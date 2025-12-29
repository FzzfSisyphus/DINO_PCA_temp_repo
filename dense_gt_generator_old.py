"""
dense_gt_generator.py
将稀疏的 (u, v, outcome) 转化为密集的 Affordance Heatmap GT
"""

import os
import json
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from transformers import AutoModel, AutoImageProcessor


@dataclass
class EpisodeData:
    """单次抓取的完整数据"""
    episode_id: str
    rgb: np.ndarray          # (H, W, 3)
    depth: np.ndarray        # (H, W)
    masks: List[np.ndarray]  # list of (H, W) binary masks
    pick_uv: Tuple[float, float]  # normalized (u, v) in [0, 1]
    pick_pixel: Tuple[int, int]   # (x, y) pixel coords
    outcome: str             # 'success' or 'failure'
    mask_idx: int           # which mask was picked


class DINOFeatureExtractor:
    """DINOv2 特征提取器"""
    
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "cuda"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # DINOv2-base: patch_size=14, hidden_dim=768
        self.patch_size = 14
        self.hidden_dim = 768
        self.feat_dim = 768
        
    @torch.no_grad()
    def extract_features(self, rgb: np.ndarray) -> torch.Tensor:
        """
        提取 patch-level 特征图
        
        Args:
            rgb: (H, W, 3) numpy array, uint8
            
        Returns:
            features: (1, h, w, D) tensor, h/w = H/W / patch_size
        """
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # 获取最后一层的 patch tokens (去掉 CLS token)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (1, num_patches, D)
        
        # 重塑为空间特征图
        h = inputs['pixel_values'].shape[2] // self.patch_size
        w = inputs['pixel_values'].shape[3] // self.patch_size
        features = patch_tokens.reshape(1, h, w, -1)  # (1, h, w, D)
        
        return features
    
    def get_point_feature(
        self, 
        features: torch.Tensor, 
        uv: Tuple[float, float],
        original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """获取指定点的特征向量"""
        _, h, w, D = features.shape
        H, W = original_size
        
        # 将 normalized uv 转换为 feature map 坐标
        feat_x = int(uv[0] * w)
        feat_y = int(uv[1] * h)
        feat_x = min(max(feat_x, 0), w - 1)
        feat_y = min(max(feat_y, 0), h - 1)
        
        return features[0, feat_y, feat_x, :]  # (D,)


class DenseGTGenerator:
    """
    密集 Ground Truth 生成器
    
    核心思想：
    1. 用 DINO 特征相似度找到语义一致的区域
    2. 用深度图约束过滤不同层的物体
    3. 用 Distance Transform 生成平滑的向心力权重
    """
    
    def __init__(
        self,
        dino_extractor: DINOFeatureExtractor,
        similarity_threshold: float = 0.7,
        depth_threshold_mm: float = 5.0,
        gaussian_sigma: float = 3.0,
        device: str = "cuda"
    ):
        self.dino = dino_extractor
        self.sim_threshold = similarity_threshold
        self.depth_threshold = depth_threshold_mm
        self.gaussian_sigma = gaussian_sigma
        self.device = device
        
    def generate_dense_gt(
        self,
        episode: EpisodeData,
        use_mask_prior: bool = True
    ) -> np.ndarray:
        """
        生成密集的 affordance heatmap GT
        
        Args:
            episode: 单次抓取数据
            use_mask_prior: 是否使用分割 mask 作为先验
            
        Returns:
            heatmap: (H, W) float32, 值域 [0, 1] for success, [-1, 0] for failure
        """
        H, W = episode.rgb.shape[:2]
        
        # Step 1: 提取 DINO 特征
        features = self.dino.extract_features(episode.rgb)  # (1, h, w, D)
        point_feat = self.dino.get_point_feature(
            features, episode.pick_uv, (H, W)
        )  # (D,)
        
        # Step 2: 计算全图相似度
        similarity_map = self._compute_similarity_map(features, point_feat, (H, W))
        
        # Step 3: 深度约束
        depth_mask = self._compute_depth_mask(
            episode.depth, 
            episode.pick_pixel,
            threshold_mm=self.depth_threshold
        )
        
        # Step 4: 结合 mask 先验（如果可用）
        object_mask = self._get_mask(episode.masks, episode.mask_idx, (H, W))

        # Step 5: 组合所有约束
        valid_region = (
            (similarity_map > self.sim_threshold) & 
            depth_mask & 
            object_mask
        )
        
        # Step 6: Distance Transform 生成向心力权重
        if valid_region.sum() > 0:
            # 计算到边界的距离
            dt = distance_transform_edt(valid_region)
            dt = dt / (dt.max() + 1e-6)  # normalize to [0, 1]
            
            # 乘以相似度作为最终权重
            heatmap = dt * similarity_map * valid_region.astype(np.float32)
        else:
            # fallback: 如果没有有效区域，使用 Gaussian blob
            heatmap = self._gaussian_blob(
                (H, W), 
                episode.pick_pixel, 
                sigma=self.gaussian_sigma
            )
        
        # Step 7: 根据 outcome 调整符号
        if episode.outcome == "failure":
            # 失败点周围标记为负值
            failure_mask = self._gaussian_blob(
                (H, W), 
                episode.pick_pixel, 
                sigma=self.gaussian_sigma * 2
            )
            heatmap = heatmap - failure_mask  # 失败区域变负
        
        # 裁剪到 [-1, 1]
        heatmap = np.clip(heatmap, -1.0, 1.0)
        
        return heatmap.astype(np.float32)
    
    def _compute_similarity_map(
        self, 
        features: torch.Tensor, 
        point_feat: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """计算特征相似度图并上采样到原图尺寸"""
        # Cosine similarity
        features_flat = features[0].reshape(-1, features.shape[-1])  # (h*w, D)
        point_feat = point_feat / (point_feat.norm() + 1e-6)
        features_flat = features_flat / (features_flat.norm(dim=-1, keepdim=True) + 1e-6)
        
        similarity = (features_flat @ point_feat).reshape(features.shape[1], features.shape[2])
        similarity = similarity.cpu().numpy()
        
        # 双线性插值到原图尺寸
        similarity = np.clip(similarity, 0, 1)
        sim_tensor = torch.from_numpy(similarity).unsqueeze(0).unsqueeze(0).float()
        sim_upsampled = F.interpolate(
            sim_tensor, 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        return sim_upsampled[0, 0].numpy()
    
    def _get_mask(self, masks_data, mask_idx: int, size: tuple) -> np.ndarray:
        """从 masks 数据中提取指定 mask"""
        H, W = size
        
        if masks_data is None or len(masks_data) == 0:
            return np.ones((H, W), dtype=bool)
        
        try:
            # 你的格式: list of dict, 每个 dict 有 'mask' 字段
            idx = min(mask_idx, len(masks_data) - 1)
            mask_item = masks_data[idx]
            
            if isinstance(mask_item, dict) and 'mask' in mask_item:
                mask = np.array(mask_item['mask'])
                return mask.astype(bool)
            elif isinstance(mask_item, np.ndarray):
                return mask_item.astype(bool)
            else:
                return np.ones((H, W), dtype=bool)
                
        except Exception as e:
            print(f"Warning: Failed to extract mask: {e}")
            return np.ones((H, W), dtype=bool)
    
    def _compute_depth_mask(
        self, 
        depth: np.ndarray, 
        pick_pixel: Tuple[int, int],
        threshold_mm: float
    ) -> np.ndarray:
        """计算深度约束 mask"""
        px, py = pick_pixel
        H, W = depth.shape
        
        # 边界检查
        px = min(max(px, 0), W - 1)
        py = min(max(py, 0), H - 1)
        
        pick_depth = depth[py, px]
        
        # 深度差在阈值内的区域
        depth_diff = np.abs(depth - pick_depth)
        mask = depth_diff < threshold_mm
        
        return mask
    
    def _gaussian_blob(
        self, 
        size: Tuple[int, int], 
        center: Tuple[int, int],
        sigma: float
    ) -> np.ndarray:
        """生成高斯 blob"""
        H, W = size
        cx, cy = center
        
        y, x = np.ogrid[:H, :W]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
        
        return gaussian.astype(np.float32)


class ReplayBufferLoader:
    """加载 replay buffer 数据"""
    
    def __init__(self, buffer_dir: str):
        self.buffer_dir = Path(buffer_dir)
        self.episode_dirs = sorted([
            d for d in self.buffer_dir.iterdir() 
            if d.is_dir() and (d / "meta.json").exists()
        ])
        
    def __len__(self) -> int:
        return len(self.episode_dirs)
    
    def load_episode(self, idx: int) -> Optional[EpisodeData]:
        """加载单个 episode"""
        episode_dir = self.episode_dirs[idx]
        
        try:
            # Load metadata
            with open(episode_dir / "meta.json", "r") as f:
                meta = json.load(f)
            
            # Load RGB
            rgb = np.array(Image.open(episode_dir / "step0_rgb.png"))
            
            # Load Depth
            depth = np.load(episode_dir / "step0_depth.npy")
            
            # Load Masks
            with open(episode_dir / "step0_masks.pkl", "rb") as f:
                masks = pickle.load(f)
            
            # Parse pick info
            step = meta["steps"][0]
            uv = tuple(step["uv"])
            pixel = tuple(step["extra"]["pixel"])
            mask_idx = step["extra"]["mask_idx"]
            
            return EpisodeData(
                episode_id=meta["episode_id"],
                rgb=rgb,
                depth=depth,
                masks=masks,
                pick_uv=uv,
                pick_pixel=pixel,
                outcome=meta["outcome"],
                mask_idx=mask_idx
            )
            
        except Exception as e:
            print(f"Error loading episode {episode_dir}: {e}")
            return None
    
    def iterate_episodes(self):
        """迭代所有 episodes"""
        for idx in range(len(self)):
            episode = self.load_episode(idx)
            if episode is not None:
                yield episode

