# dense_gt_generator_v3.py
"""
改进版 Dense GT 生成器 V3
- 全局都有信号，不再有"白色死区"
- 基于相似度 + 距离的平滑过渡
- 中心区域高值，远离区域低值但不是0
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, gaussian_filter
from dataclasses import dataclass
from typing import Tuple, List, Optional
from transformers import AutoModel, AutoImageProcessor


@dataclass
class EpisodeData:
    """单次抓取的完整数据"""
    episode_id: str
    rgb: np.ndarray
    depth: np.ndarray
    masks: List[np.ndarray]
    pick_uv: Tuple[float, float]
    pick_pixel: Tuple[int, int]
    outcome: str
    mask_idx: int


class DINOFeatureExtractor:
    """DINOv2 特征提取器"""
    
    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "cuda"):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.patch_size = 14
        
    @torch.no_grad()
    def extract_features(self, rgb: np.ndarray) -> torch.Tensor:
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        h = inputs['pixel_values'].shape[2] // self.patch_size
        w = inputs['pixel_values'].shape[3] // self.patch_size
        features = patch_tokens.reshape(1, h, w, -1)
        return features
    
    def get_point_feature(
        self, 
        features: torch.Tensor, 
        uv: Tuple[float, float],
        original_size: Tuple[int, int]
    ) -> torch.Tensor:
        _, h, w, D = features.shape
        feat_x = int(uv[0] * w)
        feat_y = int(uv[1] * h)
        feat_x = min(max(feat_x, 0), w - 1)
        feat_y = min(max(feat_y, 0), h - 1)
        return features[0, feat_y, feat_x, :]


class DenseGTGenerator:
    """
    Dense GT 生成器 V3
    
    设计理念：
    - 全图都有信号，形成从中心到边缘的平滑梯度
    - 高相似度区域 → 高值 (接近 1)
    - 低相似度区域 → 低值 (接近 0，但不是 0)
    - 可选：使用 object mask 增强对比度
    """
    
    def __init__(
        self,
        dino_extractor: DINOFeatureExtractor,
        use_object_mask: bool = True,
        mask_boost: float = 0.3,       # mask 内的额外加成
        distance_sigma: float = 100.0,  # 距离衰减的 sigma
        similarity_weight: float = 0.6, # 相似度权重
        distance_weight: float = 0.4,   # 距离权重
        background_min: float = 0.05,   # 背景最小值（不要是0）
        device: str = "cuda"
    ):
        self.dino = dino_extractor
        self.use_object_mask = use_object_mask
        self.mask_boost = mask_boost
        self.distance_sigma = distance_sigma
        self.sim_weight = similarity_weight
        self.dist_weight = distance_weight
        self.bg_min = background_min
        self.device = device
        
    def generate_dense_gt(self, episode: EpisodeData) -> np.ndarray:
        """生成密集 GT heatmap"""
        H, W = episode.rgb.shape[:2]
        px, py = episode.pick_pixel
        px, py = min(max(px, 0), W-1), min(max(py, 0), H-1)
        
        # === Step 1: 计算相似度图 ===
        features = self.dino.extract_features(episode.rgb)
        point_feat = self.dino.get_point_feature(features, episode.pick_uv, (H, W))
        similarity_map = self._compute_similarity_map(features, point_feat, (H, W))
        
        # === Step 2: 计算距离图（到 pick point 的距离）===
        distance_map = self._compute_distance_map((H, W), (px, py))
        
        # === Step 3: 组合相似度和距离 ===
        # 两者都归一化到 [0, 1]
        combined = (
            self.sim_weight * similarity_map + 
            self.dist_weight * distance_map
        )
        
        # === Step 4: 可选的 Object Mask 增强 ===
        if self.use_object_mask:
            object_mask = self._get_object_mask(episode.masks, episode.mask_idx, (H, W))
            # mask 内加成，mask 外保持但稍微降低
            mask_factor = np.where(object_mask, 1.0 + self.mask_boost, 1.0 - self.mask_boost * 0.5)
            combined = combined * mask_factor
        
        # === Step 5: 确保全图有信号 ===
        # 添加一个基础的背景值 + 到 pick point 的柔和距离衰减
        background = self._compute_distance_map((H, W), (px, py), sigma=200.0)
        background = self.bg_min + (1 - self.bg_min) * background * 0.3
        
        # 取 combined 和 background 的较大值
        heatmap = np.maximum(combined, background)
        
        # === Step 6: 归一化到 [0, 1] ===
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # === Step 7: 处理 success/failure ===
        if episode.outcome == "success":
            # 成功：保持正值
            pass
        else:
            # 失败：翻转，中心变低，远处变高
            heatmap = 1.0 - heatmap
        
        return heatmap.astype(np.float32)
    
    def _compute_similarity_map(
        self, 
        features: torch.Tensor, 
        point_feat: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """计算归一化的相似度图 [0, 1]"""
        features_flat = features[0].reshape(-1, features.shape[-1])
        point_feat = point_feat / (point_feat.norm() + 1e-6)
        features_flat = features_flat / (features_flat.norm(dim=-1, keepdim=True) + 1e-6)
        
        similarity = (features_flat @ point_feat).reshape(features.shape[1], features.shape[2])
        similarity = similarity.cpu().numpy()
        
        # 上采样
        sim_tensor = torch.from_numpy(similarity).unsqueeze(0).unsqueeze(0).float()
        sim_upsampled = F.interpolate(sim_tensor, size=target_size, mode='bilinear', align_corners=False)
        sim_map = sim_upsampled[0, 0].numpy()
        
        # 归一化到 [0, 1]，负值 clip 到 0
        sim_map = np.clip(sim_map, 0, None)
        if sim_map.max() > 0:
            sim_map = sim_map / sim_map.max()
        
        return sim_map
    
    def _compute_distance_map(
        self, 
        size: Tuple[int, int], 
        center: Tuple[int, int],
        sigma: float = None
    ) -> np.ndarray:
        """计算到中心点的距离衰减图，高斯形式，[0, 1]"""
        if sigma is None:
            sigma = self.distance_sigma
            
        H, W = size
        cx, cy = center
        y, x = np.ogrid[:H, :W]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
        return gaussian.astype(np.float32)
    
    def _get_object_mask(self, masks_data, mask_idx: int, size: Tuple[int, int]) -> np.ndarray:
        """提取并膨胀 object mask"""
        H, W = size
        
        if masks_data is None or len(masks_data) == 0:
            return np.ones((H, W), dtype=bool)
        
        try:
            idx = min(mask_idx, len(masks_data) - 1)
            mask_item = masks_data[idx]
            
            if isinstance(mask_item, dict) and 'mask' in mask_item:
                mask = np.array(mask_item['mask'])
            elif isinstance(mask_item, np.ndarray):
                mask = mask_item
            else:
                return np.ones((H, W), dtype=bool)
            
            # 膨胀 mask
            from scipy.ndimage import binary_dilation
            mask = binary_dilation(mask, iterations=10)
            
            return mask.astype(bool)
            
        except Exception as e:
            print(f"Warning: Failed to extract mask: {e}")
            return np.ones((H, W), dtype=bool)


# ============ 可视化测试 ============

def test_v3_generator(episode: EpisodeData, save_path: str = None):
    """测试 V3 生成器，展示更多细节"""
    import matplotlib.pyplot as plt
    
    dino = DINOFeatureExtractor(device="cuda")
    
    H, W = episode.rgb.shape[:2]
    px, py = episode.pick_pixel
    
    # 提取基础信息
    features = dino.extract_features(episode.rgb)
    point_feat = dino.get_point_feature(features, episode.pick_uv, (H, W))
    
    # 计算相似度
    features_flat = features[0].reshape(-1, features.shape[-1])
    point_feat_norm = point_feat / (point_feat.norm() + 1e-6)
    features_flat_norm = features_flat / (features_flat.norm(dim=-1, keepdim=True) + 1e-6)
    similarity = (features_flat_norm @ point_feat_norm).reshape(features.shape[1], features.shape[2])
    similarity = similarity.cpu().numpy()
    sim_tensor = torch.from_numpy(similarity).unsqueeze(0).unsqueeze(0).float()
    sim_map = F.interpolate(sim_tensor, size=(H, W), mode='bilinear', align_corners=False)[0, 0].numpy()
    sim_map = np.clip(sim_map, 0, None)
    if sim_map.max() > 0:
        sim_map = sim_map / sim_map.max()
    
    # 距离图
    y, x = np.ogrid[:H, :W]
    dist_sq = (x - px) ** 2 + (y - py) ** 2
    distance_map = np.exp(-dist_sq / (2 * 100.0 ** 2))
    
    # 生成最终 heatmap
    generator = DenseGTGenerator(
        dino_extractor=dino,
        use_object_mask=True,
        mask_boost=0.3,
        distance_sigma=100.0,
        similarity_weight=0.6,
        distance_weight=0.4,
        background_min=0.05
    )
    heatmap = generator.generate_dense_gt(episode)
    
    # 绘图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: 输入分解
    axes[0, 0].imshow(episode.rgb)
    axes[0, 0].scatter([px], [py], c='red', s=200, marker='*', edgecolors='white', linewidths=2)
    axes[0, 0].set_title(f'RGB ({episode.outcome})')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(sim_map, cmap='hot')
    axes[0, 1].scatter([px], [py], c='cyan', s=100, marker='*')
    axes[0, 1].set_title('DINO Similarity (normalized)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(distance_map, cmap='hot')
    axes[0, 2].scatter([px], [py], c='cyan', s=100, marker='*')
    axes[0, 2].set_title('Distance Decay (Gaussian)')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: 最终结果
    im3 = axes[1, 0].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    axes[1, 0].scatter([px], [py], c='cyan', s=100, marker='*')
    axes[1, 0].set_title(f'Final Heatmap\nmin={heatmap.min():.3f}, max={heatmap.max():.3f}')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Heatmap 叠加在 RGB 上
    axes[1, 1].imshow(episode.rgb)
    axes[1, 1].imshow(heatmap, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[1, 1].scatter([px], [py], c='cyan', s=100, marker='*')
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')
    
    # 像素值分布
    axes[1, 2].hist(heatmap.flatten(), bins=50, edgecolor='black', color='orange')
    axes[1, 2].axvline(x=heatmap.mean(), color='red', linestyle='--', label=f'mean={heatmap.mean():.3f}')
    axes[1, 2].set_xlabel('Heatmap Value')
    axes[1, 2].set_ylabel('Pixel Count')
    axes[1, 2].set_title('Heatmap Value Distribution')
    axes[1, 2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # 打印统计
    print("\n=== Heatmap Statistics ===")
    print(f"Shape: {heatmap.shape}")
    print(f"Min: {heatmap.min():.4f}")
    print(f"Max: {heatmap.max():.4f}")
    print(f"Mean: {heatmap.mean():.4f}")
    print(f"Std: {heatmap.std():.4f}")
    print(f"Pixels > 0.5: {(heatmap > 0.5).sum()} ({100*(heatmap > 0.5).mean():.1f}%)")
    print(f"Pixels > 0.8: {(heatmap > 0.8).sum()} ({100*(heatmap > 0.8).mean():.1f}%)")


if __name__ == "__main__":
    from dense_gt_generator_old import ReplayBufferLoader
    
    loader = ReplayBufferLoader("./replaybuffer")
    
    # 测试多个 episode
    for i in range(min(3, len(loader))):
        episode = loader.load_episode(i)
        if episode:
            print(f"\n{'='*50}")
            print(f"Episode {i}: {episode.episode_id}")
            test_v3_generator(episode, save_path=f"./debug_gt_v3_ep{i}.png")