"""
inference.py
推理和 UCB 探索策略
"""

import torch
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class PickCandidate:
    """抓取候选点"""
    pixel: Tuple[int, int]  # (x, y)
    uv: Tuple[float, float]  # normalized
    mean_score: float
    variance: float
    ucb_score: float


class InferenceEngine:
    """推理引擎"""
    
    def __init__(
        self,
        model: 'AffordanceDiffusion',
        dino_extractor: 'DINOFeatureExtractor',
        sampler: 'DDIMSampler' = None,
        num_samples: int = 10,
        ucb_beta: float = 1.0,
        device: str = "cuda"
    ):
        self.model = model.to(device).eval()
        self.dino = dino_extractor
        self.sampler = sampler or DDIMSampler(model, ddim_steps=50)
        self.num_samples = num_samples
        self.ucb_beta = ucb_beta
        self.device = device
        
    @torch.no_grad()
    def predict(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        top_k: int = 5
    ) -> Tuple[List[PickCandidate], np.ndarray, np.ndarray]:
        """
        预测抓取点
        
        Returns:
            candidates: Top-K 候选点列表
            mean_heatmap: 均值 heatmap
            var_heatmap: 方差 heatmap
        """
        H, W = rgb.shape[:2]
        
        # 预处理
        rgb_tensor, depth_tensor, dino_feat = self._preprocess(rgb, depth)
        
        # 采样多个 heatmap
        samples = self.sampler.sample(
            rgb_tensor, depth_tensor, dino_feat,
            num_samples=self.num_samples
        )  # (N, 1, 1, H', W')
        
        samples = samples.squeeze(2)  # (N, 1, H', W')
        
        # 计算均值和方差
        mean_heatmap = samples.mean(dim=0)  # (1, H', W')
        var_heatmap = samples.var(dim=0)    # (1, H', W')
        
        # UCB score
        ucb_heatmap = mean_heatmap + self.ucb_beta * torch.sqrt(var_heatmap)
        
        # 上采样到原图尺寸
        mean_heatmap = torch.nn.functional.interpolate(
            mean_heatmap.unsqueeze(0), size=(H, W), mode='bilinear'
        )[0, 0].cpu().numpy()
        
        var_heatmap = torch.nn.functional.interpolate(
            var_heatmap.unsqueeze(0), size=(H, W), mode='bilinear'
        )[0, 0].cpu().numpy()
        
        ucb_heatmap = torch.nn.functional.interpolate(
            ucb_heatmap.unsqueeze(0), size=(H, W), mode='bilinear'
        )[0, 0].cpu().numpy()
        
        # 选择 Top-K 点
        candidates = self._select_top_k(
            mean_heatmap, var_heatmap, ucb_heatmap, 
            image_size=(H, W), k=top_k
        )
        
        return candidates, mean_heatmap, var_heatmap
    
    def _preprocess(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """预处理输入"""
        import torchvision.transforms as T
        
        # RGB
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((self.model.image_size, self.model.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        rgb_tensor = transform(rgb).unsqueeze(0).to(self.device)
        
        # Depth
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth_tensor = torch.from_numpy(depth_norm).float().unsqueeze(0).unsqueeze(0)
        depth_tensor = T.functional.resize(
            depth_tensor, [self.model.image_size, self.model.image_size]
        ).to(self.device)
        
        # DINO features
        dino_feat = self.dino.extract_features(rgb)  # (1, h, w, D)
        dino_feat = dino_feat.permute(0, 3, 1, 2).to(self.device)  # (1, D, h, w)
        
        return rgb_tensor, depth_tensor, dino_feat
    
    def _select_top_k(
        self,
        mean_map: np.ndarray,
        var_map: np.ndarray,
        ucb_map: np.ndarray,
        image_size: Tuple[int, int],
        k: int
    ) -> List[PickCandidate]:
        """选择 Top-K 候选点（带 NMS）"""
        H, W = image_size
        
        # 找到局部最大值（简单 NMS）
        from scipy.ndimage import maximum_filter
        
        # 用最大值滤波找局部最大值
        local_max = maximum_filter(ucb_map, size=20)
        peak_mask = (ucb_map == local_max) & (ucb_map > 0)
        
        # 获取峰值点
        peaks_y, peaks_x = np.where(peak_mask)
        peak_scores = ucb_map[peaks_y, peaks_x]
        
        # 按 UCB score 排序
        sorted_idx = np.argsort(peak_scores)[::-1]
        
        candidates = []
        for i in sorted_idx[:k]:
            x, y = peaks_x[i], peaks_y[i]
            candidates.append(PickCandidate(
                pixel=(int(x), int(y)),
                uv=(x / W, y / H),
                mean_score=float(mean_map[y, x]),
                variance=float(var_map[y, x]),
                ucb_score=float(ucb_map[y, x])
            ))
        
        return candidates


class SelfImprovingLoop:
    """
    自我改进循环
    
    执行抓取 -> 收集结果 -> 生成新 GT -> 微调模型
    """
    
    def __init__(
        self,
        model: 'AffordanceDiffusion',
        gt_generator: 'DenseGTGenerator',
        buffer_dir: str,
        finetune_interval: int = 10,  # 每 N 次抓取微调一次
        finetune_epochs: int = 5
    ):
        self.model = model
        self.gt_generator = gt_generator
        self.buffer_dir = buffer_dir
        self.finetune_interval = finetune_interval
        self.finetune_epochs = finetune_epochs
        
        self.new_episodes = []
        self.pick_count = 0
        
    def on_pick_complete(self, episode: 'EpisodeData'):
        """
        抓取完成后调用
        
        Args:
            episode: 刚完成的抓取数据
        """
        # 生成密集 GT
        dense_gt = self.gt_generator.generate_dense_gt(episode)
        
        # 保存到 buffer
        self._save_to_buffer(episode, dense_gt)
        
        self.new_episodes.append(episode)
        self.pick_count += 1
        
        # 检查是否需要微调
        if self.pick_count % self.finetune_interval == 0:
            self._finetune()
    
    def _save_to_buffer(self, episode: 'EpisodeData', dense_gt: np.ndarray):
        """保存到 replay buffer"""
        import json
        from pathlib import Path
        
        episode_dir = Path(self.buffer_dir) / episode.episode_id
        episode_dir.mkdir(exist_ok=True)
        
        # 保存密集 GT
        np.save(episode_dir / "dense_gt.npy", dense_gt)
        
    def _finetune(self):
        """微调模型"""
        print(f"Finetuning on {len(self.new_episodes)} new episodes...")
        
        # 这里简化处理，实际应该使用完整的 Trainer
        # 重点是：使用新生成的 dense GT 进行训练
        
        # 清空缓存
        self.new_episodes = []