import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from diffusion_heatmap_model import AffordanceDiffusion
from dense_gt_generator import DINOFeatureExtractor
from dense_gt_generator import DenseGTGenerator
from dense_gt_generator_old import ReplayBufferLoader

# visualize_predictions.py

class AffordanceVisualizer:
    """可视化 affordance 预测结果"""
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 224
    ):
        self.device = device
        self.image_size = image_size
        
        # 加载 DINO
        self.dino = DINOFeatureExtractor(device=device)
        
        # 加载模型
        self.model = AffordanceDiffusion(
            image_size=image_size,
            dino_dim=768,
            timesteps=1000,
            device=device
        ).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
        if 'loss' in checkpoint:
            print(f"Loss: {checkpoint['loss']:.4f}")
    
    def predict(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray = None,
        num_samples: int = 1
    ) -> np.ndarray:
        """
        预测 affordance heatmap
        
        Args:
            rgb: (H, W, 3) RGB 图像
            depth: (H, W) 深度图，可选
            num_samples: 采样次数（用于不确定性估计）
            
        Returns:
            heatmap: (H, W) 预测的 affordance heatmap
        """
        H_orig, W_orig = rgb.shape[:2]
        
        # 预处理
        rgb_resized = cv2.resize(rgb, (self.image_size, self.image_size))
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).float() / 255.0
        rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        
        if depth is not None:
            depth_resized = cv2.resize(depth.astype(np.float32), (self.image_size, self.image_size))
            depth_tensor = torch.from_numpy(depth_resized).float().unsqueeze(0).unsqueeze(0)
            depth_tensor = depth_tensor.to(self.device)  # (1, 1, H, W)
        else:
            depth_tensor = torch.zeros(1, 1, self.image_size, self.image_size, device=self.device)
        
        with torch.no_grad():
            # 提取 DINO 特征
            dino_features = self.dino.extract_features(rgb)  # (1, h, w, D)
            
            # 调整 DINO 特征格式 (B, h, w, D) -> (B, D, h, w)
            dino_features = dino_features.permute(0, 3, 1, 2)
            
            # Diffusion 采样
            samples = self.model.sample(
                rgb=rgb_tensor,
                depth=depth_tensor,
                dino_features=dino_features,
                num_samples=num_samples
            )  # (num_samples, 1, 1, H, W)
            
            # 平均多次采样结果
            heatmap = samples.mean(dim=0).squeeze()  # (H, W)
            
            # 上采样到原始分辨率
            heatmap = torch.nn.functional.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=(H_orig, W_orig),
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            heatmap = heatmap.cpu().numpy()
        
        return heatmap
    
    def get_top_k_points(
        self, 
        heatmap: np.ndarray, 
        k: int = 5,
        min_distance: int = 20
    ) -> list:
        """
        获取 top-K 抓取点（带非极大值抑制）
        
        Args:
            heatmap: (H, W) affordance heatmap
            k: 返回的点数
            min_distance: 点之间的最小距离（像素）
            
        Returns:
            points: [(u, v, score), ...] 按分数降序排列
        """
        H, W = heatmap.shape
        points = []
        
        # 复制 heatmap 用于 NMS
        heatmap_copy = heatmap.copy()
        
        for _ in range(k):
            # 找最大值位置
            max_idx = np.argmax(heatmap_copy)
            v, u = np.unravel_index(max_idx, (H, W))
            score = heatmap_copy[v, u]
            
            if score <= 0:
                break
            
            points.append((int(u), int(v), float(score)))
            
            # 抑制周围区域
            y_min = max(0, v - min_distance)
            y_max = min(H, v + min_distance + 1)
            x_min = max(0, u - min_distance)
            x_max = min(W, u + min_distance + 1)
            heatmap_copy[y_min:y_max, x_min:x_max] = -np.inf
        
        return points
    
    def visualize(
        self,
        rgb: np.ndarray,
        depth: np.ndarray = None,
        query_uv: tuple = None,
        k: int = 5,
        gt_heatmap: np.ndarray = None,
        save_path: str = None,
        show: bool = True,
        num_samples: int = 1
    ):
        """
        可视化预测结果
        """
        # 预测
        pred_heatmap = self.predict(rgb, depth, num_samples=num_samples)
        top_k_points = self.get_top_k_points(pred_heatmap, k=k)
        
        # 创建图
        n_cols = 3 if gt_heatmap is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
        
        # 1. 原图 + 查询点 + Top-K 点
        ax = axes[0]
        ax.imshow(rgb)
        
        # 绘制查询点（蓝色，如果有）
        if query_uv is not None:
            ax.scatter(query_uv[0], query_uv[1], c='blue', s=200, marker='*', 
                       edgecolors='white', linewidths=2, label='Query Point', zorder=5)
        
        # 绘制 Top-K 点
        colors = plt.cm.hot(np.linspace(0.2, 0.8, len(top_k_points)))
        for i, (u, v, score) in enumerate(top_k_points):
            ax.scatter(u, v, c=[colors[i]], s=150, marker='o',
                      edgecolors='white', linewidths=2, zorder=4)
            ax.annotate(f'{i+1}', (u, v), textcoords="offset points", 
                       xytext=(5, 5), ha='left', fontsize=10, 
                       color='white', fontweight='bold')
        
        title = f'RGB + Top-{k} Points'
        if query_uv:
            title += ' + Query (★)'
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        if query_uv:
            ax.legend(loc='upper right')
        
        # 2. 预测的 Heatmap
        ax = axes[1]
        ax.imshow(rgb, alpha=0.3)
        hm = ax.imshow(pred_heatmap, cmap='jet', alpha=0.7, 
                       vmin=pred_heatmap.min(), vmax=pred_heatmap.max())
        if query_uv:
            ax.scatter(query_uv[0], query_uv[1], c='blue', s=200, marker='*',
                       edgecolors='white', linewidths=2, zorder=5)
        plt.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Predicted Heatmap', fontsize=12)
        ax.axis('off')
        
        # 3. GT Heatmap（如果有）
        if gt_heatmap is not None:
            ax = axes[2]
            ax.imshow(rgb, alpha=0.3)
            hm = ax.imshow(gt_heatmap, cmap='jet', alpha=0.7,
                          vmin=-1, vmax=1)
            if query_uv:
                ax.scatter(query_uv[0], query_uv[1], c='blue', s=200, marker='*',
                           edgecolors='white', linewidths=2, zorder=5)
            plt.colorbar(hm, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Ground Truth Heatmap', fontsize=12)
            ax.axis('off')
        
        plt.tight_layout()
        
        # 打印 Top-K 信息
        print(f"\nTop-{k} Grasp Points:")
        print("-" * 40)
        for i, (u, v, score) in enumerate(top_k_points):
            print(f"  {i+1}. ({u:4d}, {v:4d}) - score: {score:.4f}")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved to {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
        
        return pred_heatmap, top_k_points


def visualize_from_episode(
    visualizer: AffordanceVisualizer,
    episode_dir: Path,
    k: int = 5,
    save_dir: Path = None,
    show: bool = True,
    num_samples: int = 1
):
    """从 episode 目录可视化"""
    from dense_gt_generator import DenseGTGenerator
    from dense_gt_generator_old import EpisodeData
    import json
    import pickle
    
    # 检查文件是否存在
    rgb_path = episode_dir / "step0_rgb.png"
    depth_path = episode_dir / "step0_depth.npy"
    meta_path = episode_dir / "meta.json"
    masks_path = episode_dir / "step0_masks.pkl"
    
    if not rgb_path.exists():
        print(f"Skipping {episode_dir.name}: step0_rgb.png not found")
        return False
    if not depth_path.exists():
        print(f"Skipping {episode_dir.name}: step0_depth.npy not found")
        return False
    if not meta_path.exists():
        print(f"Skipping {episode_dir.name}: meta.json not found")
        return False
    if not masks_path.exists():
        print(f"Skipping {episode_dir.name}: step0_masks.pkl not found")
        return False
    
    try:
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            print(f"Skipping {episode_dir.name}: failed to read step0_rgb.png")
            return False
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        depth = np.load(str(depth_path))
        
        with open(masks_path, "rb") as f:
            masks = pickle.load(f)
        
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        # 从 steps 数组中获取第一个 step 的信息
        step_info = metadata["steps"][0]
        H, W = rgb.shape[:2]
        
        # uv 是归一化坐标，pick_pixel 是像素坐标
        uv_normalized = step_info["uv"]
        pick_uv = (uv_normalized[0], uv_normalized[1])  # 归一化坐标
        pick_pixel = tuple(step_info["extra"]["pixel"])  # 像素坐标 [u, v]
        mask_idx = step_info["extra"]["mask_idx"]
        
        episode_id = metadata["episode_id"]
        outcome = metadata["outcome"]
        
        episode = EpisodeData(
            rgb=rgb,
            depth=depth,
            pick_uv=pick_uv,
            episode_id=episode_id,
            masks=masks,
            pick_pixel=pick_pixel,
            outcome=outcome,
            mask_idx=mask_idx
        )
    except Exception as e:
        print(f"Skipping {episode_dir.name}: {e}")
        return False
    
    gt_generator = DenseGTGenerator(
        dino_extractor=visualizer.dino,
        use_object_mask=True,
        mask_boost=0.3,
        distance_sigma=100.0,
        similarity_weight=0.6,
        distance_weight=0.4,
        background_min=0.05
    )
    
    # 生成 GT
    gt_heatmap = gt_generator.generate_dense_gt(episode)
    
    # 可视化时用像素坐标
    save_path = None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{episode_dir.name}_pred.png"
    
    visualizer.visualize(
        rgb=episode.rgb,
        depth=episode.depth,
        query_uv=pick_pixel,  # 用像素坐标显示
        k=k,
        gt_heatmap=gt_heatmap,
        save_path=str(save_path) if save_path else None,
        show=show,
        num_samples=num_samples
    )
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Visualize affordance predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--episode', type=str, default=None,
                        help='Path to episode directory')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image file')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Number of top grasp points to show')
    parser.add_argument('--buffer', type=str, default='./replaybuffer',
                        help='Replay buffer directory for batch visualization')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to visualize in batch mode')
    parser.add_argument('--diffusion-samples', type=int, default=3,
                        help='Number of diffusion samples for uncertainty')
    parser.add_argument('--save-dir', type=str, default='./visualizations',
                        help='Directory to save visualizations')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Model image size')
    
    args = parser.parse_args()
    
    # 加载模型
    visualizer = AffordanceVisualizer(
        checkpoint_path=args.checkpoint,
        device=args.device,
        image_size=args.image_size
    )
    
    save_dir = Path(args.save_dir) if args.save_dir else None
    show = not args.no_show
    
    if args.episode:
        # 可视化单个 episode
        episode_dir = Path(args.episode)
        visualize_from_episode(
            visualizer, episode_dir, 
            k=args.top_k, save_dir=save_dir, show=show,
            num_samples=args.diffusion_samples
        )
    
    elif args.image:
        # 可视化单张图片
        rgb = cv2.imread(args.image)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        save_path = str(save_dir / "image_pred.png") if save_dir else None
        visualizer.visualize(
            rgb=rgb,
            k=args.top_k,
            save_path=save_path,
            show=show,
            num_samples=args.diffusion_samples
        )
    
    else:
        # 批量可视化 buffer 中的样本
        buffer_dir = Path(args.buffer)
        episode_dirs = sorted([d for d in buffer_dir.iterdir() if d.is_dir()])

        if len(episode_dirs) == 0:
            print(f"No episodes found in {buffer_dir}")
            return

        # 随机选择样本
        import random
        sample_dirs = random.sample(episode_dirs, min(args.num_samples, len(episode_dirs)))

        print(f"Visualizing {len(sample_dirs)} episodes...")
        success_count = 0
        for episode_dir in sample_dirs:
            print(f"\n{'='*50}")
            print(f"Episode: {episode_dir.name}")
            if visualize_from_episode(
                visualizer, episode_dir,
                k=args.top_k, save_dir=save_dir, show=show,
                num_samples=args.diffusion_samples
            ):
                success_count += 1

        print(f"\nDone. Successfully visualized {success_count}/{len(sample_dirs)} episodes.")


if __name__ == '__main__':
    main()