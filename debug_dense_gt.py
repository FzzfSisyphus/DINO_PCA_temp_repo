# debug_dense_gt.py
"""调试 Dense GT 生成的各个步骤"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
from PIL import Image

from dense_gt_generator import (
    DINOFeatureExtractor, 
    DenseGTGenerator, 
    EpisodeData,
    ReplayBufferLoader
)


def debug_gt_generation(episode: EpisodeData, save_dir: Path = None):
    """可视化 GT 生成的每个步骤"""
    
    dino = DINOFeatureExtractor(device="cuda")
    
    H, W = episode.rgb.shape[:2]
    
    # Step 1: DINO 特征相似度
    features = dino.extract_features(episode.rgb)
    point_feat = dino.get_point_feature(features, episode.pick_uv, (H, W))
    
    # 计算相似度（不做阈值）
    features_flat = features[0].reshape(-1, features.shape[-1])
    point_feat_norm = point_feat / (point_feat.norm() + 1e-6)
    features_flat_norm = features_flat / (features_flat.norm(dim=-1, keepdim=True) + 1e-6)
    similarity = (features_flat_norm @ point_feat_norm).reshape(features.shape[1], features.shape[2])
    similarity = similarity.cpu().numpy()
    
    # 上采样
    import torch.nn.functional as F
    import torch
    sim_tensor = torch.from_numpy(similarity).unsqueeze(0).unsqueeze(0).float()
    sim_upsampled = F.interpolate(sim_tensor, size=(H, W), mode='bilinear', align_corners=False)
    similarity_map = sim_upsampled[0, 0].numpy()
    
    # Step 2: 深度 mask
    px, py = episode.pick_pixel
    px, py = min(max(px, 0), W-1), min(max(py, 0), H-1)
    pick_depth = episode.depth[py, px]
    
    depth_diff = np.abs(episode.depth - pick_depth)
    depth_mask_5mm = depth_diff < 5.0
    depth_mask_10mm = depth_diff < 10.0
    depth_mask_20mm = depth_diff < 20.0
    depth_mask_50mm = depth_diff < 50.0
    
    # Step 3: Object mask
    masks = episode.masks
    mask_idx = min(episode.mask_idx, len(masks) - 1)
    if isinstance(masks[mask_idx], dict) and 'mask' in masks[mask_idx]:
        object_mask = np.array(masks[mask_idx]['mask']).astype(bool)
    else:
        object_mask = np.array(masks[mask_idx]).astype(bool)
    
    # Step 4: 不同阈值的相似度
    sim_thresh_05 = similarity_map > 0.5
    sim_thresh_06 = similarity_map > 0.6
    sim_thresh_07 = similarity_map > 0.7
    sim_thresh_08 = similarity_map > 0.8
    
    # 创建可视化
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    # Row 1: RGB, Depth, Pick point
    axes[0, 0].imshow(episode.rgb)
    axes[0, 0].scatter([px], [py], c='red', s=100, marker='*')
    axes[0, 0].set_title(f'RGB + Pick Point\nOutcome: {episode.outcome}')
    
    axes[0, 1].imshow(episode.depth, cmap='viridis')
    axes[0, 1].scatter([px], [py], c='red', s=100, marker='*')
    axes[0, 1].set_title(f'Depth\nPick depth: {pick_depth:.1f}mm')
    
    axes[0, 2].imshow(similarity_map, cmap='jet', vmin=0, vmax=1)
    axes[0, 2].scatter([px], [py], c='white', s=100, marker='*')
    axes[0, 2].set_title(f'DINO Similarity\nmin={similarity_map.min():.2f}, max={similarity_map.max():.2f}')
    
    axes[0, 3].imshow(object_mask, cmap='gray')
    axes[0, 3].scatter([px], [py], c='red', s=100, marker='*')
    axes[0, 3].set_title(f'Object Mask (idx={mask_idx})\nArea: {object_mask.sum()} pixels')
    
    # Row 2: Depth masks at different thresholds
    axes[1, 0].imshow(depth_mask_5mm, cmap='gray')
    axes[1, 0].set_title(f'Depth < 5mm\nArea: {depth_mask_5mm.sum()}')
    
    axes[1, 1].imshow(depth_mask_10mm, cmap='gray')
    axes[1, 1].set_title(f'Depth < 10mm\nArea: {depth_mask_10mm.sum()}')
    
    axes[1, 2].imshow(depth_mask_20mm, cmap='gray')
    axes[1, 2].set_title(f'Depth < 20mm\nArea: {depth_mask_20mm.sum()}')
    
    axes[1, 3].imshow(depth_mask_50mm, cmap='gray')
    axes[1, 3].set_title(f'Depth < 50mm\nArea: {depth_mask_50mm.sum()}')
    
    # Row 3: Similarity thresholds
    axes[2, 0].imshow(sim_thresh_05, cmap='gray')
    axes[2, 0].set_title(f'Sim > 0.5\nArea: {sim_thresh_05.sum()}')
    
    axes[2, 1].imshow(sim_thresh_06, cmap='gray')
    axes[2, 1].set_title(f'Sim > 0.6\nArea: {sim_thresh_06.sum()}')
    
    axes[2, 2].imshow(sim_thresh_07, cmap='gray')
    axes[2, 2].set_title(f'Sim > 0.7\nArea: {sim_thresh_07.sum()}')
    
    axes[2, 3].imshow(sim_thresh_08, cmap='gray')
    axes[2, 3].set_title(f'Sim > 0.8\nArea: {sim_thresh_08.sum()}')
    
    # Row 4: Combined masks and final GT
    # 使用不同参数组合
    combined_loose = (similarity_map > 0.5) & depth_mask_50mm & object_mask
    combined_medium = (similarity_map > 0.6) & depth_mask_20mm & object_mask
    combined_tight = (similarity_map > 0.7) & depth_mask_10mm & object_mask
    
    axes[3, 0].imshow(combined_loose, cmap='gray')
    axes[3, 0].set_title(f'Loose (sim>0.5, depth<50mm)\nArea: {combined_loose.sum()}')
    
    axes[3, 1].imshow(combined_medium, cmap='gray')
    axes[3, 1].set_title(f'Medium (sim>0.6, depth<20mm)\nArea: {combined_medium.sum()}')
    
    axes[3, 2].imshow(combined_tight, cmap='gray')
    axes[3, 2].set_title(f'Tight (sim>0.7, depth<10mm)\nArea: {combined_tight.sum()}')
    
    # 生成最终 heatmap（使用 medium 参数）
    from scipy.ndimage import distance_transform_edt
    if combined_medium.sum() > 0:
        dt = distance_transform_edt(combined_medium)
        dt = dt / (dt.max() + 1e-6)
        heatmap = dt * similarity_map * combined_medium.astype(np.float32)
    else:
        # Fallback
        y, x = np.ogrid[:H, :W]
        dist_sq = (x - px) ** 2 + (y - py) ** 2
        heatmap = np.exp(-dist_sq / (2 * 30 ** 2))
    
    if episode.outcome == "failure":
        y, x = np.ogrid[:H, :W]
        dist_sq = (x - px) ** 2 + (y - py) ** 2
        failure_mask = np.exp(-dist_sq / (2 * 50 ** 2))
        heatmap = heatmap - failure_mask
    
    heatmap = np.clip(heatmap, -1, 1)
    
    axes[3, 3].imshow(heatmap, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[3, 3].scatter([px], [py], c='black', s=100, marker='*')
    axes[3, 3].set_title(f'Final Heatmap\nmin={heatmap.min():.2f}, max={heatmap.max():.2f}')
    
    plt.tight_layout()
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{episode.episode_id}_debug.png", dpi=150)
        print(f"Saved to {save_dir / f'{episode.episode_id}_debug.png'}")
    
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*50)
    print(f"Episode: {episode.episode_id}")
    print(f"Outcome: {episode.outcome}")
    print(f"Pick pixel: {episode.pick_pixel}")
    print(f"Pick UV: {episode.pick_uv}")
    print(f"Depth at pick: {pick_depth:.1f}mm")
    print(f"Depth range: {episode.depth.min():.1f} - {episode.depth.max():.1f}mm")
    print(f"Similarity range: {similarity_map.min():.3f} - {similarity_map.max():.3f}")
    print(f"Object mask area: {object_mask.sum()} / {H*W} ({100*object_mask.sum()/(H*W):.1f}%)")
    print("="*50)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=str, default="./replaybuffer")
    parser.add_argument("--num-episodes", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="./debug_gt")
    args = parser.parse_args()
    
    loader = ReplayBufferLoader(args.buffer)
    print(f"Found {len(loader)} episodes")
    
    for i in range(min(args.num_episodes, len(loader))):
        episode = loader.load_episode(i)
        if episode is not None:
            debug_gt_generation(episode, save_dir=args.save_dir)


if __name__ == "__main__":
    main()