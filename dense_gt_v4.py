# dense_gt_generator_v4_curvature.py
"""
Dense GT 生成器 V4 - 融合点云曲率
===================================

设计理念:
1. 曲率提供几何可行性先验 (底层信号)
2. DINO相似度提供语义相关性 (中层信号)
3. 距离衰减 + Mask 提供任务信号 (顶层信号)

融合方式:
- flatness_score = 1 - normalized_curvature  (平坦度，越高越好)
- semantic_score = similarity * distance_decay * mask_boost
- final_score = flatness_score^alpha * semantic_score^(1-alpha)  (几何平均)

这样设计的好处:
- 曲率高的区域会被自然抑制，即使语义相似
- 全图都有信号，形成平滑梯度
- 可通过 alpha 调节几何/语义的相对重要性
"""

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from scipy.ndimage import binary_dilation, gaussian_filter
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from transformers import AutoModel, AutoImageProcessor
import cv2


# ============ 数据结构 ============

@dataclass
class EpisodeData:
    """单次抓取的完整数据"""
    episode_id: str
    rgb: np.ndarray                      # (H, W, 3) uint8
    depth: np.ndarray                    # (H, W) float32, meters
    masks: List[np.ndarray]              # 分割masks列表
    pick_uv: Tuple[float, float]         # 归一化坐标 [0,1]
    pick_pixel: Tuple[int, int]          # 像素坐标 (x, y)
    outcome: str                         # "success" or "failure"
    mask_idx: int                        # 选中的mask索引


@dataclass
class CameraIntrinsics:
    """相机内参"""
    width: int = 640
    height: int = 480
    fx: float = 781.6494140625
    fy: float = 780.643310546875
    cx: float = 632.9419555664062
    cy: float = 396.1692199707031

    
@dataclass 
class DenseGTConfig:
    """Dense GT 生成器配置 - 所有参数集中管理"""
    # 曲率相关
    curvature_radius: float = 0.015          # 曲率计算邻域半径(米)
    curvature_max_nn: int = 30               # 最大邻居数
    flatness_power: float = 2.0              # 平坦度的幂次(增强对比度)
    
    # 点云预处理
    voxel_size: float = 0.003                # 体素下采样大小(米)
    depth_trunc: float = 2.0                 # 深度截断(米)
    outlier_nb_neighbors: int = 20           # 离群点检测邻居数
    outlier_std_ratio: float = 2.0           # 离群点标准差阈值
    
    # DINO相似度
    dino_model: str = "facebook/dinov2-base"
    similarity_temperature: float = 0.1      # softmax温度
    
    # 距离衰减
    distance_sigma: float = 80.0             # 距离高斯sigma(像素)
    
    # Mask处理
    use_object_mask: bool = True
    mask_dilation_iterations: int = 10       # mask膨胀迭代次数
    mask_boost_inside: float = 1.3           # mask内部增强系数
    mask_suppress_outside: float = 0.5       # mask外部抑制系数
    
    # 融合权重
    geometry_weight: float = 0.4             # 几何信号权重
    semantic_weight: float = 0.6             # 语义信号权重 (应该 = 1 - geometry_weight)
    
    # 后处理
    gaussian_blur_sigma: float = 3.0         # 最终heatmap平滑
    background_min: float = 0.02             # 背景最小值
    
    # 设备
    device: str = "cuda"


# ============ 点云曲率计算模块 ============

class PointCloudCurvatureEstimator:
    """
    基于点云的曲率估计器
    
    核心算法: PCA特征值分解
    - 对每个点，取其邻域点
    - 计算协方差矩阵
    - 最小特征值 / 总特征值 = 曲率
    """
    
    def __init__(self, config: DenseGTConfig, intrinsics: CameraIntrinsics):
        self.config = config
        self.intrinsics = intrinsics
        self._setup_o3d_intrinsics()
        
    def _setup_o3d_intrinsics(self):
        """创建Open3D相机内参对象"""
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.intrinsics.width, self.intrinsics.height,
            self.intrinsics.fx, self.intrinsics.fy,
            self.intrinsics.cx, self.intrinsics.cy
        )
    
    def compute_curvature_map(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算曲率图和法向量图
        
        Returns:
            curvature_map: (H, W) 曲率值，范围 [0, 1]
            normal_map: (H, W, 3) 法向量
        """
        H, W = depth.shape
        
        # Step 1: 深度预处理
        depth_processed = self._preprocess_depth(depth)
        
        # Step 2: 生成点云
        pcd = self._create_point_cloud(rgb, depth_processed)
        if len(pcd.points) < 100:
            print("[WARNING] Too few points in point cloud, returning default maps")
            return np.zeros((H, W), dtype=np.float32), np.zeros((H, W, 3), dtype=np.float32)
        
        # Step 3: 计算法向量和曲率
        normals, curvature = self._compute_geometry_features(pcd)
        
        # Step 4: 投影回2D图像空间
        curvature_map, normal_map = self._project_to_image(
            pcd, curvature, normals, (H, W)
        )
        
        return curvature_map, normal_map
    
    def _preprocess_depth(self, depth: np.ndarray) -> np.ndarray:
        """深度图预处理"""
        depth = depth.copy().astype(np.float32)
        
        # 处理无效值
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        
        # 单位转换：如果最大值>10，假设是mm，转为m
        if np.nanmax(depth) > 10.0:
            depth = depth / 1000.0
            
        return depth
    
    def _create_point_cloud(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray
    ) -> o3d.geometry.PointCloud:
        """从RGB-D创建点云"""
        # 创建Open3D图像对象
        o3d_rgb = o3d.geometry.Image(rgb.astype(np.uint8))
        o3d_depth = o3d.geometry.Image(depth.astype(np.float32))
        
        # 创建RGBD图像
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb, o3d_depth,
            depth_scale=1.0,
            depth_trunc=self.config.depth_trunc,
            convert_rgb_to_intensity=False
        )
        
        # 反投影到点云
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.o3d_intrinsics
        )
        
        # 体素下采样
        pcd = pcd.voxel_down_sample(voxel_size=self.config.voxel_size)
        
        # 统计滤波去噪
        if len(pcd.points) > self.config.outlier_nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self.config.outlier_nb_neighbors,
                std_ratio=self.config.outlier_std_ratio
            )
        
        return pcd
    
    def _compute_geometry_features(
        self, 
        pcd: o3d.geometry.PointCloud
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算法向量和曲率"""
        # 估计法向量
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.config.curvature_radius * 2,
                max_nn=self.config.curvature_max_nn
            )
        )
        # 法向量朝向相机
        pcd.orient_normals_towards_camera_location(
            camera_location=np.array([0., 0., 0.])
        )
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        n_points = len(points)
        
        # 计算曲率
        curvature = np.zeros(n_points, dtype=np.float32)
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        
        for i in range(n_points):
            [k, idx, _] = kdtree.search_hybrid_vector_3d(
                pcd.points[i],
                radius=self.config.curvature_radius,
                max_nn=self.config.curvature_max_nn
            )
            
            if k < 3:
                curvature[i] = 1.0  # 邻居太少，视为高曲率
                continue
            
            # PCA计算曲率
            neighbors = points[idx, :]
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = centered.T @ centered / (k - 1)
            
            try:
                eigenvalues = np.linalg.eigvalsh(cov)
                total_var = np.sum(eigenvalues)
                if total_var > 1e-8:
                    curvature[i] = eigenvalues[0] / total_var
                else:
                    curvature[i] = 0.0
            except np.linalg.LinAlgError:
                curvature[i] = 0.5  # 数值错误时给中等曲率
        
        return normals, curvature
    
    def _project_to_image(
        self,
        pcd: o3d.geometry.PointCloud,
        curvature: np.ndarray,
        normals: np.ndarray,
        image_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """将点云特征投影回2D图像空间"""
        H, W = image_size
        points = np.asarray(pcd.points)
        
        # 初始化输出
        curvature_map = np.full((H, W), np.nan, dtype=np.float32)
        normal_map = np.zeros((H, W, 3), dtype=np.float32)
        depth_buffer = np.full((H, W), np.inf, dtype=np.float32)  # 用于z-buffer
        
        # 投影公式: u = fx * x/z + cx, v = fy * y/z + cy
        valid_z = points[:, 2] > 0.01
        
        u = (points[:, 0] * self.intrinsics.fx / points[:, 2]) + self.intrinsics.cx
        v = (points[:, 1] * self.intrinsics.fy / points[:, 2]) + self.intrinsics.cy
        
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        
        # 过滤有效像素
        valid_uv = valid_z & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        
        # Z-buffer方式写入（保留最近的点）
        for i in np.where(valid_uv)[0]:
            ui, vi = u[i], v[i]
            z = points[i, 2]
            if z < depth_buffer[vi, ui]:
                depth_buffer[vi, ui] = z
                curvature_map[vi, ui] = curvature[i]
                normal_map[vi, ui] = normals[i]
        
        # 插值填充空洞
        curvature_map = self._fill_holes(curvature_map)
        
        # 归一化曲率到 [0, 1]
        valid_curv = curvature_map[~np.isnan(curvature_map)]
        if len(valid_curv) > 0:
            # 使用百分位数归一化，避免极值影响
            p5, p95 = np.percentile(valid_curv, [5, 95])
            curvature_map = np.clip(curvature_map, p5, p95)
            curvature_map = (curvature_map - p5) / (p95 - p5 + 1e-8)
        
        curvature_map = np.nan_to_num(curvature_map, nan=0.5)
        
        return curvature_map, normal_map
    
    def _fill_holes(self, image: np.ndarray, max_iterations: int = 5) -> np.ndarray:
        """使用迭代膨胀填充空洞"""
        result = image.copy()
        mask = np.isnan(result)
        
        for _ in range(max_iterations):
            if not np.any(mask):
                break
            
            # 使用中值滤波填充
            temp = result.copy()
            temp[mask] = 0
            
            # 简单的邻域平均填充
            from scipy.ndimage import uniform_filter
            smoothed = uniform_filter(temp, size=5)
            count = uniform_filter((~mask).astype(float), size=5)
            
            fill_values = np.divide(smoothed, count, where=count > 0)
            result[mask] = fill_values[mask]
            
            mask = np.isnan(result)
        
        return result


# ============ DINO特征提取模块 ============

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
        """提取DINO patch特征"""
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        
        # 获取patch tokens (去掉CLS token)
        patch_tokens = outputs.last_hidden_state[:, 1:, :]
        
        # Reshape到空间维度
        h = inputs['pixel_values'].shape[2] // self.patch_size
        w = inputs['pixel_values'].shape[3] // self.patch_size
        features = patch_tokens.reshape(1, h, w, -1)
        
        return features
    
    def compute_similarity_map(
        self,
        features: torch.Tensor,
        point_uv: Tuple[float, float],
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """计算到指定点的相似度图"""
        _, h, w, D = features.shape
        H, W = target_size
        
        # 获取目标点特征
        feat_x = int(point_uv[0] * w)
        feat_y = int(point_uv[1] * h)
        feat_x = min(max(feat_x, 0), w - 1)
        feat_y = min(max(feat_y, 0), h - 1)
        point_feat = features[0, feat_y, feat_x, :]
        
        # 计算余弦相似度
        features_flat = features[0].reshape(-1, D)
        point_feat = point_feat / (point_feat.norm() + 1e-6)
        features_flat = features_flat / (features_flat.norm(dim=-1, keepdim=True) + 1e-6)
        
        similarity = (features_flat @ point_feat).reshape(h, w)
        similarity = similarity.cpu().numpy()
        
        # 上采样到目标尺寸
        sim_tensor = torch.from_numpy(similarity).unsqueeze(0).unsqueeze(0).float()
        sim_upsampled = F.interpolate(
            sim_tensor, size=target_size, mode='bilinear', align_corners=False
        )
        sim_map = sim_upsampled[0, 0].numpy()
        
        # 归一化到 [0, 1]
        sim_map = np.clip(sim_map, 0, None)
        if sim_map.max() > sim_map.min():
            sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
        
        return sim_map.astype(np.float32)


# ============ 主生成器类 ============

class DenseGTGeneratorV4:
    """
    Dense GT 生成器 V4 - 融合点云曲率
    
    信号融合流程:
    ┌─────────────────────────────────────────────────────────────┐
    │  RGB + Depth                                                 │
    │      │                                                       │
    │      ├──► 点云曲率 ──► 平坦度图 (flatness_map)               │
    │      │                    │                                  │
    │      │                    ▼                                  │
    │      │              ┌─────────────┐                          │
    │      ├──► DINO ──►  │   加权融合   │ ──► Dense GT Heatmap   │
    │      │              └─────────────┘                          │
    │      │                    ▲                                  │
    │      ├──► 距离衰减 ───────┤                                  │
    │      │                    │                                  │
    │      └──► Object Mask ────┘                                  │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        config: DenseGTConfig = None,
        intrinsics: CameraIntrinsics = None
    ):
        self.config = config or DenseGTConfig()
        self.intrinsics = intrinsics or CameraIntrinsics()
        
        # 初始化子模块
        self.curvature_estimator = PointCloudCurvatureEstimator(
            self.config, self.intrinsics
        )
        self.dino_extractor = DINOFeatureExtractor(
            model_name=self.config.dino_model,
            device=self.config.device
        )
        
        print(f"[DenseGTGeneratorV4] Initialized with config:")
        print(f"  - Geometry weight: {self.config.geometry_weight}")
        print(f"  - Semantic weight: {self.config.semantic_weight}")
        print(f"  - Curvature radius: {self.config.curvature_radius}m")
    
    def generate(self, episode: EpisodeData) -> Dict[str, np.ndarray]:
        """
        生成Dense GT Heatmap
        
        Returns:
            dict containing:
                - 'heatmap': 最终融合的heatmap (H, W)
                - 'flatness_map': 平坦度图 (H, W)
                - 'similarity_map': DINO相似度图 (H, W)
                - 'distance_map': 距离衰减图 (H, W)
                - 'curvature_map': 原始曲率图 (H, W)
        """
        H, W = episode.rgb.shape[:2]
        px, py = episode.pick_pixel
        px, py = min(max(px, 0), W-1), min(max(py, 0), H-1)
        
        # ========== Step 1: 计算几何特征 (曲率/平坦度) ==========
        print("  [1/5] Computing curvature from point cloud...")
        curvature_map, normal_map = self.curvature_estimator.compute_curvature_map(
            episode.rgb, episode.depth
        )
        
        # 平坦度 = 1 - 曲率，然后增强对比度
        flatness_map = 1.0 - curvature_map
        flatness_map = np.power(flatness_map, self.config.flatness_power)
        
        # ========== Step 2: 计算语义相似度 (DINO) ==========
        print("  [2/5] Computing DINO similarity...")
        features = self.dino_extractor.extract_features(episode.rgb)
        similarity_map = self.dino_extractor.compute_similarity_map(
            features, episode.pick_uv, (H, W)
        )
        
        # ========== Step 3: 计算距离衰减 ==========
        print("  [3/5] Computing distance decay...")
        distance_map = self._compute_distance_decay((H, W), (px, py))
        
        # ========== Step 4: 应用Object Mask ==========
        print("  [4/5] Applying object mask...")
        mask_weight = self._compute_mask_weight(episode, (H, W))
        
        # ========== Step 5: 融合所有信号 ==========
        print("  [5/5] Fusing signals...")
        
        # 语义信号 = 相似度 × 距离衰减 × mask权重
        semantic_signal = similarity_map * distance_map * mask_weight
        
        # 几何信号 = 平坦度
        geometry_signal = flatness_map
        
        # 融合策略：加权几何平均
        # final = geometry^α × semantic^(1-α)
        # 这保证了：曲率高(geometry低)的区域会被抑制
        alpha = self.config.geometry_weight
        
        # 避免0值导致的问题
        geometry_signal = np.clip(geometry_signal, 0.01, 1.0)
        semantic_signal = np.clip(semantic_signal, 0.01, 1.0)
        
        heatmap = np.power(geometry_signal, alpha) * np.power(semantic_signal, 1 - alpha)
        
        # ========== Step 6: 后处理 ==========
        # 添加背景最小值
        background = self._compute_distance_decay(
            (H, W), (px, py), sigma=200.0
        ) * self.config.background_min
        heatmap = np.maximum(heatmap, background)
        
        # 高斯平滑
        if self.config.gaussian_blur_sigma > 0:
            heatmap = gaussian_filter(heatmap, sigma=self.config.gaussian_blur_sigma)
        
        # 归一化到 [0, 1]
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # 处理成功/失败
        if episode.outcome != "success":
            heatmap = 1.0 - heatmap
        
        return {
            'heatmap': heatmap.astype(np.float32),
            'flatness_map': flatness_map.astype(np.float32),
            'similarity_map': similarity_map.astype(np.float32),
            'distance_map': distance_map.astype(np.float32),
            'curvature_map': curvature_map.astype(np.float32),
            'mask_weight': mask_weight.astype(np.float32)
        }
    
    def _compute_distance_decay(
        self,
        size: Tuple[int, int],
        center: Tuple[int, int],
        sigma: float = None
    ) -> np.ndarray:
        """计算高斯距离衰减"""
        if sigma is None:
            sigma = self.config.distance_sigma
        
        H, W = size
        cx, cy = center
        y, x = np.ogrid[:H, :W]
        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        gaussian = np.exp(-dist_sq / (2 * sigma ** 2))
        
        return gaussian.astype(np.float32)
    
    def _compute_mask_weight(
        self,
        episode: EpisodeData,
        size: Tuple[int, int]
    ) -> np.ndarray:
        """计算mask权重图"""
        H, W = size
        
        if not self.config.use_object_mask:
            return np.ones((H, W), dtype=np.float32)
        
        if episode.masks is None or len(episode.masks) == 0:
            return np.ones((H, W), dtype=np.float32)
        
        try:
            idx = min(episode.mask_idx, len(episode.masks) - 1)
            mask_item = episode.masks[idx]
            
            if isinstance(mask_item, dict) and 'mask' in mask_item:
                mask = np.array(mask_item['mask'])
            elif isinstance(mask_item, np.ndarray):
                mask = mask_item
            else:
                return np.ones((H, W), dtype=np.float32)
            
            # 确保mask尺寸匹配
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H)) > 0
            
            # 膨胀mask
            mask = binary_dilation(
                mask, 
                iterations=self.config.mask_dilation_iterations
            )
            
            # 创建权重图
            weight = np.where(
                mask,
                self.config.mask_boost_inside,
                self.config.mask_suppress_outside
            )
            
            return weight.astype(np.float32)
            
        except Exception as e:
            print(f"[WARNING] Failed to process mask: {e}")
            return np.ones((H, W), dtype=np.float32)


# ============ 可视化和测试 ============

def visualize_v4_results(
    episode: EpisodeData,
    results: Dict[str, np.ndarray],
    save_path: str = None
):
    """可视化V4生成器的所有中间结果"""
    import matplotlib.pyplot as plt
    
    px, py = episode.pick_pixel
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: 输入和中间结果
    axes[0, 0].imshow(episode.rgb)
    axes[0, 0].scatter([px], [py], c='red', s=200, marker='*', edgecolors='white', linewidths=2)
    axes[0, 0].set_title(f'RGB Input\n({episode.outcome})')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(results['curvature_map'], cmap='viridis')
    axes[0, 1].scatter([px], [py], c='red', s=100, marker='*')
    axes[0, 1].set_title('Curvature Map\n(from Point Cloud)')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
    
    im2 = axes[0, 2].imshow(results['flatness_map'], cmap='hot')
    axes[0, 2].scatter([px], [py], c='cyan', s=100, marker='*')
    axes[0, 2].set_title('Flatness Map\n(1 - curvature)^α')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
    
    im3 = axes[0, 3].imshow(results['similarity_map'], cmap='hot')
    axes[0, 3].scatter([px], [py], c='cyan', s=100, marker='*')
    axes[0, 3].set_title('DINO Similarity')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)
    
    # Row 2: 更多中间结果和最终输出
    im4 = axes[1, 0].imshow(results['distance_map'], cmap='hot')
    axes[1, 0].scatter([px], [py], c='cyan', s=100, marker='*')
    axes[1, 0].set_title('Distance Decay')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    
    im5 = axes[1, 1].imshow(results['mask_weight'], cmap='RdYlGn')
    axes[1, 1].scatter([px], [py], c='blue', s=100, marker='*')
    axes[1, 1].set_title('Mask Weight')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)
    
    im6 = axes[1, 2].imshow(results['heatmap'], cmap='hot', vmin=0, vmax=1)
    axes[1, 2].scatter([px], [py], c='cyan', s=100, marker='*')
    axes[1, 2].set_title(f'Final Heatmap\nmin={results["heatmap"].min():.3f}, max={results["heatmap"].max():.3f}')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)
    
    # Overlay
    axes[1, 3].imshow(episode.rgb)
    axes[1, 3].imshow(results['heatmap'], cmap='hot', alpha=0.6, vmin=0, vmax=1)
    axes[1, 3].scatter([px], [py], c='cyan', s=100, marker='*')
    axes[1, 3].set_title('RGB + Heatmap Overlay')
    axes[1, 3].axis('off')
    
    plt.suptitle(f'Dense GT V4 (Curvature-Fused) - Episode: {episode.episode_id}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Heatmap Statistics:")
    print(f"  Shape: {results['heatmap'].shape}")
    print(f"  Range: [{results['heatmap'].min():.4f}, {results['heatmap'].max():.4f}]")
    print(f"  Mean: {results['heatmap'].mean():.4f}")
    print(f"  Std: {results['heatmap'].std():.4f}")
    print(f"  Pixels > 0.5: {(results['heatmap'] > 0.5).sum()} ({100*(results['heatmap'] > 0.5).mean():.1f}%)")
    print(f"  Pixels > 0.8: {(results['heatmap'] > 0.8).sum()} ({100*(results['heatmap'] > 0.8).mean():.1f}%)")
    print("="*60)


def create_dummy_episode() -> EpisodeData:
    """创建测试用的dummy数据"""
    width, height = 640, 480
    
    # RGB: 灰色背景 + 彩色物体
    rgb = np.ones((height, width, 3), dtype=np.uint8) * 100
    rgb[100:200, 100:200] = [200, 100, 100]  # 红色方块
    rgb[200:350, 300:450] = [100, 200, 100]  # 绿色方块
    
    # Depth: 1m平面 + 凸起物体
    x = np.linspace(-1, 1, width)
    y = np.linspace(-0.75, 0.75, height)
    xv, yv = np.meshgrid(x, y)
    depth = np.ones_like(xv) * 1.0
    
    # 红色区域是平坦的，靠近相机
    depth[100:200, 100:200] = 0.8
    
    # 绿色区域有球面凸起
    center_y, center_x = 275, 375
    mask = ((xv - 0.17)**2 + (yv + 0.15)**2) < 0.15**2
    depth[mask] -= np.sqrt(0.15**2 - (xv[mask] - 0.17)**2 - (yv[mask] + 0.15)**2) * 0.3
    
    # 添加噪声
    depth += np.random.normal(0, 0.002, depth.shape)
    
    # 创建mask
    mask1 = np.zeros((height, width), dtype=bool)
    mask1[100:200, 100:200] = True
    
    mask2 = np.zeros((height, width), dtype=bool)
    mask2[200:350, 300:450] = True
    
    # pick点在红色平坦区域
    pick_pixel = (150, 150)  # x, y
    pick_uv = (150/width, 150/height)
    
    return EpisodeData(
        episode_id="dummy_test",
        rgb=rgb,
        depth=depth.astype(np.float32),
        masks=[mask1, mask2],
        pick_uv=pick_uv,
        pick_pixel=pick_pixel,
        outcome="success",
        mask_idx=0
    )


# ============ 主函数 ============

if __name__ == "__main__":
    print("="*60)
    print("Dense GT Generator V4 - Curvature Fusion Test")
    print("="*60)
    
    # 初始化生成器
    print("\n[1] Initializing generator...")
    config = DenseGTConfig(
        geometry_weight=0.4,
        semantic_weight=0.6,
        curvature_radius=0.015,
        flatness_power=4.0,
        distance_sigma=70.0,
        gaussian_blur_sigma=3.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print("\n[2] Initializing generator...")
    generator = DenseGTGeneratorV4(config=config)

    # 创建测试数据
    from dense_gt_generator_old import ReplayBufferLoader
    loader = ReplayBufferLoader("./replaybuffer")
    # 测试多个 episode
    for i in range(min(3, len(loader))):
        episode = loader.load_episode(i)
    
        # 生成GT
        print("\n[3] Generating Dense GT...")
        results = generator.generate(episode)
        
        # 可视化
        print("\n[4] Visualizing results...")
        visualize_v4_results(episode, results, save_path=f"./debug_gt_v4_ep{i}.png")
        
        print("\n[DONE] Test completed!")
