import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os

class SuctionGraspBaseline:
    """
    A baseline implementation for suction grasp detection based on 
    surface normal estimation and curvature thresholding.
    """

    def __init__(self, 
                 camera_intrinsics: dict = None, 
                 voxel_size: float = 0.005):
        """
        Args:
            camera_intrinsics: Dictionary containing fx, fy, cx, cy. 
                               If None, uses RealSense D435 defaults.
            voxel_size: Downsampling size for faster processing (in meters).
        """
        # Default to RealSense D435 640x480 intrinsics if not provided
        self.intrinsics = camera_intrinsics or {
            'width': 640, 'height': 480,
            'fx': 605.0, 'fy': 605.0,
            'cx': 320.0, 'cy': 240.0
        }
        self.voxel_size = voxel_size
        
        # Create Open3D intrinsic object
        self.o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            self.intrinsics['width'], self.intrinsics['height'],
            self.intrinsics['fx'], self.intrinsics['fy'],
            self.intrinsics['cx'], self.intrinsics['cy']
        )

    def load_data(self, rgb_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Loads aligned RGB and Depth images."""
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            raise FileNotFoundError("Input files not found.")

        # Load RGB
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load Depth (.npy usually contains float meters or mm)
        depth = np.load(depth_path)
        
        # Sanity check: Ensure depth is in meters for Open3D
        # Heuristic: if max value > 100, it's likely mm, convert to meters
        if np.nanmax(depth) > 10.0: 
            print("[INFO] Depth max > 10.0, assuming millimeters. Converting to meters.")
            depth = depth / 1000.0

        return rgb, depth

    def preprocess_point_cloud(self, rgb: np.ndarray, depth: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Converts RGB-D to PointCloud and removes outliers.
        """
        # Create Open3D images
        o3d_rgb = o3d.geometry.Image(rgb)
        o3d_depth = o3d.geometry.Image(depth.astype(np.float32))

        # Create RGBD Image (assuming depth is already aligned)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_rgb, o3d_depth, 
            depth_scale=1.0, 
            depth_trunc=2.0, # Truncate depth at 2 meters
            convert_rgb_to_intensity=False
        )

        # Back-project to Point Cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, self.o3d_intrinsics
        )

        # Voxel Downsampling (Crucial for speed and noise reduction)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        # Statistical Outlier Removal (Remove flying pixels)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        return pcd

    def compute_geometry_features(self, pcd: o3d.geometry.PointCloud, radius: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        print("      Estimating normals...")
        # 1. 计算法向量 (使用混合搜索：半径 2cm 或 30个邻居)
        # 注意：如果你的物体非常小，radius=0.02 (2cm) 可能需要调整
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        n_points = len(points)
        
        print(f"      Computing curvature for {n_points} points...")
        
        # 2. 预分配曲率数组 (关键修复：先生成全0数组，确保长度一致)
        curvature = np.zeros(n_points)
        
        # 建立 KDTree 用于搜索邻居
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        
        # 遍历所有点计算曲率
        # 为了速度，这里只用了简单的 Python 循环。如果点数 > 10万会慢，但 2万点没问题。
        for i in range(n_points):
            # 搜索邻居：半径 2cm，最多 30 个点
            [k, idx, _] = kdtree.search_hybrid_vector_3d(pcd.points[i], radius=0.02, max_nn=30)
            
            # 如果邻居太少（<3个），无法计算平面，设为高曲率（不平）
            if k < 3:
                curvature[i] = 1.0 
                continue
            
            # 获取邻居点
            neighbors = points[idx, :]
            
            # PCA (主成分分析) 计算特征值
            # 协方差矩阵
            mean = np.mean(neighbors, axis=0)
            centered = neighbors - mean
            cov = centered.T @ centered / (k - 1)
            
            # 特征值分解 (eigvalsh 返回升序排列的特征值)
            eigenvalues = np.linalg.eigvalsh(cov)
            
            # 曲率近似公式: min_eigen_value / sum_eigen_values
            # 最小特征值越小，说明在该方向变化越小（越平）
            total_variance = np.sum(eigenvalues)
            if total_variance > 1e-8:
                c = eigenvalues[0] / total_variance
                curvature[i] = c
            else:
                curvature[i] = 0.0

        print(f"      Geometry computed. Normals: {normals.shape}, Curvature: {curvature.shape}")
        return normals, curvature

    def evaluate_suction_candidates(self, 
                                    normals: np.ndarray, 
                                    curvature: np.ndarray, 
                                    curvature_thresh: float = 0.05,
                                    angle_thresh_deg: float = 30.0) -> np.ndarray:
        """
        Filters points based on flatness (curvature) and orientation.
        Robust version to handle cases with 0 valid points.
        """
        # Criteria 1: Flatness (Low curvature)
        mask_flat = curvature < curvature_thresh

        # Criteria 2: Orientation
        # Calculate angle with Camera Z-axis (assuming camera looks down -Z or +Z)
        # We use absolute value of Z component of the normal
        nz = np.abs(normals[:, 2])
        min_z_component = np.cos(np.deg2rad(angle_thresh_deg))
        mask_angle = nz > min_z_component

        # Combine masks
        valid_mask = mask_flat & mask_angle
        
        # Initialize scores with zeros
        scores = np.zeros_like(curvature)
        
        # --- FIX: Check if any points exist before assignment ---
        num_valid = np.sum(valid_mask)
        if num_valid > 0:
            # Extract only the valid curvature values (Shape: [N_valid,])
            valid_curvature = curvature[valid_mask]
            
            # Calculate scores for these points
            # Score 1.0 means perfectly flat (curvature 0)
            # Score 0.0 means curvature == curvature_thresh
            point_scores = 1.0 - (valid_curvature / curvature_thresh)
            
            # Clip to ensure range [0, 1] (in case curvature slightly exceeds thresh due to float precision)
            point_scores = np.clip(point_scores, 0.0, 1.0)
            
            # Assign back using the same mask
            scores[valid_mask] = point_scores
        
        return scores
    def visualize(self, rgb: np.ndarray, pcd: o3d.geometry.PointCloud, scores: np.ndarray):
        """
        Visualizes the results in 2D (projected) and 3D.
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # --- 3D Visualization ---
        # Colorize point cloud based on scores (Red = Good, Grey = Bad)
        # Create a new PCD for visualization
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = pcd.points
        
        # Base color: Grey
        vis_colors = np.ones((len(points), 3)) * 0.5 
        
        # High score points -> Red/Green gradient
        valid_indices = scores > 0
        if np.any(valid_indices):
            # Map score 0-1 to Yellow-Red
            # Simple: Set valid points to Red
            vis_colors[valid_indices] = [1, 0, 0] 
            
        vis_pcd.colors = o3d.utility.Vector3dVector(vis_colors)
        
        print("[INFO] Opening 3D Visualization window... (Close to continue)")
        o3d.visualization.draw_geometries([vis_pcd], window_name="Suction Candidates (Red)")

        # --- 2D Projection Visualization ---
        # Project 3D points back to 2D image plane to visualize heatmap on RGB
        # x_2d = fx * x_3d / z_3d + cx
        u = (points[:, 0] * self.intrinsics['fx'] / points[:, 2]) + self.intrinsics['cx']
        v = (points[:, 1] * self.intrinsics['fy'] / points[:, 2]) + self.intrinsics['cy']
        
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
        
        # Filter valid indices within image bounds
        h, w, _ = rgb.shape
        valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        
        # Create Heatmap Overlay
        heatmap = np.zeros((h, w), dtype=np.float32)
        # We use simple assignment here (last point wins), for better results use splatting
        heatmap[v[valid_uv], u[valid_uv]] = scores[valid_uv]
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Original RGB")
        plt.imshow(rgb)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Suction Score Map")
        plt.imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Overlay")
        plt.imshow(rgb)
        plt.imshow(heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def run(self, rgb_path: str, depth_path: str):
        print(f"[1/5] Loading data from {rgb_path}...")
        rgb, depth = self.load_data(rgb_path, depth_path)
        
        print("[2/5] Preprocessing Point Cloud...")
        pcd = self.preprocess_point_cloud(rgb, depth)
        print(f"      Points after downsampling: {len(pcd.points)}")
        
        print("[3/5] Computing Geometry (Normals & Curvature)...")
        normals, curvature = self.compute_geometry_features(pcd, radius=0.015) # 1.5cm radius
        
        print("[4/5] Evaluating Suction Candidates...")
        scores = self.evaluate_suction_candidates(normals, curvature, curvature_thresh=0.2,angle_thresh_deg=60.0 )
        num_candidates = np.sum(scores > 0)
        print(f"      Found {num_candidates} valid suction points.")
        
        print("[5/5] Visualizing...")
        self.visualize(rgb, pcd, scores)

# --- Mock Data Generator for Testing (Remove this block if you have real files) ---
def create_dummy_data():
    width, height = 640, 480
    # RGB: Gray background
    rgb = np.ones((height, width, 3), dtype=np.uint8) * 100
    # Depth: Plane at 1.0m with a sphere bump
    x = np.linspace(-1, 1, width)
    y = np.linspace(-0.75, 0.75, height)
    xv, yv = np.meshgrid(x, y)
    depth = np.ones_like(xv) * 1.0 # 1 meter flat
    
    # Add a sphere (curved surface)
    mask = (xv**2 + yv**2) < 0.3**2
    depth[mask] -= np.sqrt(0.3**2 - xv[mask]**2 - yv[mask]**2) * 0.5
    
    # Add a flat box
    depth[100:200, 100:200] = 0.8
    rgb[100:200, 100:200] = [200, 100, 100] # Red box

    # Add noise
    depth += np.random.normal(0, 0.002, depth.shape)

    cv2.imwrite("step0_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save("step0_depth.npy", depth)
    print("Generated dummy step0_rgb.png and step0_depth.npy")

if __name__ == "__main__":
    # If files don't exist, create dummy data for demonstration
    if not os.path.exists("step0_depth.npy"):
        create_dummy_data()

    # Initialize pipeline
    # Note: Adjust intrinsics if you know your camera's specific parameters
    pipeline = SuctionGraspBaseline()
    
    pipeline.run("step0_rgb.png", "step0_depth.npy")