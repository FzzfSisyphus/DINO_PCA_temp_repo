"""
example_usage.py
使用示例
"""

import numpy as np
from PIL import Image

# 初始化
device = "cuda"

# 1. 加载预训练模型
dino = DINOFeatureExtractor(device=device)
model = AffordanceDiffusion(image_size=224, device=device)
model.load_state_dict(torch.load("checkpoints/best_model.pt")['model_state_dict'])

sampler = DDIMSampler(model, ddim_steps=50)
engine = InferenceEngine(model, dino, sampler, num_samples=10)

# 2. 推理
rgb = np.array(Image.open("test_rgb.png"))
depth = np.load("test_depth.npy")

candidates, mean_heatmap, var_heatmap = engine.predict(rgb, depth, top_k=5)

# 3. 选择最佳抓取点
best = candidates[0]
print(f"Best pick: pixel={best.pixel}, UCB={best.ucb_score:.3f}")

# 4. 执行抓取（调用机器人 API）
# robot.pick(best.pixel)

# 5. 收集结果，进入自我改进循环
# self_improving_loop.on_pick_complete(episode_data)

运行
