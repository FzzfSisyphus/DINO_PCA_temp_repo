# DINO_PCA_temp_repo

# 1. 可视化单个 episode
python visualize_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --episode ./replaybuffer/2025-12-15_183425_t0s43y \
    --top-k 5

# 2. 随机可视化 5 个样本
`python visualize_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --buffer ./replaybuffer \
    --num-samples 5 \
    --top-k 5`

# 3. 保存可视化结果（不显示）
`python visualize_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --buffer ./replaybuffer \
    --num-samples 10 \
    --save-dir ./visualizations \
    --no-show`

# 4. 可视化任意图片（需要手动指定查询点）
`python visualize_predictions.py \
    --checkpoint checkpoints/best_model.pth \
    --image ./test.png \
    --query 320 240 \
    --top-k 5`