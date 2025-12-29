"""
train.py
完整的训练流程
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional
import torchvision.transforms as T
from dense_gt_generator_old import ReplayBufferLoader
from dense_gt_generator import DenseGTGenerator
from diffusion_heatmap_model import AffordanceDiffusion
from dense_gt_generator import DINOFeatureExtractor
import torch.multiprocessing as mp

class BinPickingDataset(Dataset):
    """Bin Picking 数据集"""
    
    def __init__(
        self,
        buffer_loader: 'ReplayBufferLoader',
        gt_generator: 'DenseGTGenerator',
        image_size: int = 224,
        augment: bool = True,
        cache_gt: bool = True
    ):
        self.buffer_loader = buffer_loader
        self.gt_generator = gt_generator
        self.image_size = image_size
        self.augment = augment
        self.cache_gt = cache_gt
        
        # 预加载所有有效 episode
        self.episodes = list(buffer_loader.iterate_episodes())
        print(f"Loaded {len(self.episodes)} episodes")
        
        # GT 缓存
        self.gt_cache: Dict[str, np.ndarray] = {}
        
        # 图像变换
        self.rgb_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if augment:
            self.augment_transform = T.Compose([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode = self.episodes[idx]
        
        # 生成或获取缓存的 GT
        if self.cache_gt and episode.episode_id in self.gt_cache:
            gt_heatmap = self.gt_cache[episode.episode_id]
        else:
            gt_heatmap = self.gt_generator.generate_dense_gt(episode)
            if self.cache_gt:
                self.gt_cache[episode.episode_id] = gt_heatmap
        
        # 预处理 RGB
        rgb_pil = Image.fromarray(episode.rgb)
        if self.augment:
            rgb_pil = self.augment_transform(rgb_pil)
        rgb = self.rgb_transform(rgb_pil)
        
        # 预处理深度图
        depth = episode.depth.copy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)  # normalize
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        depth = T.functional.resize(depth, [self.image_size, self.image_size])
        
        # 预处理 GT heatmap
        gt = torch.from_numpy(gt_heatmap).float().unsqueeze(0)
        gt = T.functional.resize(gt, [self.image_size, self.image_size])
        
        return {
            'rgb': rgb,            # (3, H, W)
            'depth': depth,        # (1, H, W)
            'gt_heatmap': gt,      # (1, H, W)
            'episode_id': episode.episode_id,
            'outcome': 1 if episode.outcome == 'success' else 0
        }


class Trainer:
    """训练器"""
    
    def __init__(
        self,
        model: 'AffordanceDiffusion',
        dino_extractor: 'DINOFeatureExtractor',
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        epochs: int = 100,
        device: str = "cuda",
        log_wandb: bool = True,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model.to(device)
        self.dino = dino_extractor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.log_wandb = log_wandb
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=epochs * len(train_loader),
            eta_min=1e-6
        )
        
        # Best metric tracking
        self.best_val_loss = float('inf')
        
    def train(self):
        """完整训练循环"""
        if self.log_wandb:
            wandb.init(project="bin-picking-diffusion", config={
                "epochs": self.epochs,
                "lr": self.optimizer.defaults['lr'],
            })
        
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(epoch)
            
            if self.val_loader:
                val_loss = self._validate(epoch)
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint(epoch, is_best=True)
            
            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
        
        if self.log_wandb:
            wandb.finish()
    
    def _train_epoch(self, epoch: int) -> float:
        """单个 epoch 训练"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            rgb = batch['rgb'].to(self.device)
            depth = batch['depth'].to(self.device)
            gt_heatmap = batch['gt_heatmap'].to(self.device)
            
            # 提取 DINO 特征
            with torch.no_grad():
                # 需要反归一化给 DINO
                rgb_unnorm = rgb * torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                rgb_unnorm = rgb_unnorm + torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                rgb_unnorm = (rgb_unnorm * 255).byte()
                
                # 批量提取特征
                dino_features = self._batch_dino_features(rgb_unnorm)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model.training_step(rgb, depth, dino_features, gt_heatmap)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            if self.log_wandb:
                wandb.log({
                    'train_loss': loss.item(),
                    'lr': self.scheduler.get_last_lr()[0]
                })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def _batch_dino_features(self, rgb_batch: torch.Tensor) -> torch.Tensor:
        """批量提取 DINO 特征"""
        B = rgb_batch.shape[0]
        features_list = []
        
        for i in range(B):
            rgb_np = rgb_batch[i].permute(1, 2, 0).cpu().numpy()
            feat = self.dino.extract_features(rgb_np)  # (1, h, w, D)
            feat = feat.permute(0, 3, 1, 2)  # (1, D, h, w)
            features_list.append(feat)
        
        return torch.cat(features_list, dim=0)  # (B, D, h, w)
    
    @torch.no_grad()
    def _validate(self, epoch: int) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0
        
        for batch in self.val_loader:
            rgb = batch['rgb'].to(self.device)
            depth = batch['depth'].to(self.device)
            gt_heatmap = batch['gt_heatmap'].to(self.device)
            
            # 提取 DINO 特征
            rgb_unnorm = rgb * torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            rgb_unnorm = rgb_unnorm + torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            rgb_unnorm = (rgb_unnorm * 255).byte()
            dino_features = self._batch_dino_features(rgb_unnorm)
            
            loss = self.model.training_step(rgb, depth, dino_features, gt_heatmap)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        if self.log_wandb:
            wandb.log({'val_loss': avg_loss, 'epoch': epoch})
        
        return avg_loss
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存 checkpoint"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save(state, path)
        print(f"Saved checkpoint to {path}")


# ==================== 主入口 ====================

def main():
    """训练主函数"""
    
    # 配置
    config = {
        'buffer_dir': './replaybuffer',
        'image_size': 224,
        'batch_size': 8,
        'epochs': 40,
        'lr': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Using device: {config['device']}")
    
    # 初始化 DINO
    print("Loading DINO...")
    dino = DINOFeatureExtractor(device=config['device'])
    
    # 初始化 GT 生成器
    gt_generator  = DenseGTGenerator(
        dino_extractor=dino,
        use_object_mask=True,
        mask_boost=0.3,
        distance_sigma=100.0,
        similarity_weight=0.6,
        distance_weight=0.4,
        background_min=0.05
    )
    
    # 加载数据
    print("Loading replay buffer...")
    buffer_loader = ReplayBufferLoader(config['buffer_dir'])
    
    dataset = BinPickingDataset(
        buffer_loader=buffer_loader,
        gt_generator=gt_generator,
        image_size=config['image_size'],
        augment=True
    )
    
    # 划分训练/验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # 初始化模型
    print("Initializing model...")
    model = AffordanceDiffusion(
        image_size=config['image_size'],
        dino_dim=768,
        timesteps=1000,
        device=config['device']
    )
    
    # 训练
    trainer = Trainer(
        model=model,
        dino_extractor=dino,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=config['lr'],
        epochs=config['epochs'],
        device=config['device']
    )
    
    print("Starting training...")
    trainer.train()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()