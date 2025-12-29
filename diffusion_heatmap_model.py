"""
diffusion_heatmap_model.py
基于 U-Net 的条件扩散模型，生成 Affordance Heatmap
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from einops import rearrange, repeat


# ==================== 时间嵌入 ====================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding"""
    
    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


# ==================== U-Net Building Blocks ====================

class ConvBlock(nn.Module):
    """Basic conv block with GroupNorm and GELU"""
    
    def __init__(self, in_ch: int, out_ch: int, time_dim: int = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        self.time_mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_dim, out_ch)
        ) if time_dim else None
        
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.gelu(h)
        
        if self.time_mlp is not None and t_emb is not None:
            time_emb = self.time_mlp(t_emb)[:, :, None, None]
            h = h + time_emb
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.gelu(h)
        
        return h + self.residual(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=self.num_heads)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * scale
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=H, y=W)
        
        return x + self.proj(out)


class DownBlock(nn.Module):
    """Downsample block"""
    
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, use_attention: bool = False):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, time_dim)
        self.attn = AttentionBlock(out_ch) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x, t_emb)
        h = self.attn(h)
        return self.downsample(h), h  # 返回 downsampled 和 skip connection


class UpBlock(nn.Module):
    """Upsample block"""
    
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, use_attention: bool = False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2, padding=1)
        self.conv = ConvBlock(in_ch + out_ch, out_ch, time_dim)  # concat skip
        self.attn = AttentionBlock(out_ch) if use_attention else nn.Identity()
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.upsample(x)
        
        # Handle size mismatch
        if h.shape != skip.shape:
            h = F.interpolate(h, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        h = torch.cat([h, skip], dim=1)
        h = self.conv(h, t_emb)
        h = self.attn(h)
        return h


# ==================== 条件编码器 ====================

class ConditionEncoder(nn.Module):
    """
    编码条件信息：DINO 特征 + 深度图 + RGB
    """
    
    def __init__(
        self, 
        dino_dim: int = 768,
        rgb_channels: int = 3,
        depth_channels: int = 1,
        out_channels: int = 256,
        image_size: int = 224
    ):
        super().__init__()
        self.image_size = image_size
        
        # RGB encoder (轻量级)
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(rgb_channels, 32, 7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        
        # Depth encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels, 32, 7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        
        # DINO feature projector
        self.dino_proj = nn.Sequential(
            nn.Conv2d(dino_dim, 128, 1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )
        
        # Fusion: 64 (rgb) + 64 (depth) + 128 (dino) = 256
        self.fusion = nn.Sequential(
            nn.Conv2d(64 + 64 + 128, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )
        
        # Global context (for cross-attention or global conditioning)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_proj = nn.Linear(out_channels, out_channels)
        
    def forward(
        self, 
        rgb: torch.Tensor,           # (B, 3, H, W)
        depth: torch.Tensor,         # (B, 1, H, W)
        dino_features: torch.Tensor  # (B, D, h, w) - 已经是 patch 特征图
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            local_cond: (B, C, H', W') 局部条件特征
            global_cond: (B, C) 全局条件特征
        """
        # Encode RGB and Depth
        rgb_feat = self.rgb_encoder(rgb)    # (B, 64, H/4, W/4)
        depth_feat = self.depth_encoder(depth)  # (B, 64, H/4, W/4)
        
        # Upsample DINO features to match
        target_size = rgb_feat.shape[-2:]
        dino_feat = F.interpolate(dino_features, size=target_size, mode='bilinear', align_corners=False)
        dino_feat = self.dino_proj(dino_feat)  # (B, 128, H/4, W/4)
        
        # Concatenate and fuse
        combined = torch.cat([rgb_feat, depth_feat, dino_feat], dim=1)
        local_cond = self.fusion(combined)  # (B, C, H/4, W/4)
        
        # Global context
        global_feat = self.global_pool(local_cond).squeeze(-1).squeeze(-1)
        global_cond = self.global_proj(global_feat)
        
        return local_cond, global_cond


# ==================== 主 Diffusion U-Net ====================

class DiffusionUNet(nn.Module):
    """
    条件扩散 U-Net，生成 Affordance Heatmap
    
    输入：noisy heatmap + 条件（DINO + Depth + RGB）
    输出：预测的噪声
    """
    
    def __init__(
        self,
        in_channels: int = 1,          # noisy heatmap
        out_channels: int = 1,         # predicted noise
        cond_channels: int = 256,      # condition feature channels
        base_channels: int = 64,
        channel_mults: Tuple[int] = (1, 2, 4, 8),
        attention_resolutions: Tuple[int] = (16, 8),  # at which resolution to use attention
        time_dim: int = 256,
        image_size: int = 224
    ):
        super().__init__()
        self.image_size = image_size
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(time_dim // 4, time_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels + cond_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.downs = nn.ModuleList()
        channels = [base_channels]
        current_ch = base_channels
        current_res = image_size
        
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            use_attn = current_res in attention_resolutions
            self.downs.append(DownBlock(current_ch, out_ch, time_dim, use_attn))
            channels.append(out_ch)
            current_ch = out_ch
            current_res = current_res // 2
        
        # Middle
        self.mid_block1 = ConvBlock(current_ch, current_ch, time_dim)
        self.mid_attn = AttentionBlock(current_ch)
        self.mid_block2 = ConvBlock(current_ch, current_ch, time_dim)
        
        # Upsampling path
        self.ups = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            skip_ch = channels.pop()
            current_res = current_res * 2
            use_attn = current_res in attention_resolutions
            self.ups.append(UpBlock(current_ch, out_ch, time_dim, use_attn))
            current_ch = out_ch
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, current_ch),
            nn.GELU(),
            nn.Conv2d(current_ch, out_channels, 3, padding=1),
        )
        
    def forward(
        self,
        x: torch.Tensor,           # (B, 1, H, W) noisy heatmap
        t: torch.Tensor,           # (B,) timesteps
        cond: torch.Tensor,        # (B, C, H', W') local condition
        global_cond: torch.Tensor = None  # (B, C) global condition (optional)
    ) -> torch.Tensor:
        """预测噪声"""
        
        # Time embedding
        t_emb = self.time_embed(t)
        
        # 如果有 global condition，加到 time embedding
        if global_cond is not None:
            t_emb = t_emb + global_cond
        
        # Upsample condition to match input size
        cond = F.interpolate(cond, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate input with condition
        h = torch.cat([x, cond], dim=1)
        h = self.init_conv(h)
        
        # Downsampling
        skips = []
        for down in self.downs:
            h, skip = down(h, t_emb)
            skips.append(skip)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Upsampling
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, t_emb)
        
        # Final
        return self.final_conv(h)


# ==================== Diffusion 完整模型 ====================

class AffordanceDiffusion(nn.Module):
    """
    完整的 Affordance Heatmap 扩散模型
    
    集成：
    - DINO 特征提取（外部传入，避免重复加载）
    - 条件编码
    - Diffusion U-Net
    - 训练和推理逻辑
    """
    
    def __init__(
        self,
        image_size: int = 224,
        dino_dim: int = 768,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda"
    ):
        super().__init__()
        self.image_size = image_size
        self.timesteps = timesteps
        self.device = device
        
        # 条件编码器
        self.condition_encoder = ConditionEncoder(
            dino_dim=dino_dim,
            out_channels=256,
            image_size=image_size
        )
        
        # Diffusion U-Net
        self.unet = DiffusionUNet(
            in_channels=1,
            out_channels=1,
            cond_channels=256,
            base_channels=64,
            channel_mults=(1, 2, 4, 8),
            image_size=image_size
        )
        
        # 预计算 diffusion schedule
        self._setup_diffusion_schedule(beta_start, beta_end, timesteps)
        
    def _setup_diffusion_schedule(self, beta_start: float, beta_end: float, timesteps: int):
        """预计算 diffusion 参数"""
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为 buffer（不参与训练，但随模型保存/加载）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        
        # 后验分布参数
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        
    def q_sample(
        self, 
        x_0: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向扩散：给干净数据加噪声"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    def training_step(
        self,
        rgb: torch.Tensor,           # (B, 3, H, W)
        depth: torch.Tensor,         # (B, 1, H, W)
        dino_features: torch.Tensor, # (B, D, h, w)
        gt_heatmap: torch.Tensor     # (B, 1, H, W)
    ) -> torch.Tensor:
        """训练步骤：返回 loss"""
        B = rgb.shape[0]
        device = rgb.device
        
        # 随机采样 timestep
        t = torch.randint(0, self.timesteps, (B,), device=device)
        
        # 加噪
        x_t, noise = self.q_sample(gt_heatmap, t)
        
        # 编码条件
        local_cond, global_cond = self.condition_encoder(rgb, depth, dino_features)
        
        # 预测噪声
        noise_pred = self.unet(x_t, t, local_cond, global_cond)
        
        # MSE loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        dino_features: torch.Tensor,
        num_samples: int = 1,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """
        从噪声生成 heatmap
        
        Args:
            rgb, depth, dino_features: 条件输入
            num_samples: 生成多少个样本（用于不确定性估计）
            guidance_scale: classifier-free guidance 强度
            
        Returns:
            heatmaps: (num_samples, B, 1, H, W)
        """
        B = rgb.shape[0]
        device = rgb.device
        
        # 编码条件
        local_cond, global_cond = self.condition_encoder(rgb, depth, dino_features)
        
        all_samples = []
        
        for _ in range(num_samples):
            # 从纯噪声开始
            x = torch.randn(B, 1, self.image_size, self.image_size, device=device)
            
            # DDPM 去噪循环
            for t in reversed(range(self.timesteps)):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                
                # 预测噪声
                noise_pred = self.unet(x, t_batch, local_cond, global_cond)
                
                # 去噪一步
                x = self._denoise_step(x, noise_pred, t)
            
            all_samples.append(x)
        
        return torch.stack(all_samples, dim=0)  # (num_samples, B, 1, H, W)
    
    def _denoise_step(
        self, 
        x_t: torch.Tensor, 
        noise_pred: torch.Tensor, 
        t: int
    ) -> torch.Tensor:
        """单步去噪"""
        betas_t = self.betas[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t]
        
        # 预测 x_0
        pred_x0 = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * noise_pred)
        
        if t > 0:
            noise = torch.randn_like(x_t)
            posterior_variance_t = self.posterior_variance[t]
            x_prev = pred_x0 + torch.sqrt(posterior_variance_t) * noise
        else:
            x_prev = pred_x0
        
        return x_prev


# ==================== DDIM 加速采样器 ====================

class DDIMSampler:
    """
    DDIM 采样器 - 比 DDPM 快 10-50 倍
    """
    
    def __init__(self, model: AffordanceDiffusion, ddim_steps: int = 50, eta: float = 0.0):
        self.model = model
        self.ddim_steps = ddim_steps
        self.eta = eta  # eta=0 是确定性采样
        
        # 计算 DDIM timestep 子集
        self.ddim_timesteps = self._make_ddim_timesteps()
        
    def _make_ddim_timesteps(self):
        """生成 DDIM 使用的 timestep 子集"""
        c = self.model.timesteps // self.ddim_steps
        ddim_timesteps = list(range(0, self.model.timesteps, c))
        return list(reversed(ddim_timesteps))
    
    @torch.no_grad()
    def sample(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        dino_features: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """DDIM 快速采样"""
        B = rgb.shape[0]
        device = rgb.device
        
        # 编码条件
        local_cond, global_cond = self.model.condition_encoder(rgb, depth, dino_features)
        
        all_samples = []
        
        for _ in range(num_samples):
            x = torch.randn(B, 1, self.model.image_size, self.model.image_size, device=device)
            
            for i, t in enumerate(self.ddim_timesteps):
                t_batch = torch.full((B,), t, device=device, dtype=torch.long)
                
                # 预测噪声
                noise_pred = self.model.unet(x, t_batch, local_cond, global_cond)
                
                # DDIM 更新
                x = self._ddim_step(x, noise_pred, t, i)
            
            all_samples.append(x)
        
        return torch.stack(all_samples, dim=0)
    
    def _ddim_step(self, x_t, noise_pred, t, step_idx):
        """DDIM 单步更新"""
        alpha_cumprod_t = self.model.alphas_cumprod[t]
        
        # 预测 x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        
        if step_idx < len(self.ddim_timesteps) - 1:
            t_prev = self.ddim_timesteps[step_idx + 1]
            alpha_cumprod_t_prev = self.model.alphas_cumprod[t_prev]
            
            # 计算方向
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - self.eta ** 2 * (1 - alpha_cumprod_t_prev)) * noise_pred
            
            # 可选的随机性
            if self.eta > 0:
                noise = torch.randn_like(x_t)
                sigma = self.eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t))
            else:
                noise = 0
                sigma = 0
            
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + sigma * noise
        else:
            x_prev = pred_x0
        
        return x_prev