"""CBAM 注意力模块（Convolutional Block Attention Module）

论文：Woo S, Park J, Lee J Y, et al. CBAM: Convolutional Block Attention Module.
      ECCV 2018.

结构与原论文一致，由两个串联的子模块组成：

    输入 F
      │
      ├─ Channel Attention ──► 按通道加权（"看什么"）
      │        F' = Mc(F) ⊗ F
      │
      └─ Spatial Attention ──► 按空间位置加权（"看哪里"）
               F'' = Ms(F') ⊗ F'

通道注意力同时对全局平均池化和全局最大池化做 MLP，两路结果相加后过 Sigmoid；
空间注意力把通道维的均值图和最大值图拼成 2 通道，再用一个 7×7 卷积压成 1 通道。

用于 PF-MDD 的动机
------------------
薄膜表面缺陷（晶点、划痕、气泡）在 4096×256 的宽幅图中占比很小，
经过多次下采样后特征响应弱。CBAM 的通道注意力能突出缺陷相关通道，
空间注意力能把响应集中到缺陷所在的小区域，从而提升细小目标的召回。

本模块不依赖 ultralytics 内部实现，可单独用于任何 PyTorch 模型。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力：avg-pool 与 max-pool 两路共享 MLP，相加后 Sigmoid。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 两路共享同一个 MLP（原论文做法），这里用一个 Sequential 复用
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力：通道维 avg/max 拼接成 2 通道，7×7 卷积后 Sigmoid。"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size 必须是奇数，才能保持特征图尺寸不变")
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """CBAM = 通道注意力 → 空间注意力，输出尺寸与输入一致。"""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        return self.spatial_attention(x)

    def extra_repr(self) -> str:
        return f"channels={self.channel_attention.mlp[0].in_channels}"
