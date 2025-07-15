import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# sinusoidal embedding for timestep
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# a single conv block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act = nn.ReLU()

        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, out_ch)

    def forward(self, x, t_emb=None):
        h = self.conv1(x)
        h = self.norm1(h)

        if t_emb is not None:
            t_h = self.time_mlp(t_emb)
            h = h + t_h[:, :, None, None]

        return self.act(h)


# U-Net for 64×64 images
class SimpleUNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=32, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Encoder blocks
        self.enc1 = ConvBlock(img_channels, base_channels, time_emb_dim)  # 64x64
        self.enc2 = ConvBlock(base_channels, base_channels * 2, time_emb_dim)  # 32x32
        self.enc3 = ConvBlock(
            base_channels * 2, base_channels * 4, time_emb_dim
        )  # 16x16

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(
            base_channels * 4, base_channels * 8, time_emb_dim
        )  # 8x8

        # Decoder blocks
        self.dec3 = ConvBlock(
            base_channels * 8 + base_channels * 4, base_channels * 4, time_emb_dim
        )  # 16x16
        self.dec2 = ConvBlock(
            base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim
        )  # 32x32
        self.dec1 = ConvBlock(
            base_channels * 2 + base_channels, base_channels, time_emb_dim
        )  # 64x64

        self.final = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t):
        # embed t
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)  # (B, C, 64, 64)
        e2 = self.enc2(self.pool(e1), t_emb)  # (B, 2C, 32, 32)
        e3 = self.enc3(self.pool(e2), t_emb)  # (B, 4C, 16, 16)

        # Bottleneck
        b = self.bottleneck(self.pool(e3), t_emb)  # (B, 8C, 8, 8)

        # Decoder
        d3 = F.interpolate(b, scale_factor=2, mode="nearest")  # (B, 8C, 16, 16)
        d3 = torch.cat([d3, e3], dim=1)  # (B, 8C+4C, 16, 16)
        d3 = self.dec3(d3, t_emb)  # (B, 4C, 16, 16)

        d2 = F.interpolate(d3, scale_factor=2, mode="nearest")  # (B, 4C, 32, 32)
        d2 = torch.cat([d2, e2], dim=1)  # (B, 4C+2C, 32, 32)
        d2 = self.dec2(d2, t_emb)  # (B, 2C, 32, 32)

        d1 = F.interpolate(d2, scale_factor=2, mode="nearest")  # (B, 2C, 64, 64)
        d1 = torch.cat([d1, e1], dim=1)  # (B, 2C+C, 64, 64)
        d1 = self.dec1(d1, t_emb)  # (B, C, 64, 64)

        out = self.final(d1)  # (B, img_channels, 64, 64)
        return out
