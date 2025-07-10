import torch
import torch.nn as nn
import math

# -----------------------------------------------------------
# Noise schedule (precompute)
# -----------------------------------------------------------
T = 1000  # number of diffusion steps
betas = torch.linspace(1e-4, 0.02, T)  # linear schedule
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# -----------------------------------------------------------
# Sinusoidal timestep embedding
# -----------------------------------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# -----------------------------------------------------------
# U-Net Block
# -----------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, down=True):
        super().__init__()
        self.down = down
        if down:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb):
        h = self.relu(self.conv1(x))
        t_emb_proj = self.time_mlp(t_emb).view(-1, h.shape[1], 1, 1)
        h = h + t_emb_proj
        h = self.relu(self.conv2(h))
        return h

# -----------------------------------------------------------
# Simple U-Net
# -----------------------------------------------------------
class SimpleUNet(nn.Module):
    def __init__(self, image_channels=3, base_channels=64, time_emb_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Down-sampling
        self.conv0 = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim, down=True)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim, down=True)

        # Bottleneck
        self.bot1 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1)
        self.bot2 = nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1)

        # Up-sampling
        self.up1 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim, down=False)
        self.up2 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim, down=False)

        # Output layer
        self.out = nn.Conv2d(base_channels, image_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Down
        x0 = self.conv0(x)
        x1 = self.down1(x0, t_emb)
        x2 = self.down2(x1, t_emb)

        # Bottleneck
        b = nn.ReLU()(self.bot1(x2))
        b = nn.ReLU()(self.bot2(b))

        # Up
        u1 = self.up1(b, t_emb)
        u2 = self.up2(u1, t_emb)
        out = self.out(u2)
        return out
