import torch
import math
from torch import Tensor

from libs.schedules.schedule import Schedule


class CosineBeta(Schedule):
    def __init__(self, s: float = 0.008):
        """
        Initialize the CosineBeta schedule with a small offset.

        Args:
            s (float): Small offset to prevent singularities (default 0.008).
        """
        super().__init__()
        self.s = s

    def beta_schedule(self, timesteps: int) -> Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + self.s) / (1 + self.s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to 1 at t=0
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(max=0.999)
        return betas
