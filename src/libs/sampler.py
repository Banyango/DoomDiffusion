import os

import torch
import torchvision
from torch import nn

from tqdm import tqdm

from libs.config import Paths, ModelConfig
from libs.schedules.schedule import Schedule


class SamplerConfig:
    paths: Paths()
    """Paths configuration for saving checkpoints, logs, and results."""

    model: nn.Module
    """The model to be trained, typically a neural network architecture."""

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Device on which the model will be trained, either CPU or GPU."""

    schedule: Schedule
    """Schedule for generating beta values for the diffusion process."""

    num_steps: int = 1000
    """Number of steps for the sampling process."""

    batch_size: int = 128
    """Batch size for sampling the model."""

    img_size: int = 64
    """Size of the images used in the dataset."""

    base_channels: int = 64
    """Number of channels in the images."""

    channels: int = 3
    """Number of channels in the input images, e.g., 3 for RGB images."""

    debug_step_values: bool = False
    """Flag to enable or disable debugging of step values during sampling."""

    debug_samples: bool = False
    """Flag to enable or disable saving debug samples during sampling."""


class Sampler:
    def __init__(self, config: SamplerConfig):
        self.config = config

    def sample(self):
        """Sample images from the model using the diffusion process."""
        self.config.model.eval()

        os.makedirs(self.config.paths.samples, exist_ok=True)

        betas = self.config.schedule.beta_schedule(self.config.num_steps).to(
            self.config.device
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.0], device=self.config.device), alphas_cumprod[:-1])
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_variance = posterior_variance.clamp(min=1e-20)

        x_t = torch.randn(
            self.config.batch_size,
            self.config.channels,
            self.config.img_size,
            self.config.img_size,
            device=self.config.device,
        )

        for t in tqdm(reversed(range(self.config.num_steps)), desc="Sampling"):
            t_batch = torch.full((self.config.batch_size,), t, device=self.config.device, dtype=torch.long)
            predicted_noise = self.config.model(x_t, t_batch)

            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]
            alpha_cumprod_prev_t = alphas_cumprod_prev[t]
            beta_t = betas[t]

            # predict x_0
            eps = 1e-5  # prevent division by zero
            alpha_cumprod_t_safe = torch.clamp(alpha_cumprod_t, min=eps)

            x_0_pred = (
                               x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise
                       ) / torch.sqrt(alpha_cumprod_t_safe)
            x_0_pred = x_0_pred.clamp(-1, 1)  # optional

            # posterior mean calculation
            coef1 = beta_t * torch.sqrt(alpha_cumprod_prev_t) / (1 - alpha_cumprod_t)
            coef2 = (1 - alpha_cumprod_prev_t) * torch.sqrt(alpha_t) / (1 - alpha_cumprod_t)
            mean = coef1 * x_0_pred + coef2 * x_t

            if t > 0:
                noise = torch.randn_like(x_t)
                var = posterior_variance[t]
                x_t = mean + torch.sqrt(var) * noise
            else:
                x_t = mean

            if self.config.debug_step_values:
                if t in [0, 10, 100, 500, 900]:
                    print(f"\nStep {t}:")
                    print(f"  beta_t: {beta_t.item():.6f}")
                    print(f"  posterior_variance[t]: {posterior_variance[t].item():.6e}")
                    print(f"  coef1: {coef1.item():.6f}, coef2: {coef2.item():.6f}")
                    print(
                        f"  x_0_pred min/max/mean/std: {x_0_pred.min().item():.3f} / {x_0_pred.max().item():.3f} / {x_0_pred.mean().item():.3f} / {x_0_pred.std().item():.3f}"
                    )
                    print(
                        f"  mean min/max/mean/std: {mean.min().item():.3f} / {mean.max().item():.3f} / {mean.mean().item():.3f} / {mean.std().item():.3f}"
                    )
                    print(
                        f"  x_t min/max/mean/std: {x_t.min().item():.3f} / {x_t.max().item():.3f} / {x_t.mean().item():.3f} / {x_t.std().item():.3f}"
                    )

            if t < 5 or t > self.config.num_steps - 5:
                print(
                    f"Step {t}: var={var.item() if t > 0 else 0}, mean norm={mean.norm().item()}"
                )

            if (self.config.debug_samples and t % 100 == 0) or t == 0:
                img_grid = torchvision.utils.make_grid((x_t.clamp(-1, 1) + 1) / 2, nrow=4)
                torchvision.utils.save_image(img_grid, f"{self.config.paths.samples}/step_{t:04d}.png")

        print(f"Sampling complete. Images saved to {self.config.paths.samples}")
