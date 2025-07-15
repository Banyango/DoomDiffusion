import os
import torch
import torchvision
from tqdm import tqdm

DEBUG = False
LOAD_CHECKPOINT = True


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)


@torch.no_grad()
def sample(
    model, img_size=64, channels=3, num_steps=1000, batch_size=16, save_dir="samples"
):
    device = next(model.parameters()).device
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    betas = cosine_beta_schedule(num_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device), alphas_cumprod[:-1]]
    )
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    posterior_variance = posterior_variance.clamp(min=1e-20)

    x_t = torch.randn(batch_size, channels, img_size, img_size, device=device)

    for t in tqdm(reversed(range(num_steps)), desc="Sampling"):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        predicted_noise = model(x_t, t_batch)

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

        if DEBUG:
            debug_step_values(
                beta_t, coef1, coef2, mean, posterior_variance, t, x_0_pred, x_t
            )

        if t < 5 or t > num_steps - 5:
            print(
                f"Step {t}: var={var.item() if t > 0 else 0}, mean norm={mean.norm().item()}"
            )

        if (DEBUG and t % 100 == 0) or t == 0:
            img_grid = torchvision.utils.make_grid((x_t.clamp(-1, 1) + 1) / 2, nrow=4)
            torchvision.utils.save_image(img_grid, f"{save_dir}/step_{t:04d}.png")

    print(f"Sampling complete. Images saved to {save_dir}")
    return x_t


def debug_step_values(beta_t, coef1, coef2, mean, posterior_variance, t, x_0_pred, x_t):
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


if __name__ == "__main__":
    from src.libs.models.simple_unet import SimpleUNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Remove the sample directory if it exists even if not empty
    if os.path.exists("samples"):
        for root, dirs, files in os.walk("samples", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir("samples")

    model = SimpleUNet(base_channels=64).to(device)

    if LOAD_CHECKPOINT:
        checkpoint = torch.load("checkpoints/best.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(torch.load("results/v1.pth", map_location=device))

    sample(model, img_size=64, batch_size=16, num_steps=1000, save_dir="samples")
