import shutil
import os
import torch
import torchvision
from tqdm import tqdm

# Reuse your cosine schedule from training
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(max=0.999)

@torch.no_grad()
def sample(model, img_size=32, channels=3, num_steps=1000, sample_steps=None, batch_size=16, save_dir="samples"):
    device = next(model.parameters()).device
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    betas = cosine_beta_schedule(num_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])
    posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    posterior_variance = torch.clamp(posterior_variance, min=1e-20)

    if sample_steps is None:
        sample_steps = num_steps
        timestep_indices = torch.arange(num_steps - 1, -1, -1, device=device)
    else:
        timestep_indices = torch.linspace(num_steps - 1, 0, sample_steps, device=device).long()

    x_t = torch.randn(batch_size, channels, img_size, img_size, device=device)

    for i, t in enumerate(tqdm(timestep_indices, desc="Sampling")):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        predicted_noise = model(x_t, t_batch)

        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        alpha_cumprod_prev_t = alphas_cumprod_prev[t]
        beta_t = betas[t]

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_alpha_prev = torch.sqrt(alpha_cumprod_prev_t)

        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

        mean = (beta_t * sqrt_alpha_prev) / (1 - alpha_cumprod_t) * x_0_pred + \
               ((1 - alpha_cumprod_prev_t) * sqrt_alpha_t) / (1 - alpha_cumprod_t) * x_t

        if t > 0:
            noise = torch.randn_like(x_t)
            var = posterior_variance[t]
            x_t = mean + torch.sqrt(var) * noise
        else:
            x_t = mean

        # Debugging logs near the end
        if t < 5 or t > num_steps - 5:
            print(f"Step {t}: var={var.item() if t > 0 else 0}, mean norm={mean.norm().item()}")

        if i % 100 == 0 or i == len(timestep_indices) - 1:
            img_grid = torchvision.utils.make_grid((x_t.clamp(-1, 1) + 1) / 2, nrow=4)
            torchvision.utils.save_image(img_grid, f"{save_dir}/step_{t.item():04d}.png")

    print(f"Sampling complete. Images saved to {save_dir}")
    return x_t


if __name__ == "__main__":
    from model import SimpleUNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("checkpoint.pth", map_location=device))

    # delete the sample directory if it exists
    if os.path.exists("samples"):
        shutil.rmtree("samples")

    # Sample with full 1000 steps (slow but best quality)
    sample(model, img_size=32, batch_size=16, num_steps=1000, save_dir="samples")

    # Or sample faster with 200 steps (less quality but faster)
    # sample(model, img_size=32, batch_size=16, num_steps=1000, sample_steps=200, save_dir="samples_fast")
