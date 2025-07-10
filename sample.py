import os

import torch
import torchvision

@torch.no_grad()
def sample(model, img_size=64, channels=3, num_steps=1000, batch_size=16, save_dir="samples"):
    device = next(model.parameters()).device
    model.eval()

    # Noise schedule (same as training)
    betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])

    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # Start from pure noise
    x_t = torch.randn(batch_size, channels, img_size, img_size, device=device)

    os.makedirs(save_dir, exist_ok=True)

    for t in reversed(range(num_steps)):
        t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
        predicted_noise = model(x_t, t_batch)

        alpha_t = alphas[t]
        alpha_cumprod_t = alphas_cumprod[t]
        sqrt_recip_alpha_t = sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t]

        # Predict x_0
        x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)

        # Compute mean of posterior q(x_{t-1} | x_t, x_0)
        mean = sqrt_recip_alpha_t * (x_t - (betas[t] / sqrt_one_minus_alpha_cumprod_t) * predicted_noise)

        if t > 0:
            noise = torch.randn_like(x_t)
            var = posterior_variance[t]
            x_t = mean + torch.sqrt(var) * noise
        else:
            x_t = mean

        if t % 100 == 0 or t == num_steps - 1:
            grid = torchvision.utils.make_grid((x_t.clamp(-1, 1) + 1) / 2, nrow=4)
            torchvision.utils.save_image(grid, f"{save_dir}/step_{t:04d}.png")

    print(f"Samples saved in {save_dir}")
    return x_t


if __name__ == "__main__":
    model.load_state_dict(torch.load("checkpoint.pth"))
    sample(model, img_size=64, batch_size=16)