import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from image_dataset import DoomImages
from model import SimpleUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model = SimpleUNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    mse = nn.MSELoss()

    print("Loading Doom images dataset...")
    dataset = DoomImages(folder="data/doom_images", image_size=64)
    dataloader = DataLoader(dataset, batch_size=129, shuffle=True, num_workers=4)

    print(f"Dataset size: {len(dataset)}")

    T = 1000
    betas = torch.linspace(1e-4, 0.02, T).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    def q_sample(images, t, noise):
        """
        Forward diffusion (adding noise to the clean image at timestep t)
        """
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod * images + sqrt_one_minus_alphas_cumprod * noise


    # Training loop

    epochs = 100
    for epochs in range(epochs):
        pbar = tqdm(dataloader)
        for clean_images in pbar:
            clean_images = clean_images.to(device)
            batch_size = clean_images.shape[0]

            timestep = torch.randint(0, T, (batch_size,), device=device).long()
            noise = torch.randn_like(clean_images)
            noisy_images = q_sample(clean_images, timestep, noise)

            pred_noise = model(noisy_images, timestep)

            loss = mse(noise, pred_noise)

            optimizer.zero_grad()
            loss.backward()

            pbar.set_description(f"Epoch {epochs + 1}/{epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "checkpoint.pth")


if __name__ == "__main__":
    main()
