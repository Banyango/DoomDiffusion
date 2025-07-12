import math
import os
from collections import deque

import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from image_dataset import DoomImages, RandomGaussianBlur
from model import SimpleUNet

BATCH_SIZE = 16
IMAGE_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopper:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else :
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule for betas, as proposed in Improved DDPM paper.

    Args:
        timesteps (int): total diffusion steps, e.g., 1000
        s (float): small offset to prevent singularities (default 0.008)

    Returns:
        betas (torch.Tensor): noise schedule of shape [timesteps]
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to 1 at t=0
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = betas.clamp(max=0.999)
    return betas

def main():
    model = SimpleUNet(base_channels=32).to(device)

    print(f"using device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    mse = nn.MSELoss()

    print("Loading Doom images dataset...")
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),
        transforms.RandomRotation(degrees=15),
        RandomGaussianBlur(p=0.3),  # added
        transforms.ToTensor(),
        transforms.RandomErasing(  # added
            p=0.25,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
            value='random'
        ),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    dataset = DoomImages(folder="data/", image_size=IMAGE_SIZE, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    print(f"Dataset size: {len(dataset)}")

    early_stopper = EarlyStopper(patience=50)
    training_losses = []

    save_dir = "checkpoints"
    sample_dir = "samples"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Diffusion parameters
    T = 1000  # total diffusion steps
    betas = cosine_beta_schedule(T).to(device)
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

    epochs_total = 1000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_total)
    for epoch in range(epochs_total):
        running_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs_total}", unit="batch")

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
            optimizer.step()
            scheduler.step()

            running_loss = running_loss + (loss.item() * batch_size)
            pbar.set_postfix({"batch_loss": loss.item()})

        epoch_loss = running_loss / len(dataset)
        training_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs_total}, Loss: {epoch_loss:.4f}")

        # if epoch_loss <= early_stopper.best_loss:
        #     torch.save(model.state_dict(), os.path.join(save_dir, "best.pth"))
        #     print(f"✅ Best model saved at epoch {epoch + 1}")

        # # Check early stopping
        # if early_stopper.step(epoch_loss):
        #     print(f"⏹️ Early stopping triggered at epoch {epoch + 1}")
        #     break

    torch.save(model.state_dict(), "checkpoint.pth")

    # Plot loss curve
    plt.plot(training_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))


if __name__ == "__main__":
    main()
