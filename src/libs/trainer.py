import os
from typing import Callable

import torch
from torch import Tensor
from loguru import logger

from libs.config import TrainingConfig, ModelConfig


class EarlyStopper:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        """
        Check if the validation loss has improved and update the early stopping counter.

        Args:
            val_loss (float): The current validation loss.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


class Trainer:
    def __init__(
        self,
        training_config: TrainingConfig,
        model_config: ModelConfig,
    ):
        self.model = training_config.model
        self.optimizer = training_config.optimizer
        self.loss_fn = training_config.loss_fn
        self.dataset = training_config.dataset
        self.data_loader = training_config.data_loader
        self.save_dir = "../../checkpoints"
        self.sample_dir = "../../samples"
        self.training_config = training_config
        self.model_config = model_config

    def train(self):
        """
        Train the model for a given number of epochs.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        betas = self.training_config.schedule.beta_schedule(
            self.training_config.total_diffusion_steps
        ).to(self.training_config.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        def q_sample(images, t, noise):
            """
            Forward diffusion (adding noise to the clean image at timestep t)
            """
            sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod[t])[
                :, None, None, None
            ]
            return sqrt_alphas_cumprod * images + sqrt_one_minus_alphas_cumprod * noise

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.training_config.optimizer,
            T_max=self.training_config.total_diffusion_steps,
        )
        start_epoch = 0
        if self.training_config.reload:
            checkpoint = torch.load(os.path.join(self.save_dir, "best.pth"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1

        early_stopper = EarlyStopper(patience=self.training_config.patience)
        training_losses = []

        for epoch in range(start_epoch, self.training_config.total_diffusion_steps):
            running_loss = 0.0
            scheduler.step()
            pbar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{epochs_total}", unit="batch"
            )

            for clean_images in pbar:
                clean_images = clean_images.to(self.training_config.device)
                batch_size = clean_images.shape[0]

                timestep = torch.randint(
                    0, T, (batch_size,), device=self.training_config.device
                ).long()
                noise = torch.randn_like(clean_images)
                noisy_images = q_sample(clean_images, timestep, noise)

                pred_noise = self.training_config.model(noisy_images, timestep)

                loss = self.training_config.loss_fn(noise, pred_noise)

                self.training_config.optimizer.zero_grad()
                loss.backward()
                self.training_config.optimizer.step()

                running_loss = running_loss + (loss.item() * batch_size)
                pbar.set_postfix({"batch_loss": loss.item()})

            epoch_loss = running_loss / len(self.training_config.dataset)
            training_losses.append(epoch_loss)
            logger.info(
                f"Epoch {epoch + 1}/{self.training_config.total_diffusion_steps}, Loss: {epoch_loss:.4f}"
            )

            if epoch_loss <= early_stopper.best_loss:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.training_config.model.state_dict(),
                        "optimizer_state_dict": self.training_config.optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),  # if using a scheduler
                        "loss": epoch_loss,
                    },
                    os.path.join(self.training_config.save_dir, "best.pth"),
                )
                logger.info(f"✅ Best model saved at epoch {epoch + 1}")

            if early_stopper.step(epoch_loss):
                logger.info(
                    f"Early stopping at epoch {epoch + 1} with best loss {early_stopper.best_loss:.4f}"
                )
                break

            # plot intermediate samples
            if self.training_config.should_debug_loss:
                plt.figure(figsize=(10, 5))
                plt.plot(training_losses)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Training Loss Curve")
                plt.show()

        torch.save(
            self.training_config.model.state_dict(),
            self.training_config.final_output_path,
        )

        if self.training_config.should_debug_loss:
            plt.plot(training_losses)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.savefig(os.path.join(save_dir, "loss_curve.png"))
