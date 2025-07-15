import click
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from image_dataset import DoomImages
from libs.config import TrainingConfig, Paths, ModelConfig
from libs.models.simple_unet import SimpleUNet
from libs.schedules.cosine_beta import CosineBeta
from libs.trainer import Trainer


@click.command()
@click.option(
    "--debug-loss", is_flag=True, help="Enable debug mode for loss calculations"
)
@click.option(
    "--reload", is_flag=True, help="Restart training from the cached best model"
)
def main(debug_loss: bool, reload: bool):
    """Main function to train the model."""

    config = TrainingConfig()
    model_config = ModelConfig()

    # cli options
    config.debug_loss = debug_loss
    config.reload = reload

    config.paths = Paths()

    config.model = SimpleUNet(base_channels=model_config.base_channels).to(
        config.device
    )
    config.optimizer = Adam(
        config.model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=config.image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    config.dataset = DoomImages(
        folder=config.paths.data,
        image_size=config.image_size,
        transform=train_transform,
    )
    config.data_loader = DataLoader(
        config.dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )

    config.loss_fn = nn.MSELoss()
    config.schedule = CosineBeta()

    trainer = Trainer(config, model_config)

    trainer.train()
