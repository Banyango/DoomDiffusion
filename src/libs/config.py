import os.path

import torch
from torch import nn
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from typing import Union, Sized

from libs.schedules.schedule import Schedule


class Paths:
    checkpoints: str = "checkpoints"
    """Directory where model checkpoints are saved."""

    logs: str = "logs"
    """Directory where training logs are stored."""

    samples: str = "samples"
    """Directory where sample outputs are saved."""

    results: str = "results"
    """Directory where final results and outputs are saved."""

    data: str = "data"
    """Directory where the dataset is stored."""

    final_training_model_output: str = os.path.join("final", "result.pth")
    """Path to save the final output of the training process."""


class TrainingConfig:
    reload: bool = False
    """Flag to reload the model from a checkpoint."""

    paths: Paths
    """Paths configuration for saving checkpoints, logs, and results."""

    model: nn.Module
    """The model to be trained, typically a neural network architecture."""

    optimizer: Optimizer
    """Optimizer used for training the model, e.g., Adam or SGD."""

    loss_fn: Module
    """Loss function used to compute the error during training, e.g., MSELoss or CrossEntropyLoss."""

    dataset: Union[Dataset, Sized]
    """Dataset containing the training data, e.g., images or text."""

    schedule: Schedule
    """Schedule for generating beta values for the diffusion process."""

    data_loader: DataLoader
    """DataLoader for batching and shuffling the dataset during training."""

    total_diffusion_steps = 1000
    """Total number of diffusion steps in the training process."""

    batch_size: int = 128
    """Batch size for training the model."""

    image_size: int = 64  # size of the images
    """Size of the images used in the dataset."""

    learning_rate: float = 1e-4
    """Learning rate for the optimizer."""

    weight_decay: float = 1e-4
    """Weight decay for the optimizer, used for regularization."""

    patience: int = 10
    """Number of epochs with no improvement after which training will be stopped."""

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """Device on which the model will be trained, either CPU or GPU."""

    should_debug_loss: bool = True
    """Flag to enable or disable debugging of loss values during training."""


class ModelConfig:
    img_channels: int = 3
    """Number of channels in the input images, e.g., 3 for RGB images."""

    base_channels: int = 32
    """Base number of channels for the model, used to define the architecture."""
