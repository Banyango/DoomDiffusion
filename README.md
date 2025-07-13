# Doom Image Diffusion Project

This project implements a diffusion model for generating and processing images, inspired by the game Doom. It includes a dataset class for handling image data, a U-Net-based neural network for image generation, and a noise schedule for diffusion.

## Features

- **Dataset Handling**: The `DoomImages` class processes image datasets, ensuring proper resizing, normalization, and loading.
- **Diffusion Model**: Implements a U-Net architecture with sinusoidal timestep embeddings for image generation.
- **Noise Schedule**: Precomputes a linear noise schedule for diffusion steps.

## Project Structure

- `image_dataset.py`: Contains the `DoomImages` dataset class for loading and preprocessing images.
- `model.py`: Defines the U-Net model, sinusoidal timestep embeddings, and noise schedule.
- `.gitignore`: Specifies files and directories to exclude from version control.
- `README.md`: Documentation for the project.

## Requirements

- Python 3.12+
- PyTorch
- NumPy
- Pillow

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

## Usage

### Dataset Preparation

Place your `.png` images in a folder (e.g., `data/`). The `DoomImages` class will automatically load and preprocess them.

### Training the Model

1. Import the dataset and model:
   ```python
   from image_dataset import DoomImages
   from model import SimpleUNet
   ```

2. Initialize the dataset and model:
   ```python
   dataset = DoomImages(folder='data/', image_size=64)
   model = SimpleUNet()
   ```

3. Train the model using your preferred training loop.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.