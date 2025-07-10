import numpy as np
import torch
from torch.utils.data import Dataset

from PIL import Image
import os

class DoomImages(Dataset):
    def __init__(self, folder: str, image_size: int = 64):
        self.folder = folder
        self.image_size = image_size
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        img = (torch.tensore(np.array(img)) / 127.5) - 1.0  # Normalize to [-1, 1]
        img = img.permute(2, 0, 1).float()
        return img
