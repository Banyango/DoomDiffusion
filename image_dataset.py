import random
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from PIL import Image
import os

class DoomImages(Dataset):
    def __init__(self, folder: str, image_size: int = 64, transform=None):
        self.folder = folder
        self.image_size = image_size
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img
