import random
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from PIL import Image
import os

class RandomGaussianBlur:
    """Apply Gaussian Blur with random sigma."""
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.uniform(self.radius_min, self.radius_max)
            return F.gaussian_blur(img, kernel_size=5, sigma=sigma)
        return img

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
