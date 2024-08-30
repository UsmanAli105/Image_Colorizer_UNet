from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import os
import numpy as np
import torch

def convert_to_grayscale(image_data):
    R, G, B = image_data[0, :, :], image_data[1, :, :], image_data[2, :, :]
    grayscale_image = 0.2989 * R + 0.5870 * G + 0.1140 * B
    grayscale_image = np.clip(grayscale_image, 0, 255)
    grayscale_image = grayscale_image.unsqueeze(dim=0)
    return grayscale_image




class CustomDataset(Dataset):
    def __init__(self, image_files, images_path, image_transform=None):
        self.images_path = images_path
        self.image_transform = image_transform
        self.image_files = image_files

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_data = read_image(os.path.join(self.images_path, image_file)).float()
        if self.image_transform is not None:
            image_data = self.image_transform(image_data)
        gray_image = convert_to_grayscale(image_data)
        return gray_image, image_data