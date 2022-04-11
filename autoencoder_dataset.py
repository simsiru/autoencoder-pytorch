import torch
from torch.utils.data import Dataset
import torchvision.transforms as tfs
from torchvision.io import read_image
import os
import pandas as pd


class AutoencoderDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        #self.img_labels = annotations_file
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        _, _, files = next(os.walk(img_dir))
        self.n_samples = len(files)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        #img_path = os.path.join(self.img_dir, "/" + str(index) + ".jpg")
        img_path = self.img_dir + "/" + str(index) + ".jpg"
        image = read_image(img_path)
        
        if self.transform:
            image = self.transform(image.numpy())
        image = image / 255.0
        return image