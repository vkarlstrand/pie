# Imports
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2



def default_loader(path):
    return Image.open(path).convert('RGB')

class DatasetFromCSV(Dataset):
    def __init__(self, image_root, csv_path, transforms=None, loader=default_loader):

        self.image_root = image_root

        self.data = pd.read_csv(csv_path, header=None)
        self.labels = np.asarray(self.data.iloc[:, 1])

        imgs = []
        files_names = np.array(self.data.iloc[:, 0])
        for img in files_names:
            imgs.append(os.path.join(self.image_root, str(img)))

        self.images = imgs
        self.transforms = transforms
        self.loader = loader

    def __getitem__(self, index):

        label = self.labels[index]
        img = self.images[index]
        img = np.array(self.loader(img))
        if self.transforms is not None:
            img = self.transforms(image=img)
        return img, label

    def __len__(self):
        return len(self.data.index)
