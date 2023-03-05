import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from constants import ADAPTIVE_PATH
import torchvision.transforms.functional as F
import requests
from PIL import Image
from tqdm import tqdm

def getAdaptiveExposureDataset(normal_dataset, normal_class_indx):
    class AdaptiveExposureDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None, target_transform=None, train=True, normal=True, download=False):
            self.transform = transform
            self.data = []
            try:
                file_paths = glob(os.path.join(ADAPTIVE_PATH, normal_dataset, f'{normal_class_indx}', "*.npy"), recursive=True)
                for path in file_paths:
                    self.data += np.load(path).tolist()
            except:
                raise ValueError('Wrong Exposure Address!')
            self.train = train

        def __getitem__(self, index):
            image_file = self.data[index]
            image = image_file
            if self.transform is not None:
                image = self.transform(image_file)

            target = 0

            return image, target

        def __len__(self):
            return len(self.data)
