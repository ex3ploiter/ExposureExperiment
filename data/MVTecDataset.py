import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from data.constants import MVTEC_PATH, mvtec_labels
import torchvision.transforms.functional as F
import requests
from PIL import Image
from tqdm import tqdm

def getMVTecDataset(normal_class_indx, only_anomaly_in_test=False, remove_class_indx=None):

    class MVTecDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None, target_transform=None, train=True, normal=True, download=False):
            self.transform = transform
            self.category = "*" if normal_class_indx is None else mvtec_labels[normal_class_indx]

            # Check if dataset directory exists
            dataset_dir = os.path.join(root, "mvtec_anomaly_detection")
            if not os.path.exists(dataset_dir):
                if download:
                    self.download_dataset(root)
                else:
                    raise ValueError("Dataset not found. Please set download=True to download the dataset.")

            if train:
                self.data = glob(
                    os.path.join(dataset_dir, self.category, "train", "good", "*.png"), recursive=True
                )

            else:
                image_files = glob(os.path.join(dataset_dir, self.category, "test", "*", "*.png"), recursive=True)
                normal_image_files = glob(os.path.join(dataset_dir, self.category, "test", "good", "*.png"), recursive=True)
                anomaly_image_files = list(set(image_files) - set(normal_image_files))
                if only_anomaly_in_test:
                    self.data = anomaly_image_files
                else:
                    self.data = image_files

            if remove_class_indx is not None:
                remove_class_files = glob(
                    os.path.join(dataset_dir, self.category, "**", "*.png"), recursive=True
                )
                self.data = list(set(self.data) - set(remove_class_files))

            self.data.sort(key=lambda y: y.lower())
            self.data = [Image.open(x).convert('RGB') for x in self.data]
            self.train = train

        def __getitem__(self, index):
            image_file = self.data[index]
            image = image_file
            if self.transform is not None:
                image = self.transform(image_file)

            if os.path.dirname(image_file).endswith("good"):
                target = 0
            else:
                target = 1

            return image, target

        def __len__(self):
            return len(self.data)

        def download_dataset(self, root):
            url = "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz"
            dataset_dir = os.path.join(root, "mvtec_anomaly_detection")

            # Create directory for dataset
            os.makedirs(dataset_dir, exist_ok=True)

            # Download and extract dataset
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(os.path.join(root, "mvtec_anomaly_detection.tar.xz"), 'wb') as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            progress_bar.close()

            os.system(f"tar -xf {os.path.join(root, 'mvtec_anomaly_detection.tar.xz')} -C {dataset_dir}")
    return MVTecDataset
