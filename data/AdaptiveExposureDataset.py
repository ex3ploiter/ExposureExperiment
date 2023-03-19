import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from data.constants import ADAPTIVE_PATH
import torchvision.transforms.functional as F
import requests
from PIL import Image
from tqdm import tqdm
import cv2 

def getAdaptiveExposureDataset(normal_dataset, normal_class_indx):
    class AdaptiveExposureDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None, target_transform=None, train=True, normal=True, download=False):
            self.transform = transform
            self.data = []
            try:
                # file_paths = glob(os.path.join(ADAPTIVE_PATH, normal_dataset, f'{normal_class_indx}', "*.npy"), recursive=True)
                file_paths = glob(os.path.join(ADAPTIVE_PATH,(f'{normal_dataset}_GLIDE_*{normal_class_indx}_*.npy')), recursive=True)
                
                for path in file_paths:
                    # self.data += np.load(path).tolist()
                    
                    loaded=np.load(path).transpose(0,2,3,1)
                    norm_image = cv2.normalize(loaded, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                    norm_image = norm_image.astype(np.uint8)

                    self.data += norm_image.tolist()
                                        
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
    
    return AdaptiveExposureDataset
