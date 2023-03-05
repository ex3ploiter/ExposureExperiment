
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, FashionMNIST
from MVTecAD import getMVTecDataset
from AdaptiveExposureDataset import getAdaptiveExposureDataset
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from utills import sparse2coarse
from constants import CIFAR10_PATH, CIFAR100_PATH, MNIST_PATH, FMNIST_PATH, SVHN_PATH, MVTEC_PATH, ADAPTIVE_PATH, mvtec_labels
import torchvision.transforms.functional as F
import requests
from PIL import Image
from tqdm import tqdm

dataset_paths = {
    "cifar10":CIFAR10_PATH,
    "cifar100":CIFAR100_PATH,
    "mnist":MNIST_PATH,
    "fashion":FMNIST_PATH,
    "svhn":SVHN_PATH,
    "mvtec":MVTEC_PATH,
    "adaptive":ADAPTIVE_PATH
}

tansform_224 = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ToTensor()
                                ])

tansform_224_gray = transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()
                                ])

tansform_32 = transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.ToTensor()
                                ])

tansform_32_gray = transforms.Compose([
                                    transforms.Resize(32),
                                    transforms.Grayscale(num_output_channels=3),
                                    transforms.ToTensor()
                                ])



class GeneralDataset(torch.utils.data.Dataset):
    def __init__(self, normal_data, exposure_data, transform=None):
        self.transform = transform
        normal_data = [F.to_pil_image(x).convert('RGB') for x in normal_data]
        exposure_data = [F.to_pil_image(x).convert('RGB') for x in exposure_data]
        self.data = normal_data + exposure_data
        self.targets = [0] * len(normal_data) + [1] * len(exposure_data)

    def __getitem__(self, index):
        image = self.data[index]
        target = self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.data)

def get_dataloader(normal_dataset:str, normal_class_indx:int, exposure_dataset:str, batch_size):

    transform = None

    is_big: bool = True if normal_dataset in ['mvtec', 'ctscan'] else False
    is_colorful: bool = True if normal_dataset in ['cifar10', 'cifar100', 'svhn', 'mvtec'] else False

    if is_big:
        if is_colorful:
            transform = tansform_224
        else:
            transform = tansform_224_gray
    else:
        if is_colorful:
            transform = tansform_32
        else:
            transform = tansform_32_gray

    normal_data, testset = get_normal_class(dataset=normal_dataset, normal_class_indx=normal_class_indx, transform=transform)
    exposure_data = get_exposure(dataset=exposure_dataset, normal_dataset=normal_dataset, normal_class_indx=normal_class_indx, count=len(normal_data))

    trainset = GeneralDataset(normal_data=normal_data, exposure_data=exposure_data, transform=transform)
    del exposure_data, normal_data

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader



####################
#  Normal Datastes #
####################

def get_normal_class(dataset='cifar10', normal_class_indx = 0,  transform=None):

    datasets_builders = {
        "cifar10":CIFAR10,
        "cifar100":CIFAR100,
        "mnist":MNIST,
        "fashion":FashionMNIST,
        "svhn":SVHN,
        "mvtec":getMVTecDataset(normal_class_indx, only_anomaly_in_test=False)
    }

    if dataset in datasets_builders.keys():
        trainset = datasets_builders[dataset](root=dataset_paths[dataset], train=True, download=True) #CHECK transform ?
        if dataset == "cifar100":
            trainset.targets = sparse2coarse(trainset.targets)
        trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

        testset = datasets_builders[dataset](root=dataset_paths[dataset], train=False, download=True, transform=transform)
        if dataset == "cifar100":
            testset.targets = sparse2coarse(testset.targets)
        testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

        return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in trainset.data], testset
    else:
        raise Exception("Dataset is not supported yet. ")




######################
#  Exposure Datastes #
######################

def get_exposure(dataset:str='cifar10', normal_dataset:str='cifar100', normal_class_indx:int = 0, count:int = 0):
    assert count > 0

    datasets_builders = {
        "cifar10":CIFAR10,
        "cifar100":CIFAR100,
        "mnist":MNIST,
        "fashion":FashionMNIST,
        "svhn":SVHN,
        "mvtec":getMVTecDataset(
            normal_class_indx = None,
            only_anomaly_in_test = False if normal_dataset!='mvtec' else True,
            remove_class_indx = None if normal_dataset!='mvtec' else mvtec_labels[normal_class_indx]
        ),
        "adaptive":getAdaptiveExposureDataset(normal_dataset, normal_class_indx)
    }

    if dataset in datasets_builders.keys():
        exposure_train = datasets_builders[dataset](root=dataset_paths[dataset], train=True, download=True) #CHECK transforms ?
        #exposure_test = datasets_builders[dataset](root=dataset_paths[dataset], train=False, download=True) #CHECK transforms ?
        if dataset == "cifar100":
            exposure_train.targets = sparse2coarse(exposure_train.targets)
            #exposure_test.targets = sparse2coarse(exposure_test.targets)

        if normal_dataset.lower() == dataset:
            exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]
            #exposure_test.data = exposure_test.data[np.array(exposure_test.targets) != normal_class_indx]

        exposure_data = torch.tensor(exposure_train.data)
        del exposure_train

        #if exposure_data.size(0) < count:
        #    exposure_data = torch.cat((exposure_data, torch.tensor(exposure_test.data)), 0)
        #del exposure_train

        if exposure_data.size(0) < count:
            copy_dataset(exposure_data, count)

        indices = torch.randperm(exposure_data.size(0))[:count]
        exposure_data =  exposure_data[indices]

        return [F.to_tensor(np.array(x).astype(np.uint8)) for x  in exposure_data]
    else:
        raise Exception("Dataset is not supported yet. ")


def copy_dataset(dataset , target_count:int):
    while target_count > len(dataset):
        dataset = torch.cat((dataset, dataset.data), 0)

    return dataset


def get_ADAPTIVE_exposure(normal_dataset:str, normal_class_indx:int,count:int):
    exposure_data = []
    try:
        exposure_path = glob(os.path.join(ADAPTIVE_PATH, normal_dataset, f'{normal_class_indx}', "*.npy"), recursive=True)
        for path in exposure_path:
            exposure_data += np.load(path).tolist()
    except:
        raise ValueError('Wrong Exposure Address!')
        exit()

    exposure_data = torch.tensor(exposure_data)

    if exposure_data.size(0) < count:
        copy_dataset(exposure_data, count)

    indices = torch.randperm(exposure_data.size(0))[:count]
    exposure_data =  exposure_data[indices]

    return exposure_data