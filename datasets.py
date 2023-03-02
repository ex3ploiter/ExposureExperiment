
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob
from constants import CIFAR10_PATH, CIFAR100_PATH, MNIST_PATH, FMNIST_PATH, SVHN_PATH, MVTEC_PATH

class MyDataset_Binary(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x, labels,transform):
        'Initialization'
        super(MyDataset_Binary, self).__init__()
        self.labels = labels
        self.x = x
        self.transform = transform

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        x = self.transform(self.x[index])
        y = self.labels[index]
       
        return x, y
  
tansform_224 = transforms.Compose([
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()
                                      ])

tansform_224_gray = transforms.Compose([
                                 transforms.CenterCrop(224),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor()
                                ])

tansform_32 = transforms.Compose([
                                      transforms.CenterCrop(32),
                                      transforms.ToTensor()
                                      ])

tansform_32_gray = transforms.Compose([
                                 transforms.CenterCrop(32),
                                 transforms.Grayscale(num_output_channels=3),
                                 transforms.ToTensor()
                                ])

mvtec_labels = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
                'wood', 'zipper']


def get_normal_class(dataset='cifar10', normal_class_indx = 0):

    if dataset == 'cifar10':
        return get_CIFAR10_normal(normal_class_indx)
    elif dataset == 'mnist':
        return get_MNIST_normal(normal_class_indx)
    elif dataset == 'fashion':
        return get_FASHION_MNIST_normal(normal_class_indx)
    elif dataset == 'svhn':
        return get_SVHN_normal(normal_class_indx)
    elif dataset == 'mvtec':
        return get_MVTEC_normal(normal_class_indx)
    else:
        raise Exception("Dataset is not supported yet. ")


def get_CIFAR10_normal(normal_class_indx):
    trainset = CIFAR10(root=CIFAR10_PATH, train=True, download=True)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = CIFAR10(root=CIFAR10_PATH, train=False, download=True)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return trainset.data, testset



def get_MNIST_normal(normal_class_indx):
    trainset = MNIST(root=MNIST_PATH, train=True, download=True)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = MNIST(root=MNIST_PATH, train=False, download=True)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return trainset.data, testset


def get_FASHION_MNIST_normal(normal_class_indx):
    trainset = FashionMNIST(root=FMNIST_PATH, train=True, download=True)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = FashionMNIST(root=FMNIST_PATH, train=False, download=True)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return trainset.data, testset

def get_SVHN_normal(normal_class_indx):
    trainset = SVHN(root=SVHN_PATH, split='train', download=True)
    trainset.data = trainset.data[np.array(trainset.labels) == normal_class_indx]

    testset = SVHN(root=SVHN_PATH, split='test', download=True)
    testset.labels  = [int(t!=normal_class_indx) for t in testset.labels]

    return trainset.data, testset


class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, normal=True):
        self.transform = transform
        if train:
            self.data = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
          image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
          normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          self.data = image_files

        self.data.sort(key=lambda y: y.lower())
        self.data = [Image.open(x).convert('RGB') for x in self.data]
        self.train = train

    def __getitem__(self, index):
        image_file = self.data[index]

        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.data)


def get_MVTEC_normal(normal_class_indx):
    normal_class = mvtec_labels[normal_class_indx]

    trainset = MVTecDataset(MVTEC_PATH, normal_class, train=True)
    testset = MVTecDataset(MVTEC_PATH, normal_class, train=False)

    return trainset.data, testset


def download_and_extract_mvtec(path):
    import os
    import wget
    import tarfile
    so.extractall(path=os.environ['BACKUP_DIR'])
    url = 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'
    filename = wget.download(url, out=path)
    with tarfile.open(os.path.join(path, filename)) as so:
        so.extractall(path=os.path.join(path, "mvtec_anomaly_detection"))


def get_exposure(dataset='cifar10', normal_dataset='cifar100', normal_class_indx = 0):

    if dataset == 'cifar10':
        return get_CIFAR10_exposure(normal_class_indx)
    elif dataset == 'mnist':
        return get_MNIST_exposure(normal_class_indx)
    elif dataset == 'fashion':
        return get_FASHION_MNIST_exposure(normal_class_indx)
    elif dataset == 'svhn':
        return get_SVHN_exposure(normal_class_indx)
    elif dataset == 'mvtec':
        return get_MVTEC_exposure(normal_class_indx)
    else:
        raise Exception("Dataset is not supported yet. ")
    

def copy_dataset(dataset, target_count):
    pass


def get_CIFAR10_exposure(normal_dataset, normal_class_indx:int, count:int):
    assert count > 0
    
    exposure_train = CIFAR10(root=CIFAR10_PATH, train=True, download=True)
    exposure_test = CIFAR10(root=CIFAR10_PATH, train=False, download=True)

    if normal_dataset.lower() == 'cifar10':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]
        exposure_test.data = exposure_test.data[np.array(exposure_test.targets) != normal_class_indx]

    exposure_data = exposure_train.data

    if len(exposure_data) < count:
        exposure_data += exposure_test.data
    
    if len(exposure_data) < count:
        copy_dataset(exposure_data, count)


    return exposure_data


def get_MNIST_exposure(normal_dataset, normal_class_indx:int, count:int):
    assert count > 0
    
    exposure_train = MNIST(root=MNIST_PATH, train=True, download=True)
    exposure_test = MNIST(root=MNIST_PATH, train=False, download=True)

    if normal_dataset.lower() == 'mnist':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]
        exposure_test.data = exposure_test.data[np.array(exposure_test.targets) != normal_class_indx]

    exposure_data = exposure_train.data

    if len(exposure_data) < count:
        exposure_data += exposure_test.data
    
    if len(exposure_data) < count:
        copy_dataset(exposure_data, count)


    return exposure_data


def get_FASHION_MNIST_exposure(normal_dataset, normal_class_indx:int, count:int):
    assert count > 0
    
    exposure_train = FashionMNIST(root=FMNIST_PATH, train=True, download=True)
    exposure_test = FashionMNIST(root=FMNIST_PATH, train=False, download=True)

    if normal_dataset.lower() == 'fmnist':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]
        exposure_test.data = exposure_test.data[np.array(exposure_test.targets) != normal_class_indx]

    exposure_data = exposure_train.data

    if len(exposure_data) < count:
        exposure_data += exposure_test.data
    
    if len(exposure_data) < count:
        copy_dataset(exposure_data, count)


    return exposure_data


def get_SVHN_exposure(normal_dataset, normal_class_indx:int, count:int):
    assert count > 0
    
    exposure_train = SVHN(root=SVHN_PATH, split='train', download=True)
    exposure_test = SVHN(root=SVHN_PATH, split='test', download=True)

    if normal_dataset.lower() == 'svhn':
        exposure_train.data = exposure_train.data[np.array(exposure_train.targets) != normal_class_indx]
        exposure_test.data = exposure_test.data[np.array(exposure_test.targets) != normal_class_indx]

    exposure_data = exposure_train.data

    if len(exposure_data) < count:
        exposure_data += exposure_test.data
    
    if len(exposure_data) < count:
        copy_dataset(exposure_data, count)


    return exposure_data


class MVTecDatasetExposure(Dataset):
    def __init__(self, root, category=None, transform=None):
        self.transform = transform
        self.data = glob(os.path.join(root, "**", "*.png"), recursive=True)

        if category is not None:
          class_files = glob(os.path.join(root, category, "**", "*.png"), recursive=True)
          self.data = list(set(self.data) - set(class_files))

        self.data.sort(key=lambda y: y.lower())
        self.data = [Image.open(x).convert('RGB') for x in self.data]

    def __getitem__(self, index):
        image_file = self.data[index]
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.data)
    


def get_MVTEC_exposure(normal_dataset, normal_class_indx:int, count:int):
    assert count > 0
    
    exposure_data = MVTecDatasetExposure(root=SVHN_PATH).data

    if len(exposure_data) < count:
        copy_dataset(exposure_data, count)

    return exposure_data