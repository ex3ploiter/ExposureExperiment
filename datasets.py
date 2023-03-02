
from torchvision.datasets import CIFAR10, MNIST, SVHN, FashionMNIST
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from glob import glob


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


def get_normal_class(dataset='cifar', path='~/mydataset', normal_class_indx = 0, batch_size=8, img_size=32):

    assert img_size == 32 or img_size == 224

    if dataset == 'cifar10':
        return get_CIFAR10_normal(normal_class_indx, batch_size, path, img_size)
    elif dataset == 'mnist':
        return get_MNIST_normal(normal_class_indx, batch_size, path, img_size)
    elif dataset == 'fashion':
        return get_FASHION_MNIST_normal(normal_class_indx, batch_size, path, img_size)
    elif dataset == 'svhn':
        return get_SVHN_normal(normal_class_indx, batch_size, path, img_size)
    elif dataset == 'mvtec':
        return get_MVTEC(normal_class_indx, batch_size, path, img_size)
    else:
        raise Exception("Dataset is not supported yet. ")


def get_CIFAR10_normal(normal_class_indx, batch_size, path, img_size):
    tansform_dataset = tansform_224 if img_size==224 else tansform_32

    trainset = CIFAR10(root=path, train=True, download=True, transform=tansform_dataset)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = CIFAR10(root=path, train=False, download=True, transform=tansform_dataset)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return trainset.data, testset



def get_MNIST_normal(normal_class_indx, batch_size, path, img_size):
    tansform_dataset = tansform_224_gray if img_size==224 else tansform_32_gray

    trainset = MNIST(root=path, train=True, download=True, transform=tansform_dataset)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = MNIST(root=path, train=False, download=True, transform=tansform_dataset)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return trainset.data, testset


def get_FASHION_MNIST_normal(normal_class_indx, batch_size, path, img_size):
    tansform_dataset = tansform_224_gray if img_size==224 else tansform_32_gray

    trainset = FashionMNIST(root=path, train=True, download=True)
    trainset.data = trainset.data[np.array(trainset.targets) == normal_class_indx]

    testset = FashionMNIST(root=path, train=False, download=True)
    testset.targets  = [int(t!=normal_class_indx) for t in testset.targets]

    return trainset.data, testset

def get_SVHN_normal(normal_class_indx, batch_size, path, img_size):
    tansform_dataset = tansform_224 if img_size==224 else tansform_32

    trainset = SVHN(root=path, split='train', download=True,)
    trainset.data = trainset.data[np.array(trainset.labels) == normal_class_indx]

    testset = SVHN(root=path, split='test', download=True)
    testset.labels  = [int(t!=normal_class_indx) for t in testset.labels]

    return trainset.data, testset


class MVTecDataset(Dataset):
    def __init__(self, root, category, transform=None, target_transform=None, train=True, normal=True):
        self.transform = transform
        if train:
            self.image_files = glob(
                os.path.join(root, category, "train", "good", "*.png")
            )
        else:
          image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
          normal_image_files = glob(os.path.join(root, category, "test", "good", "*.png"))
          anomaly_image_files = list(set(image_files) - set(normal_image_files))
          self.image_files = image_files

        self.image_files.sort(key=lambda y: y.lower())
        self.train = train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if os.path.dirname(image_file).endswith("good"):
            target = 0
        else:
            target = 1

        return image, target

    def __len__(self):
        return len(self.image_files)


def get_MVTEC(normal_class_indx, batch_size, path):
    normal_class = mvtec_labels[normal_class_indx]

    trainset = MVTecDataset(path, normal_class, train=True)
    testset = MVTecDataset(path, normal_class, train=False)

    return trainset[:, 0], testset


def download_and_extract_mvtec(path):
    import os
    import wget
    import tarfile
    so.extractall(path=os.environ['BACKUP_DIR'])
    url = 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'
    filename = wget.download(url, out=path)
    with tarfile.open(os.path.join(path, filename)) as so:
        so.extractall(path=os.path.join(path, "mvtec_anomaly_detection"))
