from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import preactresnet

from wideresnet import WideResNet

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

class FeatureExtractor(nn.Module):
  def __init__(self, model=18, pretrained=True):
    super(FeatureExtractor, self).__init__()

    # Load a pretrained resnet model from torchvision.models in Pytorch
    self.model = None
    
    if model == 18:
        self.model = preactresnet.PreActResNet18()
    elif model == 34:
        self.model = preactresnet.PreActResNet34()
    elif model == 50:
        self.model = preactresnet.PreActResNet50()
    elif model == 101:
        self.model = preactresnet.PreActResNet101()
    elif model == 152:
        self.model = preactresnet.PreActResNet152()
    else:
        model = WideResNet(34, 10, widen_factor=10, dropRate=0.0)

    num_ftrs = self.model.linear.in_features
    self.model.linear = nn.Flatten()
    self.head = nn.Linear(num_ftrs, 2)

  def forward(self, x):
    x = self.model(x)
    x = self.head(x)
    return x

  def get_feature_vector(self, x):
    x = self.model(x)
    return x

