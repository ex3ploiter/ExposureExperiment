from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import preactresnet

class FeatureExtractor(nn.Module):
  def __init__(self, model:int=18, pretrained:bool=True):
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
        raise ValueError('Invalid Model Type!')

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

