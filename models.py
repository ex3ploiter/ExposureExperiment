from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import preactresnet

class Net(nn.Module):
  def __init__(self, model:str='preactresnet18', pretrained:bool=True):
    super(Net, self).__init__()

    # Load a pretrained resnet model from torchvision.models in Pytorch
    self.model = None
    
    if model == 'preactresnet18':
        self.model = preactresnet.PreActResNet18()
    elif model == 'preactresnet34':
        self.model = preactresnet.PreActResNet34()
    elif model == 'preactresnet50':
        self.model = preactresnet.PreActResNet50()
    elif model == 'preactresnet101':
        self.model = preactresnet.PreActResNet101()
    elif model == 'preactresnet152':
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

