from torchvision import models
import torch
import torch.nn as nn
import torch.optim as optim
import Models.preactresnet as preactresnet

class Net(nn.Module):
  def __init__(self, model:str='preactresnet18', pretrained:bool=False):
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

    elif model == 'resnet18':
        self.model = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
    elif model == 'resnet34':
        self.model = models.resnet34(weights=None if not pretrained else models.ResNet34_Weights.DEFAULT)
    elif model == 'resnet50':
        self.model = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
    elif model == 'resnet101':
        self.model = models.resnet101(weights=None if not pretrained else models.ResNet101_Weights.DEFAULT)
    elif model == 'resnet152':
        self.model = models.resnet152(weights=None if not pretrained else models.ResNet152_Weights.DEFAULT)

    elif model == 'vit_b_16':
        self.model = models.vit_b_16(image_size=32,weights=models.ViT_B_16_Weights.DEFAULT)
    elif model == 'vit_b_32':
        self.model = models.vit_b_32(weights=None if not pretrained else models.ViT_B_32_Weights.DEFAULT)
    elif model == 'vit_l_16':
        self.model = models.vit_l_16(weights=None if not pretrained else models.ViT_L_16_Weights.DEFAULT)
    elif model == 'vit_l_32':
        self.model = models.vit_l_32(weights=None if not pretrained else models.ViT_L_32_Weights.DEFAULT)
    elif model == 'vit_h_14':
        self.model = models.vit_h_14(weights=None if not pretrained else models.ViT_H_16_Weights.DEFAULT)

    else:
      raise ValueError('Invalid Model Type!')

    self.head = nn.Linear(1000, 2)

  def forward(self, x):
    x = self.model(x)
    x = self.head(x)
    return x

  def get_feature_vector(self, x):
    x = self.model(x)
    return x