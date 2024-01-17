import torchvision
import torch
from torch.nn import Sequential, Linear, ReLU
import timm
from vision_transformer import HybridViT, DeiT

    
class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""
    def __init__(self):
        super(ResNet, self).__init__()
        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = 2048

        del self.network.fc
        self.network.fc = Identity()
    
    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)


class Identity(torch.nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def Featurizer(model_type):
    if model_type == 'hvit':
        return HybridViT()
    elif model_type == 'deit':
        return DeiT()
    elif model_type == 'resnet50':
        return ResNet()
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

