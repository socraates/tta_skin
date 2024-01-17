import torch
import timm


class Identity(torch.nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

""" 
Note: The implementation is not used main experimennts. Please use ViT2. 

class ViT(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        super().__init__()
        self.network = pytorch_pretrained_vit.ViT(
            hparams['backbone'], pretrained=True
        )
        self.n_outputs = self.network.fc.in_features
        del self.network.fc
        self.network.fc = Identity()
        self.hparams = hparams

    def forward(self, x):
        return self.network(x)
"""


class DeiT(torch.nn.Module):
    KNOWN_MODELS = {
        'DeiT': timm.models.vision_transformer.vit_deit_base_distilled_patch16_224
    }
    def __init__(self):
        super().__init__()
        func = self.KNOWN_MODELS['DeiT']
        self.network = func(pretrained=True)
        self.n_outputs = self.network.norm.normalized_shape[0]
        self.network.head = Identity()
        if hasattr(self.network, 'head_dist'):
            self.network.head_dist = None

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        y = self.network(x)
        return (y[0] + y[1]) / 2 

class HybridViT(torch.nn.Module):
    KNOWN_MODELS = {
        'HViT_SMALL': timm.models.vision_transformer_hybrid.vit_small_r26_s32_224_in21k
    }
    def __init__(self):
        super().__init__()
        func = self.KNOWN_MODELS['HViT_SMALL']
        self.network = func(pretrained=True)
        self.n_outputs = self.network.norm.normalized_shape[0]
        self.network.head = Identity()
        if hasattr(self.network, 'head_dist'):
            self.network.head_dist = None

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
