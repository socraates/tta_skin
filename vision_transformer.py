"""
DeiT class selected from https://github.com/matsuolab/T3A
"""
import torch
import timm

class Identity(torch.nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DeiT(torch.nn.Module):
    KNOWN_MODELS = {
        'DeiT-S': timm.models.deit.deit3_small_patch16_224_in21ft1k
    }
    def __init__(self):
        super().__init__()
        func = self.KNOWN_MODELS['DeiT-S']
        self.network = func(pretrained=True)
        self.n_outputs = self.network.norm.normalized_shape[0]
        self.network.head = Identity()
        if hasattr(self.network, 'head_dist'):
            self.network.head_dist = None

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.network(x)
        