import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredReLU(nn.Module):
    """
    Squared ReLU: (max(0, x))^2
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return torch.square(self.relu(x))
