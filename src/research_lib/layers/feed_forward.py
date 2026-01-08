from typing import Callable, Optional

import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        activation=None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.activation = activation if activation else nn.GELU()

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
