import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhSoftCapping(nn.Module):
    def __init__(self, soft_cap_value: float = 30.0):
        super().__init__()
        self.soft_cap_value = soft_cap_value

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.soft_cap_value * F.tanh(logits / self.soft_cap_value)
