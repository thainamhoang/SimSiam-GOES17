""" Linear Batch norm ReLU"""

import torch
import torch.nn as nn


class LnBnReLU(nn.Module):
    """ Create a layer consists of nn.Linear, nn.BatchNorm1d (if provided), and nn.ReLU (if provided).

    Attributes:
        input_dim:
            Input dimension
        output_dim:
            Output dimension

    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 batch_norm: bool = True,
                 relu: bool = True):
        super(LnBnReLU, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=not batch_norm)
        self.batch_norm = nn.BatchNorm1d(output_dim) if batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x
