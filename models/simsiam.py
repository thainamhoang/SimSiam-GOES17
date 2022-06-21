""" SimSiam """

import torch
import torch.nn as nn
from typing import Tuple


class SimSiam(nn.Module):
    """Rectangular input implementation of SimSiam model, based on https://arxiv.org/abs/2011.10566

    Attributes:
        encoder:
            Encoder backbone for the network, here we use resnet18()
        input_dim:
            Input dimension for the network
        proj_hidden_dim:
            Projection hidden dim
        pred_hidden_dim:
            Prediction hidden dim
        output_dim:
            Output dimension for the network

    """
    def __init__(self,
                 encoder: nn.Module,
                 input_dim: int = 2048,
                 proj_hidden_dim: int = 2048,
                 pred_hidden_dim: int = 512,
                 output_dim: int = 2048):
        super(SimSiam, self).__init__()
        self.encoder = encoder(num_classes=input_dim)

        self.projection_head = nn.Sequential(nn.Linear(input_dim, proj_hidden_dim, bias=False),
                                             nn.BatchNorm1d(proj_hidden_dim),
                                             nn.ReLU(inplace=True),  # 1st block
                                             nn.Linear(proj_hidden_dim, proj_hidden_dim, bias=False),
                                             nn.BatchNorm1d(proj_hidden_dim),
                                             nn.ReLU(inplace=True),  # 2nd block
                                             nn.Linear(proj_hidden_dim, output_dim, bias=False),
                                             nn.BatchNorm1d(output_dim, affine=False))  # 3rd block

        self.prediction_head = nn.Sequential(nn.Linear(output_dim, pred_hidden_dim, bias=False),
                                             nn.BatchNorm1d(pred_hidden_dim),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(pred_hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encoder(x).flatten(start_dim=1)  # Get representation
        z = self.projection_head(f)  # Get projections
        p = self.prediction_head(f)  # Get predictions
        z = z.detach()  # Stop gradient
        return z, p
