""" Negative Cosine Similarity & Symmetrized Negative Cosine Similarity """

import torch
import torch.nn as nn


class NegativeCosineSimilarity(nn.Module):
    """Negative Cosine Similarity
    """

    def __init__(self):
        super(NegativeCosineSimilarity, self).__init__()
        self.neg_cos = nn.CosineSimilarity()

    def forward(self, y1: torch.Tensor, y2: torch.Tensor) -> torch.Tensor:
        return -self.neg_cos(y1, y2)


class SymmetrizedNegativeCosineSimilarity(nn.Module):
    """Symmetrized Negative Cosine Similarity
    """

    def __init__(self):
        super(SymmetrizedNegativeCosineSimilarity, self).__init__()
        self.neg_cos = NegativeCosineSimilarity()

    def forward(self, p1: torch.Tensor, z1: torch.Tensor, p2: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        loss1 = self.neg_cos(p1, z2.detach()).mean()
        loss2 = self.neg_cos(p2, z1.detach()).mean()

        return (loss1 + loss2) * 0.5
