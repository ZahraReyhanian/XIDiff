import math

import torch
from torch import nn
import torch.nn.functional as F

"""
This code is from this link: https://github.com/GOKORURI007/pytorch_arcface/blob/main/arcface.py
"""

class ArcFace(nn.Module):
    def __init__(self, embed_size, num_classes, scale=64, margin=0.5, easy_margin=False, **kwargs):
        """
        The input of this Module should be a Tensor which size is (N, embed_size), and the size of output Tensor is (N, num_classes).

        arcface_loss =-\sum^{m}_{i=1}log
                        \frac{e^{s\psi(\theta_{i,i})}}{e^{s\psi(\theta_{i,i})}+
                        \sum^{n}_{j\neq i}e^{s\cos(\theta_{j,i})}}
        \psi(\theta)=\cos(\theta+m)
        where m = margin, s = scale
        """
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_size))
        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding: torch.Tensor, ground_truth):
        """
        This Implementation is from https://github.com/ronghuaiyang/arcface-pytorch, which takes
        54.804054962005466 ms for every 100 times of input (50, 512) and output (50, 10000) on 2080Ti.
        """

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(embedding), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        sin_theta = torch.sqrt((1.0 - torch.pow(cos_theta, 2)).clamp(-1 + 1e-7, 1 - 1e-7))
        phi = cos_theta * self.cos_m - sin_theta * self.sin_m
        if self.easy_margin:
            phi = torch.where(cos_theta > 0, phi, cos_theta)
        else:
            phi = torch.where(cos_theta > self.th, phi, cos_theta - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size(), device='cuda')
        one_hot.scatter_(1, ground_truth.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cos_theta)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.scale

        loss = self.ce(output, ground_truth)
        return loss