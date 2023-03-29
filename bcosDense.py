import torch.nn.functional as F
from torch import nn
import numpy as np
from utils import eps

"""
    Dense layer implementation of B-cos weight computation
    Implementation based on https://github.com/moboehle/B-cos
    papers: https://arxiv.org/pdf/2205.10268.pdf and
    https://arxiv.org/abs/2301.08669
"""

class NormedDense(nn.Linear):
    def forward(self, x):
        shape = self.weight.shape
        w = self.weight.view(shape[0], -1)
        w = w/w.norm(p=2, dim=1, keepdim=True)
        return F.linear(x, w.view(shape))

class BCosDense(nn.Module):
    def __init__(self, in_features, out_features, b, max_out=1, scale=None,
                 scale_fact=100, **kwargs):
        super().__init__()

        self.outc = out_features + max_out
        self.normed_dense = NormedDense(in_features, self.outc, bias=False)
        self.b = b
        self.max_out = max_out
        self.inc = in_features
        self.detach = False
        if scale is None:
            self.scale = np.sqrt(self.inc) / scale_fact
        else:
            self.scale = scale

    def forward(self, x, detach=False):
        """
            Args:
                x: input tensor of shape (batch_size, leads, signal)
                detach: if True enter in 'explanation mode'
        """

        out = self.normed_dense(x)
        bs, s = out.shape

        # maxout
        if self.max_out > 1:
            out = out.view(bs, -1, self.max_out, s)
            out = out.max(dim=2, keepdim=False)[0]

        # normalized avg_pooling

        norm = (F.avg_pool1d((x ** 2).sum(1, keepdim=True),
                             1,1,0
                             ) + eps).sqrt()
        abs_cos = (out / norm).abs()

        if self.detach:
            abs_cos = abs_cos.detach()

        # b-cos
        out = out * (abs_cos ** (self.b - 1))
        return out / self.scale
