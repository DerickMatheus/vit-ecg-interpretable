import torch.nn.functional as F
from torch import nn
import numpy as np
from utils import eps

""" 
    Implementation based on https://github.com/moboehle/B-cos
    papers: https://arxiv.org/pdf/2205.10268.pdf and 
    https://arxiv.org/abs/2301.08669
"""

class NormedConv1d(nn.Conv1d):
    """
    Normalize the weights of the convolutional layer
    """
    def forward(self, x):
        shape = self.weight.shape
        w = self.weight.view(shape[0], -1)
        w = w/(w.norm(p=2, dim=1, keepdim=True))
        return F.conv1d(x, w.view(shape),
            self.bias, self.stride, self.padding, self.dilation, self.groups)

class BCosConv1d(nn.Module):
    def __init__(self, inc, outc, kernel_size=1, stride=1, padding=0,
                 max_out=2, b=2, scale=None, scale_fact=100, **kwargs):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.outc = outc * max_out
        self.normed_conv1d = NormedConv1d(inc, self.outc, self.kernel_size,
                                   stride, padding, 1, 1, bias=False)
        self.b = b
        self.max_out = max_out
        self.inc = inc
        self.kernel_size = self.kernel_size
        self.kernel_squared = self.kernel_size ** 2
        self.padding = padding
        self.detach = False
        if scale is None:
            ks_scale = kernel_size if not \
                    isinstance(kernel_size, tuple) else np.sqrt(np.prod(ks))
            self.scale = (ks_scale * np.sqrt(self.inc)) / scale_fact
        else:
            self.scale = scale

    def forward(self, x, detach=False):
        """
        Args:
            x: Input tensor. Expected shape: (Batch_size, leads, signal)
            detach: if true enter in 'explanation mode'
        Returns:
            BcosConv1d output on the input tensor.
        """
        # Simple linear layer
        out = self.normed_conv1d(x)
        bs, _, s = out.shape

        # MaxOut computation
        if self.max_out > 1:
            out = out.view(bs, -1, self.max_out, s)
            out = out.max(dim=2, keepdim=False)[0]

        # Calculating the norm of input patches. 
        # Use average pooling and upscale by kernel size.
        norm = (F.avg_pool1d((x ** 2).sum(1, keepdim=True),
                             self.kernel_size, padding=self.padding,
                             stride=self.stride) * self.kernel_squared +
                eps).sqrt()

        # get absolute value of cos
        # abs_cos = (out / norm).abs()
        #TODO teste without stabilizer term
        abs_cos = (out / norm).abs() + eps

        # In order to compute the explanations.
        # we detach the dynamically calculated scaling from the graph.
        if self.detach:
            abs_cos = abs_cos.detach()

        # additional factor of cos^(b-1) s.t.
        # in total we have norm * cos^b with original sign
        out = out * abs_cos.pow(self.b-1)
        return out / self.scale
