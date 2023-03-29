import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
from utils import eps
from bcosDense import BCosDense
from einops import rearrange

"""
    Implementation based on: 
            https://github.com/moboehle/B-cos and
            https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_1d.py
    papers: 
            https://arxiv.org/pdf/2205.10268.pdf and
            https://arxiv.org/pdf/2301.08669.pdf
"""

## attBlock(softmax(Q*K))
##Maxout(B-cos(P,S))

class BCos(nn.Module):
    def __init__(self, dim, max_out=1, b=2, scale=None,
                 scale_fact=100, **kwargs):
        super().__init__()
        self.b = b
        self.max_out = max_out
        self.layer_norm = nn.LayerNorm(dim)

        if scale is None:
            self.scale = (dim) ** 0.5 / scale_fact
        else:
            self.scale = scale


    def forward(self, x):
        # Layer normalization
        p = self.layer_norm(x)
        bs, s = p.shape

        # MaxOut computation
        if self.max_out > 1:
            p = p.view(bs, -1, self.max_out, s)
            p = p.max(dim=2, keepdim=False)[0]

        # B-cos computation
        norm = (F.avg_pool1d((p ** 2).sum(1, keepdim=True),1,1,0)
                    + eps) ** 0.5
        abs_cos = (p / norm).abs() + eps

        out = p * abs_cos.pow(self.b-1)
        return out / self.scale

class BCosAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=64):
        super().__init__()
        
        self.scale = dim_head ** -0.5
        self.sm = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.Wv = BCos(dim)
        self.Wu = BCos(dim)
        self.qkv = nn.Linear(dim, dim_head * num_heads * 3, bias=False)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q,k,v = map(lambda aux: rearrange(aux, 'b (h d) -> b (h d)', 
                                          h=self.num_heads), qkv)
        wvp = self.Wv(x)
        ah = self.sm(torch.matmul(q,k.transpose(-2,-1)))
        x_= torch.matmul(ah, wvp)
        wvu = self.Wu(x_)
        out = torch.matmul(torch.matmul(wvu, x_.transpose(-2,-1)), x)

        return out
 
class BCosTransformer(nn.Module):
    def __init__(self, dim, depth, heads, nclass, dim_head):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                BCosAttention(dim, heads, dim_head),
                BCosDense(dim, dim - 1, b=2)
            ]))
        self.classifier = BCosDense(dim, nclass - 1, b=2)

    def forward(self, x):
        for attn, fc in self.layers:
            x = attn(x) + x
            x = fc(x) + x
        x = self.classifier(x)
        return x
