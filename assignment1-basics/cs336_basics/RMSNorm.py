import torch
from einops import rearrange, einsum
import math
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps = 1e-5,device = None,dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        #依据实验指导书，初始化为全1
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self,x):
        #为了数值稳定，先将类型转化为float32
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms
        x_normed = x_normed * self.weight
        return x_normed.to(in_dtype)