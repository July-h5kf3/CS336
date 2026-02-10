import torch
import torch.nn as nn
import math
from einops import rearrange, einsum

class SwiGLU(nn.Module):
    def __init__(self,d_model,device = None,dtype = None):
        super().__init__()
        self.d_model = d_model
        #根据指导手册要求，但是需要保证是64的整数倍
        d_ff = int(math.ceil((8/3) * d_model / 64) * 64)
        
        self.W1 = torch.nn.Parameter(torch.empty((d_ff,d_model),device=device,dtype=dtype))
        self.W2 = torch.nn.Parameter(torch.empty((d_model,d_ff),device=device,dtype=dtype))
        self.W3 = torch.nn.Parameter(torch.empty((d_ff,d_model),device=device,dtype=dtype))

        sigma1 = (2 / (d_model + d_ff)) ** 0.5
        sigma2 = (2 / (d_ff + d_model)) ** 0.5

        torch.nn.init.trunc_normal_(self.W1,mean = 0.0,std = sigma1,a = -3.0 * sigma1,b = 3.0 * sigma1)
        torch.nn.init.trunc_normal_(self.W2,mean = 0.0,std = sigma2,a = -3.0 * sigma2,b = 3.0 * sigma2)
        torch.nn.init.trunc_normal_(self.W3,mean = 0.0,std = sigma1,a = -3.0 * sigma1,b = 3.0 * sigma1)
    def forward(self,x):
        x_proj1 = einsum(x,self.W1,"... d_model,d_ff d_model->... d_ff")
        x_proj1 = x_proj1 * torch.sigmoid(x_proj1) #计算SiLU激活
        x_proj2 = einsum(x,self.W3,"... d_model,d_ff d_model->... d_ff")
        x_glu = x_proj1 * x_proj2
        out = einsum(x_glu,self.W2,"... d_ff,d_model d_ff->... d_model")
        return out