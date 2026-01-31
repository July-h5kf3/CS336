import torch
import math
from einops import rearrange, einsum
class Linear(torch.nn.Module):
    def __init__(self,in_features,out_features,device = None,dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = torch.nn.Parameter(torch.empty((out_features,in_features),device = device,dtype = dtype))
        sigma = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.W,mean = 0.0,std = sigma,a = -3.0 * sigma,b = 3.0 * sigma)
    def forward(self,x):
        return einsum(x,self.W,"... d_in,d_out d_in->... d_out")