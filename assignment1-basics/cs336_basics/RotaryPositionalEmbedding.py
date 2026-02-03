from einops import rearrange, einsum
import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta,d_k,max_seq_len,device):
        """
        d_k: int, 维度大小，必须为偶数
        theta: float, RoPE中的\Theta值
        max_seq_len: int, 最大序列长度
        device: torch.device, 设备
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        #一共有d / 2个频率
        half_dk = d_k // 2
        k = torch.arange(0,half_dk,device=device).float()
        inv_freq = 1.0 / (self.theta ** (2.0 * k / d_k))

        positions = torch.arange(0,max_seq_len,device = device).float()

        angles = einsum(positions,inv_freq,"max_seq_len,half_dk->max_seq_len half_dk")
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos",cos,persistent = False)
        self.register_buffer("sin",sin,persistent = False)

    def forward(self,x,token_positions):
        """
        inputs:
            x: ...,seq_len,d_k
            token_positions:...,seq_len
        returns:
            x_rotated: ...,seq_len,d_k
        """
        cos = self.cos[token_positions]  # ...,seq_len,half_dk
        sin = self.sin[token_positions]  # ...,seq_len,half_dk

        x_even = x[...,0::2]
        x_odd = x[...,1::2]

        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[...,0::2] = x_rot_even
        out[...,1::2] = x_rot_odd
        return out
