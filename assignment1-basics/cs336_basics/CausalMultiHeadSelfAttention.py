from einops import rearrange, einsum
from cs336_basics.Linear import Linear
from cs336_basics.RotaryPositionalEmbedding import RotaryPositionalEmbedding

import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Given a tensor of inputs, return the output of softmaxing the given `dim`
    of the input.

    Args:
        in_features (Float[Tensor, "..."]): Input features to softmax. Shape is arbitrary.
        dim (int): Dimension of the `in_features` to apply softmax to.

    Returns:
        Float[Tensor, "..."]: Tensor of with the same shape as `in_features` with the output of
        softmax normalizing the specified `dim`.
    """
    max_num = torch.max(in_features,dim=dim,keepdim=True).values
    exp_tensor = torch.exp(in_features - max_num)
    sum_exp = torch.sum(exp_tensor,dim=dim,keepdim=True)
    return exp_tensor / sum_exp

def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """
    d_k = Q.shape[-1]
    scores = einsum(Q,K,"... q d_k,... k d_k -> ... q k") / (d_k ** 0.5)
    if mask is not None:
        #对于mask的，我们直接将mask为False的位置设置为负无穷
        scores = scores.masked_fill(~mask,float("-inf"))
    attn_weights = run_softmax(scores,dim=-1)
    return einsum(attn_weights,V,"... q k,... k d_v -> ... q d_v")

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads,seq_len,device,use_rope=True):
        """
        d_model: int,模型维度
        num_heads: int,注意力头数
        device: torch.device,设备
        use_rope: bool,是否使用RoPE
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device
        self.theta = 10000.0

        self.W_q = Linear(self.d_model, self.d_model, self.device)
        self.W_k = Linear(self.d_model, self.d_model, self.device)
        self.W_v = Linear(self.d_model, self.d_model, self.device)
        self.W_o = Linear(self.d_model, self.d_model, self.device)
        if use_rope:
            self.use_rope = True
            self.RoPE = RotaryPositionalEmbedding(theta=self.theta,d_k = self.d_k, max_seq_len = seq_len, device = device)
        else:
            self.use_rope = False

    def forward(self,x,token_positions):
        """
        inputs:
        x: Float[Tensor, "batch_size seq_len d_model"]
        token_positions: Long[Tensor, "batch_size seq_len"]
        returns:
        out: Float[Tensor, "batch_size seq_len d_model"]
        """
        batch_size,seq_len,_ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        Q = rearrange(Q,"b s (h d_k) -> b h s d_k",h=self.num_heads)
        K = rearrange(K,"b s (h d_k) -> b h s d_k",h=self.num_heads)
        V = rearrange(V,"b s (h d_k) -> b h s d_k",h=self.num_heads)
        if self.use_rope:
            Q = self.RoPE(Q,token_positions)
            K = self.RoPE(K,token_positions)
        
        mask = torch.tril(torch.ones((seq_len,seq_len),device=self.device)).bool()
        attn_output = run_scaled_dot_product_attention(Q,K,V,mask=mask)
        attn_output = rearrange(attn_output,"b h s d_k -> b s (h d_k)")
        out = self.W_o(attn_output)
        return out  

        
