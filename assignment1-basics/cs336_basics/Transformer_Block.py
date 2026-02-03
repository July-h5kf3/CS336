import torch.nn as nn
from cs336_basics.CausalMultiHeadSelfAttention import CausalMultiHeadSelfAttention
from cs336_basics.Linear import Linear
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.SwiGLU import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,seq_len,ffn_hidden_dim,device,use_rope=True):
        super().__init__()
        self.attention = CausalMultiHeadSelfAttention(d_model,num_heads,seq_len,device,use_rope)
        self.ffn = SwiGLU(d_model,device)
        # pass device explicitly to avoid treating it as eps
        self.attn_norm = RMSNorm(d_model, device=device)
        self.ffn_norm = RMSNorm(d_model, device=device)
    def forward(self,x,token_positions):
        x_norm = self.attn_norm(x)
        attn_out = self.attention(x_norm,token_positions)
        x = x + attn_out
        x_norm = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x
