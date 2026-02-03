from cs336_basics.Transformer_Block import TransformerBlock
from cs336_basics.Embedding import Embedding
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.Linear import Linear
import torch.nn as nn
import torch
from einops import rearrange, einsum

class Transformer:
    def __init__(self,vocab_size,context_length,num_layers,d_model,num_heads,device):
        self.TokenEmbedding = Embedding(vocab_size,d_model,device)
        self.TransformerBlocks = nn.ModuleList([
            TransformerBlock(d_model,num_heads,context_length,ffn_hidden_dim=None,device=device,use_rope=True)
            for _ in range(num_layers)
        ])
        self.FinalNorm = RMSNorm(d_model,device=device)
        self.OutputLayer = Linear(d_model,vocab_size,device)
    def forward(self,token_ids):
        """
        inputs:
        token_ids: Long[Tensor, "batch_size seq_len"]
        returns:
        logits: Float[Tensor, "batch_size seq_len vocab_size"]
        """
        batch_size,seq_len = token_ids.shape
        x = self.TokenEmbedding(token_ids)  # [batch_size, seq_len, d_model]
        token_positions = torch.arange(seq_len,device=x.device).unsqueeze(0).expand(batch_size,-1)  # [batch_size, seq_len]
        for block in self.TransformerBlocks:
            x = block(x,token_positions)
        x = self.FinalNorm(x)
        logits = self.OutputLayer(x)
        return logits