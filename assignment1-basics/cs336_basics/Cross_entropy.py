import torch
from einops import rearrange, einsum

def cross_entropy(logits, targets):
    """
    logits: Float[Tensor, "batch vocab_size"]
    targets: Int[Tensor, "batch"]
    """
    max_logits = logits.max(dim = -1,keepdim = True).values #shape: [batch_size,1]
    logits = logits - max_logits
    
    log_sum_exp = torch.log(torch.exp(logits).sum(dim=-1))  # [batch]
    loss = log_sum_exp - logits[torch.arange(logits.shape[0]),targets]
    return loss.mean()