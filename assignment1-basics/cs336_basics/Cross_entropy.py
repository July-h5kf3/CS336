import torch
from einops import rearrange, einsum

def cross_entropy(logits, targets):
    """
    logits: Float[Tensor, "batch vocab_size"]
    targets: Int[Tensor, "batch"]
    """
    max_logits = logits.max(dim = -1,keepdim = True).values #shape: [batch_size,1]
    #如果直接对logits减去最大值，最后会因为log操作出现下溢的情况
    log_sum_exp = torch.log(torch.exp(logits - max_logits).sum(dim=-1,keepdim=True)) + max_logits  # [batch,1]
    # [batch,1] -> [batch]
    log_sum_exp = rearrange(log_sum_exp,'b 1 -> b')
    loss = log_sum_exp - logits[torch.arange(logits.shape[0]),targets]
    return loss.mean()