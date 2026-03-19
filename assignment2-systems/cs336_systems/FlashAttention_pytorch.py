import torch
import math
from einops import einsum

TILE_SIZE = 16

class FlashAttention_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V,is_causal=False):
        batch_size,seq_len,d_model = Q.shape
        device = Q.device
        dtype = Q.dtype
        # S = torch.zeros(batch_size,TILE_SIZE,TILE_SIZE)
        sqrt_d = math.sqrt(d_model)
        O = torch.zeros(batch_size,seq_len,d_model,dtype=dtype,device=device)
        L = torch.zeros(batch_size,seq_len,dtype=dtype,device=device)
        for i in range(0,seq_len // TILE_SIZE):
            m = torch.full((batch_size,TILE_SIZE),float("-inf"),device=device)
            l = torch.zeros(batch_size,TILE_SIZE,dtype=dtype,device=device)
            B_q = Q[:,i*TILE_SIZE:(i + 1) * TILE_SIZE,:]
            for j in range(0,seq_len // TILE_SIZE):
                B_k = K[:,j * TILE_SIZE:(j + 1) * TILE_SIZE,:]
                S = einsum(B_q,B_k,"... B_q d,... B_k d -> ... B_q B_k") / sqrt_d
                last_m,last_l = m.clone(),l.clone()
                m = torch.maximum(m,torch.amax(S,dim=2))
                l = last_l * torch.exp(last_m - m) + torch.sum(torch.exp(S - m[:,:,None]),dim=2) 
                S = torch.exp(S - m[:,:,None])
                O[:,i*TILE_SIZE:(i+1)*TILE_SIZE,:] = O[:,i*TILE_SIZE:(i+1)*TILE_SIZE,:] * torch.exp(last_m - m).unsqueeze(-1) + einsum(S,V[:,j*TILE_SIZE:(j+1)*TILE_SIZE,:],"... B_q B_k,... B_k d_model->... B_q d_model")
            O[:,i*TILE_SIZE:(i+1)*TILE_SIZE,:] = O[:,i*TILE_SIZE:(i+1)*TILE_SIZE,:] / l.unsqueeze(-1)
            L[:,i*TILE_SIZE:(i+1)*TILE_SIZE] = torch.log(l) + m
        ctx.save_for_backward(L,Q,K,V,O)
        return O
    def backward(ctx,grad_output):
        raise NotImplementedError








