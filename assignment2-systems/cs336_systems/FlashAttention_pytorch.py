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
            m = torch.full((batch_size,TILE_SIZE),float("-inf"),dtype=dtype,device=device)
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
        L,Q,K,V,O = ctx.saved_tensors
        batch_size,N_QUERIES,d_model = Q.shape
        _,N_KEYS,_ = K.shape
        scale = 1.0 / (d_model ** 0.5)
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        D = torch.sum(grad_output * O,dim=2)
        for i in range(0,N_QUERIES // TILE_SIZE):
            Q_b = Q[:,i*TILE_SIZE:(i+1)*TILE_SIZE,:]
            L_b  =L[:,i*TILE_SIZE:(i+1)*TILE_SIZE]
            dO_b = grad_output[:,i*TILE_SIZE:(i+1)*TILE_SIZE,:]
            D_b = D[:,i*TILE_SIZE:(i+1)*TILE_SIZE]
            for j in range(0,N_KEYS // TILE_SIZE):
                K_b = K[:,j * TILE_SIZE:(j + 1) * TILE_SIZE,:]
                V_b = V[:,j*TILE_SIZE:(j+1)*TILE_SIZE,:]
                P_ij = torch.exp(torch.matmul(Q_b,K_b.transpose(-1,-2)) * scale - L_b[:,:,None])
                dV[:,j*TILE_SIZE:(j+1)*TILE_SIZE,:] += torch.matmul(P_ij.transpose(-1,-2),dO_b)
                dP_ij = torch.matmul(dO_b,V_b.transpose(-1,-2))
                dS_ij = P_ij * (dP_ij - D_b[:,:,None])
                dQ[:,i*TILE_SIZE:(i+1)*TILE_SIZE,:] += torch.matmul(dS_ij,K_b) * scale
                dK[:,j*TILE_SIZE:(j+1)*TILE_SIZE,:] += torch.matmul(dS_ij.transpose(-1,-2),Q_b) * scale
        return dQ,dK,dV,None
                







