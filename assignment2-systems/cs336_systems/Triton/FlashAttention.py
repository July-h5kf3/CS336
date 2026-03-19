import triton
import triton.language as tl
import math

import torch
Q_TILE_SIZE = 16
K_TILE_SIZE = 16
@triton.jit
def flash_fwd_kernel(
    Q_ptr,K_ptr,V_ptr,
    O_ptr,L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        base = Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES,D),
        strides = (stride_qq,stride_qd),
        offsets = (query_tile_index * Q_TILE_SIZE,0),
        block_shape = (Q_TILE_SIZE,D),
        order = (1,0),
    )
    K_block_ptr = tl.make_block_ptr(
        base = K_ptr + batch_index * stride_kb,
        shape = (N_KEYS,D),
        strides = (stride_kk,stride_kd),
        offsets = (0,0),
        block_shape = (K_TILE_SIZE,D),
        order = (1,0),
    )
    V_block_ptr = tl.make_block_ptr(
        base = V_ptr + batch_index * stride_vb,
        shape = (N_KEYS,D),
        strides = (stride_vk,stride_vd),
        offsets = (0,0),
        block_shape = (K_TILE_SIZE,D),
        order =(1,0),
    )
    L_block_ptr = tl.make_block_ptr(
        base = L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (query_tile_index * Q_TILE_SIZE,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )
    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES,D),
        strides = (stride_oq,stride_od),
        offsets = (query_tile_index * Q_TILE_SIZE,0),
        block_shape = (Q_TILE_SIZE,D),
        order = (1,0),
    )
    
    m = tl.full((Q_TILE_SIZE,),float("-inf"),dtype=tl.float32)
    l = tl.zeros((Q_TILE_SIZE,),dtype = tl.float32)
    O = tl.zeros((Q_TILE_SIZE,D),dtype=tl.float32)
    B_q = tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")
    for j in range(tl.cdiv(N_KEYS,K_TILE_SIZE)):
        B_k = tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")
        B_v = tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero")
        S = tl.dot(B_q,tl.trans(B_k)) * scale
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0,Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0,K_TILE_SIZE)
            causal_mask = q_idx[:,None] < k_idx[None,:]
            S = S + causal_mask * (-1e6)
        last_m,last_l = m,l
        m = tl.maximum(last_m,tl.max(S,axis=1))
        S_ = tl.exp(S - m[:,None])
        l = last_l * tl.exp(last_m - m) + tl.sum(S_,axis=1)
        O = tl.dot(S_.to(B_v.dtype),B_v,acc=O * tl.exp(last_m - m)[:,None])
        K_block_ptr = tl.advance(K_block_ptr,(K_TILE_SIZE,0))
        V_block_ptr = tl.advance(V_block_ptr,(K_TILE_SIZE,0))
    O = O / l[:,None]
    L = tl.log(l) + m
    tl.store(O_block_ptr, O.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr,L,boundary_check=(0,))

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V,is_causal=False):
        batch_size,N_QUERIES,D = Q.shape
        N_KEYS = K.shape[1]
        ctx.Q_TILE_SIZE = Q_TILE_SIZE
        ctx.K_TILE_SIZE = K_TILE_SIZE
        ctx.D = D

        O = torch.zeros((batch_size,N_QUERIES,D),device=Q.device,dtype=Q.dtype)
        L = torch.zeros((batch_size,N_QUERIES),device=Q.device,dtype=Q.dtype)

        grid = ((N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE,batch_size)
        scale = 1.0 / math.sqrt(D)
        flash_fwd_kernel[grid](
            Q,K,V,
            O,L,
            Q.stride(0),Q.stride(1),Q.stride(2),
            K.stride(0),K.stride(1),K.stride(2),
            V.stride(0),V.stride(1),V.stride(2),
            O.stride(0),O.stride(1),O.stride(2),
            L.stride(0),L.stride(1),
            N_QUERIES,N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,K_TILE_SIZE,
            is_causal,
        )
        ctx.save_for_backward(L,K,Q,V,O)
        return O
    def backward(ctx,grad_output):
        raise NotImplementedError

