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
            S = tl.where(causal_mask,float("-inf"),S)
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

@triton.jit
def flash_bwd_kernel_phase1(
    Q_ptr,K_ptr,V_ptr,L_ptr,O_ptr,
    dO_ptr,dQ_ptr,
    stride_qb,stride_qq,stride_qd,
    stride_kb,stride_kk,stride_kd,
    stride_vb,stride_vk,stride_vd,
    stride_lb,stride_lq,
    stride_ob,stride_oq,stride_od,
    stride_dOb,stride_dOq,stride_dOd,
    N_QUERIES,N_KEYS,
    scale,
    D:tl.constexpr,
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
    dO_block_ptr = tl.make_block_ptr(
        base = dO_ptr + batch_index * stride_dOb,
        shape = (N_QUERIES,D),
        strides = (stride_dOq,stride_dOd),
        offsets = (query_tile_index * Q_TILE_SIZE,0),
        block_shape = (Q_TILE_SIZE,D),
        order = (1,0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        base = dQ_ptr + batch_index * stride_qb,
        shape = (N_QUERIES,D),
        strides = (stride_qq,stride_qd),
        offsets = (query_tile_index * Q_TILE_SIZE,0),
        block_shape = (Q_TILE_SIZE,D),
        order = (1,0)
    )
    B_dO = tl.load(dO_block_ptr,boundary_check=(0,1),padding_option="zero")
    O = tl.load(O_block_ptr,boundary_check=(0,1),padding_option="zero")
    D0 = tl.sum(B_dO * O,axis=1)
    B_q = tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")
    B_l = tl.load(L_block_ptr,boundary_check=(0,),padding_option="zero")
    dQ = tl.zeros((Q_TILE_SIZE,D),dtype=tl.float32)
    for j in range(tl.cdiv(N_KEYS,K_TILE_SIZE)):
        B_k = tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")
        B_v = tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero")
        S_ij = tl.dot(B_q,B_k.T) * scale
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0,Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0,K_TILE_SIZE)
            causal_mask = q_idx[:,None] < k_idx[None,:]
            S_ij = tl.where(causal_mask,float("-inf"),S_ij)
        P_ij = tl.exp(S_ij - B_l[:,None])
        dP_ij = tl.dot(B_dO,B_v.T)
        dS_ij = P_ij * (dP_ij - D0[:,None])
        dQ += tl.dot(dS_ij,B_k) * scale
        K_block_ptr = tl.advance(K_block_ptr,(K_TILE_SIZE,0))
        V_block_ptr = tl.advance(V_block_ptr,(K_TILE_SIZE,0))
    tl.store(dQ_block_ptr,dQ.to(dQ_block_ptr.type.element_ty),boundary_check=(0,1))

@triton.jit
def flash_bwd_kernel_phase2(
    Q_ptr,K_ptr,V_ptr,L_ptr,O_ptr,
    dO_ptr,dK_ptr,dV_ptr,
    stride_qb,stride_qq,stride_qd,
    stride_kb,stride_kk,stride_kd,
    stride_vb,stride_vk,stride_vd,
    stride_lb,stride_lq,
    stride_ob,stride_oq,stride_od,
    stride_dOb,stride_dOq,stride_dOd,
    N_QUERIES,N_KEYS,
    scale,
    D:tl.constexpr,
    Q_TILE_SIZE:tl.constexpr,
    K_TILE_SIZE:tl.constexpr,
    is_causal:tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        base = Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES,D),
        strides = (stride_qq,stride_qd),
        offsets = (0,0),
        block_shape = (Q_TILE_SIZE,D),
        order = (1,0),
    )
    K_block_ptr = tl.make_block_ptr(
        base = K_ptr + batch_index * stride_kb,
        shape = (N_KEYS,D),
        strides = (stride_kk,stride_kd),
        offsets = (key_tile_index * K_TILE_SIZE,0),
        block_shape = (K_TILE_SIZE,D),
        order = (1,0),
    )
    V_block_ptr = tl.make_block_ptr(
        base = V_ptr + batch_index * stride_vb,
        shape = (N_KEYS,D),
        strides = (stride_vk,stride_vd),
        offsets = (key_tile_index * K_TILE_SIZE,0),
        block_shape = (K_TILE_SIZE,D),
        order =(1,0),
    )
    L_block_ptr = tl.make_block_ptr(
        base = L_ptr + batch_index * stride_lb,
        shape = (N_QUERIES,),
        strides = (stride_lq,),
        offsets = (0,),
        block_shape = (Q_TILE_SIZE,),
        order = (0,),
    )
    O_block_ptr = tl.make_block_ptr(
        base = O_ptr + batch_index * stride_ob,
        shape = (N_QUERIES,D),
        strides = (stride_oq,stride_od),
        offsets = (0,0),
        block_shape = (Q_TILE_SIZE,D),
        order = (1,0),
    )
    dO_block_ptr = tl.make_block_ptr(
        base = dO_ptr + batch_index * stride_dOb,
        shape = (N_QUERIES,D),
        strides = (stride_dOq,stride_dOd),
        offsets = (0,0),
        block_shape = (Q_TILE_SIZE,D),
        order = (1,0),
    )
    dK_block_ptr = tl.make_block_ptr(
        base = dK_ptr + batch_index * stride_kb,
        shape = (N_KEYS,D),
        strides = (stride_kk,stride_kd),
        offsets = (key_tile_index * K_TILE_SIZE,0),
        block_shape = (K_TILE_SIZE,D),
        order = (1,0)
    )
    dV_block_ptr = tl.make_block_ptr(
        base = dV_ptr + batch_index * stride_vb,
        shape = (N_KEYS,D),
        strides = (stride_vk,stride_vd),
        offsets = (key_tile_index * K_TILE_SIZE,0),
        block_shape = (K_TILE_SIZE,D),
        order = (1,0)
    )
    B_k = tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")
    B_v = tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero")
    dK = tl.zeros((K_TILE_SIZE,D),dtype=tl.float32)
    dV = tl.zeros((K_TILE_SIZE,D),dtype=tl.float32)
    for i in range(tl.cdiv(N_QUERIES,Q_TILE_SIZE)):
        B_dO = tl.load(dO_block_ptr,boundary_check=(0,1),padding_option="zero")
        B_O = tl.load(O_block_ptr,boundary_check=(0,1),padding_option="zero")
        B_q = tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")
        B_l = tl.load(L_block_ptr,boundary_check=(0,),padding_option="zero")
        D0 = tl.sum(B_dO * B_O,axis=1)
        S_ij = tl.dot(B_q,B_k.T) * scale
        if is_causal:
            q_idx = i * Q_TILE_SIZE + tl.arange(0,Q_TILE_SIZE)
            k_idx = key_tile_index * K_TILE_SIZE + tl.arange(0,K_TILE_SIZE)
            causal_mask = q_idx[:,None] < k_idx[None,:]
            S_ij = tl.where(causal_mask,float("-inf"),S_ij)
        P_ij = tl.exp(S_ij - B_l[:,None])
        dV += tl.dot(P_ij.T,B_dO)
        dP_ij = tl.dot(B_dO,B_v.T)
        dS_ij = P_ij * (dP_ij - D0[:,None])
        dK += tl.dot(dS_ij.T,B_q) * scale
        dO_block_ptr = tl.advance(dO_block_ptr,(Q_TILE_SIZE,0))
        O_block_ptr = tl.advance(O_block_ptr,(Q_TILE_SIZE,0))
        Q_block_ptr = tl.advance(Q_block_ptr,(Q_TILE_SIZE,0))
        L_block_ptr = tl.advance(L_block_ptr,(Q_TILE_SIZE,))
    tl.store(dK_block_ptr,dK.to(dK_block_ptr.type.element_ty),boundary_check = (0,1))
    tl.store(dV_block_ptr,dV.to(dV_block_ptr.type.element_ty),boundary_check = (0,1))

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,Q,K,V,is_causal=False):
        batch_size,N_QUERIES,D = Q.shape
        N_KEYS = K.shape[1]
        ctx.is_causal = is_causal
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
        L,K,Q,V,O = ctx.saved_tensors
        is_causal = ctx.is_causal
        Q_TILE_SIZE = ctx.Q_TILE_SIZE
        K_TILE_SIZE = ctx.K_TILE_SIZE
        D = ctx.D
        batch_size,N_QUERIES,_ = Q.shape
        _,N_KEYS,_ = K.shape
        dQ = torch.zeros((batch_size,N_QUERIES,D),device=Q.device,dtype=Q.dtype)
        dK = torch.zeros((batch_size,N_KEYS,D),device=K.device,dtype=Q.dtype)
        dV = torch.zeros((batch_size,N_KEYS,D),device=V.device,dtype=V.dtype)
        grid = ((N_QUERIES + Q_TILE_SIZE - 1) // Q_TILE_SIZE,batch_size)
        scale = 1.0 / (D ** 0.5)
        flash_bwd_kernel_phase1[grid](
            Q,K,V,L,O,
            grad_output,dQ,
            Q.stride(0),Q.stride(1),Q.stride(2),
            K.stride(0),K.stride(1),K.stride(2),
            V.stride(0),V.stride(1),V.stride(2),
            L.stride(0),L.stride(1),
            O.stride(0),O.stride(1),O.stride(2),
            grad_output.stride(0),grad_output.stride(1),grad_output.stride(2),
            N_QUERIES,N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,K_TILE_SIZE,
            is_causal
        )    
        grid = ((N_KEYS + K_TILE_SIZE - 1) // K_TILE_SIZE,batch_size)
        flash_bwd_kernel_phase2[grid](
            Q,K,V,L,O,
            grad_output,dK,dV,
            Q.stride(0),Q.stride(1),Q.stride(2),
            K.stride(0),K.stride(1),K.stride(2),
            V.stride(0),V.stride(1),V.stride(2),
            L.stride(0),L.stride(1),
            O.stride(0),O.stride(1),O.stride(2),
            grad_output.stride(0),grad_output.stride(1),grad_output.stride(2),
            N_QUERIES,N_KEYS,
            scale,
            D,
            Q_TILE_SIZE,K_TILE_SIZE,
            is_causal
        )    
        return dQ,dK,dV,None


