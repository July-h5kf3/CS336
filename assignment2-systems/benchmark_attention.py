from einops import rearrange, einsum
from cs336_basics.Linear import Linear
from cs336_basics.RotaryPositionalEmbedding import RotaryPositionalEmbedding

import torch
import torch.nn as nn
import pandas as pd

import argparse

def parse_args():
    parse_args = argparse.ArgumentParser(description="Attention Benchmark")
    parse_args.add_argument("--warmup_steps",type=int,default=10)
    parse_args.add_argument("--steps",type=int,default=100)
    parse_args.add_argument("--jit",action="store_true",help="Whether to use torch.compile for attention benchmark")
    return parse_args.parse_args()

def softmax(in_feature,dim):
    max_num = torch.max(in_feature,dim=dim,keepdim=True).values
    exp_tensor = torch.exp(in_feature - max_num)
    sum_exp = torch.sum(exp_tensor,dim=dim,keepdim=True)
    return exp_tensor / sum_exp

def scaled_dot_product_attention(Q,K,V,mask=None):
    d_k = Q.shape[-1]
    scores = einsum(Q,K,"... q d_k,... k d_k -> ... q k") / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(~mask,float("-inf"))
    attn_weights = softmax(scores,dim=-1)
    return einsum(attn_weights,V,"... q k,... k d_v -> ... q d_v")

def is_oom_error(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )

class CausalSelfAttention(nn.Module):
    def __init__(self,d_model,seq_len,device,use_rope=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model
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
        batch_size,seq_len,_ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        if self.use_rope:
            Q = self.RoPE(Q,token_positions)
            K = self.RoPE(K,token_positions)

        mask = torch.tril(torch.ones((seq_len,seq_len),device=self.device)).bool()
        attn_output = scaled_dot_product_attention(Q,K,V,mask=mask)
        out = self.W_o(attn_output)
        return out
def benchmark_attention(seq_len,d_model,warm_up_step,pass_step,batch_size=8,jit=False):
    device = torch.device("cuda")
    result = {
        "batch_size": batch_size,
        "d_model": d_model,
        "seq_len": seq_len,
        "fwd_ms": "",
        "bwd_ms": "",
        "fwd_peak_mem_mib": "",
        "bwd_mem_before_backward_mib": "",
        "bwd_peak_mem_mib": "",
        "status": "ok",
    }

    attention = CausalSelfAttention(d_model=d_model,seq_len=seq_len,device=device).to(device)
    if jit:
        attention = torch.compile(attention)
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    token_positions = torch.arange(seq_len,device=device)
    token_positions = rearrange(token_positions,"k -> 1 k").expand(batch_size, -1)

    for _ in range(warm_up_step):
        _ = attention(x,token_positions)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    fwd_total_time = 0.0
    #FWD test
    for _ in range(pass_step):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        _ = attention(x,token_positions)
        end_time.record()
        torch.cuda.synchronize()
        fwd_total_time += start_time.elapsed_time(end_time)

    avg_fwd_time = fwd_total_time / pass_step
    result["fwd_ms"] = f"{avg_fwd_time:.2f}"
    result["fwd_peak_mem_mib"] = f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f}"

    #BWD test
    x = x.detach().requires_grad_(True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    bwd_total_time = 0.0
    bwd_mem_before_backward = 0.0
    for _ in range(pass_step):
        attention.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        out = attention(x,token_positions)
        loss = out.sum()
        bwd_mem_before_backward += torch.cuda.memory_allocated(device)
        loss.backward()
        end_time.record()
        torch.cuda.synchronize()
        bwd_total_time += start_time.elapsed_time(end_time)

    avg_bwd_time = bwd_total_time / pass_step
    result["bwd_ms"] = f"{avg_bwd_time:.2f}"
    result["bwd_mem_before_backward_mib"] = f"{(bwd_mem_before_backward / pass_step) / (1024 ** 2):.2f}"
    result["bwd_peak_mem_mib"] = f"{torch.cuda.max_memory_allocated(device) / (1024 ** 2):.2f}"
    return result

def main():
    args = parse_args()
    d_models = [16,32]
    seq_lens = [256,1024]
    # d_models = [16,32,64,128]
    # seq_lens = [256,1024,4096,8192,16384]
    results = []
    for d_model in d_models:
        for seq_len in seq_lens:
            print(f"Benchmarking attention with d_model={d_model} and seq_len={seq_len}")
            try:
                result = benchmark_attention(
                    batch_size=8,
                    seq_len=seq_len,
                    d_model=d_model,
                    warm_up_step=args.warmup_steps,
                    pass_step=args.steps,
                    jit=args.jit
                )
                results.append(result)
            except Exception as exc:
                if is_oom_error(exc):
                    print(f"OOM at d_model={d_model}, seq_len={seq_len}")
                    results.append({
                        "batch_size": 8,
                        "d_model": d_model,
                        "seq_len": seq_len,
                        "fwd_ms": "",
                        "bwd_ms": "",
                        "fwd_peak_mem_mib": "",
                        "bwd_mem_before_backward_mib": "",
                        "bwd_peak_mem_mib": "",
                        "status": "oom",
                    })
                    torch.cuda.empty_cache()
                else:
                    raise

    print("\nBenchmark Results:")
    print(pd.DataFrame(results).to_markdown(index=False))

if __name__ == "__main__":    
    main()
