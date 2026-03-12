import timeit
import argparse
import yaml
import torch
import pandas as pd
import math

from cs336_basics.Transformer import Transformer
from cs336_basics.Cross_entropy import cross_entropy
from cs336_basics.Adamw import Adamw
import cs336_basics.CausalMultiHeadSelfAttention as attention_module

"""
Profile 相关
"""
from contextlib import contextmanager
import torch.cuda.nvtx as nvtx

@contextmanager
def nvtx_range(name):
    nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()

def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    with nvtx_range("ATTN_QK_MATMUL"):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    with nvtx_range("ATTN_SOFTMAX"):
        attn_weights = torch.softmax(scores, dim=-1)
    with nvtx_range("ATTN_AV_MATMUL"):
        return torch.matmul(attn_weights, V)

def install_annotated_attention():
    attention_module.run_scaled_dot_product_attention = annotated_scaled_dot_product_attention

def parse_args():
    parse_args = argparse.ArgumentParser(description="End-to-end Transformer training")
    parse_args.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parse_args.add_argument("--warmup_steps",type=int,default=10)
    parse_args.add_argument("--steps",type=int,default=100)
    parse_args.add_argument("--mix_precision",action="store_true",help="Whether to use mixed precision for training benchmark")
    parse_args.add_argument("--jit",action="store_true",help="Whether to use torch.compile for training benchmark")
    return parse_args.parse_args()
def load_config(config_path):
    with open(config_path,"r") as f:
        return yaml.safe_load(f)

def get_random_batch(batch_size,context_length,vocab_size,device):
    x = torch.randint(0,vocab_size,(batch_size,context_length),device=device)
    y = torch.randint(0,vocab_size,(batch_size,context_length),device=device)
    return x,y

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(num_params):
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.3f}B"
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.3f}M"
    return str(num_params)

def build_model(model_config, device, jit=False):
    model = Transformer(
        vocab_size=model_config['vocab_size'],
        context_length=model_config['context_length'],
        num_layers=model_config['num_layers'],
        d_model=model_config['d_model'],
        num_heads=model_config['num_heads'],
        device=device
    )
    if jit:
        model = torch.compile(model)
    return model.to(device)

def benchmark_forward(model, x, warmup_steps, steps,mix_precision=False):
    model.eval()
    for _ in range(warmup_steps):
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()

    start_time = timeit.default_timer()
    if mix_precision:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # torch.cuda.memory._record_memory_history(max_entries=1000000)
            for _ in range(steps):
                with torch.no_grad():
                    with nvtx_range("FWD_STEP"):
                        _ = model(x)
                torch.cuda.synchronize()
            # torch.cuda.memory._dump_snapshot("memory_FWD_STEP_MIX.pickle")
            # torch.cuda.memory._record_memory_history(enabled=None)
    else:
        # torch.cuda.memory._record_memory_history(max_entries=1000000)
        for _ in range(steps):
            with torch.no_grad():
                with nvtx_range("FWD_STEP"):
                    _ = model(x)
            torch.cuda.synchronize()
        # torch.cuda.memory._dump_snapshot("memory_FWD_STEP.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)
    return (timeit.default_timer() - start_time) / steps

def benchmark_train(model, x, y, vocab_size, optimizer, warmup_steps, steps,mix_precision=False):
    model.train()
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

    start_time = timeit.default_timer()
    if mix_precision:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # torch.cuda.memory._record_memory_history(max_entries=1000000)
            for _ in range(steps):
                optimizer.zero_grad()
                with nvtx_range("FWD_STEP"):
                    logits = model(x)
                loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
                with nvtx_range("BWD_STEP"):
                    loss.backward()
                with nvtx_range("OPTIM_STEP"):
                    optimizer.step()
                torch.cuda.synchronize()
            # torch.cuda.memory._dump_snapshot("memory_TRAIN_STEP_MIX.pickle")
            # torch.cuda.memory._record_memory_history(enabled=None)
    else:
        # torch.cuda.memory._record_memory_history(max_entries=1000000)
        for _ in range(steps):
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            with nvtx_range("BWD_STEP"):
                loss.backward()
            with nvtx_range("OPTIM_STEP"):
                optimizer.step()
            torch.cuda.synchronize()
        # torch.cuda.memory._dump_snapshot("memory_TRAIN_STEP.pickle")
        # torch.cuda.memory._record_memory_history(enabled=None)
    return (timeit.default_timer() - start_time) / steps

def empty_result_row(size_name, params):
    return {
        "Size": size_name,
        "d_model": params["d_model"],
        "d_ff": params["d_ff"],
        "num_layers": params["num_layers"],
        "num_heads": params["num_heads"],
        "Parameters": "",
        "Forward (s)": "",
        "Forward+Backward (s)": "",
    }

def is_oom_error(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )

def main():
    args = parse_args()
    config = load_config(args.config)
    install_annotated_attention()
    model_config = config['model']
    device = config['training']['device']
    assert device == "cuda","Please run bench in cuda!"
    warmup_steps = args.warmup_steps
    steps = args.steps

    model_configs = {
        # "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        # "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
        # "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
        # "7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    }

    results = []
    skip_inference_remaining = False

    for size_name, params in model_configs.items():
        row = empty_result_row(size_name, params)
        if skip_inference_remaining:
            print(f"Skipping {size_name} because a smaller model hit inference OOM.")
            results.append(row)
            continue

        print(f"Benchmarking {size_name} model...")
        current_model_config = model_config.copy()
        current_model_config.update(params)
        x = None
        y = None
        model = None
        optimizer = None

        # Inference stage: OOM here means this and larger models are all skipped.
        try:
            x, y = get_random_batch(
                config['training']['batch_size'],
                current_model_config['context_length'],
                current_model_config['vocab_size'],
                device
            )
            model = build_model(current_model_config, device, args.jit)
            row["Parameters"] = format_params(count_parameters(model))
            fwd_time = benchmark_forward(model, x, warmup_steps, steps,args.mix_precision)
            row["Forward (s)"] = f"{fwd_time:.6f}"
        except Exception as exc:
            if is_oom_error(exc):
                print(f"Inference OOM on {size_name}; skipping this and larger models.")
                skip_inference_remaining = True
                results.append(row)
                continue
            raise
        finally:
            if model is not None:
                del model
                model = None
            if optimizer is not None:
                del optimizer
                optimizer = None
            torch.cuda.empty_cache()

        # Training stage: OOM here only affects current model's training metrics.
        try:
            model = build_model(current_model_config, device)
            optimizer = Adamw(
                model.parameters(),
                lr=float(config['training']['max_lr']),
                weight_decay=float(config['training']['weight_decay'])
            )
            fwd_bwd_time = benchmark_train(
                model,
                x,
                y,
                current_model_config['vocab_size'],
                optimizer,
                warmup_steps,
                steps,
                args.mix_precision
            )
            row["Forward+Backward (s)"] = f"{fwd_bwd_time:.6f}"
        except Exception as exc:
            if is_oom_error(exc):
                print(
                    f"Training OOM on {size_name}; keeping inference result and continuing to larger-model inference."
                )
            else:
                raise
        finally:
            if model is not None:
                del model
                model = None
            if optimizer is not None:
                del optimizer
                optimizer = None
            torch.cuda.empty_cache()
        results.append(row)

    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()
