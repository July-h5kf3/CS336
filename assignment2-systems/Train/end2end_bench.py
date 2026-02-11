import timeit
import argparse
import yaml
import torch
import numpy as np
import pandas as pd

from cs336_basics.Transformer import Transformer
from cs336_basics.Cross_entropy import cross_entropy
from cs336_basics.Adamw import Adamw

def parse_args():
    parse_args = argparse.ArgumentParser(description="End-to-end Transformer training")
    parse_args.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parse_args.add_argument("--warmup_steps",type=int,default=10)
    parse_args.add_argument("--steps",type=int,default=100)
    return parse_args.parse_args()
def load_config(config_path):
    with open(config_path,"r") as f:
        return yaml.safe_load(f)

def get_random_batch(batch_size,context_length,vocab_size,device):
    x = torch.randint(0,vocab_size,(batch_size,context_length),device=device)
    y = torch.randint(0,vocab_size,(batch_size,context_length),device=device)
    return x,y

def main():
    args = parse_args()
    config = load_config(args.config)
    model_config = config['model']
    device = config['training']['device']
    assert device == "cuda","Please run bench in cuda!"
    warmup_steps = args.warmup_steps
    steps = args.steps

    model_configs = {
        "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
        "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
        "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    }

    results = []

    for size_name, params in model_configs.items():
        print(f"Benchmarking {size_name} model...")
        
        current_model_config = model_config.copy()
        current_model_config.update(params)
        
        model = Transformer(
            vocab_size=current_model_config['vocab_size'],
            context_length=current_model_config['context_length'],
            num_layers=current_model_config['num_layers'],
            d_model=current_model_config['d_model'],
            num_heads=current_model_config['num_heads'],
            device=device
        )
        model.to(device)

        x, y = get_random_batch(config['training']['batch_size'], current_model_config['context_length'], current_model_config['vocab_size'], device)

        for _ in range(warmup_steps):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        for _ in range(steps):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        fwd_time = (timeit.default_timer() - start_time) / steps

        optimizer = Adamw(
            model.parameters(),
            lr=float(config['training']['max_lr']),
            weight_decay=float(config['training']['weight_decay'])
        )
        
        for _ in range(warmup_steps):
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits.view(-1, current_model_config['vocab_size']), y.view(-1))
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        start_time = timeit.default_timer()
        for _ in range(steps):
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits.view(-1, current_model_config['vocab_size']), y.view(-1))
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        fwd_bwd_time = (timeit.default_timer() - start_time) / steps

        results.append({
            "Size": size_name,
            "d_model": params["d_model"],
            "d_ff": params["d_ff"],
            "num_layers": params["num_layers"],
            "num_heads": params["num_heads"],
            "Forward (s)": f"{fwd_time:.6f}",
            "Forward+Backward (s)": f"{fwd_bwd_time:.6f}"
        })
        
        # Clean up to avoid OOM
        del model
        del optimizer
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print("\nBenchmark Results:")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()