import argparse
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import triton
from einops import einsum

from cs336_systems import FlashAttention_pytorch as flash_pytorch_module
from cs336_systems.FlashAttention_pytorch import FlashAttention_pytorch
from cs336_systems.Triton import FlashAttention as flash_triton_module
from cs336_systems.Triton.FlashAttention import FlashAttention


LOCAL_SEQ_LENS = [2048,4096]
LOCAL_D_MODELS = [64,128]
FULL_SEQ_LENS = [2 ** exp for exp in range(7, 17)]
FULL_D_MODELS = [2 ** exp for exp in range(4, 8)]
DTYPE_CHOICES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

torch.set_float32_matmul_precision("high")


def to_markdown_table(rows):
    columns = list(rows[0].keys())
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    header = "| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |"
    separator = "| " + " | ".join("-" * widths[column] for column in columns) + " |"
    body = [
        "| " + " | ".join(str(row[column]).ljust(widths[column]) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark naive/PyTorch/Triton attention")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup iterations for triton.testing.do_bench")
    parser.add_argument("--rep", type=int, default=1000, help="Measured iterations for triton.testing.do_bench")
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=["naive", "pytorch_flash", "triton_flash"],
        default=["naive", "triton_flash"],
        help="Attention implementations to benchmark",
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=None,
        help="Sequence lengths to benchmark; defaults to the full assignment sweep",
    )
    parser.add_argument(
        "--d-models",
        type=int,
        nargs="+",
        default=None,
        help="Embedding dimensions to benchmark; defaults to the full assignment sweep",
    )
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=sorted(DTYPE_CHOICES.keys()),
        default=["float32", "bfloat16"],
        help="Tensor dtypes to benchmark",
    )
    parser.add_argument(
        "--local-sweep",
        action="store_true",
        help="Use the smaller local sweep: seq_len=[256,1024], d_model=[16,32]",
    )
    parser.add_argument("--q-tile-size", type=int, default=None, help="Override Triton Q tile size")
    parser.add_argument("--k-tile-size", type=int, default=None, help="Override Triton K tile size")
    parser.add_argument("--pytorch-tile-size", type=int, default=None, help="Override PyTorch FlashAttention tile size")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmark inputs")
    parser.add_argument("--n-heads", type=int, default=16, help="Number of attention heads to benchmark")
    parser.add_argument(
        "--plot-output",
        type=str,
        default=None,
        help="Path to save the flash attention comparison plot",
    )
    parser.add_argument(
        "--compile-pytorch-flash",
        action="store_true",
        default=True,
        help="Benchmark the PyTorch FlashAttention implementation through torch.compile",
    )
    parser.add_argument("--causal", action="store_true", default=True, help="Benchmark causal attention")
    return parser.parse_args()


def softmax(in_feature, dim):
    max_num = torch.max(in_feature, dim=dim, keepdim=True).values
    exp_tensor = torch.exp(in_feature - max_num)
    sum_exp = torch.sum(exp_tensor, dim=dim, keepdim=True)
    return exp_tensor / sum_exp


def scaled_dot_product_attention(q, k, v, is_causal):
    d_k = q.shape[-1]
    scores = einsum(q, k, "... q d_k, ... k d_k -> ... q k") / (d_k ** 0.5)
    if is_causal:
        q_idx = torch.arange(q.shape[-2], device=q.device)
        k_idx = torch.arange(k.shape[-2], device=k.device)
        mask = q_idx[:, None] >= k_idx[None, :]
        scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
    attn_weights = softmax(scores, dim=-1)
    return einsum(attn_weights, v, "... q k, ... k d_v -> ... q d_v")


def is_oom_error(exc):
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


@contextmanager
def configured_tile_sizes(args):
    old_pytorch_tile = flash_pytorch_module.TILE_SIZE
    old_q_tile = flash_triton_module.Q_TILE_SIZE
    old_k_tile = flash_triton_module.K_TILE_SIZE
    if args.pytorch_tile_size is not None:
        flash_pytorch_module.TILE_SIZE = args.pytorch_tile_size
    if args.q_tile_size is not None:
        flash_triton_module.Q_TILE_SIZE = args.q_tile_size
    if args.k_tile_size is not None:
        flash_triton_module.K_TILE_SIZE = args.k_tile_size
    try:
        yield
    finally:
        flash_pytorch_module.TILE_SIZE = old_pytorch_tile
        flash_triton_module.Q_TILE_SIZE = old_q_tile
        flash_triton_module.K_TILE_SIZE = old_k_tile


def make_impl(name, compile_pytorch_flash=False):
    if name == "naive":
        return lambda q, k, v, is_causal: scaled_dot_product_attention(q, k, v, is_causal)
    if name == "pytorch_flash":
        impl = lambda q, k, v, is_causal: FlashAttention_pytorch.apply(q, k, v, is_causal)
        if compile_pytorch_flash:
            impl = torch.compile(impl)
        return impl
    if name == "triton_flash":
        return lambda q, k, v, is_causal: FlashAttention.apply(q, k, v, is_causal)
    raise ValueError(f"Unknown implementation: {name}")


def benchmark_case(
    impl_name,
    seq_len,
    d_model,
    dtype,
    warmup,
    rep,
    is_causal,
    compile_pytorch_flash=False,
    batch_size=1,
    n_heads=1,
):
    device = torch.device("cuda")
    impl = make_impl(impl_name, compile_pytorch_flash=compile_pytorch_flash)

    q = torch.randn(batch_size, n_heads, seq_len, d_model, device=device, dtype=dtype)
    k = torch.randn(batch_size, n_heads, seq_len, d_model, device=device, dtype=dtype)
    v = torch.randn(batch_size, n_heads, seq_len, d_model, device=device, dtype=dtype)
    do = torch.randn(batch_size, n_heads, seq_len, d_model, device=device, dtype=dtype)

    q = q.reshape(batch_size * n_heads, seq_len, d_model)
    k = k.reshape(batch_size * n_heads, seq_len, d_model)
    v = v.reshape(batch_size * n_heads, seq_len, d_model)
    do = do.reshape(batch_size * n_heads, seq_len, d_model)

    def fwd_only():
        out = impl(q, k, v, is_causal)
        torch.cuda.synchronize()
        return out

    def fwd_bwd():
        q_local = q.detach().clone().requires_grad_(True)
        k_local = k.detach().clone().requires_grad_(True)
        v_local = v.detach().clone().requires_grad_(True)
        out = impl(q_local, k_local, v_local, is_causal)
        out.backward(do)
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    fwd_ms = triton.testing.do_bench(fwd_only, warmup=warmup, rep=rep)
    fwd_peak_mem_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    bwd_ms = triton.testing.do_bench(fwd_bwd, warmup=warmup, rep=rep)
    bwd_peak_mem_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    return {
        "implementation": impl_name,
        "compiled": impl_name == "pytorch_flash" and compile_pytorch_flash,
        "batch_size": batch_size,
        "n_heads": n_heads,
        "seq_len": seq_len,
        "d_head": d_model,
        "d_model": n_heads * d_model,
        "dtype": str(dtype).replace("torch.", ""),
        "fwd_ms": f"{float(fwd_ms):.3f}",
        "bwd_ms": f"{float(bwd_ms):.3f}",
        "fwd_peak_mem_mib": f"{fwd_peak_mem_mib:.2f}",
        "bwd_peak_mem_mib": f"{bwd_peak_mem_mib:.2f}",
        "status": "ok",
    }


def make_plot(results, output_path):
    plot_rows = []
    for row in results:
        if row["implementation"] not in {"naive", "triton_flash"}:
            continue
        if row["status"] != "ok":
            continue
        plot_rows.append(
            {
                **row,
                "fwd_ms_value": float(row["fwd_ms"]),
                "bwd_ms_value": float(row["bwd_ms"]),
                "total_ms": float(row["fwd_ms"]) + float(row["bwd_ms"]),
            }
        )

    if not plot_rows:
        print("\nNo torch/triton flash results available for plotting.")
        return

    dtypes = sorted({row["dtype"] for row in plot_rows})
    shapes = sorted({(row["n_heads"], row["d_head"]) for row in plot_rows})
    n_rows = len(dtypes)
    n_cols = len(shapes)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )
    colors = {
        "naive": "#d95f02",
        "triton_flash": "#1b9e77",
    }
    labels = {
        "naive": "PyTorch Attention",
        "triton_flash": "Triton FlashAttention",
    }

    for row_idx, dtype in enumerate(dtypes):
        for col_idx, (n_heads, d_head) in enumerate(shapes):
            ax = axes[row_idx][col_idx]
            subset = [
                row
                for row in plot_rows
                if row["dtype"] == dtype and row["n_heads"] == n_heads and row["d_head"] == d_head
            ]
            for impl_name in ("naive", "triton_flash"):
                impl_rows = sorted(
                    [row for row in subset if row["implementation"] == impl_name],
                    key=lambda row: row["seq_len"],
                )
                if not impl_rows:
                    continue
                ax.plot(
                    [row["seq_len"] for row in impl_rows],
                    [row["total_ms"] for row in impl_rows],
                    marker="o",
                    linewidth=2,
                    color=colors[impl_name],
                    label=labels[impl_name],
                )
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Sequence Length")
            ax.set_ylabel("Forward + Backward Time (ms)")
            ax.set_title(f"dtype={dtype}, n_heads={n_heads}, d_head={d_head}")
            ax.grid(True, alpha=0.3)

    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=len(legend_labels), frameon=False)
    fig.suptitle("FlashAttention Benchmark: PyTorch vs Triton", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot to {output_path}")


def main():
    args = parse_args()
    if args.seq_lens is not None:
        seq_lens = args.seq_lens
    elif args.local_sweep:
        seq_lens = LOCAL_SEQ_LENS
    else:
        seq_lens = FULL_SEQ_LENS

    if args.d_models is not None:
        d_models = args.d_models
    elif args.local_sweep:
        d_models = LOCAL_D_MODELS
    else:
        d_models = FULL_D_MODELS

    dtypes = [DTYPE_CHOICES[name] for name in args.dtypes]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    results = []
    with configured_tile_sizes(args):
        for impl_name in args.implementations:
            for seq_len in seq_lens:
                for d_model in d_models:
                    for dtype in dtypes:
                        print(
                            f"Benchmarking impl={impl_name} seq_len={seq_len} d_head={d_model} "
                            f"dtype={dtype} batch_size={args.batch_size} n_heads={args.n_heads}"
                        )
                        try:
                            results.append(
                                benchmark_case(
                                    impl_name=impl_name,
                                    seq_len=seq_len,
                                    d_model=d_model,
                                    dtype=dtype,
                                    warmup=args.warmup,
                                    rep=args.rep,
                                    is_causal=args.causal,
                                    compile_pytorch_flash=args.compile_pytorch_flash,
                                    batch_size=args.batch_size,
                                    n_heads=args.n_heads,
                                )
                            )
                        except Exception as exc:
                            if is_oom_error(exc):
                                print(
                                    f"OOM for impl={impl_name} seq_len={seq_len} d_head={d_model} "
                                    f"dtype={dtype} batch_size={args.batch_size} n_heads={args.n_heads}"
                                )
                                results.append(
                                    {
                                        "implementation": impl_name,
                                        "compiled": impl_name == "pytorch_flash" and args.compile_pytorch_flash,
                                        "batch_size": args.batch_size,
                                        "n_heads": args.n_heads,
                                        "seq_len": seq_len,
                                        "d_head": d_model,
                                        "d_model": args.n_heads * d_model,
                                        "dtype": str(dtype).replace("torch.", ""),
                                        "fwd_ms": "",
                                        "bwd_ms": "",
                                        "fwd_peak_mem_mib": "",
                                        "bwd_peak_mem_mib": "",
                                        "status": "oom",
                                    }
                                )
                                torch.cuda.empty_cache()
                            else:
                                raise

    if not results:
        raise RuntimeError("No benchmark points completed successfully.")

    print("\nBenchmark Results:")
    print(to_markdown_table(results))
    plot_output = Path(args.plot_output) if args.plot_output else Path(__file__).with_name("flash_attention_benchmark.png")
    make_plot(results, plot_output)


if __name__ == "__main__":
    main()
