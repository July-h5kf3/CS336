import argparse
import os
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


BYTES_PER_MB = 1024 * 1024
WORLD_SIZES = [2, 4, 6, 8]
TENSOR_SIZES_MB = [1, 10, 100, 1024]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["gloo-cpu", "nccl-cuda"])
    parser.add_argument("--world-sizes", nargs="+", type=int, default=WORLD_SIZES)
    parser.add_argument("--tensor-sizes-mb", nargs="+", type=int, default=TENSOR_SIZES_MB)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--master-addr", type=str, default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument("--plot-output", type=Path, default=None)
    return parser.parse_args()


def parse_target(target):
    backend, device = target.split("-")
    return backend, device


def numel_from_size(size_mb):
    size_bytes = size_mb * BYTES_PER_MB
    return size_bytes // 4


def tensor_size_label(size_mb):
    if size_mb >= 1024:
        return f"{size_mb / 1024:.0f}GB"
    return f"{size_mb}MB"


def format_table(rows):
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


def setup(rank, world_size, backend, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def benchmark_worker(rank, config, queue):
    backend = config["backend"]
    device_type = config["device"]
    world_size = config["world_size"]

    if device_type == "cuda":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")

    setup(rank, world_size, backend, config["master_addr"], config["master_port"])

    tensor = torch.ones(numel_from_size(config["tensor_size_mb"]), dtype=torch.float32, device=device)

    # Warm up collective communication before starting measured iterations.
    for _ in range(config["warmup_iters"]):
        dist.barrier()
        dist.all_reduce(tensor)
        tensor.div_(world_size)

    dist.barrier()
    if device_type == "cuda":
        torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(config["iters"]):
        dist.barrier()
        if device_type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(tensor)
        if device_type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000)
        tensor.div_(world_size)

    if rank == 0:
        median_ms = statistics.median(latencies_ms)
        size_gib = config["tensor_size_mb"] / 1024
        algo_gibps = size_gib * (2 * (world_size - 1) / world_size) / (median_ms / 1000)
        queue.put(
            {
                "target": config["target"],
                "world_size": world_size,
                "tensor_size_mb": config["tensor_size_mb"],
                "tensor_size": tensor_size_label(config["tensor_size_mb"]),
                "median_ms": median_ms,
                "mean_ms": statistics.mean(latencies_ms),
                "algo_gibps": algo_gibps,
            }
        )

    cleanup()


def run_case(case_id, target, world_size, tensor_size_mb, args):
    backend, device = parse_target(target)
    config = {
        "target": target,
        "backend": backend,
        "device": device,
        "world_size": world_size,
        "tensor_size_mb": tensor_size_mb,
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "master_addr": args.master_addr,
        "master_port": args.master_port + case_id,
    }
    queue = mp.SimpleQueue()
    mp.spawn(benchmark_worker, args=(config, queue), nprocs=world_size, join=True)
    return queue.get()


def markdown_rows(results):
    rows = []
    for row in results:
        rows.append(
            {
                "target": row["target"],
                "world_size": row["world_size"],
                "tensor_size": row["tensor_size"],
                "median_ms": f"{row['median_ms']:.3f}",
                "mean_ms": f"{row['mean_ms']:.3f}",
                "algo_GiBps": f"{row['algo_gibps']:.3f}",
            }
        )
    return rows


def make_report(results, args):
    lines = [
        "# Distributed Benchmark",
        "",
        f"- targets: {', '.join(args.targets)}",
        f"- world_sizes: {', '.join(str(x) for x in args.world_sizes)}",
        f"- tensor_sizes_mb: {', '.join(str(x) for x in args.tensor_sizes_mb)}",
        "",
        format_table(markdown_rows(results)),
    ]
    if args.plot_output is not None:
        lines.extend(["", f"![plot]({args.plot_output.name})"])
    return "\n".join(lines)


def save_plot(results, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for target in sorted({row["target"] for row in results}):
        for world_size in sorted({row["world_size"] for row in results}):
            rows = [
                row for row in results
                if row["target"] == target and row["world_size"] == world_size
            ]
            if not rows:
                continue
            rows.sort(key=lambda row: row["tensor_size_mb"])
            label = f"{target}, p={world_size}"
            x = [row["tensor_size_mb"] for row in rows]
            axes[0].plot(x, [row["median_ms"] for row in rows], marker="o", label=label)
            axes[1].plot(x, [row["algo_gibps"] for row in rows], marker="o", label=label)

    axes[0].set_xscale("log", base=10)
    axes[1].set_xscale("log", base=10)
    axes[0].set_title("Median Latency")
    axes[1].set_title("Algorithmic Bandwidth")
    axes[0].set_xlabel("Tensor Size (MB)")
    axes[1].set_xlabel("Tensor Size (MB)")
    axes[0].set_ylabel("ms")
    axes[1].set_ylabel("GiB/s")
    axes[0].grid(True)
    axes[1].grid(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)

    results = []
    case_id = 0
    for target in args.targets:
        for world_size in args.world_sizes:
            for tensor_size_mb in args.tensor_sizes_mb:
                results.append(run_case(case_id, target, world_size, tensor_size_mb, args))
                case_id += 1

    report = make_report(results, args)
    print(report)

    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(report + "\n", encoding="utf-8")

    if args.plot_output is not None:
        args.plot_output.parent.mkdir(parents=True, exist_ok=True)
        save_plot(results, args.plot_output)


if __name__ == "__main__":
    main()
