import triton
import triton.language as tl

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


@triton.jit
def weighted_sum_fwd(
    x_ptr,
    weight_ptr,
    output_ptr,
    x_stride_row,
    x_stride_dim,
    weight_stride_dim,
    output_stride_dim,
    ROWS,
    D,
    ROW_TILE_SIZE: tl.constexpr,
    DIM_TILE_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(ROWS, D),
        strides=(x_stride_row, x_stride_dim),
        offsets=(row_idx * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, DIM_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),
        offsets=(0,),
        block_shape=(DIM_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(ROWS,),
        strides=(output_stride_dim,),
        offsets=(row_idx * ROW_TILE_SIZE,),
        block_shape=(ROW_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROW_TILE_SIZE,), dtype=tl.float32)
    for _ in range(tl.cdiv(D, DIM_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        output += tl.sum(row * weight[None, :], axis=1)
        x_block_ptr = tl.advance(x_block_ptr, (0, DIM_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (DIM_TILE_SIZE,))
    tl.store(output_block_ptr, output, boundary_check=(0,))


@triton.jit
def weighted_sum_bwd(
    x_ptr,
    weight_ptr,
    grad_output_ptr,
    grad_x_ptr,
    partial_grad_weight_ptr,
    stride_xr,
    stride_xd,
    stride_wd,
    stride_gr,
    stride_gxr,
    stride_gxd,
    stride_gwb,
    stride_gwd,
    NUM_ROWS,
    D,
    ROW_TILE_SIZE: tl.constexpr,
    DIM_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(
        base=grad_output_ptr,
        shape=(NUM_ROWS,),
        strides=(stride_gr,),
        offsets=(row_tile_idx * ROW_TILE_SIZE,),
        block_shape=(ROW_TILE_SIZE,),
        order=(0,),
    )
    x_block_ptr = tl.make_block_ptr(
        base=x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_xr, stride_xd),
        offsets=(row_tile_idx * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, DIM_TILE_SIZE),
        order=(1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(
        base=weight_ptr,
        shape=(D,),
        strides=(stride_wd,),
        offsets=(0,),
        block_shape=(DIM_TILE_SIZE,),
        order=(0,),
    )
    grad_x_block_ptr = tl.make_block_ptr(
        base=grad_x_ptr,
        shape=(NUM_ROWS, D),
        strides=(stride_gxr, stride_gxd),
        offsets=(row_tile_idx * ROW_TILE_SIZE, 0),
        block_shape=(ROW_TILE_SIZE, DIM_TILE_SIZE),
        order=(1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(
        base=partial_grad_weight_ptr,
        shape=(n_row_tiles, D),
        strides=(stride_gwb, stride_gwd),
        offsets=(row_tile_idx, 0),
        block_shape=(1, DIM_TILE_SIZE),
        order=(1, 0),
    )

    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    for _ in range(tl.cdiv(D, DIM_TILE_SIZE)):
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        grad_x_row = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        grad_weight_row = tl.sum(row * grad_output[:, None], axis=0)[None, :]
        tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,))

        x_block_ptr = tl.advance(x_block_ptr, (0, DIM_TILE_SIZE))
        weight_block_ptr = tl.advance(weight_block_ptr, (DIM_TILE_SIZE,))
        grad_x_block_ptr = tl.advance(grad_x_block_ptr, (0, DIM_TILE_SIZE))
        partial_grad_weight_block_ptr = tl.advance(partial_grad_weight_block_ptr, (0, DIM_TILE_SIZE))


class WeightedSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]
        input_shape = x.shape
        x = x.reshape(-1, D)
        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"
        assert weight.is_contiguous(), "Expected contiguous weight"

        ctx.D_TILE_SIZE = min(128, triton.next_power_of_2(D))
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        y = torch.empty((x.shape[0],), device=x.device, dtype=x.dtype)
        n_rows = y.numel()

        grid = (triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)
        weighted_sum_fwd[grid](
            x,
            weight,
            y,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            y.stride(0),
            n_rows,
            D,
            ROW_TILE_SIZE=ctx.ROWS_TILE_SIZE,
            DIM_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(output_dims)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape
        grad_output = grad_output.contiguous().view(-1)

        partial_grad_weight = torch.empty(
            (triton.cdiv(n_rows, ROWS_TILE_SIZE), D),
            device=x.device,
            dtype=torch.float32,
        )
        grad_x = torch.empty_like(x)

        grid = (triton.cdiv(n_rows, ROWS_TILE_SIZE),)
        weighted_sum_bwd[grid](
            x,
            weight,
            grad_output,
            grad_x,
            partial_grad_weight,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            grad_output.stride(0),
            grad_x.stride(0),
            grad_x.stride(1),
            partial_grad_weight.stride(0),
            partial_grad_weight.stride(1),
            n_rows,
            D,
            ROW_TILE_SIZE=ROWS_TILE_SIZE,
            DIM_TILE_SIZE=D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(dim=0).to(weight.dtype)
        return grad_x.view(ctx.input_shape), grad_weight


def weighted_sum(x, weight):
    return WeightedSumFunction.apply(x, weight)


def test(seed=42):
    torch.manual_seed(seed)
    x = torch.randn(1024, 512, device="cuda")
    weight = torch.randn(512, device="cuda")
    x.requires_grad_()
    weight.requires_grad_()

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)

    y = WeightedSumFunction.apply(x, weight)
    print("Forward pass output:", y)
    grad_output = torch.randn_like(y)
    y.backward(grad_output)

    expected_y = (x_ref * weight_ref).sum(dim=-1)
    print("Expected output:", expected_y)
    expected_y.backward(grad_output)

    assert torch.allclose(y, expected_y, atol=1e-5, rtol=1e-5), "Forward pass mismatch"
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5), "Grad x mismatch"
    assert torch.allclose(weight.grad, weight_ref.grad, atol=1e-5, rtol=1e-5), "Grad weight mismatch"
    print("Test passed!")


def benchmark(
    sizes=None,
    warmup=10,
    rep=50,
    output_path=None,
):
    if sizes is None:
        sizes = [2 ** exp for exp in range(12, 28)]

    device = torch.device("cuda")
    results = []
    output_path = Path(output_path or Path(__file__).with_name("weighted_sum_benchmark.png"))

    for size in sizes:
        x = None
        weight = None
        print(f"Benchmarking size={size}")
        try:
            x = torch.randn(size, size, device=device, dtype=torch.float32)
            weight = torch.randn(size, device=device, dtype=torch.float32)

            for _ in range(warmup):
                _ = weighted_sum(x, weight)
                _ = (x * weight).sum(dim=-1)
            torch.cuda.synchronize()

            triton_start_event = torch.cuda.Event(enable_timing=True)
            triton_end_event = torch.cuda.Event(enable_timing=True)
            triton_start_event.record()
            for _ in range(rep):
                _ = weighted_sum(x, weight)
            triton_end_event.record()
            torch.cuda.synchronize()

            torch_start_event = torch.cuda.Event(enable_timing=True)
            torch_end_event = torch.cuda.Event(enable_timing=True)
            torch_start_event.record()
            for _ in range(rep):
                _ = (x * weight).sum(dim=-1)
            torch_end_event.record()
            torch.cuda.synchronize()

            triton_avg_ms = triton_start_event.elapsed_time(triton_end_event) / rep
            torch_avg_ms = torch_start_event.elapsed_time(torch_end_event) / rep
            moved_bytes = (x.numel() + weight.numel() + size) * x.element_size()
            triton_gbps = moved_bytes / (triton_avg_ms * 1e-3) / 1e9
            torch_gbps = moved_bytes / (torch_avg_ms * 1e-3) / 1e9

            results.append((size, triton_gbps, torch_gbps, triton_avg_ms, torch_avg_ms))
            print(
                f"size={size}, "
                f"triton_ms={triton_avg_ms:.4f}, triton_GB/s={triton_gbps:.2f}, "
                f"torch_ms={torch_avg_ms:.4f}, torch_GB/s={torch_gbps:.2f}"
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower() and not isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
            torch.cuda.empty_cache()
            print(f"Skipping size={size} due to OOM")
        finally:
            if x is not None:
                del x
            if weight is not None:
                del weight
            torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("No benchmark points completed successfully.")

    xs = [size for size, _, _, _, _ in results]
    triton_ys = [triton_gbps for _, triton_gbps, _, _, _ in results]
    torch_ys = [torch_gbps for _, _, torch_gbps, _, _ in results]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, triton_ys, marker="o", label="Triton WeightedSum")
    plt.plot(xs, torch_ys, marker="s", label="PyTorch sum(x * weight)")
    plt.xscale("log", base=2)
    plt.xlabel("size")
    plt.ylabel("GB/s")
    plt.title("WeightedSum Forward Throughput")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Saved plot to {output_path}")
    return results, output_path


def parse_args():
    parser = argparse.ArgumentParser(description="WeightedSum Triton test and benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Run throughput benchmark")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=50, help="Measured iterations for benchmark")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the benchmark plot",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.benchmark:
        benchmark(warmup=args.warmup, rep=args.rep, output_path=args.output)
    else:
        test()
