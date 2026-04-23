import torch
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "cs336-basics"))
from cs336_basics.Transformer import Transformer

VOCAB_SIZE = 10000
CONTEXT_LENGTH = 128
BATCH_SIZE_PER_RANK = 1
NUM_STEPS = 300
WARMUP_STEPS = 20


def setup(rank, world_size, backend, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    init_kwargs = {"backend": backend, "rank": rank, "world_size": world_size}
    if backend == "nccl":
        init_kwargs["device_id"] = torch.device(f"cuda:{rank}")
    dist.init_process_group(**init_kwargs)


def get_device(rank, backend):
    if backend == "nccl":
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_worker(rank, world_size, backend, use_flatten, master_port):
    torch.manual_seed(rank)
    setup(rank, world_size, backend, master_port)
    device = get_device(rank, backend)
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        num_layers=24,
        d_model=1024,
        num_heads=16,
        device=device,
    ).to(device)
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)

    local_tokens = torch.randint(
        0, VOCAB_SIZE, (BATCH_SIZE_PER_RANK, CONTEXT_LENGTH), device=device, dtype=torch.long
    )
    local_targets = torch.randint(
        0, VOCAB_SIZE, (BATCH_SIZE_PER_RANK, CONTEXT_LENGTH), device=device, dtype=torch.long
    )

    step_times = []
    total_steps = WARMUP_STEPS + NUM_STEPS
    for step in range(total_steps):
        dist.barrier()
        sync_device(device)
        start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        logits = model(local_tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), local_targets.reshape(-1))
        loss.backward()
        if use_flatten:
            params = [p for p in model.parameters() if p.grad is not None]
            grads = [p.grad for p in params]
            flat_grads = _flatten_dense_tensors(grads)
            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
            flat_grads /= world_size
            synced_grads = _unflatten_dense_tensors(flat_grads, grads)
            for param, grad in zip(params, synced_grads):
                param.grad.copy_(grad)
        else:
            for param in model.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size
        optimizer.step()

        sync_device(device)
        dist.barrier()
        end = time.perf_counter()

        if step >= WARMUP_STEPS:
            step_time = torch.tensor(end - start, device=device)
            dist.all_reduce(step_time, op=dist.ReduceOp.MAX)
            step_times.append(step_time.item())

    if rank == 0:
        label = "flattened all-reduce" if use_flatten else "per-parameter all-reduce"
        avg_time = sum(step_times) / len(step_times)
        print(f"{label}: {avg_time:.3f} s/step")
        print("model config: d_model=1600, d_ff=6400, num_layers=48, num_heads=25")

    dist.destroy_process_group()

if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        backend = "nccl"
    else:
        world_size = 4
        backend = "gloo"
    mp.spawn(
        fn=train_worker,
        args=(world_size, backend, False, 29500),
        nprocs=world_size,
        join=True,
    )
    mp.spawn(
        fn=train_worker,
        args=(world_size, backend, True, 29501),
        nprocs=world_size,
        join=True,
    )
