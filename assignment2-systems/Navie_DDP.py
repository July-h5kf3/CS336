import torch
import torch.nn as nn
from einops import rearrange

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import AdamW

import os
from copy import deepcopy


class ToyMLP(nn.Module):
    def __init__(self, d_in=16, d_hidden=32, d_out=4):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = self.fc2(self.act(self.fc1(x)))
        return x


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    init_kwargs = {"backend": backend, "rank": rank, "world_size": world_size}
    if backend == "nccl":
        init_kwargs["device_id"] = torch.device(f"cuda:{rank}")
    dist.init_process_group(**init_kwargs)


def get_device(rank, backend):
    if backend == "nccl":
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def distributed_train(rank, world_size, backend, x_rand, y_rand):
    torch.manual_seed(rank)
    setup(rank, world_size, backend)
    device = get_device(rank, backend)
    model = ToyMLP().to(device)
    x_rand = x_rand.to(device)
    y_rand = y_rand.to(device)

    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param, src=0)

    if rank == 0:
        model_baseline = deepcopy(model)
        baseline_opt = AdamW(
            model_baseline.parameters(),
            lr=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
        )
        loss_fn = nn.MSELoss(reduction="mean")
        baseline_opt.zero_grad()
        pred = model_baseline(x_rand)
        loss = loss_fn(pred, y_rand)
        loss.backward()
        baseline_opt.step()

    x_local = rearrange(x_rand, "(d b) f -> d b f", d=world_size)[rank]
    y_local = rearrange(y_rand, "(d b) f -> d b f", d=world_size)[rank]
    optimizer = AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
    )
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer.zero_grad()
    pred = model(x_local)
    loss = loss_fn(pred, y_local)
    loss.backward()

    for param in model.parameters():
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad /= world_size

    optimizer.step()

    dist.barrier()

    if rank == 0:
        for i, (p_base, p_ddp) in enumerate(zip(model_baseline.parameters(), model.parameters())):
            max_diff = (p_base - p_ddp).abs().max().item()
            print(f"param {i}: max_diff = {max_diff:.8e}")

    dist.destroy_process_group()


if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        backend = "nccl"
    else:
        world_size = 4
        backend = "gloo"
    batch_size = 128
    x_rand = torch.randn(batch_size, 16)
    y_rand = torch.randn(batch_size, 4)
    mp.spawn(fn=distributed_train, args=(world_size, backend, x_rand, y_rand), nprocs=world_size, join=True)
