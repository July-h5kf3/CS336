import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.multiprocessing as mp

class DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)

        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            def make_hook(p):
                def hook(_):
                    if p.grad is None:
                        return
                    handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                    self.handles.append((handle, p))
                return hook
            param.register_post_accumulate_grad_hook(make_hook(param))
    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle, param in self.handles:
            handle.wait()
            param.grad /= self.world_size
        self.handles = []

class Bucket_DDP(nn.Module):
    def __init__(self, module,bucket_size_mb=10):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.bucket_size_mb = bucket_size_mb

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param, src=0)
        params = [p for p in self.module.parameters() if p.requires_grad]
        params = list(reversed(params))
        
        self.buckets = []
        current_bucket = []
        current_size = 0

        for param in params:
            p_size = param.numel()
            if current_bucket and current_size + p_size > bucket_size_mb * 1024 * 1024 // 4:
                self.buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            current_bucket.append(param)
            current_size += p_size
        if current_bucket:
            self.buckets.append(current_bucket)
        self.param2bucket = {}
        for bucket_idx, bucket in enumerate(self.buckets):
            for p in bucket:
                self.param2bucket[p] = bucket_idx

        self.bucket_ready_count = [0 for _ in self.buckets]
        self.bucket_handles = [None for _ in self.buckets]
        self.bucket_flat_grads = [None for _ in self.buckets]

        for p in params:
            if not p.requires_grad:
                continue
            def make_hook(p):
                def hook(_):
                    if p.grad is None:
                        return
                    bucket_idx = self.param2bucket[p]
                    self.bucket_ready_count[bucket_idx] += 1
                    if self.bucket_ready_count[bucket_idx] == len(self.buckets[bucket_idx]):
                        grads = [param.grad for param in self.buckets[bucket_idx]]
                        flat_grads = _flatten_dense_tensors(grads)
                        handle = dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, async_op=True)
                        self.bucket_flat_grads[bucket_idx] = flat_grads
                        self.bucket_handles[bucket_idx] = handle
                return hook
            p.register_post_accumulate_grad_hook(make_hook(p))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for bucket_idx, handle in enumerate(self.bucket_handles):
            handle = self.bucket_handles[bucket_idx]
            if handle is None:
                continue
            handle.wait()
            flat_grads = self.bucket_flat_grads[bucket_idx]
            flat_grads /= self.world_size
            grads = [p.grad for p in self.buckets[bucket_idx]]
            synced_grads = _unflatten_dense_tensors(flat_grads, grads)
            for param, grad in zip(self.buckets[bucket_idx], synced_grads):
                param.grad.copy_(grad)
            self.bucket_ready_count[bucket_idx] = 0
            self.bucket_handles[bucket_idx] = None
            self.bucket_flat_grads[bucket_idx] = None