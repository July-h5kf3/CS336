import math
import torch

def gradient_clip(params, max_norm, epsilon=1e-6):
    total_norm = torch.sqrt(sum([torch.norm(p.grad.data, p=2) ** 2 for p in params if p.grad is not None]))
    scale = max_norm / (total_norm + epsilon)
    if scale < 1:
        for p in params:
            if p.grad is not None:
                p.grad.data *= scale

        