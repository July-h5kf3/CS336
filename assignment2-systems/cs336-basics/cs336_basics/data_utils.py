from torch.utils.data import Dataset
import numpy as np
import torch

def get_batch(x, batch_size, context_length, device):
    """
    input:
        x:  Int[Tensor, "seq_len"] 或 numpy array
        batch_size: int
        context_length: int
        device: torch.device
    output:
        xb: Int[Tensor, "batch_size context_length"]
        yb: Int[Tensor, "batch_size context_length"]
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    
    idx = torch.randint(0, len(x) - context_length, size=(batch_size,))
    xb = torch.stack([x[i:i+context_length] for i in idx]).long()
    yb = torch.stack([x[i+1:i+context_length+1] for i in idx]).long()
    xb, yb = xb.to(device), yb.to(device)
    return xb, yb

def save_checkpoint(model,optimizer,epoch, out):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, out)

def load_checkpoint(src, model, optimizer=None):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]
