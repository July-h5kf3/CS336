import torch
from einops import rearrange, einsum
import math

class Adamw(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,weight_decay=0.01):
        defaults = dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(params,defaults)
    def step(self,closure=None):
        loss = None
        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            # step 放在 group 级别，所有参数共享
            step = group.get('step', 0) + 1
            group['step'] = step
            
            # 计算偏差修正后的学习率
            lr_t = lr * math.sqrt(1 - betas[1] ** step) / (1 - betas[0] ** step)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]  # 只包含一阶矩、二阶矩
                m = state.get('m', torch.zeros_like(p))
                v = state.get('v', torch.zeros_like(p))
                
                # 更新一阶矩、二阶矩
                m = betas[0] * m + (1 - betas[0]) * p.grad
                v = betas[1] * v + (1 - betas[1]) * p.grad * p.grad

                # 更新参数
                p.data = p.data - lr_t * (m / (torch.sqrt(v) + eps))
                # 参数衰减
                p.data = p.data * (1 - weight_decay * lr)
                
                state['m'] = m
                state['v'] = v
        return loss