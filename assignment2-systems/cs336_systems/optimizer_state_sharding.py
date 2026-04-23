from __future__ import annotations

from typing import Any, Type

import torch
import torch.distributed as dist
import torch.optim as optim


class OptimizerStateSharding(optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs: Any):
        if not dist.is_initialized():
            raise RuntimeError("OptimizerStateSharding requires torch.distributed to be initialized.")

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = dict(kwargs)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self._owned_rank_by_param = {}
        self._local_param_groups = []
        self._next_param_index = 0
        self.local_optimizer = None
        self._initializing = True

        super().__init__(params, kwargs)

        self._initializing = False
        non_empty_local_groups = [group for group in self._local_param_groups if group["params"]]
        if non_empty_local_groups:
            self.local_optimizer = optimizer_cls(non_empty_local_groups, **kwargs)

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        loss = None
        if self.local_optimizer is not None:
            if closure is not None:
                with torch.enable_grad():
                    loss = self.local_optimizer.step(closure=closure, **kwargs)
            else:
                loss = self.local_optimizer.step(**kwargs)

        for group in self.param_groups:
            for param in group["params"]:
                dist.broadcast(param.data, src=self._owned_rank_by_param[param])

        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)

        added_group = self.param_groups[-1]
        local_params = []
        for param in added_group["params"]:
            owner_rank = self._next_param_index % self.world_size
            self._owned_rank_by_param[param] = owner_rank
            if owner_rank == self.rank:
                local_params.append(param)
            self._next_param_index += 1

        local_group = {key: value for key, value in added_group.items() if key != "params"}
        local_group["params"] = local_params
        self._local_param_groups.append(local_group)

        if self._initializing or not local_params:
            return

        if self.local_optimizer is None:
            self.local_optimizer = self.optimizer_cls([local_group], **self.optimizer_kwargs)
        else:
            self.local_optimizer.add_param_group(local_group)
