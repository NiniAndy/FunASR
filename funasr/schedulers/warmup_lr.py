"""Warm up learning rate scheduler module."""

import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Union

from funasr.schedulers.abs_scheduler import AbsBatchStepScheduler


class WarmupLR(_LRScheduler, AbsBatchStepScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: Union[int, float] = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr * self.warmup_steps**0.5 * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]


# class WarmupLR(_LRScheduler):
#     """The WarmupLR scheduler
#
#     This scheduler is almost same as NoamLR Scheduler except for following
#     difference:
#
#     NoamLR:
#         lr = optimizer.lr * model_size ** -0.5
#              * min(step ** -0.5, step * warmup_step ** -1.5)
#     WarmupLR:
#         lr = optimizer.lr * warmup_step ** 0.5
#              * min(step ** -0.5, step * warmup_step ** -1.5)
#
#     Note that the maximum lr equals to optimizer.lr in this scheduler.
#
#     """
#
#     def __init__(
#         self,
#         optimizer: torch.optim.Optimizer,
#         warmup_steps: Union[int, float, List[Union[int, float]]] = 25000,
#         last_epoch: int = -1,
#     ):
#         self.warmup_steps = warmup_steps
#         # __init__() must be invoked before setting field
#         # because step() is also invoked in __init__()
#         super().__init__(optimizer, last_epoch)
#
#     def __repr__(self):
#         return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"
#
#     def get_lr(self):
#         step_num = self.last_epoch + 1
#         if step_num <= 0:
#             step_num = 1
#         warmup_steps = self.warmup_steps
#         if not isinstance(warmup_steps, List):
#             warmup_steps = [self.warmup_steps] * len(self.base_lrs)
#
#         def initlr_fn(lr):
#             return lr * step_num**-0.5
#
#         def warmuplr_fn(lr, warmup_step):
#             return lr * warmup_step**0.5 * min(step_num**-0.5, step_num * warmup_step**-1.5)
#
#         return [initlr_fn(lr) if warmup_steps[i] == 0 else warmuplr_fn(lr, warmup_steps[i]) for (i, lr) in enumerate(self.base_lrs)]
#
#     def set_step(self, step: int):
#         self.last_epoch = step
