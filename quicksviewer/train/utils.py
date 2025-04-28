import copy
import math
import torch.distributed as dist

def calculate_loss_weight(labels, ignore_index=-100):
    # (Qinghao): Weighted loss based on num_active_elements
    # To achieve accurate sequence parallel loss calculation, we need to get
    # the real active_elements of each sequence partitions.
    # For data parallelism, the loss almost remains the same (also more accurate).
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.view(-1)

    padding_mask = shift_labels.eq(ignore_index)  # IGNORE_INDEX = -100 by default
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    global_active_sum = copy.deepcopy(num_active_elements)
    dist.all_reduce(global_active_sum)
    loss_weight = num_active_elements / global_active_sum * dist.get_world_size()
    return loss_weight




class CosineAnnealer:
    def __init__(self, max_update, base_lr=1.0, final_lr=0.001,
               warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps
        # initialize base_lr as final_lr
        self.base_lr = final_lr

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr
