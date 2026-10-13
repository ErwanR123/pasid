import math

from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(
    optimizer, warmup_steps, cosine_steps, warmup_factor=0.1, final_factor=0.1
):
    """Learning rate, as a factor of the base one : linear warmup from warmup_factor to 1
    over warmup_steps, then cosine decay to final_factor over cosine_steps, then flat."""

    def factor(step):
        if step < warmup_steps:
            return warmup_factor + (1 - warmup_factor) * step / warmup_steps
        progress = min((step - warmup_steps) / cosine_steps, 1.0)
        return final_factor + (1 - final_factor) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, factor)
