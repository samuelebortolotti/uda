import torch


def exp_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    step: int,
    init_lr: float,
    lr_decay_step: int,
    step_decay_weight: int,
) -> torch.optim.Optimizer:
    r"""
    Exponential learning rate schedule

    Args:

    - optimizer [torch.optim.Optimizer]: oprimizers
    - step [float]: step of the exponential learning rate schedule
    - init_lr [float]: initial learning rate
    - step_decay_weight [int]: step decay

    Returns:

    - optimizer [torch.optim.Optimizer]: optimizer with updated schedule
    """

    # Learning rate decay
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

    # update of the learning rate: the same for all the parameters
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer
