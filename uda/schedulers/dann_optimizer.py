import torch
import numpy as np


def dann_optimizer_scheduler(
    optimizer: torch.optim.Optimizer, p: np.double
) -> np.double:
    r"""
    Custom scheduler for DANN's optimizer. The update of the learning rate
    follows the formula described in the paper (Domain-Adversarial Training of
    Neural Networks). The learning rate for the feature extraction layers is
    exactly the calculated one, while the learning rate for the adaptation layer
    and the classifiers is multiplied by 10, since they don't start pre-trained.

    Args:

    - optimizer [torch.optim.Optimizer]: optimizer
    - p [np.double]: p parameter

    Return:

    - lr np.double
    """
    lr = 0.01 / (1.0 + 10 * p) ** 0.75
    for param_group in optimizer.param_groups[:3]:
        param_group["lr"] = lr
    for param_group in optimizer.param_groups[3:]:
        param_group["lr"] = lr * 10
    return lr
