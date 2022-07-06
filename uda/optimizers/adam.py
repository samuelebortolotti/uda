import torch.nn as nn
import torch


def get_adam_optimizer(net: nn.Module, lr: float) -> torch.optim.Optimizer:
    r"""
    ADAM optimizer

    Args:

    - net [nn.Module]: network architecture
    - lr [float]: learning rate

    Returns:
    - optimizer [nn.Optimizer]
    """
    return torch.optim.Adam(net.parameters(), lr=lr)
