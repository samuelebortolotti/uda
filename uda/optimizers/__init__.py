"""Optimizers module
It deals with all the optimizers we have employed according to the papers we have implemented or the approaches we have experimented
"""
from .adam import get_adam_optimizer
from .sdg import (
    get_dann_alexnet_optimizer,
    get_dann_resnet_optimizer,
    get_ddc_alexnet_optimizer,
    get_ddc_resnet_optimizer,
    get_dsn_resnet_optimizer,
    get_native_alexnet_optimizer,
    get_native_resnet_optimizer,
    get_rotation_resnet_optimizer,
)
