import torch.nn as nn
import torch

############# AlexNet optimizers ##################


def get_native_alexnet_optimizer(
    net: nn.Module, lr: float, wd: float, momentum: float
) -> torch.optim.Optimizer:
    r"""
    Stochastic gradient descent optimizer
    It is employed to train the native AlexNet or any other nn.Module
    architecture

    Args:
    - net [nn.Module]: network architecture
    - lr [float]: learning rate
    - wd [float]: weight decay
    - momentum [float]: momentum

    Returns:
    - optimizer [torch.optim.Optimizer]
    """
    return torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)


def get_ddc_alexnet_optimizer(
    net: nn.Module, lr: float, wd: float, momentum: float
) -> torch.optim.Optimizer:
    r"""
    Stochastic gradient descent optimizer
    Optimizer for the DDCAlexNet architecture, with a learning rate multiplied
    by ten for the last two layers, as explained in the Deep Domain Confusion:
    Maximizing for Domain Invariance paper

    Args:
    - net [nn.Module]: network architecture
    - lr [float]: learning rate
    - wd [float]: weight decay
    - momentum [float]: momentum

    Returns:
    - optimizer [torch.optim.Optimizer]
    """
    return torch.optim.SGD(
        [
            {"params": net.features.parameters(), "lr": lr},
            {"params": net.avgpool.parameters(), "lr": lr},
            {"params": net.mid_classifier.parameters(), "lr": lr},
            {"params": net.adapt.parameters(), "lr": lr * 10},
            {"params": net.classifier.parameters(), "lr": lr * 10},
        ],
        lr=lr,
        weight_decay=wd,
        momentum=momentum,
    )


def get_dann_alexnet_optimizer(
    net: nn.Module, lr: float, wd: float, momentum: float
) -> torch.optim.Optimizer:
    r"""
    Stochastic gradient descent optimizer
    It is employed to train the DANNAlexNet as described in
    the Domain-Adversarial Training of Neural Networks paper

    Args:
    - net [nn.Module]: network architecture
    - lr [float]: learning rate
    - wd [float]: weight decay
    - momentum [float]: momentum

    Returns:
    - optimizer nn.Optimizer
    """
    return torch.optim.SGD(
        [
            {"params": net.features.parameters(), "lr": 0.001},
            {"params": net.avgpool.parameters(), "lr": 0.001},
            {"params": net.mid_classifier.parameters(), "lr": 0.001},
            {"params": net.adapt.parameters()},
            {"params": net.classifier.parameters()},
            {"params": net.domain_classifier.parameters()},
        ],
        lr=lr,
        momentum=momentum,
    )


########## SDG for ResNet architecture ###############


def get_native_resnet_optimizer(
    net: nn.Module, lr: float, wd: float, momentum: float
) -> torch.optim.Optimizer:
    r"""
    Stochastic gradient descent optimizer
    It is employed to train the native Resnet or any other nn.Module
    architecture

    Args:
    - net [nn.Module]: network architecture
    - lr [float]: learning rate
    - wd [float]: weight decay
    - momentum [float]: momentum

    Returns:
    - optimizer [torch.optim.Optimizer]
    """
    return torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)


def get_ddc_resnet_optimizer(
    net: nn.Module, lr: float, wd: float, momentum: float
) -> torch.optim.Optimizer:
    r"""
    Stochastic gradient descent optimizer
    Optimizer for the DDCResnet architecture, with a learning rate multiplied
    by ten for the last two layers, as explained in the Deep Domain Confusion:
    Maximizing for Domain Invariance paper

    Args:
    - net [nn.Module]: network architecture
    - lr [float]: learning rate
    - wd [float]: weight decay
    - momentum [float]: momentum

    Returns:
    - optimizer [torch.optim.Optimizer]
    """
    return torch.optim.SGD(
        [
            {"params": net.features.parameters(), "lr": lr},
            {"params": net.adapt.parameters(), "lr": lr * 10},
            {"params": net.classifier.parameters(), "lr": lr * 10},
        ],
        lr=lr,
        weight_decay=wd,
        momentum=momentum,
    )


def get_dann_resnet_optimizer(
    net: nn.Module, lr: float, wd: float, momentum: float
) -> torch.optim.Optimizer:
    r"""
    Stochastic gradient descent optimizer
    It is employed to train the DANNResnet as described in
    the Domain-Adversarial Training of Neural Networks paper

    Args:
    - net [nn.Module]: network architecture
    - lr [float]: learning rate
    - wd [float]: weight decay
    - momentum [float]: momentum

    Returns:
    - optimizer nn.Optimizer
    """
    print("RESNET DANN OPTIMIZER")
    return torch.optim.SGD(
        [
            {"params": net.features.parameters(), "lr": 0.001},
            {"params": net.adapt.parameters()},
            {"params": net.classifier.parameters()},
            {"params": net.domain_classifier.parameters()},
        ],
        lr=lr,
        momentum=momentum,
    )


def get_rotation_resnet_optimizer(
    net: nn.Module, lr: float, wd: float, momentum: float
) -> torch.optim.Optimizer:
    r"""
    Stochastic gradient descent optimizer
    It is employed to train the DANNResnet as described in
    the Domain-Adversarial Training of Neural Networks paper

    Args:
    - net [nn.Module]: network architecture
    - lr [float]: learning rate
    - wd [float]: weight decay
    - momentum [float]: momentum

    Returns:
    - optimizer nn.Optimizer
    """
    print("RESNET ROTATION OPTIMIZER")
    return torch.optim.SGD(
        [
            {"params": net.features.parameters(), "lr": 0.001},
            {"params": net.adapt.parameters()},
            {"params": net.classifier.parameters()},
            {"params": net.domain_adversarial_network.parameters()},
            {"params": net.rotation_classifier.parameters()},
        ],
        lr=lr,
        momentum=momentum,
    )


def get_dsn_resnet_optimizer(
    net: nn.Module, lr: float, wd: float, momentum: float
) -> torch.optim.Optimizer:
    r"""
    Stochastic gradient descent optimizer
    It is employed to train the DANNResnet as described in
    the Domain-Adversarial Training of Neural Networks paper

    Args:
    - net [nn.Module]: network architecture
    - lr [float]: learning rate
    - wd [float]: weight decay
    - momentum [float]: momentum

    Returns:
    - optimizer nn.Optimiser
    """
    return torch.optim.SGD(
        [
            {"params": net.shared_encoder.parameters(), "lr": 0.001},
            {"params": net.source_encoder.parameters(), "lr": 0.001},
            {"params": net.target_encoder.parameters(), "lr": 0.001},
            {"params": net.bottle_neck.parameters()},
            {"params": net.classifier.parameters()},
            {"params": net.domain_adversarial_network.parameters()},
        ],
        lr=lr,
        momentum=momentum,
    )
