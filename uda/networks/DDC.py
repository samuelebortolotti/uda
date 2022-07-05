import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet18
from typing import Tuple
import numpy as np

################### Loss function ###################


def get_mmd(x_source: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    r"""
    Maximum Mean Discrepancy (MMD) distance implementation.
    It is MMD distance employed in the Deep Domain Confusion: Maximizing for Domain Invariance
    paper by Eric Tzeng et al.

    The implementation is taken from
    [link]: https://github.com/jindongwang/transferlearning/blob/master/code/distance/mmd_numpy_sklearn.py#L7

    Args:
    - x_source [torch.Tensor]: source sample
    - x_target [torch.Tensor]: target sample

    Returns:
    - loss [torch.Tensor]: distance
    """
    delta = x_source.mean(0) - x_target.mean(0)
    return delta.dot(delta.T)


################### AlexNet ###################


class DDCAlexNet(nn.Module):
    r"""
    Customized AlexNet architecture according to the Deep Domain Confusion:
    Maximizing for Domain Invariance paper by Eric Tzeng et al.
    This implementation takes inspiration from the Deep Domain Confusion:
    Maximizing for Domain Invariance paper by Eric Tzeng et al.
    In practice, we aim to reproduce the principles of such architecture even if
    we have not widely explored it like Eric Tzeng et al.
    """

    def __init__(
        self,
        num_classes: int = 20,
        dropout: float = 0.5,
        pretrained: bool = True,
        feature_extractor: bool = False,
    ) -> None:
        r"""
        Initialize the custom DDCAlexNet model which implements
        the Deep Domain Confusion approach

        Default:
        - num_classes [int] = 20
        - dropout [float] = 0.5
        - pretrained [bool] = False
        - feature_extractor [bool] = False

        Args:
        - num_classes [int]: number of classes [used in the last layer]
        - dropout [float]: probability of dropout [value between 0-1]
        - pretrained [bool]: whether to pretrain the model or not
        - feature_extractor [bool]: whether to return the feature extracted from
        the neural network or not in the forward step.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.mid_classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        # Adapt layer before the last linear layer
        # as described in the Deep Domain Confusion paper
        self.adapt = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(inplace=True),
        )
        # Last fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
        )
        # Whether to return only the features in the forward step
        self.feature_extractor = feature_extractor

        # whether to load a pretrained model
        if pretrained:
            from torchvision.models.alexnet import model_urls

            pretrained_params = load_state_dict_from_url(
                model_urls["alexnet"], progress=True
            )

            # load weights from alexnet base net
            current_params = self.state_dict()

            # loading weights coming from common layers
            for key, value in current_params.items():
                # the common layers between the two nets are:
                # features->features and classifier->mid_classifier
                if key.split(".")[0] == "features":
                    print("Adding key", key, "in current params")
                    current_params[key] = pretrained_params[key]
                if key.split(".")[0] == "mid_classifier":
                    print("Adding key", key, "in current params")
                    current_params[key] = pretrained_params[
                        key.replace("mid_classifier", "classifier")
                    ]
            # load weigths
            self.load_state_dict(current_params)

    def forward(
        self, x: torch.Tensor, x_target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward method

        **Note**: in the case of `feature_extractor` set to `True` it returns
        only the features [torch.Tensor]

        Args:
        - x [torch.Tensor]: source sample
        - x_target [torch.Tensor]: target sample

        Returns:
        - prediction, loss [Tuple[torch.Tensor, torch.Tensor]]: prediction and mmd loss
        """
        x = self.features(x)
        # if in the feature extractor mode
        if self.feature_extractor:
            return x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.mid_classifier(x)
        x = self.adapt(x)

        # If we are in training mode, find also the adapt layer's result
        # for target data and compute the mmd distance between source and target
        loss = 0
        if self.training:
            x_target = self.features(x_target)
            x_target = self.avgpool(x_target)
            x_target = torch.flatten(x_target, 1)
            x_target = self.mid_classifier(x_target)
            x_target = self.adapt(x_target)
            loss += get_mmd(x, x_target)

        x = self.classifier(x)

        # Return also the MMD loss, not only the actual ouptut of the net
        # loss is simply zero if we are not training
        return x, loss


################### ResNet18 ###################


class DDCResNet18(nn.Module):
    r"""
    Customized DDCResNet18 architecture.
    This implementation takes inspiration from the Deep Domain Confusion:
    Maximizing for Domain Invariance paper by Eric Tzeng et al.
    In practice, we aim to reproduce the principles of such architecture even if
    we have not widely explored it like Eric Tzeng et al.
    """

    def __init__(
        self,
        num_classes: int = 20,
        pretrained: bool = False,
        feature_extractor: bool = False,
    ) -> None:
        super(DDCResNet18, self).__init__()
        r"""
        Initialize the custom DDCResNet18 model which tries to follow
        the general idea behind Deep Domain Confusion approach

        Default:
        - num_classes [int] = 20
        - pretrained [bool] = False
        - feature_extractor [bool] = False

        Args:
        - num_classes [int]: number of classes [used in the last layer]
        - dropout [float]: probability of dropout [value between 0-1]
        - pretrained [bool]: whether to pretrain the model or not
        - feature_extractor [bool]: whether to return the feature extracted from
        the neural network or not in the forward step.
        """
        # Take the resNet18 module and discard the last layer
        features = nn.ModuleList(resnet18(pretrained=pretrained).children())[:-1]

        # Use it as a feature extractor
        self.features = nn.Sequential(*features)

        # add an adapt layer before the last one
        # similar to what Tzeng et al. did with AlexNet
        self.adapt = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # final fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
        )

        # in the forward return only the features!
        self.feature_extractor = feature_extractor

    def forward(
        self, x: torch.Tensor, x_target: torch.Tensor
    ) -> Tuple[torch.Tensor, np.double]:
        r"""
        Forward method

        **Note**: in the case of `feature_extractor` set to `True` it returns
        only the features [torch.Tensor]

        Args:
        - x [torch.Tensor]: source sample
        - x_target [torch.Tensor]: target sample

        Returns:
        - prediction, loss [Tuple[torch.Tensor, float]]: prediction and mmd loss
        """

        x = self.features(x)

        # if in the feature extractor mode
        if self.feature_extractor:
            return x

        # flatten to prepare for the classifier
        x = torch.flatten(x, 1)
        x = self.adapt(x)

        loss = 0
        if self.training:
            x_target = self.features(x_target)
            x_target = torch.flatten(x_target, 1)
            x_target = self.adapt(x_target)
            loss += get_mmd(x, x_target)

        x = self.classifier(x)

        # Return also the MMD loss, not only the actual ouptut of the net
        # loss is simply zero if we are not training
        return x, loss
