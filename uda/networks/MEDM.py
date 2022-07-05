from torchvision.models import resnet18
import torch
import torch.nn as nn
from typing import Tuple
from uda.networks.DANN import GradientReversalFn


class MEDM(nn.Module):
    r"""
    Original ResNet18 architecture, integrated with MEDM
    """

    def __init__(
        self,
        num_classes: int = 20,
        pretrained: bool = False,
        feature_extractor: bool = False,
    ) -> None:
        r"""
        Initialize the basic ResNet18 architecture

        Default:
        - num_classes [int] = 20
        - pretrained [bool] = False
        - feature_extractor [bool] = False

        Args:
        - num_classes [int]: number of classes [used in the last layer]
        - pretrained [bool]: whether to pretrain the model or not
        - feature_extractor [bool]: whether to return the feature extracted from
        the neural network or not in the forward step.
        """
        super(MEDM, self).__init__()

        # Take the resNet18 module and discard the last layer
        features = nn.ModuleList(resnet18(pretrained=pretrained).children())[:-1]

        # set the ResNet18 backbone as feature extractor
        self.features = nn.Sequential(*features)

        # classifier adapted from the original paper's approach
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(),
        )

        # As usual the mode for returning the features instead of the
        # predictions
        self.feature_extractor = feature_extractor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward method

        Args:
        - x [torch.Tensor]: source sample

        Returns:
        - prediction [torch.Tensor]: prediction
        """
        x = self.features(x)
        # if in the feature extractor mode
        if self.feature_extractor:
            # return the features
            return x
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DANNMEDM(nn.Module):
    r"""
    ResNet18 architecture, integrated with a new method exploiting both DANN
    and MEDM approaches
    """

    def __init__(
        self,
        num_classes: int = 20,
        pretrained: bool = False,
        dropout: float = 0.5,
        feature_extractor: bool = False,
    ) -> None:
        r"""
        Initialize the architecture

        Default:
        - num_classes [int] = 20
        - pretrained [bool] = False
        - feature_extractor [bool] = False

        Args:
        - num_classes [int]: number of classes [used in the last layer]
        - pretrained [bool]: whether to pretrain the model or not
        - feature_extractor [bool]: whether to return the feature extracted from
        the neural network or not in the forward step.
        """
        super(DANNMEDM, self).__init__()

        # Take the resNet18 module and discard the last layer
        features = nn.ModuleList(resnet18(pretrained=pretrained).children())[:-1]

        # set the ResNet18 backbone as feature extractor
        self.features = nn.Sequential(*features)

        # features
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
            nn.Softmax(),
        )

        # As adviced by the paper, we follow the previous structure
        # for deep domain confusion, adding an adapt layer before the classifiers
        self.adapt = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # Domain classifier similar to the one described in the paper
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
            nn.Softmax(),
        )

        # in the forward return only the features!
        self.feature_extractor = feature_extractor

    def forward(
        self, x: torch.Tensor, grl_lambda: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward method

        Args:
        - x [torch.Tensor]: source sample

        Returns:
        - prediction [torch.Tensor]: prediction
        """
        x = self.features(x)
        # if in the feature extractor mode
        if self.feature_extractor:
            # return the features
            return x
        # extract the features
        features = x.view(x.size(0), -1)
        features = self.adapt(features)

        # gradient reversal
        reverse_features = GradientReversalFn.apply(features, grl_lambda)

        # classify both the domain and the class
        class_pred = self.classifier(features)
        domain_pred = self.domain_classifier(reverse_features)
        return class_pred, domain_pred
