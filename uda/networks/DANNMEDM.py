import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.hub import load_state_dict_from_url
from typing import Tuple
from uda.networks.DANN import GradientReversalFn


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
        if pretrained:
            backbone = resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        else:
            backbone = resnet18()
        features = nn.ModuleList(backbone.children())[:-1]

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
