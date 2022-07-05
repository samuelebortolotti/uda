import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18(nn.Module):
    r"""
    Original ResNet18 architecture, which is taken directly from the torchvision
    models
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
        super(ResNet18, self).__init__()

        # Take the resNet18 module and discard the last layer
        features = nn.ModuleList(resnet18(pretrained=pretrained).children())[:-1]

        # set the ResNet18 backbone as feature extractor
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes),
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
