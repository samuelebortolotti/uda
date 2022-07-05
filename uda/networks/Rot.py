import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import Tuple


class RotationArch(nn.Module):
    r"""
    Architecture employing the rotation loss.
    This implementation takes inspiration from the Self-Supervised Domain adaptation
    with Consistency training by Liang Xiao et al.
    This is a self-supervised approach that employs a pretext task of predicting
    rotation angles of the images to learn important features of target domain data
    """

    def __init__(self, num_classes: int = 20, dropout: float = 0.5) -> None:
        super(RotationArch, self).__init__()

        # Take the resNet18 module and discard the last layer
        features = nn.ModuleList(resnet18(pretrained=True).children())[:-1]

        # Use it as a feature extractor
        self.features = nn.Sequential(*features)

        # final fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(512, 20),
        )

        self.rotation_classifier = nn.Sequential(
            nn.Linear(512, 4),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward method

        Args:
        - x [torch.Tensor]: source sample

        Returns:
        - class_pred [torch.Tensor]: class prediction
        - rotation_pred [torch.Tensor]: rotation prediction
        """

        x = self.features(x)

        # extract the features
        features = x.view(x.size(0), -1)

        # classify both the domain and the class
        class_pred = self.classifier(features)

        # get the rotation prediction
        rotation_pred = self.rotation_classifier(features)

        return class_pred, rotation_pred
