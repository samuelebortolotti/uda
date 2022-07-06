import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url


class AlexNet(nn.Module):
    r"""
    AlexNet architecture taken from the PyTorch source code.
    The reference is taken from
    [link]: https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
    """

    def __init__(
        self,
        num_classes: int = 20,
        dropout: float = 0.5,
        pretrained: bool = False,
        feature_extractor: bool = False,
    ) -> None:
        r"""
        Initialize the AlexNet model

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
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # number of classes as output channel of the last fc layer
            nn.Linear(4096, num_classes),
        )
        # Whether to return only the features in the forward step!
        self.feature_extractor = feature_extractor

        # automatic pretrained model
        if pretrained:
            # url of the AlexNet weights
            from torchvision.models import alexnet as anet

            if pretrained:
                backbone = anet(weights="AlexNet_Weights.IMAGENET1K_V1")
            else:
                backbone = anet()

            # load the weights
            state_dict = backbone.state_dict()
            # remove the last layer weights
            state_dict["classifier.6.weight"] = self.state_dict()["classifier.6.weight"]
            state_dict["classifier.6.bias"] = self.state_dict()["classifier.6.bias"]
            # load the weights
            self.load_state_dict(state_dict)

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
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
