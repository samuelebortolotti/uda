import torch
import torch.nn as nn
from typing import Tuple
from torch.autograd import Function
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet18

# Autograd Function objects are what record operation history on tensors,
# and define formulas for the forward and backprop.


class GradientReversalFn(Function):
    r"""
    Reverse gradient layer as described in Domain-Adversarial Training of Neural Networks
    paper by Yaroslav Ganin et al.

    Implementation taken from:
    [link]: https://nbviewer.org/github/vistalab-technion/cs236605-tutorials/blob/master/tutorial6/tutorial6-TL_DA.ipynb
    """
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x: torch.Tensor, grl_lambda: float) -> torch.Tensor:
        r"""Forward method, the aim is to store the GRL lambda parameter

        Args:
        - ctx: context
        - x [torch.Tensor]: source sample
        - grl_lambda [float]: Gradient Reversal Layer (GRL) lambda, which is a
        parameter described in the Domain-Adversarial Training of Neural Networks
        paper

        Returns:
        - x [torch.Tensor]: forward as it is
        """
        # Store context for backprop
        ctx.grl_lambda = grl_lambda

        # Forward pass is a no-op
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        r"""
        Backward method, the aim is to backpropagate the negative value
        of the gradient

        Args:
        - ctx: context
        - grad_out [torch.Tensor]: gradient

        Returns:
        - output, None [Tuple[torch.Tensor, None]]: reversed gradient and None
        """
        # Backward pass is just to -grl_lambda the gradient
        output = grad_output.neg() * ctx.grl_lambda

        # Must return same number as inputs to forward()
        return output, None


################### AlexNet ###################


class DANNAlexNet(nn.Module):
    r"""
    Customized AlexNet architecture as described in Domain-Adversarial Training of Neural Networks
    paper by Yaroslav Ganin et al.
    """

    def __init__(
        self,
        num_classes: int = 20,
        dropout: float = 0.5,
        pretrained: bool = False,
        feature_extractor: bool = False,
    ) -> None:
        r"""
        Initialize the custom DANNAlexNet model which implements
        the Domain-Adversarial Neural Network approach approach

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
        super(DANNAlexNet, self).__init__()
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
        # As adviced by the paper, we follow the previous structure
        # for deep domain confusion, adding an adapt layer before the classifiers
        self.adapt = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(nn.Linear(2048, num_classes))
        # Domain classifier as described in the paper
        self.domain_classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 2),
        )

        # As usual the mode for returning the features instead of the
        # predictions
        self.feature_extractor = feature_extractor

        # Load the pretrained model if pretrained is set to True
        if pretrained:
            from torchvision.models.alexnet import model_urls

            # load AlexNet pretrained weights on ImageNet
            state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=True)

            # load weights from alexnet base net
            current_params = self.state_dict()

            # As for the Deep Domain Confusion network we load the common
            # parameters
            for key, value in current_params.items():
                # the common layers between the two nets are features->features and classifier->mid_classifier
                if key.split(".")[0] == "features":
                    print("Adding key", key, "in current params")
                    current_params[key] = state_dict[key]
                if key.split(".")[0] == "mid_classifier":
                    print("Adding key", key, "in current params")
                    current_params[key] = state_dict[
                        key.replace("mid_classifier", "classifier")
                    ]
            self.load_state_dict(current_params)

    def forward(
        self, x: torch.Tensor, grl_lambda: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward method

        **Note**: in the case of `feature_extractor` set to `True` it returns
        only the features [torch.Tensor]

        Args:
        - x [torch.Tensor]: source sample
        - grl_lambda [float]: Gradient Reversal Layer (GRL) lambda parameter

        Returns:
        - class_prediction, domain_prediction [Tuple[torch.Tensor, float]]:
        class prediction and domain prediction
        """
        x = self.features(x)
        # if feature extractor then return the features only
        if self.feature_extractor:
            return x
        x = self.avgpool(x)

        # extract the features
        features = x.view(x.size(0), -1)
        features = self.mid_classifier(features)
        features = self.adapt(features)

        # gradient reversal
        reverse_features = GradientReversalFn.apply(features, grl_lambda)

        # classify both the domain and the class
        class_pred = self.classifier(features)
        domain_pred = self.domain_classifier(reverse_features)
        return class_pred, domain_pred


################### ResNet18 ###################


class DANNResNet18(nn.Module):
    r"""
    Customized DANNResNet18 architecture.
    This implementation takes inspiration from the Domain-Adversarial Training
    of Neural Networks paper by Yaroslav Ganin et al.
    In practice, we aim to reproduce the principles of such architecture even if
    we have not widely explored the architecture like Yaroslav Ganin et al.
    """

    def __init__(
        self,
        num_classes: int = 20,
        pretrained: bool = False,
        feature_extractor: bool = False,
        dropout: float = 0.5,
    ) -> None:
        r"""
        Initialize the custom DANNResNet18 model which aim to simulate
        the Domain-Adversarial Neural Network approach approach

        Default:
        - num_classes [int] = 20
        - pretrained [bool] = False
        - feature_extractor [bool] = False
        - dropout [float] = 0.5

        Args:
        - num_classes [int]: number of classes [used in the last layer]
        - dropout [float]: probability of dropout [value between 0-1]
        - pretrained [bool]: whether to pretrain the model or not
        - feature_extractor [bool]: whether to return the feature extracted from
        the neural network or not in the forward step.
        - dropout [float]: probability of dropout [value between 0-1]
        """
        super(DANNResNet18, self).__init__()

        # Take the resNet18 module and discard the last layer
        features = nn.ModuleList(resnet18(pretrained=pretrained).children())[:-1]

        # Use it as a feature extractor
        self.features = nn.Sequential(*features)

        # As adviced by the paper, we follow the previous structure
        # for deep domain confusion, adding an adapt layer before the classifiers
        self.adapt = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # final fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(256, num_classes),
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
        )

        # in the forward return only the features!
        self.feature_extractor = feature_extractor

    def forward(
        self, x: torch.Tensor, grl_lambda: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward method

        **Note**: in the case of `feature_extractor` set to `True` it returns
        only the features [torch.Tensor]

        Args:
        - x [torch.Tensor]: source sample
        - grl_lambda [float]: Gradient Reversal Layer (GRL) lambda parameter

        Returns:
        - class_prediction, domain_prediction [Tuple[torch.Tensor, float]]:
        class prediction and domain prediction
        """
        x = self.features(x)
        # if feature extractor then return the features only
        if self.feature_extractor:
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
