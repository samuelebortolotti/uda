import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from typing import Tuple
from enum import Enum
from uda.networks.DANN import GradientReversalFn

################### Loss functions ###################


class DiffLoss(nn.Module):
    r"""
    L difference loss depicted in the paper
    """

    def __init__(self) -> None:
        r"""
        Initialize the Difference Loss
        """
        super(DiffLoss, self).__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        r"""
        Forward method, computes the difference loss between the two inputs
        Taken from: https://github.com/fungtion/DSN/blob/master/functions.py

        Args:
        - input_1 [torch.Tensor]: source data representation
        - input_2 [torch.Tensor]: target data representation

        Returns:
        - diff_loss [torch.Tensor]: difference loss
        """
        batch_size = input1.size(0)
        input_1 = input1.view(batch_size, -1)
        input_2 = input2.view(batch_size, -1)

        # L2 norms
        input1_l2_norm = torch.norm(input_1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input_2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input_2) + 1e-6)

        # mean and squared
        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class MSE(nn.Module):
    r"""
    Mean squared error as depicted in the paper
    """

    def __init__(self) -> None:
        r"""
        Initialize the Mean squared error
        """
        super(MSE, self).__init__()

    def forward(self, pred: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        r"""
        Forward method, computes the MSE between prediction and real value
        Taken from: https://github.com/fungtion/DSN/blob/master/functions.py

        Args:
        - pred [torch.Tensor]: prediction
        - real [torch.Tensor]: real

        Returns:
        - mse [Torch.tensor]: mean squared error
        """
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):
    r"""
    Scale Invariant Mean Squared Error as depicted in the paper for reconstruction purposes
    """

    def __init__(self) -> None:
        r"""
        Initialize the SIMSE error
        """
        super(SIMSE, self).__init__()

    def forward(self, pred: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        r"""
        Forward method, computes the SIMSE between the two inputs
        Taken from: https://github.com/fungtion/DSN/blob/master/functions.py

        Args:
        - pred [torch.Tensor]: source data representation
        - real [torch.Tensor]: target data representation

        Returns:
        - simse [Torch.tensor]: scale invariant mean squared error
        """
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n**2)

        return simse


################### Reconstruction code enums ###################


class ReconstructionCode(Enum):
    r"""
    Enumerator employed in order to understand which encoder to use (source
    or target) to perform the image reconstruction using the decoder
    """
    SOURCE = 0
    TARGET = 1


class ReconstructionSheme(Enum):
    r"""
    Enumerator employed in order to understand how to merge the reconstruction
    codes either the shared code, the private or all of them.
    """
    ONLY_PRIVATE_CODE = 0
    BOTH_SHARED_AND_PRIVATE = 1
    ONLY_SHARED_CODE = 2


################### Custom nn.Module ###################


class Flatten(nn.Module):
    r"""
    Module employed in order to reshape the tensor with
    `torch.flatten(x)`.
    In the architecture it is employed so as to flatten a 4 dimensional
    tensor into a 2 dimensional one
    """

    def __init__(self, *args: str) -> None:
        r"""
        Initialize the Flatten module

        Args:
        - args [str]: additional arguments, they will be ignored
        """
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward method

        Args:
        - x [torch.Tensor]: source sample

        Returns:
        - x after the reshape
        """
        return torch.flatten(x, 1)


class Reshape3D(nn.Module):
    r"""
    Module employed in order to reshape a tensor into a 3 dimensional one
    """

    def __init__(self, dim1: int, dim2: int, dim3: int, *args: str) -> None:
        r"""
        Initialize the Reshape3D module

        Args:
        - dim1 [int]: first dimension for the resize
        - dim2 [int]: second dimension for the resize
        - dim3 [int]: third dimension for the resize
        - args [str]: additional arguments, they will be ignored
        """
        super(Reshape3D, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward method

        Args:
        - x [torch.Tensor]: source sample

        Returns:
        - x after the 3D resize
        """
        return x.view(-1, self.dim1, self.dim2, self.dim3)


################### DSN ###################


class DSN(nn.Module):
    r"""
    Customized DSN architecture.
    This implementation takes inspiration from the Domain Separation Networks
    by Konstantinos Bousmalis et al.
    In practice, we aim to reproduce the main principles behind the architecture
    proposed for the LineMod dataset. However, since we have images of size 224x224 which
    is different from both MNIST and LinMod, we have added 3 more convolutional layers
    and 3 more maxpooling in order to reduce the tensor sizes while exploiting the
    advantages of low dimensional kernels, otherwise for Colab flattening such
    features would have become impossible.

    **Note***: it is designed to work with `batch_size` equal to 32, which is the sample size
    chosen for the experiments of Konstantinos Bousmalis et al
    """

    def __init__(
        self,
        num_classes: int = 20,
        dropout: float = 0.5,
        kaiming: bool = False,
        feature_extractor: bool = False,
    ) -> None:
        r"""
        Initialize the custom DSN network architecture employed on Linemod data to ours.
        We have added 3 extra layers for the encoders and the decoders so as to match
        the image size and avoid memory problems in for the fc 128 output layers

        The code size is 128 as depicted in the paper.

        According to the paper the layer provided are:
        - private source encoder
        - private target encoder
        - shared encoder
        - shared decoder
        - task specific network (in our case it is the classifier)
        - domain adversarial network (to predict the domain)

        Default:
        - num_classes [int] = 20
        - kaiming [bool] = False
        - feature_extractor [bool] = False

        Args:
        - num_classes [int]: first dimension for the resize
        - dropout [float]: dropout probability, in the original network it is implemented
        in the shared encoder before the previous layer.
        - kaiming [bool]: whether to use the kaiming initialization on the convolutional
        and linear layers instead of the default initialization.
        - feature_extractor [bool]: whether to use the model only to extract the features
        """
        super(DSN, self).__init__()

        # In the original tensorfrow code: default_encoder
        self.source_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # The Conv and MaxPool layers below are added in order
            # to reduce the image size, so as to fit the actual image size
            nn.Conv2d(64, 128, kernel_size=3, padding=2),  # 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=2),  # 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=2),  # 3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            # output 32, 128
        )

        # identical to the source_encoder
        self.target_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            # output 32, 128
        )

        # indentical to the source_encoder except for the dropout
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.Dropout(p=dropout),  # present in the original implementation
            # output 32, 128
        )

        # The decoder aims to reconstruct the picture, thus its output
        # needs to convey to the original image size: [32, 3, 224, 224]
        # It is basically the reverse of the encoder
        # up to know it works, but it should not be that good since
        # the upsampling with 2 scale factor is not equal to the maxpool2d
        self.shared_decoder = nn.Sequential(
            Reshape3D(32, 4, 4),
            nn.Conv2d(32, 64, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(224, 224)),
            nn.Conv2d(32, 3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            # output 32, 128
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=num_classes),
        )

        # Domain adversarial network, implemented following the paper
        self.domain_adversarial_network = nn.Sequential(
            nn.Linear(in_features=512, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=2),
        )

        # Check whether we should just return the features
        self.feature_extractor = feature_extractor

        # module to initialize
        init_modules = [
            self.source_encoder,
            self.target_encoder,
            self.shared_decoder,
            self.classifier,
            self.domain_adversarial_network,
        ]

        # Init modules with kaiming initialization if requested
        if kaiming:
            for m in init_modules:
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        nn.init.constant_(m.bias.data, 0)

    def forward(
        self,
        x: torch.Tensor,
        rec_code: ReconstructionCode = ReconstructionCode.SOURCE,
        rec_scheme: ReconstructionSheme = ReconstructionSheme.ONLY_SHARED_CODE,
        grl_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward method

        **Note**: in the case of `feature_extractor` set to `True` it returns
        only the features [torch.Tensor]

        Args:
        - x [torch.Tensor]: source sample
        - rec_code [ReconstructionCode]: which private features to employ
        - rec_scheme [ReconstructionSheme]: which scheme to use in order to merge
        the features for the reconstruction
        - grl_lambda [float]: Gradient Reversal Layer (GRL) lambda parameter
        Since according to the paper it implements the DANN loss, using the same
        idea of adversarial scheme.

        Returns:
        - class_pred [torch.Tensor]: class prediction
        - domain_pred [torch.Tensor]: domain prediction for the DANN
        - private_code [torch.Tensor]: code coming from a private
        encoder either of the SOURCE or TARGET depends on the rec_code value
        - shared_code [torch.Tensor]: code coming from the shared encoder
        - rec_code [torch.Tensor]: reconstructed code
        """

        # shared encoder
        shared_code = self.shared_encoder(x)
        # get the class prediction
        class_pred = self.classifier(shared_code)
        # reverse the shared code in order to employ the reverse gradient
        rev_shared_code = GradientReversalFn.apply(shared_code, grl_lambda)
        # get the domain prediction
        domain_pred = self.domain_adversarial_network(rev_shared_code)

        # private encoder
        private_code = None
        if rec_code == ReconstructionCode.SOURCE:
            private_code = self.source_encoder(x)
        elif rec_code == ReconstructionCode.TARGET:
            private_code = self.target_encoder(x)
        else:
            raise Exception("No valid code selected")

        # shared decoder
        merged_code = None
        # clone is needed in order to avoid backpropagation issues, as the
        # merged code is a function of shared and private code
        if rec_scheme == ReconstructionSheme.ONLY_PRIVATE_CODE:
            # overriding to private code
            merged_code = private_code.clone()
        elif rec_scheme == ReconstructionSheme.BOTH_SHARED_AND_PRIVATE:
            # merging both shared and private
            merged_code = shared_code.clone()
            merged_code += private_code.clone()
        elif rec_scheme == ReconstructionSheme.ONLY_SHARED_CODE:
            merged_code = shared_code.clone()
        else:
            raise Exception("No valid scheme selected")

        # shared decoder
        rec_input = self.shared_decoder(merged_code)

        return class_pred, domain_pred, private_code, shared_code, rec_input


################### Decoder ResNet18 ###################


class ResizeConv2d(nn.Module):
    r"""
    ResNet18 module for the Upsample operation following a convolutional one
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, scale_factor, mode="nearest"
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockDec(nn.Module):
    r"""
    ResNet18 deconvolution block with residual connections
    """

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(
                in_planes, planes, kernel_size=3, scale_factor=stride
            )
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):
    r"""
    ResNet18 Deconvolutional architecture
    """

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_idim=10, nc=3):
        super().__init__()
        self.in_planes = 512
        self.linear = nn.Linear(z_idim, 512)
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=2)
        self.conv1 = ResizeConv2d(32, nc, kernel_size=3, scale_factor=3.5)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        # Sigmoid is employed since we are expecting to produce an image, thus
        # 3 channels with values within the interval [0, 1]
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 224, 224)
        return x


################### ResNet18 ###################


class ResNet18DSN(nn.Module):
    r"""
    Customized ResNet18 DSN architecture.
    This implementation takes inspiration from the Domain Separation Networks
    by Konstantinos Bousmalis et al.
    In practice, we aim to reproduce the main principles behind the architecture
    proposed for the LineMod dataset. However, since we have images of size 224x224 which
    is different from both MNIST and LinMod.
    Moreover, we have added 3 more convolutional layers and 3 more maxpooling in order to reduce
    the thensor sizes while exploiting the advantages of low dimensional kernels, otherwise
    for Colab flatten such features would have become impossible.

    **Note***: it is designed to work with `batch_size` equal to 32, which is the sample size
    chosen for the experiments of Konstantinos Bousmalis et al
    """

    def __init__(
        self,
        num_classes: int = 20,
        pretrained: bool = False,
        feature_extractor: bool = False,
        dropout: float = 0.5,
    ) -> None:
        r"""
        Initialize the custom DSN network architecture employed on Linemod data to ours.
        We have added 3 extra layers for the encoders and the decoders so as to match
        the image size and avoid memory problems in for the fc 128 output layers

        The code size is 128 as depicted in the paper.

        According to the paper the layer provided are:
        - private source encoder
        - private target encoder
        - shared encoder
        - shared decoder
        - task specific network (in our case it is the classifier)
        - domain adversarial network (to predict the domain)

        Default:
        - num_classes [int] = 20
        - pretrained [bool] = False
        - feature_extractor [bool] = False
        - dropout [float] = 0.5

        Args:
        - num_classes [int]: first dimension for the resize
        - dropout [float]: dropout probability, in the original network it is implemented
        in the shared encoder before the previous layer.
        - pretrained [bool]: whether to use the pretrained weights for the encoder
        - feature_extractor [bool]: whether to use the model only to extract the features
        """
        super(ResNet18DSN, self).__init__()

        # Take the resNet18 module and discard the last layer
        features = nn.ModuleList(resnet18(pretrained=pretrained).children())[:-1]

        # In the original tensorfrow code: default_encoder
        self.source_encoder = nn.Sequential(
            *features,
            Flatten(),
        )

        # identical to the source_encoder
        self.target_encoder = nn.Sequential(
            *features,
            Flatten(),
        )

        # indentical to the source_encoder except for the dropout
        self.shared_encoder = nn.Sequential(
            *features,
            Flatten(),
            nn.Dropout(p=dropout),  # present in the original implementation
        )

        self.shared_decoder = nn.Sequential(
            ResNet18Dec(z_idim=512),
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=num_classes),
        )

        # domain adversarial network according to the paper
        self.domain_adversarial_network = nn.Sequential(
            nn.Linear(in_features=512, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=2),
        )

        # in the forward return only the features!
        self.feature_extractor = feature_extractor

        # module to initialize
        init_modules = [self.classifier, self.domain_adversarial_network]

        # init modules
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(
        self,
        x: torch.Tensor,
        rec_code: ReconstructionCode = ReconstructionCode.SOURCE,
        rec_scheme: ReconstructionSheme = ReconstructionSheme.ONLY_SHARED_CODE,
        grl_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward method

        **Note**: in the case of `feature_extractor` set to `True` it returns
        only the features [torch.Tensor]

        Args:
        - x [torch.Tensor]: source sample
        - rec_code [ReconstructionCode]: which private features to employ
        - rec_scheme [ReconstructionSheme]: which scheme to use in order to merge
        the features for the reconstruction
        - grl_lambda [float]: Gradient Reversal Layer (GRL) lambda parameter
        Since according to the paper it implements the DANN loss, using the same
        idea of adversarial scheme.

        Returns:
        - class_pred [torch.Tensor]: class prediction
        - domain_pred [torch.Tensor]: domain prediction for the DANN
        - private_code [torch.Tensor]: code coming from a private
        encoder either of the SOURCE or TARGET depends on the rec_code value
        - shared_code [torch.Tensor]: code coming from the shared encoder
        - rec_code [torch.Tensor]: reconstructed code
        """

        # shared encoder
        shared_code = self.shared_encoder(x)
        # return the features if the feature extractor flag is set
        if self.feature_extractor:
            return shared_code
        # get the class prediction
        class_pred = self.classifier(shared_code)
        # reverse the shared code in order to employ the reverse gradient
        rev_shared_code = GradientReversalFn.apply(shared_code, grl_lambda)
        # get the domain prediction
        domain_pred = self.domain_adversarial_network(rev_shared_code)

        # private encoder
        private_code = None
        if rec_code == ReconstructionCode.SOURCE:
            private_code = self.source_encoder(x)
        elif rec_code == ReconstructionCode.TARGET:
            private_code = self.target_encoder(x)
        else:
            raise Exception("No valid code selected")

        # shared decoder
        merged_code = None
        # clone is needed in order to avoid backpropagation issues, as the
        # merged code is a function of shared and private code
        if rec_scheme == ReconstructionSheme.ONLY_PRIVATE_CODE:
            # overriding to private code
            merged_code = private_code.clone()
        elif rec_scheme == ReconstructionSheme.BOTH_SHARED_AND_PRIVATE:
            # merging both shared and private
            merged_code = shared_code.clone()
            merged_code += private_code.clone()
        elif rec_scheme == ReconstructionSheme.ONLY_SHARED_CODE:
            merged_code = shared_code.clone()
        else:
            raise Exception("No valid scheme selected")

        # shared decoder
        rec_input = self.shared_decoder(merged_code)

        return class_pred, domain_pred, private_code, shared_code, rec_input


################### AlexNet ###################


class AlexNetDSN(nn.Module):
    r"""
    Customized AlexNet DSN architecture.
    This implementation takes inspiration from the Domain Separation Networks
    by Konstantinos Bousmalis et al.
    In practice, we aim to reproduce the main principles behind the architecture
    proposed for the LineMod dataset. However, since we have images of size 224x224 which
    is different from both MNIST and LinMod.
    Moreover, we have added 3 more convolutional layers and 3 more maxpooling in order to reduce
    the thensor sizes while exploiting the advantages of low dimensional kernels, otherwise
    for Colab flatten such features would have become impossible.

    **Note***: it is designed to work with `batch_size` equal to 32, which is the sample size
    chosen for the experiments of Konstantinos Bousmalis et al
    """

    def __init__(
        self,
        num_classes: int = 20,
        dropout: float = 0.5,
        pretrained: bool = True,
        feature_extractor: bool = False,
    ) -> None:
        r"""
        Initialize the custom DSN network architecture employed on Linemod data to ours.
        We have added 3 extra layers for the encoders and the decoders so as to match
        the image size and avoid memory problems in for the fc 128 output layers

        The code size is 128 as depicted in the paper.

        According to the paper the layer provided are:
        - private source encoder
        - private target encoder
        - shared encoder
        - shared decoder
        - task specific network (in our case it is the classifier)
        - domain adversarial network (to predict the domain)

        Default:
        - num_classes [int] = 20
        - pretrained [bool] = True
        - feature_extractor [bool] = False

        Args:
        - num_classes [int]: first dimension for the resize
        - dropout [float]: dropout probability, in the original network it is implemented
        in the shared encoder before the previous layer.
        - pretrained [bool]: whether the network is pre-trained or not
        - feature_extractor [bool]: whether to use the model only to extract the features
        """
        super(AlexNetDSN, self).__init__()

        self.source_encoder = nn.Sequential(
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

        self.target_encoder = nn.Sequential(
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

        # indentical to the source_encoder except for the dropout
        self.shared_encoder = nn.Sequential(
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

        self.mid_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
        )

        # The decoder aims to reconstruct the picture, thus its output
        # needs to convey to the original image size: [32, 3, 224, 224]
        # Here we employ the ResNet one for the sake of simplicity and reuse
        # of component
        self.shared_decoder = nn.Sequential(
            ResNet18Dec(z_idim=512),
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=num_classes),
        )

        # domain adversarial network according to the paper
        self.domain_adversarial_network = nn.Sequential(
            nn.Linear(in_features=4096, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=2),
        )

        # merge features layer
        self.merge_features = nn.Sequential(nn.Linear(8192, 512), nn.ReLU(inplace=True))

        # in the forward return only the features!
        self.feature_extractor = feature_extractor

        # module to initialize
        init_modules = [
            self.classifier,
            self.domain_adversarial_network,
            self.shared_decoder,
            self.mid_classifier,
            self.merge_features,
        ]

        # init modules
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

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
                if key.split(".")[0] in ["source_encoder"]:
                    print("Adding key", key, "in current params")
                    current_params[key] = pretrained_params[
                        key.replace("source_encoder", "features")
                    ]
                elif key.split(".")[0] in ["target_encoder"]:
                    print("Adding key", key, "in current params")
                    current_params[key] = pretrained_params[
                        key.replace("target_encoder", "features")
                    ]
                elif key.split(".")[0] in ["shared_encoder"]:
                    print("Adding key", key, "in current params")
                    current_params[key] = pretrained_params[
                        key.replace("shared_encoder", "features")
                    ]
            # load weigths
            self.load_state_dict(current_params)

    def forward(
        self,
        x: torch.Tensor,
        rec_code: ReconstructionCode = ReconstructionCode.SOURCE,
        rec_scheme: ReconstructionSheme = ReconstructionSheme.ONLY_SHARED_CODE,
        grl_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward method

        **Note**: in the case of `feature_extractor` set to `True` it returns
        only the features [torch.Tensor]

        Args:
        - x [torch.Tensor]: source sample
        - rec_code [ReconstructionCode]: which private features to employ
        - rec_scheme [ReconstructionSheme]: which scheme to use in order to merge
        the features for the reconstruction
        - grl_lambda [float]: Gradient Reversal Layer (GRL) lambda parameter
        Since according to the paper it implements the DANN loss, using the same
        idea of adversarial scheme.

        Returns:
        - class_pred [torch.Tensor]: class prediction
        - domain_pred [torch.Tensor]: domain prediction for the DANN
        - private_code [torch.Tensor]: code coming from a private
        encoder either of the SOURCE or TARGET depends on the rec_code value
        - shared_code [torch.Tensor]: code coming from the shared encoder
        - rec_code [torch.Tensor]: reconstructed code
        """

        # shared encoder
        shared_code = self.mid_classifier(self.shared_encoder(x))
        # return the features if the feature extractor flag is set
        if self.feature_extractor:
            return shared_code
        # get the class prediction
        class_pred = self.classifier(shared_code)
        # reverse the shared code in order to employ the reverse gradient
        rev_shared_code = GradientReversalFn.apply(shared_code, grl_lambda)
        # get the domain prediction
        domain_pred = self.domain_adversarial_network(rev_shared_code)

        # private encoder
        private_code = None
        if rec_code == ReconstructionCode.SOURCE:
            private_code = self.mid_classifier(self.source_encoder(x))
        elif rec_code == ReconstructionCode.TARGET:
            private_code = self.mid_classifier(self.target_encoder(x))
        else:
            raise Exception("No valid code selected")

        # shared decoder
        merged_code = None
        # clone is needed in order to avoid backpropagation issues, as the
        # merged code is a function of shared and private code
        if rec_scheme == ReconstructionSheme.ONLY_PRIVATE_CODE:
            # overriding to private code
            merged_code = self.merge_features(
                torch.cat((private_code.clone(), private_code.clone()), 1)
            )
        elif rec_scheme == ReconstructionSheme.BOTH_SHARED_AND_PRIVATE:
            merged_code = self.merge_features(
                torch.cat((shared_code.clone(), private_code.clone()), 1)
            )
        elif rec_scheme == ReconstructionSheme.ONLY_SHARED_CODE:
            merged_code = self.merge_features(
                torch.cat((shared_code.clone(), shared_code.clone()), 1)
            )
        else:
            raise Exception("No valid scheme selected")

        rec_input = self.shared_decoder(merged_code)

        return class_pred, domain_pred, private_code, shared_code, rec_input


################### ResNet18 but improved with respect to our task ###################


class ResNet18DSNImproved(nn.Module):
    r"""
    Improved ResNet18 DSN architecture.
    This implementation takes inspiration from the Domain Separation Networks
    by Konstantinos Bousmalis et al.
    In practice, we aim to reproduce the main principles behind the architecture
    proposed for the LineMod dataset. However, since we have images of size 224x224 which
    is different from both MNIST and LinMod.
    Moreover, we have added 3 more convolutional layers and 3 more maxpooling in order to reduce
    the thensor sizes while exploiting the advantages of low dimensional kernels, otherwise
    for Colab flatten such features would have become impossible.

    Moreover, we have changed the way in which the final image has been reconstructed,
    since we have proven it provides better results for our tasks of domain adaptation

    **Note***: it is designed to work with `batch_size` equal to 32, which is the sample size
    chosen for the experiments of Konstantinos Bousmalis et al
    """

    def __init__(
        self,
        decoder_location: str,
        num_classes: int = 20,
        pretrained: bool = True,
        feature_extractor: bool = False,
        dropout: float = 0.5,
    ) -> None:
        r"""
        Initialize the custom DSN network architecture employed on Linemod data to ours.
        We have added 3 extra layers for the encoders and the decoders so as to match
        the image size and avoid memory problems in for the fc 128 output layers

        The code size is 128 as depicted in the paper.

        According to the paper the layer provided are:
        - private source encoder
        - private target encoder
        - shared encoder
        - shared decoder
        - task specific network (in our case it is the classifier)
        - domain adversarial network (to predict the domain)

        Default:
        - num_classes [int] = 20
        - pretrained [bool] = False
        - feature_extractor [bool] = False
        - pretrained [bool] = True

        Args:
        - decoder_location [str]: pre-trained decoder weights
        - num_classes [int]: first dimension for the resize
        - dropout [float]: dropout probability, in the original network it is implemented
        in the shared encoder before the previous layer.
        - pretrained [bool]: whether to use the pretrained network or not
        - feature_extractor [bool]: whether to use the model only to extract the features
        """
        super(ResNet18DSNImproved, self).__init__()

        # In the original tensorfrow code: default_encoder
        self.source_encoder = nn.Sequential(
            *nn.ModuleList(resnet18(pretrained=pretrained).children())[:-1],
            Flatten(),
        )

        # identical to the source_encoder
        self.target_encoder = nn.Sequential(
            *nn.ModuleList(resnet18(pretrained=pretrained).children())[:-1],
            Flatten(),
        )

        # indentical to the source_encoder
        self.shared_encoder = nn.Sequential(
            *nn.ModuleList(resnet18(pretrained=pretrained).children())[:-1],
            Flatten(),
        )

        # layer which defines the function `h` defined before, namely the map
        # to a lower dimensional embedding
        self.merge_features = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))

        # shared decoder, namely the deconvolutional ResNet18
        self.shared_decoder = nn.Sequential(
            ResNet18Dec(z_idim=512),
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=num_classes),
        )

        # domain adversarial network part
        self.domain_adversarial_network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

        # domain adversarial network according to the paper
        self.bottle_neck = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # in the forward return only the features!
        self.feature_extractor = feature_extractor

        # module to initialize
        init_modules = [
            self.classifier,
            self.bottle_neck,
            self.domain_adversarial_network,
        ]

        # init modules
        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

        # load weights of decoder
        current_params = self.state_dict()
        # decoder we have trained for 200 epochs
        decoder_state_dict = torch.load(decoder_location)
        for key, value in current_params.items():
            if (
                key.split(".")[0] == "shared_decoder"
                or key.split(".")[0] == "merge_features"
            ):
                print("Adding ", key)
                current_params[key] = decoder_state_dict[key]
        self.load_state_dict(current_params)

    def forward(
        self,
        x: torch.Tensor,
        rec_code: ReconstructionCode = ReconstructionCode.SOURCE,
        rec_scheme: ReconstructionSheme = ReconstructionSheme.ONLY_SHARED_CODE,
        grl_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward method

        **Note**: in the case of `feature_extractor` set to `True` it returns
        only the features [torch.Tensor]

        Args:
        - x [torch.Tensor]: source sample
        - rec_code [ReconstructionCode]: which private features to employ
        - rec_scheme [ReconstructionSheme]: which scheme to use in order to merge
        the features for the reconstruction
        - grl_lambda [float]: Gradient Reversal Layer (GRL) lambda parameter
        Since according to the paper it implements the DANN loss, using the same
        idea of adversarial scheme.

        Returns:
        - class_pred [torch.Tensor]: class prediction
        - domain_pred [torch.Tensor]: domain prediction for the DANN
        - private_code [torch.Tensor]: code coming from a private
        encoder either of the SOURCE or TARGET depends on the rec_code value
        - shared_code [torch.Tensor]: code coming from the shared encoder
        - rec_code [torch.Tensor]: reconstructed code
        """

        # shared encoder
        shared_code = self.shared_encoder(x)
        # return the features if the feature extractor flag is set
        if self.feature_extractor:
            return shared_code

        # bottle_neck
        bottle_neck_res = self.bottle_neck(shared_code)

        # get the class prediction
        class_pred = self.classifier(bottle_neck_res)
        # reverse the shared code in order to employ the reverse gradient
        rev_shared_code = GradientReversalFn.apply(bottle_neck_res, grl_lambda)
        # get the domain prediction
        domain_pred = self.domain_adversarial_network(rev_shared_code)

        # private encoder
        private_code = None
        if rec_code == ReconstructionCode.SOURCE:
            private_code = self.source_encoder(x)
        elif rec_code == ReconstructionCode.TARGET:
            private_code = self.target_encoder(x)
        else:
            raise Exception("No valid code selected")

        # shared decoder
        merged_code = None
        # clone is needed in order to avoid backpropagation issues, as the
        # merged code is a function of shared and private code
        if rec_scheme == ReconstructionSheme.ONLY_PRIVATE_CODE:
            # overriding to private code
            merged_code = self.merge_features(
                torch.cat((private_code, private_code), 1)
            )
        elif rec_scheme == ReconstructionSheme.BOTH_SHARED_AND_PRIVATE:
            # merging both shared and private
            merged_code = self.merge_features(torch.cat((shared_code, private_code), 1))
        elif rec_scheme == ReconstructionSheme.ONLY_SHARED_CODE:
            # overriding the merge code
            merged_code = self.merge_features(torch.cat((shared_code, shared_code), 1))
        else:
            raise Exception("No valid scheme selected")

        # shared decoder output
        rec_input = self.shared_decoder(merged_code)

        return class_pred, domain_pred, private_code, shared_code, rec_input
