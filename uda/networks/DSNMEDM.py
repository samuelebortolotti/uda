import torch.nn as nn
import torch
from torchvision.models import resnet18
from typing import Tuple
from .DSN import ResNet18Dec, Flatten, ReconstructionCode, ReconstructionSheme


class DSNMEDM(nn.Module):
    r"""
    MEDM DSN architecture.
    This implementation takes inspiration from the Domain Separation Networks
    by Konstantinos Bousmalis et al. and from the by Entropy Minimization vs. Diversity
    Maximization paper for Domain Adaptation et al. Xiaofu Wu.
    In practice, we aim to taake both principles behind the architectures, by employing the
    reconstruction loss to better enforce the encoders to learn better feature representation
    and as domain adaptation losses we have employed the `entropy` and the `divergence` loss
    as stated in the MEDM.

    **Note***: it is designed to work with `batch_size` equal to 32, which is the sample size
    chosen for the experiments of Konstantinos Bousmalis et al
    """

    def __init__(
        self,
        num_classes: int = 20,
        pretrained: bool = True,
        feature_extractor: bool = False,
    ) -> None:
        r"""
        Initialize the custom DSN+MEDM network architecture exploiting both the
        reconstruction loss and the MEDM loss.

        The code size is 128 as depicted in the paper.

        According to the paper the layer provided are:
        - private source encoder
        - private target encoder
        - shared encoder
        - shared decoder
        - task specific network (in our case it is the classifier)

        we remove the domain adversarial piece since we have decided to rely on the losses
        of MEDM so as to accomplish domain adaptation.

        Default:
        - num_classes [int] = 20
        - pretrained [bool] = True
        - feature_extractor [bool] = False

        Args:
        - num_classes [int]: first dimension for the resize
        - pretrained [bool]: whether to use pre-trained weights.
        - feature_extractor [bool]: whether to use the model only to extract the features
        """
        super(DSNMEDM, self).__init__()

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

        # indentical to the source_encoder except for the dropout
        self.shared_encoder = nn.Sequential(
            *nn.ModuleList(resnet18(pretrained=pretrained).children())[:-1],
            Flatten(),
        )

        # merge feature layer, as we have seen it works pretty well
        self.merge_features = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(inplace=True))

        # shared encoder network (the reverse of the ResNet)
        self.shared_decoder = nn.Sequential(
            ResNet18Dec(z_idim=512),
        )

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=num_classes), nn.Softmax()
        )

        # bottle neck layer
        self.bottle_neck = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        # in the forward return only the features!
        self.feature_extractor = feature_extractor

        # module to initialize
        init_modules = [self.classifier, self.bottle_neck]

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Forward method

        **Note**: in the case of `feature_extractor` set to `True` it returns
        only the features [torch.Tensor]

        Args:
        - x [torch.Tensor]: source sample
        - rec_code [ReconstructionCode]: which private features to employ
        - rec_scheme [ReconstructionSheme]: which scheme to use in order to merge
        the features for the reconstruction

        Returns:
        - class_pred [torch.Tensor]: class prediction
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

        bottle_neck_res = self.bottle_neck(shared_code)

        # get the class prediction
        class_pred = self.classifier(bottle_neck_res)

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
        # merged code is a function of shared and private code
        if rec_scheme == ReconstructionSheme.ONLY_PRIVATE_CODE:
            # overriding to private code
            merged_code = self.merge_features(
                torch.cat((private_code, private_code), 1)
            )
            # merged_code = private_code
        elif rec_scheme == ReconstructionSheme.BOTH_SHARED_AND_PRIVATE:
            # merging both shared and private
            merged_code = self.merge_features(torch.cat((shared_code, private_code), 1))
        elif rec_scheme == ReconstructionSheme.ONLY_SHARED_CODE:
            # shared code
            merged_code = self.merge_features(torch.cat((shared_code, shared_code), 1))
        else:
            raise Exception("No valid scheme selected")

        # shared decoder
        rec_input = self.shared_decoder(merged_code)

        return class_pred, private_code, shared_code, rec_input
