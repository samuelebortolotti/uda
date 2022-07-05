"""Networks module
This module contains all the networks architectures we have employed in our experiments
"""
from .AlexNet import AlexNet
from .ResNet18 import ResNet18
from .DANN import DANNAlexNet, DANNResNet18
from .DANNMEDM import DANNMEDM
from .DDC import DDCAlexNet, DDCResNet18
from .DSN import (
    AlexNetDSN,
    ResNet18Dec,
    ResNet18DSN,
    DiffLoss,
    SIMSE,
    ReconstructionCode,
    ReconstructionSheme,
    ResNet18DSNImproved,
)
from .DSNMEDM import DSNMEDM
from .MEDM import DANNMEDM, MEDM
from .Rot import RotationArch
