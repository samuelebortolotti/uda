"""
Main module of the `uda` project
"""
from enum import Enum


class Technique(Enum):
    r"""
    Technique enumerator.
    It is employed for letting understand which method to use
    in order to train the neural network
    """
    SOURCE_ONLY = 1  # Train using only the source domain
    UPPER_BOUND = 2  # Train using the target domain too, to fix an upper bound
    DDC = 3  # Deep Domain Confusion
    DANN = 4  # Domain-Adversarial Neural Network
    DSN = 5  # Domain Separation Network
    ROTATION = 6  # Rotation Loss
    MEDM = 7  # Entropy Minimization vs. Diversity Maximization
    DANN_MEDM = 8  # DANN with Entropy Minimization vs. Diversity Maximization
    DSN_MEDM = 9  # DSN with MEDM
