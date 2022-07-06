import torch
import torch.nn as nn
from uda import Technique
from uda.networks.DSN import ReconstructionCode, ReconstructionSheme
from typing import Tuple
import tqdm
import torch.nn.functional as F


def tr_image(img: torch.Tensor) -> torch.Tensor:
    r"""
    Function which computes the average of the image, to better visualize it
    in Tensor board

    Args:

    - img [torch.Tensor]: image

    Returns:
    - image after the processing [torch.Tensor]
    """
    return (img + 1) / 2


def test_step(
    net: nn.Module,
    source_validation_loader: torch.utils.data.DataLoader,
    cost_function: nn.CrossEntropyLoss,
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter,
    title: str,
    technique: Technique,
    rec_code: ReconstructionCode,
    device="cuda",
) -> Tuple[float, float]:
    r"""
    Function which performs one test step, it is able to adapt to
    different architectures according to the `technique` passed.

    Namely:

    - Source-Only (baseline): learns only from the source domain
    - Upper-Bound: learns both from the source and the target domain
    - Deep Domain Confusion: https://arxiv.org/pdf/1412.3474.pdf
    - Domain-Adversarial Neural Network: https://arxiv.org/pdf/1505.07818.pdf
    - Domain Separation Networks: https://arxiv.org/pdf/1608.06019.pdf
    - Entropy Minimization vs. Diversity Maximization for Domain Adaptation: https://arxiv.org/pdf/2002.01690.pdf
    - Domain Separation Networks with Entropy Minimization vs. Diversity Maximization

    Args:

    - net [nn.Module]: network architecture
    - source_validation_loader [torch.utils.data.DataLoader]: source validation data loader
    - cost_function [nn.CrossEntropyLoss]: cost function
    - epoch [int]: current epoch
    - writer [torch.utils.tensorboard.SummaryWriter]: summary writer
    - title [str]: title for the `tqdm` loop
    - technique [Technique]: which training technique to employ, of course the neural network needs to be adapted according to the chosen technique
    - device [str] = "cuda": device on which to perform the training

    Returns:

    - batch loss, batch accuracy [Tuple[float, float]]
    """
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # set the network to evaluation mode
    net.eval()

    # disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
    with torch.no_grad():

        # iterate over the test set
        for batch_idx, (inputs, targets) in tqdm.tqdm(
            enumerate(source_validation_loader), desc=title
        ):

            # load data into device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward pass
            if technique == Technique.DDC:
                outputs, mmd_loss = net(inputs, targets)
            elif technique == Technique.DANN:
                outputs, _ = net(inputs)
            elif technique == Technique.DSN:
                # class prediction
                outputs, _, _, _, reconstructed_code = net(inputs, rec_code=rec_code)

                # image reconstructions
                rec_img_share = tr_image(reconstructed_code.data)
                _, _, _, _, reconstructed_code = net(
                    inputs,
                    rec_code=rec_code,
                    rec_scheme=ReconstructionSheme.BOTH_SHARED_AND_PRIVATE,
                )
                rec_img_all = tr_image(reconstructed_code.data)
                _, _, _, _, reconstructed_code = net(
                    inputs,
                    rec_code=rec_code,
                    rec_scheme=ReconstructionSheme.ONLY_PRIVATE_CODE,
                )
                rec_img_private = tr_image(reconstructed_code.data)

                # log images
                # log_images(writer, rec_img_share, epoch, 'DSN share images')
                # log_images(writer, rec_img_all, epoch, 'DSN all images')
                # log_images(writer, rec_img_private, epoch, 'DSN private images')
                # log_images(writer, tr_image(inputs), epoch, 'Original')
            elif technique == Technique.ROTATION or technique == Technique.DANN_MEDM:
                # outputs, _, _ = net(inputs)
                outputs, _ = net(inputs)
            elif technique == Technique.DSN_MEDM:
                # class prediction
                outputs, _, _, reconstructed_code = net(inputs, rec_code=rec_code)

                # image reconstructions
                rec_img_share = tr_image(reconstructed_code.data)
                _, _, _, reconstructed_code = net(
                    inputs,
                    rec_code=rec_code,
                    rec_scheme=ReconstructionSheme.BOTH_SHARED_AND_PRIVATE,
                )
                rec_img_all = tr_image(reconstructed_code.data)
                _, _, _, reconstructed_code = net(
                    inputs,
                    rec_code=rec_code,
                    rec_scheme=ReconstructionSheme.ONLY_PRIVATE_CODE,
                )
                rec_img_private = tr_image(reconstructed_code.data)
            else:
                outputs = net(inputs)

            # loss computation
            loss = cost_function(outputs, targets)
            if technique == Technique.DDC:
                loss += 0.25 * mmd_loss
            elif technique == Technique.MEDM or technique == Technique.DANN_MEDM:
                loss = F.nll_loss(outputs.log(), targets, size_average=False)

            # fetch prediction and loss value
            samples += inputs.shape[0]
            cumulative_loss += (
                loss.item()
            )  # Note: the .item() is needed to extract scalars from tensors
            _, predicted = outputs.max(1)

            # compute accuracy
            cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples, cumulative_accuracy / samples * 100
