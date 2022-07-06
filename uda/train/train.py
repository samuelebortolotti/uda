import torch
import torch.nn as nn
import numpy as np
from uda.networks.DSN import DiffLoss, SIMSE, ReconstructionCode, ReconstructionSheme
from uda import Technique
from typing import Tuple
import tqdm
from uda.schedulers import dann_optimizer_scheduler
import torch.nn.functional as F


def training_step(
    net: nn.Module,
    source_training_loader: torch.utils.data.DataLoader,
    target_training_loader: torch.utils.data.DataLoader,
    rotation_training_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    cost_function: nn.CrossEntropyLoss,
    cost_function_domain: nn.CrossEntropyLoss,
    epoch: int,
    total_epochs: int,
    writer: torch.utils.tensorboard.SummaryWriter,
    current_step: int,
    active_domain_loss_step: int,
    loss_diff: DiffLoss,
    loss_recon1: SIMSE,
    loss_recon2: SIMSE,
    gamma_weight: float,
    beta_weight: float,
    alpha_weight: float,
    learning_rate: float,
    lr_decay_step: int,
    step_decay_weight: float,
    title: str,
    technique: Technique,
    device: str = "cuda",
) -> Tuple[float, float]:
    r"""
    Function which performs one training step, it is able to adapt to
    different architectures according to the technique passed.

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
    - source_training_loader [torch.utils.data.DataLoader]: source training data loader
    - target_training_loader torch.utils.data.DataLoader]: target training data loader
    - optimizer [torch.optim.Optimizer]: optimizer
    - scheduler [torch.optim.lr_scheduler.ReduceLROnPlateau]: scheduler
    - cost_function [nn.CrossEntropyLoss]: cost function
    - cost_function_domain [nn.CrossEntropyLoss]: cost function for the domain classifier in the case of the Domain-Adversarial Neural NetworkNetwork
    - epoch [int]: current epoch
    - total_epochs [int]: total number of epoch
    - writer [torch.utils.tensorboard.SummaryWriter]: summary writer
    - current_step [int]: current step (total number of batches analyzed), for DSN
    - active_domain_loss_step [int]: amount of steps after which to employ the domain adaptation method, for DSN
    - loss_diff [DiffLoss]: difference loss for the DSN architecture
    - loss_recon1 [SIMSE]: first reconstruction loss for DSN
    - loss_recon2 [SIMSE]: second reconstruction loss for DSN
    - gamma_weight [float]: gamma parameter in the DSN loss function
    - beta_weight [float]: beta parameter in the DSN loss function
    - alpha_weight [float]: alpha parameter in the DSN loss function
    - learning_rate [float]: learning rate
    - lr_decay_step [int]: learning rate decay step for the scheduler of the DSN
    - step_decay_weight [float]: learning rate weight decay step for the scheduler of the DSN
    - title [str]: title for the tqdm loop
    - technique [Technique]: which training technique to employ, of course the neural network needs to be adapted according to the chosen technique
    - device [str] = "cuda": device on which to perform the training

    Returns:

    - batch loss, batch accuracy [Tuple[float, float]]
    """
    samples = 0.0
    samples_domain_class = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0
    cumulative_accuracy_domain_class = 0.0

    # set the network to training mode
    net.train()

    # training target iterator
    target_iter = iter(target_training_loader)

    # Iterator for the rotation dataloader
    rotation_training_iter = iter(rotation_training_loader)

    # iterate over the training set
    for batch_idx, (inputs, label) in tqdm.tqdm(
        enumerate(source_training_loader), desc=title
    ):

        # Load target data
        try:
            target_data, _ = next(target_iter)
        except:
            target_iter = iter(target_training_loader)
            target_data, _ = next(target_iter)

        # Load rotation data
        if technique == Technique.ROTATION:
            try:
                target_data, _, rotation_label = next(rotation_training_iter)
            except:
                rotation_training_iter = iter(rotation_training_loader)
                target_data, _, rotation_label = next(rotation_training_iter)

        # load data into device
        inputs = inputs.to(device)
        label = label.to(device)
        if technique == Technique.ROTATION:
            rotation_label = rotation_label.to(device)
        target_data = target_data.to(device)

        # Training progress and GRL lambda computation
        p = (
            float(batch_idx + epoch * len(source_training_loader))
            / total_epochs
            / len(source_training_loader)
        )
        lambda_parameter = 2.0 / (1.0 + np.exp(-10 * p)) - 1

        if technique == Technique.DANN:
            # Set the learning rate according to the DANN paper
            dann_optimizer_scheduler(optimizer, p)

        # gradients reset
        optimizer.zero_grad()

        # forward pass [it changes according to the technique we are employing]
        if technique == Technique.DDC:
            outputs, mmd_loss = net(inputs, target_data)
        elif technique == Technique.DANN:
            # Get classifier outputs and domain classifier
            outputs, domain_prediction_source = net(inputs, lambda_parameter)
            # Get the domain prediction target
            _, domain_prediction_target = net(target_data, lambda_parameter)
        elif technique == Technique.DSN:
            # set lambda only in dann mode
            lambda_dsn = (
                lambda_parameter if current_step > active_domain_loss_step else 0
            )

            # source prediction
            (
                outputs,
                source_domain_pred,
                source_private_code,
                source_shared_code,
                source_rec_code,
            ) = net(
                inputs,
                rec_code=ReconstructionCode.SOURCE,
                rec_scheme=ReconstructionSheme.BOTH_SHARED_AND_PRIVATE,
                grl_lambda=lambda_dsn,
            )

            # target prediction
            (
                _,
                target_domain_pred,
                target_private_code,
                target_shared_code,
                target_rec_code,
            ) = net(
                target_data,
                rec_code=ReconstructionCode.TARGET,
                rec_scheme=ReconstructionSheme.BOTH_SHARED_AND_PRIVATE,
                grl_lambda=lambda_dsn,
            )
        elif technique == Technique.ROTATION:
            # Source domain prediction
            outputs, _ = net(inputs)

            # Target domain prediction
            _, target_rotation_pred = net(target_data)
        elif technique == Technique.MEDM:
            # Source domain prediction
            outputs = net(inputs)

            # Target domain prediction
            clabel_tgt = net(target_data)
        elif technique == Technique.DSN_MEDM:
            lambda_dsn = (
                lambda_parameter if current_step > active_domain_loss_step else 0
            )

            # Source domain prediction
            outputs, source_private_code, source_shared_code, source_rec_code = net(
                inputs,
                rec_code=ReconstructionCode.SOURCE,
                rec_scheme=ReconstructionSheme.BOTH_SHARED_AND_PRIVATE,
            )

            # Target domain prediction
            clabel_tgt, target_private_code, target_shared_code, target_rec_code = net(
                target_data,
                rec_code=ReconstructionCode.TARGET,
                rec_scheme=ReconstructionSheme.BOTH_SHARED_AND_PRIVATE,
            )
        elif technique == Technique.DANN_MEDM:
            # source prediction
            outputs, domain_prediction_source = net(inputs, lambda_parameter)
            # Get the domain prediction target
            clabel_tgt, domain_prediction_target = net(target_data, lambda_parameter)
        else:
            # general output
            outputs = net(inputs)

        # general prediction loss computation
        loss = cost_function(outputs, label)

        # Additive losses, according to the different techniques
        if technique == Technique.DDC:
            # 0.25 comes from a lambda factor in the Deep Domain Confusion:
            # Maximizing for Domain Invariance paper, and it has been
            # proven to provide good results
            loss += 0.25 * mmd_loss
        elif technique == Technique.DSN:
            # The network performs better if we start immediately using the DANN loss
            # so we don't wait for the current_step to reach active_domain_loss_step
            if current_step > active_domain_loss_step:
                source_domain_label = torch.zeros(len(inputs), dtype=torch.long).to(
                    device
                )
                target_domain_label = torch.ones(len(inputs), dtype=torch.long).to(
                    device
                )

                source_dann = gamma_weight * cost_function_domain(
                    source_domain_pred, source_domain_label
                )
                loss += source_dann

                target_dann = gamma_weight * cost_function_domain(
                    target_domain_pred, target_domain_label
                )
                loss += target_dann
            else:
                # only for the print
                source_dann = torch.zeros(1).float().cpu()
                target_dann = torch.zeros(1).float().cpu()

            # loss on the source
            input_img = inputs.clone().detach()
            source_diff = beta_weight * loss_diff(
                source_private_code, source_shared_code
            )
            source_simse = alpha_weight * loss_recon2(source_rec_code, input_img)
            loss += source_diff + source_simse

            # loss on target
            target_diff = beta_weight * loss_diff(
                target_private_code, target_shared_code
            )
            target_simse = alpha_weight * loss_recon2(target_rec_code, target_data)
            loss += target_diff + target_simse

        elif technique == Technique.DANN:
            # Define the correct labels for the source domain: 0
            zeros = torch.zeros(len(inputs), dtype=torch.long)
            zeros = zeros.to(device)

            # Define the correct labels for the target domain: 1
            ones = torch.ones(len(inputs), dtype=torch.long)
            ones = ones.to(device)

            # Calculate the loss for the source domain predictions
            pred_source_loss = cost_function_domain(domain_prediction_source, zeros)

            # Calculate the loss for the target domain predictions
            pred_target_loss = cost_function_domain(domain_prediction_target, ones)

            # Calculate the total loss for the domain classifier
            loss_domain = pred_source_loss + pred_target_loss

            # Add the calculated loss to the final result
            loss += loss_domain

        elif technique == Technique.ROTATION:
            # Calculate the loss for the rotation classification task (only on target)
            rotation_loss = cost_function(target_rotation_pred, rotation_label)
            loss += 0.4 * rotation_loss

        elif technique == Technique.MEDM:
            loss = F.nll_loss(outputs.log(), label)

            ## Target category diversity loss
            pb_pred_tgt = clabel_tgt.sum(dim=0)
            pb_pred_tgt = (
                1.0 / pb_pred_tgt.sum() * pb_pred_tgt
            )  # normalizatoin to a prob. dist.
            target_div_loss = -torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-6)))

            target_entropy_loss = -torch.mean(
                (clabel_tgt * torch.log(clabel_tgt + 1e-6)).sum(dim=1)
            )
            total_loss = loss + 1.0 * target_entropy_loss - 0.4 * target_div_loss

            ##1: Training shared network and label classifier
            print(
                "Train Epoch: {} \tentropy_Loss: {:.6f}\tlabel_Loss: {:.6f}\tdiv_Loss: {:.6f}".format(
                    epoch,
                    target_entropy_loss.item(),
                    loss.item(),
                    target_div_loss.item(),
                )
            )

            loss = total_loss
        elif technique == Technique.DSN_MEDM:
            loss = F.nll_loss(outputs.log(), label)

            ## Target category diversity loss
            pb_pred_tgt = clabel_tgt.sum(dim=0)
            pb_pred_tgt = (
                1.0 / pb_pred_tgt.sum() * pb_pred_tgt
            )  # normalizatoin to a prob. dist.
            target_div_loss = -torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-6)))

            target_entropy_loss = -torch.mean(
                (clabel_tgt * torch.log(clabel_tgt + 1e-6)).sum(dim=1)
            )
            total_loss = loss + 1.0 * target_entropy_loss - 0.4 * target_div_loss

            # loss on the source
            input_img = inputs.clone().detach()
            source_diff = beta_weight * loss_diff(
                source_private_code, source_shared_code
            )
            source_simse = alpha_weight * loss_recon2(source_rec_code, input_img)
            total_loss += source_diff + source_simse

            # loss on target
            target_diff = beta_weight * loss_diff(
                target_private_code, target_shared_code
            )
            target_simse = alpha_weight * loss_recon2(target_rec_code, target_data)
            total_loss += target_diff + target_simse

            ##1: Training shared network and label classifier
            print(
                "Train Epoch: {} \tentropy_Loss: {:.6f}\tlabel_Loss: {:.6f}\tdiv_Loss: {:.6f}".format(
                    epoch,
                    target_entropy_loss.item(),
                    loss.item(),
                    target_div_loss.item(),
                )
            )

            loss = total_loss
        elif technique == Technique.DANN_MEDM:

            # Define the correct labels for the source domain: 0
            zeros = torch.zeros(len(inputs), dtype=torch.long)
            zeros = zeros.to(device)

            # Define the correct labels for the target domain: 1
            ones = torch.ones(len(inputs), dtype=torch.long)
            ones = ones.to(device)

            # Calculate the loss for the source domain predictions
            pred_source_loss = F.nll_loss(domain_prediction_source.log(), zeros)

            # Calculate the loss for the target domain predictions
            pred_target_loss = F.nll_loss(domain_prediction_target.log(), ones)

            # Calculate the total loss for the domain classifier
            loss_domain = pred_source_loss + pred_target_loss

            # MEDM
            loss = F.nll_loss(outputs.log(), label)

            ## Target category diversity loss
            # Get the sum of all probabilities in a batch, grouped by class, e.g.
            # - predictions returned by the net: [[0.6, 0.4], [0.2, 0.8]]
            # - result: [0.8, 1.2]
            pb_pred_tgt = clabel_tgt.sum(dim=0)
            # Normalize to a probability distribution, e.g.
            # - from the previous example:
            #   - sum: 2
            #   - result: [0.4, 0.6]
            pb_pred_tgt = 1.0 / pb_pred_tgt.sum() * pb_pred_tgt
            # Calculate the entropy of pb_pred_tgt
            target_div_loss = -torch.sum((pb_pred_tgt * torch.log(pb_pred_tgt + 1e-6)))

            ## Target classification entropy loss
            target_entropy_loss = -torch.mean(
                (clabel_tgt * torch.log(clabel_tgt + 1e-6)).sum(dim=1)
            )

            # Calculate the final loss, as L_source_classification + lambda*L_entropy_classification - beta*L_entropy_category_diversity
            total_loss = (
                loss + 1.0 * target_entropy_loss + loss_domain - 0.4 * target_div_loss
            )

            ##1: Training shared network and label classifier
            print(
                "Train Epoch: {} \tentropy_Loss: {:.6f}\tlabel_Loss: {:.6f}\tdiffloss: {:.6f}\tdann: {:.6f}".format(
                    epoch,
                    target_entropy_loss.item(),
                    loss.item(),
                    target_div_loss.item(),
                    loss_domain.item(),
                )
            )

        # backward pass
        loss.backward()

        # optimizer
        optimizer.step()

        # add current step
        current_step += 1

        # fetch prediction and loss value
        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(dim=1)

        # compute training accuracy
        cumulative_accuracy += predicted.eq(label).sum().item()

    return cumulative_loss / samples, cumulative_accuracy / samples * 100
