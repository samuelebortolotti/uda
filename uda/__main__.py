"""Main module of the `uda` project

This project has been developed by Samuele Bortolotti and Luca De Menego as a project for the Deep Learning course
of the master's degree program in Computer Science at University of Trento
"""

import argparse
from argparse import Namespace
import matplotlib
import uda.dataset.prepare_data as data
import uda.dataset.visualize_dataset as viz_data
from uda.dataset import classes
from uda import Technique
import torch
import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import Any, List
import sys
import os
from uda.networks import *
from uda.optimizers import *
from uda.schedulers import *
from uda.visualization import *
from uda.utils.utils import (
    load_best_weights,
    log_values,
    load_best_weights,
    resume_training,
    get_lr,
)
from uda.train import training_step
from uda.test import test_step
from uda.dataset.dataloaders import load_dataloaders
from argparse import _SubParsersAction as Subparser
from argparse import Namespace


def get_args() -> Namespace:
    """Parse command line arguments.

    Returns:
      Namespace: command line arguments
    """
    # main parser
    parser = argparse.ArgumentParser(
        prog="Unsupervised-domain-adaptation",
        description="""
        Project of the Deep Learning course at University of Trento
        """,
    )

    # matplotlib interactive backend
    parser.add_argument(
        "--matplotlib-backend",
        "-mb",
        choices=matplotlib.rcsetup.interactive_bk,
        default="QtAgg",
        help="Matplotlib interactive backend [default: QtAgg]",
    )

    # subparsers
    subparsers = parser.add_subparsers(help="sub-commands help")
    # configure the dataset subparser
    data.configure_subparsers(subparsers)
    # configure the visualize dataset subparser
    viz_data.configure_subparsers(subparsers)
    # configure the experiment subparser
    configure_subparsers(subparsers)

    # parse the command line arguments
    parsed_args = parser.parse_args()

    # if function not passed, then print the usage and exit the program
    if "func" not in parsed_args:
        parser.print_usage()
        parser.exit(1)

    return parsed_args


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the unsupervised domain adaptation experiment
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser(
        "experiment", help="Unsupervised Domain Adaptation experiment helper"
    )
    parser.add_argument(
        "--wandb",
        "-w",
        type=bool,
        default=False,
        help="employ wandb to keep track of the runs",
    )
    parser.add_argument(
        "technique",
        type=int,
        choices=range(10),
        help="which network to run, see the `Technique` enumerator in `uda/__init__.py` and select the one you prefer",
    )
    parser.add_argument("source_data", type=str, help="source domain dataset")
    parser.add_argument("target_data", type=str, help="target domain dataset")
    parser.add_argument("exp_name", type=str, help="name of the experiment")
    parser.add_argument(
        "num_classes", type=int, default=20, help="number of classes [default 20]"
    )
    parser.add_argument("pretrained", type=bool, default=True, help="pretrained model")
    parser.add_argument("epochs", type=int, help="number of training epochs")
    parser.add_argument(
        "net_name", type=str, choices=["alexnet", "resnet"], help="backbone network"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        choices=["cuda", "cpu"],
        help="device on which to run the experiment",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--test-batch-size", type=int, default=256, help="test batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.000001, help="sdg weight decay"
    )
    parser.add_argument("--momentum", type=float, default=0.95, help="sdg momentum")
    parser.add_argument(
        "--step-decay-weight", type=float, default=0.95, help="sdg step weight decay"
    )
    parser.add_argument(
        "--active-domain-loss-step",
        type=int,
        default=10000,
        help="active domain loss step",
    )
    parser.add_argument(
        "--lr-decay-step", type=int, default=20000, help="learning rate decay step"
    )
    parser.add_argument(
        "--alpha-weight",
        type=float,
        default=0.01,
        help="alpha weight factor for Domain Separation Networks",
    )
    parser.add_argument(
        "--beta-weight",
        type=float,
        default=0.075,
        help="beta weight factor for Domain Separation Networks",
    )
    parser.add_argument(
        "--gamma-weight",
        type=float,
        default=0.25,
        help="gamma weight factor for Domain Separation Networks",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=20,
        help="how frequent to save the model",
    )
    parser.add_argument(
        "--reverse-domains",
        type=bool,
        default=False,
        help="switch source and target domain",
    )
    parser.add_argument(
        "--dry", type=bool, default=False, help="do not save checkpoints"
    )
    parser.add_argument(
        "--project" "-w", type=str, default="dl-project", help="wandb project"
    )
    parser.add_argument(
        "--entity" "-w", type=str, default="deep-learning-project", help="wandb entity"
    )
    parser.add_argument(
        "--classes",
        "-C",
        nargs="+",
        default=classes,
        help="classes provided in the dataset, by default they are those employed for the project",
    )
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=experiment)


def unsupervised_domain_adaptation_main(
    net: nn.Module,
    exp_name: str,
    technique: Technique,
    source_data: str,
    target_data: str,
    net_name: str,
    resume: bool = False,
    device: str = "cuda",
    batch_size: int = 128,
    test_batch_size: int = 256,
    learning_rate: float = 0.01,
    weight_decay: float = 0.000001,
    momentum: float = 0.9,
    step_decay_weight: float = 0.95,
    active_domain_loss_step: int = 10000,
    lr_decay_step: int = 20000,
    alpha_weight: float = 0.01,
    beta_weight: float = 0.075,
    gamma_weight: float = 0.25,
    epochs: int = 10,
    save_every_epochs: int = 10,
    reverse_domains: bool = False,
    dry: bool = False,
    additional_transformations: List[nn.Module] = [],
    wandb: bool = False,
    **kwargs: Any,
) -> None:
    r"""
    Function which performs both training and test step, it is able to adapt to
    different architectures according to the `technique` passed.

    Namely:

    - Source-Only (baseline): learns only from the source domain
    - Upper-Bound: learns both from the source and the target domain
    - Deep Domain Confusion: https://arxiv.org/pdf/1412.3474.pdf
    - Domain-Adversarial Neural Network: https://arxiv.org/pdf/1505.07818.pdf
    - Domain Separation Networks: https://arxiv.org/pdf/1608.06019.pdf
    - Entropy Minimization vs. Diversity Maximization for Domain Adaptation: https://arxiv.org/pdf/2002.01690.pdf
    - Domain Separation Networks with Entropy Minimization vs. Diversity Maximization

    Defaults:

    - resume [bool] = False: by default do not resume last training
    - device [str] = "cuda": move tensors on GPU
    - batch_size [int] = 128
    - test_batch_size [int] = 256
    - learning_rate [float] = 0.01
    - weight_decay [float] = 0.000001
    - momentum [float] = 0.9
    - epochs [int] = 10
    - step_decay_weight [float] = 0.95,
    - active_domain_loss_step [int] = 10000,
    - lr_decay_step [int] = 20000,
    - alpha_weight [float] = 0.01,
    - beta_weight [float] = 0.075,
    - gamma_weight [float] = 0.25,
    - save_every_epochs [int] = 10: save a checkpoint every 10 epoch
    - reverse_domains [bool] = False: by default the domain are in the order specified
    - dry [bool] = False: by default save weights
    - additional_transformations [List[nn.Module]] = []
    - wandb [bool] = False

    Args:

    - net [nn.Module]: network architecture
    - exp_name [str]: name of the experiment, basically where to save the logs of the SummaryWriter
    - technique [Technique]: technique to employ
    - resume [bool] = False: whether to resume a checkpoint
    - device [str] = "cuda": where to load the tensors
    - batch_size [int] = 128: default batch size
    - test_batch_size [int] = 256: default test batch size
    - learning_rate [float] = 0.01: initial learning rate
    - weight_decay [float] = 0.000001 : initial weight decay
    - momentum [float] = 0.9: initial momentum
    - step_decay_weight [float] = 0.95: learning rate weight decay step for DSN
    - active_domain_loss_step [int] = 10000: how many steps to wait for applying the Domain Adaptation technique
    - lr_decay_step [int] = 20000: learning rate weight decay step
    - alpha_weight [float] = 0.01: alpha parameter for the DSN loss function
    - beta_weight [float] = 0.075: beta parameter for the DSN loss function
    - gamma_weight [float] = 0.25: gamma parameter for the DSN loss function
    - epochs [int] = 10: number of epochs
    - save_every_epochs: int = 10: save a checkpoint every `save_every_epochs` epoch
    - dry [bool] = False: whether to do not save weights
    - reverse_domains [bool] = False: whether to swap or not the domains (source and target)
    - additional_transformations [List[nn.Module]] = []: additional transformation for the source domain dataloader
    - wandb [bool] = False: whether to log values on wandb
    - \*\*kwargs [Any]: additional key-value arguments
    """
    # create a logger for the experiment
    log_directory = "runs/exp_{}".format(exp_name)
    if technique == Technique.DDC:
        log_directory = "runs/deep_domain_confusion_{}".format(exp_name)
    elif technique == Technique.DANN:
        log_directory = "runs/dann_{}".format(exp_name)
    elif technique == Technique.DSN:
        log_directory = "runs/dsn_{}".format(exp_name)
    elif technique == Technique.ROTATION:
        log_directory = "runs/rotation_{}".format(exp_name)

    # create a logger for the experiment
    writer = SummaryWriter(log_dir="runs/deep_domain_confusion_{}".format(exp_name))

    # create folder for the experiment
    os.makedirs(exp_name, exist_ok=True)

    # Set up the metrics
    metrics = {
        "loss": {"train": 0, "val": 0, "test": 0},
        "acc": {"train": 0, "val": 0, "test": 0},
    }

    # get dataloaders
    source_data = source_data  # source
    target_data = target_data  # target

    # reverse domains
    if reverse_domains:
        source_data, target_data = target_data, source_data

    dataloaders = load_dataloaders(
        (224, 224),  # img_size
        source_data,
        target_data,
        batch_size=batch_size,  # default batch size
        test_batch_size=test_batch_size,  # default test batch size
        additional_transformations=additional_transformations,  # additional transoformations
    )

    if technique == Technique.UPPER_BOUND:
        # train supervisely on both datasets
        train_loader = dataloaders["train_source_target"]
    else:
        train_loader = dataloaders["train_source"]

    if technique == Technique.ROTATION:
        rotation_training_loader = dataloaders["train_rotated_target"]
    else:
        rotation_training_loader = []

    val_loader = dataloaders["val_source"]
    training_target = dataloaders["train_target"]
    test_target = dataloaders["test_target"]

    print("Technique {}".format(technique))

    optimizer = get_adam_optimizer(net, learning_rate)

    # instantiate the optimizer
    if technique == Technique.DDC:
        if net_name == "alexnet":
            optimizer = get_ddc_alexnet_optimizer(
                net, learning_rate, weight_decay, momentum
            )
        elif net_name == "resnet":
            optimizer = get_ddc_resnet_optimizer(
                net, learning_rate, weight_decay, momentum
            )
    elif technique == Technique.DANN:
        if net_name == "alexnet":
            optimizer = get_dann_alexnet_optimizer(
                net, learning_rate, weight_decay, momentum
            )
        elif net_name == "resnet":
            optimizer = get_dann_resnet_optimizer(
                net, learning_rate, weight_decay, momentum
            )
    elif technique == Technique.DSN:
        optimizer = get_adam_optimizer(net, learning_rate)
    elif technique == Technique.ROTATION:
        optimizer = get_adam_optimizer(net, learning_rate)
    elif technique == Technique.MEDM:
        optimizer = torch.optim.Adam(
            [
                {"params": net.features.parameters(), "lr": learning_rate / 100},
                {"params": net.classifier.parameters(), "lr": learning_rate},
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif technique == Technique.DSN_MEDM:
        optimizer = torch.optim.Adam(
            [
                {"params": net.source_encoder.parameters(), "lr": learning_rate / 50},
                {"params": net.target_encoder.parameters(), "lr": learning_rate / 50},
                {"params": net.shared_encoder.parameters(), "lr": learning_rate / 50},
                {"params": net.classifier.parameters(), "lr": learning_rate},
                {"params": net.merge_features.parameters(), "lr": learning_rate},
                {"params": net.bottle_neck.parameters(), "lr": learning_rate},
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif technique == Technique.DANN_MEDM:
        optimizer = torch.optim.Adam(
            [
                {"params": net.features.parameters(), "lr": learning_rate / 100},
                {"params": net.classifier.parameters(), "lr": learning_rate},
                {"params": net.adapt.parameters(), "lr": learning_rate},
                {"params": net.domain_classifier.parameters(), "lr": learning_rate},
            ],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        if net_name == "alexnet":
            optimizer = get_native_alexnet_optimizer(
                net, learning_rate, weight_decay, momentum
            )
        elif net_name == "resnet":
            optimizer = get_native_resnet_optimizer(
                net, learning_rate, weight_decay, momentum
            )

    # define the cost function
    cost_function = torch.nn.CrossEntropyLoss()

    # define the domain cost function, useful only if we use DANN
    cost_function_domain = torch.nn.CrossEntropyLoss()

    # define the cost functions for the reconstruction, difference and similarity
    loss_recon1 = SIMSE()
    loss_recon2 = SIMSE()
    loss_diff = DiffLoss()

    # Resume training or start a new experiment
    training_params, val_params, start_epoch = resume_training(
        resume, exp_name, net, optimizer
    )

    current_step = 0

    # log on wandb if and only if the module is loaded
    if wandb:
        wandb.watch(net)

    # for each epoch, train the network and then compute evaluation results
    for e in tqdm.tqdm(range(start_epoch, epochs), desc="Epochs"):
        # training step [supervisely on source, unsupervisely on target {if upperbound, it
        # will train supervisely on both source and target }]
        train_loss, train_accuracy = training_step(
            net,
            iter(train_loader),
            training_target,
            rotation_training_loader,
            optimizer,
            cost_function,
            cost_function_domain,
            e,
            epochs,
            writer,
            title="Training",
            technique=technique,
            device=device,
            current_step=current_step,
            active_domain_loss_step=active_domain_loss_step,
            loss_diff=loss_diff,
            loss_recon1=loss_recon1,
            loss_recon2=loss_recon2,
            gamma_weight=gamma_weight,
            beta_weight=beta_weight,
            alpha_weight=alpha_weight,
            learning_rate=learning_rate,
            lr_decay_step=lr_decay_step,
            step_decay_weight=step_decay_weight,
        )
        # update the current step
        current_step += len(iter(train_loader))
        # save the values in the metrics
        metrics["loss"]["train"] = train_loss
        metrics["acc"]["train"] = train_accuracy

        # test step on the test [target]
        # **Note** it is used ONLY for logging purposes during the debugging phase
        # in order to see whether the performances improve over the target domain
        test_loss, test_accuracy = test_step(
            net,
            iter(test_target),
            cost_function,
            epochs,
            writer,
            "Test",
            technique=technique,
            device=device,
            rec_code=ReconstructionCode.TARGET,
        )
        # save the metrics
        metrics["loss"]["test"] = test_loss
        metrics["acc"]["test"] = test_accuracy

        # save model and checkpoint
        training_params["start_epoch"] = e + 1  # epoch where to start
        training_params["technique"] = technique  # technique used

        # check if I have outperformed the best loss in the validation set
        # Here for the scope of the project I save the best according to the test
        if val_params["best_loss"] > metrics["loss"]["test"]:
            val_params["best_loss"] = metrics["loss"]["test"]
            # save best weights
            if not dry:
                torch.save(net.state_dict(), os.path.join(exp_name, "best.pth"))
        # what to save
        save_dict = {
            "state_dict": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "training_params": training_params,
            "val_params": val_params,
        }
        # save current weights
        if not dry:
            torch.save(net.state_dict(), os.path.join(exp_name, "net.pth"))
            # save current settings
            torch.save(save_dict, os.path.join(exp_name, "ckpt.pth"))
            if e % save_every_epochs == 0:
                # Dump every checkpoint
                torch.save(
                    save_dict,
                    os.path.join(exp_name, "ckpt_e{}.pth".format(e + 1)),
                )
        del save_dict

        # logs to TensorBoard
        log_values(writer, e, train_loss, train_accuracy, "Train")
        log_values(writer, e, test_loss, test_accuracy, "Test")
        writer.add_scalar("Learning rate", get_lr(optimizer), e)

        # log on wandb if and only if the module is loaded
        if wandb:
            wandb.log(
                {
                    "train/train_loss": train_loss,
                    "train/train_accuracy": train_accuracy,
                    "test/test_loss": test_loss,
                    "test/test_accuracy": test_accuracy,
                    "learning_rate": get_lr(optimizer),
                }
            )

        # test value

        print("\nEpoch: {:d}".format(e + 1))
        print(
            "\t Training loss {:.5f}, Training accuracy {:.2f}".format(
                train_loss, train_accuracy
            )
        )
        print(
            "\t Test loss {:.5f}, Test accuracy {:.2f}".format(test_loss, test_accuracy)
        )
        print("-----------------------------------------------------")

    # compute final evaluation results
    print("#> After training:")
    train_loss, train_accuracy = test_step(
        net,
        iter(train_loader),
        cost_function,
        epochs,
        writer,
        "Training",
        technique=technique,
        device=device,
        rec_code=ReconstructionCode.SOURCE,
    )
    val_loss, val_accuracy = test_step(
        net,
        iter(val_loader),
        cost_function,
        epochs,
        writer,
        "Validation",
        technique=technique,
        device=device,
        rec_code=ReconstructionCode.SOURCE,
    )
    test_loss, test_accuracy = test_step(
        net,
        iter(test_target),
        cost_function,
        epochs,
        writer,
        "Test",
        technique=technique,
        device=device,
        rec_code=ReconstructionCode.TARGET,
    )

    # log to TensorBoard
    log_values(writer, epochs, train_loss, train_accuracy, "Train")
    log_values(writer, epochs, val_loss, val_accuracy, "Validation")
    log_values(writer, epochs, test_loss, test_accuracy, "Test")

    # log on wandb if and only if the module is loaded
    if wandb:
        wandb.log(
            {
                "train/train_loss": train_loss,
                "train/train_accuracy": train_accuracy,
                "validation/validation_loss": val_loss,
                "validation/validation_accuracy": val_accuracy,
                "test/test_loss": test_loss,
                "test/test_accuracy": test_accuracy,
            }
        )

    print(
        "\t Training loss {:.5f}, Training accuracy {:.2f}".format(
            train_loss, train_accuracy
        )
    )
    print(
        "\t Validation loss {:.5f}, Validation accuracy {:.2f}".format(
            val_loss, val_accuracy
        )
    )
    print(
        "\t [Current] Test loss {:.5f}, Test accuracy {:.2f}".format(
            test_loss, test_accuracy
        )
    )

    # Test on best weights
    load_best_weights(net, exp_name)

    test_loss, test_accuracy = test_step(
        net,
        iter(test_target),
        cost_function,
        epochs,
        writer,
        "Test",
        technique=technique,
        device=device,
        rec_code=ReconstructionCode.SOURCE,
    )

    print(
        "\n\t [Best] Test loss {:.5f}, Test accuracy {:.2f}".format(
            test_loss, test_accuracy
        )
    )
    print("-----------------------------------------------------")

    # closes the logger
    writer.close()


def experiment(args: Namespace) -> None:
    """Experiment wrapper function

    Args:
      args (Namespace): command line arguments
    """

    # mapping number to technique
    try:
        args.technique = Technique(args.technique)
    except:
        print("No valid technique selected")
        exit(1)

    print("\n### Experiment ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # set wandb if needed
    if args.wandb:
        # import wandb
        import wandb

        # Log in to your W&B account
        wandb.login()

    if args.wandb:
        # start the log
        wandb.init(project=args.project, entity=args.entity)

    # network initialization
    net = None

    # technique switch
    if (
        args.technique == Technique.SOURCE_ONLY
        or args.technique == Technique.UPPER_BOUND
    ):
        # Train using only the source domain
        # or train using the target domain too, to fix an upper bound
        net = (
            AlexNet(num_classes=args.num_classes, pretrained=args.pretrained)
            if args.net_name == "alexnet"
            else ResNet18(num_classes=args.num_classes, pretrained=args.pretrained)
        )
    elif args.technique == Technique.DDC:
        # Deep Domain Confusion
        net = (
            DDCAlexNet(num_classes=args.num_classes, pretrained=args.pretrained)
            if args.net_name == "alexnet"
            else DDCResNet18(num_classes=args.num_classes, pretrained=args.pretrained)
        )
    elif args.technique == Technique.DANN:
        # Domain-Adversarial Neural Network
        net = (
            DANNAlexNet(num_classes=args.num_classes, pretrained=args.pretrained)
            if args.net_name == "alexnet"
            else DANNResNet18(num_classes=args.num_classes, pretrained=args.pretrained)
        )
    elif args.technique == Technique.DSN:
        # Domain Separation Network
        if args.net_name == "alexnet":
            net = AlexNetDSN(num_classes=args.num_classes, pretrained=args.pretrained)
        else:
            decoder_location = input("Enter the weight of the decoder: ").strip()
            if not os.path.isfile(decoder_location):
                print("Missing weights")
                exit(1)
            net = ResNet18DSNImproved(
                num_classes=args.num_classes,
                pretrained=args.pretrained,
                decoder_location=decoder_location,
            )
    elif args.technique == Technique.ROTATION:
        # Rotation Loss
        net = RotationArch(num_classes=args.num_classes)
    elif args.technique == Technique.MEDM:
        # Entropy Minimization vs. Diversity Maximization
        net = MEDM(num_classes=args.num_classes, pretrained=args.pretrained)
    elif args.technique == Technique.DANN_MEDM:
        # DANN with Entropy Minimization vs. Diversity Maximization
        net = DANNMEDM(num_classes=args.num_classes, pretrained=args.pretrained)
    elif args.technique == Technique.DSN_MEDM:
        # DSN with MEDM
        net = DSNMEDM(num_classes=args.num_classes, pretrained=args.pretrained)
    else:
        print("No technique selected")
        exit(1)
    
    net.to(args.device)

    # run the experiment
    unsupervised_domain_adaptation_main(net=net, **vars(args))

    dataloaders = load_dataloaders(
        (224, 224),  # img_size
        args.source_data,
        args.target_data,
        batch_size=32,
        additional_transformations=[],
    )

    # generate teh features
    targets, outputs = gen_features_source_domain(
        net,
        dataloaders["full_source"]
        if not args.reverse_domains
        else dataloaders["full_target"],
        dataloaders["full_target"]
        if not args.reverse_domains
        else dataloaders["full_source"],
        technique=args.technique,
        device=args.device,
    )

    feature_visualizer(
        "Upper bound ResNet18 source vs target features",
        targets,
        outputs,
        2,
        legend=["source", "target"]
        if not args.reverse_domains
        else ["target", "source"],
    )

    targets, outputs = gen_predictions_targets(
        net,
        dataloaders["full_target"]
        if not args.reverse_domains
        else dataloaders["full_source"],
        device=args.device,
        technique=args.technique,
    )

    feature_visualizer(
        "Upper bound ResNet18 target dataset classes distribution",
        targets,
        outputs,
        20,
        legend=args.classes,
    )

    if args.wandb:
        # finish the log
        wandb.finish()


def main(args: Namespace) -> None:
    """Main function

    It runs the `func` function passed to the parser with the respective
    parameters
    Args:
      args (Namespace): command line arguments
    """
    # set matplotlib backend
    matplotlib.use(args.matplotlib_backend)
    # execute the function `func` with args as arguments
    args.func(
        args,
    )


if __name__ == "__main__":
    """
    Main

    Calls the main function with the command line arguments passed as parameters
    """
    main(get_args())
