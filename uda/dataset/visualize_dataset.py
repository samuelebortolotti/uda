import torch
import torchvision
import matplotlib.pyplot as plt
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from uda.dataset.dataloaders import load_dataloaders


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the dataset visualization
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser("visualize", help="Dataset visualization subparser")
    parser.add_argument(
        "dataset_source", type=str, help="location of the source dataset"
    )
    parser.add_argument("dataset_dest", type=str, help="location of the target dataset")
    parser.add_argument(
        "label", type=int, choices=range(0, 20), help="which label to display"
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=128, help="batch size of the datasets"
    )
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=main)


def visualize_train_datasets(
    source_train: torch.utils.data.DataLoader,
    target_train: torch.utils.data.DataLoader,
    image_label: int,
    rows: int = 3,
    cols: int = 3,
) -> None:
    r"""
    Show the data from both dataloaders according to the label which we pass
    as a parameter

    **Note**: the number of images displayed depends on the size of the batch, since
    the bigger the batch size, the more the probability there are enogh samples of the
    desired label to be displayed.

    Default:

    - rows [int] = 3
    - cols [int] = 3

    Args:

    - source_train [torch.utils.data.DataLoader]
    - target_train [torch.utils.data.DataLoader]
    - image_label [int]: image label index
    - rows [int]
    - cols [int]
    """

    # define iterators over both datasets
    train_iter_source, train_iter_target = iter(source_train), iter(target_train)

    # get labels of source data
    data_source, labels_source = next(train_iter_source)

    # get labels of target data
    data_target, labels_target = next(train_iter_target)

    # get first rows*cols indices of the chosen label for source data
    get_idx = (labels_source == image_label).nonzero().squeeze(-1)[0 : (rows * cols)]

    # get first rows*cols indices of the chosen label for target data
    get_idx_target = (
        (labels_target == image_label).nonzero().squeeze(-1)[0 : (rows * cols)]
    )

    # get the data and labels for the chosen data
    get_data_source, get_labels_source = (
        data_source[get_idx, :, :, :],
        labels_source[get_idx],
    )
    get_data_target, get_labels_target = (
        data_target[get_idx_target, :, :, :],
        labels_target[get_idx_target],
    )

    if len(get_data_source) == 0 or len(get_data_target) == 0:
        print("No data retrieved with given label")
        return

    # How many image it was able to retrieve
    print(
        "Retreived {} source and {} target images".format(
            len(get_data_source), len(get_data_target)
        )
    )

    # source display
    display_grid = torchvision.utils.make_grid(
        get_data_source,
        nrow=rows,
        padding=2,
        pad_value=1,
        normalize=True,
        value_range=(get_data_source.min(), get_data_source.max()),
    )
    plt.subplot(1, 2, 1)
    plt.imshow((display_grid.numpy().transpose(1, 2, 0)))
    plt.axis("off")
    plt.title(f"Source Train Dataset label {image_label}")

    # target display
    display_grid_translated = torchvision.utils.make_grid(
        get_data_target,
        nrow=rows,
        padding=2,
        pad_value=1,
        normalize=True,
        value_range=(get_data_source.min(), get_data_source.max()),
    )
    plt.subplot(1, 2, 2)
    plt.imshow((display_grid_translated.numpy().transpose(1, 2, 0)))
    plt.axis("off")
    plt.title(f"Target Train Dataset label: {image_label}")
    plt.tight_layout()
    plt.show()


def main(args: Namespace) -> None:
    r"""Checks the command line arguments and then runs the dataset visualization
    Args:
      args (Namespace): command line arguments
    """
    print("\n### Dataset visualization ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # Load dataloaders
    print("Load dataloaders...")
    dataloaders = load_dataloaders(
        (224, 224),  # img_size
        args.dataset_source,
        args.dataset_dest,
        batch_size=args.batch_size,
        additional_transformations=[],
    )

    # visualize
    print("Visualizing...")
    visualize_train_datasets(
        dataloaders["train_source"],  # train source loader
        dataloaders["train_target"],  # train target loader
        args.label,  # image label
    )
