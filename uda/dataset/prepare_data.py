from typing import List, Optional
from os import makedirs
from os.path import join
import os
from shutil import copytree, rmtree
from argparse import _SubParsersAction as Subparser
from argparse import Namespace
from . import classes
from tqdm import tqdm
import gdown
import zipfile


def configure_subparsers(subparsers: Subparser) -> None:
    """Configure a new subparser for running the data dowload and preparation
    Args:
      subparser (Subparser): argument parser
    """
    parser = subparsers.add_parser("dataset", help="Dataset preparation subparser")
    parser.add_argument(
        "--classes",
        "-C",
        nargs="+",
        default=classes,
        help="classes to filter, those provided by default are the ones used for the project",
    )
    parser.add_argument(
        "--destination-location",
        "-D",
        type=str,
        default="Adaptiope",
        help="where to store the dataset",
    )
    parser.add_argument(
        "--new-dataset-name",
        "-N",
        type=str,
        default="adaptiope_small",
        help="filtered dataset name",
    )
    # set the main function to run when blob is called from the command line
    parser.set_defaults(func=main)


def download_file_from_google_drive(id: str, destination: str) -> None:
    """Dowloads a file from a Google Drive link
    Args:
      id (str): shared document id
      destination (str): where to store the dowloaded data
    """
    BASE_URL = "https://drive.google.com/u/1/uc?id="
    gdown.download(f"{BASE_URL}{id}&export=download", destination, quiet=False)


def unzip_file(source: str, dest: str) -> None:
    """Unzip a file in a given destination location
    Args:
      source (str): source zip file
      dest (str): where to store the unzipped data
    """
    with zipfile.ZipFile(source, "r") as zp:
        zp.extractall(dest)


def filter_data(
    classes: List[str], original_dataset: List[str], smaller_dataset: List[str]
) -> None:
    """Filter the full dataset to only preserve the needed classes and considers
    only the `real world` and `product` domain.
    Args:
      classes (List[str]): list of classes to preserve
      original_dataset (List[str]): original datasets domains location
      smaller_dataset (List[str]): smaller datasets domains location
    """
    for d, td in zip(original_dataset, smaller_dataset):
        makedirs(td)
        for c in tqdm(classes):
            c_path = join(d, c)
            c_target = join(td, c)
            copytree(c_path, c_target)


def main(args: Namespace) -> None:
    r"""Checks the command line arguments and then runs the download and the prepare data script
    Args:
      args (Namespace): command line arguments
    """

    ADAPTIOPE_DOWNLOAD_ID = "1FmdsvetC0oVyrFJ9ER7fcN-cXPOWx2gq"

    print("\n### Dataset preparation ###")
    print("> Parameters:")
    for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
        print("\t{}: {}".format(p, v))
    print("\n")

    # dowload the full dataset from google drive
    download_file_from_google_drive(
        ADAPTIOPE_DOWNLOAD_ID, f"{args.destination_location}.zip"
    )

    # unzip the file
    print("Unzipping the full dataset")
    unzip_file(f"{args.destination_location}.zip", args.destination_location)

    # filter data
    filter_data(
        classes,
        [
            f"{args.destination_location}/Adaptiope/product_images",
            f"{args.destination_location}/Adaptiope/real_life",
        ],
        [
            f"{args.new_dataset_name}/product_images",
            f"{args.new_dataset_name}/real_life",
        ],
    )

    # removing the extra file
    print("Removing the full dataset")
    rmtree(args.destination_location)
