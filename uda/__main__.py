"""Main module of the `uda` project

This project has been developed by Samuele Bortolotti and Luca De Menego as a project for the Deep Learning course
of the master's degree program in Computer Science at University of Trento
"""

import argparse
from argparse import Namespace
import matplotlib


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

    # parse the command line arguments
    parsed_args = parser.parse_args()

    # if function not passed, then print the usage and exit the program
    if "func" not in parsed_args:
        parser.print_usage()
        parser.exit(1)

    return parsed_args


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
