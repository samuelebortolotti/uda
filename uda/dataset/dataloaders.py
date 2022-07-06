from typing import Tuple, List, Dict, Any, Optional
import torchvision
import torch
from uda.dataset.rotation_dataset import CustomImageFolder
from tqdm import tqdm

#### Compute Mean and Stdev ################


def get_mean_std(
    img_size: Tuple[int, int], source_data: str, target_data: str
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    r"""
    Computes mean and standard deviation over the source and
    target dataloader

    This function has been adapted from
    [link]: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py

    Args:

    - img_size [Tuple[int, int]]: image shape
    - source_data [str]: path to the source data
    - target_data [str]: path to the target data

    Returns:
        source mean and stdev, target mean and stdev [Tuple[Tuple[float, float], Tuple[float, float]]
    """

    # basic transformations
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
        ]
    )

    # datasets
    source_data = torchvision.datasets.ImageFolder(source_data, transform=transform)
    target_data = torchvision.datasets.ImageFolder(target_data, transform=transform)

    # data loader
    source_loader = torch.utils.data.DataLoader(
        dataset=source_data, batch_size=64, shuffle=True
    )
    target_loader = torch.utils.data.DataLoader(
        dataset=target_data, batch_size=64, shuffle=True
    )

    def compute_mean_std(loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        r"""
        Computes mean and standard deviation over the source and
        target dataloader

        Args:

        - loader [torch.utils.data.DataLoader]: dataloader

        Returns:
          dataloader mean and stdev [Tuple[float, float]]
        """
        channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

        for data, _ in tqdm(loader):
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        # std = sqrt(E[X^2] - (E[X])^2)
        std = (channels_sqrd_sum / num_batches - mean**2) ** 0.5

        return mean, std

    return (compute_mean_std(source_loader), compute_mean_std(target_loader))


############ Load dataloaders ############


def load_dataloaders(
    img_size: Tuple[int, int],
    source_data: str,
    target_data: str,
    batch_size: int = 128,
    test_batch_size: int = 256,
    # those parameters of mean and
    # standard deviation have been observed
    # empirically by computing mean and stdev over the
    # dataloaders
    mean_source: List[float] = [
        0.8023,
        0.7918,
        0.7904,
    ],
    stdev_source: List[float] = [0.3085, 0.3149, 0.3162],
    mean_target: List[float] = [0.4984, 0.4485, 0.4105],
    stdev_target: List[float] = [0.2733, 0.2635, 0.2590],
    additional_transformations: Optional[List[Any]] = None,
    normalize: bool = True,
) -> Dict[str, torch.utils.data.DataLoader]:
    r"""
    Load the dataloaders (for both training, validation and test)
    according to what has been specified by the arguments

    Default:

    - batch_size [int] = 128
    - test_batch_size [int] = 256
    - mean_source [List[float]] = [0.8023, 0.7918, 0.7904]
    - stdev_source [List[float]] = [0.3085, 0.3149, 0.3162]
    - mean_target [List[float]] = [0.4984, 0.4485, 0.4105]
    - stdev_target [List[float]] = [0.2733, 0.2635, 0.2590]
    - normalize [bool] = True

    Args:

    - img_size: Tuple[int, int]: image shape
    - source_data [str]: path to the source daa
    - target_data [str]: path to the target data
    - batch_size [int]: batch size
    - test_batch_size [int]: batch size for the test data
    - mean_source [List[float]]
    - stdev_source [List[float]]
    - mean_target [List[float]]
    - stdev_target [List[float]]
    - additional_transformations = None
    - normalize [bool] = True

    Returns:
        dataloaders [Dict[str, torch.utils.data.DataLoader]]: a dictionary containing the dataloaders, for training [source, target, source+target], validation [source] and test [target]
    """

    # Use mean and stdev recommended by Pytorch for ResNet and AlexNet
    # Using mean and stdev calculated directly from our dataset, in fact,
    # has brought worse results
    mean_source = [0.485, 0.456, 0.406]
    stdev_source = [0.229, 0.224, 0.225]
    mean_target = [0.485, 0.456, 0.406]
    stdev_target = [0.229, 0.224, 0.225]

    def split_dataset(
        dataset: torch.utils.data.dataset.Subset, perc: float
    ) -> Tuple[torchvision.datasets.ImageFolder, torchvision.datasets.ImageFolder]:
        r"""Split the dataset into a training and validation set randomly according to the specified
        percentage value.

        Args:

        - dataset [torchvision.datasets.Subset]: datset
        - perc [float]: split percentage must be a value between 0 and 1

        Returns:
          [training, validation] Tuple[torchvision.datasets.ImageFolder]: respectively the training set,
           which will have (perc*100)% of samples and validation set, which will have (100-(100*perc))% samples
        """
        training_samples = int(len(dataset) * perc + 1)
        validation_samples = len(dataset) - training_samples
        training, validation = torch.utils.data.random_split(
            dataset,
            [training_samples, validation_samples],
            generator=torch.Generator().manual_seed(32),
        )
        return training, validation

    print("#> Loading dataloader ...")

    # transformations
    transform_source = [
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]

    # rotated transforms - we won't add additional transformations in this case
    rotated_transform = [
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
    ]

    # target transforms
    transform_target = [
        torchvision.transforms.Resize(img_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ]

    # Additional transformations
    if additional_transformations:
        transform_source.append(*additional_transformations)
        transform_target.append(*additional_transformations)

    # normalization
    if normalize:
        transform_source.append(
            torchvision.transforms.Normalize(mean_source, stdev_source)
        )
        transform_target.append(
            torchvision.transforms.Normalize(mean_target, stdev_target)
        )

    # compose
    transform_source = torchvision.transforms.Compose(transform_source)
    transform_target = torchvision.transforms.Compose(transform_target)
    rotated_transform = torchvision.transforms.Compose(rotated_transform)

    # target rotated data
    target_rotated_data = CustomImageFolder(target_data, transform=rotated_transform)

    # datasets
    source_data = torchvision.datasets.ImageFolder(
        source_data, transform=transform_source
    )

    target_data = torchvision.datasets.ImageFolder(
        target_data, transform=transform_target
    )

    # split
    target_rotated_training, target_rotated_validation = split_dataset(
        target_rotated_data, 0.8
    )
    source_training, source_validation = split_dataset(source_data, 0.8)
    target_training, target_test = split_dataset(target_data, 0.8)
    source_target_training = torch.utils.data.ConcatDataset(
        [source_training, target_training]
    )

    # Dataloaders
    target_rotated_training_loader = torch.utils.data.DataLoader(
        target_rotated_training, batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    target_rotated_validation_loader = torch.utils.data.DataLoader(
        target_rotated_validation,
        batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    source_training_loader = torch.utils.data.DataLoader(
        source_training, batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    source_validation_loader = torch.utils.data.DataLoader(
        source_validation, batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    target_training_loader = torch.utils.data.DataLoader(
        target_training, batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test, test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )
    source_target_training_loader = torch.utils.data.DataLoader(
        source_target_training, batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    source_data_loader = torch.utils.data.DataLoader(
        source_data, batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    target_data_loader = torch.utils.data.DataLoader(
        target_data, batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    print(
        "\t# of rotated training samples [target]: %d\n"
        % int(len(target_rotated_training))
    )
    print(
        "\t# of rotated validation samples [target]: %d\n"
        % int(len(target_rotated_validation))
    )
    print("\t# of training samples [source]: %d\n" % int(len(source_training)))
    print("\t# of validation samples [source]: %d\n" % int(len(source_validation)))
    print("\t# of training samples [target]: %d\n" % int(len(target_training)))
    print("\t# of test samples [target]: %d\n" % int(len(target_test)))
    print("\t# of full source: %d\n" % int(len(source_data)))
    print("\t# of full target: %d\n" % int(len(target_data)))
    print(
        "\t# of training samples [source + target]: %d\n"
        % int(len(source_target_training))
    )

    # dictionary of loaders
    dataloaders = {
        "train_source": source_training_loader,
        "val_source": source_validation_loader,
        "train_target": target_training_loader,
        "test_target": target_test_loader,
        "train_rotated_target": target_rotated_training_loader,
        "val_rotated_target": target_rotated_validation_loader,
        "train_source_target": source_target_training_loader,
        "full_source": source_data_loader,
        "full_target": target_data_loader,
    }

    return dataloaders
