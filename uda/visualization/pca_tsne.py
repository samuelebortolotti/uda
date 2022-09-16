from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
from uda import Technique
from typing import Tuple, List
import numpy as np
from uda.networks.DSN import ReconstructionCode, Flatten
import os
import matplotlib.pyplot as plt


def gen_predictions_targets(
    net: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    technique: Technique,
    device: str = "cuda",
) -> Tuple[np.double, np.double]:
    r"""
    Function which generates the prediction for a dataloader according to the actual
    technique which has been chosen.

    In this case, the prediction is associated to the actual ground-truth label

    This is used in order to see, thanks to PCA and tSNE, whether the model has
    learned to cluster the features according to each of the different classes

    The code has been adapted from:
    [link]: https://github.com/2-Chae/PyTorch-tSNE/blob/main/main.py

    Args:

    - net [nn.Module]: network architecture
    - dataloader [torch.utils.data.DataLoader]: dataloader where to compute the features
    - technique [Technique]: technique to employ
    - device [str] = "cuda": where to load the tensors

    Returns:

    - targets [np.double]: targets
    - outputs [np.double]: features
    """

    # This prints only data with the corresponding labels
    net.eval()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        # loop over the dataloader
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # get data on cpu
            targets_np = targets.data.cpu().numpy()

            # according to the technique get the prediction
            if technique == Technique.DDC:
                outputs, mmd_loss = net(inputs, targets)
            elif technique == Technique.DANN:
                outputs, _ = net(inputs)
            elif technique == Technique.DSN_MEDM:
                outputs, _, _, _ = net(inputs, rec_code=ReconstructionCode.TARGET)
            else:
                outputs = net(inputs)

            # prediction in numpy
            outputs_np = outputs.data.cpu().numpy()

            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)

    # get targets and output
    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    return targets, outputs


def gen_features_source_domain(
    net: nn.Module,
    source_dataloader: torch.utils.data.DataLoader,
    target_dataloader: torch.utils.data.DataLoader,
    technique: Technique,
    device: str = "cuda",
) -> Tuple[np.double, np.double]:
    r"""
    Function which generates the features for both the dataloaders according to
    the actual technique which has been chosen.

    In this case, each prediction is associated either to:

    - 0 in the case of source set samples
    - 1 in the case of target set samples

    This is used in order to see, thanks to PCA and tSNE, whether the distribution
    of the feature extracted are aligned or not.

    The code has been adapted from:
    [link]: https://github.com/2-Chae/PyTorch-tSNE/blob/main/main.py

    Args:

    - net [nn.Module]: network architecture
    - source_dataloader [torch.utils.data.DataLoader]: source dataloader where to compute the features associated to
    - target_dataloader [torch.utils.data.DataLoader]: target dataloader where to compute the features
    - technique [Technique]: technique to employ
    - device [str] = "cuda": where to load the tensors

    Returns:

    - targets [np.double]: targets
    - outputs [np.double]: features
    """
    net.eval()

    # set the network to feature extraction mode!
    net.feature_extractor = True

    if technique == Technique.DSN_MEDM:
        # remove the Flatten layer from the source_encoder sequential
        net.shared_encoder[9] = nn.Identity()
    targets_list = []
    outputs_list = []

    with torch.no_grad():
        # loop over the source dataloader
        for idx, (inputs, label) in enumerate(source_dataloader):
            inputs = inputs.to(device)
            label = label.to(device)

            # label here is zero for each of the source domain sample
            targets = torch.zeros(len(inputs)).to(device)
            targets_np = targets.data.cpu().numpy()

            if technique == Technique.DDC:
                outputs = net(inputs, label)
            elif technique == Technique.DSN_MEDM:
                outputs = net(inputs, rec_code=ReconstructionCode.SOURCE)
            else:
                outputs = net(inputs)

            outputs_np = outputs.data.cpu().numpy()

            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)

            if ((idx + 1) % 10 == 0) or (idx + 1 == len(source_dataloader)):
                print(idx + 1, "/", len(source_dataloader))

        # loop over the target dataloader
        for idx, (inputs, label) in enumerate(target_dataloader):
            inputs = inputs.to(device)
            label = label.to(device)

            # label here is one for the each of the target domain sample
            targets = torch.ones(len(inputs)).to(device)
            targets_np = targets.data.cpu().numpy()

            if technique == Technique.DDC:
                outputs = net(inputs, label)
            elif technique == Technique.DSN_MEDM:
                outputs = net(inputs, rec_code=ReconstructionCode.TARGET)
            else:
                outputs = net(inputs)

            outputs_np = outputs.data.cpu().numpy()
            targets_list.append(targets_np[:, np.newaxis])
            outputs_list.append(outputs_np)

            if ((idx + 1) % 10 == 0) or (idx + 1 == len(target_dataloader)):
                print(idx + 1, "/", len(target_dataloader))

    # set the network to prediction mode
    net.feature_extractor = False

    if technique == Technique.DSN_MEDM:
        # put once again the Flatten layer in the source_encoder sequential
        net.shared_encoder[9] = Flatten()

    print(np.asarray(outputs_list).shape)

    targets = np.concatenate(targets_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0).astype(np.float64)

    # flatten the features, since the feature map are not flattened and
    # tSNE and PCA require flat feature map
    nsamples, nx, ny, nz = outputs.shape
    outputs = outputs.reshape((nsamples, nx * ny * nz))

    return targets, outputs


def feature_visualizer(
    title: str,
    targets: np.double,
    outputs: np.double,
    n_classes: int,
    nc: float = 0.95,
    legend: List[str] = None,
) -> None:
    r"""
    Function which displays a PCA (preserving 95% of the data variance) and
    t-SNE according to the features passed, whose color is associated to the
    actual target value.

    The code has been adapted from:
    [link]: https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

    Args:

    - title [str]: network architecture
    - targets [np.double]: features to feed to the dimensionality reduction algorithm
    - outputs [np.double]: labels associated to the features
    - n_classes [int]: number of classes
    - n_c [float] = 0.95: number of PCA components, in this case it is the amount of variance to preserve
    - legend [List[str]]: legend to display
    """

    # font size
    SMALL_SIZE = 8
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 20

    print(" #> Generating PCA and t-SNE plots...")

    # Scale the PCA in order to employ 95% variance preserve data
    from sklearn.preprocessing import MinMaxScaler

    # color palette
    import colorcet as cc

    # rescale the data before PCA
    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(outputs)

    # running PCA
    pca = PCA(n_components=nc)
    pca_result = pca.fit_transform(data_rescaled)
    df_pca = pd.DataFrame(pca_result)
    # drop the columns which cannot be depicted in a two dimensional space
    # and save only the first two
    column_to_drop = [x for x in range(2, df_pca.shape[1])]
    df_pca = df_pca.drop(df_pca[column_to_drop], axis=1)
    df_pca = df_pca.rename({0: "pca-one", 1: "pca-two"}, axis=1)
    df_pca["targets"] = targets

    # running TSNE
    tsne = TSNE(random_state=0, learning_rate="auto", init="random")
    # use the 95% of variance of the PCA to save some time and still obtain good results
    # if something is not ok, we can change it to outputs
    tsne_output = tsne.fit_transform(pca_result)

    # taking the first two columns
    df_tsne = pd.DataFrame(tsne_output, columns=["x", "y"])
    df_tsne["targets"] = targets

    plt.figure(figsize=(20, 10))
    plt.suptitle(
        "PCA + t-SNE {}".format(title),
        x=0.1,
        y=0.95,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=20,
    )

    ax1 = plt.subplot(1, 2, 1)
    ax1.title.set_text("PCA")

    # TSNE plot
    sns.scatterplot(
        x="pca-one",
        y="pca-two",
        hue="targets",
        palette=sns.color_palette(cc.glasbey, n_colors=n_classes),
        data=df_pca,
        legend="full",
        alpha=0.3,
        ax=ax1,
    )

    # Put the legend out of the figure
    ax1.legend(
        labels=legend, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=2
    )

    # PCA plot
    ax2 = plt.subplot(1, 2, 2)
    ax2.title.set_text("t-SNE")
    sns.scatterplot(
        x="x",
        y="y",
        hue="targets",
        palette=sns.color_palette(cc.glasbey, n_colors=n_classes),
        data=df_tsne,
        marker="o",
        legend="full",
        alpha=0.5,
    )

    # Put the legend out of the figure
    ax2.legend(
        labels=legend, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, markerscale=2
    )

    # spacing between the subplots
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=1.9, top=0.9, wspace=0.4, hspace=0.4
    )
    plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc("legend", fontsize=BIGGER_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
    # save plots in a figure
    os.makedirs("plot", exist_ok=True)
    plt.savefig(
        os.path.join("plot/", "{}_pca+tsne.png".format(title)), bbox_inches="tight"
    )

    # show plots
    plt.show()
