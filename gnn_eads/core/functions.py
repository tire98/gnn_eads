"""Functions for converting DFT data to graphs and for learning process purposes."""

import math
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn_eads.core.constants import ELEMENT_LIST, METALS, MOL_ELEM


def split_percentage(splits: int, test: bool = True) -> tuple:
    """Return split percentage of the train, validation and test sets.

    Args:
        split (int): number of initial splits of the entire initial dataset

    Returns:
        a, b, c: train, validation, test percentage of the sets.
    """
    if test:
        a = int(100 - 200 / splits)
        b = math.ceil(100 / splits)
        return a, b, b
    else:
        return int((1 - 1 / splits) * 100), math.ceil(100 / splits)


def get_energy(dataset: str, paths_dict: dict) -> dict:
    """
    Extract the ground energy for each sample of the dataset from the energies.dat file.
    Args:
        dataset(str): Dataset's title.
    Returns:
        ener_dict(dict): Dictionary with raw total energy (sigma->0) [eV].
    """
    with open(paths_dict[dataset]["ener"], "r") as infile:
        lines = infile.readlines()
    ener_dict = {}
    for line in lines:
        split = line.split()
        ener_dict[split[0]] = float(split[1])
    return ener_dict


def export_tuples(filename: str, tuple_dict: dict):
    """
    Export processed DFT dataset into text file.
    Args:
        filename (str): file to write.
        tuple_dict (tuple): tuple dictionary containing all the graph information.
    """
    with open(filename, "w") as outfile:
        for code, inter in tuple_dict.items():
            lst_trans = lambda x: " ".join([str(y) for y in x])
            outfile.write(f"{code}\n")
            outfile.write(f"{lst_trans(inter.graph[1][0])}\n")
            outfile.write(f"{lst_trans(inter.graph[1][1][0])}\n")
            outfile.write(f"{lst_trans(inter.graph[1][1][1])}\n")
            outfile.write(f"{inter.energy}\n")


def geometry_to_graph_analysis(dataset: str, paths_dict: dict):
    """
    Check that all adsorption samples in the dataset are correctly
    converted to a graph.
    Args:
        dataset(str): Dataset's title.
    Returns:
        wrong_graphs(int): number of uncorrectly-converted samples, i.e., no metal atom is
                           present as node in the graph representation.
        wrong_samples(list): list of the badly represented data.
        dataset_size(int): dataset size.
    """
    with open(paths_dict[dataset]["dataset"]) as f:
        all_lines = f.readlines()
    dataset_size = int(len(all_lines) / 5)
    if dataset[:3] == "gas":
        print("{}: dataset of gas phase molecules".format(dataset))
        print("------------------------------------------")
        return 0, [], dataset_size

    lines = []
    labels = []
    for i in range(dataset_size):
        lines.append(
            all_lines[1 + 5 * i]
        )  # Read the second line of each graph (ex. "C H C H Ag")
        labels.append(all_lines[5 * i])  # Read label of each sample (ex. "ag-4a01-a")
    for i in range(dataset_size):
        lines[i] = lines[i].strip("\n")
        lines[i] = lines[i].split()
        labels[i] = labels[i].strip("\n")
    new_list = [[]] * dataset_size
    wrong_samples = []
    for i in range(dataset_size):
        new_list[i] = [
            lines[i][j] for j in range(len(lines[i])) if lines[i][j] not in MOL_ELEM
        ]
        if new_list[i] == []:
            wrong_samples.append(labels[i])
    wrong_graphs = int(new_list.count([]))
    print("Dataset: {}".format(dataset))
    print("Size: {}".format(dataset_size))
    print("Bad representations: {}".format(wrong_graphs))
    print(
        "Percentage of bad representations: {:.2f}%".format(
            (wrong_graphs / dataset_size) * 100
        )
    )
    print("-------------------------------------------")
    return wrong_graphs, wrong_samples, dataset_size


def get_graph_formula(
    graph: Data, categories: list = ELEMENT_LIST, metal_list: list = METALS
) -> str:
    """
    Create a string label for the selected graph.
    String format: xxxxxxxxxxxxxx (len=14)
    CxHyOzNwSt-mmx
    Args:
        graph(torch_geometric.data.Data): graph object.
        categories(list): list with element string labels.
        metal_list(list): list of metal atoms string.
    Returns:
        label(str): brute formula of the graph.
    """
    element_list = []
    for i in range(graph.num_nodes):
        for j in range(graph.num_features):
            if graph.x[i, j] == 1:
                element_list.append(j)
    element_array = [0] * len(categories)
    for element in range(len(categories)):
        for index in element_list:
            if element == index:
                element_array[element] += 1
    element_array = list(element_array)
    element_array = [int(i) for i in element_array]
    element_dict = dict(zip(categories, element_array))
    label = ""
    ss = ""
    for key in element_dict.keys():
        if element_dict[key] == 0:
            pass
        else:
            label += key
            label += str(element_dict[key])
    for metal in metal_list:
        if metal in label:
            index = label.index(metal)
            ss = label[index : index + 3]
    label = label.replace(ss, "")
    label += "-" + ss
    # label = label.replace("1", "")
    counter = 0
    for metal in metal_list:
        if metal in label:
            counter += 1
    if counter == 0:
        label += "(g)"
    # Standardize string length to 14
    diff = 14 - len(label)
    if diff > 0:
        extra_space = " " * diff
        label += extra_space
    return label


def get_number_atoms_from_label(formula: str, H_count: bool = True) -> int:
    """Get the total number of atoms in the adsorbate from a graph formula
    got from get_graph_formula."""

    # match elements and their counts
    pattern = r'([A-Z][a-z]*)(\d*)'
    # initialize a dictionary to store the element counts
    elements = {}
    # loop through each element in the formula
    for element, count in re.findall(pattern, formula):
        # if no count is given, assume 1
        count = int(count) if count else 1
        # add or update the element count in the dictionary
        elements[element] = elements.get(element, 0) + count
    # return the total number of atoms
    return sum(elements.values())


def create_loaders_fg_train_val(
    datasets,
    split: int = 5,
    batch_size: int = 32,
    test: bool = True,
) -> tuple:
    """
    Create dataloaders for training, validation and test.
    Args:
        datasets (tuple): tuple containing the HetGraphDataset objects.
        split (int): number of splits to generate train/val/test sets.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate train/val/test loaders or just train/val.
    Returns:
        (tuple): tuple with dataloaders for training, validation and test.
    """

    train_loader = []
    val_loader = []
    test_loader = []

    for dataset in datasets:
        # if dataset label is not "radicals" add the data set to train/val loaders
        if dataset.family not in ["radicals", "biomass", "cyclic"]:
            n_items = len(dataset)
            sep = n_items // split
            dataset = dataset.shuffle()
            val_loader += dataset[:sep]
            train_loader += dataset[sep:]
        else:
            n_items = len(dataset)
            sep = n_items // split
            dataset = dataset.shuffle()
            if test:
                test_loader += dataset[:sep]
                val_loader += dataset[sep : sep * 2]
                train_loader += dataset[sep * 2 :]
            else:
                val_loader += dataset[:sep]
                train_loader += dataset[sep:]
    train_n = len(train_loader)
    val_n = len(val_loader)
    test_n = len(test_loader)
    total_n = train_n + val_n + test_n
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)
    if test:
        test_loader = DataLoader(test_loader, batch_size=batch_size)
        a, b, c = split_percentage(split)
        print("Data split (train/val/test): {}/{}/{} %".format(a, b, c))
        print(
            "Training data = {} Validation data = {} Test data = {} (Total = {})".format(
                train_n, val_n, test_n, total_n
            )
        )
        return (train_loader, val_loader, test_loader)
    else:
        print(
            "Data split (train/val): {}/{} %".format(
                int(100 * (split - 1) / split), int(100 / split)
            )
        )
        print(
            "Training data = {} Validation data = {} (Total = {})".format(
                train_n, val_n, total_n
            )
        )
        return (train_loader, val_loader, None)


def create_loaders(
    datasets, split: int = 5, batch_size: int = 32, test: bool = True
) -> tuple:
    """
    Create dataloaders for training, validation and test.
    Args:
        datasets (tuple): tuple containing the HetGraphDataset objects.
        split (int): number of splits to generate train/val/test sets.
        batch_size (int): batch size. Default to 32.
        test (bool): Whether to generate train/val/test loaders or just train/val.
    Returns:
        (tuple): tuple with dataloaders for training, validation and test.
    """

    train_loader = []
    val_loader = []
    test_loader = []

    for dataset in datasets:
        # if dataset label is not "radicals" add the data set to train/val loaders

        n_items = len(dataset)
        sep = n_items // split
        dataset = dataset.shuffle()
        if test:
            test_loader += dataset[:sep]
            val_loader += dataset[sep : sep * 2]
            train_loader += dataset[sep * 2 :]
        else:
            val_loader += dataset[:sep]
            train_loader += dataset[sep:]
    train_n = len(train_loader)
    val_n = len(val_loader)
    test_n = len(test_loader)
    total_n = train_n + val_n + test_n
    train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)
    if test:
        test_loader = DataLoader(test_loader, batch_size=batch_size)
        a, b, c = split_percentage(split)
        print("Data split (train/val/test): {}/{}/{} %".format(a, b, c))
        print(
            "Training data = {} Validation data = {} Test data = {} (Total = {})".format(
                train_n, val_n, test_n, total_n
            )
        )
        return (train_loader, val_loader, test_loader)
    else:
        print(
            "Data split (train/val): {}/{} %".format(
                int(100 * (split - 1) / split), int(100 / split)
            )
        )
        print(
            "Training data = {} Validation data = {} (Total = {})".format(
                train_n, val_n, total_n
            )
        )
        return (train_loader, val_loader, None)


def scale_target(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader = None,
    mode: str = "std",
    verbose: bool = True,
    test: bool = True,
):
    """
    Apply target scaling to the whole dataset using labels of
    training and validation sets.
    Args:
        train_loader (torch_geometric.loader.DataLoader): training dataloader
        val_loader (torch_geometric.loader.DataLoader): validation dataloader
        test_loader (torch_geometric.loader.DataLoader): test dataloader
    Returns:
        train, val, test: dataloaders with scaled target values
        mean_tv, std_tv: mean and std (standardization)
        min_tv, max_tv: min and max (normalization)
    """
    # 1) Get mean-std/min-max from train and validation sets
    y_list = []
    for graph in train_loader.dataset:
        y_list.append(graph.y.item())
    for graph in val_loader.dataset:
        y_list.append(graph.y.item())
    y_tensor = torch.tensor(y_list)
    mean_tv = y_tensor.mean(
        dim=0, keepdim=True
    )  # _tv stands for "train+validation sets"
    std_tv = y_tensor.std(dim=0, keepdim=True)
    max_tv = y_tensor.max()
    min_tv = y_tensor.min()
    # 2) Apply Scaling (Standardization or Normalization)
    for graph in train_loader.dataset:
        if mode == "std":
            graph.y = (graph.y - mean_tv) / std_tv
        elif mode == "norm":
            graph.y = (graph.y - min_tv) / (max_tv - min_tv)
        else:
            pass
    for graph in val_loader.dataset:
        if mode == "std":
            graph.y = (graph.y - mean_tv) / std_tv
        elif mode == "norm":
            graph.y = (graph.y - min_tv) / (max_tv - min_tv)
        else:
            pass
    if test:
        for graph in test_loader.dataset:
            if mode == "std":
                graph.y = (graph.y - mean_tv) / std_tv
            elif mode == "norm":
                graph.y = (graph.y - min_tv) / (max_tv - min_tv)
            else:
                pass
    if mode == "std":
        if verbose:
            print("Target Scaling (Standardization) applied successfully")
            print("(Train+Val) mean: {:.2f} eV".format(mean_tv.item()))
            print("(Train+Val) standard deviation: {:.2f} eV".format(std_tv.item()))
        if test:
            return train_loader, val_loader, test_loader, mean_tv.item(), std_tv.item()
        else:
            return train_loader, val_loader, None, mean_tv.item(), std_tv.item()
    elif mode == "norm":
        if verbose:
            print("Target Scaling (Normalization) applied successfully")
            print("(Train+Val) min: {:.2f} eV".format(min_tv.item()))
            print("(Train+Val) max: {:.2f} eV".format(max_tv.item()))
        if test:
            return train_loader, val_loader, test_loader, min_tv.item(), max_tv.item()
        else:
            return train_loader, val_loader, None, min_tv.item(), max_tv.item()
    else:
        print("Target Scaling NOT applied")
        return train_loader, val_loader, test_loader, 0, 1


def train_loop(model, device: str, train_loader: DataLoader, optimizer, loss_fn):
    """
    Helper function for model training over an epoch.
    For each batch in the epoch, the following actions are performed:
    1) Move the batch to the selected device for training
    2) Forward pass through the GNN model and loss function computation
    3) Compute gradient of loss function wrt model parameters
    4) Update model parameters
    Args:
        model(): GNN model object.
        device(str): device on which training is performed.
        train_loader(): Training dataloader.
        optimizer(): optimizer used during training.
        loss_fn(): Loss function used for the training.
    Returns:
        loss_all, mae_all (tuple[float]): Loss function and MAE of the whole epoch.
    """
    model.train()
    loss_all = 0
    mae_all = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()  # Set gradients of all tensors to zero
        loss = loss_fn(model(batch), batch.y)
        mae = F.l1_loss(model(batch), batch.y)  # For comparison with val/test data
        loss.backward()  # Get gradient of loss function wrt parameters
        loss_all += loss.item() * batch.num_graphs
        mae_all += mae.item() * batch.num_graphs
        optimizer.step()  # Update model parameters
    loss_all /= len(train_loader.dataset)
    mae_all /= len(train_loader.dataset)
    return loss_all, abs(mae_all)


def test_loop(
    model,
    loader: DataLoader,
    device: str,
    std: float,
    mean: float = None,
    scaled_graph_label: bool = True,
    verbose: int = 0,
) -> float:
    """
    Helper function for validation/testing loop.
    For each batch in the validation/test epoch, the following actions are performed:
    1) Set the GNN model in evaluation mode
    2) Move the batch to the selected device where the model is stored
    3) Compute the Mean Absolute Error (MAE)
    Args:
        model (): GNN model object.
        loader (Dataloader object): Dataset for validation/testing.
        device (str): device on which training is performed.
        std (float): standard deviation of the training+validation datasets [eV]
        mean (float): mean of the training+validation datasets [eV]
        scaled_graph_label (bool): whether the graph labels are in eV or in a scaled format.
        verbose (int): 0=no printing info 1=printing information
    Returns:
        error(float): Mean Absolute Error (MAE) of the test loader.
    """
    model.eval()
    error = 0
    for batch in loader:
        batch = batch.to(device)
        if scaled_graph_label is False:  # label in eV
            error += (model(batch) * std + mean - batch.y).abs().sum().item()
        else:  # Scaled label
            error += (model(batch) * std - batch.y * std).abs().sum().item()
    error /= len(loader.dataset)  # Mean Absolute Error

    if verbose == 1:
        print("Dataset size = {}".format(len(loader.dataset)))
        print("Mean Absolute Error = {} eV".format(error))
    return error


def get_mean_std_from_model(path: str) -> tuple:
    """Get mean and standard deviation used for scaling the target values
       from the selected trained model.

    Args:
        model_name (str): GNN model path.

    Returns:
        mean, std (tuple[float]): mean and standard deviation for scaling the targets.
    """
    file = open("{}/performance.txt".format(path), "r")
    lines = file.readlines()
    for line in lines:
        if "(train+val) mean" in line:
            mean = float(line.split()[-2])
        if "(train+val) standard deviation" in line:
            std = float(line.split()[-2])
    return mean, std


def get_graph_conversion_params(path: str) -> tuple:
    """Get the hyperparameters for geometry->graph conversion algorithm.

    Args:
        path (str): path to GNN model.

    Returns:
        tuple: voronoi tolerance (float), scaling factor (float), metal nearest neighbours inclusion (bool)
    """
    file = open("{}/performance.txt".format(path), "r")
    lines = file.readlines()
    for line in lines:
        if "Voronoi" in line:
            voronoi_tol = float(line.split()[-2])
        if "scaling factor" in line:
            scaling_factor = float(line.split()[-1])
        if "Second order" in line:
            if line.split()[-1] == "True":
                second_order_nn = True
            else:
                second_order_nn = False
    return voronoi_tol, scaling_factor, second_order_nn


def get_id(graph_params: dict) -> str:
    """
    Returns string identifier associated to a specific graph representation setting,
    consistsing of tolerance, scaling factor, 2-hop metals inclusion in the
    conversion from geometry to graph.
    Args
        graph_params (dict): dictionary containing graph settings:
            {"voronoi_tol": (float), "second_order_nn": (bool), "scaling_factor": float}
    Returns
        identifier (str): String defining graph settings.
    """
    identifier = str(graph_params["voronoi_tol"]).replace(".", "")
    identifier += "_"
    identifier += str(graph_params["second_order_nn"])
    identifier += "_"
    identifier += str(graph_params["scaling_factor"]).replace(".", "")
    identifier += ".dat"
    return identifier


def set_seed(seed: int):
    """Set seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopper:
    """Early stopper for training loop."""

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_validation_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
