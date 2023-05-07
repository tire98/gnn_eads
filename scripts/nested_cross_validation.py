"""Perform nested cross validation for GNN using the FG-dataset.
To run it, you can use the following command:
python nested_cross_validation_GNN.py -i input_hyperparams.toml -o ../results_dir
The input toml has the same structure as the one used for the training process."""

import argparse
from os.path import isdir, join
from os import mkdir, listdir
import time
import copy

import torch
import toml
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader

from gnn_eads.core.functions import scale_target, train_loop, test_loop
from gnn_eads.core.nets import FlexibleNet
from gnn_eads.core.post_training import create_model_report
from gnn_eads.core.process_ase_db_to_PyG_dataset import load_FG_dataset
from gnn_eads.core.constants import (
    loss_dict,
    pool_seq_dict,
    conv_layer,
    sigma_dict,
    pool_dict,
)
from gnn_eads.data import DATAPATH
from gnn_eads.models import MODELSPATH
from gnn_eads.toml import TOMLPATH


PARSER = argparse.ArgumentParser(
    description="Perform nested cross validation for GNN using the FG-dataset."
)
PARSER.add_argument(
    "-i",
    "--input",
    type=str,
    dest="i",
    help="Input toml file with hyperparameters for the nested cross validation.",
)
PARSER.add_argument("-o", "--output", type=str, dest="o", help="Output name.")
PARSER.add_argument("-d", "--dataset", type=str, dest="d", help="Dataset name.")
PARSER.add_argument("-db", "--database", type=str, dest="db", help="Database name.")
ARGS = PARSER.parse_args()


output_name = ARGS.o
output_directory = join(MODELSPATH, output_name)
if isdir("{}/{}".format(output_directory, output_name)):
    output_name = input(
        "There is already a model with the chosen name in the provided directory, provide a new one: "
    )
mkdir(output_directory)
# Upload training hyperparameters from toml file
toml_file = join(TOMLPATH, ARGS.i + ".toml")
HYPERPARAMS = toml.load(toml_file)
data_path = join(DATAPATH, ARGS.d)
graph_settings = HYPERPARAMS["graph"]
train = HYPERPARAMS["train"]
architecture = HYPERPARAMS["architecture"]

print("Nested cross validation for GNN using the FG-dataset")
print("Number of splits: {}".format(train["splits"]))
print("Total number of runs: {}".format(train["splits"] * (train["splits"] - 1)))
print("--------------------------------------------")
# Select device (GPU/CPU)
device_dict = {}
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Device name: {} (GPU)".format(torch.cuda.get_device_name(0)))
    device_dict["name"] = torch.cuda.get_device_name(0)
    device_dict["CudaDNN_enabled"] = torch.backends.cudnn.enabled
    device_dict["CUDNN_version"] = torch.backends.cudnn.version()
    device_dict["CUDA_version"] = torch.version.cuda
else:
    print("Device name: CPU")
    device_dict["name"] = "CPU"


def split_list(a: list, n: int):
    """
    Split a list into n chunks (for nested cross-validation)
    Args:
        a(list): list to split
        n(int): number of chunks
    Returns:
        (list): list of chunks
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def get_model_params(model_id):
    # get path from current working directory
    dirs = os.listdir(os.getcwd())
    for dir in dirs:
        if f"_{model_id}_" in dir:
            model_path = os.path.join(os.getcwd(), dir)
            break
    # read json as dict
    json_file_path = os.path.join(model_path, "params.json")

    with open(json_file_path, 'r') as j:
         contents = json.loads(j.read())
    
    return contents


def create_loaders_nested_cv(datasets, split, batch_size):
    """
    Create dataloaders for training, validation and test sets for nested cross-validation.
    Args:
        datasets(tuple): tuple containing the HetGraphDataset objects.
        split(int): number of splits to generate train/val/test sets
        batch(int): batch size
    Returns:
        (tuple): tuple with dataloaders for training, validation and testing.
    """
    # Create list of lists, where each list contains the datasets for a split.
    chunk = [[] for _ in range(split)]
    for dataset in datasets:
        dataset.shuffle()
        iterator = split_list(dataset, split)
        for index, item in enumerate(iterator):
            chunk[index] += item
        chunk = sorted(chunk, key=len)
    # Create dataloaders for each split.
    for index in range(len(chunk)):
        proxy = copy.copy(chunk)
        test_loader = DataLoader(proxy.pop(index), batch_size=batch_size, shuffle=False)
        for index2 in range(len(proxy)):  # length is reduced by 1 here
            proxy2 = copy.copy(proxy)
            val_loader = DataLoader(
                proxy2.pop(index2), batch_size=batch_size, shuffle=False
            )
            flatten_training = [
                item for sublist in proxy2 for item in sublist
            ]  # flatten list of lists
            train_loader = DataLoader(
                flatten_training, batch_size=batch_size, shuffle=True
            )
            yield copy.deepcopy((train_loader, val_loader, test_loader))


def main():
    FG_dataset = load_FG_dataset(
        root=data_path,
        database=ARGS.db,
        second_order=graph_settings["second_order"],
        scale_factor=graph_settings["scaling_factor"],
        tolerance=graph_settings["voronoi_tol"],
        edge_features=graph_settings["edge_features"],
        ring_features=graph_settings["ring_features"],
        aromatic_features=graph_settings["aromatic_features"],
        radical_features=graph_settings["radical_features"],
        relax=graph_settings["relax"],
        num_el=graph_settings["num_el"],
        family=graph_settings["family"],
    )

    # Instantiate iterator for nested cross validation: Each iteration yields a different train/val/test set combination
    ncv_iterator = create_loaders_nested_cv(
        FG_dataset, split=train["splits"], batch_size=train["batch_size"]
    )
    MAE_outer = []
    counter = 0
    TOT_RUNS = train["splits"] * (train["splits"] - 1)
    
    params_dict = get_model_params(ARGS.m)
    for outer in range(train["splits"]):
        MAE_inner = []
        for inner in range(train["splits"] - 1):
            counter += 1
            train_loader, val_loader, test_loader = next(ncv_iterator)
            train_loader, val_loader, test_loader, mean, std = scale_target(
                train_loader,
                val_loader,
                test_loader,
                mode=train["target_scaling"],
                test=True,
            )  # True is necessary condition for nested CV
            # Instantiate model, optimizer and lr-scheduler
            model = FlexibleNet(
                    dim=params_dict["dim"],
                    N_linear=params_dict["n_linear"],
                    N_conv=params_dict["n_conv"],
                    adj_conv=params_dict["adj_conv"],
                    sigma="ReLU",
                    in_features=19,
                    conv=conv_layer[params_dict["conv_layer"]],
                    pool=pool_dict["GMT"],
                    pool_ratio=params_dict["pool_ratio"],
                    pool_heads=params_dict["pool_heads"],
                    pool_seq=pool_seq_dict["1"],
                    pool_layer_norm=params_dict["pool_layer_norm"],
                    bias_input=params_dict["bias_input"],
                    bias_conv=params_dict["bias_conv"],
                    bias_dense=params_dict["bias_dense"],
                    bias_adj=params_dict["bias_adj"],
                    ).to(device)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=train["lr0"],
                eps=train["eps"],
                weight_decay=train["weight_decay"],
                amsgrad=train["amsgrad"],
            )
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=train["factor"],
                patience=train["patience"],
                min_lr=train["minlr"],
            )
            loss_list, train_list, val_list, test_list, lr_list = [], [], [], [], []
            t0 = time.time()
            # Run the learning
            for epoch in range(1, train["epochs"] + 1):
                torch.cuda.empty_cache()
                lr = lr_scheduler.optimizer.param_groups[0]["lr"]
                loss, train_MAE = train_loop(
                    model,
                    device,
                    train_loader,
                    optimizer,
                    loss_dict[train["loss_function"]],
                )
                val_MAE = test_loop(model, val_loader, device, std)
                lr_scheduler.step(val_MAE)
                test_MAE = test_loop(model, test_loader, device, std, mean)
                print(
                    "{}/{}-Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Validation MAE: {:.4f} eV "
                    "Test MAE: {:.4f} eV".format(
                        counter, TOT_RUNS, epoch, lr, train_MAE * std, val_MAE, test_MAE
                    )
                )
                test_list.append(test_MAE)
                loss_list.append(loss)
                train_list.append(train_MAE * std)
                val_list.append(val_MAE)
                lr_list.append(lr)
                if epoch == train["epochs"]:
                    MAE_inner.append(test_MAE)
            print(
                "-----------------------------------------------------------------------------------------"
            )
            training_time = (time.time() - t0) / 60
            print("Training time: {:.2f} min".format(training_time))
            device_dict["training_time"] = training_time
            create_model_report(
                "{}_{}".format(outer + 1, inner + 1),
                output_directory + "/" + output_name,
                HYPERPARAMS,
                model,
                (train_loader, val_loader, test_loader),
                (mean, std),
                (train_list, val_list, test_list, lr_list),
                device_dict,
            )
            del model, optimizer, lr_scheduler, train_loader, val_loader, test_loader
            if device == "cuda":
                torch.cuda.empty_cache()
        MAE_outer.append(np.mean(MAE_inner))
    MAE = np.mean(MAE_outer)
    print("Nested CV MAE: {:.4f} eV".format(MAE))
    # Generate report of the whole experiment
    ncv_results = listdir(output_directory + "/" + output_name)
    df = pd.DataFrame()
    for run in ncv_results:
        results = pd.read_csv(
            output_directory + "/" + output_name + "/" + run + "/test_set.csv", sep="\t"
        )
        # add column with run number
        results["run"] = run
        df = pd.concat([df, results], axis=0)
    df = df.reset_index(drop=True)
    df.to_csv(output_directory + "/" + output_name + "/summary.csv", index=False)


if __name__ == "__main__":
    main()
