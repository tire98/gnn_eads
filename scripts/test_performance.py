"""script for testing the model performance on the C3 dataset"""
import argparse
import os

import numpy as np
import pandas as pd
import toml
import torch
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from ase.db import connect

from gnn_eads.core.nets import PreTrainedModel, FlexibleNet
from gnn_eads.core.constants import (conv_layer, pool_seq_dict, pool_dict, sigma_dict)
from gnn_eads.core.process_ase_db_to_PyG_dataset import load_FG_dataset
from gnn_eads.data import DATAPATH
from gnn_eads.models import MODELSPATH
from gnn_eads.toml import TOMLPATH

# Create the parser
parser = argparse.ArgumentParser(
    description="Get performance stats of model on C3 dataset."
)
parser.add_argument("--model_name", type=str, help="Name of the model to be tested")
parser.add_argument("--dataset", type=str, help="Dataset name to be evaluated")
parser.add_argument(
    "--database", type=str, help="Database name the dataset is derived from"
)
parser.add_argument(
    "--toml_file", type=str, help="Name of the toml file the test set is defined in. This file should contain the hyperparameters of the model."
)
parser.add_argument("--mean", type=float, help="Mean of the training + val set.")
parser.add_argument("--std", type=float, help="Standard deviation of the training + val set.")
args = parser.parse_args()

# Get information from provided .toml file
toml_file = os.path.join(TOMLPATH, args.toml_file + ".toml")
HYPERPARAMS = toml.load(toml_file)
data_path = os.path.join(DATAPATH, args.dataset)
model_path = os.path.join(MODELSPATH, "final_trainings", args.model_name)
graph_settings = HYPERPARAMS["graph"]
train_settings = HYPERPARAMS["train"]
architecture = HYPERPARAMS["architecture"]
seed = train_settings["seed"]
print("Model is loaded from: ", model_path)

# Load dataset
print("Data is loaded from: ", data_path)
seed_everything(0)
proc_data = load_FG_dataset(
    root=data_path,
    database=args.database,
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
dataset = []
for i in proc_data:
    for j in i:
        dataset.append(j)

dataloader = DataLoader(
    dataset, batch_size=16, shuffle=False
)


# Load model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
if "GNN.pth" in os.listdir(model_path):
    model = PreTrainedModel(model_path)
    mean = model.mean
    std = model.std
    model = model.model.to(device)
else:
    model = FlexibleNet(
        dim=architecture["dim"],
        N_linear=architecture["n_linear"],
        N_conv=architecture["n_conv"],
        adj_conv=architecture["adj_conv"],
        sigma=sigma_dict[architecture["sigma"]],
        edge_dim=proc_data[0].num_edge_features,
        in_features=proc_data[0].num_node_features,
        conv=conv_layer[architecture["conv_layer"]],
        pool=pool_dict[architecture["pool_layer"]],
        pool_ratio=architecture["pool_ratio"],
        pool_heads=architecture["pool_heads"],
        pool_seq=pool_seq_dict[architecture["pool_seq"]],
        pool_layer_norm=architecture["pool_layer_norm"],
        bias_input=architecture["bias_input"],
        bias_conv=architecture["bias_conv"],
        bias_dense=architecture["bias_dense"],
        bias_adj=architecture["bias_adj"],
    ).to(device)

    state_dict = os.path.join(model_path, "model.pth")
    model.load_state_dict(torch.load(state_dict, map_location=torch.device(str(device))))
    mean = args.mean
    std = args.std

formula, data_id, category, y_true, y_pred = [], [], [], [], []
for batch in dataloader:
    batch = batch.to(device)
    with torch.no_grad():
        y_pred += model(batch)
        y_true += batch.y
y_pred = [y_pred[i].item() * std + mean for i in range(len(dataloader.dataset))]
y_true = [y_true[i].item() for i in range(len(dataloader.dataset))]
# Test model
for i in range(len(dataloader.dataset)):
    graph = dataset[i]
    formula.append(graph.formula)
    data_id.append(graph.id.item())
    category.append(graph.family)
    # true.append(graph.ener.item())


# Create report
# create dataframe
df_dataset = pd.DataFrame(
    {
        "formula": formula,
        "data_id": data_id,
        "family": category,
        "true": y_true,
        "pred": y_pred,
    }
)
# get e_frag from database


def get_e_frag(data_id):
    with connect(proc_data[0].db_name) as conn:
        return conn.get(id=data_id).get("e_frag")


# define new column by applying get_e_frag function
df_dataset["e_gas"] = df_dataset["data_id"].apply(get_e_frag)

df_dataset["true_scaled"] = df_dataset["true"] - df_dataset["e_gas"]
df_dataset["pred_scaled"] = df_dataset["pred"] - df_dataset["e_gas"]

df_dataset["error"] = df_dataset["true"] - df_dataset["pred"]
df_dataset["abs_error"] = np.abs(df_dataset["error"])
df_dataset["abs_error_scaled"] = np.abs(df_dataset["true_scaled"] - df_dataset["pred_scaled"])
df_dataset

std_error = df_dataset["error"].std()

# calculate statistics
MAE = df_dataset["abs_error"].mean()
RMSE = np.sqrt((df_dataset["error"] ** 2).mean())
R2 = (
    1
    - (df_dataset["abs_error_scaled"] ** 2).sum()
    / ((df_dataset["true_scaled"] - df_dataset["true_scaled"].mean()) ** 2).sum()
)
MDAE = df_dataset["abs_error"].median()

## error distribution plot metrics
mean = df_dataset["error"].mean()
median = df_dataset["error"].median()
std = df_dataset["error"].std()


# print statistics
print("MAE: {:.2f} eV".format(MAE))
print("RMSE: {:.2f} eV".format(RMSE))
print("R2: {:.2f}".format(R2))
print("MDAE: {:.2f} eV".format(MDAE))
print("Mean: {:.2f} eV".format(mean))
print("Median: {:.2f} eV".format(median))
print("Std: {:.2f} eV".format(std))

# get outliers
df_outliers = df_dataset[df_dataset["abs_error"] >= 3 * std_error]
df_outliers_formulas_enery = df_outliers[["formula", "true", "pred", "abs_error"]]

# write report
with open(os.path.join(model_path, args.dataset + "_report.txt"), "w") as f:
    f.write(
        "Report for model: "
        + args.model_name
        + " on dataset: "
        + args.dataset
        + " from database: "
        + args.database
        + "\n"
    )
    f.write("--------------------------------------------\n")
    f.write("Statistics: \n")
    f.write("MAE: {:.2f} eV \n".format(MAE))
    f.write("RMSE: {:.2f} eV \n".format(RMSE))
    f.write("R2: {:.2f} \n".format(R2))
    f.write("MDAE: {:.2f} eV \n".format(MDAE))
    f.write("Mean: {:.2f} eV \n".format(mean))
    f.write("Median: {:.2f} eV \n".format(median))
    f.write("Std: {:.2f} eV \n".format(std))
    f.write("Dataset size: " + str(len(df_dataset)) + "\n")
    f.write("--------------------------------------------\n")
    f.write("Outliers: \n")
    f.write("Systems: " + str(df_outliers_formulas_enery) + "\n")
    f.write("--------------------------------------------\n")
    f.write("End of report. \n")
# save dataframe to csv

df_dataset.to_csv(os.path.join(model_path, args.dataset + "_report.csv"))