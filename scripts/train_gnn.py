"""Script to run GNN training on tekla2"""
import argparse
import os
import time

import toml
import torch

from gnn_eads.core.constants import (conv_layer, loss_dict, pool_dict,
                                     pool_seq_dict, sigma_dict)
from gnn_eads.core.functions import (create_loaders, scale_target, set_seed,
                                     test_loop, train_loop)
from gnn_eads.core.nets import FlexibleNet
from gnn_eads.core.post_training import create_model_report
from gnn_eads.core.process_ase_db_to_PyG_dataset import load_FG_dataset
from gnn_eads.data import DATAPATH
from gnn_eads.toml import TOMLPATH

# get specific toml file name from command line
parser = argparse.ArgumentParser(
    description="Run training with settings from .toml file."
)
parser.add_argument(
    "--toml_file", type=str, help="toml file name which is in the toml folder"
)
parser.add_argument("--model_name", type=str, help="name of the model to be saved")
parser.add_argument("--dataset", type=str, help="dataset name")
parser.add_argument("--database", type=str, help="database name")
args = parser.parse_args()

# load specifications from toml file
toml_file = os.path.join(TOMLPATH, args.toml_file + ".toml")
HYPERPARAMS = toml.load(toml_file)
data_path = os.path.join(DATAPATH, args.dataset)
data_path_propylene = os.path.join(DATAPATH, "propylene_test")
graph_settings = HYPERPARAMS["graph"]
train = HYPERPARAMS["train"]
architecture = HYPERPARAMS["architecture"]
print("Data is loaded from: ", data_path)

# load dataset
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

print(proc_data)


set_seed(train["seed"])

train_loader, val_loader, test_loader = create_loaders(
    proc_data, train["splits"], train["batch_size"], train["test_set"]
)

train_loader, val_loader, test_loader, mean_min, std_max = scale_target(
    train_loader,
    val_loader,
    test_loader=test_loader,
    mode=train["target_scaling"],
    test=train["test_set"],
)
if train["target_scaling"] == "std":
    res_factor = std_max
elif train["target_scaling"] == "norm":
    res_factor = std_max - mean_min

# device selection
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

# GNN model instantiation
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

# Optimizer and loss function
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=train["lr0"],
    eps=train["eps"],
    weight_decay=train["weight_decay"],
    amsgrad=train["amsgrad"],
)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=train["factor"],
    patience=train["patience"],
    min_lr=train["minlr"],
)

# Run training
loss_list, train_list, val_list, test_list = [], [], [], []
t0 = time.time()

for epoch in range(1, train["epochs"] + 1):
    torch.cuda.empty_cache()
    lr = lr_scheduler.optimizer.param_groups[0]["lr"]
    loss, train_MAE = train_loop(
        model, device, train_loader, optimizer, loss_dict[train["loss_function"]]
    )
    val_MAE = test_loop(model, val_loader, device, res_factor)
    lr_scheduler.step(val_MAE)
    if train["test_set"]:
        test_MAE = test_loop(model, test_loader, device, res_factor)
        print(
            "Epoch {:03d}: LR={:.7f}  Train MAE: {:.6f} eV  Validation MAE: {:.6f} eV  Test MAE: {:.6f} eV ".format(
                epoch, lr, train_MAE * res_factor, val_MAE, test_MAE
            )
        )
    else:
        print(
            "Epoch {:03d}: LR={:.7f}  Train MAE: {:.6f} eV  Validation MAE: {:.6f} eV ".format(
                epoch, lr, train_MAE * res_factor, val_MAE
            )
        )
    train_list.append(train_MAE)
    loss_list.append(loss)
    val_list.append(val_MAE)
print(
    "---------------------------------------------------------------------------------"
)
training_time = (time.time() - t0) / 60
print("Training time: {:.2f} min".format(training_time))
device_dict["training_time"] = training_time
# Save model
create_model_report(
    args.model_name,
    "../game_net/models/final_trainings",
    # Provide a name different from models present in the directory "models"
    HYPERPARAMS,
    model,
    (train_loader, val_loader, test_loader),
    (mean_min, std_max),
    (train_list, val_list, test_list),
    device_dict,
    toml_file=toml_file,
)
