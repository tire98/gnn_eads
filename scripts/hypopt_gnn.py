"""Run Hyperparameter optimization with Ray Tune"""

import argparse
import os

import pandas as pd
import ray
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphMultisetTransformer

from gnn_eads import MODULEROOT
from gnn_eads.core.constants import conv_layer, pool_seq_dict
from gnn_eads.core.functions import (create_loaders, scale_target, test_loop,
                                     train_loop)
from gnn_eads.core.nets import FlexibleNet
from gnn_eads.core.process_ase_db_to_PyG_dataset import load_FG_dataset
from gnn_eads.data import DATAPATH
from gnn_eads.hypopt import HYPOPTPATH
os.environ["TUNE_RESULT_DELIM"] = "/"

# --------------------------------------------------------------------------------------------#

HYPERPARAMS = {}
# Dataset

# Graph settings (THESE ARE NOT HYPERPARAMS, DO NOT SWITCH THEM TO VARIABLES!)
HYPERPARAMS["voronoi_tol"] = 0.5  # tune.choice([0.25, 0.50])
HYPERPARAMS["scaling_factor"] = 1.5  # tune.choice([1.3, 1.5, 1.7])
HYPERPARAMS["second_order"] = False
HYPERPARAMS["ring_features"] = False
HYPERPARAMS["edge_features"] = False
HYPERPARAMS["aromatic_features"] = False
HYPERPARAMS["radical_features"] = True
HYPERPARAMS["relax"] = True
HYPERPARAMS["num_el"] = True
HYPERPARAMS["family"] = [
    "amides",
    "amidines",
    "group2",
    "group2b",
    "group3S",
    "group3N",
    "group4",
    "carbamate_esters",
    "oximes",
    "radicals",
]
HYPERPARAMS["write_db"] = False

# Process-related
HYPERPARAMS["test_set"] = False
HYPERPARAMS["splits"] = 10
HYPERPARAMS["target_scaling"] = "std"  # tune.choice(["std", "norm"])
HYPERPARAMS["batch_size"] = 16
HYPERPARAMS["epochs"] = 200
HYPERPARAMS["loss_function"] = torch.nn.L1Loss()
HYPERPARAMS["lr0"] = 0.001
HYPERPARAMS["patience"] = 7
HYPERPARAMS["factor"] = 0.7
HYPERPARAMS["minlr"] = 1e-7
HYPERPARAMS["betas"] = (0.9, 0.999)
HYPERPARAMS["eps"] = 1e-9
HYPERPARAMS["weight_decay"] = 0
HYPERPARAMS["amsgrad"] = True
# Model-related
HYPERPARAMS["dim"] = tune.choice([160, 176, 192, 208, 224, 240, 256])
HYPERPARAMS["sigma"] = torch.nn.ReLU()
HYPERPARAMS["bias_input"] = tune.choice([True, False])
HYPERPARAMS["bias_conv"] = tune.choice([True, False])
HYPERPARAMS["bias_dense"] = tune.choice([True, False])
HYPERPARAMS["bias_adj"] = False
HYPERPARAMS["adj_conv"] = False
HYPERPARAMS["n_linear"] = tune.choice([0, 1, 2])
HYPERPARAMS["n_conv"] = tune.choice([1, 2, 3])
HYPERPARAMS["conv_layer"] = "SAGE"
HYPERPARAMS["pool_layer"] = GraphMultisetTransformer
HYPERPARAMS["conv_normalize"] = False
HYPERPARAMS["conv_root_weight"] = True
HYPERPARAMS["pool_ratio"] = tune.choice([0.25, 0.5, .75])
HYPERPARAMS["pool_heads"] = tune.choice([1, 2])
HYPERPARAMS["pool_seq"] = "1"
HYPERPARAMS["pool_layer_norm"] = False
# --------------------------------------------------------------------------------------------#
PARSER = argparse.ArgumentParser(
    description='Perform hyperparameter optimization with Ray-Tune (ASHA algorithm). \
                                 The output is stored in the "hyperparameter_optimization" directory.'
)
PARSER.add_argument(
    "-o",
    "--output",
    type=str,
    dest="o",
    help="Name of the hyperparameter optimization run.",
)
PARSER.add_argument(
    "-s",
    "--samples",
    default=5,
    type=int,
    dest="s",
    help="Number of trials of the search.",
)
PARSER.add_argument(
    "-v",
    "--verbose",
    default=1,
    type=int,
    dest="v",
    help="Verbosity of tune.run() function",
)
PARSER.add_argument(
    "-gr", "--grace", default=15, type=int, dest="grace", help="Grace period of ASHA."
)
PARSER.add_argument(
    "-maxit",
    "--max-iterations",
    default=150,
    type=int,
    dest="max_iter",
    help="Maximum number of training iterations (epochs) allowed by ASHA.",
)
PARSER.add_argument(
    "-rf",
    "--reduction-factor",
    default=4,
    type=int,
    dest="rf",
    help="Reduction factor of ASHA.",
)
PARSER.add_argument(
    "-bra",
    "--brackets",
    default=1,
    type=int,
    dest="bra",
    help="Number of brackets of ASHA.",
)
PARSER.add_argument(
    "-gpt",
    "--gpu-per-trial",
    default=1.0,
    type=float,
    dest="gpu_per_trial",
    help="Number of gpus per trial (can be fractional).",
)
ARGS = PARSER.parse_args()


def get_hyp_space(config: dict):
    """Return the total number of possible hyperparameters combinations.

    Args:
        config (dict): Hyperparameter configuration setting
    """
    x = 1
    counter = 0
    for key in list(config.keys()):
        if type(config[key]) == tune.search.sample.Categorical:
            x *= len(config[key])
            counter += 1
    return counter, x


def train_function(config: dict):
    """
    Helper function for hyperparameter tuning with RayTune.
    Args:
        config (dict): Dictionary with search space (hyperparameters)
    """
    # Generate graph dataset for training
    seed_everything(42)
    FG_dataset = load_FG_dataset(
        root=os.path.join(DATAPATH, "GAME_Net_2"),
        database="game_net_2_all.db",
        second_order=HYPERPARAMS["second_order"],
        scale_factor=HYPERPARAMS["scaling_factor"],
        tolerance=HYPERPARAMS["voronoi_tol"],
        edge_features=HYPERPARAMS["edge_features"],
        ring_features=HYPERPARAMS["ring_features"],
        aromatic_features=HYPERPARAMS["aromatic_features"],
        radical_features=HYPERPARAMS["radical_features"],
        relax=HYPERPARAMS["relax"],
        num_el=HYPERPARAMS["num_el"],
        family=HYPERPARAMS["family"],
    )

    # Load C3 dataloader only for testing
    P_dataset = load_FG_dataset(
        root=os.path.join(DATAPATH, "propylene_test"),
        database="propylene_c3_cu111_database.db",
        second_order=HYPERPARAMS["second_order"],
        scale_factor=HYPERPARAMS["scaling_factor"],
        tolerance=HYPERPARAMS["voronoi_tol"],
        edge_features=HYPERPARAMS["edge_features"],
        ring_features=HYPERPARAMS["ring_features"],
        aromatic_features=HYPERPARAMS["aromatic_features"],
        radical_features=HYPERPARAMS["radical_features"],
        relax=HYPERPARAMS["relax"],
        num_el=HYPERPARAMS["num_el"],
        family=["propylene_111"],
    )
    print(P_dataset)
    P_dataloader = DataLoader(P_dataset[0], batch_size=16, shuffle=False)

    NODE_FEATURES = FG_dataset[0].num_node_features
    # Data splitting and target scaling
    train_loader, val_loader, test_loader = create_loaders(
        FG_dataset, config["splits"], config["batch_size"], config["test_set"]
    )
    train_loader, val_loader, test_loader, mean_min, std_max = scale_target(
        train_loader,
        val_loader,
        test_loader,
        mode=config["target_scaling"],
        test=config["test_set"],
    )
    if config["target_scaling"] == "std":
        res_factor = std_max
    elif config["target_scaling"] == "norm":
        res_factor = std_max - mean_min

    # Define device, model, optimizer and lr-scheduler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FlexibleNet(
        dim=config["dim"],
        N_linear=config["n_linear"],
        N_conv=config["n_conv"],
        adj_conv=config["adj_conv"],
        in_features=NODE_FEATURES,
        edge_dim=FG_dataset[0].num_edge_features,
        sigma=config["sigma"],
        conv=conv_layer[config["conv_layer"]],
        pool=config["pool_layer"],
        pool_ratio=config["pool_ratio"],
        pool_heads=config["pool_heads"],
        pool_seq=pool_seq_dict[config["pool_seq"]],
        pool_layer_norm=config["pool_layer_norm"],
        bias_input=config["bias_input"],
        bias_conv=config["bias_conv"],
        bias_dense=config["bias_dense"],
        bias_adj=config["bias_adj"],
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config["lr0"],
        betas=config["betas"],
        eps=config["eps"],
        weight_decay=config["weight_decay"],
        amsgrad=config["amsgrad"],
    )

    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["factor"],
        patience=config["patience"],
        min_lr=config["minlr"],
    )

    # Training process
    train_list, val_list, test_list = [], [], []
    P_list = []
    for iteration in range(1, config["epochs"] + 1):
        torch.cuda.empty_cache()
        lr = lr_scheduler.optimizer.param_groups[0]["lr"]
        _, train_MAE = train_loop(
            model, device, train_loader, optimizer, config["loss_function"]
        )
        val_MAE = test_loop(model, val_loader, device, res_factor)
        lr_scheduler.step(val_MAE)
        if config["test_set"]:
            test_MAE = test_loop(model, test_loader, device, res_factor)
            test_list.append(test_MAE)
            p_MAE = test_loop(model, P_dataloader, device, res_factor, mean_min, scaled_graph_label=False)
            print(
                "Epoch {:03d}: LR={:.7f}  Train MAE: {:.4f} eV  Validation MAE: {:.4f} eV "
                "Test MAE: {:.4f} eV  P_MAE: {:.3f} eV".format(
                    iteration, lr, train_MAE * res_factor, val_MAE, test_MAE, p_MAE
                )
            )
        else:
            p_MAE = test_loop(model, P_dataloader, device, res_factor, mean_min, scaled_graph_label=False)
            print(
                "Epoch {:03d}: LR={:.7f}  Train MAE: {:.6f} eV  Validation MAE: {:.6f} eV  P_MAE: {:.3f} eV".format(
                    iteration, lr, train_MAE * res_factor, val_MAE, p_MAE
                )
            )

        # Test on C3 dataset

        train_list.append(train_MAE * res_factor)
        val_list.append(val_MAE)
        P_list.append(p_MAE)
        tune.report(P_MAE=p_MAE, epoch=iteration)
        # if P_MAE <= 0.4, save the model with this model with the hypopt run name
        if p_MAE <= 0.25:
            torch.save(model.state_dict(), os.path.join(MODULEROOT, "models", "hypopt", "model_{}_epoch{}.pt".format(tune.get_trial_id(), iteration)))


hypopt_scheduler = ASHAScheduler(
    time_attr="epoch",
    metric="P_MAE",
    mode="min",
    grace_period=ARGS.grace,
    reduction_factor=ARGS.rf,
    max_t=ARGS.max_iter,
    brackets=ARGS.bra,
)


def main():
    counter, x = get_hyp_space(HYPERPARAMS)
    print("Hyperparameters investigated: {}".format(counter))
    print("Hyperparameter space: {} possible combinations".format(x))
    print("Number of trials: {} ({:.2f} % of space covered)".format(ARGS.s, ARGS.s * 100.0 / x))
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": MODULEROOT,
            "excludes": ["*.pt", "*.db", "*.csv", "*.pth", "*.git", "*.dat", "*.ipynb", "*.json", "*.txt", "*.log", "*.out", "*.err", "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.exe", "*.obj", "*.o", "*.a", "*.lib", ],
        },
        num_gpus=11,
    )
    # CUDA_VISIBLE_DEVICES = os.environ("export CUDA_VISIBLE_DEVICES=GPU-18cb81a0-cb62-fc7c-aff9-d3c02648a90d,GPU-fab09db5-f9a2-dc33-2826-aab596c71a70,MIG-7ef07780-7be3-5b08-bd37-4fbf0eaeea86,MIG-4c2030cb-c106-53ed-bd51-4b5b779e6c1e,MIG-e202cd9d-7d48-55ab-8b14-3d917453976c,MIG-7eb7da44-8f92-5c22-b096-1e06cebf6223,MIG-ef854c66-26c6-539c-94ea-1d4557ef5351,MIG-cf654db3-9e25-5ddb-b244-6a2fe369d747,MIG-53a06fed-f282-579c-ba8d-66bbe12b4104,MIG-fc2b0859-e65e-5543-b539-9d44d6ef78f9,MIG-59c7d5b0-3d07-5a23-9e1e-1303a6639353")
    result = tune.run(
        train_function,
        name=ARGS.o,
        time_budget_s=3600 * 24,
        config=HYPERPARAMS,
        scheduler=hypopt_scheduler,
        resources_per_trial={"cpu": 2, "gpu": ARGS.gpu_per_trial},
        num_samples=ARGS.s,
        verbose=ARGS.v,
        log_to_file=True,
        local_dir=HYPOPTPATH,
        raise_on_failed_trial=False,
    )
    ray.shutdown()
    best_config = result.get_best_config(metric="P_MAE", mode="min", scope="last")
    best_config_df = pd.DataFrame.from_dict(best_config, orient="index")
    best_config_df.to_csv("{}/{}/best_config.csv".format(HYPOPTPATH, ARGS.o), sep="/")
    print(best_config)
    exp_df = result.results_df
    exp_df.to_csv("{}/{}/summary.csv".format(HYPOPTPATH, ARGS.o), sep="/")


if __name__ == "__main__":
    main()
