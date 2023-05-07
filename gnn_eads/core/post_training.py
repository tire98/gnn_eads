"""
Module for post-processing and collecting data after GNN model training.
"""

import csv
import json
import os
from datetime import date, datetime

import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import r2_score

from gnn_eads.core.constants import DPI, ENCODER, FG_RAW_GROUPS, METALS
from gnn_eads.core.functions import (get_graph_formula,
                                     get_number_atoms_from_label,
                                     split_percentage)
from gnn_eads.core.plot_functions import (DFTvsGNN_plot, hist_num_atoms,
                                          label_dist_train_val_test, pred_real)

# from torchinfo import summary


def create_model_report(
    model_name: str,
    model_path: str,
    configuration_dict: dict,
    model,
    loaders: tuple,
    scaling_params: tuple,
    mae_lists: tuple,
    device: dict = None,
    toml_file: str = None,
):
    """Create full report of the performed model training.

    Args:
        model_name (str): name of the model.
        model_path (str): path to the model folder.
        configuration_dict (dict): input hyperparams dict from toml input file.
        model (_type_): model object.
        loaders (tuple[DataLoader]): train/val/test sets(loaders).
        scaling_params (tuple[float]): Scaling params (mean and std of train+val sets).
        mae_lists (tuple[list]): MAE trends of train/val/test sets during learning process.
        device (dict, optional): Dictionary containing device info. Defaults to None.
        toml_file (str, optional): Name of the input toml file. Defaults to None.

    Returns:
        (str): Confirmation that model has been saved.
    """
    print("Saving the model ...")

    # Time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    # Unfold  train/val/test sets(loaders)
    train_loader = loaders[0]
    val_loader = loaders[1]
    test_loader = loaders[2]

    # Get data labels in train/val/test sets
    train_label_list = [
        get_graph_formula(graph, ENCODER.categories_[0])
        for graph in train_loader.dataset
    ]
    val_label_list = [
        get_graph_formula(graph, ENCODER.categories_[0]) for graph in val_loader.dataset
    ]
    train_family_set = [graph.family for graph in train_loader.dataset]
    val_family_set = [graph.family for graph in val_loader.dataset]
    if test_loader is not None:
        test_family_set = [graph.family for graph in test_loader.dataset]
    else:
        test_family_set = 0
    train_id_set = [graph.id.item() for graph in train_loader.dataset]
    val_id_set = [graph.id.item() for graph in val_loader.dataset]
    if test_loader is not None:
        test_id_set = [graph.id.item() for graph in test_loader.dataset]
    else:
        test_id_set = 0
    graph = configuration_dict["graph"]
    train = configuration_dict["train"]
    architecture = configuration_dict["architecture"]

    # Extract graph conversion parameters
    voronoi_tol = graph["voronoi_tol"]
    second_order_nn = graph["second_order"]
    scaling_factor = graph["scaling_factor"]

    # Scaling parameters
    if train["target_scaling"] == "std":
        mean_tv = scaling_params[0]
        std_tv = scaling_params[1]
    else:
        pass

    # Create directory where to store model files
    try:
        os.mkdir("{}/{}".format(model_path, model_name))
    except FileExistsError:
        model_name = input(
            "The name defined already exists in the provided directory: Provide a new one: "
        )
        os.mkdir("{}/{}".format(model_path, model_name))
    os.mkdir("{}/{}/Outliers".format(model_path, model_name))
    torch.save(train_loader, "{}/{}/train_loader.pth".format(model_path, model_name))
    torch.save(val_loader, "{}/{}/val_loader.pth".format(model_path, model_name))
    # Save model architecture and parameters
    torch.save(model, "{}/{}/model.pth".format(model_path, model_name))
    torch.save(model.state_dict(), "{}/{}/GNN.pth".format(model_path, model_name))

    # Store info of device on which model training has been performed
    if device is not None:
        with open("{}/{}/device.txt".format(model_path, model_name), "w") as f:
            print(device, file=f)

    # Store Hyperparameters dict from input file
    with open("{}/{}/input.txt".format(model_path, model_name), "w") as g:
        print(configuration_dict, file=g)

    # Copy input toml file to model folder
    os.system("cp {} {}/{}/input.toml".format(toml_file, model_path, model_name))

    loss = train["loss_function"]

    N_train = len(train_loader.dataset)
    N_val = len(val_loader.dataset)


    if train["test_set"] is False:
        N_tot = N_train + N_val
        file1 = open("{}/{}/performance.txt".format(model_path, model_name), "w")
        file1.write("GRAPH REPRESENTATION PARAMETERS\n")
        file1.write("Voronoi tolerance = {} Angstrom\n".format(voronoi_tol))
        file1.write("Atomic radii scaling factor = {}\n".format(scaling_factor))
        file1.write(
            "Second order metal neighbours inclusion = {}\n".format(second_order_nn)
        )
        file1.write("TRAINING PROCESS\n")
        file1.write(run_period)
        file1.write("Dataset Size = {}\n".format(N_tot))
        file1.write(
            "Data Split (Train/Val) = {}-{} %\n".format(
                *split_percentage(train["splits"], train["test_set"])
            )
        )
        file1.write("Target scaling = {}\n".format(train["target_scaling"]))
        file1.write("Dataset (train+val) mean = {:.6f} eV\n".format(scaling_params[0]))
        file1.write(
            "Dataset (train+val) standard deviation = {:.6f} eV\n".format(
                scaling_params[1]
            )
        )
        file1.write("Epochs = {}\n".format(train["epochs"]))
        file1.write("Batch Size = {}\n".format(train["batch_size"]))
        file1.write("Optimizer = Adam\n")  # Kept fixed in this project
        file1.write(
            "Learning Rate scheduler = Reduce Loss On Plateau\n"
        )  # Kept fixed in this project
        file1.write("Initial Learning Rate = {}\n".format(train["lr0"]))
        file1.write("Minimum Learning Rate = {}\n".format(train["minlr"]))
        file1.write("Patience (lr-scheduler) = {}\n".format(train["patience"]))
        file1.write("Factor (lr-scheduler) = {}\n".format(train["factor"]))
        file1.write("Loss function = {}\n".format(loss))
        file1.close()
        try:
            # add .csv file with errors for each epoch for train and val set
            w_pred, w_true = [], []  # Test set
            x_pred, x_true = [], []  # Train set
            a_pred, a_true = [], []  # Validation set
            for batch in train_loader:
                batch = batch.to("cpu")
                x_pred += model(batch)
                x_true += batch.y
            for batch in val_loader:
                batch = batch.to("cpu")
                a_pred += model(batch)
                a_true += batch.y

            z_pred = [x_pred[i].item() * std_tv + mean_tv for i in range(N_train)]  # Train set
            z_true = [x_true[i].item() * std_tv + mean_tv for i in range(N_train)]
            b_pred = [a_pred[i].item() * std_tv + mean_tv for i in range(N_val)]  # Val set
            b_true = [a_true[i].item() * std_tv + mean_tv for i in range(N_val)]
            error_train = [(z_pred[i] - z_true[i]) for i in range(N_train)]  # Error (train set)
            error_val = [(b_pred[i] - b_true[i]) for i in range(N_val)]  # Error validations et)

            abs_error_val = [abs(error_val[i]) for i in range(N_val)]
            abs_error_train = [abs(error_train[i]) for i in range(N_train)]
            # write .csv file with errors for each epoch for train and val set
            with open("{}/{}/train_set.csv".format(model_path, model_name), "w") as file4:
                writer = csv.writer(file4, delimiter="\t")
                writer.writerow(["System", "Family", "ID", "True [eV]", "Prediction [eV]", "Error [eV]", "Abs. error [eV]"])
                writer.writerows(zip(train_label_list, train_id_set, train_family_set, z_true, z_pred, error_train, abs_error_train)
            )
            with open("{}/{}/validation_set.csv".format(model_path, model_name), "w") as file4:
                writer = csv.writer(file4, delimiter="\t")
                writer.writerow(
                    ["System", "Family", "ID", "True [eV]", "Prediction [eV]", "Error [eV]", "Abs. error [eV]"])
                writer.writerows(zip(val_label_list, val_family_set, val_id_set, b_true, b_pred, error_val, abs_error_val))
            # write .json file with ids for each set
            id_dict = {"train": train_id_set, "val": val_id_set}
            for key in id_dict.keys():
                id_dict[key] = list(id_dict[key])
            with open("{}/{}/id_dict.json".format(model_path, model_name), "w") as file5:
                json.dump(id_dict, file5)
        except:
            print("Error in writing .csv file")
        return "Model saved in {}/{}".format(model_path, model_name)

    torch.save(test_loader, "{}/{}/test_loader.pth".format(model_path, model_name))

    test_label_list = [
        get_graph_formula(graph, ENCODER.categories_[0])
        for graph in test_loader.dataset
    ]
    N_test = len(test_loader.dataset)
    N_tot = N_train + N_val + N_test
    model.eval()
    model.to("cpu")

    w_pred, w_true = [], []  # Test set
    x_pred, x_true = [], []  # Train set
    a_pred, a_true = [], []  # Validation set

    for batch in test_loader:
        batch = batch.to("cpu")
        w_pred += model(batch)
        w_true += batch.y
    for batch in train_loader:
        batch = batch.to("cpu")
        x_pred += model(batch)
        x_true += batch.y
    for batch in val_loader:
        batch = batch.to("cpu")
        a_pred += model(batch)
        a_true += batch.y
    y_pred = [w_pred[i].item() * std_tv + mean_tv for i in range(N_test)]  # Test set
    y_true = [w_true[i].item() * std_tv + mean_tv for i in range(N_test)]
    z_pred = [x_pred[i].item() * std_tv + mean_tv for i in range(N_train)]  # Train set
    z_true = [x_true[i].item() * std_tv + mean_tv for i in range(N_train)]
    b_pred = [a_pred[i].item() * std_tv + mean_tv for i in range(N_val)]  # Val set
    b_true = [a_true[i].item() * std_tv + mean_tv for i in range(N_val)]
    
    # Error analysis
    error_test = [(y_pred[i] - y_true[i]) for i in range(N_test)]  # Error (test set)
    error_train = [(z_pred[i] - z_true[i]) for i in range(N_train)]  # Error (train set)
    error_val = [
        (b_pred[i] - b_true[i]) for i in range(N_val)
    ]  # Error (validation set)
    abs_error_test = [
        abs(error_test[i]) for i in range(N_test)
    ]  # Absolute Error (test set)
    abs_error_train = [
        abs(error_train[i]) for i in range(N_train)
    ]  # Absolute Error (train set)
    abs_error_val = [
        abs(error_val[i]) for i in range(N_val)
    ]  # Absolute Error (val set)
    squared_error_test = [error_test[i] ** 2 for i in range(N_test)]  # Squared Error
    abs_pctg_error_test = [
        abs(error_test[i] / y_true[i]) for i in range(N_test)
    ]  # Absolute Percentage Error
    std_error_test = np.std(error_test)  # eV
    # # Test set: Error distribution plot
    # sns.displot(error_test, bins=50, kde=True)
    plt.tight_layout()
    plt.savefig("{}/{}/test_error_dist.svg".format(model_path, model_name), dpi=DPI, bbox_inches='tight')
    plt.close()
    # Performance Report
    file1 = open("{}/{}/performance.txt".format(model_path, model_name), "w")
    file1.write(run_period)
    if device is not None:
        file1.write("Device = {}\n".format(device["name"]))
        file1.write("Training time = {:.2f} min\n".format(device["training_time"]))
    file1.write("---------------------------------------------------------\n")
    file1.write("GRAPH REPRESENTATION PARAMETERS\n")
    file1.write("Voronoi tolerance = {} Angstrom\n".format(voronoi_tol))
    file1.write("Atomic radius scaling factor = {}\n".format(scaling_factor))
    file1.write(
        "Second order metal neighbours inclusion = {}\n".format(second_order_nn)
    )
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN ARCHITECTURE\n")
    file1.write("Activation function = {}\n".format(architecture["sigma"]))
    file1.write("Convolutional layer = {}\n".format(architecture["conv_layer"]))
    file1.write("Pooling layer = {}\n".format(architecture["pool_layer"]))
    file1.write("Number of convolutional layers = {}\n".format(architecture["n_conv"]))
    file1.write(
        "Number of fully connected layers = {}\n".format(architecture["n_linear"])
    )
    file1.write("Depth of the layers = {}\n".format(architecture["dim"]))
    file1.write("Bias presence in inout layer = {}\n".format(architecture["bias_input"]))
    file1.write("---------------------------------------------------------\n")
    file1.write("TRAINING PROCESS\n")
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write(
        "Data Split (Train/Val/Test) = {}-{}-{} %\n".format(
            *split_percentage(train["splits"])
        )
    )
    file1.write("Target scaling = {}\n".format(train["target_scaling"]))
    file1.write("Target (train+val) mean = {:.6f} eV\n".format(mean_tv))
    file1.write("Target (train+val) standard deviation = {:.6f} eV\n".format(std_tv))
    file1.write("Epochs = {}\n".format(train["epochs"]))
    file1.write("Batch size = {}\n".format(train["batch_size"]))
    file1.write("Optimizer = Adam\n")  # Kept fixed in this project
    file1.write(
        "Learning Rate scheduler = Reduce Loss On Plateau\n"
    )  # Kept fixed in this project
    file1.write("Initial learning rate = {}\n".format(train["lr0"]))
    file1.write("Minimum learning rate = {}\n".format(train["minlr"]))
    file1.write("Patience (lr-scheduler) = {}\n".format(train["patience"]))
    file1.write("Factor (lr-scheduler) = {}\n".format(train["factor"]))
    file1.write("Loss function = {}\n".format(loss))
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN PERFORMANCE\n")
    file1.write("Test set size = {}\n".format(N_test))
    file1.write("Mean Bias Error (MBE) = {:.3f} eV\n".format(np.mean(error_test)))
    file1.write(
        "Mean Absolute Error (MAE) = {:.3f} eV\n".format(np.mean(abs_error_test))
    )
    file1.write(
        "Root Mean Square Error (RMSE) = {:.3f} eV\n".format(
            np.sqrt(np.mean(squared_error_test))
        )
    )
    file1.write(
        "Mean Absolute Percentage Error (MAPE) = {:.3f} %\n".format(
            np.mean(abs_pctg_error_test) * 100.0
        )
    )
    file1.write("Error Standard Deviation = {:.3f} eV\n".format(np.std(error_test)))
    file1.write("R2 = {:.3f} \n".format(r2_score(y_true, y_pred)))
    file1.write("---------------------------------------------------------\n")
    file1.write("OUTLIERS (TEST SET)\n")
    outliers_list, outliers_error_list, index_list = [], [], []
    counter = 0
    for sample in range(N_test):
        if abs_error_test[sample] >= 3 * std_error_test:
            counter += 1
            outliers_list.append(test_label_list[sample])
            outliers_error_list.append(error_test[sample])
            index_list.append(sample)
            if counter < 10:
                file1.write(
                    "0{}) {}    Error: {:.2f} eV    (index={})\n".format(
                        counter, test_label_list[sample], error_test[sample], sample
                    )
                )
            else:
                file1.write(
                    "{}) {}    Error: {:.2f} eV    (index={})\n".format(
                        counter, test_label_list[sample], error_test[sample], sample
                    )
                )
    file1.close()

    # Save train, val, test set error of the samples
    with open("{}/{}/train_set.csv".format(model_path, model_name), "w") as file4:
        writer = csv.writer(file4, delimiter="\t")
        writer.writerow(
            ["System", "Family", "ID", "True [eV]", "Prediction [eV]", "Error [eV]", "Abs. error [eV]"]
        )
        writer.writerows(
            zip(train_label_list, train_id_set, train_family_set, z_true, z_pred, error_train, abs_error_train)
        )
    with open("{}/{}/validation_set.csv".format(model_path, model_name), "w") as file4:
        writer = csv.writer(file4, delimiter="\t")
        writer.writerow(
            ["System", "Family", "ID", "True [eV]", "Prediction [eV]", "Error [eV]", "Abs. error [eV]"]
        )
        writer.writerows(zip(val_label_list, val_family_set, val_id_set, b_true, b_pred, error_val, abs_error_val))

    id_dict = {"train": train_id_set, "val": val_id_set}
    # save id_dict for storing the shuffling of the dataset
    with open("{}/{}/test_set.csv".format(model_path, model_name), "w") as file4:
        writer = csv.writer(file4, delimiter="\t")
        writer.writerow(
            ["System", "Family", "ID", "True [eV]", "Prediction [eV]", "Error [eV]", "Abs. error [eV]"]
        )
        writer.writerows(
            zip(test_label_list, test_family_set, test_id_set, y_true, y_pred, error_test, abs_error_test)
        )

    for key in id_dict.keys():
        id_dict[key] = list(id_dict[key])
    with open("{}/{}/id_dict.json".format(model_path, model_name), "w") as file5:
        json.dump(id_dict, file5)

    return "Model saved in {}/{}".format(model_path, model_name)
