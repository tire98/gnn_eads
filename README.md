# Fast estimation of adsorption energies of open- and closed-shell molecules on metal surfaces using Graph Neural Networks
# About
This repository contains the code for the Master Thesis *Graph Neural Networks for the fast estimation of the adsorption energy of open- and closed-shell molecules on metal catalysts* which was used to train the models and generate the results. The work extends the current [GAME-Net model] (https://doi.org/10.1038/s43588-023-00437-y) and allows for the prediction of adsorption energies of open- and closed-shell fragments on 14 metal surfaces. 
 The repository is structured as follows
- `gnn_eads`: Python package containing the code and data used for the training the models, and the trained models.
- `scripts`: Python scripts used for running various tasks.
- `requirements.txt`: File containing the required packages to run the code.
This repository does not contain any data or trained models and is only used to provide the code used for the training and inference of the models. The following instructions will explain how the code was used to train the models and generate the results.
# Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. The instructions are for Linux systems and have been tested on Ubuntu 20.04.6 LTS. Windows users can use the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) (WSL) to run the code. The code has not been tested on macOS.

## Pre-requisites: 
- Python 3.9
- git 
- conda
- pip
1. Clone the repository from GitHub. Execute the following command in the terminal:
`git clone https://github.com/tire98/gnn_eads.git`
2. Create a conda environment with the required packages from the requirements.txt file. Execute the following command in the terminal:
`conda create --name gnn_eads --file requirements.txt`
3. Activate the conda environment. Execute the following command in the terminal:
`conda activate gnn_eads`
4. Install ray from source. Execute the following command in the terminal using pip.
`pip install ray`
5. Install the gnn_eads package. Execute the following commands in the terminal using conda.
`conda develop .`

# Usage
The code can be used for two purposes:
1. **Training the models**: Starting from your own (DFT) data or the data provided in `gnn_eads/data`, you can train models for your own purposes and investigate, e.g., the effect of different features or hyperparameters. Hyperparameter optimisations can be performed which is only advised if sufficient computational resources are available. 
2. **Inference**: Using the provided trained models in `gnn_eads/models`, you can perform inference on your own data. The data should be in the same format as the data provided in `gnn_eads/data`. Alternatively, the provided script `scripts/vasp_out_to_database.py` can be used to convert VASP `POSCAR`s, CONTCARs, or `OUTCAR`s to the required format.


## Training
1. New models can be trained on the provided dataset or on your own dataset. The training of  the models is performed using the `train_gnn.py` script.
    Run the following command in the terminal to see the available options:
    `python train_gnn.py --help`
    The following options are available:
    - `--dataset`: The name of the dataset to use. The dataset should be located in `gnn_eads/data`. 
    - `--database`: The name of the database to use. The database should be located in `gnn_eads/data/raw`. Note, even if the filtered database is present, the raw database should be used since the script will identify the filtered database and select it automatically.
    - `--toml_file`: The name of the toml file containing the hyperparameters to use. The toml file should be located in `gnn_eads/toml` 
    - `--model_name`: The name of the model which will be used to save the model. The model will be saved in a new directory with the model's name in `gnn_eads/models`.
2. Perform a nested cross validation with `nested_cross_validation.py`. 
    Run the following command in the terminal to see the available options:
    `python nested_cross_validation.py --help`
    The following options are available:
    - `--dataset`: The name of the dataset to use. The dataset should be located in `gnn_eads/data`. 
    - `--database`: The name of the database to use. The database should be located in `gnn_eads/data/raw`. Note, even if the filtered database is present, the raw database should be used since the script will identify the filtered database and select it automatically.
    - `--input`: Input `.toml` file containing the hyperparameters to use. The toml file should be located in `gnn_eads/toml`
    - `--output`: Output directory name where the results will be saved. The directory will be created in `gnn_eads/models` and should not exist.
3. Perform a hyperparameter optimisation with the hyperparameters of interest. Choose the hyperparameters of interest in the `hypopt_gnn.py` script.

## Inference
1. Assess the performance of a trained model on a test set with `test_performance.py`.
Run the following command in the terminal to see the available options:
`python test_performance.py --help`
The following options are available:
- `--model_name`: The name of the model to use. The model should be located in `gnn_eads/models`.
- `--dataset`: The name of the test dataset to use. The dataset should be located in `gnn_eads/data`.
- `--database`: The name of the test database to use. The database should be located in `gnn_eads/data/raw`. Note, even if the filtered database is present, the raw database should be used since the script will identify the filtered database and select it automatically.
- `--toml_file`: The name of the toml file containing the hyperparameters to use. The toml file should be located in `gnn_eads/toml`. Here, only the graph hyperparameters are used.
- `--mean`: If no `performance.txt` file is present in the model directory, the mean of the mean of the train+validation energy target needs to be provided. Ignore this option if the `performance.txt` file is present.
- `--std`: If no `performance.txt` file is present in the model directory, the mean of the standard deviation of the train+validation energy target needs to be provided. Ignore this option if the `performance.txt` file is present.

2. Perform a more detailed analysis with interactive plots as outlined any of the notebooks in `gnn_eads/notebooks`. The notebooks can be run in Jupyter Lab or Jupyter Notebook and should be run with the `gnn_eads` conda environment.


## Database generation
You can either train your own models on the provided data, or create your own dataset which needs to be saved in form of an `ase` database in `gnn_eads/data`. The database should contain the following keys:
- `metal`: The metal symbol.	
- `family`: The family the adsorbate belongs to, e.g., aromatic, cylic, alcohol, etc.
- `eads`: The target adsorption energy (in eV).
- `e_tot`: The total energy of the adsorbate-metal system (in eV).
- `c_atoms`: The number of carbon atoms in the adsorbate.
- `h_atoms`: The number of hydrogen atoms in the adsorbate.
- `o_atoms`: The number of oxygen atoms in the adsorbate.
- `n_atoms`: The number of nitrogen atoms in the adsorbate.
- `s_atoms`: The number of sulphur atoms in the adsorbate.
- `m_metal`: The number of metal atoms in the slab.
- `e_gas`: The total energy of the gas phase adsorbate (in eV).
The database can be generated using the provided script `scripts/vasp_out_to_database.py` which converts VASP ouput files to the required format. However, this script is only suitable for metal surfaces presented in the work, unless the column `e_ads`gets updated with the correct values after database generation. 
The columns concerning the 

## Features
### Node features
Currently, the graph representation can encode the following features for the adsorbate atoms in addition to the one-hot encoded atom type:
- degree of unsaturation. The maximum valence from `rdkit` is used to calculate the degree of unsaturation.
- aromaticity. The aromaticity is calculated using the `rdkit` implementation. This feature is not used in the current implementation and might not work properly.
- cyclicity. The non-aromatic rings are identified using the `rdkit` implementation. This feature is not used in the current implementation and might not work properly.
### Edge features
Edge features can be added as well, but are currently not used and might not work properly. The GraphSAGE implementation does not support edge features and the GATv2 needs to be used instead. However, initial trials suggest that the GATv2 implementation performs worse than GraphSAGE. The MAE are higher and the training time is longer. The edge features distinguish between the following types of edges:
- fragment-fragment bonds
- metal-fragment bonds

