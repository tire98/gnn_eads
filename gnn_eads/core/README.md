# core
This package contains the functions and classes used to design the Graph Neural Network (GNN) architectures and graph data structures.
- `constants.py`: Contains the variables used in the package.
- `featurisers.py`: Contains the functions used to featurise the graphs, filter the graphs and obtain the adsorbate-metal ensemble.
- `functions.py`: Contains the functions used to train the GNNs.
- `graph_tools.py`: Functions for converting `networkx` graphs to `torch_geometric` graphs. Tools for visualising graphs.
- `nets.py`: Contains the classes used to design the GNN architectures.
- `post_training.py`: Contains the functions used to analyse the GNNs after training.
- `process_ase_db_to_PyG.py`: Contains the functions used to create the `torch_geometric` graphs and the `torch_geometric` `Dataset` from the ASE database.