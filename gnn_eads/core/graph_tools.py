"""Functions for graphs handling and visualization purposes."""

import matplotlib.pyplot as plt
import networkx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

from gnn_eads.core.constants import (ELEMENT_LIST, ENCODER, FULL_ELEM_LIST,
                                     METALS, MOL_ELEM, rgb_colors)
from gnn_eads.core.functions import get_graph_formula


def convert_gpytorch_to_networkx(graph: Data) -> networkx.Graph:
    """
    Convert graph in pytorch_geometric format to NetworkX type.
    For each node in the graph, the label corresponding to the atomic species
    is added as attribute together with a corresponding color.
    Args:
        graph(torch_geometric.data.Data): torch_geometric graph object.
    Returns:
        new_g(networkx.classes.graph.Graph): networkx graph object.
    """
    node_features_matrix = graph.x.numpy()
    n_nodes = graph.num_nodes
    atom_list = []
    for i in range(n_nodes):
        index = np.where(node_features_matrix[i, :] == 1)[0][0]
        atom_list.append(FULL_ELEM_LIST[index])
    g = torch_geometric.utils.to_networkx(graph, to_undirected=False)
    connections = list(g.edges)
    nx_graph = networkx.DiGraph()
    for i in range(n_nodes):
        nx_graph.add_node(i, element=atom_list[i], rgb=rgb_colors[atom_list[i]])
    nx_graph.add_edges_from(connections, minlen=2)
    return nx_graph


def convert_networkx_to_gpytorch(graph: networkx.Graph) -> Data:
    """
    Convert graph object from networkx to pytorch_geometric type.
    Args:
        graph(networkx.classes.graph.Graph): networkx graph object
    Returns:
        new_g(torch_geometric.data.Data): torch_geometric graph object
    """
    return None


def plotter(
    graph: Data,
    node_size: int = 400,
    font_color: str = "white",
    font_weight: str = "bold",
    alpha: float = 0.9,
    arrowsize: int = 10,
    width: float = 1.2,
    k: float = 0.01,
    scale: float = 1,
    dpi: int = 600,
    figsize: tuple[int, int] = (4, 4),
    name: str = None,
    adsorbate: bool = False, # if True, only adsorbate is plotted
):
    """
    Visualize graph with atom labels and colors.
    Kamada_kawai_layout engine gives the best visualization appearance.
    Args:
        graph(torch_geometric.data.Data): graph object in pyG format.
    """
    if type(graph) != networkx.DiGraph and type(graph) != networkx.Graph:
        graph = convert_gpytorch_to_networkx(graph)
    else:
        # rename element nodes to atom nodes
        for node in graph.nodes:
            atom = graph.nodes[node]["element"]
            networkx.set_node_attributes(graph, {node: {"atom": atom, "rgb": rgb_colors[atom]}})
    if adsorbate:
        # remove metal nodes
        rm_nodes = []
        for node in list(graph.nodes):
            if graph.nodes[node]["atom"] in METALS:
                # get neighbors of metal nodes
                neighbors = list(graph.neighbors(node))
                for neighbor in neighbors:
                    graph.remove_edge(node, neighbor)
                    graph.remove_edge(neighbor, node)
                # remove edges connected to metal nodes
                rm_nodes.append(node)
        graph.remove_nodes_from(rm_nodes)
            
    labels = networkx.get_node_attributes(graph, "atom")
    colors = list(networkx.get_node_attributes(graph, "rgb").values())
    plt.figure(figsize=figsize, dpi=dpi)
    networkx.draw_networkx(
        graph,
        labels=labels,
        node_size=node_size,
        font_color=font_color,
        font_weight=font_weight,
        node_color=colors,
        alpha=alpha,
        arrows=None,
        arrowsize=arrowsize,
        arrowstyle="->",
        width=width,
        pos=networkx.kamada_kawai_layout(graph),
    )
    plt.draw()
    # save as svg
    if name != None:
        plt.savefig(name, format="svg")


def visualize_graph(
    graph: Data,
    node_size: int = 400,
    font_color: str = "white",
    font_weight: str = "bold",
    alpha: float = 0.9,
    arrowsize: int = 10,
    width: float = 1.2,
    k: float = 0.01,
    scale: float = 1,
    dpi: int = 600,
    figsize: tuple[int, int] = (4, 4),
):
    """
    Visualize graph with atom labels and colors.
    Kamada_kawai_layout engine is applied as it gives the best visualization appearance.
    Args:
        graph(torch_geometric.data.Data): graph object in pyG format.
    """
    graph = convert_gpytorch_to_networkx(graph)
    labels = networkx.get_node_attributes(graph, "atom")
    colors = list(networkx.get_node_attributes(graph, "rgb").values())
    networkx.draw_networkx(
        graph,
        labels=labels,
        node_size=node_size,
        font_color=font_color,
        font_weight=font_weight,
        node_color=colors,
        alpha=alpha,
        arrowsize=arrowsize,
        width=width,
        pos=networkx.kamada_kawai_layout(graph),
    )
    plt.draw()


def extract_adsorbate(graph: Data) -> Data:
    """Extract molecule from the adsorption graph,
    removing metals and connections between metal and molecule.

    Args:
        graph (torch_geometric.data.Data): Adsorption system in graph format
    Returns:
        adsorbate(torch_geometric.data.Data): Molecule in graph format
    """
    CHONS = [
        ELEMENT_LIST.index(element) for element in MOL_ELEM
    ]  # Get indeces of C,H,O,N,S in the encoder
    y = [None] * graph.num_nodes  # function for new indexing
    node_list = []
    node_index = []
    edge_list = []
    edge_index = []
    # 1) Node selection
    counter = 0
    for atom in range(graph.num_nodes):
        index = torch.where(graph.x[atom, :] == 1)[0].item()
        if index in CHONS:
            y[atom] = counter
            node_index.append(atom)
            node_list.append(graph.x[atom, :])
            counter += 1

    def ff(num):  # new indexing for the new graph (important!)
        return y[num]

    # 2) Edge selection
    for link in range(graph.num_edges):
        nodes = graph.edge_index[:, link]
        switch = 0
        for node in nodes:
            if node not in node_index:
                switch = 1
        if switch == 0:
            edge_list.append(nodes)
            edge_index.append(link)
        switch = 0
    # 3) Graph construction
    x = torch.zeros((len(node_list), len(FULL_ELEM_LIST)))
    edge = torch.zeros((2, len(edge_index)))
    for i in range(x.shape[0]):
        x[i, :] = node_list[i]
    for j in range(2):
        for k in range(edge.shape[1]):
            edge[j, k] = ff(edge_list[k][j])
    adsorbate = torch_geometric.data.Data(x, edge)
    return adsorbate


def get_number_atoms(graph: Data, atom: str) -> int:
    """Return number of atoms of a specific element in the graph.

    Args:
        graph (torch_geometric.data.Data): graph sample
        atom (str): atomic element present in the encoder
    Returns:
        n(int): number of atoms of the specified element in the graph
    """
    formula = get_graph_formula(graph)
    if atom in formula:
        index = formula.find(atom)
        return int(formula[index + 1])
    else:
        return "The defined element is not present in the graph."
