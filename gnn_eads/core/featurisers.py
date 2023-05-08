"""module containing deepchem based featuriser classes"""
import tempfile
from itertools import product
import copy

import numpy as np
import scipy.sparse as sp
import torch
from ase.atoms import Atoms
from networkx import (
    Graph,
    from_scipy_sparse_array,
    get_edge_attributes,
    get_node_attributes,
    is_connected,
    is_isomorphic,
    set_node_attributes,
    to_scipy_sparse_array,
)
from pymatgen.core.structure import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from scipy.spatial import Voronoi
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from torch import Tensor

from gnn_eads.core.constants import CORDERO, METALS, MOL_ELEM

# load metal and molecule elements
frag_elem = set(MOL_ELEM)
metals = set(METALS)
max_bond_degree = {"C": 4, "H": 1, "O": 2, "N": 3, "S": 2}


# load one-hot encoder
encoder = OneHotEncoder().fit(np.array([*frag_elem] + [*metals]).reshape(-1, 1))


def get_surf_atoms(
    atoms: Atoms,
    scale_factor: float = 1.0,
    tolerance: float = 0.25,
    second_order: bool = False,
) -> Atoms:
    """Create adsorbed fragment and metal atoms it is bonded to

    Args:
        atoms (Atoms): slab atoms with adsorbed fragment
        tolerance (float, optional): tolerance for voronoi neighbourlist in angstrom. Defaults to 0.25.
        scale_factor (float, optional): scale factor for voronoi neighbourlist. Defaults to 1.0.
        second_order (bool, optional): whether to include second order neighbours of metals. Defaults to False.
    Returns:
        Atoms: Atoms object of adsorbed fragment and metal atoms it is bonded to
    """
    # Get the indices of the atoms in the fragment
    indexes = {atom.index for atom in atoms if atom.symbol not in metals}
    metal_neighbours = set()

    # Get connectivity of the entire syste, slab + fragment
    nl = get_voronoi_neighbourlist(
        atoms, scale_factor=scale_factor, tolerance=tolerance
    )
    if len(nl) == 0:
        return Atoms(atoms[[*indexes]], pbc=atoms.pbc, cell=atoms.cell)

    # append those metal atoms which are bonded to any atom in indexes
    for pair in nl:
        # adsorbate-metal interaction
        if (pair[0] in indexes) and (atoms[pair[1]].symbol in metals):
            metal_neighbours.add(pair[1])
        # metal-adsorbate interaction
        elif (pair[1] in indexes) and (atoms[pair[0]].symbol in metals):
            metal_neighbours.add(pair[0])

    # add metal atoms which are bonded to other metal atoms
    if second_order:
        for metal_atom_index in metal_neighbours:
            # append to nl the index of neighbours of the metal atom
            for pair in nl:
                if (pair[0] == metal_atom_index) and (atoms[pair[1]].symbol in metals):
                    metal_neighbours.add(pair[1])
                elif (pair[1] == metal_atom_index) and (
                    atoms[pair[0]].symbol in metals
                ):
                    metal_neighbours.add(pair[0])
                else:
                    continue

    return Atoms(atoms[[*indexes, *metal_neighbours]], pbc=atoms.pbc, cell=atoms.cell)


def wrap_adsorbate(adsorbate):
    ads = adsorbate.copy()
    ads_g = Atoms(ads.get_chemical_symbols(), ads.get_positions(), cell=ads.cell)
    ads_graph = atoms_to_graph(ads_g, 1.5, 0.5)
    if not is_connected(ads_graph):
        ads = adsorbate.copy()
    else:
        return ads
    ads.wrap(center=(0.5, 0.5, 0.5), pbc=[True, True, True], pretty_translation=False)
    count = 1
    count_ = 1
    while True:
        ads_g = Atoms(ads.get_chemical_symbols(), ads.get_positions(), cell=ads.cell)
        ads_graph = atoms_to_graph(ads_g, 1.5, 0.5)
        if not is_connected(ads_graph) and count == 1:
            ads = adsorbate.copy()
            ads.wrap(
                center=(0.425, 0.378, 0.5),
                pbc=[True, True, False],
                pretty_translation=False,
            )
            count += 1
        elif not is_connected(ads_graph) and count == 2:
            ads = adsorbate.copy()
            ads.wrap(
                center=(0.425, 0.378, 0.5),
                pbc=[True, True, False],
                pretty_translation=True,
            )
            count += 1
        elif not is_connected(ads_graph) and count >= 7:
            print(
                "WARNING: No connected adsorbate could be found. Place replace the adsorbate manually to the centre of the slab."
            )
            break
        elif not is_connected(ads_graph) and count == 3:
            ads = adsorbate.copy()
            ads.wrap(
                center=(0.6, 0.6, 0.5),
                pbc=[True, True, False],
                pretty_translation=False,
            )
            count += 1
        elif not is_connected(ads_graph) and count == 4:
            ads = adsorbate.copy()
            ads.wrap(
                center=(0.5, 0.4, 0.5),
                pbc=[True, True, False],
                pretty_translation=False,
            )
            count += 1
        elif not is_connected(ads_graph) and count == 5:
            ads = adsorbate.copy()
            ads.wrap(
                center=(0.6, 0.5, 0.5),
                pbc=[True, True, False],
                pretty_translation=False,
            )
            count += 1
        elif not is_connected(ads_graph) and count == 6:
            ads = adsorbate.copy()
            ads.wrap(
                center=(0.525, 0.5, 0.5),
                pbc=[True, True, False],
                pretty_translation=False,
            )
            count += 1

            ads_g = Atoms(
                ads.get_chemical_symbols(), ads.get_positions(), cell=ads.cell
            )
            ads_graph = atoms_to_graph(ads_g, 1.5, 0.5)
            if not is_connected(ads_graph) and count_ == 1:
                adsorbate = ads
                count = 1
                count_ += 1
            else:
                count += 1
        else:
            break
    return ads


def atoms_to_mol(atoms: Atoms, relax: bool, bond_order: bool = False):  # -> RDKitMol:
    """Convert atoms object to rdkit mol object via temporary .pdb file.

    Args:
        atoms (Atoms): ase Atoms object of interest
        reax (bool): whether to relax the molecule with openbabel
        sclae_factor (float): scale factor for voronoi neighbourlist
        tolerance (float): tolerance for voronoi neighbourlist in angstrom

    Returns:
        RDKitMol: converted RDKit molecule object
    """
    # use ase wrap function to wrap adsorbate to the centre of the slab
    atoms = wrap_adsorbate(atoms)
    pmg_struct = Molecule(atoms.get_chemical_symbols(), atoms.get_positions())
    # convert to openbabel molecule object
    ob_mol = BabelMolAdaptor(pmg_struct)

    # this object can be optimized with the "mmff94" force field
    # this is necessary to correct distortions from adsorption; rdkit featurisation relies and relaxed gas phase geometry
    if relax:
        ob_mol.localopt(forcefield="mmff94", steps=3000)
    # convert back to pymatgen molecule object and then to rdkit molecule object
    pmg_mol = ob_mol.pymatgen_mol
    # try to convert to rdkit mol object using xyz file; rdDeterminEBonds.
    # write temp .xyz file and create rdkit mol object
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mol", delete=True) as temp:
        pmg_mol.to(filename=temp.name, fmt="mol")
        rdkit_mol = Chem.MolFromMolFile(temp.name, removeHs=False, sanitize=False)
    # rdDetermineBonds.DetermineConnectivity(rdkit_mol)
    # rdDetermineBonds.DetermineBondOrders(rdkit_mol)
    Chem.SanitizeMol(rdkit_mol, Chem.SANITIZE_FINDRADICALS ^ Chem.SANITIZE_SETHYBRIDIZATION)

    # Chem.SetAromaticity(rdkit_mol, Chem.AromaticityModel.AROMATICITY_SIMPLE)
    # sanitize mol object to recognise aromaticity and radicals
    # set aromaticity and conjugation; other ArromaticityModel options are available
    return rdkit_mol


def atoms_to_graph(
    atoms: Atoms,
    scale_factor: float = 1.0,
    tolerance: float = 0.25,
    second_order: bool = False,
) -> Graph:
    """Convert atoms object to graph object

    Args:
        atoms (Atoms): ase atoms object of interest
        scale_factor (float, optional): Scale factor for covalent radii of metal atoms.
        Defaults to 1.0.
        tolerance (float, optional): Tolerance for covalent radii of atoms.

    Returns:
        nx.Graph: networkx graph object
    """

    nl = get_voronoi_neighbourlist(
        atoms, scale_factor=scale_factor, tolerance=tolerance
    )

    if len(nl) == 0:
        graph = Graph()
        graph.add_nodes_from(range(len(atoms)))
        set_node_attributes(
            graph, dict(zip(range(len(atoms)), atoms.get_chemical_symbols())), "element"
        )
        return graph

    # convert neighbour list to scipy coo_matrix
    # add pair for both directions
    nl = np.concatenate((nl, nl[:, [1, 0]]))
    scipy_coo_matrix = sp.coo_matrix(
        (np.ones(len(nl)), (np.array(nl)[:, 0], np.array(nl)[:, 1]))
    )

    # built graph from scipy_coo_matrix
    graph = from_scipy_sparse_array(scipy_coo_matrix)
    if graph.number_of_nodes() != len(atoms):
        # create disconnected graph
        graph = Graph()
        graph.add_nodes_from(range(len(atoms)))
        graph.add_edges_from(nl)
    elem_list = [atom.symbol for atom in atoms]
    set_node_attributes(graph, dict(zip(range(len(elem_list)), elem_list)), "element")

    # remove metal-metal bonds if second_order is False
    if not second_order:
        m_m_edges = []
        # get metal-metal edges
        for edge in graph.edges:
            if (
                graph.nodes[edge[0]]["element"] in metals
                and graph.nodes[edge[1]]["element"] in metals
            ):
                m_m_edges.append(edge)
        # remove metal-metal edges
        graph.remove_edges_from(m_m_edges)

    return graph


def get_voronoi_neighbourlist(
    atoms: Atoms,
    scale_factor: float,
    tolerance: float,
) -> np.ndarray:
    """Get connectivity list from Voronoi analysis, considering periodic boundary conditions.
    To have two atoms connected, these must satisfy two conditions:
    1. They must share a Voronoi facet
    2. The distance between them must be less than the sum of their covalent radii (plus a tolerance)

    Args:
        atoms (Atoms): ase Atoms object.
        scaling_factor (float): Scaling factor for covalent radii of metal atoms.
        tolerance (float): Tolerance for second condition.

    Returns:
        np.ndarray: N_edges x 2 array with the connectivity list.

    Notes:
        The array contains all the edges just in one direction!
    """
    # First condition to have two atoms connected: They must share a Voronoi facet
    coords_arr = np.copy(atoms.get_scaled_positions())
    coords_arr = np.expand_dims(coords_arr, axis=0)
    coords_arr = np.repeat(coords_arr, 27, axis=0)
    mirrors = [-1, 0, 1]
    mirrors = np.asarray(list(product(mirrors, repeat=3)))
    mirrors = np.expand_dims(mirrors, 1)
    mirrors = np.repeat(mirrors, coords_arr.shape[1], axis=1)
    corrected_coords = np.reshape(
        coords_arr + mirrors,
        (coords_arr.shape[0] * coords_arr.shape[1], coords_arr.shape[2]),
    )
    corrected_coords = np.dot(corrected_coords, atoms.get_cell())
    translator = np.tile(np.arange(coords_arr.shape[1]), coords_arr.shape[0])
    vor_bonds = Voronoi(corrected_coords)
    pairs_corr = translator[vor_bonds.ridge_points]
    pairs_corr = np.unique(np.sort(pairs_corr, axis=1), axis=0)
    true_arr = pairs_corr[:, 0] == pairs_corr[:, 1]
    true_arr = np.argwhere(true_arr)
    pairs_corr = np.delete(pairs_corr, true_arr, axis=0)
    # Second condition for two atoms to be connected: Their distance must be smaller than the sum of their
    # covalent radii plus a tolerance.
    dst_d = {}
    pairs_lst = []
    for pair in pairs_corr:
        distance = atoms.get_distance(
            pair[0], pair[1], mic=True
        )  # mic=True for periodic boundary conditions
        elem_pair = (atoms[pair[0]].symbol, atoms[pair[1]].symbol)
        fr_elements = frozenset(elem_pair)
        if fr_elements not in dst_d:
            dst_d[fr_elements] = (
                CORDERO[atoms[pair[0]].symbol]
                + CORDERO[atoms[pair[1]].symbol]
                + tolerance
            )
            if atoms[pair[0]].symbol in METALS:
                dst_d[fr_elements] += (scale_factor - 1.0) * CORDERO[
                    atoms[pair[0]].symbol
                ]
            if atoms[pair[1]].symbol in METALS:
                dst_d[fr_elements] += (scale_factor - 1.0) * CORDERO[
                    atoms[pair[1]].symbol
                ]
        if distance <= dst_d[fr_elements]:
            pairs_lst.append(pair)
    if len(pairs_lst) == 0:
        return np.array([])
    else:
        return np.sort(np.array(pairs_lst), axis=1)


class Featurizer:
    """Featurise atoms of a molecule.
       Use OneHot for encoding of atom symbols.
       Ring, aromatic and radical features are optional.
       Takes ase atoms object as input

    Args:
        mol (Graph): nx.Graph representation of system of interest
        atoms_frag (Atoms): Atoms object of the fragment for ring, aromatic and radical features
    """

    def __init__(
        self,
        mol: Graph,
        atoms_frag: Atoms,
        scale_factor: float = 1.0,
        tolerance: float = 0.25,
        OneHot: bool = True,
        ring: bool = False,
        aromatic: bool = False,
        radical: bool = False,
        second_order: bool = False,
        relax: bool = True,
        num_el: bool = False,
    ):
        """_summary_

        Args:
            mol (Graph): nx.Graph representation of system of interest.
            atoms_frag (Atoms): Atoms object of the fragment for featurisation.
            scale_factor (float, optional): scale factor for covalent radii of metal atoms. Defaults to 1.0.
            tolerance (float, optional): tolerance for assigning edges. Defaults to 0.25.
            OneHot (bool, optional): OneHot encoding of atom symbols. Defaults to True.
            ring (bool, optional): OneHot encoding for determining if atom is in a ring. Defaults to False.
            aromatic (bool, optional): OneHot encoding for determining if atom is in an aromatic ring. Defaults to False.
            radical (bool, optional): OneHot encoding for determining if atom is a radical. Defaults to False.
            second_order (bool, optional): determines if second order metal atoms are considered, and if edge feature "metal-metal" is added. Defaults to False.
            relax (bool, optional): determines if relaxation on the fragment is performed. Defaults to True.
            num_el (bool, optional): Use of number of electrons (true) for radical feature instead of OneHot encoding (false). Defaults to False.
        """
        self.mol = mol
        self.atoms_frag = atoms_frag
        self.scale_factor = scale_factor
        self.tolerance = tolerance
        self.OneHot = OneHot  # OneHot encoding of atom symbols
        self.ring = ring
        self.aromatic = aromatic
        self.radical = radical
        self.relax = relax
        self.num_el = num_el  # radical feature with number of electrons (true) or one-hot encoding (false)
        self.second_order = second_order

    def featurize(self) -> Tensor:
        """Featurise atoms of a molecule with electronic state indices
            and OneHot encoding

        Args:
            self (Featurizer): Featurizer object

        Returns:
            torch.tensor: torch tensor of shape (n_atoms, n_features, dtype=torch.long)
        """
        # get node array of graph
        G1 = self.mol
        node_array = np.array([node for node in G1]).reshape(-1, 1)

        # OneHot encoding of atom symbols
        if self.OneHot:
            attribute_vector = np.vectorize(lambda x: G1.nodes[x]["element"])
            matrix = attribute_vector(node_array)
            features = encoder.transform(matrix).toarray()
        else:
            features = np.array(
                [get_node_attributes(G1, "element")[node] for node in G1]
            ).reshape(-1, 1)

        # add ring, aromatic and radical features
        if any([self.ring, self.aromatic, self.radical]):
            # get rdkit mol object
            rdkit_mol_connect = atoms_to_mol(self.atoms_frag, relax=self.relax)
            atom_array_connect = np.array([i for i in rdkit_mol_connect.GetAtoms()]).reshape(-1, 1)
            # Instantiate OneHotEncoder for True/False features
            true_false_encoder = LabelBinarizer()
            true_false_encoder.fit(np.array([[False], [True]]).reshape(-1, 1))
            zero_array = np.zeros(
                (features.shape[0] - len(self.atoms_frag), 1), dtype=int
            )

            # add ring feature
            if self.ring:
                ring_vector = np.vectorize(lambda x: x.IsInRing())
                ring_array = ring_vector(atom_array_connect)
                # encode True/False features
                ring_array = true_false_encoder.transform(ring_array)
                # expand array by entries for metals which are not in rdkit mol object
                ring_array = np.concatenate((ring_array, zero_array), axis=0)
                features = np.concatenate((features, ring_array), axis=1)

            # add aromatic feature
            if self.aromatic:
                aromatic_vector = np.vectorize(lambda x: x.GetIsAromatic())
                aromaticity_array = aromatic_vector(atom_array_connect)
                aromaticity_array = true_false_encoder.transform(aromaticity_array)
                aromaticity_array = np.concatenate(
                    (aromaticity_array, zero_array), axis=0
                )
                features = np.concatenate((features, aromaticity_array), axis=1)

            # add radical feature
            def get_radical_array(atom_array):
                rad_vector = np.vectorize(lambda x: x.GetNumRadicalElectrons())
                rad_array = rad_vector(atom_array)
                rad_array = rad_array.reshape(-1, 1)
                return rad_array
            def get_degree_array(atom_array):
                degree_vector = np.vectorize(lambda x: x.GetDegree())
                max_degree_vector = np.vectorize(lambda x: Chem.GetPeriodicTable().GetDefaultValence(x.GetAtomicNum()))
                degree_array = max_degree_vector(atom_array) - degree_vector(atom_array)
                degree_array = degree_array.reshape(-1, 1)
                return degree_array
       
            if self.radical:
                radical_connect_array = get_degree_array(atom_array_connect)
                zero_array = np.zeros(
                    (features.shape[0] - len(self.atoms_frag), 1), dtype=int
                )
                self.radical_connect_array = np.concatenate((radical_connect_array, zero_array), axis=0)
                radical_connect_array = copy.deepcopy(self.radical_connect_array)
                # set radical array to 0
                if self.num_el is False:
                    radical_connect_array[radical_connect_array > 0] = 1
                features = np.concatenate((features, radical_connect_array), axis=1)
                
                self.features = torch.tensor(features, dtype=torch.float)
                    

        return self.features

    def get_edge_index(self) -> Tensor:
        """Get edge_index of atoms in atoms object
        Returns:
            torch.tensor: edge_index of atoms in atoms object
        """

        G1 = self.mol
        # get adjacency matrix in coo format
        matrix = to_scipy_sparse_array(G1, format="coo")
        edge_index = torch.tensor(np.array([matrix.row, matrix.col]), dtype=torch.long)
        return edge_index

    def get_edge_features(self) -> Tensor:
        """Get edge features of atoms in atoms object
        Returns:
            torch.tensor: edge features of atoms in atoms object
        """

        G1 = self.mol
        # if node is frag_elem and neighbour is metal, add "frag-surf" to edge features
        # if node is frag_elem and neighbour is frag_elem, add "frag-frag" to edge features
        # if node is metal and neighbour is metal, add "metal-metal" to edge features
        for node in G1.nodes:
            for neighbour in G1.neighbors(node):
                if (
                    G1.nodes[node]["element"] in ["C", "N", "O", "S", "H"]
                    and G1.nodes[neighbour]["element"] in metals
                ):
                    G1[node][neighbour]["edge_feat"] = "frag-surf"
                # if (
                #     G1.nodes[node]["element"] == "H"
                #     and G1.nodes[neighbour]["element"] in metals
                # ):
                #     G1[node][neighbour]["edge_feat"] = "hydr-surf"
                if (
                    G1.nodes[node]["element"] in frag_elem
                    and G1.nodes[neighbour]["element"] in frag_elem
                ):
                    G1[node][neighbour]["edge_feat"] = "frag-frag"
                # metal-metal edges are only considered in second order
                if (
                    self.second_order
                    and G1.nodes[node]["element"] in metals
                    and G1.nodes[neighbour]["element"] in metals
                ):
                    G1[node][neighbour]["edge_feat"] = "metal-metal"
        edge_attr = get_edge_attributes(G1, "edge_feat")
        # find edge attribute for each pair in edge_index in coo format
        edge_index = self.get_edge_index()
        # use key error for edge_attr dict to create edge_attr_1
        # create 1d empty array
        edge_attr_1 = np.array([])
        for i in range(edge_index.shape[1]):
            try:
                feature = np.array(
                    [edge_attr[(edge_index[0][i].item(), edge_index[1][i].item())]]
                )
            except KeyError:
                try:
                    feature = np.array(
                        [edge_attr[(edge_index[1][i].item(), edge_index[0][i].item())]]
                    )
                except KeyError:
                    feature = np.array(["unsaturated"])
            # extend edge_attr_1 with feature
            edge_attr_1 = np.append(edge_attr_1, feature)
        edge_attr_1 = edge_attr_1.reshape(-1, 1)

        # OneHot encoding of edge features
        edge_enc = OneHotEncoder()
        edge_enc_arr = np.array(["frag-surf", "frag-frag", "unsaturated"]).reshape(-1, 1)
        if self.second_order:
            edge_enc_arr = np.array(["frag-surf", "frag-frag", "unsaturated", "metal-metal"]).reshape(
                -1, 1
            )
        edge_enc.fit(edge_enc_arr)
        edge_features = edge_enc.transform(edge_attr_1).toarray()
        edge_features = torch.tensor(edge_features, dtype=torch.float)
        return edge_features


def isomorphism_test(
    formula: str, graph_ads: Graph, energy: float, graphs: dict, atoms_frag: Atoms
):
    """Check if graph is isomorphic to any graph in graphs[formula] and replace it with the lower energy graph

    Args:
        formula (str): Formula of ensemble
        graph_ads (Graph): Graph representation of ensemble
        energy (float): Energy of ensemble
        graphs (dict): Dictionary of graphs {formula: [(graph, energy, formula, atoms_frag), ...]}
        atoms_frag (Atoms): adsorbate atoms object

    Returns:
        _type_: _description_
    """
    for i in range(len(graphs[formula])):
        g_2, energy2, _, _ = graphs[formula][i]

        # replace the graph with the lower energy
        if is_isomorphic(graph_ads, g_2) and energy < energy2:
            graphs[formula][i] = (graph_ads, energy, formula, atoms_frag)

        elif is_isomorphic(graph_ads, g_2) and energy > energy2:
            continue

        else:
            graphs[formula].append((graph_ads, energy, formula, atoms_frag))

    return graphs


def check_fragment(graph_frag: Graph, symbols: list) -> list:
    """
    Summary line
    Check if:
        1. Any frag_elem is present
        2. Fragment is connected
        3. Carbon has no more than 4 connections

    Args:
        atoms_frag (Atoms): adsorbate atoms object
        symbols (list): list of symbols in atoms_frag

    returns:
        list: for each condition: [empty, connected, c_conn, h_conn]
    """

    # check if any frag_elem is present
    if graph_frag.number_of_nodes() == 0:
        print(f"Empty graph{graph_frag}: {graph_frag.nodes}")
        empty = True
    else:
        empty = False
    # if empty, return False and break the loop
    if empty:
        connected = False
        c_conn = False
        h_conn = False
        return [empty, connected, c_conn, h_conn]

    # check if the adsorbate is connected
    connected = is_connected(graph_frag)
    # if not connected, return False and break the loop
    if not connected:
        empty = False
        h_conn = False
        c_conn = False
        return [empty, connected, c_conn, h_conn]

    # check if H and C are present
    if "C" not in symbols:
        c_conn = True
    if "H" not in symbols:
        h_conn = True
    if "C" in symbols or "H" in symbols:
        # check if carbon atoms have degree > 4
        # check if hydrogen atoms have degree > 1
        for node in graph_frag.nodes:
            if graph_frag.nodes[node]["element"] == "C" and graph_frag.degree(node) > 4:
                c_conn = False
                continue
            elif (
                graph_frag.nodes[node]["element"] == "C"
                and graph_frag.degree(node) <= 4
            ):
                c_conn = True
                continue
            elif graph_frag.nodes[node]["element"] == "H" and (
                graph_frag.degree(node) > 1 or graph_frag.degree(node) == 0
            ):
                h_conn = False
                continue
            elif (
                graph_frag.nodes[node]["element"] == "H"
                and graph_frag.degree(node) == 1
            ):
                h_conn = True
                continue

    return [empty, connected, c_conn, h_conn]


def check_ensemble(graph_ads: Graph, symbols: list) -> list:
    """Check if ensemble is valid.
        1. Check if graph is empty
        2. Check if graph is connected
        3. Check if graph contains metal if system contains metal

    Args:
        graph_ads (Graph): Graqph representation of ensemble: metal + fragment
        symbol (list): list of all symbols present in the initial input

    Returns:
        list: for each condition, [empty, connected, metal]
    """

    # 1. check if any atoms is present
    # 2. check if the ensemble graph is connected (adsorbate-surface interaction)
    if graph_ads.number_of_nodes() == 0 or not is_connected(graph_ads):
        empty = True
        connected = False
        metal = False
        return [empty, connected, metal]
    else:
        empty = False
        connected = True

    # 3. check if metal is present in initial input
    if not any(i in symbols for i in metals):
        metal = True
        return [empty, connected, metal]

    # if metal is present, check if metal is present in ensemble
    elif any(graph_ads.nodes[node]["element"] in metals for node in graph_ads.nodes):
        metal = True
    else:
        metal = False

    return [empty, connected, metal]
