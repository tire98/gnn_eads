"""Converts ase database to PyG dataset"""
import os
from multiprocessing import Pool

import torch
from ase.db import connect
from ase.atoms import Atoms
from networkx import is_isomorphic
from torch_geometric.data import Data, InMemoryDataset

import gnn_eads.core.featurisers as ft
from gnn_eads.core.constants import METALS, MOL_ELEM, REF_ENERGIES, FG_RAW_GROUPS

torch.multiprocessing.set_sharing_strategy("file_system")

slabs_energy_dict = REF_ENERGIES


metals = METALS
frag_elem = MOL_ELEM
families = FG_RAW_GROUPS


class FGGraphDataset_Feat(InMemoryDataset):
    """processes ase database to PyG InMemoryDataset
    Args:
        InMemoryDataset (_type_): _description_
    """

    def __init__(
        self,
        root: str,
        database: str,
        family: str,
        second_order: bool = False,
        scale_factor: float = 1.0,
        tolerance: float = 0.25,
        edge_features: bool = True,
        ring_features: bool = True,
        aromatic_features: bool = True,
        radical_features: bool = True,
        relax: bool = True,
        num_el: bool = True,
        write_db: bool = False,
    ):
        """Inherits from InMemoryDataset

        Args:
            root (str): root to the directory where the dataset should be saved
            database (str): database to be processed
            family (str): FG family of the dataset
            es_index (bool, optional): If True, EState index is added to node features. Defaults to True.
        """
        self.root = str(root)
        self.database = database
        self.family = family
        self.second_order = second_order
        self.scale_factor = scale_factor
        self.tolerance = tolerance
        self.edge_features = edge_features
        self.ring_features = ring_features
        self.aromatic_features = aromatic_features
        self.radical_features = radical_features
        self.relax = relax
        self.num_el = num_el
        
        self.post_data = (
            str(root)
            + "/processed/"
            + "FGGraphDataset_"
            + self.family
            + "_"
            + str(self.second_order)
            + "_"
            + str(int(self.scale_factor * 10))
            + "_"
            + str(int(self.tolerance * 100))
            + "_"
            + str(self.edge_features)
            + "_"
            + str(self.ring_features)
            + "_"
            + str(self.aromatic_features)
            + "_"
            + str(self.radical_features)
            + "_"
            + str(self.num_el)
            + "_"
            + str(self.relax)
            + ".pt"
        )
        self.write_db = write_db
        self.db_name = (
            str(root)
            + "/raw/"
            + self.database.removesuffix(".db")
            + "_"
            + str(int(self.scale_factor * 10))
            + "_"
            + str(int(self.tolerance * 100))
            + ".db"
        )
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> str:
        """If this file exists in pre_dir, download is not triggered"""
        return self.database

    @property
    def processed_file_names(self) -> list:
        """If this file exists in processed_dir, process is not triggered
        Returns:
            dataset: returns the processed dataset
        """
        return self.post_data

    def read_filter_data_from_db(self, metal: str, database: str) -> None:
        """Reads from ase database and filters data.
            Filters applied:
                1. Adsorbate is disconnected or shows wrong connectivity
                2. Adsorbate shows no connectivity to surface
                3. Calculation was not converged or relaxed
                4. Isomorphic graphs are discarded

        Args:
            metal (str): Metal of interest
            db (str): ase database that is iterated through
        """
        db = connect(database)
        graphs = {}
        # check metal is present for family
        final_log = []
        final_log.append(f"\n\nLog for {self.family} on {metal}:")
        # if db.count(family=self.family, metal=metal) == 0:
        #     final_log.append(f"No {self.family} on {metal} found in database")
        #     return final_log
        for row in db.select(family=self.family, metal=metal):
            log_file = []
            # check if DFT calculation converged and relaxed, if available!
            if row.get("converged") == "False" or row.get("relaxed") == "False":
                continue
            else:
                # check if any frag_elem is present
                atoms = row.toatoms()
                metal = row.get("metal")
                energy = row.get("eads")
                if energy == "N.A.":
                    energy = row.get("energy")
                symbols_sys = atoms.get_chemical_symbols()
                family = row.get("family")
                row_id = row.get("id")
                if not any(i in symbols_sys for i in frag_elem):
                    continue

                # check if fragment is connected
                atoms_frag = Atoms(atoms[
                    [atom.index for atom in atoms if atom.symbol != metal]
                ], pbc=atoms.pbc, cell=atoms.cell)

                graph_frag = ft.atoms_to_graph(
                    atoms_frag, self.scale_factor, self.tolerance, self.second_order
                )

                symbols_frag = atoms_frag.get_chemical_symbols()
                formula_frag = atoms_frag.get_chemical_formula(mode="metal")
                check_frag_list = ft.check_fragment(graph_frag, symbols_frag)
                # check if a fragment is present
                if check_frag_list[0]:
                    print(
                        f"No adsorbate present. Removing {formula_frag} on {metal} from {family} from dataset. "
                    )
                    # add to log file
                    log_file.append(
                        f"No adsorbate present. Removing {formula_frag} on {metal} from {family} from dataset. "
                    )
                    final_log.extend(log_file)
                    continue
                # check if fragment is connected and shows correct connectivity
                if any(i is False for i in check_frag_list[1:]):
                    print(
                        f"Adsorbate is disconnected or shows wrong connectivity. Removing {formula_frag} on {metal} from {family} from dataset. "
                    )
                    # add to log file
                    log_file.append(
                        f"Adsorbate is disconnected or shows wrong connectivity. Removing {formula_frag} on {metal} from {family} from dataset. "
                    )
                    final_log.extend(log_file)
                    continue

                atoms_ads = ft.get_surf_atoms(
                    atoms, self.scale_factor, self.tolerance, self.second_order
                )
                formula = atoms_ads.get_chemical_formula(mode="metal")
                graph_ads = ft.atoms_to_graph(
                    atoms_ads, self.scale_factor, self.tolerance, self.second_order
                )
                check_ensemble_list = ft.check_ensemble(graph_ads, symbols_sys)
                # check if graph is not empty
                if check_ensemble_list[0]:
                    print(
                        f"Adsorbate shows no connectivity to surface. Removing {formula} on {metal} from {family} from dataset. "
                    )
                    # add to log file
                    log_file.append(
                        f"Adsorbate shows no connectivity to surface. Removing {formula} on {metal} from {family} from dataset. "
                    )
                    final_log.extend(log_file)
                    continue

                # check if graph is connected and metal is present in the graph
                if any(i is False for i in check_ensemble_list[1:]):
                    print(
                        f"Adsorbate shows no connectivity to metal. Removing {formula} on {metal} from {family} from dataset. "
                    )
                    # add to log file
                    log_file.append(
                        f"Adsorbate shows no connectivity to metal. Removing {formula} on {metal} from {family} from dataset. "
                    )
                    final_log.extend(log_file)
                    continue

                # check if graph_ads is isomorphic to another fragment with same formula
                if formula in graphs.keys():
                    # check if any graph is isomorphic to the current graph,
                    # use node and edge attributes
                    if any(
                        is_isomorphic(
                            graph_ads,
                            graphs[formula][g][0],
                            node_match=lambda x, y: x["element"] == y["element"],
                        )
                        for g in range(len(graphs[formula]))
                    ):
                        for i in range(len(graphs[formula])):
                            g_2, energy2, _, _, _ = graphs[formula][i]
                            # check if graphs are isomorphic and if energy is lower
                            if (
                                is_isomorphic(
                                    graph_ads,
                                    g_2,
                                    node_match=lambda x, y: x["element"]
                                    == y["element"],
                                )
                                and energy < energy2
                            ):
                                print(
                                    f"Replacing {formula} from {family} by a graph with lower energy. "
                                )
                                # add to log file
                                log_file.append(
                                    f"Replacing {formula} from {family} by a graph with lower energy. "
                                )
                                # replace the graph with the lower energy
                                graphs[formula][i] = (
                                    graph_ads,
                                    energy,
                                    formula,
                                    atoms_frag,
                                    row_id,
                                )
                                final_log.extend(log_file)
                                break

                            elif (
                                is_isomorphic(
                                    graph_ads,
                                    g_2,
                                    node_match=lambda x, y: x["element"]
                                    == y["element"],
                                )
                                and energy > energy2
                            ):
                                print(
                                    f"Discard {formula} from {family} from dataset. Energy is higher than other isomorphic graph. "
                                )
                                # add to log file
                                log_file.append(
                                    f"Discard {formula} from {family} from dataset. Energy is higher than other isomorphic graph. "
                                )
                                final_log.extend(log_file)
                                continue
                    else:
                        graphs[formula].append(
                            (graph_ads, energy, formula, atoms_frag, row_id)
                        )

                elif formula not in graphs.keys():
                    graphs[formula] = [(graph_ads, energy, formula, atoms_frag, row_id)]
        if len(final_log) == 1:
            final_log.append(f"Nothing to report for {metal}. \n")
        # export log_file
        with open(
            f"{self.root}/processed/FGGraphDataset_Log_{int(self.scale_factor * 10)}_{int(self.tolerance * 100)}.txt",
            "a",
        ) as log:
            # write log file with new line for each item in list
            log.write("\n".join(final_log))
        return graphs

    def read_pre_processed_db(self, metal: str, database: str) -> None:
        """Read data from database which already pre-filtered adsorption systems
            Filters applied:
                1. Adsorbate is disconnected or shows wrong connectivity
                2. Adsorbate shows no connectivity to surface
                3. Calculation was not converged or relaxed
                4. Isomorphic graphs are discarded

        Args:
            metal (str): Metal of interest
            result_queue (Queue): Queue to store the results
        """
        data_list = []
        db = connect(database)
        for row in db.select(family=self.family, metal=metal):
            atoms = row.toatoms()
            metal = row.get("metal")
            energy = row.get("eads")
            if energy == "N.A.":
                energy = row.get("energy")
            row_id = row.get("id")
            atoms_frag = atoms[[atom.index for atom in atoms if atom.symbol != metal]]
            # remove S and N atoms for now
            # if any(atom.symbol in ["S", "N"] for atom in atoms_frag):
            #     continue
            atoms_ads = ft.get_surf_atoms(
                atoms, self.scale_factor, self.tolerance, self.second_order
            )
            formula = atoms_ads.get_chemical_formula(mode="metal")
            graph_ads = ft.atoms_to_graph(
                atoms_ads, self.scale_factor, self.tolerance, self.second_order
            )
            featuriser = ft.Featurizer(
                mol=graph_ads,
                atoms_frag=atoms_frag,
                scale_factor=self.scale_factor,
                tolerance=self.tolerance,
                second_order=self.second_order,
                ring=self.ring_features,
                aromatic=self.aromatic_features,
                radical=self.radical_features,
                relax=self.relax,
                num_el=self.num_el,
            )
            features = featuriser.featurize()
            edges = featuriser.get_edge_index()
            energy = torch.tensor([energy], dtype=torch.float)
            data = Data(
                x=features,
                edge_index=edges,
                y=energy,
                ener=energy,
                formula=formula,
                family=self.family,
                id=row_id,
            )

            # add edge features if requested
            if self.edge_features:
                edge_features = featuriser.get_edge_features()
                data.edge_attr = edge_features

            # Add filters here, use those present in the original code
            data_list.append(data)

        return data_list

    def process(self) -> None:
        """Process the database and saves it to processed_dir. Multiprocessing is used to speed up the process."""


        print("Selecting database... ")
        data_list = []
        graphs = {}
        # check if preprocessed db is present
        if os.path.isfile(self.db_name) and self.write_db is False:
            print("Preprocessed database already present. ")
            print(f"Reading data from {self.db_name}. ")
            database = self.db_name
            task = self.read_pre_processed_db
        else:
            print(f"Reading data from {self.raw_paths[0]}. ")
            database = self.raw_paths[0]
            task = self.read_filter_data_from_db
        # gas phase data do not have a metal
        if "gas_" in self.family:
            starmap_args = [("", database)]
        else:
            starmap_args = [(metal, database) for metal in metals]
        # instantiate pool
        with Pool() as pool:
            process_result = pool.starmap(task, starmap_args)
            if task == self.read_pre_processed_db:
                for metal in process_result:
                    data_list.extend(metal)
            else:
                for metal in process_result:
                    graphs.update(metal)

        # if data is read from raw db, a dict is returned and needs to
        # be converted to a list of Data objects
        if task == self.read_filter_data_from_db:
            row_ids = set()
            for formula in graphs.keys():
                for i in range(len(graphs[formula])):
                    G1, energy, formula, atoms_frag, row_id = graphs[formula][i]
                    row_ids.add(row_id)
                    featuriser = ft.Featurizer(
                        mol=G1,
                        atoms_frag=atoms_frag,
                        scale_factor=self.scale_factor,
                        tolerance=self.tolerance,
                        second_order=self.second_order,
                        ring=self.ring_features,
                        aromatic=self.aromatic_features,
                        radical=self.radical_features,
                        relax=self.relax,
                        num_el=self.num_el,
                    )
                    features = featuriser.featurize()
                    edges = featuriser.get_edge_index()
                    energy = torch.tensor([energy], dtype=torch.float)
                    data = Data(
                        x=features,
                        edge_index=edges,
                        y=energy,
                        ener=energy,
                        formula=formula,
                        family=self.family,
                        id=row_id,
                    )

                    # add edge features if requested
                    if self.edge_features:
                        edge_features = featuriser.get_edge_features()
                        data.edge_attr = edge_features

                    data_list.append(data)
            # if write_db is True, write the filtered data to a new database
            if self.write_db:
                db = connect(self.raw_paths[0])
                with connect(self.db_name) as db_new:
                    print("Writing to new database... ")
                    for row_id in row_ids:
                        row = [row for row in db.select(id=row_id)][0]
                        atoms = row.toatoms()
                        db_new.write(atoms, key_value_pairs=row.key_value_pairs)

        # Then, create the dataset
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def load_FG_dataset(
    root: str,
    database: str,
    second_order: bool = False,
    scale_factor: float = 1.0,
    tolerance: float = 0.25,
    edge_features: bool = False,
    ring_features: bool = False,
    aromatic_features: bool = False,
    radical_features: bool = False,
    family: list = None,
    relax: bool = True,
    num_el: bool = True,
    write_db: bool = False,
) -> FGGraphDataset_Feat:
    """Create the dataset and initialize the RadicalGraphDataset class

    Args:
        root (str): root to the directory where the dataset is stored in a 'raw' subdirectory
        database (str): name of the database
        second_order (bool, optional): whether to include second neighbours of metal atoms. Defaults to False.
        scale_factor (float, optional): scale factor for the cutoff radius. Defaults to 1.0.
        tolerance (float, optional): tolerance for the cutoff radius. Defaults to 0.25.
        edge_features (bool, optional): whether to include edge features. Defaults to False.
        OneHot encoding of the bond. fragment-fragment, fragment-metal; metal-metal only if second_order is True.

        These features are used OneHot encoded. 0: False, 1: True.
        ring_features (bool, optional): whether to include ring features. Defaults to False.
        aromatic_features (bool, optional): whether to include aromatic features. Defaults to False.
        radical_features (bool, optional): whether to include radical features. Defaults to False.

    Returns:
        FGGraphDataset_Feat: returns list with the datasets
    """
    dataset_list = []
    if family[0] != "all":
        for fam in family:
            db_path = os.path.join(root, "raw", database)
            if connect(os.path.join(root, "raw", database)).count(family=fam) == 0:
                print("No data for family: ", fam)
                continue
            print('fam: ', fam)
            data = FGGraphDataset_Feat(
                root,
                database,
                family=fam,
                second_order=second_order,
                scale_factor=scale_factor,
                tolerance=tolerance,
                edge_features=edge_features,
                ring_features=ring_features,
                aromatic_features=aromatic_features,
                radical_features=radical_features,
                relax=relax,
                num_el=num_el,
                write_db=write_db,
            )
            dataset_list.append(data)
    else:
        for fam in families:
            db_path = os.path.join(root, "raw", database)
            if connect(db_path).count(family=fam) == 0:
                print("No data for family: ", fam)
                continue
            print("Processing family: ", fam)
            data = FGGraphDataset_Feat(
                root,
                database,
                family=fam,
                second_order=second_order,
                scale_factor=scale_factor,
                tolerance=tolerance,
                edge_features=edge_features,
                ring_features=ring_features,
                aromatic_features=aromatic_features,
                radical_features=radical_features,
                relax=relax,
                num_el=num_el,
                write_db=write_db,
            )
            dataset_list.append(data)

    return dataset_list
