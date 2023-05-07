#! /home/trenningholtz/miniconda3/envs/GNN/bin/python
"""Script to extract data from vasp output files and write them to a database.
"""

import argparse
import os

from ase.db import connect
from ase.calculators.vasp import Vasp
from ase.io.vasp import read_vasp_out
from gnn_eads.core.functions_data_extraction import (get_eads, get_metal,
                                                     get_no_of_atom_species,
                                                     get_vasp_xml_output)
from gnn_eads.core.constants import METALS as metals
from pymatgen.io.vasp.outputs import Oszicar, Vasprun, Outcar

# get the path to the output files and the database from the command line

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="path to the folder with vasp output files")
parser.add_argument("--db_path", type=str, help="path to the database")
parser.add_argument("--db_name", type=str, help="Name of the database file")
parser.add_argument("--family", type=str, help="Family name of the adsorbates. Custom value to key 'family' in database")

args = parser.parse_args()


path = args.data_path
family = args.family
with connect(os.path.join(args.db_path, args.db_name)) as db:
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("run.xml"):
                try:
                    print(os.path.join(root, file))
                    vasp_out = get_vasp_xml_output(os.path.join(root, file))
                except:
                    print("Could not read xml file")
                    print(os.path.join(root, "OUTCAR"))
                    vasp_out = read_vasp_out(os.path.join(root, "OUTCAR"))
            elif file == "OUTCAR" and not os.path.isfile(os.path.join(root, "vasprun.xml")):
                vasp_out = read_vasp_out(os.path.join(root, "OUTCAR"))
                print(os.path.join(root, "OUTCAR"))
            else:
                continue
            metal = get_metal(vasp_out)
            # if metal not in metals:
                # continue
            no_atoms = get_no_of_atom_species(vasp_out)
            m_metal = no_atoms[metal]
            energy = vasp_out.get_potential_energy()
            if metal == "":
                metal = "gas_phase"
                eads = 0
            else:
                eads = get_eads(metal, energy, m_metal)
            if eads == 0:
                eads = "N.A."
            path_ = os.path.relpath(root, path)
            try:
                converged = Vasprun(os.path.join(root, file)).converged_electronic
                relaxed = Vasprun(os.path.join(root, file)).converged_ionic
            except:
                converged = "N.A."
                relaxed = "N.A."
            if metal in ["Ni", "Co", "Fe"]  and os.path.isfile(os.path.join(root, "OSZICAR")):
                oszicar_dict = Oszicar(os.path.join(root, "OSZICAR")).as_dict()
                magmom = oszicar_dict["ionic_steps"][-1]["mag"]
            elif metal in ["Ni", "Co", "Fe"]  and not os.path.isfile(os.path.join(root, "OSZICAR")):
                magmom = Outcar(os.path.join(root, "OUTCAR")).total_mag
            else:
                magmom = ""
            db.write(
            vasp_out,
            m_metal=m_metal,
            c_atoms=no_atoms.get("C"),
            h_atoms=no_atoms.get("H"),
            o_atoms=no_atoms.get("O"),
            n_atoms=no_atoms.get("N"),
            s_atoms=no_atoms.get("S"),
            metal=metal,
            eads=eads,
            family=family,
            tot_magnet=magmom,
            # path=path_,
            converged=converged,
            relaxed=relaxed,
            )
