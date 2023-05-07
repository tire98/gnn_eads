"""This module contains functions to extract data from vasp output files and\
    write them to an ase database."""
from ase.atoms import Atoms
from ase.formula import Formula
from ase.io import vasp
from pymatgen.core.composition import Composition

from gnn_eads.core.constants import METALS, MOL_ELEM, REF_ENERGIES

# Global variables
ref_energies_dict = REF_ENERGIES
frag_elem = MOL_ELEM
metals = METALS


def get_vasp_xml_output(vasp_xml: str) -> Atoms:
    """Return an Atoms object from a vasp xml file

    Args:
        vasp_xml (file): vasprun.xml file

    Returns:
        Atoms: Atoms object
    """
    try:
        xml = vasp.read_vasp_xml(vasp_xml)
        for i in xml:
            return i
    except AttributeError:
        raise FileNotFoundError


def get_metal(output: Atoms) -> str:
    """Get the metal the slab is built of from the output

    Args:
        output (Atoms): Atoms object read in from vasp output file
    """
    if Composition(str(output.symbols)).contains_element_type("metal"):
        for i in range(len(output.symbols)):
            if Composition(str(output[i].symbol)).contains_element_type("metal"):
                metal = output[i].symbol
    else:
        metal = ""
    return metal


def get_no_of_atom_species(readincontcar: Atoms) -> dict:
    """Return number of atom species in a structure

    Args:
        readincontcar (Atoms): Atoms object read in from CONTCAR, OUTCAR, or vasprun.xml

    Returns:
        dict: returns a dict with atom symbol as key and its occurence as value
    """
    formula = str(readincontcar.symbols)  # retrieve chemical formula from Atoms-object
    formula = Formula(formula).format(
        "metal"
    )  # alphabetically ordered with metal first
    elements = ["C", "H", "O", "N", "S"]  # elements of interest
    formula = Formula(formula).format("metal")
    metal = get_metal(readincontcar)
    formula = Formula(formula).count()  # return dictionary; e.g. {'C': 1, 'O': 2}
    no_metal = readincontcar.get_global_number_of_atoms()
    # expand dict by elements with zero value which are not present in formula
    for i in range(len(elements)):
        if elements[i] not in formula:
            formula[elements[i]] = 0
        elif elements[i] in formula:
            no_metal -= formula[elements[i]]
            formula[metal] = no_metal
        else:
            formula[metal] = no_metal
    return formula


def get_eads(metal: str, energy: float, m_metal: int = 48) -> float:
    """Calculate the proxy adsorption energy of a fragment
    E_ads = E_sys + E_slab

    Args:
        metal (str): metal symbol of slab
        m_metal (int): slab size (either 48 or 36)
        energy (float): DFT energy of vasp calculation (E_frag + E_slab)

    Returns:
        float: "adsorption energy" in eV
    """
    return energy - ref_energies_dict[str(metal) + str(m_metal)]

def get_fragment_energy(structure: list[int]) -> float:
    """Calculate fragment energy from closed shell structures.
    Args:
        structure (list[int]): list of atom numbers in the order C, H, O, N, S
    Returns:
        e_fragment (float): fragment energy in eV
    """ 
    e_H2O = ref_energies_dict["H2O"]
    e_H2 = ref_energies_dict["H2"]
    e_CH4 = ref_energies_dict["CH4"]
    e_NH3 = ref_energies_dict["NH3"]
    e_H2S = ref_energies_dict["H2S"]
    e_CO2 = ref_energies_dict["CO2"]
    # Count elemens in the structure
    n_C = int(structure[0])
    n_H = int(structure[1])
    n_O = int(structure[2])
    n_N = int(structure[3])
    n_S = int(structure[4])
    # Calculate adsorbate energy
    # e_fragment = (n_C * e_CH4) + (n_O * e_H2O) + (n_N * e_NH3) + (n_S * e_H2S)  + ((0.5 * n_H) - (2 * n_C) - n_O - (3/2 * n_N) - n_S) * e_H2
    e_fragment = n_C * e_CO2 + (n_O - 2*n_C) * e_H2O + (4*n_C + n_H - 2*n_O - (3/2 * n_N) - n_S) * e_H2 * 0.5 + (n_N * e_NH3) + (n_S * e_H2S) 

    return e_fragment