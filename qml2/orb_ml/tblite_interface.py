"""
Note that tblite uses atomic units for everything, including orbital energies and atomic coordinates.
"""
import numpy as np

from ..basic_utils import ExceptionRaisingClass
from ..data import conversion_coefficient
from .aux_classes import PseudoMF, PseudoMole

try:
    from tblite.interface import Calculator
except ModuleNotFoundError as ex:
    Calculator = ExceptionRaisingClass(ex)


tblite_max_iter = (
    2048  # increased compared to the default 250 due to some twisted molecules in RedDB.
)

default_calc_type = "GFN2-xTB"
default_verbosity = 0  # set 1 for verbose mode of tblite.

num_valence_electrons = {1: 1, 6: 4, 7: 5, 8: 6, 9: 7, 15: 5, 16: 6}


# Extracts atom AO ranges and angular momenta from a tblite calculator object.
def tblite_ao_ranges_angular_momenta(tblite_calc):
    ao_map = tblite_calc.get("orbital-map")
    shell_map = tblite_calc.get("shell-map")
    shell_angular_momentum = tblite_calc.get("angular-momenta")
    ao_ranges = []
    angular_momenta = np.zeros((len(ao_map),), dtype=int)
    prev_range_start = 0
    prev_atom = 0
    for ao_id, sh_id in enumerate(ao_map):
        angular_momenta[ao_id] = shell_angular_momentum[sh_id]
        if shell_map[sh_id] != prev_atom:
            prev_atom = shell_map[sh_id]
            ao_ranges.append((prev_range_start, ao_id))
            prev_range_start = ao_id
    ao_ranges.append((prev_range_start, len(ao_map)))
    return ao_ranges, angular_momenta


def generate_pyscf_mf_mol(
    compound,
    coordinate_units="Angstrom",
    verbosity=default_verbosity,
    calc_type=default_calc_type,
    spin=None,
    charge=None,
    solvent=None,
):
    if coordinate_units == "Bohr":
        used_coordinates = compound.coordinates
    else:
        used_coordinates = compound.coordinates * conversion_coefficient["Angstrom_Bohr"]
    calculator_kwargs = {}
    if spin is not None:
        calculator_kwargs = {"uhf": spin}
    if charge is not None:
        calculator_kwargs = {"charge": charge}

    calculator = Calculator(
        calc_type, compound.nuclear_charges, used_coordinates, **calculator_kwargs
    )
    if solvent is not None:
        calculator.add("alpb-solvation", solvent)

    ao_ranges, angular_momenta = tblite_ao_ranges_angular_momenta(calculator)

    calculator.set("save-integrals", 1)
    calculator.set("max-iter", tblite_max_iter)
    calculator.set("verbosity", verbosity)
    res = calculator.singlepoint()
    output = res.dict()

    mf = PseudoMF(
        e_tot=output["energy"],
        mo_coeff=output["orbital-coefficients"],
        mo_occ=output["orbital-occupations"],
        mo_energy=output["orbital-energies"],
    )

    mol = PseudoMole(
        ovlp_mat=output["overlap-matrix"],
        atom_ao_ranges=ao_ranges,
        angular_momenta=angular_momenta,
    )

    return mf, mol
