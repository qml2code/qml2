from ase.atoms import Atoms

from .compound import Compound


def ASE2Compound(ase_in: Atoms, **other_kwargs):
    """
    Convert an ASE object into Compound.
    """
    return Compound(
        coordinates=ase_in.get_positions(), atomtypes=ase_in.get_chemical_symbols(), **other_kwargs
    )
