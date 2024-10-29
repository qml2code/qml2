from .basic_utils import nuclear_charge
from .jit_interfaces import array_, int_
from .representations import (
    generate_bob,
    generate_cmbdf,
    generate_coulomb_matrix,
    generate_fchl19,
    generate_slatm,
)
from .utils import read_xyz_file, read_xyz_lines, str_atom_corr


class Compound:
    """
    Compound class is used to store all the data associated with a molecule along with

    xyz             - xyz file use to create base Compound object.
    """

    def __init__(
        self,
        xyz=None,
        xyz_lines=None,
        coordinates=None,
        nuclear_charges=None,
        atomtypes=None,
    ):
        self.coordinates = None
        self.nuclear_charges = None
        self.atomtypes = None
        self.name = None
        if (xyz is not None) or (xyz_lines is not None):
            if xyz_lines is not None:
                xyz_input = read_xyz_lines(xyz_lines)
            else:
                xyz_input = read_xyz_file(xyz)
            (
                self.nuclear_charges,
                self.atomtypes,
                self.coordinates,
                self.add_attr_dict,
            ) = xyz_input
        if coordinates is not None:
            self.coordinates = coordinates
        if nuclear_charges is not None:
            self.nuclear_charges = nuclear_charges
        if atomtypes is not None:
            self.atomtypes = atomtypes
            self.nuclear_charges = array_(
                [nuclear_charge(atomtype) for atomtype in self.atomtypes], dtype=int_
            )
        if (self.atomtypes is None) and (self.nuclear_charges is not None):
            self.atomtypes = [str_atom_corr(charge) for charge in self.nuclear_charges]

        if isinstance(xyz, str):
            self.name = xyz
        else:
            self.name = repr(xyz)
        self.representation = None

    def generate_coulomb_matrix(self, size=29):
        self.representation = generate_coulomb_matrix(
            self.nuclear_charges, self.coordinates, size=size
        )

    def generate_fchl19(self, gradients=False, **kwargs):
        if gradients:
            (
                self.representation,
                self.grad_representation,
                self.grad_relevant_atom_ids,
                self.grad_relevant_atom_nums,
            ) = generate_fchl19(
                self.nuclear_charges, self.coordinates, gradients=gradients, **kwargs
            )
        else:
            self.representation = generate_fchl19(self.nuclear_charges, self.coordinates, **kwargs)

    def generate_bob(self, bags, **kwargs):
        self.representation = generate_bob(self.nuclear_charges, self.coordinates, bags, **kwargs)

    def generate_slatm(self, mbtypes, **kwargs):
        self.representation = generate_slatm(
            self.nuclear_charges, self.coordinates, mbtypes, **kwargs
        )

    def generate_cmbdf(self, convolutions, gradients=False, **kwargs):
        if gradients:
            (
                self.representation,
                self.grad_representation,
                self.grad_relevant_atom_ids,
                self.grad_relevant_atom_nums,
            ) = generate_cmbdf(
                self.nuclear_charges, self.coordinates, convolutions, gradients=gradients, **kwargs
            )
        else:
            self.representation = generate_cmbdf(
                self.nuclear_charges, self.coordinates, convolutions, **kwargs
            )


# Additional constructors.


def ASE2Compound(ase_in, **other_kwargs):
    """
    Convert an ASE object into qml2.compound.
    """
    return Compound(
        coordinates=ase_in.get_positions(), atomtypes=ase_in.get_chemical_symbols(), **other_kwargs
    )
