"""
Procedures for conveniently initializing correct arguments and keyword arguments for representation functions and ensuring their consistency.
"""
from ..compound import Compound
from ..utils import get_possible_nuclear_charges
from .cMBDF import generate_cmbdf, get_asize, get_convolutions
from .fchl19 import generate_fchl19
from .slatm import generate_slatm, get_slatm_mbtypes
from .standard_geometric import compute_ncm, generate_bob, generate_coulomb_matrix, get_bob_bags


def get_check_all_nuclear_charges(all_nuclear_charges=None, all_compounds=None):
    if all_nuclear_charges is None:
        assert (
            all_compounds is not None
        ), "Either `all_nuclear_charges` or `all_compounds` keyword argument should be defined!"
        all_nuclear_charges = [comp.nuclear_charges for comp in all_compounds]
    return all_nuclear_charges


def get_check_elements(elements=None, all_nuclear_charges=None, all_compounds=None):
    if elements is None:
        assert (all_nuclear_charges is not None) or (
            all_compounds is not None
        ), "Either `elements`, `all_nuclear_charges`, or `all_compounds` should be defined!"
        all_nuclear_charges = get_check_all_nuclear_charges(
            all_nuclear_charges=all_nuclear_charges, all_compounds=all_compounds
        )
        elements = get_possible_nuclear_charges(all_nuclear_charges)
    return elements


class RepresentationCalculator:
    def __init__(self, rep_function, *rep_args, **rep_kwargs):
        self.rep_function = rep_function
        self.rep_args = rep_args
        self.rep_kwargs = rep_kwargs

    def __call__(self, nuclear_charges, coordinates, **kwargs):
        assert nuclear_charges.shape[0] == coordinates.shape[0]
        return self.rep_function(
            nuclear_charges, coordinates, *self.rep_args, **self.rep_kwargs, **kwargs
        )

    def calc_from_compound(self, compound: Compound, **kwargs):
        return self(compound.nuclear_charges, compound.coordinates, **kwargs)


class SLATMCalculator(RepresentationCalculator):
    def __init__(self, all_nuclear_charges=None, all_compounds=None, **rep_kwargs):
        """
        Calculator for SLATM representation. Use keyword argument `local=True` to generate aSLATM instead.
        """
        all_nuclear_charges = get_check_all_nuclear_charges(
            all_nuclear_charges=all_nuclear_charges, all_compounds=all_compounds
        )
        self.slatm_mbtypes = get_slatm_mbtypes(all_nuclear_charges)
        super().__init__(generate_slatm, self.slatm_mbtypes, **rep_kwargs)


class cMBDFCalculator(RepresentationCalculator):
    def __init__(
        self,
        all_nuclear_charges=None,
        all_compounds=None,
        local=True,
        pad=None,
        rcut=10.0,
        n_atm=2.0,
        cell=None,
        gradients=False,
        **convolution_kwargs,
    ):
        """
        Calculator for cMBDF representation.
        """
        all_nuclear_charges = get_check_all_nuclear_charges(
            all_nuclear_charges=all_nuclear_charges, all_compounds=all_compounds
        )
        self.asize = get_asize(all_nuclear_charges)
        self.convs = get_convolutions(rcut=rcut, **convolution_kwargs)
        super().__init__(
            generate_cmbdf,
            self.convs,
            asize=self.asize,
            local=local,
            pad=pad,
            rcut=rcut,
            n_atm=n_atm,
            cell=cell,
            gradients=gradients,
        )


class FCHL19Calculator(RepresentationCalculator):
    def __init__(self, elements=None, all_nuclear_charges=None, all_compounds=None, **kwargs):
        """
        Calculator for FCHL19 representation.
        """
        self.elements = get_check_elements(
            elements=elements, all_nuclear_charges=all_nuclear_charges, all_compounds=all_compounds
        )
        super().__init__(generate_fchl19, elements=self.elements, **kwargs)


class CMCalculator(RepresentationCalculator):
    def __init__(self, size=None, all_compounds=None, all_nuclear_charges=None):
        """
        Calculator for Coulomb Matrix representation.
        """
        if size is None:
            assert not (
                all_compounds is None and all_nuclear_charges is None
            ), "`size`, `all_compounds`, or `all_nuclear_charges` should be defined!"
            all_nuclear_charges = get_check_all_nuclear_charges(
                all_nuclear_charges=all_nuclear_charges, all_compounds=all_compounds
            )
            size = max([len(nc) for nc in all_nuclear_charges])
        super().__init__(generate_coulomb_matrix, size=size)


class BoBCalculator(RepresentationCalculator):
    def __init__(self, all_nuclear_charges=None, elements=None, all_compounds=None):
        """
        Calculator for Bag-of-Bonds representation.
        """
        all_nuclear_charges = get_check_all_nuclear_charges(
            all_nuclear_charges=all_nuclear_charges, all_compounds=all_compounds
        )
        if elements is None:
            self.elements = get_possible_nuclear_charges(all_nuclear_charges)
        self.elements = get_check_elements(
            elements=elements, all_nuclear_charges=all_nuclear_charges, all_compounds=all_compounds
        )
        self.bags = get_bob_bags(all_nuclear_charges, elements=self.elements)
        self.ncm = compute_ncm(self.bags)
        super().__init__(generate_bob, self.bags, ncm=self.ncm, elements=self.elements)
