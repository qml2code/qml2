# Convenience classes storing temporary data required by some representations and doing the calculation.
from .cMBDF import generate_cmbdf, get_asize, get_convolutions
from .slatm import generate_slatm, get_slatm_mbtypes


class RepresentationCalculator:
    def __call__(self, nuclear_charges, coordinates, **kwargs):
        return self.calc_representation(nuclear_charges, coordinates, **kwargs)


class SLATMCalculator(RepresentationCalculator):
    def __init__(self, all_nuclear_charges):
        self.slatm_mbtypes = get_slatm_mbtypes(all_nuclear_charges)

    def calc_representation(self, nuclear_charges, coordinates, **kwargs):
        return generate_slatm(nuclear_charges, coordinates, self.slatm_mbtypes, **kwargs)


class cMBDFCalculator(RepresentationCalculator):
    def __init__(self, all_nuclear_charges, **conv_kwargs):
        self.asize = get_asize(all_nuclear_charges)
        self.convs = get_convolutions(**conv_kwargs)

    def calc_representation(self, nuclear_charges, coordinates, **kwargs):
        return generate_cmbdf(nuclear_charges, coordinates, self.convs, asize=self.asize, **kwargs)
