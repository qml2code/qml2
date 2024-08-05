from .compound import Compound
from .parallelization import parallelized_inheritance

inherited_attribute_names = [
    "generate_coulomb_matrix",
    "generate_bob",
    "generate_fchl19",
    "generate_slatm",
]


@parallelized_inheritance(*inherited_attribute_names, base_class=Compound)
class CompoundList(list):
    """
    Provides a convenient way to embarrasingly parallelize operations with Compound class.
    """

    def all_nuclear_charges(self):
        """
        List of nuclear charges of all list members
        """
        return [comp.nuclear_charges for comp in self]

    def all_representations(self):
        """
        List of representation arrays of all list members
        """
        return [comp.representation for comp in self]
