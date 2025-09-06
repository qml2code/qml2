from .compound import Compound
from .parallelization import parallelized_inheritance

inherited_attribute_names = [
    "generate_coulomb_matrix",
    "generate_bob",
    "generate_fchl19",
    "generate_slatm",
    "generate_cmbdf",
]


@parallelized_inheritance(*inherited_attribute_names, base_class=Compound)
class CompoundList(list):
    """
    Acts as a list of `Compound` class objects, except it has several attributes `attr` such that running `attr(**kwargs, **parallel_kwargs)` runs `attr(**kwargs)` for each member of the list (where `attr` has a name in the form of `generate_*`). The `parallel_kwargs` are keyword arguments `num_procs` and `fixed_num_threads` as defined in `.parallelization.embarrassingly_parallel`.
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
