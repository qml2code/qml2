from conftest import (
    add_checksum_to_dict,
    compare_or_create,
    int2rng,
    perturbed_xyz_examples,
    qm7_nuclear_charges,
)
from test_geometric_representations import run_single_representation_test
from test_slatm import run_slatm_test_for_compounds

from qml2 import Compound, CompoundList
from qml2.jit_interfaces import array_, concatenate_
from qml2.representations.numba_calculators import (
    FCHL19JITCalculator,
    SLATMJITCalculator,
    cMBDFJITCalculator,
)

ntest_mols = 2000


def test_fchl19():
    rep_func = FCHL19JITCalculator(
        elements=qm7_nuclear_charges, rcut=8.0, acut=12.0
    ).single_coordinate_calc
    run_single_representation_test("fchl19", local=True, rep_func=rep_func)


def test_slatm(local=False):
    rng = int2rng(2)
    xyzs = perturbed_xyz_examples(rng, ntest_mols)
    compounds = CompoundList([Compound(xyz=xyz) for xyz in xyzs])
    calculators = {}
    for alchemy in [True, False]:
        calculators[alchemy] = SLATMJITCalculator(
            all_compounds=compounds, local=local, alchemy=alchemy
        ).single_coordinate_calc
    run_slatm_test_for_compounds(compounds, local=local, calculators=calculators)


def test_aslatm():
    test_slatm(local=True)


def test_cmbdf():
    # written to mirror test_cMBDF inside the test_cMBDF.py script and capitalize on the same tests.
    rng = int2rng(2)

    xyzs = perturbed_xyz_examples(rng, ntest_mols)
    compounds = CompoundList([Compound(xyz=xyz) for xyz in xyzs])
    all_nuclear_charges = compounds.all_nuclear_charges()

    checksums = {}

    for local, local_lbl, rep_merger in [(True, "local", concatenate_), (False, "global", array_)]:
        cmbdf_calculator = cMBDFJITCalculator(
            local=local, all_nuclear_charges=all_nuclear_charges
        ).single_coordinate_calc
        compounds.generate_rep_from_func(cmbdf_calculator, test_mode=True)
        merged_reps = rep_merger(compounds.all_representations())
        add_checksum_to_dict(
            checksums, f"cMBDF_{local_lbl}", merged_reps, rng, nstack_checksums=8, stacks=32
        )
    compare_or_create(checksums, "cMBDF", max_rel_difference=1.0e-10)


if __name__ == "__main__":
    test_fchl19()
    test_aslatm()
    test_slatm()
    test_cmbdf()
