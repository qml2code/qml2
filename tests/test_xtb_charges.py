"""Do the tight-binding atomic charges survive OML_Compound, and respond to what they should?

The charges are the one thing OML_Compound keeps that no representation reads, so nothing
else in the suite would notice if they were lost, stale, or silently recomputed. They are
also the one thing that cannot be reconstructed exactly from what else is stored: Mulliken
analysis of the stored density reproduces them only up to the difference the occupation
refilling makes, which is 8.7e-04 per atom on a molecule the backend was smearing.

Three things are checked for every configuration:

* the charges are ARITHMETICALLY CONSISTENT -- one per atom, summing to the molecule's
  total charge. This is what catches a configuration keyword that never reached the
  calculation, since a cation whose charges sum to zero was never a cation;
* they SURVIVE A SAVEFILE bit for bit, and are restored rather than recomputed. The
  distinction matters: get_xtb_charges falls back on redoing the calculation when a
  savefile predates the field, so a test that only called it would pass even if nothing
  were ever stored;
* they RESPOND to the configuration. Charging, spin-exciting and solvating a molecule all
  have to move them, or the keyword is being accepted and ignored.

The configurations are the ones from the rest of the suite that mean something here.
HOMO_removed and LUMO_added are deliberately absent from that list and tested separately:
they rewrite occupations after convergence and never re-run the calculation, so the charges
they leave behind are the unexcited molecule's. That is a trap rather than a bug, and the
test pins it so that it stays a known one.
"""

import numpy as np
import pytest
from conftest import unsaturated_xyz_list

min_nhatoms = 5
max_nhatoms = 6
num_molecules = 32

BASE_KWARGS = {
    "software": "tblite",
    "calc_type": "GFN2-xTB",
    # IAO/IBO is ill-posed on GFN2's valence-only occupied space, as elsewhere.
    "localization_procedure": "Pipek-Mezey",
}

# Keyword arguments, and the total charge they imply.
CONFIGURATIONS = {
    "ground_state": ({}, 0),
    "cation": ({"charge": 1}, 1),
    "anion": ({"charge": -1}, -1),
    "triplet": ({"spin": 2}, 0),
    "solvated": ({"smd_solvent": "water"}, 0),
}

# Charges are in electrons, and the smallest effect tested here -- solvating a molecule --
# moves them by ~0.01 on average and ~0.06 at most. A configuration that moved nothing at
# all is the failure being looked for, so the bar is set well below the smallest real
# effect rather than close to it.
MINIMUM_RESPONSE = 1e-3


def molecules():
    return unsaturated_xyz_list(min_nhatoms, max_nhatoms, num_molecules)


def charges_of(xyz, extra_kwargs, savefile_prefix=None):
    from qml2.orb_ml import OML_Compound

    compound = OML_Compound(xyz=xyz, mats_savefile=savefile_prefix, **BASE_KWARGS, **extra_kwargs)
    compound.run_calcs()
    return compound, np.array(compound.get_xtb_charges())


@pytest.fixture(scope="module")
def ground_state_charges():
    """One ground-state calculation per molecule, shared by every configuration."""
    pytest.importorskip("tblite")
    return {xyz: charges_of(xyz, {})[1] for xyz in molecules()}


@pytest.mark.parametrize("configuration_name", list(CONFIGURATIONS))
def test_xtb_charges(configuration_name, ground_state_charges, tmp_path):
    pytest.importorskip("tblite")
    from qml2.orb_ml import OML_Compound

    extra_kwargs, total_charge = CONFIGURATIONS[configuration_name]

    for molecule_id, xyz in enumerate(molecules()):
        context = f"{configuration_name}/{xyz}"
        prefix = str(tmp_path / str(molecule_id))
        compound, charges = charges_of(xyz, extra_kwargs, savefile_prefix=prefix)

        natoms = len(compound.nuclear_charges)
        assert charges.shape == (natoms,), f"{context}: {charges.shape} charges for {natoms} atoms"
        assert abs(charges.sum() - total_charge) < 1e-6, (
            f"{context}: charges sum to {charges.sum():+.6f}, not to the total charge "
            f"{total_charge:+d} -- the configuration did not reach the calculation"
        )

        restored = OML_Compound(xyz=xyz, mats_savefile=prefix, **BASE_KWARGS, **extra_kwargs)
        restored.check_saved_files()
        assert restored.mats_created, f"{context}: the savefile was not picked up"
        assert restored.xtb_charges is not None, (
            f"{context}: the savefile carried no charges, so get_xtb_charges would have "
            "recomputed them and this would prove nothing"
        )
        assert np.array_equal(
            charges, np.array(restored.get_xtb_charges())
        ), f"{context}: charges changed across the savefile"

        if configuration_name != "ground_state":
            difference = np.abs(charges - ground_state_charges[xyz]).max()
            assert difference > MINIMUM_RESPONSE, (
                f"{context}: charges moved by only {difference:.2e} from the ground state, "
                "so the configuration was accepted and ignored"
            )


def test_xtb_charges_are_the_unexcited_molecule_s(ground_state_charges, tmp_path):
    """HOMO_removed and LUMO_added leave the charges alone, and are meant to.

    Those excitations rewrite occupation numbers -- on the electron counts, for this
    backend -- after the calculation has converged, and never run a second one. So the
    charges belong to the molecule that was calculated, not to the state being represented.
    Anyone reading them off an excited compound is reading the neutral's numbers, and this
    fails if that ever silently changes.
    """
    pytest.importorskip("tblite")

    for xyz in molecules()[:4]:
        for used_orb_type in ("HOMO_removed", "LUMO_added"):
            _, charges = charges_of(xyz, {"used_orb_type": used_orb_type})
            assert np.array_equal(charges, ground_state_charges[xyz]), (
                f"{xyz}/{used_orb_type}: the charges moved, so something re-ran the "
                "calculation and this test's premise no longer holds"
            )
