"""Does a compound restored from a savefile reproduce the one that wrote it?

Every workflow that matters caches its ab initio results -- the paper scripts run over tens
of thousands of molecules and would be unaffordable otherwise -- so a representation that
cannot be rebuilt from a savefile is a representation that works exactly once. The failures
this guards against are all of the same shape: nothing raises, and the restored compound
quietly produces DIFFERENT features from the run that saved them. Two real examples:

* FJD's dipole integrals are deliberately not stored (3*nao**2 per compound per conformer
  would dominate the savefile), so they are recomputed from an integral source -- which is
  created in run_calcs, and a restored compound never calls run_calcs;
* an excitation applied to the electron counts leaves the represented state with a
  different multiplicity from the calculated one, and num_spins() reads that rather than
  `spin`. Until it was saved too, a restored excited compound reported one spin channel
  over two channels' worth of stored occupations and built 3 representations where a fresh
  run built 7.

Neither is visible from a single run, and neither raises. So the test is an INVARIANCE
test rather than a benchmark: it runs each configuration twice against the same savefile
prefix and requires the second pass to reproduce the first bit for bit. Nothing is compared
against stored reference values, so there is nothing to regenerate and nothing that can
drift between machines.

Two properties make it water-tight rather than vacuous:

* the restore is CONFIRMED before anything is compared. If the savefile were not found the
  second pass would simply redo the calculation and of course agree -- a check that passes
  while proving nothing -- so check_saved_files is consulted directly to establish that the
  matrices came back from disk and that no integral source came with them, which is the
  state the failures above occur in;
* the comparison covers STRUCTURE as well as values: how many orbitals, which atoms each
  keeps, and their weights, not only the feature vectors. A restored compound that silently
  drops half its orbitals has to fail, and comparing concatenated feature arrays alone
  would let that through.

The configurations mirror those used elsewhere in the suite: both backends, restricted and
unrestricted, plain compounds and Slater pairs excited the same ways test_FJK and
test_msorf_features excite them.
"""

import numpy as np
import pytest
from conftest import unsaturated_xyz_list

min_nhatoms = 5
max_nhatoms = 6
num_molecules = 16


# Every backend gets the flavours it can support: FJK needs a Coulomb and an exchange
# matrix, which a tight-binding calculation does not have, while FJD works from either.
PYSCF_FLAVOURS = [{"max_angular_momentum": 1}, {"max_angular_momentum": 1, "fjd_coup_mats": True}]
TBLITE_FLAVOURS = [{"max_angular_momentum": 1, "fjd_coup_mats": True}]

BACKENDS = {
    "pyscf_restricted": ({"basis": "sto-3g", "calc_type": "HF"}, PYSCF_FLAVOURS, "pyscf"),
    "pyscf_unrestricted": ({"basis": "sto-3g", "calc_type": "UHF"}, PYSCF_FLAVOURS, "pyscf"),
    "tblite_restricted": (
        {
            "software": "tblite",
            "calc_type": "GFN2-xTB",
            # IAO/IBO is ill-posed on GFN2's valence-only occupied space, as elsewhere.
            "localization_procedure": "Pipek-Mezey",
        },
        TBLITE_FLAVOURS,
        "tblite",
    ),
    "tblite_spin_polarized": (
        {
            "software": "tblite",
            "calc_type": "spGFN2-xTB",
            "localization_procedure": "Pipek-Mezey",
        },
        TBLITE_FLAVOURS,
        "tblite",
    ),
}

# `None` is a plain OML_Compound; anything else is an OML_Slater_pair whose second
# determinant differs from the first by these keyword arguments.
SECOND_COMPOUND_KWARGS = {
    "single": None,
    "HOMO_removed": {"used_orb_type": "HOMO_removed"},
    "LUMO_added": {"used_orb_type": "LUMO_added"},
    "charge": {"charge": 1},
    "solvation": {"smd_solvent": "water"},
}


def build(xyz, savefile_prefix, compound_kwargs, second_compound_kwargs):
    from qml2.orb_ml import OML_Compound, OML_Slater_pair

    kwargs = {"xyz": xyz, "mats_savefile": savefile_prefix, **compound_kwargs}
    if second_compound_kwargs is None:
        return OML_Compound(**kwargs)
    return OML_Slater_pair(second_oml_comp_kwargs=second_compound_kwargs, **kwargs)


def compounds_of(oml_object):
    """A Slater pair's determinants, or the single compound, uniformly."""
    return getattr(oml_object, "comps", [oml_object])


def restored_from_savefile(oml_object):
    """Whether every compound came back from disk rather than being recalculated.

    check_saved_files is what run_calcs consults before deciding to compute anything, so
    calling it directly settles the question without running a calculation either way. A
    restored compound has its matrices in place and -- the state the failures above live
    in -- no integral source to recompute anything from.
    """
    for compound in compounds_of(oml_object):
        compound.check_saved_files()
        if not compound.mats_created or compound.integral_source is not None:
            return False
    return True


def orbital_contributions(oml_object, rep_params):
    """Every atomic contribution of every orbital, tagged with where it came from.

    The atom id and the weight travel with the feature vector so that a restored compound
    which keeps a different set of atoms, or orders its orbitals differently, fails on the
    structure rather than slipping through on the numbers.
    """
    oml_object.generate_orb_reps(rep_params=rep_params)
    return [
        (
            compound_id,
            orbital_id,
            atom_rep.atom_id,
            float(atom_rep.rho),
            np.array(atom_rep.scalar_reps),
        )
        for compound_id, compound in enumerate(compounds_of(oml_object))
        for orbital_id, orbital in enumerate(compound.orb_reps)
        for atom_rep in orbital.orb_atom_reps
    ]


def backend_charges(oml_object):
    """The backend's own atomic charges, per compound; None where it computes none.

    They are stored rather than recomputed, so they have to survive the savefile like
    everything else -- and unlike the features, nothing else in this file would notice if
    they came back wrong, since no representation reads them.
    """
    from qml2.basic_utils import OptionUnavailableError

    charges = []
    for compound in compounds_of(oml_object):
        try:
            charges.append(np.array(compound.get_xtb_charges()))
        except OptionUnavailableError:
            return None
    return charges


def assert_identical(fresh, restored, context):
    assert len(fresh) == len(
        restored
    ), f"{context}: {len(restored)} atomic contributions after restart, {len(fresh)} before"
    for first, second in zip(fresh, restored):
        assert (
            first[:3] == second[:3]
        ), f"{context}: contribution moved: {first[:3]} -> {second[:3]}"
        assert first[3] == second[3], f"{context}: weight changed: {first[3]} -> {second[3]}"
        assert np.array_equal(first[4], second[4]), f"{context}: features changed"


@pytest.mark.parametrize("backend_name", list(BACKENDS))
@pytest.mark.parametrize("excitation_name", list(SECOND_COMPOUND_KWARGS))
def test_FJK_restart(backend_name, excitation_name, tmp_path):
    compound_kwargs, flavours, required_module = BACKENDS[backend_name]
    pytest.importorskip(required_module)
    second_compound_kwargs = SECOND_COMPOUND_KWARGS[excitation_name]

    from qml2.orb_ml.representations import OML_rep_params

    xyzs = unsaturated_xyz_list(min_nhatoms, max_nhatoms, num_molecules)
    assert len(xyzs) == num_molecules

    for molecule_id, xyz in enumerate(xyzs):
        for flavour_id, flavour_kwargs in enumerate(flavours):
            prefix = str(tmp_path / f"{molecule_id}_{flavour_id}")
            context = f"{backend_name}/{excitation_name}/{xyz}/flavour {flavour_id}"
            rep_params = OML_rep_params(**flavour_kwargs)

            fresh_object = build(xyz, prefix, compound_kwargs, second_compound_kwargs)
            fresh = orbital_contributions(fresh_object, rep_params)
            fresh_charges = backend_charges(fresh_object)
            second_pass = build(xyz, prefix, compound_kwargs, second_compound_kwargs)
            assert restored_from_savefile(
                second_pass
            ), f"{context}: the second pass recalculated instead of restoring, nothing was tested"
            assert_identical(fresh, orbital_contributions(second_pass, rep_params), context)
            if fresh_charges is not None:
                restored_charges = backend_charges(second_pass)
                assert all(
                    np.array_equal(first, second)
                    for first, second in zip(fresh_charges, restored_charges)
                ), f"{context}: the backend's atomic charges changed across the restart"


def test_FJK_restart_notices_a_broken_savefile(tmp_path):
    """The mutation check: with a field missing from the savefile, does the comparison fail?

    Everything above passing shows the code is right. It does not show that the comparison
    could reject anything -- a test that agrees with itself no matter what is not a guard.
    So one savefile is rewritten without `representation_spin`, which is precisely what a
    savefile written before that field existed looks like, and which is what made a restored
    excited compound report a single spin channel over two channels' worth of stored
    occupations: 3 representations where a fresh run built 7, with nothing raised.

    The restore is still confirmed first, for the same reason as above, so this fails for
    the intended reason rather than because the savefile went missing entirely.
    """
    pytest.importorskip("tblite")
    from qml2.basic_utils import dump2pkl, loadpkl
    from qml2.orb_ml.representations import OML_rep_params

    compound_kwargs, flavours, _ = BACKENDS["tblite_restricted"]
    second_compound_kwargs = SECOND_COMPOUND_KWARGS["HOMO_removed"]
    rep_params = OML_rep_params(**flavours[0])
    xyz = unsaturated_xyz_list(min_nhatoms, max_nhatoms, num_molecules)[0]
    prefix = str(tmp_path / "mutated")

    fresh = orbital_contributions(
        build(xyz, prefix, compound_kwargs, second_compound_kwargs), rep_params
    )

    mutated = 0
    for savefile in tmp_path.iterdir():
        contents = loadpkl(str(savefile))
        if isinstance(contents, dict) and "representation_spin" in contents:
            del contents["representation_spin"]
            dump2pkl(contents, str(savefile))
            mutated += 1
    assert mutated, "no savefile carried representation_spin, so nothing was mutated"

    second_pass = build(xyz, prefix, compound_kwargs, second_compound_kwargs)
    assert restored_from_savefile(second_pass), "the mutated savefile was not used at all"
    # Matched on the message, not just the type: the count is what the missing field
    # actually moves -- 112 atomic contributions become 74 -- and an AssertionError raised
    # by anything else here would mean this passed for the wrong reason.
    with pytest.raises(AssertionError, match="atomic contributions after restart"):
        assert_identical(fresh, orbital_contributions(second_pass, rep_params), "mutation check")
