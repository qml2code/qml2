"""
Note that tblite uses atomic units for everything, including orbital energies and atomic coordinates.
"""
from functools import partial

import numpy as np

from ...basic_utils import ExceptionRaisingClass, OptionUnavailableError
from ...data import conversion_coefficient
from ..aux_classes import PseudoMF, PseudoMole
from .data import ANGULAR_MOMENTUM, slater_to_gauss

try:
    from tblite import library as tblite_library
    from tblite.interface import Calculator
except ModuleNotFoundError as ex:
    Calculator = ExceptionRaisingClass(ex)
    tblite_library = ExceptionRaisingClass(ex)


tblite_max_iter = (
    2048  # increased compared to the default 250 due to some twisted molecules in RedDB.
)

default_calc_type = "GFN2-xTB"
default_verbosity = 0  # set 1 for verbose mode of tblite.

num_valence_electrons = {1: 1, 6: 4, 7: 5, 8: 6, 9: 7, 15: 5, 16: 6}


def gfn2_element_parameters():
    """Per-element GFN2 basis parameters, straight from the tblite library.

    ``{symbol: {"shells": ["2s", "2p"], "slater": [...], "ngauss": [...], ...}}`` -- the
    shell labels carry the principal and azimuthal quantum numbers, ``slater`` the Slater
    exponents, and ``ngauss`` the STO-nG expansion order, which VARIES by element and by
    shell (H is 3, C/N/O are 4, sulfur's d shell is 3), so it must be read rather than
    assumed. Feed these to :func:`.data.slater_to_gauss` to obtain the Gaussian primitives.

    Taken from the library rather than the ``tblite`` CLI or a molden dump: this is the
    same parametrization the calculator itself runs with, so the reconstructed basis
    cannot drift from the one a given calculation used, and there is no CLI-versus-library
    version skew to police.

    NB ``dump_param`` writes into a tblite TABLE object, not to a file, despite the name.
    """
    parametrization = tblite_library.new_param()
    tblite_library.export_gfn2_param(parametrization)
    table = tblite_library.new_table()
    tblite_library.dump_param(parametrization, table)
    return tblite_library.table_to_dict(table)["element"]


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


#: Reordering of pySCF's spherical components into tblite's, per angular momentum.
#: Measured, not assumed -- and pinned rather than re-derived, so that a convention change
#: on either side fails sanity_checks/tblite/check_gfn2_overlap.py loudly. Absent entries
#: (here l=0, and l=2 which turns out to be the identity) need no permutation.
PYSCF_TO_TBLITE_COMPONENT_ORDER = {1: (1, 2, 0)}


def gfn2_pyscf_basis(nuclear_charges):
    """A pySCF basis dict reproducing the GFN2 basis for the elements given.

    Each shell is the STO-nG expansion of one Slater function, with the exponents and
    expansion orders read out of the running tblite library
    (:func:`gfn2_element_parameters`) and the tabulated expansion applied by
    :func:`.data.slater_to_gauss`.

    The coefficients go in WITHOUT tblite's primitive normalization: pySCF's basis input
    already takes contraction coefficients over normalized primitives, so passing the
    normalized form would apply the factor twice.
    """
    from tblite.interface import SYMBOL_TO_NUMBER

    number_to_symbol = {number: symbol for symbol, number in SYMBOL_TO_NUMBER.items()}
    element_parameters = gfn2_element_parameters()
    basis = {}
    for nuclear_charge in set(int(charge) for charge in nuclear_charges):
        symbol = number_to_symbol[nuclear_charge]
        entry = element_parameters[symbol]
        shells = []
        for label, zeta, ngauss in zip(entry["shells"], entry["slater"], entry["ngauss"]):
            angular_momentum = ANGULAR_MOMENTUM[label[1].lower()]
            alpha, coeff = slater_to_gauss(
                int(label[0]), angular_momentum, float(zeta), int(ngauss)
            )
            shells.append(
                [angular_momentum] + [[float(a), float(c)] for a, c in zip(alpha, coeff)]
            )
        basis[nuclear_charge] = shells
    return basis


def pseudo_mole_from_calculator(calculator, overlap_matrix, nuclear_charges, coordinates):
    """A :class:`.aux_classes.PseudoMole` carrying tblite's AO data and its basis.

    The overlap, atom-to-AO partition and angular momenta are tblite's own and are used as
    given. The basis is attached as a FACTORY rather than a built dict, so integrals tblite
    does not expose (the dipole matrices, in particular) can be taken from pySCF and
    converted into tblite's AO convention on the way out -- without every ordinary
    calculation paying for a basis it never uses, and without requiring the parametrization
    export (tblite >= 0.7.0) of callers that only ever want the overlap.
    """
    ao_ranges, angular_momenta = tblite_ao_ranges_angular_momenta(calculator)
    return pseudo_mole_from_ao_data(
        overlap_matrix, ao_ranges, angular_momenta, nuclear_charges, coordinates
    )


def pseudo_mole_from_ao_data(
    overlap_matrix, atom_ao_ranges, angular_momenta, nuclear_charges, coordinates
):
    """The same, from AO data already in hand rather than from a live calculator.

    Split out because a compound restored from a savefile has all of this -- the overlap,
    the atom-to-AO partition and the angular momenta are stored with the rest of the
    matrices -- but no calculator to ask, and re-running the calculation just to rebuild a
    wrapper around data already on disk would be absurd.
    """
    return PseudoMole(
        ovlp_mat=overlap_matrix,
        atom_ao_ranges=atom_ao_ranges,
        angular_momenta=angular_momenta,
        nuclear_charges=np.array(nuclear_charges),
        coordinates=np.array(coordinates),
        basis=partial(gfn2_pyscf_basis, tuple(int(c) for c in nuclear_charges)),
        component_order=PYSCF_TO_TBLITE_COMPONENT_ORDER,
    )


SPIN_POLARIZED_PREFIX = "sp"


def tblite_method_spin_treatment(calc_type):
    """Split a calc_type into the method tblite knows and whether to spin-polarize it.

    ``spGFN2-xTB`` is the literature's name for spin-polarized GFN2, but tblite exposes no
    method under it -- ``Calculator("spGFN2-xTB", ...)`` raises ``TBLiteValueError``. The
    spin-polarized variants are built instead as the plain method plus the
    spin-polarization container, which is also how tblite's own tests construct spGFN1.
    """
    if calc_type.startswith(SPIN_POLARIZED_PREFIX):
        return calc_type[len(SPIN_POLARIZED_PREFIX) :], True
    return calc_type, False


def fill_channel(energies, electrons, degeneracy_tolerance):
    """One spin channel filled from the bottom up, sharing inside a degenerate shell.

    Orbitals take one electron each in order of energy. When the electrons run out part
    way through a set of orbitals that are degenerate to within `degeneracy_tolerance`,
    the remainder is split evenly over that set rather than assigned to whichever of them
    sorted first -- the sort cannot tell them apart, and on a symmetric molecule the
    separation is numerical noise (8.5e-14 Hartree across the methane cation's t2 shell).
    """
    occupations = np.zeros(energies.shape[0])
    order = np.argsort(energies)
    remaining = electrons
    position = 0
    while remaining > 0 and position < order.shape[0]:
        shell = [order[position]]
        while (
            position + len(shell) < order.shape[0]
            and abs(energies[order[position + len(shell)]] - energies[shell[0]])
            <= degeneracy_tolerance
        ):
            shell.append(order[position + len(shell)])
        share = min(remaining, len(shell)) / len(shell)
        for orbital in shell:
            occupations[orbital] = share
        remaining -= min(remaining, len(shell))
        position += len(shell)
    return occupations


# How many electrons each spin channel holds relative to the ground state. The excitation
# is expressed HERE, on the counts, rather than by overwriting occupation numbers after the
# fact: the counts are what fill_channel consumes, so the HOMO is orbital n_alpha - 1 in the
# energy ordering and the LUMO is orbital n_alpha, by construction. Looking for them instead
# as a gap in the occupations -- which is what alter_mo_occ does, and what works on pySCF --
# has no reliable version on a backend that fills at a finite electronic temperature: the
# gap is smeared shut, and the orbital found by the scan is whichever of a near-degenerate
# set happened to sort last.
#
# The shift is +/-1 electron, not the +/-2 that zeroing a doubly occupied orbital amounts
# to, so a closed-shell molecule excited this way becomes genuinely open-shell. See
# `representation_spin` for what that means downstream.
CHANNEL_ELECTRON_SHIFTS = {
    "standard": (0, 0),
    "HOMO_removed": (-1, 0),
    "LUMO_added": (1, 0),
}


def aufbau_channel_occupations(
    orbital_energies,
    reported_occupations,
    spin,
    used_orb_type="standard",
    degeneracy_tolerance=1e-6,
):
    """Occupations rebuilt from the orbital ordering and the electron count alone.

    A tight-binding backend fills its orbitals at a finite electronic temperature, so what
    it reports can be fractional: octatetraene comes back with 1.9987 and 0.0013 where an
    integral filling would say 2 and 0. Everything downstream then asks a yes/no question
    of those numbers -- ``occ > neglect_orb_occ`` decides which orbitals are localized and
    which spin channels each appears in -- so a molecule with a small gap can gain or lose
    an orbital on numerical noise, changing the NUMBER of representations it produces
    rather than their values. That discontinuity is what this removes.

    The electron count is recovered from the reported occupations, whose SUM is integral
    even where the individual values are not, and dealt out one channel at a time: the
    restricted case is the two channels added back together, which is also what makes a
    singly occupied orbital come out as exactly 1.

    Fractional occupations are not abolished, only the accidental ones. A genuinely
    degenerate partly-filled shell keeps its even share -- anything else would pick one of
    a set of orbitals the energies cannot distinguish -- so the methane cation still comes
    back as [2, 5/3, 5/3, 5/3]. What goes away is smearing across a shell that is NOT
    degenerate, which is where the thresholds downstream were exposed.

    Orbitals are ordered here rather than assumed sorted. tblite does return them
    ascending, but nothing in its interface promises to keep doing so.
    """
    energies = np.asarray(orbital_energies)
    reported = np.asarray(reported_occupations)
    electrons = int(round(float(reported.sum())))
    unpaired = 0 if spin is None else int(spin)
    if unpaired < 0 or electrons < unpaired or (electrons - unpaired) % 2 != 0:
        raise ValueError(
            f"cannot fill {electrons} electrons with {unpaired} of them unpaired: the two "
            "have to agree in parity and the unpaired ones have to fit. Check `charge` and "
            "`spin` -- note that a tight-binding backend counts VALENCE electrons, so this "
            "count is smaller than the sum of the nuclear charges."
        )
    if used_orb_type not in CHANNEL_ELECTRON_SHIFTS:
        # first_excitation is the one that cannot be written as a shift of counts: moving
        # an electron from the HOMO to the LUMO is (n - 1) then (n + 1), which nets back to
        # the ground state. It needs the HOMO shell skipped rather than the count changed,
        # and nothing in the tree asks for it on this backend.
        raise OptionUnavailableError(
            f"used_orb_type '{used_orb_type}' has no electron-count form, so it cannot be "
            "applied to a tight-binding calculation; see CHANNEL_ELECTRON_SHIFTS"
        )
    alpha_shift, beta_shift = CHANNEL_ELECTRON_SHIFTS[used_orb_type]
    per_channel = (
        (electrons + unpaired) // 2 + alpha_shift,
        (electrons - unpaired) // 2 + beta_shift,
    )
    if energies.ndim == 1:
        # Restricted: both channels are filled over the SAME orbitals, and the two rows are
        # returned separately rather than added. Adding them is the caller's business for
        # `mo_occ`; keeping them apart is the entire point of this function, because a
        # restricted open shell cannot be split back apart from the sum without guessing.
        return np.array(
            [fill_channel(energies, count, degeneracy_tolerance) for count in per_channel]
        )
    # Spin-polarized: a set of orbitals per channel, each with its own ordering.
    return np.array(
        [
            fill_channel(energies[channel], count, degeneracy_tolerance)
            for channel, count in enumerate(per_channel)
        ]
    )


def generate_pyscf_mf_mol(
    compound,
    coordinate_units="Angstrom",
    verbosity=default_verbosity,
    calc_type=default_calc_type,
    spin=None,
    used_orb_type="standard",
    charge=None,
    solvent=None,
):
    if coordinate_units == "Bohr":
        used_coordinates = compound.coordinates
    else:
        used_coordinates = compound.coordinates * conversion_coefficient["Angstrom_Bohr"]
    # Both assignments, not one replacing the other: `uhf` says how many electrons are
    # unpaired, `charge` how many are missing, and a charged radical has to state both.
    # Dropping `uhf` went unnoticed for as long as it did because it changes nothing
    # without spin polarization -- but it decides the occupations as soon as there is any.
    calculator_kwargs = {}
    if spin is not None:
        calculator_kwargs["uhf"] = spin
    if charge is not None:
        calculator_kwargs["charge"] = charge

    method, spin_polarized = tblite_method_spin_treatment(calc_type)

    calculator = Calculator(
        method, compound.nuclear_charges, used_coordinates, **calculator_kwargs
    )
    if spin_polarized:
        # No scaling argument: the default is what spGFNn-xTB means, and it is what
        # tblite's own spGFN1 test asserts its reference energy against.
        calculator.add("spin-polarization")
    if solvent is not None:
        # The caller's keyword is `smd_solvent` (OML_compound), but what happens here is ALPB:
        # tblite has no SMD, and ALPB is its native implicit solvation. So on this backend
        # "smd_solvent" names the solvent, NOT the model -- a naming oversight kept for
        # compatibility with existing savefiles. The pySCF backend of the same keyword really
        # does run SMD (pyscf.solvent.smd in oml_compound.generate_pyscf_mf), so one keyword
        # selects two different models depending on `software`.
        calculator.add("alpb-solvation", solvent)

    ao_ranges, angular_momenta = tblite_ao_ranges_angular_momenta(calculator)

    calculator.set("save-integrals", 1)
    calculator.set("max-iter", tblite_max_iter)
    calculator.set("verbosity", verbosity)
    res = calculator.singlepoint()
    output = res.dict()

    channel_occupations = aufbau_channel_occupations(
        output["orbital-energies"],
        output["orbital-occupations"],
        spin,
        used_orb_type=used_orb_type,
    )

    mf = PseudoMF(
        e_tot=output["energy"],
        mo_coeff=output["orbital-coefficients"],
        # Not output["orbital-occupations"]: those can be fractional, and what reads them
        # downstream only ever thresholds them. See aufbau_channel_occupations.
        #
        # Summed across the channels for a restricted calculation, because one row carrying
        # 2 per doubly occupied orbital is the shape the rest of the module reads. A
        # spin-polarized result is already per channel and stays that way. Either way the
        # unsummed breakdown travels alongside in `channel_occupations`, so a restricted
        # OPEN shell never has to be divided up again by thresholding the sum.
        mo_occ=(channel_occupations if spin_polarized else channel_occupations.sum(axis=0)),
        channel_occupations=channel_occupations,
        xtb_mo_occ=output["orbital-occupations"],
        xtb_charges=output["charges"],
        # The multiplicity of the state being REPRESENTED, which an excitation of one
        # electron makes differ from the multiplicity that was calculated.
        representation_spin=abs(
            int(round(channel_occupations[0].sum() - channel_occupations[1].sum()))
        ),
        mo_energy=output["orbital-energies"],
        # Verified to be the zeroth-order, charge-independent Hamiltonian rather than the
        # converged Fock matrix: identical for neutral and +1 at a frozen geometry, and
        # distinct from S C eps C^T S. tblite retains no converged Fock matrix, so F has
        # to be reconstructed from the orbitals.
        h0_mat=output["hamiltonian-matrix"],
    )

    mol = pseudo_mole_from_calculator(
        calculator,
        output["overlap-matrix"],
        compound.nuclear_charges,
        used_coordinates,
    )

    return mf, mol
