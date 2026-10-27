# NOTE: Representations of spin and charge excitation Slater pairs from restricted calculation results
# was implemented here but never tested in terms of utility for ML.

from os.path import isfile

import numpy as np
from pyscf import dft, gto, lo, scf

from ..basic_utils import OptionUnavailableError, dump2pkl, loadpkl, overwrite_when_possible
from ..compound import Compound
from ..data import conversion_coefficient
from ..jit_interfaces import array_, matmul_, zeros_
from .aux_classes import OML_pyscf_calc_params, converged_mf, generate_ao_arr
from .optimize_basis_sets import optimize_molecule_basis_rescalings, rescale_mol_basis
from .representations import (
    OML_rep_params,
    dipole_coupling_matrices,
    gen_atom_sorted_pseudo_orbs,
    generate_atom_ao_ranges,
    generate_orb_rep_array,
    neglect_orb_occ,
    reconstr_mat,
)

# WARNING: THIS INTERFACE HAS PROBLEMS
try:
    from .tblite_interface import generate_pyscf_mf_mol as tblite_generate_pyscf_mf_mol
except ModuleNotFoundError:
    pass
# interface to gpu4pyscf
try:
    from gpu4pyscf import dft as dft_gpu
    from gpu4pyscf import scf as scf_gpu

    mf_creator_gpu = {"HF": scf_gpu.RHF, "UHF": scf_gpu.UHF, "KS": dft_gpu.RKS, "UKS": dft_gpu.UKS}
except ModuleNotFoundError:
    mf_creator_gpu = None

mf_creator = {"HF": scf.RHF, "UHF": scf.UHF, "KS": dft.RKS, "UKS": dft.UKS}

# Calculation types that carry their own spin treatment in the name. Every backend spells
# the pair out -- HF/UHF and KS/UKS on pySCF, GFN2-xTB/spGFN2-xTB on a tight-binding one,
# the "sp" prefix being the literature's own (spGFNn-xTB) -- so `calc_type` settles the
# restricted-or-not question by itself and `restricted` never has to guess.
unrestricted_methods = ["UHF", "UKS", "spGFN2-xTB", "spGFN1-xTB"]

# Each calculation type beside the one differing from it ONLY in spin treatment, so that a
# contradiction between `restricted` and `calc_type` can name the alternative meant.
spin_treatment_counterparts = {
    "HF": "UHF",
    "KS": "UKS",
    "GFN2-xTB": "spGFN2-xTB",
    "GFN1-xTB": "spGFN1-xTB",
}
spin_treatment_counterparts.update(
    {unrestricted: restricted for restricted, unrestricted in spin_treatment_counterparts.items()}
)

HF_methods = ["HF", "UHF"]

KS_methods = ["KS", "UKS"]

#   TO-DO rename to "available_orbital_types" now that Boys is also used?
available_orb_types = ["standard", "HOMO_removed", "LUMO_added", "first_excitation"]

available_localization_procedures = ["Boys", "IBO", "Pipek-Mezey", None]

available_software = ["pySCF", "molpro", "tblite"]

available_calc_types = {
    "pySCF": ["HF", "UHF", "KS", "UKS"],
    # Restricted spelling first in each pair: entry 0 is what `calc_type=None` falls back on.
    "tblite": ["GFN2-xTB", "spGFN2-xTB", "GFN1-xTB", "spGFN1-xTB"],
    "molpro": ["HF", "UHF"],
}


def assign_avail_check(val, avail_val):
    """The canonical spelling of `val` from `avail_val`, matched ignoring case.

    Returns the entry AS LISTED rather than as passed. That matters more than the
    case-insensitivity itself: the rest of the module compares against those exact strings
    (`software == "pySCF"`, `calc_type in HF_methods`), and `calc_type` is handed on to
    backends that accept only their own spelling -- tblite wants "GFN2-xTB" and nothing
    else. So case is forgiven at the boundary and there is exactly one spelling past it.

    Non-string options (`None` among the localization procedures) are matched by equality.
    """
    for available_val in avail_val:
        if val == available_val:
            return available_val
        if isinstance(val, str) and isinstance(available_val, str):
            if val.lower() == available_val.lower():
                return available_val
    print("Not found:", val, avail_val)
    raise OptionUnavailableError


def assign_restrictedness(restricted, calc_type):
    """Whether the calculation is spin-restricted, reconciled with `calc_type`.

    `calc_type` already decides this wherever a backend spells both variants out, so the
    default -- `None` -- reads it off, and every backend then behaves the same way. An
    explicit value is honoured when it agrees and REFUSED when it does not: letting either
    one quietly win would leave the compound disagreeing with itself about what was
    computed, while the savefile is named after `calc_type` regardless.

    Restricted is not the same as closed-shell. A restricted calculation of a molecule with
    unpaired electrons is an ordinary thing to ask for -- ROHF on pySCF, plain GFN2 on a
    tight-binding backend -- and `spin` goes on describing the state independently of the
    ansatz used to reach it.
    """
    implied_restricted = calc_type not in unrestricted_methods
    if restricted is None:
        return implied_restricted
    if bool(restricted) != implied_restricted:
        counterpart = spin_treatment_counterparts.get(calc_type)
        message = (
            f"calc_type '{calc_type}' is "
            + ("restricted" if implied_restricted else "unrestricted")
            + f", but restricted={restricted} was requested"
        )
        if counterpart is not None:
            message += f"; use calc_type '{counterpart}' for that"
        raise OptionUnavailableError(message)
    return implied_restricted


class OML_Compound(Compound):
    """'Orbital Machine Learning (OML) Compound' extends the Compound class to include
    results of ab initio calculations performed with pySCF or tblite packages.

    xyz             - xyz file use to create base Compound object.
    mats_savefile   - the file where results of the ab initio calculations are stored;
                      if it is the name is specified then the results would be imported from the file if it exists
                      or saved to the file otherwise.
    calc_type       - type of the calculation (for now only HF with orb localization and the default basis set are supported).
    """

    def __init__(
        self,
        xyz=None,
        xyz_lines=None,
        coordinates=None,
        nuclear_charges=None,
        atomtypes=None,
        mats_savefile=None,
        calc_type=None,
        restricted=None,
        basis="sto-3g",
        used_orb_type="standard",
        use_Huckel=False,
        optimize_geometry=False,
        dft_xc="lda,vwn",
        dft_nlc="",
        optimize_ao_rescalings=False,
        ao_rescalings_optimization_kwargs={},
        basis_rescaled_orbitals=None,
        software="pySCF",
        pyscf_calc_params=None,
        use_pyscf_localization=True,
        write_full_pyscf_chkfile=False,
        # NOTE: the model this selects depends on `software`. With pySCF it is SMD
        # (pyscf.solvent.smd, see generate_pyscf_mf); with tblite it is ALPB
        # (`calculator.add("alpb-solvation", ...)` in tblite_interface), because tblite
        # has no SMD. The name predates the tblite backend and is kept for compatibility.
        smd_solvent=None,
        localization_procedure="IBO",
        use_gpu=False,
        temp_calc_dir=None,
        **kwargs,
    ):
        super().__init__(
            xyz=xyz,
            xyz_lines=xyz_lines,
            coordinates=coordinates,
            nuclear_charges=nuclear_charges,
            atomtypes=atomtypes,
            **kwargs,
        )

        self.software = assign_avail_check(software, available_software)
        self.use_gpu = use_gpu

        if calc_type is None:
            self.calc_type = available_calc_types[self.software][0]
        else:
            self.calc_type = assign_avail_check(calc_type, available_calc_types[self.software])
        self.restricted = assign_restrictedness(restricted, self.calc_type)
        # How many unpaired electrons the REPRESENTED state has, as opposed to the one that
        # was calculated. They differ only where a backend applies an excitation by moving a
        # single electron between the spin channels; everywhere else this is `spin`, which
        # goes on being what reaches the backend and what the savefile name records.
        self.representation_spin = self.spin
        # Backend-reported quantities that nothing else can reproduce exactly. None on any
        # backend that does not provide them; see get_xtb_charges.
        self.xtb_mo_occ = None
        self.xtb_charges = None

        self.mats_savefile = mats_savefile
        self.mats_savefile_final = None

        self.basis = basis
        self.used_orb_type = assign_avail_check(used_orb_type, available_orb_types)
        self.use_Huckel = use_Huckel
        self.optimize_geometry = optimize_geometry
        self.use_pyscf_localization = use_pyscf_localization
        self.smd_solvent = smd_solvent
        # Canonicalised here rather than only checked at localization time, so that a
        # misspelling is caught at construction and the stored value is the one the
        # localizer branches on below.
        self.localization_procedure = assign_avail_check(
            localization_procedure, available_localization_procedures
        )
        # related to generating and using ao rescalings.
        self.optimize_ao_rescalings = optimize_ao_rescalings
        self.ao_rescalings_optimization_kwargs = ao_rescalings_optimization_kwargs
        self.ao_rescalings = None
        self.basis_rescaled_orbitals = basis_rescaled_orbitals

        self.temp_calc_dir = temp_calc_dir

        self.pyscf_chkfile = None
        self.full_pyscf_chkfile = None

        self.write_full_pyscf_chkfile = write_full_pyscf_chkfile

        if pyscf_calc_params is None:
            self.pyscf_calc_params = OML_pyscf_calc_params()
        else:
            self.pyscf_calc_params = pyscf_calc_params
        self.dft_xc = dft_xc
        self.dft_nlc = dft_nlc

        self.mats_created = None

        self.mo_coeff = None
        self.mo_occ = None
        self.mo_energy = None
        self.aos = None
        self.atom_ao_ranges = None
        self.e_tot = None

        self.ovlp_mat = None
        self.dipole_mats = None
        self.atom_coords_bohr = None
        self.h0_mat = None
        self.integral_source = None
        self.orb_mat = None
        if self.calc_type in HF_methods:
            self.j_mat = None
            self.k_mat = None
            self.fock_mat = None
        if self.optimize_geometry:
            self.opt_coords = None

        self.orb_reps = None

    # Parameters of the reference ab initio calculations.
    def calc_params(self):
        return {
            "calc_type": self.calc_type,
            "basis": self.basis,
            "software": self.software,
            "use_pyscf_localization": self.use_pyscf_localization,
            "used_orb_type": self.used_orb_type,
            "use_Huckel": self.use_Huckel,
            "pyscf_calc_params": self.pyscf_calc_params,
            "smd_solvent": self.smd_solvent,
            "localization_procedure": self.localization_procedure,
            "dft_xc": self.dft_xc,
            "dft_nlc": self.dft_nlc,
        }

    def assign_calc_res(self, calc_res_dict):
        self.mo_coeff = calc_res_dict["mo_coeff"]
        self.mo_occ = calc_res_dict["mo_occ"]
        self.mo_energy = calc_res_dict["mo_energy"]
        self.aos = calc_res_dict["aos"]
        self.atom_ao_ranges = calc_res_dict["atom_ao_ranges"]
        self.e_tot = calc_res_dict["e_tot"]

        self.ovlp_mat = calc_res_dict["ovlp_mat"]
        self.orb_mat = calc_res_dict["orb_mat"]
        # .get, not [...]: savefiles written before FJD existed are still perfectly good
        # for every other representation and should keep loading. Asking for FJD without
        # them raises where the need arises instead.
        self.atom_coords_bohr = calc_res_dict.get("atom_coords_bohr")
        self.h0_mat = calc_res_dict.get("h0_mat")
        # Falls back on `spin`, which is what every savefile written before the two could
        # differ was implicitly recording.
        self.representation_spin = calc_res_dict.get("representation_spin", self.spin)
        self.xtb_mo_occ = calc_res_dict.get("xtb_mo_occ")
        self.xtb_charges = calc_res_dict.get("xtb_charges")

        if self.optimize_ao_rescalings:
            self.ao_rescalings = calc_res_dict["ao_rescalings"]

        if self.calc_type in HF_methods:
            self.j_mat = calc_res_dict["j_mat"]
            self.k_mat = calc_res_dict["k_mat"]
            self.fock_mat = calc_res_dict["fock_mat"]
        if self.optimize_geometry:
            self.opt_coords = calc_res_dict["opt_coords"]

    def default_savefile_prename(self, mats_savefile):
        """
        The beginning of *.pkl file used to dump ab initio results for later re-use.
        """
        savefile_prename = mats_savefile + "." + self.calc_type + "." + self.basis
        if self.use_Huckel:
            savefile_prename += ".Huckel"
        if self.optimize_geometry:
            savefile_prename += ".geom_opt"
        if self.charge != 0:
            savefile_prename += ".charge_" + str(self.charge)
        if self.spin != self.default_spin_val():
            savefile_prename += ".spin_" + str(self.spin)
        if self.calc_type in KS_methods:
            savefile_prename += ".xc_" + self.dft_xc + ".nlc_" + str(self.dft_nlc)
        if self.software != "pySCF":
            savefile_prename += (
                "." + self.software + ".pySCF_loc_" + str(self.use_pyscf_localization)
            )
        if self.optimize_ao_rescalings:
            savefile_prename += ".optimized_ao"
        return savefile_prename

    def savefile_orb_component(self):
        output = "." + self.used_orb_type
        if self.localization_procedure is not None:
            output += ".localization_" + self.localization_procedure
        return output

    def check_saved_files(self):
        if self.mats_savefile is None:
            return
        if self.mats_savefile.endswith(".pkl"):
            self.pyscf_chkfile = self.mats_savefile[:-3] + "chkfile"
            self.mats_savefile_final = self.mats_savefile
        else:
            savefile_prename = self.default_savefile_prename(self.mats_savefile)
            self.pyscf_chkfile = savefile_prename + ".chkfile"
            savefile_prename += self.savefile_orb_component()
            self.mats_savefile_final = savefile_prename + ".pkl"
        if self.write_full_pyscf_chkfile:
            self.full_pyscf_chkfile = self.pyscf_chkfile + "_full"
        self.mats_created = ext_isfile(self.mats_savefile_final)
        if self.mats_created:
            # Import ab initio results from the savefile. A file that will not load is
            # treated as absent and recalculated rather than killing the run: savefiles
            # written before dump2pkl became atomic can be damaged, and one bad molecule
            # should not take a job over the whole dataset with it. It is reported, because
            # a damaged savefile is worth knowing about.
            try:
                precalc_vals = loadpkl(self.mats_savefile_final)
            except Exception as loading_error:
                print("WARNING: could not read", self.mats_savefile_final, "-", loading_error)
                print("WARNING: recalculating it instead")
                self.mats_created = False
            else:
                self.assign_calc_res(precalc_vals)

    def run_calcs(self, initial_guess_comp=None):
        """Runs the ab initio calculations if they are necessary.

        pyscf_calc_params   - object of OML_pyscf_calc_params class containing parameters of the pySCF calculation. (To be made more useful.)
        """
        self.check_saved_files()
        if self.mats_created:
            return
        if self.software == "molpro":
            if self.used_orb_type != "standard_orb":
                raise OptionUnavailableError
            from .molpro_interface import get_molpro_calc_data

            calc_data = get_molpro_calc_data[self.calc_type](self)
            self.assign_calc_res(calc_data)
            # TO-DO IMPORT BASIS FROM MOLPRO BEFORE LOCALIZATION???
            if self.use_pyscf_localization:
                self.orb_mat = self.localized_orbs()
            self.create_mats_savefile()
            return
        # Run the pySCF calculations.
        mf, pyscf_mol = self.generate_pyscf_mf_mol(initial_guess_comp=initial_guess_comp)
        # Special operations.
        # alter_mo_occ rewrites occupation numbers to express an excitation, by finding the
        # gap between occupied and empty. A tight-binding backend has no reliable gap to
        # find, so it applies the excitation upstream instead, as a shift of the per-channel
        # electron counts -- see tblite_interface.CHANNEL_ELECTRON_SHIFTS -- and its
        # occupations arrive already excited.
        if self.used_orb_type != "standard_orb" and self.software != "tblite":
            self.mo_occ = self.alter_mo_occ(mf.mo_occ)
        backend_representation_spin = getattr(mf, "representation_spin", None)
        # `or` would be wrong here: a represented multiplicity of 0 is a real answer, not a
        # missing one, and falling back on `spin` would then re-open the channels an
        # excitation just closed.
        if backend_representation_spin is not None:
            self.representation_spin = backend_representation_spin
        self.xtb_mo_occ = getattr(mf, "xtb_mo_occ", None)
        self.xtb_charges = getattr(mf, "xtb_charges", None)

        self.mo_coeff = self.adjust_spin_mat_dimen(mf.mo_coeff)
        # The per-channel breakdown is passed on only for an UNALTERED ground state.
        # alter_mo_occ above rewrites the occupations in place for HOMO_removed /
        # LUMO_added / first_excitation, and expresses that rewrite on the summed array;
        # past that point the backend's own filling no longer describes what is being
        # represented, so the threshold split has to interpret the altered array instead.
        self.mo_occ = self.adjust_spin_mo_occ(
            mf.mo_occ,
            channel_occupations=(
                getattr(mf, "channel_occupations", None)
                # The tblite path applies the excitation to the counts, so its breakdown
                # already describes the excited state and is used for every orbital type.
                if self.software == "tblite" or self.used_orb_type == "standard"
                else None
            ),
        )
        self.mo_energy = self.adjust_spin_mat_dimen(mf.mo_energy)
        self.aos = generate_ao_arr(pyscf_mol)
        self.atom_ao_ranges = generate_atom_ao_ranges(pyscf_mol)
        self.e_tot = mf.e_tot
        self.ovlp_mat = array_(pyscf_mol.intor_symmetric("int1e_ovlp"))
        # Bohr, matching the integrals -- taken from the mol rather than self.coordinates,
        # which is Angstrom on the pySCF path. Free for both backends.
        self.atom_coords_bohr = array_(pyscf_mol.atom_coords())
        # The backend's zeroth-order Hamiltonian, where it has one; free, it is already in
        # the result. Used for the electrostatic channel of FJD as F - H0.
        self.h0_mat = getattr(mf, "h0_mat", None)
        # NOT the dipole integrals: those are deferred to dipole_matrices(). On a
        # tight-binding backend they are reconstructed through pySCF, which needs the
        # parametrization export and hence a recent backend version -- a cost no
        # calculation should pay for a representation it was never asked for.
        self.integral_source = pyscf_mol

        if self.calc_type in HF_methods:
            self.j_mat = self.adjust_spin_mat_dimen(mf.get_j(), already_spin_adj=True)
            self.k_mat = self.adjust_spin_mat_dimen(mf.get_k(), already_spin_adj=True)
            if self.smd_solvent is None:
                fock_kwargs = {}
            else:
                fock_kwargs = {"dm": mf.make_rdm1()}
            self.fock_mat = self.adjust_spin_mat_dimen(mf.get_fock(**fock_kwargs))

        self.orb_mat = self.localized_orbs(pyscf_mol=pyscf_mol)
        self.create_mats_savefile()

    def localized_orbs(self, pyscf_mol=None):
        occ_orb_arrs = self.occ_orbs()
        if pyscf_mol is None:
            pyscf_mol = self.generate_pyscf_mol()
        orb_mat = []
        for occ_orb_arr in occ_orb_arrs:
            if occ_orb_arr.size == 0:
                orb_mat.append(None)
            else:
                if self.localization_procedure not in available_localization_procedures:
                    raise OptionUnavailableError
                if self.localization_procedure is None:
                    new_orb_mat = occ_orb_arr
                elif self.localization_procedure == "IBO":
                    new_orb_mat = lo.ibo.ibo(
                        pyscf_mol, occ_orb_arr, **self.pyscf_calc_params.orb_kwargs
                    )
                else:
                    kernel_kwargs = {}
                    if self.localization_procedure == "Boys":
                        loc_obj = lo.boys.Boys(
                            pyscf_mol, mo_coeff=occ_orb_arr
                        )  # TO-DO: self.pyscf_calc_params.orb_kwargs?
                    else:  # Pipek-Mezey
                        loc_obj = lo.pipek.PipekMezey(pyscf_mol, occ_orb_arr)
                        loc_obj.pop_method = "Mulliken"
                        if self.software != "pySCF":
                            kernel_kwargs = {"mo_coeff": occ_orb_arr}
                    loc_obj.kernel(**kernel_kwargs)
                    new_orb_mat = loc_obj.mo_coeff
                orb_mat.append(array_(new_orb_mat))
        return orb_mat

    def create_mats_savefile(self):
        self.mats_created = True
        if self.mats_savefile is None:
            return
        # TO-DO Check ways for doing it in a less ugly way.
        saved_data = {
            "mo_coeff": self.mo_coeff,
            "mo_occ": self.mo_occ,
            "mo_energy": self.mo_energy,
            "aos": self.aos,
            "ovlp_mat": self.ovlp_mat,
            "orb_mat": self.orb_mat,
            "atom_ao_ranges": self.atom_ao_ranges,
            "e_tot": self.e_tot,
            "atom_coords_bohr": self.atom_coords_bohr,
            "h0_mat": self.h0_mat,
            # Not derivable from `spin` on restore: an excitation applied to the electron
            # COUNTS leaves the represented state with a different multiplicity from the
            # calculated one, and num_spins() reads this rather than `spin`. Without it a
            # restored excited compound reports one spin channel over two channels' worth
            # of stored occupations, and quietly builds half its representations.
            "representation_spin": self.representation_spin,
            "xtb_mo_occ": self.xtb_mo_occ,
            "xtb_charges": self.xtb_charges,
        }
        if self.calc_type in HF_methods:
            saved_data = {
                **saved_data,
                "j_mat": self.j_mat,
                "k_mat": self.k_mat,
                "fock_mat": self.fock_mat,
            }
        if self.optimize_ao_rescalings:
            saved_data = {**saved_data, "ao_rescalings": self.ao_rescalings}
        if self.optimize_geometry:
            saved_data["opt_coords"] = self.opt_coords
        dump2pkl(saved_data, self.mats_savefile_final)

    def generate_orb_reps(
        self, rep_params: OML_rep_params = OML_rep_params(), initial_guess_comp=None
    ):
        """Generates orbital representation.

        rep_params  - object of oml_representations.OML_rep_params class containing parameters of the orbital representation.
        initial_guess_comp - if present defined OML_Compound class instance whose electronic density can be used as initial guess for the calculations.
        """
        if not self.mats_created:
            self.run_calcs(initial_guess_comp=initial_guess_comp)
        self.orb_reps = []
        for spin in range(self.num_spins()):
            if self.orb_mat[spin] is None:
                continue
            #   Generate the array of orbital representations.
            coupling_matrices = None
            only_other_couplings = None
            l2_norm_merges = None
            if rep_params.fjd_coup_mats:
                (
                    coupling_matrices,
                    only_other_couplings,
                    l2_norm_merges,
                ) = self.fjd_coupling_matrices(spin)
            if coupling_matrices is None:
                coupling_matrices = (
                    self.fock_mat[spin],
                    self.jk_mat_spin_rescaled(self.j_mat[spin]),
                    self.jk_mat_spin_rescaled(self.k_mat[spin]),
                )
            cur_orb_rep_array = generate_orb_rep_array(
                self.orb_mat[spin],
                rep_params,
                self.aos,
                self.atom_ao_ranges,
                self.ovlp_mat,
                array_(coupling_matrices),
                only_other_couplings=only_other_couplings,
                l2_norm_merges=l2_norm_merges,
            )
            self.orb_reps += cur_orb_rep_array
            orb_occ = orb_occ_prop_coeff(self)
            for orb_rep_counter in range(len(self.orb_reps)):
                if not self.orb_reps[orb_rep_counter].virtual:
                    self.orb_reps[orb_rep_counter].rho = orb_occ
            if rep_params.atom_sorted_pseudo_orbs:
                self.orb_reps = gen_atom_sorted_pseudo_orbs(self.orb_reps)

    def rebuild_integral_source(self):
        """Recreate the object integrals are taken from, without redoing the calculation.

        run_calcs sets self.integral_source, but a compound restored from a savefile never
        runs it, and the dipole integrals themselves are deliberately not stored -- at
        3*nao**2 per compound per conformer they would dominate the savefiles. Everything
        needed to rebuild the SOURCE is stored, though, so it is remade rather than
        recomputed:

        * pySCF: a Mole over the same geometry and basis. No SCF is involved; it only has
          to be able to evaluate one-electron integrals.
        * a tight-binding backend: a PseudoMole around the overlap and atom-to-AO partition
          that were saved with the other matrices, which is all it needs to serve integrals
          in the backend's own AO convention.
        """
        if self.software == "tblite":
            from .tblite_interface import pseudo_mole_from_ao_data

            return pseudo_mole_from_ao_data(
                self.ovlp_mat,
                self.atom_ao_ranges,
                array_([ao.angular for ao in self.aos]),
                self.nuclear_charges,
                self.coordinates * conversion_coefficient["Angstrom_Bohr"],
            )
        return self.generate_pyscf_mol()

    def integral_source_object(self):
        """The integral source, rebuilt from stored data if this came from a savefile."""
        if self.integral_source is None:
            self.integral_source = self.rebuild_integral_source()
        return self.integral_source

    def atom_coordinates_bohr(self):
        """Nuclear coordinates in Bohr, matching the integrals.

        Taken from the integral source when absent, which covers savefiles written before
        they were stored alongside the other matrices.
        """
        if self.atom_coords_bohr is None:
            self.atom_coords_bohr = array_(self.integral_source_object().atom_coords())
        return self.atom_coords_bohr

    def dipole_matrices(self):
        """AO dipole integrals about the global origin, computed once, on demand.

        Deferred rather than taken during run_calcs: on a tight-binding backend these come
        from a pySCF Mole rebuilt over the backend's basis, which needs the parametrization
        export, and no calculation should require that for a representation it does not use.
        """
        if self.dipole_mats is None:
            self.dipole_mats = array_(self.integral_source_object().intor("int1e_r"))
        return self.dipole_mats

    def fjd_coupling_matrices(self, spin):
        """The FJD coupling matrices: Fock, the electrostatic operator, and the D block.

        Returns ``(matrices, only_other_couplings, l2_norm_merges)``.

        The D block is four matrices that become two channels. The three shifted-dipole
        components are one vector operator, contracted to a scalar per coupling by the l2
        merge; the bond projection is a scalar operator whose self coupling vanishes
        identically, so its self slot is dropped rather than carried as a constant-zero
        column.

        The F and J slots have the same meaning in both backends but different providers:

        * pySCF: the Fock and Coulomb matrices the mean-field object supplies.
        * a tight-binding backend: no converged Fock matrix is retained, so F is
          reconstructed from the FULL orbital set as ``S C eps C^T S`` -- occupied-only
          would give the energy-weighted density, which agrees with F only on
          occupied-occupied couplings -- and J is ``F - H0``, the self-consistent-charge
          potential. That is a density-response operator rather than a clean Coulomb one;
          carry HF intuition about the J channel loosely.
        """
        if self.calc_type in HF_methods:
            fock_mat = self.fock_mat[spin]
            electrostatic_mat = self.jk_mat_spin_rescaled(self.j_mat[spin])
        else:
            if self.h0_mat is None:
                raise Exception(
                    "FJD needs either a Fock and Coulomb matrix or a zeroth-order "
                    f"Hamiltonian; software '{self.software}' provided neither."
                )
            fock_mat = reconstr_mat(
                matmul_(self.ovlp_mat, self.mo_coeff[spin]),
                mo_energy=self.mo_energy[spin],
                mat_type="Fock",
            )
            electrostatic_mat = fock_mat - self.h0_mat
        dipole_channels = dipole_coupling_matrices(
            self.dipole_matrices(),
            self.ovlp_mat,
            self.atom_coordinates_bohr(),
            array_([ao.atom_id for ao in self.aos]),
        )
        matrices = (fock_mat, electrostatic_mat, *dipole_channels)
        # Only the bond projection (last) lacks a self coupling; only the three dipole
        # components (2, 3, 4) merge into one channel.
        only_other_couplings = array_([False] * 5 + [True])
        return matrices, only_other_couplings, [[2, 3, 4]]

    def jk_mat_spin_rescaled(self, mat):
        return mat / jk_mat_spin_rescaling(self)

    #   Find maximal value of angular momentum for AOs of current molecule.
    def find_max_angular_momentum(self):
        if not self.mats_created:
            self.run_calcs()
        max_angular_momentum = 0
        for ao in self.aos:
            max_angular_momentum = max(max_angular_momentum, ao.angular)
        return max_angular_momentum

    def generate_pyscf_mol(self):
        # Convert between the Mole class used in pySCF and the Compound class used in the rest of QML.
        mol = gto.Mole()
        # atom_coords should be in Angstrom.
        mol.atom = [
            [atom_type, atom_coords]
            for atom_type, atom_coords in zip(self.atomtypes, self.coordinates)
        ]
        mol.charge = self.charge
        mol.spin = self.spin
        mol.basis = self.basis
        try:
            mol.build()
        except KeyError as KE:
            if str(KE)[1:-1] == mol.basis:
                import basis_set_exchange as bse

                # WARNING: was never used, therefore not %100 sure it works correctly.
                mol.basis = bse.get_basis(mol.basis, fmt="nwchem")
                mol.build()
            else:
                raise KE
        if self.optimize_ao_rescalings and (self.ao_rescalings is None):
            self.ao_rescalings = optimize_molecule_basis_rescalings(
                mol,
                pyscf_calc_params=self.pyscf_calc_params,
                basis_rescaled_orbitals=self.basis_rescaled_orbitals,
                **self.ao_rescalings_optimization_kwargs,
            )
        if self.ao_rescalings is not None:
            rescale_mol_basis(
                mol, self.ao_rescalings, basis_rescaled_orbitals=self.basis_rescaled_orbitals
            )
        return mol

    def generate_pyscf_mf(self, pyscf_mol, initial_guess_comp=None):
        if self.use_gpu:
            used_mf_creator = mf_creator_gpu
            assert (
                used_mf_creator is not None
            ), "check that gpu4pyscf is installed: https://github.com/pyscf/gpu4pyscf"
        else:
            used_mf_creator = mf_creator
        mf = used_mf_creator[self.calc_type](pyscf_mol)
        if self.calc_type in KS_methods:
            mf.xc = self.dft_xc
            mf.nlc = self.dft_nlc
        mf.chkfile = self.pyscf_chkfile
        if (not self.use_Huckel) and ext_isfile(self.pyscf_chkfile):
            mf.init_guess = "chkfile"
        if self.smd_solvent is not None:
            if self.use_gpu:
                from gpu4pyscf.solvent.smd import smd_for_scf as smd_for_scf_gpu

                mf = smd_for_scf_gpu(mf)
            else:
                from pyscf.solvent.smd import smd_for_scf

                mf = smd_for_scf(mf)
            mf.with_solvent.solvent = self.smd_solvent
        # TODO why this does not work???
        if initial_guess_comp is None:
            dm_init_guess = None
        else:
            dm_init_guess = create_dm_init_guess(
                mf.make_rdm1, initial_guess_comp, self.num_spins()
            )
        mf = converged_mf(
            mf,
            pyscf_calc_params=self.pyscf_calc_params,
            dm_init_guess=dm_init_guess,
            use_Huckel=self.use_Huckel,
            mats_savefile=self.mats_savefile_final,
            use_gpu=self.use_gpu,
        )
        return mf

    def generate_pyscf_mf_mol(self, initial_guess_comp=None):
        if self.software == "tblite":
            return tblite_generate_pyscf_mf_mol(
                self,
                calc_type=self.calc_type,
                charge=self.charge,
                spin=self.spin,
                used_orb_type=self.used_orb_type,
                solvent=self.smd_solvent,
            )
        #        if self.software == "tblite":
        #            return tblite_generate_pyscf_mf_mol(self)
        if ext_isfile(self.full_pyscf_chkfile):
            return loadpkl(self.full_pyscf_chkfile)
        pyscf_mol = self.generate_pyscf_mol()
        mf = self.generate_pyscf_mf(pyscf_mol, initial_guess_comp=initial_guess_comp)
        if self.optimize_geometry:
            from pyscf.geomopt.geometric_solver import optimize

            pyscf_mol = optimize(mf)
            self.opt_coords = pyscf_mol.atom_coords(unit="Ang")
            mf = self.generate_pyscf_mf(pyscf_mol)
        output = (mf, pyscf_mol)
        if self.full_pyscf_chkfile is not None:
            dump2pkl(output, self.full_pyscf_chkfile)
        return output

    def get_xtb_charges(self):
        """The tight-binding backend's own atomic partial charges.

        Stored with the other results, so a restored compound has them without recomputing
        anything. A savefile written before they were stored has none, and rather than
        refuse, the calculation is redone -- it is a single point on a geometry that is
        already known, and the alternative is a field that stays empty forever on
        everything cached so far.

        They are the GROUND STATE's charges. `used_orb_type` excitations rewrite
        occupations after convergence and never re-run the calculation, so an excited
        compound's charges are the ones its unexcited calculation produced; they do not
        describe the state being represented.

        Mulliken analysis of the stored density reproduces them closely but not exactly --
        `mo_occ` has been refilled from the orbital ordering, and where the backend was
        smearing, the two densities differ (8.7e-04 per atom on octatetraene). Hence
        keeping the backend's own numbers rather than deriving them.
        """
        if self.xtb_charges is None:
            if self.software != "tblite":
                raise OptionUnavailableError(
                    f"software '{self.software}' computes no tight-binding charges; "
                    "get_xtb_charges is available for tblite calculations only"
                )
            mf, _ = self.generate_pyscf_mf_mol()
            self.xtb_charges = mf.xtb_charges
            if self.xtb_mo_occ is None:
                self.xtb_mo_occ = mf.xtb_mo_occ
        return self.xtb_charges

    def calc_unrestricted(self):
        match self.software:
            case "pySCF" | "tblite":
                return not self.restricted
            case _:
                raise Exception("Unknown/untested software")

    def alter_mo_occ(self, mo_occ):
        if self.calc_unrestricted():
            true_mo_occ = mo_occ[0]
        else:
            true_mo_occ = mo_occ
        for i, orb_occ in enumerate(true_mo_occ):
            if orb_occ < neglect_orb_occ:
                if self.calc_unrestricted():
                    if add_LUMO(self):
                        mo_occ[0][i] = 1.0
                    if remove_HOMO(self):
                        mo_occ[0][i - 1] = 0.0
                else:
                    if add_LUMO(self):
                        mo_occ[i] = 2.0
                    if remove_HOMO(self):
                        mo_occ[i - 1] = 0.0
                break
        return mo_occ

    def occ_orbs(self):
        output = []
        for mo_occ_arr, mo_coeff_arr in zip(self.mo_occ, self.mo_coeff):
            cur_occ_orbs = []
            for basis_func in mo_coeff_arr:
                cur_line = []
                for occ, orb_coeff in zip(mo_occ_arr, basis_func):
                    if occ > neglect_orb_occ:
                        cur_line.append(orb_coeff)
                cur_occ_orbs.append(cur_line)
            output.append(array_(cur_occ_orbs))
        return output

    def HOMO_en(self):
        # HOMO energy.
        return self.mo_energy[0][self.LUMO_orb_id() - 1]

    def LUMO_en(self):
        # LUMO energy.
        return self.mo_energy[0][self.LUMO_orb_id()]

    def HOMO_LUMO_gap(self):
        return self.LUMO_en() - self.HOMO_en()

    def LUMO_orb_id(self):
        # index of the LUMO orbital.
        for orb_id, occ in enumerate(self.mo_occ[0]):
            if occ < neglect_orb_occ:
                return orb_id

    def adjust_spin_mat_dimen(self, matrices, already_spin_adj=False):
        if self.calc_unrestricted():
            return matrices
        if already_spin_adj and self.representation_spin != 0:
            return array_(matrices)
        return array_([matrices for _ in range(self.num_spins())])

    def adjust_spin_mo_occ(self, mo_occ, channel_occupations=None):
        """Occupations laid out one row per spin channel.

        Three ways in, in decreasing order of how much the calculation already told us:

        * an unrestricted calculation answers per channel to begin with, so its array is
          taken as it stands;
        * a restricted calculation of a closed shell has a single channel, and the summed
          occupations are already it;
        * a restricted calculation with unpaired electrons has to be SPLIT, and that is
          what `channel_occupations` is for.

        The split used to be inferred by thresholding the summed array: an orbital joins
        channel s when its total occupation clears s + neglect_orb_occ. That reads an
        integral filling correctly and a smeared one incorrectly. A radical cation whose
        hole is spread over three near-degenerate orbitals comes back as
        (1.80, 1.80, 1.40) -- all three above the 1.1 cut -- so the beta channel is handed
        as many orbitals as alpha and the molecule ends up represented as though no
        electron were missing. pySCF's HF, which this module was written around, fills
        integrally and never provokes it; a tight-binding backend fills at a finite
        electronic temperature and provokes it routinely.

        So where the backend can say which channel each electron went into -- computed
        from the orbital ordering and the electron count, see
        tblite_interface.aufbau_channel_occupations -- that breakdown is used directly and
        nothing is inferred. A genuinely degenerate partly-filled shell still comes back
        fractional in both channels, because sharing evenly over orbitals the energies
        cannot tell apart is the honest description of it; what goes away is the accidental
        case, where the smearing reflected the electronic temperature rather than symmetry.

        The threshold path stays for everything that provides no breakdown, and for the
        altered occupations of HOMO_removed / LUMO_added / first_excitation -- those are
        expressed on the summed array by alter_mo_occ, and describe an occupation the
        ground-state filling no longer matches.
        """
        if self.calc_unrestricted():
            return array_(mo_occ)
        if self.num_spins() == 1:
            return self.adjust_spin_mat_dimen(mo_occ)
        if channel_occupations is not None:
            return array_(channel_occupations)
        new_mo_occ = zeros_((2, mo_occ.shape[0]))
        for orb_id, occ_num in enumerate(mo_occ):
            for spin_id in range(self.num_spins()):
                if occ_num > spin_id + neglect_orb_occ:
                    new_mo_occ[spin_id, orb_id] = 1.0
        return new_mo_occ

    def num_spins(self):
        if self.calc_unrestricted() or self.representation_spin != 0:
            return 2
        else:
            return 1


def create_dm_init_guess(rdm_maker, initial_guess_comp, nspins):
    copied_nspins = initial_guess_comp.num_spins()
    dm = []
    for spin_id in range(nspins):
        if copied_nspins == 1:
            copied_spin_id = 0
        else:
            copied_spin_id = spin_id
        # KK.: I think that rdm_maker requires np.array as input, but not sure
        dm.append(
            rdm_maker(
                mo_coeff=np.array(initial_guess_comp.mo_coeff[copied_spin_id]),
                mo_occ=np.array(initial_guess_comp.mo_occ[copied_spin_id]),
            )
        )
    if nspins == 1:
        return dm[0]
    else:
        return array_(dm)


def remove_HOMO(oml_comp):
    used_orb_type = oml_comp.used_orb_type
    return (used_orb_type == "HOMO_removed") or (used_orb_type == "first_excitation")


def add_LUMO(oml_comp):
    used_orb_type = oml_comp.used_orb_type
    return (used_orb_type == "LUMO_added") or (used_orb_type == "first_excitation")


def orb_occ_prop_coeff(comp):
    if comp.num_spins() == 1:
        return 2.0
    else:
        return 1.0


def jk_mat_spin_rescaling(comp):
    nspins = comp.num_spins()
    if nspins == 1:
        return 2.0
    else:
        return 1.0


def ext_isfile(filename):
    if filename is None:
        return False
    else:
        return isfile(filename)


class OML_Slater_pair:
    def __init__(
        self,
        second_oml_comp_kwargs={},
        initial_guess_from_first=False,
        comp1=None,
        comp2=None,
        **first_oml_comp_kwargs,
    ):
        """
        Defines a pair of Slater determinants representing an eletronic process.
        """
        self.initial_guess_from_first = initial_guess_from_first
        if comp1 is None:
            comp1 = OML_Compound(**first_oml_comp_kwargs)

        used_second_oml_comp_kwargs = overwrite_when_possible(
            first_oml_comp_kwargs, second_oml_comp_kwargs
        )
        if comp2 is None:
            comp2 = OML_Compound(**used_second_oml_comp_kwargs)
        self.comps = [comp1, comp2]

    def get_initial_guess_comp(self):
        if self.initial_guess_from_first:
            return self.comps[0]
        else:
            return None

    def get_other_comp_ids(self):
        return range(1, len(self.comps))

    def run_calcs(self):
        self.comps[0].run_calcs()
        initial_guess_comp = self.get_initial_guess_comp()
        for i in self.get_other_comp_ids():
            self.comps[i].run_calcs(initial_guess_comp=initial_guess_comp)

    def generate_orb_reps(
        self,
        rep_params: OML_rep_params = OML_rep_params(),
    ):
        self.comps[0].generate_orb_reps(rep_params=rep_params)
        initial_guess_comp = self.get_initial_guess_comp()
        for i in self.get_other_comp_ids():
            self.comps[i].generate_orb_reps(
                rep_params=rep_params, initial_guess_comp=initial_guess_comp
            )


class OML_Slater_pairs(OML_Slater_pair):
    def __init__(
        self,
        other_oml_comp_kwargs_list=[],
        initial_guess_from_first=False,
        **first_oml_comp_kwargs,
    ):
        """
        For conveniently managing situations when same initial point is used in several Slater pairs.
        """
        self.initial_guess_from_first = initial_guess_from_first
        self.comps = [OML_Compound(**first_oml_comp_kwargs)]
        for other_oml_comp_kwargs in other_oml_comp_kwargs_list:
            other_kwargs = overwrite_when_possible(first_oml_comp_kwargs, other_oml_comp_kwargs)
            other_comp = OML_Compound(**other_kwargs)
            self.comps.append(other_comp)

    def get_Slater_pairs(self):
        return [
            OML_Slater_pair(
                comp1=self.comps[0],
                comp2=self.comps[i],
                initial_guess_from_first=self.initial_guess_from_first,
            )
            for i in self.get_other_comp_ids()
        ]


def ASE2OML_Compound(ase_in, **other_kwargs):
    """
    Convert an ASE object into OML_Compound.
    """
    return OML_Compound(
        coordinates=ase_in.get_positions(), atomtypes=ase_in.get_chemical_symbols(), **other_kwargs
    )
