# Subroutines for finding optimal basis sets (used to create training data for adaptive basis sets, see arXiv:2404.16942).
# The procedure is mostly numerically equivalent to the one used in arXiv:2404.16942.
from copy import deepcopy

import numpy as np
import scipy.optimize as spo
from pyscf import scf
from pyscf.grad.rhf import make_rdm1e

from ..basic_utils import display_scipy_convergence
from ..finite_difference import grid_finite_difference_coefficients_1var
from .aux_classes import OML_pyscf_calc_params, converged_mf


def last_contracted_and_uncontracted_orbital_ids(basis_list):
    encountered_momenta = []
    last_contracted = []
    uncontracted = []
    en_basis_list = list(enumerate(basis_list))
    for bo_id, bo in en_basis_list[::-1]:
        if len(bo) == 2:
            uncontracted.append(bo_id)
            continue
        momentum = bo[0]
        if momentum not in encountered_momenta:
            encountered_momenta.append(momentum)
            last_contracted.append(bo_id)
    return last_contracted, uncontracted


def default_basis_rescaled_orbitals(basis):
    """
    Define a rescaling procedure that, for each element, assigns one rescaling factor
    to the last contracted shell, and one rescaling factor corresponding to all
    uncontracted shells at once.
    """
    rescaling_dictionnary = {}
    for el, basis_list in basis.items():
        rescaling_dictionnary[el] = []
        for orb_list in last_contracted_and_uncontracted_orbital_ids(basis_list):
            if orb_list:
                rescaling_dictionnary[el].append(orb_list)
    return rescaling_dictionnary


def lookup_rescaled_orbitals(label, basis_rescaled_orbitals):
    label_no_nums = label
    while label_no_nums not in basis_rescaled_orbitals:
        try:
            int(label_no_nums[-1])
        except ValueError:
            raise Exception(
                "Element not defined in basis_rescaled_orbitals:", label, label_no_nums
            )
        label_no_nums = label_no_nums[:-1]
    return basis_rescaled_orbitals[label_no_nums]


def rescale_basis(initial_basis, rescaling_dictionnary, basis_rescaled_orbitals=None):
    if basis_rescaled_orbitals is None:
        basis_rescaled_orbitals = default_basis_rescaled_orbitals(initial_basis)
    for el, basis_list in initial_basis.items():
        rescalings = rescaling_dictionnary[el]
        orbitals_of_interest_lists = lookup_rescaled_orbitals(el, basis_rescaled_orbitals)
        resc_id = 0
        for orbitals_of_interest_list in orbitals_of_interest_lists:
            if not orbitals_of_interest_list:
                continue
            for orb_id in orbitals_of_interest_list:
                orb_def_list = basis_list[orb_id]
                nctr = len(orb_def_list) - 1
                for ctr_id in range(nctr):
                    orb_def_list[ctr_id + 1][0] *= rescalings[resc_id]
            resc_id += 1


def rescale_mol_basis_from_elements(
    mol, rescaling_dictionnary, init_basis=None, basis_rescaled_orbitals=None
):
    if init_basis is None:
        old_basis = mol._basis
    else:
        old_basis = deepcopy(init_basis)
    rescale_basis(
        old_basis, rescaling_dictionnary, basis_rescaled_orbitals=basis_rescaled_orbitals
    )
    mol.basis = old_basis
    mol.build()


def rescale_mol_basis_from_array(
    rescalings_array, mol, rescaling_atom_mapping, init_basis=None, basis_rescaled_orbitals=None
):
    resc_dict = {}
    for el, resc_bounds in rescaling_atom_mapping.items():
        resc_dict[el] = rescalings_array[resc_bounds[0] : resc_bounds[1]]
    rescale_mol_basis_from_elements(
        mol, resc_dict, init_basis=init_basis, basis_rescaled_orbitals=basis_rescaled_orbitals
    )


def S_G_h1(
    rescalings_array, mol, rescaling_atom_mapping, init_basis=None, basis_rescaled_orbitals=None
):
    rescale_mol_basis_from_array(
        rescalings_array,
        mol,
        rescaling_atom_mapping,
        init_basis=init_basis,
        basis_rescaled_orbitals=basis_rescaled_orbitals,
    )
    S = mol.intor("int1e_ovlp")
    # K.Karan:@Danish: what is 8?
    G = mol.intor("int2e", aosym="8")
    h1 = mol.intor("int1e_nuc") + mol.intor("int1e_kin")
    return S, G, h1


def energy_1d_gradient_wrt_rescaling(
    mol,
    cur_rescalings_array,
    diff_id,
    rescaling_atom_mapping,
    P,
    W,
    fd_coeffs,
    rel_changes,
    init_basis,
    basis_rescaled_orbitals=None,
):
    grad_el = 0.0

    # Evaluate some necessary components via finite difference
    dS = None
    dG = None
    dh1 = None

    backup_el_val = cur_rescalings_array[diff_id]
    for fd_coeff, rel_change in zip(fd_coeffs, rel_changes):
        cur_rescalings_array[diff_id] = backup_el_val * rel_change
        step_S, step_G, step_h1 = S_G_h1(
            cur_rescalings_array,
            mol,
            rescaling_atom_mapping,
            init_basis,
            basis_rescaled_orbitals=basis_rescaled_orbitals,
        )
        if dS is None:
            dS = np.zeros_like(step_S)
            dG = np.zeros_like(step_G)
            dh1 = np.zeros_like(step_h1)
        dS += step_S * fd_coeff
        dG += step_G * fd_coeff
        dh1 += step_h1 * fd_coeff
    cur_rescalings_array[diff_id] = backup_el_val

    uJ, uK = scf._vhf.incore(dG, P)
    grad_el = (
        np.einsum("ij,ij", P, dh1)
        + 0.5 * np.einsum("ij,ij", P, uJ)
        - np.einsum("ij,ji", P, uK) / 4
        - np.einsum("ij,ij", W, dS)
    )

    return grad_el


def energy_wgradient_wrt_rescaling(
    log_rescalings_array,
    mol,
    rescaling_atom_mapping,
    init_basis=None,
    fd_log_step=1.0e-6,
    pn=1,
    pyscf_calc_params=OML_pyscf_calc_params(),
    basis_rescaled_orbitals=None,
):
    # Initialize finite difference steps in log space.
    fd_coeffs, grid_pos = grid_finite_difference_coefficients_1var(pn, 1, fd_log_step)
    grid_mult = np.exp(grid_pos * fd_log_step)

    if init_basis is None:
        init_basis = deepcopy(mol._basis)
    rescalings_array = np.exp(log_rescalings_array)
    rescale_mol_basis_from_array(
        rescalings_array,
        mol,
        rescaling_atom_mapping,
        init_basis=init_basis,
        basis_rescaled_orbitals=basis_rescaled_orbitals,
    )

    nresc = rescalings_array.shape[0]
    grad = np.zeros((nresc,))

    # K.Karan: originally thought to unify it with OML_Compound.generate_pyscf_mf, decided it's not worth it.

    mf = mol.HF()
    mf = converged_mf(mf, pyscf_calc_params=pyscf_calc_params)

    e0 = mf.e_tot
    P = mf.make_rdm1()
    C = mf.mo_coeff
    O = mf.mo_occ
    e = mf.mo_energy
    W = make_rdm1e(e, C, O)

    temp_rescalings = np.copy(rescalings_array)
    for i in range(nresc):
        grad[i] = energy_1d_gradient_wrt_rescaling(
            mol,
            temp_rescalings,
            i,
            rescaling_atom_mapping,
            P,
            W,
            fd_coeffs,
            grid_mult,
            init_basis,
            basis_rescaled_orbitals=basis_rescaled_orbitals,
        )
    return e0, grad


def make_atoms_distinguishable(mol):
    el_counter = {}
    new_atoms = []
    for el, (_, coords) in zip(mol.elements, mol.atom):
        if el in el_counter:
            el_counter[el] += 1
        else:
            el_counter[el] = 1
        new_atoms.append((el + str(el_counter[el]), coords))
    mol.atom = new_atoms
    mol.build()


def get_rescaling_atom_mapping_nrescs(label_list, basis_rescaled_orbitals):
    lb = 0
    output = {}
    for label in label_list:
        nrescs = len(lookup_rescaled_orbitals(label, basis_rescaled_orbitals))
        ub = lb + nrescs
        output[label] = (lb, ub)
        lb = ub
    return output, ub


def get_label_list(mol):
    return [t[0] for t in mol.atom]


def optimize_molecule_basis_rescalings(
    init_mol,
    pyscf_calc_params=OML_pyscf_calc_params(),
    opt_rescaling_guess=None,
    atoms_distinguishable=False,
    tolerance=1e-9,
    fd_log_step=1.0e-6,
    pn=1,
    return_energy=False,
    method="L-BFGS-B",
    basis_rescaled_orbitals=None,
):
    mol = deepcopy(init_mol)
    if basis_rescaled_orbitals is None:
        basis_rescaled_orbitals = default_basis_rescaled_orbitals(init_mol._basis)
    if not atoms_distinguishable:
        make_atoms_distinguishable(mol)
    llist = get_label_list(mol)
    rescaling_atom_mapping, nrecs = get_rescaling_atom_mapping_nrescs(
        llist, basis_rescaled_orbitals
    )
    init_log_rescalings = np.zeros(nrecs)
    if opt_rescaling_guess is not None:
        for atom_id, atom_label in enumerate(llist):
            bound_tuple = rescaling_atom_mapping[atom_label]
            init_log_rescalings[bound_tuple[0] : bound_tuple[1]] = np.log(
                opt_rescaling_guess[atom_id]
            )

    init_basis = deepcopy(mol._basis)
    optimization_result = spo.minimize(
        energy_wgradient_wrt_rescaling,
        init_log_rescalings,
        args=(
            mol,
            rescaling_atom_mapping,
            init_basis,
            fd_log_step,
            pn,
            pyscf_calc_params,
            basis_rescaled_orbitals,
        ),
        method=method,
        jac=True,
        options={"disp": display_scipy_convergence},
        tol=tolerance,
    )

    optimized_rescalings = np.exp(optimization_result.x)
    atom_rescalings = [None for _ in llist]
    for atom_id, atom_label in enumerate(llist):
        bound_tuple = rescaling_atom_mapping[atom_label]
        atom_rescalings[atom_id] = np.copy(optimized_rescalings[bound_tuple[0] : bound_tuple[1]])
    if return_energy:
        return atom_rescalings, optimization_result.fun
    else:
        return atom_rescalings


def rescale_mol_basis(mol, rescaling_list, init_basis=None, basis_rescaled_orbitals=None):
    make_atoms_distinguishable(mol)
    rescaling_label_dictionnary = {}
    for atom_id, atom_label in enumerate(get_label_list(mol)):
        rescaling_label_dictionnary[atom_label] = rescaling_list[atom_id]
    rescale_mol_basis_from_elements(
        mol,
        rescaling_label_dictionnary,
        init_basis=init_basis,
        basis_rescaled_orbitals=basis_rescaled_orbitals,
    )
