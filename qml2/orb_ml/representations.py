from ..jit_interfaces import (
    append_,
    array_,
    bool_,
    int_,
    jit_,
    matmul_,
    ones_,
    sqrt_,
    sum_,
    where_,
    zeros_,
)
from ..utils import weighted_array

neglect_orb_occ = 0.1


# Some auxiliary numba functions.
@jit_
def add_aos_coupling(
    mat,
    coeffs,
    angular_momenta,
    ang_mom2: int_,
    lower_index: int_,
    upper_index: int_,
    ao1: int_,
    cur_coupling_val,
):
    for ao2 in range(lower_index, upper_index):
        if angular_momenta[ao2] == ang_mom2:
            # mat[ao1, ao2], not mat[ao2, ao1]: ao1 is the AO on the single atom, which by
            # convention is the BRA, so this is <c_a|O|c_partner>. Identical for the
            # Hermitian coupling matrices used so far; the distinction only bites for a
            # non-Hermitian operator, where the two orderings are genuinely different.
            cur_coupling_val[:] += mat[ao1, ao2] * coeffs[ao1] * coeffs[ao2]


@jit_
def add_orb_atom_coupling(
    mat,
    coeffs,
    angular_momenta,
    atom_ao_range,
    same_atom: int_,
    ang_mom1: int_,
    ang_mom2: int_,
    cur_coupling_val,
):
    ao_lb = int(atom_ao_range[0])
    ao_ub = int(atom_ao_range[1])
    for ao1 in range(ao_lb, ao_ub):
        if angular_momenta[ao1] == ang_mom1:
            if same_atom == 0:
                add_aos_coupling(
                    mat,
                    coeffs,
                    angular_momenta,
                    ang_mom2,
                    ao_lb,
                    ao_ub,
                    ao1,
                    cur_coupling_val,
                )
            else:
                add_aos_coupling(
                    mat,
                    coeffs,
                    angular_momenta,
                    ang_mom2,
                    0,
                    ao_lb,
                    ao1,
                    cur_coupling_val,
                )
                add_aos_coupling(
                    mat,
                    coeffs,
                    angular_momenta,
                    ang_mom2,
                    ao_ub,
                    coeffs.shape[0],
                    ao1,
                    cur_coupling_val,
                )


@jit_(numba_cache=True)
def gen_orb_atom_scalar_rep(
    coup_mats,
    coeffs,
    angular_momenta,
    atom_ao_range,
    max_ang_mom: int_,
    scalar_reps,
    only_other_couplings,
):
    cur_array_position = 0
    num_coup_mats = coup_mats.shape[0]
    scalar_reps[:] = 0.0

    for mat_counter in range(num_coup_mats):
        for same_atom_check in range(2):
            # same_atom_check == 0 is the self block; a matrix flagged as offering only
            # "other" couplings has no self slot at all (its self term is structurally zero,
            # and a constant-zero column would break per-feature variance normalization).
            if only_other_couplings[mat_counter] and (same_atom_check == 0):
                continue
            for ang_mom1 in range(max_ang_mom + 1):
                for ang_mom2 in range(max_ang_mom + 1):
                    # The self block is symmetric in (ang_mom1, ang_mom2) -- for the shifted
                    # dipole too, since the subtracted R_A(mu) is constant across it -- so only
                    # the upper triangle is stored. The "other" block is directed and stored full.
                    if (same_atom_check == 0) and (ang_mom1 > ang_mom2):
                        continue
                    add_orb_atom_coupling(
                        coup_mats[mat_counter],
                        coeffs,
                        angular_momenta,
                        atom_ao_range,
                        same_atom_check,
                        ang_mom1,
                        ang_mom2,
                        scalar_reps[cur_array_position : cur_array_position + 1],
                    )
                    cur_array_position = cur_array_position + 1


@jit_(numba_cache=True)
def ang_mom_descr(
    ovlp_mat, coeffs, angular_momenta, atom_ao_range, max_ang_mom: int_, scalar_reps, rho_val
):
    scalar_reps[:] = 0.0
    for ang_mom in range(1, max_ang_mom + 1):
        add_orb_atom_coupling(
            ovlp_mat,
            coeffs,
            angular_momenta,
            atom_ao_range,
            0,
            ang_mom,
            ang_mom,
            scalar_reps[ang_mom - 1 : ang_mom],
        )
    rho_val[0] = sum_(scalar_reps)
    add_orb_atom_coupling(ovlp_mat, coeffs, angular_momenta, atom_ao_range, 0, 0, 0, rho_val)


#   Parameters defining how orbitals are represented.


class OML_rep_params:
    def __init__(
        self,
        orb_atom_rho_comp=0.95,
        max_angular_momentum=3,
        atom_sorted_pseudo_orbs=False,
        fjd_coup_mats=False,
    ):
        self.orb_atom_rho_comp = orb_atom_rho_comp
        self.max_angular_momentum = max_angular_momentum

        self.atom_sorted_pseudo_orbs = atom_sorted_pseudo_orbs
        self.fjd_coup_mats = fjd_coup_mats

    def __str__(self):
        str_out = (
            "orb_atom_rho_comp:"
            + str(self.orb_atom_rho_comp)
            + ",max_ang_mom:"
            + str(self.max_angular_momentum)
        )
        return str_out


def reconstr_mat(inv_mo_coeff, mo_energy=None, mo_occ=None, mat_type="ovlp"):
    norbs = inv_mo_coeff.shape[1]
    match mat_type:
        case "density":
            # NOTE: only using 1. and 0. to make representation invariant w.r.t. UHF<->RHF change.
            mo_arr = zeros_(norbs)
            for orb_id in range(norbs):
                if mo_occ[orb_id] > neglect_orb_occ:
                    mo_arr[orb_id] = 1.0
        case "Fock":
            mo_arr = mo_energy
        case "ovlp":
            mo_arr = ones_(norbs)
        case _:
            raise Exception("Unknown reconstructed matrix type.")
    return matmul_(inv_mo_coeff * mo_arr, inv_mo_coeff.T)


def bra_recentred_dipole_matrices(dipole_mats, ovlp_mat, atom_coords, ao_atom_ids):
    """The dipole operator measured from the atom owning the BRA index.

    ``Dtilde[gamma, mu, nu] = <mu|r_gamma|nu> - R[A(mu), gamma] * S[mu, nu]``

    i.e. each atomic contribution is measured against its own nucleus, the CAMM
    convention, which is what makes the resulting couplings translationally invariant.
    Recentring on the bra also makes the operator non-Hermitian, so the ordering matters:
    couplings must be formed as <single atom|O|partner>, never the reverse.

    Args:
        dipole_mats: ``(3, nao, nao)`` dipole integrals about the GLOBAL origin.
        ovlp_mat: ``(nao, nao)`` overlap, in the same AO convention.
        atom_coords: ``(natoms, 3)`` nuclear coordinates, Bohr.
        ao_atom_ids: ``(nao,)`` index of the atom owning each AO.
    """
    shifts = atom_coords[ao_atom_ids]  # (nao, 3), the bra atom's position per AO
    return array_([dipole_mats[gamma] - shifts[:, gamma, None] * ovlp_mat for gamma in range(3)])


def bond_projected_dipole_matrix(recentred_dipole_mats, atom_coords, ao_atom_ids):
    """The bra-recentred dipole projected onto each AO pair's own bond axis.

    ``M[mu, nu] = sum_gamma Dtilde[gamma, mu, nu] * u[mu, nu, gamma]``, with ``u`` the UNIT
    vector from the bra atom to the ket atom. The unit vector, not the raw separation: the
    raw form scales as the bond length squared and would up-weight distant atoms in the
    union sum, backwards from chemical locality.

    Intra-atomic blocks (``A(mu) == A(nu)``) carry no bond direction and would divide by
    zero, so they are set to zero explicitly here. That is an AO-level guard, distinct from
    the orbital-level fact that this channel's self coupling vanishes identically -- which
    is handled separately, by dropping its self slot from the feature vector.
    """
    separations = atom_coords[ao_atom_ids][None, :, :] - atom_coords[ao_atom_ids][:, None, :]
    distances = sqrt_(sum_(separations**2, axis=2))
    same_atom = ao_atom_ids[:, None] == ao_atom_ids[None, :]
    # Guard the division itself, not just its result: a zero distance would otherwise raise
    # or produce a NaN that no later masking can be trusted to remove.
    safe_distances = where_(same_atom, 1.0, distances)
    output = zeros_(distances.shape)
    for gamma in range(3):
        output += recentred_dipole_mats[gamma] * separations[:, :, gamma] / safe_distances
    return where_(same_atom, 0.0, output)


def dipole_coupling_matrices(dipole_mats, ovlp_mat, atom_coords, ao_atom_ids):
    """The D block of FJD: three shifted-dipole components plus the bond projection.

    Returned in the order the coupling machinery consumes them, ``(Dx, Dy, Dz, M)``. The
    first three are one vector operator whose per-orbital couplings are reduced to a scalar
    by Euclidean norm (``l2_norm_merges``); the last is a scalar operator with no self
    coupling (``only_other_couplings``). They are kept as separate channels rather than one
    concatenated block because their self terms behave differently and because either must
    be ablatable on its own.
    """
    recentred = bra_recentred_dipole_matrices(dipole_mats, ovlp_mat, atom_coords, ao_atom_ids)
    bond_projected = bond_projected_dipole_matrix(recentred, atom_coords, ao_atom_ids)
    return (*recentred, bond_projected)


#   Representation of contribution of a single atom to an orb.
class OML_orb_atom_rep:
    def __init__(self, atom_ao_range, coeffs, rep_params, angular_momenta, ovlp_mat):
        self.atom_ao_range = atom_ao_range
        self.scalar_reps = zeros_(rep_params.max_angular_momentum)
        rho_arr = zeros_(1)
        ang_mom_descr(
            ovlp_mat,
            coeffs,
            angular_momenta,
            self.atom_ao_range,
            rep_params.max_angular_momentum,
            self.scalar_reps,
            rho_arr,
        )
        self.rho = rho_arr[0]
        self.pre_renorm_rho = self.rho
        self.atom_id = None

    def completed_scalar_reps(
        self,
        coeffs,
        rep_params: OML_rep_params,
        angular_momenta,
        coup_mats,
        only_other_couplings,
        l2_norm_merges=None,
    ):
        couplings = zeros_(
            scalar_coup_length(
                rep_params, coup_mats, only_other_couplings, l2_norm_merges=l2_norm_merges
            )
        )
        gen_orb_atom_scalar_rep(
            coup_mats,
            coeffs,
            angular_momenta,
            self.atom_ao_range,
            rep_params.max_angular_momentum,
            couplings,
            only_other_couplings,
        )

        if l2_norm_merges is not None:
            pre_l2_merge_couplings = couplings
            couplings = zeros_(
                scalar_coup_length(
                    rep_params,
                    coup_mats,
                    only_other_couplings,
                    l2_norm_merges=l2_norm_merges,
                    pre_l2_merge=False,
                )
            )
            merge_couplings_l2(
                pre_l2_merge_couplings,
                couplings,
                rep_params,
                coup_mats,
                only_other_couplings,
                l2_norm_merges,
            )

        self.scalar_reps = append_(self.scalar_reps, couplings)
        self.scalar_reps /= self.pre_renorm_rho

    def __str__(self):
        return "OML_orb_atom_rep,rho:" + str(self.rho)

    def __repr__(self):
        return str(self)


def coupling_block_bounds(rep_params, coup_mats, only_other_couplings):
    """(start, stop) of each coupling matrix's slice of the pre-merge coupling array.

    The layout is implicit in gen_orb_atom_scalar_rep's loop order: for each matrix in turn,
    the self block (upper triangle only, and absent when the matrix offers only "other"
    couplings) followed by the full "other" block. Kept beside scalar_coup_length because the
    two must agree -- the sum of these lengths IS the pre-merge length.
    """
    nam = num_ang_mom(rep_params)
    self_len = (nam * (nam + 1)) // 2
    other_len = nam**2
    bounds = []
    position = 0
    for mat_id in range(len(coup_mats)):
        length = other_len
        if not only_other_couplings[mat_id]:
            length += self_len
        bounds.append((position, position + length))
        position += length
    return bounds


def merge_couplings_l2(
    couplings, merged_couplings, rep_params, coup_mats, only_other_couplings, l2_norm_merges
):
    """Reduce each group of coupling matrices to one via the slot-wise l2 norm.

    Slot-wise: corresponding (same_atom, ang_mom1, ang_mom2) cells across the group are
    combined, so a vector-valued operator supplied as one matrix per Cartesian component
    collapses to the norm of that vector, per coupling, which is what the FJK feature wants.

    A merged group occupies the position of its first member and the other members are
    dropped, so the output keeps the input's matrix order with each group contracted in
    place. Done before the pre_renorm_rho division purely for neatness: the norm commutes
    with a positive scaling, so the order carries no numerical consequence.
    """
    bounds = coupling_block_bounds(rep_params, coup_mats, only_other_couplings)
    merged_away = set()
    for l2_norm_merge in l2_norm_merges:
        first = l2_norm_merge[0]
        for mat_id in l2_norm_merge[1:]:
            # scalar_coup_length sizes a group from its first member alone, so members
            # disagreeing here would silently mis-size the output.
            assert (
                only_other_couplings[mat_id] == only_other_couplings[first]
            ), "matrices merged by l2 norm must agree on only_other_couplings"
            merged_away.add(mat_id)

    merge_of = {}
    for l2_norm_merge in l2_norm_merges:
        for mat_id in l2_norm_merge:
            merge_of[mat_id] = l2_norm_merge

    position = 0
    for mat_id in range(len(coup_mats)):
        if mat_id in merged_away:
            continue
        start, stop = bounds[mat_id]
        length = stop - start
        if mat_id in merge_of:
            for member in merge_of[mat_id]:
                member_start, member_stop = bounds[member]
                merged_couplings[position : position + length] += (
                    couplings[member_start:member_stop] ** 2
                )
            merged_couplings[position : position + length] = sqrt_(
                merged_couplings[position : position + length]
            )
        else:
            merged_couplings[position : position + length] = couplings[start:stop]
        position += length
    return position


def scalar_coup_length(
    rep_params, coup_mats, only_other_couplings, l2_norm_merges=None, pre_l2_merge=True
):
    nam = num_ang_mom(rep_params)
    total_coupling_num = nam**2 * len(coup_mats) + (nam * (nam + 1)) // 2 * (
        len(coup_mats) - sum_(only_other_couplings)
    )
    if l2_norm_merges is not None and not pre_l2_merge:
        for l2_norm_merge in l2_norm_merges:
            cur_num_couplings = nam**2
            if not only_other_couplings[l2_norm_merge[0]]:
                cur_num_couplings += (nam * (nam + 1)) // 2
            total_coupling_num -= cur_num_couplings * (len(l2_norm_merge) - 1)
    return total_coupling_num


def num_ang_mom(rep_params):
    return rep_params.max_angular_momentum + 1


#   Auxiliary functions.
def scalar_rep_length(oml_comp):
    try:  # if that's a Slater pair list
        first_comp = oml_comp.comps[0]
    except AttributeError:  # if that's a compound list
        first_comp = oml_comp
    return len(first_comp.orb_reps[0].orb_atom_reps[0].scalar_reps)


def component_id_ang_mom_map(rep_params: OML_rep_params):
    """
    What different components of atomic contribution vector representations correspond to.
    """
    output = []
    # Describes the feature vector AS STORED, which for FJD means after the l2 merge and
    # after the self slot of the self-less channel has been dropped -- not the six matrices
    # that went in. Getting this wrong does not mis-label anything quietly: the map is used
    # to index the features, so it reads off the end of them.
    if rep_params.fjd_coup_mats:
        # F, J, the merged shifted-dipole channel, then the bond projection, whose self
        # coupling vanishes identically and carries no self block.
        coup_mat_has_self_block = [True, True, True, False]
    else:
        coup_mat_has_self_block = [True] * 3  # F, J, K; or overlap, F, density
    # Components corresponding to the angular momentum distribution.
    for ang_mom in range(1, num_ang_mom(rep_params)):
        output.append((ang_mom, ang_mom, -1, True))
    for coup_mat_id, has_self_block in enumerate(coup_mat_has_self_block):
        for same_atom in [True, False]:
            if same_atom and not has_self_block:
                continue
            for ang_mom1 in range(num_ang_mom(rep_params)):
                for ang_mom2 in range(num_ang_mom(rep_params)):
                    if not (same_atom and (ang_mom1 > ang_mom2)):
                        output.append((ang_mom1, ang_mom2, coup_mat_id, same_atom))
    return output


def component_id_mat_map(rep_params: OML_rep_params):
    """
    Which coupling matrices different components of atomic contribution vector representations correspond to.
    """
    output = {}
    for comp_id, (_, _, coup_mat_id, _) in enumerate(component_id_ang_mom_map(rep_params)):
        if coup_mat_id not in output:
            output[coup_mat_id] = []
        output[coup_mat_id].append(comp_id)
    return [array_(comp_ids) for comp_ids in output.values()]


# Auxiliary functions
def generate_atom_ao_ranges(mol):
    ao_sliced_with_shells = mol.aoslice_by_atom()
    output = []
    for atom_data in ao_sliced_with_shells:
        output.append(atom_data[2:4])
    return array_(output)


def placeholder_orb_rep(
    rep_params, atom_ids, atom_ao_ranges, angular_momenta, ovlp_mat, coupling_mats
):
    placeholder_orb_coeffs = zeros_(ovlp_mat.shape[0])
    placeholder_orb_coeffs[0] = 1.0
    placeholder_rep = OML_orb_rep(
        placeholder_orb_coeffs,
        rep_params,
        atom_ids,
        atom_ao_ranges,
        angular_momenta,
        ovlp_mat,
        coupling_mats,
    )
    placeholder_rep.virtual = True
    return placeholder_rep


#   Generate an array of orb representations.
def generate_orb_rep_array(
    orb_mat,
    rep_params,
    aos,
    atom_ao_ranges,
    ovlp_mat,
    coupling_mats,
    only_other_couplings=None,
    l2_norm_merges=None,
):
    atom_ids = [ao.atom_id for ao in aos]
    angular_momenta = array_([ao.angular for ao in aos])
    if len(orb_mat) == 0:
        return [
            placeholder_orb_rep(
                rep_params,
                atom_ids,
                atom_ao_ranges,
                angular_momenta,
                ovlp_mat,
                coupling_mats,
            )
        ]
    # It's important that ovlp_mat appears first in this array.
    orb_tmat = orb_mat.T
    output = [
        OML_orb_rep_from_coeffs(
            orb_coeffs,
            rep_params,
            atom_ids,
            atom_ao_ranges,
            angular_momenta,
            ovlp_mat,
            coupling_mats,
            only_other_couplings=only_other_couplings,
            l2_norm_merges=l2_norm_merges,
        )
        for orb_coeffs in orb_tmat
    ]
    return output


def gen_atom_sorted_pseudo_orbs(orb_rep_arr):
    backup_placeholder_arr = []
    atom_sorted_areps = {}
    orb_occ = None
    for orb_rep in orb_rep_arr:
        if orb_rep.virtual:
            backup_placeholder_arr.append(orb_rep)
        else:
            if orb_occ is None:
                orb_occ = orb_rep.rho
            for orb_atom_rep in orb_rep.orb_atom_reps:
                cur_atom_id = orb_atom_rep.atom_id
                if cur_atom_id in atom_sorted_areps:
                    atom_sorted_areps[cur_atom_id].append(orb_atom_rep)
                else:
                    atom_sorted_areps[cur_atom_id] = [orb_atom_rep]
    output = []
    for atom_id in atom_sorted_areps:
        output.append(OML_orb_rep(None, atom_sorted_areps[atom_id]))
        output[-1].rho = orb_occ
    if len(output) == 0:
        return backup_placeholder_arr
    else:
        return output


#   Representation of an orb from atomic contributions.
class OML_orb_rep:
    def __init__(self, orb_coeffs, orb_atom_reps):
        self.rho = 0.0
        self.full_coeffs = orb_coeffs
        self.virtual = False
        self.orb_atom_reps = orb_atom_reps


def OML_orb_rep_from_coeffs(
    orb_coeffs,
    rep_params,
    atom_ids,
    atom_ao_ranges,
    angular_momenta,
    ovlp_mat,
    coup_mats,
    only_other_couplings=None,
    l2_norm_merges=None,
):
    atom_list = []
    prev_atom = -1
    for atom_id in atom_ids:
        if prev_atom != atom_id:
            atom_list.append(atom_id)
            prev_atom = atom_id
    # Each of the resulting groups of AOs is represented with OML_orb_atom_rep object.
    if only_other_couplings is None:
        only_other_couplings = zeros_(len(coup_mats), dtype=bool_)
    orb_atom_reps = []
    for atom_id, atom_ao_range in enumerate(atom_ao_ranges):
        cur_orb_atom_rep = OML_orb_atom_rep(
            atom_ao_range, orb_coeffs, rep_params, angular_momenta, ovlp_mat
        )
        cur_orb_atom_rep.atom_id = atom_id
        orb_atom_reps.append(cur_orb_atom_rep)
    orb_atom_reps = weighted_array(orb_atom_reps)

    # Try to decrease the number of atomic representations, leaving only the most relevant ones.
    orb_atom_reps.normalize_sort_rhos_wcutoff(remaining_rho=rep_params.orb_atom_rho_comp)
    for orb_arep_counter in range(len(orb_atom_reps)):
        orb_atom_reps[orb_arep_counter].completed_scalar_reps(
            orb_coeffs,
            rep_params,
            angular_momenta,
            coup_mats,
            only_other_couplings,
            l2_norm_merges=l2_norm_merges,
        )
    return OML_orb_rep(orb_coeffs, orb_atom_reps)
