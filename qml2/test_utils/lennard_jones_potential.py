from ..data import nCartDim
from ..jit_interfaces import (
    array_,
    dim0float_array_,
    dint_,
    dtype_,
    empty_,
    int_,
    jit_,
    l2_norm_,
    randint_,
    random_,
    standard_normal_,
    zeros_,
)
from ..utils import l2_norm_sq_dist


@jit_
def lj_energy_force(nuclear_charges, coordinates):
    energy = 0.0
    force = zeros_(coordinates.shape)
    natoms = nuclear_charges.shape[0]
    for atom_id in range(natoms):
        atom_coords = coordinates[atom_id]
        for other_atom_id in range(atom_id):
            other_atom_coords = coordinates[other_atom_id]
            inv_d2 = 1.0 / l2_norm_sq_dist(atom_coords, other_atom_coords)
            prefac = nuclear_charges[atom_id] * nuclear_charges[other_atom_id]
            energy += (inv_d2**6 - inv_d2**3) * prefac
            radial_force = (
                (atom_coords - other_atom_coords) * (12 * inv_d2**7 - 6 * inv_d2**4) * prefac
            )
            force[atom_id] += radial_force
            force[other_atom_id] -= radial_force
    return energy, force


@jit_
def too_close(coord, other_coord_arr, min_dist):
    min_dist_sq = min_dist**2
    for other_coord in other_coord_arr:
        if l2_norm_sq_dist(coord, other_coord) < min_dist_sq:
            return True
    return False


@jit_
def rand_unit_sphere(nCartDim: int_ = nCartDim):
    vec = standard_normal_((nCartDim,))
    vec /= l2_norm_(vec)
    return vec


@jit_
def random_lj_configuration(
    natoms: int_,
    min_dist: dim0float_array_ = array_(0.75),
    max_dist: dim0float_array_ = array_(1.5),
    nCartDim: int_ = nCartDim,
):
    """
    Generate random natoms LJ particles not closer to each other than min dist.
    """
    output = zeros_((natoms, nCartDim))
    for i in range(1, natoms):
        while too_close(output[i], output[:i], min_dist):
            seed_id = int(random_() * i)
            output[i] = output[seed_id] + rand_unit_sphere() * (
                min_dist + (max_dist - min_dist) * random_()
            )
    return output


@jit_
def random_lj_molecule(
    natoms_min: int_,
    natoms_max: int_,
    possible_nuclear_charges=array_([1, 3]),
    min_dist: dim0float_array_ = array_(0.75),
    max_dist: dim0float_array_ = array_(1.5),
    dint_: dtype_ = dint_,
):
    nelements = len(possible_nuclear_charges)
    natoms = int(randint_(natoms_min, natoms_max + 1))
    element_ids = randint_(0, nelements, (natoms,))
    nuclear_charges = empty_((natoms,), dtype=dint_)
    for atom_id in range(natoms):
        nuclear_charges[atom_id] = possible_nuclear_charges[element_ids[atom_id]]
    return nuclear_charges, random_lj_configuration(natoms, min_dist=min_dist, max_dist=max_dist)
