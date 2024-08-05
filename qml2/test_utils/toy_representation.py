# A toy representation simply assigning distances to other atoms. Introduced for convenient testing of procedures calculating force
# with 2body-only potentials (like Lennard-Jones).
from ..data import nCartDim
from ..jit_interfaces import dint_, dtype_, empty_, inf_, int_, jit_, zeros_
from ..representations.basic_utilities import calculate_distances_wrcut


@jit_
def all_atoms_relevant(natoms: int_, max_natoms: int_, dint_: dtype_ = dint_):
    output = empty_((natoms, max_natoms), dtype=dint_)
    for i in range(natoms):
        output[:, i] = i
    return output


# @jit_
def generate_toy_representation_with_gradients(
    nuclear_charges, coordinates, max_natoms=None, nCartDim: int_ = nCartDim, inf_=inf_
):
    """
    Generates an array consisting of nuclear charge and inverse distance values for neighbors of each atoms,
    along with the corresponding derivatives. Designed for testing purposes.

    Output:
        representations - natoms x repsize array of representation vectors
        num_neighbors - natoms integer array - for each "central" atom contains total number of "neighbor" atoms,
                        defined such that derivative of central atom's representation w.r.t. their positions
                        can be non-zero.
        rel_neighbor_list - natoms x max_num_neighbors integer array, for each atom `i` contains indices
                        of its neighbor atoms (including index of the "central atom" itself) in the first
                        `num_neighbors[i]` entries of `rel_neighbor_list[i]`.
        representation_gradients - natoms x repsize x max_num_neighbors x CartDim - for a central atom with index `i`,
                        a neighbor atom with index `j`, and representation
                        component `k`, `representation_gradients[i, k, j, :]` contains derivative of `representations[i, k]`
                        w.r.t. `coordinates[rel_neighbor_list[j], :]`

    """
    natoms = nuclear_charges.shape[0]
    if max_natoms is None:
        max_natoms = natoms
    rep_length = int((max_natoms - 1) * 2)
    representations = zeros_((natoms, rep_length))
    representation_gradients = zeros_((natoms, rep_length, max_natoms, nCartDim))
    rel_neighbor_list = all_atoms_relevant(natoms, max_natoms)  # every atom is relevant
    dist_mat, neighbor_ids, num_neighbors = calculate_distances_wrcut(
        coordinates, natoms, max_rcut=inf_
    )
    for i_atom, (dist_mat_row, neighbor_ids_row, num_neigh) in enumerate(
        zip(dist_mat, neighbor_ids, num_neighbors)
    ):
        sorted_tuples = []
        for d, i in zip(dist_mat_row[:num_neigh], neighbor_ids_row[:num_neigh]):
            sorted_tuples.append((nuclear_charges[i], 1.0 / d, i))
        sorted_tuples.sort()
        for st_id, st in enumerate(sorted_tuples):
            inv_d = st[1]
            other_atom_id = st[2]
            representations[i_atom, st_id * 2] = inv_d
            representations[i_atom, st_id * 2 + 1] = st[0]
            radial_der = -(coordinates[i_atom] - coordinates[other_atom_id]) * inv_d**3
            representation_gradients[i_atom, st_id * 2, i_atom] = radial_der
            representation_gradients[i_atom, st_id * 2, other_atom_id] = -radial_der
    num_neighbors += 1  # need to count atom itself as neighbor
    return representations, representation_gradients, rel_neighbor_list, num_neighbors


def generate_toy_representation(nuclear_charges, coordinates, max_natoms=None):
    wgrads = generate_toy_representation_with_gradients(
        nuclear_charges, coordinates, max_natoms=max_natoms
    )
    return wgrads[0]
