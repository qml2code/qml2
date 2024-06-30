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
