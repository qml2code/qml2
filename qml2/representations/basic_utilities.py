from ..jit_interfaces import (
    all_,
    any_,
    bool_,
    constr_dfloat_,
    constr_dint_,
    constr_int_,
    copy_,
    dbool_,
    dint_,
    dtype_,
    empty_,
    float_,
    floor_,
    int_,
    jit_,
    l2_norm_,
    max_,
    min_,
    ndarray_,
    ones_,
    prange_,
    repeat_,
    save_,
    searchsorted_,
    sqrt_,
    sum_,
    zeros_,
)
from ..utils import get_atom_environment_ranges, l2_norm_sq_dist


# For introducing periodic boundary conditions.
@jit_
def generate_cell_int_coords(
    cell_int_coords: ndarray_,
    nExtend: ndarray_,
    ndim: int_,
    is_on_lower_border: ndarray_,
    is_on_upper_border: ndarray_,
    started: bool_,
):
    if started:
        cell_int_coords[:] = -nExtend[:]
        is_on_lower_border[:] = True
        is_on_upper_border[:] = False
        return True
    for dim in range(ndim):
        if cell_int_coords[dim] == nExtend[dim]:
            cell_int_coords[dim] = -nExtend[dim]
            is_on_lower_border[dim] = True
            is_on_upper_border[dim] = False
        else:
            cell_int_coords[dim] += 1
            is_on_lower_border[dim] = False
            if cell_int_coords[dim] == nExtend[dim]:
                is_on_upper_border[dim] = True
            return True
    return False


@jit_
def should_be_added(
    atom_coords: ndarray_,
    cell_dir_lower_reach: ndarray_,
    cell_dir_upper_reach: ndarray_,
    is_on_lower_border: ndarray_,
    is_on_upper_border: ndarray_,
    ndim: int_,
):
    if not (any_(is_on_lower_border) or any_(is_on_upper_border)):
        return True
    for dim in range(ndim):
        if is_on_lower_border[dim] and (atom_coords[dim] > cell_dir_upper_reach[dim]):
            return True
        if is_on_upper_border[dim] and (atom_coords[dim] < cell_dir_lower_reach[dim]):
            return True
    return False


@jit_(numba_parallel=True)
def count_atom_copies(
    cell_dir_coordinates: ndarray_,
    cell_dir_lower_reach: ndarray_,
    cell_dir_upper_reach: ndarray_,
    natoms: int_,
    nExtend: ndarray_,
    ndim: int_,
    dint_: dtype_ = dint_,
    dbool_: dtype_ = dbool_,
):
    # Number of created cells not on the supercell's border and thus containing copies of all atoms.
    num_inside_cells = 1
    for nex in nExtend:
        num_inside_cells *= 2 * constr_int_(nex) - 1
    # Not counting the original cell.
    num_inside_cells -= 1
    # We know that at least num_inside_cells copies of atoms are added.
    atom_additional_copy_number = repeat_(constr_dint_(num_inside_cells), natoms)
    for atom_id in prange_(natoms):
        atom_coords = cell_dir_coordinates[atom_id]
        cell_int_coords = empty_((ndim,), dtype=dint_)
        is_on_lower_border = empty_((ndim,), dtype=dbool_)
        is_on_upper_border = empty_((ndim,), dtype=dbool_)
        started = True
        while generate_cell_int_coords(
            cell_int_coords, nExtend, ndim, is_on_lower_border, is_on_upper_border, started
        ):
            started = False
            if not (any_(is_on_lower_border) or any_(is_on_upper_border)):
                continue
            if should_be_added(
                atom_coords,
                cell_dir_lower_reach,
                cell_dir_upper_reach,
                is_on_lower_border,
                is_on_upper_border,
                ndim,
            ):
                atom_additional_copy_number[atom_id] += 1

    return atom_additional_copy_number


@jit_(numba_parallel=True)
def extend_for_pbc(
    coordinates: ndarray_,
    element_types: ndarray_,
    natoms: int_,
    rcut: float_,
    cell: ndarray_,
    dint_: dtype_ = dint_,
    dbool_: dtype_ = dbool_,
) -> tuple[ndarray_, ndarray_, int_, ndarray_]:
    # Cartesian space dimensionality.
    ndim = coordinates.shape[1]
    # Normalized directions corresponding to different cells.
    cell_directions = copy_(cell)
    cell_lengths = empty_((ndim,))
    for dim in range(ndim):
        cell_lengths[dim] = l2_norm_(cell_directions[dim])
        cell_directions[dim] /= cell_lengths[dim]

    # Transform coordinates to cell.
    cell_dir_coordinates = coordinates @ cell_directions.T
    # maximum and minimum cell coordinates of atoms that can be reached from atoms in neighboring cells.
    cell_dir_lower_reach = empty_((ndim,))
    cell_dir_upper_reach = empty_((ndim,))
    # how many cells in different directions we need to create
    nExtend = empty_((ndim,), dtype=dint_)
    for dim in range(ndim):
        cell_dir_lower_reach[dim] = max_(cell_dir_coordinates[:, dim]) + rcut - cell_lengths[dim]
        cell_dir_upper_reach[dim] = min_(cell_dir_coordinates[:, dim]) - rcut + cell_lengths[dim]
        nExtend[dim] = floor_(rcut / cell_lengths[dim]) + 1
    # how many copies of each atom do we need to make
    atom_additional_copy_number = count_atom_copies(
        cell_dir_coordinates, cell_dir_lower_reach, cell_dir_upper_reach, natoms, nExtend, ndim
    )
    # total number of atoms
    natoms_tot = natoms + constr_int_(sum_(atom_additional_copy_number))
    # extended arrays
    extended_coordinates = empty_((natoms_tot, ndim))
    extended_element_types = empty_((natoms_tot,), dtype=dint_)

    # the oridinal cell is just copied.
    extended_coordinates[:natoms, :] = coordinates[:, :]
    extended_element_types[:natoms] = element_types[:]

    atom_copy_start_ids = get_atom_environment_ranges(atom_additional_copy_number) + natoms

    for atom_id in prange_(natoms):
        atom_coords = coordinates[atom_id]
        cell_atom_coords = cell_dir_coordinates[atom_id]
        cur_copy_id = atom_copy_start_ids[atom_id]
        extended_element_types[cur_copy_id : atom_copy_start_ids[atom_id + 1]] = element_types[
            atom_id
        ]
        # create new coordinates
        cell_int_coords = empty_((ndim,), dtype=dint_)
        is_on_lower_border = empty_((ndim,), dtype=dbool_)
        is_on_upper_border = empty_((ndim,), dtype=dbool_)
        started = True
        while generate_cell_int_coords(
            cell_int_coords, nExtend, ndim, is_on_lower_border, is_on_upper_border, started
        ):
            started = False
            if all_(cell_int_coords == 0):
                continue
            if should_be_added(
                cell_atom_coords,
                cell_dir_lower_reach,
                cell_dir_upper_reach,
                is_on_lower_border,
                is_on_upper_border,
                ndim,
            ):
                extended_coordinates[cur_copy_id, :] = atom_coords
                for dim in range(ndim):
                    extended_coordinates[cur_copy_id, :] += cell_int_coords[dim] * cell[dim]
                cur_copy_id += 1
    return extended_coordinates, extended_element_types, int(natoms_tot), atom_copy_start_ids


# For creating sparse matrices of functions of distance.
@jit_
def box_bounds(atom_coords: ndarray_, rcut: float_):
    return atom_coords - rcut, atom_coords + rcut


@jit_
def is_inside_box(atom_coords: ndarray_, box_lcoords: ndarray_, box_ucoords: ndarray_):
    return all_(atom_coords > box_lcoords) and all_(atom_coords < box_ucoords)


@jit_(numba_parallel=True)
def get_rough_num_neighbors(
    coordinates: ndarray_, rcut: float_, natoms: int_, dint_: dtype_ = dint_
):
    """
    (Rough) estimates of numbers of atoms close enough to a given atom. Uses boxes instead of spheres for computational efficiency.
    """
    all_num_neighbors = -ones_(
        (natoms,), dtype=dint_
    )  # KK: making sure self is skipped not worth the hassle.
    for atom_id in prange_(natoms):
        box_lcoords, box_ucoords = box_bounds(coordinates[atom_id], rcut)
        for other_coords in coordinates:
            if is_inside_box(other_coords, box_lcoords, box_ucoords):
                all_num_neighbors[atom_id] += 1
    return all_num_neighbors


@jit_
def any_is_inside_box(all_atom_coords: ndarray_, box_lcoords: ndarray_, box_ucoords: ndarray_):
    for atom_coords in all_atom_coords:
        if is_inside_box(atom_coords, box_lcoords, box_ucoords):
            return True
    return False


@jit_
def get_neighbor_id(i: int_, j: int_, neighbor_ids: ndarray_, neighbor_nums: ndarray_) -> int_:
    j_id = searchsorted_(neighbor_ids[i, : neighbor_nums[i]], j)
    return int(j_id)


@jit_
def get_from_sparse_matrix(
    matrix: ndarray_,
    i: int_,
    j: int_,
    neighbor_ids: ndarray_,
    neighbor_nums: ndarray_,
    default_val: float_ = 0.0,
):
    j_id = get_neighbor_id(i, j, neighbor_ids, neighbor_nums)
    if neighbor_ids[i, j_id] == j:
        return matrix[i, j_id]
    else:  # j must've been on the edge of cutoff
        return constr_dfloat_(default_val)


@jit_
def symmetrize_neighbor_ids(neighbor_ids: ndarray_, num_neighbors: ndarray_, natoms: int_):
    for i in range(natoms):
        i_num_neighbors = int(num_neighbors[i])
        for j in neighbor_ids[i, :i_num_neighbors]:
            neighbor_ids[int(j), int(num_neighbors[j])] = i
            num_neighbors[int(j)] += 1


@jit_(numba_parallel=True)
def symmetrize_sparse_matrix(
    matrix: ndarray_, neighbor_ids: ndarray_, num_neighbors: ndarray_, natoms: int_
):
    matrix_saved = save_(matrix)
    for i in prange_(natoms):
        for j_id in range(int(num_neighbors[i])):
            j = neighbor_ids[i, j_id]
            if j > i:
                break
            other_i_id = get_neighbor_id(j, i, neighbor_ids, num_neighbors)
            matrix[j, other_i_id] = matrix_saved[i, j_id]


@jit_(numba_parallel=True)
def calculate_distances_wrcut(
    coordinates: ndarray_, natoms: int_, max_rcut: float_, dint_: dtype_ = dint_
):
    # First estimate how much space should be allocated for those distances and neighbor indices.
    rough_num_neighbors = get_rough_num_neighbors(coordinates, max_rcut, natoms, dint_=dint_)
    rough_max_num_neighbors = int(max_(rough_num_neighbors))
    num_neighbors = zeros_((natoms,), dtype=dint_)
    neighbor_ids = empty_((natoms, rough_max_num_neighbors), dtype=dint_)
    distances = empty_((natoms, rough_max_num_neighbors))
    # Calculate half of the distance matrix.
    max_sqrcut = max_rcut**2
    for i in prange_(natoms):
        atom_coords = coordinates[i]
        box_lcoords, box_ucoords = box_bounds(coordinates[i], max_rcut)
        found_neighbors = 0
        for j in range(i):
            # Quickly check it's inside the box.
            if not is_inside_box(coordinates[j], box_lcoords, box_ucoords):
                continue
            # Square distance.
            sq_dist = l2_norm_sq_dist(atom_coords, coordinates[j])
            if sq_dist >= max_sqrcut:
                continue
            dist = sqrt_(sq_dist)
            distances[i, found_neighbors] = dist
            neighbor_ids[i, found_neighbors] = j
            found_neighbors += 1
        num_neighbors[i] = found_neighbors
    # Symmetrize distance matrix.
    symmetrize_neighbor_ids(neighbor_ids, num_neighbors, natoms)
    symmetrize_sparse_matrix(distances, neighbor_ids, num_neighbors, natoms)
    return distances, neighbor_ids, num_neighbors
