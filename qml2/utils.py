# Miscellaneous functions and classes used throughout the code.
# TODO : KK: should be broken down into several components at some point.
from itertools import product
from typing import Union

from .basic_utils import canonical_atomtype, str_atom_corr
from .data import NUCLEAR_CHARGE
from .jit_interfaces import (
    DimsSequenceType_,
    abs_,
    all_tuple_dim_smaller_,
    append_,
    delete_,
    dfloat_,
    dim0int_array_,
    dint_,
    dtype_,
    empty_,
    int_,
    is_scalar_,
    jit_,
    logical_not_,
    ndarray_,
    prange_,
    searchsorted_,
    sort_,
    square_,
    sum_,
    where_,
    zeros_,
)


# Some auxiliary functions.
def np_resize(np_arr, new_size):
    """
    Expand or cut a NumPy array.
    """
    new_arr = np_arr
    for dim_id, new_dim in enumerate(new_size):
        cur_dim = new_arr.shape[dim_id]
        if new_dim is None:
            continue
        if cur_dim < new_dim:
            add_arr_dims = list(new_arr.shape)
            add_arr_dims[dim_id] = new_dim - cur_dim
            add_arr = zeros_(tuple(add_arr_dims))
            new_arr = append_(new_arr, add_arr, dim_id)
        if cur_dim > new_dim:
            new_arr = delete_(new_arr, slice(new_dim, cur_dim), axis=dim_id)
    return new_arr


def all_possible_indices(shape_tuple: DimsSequenceType_):
    if shape_tuple == ():
        return [()]
    else:
        return product(*[range(ubound) for ubound in shape_tuple])


def expanded_shape(arr):
    is_scalar = is_scalar_(arr)
    if is_scalar_(arr):
        shape = ()
    else:
        shape = arr.shape
    return is_scalar, shape


@jit_
def get_atom_environment_ranges(natoms_arr: ndarray_, dint_: dtype_ = dint_):
    nreps = natoms_arr.shape[0]
    ubound_arr = empty_((nreps + 1,), dtype=dint_)
    ubound_arr[0] = 0
    for rep_id in range(nreps):
        ubound_arr[rep_id + 1] = ubound_arr[rep_id] + natoms_arr[rep_id]
    return ubound_arr


# KK: Introduced not for numerical efficiency, but due to Torch
# not liking "nc not in output" in get_sorted_elements.
@jit_
def int_in_sorted(int_val: int, sorted_int_arr: ndarray_):
    i = searchsorted_(sorted_int_arr, int_val)
    if i == sorted_int_arr.shape[0]:
        return False
    return bool(sorted_int_arr[i] == int_val)


@jit_
def get_sorted_elements(ncharges: ndarray_, dint_: dtype_ = dint_):
    output = zeros_((1,), dtype=dint_)
    output[0] = ncharges[0]
    natoms = ncharges.shape[0]
    for i in range(1, natoms):
        nc = int(ncharges[i])
        if not int_in_sorted(nc, output):
            cur_nelements = output.shape[0]
            new_output = zeros_((int(cur_nelements + 1),), dtype=dint_)
            new_output[:cur_nelements] = output[:cur_nelements]
            new_output[-1] = nc
            output = sort_(new_output)
    return output


@jit_
def check_allocation(
    size_tuple: DimsSequenceType_, output: Union[ndarray_, None] = None, dtype: dtype_ = dfloat_
):
    if output is not None:
        if all_tuple_dim_smaller_(size_tuple, output.shape):
            return output
    return empty_(size_tuple, dtype=dtype)


@jit_
def searchsorted_wexception(sorted_int_arr: ndarray_, int_val: dim0int_array_):
    found = searchsorted_(sorted_int_arr, int_val)
    if sorted_int_arr[found] != int_val:
        raise Exception
    return found


@jit_
def get_element_ids_from_sorted(
    ncharges: ndarray_,
    sorted_elements: ndarray_,
    dint_: dtype_ = dint_,
    output: Union[ndarray_, None] = None,
    natoms: Union[int_, None] = None,
):
    output = check_allocation(ncharges.shape, output, dtype=dint_)
    if natoms is None:
        natoms = ncharges.shape[0]
    for i in prange_(natoms):
        output[i] = searchsorted_wexception(sorted_elements, ncharges[i])
    return output


@jit_
def l1_norm_dist(vec1, vec2):
    return sum_(abs_(vec1 - vec2))


@jit_
def l2_sq_norm(vec):
    return sum_(square_(vec))


@jit_
def l2_norm_sq_dist(vec1, vec2):
    return l2_sq_norm(vec1 - vec2)


@jit_
def multiply_transposed(mat, vec):
    for i in prange_(vec.shape[0]):
        mat[i] *= vec[i]


#   XYZ processing.
def check_byte(byte_or_str):
    if isinstance(byte_or_str, str):
        return byte_or_str
    else:
        return byte_or_str.decode("utf-8")


#   For processing xyz files.
def checked_input_readlines(file_input):
    try:
        lines = file_input.readlines()
    except AttributeError:
        with open(file_input, "r") as input_file:
            lines = input_file.readlines()
    return lines


def write_compound_to_xyz_file(compound, xyz_file_name):
    write_xyz_file(compound.coordinates, xyz_file_name, elements=compound.atomtypes)


def xyz_string(coordinates, elements=None, nuclear_charges=None, extra_string=""):
    """
    Create an xyz-formatted string from coordinates and elements or nuclear charges.
    coordinates : coordinate array
    elements : string array of element symbols
    nuclear_charges : integer array; used to generate element list if elements is set to None
    """
    if elements is None:
        elements = [str_atom_corr(charge) for charge in nuclear_charges]
    output = str(len(coordinates)) + "\n" + extra_string
    for atom_coords, element in zip(coordinates, elements):
        output += (
            "\n" + element + " " + " ".join([str(float(atom_coord)) for atom_coord in atom_coords])
        )
    return output


def write_xyz_file(
    coordinates, xyz_file_name, elements=None, nuclear_charges=None, extra_string=""
):
    xyz_file = open(xyz_file_name, "w")
    xyz_file.write(
        xyz_string(
            coordinates,
            elements=elements,
            nuclear_charges=nuclear_charges,
            extra_string=extra_string,
        )
    )
    xyz_file.close()


def read_xyz_file(xyz_input, additional_attributes=["charge"]):
    lines = checked_input_readlines(xyz_input)
    return read_xyz_lines(lines, additional_attributes=additional_attributes)


def read_xyz_lines(unchecked_xyz_lines, additional_attributes=["charge"]):
    add_attr_dict = {}
    for add_attr in additional_attributes:
        add_attr_dict = {add_attr: None, **add_attr_dict}

    xyz_lines = [check_byte(l) for l in unchecked_xyz_lines]

    num_atoms = int(xyz_lines[0])
    xyz_coordinates = empty_((num_atoms, 3))
    nuclear_charges = empty_((num_atoms,), dtype=dint_)
    atomic_symbols = []

    lsplit = xyz_lines[1].split()
    for l in lsplit:
        for add_attr in additional_attributes:
            add_attr_eq = add_attr + "="
            if add_attr_eq == l[: len(add_attr_eq)]:
                add_attr_dict[add_attr] = int(l.split("=")[1])

    for atom_id, atom_line in enumerate(xyz_lines[2 : num_atoms + 2]):
        lsplit = atom_line.split()
        atomic_symbol = lsplit[0]
        atomic_symbols.append(atomic_symbol)
        nuclear_charges[atom_id] = NUCLEAR_CHARGE[canonical_atomtype(atomic_symbol)]
        for i in range(3):
            xyz_coordinates[atom_id, i] = float(lsplit[i + 1])

    return nuclear_charges, atomic_symbols, xyz_coordinates, add_attr_dict


def xyz_file_stochiometry(xyz_input, by_atom_symbols=True):
    """
    Stochiometry of the xyz input
    xyz_input : either name of an xyz file or the corresponding _io.TextIOWrapper instance
    by_atom_symbols : if "True" use atomic symbols as keys, otherwise use nuclear charges.
    """
    nuclear_charges, atomic_symbols, _, _ = read_xyz_file(xyz_input, additional_attributes=[])
    if by_atom_symbols:
        identifiers = atomic_symbols
    else:
        identifiers = nuclear_charges
    output = {}
    for i in identifiers:
        if i in output:
            output[i] += 1
        else:
            output[i] = 1
    return output


def write2file(string, file_name):
    with open(file_name, "w") as f:
        f.write(string)


# KK: IIRC the two functions were used for G2S in bmapqml repo.
# TODO: jit?
def where2slice(indices_to_ignore):
    return where_(logical_not_(indices_to_ignore))[0]


def nullify_ignored(arr, indices_to_ignore):
    if indices_to_ignore is not None:
        for row_id, cur_ignore_indices in enumerate(indices_to_ignore):
            arr[row_id][where2slice(logical_not_(cur_ignore_indices))] = 0.0


class weighted_array(list):
    def normalize_rhos(self, normalization_constant=None):
        if normalization_constant is None:
            normalization_constant = sum(el.rho for el in self)
        for i in range(len(self)):
            self[i].rho /= normalization_constant

    def sort_rhos(self):
        self.sort(key=lambda x: x.rho, reverse=True)

    def normalize_sort_rhos(self):
        self.normalize_rhos()
        self.sort_rhos()

    def cutoff_minor_weights(self, remaining_rho=None):
        if (remaining_rho is not None) and (len(self) > 1):
            ignored_rhos = 0.0
            for remaining_length in range(len(self), 0, -1):
                upper_cutoff = self[remaining_length - 1].rho
                cut_rho = upper_cutoff * remaining_length + ignored_rhos
                if cut_rho > (1.0 - remaining_rho):
                    density_cut = (1.0 - remaining_rho - ignored_rhos) / remaining_length
                    break
                else:
                    ignored_rhos += upper_cutoff
            del self[remaining_length:]
            for el_id in range(remaining_length):
                self[el_id].rho = max(
                    0.0, self[el_id].rho - density_cut
                )  # max was introduced in case there is some weird numerical noise.
            self.normalize_rhos()
