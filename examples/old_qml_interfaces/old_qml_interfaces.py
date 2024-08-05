# A collection of functions that take input analogous to the ones in the old qml package,
# but do calculation with qml2 package. Please contact the repo maintainers for functions
# you would want added here.
import numpy as np

from qml2.kernels import local_dn_gaussian_kernel


def get_natoms_arr_all_ncharges(ncharges_lists):
    return np.array([len(ncharges_list) for ncharges_list in ncharges_lists]), np.concatenate(
        ncharges_lists
    )


def reps_3D_to_2D(merged_representations, natoms_arr):
    output = np.empty((np.sum(natoms_arr), merged_representations.shape[-1]))
    lb = 0
    for mol_id, natoms in enumerate(natoms_arr):
        ub = lb + natoms
        output[lb:ub, :] = merged_representations[mol_id, :natoms]
        lb = ub
    return output


def get_local_kernel_qml2(
    merged_representations1, merged_representations2, ncharges_lists1, ncharges_lists2, sigma
):
    """
    A substitute for qml.kernels.gradient_kernels.get_local_kernel
    """
    natoms_arr1, all_ncharges1 = get_natoms_arr_all_ncharges(ncharges_lists1)
    natoms_arr2, all_ncharges2 = get_natoms_arr_all_ncharges(ncharges_lists2)
    reps1 = reps_3D_to_2D(merged_representations1, natoms_arr1)
    reps2 = reps_3D_to_2D(merged_representations2, natoms_arr2)

    return local_dn_gaussian_kernel(
        reps1, reps2, natoms_arr1, natoms_arr2, all_ncharges1, all_ncharges2, sigma
    )
