# TODO: KK: Can be optimized a bit by avoiding to calculate element_ids inside and outside the reductor functions.

from typing import Tuple, Union

from .jit_interfaces import (
    dot_,
    elements_where_,
    empty_,
    int_,
    jit_,
    ndarray_,
    permuted_range_,
    prange_,
    standard_normal_,
    sum_,
    svd_,
)
from .utils import check_allocation, get_sorted_elements


@jit_
def get_rand_reductor(nreps: int_, npcas: int_):
    """
    Creates random reductor in a way compilable by both Numba and Torch.
    TODO: I think it's unbiased, but should check.
    """
    print("WARNING! get_rand_reductor had to be called for a missing element.")
    max_dim = max(nreps, npcas)
    min_dim = min(nreps, npcas)
    rand_vecs = standard_normal_((max_dim, min_dim))
    rand_reductor, _, _ = svd_(rand_vecs, full_matrices=False)
    if nreps < npcas:
        return rand_reductor.T
    return rand_reductor


# KK: Only difference from what Nickolas Browning used is added option to use all atomic environments in the PCA.
@jit_
def get_reductor(representations, npcas: int_, num_samples: int_ = 1024):
    nreps = representations.shape[1]
    if (nreps <= npcas) or (num_samples <= npcas):
        # TODO KK: I suspect creating a random reductor should make the procedure less
        # demanding w.r.t. ntransforms, but never checked it.
        # Just pad with zeros. Useful for fitting into Hadamard transform.
        # reductor = zeros_((nreps, npcas))
        # for i in range(nreps):
        #     reductor[i, i] = 1.0
        # return reductor
        return get_rand_reductor(nreps, npcas)

    if num_samples is None:
        used_representations = representations
    else:
        permuted_indices = permuted_range_(representations.shape[0])
        sample_indices = permuted_indices[:num_samples]
        true_nsamples = sample_indices.shape[0]
        used_representations = representations[sample_indices]

    if true_nsamples >= npcas:
        eigvecs, eigvals, _ = svd_(used_representations.T, full_matrices=False)
    else:
        print("WARNING: sample representation vectors insufficient to generate valid reductor.")
        # KK: such a situation only occures in tests, but a placeholder should still be valuable.
        # TODO: double-check it actually works as intended.
        eigvecs, eigvals, _ = svd_(
            dot_(used_representations.T, used_representations), full_matrices=False
        )

    all_eigvals_sum = sum_(eigvals)

    cev = 100 - (all_eigvals_sum - sum_(eigvals[:npcas])) / all_eigvals_sum * 100
    reductor = eigvecs[:, :npcas]
    size_from = reductor.shape[0]
    size_to = reductor.shape[1]
    print(size_from, "->", size_to, "Cumulative Explained Feature Variance =", float(cev), "%")

    return reductor


@jit_
def get_reductors_diff_species(
    representations: ndarray_,
    ncharges: ndarray_,
    npcas: int_,
    num_samples: int_ = 1024,
    sorted_elements: Union[None, ndarray_] = None,
) -> Tuple[ndarray_, ndarray_]:
    if sorted_elements is None:
        sorted_elements = get_sorted_elements(ncharges)
    nelements = sorted_elements.shape[0]
    nreps = representations.shape[1]
    all_reductors = empty_((nelements, nreps, npcas))
    for i_element in range(nelements):
        element = sorted_elements[i_element]
        # TODO:KK:this use of where_ causes errors with TorchScript.
        # el_reps = representations[where_(ncharges == element)]
        el_reps = elements_where_(representations, (ncharges == element))
        all_reductors[i_element, :, :] = get_reductor(el_reps, npcas, num_samples=num_samples)[
            :, :
        ]
    return all_reductors, sorted_elements


@jit_
def project_representation(X, reductor):
    """
    projects the representation from shape:
    nsamples x repsize
    to
    nsamples x npcas

    """

    return dot_(X, reductor)


@jit_
def project_scale_representations(
    X, reductor, sigma, output: Union[ndarray_, None] = None, nmols: int_ = -1
):
    npcas = reductor.shape[1]
    if nmols == -1:
        nmols = X.shape[0]
    output = check_allocation((nmols, npcas), output=output)
    output[:nmols, :] = project_representation(X, reductor) / sigma
    return output


@jit_(numba_parallel=True)
def project_local_representations(
    all_X, element_ids, all_reductors, output: Union[ndarray_, None] = None, natoms: int_ = -1
) -> ndarray_:
    npcas = all_reductors.shape[-1]
    if natoms == -1:
        natoms = element_ids.shape[0]
    output = check_allocation((natoms, npcas), output=output)
    for i_atom in prange_(natoms):
        el_id = int(element_ids[i_atom])
        output[i_atom, :] = project_representation(all_X[i_atom], all_reductors[el_id])
    return output


@jit_(numba_parallel=True)
def project_scale_local_representations(
    all_X,
    element_ids,
    all_reductors,
    sigma,
    output: Union[ndarray_, None] = None,
    natoms: int_ = -1,
) -> ndarray_:
    output = project_local_representations(
        all_X, element_ids, all_reductors, output=output, natoms=natoms
    )
    output[:, :] /= sigma
    return output
