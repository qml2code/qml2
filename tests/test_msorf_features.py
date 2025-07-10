from conftest import (
    add_checksum_to_dict,
    add_reshaped_to_dict,
    compare_or_create,
    comparison_sanity_check,
    get_shuffled_QM7_SMILES,
    str2rng,
)

# NOTE: K.Karan. nfeatures and r_cut are not imported here because in this test it makes sense to use different values.
from test_msorf_models import get_parameter_list

from qml2.ensemble import Ensemble
from qml2.jit_interfaces import array_
from qml2.multilevel_sorf.base_constructors import get_transform_list_dict2datatype
from qml2.multilevel_sorf.processed_object_constructors import ProcessedRepresentationListCalc
from qml2.multilevel_sorf.processed_object_constructors.ensemble import (
    EnsembleRepresentationCalc,
    MinConformerRepresentationCalc,
    fjk_comp_ensemble_list_dict2datatype,
)
from qml2.multilevel_sorf.processed_object_constructors.fjk import (
    FJKRepresentationCalc,
    comp_rep_def,
    slater_pairs_rep_def,
)
from qml2.multilevel_sorf.sorf_calculation import MultilevelSORF
from qml2.orb_ml.representations import OML_rep_params

num_conformers = 8

FJK_function_definition_list = [
    "resize",
    "rescaling",
    "unscaled_sorf",
    "weighted_component_sum",
    "normalization",
    "sorf",
    "weighted_component_sum",
]
FJK_rep_params = OML_rep_params(max_angular_momentum=1)
FJK_rep_generator = FJKRepresentationCalc(oml_rep_params=FJK_rep_params)
qm7_possible_nuclear_charges = array_([1, 6, 7, 8, 16])


def create_ensemble_representations(
    SMILES_list,
    base_class_name="Compound",
    upper_nhatoms_bound=None,
    max_num_created=None,
    r_cut=0.05,
    verbose=False,
    morfeus_random_seed=8,
    **compound_kwargs
):
    ensembles = []
    for SMILES in SMILES_list:
        ensemble = Ensemble(
            SMILES=SMILES,
            num_conformers=num_conformers,
            base_class_name=base_class_name,
            r_cut=r_cut,
            compound_kwargs=compound_kwargs,
            random_seed=morfeus_random_seed,
        )
        if upper_nhatoms_bound is not None:
            nuclear_charges = ensemble.get_nuclear_charges()
            nhatoms = sum(nuclear_charges != 1)
            if nhatoms >= upper_nhatoms_bound:
                continue
        ensembles.append(ensemble)
        if verbose:
            print(SMILES)
        if max_num_created is not None:
            if len(ensembles) == max_num_created:
                break
    return ensembles


def generate_check_features(
    compound_representations,
    function_definition_list,
    benchmark_suffix,
    max_rel_difference=1.0e-6,
    **parameter_list_kwargs
):
    from qml2.jit_interfaces import dot_

    parameter_list = get_parameter_list(
        function_definition_list, nfeatures=1024, **parameter_list_kwargs
    )
    sorf_rng = str2rng("sorf")
    sorf_calculator = MultilevelSORF(function_definition_list, parameter_list, rng=sorf_rng)
    reasonable_hyperparameters = sorf_calculator.hyperparameter_initial_guesses(
        compound_representations
    )
    feature_matrix1 = sorf_calculator.calc_Z_matrix(
        compound_representations, reasonable_hyperparameters
    )
    feature_matrix2, feature_matrix_derivatives = sorf_calculator.calc_Z_matrix(
        compound_representations, reasonable_hyperparameters, gradient=True
    )

    assert comparison_sanity_check(
        feature_matrix1,
        feature_matrix2,
        "features calculated with and without gradient",
        max_rel_difference=1e-7,
    )

    final_checksums = {}
    add_reshaped_to_dict(final_checksums, "hyperparameters", reasonable_hyperparameters)
    # K.Karan.: 1. AFAIK feature approximations for kernel function are more numerically stable than the features themselves, hence I use them for benchmarking.
    # 2. Should solve sign-flipping problem potentially happening for SLATM
    K_approx = dot_(feature_matrix2, feature_matrix2.T)
    add_checksum_to_dict(final_checksums, "K_approx", K_approx, nstack_checksums=32)
    K_approx_der = dot_(feature_matrix_derivatives, feature_matrix2.T)
    add_checksum_to_dict(
        final_checksums,
        "K_approx_der",
        K_approx_der,
        nstack_checksums=32,
    )

    final_benchmark_name = "MSORF_features_" + benchmark_suffix
    compare_or_create(final_checksums, final_benchmark_name, max_rel_difference=max_rel_difference)


def test_reduced_SLATM_MSORF_features():
    """
    Test generating ensemble representations with SLATM using the "training conformer projection" procedure used to decrease dimensionality of SLATM representation vectors in the MSORF paper.
    """
    from qml2.jit_interfaces import copy_
    from qml2.math import inplace_orthogonalize_vectors
    from qml2.multilevel_sorf.processed_object_constructors.ensemble import (
        EnsembleGlobalRepresentationCalc,
        global_rep_ensemble_list_dict2datatype,
    )
    from qml2.multilevel_sorf.processed_object_constructors.standard import (
        GlobalRepresentationCalc,
    )
    from qml2.multilevel_sorf.processed_object_constructors.utils import (
        extract_all_conformer_representations,
        transform_all_conformer_representations_to_principal_components,
    )
    from qml2.representations.calculators import SLATMCalculator

    example_SMILES = get_shuffled_QM7_SMILES()
    ensembles = create_ensemble_representations(example_SMILES, upper_nhatoms_bound=5)
    nmols = len(ensembles)
    print("nensembles:", len(ensembles))
    all_nuclear_charges = [ensemble.get_nuclear_charges() for ensemble in ensembles]

    rep_calculator = GlobalRepresentationCalc(SLATMCalculator(all_nuclear_charges))

    ensemble_rep_calculator = EnsembleGlobalRepresentationCalc(rep_calculator)

    parallel_calculator = ProcessedRepresentationListCalc(ensemble_rep_calculator)

    ensemble_dicts = parallel_calculator(ensembles)

    processed_ensembles = global_rep_ensemble_list_dict2datatype(ensemble_dicts)

    basis_reps = extract_all_conformer_representations(processed_ensembles[: nmols // 2])
    print("basis reps shape:", basis_reps.shape)

    orthogonal_basis = copy_(basis_reps)
    inplace_orthogonalize_vectors(orthogonal_basis)

    function_definition_list = [
        "resize",
        "sorf",
        "weighted_component_sum",
        "normalization",
        "component_sum",
        "sorf",
    ]

    processed_ensembles = transform_all_conformer_representations_to_principal_components(
        processed_ensembles, orthogonal_basis, add_remainder=True
    )

    generate_check_features(
        processed_ensembles,
        function_definition_list,
        "projected_SLATM",
        rep_size=orthogonal_basis.shape[0] + 1,
    )


def run_feature_test(
    function_definition_list,
    rep_calculator,
    rep_list_processor,
    benchmark_suffix,
    upper_nhatoms_bound=None,
    max_num_created=None,
    base_class_name="Compound",
    ensemble_rep_wrapper=EnsembleRepresentationCalc,
    parameter_list_kwargs={},
    **compound_kwargs
):
    SMILES_list = get_shuffled_QM7_SMILES()
    ensembles = create_ensemble_representations(
        SMILES_list,
        base_class_name=base_class_name,
        upper_nhatoms_bound=upper_nhatoms_bound,
        max_num_created=max_num_created,
        **compound_kwargs,
    )

    ensemble_rep_calculator = ensemble_rep_wrapper(rep_calculator)
    parallel_calculator = ProcessedRepresentationListCalc(ensemble_rep_calculator)
    ensemble_dicts = parallel_calculator(ensembles)
    processed_ensembles = rep_list_processor(ensemble_dicts)

    generate_check_features(
        processed_ensembles,
        function_definition_list,
        benchmark_suffix,
        parallel_representation_calculator=parallel_calculator,
        **parameter_list_kwargs,
    )


def test_FCHL19_MSORF_features():
    """
    Test generating 'local dn' features from FCHL19 representation.
    """
    from qml2.multilevel_sorf.processed_object_constructors import (
        ElementAugmentedLocalRepresentationCalc,
    )
    from qml2.multilevel_sorf.processed_object_constructors.ensemble import (
        local_dn_rep_ensemble_list_dict2datatype,
    )
    from qml2.representations import generate_fchl19

    function_definition_list = [
        "resize",
        "sorf",
        "element_id_switch",
        "component_sum",
        "mixed_extensive_sorf",
        "weighted_component_sum",
        "component_sum",
        "mixed_extensive_sorf",
    ]
    rep_calculator = ElementAugmentedLocalRepresentationCalc(
        representation_function=generate_fchl19,
        possible_nuclear_charges=qm7_possible_nuclear_charges,
        representation_function_kwargs={"elements": qm7_possible_nuclear_charges},
    )
    run_feature_test(
        function_definition_list,
        rep_calculator,
        local_dn_rep_ensemble_list_dict2datatype,
        "FCHL19",
        upper_nhatoms_bound=5,
        parameter_list_kwargs={"possible_nuclear_charges": qm7_possible_nuclear_charges},
    )


def test_FJK_MSORF_features():
    """
    Test generating 'single determinant' features from FJK.
    """
    run_feature_test(
        FJK_function_definition_list + ["weighted_component_sum", "component_sum"],
        FJK_rep_generator,
        fjk_comp_ensemble_list_dict2datatype,
        "FJK",
        upper_nhatoms_bound=5,
        base_class_name="OML_Compound",
    )


def test_pair_FJK_MSORF_features():
    """
    Test generating 'pair determinant' features from FJK (example used here generates HOMO-unoccupied pair).
    """
    compound_kwargs = {
        "second_oml_comp_kwargs": {"used_orb_type": "HOMO_removed"},
    }
    comp_list_dict_list2reps = get_transform_list_dict2datatype(comp_rep_def + ["list"])
    run_feature_test(
        FJK_function_definition_list + ["component_sum"],
        FJK_rep_generator,
        comp_list_dict_list2reps,
        "FJK_Slater_pair",
        base_class_name="OML_Slater_pair",
        upper_nhatoms_bound=5,
        ensemble_rep_wrapper=MinConformerRepresentationCalc,
        **compound_kwargs,
    )


def test_pairs_FJK_MSORF_features():
    """
    Test generating 'pairs of determinants' features from FJK (example used here combines a pair with HOMO unoccupied and LUMO occupied).

    NOTE: the option was introduced to try model excitation energies by combining the representation of HOMO and LUMO. It did not appear in any publications because the resulting methods didn't improve on FCHL19 for spectra prediction.
    """
    compound_kwargs = {
        "other_oml_comp_kwargs_list": [
            {"used_orb_type": "HOMO_removed"},
            {"used_orb_type": "LUMO_added"},
        ],
    }
    slater_pairs_list_dict_list2reps = get_transform_list_dict2datatype(
        slater_pairs_rep_def + ["list"]
    )
    run_feature_test(
        FJK_function_definition_list + ["concatenation", "component_sum"],
        FJK_rep_generator,
        slater_pairs_list_dict_list2reps,
        "FJK_Slater_pairs",
        base_class_name="OML_Slater_pairs",
        upper_nhatoms_bound=5,
        ensemble_rep_wrapper=MinConformerRepresentationCalc,
        parameter_list_kwargs={"num_components": 2},
        **compound_kwargs,
    )


if __name__ == "__main__":
    test_reduced_SLATM_MSORF_features()
    test_FCHL19_MSORF_features()
    test_FJK_MSORF_features()
    test_pair_FJK_MSORF_features()
    test_pairs_FJK_MSORF_features()
