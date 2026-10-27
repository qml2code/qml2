import pytest
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
    concatenated_pair_rep_def,
    slater_pairs_rep_def,
)
from qml2.multilevel_sorf.sorf_calculation import MultilevelSORF

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


def FJK_rep_generator(concatenate_components=False, **rep_params_kwargs):
    from qml2.orb_ml.representations import OML_rep_params

    FJK_rep_params = OML_rep_params(max_angular_momentum=1, **rep_params_kwargs)
    return FJKRepresentationCalc(
        oml_rep_params=FJK_rep_params, concatenate_components=concatenate_components
    )


qm7_possible_nuclear_charges = array_([1, 6, 7, 8, 16])


def create_ensemble_representations(
    SMILES_list,
    base_class_name="Compound",
    upper_nhatoms_bound=None,
    max_num_created=None,
    r_cut=0.05,
    verbose=False,
    morfeus_random_seed=8,
    **compound_kwargs,
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
    max_rel_difference=1.0e-5,
    extra_level_parameters=None,
    **parameter_list_kwargs,
):
    """`extra_level_parameters` maps a level name to parameters `get_parameter_list` cannot supply.

    It exists for compile-time parameters: `get_parameter_list` walks the definition list and knows
    what each level needs, but a baked-in CALLABLE is not something it can derive from sizes and
    counts. Merged rather than replaced, so a level can take both kinds. Raises on a level that is not
    in the stack, since silently ignoring it would leave the caller with a model built from defaults
    while believing otherwise.
    """
    from qml2.jit_interfaces import dot_

    parameter_list = get_parameter_list(
        function_definition_list, nfeatures=1024, **parameter_list_kwargs
    )
    for level_name, extra in (extra_level_parameters or {}).items():
        if level_name not in function_definition_list:
            raise ValueError(
                f"extra_level_parameters names level {level_name!r}, which is not in "
                f"{function_definition_list}"
            )
        parameter_list[function_definition_list.index(level_name)].update(extra)
    sorf_rng = str2rng("sorf")
    sorf_calculator = MultilevelSORF(
        function_definition_list, parameter_list, rng=sorf_rng, test_mode=True
    )
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
    # NOTE: point 2 is narrower than it reads, and worth stating exactly because the
    # benchmark below depends on it. K = Z @ Z.T is invariant under Z -> -Z, i.e. a GLOBAL
    # sign on the feature matrix. A per-row sign flip of a projection basis is not that: it
    # flips one component of every projected vector, which changes Z non-trivially, and no
    # tolerance absorbs it either -- SORF's own approximation error at nfeatures=1024 is
    # around 3%, against the 1e-5 this is compared at. What actually covers it is
    # `inplace_orthogonalize_vectors` now returning a CANONICAL basis (a QR whose R has a
    # positive diagonal, which is unique for full-rank input) where the SVD it used to use
    # was determined only up to a sign per row.
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

    pytest.importorskip("morfeus")
    pytest.importorskip("boss")

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
    extra_level_parameters=None,
    **compound_kwargs,
):
    pytest.importorskip("morfeus")
    pytest.importorskip("boss")
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
        extra_level_parameters=extra_level_parameters,
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


def test_numba_FCHL19_MSORF_features():
    """The same features as test_FCHL19_MSORF_features, with FCHL19 computed INSIDE the SORF loop.

    Checked against the same benchmark, deliberately. The `calculate_representation` level is supposed
    to be invisible to everything below it: the per-atom levels see the object they always saw, so the
    features must come out identical rather than merely close. Sharing one benchmark file with the
    precomputed test is what makes that claim testable -- if the two ever disagree, one of them is
    wrong, and no separately-generated reference could tell us which.

    What differs is only WHERE the representation is computed, and therefore what is stored: the
    conformers carry their geometry, and each atom's vector exists only while the loop is on that atom.

    This also exercises the import pass across a real process boundary. A jitclass cannot be pickled,
    so the generator emits plain dicts and the parent builds the datatype -- and since
    `ProcessedRepresentationListCalc` is called here without `num_procs`, that path is live in this
    test rather than assumed.
    """
    from qml2.multilevel_sorf.base_constructors import get_transform_list_dict2datatype
    from qml2.multilevel_sorf.processed_object_constructors.ensemble import comp2ensemble_rep_def
    from qml2.multilevel_sorf.processed_object_constructors.numba_constructors import (
        get_el_aug_rep_constructor,
    )
    from qml2.multilevel_sorf.processed_object_constructors.standard import (
        NumbaConformerGenerator,
        conformer_lvl,
    )
    from qml2.representations.numba_calculators import FCHL19JITCalculator

    # The stack of test_FCHL19_MSORF_features with one level inserted, immediately after the per-atom
    # `component_sum`: that is the boundary where a conformer stops being the unit being handled.
    function_definition_list = [
        "resize",
        "sorf",
        "element_id_switch",
        "component_sum",
        "calculate_representation",
        "mixed_extensive_sorf",
        "weighted_component_sum",
        "component_sum",
        "mixed_extensive_sorf",
    ]

    base_calculator = FCHL19JITCalculator(elements=qm7_possible_nuclear_charges)
    # Measured with the same jitted function the level bakes in. One region: unlike the
    # charge-augmented scripts there is no extra per-atom column here, so there is one sigma.
    probe_coordinates = array_([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [-0.6, 1.2, 0.0]])
    rep_width = base_calculator.single_coordinate_calc(array_([6, 1, 1]), probe_coordinates).shape[
        1
    ]
    component_bounds = array_([[0, rep_width]])

    rep_calculator = NumbaConformerGenerator(component_bounds)
    rep_list_processor = get_transform_list_dict2datatype(comp2ensemble_rep_def([conformer_lvl]))

    run_feature_test(
        function_definition_list,
        rep_calculator,
        rep_list_processor,
        # NOT a new suffix: the point is to land on the precomputed test's benchmark.
        "FCHL19",
        upper_nhatoms_bound=5,
        parameter_list_kwargs={"possible_nuclear_charges": qm7_possible_nuclear_charges},
        extra_level_parameters={
            "calculate_representation": {
                "compilation_parameters": {
                    "representation_function": base_calculator.numba_calculator,
                    "processed_object_constructor": get_el_aug_rep_constructor(
                        qm7_possible_nuclear_charges
                    ),
                }
            }
        },
    )


def test_FJK_MSORF_features(
    fjd_coup_mats=False,
    calc_type="HF",
    software="pySCF",
    orb_rep_name="FJK",
    localization_procedure="IBO",
):
    """
    Test generating 'single determinant' features from FJK.
    """
    pytest.importorskip("pyscf")
    run_feature_test(
        FJK_function_definition_list + ["weighted_component_sum", "component_sum"],
        FJK_rep_generator(fjd_coup_mats=fjd_coup_mats),
        fjk_comp_ensemble_list_dict2datatype,
        orb_rep_name,
        upper_nhatoms_bound=5,
        base_class_name="OML_Compound",
        calc_type=calc_type,
        software=software,
        localization_procedure=localization_procedure,
    )


def test_FJD_MSORF_features():
    # Pipek-Mezey rather than the IBO default: IAO/IBO is ill-posed on GFN2's valence-only
    # occupied space -- the MINAO reference carries core shells with nothing to match.
    test_FJK_MSORF_features(
        fjd_coup_mats=True,
        calc_type="GFN2-xTB",
        software="tblite",
        orb_rep_name="FJD",
        localization_procedure="Pipek-Mezey",
    )


def test_pair_FJK_MSORF_features(
    fjd_coup_mats=False,
    calc_type="HF",
    software="pySCF",
    second_oml_comp_kwargs={"used_orb_type": "HOMO_removed"},
    orb_rep_name="FJK",
    localization_procedure="IBO",
):
    """
    Test generating 'pair determinant' features from FJK (example used here generates HOMO-unoccupied pair).
    """
    pytest.importorskip("pyscf")
    compound_kwargs = {
        "calc_type": calc_type,
        "software": software,
        "localization_procedure": localization_procedure,
        "second_oml_comp_kwargs": second_oml_comp_kwargs,
    }
    comp_list_dict_list2reps = get_transform_list_dict2datatype(comp_rep_def + ["list"])
    run_feature_test(
        FJK_function_definition_list + ["component_sum"],
        FJK_rep_generator(fjd_coup_mats=fjd_coup_mats),
        comp_list_dict_list2reps,
        f"{orb_rep_name}_Slater_pair",
        base_class_name="OML_Slater_pair",
        upper_nhatoms_bound=5,
        ensemble_rep_wrapper=MinConformerRepresentationCalc,
        **compound_kwargs,
    )


def test_pair_FJD_MSORF_features():
    test_pair_FJK_MSORF_features(
        fjd_coup_mats=True,
        calc_type="GFN2-xTB",
        software="tblite",
        second_oml_comp_kwargs={"charge": 1},
        orb_rep_name="FJD",
        localization_procedure="Pipek-Mezey",
    )


def test_pairs_FJK_MSORF_features():
    """
    Test generating 'pairs of determinants' features from FJK (example used here combines a pair with HOMO unoccupied and LUMO occupied).

    NOTE: the option was introduced to try model excitation energies by combining the representation of HOMO and LUMO. It did not appear in any publications because the resulting methods didn't improve on FCHL19 for spectra prediction.
    """
    pytest.importorskip("pyscf")
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
        FJK_rep_generator(),
        slater_pairs_list_dict_list2reps,
        "FJK_Slater_pairs",
        base_class_name="OML_Slater_pairs",
        upper_nhatoms_bound=5,
        ensemble_rep_wrapper=MinConformerRepresentationCalc,
        parameter_list_kwargs={"num_components": 2},
        **compound_kwargs,
    )


def test_concatenated_pair_FJD_MSORF_features():
    """A Slater pair whose two determinants are concatenated rather than subtracted.

    The pair is the molecule in vacuum and the same molecule in water, which is the
    FreeSolv configuration; the representation is FJD on GFN2, where the solvent enters
    through ALPB. Where test_pair_FJD_MSORF_features sums the two determinants' orbitals
    into one weighted list with the first negated -- so the features are a DIFFERENCE --
    this keeps them apart and lets a "concatenation" level lay them end to end, doubling
    the feature vector instead of cancelling within it.

    The architecture is the one test_pairs_FJK_MSORF_features already uses for several
    determinants: the same FJK stack down to one vector per determinant, then
    "concatenation" with num_components=2. What differs is where the components come from
    -- there, several Slater pairs of one molecule; here, the two halves of a single pair.
    """
    pytest.importorskip("tblite")
    concatenated_pair_list_dict_list2reps = get_transform_list_dict2datatype(
        concatenated_pair_rep_def + ["list"]
    )
    run_feature_test(
        FJK_function_definition_list + ["concatenation", "component_sum"],
        FJK_rep_generator(concatenate_components=True, fjd_coup_mats=True),
        concatenated_pair_list_dict_list2reps,
        "FJD_concatenated_pair",
        base_class_name="OML_Slater_pair",
        upper_nhatoms_bound=5,
        ensemble_rep_wrapper=MinConformerRepresentationCalc,
        parameter_list_kwargs={"num_components": 2},
        calc_type="GFN2-xTB",
        software="tblite",
        # Pipek-Mezey rather than the IBO default, as everywhere GFN2 is used here.
        localization_procedure="Pipek-Mezey",
        second_oml_comp_kwargs={"smd_solvent": "water"},
    )


if __name__ == "__main__":
    test_reduced_SLATM_MSORF_features()
    test_FCHL19_MSORF_features()
    test_FJK_MSORF_features()
    test_FJD_MSORF_features()
    test_pair_FJK_MSORF_features()
    test_pair_FJD_MSORF_features()
    test_pairs_FJK_MSORF_features()
    test_concatenated_pair_FJD_MSORF_features()
