"""
K.Karan.: Originally (as can be seen in earlier commits in `qml2-dev`) the script build several learning curves for cMBDF + LJ energy and CM + LJ dipole, also iterating over keyword arguments that should not affect results of the calculation (`sd_reuse_Z_derivatives` and `init_thread_assignments`). However, first LJ dipole had to be replaced with LJ energy because otherwise the benchmarks were too unstable (I don't know why; perhaps too many values that are zero?), then the number of full learning curves being built had to be cut down because the test was taking too much CPU time (it still does, but to a lesser extent).
"""

import random

import numpy as np
import pytest
from conftest import (
    add_checksum_to_dict,
    add_reshaped_to_dict,
    compare_or_create,
    comparison_sanity_check,
    kwargs_to_oneliner,
    str2rng,
)

from qml2.jit_interfaces import array_, log_
from qml2.multilevel_sorf.models import MultilevelSORFModel
from qml2.multilevel_sorf.processed_object_constructors import (
    ElementAugmentedLocalRepresentationCalc,
    ProcessedRepresentationListCalc,
)
from qml2.multilevel_sorf.processed_object_constructors.ensemble import (
    EnsembleGlobalRepresentationCalc,
    EnsembleRepresentationCalc,
    global_rep_ensemble_list_dict2datatype,
    local_dn_rep_ensemble_list_dict2datatype,
)
from qml2.multilevel_sorf.processed_object_constructors.standard import GlobalRepresentationCalc
from qml2.representations import generate_coulomb_matrix
from qml2.representations.calculators import cMBDFCalculator
from qml2.test_utils.lennard_jones_potential import RandomLJEnsemble
from qml2.utils import get_sorted_elements, roundup_power2

# test ensemble parameters
natoms_min = 2
natoms_max = 4

num_conformers_min = 2
num_conformers_max = 3

r_cut = 0.25

# for RNG
global_RNG_seed = 16
checksum_RNG = 16

# SORF parameters
nfeatures = 1024
ntransforms = 3

# these combinations of keyword arguments should not affect the output of run_multilevel_sorf_calc, only numerical details of how its hyperparameters are optimized
# default_irrelevant_kwarg_list = [
#    {"init_thread_assignments": False, "sd_reuse_Z_derivatives": False},
#    {"init_thread_assignments": True, "sd_reuse_Z_derivatives": False},
#    {"init_thread_assignments": False, "sd_reuse_Z_derivatives": True},
# ]
default_irrelevant_kwarg_list = [{}]


# generated ensembles for testing.
def generate_ensemble_set(nmols, rng, quantity_name):
    ensembles = []
    all_nuclear_charges = []
    quantities = []
    for _ in range(nmols):
        ensemble = RandomLJEnsemble(
            natoms_min,
            natoms_max,
            num_conformers_min=num_conformers_min,
            num_conformers_max=num_conformers_max,
            r_cut=r_cut,
            rng=rng,
        )
        ensembles.append(ensemble)
        all_nuclear_charges.append(ensemble.get_nuclear_charges())
        match quantity_name:
            case "dipole":
                quantity = ensemble.mean_lj_dipole()
            case "energy":
                quantity = ensemble.mean_lj_energy()
            case _:
                raise Exception("Unknown quantity")
        quantities.append(quantity)
    return ensembles, np.array(quantities), all_nuclear_charges


def get_possible_nuclear_charges(all_nuclear_charges):
    present_nuclear_charges = []
    for nuclear_charges in all_nuclear_charges:
        present_nuclear_charges += list(nuclear_charges)
    return get_sorted_elements(np.array(present_nuclear_charges))


def get_cMBDF_calculator(all_nuclear_charges):
    present_nuclear_charges = get_possible_nuclear_charges(all_nuclear_charges)
    return ElementAugmentedLocalRepresentationCalc(
        representation_function=cMBDFCalculator(all_nuclear_charges),
        possible_nuclear_charges=present_nuclear_charges,
    )


def generate_train_test_sets(quantity_name, representation_name, ntrain=256, ntest=128):
    rng = str2rng("ensembles")
    train_ensembles, train_quantities, train_nuclear_charges = generate_ensemble_set(
        ntrain, rng, quantity_name
    )
    test_ensembles, test_quantities, test_nuclear_charges = generate_ensemble_set(
        ntest, rng, quantity_name
    )
    match representation_name:
        case "cMBDF":
            representation_calculator = get_cMBDF_calculator(
                train_nuclear_charges + test_nuclear_charges
            )
            processed_object_creator = local_dn_rep_ensemble_list_dict2datatype
            ensemble_rep_calculator = EnsembleRepresentationCalc(representation_calculator)
        case "CM":
            representation_calculator = GlobalRepresentationCalc(
                generate_coulomb_matrix, representation_function_kwargs={"size": natoms_max}
            )
            processed_object_creator = global_rep_ensemble_list_dict2datatype
            ensemble_rep_calculator = EnsembleGlobalRepresentationCalc(representation_calculator)
        case _:
            raise Exception("Unknown representation")

    parallel_representation_calculator = ProcessedRepresentationListCalc(ensemble_rep_calculator)

    def final_processor(ensembles):
        return processed_object_creator(parallel_representation_calculator(ensembles))

    train_representations = final_processor(train_ensembles)
    test_representations = final_processor(test_ensembles)

    return (
        train_representations,
        train_quantities,
        train_nuclear_charges,
        test_representations,
        test_quantities,
        test_nuclear_charges,
        parallel_representation_calculator,
    )


#
def get_parameter_list(
    function_definition_list,
    parallel_representation_calculator=None,
    rep_size=None,
    possible_nuclear_charges=None,
    nfeatures=nfeatures,
    num_components=None,
):
    parameter_list = []
    init_size = None
    first_sorf_lvl = True
    for lvl in function_definition_list:
        if lvl == "resize":
            if rep_size is None:
                rep_size = parallel_representation_calculator.max_component_bound()
            # just make smallest possible init_size
            init_size = roundup_power2(rep_size)
            parameter_list.append({"output_size": init_size})
            continue
        if lvl == "rescaling":
            parameter_list.append(
                {"resc_bounds": parallel_representation_calculator.component_bounds}
            )
            continue
        if lvl == "concatenation":
            assert num_components is not None
            parameter_list.append({"num_components": num_components})
            continue
        if lvl in ["sorf", "unscaled_sorf", "mixed_extensive_sorf"]:
            if first_sorf_lvl:
                first_sorf_lvl = False
                assert init_size is not None
                nfeature_stacks = nfeatures // init_size
            else:
                nfeature_stacks = 1
            parameter_list.append({"nfeature_stacks": nfeature_stacks, "ntransforms": ntransforms})
            continue
        if lvl == "element_id_switch":
            assert possible_nuclear_charges is not None
            parameter_list.append({"num_element_ids": len(possible_nuclear_charges)})
            continue
        parameter_list.append({})
    return parameter_list


def run_multilevel_sorf_calc(
    train_representations,
    train_quantities,
    train_nuclear_charges,
    test_representations,
    test_quantities,
    test_nuclear_charges,
    function_definition_list,
    parameter_list,
    extensive_quantity_shift=False,
    intensive_quantity_shift=False,
    init_thread_assignments=False,
    full_optimization_cycle=True,
    lc_hyperparameter_reoptimization=True,
    **other_model_kwargs,
):
    """
    NOTE: changing values of `init_thread_assigments` and `sd_reuse_Z_derivatives` (the latter is part of `other_model_kwargs`) should not change the outcome of the routine.
    """
    # TODO K.Karan. 2025.07.06: introduced full_optimization_cycle option to decrease cost of the test further, realized it only works correctly for the experimental-sorf1 branch (estimated to merge with main in August 2025). Option should be revisited at some point this year.

    # to control RNG inside BOSS, which is used for hyperparameter optimization
    np.random.seed(global_RNG_seed)
    random.seed(global_RNG_seed)
    model_rng = str2rng("model")
    # create the model
    model = MultilevelSORFModel(
        function_definition_list,
        parameter_list,
        extensive_quantity_shift=extensive_quantity_shift,
        intensive_quantity_shift=intensive_quantity_shift,
        rng=model_rng,
        sd_ln_steps=[0.5, 0.25],
        sd_compromise_coeffs=[0.5, 0.25],
        **other_model_kwargs,
    )
    # create a learning curve while reoptimizing hyperparameters
    ntrain = len(train_representations)
    minor_train_representations = train_representations[: ntrain // 2]
    minor_train_quantities = train_quantities[: ntrain // 2]
    if train_nuclear_charges is None:
        minor_train_nuclear_charges = None
    else:
        minor_train_nuclear_charges = train_nuclear_charges[: ntrain // 2]
    checksums = {}
    if full_optimization_cycle:
        if not lc_hyperparameter_reoptimization:
            model.assign_training_set(
                minor_train_representations,
                minor_train_quantities,
                training_nuclear_charges=minor_train_nuclear_charges,
            )
            model.optimize_hyperparameters()
        mean_MAE, std_MAE = model.learning_curve(
            train_representations,
            train_quantities,
            test_representations,
            test_quantities,
            [ntrain // 2, 3 * ntrain // 4],
            max_subset_num=2,
            hyperparameter_reoptimization=lc_hyperparameter_reoptimization,
            training_nuclear_charges=train_nuclear_charges,
            test_nuclear_charges=test_nuclear_charges,
            rng=random,
            init_thread_assignments=init_thread_assignments,
        )
        # include the MAE info into checksum
        add_reshaped_to_dict(checksums, "mean_MAEs", array_(mean_MAE))
        add_checksum_to_dict(checksums, "std_MAEs", array_(std_MAE), nstack_checksums=1)
    else:
        model.assign_training_set(
            minor_train_representations,
            minor_train_quantities,
            training_nuclear_charges=minor_train_nuclear_charges,
        )
        model.hyperparameters = model.find_hyperparameter_guesses()
        model.optimize_quantity_shifts_l2reg()
    # add (sigma) hyperparameters to the checksum too
    add_reshaped_to_dict(checksums, "sd_hyperparameters", model.hyperparameters)
    # NOTE K.Karan.: I think l2 regularization and quantity shifts obtained in the end of an optimization loop can be prone to instabilities, making them unsuitable for this kind of benchmarking. Hence l2reg and shift optimizations are tested separately later.

    # use last optimized hyperparameters to fit model with full training set
    model.fit(
        train_representations, train_quantities, training_nuclear_charges=train_nuclear_charges
    )
    # use last fit for predictions that are also included in the checksums
    predictions = model.get_all_predictions(
        test_representations, all_query_nuclear_charges=test_nuclear_charges
    )
    add_checksum_to_dict(checksums, "predictions", predictions, nstack_checksums=8)
    # run a single calculation of loss function + gradients.
    model.assign_training_set(
        train_representations, train_quantities, training_nuclear_charges=train_nuclear_charges
    )
    guess_hyperparameters = model.get_initial_hyperparameter_guesses()
    ln_guess_hyperparameters = log_(guess_hyperparameters)
    loss1 = model.get_loss_function_ln_hyperparameters(ln_guess_hyperparameters)
    loss2, loss_gradient, Z_quantities = model.get_loss_function_ln_hyperparameters(
        ln_guess_hyperparameters, gradient=True, return_Z_matrices=True
    )
    assert comparison_sanity_check(
        loss1,
        loss2,
        "loss calculated with and without gradients match",
        max_rel_difference=1.0e-10,
    )
    add_reshaped_to_dict(checksums, "single_loss_calc", loss2)
    add_reshaped_to_dict(checksums, "single_loss_calc_der", loss_gradient)
    add_checksum_to_dict(
        checksums,
        "single_loss_calc_singular_values",
        Z_quantities["Z_singular_values"],
        nstack_checksums=8,
    )
    # NOTE: did not include Z_U and Z_Vh into the benchmark because they are much less numerically consistent.
    #    for k, arr in Z_quantities.items():
    #        if k != "Z_singular_values":
    #            arr=copy_(arr)
    #            if k == "Z_Vh":
    #                arr=arr.T
    #            fix_reductor_signs(arr)
    #        full_name="single_loss_calc_"+k
    #        add_checksum_to_dict(checksums, full_name, arr, nstack_checksums=8)
    # also save leaveoneout errors.
    add_checksum_to_dict(
        checksums, "single_loss_calc_loo_errs", model.leaveoneout_errors, nstack_checksums=8
    )
    # calculate loss for a non-optimal l2reg (NOTE: the value of sigma hyperparameters used is the one called last)
    ln_l2reg = log_(model.av_K_diag_element()) - 1
    loss_l2reg1 = model.get_loss_function_ln_l2reg(ln_l2reg)
    loss_l2reg2, loss_l2reg_der = model.get_loss_function_ln_l2reg(ln_l2reg, gradient=True)
    assert comparison_sanity_check(
        loss_l2reg1,
        loss_l2reg2,
        "loss calculated with and without l2reg derivative match",
        max_rel_difference=1.0e-10,
    )
    add_reshaped_to_dict(checksums, "single_l2reg_loss_calc", loss_l2reg2)
    add_reshaped_to_dict(checksums, "single_l2reg_loss_calc_der", loss_l2reg_der)
    # if a quantity shift was used save both the optimal shifts that were calculated last.
    if intensive_quantity_shift or extensive_quantity_shift:
        add_reshaped_to_dict(checksums, "quantity_shift", model.quantity_shift)
        # TODO: rewrite qml2.multilevel_sorf.models to make the matrices used to calculate quantity shifts more accessible? Include them into benchmark here?

    return checksums


def run_several_multilevel_sorf_calcs(
    benchmark_name,
    quantity_name,
    representation_name,
    function_definition_list,
    relevant_kwargs_list,
    irrelevant_kwargs_list=default_irrelevant_kwarg_list,
):
    """
    Run several run_multilevel_sorf_calc instances
    """
    _ = pytest.importorskip("boss")
    # create train and test sets
    (
        train_representations,
        train_quantities,
        train_nuclear_charges,
        test_representations,
        test_quantities,
        test_nuclear_charges,
        parallel_representation_calculator,
    ) = generate_train_test_sets(quantity_name, representation_name)
    # generate parameters for the model
    parameter_list = get_parameter_list(
        function_definition_list,
        parallel_representation_calculator,
        possible_nuclear_charges=get_possible_nuclear_charges(
            train_nuclear_charges + test_nuclear_charges
        ),
    )
    for kwarg_name, relevant_kwargs in relevant_kwargs_list:
        final_benchmark_name = benchmark_name + "_" + kwarg_name
        for irrelevant_kwargs in irrelevant_kwargs_list:
            checksums = run_multilevel_sorf_calc(
                train_representations,
                train_quantities,
                train_nuclear_charges,
                test_representations,
                test_quantities,
                test_nuclear_charges,
                function_definition_list,
                parameter_list,
                **relevant_kwargs,
                **irrelevant_kwargs,
            )
            if irrelevant_kwargs:
                extra_print_string = kwargs_to_oneliner(irrelevant_kwargs)
            else:
                extra_print_string = None
            compare_or_create(
                checksums,
                final_benchmark_name,
                max_rel_difference=5.0e-5,
                extra_print_string=extra_print_string,
            )


def test_cMBDF_multilevel_sorf():
    relevant_kwargs_list = [
        (
            "ext_shift",
            {
                "extensive_quantity_shift": True,
                "init_thread_assignments": False,
                "sd_reuse_Z_derivatives": True,
                "lc_hyperparameter_reoptimization": False,
            },
        ),
        (
            "no_ext_shift",
            {
                "extensive_quantity_shift": False,
                "init_thread_assignments": True,
                "sd_reuse_Z_derivatives": True,
            },
        ),
    ]
    run_several_multilevel_sorf_calcs(
        "MSORF_cMBDF",
        "energy",
        "cMBDF",
        [
            "resize",
            "sorf",
            "element_id_switch",
            "component_sum",
            "mixed_extensive_sorf",
            "weighted_component_sum",
            "component_sum",
            "mixed_extensive_sorf",
        ],
        relevant_kwargs_list,
    )


def test_CM_multilevel_sorf():
    relevant_kwargs_list = [
        (
            "int_shift",
            {
                "intensive_quantity_shift": True,
                "init_thread_assignments": True,
                "sd_reuse_Z_derivatives": False,
            },
        ),
        (
            "no_int_shift",
            {
                "intensive_quantity_shift": False,
                "init_thread_assignments": False,
                "sd_reuse_Z_derivatives": False,
                "lc_hyperparameter_reoptimization": False,
            },
        ),
    ]
    run_several_multilevel_sorf_calcs(
        "MSORF_CM",
        "energy",
        "CM",
        [
            "resize",
            "sorf",
            "weighted_component_sum",
            "normalization",
            "component_sum",
            "sorf",
        ],
        relevant_kwargs_list,
    )


if __name__ == "__main__":
    test_CM_multilevel_sorf()
    test_cMBDF_multilevel_sorf()
