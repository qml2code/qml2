import numpy as np
from conftest import (
    add_checksum_to_dict,
    add_reshaped_to_dict,
    compare_or_create,
    get_qm7_hof,
    perturbed_xyz_list,
    qm7_nuclear_charges,
    str2rng,
)

from qml2 import Compound
from qml2.jit_interfaces import array_, default_rng_, empty_, log_
from qml2.models.krr import KRRLocalModel, KRRModel
from qml2.models.sorf import SORFLocalModel, SORFModel
from qml2.representations.calculators import CMCalculator, FCHL19Calculator

default_num_train_mols = 32
default_num_test_mols = 16
default_lc_training_set_sizes = [15, 30]

# calculators use parameters for QM7, which is the test dataset we use.
CMCalc = CMCalculator(size=29)
FCHL19Calc = FCHL19Calculator(elements=np.array([1, 6, 7, 8, 16], dtype=int))


def tuples_to_set(qm7_tuples):
    energies = []
    compounds = []
    for t in qm7_tuples:
        energies.append(t[0])
        compounds.append(Compound(xyz=t[1]))
    return array_(energies), compounds


def generate_train_test_sets(
    rng=None, num_train_mols=default_num_train_mols, num_test_mols=default_num_test_mols
):
    if rng is None:
        rng = str2rng("sets")
    qm7_hofs = get_qm7_hof()
    qm7_xyzs = perturbed_xyz_list()
    qm7_tuples = list(zip(qm7_hofs, qm7_xyzs))
    rng.shuffle(qm7_tuples)
    training_energies, training_compounds = tuples_to_set(qm7_tuples[:num_train_mols])
    test_energies, test_compounds = tuples_to_set(
        qm7_tuples[num_train_mols : num_train_mols + num_test_mols]
    )
    return training_energies, training_compounds, test_energies, test_compounds


def run_model_type_check(
    model_type,
    benchmark_name,
    lc_training_set_sizes=default_lc_training_set_sizes,
    model_kwargs={},
):
    # define model
    # custom ln_l2reg_diag_ratio_bounds are needed because l2reg optimization for lower log values becomes numerically unstable (e.g. sometimes gives -24., sometimes -22.)
    # hyp_1D_opt_method="grid" is also for numerical stability
    # training_reps_suppress_openmp=False avoids representation recompilation (only noticeable for small sizes)
    model = model_type(
        **model_kwargs,
        l2reg_total_iterpts=8,
        sigma_total_iterpts=8,
        hyp_1D_opt_method="grid",
        ln_l2reg_diag_ratio_bounds=[-8.0, 2.0],
        training_reps_suppress_openmp=False,
    )

    (
        training_energies,
        training_compounds,
        test_energies,
        test_compounds,
    ) = generate_train_test_sets(
        num_train_mols=default_num_train_mols, num_test_mols=default_num_test_mols
    )

    checksums = {}
    lc_rng = str2rng("learning")
    # fix the global RNG seed for test reproducibility
    np.random.seed(8)
    # first build learning curve while re-optimizing hyperparameters
    mean_MAE, std_MAE = model.learning_curve(
        training_compounds,
        training_energies,
        test_compounds,
        test_energies,
        lc_training_set_sizes,
        rng=lc_rng,
        hyperparameter_reoptimization=True,
        num_procs=1,
        suppress_openmp=False,
    )
    add_reshaped_to_dict(checksums, "mean_MAEs", array_(mean_MAE))
    add_checksum_to_dict(checksums, "std_MAEs", array_(std_MAE), nstack_checksums=1)
    # use last considered hyperparameters to do a model fit.
    # num_procs=1 avoids JIT recompilation
    model.fit(
        training_compounds=training_compounds,
        training_quantities=training_energies,
        num_procs=1,
        suppress_openmp=False,
    )
    predictions = empty_(test_energies.shape)
    for i_comp, compound in enumerate(test_compounds):
        predictions[i_comp] = model.predict_from_compound(compound)
    add_checksum_to_dict(checksums, "predictions", predictions)
    unopt_mean_MAE, unopt_std_MAE = model.learning_curve(
        training_compounds,
        training_energies,
        test_compounds,
        test_energies,
        lc_training_set_sizes,
        rng=lc_rng,
        hyperparameter_reoptimization=False,
        num_procs=1,
        suppress_openmp=False,
    )
    add_reshaped_to_dict(checksums, "unopt_mean_MAEs", array_(unopt_mean_MAE))
    add_checksum_to_dict(checksums, "unopt_std_MAEs", array_(unopt_std_MAE), nstack_checksums=1)
    # lastly, add a check for hyperparameters
    add_reshaped_to_dict(checksums, "hyperparameters", log_([model.sigma, model.l2reg_diag_ratio]))
    compare_or_create(
        checksums,
        benchmark_name,
        max_rel_difference=5.0e-5,
    )


def test_global_model():
    for shift_quantity in [False, True]:
        run_model_type_check(
            KRRModel,
            "global_model_shift_" + str(shift_quantity),
            model_kwargs={"representation_function": CMCalc, "shift_quantities": shift_quantity},
        )


def test_local_model():
    for shift_quantity in [False, True]:
        run_model_type_check(
            KRRLocalModel,
            "local_model_shift_" + str(shift_quantity),
            model_kwargs={
                "representation_function": FCHL19Calc,
                "shift_quantities": shift_quantity,
                "possible_nuclear_charges": qm7_nuclear_charges,
            },
        )


def test_sorf_models():
    kwargs1 = {"shift_quantities": True}
    kwargs2 = {
        "shift_quantities": False,
        "npcas": 8,
        "use_rep_reduction": True,
        "test_mode": True,
    }
    for name, constructor, extra_kwargs in [
        ("global", SORFModel, {"representation_function": CMCalc}),
        (
            "local",
            SORFLocalModel,
            {
                "possible_nuclear_charges": np.array([1, 6, 7, 8, 16]),
                "representation_function": FCHL19Calc,
            },
        ),
    ]:
        for i, kwargs in enumerate([kwargs1, kwargs2]):
            full_kwargs = {"nfeatures": 1024, "rng": default_rng_(8), **kwargs, **extra_kwargs}
            run_model_type_check(
                constructor, name + "_sorf_model_" + str(i), model_kwargs=full_kwargs
            )


if __name__ == "__main__":
    test_global_model()
    test_local_model()
    test_sorf_models()
