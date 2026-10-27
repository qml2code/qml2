import csv
import random
import tarfile

import numpy as np

from qml2 import Compound
from qml2.models.loss_functions import MAE
from qml2.models.sorf import SORFLocalModel
from qml2.representations.calculators import SLATMCalculator


def main():
    xyzs = []
    energies = []

    training_set_size = 501
    test_set_size = 1000
    num_mols = training_set_size + test_set_size

    with open("../../tests/test_data/hof_qm7.txt") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        all_rows = list(reader)
        random.shuffle(all_rows)
        for row in all_rows[:num_mols]:
            xyzs.append(row[0])
            energies.append(float(row[1]))

    energies = np.array(energies)
    all_nuclear_charges = []

    compounds = []
    with tarfile.open("../../tests/test_data/qm7.tar.gz") as tar:
        for xyz_name in xyzs:
            xyz = tar.extractfile(xyz_name)
            comp = Compound(xyz=xyz)
            compounds.append(comp)
            all_nuclear_charges.append(comp.nuclear_charges)

    train_compounds = compounds[:training_set_size]
    test_compounds = compounds[training_set_size:]

    train_quantities = energies[:training_set_size]
    test_quantities = energies[training_set_size:]

    slatm_calculator = SLATMCalculator(all_nuclear_charges, local=True)
    possible_nuclear_charges = np.unique(np.concatenate(all_nuclear_charges))
    print("Nuclear charges found:", possible_nuclear_charges)

    # using "shift_quantites=True" means using dressed atom approach; requires defining `possible_nuclear_charges` though.
    # keyword arguments for hyperparameter optimization are:
    # ln_l2reg_diag_ratio_bounds : bounds in which logarithm of l2reg_diag_ratio is optimized.
    # ln_sigma_shift_bounds : bounds in which logarithm of sigma is optimized, shifted by logarithm of initial guess.
    # l2reg_total_iterpts (int): total number of function calls as l2reg is optimized (e.g. with BOSS).
    # sigma_total_iterpts (int): total number of function calls as sigma is optimized (e.g. with BOSS).
    # hyp_1D_opt_method (str): method for 1D optimization used for hyperparameters (default is "boss" - BOSS optimization, also supports "grid").
    # NOTE: nfeatures might need to be increased if aSLATM representation is too large.
    model = SORFLocalModel(
        shift_quantities=True,
        possible_nuclear_charges=possible_nuclear_charges,
        representation_function=slatm_calculator,
        nfeatures=32768,
        ntransforms=3,
        ln_l2reg_diag_ratio_bounds=[-25.0, 0.0],
        ln_sigma_shift_bounds=[-10.0, 10.0],
        l2reg_total_iterpts=32,
        sigma_total_iterpts=32,
        hyp_1D_opt_method="boss",
    )

    model.train(training_compounds=train_compounds, training_quantities=train_quantities)

    print("Optimized sigma:", model.sigma)
    print("Optimized l2reg divided by average kernel element:", model.l2reg_diag_ratio)

    predictions = model.predict_from_compounds(test_compounds)
    print("Prediction MAE:", MAE()(predictions - test_quantities))
    print("Test set quantity STD:", np.std(test_quantities))


if __name__ == "__main__":
    main()
