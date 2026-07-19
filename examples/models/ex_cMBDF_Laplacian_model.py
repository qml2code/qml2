import csv
import random
import tarfile

import numpy as np

from qml2 import Compound
from qml2.kernels import local_dn_laplacian_kernel, local_dn_laplacian_kernel_symmetric
from qml2.models.krr import KRRLocalModel
from qml2.models.loss_functions import MAE
from qml2.representations.calculators import cMBDFCalculator


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

    slatm_calculator = cMBDFCalculator(all_nuclear_charges)
    possible_nuclear_charges = np.unique(np.concatenate(all_nuclear_charges))
    print("Nuclear charges found:", possible_nuclear_charges)

    # using "shift_quantites=True" means using dressed atom approach; requires defining `possible_nuclear_charges` though.
    model = KRRLocalModel(
        shift_quantities=True,
        possible_nuclear_charges=possible_nuclear_charges,
        representation_function=slatm_calculator,
        kernel_function=local_dn_laplacian_kernel,
        kernel_function_symmetric=local_dn_laplacian_kernel_symmetric,
    )

    model.train(training_compounds=train_compounds, training_quantities=train_quantities)

    print("Optimized sigma:", model.sigma)
    print("Optimized l2reg divided by average kernel element:", model.l2reg_diag_ratio)

    predictions = model.predict_from_compounds(test_compounds)
    print("Prediction MAE:", MAE()(predictions - test_quantities))
    print("Test set quantity STD:", np.std(test_quantities))


if __name__ == "__main__":
    main()
