import csv
import random
import tarfile

import numpy as np

from qml2 import Compound, CompoundList
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

    compounds = []
    with tarfile.open("../../tests/test_data/qm7.tar.gz") as tar:
        for xyz_name in xyzs:
            xyz = tar.extractfile(xyz_name)
            comp = Compound(xyz=xyz)
            compounds.append(comp)

    # calculate SLATM representations for all compounds.
    aslatm_calculator = SLATMCalculator(all_compounds=compounds, local=True)
    compounds = CompoundList(compounds)
    # note that the execution will be parallel
    compounds.generate_rep_from_func(aslatm_calculator)

    train_compounds = compounds[:training_set_size]
    test_compounds = compounds[training_set_size:]

    train_quantities = energies[:training_set_size]
    test_quantities = energies[training_set_size:]

    possible_nuclear_charges = np.unique(
        np.concatenate([comp.nuclear_charges for comp in compounds])
    )
    print("Nuclear charges found:", possible_nuclear_charges)

    # using "shift_quantites=True" means using dressed atom approach; requires defining `possible_nuclear_charges` though.
    # setting `representation_function=None` means that each `Compound` instance is represented by its pre-calculated `representation` attribute.
    # NOTE: nfeatures might need to be increased if aSLATM representation is too large.
    model = SORFLocalModel(
        shift_quantities=True,
        possible_nuclear_charges=possible_nuclear_charges,
        nfeatures=32768,
        ntransforms=3,
    )

    model.train(training_compounds=train_compounds, training_quantities=train_quantities)

    print("Optimized sigma:", model.sigma)
    print("Optimized l2reg divided by average kernel element:", model.l2reg_diag_ratio)

    train_sizes = [125, 250, 500]
    MAEs, MAE_stds = model.learning_curve(
        train_compounds,
        train_quantities,
        test_compounds,
        test_quantities,
        train_sizes,
        hyperparameter_reoptimization=False,
    )

    print("Learning curve:")
    for train_size, MAE_val, MAE_std in zip(train_sizes, MAEs, MAE_stds):
        print(train_size, MAE_val, MAE_std)


if __name__ == "__main__":
    main()
