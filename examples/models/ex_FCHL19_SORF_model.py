import csv
import random
import tarfile

import numpy as np

from qml2 import Compound
from qml2.models.loss_functions import MAE
from qml2.models.sorf import SORFLocalModel

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

train_compounds = compounds[:training_set_size]
test_compounds = compounds[training_set_size:]

train_quantities = energies[:training_set_size]
test_quantities = energies[training_set_size:]

# using "shift_quantites=True" means using dressed atom approach; requires defining `possible_nuclear_charges` though.
model = SORFLocalModel(shift_quantities=True, possible_nuclear_charges=np.array([1, 6, 7, 8, 16]))

model.train(training_compounds=train_compounds, training_quantities=train_quantities)

print("Optimized sigma:", model.sigma)
print("Optimized l2reg divided by average kernel element:", model.l2reg_diag_ratio)

predictions = model.predict_from_compounds(test_compounds)
print("Prediction MAE:", MAE()(predictions - test_quantities))
print("Test set quantity STD:", np.std(test_quantities))
