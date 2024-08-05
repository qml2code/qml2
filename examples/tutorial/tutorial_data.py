import os
import random
import shutil
import tarfile

import numpy as np

from qml2 import Compound, CompoundList


def get_energies(filename, key="dft"):
    """Returns a dictionary with heats of formation for each xyz-file."""

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    energies = dict()

    for line in lines:
        tokens = line.split()

        xyz_name = tokens[0]
        hof = float(tokens[1])
        dftb = float(tokens[2])

        if key == "dft":
            energies[xyz_name] = hof

        elif key == "delta":
            energies[xyz_name] = hof - dftb
        else:
            energies[xyz_name] = hof

    return energies


# Ensure all necessary data is copied from qml2/tests/test_data
def copy_data():
    if os.path.isfile("hof_qm7.txt") and os.path.isdir("qm7/"):
        return
    print("CREATING EXERCISE FILES")
    shutil.copyfile("../../tests/test_data/hof_qm7.txt", "hof_qm7.txt")
    compressed_xyzs = tarfile.open("../../tests/test_data/qm7.tar.gz")
    compressed_xyzs.extractall(".")
    print("EXERCISE FILES CREATED")


copy_data()

qm7_dft_energy = get_energies("hof_qm7.txt", key="dft")
qm7_delta_energy = get_energies("hof_qm7.txt", key="delta")

compounds = CompoundList([Compound(xyz="qm7/" + f) for f in sorted(os.listdir("qm7/"))])

for mol in compounds:
    mol.properties = qm7_dft_energy[mol.name]
    mol.properties2 = qm7_delta_energy[mol.name]

random.seed(4)
random.shuffle(compounds)

energy_pbe0 = np.array([mol.properties for mol in compounds])
energy_delta = np.array([mol.properties2 for mol in compounds])
