# Ensures data used in the exercises is copied.
from tutorial_data import copy_data

from qml2.compound import Compound

copy_data()

# Create the compound object mol from the file qm7/0001.xyz which happens to be methane
mol = Compound(xyz="qm7/0001.xyz")

# Generate and print a coulomb matrix for compound with 5 atoms
mol.generate_coulomb_matrix(size=5)
print(mol.representation)


# Print other properties stored in the object
print(mol.coordinates)
print(mol.atomtypes)
print(mol.nuclear_charges)
