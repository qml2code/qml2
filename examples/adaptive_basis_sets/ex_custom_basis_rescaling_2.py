# NOTE: QML2's implementation of adaptive basis sets differs from the one in arXiv:2404.16942, but is still largely numerically equivalent.
import tarfile

from qml2.orb_ml import OML_Compound

basis = "6-31G"

# Get the compound corresponding to methane.
with tarfile.open("../../tests/test_data/qm7.tar.gz") as tar, tar.extractfile(
    "qm7/0001.xyz"
) as xyz:
    methane_comp = OML_Compound(xyz=xyz, basis=basis)
# run the calculations and save the energy
methane_comp.run_calcs()
energy_unopt = methane_comp.e_tot

# Let's print the basis.
methane_mol = methane_comp.generate_pyscf_mol()
print("Initial basis:")
for el, basis in methane_mol._basis.items():
    print(el + ":")
    for i, coeffs in enumerate(basis):
        print(i, coeffs)
# Assume that we want to optimize separately rescaling for each shell. The corresponding dictionary reads:
basis_rescaled_orbitals = {"C": [[0], [1, 3], [2, 4]], "H": [[0], [1]]}
# Assign the rescaling to methane_comp, enable basis optimization, then run pySCF calculations.
methane_comp.optimize_ao_rescalings = True
methane_comp.basis_rescaled_orbitals = basis_rescaled_orbitals
methane_comp.run_calcs()
new_methane_mol = methane_comp.generate_pyscf_mol()
print("Energy of methane with standard basis:", energy_unopt)
print("Energy of methane with optimized basis:", methane_comp.e_tot)
print("Found rescaling:", methane_comp.ao_rescalings)
print("Optimized basis:")
for el, basis in new_methane_mol._basis.items():
    print(el + ":")
    for i, coeffs in enumerate(basis):
        print(i, coeffs)
