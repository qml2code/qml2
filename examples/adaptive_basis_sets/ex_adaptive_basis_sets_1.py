# NOTE: QML2's implementation of adaptive basis sets differs from the one in arXiv:2404.16942, but is still largely numerically equivalent.
import tarfile

from qml2.orb_ml import OML_Compound

basis = "STO-3G"  # "6-31G"


def get_xyz(xyz_name, optimize_ao_rescalings=False):
    # We'll be using xyz's compressed in qml2/tests.
    with tarfile.open("../../tests/test_data/qm7.tar.gz") as tar, tar.extractfile(
        "qm7/" + xyz_name
    ) as xyz:
        return OML_Compound(xyz=xyz, optimize_ao_rescalings=optimize_ao_rescalings, basis=basis)


# Import methane's xyz.
methane_comp = get_xyz("0001.xyz", optimize_ao_rescalings=True)

# Run the pySCF calculations.
methane_comp.run_calcs()
print("Energy of methane with optimized basis:", methane_comp.e_tot)
# Print AO rescalings found to be optimal.
methane_opt_resc = methane_comp.ao_rescalings
print("ao rescalings:", methane_opt_resc)

# Re-run them without adaptive basis.
methane_comp.mats_created = False  # ignore previousy calculated results
methane_comp.ao_rescalings = None
methane_comp.optimize_ao_rescalings = False
methane_comp.run_calcs()
print("Without optimized basis:", methane_comp.e_tot)

# Import ethane next.
ethane_comp = get_xyz("0002.xyz")
# Run pySCF calculations and check the resulting energy.
ethane_comp.run_calcs()
print("Ethane energy unoptimized basis:", ethane_comp.e_tot)

# Training a proper model for rescalings is beyond this example,
# but let's see how the energy changes if we assign basis parameters
# previously observed for methane.
ethane_rescalings = []
for nc in ethane_comp.nuclear_charges:
    if nc == 6:
        resc = methane_opt_resc[0]
    else:
        resc = methane_opt_resc[-1]
    ethane_rescalings.append(resc)

ethane_comp.ao_rescalings = ethane_rescalings
ethane_comp.mats_created = False
ethane_comp.run_calcs()
print("Ethane energy with basis optimal for methane:", ethane_comp.e_tot)
