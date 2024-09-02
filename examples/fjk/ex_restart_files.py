# The script exemplifies how QML2 stores ab initio data used in FJK representation
# for potential later reuse
import tarfile

from qml2.orb_ml import OML_Compound, OML_Slater_pair

# Get methane xyz.
with tarfile.open("../../tests/test_data/qm7.tar.gz") as tar:
    methane_xyz_lines = tar.extractfile("qm7/0001.xyz").readlines()

methane_comp1 = OML_Compound(xyz_lines=methane_xyz_lines, mats_savefile="temp.pkl")
# Note that how this creates temp.pkl.
methane_comp1.run_calcs()

# Instead of redoing calculations for another instance, QML2 will import the results from temp.pkl
methane_comp2 = OML_Compound(xyz_lines=methane_xyz_lines, mats_savefile="temp.pkl")
methane_comp2.run_calcs()

# Setting "mats_savefile" not ending in "pkl" make the code use the string as a prefix, which
# is extended into the filename with some extra information.

methane_comp3 = OML_Compound(xyz_lines=methane_xyz_lines, mats_savefile="methane")

second_kwargs = {"used_orb_type": "HOMO_removed", "calc_type": "UHF"}
methane_pair = OML_Slater_pair(
    second_oml_comp_kwargs=second_kwargs, xyz_lines=methane_xyz_lines, mats_savefile="methane"
)
# QML2 will reuse the previously stored ground state results in the calculation.
methane_pair.run_calcs()
