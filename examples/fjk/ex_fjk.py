# FJK example (DOI:10.1063/5.0083301).
import tarfile

import numpy as np

from qml2.orb_ml import OML_Compound, OML_Slater_pair
from qml2.orb_ml.kernels import gaussian_kernel, gaussian_kernel_symmetric, rep_stddevs
from qml2.orb_ml.oml_compound_list import OML_CompoundList
from qml2.orb_ml.representations import OML_rep_params

# Get methane and ethane xyz files.
with tarfile.open("../../tests/test_data/qm7.tar.gz") as tar:
    methane_xyz_lines = tar.extractfile("qm7/0001.xyz").readlines()
    ethane_xyz_lines = tar.extractfile("qm7/0002.xyz").readlines()

# For machine learning of extensive properties create OML_Compound objects.
methane_comp = OML_Compound(xyz_lines=methane_xyz_lines)
ethane_comp = OML_Compound(xyz_lines=ethane_xyz_lines)

# For machine learning intensive properties create pairs of OML_Compound objects
# grouped into "Slater determinant pairs". Here we look at Slater determinant pairs
# created by vacating ground state's HOMO, used for HOMO energy calculation.
# NOTE: since vacating HOMO changes spin we need to specify using UHF instead of HF.
second_kwargs = {"used_orb_type": "HOMO_removed", "calc_type": "UHF"}
methane_pair = OML_Slater_pair(second_oml_comp_kwargs=second_kwargs, xyz_lines=methane_xyz_lines)
ethane_pair = OML_Slater_pair(second_oml_comp_kwargs=second_kwargs, xyz_lines=ethane_xyz_lines)

# Define parameters of the representation. We'll just set maximum angular momentum value
# as 1 since we'll be working with STO-3G.
rep_params = OML_rep_params(max_angular_momentum=1)

# Representations can be generated by calling generate_re attributes of OML_Compound, OML_Slater_pair,
# or OML_Compound_list instances. In the latter case execution is embarassingly parallalized over list members.
comp_list = OML_CompoundList([methane_comp, ethane_comp])
pair_list = OML_CompoundList([methane_pair, ethane_pair])

comp_list.generate_orb_reps(rep_params=rep_params)
pair_list.generate_orb_reps(rep_params=rep_params)

# Calculate and print kernels.
print("Kernels:")
for obj_list in [comp_list, pair_list]:
    # get reasonable sigma parameters; note that in DOI:10.1063/5.0083301 they were additionally rescaled.
    sigmas = rep_stddevs(obj_list)
    sigmas *= np.sqrt(len(sigmas))

    global_sigma = 0.5
    # calculate the kernel.
    kernel_asym = gaussian_kernel(obj_list, obj_list, sigmas, global_sigma)
    # same result, but with symmetric version of the function
    kernel_sym = gaussian_kernel_symmetric(obj_list, sigmas, global_sigma)
    print(kernel_asym)
    print(kernel_sym)
