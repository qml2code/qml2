from ..compound_list import CompoundList
from ..parallelization import parallelized_inheritance
from .oml_compound import ASE2OML_Compound, OML_Compound, OML_Slater_pair


@parallelized_inheritance(
    "run_calcs", "generate_orb_reps", base_classes=[OML_Compound, OML_Slater_pair]
)
class OML_CompoundList(CompoundList):
    """The class was created to allow easy embarrassing parallelization of operations with lists of OML_Compound or OML_Slater_pair objects."""

    def mats_savefile2temp_calc_dirs(self):
        # not_pairs = isinstance(self[0], OML_Compound)
        # TODO: Double-check it works correctly. Why did I consider not_pairs here?
        for i in range(len(self)):
            self[i].temp_calc_dir = self[i].mats_savefile[:-4]


def OML_Compound_list_from_xyzs(xyz_files, **oml_comp_kwargs):
    return OML_CompoundList(
        [
            OML_Compound(xyz=xyz_file, mats_savefile=xyz_file, **oml_comp_kwargs)
            for xyz_file in xyz_files
        ]
    )


def OML_Slater_pair_list_from_xyzs(xyz_files, **slater_pair_kwargs):
    return OML_CompoundList(
        [
            OML_Slater_pair(xyz=xyz_file, mats_savefile=xyz_file, **slater_pair_kwargs)
            for xyz_file in xyz_files
        ]
    )


def OML_Slater_pair_list_from_xyz_pairs(xyz_file_pairs, **slater_pair_kwargs):
    if ["second_oml_comp_kwargs" in slater_pair_kwargs]:
        add_second_oml_comp_kwargs = slater_pair_kwargs["second_oml_comp_kwargs"]
    else:
        add_second_oml_comp_kwargs = {}

    return OML_CompoundList(
        [
            OML_Slater_pair(
                xyz=xyz_file_pair[0],
                mats_savefile=xyz_file_pair[0],
                second_oml_comp_kwargs={
                    "xyz": xyz_file_pair[1],
                    "mats_savefile": xyz_file_pair[1],
                    **add_second_oml_comp_kwargs,
                },
                **slater_pair_kwargs
            )
            for xyz_file_pair in xyz_file_pairs
        ]
    )


def OML_CompoundList_from_ASEs(ase_list, **oml_comp_kwargs):
    return OML_CompoundList([ASE2OML_Compound(ase_obj, **oml_comp_kwargs) for ase_obj in ase_list])
