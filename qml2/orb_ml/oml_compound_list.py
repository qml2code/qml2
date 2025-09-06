from ..compound_list import CompoundList
from ..parallelization import parallelized_inheritance
from .oml_compound import ASE2OML_Compound, OML_Compound, OML_Slater_pair, OML_Slater_pairs


@parallelized_inheritance(
    "run_calcs",
    "generate_orb_reps",
    base_classes=[OML_Compound, OML_Slater_pair, OML_Slater_pairs],
)
class OML_CompoundList(CompoundList):
    """
    Acts like a list of OML_Compound, OML_Slater_pair, or OML_Slater_pairs objects, but has two additional attributes:
    - `run_calcs(**kwags, **parallel_kwargs)` - perform `run_calcs(**kwargs)` for each list member;
    - `generate_orb_reps(**kwags, **parallel_kwargs)` - perform `generate_orb_reps(**kwargs)` for each list member.
    The `parallel_kwargs` are keyword arguments `num_procs` and `fixed_num_threads` as defined in `.parallelization.embarrassingly_parallel`.
    """

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
