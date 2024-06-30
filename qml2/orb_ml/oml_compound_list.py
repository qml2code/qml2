from ..parallelization import embarrassingly_parallel
from .oml_compound import ASE2OML_Compound, OML_Compound, OML_Slater_pair
from .representations import OML_rep_params


class OML_Compound_list(list):
    """The class was created to allow easy embarrassing parallelization of operations with lists of OML_compound objects."""

    def run_calcs(self, **emb_paral_kwargs):
        self.embarrassingly_parallelize(after_run_calcs, (), **emb_paral_kwargs)

    def generate_orb_reps(self, rep_params: OML_rep_params = OML_rep_params(), **emb_paral_kwargs):
        self.embarrassingly_parallelize(after_gen_orb_reps, rep_params, **emb_paral_kwargs)

    def embarrassingly_parallelize(
        self, func_in, other_args, num_procs=None, fixed_num_threads=None
    ):
        new_vals = embarrassingly_parallel(
            func_in,
            self,
            other_args,
            num_procs=num_procs,
            fixed_num_threads=fixed_num_threads,
        )
        for i in range(len(self)):
            self[i] = new_vals[i]

    def mats_savefile2temp_calc_dirs(self):
        # not_pairs = isinstance(self[0], OML_Compound)
        # TODO: Double-check it works correctly. Why did I consider not_pairs here?
        for i in range(len(self)):
            self[i].temp_calc_dir = self[i].mats_savefile[:-4]


#   Both functions are dirty as they modify the arguments, but it doesn't matter in this particular case.
def after_run_calcs(oml_comp):
    oml_comp.run_calcs()
    return oml_comp


def after_gen_orb_reps(oml_comp, rep_params):
    oml_comp.generate_orb_reps(rep_params)
    return oml_comp


def OML_Compound_list_from_xyzs(xyz_files, **oml_comp_kwargs):
    return OML_Compound_list(
        [
            OML_Compound(xyz=xyz_file, mats_savefile=xyz_file, **oml_comp_kwargs)
            for xyz_file in xyz_files
        ]
    )


def OML_Slater_pair_list_from_xyzs(xyz_files, **slater_pair_kwargs):
    return OML_Compound_list(
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

    return OML_Compound_list(
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


def OML_Compound_list_from_ASEs(ase_list, **oml_comp_kwargs):
    return OML_Compound_list(
        [ASE2OML_Compound(ase_obj, **oml_comp_kwargs) for ase_obj in ase_list]
    )
