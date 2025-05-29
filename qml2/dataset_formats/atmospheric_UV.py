# for processing data published in https://arxiv.org/abs/2101.07301
from qml2.orb_ml.oml_compound import OML_Compound


def HOMO_en(xyz_name, **kwargs):
    oml_comp = OML_Compound(xyz=xyz_name, mats_savefile=xyz_name, **kwargs)
    oml_comp.run_calcs()
    return oml_comp.HOMO_en()


def LUMO_en(xyz_name, **kwargs):
    oml_comp = OML_Compound(xyz=xyz_name, mats_savefile=xyz_name, **kwargs)
    oml_comp.run_calcs()
    return oml_comp.LUMO_en()


def first_excitation(xyz_name, **kwargs):
    return LUMO_en(xyz_name, **kwargs) - HOMO_en(xyz_name, **kwargs)


def potential_energy(xyz_name, **kwargs):
    oml_comp = OML_Compound(xyz=xyz_name, mats_savefile=xyz_name, **kwargs)
    oml_comp.run_calcs()
    return oml_comp.e_tot


def blank_function(*args, **kwargs):
    return 0.0


quant_properties = {
    "ground_state_energy": (0, "Hartree", potential_energy),
    "S1_excitation": (1, "Hartree", first_excitation),
    "S1_oscillator_strength": (2, "Hartree", blank_function),
    "S2_excitation": (3, "Hartree", blank_function),
    "S2_oscillator_strength": (4, "Hartree", blank_function),
    "S3_excitation": (5, "Hartree", blank_function),
    "S3_oscillator_strength": (6, "Hartree", blank_function),
}


class Quantity:
    def __init__(self, quant_name):
        self.name = quant_name
        self.quant_pos = quant_properties[quant_name][0]
        self.dimensionality = quant_properties[quant_name][1]

    def extract_xyz(self, filename):
        file = open(filename, "r")
        l = file.readlines()[1]
        file.close()
        split_l = l.split()
        if len(split_l) <= self.quant_pos:
            return None
        return float(split_l[self.quant_pos])

    def OML_calc_quant(self, xyz_name, **kwargs):
        return quant_properties[self.name][2](xyz_name, **kwargs)

    def write_byprod_result(self, val, io_out):
        io_out.write(str(self.quant_pos) + " " + str(val) + "\n")
