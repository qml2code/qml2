# IMPORTANT: Note that for the QM9 total potential energy results what was actually predicted was internal energy at 0K with
# total potential energy used as baseline in the Delta-ML scheme.

from ..utils import checked_input_readlines, xyz_file_stochiometry

try:
    from ..orb_ml.oml_compound import OML_compound
except ImportError:
    pass

try:
    # If MOSAiCS (https://github.com/chemspacelab/mosaics) and RdKIT are installed these can be used to check that an xyz's coordinates agree with the SMILES.
    from mosaics.rdkit_utils import RdKitFailure, SMILES_to_egc
    from mosaics.valence_treatment import InvalidAdjMat
    from mosaics.xyz2graph import xyz2mol_extgraph
except ImportError:
    pass

atom_energies = {
    "H": -0.500273,
    "C": -37.846772,
    "N": -54.583861,
    "O": -75.064579,
    "F": -99.718730,
}
valence_electrons = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}


def HOMO_en(xyz_name, **kwargs):
    oml_comp = OML_compound(xyz=xyz_name, mats_savefile=xyz_name, **kwargs)
    oml_comp.run_calcs()
    return oml_comp.HOMO_en()


def LUMO_en(xyz_name, **kwargs):
    oml_comp = OML_compound(xyz=xyz_name, mats_savefile=xyz_name, **kwargs)
    oml_comp.run_calcs()
    return oml_comp.LUMO_en()


def HOMO_LUMO_gap(xyz_name, **kwargs):
    return LUMO_en(xyz_name, **kwargs) - HOMO_en(xyz_name, **kwargs)


def potential_energy(xyz_name, **kwargs):
    oml_comp = OML_compound(xyz=xyz_name, mats_savefile=xyz_name, **kwargs)
    oml_comp.run_calcs()
    return oml_comp.e_tot


quant_properties = {
    "Dipole moment": (6, "Debye"),
    "Isotropic polarizability": (7, "Bohr^3"),
    "HOMO eigenvalue": (8, "Hartree", HOMO_en),
    "LUMO eigenvalue": (9, "Hartree", LUMO_en),
    "HOMO-LUMO gap": (10, "Hartree", HOMO_LUMO_gap),
    "Electronic spacial extent": (11, "Bohr^2"),
    "Zero point vibrational energy": (12, "Hartree"),
    "Internal energy at 0 K": (13, "Hartree", potential_energy),
    "Internal energy at 298.15 K": (14, "Hartree"),
    "Enthalpy at 298.15 K": (15, "Hartree"),
    "Free energy at 298.15 K": (16, "Hartree"),
    "Heat capacity at 298.15 K": (17, "cal/(mol K)"),
    "Highest vibrational frequency": (18, "cm^-1"),
}


def standard_extract_xyz(filename, qm9_id, quant_name):
    file = open(filename, "r")
    lines = file.readlines()
    output = None
    if quant_name == "Highest vibrational frequency":
        first_line_passing = True
    for l in lines:
        lsplit = l.split()
        if quant_name == "Highest vibrational frequency":
            if first_line_passing:
                first_line_passing = False
                continue
            try:
                output = max(
                    [float(freq_str) for freq_str in lsplit]
                )  # this will fail for all lines but for the one with molecule number (first line) and frequencies
                break
            except ValueError:
                continue
        else:
            if lsplit[0] == "gdb":
                output = float(lsplit[qm9_id - 1])
                break
    file.close()
    return output


def atomization_energy(filename, stochiometry=None):
    if stochiometry is None:
        stochiometry = xyz_file_stochiometry(filename)
    baseline_name = "Internal energy at 0 K"
    baseline_qm9_id = quant_properties[baseline_name][0]
    output = standard_extract_xyz(filename, baseline_qm9_id, baseline_name)
    for atom_symbol, num_atoms in stochiometry.items():
        output -= atom_energies[atom_symbol] * num_atoms
    return output


def normalized_atomization_energy(filename):
    stochiometry = xyz_file_stochiometry(filename)
    normalization = 0
    for atom_symbol, num_atoms in stochiometry.items():
        normalization += num_atoms * valence_electrons[atom_symbol]
    return atomization_energy(filename, stochiometry=stochiometry) / normalization


special_extract_functions = {
    "Atomization energy": atomization_energy,
    "Normalized atomization energy": normalized_atomization_energy,
}


class Quantity:
    def __init__(self, quant_name):
        self.name = quant_name
        self.qm9_id = None
        self.dimensionality = None
        if self.name in special_extract_functions:
            self.special_extract_func = special_extract_functions[self.name]
        else:
            self.special_extract_func = None
            self.qm9_id = quant_properties[self.name][0]
            self.dimensionality = quant_properties[self.name][1]

    def extract_xyz(self, filename):
        if self.special_extract_func is None:
            return standard_extract_xyz(filename, self.qm9_id, self.name)
        else:
            return self.special_extract_func(filename)

    def extract_byprod_result(self, filename):
        file = open(filename, "r")
        lines = file.readlines()
        output = None
        for l in lines:
            lsplit = l.split()
            if int(lsplit[0]) == self.qm9_id:
                output = float(lsplit[1])
                break
        file.close()
        return output

    def OML_calc_quant(
        self,
        xyz_name,
        calc_type="HF",
        basis="sto-3g",
        dft_xc="lda,vwn",
        dft_nlc="",
        **other_kwargs
    ):
        return quant_properties[self.name][2](
            xyz_name,
            calc_type=calc_type,
            basis=basis,
            dft_xc=dft_xc,
            dft_nlc=dft_nlc,
            **other_kwargs
        )

    def write_byprod_result(self, val, io_out):
        io_out.write(str(self.qm9_id) + " " + str(val) + "\n")


def read_str_rep(xyz_input, offset):
    lines = checked_input_readlines(xyz_input)
    natoms = int(lines[0])
    str_rep_line_id = natoms + offset
    return lines[str_rep_line_id].split()[0]


def read_SMILES(xyz_input):
    return read_str_rep(xyz_input, 3)


def read_InChI(xyz_input):
    return read_str_rep(xyz_input, 4)


def xyz_SMILES_consistent(xyz_file):
    SMILES = read_SMILES(xyz_file)
    try:
        egc1 = SMILES_to_egc(SMILES)
    except InvalidAdjMat:
        return False
    except RdKitFailure:
        return False
    egc2 = xyz2mol_extgraph(xyz_file)
    if egc2 is None:  # happens if chemical graph couldn't be determined
        return False
    return egc1 == egc2
