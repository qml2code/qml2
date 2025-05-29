# For processing data published at: https://iopscience.iop.org/article/10.1088/2632-2153/ad2f52
import ast
import csv

import numpy as np
from scipy.stats import sem

quant_name = "Oxidation potential SCE"

SMILES_name = "SMILES"

data_filename = "grouped_dataset_acetonitrile_neutral.csv"


def get_oxidation_potentials(containing_folder="./", data_filename=data_filename):
    full_data_filename = containing_folder + "/" + data_filename
    with open(full_data_filename, "r") as csv_input:
        reader = csv.DictReader(csv_input)
        data_dict = {}
        for d in reader:
            quant_vals = ast.literal_eval(d[quant_name])
            mean_val = np.mean(quant_vals)
            if len(quant_vals) == 1:
                val_stat_err = None
            else:
                val_stat_err = sem(quant_vals)
            SMILES = d[SMILES_name]
            data_dict[SMILES] = (mean_val, val_stat_err)
    return data_dict
