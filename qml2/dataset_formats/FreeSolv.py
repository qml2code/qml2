# Written for FreeSolv-0.52/0.51
import csv

data_filename = "database.txt"

# columns at which different value aspects are stored.
SMILES_id = 1
value_id = 3
error_id = 4
note_id = -1

default_error_note = (
    " Experimental uncertainty not presently available, so assigned a default value.  "
)


def get_free_solvation_energies(containing_folder="./", data_filename=data_filename):
    full_data_filename = containing_folder + "/" + data_filename
    csv_input = open(full_data_filename, "r")
    # skip first three lines
    for _ in range(3):
        _ = csv_input.readline()
    reader = csv.reader(csv_input, delimiter=";")
    data_dict = {}
    for l in reader:
        value = float(l[value_id])
        note = l[note_id]
        if note == default_error_note:
            error = None
        else:
            error = float(l[error_id])
        # cutting off the first blank space
        SMILES = l[SMILES_id][1:]
        if SMILES in data_dict:
            print("WARNING: duplicated", SMILES)
        data_dict[SMILES] = (value, error)
    return data_dict
