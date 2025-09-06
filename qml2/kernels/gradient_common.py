from ..data import nCartDim

atom_force_dim = nCartDim


def change_atom_force_dim(new_atom_force_dim):
    """
    Changes the assumed dimensionality of the force acting on an atom. (Introduced as a quickfix to allow alchemical derivatives.)

    WARNING: should be used before any kernel functions or model classes are imported from elsewhere in QML2.
    """
    global atom_force_dim
    atom_force_dim = new_atom_force_dim
