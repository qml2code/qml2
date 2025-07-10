from ..data import nCartDim
from ..ensemble import Ensemble, WeightedCompound, base_compound_class_dict
from ..jit_interfaces import (
    OptionalGenerator_,
    array_,
    dim0float_array_,
    dint_,
    dot_,
    dtype_,
    empty_,
    int_,
    jit_,
    l2_norm_,
    mean_,
    randint_array_from_rng_,
    randint_from_rng_,
    random_array_from_rng_,
    random_from_rng_,
    standard_normal_array_from_rng_,
    zeros_,
)
from ..utils import l2_norm_sq_dist, weighted_array

default_possible_nuclear_charges = array_([1, 3])


@jit_
def lj_energy_force(nuclear_charges, coordinates):
    energy = 0.0
    force = zeros_(coordinates.shape)
    natoms = nuclear_charges.shape[0]
    for atom_id in range(natoms):
        atom_coords = coordinates[atom_id]
        for other_atom_id in range(atom_id):
            other_atom_coords = coordinates[other_atom_id]
            inv_d2 = 1.0 / l2_norm_sq_dist(atom_coords, other_atom_coords)
            prefac = nuclear_charges[atom_id] * nuclear_charges[other_atom_id]
            energy += (inv_d2**6 - inv_d2**3) * prefac
            radial_force = (
                (atom_coords - other_atom_coords) * (12 * inv_d2**7 - 6 * inv_d2**4) * prefac
            )
            force[atom_id] += radial_force
            force[other_atom_id] -= radial_force
    return energy, force


@jit_
def too_close(coord, other_coord_arr, min_dist):
    min_dist_sq = min_dist**2
    for other_coord in other_coord_arr:
        if l2_norm_sq_dist(coord, other_coord) < min_dist_sq:
            return True
    return False


@jit_
def rand_unit_sphere(nCartDim: int_ = nCartDim, rng: OptionalGenerator_ = None):
    vec = standard_normal_array_from_rng_((nCartDim,), rng=rng)
    vec /= l2_norm_(vec)
    return vec


@jit_
def random_lj_configuration(
    natoms: int_,
    min_dist: dim0float_array_ = array_(0.75),
    max_dist: dim0float_array_ = array_(1.5),
    nCartDim: int_ = nCartDim,
    rng: OptionalGenerator_ = None,
):
    """
    Generate random natoms LJ particles not closer to each other than min dist.
    """
    output = zeros_((natoms, nCartDim))
    for i in range(1, natoms):
        while too_close(output[i], output[:i], min_dist):
            seed_id = int(random_from_rng_(rng=rng) * i)
            output[i] = output[seed_id] + rand_unit_sphere(rng=rng) * (
                min_dist + (max_dist - min_dist) * random_from_rng_(rng=rng)
            )
    return output


@jit_
def random_nuclear_charges(
    natoms: int,
    possible_nuclear_charges=default_possible_nuclear_charges,
    rng: OptionalGenerator_ = None,
    dint_: dtype_ = dint_,
):
    nelements = len(possible_nuclear_charges)
    element_ids = randint_array_from_rng_(0, nelements, (natoms,), rng=rng)
    nuclear_charges = empty_((natoms,), dtype=dint_)
    for atom_id in range(natoms):
        nuclear_charges[atom_id] = possible_nuclear_charges[element_ids[atom_id]]
    return nuclear_charges


@jit_
def random_lj_molecule(
    natoms_min: int_,
    natoms_max: int_,
    possible_nuclear_charges=default_possible_nuclear_charges,
    min_dist: dim0float_array_ = array_(0.75),
    max_dist: dim0float_array_ = array_(1.5),
    rng: OptionalGenerator_ = None,
):
    natoms = randint_from_rng_(natoms_min, natoms_max + 1, rng=rng)
    nuclear_charges = random_nuclear_charges(
        natoms, possible_nuclear_charges=possible_nuclear_charges, rng=rng
    )
    return nuclear_charges, random_lj_configuration(
        natoms, min_dist=min_dist, max_dist=max_dist, rng=rng
    )


@jit_
def lj_dipole(nuclear_charges, coordinates):
    """
    A somewhat well-behaved quantity for tests and proof-of-concept benchmarks.
    """
    av_ncharge = mean_(nuclear_charges)
    atomic_charges = av_ncharge - nuclear_charges
    dipole_vector = dot_(atomic_charges, coordinates)
    return l2_norm_(dipole_vector)


# everything related to generating test ensembles.
def lj_dipole_from_wcompound(wcompound: WeightedCompound):
    """
    Find lj_dipole for a weighted compound
    """
    compound = wcompound.compound
    return lj_dipole(compound.nuclear_charges, compound.coordinates)


def lj_energy_from_wcompound(wcompound: WeightedCompound):
    compound = wcompound.compound
    energy, _ = lj_energy_force(compound.nuclear_charges, compound.coordinates)
    return energy


class RandomLJEnsemble(Ensemble):
    def __init__(
        self,
        natoms_min: int_,
        natoms_max: int_,
        possible_nuclear_charges=default_possible_nuclear_charges,
        min_dist: dim0float_array_ = array_(0.75),
        max_dist: dim0float_array_ = array_(1.5),
        num_conformers_min=8,
        num_conformers_max=16,
        num_conformer_generations=1,
        r_cut=None,
        rng=None,
    ):
        self.rng = rng
        self.natoms = randint_from_rng_(natoms_min, natoms_max + 1, rng=self.rng)
        self.num_conformer_generations = (1,)
        self.num_conformers_min = num_conformers_min
        self.num_conformers_max = num_conformers_max
        self.nuclear_charges = random_nuclear_charges(
            self.natoms, possible_nuclear_charges=possible_nuclear_charges, rng=self.rng
        )
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.base_class_name = "Compound"
        self.r_cut = r_cut
        self.num_conformer_generations = num_conformer_generations
        # TODO: make a separate routine
        self.base_compound_class = base_compound_class_dict[self.base_class_name]
        self.savefile_prefix = None
        self.conformers = None
        self.processed_conformers = None
        self.filtered_conformers = None
        self.compound_kwargs = {}

    def conformer_generation(self):
        num_conformers = randint_from_rng_(
            self.num_conformers_min, self.num_conformers_max + 1, rng=self.rng
        )
        boltzmann_weights = random_array_from_rng_(size=(num_conformers,), rng=self.rng)
        output = weighted_array()
        for bw in boltzmann_weights:
            coordinates = random_lj_configuration(
                self.natoms, min_dist=self.min_dist, max_dist=self.max_dist, rng=self.rng
            )
            compound = self.create_compound(self.nuclear_charges, coordinates)
            output.append(WeightedCompound(compound, bw, None))
        return output

    def get_nuclear_charges(self):
        return self.nuclear_charges

    def mean_lj_dipole(self):
        return self.mean_quant(lj_dipole_from_wcompound)

    def mean_lj_energy(self):
        return self.mean_quant(lj_energy_from_wcompound)
