"""GFN2 basis data: tblite's STO-nG expansion tables, pinned.

tblite builds its AO basis by expanding one Slater function per shell into ``ngauss``
Gaussians (``basis/slater.f90``). The per-element inputs to that expansion -- the shell
labels, the Slater exponents ``zeta`` and ``ngauss`` -- are exported at runtime by the
library itself (see :func:`..gfn2_element_parameters`), but the STO-nG expansion tables
are compiled-in constants: the installed wheel ships a shared object and no Fortran, and
the C API exposes no accessor for them. They are therefore transcribed here.

This copy is PINNED, not derived. That is deliberate: if upstream ever revises the
tables, a pinned copy makes the overlap sanity check fail loudly instead of letting the
reconstructed basis drift silently away from the one tblite computed with.

To check the copy against a tblite source tree::

    python -m qml2.orb_ml.tblite_interface.data /path/to/tblite

which re-extracts the tables from ``src/tblite/basis/slater.f90`` and reports any
disagreement (exit status 1 if the two differ, 0 if they match).

The tables are indexed ``[ngauss][ityp][primitive]``, with ``ityp`` derived from the
principal and azimuthal quantum numbers exactly as tblite does it -- see
:func:`sto_ng_index`.
"""

import re
import sys
from pathlib import Path

import numpy as np

#: Angular momentum of each shell letter, for reading tblite's shell labels.
ANGULAR_MOMENTUM = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

#: Path of the expansion tables inside a tblite checkout.
SLATER_SOURCE = "src/tblite/basis/slater.f90"

#: Number of (n, l) function types tabulated per expansion order (Fortran ``nf``).
N_FUNCTION_TYPES = 15


# fmt: off
STO_NG_ALPHA = {
    1: [
        [0.2709498091],
        [0.1012151084],
        [0.05296881757],
        [0.03264600274],
        [0.02216912938],
        [0.1759666885],
        [0.09113614253],
        [0.05578350235],
        [0.03769845216],
        [0.1302270363],
        [0.07941656339],
        [0.05352200793],
        [0.1033434062],
        [0.06952785407],
        [0.08565417784],
    ],
    2: [
        [0.8518186635, 0.1516232927],
        [0.1292278611, 0.04908584205],
        [0.6694095822, 0.05837135094],
        [0.2441785453, 0.04051097664],
        [0.1213425654, 0.03133152144],
        [0.4323908358, 0.1069439065],
        [0.1458620964, 0.05664210742],
        [0.0619005268, 0.02648418407],
        [0.2691294191, 0.03980805011],
        [0.2777427345, 0.08336507714],
        [0.1330958892, 0.05272119659],
        [0.06906014388, 0.03399457777],
        [0.2006693538, 0.068653849],
        [0.1156094555, 0.04778940916],
        [0.1554531559, 0.05854079811],
    ],
    3: [
        [2.227660584, 0.4057711562, 0.1098175104],
        [2.581578398, 0.1567622104, 0.06018332272],
        [0.5641487709, 0.06924421391, 0.03269529097],
        [0.2267938753, 0.04448178019, 0.02195294664],
        [0.1080198458, 0.04408119382, 0.0261081181],
        [0.9192379002, 0.2359194503, 0.08009805746],
        [2.692880368, 0.1489359592, 0.0573958504],
        [0.485969222, 0.07430216918, 0.03653340923],
        [0.2127482317, 0.0472964862, 0.02604865324],
        [0.5229112225, 0.1639595876, 0.06386630021],
        [0.1777717219, 0.0804064735, 0.03949855551],
        [0.491335295, 0.07329090601, 0.0359420929],
        [0.3483826963, 0.1249380537, 0.05349995725],
        [0.1649233885, 0.07487066646, 0.03735787219],
        [0.2545432122, 0.1006544376, 0.04624463922],
    ],
    4: [
        [5.216844534, 0.954618276, 0.2652034102, 0.08801862774],
        [11.61525551, 2.000243111, 0.1607280687, 0.06125744532],
        [1.513265591, 0.4262497508, 0.07643320863, 0.03760545063],
        [0.3242212833, 0.1663217177, 0.05081097451, 0.028290666],
        [0.8602284252, 0.11890502, 0.03446076176, 0.01974798796],
        [1.798260992, 0.4662622228, 0.164371862, 0.06543927065],
        [1.853180239, 0.1915075719, 0.08655487938, 0.04184253862],
        [1.49260788, 0.4327619272, 0.07553156064, 0.03706272183],
        [0.3962838833, 0.1838858552, 0.04943555157, 0.02750222273],
        [0.9185846715, 0.2920461109, 0.118756889, 0.05286755896],
        [1.995825422, 0.182346128, 0.08197240896, 0.04000634951],
        [0.4230617826, 0.08293863702, 0.04590326388, 0.02628744797],
        [0.5691670217, 0.2074585819, 0.09298346885, 0.04473508853],
        [0.2017831152, 0.1001952178, 0.0544700663, 0.03037569283],
        [0.3945205573, 0.1588100623, 0.07646521729, 0.03898703611],
    ],
    5: [
        [11.30563696, 2.071728178, 0.5786484833, 0.1975724573, 0.07445271746],
        [8.984956862, 1.673710636, 0.1944726668, 0.08806345634, 0.04249068522],
        [4.275877914, 1.132409433, 0.4016256968, 0.0773237062, 0.03800708627],
        [2.980263783, 0.3792228833, 0.1789717224, 0.0500211036, 0.02789361681],
        [0.7403763257, 0.1367990863, 0.09135301779, 0.03726907315, 0.02241490836],
        [3.320386533, 0.8643257633, 0.3079819284, 0.1273309895, 0.05606243164],
        [6.466803859, 1.555914802, 0.1955925255, 0.08809647701, 0.04234835707],
        [1.091977298, 0.3719985051, 0.08590019352, 0.0478650386, 0.0273047999],
        [0.3422168934, 0.16650999, 0.05443732013, 0.03367775277, 0.02091949042],
        [1.539033958, 0.4922090297, 0.2029756928, 0.09424112917, 0.04569058269],
        [1.522122079, 0.2173041823, 0.1084876577, 0.05836797641, 0.03206682246],
        [0.970294647, 0.3603270196, 0.08668717752, 0.04833708379, 0.02751899341],
        [0.8925960415, 0.327758912, 0.1492869962, 0.07506099109, 0.03892475795],
        [1.670735676, 0.2072477219, 0.1024709357, 0.05537913898, 0.03072866652],
        [0.5895429375, 0.239334378, 0.1172646904, 0.06254074479, 0.03411243214],
    ],
    6: [
        [23.10303149, 4.235915534, 1.185056519, 0.4070988982, 0.1580884151, 0.06510953954],
        [27.68496241, 5.077140627, 1.42678605, 0.2040335729, 0.09260298399, 0.04416183978],
        [3.273031938, 0.9200611311, 0.3593349765, 0.08636686991, 0.04797373812, 0.02724741144],
        [3.232838646, 0.3605788802, 0.1717905487, 0.05277666487, 0.03163400284, 0.01874093091],
        [1.410128298, 0.5077878915, 0.1847926858, 0.1061070594, 0.03669584901, 0.0221355843],
        [5.868285913, 1.530329631, 0.5475665231, 0.2288932733, 0.1046655969, 0.04948220127],
        [5.077973607, 1.34078694, 0.2248434849, 0.1131741848, 0.06076408893, 0.03315424265],
        [2.389722618, 0.7960947826, 0.341554138, 0.08847434525, 0.04958248334, 0.02816929784],
        [3.778623374, 0.3499121109, 0.1683175469, 0.05404070736, 0.03328911801, 0.02063815019],
        [2.488296923, 0.7981487853, 0.331132749, 0.1559114463, 0.07877734732, 0.04058484363],
        [4.63423942, 1.341648295, 0.2209593028, 0.1101467943, 0.0590419037, 0.03232628887],
        [0.8820520428, 0.3410838409, 0.0920430884, 0.05472831774, 0.0339120283, 0.02108227374],
        [1.357718039, 0.5004907278, 0.2296565064, 0.1173146814, 0.06350097171, 0.03474556673],
        [1.33409684, 0.2372312347, 0.1269485744, 0.07290318381, 0.04351355997, 0.02598071843],
        [0.8574668996, 0.3497184772, 0.172791706, 0.09373643151, 0.05340032759, 0.03057364464],
    ],
}

STO_NG_COEFF = {
    1: [
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
    ],
    2: [
        [0.4301284983, 0.6789135305],
        [0.7470867124, 0.2855980556],
        [-0.1529645716, 1.05137011],
        [-0.3046656896, 1.146877294],
        [-0.5114756049, 1.307377277],
        [0.4522627513, 0.6713122642],
        [0.5349653144, 0.5299607212],
        [0.8743116767, 0.1513640107],
        [-0.103422701, 1.033376378],
        [0.4666137923, 0.6644706516],
        [0.4932764167, 0.5918727866],
        [0.6539405185, 0.3948945302],
        [0.4769346276, 0.6587383976],
        [0.4856637346, 0.6125980914],
        [0.4848298074, 0.6539381621],
    ],
    3: [
        [0.1543289673, 0.5353281423, 0.4446345422],
        [-0.05994474934, 0.5960385398, 0.4581786291],
        [-0.1782577972, 0.8612761663, 0.2261841969],
        [-0.3349048323, 1.056744667, 0.125666168],
        [-0.6617401158, 0.7467595004, 0.7146490945],
        [0.1623948553, 0.5661708862, 0.4223071752],
        [-0.01061945788, 0.5218564264, 0.5450015143],
        [-0.06147823411, 0.6604172234, 0.3932639495],
        [-0.1389529695, 0.8076691064, 0.2726029342],
        [0.168659606, 0.5847984817, 0.4056779523],
        [0.2308552718, 0.6042409177, 0.2595768926],
        [-0.02010175008, 0.5899370608, 0.465844596],
        [0.1737856685, 0.5973380628, 0.3929395614],
        [0.1909729355, 0.6146060459, 0.3059611271],
        [0.1780980905, 0.6063757846, 0.3828552923],
    ],
    4: [
        [0.0567524208, 0.260141355, 0.5328461143, 0.2916254405],
        [-0.01198411747, -0.05472052539, 0.5805587176, 0.4770079976],
        [-0.03295496352, -0.1724516959, 0.7518511194, 0.3589627317],
        [-0.1120682822, -0.2845426863, 0.8909873788, 0.3517811205],
        [0.01103657561, -0.5606519023, 1.179429987, 0.1734974376],
        [0.05713170255, 0.2857455515, 0.5517873105, 0.2632314924],
        [-0.01434249391, 0.2755177589, 0.5846750879, 0.2144986514],
        [-0.006035216774, -0.06013310874, 0.64515182, 0.411792382],
        [-0.01801459207, -0.1360777372, 0.7533973719, 0.3409304859],
        [0.05799057705, 0.3045581349, 0.5601358038, 0.2432423313],
        [-0.00281670262, 0.2177095871, 0.6058047348, 0.2717811257],
        [-0.02421626009, 0.3937644956, 0.5489520286, 0.1190436963],
        [0.05902730589, 0.3191828952, 0.5639423893, 0.2284796537],
        [0.0917426883, 0.4023496947, 0.49374321, 0.1254001522],
        [0.0601048425, 0.3309738329, 0.5655207585, 0.2171122608],
    ],
    5: [
        [0.02214055312, 0.113541152, 0.3318161484, 0.4825700713, 0.1935721966],
        [-0.01596349096, -0.05685884883, 0.3698265599, 0.5480512593, 0.1472634893],
        [-0.00392035885, -0.04168430506, -0.163744099, 0.7419373723, 0.3724364929],
        [0.001513948997, -0.07316801518, -0.3143703799, 0.9032615169, 0.3294210848],
        [0.01375523371, -0.3097344179, -0.3199192259, 1.084547038, 0.3345288361],
        [0.02079051117, 0.1235472099, 0.3667738886, 0.483493029, 0.1653444074],
        [-0.002329023747, -0.01357395221, 0.2632185383, 0.5880427024, 0.2242794445],
        [-0.01143929558, -0.06322651538, 0.4398907721, 0.5245859166, 0.1017072253],
        [-0.03113958289, -0.1374007017, 0.5573881018, 0.48554281, 0.06605423564],
        [0.02020869128, 0.1321157923, 0.3911240346, 0.4779609701, 0.1463662294],
        [-0.003673711876, 0.1167122499, 0.4216476416, 0.4547673415, 0.1037803318],
        [-0.003231527611, -0.02434931372, 0.3440817054, 0.5693674376, 0.1511340183],
        [0.01999839052, 0.139542744, 0.4091508237, 0.4708252119, 0.1328082566],
        [-0.0007301193568, 0.08414991343, 0.3923683153, 0.5040033146, 0.13289793],
        [0.01998085812, 0.146038405, 0.4230565459, 0.4635699665, 0.1226411691],
    ],
    6: [
        [0.00916359628, 0.04936149294, 0.1685383049, 0.3705627997, 0.4164915298, 0.1303340841],
        [-0.004151277819, -0.02067024148, -0.05150303337, 0.3346271174, 0.5621061301, 0.1712994697],
        [-0.006775596947, -0.05639325779, -0.1587856086, 0.5534527651, 0.501535102, 0.07223633674],
        [0.001374817488, -0.08666390043, -0.3130627309, 0.7812787397, 0.4389247988, 0.02487178756],
        [0.002695439582, 0.01850157487, -0.09588628125, -0.520067356, 1.08761949, 0.3103964343],
        [0.007924233646, 0.05144104825, 0.189840006, 0.4049863191, 0.4012362861, 0.1051855189],
        [-0.00332992984, -0.0141948834, 0.163939577, 0.4485358256, 0.390881305, 0.07411456232],
        [-0.001665913575, -0.01657464971, -0.05958513378, 0.4053115554, 0.5433958189, 0.1204970491],
        [0.0001163246387, -0.02920771322, -0.1381051233, 0.5706134877, 0.476880814, 0.06021665516],
        [0.007283828112, 0.05386799363, 0.2072139149, 0.4266269092, 0.3843100204, 0.08902827546],
        [-0.0004749842876, -0.003566777891, 0.1108670481, 0.415964693, 0.4621672517, 0.1081250196],
        [-0.004097377019, -0.02508271857, 0.2648458555, 0.5097437054, 0.2654483467, 0.02623132212],
        [0.006930234381, 0.05634263745, 0.2217065797, 0.4411388883, 0.3688112625, 0.07787514504],
        [-0.0009486751531, 0.04624275998, 0.2373699784, 0.4589112231, 0.3205010548, 0.05077063693],
        [0.006729778096, 0.0587414517, 0.2339955227, 0.4512983737, 0.3552053926, 0.06974153145],
    ],
}
# fmt: on

# (2l-1)!! table, tblite's `dfactorial`, used in the primitive normalization.
DFACTORIAL = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0]


def sto_ng_index(n: int, l: int) -> int:
    """1-based ``ityp`` for a shell, as in ``slater_to_gauss_array``'s integer hack.

    ityp: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
       n: 1 2 3 4 5 2 3 4 5  3  4  5  4  5  5
       l: 0 0 0 0 0 1 1 1 1  2  2  2  3  3  4
    """
    if l == 0:
        return n
    if l == 1:
        return 4 + n
    if l == 2:
        return 7 + n
    if l == 3:
        return 9 + n
    if l == 4:
        return 10 + n
    raise ValueError(f"no tabulated expansion for l={l}")


def slater_to_gauss(n: int, l: int, zeta: float, ngauss: int, normalize: bool = False):
    """``(alpha, coeff)`` of the STO-nG expansion, mirroring tblite's ``slater_to_gauss``.

    Exponents scale as ``zeta**2``; the coefficients are the tabulated values, optionally
    carrying the primitive normalization ``N = (2a/pi)^(3/4) sqrt(4a)^l / sqrt((2l-1)!!)``.

    ``normalize=False`` is what PySCF wants: its basis input takes contraction
    coefficients over NORMALIZED primitives, so handing it the normalized form would
    apply the factor twice.
    """
    ityp = sto_ng_index(n, l)
    alpha = np.array(STO_NG_ALPHA[ngauss][ityp - 1]) * zeta**2
    coeff = np.array(STO_NG_COEFF[ngauss][ityp - 1], dtype=float)
    if normalize:
        coeff = (
            coeff
            * (2.0 / np.pi * alpha) ** 0.75
            * np.sqrt(4.0 * alpha) ** l
            / np.sqrt(DFACTORIAL[l])
        )
    return alpha, coeff


def _fortran_array_literal(text: str, name: str):
    """Flat list of the floats assigned to Fortran parameter array ``name``.

    Bracket-matched rather than regexed to the first ``]``: the tables are written as
    ``reshape([...], shape(...))``, so a non-nesting match truncates the array.
    """
    match = re.search(rf"::\s*{name}\s*\([^)]*\)\s*=\s*(reshape\()?\s*\[", text)
    if match is None:
        return None
    position = match.end()
    depth = 1
    collected = []
    while depth:
        char = text[position]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                break
        collected.append(char)
        position += 1
    literal = re.sub(r"!.*", "", "".join(collected))  # strip the ! 1s / ! 2s labels
    literal = literal.replace("&", " ")  # and the line continuations
    numbers = re.findall(r"[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?", literal)
    return [float(n.replace("d", "e").replace("D", "e")) for n in numbers]


def extract_sto_ng_tables(checkout):
    """Re-extract the expansion tables from a tblite source tree.

    Args:
        checkout: path to a tblite checkout (the directory holding ``src/tblite``).

    Returns:
        ``(alpha, coeff)``, each ``{ngauss: [[primitive, ...], ...]}`` indexed the same
        way as the pinned tables in this module.
    """
    source = Path(checkout) / SLATER_SOURCE
    if not source.is_file():
        raise FileNotFoundError(f"no {SLATER_SOURCE} under {checkout}")
    text = source.read_text()

    alpha_tables, coeff_tables = {}, {}
    for ngauss in range(1, 7):
        flat_alpha = _fortran_array_literal(text, f"pAlpha{ngauss}")
        if flat_alpha is None:
            continue
        expected = ngauss * N_FUNCTION_TYPES
        if len(flat_alpha) != expected:
            raise ValueError(
                f"pAlpha{ngauss}: parsed {len(flat_alpha)} values, expected {expected}"
            )
        flat_coeff = _fortran_array_literal(text, f"pCoeff{ngauss}")
        if flat_coeff is None:  # ng == 1 carries no coefficient table
            flat_coeff = [1.0] * expected
        elif len(flat_coeff) != expected:
            raise ValueError(
                f"pCoeff{ngauss}: parsed {len(flat_coeff)} values, expected {expected}"
            )
        # Fortran reshape fills the first index fastest, so [primitive, ityp] lives at
        # flat position ityp * ngauss + primitive.
        alpha_tables[ngauss] = [
            [flat_alpha[t * ngauss + g] for g in range(ngauss)] for t in range(N_FUNCTION_TYPES)
        ]
        coeff_tables[ngauss] = [
            [flat_coeff[t * ngauss + g] for g in range(ngauss)] for t in range(N_FUNCTION_TYPES)
        ]
    if not alpha_tables:
        raise ValueError(f"no pAlpha* tables found in {source}")
    return alpha_tables, coeff_tables


def compare_with_checkout(checkout):
    """Differences between the pinned tables and a tblite checkout's.

    Returns a list of human-readable discrepancies; empty means the copy is current.
    """
    alpha_tables, coeff_tables = extract_sto_ng_tables(checkout)
    problems = []
    for label, pinned, extracted in (
        ("alpha", STO_NG_ALPHA, alpha_tables),
        ("coeff", STO_NG_COEFF, coeff_tables),
    ):
        missing = set(pinned) - set(extracted)
        added = set(extracted) - set(pinned)
        if missing:
            problems.append(f"{label}: checkout lacks ngauss={sorted(missing)}")
        if added:
            problems.append(f"{label}: checkout adds ngauss={sorted(added)}")
        for ngauss in sorted(set(pinned) & set(extracted)):
            for ityp, (here, there) in enumerate(zip(pinned[ngauss], extracted[ngauss]), start=1):
                if here != there:
                    problems.append(
                        f"{label}: ngauss={ngauss} ityp={ityp} differs\n"
                        f"    pinned:   {here}\n"
                        f"    checkout: {there}"
                    )
    return problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(f"usage: python -m {__spec__.name} /path/to/tblite", file=sys.stderr)
        return 2
    checkout = argv[0]
    try:
        problems = compare_with_checkout(checkout)
    except (FileNotFoundError, ValueError) as error:
        print(f"could not read the tables: {error}", file=sys.stderr)
        return 2
    if problems:
        print(f"PINNED TABLES DIFFER from {checkout}:")
        for problem in problems:
            print(f"  {problem}")
        return 1
    orders = ", ".join(str(n) for n in sorted(STO_NG_ALPHA))
    print(f"pinned STO-nG tables match {checkout} (expansion orders: {orders})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
