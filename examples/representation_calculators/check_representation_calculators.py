import csv
import tarfile

from qml2 import Compound
from qml2.compound_list import CompoundList
from qml2.representations.calculators import (
    BoBCalculator,
    CMCalculator,
    FCHL19Calculator,
    SLATMCalculator,
    cMBDFCalculator,
)


def main():
    xyzs = []
    with open("../../tests/test_data/hof_qm7.txt") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        all_rows = list(reader)
        for row in all_rows[:50]:
            xyzs.append(row[0])

    compounds = []
    with tarfile.open("../../tests/test_data/qm7.tar.gz") as tar:
        for xyz_name in xyzs:
            xyz = tar.extractfile(xyz_name)
            comp = Compound(xyz=xyz)
            compounds.append(comp)

    all_nuclear_charges = [comp.nuclear_charges for comp in compounds]

    # initiate representation calculators
    SLATMCalc = SLATMCalculator(all_nuclear_charges)

    aSLATMCalc = SLATMCalculator(all_nuclear_charges, local=True)

    cMBDFCalc = cMBDFCalculator(all_nuclear_charges)

    global_cMBDFCalc = cMBDFCalculator(all_nuclear_charges, local=False)

    FCHL19Calc = FCHL19Calculator(all_nuclear_charges=all_nuclear_charges)

    BoBCalc = BoBCalculator(all_nuclear_charges=all_nuclear_charges)

    CMCalc = CMCalculator(all_nuclear_charges=all_nuclear_charges)

    # make some sample calculations
    for comp in compounds:
        nuclear_charges = comp.nuclear_charges
        coordinates = comp.coordinates
        rep1 = SLATMCalc(nuclear_charges, coordinates)
        rep2 = cMBDFCalc(nuclear_charges, coordinates)
        rep3 = FCHL19Calc.calc_from_compound(comp)
        rep4 = BoBCalc(nuclear_charges, coordinates)
        rep5 = CMCalc.calc_from_compound(comp)
        rep6 = aSLATMCalc.calc_from_compound(comp)
        rep7 = global_cMBDFCalc(nuclear_charges, coordinates)
        print(rep1.shape, rep2.shape, rep3.shape, rep4.shape, rep5.shape, rep6.shape, rep7.shape)

    # also an example of parallel representation evaluation
    compounds = CompoundList(compounds)
    compounds.generate_rep_from_func(SLATMCalc)
    print("Sample rep shape:", compounds[0].representation.shape)


if __name__ == "__main__":
    main()
