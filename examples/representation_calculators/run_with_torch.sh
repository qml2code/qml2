#!/bin/bash
# by default the representation calculation example will use Numba+Numpy to calculate representations; this script show how to set up your environment to use TORCH instead

# specify new JIT compiler
export QML2_DEFAULT_JIT=TORCH
# make sure errors in other parts of the code do not shut down the code
export QML2_SKIP_JIT_FAILURES=1
# run the script
python check_representation_calculators.py
# note that the representation test scripts should run correctly too
for test_name in SLATM aSLATM cMBDF geometric_representations
do
	echo "Test name: $test_name"
	python ../../tests/test_$test_name.py
done
# In general though maintaining usability with both Torch is Numba for all functions is quite annoying, so 1. don't be surprised something does not work 2. if something doesn't work make it a repo issue
