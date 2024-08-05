# K.Karandashev: Largely inspired by the makefile from https://github.com/qmlcode/qmllib
# I made it more lightweight though, giving the developer more leeway on what environment
# to base and test their commit in.

# if in your environment "python" or "pip" are aliases for another command modify these
# lines accordingly.
python=python
pip=pip

all: install

dev-env:
	$(pip) install pre-commit

./.git/hooks/commit-msg: dev-env
	pre-commit install --hook-type commit-msg

./.git/hooks/pre-commit: dev-env
	pre-commit install

conventional-commits: ./.git/hooks/commit-msg

dev-setup: dev-env ./.git/hooks/pre-commit

review: dev-setup
	pre-commit run --all-files

test-env:
	$(pip) install pytest

test: test-env
	$(python) -m pytest -rs ./tests

install:
	$(pip) install .

clean:
	rm -Rf ./build
	rm -Rf ./dist
	rm -Rf ./qml2.egg-info
	rm -Rf ./tests/test_data/perturbed_qm7
