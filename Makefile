# K.Karandashev: Largely inspired by the makefile from https://github.com/qmlcode/qmllib
# I made it more lightweight though, giving the developer more leeway on what environment
# to base and test their commit in.

all: install

dev-env:
	pip install pre-commit

./.git/hooks/pre-commit: dev-env
	pre-commit install

dev-setup: dev-env ./.git/hooks/pre-commit

review: dev-setup
	pre-commit run --all-files

test-env:
	pip install pytest

test: test-env
	python -m pytest -rs ./tests

install:
	pip install .

clean:
	rm -Rf ./build
	rm -Rf ./dist
	rm -Rf ./qml2.egg-info
