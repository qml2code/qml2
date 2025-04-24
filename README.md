# qml2

Code repo for convenient usage of methods developed in the [`chemspacelab`](https://github.com/chemspacelab) group. Successor to [`qmlcode/qml`](https://github.com/qmlcode/qml) repo that also draws a lot of inspiration from [`qmllib`](https://github.com/qmlcode/qmllib) and [`qml-lightning`](https://github.com/nickjbrowning/qml-lightning) repos.

## :wrench: Installation

Install with:

   ```bash
   pip install .
   ```
or, if `makefile` is installed,

   ```bash
   make install
   ```
**NOTE:** Using FJK representation or constructing adaptive basis sets additionally requires installing `pyscf` package via

   ```bash
   pip install pyscf
   ```

## :clipboard: Testing

To check that the installed repo works correctly run

   ```bash
   make test
   ```
**NOTE:** The command assumes that `python` environmental variable points towards a valid executable. If you use an environment alias change definition of the `python` variable in the beginning of the Makefile.

## :books: API documentation

API documentation can be generated with [Doxygen](https://www.doxygen.nl/) by running

   ```bash
   make docs
   ```
or, if Makefile is not installed, running
   ```bash
   python doxygen/run_doxygen.py
   ```

This will create `manual.html` file that can be opened with an Internet browser.

## :computer: Environmental variables

### Calculation management

`NUMBA_NUM_THREADS` - since the code is written in Numba OpenMP parallelization is mostly controlled via this variable.

`QML2_NUM_PROCS` - number of processes spawned by parts of the code parallelized via `python.multiprocessing` (training set representations in model-related classes, `pyscf` calculations made by `OML_Compound_list` attribute calls). For limiting number of OpenMP threads spawned in turn by these processes use suppress options (such as `KRRModel` class's `training_reps_suppress_openmp` option). Also see `parallelization.set_default_num_procs`.

### Experimental

`QML2_DEFAULT_JIT` - setting to `NUMBA` (default) or `TORCH` (both are case insensitive) determines whether Numba or TorchScript JIT compilation is used. Also see `jit_interfaces.set_default_jit`.

### Debugging

`QML2_DEBUG` - if 1 add `debug=True` to all Numba `@njit` instances.

`QML2_SKIP_JIT` - if 1 do not use JIT.

`QML2_SKIP_JIT_FAILURES` - if 1 TorchScript does not terminate when encountering uncompilable part of the code.

## :handshake: Contributing

We use several packages that maximize code readability, listed in `requirements-dev.txt`; hence should you decide to commit make sure you have a conda environment you are prepared to modify. Having, for example, created a fresh conda environment named `qml2dev` with

   ```bash
   conda create --name qml2dev
   conda activate qml2dev
   ```
one prepares the environment and the pre-commit scripts in the repository with

   ```bash
   make dev-setup
   ```
This allows automatic formatting/readability checks for the committed code. It is also possible to enforce adherence to [Conventional Commits](https://www.conventionalcommits.org/) [format](https://github.com/pvdlg/conventional-changelog-metahub) of commit messages inside your fork with

   ```bash
   make conventional-commits
   ```

## :pager: Slack channel

If you plan to use `qml2` regularly consider contacting the developers to join their Slack channel for quick means of communication.

## :lock: Developer repository

`qml2-dev` is the private "developer" version of `qml2` repository created to allow package developers creation and easy management of private forks, as well as sharing messier code drafts. If you interact with the code and one of the package authors on a regular basis consider asking them to be added; though in general we try to prevent `qml2` from lagging behind `qml2-dev` for more than two months.

## :name_badge: Developer list

* [Konstantin Karandashev](https://github.com/kvkarandashev) (<kvkarandashev@gmail.com>)
* [Stefan Heinen](https://github.com/heini-phys-chem)
* [Danish Khan](https://github.com/dkhan42)
* [Jan Weinrech](https://github.com/janweinreich)
