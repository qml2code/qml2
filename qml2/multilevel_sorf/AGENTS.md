# qml2.multilevel_sorf

Notes on the parts of this module that the code does not state and that have cost debugging
time. Not a tour: read `models.py` and `sorf_calculation.py` for what things do, and
`tests/test_msorf_models.py` for working examples of every construction below.

Repository-wide conventions -- how to run the tests, why the benchmarks are checksums, what
`sanity_checks/` is for -- are in the `AGENTS.md` at the root.

## The two lists are a positional contract, and nothing checks it

`MultilevelSORF(function_definition_list, parameter_list)` takes a list of level NAMES and a
list of parameter DICTS, and pairs them by index. A parameter dict that belongs to the wrong
level is not an error -- it is a different model. There is no validation, and the failure
surfaces much later as a shape mismatch or, worse, as plausible numbers.

`get_parameter_list` in `tests/test_msorf_models.py` is the canonical builder: give it the
function list and it produces the matching parameter list, taking `resc_bounds` and sizes
from the representation calculator. Prefer it to writing the dicts by hand; the paper
scripts write them out explicitly, which is why they have to be edited in pairs.

Each level of nesting in the data is collapsed by one summation level. The FJK stack goes
atom contribution -> orbital -> compound (-> Slater pair), so a compound needs one more
`weighted_component_sum` than an orbital does, and a pair one more `component_sum` than a
compound. Getting this wrong produces a model that runs.

## Compile-time parameters: a third kind, new and unused so far

`parameter_list` shapes the INPUT OBJECTS -- sizes, counts, element-id tables -- read when the
jitclass instances are built. Hyperparameters arrive as an array at call time. Neither can supply a
**callable that the generated numba function closes over**, which is what you need in order to bake a
representation function into a level -- `qml2.representations.numba_calculators` supplies them, e.g.
`aSLATMJITCalculator(...).numba_calculator`. Swapping a level's activation function is the same shape
of problem. That is what compile-time parameters are for.

Spelling: a reserved key inside the level's ordinary `parameter_list` entry.

```python
function_definition_list = ["resize", "sorf", "component_sum"]
parameter_list = [
    {"output_size": 1024},
    {"nfeature_stacks": 4, "ntransforms": 3,
     "compilation_parameters": {"representation_function": aslatm_calc.numba_calculator}},
    {},
]
```

`MultilevelSORF.__init__` splits them out with `split_compilation_parameters` **before** anything
else reads `parameter_list`, so every existing consumer sees the dict it always saw, and stores the
parallel `self.compilation_parameter_list`. The routine getters (`get_routine_from_def`,
`get_copy_from_def`, `get_cpuest_from_def`) take it as an optional argument and thread it down the
recursion alongside the definition list, sliced the same way `create_input_object` already slices
`parameter_list`.

**The part that can fail silently, and the reason any of this needs care.** Routines are memoised in
`base_constructors/common.py`'s module globals under a name built from the definition list and
nothing else. Two models differing ONLY in a compile-time parameter would therefore share one
compiled routine, and the second would quietly run the first's baked-in function while every number
it produced looked plausible. So a level carrying compilation parameters contributes a token to that
name (`sorf` becomes `sorf_c1f3a09b2e`), and a level carrying none contributes exactly the name it
always did -- `get_name_from_definition_list(d) == get_name_from_definition_list(d, [{}, {}, {}])`
is asserted, because that identity is what keeps every pre-existing model bit-for-bit unchanged.

The token identifies a callable by `module.qualname`, not `repr`. `repr` of a function embeds its
`id()`, which differs between a run and its own restart, so the routine would be recompiled under a
fresh name every process.

**`module.qualname` cannot see what a CLOSURE baked in, and that is a known open hole.** Every
function a factory returns shares one qualname, so two constructors built by one factory over
different baked-in values -- two element lists, say -- token identically. A factory therefore
declares its baked-in values through `set_compilation_identity(func, *values)`, and `_canonical`
prefers them when present. `numba_constructors.get_el_aug_rep_constructor` does this with its
`sorted_elements`, so the el-aug-rep path is safe.

But the declaration is **opt-in**, and an undeclared closure still collides silently. Two facts bound
the risk. First, the collision is scoped to a PROCESS, not to a model: the cache is
`base_constructors/common.py`'s module `globals()`, which no amount of rebuilding a `MultilevelSORF`
clears, while separate processes never share it at all. Second, the project convention is **one model
type per process**, which is what every paper job does, so nothing currently run can hit it. A single
process building two models whose levels bake in different values is the case that breaks, and a
pytest process is the likeliest first place to meet it.

**TODO (design, deferred).** The root cause is that intermediate constructed functions are dumped
into module `globals()` rather than into a cache owned by something with a lifetime -- the model, or
an explicit registry object. The original reason for that choice is no longer remembered. Revisiting
it would retire this hole properly, along with the process-lifetime surprise, instead of asking each
factory to remember to declare. Until then: keep it one model type per process. A guard that raised
on an undeclared `<locals>` closure was considered and deliberately NOT added, to avoid hard-failing
callers over a case the convention already excludes.

**How a level opts in:** its constructor declares a `compilation_parameters` argument. That
declaration IS the registration -- there is no second registry to keep in step. A level whose
constructor does not declare it and is handed parameters anyway **raises**; dropping them silently
is the failure to avoid, since the caller would get a working model built with the default function
while believing its own had been baked in.

Note this is deliberately generic rather than keyed on level names. The original sketch was "if the
level is `conformer_representation`, pull the representation function out of the parameter list" --
the same thing happens here, but the generic machinery never mentions any level by name and the
level's own constructor reads whatever keys it wants out of the dict it is handed. One mechanism
serves the representation case and any other baked-in callable without a name check per use.

One asymmetry to know about: the COPY and CPUEST routines are not threaded, and `get_copy_from_def` /
`get_cpuest_from_def` do not take the list at all. Both act on the INPUT OBJECT, whose type the
definition list fixes, so neither can depend on a callable baked into a level's compute function.
Threading them also breaks outright -- their fallbacks are `cadd_simple` and `cpuest_add_simple`,
neither of which declares `compilation_parameters`, so `call_level_addition` refuses rather than drop
them silently. Consequence: nested copy routines keep their untokened names and existing cache
entries. Revisit only if a copy routine ever needs to know a baked-in callable.

CPUEST routines, by contrast, ARE threaded, and the reason is worth stating because the opposite was
written here first. A cpuest acts on the PROCESSED object, not the input object, so a level that
substitutes the processed object must substitute it there too. `cpuest_add_simple` forwards what it is
handed unchanged, which leaves the estimator chain below typed against a conformer instead of the
per-atom object -- a hard TypingError the first time anything asks for a CPU estimate, not the
load-balancing inaccuracy it was first described as. `calculate_representation` therefore registers
`cpuest_add_calculate_representation` in `cpuest_level_additions`.

That addition does NOT compute the representation, and the reason matters: a cpuest routine only ever
counts. `cpuest_resize_` returns 0.0 without reading its argument, `cpuest_add_single` adds 1 per call,
the summation levels iterate their components. So the level hands down a `(natoms, 1)` array of zeros
through the same constructor -- right type, right atom count -- and the count comes out identical.
Computing the real representation would be self-defeating at the sizes this level exists for:
estimating the cost of an aSLATM would cost an aSLATM. The representation's own cost is not counted
either; the estimate is used for relative balancing, and the nested count already scales with the atom
count that dominates it.

**Its first real consumer is `calculate_representation`**, in
`base_constructors/representations.py` -- see that module's docstring for where the level sits and
why. The mechanism, its cache-key invariant and its refusal behaviour are additionally pinned by
`tests/test_msorf_compilation_parameters.py`, which exercises delivery through a level registered only
for the duration of the test, so no real level's design is presumed there either.

## Rescaling regions must tile the feature vector

`rescaling` takes `resc_bounds`, which comes from
`ProcessedRepresentationListCalc.component_bounds`, and `calc_hyperparameter_num` returns
`resc_bounds.shape[0]` -- **one sigma per region**. Nothing verifies that the regions cover
the features.

This has already gone wrong. `fjk_rep_component_bounds` derived its regions from the number
of coupling matrices, which is right for FJK and wrong for FJD, where three dipole
components merge into one channel and the bond projection has no self block. FJD silently
got regions covering 22 of its 26 features and 4 sigmas instead of 5: no exception, just a
tail of features with no length scale of their own.
`sanity_checks/fjd_tblite/check_fjd_features.py` pins the coverage now.

## After `rescaling`, use `unscaled_sorf` -- not `sorf`

`sorf` is not a bare SORF transform. `add_sorf` is literally a `full_rescaling` wrapped around
`add_unscaled_sorf` (`base_constructors/sorf_related.py`), and `full_rescaling` divides the whole
output by one more hyperparameter (`base_constructors/rescalings.py`). So a `sorf` level carries a
sigma of its own.

Put it straight after `rescaling` and that sigma is **redundant**: `rescaling` has already divided
region *i* by its own sigma, so the feature that reaches the transform is scaled by the product
`sigma_i * sigma_global`. Any `(sigma_i, sigma_global)` describes the same model as
`(sigma_i * sigma_global, 1)` -- an exactly flat direction in hyperparameter space, which the
optimizer then has to wander along for nothing.

Use `["resize", "rescaling", "unscaled_sorf", ...]`, as `FJK_mixed_sorf_sdet_FreeSolv_full.py` and
the FJD scripts do. `sorf` is the right level where no `rescaling` precedes it -- after
`normalization`, say, which has no sigma of its own, or at a level that needs a length scale it
does not otherwise have.

## Everything in one kernel must share one `component_bounds`

`ProcessedRepresentationListCalc` accumulates `component_bounds` over the objects it
processes and asserts they agree. Two calculators produce two sets of bounds, and features
built against different bounds are not comparable even when their shapes match. Process
everything that will meet in a kernel through a single calculator.

## Quantity shifts: `simple_quantity_shift` is a modifier, not a switch

Three flags on `MultilevelSORFModel`, and they are not independent:

* `intensive_quantity_shift` adds a constant column to the shift basis -- an adjustment by
  the mean;
* `extensive_quantity_shift` adds one column per element holding that element's count --
  the dressed-atom, stoichiometry-dependent linear adjustment;
* `simple_quantity_shift` chooses HOW the coefficients over that basis are optimized:
  fitted directly to the training quantities when True, or against the model's
  cross-validation errors when False.

The False branch is a **hyperparameter** shift, and worth naming that way: the coefficients are
re-optimized inside `optimize_quantity_shifts` on every loss evaluation, i.e. once per sigma
setting ("Optimize quantity shifts as part of loss function calculation for given sigma values"),
so the shift is part of what the hyperparameter search explores. `simple_quantity_shift=True`
instead calls `simple_guess_quantity_shift` once and returns, independently of the kernel.

Which cross-validation the False branch uses is not fixed either -- it follows the split:

* no `training_group_ids` on `assign_training_set` -> `error_matrices_for_shifts`, leave-one-out;
* `training_group_ids` assigned (a scaffold split, say) -> `group_error_matrices_for_shifts`,
  leave-one-GROUP-out, using `group_stat_matrices` instead of `stat_factors`.

So "LOO shift" is the wrong label for a run with groups. Call it the hyperparameter shift.

`optimize_quantity_shifts` returns immediately unless one of the first two is True, so
`simple_quantity_shift=True` on its own applies **no shift at all**. Setting it without
also setting an intensive or extensive shift is the mistake to watch for.

Which of the first two to set follows the QUANTITY, not the representation:

* an **extensive** target -- one that grows with molecular size, like FreeSolv's hydration
  free energy -- wants `extensive_quantity_shift=True`, the dressed-atom shift with one
  column per element count;
* an **intensive** target -- one that does not, like LES's E(cation) - E(neutral) per
  molecule -- wants `intensive_quantity_shift=True`, the adjustment by the mean.

So every FreeSolv script in `paper_scripts-dev` is `extensive=True, intensive=False` and every
LES one is the reverse. This has already gone wrong once: the LES `charge_augmented` script
carried FreeSolv's extensive shift, fitting per-element counts to an energy difference that
does not scale with them.

## Hyperparameters

`MultilevelSORF.hyperparameter_initial_guesses(processed_objects)` produces usable sigmas
from the data, which is why SORF is easier to set up ad hoc than the standard kernels --
those need `rep_stddevs` over a set large enough to have variance in every component, and
return zeros for components that do not.

## Pickling

Objects in `sorf_calculation.py` and `models.py` hold numba typed structures that do not
pickle. `pickle.py`'s `Pickler` converts those attributes to dicts on the way out and back
on the way in; attributes listed in `deleted_attrs` are dropped rather than saved. A new
attribute holding a numba type needs registering there, or saving the model starts failing.

## Leave-one-group-out: one argument on `assign_training_set`

`assign_training_set(..., training_group_ids=...)` takes a **list of lists** -- each naming
the training-point indices of one group -- and switches the hyperparameter criterion from
leave-one-out to leave-one-GROUP-out. Every point still contributes exactly one held-out
error; it is just the error obtained with that point's whole group removed. The Ntrain
errors are then pooled and handed to the loss exactly as before, which is micro-averaging:
each MOLECULE weighted equally, and so each group in proportion to its size.

Nothing else changes name: the errors still live in `self.leaveoneout_errors`, because
everything downstream consumes them identically. Under grouping `self.stat_factors` is None
and `self.group_stat_matrices` holds one `(I - P_VV)` matrix per group instead, so code
reaching for `stat_factors` must branch.

The groups must **partition** the training set, and `init_training_group_ids` asserts it.
"One error per point" is otherwise undefined: a point in two groups gets written twice and a
point in none is left uninitialized, and neither shows up as anything but a slightly odd
loss.

Two things to know, neither an oversight:

* **The l2reg gradient refinement is worth switching off by hand.** `optimize_quantity_shifts_l2reg`
  finishes the global search with `scipy.optimize.minimize`. Under grouping each extra
  evaluation costs a Cholesky solve per group rather than one elementwise multiplication, so
  a re-evaluation is no longer close to free -- but it is still cheap for small groups and
  still buys accuracy, so grouping does NOT disable it for you. Pass
  `l2reg_local_opt_method=None` when you want it off. That off-switch did NOT work before: None
  was passed straight to `minimize`, and scipy reads it as "choose a method for me" and runs
  anyway.
* **The memory-lean hyperparameter-gradient path refuses.** `calc_loss_function_hyperparameter_grad`
  in `sorf_calculation.py` is jit-compiled around the leave-one-out stat FACTORS and has no
  way to take the stat matrices, so under grouping it raises rather than quietly computing a
  leave-one-out gradient for a leave-one-group-out loss. The requirement is therefore just to
  keep `sd_reuse_Z_derivatives=True` (the default), which materializes the Z derivatives and
  goes through `group_loss_der_wrt_features`.

  Every optimizer asks for that route by itself, so none of them needs special handling:
  `run_sd_hyperparameter_optimization_cycle` always passes `return_Z_matrices=True`, and
  `get_ninv_loss_function_wders` (gradient-based BOSS) requests it whenever a grouping is
  assigned -- deliberately only then, since the ungrouped path is exactly what the leaner
  route exists for and forcing materialization there would raise its memory use. So steepest
  descent, gradient-free BOSS and gradient-based BOSS all work under grouping; the one
  configuration that is refused is `sd_reuse_Z_derivatives=False`.

The invariant worth keeping: with every point its own group, leave-one-group-out IS
leave-one-out, and the two must agree to the last bit despite sharing almost no code.
`sanity_checks/leaveoneout_MAEs/check_msorf_group_loss.py` pins that, and it is what catches
a mis-wiring.

## The l2reg pass returns early, and later methods depend on its side effects

`get_loss_function_ln_l2reg(ln_l2reg, gradient=False)` returns at the end of its
`if not gradient:` block, BEFORE setting `self.error_loss_error_ders` and (under grouping)
`self.group_loss_multipliers`. Those two are then read by
`get_loss_function_ln_hyperparameters` when it builds the hyperparameter gradient. So a
cached attribute crosses a method boundary and only one of the three call sites populates it:

| call site | gradient | populates the gradient attributes |
| --- | --- | --- |
| `global_optimize_1D(ninv_f(...))` global search | False | no |
| `minimize(...)` local refinement, via `..._wder` | True | yes |
| the "finalize calculations" call | now True | yes |

The finalize call used to default to `gradient=False`, which made `minimize` the only
populating caller -- so switching the refinement off with `l2reg_local_opt_method=None` left
the attributes unset and the next hyperparameter gradient raised `AttributeError`. Neither is
initialized in `__init__` (unlike `leaveoneout_errors` and `error_loss`), and this bit the
leave-one-out path exactly as hard as the grouped one. Fixed by passing `gradient=True` at
finalize, which also makes them correspond to `ln_l2reg_final_min` rather than to whatever
point scipy last probed.

Two things that make this class of bug easy to reintroduce:

* The loss VALUE is identical either way -- every loss class returns the same number from
  `calc_wders` as from `__call__` (checked across MAE/MSE/RMSE and the whole self-consistent
  Huber/LogCosh family). So a wrong `gradient` flag never shows up as a wrong loss; only the
  gradient path notices, and only later.
* Adding an early return, or an off-switch that skips a call, can silently remove a side
  effect something else relies on. Before doing either, grep the attributes the skipped code
  WRITES, not just what it returns.

## Two different element lists: the representation's and the shift's

`possible_nuclear_charges` given to the representation calculators and the element basis of the
extensive ("dressed atom") shift are separate objects, and confusing them is easy:

| | set from | changes with the training set? |
| --- | --- | --- |
| representation `possible_nuclear_charges` | fixed by the caller, over ALL molecules | no -- the feature layout must be stable |
| shift basis (`training_set_possible_nuclear_charges`) | `get_possible_nuclear_charges(self.training_nuclear_charges)` | yes, rebuilt per `assign_training_set` |

`MultilevelSORFModel` holds no model-wide `possible_nuclear_charges` at all; the shift machinery
never consults the representation's list. So each learning-curve point fits dressed-atom
coefficients for exactly the elements ITS OWN subset contains, which is the intended behaviour --
a 50-molecule point must not borrow element information from the full training set.

The consequence worth spelling out: an element the representation supports but the training
subset never sees gets **no column**, rather than a column fitted to zero. The two are
indistinguishable in the output -- `element_counts` iterates over the basis and silently ignores
charges outside it, so at prediction a test molecule's atoms of an unseen element contribute zero
to its shift and nothing raises -- but they are different mechanisms, and only one of them exists
here. Do not go looking for an expansion step that pads the coefficient vector with zeros.

Note the opposite convention next door: `qml2.utils.get_element_ids_from_sorted` resolves charges
THROUGH a basis with `searchsorted_wexception` and DOES raise on an unknown element. Shifts fail
softly, element ids fail loudly; do not reason from one to the other. A rare element missing from
a small subset therefore shows up as a quiet bias in that subset's shift, not as a crash.

## Testing the model: pass `rng` AND `test_mode=True`

The SORF diagonals are random, so a model built twice is two different models. Any check
that compares two model objects, or that finite-differences anything, needs `rng=` -- and
that alone is not enough.

`test_mode=True` replaces the sign matrix `sign(r - 0.5)` with `rand_sign_polynom`, a
continuous quartic through the boundary (`qml2/kernels/sorf.py`). With the hard sign, a draw
sitting near 0.5 flips an entire feature vector on a floating-point wobble. For a
finite-difference derivative check that is fatal, because the two displaced evaluations can
be of *different* models; switching it on took one such check from 2e-8 to 2e-10 agreement.

A related trap in writing those checks: an inequality that holds "obviously" may hold only
for the draw you tried. Asserting that a grouped loss exceeds the leave-one-out loss looks
safe -- holding out three points at a time is harder -- but it is not an invariant, and it
flipped the moment `test_mode` changed the draw. Assert what is invariant (the errors move,
the singleton case reduces exactly) and leave the rest as printed diagnostics.

## Sanity-check scripts need `PYTHONPATH`, or they silently test the installed qml2

The scripts under `sanity_checks/` are run from their own directory, and the test conda envs
have a non-editable `qml2` in `site-packages`. The repo is on `sys.path` only when the cwd
happens to be the repo root, so from `sanity_checks/leaveoneout_MAEs/` a plain
`python check_*.py` imports the INSTALLED package:

```
cwd=/tmp        -> .../envs/qml2_test1/lib/python3.12/site-packages/qml2/__init__.py
cwd=<repo root> -> <repo>/qml2/__init__.py
```

Run them as `PYTHONPATH=<repo root> python check_*.py`. This bit once already, and the loud
version is the lucky one: a new keyword gave `assign_training_set() got an unexpected
keyword argument 'training_group_ids'`. When the signature has NOT changed -- editing a
function body, or renaming something consistently on both sides -- there is no error at all
and the check quietly validates the stale install instead. Two consequences worth keeping in
mind when a check passes suspiciously easily: a genuinely broken edit can look green, and a
green result says nothing about the working tree unless the import path was set.

Related: a check that pokes an attribute on the model (`model.<param> = value`) cannot fail
loudly if that parameter gets renamed -- Python creates a new attribute rather than raising.
Such a check has to assert on the resulting BEHAVIOR, not on the assignment going through.

## Which routine to reach for, and what it assumes

The loss machinery is layered, and the layers differ in what they hold fixed. Picking the wrong
one is the most common way to measure something other than what was intended.

| routine | varies | holds fixed | prerequisite |
| --- | --- | --- | --- |
| `init_Z_matrix_decomposition` | -- | -- | sets `Z_U`/`Z_singular_values`/`Z_Vh` from a weighted Z matrix; everything below reads them off self |
| `get_loss_function_ln_l2reg(ln_l2reg, gradient=)` | l2reg | features, hyperparameters | the decomposition above; `error_loss_function` must not be None |
| `get_loss_function_ln_hyperparameters(x, gradient=)` | hyperparameters | nothing -- it re-optimizes l2reg AND the shifts internally | an assigned training set |
| `leaveoneout_loss_der_wrt_features` / `group_loss_der_wrt_features` | features | l2reg, shifts | the state a `gradient=True` l2reg call leaves behind |
| `optimize_quantity_shifts_l2reg` | l2reg + shifts | features | as above; calls the decomposition itself |
| `optimize_hyperparameters` | everything | -- | dispatches on `hyperparameter_optimizer` |

Two consequences worth knowing before differentiating anything:

* **`get_loss_function_ln_hyperparameters` is not a function of the hyperparameters alone.** It
  runs the inner l2reg optimization on every call, so its value depends on where that
  optimization lands. With `l2reg_global_opt_method="grid"` the inner optimum is a STEP function
  of the hyperparameters, so a finite-difference interval straddling a jump yields a spurious
  difference -- large enough to flip a component's sign. With BOSS it wanders stochastically
  unless `boss_seed` is set. Neither is a gradient error; the envelope theorem makes the total
  derivative equal the partial one only when the inner optimum is located consistently.
* **For a deterministic derivative check, hold l2reg fixed.** There is no entry point for this,
  but the pieces are public: set the hyperparameters, build and decompose the Z matrix, call
  `get_loss_function_ln_l2reg(ln_l2reg, gradient=True)` at a fixed ABSOLUTE ln_l2reg (fixing
  `l2reg_diag_ratio` instead would let l2reg drift with the hyperparameters), then call the
  feature-gradient routine and multiply by `self.hyperparameters` for the ln chain rule. That is
  what `sanity_checks/multilevel_sorf/hyperparameter_gradients/check_gradients.py` does, and it agrees with
  central differences to ~1e-4 across shift settings, importance multipliers and grouping --
  where the same comparison through the optimized-l2reg loss appears to fail by ~4e-2. That
  4e-2 is not an expected scale: the discrepancy is however far the inner optimum jumps, so it
  is unbounded. `sanity_checks/multilevel_sorf/full_hyperparameter_optimization/param_der_fd_check.py`
  differences the optimized-l2reg loss on QM7 and shows O(1) disagreement in 2 of its 4
  configurations, uncorrelated with the importance-multiplier and grouping paths (each appears
  once in a pass and once in a failure) -- which is the signature of the artefact rather than of
  a defect in either path. That script stays useful as the end-to-end smoke test and for its
  l2reg line, which is a real check and should sit at 1e-10; treat its hyperparameter lines as
  uninformative.

State the gradient routines depend on, all populated by a `gradient=True` l2reg call and NOT by
a `gradient=False` one: `error_loss_error_ders`, `reproduced_quantities`, `stat_factors`,
`transformed_inv_K_Z`, `transformed_alphas_rhs`, `eigenvalue_multipliers`,
`shifted_training_weighted_quantities`, and under grouping `group_loss_multipliers` and
`group_stat_matrices`.

Scripts under `sanity_checks/` are not run by CI and rot; `git log -S "<old name>" -- qml2/`
recovers what a vanished method became, and `git show <commit>^:<path>` shows what its body did.
Anything that builds representations needs `if __name__ == "__main__":`, because qml2 sets
`default_pool_start_method = "spawn"` in `qml2/parallelization.py` regardless of the
interpreter's own default (`change_default_pool_start_method("fork")` opts out).

## A fresh process spends minutes in JIT compilation

MSORF compiles a great deal of numba per `function_definition_list`, and the cost lands
before the first loss value. A check that appears to hang usually has not started computing.
Budget minutes, not seconds, for any harness timeout; within one process later model builds
are comparatively cheap.

## Small traps when driving the loss directly

Bypassing `optimize_hyperparameters` to call the loss yourself hits four things the
signatures do not say:

* `resize`'s `output_size` must be **at least the representation width** and a power of two.
  FCHL19 over two elements is 168 wide, so 256 -- not 64. Too small does not raise anything
  legible; numba reports `SystemError: CPUDispatcher(...) returned a result with an exception
  set`, which says nothing about sizes.
* `get_loss_function_ln_l2reg` wants an **array**, not a float: it runs its argument through
  `flatten_to_scalar` for BOSS's sake, and that indexes `.shape`.
* `error_loss_function` is **not** an `__init__` argument. It is built by
  `init_default_error_loss` / `construct_error_loss_function` (MSE by default) on the
  hyperparameter-optimization path; driving the loss directly means setting it yourself.
* `l2reg_global_opt_method` defaults to `"boss"`. Checks that only care which branch runs
  should set `"grid"` and a small `l2reg_total_iterpts`.
