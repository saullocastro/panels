# Changelog

## 0.6.13 (2026-09-18)

### Requirements

- `structsolve>=0.4.3`, with the new non-linear solvers (full Newton-Raphson,
  rewritten arc-length methods of Riks and Crisfield) and the verified sparse
  solver of `lb`, which could return wrong buckling loads without any warning.

### Breaking: limits of the integration domain

The attributes `Shell.x1, x2, y1, y2`, the physical limits of the integration
domain, default to `None` instead of the sentinels `x1 = y1 = -1` and
`x2 = y2 = +1`. A `None` limit is the corresponding edge of the shell, `0`,
`a` or `b`, and each limit is independent of the others. The new
`Shell.integration_limits()` returns the four limits as floats and raises
`ValueError` unless `0 <= x1 < x2 <= a` and `0 <= y1 < y2 <= b`. It is called
by `Shell._rebuild`, so an invalid limit fails when the shell is built.

Code that sets `x1 = -1` or `y1 = -1` to request the full domain now raises
`ValueError`, and must set `None` instead, or leave the default. There is no
backward compatibility for the old sentinels, because `+1` is also a valid
coordinate. For the same reason, code that sets `x2 = +1` or `y2 = +1` alone
to restore the full domain cannot be detected: it now integrates up to
`1.0`.

The old convention had two defects, both giving wrong results with no warning:

- `x2 = 1.0` (or `y2 = 1.0`) could not be told apart from the sentinel, so
  on a shell with `a > 1` the requested limit was replaced by the full
  domain. A change of 1e-9 in that limit changed the stiffness matrix by
  34 %. This was reachable through `StiffPanelBay.add_panel`: in a bay of
  width `b = 1.0` the last panel was integrated over the whole bay, counting
  its skin twice and giving a buckling load 4.9 % too low, and a panel edge
  at `y = 1.0` in a wider bay did the same to the first panel.
- A partial domain was recognized only when both limits of a pair departed
  from the sentinels, so a single limit, e.g. `x1 = 0.2`, was ignored and the
  full `x` range was integrated. No caller in panels set a single limit.

The numerical kernels of `plate_clpt_donnell` and `cylshell_clpt_donnell`
(`fkC_num`, `fkG_num`, `fkM_num`, `fkAx_num`, `fkAy_num` and `calc_fint`) now
always map the resolved limits to the natural coordinates, which for the full
domain gives exactly `xi = eta = -1, +1`. The full-domain matrices are
bit-identical to those of 0.6.9, through both the analytical and the numerical
integration, and so are the matrices of domains limited on both sides.
`Shell.is_partial_domain` compares the resolved limits with the edges, and
`Shell.calc_kA` uses it instead of repeating the old test.

### Breaking: tangent stiffness matrix of the non-linear models

The geometric stiffness matrix due to the membrane stress of the non-linear
strains, `KGNL`, moved from `fkG_num` to `fkC_num` in the models
`plate_clpt_donnell` and `cylshell_clpt_donnell`:

- `calc_kC(c, NLgeom=True)` returns `K0 + K0L + KL0 + KLL + KGNL`.
- `calc_kG(c, NLgeom=True)` returns `KG(N0 + N_L)`, which is homogeneous of
  degree one in `c`, as required by linear buckling analyses.
- The tangent stiffness matrix `calc_kT(c) = calc_kC(c, NLgeom=True) +
  calc_kG(c, NLgeom=True)` is unchanged, and it is the exact derivative of
  `calc_fint(c)`, giving quadratic convergence to the Newton-Raphson method.

Code that calls `calc_kG(c, NLgeom=True)` alone to obtain the geometric
stiffness matrix of a deformed state now obtains it without `KGNL`.

Following that move, the `NLgeom` argument of the kernels
`panels.models.plate_clpt_donnell_num.fkG_num` and
`panels.models.cylshell_clpt_donnell_num.fkG_num` was removed. It had become
inert, since `KGNL` is assembled by `fkC_num`, so the signature advertised an
effect it did not have. Code calling these kernels directly must drop the
argument, which was positional before `Nxx0`. `Shell.calc_kC` and
`Shell.calc_kG` keep their own `NLgeom` argument, where it still selects the
numerical integration.

### Breaking: renamed modules

The suffix `_bardell` was removed from the modules of the models:

- `panels.models.clpt_bardell_field` -> `panels.models.clpt_field`
- `panels.models.plate_clpt_donnell_bardell` and `_num` ->
  `panels.models.plate_clpt_donnell` and `_num`
- `panels.models.cylshell_clpt_donnell_bardell` and `_num` ->
  `panels.models.cylshell_clpt_donnell` and `_num`
- `panels.stiffener.models.bladestiff1d_clt_donnell_bardell` ->
  `panels.stiffener.models.bladestiff1d_clt_donnell`
- `panels.stiffener.models.bladestiff2d_clt_donnell_bardell` ->
  `panels.stiffener.models.bladestiff2d_clt_donnell`

The model names `Shell.model = 'plate_clpt_donnell'` and
`'cylshell_clpt_donnell'` are the only ones accepted. The legacy names
`'plate_clpt_donnell_bardell'` and `'cylshell_clpt_donnell_bardell'` were
removed from `panels.modelDB.db`, and `Shell._rebuild` now raises
`ValueError` listing the valid models when one of them is used.

### Bug fixes

- `Shell.calc_kC` forwarded an invalid Ritz-constant vector into the nogil
  kernels. The zero-filled default for `c` was installed only when both `c`
  and `ABDnxny` were absent, so `calc_kC(ABDnxny=...)` without `c` reached
  `np.ascontiguousarray(None)`, which returns the shape `(1,)` array `[nan]`.
  The kernels declare `double [::1] cs` and are compiled with
  `boundscheck=False`, so with `NLgeom=True` this was read far out of bounds
  instead of raising. The same gap on `c_cte` was worse: when a constant
  stress state was set through `Nxx_cte`, `Nyy_cte` or `Nxy_cte` but no
  `c_cte` was given, `None` reached `fkG_num`, and dereferencing a `None`
  memoryview segfaulted the interpreter. This hit `calc_kT` on any preloaded
  panel, since `NLgeom=True` forces the numerical integration.
- `Shell.calc_fint`, `Shell.uvw` and `Shell.strain` also forwarded `c` into a
  `double [::1]` argument without checking its size. They now validate it.
- An unset `Shell.r` silently became `r = 0`, which the cylindrical kernels
  divide by. The failure surfaced as a bare `AssertionError` from
  `structsolve.sparseutils.finalize_symmetric_matrix`, with nothing pointing
  at the radius. The flat-plate limit of a cylindrical shell is
  `r -> infinity`, not `r = 0`, so `r = 0` is never a valid radius. Models
  that need one are flagged with `requires_r` in `panels.modelDB`, and
  `Shell.calc_kC`, `calc_kG`, `calc_kM`, `calc_kA` and `calc_fint` now raise a
  `ValueError` naming the model and pointing at `'plate_clpt_donnell'` when
  `r` is `None` or non-positive. The `r = 0` fallback is kept only for the
  plate models, whose kernels never read it.
- `StiffPanelBay.add_panel` raised `ValueError: stack must be defined` and made
  `StiffPanelBay` unusable. It built its `Shell` without the laminate
  attributes and assigned them only afterwards, but `Shell.__init__` already
  calls `Shell._rebuild`, which needs them. The laminate, the model and the
  density are now passed to the constructor.
- The analytical closed-form matrices `fk0`, `fkG0` and `fkM` always integrate
  the full shell domain and silently ignored the `(x1, x2, y1, y2)` limits that
  only the numerically integrated matrices honour. Any assembly with more than
  one panel over the same approximation functions therefore counted the skin
  once per panel. `Shell.calc_kC`, `Shell.calc_kG` and `Shell.calc_kM` now
  select the numerical matrices whenever the new `Shell.is_partial_domain`
  returns `True`.
- `MultiDomain` connection `'SB'` overwrote the penalty given by
  `calc_kt_kr` with a hard-coded `kt = 2e5`. That is a N/mm**3 value, about
  6e-7 of the default penalty in SI units, so a stiffener base was practically
  disconnected from the skin. The reference values of
  `tests/multidomain/test_tstiff2d_assembly.py` had been regenerated to match
  it; restoring the penalty restores their original values.
- `MultiDomain` connections `'SB'` and `'SB_TSL'` took the panel coming first
  in the assembly as the top one, so the offset between the mid-surfaces had
  the wrong sign whenever the bottom panel came first, with errors of up to
  30 % in a buckling load. `p1` is now always the top panel and `p2` the
  bottom one, as documented, and panels of different dimensions raise
  `ValueError`.
- `MultiDomain.get_kC_conn` swapped `xcte1`/`xcte2` and `ycte1`/`ycte2` inside
  the user's connection dictionaries when `p1` came after `p2` in the
  assembly, so every second call, e.g. `calc_kC()` followed by `calc_kT()`,
  used them swapped back. The dictionaries are no longer modified. A
  connection of a panel to itself raises `ValueError` instead of
  `UnboundLocalError`.

### Enhancements

- `MultiDomain.calc_kC` and `MultiDomain.calc_kG` accept `NLgeom`, forwarded to
  each domain, such that the methods `calc_fext`, `calc_fint`, `calc_kC` and
  `calc_kG` of both `Shell` and `MultiDomain` can be passed directly to
  `structsolve.Analysis`.
- The Gauss-Legendre points and weights come from
  `scipy.special.roots_legendre`, instead of hardcoded tables, which matches
  the old tables to `1e-13` and has no upper limit on the number of
  integration points (one of the previous tables stopped at 30 points). The
  C++ sources of the tables (about 7.5 MB) were removed.
- `num_eigvalues` argument in `tstiff2d_1stiff_compression`,
  `tstiff2d_1stiff_freq` and `tstiff2d_1stiff_flutter`.
- The `MultiDomain` connection dictionaries accept the keys `'kt'` and `'kr'`
  to override the penalty constants of `calc_kt_kr`. The default rotation
  penalty does not grow as the domains become narrower, and for thin skins
  split in narrow strips it leaves a converged error of about 2 %.

### Documentation

- New Sphinx documentation, in the same style as `structsolve`, with usage
  examples extracted from the unit tests: linear static, linear buckling,
  frequency, flutter, Newton-Raphson, arc-length, multi-domain cylinders and
  panels with a T-stiffener.
- API reference with one page per module, and docstrings fixed such that the
  documentation builds without warnings.
- `Shell.calc_kC` documents that its return value is not purely constitutive:
  with `NLgeom=True` it contains `KGNL`, and with `c_cte` or a non-zero
  `Nxx_cte`, `Nyy_cte` or `Nxy_cte` it contains `KG(N_cte)`, the device used
  to superpose combined load cases. `Shell.calc_kG` documents that it stays
  homogeneous of degree one in `c`, and `Shell.calc_kT`, which had no
  docstring, documents the grouping of the tangent stiffness matrix.
- `Shell.calc_kC` claimed that passing `c` gives the large displacement
  matrix. It does not: `fkC_num` zeroes `w,x` and `w,y` unless `NLgeom == 1`,
  and `K0L`, `KL0`, `KLL` and `KGNL` are all built from them, so `c` alone
  still returns `K0`. What `c` selects is the numerical integration over the
  analytical closed form. The docstring now says so.
- `CHANGELOG.md` and `CITATION.cff`.
- `notebooks/stamatelos_labeas_2023.ipynb`, reproducing every result of
  Stamatelos and Labeas (Computation 2023, 11, 110) and documenting where the
  published data is incomplete or inconsistent.

### Tests

- Consistency of the tangent stiffness matrix
  (`tests/tests_shell/test_tangent_consistency.py`): directional Taylor test,
  finite-difference Jacobian, undeformed state, quadratic convergence of the
  Newton-Raphson method and homogeneity of `kG`.
- Geometrically non-linear analysis deep in the postbuckling regime with a full
  Newton-Raphson method, verifying the quadratic convergence
  (`tests/tests_shell/test_nonlinear.py`).
- Snap-through of a hinged cylindrical shell (Sabir and Lock) traced with the
  Riks method of `structsolve`
  (`tests/tests_shell/test_nonlinear_riks.py`).
- Partial-domain integration of `calc_kC`, `calc_kG` and `calc_kM`, checking
  that adjacent strips sum to the full-domain matrices
  (`tests/tests_shell/test_partial_domain_integration.py`).
- `StiffPanelBay` with blade stiffeners modelled by the 1D flange formulation
  (`tests/tests_stiffpanelbay/test_stiffpanelbay_lb.py`).
- Buckling loads of the laminated stiffened plates of Stamatelos and Labeas
  (Computation 2023, 11, 110), with the published reference values kept in a
  dictionary (`tests/tests_stiffpanelbay/test_stamatelos_labeas_2023.py`).
- Argument handling of the stiffness API
  (`tests/tests_shell/test_argument_validation.py`): the defaults for `c` and
  `c_cte`, the size checks of `calc_fint`, the radius validation, the
  convergence of the cylindrical kernels to the plate kernels as
  `r -> infinity`, and the removal of the `NLgeom` argument of `fkG_num` and
  of the legacy model names.
- `tests/multidomain/test_stamatelos_labeas_2023_multidomain.py`, the
  `MultiDomain` counterpart of the Stamatelos and Labeas (2023) test, with
  plate domains only and every connection strategy: `SSycte` and `SSxcte`
  (skin strips), `BFycte` (blades as plate domains, at a strip edge and along
  an interior line, Tables 4 and 5 and Figure 7), `BFxcte` (the same model
  rotated by 90 degrees, equal to machine precision) and `SB` and `SB_TSL`
  (the unsymmetric skin split in two bonded sub-laminates).
- `tests/multidomain/test_conn_kCBFxcte.py` compared the `BFycte` eigenvalue
  with itself; it now compares it with the `BFxcte` one.
- `tests/tests_shell/test_partial_domain_limits.py` and
  `test_stiffpanelbay_lb.py::test_panel_edge_at_one_meter` cover the
  integration limits.

## 0.5.4 (2026-04-09)

- Last version before this changelog, see the git history for details.
