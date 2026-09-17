# Changelog

## 0.6.0 (2026-09-17)

### Requirements

- `structsolve>=0.4.3`, with the new non-linear solvers (full Newton-Raphson,
  rewritten arc-length methods of Riks and Crisfield) and the verified sparse
  solver of `lb`, which could return wrong buckling loads without any warning.

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
`'cylshell_clpt_donnell'` are the new defaults. The legacy names
`'plate_clpt_donnell_bardell'` and `'cylshell_clpt_donnell_bardell'` are kept
in `panels.modelDB.db` for backward compatibility.

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

### Documentation

- New Sphinx documentation, in the same style as `structsolve`, with usage
  examples extracted from the unit tests: linear static, linear buckling,
  frequency, flutter, Newton-Raphson, arc-length, multi-domain cylinders and
  panels with a T-stiffener.
- API reference with one page per module, and docstrings fixed such that the
  documentation builds without warnings.
- `CHANGELOG.md` and `CITATION.cff`.

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

## 0.5.4 (2026-04-09)

- Last version before this changelog, see the git history for details.
