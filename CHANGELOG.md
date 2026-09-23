# Changelog

## 0.8.0 (2026-09-23)

### Breaking: default model of cylindrical shells

A `Shell` with a radius `r` and no `model` uses the Sanders-Koiter kinematics,
`'cylshell_clpt_sanders'`, instead of the Donnell kinematics. The Donnell
kinematics remain available with `model='cylshell_clpt_donnell'`. The
regression values of the multi-domain cylinders of
`tests/multidomain/test_cylinder.py` and
`tests/multidomain/test_cylinder_blade_stiffened.py` decreased by 0.2% to
0.6%, the Donnell values are kept in the comments.

### New models

- `'cylshell_clpt_sanders'`: cylindrical shells with the classical laminated
  plate theory (CLPT) and the Sanders-Koiter kinematics, whose strains vanish
  for any rigid-body motion. The rotation of the normal about the axis,
  `phiy = -w,y + v/r`, enters the non-linear strains, the geometric stiffness
  matrix, the mass matrix, the field outputs and the point constraints.
- `'plate_fsdt_donnell'`: plates with the first-order shear deformation
  theory (FSDT) and von Karman kinematics, with 5 DOFs `u, v, w, phix, phiy`
  per term. The shear correction is controlled by
  `Shell.fsdt_shear_correction`: a float `k` multiplies the uncorrected
  transverse shear stiffness, `5/6` by default, whereas `'rohwer'`,
  `'vlachoutsis'`, `'constant'` or `None` select the method of
  `composites.Laminate.calc_transverse_shear_stiffness`.
- `'plate_tsdt_donnell'`: plates with the third-order shear deformation
  theory of Reddy (1984) and von Karman kinematics, with 5 DOFs per term.
- `'cylshell_fsdt_donnell'`, `'cylshell_tsdt_donnell'`: cylindrical shells
  with the FSDT and the TSDT and the Donnell kinematics, 5 DOFs per term.
- `'cylshell_fsdt_sanders'`, `'cylshell_tsdt_sanders'`: cylindrical shells
  with the FSDT and the TSDT and the Sanders-Koiter kinematics, 5 DOFs per
  term, whose strains, including the transverse shear strains, vanish for
  any rigid-body motion. The rotation of the normal about the axis, `phiy +
  v/r`, enters the displacement field and the mass matrix, the changes of
  curvature include `v,y/r` and the rotation about the normal, and the
  non-linear terms use `w,y - v/r`, following Sanders (1959, 1963) and, for
  the TSDT, Reddy and Liu (1985), with no Sanders-Koiter terms in the
  third-order terms. The field output `phiy` and the approximation `fg` of
  `phiy` are the rotation of the normal `phiy + v/r`, which the connections
  `'SSxcte'`, `'SSycte'`, `'SB'` and `'BFycte'` also use.
- All the new models provide the analytical and the numerically integrated
  constitutive, geometric, mass and aerodynamic matrices, the internal force
  vector and the exact tangent stiffness matrix, such that linear static,
  linear buckling, frequency, flutter and non-linear static analyses are
  available. The integrands are generated from the kinematics in
  `theory/shells/cylshell_clpt_sanders/` and `theory/shells/fsdt_tsdt/`,
  which generates the six models based on the FSDT and TSDT, plates and
  cylinders, and replaces `theory/shells/plate_fsdt_tsdt_donnell/`.

### API

- The boundary condition flags of the rotations, `x1phix, x1phixr, ...,
  y2phiyr`, used by the FSDT and TSDT models. Their default is the hard simply
  supported condition, with the rotation tangential to each edge removed.
- `Shell.strain()` and `Shell.stress()` also return the transverse shear
  strains and forces, and the higher-order terms of the TSDT, for the models
  based on shear deformation theories.
- `MultiDomain` supports the FSDT and TSDT models with the connections
  `'SSxcte'`, `'SSycte'` and `'SB'`, implemented in
  `panels.multidomain.connections.kCsdt`, and `'BFycte'` and `'BFxcte'`,
  implemented in the functions `fkCBFycte*_sdt` and `fkCBFxcte*_sdt` of
  `kCBFycte.pyx` and `kCBFxcte.pyx`. The edge connections penalize
  `u, v, w` with `kt` and the rotations `phix, phiy` with `kr`, and for the
  TSDT also the normal derivative of `w`. The `'SB'` connection penalizes the
  displacements at the interface of the two laminates and, when `kr` is
  given, the difference of their rotations, with which two FSDT laminates
  behave like a single laminate with both stacking sequences. The
  base-flange connections penalize the rotation of the normals about the
  axis of the connection, `phiy` (`'BFycte'`) and `phix` (`'BFxcte'`), where
  the models based on the classical laminated plate theory use `-w,y` and
  `-w,x`; for the TSDT the derivative of `w` is not penalized. `'SB_TSL'`,
  and connections between models with a different number of DOFs, raise
  `NotImplementedError`.
- `MultiDomain.strain()` and `MultiDomain.stress()` also return the
  transverse shear strains and forces, and the higher-order terms of the
  TSDT, for the panels of the models based on shear deformation theories.
- `StiffPanelBay` raises `NotImplementedError` for models with 5 DOFs, since
  its stiffeners assume 3 DOFs.

### Tests

- Literature benchmarks for the new models: Leissa (1973), Loy et al. (1997)
  and Batdorf for Donnell against Sanders kinematics, Sanders' rigid-body
  criterion, Noor (1973, 1975), Whitney and Pagano (1970), Pagano (1970),
  Reddy (1984), Liew et al. (1993), Hashemi and Arsanjani (2005) and Leissa
  (1973) for the plates. The existing tests also run the new models.
- Multi-domain FSDT and TSDT plates, `tests/multidomain/test_multidomain_fsdt_tsdt.py`:
  plates divided in two and four domains reproduce the frequencies of Noor
  (1973), the deflection of Pagano (1970) at the interior corner of four
  domains and the single-domain results; two FSDT laminates connected with
  `'SB'` reproduce the single laminate with both stacking sequences, and
  without the rotations tied converge to it with `(h/a)^2`, as the TSDT
  laminates do.
- Cylindrical shells with the FSDT and TSDT,
  `tests/tests_shell/test_fsdt_tsdt_cylinders.py`: Leissa (1973), Table 2.8,
  where the Sanders-Koiter FSDT and TSDT are within 0.07 % of 3D elasticity
  for `R/h = 20`; Batdorf's axial buckling of a thin panel; Sanders'
  rigid-body criterion, including the transverse shear strains; an exact
  Navier solution of a thick cross-ply shell; the consistency of the
  analytical and numerical matrices and of the tangent stiffness matrix; a
  cylinder divided in two domains, and the base-flange connection of a
  Sanders-Koiter skin. The generic tests also run the new models.
- `tests/tests_shell/test_tangent_consistency.py`: the Newton-Raphson test
  measures the order of convergence with `log(e_k+1/e_k)/log(e_k/e_k-1)`,
  which is 2 for `e_k+1 = C e_k^2` whatever the constant `C`, instead of
  `log(e_k+1)/log(e_k)`, which falls below 2 for large `C`, and discards the
  manufactured equilibrium states whose tangent stiffness is nearly
  singular, where `C` grows with the norm of its inverse.

### Bug fixes

- Curvature term `gamma` of the piston theory, Krumhaar's correction for the
  external flow over a cylinder, made consistent in all kernels. With the
  aerodynamic load `q = beta*w,x + gamma*w` along `w`, positive outwards,
  `gamma` softens the shell independently of the flow direction, as obtained
  expanding the exact potential flow over a cylinder with a sinusoidal radial
  displacement (`tests/tests_shell/test_aero_curvature.py`). The numerical
  kernel `cylshell_clpt_donnell_num.fkAx_num`, also used by
  `'cylshell_clpt_sanders'`, had the opposite sign of the analytical
  `fkAx`, which changed the flutter of cylinders with partial integration
  domains. Flat plates have `gamma = 0`: the kernels of the plate models no
  longer include the term, `Shell.calc_kA()` ignores a `gamma` given for a
  plate with a warning, and `StiffPanelBay` computes `gamma` only for the
  cylindrical models. The load and the sign convention are documented in
  `Shell.calc_kA()`.
- Mass matrix of the 1D flange of `BladeStiff1D`, kernel
  `bladestiff1d_clt_donnell.fkMf`: the terms coupling the displacements
  `u, v` and the rotations `phix, phiy` were twice the value given by the
  integration of the kinetic energy through the height of the flange. They
  followed Eq. (26) of Castro et al. (2016),
  https://doi.org/10.1016/j.compstruct.2015.12.056, whose matrix `kmf` has
  `-2*df` where `-df` is correct. The correction is documented in the
  docstring of `fkMf` and checked against the kinetic energy integrated from
  the displacement field (`tests/tests_stiffpanelbay/test_bladestiff1d_mass.py`).
  The base of the stiffener is a `Shell` with offset and was not affected,
  although Eq. (27) of the same paper has the same factor.
- `BladeStiff1D.hb` stayed zero with a base, which removed the thickness of
  the base from the rotary inertia of the flange in `fkMf`.
- The rotation penalty of the connection `'BFycte'` used `w,y` for the skin
  also with the Sanders-Koiter kinematics, whose rotation about `x` is
  `w,y - v/r`. The kernels of `kCBFycte.pyx` take the optional arguments
  `rinv1` and `rinv2`, `1/r` of each panel with the Sanders-Koiter kinematics
  and zero otherwise, set by `MultiDomain`. A rigid-body rotation of a
  stiffened cylinder about its axis is now free of energy
  (`tests/multidomain/test_bf_sanders.py`). The matrices of the Donnell
  kinematics and of flat plates are unchanged, and `'BFxcte'` needs no
  change, since the rotation about `y` is `-w,x` for both kinematics. The
  buckling loads of the blade-stiffened cylinders of
  `tests/multidomain/test_cylinder_blade_stiffened.py` increased by 0.003%
  and 0.18%.
- `'BFycte'` and `'BFxcte'` took the first panel in the assembly as the
  base, whatever `p1` and `p2`: with the flange before the base the flange
  was rotated by 90 degrees in the opposite direction. `p1` is now always
  the base and `p2` the flange, as documented. The assemblies of the
  repository have the base first and are not affected
  (`tests/multidomain/test_bf_sdt.py`, both orders).
- `'SB_TSL'`: removed the fallback to the kernels of `kCSB_dmg.pyx` for
  panels of different dimensions, which could not be reached, since both
  assemblies require panels of the same dimensions and `MultiDomain` raises
  an error otherwise.

### Theory

- `theory/multidomain_penalization/connections.py` matches the current
  kernels: the curvature penalty removed from `kCSSxcte` is no longer
  derived, `kCBFxcte`, the rotation of the Sanders-Koiter kinematics of
  `kCBFycte` and the connections of the FSDT and TSDT models of `kCsdt.py`
  and of `kCBFycte.pyx` and `kCBFxcte.pyx` were added, and the script runs
  from any location.
- The `^` of `theory/shells/cylshell_clpt_donnell/cylshell_clpt_donnell.py`,
  which sympy took as a logical XOR in the mass matrix, is now a power.
- The Mathematica notebooks were replaced by SymPy scripts, verified against
  the notebooks and against the kernels:
  - `theory/func/bardell/bardell_integrals_C.py`, which writes
    `panels/core/src/bardell.cpp` and `panels/core/include/bardell.hpp`
    byte for byte, instead of `bardell_integrals*.nb`;
  - `theory/func/bardell/fuvw.py`, the displacement field of the CLPT with
    the Donnell and Sanders-Koiter kinematics and of the FSDT and TSDT,
    checked against `fg` of the field modules, instead of `fuvw.nb`;
  - `bladestiff1d_clt_donnell.py` and `bladestiff2d_clt_donnell.py` in
    `theory/multidomain_panels/`, and
    `theory/stiffener/mass_matrix_1D_stiffeners.py`, which also replaces
    `theory/multidomain_panels/mass_matrix.nb`;
  - `connections.py`, `plate_clpt_donnell.py` and `cylshell_clpt_donnell.py`,
    which already derived the matrices of their notebooks.

  The notebook of the cone is kept, and the T stiffener
  (`tstiff2d_clt_donnell`) is kept as it was. Elsewhere, the scripts that
  converted the Mathematica output (`print_expressions_python.py`,
  `print_bardell_integrals_*.py`) and all the old outputs of the theory
  folder were removed. The generator of the
  Sanders cylinder is now `theory/shells/cylshell_clpt_sanders/derive_expressions.py`.

### Documentation

- `doc/source/cohesive_zone.rst`, the theory of the cohesive zone of the
  `'SB_TSL'` connection, its deviations from the thesis of D'Souza (2024),
  its verification, the convergence study and the validation against the
  literature, replaces
  `theory/multidomain_penalization/cohesive_zone_deviations_from_thesis.tex`.
  What is specific to a function is now in its docstring:
  `panels.multidomain.connections.kCSB_dmg` and the methods of `MultiDomain`
  of the `'SB_TSL'` connection.
- `doc/source/bardell.rst` documents the SymPy scripts of the integrals and
  of the displacement field.
- The tables of the validation of `doc/source/cohesive_zone.rst` include the
  DCB benchmarks of Krueger (2008).

## 0.7.1 (2026-09-21)

### Requirements

- `composites>=0.9.0`, whose laminates cannot be pickled, see the fix of
  `StiffPanelBay.save()` below.

### Breaking: kernels of the `'SB_TSL'` connection

The Cython kernels of `panels.multidomain.connections.kCSB_dmg` take the
distance from the mid-surface of each panel to the interface, instead of a
single distance `dsb` between the two mid-surfaces:

- `fkCSB11_dmg(dt, p1, ...)`, `dsb` renamed to `dt`
- `fkCSB12_dmg(dt, db, p1, p2, ...)`, `dsb` replaced by `dt` and `db`
- `fkCSB22_dmg(db, p1, p2, ...)`, new first argument `db`

where `dt = sum(p_top.plyts)/2` and `db = sum(p_bot.plyts)/2`. Code calling
these kernels directly must be updated. The results of `'SB_TSL'`
connections change with the bug fixes below.

### Bug fixes

- The cohesive zone of the `MultiDomain` connection `'SB_TSL'`, after
  D'Souza (2024), could not capture the onset of failure of the double
  cantilever beam (DCB):
  - The tangential separation used the rotation of the top panel only, over
    the full distance between the mid-surfaces. It now uses the rotation of
    each panel over the distance from its mid-surface to the interface, and
    the 12 block uses the `eta` flags of the top panel.
  - `MultiDomain.force_out_plane_damage` integrated the tractions with the
    corrected separation, which drops the compressive tractions that balance
    the tensile ones. It now uses the uncorrected separation, the same that
    enters the internal force vector.
  - The correction of the separation, now in
    `MultiDomain.correct_separation`, zeroed each row of Gauss points up to
    its last non-positive point. It now follows Section 6.3.1 of the thesis:
    from the point of maximum separation towards `x = 0`, it zeroes the
    separation up to the first non-positive point only.
  - `MultiDomain.force_out_plane_damage` raised a `NameError` when no damage
    history (`dmg_index`) was set, e.g. before the first converged
    increment.
- `StiffPanelBay.save()` failed with `composites>=0.9.0`, because
  `_clear_matrices()` did not clear the laminates of the base and flange of
  the stiffeners, nor `BladeStiff1D.flam`. They are rebuilt before each
  calculation, so they are now cleared.

### Enhancements

- `MultiDomain.calc_kT_TSL(c)`, the damage-rate part of the tangent stiffness
  of the cohesive zone, non-symmetric, such that the secant stiffness plus
  this matrix is the consistent tangent of the `'SB_TSL'` connection.
- `MultiDomain.reaction_line_pd_xcte`, the reaction of a displacement
  prescribed along `x = cte`, which is the load measured by the load cell,
  exact for any state of damage.
- The secant stiffness of an `'SB_TSL'` connection between panels over the
  same domain is assembled by matrix products, with the pristine stiffness
  computed once and an update restricted to the damaged points. The Cython
  kernels are still used for panels over different domains, or with
  `'use_kernels': True` in the connection dictionary.
- The DCB driver of `tests/multidomain/test_dcb_damage.py` measures the load
  as the reaction of the prescribed displacement, evaluates the cohesive
  secant stiffness at every iteration as part of the internal force vector,
  uses a convergence criterion based on the physical forces, stiffer edge
  penalties, a predictor, a backtracking line search and the bisection of
  increments.

### Documentation

- `theory/multidomain_penalization/cohesive_zone_deviations_from_thesis.tex`,
  with the theory of the cohesive zone, the deviations from the thesis of
  D'Souza (2024) and their verification, a convergence study and the
  validation against DCB results in the literature.
- Validation notebooks of the `'SB_TSL'` cohesive zone in mode I DCB tests:
  `alfano2001_dcb.ipynb`, `camanho2003_dcb.ipynb`, `krueger2008_dcb.ipynb`, `turon2007_dcb.ipynb`,  `tijs2022_dcb.ipynb`, `lecinana2023_dcb.ipynb` and `tijs2023_phd_dcb.ipynb`,
  with the Krueger (2012) MMB benchmark data in
  `krueger2012_mmb_benchmark.ipynb` (not simulated). The shared model builder
  and solver are in `notebooks/dcb_utils.py`, and the non-linear results are
  cached in `notebooks/results`.
- `notebooks/validation_summary.ipynb` collects the peak loads of all
  validation notebooks, and `notebooks/convergence_study.py` reproduces the
  convergence study of the theory document.

### Tests

- `tests/multidomain/test_sb_tsl.py`: energy of the `kCSB_dmg` kernels with
  the bottom panel first in the assembly, equality of the matrix-product and
  kernel assemblies, and a finite-difference check of
  `MultiDomain.calc_kT_TSL`.

## 0.6.21 (2026-09-18)

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
- `BladeStiff2D` lost the ply densities when rebuilding the laminates of the
  base and of the flange. The base, integrated numerically over its partial
  domain, had no mass at all, so natural frequencies and flutter results of
  bays with a `BladeStiff2D` base were wrong with no warning. The base now
  adds exactly the mass of a skin strip of the same laminate and width.
- `MultiDomain.force()` integrated the stress resultants over the natural
  coordinate without the Jacobian, returning `2*mean(N)` instead of the
  section force: along `x = cte` the result is now multiplied by `b/2`, and
  along `y = cte` by `a/2`. `MultiDomain.calc_results()` with
  `vec='Fxx'`, `'Fyy'` or `'Fxy'` is affected the same way.
- `StiffPanelBay.calc_kA()` used only the first panel, integrated over its
  own partial domain, instead of the whole bay. It now sums the contribution
  of every panel, as `calc_kC()` does, and uses the piston theory parameters
  of the bay.
- Methods that always raised an exception:
  - `Shell.calc_cA()` looked `fcA` up in the numerical module instead of the
    analytical one. It now also returns the matrix.
  - `Shell.strain()` and `Shell.stress()`, and therefore `plot_shell()` with
    a strain or stress field, failed for a plate with an unset radius
    (`r=None`).
  - `StiffPanelBay.calc_kA()` and `StiffPanelBay.calc_cA()`.
  - `StiffPanelBay.calc_fext()` failed with any skin load, and for any bay
    with a `BladeStiff2D`. Flange loads are read from
    `BladeStiff2D.forces_flange`.
  - `StiffPanelBay.save()`.
  - `StiffPanelBay.get_size()` for a `BladeStiff2D` without flange.
  - `StiffPanelBay.uvw_stiffener()` for any stiffener after the first one.
- Removed `StiffPanelBay.tstiff2ds` and the code handling it, left over from
  the removal of `TStiff2D`: nothing could fill the list, and the external
  force vector of that code wrote past the end of its buffer.

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
- New tests for `Shell` (`tests/tests_shell/test_shell_api.py`),
  `plot_shell` (`tests/tests_shell/test_plot_shell.py`), `StiffPanelBay` with
  `BladeStiff1D` and `BladeStiff2D`
  (`tests/tests_stiffpanelbay/test_stiffpanelbay_api.py`) and the
  post-processing of `MultiDomain`
  (`tests/multidomain/test_multidomain_postprocessing.py`), raising the line
  coverage of the Python modules from 67 % to 92 %.

## 0.5.4 (2026-04-09)

- Last version before this changelog, see the git history for details.
