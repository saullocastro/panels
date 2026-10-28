# Changelog

## 0.14.1 (2026-10-28)

### Python versions

- Python 3.9 is no longer supported, it reached its end of life in October 2025: Python 3.10 or newer is required, and the tests and wheels cover Python 3.10 to 3.14.

### Packaging

- Packaging: the metadata, with the version, in `pyproject.toml` (PEP 621), built with `setuptools>=77` and Cython; `setup.py` keeps only the Cython extensions with the compiler flags, including those of the WebAssembly build, and writes `panels/version.py` from `pyproject.toml`, with the same fields as before (`MAJOR`, `MINOR`, `MICRO` and `ISRELEASED` are gone from `setup.py`; `isreleased` is true for a version without suffix). The documentation reads the version from `pyproject.toml`. The license is given as the SPDX expression `BSD-3-Clause`, wheel is no longer a build requirement, README.md and LICENSE are no longer installed in the root of the Python environment (`data_files`), and two keywords merged by a missing comma are separated.

### Bug fixes: mass matrices

- The closed-form mass matrices `fkM` of the nine models read `shell.rho`,
  whereas `Shell.calc_kM` and the numerical `fkM_num` use the density of the
  laminate, `lam.intrho/lam.h`: with per-ply densities `rhos` the two
  differed, and a `Shell` given only `rhos` had a zero mass matrix.
  `Shell._rebuild` keeps the lists `plyts`, `laminaprops` and `rhos` given
  by the user, and `json_io` saves which lists were derived (key
  `'derived_lists'`; the lists of older files are kept as given).
- The mass matrices of all the models, closed-form and numerical, integrated
  a uniform density over `-h/2 - d <= z <= h/2 - d`, whereas the stiffness
  matrices and composites place the laminate at `-h/2 + d <= z <= h/2 + d`:
  with an offset the coupling of `u` and `phi` had the wrong sign. The mass
  matrices now use the moments of the density `I_0` to `I_6` of
  `composites.Laminate`, exact for per-ply densities. `fkM(shell, size,
  row0, col0)` reads them from `shell.lam` (the argument `d` is dropped),
  `fkM_num(shell, I_input, size, row0, col0, nx, ny)` receives them, and
  `Shell.calc_kM` gains `I_nxny`. With offset 0 and a uniform density the
  matrices are unchanged to `2e-16`.

### Bug fixes: third-order shear deformation theory with an offset

- With an offset `d`, the cubic term `-c1 z^3 (phi + w,x)` did not make the
  transverse shear vanish at the faces `z = d -+ h/2`. The field is now
  `u(z) = u + z phi - c1 [(z - d)^3 + d^3] (phi + w,x)`, Reddy's field for
  `d = 0`, with `phi` the rotation at the mid-surface of the laminate; the
  strain energy equals a ply-by-ply integration to `2e-15` (14% off before
  for `d = 1.2` mm).
- The connections of the TSDT models tie `u, v, w` and the rotations at the
  common reference surface `z = 0`, `theta = u,z(0) = phi - (4 d^2/h^2)(phi
  + w,x)`, instead of the mid-surface rotations `phi`, which differ between
  panels with different offsets. The skin-base connection evaluates each
  laminate at its own face, `z = d -+ h/2`. Nothing changes for `d = 0`, for
  the FSDT and for the CLPT.
- The rotations `phix` and `phiy` prescribed with `Shell.add_point_pd`,
  `Shell.add_distr_pd_fixed_x` and `Shell.add_distr_pd_fixed_y`
  (`pd_method='null-space'`) are, for the TSDT, the rotations at the
  reference surface `z = 0`, `theta = phi - (4 d^2/h^2)(phi + w,x)`, those
  tied by the connections, instead of the mid-surface rotations `phi`: with
  an offset the rotation at `z = 0` missed the prescribed value by the term
  of the transverse shear, e.g. `2.15e-3` instead of the prescribed `2e-3`
  at a point of a plate with `d = 0.3 h`. Nothing changes for `d = 0`, for
  the FSDT and for the CLPT.
- New page `tsdt_offset.rst`.

### Clamped edges of the third-order shear deformation theory

- A clamped TSDT edge keeps the cross-section planar and unrotated, `u(z) =
  v(z) = 0` for every `z`, which requires `u = phi = w,n = 0`: the slope
  flag, e.g. `x1wr = 0` on `x = 0`, is removed together with the rotation
  flags. With the slope free the warping term `p(z) (phi + w,x)` remained
  at the edge: for a square isotropic plate with `h/a = 0.1` clamped on
  four edges the first frequency `omega a^2 sqrt(rho h/D)` is 32.64 instead
  of 30.18. The FSDT clamp is unchanged, `w,n` free. The function
  `set_edge` of `tests/tests_shell/test_fsdt_tsdt_plates.py`, the pages
  `boundary_conditions.rst` and `tsdt_offset.rst` and the docstring of
  `Shell` are updated; the thin plates of Leissa (1973) move by less than
  `1.2e-5`, e.g. `'SCSC'` 28.95027 to 28.95059 (reference 28.951).
- New keyword `z` of `Shell.add_point_pd`, `Shell.add_distr_pd_fixed_x`
  and `Shell.add_distr_pd_fixed_y`, and of the penalty kernels `fkCpd`,
  `fkCld_xcte` and `fkCld_ycte`: the in-plane displacements `u` and `v` are
  prescribed at the thickness coordinate `z`, measured from the reference
  surface, `u(z) = u + (z - c1 g) phi_x - c1 g w,x` with `g = (z - d)^3 +
  d^3` for the TSDT (`c1 = 0` for the FSDT, `u(z) = u - z w,x` for the
  CLPT), with both `pd_method='penalty'` and `pd_method='null-space'`.
  Default `z = 0`, unchanged. The stored lists `point_pds`, `distr_pds`
  and `*_values` end with `z`; `json_io` completes the lists of older files
  with `z = 0`. New function `nullspace.pd_rows`.
- Face clamp, `u = v = 0` at the faces `z = d -+ h/2` and `w = 0`: the
  clamp of the flags for the CLPT and the FSDT, and for the TSDT the two
  conditions `phi = w,x/2` and `u = -(d/2)(1 - 4 d^2/h^2) w,x`, which leave
  the section free to warp and which the flags cannot express. New test
  file `tests/multidomain/test_face_clamp.py` and example page
  `ex_face_clamp.rst`.

### Bug fixes: stiffeners, connections and post-processing

- `BladeStiff1D`: the torsion of the flange used `bf*Jxx*w,xy^2`, without a
  shear modulus; it now uses the torsional stiffness `GJ = 4 bf D66` of a
  thin laminated strip, new attribute `BladeStiff1D.GJ`, also saved by
  `json_io`. The results of Tables 4 and 5 of Stamatelos and Labeas (2023)
  rise by 1.9% to 3.3%, e.g. from 115.22 to 119.01 N/mm, those of Figure 7
  by up to 0.04%; `notebooks/stamatelos_labeas_2023.ipynb`,
  `stamatelos_labeas_2023_multidomain.ipynb` and the Stamatelos section of
  `conn_method_comparison.ipynb` are updated.
- `BladeStiff1D`: the coupling `S1` uses the condensed stiffness `Q16 - Q12
  Q26/Q22` instead of `Q16 - Q12 Q16/Q22`, as in Eq. (16) of Castro et al.,
  Compos. Struct. 140 (2016), with `y_k` measured from the mid-plane of the
  flange instead of its first face: `S1` is zero for a symmetric flange.
  Balanced cross-ply flanges are unaffected.
- The skin-base connections `'SB'` and `'SB_TSL'` connect the lower face of
  `p1`, `z = d1 - h1/2`, to the upper face of `p2`, `z = d2 + h2/2`, with the
  offset `d` of each laminate (`Shell.offset`), for all the models; the CLPT
  and FSDT kernels placed the faces at `z = -+h/2`, ignoring the offsets. Two
  laminates bonded with `'SB'` with offsets had their first frequency 25%
  (CLPT) and 41% (FSDT) above that of the single laminate, which they now
  reproduce to `1e-12` for any offsets. New `kCsdt._sb_distances`, used by
  the penalty and null-space `'SB'`, the kernels of `'SB_TSL'` and
  `MultiDomain._tsl_operators`. Without offsets nothing changes.
- The penalty kernels of the prescribed displacements `fkCpd`, `fkCld_xcte`
  and `fkCld_ycte` (`panels/multidomain/connections/kCpd.pyx`) used a stride
  of 3 DOFs per term, `row0 + 3*(j*m + i)`, for all the models: with the
  FSDT and TSDT models, 5 DOFs per term, the penalty stiffnesses of `u, v, w`
  landed silently on other DOFs, inconsistent with the forces `k u_p` of
  `shell_fext`, e.g. `w = 1.1e5` instead of the prescribed `1e-3` at a point
  of a simply supported `plate_fsdt_donnell` with `k = 1e4 E h`. The kernels
  read the number of DOFs per term from `modelDB` (signatures unchanged, a
  `ValueError` for an unknown `p.model`); for the FSDT and TSDT the penalty
  method now converges to the null-space method with `1/k`, as for the
  CLPT, whose results are unchanged.
- `BladeStiff1D` without a flange accepts `bf=None`.
- `StiffPanelBay.calc_kA` sums the aerodynamic matrices of the panels without
  skew-symmetrizing them.
- The cylinder assembly functions of `panels.multidomain` give each panel its
  own load; the lambdas bound late and every panel got the load of the last
  one.
- `MultiDomain.force_out_plane` integrates `Qx` along `x = x_cte_force` (or
  `Qy` along `y = y_cte_force`), instead of `Qx + Qy` along a hard-coded
  line, and checks its arguments.
- `MultiDomain.plot` computes the field of `eval_panel` and uses `xs` and
  `ys`.
- `cylinder_blade_stiffened_compression_lb_pd_from_static` imposes the
  prescribed end-shortening exactly, with `pd_method='null-space'` and
  `MultiDomain.linear_buckling`: the forces `ku*pd` of the penalty method
  entered `calc_fext` without the penalty stiffness in the stiffness matrix,
  so the analysis was load controlled, `u` at `x = height` reaching 1.2% of
  the prescribed value. The returned assembly stores the connections and has
  `pd_method='null-space'`, the prescriptions are added with `cte=False`, and
  `eigvals` multiply the prescribed displacements: 0.4785 for the
  five-panel cylinder of the tests (`m = 8`, `n = 12`), instead of 39.99, the
  multiplier of the forces `ku*pd` with `ku = 1e6`. The signature is
  unchanged; `ku` only flags the prescribed component.
- `MultiDomain.linear_buckling`: the constant stress resultants `Nxx`,
  `Nyy` and `Nxy` of the panels are a pre-stress, in the geometric stiffness
  of the constant state `c0` only. With constant prescribed displacements
  they entered both `K_G(c0)` and `K_G(c1)`, i.e. they were applied `1 +
  lambda` times: a constant prescribed rigid translation, which produces no
  stress, lowered the critical load factor by exactly one (161.93 to 160.93
  for a plate with `Nxx = -1`). Now `K_G(c1)` is computed with them set to
  zero, also without prescribed displacements and for a given `c`; a
  `ValueError` is raised when the incremented state has no geometric
  stiffness, e.g. when `Nxx` is the only load, which `linear_buckling` used
  to multiply by the load factor: apply such a load with
  `Shell.add_distr_load_fixed_x` instead.
- Skin-base connections of curved panels with the Sanders-Koiter kinematics:
  the tangential displacement at the interface, `v(z) = v + z (v/r - w,y)`,
  now has the term `v/r` in the CLPT connection `'SB'`, penalty and
  null-space, see the unified `'SB'` below, and in the kernels
  `fkCSB11_dmg`, `fkCSB12_dmg`,
  `fkCSB22_dmg` of `'SB_TSL'` with `use_kernels=True` (new optional
  arguments `rinv1`, `rinv2`), which differed from the default matrix
  products by 1.2e-3 relative and now agree to 1.5e-15. A rigid rotation
  about the axis of two bonded Sanders panels, the radius of each that of
  its reference surface, had a connection energy of `(dsb/r)^2` = 2e-6 times
  that of a separation `r Omega`, now zero to round-off. The FSDT and TSDT
  `'SB'` already had the term. Nothing changes for the Donnell kinematics
  and for plates.
- One skin-base connection `'SB'` for every model: `u1(z1) = u2(z2)`,
  `v1(z1) = v2(z2)` and `w1 = w2` at the faces in contact, `z1 = d1 - h1/2`
  of `p1` and `z2 = d2 + h2/2` of `p2`, each panel with its own kinematics,
  `u(z) = u - z w,x` and `v(z) = (1 + z/r) v - z w,y` for the CLPT (the
  term `z/r` only with the Sanders-Koiter kinematics), with the operator
  `kCsdt._surface_terms` of the FSDT and TSDT `'SB'` and of the prescribed
  displacements at a thickness coordinate. The null-space constraints and
  the penalty matrix, `kt` times the integral of the products of the same
  quantities, come from `kCsdt._sb_components`. The CLPT used the form of
  the thesis of D'Souza (2024), `u1 + dsb w1,x = u2` and `(1 - dt/r1) v1 +
  dsb w1,y = (1 + db/r2) v2`, with the slope of `p1` only, from the
  hand-written kernels of `kCSB.pyx`. The null-space results do not change:
  `w1 = w2` at the Gauss-Legendre grid ties the slopes over the whole area,
  and the two sets of constraints have the same null space to `2.5e-15`.
  The penalty results change at finite `kt`, all within the tolerances of
  the tests: for the laminates `[0/90]` over `[0/90]` with `1e5` times the
  default `kt`, the largest error of the first 8 frequencies with respect to
  the single laminate, between `6.94e-6` and `8.31e-6` depending on the
  offsets, is `6.94e-6` for all of them; the
  T-stiffened panel of `tstiff2d_1stiff_freq` 48.27331 to 48.27327,
  `tstiff2d_1stiff_compression` 142.6822 to 142.6811 (114.8576 to 114.8542
  with the static pre-stress), `tstiff2d_1stiff_flutter` 446.1180 to
  446.1153; the bonded sub-laminates of Stamatelos and Labeas (2023) by
  `-3.0e-6` at most, and now agree to `5e-8` with the pristine `'SB_TSL'`,
  whose
  separations are the same quantities: with `k_o` at every point its matrix
  is that of `'SB'` with `kt = k_o`. The penalty method converges to the
  null-space method with `1/kt`, e.g. from `2.7e-2` with the default `kt`
  to `2.9e-5` with `1000 kt` for two bonded laminates with offsets. The key
  `'kr'` of `'SB'` is still used only by the FSDT and TSDT, whose rotations
  are independent: the rotations of the CLPT follow from `w`, and tying them
  too would be redundant or, with the Sanders-Koiter kinematics and distinct
  reference surfaces, would impose `(z1 - z2) w,y = 0`. A `'kr'` given to
  `'SB'` of the CLPT is not used and now issues a `UserWarning`. New test
  file `tests/multidomain/test_sb_unified.py`.
- Default numbers of integration points: `Shell.nx` and `Shell.ny` follow
  `m` and `n` at the time of the calculation, `max(2*m - 1, 7)` and
  `max(2*n - 1, 7)`, new function `panels.shell.gauss_points`, unless a
  value is assigned (kept, the rebuild only raising it to `m`, `n`;
  `None` restores the default). Before, `nx = 2*m` was set only by the
  constructor, so that `Shell()` followed by `s.m = 20` integrated with 22
  points, and for `m <= 3` the 6 points of `2*m` did not integrate exactly
  the internal force vector and the tangent stiffness matrix, products of
  four functions of degree up to `4 max(m - 1, 3) = 12`. `2*max(m - 1, 3) + 1`
  points are the fewest that integrate them exactly; for `m >= 4` the new
  default has one point fewer than `2*m` and the same results to round-off
  for a constant constitutive matrix and polynomial pressures. `json_io`
  saves `nx`, `ny` with their value and lists them in `'derived_lists'`
  when they follow `m`, `n`; the values of older files are kept as
  assigned.

### Breaking: the null-space basis is `Z`

- The basis of the null space of the connections and prescribed
  displacements, `c = Z c_r + c_p0 + lbd*c_p1`, is `[Z]` (was `[T]`), which
  no longer clashes with the transformation matrices `T`, e.g. that of the
  strains of the TSDT with an offset. Renamed, without aliases (all public
  in 0.13.11): `MultiDomain.get_T()` -> `MultiDomain.get_Z()`, the
  attribute `MultiDomain.T` -> `MultiDomain.Z`, the attribute
  `nullspace.Elimination.T` -> `nullspace.Elimination.Z`, and the keys
  `'Te'` and `'Bte'` of the dictionaries of `Elimination.steps` -> `'Ze'`
  and `'Bze'`. The JSON files store the tolerance of the basis as
  `'Z_tol'` (was `'T_tol'`); the files written by 0.13.11 are still loaded.

### API

- `MultiDomain.get_kC_conn` loses the argument `kw_tsl`, which was not used:
  the stiffness of each `'SB_TSL'` connection is computed with
  `MultiDomain.calc_k_dmg` at `c`. Calls passing `kw_tsl` must drop it.
- New module `panels.multidomain.connections.skin_base` with
  `fkCSB(kt, p_top, p_bot, size, kr=0.)`, the upper triangle of the matrix
  of `'SB'` for every model; `kCsdt.fkCSB_sdt` gives the same matrix for
  the FSDT and TSDT, signature unchanged.
- The Cython module `panels.multidomain.connections.kCSB` is removed.
  `fkCSB11`, `fkCSB12` and `fkCSB22`, still importable from
  `panels.multidomain.connections`, are deprecated and issue a
  `DeprecationWarning`: they return the blocks of `fkCSB`, with the faces
  and the kinematics of the panels, and ignore the arguments `dsb`, `dt`,
  `db`, `rinv1` and `rinv2`. `nullspace._clpt_components` loses the argument
  `dsb`.

### Tests

- The Riks test of the hinged cylindrical shell no longer depends on how the
  arc-length adaptation samples the path, and passes with both convergence
  criteria of structsolve.
- `tests/tests_shell/test_kernel_python_calls.py` checks that the kernels
  are finite and agree with the numerically integrated ones. The
  constitutive matrix read by the kernels is assembled without
  matrix-matrix BLAS products, because the `cblas_dgemm` of Intel oneMKL
  2024.2.0 to 2025.0.0 on Windows x64 changes the registers `xmm8`-`xmm15`
  that the caller expects to be preserved; oneMKL 2025.0.1 and later, or
  `MKL_CBWR=COMPATIBLE`, preserve them.
- New `tests/multidomain/test_dcb_linear_measures.py`: linear solutions of
  the DCB model of the thesis of D'Souza (2024), asserting the measures of
  the load and the effect of the penalties of the edge connections given in
  `doc/source/cohesive_zone.rst`, to `1e-4` relative for the integrals of
  the traction and the largest separation, to `5e-4` for the reaction and
  the integral of `Q_x`, which carry a round-off error of up to `3e-4` from
  the 0.015 mm connecting domain (it changes with the number of integration
  points, all exact), and to `1e-5` for the reaction with a 5 mm connecting
  domain.
- The tests that recompute the residual of the solvers of structsolve use
  `R = F_int - lambda F_ext`, the convention of structsolve from 0.9.3 and
  of the drivers of panels (were `lambda F_ext - F_int`): the `Recorder` of
  `tests/tests_shell/test_follower_pressure.py` and its docstring, the
  equilibrium checks of `test_follower_pressure.py` and
  `tests/tests_shell/test_nonlinear_riks.py`, and the finite-difference
  check of `tests/multidomain/test_null_space_pd.py`, which now asserts
  `dR/dlambda = -calc_fext(c, lbd)` as its docstring states; also
  `theory/shells/follower_pressure/follower_pressure.md`. The tests do not
  depend on the convention of the installed structsolve, and the solutions
  of `Analysis.static` over the suite are bitwise identical with both
  conventions of structsolve.

### Documentation

- Functional docstrings of the public API, and Notes and References with
  the theory of the models, connections, stiffeners, cohesive zone and
  multidomain assembly.
- The sign convention of the piston-theory matrices, with the sign of the
  term of Krumhaar's correction in Eqs. (18), (21) and (29) of Castro et
  al., Compos. Struct. 140 (2016), stated as a mistake of the paper; the
  code is unchanged.
- `doc/source/cohesive_zone.rst`: the tables of the out-of-plane force of
  the DCB for a tip displacement of 1 mm and of its elastic stiffness against
  the factor on the default edge penalties, and the stiffness with a
  connecting domain of 5 mm, recomputed with the tangential separation of
  `'SB_TSL'`, with the slope of each panel; they had been computed with that
  of Eq. 4.62 of the thesis, with the slope of the top panel only. The
  reactions are 1.0-1.2% lower and the largest separation per unit of
  imposed displacement 20% higher; with the default penalties the separation
  lags the imposed displacement by 21%, not 20%. The linear solutions are
  computed at 0.5 mm, below the damage onset, and scaled to 1 mm.
- `doc/source/cohesive_zone.rst`: the rotational penalty of an `'SSxcte'`
  connection is a spring of stiffness `k_r` per unit width, with the energy
  `k_r (w^i_x - w^j_x)^2/2` of the kernels, which adds `3h/a`, about 13%, to
  the bending compliance of the DCB arm, not `2 k_r` and 7%.
- `doc/source/tangent_product.rst`: the single-shell timings and the rows of
  0.14.0 of the displacement-controlled post-buckling re-measured with the
  default integration points of `gauss_points`; the times of
  `doc/source/tabulation.rst` and of the older rows, measured with
  `nx = 2m`, `ny = 2n`, are labelled with these points.
- Statements checked against the sources: the non-linear terms of the
  Sanders-Koiter models, without the rotation about the normal, are a
  simplification of Sanders' theory for small strains and moderately small
  rotations, whose mid-surface strains also contain `omega3^2/2` (Amabili
  2008, Sec. 1.3.8, Eqs. (1.133)-(1.134)), see `panels.models`; the
  symmetry conditions of the follower load stiffness with their locations in
  Hibbitt (1979), pp. 938-939, and Schweizerhof and Ramm (1984), Table 2 and
  Eqs. (32)-(35), whose Table 3 lists the ring values 3, 4 and 4.5 `EI/R^3`;
  the TSDT of the Sanders cylinders as in Reddy and Liu (1985), Eqs. (5),
  (7) and (11).
- Notation of `structsolve` and `pyfe3d`: the linear constitutive stiffness
  matrix `K_C0` (was `K_0`), the non-linear one `K_CNL = K_0L + K_L0 +
  K_LL`, with the terms of `B_NL` only, and the diagonal scaling vector `d`
  of Crisfield's criterion (was `D`).
- One notation in the docstrings, `doc/source`, the markdown of the
  notebooks, the theory of `theory/` and the comments of the kernels
  written by its generators. Vectors are bold lower case, without braces or
  overbar, e.g. the Ritz constants `\boldsymbol{c}` (were `{c}`) and the
  increment `\delta \boldsymbol{c}`, except the force vectors and the
  residual, upper case, `\boldsymbol{F}_{ext}`, `\boldsymbol{F}_{int}` and
  `\boldsymbol{R}` (also for the lower case `{f}`, `{f_int}`, `{f_ext}` of
  the null-space method and of the cohesive zone); matrices are bold upper
  case, without brackets, e.g. `\boldsymbol{K}` (was `[K]`); the overbar is
  kept for prescribed values, the membrane stress resultants `Nxx`, `Nyy`,
  `Nxy` being `\bar{\boldsymbol{N}}` (were `N_0`, `N^0`; `KG(Nbar + N_L)` in
  the comments of the kernels, were `KG(N0 + N_L)`), and the prescribed
  values of the null-space method `\bar{\boldsymbol{d}}_0`,
  `\bar{\boldsymbol{d}}_1`, the right-hand side of a block of the
  elimination being `\boldsymbol{b}`. The residual is `R = F_int - F_ext`
  in all the pages, also in `follower_pressure.rst` and in the null-space
  method (were `F_ext - F_int`), with `K_T = dR/dc` and the load vector
  `-dR/dlambda` of the arc-length methods. `K_C0` also in
  `tangent_product.rst` (was `K_0`); `K_CNL` as the sum of its three
  integrals of `B_L` and `B_NL` in `Shell.calc_kC`, and `K_CNL + K_GNL`
  written out where the matrix of `fkC_num` is meant, in
  `tangent_product.rst` (`K_CNL` included `K_GNL` there) and in the
  comments of the kernels, `KC0 + KCNL + KGNL` (were `K0 + K0L + KL0 + KLL
  + KGNL`); `K_GNL` (was `K_{G_{NL}}`); the load stiffness matrix `K_F =
  -dF_p/dc` of `Shell.calc_kCfollower` (was `K_{C_{follower}}`, and `K_p =
  -K_F`); the rows `S^u`, `S^v`, `S^w` of the approximation functions (were
  `{N_u}`, `{N_v}`, `{N_w}`); the tangent vectors `X_{,x}`, `X_{,y}` of the
  follower pressure (were `A`, `B`, the symbols of the laminate); the
  constitutive matrix `C` of the laminate (was `[F]`); the factor `U` of
  the QR decomposition `Q U` of the null-space method (was `[R]`, now
  free for the residual); the strain operator `B_L + B_NL`
  (was `[B(beta)]`); the pre-buckling state `c-hat`; the aerodynamic
  damping matrix `C_A` (was `[c_A]`); Bardell's functions `P_i` (were `f_i`
  and `g_j`); in `cohesive_zone.rst` the residual `R = F_int - F_ext` with
  `F_int` including the cohesive and the penalty forces, and the first term
  of the denominator of the convergence criterion `F_int - K_PD c`, the
  internal force vector of the drivers, the diagonal scaling vector `d`
  (was `[D]`) and the tangent stiffness matrix `K_T` (was the Jacobian
  `[J]`); the direction `v` of the Taylor test of the tangent (was `d`); `\boldsymbol` (was `\mathbf`,
  `\vec`) and the transpose `^\top` (was `^T`). The API names, e.g.
  `calc_kCfollower`, the key `'kCfollower'` and the argument `c`, are
  unchanged.
- New `notebooks/fk1952_shallow_arches.ipynb`: snap-through of a hinged
  shallow circular arch, `lambda_1 = 2.2`, under a sinusoidal dead load,
  a strip of `cylshell_clpt_sanders` with `nu = 0`, traced through both
  limit points with the Riks and the Crisfield methods of `structsolve`,
  each with the convergence criteria `'crisfield'` and `'relative'` at the
  default tolerance, against the symmetric path of Eqs. (15) and (45) of
  Fung and Kaplan, NACA TN 2840 (1952), solved in the notebook. The four
  paths coincide; the maximum load is 4.9858 against 4.9822 (+0.074%) and
  the minimum -0.5883 against -0.5822, differences of the order of
  `theta^2 = 2.4e-3`, `theta` being the half-angle of the arc, the order of
  the terms in which the model differs from the shallow-arch theory; results
  stored in `notebooks/results/fk1952_shallow_arches.json`.
- The rotations of the von Karman terms of the FSDT and TSDT models,
  `beta_x = -w,x` and `beta_y = -w,y + v/r` (Sanders), and of the tangent
  product of the CLPT models with Donnell kinematics, `beta_x = -w,x`,
  `beta_y = -w,y`, take the sign of the rotations of the normal, as
  `phi_x`, `phi_y` of `cylshell_clpt_sanders` (were `w,x` and `w,y -
  v/r`), in the generators in `theory/shells/fsdt_tsdt`, `tangent_product`
  and `follower_pressure`, in the kernels `<model>_num.pyx` written by them,
  in `fsdt_tsdt_field.pyx`, and in the docstrings of `panels.models`,
  `Shell.calc_kC`, `Shell.calc_kG`, `doc/source/tangent_product.rst` and
  `follower_pressure.rst`. They enter only squared or multiplied by the
  operator `G` of the same sign, and the matrices, internal forces and
  strains are unchanged, bit for bit.
- `doc/source/follower_pressure.rst`, ring under uniform pressure: the
  ring values with the reference of Pflueger (1975), p. 403, and Eq. (97b),
  p. 150, for the long cylinder; why the Donnell row differs from the
  classical `3D/r^3`, which Kardomateas (1993), Eq. (29a), p. 199, obtains
  from Donnell's equations with the rotation `-w,y + v/r` of Sanders; the
  centrally directed load checked only with `central_load_stiffness` of
  the tests; DOI of Kardomateas (1993) in `ref.rst`.
- `theory/shells/fsdt_tsdt/fsdt_tsdt.py`: the TSDT takes the third-order
  displacement field of Reddy and Liu (1985), Eqs. (5)-(9), with Sanders'
  corrections of the bending strains added, since Eq. (11) of Reddy and
  Liu has no Sanders-Koiter terms.

### Clean-up

- The internal names of partial derivatives follow one convention, an
  underscore introducing the derivative: `fAu_xi`, `gAw_etaeta`, `f1Aw_xi`,
  `sAw_eta` (were `fAuxi`, `gAwetaeta`, `f1Awxi`, `sAweta`), `w_xi` and
  `w_x` (were `wxi`, `wx`) in the kernels of the models, connections and
  stiffeners and in their generators in `theory/`; `fu_xi`, `gw_eta`, `w_x`
  and `cfw_x` in `panels/models/clpt_field.pyx`; `f_xi`, `g_eta`, `val_x`
  and `val_xx` (were `fxi`, `geta`, `dx`, `dxx`) in
  `panels/models/fsdt_tsdt_field.pyx`; `u_x` to `w_y` in the
  follower-pressure kernels; `w_x`, `w_xx`, `Mxx_x` and `Qx_x` (were
  `dw_dx`, `d2w_dx2`, `dMxx_dx`, `dQx_dx`) in `MultiDomain.force_out_plane`;
  also in the tests, notebooks and theory scripts. The public API, the
  Bardell functions `fp` and `fpp` (`calc_vec_fp`, `integral_ffp`, ...),
  the names of the integrals of products (e.g. `fAufBuxi`) and
  the quadrature weights (`wx`, `wxi`, ...) are unchanged, and so are the
  results, bit for bit.
- The argument of the kernel `fkCf` of
  `panels.stiffener.models.bladestiff1d_clt_donnell` that receives `GJ/bf`
  is `GJperbf` (was `Jxx`, the name of the polar moment of area of
  `BladeStiff1D`), also in the generator in
  `theory/multidomain_panels/bladestiff1d_clt_donnell`.

### Dependencies

- Requires `composites>=0.10.0`, for `Laminate.intrhoz3`, `intrhoz4` and
  `intrhoz6`, and `structsolve>=0.9.3`, whose residual is `R = F_int -
  lambda F_ext`, with `K_T = dR/dc`, the convention of the documentation of
  `panels` (was `lambda F_ext - F_int` up to 0.9.2). The default
  convergence criterion of `structsolve` is Crisfield's since 0.9.0. 0.9.2
  fixes the arc-length methods, which could accept a step that converged
  back to the point
  before the last one and then trace the path backwards ("doubling back"):
  with 0.9.0 and 0.9.1 the Riks path of the `[90/0/90]` hinged roof with
  `h = 12.7` and the TSDT of `notebooks/sze2004_hinged_roofs.ipynb` turned
  back down the loading branch at the step through the limit point, and its
  limit load was 1780.7 instead of 1783.8.

## 0.13.11 (2026-10-16)

### Bug fix: one constitutive matrix for all the kernels

- The analytical `fk0` of the three models based on the classical laminated
  plate theory (`plate_clpt_donnell`, `cylshell_clpt_donnell`,
  `cylshell_clpt_sanders`) read `shell.lam.ABD` instead of `shell.ABD`, like
  every other kernel, therefore ignored `force_orthotropic_laminate=True`:
  the linear static and linear buckling analyses with the analytical `kC`
  used the anisotropic laminate, which was inconsistent with `calc_fint`,
  with the numerical tangent stiffness matrix and with the partial domains,
  e.g. a relative difference of `1.2e-1` between the analytical and the
  numerical `kC` for the laminate `[30, -60, 15]`. They now read
  `shell.ABD` and check that it is a 6 x 6 matrix; the generator
  `theory/shells/cylshell_clpt_sanders/derive_expressions.py` is fixed
  accordingly. No test, notebook or documented result used
  `force_orthotropic_laminate=True` with these models, no reference value
  changes.
- `Shell.calc_kG` with `c` integrated a new copy of the laminate matrix
  instead of the one of the other kernels.

### Changed: `Shell.ABD` is calculated from the laminate, new `Shell.ABDnxny`

- `Shell.ABD` is no longer an attribute, whose value set by the user was
  silently replaced by `_rebuild()` in `calc_kC`, `calc_kG`, `calc_fext`,
  etc.: it is a read-only property, calculated from `Shell.lam` at each
  access with the coupling terms zeroed by `force_orthotropic_laminate`,
  `None` before the laminate is built. Assigning it, also as
  `Shell(ABD=...)`, raises an `AttributeError`.
- New attribute `Shell.ABDnxny`, `None` by default, never allocated
  otherwise: a stiffness that is not that of the laminate, either constant,
  `(ne, ne)`, or one matrix per integration point, `(nx, ny, ne, ne)`, with
  `ne` = 6 (CLPT), 8 (FSDT) or 13 (TSDT). When set, `calc_kC`, `calc_kG`,
  `calc_kT`, `calc_kTv` and `calc_fint` integrate it numerically, the
  analytical `kC` being skipped, also within a `MultiDomain` and in the
  analyses of `structsolve`, which call these methods; the argument
  `ABDnxny` of these methods still takes precedence. With one matrix per
  point, `nx` and `ny` default to its shape. The mass, the thickness of the
  TSDT kinematics and the penalty constants of the connections still come
  from the laminate, as documented in `calc_kt_kr` and `tsl_resolution`.
- New `Shell.integration_points(nx=None, ny=None)`: the physical
  coordinates and weights of the Legendre-Gauss points, mapped to
  `Shell.integration_limits()`, `ABDnxny[i, j]` being the matrix of the
  point `(xs[i], ys[j])`.
- `Shell.stress` accepts a stiffness per point, `ABD` of shape `xs.shape +
  (ne, ne)`, and by default uses `Shell.ABDnxny`; with one matrix per
  integration point and without `xs`, `ys`, the stresses are calculated at
  the integration points. `MultiDomain.stress`, and through it
  `MultiDomain.calc_results`, `MultiDomain.force` and
  `MultiDomain.force_out_plane`, use a constant `ABDnxny` and raise a
  `ValueError` for one per point.
- The docstrings of `calc_kC`, `calc_fint` and the related methods document
  the shape and the point order of `ABDnxny`.
- `ABDnxny` is checked before it is integrated, since the kernels read only
  the upper entries of each block: a `ValueError`, naming the block and the
  integration point, is raised unless the matrix and its blocks `B` and, for
  the TSDT, `E`, `F` and `Dts` are symmetric, and the transverse shear is
  uncoupled from the other strains, as for any laminate, to `1e-10` times
  the largest entry. A matrix symmetric as a whole but with, e.g., `B12 !=
  B21` was integrated wrongly without error. The shape is checked as well.
- Saving and loading: `ABDnxny` is saved instead of `ABD`; the `ABD` of the
  files of older versions is read and discarded, being that of the
  laminate.
- Tests: `tests/tests_shell/test_abd_consistency.py`, for the 9 models: the
  analytical `kC` against the numerical one with and without
  `force_orthotropic_laminate`, for an anisotropic, unsymmetric laminate;
  the equilibrium of the linear static solution according to `calc_fint`;
  the buckling load of a forced orthotropic plate against the closed-form
  solution, single and divided in domains; the point order of `ABDnxny`
  against the strain energy summed over `Shell.integration_points()`, for
  full and partial domains with `nx != ny`, which a reversed field fails;
  `Shell.ABDnxny` against the argument, within a `MultiDomain`, in
  `Shell.stress` and saved and loaded; the checks of `ABDnxny`.

### Tests and coverage

- Every Python line added or changed for this version is covered by the
  tests, and so are the new checks of the analytical CLPT kernels, by
  calling them with a stand-in for `Shell`.
- New tests of the argument handling of `Shell`, in
  `tests/tests_shell/test_argument_validation.py`: `size='+n'` of
  `calc_kC`, `calc_kG`, `calc_kM`, `calc_kA`, `calc_kCfollower`,
  `calc_fint`, `calc_kTv` and `calc_fext_follower`; `Nxx_cte` in the
  analytical `calc_kC`; the errors of `_get_lam_ABD`,
  `integration_limits`, `global_frame`, `strain`,
  `add_distr_load_fixed_x` and `add_distr_pd_fixed_x`; and the errors of a
  model registered without the kernels of `calc_kTv`, of the follower
  pressure loads or of `calc_fint`.
- New tests of `MultiDomain.stress` and `MultiDomain.force` with
  `ABDnxny`, and of the penalty constants of the connections
  `'xcte-ycte'` and `'ycte-xcte'`.
- `Shell.global_frame` checks that `point_x` and `point_xy` have three
  coordinates before using them, instead of failing with a broadcasting
  error.
- `connections.build_panel_lam` is reduced to `Shell._rebuild()`, the
  laminate being always built by it, without the unreachable fallback.
- The branches of `MultiDomain.force_out_plane` disabled by hard-coded
  switches, the area integral and the von Kármán integral, are excluded
  from the coverage.

### Documentation: tangent stiffness of the CLPT models with Donnell kinematics

- The derivation of the internal force vector and of the tangent stiffness
  matrix of `plate_clpt_donnell` and `cylshell_clpt_donnell`, kept as the
  LaTeX file `theory/shells/tangent_stiffness_clpt_donnell.tex`, outside the
  documentation and with line numbers of the code that were out of date, is
  now a section of `doc/source/tangent_product.rst`, renamed "Tangent
  stiffness matrix and its product by a vector": the two integration paths
  and when each is taken, the approximation functions, the kinematics, the
  operators `B_L`, `B_NL` and `G`, the internal force vector, the grouping
  of the terms between `calc_kC` and `calc_kG` and why `K_GNL` is in
  `calc_kC`, the closed-form path, the quadrature, the multi-domain
  assembly, the symbols against the variables of the kernels, and the
  verification. Each equation names the function or kernel that evaluates
  it, checked against the current code, and the equations it needed from
  outside are written out. The `.tex` is removed and `*.tex` ignored.
- New docstrings of `MultiDomain.calc_kT` and `MultiDomain.calc_fint`.
- `MultiDomain.calc_kT` no longer has the argument `kw_tsl`, which was not
  used: the stiffness of a damaged cohesive connection is part of the
  matrix of the connections, `kC_conn`.

### New: product of the tangent stiffness matrix by a vector, without assembly

- `Shell.calc_kTv(c, v, ..., inc=1.)` and `MultiDomain.calc_kTv(c, v,
  inc=1.)` return `K_T(c) v`, with `K_T` of `Shell.calc_kT` and of
  `MultiDomain.calc_kC + calc_kG` with `NLgeom=True`, for all the 9
  geometrically non-linear models, at the cost of about one `calc_fint`
  instead of the assembly of `K_T`, which grows with the square of the
  number of terms: 20 to 300 times faster than the assembly for `m = n = 8`
  to `16`, see `doc/source/tangent_product.rst`. Exact to round-off,
  including the constant membrane stresses `Nxx_cte`, `Nyy_cte`, `Nxy_cte`,
  one constitutive matrix per integration point, partial integration
  domains, the follower pressure loads (an exact central difference of their
  force vector, at most quadratic in `c`) and, for `MultiDomain`, the
  penalty connections.
- New kernels `calc_kTv` in every `panels/models/<model>_num.pyx`,
  generated by the new `theory/shells/tangent_product`: at each integration
  point `K_T v = [B(beta)]^T [F] [B(beta)] v + [G]^T [N] [G] v`, with the
  approximation functions tabulated once per point; `tangent_product.py`
  verifies with SymPy, for every model, that it is the directional
  derivative of the internal force vector, and `tangent_product.md`
  describes the derivation.
- New documentation page `doc/source/tangent_product.rst`; a note in the
  Newton-Raphson example `doc/source/ex_newton_raphson.rst`.
- Tests: `tests/tests_shell/test_tangent_product.py` (the checks of
  `test_tangent_consistency.py` for the product: equality with `calc_kT @
  v`, Taylor test, finite-difference Jacobian, undeformed state, quadratic
  convergence of Newton-Raphson) and
  `tests/multidomain/test_tangent_product_multidomain.py`.

### Performance: prescribed displacements imposed exactly

- `MultiDomain.get_reduced_functions()` computes `K_T(c) c_p1` of
  `dR_r/dlbd = T^T (F_ext - K_T(c) c_p1)` with `MultiDomain.calc_kTv`
  instead of assembling `K_T`; the cache of the full tangent stiffness
  matrices of the last configuration, used only for this product, is
  removed, and `calc_kC` and `calc_kG` of the reduced functions assemble
  the full matrices directly. With the tabulation below and `structsolve`
  0.8.0, see Dependencies, the displacement-controlled post-buckling of a plate with
  `m = n = 12` runs in 11.8 s instead of 285 s (Newton-Raphson), 7.8 s
  instead of 705 s (modified Newton-Raphson) and 18.4 s instead of 363 s
  (Riks), with the same reactions. The run times of the penalty-vs-null-space
  study in `doc/source/multidomain_null_space.rst` are updated.

### Performance: approximation functions tabulated at the integration points

- Every numerically integrated kernel evaluates the approximation functions
  once per integration point and term, in tables read by the loops over the
  terms, instead of again in every loop, the innermost one included: the 9
  models (`fkC_num`, `fkG_num`, `calc_fint`, `fkM_num`, `fkAx_num`,
  `fkAy_num`, `calc_fext_follower`, `fkCfollower_num`), the connections
  (`kCSSxcte`, `kCSSycte`, `kCBFxcte`, `kCBFycte` with the shear deformation
  theories, `kCpd`, `kCSB_dmg`) and the blade stiffeners
  (`bladestiff1d_clt_donnell`, `bladestiff2d_clt_donnell`). Per integration
  point the number of evaluations falls from `O(m^2 n^2)` to `O(m + n)`:
  for `m = n = 12`, `fkC_num` with `NLgeom` 3.5 to 17 times faster, `fkG_num`
  30 to 60, `calc_fint` 10, `fkCfollower_num` 16 to 26, `fkM_num` 9 to 36,
  `kCSB_dmg` 57 and `kCSSxcte` 15; the DCB test with cohesive zone runs in
  2 min 45 s instead of 7 min 28 s. The results are identical, bit for bit:
  the outputs of 73946 calls of the 92 kernels recorded while running the
  tests of the models, follower pressures, partial domains, connections,
  cohesive zone and stiffened panels are the same as before.
- New `panels.dev.tabulate`, which rewrites the kernels, applied by the
  generators `fsdt_tsdt/write_pyx.py`, `cylshell_clpt_sanders/write_pyx.py`
  and `follower_pressure/write_pyx.py`, and with `python -m
  panels.dev.tabulate <file.pyx>` to the modules written by hand. The
  helper `_terms_sdt` of `kCBFxcte` and `kCBFycte` was tabulated by hand.
- New documentation page `doc/source/tabulation.rst`, comparing with the
  previous implementation.
- Tests: `tests/test_tabulate.py`, including a check that no kernel of the
  package evaluates the approximation functions in its loops any more.

### Cohesive zone (`'SB_TSL'`): discretisation rules and checks

- New module `panels.multidomain.connections.tsl` with the rules of the
  convergence study of `doc/source/cohesive_zone.rst`:
  `tsl_resolution(p1, k_o, tau_o, G1c, p2=None)` returns the decay rate of
  the opening `w1 - w2` of the two panels of the connection,
  `lambda = (k_o (1/D11_1 + 1/D11_2)/4)**0.25`, `(k_o/(2 D11))**0.25` for
  the symmetric DCB (by default `p2` has the `D11` of `p1`), and the cohesive
  zone length `l_cz = 0.88 E2 G1c/tau_o**2`; `tsl_margin(lam, lcz)` the
  length `3 l_cz + 8/lambda` needed ahead of the crack front, the process
  zone and the region where the opening decays as `exp(-lambda x)`, the
  solution of beams on an elastic foundation (Kanninen 1973), to below
  `exp(-8)` of its value; and
  `tsl_discretization(lam, lcz, L_tsl)` the terms `m = 1.25 lambda L_tsl`
  and Gauss points `nx = 10 L_tsl/l_cz` along `x`. The loads of the DCB are
  converged to 0.1% from `m = 1.0 lambda L_tsl`. The tag 0.13.9 used
  `(k_o/(4 D11))**0.25`, the rate of one panel on a rigid foundation,
  `2**-0.25` times the rate of the symmetric DCB, with the factors 1.5 and
  1.2 instead of 1.25 and 1.0; the linear compliance of the DCB of Krueger
  (2008) confirms the corrected rate (1.1005 times that of the simple beam
  theory with panels, 1.0973 for two beams connected by the foundation
  `k_o`, 1.116 with the rate of the tag 0.13.9). The margin of the tag
  0.13.9, `2 pi/lambda + 3 l_cz`, is `7.47/lambda + 3 l_cz` with the
  corrected rate; `2 pi/lambda + 3 l_cz` with the corrected rate would let
  the tractions reach the perfect bond in the last openings of the DCB
  validation runs, with differences between the load and the integral of
  the cohesive tractions up to 10%, against 0.25% at most with
  `3 l_cz + 8/lambda` when the basis follows the rules and the perfect bond
  is not wider than the bonded strip. The validation runs of the DCB notebooks were
  repeated with the corrected rules.
- `MultiDomain.calc_k_dmg` prints a warning, once, the first time a point is
  damaged, when the cohesive domain has fewer than `1.0 lambda a` terms or
  `7 a/l_cz` Gauss points along `x`, with `lambda` of both panels.
- The Gauss-Legendre points of `MultiDomain.calc_results` and
  `MultiDomain.force` are no longer limited to 304: the rules come from
  `scipy.special.roots_legendre`, without an upper limit.
- `connections.calc_kw_tsl` requires `k_i`, `tau_o` and `G1c` instead of
  silent defaults, accepts only `tsl_type='bilinear'`, and no longer has the
  unused interpenetration stiffness `k_ipen`; the damage and the stiffness
  are unchanged.
- Tests: `tests/multidomain/test_tsl_resolution.py`, including the rate of
  two different panels and its limit for a rigid second panel; in
  `tests/multidomain/test_multidomain_postprocessing.py`, the checks that
  more than 304 Gauss points raise a `ValueError` are replaced by checks
  that 305 points along `x` or `y` give the same results.

### Generators

- The whole-module generators `theory/shells/fsdt_tsdt/write_pyx.py` and
  `theory/shells/cylshell_clpt_sanders/write_pyx.py` tabulate the kernels,
  append `calc_kTv`, and write the `initializedcheck=False` directive of the
  committed modules, such that they reproduce them exactly;
  `theory/shells/follower_pressure/write_pyx.py` tabulates its kernels.

### Notebooks

- `doc/source/cohesive_zone.rst`: new convergence study of the cohesive
  domain, on the DCB of Tijs et al. (2022) with `L = 100` mm, and run times
  of panels 0.13.11.
- `notebooks/dcb_utils.py`: `discretize` without a cap on the Gauss points,
  `build_dcb` fills a missing `m_tsl` or `nx` from the rules, and the
  results store the case, the solver settings, the version of `panels` and
  the connection method; `run_or_load` reports a stored run of a different
  model.
- New `notebooks/README.md`, `notebooks/dcb_mode_I_solver_study.py` and
  `notebooks/dcb_mode_I_validation_rerun.py`.
- Renamed, after their scope, the mode I DCB: `notebooks/convergence_study.py`
  to `notebooks/dcb_mode_I_convergence_study.py`, with its results
  `notebooks/results/dcb_mode_I_convergence_study.json`, and
  `notebooks/validation_summary.ipynb` to
  `notebooks/dcb_mode_I_validation_summary.ipynb`.
- Renamed the DCB notebooks with the prefix `dcb_`: `alfano2001_dcb.ipynb`
  to `dcb_alfano2001.ipynb`, and in the same way `dcb_camanho2003.ipynb`,
  `dcb_krueger2008.ipynb`, `dcb_lecinana2023.ipynb`, `dcb_tijs2022.ipynb`,
  `dcb_tijs2023_phd.ipynb` and `dcb_turon2007.ipynb`.
- New `dcb_utils.opening_at_crack_length`: the last opening of the DCB
  notebooks is that at which the crack front of the model reaches the crack
  growth covered by the cohesive domain, from the effective crack length
  `a + l_cz + 1/lambda`, instead of the corrected beam theory, whose
  shorter crack let the last openings go beyond the cohesive domain.
- `dcb_tijs2022.ipynb`: the specimen with `a_0 = 40` mm is run with
  `G_Ic = 0.95` N/mm, the propagation toughness of the test DCB (2) of the
  paper, and the LEFM curves are given with the toughnesses of the model and
  the initiation toughness, 0.7 N/mm.

### Dependencies

- Requires `structsolve>=0.8.0`, which evaluates `dR_r/dlbd` only together
  with the tangent stiffness matrix, saving evaluations of
  `MultiDomain.calc_kTv`, and reuses the factorizations of the tangent
  stiffness matrix in the modified Newton-Raphson method.

### Maintenance

- `.gitignore` without the runs of `thesis_timing.py`, which was moved out
  of the repository.

## 0.12.0 (2026-10-14)

### New: prescribed displacements imposed exactly with the null-space method

`MultiDomain(..., pd_method='null-space')` imposes the prescribed
displacements of `Shell.add_point_pd`, `Shell.add_distr_pd_fixed_x` and
`Shell.add_distr_pd_fixed_y` exactly, as the inhomogeneous linear constraints
`B_d c = d_0 + lbd*d_1`, `d_0` from the constant prescriptions
(`cte=True`) and `d_1` from those incremented with the load factor
(`cte=False`), eliminated together with the connections:
`c = T c_r + c_p0 + lbd*c_p1`. The default `pd_method='penalty'` keeps the
previous behaviour, bit for bit. See the new section "Prescribed
displacements imposed exactly" of `doc/source/multidomain_null_space.rst`.

- `nullspace.Elimination`: the elimination of `null_space_basis` one block
  at a time, unchanged for the connections (`T` is bit-identical), keeping
  the data of the blocks with a right-hand side, from which the particular
  solutions are computed for any prescribed values without a new
  decomposition. New `nullspace.pd_constraint_matrix`, `pd_points`,
  `pd_blocks`, `pd_entries`, `pd_rhs` and `PD_COMPONENTS`. The constraints
  are taken at `max(n, 4)` (or `max(m, 4)`) Gauss-Legendre points along a
  line, exact along the whole line for polynomial prescribed functions up to
  that degree.
- `MultiDomain.get_T(conn=None, tol=None, rhs_tol=None)` also eliminates the
  prescribed displacements; `T` is rebuilt only when their positions or
  components change. New `MultiDomain.get_cp()` returns `c_p0` and `c_p1`
  (attributes `c_p0`, `c_p1`), reading the prescribed values again from the
  panels.
- `MultiDomain.expand(c_r, lbd=None)`: with `lbd`, the configuration
  `T c_r + c_p0 + lbd*c_p1`; without it, `T c_r`, as before, e.g. for modes.
- `MultiDomain.get_reduced_functions()`: with prescribed displacements, the
  callables build the configuration from `inc`, `calc_kG` declares `inc`,
  and `calc_fext(inc, c=c, lbd=lbd)` returns `dR_r/dlbd =
  T^T (F_ext - K_T(c) c_p1)`, which `structsolve>=0.7.0` uses in the
  predictors, in the arc-length constraint and in the convergence criterion.
  The full tangent stiffness matrix of the last configuration is cached, so
  that `dR_r/dlbd` costs no extra assembly in the arc-length methods.
  `Analysis.static(NLgeom=False)` raises an error pointing to the new
  `MultiDomain.linear_static()`. Without prescribed displacements the
  callables are unchanged.
- New `MultiDomain.linear_static(lbd=1.)` and
  `MultiDomain.linear_buckling(num_eigvalues=25, c=None, **kwargs)`, whose
  pre-buckling state `c_0 + lbd*c_1` separates the constant prescribed
  displacements.
- Reactions: `MultiDomain.reaction_pd(c, lbd=1., NLgeom=True)`, the
  generalized reactions `c_pj^T (F_int(c) - lbd*F_ext)` work-conjugate to the
  constant and the incremented prescriptions, independent of the particular
  solution; `MultiDomain.reaction_resultant(c, lines, lbd=1., NLgeom=True)`,
  the resultant forces along `u`, `v`, `w` of selected prescribed lines or
  points.
- Redundant prescriptions are dropped when consistent, e.g. at corners with
  connections or other prescriptions, or on terms removed by boundary
  condition flags; inconsistent ones raise a `ValueError` naming the panels,
  the prescriptions, the flags to release and the constraints involved.
- Prescribed rotations, `kphix`, `phixp`, `kphiy`, `phiyp` of
  `Shell.add_point_pd` and `kphix`, `funcphix`, `kphiy`, `funcphiy` of the
  distributed ones, only with `pd_method='null-space'`; with the penalty
  method `calc_fext` raises `NotImplementedError`.
- `Shell` stores the prescribed values in the new attributes
  `point_pds_values`, `point_pds_values_inc`, `distr_pds_values` and
  `distr_pds_values_inc`. `json_io` saves the point values, together with
  `MultiDomain.pd_method` and the `rhs_tol` of `MultiDomain.get_T()`; the
  distributed ones contain functions and, as `distr_pds`, cannot be saved.
  A penalty constant `None` with a value now prescribes nothing in
  `add_point_pd`, instead of raising `TypeError`.
- `Shell.calc_fext(..., include_pds=True)` and `shell_fext(...,
  include_pds=True)`; `MultiDomain.calc_fext` excludes the forces of the
  prescribed displacements with `pd_method='null-space'`.
- A single panel is wrapped as `MultiDomain([shell], conn=[],
  pd_method='null-space')`.
- Tests: `tests/multidomain/test_null_space_pd.py`.

### Bug fix

- `fkCld_xcte` and `fkCld_ycte` (`panels/multidomain/connections/kCpd.pyx`)
  placed the penalty stiffnesses of `ku` and `kv` at the rows and columns of
  `w` instead of `u` and `v`, a transcription error with respect to
  `theory/multidomain_penalization/kCpd.py`, which generates the correct
  entries. Only the line prescribed displacements of `u` and `v` with the
  penalty method are affected; the prescribed `w` of the cohesive zone and
  DCB examples is unchanged. New test `test_line_penalty_kernels`.

### Tests

- `tests/test_json.py` compares the eigenpairs of the saved and loaded
  models with a tolerance, eigenvectors up to the sign, since they are not
  reproducible to the last bit across calls of the eigensolver on macOS.
  The matrices, stored results and fields are still compared exactly.

### Dependencies

- Requires `structsolve>=0.7.0`.

## 0.11.1 (2026-10-08)

### Saving and loading in a JSON + NumPy zip file

A zip file replaces pickle as the way to save objects to files. The new
module `panels.json_io` mirrors `composites.json_io`:

- `Shell.save(fname=None)`, `StiffPanelBay.save(fname=None)` and the new
  `MultiDomain.save(fname=None)` write `<name>.shell.zip`,
  `<name>.stiffpanelbay.zip` and `<name>.multidomain.zip` by default, or to
  the given path or binary file object, e.g. `io.BytesIO`, which is how a
  browser application saves without a file system.
- `panels.shell.load(name)`, `panels.stiffpanelbay.load(name)` and the new
  `panels.multidomain.load(name)` accept the name with or without the
  extension, a path or a binary file object, and check the type of the saved
  object. `panels.json_io.save(obj, fname)` and `panels.json_io.load(fname)`
  handle the three classes.
- New attribute `MultiDomain.name`, also a keyword argument of
  `MultiDomain`, default `'multidomain'`.
- The zip file, compressed with `ZIP_DEFLATED`, contains `model.json`, with
  the envelope `{"type", "format_version", "panels_version", "data"}`, and
  one `arrays/<key>.npy` member per NumPy array, referenced in the JSON as
  `{"__ndarray__": "arrays/<key>.npy"}`. The inputs (geometry, `m`, `n`,
  boundary conditions, `stack`, `plyts`, `laminaprops`, `offset`, loads,
  names) are stored in JSON, the arrays (`ABD`, `fields`, `plot_mesh`,
  `results`, `StiffPanelBay.u`, `MultiDomain.dmg_index`, ...) in `.npy`.
  The laminates, e.g. `Shell.lam` and `BladeStiff1D.flam`, are embedded with
  `composites.to_dict`. The JSON is strict: `nan` and `inf` are stored as
  the strings `"NaN"`, `"Infinity"` and `"-Infinity"`.
- The panels and stiffeners of a `StiffPanelBay` and the panels and
  connections of a `MultiDomain` are saved with them, and the references to
  the panels (`panel1`, `panel2`, `p1`, `p2`) are restored as references to
  the loaded panels.
- An explicit list of attributes is stored for each class. Saving an
  attribute whose value is not supported, e.g. the function of a distributed
  load or of a pressure `p(x, y)`, raises `TypeError` and writes no file. The
  matrices (`Shell.matrices`, `StiffPanelBay.kC`, `MultiDomain.kC`,
  `MultiDomain.T`, ...) and the number of threads `out_num_cores` are not
  stored.
- `save()` no longer modifies the object: the pickle version cleared its
  matrices, laminate and field outputs.
- Loading is safe for files from untrusted sources: the arrays are read with
  `allow_pickle=False`, the members are read in memory and never extracted,
  only the members referenced by `model.json` are read, after checking that
  their uncompressed sizes declared in the zip file do not exceed the new
  argument `max_size` of the functions `load`, 4 GiB by default, and unknown
  keys, missing or extra members, object arrays and newer format versions
  raise `ValueError`.
- The pickle files `.Shell` and `.StiffPanelBay` of older versions are no
  longer loaded, since a pickle file can execute arbitrary code: `load()`
  raises `ValueError` for any file that is not a zip file saved by panels.
  One from a trusted source can be converted by loading it with
  `pickle.load()` and saving the object again. Pickling is still supported
  for `copy.deepcopy` and for passing objects between processes, e.g. with
  `multiprocessing`.
- Requires `composites>=0.9.21`, which provides `composites.json_io`.
- Tests in `tests/test_json.py`.

### Pyodide (WebAssembly)

- A WebAssembly wheel for Pyodide 314 (CPython 3.14), tagged
  `pyemscripten_2026_0_wasm32` (PEP 783), is built with cibuildwheel 4.3.0,
  tested in Pyodide on every push (new workflow `pyodide.yml`, with the test
  suite split in two parallel jobs to stay within the 6 h limit of the
  runners) and published on PyPI on tags (new job `deploy-pyodide` of
  `pythonpublish.yml`), such
  that `await micropip.install("panels")` works in the browser. NumPy, SciPy
  and matplotlib come with Pyodide, `structsolve` is pure Python and
  `composites` has its WebAssembly wheel on PyPI.
- `setup.py` detects the Emscripten cross-compilation of `pyodide-build` and
  builds without OpenMP, with which the `prange` loops of
  `panels.models.clpt_field` run serially, and without the GCC runtime link
  flags. A coverage build with line tracing is never made for Emscripten.
- The runtime dependencies are declared in `pyproject.toml`: `numpy`,
  `scipy`, `matplotlib`, `composites>=0.9.21` and `structsolve>=0.6.1`,
  the first version of structsolve supporting Pyodide.
  The build requires only `setuptools`, `wheel` and `cython`, since no Cython
  module cimports NumPy or composites.
- Fixed a heap buffer overflow of `panels.models.clpt_field`, harmless on
  64-bit platforms only: its work arrays of doubles were allocated with the
  size of pointers, `malloc(NMAX * sizeof(double *))`, half the size needed
  in WebAssembly, where pointers have 4 bytes, crashing Pyodide.
- `out_num_cores` defaults to `os.cpu_count() or 1` instead of
  `multiprocessing.cpu_count()`, which raises where the number of CPUs is
  unknown.
- The unused import of `multiprocessing.Pool` was removed from
  `tests/multidomain/test_dcb_damage.py`.
- The tests run with `MPLBACKEND=Agg` in Pyodide.
- New classifier `Environment :: WebAssembly :: Emscripten`.
- Installation in Pyodide, with a browser example saving to `io.BytesIO`
  for download, in the README and the documentation.

### Performance

- Removed the explicit `gc.collect()` after each matrix and internal force
  vector of `Shell`, `MultiDomain`, `StiffPanelBay`, `BladeStiff1D` and
  `BladeStiff2D`. The arrays are freed as soon as they are no longer
  referenced, a full collection only reclaims reference cycles, which Python
  collects automatically, and it walks every object of the process: it took
  70 % of the time of the non-linear analysis with the cohesive zone of
  `tests/multidomain/test_dcb_damage.py`. The memory of 300 repeated
  calculations of `calc_kC`, `calc_kG` and `calc_fint` is the same with and
  without it, the memory leak mentioned in the removed comments is no longer
  observed. `_clear_matrices()` still calls it.

### Tests

- `tests/multidomain/test_dcb_damage.py` checks its results, it had no
  assertion: every increment converged, the driver aborting silently
  otherwise, the reaction equals the area integral of the cohesive tractions,
  the load-displacement curve and the damage match the reference results.
  Its results are written to a temporary directory instead of the working
  directory. 446 s instead of 1514 s with the removal of `gc.collect()`.

### Maintenance

- GitHub Actions updated: `actions/checkout@v7`, `actions/setup-python@v7`,
  `actions/upload-artifact@v7`, `codecov/codecov-action@v7`,
  `softprops/action-gh-release@v3` and
  `JamesIves/github-pages-deploy-action@v4`.
- Removed the outdated comment that composites laminates cannot be pickled,
  which is no longer true since composites 0.9.2.
- The coverage workflow, cancelled at the limit of 6 h of the runners since
  2026-09-30, runs the test suite in three parallel parts, each uploading its
  report to Codecov, which waits for the 3 reports (new `codecov.yml`). The
  coverage build is optimized, `/O2` or `-O3` instead of `/Od` or `-O0`: every
  traced line being a call with its line number, the coverage report is the
  same, and the traced tests run 24 % faster.

## 0.11.0 (2026-10-07)

### New: follower (hydrostatic) pressure loads

`Shell.add_pressure_load(p, ..., follower=False, zp=0.)`:

- `follower=True` (or `'linear'`) makes the pressure follow the deformed
  mid-surface, normal to it and acting on its deformed area, with the area
  vector truncated to first order in the displacement gradients, consistent
  with the moderate-rotation strains of all models: `a = (-w,x, -w,y + kv v,
  1 + u,x + v,y + kw w)`, `kw = 1/r` for the cylinders and `kv = 1/r` for the
  Sanders kinematics. `follower='quadratic'` keeps the complete (bilinear)
  area vector, for sensitivity checks. See
  `theory/shells/follower_pressure/follower_pressure.md`.
- `zp` places the loaded surface at `z = zp`, e.g. the outer face of a thick
  sandwich: the pressure is applied as `p (1 + zp/r)` per unit mid-surface
  area of the cylinders (shallow shell).
- In the undeformed state a follower load equals the dead load, so
  `Shell.calc_fext()` is unchanged. `Shell.calc_fext(inc, c=c)` gives the load
  vector of the configuration `c`.
- `Shell.calc_fint(c, inc=1.)` subtracts the configuration-dependent part of
  the follower loads, scaled by the load factor `inc` for `cte=False`.
- New `Shell.calc_kCfollower(c=None, inc=1.)`: the load stiffness
  `-dF_p/dc`, unsymmetric in general, assembled in full and never
  symmetrized, stored in `Shell.matrices['kCfollower']`. It is added by
  `Shell.calc_kC(c, NLgeom=True, inc=1.)` (with `finalize=True`) and therefore
  by `Shell.calc_kT(c, inc=1.)`, the exact Jacobian of `calc_fint`. Linear
  buckling under hydrostatic pressure: `lb(kC0, kG + kCfollower)`.
- New `Shell.calc_fext_follower(c, inc=1.)` and `Shell.has_follower_loads()`.
- The kernels `calc_fext_follower` and `fkCfollower_num` of the 9 models are
  generated by `theory/shells/follower_pressure/write_pyx.py`, also called by
  the generators of the FSDT/TSDT and Sanders CLPT modules.
- `MultiDomain.calc_fint`, `calc_kC`, `calc_kT`, `calc_fext` and
  `get_reduced_functions()` forward `inc` and `c`, and add the follower load
  stiffness of the panels after the symmetrization.
- Requires `structsolve>=0.5.0`: its solvers pass the load factor `inc` to
  the callables that accept it, the arc-length methods use the load vector
  of the current configuration, `lb` and `freq` handle unsymmetric matrices,
  `Analysis.static(NLgeom=False)` solves the geometrically linear problem
  with the follower load in the current configuration, `(k0 + kCfollower) c
  = fext`, and `freq` gives the kinetic criterion (flutter) of the tangent
  stiffness matrix with the unsymmetric load stiffness.
- Theory: new documentation page *Follower pressure loads: theory*,
  `doc/source/follower_pressure.rst`, with the derivation of the force
  vector, the internal force vector, the load stiffness `kCfollower` and its
  symmetry conditions.
- Verification: `tests/tests_shell/test_follower_pressure.py`; validation
  against closed forms and the literature (Han et al. 2004, Kardomateas 1993,
  Kardomateas and Simitses 2003, Schweizerhof and Ramm 1984, NASA/SP-8007-2020/REV 2):
  `tests/tests_shell/test_follower_pressure_validation.py` and, per
  reference, the notebooks `han2004_sandwich_rings.ipynb`,
  `kardomateas2003_sandwich_rings.ipynb`, `kardomateas1993_thick_rings.ipynb`,
  `schweizerhof1984_pressure_loads.ipynb` and
  `nasa_sp8007_external_pressure.ipynb`, with the helpers of
  `notebooks/follower_utils.py`.

### New: interactive 3D plots of multi-domain assemblies

`MultiDomain.plot3d(c)` draws the assembly in 3D with plotly, which is an
optional dependency (`pip install plotly`). Each domain is placed in the
global coordinate system by its origin `(x0, y0, z0)`, a point `point_x` on
its `x` axis and a point `point_xy` on its `xy` plane, the new attributes
`Shell.z0`, `Shell.point_x` and `Shell.point_xy`:

    vec_x = point_x - (x0, y0, z0)
    vec_xy = point_xy - (x0, y0, z0)
    vec_z = np.cross(vec_x, vec_xy)
    vec_y = np.cross(vec_z, vec_x)

By default the local axes are the global ones, as in the 2D plots of
`MultiDomain.plot()`. The cylindrical shells are drawn curved, with `w`
positive outwards. The domains are drawn deformed by the displacements `u`,
`v` and `w` times a scale factor, by default such that the largest
displacement is 10 % of the size of the assembly, and colored by one of the
outputs of `MultiDomain.plot()`, `'u'`, ..., `'exx'`, ..., `'Nxx'`, ...,
selected in a drop-down menu. Saved with `filename` to an HTML file, or shown
with `show=True` in a browser or a notebook, the figure has controls to
toggle `u`, `v` and `w` and to change the scale factor, and keyboard
shortcuts while the mouse pointer is over it: `u`, `v`, `w` toggle the
displacements, `+` and `-` change the scale factor, `0` shows the undeformed
geometry, `r` resets, `n` and `p` select the next and previous outputs and
`o` toggles the orthographic and perspective projections.

- `Shell.global_frame()` gives the origin and the local axes of a domain,
  `Shell.global_coords(x, y)` the global coordinates of points of its
  mid-surface and the directions of `u`, `v` and `w` at them, and
  `Shell.is_curved()` tells whether it is drawn as a cylindrical surface.

### Build: Cython directives and compiler flags

- Every module is compiled with `initializedcheck=False`. Cython no longer
  checks, at each memoryview access, that the memoryview has been assigned.
  Reading an unassigned memoryview attribute from Python still raises
  `AttributeError`, because the property getters keep their own check.
- `setup.py` passed `linetrace=True` to every build. Without `CYTHON_TRACE`
  the line tracing itself stays compiled out, but the directive also
  generates profiling hooks that are active by default and run at every call
  of a Python-visible function. The directive is now set only for a coverage
  build, see below.
- Compiler flags: `-O3` and `-fno-math-errno` for GCC and Clang, and an
  explicit `/O2` for MSVC, which setuptools already combines with `/GL`.
  OpenMP stays, since `panels/models/clpt_field.pyx` uses `prange`. Flags
  that change floating-point results, such as `/fp:fast` or `-ffast-math`,
  and flags that tie a wheel to the CPU that built it, such as
  `-march=native`, are deliberately not used.

The same changes made `pyfe3d` and `composites` up to 2.8 times and 24%
faster, but not `panels`. Its memoryviews are local variables, which Cython
already knows to be assigned, so the generated code had almost no checks for
`initializedcheck=False` to remove (2 per model), and each matrix is built by
a handful of calls, which makes the per-call cost of the profiling hooks
negligible. The cost does not change measurably. Measured on an AMD x86-64 laptop CPU (family 25, model 117),
Windows 11, Python 3.13, NumPy 2.4, Cython 3.2, MSVC, with the process pinned
to one core at high priority, minimum of 3 interleaved runs of 5 repetitions
each (`benchmarks/bench_build_flags.py`), in milliseconds per call for
`m = n = 12`; `kC`, `kG` and `kM` are integrated analytically, and `kC(c)`,
`kG(c)` and `fint(c)` numerically. Each cell gives the cost before and after
this change, with the relative difference:

| Model | kC | kG | kM | kC(c) | kG(c) | fint(c) |
| - | - | - | - | - | - | - |
| `plate_clpt_donnell` | 24.09 / 23.86 (-1%) | 7.19 / 7.45 (+4%) | 17.49 / 17.89 (+2%) | 1583.66 / 1602.44 (+1%) | 693.77 / 686.89 (-1%) | 41.69 / 41.18 (-1%) |
| `cylshell_clpt_sanders` | 25.31 / 24.53 (-3%) | 12.36 / 12.86 (+4%) | 17.63 / 17.82 (+1%) | 2146.72 / 2133.46 (-1%) | 723.22 / 722.30 (-0%) | 44.60 / 44.83 (+1%) |
| `plate_fsdt_donnell` | 43.09 / 42.57 (-1%) | 7.18 / 7.02 (-2%) | 20.34 / 20.55 (+1%) | 2753.77 / 2714.43 (-1%) | 486.27 / 492.08 (+1%) | 60.70 / 60.13 (-1%) |
| `cylshell_fsdt_sanders` | 50.74 / 50.03 (-1%) | 12.78 / 12.92 (+1%) | 20.50 / 20.38 (-1%) | 3094.37 / 3032.40 (-2%) | 732.51 / 727.61 (-1%) | 64.52 / 63.79 (-1%) |
| `cylshell_tsdt_donnell` | 53.26 / 53.72 (+1%) | 7.09 / 7.08 (-0%) | 32.71 / 31.68 (-3%) | 4439.99 / 4463.94 (+1%) | 489.35 / 489.04 (-0%) | 62.79 / 61.77 (-2%) |

All differences are within the measurement noise of about 4%. The whole test
suite passes with the new build.

### Coverage build

`setup.py` now recognises a coverage build, requested either with
`CYTHON_TRACE_NOGIL` in the environment or with `--define CYTHON_TRACE_NOGIL`,
and compiles it unoptimized with line tracing. Since Python 3.12 Cython
traces through `sys.monitoring` by default, which the `Cython.Coverage`
plugin does not follow, so the coverage build now requests the legacy tracing
with `CYTHON_USE_SYS_MONITORING=0`. `.github/workflows/coverage.yml` runs on
Python 3.13, where the same setup left every `.pyx` file out of the coverage
report of `composites`; this was not measured for `panels`. Switching between
a coverage build and a normal one regenerates the C++ files of the Cython
modules, which cythonize would otherwise reuse from the other mode, and
leaves the hand-written sources in `panels/core/src` alone.

### Changed

- The TSDT constitutive matrix `Shell.ABD` uses `Laminate.Dts` and
  `Laminate.Fts` of `composites`, which replace the deprecated
  `Laminate.Dtrans` and `Laminate.Ftrans`. The values are the same.
- Requires `composites>=0.9.12`, the first version with `Laminate.Dts` and
  `Laminate.Fts`.

## 0.9.0 (2026-09-24)

### Breaking: the multi-domain connections are exact by default

`MultiDomain` and the functions of `panels.multidomain.cylinder`,
`panels.multidomain.cylinder_blade_stiffened` and
`panels.multidomain.tstiff2d_1stiff_*` use `conn_method='null-space'` by
default, see below. The matrices and vectors of `MultiDomain`, e.g.
`calc_kC()`, `calc_kG()`, `calc_kM()` and `calc_fext()`, then no longer
contain the connections imposed exactly: they must be reduced with
`MultiDomain.reduce()` before solving, and the solutions expanded with
`MultiDomain.expand()`, otherwise the domains are disconnected. Scripts that
solve these matrices directly must either reduce them or create the assembly
with `conn_method='penalty'`, which gives the previous results. The functions
of `panels.multidomain` already return the expanded solutions. The tests of
the penalty kernels and the notebooks
`stamatelos_labeas_2023_multidomain.ipynb` and those of the DCB, through
`notebooks/dcb_utils.py`, use `conn_method='penalty'`, with which their
stored results were obtained.

### New: exact multi-domain connections with the null-space method

`MultiDomain(panels, conn, conn_method='null-space')` imposes the connections
`'SSxcte'`, `'SSycte'`, `'BFxcte'`, `'BFycte'` and `'SB'` exactly, for all
the models, instead of adding penalty stiffnesses. The quantities penalized
by each kernel become linear constraints `B c = 0`, eliminated with a sparse
basis `T` of their null space, `c = T c_r`, see
`panels.multidomain.connections.nullspace` and the documentation page
"Exact connections with the null-space method". No penalty constant is
needed, the results are the limit of the penalty method for penalty
constants going to infinity, and the reduced stiffness matrix is much better
conditioned. The penalty method remains available with
`conn_method='penalty'`, whose results are unchanged.

- `MultiDomain.get_T()`, `MultiDomain.reduce()` and `MultiDomain.expand()`
  reduce the matrices and vectors to the independent Ritz constants and
  expand the solutions back, and `MultiDomain.get_reduced_functions()` gives
  the functions of the reduced problem for `structsolve.Analysis`, for the
  non-linear analyses. With `conn_method='penalty'`, `T` is the identity.
- Redundant constraints, e.g. around a closed cylinder or at a corner shared
  by four domains, are detected and dropped.
- The damaged connection `'SB_TSL'` and the prescribed displacements remain
  penalty stiffnesses.
- `MultiDomain.get_kC_conn()`, and therefore `calc_kC()`, `calc_kT()` and
  `calc_fint()`, called without connections on an assembly created without
  them, use the connections of the last call, e.g. `calc_kC(conns)`, instead
  of raising a `RuntimeError`.
- The functions of `panels.multidomain.cylinder`,
  `panels.multidomain.cylinder_blade_stiffened` and
  `panels.multidomain.tstiff2d_1stiff_*` take the argument `conn_method`.
- A plate divided in domains gives the frequencies of the single-domain
  plate to `1e-8` or better with the FSDT and TSDT, where the penalty method
  needs 1000 times the default penalty constants to reach `1e-4`. Two
  laminates connected with `'SB'` are the single laminate with both stacking
  sequences to machine precision, also in large deflection. The default
  penalty constants underestimate the buckling load of the blade-stiffened
  cylinder of `tests/multidomain/test_cylinder_blade_stiffened.py`, with the
  pre-buckling state of a static analysis, by 4%: -39993.8 N/m against
  -41696.7 N/m.

### Documentation

- The panels `p1` and `p2` of the connections `'SB'` and `'SB_TSL'` are
  defined by the `z` axis common to both panels, normal to their
  mid-surfaces, instead of top and bottom: `p1` is the panel on the positive
  side of the interface along `z` and `p2` the one on the negative side, so
  that the face `z = -h1/2` of `p1` is connected to the face `z = +h2/2` of
  `p2`, e.g. a stiffener base on the side of positive `z` of a skin is `p1`,
  and a pad-up on the side of negative `z` is `p2`. The convention of the
  kernels is unchanged.

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
  `Shell.fsdt_shear_correction`: `'rohwer'`, the default, `'vlachoutsis'`,
  `'constant'` or `None` select the method of
  `composites.Laminate.calc_transverse_shear_stiffness`, whereas a float `k`
  multiplies the uncorrected transverse shear stiffness. The equilibrium
  approach of Rohwer (1988) gives `k = 5/6` for a homogeneous plate and
  accounts for the stacking sequence otherwise. The tests compared with
  references that use `k = 5/6` set it explicitly.
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
