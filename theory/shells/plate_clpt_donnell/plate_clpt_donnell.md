# Flat plate, CLPT with von Kármán strains (`plate_clpt_donnell`)

Derivation: `plate_clpt_donnell.py` (sympy, run `python
plate_clpt_donnell.py`), which writes the integrands of the matrices to
`output_expressions_python/sympy_<name>.txt`. No script writes the kernels
of this model, `panels/models/plate_clpt_donnell.pyx` (integrated
analytically) and `panels/models/plate_clpt_donnell_num.pyx` (integrated
numerically); the kernels `calc_kTv`, `calc_fext_follower` and
`fkCfollower_num` at the end of the latter are written by
`../tangent_product/write_pyx.py` and `../follower_pressure/write_pyx.py`.

## Kinematics

Degrees of freedom of a term: `u, v, w`, the displacements of the
mid-surface along the fixed axes `x, y, z` of the plate. Each is a sum of
products of Bardell's functions along `x` and `y`,

    u = sum_ij c_ij^u f_i(xi) g_j(eta),   xi = 2x/a - 1,   eta = 2y/b - 1

and likewise `v`, `w`, with their own boundary flags. Displacement field of
the classical laminated plate theory (Kirchhoff-Love), `z` measured from the
reference surface:

    u(z) = u - z w,x,   v(z) = v - z w,y,   w(z) = w

## Strains

Von Kármán strains, the quadratic terms of the rotations `w,x`, `w,y` only:

    exx = u,x + w,x^2/2
    eyy = v,y + w,y^2/2
    gxy = u,y + v,x + w,x w,y
    kxx = -w,xx,   kyy = -w,yy,   kxy = -2 w,xy

In the script the strain-displacement matrix `BA` (6 x 3 per term) has the
linear part `B0A` and the part linear in `w_xi`, `w_eta`, the derivatives
of `w` at the point, and `GA = (w,x, w,y)` gives the rotations.

## Constitutive law

Classical laminated plate theory:

    (Nxx, Nyy, Nxy, Mxx, Myy, Mxy) = [ABD] (exx, eyy, gxy, kxx, kyy, kxy)

with the `6 x 6` matrix `[ABD] = [[A, B], [B, D]]` of the laminate,
`Shell.ABD`, about the reference surface (laminate between `-h/2 + d` and
`h/2 + d`, offset `d`).

## Matrices

All integrands are per unit area of the natural domain, the factor
`intx*inty/4` mapping them onto `x1 <= x <= x2`, `y1 <= y <= y2`:

    kC   = BA^T [ABD] BB          constitutive stiffness, with KCNL
    kC0  = B0A^T [ABD] B0B        linear constitutive stiffness
    kG   = GA^T [N] GB            geometric stiffness, [N] = [[Nxx, Nxy], [Nxy, Nyy]]
    fint = BA^T (N, M)            internal force vector
    kM   = gA5^T [I] gB5          mass, gA5 = (u, v, w, -w,x, -w,y)
    kAx  = -beta  w,x^T w         piston theory, flow along x
    kAy  = -beta  w,y^T w         piston theory, flow along y
    cA   = -aeromu w^T w          aerodynamic damping

with `[I]` built from the moments `I_n = int rho z^n dz`, `n = 0, 1, 2`
(`intrho`, `intrhoz`, `intrhoz2` of `composites.Laminate`). Flat plates
have no curvature term `gamma` in `kAx`, see `Shell.calc_kA`.

Kernels of `plate_clpt_donnell.pyx`: `fk0`, `fkG0` (constant `Nxx`, `Nyy`,
`Nxy`), `fkM`, `fkAx`, `fkAy`, `fcA`, integrated in closed form with the
integrals of Bardell's functions (constant `[ABD]` and density moments,
whole domain). Kernels of `plate_clpt_donnell_num.pyx`: `fkC_num`,
`fkG_num`, `fkM_num`, `fkAx_num`, `fkAy_num`, `calc_fint`, with the
Gauss-Legendre rule, `nx = max(2m - 1, 7)` points along `x` by default
(`panels.shell.gauss_points`), which admits `[ABD]` varying over the
domain, partial domains and the non-linear terms.

## Selection in panels

`Shell(..., model='plate_clpt_donnell')`; it is the default model of a
`Shell` without radius (`r=None`). `panels.modelDB.db['plate_clpt_donnell']`:
`dofs = 3`, `e_num = 6`, `requires_r = False`. It is also the model of the
flange of `BladeStiff2D`.

## Limitations

- The von Kármán strains are not invariant under large rigid rotations,
  e.g. a rigid in-plane rotation `t` gives `exx = cos t - 1`. For
  assemblies of plates with large rotations use `plate_clpt_gl` or
  `plate_clpt_tl` (`../plate_clpt_gl`, `../plate_clpt_tl`).
- No transverse shear deformation; see `plate_fsdt_donnell` and
  `plate_tsdt_donnell` (`../fsdt_tsdt`).

## References

- Bardell, N. S., "Free vibration analysis of a flat plate using the
  hierarchical finite element method", Journal of Sound and Vibration,
  151(2), 263-289, 1991. https://doi.org/10.1016/0022-460X(91)90855-E
- Castro, S. G. P., and Donadon, M. V., "Assembly of semi-analytical models
  to address linear buckling and vibration of stiffened composite panels
  with debonding defect", Composite Structures, 160, 232-247, 2017.
  https://doi.org/10.1016/j.compstruct.2016.10.026
