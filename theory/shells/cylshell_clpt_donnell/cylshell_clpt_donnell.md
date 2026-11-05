# Cylindrical shell, CLPT with Donnell kinematics (`cylshell_clpt_donnell`)

Derivation: `cylshell_clpt_donnell.py` (sympy, run `python
cylshell_clpt_donnell.py`), which writes the integrands of the matrices to
`output_expressions_python/sympy_<name>.txt`. No script writes the kernels
of this model, `panels/models/cylshell_clpt_donnell.pyx` (integrated
analytically) and `panels/models/cylshell_clpt_donnell_num.pyx` (integrated
numerically); the kernels `calc_kTv`, `calc_fext_follower` and
`fkCfollower_num` at the end of the latter are written by
`../tangent_product/write_pyx.py` and `../follower_pressure/write_pyx.py`.

The script is that of `../plate_clpt_donnell` with the hoop strain `w/r`
and the curvature term of the piston theory.

## Kinematics

Cylinder of radius `r`, `x` along the axis, `y = r theta` the arc length
along the circumference, `w` positive outwards, and the shallow-shell
assumption `r + z ~ r`. Degrees of freedom of a term: `u, v, w`, each a sum
of products of Bardell's functions `f_i(xi) g_j(eta)`, `xi = 2x/a - 1`,
`eta = 2y/b - 1`. Displacement field of the classical laminated plate
theory, `z` measured from the reference surface:

    u(z) = u - z w,x,   v(z) = v - z w,y,   w(z) = w

## Strains

Donnell-Mushtari-Vlasov kinematics, with the von Kármán terms of the
rotations `w,x`, `w,y`:

    exx = u,x + w,x^2/2
    eyy = v,y + w/r + w,y^2/2
    gxy = u,y + v,x + w,x w,y
    kxx = -w,xx,   kyy = -w,yy,   kxy = -2 w,xy

With `1/r = 0` these are the strains of `plate_clpt_donnell`.

## Constitutive law

Classical laminated plate theory, `(N, M) = [ABD] (eps, kappa)`, with the
`6 x 6` matrix `[ABD]` of the laminate about the reference surface
(laminate between `-h/2 + d` and `h/2 + d`, offset `d`).

## Matrices

Integrands per unit area of the natural domain, times `intx*inty/4`:

    kC   = BA^T [ABD] BB          constitutive stiffness, with KCNL
    kC0  = B0A^T [ABD] B0B        linear constitutive stiffness
    kG   = GA^T [N] GB            geometric stiffness
    fint = BA^T (N, M)            internal force vector
    kM   = gA5^T [I] gB5          mass, gA5 = (u, v, w, -w,x, -w,y)
    kAx  = -(beta w,x^T w + gamma w^T w)
    kAy  = -beta w,y^T w
    cA   = -aeromu w^T w

`[I]` holds the moments `I_n = int rho z^n dz`, `n = 0, 1, 2`, of
`composites.Laminate`. The aerodynamic matrices follow the piston theory,
with the load `q = beta w,x + gamma w` along `w` and the curvature term
`gamma`, see `Shell.calc_kA`.

Kernels of `cylshell_clpt_donnell.pyx`: `fk0`, `fkG0`, `fkM`, `fkAx`,
`fkAy`, `fcA`, integrated in closed form with the integrals of Bardell's
functions. Kernels of `cylshell_clpt_donnell_num.pyx`: `fkC_num`, `fkG_num`,
`fkM_num`, `fkAx_num`, `fkAy_num`, `calc_fint`, with the Gauss-Legendre
rule, `max(2m - 1, 7)` points along `x` by default.

## Selection in panels

`Shell(..., r=r, model='cylshell_clpt_donnell')`; the default model of a
`Shell` with a radius is `cylshell_clpt_sanders`.
`panels.modelDB.db['cylshell_clpt_donnell']`: `dofs = 3`, `e_num = 6`,
`requires_r = True` (the kernels divide by `r`).

## Limitations

- The strains do not vanish for every small rigid-body motion of the
  cylinder (no `v/r` terms in the rotations and the changes of curvature),
  which matters for long shells and deformations of long wavelength, e.g.
  low circumferential wave numbers; see `cylshell_clpt_sanders`
  (`../cylshell_clpt_sanders`).
- No transverse shear deformation; see `cylshell_fsdt_donnell` and
  `cylshell_tsdt_donnell` (`../fsdt_tsdt`).

## References

- Amabili, M., "Nonlinear Vibrations and Stability of Shells and Plates",
  Cambridge University Press, 2008.
  https://doi.org/10.1017/CBO9780511619694
- Castro, S. G. P., and Donadon, M. V., "Assembly of semi-analytical models
  to address linear buckling and vibration of stiffened composite panels
  with debonding defect", Composite Structures, 160, 232-247, 2017.
  https://doi.org/10.1016/j.compstruct.2016.10.026
