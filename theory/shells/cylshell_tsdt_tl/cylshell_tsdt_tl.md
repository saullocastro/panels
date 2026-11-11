# Cylindrical shell, total-Lagrangian TSDT (`cylshell_tsdt_tl`)

Derivation and symbolic checks: `cylshell_tsdt_tl.py` (sympy, run `python
cylshell_tsdt_tl.py`, about twenty minutes, mostly `check_tangent`). Kernels: `write_pyx.py`, which writes
`panels/models/cylshell_tsdt_tl_num.pyx` (`fkC_num`, `fkG_num`, `calc_fint`,
`calc_kTv`, `calc_fext_follower`, `fkCfollower_num`), self-contained, as
`../plate_tsdt_tl/write_pyx.py` with the radius. No analytical module:
`modelDB` takes `fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA` from
`panels.models.cylshell_tsdt_gl` (`check_tangent`).

## Why

`cylshell_tsdt_gl` has the complete Green-Lagrange membrane strains but the
linear curvatures and third-order strains of Reddy's TSDT with Sanders'
kinematics. This model measures them with the second-order director of
`cylshell_fsdt_tl` and the normal of the deformed mid-surface, as
`plate_tsdt_tl` does for the plate (see `../plate_tsdt_tl/plate_tsdt_tl.md`
for the choice of the third-order terms and the comparison with linear
third-order terms).

## Kinematics

Local basis `(e_x, e_y, e_z)`, `e_y,y = -e_z/r`, `e_z,y = e_y/r`, `w`
outwards, `r + z ~ r`; the derivative along `y` of a vector `(a_x, a_y, a_z)`
is `(a_x,y, a_y,y + a_z/r, a_z,y - a_y/r)`.

    X,x  = (1 + u,x, v,x, w,x)          X,y = (u,y, 1 + v,y + w/r, w,y - v/r)
    X,xx = (u,xx, v,xx, w,xx)            X,xy = (u,xy, v,xy + w,x/r, w,xy - v,x/r)
    X,yy = (u,yy, v,yy + 2 w,y/r - v/r^2, w,yy - 2 v,y/r - 1/r - w/r^2)
    d    = (psix, psiy, 1 - (psix^2 + psiy^2)/2),  psix = phix,  psiy = phiy + v/r
    N    = X,x x X,y,   N,x = X,xx x X,y + X,x x X,xy,   N,y = X,xy x X,y + X,x x X,yy
    r(z) = X + z d + z^3 q,   q = -c1 (d - N),   c1 = 4/(3 h^2)

Linear part of `d - N`: `(phix + w,x, phiy + w,y, -(u,x + v,y + w/r))`.

Strains:

    exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz:   those of cylshell_fsdt_tl
    kxx3 = X,x . q,x
    kyy3 = X,y . q,y - c1 (exx + eyy)/r
    kxy3 = X,x . q,y + X,y . q,x
    gyz2 = 3 X,y . q = -3 c1 gyz,   gxz2 = 3 X,x . q = -3 c1 gxz

Without `-c1 (exx + eyy)/r` (and the Koiter terms `-eyy/r`, `-gxy/(2r)` of
`kyy`, `kxy`), these are the coefficients of `z^0`, `z`, `z^3` of the
Green-Lagrange strains of `r(z)` with the reference vectors `e_x`, `(1 +
z/r) e_y`, and of `z^0`, `z^2` of `2 E_a3` (`check_strain_measures`, on a
Cartesian embedding with polynomial fields).

## The term `-c1 (exx + eyy)/r`

The `z` component of `N` is `1 + exx + eyy` to first order (stretch of the
area). Through `e_z,y = e_y/r` it enters the `y` component of `q,y`, and the
linear part of `X,y . q,y` is

    -c1 (phiy,y + w,yy) + c1 (u,x + v,y + w/r)/r

(`check_linear_part`). The term with the Green-Lagrange membrane strains,
invariant under rigid rotations, removes `c1 (u,x + v,y + w/r)/r`, as the
Koiter terms of `cylshell_fsdt_tl` remove `eyy/r`, `gxy/(2r)` from the
linear curvatures. Then the linear part is that of `cylshell_tsdt_sanders`:

    kxx3 = -c1 (phix,x + w,xx)
    kyy3 = -c1 (phiy,y + w,yy)
    kxy3 = -c1 (phix,y + phiy,x + 2 w,xy)
    gyz2 = -3 c1 (phiy + w,y),   gxz2 = -3 c1 (phix + w,x)

In the plate the same stretch only enters the dropped `z^3` coefficient of
`2 E_a3`.

## Checks (`python cylshell_tsdt_tl.py`)

- `check_strain_measures`: as above (Cartesian embedding `X0 = (x, r sin th,
  r cos th)`, random polynomial fields, exact arithmetic at a point).
- `check_linear_part`: `[B(0)] [Phi]` = `B0` of `cylshell_tsdt_sanders`;
  linear and membrane strains = `cylshell_tsdt_gl`; first-order strains =
  `cylshell_fsdt_tl`; the linear part of the raw `X,y . q,y`.
- `check_face_shear`: `gamma + (h^2/4) gamma2 = 0` for any deformation.
- `check_rigid_rotation`: rigid rotations of the undeformed cylinder and of
  a state bent along the axis (`w = 0.4 x^2/2`, at `x = 0`), random axes,
  director components those of the rotated normal: membrane strains
  unchanged, the others change by `O(t^4)` (fitted orders 3.90-4.04
  between `t = 0.1` and `0.05`); the linear third-order strains of
  `cylshell_tsdt_gl` change at the second order (undeformed, 2.00) or the
  first (bent, 0.87-1.07).
- `check_tangent`, `check_numerical`, `check_integrand_degree` (`6p` along
  `x` and `y`) as for `plate_tsdt_tl`.

## Limitations

As `cylshell_fsdt_tl`; the connections of `MultiDomain` use the Sanders
rotation `phiy + v/r` (`kCsdt._rinv_sanders`).

## References

- Reddy, J. N., Liu, C. F., Int. J. Eng. Sci. 23(3), 319-330, 1985.
  https://doi.org/10.1016/0020-7225(85)90051-5
- Schmidt, R., Reddy, J. N., J. Appl. Mech. 55(3), 611-617, 1988.
  https://doi.org/10.1115/1.3125837
- Palmerio, A. F., Reddy, J. N., Schmidt, R., Int. J. Non-Linear Mech.
  25(6), 687-700, 1990. https://doi.org/10.1016/0020-7462(90)90007-V
- van der Heijden, A. M. A. (ed.), "W. T. Koiter's Elastic Stability of
  Solids and Structures", Cambridge University Press, 2008.
  https://doi.org/10.1017/CBO9780511546174
