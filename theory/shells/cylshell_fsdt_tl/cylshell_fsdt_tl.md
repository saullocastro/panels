# Cylindrical shell, total-Lagrangian FSDT with a second-order director (`cylshell_fsdt_tl`)

Derivation and symbolic checks: `cylshell_fsdt_tl.py` (sympy, run `python
cylshell_fsdt_tl.py`, about two minutes). Kernels: `write_pyx.py`, which
writes `panels/models/cylshell_fsdt_tl_num.pyx`, self-contained, as
`../plate_fsdt_tl/write_pyx.py` with the radius. No analytical module:
`modelDB` takes `fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA` from
`panels.models.cylshell_fsdt_gl` (`check_tangent`).

## Why

`cylshell_fsdt_gl` has the complete Green-Lagrange membrane strains but the
linear changes of curvature of Sanders: a rotation `t` of a part of the
shell about an axis normal to the cylinder axis gives bending strains of
order `(1 - cos t)/r`. This model measures the bending and transverse shear
strains with the second-order director of `plate_fsdt_tl`, as
`cylshell_clpt_tl` does with the deformed normal.

## Kinematics

Local basis `(e_x, e_y, e_z)`, `e_y,y = -e_z/r`, `e_z,y = e_y/r`, `w`
outwards, `r + z ~ r`. Displacement field of Schmidt and Reddy (1988), Eq.
(16), Palmerio et al. (1990, Part 1), Eq. (16), with `E33(0) = 0`, Eq. (20).

Director components, from Palmerio et al.: the
linear transverse shear strain of Eq. (20) is `2 E0_a3 = Phi0_a3 + u1_a`,
`Phi0_a3 = u3,a + b^l_a u_l` (Eq. (15c)); with `b^y_y = -1/r`, `Phi0_y3 =
w,y - v/r`, and the shear strain `gyz = phiy + w,y` of
`cylshell_fsdt_sanders` gives `u1_y = phiy + v/r`:

    psix = phix,   psiy = phiy + v/r,   psiz = 1 - (psix^2 + psiy^2)/2
    X,x = (1 + u,x, v,x, w,x)          X,y = (u,y, 1 + v,y + w/r, w,y - v/r)
    d,x = (psix,x, psiy,x, psiz,x)      d,y = (psix,y, psiy,y + psiz/r, psiz,y - psiy/r)

With `psiy = phiy` a rigid rotation about the cylinder axis (`phiy = 0`,
`v/r` the rotation) would not be free of strain.

Strains:

    exx, eyy, gxy: Green-Lagrange (those of cylshell_fsdt_gl)
    kxx = X,x . d,x
    kyy = X,y . d,y - 1/r - eyy/r
    kxy = X,x . d,y + X,y . d,x - gxy/(2r)
    gyz = X,y . d      gxz = X,x . d

## Linear part: Sanders' FSDT (`check_linear_part`)

Linear parts of the dot products (`psiz = 1 + O(psi^2)`, `eyyL = v,y + w/r`,
`gxyL = u,y + v,x`), each asserted in `check_linear_part`:

    X,x . d,x       = phix,x
    X,y . d,y - 1/r = psiy,y + (v,y + w/r)/r = phiy,y + v,y/r + eyyL/r
                      (the 1/r and the hoop term from (1 + v,y + w/r) psiz/r)
    X,x . d,y       = phix,y + v,x/r            (from v,x psiz/r)
    X,y . d,x       = psiy,x = phiy,x + v,x/r
    X,y . d         = psiy + w,y - v/r = phiy + w,y
    X,x . d         = phix + w,x

so without the Koiter terms the linear curvatures are Sanders' plus
`(0, eyyL/r, gxyL/(2r))`; with `-eyy/r` and `-gxy/(2r)`:

    kxx = phix,x
    kyy = phiy,y + v,y/r
    kxy = phix,y + phiy,x + 2 v,x/r - (u,y + v,x)/(2r)
        = phix,y + phiy,x + (3/2) v,x/r - (1/2) u,y/r
    gyz = phiy + w,y,   gxz = phix + w,x

the linear strains of `cylshell_fsdt_sanders`; `check_linear_part` also
asserts `[B(0)] [Phi] = B0` of that model (`../cylshell_fsdt_sanders/cylshell_fsdt_sanders.py`).
The Koiter terms use the complete Green-Lagrange `eyy`, `gxy`, so they do
not spoil the invariance of the strains.

`check_strain_measures`: without `-eyy/r`, `-gxy/(2r)` these are the
coefficients of `z^0`, `z` of the Green-Lagrange strains of `X + z d` with
the reference vectors `e_x`, `(1 + z/r) e_y`. Their linear part is Sanders'
plus `(0, eyy/r, gxy/(2r))`; the two Koiter terms of `cylshell_clpt_tl`
(van der Heijden 2008, p. 202) make it Sanders' (`check_linear_part`,
against `../cylshell_fsdt_sanders/cylshell_fsdt_sanders.py`).

## Checks

- `check_palmerio`: Eq. (20) of Palmerio et al. for the cylinder (`b_yy =
  -1/r`, Cartesian surface coordinates, `E33(0) = 0`) equals the strains
  without the Koiter terms to `O(eps^3)` (orders of `_scaling`, `v`, `w`
  of `eps^2`); the `z` term of `2 E_a3` is `O(eps^4)`.
- `check_small_rotations`: against `cylshell_fsdt_gl`, membrane equal,
  shear `O(eps^3)`, `kxx` `O(eps^4)`, `kyy` and `kxy` `O(eps^2)`: Palmerio's
  `Phi0_y3 Phi1_y3`, `Phi1_y3 = u13,y - psiy/r`, is of the order of the
  linear strains and missing from the linear curvatures of the GL model.
- `check_rigid_rotation`: rigid rotations of the undeformed cylinder about
  random axes, `psi` the components of the rotated normal: membrane zero,
  bending and shear `O(t^4)` (fitted order between `t = 0.1` and `0.05`),
  `cylshell_fsdt_gl` `O(t^2)`.
- `check_tangent`, `check_numerical`, `check_integrand_degree` (`6p` along
  `x` and `y`) as for `plate_fsdt_tl`.

## Limitations

As `plate_fsdt_tl`; the connections of `MultiDomain` use the Sanders
rotation `phiy + v/r` (`kCsdt._rinv_sanders`).

## References

- Schmidt, R., Reddy, J. N., J. Appl. Mech. 55(3), 611-617, 1988.
  https://doi.org/10.1115/1.3125837
- Palmerio, A. F., Reddy, J. N., Schmidt, R., Int. J. Non-Linear Mech.
  25(6), 687-700, 1990. https://doi.org/10.1016/0020-7462(90)90007-V
- van der Heijden, A. M. A. (ed.), "W. T. Koiter's Elastic Stability of
  Solids and Structures", Cambridge University Press, 2008.
  https://doi.org/10.1017/CBO9780511546174
