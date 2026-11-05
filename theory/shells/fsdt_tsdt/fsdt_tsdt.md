# Plates and cylindrical shells, FSDT and Reddy's TSDT

Six models with five degrees of freedom per term, `u, v, w, phix, phiy`
(`fsdt_tsdt.MODELS`):

| model                   | theory | geometry | kinematics            |
|-------------------------|--------|----------|-----------------------|
| `plate_fsdt_donnell`    | FSDT   | plate    | von Kármán            |
| `plate_tsdt_donnell`    | TSDT   | plate    | von Kármán            |
| `cylshell_fsdt_donnell` | FSDT   | cylinder | Donnell               |
| `cylshell_fsdt_sanders` | FSDT   | cylinder | Sanders-Koiter        |
| `cylshell_tsdt_donnell` | TSDT   | cylinder | Donnell               |
| `cylshell_tsdt_sanders` | TSDT   | cylinder | Sanders-Koiter        |

Derivation: `fsdt_tsdt.py` (sympy; `build(theory, geometry, kinematics)`
gives the integrands, and `python fsdt_tsdt.py` writes them to
`output_expressions_python/sympy_<model>_<name>.txt`). Kernels:
`write_pyx.py`, which writes `panels/models/<model>.pyx` (integrated
analytically) and `panels/models/<model>_num.pyx` (integrated numerically)
for every model, or for the models given as arguments, `python write_pyx.py
[model ...]`, and appends to the numerical modules the kernels of
`../follower_pressure/write_pyx.py` and `../tangent_product/write_pyx.py`.
The field variables of all six models are evaluated by
`panels/models/fsdt_tsdt_field.pyx`.

## Kinematics

Domain `0 <= x <= a`, `0 <= y <= b`, mapped onto `-1 <= xi, eta <= +1`; for
the cylinders `y = r theta` is the arc length, `w` positive outwards and
`r + z ~ r`. With `z` measured from the reference surface, the laminate
between `d - h/2` and `d + h/2` (offset `d`):

    u(z) = u + z phix - c1 [(z - d)^3 + d^3] (phix + w,x)
    v(z) = v + z (phiy + sk v/r) - c1 [(z - d)^3 + d^3] (phiy + w,y)
    w(z) = w

with `c1 = 0` for the FSDT and `c1 = 4/(3 h^2)` for the TSDT of Reddy
(1984), and `sk = 1` for the Sanders-Koiter kinematics, `0` otherwise. The
cubic is written about the mid-surface, `z = d`: the transverse shear
strains vanish at both faces for any offset, `u`, `v` remain the
displacements of the reference surface, and `phix`, `phiy` are the
rotations of the normal at the mid-surface (`tsdt_offset_cubic()`). With
`d = 0` this is Reddy's field. For the cylinders the TSDT follows the shell
field of Reddy and Liu (1985), Eqs. (5)-(9), with the shallow-shell
assumption.

## Strains

`eps = eps0 + z eps1 + [(z - d)^3 + d^3] eps3`, `gam = gam0 + (z - d)^2
gam2`, with the rotations `bx = -w,x`, `by = -w,y + sk v/r` of the von
Kármán terms:

    eps0 = (u,x + bx^2/2,  v,y + w/r + by^2/2,  u,y + v,x + bx by)
    eps1 = (phix,x,  phiy,y + sk v,y/r,
            phix,y + phiy,x + sk [(3/2) v,x/r - (1/2) u,y/r])
    eps3 = -c1 (phix,x + w,xx,  phiy,y + w,yy,  phix,y + phiy,x + 2 w,xy)
    gam0 = (gyz, gxz)^(0) = (phiy + w,y,  phix + w,x)
    gam2 = -3 c1 gam0

`1/r = 0` for the plates. The `sk` terms (Sanders 1963) make all the
strains vanish for any small rigid-body motion; the term `omega3^2/2` of
the rotation about the normal is omitted, as in `../cylshell_clpt_sanders`.
No Sanders-Koiter term enters `eps3` and the transverse shear strains.

## Constitutive law

- FSDT, `8 x 8`: `[[A, B, 0], [B, D, 0], [0, 0, Ats]]`, with `Ats` the
  shear corrected transverse shear stiffness of the `(yz, xz)` components,
  set by `Shell.fsdt_shear_correction`: `'rohwer'` (Rohwer 1988, the
  default), `'vlachoutsis'`, `'constant'`, `None`, or a factor `k`
  multiplying the uncorrected stiffness.
- TSDT, `13 x 13`: `T^T F_m T`, with `F_m = [[A, B, E, 0, 0], [B, D, F, 0,
  0], [E, F, H, 0, 0], [0, 0, 0, Abar_ts, Dts], [0, 0, 0, Dts, Fts]]` the
  matrix of the laminate about its mid-surface, without shear correction,
  and `T` the identity but for `T[0:3, 3:6] = d I`, `T[0:3, 6:9] = d^3 I`,
  see `Shell._get_lam_ABD`.

## Matrices

`shape_functions` gives, per term, `B` (8 x 5 or 13 x 5, with the parts
linear in `bx`, `by` at the integration point), `B0` (`bx = by = 0`) and `G
= d(bx, by)/dc`; `build` gives, per unit area of the natural domain:

    kC   = BA^T [F] BB        constitutive stiffness, with KCNL
    kC0  = B0A^T [F] B0B      linear constitutive stiffness
    fint = BA^T sigma         internal force vector
    kG   = GA^T [N] GB        geometric stiffness
    kM                        mass, kinetic energy of u(z), v(z), w(z)
    kAx, kAy, cA              piston theory, q = beta w,x + gamma w (gamma: cylinders)

with the generalized stresses `(Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qy, Qx)`
(FSDT) or `(N, M, P, Qy, Qx, Ry, Rx)` (TSDT). The mass uses the moments `I_n
= int rho z^n dz`, `n = 0, 1, 2` (FSDT), or the moments about the
mid-surface `J_n`, `n = 0, 1, 2, 3, 4, 6` (TSDT).

Kernels of `<model>.pyx`: `fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA`
(closed form, integrals of Bardell's functions). Kernels of
`<model>_num.pyx`: `fkC_num`, `fkG_num`, `fkM_num`, `fkAx_num`,
`fkAy_num`, `calc_fint`, `calc_fext_follower`, `fkCfollower_num`,
`calc_kTv` (Gauss-Legendre, `max(2m - 1, 7)` points along `x` by default).

## Selection in panels

`Shell(..., model='plate_fsdt_donnell')` and likewise for the other five
names; `panels.modelDB.db[model]`: `dofs = 5`, `e_num = 8` (FSDT) or `13`
(TSDT), `requires_r = True` for the cylinders. The models are supported by
`MultiDomain`, with the connections of
`panels.multidomain.connections.kCsdt` and the functions `*_sdt` of
`kCBFycte`, `kCBFxcte`, `kCBFxycte` (`../../multidomain_penalization`), but
not by `StiffPanelBay`, whose stiffeners assume the 3 DOFs of the CLPT. For
the Sanders models the field `phiy` returned by `fsdt_tsdt_field.fuvw` is
the rotation of the normal `phiy + v/r`.

## References

- Reddy, J. N., "A simple higher-order theory for laminated composite
  plates", Journal of Applied Mechanics, 51(4), 745-752, 1984.
  https://doi.org/10.1115/1.3167719
- Reddy, J. N., and Liu, C. F., "A higher-order shear deformation theory of
  laminated elastic shells", International Journal of Engineering Science,
  23(3), 319-330, 1985. https://doi.org/10.1016/0020-7225(85)90051-5
- Rohwer, K., "Improved transverse shear stiffnesses for layered finite
  elements", DFVLR-FB 88-32, Deutsche Forschungs- und Versuchsanstalt für
  Luft- und Raumfahrt (DFVLR), Institut für Strukturmechanik, Braunschweig,
  1988.
- Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
  Applied Mathematics, 21(1), 21-36, 1963. https://doi.org/10.1090/qam/147023
- Amabili, M., "Nonlinear Vibrations and Stability of Shells and Plates",
  Cambridge University Press, 2008.
  https://doi.org/10.1017/CBO9780511619694
