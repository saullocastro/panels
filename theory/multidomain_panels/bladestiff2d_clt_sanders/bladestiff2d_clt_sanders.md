# Blade stiffener with a 2D flange on a Sanders cylindrical skin (`bladestiff2d_clt_sanders`)

Derivation and kernels: `bladestiff2d_clt_sanders.py` (sympy, run `python
bladestiff2d_clt_sanders.py`), which checks the derivation symbolically
(`check_rigid_rotation`, `check_base_is_cylshell_clpt_sanders`) and writes
`panels/stiffener/models/bladestiff2d_clt_sanders.pyx`, the kernels of the
connection `kCss: fkCss`, `kCsf: fkCsf`, `kCff: fkCff`. The matrices of
the base and of the flange are those of the shell models, computed by the
`Shell` objects `BladeStiff2D.base` and `BladeStiff2D.flange`.

## Model

Blade stiffener of a curved `StiffPanelBay` (radius `r`) along `x` at `y =
ys`, `y` being the arc length, with the flange inside the cylinder:

                 || --> flange (plate, width bf, thickness hf)
               ======  --> base (padup, width bb, thickness hb)
      =========================  --> skin (thickness h, radius r)

- Base: a cylindrical shell `cylshell_clpt_sanders`, the kinematics of the
  default cylindrical skin, with the approximation functions, Ritz
  constants and boundary conditions of the bay, integrated over `y1 = ys -
  bb/2 <= y <= y2 = ys + bb/2` (matrices `*y1y2`) with the offset `-(h/2 +
  hb/2)`: perfectly bonded below the skin, it follows the skin's
  displacement field.
- Flange: it remains a plate, a `Shell` with the model
  `'plate_clpt_donnell'`, length `a`, width `bf`, its own approximation
  functions (`r` along `x`, `s` along the width) and Ritz constants, `mf x
  nf` terms (default `14 x 11`). It is simply supported at `x = 0` and `x =
  a`, where only `w,x` is free, and free along its two edges.

## Kinematics, strains and constitutive law

Classical laminated plate theory with the `6 x 6` matrix `[ABD]`. The base
has the linear strains of Sanders (1963), with the changes of curvature as
written by Amabili (2008), Sec. 1.3.8, Eqs. (1.136)-(1.138):

    exx = u,x,   eyy = v,y + w/r,   gxy = u,y + v,x
    kxx = -w,xx, kyy = -w,yy + v,y/r,
    kxy = -2 w,xy + (3/2) v,x/r - (1/2) u,y/r

and the rotations of the normal

    phix = -w,x,   phiy = -w,y + v/r

which enter the mass matrix through `u(z) = u + z phix`, `v(z) = v + z
phiy`. The flange has the strains of the plate (`1/r = 0`). The script
writes the linear stiffness and the mass of the base (`k0y1y2`, `kMy1y2`)
and of the flange (`k0f`, `kMf`), and checks that those of the base equal
`ab/4` times the integrands `kC0` and `kM` of
`../../shells/cylshell_clpt_sanders/cylshell_clpt_sanders.py`; `panels`
computes them with the kernels `fk0`, `fkM` of the model of each `Shell`.
The base adds no geometric stiffness (`BladeStiff2D.calc_kG`).

## Connection of the flange

The edge `eta_f = -1` of the flange is connected to the base along `y =
ys` (or to the skin, when there is no base) with the flange normal to the
skin:

    u = uf,   v = wf,   w = -vf,   -phiy = w,y - v/r = wf,y

by the penalty energy

    U = 1/2 int_0^a [kt (u - uf)^2 + kt (v - wf)^2 + kt (w + vf)^2
                     + kr (w,y - v/r - wf,y)^2] dx

which gives the blocks `kCss` (base-base), `kCsf` (base-flange; only the
upper triangle is assembled) and `kCff` (flange-flange, the same as
`../bladestiff2d_clt_donnell`). Compared with `../bladestiff2d_clt_donnell`,
whose rotation is `w,y`, the rotation `-v/r` adds to `kCss` the terms
`kr/r^2` in `v-v` and `-kr/r` in `v-w,y`, and to `kCsf` the term `kr/r` in
`v-wf,y`. In the kernels the functions `g` of the base and `s` of the
flange are evaluated at `eta(ys)` and `eta_f = -1`, and only `f` and `r`
are integrated along `x`, in closed form with the integrals of Bardell's
functions. The penalty constants `kt`, `kr` are those of
`panels.multidomain.connections.calc_kt_kr` of type `'ycte'`.

Why Sanders' rotation: under a rigid rotation `alpha` of the cylinder about
its axis, `v = r alpha`, `w = 0`, and the flange, pointing inwards, turns
with it, `wf = (r - yf) alpha`, `wf,y = -alpha`. Then `w,y - v/r = -alpha =
wf,y` and the connection stores no energy; with Donnell's `w,y = 0` it
stores `kr alpha^2 a/2` (`check_rigid_rotation`). The same `v/r` enters
the connections of `MultiDomain` between cylindrical domains with the
Sanders kinematics.

The base as a structural element follows Castro et al. (2016), Sec. 2,
where the flange is a beam; the plate flange and its connection are a
model of this package.

## Selection in panels

`StiffPanelBay.add_bladestiff2d(ys, ..., model=None)` creates a
`BladeStiff2D`; with the default `model=None` a curved skin whose model is
not `cylshell_clpt_donnell` (i.e. `cylshell_clpt_sanders`, the default
with a radius, or `cylshell_clpt_gl`, whose linear part is that of
Sanders) gets `'bladestiff2d_clt_sanders'`, a flat bay or a skin
`cylshell_clpt_donnell` gets `'bladestiff2d_clt_donnell'`; `model` may be
given explicitly. `panels.stiffener.modelDB.db['bladestiff2d_clt_sanders']`:
keys `'connections'` and `'base models'` (`'curved':
'cylshell_clpt_sanders'`, no flat base: a flat bay with this model raises
`ValueError`), linear static and linear buckling analyses (no non-linear
static analysis). The skin must use a model with 3 DOFs.

## Limitations

- The edge of the flange follows the mid-surface of the skin; the offset
  of the base is not considered in the position of the flange.
- The flange is a von Kármán plate: its geometric stiffness lacks
  `Nxx v,x^2` of a load in its own plane (see `plate_clpt_gl`).
- The flange always points inwards (`w = -vf`).

## References

- Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
  Applied Mathematics, 21(1), 21-36, 1963. https://doi.org/10.1090/qam/147023
- Amabili, M., "Nonlinear Vibrations and Stability of Shells and Plates",
  Cambridge University Press, 2008.
  https://doi.org/10.1017/CBO9780511619694
- Castro, S. G. P., Guimarães, T. A. M., Rade, D. A., and Donadon, M. V.,
  "Flutter of stiffened composite panels considering the stiffener's base
  as a structural element", Composite Structures, 140, 36-43, 2016.
  https://doi.org/10.1016/j.compstruct.2015.12.056
