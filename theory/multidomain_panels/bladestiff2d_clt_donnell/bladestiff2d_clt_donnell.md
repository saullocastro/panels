# Blade stiffener with a 2D flange (`bladestiff2d_clt_donnell`)

Derivation: `bladestiff2d_clt_donnell.py` (sympy, run `python
bladestiff2d_clt_donnell.py`), which writes the integrands to
`output_expressions_python/sympy_<name>.txt`. Kernels of the connection:
`panels/stiffener/models/bladestiff2d_clt_donnell.pyx`, `kCss: fkCss`,
`kCsf: fkCsf`, `kCff: fkCff`; no script writes this module. The matrices of
the base and of the flange are those of the shell models, computed by the
`Shell` objects `BladeStiff2D.base` and `BladeStiff2D.flange`.

## Model

Blade stiffener of a `StiffPanelBay` along `x` at `y = ys`, below the skin:

                 || --> flange (plate, width bf, thickness hf)
               ======  --> base (padup, width bb, thickness hb)
      =========================  --> skin (thickness h)

- Base: a `Shell` with the approximation functions, Ritz constants and
  boundary conditions of the bay, integrated over `y1 = ys - bb/2 <= y <=
  y2 = ys + bb/2` (matrices `*y1y2`) with the offset `-(h/2 + hb/2)`:
  perfectly bonded below the skin, it follows the skin's displacement
  field. It is built without a `model`, i.e. with the default model of
  `Shell`: `plate_clpt_donnell` for a flat bay, `cylshell_clpt_sanders`
  for a bay with a radius.
- Flange: a `Shell` with the model `'plate_clpt_donnell'`, length `a`,
  width `bf`, its own approximation functions (`r` along `x`, `s` along the
  width) and Ritz constants, `mf x nf` terms (default `14 x 11`). It is
  simply supported at `x = 0` and `x = a`, where only `w,x` is free, and
  free along its two edges.

## Kinematics, strains and constitutive law

Those of the shell models of the base and of the flange, see
`../../shells`: classical laminated plate theory with the `6 x 6` matrix
`[ABD]`. The script restates the linear matrices with the Donnell strains

    exx = u,x,   eyy = v,y + w/r,   gxy = u,y + v,x
    kxx = -w,xx, kyy = -w,yy,       kxy = -2 w,xy

(`1/r = 0` for the flange), the geometric stiffness of the membrane forces
`(Nxx, Nyy, Nxy)` on `(w,x, w,y)`, the mass of a homogeneous laminate with
the offset `d` and the piston-theory matrices `kAx`, `kAy`, `cA` (`k0y1y2`,
`kG0y1y2`, `kMy1y2`, `k0f`, `kG0f`, `kMf`, ...); `panels` computes them
with the kernels `fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA` of the model
of each `Shell`.

## Connection of the flange

The edge `eta_f = -1` of the flange is connected to the base along `y =
ys` (or to the skin, when there is no base) with the flange normal to the
base:

    u = uf,   v = wf,   w = -vf,   w,y = wf,y

by the penalty energy

    U = 1/2 int_0^a [kt (u - uf)^2 + kt (v - wf)^2 + kt (w + vf)^2
                     + kr (w,y - wf,y)^2] dx

which gives the blocks `kCss` (base-base), `kCsf` (base-flange; only the
upper triangle is assembled) and `kCff` (flange-flange). In the kernels the
functions `g` of the base and `s` of the flange are evaluated at `eta(ys)`
and `eta_f = -1`, and only `f` and `r` are integrated along `x`. The
penalty constants `kt`, `kr` are those of
`panels.multidomain.connections.calc_kt_kr` of type `'ycte'`.

The flange as a plate represents its local buckling, unlike
`../bladestiff1d_clt_donnell`. The base as a structural element follows
Castro et al. (2016), Sec. 2, where the flange is a beam; the plate flange
and its connection are a model of this package.

## Selection in panels

`StiffPanelBay.add_bladestiff2d(ys, ...)` creates a `BladeStiff2D`;
`panels.stiffener.modelDB.db['bladestiff2d_clt_donnell']`: key
`'connections'`, linear static and linear buckling analyses (no non-linear
static analysis). The skin must use a model with 3 DOFs.

## References

- Castro, S. G. P., Guimarães, T. A. M., Rade, D. A., and Donadon, M. V.,
  "Flutter of stiffened composite panels considering the stiffener's base
  as a structural element", Composite Structures, 140, 36-43, 2016.
  https://doi.org/10.1016/j.compstruct.2015.12.056
