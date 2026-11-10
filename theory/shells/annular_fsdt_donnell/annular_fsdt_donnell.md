# Annular-sector plate, FSDT with von Kármán strains in polar coordinates (`annular_fsdt_donnell`)

Derivation and symbolic checks: `annular_fsdt_donnell.py` (sympy, run
`python annular_fsdt_donnell.py`, a few seconds). Kernels: `write_pyx.py`,
the same file in every folder `annular_fsdt_*` (the model is the name of the
folder), which writes `panels/models/annular_fsdt_donnell_num.pyx` and
`panels/models/annular_fsdt_donnell.pyx` (`python write_pyx.py [folder]`).

## Why

The annular sector of `annular_clpt_donnell` with the first-order shear
deformation theory (FSDT): thick or sandwich frame webs, and annular plates
whose buckling load depends on the transverse shear. Bagheri, Kiani and
Eslami (2019, Table 2) give buckling torques of annular Mindlin plates up to
`h/a = 0.1`, where the shear lowers the classical torque by up to 68 %.

## Kinematics used in the literature

- Mindlin plate in polar coordinates, von Kármán strains: Bagheri, Kiani and
  Eslami (2019), Eqs. (1), (2) and (6), the strains of this model
  (`check_bagheri`).
- The membrane strains are those of `annular_clpt_donnell` (Amabili 2008,
  Eqs. (1.175)-(1.177)); with `phix = -w,x`, `phiy = -s w,y` the transverse
  shear strains vanish and the curvatures are those of `annular_clpt_donnell`
  (`check_clpt_limit`).

## Coordinates and degrees of freedom

Those of `annular_clpt_donnell` (`x = R - rho` inwards from the outer edge, `y
= R theta`, `e_x = -e_rho`, `e_y = e_theta`, `e_z = e_x x e_y`), with the
rotations `phix`, `phiy` of the normal as independent fields, `u(z) = u + z
phix`, `v(z) = v + z phiy`, `w(z) = w`, as in `plate_fsdt_donnell`. In
Bagheri's notation `u0 = -u`, `v0 = v`, `w0 = -w`, `psi_r = phix`,
`psi_theta = -phiy`, his `z` along `-e_z`. With `s = R/rho`:

    exx = u,x + w,x^2/2
    eyy = s v,y - u/rho + s^2 w,y^2/2
    gxy = s u,y + v,x + v/rho + s w,x w,y
    kxx = phix,x
    kyy = s phiy,y - phix/rho
    kxy = s phix,y + phiy,x + phiy/rho
    gyz = phiy + s w,y
    gxz = phix + w,x

and `dA = (rho/R) dx dy`. The laminate offset enters through `Shell.ABD` (8 x
8, the transverse shear stiffness with the shear correction of `Shell`) and
the moments `I0, I1, I2` of the density, with `J d = (u, v, w, phix, phiy)` in
the mass matrix.

## Checks of `annular_fsdt_donnell.py`

`check_linear_strains` (3-D FSDT field in the metric of the plane, the
transverse shear strains included), `check_bagheri`, `check_rigid_body` (six
small rigid motions with the rotations of the rotated normal),
`check_clpt_limit`, `check_plate_limit` (`R -> infinity`:
`plate_fsdt_donnell`), `check_tangent`.

## Kernels

As for `annular_clpt_donnell` (all matrices integrated numerically, the
integrands being rational in `x`), with the 15 generalized derivatives `(u,
u,x, u,y, ..., phiy, phiy,x, phiy,y)`, the 8 x 8 constitutive matrix, 5 DOFs
per term and the fields `fuvw`, `fstrain`, `strain_names`, `stress_names`
and `fg` (of `fsdt_tsdt_field`). The piston-theory kernels raise
`NotImplementedError`; the follower pressure is not generated. The split
`fkC_num` = `[B]^T [F] [B] + [H(sigma, d)] - [H(sigma_L, 0)]`, `fkG_num` =
`[H(sigma_L + Nbar, 0)]`, as in every model.

## References

- Bagheri, H., Kiani, Y., and Eslami, M. R., Buckling of moderately thick
  annular plates subjected to torque, Archive of Mechanical Engineering
  66(2), 2019, 209-227. https://doi.org/10.24425/ame.2019.128445
- Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
  Cambridge University Press, 2008. https://doi.org/10.1017/CBO9780511619694
