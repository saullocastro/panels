# Annular-sector plate, `annular_fsdt_tl`

Derivation and symbolic checks: `annular_fsdt_tl.py` (sympy, run
`python annular_fsdt_tl.py`, about 1 min). Kernels: `write_pyx.py`, the same
file in every folder `annular_fsdt_*` (the model is the name of the folder),
which writes `panels/models/annular_fsdt_tl_num.pyx` and
`panels/models/annular_fsdt_tl.pyx` (`python write_pyx.py [folder]`).

Coordinates, local basis, degrees of freedom, inputs and the laminate offset
as in `../annular_fsdt_donnell/annular_fsdt_donnell.md`. The kinematics are
those of `plate_fsdt_tl`: the second-order director of Schmidt and Reddy
(1988), Eq. (16), and Palmerio, Reddy and Schmidt (1990, Part 1), Eqs. (16)
and (20), with an inextensible normal, and the exact strains of `X + z d`,
written in the local basis of the annulus, which turns with `theta`:

    d     = (phix, phiy, 1 - (phix^2 + phiy^2)/2)
    d,x   = (phix,x, phiy,x, -(phix phix,x + phiy phiy,x))
    s d,y = (s phix,y + phiy/rho, s phiy,y - phix/rho,
             -s (phix phix,y + phiy phiy,y))

with `g1 = X,x`, `g2 = s X,y` of `annular_clpt_gl`:

    exx = (g1.g1 - 1)/2,  eyy = (g2.g2 - 1)/2,  gxy = g1.g2
    kxx = g1.d,x,  kyy = g2.(s d,y),  kxy = g1.(s d,y) + g2.d,x
    gyz = g2.d,  gxz = g1.d

the coefficients of `z^0`, `z^1` of the Green-Lagrange strains of `X + z d` in
the metric of the flat reference plane, which does not depend on `z`.
Dropped, as in `plate_fsdt_tl`: the `z^2` terms of `E_ab` and `d,a . d` in `2
E_a3`, both of fourth order in the rotations. The reference plane has no
curvature, so no Koiter term is added (unlike `cylshell_fsdt_tl`): the linear
part is exactly that of `annular_fsdt_donnell`.

Checks of `annular_fsdt_tl.py`: `check_vectors` (the local components
against the Cartesian derivatives), `check_strain_measures`,
`check_linear_part` (`annular_fsdt_donnell`; membrane = `annular_fsdt_gl`),
`check_rigid_rotation` (undeformed annulus rotated about random axes:
membrane and bending strains zero, shear `O(t^5)`; a bent state rotated about
in-plane axes: changes `O(t^4)` against `O(t^2)` for the curvatures of
`annular_fsdt_gl`), `check_plate_limit` (`R -> infinity`: `plate_fsdt_tl`),
`check_tangent`. No published total-Lagrangian shear-deformable model in
polar coordinates was found; the plate limit, the linear part and these
checks anchor the model.

The kernels use `[H(sigma, d)]`, linear in `d`, split as in every model:
`fkG_num` = `[H(sigma_L + Nbar, 0)]`, `fkC_num` = `[B]^T [F] [B] + [H(sigma,
d)] - [H(sigma_L, 0)]`; the mass matrix is that of the linear kinematics.
`integrand_degree` (6, 0) in `modelDB`, as `plate_fsdt_tl`.
