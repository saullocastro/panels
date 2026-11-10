# Annular-sector plate, `annular_fsdt_gl`

Derivation and symbolic checks: `annular_fsdt_gl.py` (sympy, run
`python annular_fsdt_gl.py`, about 10 s). Kernels: `write_pyx.py`, the same
file in every folder `annular_fsdt_*` (the model is the name of the folder),
which writes `panels/models/annular_fsdt_gl_num.pyx` and
`panels/models/annular_fsdt_gl.pyx` (`python write_pyx.py [folder]`).

Coordinates, local basis, degrees of freedom, inputs and the laminate offset
as in `../annular_fsdt_donnell/annular_fsdt_donnell.md`. The model has the
complete Green-Lagrange membrane strains of `annular_clpt_gl` (Amabili 2008,
Eqs. (1.166)-(1.168), from `g1 = X,x`, `g2 = s X,y`) and the linear changes of
curvature and transverse shear strains of `annular_fsdt_donnell`, as
`plate_fsdt_gl` for the flat plate. All eight strains vanish for any finite
rigid motion with `phix = -w,x`, `phiy = -s w,y`; with the rotations of the
rotated normal, the transverse shear strains are of the second order in the
angle of a rotation about an axis inclined to the plane.

Checks of `annular_fsdt_gl.py`: `check_linear_part` (`annular_fsdt_donnell`),
`check_amabili`, `check_rigid_body` (finite rotations of 1.1 and 2.9 rad),
`check_plate_limit` (`plate_fsdt_gl`), `check_tangent`. The kernels use
`[H(sigma, d)] = sum_r sigma_r d^2 eps_r/d d^2`, constant in `d`, split as in
every model: `fkG_num` = `[H(sigma_L + Nbar, 0)]`, `fkC_num` = `[B]^T [F] [B]
+ [H(sigma, d)] - [H(sigma_L, 0)]`.
