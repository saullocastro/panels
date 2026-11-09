# Flat plate, total-Lagrangian FSDT with a second-order director (`plate_fsdt_tl`)

Derivation and symbolic checks: `plate_fsdt_tl.py` (sympy, run `python
plate_fsdt_tl.py`, about ten seconds). Kernels: `write_pyx.py`, which writes
`panels/models/plate_fsdt_tl_num.pyx` (`fkC_num`, `fkG_num`, `calc_fint`,
`calc_kTv`, `calc_fext_follower`, `fkCfollower_num`), self-contained:
`python write_pyx.py [outdir]`. Only the follower-pressure kernels come from
`../follower_pressure/write_pyx.py`, as for every model. No analytical
module: `modelDB` takes `fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA` from
`panels.models.plate_fsdt_gl` (`check_tangent`).

## Why

`plate_fsdt_gl` has the complete Green-Lagrange membrane strains but the
linear bending strains `phi,a` of the fixed local frame: a bending
deformation superposed on a rotation `t` about an in-plane axis is seen with
the factor `cos t`, and a cantilever bent out of its plane follows the
linear beam. This model measures the bending and transverse shear strains
with a director that rotates with the deformation, as `plate_clpt_tl` does
with the deformed normal.

## Kinematics (Schmidt and Reddy 1988; Palmerio, Reddy and Schmidt 1990)

Displacement field linear through the thickness, Schmidt and Reddy (1988),
Eq. (16), Palmerio et al. (1990, Part 1), Eq. (16): `u(z) = u0 + z u1`, with
`u1 = (phix, phiy, u13)`. Palmerio's transverse normal strain, Eq. (20),
`E33(0) = u13 + (phix^2 + phiy^2)/2`, set to zero (inextensible normal; their
theory keeps `u13` free): `u13 = -(phix^2 + phiy^2)/2`, and the point at `z`
goes to `X + z d`,

    X = (x + u, y + v, w),    d = (phix, phiy, 1 - (phix^2 + phiy^2)/2)

Strains: coefficients of `z^0`, `z^1` of `E_ab(z) = (G_a . G_b - delta_ab)/2`,
`G_a = X,a + z d,a`, and `2 E_a3(0) = X,a . d` (`check_strain_measures`):

    exx, eyy, gxy: Green-Lagrange (those of plate_fsdt_gl)
    kxx = X,x . d,x    kyy = X,y . d,y    kxy = X,x . d,y + X,y . d,x
    gyz = X,y . d      gxz = X,x . d

Dropped: the `z^2` terms of `E_ab` and the `z` term of `2 E_a3`,
`d,a . d = |phi|^2 (phi . phi,a)/2`, fourth order; `|d|^2 = 1 + |phi|^4/4`.

## Checks

- `check_palmerio`: with Palmerio's orders, Eqs. (9), (18) (strains
  `eps^2`; `w,a` `eps`; in-plane gradients, rotation gradients `eps^2`;
  `phi_a = -w,a + gamma_a`, `gamma_a` `eps^2`), the strains equal Eq. (20)
  (plate, `b_ab = 0`, `E33(0) = 0`) to `O(eps^3)`, and Palmerio's `z` term of
  `2 E_a3` is `O(eps^4)`.
- `check_linear_part`: `[B(0)] [Phi]` is `B0` of `plate_fsdt_donnell`
  (`../fsdt_tsdt/fsdt_tsdt.py`); linear and membrane strains are those of
  `plate_fsdt_gl`.
- `check_small_rotations`: with the same orders, `plate_fsdt_tl -
  plate_fsdt_gl` is `O(eps^3)` in the shear strains, `O(eps^4)` in the
  curvatures, zero in the membrane strains. For the plate the cubic
  curvature terms are beyond Palmerio's order: they are kept because the
  strains are then exact for `X + z d`.
- `check_rigid_rotation`: rigid rotation of the undeformed plate with `phi`
  the components of `R e_z`: membrane and bending zero, shear `O(t^5)`;
  rotation of a bent state about an in-plane axis: all strains change by
  `O(t^4)`, the linear curvatures of `plate_fsdt_gl` by `O(t^2)` (at least
  20 times more at `t = 0.1`).
- `check_tangent`: exact Jacobian `B^T F B + H(sigma, d)`, `H` linear in
  `d`; split `fkG_num = H0(sigma_L + Nbar)`, `H0 = H(., d = 0)`, homogeneous
  of degree one; `fkC_num = B^T F B + H(sigma, d) - H0(sigma_L)`; with a
  constant membrane stress `H0` is the `H` of `plate_fsdt_gl`.
- `check_numerical`: central differences of `fint` against `K_T`.
- `check_integrand_degree`: `6p` along both directions (`phix`, `phiy`
  without derivative), `3p + 1` Gauss-Legendre points (`modelDB`
  `'integrand_degree': (6, 0)`).

## Pattern for the other total-Lagrangian shear models

The folder differs from `../plate_fsdt_gl/` in `<model>.py` (strains,
`H` depending on `d`, `H0`, the checks) and in `write_pyx.py` only in the
model settings, the notes and `fkC_num`/`fkG_num` (`H(sigma, d) -
H0(sigma_L)` and `H0`) and `generate` (numerical module only). A TSDT model
adds the third-order strains of `../plate_tsdt_gl/` (13 strains, `w,xx`,
`w,xy`, `w,yy` in `d`); an annular model takes the tangent vectors and the
basis derivatives of its geometry.

## Limitations

- Moderate rotations: second-order director, `|d|^2 = 1 + |phi|^4/4`.
- Small strains, inextensible normal, linear connections of `MultiDomain`.

## References

- Schmidt, R., Reddy, J. N., "A Refined Small Strain and Moderate Rotation
  Theory of Elastic Anisotropic Shells", J. Appl. Mech. 55(3), 611-617,
  1988. https://doi.org/10.1115/1.3125837
- Palmerio, A. F., Reddy, J. N., Schmidt, R., "On a moderate rotation theory
  of laminated anisotropic shells - Part 1. Theory", Int. J. Non-Linear
  Mech. 25(6), 687-700, 1990. https://doi.org/10.1016/0020-7462(90)90007-V
