# Annular-sector plate, total-Lagrangian TSDT in polar coordinates (`annular_tsdt_tl`)

Derivation and symbolic checks: `annular_tsdt_tl.py` (sympy, run
`python annular_tsdt_tl.py`, about 30 s). Kernels: `write_pyx.py`, the writer
of `annular_tsdt_donnell` and `annular_tsdt_gl` with all the stresses in
`[H]` (`NH = NE`), which writes `panels/models/annular_tsdt_tl_num.pyx` and
`panels/models/annular_tsdt_tl.pyx` (`python write_pyx.py [folder]`).

Coordinates, local basis, degrees of freedom, inputs and the laminate offset
as in `../annular_tsdt_donnell/annular_tsdt_donnell.py`.

## Kinematics

Reddy's TSDT field written for finite rotations, as `plate_tsdt_tl`:

    r(z) = X + z d + p(z) q,   p(z) = (z - d)^3 + d^3,   q = -c1 (d - N)

with `c1 = 4/(3 h^2)`, the offset `d` of the laminate, `X` the deformed
reference plane, the second-order director of `annular_fsdt_tl` (Schmidt and
Reddy 1988, Eq. (16); Palmerio, Reddy and Schmidt 1990, Eqs. (16), (20)),

    d = (phix, phiy, 1 - (phix^2 + phiy^2)/2)

in the local basis, and `N = g1 x g2`, the normal of the deformed plane, not
normalized, `g1 = X,x`, `g2 = s X,y`. The linear part of `d - N` is Reddy's
vector `(phix + w,x, phiy + s w,y, .)`. In the local basis, which turns with
`theta`, `V,x = (V1,x, V2,x, V3,x)` and `s V,y = (s V1,y + V2/rho, s V2,y -
V1/rho, s V3,y)`; `N,x` and `s N,y` need `u,xx, u,yy, u,xy, v,xx, v,yy,
v,xy`: 24 generalized derivatives (those of `annular_tsdt_gl` and these six).

## Strains

The coefficients of `1`, `z`, `p(z)` of the Green-Lagrange strains `E_ab` of
`r(z)` (the metric of the flat reference plane does not depend on `z`) and of
`(z - d)^2` of `2 E_a3`:

    exx, eyy, gxy      = (g1.g1 - 1)/2, (g2.g2 - 1)/2, g1.g2
    kxx, kyy, kxy      = g1.d,x,  g2.(s d,y),  g1.(s d,y) + g2.d,x
    kxx3, kyy3, kxy3   = g1.q,x,  g2.(s q,y),  g1.(s q,y) + g2.q,x
    gyz, gxz           = g2.d,  g1.d
    gyz2, gxz2         = 3 g2.q, 3 g1.q  = -3 c1 (gyz, gxz)   (g_a . N = 0)

Membrane, `kappa` and `gamma` are those of `annular_fsdt_tl`; since `g_a . N,b
= -g_a,b . N`, `kappa3 = -c1 (kappa + second fundamental form)`. Dropped, as
in the linear TSDT and `plate_tsdt_tl`: the `z^2`, `z p`, `p^2` terms of
`E_ab`, the other powers of `2 E_a3`, and `E_33`.

## Why the third-order term is built from `d` and `N`

The alternative keeps the third-order strains of the linear theory (those of
`annular_tsdt_gl`) with the director part above. Its transverse shear strain
at the faces, `gamma + (h^2/4) gamma2`, is then `gamma_TL - gamma_lin`: the
shear of the director `g_a . d` minus the linear `phi_a + w,a`, which cannot
both vanish in large bending, so that the transverse shear energy stiffens
the bending. For the flat plate (both options prototyped, `plate_tsdt_tl`)
the thin cantilever strip, `L/h = 500`, gives at a tip rotation of 0.46 rad
the tip deflection and shortening -25.5 % and -46.7 % from the elastica (the
construction above: -0.22 % and -0.47 %), at 0.99 rad -49.9 % and -79.4 %
(-1.44 %, -3.84 %); under a rigid rotation about an oblique axis its internal
forces are `O(t^2)`, against `O(t^5)`. With `q = -c1 (d - N)` the face shear is
identically zero (Reddy's condition kept for any deformation), and the
invariance is that of the second-order director.

## Checks (`annular_tsdt_tl.py`, all pass)

- `check_vectors`: `g1, g2, d, d,x, s d,y, N, N,x, s N,y` in the local basis
  against the derivatives of the Cartesian vectors (random cubic fields);
- `check_strain_measures`: the strains are the coefficients of the
  Green-Lagrange strains of `r(z)` (random cubic fields);
- `check_linear_part`: linear strains of `annular_tsdt_donnell`, membrane
  strains of `annular_tsdt_gl`;
- `check_face_shear`: `gamma + (h^2/4) gamma2` identically zero;
- `check_rigid_rotation` (`c1 = 1`, `t = 0.2/0.1`): undeformed annulus about
  random axes, membrane, `kappa`, `kappa3` below 1e-16, `gamma` 2.5e-5/7.9e-7
  and `gamma2` 7.5e-5/2.4e-6, order 4.98; bent state (`k0 = 0.3`) about
  in-plane axes, changes of `kappa`, `kappa3` 2.8e-4/1.8e-5 (order 3.98), of
  `gamma`, `gamma2` order 4.99, membrane 0, against 7.1e-3/1.8e-3 (order 2.00)
  for the linear curvatures and third-order strains of `annular_tsdt_gl`;
- `check_plate_limit`: `R -> infinity` gives the strains of `plate_tsdt_tl`
  (derivation loaded from the path `PLATE_TL` at the top of the file);
- `check_tangent`: `[B]^T [F] [B] + [H(sigma, d)]` against central
  differences of `[B]^T sigma`, `[H]` linear in `d`, `[H(sigma, 0)]` with
  membrane stresses equal to the Hessian of `annular_tsdt_gl`;
- `check_mass`: the mass integrand of the linear field, equal to
  `annular_tsdt_gl`'s;
- `integrand_degree`: `(6, 0)` along `x` and `y`, the value in `modelDB`.

No published total-Lagrangian TSDT in polar coordinates was found; the linear
part, the plate limit and these checks anchor the model.

## Kernels

`[B] = d eps/d d` (13 x 24) quadratic, `[H(sigma, d)]` linear in `d` with all
13 stresses. `fkC_num` = `[B]^T [F] [B] + [H(sigma, d)] - [H(sigma_L, 0)]`,
`fkG_num` = `[H(sigma_L + Nbar, 0)]`, `calc_fint`, `calc_kTv`, `fstrain` (13
strains, non-linear terms times `NLgeom`), `fkM_num` of the linear field; no
piston theory.

## References

- Reddy, J. N., A simple higher-order theory for laminated composite plates,
  J. Appl. Mech. 51 (1984) 745-752, doi:10.1115/1.3167719.
- Schmidt, R., and Reddy, J. N., A refined small strain and moderate rotation
  theory of elastic anisotropic shells, J. Appl. Mech. 55(3) (1988) 611-617,
  doi:10.1115/1.3125837.
- Palmerio, A. F., Reddy, J. N., and Schmidt, R., On a moderate rotation
  theory of laminated anisotropic shells - Part 1. Theory, Int. J. Non-Linear
  Mech. 25(6) (1990) 687-700, doi:10.1016/0020-7462(90)90007-V.
- Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
  Cambridge University Press, 2008, Sec. 1.4, doi:10.1017/CBO9780511619694.
