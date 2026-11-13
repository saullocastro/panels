# Flat plate, total-Lagrangian TSDT (`plate_tsdt_tl`)

Derivation and symbolic checks: `plate_tsdt_tl.py` (sympy, run `python
plate_tsdt_tl.py`, about two minutes). Kernels: `write_pyx.py`, which writes
`panels/models/plate_tsdt_tl_num.pyx` (`fkC_num`, `fkG_num`, `calc_fint`,
`calc_kTv`, `calc_fext_follower`, `fkCfollower_num`), self-contained:
`python write_pyx.py [outdir]`. Only the follower-pressure kernels come from
`../follower_pressure/write_pyx.py`, as for every model. No analytical
module: `modelDB` takes `fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA` from
`panels.models.plate_tsdt_gl` (`check_tangent`).

## Why

`plate_tsdt_gl` has the complete Green-Lagrange membrane strains but the
linear curvatures and third-order strains of Reddy's TSDT in the fixed local
frame: a cantilever bent out of its plane follows the linear beam (+10.5 %
in the tip deflection at a tip rotation of 0.46 rad). `plate_fsdt_tl`
measures the bending and transverse shear strains with the second-order
director; this model does the same for Reddy's third-order field, keeping
its traction-free faces for any deformation.

## Kinematics

Reddy's field `u(z) = u + z (phix, phiy, 0) - c1 z^3 (phix + w,x, phiy +
w,y, 0)`, `c1 = 4/(3 h^2)`, written for finite rotations:

    r(z) = X + z d + z^3 q,      q = -c1 (d - N)
    X = (x + u, y + v, w)        d = (phix, phiy, 1 - (phix^2 + phiy^2)/2)
    N = X,x x X,y                (normal of the deformed mid-surface, not normalized)

`d` is the second-order director of `plate_fsdt_tl` (Schmidt and Reddy 1988,
Eq. (16); Palmerio et al. 1990, Part 1, Eq. (20) with `E33(0) = 0`). The
linear part of `d - N` is `(phix + w,x, phiy + w,y, 0)`, Reddy's vector.

Strains: coefficients of `z^0`, `z`, `z^3` of `E_ab(z) = (G_a . G_b -
delta_ab)/2`, `G_a = r,a`, and of `z^0`, `z^2` of `2 E_a3(z) = G_a . r,z`
(`check_strain_measures`). With `X,a . N = 0`, `X,a . N,b = -X,ab . N`:

    exx, eyy, gxy:     Green-Lagrange (plate_tsdt_gl)
    kxx, kyy, kxy:     X,x . d,x, X,y . d,y, X,x . d,y + X,y . d,x (plate_fsdt_tl)
    kxx3 = X,x . q,x = -c1 (kxx + X,xx . N)
    kyy3 = X,y . q,y = -c1 (kyy + X,yy . N)
    kxy3 = X,x . q,y + X,y . q,x = -c1 (kxy + 2 X,xy . N)
    gyz, gxz:          X,y . d, X,x . d (plate_fsdt_tl)
    gyz2, gxz2 = 3 X,a . q = -3 c1 (gyz, gxz)

Reddy's `phi_a,b` becomes the director curvature and `w,ab` the second
fundamental form `X,ab . N`. Dropped, as in the linear TSDT: `z^2`, `z^4`,
`z^6` of `E_ab`; `z`, `z^3`, `z^5` of `2 E_a3`; `E33`. The dropped `z^3`
coefficient of `2 E_a3`, `q,a . d + 3 d,a . q`, has the linear part
`c1 (u,x + v,y),a`, from the stretch of `N`. Generalized derivatives: 21,
the 15 of `plate_tsdt_gl` plus `u,xx, u,yy, u,xy, v,xx, v,yy, v,xy` (through
`X,ab`).

Consequences:

- face shear `gamma + (h^2/4) gamma2 = gamma (1 - 3 c1 h^2/4) = 0` for any
  deformation: Reddy's traction-free faces are kept;
- every strain is a dot product of vectors that rotate with the body: a
  rigid rotation changes them only through `|d|^2 - 1 = |phi|^4/4`.

## Why not the third-order terms linear

The alternative keeps the first-order part of `plate_fsdt_tl` with the
linear third-order strains of `plate_tsdt_gl`, `kappa3 = -c1 (phi_a,b + w,ab)`,
`gamma2 = -3 c1 (phi_a + w,a)` (`option='A'` of `strains`). Same linear part,
cheaper, but:

- the face shear is `X,a . d - (phi_a + w,a)`, cubic, not zero
  (`check_face_shear`);
- a rigid rotation about an oblique axis gives `gamma2` of `O(t^2)` and a
  rotated bent state `kappa3` changes of `O(t^2)` (`check_rigid_rotation`;
  `c1 = 1`, `t = 0.2/0.1`: `gamma2` 4.1e-2/1.0e-2 against 1.2e-4/3.7e-6 of this
  model, `kappa3` 5.9e-3/1.5e-3 against 1.2e-4/7.4e-6);
- in large bending the director shear `X,x . d` and the linear shear
  `phix + w,x` cannot both vanish; the transverse shear energy contains both,
  `G h [g^2 - (2/3) g gL + gL^2/5]`, which then penalizes the bending by
  `(L/h)^2`: it locks.

A prototype of each option, compiled and run under the same conditions
(14 x 4 terms for the strips, 12 x 12 for the plate, Newton-Raphson):

| Case | linear third order | this model | `plate_fsdt_tl` | `plate_tsdt_gl` |
|---|---|---|---|---|
| internal forces of a rigid rotation, free plate a/h = 6, axis (1, 2, 3), `|fint|/|KC0 c|` at 0.2/0.1 rad | 8.4e-3/8.3e-3 | 4.3e-6/5.3e-7 | 5.1e-6/6.3e-7 | 3.3e-2/3.3e-2 |
| thin strip L/h = 500, tip force, alpha = 1 (0.46 rad), w and -u against the elastica | -25.5 %, -46.7 % | -0.22 %, -0.47 % | -0.27 %, -0.59 % | +10.5 %, +24.4 % |
| idem alpha = 3 (0.99 rad) | -49.9 %, -79.4 % | -1.44 %, -3.84 % | -1.77 %, -4.69 % | +20.6 %, +96.9 % |
| thick strip L/h = 5 against the Reissner beam (shear stiffness from the linear deflection of each model), alpha = 3 (0.98 rad) | -3.43 %, -7.90 % | -0.81 %, -0.20 % | -1.44 %, -2.15 % | +20.3 %, +106 % |
| thick strip b = h = 0.2, oblique tip force (0, 1, 1)/sqrt(2), alpha = 2: (u, v, w)/L | (-0.073, 0.277, 0.216) | (-0.170, 0.361, 0.358) | (-0.169, 0.362, 0.357) | (-0.220, 0.364, 0.428) |
| simply supported plate a/h = 5, nu = 0.3, pressure 1.5e9 Pa, centre w/h | 0.67451 | 0.67582 | 0.67686 | 0.69187 |
| time of kC (NLgeom), kG, fint, 12 x 12 terms [s] | 0.66, 0.21, 0.047 | 0.88, 0.36, 0.054 | 0.56, 0.19, 0.040 | 0.32, 0.07, 0.022 |

The elastica (`theta'' = -alpha cos theta`, alpha = P L^2/EI) gives w =
0.3017208 L, -u = 0.0564332 L at alpha = 1 and w = 0.6032534 L, -u =
0.2544202 L at alpha = 3. The Reissner beam: `r' = (1 + e) a1 + g a3`, `e =
P sin(th)/EA`, `g = P cos(th)/kGA`, `th' = M/EI`, clamped, `M(L) = 0`.

Literature: Amabili (2018), Sec. 3.9, keeps the third-order terms (and the
curvatures) linear in a model with von Kármán-type nonlinearity; Schmidt and
Reddy (1988), Eqs. (17)-(19), for an expansion of any order, have products
of the rotation terms of all orders in the strains, i.e. nonlinear
third-order strains, as here.

## Checks (`python plate_tsdt_tl.py`)

- `check_strain_measures`: first-order strains = `plate_fsdt_tl`; `kappa3`,
  `gamma2` = the `z^3`, `z^2` coefficients of the Green-Lagrange strains of
  `r(z)`; linear part of the dropped `z^3` term of `2 E_a3`.
- `check_linear_part`: `[B(0)] [Phi]` = `B0` of `plate_tsdt_donnell`; linear
  strains = `plate_tsdt_gl` (both options); membrane = `plate_tsdt_gl`.
- `check_face_shear`: zero; option A: `X,a . d - (phi_a + w,a)`.
- `check_rigid_rotation`: undeformed plate rotated about random axes:
  membrane, `kappa`, `kappa3` zero, `gamma`, `gamma2` `O(t^5)`; bent state
  (`k0 = 0.3`) rotated about in-plane axes: all `O(t^4)`; option A:
  `gamma2`, `kappa3` `O(t^2)`.
- `check_tangent`: `d(B^T sigma)/dD = B^T F B + H(sigma, D)`, `H` linear in
  `D`; `fkG_num` homogeneous; `H0` with membrane stresses = Hessian of
  `plate_tsdt_gl`.
- `check_numerical`: central differences of `fint` of a pair of terms.
- `check_integrand_degree`: `6p` (`modelDB` `integrand_degree (6, 0)`).

## References

- Reddy, J. N., Mechanics of Laminated Composite Plates and Shells, 2nd ed.,
  CRC Press, 2003. https://doi.org/10.1201/b12409
- Schmidt, R., Reddy, J. N., J. Appl. Mech. 55(3), 611-617, 1988.
  https://doi.org/10.1115/1.3125837
- Palmerio, A. F., Reddy, J. N., Schmidt, R., Int. J. Non-Linear Mech.
  25(6), 687-700, 1990. https://doi.org/10.1016/0020-7462(90)90007-V
- Amabili, M., Nonlinear Mechanics of Shells and Plates in Composite, Soft
  and Biological Materials, Cambridge University Press, 2018.
  https://doi.org/10.1017/9781316422892
