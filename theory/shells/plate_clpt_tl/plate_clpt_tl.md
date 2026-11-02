# Flat plate, total-Lagrangian Kirchhoff-Love (`plate_clpt_tl`)

Derivation and symbolic checks: `plate_clpt_tl.py` (sympy, run
`python plate_clpt_tl.py`, about one minute). Kernels: `write_pyx.py`, which
writes `panels/models/plate_clpt_tl_num.pyx` with the helpers of
`plate_clpt_gl/write_pyx.py`.

## Why

`plate_clpt_gl` has the complete Green-Lagrange membrane strains, exact for
any rigid rotation, but the changes of curvature `-w,ab` of the fixed local
frame. A bending deformation superposed on a rigid rotation `t` about an
in-plane axis, e.g. a local buckle of a rotated skin, is then seen with the
factor `cos t`, and its bending energy with `cos^2 t`. Measuring the
curvatures with the normal of the deformed mid-surface removes this error:
all the strains become invariant under any rigid rotation of any deformed
state, and the model is a total-Lagrangian Kirchhoff-Love plate.

## Kinematics

Degrees of freedom: `u, v, w` along the fixed local axes, as in
`plate_clpt_donnell` and `plate_clpt_gl`. With `X = (x, y, 0) + u` and the
area vector of the deformed mid-surface

    a = X,x x X,y = (v,x w,y - w,x (1 + v,y),
                     w,x u,y - (1 + u,x) w,y,
                     (1 + u,x)(1 + v,y) - v,x u,y)

the strains are

    exx = u,x + (u,x^2 + v,x^2 + w,x^2)/2
    eyy = v,y + (u,y^2 + v,y^2 + w,y^2)/2
    gxy = u,y + v,x + u,x u,y + v,x v,y + w,x w,y
    kxx = -a . u,xx,   kyy = -a . u,yy,   kxy = -2 a . u,xy

with `u,ab = (u,ab, v,ab, w,ab) = X,ab`, the plate being flat. The linear
part is that of the other plates; the quadratic part of `kxx` is
`w,x u,xx + w,y v,xx - (u,x + v,y) w,xx` (`check_membrane_and_linear_part`).

At a point the strains depend on fifteen generalized derivatives, the six
gradients and the nine second derivatives of `u, v, w`.

## Area vector rather than unit normal

`|a|^2 = det(F^T F) = (1 + 2 exx)(1 + 2 eyy) - gxy^2`
(`check_area_vector_norm`): `|a| = 1` for any rigid motion and
`1 + exx + eyy + O(eps^2)` otherwise, so the curvatures are `|a|` times
those of the unit normal. The choice of `a`:

- keeps the strains polynomial (cubic) in the displacement derivatives, the
  integrands polynomial, and the Gauss-Legendre rule exact; `n = a/|a|`
  would bring `1/|a|` and its derivatives into the kernels and no rule would
  be exact;
- keeps the strains exactly invariant under rigid rotations of a deformed
  state, `(R A) x (R B) = R (A x B)` (`check_rigid_rotation`, symbolically
  about `x`, `y`, `z`, numerically about oblique axes);
- scales the bending stiffness by `|a|^2 = 1 + 2 (exx + eyy) + O(eps^2)`, a
  relative error of the order of the membrane strain (0.2 % at a membrane
  strain of 1e-3), the order of the terms already neglected by the
  Kirchhoff-Love kinematics with a constitutive law linear in the
  Green-Lagrange strains. For a plate under pressure with a centre
  deflection of 11 times its thickness (membrane strains about 3e-3), the
  deflection differs from that of `plate_clpt_gl` by 0.13 %.

## Internal force vector and tangent stiffness

    fint = int [Phi]^T [B]^T sigma dA,          [B] = d eps/d d  (6 x 15, quadratic)
    K_T  = int [Phi]^T ([B]^T [ABD] [B] + [H(sigma, d)]) [Phi] dA
    [H(sigma, d)] = sum_r sigma_r d^2 eps_r/d d^2   (linear in d)

The membrane part of `[H]` is `diag(N, N, N)` on the gradients, as in
`plate_clpt_gl`; the moment part couples the gradients with the second
derivatives.

Split (`check_tangent`):

    fkG_num = [H0(sigma_L + Nbar)],      H0 = H(., d = 0)
    fkC_num = [B]^T [ABD] [B] + [H(sigma, d)] - [H0(sigma_L)]

`fkG_num` is homogeneous of degree one in `c`, as the linear buckling
analysis requires, and `fkC_num + fkG_num` is the exact Jacobian of
`calc_fint`. `[H0]` has the `u-u`, `v-v`, `w-w` blocks of `plate_clpt_gl`
(of `N_L + Nbar`) and `u-w`, `v-w` blocks of the moments `M_L` of the linear
strains. With a constant membrane stress only it is the geometric stiffness
of `plate_clpt_gl`, so the analytical kernels (`fk0`, `fkG0`, `fkM`, `fkAx`,
`fkAy`, `fcA`) are those of `panels.models.plate_clpt_gl` and no analytical
module is generated. A linear buckling analysis with the stresses of a
linear static solution differs from `plate_clpt_gl` only through the
prebuckling moments, i.e. not at all for a symmetric laminate under
in-plane loads.

The split, the null-space and penalty connections and the follower
pressures are unaffected: the degrees of freedom are still the
displacements in the fixed local frames, the connections stay linear in
them, and the follower pressure uses the exact area vector of the plate
(`follower_pressure.MODELS`, geometry `plate`).

## Product of the tangent stiffness by a vector

The strains are no longer of the form `([B_L] + [B_NL(beta)]/2) d` with
`[B_NL]` linear in `beta` assumed by `theory/shells/tangent_product`.
`calc_kTv` evaluates, at each point, with `d = [Phi] c` and `dv = [Phi] v`,

    kTv += [Phi]^T ([B(d)]^T [ABD] [B(d)] dv + [H(sigma + Nbar, d)] dv)

at the cost of about one `calc_fint`.

## Integration

Along `x`, an approximation function of degree `p` and its first and
second derivatives have the degrees `p`, `p - 1`, `p - 2`, and each
quadratic term of `a` has one derivative in `x` and one in `y`. The
integrands of `fint` and `K_T` have the degree `6p - 2` along each
direction (`Mxx` cubic times the variation of `kxx`, e.g. `u,x v,y w,yy`),
against `4p` for `plate_clpt_gl` (`check_integrand_degree`). The
Gauss-Legendre rule of `n` points is exact for `2n - 1 >= 6p - 2`, i.e.
`n >= 3p`: the default of `Shell.nx` for this model (`modelDB` key
`'integrand_degree': (6, -2)`, `panels.shell.gauss_points`), 21 points for
`m = 8` against 15. The terms of the highest degree are products of five or
six of the highest approximation functions: with the 15 points of
`plate_clpt_gl` and gradients of order 0.5 the internal forces differ by
3e-13 (`tests/tests_shell/test_plate_clpt_tl.py`), so that `nx =
gauss_points(m)` is a cheaper alternative in practice.

## Limitations

- The connections of `MultiDomain` stay linear: the rotation continuity
  `w,y` (penalty or null space) is that of small rotations, with a mismatch
  of order `1 - cos t` at a joint rotated by `t`.
- Small strains: constitutive law linear in the Green-Lagrange strains, the
  thickness does not change, and the curvature is `|a|` times that of the
  unit normal.
