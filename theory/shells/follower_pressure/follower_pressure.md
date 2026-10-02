# Follower (hydrostatic) pressure

Derivation: `follower_pressure.py` (sympy), kernels: `write_pyx.py`, which
writes `calc_fext_follower` and `fkCfollower_num` at the end of every
`panels/models/<model>_num.pyx` (also called by `fsdt_tsdt/write_pyx.py` and
`cylshell_clpt_sanders/write_pyx.py`). Tests:
`tests/tests_shell/test_follower_pressure.py` (verification),
`tests/tests_shell/test_follower_pressure_validation.py` (rings, Han et al.
2004), `tests/multidomain/test_follower_pressure_multidomain.py`.

## Conventions

- `x`, `y` are the coordinates of the undeformed mid-surface, `y` being the
  arc length of the cylinders; `u, v, w` are the displacements of the
  mid-surface along `ex, ey, ez` (all models, the FSDT/TSDT rotations
  `phix, phiy` never enter the load).
- `ez` is the normal, **`w` positive outwards** for the cylinders,
  `ey,y = -ez/r`, `ez,y = ey/r`.
- A positive `p` pushes along `+z`, as `fz` of `add_point_load`, such that an
  **external pressure is negative**. `p` is a force per unit undeformed area
  and a function of the undeformed coordinates, `p(x, y)`.

## Virtual work and area vector

    dW_p = int p n.du dA = int p a.du dx dy,    a = X,x × X,y,   X = X0 + u

    A = X,x = (1 + u,x,  v,x,  w,x)
    B = X,y = (u,y,  1 + v,y + kw w,  w,y - kv v)

| model              | `kw`  | `kv`  |
|--------------------|-------|-------|
| plate (CLPT, FSDT, TSDT) | 0 | 0 |
| Donnell cylinder   | `1/r` | 0     |
| Sanders cylinder   | `1/r` | `1/r` |

For the plate and the Sanders cylinder `A × B` is the exact area vector; the
Donnell cylinder neglects `v/r` in the rotation, as its strains do. With
`a = a0 + a1 + a2` (orders 0, 1, 2 in the displacement gradients):

    a0 = (0, 0, 1)
    a1 = (-w,x,  -w,y + kv v,  u,x + v,y + kw w)
    a2 = (v,x (w,y - kv v) - w,x (v,y + kw w),
          w,x u,y - u,x (w,y - kv v),
          u,x (v,y + kw w) - v,x u,y)

## Truncation

With the moderate-rotation ordering of the strains of all models
(`u,x ~ eps`, `w,x ~ eps^(1/2)`, so `du ~ eps^(1/2) dw`), `p a.du` keeps every
term up to `O(eps) p dw` with `a0 + a1`, and `a2` contributes `O(eps^2) p dw`,
the order of what the von Kármán, Donnell and Sanders strains neglect.

- `follower=True` / `'linear'` (default follower): `a = a0 + a1`, the
  consistent truncation. `F_p(c)` is affine, `K_p` constant.
- `follower='quadratic'`: `a = a0 + a1 + a2`, the complete area vector of the
  kinematics, for sensitivity checks of the truncation; `K_p` depends on `c`.

## Force vector and load stiffness

    F_p(c) = int p (Nu a_x + Nv a_y + Nw a_z) dx dy

    kCfollower = -dF_p/dc

    kCfollower[i, j] = -int p N_i . (da/dq_k) (dq_k/dc_j) dx dy

with the generated blocks (per unit `p`, linear part; `'quadratic'` adds the
terms in `qf`, see `output_expressions_python/`):

    [u, w] = +Nu Nw,x          [v, w] = +Nv Nw,y          [v, v] = -kv Nv Nv
    [w, u] = -Nw Nu,x          [w, v] = -Nw Nv,y          [w, w] = -kw Nw Nw

Residual and tangent, `mult = inc` for `cte=False` and `1` for `cte=True`:

    R(c, inc) = fext(inc) - fint(c, inc)
    fint(c, inc) = fint_elastic(c) - sum mult (F_p(c) - F_p(0))
    K_T(c, inc) = kC(c, NLgeom=True, inc) + kG(c, NLgeom=True)
                = ... + kCfollower(c, inc)
    dR/dinc = fext(inc=1., c=c)        (Shell.calc_fext with c)

At `c = 0` the follower load equals the dead load: `calc_fext` returns
`F_p(0)` as for `follower=False`.

Linear buckling with a hydrostatic prestress, `p = lambda p0`:

    c0 = solve(kC0, fext)
    lb(kC0, kG(c0) + kCfollower)   # unsymmetric solvers when needed

## Symmetry

For `a1`, the skew part of `P(d, D) = int p d.a1(D) dA` is (verified
symbolically by `check_symmetry`)

    P(d, D) - P(D, d) = - oint p [(du Dw - dw Du) nx + (dv Dw - dw Dv) ny] ds
                        + int [p,x (du Dw - dw Du) + p,y (dv Dw - dw Dv)] dA

over the loaded region, independently of `kw` and `kv`. Hence `K_p` is
symmetric if and only if (for all admissible fields) `p` is uniform over the
whole integration domain and on each edge `w = 0` or the normal displacement
is zero (`u` on `x = const`, `v` on `y = const`): simply supported or clamped
panels, symmetry planes, closed rings. It is unsymmetric (non-conservative
load) for free or partially restrained edges, for patches (their edges are
inside the shell) and for a varying `p(x, y)`. This is the result of Hibbitt
(1979) and Schweizerhof and Ramm (1984) for a pressure depending on the
reference coordinates: symmetric for closed surfaces or a fixed boundary of
the loaded surface. A hydrostatic head `p(X)` of the current position is
not implemented. For `'quadratic'` the load stiffness at `c != 0` is not
symmetric in general.

## Load surface

`add_pressure_load(..., zp=...)`: a pressure on the surface `z = zp`, e.g.
`zp = h/2` for the outer face, is applied as `p (1 + zp/r)` per unit area of
mid-surface (cylinders only), i.e. its resultant on the radius `r + zp`. The
direction and the change of area are those of the mid-surface (shallow
shell): the terms of relative order `zp k` (`k` the change of curvature) in
the area change and the moment of the in-plane components about the
mid-surface are neglected, of relative order `n^2 zp/r` in the load
stiffness of a ring mode `n` (`~0.13` for `n = 2`, `zp = h/2`, `r/h = 15`).

## Ring closed forms (`ring_buckling`, generalized plane strain, `n = 2`)

Inextensional limit, `p r^3/D`:

| kinematics | follower (hydrostatic) | dead (constant direction) | centrally directed |
|------------|------------------------|---------------------------|--------------------|
| Sanders    | -3                     | -4                        | -9/2               |
| Donnell    | -16/5                  | -4                        | -64/15             |

The Sanders values are the analytical values of Schweizerhof and Ramm (1984),
Table 3 (hydrostatic 3, constant directional 4, centrally directed 4.5).

FSDT, Sanders, follower, inextensional: `p = -3 D/(r^3 (1 + 4 D/(S r^2)))`,
`S` the transverse shear stiffness, which is the shell formula of Han,
Kardomateas and Simitses (2004), Eq. (13a).

With Sanders' kinematics and a dead pressure, the rigid rotation `v = const`
of a ring has the energy `int Nyy (v/r)^2 dA < 0` under an external
pressure, a spurious mode at a vanishing load whenever `v` is free; the
follower load stiffness adds `-int p v^2/r dA`, which cancels it exactly
(`test_rigid_rotation_of_ring_is_neutral_only_for_follower`).
