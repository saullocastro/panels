# Cylindrical shell, total-Lagrangian Kirchhoff-Love (`cylshell_clpt_tl`)

Derivation and symbolic checks: `cylshell_clpt_tl.py` (sympy, run
`python cylshell_clpt_tl.py`, about three minutes). Kernels: `write_pyx.py`,
which writes `panels/models/cylshell_clpt_tl_num.pyx`, self-contained: the
layout helpers of `../plate_clpt_gl/write_pyx.py` are repeated in it, with
the radius added.

## Why

`cylshell_clpt_gl` has the complete Green-Lagrange membrane strains, exact for
any rigid motion, but the linear changes of curvature of Sanders, measured in
the fixed local basis. When a part of the shell rotates through a large angle
about an axis normal to the cylinder axis, e.g. the cap under a point load
that snaps through and inverts its curvature, the bending deformation is
measured with an error of order `(1 - cos t)/r` and the bending energy with
the factor `cos^2 t`. Measuring the curvatures with the normal of the deformed
mid-surface removes this error, as `plate_clpt_tl` does for the plate.

## Kinematics

Assumptions: those of `cylshell_clpt_gl` (Kirchhoff, `r + z ~ r`, small
strains, complete Green-Lagrange membrane strains), with the changes of
curvature of the deformed mid-surface.

With the local basis `(e_x, e_y, e_z)`, `e_y,y = -e_z/r`, `e_z,y = e_y/r`,
and `X = X0 + u e_x + v e_y + w e_z` (`check_position_derivatives`, from the
global position):

    X,x  = (1 + u,x, v,x, w,x)
    X,y  = (u,y, 1 + v,y + w/r, w,y - v/r)
    X,xx = (u,xx, v,xx, w,xx)
    X,xy = (u,xy, v,xy + w,x/r, w,xy - v,x/r)
    X,yy = (u,yy, v,yy + 2 w,y/r - v/r^2, w,yy - 2 v,y/r - w/r^2 - 1/r)

Membrane strains `E = (F^T F - I)/2`, `F = [X,x  X,y]`: those of
`cylshell_clpt_gl`, the Flügge-Lur'e-Byrne strains of Amabili (2008), Eqs.
(1.118)-(1.120) (`check_membrane`).

Changes of curvature, with the area vector `a = X,x x X,y`:

    kxx = -a . X,xx
    kyy = -a . X,yy - 1/r - (exx + 2 eyy)/r
    kxy = -2 a . X,xy - gxy/(2 r)

- `-(a . X,ab - (1 + exx + eyy) b0_ab)` is the change of the second
  fundamental form `b_ab = n . X,ab` (van der Heijden 2008, Sec. 3.16,
  p. 182), `b0_yy = -1/r` the only non-zero component of the undeformed
  cylinder, times the area stretch `|a| = 1 + exx + eyy + O(eps^2)`
  (`check_area_vector_norm`), with `|a|` replaced by its first-order part in
  the reference term: the strains stay cubic polynomials in the displacement
  derivatives and the Gauss-Legendre rule is exact.
- `-eyy/r` in `kyy` and `-gxy/(2 r)` in `kxy` make the linear part Sanders'
  linear changes of curvature (`check_linear_part`), derived step by step
  below.

## Why the linear part is Sanders'

Every equation of this section is checked by `sanders_linear_part.py`
(sympy, about ten seconds; labels `cyl-tl-*` of
`doc/source/cylshell_clpt_tl.rst`, where the derivation is written out).
Koiter's quantities, van der Heijden (2008), Eqs. (3.16.5)-(3.16.8), p.
183, for the cylinder Eq. (3.17.3), p. 202 (there in the coordinates `x/R`,
`y/R`, outward normal):

    theta_xx = u,x   theta_yy = v,y + w/r   theta_xy = (u,y + v,x)/2
    omega_xy = (v,x - u,y)/2   phi_x = w,x   phi_y = w,y - v/r

1. Area vector (exact, `cyl-tl-a`):

       a = (v,x (w,y - v/r) - w,x (1 + v,y + w/r),
            w,x u,y - (1 + u,x)(w,y - v/r),
            (1 + u,x)(1 + v,y + w/r) - v,x u,y)

   `|a|^2 = (1 + 2 exx)(1 + 2 eyy) - gxy^2`, and to first order
   `a = (1 + theta_xx + theta_yy) e_z - phi_x e_x - phi_y e_y`.
2. Linearization, `a . X,ab ~ e_z . X0,ab + da . X0,ab + e_z . dX,ab`; only
   `X0,yy = -e_z/r` is not zero, normal to the surface:

       a . X,xx ~ w,xx
       a . X,xy ~ w,xy - v,x/r
       a . X,yy ~ -1/r - (theta_xx + theta_yy)/r + w,yy - 2 v,y/r - w/r^2

3. Dividing by `|a| ~ 1 + theta_xx + theta_yy` removes the stretch term
   `-(theta_xx + theta_yy)/r` and leaves the linear changes of the second
   fundamental form:

       db_xx = w,xx   db_xy = w,xy - v,x/r   db_yy = w,yy - 2 v,y/r - w/r^2

4. Koiter's Eq. (3.16.8) for the cylinder (`b^y_y = -1/r`, Cartesian
   surface coordinates):

       rho_xx = phi_x,x = w,xx
       rho_yy = phi_y,y = w,yy - v,y/r
       rho_xy = (phi_x,y + phi_y,x)/2 - omega_xy/(2r)
              = w,xy - (3/4) v,x/r + (1/4) u,y/r

   and `(kxx, kyy, kxy) = (-rho_xx, -rho_yy, -2 rho_xy)` are Sanders' linear
   changes of curvature of `cylshell_clpt_sanders`:

       kxx = -w,xx,   kyy = -w,yy + v,y/r,   kxy = -2 w,xy + (3/2) v,x/r - (1/2) u,y/r

5. The plain change of the second fundamental form is not Sanders':
   `db_yy = rho_yy - theta_yy/r`, `db_xy = rho_xy - theta_xy/(2r)` (with
   `v,x = theta_xy + omega_xy`, the second fundamental form sees the strain
   and the rotation about the normal), i.e. in the engineering components of
   panels `kb = kS + (0, eyy/r, gxy/(2r))`. For a uniform radial
   displacement `w0`, `kb_yy = w0/r^2`, whereas Sanders' and this model's
   `kyy` are exactly zero (`cyl-tl-expansion`). Both measures are equally
   accurate (Koiter, van der Heijden 2008, p. 202); only the second has the
   linear part of `cylshell_clpt_sanders`.
6. The model, term by term (`cyl-tl-k-lin`): in `kyy`, the `1/r` of `-a .
   X,yy` cancels with `-1/r`; its stretch term `(theta_xx + theta_yy)/r`
   cancels with the linear part of `-(exx + eyy)/r` (the stretch put on the
   reference term); `-eyy/r` cancels `w/r^2` and half of `2 v,y/r`. In
   `kxy`, `-gxy/(2r) = -(u,y + v,x)/(2r)` turns `2 v,x/r` into `(3/2) v,x/r
   - (1/2) u,y/r`. Without the stretch on the reference term, `-a . X,yy -
   1/r - eyy/r` would keep the further linear term `(theta_xx +
   theta_yy)/r`.
7. Invariance (`cyl-tl-invariance`): the model's curvatures and the plain
   change of the second fundamental form do not change under a rigid
   rotation of a deformed state; Sanders' linear curvatures (those of
   `cylshell_clpt_gl`) do. The model has both properties: invariant, with
   Sanders' linear part.

Every term is built from `a`, `X,ab` and the Green-Lagrange strains, which do
not change under a rigid rotation `X -> R X` of a deformed state: the six
strains are invariant under any rigid rotation of any deformed state
(`check_rigid_rotation`, random oblique axes up to 3 rad, rigid motions of the
undeformed cylinder and rotations of a random deformed state) and vanish for
any rigid motion.

## Comparison with Novozhilov's curvatures (Amabili 2008, Sec. 1.3.7)

Amabili's Eqs. (1.130)-(1.132), `w0 = 0`, are the coefficients of `z` of the
three-dimensional Green-Lagrange strains of Novozhilov's displacement field
(1.124)-(1.129), quadratic in the displacements. Term by term
(`check_novozhilov`), Amabili's minus the curvatures above:

- `kxx`: zero, linear and quadratic parts, when `u,y = v,x`;
- `kyy`: `exx/r`, linear and quadratic parts;
- `kxy`: `-gxy/(2 r)`, linear and quadratic parts, plus `v,x w,xx`, which is
  `gxy w,xx/2` when `u,y = v,x`.

The differences are terms of the order of the membrane strain divided by the
radius or times the curvature (`plate_clpt_tl` scales its curvatures by the
area stretch, a term of the same kind), except when the rotation about the
normal, `(v,x - u,y)/2`, does not vanish: Novozhilov's rotations of the
normal (1.127)-(1.128) have `u,y` where the area vector has `v,x`, and
conversely, exact for a symmetric in-plane displacement gradient only.

## Internal force vector and tangent stiffness

Seventeen generalized derivatives at a point, `u,x, u,y, v,x, v,y, w,x, w,y,
v, w` and the nine second derivatives of `u, v, w`:

    fint = int [Phi]^T [B]^T sigma dA,          [B] = d eps/d d  (6 x 17, quadratic)
    K_T  = int [Phi]^T ([B]^T [ABD] [B] + [H(sigma, d)]) [Phi] dA

and the split of `plate_clpt_tl` (`check_tangent`):

    fkG_num = [H0(sigma_L + Nbar)],      H0 = H(., d = 0)
    fkC_num = [B]^T [ABD] [B] + [H(sigma, d)] - [H0(sigma_L)]

With a constant membrane stress only, `[H0]` is the `[H]` of
`cylshell_clpt_gl`, so that the analytical kernels of `cylshell_clpt_gl`
(`fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA`, the linear ones being those of
`cylshell_clpt_sanders`) are those of this model, and no analytical module is
generated: linear static solutions, natural frequencies and linear buckling
loads with a membrane prebuckling state are those of `cylshell_clpt_gl`.
`calc_kTv` evaluates `[B]^T [ABD] [B] v + [H] v` at each point, the strains
not being of the form assumed by `theory/shells/tangent_product`. Follower
pressures: the exact area vector of the Sanders cylinder
(`follower_pressure.MODELS`). Connections of `MultiDomain`: those of the
Sanders-Koiter rotations, `phiy = -w,y + v/r` (`_rinv_sanders`).

## Integration

Degree of the integrands (`check_integrand_degree`): `6p - 2` along `x`, as
`plate_clpt_tl`, and `6p` along `y`, where `v` and `w` enter without
derivative. The default number of Gauss-Legendre points is `3p + 1` along
both directions (`modelDB` key `'integrand_degree': (6, 0)`), 25 for
`m = 9`.

## Limitations

- The connections of `MultiDomain` stay linear (small rotations at the
  joints), as for `plate_clpt_tl`.
- Small strains, `r + z ~ r` through the thickness, and the curvatures are
  those of the area vector, with the reference term to first order in the
  membrane strains.
