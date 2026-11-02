# Flat plate, CLPT with Green-Lagrange membrane strains (`plate_clpt_gl`)

Derivation and symbolic checks: `plate_clpt_gl.py` (sympy, run
`python plate_clpt_gl.py`). Kernels: `write_pyx.py`, which writes
`panels/models/plate_clpt_gl.pyx` and `panels/models/plate_clpt_gl_num.pyx`.

## Why

The von Kármán strains of `plate_clpt_donnell` keep the quadratic terms of
`w` only. In an assembly of plates at right angles, e.g. a wing box, the
rotation of the box about a horizontal axis is an out-of-plane rotation of the
skins (`w,x`) and an in-plane rotation of the spars (`v,x`). The skins see
`w,x^2/2`, the spars nothing, so a rigid rotation `t` of the box stretches
one against the other by `t^2/2`: the box stiffens once its tip deflection is
of the order of its height. The Green-Lagrange membrane strains treat both
rotations alike and vanish for any rigid rotation.

## Kinematics

Degrees of freedom of a term: `u, v, w`, the displacements of the mid-surface
along the fixed local axes `x, y, z` of the plate, as in `plate_clpt_donnell`.
At a point the strains depend on the nine generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, w,xx, w,yy, w,xy) = [Phi] c

with `[Phi]` (9 x 3 per term) built from the approximation functions.

Membrane strains, the complete Green-Lagrange strains of the mid-surface,
`E = (F^T F - I)/2`, `F = [e_x + u,x  e_y + u,y]` (3 x 2), `u = (u, v, w)`:

    exx = u,x + (u,x^2 + v,x^2 + w,x^2)/2
    eyy = v,y + (u,y^2 + v,y^2 + w,y^2)/2
    gxy = u,y + v,x + u,x u,y + v,x v,y + w,x w,y

Changes of curvature, as in `plate_clpt_donnell`:

    kxx = -w,xx,   kyy = -w,yy,   kxy = -2 w,xy

Dropping the quadratic terms of `u,x, u,y, v,x, v,y` gives the von Kármán
strains (`check_von_karman_limit`). For a rigid rotation `u = (R - I) X`,
the columns of `F` are two columns of `R` and `F^T F = I`: the membrane strains
vanish for any angle and axis (`check_rigid_rotation`), whereas the von Kármán
`exx` of an in-plane rotation `t` is `cos t - 1`.

## Internal force vector and tangent stiffness

With `sigma = (Nxx, Nyy, Nxy, Mxx, Myy, Mxy) = [ABD] eps` and
`[B] = d eps/d d` (6 x 9, linear in the six gradients):

    fint = int [Phi]^T [B]^T sigma dA

    K_T = int [Phi]^T ( [B]^T [ABD] [B] + [H(N)] ) [Phi] dA

    [H(N)] = sum_r sigma_r d^2 eps_r/d d^2 = diag(N, N, N, 0),   N = [[Nxx, Nxy], [Nxy, Nyy]]

on the pairs `(u,x, u,y)`, `(v,x, v,y)` and `(w,x, w,y)`: the geometric
stiffness has `u-u` and `v-v` blocks besides the `w-w` block of the von
Kármán plate (`check_tangent`, `check_numerical`).

Split, as in every model of `panels`:

    K_T = KC0 + KCNL + KGNL   (fkC_num, NLgeom=1: [B]^T [ABD] [B] + [H(N_NL)])
        + KG(Nbar + N_L)      (fkG_num: [H(N_L + Nbar)])

with `N_L = [A] eps_L + [B] kappa` the membrane stresses of the linear strains,
`N_NL = [A] eps_NL` those of the quadratic strains, and `Nbar` a constant
membrane stress state. `fkG_num` is homogeneous of degree one in `c`, as the
linear buckling analysis requires. The analytical `fkG0` (constant `Nbar`) has
the same three blocks.

Form of `theory/shells/tangent_product`: `eps = ([B_L] + [B_NL(beta)]/2) d`,
`[B_NL]` linear in the six gradients `beta = (u,x, u,y, v,x, v,y, w,x, w,y)`,
and `[G]^T diag(N, N, N) [G]` (`check_tangent_product_form`); `calc_kTv` is
generated from it, with `beta` of six components instead of two rotations.

## Linear kernels

With zero gradients the strains are those of `plate_clpt_donnell`, so the
linear constitutive stiffness, the mass and the aerodynamic matrices (`fk0`,
`fkM`, `fkAx`, `fkAy`, `fcA`, `fkM_num`, `fkAx_num`, `fkAy_num`) are imported
from its modules. A linear static solution is identical; a linear buckling
eigenvalue differs only through the `u-u` and `v-v` blocks of `KG`, which
vanish for a single symmetric laminate (in-plane and bending decoupled) and are
of the order of the membrane strain otherwise.

## Integration

`[B]` is linear in the gradients, so the integrands of `fint`, `KC` and `KG`
remain products of four approximation functions, as for `plate_clpt_donnell`,
and the default Gauss-Legendre rule `nx = max(2m - 1, 7)` is unchanged.

## Kernel layout

At each integration point `[B]`, `K = [B]^T [ABD] [B] + [H]` (9 x 9) are
computed once. The entry of the pair of terms `A = (i, j)`, `B = (k, l)` and
the degrees of freedom `d`, `e` is `sum_pq Phi_A[p, d] K[p, q] Phi_B[q, e]`:
`TA[d, q] = sum_p Phi_A[p, d] K[p, q]` in the loop over `A`, then
`sum_q TA[d, q] Phi_B[q, e]` in the loop over `B`, i.e. 27 products per pair
of terms. The approximation functions are tabulated per integration point with
their derivatives in `x` and `y`.

## Follower pressure

`calc_fext_follower` and `fkCfollower_num` use the total gradients and the
exact area vector of a plate (`follower_pressure.MODELS`, geometry `plate`).
With large rotations use `follower='quadratic'`, the complete area vector;
`'linear'` truncates it under the moderate-rotation ordering of the von Kármán
strains.

## Limitations

- The curvatures are measured in the fixed local frame: a bending deformation
  superposed on a rigid rotation `t` (e.g. a local buckle of a rotated skin) is
  seen with the factor `cos t`, and its bending energy with `cos^2 t`.
- The connections of `MultiDomain` stay linear in the local displacements:
  displacement continuity is exact for any rotation, the rotation continuity
  `w,y` (penalty or null space) is that of small rotations, with a mismatch of
  order `1 - cos t` at large rotations of the joint.
- Small strains: the constitutive law is linear in the Green-Lagrange strains
  (St. Venant-Kirchhoff).
