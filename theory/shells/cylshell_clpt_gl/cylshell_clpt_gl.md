# Cylindrical shell, CLPT with Green-Lagrange membrane strains (`cylshell_clpt_gl`)

Derivation and symbolic checks: `cylshell_clpt_gl.py` (sympy, run
`python cylshell_clpt_gl.py`). Kernels: `write_pyx.py`, which writes
`panels/models/cylshell_clpt_gl.pyx` and `panels/models/cylshell_clpt_gl_num.pyx`
with the kernel templates of `../plate_clpt_gl/write_pyx.py`.

## Why

`cylshell_clpt_sanders` keeps, of the non-linear membrane strains, the
squares of the two rotations of the normal, `phix = -w,x` and `phiy = -w,y +
v/r`. It omits the rotation about the normal, `omega = (v,x - u,y)/2`, which
Sanders (1963) keeps as `omega^2/2` in `exx` and `eyy` (Amabili 2008, Eqs.
(1.133)-(1.134)), and the squares of the in-plane gradients. Under an axial
force `Nxx` its geometric stiffness is then `Nxx w,x^2`, that of the Donnell
kinematics. When a long tube bends as a beam, `w = W(x) cos(theta)`, `v =
-W(x) sin(theta)`, the cross-section translates rigidly and `Nxx v,x^2` is
as large as `Nxx w,x^2`: without it the work of the axial load is halved and
the Euler load of a pinned tube comes out about twice `pi^2 E I/L^2`. The
same holds for any deformation where `v,x` is of the order of `w,x`: low
circumferential wave numbers, and curved panels or blades loaded in their
own plane. For shell-like modes the difference decreases as the
circumferential waves shorten: axially compressed panels with `R/h = 100`,
`L/R = 2`, supported by shear diaphragms, buckle 4.2 % below
`cylshell_clpt_sanders` for a panel of 180 degrees and 1.0 % below for a
panel of 40 degrees (`tests/tests_shell/test_cylshell_clpt_gl.py`); the
pinned tube with `R/h = 100`, `L/R = 50` buckles at 0.989 `N_E`, 1.984 `N_E`
with `cylshell_clpt_sanders`.

The Green-Lagrange membrane strains treat the three rotations alike and
vanish for any rigid motion.

## Kinematics

Assumptions:

- Kirchhoff (CLPT): straight normals that stay normal, no transverse shear;
  degrees of freedom of a term `u, v, w` along the local axes `x` (axial),
  `y = r theta` (arc length), `z` (outward normal);
- shallow through the thickness, `r + z ~ r`, as the other cylinders;
- small strains, constitutive law linear in the Green-Lagrange strains;
- membrane strains: complete Green-Lagrange strains of the mid-surface;
- changes of curvature: the linear ones of `cylshell_clpt_sanders`
  (Amabili 2008, Eqs. (1.136)-(1.138)).

With the local basis `(e_x, e_y, e_z)`, `e_y,y = -e_z/r`, `e_z,y = e_y/r`,
the tangent vectors of the deformed mid-surface are the columns of

    F = [1 + u,x     u,y          ]
        [v,x         1 + v,y + w/r]
        [w,x         w,y - v/r    ]

and `E = (F^T F - I)/2` gives (the strains of the Flügge-Lur'e-Byrne non-linear theory as given by
Amabili 2008, Sec. 1.3.6, Eqs. (1.118)-(1.120), without imperfection and
with `y = R theta`):

    exx = u,x + (u,x^2 + v,x^2 + w,x^2)/2
    eyy = v,y + w/r + (u,y^2 + (v,y + w/r)^2 + (w,y - v/r)^2)/2
    gxy = u,y + v,x + u,x u,y + v,x (v,y + w/r) + w,x (w,y - v/r)

    kxx = -w,xx
    kyy = -w,yy + v,y/r
    kxy = -2 w,xy + (3/2) v,x/r - (1/2) u,y/r

`check_deformation_gradient` derives `F` from the position of the deformed
mid-surface. Compared with `cylshell_clpt_sanders` the membrane strains add
`(u,x^2 + v,x^2)/2`, `(u,y^2 + (v,y + w/r)^2)/2` and `u,x u,y + v,x (v,y +
w/r)`, which contain Sanders' `omega^2/2` (`check_sanders_limit`). For any
rigid rotation, of any angle about any axis, the membrane strains vanish
(`check_rigid_rotation`, numerically at random points, against strains of
the order of the angle squared for the Sanders strains). Under `Nxx` alone
the geometric stiffness is `Nxx (u,x^2 + v,x^2 + w,x^2)`, twice that of the
Sanders strains for the beam mode (`check_axial_geometric_stiffness`).

## Internal force vector and tangent stiffness

At a point the strains depend on the eleven derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, v, w, w,xx, w,yy, w,xy) = [Phi] c

(the values `v` and `w` enter through the curvature of the reference
surface). With `[B] = d eps/d d` (6 x 11, linear in the first eight) and the
six local gradients `beta = (u,x, u,y, v,x, v,y + w/r, w,x, w,y - v/r) = [Gm]
d`:

    fint = int [Phi]^T [B]^T sigma dA
    K_T  = int [Phi]^T ([B]^T [ABD] [B] + [Gm]^T diag(N, N, N) [Gm]) [Phi] dA

split as in `plate_clpt_gl`: `fkC_num` (`NLgeom=1`) `= KC0 + KCNL + KGNL`,
`fkG_num = KG(Nbar + N_L)`, homogeneous of degree one in `c` (`check_tangent`).
The geometric stiffness couples `v` and `w` through `v,y + w/r` and `w,y -
v/r`, so `fkG_num` and `fkG0` have five blocks (`u-u`, `v-v`, `v-w`, `w-v`,
`w-w`). The strains have the form `eps = ([B_L] + [B_NL(beta)]/2) d` of
`theory/shells/tangent_product` (`check_tangent_product_form`), from which
`calc_kTv` is generated with the six `beta`.

## Linear kernels

Without the quadratic terms the strains are those of
`cylshell_clpt_sanders`, so `fk0`, `fkM`, `fkAx`, `fkAy`, `fcA`, `fkM_num`,
`fkAx_num` and `fkAy_num` are imported from its modules. Linear static
solutions, natural frequencies and the linear constitutive stiffness are
those of `cylshell_clpt_sanders`; linear buckling differs through the
geometric stiffness only.

## Integration

`[B]` is linear in the derivatives, so the integrands remain products of four
approximation functions and the default Gauss-Legendre rule `max(2m - 1, 7)`
is exact, as for `plate_clpt_gl`.

## Follower pressure

`calc_fext_follower` and `fkCfollower_num` are those of the Sanders
cylinder, whose area vector `X,x x X,y` is that of the exact deformed
geometry (`follower_pressure.MODELS`). With large rotations use
`follower='quadratic'`.

## Selection in panels

`Shell(..., r=r, model='cylshell_clpt_gl')`;
`panels.modelDB.db['cylshell_clpt_gl']`: `dofs = 3`, `e_num = 6`,
`requires_r = True`. The field module `clpt_field` treats it as a Sanders
model (`is_sanders`, rotation `phiy = -w,y + v/r`) with the Green-Lagrange
strains (`is_gl`), and the connections of `MultiDomain` include the `v/r` of
the rotation (`_rinv_sanders`), as for `cylshell_clpt_sanders`.

## Limitations

- The changes of curvature are linear: for a finite rigid rotation `t`
  about an axis normal to the cylinder axis they are of order `(1 - cos
  t)/r` instead of zero. A total-Lagrangian cylinder (curvatures of the
  deformed normal, as `plate_clpt_tl`) is not implemented.
- The connections of `MultiDomain` stay linear in the local displacements.
- No FSDT or TSDT counterpart.

## References

- Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
  Applied Mathematics, 21(1), 21-36, 1963. https://doi.org/10.1090/qam/147023
- Amabili, M., "Nonlinear Vibrations and Stability of Shells and Plates",
  Cambridge University Press, 2008.
  https://doi.org/10.1017/CBO9780511619694
