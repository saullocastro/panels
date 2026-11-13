# Cylindrical shell, TSDT with Green-Lagrange membrane strains (`cylshell_tsdt_gl`)

Derivation and symbolic checks: `cylshell_tsdt_gl.py` (sympy, run `python
cylshell_tsdt_gl.py`, a few seconds). Kernels: `write_pyx.py`, which writes
`panels/models/cylshell_tsdt_gl.pyx` (analytical geometric stiffness `fkG0`) and
`panels/models/cylshell_tsdt_gl_num.pyx` (`fkC_num`, `fkG_num`, `calc_fint`,
`calc_kTv`, `calc_fext_follower`, `fkCfollower_num`), self-contained:
`python write_pyx.py [outdir]`. Only the follower-pressure kernels come
from `../follower_pressure/write_pyx.py`, as for every model.

## Why

`cylshell_tsdt_sanders` keeps, of the non-linear membrane strains, the squares of the
rotations of the normal, `-w,x` and `-w,y + v/r` (Sanders-Koiter), without the rotation about the normal and the squares of the in-plane gradients. The complete Green-Lagrange membrane
strains of `cylshell_clpt_gl` treat the in-plane rotations alike and vanish for any
rigid motion; this model adds them to the third-order shear deformation theory of Reddy of `cylshell_tsdt_sanders`, for the
same reasons as `cylshell_clpt_gl` (in-plane rotations of plates in an assembly,
beam modes of tubes, curved panels loaded in their own plane), with
transverse shear deformation.

## Kinematics

Degrees of freedom of a term: `u, v, w, phix, phiy`, as in `cylshell_tsdt_sanders`.
Membrane strains: the Green-Lagrange strains of the mid-surface of
`cylshell_clpt_gl`, the Flügge-Lur'e-Byrne strains of Amabili (2008), Eqs. (1.118)-(1.120):

    exx = u,x + (u,x^2 + v,x^2 + w,x^2)/2
    eyy = v,y + w/r + (u,y^2 + (v,y + w/r)^2 + (w,y - v/r)^2)/2
    gxy = u,y + v,x + u,x u,y + v,x (v,y + w/r) + w,x (w,y - v/r)

Bending, third-order and transverse shear strains: the linear ones of
`cylshell_tsdt_sanders`:

    kxx = phix,x
    kyy = phiy,y + v,y/r
    kxy = phix,y + phiy,x + (3/2) v,x/r - (1/2) u,y/r
    gyz = phiy + w,y
    gxz = phix + w,x
    kxx3 = -c1 (phix,x + w,xx)
    kyy3 = -c1 (phiy,y + w,yy)
    kxy3 = -c1 (phix,y + phiy,x + 2 w,xy)
    gyz2 = -3 c1 gyz,   gxz2 = -3 c1 gxz,   c1 = 4/(3 h^2)

`check_membrane`: the membrane strains are `E = (F^T F - I)/2` of the
tangent vectors of the deformed mid-surface. `check_linear_part`: `[B_L]
[Phi]` is the linear operator `B0` of `cylshell_tsdt_sanders` in
`../cylshell_tsdt_sanders/cylshell_tsdt_sanders.py`, and the operator `B` of that model at
`bx = -w,x`, `by = -w,y + v/r` is that of these strains without the
quadratic terms of the in-plane gradients: the linear part is exactly
`cylshell_tsdt_sanders`'s, and the non-linear part contains its terms.
`check_rigid_rotation`: for finite rigid rotations about any axis the membrane strains vanish, those of `cylshell_tsdt_sanders` are of the order of the angle squared; the bending strains, linear, are not zero (order `(1 - cos t)/r`), as for `cylshell_clpt_gl`.

## Internal force vector and tangent stiffness

At a point the strains depend on the seventeen generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, v, w, phix, phiy, phix,x, phix,y, phiy,x, phiy,y, w,xx, w,yy, w,xy) = [Phi] c

    fint = int [Phi]^T [B]^T sigma dA,          [B] = d eps/d d (13 x 17)
    K_T  = int [Phi]^T ([B]^T [F] [B] + [H(N)]) [Phi] dA
    [H(N)] = [Gm]^T diag(N, N, N) [Gm]

with `N = [[Nxx, Nxy], [Nxy, Nyy]]` and the six local gradients `beta =
(u,x, u,y, v,x, v,y + w/r, w,x, w,y - v/r) = [Gm] d` (`check_tangent`). Only the membrane strains are
non-linear, quadratic, so `[H]` does not depend on `d`; split as in
`cylshell_clpt_gl`: `fkC_num` (`NLgeom=1`) `= [B]^T [F] [B] + [H(N - N_L)]`,
`fkG_num = [H(N_L + Nbar)]`, homogeneous of degree one in `c`, with `N_L`
the membrane stresses of the linear strains and `Nbar` a constant membrane
state. `calc_kTv` evaluates `[B]^T [F] [B] v + [H(N + Nbar)] v` at each
point. `check_numerical`: central differences of `fint` of a pair of terms
against `K_T`.

## Linear kernels

`fk0`, `fkM`, `fkAx`, `fkAy`, `fcA`, `fkM_num`, `fkAx_num` and `fkAy_num`
are imported from the modules of `cylshell_tsdt_sanders`: linear static solutions,
natural frequencies and the linear constitutive stiffness are those of
`cylshell_tsdt_sanders`; linear buckling differs through the `u-u` and `v-v`, `v-w`
blocks of the geometric stiffness.

## Integration

`[B]` is linear in the derivatives: the integrands are products of four
approximation functions, as for `cylshell_tsdt_sanders`, and the default Gauss-Legendre
rule is unchanged (`modelDB` without `integrand_degree`).

## Follower pressure

`calc_fext_follower` and `fkCfollower_num` use the area vector of the
Sanders cylinder, that of the exact deformed mid-surface, as `cylshell_clpt_gl` (`follower_pressure.MODELS`).

## Selection in panels

`Shell(..., model='cylshell_tsdt_gl')`; `panels.modelDB.db['cylshell_tsdt_gl']`: `dofs = 5`,
`e_num = 13`, `requires_r = True`. The field module `fsdt_tsdt_field`
gives the Green-Lagrange membrane strains with `NLgeom=1` (`is_gl`), and treats the model as a Sanders model (`is_sanders`: the field `phiy` is the rotation of the normal `phiy + v/r`); the connections include `v/r` (`kCsdt._rinv_sanders`).

## Limitations

- The bending, third-order and transverse shear strains are linear, measured
  in the fixed local basis: a bending deformation superposed on a finite
  rotation `t` about an in-plane axis is seen with the factor `cos t`, and a finite rigid rotation about an axis normal to the cylinder axis gives bending strains of order `(1 - cos t)/r`.
  A total-Lagrangian counterpart (finite rotation of the director) is not
  implemented: its definition for independent rotations is open.
- The connections of `MultiDomain` stay linear in the local displacements
  and rotations.
- Small strains: constitutive law linear in the Green-Lagrange strains.

## References

- Amabili, M., "Nonlinear Vibrations and Stability of Shells and Plates",
  Cambridge University Press, 2008. https://doi.org/10.1017/CBO9780511619694
- Reddy, J. N., "Mechanics of Laminated Composite Plates and Shells: Theory
  and Analysis", 2nd ed., CRC Press, 2003. https://doi.org/10.1201/b12409
- Reddy, J. N., and Liu, C. F., "A higher-order shear deformation theory of
  laminated elastic shells", International Journal of Engineering Science,
  23(3), 319-330, 1985. https://doi.org/10.1016/0020-7225(85)90051-5
- Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
  Applied Mathematics, 21(1), 21-36, 1963. https://doi.org/10.1090/qam/147023
- Reddy, J. N., "A simple higher-order theory for laminated composite
  plates", Journal of Applied Mechanics, 51(4), 745-752, 1984.
  https://doi.org/10.1115/1.3167719
