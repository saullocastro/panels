# Flat plate, TSDT with von Kármán kinematics (`plate_tsdt_donnell`)

Derivation and symbolic checks: `plate_tsdt_donnell.py` (sympy, run `python
plate_tsdt_donnell.py`). Kernels: `write_pyx.py`, which writes
`panels/models/plate_tsdt_donnell.pyx` (`fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA`,
integrated analytically) and `panels/models/plate_tsdt_donnell_num.pyx` (`fkC_num`,
`fkG_num`, `calc_fint`, `calc_kTv`, `fkM_num`, `fkAx_num`, `fkAy_num`,
`calc_fext_follower`, `fkCfollower_num`, integrated numerically),
self-contained: `python write_pyx.py [outdir]`. Only the follower-pressure
kernels come from `../follower_pressure/write_pyx.py`, as for every model.
The field variables are evaluated by `panels/models/fsdt_tsdt_field.pyx`.

## Kinematics

Flat plate, domain `0 <= x <= a`, `0 <= y <= b`, mapped onto `-1 <= xi, eta
<= +1`. With `z` measured from the reference surface and the laminate
between `d - h/2` and `d + h/2` (offset `d`), the displacement field of the
third-order shear deformation theory (TSDT) of Reddy (1984) is:

    u(z) = u + z phix - c1 [(z - d)^3 + d^3] (phix + w,x)
    v(z) = v + z phiy - c1 [(z - d)^3 + d^3] (phiy + w,y)
    w(z) = w

with `c1 = 4/(3 h^2)`. The cubic is written about the mid-surface, `z = d`: the
transverse shear strains vanish at both faces for any offset, `u`, `v`
remain the displacements of the reference surface, and `phix`, `phiy`
are the rotations of the normal at the mid-surface (`check_offset_cubic`).
With `d = 0` this is Reddy's field.

## Strains

`eps = eps0 + z eps1 + [(z - d)^3 + d^3] eps3`, `gam = gam0 + (z - d)^2
gam2`, with the rotations of the
von Kármán terms `bx = -w,x`, `by = -w,y`:

    eps0 = (u,x + bx^2/2,  v,y + by^2/2,  u,y + v,x + bx by)
    eps1 = (phix,x,  phiy,y,  phix,y + phiy,x)
    eps3 = -c1 (phix,x + w,xx,  phiy,y + w,yy,  phix,y + phiy,x + 2 w,xy)
    gam0 = (gyz, gxz)^(0) = (phiy + w,y,  phix + w,x)
    gam2 = -3 c1 gam0

For a flat plate, Donnell's equations reduce to von Kármán's.

## Constitutive law

`13 x 13`: `T^T F_m T`, with `F_m = [[A, B, E, 0, 0], [B, D, F, 0, 0], [E,
F, H, 0, 0], [0, 0, 0, Abar_ts, Dts], [0, 0, 0, Dts, Fts]]` the matrix of
the laminate about its mid-surface, without shear correction, and `T`
the identity but for `T[0:3, 3:6] = d I`, `T[0:3, 6:9] = d^3 I`, see
`Shell._get_lam_ABD`.

## Internal force vector and tangent stiffness

At a point the strains depend on the fifteen generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, phix, phiy, phix,x, phix,y, phiy,x, phiy,y, w,xx, w,yy, w,xy) = [Phi] c

    fint = int [Phi]^T [B]^T sigma dA,          [B] = d eps/d d (13 x 15)
    K_T  = int [Phi]^T ([B]^T [F] [B] + [H(N)]) [Phi] dA
    [H(N)] = [G]^T [N] [G],                     [G] = d(bx, by)/d d

with `N = [[Nxx, Nxy], [Nxy, Nyy]]` (`check_tangent`). Only the membrane
strains are non-linear, quadratic, so `[H]` does not depend on `d`. Split:
`fkC_num` (`NLgeom=1`) `= [B]^T [F] [B] + [H(N - N_L)]`, `fkG_num = [H(N_L
+ Nbar)]`, homogeneous of degree one in `c`, with `N_L` the membrane
stresses of the linear strains and `Nbar` a constant membrane state.
`calc_kTv` evaluates `[B]^T [F] [B] v + [H(N + Nbar)] v` at each point.
`check_numerical`: central differences of `fint` of a pair of terms
against `K_T`.

## Linear kernels

`shape_functions` gives the operators of a term: `B0`, the linear strain
operator, equal to `[B_L] [Phi]` (`check_linear_part`), `G` of the
rotations `(bx, by)`, and the displacements through the thickness;
`build_linear` gives, per unit area of the natural domain:

    kC0  = B0A^T [F] B0B      linear constitutive stiffness
    kG   = GA^T [N] GB        geometric stiffness
    kM                        mass, kinetic energy of u(z), v(z), w(z)
    kAx, kAy, cA              piston theory, q = beta w,x

The mass uses the moments about the mid-surface `J_n`, `n = 0, 1, 2, 3, 4,
6`.
The kernels `fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA` are written in
closed form, with the integrals of Bardell's functions, and `fkM_num`,
`fkAx_num`, `fkAy_num` with the Gauss-Legendre rule, from the expanded
expression of each entry.

## Selection in panels

`Shell(..., model='plate_tsdt_donnell')`; `panels.modelDB.db['plate_tsdt_donnell']`: `dofs = 5`,
`e_num = 13`. The model is supported by `MultiDomain`, with the
connections of `panels.multidomain.connections.kCsdt` and the functions
`*_sdt` of `kCBFycte`, `kCBFxcte`, `kCBFxycte`
(`../../multidomain_penalization`), but not by `StiffPanelBay`, whose
stiffeners assume the 3 DOFs of the CLPT.

## References

- Reddy, J. N., "A simple higher-order theory for laminated composite
  plates", Journal of Applied Mechanics, 51(4), 745-752, 1984.
  https://doi.org/10.1115/1.3167719
- Reddy, J. N., "Mechanics of Laminated Composite Plates and Shells: Theory
  and Analysis", 2nd ed., CRC Press, 2003. https://doi.org/10.1201/b12409
