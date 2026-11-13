# Flat plate, FSDT with von Kármán kinematics (`plate_fsdt_donnell`)

Derivation and symbolic checks: `plate_fsdt_donnell.py` (sympy, run `python
plate_fsdt_donnell.py`). Kernels: `write_pyx.py`, which writes
`panels/models/plate_fsdt_donnell.pyx` (`fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA`,
integrated analytically) and `panels/models/plate_fsdt_donnell_num.pyx` (`fkC_num`,
`fkG_num`, `calc_fint`, `calc_kTv`, `fkM_num`, `fkAx_num`, `fkAy_num`,
`calc_fext_follower`, `fkCfollower_num`, integrated numerically),
self-contained: `python write_pyx.py [outdir]`. Only the follower-pressure
kernels come from `../follower_pressure/write_pyx.py`, as for every model.
The field variables are evaluated by `panels/models/fsdt_tsdt_field.pyx`.

## Kinematics

Flat plate, domain `0 <= x <= a`, `0 <= y <= b`, mapped onto `-1 <= xi, eta
<= +1`. With `z` measured from the reference surface and the laminate
between `d - h/2` and `d + h/2` (offset `d`), the displacement field of the
first-order shear deformation theory (FSDT) is:

    u(z) = u + z phix
    v(z) = v + z phiy
    w(z) = w


## Strains

`eps = eps0 + z eps1`, `gam = gam0`, with the rotations of the
von Kármán terms `bx = -w,x`, `by = -w,y`:

    eps0 = (u,x + bx^2/2,  v,y + by^2/2,  u,y + v,x + bx by)
    eps1 = (phix,x,  phiy,y,  phix,y + phiy,x)
    gam0 = (gyz, gxz)^(0) = (phiy + w,y,  phix + w,x)

For a flat plate, Donnell's equations reduce to von Kármán's.

## Constitutive law

`8 x 8`: `[[A, B, 0], [B, D, 0], [0, 0, Ats]]`, with `Ats` the shear
corrected transverse shear stiffness of the `(yz, xz)` components, set by
`Shell.fsdt_shear_correction`: `'rohwer'` (Rohwer 1988, the default),
`'vlachoutsis'`, `'constant'`, `None`, or a factor `k` multiplying the
uncorrected stiffness.

## Internal force vector and tangent stiffness

At a point the strains depend on the twelve generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, phix, phiy, phix,x, phix,y, phiy,x, phiy,y) = [Phi] c

    fint = int [Phi]^T [B]^T sigma dA,          [B] = d eps/d d (8 x 12)
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

The mass uses the moments `I_n = int rho z^n dz`, `n = 0, 1, 2`.
The kernels `fk0`, `fkG0`, `fkM`, `fkAx`, `fkAy`, `fcA` are written in
closed form, with the integrals of Bardell's functions, and `fkM_num`,
`fkAx_num`, `fkAy_num` with the Gauss-Legendre rule, from the expanded
expression of each entry.

## Selection in panels

`Shell(..., model='plate_fsdt_donnell')`; `panels.modelDB.db['plate_fsdt_donnell']`: `dofs = 5`,
`e_num = 8`. The model is supported by `MultiDomain`, with the
connections of `panels.multidomain.connections.kCsdt` and the functions
`*_sdt` of `kCBFycte`, `kCBFxcte`, `kCBFxycte`
(`../../multidomain_penalization`), but not by `StiffPanelBay`, whose
stiffeners assume the 3 DOFs of the CLPT.

## References

- Reddy, J. N., "Mechanics of Laminated Composite Plates and Shells: Theory
  and Analysis", 2nd ed., CRC Press, 2003. https://doi.org/10.1201/b12409
- Rohwer, K., "Improved transverse shear stiffnesses for layered finite
  elements", DFVLR-FB 88-32, Deutsche Forschungs- und Versuchsanstalt für
  Luft- und Raumfahrt (DFVLR), Institut für Strukturmechanik, Braunschweig,
  1988.
