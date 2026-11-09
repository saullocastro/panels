# Cylindrical shell, CLPT with Sanders-Koiter kinematics (`cylshell_clpt_sanders`)

Derivation: `cylshell_clpt_sanders.py` (sympy; `build()` gives the
integrands, and `python cylshell_clpt_sanders.py` writes them to
`output_expressions_python/sympy_<name>.txt`). Kernels, in two steps:

    python derive_expressions.py    # writes sanders_exprs.py
    python write_pyx.py             # reads sanders_exprs.py, writes the .pyx

`write_pyx.py` writes `panels/models/cylshell_clpt_sanders.pyx` (integrated
analytically) and `panels/models/cylshell_clpt_sanders_num.pyx` (integrated
numerically, with the approximation functions tabulated per integration
point by `panels/dev/tabulate.py`), and appends to the latter the kernels of
`../follower_pressure/write_pyx.py` and `../tangent_product/write_pyx.py`.

## Kinematics

Cylinder of radius `r`, `x` along the axis, `y = r theta` the arc length
along the circumference (`(1/r) (.),theta = (.),y`), `w` positive outwards,
shallow-shell assumption `r + z ~ r`. Degrees of freedom of a term: `u, v,
w`, sums of products of Bardell's functions `f_i(xi) g_j(eta)`, `xi = 2x/a -
1`, `eta = 2y/b - 1`. Rotations of the normal and displacement field, `z`
measured from the reference surface:

    phix = -w,x,   phiy = -w,y + v/r
    u(z) = u + z phix,   v(z) = v + z phiy,   w(z) = w

## Strains

Sanders (1963), with the changes of curvature as written by Amabili (2008),
Sec. 1.3.8, Eqs. (1.136)-(1.138):

    exx = u,x + phix^2/2
    eyy = v,y + w/r + phiy^2/2
    gxy = u,y + v,x + phix phiy
    kxx = -w,xx
    kyy = -w,yy + v,y/r
    kxy = -2 w,xy + (3/2) v,x/r - (1/2) u,y/r

All the strains vanish for any small rigid-body motion, which matters for
long shells and deformations of long wavelength, e.g. low circumferential
wave numbers. Dropping the `v/r` of `phiy` and the `1/r` terms of the
changes of curvature gives `cylshell_clpt_donnell`.

The non-linear terms are a simplification of the Sanders-Koiter theory for
small strains and moderately small rotations: the term `omega3^2/2 =
(v,x - u,y)^2/8` of the rotation about the normal, `omega3 = (v,x -
u,y)/2`, in `exx` and `eyy` (Amabili 2008, Sec. 1.3.8, Eqs.
(1.133)-(1.134)) is omitted, i.e. the rotation about the normal is assumed
small compared with the other two. The geometric stiffness of an axial load
is then that of the Donnell kinematics.

## Constitutive law

Classical laminated plate theory, `(N, M) = [ABD] (eps, kappa)`, with the
`6 x 6` matrix `[ABD]` of the laminate about the reference surface
(laminate between `-h/2 + d` and `h/2 + d`, offset `d`).

## Matrices

`shape_functions(side)` gives, per term, `B` (6 x 3, with the parts linear
in the values `phix`, `phiy` at the integration point), `B0` (`phix = phiy
= 0`), `G = d(phix, phiy)/dc` and `g5 = (u, v, w, phix, phiy)`. `build()`:

    kC   = BA^T [ABD] BB       constitutive stiffness, with KCNL
    kC0  = B0A^T [ABD] B0B     linear constitutive stiffness
    fint = BA^T (N, M)         internal force vector
    kG   = GA^T [N] GB         geometric stiffness, [N] = [[Nxx, Nxy], [Nxy, Nyy]]
    kM   = gA5^T [I] gB5       mass, [I] from I_n = int rho z^n dz, n = 0, 1, 2

The rotation `phiy` enters the mass matrix and the point constraints of the
rotation about `x`.

Kernels of `cylshell_clpt_sanders.pyx`: `fk0`, `fkG0`, `fkM`, integrated in
closed form with the integrals of Bardell's functions (`ab/4` the
Jacobian), and `fkAx`, `fkAy`, `fcA` of the piston theory (load `q = beta
w,x + gamma w` along `w`, see `Shell.calc_kA`). Kernels of
`cylshell_clpt_sanders_num.pyx`: `fkC_num`, `fkG_num`, `fkM_num`,
`fkAx_num`, `fkAy_num`, `calc_fint`, with the Gauss-Legendre rule
(`max(2m - 1, 7)` points along `x` by default), plus `calc_fext_follower`,
`fkCfollower_num` and `calc_kTv`.

## Selection in panels

Default model of a `Shell` with a radius `r`; explicitly
`Shell(..., r=r, model='cylshell_clpt_sanders')`.
`panels.modelDB.db['cylshell_clpt_sanders']`: `dofs = 3`, `e_num = 6`,
`requires_r = True`. The FSDT and TSDT counterparts are
`cylshell_fsdt_sanders` and `cylshell_tsdt_sanders` (`../cylshell_fsdt_sanders`,
`../cylshell_tsdt_sanders`).

## References

- Sanders, J. L., "An improved first-approximation theory for thin
  shells", NASA Technical Report R-24, Langley Research Center, 1959.
- Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
  Applied Mathematics, 21(1), 21-36, 1963. https://doi.org/10.1090/qam/147023
- Amabili, M., "Nonlinear Vibrations and Stability of Shells and Plates",
  Cambridge University Press, 2008.
  https://doi.org/10.1017/CBO9780511619694
