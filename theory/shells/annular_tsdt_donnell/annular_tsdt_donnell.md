# Annular-sector plate, TSDT with von Kármán strains in polar coordinates (`annular_tsdt_donnell`)

Derivation and symbolic checks: `annular_tsdt_donnell.py` (sympy, run
`python annular_tsdt_donnell.py`, about a minute). Kernels: `write_pyx.py`,
which writes `panels/models/annular_tsdt_donnell_num.pyx` and
`panels/models/annular_tsdt_donnell.pyx` (`python write_pyx.py [folder]`,
a few seconds).

## Why

`annular_clpt_donnell` with Reddy's third-order shear deformation theory
(TSDT), for thick annular plates and frame webs, and for webs connected to
the shear-deformable cylinders `cylshell_tsdt_*`.

## Coordinates and degrees of freedom

As `annular_clpt_donnell`: `x = R - rho` from the outer edge inwards,
`y = R theta` along the outer edge, local basis `e_x = -e_rho`,
`e_y = e_theta`, `e_z = e_x x e_y`, `s = R/rho`, `dA = (rho/R) dx dy`.
Degrees of freedom `u, v, w, phix, phiy` (the order of `plate_tsdt_donnell`).

## Displacement field

Reddy (1984), with the cubic term about the mid-surface of the laminate, as
in `plate_tsdt_donnell` (offset `d`, `c1 = 4/(3 h^2)`; see
`doc/source/tsdt_offset.rst` for the choice of the cubic):

    u(z) = u + z phix - c1 [(z - d)^3 + d^3] (phix + w,x)
    v(z) = v + z phiy - c1 [(z - d)^3 + d^3] (phiy + s w,y)
    w(z) = w

`s w,y` is the derivative of `w` along the arc, so the transverse shear
strains vanish at both faces.

## Strains

The plane is flat: the metric does not depend on `z` and the in-plane
strains at `z` are the linear membrane operator in polar coordinates,
`L(f, g) = (f,x, s g,y - f/rho, s f,y + g,x + g/rho)`, applied to
`(u(z), v(z))`. With `d(s w,y)/dx = s w,xy + s w,y/rho`:

    eps0   = L(u, v) + (w,x^2/2, s^2 w,y^2/2, s w,x w,y)     (von Kármán)
    eps1   = L(phix, phiy)
    eps3   = -c1 L(phix + w,x, phiy + s w,y)
    gamma0 = (phiy + s w,y, phix + w,x),   gamma2 = -3 c1 gamma0

in the order of `Shell.ABD` for the TSDT (13 strains). The offset enters
the stiffness through `Shell.ABD` (`T^T F_m T`, see
`Shell._get_lam_ABD`) and the mass through the moments about the
mid-surface `J0, ..., J6` with `d` and `c1` in the coefficients, as for
`plate_tsdt_donnell`.

## Checks (`annular_tsdt_donnell.py`)

- the linear strains are those of the 3-D displacement field in the metric
  of the plane, at every `z`;
- membrane strains equal to `annular_clpt_donnell`; with `phix = -w,x`,
  `phiy = -s w,y` the changes of curvature too, and the third-order and
  shear strains vanish;
- six small rigid-body motions are strain-free;
- `R -> infinity` gives `plate_tsdt_donnell`;
- exact tangent `B^T F B + H(sigma)`;
- mass integrand equal to the integral through the thickness.

## Kernels

`write_pyx.py` is the writer of the `annular_clpt_*` folders generalized to
5 degrees of freedom and 13 strains (18 generalized derivatives). All the
matrices are integrated with Gauss-Legendre points, including those of
the whole domain (`fk0`, `fkG0`, `fkM`); the fields `fuvw`, `fg` and the
names of the strains and stresses come from `fsdt_tsdt_field`, the strains
`fstrain` are generated. The piston theory is not implemented.

## References

- Reddy, J. N., A simple higher-order theory for laminated composite plates,
  J. Appl. Mech. 51 (1984) 745-752, doi:10.1115/1.3167719.
- Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
  Cambridge University Press, 2008, Sec. 1.4, doi:10.1017/CBO9780511619694.
