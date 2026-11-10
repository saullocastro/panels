# Annular-sector plate, TSDT with Green-Lagrange membrane strains in polar coordinates (`annular_tsdt_gl`)

Derivation and symbolic checks: `annular_tsdt_gl.py` (sympy, run
`python annular_tsdt_gl.py`, a few seconds). Kernels: `write_pyx.py`, the
same file as in `annular_tsdt_donnell`, which writes
`panels/models/annular_tsdt_gl_num.pyx` and `panels/models/annular_tsdt_gl.pyx`.

## Strains

`annular_tsdt_donnell` with the complete Green-Lagrange membrane strains of
`annular_clpt_gl` (Amabili 2008, Eqs. (1.166)-(1.168)):

    g1 = X,x   = (1 + u,x, v,x, w,x)
    g2 = s X,y = (s u,y + v/rho, 1 + s v,y - u/rho, s w,y)
    exx = (g1.g1 - 1)/2,   eyy = (g2.g2 - 1)/2,   gxy = g1.g2

The bending, third-order and transverse shear strains are the linear ones
of `annular_tsdt_donnell`, as `plate_tsdt_gl` is built from
`plate_clpt_gl` and `plate_tsdt_donnell`. The large-rotation (TL) variant
waits for the decision on the TSDT TL models.

## Checks (`annular_tsdt_gl.py`)

- linear strains equal to `annular_tsdt_donnell`;
- membrane strains equal to `annular_clpt_gl`;
- a finite rigid motion (1.1 rad about a random axis) leaves zero membrane
  strains, a small one zero linear strains;
- `R -> infinity` gives `plate_tsdt_gl`;
- exact tangent, mass integrand as in `annular_tsdt_donnell`.

## References

- Reddy, J. N., A simple higher-order theory for laminated composite plates,
  J. Appl. Mech. 51 (1984) 745-752, doi:10.1115/1.3167719.
- Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
  Cambridge University Press, 2008, Sec. 1.4, doi:10.1017/CBO9780511619694.
