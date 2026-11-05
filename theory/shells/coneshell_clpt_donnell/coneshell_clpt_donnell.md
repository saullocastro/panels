# Conical shell, linear CLPT (`coneshell_clpt_donnell`)

Derivation only: `coneshell_clpt_donnell.nb` (Mathematica 9), which writes
the matrices in Fortran form to `output_expressions_mathematica/`
(`fortran_k0.txt`, `fortran_kG0.txt`, `fortran_kM.txt`). No model of
`panels` uses it: there is no `coneshell_*` entry in `panels.modelDB.db`
and no kernel in `panels/models`.

## Kinematics

Degrees of freedom of a term: `u, v, w`, each a product of approximation
functions `f(xi) g(eta)`, `xi = 2x/a - 1`, `eta = 2y/b - 1`. The geometry
enters through `sina`, `cosa` (the sine and cosine of the cone angle
`alpha`) and the radius `r`, which the notebook treats as a constant over
the domain: the factor of the area is `a b/4`. The notebook does not define
the coordinates nor the angle; the strains below are as written in it.

## Strains

Linear strains only (no von Kármán terms), as written in the notebook:

    exx = u,x
    eyy = (sina u + cosa w)/r + v,y
    gxy = u,y + v,x - sina v/r
    kxx = -w,xx
    kyy = -w,yy - sina w,x/r
    kxy = -2 w,xy + sina w,y/r

For `alpha = 0` (`sina = 0`, `cosa = 1`) these are the linear strains of
`cylshell_clpt_donnell`.

## Constitutive law

Classical laminated plate theory, `(N, M) = [ABD] (eps, kappa)`, `6 x 6`.

## Matrices

    k0  = (a b/4) B0A^T [ABD] B0B          linear constitutive stiffness
    kG0 = (a b/4) GA^T [N] GB              geometric stiffness, GA = (w,x, w,y)
    kM  = (a b/4) h mu gA5^T [m] gB5       mass, gA5 = (u, v, w, -w,x, -w,y)

with `[N] = [[Nxx, Nxy], [Nxy, Nyy]]` constant and `[m]` the matrix of a
homogeneous laminate of thickness `h`, density `mu` (the notebook's
name) and offset `d`
(`h^2/12 + d^2` on the rotations, `-d` coupling).

## Limitations

- The notebook keeps the radius `r` constant; on a cone it varies along
  the meridian.
- Only the linear matrices; no internal force vector, no numerically
  integrated kernels.
