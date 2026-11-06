# Web of a ring frame, CLPT with von Kármán strains in polar coordinates (`frame_web_clpt_donnell`)

Derivation and symbolic checks: `frame_web_clpt_donnell.py` (sympy, run
`python frame_web_clpt_donnell.py`, a few seconds). Kernels: `write_pyx.py`,
which writes `panels/models/frame_web_clpt_donnell_num.pyx` and
`panels/models/frame_web_clpt_donnell.pyx` (`python write_pyx.py [folder]`).

## Why

A ring frame of a stiffened cylinder has a web in the plane normal to the
axis: an annular sector of a flat plate, between the outer radius `R`, where
it meets the skin, and the inner radius `R - a`. Its strains have the terms
`1/rho` of polar coordinates, which neither `plate_clpt_*` nor
`cylshell_clpt_*` have.

## Kinematics used in the literature

- Linear Kirchhoff plate in polar coordinates: Leissa (1969, Chap. 2, the
  vibration of circular and annular plates), and the linear part of the
  strains below.
- Moderately large deflections, von Kármán: Amabili (2008), Sec. 1.4.2,
  Eqs. (1.175)-(1.180), "Strains for finite deflection of circular plates:
  von Kármán theory", and the equation of motion (1.181)-(1.182).
- Complete Green-Lagrange strains in polar coordinates: Amabili (2008),
  Sec. 1.4.1, Eqs. (1.166)-(1.168).

Donnell's shallow-shell kinematics are the von Kármán strains plus the
curvature terms of the shell (Amabili 2008, p. 42, comparing Eqs.
(1.160)-(1.161) with (1.72)-(1.73)); a flat plate has no curvature, and
the literature calls the circular-plate strains "von Kármán". `panels`
names the von Kármán flat plate `plate_clpt_donnell`; the web follows that
convention: `frame_web_clpt_donnell` has the von Kármán strains in polar
coordinates. The variants `frame_web_clpt_gl` (Eqs. (1.166)-(1.168) for the
membrane strains) and `frame_web_clpt_tl` (curvatures of the deformed
normal) would follow `plate_clpt_gl` and `plate_clpt_tl`.

## Coordinates and degrees of freedom

`Shell(model='frame_web_clpt_donnell', a=height, b=arc length of the outer
edge, r=outer radius)`:

    x = R - rho   (0 at the outer edge, a at the inner edge)
    y = R theta   (arc length along the outer edge, 0 <= y <= b)

The Ritz series is that of every model, `f_i(xi) g_j(eta)`, `xi = 2x/a -
1`, `eta = 2y/b - 1`. The local basis is `e_x = -e_rho`, `e_y = e_theta`,
`e_z = e_x x e_y`, opposite to the axis `e_rho x e_theta` of the polar
coordinates; `u, v, w` are the displacements along it. In Amabili's
notation `u1 = -u`, `u2 = v`, `u3 = -w`, `d/drho = -d/dx`, `d/dtheta = R
d/dy`. With `s = R/rho`:

    exx = u,x + w,x^2/2
    eyy = s v,y - u/rho + s^2 w,y^2/2
    gxy = s u,y + v,x + v/rho + s w,x w,y
    kxx = -w,xx
    kyy = w,x/rho - s^2 w,yy
    kxy = -2 s (w,xy + w,y/rho)

and `dA = (rho/R) dx dy`. For `R -> infinity` these are the strains of
`plate_clpt_donnell`. Displacement through the thickness: `u(z) = u - z
w,x`, `v(z) = v - z s w,y`, `w(z) = w`.

## Checks of `frame_web_clpt_donnell.py`

- `check_linear_strains`: the linear strains and curvatures above are the
  linear strains, and their derivatives in `z`, of the three-dimensional
  Kirchhoff field in the metric of the plane; the transverse shear strains
  of that field vanish.
- `check_amabili`: the strains are Amabili's Eqs. (1.175)-(1.180), and the
  Green-Lagrange strains of the deformation gradient of the mid-surface
  are Eqs. (1.166)-(1.168), after the change of variables
  (`exx = eps_r`, `eyy = eps_theta`, `gxy = -gamma_rtheta`, `kxx = -k_r`,
  `kyy = -k_theta`, `kxy = k_rtheta`).
- `check_rigid_body`: three translations and three rotations give zero
  linear strains.
- `check_plate_limit`: `R -> infinity` gives `plate_clpt_donnell`.
- `check_tangent`: `[B]^T [ABD] [B] + [H(sigma)]` is the derivative of
  `[B]^T sigma`.

## Kernels

`frame_web_clpt_donnell_num.pyx`: `fkC_num`, `fkG_num`, `fkM_num`,
`calc_fint`, `calc_kTv` and the fields `fuvw`, `fstrain` (with `fg` of `clpt_field`),
integrated with the Gauss-Legendre rule over the domain of
`Shell.integration_limits`. `frame_web_clpt_donnell.pyx`: `fk0`, `fkG0`,
`fkM` of the whole domain, which call the numerical kernels with
`shell.nx`, `shell.ny`. The coefficients `1/rho` make the integrands
rational in `x`: no rule is exact, and no closed form with the integrals
of Bardell's functions exists. The piston-theory kernels raise
`NotImplementedError`; the follower pressure is not generated.
`calc_kTv` evaluates `[Phi]^T ([B]^T [ABD] [B] + [H(N + Nbar)]) [Phi] v` at
each point, about the cost of `calc_fint`.

## Connection to a cylinder

The local axes of the web are those of the flange of `'BFxcte'`: the axes
of the base, the cylinder, rotated by 90 degrees about the common axis `y`
(`e_x` of the web = `-e_z` of the cylinder, inwards; `e_z` of the web = `e_x`
of the cylinder). At the edge `x = 0` of the web the polar angle is that of
the cylinder at `y_cyl = R theta`, so both edges share `eta` when the web
spans the same angle as the cylinder domain and `R` is the radius of the
cylinder. `'BFxcte'` then imposes `u_cyl = w_web`, `v_cyl = v_web`,
`w_cyl = -u_web` and `w_cyl,x = w_web,x` (the rotation about `e_theta`),
with the penalty constants or the null space of `MultiDomain`:

    dict(p1=cylinder, p2=web, func='BFxcte', xcte1=x_frame, xcte2=0)

No new kernel is needed.

## References

- Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
  Cambridge University Press, 2008. https://doi.org/10.1017/CBO9780511619694
- Leissa, A. W., Vibration of Plates, NASA SP-160, 1969.
