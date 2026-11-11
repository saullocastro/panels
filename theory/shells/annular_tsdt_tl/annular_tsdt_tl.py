r"""Annular-sector flat plate, total-Lagrangian TSDT with the second-order
director and the deformed normal, in polar coordinates (the model named
after this file)

Geometry and coordinates
------------------------
As in ``annular_tsdt_donnell``: the annular sector lies in a plane between
the outer radius ``R`` and the inner radius ``R - a`` (``a`` the radial
width, ``b`` the arc length of the outer edge), with

    x = R - rho,   0 <= x <= a      (from the outer edge inwards)
    y = R theta,   0 <= y <= b      (arc length along the outer edge)

the local basis ``e_x = -e_rho``, ``e_y = e_theta``, ``e_z = e_x x e_y``,
the displacements ``u, v, w`` along it, the rotations ``phix``, ``phiy`` of
the normal, ``s = R/rho`` and ``dA = (rho/R) dx dy``. The laminate spans
``d - h/2 <= z <= d + h/2`` with the offset ``d`` (``Shell.offset``), which
enters through the matrix ``ABD`` of the TSDT and the moments of the
density.

Kinematics
----------
Reddy's third-order shear deformation theory (TSDT), ``c1 = 4/(3 h^2)``,
written for finite rotations as ``plate_tsdt_tl``: the deformed position of
a point is

    r(z) = X + z d + p(z) q,   p(z) = (z - d)^3 + d^3,   q = -c1 (d - N)

with the position ``X = X0 + u e_x + v e_y + w e_z`` of the deformed
reference plane, ``X0 = rho e_rho``, the second-order director of
``annular_fsdt_tl``

    d = phix e_x + phiy e_y + (1 - (phix^2 + phiy^2)/2) e_z

and the normal of the deformed reference plane, not normalized, ``N = g1 x
g2``. Reddy's vector ``(phix + w,x, phiy + s w,y, 0)`` is the linear part of
``d - N``. The local basis turns with ``theta``, ``e_x,y = -e_y/R`` and
``e_y,y = e_x/R``, so that, per unit length of the reference lines and in
the local basis, a vector ``V = (V1, V2, V3)`` has ``V,x = (V1,x, V2,x,
V3,x)`` and ``s V,y = (s V1,y + V2/rho, s V2,y - V1/rho, s V3,y)``, with

    g1 = X,x       = (1 + u,x, v,x, w,x)
    g2 = s X,y     = (s u,y + v/rho, 1 + s v,y - u/rho, s w,y)

Strains
-------
The reference plane being flat, the metric of the undeformed body does not
depend on ``z``. The strains are the coefficients of ``1``, ``z`` and
``p(z)`` of the Green-Lagrange strains ``E_ab`` of ``r(z)`` and of
``(z - d)^2`` in ``2 E_a3`` (``p,z = 3 (z - d)^2``):

    exx = (g1.g1 - 1)/2,  eyy = (g2.g2 - 1)/2,  gxy = g1.g2
    kxx = g1.d,x,  kyy = g2.(s d,y),  kxy = g1.(s d,y) + g2.d,x
    kxx3 = g1.q,x,  kyy3 = g2.(s q,y),  kxy3 = g1.(s q,y) + g2.q,x
    gyz = g2.d,  gxz = g1.d
    gyz2 = 3 g2.q = -3 c1 gyz,  gxz2 = 3 g1.q = -3 c1 gxz

the last two since ``g_a . N = 0``. Membrane, ``kappa`` and ``gamma`` are
those of ``annular_fsdt_tl``. Since ``g_a . N,b = -g_a,b . N``, ``kappa3 =
-c1 (kappa + second fundamental form of the deformed plane)``. The
transverse shear strain ``gamma (1 - 4 (z - d)^2/h^2)`` vanishes at both
faces for any deformation. Dropped, as in the linear TSDT and in
``plate_tsdt_tl``: the terms in ``z^2``, ``z p``, ``p^2`` of ``E_ab``, the
other powers of ``2 E_a3`` and ``E_33``. The linear part is that of
``annular_tsdt_donnell``, the membrane strains are those of
``annular_tsdt_gl``, and for ``R -> infinity`` the strains are those of
``plate_tsdt_tl``.

Checks (run this file):

- :func:`check_vectors`: ``g1``, ``g2``, ``d``, ``d,x``, ``s d,y``, ``N``,
  ``N,x``, ``s N,y`` against the derivatives of the Cartesian vectors;
- :func:`check_strain_measures`: the strains are the coefficients of the
  Green-Lagrange strains of ``r(z)`` above;
- :func:`check_linear_part`: the linear strains are those of
  ``annular_tsdt_donnell``, the membrane strains those of
  ``annular_tsdt_gl``;
- :func:`check_face_shear`: ``gamma + (h^2/4) gamma2`` is identically zero;
- :func:`check_rigid_rotation`: finite rigid rotations of the undeformed
  annulus and of a bent state change the strains by ``O(t^4)`` or less,
  the linear curvatures of ``annular_tsdt_gl`` by ``O(t^2)``;
- :func:`check_plate_limit`: for ``R -> infinity`` the strains are those of
  ``plate_tsdt_tl``;
- :func:`check_tangent`: ``[B]^T [F] [B] + [H(sigma, d)]`` is the
  derivative of ``[B]^T sigma``, ``[H]`` is linear in ``d`` and ``[H(sigma,
  0)]`` with membrane stresses is that of ``annular_tsdt_gl``;
- :func:`check_mass`: the mass integrand of the linear kinematics, as
  ``annular_tsdt_gl``;
- :func:`integrand_degree`: the degree of the integrands for ``modelDB``.

Kernels
-------
``write_pyx.py`` writes ``panels/models/<model>.pyx`` and
``panels/models/<model>_num.pyx`` from :func:`build`, with the twenty-four
generalized derivatives

    d = (u, u,x, u,y, u,xx, u,yy, u,xy, v, ..., v,xy, w, ..., w,xy,
         phix, phix,x, phix,y, phiy, phiy,x, phiy,y) = [Phi] c

``[B] = d eps/d d`` (13 x 24) is quadratic and ``[H(sigma, d)] = sum_r
sigma_r d^2 eps_r/d d^2`` linear in ``d``, with all the stresses.

References
----------
Reddy, J. N., A simple higher-order theory for laminated composite plates,
J. Appl. Mech. 51 (1984) 745-752. doi:10.1115/1.3167719

Schmidt, R., and Reddy, J. N., A refined small strain and moderate rotation
theory of elastic anisotropic shells, Journal of Applied Mechanics 55(3),
1988, 611-617, Eq. (16). doi:10.1115/1.3125837

Palmerio, A. F., Reddy, J. N., and Schmidt, R., On a moderate rotation
theory of laminated anisotropic shells - Part 1. Theory, International
Journal of Non-Linear Mechanics 25(6), 1990, 687-700, Eqs. (16) and (20).
doi:10.1016/0020-7462(90)90007-V

Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
Cambridge University Press, 2008, Sec. 1.4. doi:10.1017/CBO9780511619694

"""
import importlib.util
import os

import numpy as np
import sympy
from sympy import Matrix as M, Poly, Symbol, cos, diff, sin, sqrt, symbols

HERE = os.path.dirname(os.path.abspath(__file__))
#: name of the model, that of this file
MODEL = os.path.splitext(os.path.basename(__file__))[0]
#: derivation of the flat-plate model with the same kinematics, compared by
#: :func:`check_plate_limit`, relative to ``theory/shells``
PLATE_TL = ('plate_tsdt_tl', 'plate_tsdt_tl.py')

x, y, z, R = symbols('x, y, z, R', real=True)
rho = symbols('rho', positive=True)
#: the constant of the TSDT, 4/(3 h^2), the thickness and the offset
c1, h, d = symbols('c1, h, d', real=True)
#: the thickness coordinate from the mid-surface, `zeta = z - d`
zeta = Symbol('zeta', real=True)

FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
#: generalized derivatives of each field: second derivatives of u, v, w
_SUFFIXES = {'u': ('', '_x', '_y', '_xx', '_yy', '_xy'),
             'v': ('', '_x', '_y', '_xx', '_yy', '_xy'),
             'w': ('', '_x', '_y', '_xx', '_yy', '_xy'),
             'phix': ('', '_x', '_y'), 'phiy': ('', '_x', '_y')}
D_NAMES = tuple(f + sfx for f in FIELDS for sfx in _SUFFIXES[f])
#: generalized derivatives at a point, grouped by degree of freedom
D = tuple(Symbol(nm, real=True) for nm in D_NAMES)
SYM = dict(zip(D_NAMES, D))
#: field and the derivatives along x and y of each generalized derivative
D_OF = {f + sfx: (f, sfx.count('x'), sfx.count('y'))
        for f in FIELDS for sfx in _SUFFIXES[f]}
(u, u_x, u_y, u_xx, u_yy, u_xy, v, v_x, v_y, v_xx, v_yy, v_xy,
 w, w_x, w_y, w_xx, w_yy, w_xy, phix, phix_x, phix_y, phiy, phiy_x,
 phiy_y) = D
DOF = 5
NE = 13

A11, A12, A16, A22, A26, A66 = symbols('A11, A12, A16, A22, A26, A66')
B11, B12, B16, B22, B26, B66 = symbols('B11, B12, B16, B22, B26, B66')
D11, D12, D16, D22, D26, D66 = symbols('D11, D12, D16, D22, D26, D66')
A44, A45, A55 = symbols('A44, A45, A55')
D44, D45, D55 = symbols('D44, D45, D55')
E11, E12, E16, E22, E26, E66 = symbols('E11, E12, E16, E22, E26, E66')
F11, F12, F16, F22, F26, F66 = symbols('F11, F12, F16, F22, F26, F66')
H11, H12, H16, H22, H26, H66 = symbols('H11, H12, H16, H22, H26, H66')
F44, F45, F55 = symbols('F44, F45, F55')
#: constitutive matrix of the TSDT, ``[[A, B, E, 0, 0], [B, D, F, 0, 0], [E, F,
#: H, 0, 0], [0, 0, 0, Ats, Dts], [0, 0, 0, Dts, Fts]]``
F = M([[A11, A12, A16, B11, B12, B16, E11, E12, E16, 0, 0, 0, 0],
       [A12, A22, A26, B12, B22, B26, E12, E22, E26, 0, 0, 0, 0],
       [A16, A26, A66, B16, B26, B66, E16, E26, E66, 0, 0, 0, 0],
       [B11, B12, B16, D11, D12, D16, F11, F12, F16, 0, 0, 0, 0],
       [B12, B22, B26, D12, D22, D26, F12, F22, F26, 0, 0, 0, 0],
       [B16, B26, B66, D16, D26, D66, F16, F26, F66, 0, 0, 0, 0],
       [E11, E12, E16, F11, F12, F16, H11, H12, H16, 0, 0, 0, 0],
       [E12, E22, E26, F12, F22, F26, H12, H22, H26, 0, 0, 0, 0],
       [E16, E26, E66, F16, F26, F66, H16, H26, H66, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, A44, A45, D44, D45],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, A45, A55, D45, D55],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, D44, D45, F44, F45],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, D45, D55, F45, F55]])
#: generalized stresses, conjugate to the strains
STRESSES = symbols('Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Pxx, Pyy, Pxy, Qy, Qx, Ry, Rx')
#: number of the stresses that enter ``[H]``: all of them
NH = NE
#: the moments of the density about the mid-surface, `J_n = \int \rho
#: \zeta^n d\zeta`, `n = 0, ..., 6` (`J_5` never appears)
MOMENTS = symbols('J0, J1, J2, J3, J4, J5, J6', real=True)

s = R/rho


def _shift(sym, ax):
    """The generalized derivative of ``sym`` along ``ax`` (``'x'`` or
    ``'y'``)"""
    fld, ox, oy = D_OF[sym.name]
    ox, oy = (ox + 1, oy) if ax == 'x' else (ox, oy + 1)
    sfx = '_' + 'x'*ox + 'y'*oy if ox + oy else ''
    if fld + sfx not in SYM:
        raise ValueError('%s,%s is not a generalized derivative' % (sym, ax))
    return SYM[fld + sfx]


def dx(e):
    """Derivative along ``x`` of an expression of ``d`` and ``rho``,
    ``rho,x = -1``"""
    e = sympy.sympify(e)
    out = -diff(e, rho)
    for sym in e.free_symbols & set(D):
        out += diff(e, sym)*_shift(sym, 'x')
    return out


def dy(e):
    """Derivative along ``y`` of an expression of ``d`` and ``rho``"""
    e = sympy.sympify(e)
    out = sympy.S(0)
    for sym in e.free_symbols & set(D):
        out += diff(e, sym)*_shift(sym, 'y')
    return out


def local_x(V):
    """``V,x`` of the vector of local components ``V``"""
    return V.applyfunc(dx)


def local_sy(V):
    """``s V,y`` of the vector of local components ``V``, the basis turning
    with ``theta``"""
    return M([s*dy(V[0]) + V[1]/rho, s*dy(V[1]) - V[0]/rho, s*dy(V[2])])


def tangent_vectors():
    """``g1 = X,x`` and ``g2 = s X,y`` in the local basis"""
    g1 = M([1 + u_x, v_x, w_x])
    g2 = M([s*u_y + v/rho, 1 + s*v_y - u/rho, s*w_y])
    return g1, g2


def director():
    """Second-order director ``d``, ``d,x`` and ``s d,y`` in the local
    basis"""
    dd = M([phix, phiy, 1 - (phix**2 + phiy**2)/2])
    return dd, local_x(dd), local_sy(dd)


def normal():
    """``N = g1 x g2``, ``N,x`` and ``s N,y`` in the local basis"""
    g1, g2 = tangent_vectors()
    N = g1.cross(g2).applyfunc(sympy.expand)
    return N, local_x(N), local_sy(N)


def third_vector():
    """``q = -c1 (d - N)``, ``q,x`` and ``s q,y``"""
    dd, d_x, sd_y = director()
    N, N_x, sN_y = normal()
    return -c1*(dd - N), -c1*(d_x - N_x), -c1*(sd_y - sN_y)


def linear_part(e):
    """Terms of degree one in ``d`` of the polynomial ``e``"""
    e = sympy.expand(e)
    return sum((tt for tt in sympy.Add.make_args(e)
                if sum(sympy.degree(tt, dd) for dd in D) == 1), sympy.S(0))


def strains(nonlinear=True):
    """Generalized strains, in the order of ``Shell.ABD``"""
    g1, g2 = tangent_vectors()
    dd, d_x, sd_y = director()
    q, q_x, sq_y = third_vector()
    eps = [(g1.dot(g1) - 1)/2, (g2.dot(g2) - 1)/2, g1.dot(g2),
           g1.dot(d_x), g2.dot(sd_y), g1.dot(sd_y) + g2.dot(d_x),
           g1.dot(q_x), g2.dot(sq_y), g1.dot(sq_y) + g2.dot(q_x),
           g2.dot(dd), g1.dot(dd), 3*g2.dot(q), 3*g1.dot(q)]
    eps = [sympy.expand(e) for e in eps]
    if not nonlinear:
        eps = [linear_part(e) for e in eps]
    return M(eps)


def displacement_map():
    """``J(z)`` such that ``(u(z), v(z), w(z)) = J(z) d``, the linear
    kinematics"""
    p = (z - d)**3 + d**3
    qv = M([u + z*phix - c1*p*(phix + w_x),
            v + z*phiy - c1*p*(phiy + s*w_y),
            w])
    return qv.jacobian(M(D))


def thickness_moments(e):
    r"""Integral through the thickness of ``rho*e``, ``e`` a polynomial in
    `z`, with `z = \zeta + d` and `\int \rho \zeta^n d\zeta` replaced by the
    moment `J_n` about the mid-surface"""
    p = Poly(sympy.expand(e.subs(z, zeta + d)), zeta)
    return sum(coef*MOMENTS[n] for (n, ), coef in p.terms())


def mass_integrand():
    """``int rho J(z)^T J(z) dz`` in the moments ``J_n``"""
    J = displacement_map()
    return (J.T*J).applyfunc(thickness_moments)


def hessian_stress(sig, eps=None):
    """``sum_r sigma_r d^2 eps_r/d d^2`` for the stresses ``sig``"""
    eps = strains() if eps is None else eps
    H = sympy.zeros(len(D), len(D))
    for r_, sr in enumerate(sig):
        if sr != 0:
            H += sr*sympy.hessian(eps[r_], D)
    return H.applyfunc(sympy.expand)


def build():
    """Expressions used by ``write_pyx.py``: the strains ``eps`` and their
    linear part ``epsL``, ``[B] = d eps/d d``, ``[H(sigma, d)]`` and
    ``[H(sigma, 0)]`` with the symbols ``STRESSES``, and the mass integrand
    ``KM``"""
    eps = strains()
    epsL = M([linear_part(e) for e in eps])
    B = eps.jacobian(M(D))
    H = hessian_stress(STRESSES, eps)
    return dict(eps=eps, epsL=epsL, B=B, H=H,
                H0=H.subs({dd: 0 for dd in D}, simultaneous=True),
                KM=mass_integrand())


#: first lines of the docstrings of the generated modules
TITLE = ("Annular-sector flat plate, using Reddy's third-order shear "
         "deformation\ntheory (TSDT), total-Lagrangian, in polar coordinates")

#: notes of the generated modules, the kinematics
KINEMATICS_NOTES = r"""The displacement field of Reddy's third-order shear deformation theory
[1]_ (TSDT), with the cubic term about the mid-surface of the laminate,
`c_1 = 4/(3 h^2)`, as in ``annular_tsdt_donnell``, is written for finite
rotations as in ``plate_tsdt_tl``: the deformed position of a point is

.. math::

    \boldsymbol{r}(z) = \boldsymbol{X} + z \boldsymbol{d} + p(z)
    \boldsymbol{q}, \qquad p(z) = (z - d)^3 + d^3, \qquad \boldsymbol{q} =
    -c_1 (\boldsymbol{d} - \boldsymbol{N})

with the position `\boldsymbol{X}` of the deformed reference plane, the
second-order director of Schmidt and Reddy [2]_, Eq. (16), and Palmerio,
Reddy and Schmidt [3]_, Eqs. (16) and (20), `\boldsymbol{d} = \phi_x
\boldsymbol{e}_x + \phi_y \boldsymbol{e}_y + (1 - (\phi_x^2 + \phi_y^2)/2)
\boldsymbol{e}_z`, as in ``annular_fsdt_tl``, and the normal of the deformed
reference plane, not normalized, `\boldsymbol{N} = \boldsymbol{g}_1 \times
\boldsymbol{g}_2`. The linear part of `\boldsymbol{d} - \boldsymbol{N}` is
Reddy's vector `(\phi_x + w_{,x}, \phi_y + s w_{,y}, \cdot)`, `s =
r/\rho`.
"""

#: notes of the generated modules, the strains
STRAIN_NOTES = r"""With, per unit length of the reference lines and in the local basis,
which turns with `\theta`, the tangent vectors `\boldsymbol{g}_1 =
\boldsymbol{X}_{,x} = \{1 + u_{,x}, v_{,x}, w_{,x}\}^\top` and
`\boldsymbol{g}_2 = s \boldsymbol{X}_{,y} = \{s u_{,y} + v/\rho, 1 + s
v_{,y} - u/\rho, s w_{,y}\}^\top`, and, for a vector `\boldsymbol{V} =
\{V_1, V_2, V_3\}^\top`, `\boldsymbol{V}_{,x} = \{V_{1,x}, V_{2,x},
V_{3,x}\}^\top` and `s \boldsymbol{V}_{,y} = \{s V_{1,y} + V_2/\rho, s
V_{2,y} - V_1/\rho, s V_{3,y}\}^\top`, the strains are the coefficients of
`1`, `z`, `p(z)` of the Green-Lagrange strains of `\boldsymbol{r}(z)`, and
of `(z - d)^2` in the transverse shear strains:

.. math::

    \varepsilon_{xx} = \frac{1}{2} \left(\boldsymbol{g}_1 \cdot
    \boldsymbol{g}_1 - 1\right) \qquad
    \varepsilon_{yy} = \frac{1}{2} \left(\boldsymbol{g}_2 \cdot
    \boldsymbol{g}_2 - 1\right) \qquad
    \gamma_{xy} = \boldsymbol{g}_1 \cdot \boldsymbol{g}_2

    \kappa_{xx} = \boldsymbol{g}_1 \cdot \boldsymbol{d}_{,x} \qquad
    \kappa_{yy} = \boldsymbol{g}_2 \cdot s \boldsymbol{d}_{,y} \qquad
    \kappa_{xy} = \boldsymbol{g}_1 \cdot s \boldsymbol{d}_{,y} +
    \boldsymbol{g}_2 \cdot \boldsymbol{d}_{,x}

    \kappa^{(3)}_{xx} = \boldsymbol{g}_1 \cdot \boldsymbol{q}_{,x} \qquad
    \kappa^{(3)}_{yy} = \boldsymbol{g}_2 \cdot s \boldsymbol{q}_{,y} \qquad
    \kappa^{(3)}_{xy} = \boldsymbol{g}_1 \cdot s \boldsymbol{q}_{,y} +
    \boldsymbol{g}_2 \cdot \boldsymbol{q}_{,x}

    \gamma_{yz} = \boldsymbol{g}_2 \cdot \boldsymbol{d} \qquad
    \gamma_{xz} = \boldsymbol{g}_1 \cdot \boldsymbol{d} \qquad
    \gamma^{(2)}_{yz} = -3 c_1 \gamma_{yz} \qquad
    \gamma^{(2)}_{xz} = -3 c_1 \gamma_{xz}

the last two because `\boldsymbol{g}_\alpha \cdot \boldsymbol{N} = 0`, with
`\boldsymbol{\varepsilon}(z) = \boldsymbol{\varepsilon}^{(0)} + z
\boldsymbol{\kappa} + p(z) \boldsymbol{\kappa}^{(3)}` and
`\boldsymbol{\gamma}(z) = \boldsymbol{\gamma} + (z - d)^2
\boldsymbol{\gamma}^{(2)}`, in the order of the rows of ``shell.ABD``, the
constitutive matrix of the TSDT with the offset `d`, see
:meth:`.Shell._get_lam_ABD`. The transverse shear strains `\boldsymbol{\gamma}
(1 - 4 (z - d)^2/h^2)` vanish at both faces for any deformation. Dropped,
as in the linear TSDT: the terms in `z^2`, `z p`, `p^2` of the in-plane
strains, the other powers of the transverse shear strains and the
transverse normal strain. The membrane strains are those of
``annular_tsdt_gl``, the Green-Lagrange strains of Amabili [4]_, Eqs.
(1.166)-(1.168); `\boldsymbol{\kappa}` and `\boldsymbol{\gamma}` are those
of ``annular_fsdt_tl``; the linear part is that of
``annular_tsdt_donnell``, and for `r \to \infty` the strains are those of
``plate_tsdt_tl``. A finite rigid rotation `t` changes the strains by
`O(t^4)` or less, from the second-order director. The second derivatives
of the strains with respect to the generalized derivatives are linear in
them. The mass matrix is that of the linear kinematics, as in
``annular_tsdt_gl``.
"""

#: references of the generated modules
REFERENCES = r""".. [1] Reddy, J. N., A simple higher-order theory for laminated composite
   plates, J. Appl. Mech. 51 (1984) 745-752.
   https://doi.org/10.1115/1.3167719
.. [2] Schmidt, R., and Reddy, J. N., A refined small strain and moderate
   rotation theory of elastic anisotropic shells, Journal of Applied
   Mechanics, Vol. 55, No. 3, 1988, pp. 611-617, Eq. (16).
   https://doi.org/10.1115/1.3125837
.. [3] Palmerio, A. F., Reddy, J. N., and Schmidt, R., On a moderate
   rotation theory of laminated anisotropic shells - Part 1. Theory,
   International Journal of Non-Linear Mechanics, Vol. 25, No. 6, 1990,
   pp. 687-700, Eqs. (16) and (20).
   https://doi.org/10.1016/0020-7462(90)90007-V
.. [4] Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
   Cambridge University Press, 2008, Sec. 1.4.
   https://doi.org/10.1017/CBO9780511619694
"""


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def _load(name, *path):
    """The derivation of the model ``name``, ``theory/shells/<path>``"""
    path = path or (name, name + '.py')
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, '..', *path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def basis():
    """``e_x, e_y, e_z`` at ``theta = y/R``"""
    th_ = y/R
    e_rho = M([cos(th_), sin(th_), 0])
    e_th = M([-sin(th_), cos(th_), 0])
    e_ax = M([0, 0, 1])
    return -e_rho, e_th, -e_ax


def values(funcs, xv, yv, Rv, c1v=1.):
    """Numerical generalized derivatives at ``(xv, yv)`` of the fields
    ``funcs`` (expressions of ``x``, ``y`` and ``R``), with ``R``, ``rho``
    and ``c1``"""
    vals = {}
    for name, (fld, nx_, ny_) in D_OF.items():
        e = funcs[fld].subs(R, Rv)
        if nx_:
            e = diff(e, x, nx_)
        if ny_:
            e = diff(e, y, ny_)
        vals[SYM[name]] = float(e.subs({x: xv, y: yv}))
    vals.update({R: Rv, rho: Rv - xv, c1: c1v})
    return vals


def _cartesian(funcs):
    """Cartesian position ``X``, director ``d``, normal ``N = X,x x s X,y``
    and ``q`` of the fields ``funcs``"""
    ex, ey, ez = basis()
    X = (R - x)*(-ex) + funcs['u']*ex + funcs['v']*ey + funcs['w']*ez
    px, py = funcs['phix'], funcs['phiy']
    dd = px*ex + py*ey + (1 - (px**2 + py**2)/2)*ez
    N = diff(X, x).cross(R/(R - x)*diff(X, y))
    return X, dd, N, -c1*(dd - N)


def _random_fields(rng):
    """Random cubic polynomials in ``x``, ``y`` for the five fields"""
    mons = [x**i*y**j for i in range(4) for j in range(4 - i)]
    return {f: sum(float(rng.uniform(-0.3, 0.3))*mn for mn in mons)
            for f in FIELDS}


def _ev(e, vals):
    return np.array([float(sympy.sympify(t).xreplace(vals)) for t in e])


def check_vectors(seed=0):
    """``g1``, ``g2``, ``d``, ``d,x``, ``s d,y``, ``N``, ``N,x``, ``s N,y``
    in the local basis, against the Cartesian derivatives, for random cubic
    fields at random points"""
    rng = np.random.default_rng(seed)
    Rv = 2.
    g1, g2 = tangent_vectors()
    dd, d_x, sd_y = director()
    N, N_x, sN_y = normal()
    for _ in range(3):
        funcs = _random_fields(rng)
        xv, yv = rng.uniform(0., 0.8), rng.uniform(0., 1.)
        ex, ey, ez = [e.subs(R, Rv) for e in basis()]
        X, dc, Nc, _ = [e.subs(R, Rv) for e in _cartesian(funcs)]
        sc = Rv/(Rv - x)
        pt = {x: xv, y: yv}

        def loc(vec):
            return np.array([float(vec.dot(e).subs(pt)) for e in (ex, ey, ez)])
        vals = values(funcs, xv, yv, Rv)
        pairs = ((loc(diff(X, x)), g1), (loc(sc*diff(X, y)), g2),
                 (loc(dc), dd), (loc(diff(dc, x)), d_x),
                 (loc(sc*diff(dc, y)), sd_y), (loc(Nc), N),
                 (loc(diff(Nc, x)), N_x), (loc(sc*diff(Nc, y)), sN_y))
        for ref, e in pairs:
            assert np.allclose(ref, _ev(e, vals), rtol=1e-12, atol=1e-12), (ref, e)
    print('check_vectors: OK')


def check_strain_measures(seed=0):
    """The strains are the coefficients of the Green-Lagrange strains of
    ``r(z) = X + z d + p q``, in the metric of the plane, for random cubic
    fields

    The reference point is ``X0 + z e_z``: ``E_xx = (P,x.P,x - 1)/2``,
    ``E_yy = (s^2 P,y.P,y - 1)/2``, ``2 E_xy = s P,x.P,y``, ``2 E_y3 = s
    P,y.P,z`` and ``2 E_x3 = P,x.P,z``. In ``E_ab``, ``p`` is a symbol: the
    coefficients of ``1``, ``z``, ``p``; in ``2 E_a3``, ``d = 0`` and ``p =
    z^3``: the coefficients of ``z^0`` and ``z^2``.
    """
    rng = np.random.default_rng(seed)
    Rv = 2.
    pz = Symbol('pz')
    eps = strains()
    for _ in range(2):
        funcs = _random_fields(rng)
        xv, yv = rng.uniform(0., 0.8), rng.uniform(0., 1.)
        X, dc, Nc, qc = [e.subs({R: Rv, c1: 1.}) for e in _cartesian(funcs)]
        sc = Rv/(Rv - x)
        pt = {x: xv, y: yv}
        e_num = _ev(eps, values(funcs, xv, yv, Rv))
        P = X + z*dc + pz*qc
        Px, Py = diff(P, x).subs(pt), (sc*diff(P, y)).subs(pt)
        E = [sympy.expand((Px.dot(Px) - 1)/2), sympy.expand((Py.dot(Py) - 1)/2),
             sympy.expand(Px.dot(Py))]
        for k in range(3):
            Ek = Poly(E[k], z, pz)
            for (mz, mp), r in (((0, 0), k), ((1, 0), 3 + k), ((0, 1), 6 + k)):
                assert abs(float(Ek.coeff_monomial(z**mz*pz**mp)) - e_num[r]) < 1e-11, (k, r)
        P3 = X + z*dc + z**3*qc
        Pz = diff(P3, z).subs(pt)
        S = [sympy.expand((sc*diff(P3, y)).subs(pt).dot(Pz)),
             sympy.expand(diff(P3, x).subs(pt).dot(Pz))]
        for k in range(2):
            Sk = Poly(S[k], z)
            assert abs(float(Sk.coeff_monomial(1)) - e_num[9 + k]) < 1e-11
            assert abs(float(Sk.coeff_monomial(z**2)) - e_num[11 + k]) < 1e-11
    print('check_strain_measures: OK')


def check_linear_part():
    """Linear strains of ``annular_tsdt_donnell``; membrane strains of
    ``annular_tsdt_gl``"""
    lin = _load('annular_tsdt_donnell').strains(nonlinear=False)
    for g, e in zip(strains(nonlinear=False), lin):
        assert sympy.simplify(g - e) == 0, (g, e)
    gl = _load('annular_tsdt_gl').strains()
    eps = strains()
    for k in range(3):
        assert sympy.simplify(eps[k] - gl[k]) == 0, k
    print('check_linear_part: OK')


def check_face_shear():
    """``gamma + (h^2/4) gamma2``, ``h^2 = 4/(3 c1)``, identically zero"""
    eps = strains()
    for k in range(2):
        assert sympy.expand(eps[9 + k] + eps[11 + k]/(3*c1)) == 0
    print('check_face_shear: OK')


def rotation(axis, angle):
    """Rotation matrix of ``angle`` about the unit vector ``axis``"""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*K @ K


def _state(Xc, nc, xv, yv, Rv):
    """Generalized derivatives at ``(xv, yv)`` of the position ``Xc`` and of
    the director whose components along ``e_x``, ``e_y`` are those of the
    vector ``nc``, both Cartesian expressions of ``x``, ``y``"""
    ex, ey, ez = basis()
    Ug = Xc - (R - x)*(-ex)
    funcs = dict(u=Ug.dot(ex), v=Ug.dot(ey), w=Ug.dot(ez), phix=nc.dot(ex),
                 phiy=nc.dot(ey))
    return values(funcs, xv, yv, Rv)


GROUPS = {'membrane': (0, 1, 2), 'kappa': (3, 4, 5), 'kappa3': (6, 7, 8),
          'gamma': (9, 10), 'gamma2': (11, 12)}


def rigid_rotation_residuals(seed=0):
    r"""Largest strain of each group of ``GROUPS``, ``c1 = 1``, for rigid
    rotations `t = 0.2, 0.1` of the undeformed annulus about three random
    axes, and largest change of each group when the bent state `X_0 + w_0
    e_z`, `w_0 = k_0 (x - x_p)^2/2 + k_0 (y - y_p)^2/4`, `k_0 = 0.3`, with
    the director along its unit normal, is rotated about three random
    in-plane axes; also the change of the linear curvatures and
    third-order strains of ``annular_tsdt_gl`` (group ``'gl'``)"""
    rng = np.random.default_rng(seed)
    Rv, xv, yv = 2., 0.7, 0.9
    eps = strains()
    lin = M([linear_part(e) for e in eps[3:9]])
    ex, ey, ez = basis()
    X0 = (R - x)*(-ex)
    out = {'undeformed': {g: [0., 0.] for g in GROUPS},
           'bent': {g: [0., 0.] for g in list(GROUPS) + ['gl']}}
    for _ in range(3):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        for i, t in enumerate((0.2, 0.1)):
            Q = M(rotation(axis, t))
            e = _ev(eps, _state(Q*X0, Q*ez, xv, yv, Rv))
            for g, idx in GROUPS.items():
                out['undeformed'][g][i] = max(out['undeformed'][g][i],
                                              np.abs(e[list(idx)]).max())
    k0 = 0.3
    w0 = k0*(x - xv)**2/2 + k0*(y - yv)**2/4
    Xb = X0 + w0*ez
    nb = diff(Xb, x).cross(diff(Xb, y))
    nb = nb/sqrt(nb.dot(nb))
    st0 = _state(Xb, nb, xv, yv, Rv)
    base, base_gl = _ev(eps, st0), _ev(lin, st0)
    for _ in range(3):
        axis = np.array([*rng.standard_normal(2), 0.])
        axis /= np.linalg.norm(axis)
        for i, t in enumerate((0.2, 0.1)):
            Q = M(rotation(axis, t))
            st = _state(Q*Xb, Q*nb, xv, yv, Rv)
            e = _ev(eps, st) - base
            for g, idx in GROUPS.items():
                out['bent'][g][i] = max(out['bent'][g][i],
                                        np.abs(e[list(idx)]).max())
            out['bent']['gl'][i] = max(out['bent']['gl'][i],
                                       np.abs(_ev(lin, st) - base_gl).max())
    return out


def _order(v):
    return np.log2(v[0]/v[1]) if v[1] > 1e-14 else 99.


def check_rigid_rotation():
    r"""Undeformed annulus: membrane strains zero, the others `O(t^4)` or
    smaller (fitted order between `t = 0.2` and `0.1` above 3.8); bent
    state: every group changes by `O(t^4)` or less, the linear curvatures
    of ``annular_tsdt_gl`` by `O(t^2)` (order between 1.8 and 2.2), at
    least 20 times more at `t = 0.1`"""
    res = rigid_rotation_residuals()
    und, bent = res['undeformed'], res['bent']
    assert und['membrane'][0] < 1e-12
    for g in ('kappa', 'kappa3', 'gamma', 'gamma2'):
        assert und[g][0] < 1e-12 or _order(und[g]) > 3.8, (g, und[g])
    for g in GROUPS:
        assert bent[g][0] < 1e-12 or _order(bent[g]) > 3.8, (g, bent[g])
    assert 1.8 < _order(bent['gl']) < 2.2, bent['gl']
    assert max(bent[g][1] for g in GROUPS) < 0.05*bent['gl'][1]
    print('check_rigid_rotation: OK')
    for case, groups in res.items():
        print('   ', case, ', '.join('%s: %.2e/%.2e (order %.2f)'
                                     % (g, v[0], v[1], _order(v))
                                     for g, v in groups.items()))
    return res


def check_plate_limit():
    """``R -> infinity``: the strains of ``plate_tsdt_tl`` (the derivation
    at ``PLATE_TL``); the strains are polynomials in ``s`` and ``1/rho``,
    whose limits are ``1`` and ``0``"""
    plate = _load('plate_tsdt_tl', *PLATE_TL)
    sub = {getattr(plate, sy.name): SYM[sy.name] for sy in plate.D}
    sub[plate.c1] = c1
    ref = plate.strains().xreplace(sub)
    S_, IR = symbols('S_, IR')
    eps = strains().subs(R, S_*rho).subs(rho, 1/IR).applyfunc(sympy.expand)
    lim = eps.subs({S_: 1, IR: 0})
    for g, e in zip(lim, ref):
        assert sympy.expand(g - e) == 0, (g, e)
    print('check_plate_limit: OK')


def check_tangent(seed=0):
    """``[B]^T [F] [B] + [H(sigma, d)]`` against central differences of
    ``[B]^T sigma``, ``sigma = F eps``; ``[H]`` linear in ``d``; ``[H(sigma,
    0)]`` with membrane stresses equal to that of ``annular_tsdt_gl``"""
    rng = np.random.default_rng(seed)
    m = build()
    eps, B, H, H0 = m['eps'], m['B'], m['H'], m['H0']
    tt = Symbol('tt')
    Hs = H.subs({dd: tt*dd for dd in D}, simultaneous=True).applyfunc(sympy.expand)
    assert all(sympy.degree(e, tt) <= 1 for e in Hs if e != 0)
    rep = {sy: rng.random() for sy in F.free_symbols}
    rep.update({R: 2., rho: 1.3, c1: 0.7})
    Fn = np.array(F.xreplace(rep).tolist(), dtype=float)
    fe = sympy.lambdify(D, eps.xreplace(rep), 'numpy')
    fB = sympy.lambdify(D, B.xreplace(rep), 'numpy')
    fH = sympy.lambdify(D + STRESSES, H.xreplace(rep), 'numpy')
    dv = rng.uniform(-0.4, 0.4, len(D))
    sig = Fn @ np.array(fe(*dv), dtype=float).ravel()
    Bv = np.array(fB(*dv), dtype=float)
    KT = Bv.T @ Fn @ Bv + np.array(fH(*dv, *sig), dtype=float)

    def g(dd_):
        Bd = np.array(fB(*dd_), dtype=float)
        return Bd.T @ (Fn @ np.array(fe(*dd_), dtype=float).ravel())
    step = 1e-6
    for k in range(len(D)):
        dd_ = np.zeros(len(D))
        dd_[k] = step
        fd = (g(dv + dd_) - g(dv - dd_))/(2*step)
        assert np.allclose(fd, KT[:, k], rtol=1e-6, atol=1e-6), k
    gl = _load('annular_tsdt_gl')
    Hgl = gl.build()['H']
    idx = [D_NAMES.index(sy.name) for sy in gl.D]
    memb = {sg: 0 for sg in STRESSES[3:]}
    H0m = H0.xreplace(memb)
    subgl = {getattr(gl, sg.name) if hasattr(gl, sg.name) else sg: sg
             for sg in STRESSES[:3]}
    for i, p in enumerate(idx):
        for j, q in enumerate(idx):
            ref = Hgl[i, j].xreplace(subgl)
            assert sympy.simplify(H0m[p, q] - ref) == 0, (p, q)
    rest = [p for p in range(len(D)) if p not in idx]
    assert all(H0m[p, q] == 0 for p in rest for q in range(len(D)))
    print('check_tangent: OK')


def check_mass(seed=0):
    """``KM`` with the moments of a uniform density about the mid-surface
    equals ``int rho J(z)^T J(z) dz`` over ``d - h/2 <= z <= d + h/2``,
    ``c1 = 4/(3 h^2)``, and the integrand of ``annular_tsdt_gl``"""
    hv, dv, rv = 0.3, 0.11, 1.7
    vals = {h: hv, d: dv, c1: 4/(3*hv**2), R: 2., rho: 1.3}
    vals.update({MOMENTS[n]: rv*(((hv/2)**(n + 1) - (-hv/2)**(n + 1))/(n + 1))
                 for n in range(7)})
    KM = mass_integrand().xreplace(vals)
    J = displacement_map().xreplace({k: vals[k] for k in (d, c1, R, rho)})
    full = (J.T*J).applyfunc(lambda e: sympy.integrate(rv*e, (z, dv - hv/2, dv + hv/2)))
    assert max(abs(float(e)) for e in (full - KM)) < 1e-12
    gl = _load('annular_tsdt_gl')
    KMgl = gl.mass_integrand()
    idx = [D_NAMES.index(sy.name) for sy in gl.D]
    KMt = mass_integrand()
    for i, p in enumerate(idx):
        for j, q in enumerate(idx):
            assert sympy.simplify(KMt[p, q] - KMgl[i, j]) == 0
    print('check_mass: OK')


def integrand_degree():
    r"""``(k, j)`` such that the integrands of ``K_T`` are polynomials of
    degree at most ``k p + j`` along `x` and along `y` in the generalized
    derivatives, for approximation functions of degree ``p``, the
    coefficients in `\rho` aside: a monomial of degree ``n`` in ``d``
    between the two functions has ``k = n + 2``, and each derivative along
    the direction lowers ``j`` by one"""
    m = build()
    eps, B, H = m['eps'], m['B'], m['H']
    best = None
    for ax in (1, 2):
        deg = {sy: -D_OF[sy.name][ax] for sy in D}

        def top(e):
            e = sympy.expand(e)
            if e == 0:
                return None
            mons = Poly(e, *D).monoms()
            return max((sum(mo), sum(n*deg[sy] for n, sy in zip(mo, D)))
                       for mo in mons)

        def add(*ps):
            return None if None in ps else tuple(map(sum, zip(*ps)))
        tb = [[top(B[r, p]) for p in range(len(D))] for r in range(NE)]
        te = [top(e) for e in eps]
        tsig = [max([te[c] for c in range(NE) if F[r, c] != 0
                     and te[c] is not None], default=None) for r in range(NE)]
        hs = [hessian_stress([0]*r_ + [1] + [0]*(NE - r_ - 1), eps)
              for r_ in range(NE)]
        for p in range(len(D)):
            for q in range(len(D)):
                cand = []
                for r_ in range(NE):
                    for c in range(NE):
                        if F[r_, c] != 0:
                            cand.append(add(tb[r_][p], tb[c][q]))
                    cand.append(add(tsig[r_], top(hs[r_][p, q])))
                cand = [cc for cc in cand if cc is not None]
                if not cand:
                    continue
                k, j = max(cand)
                kj = (k + 2, j + deg[D[p]] + deg[D[q]])
                best = kj if best is None else max(best, kj)
    return best


if __name__ == '__main__':
    check_vectors()
    check_strain_measures()
    check_linear_part()
    check_face_shear()
    check_rigid_rotation()
    check_plate_limit()
    check_tangent()
    check_mass()
    print('integrand_degree:', integrand_degree())
