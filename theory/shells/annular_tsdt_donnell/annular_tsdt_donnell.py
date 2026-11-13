r"""Annular-sector flat plate, TSDT with von Kármán strains in polar
coordinates (the model named after this file)

Geometry and coordinates
------------------------
As in ``annular_clpt_donnell``: the annular sector lies in a plane between
the outer radius ``R`` and the inner radius ``R - a`` (``a`` the radial
width, ``b`` the arc length of the outer edge), with

    x = R - rho,   0 <= x <= a      (from the outer edge inwards)
    y = R theta,   0 <= y <= b      (arc length along the outer edge)

the local basis ``e_x = -e_rho``, ``e_y = e_theta``, ``e_z = e_x x e_y``,
``s = R/rho`` (``d/ds_theta = s d/dy``) and ``dA = (rho/R) dx dy``. The web
of a ring frame is an application (``a`` the height of the web, ``b`` the
arc length of its outer edge, ``R`` its outer radius). The laminate spans
``d - h/2 <= z <= d + h/2`` with the offset ``d`` (``Shell.offset``).

Kinematics
----------
Reddy's third-order shear deformation theory (TSDT), degrees of freedom of
a term ``u, v, w, phix, phiy``, written as in ``plate_tsdt_donnell``, with
the cubic term about the mid-surface of the laminate (the offset
convention of ``plate_tsdt_donnell``, ``c1 = 4/(3 h^2)``):

    u(z) = u + z phix - c1 [(z - d)^3 + d^3] (phix + w,x)
    v(z) = v + z phiy - c1 [(z - d)^3 + d^3] (phiy + s w,y)
    w(z) = w

``s w,y`` being the derivative of ``w`` along the arc, so that the
transverse shear strains ``gxz = (phix + w,x) (1 - 3 c1 (z - d)^2)`` and
``gyz = (phiy + s w,y) (1 - 3 c1 (z - d)^2)`` vanish at both faces.

Strains
-------
The reference plane being flat, the in-plane strains at ``z`` are the
linear operator ``L`` of the membrane strains of ``annular_clpt_donnell``
applied to ``(u(z), v(z))``,

    L(u, v) = (u,x,  s v,y - u/rho,  s u,y + v,x + v/rho)

so that ``eps(z) = eps0 + z eps1 + [(z - d)^3 + d^3] eps3``, ``gamma(z) =
gamma0 + (z - d)^2 gamma2``, with the von Kármán terms of
``annular_clpt_donnell`` (Amabili 2008, Eqs. (1.175)-(1.177)) added to the
membrane strains, ``ds/dx = s/rho``:

    eps0   = L(u, v) + (w,x^2/2, s^2 w,y^2/2, s w,x w,y)
    eps1   = L(phix, phiy)
           = (phix,x,  s phiy,y - phix/rho,  s phix,y + phiy,x + phiy/rho)
    eps3   = -c1 L(phix + w,x, phiy + s w,y)
           = -c1 (phix,x + w,xx,
                  s phiy,y + s^2 w,yy - (phix + w,x)/rho,
                  s phix,y + phiy,x + phiy/rho + 2 s (w,xy + w,y/rho))
    gamma0 = (gyz, gxz) = (phiy + s w,y, phix + w,x)
    gamma2 = -3 c1 gamma0

in the order ``(exx, eyy, gxy, kxx, kyy, kxy, kxx3, kyy3, kxy3, gyz, gxz,
gyz2, gxz2)`` of the constitutive matrix ``Shell.ABD`` of the TSDT,
``T^T F_m T`` (see ``plate_tsdt_donnell``); the offset enters the
stiffness only through it. With ``phix = -w,x``, ``phiy = -s w,y`` the
strains reduce to those of ``annular_clpt_donnell``.

Checks (run this file):

- :func:`check_linear_strains`: the linear strains follow from the
  three-dimensional displacement field in the metric of the plane, at every
  ``z``;
- :func:`check_clpt_limit`: the membrane strains are those of
  ``annular_clpt_donnell`` and, with ``phix = -w,x``, ``phiy = -s w,y``, so
  are the changes of curvature, the other strains vanishing;
- :func:`check_rigid_body`: the six small rigid-body motions give zero
  linear strains;
- :func:`check_plate_limit`: for ``R -> infinity`` the strains are those of
  ``plate_tsdt_donnell``;
- :func:`check_tangent`: the tangent stiffness of the kernels is the
  derivative of the internal force vector;
- :func:`check_mass`: the mass integrand from the moments of the density
  about the mid-surface equals the integral through the thickness of the
  kinetic energy of the field above.

Kernels
-------
``write_pyx.py`` writes ``panels/models/<model>.pyx`` and
``panels/models/<model>_num.pyx`` from :func:`build`, with the eighteen
generalized derivatives

    d = (u, u,x, u,y, v, v,x, v,y, w, w,x, w,y, w,xx, w,yy, w,xy,
         phix, phix,x, phix,y, phiy, phiy,x, phiy,y) = [Phi] c

``[B] = d eps/d d`` (13 x 18) and ``[H(N)] = sum_r sigma_r d^2 eps_r/d
d^2``, whose only block is on ``(w,x, w,y)``:

    [H(N)] = [[Nxx, s Nxy], [s Nxy, s^2 Nyy]]

The mass integrand ``int rho J(z)^T J(z) dz``, ``(u(z), v(z), w(z)) = J(z)
d``, is written with the moments ``J_n = int rho zeta^n dzeta``, ``zeta = z
- d``, ``n = 0, 1, 2, 3, 4, 6`` (those of :meth:`.Shell.calc_kM` for the
TSDT), the offset ``d`` and ``c1``.

References
----------
Reddy, J. N., A simple higher-order theory for laminated composite plates,
J. Appl. Mech. 51 (1984) 745-752. doi:10.1115/1.3167719

Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
Cambridge University Press, 2008, Sec. 1.4. doi:10.1017/CBO9780511619694

"""
import importlib.util
import os

import numpy as np
import sympy
from sympy import Function, Matrix as M, Poly, Symbol, diff, simplify, symbols

HERE = os.path.dirname(os.path.abspath(__file__))
#: name of the model, that of this file
MODEL = os.path.splitext(os.path.basename(__file__))[0]

x, y, z, R = symbols('x, y, z, R', real=True)
rho = symbols('rho', positive=True)
#: the constant of the TSDT, 4/(3 h^2), the thickness and the offset
c1, h, d = symbols('c1, h, d', real=True)
#: the thickness coordinate from the mid-surface, `zeta = z - d`
zeta = Symbol('zeta', real=True)

(u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy,
 phix, phix_x, phix_y, phiy, phiy_x, phiy_y) = symbols(
    'u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy, '
    'phix, phix_x, phix_y, phiy, phiy_x, phiy_y', real=True)

#: generalized derivatives at a point, grouped by degree of freedom
D = (u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy,
     phix, phix_x, phix_y, phiy, phiy_x, phiy_y)
D_NAMES = tuple(sy.name for sy in D)
#: field and the derivatives along x and y of each generalized derivative
D_OF = dict(u=('u', 0, 0), u_x=('u', 1, 0), u_y=('u', 0, 1),
            v=('v', 0, 0), v_x=('v', 1, 0), v_y=('v', 0, 1),
            w=('w', 0, 0), w_x=('w', 1, 0), w_y=('w', 0, 1),
            w_xx=('w', 2, 0), w_yy=('w', 0, 2), w_xy=('w', 1, 1),
            phix=('phix', 0, 0), phix_x=('phix', 1, 0), phix_y=('phix', 0, 1),
            phiy=('phiy', 0, 0), phiy_x=('phiy', 1, 0), phiy_y=('phiy', 0, 1))
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
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
#: number of the stresses that enter ``[H]``, the membrane ones
NH = 3
#: the moments of the density about the mid-surface, `J_n = \int \rho
#: \zeta^n d\zeta`, `n = 0, ..., 6` (`J_5` never appears)
MOMENTS = symbols('J0, J1, J2, J3, J4, J5, J6', real=True)

s = R/rho


def L(f, g, f_x, f_y, g_x, g_y):
    """Linear membrane operator in polar coordinates of the fields ``f``,
    ``g`` along ``e_x``, ``e_y``, given with their derivatives"""
    return [f_x, s*g_y - f/rho, s*f_y + g_x + g/rho]


def membrane(nonlinear=True):
    """Membrane strains ``(exx, eyy, gxy)``, von Kármán"""
    k = 1 if nonlinear else 0
    lin = L(u, v, u_x, u_y, v_x, v_y)
    return [lin[0] + k*w_x**2/2, lin[1] + k*s**2*w_y**2/2,
            lin[2] + k*s*w_x*w_y]


def bending_shear():
    """The ten linear strains ``(kxx, kyy, kxy, kxx3, kyy3, kxy3, gyz, gxz,
    gyz2, gxz2)``; ``d(s w,y)/dx = s w,xy + s w,y/rho``"""
    eps1 = L(phix, phiy, phix_x, phix_y, phiy_x, phiy_y)
    psx = phix + w_x
    psy = phiy + s*w_y
    eps3 = [-c1*e for e in L(psx, psy, phix_x + w_xx, phix_y + w_xy,
                             phiy_x + s*w_xy + s*w_y/rho, phiy_y + s*w_yy)]
    gam0 = [psy, psx]
    gam2 = [-3*c1*g for g in gam0]
    return eps1 + eps3 + gam0 + gam2


def strains(nonlinear=True):
    """Generalized strains, in the order of ``Shell.ABD``"""
    return M(membrane(nonlinear) + bending_shear()).applyfunc(sympy.expand)


def displacement_map():
    """``J(z)`` such that ``(u(z), v(z), w(z)) = J(z) d``"""
    p = (z - d)**3 + d**3
    q = M([u + z*phix - c1*p*(phix + w_x),
           v + z*phiy - c1*p*(phiy + s*w_y),
           w])
    return q.jacobian(M(D))


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


def hessian_stress(sig):
    """``sum_r sigma_r d^2 eps_r/d d^2`` for the stresses ``sig``"""
    eps = strains()
    H = sympy.zeros(len(D), len(D))
    for r_, sr in enumerate(sig):
        if sr != 0:
            H += sr*sympy.hessian(eps[r_], D)
    return H


def build():
    """Expressions used by ``write_pyx.py``: the strains ``eps`` and their
    linear part ``epsL``, ``[B] = d eps/d d``, ``[H(sigma, d)]`` and
    ``[H(sigma, 0)]`` with the symbols ``STRESSES``, and the mass integrand
    ``KM``"""
    eps = strains()
    epsL = strains(nonlinear=False)
    B = eps.jacobian(M(D))
    H = hessian_stress(STRESSES)
    return dict(eps=eps, epsL=epsL, B=B, H=H,
                H0=H.subs({dd: 0 for dd in D}, simultaneous=True),
                KM=mass_integrand())


#: first lines of the docstrings of the generated modules
TITLE = ("Annular-sector flat plate, using Reddy's third-order shear "
         "deformation\ntheory (TSDT) with von Kármán strains in polar "
         "coordinates")

#: notes of the generated modules, the kinematics
KINEMATICS_NOTES = r"""The displacement field is that of Reddy's third-order shear deformation
theory [1]_ (TSDT), with the cubic term about the mid-surface of the
laminate, `c_1 = 4/(3 h^2)`, as in :mod:`panels.models.plate_tsdt_donnell`,
and the derivative `s w_{,y}` of `w` along the arc, `s = r/\rho`:

.. math::

    u(z) = u + z \phi_x - c_1 \left[(z - d)^3 + d^3\right]
    (\phi_x + w_{,x})

    v(z) = v + z \phi_y - c_1 \left[(z - d)^3 + d^3\right]
    (\phi_y + s w_{,y}) \qquad w(z) = w
"""

#: notes of the generated modules, the strains
STRAIN_NOTES = r"""With the linear membrane operator in polar coordinates
`\mathcal{L}(f, g) = \{f_{,x},\ s g_{,y} - f/\rho,\ s f_{,y} + g_{,x} +
g/\rho\}^\top`, the in-plane strains at `z` are
`\mathcal{L}(u(z), v(z))` plus the von Kármán terms of Amabili [2]_, Eqs.
(1.175)-(1.177), at the reference plane, and the transverse shear strains
`\{\gamma_{yz}, \gamma_{xz}\} = \{v_{,z} + s w_{,y},\ u_{,z} + w_{,x}\}`:

.. math::

    \boldsymbol{\varepsilon}^{(0)} = \mathcal{L}(u, v) + \left\{
    \tfrac{1}{2} w_{,x}^2,\ \tfrac{1}{2} s^2 w_{,y}^2,\ s w_{,x} w_{,y}
    \right\}^\top \qquad
    \boldsymbol{\varepsilon}^{(1)} = \mathcal{L}(\phi_x, \phi_y)

    \boldsymbol{\varepsilon}^{(3)} = -c_1 \mathcal{L}(\phi_x + w_{,x},
    \phi_y + s w_{,y}) \qquad
    \boldsymbol{\gamma}^{(0)} = \{\phi_y + s w_{,y},\ \phi_x + w_{,x}\}^\top
    \qquad \boldsymbol{\gamma}^{(2)} = -3 c_1 \boldsymbol{\gamma}^{(0)}

with `\boldsymbol{\varepsilon}(z) = \boldsymbol{\varepsilon}^{(0)} + z
\boldsymbol{\varepsilon}^{(1)} + [(z - d)^3 + d^3]
\boldsymbol{\varepsilon}^{(3)}` and `\boldsymbol{\gamma}(z) =
\boldsymbol{\gamma}^{(0)} + (z - d)^2 \boldsymbol{\gamma}^{(2)}`, in the
order of the rows of ``shell.ABD``, the constitutive matrix of the TSDT
with the offset `d`, see :meth:`.Shell._get_lam_ABD`. Since `\partial
s/\partial x = s/\rho`, `\mathcal{L}` applied to `s w_{,y}` gives `s
w_{,xy} + s w_{,y}/\rho` and `s^2 w_{,yy}`. With `\phi_x = -w_{,x}`,
`\phi_y = -s w_{,y}` the strains are those of
``annular_clpt_donnell``; for `r \to \infty` those of
``plate_tsdt_donnell``. The second derivatives of the strains with respect
to the generalized derivatives are constant and only on `(w_{,x},
w_{,y})`.
"""

#: references of the generated modules
REFERENCES = r""".. [1] Reddy, J. N., A simple higher-order theory for laminated composite
   plates, J. Appl. Mech. 51 (1984) 745-752.
   https://doi.org/10.1115/1.3167719
.. [2] Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
   Cambridge University Press, 2008, Sec. 1.4.
   https://doi.org/10.1017/CBO9780511619694
"""


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
U = Function('U', real=True)(x, y)
V = Function('V', real=True)(x, y)
W = Function('W', real=True)(x, y)
PX = Function('PX', real=True)(x, y)
PY = Function('PY', real=True)(x, y)
FUN = dict(u=U, v=V, w=W, phix=PX, phiy=PY)


def at_point(expr):
    """Replace ``d`` by the derivatives of the functions ``U, V, W, PX,
    PY``"""
    rep = {}
    for name, (fld, nx_, ny_) in D_OF.items():
        e = FUN[fld]
        if nx_:
            e = diff(e, x, nx_)
        if ny_:
            e = diff(e, y, ny_)
        rep[symbols(name, real=True)] = e
    return expr.subs(rep, simultaneous=True).subs(rho, R - x)


def basis():
    th = y/R
    e_rho = M([sympy.cos(th), sympy.sin(th), 0])
    e_th = M([-sympy.sin(th), sympy.cos(th), 0])
    e_ax = M([0, 0, 1])
    return -e_rho, e_th, -e_ax


def _load(name):
    """The derivation of the model ``name``, ``theory/shells/<name>``"""
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(HERE, '..', name, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_linear_strains():
    """Linear strains from the 3-D field in the metric of the plane

    ``P(x, y, z) = X0(x, y) + z e_z``, ``X0 = (R - x) e_rho``; the plane
    being flat, ``P,x = e_x``, ``P,y = (rho/R) e_y``, ``P,z = e_z``. The
    linear strains of ``U3 = u(z) e_x + v(z) e_y + w e_z`` are ``e_xx = U3,x
    . e_x``, ``e_yy = U3,y . e_y/(rho/R)``, ``g_xy = U3,x . e_y + U3,y .
    e_x/(rho/R)``, ``g_yz = U3,z . e_y + U3,y . e_z/(rho/R)``, ``g_xz =
    U3,z . e_x + U3,x . e_z``, equal to ``eps0 + z eps1 + p(z) eps3`` and
    ``gamma0 + (z - d)^2 gamma2`` at every ``z``.
    """
    ex, ey, ez = basis()
    sc = (R - x)/R
    p = (z - d)**3 + d**3
    uz = U + z*PX - c1*p*(PX + diff(W, x))
    vz = V + z*PY - c1*p*(PY + diff(W, y)/sc)
    U3 = uz*ex + vz*ey + W*ez
    got = [diff(U3, x).dot(ex), diff(U3, y).dot(ey)/sc,
           diff(U3, x).dot(ey) + diff(U3, y).dot(ex)/sc,
           diff(U3, z).dot(ey) + diff(U3, y).dot(ez)/sc,
           diff(U3, z).dot(ex) + diff(U3, x).dot(ez)]
    e = at_point(strains(nonlinear=False))
    ref = [e[k] + z*e[3 + k] + p*e[6 + k] for k in range(3)]
    ref += [e[9] + (z - d)**2*e[11], e[10] + (z - d)**2*e[12]]
    for g, r_ in zip(got, ref):
        assert simplify(g - r_) == 0, (g, r_)
    print('check_linear_strains: OK')


def check_clpt_limit():
    """Membrane strains of ``annular_clpt_donnell``; with ``phix = -w,x``,
    ``phiy = -s w,y`` its changes of curvature, the other strains zero"""
    clpt = _load('annular_clpt_donnell')
    names = {c.name: c for c in clpt.D}
    sub = {names[k.name]: k for k in D if k.name in names}
    sub.update({clpt.R: R, clpt.rho: rho})
    ec = clpt.strains().xreplace(sub)
    eps = strains()
    for k in range(3):
        assert simplify(eps[k] - ec[k]) == 0
    kirchhoff = at_point(eps).subs({PX: -diff(W, x), PY: -R/(R - x)*diff(W, y)}).doit()
    eck = at_point(ec)
    for k in range(3, 6):
        assert simplify(kirchhoff[k] - eck[k]) == 0, k
    for k in range(6, NE):
        assert simplify(kirchhoff[k]) == 0, k
    print('check_clpt_limit: OK')


def check_rigid_body():
    """Six small rigid-body motions ``t + o x X0`` with the rotations of the
    normal ``(phix, phiy) = (o . e_y, -o . e_x)`` give zero linear strains"""
    ex, ey, ez = basis()
    t1, t2, t3, o1, o2, o3 = symbols('t1, t2, t3, o1, o2, o3', real=True)
    X0 = (R - x)*(-ex)
    om = M([o1, o2, o3])
    Ug = M([t1, t2, t3]) + om.cross(X0)
    rep = {U: Ug.dot(ex), V: Ug.dot(ey), W: Ug.dot(ez),
           PX: om.dot(ey), PY: -om.dot(ex)}
    for e in at_point(strains(nonlinear=False)):
        assert simplify(e.subs(rep).doit()) == 0, e
    print('check_rigid_body: OK')


def check_plate_limit():
    """``R -> infinity``: the strains of ``plate_tsdt_donnell``"""
    plate = _load('plate_tsdt_donnell')
    sub = {sy: Symbol(sy.name, real=True) for sy in plate.D}
    sub[plate.c1] = c1
    ref = plate.strains().xreplace(sub)
    eps = strains().subs(rho, R - x)
    for g, e in zip(eps, ref):
        assert simplify(sympy.limit(g, R, sympy.oo) - e) == 0, (g, e)
    print('check_plate_limit: OK')


def check_tangent(seed=0):
    """``[B]^T [F] [B] + [H(sigma)]`` is the derivative of ``[B]^T sigma``"""
    rng = np.random.default_rng(seed)
    eps = strains()
    B = eps.jacobian(M(D))
    sig = F*eps
    KT = (B.T*sig).jacobian(M(D))
    H = hessian_stress(list(sig))
    rep = {sy: rng.random() for sy in F.free_symbols}
    rep.update({dd: rng.random() - 0.5 for dd in D})
    rep.update({R: 2., rho: 1.3, c1: 0.7})
    diffm = (KT - B.T*F*B - H).xreplace(rep)
    assert max(abs(float(e)) for e in diffm) < 1e-12
    print('check_tangent: OK')


def check_mass(seed=0):
    """``KM`` with the moments of a uniform density about the mid-surface
    equals ``int rho J(z)^T J(z) dz`` over ``d - h/2 <= z <= d + h/2``,
    ``c1 = 4/(3 h^2)``"""
    rng = np.random.default_rng(seed)
    hv, dv, rv = 0.3, 0.11, 1.7
    vals = {h: hv, d: dv, c1: 4/(3*hv**2), R: 2., rho: 1.3}
    vals.update({MOMENTS[n]: rv*(((hv/2)**(n + 1) - (-hv/2)**(n + 1))/(n + 1))
                 for n in range(7)})
    KM = mass_integrand().xreplace(vals)
    J = displacement_map().xreplace({k: vals[k] for k in (d, c1, R, rho)})
    for _ in range(3):
        i, j = rng.integers(0, len(D), 2)
        ref = sympy.integrate(rv*(J.T*J)[i, j], (z, dv - hv/2, dv + hv/2))
        assert abs(float(ref - KM[i, j])) < 1e-12*(1 + abs(float(ref))), (i, j)
    full = (J.T*J).applyfunc(lambda e: sympy.integrate(rv*e, (z, dv - hv/2, dv + hv/2)))
    assert max(abs(float(e)) for e in (full - KM)) < 1e-12
    print('check_mass: OK')


if __name__ == '__main__':
    check_linear_strains()
    check_clpt_limit()
    check_rigid_body()
    check_plate_limit()
    check_tangent()
    check_mass()
