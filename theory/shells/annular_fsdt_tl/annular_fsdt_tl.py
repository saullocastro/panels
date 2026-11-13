r"""Annular-sector flat plate, total-Lagrangian FSDT with a second-order
director, in polar coordinates (the model named after this file)

Geometry and coordinates
------------------------
As in ``annular_fsdt_donnell``: the annular sector lies in a plane between
the outer radius ``R`` and the inner radius ``R - a`` (``a`` the radial
width, ``b`` the arc length of the outer edge), with

    x = R - rho,   0 <= x <= a      (from the outer edge inwards)
    y = R theta,   0 <= y <= b      (arc length along the outer edge)

the local basis ``e_x = -e_rho``, ``e_y = e_theta``, ``e_z = e_x x e_y``,
the displacements ``u, v, w`` along it, the rotations ``phix``, ``phiy`` of
the normal, ``s = R/rho`` and ``dA = (rho/R) dx dy``. The laminate spans
``d - h/2 <= z <= d + h/2`` with the offset ``d`` (``Shell.offset``), which
enters through the matrix ``ABD`` and the moments of the density.

Kinematics
----------
The displacement field of ``plate_fsdt_tl`` (Schmidt and Reddy 1988, Eq.
(16); Palmerio, Reddy and Schmidt 1990, Part 1, Eqs. (16) and (20), with an
inextensible normal), in the local basis of the annulus: the deformed
position of a point is ``X + z d`` with

    X = X0 + u e_x + v e_y + w e_z,   X0 = rho e_rho
    d = phix e_x + phiy e_y + (1 - (phix^2 + phiy^2)/2) e_z

the second-order director. The local basis turns with ``theta``, ``e_x,y =
-e_y/R`` and ``e_y,y = e_x/R``, so that, per unit length of the reference
lines and in the local basis,

    g1 = X,x       = (1 + u,x, v,x, w,x)
    g2 = s X,y     = (s u,y + v/rho, 1 + s v,y - u/rho, s w,y)
    d,x            = (phix,x, phiy,x, -(phix phix,x + phiy phiy,x))
    s d,y          = (s phix,y + phiy/rho, s phiy,y - phix/rho,
                      -s (phix phix,y + phiy phiy,y))

The reference plane being flat, the metric of the undeformed body does not
depend on ``z``; the strains are the coefficients of ``z^0`` and ``z^1`` of
the Green-Lagrange strains of ``X + z d`` and ``2 E_a3`` at ``z = 0``:

    exx = (g1.g1 - 1)/2,  eyy = (g2.g2 - 1)/2,  gxy = g1.g2
    kxx = g1.d,x,  kyy = g2.(s d,y),  kxy = g1.(s d,y) + g2.d,x
    gyz = g2.d,  gxz = g1.d

The membrane strains are those of ``annular_fsdt_gl`` and
``annular_clpt_gl`` (Amabili 2008, Eqs. (1.166)-(1.168)); the linear part
is ``annular_fsdt_donnell``'s, and for ``R -> infinity`` the strains are
those of ``plate_fsdt_tl``. Dropped, as in ``plate_fsdt_tl``: the
coefficient of ``z^2`` of ``E_ab`` and the coefficient of ``z`` of ``2
E_a3``, ``d,a . d``, of fourth order in the rotations.

Checks (run this file):

- :func:`check_vectors`: ``g1``, ``g2``, ``d``, ``d,x``, ``s d,y`` against the
  derivatives of the Cartesian vectors;
- :func:`check_strain_measures`: the strains are the coefficients of the
  Green-Lagrange strains of ``X + z d`` above;
- :func:`check_linear_part`: the linear strains are those of
  ``annular_fsdt_donnell``, and the membrane strains those of
  ``annular_fsdt_gl``;
- :func:`check_rigid_rotation`: finite rigid rotations of the undeformed
  annulus and of a bent state, with the director components those of the
  rotated normal: the strains change by ``O(t^4)``, the linear curvatures
  of ``annular_fsdt_gl`` by ``O(t^2)``;
- :func:`check_plate_limit`: for ``R -> infinity`` the strains are those of
  ``plate_fsdt_tl``;
- :func:`check_tangent`: ``[B]^T [F] [B] + [H(sigma, d)]`` is the
  derivative of ``[B]^T sigma``.

Kernels
-------
``write_pyx.py`` writes ``panels/models/<model>.pyx`` and
``panels/models/<model>_num.pyx`` from :func:`build`, with the fifteen
generalized derivatives

    d = (u, u,x, u,y, v, v,x, v,y, w, w,x, w,y,
         phix, phix,x, phix,y, phiy, phiy,x, phiy,y) = [Phi] c

``[B] = d eps/d d`` is quadratic and ``[H(sigma, d)] = sum_r sigma_r
d^2 eps_r/d d^2`` linear in ``d``.

References
----------
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
import os

import numpy as np
import sympy
from sympy import (Function, Matrix as M, cos, diff, expand, simplify, sin,
                   sqrt, symbols)

HERE = os.path.dirname(os.path.abspath(__file__))
#: name of the model, that of this file
MODEL = os.path.splitext(os.path.basename(__file__))[0]

x, y, z, R = symbols('x, y, z, R', real=True)
rho = symbols('rho', positive=True)

(u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, phix, phix_x, phix_y, phiy, phiy_x,
 phiy_y) = symbols('u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, phix, phix_x, '
                   'phix_y, phiy, phiy_x, phiy_y', real=True)

#: generalized derivatives at a point, grouped by degree of freedom
D = (u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, phix, phix_x, phix_y, phiy,
     phiy_x, phiy_y)
D_NAMES = tuple(s.name for s in D)
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
#: field and the derivatives along x and y of each generalized derivative
D_OF = {'%s%s' % (f, sfx): (f, ox, oy) for f in FIELDS
        for sfx, ox, oy in (('', 0, 0), ('_x', 1, 0), ('_y', 0, 1))}
DOF = 5
NE = 8

A11, A12, A16, A22, A26, A66 = symbols('A11, A12, A16, A22, A26, A66')
B11, B12, B16, B22, B26, B66 = symbols('B11, B12, B16, B22, B26, B66')
D11, D12, D16, D22, D26, D66 = symbols('D11, D12, D16, D22, D26, D66')
A44, A45, A55 = symbols('A44, A45, A55')
#: constitutive matrix of the FSDT, ``[[A, B, 0], [B, D, 0], [0, 0, Ats]]``
F = M([[A11, A12, A16, B11, B12, B16, 0, 0],
       [A12, A22, A26, B12, B22, B26, 0, 0],
       [A16, A26, A66, B16, B26, B66, 0, 0],
       [B11, B12, B16, D11, D12, D16, 0, 0],
       [B12, B22, B26, D12, D22, D26, 0, 0],
       [B16, B26, B66, D16, D26, D66, 0, 0],
       [0, 0, 0, 0, 0, 0, A44, A45],
       [0, 0, 0, 0, 0, 0, A45, A55]])
STRESSES = symbols('Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qy, Qx')

s = R/rho


def tangent_vectors():
    """``g1 = X,x`` and ``g2 = s X,y`` in the local basis"""
    g1 = M([1 + u_x, v_x, w_x])
    g2 = M([s*u_y + v/rho, 1 + s*v_y - u/rho, s*w_y])
    return g1, g2


def director():
    """Second-order director ``d``, ``d,x`` and ``s d,y`` in the local
    basis"""
    d = M([phix, phiy, 1 - (phix**2 + phiy**2)/2])
    d_x = M([phix_x, phiy_x, -(phix*phix_x + phiy*phiy_x)])
    sd_y = M([s*phix_y + phiy/rho, s*phiy_y - phix/rho,
              -s*(phix*phix_y + phiy*phiy_y)])
    return d, d_x, sd_y


def linear_part(e):
    """Terms of degree one in ``d`` of the polynomial ``e``"""
    e = sympy.expand(e)
    return sum((t for t in sympy.Add.make_args(e)
                if sum(sympy.degree(t, d) for d in D) == 1), sympy.S(0))


def strains(nonlinear=True):
    """Generalized strains (exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz)"""
    g1, g2 = tangent_vectors()
    d, d_x, sd_y = director()
    eps = [(g1.dot(g1) - 1)/2, (g2.dot(g2) - 1)/2, g1.dot(g2),
           g1.dot(d_x), g2.dot(sd_y), g1.dot(sd_y) + g2.dot(d_x),
           g2.dot(d), g1.dot(d)]
    eps = [sympy.expand(e) for e in eps]
    if not nonlinear:
        eps = [linear_part(e) for e in eps]
    return M(eps)


def mass_map():
    """Map ``J`` from ``d`` to ``(u, v, w, phix, phiy)``, the displacement
    ``(u(z), v(z), w(z)) = (u, v, w) + z (phix, phiy, 0)`` of the linear
    kinematics"""
    q = M([u, v, w, phix, phiy])
    return q.jacobian(M(D))


def hessian_stress(sig):
    """``sum_r sigma_r d^2 eps_r/d d^2`` for the stresses ``sig``"""
    eps = strains()
    H = sympy.zeros(len(D), len(D))
    for r_, sr in enumerate(sig):
        H += sr*sympy.hessian(eps[r_], D)
    return H


def build():
    """Expressions used by ``write_pyx.py``: the strains ``eps`` and their
    linear part ``epsL``, ``[B] = d eps/d d``, ``[H(sigma, d)]`` and
    ``[H(sigma, 0)]`` with the symbols ``Nxx, ..., Qx``, and the map ``J``
    of the mass matrix"""
    eps = strains()
    epsL = strains(nonlinear=False)
    B = eps.jacobian(M(D))
    H = hessian_stress(STRESSES)
    return dict(eps=eps, epsL=epsL, B=B, H=H,
                H0=H.subs({d: 0 for d in D}, simultaneous=True),
                J=mass_map())


#: first lines of the docstrings of the generated modules
TITLE = ('Annular-sector flat plate, using the total-Lagrangian first-order '
         'shear\ndeformation theory (FSDT) with a second-order director, in '
         'polar coordinates')

#: notes of the generated modules, the strains
STRAIN_NOTES = r"""The deformed position of a point is `\boldsymbol{X} + z \boldsymbol{d}`,
with the position `\boldsymbol{X}` of the deformed reference plane and the
second-order director of Schmidt and Reddy [1]_, Eq. (16), and Palmerio,
Reddy and Schmidt [2]_, Eqs. (16) and (20), with an inextensible normal,
`\boldsymbol{d} = \phi_x \boldsymbol{e}_x + \phi_y \boldsymbol{e}_y + (1
- (\phi_x^2 + \phi_y^2)/2) \boldsymbol{e}_z`, as in ``plate_fsdt_tl``. With
`s = r/\rho` and, per unit length of the reference lines and in the local
basis, which turns with `\theta`,

.. math::

    \boldsymbol{g}_1 = \boldsymbol{X}_{,x} = \begin{Bmatrix} 1 + u_{,x} \\
    v_{,x} \\ w_{,x} \end{Bmatrix} \qquad
    \boldsymbol{g}_2 = s \boldsymbol{X}_{,y} = \begin{Bmatrix} s u_{,y} +
    v/\rho \\ 1 + s v_{,y} - u/\rho \\ s w_{,y} \end{Bmatrix}

    \boldsymbol{d}_{,x} = \begin{Bmatrix} \phi_{x,x} \\ \phi_{y,x} \\
    -(\phi_x \phi_{x,x} + \phi_y \phi_{y,x}) \end{Bmatrix} \qquad
    s \boldsymbol{d}_{,y} = \begin{Bmatrix} s \phi_{x,y} + \phi_y/\rho \\
    s \phi_{y,y} - \phi_x/\rho \\ -s (\phi_x \phi_{x,y} + \phi_y
    \phi_{y,y}) \end{Bmatrix}

the strains are the coefficients of `z^0` and `z^1` of the Green-Lagrange
strains of `\boldsymbol{X} + z \boldsymbol{d}`:

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

    \gamma_{yz} = \boldsymbol{g}_2 \cdot \boldsymbol{d} \qquad
    \gamma_{xz} = \boldsymbol{g}_1 \cdot \boldsymbol{d}

The membrane strains are the Green-Lagrange strains of Amabili [3]_, Eqs.
(1.166)-(1.168); the linear part is that of ``annular_fsdt_donnell``, and
for `r \to \infty` the strains are those of ``plate_fsdt_tl``. A finite
rigid rotation `t` changes the strains by `O(t^4)`, from the second-order
director. The second derivatives of the strains with respect to the
generalized derivatives are linear in them. The mass matrix is that of
the linear kinematics, `u(z) = u + z \phi_x`, `v(z) = v + z \phi_y`, `w(z)
= w`.
"""

#: references of the generated modules
REFERENCES = r""".. [1] Schmidt, R., and Reddy, J. N., A refined small strain and moderate
   rotation theory of elastic anisotropic shells, Journal of Applied
   Mechanics, Vol. 55, No. 3, 1988, pp. 611-617, Eq. (16).
   https://doi.org/10.1115/1.3125837
.. [2] Palmerio, A. F., Reddy, J. N., and Schmidt, R., On a moderate
   rotation theory of laminated anisotropic shells - Part 1. Theory,
   International Journal of Non-Linear Mechanics, Vol. 25, No. 6, 1990,
   pp. 687-700, Eqs. (16) and (20).
   https://doi.org/10.1016/0020-7462(90)90007-V
.. [3] Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
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
FUNCS = dict(u=U, v=V, w=W, phix=PX, phiy=PY)


def at_point(expr, funcs=None):
    """Replace ``d`` by the derivatives of the functions ``U, V, W, PX,
    PY``, or of the expressions ``funcs`` of each field"""
    funcs = FUNCS if funcs is None else funcs
    rep = {}
    for name, (fld, nx_, ny_) in D_OF.items():
        e = funcs[fld]
        if nx_:
            e = diff(e, x, nx_)
        if ny_:
            e = diff(e, y, ny_)
        rep[symbols(name, real=True)] = e
    return expr.subs(rep, simultaneous=True).subs(rho, R - x)


def basis():
    """``e_x, e_y, e_z`` at ``theta = y/R``"""
    th_ = y/R
    e_rho = M([cos(th_), sin(th_), 0])
    e_th = M([-sin(th_), cos(th_), 0])
    e_ax = M([0, 0, 1])
    return -e_rho, e_th, -e_ax


def _cartesian():
    """Position ``X`` and director ``d`` of the functions ``U, ..., PY``"""
    ex, ey, ez = basis()
    X = (R - x)*(-ex) + U*ex + V*ey + W*ez
    d = PX*ex + PY*ey + (1 - (PX**2 + PY**2)/2)*ez
    return X, d


def check_vectors():
    """``g1``, ``g2``, ``d``, ``d,x`` and ``s d,y`` in the local basis"""
    ex, ey, ez = basis()
    X, d = _cartesian()
    loc = lambda vec: M([vec.dot(ex), vec.dot(ey), vec.dot(ez)])
    g1, g2 = tangent_vectors()
    dd, d_x, sd_y = director()
    sc = R/(R - x)
    pairs = ((loc(diff(X, x)), g1), (loc(diff(X, y))*sc, g2),
             (loc(d), dd), (loc(diff(d, x)), d_x), (loc(diff(d, y))*sc, sd_y))
    for g, e in pairs:
        for gi, ei in zip(g, at_point(e)):
            assert simplify(gi - ei) == 0, (gi, ei)
    print('check_vectors: OK')


def check_strain_measures():
    """The strains are the coefficients of ``z^0`` and ``z^1`` of the
    Green-Lagrange strains of ``X + z d``, in the metric of the plane

    The reference point is ``X0 + z e_z``, with ``X0,x = e_x``, ``X0,y =
    (rho/R) e_y`` for every ``z``: ``E_xx = (P,x.P,x - 1)/2``, ``E_yy = (s^2
    P,y.P,y - 1)/2``, ``2 E_xy = s P,x.P,y``, ``2 E_y3 = s P,y.P,z`` and
    ``2 E_x3 = P,x.P,z``, with ``P = X + z d``; the transverse shear strains
    are taken at ``z = 0``.
    """
    X, d = _cartesian()
    P = X + z*d
    sc = R/(R - x)
    Px, Py = diff(P, x), diff(P, y)
    E = [(Px.dot(Px) - 1)/2, (sc**2*Py.dot(Py) - 1)/2, sc*Px.dot(Py)]
    eps = at_point(strains())
    for k in range(3):
        Ek = expand(E[k])
        assert simplify(Ek.coeff(z, 0) - eps[k]) == 0, k
        assert simplify(Ek.coeff(z, 1) - eps[3 + k]) == 0, k
    assert simplify((sc*diff(X, y).dot(d)) - eps[6]) == 0
    assert simplify(diff(X, x).dot(d) - eps[7]) == 0
    print('check_strain_measures: OK')


def check_linear_part():
    """Linear strains of ``annular_fsdt_donnell``; membrane strains of
    ``annular_fsdt_gl``"""
    donnell = M([u_x, s*v_y - u/rho, s*u_y + v_x + v/rho,
                 phix_x, s*phiy_y - phix/rho, s*phix_y + phiy_x + phiy/rho,
                 phiy + s*w_y, phix + w_x])
    for g, e in zip(strains(nonlinear=False), donnell):
        assert simplify(g - e) == 0, (g, e)
    g1, g2 = tangent_vectors()
    gl = [(g1.dot(g1) - 1)/2, (g2.dot(g2) - 1)/2, g1.dot(g2)]
    for g, e in zip(strains()[:3], gl):
        assert simplify(g - e) == 0, (g, e)
    print('check_linear_part: OK')


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
    X0 = (R - x)*(-ex)
    Ug = Xc - X0
    funcs = dict(u=Ug.dot(ex), v=Ug.dot(ey), w=Ug.dot(ez), phix=nc.dot(ex),
                 phiy=nc.dot(ey))
    vals = {}
    for name, (fld, nx_, ny_) in D_OF.items():
        e = funcs[fld].subs(R, Rv)
        if nx_:
            e = diff(e, x, nx_)
        if ny_:
            e = diff(e, y, ny_)
        vals[symbols(name, real=True)] = float(e.subs({x: xv, y: yv}))
    vals.update({R: Rv, rho: Rv - xv})
    return vals


def check_rigid_rotation(seed=0):
    r"""Finite rigid rotations `t` about random axes, the director
    components being those of the rotated unit normal

    Undeformed annulus: the membrane strains vanish and the other strains
    are of `O(t^4)` (fitted order between `t = 0.2` and `0.1` above 3.8).
    Bent state, the position `X0 + w_0 e_z` with `w_0 = k_0 (x - x_p)^2/2 +
    k_0 (y - y_p)^2/4`, `(x_p, y_p)` the point where the strains are
    evaluated, and its unit normal, rotated about an axis in the
    plane: the strains change by `O(t^4)` (order above 3.8), the linear
    curvatures of ``annular_fsdt_gl`` by `O(t^2)` (order between 1.8 and
    2.2), at least 20 times more at `t = 0.1`.
    """
    rng = np.random.default_rng(seed)
    Rv, xv, yv = 2., 0.7, 0.9
    eps = strains()
    kap_gl = M([linear_part(e) for e in eps[3:6]])
    ex, ey, ez = basis()
    X0 = (R - x)*(-ex)

    def ev(e, vals):
        return np.array([float(t.xreplace(vals)) for t in e])

    def order(c):
        return np.log2(c[0]/c[1])
    for _ in range(3):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        ch = []
        for t in (0.2, 0.1):
            Q = M(rotation(axis, t))
            e = ev(eps, _state(Q*X0, Q*ez, xv, yv, Rv))
            assert np.abs(e[:3]).max() < 1e-12
            ch.append(np.abs(e[3:]).max())
        assert order(ch) > 3.8, ch
    k0 = 0.3
    w0 = k0*(x - xv)**2/2 + k0*(y - yv)**2/4
    Xb = X0 + w0*ez
    nb = diff(Xb, x).cross(diff(Xb, y))
    nb = nb/sqrt(nb.dot(nb))
    base = ev(eps, _state(Xb, nb, xv, yv, Rv))
    base_gl = ev(kap_gl, _state(Xb, nb, xv, yv, Rv))
    for _ in range(3):
        axis = np.array([*rng.standard_normal(2), 0.])
        axis /= np.linalg.norm(axis)
        ch, ch_gl = [], []
        for t in (0.2, 0.1):
            Q = M(rotation(axis, t))
            st = _state(Q*Xb, Q*nb, xv, yv, Rv)
            ch.append(np.abs(ev(eps, st) - base).max())
            ch_gl.append(np.abs(ev(kap_gl, st) - base_gl).max())
        assert order(ch) > 3.8, ch
        assert 1.8 < order(ch_gl) < 2.2, ch_gl
        assert ch[1] < 0.05*ch_gl[1], (ch, ch_gl)
    print('check_rigid_rotation: OK')


def check_plate_limit():
    """``R -> infinity``: the strains of ``plate_fsdt_tl``"""
    Xx, Xy = M([1 + u_x, v_x, w_x]), M([u_y, 1 + v_y, w_y])
    d = M([phix, phiy, 1 - (phix**2 + phiy**2)/2])
    d_x = M([phix_x, phiy_x, -(phix*phix_x + phiy*phiy_x)])
    d_y = M([phix_y, phiy_y, -(phix*phix_y + phiy*phiy_y)])
    plate = M([(Xx.dot(Xx) - 1)/2, (Xy.dot(Xy) - 1)/2, Xx.dot(Xy),
               Xx.dot(d_x), Xy.dot(d_y), Xx.dot(d_y) + Xy.dot(d_x),
               Xy.dot(d), Xx.dot(d)])
    eps = strains().subs(rho, R - x)
    for g, e in zip(eps, plate):
        assert simplify(sympy.limit(g, R, sympy.oo) - e) == 0
    print('check_plate_limit: OK')


def check_tangent(seed=0):
    """``[B]^T [F] [B] + [H(sigma, d)]`` is the derivative of ``[B]^T
    sigma``"""
    rng = np.random.default_rng(seed)
    eps = strains()
    B = eps.jacobian(M(D))
    sig = F*eps
    KT = (B.T*sig).jacobian(M(D))
    H = hessian_stress(list(sig))
    rep = {sym: rng.random() for sym in F.free_symbols}
    rep.update({d: rng.random() - 0.5 for d in D})
    rep.update({R: 2., rho: 1.3})
    diffm = (KT - B.T*F*B - H).subs(rep)
    assert max(abs(float(e)) for e in diffm) < 1e-12
    print('check_tangent: OK')


if __name__ == '__main__':
    check_vectors()
    check_strain_measures()
    check_linear_part()
    check_rigid_rotation()
    check_plate_limit()
    check_tangent()
