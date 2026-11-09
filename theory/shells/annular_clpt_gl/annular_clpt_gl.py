r"""Annular-sector flat plate, CLPT with the Green-Lagrange membrane strains
in polar coordinates (the model named after this file)

Geometry and coordinates
------------------------
As in ``annular_clpt_donnell``: the annular sector lies in a plane between
the outer radius ``R`` and the inner radius ``R - a`` (``a`` the radial
width, ``b`` the arc length of the outer edge), with

    x = R - rho,   0 <= x <= a      (from the outer edge inwards)
    y = R theta,   0 <= y <= b      (arc length along the outer edge)

the local basis ``e_x = -e_rho``, ``e_y = e_theta``, ``e_z = e_x x e_y``,
the displacements ``u, v, w`` along it, ``s = R/rho`` and ``dA = (rho/R) dx
dy``. The web of a ring frame is an application (``a`` the height of the
web, ``b`` the arc length of its outer edge, ``R`` its outer radius),
connected to a cylinder with ``'BFxcte'``. The laminate spans ``d - h/2 <=
z <= d + h/2`` with the offset ``d`` (``Shell.offset``), which enters
through the matrix ``ABD`` and the moments of the density.

Kinematics
----------
Kirchhoff-Love. The position of the deformed reference plane is ``X = X0 +
u e_x + v e_y + w e_z``, ``X0 = rho e_rho``; with ``e_x,y = -e_y/R`` and
``e_y,y = e_x/R`` its tangent vectors per unit length of the reference
lines are, in the local basis,

    g1 = X,x       = (1 + u,x, v,x, w,x)
    g2 = s X,y     = (s u,y + v/rho, 1 + s v,y - u/rho, s w,y)

The membrane strains are the complete Green-Lagrange strains of the
reference plane, ``exx = (g1.g1 - 1)/2``, ``eyy = (g2.g2 - 1)/2``, ``gxy =
g1.g2``, Amabili (2008), Eqs. (1.166)-(1.168), and the changes of curvature
are linear, those of ``annular_clpt_donnell``:

    kxx = -w,xx,   kyy = w,x/rho - s^2 w,yy,   kxy = -2 s (w,xy + w,y/rho)

as ``plate_clpt_gl`` for the flat plate. The membrane strains vanish for
any finite rigid motion of the plane.

Checks (run this file):

- :func:`check_linear_part`: the linear strains are those of
  ``annular_clpt_donnell``;
- :func:`check_amabili`: the membrane strains are Eqs. (1.166)-(1.168) of
  Amabili (2008) and those of the deformation gradient of the reference
  plane;
- :func:`check_rigid_body`: the membrane strains of a finite rigid motion
  vanish, and the linear strains of a small one;
- :func:`check_plate_limit`: for ``R -> infinity`` the strains are those of
  ``plate_clpt_gl``;
- :func:`check_tangent`: ``[B]^T [ABD] [B] + [H(sigma)]`` is the derivative
  of ``[B]^T sigma``.

Kernels
-------
``write_pyx.py`` writes ``panels/models/<model>.pyx`` and
``panels/models/<model>_num.pyx`` from :func:`build`, with the twelve
generalized derivatives

    d = (u, u,x, u,y, v, v,x, v,y, w, w,x, w,y, w,xx, w,yy, w,xy) = [Phi] c

``[H(sigma)] = sum_r sigma_r d^2 eps_r/d d^2`` is constant in ``d``, with
the membrane stresses on ``(u, u,x, u,y, v, v,x, v,y)`` and ``(w,x, w,y)``.

References
----------
Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
Cambridge University Press, 2008, Sec. 1.4. doi:10.1017/CBO9780511619694

"""
import os

import numpy as np
import sympy
from sympy import Function, Matrix as M, cos, diff, simplify, sin, symbols

HERE = os.path.dirname(os.path.abspath(__file__))
#: name of the model, that of this file
MODEL = os.path.splitext(os.path.basename(__file__))[0]

x, y, z, R = symbols('x, y, z, R', real=True)
rho = symbols('rho', positive=True)

u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy = symbols(
    'u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy', real=True)

#: generalized derivatives at a point, grouped by degree of freedom
D = (u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy)
#: field and the derivatives along x and y of each generalized derivative
D_OF = dict(u=('u', 0, 0), u_x=('u', 1, 0), u_y=('u', 0, 1),
            v=('v', 0, 0), v_x=('v', 1, 0), v_y=('v', 0, 1),
            w=('w', 0, 0), w_x=('w', 1, 0), w_y=('w', 0, 1),
            w_xx=('w', 2, 0), w_yy=('w', 0, 2), w_xy=('w', 1, 1))
FIELDS = ('u', 'v', 'w')
DOF = 3

A11, A12, A16, A22, A26, A66 = symbols('A11, A12, A16, A22, A26, A66')
B11, B12, B16, B22, B26, B66 = symbols('B11, B12, B16, B22, B26, B66')
D11, D12, D16, D22, D26, D66 = symbols('D11, D12, D16, D22, D26, D66')
F = M([[A11, A12, A16, B11, B12, B16],
       [A12, A22, A26, B12, B22, B26],
       [A16, A26, A66, B16, B26, B66],
       [B11, B12, B16, D11, D12, D16],
       [B12, B22, B26, D12, D22, D26],
       [B16, B26, B66, D16, D26, D66]])
Nxx, Nyy, Nxy, Mxx, Myy, Mxy = symbols('Nxx, Nyy, Nxy, Mxx, Myy, Mxy')
STRESSES = (Nxx, Nyy, Nxy, Mxx, Myy, Mxy)

s = R/rho


def tangent_vectors():
    """``g1 = X,x`` and ``g2 = s X,y`` in the local basis"""
    g1 = M([1 + u_x, v_x, w_x])
    g2 = M([s*u_y + v/rho, 1 + s*v_y - u/rho, s*w_y])
    return g1, g2


def linear_part(e):
    """Terms of degree one in ``d`` of the polynomial ``e``"""
    e = sympy.expand(e)
    return sum((t for t in sympy.Add.make_args(e)
                if sum(sympy.degree(t, d) for d in D) == 1), sympy.S(0))


def strains(nonlinear=True):
    """Generalized strains (exx, eyy, gxy, kxx, kyy, kxy) at a point"""
    g1, g2 = tangent_vectors()
    mem = [sympy.expand((g1.dot(g1) - 1)/2), sympy.expand((g2.dot(g2) - 1)/2),
           sympy.expand(g1.dot(g2))]
    if not nonlinear:
        mem = [linear_part(e) for e in mem]
    return M(mem + [-w_xx, w_x/rho - s**2*w_yy, -2*s*(w_xy + w_y/rho)])


def mass_map():
    """Map ``J`` from ``d`` to ``(u, v, w, -w,x, -s w,y)``, the displacement
    ``(u(z), v(z), w(z)) = (u, v, w) + z (-w,x, -s w,y, 0)``"""
    q = M([u, v, w, -w_x, -s*w_y])
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
    ``[H(sigma, 0)]`` with the symbols ``Nxx, ..., Mxy``, and the map ``J``
    of the mass matrix"""
    eps = strains()
    epsL = strains(nonlinear=False)
    B = eps.jacobian(M(D))
    H = hessian_stress(STRESSES)
    return dict(eps=eps, epsL=epsL, B=B, H=H,
                H0=H.subs({d: 0 for d in D}, simultaneous=True),
                J=mass_map())


#: first lines of the docstrings of the generated modules
TITLE = ('Annular-sector flat plate, using the classical laminated plate '
         'theory\n(CLPT) with Green-Lagrange membrane strains in polar '
         'coordinates')

#: notes of the generated modules, the strains
STRAIN_NOTES = r"""With `s = r/\rho` and the tangent vectors of the deformed reference plane
per unit length of the reference lines, in the local basis,
`\boldsymbol{g}_1 = \{1 + u_{,x}, v_{,x}, w_{,x}\}^\top` and
`\boldsymbol{g}_2 = \{s u_{,y} + v/\rho, 1 + s v_{,y} - u/\rho, s
w_{,y}\}^\top`, the membrane strains are the complete Green-Lagrange
strains of Amabili [1]_, Eqs. (1.166)-(1.168), in these coordinates, and
the changes of curvature are linear:

.. math::

    \varepsilon_{xx} = \frac{1}{2} \left(\boldsymbol{g}_1 \cdot
    \boldsymbol{g}_1 - 1\right) \qquad
    \varepsilon_{yy} = \frac{1}{2} \left(\boldsymbol{g}_2 \cdot
    \boldsymbol{g}_2 - 1\right) \qquad
    \gamma_{xy} = \boldsymbol{g}_1 \cdot \boldsymbol{g}_2

    \kappa_{xx} = -w_{,xx} \qquad
    \kappa_{yy} = \frac{w_{,x}}{\rho} - s^2 w_{,yy} \qquad
    \kappa_{xy} = -2 s \left(w_{,xy} + \frac{w_{,y}}{\rho}\right)

The membrane strains vanish for any finite rigid motion; for `r \to
\infty` the strains are those of ``plate_clpt_gl``. The second derivatives
of the strains with respect to the generalized derivatives are constant.
"""

#: references of the generated modules
REFERENCES = r""".. [1] Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
   Cambridge University Press, 2008, Sec. 1.4.
   https://doi.org/10.1017/CBO9780511619694
"""


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
U = Function('U', real=True)(x, y)
V = Function('V', real=True)(x, y)
W = Function('W', real=True)(x, y)


def at_point(expr):
    """Replace ``d`` by the derivatives of the functions ``U, V, W``"""
    rep = {}
    for name, (fld, nx_, ny_) in D_OF.items():
        e = dict(u=U, v=V, w=W)[fld]
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


def check_linear_part():
    """Linear strains of ``annular_clpt_donnell``"""
    donnell = M([u_x, s*v_y - u/rho, s*u_y + v_x + v/rho,
                 -w_xx, w_x/rho - s**2*w_yy, -2*s*(w_xy + w_y/rho)])
    for g, e in zip(strains(nonlinear=False), donnell):
        assert simplify(g - e) == 0, (g, e)
    print('check_linear_part: OK')


def check_amabili():
    """Membrane strains of Amabili (2008) Eqs. (1.166)-(1.168) and of the
    deformation gradient

    Amabili's displacements ``u1(r, theta)``, ``u2``, ``u3`` (radial,
    circumferential, along ``e_axis``) are ``u1 = -U``, ``u2 = V``, ``u3 =
    -W``, with ``r = R - x``, ``theta = y/R``: ``exx = eps_r``, ``eyy =
    eps_theta``, ``gxy = -gamma_rtheta``.
    """
    r_ = R - x
    u1, u2, u3 = -U, V, -W

    def dr(f):
        return -diff(f, x)

    def dt(f):
        return R*diff(f, y)

    e_rr = dr(u1) + (dr(u1)**2 + dr(u2)**2 + dr(u3)**2)/2
    e_tt = ((dt(u2) + u1)/r_ + ((dt(u2) + u1)**2 + (dt(u1) - u2)**2
            + dt(u3)**2)/(2*r_**2))
    g_rt = (dt(u1)/r_ + dr(u2) - u2/r_ + (dr(u1)*(dt(u1) - u2)
            + dr(u2)*(dt(u2) + u1) + dr(u3)*dt(u3))/r_)
    mem = at_point(strains())[:3]
    for g, e in zip(mem, (e_rr, e_tt, -g_rt)):
        assert simplify(sympy.expand(g - e)) == 0, (g, e)
    ex, ey, ez = basis()
    X = (R - x)*(-ex) + U*ex + V*ey + W*ez
    sc = r_/R
    Exx = (diff(X, x).dot(diff(X, x)) - 1)/2
    Eyy = (diff(X, y).dot(diff(X, y))/sc**2 - 1)/2
    Gxy = diff(X, x).dot(diff(X, y))/sc
    for g, e in zip(mem, (Exx, Eyy, Gxy)):
        assert simplify(sympy.expand(g - e)) == 0, (g, e)
    print('check_amabili: OK')


def rigid_point(Q, t, xv, yv, Rv):
    """``d`` of the rigid motion ``X = Q X0 + t`` at ``(xv, yv)``"""
    ex, ey, ez = basis()
    X0 = (R - x)*(-ex)
    Ug = M(Q)*X0 + M(t) - X0
    fields = dict(u=Ug.dot(ex), v=Ug.dot(ey), w=Ug.dot(ez))
    vals = {}
    for name, (fld, nx_, ny_) in D_OF.items():
        e = fields[fld]
        if nx_:
            e = diff(e, x, nx_)
        if ny_:
            e = diff(e, y, ny_)
        vals[symbols(name, real=True)] = float(e.subs({x: xv, y: yv, R: Rv}))
    vals.update({R: Rv, rho: Rv - xv})
    return vals


def rotation(axis, angle):
    """Rotation matrix of ``angle`` about the unit vector ``axis``"""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*K @ K


def check_rigid_body(seed=0):
    """Finite rigid motions: zero membrane strains; small ones: linear
    strains of the order of the square of the rotation"""
    rng = np.random.default_rng(seed)
    for angle, eps, rows in ((1.1, strains(), 3),
                             (1e-6, strains(nonlinear=False), 6)):
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        vals = rigid_point(rotation(axis, angle), angle*rng.standard_normal(3),
                           0.7, 0.9, 2.)
        for e in eps[:rows]:
            assert abs(float(e.subs(vals))) < 1e-10, e
    print('check_rigid_body: OK')


def check_plate_limit():
    """``R -> infinity``: the strains of ``plate_clpt_gl``"""
    plate = M([u_x + (u_x**2 + v_x**2 + w_x**2)/2,
               v_y + (u_y**2 + v_y**2 + w_y**2)/2,
               u_y + v_x + u_x*u_y + v_x*v_y + w_x*w_y,
               -w_xx, -w_yy, -2*w_xy])
    eps = strains().subs(rho, R - x)
    for g, e in zip(eps, plate):
        assert simplify(sympy.limit(g, R, sympy.oo) - e) == 0
    print('check_plate_limit: OK')


def check_tangent(seed=0):
    """``[B]^T [ABD] [B] + [H(sigma)]`` is the derivative of ``[B]^T sigma``"""
    rng = np.random.default_rng(seed)
    eps = strains()
    B = eps.jacobian(M(D))
    sig = F*eps
    KT = (B.T*sig).jacobian(M(D))
    H = hessian_stress(list(sig))
    Fvals = rng.random((6, 6))
    Fvals = Fvals + Fvals.T
    rep = {F[i, j]: Fvals[i, j] for i in range(6) for j in range(6)}
    rep.update({d: rng.random() - 0.5 for d in D})
    rep.update({R: 2., rho: 1.3})
    diffm = (KT - B.T*F*B - H).subs(rep)
    assert max(abs(float(e)) for e in diffm) < 1e-12
    print('check_tangent: OK')


if __name__ == '__main__':
    check_linear_part()
    check_amabili()
    check_rigid_body()
    check_plate_limit()
    check_tangent()
