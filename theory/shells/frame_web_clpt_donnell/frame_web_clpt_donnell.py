r"""Web of a ring frame, an annular-sector flat plate, CLPT with von Kármán
strains in polar coordinates (``frame_web_clpt_donnell``)

Geometry and coordinates
------------------------
The web lies in a plane normal to the axis of a cylinder, between the outer
radius ``R`` and the inner radius ``R - a``; ``a`` is the height of the web
and ``b`` the arc length of its outer edge, ``b = R theta_b``. With the
polar coordinates ``(rho, theta)`` of the plane, the coordinates of the
model are

    x = R - rho,   0 <= x <= a      (from the outer edge inwards)
    y = R theta,   0 <= y <= b      (arc length along the outer edge)

and the Ritz series is that of the other models, ``u = sum c f_i(xi)
g_j(eta)``, ``xi = 2x/a - 1``, ``eta = 2y/b - 1``. The local basis of the
web at a point is

    e_x = -e_rho,   e_y = e_theta,   e_z = e_x x e_y = -e_axis

with ``e_axis = e_rho x e_theta``, and ``u, v, w`` are the displacements of
the mid-surface along ``e_x, e_y, e_z``. In the polar notation of Amabili
(2008, Sec. 1.4, Fig. 1.14), with the radial, circumferential and normal
displacements ``u1, u2, u3`` and the normal ``e_axis``:

    u1 = -u,   u2 = v,   u3 = -w,   d/drho = -d/dx,   d/dtheta = R d/dy

This is the orientation of the flange of the connection ``'BFxcte'`` of
``panels`` (the local axes of the flange are those of the base rotated by 90
degrees about the common axis ``y``): the web connects to a cylinder of
radius ``R`` along its edge ``x = 0`` with ``'BFxcte'``, the cylinder being
the base, see ``frame_web_clpt_donnell.md``.

The scale factor of the arc length is ``s = R/rho``: ``d/ds_theta = (1/rho)
d/dtheta = s d/dy``, and the element of area is ``dA = (rho/R) dx dy``.

Kinematics
----------
Kirchhoff-Love (classical laminated plate theory), ``z`` measured from the
reference surface along ``e_z``:

    u(z) = u - z w,x,   v(z) = v - z s w,y,   w(z) = w

Strains, von Kármán (Amabili 2008, Eqs. (1.175)-(1.180), in the coordinates
above):

    exx = u,x + w,x^2/2
    eyy = s v,y - u/rho + s^2 w,y^2/2
    gxy = s u,y + v,x + v/rho + s w,x w,y
    kxx = -w,xx
    kyy = w,x/rho - s^2 w,yy
    kxy = -2 s (w,xy + w,y/rho)

Checks (run ``python frame_web_clpt_donnell.py``):

- :func:`check_linear_strains`: the linear strains and changes of curvature
  follow from the linear strains of the three-dimensional Kirchhoff
  displacement field in the metric of the plane;
- :func:`check_amabili`: the strains above are Eqs. (1.175)-(1.180) of
  Amabili (2008) and the complete Green-Lagrange membrane strains of the
  deformation gradient are Eqs. (1.166)-(1.168), after the change of
  variables above;
- :func:`check_rigid_body`: the six small rigid-body motions give zero
  linear strains;
- :func:`check_plate_limit`: for ``R -> infinity`` the strains are those of
  ``plate_clpt_donnell``;
- :func:`check_tangent`: the tangent stiffness of the kernels is the
  derivative of the internal force vector.

Kernels
-------
``write_pyx.py`` writes ``panels/models/frame_web_clpt_donnell.pyx`` and
``panels/models/frame_web_clpt_donnell_num.pyx`` from :func:`build`. At each
integration point the twelve generalized derivatives

    d = (u, u,x, u,y, v, v,x, v,y, w, w,x, w,y, w,xx, w,yy, w,xy) = [Phi] c

give the strains, ``[B] = d eps/d d`` (6 x 12) and ``[H(N)] = sum_r sigma_r
d^2 eps_r/d d^2``, whose only block is on ``(w,x, w,y)``:

    [H(N)] = [[Nxx, s Nxy], [s Nxy, s^2 Nyy]]

References
----------
Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
Cambridge University Press, 2008, Sec. 1.4. doi:10.1017/CBO9780511619694

"""
import os
import sys

import numpy as np
import sympy
from sympy import Function, Matrix as M, Rational, cos, diff, simplify, sin, symbols

HERE = os.path.dirname(os.path.abspath(__file__))

x, y, z, R = symbols('x, y, z, R', real=True)
rho = symbols('rho', positive=True)

u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy = symbols(
    'u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy', real=True)

#: generalized derivatives at a point, grouped by degree of freedom
D = (u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy)
D_NAMES = tuple(s.name for s in D)
#: field and the derivatives along x and y of each generalized derivative
D_OF = dict(u=('u', 0, 0), u_x=('u', 1, 0), u_y=('u', 0, 1),
            v=('v', 0, 0), v_x=('v', 1, 0), v_y=('v', 0, 1),
            w=('w', 0, 0), w_x=('w', 1, 0), w_y=('w', 0, 1),
            w_xx=('w', 2, 0), w_yy=('w', 0, 2), w_xy=('w', 1, 1))
FIELDS = ('u', 'v', 'w')
DOF = 3
NE = 6

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


def strains(nonlinear=True):
    """Generalized strains (exx, eyy, gxy, kxx, kyy, kxy) at a point"""
    k = 1 if nonlinear else 0
    return M([u_x + k*w_x**2/2,
              s*v_y - u/rho + k*s**2*w_y**2/2,
              s*u_y + v_x + v/rho + k*s*w_x*w_y,
              -w_xx,
              w_x/rho - s**2*w_yy,
              -2*s*(w_xy + w_y/rho)])


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
    """Expressions used by ``write_pyx.py``"""
    eps = strains()
    epsL = strains(nonlinear=False)
    epsNL = eps - epsL
    B = eps.jacobian(M(D))
    return dict(eps=eps, epsL=epsL, epsNL=epsNL, B=B,
                BL=B.subs({w_x: 0, w_y: 0}),
                H=hessian_stress(STRESSES), J=mass_map())


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
        F_ = dict(u=U, v=V, w=W)[fld]
        e = F_
        if nx_:
            e = diff(e, x, nx_)
        if ny_:
            e = diff(e, y, ny_)
        rep[symbols(name, real=True)] = e
    return expr.subs(rep, simultaneous=True).subs(rho, R - x)


def basis():
    th = y/R
    e_rho = M([cos(th), sin(th), 0])
    e_th = M([-sin(th), cos(th), 0])
    e_ax = M([0, 0, 1])
    return th, e_rho, e_th, e_ax


def check_linear_strains():
    """Linear strains from the 3-D Kirchhoff field in the metric of the plane

    ``P(x, y, z) = X0(x, y) + z e_z`` with ``X0 = (R - x) e_rho``; the
    plane being flat, ``P,x = e_x``, ``P,y = (rho/R) e_y``, ``P,z = e_z``
    for every ``z``. The linear strains of the displacement ``U3 = u(z) e_x
    + v(z) e_y + w e_z`` are ``e_xx = U3,x . e_x``, ``e_yy = U3,y .
    e_y/(rho/R)``, ``g_xy = U3,x . e_y + U3,y . e_x/(rho/R)``, and their
    derivatives in ``z`` the changes of curvature.
    """
    th, e_rho, e_th, e_ax = basis()
    ex, ey, ez = -e_rho, e_th, -e_ax
    sc = (R - x)/R
    uz = U - z*diff(W, x)
    vz = V - z*R/(R - x)*diff(W, y)
    U3 = uz*ex + vz*ey + W*ez
    exx = diff(U3, x).dot(ex)
    eyy = diff(U3, y).dot(ey)/sc
    gxy = diff(U3, x).dot(ey) + diff(U3, y).dot(ex)/sc
    # transverse shear of the Kirchhoff field
    gyz = diff(U3, z).dot(ey) + diff(U3, y).dot(ez)/sc
    gxz = diff(U3, z).dot(ex) + diff(U3, x).dot(ez)
    assert simplify(gyz) == 0 and simplify(gxz) == 0
    lin = at_point(strains(nonlinear=False))
    got = [exx.subs(z, 0), eyy.subs(z, 0), gxy.subs(z, 0),
           diff(exx, z), diff(eyy, z), diff(gxy, z)]
    for g, e in zip(got, lin):
        assert simplify(g - e) == 0, (g, e)
    print('check_linear_strains: OK')


def check_amabili():
    """Strains of Amabili (2008) Eqs. (1.166)-(1.168), (1.175)-(1.180)

    Amabili's displacements ``u1(r, theta)``, ``u2``, ``u3`` (radial,
    circumferential, along ``e_axis``) are written with ``u1 = -U``, ``u2 =
    V``, ``u3 = -W``, ``r = R - x``, ``theta = y/R``, i.e. ``d/dr = -d/dx``,
    ``d/dtheta = R d/dy``. With ``e_x = -e_rho`` and ``e_z = -e_axis``:
    ``exx = eps_r``, ``eyy = eps_theta``, ``gxy = -gamma_rtheta``,
    ``kxx = -k_r``, ``kyy = -k_theta``, ``kxy = k_rtheta``; the curvatures
    change sign with ``z``, the shear strain with ``e_x``.
    """
    r_ = R - x
    u1, u2, u3 = -U, V, -W

    def dr(f):
        return -diff(f, x)

    def dt(f):
        return R*diff(f, y)

    # von Kármán, Eqs. (1.175)-(1.180)
    er = dr(u1) + dr(u3)**2/2
    et = dt(u2)/r_ + u1/r_ + (dt(u3)/r_)**2/2
    grt = dt(u1)/r_ + dr(u2) - u2/r_ + dr(u3)*dt(u3)/r_
    kr = -dr(dr(u3))
    kt = -(dr(u3)/r_ + dt(dt(u3))/r_**2)
    krt = -2*(dr(dt(u3))/r_ - dt(u3)/r_**2)
    ama = [er, et, -grt, -kr, -kt, krt]
    vk = at_point(strains())
    for g, e in zip(vk, ama):
        assert simplify(g - e) == 0, (g, e)
    # complete Green-Lagrange membrane strains, Eqs. (1.166)-(1.168), against
    # the deformation gradient of the mid-surface
    th, e_rho, e_th, e_ax = basis()
    ex, ey, ez = -e_rho, e_th, -e_ax
    X = (R - x)*e_rho + U*ex + V*ey + W*ez
    sc = r_/R
    Exx = (diff(X, x).dot(diff(X, x)) - 1)/2
    Eyy = (diff(X, y).dot(diff(X, y))/sc**2 - 1)/2
    Gxy = diff(X, x).dot(diff(X, y))/sc
    e_rr = dr(u1) + (dr(u1)**2 + dr(u2)**2 + dr(u3)**2)/2
    e_tt = ((dt(u2) + u1)/r_ + ((dt(u2) + u1)**2 + (dt(u1) - u2)**2
            + dt(u3)**2)/(2*r_**2))
    g_rt = (dt(u1)/r_ + dr(u2) - u2/r_ + (dr(u1)*(dt(u1) - u2)
            + dr(u2)*(dt(u2) + u1) + dr(u3)*dt(u3))/r_)
    for g, e in zip((Exx, Eyy, Gxy), (e_rr, e_tt, -g_rt)):
        assert simplify(sympy.expand(g - e)) == 0, (g, e)
    print('check_amabili: OK')


def check_rigid_body():
    """Six small rigid-body motions ``t + o x X0`` give zero linear strains"""
    th, e_rho, e_th, e_ax = basis()
    ex, ey, ez = -e_rho, e_th, -e_ax
    t1, t2, t3, o1, o2, o3 = symbols('t1, t2, t3, o1, o2, o3', real=True)
    X0 = (R - x)*e_rho
    Ug = M([t1, t2, t3]) + M([o1, o2, o3]).cross(X0)
    rep = {U: Ug.dot(ex), V: Ug.dot(ey), W: Ug.dot(ez)}
    lin = at_point(strains(nonlinear=False))
    for e in lin:
        assert simplify(e.subs(rep).doit()) == 0, e
    print('check_rigid_body: OK')


def check_plate_limit():
    """``R -> infinity``: the strains of ``plate_clpt_donnell``"""
    plate = M([u_x + w_x**2/2, v_y + w_y**2/2, u_y + v_x + w_x*w_y,
               -w_xx, -w_yy, -2*w_xy])
    eps = strains().subs(rho, R - x)
    for g, e in zip(eps, plate):
        assert sympy.limit(g, R, sympy.oo) == e
    print('check_plate_limit: OK')


def check_tangent(seed=0):
    """``[B]^T [ABD] [B] + [H(sigma)]`` is the derivative of ``[B]^T sigma``"""
    rng = np.random.default_rng(seed)
    eps = strains()
    B = eps.jacobian(M(D))
    sig = F*eps
    fint = B.T*sig
    KT = fint.jacobian(M(D))
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
    check_linear_strains()
    check_amabili()
    check_rigid_body()
    check_plate_limit()
    check_tangent()
