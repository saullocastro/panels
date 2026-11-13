r"""Annular-sector flat plate, FSDT with von Kármán strains in polar
coordinates (the model named after this file)

Geometry and coordinates
------------------------
Those of ``annular_clpt_donnell``. The annular sector lies in a plane
between the outer radius ``R`` and the inner radius ``R - a``; ``a`` is its
radial width and ``b`` the arc length of its outer edge, ``b = R
theta_b``. With the polar coordinates ``(rho, theta)`` of the plane:

    x = R - rho,   0 <= x <= a      (from the outer edge inwards)
    y = R theta,   0 <= y <= b      (arc length along the outer edge)

The local basis is ``e_x = -e_rho``, ``e_y = e_theta``, ``e_z = e_x x e_y =
-e_axis``; ``u, v, w`` are the displacements of the reference plane along
it and ``phix``, ``phiy`` the rotations of the normal, the derivatives
along ``z`` of the in-plane displacements. In the polar notation of
Bagheri, Kiani and Eslami (2019), with the radial, circumferential and
transverse displacements ``u0, v0, w0``, the rotations ``psi_r``,
``psi_theta`` and ``z_B`` along ``e_axis``:

    u0 = -u,  v0 = v,  w0 = -w,  psi_r = phix,  psi_theta = -phiy,
    z_B = -z,  d/drho = -d/dx,  d/dtheta = R d/dy

The scale factor of the arc length is ``s = R/rho``, the element of area
``dA = (rho/R) dx dy``. The laminate spans ``d - h/2 <= z <= d + h/2`` with
the offset ``d`` (``Shell.offset``), which enters the stiffness through the
matrix ``ABD`` and the mass through the moments ``I0, I1, I2``.

Kinematics
----------
First-order shear deformation theory (FSDT), as ``plate_fsdt_donnell``:

    u(z) = u + z phix,   v(z) = v + z phiy,   w(z) = w

Strains, von Kármán in polar coordinates (Bagheri, Kiani and Eslami 2019,
Eqs. (2) and (6), in the coordinates above):

    exx = u,x + w,x^2/2
    eyy = s v,y - u/rho + s^2 w,y^2/2
    gxy = s u,y + v,x + v/rho + s w,x w,y
    kxx = phix,x
    kyy = s phiy,y - phix/rho
    kxy = s phix,y + phiy,x + phiy/rho
    gyz = phiy + s w,y
    gxz = phix + w,x

The membrane strains are those of ``annular_clpt_donnell``. With ``phix =
-w,x`` and ``phiy = -s w,y`` the transverse shear strains vanish and the
changes of curvature are those of ``annular_clpt_donnell``.

Checks (run this file):

- :func:`check_linear_strains`: the linear strains follow from the linear
  strains of the three-dimensional FSDT displacement field in the metric of
  the plane;
- :func:`check_bagheri`: the strains are Eqs. (2) and (6) of Bagheri, Kiani
  and Eslami (2019), after the change of variables above;
- :func:`check_rigid_body`: the six small rigid-body motions, with the
  rotations of the rotated normal, give zero linear strains;
- :func:`check_clpt_limit`: with the Kirchhoff rotations the strains are
  those of ``annular_clpt_donnell``;
- :func:`check_plate_limit`: for ``R -> infinity`` the strains are those of
  ``plate_fsdt_donnell``;
- :func:`check_tangent`: the tangent stiffness of the kernels is the
  derivative of the internal force vector.

Kernels
-------
``write_pyx.py`` writes ``panels/models/<model>.pyx`` and
``panels/models/<model>_num.pyx`` from :func:`build`. At each integration
point the fifteen generalized derivatives

    d = (u, u,x, u,y, v, v,x, v,y, w, w,x, w,y,
         phix, phix,x, phix,y, phiy, phiy,x, phiy,y) = [Phi] c

give the strains, ``[B] = d eps/d d`` (8 x 15) and ``[H(N)] = sum_r sigma_r
d^2 eps_r/d d^2``, whose only block is on ``(w,x, w,y)``:

    [H(N)] = [[Nxx, s Nxy], [s Nxy, s^2 Nyy]]

References
----------
Bagheri, H., Kiani, Y., and Eslami, M. R., Buckling of moderately thick
annular plates subjected to torque, Archive of Mechanical Engineering 66(2),
2019, 209-227, Eqs. (1), (2) and (6). doi:10.24425/ame.2019.128445

Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
Cambridge University Press, 2008, Sec. 1.4. doi:10.1017/CBO9780511619694

"""
import os
import sys

import numpy as np
import sympy
from sympy import Function, Matrix as M, cos, diff, simplify, sin, symbols

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


def strains(nonlinear=True):
    """Generalized strains (exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz)"""
    k = 1 if nonlinear else 0
    return M([u_x + k*w_x**2/2,
              s*v_y - u/rho + k*s**2*w_y**2/2,
              s*u_y + v_x + v/rho + k*s*w_x*w_y,
              phix_x,
              s*phiy_y - phix/rho,
              s*phix_y + phiy_x + phiy/rho,
              phiy + s*w_y,
              phix + w_x])


def mass_map():
    """Map ``J`` from ``d`` to ``(u, v, w, phix, phiy)``, the displacement
    ``(u(z), v(z), w(z)) = (u, v, w) + z (phix, phiy, 0)``"""
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
TITLE = ('Annular-sector flat plate, using the first-order shear deformation '
         'theory\n(FSDT) with von Kármán strains in polar coordinates')

#: notes of the generated modules, the strains
STRAIN_NOTES = r"""With `s = r/\rho`, the von Kármán strains in polar coordinates of the
first-order shear deformation theory, Bagheri, Kiani and Eslami [1]_, Eqs.
(2) and (6), are, in these coordinates:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} w_{,x}^2 \qquad
    \varepsilon_{yy} = s v_{,y} - \frac{u}{\rho} + \frac{1}{2} s^2 w_{,y}^2

    \gamma_{xy} = s u_{,y} + v_{,x} + \frac{v}{\rho} + s w_{,x} w_{,y}

    \kappa_{xx} = \phi_{x,x} \qquad
    \kappa_{yy} = s \phi_{y,y} - \frac{\phi_x}{\rho} \qquad
    \kappa_{xy} = s \phi_{x,y} + \phi_{y,x} + \frac{\phi_y}{\rho}

    \gamma_{yz} = \phi_y + s w_{,y} \qquad
    \gamma_{xz} = \phi_x + w_{,x}

with the displacement `u(z) = u + z \phi_x`, `v(z) = v + z \phi_y`, `w(z)
= w`. The membrane strains are those of ``annular_clpt_donnell`` (Amabili
[2]_, Eqs. (1.175)-(1.177)); with `\phi_x = -w_{,x}` and `\phi_y = -s
w_{,y}` the transverse shear strains vanish and the changes of curvature
are those of ``annular_clpt_donnell``. For `r \to \infty` the strains are
those of ``plate_fsdt_donnell``. The second derivatives of the strains with
respect to the generalized derivatives are constant and only on `(w_{,x},
w_{,y})`.
"""

#: references of the generated modules
REFERENCES = r""".. [1] Bagheri, H., Kiani, Y., and Eslami, M. R., Buckling of moderately
   thick annular plates subjected to torque, Archive of Mechanical
   Engineering, Vol. 66, No. 2, 2019, pp. 209-227, Eqs. (1), (2) and (6).
   https://doi.org/10.24425/ame.2019.128445
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
FUNCS = dict(u=U, v=V, w=W, phix=PX, phiy=PY)


def at_point(expr):
    """Replace ``d`` by the derivatives of the functions ``U, V, W, PX,
    PY``"""
    rep = {}
    for name, (fld, nx_, ny_) in D_OF.items():
        e = FUNCS[fld]
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


def check_linear_strains():
    """Linear strains from the 3-D FSDT field in the metric of the plane

    ``P(x, y, z) = X0(x, y) + z e_z`` with ``X0 = (R - x) e_rho``; the plane
    being flat, ``P,x = e_x``, ``P,y = (rho/R) e_y``, ``P,z = e_z`` for every
    ``z``. The linear strains of the displacement ``U3 = u(z) e_x + v(z) e_y
    + w e_z`` are ``e_xx = U3,x . e_x``, ``e_yy = U3,y . e_y/(rho/R)``,
    ``g_xy = U3,x . e_y + U3,y . e_x/(rho/R)``, ``g_yz = U3,z . e_y + U3,y .
    e_z/(rho/R)``, ``g_xz = U3,z . e_x + U3,x . e_z``; the in-plane ones are
    linear in ``z``, the changes of curvature being their derivatives, and
    the transverse shear strains are constant.
    """
    ex, ey, ez = basis()
    sc = (R - x)/R
    U3 = (U + z*PX)*ex + (V + z*PY)*ey + W*ez
    exx = diff(U3, x).dot(ex)
    eyy = diff(U3, y).dot(ey)/sc
    gxy = diff(U3, x).dot(ey) + diff(U3, y).dot(ex)/sc
    gyz = diff(U3, z).dot(ey) + diff(U3, y).dot(ez)/sc
    gxz = diff(U3, z).dot(ex) + diff(U3, x).dot(ez)
    lin = at_point(strains(nonlinear=False))
    got = [exx.subs(z, 0), eyy.subs(z, 0), gxy.subs(z, 0),
           diff(exx, z), diff(eyy, z), diff(gxy, z), gyz, gxz]
    for g, e in zip(got, lin):
        assert simplify(g - e) == 0, (g, e)
    for e in (exx, eyy, gxy):
        assert simplify(diff(e, z, 2)) == 0
    print('check_linear_strains: OK')


def check_bagheri():
    """Strains of Bagheri, Kiani and Eslami (2019), Eqs. (2) and (6)

    Their radial, circumferential and transverse displacements and rotations
    are ``u0 = -U``, ``v0 = V``, ``w0 = -W``, ``psi_r = PX``, ``psi_theta =
    -PY``, with ``r = R - x``, ``theta = y/R`` and their ``z`` along
    ``e_axis = -e_z``: ``exx = eps_rr``, ``eyy = eps_thth``, ``gxy =
    -gamma_rth``, ``kxx = -k_rr``, ``kyy = -k_thth``, ``kxy = k_rth``,
    ``gyz = -gamma_thz``, ``gxz = gamma_rz``.
    """
    r_ = R - x
    u0, v0, w0, pr, pt = -U, V, -W, PX, -PY

    def dr(f):
        return -diff(f, x)

    def dt(f):
        return R*diff(f, y)

    # Eqs. (2): von Kármán membrane strains and transverse shear strains
    err = dr(u0) + dr(w0)**2/2
    ett = u0/r_ + dt(v0)/r_ + dt(w0)**2/(2*r_**2)
    grt = dt(u0)/r_ + dr(v0) - v0/r_ + dr(w0)*dt(w0)/r_
    grz = dr(w0) + pr
    gtz = dt(w0)/r_ + pt
    # Eqs. (6): changes of curvature, the strains of M
    krr = dr(pr)
    ktt = pr/r_ + dt(pt)/r_
    krt = dt(pr)/r_ + dr(pt) - pt/r_
    ref = [err, ett, -grt, -krr, -ktt, krt, -gtz, grz]
    got = at_point(strains())
    for g, e in zip(got, ref):
        assert simplify(g - e) == 0, (g, e)
    print('check_bagheri: OK')


def check_rigid_body():
    """Six small rigid-body motions ``t + o x X0``, with the rotations of
    the normal ``(o x e_z) . e_x`` and ``(o x e_z) . e_y``, give zero linear
    strains"""
    ex, ey, ez = basis()
    t1, t2, t3, o1, o2, o3 = symbols('t1, t2, t3, o1, o2, o3', real=True)
    X0 = (R - x)*(-ex)
    o = M([o1, o2, o3])
    Ug = M([t1, t2, t3]) + o.cross(X0)
    dn = o.cross(ez)
    rep = {U: Ug.dot(ex), V: Ug.dot(ey), W: Ug.dot(ez), PX: dn.dot(ex),
           PY: dn.dot(ey)}
    lin = at_point(strains(nonlinear=False))
    for e in lin:
        assert simplify(e.subs(rep).doit()) == 0, e
    print('check_rigid_body: OK')


def check_clpt_limit():
    """``phix = -w,x``, ``phiy = -s w,y``: zero transverse shear strains and
    the strains of ``annular_clpt_donnell``"""
    clpt = M([u_x + w_x**2/2, s*v_y - u/rho + s**2*w_y**2/2,
              s*u_y + v_x + v/rho + s*w_x*w_y,
              -diff(W, x, 2), w_x/rho - s**2*diff(W, y, 2),
              -2*s*(diff(W, x, y) + w_y/rho), 0, 0])
    rep = {PX: -diff(W, x), PY: -R/(R - x)*diff(W, y)}
    got = at_point(strains()).subs(rep).doit()
    ref = at_point(clpt)
    for g, e in zip(got, ref):
        assert simplify(g - e) == 0, (g, e)
    print('check_clpt_limit: OK')


def check_plate_limit():
    """``R -> infinity``: the strains of ``plate_fsdt_donnell``"""
    plate = M([u_x + w_x**2/2, v_y + w_y**2/2, u_y + v_x + w_x*w_y,
               phix_x, phiy_y, phix_y + phiy_x, phiy + w_y, phix + w_x])
    eps = strains().subs(rho, R - x)
    for g, e in zip(eps, plate):
        assert sympy.limit(g, R, sympy.oo) == e
    print('check_plate_limit: OK')


def check_tangent(seed=0):
    """``[B]^T [F] [B] + [H(sigma)]`` is the derivative of ``[B]^T sigma``"""
    rng = np.random.default_rng(seed)
    eps = strains()
    B = eps.jacobian(M(D))
    sig = F*eps
    fint = B.T*sig
    KT = fint.jacobian(M(D))
    H = hessian_stress(list(sig))
    rep = {sym: rng.random() for sym in F.free_symbols}
    rep.update({d: rng.random() - 0.5 for d in D})
    rep.update({R: 2., rho: 1.3})
    diffm = (KT - B.T*F*B - H).subs(rep)
    assert max(abs(float(e)) for e in diffm) < 1e-12
    print('check_tangent: OK')


if __name__ == '__main__':
    check_linear_strains()
    check_bagheri()
    check_rigid_body()
    check_clpt_limit()
    check_plate_limit()
    check_tangent()
