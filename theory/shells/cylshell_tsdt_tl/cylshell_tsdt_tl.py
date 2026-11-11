r"""Cylindrical shell, total-Lagrangian TSDT (``cylshell_tsdt_tl``)

Derivation of the strains, of their derivatives and of the matrices at an
integration point, with symbolic checks, run ``python cylshell_tsdt_tl.py``.
The kernels are written by ``write_pyx.py``.

Kinematics
----------
Third-order shear deformation theory (TSDT) of Reddy of a circular cylinder
of radius `r`, `x` along the axis, `y = r \theta` the arc length, `z` along
the outward normal, shallow through the thickness, `r + z \approx r`, as in
``cylshell_tsdt_sanders`` and ``cylshell_tsdt_gl``. Degrees of freedom of a
term: `u, v, w, \phi_x, \phi_y`. Local basis `(\mathbf{e}_x, \mathbf{e}_y,
\mathbf{e}_z)` with `\mathbf{e}_{y,y} = -\mathbf{e}_z/r`, `\mathbf{e}_{z,y} =
\mathbf{e}_y/r`; the derivative along `y` of a vector of components `(a_x,
a_y, a_z)` has the components `(a_{x,y}, a_{y,y} + a_z/r, a_{z,y} - a_y/r)`.

The first-order part is that of ``cylshell_fsdt_tl``: the deformed
mid-surface `\mathbf{X}` and the second-order director `\mathbf{d} = (\psi_x,
\psi_y, 1 - (\psi_x^2 + \psi_y^2)/2)`, `\psi_x = \phi_x`, `\psi_y = \phi_y +
v/r`, with

.. math::

    \mathbf{X}_{,x} = (1 + u_{,x}, \; v_{,x}, \; w_{,x}) \qquad
    \mathbf{X}_{,y} = (u_{,y}, \; 1 + v_{,y} + w/r, \; w_{,y} - v/r)

    \mathbf{X}_{,xx} = (u_{,xx}, \; v_{,xx}, \; w_{,xx}) \qquad
    \mathbf{X}_{,xy} = (u_{,xy}, \; v_{,xy} + w_{,x}/r, \; w_{,xy} -
    v_{,x}/r)

    \mathbf{X}_{,yy} = \left(u_{,yy}, \; v_{,yy} + \frac{2 w_{,y}}{r} -
    \frac{v}{r^2}, \; w_{,yy} - \frac{2 v_{,y}}{r} - \frac{1}{r} -
    \frac{w}{r^2}\right)

As for ``plate_tsdt_tl``, the third-order term is built with the director
and the normal of the deformed mid-surface, not normalized, `\mathbf{N} =
\mathbf{X}_{,x} \times \mathbf{X}_{,y}` (components in the local basis):
the deformed position of a point is

.. math::

    \mathbf{r}(z) = \mathbf{X} + z \mathbf{d} + z^3 \mathbf{q}, \qquad
    \mathbf{q} = -c_1 (\mathbf{d} - \mathbf{N}), \qquad
    c_1 = \frac{4}{3 h^2}

whose linear part is Reddy's field with Sanders' rotation, the linear part
of `\mathbf{d} - \mathbf{N}` being `(\phi_x + w_{,x}, \; \phi_y + w_{,y},
\; -(u_{,x} + v_{,y} + w/r))`. The strains: membrane, `\kappa` and
`\gamma` of ``cylshell_fsdt_tl``, and

.. math::

    \kappa^{(3)}_{xx} = \mathbf{X}_{,x} \cdot \mathbf{q}_{,x} \qquad
    \kappa^{(3)}_{yy} = \mathbf{X}_{,y} \cdot \mathbf{q}_{,y} -
    \frac{c_1}{r} (\varepsilon_{xx} + \varepsilon_{yy}) \qquad
    \kappa^{(3)}_{xy} = \mathbf{X}_{,x} \cdot \mathbf{q}_{,y} +
    \mathbf{X}_{,y} \cdot \mathbf{q}_{,x}

    \gamma^{(2)}_\alpha = 3 \mathbf{X}_{,\alpha} \cdot \mathbf{q} = -3 c_1
    \gamma_\alpha

Without the term `-c_1 (\varepsilon_{xx} + \varepsilon_{yy})/r`, these are
the coefficients of `z^3` of the Green-Lagrange strains of `\mathbf{r}(z)`
with the reference vectors `\mathbf{e}_x`, `(1 + z/r) \mathbf{e}_y`, and of
`z^2` of `2 E_{\alpha 3}` (``check_strain_measures``). The `z` component of
`\mathbf{N}`, `1 + \varepsilon_{xx} + \varepsilon_{yy}` to first order (the
stretch of the area), enters `\mathbf{q}_{,y}` through the curvature of the
reference, `\mathbf{e}_{z,y} = \mathbf{e}_y/r`, and adds `c_1 (u_{,x} +
v_{,y} + w/r)/r` to the linear part of `\kappa^{(3)}_{yy}`; the term with
the Green-Lagrange membrane strains removes it, as the Koiter terms of
``cylshell_fsdt_tl`` remove the membrane strains from the linear
curvatures, and the linear part is that of ``cylshell_tsdt_sanders``
(``check_linear_part``). Built from dot products and Green-Lagrange
strains, the strains change under a rigid rotation only through the
second-order director (``check_rigid_rotation``). Since
`\mathbf{X}_{,\alpha} \cdot \mathbf{N} = 0`, the transverse shear strain
`\gamma_\alpha (1 - 4 z^2/h^2)` vanishes at the faces for any deformation
(``check_face_shear``).

At a point the strains depend on the 23 generalized derivatives

    D = (u,x, u,y, v,x, v,y, w,x, w,y, v, w, phix, phiy,
         phix,x, phix,y, phiy,x, phiy,y, w,xx, w,yy, w,xy,
         u,xx, u,yy, u,xy, v,xx, v,yy, v,xy) = [Phi] c

and the matrices are those of ``plate_tsdt_tl``.

"""
import importlib.util
import math
import os
import random

import sympy
from sympy import (Matrix as M, Symbol, symbols, cos, sin, expand, zeros,
                   eye, diff)

HERE = os.path.dirname(os.path.abspath(__file__))

a, b, r = symbols('a, b, r', positive=True)

#: generalized derivatives at a point, physical coordinates
(u_x, u_y, v_x, v_y, w_x, w_y, v, w, phix, phiy, phix_x, phix_y, phiy_x,
 phiy_y, w_xx, w_yy, w_xy, u_xx, u_yy, u_xy, v_xx, v_yy, v_xy) = symbols(
     'u_x, u_y, v_x, v_y, w_x, w_y, v, w, phix, phiy, phix_x, phix_y, '
     'phiy_x, phiy_y, w_xx, w_yy, w_xy, u_xx, u_yy, u_xy, v_xx, v_yy, v_xy')
D = (u_x, u_y, v_x, v_y, w_x, w_y, v, w, phix, phiy, phix_x, phix_y, phiy_x,
     phiy_y, w_xx, w_yy, w_xy, u_xx, u_yy, u_xy, v_xx, v_yy, v_xy)
#: names of the generalized derivatives in the kernels
D_NAMES = tuple(s.name for s in D)
#: field and orders of the derivatives along x and y
D_DEF = {nm: (nm.partition('_')[0], nm.partition('_')[2].count('x'),
              nm.partition('_')[2].count('y')) for nm in D_NAMES}
D_DEF.update({'phix': ('phix', 0, 0), 'phiy': ('phiy', 0, 0),
              'v': ('v', 0, 0), 'w': ('w', 0, 0)})
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
#: the constant of the TSDT, 4/(3 h^2)
c1 = Symbol('c1')
Nxx, Nyy, Nxy = symbols('Nxx, Nyy, Nxy')
STRESSES = (Nxx, Nyy, Nxy) + symbols('Mxx, Myy, Mxy, Pxx, Pyy, Pxy, Qy, Qx, '
                                     'Ry, Rx')

#: constitutive matrix of the TSDT, that of ``cylshell_tsdt_gl``
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


def tangents():
    r"""``X,x``, ``X,y`` of the deformed mid-surface in the local basis"""
    return M([1 + u_x, v_x, w_x]), M([u_y, 1 + v_y + w/r, w_y - v/r])


def second_derivatives():
    r"""``X,xx``, ``X,yy``, ``X,xy`` in the local basis"""
    return (M([u_xx, v_xx, w_xx]),
            M([u_yy, v_yy + 2*w_y/r - v/r**2, w_yy - 2*v_y/r - 1/r - w/r**2]),
            M([u_xy, v_xy + w_x/r, w_xy - v_x/r]))


def director():
    r"""Second-order director ``d`` and its derivatives ``d,x``, ``d,y`` in
    the local basis"""
    px, py = phix, phiy + v/r
    px_x, px_y = phix_x, phix_y
    py_x, py_y = phiy_x + v_x/r, phiy_y + v_y/r
    pz = 1 - (px**2 + py**2)/2
    pz_x = -(px*px_x + py*py_x)
    pz_y = -(px*px_y + py*py_y)
    return (M([px, py, pz]), M([px_x, py_x, pz_x]),
            M([px_y, py_y + pz/r, pz_y - py/r]))


def normal():
    r"""``N = X,x x X,y`` and ``N,x``, ``N,y`` in the local basis"""
    Xx, Xy = tangents()
    Xxx, Xyy, Xxy = second_derivatives()
    return (Xx.cross(Xy), Xxx.cross(Xy) + Xx.cross(Xxy),
            Xxy.cross(Xy) + Xx.cross(Xyy))


def membrane():
    r"""Green-Lagrange membrane strains of the mid-surface"""
    Xx, Xy = tangents()
    return M([(Xx.dot(Xx) - 1)/2, (Xy.dot(Xy) - 1)/2, Xx.dot(Xy)])


def bending_shear():
    r"""Curvatures and transverse shear strains of ``cylshell_fsdt_tl``,
    with the Koiter terms ``-eyy/r`` and ``-gxy/(2r)``"""
    Xx, Xy = tangents()
    d, d_x, d_y = director()
    em = membrane()
    return M([Xx.dot(d_x),
              Xy.dot(d_y) - 1/r - em[1]/r,
              Xx.dot(d_y) + Xy.dot(d_x) - em[2]/(2*r),
              Xy.dot(d),
              Xx.dot(d)])


def third_order():
    r"""``(kappa3, gamma2)``, with the term ``-c1 (exx + eyy)/r`` in
    ``kyy3``"""
    Xx, Xy = tangents()
    d, d_x, d_y = director()
    N, N_x, N_y = normal()
    q, q_x, q_y = -c1*(d - N), -c1*(d_x - N_x), -c1*(d_y - N_y)
    em = membrane()
    return (M([Xx.dot(q_x), Xy.dot(q_y) - c1*(em[0] + em[1])/r,
               Xx.dot(q_y) + Xy.dot(q_x)]),
            M([3*Xy.dot(q), 3*Xx.dot(q)]))


def strains(nonlinear=True):
    r"""``eps = (exx, eyy, gxy, kxx, kyy, kxy, kxx3, kyy3, kxy3, gyz, gxz,
    gyz2, gxz2)``; with ``nonlinear=False`` the linear part"""
    bs = bending_shear()
    k3, g2 = third_order()
    eps = M(list(membrane()) + list(bs[:3]) + list(k3) + list(bs[3:])
            + list(g2)).applyfunc(expand)
    if nonlinear:
        return eps
    t = Symbol('t')
    return eps.subs({s: t*s for s in D}, simultaneous=True).applyfunc(
        lambda e: expand(e).coeff(t, 1)).applyfunc(expand)


def basis(side):
    r"""``[Phi]`` of a term of side ``'A'`` or ``'B'``: the 23 x 5 matrix such
    that ``D = [Phi] c``, with the approximation functions in the notation
    of the kernels, e.g. ``fAu_xixi`` for `\partial^2 f_i^u/\partial \xi^2`"""
    P = zeros(len(D), DOF)
    for p, name in enumerate(D_NAMES):
        field, dxi, deta = D_DEF[name]
        f = Symbol('f{0}{1}{2}'.format(side, field, '_' + 'xi'*dxi if dxi else ''))
        g = Symbol('g{0}{1}{2}'.format(side, field, '_' + 'eta'*deta if deta else ''))
        P[p, FIELDS.index(field)] = (2/a)**dxi*(2/b)**deta*f*g
    return P


def hessian_stress(S, eps=None):
    r"""``[H(S, D)] = sum_r S_r d^2 eps_r/d D^2``, linear in ``D``"""
    eps = strains() if eps is None else eps
    H = zeros(len(D), len(D))
    for k in range(NE):
        H += S[k]*sympy.hessian(eps[k], D)
    return H.applyfunc(expand)


def build():
    r"""Dict with ``'eps'``, ``'eps_L'``, ``'B'``, ``'F'``, ``'H'``,
    ``'H0'`` (``'H'`` at ``D = 0``), ``'PhiA'``, ``'PhiB'``"""
    eps = strains()
    eps_L = strains(nonlinear=False)
    B = eps.jacobian(D)
    H = hessian_stress(STRESSES, eps)
    H0 = H.subs({s: 0 for s in D})
    return dict(eps=eps, eps_L=eps_L, B=B, F=F, H=H, H0=H0,
                PhiA=basis('A'), PhiB=basis('B'))


def _load(name, *path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, *path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _embedding(funcs, xs, th):
    r"""Cartesian position, local basis and director of the cylinder `X_0 =
    (x, r \sin\theta, r \cos\theta)`, `\mathbf{e}_y = (0, \cos\theta,
    -\sin\theta)`, `\mathbf{e}_z = (0, \sin\theta, \cos\theta)`, for the
    fields ``funcs = (U, V, W, PX, PY)`` of ``x`` and ``theta``"""
    U, V, W, PX, PY = funcs
    ex = M([1, 0, 0])
    ey = M([0, cos(th), -sin(th)])
    ez = M([0, sin(th), cos(th)])
    X = M([xs, r*sin(th), r*cos(th)]) + U*ex + V*ey + W*ez
    psx, psy = PX, PY + V/r
    dd = psx*ex + psy*ey + (1 - (psx**2 + psy**2)/2)*ez
    return X, dd, ex, ey, ez


def _point_values(funcs, xs, th, pt):
    r"""Generalized derivatives of the fields ``funcs`` at the point ``pt``"""
    U, V, W, PX, PY = funcs
    dy = lambda f: diff(f, th)/r
    dx = lambda f: diff(f, xs)
    exprs = {u_x: dx(U), u_y: dy(U), v_x: dx(V), v_y: dy(V), w_x: dx(W),
             w_y: dy(W), v: V, w: W, phix: PX, phiy: PY, phix_x: dx(PX),
             phix_y: dy(PX), phiy_x: dx(PY), phiy_y: dy(PY),
             w_xx: dx(dx(W)), w_yy: dy(dy(W)), w_xy: dx(dy(W)),
             u_xx: dx(dx(U)), u_yy: dy(dy(U)), u_xy: dx(dy(U)),
             v_xx: dx(dx(V)), v_yy: dy(dy(V)), v_xy: dx(dy(V))}
    return {k: e.subs(pt) for k, e in exprs.items()}


def check_strain_measures(seed=0):
    r"""With polynomial fields of random coefficients and the Cartesian
    embedding of :func:`_embedding`, at a random point: the strains without
    the Koiter terms (``-eyy/r``, ``-gxy/(2r)``) and without ``-c1 (exx +
    eyy)/r`` are the coefficients of `z^0`, `z`, `z^3` of the Green-Lagrange
    strains of `\mathbf{X} + z \mathbf{d} + z^3 \mathbf{q}` with the
    reference vectors `\mathbf{e}_x`, `(1 + z/r) \mathbf{e}_y`, and of
    `z^0`, `z^2` of `2 E_{\alpha 3}`; `\mathbf{N}` is the Cartesian
    `\mathbf{X}_{,x} \times \mathbf{X}_{,y}`, `\mathbf{X}_{,y} =
    \mathbf{X}_{,\theta}/r`"""
    rnd = random.Random(seed)
    xs, th, z = symbols('x_s, theta, z', real=True)
    rr = sympy.Rational(13, 10)
    funcs = []
    for _ in range(5):
        funcs.append(sum(sympy.Rational(rnd.randint(-9, 9), 40)*xs**i*th**j
                         for i in range(3) for j in range(3) if i + j <= 3))
    X, dd, ex, ey, ez = [e.subs(r, rr) for e in _embedding(funcs, xs, th)]
    Xx, Xy = diff(X, xs), diff(X, th)/rr
    N = Xx.cross(Xy)
    q = -c1*(dd - N)
    pos = X + z*dd + z**3*q
    Gx, Gy, Gz = diff(pos, xs), diff(pos, th)/rr, diff(pos, z)
    pt = {xs: sympy.Rational(3, 10), th: sympy.Rational(-2, 10)}
    E = [((Gx.dot(Gx) - 1)/2).subs(pt), ((Gy.dot(Gy) - (1 + z/rr)**2)/2).subs(pt),
         Gx.dot(Gy).subs(pt)]
    S = [Gy.dot(Gz).subs(pt), Gx.dot(Gz).subs(pt)]
    vals = _point_values([f.subs(r, rr) for f in funcs], xs, th, pt)
    vals = {k: e.subs(r, rr) for k, e in vals.items()}
    vals[r] = rr
    eps = [expand(e.xreplace(vals)) for e in strains()]
    E = [sympy.Poly(expand(e), z) for e in E]
    S = [sympy.Poly(expand(e), z) for e in S]
    coef = lambda P, k: P.coeff_monomial(z**k)
    # exact values with sines and cosines of the point, compared numerically
    zero = lambda e: abs(sympy.N(sympy.sympify(e).subs(c1, sympy.Rational(7, 10)), 40)) < 1e-30
    koiter = [0, eps[1]/rr, eps[2]/(2*rr)]
    k3 = [0, c1*(eps[0] + eps[1])/rr, 0]
    for k in range(3):
        assert zero(coef(E[k], 0) - eps[k])
        assert zero(coef(E[k], 1) - eps[3 + k] - koiter[k])
        assert zero(coef(E[k], 3) - eps[6 + k] - k3[k])
    for k in range(2):
        assert zero(coef(S[k], 0) - eps[9 + k])
        assert zero(coef(S[k], 2) - eps[11 + k])
    return True


def check_linear_part():
    r"""``[B(0)] [Phi]`` equal to the linear operator ``B0`` of
    ``cylshell_tsdt_sanders``; the linear strains and the membrane strains
    those of ``cylshell_tsdt_gl``; the first-order strains those of
    ``cylshell_fsdt_tl``; without ``-c1 (exx + eyy)/r`` the linear part of
    ``kyy3`` would have ``c1 (u,x + v,y + w/r)/r`` more"""
    lm = _load('cylshell_tsdt_sanders', '..', 'cylshell_tsdt_sanders',
               'cylshell_tsdt_sanders.py')
    m = build()
    BL = m['B'].subs({s: 0 for s in D})
    sub = {lm.r: r, lm.a: a, lm.b: b}
    if hasattr(lm, 'c1'):
        sub[lm.c1] = c1
    B0 = lm.shape_functions('A')['B0'].xreplace(sub)
    assert (BL*m['PhiA'] - B0).applyfunc(expand) == zeros(NE, DOF)
    gl = _load('cylshell_tsdt_gl', '..', 'cylshell_tsdt_gl', 'cylshell_tsdt_gl.py')
    glsub = {getattr(gl, s.name): s for s in gl.D}
    glsub.update({gl.c1: c1, gl.r: r})
    assert (gl.strains(nonlinear=False).xreplace(glsub) - m['eps_L']).applyfunc(expand) == zeros(NE, 1)
    assert (gl.strains().xreplace(glsub)[:3, :] - m['eps'][:3, :]).applyfunc(expand) == zeros(3, 1)
    ft = _load('cylshell_fsdt_tl', '..', 'cylshell_fsdt_tl', 'cylshell_fsdt_tl.py')
    ftsub = {getattr(ft, s.name): s for s in ft.D}
    ftsub[ft.r] = r
    e8 = ft.strains().xreplace(ftsub)
    for k, j in enumerate((0, 1, 2, 3, 4, 5, 9, 10)):
        assert expand(m['eps'][j] - e8[k]) == 0
    t = Symbol('t')
    lin = lambda e: expand(expand(e.subs({s: t*s for s in D},
                                         simultaneous=True)).coeff(t, 1))
    Xx, Xy = tangents()
    d, d_x, d_y = director()
    N, N_x, N_y = normal()
    raw = Xy.dot(-c1*(d_y - N_y))
    assert expand(lin(raw) - m['eps_L'][7] - c1*(u_x + v_y + w/r)/r) == 0
    return True


def check_face_shear():
    r"""`\gamma_\alpha + (h^2/4) \gamma^{(2)}_\alpha`, `h^2 = 4/(3 c_1)`,
    zero for any deformation"""
    eps = strains()
    assert [expand(eps[9 + k] + eps[11 + k]/(3*c1)) for k in range(2)] == [0, 0]
    return True


def _rigid_values(rnd, tval, bend=0.):
    r"""Generalized derivatives at a random point of a rigid rotation
    ``tval`` about a random axis through a random point of the cylinder,
    optionally bent before by ``w = bend x^2/2`` (then at ``x = 0``), with
    the director components those of the rotated normal; also the radius"""
    xs, th = symbols('x_s, theta', real=True)
    nn = [rnd.uniform(-1, 1) for _ in range(3)]
    norm = sum(c**2 for c in nn)**0.5
    n1, n2, n3 = [c/norm for c in nn]
    K = M([[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]])
    R = eye(3) + sin(tval)*K + (1 - cos(tval))*K*K
    rr = rnd.uniform(0.5, 2.)
    ex = M([1, 0, 0])
    ey = M([0, cos(th), -sin(th)])
    ez = M([0, sin(th), cos(th)])
    X0 = M([xs, rr*sin(th), rr*cos(th)])
    p = M([rnd.uniform(-1, 1) for _ in range(3)])
    # the bent state: w = bend x^2/2 with the director its normal
    Xb = X0 + bend*xs**2/2*ez
    nb = (ez - bend*xs*ex)/sympy.sqrt(1 + bend**2*xs**2)
    disp = R*(Xb - p) + p - X0
    nrm = R*nb
    U, V, W = disp.dot(ex), disp.dot(ey), disp.dot(ez)
    PX = nrm.dot(ex)
    PY = nrm.dot(ey) - V/rr
    # the bent state at x = 0, where its normal is e_z: the error of the
    #     second-order director is then that of the rotation alone
    pt = {xs: rnd.uniform(-1, 1) if bend == 0 else 0., th: rnd.uniform(-3, 3)}
    vals = _point_values([U, V, W, PX, PY], xs, th, pt)
    out = {k: float(e.subs(r, rr)) for k, e in vals.items()}
    out[r] = rr
    return out


def check_rigid_rotation():
    r"""Rigid rotations `t` of the undeformed cylinder, and of a state bent
    along the axis, about random axes, with the director components those
    of the rotated normal: the membrane strains do not change, the other
    strains change by `O(t^4)` (fitted order between `t = 0.1` and `0.05` at
    least 3.8); the linear third-order strains of ``cylshell_tsdt_gl``
    change by `O(t)` or `O(t^2)`, at least a hundred times more"""
    gl = _load('cylshell_tsdt_gl', '..', 'cylshell_tsdt_gl', 'cylshell_tsdt_gl.py')
    glsub = {getattr(gl, s.name): s for s in gl.D}
    glsub.update({gl.c1: c1, gl.r: r})
    eps = strains().xreplace({c1: 1})
    eps_gl = gl.strains().xreplace(glsub).xreplace({c1: 1})
    orders = []
    for bend in (0., 0.4):
        for k in range(4):
            res, res_gl = [], []
            for tval in (0.1, 0.05):
                ev = lambda ee, vals: [float(x.xreplace(vals)) for x in ee]
                v0 = _rigid_values(random.Random(k), 0., bend)
                vt = _rigid_values(random.Random(k), tval, bend)
                e0, et = ev(eps, v0), ev(eps, vt)
                g0, gt = ev(eps_gl, v0), ev(eps_gl, vt)
                dif = [abs(p - q) for p, q in zip(et, e0)]
                assert max(dif[:3]) < 1e-12
                res.append(max(dif[3:]))
                res_gl.append(max(abs(p - q) for p, q in zip(gt[6:9], g0[6:9])))
            o, o_gl = math.log2(res[0]/res[1]), math.log2(res_gl[0]/res_gl[1])
            assert o > 3.8, res
            assert o_gl < 2.2 and res_gl[1] > 100*res[1], res_gl
            orders.append((bend, o, o_gl))
    return orders


def check_tangent():
    r"""``d(B^T sigma)/dD = B^T F B + H(sigma, D)``, ``fkG_num``
    homogeneous of degree one, and ``H0`` with membrane stresses only equal
    to the Hessian of ``cylshell_tsdt_gl`` (whose analytical ``fkG0`` the
    model uses)"""
    m = build()
    eps, B, H, H0 = m['eps'], m['B'], m['H'], m['H0']
    nd = len(D)
    sigma = F*eps
    exact = (B.T*sigma).applyfunc(expand).jacobian(D).applyfunc(expand)
    KT = (B.T*F*B + H.xreplace(dict(zip(STRESSES, sigma)))).applyfunc(expand)
    assert (exact - KT).applyfunc(expand) == zeros(nd, nd)
    KG = H0.xreplace(dict(zip(STRESSES, F*m['eps_L'])))
    tt = Symbol('tt')
    KGs = KG.subs({dd: tt*dd for dd in D}, simultaneous=True)
    assert (KGs - tt*KG).applyfunc(expand) == zeros(nd, nd)
    gl = _load('cylshell_tsdt_gl', '..', 'cylshell_tsdt_gl', 'cylshell_tsdt_gl.py')
    Hgl = gl.build()['H']
    idx = [D_NAMES.index(s.name) for s in gl.D]
    sub = {getattr(gl, s.name): s for s in STRESSES[:3]}
    sub[gl.r] = r
    H0m = H0.xreplace({s: 0 for s in STRESSES[3:]})
    for i, p in enumerate(idx):
        for j, q in enumerate(idx):
            assert expand(H0m[p, q] - Hgl[i, j].xreplace(sub)) == 0
    rest = [p for p in range(nd) if p not in idx]
    assert all(H0m[p, q] == 0 for p in rest for q in range(nd))
    return True


def check_numerical(seed=0):
    r"""Central differences of ``fint`` of a pair of terms against
    ``[Phi_A]^T K_T [Phi_B]``"""
    import numpy as np
    rnd = random.Random(seed)
    m = build()
    vals = {s: rnd.uniform(-1, 1) for s in F.free_symbols}
    vals.update({a: 0.7, b: 0.4, c1: 0.6, r: 0.9})
    for s in m['PhiA'].free_symbols | m['PhiB'].free_symbols:
        vals.setdefault(s, rnd.uniform(-1, 1))
    PhiA, PhiB, Fn = [x.xreplace(vals) for x in (m['PhiA'], m['PhiB'], F)]
    eps, Bn, Hn = [m[k].xreplace({c1: vals[c1], r: vals[r]}) for k in ('eps', 'B', 'H')]
    fe = sympy.lambdify(D, eps, 'numpy')
    fB = sympy.lambdify(D, Bn, 'numpy')
    PA, PB, Fm = [np.array(x.tolist(), dtype=float) for x in (PhiA, PhiB, Fn)]
    cA = np.array([rnd.uniform(-0.3, 0.3) for _ in range(DOF)])
    cB = np.array([rnd.uniform(-0.3, 0.3) for _ in range(DOF)])

    def fint_A(cB_):
        dd = PA@cA + PB@cB_
        return PA.T@np.array(fB(*dd), dtype=float).T@(Fm@np.array(fe(*dd), dtype=float).ravel())
    dd = PA@cA + PB@cB
    sub = dict(zip(D, dd))
    sig = Fm@np.array(fe(*dd), dtype=float).ravel()
    Bv = np.array(fB(*dd), dtype=float)
    Hv = np.array(Hn.xreplace(dict(zip(STRESSES, sig))).xreplace(sub).tolist(), dtype=float)
    KAB = PA.T@(Bv.T@Fm@Bv + Hv)@PB
    h = 1e-6
    for e in range(DOF):
        dc = np.zeros(DOF)
        dc[e] = h
        fd = (fint_A(cB + dc) - fint_A(cB - dc))/(2*h)
        assert np.allclose(fd, KAB[:, e], rtol=1e-6, atol=1e-6)
    return True


def degrees():
    r"""Degrees along ``x`` and ``y`` of the integrands of ``K_T``, each as
    ``(k, j)`` for ``k p + j`` with approximation functions of degree
    ``p``"""
    m = build()
    KT = m['B'].T*F*m['B'] + m['H'].xreplace(dict(zip(STRESSES, F*m['eps'])))
    out = []
    for axis in (1, 2):
        deg = {s: -D_DEF[s.name][axis] for s in D}
        best = None
        for p in range(len(D)):
            for q in range(len(D)):
                e = expand(KT[p, q])
                if e == 0:
                    continue
                for mono in sympy.Poly(e, *D).monoms():
                    k = sum(mono) + 2
                    j = sum(n*deg[s] for n, s in zip(mono, D)) + deg[D[p]] + deg[D[q]]
                    if best is None or (k, j) > best:
                        best = (k, j)
        out.append(best)
    return out


def check_integrand_degree():
    r"""Degree ``6p`` along both directions, as ``cylshell_fsdt_tl``"""
    assert degrees() == [(6, 0), (6, 0)], degrees()
    return True


CHECKS = (check_strain_measures, check_linear_part, check_face_shear,
          check_rigid_rotation, check_tangent, check_numerical,
          check_integrand_degree)


if __name__ == '__main__':
    for check in CHECKS:
        res = check()
        print(check.__name__, 'verified')
        if isinstance(res, list):
            for row in res:
                print('    bend {0}: order {1:.2f}, cylshell_tsdt_gl {2:.2f}'.format(*row))
