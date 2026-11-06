r"""Cylindrical shell, total-Lagrangian Kirchhoff-Love (CLPT) with the deformed normal

Kinematics
----------
Cylinder of radius ``r``, ``x`` along the axis, ``y = r theta`` the arc
length along the circumference, ``z`` along the outward normal; the local
orthonormal basis of the undeformed mid-surface is ``(e_x, e_y, e_z)``, with
``e_y,y = -e_z/r`` and ``e_z,y = e_y/r``. The degrees of freedom of a term of
the Ritz series are ``u, v, w``, the displacements of the mid-surface along
``e_x, e_y, e_z``, as in ``cylshell_clpt_sanders`` and ``cylshell_clpt_gl``.
Assumptions:

- Kirchhoff: the normal stays straight and normal, the rotations derive from
  ``u, v, w``, the transverse shear strains vanish (CLPT);
- shallow through the thickness, ``r + z ~ r``, as in the other cylinders
  of panels;
- small strains, the constitutive law linear in the Green-Lagrange strains;
- membrane strains: the complete Green-Lagrange strains of the mid-surface,
  those of ``cylshell_clpt_gl``;
- changes of curvature: those of the second fundamental form of the
  deformed mid-surface, measured with its area vector, plus terms of the
  order of the membrane strains divided by ``r`` that make their linear
  part the linear changes of curvature of Sanders (1963), those of
  ``cylshell_clpt_sanders`` and ``cylshell_clpt_gl``.

Position and derivatives
------------------------
The deformed mid-surface is ``X = X0 + u e_x + v e_y + w e_z``. In the local
basis (:func:`check_position_derivatives`)::

    X,x  = (1 + u,x, v,x, w,x)
    X,y  = (u,y, 1 + v,y + w/r, w,y - v/r)
    X,xx = (u,xx, v,xx, w,xx)
    X,xy = (u,xy, v,xy + w,x/r, w,xy - v,x/r)
    X,yy = (u,yy, v,yy + 2 w,y/r - v/r^2, w,yy - 2 v,y/r - w/r^2 - 1/r)

The membrane strains are ``E = (F^T F - I)/2`` with ``F = [X,x  X,y]``::

    exx = u,x + (u,x^2 + v,x^2 + w,x^2)/2
    eyy = v,y + w/r + (u,y^2 + (v,y + w/r)^2 + (w,y - v/r)^2)/2
    gxy = u,y + v,x + u,x u,y + v,x (v,y + w/r) + w,x (w,y - v/r)

the middle-surface strains of the Flügge-Lur'e-Byrne non-linear theory as
given by Amabili (2008), Eqs. (1.118)-(1.120), those of
``cylshell_clpt_gl``.

Changes of curvature
--------------------
The second fundamental form of a surface is ``b_ab = n . X,ab``, ``n`` its
unit normal (Koiter's notation, van der Heijden 2008, Sec. 3.16, p. 182);
for the
undeformed cylinder ``b0_xx = b0_xy = 0``, ``b0_yy = e_z . X0,yy = -1/r``.
With the area vector ``a = X,x x X,y = |a| n``, ``|a| = 1 + exx + eyy +
O(eps^2)`` (:func:`check_area_vector_norm`), the changes of curvature are::

    kxx = -a . X,xx
    kyy = -(a . X,yy - (1 + exx + eyy) b0_yy) - eyy/r
        = -a . X,yy - 1/r - (exx + 2 eyy)/r
    kxy = -2 a . X,xy - gxy/(2 r)

- ``-(a . X,ab - (1 + exx + eyy) b0_ab)`` is ``-(b_ab - b0_ab)``, the change
  of the second fundamental form, multiplied by the area stretch ``|a|``,
  with ``|a|`` replaced by its first-order part
  in the reference term, which keeps the strains polynomial in the
  displacement derivatives; for a flat plate it is ``-a . X,ab``, the
  curvatures of ``plate_clpt_tl``;
- ``-eyy/r`` and ``-gxy/(2 r)`` turn the linear part into Sanders' linear
  changes of curvature (Sanders 1963; Koiter's form, van der Heijden 2008,
  Eq. (3.16.8), p. 183, for the cylinder Eq. (3.17.3), p. 202), which
  contain the rotation about the normal instead of the membrane strains;
  changes of curvature that differ by terms of the order of the membrane
  strain divided by the radius are equivalent within the accuracy of the
  classical theory of thin shells (a result of Koiter, as given by van der
  Heijden 2008, p. 202). The derivation, step by step from the position of
  the deformed mid-surface, is checked equation by equation in
  ``sanders_linear_part.py``.

The linear part is (:func:`check_linear_part`)::

    kxx = -w,xx
    kyy = -w,yy + v,y/r
    kxy = -2 w,xy + (3/2) v,x/r - (1/2) u,y/r

so that the linear analyses (static, buckling with a membrane prebuckling
state, vibration) are those of ``cylshell_clpt_sanders`` and
``cylshell_clpt_gl``. Every term of the strains is built from ``a``, ``X,ab``
and the Green-Lagrange strains, which do not change when a deformed state is
rotated rigidly, ``X -> R X``: the six strains are invariant under any rigid
rotation of any deformed state (:func:`check_rigid_rotation`), and vanish for
any rigid motion of the cylinder.

Comparison with Novozhilov's non-linear curvatures
--------------------------------------------------
Amabili (2008), Sec. 1.3.7, Eqs. (1.130)-(1.132), gives the non-linear
changes of curvature of an improved Novozhilov theory, the coefficients of
``z`` of the three-dimensional Green-Lagrange strains of the displacements
(1.124)-(1.129), quadratic in the displacements. Term by term, without
imperfection, Amabili's minus the curvatures above (:func:`check_novozhilov`):

- ``kxx``: zero, linear and quadratic parts, when ``u,y = v,x``, i.e. when
  the rotation about the normal ``(v,x - u,y)/2`` vanishes;
- ``kyy``: ``exx/r``, linear and quadratic parts;
- ``kxy``: ``-gxy/(2r)``, linear and quadratic parts, plus ``v,x w,xx``,
  which is ``gxy w,xx/2`` when ``u,y = v,x``.

These are terms of the order of the membrane strain divided by the radius or
times the curvature, within the accuracy of the classical theory. When the
rotation about the normal does not vanish, the quadratic parts also differ
by terms in ``u,y - v,x``: the rotations of the normal (1.127)-(1.128) have
``u,y`` where the area vector has ``v,x``, and conversely, which is exact
only for a symmetric in-plane displacement gradient.

Internal force vector and tangent stiffness
-------------------------------------------
At a point the strains depend on the seventeen generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, v, w,
         u,xx, u,yy, u,xy, v,xx, v,yy, v,xy, w,xx, w,yy, w,xy) = [Phi] c

With ``sigma = (Nxx, Nyy, Nxy, Mxx, Myy, Mxy) = [ABD] eps`` and ``[B] = d
eps/d d`` (6 x 17, quadratic in ``d``)::

    fint = int [Phi]^T [B]^T sigma dA

    K_T = int [Phi]^T ([B]^T [ABD] [B] + [H(sigma, d)]) [Phi] dA

    [H(sigma, d)] = sum_r sigma_r d^2 eps_r/d d^2

Split, as in every model of ``panels`` (:func:`check_tangent`):

    fkG_num: [H0(sigma_L + Nbar)],   H0 = H(., d = 0)
    fkC_num: [B]^T [ABD] [B] + [H(sigma, d)] - [H0(sigma_L)]

``fkG_num`` is homogeneous of degree one in ``c``. With a constant membrane
stress only, ``[H0]`` is the ``[H]`` of ``cylshell_clpt_gl``, so that the
analytical kernels of ``cylshell_clpt_gl`` (``fk0``, ``fkG0``, ``fkM``,
``fkAx``, ``fkAy``, ``fcA``) are those of this model. The strains are not of
the form ``([B_L] + [B_NL(beta)]/2) d`` assumed by
``theory/shells/tangent_product``; the product ``K_T v`` is evaluated from
``[B]`` and ``[H]`` at each point, as for ``plate_clpt_tl``.

Integration
-----------
Along ``x`` and ``y``, an approximation function of degree ``p`` has the
degree ``p`` and its first and second derivatives ``p - 1`` and ``p - 2``;
``v`` and ``w`` enter the strains without derivative through the curvature
of the reference surface. The integrands of ``fint`` and ``K_T`` are
polynomials of degree at most ``6p - 2`` along ``x`` and ``6p`` along ``y``
(:func:`check_integrand_degree`); the Gauss-Legendre rule of ``n`` points is
exact for ``2n - 1 >= 6p``, i.e. ``n >= 3p + 1``.

Run ``python cylshell_clpt_tl.py`` to run the checks.

References
----------
Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of Applied
Mathematics, 21(1), 21-36, 1963. https://doi.org/10.1090/qam/147023

Amabili, M., "Nonlinear Vibrations and Stability of Shells and Plates",
Cambridge University Press, 2008. https://doi.org/10.1017/CBO9780511619694

van der Heijden, A. M. A. (ed.), "W. T. Koiter's Elastic Stability of
Solids and Structures", Cambridge University Press, 2008.
https://doi.org/10.1017/CBO9780511546174

"""
import os
import random
import sys

import sympy
from sympy import (Matrix as M, Symbol, Function, symbols, cos, sin, expand,
                   zeros, eye, diff, sqrt, Rational)

HERE = os.path.dirname(os.path.abspath(__file__))

a, b, r = symbols('a, b, r', positive=True)

u_x, u_y, v_x, v_y, w_x, w_y, v, w = symbols(
    'u_x, u_y, v_x, v_y, w_x, w_y, v, w')
u_xx, u_yy, u_xy, v_xx, v_yy, v_xy, w_xx, w_yy, w_xy = symbols(
    'u_xx, u_yy, u_xy, v_xx, v_yy, v_xy, w_xx, w_yy, w_xy')
#: the six gradients and the two values of the displacements, which enter
#: the deformation gradient
GRAD = (u_x, u_y, v_x, v_y, w_x, w_y, v, w)
#: the nine second derivatives of the displacements
SECOND = (u_xx, u_yy, u_xy, v_xx, v_yy, v_xy, w_xx, w_yy, w_xy)
#: generalized displacement derivatives at a point, physical coordinates
D = GRAD + SECOND
#: names of the generalized derivatives in the kernels
D_NAMES = tuple(s.name for s in D)
FIELDS = ('u', 'v', 'w')
DOF = 3
NE = 6

A11, A12, A16, A22, A26, A66 = symbols('A11, A12, A16, A22, A26, A66')
B11, B12, B16, B22, B26, B66 = symbols('B11, B12, B16, B22, B26, B66')
D11, D12, D16, D22, D26, D66 = symbols('D11, D12, D16, D22, D26, D66')
Nxx, Nyy, Nxy = symbols('Nxx, Nyy, Nxy')
Mxx, Myy, Mxy = symbols('Mxx, Myy, Mxy')
STRESSES = (Nxx, Nyy, Nxy, Mxx, Myy, Mxy)

#: constitutive matrix of the CLPT, ``[ABD]``
F = M([[A11, A12, A16, B11, B12, B16],
       [A12, A22, A26, B12, B22, B26],
       [A16, A26, A66, B16, B26, B66],
       [B11, B12, B16, D11, D12, D16],
       [B12, B22, B26, D12, D22, D26],
       [B16, B26, B66, D16, D26, D66]])


def deformation_gradient():
    r"""``F = [X,x  X,y]`` in the local basis, 3 x 2"""
    return M([[1 + u_x, u_y],
              [v_x, 1 + v_y + w/r],
              [w_x, w_y - v/r]])


def second_derivatives():
    r"""``X,xx``, ``X,xy``, ``X,yy`` in the local basis"""
    Xxx = M([u_xx, v_xx, w_xx])
    Xxy = M([u_xy, v_xy + w_x/r, w_xy - v_x/r])
    Xyy = M([u_yy, v_yy + 2*w_y/r - v/r**2, w_yy - 2*v_y/r - w/r**2 - 1/r])
    return Xxx, Xxy, Xyy


def area_vector():
    r"""``a = X,x x X,y``, the normal of the deformed mid-surface times its
    area stretch"""
    Fg = deformation_gradient()
    return Fg[:, 0].cross(Fg[:, 1])


def membrane(nonlinear=True):
    r"""Green-Lagrange membrane strains ``(exx, eyy, gxy)``, or their linear
    part"""
    Fg = deformation_gradient()
    E = (Fg.T*Fg - eye(2))/2
    eps = M([E[0, 0], E[1, 1], 2*E[0, 1]]).applyfunc(expand)
    if nonlinear:
        return eps
    t = Symbol('t')
    return eps.subs({d: t*d for d in GRAD}, simultaneous=True).applyfunc(
        lambda e: expand(diff(e, t).subs(t, 0)))


def strains(nonlinear=True):
    r"""Generalized strains ``(exx, eyy, gxy, kxx, kyy, kxy)`` in terms of
    the derivatives ``D``, or their linear part"""
    mem = membrane()
    av = area_vector()
    Xxx, Xxy, Xyy = second_derivatives()
    kxx = -av.dot(Xxx)
    kyy = -av.dot(Xyy) - 1/r - (mem[0] + 2*mem[1])/r
    kxy = -2*av.dot(Xxy) - mem[2]/(2*r)
    eps = M([mem[0], mem[1], mem[2], kxx, kyy, kxy]).applyfunc(expand)
    if nonlinear:
        return eps
    t = Symbol('t')
    return eps.subs({d: t*d for d in D}, simultaneous=True).applyfunc(
        lambda e: expand(diff(e, t).subs(t, 0)))


Sym32 = Rational(3, 2)
Sym12 = Rational(1, 2)


def strains_sanders_linear():
    r"""Linear strains of ``cylshell_clpt_sanders``"""
    return M([u_x, v_y + w/r, u_y + v_x,
              -w_xx, -w_yy + v_y/r, -2*w_xy + Sym32*v_x/r - Sym12*u_y/r])


def basis(side):
    r"""``[Phi]`` of a term of side ``'A'`` or ``'B'``: the 17 x 3 matrix
    such that ``d = [Phi] c``, in the notation of the kernels, e.g.
    ``fAu_xixi`` for `\partial^2 f_i^u/\partial \xi^2`"""
    s = lambda name: Symbol(name.format(side))
    P = zeros(len(D), 3)
    for k, field in enumerate(FIELDS):
        f = lambda der: s('f{0}' + field + der)
        g = lambda der: s('g{0}' + field + der)
        P[D.index(Symbol(field + '_x')), k] = (2/a)*f('_xi')*g('')
        P[D.index(Symbol(field + '_y')), k] = (2/b)*f('')*g('_eta')
        P[D.index(Symbol(field + '_xx')), k] = (2/a)**2*f('_xixi')*g('')
        P[D.index(Symbol(field + '_yy')), k] = (2/b)**2*f('')*g('_etaeta')
        P[D.index(Symbol(field + '_xy')), k] = (2/a)*(2/b)*f('_xi')*g('_eta')
    P[D.index(v), 1] = s('f{0}v')*s('g{0}v')
    P[D.index(w), 2] = s('f{0}w')*s('g{0}w')
    return P


def hessian_stress(sigma, at_zero=False):
    r"""``[H(sigma, d)] = sum_r sigma_r d^2 eps_r/d D^2``, 17 x 17, linear
    in ``d``; with ``at_zero`` its value ``[H0(sigma)]`` at ``d = 0``"""
    eps = strains()
    H = zeros(len(D), len(D))
    for k in range(NE):
        H += sigma[k]*sympy.hessian(eps[k], D)
    if at_zero:
        H = H.subs({d: 0 for d in D})
    return H.applyfunc(expand)


def build():
    r"""Matrices of the derivation at an integration point

    Returns a dict with

    - ``'eps'``, ``'eps_L'``, ``'eps_NL'``: the strains and their linear and
      non-linear parts, in terms of ``D``;
    - ``'B'``, ``'B_L'``: ``d eps/d D`` (6 x 17) and its value at ``d = 0``;
    - ``'F'``: the constitutive matrix;
    - ``'H'``: ``[H(sigma, d)]`` with the symbols ``Nxx, ..., Mxy``;
    - ``'H0'``: ``[H0(sigma)]``, its value at ``d = 0``;
    - ``'PhiA'``, ``'PhiB'``: the matrices ``[Phi]`` of the two terms.

    """
    eps = strains()
    eps_L = strains(nonlinear=False)
    eps_NL = (eps - eps_L).applyfunc(expand)
    B = eps.jacobian(D).applyfunc(expand)
    B_L = B.subs({d: 0 for d in D})
    assert (B_L - eps_L.jacobian(D)).applyfunc(expand) == zeros(NE, len(D))
    H = hessian_stress(STRESSES)
    H0 = hessian_stress(STRESSES, at_zero=True)
    return dict(eps=eps, eps_L=eps_L, eps_NL=eps_NL, B=B, B_L=B_L, F=F,
                H=H, H0=H0, PhiA=basis('A'), PhiB=basis('B'))


def _global_frame():
    r"""Global components of the local basis and of ``X0`` as functions of
    ``x`` and ``y = r theta``"""
    x, y = symbols('x, y', real=True)
    th = y/r
    ex = M([1, 0, 0])
    ey = M([0, cos(th), sin(th)])
    ez = M([0, sin(th), -cos(th)])
    X0 = M([x, r*sin(th), -r*cos(th)])
    return x, y, ex, ey, ez, X0


def check_position_derivatives():
    r"""``X,a`` and ``X,ab`` in the local basis, from the global position
    ``X0 + u e_x + v e_y + w e_z``, and ``e_z . X0,yy = -1/r``"""
    x, y, ex, ey, ez, X0 = _global_frame()
    simp = lambda e: e.applyfunc(sympy.simplify)
    assert simp(diff(ey, y) + ez/r) == zeros(3, 1)
    assert simp(diff(ez, y) - ey/r) == zeros(3, 1)
    assert sympy.simplify(ez.dot(diff(X0, y, 2)) + 1/r) == 0
    U, V, W = [Function(nm)(x, y) for nm in 'UVW']
    X = X0 + U*ex + V*ey + W*ez
    Q = M.hstack(ex, ey, ez)
    sub = {diff(U, x, 2): u_xx, diff(U, y, 2): u_yy, diff(U, x, y): u_xy,
           diff(V, x, 2): v_xx, diff(V, y, 2): v_yy, diff(V, x, y): v_xy,
           diff(W, x, 2): w_xx, diff(W, y, 2): w_yy, diff(W, x, y): w_xy}
    sub1 = {diff(U, x): u_x, diff(U, y): u_y, diff(V, x): v_x,
            diff(V, y): v_y, diff(W, x): w_x, diff(W, y): w_y, V: v, W: w}
    loc = lambda vec: simp(Q.T*vec).subs(sub).subs(sub1)
    Fg = M.hstack(loc(diff(X, x)), loc(diff(X, y)))
    assert (Fg - deformation_gradient()).applyfunc(expand) == zeros(3, 2)
    Xxx, Xxy, Xyy = second_derivatives()
    assert (loc(diff(X, x, 2)) - Xxx).applyfunc(expand) == zeros(3, 1)
    assert (loc(diff(X, x, y)) - Xxy).applyfunc(expand) == zeros(3, 1)
    assert (loc(diff(X, y, 2)) - Xyy).applyfunc(expand) == zeros(3, 1)
    return True


def check_membrane():
    r"""The membrane strains are those of ``cylshell_clpt_gl``, the
    Flügge-Lur'e-Byrne strains of Amabili (2008), Eqs. (1.118)-(1.120) with
    ``w0 = 0``, ``y = R theta``"""
    sys.path.insert(0, os.path.join(HERE, '..', 'cylshell_clpt_gl'))
    import cylshell_clpt_gl as gl
    eps_gl = gl.strains()[:3, :].xreplace({gl.r: r})
    assert (membrane() - eps_gl).applyfunc(expand) == zeros(3, 1)
    # Amabili (2008), Eqs. (1.118)-(1.120), w0 = 0, d/dtheta = r d/dy
    am = M([u_x + (u_x**2 + v_x**2 + w_x**2)/2,
            v_y + w/r + (u_y**2 + (v_y + w/r)**2 + (w_y - v/r)**2)/2,
            v_x + u_y + u_x*u_y + v_x*(v_y + w/r) + w_x*(w_y - v/r)])
    assert (membrane() - am).applyfunc(expand) == zeros(3, 1)
    return True


def check_linear_part():
    r"""The linear strains are those of ``cylshell_clpt_sanders`` (and of
    ``cylshell_clpt_gl``), Sanders' linear changes of curvature as in
    Koiter's form, van der Heijden (2008), Eqs. (3.16.5)-(3.16.8), p. 183,
    and Eq. (3.17.3), p. 202, step by step in ``sanders_linear_part.py``"""
    assert (strains(nonlinear=False) - strains_sanders_linear()).applyfunc(
        expand) == zeros(6, 1)
    # without the terms -eyy/r and -gxy/(2r), the linear part is the
    # linearized change of the second fundamental form, which differs from
    # Sanders' by eyy/r and gxy/(2r)
    mem = membrane(nonlinear=False)
    kyy_b = -(area_vector().dot(second_derivatives()[2]) + 1/r
              + (membrane()[0] + membrane()[1])/r)
    t = Symbol('t')
    lin = lambda e: expand(diff(e.subs({d: t*d for d in D}, simultaneous=True),
                                t).subs(t, 0))
    assert expand(lin(kyy_b) - (-w_yy + v_y/r) - mem[1]/r) == 0
    kxy_b = -2*area_vector().dot(second_derivatives()[1])
    assert expand(lin(kxy_b) - (-2*w_xy + Sym32*v_x/r - Sym12*u_y/r)
                  - mem[2]/(2*r)) == 0
    return True


def check_area_vector_norm():
    r"""``|a|^2 = det(F^T F) = (1 + 2 exx)(1 + 2 eyy) - gxy^2``: ``|a| = 1``
    whenever the membrane strains vanish, and ``1 + exx + eyy + O(eps^2)``
    otherwise; with ``|a| = 1`` and the strains of a rigid motion, ``a . X,yy
    = b0_yy = -1/r``"""
    av = area_vector()
    Fg = deformation_gradient()
    C = Fg.T*Fg
    mem = membrane()
    assert expand(av.dot(av) - C.det()) == 0
    assert expand(C.det() - ((1 + 2*mem[0])*(1 + 2*mem[1]) - mem[2]**2)) == 0
    return True


def _random_field(rnd, scale):
    r"""A random polynomial displacement field ``(U, V, W)`` of ``x`` and
    ``y``, global functions"""
    x, y = symbols('x, y', real=True)
    mons = [1, x, y, x*y, x**2, y**2, x**2*y, x*y**2]
    return [sum(rnd.uniform(-scale, scale)*mo for mo in mons) for _ in range(3)]


def _strains_of(Ug, x, y, pts, ex, ey, ez, X0):
    r"""Strains of the global displacement ``Ug`` (3-vector of functions of
    ``x``, ``y``) at the points ``pts`` (``(x, y, r)``), as floats"""
    U, V, W = Ug.dot(ex), Ug.dot(ey), Ug.dot(ez)
    vals = {u_x: diff(U, x), u_y: diff(U, y), v_x: diff(V, x),
            v_y: diff(V, y), w_x: diff(W, x), w_y: diff(W, y), v: V, w: W,
            u_xx: diff(U, x, 2), u_yy: diff(U, y, 2), u_xy: diff(U, x, y),
            v_xx: diff(V, x, 2), v_yy: diff(V, y, 2), v_xy: diff(V, x, y),
            w_xx: diff(W, x, 2), w_yy: diff(W, y, 2), w_xy: diff(W, x, y)}
    eps = strains()
    out = []
    for xx, yy, rr in pts:
        num = {x: xx, y: yy, r: rr}
        dv = {k: sympy.N(e.subs(num), 30) for k, e in vals.items()}
        dv[r] = rr
        out.append([sympy.N(e.xreplace(dv), 30) for e in eps])
    return out


def check_rigid_rotation(samples=3, seed=0):
    r"""The strains vanish for any rigid motion of the cylinder and do not
    change under any rigid rotation of any deformed state, ``X -> R X``

    A rotation ``R`` (Rodrigues, random angle up to 3 rad about a random
    axis) of the undeformed cylinder gives the displacement ``(R - I) X0``;
    a deformed state ``X = X0 + U`` with a random polynomial ``U`` rotated
    by ``R`` gives ``R (X0 + U) - X0``. The six strains are evaluated at
    random points from the local components of the displacement and their
    exact derivatives.

    """
    rnd = random.Random(seed)
    x, y, ex, ey, ez, X0 = _global_frame()
    rr = sympy.Rational(9, 10)
    for _ in range(samples):
        nn = M([sympy.Float(rnd.uniform(-1, 1), 30) for _ in range(3)])
        nn = nn/sqrt(nn.dot(nn))
        t = sympy.Float(rnd.uniform(0.3, 3.), 30)
        K = M([[0, -nn[2], nn[1]], [nn[2], 0, -nn[0]], [-nn[1], nn[0], 0]])
        R = eye(3) + sin(t)*K + (1 - cos(t))*K*K
        pts = [(sympy.Float(rnd.uniform(-1, 1), 30),
                sympy.Float(rnd.uniform(-2, 2), 30), rr) for _ in range(2)]
        X0r = X0.xreplace({r: rr})
        exr, eyr, ezr = [e_.xreplace({r: rr}) for e_ in (ex, ey, ez)]
        # rigid motion of the undeformed cylinder, with a translation
        p = M([rnd.uniform(-1, 1) for _ in range(3)])
        Ug = (R - eye(3))*(X0r - p)
        for row in _strains_of(Ug, x, y, pts, exr, eyr, ezr, X0r):
            assert max(abs(e_) for e_ in row) < 1e-12, row
        # rotation of a deformed state
        U0 = M(_random_field(rnd, 0.3))
        Ur = R*(X0r + U0) - X0r
        e0 = _strains_of(U0, x, y, pts, exr, eyr, ezr, X0r)
        e1 = _strains_of(Ur, x, y, pts, exr, eyr, ezr, X0r)
        for r0, r1 in zip(e0, e1):
            assert max(abs(p_ - q_) for p_, q_ in zip(r0, r1)) < 1e-12
            assert max(abs(p_) for p_ in r0[3:]) > 1e-2
    return True


def novozhilov_amabili():
    r"""Changes of curvature of Amabili (2008), Eqs. (1.130)-(1.132), with
    ``w0 = 0`` and the derivatives with respect to ``theta`` written with
    ``y = R theta``: ``d/dtheta = R d/dy``"""
    R = r
    kx = (-w_xx + (v_x/R)*(-u_y + v_x - R*w_xy) + w_x*u_xx
          + u_xy*(-v/R + w_y) + w_xy*u_y
          - w_xx*(w/R + v_y + u_x))
    kt = (-w_yy + u_x/R + v_y/R - (w/R)*(w_yy - v_y/R - 2*u_x/R)
          - (u_y/R)*(u_y + v_x + R*w_xy)
          + (v_y/R)*(v_y + 3*u_x - R*w_yy)
          + v_yy*(w_y - v/R)
          + (w_y/R)*(-v/R + w_y)
          - w_yy*u_x
          + w_x*v_xy + v_x*w_xy)
    kxt = (-2*w_xy - u_y/R + v_x/R
           + (u_y/R)*(-u_x + R*w_yy - v_y)
           + w_xx*v_x
           - (v/R**2)*(R*u_yy + R*v_xy + w_x)
           + (w_y/R)*(R*u_yy + w_x + R*v_xy)
           + (v_x/R)*(w/R + 2*v_y - R*w_yy + 2*u_x)
           + w_x*(u_xy + v_xx)
           - 2*w_xy*(w/R + v_y + u_x))
    return M([kx, kt, kxt]).applyfunc(expand)


def _order(e, n):
    r"""Part of degree ``n`` in ``D`` of ``e``"""
    t = Symbol('t')
    es = e.subs({d: t*d for d in D}, simultaneous=True)
    return expand(diff(es, t, n).subs(t, 0)/sympy.factorial(n))


def check_novozhilov():
    r"""Term-by-term comparison with the curvatures of Amabili (2008), Eqs.
    (1.130)-(1.132), ``w0 = 0``, up to the quadratic terms

    Amabili's minus the curvatures of this model:

    - ``kxx``: no linear difference; the quadratic parts are equal when the
      rotation about the normal vanishes, ``u,y = v,x`` (and so ``u,xy =
      v,xx``, ``u,yy = v,xy``), and differ otherwise;
    - ``kyy``: ``exx/r``, linear and quadratic parts (with ``u,y = v,x``
      for the quadratic part);
    - ``kxy``: ``-gxy/(2r)``, linear and quadratic parts, plus the term
      ``v,x w,xx``, which is ``gxy w,xx/2`` to this order when ``u,y = v,x``:
      a membrane strain times a curvature, as the area stretch of the
      curvatures of this model.

    The differences are then of the order of the membrane strain divided by
    the radius, or of the membrane strain times the curvature, both within
    the accuracy of the classical theory of thin shells, except for the
    terms in ``u,y - v,x``.

    """
    am = novozhilov_amabili()
    mem = membrane()
    k = strains()[3:, :]
    lin = [_order(am[i] - k[i], 1) for i in range(3)]
    memL = [_order(mem[i], 1) for i in range(3)]
    assert lin[0] == 0
    assert expand(lin[1] - memL[0]/r) == 0
    assert expand(lin[2] + memL[2]/(2*r)) == 0
    sym = {u_y: v_x, u_xy: v_xx, u_yy: v_xy}
    memQ = [_order(mem[i], 2).xreplace(sym) for i in range(3)]
    quad = [_order(am[i] - k[i], 2) for i in range(3)]
    assert expand(quad[0].xreplace(sym)) == 0
    assert expand(quad[1].xreplace(sym) - memQ[0]/r) == 0
    assert expand(quad[2].xreplace(sym) + memQ[2]/(2*r) - v_x*w_xx) == 0
    assert expand(memL[2].xreplace(sym) - 2*v_x) == 0
    # Novozhilov's rotations of the normal, Eqs. (1.127)-(1.128), w0 = 0,
    # against the area vector: u,y and v,x interchanged
    Theta = -w_x*(1 + v_y + w/r) + (w_y - v/r)*u_y
    Psi = -(w_y - v/r)*(1 + u_x) + w_x*v_x
    av = area_vector()
    assert expand(av[0] - Theta.xreplace({u_y: v_x})) == 0
    assert expand(av[1] - Psi.xreplace({v_x: u_y})) == 0
    # with a rotation about the normal, u,y - v,x = -2 omega, the quadratic
    # part of kxx differs by terms in omega
    om = Symbol('omega')
    rot = {u_y: v_x - 2*om}
    assert expand(quad[0].xreplace(rot)) != 0
    assert expand(quad[0].xreplace(rot).xreplace({om: 0}).xreplace(
        {u_xy: v_xx})) == 0
    return True


def check_tangent():
    r"""``d(B^T sigma)/dD = B^T F B + H(sigma, d)``, ``sigma = F eps``, and
    the split into ``fkC_num`` and ``fkG_num``; with a constant membrane
    stress only, ``H0`` is the ``H`` of ``cylshell_clpt_gl``

    With a symbolic ``F`` the expansion is long, so the identity is checked
    one row of ``B^T sigma`` at a time.

    """
    m = build()
    eps, B, H, H0 = m['eps'], m['B'], m['H'], m['H0']
    sigma = (F*eps).applyfunc(expand)
    sub = dict(zip(STRESSES, sigma))
    KT = (B.T*F*B + H.xreplace(sub))
    for p in range(len(D)):
        row = expand((B[:, p].T*sigma)[0, 0])
        for q in range(len(D)):
            assert expand(diff(row, D[q]) - KT[p, q]) == 0, (p, q)
    sigma_L = F*m['eps_L']
    KG = H0.xreplace(dict(zip(STRESSES, sigma_L))).applyfunc(expand)
    tt = Symbol('tt')
    KGs = KG.subs({d: tt*d for d in D}, simultaneous=True)
    assert (KGs - tt*KG).applyfunc(expand) == zeros(len(D), len(D))
    KC = B.T*F*B + H.xreplace(sub) - KG
    zero = {d: 0 for d in D}
    assert (KC.subs(zero) - m['B_L'].T*F*m['B_L']).applyfunc(expand) == \
        zeros(len(D), len(D))
    sys.path.insert(0, os.path.join(HERE, '..', 'cylshell_clpt_gl'))
    import cylshell_clpt_gl as gl
    Hgl = gl.hessian_stress((Nxx, Nyy, Nxy)).xreplace({gl.r: r})
    idx = [D.index(Symbol(s.name)) for s in gl.D]
    Nonly = {Mxx: 0, Myy: 0, Mxy: 0}
    assert (H0.xreplace(Nonly)[idx, idx] - Hgl).applyfunc(expand) == \
        zeros(len(idx), len(idx))
    rest = [k for k in range(len(D)) if k not in idx]
    assert H0.xreplace(Nonly)[rest, :] == zeros(len(rest), len(D))
    return True


def check_numerical(seed=0):
    r"""Finite differences of ``fint`` of one pair of terms against
    ``[Phi_A]^T K_T [Phi_B]`` at random values"""
    rnd = random.Random(seed)
    m = build()
    vals = {s: rnd.uniform(-1, 1) for s in F.free_symbols}
    vals.update({a: 0.7, b: 0.4, r: 0.9})
    for s in m['PhiA'].free_symbols | m['PhiB'].free_symbols:
        vals.setdefault(s, rnd.uniform(-1, 1))
    PhiA = m['PhiA'].xreplace(vals)
    PhiB = m['PhiB'].xreplace(vals)
    Fn = F.xreplace(vals)
    eps = m['eps'].xreplace({r: vals[r]})
    Bn = m['B'].xreplace({r: vals[r]})
    Hn = m['H'].xreplace({r: vals[r]})
    cA = M([rnd.uniform(-0.3, 0.3) for _ in range(3)])
    cB = M([rnd.uniform(-0.3, 0.3) for _ in range(3)])

    def fint_A(cB_):
        sub = dict(zip(D, PhiA*cA + PhiB*cB_))
        sigma = Fn*eps.xreplace(sub)
        return PhiA.T*Bn.xreplace(sub).T*sigma

    sub = dict(zip(D, PhiA*cA + PhiB*cB))
    sigma = Fn*eps.xreplace(sub)
    K = (Bn.T*Fn*Bn + Hn.xreplace(dict(zip(STRESSES, sigma)))).xreplace(sub)
    KAB = PhiA.T*K*PhiB
    h = 1e-6
    for e in range(3):
        dc = zeros(3, 1)
        dc[e] = h
        fd = (fint_A(cB + dc) - fint_A(cB - dc))/(2*h)
        for dd in range(3):
            assert abs(float(fd[dd] - KAB[dd, e])) < 1e-6*(1 + abs(float(KAB[dd, e])))
    return True


def degree(direction):
    r"""Degree along ``direction`` (``'x'`` or ``'y'``) of the integrands of
    ``fint`` and ``K_T``, as ``(k, j)`` for ``k p + j``, with the
    approximation functions of degree ``p``: a derivative of order ``n``
    along the direction has the degree ``p - n``, any other factor ``p``"""
    eps = strains()
    deg = {s: -s.name.count(direction) for s in D}
    deg[v] = deg[w] = 0
    B = eps.jacobian(D)
    sigma = F*eps
    H = hessian_stress(STRESSES).xreplace(dict(zip(STRESSES, sigma)))
    KT = B.T*F*B + H
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
    return best


def check_integrand_degree():
    r"""Degree ``6p - 2`` along ``x`` and ``6p`` along ``y``"""
    assert degree('x') == (6, -2), degree('x')
    assert degree('y') == (6, 0), degree('y')
    return True


CHECKS = (check_position_derivatives, check_membrane, check_linear_part,
          check_area_vector_norm, check_rigid_rotation, check_novozhilov,
          check_tangent, check_numerical, check_integrand_degree)


if __name__ == '__main__':
    for check in CHECKS:
        check()
        print(check.__name__, 'verified')
