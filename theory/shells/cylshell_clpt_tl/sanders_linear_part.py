r"""Why the linear part of the curvatures of ``cylshell_clpt_tl`` is Sanders'

Step-by-step derivation of the linear part of the changes of curvature of
``cylshell_clpt_tl``, from the position vector of the deformed mid-surface
to Sanders' linear changes of curvature, with every equation of the page
``doc/source/cylshell_clpt_tl.rst`` (labels ``cyl-tl-*``) and of
``cylshell_clpt_tl.md`` checked with sympy. Run::

    python sanders_linear_part.py

(about ten seconds). Each check prints the label of the equations it
verifies.

Notation
--------
``theta_xx, theta_yy, theta_xy`` (the linear membrane strains, tensor
components), ``omega_xy`` (the rotation about the normal), ``phi_x, phi_y``
(the rotations of the normal) and ``rho_xx, rho_yy, rho_xy`` (the linear
changes of curvature, tensor components) are those of Koiter in van der
Heijden (2008), Sec. 3.16, Eqs. (3.16.5)-(3.16.8), p. 183, and, for the
circular cylinder, Sec. 3.17, Eq. (3.17.3), p. 202, written there with the
dimensionless coordinates ``x/R``, ``y/R`` (here the physical ``x``, ``y =
r theta``) and the outward normal. The changes of curvature of panels are
the engineering components with the opposite sign, ``kxx = -rho_xx``,
``kyy = -rho_yy``, ``kxy = -2 rho_xy``.

References
----------
Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of Applied
Mathematics, 21(1), 21-36, 1963. https://doi.org/10.1090/qam/147023

van der Heijden, A. M. A. (ed.), "W. T. Koiter's Elastic Stability of
Solids and Structures", Cambridge University Press, 2008.
https://doi.org/10.1017/CBO9780511546174

"""
import os
import random
import sys

import sympy
from sympy import (Matrix as M, Rational, Symbol, Function, cos, sin, sqrt,
                   diff, expand, symbols, zeros, eye)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cylshell_clpt_tl as th
from cylshell_clpt_tl import (r, u_x, u_y, v_x, v_y, w_x, w_y, v, w, u_xx,
                              u_yy, u_xy, v_xx, v_yy, v_xy, w_xx, w_yy, w_xy)

D = th.D
half = Rational(1, 2)
t = Symbol('t')


def lin(e):
    r"""Linear part of ``e`` (scalar or matrix) in the displacement
    derivatives ``D``"""
    if hasattr(e, 'applyfunc'):
        return e.applyfunc(lin)
    es = e.subs({d: t*d for d in D}, simultaneous=True)
    return expand(diff(es, t).subs(t, 0))


def const(e):
    r"""Value of ``e`` at zero displacements"""
    if hasattr(e, 'applyfunc'):
        return e.applyfunc(const)
    return expand(e.subs({d: 0 for d in D}))


def zero(e):
    if hasattr(e, 'applyfunc'):
        return e.applyfunc(expand) == zeros(*e.shape)
    return expand(e) == 0


# Koiter's linear quantities of the cylinder, van der Heijden (2008), Eq.
# (3.17.3), p. 202, in the physical coordinates x = R x', y = R y'
theta_xx = u_x
theta_yy = v_y + w/r
theta_xy = half*(u_y + v_x)
omega_xy = half*(v_x - u_y)
phi_x = w_x
phi_y = w_y - v/r
rho_xx = w_xx
rho_yy = w_yy - v_y/r
rho_xy = w_xy - Rational(3, 4)*v_x/r + Rational(1, 4)*u_y/r


def frame():
    r"""Right-handed frame of the undeformed cylinder: ``X0(x, y)``, ``y = r
    theta``, and the basis ``(e_x, e_y, e_z)``, ``e_z`` the outward normal,
    as functions of ``x`` and ``y``"""
    x, y = symbols('x, y', real=True)
    th_ = y/r
    X0 = M([x, r*sin(th_), r*cos(th_)])
    ex = diff(X0, x)
    ey = diff(X0, y)
    ez = M([0, sin(th_), cos(th_)])
    return x, y, X0, ex, ey, ez


def check_reference_surface():
    r"""cyl-tl-basis, cyl-tl-b0: ``e_x x e_y = e_z`` (outward), ``e_y,y =
    -e_z/r``, ``e_z,y = e_y/r``, the other derivatives of the basis zero;
    ``X0,yy = -e_z/r``, ``X0,xx = X0,xy = 0``; ``b0_ab = e_z . X0,ab``:
    ``b0_xx = b0_xy = 0``, ``b0_yy = -1/r``"""
    x, y, X0, ex, ey, ez = frame()
    simp = lambda e: e.applyfunc(sympy.simplify)
    assert simp(ex.cross(ey) - ez) == zeros(3, 1)
    assert [sympy.simplify(ez.dot(e_)) for e_ in (ex, ey, ez)] == [0, 0, 1]
    assert simp(X0.dot(ez)*ez - X0 + x*ex) == zeros(3, 1)  # ez outward, |X0 - x ex| = r
    assert sympy.simplify(sqrt((X0 - x*ex).dot(X0 - x*ex)) - r) == 0
    assert simp(diff(ey, y) + ez/r) == zeros(3, 1)
    assert simp(diff(ez, y) - ey/r) == zeros(3, 1)
    for e_ in (ex, ey, ez):
        assert simp(diff(e_, x)) == zeros(3, 1)
    assert simp(diff(ex, y)) == zeros(3, 1)
    assert simp(diff(X0, y, 2) + ez/r) == zeros(3, 1)
    assert simp(diff(X0, x, 2)) == zeros(3, 1)
    assert simp(diff(X0, x, y)) == zeros(3, 1)
    b0 = [sympy.simplify(ez.dot(diff(X0, *der)))
          for der in ((x, 2), (x, y), (y, 2))]
    assert b0 == [0, 0, -1/r]
    return True


def deformed(local=True):
    r"""``X,x, X,y, X,xx, X,xy, X,yy`` of ``X = X0 + u e_x + v e_y + w e_z``,
    from the global position: components in the local basis, or (with
    ``local=False``) the global vectors and the frame"""
    x, y, X0, ex, ey, ez = frame()
    U, V, W = [Function(nm)(x, y) for nm in 'UVW']
    X = X0 + U*ex + V*ey + W*ez
    Q = M.hstack(ex, ey, ez)
    sub2 = {diff(U, x, 2): u_xx, diff(U, y, 2): u_yy, diff(U, x, y): u_xy,
            diff(V, x, 2): v_xx, diff(V, y, 2): v_yy, diff(V, x, y): v_xy,
            diff(W, x, 2): w_xx, diff(W, y, 2): w_yy, diff(W, x, y): w_xy}
    sub1 = {diff(U, x): u_x, diff(U, y): u_y, diff(V, x): v_x,
            diff(V, y): v_y, diff(W, x): w_x, diff(W, y): w_y, V: v, W: w}
    to = lambda e: e.subs(sub2).subs(sub1)
    ders = [diff(X, x), diff(X, y), diff(X, x, 2), diff(X, x, y),
            diff(X, y, 2)]
    if not local:
        return [to(d_) for d_ in ders], (x, y, ex, ey, ez)
    return [to((Q.T*d_).applyfunc(sympy.simplify)) for d_ in ders]


def check_position_derivatives():
    r"""cyl-tl-Xa, cyl-tl-Xab: the derivatives of the deformed position in
    the local basis, those of ``cylshell_clpt_tl.py``"""
    Xx, Xy, Xxx, Xxy, Xyy = deformed()
    Fg = th.deformation_gradient()
    assert zero(Xx - Fg[:, 0]) and zero(Xy - Fg[:, 1])
    assert zero(Xx - M([1 + u_x, v_x, w_x]))
    assert zero(Xy - M([u_y, 1 + v_y + w/r, w_y - v/r]))
    assert zero(Xxx - M([u_xx, v_xx, w_xx]))
    assert zero(Xxy - M([u_xy, v_xy + w_x/r, w_xy - v_x/r]))
    assert zero(Xyy - M([u_yy, v_yy + 2*w_y/r - v/r**2,
                         w_yy - 2*v_y/r - w/r**2 - 1/r]))
    S = th.second_derivatives()
    assert zero(Xxx - S[0]) and zero(Xxy - S[1]) and zero(Xyy - S[2])
    return True


def area_vector():
    r"""cyl-tl-a: ``a = X,x x X,y`` in the local basis"""
    return M([v_x*(w_y - v/r) - w_x*(1 + v_y + w/r),
              w_x*u_y - (1 + u_x)*(w_y - v/r),
              (1 + u_x)*(1 + v_y + w/r) - v_x*u_y])


def check_area_vector():
    r"""cyl-tl-a, cyl-tl-a-norm, cyl-tl-a-lin: the components of ``a``, from
    the cross product of the global vectors ``X,x`` and ``X,y``; ``|a|^2 =
    (1 + 2 exx)(1 + 2 eyy) - gxy^2``; its linear part ``a = e_z - phi_x e_x
    - phi_y e_y + (theta_xx + theta_yy) e_z``, and ``|a| = 1 + theta_xx +
    theta_yy`` to first order"""
    (Xx, Xy, _, _, _), (x, y, ex, ey, ez) = deformed(local=False)
    a_glob = Xx.cross(Xy)
    Q = M.hstack(ex, ey, ez)
    a_loc = (Q.T*a_glob).applyfunc(sympy.simplify)
    assert zero(a_loc - area_vector())
    assert zero(area_vector() - th.area_vector())
    mem = th.membrane()
    a_ = area_vector()
    assert zero(a_.dot(a_) - ((1 + 2*mem[0])*(1 + 2*mem[1]) - mem[2]**2))
    assert zero(const(a_) - M([0, 0, 1]))
    assert zero(lin(a_) - M([-phi_x, -phi_y, theta_xx + theta_yy]))
    assert zero(lin(sqrt(a_.dot(a_))) - (theta_xx + theta_yy))
    assert zero(lin(mem) - M([theta_xx, theta_yy, 2*theta_xy]))
    return True


def check_linearized_products():
    r"""cyl-tl-aX-lin: to first order ``a . X,xx = w,xx``, ``a . X,xy =
    w,xy - v,x/r``, ``a . X,yy = -1/r - (theta_xx + theta_yy)/r + w,yy - 2
    v,y/r - w/r^2``, the last from ``e_z . X0,yy = -1/r``, ``delta a .
    X0,yy = -(theta_xx + theta_yy)/r`` and ``e_z . delta X,yy = w,yy - 2
    v,y/r - w/r^2``"""
    Xx, Xy, Xxx, Xxy, Xyy = deformed()
    a_ = area_vector()
    assert const(a_.dot(Xxx)) == 0 and const(a_.dot(Xxy)) == 0
    assert zero(lin(a_.dot(Xxx)) - w_xx)
    assert zero(lin(a_.dot(Xxy)) - (w_xy - v_x/r))
    assert zero(const(a_.dot(Xyy)) + 1/r)
    X0yy = const(Xyy)
    assert zero(X0yy - M([0, 0, -1/r]))
    assert zero(lin(a_).dot(X0yy) + (theta_xx + theta_yy)/r)
    assert zero(M([0, 0, 1]).dot(lin(Xyy)) - (w_yy - 2*v_y/r - w/r**2))
    assert zero(lin(a_.dot(Xyy)) - (-(theta_xx + theta_yy)/r + w_yy
                                    - 2*v_y/r - w/r**2))
    return True


def second_fundamental_form():
    r"""``b_ab = n . X,ab``, ``n = a/|a|``, exact"""
    Xx, Xy, Xxx, Xxy, Xyy = deformed()
    a_ = area_vector()
    na = sqrt(a_.dot(a_))
    return [a_.dot(Xab)/na for Xab in (Xxx, Xxy, Xyy)]


def check_change_second_fundamental_form():
    r"""cyl-tl-db: the linear changes of the second fundamental form, ``db_xx
    = w,xx``, ``db_xy = w,xy - v,x/r``, ``db_yy = w,yy - 2 v,y/r - w/r^2``:
    the term ``-(theta_xx + theta_yy)/r`` of ``a . X,yy`` is the area stretch
    times ``b0_yy`` and does not enter ``b_yy``"""
    bxx, bxy, byy = second_fundamental_form()
    assert const(bxx) == 0 and const(bxy) == 0 and zero(const(byy) + 1/r)
    assert zero(lin(bxx) - w_xx)
    assert zero(lin(bxy) - (w_xy - v_x/r))
    assert zero(lin(byy) - (w_yy - 2*v_y/r - w/r**2))
    # a . X,ab = |a| b_ab: the stretch multiplies b0_yy = -1/r
    a_ = area_vector()
    Xyy = deformed()[4]
    assert zero(lin(a_.dot(Xyy)) - lin(byy) + (theta_xx + theta_yy)/r)
    return True


def check_koiter():
    r"""cyl-tl-koiter: Koiter's Eq. (3.16.8) for the cylinder, with ``b_x^x =
    b_x^y = 0``, ``b_y^y = -1/r`` and Cartesian surface coordinates (the
    covariant derivatives are partial derivatives): ``rho_xx = phi_x,x``,
    ``rho_yy = phi_y,y``, ``rho_xy = (phi_x,y + phi_y,x)/2 - omega_xy/(2r)``,
    which are the components of Eq. (3.17.3); cyl-tl-sanders: ``kxx =
    -rho_xx``, ``kyy = -rho_yy``, ``kxy = -2 rho_xy`` are the linear
    changes of curvature of ``cylshell_clpt_sanders``"""
    x, y = symbols('x, y', real=True)
    U, V, W = [Function(nm)(x, y) for nm in 'UVW']
    sub2 = {diff(U, x, 2): u_xx, diff(U, y, 2): u_yy, diff(U, x, y): u_xy,
            diff(V, x, 2): v_xx, diff(V, y, 2): v_yy, diff(V, x, y): v_xy,
            diff(W, x, 2): w_xx, diff(W, y, 2): w_yy, diff(W, x, y): w_xy}
    sub1 = {diff(U, x): u_x, diff(U, y): u_y, diff(V, x): v_x,
            diff(V, y): v_y, diff(W, x): w_x, diff(W, y): w_y, V: v, W: w}
    to = lambda e: expand(e.subs(sub2).subs(sub1))
    # u_alpha = (U, V), b_alpha^kappa = diag(0, -1/r), (3.16.5)-(3.16.7)
    ua = [U, V]
    X = [x, y]
    bm = M([[0, 0], [0, -1/r]])
    th_ = [[half*(diff(ua[al], X[be]) + diff(ua[be], X[al])) - bm[al, be]*W
            for be in range(2)] for al in range(2)]
    om = [[half*(diff(ua[be], X[al]) - diff(ua[al], X[be])) for be in range(2)]
          for al in range(2)]
    ph = [diff(W, X[al]) + sum(bm[al, k]*ua[k] for k in range(2))
          for al in range(2)]
    rho = [[half*(diff(ph[al], X[be]) + diff(ph[be], X[al]))
            - half*sum(bm[al, k]*om[k][be] + bm[be, k]*om[k][al]
                       for k in range(2))
            for be in range(2)] for al in range(2)]
    assert zero(to(th_[0][0]) - theta_xx) and zero(to(th_[1][1]) - theta_yy)
    assert zero(to(th_[0][1]) - theta_xy) and zero(to(om[0][1]) - omega_xy)
    assert zero(to(ph[0]) - phi_x) and zero(to(ph[1]) - phi_y)
    assert zero(to(rho[0][0]) - rho_xx)
    assert zero(to(rho[1][1]) - rho_yy)
    assert zero(to(rho[0][1]) - rho_xy)
    # the compact forms of the page
    assert zero(rho_xx - diff(phi_x, w_x)*w_xx)  # phi_x,x = w,xx
    assert zero(rho_yy - (w_yy - v_y/r))         # phi_y,y
    assert zero(rho_xy - (half*(w_xy + (w_xy - v_x/r)) - omega_xy/(2*r)))
    # engineering components with the sign of panels
    kS = M([-rho_xx, -rho_yy, -2*rho_xy])
    assert zero(kS - th.strains_sanders_linear()[3:, :])
    assert zero(kS - M([-w_xx, -w_yy + v_y/r,
                        -2*w_xy + Rational(3, 2)*v_x/r - half*u_y/r]))
    return True


def check_db_minus_rho():
    r"""cyl-tl-db-rho: ``db_xx = rho_xx``, ``db_yy = rho_yy - theta_yy/r``,
    ``db_xy = rho_xy - theta_xy/(2r)``, from ``db_yy = phi_y,y -
    theta_yy/r`` and ``db_xy = (phi_x,y + phi_y,x)/2 - v,x/(2r)`` with
    ``v,x = theta_xy + omega_xy``; cyl-tl-kb: the engineering changes of the
    second fundamental form ``kb = (-db_xx, -db_yy, -2 db_xy) = kS + (0,
    eyy/r, gxy/(2r))`` to first order"""
    bxx, bxy, byy = [lin(b_) for b_ in second_fundamental_form()]
    assert zero(bxx - rho_xx)
    assert zero(byy - (rho_yy - theta_yy/r))
    assert zero(bxy - (rho_xy - theta_xy/(2*r)))
    assert zero(byy - ((w_yy - v_y/r) - theta_yy/r))
    assert zero(bxy - (half*(w_xy + (w_xy - v_x/r)) - v_x/(2*r)))
    assert zero(v_x - (theta_xy + omega_xy))
    kS = th.strains_sanders_linear()[3:, :]
    memL = lin(th.membrane())
    kb = M([-bxx, -byy, -2*bxy])
    assert zero(kb - kS - M([0, memL[1]/r, memL[2]/(2*r)]))
    return True


def check_model_linear_part():
    r"""cyl-tl-k, cyl-tl-k-lin: the curvatures of the model and their linear
    parts, term by term: ``kxx = -w,xx``; ``kyy``: the constant ``1/r`` of
    ``-a . X,yy`` cancels with ``-1/r``, its stretch term ``(theta_xx +
    theta_yy)/r`` with the linear part of ``-(exx + eyy)/r``, leaving ``-db_yy
    = -w,yy + 2 v,y/r + w/r^2``, and ``-eyy/r`` turns it into ``-w,yy +
    v,y/r``; ``kxy = -2 db_xy - gxy/(2r) = -2 w,xy + (3/2) v,x/r - (1/2)
    u,y/r``; these are ``kS``"""
    Xx, Xy, Xxx, Xxy, Xyy = deformed()
    a_ = area_vector()
    mem = th.membrane()
    memL = lin(mem)
    kxx = -a_.dot(Xxx)
    kyy = -a_.dot(Xyy) - 1/r - (mem[0] + 2*mem[1])/r
    kxy = -2*a_.dot(Xxy) - mem[2]/(2*r)
    assert zero(M([kxx, kyy, kxy]) - th.strains()[3:, :])
    # the same, written as the change of the second fundamental form times
    # the stretch, plus the two terms
    stretch1 = 1 + mem[0] + mem[1]
    assert zero(kyy - (-(a_.dot(Xyy) - stretch1*(-1/r)) - mem[1]/r))
    assert const(kxx) == 0 and const(kyy) == 0 and const(kxy) == 0
    # term by term
    lyy_a = lin(-a_.dot(Xyy))
    assert zero(lyy_a - ((theta_xx + theta_yy)/r - w_yy + 2*v_y/r + w/r**2))
    assert zero(lin(-(mem[0] + mem[1])/r) + (theta_xx + theta_yy)/r)
    assert zero(lyy_a + lin(-(mem[0] + mem[1])/r) - (-w_yy + 2*v_y/r
                                                    + w/r**2))
    assert zero(-w_yy + 2*v_y/r + w/r**2 - memL[1]/r - (-w_yy + v_y/r))
    assert zero(-2*(w_xy - v_x/r) - memL[2]/(2*r)
                - (-2*w_xy + Rational(3, 2)*v_x/r - half*u_y/r))
    kS = th.strains_sanders_linear()[3:, :]
    assert zero(lin(M([kxx, kyy, kxy])) - kS)
    # without the stretch on the reference term, -a . X,yy - 1/r - eyy/r has
    # the further linear term (theta_xx + theta_yy)/r
    assert zero(lin(-a_.dot(Xyy) - 1/r - mem[1]/r) - kS[1]
                - (theta_xx + theta_yy)/r)
    return True


def check_uniform_expansion():
    r"""cyl-tl-expansion: a uniform radial displacement ``w = w0``: ``eyy =
    w0/r + w0^2/(2 r^2)``, exactly ``kyy = 0`` for the model and for
    Sanders' linear curvature, and ``-(b_yy - b0_yy) = w0/r^2`` for the
    change of the second fundamental form"""
    w0 = Symbol('w0', positive=True)
    sub = {d: 0 for d in D}
    sub[w] = w0
    mem = th.membrane().xreplace(sub)
    assert zero(mem - M([0, w0/r + w0**2/(2*r**2), 0]))
    k = th.strains()[3:, :].xreplace(sub)
    assert zero(k)
    assert zero(th.strains_sanders_linear()[3:, :].xreplace(sub))
    byy = second_fundamental_form()[2].xreplace(sub)
    assert sympy.simplify(-(byy + 1/r) - w0/r**2) == 0
    return True


def _rotation(rnd):
    nn = M([sympy.Float(rnd.uniform(-1, 1), 30) for _ in range(3)])
    nn = nn/sqrt(nn.dot(nn))
    ang = sympy.Float(rnd.uniform(0.5, 1.5), 30)
    K = M([[0, -nn[2], nn[1]], [nn[2], 0, -nn[0]], [-nn[1], nn[0], 0]])
    return eye(3) + sin(ang)*K + (1 - cos(ang))*K*K


def check_invariance(seed=1):
    r"""cyl-tl-invariance: under a rigid rotation ``X -> R X`` of a deformed
    state, the curvatures of the model and the changes of the second
    fundamental form do not change, whereas Sanders' linear changes of
    curvature (those of ``cylshell_clpt_gl``) do"""
    rnd = random.Random(seed)
    x, y, X0, ex, ey, ez = frame()
    rr = Rational(9, 10)
    X0, ex, ey, ez = [e_.xreplace({r: rr}) for e_ in (X0, ex, ey, ez)]
    mons = [1, x, y, x*y, x**2, y**2, x**2*y, x*y**2]
    U0 = M([sum(rnd.uniform(-0.2, 0.2)*mo for mo in mons) for _ in range(3)])
    R = _rotation(rnd)
    Ur = R*(X0 + U0) - X0
    model = th.strains()[3:, :]
    sanders = th.strains_sanders_linear()[3:, :]
    # -(b_xx - b0_xx), -(b_yy - b0_yy), -2 (b_xy - b0_xy)
    bxx, bxy, byy = second_fundamental_form()
    plain = M([-bxx, -byy - 1/r, -2*bxy])
    pt = {x: sympy.Float(0.31, 30), y: sympy.Float(-0.47, 30)}

    def values(Ug, exprs):
        Uu, Vv, Ww = Ug.dot(ex), Ug.dot(ey), Ug.dot(ez)
        vals = {u_x: diff(Uu, x), u_y: diff(Uu, y), v_x: diff(Vv, x),
                v_y: diff(Vv, y), w_x: diff(Ww, x), w_y: diff(Ww, y),
                v: Vv, w: Ww, u_xx: diff(Uu, x, 2), u_yy: diff(Uu, y, 2),
                u_xy: diff(Uu, x, y), v_xx: diff(Vv, x, 2),
                v_yy: diff(Vv, y, 2), v_xy: diff(Vv, x, y),
                w_xx: diff(Ww, x, 2), w_yy: diff(Ww, y, 2),
                w_xy: diff(Ww, x, y)}
        dv = {k_: sympy.N(e_.subs(pt), 30) for k_, e_ in vals.items()}
        dv[r] = rr
        return [sympy.N(e_.xreplace(dv), 30) for e_ in exprs]

    m0, m1 = values(U0, model), values(Ur, model)
    p0, p1 = values(U0, plain), values(Ur, plain)
    s0, s1 = values(U0, sanders), values(Ur, sanders)
    assert max(abs(a_ - b_) for a_, b_ in zip(m0, m1)) < 1e-20
    assert max(abs(a_ - b_) for a_, b_ in zip(p0, p1)) < 1e-20
    assert max(abs(a_ - b_) for a_, b_ in zip(s0, s1)) > 1e-2
    return True


CHECKS = (check_reference_surface, check_position_derivatives,
          check_area_vector, check_linearized_products,
          check_change_second_fundamental_form, check_koiter,
          check_db_minus_rho, check_model_linear_part,
          check_uniform_expansion, check_invariance)


if __name__ == '__main__':
    for check in CHECKS:
        check()
        labels = [w_.strip(',:;') for w_ in check.__doc__.split()
                  if w_.startswith('cyl-tl-')]
        print(check.__name__, 'verified:', ', '.join(labels))
