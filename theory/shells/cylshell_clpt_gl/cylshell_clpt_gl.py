r"""Cylindrical shell, CLPT with Green-Lagrange membrane strains

Kinematics
----------
Cylinder of radius ``r``, ``x`` along the axis, ``y = r theta`` the arc
length along the circumference, ``z`` along the outward normal; the local
orthonormal basis of the undeformed mid-surface is ``(e_x, e_y, e_z)``, with
``e_y,y = -e_z/r`` and ``e_z,y = e_y/r``. The degrees of freedom of a term of
the Ritz series are ``u, v, w``, the displacements of the mid-surface along
``e_x, e_y, e_z``, as in ``cylshell_clpt_sanders``. Assumptions:

- Kirchhoff: the normal stays straight and normal, the rotations derive from
  ``u, v, w``, the transverse shear strains vanish (CLPT);
- shallow through the thickness, ``r + z ~ r``, as in the other cylinders
  of panels;
- small strains, the constitutive law linear in the Green-Lagrange strains;
- membrane strains: the complete Green-Lagrange strains of the mid-surface
  (all the quadratic terms kept);
- changes of curvature: the linear ones of Sanders (1963), as written by
  Amabili (2008), Sec. 1.3.8, Eqs. (1.136)-(1.138), those of
  ``cylshell_clpt_sanders``.

The tangent vectors of the deformed mid-surface ``X = X0 + u e_x + v e_y +
w e_z``, in the local basis, are the columns of the 3 x 2 deformation
gradient::

    X,x = (1 + u,x) e_x + v,x e_y + w,x e_z
    X,y = u,y e_x + (1 + v,y + w/r) e_y + (w,y - v/r) e_z

and the membrane strains are ``E = (F^T F - I)/2``::

    exx = u,x + (u,x^2 + v,x^2 + w,x^2)/2
    eyy = v,y + w/r + (u,y^2 + (v,y + w/r)^2 + (w,y - v/r)^2)/2
    gxy = u,y + v,x + u,x u,y + v,x (v,y + w/r) + w,x (w,y - v/r)

the middle-surface strains of the Flügge-Lur'e-Byrne non-linear theory
as given by Amabili (2008), Sec. 1.3.6, Eqs. (1.118)-(1.120), without
imperfection and with ``y = R theta``. The six local gradients

    beta = (u,x, u,y, v,x, v,y + w/r, w,x, w,y - v/r)

are the entries of ``F - I`` (:func:`beta_of_D`). The changes of curvature::

    kxx = -w,xx
    kyy = -w,yy + v,y/r
    kxy = -2 w,xy + (3/2) v,x/r - (1/2) u,y/r

Relation with the other cylinders: with ``phix = -w,x``, ``phiy = -w,y +
v/r``, the terms ``w,x^2/2``, ``phiy^2/2`` and ``phix phiy`` are the
non-linear strains of ``cylshell_clpt_sanders``; the Green-Lagrange strains
add ``(u,x^2 + v,x^2)/2``, ``(u,y^2 + (v,y + w/r)^2)/2`` and ``u,x u,y + v,x
(v,y + w/r)`` (:func:`check_sanders_limit`). In particular the rotation
about the normal, ``omega = (v,x - u,y)/2``, which ``cylshell_clpt_sanders``
omits and which Sanders (1963) keeps as ``omega^2/2`` in ``exx`` and ``eyy``,
is contained in ``v,x^2/2`` and ``u,y^2/2``. Under an axial membrane force
``Nxx`` alone, the geometric stiffness is ``Nxx (u,x^2 + v,x^2 + w,x^2)``
instead of ``Nxx w,x^2`` (:func:`check_axial_geometric_stiffness`): the
work of ``Nxx`` on the bending of the tube as a beam, ``w = cos(theta)``,
``v = -sin(theta)``, is twice that of the Sanders strains, which therefore
give twice the Euler load of a long tube.

The membrane strains vanish for any rigid motion, translation or rotation
of any angle about any axis (:func:`check_rigid_rotation`), because the
columns of ``F`` are then the local components of two orthonormal vectors.
The linear changes of curvature vanish for any small rigid motion; for a
finite rotation ``t`` about an axis normal to the cylinder axis they are of
order ``(1 - cos t)/r``, the limitation of the model, as the curvatures of
``plate_clpt_gl`` measured in the fixed frame.

Internal force vector and tangent stiffness
-------------------------------------------
At a point the strains depend on the eleven generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, v, w, w,xx, w,yy, w,xy) = [Phi] c

With ``sigma = [ABD] eps``, ``[B] = d eps/d d`` (6 x 11, linear in the
first eight derivatives)::

    fint = int [Phi]^T [B]^T sigma dA
    K_T = int [Phi]^T ([B]^T [ABD] [B] + [H(N)]) [Phi] dA

with ``[H(N)] = sum_r sigma_r d^2 eps_r/d d^2 = [Gm]^T diag(N, N, N) [Gm]``,
``[Gm] = d beta/d d`` (6 x 11, constant), ``N = [[Nxx, Nxy], [Nxy, Nyy]]``
(:func:`check_tangent`). The split of panels, ``K_T = (KC0 + KCNL + KGNL)
[fkC_num] + KG(Nbar + N_L) [fkG_num]``, is that of ``plate_clpt_gl``. With
``beta = [Gm] d``, the strains have the form ``eps = ([B_L] + [B_NL(beta)]/2)
d`` of ``theory/shells/tangent_product`` (:func:`check_tangent_product_form`).

Run ``python cylshell_clpt_gl.py`` to run the checks.

References
----------
Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of Applied
Mathematics, 21(1), 21-36, 1963. https://doi.org/10.1090/qam/147023

Amabili, M., "Nonlinear Vibrations and Stability of Shells and Plates",
Cambridge University Press, 2008. https://doi.org/10.1017/CBO9780511619694

"""
import os
import sys

import sympy
from sympy import (Matrix as M, Symbol, Function, symbols, cos, sin, expand,
                   zeros, eye, diff, sqrt)

HERE = os.path.dirname(os.path.abspath(__file__))

a, b, r = symbols('a, b, r', positive=True)

#: generalized displacement derivatives at a point, physical coordinates
u_x, u_y, v_x, v_y, w_x, w_y, v, w, w_xx, w_yy, w_xy = symbols(
    'u_x, u_y, v_x, v_y, w_x, w_y, v, w, w_xx, w_yy, w_xy')
D = (u_x, u_y, v_x, v_y, w_x, w_y, v, w, w_xx, w_yy, w_xy)
#: the derivatives of the non-linear terms of the strains
GRAD = D[:8]
#: names of the generalized derivatives in the kernels
D_NAMES = tuple(s.name for s in D)
#: the six local gradients, the entries of F - I
BETA = symbols('beta_ux, beta_uy, beta_vx, beta_vy, beta_wx, beta_wy')
FIELDS = ('u', 'v', 'w')
DOF = 3
NE = 6

A11, A12, A16, A22, A26, A66 = symbols('A11, A12, A16, A22, A26, A66')
B11, B12, B16, B22, B26, B66 = symbols('B11, B12, B16, B22, B26, B66')
D11, D12, D16, D22, D26, D66 = symbols('D11, D12, D16, D22, D26, D66')
Nxx, Nyy, Nxy = symbols('Nxx, Nyy, Nxy')
STRESSES = (Nxx, Nyy, Nxy) + symbols('Mxx, Myy, Mxy')

#: constitutive matrix of the CLPT, ``[ABD]``
F = M([[A11, A12, A16, B11, B12, B16],
       [A12, A22, A26, B12, B22, B26],
       [A16, A26, A66, B16, B26, B66],
       [B11, B12, B16, D11, D12, D16],
       [B12, B22, B26, D12, D22, D26],
       [B16, B26, B66, D16, D26, D66]])


def beta_of_D():
    r"""The six local gradients ``beta``, entries of ``F - I``, in terms of
    ``D``"""
    return M([u_x, u_y, v_x, v_y + w/r, w_x, w_y - v/r])


def membrane_of_beta(bt, nonlinear=True):
    r"""Green-Lagrange membrane strains ``(exx, eyy, gxy)`` of the gradients
    ``bt = (ux, uy, vx, ty, wx, py)``, ``ty = v,y + w/r``, ``py = w,y -
    v/r``"""
    ux, uy, vx, ty, wx, py = bt
    t = 1 if nonlinear else 0
    return M([ux + t*(ux**2 + vx**2 + wx**2)/2,
              ty + t*(uy**2 + ty**2 + py**2)/2,
              uy + vx + t*(ux*uy + vx*ty + wx*py)])


Sym32 = sympy.Rational(3, 2)
Sym12 = sympy.Rational(1, 2)


def curvatures():
    r"""Linear changes of curvature of Sanders, as in
    ``cylshell_clpt_sanders``"""
    return M([-w_xx,
              -w_yy + v_y/r,
              -2*w_xy + Sym32*v_x/r - Sym12*u_y/r])


def strains(nonlinear=True):
    r"""Generalized strains ``(exx, eyy, gxy, kxx, kyy, kxy)`` in terms of
    ``D``"""
    mem = membrane_of_beta(beta_of_D(), nonlinear)
    return M(list(mem) + list(curvatures())).applyfunc(expand)


def strains_sanders():
    r"""Strains of ``cylshell_clpt_sanders`` in terms of ``D``"""
    phix = -w_x
    phiy = -w_y + v/r
    return M([u_x + phix**2/2,
              v_y + w/r + phiy**2/2,
              u_y + v_x + phix*phiy] + list(curvatures())).applyfunc(expand)


def basis(side):
    r"""``[Phi]`` of a term of side ``'A'`` or ``'B'``: the 11 x 3 matrix
    such that ``d = [Phi] c``, with the approximation functions in the
    notation of the kernels, e.g. ``fAu_xi`` for `\partial f_i^u/\partial
    \xi`"""
    s = lambda name: Symbol(name.format(side))
    P = zeros(11, 3)
    P[0, 0] = (2/a)*s('f{0}u_xi')*s('g{0}u')
    P[1, 0] = (2/b)*s('f{0}u')*s('g{0}u_eta')
    P[2, 1] = (2/a)*s('f{0}v_xi')*s('g{0}v')
    P[3, 1] = (2/b)*s('f{0}v')*s('g{0}v_eta')
    P[4, 2] = (2/a)*s('f{0}w_xi')*s('g{0}w')
    P[5, 2] = (2/b)*s('f{0}w')*s('g{0}w_eta')
    P[6, 1] = s('f{0}v')*s('g{0}v')
    P[7, 2] = s('f{0}w')*s('g{0}w')
    P[8, 2] = (2/a)**2*s('f{0}w_xixi')*s('g{0}w')
    P[9, 2] = (2/b)**2*s('f{0}w')*s('g{0}w_etaeta')
    P[10, 2] = (2/a)*(2/b)*s('f{0}w_xi')*s('g{0}w_eta')
    return P


def hessian_stress(N):
    r"""``[H(N)] = sum_r N_r d^2 eps_r/d D^2`` for the membrane stresses
    ``N = (Nxx, Nyy, Nxy)``, 11 x 11"""
    eps = strains()
    H = zeros(len(D), len(D))
    for k in range(3):
        H += N[k]*sympy.hessian(eps[k], D)
    return H.applyfunc(expand)


def build():
    r"""Matrices of the derivation at an integration point

    Returns a dict with the keys of ``plate_clpt_gl.build()``, ``'eps'``,
    ``'eps_L'``, ``'eps_NL'``, ``'B'``, ``'B_L'``, ``'F'``, ``'N_NL'``,
    ``'H'``, ``'PhiA'``, ``'PhiB'``, and, for
    ``theory/shells/tangent_product``, ``'BA'``, the 6 x 3 matrix ``([B_L] +
    [B_NL(beta)]) [Phi_A]`` with ``beta`` the symbols ``BETA``, and ``'GA'``,
    ``[Gm] [Phi_A]``, the 6 x 3 matrix of the gradients ``beta`` of a term;
    ``'Gm'`` is ``d beta/d D``.

    """
    eps = strains()
    eps_L = strains(nonlinear=False)
    eps_NL = (eps - eps_L).applyfunc(expand)
    B = eps.jacobian(D)
    B_L = B.subs({g: 0 for g in GRAD})
    assert B_L == eps_L.jacobian(D)
    N_NL = (F[:3, :3]*eps_NL[:3, :]).applyfunc(expand)
    H = hessian_stress((Nxx, Nyy, Nxy))
    PhiA, PhiB = basis('A'), basis('B')
    Gm = beta_of_D().jacobian(D)
    # [B_NL] in terms of the symbols beta: d eps_NL/d beta [Gm]
    memNL = membrane_of_beta(BETA) - membrane_of_beta(BETA, nonlinear=False)
    J = zeros(6, 6)
    J[:3, :] = memNL.jacobian(BETA)
    B_beta = B_L + J*Gm
    return dict(eps=eps, eps_L=eps_L, eps_NL=eps_NL, B=B, B_L=B_L, F=F,
                N_NL=N_NL, H=H, PhiA=PhiA, PhiB=PhiB, Gm=Gm,
                BA=(B_beta*PhiA).applyfunc(expand),
                GA=(Gm*PhiA).applyfunc(expand))


def check_deformation_gradient():
    r"""The strains are ``(F^T F - I)/2``, with ``F`` the local components
    of ``X,x`` and ``X,y`` of the deformed mid-surface

    With ``u, v, w`` functions of ``x, y`` and the derivatives of the local
    basis ``e_y,y = -e_z/r``, ``e_z,y = e_y/r``, the derivatives of
    ``u e_x + v e_y + w e_z`` give the columns of ``F - I``.

    """
    x, y = symbols('x, y')
    U, V, W = [Function(nm)(x, y) for nm in 'UVW']
    # global components of the local basis and of the position X0 =
    # (x, r sin(y/r), -r cos(y/r)) of the undeformed mid-surface
    th = y/r
    ex = M([1, 0, 0])
    ey = M([0, cos(th), sin(th)])
    ez = M([0, sin(th), -cos(th)])
    assert (diff(ey, y) + ez/r).applyfunc(sympy.simplify) == zeros(3, 1)
    assert (diff(ez, y) - ey/r).applyfunc(sympy.simplify) == zeros(3, 1)
    X0 = r*M([x/r, sin(th), -cos(th)])
    assert (diff(X0, y) - ey).applyfunc(sympy.simplify) == zeros(3, 1)
    X = X0 + U*ex + V*ey + W*ez
    Q = M.hstack(ex, ey, ez)
    Fx = (Q.T*diff(X, x)).applyfunc(sympy.simplify)
    Fy = (Q.T*diff(X, y)).applyfunc(sympy.simplify)
    sub = {diff(U, x): u_x, diff(U, y): u_y, diff(V, x): v_x,
           diff(V, y): v_y, diff(W, x): w_x, diff(W, y): w_y, V: v, W: w}
    Fg = M.hstack(Fx, Fy).subs(sub)
    bt = beta_of_D()
    assert (Fg - M([[1 + bt[0], bt[1]], [bt[2], 1 + bt[3]],
                    [bt[4], bt[5]]])).applyfunc(expand) == zeros(3, 2)
    E = (Fg.T*Fg - eye(2))/2
    eps = strains()
    assert expand(E[0, 0] - eps[0]) == 0
    assert expand(E[1, 1] - eps[1]) == 0
    assert expand(2*E[0, 1] - eps[2]) == 0
    return True


def check_rigid_rotation(samples=20, seed=0):
    r"""Zero membrane strains for finite rigid rotations about any axis

    The displacement ``(R - I) X0`` of a rotation ``R`` (Rodrigues, random
    angle up to 3 rad about a random axis through a random point) is
    projected on the local basis and differentiated exactly; the
    Green-Lagrange membrane strains vanish to round-off at random points,
    while the Sanders strains are of the order of the angle squared.

    """
    import random
    rnd = random.Random(seed)
    x, th, t = symbols('x, theta, t', real=True)
    n1, n2, n3, p1, p2, p3 = symbols('n1, n2, n3, p1, p2, p3', real=True)
    n = M([n1, n2, n3])
    K = M([[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]])
    R = eye(3) + sin(t)*K + (1 - cos(t))*K*K
    ex = M([1, 0, 0])
    ey = M([0, cos(th), sin(th)])
    ez = M([0, sin(th), -cos(th)])
    X0 = M([x, r*sin(th), -r*cos(th)])
    p = M([p1, p2, p3])
    disp = (R - eye(3))*(X0 - p)
    U, V, W = disp.dot(ex), disp.dot(ey), disp.dot(ez)
    # y = r theta
    vals = {u_x: diff(U, x), u_y: diff(U, th)/r, v_x: diff(V, x),
            v_y: diff(V, th)/r, w_x: diff(W, x), w_y: diff(W, th)/r,
            v: V, w: W}
    eps = strains()[:3, :]
    eps_s = strains_sanders()[:3, :]
    worst_gl, worst_s = 0., 0.
    for _ in range(samples):
        nn = [rnd.uniform(-1, 1) for _ in range(3)]
        norm = sum(c**2 for c in nn)**0.5
        num = {n1: nn[0]/norm, n2: nn[1]/norm, n3: nn[2]/norm,
               t: rnd.uniform(-3, 3), r: rnd.uniform(0.5, 2.),
               x: rnd.uniform(-1, 1), th: rnd.uniform(-3, 3),
               p1: rnd.uniform(-1, 1), p2: rnd.uniform(-1, 1),
               p3: rnd.uniform(-1, 1)}
        dv = {k: float(e.subs(num)) for k, e in vals.items()}
        dv[r] = num[r]
        worst_gl = max(worst_gl, max(abs(float(e.subs(dv))) for e in eps))
        worst_s = max(worst_s, max(abs(float(e.subs(dv))) for e in eps_s))
    assert worst_gl < 1e-12, worst_gl
    assert worst_s > 1e-2, worst_s
    return True


def check_sanders_limit():
    r"""The linear strains are those of ``cylshell_clpt_sanders``, and the
    difference of the non-linear strains is the quadratic terms of the
    in-plane gradients ``u,x, u,y, v,x, v,y + w/r``"""
    sys.path.insert(0, os.path.join(HERE, '..', 'cylshell_clpt_sanders'))
    import cylshell_clpt_sanders as cs
    eps, eps_s = strains(), strains_sanders()
    ty = v_y + w/r
    extra = M([(u_x**2 + v_x**2)/2, (u_y**2 + ty**2)/2, u_x*u_y + v_x*ty,
               0, 0, 0])
    assert (eps - eps_s - extra).applyfunc(expand) == zeros(6, 1)
    # B_L [Phi_A] = B0 of cylshell_clpt_sanders.py
    m = build()
    B0 = cs.shape_functions('A')[1]
    BL = (m['B_L']*m['PhiA']).xreplace({a: cs.a, b: cs.b, r: cs.r})
    assert (BL - B0).applyfunc(expand) == zeros(6, 3)
    # the terms omega^2/2 of exx and eyy of Sanders (1963), omega = (v,x -
    # u,y)/2, are contained in the added terms, the rest being squares
    omega = (v_x - u_y)/2
    rest = expand(extra[0] + extra[1] - omega**2)
    assert expand(rest - ((u_x**2 + ty**2)/2 + (v_x + u_y)**2/4)) == 0
    return True


def check_axial_geometric_stiffness():
    r"""Under ``Nxx`` alone, ``d^T [H] d = Nxx (u,x^2 + v,x^2 + w,x^2)``,
    ``Nxx w,x^2`` for ``cylshell_clpt_sanders``; for the beam mode of a
    tube, ``w = W(x) cos(theta)``, ``v = -W(x) sin(theta)``, ``u = 0``, the
    ``v,x^2`` term doubles the work of ``Nxx`` integrated over the
    circumference"""
    H = hessian_stress((Nxx, 0, 0))
    d = M(D)
    assert expand((d.T*H*d)[0] - Nxx*(u_x**2 + v_x**2 + w_x**2)) == 0
    x, th = symbols('x, theta', real=True)
    Wx = Symbol('Wx')
    wx = Wx*cos(th)
    vx = -Wx*sin(th)
    gl = sympy.integrate(wx**2 + vx**2, (th, 0, 2*sympy.pi))
    sd = sympy.integrate(wx**2, (th, 0, 2*sympy.pi))
    assert sympy.simplify(gl/sd) == 2
    return True


def check_tangent():
    r"""``d(B^T sigma)/dD = B^T F B + H(N)``, ``H = Gm^T diag(N, N, N) Gm``,
    and the split into ``fkC_num`` and ``fkG_num``, as in
    ``plate_clpt_gl.check_tangent``"""
    m = build()
    eps, B, H = m['eps'], m['B'], m['H']
    sigma = F*eps
    res = (B.T*sigma).applyfunc(expand)
    exact = res.jacobian(D).applyfunc(expand)
    N = {Nxx: sigma[0], Nyy: sigma[1], Nxy: sigma[2]}
    KT = (B.T*F*B + H.xreplace(N)).applyfunc(expand)
    assert (exact - KT).applyfunc(expand) == zeros(len(D), len(D))
    N2 = M([[Nxx, Nxy], [Nxy, Nyy]])
    Gm = m['Gm']
    assert (Gm.T*sympy.diag(N2, N2, N2)*Gm - H).applyfunc(expand) == \
        zeros(len(D), len(D))
    N_L = F[:3, :]*m['eps_L']
    KC = B.T*F*B + H.xreplace({Nxx: m['N_NL'][0], Nyy: m['N_NL'][1],
                                Nxy: m['N_NL'][2]})
    KG = H.xreplace({Nxx: N_L[0], Nyy: N_L[1], Nxy: N_L[2]})
    assert (KC + KG - KT).applyfunc(expand) == zeros(len(D), len(D))
    tt = Symbol('tt')
    KGs = KG.subs({dd: tt*dd for dd in D}, simultaneous=True)
    assert (KGs - tt*KG).applyfunc(expand) == zeros(len(D), len(D))
    return True


def check_tangent_product_form():
    r"""``eps = (B_L + B_NL(beta)/2) d`` with ``B_NL`` linear in ``beta =
    Gm d``, the form of ``theory/shells/tangent_product``, and ``BA`` of
    :func:`build` equal to ``B [Phi_A]``"""
    m = build()
    beta = m['Gm']*M(D)
    sub = dict(zip(BETA, beta))
    BA = m['BA'].xreplace(sub)
    assert (BA - m['B']*m['PhiA']).applyfunc(expand) == zeros(6, 3)
    B_L = m['B_L']
    B_NL = (m['B'] - B_L)
    for e in B_NL:
        assert sympy.Poly(e, *GRAD).total_degree() <= 1
    d = M(D)
    assert ((B_L + B_NL/2)*d - m['eps']).applyfunc(expand) == zeros(6, 1)
    return True


def check_numerical(seed=0):
    r"""Finite differences of ``fint`` of one pair of terms against
    ``[Phi_A]^T K_T [Phi_B]`` at random values"""
    import random
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
    K = (Bn.T*Fn*Bn + Hn.xreplace(
        {Nxx: sigma[0], Nyy: sigma[1], Nxy: sigma[2]})).xreplace(sub)
    KAB = PhiA.T*K*PhiB
    h = 1e-6
    for e in range(3):
        dc = zeros(3, 1)
        dc[e] = h
        fd = (fint_A(cB + dc) - fint_A(cB - dc))/(2*h)
        for dd in range(3):
            assert abs(float(fd[dd] - KAB[dd, e])) < 1e-6*(1 + abs(float(KAB[dd, e])))
    return True


CHECKS = (check_deformation_gradient, check_rigid_rotation,
          check_sanders_limit, check_axial_geometric_stiffness, check_tangent,
          check_tangent_product_form, check_numerical)


if __name__ == '__main__':
    for check in CHECKS:
        check()
        print(check.__name__, 'verified')
