r"""Flat plate, CLPT with Green-Lagrange membrane strains (total Lagrangian)

Kinematics
----------
The degrees of freedom of a term of the Ritz series are ``u, v, w``, the
displacements of the mid-surface along the fixed local axes ``x, y, z`` of
the plate, as in ``plate_clpt_donnell``. At a point the strains depend on the
nine generalized displacement derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, w,xx, w,yy, w,xy)

which are linear in the Ritz constants, ``d = [Phi] c`` (``Phi`` below). The
membrane strains are the complete Green-Lagrange strains of the mid-surface,
``E = (F^T F - I)/2`` with the 3 x 2 deformation gradient
``F = [X,x  X,y] = [e_x + u,x  e_y + u,y]``, ``u = (u, v, w)``::

    exx = u,x + (u,x^2 + v,x^2 + w,x^2)/2
    eyy = v,y + (u,y^2 + v,y^2 + w,y^2)/2
    gxy = u,y + v,x + u,x u,y + v,x v,y + w,x w,y

and the changes of curvature those of ``plate_clpt_donnell``::

    kxx = -w,xx,   kyy = -w,yy,   kxy = -2 w,xy

Dropping the quadratic terms in ``u,x, u,y, v,x, v,y`` recovers the von
Kármán strains of ``plate_clpt_donnell``. The membrane strains vanish for
any rigid rotation ``u = (R - I) X``, because ``F^T F = I`` when the columns
of ``F`` are two columns of a rotation matrix, for in-plane rotations
(about ``z``) as well as out-of-plane ones (about ``x`` or ``y``); the von
Kármán strains give ``exx = cos(t) - 1`` for an in-plane rotation ``t``. See
:func:`check_rigid_rotation`.

The changes of curvature are measured in the fixed local frame, such that a
bending deformation superposed on a large rigid rotation ``t`` is seen with
the factor ``cos(t)``; this is the limitation of the model, see
``plate_clpt_gl.md``.

Internal force vector and tangent stiffness
-------------------------------------------
With the generalized stresses ``sigma = (Nxx, Nyy, Nxy, Mxx, Myy, Mxy) =
[ABD] eps`` and ``[B] = d eps/d d`` (6 x 9, linear in the gradients)::

    fint = int [Phi]^T [B]^T sigma dA

    K_T = d fint/d c = int [Phi]^T ([B]^T [ABD] [B] + [H(N)]) [Phi] dA

with ``[H(N)] = sum_r sigma_r d^2 eps_r/d d^2``, which, the curvatures being
linear, only involves the membrane stresses ``N = [[Nxx, Nxy], [Nxy,
Nyy]]``: ``[H(N)] = diag(N, N, N, 0)`` on the pairs ``(u,x, u,y)``,
``(v,x, v,y)``, ``(w,x, w,y)``. The geometric stiffness thus has ``u-u``
and ``v-v`` blocks besides the ``w-w`` block of the von Kármán plate.

As in all the models, ``K_T`` is split as

    K_T = (KC0 + KCNL + KGNL)  [fkC_num]  +  KG(Nbar + N_L)  [fkG_num]

with ``N_L`` the membrane stresses of the linear strains ``[B_L] d``,
``B_L = B(gradients = 0)``, and ``N_NL = [A] eps_NL`` those of the quadratic
strains, such that ``fkG_num`` is homogeneous of degree one in ``c``, as the
linear buckling analysis requires. :func:`check_tangent` verifies that the
derivative of ``fint`` is ``K_T``.

The quadratic strains also fit the form ``eps = ([B_L] + [B_NL(beta)]/2) d``
of ``theory/shells/tangent_product``, with ``beta`` the six gradients
``(u,x, u,y, v,x, v,y, w,x, w,y)`` instead of the two rotations of the von
Kármán and Sanders strains, and ``[G]^T [N] [G]`` with ``[N] = diag(N, N,
N)``, see :func:`check_tangent_product_form`.

Kernels
-------
``write_pyx.py`` generates ``panels/models/plate_clpt_gl_num.pyx`` and
``panels/models/plate_clpt_gl.pyx`` from :func:`build`: at each integration
point the 9 x 9 matrix ``[B]^T [ABD] [B] + [H]`` is formed once, and the
entry of the pair of terms ``(i, j)``, ``(k, l)`` is
``[Phi_A]^T K [Phi_B]``, with ``[Phi]`` evaluated from the tabulated
approximation functions.

Run ``python plate_clpt_gl.py`` to run the checks.

"""
import os
import sys

import sympy
from sympy import (Matrix as M, Symbol, symbols, cos, sin, expand, simplify,
                   zeros, eye, diff, trigsimp)

HERE = os.path.dirname(os.path.abspath(__file__))

a, b = symbols('a, b', positive=True)

#: generalized displacement derivatives at a point, physical coordinates
u_x, u_y, v_x, v_y, w_x, w_y, w_xx, w_yy, w_xy = symbols(
    'u_x, u_y, v_x, v_y, w_x, w_y, w_xx, w_yy, w_xy')
D = (u_x, u_y, v_x, v_y, w_x, w_y, w_xx, w_yy, w_xy)
#: the six displacement gradients, the variables of the quadratic strains
GRAD = D[:6]
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


def strains(nonlinear=True):
    r"""Generalized strains ``(exx, eyy, gxy, kxx, kyy, kxy)`` in terms of
    the derivatives ``D``, Green-Lagrange membrane strains"""
    t = 1 if nonlinear else 0
    return M([u_x + t*(u_x**2 + v_x**2 + w_x**2)/2,
              v_y + t*(u_y**2 + v_y**2 + w_y**2)/2,
              u_y + v_x + t*(u_x*u_y + v_x*v_y + w_x*w_y),
              -w_xx,
              -w_yy,
              -2*w_xy])


def strains_von_karman():
    r"""Strains of ``plate_clpt_donnell`` in terms of ``D``"""
    return M([u_x + w_x**2/2,
              v_y + w_y**2/2,
              u_y + v_x + w_x*w_y,
              -w_xx,
              -w_yy,
              -2*w_xy])


def basis(side):
    r"""``[Phi]`` of a term of side ``'A'`` (row, test function) or ``'B'``
    (column, trial function): the 9 x 3 matrix such that ``d = [Phi] c``
    for the three Ritz constants ``c = (c_u, c_v, c_w)`` of the term, with
    the approximation functions in the notation of the kernels, e.g.
    ``fAu_xi`` for `\partial f_i^u/\partial \xi` and ``gAw_etaeta`` for
    `\partial^2 g_j^w/\partial \eta^2`"""
    s = lambda name: Symbol(name.format(side))
    P = zeros(9, 3)
    P[0, 0] = (2/a)*s('f{0}u_xi')*s('g{0}u')
    P[1, 0] = (2/b)*s('f{0}u')*s('g{0}u_eta')
    P[2, 1] = (2/a)*s('f{0}v_xi')*s('g{0}v')
    P[3, 1] = (2/b)*s('f{0}v')*s('g{0}v_eta')
    P[4, 2] = (2/a)*s('f{0}w_xi')*s('g{0}w')
    P[5, 2] = (2/b)*s('f{0}w')*s('g{0}w_eta')
    P[6, 2] = (2/a)**2*s('f{0}w_xixi')*s('g{0}w')
    P[7, 2] = (2/b)**2*s('f{0}w')*s('g{0}w_etaeta')
    P[8, 2] = (2/a)*(2/b)*s('f{0}w_xi')*s('g{0}w_eta')
    return P


def hessian_stress(N):
    r"""``[H(N)] = sum_r N_r d^2 eps_r/d D^2`` for the membrane stresses
    ``N = (Nxx, Nyy, Nxy)``, 9 x 9"""
    eps = strains()
    H = zeros(9, 9)
    for r in range(3):
        H += N[r]*sympy.hessian(eps[r], D)
    return H.applyfunc(expand)


def build():
    r"""Matrices of the derivation at an integration point

    Returns a dict with

    - ``'eps'``, ``'eps_L'``, ``'eps_NL'``: the strains and their linear and
      quadratic parts, in terms of ``D``;
    - ``'B'``, ``'B_L'``: ``d eps/d D`` (6 x 9) and its value at zero
      gradients;
    - ``'F'``: the constitutive matrix;
    - ``'N_NL'``: the membrane stresses of the quadratic strains;
    - ``'H'``: ``[H(N)]`` with the symbols ``Nxx, Nyy, Nxy``;
    - ``'PhiA'``, ``'PhiB'``: the matrices ``[Phi]`` of the two terms;
    - ``'BA'``: ``[B][Phi_A]``, the 6 x 3 strain-displacement matrix of a
      term, comparable to ``BA`` of ``plate_clpt_donnell.py``;
    - ``'GA'``: the 6 x 3 matrix of the gradients of a term.

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
    return dict(eps=eps, eps_L=eps_L, eps_NL=eps_NL, B=B, B_L=B_L, F=F,
                N_NL=N_NL, H=H, PhiA=PhiA, PhiB=PhiB, BA=B*PhiA,
                GA=PhiA[:6, :])


def check_rigid_rotation():
    r"""Zero membrane strains for any rigid rotation, about any axis

    With the rotation matrix of Rodrigues, angle `t` about the unit axis
    `n`, the displacement ``u = (R - I) X`` of the plane ``X = (x, y, 0)``
    has the gradients ``u,x = R[:, 0] - e_x``, ``u,y = R[:, 1] - e_y``;
    the Green-Lagrange strains vanish identically, the von Kármán ones do
    not, e.g. ``exx = cos(t) - 1`` for an in-plane rotation.

    """
    t = Symbol('t', real=True)
    n1, n2, n3 = symbols('n1, n2, n3', real=True)
    n = M([n1, n2, n3])
    K = M([[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]])
    R = eye(3) + sin(t)*K + (1 - cos(t))*K*K
    unit = {n3: sympy.sqrt(1 - n1**2 - n2**2)}
    grads = {u_x: R[0, 0] - 1, v_x: R[1, 0], w_x: R[2, 0],
             u_y: R[0, 1], v_y: R[1, 1] - 1, w_y: R[2, 1]}
    eps = strains()[:3, :].subs(grads).subs(unit)
    for e in eps:
        assert simplify(trigsimp(expand(e))) == 0, e
    # the Green-Lagrange membrane strains are (F^T F - I)/2
    Fg = M([[1 + u_x, u_y], [v_x, 1 + v_y], [w_x, w_y]])
    E = (Fg.T*Fg - eye(2))/2
    gl = strains()
    assert expand(E[0, 0] - gl[0]) == 0
    assert expand(E[1, 1] - gl[1]) == 0
    assert expand(2*E[0, 1] - gl[2]) == 0
    # the von Karman strains of an in-plane rotation, axis n = e_z
    inplane = {n1: 0, n2: 0, n3: 1}
    vk = strains_von_karman()[:3, :].subs(
        {g: v.subs(inplane) for g, v in grads.items()})
    assert simplify(vk[0] - (cos(t) - 1)) == 0
    return True


def check_von_karman_limit():
    r"""Without the in-plane gradients, the strains and ``BA`` are those of
    ``plate_clpt_donnell.py``"""
    m = build()
    inplane0 = {u_x: 0, u_y: 0, v_x: 0, v_y: 0}
    eps_vk = strains_von_karman()
    # the quadratic terms of the in-plane gradients are the only difference
    diff_eps = (m['eps'] - eps_vk).applyfunc(expand)
    assert diff_eps.subs(inplane0) == zeros(6, 1)
    # BA of plate_clpt_donnell.py, with w_xi, w_eta the derivatives of w in
    # the natural coordinates at the point
    w_xi, w_eta = symbols('w_xi, w_eta')
    s = lambda name: Symbol(name)
    swA_xi = M([[0, 0, s('fAw_xi')*s('gAw')]])
    swA_eta = M([[0, 0, s('fAw')*s('gAw_eta')]])
    suA_xi = M([[s('fAu_xi')*s('gAu'), 0, 0]])
    suA_eta = M([[s('fAu')*s('gAu_eta'), 0, 0]])
    svA_xi = M([[0, s('fAv_xi')*s('gAv'), 0]])
    svA_eta = M([[0, s('fAv')*s('gAv_eta'), 0]])
    swA_xixi = M([[0, 0, s('fAw_xixi')*s('gAw')]])
    swA_xieta = M([[0, 0, s('fAw_xi')*s('gAw_eta')]])
    swA_etaeta = M([[0, 0, s('fAw')*s('gAw_etaeta')]])
    BA_vk = M([(2/a)*suA_xi + (2/a)**2*swA_xi*w_xi,
               (2/b)*svA_eta + (2/b)**2*swA_eta*w_eta,
               (2/b)*suA_eta + (2/a)*svA_xi
               + (2/a)*(2/b)*(w_eta*swA_xi + w_xi*swA_eta),
               -(2/a)*(2/a)*swA_xixi,
               -(2/b)*(2/b)*swA_etaeta,
               -(2/a)*(2/b)*2*swA_xieta])
    BA = m['BA'].subs(inplane0).subs({w_x: (2/a)*w_xi, w_y: (2/b)*w_eta})
    assert (BA - BA_vk).applyfunc(expand) == zeros(6, 3)
    return True


def check_tangent():
    r"""``d(B^T sigma)/dD = B^T F B + H(N)``, ``N = N_L + N_NL``, and the
    split into ``fkC_num`` and ``fkG_num``

    Since ``d = [Phi] c`` is linear in ``c``, the identity at a point gives
    ``d fint/d c = K_T`` for any number of terms.

    """
    m = build()
    eps, B, H = m['eps'], m['B'], m['H']
    sigma = F*eps
    r = (B.T*sigma).applyfunc(expand)
    exact = r.jacobian(D).applyfunc(expand)
    N = {Nxx: sigma[0], Nyy: sigma[1], Nxy: sigma[2]}
    KT = (B.T*F*B + H.xreplace(N)).applyfunc(expand)
    assert (exact - KT).applyfunc(expand) == zeros(9, 9)
    # fkC_num with NLgeom: B^T F B + H(N_NL); fkG_num: H(N_L)
    N_L = F[:3, :]*m['eps_L']
    KC = B.T*F*B + H.xreplace({Nxx: m['N_NL'][0], Nyy: m['N_NL'][1],
                                Nxy: m['N_NL'][2]})
    KG = H.xreplace({Nxx: N_L[0], Nyy: N_L[1], Nxy: N_L[2]})
    assert (KC + KG - KT).applyfunc(expand) == zeros(9, 9)
    # KG is homogeneous of degree one in the derivatives
    tt = Symbol('tt')
    KGs = KG.subs({d: tt*d for d in D}, simultaneous=True)
    assert (KGs - tt*KG).applyfunc(expand) == zeros(9, 9)
    return True


def check_tangent_product_form():
    r"""``eps = (B_L + B_NL(beta)/2) d`` with ``B_NL`` linear in the six
    gradients ``beta``, and ``H(N) = G^T diag(N, N, N) G`` on them, the form
    assumed by ``theory/shells/tangent_product``"""
    m = build()
    B, B_L, eps = m['B'], m['B_L'], m['eps']
    B_NL = (B - B_L)
    for e in B_NL:
        assert sympy.Poly(e, *GRAD).total_degree() <= 1
    d = M(D)
    assert ((B_L + B_NL/2)*d - eps).applyfunc(expand) == zeros(6, 1)
    N2 = M([[Nxx, Nxy], [Nxy, Nyy]])
    Nbig = sympy.diag(N2, N2, N2)
    H6 = m['H'][:6, :6]
    assert (H6 - Nbig).applyfunc(expand) == zeros(6, 6)
    assert m['H'][6:, :] == zeros(3, 9) and m['H'][:, 6:] == zeros(9, 3)
    return True


def check_numerical(seed=0):
    r"""Finite differences of ``fint`` of one pair of terms against
    ``[Phi_A]^T K_T [Phi_B]`` at random values"""
    import random
    rnd = random.Random(seed)
    m = build()
    vals = {s: rnd.uniform(-1, 1) for s in F.free_symbols}
    vals.update({a: 0.7, b: 0.4})
    for side in 'AB':
        for s in m['PhiA'].free_symbols | m['PhiB'].free_symbols:
            vals.setdefault(s, rnd.uniform(-1, 1))
    PhiA = m['PhiA'].xreplace(vals)
    PhiB = m['PhiB'].xreplace(vals)
    Fn = F.xreplace(vals)
    # one term per side: d = PhiA cA + PhiB cB, fint of side A
    cA = M([rnd.uniform(-0.3, 0.3) for _ in range(3)])
    cB = M([rnd.uniform(-0.3, 0.3) for _ in range(3)])

    def fint_A(cB_):
        dvals = PhiA*cA + PhiB*cB_
        sub = dict(zip(D, dvals))
        sigma = Fn*m['eps'].xreplace(sub)
        return PhiA.T*m['B'].xreplace(sub).T*sigma

    dvals = PhiA*cA + PhiB*cB
    sub = dict(zip(D, dvals))
    sigma = Fn*m['eps'].xreplace(sub)
    K = (m['B'].T*Fn*m['B'] + m['H'].xreplace(
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


if __name__ == '__main__':
    for check in (check_rigid_rotation, check_von_karman_limit, check_tangent,
                  check_tangent_product_form, check_numerical):
        check()
        print(check.__name__, 'verified')
