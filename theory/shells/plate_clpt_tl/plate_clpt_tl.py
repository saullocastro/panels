r"""Flat plate, total-Lagrangian Kirchhoff-Love (CLPT) with the deformed normal

Kinematics
----------
The degrees of freedom of a term of the Ritz series are ``u, v, w``, the
displacements of the mid-surface along the fixed local axes ``x, y, z`` of
the plate, as in ``plate_clpt_donnell`` and ``plate_clpt_gl``. The position
of the deformed mid-surface is ``X = (x, y, 0) + u`` and its tangent vectors
are the columns of the 3 x 2 deformation gradient
``F = [X,x  X,y] = [e_x + u,x  e_y + u,y]``.

The membrane strains are the Green-Lagrange strains ``E = (F^T F - I)/2`` of
``plate_clpt_gl``::

    exx = u,x + (u,x^2 + v,x^2 + w,x^2)/2
    eyy = v,y + (u,y^2 + v,y^2 + w,y^2)/2
    gxy = u,y + v,x + u,x u,y + v,x v,y + w,x w,y

and the changes of curvature are measured with the normal of the deformed
mid-surface, through its area vector ``a = X,x x X,y``::

    kxx = -a . u,xx,    kyy = -a . u,yy,    kxy = -2 a . u,xy

``u,xx = (u,xx, v,xx, w,xx)`` and so on, the reference plate being flat,
``X,ab = u,ab``. The curvatures are then cubic in the displacement
derivatives; their linear part is ``-w,xx, -w,yy, -2 w,xy``, those of
``plate_clpt_donnell`` and ``plate_clpt_gl``.

Area vector or unit normal
--------------------------
The unit normal is ``n = a/|a|`` with ``|a|^2 = det(F^T F) = (1 + 2 exx)(1 +
2 eyy) - gxy^2`` (:func:`check_area_vector_norm`), such that the curvatures
above are ``|a|`` times the exact changes of the second fundamental form,
``-n . X,ab``: ``|a| = 1 + exx + eyy + O(eps^2)`` is the area stretch of the
mid-surface, which is one for any rigid motion and differs from one by the
order of the membrane strains otherwise. Using ``a`` instead of ``n``

- keeps every integrand polynomial in the approximation functions, such that
  the Gauss-Legendre rule integrates them exactly (``n`` would bring
  ``1/|a|`` and a rule that is never exact);
- leaves the strains exactly invariant under any rigid rotation of a
  deformed state, ``(R A) x (R B) = R (A x B)`` (:func:`check_rigid_rotation`);
- scales the bending stiffness by ``|a|^2 = 1 + 2 (exx + eyy) + O(eps^2)``,
  an error of the order of the membrane strain, e.g. 0.2 % for a membrane
  strain of 1e-3, of the order of the terms already neglected by the
  Kirchhoff-Love kinematics with a constitutive law linear in the
  Green-Lagrange strains.

Internal force vector and tangent stiffness
-------------------------------------------
At a point the strains depend on the fifteen generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y,
         u,xx, u,yy, u,xy, v,xx, v,yy, v,xy, w,xx, w,yy, w,xy) = [Phi] c

With ``sigma = (Nxx, Nyy, Nxy, Mxx, Myy, Mxy) = [ABD] eps`` and ``[B] = d
eps/d d`` (6 x 15, now quadratic in ``d``)::

    fint = int [Phi]^T [B]^T sigma dA

    K_T = int [Phi]^T ([B]^T [ABD] [B] + [H(sigma, d)]) [Phi] dA

    [H(sigma, d)] = sum_r sigma_r d^2 eps_r/d d^2

The membrane part of ``[H]`` is ``diag(N, N, N)`` on the gradients, as in
``plate_clpt_gl``; the bending part, ``Mxx d^2 kxx/d d^2 + ...``, couples the
gradients with the second derivatives and is linear in ``d``
(:func:`build`, ``'H'``).

Split, as in every model of ``panels`` (:func:`check_tangent`):

    K_T = (KC0 + KCNL + KGNL)  [fkC_num]  +  KG(Nbar + sigma_L)  [fkG_num]

    fkG_num: [H0(sigma_L + Nbar)],   H0 = H(., d = 0)
    fkC_num: [B]^T [ABD] [B] + [H(sigma, d)] - [H0(sigma_L)]

with ``sigma_L = [ABD] eps_L`` the stresses of the linear strains and
``Nbar`` a constant membrane stress state. ``fkG_num`` is homogeneous of
degree one in ``c``, as the linear buckling analysis requires. It has the
``u-u``, ``v-v`` and ``w-w`` blocks of ``plate_clpt_gl``, of ``N_L +
Nbar``, and ``u-w`` and ``v-w`` blocks of the moments ``M_L`` of the linear
strains, from the quadratic terms ``w,x u,xx + w,y v,xx - (u,x + v,y)
w,xx`` of ``kxx`` and their analogues. With a constant membrane stress only,
the geometric stiffness is that of ``plate_clpt_gl`` (``fkG0``).

The strains are no longer of the form ``([B_L] + [B_NL(beta)]/2) d`` with
``[B_NL]`` linear in ``beta``, assumed by ``theory/shells/tangent_product``;
the product ``K_T v`` is evaluated from ``[B]`` and ``[H]`` at each point
instead, see ``write_pyx.py``.

Integration
-----------
Along ``x``, an approximation function of degree ``p`` and its first and
second derivatives have the degrees ``p``, ``p - 1`` and ``p - 2``; the
quadratic part of ``a`` has one derivative in ``x`` and one in ``y`` in each
of its terms. The integrands of ``fint`` and ``K_T`` are polynomials of
degree at most ``6p - 2`` along each direction (``Mxx`` cubic times the
variation of ``kxx`` quadratic, each with one derivative), against ``4p``
for ``plate_clpt_gl`` (:func:`check_integrand_degree`). The Gauss-Legendre
rule of ``n`` points is exact for ``2n - 1 >= 6p - 2``, i.e. ``n >= 3p``.

Run ``python plate_clpt_tl.py`` to run the checks.

"""
import os
import random
import sys

import sympy
from sympy import (Matrix as M, Symbol, symbols, cos, sin, expand, zeros,
                   eye, sqrt)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'plate_clpt_gl'))

import plate_clpt_gl as gl

a, b = gl.a, gl.b
u_x, u_y, v_x, v_y, w_x, w_y = gl.GRAD
w_xx, w_yy, w_xy = gl.w_xx, gl.w_yy, gl.w_xy
u_xx, u_yy, u_xy, v_xx, v_yy, v_xy = symbols(
    'u_xx, u_yy, u_xy, v_xx, v_yy, v_xy')
#: the six displacement gradients
GRAD = gl.GRAD
#: the nine second derivatives of the displacements
SECOND = (u_xx, u_yy, u_xy, v_xx, v_yy, v_xy, w_xx, w_yy, w_xy)
#: generalized displacement derivatives at a point, physical coordinates
D = GRAD + SECOND
#: names of the generalized derivatives in the kernels
D_NAMES = tuple(s.name for s in D)
FIELDS = gl.FIELDS
DOF = gl.DOF
NE = gl.NE

F = gl.F
Nxx, Nyy, Nxy = gl.Nxx, gl.Nyy, gl.Nxy
Mxx, Myy, Mxy = gl.Mxx, gl.Myy, gl.Mxy
STRESSES = gl.STRESSES


def deformation_gradient():
    r"""``F = [e_x + u,x  e_y + u,y]``, 3 x 2"""
    return M([[1 + u_x, u_y], [v_x, 1 + v_y], [w_x, w_y]])


def area_vector():
    r"""``a = X,x x X,y``, the normal of the deformed mid-surface times its
    area stretch"""
    Fg = deformation_gradient()
    return Fg[:, 0].cross(Fg[:, 1])


def strains(nonlinear=True):
    r"""Generalized strains ``(exx, eyy, gxy, kxx, kyy, kxy)`` in terms of
    the derivatives ``D``: Green-Lagrange membrane strains and the changes
    of curvature of the area vector, or their linear part"""
    membrane = gl.strains(nonlinear)[:3, :]
    av = area_vector() if nonlinear else M([0, 0, 1])
    uxx = M([u_xx, v_xx, w_xx])
    uyy = M([u_yy, v_yy, w_yy])
    uxy = M([u_xy, v_xy, w_xy])
    return M([membrane[0], membrane[1], membrane[2],
              expand(-av.dot(uxx)),
              expand(-av.dot(uyy)),
              expand(-2*av.dot(uxy))])


def basis(side):
    r"""``[Phi]`` of a term of side ``'A'`` or ``'B'``: the 15 x 3 matrix
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
    return P


def hessian_stress(sigma, at_zero=False):
    r"""``[H(sigma, d)] = sum_r sigma_r d^2 eps_r/d D^2``, 15 x 15, linear
    in ``d``; with ``at_zero`` its value ``[H0(sigma)]`` at ``d = 0``"""
    eps = strains()
    H = zeros(len(D), len(D))
    for r in range(NE):
        H += sigma[r]*sympy.hessian(eps[r], D)
    if at_zero:
        H = H.subs({d: 0 for d in D})
    return H.applyfunc(expand)


def build():
    r"""Matrices of the derivation at an integration point

    Returns a dict with

    - ``'eps'``, ``'eps_L'``, ``'eps_NL'``: the strains and their linear and
      non-linear parts, in terms of ``D``;
    - ``'B'``, ``'B_L'``: ``d eps/d D`` (6 x 15) and its value at ``d = 0``;
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
    assert B_L == eps_L.jacobian(D)
    H = hessian_stress(STRESSES)
    H0 = hessian_stress(STRESSES, at_zero=True)
    return dict(eps=eps, eps_L=eps_L, eps_NL=eps_NL, B=B, B_L=B_L, F=F,
                H=H, H0=H0, PhiA=basis('A'), PhiB=basis('B'))


def check_membrane_and_linear_part():
    r"""The membrane strains are those of ``plate_clpt_gl``, the linear
    strains those of ``plate_clpt_donnell`` and ``plate_clpt_gl``, and the
    quadratic part of the curvatures is ``w,x u,xx + w,y v,xx - (u,x + v,y)
    w,xx`` for ``kxx``"""
    eps = strains()
    assert (eps[:3, :] - gl.strains()[:3, :]).applyfunc(expand) == zeros(3, 1)
    assert strains(nonlinear=False) == gl.strains(nonlinear=False)
    t = Symbol('t')
    kxx = eps[3].subs({d: t*d for d in D}, simultaneous=True)
    quad = expand(sympy.diff(kxx, t, 2).subs(t, 0)/2)
    assert expand(quad - (w_x*u_xx + w_y*v_xx - (u_x + v_y)*w_xx)) == 0
    return True


def check_area_vector_norm():
    r"""``|a|^2 = det(F^T F) = (1 + 2 exx)(1 + 2 eyy) - gxy^2``: ``|a| = 1``
    whenever the membrane strains vanish, and ``1 + exx + eyy + O(eps^2)``
    otherwise"""
    av = area_vector()
    Fg = deformation_gradient()
    C = Fg.T*Fg
    eps = strains()
    assert expand(av.dot(av) - C.det()) == 0
    assert expand(C.det() - ((1 + 2*eps[0])*(1 + 2*eps[1]) - eps[2]**2)) == 0
    return True


def _rotated(values, R):
    r"""Derivatives of the displacement of the configuration ``R X``, with
    ``X = (x, y, 0) + u`` of ``values``: ``X',a = R X,a`` and ``u',ab = R
    u,ab``"""
    Fg = deformation_gradient().xreplace(values)
    Fr = R*Fg
    out = {u_x: Fr[0, 0] - 1, v_x: Fr[1, 0], w_x: Fr[2, 0],
           u_y: Fr[0, 1], v_y: Fr[1, 1] - 1, w_y: Fr[2, 1]}
    for trio in ((u_xx, v_xx, w_xx), (u_yy, v_yy, w_yy), (u_xy, v_xy, w_xy)):
        vec = R*M([values[s] for s in trio])
        out.update(dict(zip(trio, vec)))
    return out


def check_rigid_rotation(seed=0):
    r"""The strains are invariant under a rigid rotation ``R`` of any
    deformed configuration, ``X -> R X``

    Symbolically for a rotation of angle ``t`` about each axis ``x``, ``y``
    and ``z``, with the derivatives ``D`` arbitrary, reducing with
    ``cos(t)^2 + sin(t)^2 = 1``; numerically for oblique axes. The
    curvatures ``-w,ab`` of ``plate_clpt_gl`` of a bent state rotated about
    ``y`` are those of the bent state times ``cos(t)``.

    """
    c, s = symbols('c, s')
    unit = c**2 + s**2 - 1
    Rs = [M([[1, 0, 0], [0, c, -s], [0, s, c]]),
          M([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
          M([[c, -s, 0], [s, c, 0], [0, 0, 1]])]
    eps = strains()
    values = {d: d for d in D}
    for R in Rs:
        rot = _rotated(values, R)
        for k in range(NE):
            diff = expand(eps[k].xreplace(rot) - eps[k])
            assert sympy.rem(diff, unit, c) == 0, (R, k)
    # oblique axes, numerically
    rnd = random.Random(seed)
    for _ in range(5):
        n = M([sympy.Float(rnd.uniform(-1, 1), 30) for _ in range(3)])
        n = n/sqrt(n.dot(n))
        t = sympy.Float(rnd.uniform(0.1, 3.), 30)
        K = M([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        R = (eye(3) + sin(t)*K + (1 - cos(t))*K*K).evalf(30)
        values = {d: sympy.Float(rnd.uniform(-0.5, 0.5), 30) for d in D}
        rot = _rotated(values, R)
        for k in range(NE):
            e0 = eps[k].xreplace(values)
            e1 = eps[k].xreplace(rot)
            assert abs(e1 - e0) < 1e-25
    # plate_clpt_gl: kxx of a bent state rotated about y, cos(t) of kxx
    t = Symbol('t')
    Ry = M([[cos(t), 0, sin(t)], [0, 1, 0], [-sin(t), 0, cos(t)]])
    bent = {d: 0 for d in D}
    bent[w_xx] = Symbol('k')
    rot = _rotated(bent, Ry)
    assert sympy.simplify(gl.strains()[3].xreplace(rot) - cos(t)*(-Symbol('k'))) == 0
    assert sympy.simplify(eps[3].xreplace(rot) + Symbol('k')) == 0
    return True


def check_tangent():
    r"""``d(B^T sigma)/dD = B^T F B + H(sigma, d)``, ``sigma = F eps``, and
    the split into ``fkC_num`` and ``fkG_num``

    Since ``d = [Phi] c`` is linear in ``c``, the identity at a point gives
    ``d fint/d c = K_T`` for any number of terms. With a symbolic ``F`` the
    expansion is long, so the identity is checked with ``F`` symbolic and
    ``d`` symbolic, but one row of ``B^T sigma`` at a time.

    """
    m = build()
    eps, B, H, H0 = m['eps'], m['B'], m['H'], m['H0']
    sigma = (F*eps).applyfunc(expand)
    sub = dict(zip(STRESSES, sigma))
    KT = (B.T*F*B + H.xreplace(sub))
    for p in range(len(D)):
        r = expand((B[:, p].T*sigma)[0, 0])
        for q in range(len(D)):
            assert expand(sympy.diff(r, D[q]) - KT[p, q]) == 0, (p, q)
    # fkG_num: H0(sigma_L), homogeneous of degree one in d
    sigma_L = F*m['eps_L']
    KG = H0.xreplace(dict(zip(STRESSES, sigma_L))).applyfunc(expand)
    tt = Symbol('tt')
    KGs = KG.subs({d: tt*d for d in D}, simultaneous=True)
    assert (KGs - tt*KG).applyfunc(expand) == zeros(len(D), len(D))
    # fkC_num: B^T F B + H(sigma, d) - H0(sigma_L), KC0 at d = 0
    KC = B.T*F*B + H.xreplace(sub) - KG
    assert (KC + KG - KT).applyfunc(expand) == zeros(len(D), len(D))
    zero = {d: 0 for d in D}
    assert (KC.subs(zero) - m['B_L'].T*F*m['B_L']).applyfunc(expand) == zeros(len(D), len(D))
    # with a constant membrane stress only, H0 is the H of plate_clpt_gl
    Nonly = {Mxx: 0, Myy: 0, Mxy: 0}
    Hgl = gl.hessian_stress((Nxx, Nyy, Nxy))
    idx = [D.index(s) for s in gl.D]
    assert H0.xreplace(Nonly)[idx, idx] == Hgl
    rest = [k for k in range(len(D)) if k not in idx]
    assert H0.xreplace(Nonly)[rest, :] == zeros(len(rest), len(D))
    return True


def check_numerical(seed=0):
    r"""Finite differences of ``fint`` of one pair of terms against
    ``[Phi_A]^T K_T [Phi_B]`` at random values"""
    rnd = random.Random(seed)
    m = build()
    vals = {s: rnd.uniform(-1, 1) for s in F.free_symbols}
    vals.update({a: 0.7, b: 0.4})
    for s in m['PhiA'].free_symbols | m['PhiB'].free_symbols:
        vals.setdefault(s, rnd.uniform(-1, 1))
    PhiA = m['PhiA'].xreplace(vals)
    PhiB = m['PhiB'].xreplace(vals)
    Fn = F.xreplace(vals)
    cA = M([rnd.uniform(-0.3, 0.3) for _ in range(3)])
    cB = M([rnd.uniform(-0.3, 0.3) for _ in range(3)])

    def state(cB_):
        sub = dict(zip(D, PhiA*cA + PhiB*cB_))
        sigma = Fn*m['eps'].xreplace(sub)
        return sub, sigma

    def fint_A(cB_):
        sub, sigma = state(cB_)
        return PhiA.T*m['B'].xreplace(sub).T*sigma

    sub, sigma = state(cB)
    K = (m['B'].T*Fn*m['B'] + m['H'].xreplace(dict(zip(STRESSES, sigma)))).xreplace(sub)
    KAB = PhiA.T*K*PhiB
    h = 1e-6
    for e in range(3):
        dc = zeros(3, 1)
        dc[e] = h
        fd = (fint_A(cB + dc) - fint_A(cB - dc))/(2*h)
        for dd in range(3):
            assert abs(float(fd[dd] - KAB[dd, e])) < 1e-6*(1 + abs(float(KAB[dd, e])))
    return True


def x_degree(model='tl'):
    r"""Degree along ``x`` of the integrands of ``fint`` and ``K_T``, as
    ``(k, j)`` for ``k p + j``, with the approximation functions of degree
    ``p``: a derivative ``u,x`` has the degree ``p - 1``, ``u,xx`` ``p - 2``
    and ``u,y`` ``p``; by symmetry the degree along ``y`` is the same"""
    if model == 'tl':
        eps, Dm = strains(), D
    else:
        eps, Dm = gl.strains(), gl.D
    deg = {s: -s.name.count('x') for s in Dm}
    B = eps.jacobian(Dm)
    sigma = F*eps
    # K_T entries [p, q] times Phi_A[p] Phi_B[q]; fint entries [p] times
    # Phi_A[p]: the terms of fint are those of K_T with one Phi_B replaced
    # by a derivative of the same order, so K_T suffices
    sub = dict(zip(STRESSES, sigma))
    if model == 'tl':
        H = hessian_stress(STRESSES).xreplace(sub)
    else:
        H = gl.hessian_stress((Nxx, Nyy, Nxy)).xreplace(sub)
    KT = (B.T*F*B + H)
    best = None
    for p in range(len(Dm)):
        for q in range(len(Dm)):
            e = expand(KT[p, q])
            if e == 0:
                continue
            for mono in sympy.Poly(e, *Dm).monoms():
                k = sum(mono) + 2
                j = sum(n*deg[s] for n, s in zip(mono, Dm)) + deg[Dm[p]] + deg[Dm[q]]
                if best is None or (k, j) > best:
                    best = (k, j)
    return best


def check_integrand_degree():
    r"""Degree ``6p - 2`` along each direction for ``plate_clpt_tl``, ``4p``
    for ``plate_clpt_gl``"""
    assert x_degree('tl') == (6, -2)
    assert x_degree('gl') == (4, 0)
    return True


if __name__ == '__main__':
    for check in (check_membrane_and_linear_part, check_area_vector_norm,
                  check_rigid_rotation, check_tangent, check_numerical,
                  check_integrand_degree):
        check()
        print(check.__name__, 'verified')
