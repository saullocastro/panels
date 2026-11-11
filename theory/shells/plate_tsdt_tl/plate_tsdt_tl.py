r"""Flat plate, total-Lagrangian TSDT (``plate_tsdt_tl``)

Derivation of the strains, of their derivatives and of the matrices at an
integration point, with symbolic checks, run ``python plate_tsdt_tl.py``
(about two minutes). The kernels are written by ``write_pyx.py``.

Kinematics
----------
Degrees of freedom of a term `u, v, w, \phi_x, \phi_y` along the fixed local
axes `x, y, z`, as in ``plate_tsdt_donnell``. Reddy's displacement field,

.. math::

    \mathbf{u}(z) = \mathbf{u} + z (\phi_x, \phi_y, 0)
    - c_1 z^3 (\phi_x + w_{,x}, \; \phi_y + w_{,y}, \; 0), \qquad
    c_1 = \frac{4}{3 h^2}

is written for finite rotations with the deformed mid-surface `\mathbf{X} =
(x + u, y + v, w)`, the second-order director of ``plate_fsdt_tl``,
`\mathbf{d} = (\phi_x, \phi_y, 1 - (\phi_x^2 + \phi_y^2)/2)`, and the normal
of the deformed mid-surface, not normalized, `\mathbf{N} = \mathbf{X}_{,x}
\times \mathbf{X}_{,y}`. The vector of the third-order term of Reddy, `(\phi_x
+ w_{,x}, \phi_y + w_{,y}, 0)`, is the linear part of `\mathbf{d} -
\mathbf{N}`, and the deformed position of a point is

.. math::

    \mathbf{r}(z) = \mathbf{X} + z \mathbf{d} + z^3 \mathbf{q}, \qquad
    \mathbf{q} = -c_1 (\mathbf{d} - \mathbf{N})

The strains are the coefficients of `z^0`, `z^1`, `z^3` of the
Green-Lagrange strains `E_{\alpha\beta}(z) = (\mathbf{G}_\alpha \cdot
\mathbf{G}_\beta - \delta_{\alpha\beta})/2`, `\mathbf{G}_\alpha =
\mathbf{r}_{,\alpha}`, and of `z^0`, `z^2` of `2 E_{\alpha 3}(z) =
\mathbf{G}_\alpha \cdot \mathbf{r}_{,z}`, the powers of the linear TSDT:
the membrane strains, the curvatures `\kappa` and the transverse shear
strains `\gamma` of ``plate_fsdt_tl`` and, since `\mathbf{X}_{,\alpha} \cdot
\mathbf{N} = 0` and therefore `\mathbf{X}_{,\alpha} \cdot \mathbf{N}_{,\beta}
= -\mathbf{X}_{,\alpha\beta} \cdot \mathbf{N}`,

.. math::

    \kappa^{(3)}_{xx} = -c_1 (\kappa_{xx} + \mathbf{X}_{,xx} \cdot
    \mathbf{N}) \quad
    \kappa^{(3)}_{yy} = -c_1 (\kappa_{yy} + \mathbf{X}_{,yy} \cdot
    \mathbf{N}) \quad
    \kappa^{(3)}_{xy} = -c_1 (\kappa_{xy} + 2 \mathbf{X}_{,xy} \cdot
    \mathbf{N})

    \gamma^{(2)}_\alpha = 3 \mathbf{X}_{,\alpha} \cdot \mathbf{q}
    = -3 c_1 \gamma_\alpha

Reddy's `\phi_{\alpha,\beta}` becomes the director curvature and
`w_{,\alpha\beta}` the second fundamental form of the deformed mid-surface,
`\mathbf{X}_{,\alpha\beta} \cdot \mathbf{N}`. The transverse shear strain
`\gamma_\alpha (1 - 4 z^2/h^2)` vanishes at the faces `z = \pm h/2` for any
deformation (Reddy's condition), and every strain is invariant under a
rigid rotation up to the error of the second-order director. Dropped, as in
the linear TSDT: the coefficients of `z^2`, `z^4`, `z^6` of
`E_{\alpha\beta}`, of `z^1`, `z^3`, `z^5` of `2 E_{\alpha 3}` and `E_{33}`.
The coefficient of `z^3` of `2 E_{\alpha 3}`, `\mathbf{q}_{,\alpha} \cdot
\mathbf{d} + 3 \mathbf{d}_{,\alpha} \cdot \mathbf{q}`, has the linear part
`c_1 (u_{,x} + v_{,y})_{,\alpha}`, from the stretch of `\mathbf{N}`
(``check_strain_measures``). The second derivatives `u_{,\alpha\beta}`,
`v_{,\alpha\beta}` enter through `\mathbf{X}_{,\alpha\beta}`.

The alternative with linear third-order terms
---------------------------------------------
Keeping the first-order part of ``plate_fsdt_tl`` with the linear
third-order strains of Reddy, those of ``plate_tsdt_gl``,
`\kappa^{(3)}_{\alpha\beta} = -c_1 (\phi_{\alpha,\beta} + \phi_{\beta,\alpha}
+ 2 w_{,\alpha\beta})/(1 + \delta_{\alpha\beta})`, `\gamma^{(2)}_\alpha =
-3 c_1 (\phi_\alpha + w_{,\alpha})` (``option='A'`` of :func:`strains`),
has the same linear part but not the properties above: the shear strain at
the faces is the difference between the shear strain of the director and
the linear one, `\mathbf{X}_{,\alpha} \cdot \mathbf{d} - (\phi_\alpha +
w_{,\alpha})`, cubic (``check_face_shear``), and a rigid rotation `t` about
an oblique axis gives third-order strains of `O(t^2)`
(``check_rigid_rotation``). In large bending the two shear strains cannot
both vanish and both enter the transverse shear energy, which then
penalizes the bending by `(L/h)^2`: the alternative locks.

At a point the strains depend on the 21 generalized derivatives ``D``,
``[B] = d eps/d D`` (13 x 21, quadratic), ``fint = int [Phi]^T [B]^T sigma
dA``, ``K_T = int [Phi]^T ([B]^T [F] [B] + [H(sigma, D)]) [Phi] dA``,
``[H(sigma, D)] = sum_r sigma_r d^2 eps_r/d D^2``, linear in ``D``, as
``plate_fsdt_tl``.

"""
import importlib.util
import math
import os
import random

import sympy
from sympy import Matrix as M, Symbol, symbols, expand, zeros, eye

HERE = os.path.dirname(os.path.abspath(__file__))

a, b = symbols('a, b', positive=True)

(u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy, phix_x, phix_y, phiy_x,
 phiy_y, w_xx, w_yy, w_xy) = symbols(
     'u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy, phix_x, phix_y, phiy_x, '
     'phiy_y, w_xx, w_yy, w_xy')
u_xx, u_yy, u_xy, v_xx, v_yy, v_xy = symbols('u_xx, u_yy, u_xy, v_xx, '
                                             'v_yy, v_xy')
#: generalized derivatives at a point, physical coordinates
D = (u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy, phix_x, phix_y, phiy_x,
     phiy_y, w_xx, w_yy, w_xy, u_xx, u_yy, u_xy, v_xx, v_yy, v_xy)
D_NAMES = tuple(s.name for s in D)
#: field and orders of the derivatives along x and y
D_DEF = {nm: (nm.partition('_')[0], nm.partition('_')[2].count('x'),
              nm.partition('_')[2].count('y')) for nm in D_NAMES}
D_DEF.update({'phix': ('phix', 0, 0), 'phiy': ('phiy', 0, 0)})
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

#: constitutive matrix of the TSDT, that of ``plate_tsdt_gl``
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
    r"""``X,x``, ``X,y`` of the deformed mid-surface"""
    return M([1 + u_x, v_x, w_x]), M([u_y, 1 + v_y, w_y])


def second_derivatives():
    r"""``X,xx``, ``X,yy``, ``X,xy``"""
    return M([u_xx, v_xx, w_xx]), M([u_yy, v_yy, w_yy]), M([u_xy, v_xy, w_xy])


def director():
    r"""Second-order director ``d`` and ``d,x``, ``d,y``"""
    dz = 1 - (phix**2 + phiy**2)/2
    dz_x = -(phix*phix_x + phiy*phiy_x)
    dz_y = -(phix*phix_y + phiy*phiy_y)
    return (M([phix, phiy, dz]), M([phix_x, phiy_x, dz_x]),
            M([phix_y, phiy_y, dz_y]))


def normal():
    r"""``N = X,x x X,y`` and ``N,x``, ``N,y``"""
    Xx, Xy = tangents()
    Xxx, Xyy, Xxy = second_derivatives()
    return Xx.cross(Xy), Xxx.cross(Xy) + Xx.cross(Xxy), Xxy.cross(Xy) + Xx.cross(Xyy)


def first_order():
    r"""Membrane, bending and transverse shear strains of ``plate_fsdt_tl``"""
    Xx, Xy = tangents()
    d, d_x, d_y = director()
    return (M([(Xx.dot(Xx) - 1)/2, (Xy.dot(Xy) - 1)/2, Xx.dot(Xy)]),
            M([Xx.dot(d_x), Xy.dot(d_y), Xx.dot(d_y) + Xy.dot(d_x)]),
            M([Xy.dot(d), Xx.dot(d)]))


def third_order_linear():
    r"""The linear third-order strains of Reddy, those of ``plate_tsdt_gl``
    (the alternative, ``option='A'``)"""
    return (M([-c1*(phix_x + w_xx), -c1*(phiy_y + w_yy),
               -c1*(phix_y + phiy_x + 2*w_xy)]),
            M([-3*c1*(phiy + w_y), -3*c1*(phix + w_x)]))


def third_order(option='B'):
    r"""``(kappa3, gamma2)``: of the model, built with the director and the
    deformed normal (``option='B'``), or the linear ones of the alternative
    (``option='A'``)"""
    if option == 'A':
        return third_order_linear()
    _, k, g = first_order()
    Xxx, Xyy, Xxy = second_derivatives()
    N = normal()[0]
    return (M([-c1*(k[0] + Xxx.dot(N)), -c1*(k[1] + Xyy.dot(N)),
               -c1*(k[2] + 2*Xxy.dot(N))]), -3*c1*g)


def strains(nonlinear=True, option='B'):
    r"""``eps = (exx, eyy, gxy, kxx, kyy, kxy, kxx3, kyy3, kxy3, gyz, gxz,
    gyz2, gxz2)``; with ``nonlinear=False`` the linear part; ``option='A'``
    the alternative with linear third-order strains"""
    m, k, g = first_order()
    k3, g2 = third_order(option)
    eps = M(list(m) + list(k) + list(k3) + list(g) + list(g2)).applyfunc(expand)
    if nonlinear:
        return eps
    t = Symbol('t')
    return eps.subs({s: t*s for s in D}, simultaneous=True).applyfunc(
        lambda e: expand(e).coeff(t, 1)).applyfunc(expand)


def basis(side):
    r"""``[Phi]`` of a term of side ``'A'`` or ``'B'``: ``D = [Phi] c``, with
    the approximation functions in the notation of the kernels, e.g.
    ``fAu_xixi`` for `\partial^2 f_i^u/\partial \xi^2`"""
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


def check_strain_measures():
    r"""The first-order strains are those of ``plate_fsdt_tl``; the
    third-order strains are the coefficients of `z^3` of `E_{\alpha\beta}`
    and `z^2` of `2 E_{\alpha 3}` of `\mathbf{X} + z \mathbf{d} + z^3
    \mathbf{q}`; the linear part of the dropped coefficient of `z^3` of `2
    E_{\alpha 3}` is `c_1 (u_{,x} + v_{,y})_{,\alpha}`"""
    ft = _load('plate_fsdt_tl', '..', 'plate_fsdt_tl', 'plate_fsdt_tl.py')
    sub = {getattr(ft, s.name): s for s in ft.D}
    eps = strains()
    e8 = ft.strains().xreplace(sub)
    for k, j in enumerate((0, 1, 2, 3, 4, 5, 9, 10)):
        assert expand(eps[j] - e8[k]) == 0
    z = Symbol('z')
    Xx, Xy = tangents()
    d, d_x, d_y = director()
    N, N_x, N_y = normal()
    q, q_x, q_y = -c1*(d - N), -c1*(d_x - N_x), -c1*(d_y - N_y)
    Gx, Gy = Xx + z*d_x + z**3*q_x, Xy + z*d_y + z**3*q_y
    G3 = d + 3*z**2*q
    E = [expand((Gx.dot(Gx) - 1)/2), expand((Gy.dot(Gy) - 1)/2),
         expand(Gx.dot(Gy))]
    for k in range(3):
        assert expand(E[k].coeff(z, 0) - eps[k]) == 0
        assert expand(E[k].coeff(z, 1) - eps[3 + k]) == 0
        assert expand(E[k].coeff(z, 3) - eps[6 + k]) == 0
    S = [expand(Gy.dot(G3)), expand(Gx.dot(G3))]
    t = Symbol('t')
    lin = lambda e: expand(e.subs({s: t*s for s in D}, simultaneous=True)).coeff(t, 1)
    for k, (Gk, ux_a, vy_a) in enumerate(((S[0], u_xy, v_yy), (S[1], u_xx, v_xy))):
        assert expand(Gk.coeff(z, 0) - eps[9 + k]) == 0
        assert expand(Gk.coeff(z, 2) - eps[11 + k]) == 0
        assert expand(lin(Gk.coeff(z, 3)) - c1*(ux_a + vy_a)) == 0
    return True


def check_linear_part():
    r"""``[B(0)] [Phi]`` equal to ``B0`` of ``plate_tsdt_donnell``; the
    linear strains those of ``plate_tsdt_gl`` (also those of the
    alternative ``option='A'``) and the membrane strains those of
    ``plate_tsdt_gl``"""
    lm = _load('plate_tsdt_donnell', '..', 'plate_tsdt_donnell', 'plate_tsdt_donnell.py')
    m = build()
    BL = m['B'].subs({s: 0 for s in D})
    B0 = lm.shape_functions('A')['B0'].xreplace({lm.a: a, lm.b: b, lm.c1: c1}
                                                if hasattr(lm, 'c1') else {lm.a: a, lm.b: b})
    assert (BL*m['PhiA'] - B0).applyfunc(expand) == zeros(NE, DOF)
    gl = _load('plate_tsdt_gl', '..', 'plate_tsdt_gl', 'plate_tsdt_gl.py')
    glsub = {getattr(gl, s.name): s for s in gl.D}
    glsub[gl.c1] = c1
    epsL_gl = gl.strains(nonlinear=False).xreplace(glsub)
    assert (epsL_gl - m['eps_L']).applyfunc(expand) == zeros(NE, 1)
    assert (epsL_gl - strains(False, 'A')).applyfunc(expand) == zeros(NE, 1)
    assert (gl.strains().xreplace(glsub)[:3, :] - m['eps'][:3, :]).applyfunc(expand) == zeros(3, 1)
    return True


def face_shear(option='B'):
    r"""Transverse shear strains at the faces, `\gamma_\alpha + (h^2/4)
    \gamma^{(2)}_\alpha` with `h^2 = 4/(3 c_1)`"""
    eps = strains(option=option)
    return [expand(eps[9 + k] + eps[11 + k]/(3*c1)) for k in range(2)]


def check_face_shear():
    r"""Zero for any deformation; for the alternative ``option='A'`` the
    difference between the shear strain of the director and the linear
    one, cubic in ``D``"""
    assert face_shear() == [0, 0]
    _, _, g = first_order()
    gL = M([phiy + w_y, phix + w_x])
    fsA = face_shear('A')
    assert [expand(fsA[k] - g[k] + gL[k]) for k in range(2)] == [0, 0]
    assert all(e != 0 for e in fsA)
    return True


def _rotation(n, t):
    K = M([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return eye(3) + sympy.sin(t)*K + (1 - sympy.cos(t))*K*K


def _state(Xx, Xy, Xxx, Xxy, Xyy, nrm, nrm_x, nrm_y):
    r"""Generalized derivatives of the state given by the tangent vectors,
    their derivatives and the unit normal with its derivatives, the director
    components being those of the normal"""
    return {u_x: Xx[0] - 1, v_x: Xx[1], w_x: Xx[2],
            u_y: Xy[0], v_y: Xy[1] - 1, w_y: Xy[2],
            phix: nrm[0], phiy: nrm[1], phix_x: nrm_x[0], phiy_x: nrm_x[1],
            phix_y: nrm_y[0], phiy_y: nrm_y[1],
            w_xx: Xxx[2], w_yy: Xyy[2], w_xy: Xxy[2],
            u_xx: Xxx[0], v_xx: Xxx[1], u_yy: Xyy[0], v_yy: Xyy[1],
            u_xy: Xxy[0], v_xy: Xxy[1]}


GROUPS = {'membrane': (0, 1, 2), 'kappa': (3, 4, 5), 'kappa3': (6, 7, 8),
          'gamma': (9, 10), 'gamma2': (11, 12)}


def rigid_rotation_residuals(option='B', seed=0, c1v=1.):
    r"""Largest strain of each group (``GROUPS``) for a rigid rotation `t =
    0.2, 0.1` of the undeformed plate about random axes, and largest change
    of each group when a cylindrical bending state of curvature `k_0 = 0.3`
    is rotated by `t` about random in-plane axes; ``c1 = c1v``"""
    rnd = random.Random(seed)
    eps = strains(option=option).xreplace({c1: c1v})

    def ev(vals):
        return [float(x.xreplace(vals)) for x in eps]

    def axis(inplane=False):
        nn = [rnd.uniform(-1, 1) for _ in range(3)]
        if inplane:
            nn[2] = 0.
        nrm = sum(c**2 for c in nn)**0.5
        return [c/nrm for c in nn]
    z3 = M([0, 0, 0])
    out = {'undeformed': {g: [0., 0.] for g in GROUPS},
           'bent': {g: [0., 0.] for g in GROUPS}}
    for _ in range(5):
        n = axis()
        for i, t in enumerate((0.2, 0.1)):
            R = _rotation(n, t)
            e = ev(_state(R[:, 0], R[:, 1], z3, z3, z3, R[:, 2], z3, z3))
            for g, idx in GROUPS.items():
                out['undeformed'][g][i] = max(out['undeformed'][g][i],
                                              max(abs(e[k]) for k in idx))
    k0 = 0.3
    for _ in range(5):
        al = rnd.uniform(0, 2*math.pi)
        cs, sn = math.cos(al), math.sin(al)
        e1 = M([cs, sn, 0])
        ez = M([0, 0, 1])
        Xx, Xy = M([1, 0, 0]), M([0, 1, 0])
        Xss, nrm_s = k0*ez, -k0*e1
        Xxx, Xxy, Xyy = cs*cs*Xss, cs*sn*Xss, sn*sn*Xss
        nrm_x, nrm_y = cs*nrm_s, sn*nrm_s
        base = ev(_state(Xx, Xy, Xxx, Xxy, Xyy, ez, nrm_x, nrm_y))
        n = axis(inplane=True)
        for i, t in enumerate((0.2, 0.1)):
            R = _rotation(n, t)
            e = ev(_state(R*Xx, R*Xy, R*Xxx, R*Xxy, R*Xyy, R*ez, R*nrm_x, R*nrm_y))
            for g, idx in GROUPS.items():
                out['bent'][g][i] = max(out['bent'][g][i],
                                        max(abs(e[k] - base[k]) for k in idx))
    return out


def check_rigid_rotation():
    r"""Every group of strains `O(t^4)` or smaller (undeformed plate: zero
    but the shear, `O(t^5)`); the alternative ``option='A'``: the
    third-order shear strains of the undeformed plate and the third-order
    strains of the bent state of `O(t^2)`"""
    order = lambda v: math.log2(v[0]/v[1]) if v[1] > 1e-14 else 99
    out = {}
    for option in ('B', 'A'):
        res = rigid_rotation_residuals(option)
        und, bent = res['undeformed'], res['bent']
        for g in ('membrane', 'kappa'):
            assert und[g][0] < 1e-13
        assert order(und['gamma']) > 4.7
        assert order(bent['membrane']) == 99 or bent['membrane'][0] < 1e-13
        assert order(bent['kappa']) > 3.8 and order(bent['gamma']) > 3.8
        if option == 'B':
            assert und['kappa3'][0] < 1e-13 and order(und['gamma2']) > 4.7
            assert order(bent['kappa3']) > 3.8 and order(bent['gamma2']) > 3.8
        else:
            assert 1.8 < order(und['gamma2']) < 2.2
            assert 1.8 < order(bent['kappa3']) < 2.2
        out[option] = res
    return out


def check_tangent():
    r"""``d(B^T sigma)/dD = B^T F B + H(sigma, D)``, the split of
    ``fkC_num``/``fkG_num``, ``fkG_num`` homogeneous of degree one, and
    ``H0`` with membrane stresses only equal to the Hessian of
    ``plate_tsdt_gl`` (whose analytical ``fkG0`` the model uses)"""
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
    gl = _load('plate_tsdt_gl', '..', 'plate_tsdt_gl', 'plate_tsdt_gl.py')
    Hgl = gl.build()['H']
    idx = [D_NAMES.index(s.name) for s in gl.D]
    sub = {getattr(gl, s.name): s for s in STRESSES[:3]}
    memb = {s: 0 for s in STRESSES[3:]}
    H0m = H0.xreplace(memb)
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
    vals.update({a: 0.7, b: 0.4, c1: 0.6})
    for s in m['PhiA'].free_symbols | m['PhiB'].free_symbols:
        vals.setdefault(s, rnd.uniform(-1, 1))
    PhiA, PhiB, Fn = [x.xreplace(vals) for x in (m['PhiA'], m['PhiB'], F)]
    eps, Bn, Hn = [m[k].xreplace({c1: vals[c1]}) for k in ('eps', 'B', 'H')]
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


def x_degree():
    r"""Degree along ``x`` of the integrands of ``K_T``, ``(k, j)`` for ``k
    p + j`` with approximation functions of degree ``p``"""
    deg = {s: -D_DEF[s.name][1] for s in D}
    m = build()
    KT = m['B'].T*F*m['B'] + m['H'].xreplace(dict(zip(STRESSES, F*m['eps'])))
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
    r"""Degree ``6p`` along each direction, as ``plate_fsdt_tl``"""
    assert x_degree() == (6, 0), x_degree()
    return True


CHECKS = (check_strain_measures, check_linear_part, check_face_shear,
          check_rigid_rotation, check_tangent, check_numerical,
          check_integrand_degree)


if __name__ == '__main__':
    for check in CHECKS:
        r = check()
        print(check.__name__, 'verified')
        if isinstance(r, dict):
            for option, res in r.items():
                for case, groups in res.items():
                    print('   ', option, case, ', '.join(
                        '{0}: {1:.2e}/{2:.2e}'.format(g, *v) for g, v in groups.items()))
