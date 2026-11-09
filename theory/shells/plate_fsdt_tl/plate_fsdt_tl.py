r"""Flat plate, total-Lagrangian FSDT with a second-order director (``plate_fsdt_tl``)

Derivation of the strains, of their derivatives and of the matrices at an
integration point, with symbolic checks, run ``python plate_fsdt_tl.py``.
The kernels are written by ``write_pyx.py``.

Kinematics
----------
First-order shear deformation theory (FSDT) of a flat plate, degrees of
freedom of a term `u, v, w, \phi_x, \phi_y` along the fixed local axes `x,
y, z`, as in ``plate_fsdt_donnell`` and ``plate_fsdt_gl``.

Displacement field of Schmidt and Reddy (1988), Eq. (16), and Palmerio,
Reddy and Schmidt (1990, Part 1), Eq. (16), linear through the thickness,
`\mathbf{u}(z) = \mathbf{u}^{(0)} + z \mathbf{u}^{(1)}`, with
`u^{(0)}_\alpha = (u, v)`, `u^{(0)}_3 = w`, `u^{(1)}_\alpha = (\phi_x,
\phi_y)`. Their transverse normal strain, Eq. (20) of Palmerio et al.,

.. math::

    E^{(0)}_{33} = u^{(1)}_3 + \tfrac{1}{2} u^{(1)}_\lambda u^{(1)\lambda}

is set to zero (inextensible normal, the second Kirchhoff-Love assumption,
which their theory does not use, Schmidt and Reddy 1988, p. 613), so that
`u^{(1)}_3 = -(\phi_x^2 + \phi_y^2)/2` and the deformed position of a
point of the plate is `\mathbf{X} + z \mathbf{d}` with

.. math::

    \mathbf{X} = (x + u, \; y + v, \; w), \qquad
    \mathbf{d} = (\phi_x, \; \phi_y, \; 1 - \tfrac{1}{2}(\phi_x^2 +
    \phi_y^2))

the second-order director. The strains are the coefficients of `z^0` and
`z^1` of the Green-Lagrange strains of this field, `E_{\alpha\beta}(z) =
(\mathbf{G}_\alpha \cdot \mathbf{G}_\beta - \delta_{\alpha\beta})/2`,
`\mathbf{G}_\alpha = \mathbf{X}_{,\alpha} + z \mathbf{d}_{,\alpha}`, and
`2 E_{\alpha 3}(0) = \mathbf{X}_{,\alpha} \cdot \mathbf{d}`:

.. math::

    \varepsilon_{xx} = u_{,x} + \tfrac{1}{2}(u_{,x}^2 + v_{,x}^2 + w_{,x}^2)
    \qquad
    \varepsilon_{yy} = v_{,y} + \tfrac{1}{2}(u_{,y}^2 + v_{,y}^2 + w_{,y}^2)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} +
    w_{,x} w_{,y}

    \kappa_{xx} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,x} \qquad
    \kappa_{yy} = \mathbf{X}_{,y} \cdot \mathbf{d}_{,y} \qquad
    \kappa_{xy} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,y} + \mathbf{X}_{,y}
    \cdot \mathbf{d}_{,x}

    \gamma_{yz} = \mathbf{X}_{,y} \cdot \mathbf{d} \qquad
    \gamma_{xz} = \mathbf{X}_{,x} \cdot \mathbf{d}

The membrane strains are those of ``plate_fsdt_gl``; the bending and
transverse shear strains are cubic polynomials of the generalized
derivatives. Dropped, as in every FSDT model of panels: the coefficient of
`z^2` of `E_{\alpha\beta}`, and the coefficient of `z` of `2 E_{\alpha 3}`,
`\mathbf{d}_{,\alpha} \cdot \mathbf{d} = (\phi_x^2 + \phi_y^2)
(\phi_x \phi_{x,\alpha} + \phi_y \phi_{y,\alpha})/2`, of fourth order
(``check_strain_measures``).

Relation to the moderate rotation theory
----------------------------------------
With the orders of magnitude of Palmerio et al. (1990, Part 1), Eqs. (9)
and (18) (strains `O(\epsilon^2)`, rotations of the normal `w_{,\alpha}`,
`\phi_\alpha` of `O(\epsilon)`, in-plane gradients and `h` times the
gradients of the rotations of `O(\epsilon^2)`), their strains, Eq. (20),
are those above to `O(\epsilon^3)` (``check_palmerio``): the difference is
of fourth order, as are the cubic terms of the curvatures, which the
moderate rotation theory does not need for a plate once the normal is
inextensible. They are kept because they make the strains exact for the
field `\mathbf{X} + z \mathbf{d}`, the plate counterpart of
``plate_clpt_tl``: a finite rigid rotation `t` of a bent state changes the
curvatures by `O(t^4)` instead of `O(t^2)` for ``plate_fsdt_gl``
(``check_rigid_rotation``), the remainder coming from the second-order
director, `|\mathbf{d}|^2 = 1 + (\phi_x^2 + \phi_y^2)^2/4`.

At a point the strains depend on the twelve generalized derivatives of
``plate_fsdt_gl``

    d = (u,x, u,y, v,x, v,y, w,x, w,y, phix, phiy,
         phix,x, phix,y, phiy,x, phiy,y) = [Phi] c

and the matrices are those of ``plate_clpt_tl`` with the 8 x 8
constitutive matrix of the FSDT: ``[B] = d eps/d d`` (8 x 12,
quadratic), ``fint = int [Phi]^T [B]^T sigma dA``, ``K_T = int [Phi]^T
([B]^T [F] [B] + [H(sigma, d)]) [Phi] dA``, ``[H(sigma, d)] = sum_r
sigma_r d^2 eps_r/d d^2``, linear in ``d``.

"""
import importlib.util
import os

import sympy
from sympy import Matrix as M, Symbol, symbols, expand, zeros, eye

HERE = os.path.dirname(os.path.abspath(__file__))

a, b = symbols('a, b', positive=True)

#: generalized derivatives at a point, physical coordinates
(u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy, phix_x, phix_y, phiy_x,
 phiy_y) = symbols('u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy, '
                   'phix_x, phix_y, phiy_x, phiy_y')
D = (u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy, phix_x, phix_y, phiy_x,
     phiy_y)
#: names of the generalized derivatives in the kernels
D_NAMES = tuple(s.name for s in D)
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
DOF = 5
NE = 8

A11, A12, A16, A22, A26, A66 = symbols('A11, A12, A16, A22, A26, A66')
B11, B12, B16, B22, B26, B66 = symbols('B11, B12, B16, B22, B26, B66')
D11, D12, D16, D22, D26, D66 = symbols('D11, D12, D16, D22, D26, D66')
A44, A45, A55 = symbols('A44, A45, A55')
Nxx, Nyy, Nxy = symbols('Nxx, Nyy, Nxy')
STRESSES = (Nxx, Nyy, Nxy) + symbols('Mxx, Myy, Mxy, Qy, Qx')

#: constitutive matrix of the FSDT, ``[[A, B, 0], [B, D, 0], [0, 0, Ats]]``
F = M([[A11, A12, A16, B11, B12, B16, 0, 0],
       [A12, A22, A26, B12, B22, B26, 0, 0],
       [A16, A26, A66, B16, B26, B66, 0, 0],
       [B11, B12, B16, D11, D12, D16, 0, 0],
       [B12, B22, B26, D12, D22, D26, 0, 0],
       [B16, B26, B66, D16, D26, D66, 0, 0],
       [0, 0, 0, 0, 0, 0, A44, A45],
       [0, 0, 0, 0, 0, 0, A45, A55]])


def tangents():
    r"""Tangent vectors ``X,x``, ``X,y`` of the deformed mid-surface in the
    fixed local basis"""
    return M([1 + u_x, v_x, w_x]), M([u_y, 1 + v_y, w_y])


def director(t=1):
    r"""Second-order director ``d`` and its derivatives ``d,x``, ``d,y``;
    ``t = 0`` gives the linear director ``(phix, phiy, 1)``"""
    dz = 1 - t*(phix**2 + phiy**2)/2
    dz_x = -t*(phix*phix_x + phiy*phiy_x)
    dz_y = -t*(phix*phix_y + phiy*phiy_y)
    return (M([phix, phiy, dz]), M([phix_x, phiy_x, dz_x]),
            M([phix_y, phiy_y, dz_y]))


def membrane():
    r"""Green-Lagrange membrane strains of the mid-surface"""
    Xx, Xy = tangents()
    return M([(Xx.dot(Xx) - 1)/2, (Xy.dot(Xy) - 1)/2, Xx.dot(Xy)])


def bending_shear():
    r"""Curvatures and transverse shear strains of the second-order
    director"""
    Xx, Xy = tangents()
    d, d_x, d_y = director()
    return M([Xx.dot(d_x),
              Xy.dot(d_y),
              Xx.dot(d_y) + Xy.dot(d_x),
              Xy.dot(d),
              Xx.dot(d)])


def strains(nonlinear=True):
    r"""``eps = (exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz)``; with
    ``nonlinear=False`` the linear part"""
    eps = M(list(membrane()) + list(bending_shear())).applyfunc(expand)
    if nonlinear:
        return eps
    t = Symbol('t')
    return eps.subs({s: t*s for s in D}, simultaneous=True).applyfunc(
        lambda e: expand(e).coeff(t, 1)).applyfunc(expand)


def basis(side):
    r"""``[Phi]`` of a term of side ``'A'`` or ``'B'``: the 12 x 5 matrix such
    that ``d = [Phi] c``, with the approximation functions in the notation of
    the kernels, e.g. ``fAu_xi`` for `\partial f_i^u/\partial \xi`"""
    s = lambda name: Symbol(name.format(side))
    P = zeros(len(D), DOF)
    P[0, 0] = (2/a)*s('f{0}u_xi')*s('g{0}u')
    P[1, 0] = (2/b)*s('f{0}u')*s('g{0}u_eta')
    P[2, 1] = (2/a)*s('f{0}v_xi')*s('g{0}v')
    P[3, 1] = (2/b)*s('f{0}v')*s('g{0}v_eta')
    P[4, 2] = (2/a)*s('f{0}w_xi')*s('g{0}w')
    P[5, 2] = (2/b)*s('f{0}w')*s('g{0}w_eta')
    P[6, 3] = s('f{0}phix')*s('g{0}phix')
    P[7, 4] = s('f{0}phiy')*s('g{0}phiy')
    P[8, 3] = (2/a)*s('f{0}phix_xi')*s('g{0}phix')
    P[9, 3] = (2/b)*s('f{0}phix')*s('g{0}phix_eta')
    P[10, 4] = (2/a)*s('f{0}phiy_xi')*s('g{0}phiy')
    P[11, 4] = (2/b)*s('f{0}phiy')*s('g{0}phiy_eta')
    return P


def hessian_stress(S, eps=None):
    r"""``[H(S, d)] = sum_r S_r d^2 eps_r/d D^2`` for the eight stresses
    ``S``, linear in ``d``"""
    eps = strains() if eps is None else eps
    H = zeros(len(D), len(D))
    for k in range(NE):
        H += S[k]*sympy.hessian(eps[k], D)
    return H.applyfunc(expand)


def build():
    r"""Matrices of the derivation at an integration point

    Returns a dict with ``'eps'``, ``'eps_L'``, ``'B'``, ``'F'``, ``'H'``
    (the Hessian of the stresses ``STRESSES``, linear in ``d``), ``'H0'``
    (``'H'`` at ``d = 0``), ``'PhiA'`` and ``'PhiB'``.

    """
    eps = strains()
    eps_L = strains(nonlinear=False)
    B = eps.jacobian(D)
    H = hessian_stress(STRESSES, eps)
    H0 = H.subs({s: 0 for s in D})
    PhiA, PhiB = basis('A'), basis('B')
    return dict(eps=eps, eps_L=eps_L, B=B, F=F, H=H, H0=H0, PhiA=PhiA,
                PhiB=PhiB)


def _load(name, *path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, *path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_strain_measures():
    r"""The strains are the coefficients of ``z^0`` and ``z^1`` of the
    Green-Lagrange strains of ``X + z d``; the dropped terms are of fourth
    order: ``|d|^2 - 1``, ``d,a . d``"""
    z = Symbol('z')
    Xx, Xy = tangents()
    d, d_x, d_y = director()
    Gx, Gy = Xx + z*d_x, Xy + z*d_y
    E = [expand((Gx.dot(Gx) - 1)/2), expand((Gy.dot(Gy) - 1)/2),
         expand(Gx.dot(Gy))]
    eps = strains()
    for k in range(3):
        assert expand(E[k].coeff(z, 0) - eps[k]) == 0
        assert expand(E[k].coeff(z, 1) - eps[3 + k]) == 0
    assert expand(Xy.dot(d) - eps[6]) == 0
    assert expand(Xx.dot(d) - eps[7]) == 0
    q = phix**2 + phiy**2
    assert expand(d.dot(d) - 1 - q**2/4) == 0
    assert expand(d_x.dot(d) - q*(phix*phix_x + phiy*phiy_x)/2) == 0
    assert expand(d_y.dot(d) - q*(phix*phix_y + phiy*phiy_y)/2) == 0
    return True


def _scaling():
    r"""Orders of magnitude of Palmerio et al. (1990, Part 1), Eqs. (9) and
    (18), as a substitution: the in-plane gradients and the gradients of the
    rotations times `\epsilon^2`, `w_{,\alpha}` times `\epsilon`, and
    `\phi_\alpha = -w_{,\alpha} + \gamma_\alpha` with the transverse shear
    strains `\gamma_\alpha` times `\epsilon^2`"""
    e, gx, gy = symbols('e, gx, gy')
    sc = {s: e**2*s for s in (u_x, u_y, v_x, v_y, phix_x, phix_y, phiy_x,
                              phiy_y)}
    sc.update({w_x: e*w_x, w_y: e*w_y, phix: -e*w_x + e**2*gx,
               phiy: -e*w_y + e**2*gy})
    return e, sc


def check_palmerio():
    r"""The strains agree with Eq. (20) of Palmerio et al. (1990, Part 1),
    flat plate, `E^{(0)}_{33} = 0`, to `O(\epsilon^3)`

    Plate, `b_{\alpha\beta} = 0`, Cartesian coordinates: `\Phi^{(0)}_{\alpha
    3} = w_{,\alpha}`, `\Phi^{(n)}_{\alpha\beta} = u^{(n)}_{\alpha,\beta}`,
    `\Phi^{(1)}_{\alpha 3} = u^{(1)}_{3,\alpha}`, Eq. (15c); `u^{(1)}_3 =
    -(\phi_x^2 + \phi_y^2)/2` from `E^{(0)}_{33} = 0`. Scaling of Eq. (18):
    the in-plane gradients and the gradients of the rotations times `\epsilon^2`,
    `w_{,\alpha}` and `\phi_\alpha` times `\epsilon`; the strains are then of
    `O(\epsilon^2)` and the differences have no term below `\epsilon^4`. The
    coefficient of `z` of `2 E_{\alpha 3}`, which the FSDT drops, is of
    `O(\epsilon^4)` too.

    """
    u0 = M([[u_x, u_y], [v_x, v_y]])          # u_a,b
    u1 = M([[phix_x, phix_y], [phiy_x, phiy_y]])
    w0 = [w_x, w_y]
    ph = [phix, phiy]
    u13 = -(phix**2 + phiy**2)/2
    u13_ = [-(phix*phix_x + phiy*phiy_x), -(phix*phix_y + phiy*phiy_y)]
    E0 = lambda i, j: (u0[i, j] + u0[j, i])/2 + w0[i]*w0[j]/2
    E1 = lambda i, j: ((u1[i, j] + u1[j, i])/2
                       + (w0[i]*u13_[j] + w0[j]*u13_[i])/2)
    E03x2 = lambda i: (w0[i] + ph[i] + sum(ph[l]*u0[l, i] for l in range(2))
                       + u13*(w0[i] - ph[i])/2)
    E13x2 = lambda i: (u13_[i] + sum(ph[l]*u1[l, i] for l in range(2))
                       - sum(u13_[l]*u0[l, i] for l in range(2))/2)
    pal = M([E0(0, 0), E0(1, 1), 2*E0(0, 1), E1(0, 0), E1(1, 1), 2*E1(0, 1),
             E03x2(1), E03x2(0)])
    eps = strains()
    e, sc = _scaling()

    def low(expr):
        p = sympy.Poly(expand(expr.xreplace(sc)), e)
        return min((m[0] for m in p.monoms()), default=99)
    for k in range(NE):
        assert low(eps[k]) >= 2 and low(pal[k]) >= 2
        assert low(eps[k] - pal[k]) >= 4, (k, low(eps[k] - pal[k]))
    for i in range(2):
        assert low(E13x2(i)) >= 4
    return True


def check_linear_part():
    r"""``[B(0)] [Phi]`` equal to the linear operator ``B0`` of
    ``plate_fsdt_donnell`` (``theory/shells/plate_fsdt_donnell``), and the linear
    strains equal to those of ``plate_fsdt_gl``"""
    lm = _load('plate_fsdt_donnell', '..', 'plate_fsdt_donnell', 'plate_fsdt_donnell.py')
    m = build()
    BL = m['B'].subs({s: 0 for s in D})
    sub = {lm.a: a, lm.b: b}
    B0 = lm.shape_functions('A')['B0'].xreplace(sub)
    assert (BL*m['PhiA'] - B0).applyfunc(expand) == zeros(NE, DOF)
    gl = _load('plate_fsdt_gl', '..', 'plate_fsdt_gl', 'plate_fsdt_gl.py')
    glsub = {getattr(gl, s.name): s for s in D}
    epsL_gl = gl.strains(nonlinear=False).xreplace(glsub)
    assert (epsL_gl - m['eps_L']).applyfunc(expand) == zeros(NE, 1)
    eps_gl = gl.strains().xreplace(glsub)
    assert (eps_gl[:3, :] - m['eps'][:3, :]).applyfunc(expand) == zeros(3, 1)
    return True


def check_small_rotations():
    r"""Agreement with ``plate_fsdt_gl`` at small rotations: with the scaling
    of ``check_palmerio`` the strains of the two models differ at
    `O(\epsilon^3)`, relative order `\epsilon`, through the transverse
    shear (`\mathbf{X}_{,\alpha} \cdot \mathbf{d}` against `\phi_\alpha +
    w_{,\alpha}`); the curvatures differ at `O(\epsilon^4)`"""
    gl = _load('plate_fsdt_gl', '..', 'plate_fsdt_gl', 'plate_fsdt_gl.py')
    glsub = {getattr(gl, s.name): s for s in D}
    diff = strains() - gl.strains().xreplace(glsub)
    e, sc = _scaling()
    lows = []
    for k in range(NE):
        ex = expand(diff[k].xreplace(sc))
        lows.append(99 if ex == 0 else
                    min(mm[0] for mm in sympy.Poly(ex, e).monoms()))
    assert lows == [99, 99, 99, 4, 4, 4, 3, 3], lows
    return True


def _rotation(n, t):
    K = M([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return eye(3) + sympy.sin(t)*K + (1 - sympy.cos(t))*K*K


def _state(Xx, Xy, Xxx, Xxy, Xyy, nrm, nrm_x, nrm_y):
    r"""Generalized derivatives of a state given by the tangent vectors,
    their derivatives and the unit normal field with its derivatives, the
    director components being those of the normal"""
    return {u_x: Xx[0] - 1, v_x: Xx[1], w_x: Xx[2],
            u_y: Xy[0], v_y: Xy[1] - 1, w_y: Xy[2],
            phix: nrm[0], phiy: nrm[1], phix_x: nrm_x[0], phiy_x: nrm_x[1],
            phix_y: nrm_y[0], phiy_y: nrm_y[1]}


def check_rigid_rotation(seed=0):
    r"""Rigid rotations, of the undeformed plate and of a bent state

    A rigid rotation `t` about a random axis of the undeformed plate, with
    the director components those of the rotated normal: the membrane
    strains and the curvatures vanish, the transverse shear strains are of
    `O(t^5)` (``|d|^2 - 1``). The bent state of a cylindrical bending of
    curvature `k_0` about a random in-plane axis, rotated by `t` about
    another random in-plane axis: the changes of all eight strains are of `O(t^4)` for this
    model (fitted order of the change between `t` and `t/2` at least 3.8),
    of `O(t^2)` for the curvatures of ``plate_fsdt_gl``, at least 20 times
    larger at `t = 0.1` with `k_0 = 0.3`.

    """
    import random
    rnd = random.Random(seed)
    gl = _load('plate_fsdt_gl', '..', 'plate_fsdt_gl', 'plate_fsdt_gl.py')
    glsub = {s: getattr(gl, s.name) for s in D}
    eps = strains()
    eps_gl = gl.strains()

    def ev(e, vals):
        return [float(x.xreplace(vals)) for x in e]

    def axis(inplane=False):
        nn = [rnd.uniform(-1, 1) for _ in range(3)]
        if inplane:
            nn[2] = 0.
        nrm = sum(c**2 for c in nn)**0.5
        return [c/nrm for c in nn]
    # undeformed plate
    for _ in range(5):
        n = axis()
        shear = []
        for t in (0.2, 0.1):
            R = _rotation(n, t)
            ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
            z3 = M([0, 0, 0])
            vals = _state(ex, ey, z3, z3, z3, ez, z3, z3)
            e = ev(eps, vals)
            assert max(abs(x) for x in e[:6]) < 1e-14
            shear.append(max(abs(x) for x in e[6:]))
        assert 4.8 < sympy.log(shear[0]/shear[1], 2) < 5.2
    # bent state: X = (sin(k0 s)/k0, ., (1 - cos(k0 s))/k0) along the unit
    # in-plane direction e1 = (c, s, 0), at s = 0
    k0 = 0.3
    for _ in range(5):
        al = rnd.uniform(0, 2*3.14159)
        e1 = M([sympy.cos(al), sympy.sin(al), 0])
        e2 = M([-sympy.sin(al), sympy.cos(al), 0])
        ez = M([0, 0, 1])
        # derivatives along x and y of X and of the normal at the point
        c1, s1 = float(sympy.cos(al)), float(sympy.sin(al))
        Xx = M([1, 0, 0])
        Xy = M([0, 1, 0])
        Xss = k0*ez                      # X,ss
        nrm_s = -k0*e1                   # normal (-sin, ., cos) along e1
        Xxx, Xxy, Xyy = c1*c1*Xss, c1*s1*Xss, s1*s1*Xss
        nrm_x, nrm_y = c1*nrm_s, s1*nrm_s
        base = ev(eps, _state(Xx, Xy, Xxx, Xxy, Xyy, ez, nrm_x, nrm_y))
        base_gl = ev(eps_gl, {glsub[k]: v for k, v in _state(
            Xx, Xy, Xxx, Xxy, Xyy, ez, nrm_x, nrm_y).items()})
        n = axis(inplane=True)
        ch, ch_gl = [], []
        for t in (0.2, 0.1):
            R = _rotation(n, t)
            st = _state(R*Xx, R*Xy, R*Xxx, R*Xxy, R*Xyy, R*ez, R*nrm_x,
                        R*nrm_y)
            e = ev(eps, st)
            # for the linear curvatures of plate_fsdt_gl the rotations are
            # those of the rotated normal as well
            e_gl = ev(eps_gl, {glsub[k]: v for k, v in st.items()})
            ch.append(max(abs(x - y) for x, y in zip(e, base)))
            ch_gl.append(max(abs(x - y) for x, y in zip(e_gl[3:6], base_gl[3:6])))
        assert sympy.log(ch[0]/ch[1], 2) > 3.8, ch
        assert 1.8 < sympy.log(ch_gl[0]/ch_gl[1], 2) < 2.2, ch_gl
        assert ch[1] < 0.05*ch_gl[1]
    return True


def check_tangent():
    r"""``d(B^T sigma)/dD = B^T F B + H(sigma, d)`` and the split ``K_T =
    (B^T F B + H(sigma, d) - H0(sigma_L)) + H0(sigma_L)`` of ``fkC_num``
    and ``fkG_num``, the latter homogeneous of degree one in ``d``"""
    m = build()
    eps, B, H, H0 = m['eps'], m['B'], m['H'], m['H0']
    nd = len(D)
    sigma = F*eps
    exact = (B.T*sigma).applyfunc(expand).jacobian(D).applyfunc(expand)
    KT = (B.T*F*B + H.xreplace(dict(zip(STRESSES, sigma)))).applyfunc(expand)
    assert (exact - KT).applyfunc(expand) == zeros(nd, nd)
    sigma_L = F*m['eps_L']
    KG = H0.xreplace(dict(zip(STRESSES, sigma_L)))
    KC = (B.T*F*B + H.xreplace(dict(zip(STRESSES, sigma))) - KG)
    assert (KC + KG - KT).applyfunc(expand) == zeros(nd, nd)
    tt = Symbol('tt')
    KGs = KG.subs({dd: tt*dd for dd in D}, simultaneous=True)
    assert (KGs - tt*KG).applyfunc(expand) == zeros(nd, nd)
    # with a constant membrane stress only, H0 is the Hessian of
    # plate_fsdt_gl: the analytical fkG0 of plate_fsdt_gl applies
    gl = _load('plate_fsdt_gl', '..', 'plate_fsdt_gl', 'plate_fsdt_gl.py')
    glsub = {getattr(gl, s.name): s for s in D}
    Hgl = gl.build()['H'].xreplace(glsub).xreplace(
        {getattr(gl, s.name): s for s in STRESSES[:3]})
    memb = {s: 0 for s in STRESSES[3:]}
    assert (H0.xreplace(memb) - Hgl).applyfunc(expand) == zeros(nd, nd)
    return True


def check_numerical(seed=0):
    r"""Central differences of ``fint`` of one pair of terms against
    ``[Phi_A]^T K_T [Phi_B]`` at random values"""
    import random
    rnd = random.Random(seed)
    m = build()
    vals = {s: rnd.uniform(-1, 1) for s in F.free_symbols}
    vals.update({a: 0.7, b: 0.4})
    for s in m['PhiA'].free_symbols | m['PhiB'].free_symbols:
        vals.setdefault(s, rnd.uniform(-1, 1))
    PhiA = m['PhiA'].xreplace(vals)
    PhiB = m['PhiB'].xreplace(vals)
    Fn = F.xreplace(vals)
    eps, Bn, Hn = m['eps'], m['B'], m['H']
    cA = M([rnd.uniform(-0.3, 0.3) for _ in range(DOF)])
    cB = M([rnd.uniform(-0.3, 0.3) for _ in range(DOF)])

    def fint_A(cB_):
        sub = dict(zip(D, PhiA*cA + PhiB*cB_))
        sigma = Fn*eps.xreplace(sub)
        return PhiA.T*Bn.xreplace(sub).T*sigma

    sub = dict(zip(D, PhiA*cA + PhiB*cB))
    sigma = Fn*eps.xreplace(sub)
    K = (Bn.T*Fn*Bn + Hn.xreplace(dict(zip(STRESSES, sigma)))).xreplace(sub)
    KAB = PhiA.T*K*PhiB
    h = 1e-6
    for e in range(DOF):
        dc = zeros(DOF, 1)
        dc[e] = h
        fd = (fint_A(cB + dc) - fint_A(cB - dc))/(2*h)
        for dd in range(DOF):
            assert abs(float(fd[dd] - KAB[dd, e])) < 1e-6*(1 + abs(float(KAB[dd, e])))
    return True


def x_degree():
    r"""Degree along ``x`` of the integrands of ``K_T`` (and ``fint``), as
    ``(k, j)`` for ``k p + j``, with the approximation functions of degree
    ``p``: a derivative ``u,x`` has the degree ``p - 1``, ``u,y`` and
    ``phix`` the degree ``p``"""
    deg = {s: -s.name.count('_x') for s in D}
    m = build()
    B = m['B']
    sigma = F*m['eps']
    KT = B.T*F*B + m['H'].xreplace(dict(zip(STRESSES, sigma)))
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
    r"""Degree ``6p`` along each direction (``plate_fsdt_gl``: ``4p``):
    ``phix`` and ``phiy`` enter without derivative"""
    assert x_degree() == (6, 0), x_degree()
    return True


CHECKS = (check_strain_measures, check_palmerio, check_linear_part,
          check_small_rotations, check_rigid_rotation, check_tangent,
          check_numerical, check_integrand_degree)


if __name__ == '__main__':
    for check in CHECKS:
        check()
        print(check.__name__, 'verified')
