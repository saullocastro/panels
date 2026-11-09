r"""Cylindrical shell, total-Lagrangian FSDT with a second-order director (``cylshell_fsdt_tl``)

Derivation of the strains, of their derivatives and of the matrices at an
integration point, with symbolic checks, run ``python cylshell_fsdt_tl.py``.
The kernels are written by ``write_pyx.py``.

Kinematics
----------
First-order shear deformation theory (FSDT) of a circular cylinder of
radius `r`, `x` along the axis, `y = r \theta` the arc length, `z` along
the outward normal, shallow through the thickness, `r + z \approx r`, as in
``cylshell_fsdt_sanders`` and ``cylshell_fsdt_gl``. Degrees of freedom of a
term: `u, v, w, \phi_x, \phi_y`. Local basis `(\mathbf{e}_x,
\mathbf{e}_y, \mathbf{e}_z)` with `\mathbf{e}_{y,y} = -\mathbf{e}_z/r`,
`\mathbf{e}_{z,y} = \mathbf{e}_y/r`.

Displacement field of Schmidt and Reddy (1988), Eq. (16), and Palmerio,
Reddy and Schmidt (1990, Part 1), Eq. (16), `\mathbf{u}(z) =
\mathbf{u}^{(0)} + z \mathbf{u}^{(1)}`, with the transverse normal strain
of Eq. (20) of Palmerio et al., `E^{(0)}_{33} = u^{(1)}_3 + \tfrac{1}{2}
u^{(1)}_\lambda u^{(1)\lambda}`, set to zero (inextensible normal), as for
``plate_fsdt_tl``. The director components `u^{(1)}_\alpha` follow from the
linear transverse shear strain of Eq. (20), `2 E^{(0)}_{\alpha 3} =
\Phi^{(0)}_{\alpha 3} + u^{(1)}_\alpha + \ldots`, with `\Phi^{(0)}_{\alpha 3}
= u_{3,\alpha} + b^\lambda_\alpha u_\lambda`, Eq. (15c): for the cylinder
`b^y_y = -1/r`, `\Phi^{(0)}_{y3} = w_{,y} - v/r`, and the transverse shear
strain `\gamma_{yz} = \phi_y + w_{,y}` of ``cylshell_fsdt_sanders`` gives
`u^{(1)}_y = \phi_y + v/r`, the rotation of the normal (in the
displacement field of these models the tangential displacement at `z` is
`(1 + z/r) v + z \phi_y`). With

.. math::

    \psi_x = \phi_x, \qquad \psi_y = \phi_y + v/r, \qquad
    \psi_z = 1 - \tfrac{1}{2}(\psi_x^2 + \psi_y^2)

the deformed position of a point is `\mathbf{X} + z \mathbf{d}`, with
`\mathbf{X} = \mathbf{X}_0 + u \mathbf{e}_x + v \mathbf{e}_y + w
\mathbf{e}_z` and the second-order director `\mathbf{d} = \psi_x
\mathbf{e}_x + \psi_y \mathbf{e}_y + \psi_z \mathbf{e}_z`:

.. math::

    \mathbf{X}_{,x} = (1 + u_{,x}, \; v_{,x}, \; w_{,x}) \qquad
    \mathbf{X}_{,y} = (u_{,y}, \; 1 + v_{,y} + w/r, \; w_{,y} - v/r)

    \mathbf{d}_{,x} = (\psi_{x,x}, \; \psi_{y,x}, \; \psi_{z,x}) \qquad
    \mathbf{d}_{,y} = (\psi_{x,y}, \; \psi_{y,y} + \psi_z/r, \;
    \psi_{z,y} - \psi_y/r)

Strains: the membrane strains of ``cylshell_fsdt_gl`` (Green-Lagrange),
and

.. math::

    \kappa_{xx} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,x} \qquad
    \kappa_{yy} = \mathbf{X}_{,y} \cdot \mathbf{d}_{,y} - \frac{1}{r} -
    \frac{\varepsilon_{yy}}{r} \qquad
    \kappa_{xy} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,y} + \mathbf{X}_{,y}
    \cdot \mathbf{d}_{,x} - \frac{\gamma_{xy}}{2 r}

    \gamma_{yz} = \mathbf{X}_{,y} \cdot \mathbf{d} \qquad
    \gamma_{xz} = \mathbf{X}_{,x} \cdot \mathbf{d}

Without the terms `-\varepsilon_{yy}/r` and `-\gamma_{xy}/(2r)` they are the
coefficients of `z^0` and `z` of the Green-Lagrange strains of `\mathbf{X}
+ z \mathbf{d}` with the reference `\mathbf{e}_x`, `(1 + z/r)
\mathbf{e}_y` (``check_strain_measures``), whose linear part is Sanders'
plus `(0, \varepsilon_{yy}/r, \gamma_{xy}/(2r))`; the two terms, those of
``cylshell_clpt_tl`` (Koiter's modification, van der Heijden 2008, p. 202),
make the linear part exactly that of ``cylshell_fsdt_sanders``
(``check_linear_part``). Built from dot products and Green-Lagrange strains,
the strains change under a rigid rotation only through the second-order
director, `|\mathbf{d}|^2 = 1 + (\psi_x^2 + \psi_y^2)^2/4`.

Relation to the moderate rotation theory
----------------------------------------
With the orders of magnitude of Palmerio et al. (1990, Part 1), Eqs. (9)
and (18), their strains, Eq. (20), with the curvature tensor of the
cylinder, are those above without the two Koiter terms, to `O(\epsilon^3)`
(``check_palmerio``). Unlike the plate, their curvatures contain quadratic
terms of the order of the linear ones, from `\Phi^{(0)}_{y3}
\Phi^{(1)}_{y3}` with `\Phi^{(1)}_{y3} = u^{(1)}_{3,y} - u^{(1)}_y/r`, which
the linear curvatures of ``cylshell_fsdt_gl`` lack (``check_small_rotations``).

At a point the strains depend on the fourteen generalized derivatives of
``cylshell_fsdt_gl``

    d = (u,x, u,y, v,x, v,y, w,x, w,y, v, w, phix, phiy,
         phix,x, phix,y, phiy,x, phiy,y) = [Phi] c

and the matrices are those of ``plate_fsdt_tl``.

"""
import importlib.util
import os

import sympy
from sympy import (Matrix as M, Symbol, symbols, cos, sin, expand, zeros,
                   eye, diff)

HERE = os.path.dirname(os.path.abspath(__file__))

a, b, r = symbols('a, b, r', positive=True)

#: generalized derivatives at a point, physical coordinates
(u_x, u_y, v_x, v_y, w_x, w_y, v, w, phix, phiy, phix_x, phix_y, phiy_x,
 phiy_y) = symbols('u_x, u_y, v_x, v_y, w_x, w_y, v, w, phix, phiy, '
                   'phix_x, phix_y, phiy_x, phiy_y')
D = (u_x, u_y, v_x, v_y, w_x, w_y, v, w, phix, phiy, phix_x, phix_y, phiy_x,
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
    local basis"""
    return M([1 + u_x, v_x, w_x]), M([u_y, 1 + v_y + w/r, w_y - v/r])


def director(t=1):
    r"""Second-order director ``d`` and its derivatives ``d,x``, ``d,y`` in
    the local basis; ``t = 0`` gives the linear director ``(psix, psiy,
    1)``"""
    px, py = phix, phiy + v/r
    px_x, px_y = phix_x, phix_y
    py_x, py_y = phiy_x + v_x/r, phiy_y + v_y/r
    pz = 1 - t*(px**2 + py**2)/2
    pz_x = -t*(px*px_x + py*py_x)
    pz_y = -t*(px*px_y + py*py_y)
    return (M([px, py, pz]), M([px_x, py_x, pz_x]),
            M([px_y, py_y + pz/r, pz_y - py/r]))


def membrane():
    r"""Green-Lagrange membrane strains of the mid-surface"""
    Xx, Xy = tangents()
    return M([(Xx.dot(Xx) - 1)/2, (Xy.dot(Xy) - 1)/2, Xx.dot(Xy)])


def bending_shear():
    r"""Curvatures and transverse shear strains of the second-order
    director, with the Koiter terms ``-eyy/r`` and ``-gxy/(2r)``"""
    Xx, Xy = tangents()
    d, d_x, d_y = director()
    em = membrane()
    return M([Xx.dot(d_x),
              Xy.dot(d_y) - 1/r - em[1]/r,
              Xx.dot(d_y) + Xy.dot(d_x) - em[2]/(2*r),
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
    r"""``[Phi]`` of a term of side ``'A'`` or ``'B'``: the 14 x 5 matrix such
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
    P[6, 1] = s('f{0}v')*s('g{0}v')
    P[7, 2] = s('f{0}w')*s('g{0}w')
    P[8, 3] = s('f{0}phix')*s('g{0}phix')
    P[9, 4] = s('f{0}phiy')*s('g{0}phiy')
    P[10, 3] = (2/a)*s('f{0}phix_xi')*s('g{0}phix')
    P[11, 3] = (2/b)*s('f{0}phix')*s('g{0}phix_eta')
    P[12, 4] = (2/a)*s('f{0}phiy_xi')*s('g{0}phiy')
    P[13, 4] = (2/b)*s('f{0}phiy')*s('g{0}phiy_eta')
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
    r"""Without the Koiter terms, the strains are the coefficients of ``z^0``
    and ``z^1`` of the Green-Lagrange strains of ``X + z d`` with the
    reference vectors ``e_x``, ``(1 + z/r) e_y``; the dropped terms
    ``|d|^2 - 1``, ``d,a . d`` are of fourth order"""
    z = Symbol('z')
    Xx, Xy = tangents()
    d, d_x, d_y = director()
    Gx, Gy = Xx + z*d_x, Xy + z*d_y
    E = [expand((Gx.dot(Gx) - 1)/2), expand((Gy.dot(Gy) - (1 + z/r)**2)/2),
         expand(Gx.dot(Gy))]
    eps = strains()
    koiter = [0, eps[1]/r, eps[2]/(2*r)]
    for k in range(3):
        assert expand(E[k].coeff(z, 0) - eps[k]) == 0
        assert expand(E[k].coeff(z, 1) - eps[3 + k] - koiter[k]) == 0
    assert expand(Xy.dot(d) - eps[6]) == 0
    assert expand(Xx.dot(d) - eps[7]) == 0
    q = phix**2 + (phiy + v/r)**2
    assert expand(d.dot(d) - 1 - q**2/4) == 0
    assert expand(d_x.dot(d) - q*(phix*phix_x + (phiy + v/r)*(phiy_x + v_x/r))/2) == 0
    # in d,y . d the terms psiz psiy/r of the derivatives of the basis cancel
    assert expand(d_y.dot(d) - q*(phix*phix_y + (phiy + v/r)*(phiy_y + v_y/r))/2) == 0
    return True


def _scaling():
    r"""Orders of magnitude of Palmerio et al. (1990, Part 1), Eqs. (9) and
    (18), as a substitution: the in-plane gradients, `v`, `w` and the
    gradients of the rotations times `\epsilon^2`, `w_{,\alpha}` times
    `\epsilon`, and `\phi_\alpha = -w_{,\alpha} + \gamma_\alpha` with the
    transverse shear strains `\gamma_\alpha` times `\epsilon^2`; `r` and the
    thickness of order one"""
    e, gx, gy = symbols('e, gx, gy')
    sc = {s: e**2*s for s in (u_x, u_y, v_x, v_y, v, w, phix_x, phix_y,
                              phiy_x, phiy_y)}
    sc.update({w_x: e*w_x, w_y: e*w_y, phix: -e*w_x + e**2*gx,
               phiy: -e*w_y + e**2*gy})
    return e, sc


def _low(expr, e, sc):
    ex = expand(expr.xreplace(sc))
    if ex == 0:
        return 99
    return min(mm[0] for mm in sympy.Poly(ex, e).monoms())


def check_palmerio():
    r"""The strains without the Koiter terms agree with Eq. (20) of Palmerio
    et al. (1990, Part 1), cylinder, `E^{(0)}_{33} = 0`, to `O(\epsilon^3)`

    Cartesian surface coordinates `(x, y)`, `b_{yy} = b^y_y = -1/r` the only
    component of the curvature tensor, Eq. (15c): `\Phi^{(n)}_{\alpha 3} =
    u^{(n)}_{3,\alpha} + b^\lambda_\alpha u^{(n)}_\lambda`,
    `\Phi^{(n)}_{\alpha\beta} = u^{(n)}_{\alpha,\beta} - b_{\alpha\beta}
    u^{(n)}_3`, with `u^{(0)} = (u, v, w)`, `u^{(1)} = (\psi_x, \psi_y,
    -(\psi_x^2 + \psi_y^2)/2)`. Scaling of ``_scaling``: the strains are of
    `O(\epsilon^2)` and the differences have no term below `\epsilon^4`; the
    coefficient of `z` of `2 E_{\alpha 3}` is of `O(\epsilon^4)`.

    """
    px, py = phix, phiy + v/r
    p_ = [[phix_x, phix_y], [phiy_x + v_x/r, phiy_y + v_y/r]]   # psi_a,b
    ph = [px, py]
    u13 = -(px**2 + py**2)/2
    u13_ = [-(px*p_[0][0] + py*p_[1][0]), -(px*p_[0][1] + py*p_[1][1])]
    bm = M([[0, 0], [0, -1/r]])               # b_ab = b^a_b
    grad0 = M([[u_x, u_y], [v_x, v_y]])       # u_a,b
    P0a3 = [w_x, w_y - v/r]
    P0 = M([[grad0[i, j] - bm[i, j]*w for j in range(2)] for i in range(2)])
    P1a3 = [u13_[i] + sum(bm[l, i]*ph[l] for l in range(2)) for i in range(2)]
    P1 = M([[p_[i][j] - bm[i, j]*u13 for j in range(2)] for i in range(2)])
    th0 = lambda i, j: (P0[i, j] + P0[j, i])/2
    th1 = lambda i, j: (P1[i, j] + P1[j, i])/2
    E0 = lambda i, j: th0(i, j) + P0a3[i]*P0a3[j]/2
    E1 = lambda i, j: (th1(i, j)
                       - sum(bm[l, i]*P0[l, j] + bm[l, j]*P0[l, i]
                             for l in range(2))/2
                       + (P0a3[i]*P1a3[j] + P0a3[j]*P1a3[i])/2)
    E03x2 = lambda i: (P0a3[i] + ph[i] + sum(ph[l]*P0[l, i] for l in range(2))
                       + u13*(P0a3[i] - ph[i])/2)
    E13x2 = lambda i: (u13_[i] + sum(ph[l]*p_[l][i] for l in range(2))
                       - sum(u13_[l]*P0[l, i] for l in range(2))/2)
    pal = M([E0(0, 0), E0(1, 1), 2*E0(0, 1), E1(0, 0), E1(1, 1), 2*E1(0, 1),
             E03x2(1), E03x2(0)])
    eps = strains()
    koiter = M([0, 0, 0, 0, -eps[1]/r, -eps[2]/(2*r), 0, 0])
    e, sc = _scaling()
    for k in range(NE):
        assert _low(eps[k], e, sc) >= 2 and _low(pal[k], e, sc) >= 2
        assert _low(eps[k] - koiter[k] - pal[k], e, sc) >= 4, k
    for i in range(2):
        assert _low(E13x2(i), e, sc) >= 4
    return True


def check_linear_part():
    r"""``[B(0)] [Phi]`` equal to the linear operator ``B0`` of
    ``cylshell_fsdt_sanders`` in ``theory/shells/fsdt_tsdt``, the linear
    strains and the membrane strains equal to those of
    ``cylshell_fsdt_gl``, and the derivation step by step: the linear parts
    ``X,y . d,y - 1/r = phiy,y + v,y/r + eyy/r``, ``X,x . d,y = phix,y +
    v,x/r``, ``X,y . d,x = phiy,x + v,x/r``, ``X,x . d,x = phix,x``, ``X,y . d
    = phiy + w,y``, ``X,x . d = phix + w,x``, which with the Koiter terms
    ``-eyy/r`` and ``-gxy/(2r)`` give Sanders' FSDT"""
    ft = _load('fsdt_tsdt', '..', 'fsdt_tsdt', 'fsdt_tsdt.py')
    m = build()
    BL = m['B'].subs({s: 0 for s in D})
    rsub = {ft.r: r, ft.a: a, ft.b: b}
    ref = ft.shape_functions('A', 'fsdt', 'cylshell', 'sanders')
    B0 = ref['B0'].xreplace(rsub)
    assert (BL*m['PhiA'] - B0).applyfunc(expand) == zeros(NE, DOF)
    gl = _load('cylshell_fsdt_gl', '..', 'cylshell_fsdt_gl', 'cylshell_fsdt_gl.py')
    epsL_gl = gl.strains(nonlinear=False)
    assert (epsL_gl - m['eps_L']).applyfunc(expand) == zeros(NE, 1)
    eps_gl = gl.strains()
    assert (eps_gl[:3, :] - m['eps'][:3, :]).applyfunc(expand) == zeros(3, 1)
    # the derivation step by step: linear parts of the dot products, the
    # Koiter terms, and Sanders' FSDT
    t = Symbol('t')
    lin = lambda e: expand(expand(e.subs({s: t*s for s in D},
                                         simultaneous=True)).coeff(t, 1))
    Xx, Xy = tangents()
    d, d_x, d_y = director()
    eyy_L, gxy_L = v_y + w/r, u_y + v_x
    assert lin(Xy.dot(d_y) - 1/r) == expand(phiy_y + v_y/r + eyy_L/r)
    assert lin(Xx.dot(d_y)) == expand(phix_y + v_x/r)
    assert lin(Xy.dot(d_x)) == expand(phiy_x + v_x/r)
    assert lin(Xx.dot(d_x)) == phix_x
    assert lin(Xy.dot(d)) == expand(phiy + w_y)
    assert lin(Xx.dot(d)) == expand(phix + w_x)
    sanders = M([u_x, eyy_L, gxy_L, phix_x, phiy_y + v_y/r,
                 phix_y + phiy_x + sympy.Rational(3, 2)*v_x/r
                 - sympy.Rational(1, 2)*u_y/r, phiy + w_y, phix + w_x])
    assert (m['eps_L'] - sanders).applyfunc(expand) == zeros(NE, 1)
    return True


def check_small_rotations():
    r"""Comparison with ``cylshell_fsdt_gl`` at small rotations: with the
    scaling of ``check_palmerio`` the membrane strains are equal, the
    transverse shear strains differ at `O(\epsilon^3)`, the curvatures
    `\kappa_{yy}` and `\kappa_{xy}` at `O(\epsilon^2)`, the order of the
    linear strains, through the quadratic terms divided by `r` (e.g.
    `-\psi_x^2/(2 r)` in `\kappa_{yy}`), and `\kappa_{xx}` at
    `O(\epsilon^4)`"""
    gl = _load('cylshell_fsdt_gl', '..', 'cylshell_fsdt_gl', 'cylshell_fsdt_gl.py')
    diff_ = strains() - gl.strains()
    e, sc = _scaling()
    lows = [_low(diff_[k], e, sc) for k in range(NE)]
    assert lows == [99, 99, 99, 4, 2, 2, 3, 3], lows
    return True


def _rigid_values(rnd, tval):
    r"""Generalized derivatives of a rigid rotation ``tval`` of the
    undeformed cylinder about a random axis through a random point, at a
    random point, with the director components those of the rotated normal,
    and the radius"""
    x, th = symbols('x, theta', real=True)
    nn = [rnd.uniform(-1, 1) for _ in range(3)]
    norm = sum(c**2 for c in nn)**0.5
    n1, n2, n3 = [c/norm for c in nn]
    K = M([[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]])
    R = eye(3) + sin(tval)*K + (1 - cos(tval))*K*K
    rr = rnd.uniform(0.5, 2.)
    ex = M([1, 0, 0])
    ey = M([0, cos(th), sin(th)])
    ez = M([0, sin(th), -cos(th)])
    X0 = M([x, rr*sin(th), -rr*cos(th)])
    p = M([rnd.uniform(-1, 1) for _ in range(3)])
    disp = (R - eye(3))*(X0 - p)
    U, V, W = disp.dot(ex), disp.dot(ey), disp.dot(ez)
    PX = (R*ez).dot(ex)
    PY = (R*ez).dot(ey) - V/rr
    vals = {u_x: diff(U, x), u_y: diff(U, th)/rr, v_x: diff(V, x),
            v_y: diff(V, th)/rr, w_x: diff(W, x), w_y: diff(W, th)/rr,
            v: V, w: W, phix: PX, phiy: PY, phix_x: diff(PX, x),
            phix_y: diff(PX, th)/rr, phiy_x: diff(PY, x),
            phiy_y: diff(PY, th)/rr}
    pt = {x: rnd.uniform(-1, 1), th: rnd.uniform(-3, 3)}
    out = {k: float(ex_.subs(pt)) for k, ex_ in vals.items()}
    out[r] = rr
    return out


def check_rigid_rotation(seed=0):
    r"""Rigid rotations `t` of the undeformed cylinder (a bent state) about
    random axes, with the director components those of the rotated normal:
    the membrane strains vanish, the curvatures and transverse shear
    strains are of `O(t^4)` (fitted order between `t = 0.1` and `0.05` at
    least 3.8); the linear curvatures of ``cylshell_fsdt_gl`` are of
    `O(t^2)`"""
    import random
    gl = _load('cylshell_fsdt_gl', '..', 'cylshell_fsdt_gl', 'cylshell_fsdt_gl.py')
    eps = strains()
    eps_gl = gl.strains()
    for k in range(5):
        res, res_gl = [], []
        for tval in (0.1, 0.05):
            vals = _rigid_values(random.Random(k), tval)
            e = [float(x.xreplace(vals)) for x in eps]
            e_gl = [float(x.xreplace(vals)) for x in eps_gl]
            assert max(abs(x) for x in e[:3]) < 1e-13
            res.append(max(abs(x) for x in e[3:]))
            res_gl.append(max(abs(x) for x in e_gl[3:6]))
        assert sympy.log(res[0]/res[1], 2) > 3.8, res
        assert 1.8 < sympy.log(res_gl[0]/res_gl[1], 2) < 2.2, res_gl
    return True


def check_tangent():
    r"""``d(B^T sigma)/dD = B^T F B + H(sigma, d)`` and the split ``K_T =
    (B^T F B + H(sigma, d) - H0(sigma_L)) + H0(sigma_L)`` of ``fkC_num``
    and ``fkG_num``, the latter homogeneous of degree one in ``d``; with a
    constant membrane stress only, ``H0`` is the Hessian of
    ``cylshell_fsdt_gl``"""
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
    gl = _load('cylshell_fsdt_gl', '..', 'cylshell_fsdt_gl', 'cylshell_fsdt_gl.py')
    Hgl = gl.build()['H']
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
    vals.update({a: 0.7, b: 0.4, r: 0.9})
    for s in m['PhiA'].free_symbols | m['PhiB'].free_symbols:
        vals.setdefault(s, rnd.uniform(-1, 1))
    PhiA = m['PhiA'].xreplace(vals)
    PhiB = m['PhiB'].xreplace(vals)
    Fn = F.xreplace(vals)
    eps = m['eps'].xreplace({r: 0.9})
    Bn = m['B'].xreplace({r: 0.9})
    Hn = m['H'].xreplace({r: 0.9})
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


def degrees():
    r"""Degrees along ``x`` and ``y`` of the integrands of ``K_T`` (and
    ``fint``), each as ``(k, j)`` for ``k p + j``, with the approximation
    functions of degree ``p``: a derivative along the direction lowers the
    degree by one"""
    m = build()
    B = m['B']
    sigma = F*m['eps']
    KT = B.T*F*B + m['H'].xreplace(dict(zip(STRESSES, sigma)))
    out = []
    for tag in ('_x', '_y'):
        deg = {s: -s.name.count(tag) for s in D}
        best = None
        for p in range(len(D)):
            for q in range(len(D)):
                e = expand(KT[p, q])
                if e == 0:
                    continue
                for mono in sympy.Poly(e, *D).monoms():
                    k = sum(mono) + 2
                    j = (sum(n*deg[s] for n, s in zip(mono, D)) + deg[D[p]]
                         + deg[D[q]])
                    if best is None or (k, j) > best:
                        best = (k, j)
        out.append(best)
    return out


def check_integrand_degree():
    r"""Degree ``6p`` along both directions (``cylshell_fsdt_gl``: ``4p``)"""
    assert degrees() == [(6, 0), (6, 0)], degrees()
    return True


CHECKS = (check_strain_measures, check_palmerio, check_linear_part,
          check_small_rotations, check_rigid_rotation, check_tangent,
          check_numerical, check_integrand_degree)


if __name__ == '__main__':
    for check in CHECKS:
        check()
        print(check.__name__, 'verified')
