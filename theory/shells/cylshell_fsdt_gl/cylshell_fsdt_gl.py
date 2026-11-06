r"""Cylindrical shell, FSDT with Green-Lagrange membrane strains (``cylshell_fsdt_gl``)

Derivation of the strains, of their derivatives and of the matrices at an
integration point, with symbolic checks, run ``python cylshell_fsdt_gl.py``.
The kernels are written by ``write_pyx.py``.

Kinematics
----------
First-order shear deformation theory (FSDT) of a circular cylinder of
radius `r`, `x` along the axis, `y = r \theta` the arc length, `z` along
the outward normal, shallow through the thickness, `r + z \approx r`, as in
``cylshell_fsdt_sanders``. Degrees of freedom of a term: `u, v, w, \phi_x,
\phi_y`.

Membrane strains: the complete Green-Lagrange strains of the mid-surface,
`E = (F^T F - I)/2`, with the tangent vectors of the deformed mid-surface in
the local basis, the columns of

.. math::

    F = \begin{bmatrix} 1 + u_{,x} & u_{,y} \\ v_{,x} & 1 + v_{,y} + w/r \\
        w_{,x} & w_{,y} - v/r \end{bmatrix}

i.e. the membrane strains of ``cylshell_clpt_gl``, those of the
Flügge-Lur'e-Byrne non-linear theory as given by Amabili (2008), Sec. 1.3.6,
Eqs. (1.118)-(1.120), without imperfection and with `y = R \theta`:

.. math::

    \varepsilon_{xx} = u_{,x} + \tfrac{1}{2}(u_{,x}^2 + v_{,x}^2 + w_{,x}^2)

    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \tfrac{1}{2}\left(u_{,y}^2 +
    (v_{,y} + w/r)^2 + (w_{,y} - v/r)^2\right)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x}(v_{,y} + w/r) +
    w_{,x}(w_{,y} - v/r)

Bending and transverse shear strains: the linear ones of
``cylshell_fsdt_sanders`` (Sanders-Koiter kinematics of the FSDT, see
``theory/shells/fsdt_tsdt``):

.. math::

    \kappa_{xx} = \phi_{x,x} \qquad \kappa_{yy} = \phi_{y,y} + v_{,y}/r
    \qquad \kappa_{xy} = \phi_{x,y} + \phi_{y,x} + \tfrac{3}{2} v_{,x}/r -
    \tfrac{1}{2} u_{,y}/r

    \gamma_{yz} = \phi_y + w_{,y} \qquad \gamma_{xz} = \phi_x + w_{,x}

Compared with ``cylshell_fsdt_sanders``, whose non-linear terms are those
of the two rotations `\beta_x = -w_{,x}`, `\beta_y = -w_{,y} + v/r`, the
membrane strains add the quadratic terms of the in-plane gradients
(``check_sanders_limit``), among which the rotation about the normal. The
linear part is exactly that of ``cylshell_fsdt_sanders``
(``check_linear_part``, against ``theory/shells/fsdt_tsdt``).

At a point the strains depend on the fourteen generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, v, w, phix, phiy,
         phix,x, phix,y, phiy,x, phiy,y) = [Phi] c

and the matrices are those of ``cylshell_clpt_gl`` with the 8 x 8
constitutive matrix of the FSDT: ``[B] = d eps/d d`` (8 x 14, linear in
the first eight derivatives), ``fint = int [Phi]^T [B]^T sigma dA``, ``K_T
= int [Phi]^T ([B]^T [F] [B] + [H(N)]) [Phi] dA``, ``[H(N)] = [Gm]^T
diag(N, N, N) [Gm]`` with the six local gradients ``beta = (u,x, u,y, v,x,
v,y + w/r, w,x, w,y - v/r) = [Gm] d``.

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
#: the derivatives of the non-linear terms of the strains
GRAD = D[:8]
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


def beta_of_D():
    r"""The six local gradients, the entries of ``F - I``"""
    return M([u_x, u_y, v_x, v_y + w/r, w_x, w_y - v/r])


def membrane(nonlinear=True):
    r"""Green-Lagrange membrane strains of the mid-surface"""
    ux, uy, vx, vy, wx, wy = beta_of_D()
    t = 1 if nonlinear else 0
    return M([ux + t*(ux**2 + vx**2 + wx**2)/2,
              vy + t*(uy**2 + vy**2 + wy**2)/2,
              uy + vx + t*(ux*uy + vx*vy + wx*wy)])


def bending_shear():
    r"""Linear bending and transverse shear strains of the Sanders FSDT"""
    return M([phix_x,
              phiy_y + v_y/r,
              phix_y + phiy_x + sympy.Rational(3, 2)*v_x/r
              - sympy.Rational(1, 2)*u_y/r,
              phiy + w_y,
              phix + w_x])


def strains(nonlinear=True):
    r"""``eps = (exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz)``"""
    return M(list(membrane(nonlinear)) + list(bending_shear())).applyfunc(expand)


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


def hessian_stress(N):
    r"""``[H(N)] = sum_r N_r d^2 eps_r/d D^2`` for the membrane stresses
    ``N = (Nxx, Nyy, Nxy)``, the only strains with non-linear terms"""
    eps = strains()
    H = zeros(len(D), len(D))
    for k in range(3):
        H += N[k]*sympy.hessian(eps[k], D)
    return H.applyfunc(expand)


def build():
    r"""Matrices of the derivation at an integration point

    Returns a dict with ``'eps'``, ``'eps_L'``, ``'B'``, ``'F'``, ``'H'``
    (the Hessian of the stresses ``STRESSES``, which involves only the
    membrane stresses and is independent of ``d``), ``'H0'`` (equal to
    ``'H'``), ``'PhiA'``, ``'PhiB'`` and ``'Gm'`` (``d beta/d D``).

    """
    eps = strains()
    eps_L = strains(nonlinear=False)
    B = eps.jacobian(D)
    H = hessian_stress(STRESSES[:3])
    PhiA, PhiB = basis('A'), basis('B')
    Gm = beta_of_D().jacobian(D)
    return dict(eps=eps, eps_L=eps_L, B=B, F=F, H=H, H0=H, PhiA=PhiA,
                PhiB=PhiB, Gm=Gm)


def _fsdt_tsdt():
    r"""The derivation of the six models of ``theory/shells/fsdt_tsdt``"""
    spec = importlib.util.spec_from_file_location(
        'fsdt_tsdt', os.path.join(HERE, '..', 'fsdt_tsdt', 'fsdt_tsdt.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_membrane():
    r"""Membrane strains equal to ``E = (F^T F - I)/2`` of the tangent
    vectors and to Amabili (2008), Eqs. (1.118)-(1.120), as transcribed in
    the module docstring"""
    Fd = M([[1 + u_x, u_y], [v_x, 1 + v_y + w/r], [w_x, w_y - v/r]])
    E = (Fd.T*Fd - eye(2))/2
    eps = strains()
    assert expand(eps[0] - E[0, 0]) == 0
    assert expand(eps[1] - E[1, 1]) == 0
    assert expand(eps[2] - 2*E[0, 1]) == 0
    amabili = [u_x + (u_x**2 + v_x**2 + w_x**2)/2,
               v_y + w/r + (u_y**2 + (v_y + w/r)**2 + (w_y - v/r)**2)/2,
               u_y + v_x + u_x*u_y + v_x*(v_y + w/r) + w_x*(w_y - v/r)]
    for k in range(3):
        assert expand(eps[k] - amabili[k]) == 0
    return True


def check_linear_part():
    r"""``[B_L] [Phi]`` equal to the linear operator ``B0`` of
    ``cylshell_fsdt_sanders`` in ``theory/shells/fsdt_tsdt``, and the
    non-linear operator of that model at ``bx = -w,x``, ``by = -w,y + v/r``
    equal to ``[B]`` without the quadratic terms of the in-plane
    gradients"""
    ft = _fsdt_tsdt()
    m = build()
    BL = m['B'].subs({g: 0 for g in GRAD})
    rsub = {ft.r: r, ft.a: a, ft.b: b}
    ref = ft.shape_functions('A', 'fsdt', 'cylshell', 'sanders')
    B0 = ref['B0'].xreplace(rsub)
    assert (BL*m['PhiA'] - B0).applyfunc(expand) == zeros(NE, DOF)
    # Sanders limit: drop the squares of the in-plane gradients
    sanders = {ft.bx: -w_x, ft.by: -(w_y - v/r)}
    Bs = ref['B'].xreplace(rsub).xreplace(sanders)
    # the strains with the in-plane gradients frozen in the quadratic terms
    t = Symbol('t')
    mem = M([u_x + t*(u_x**2 + v_x**2)/2 + w_x**2/2,
             v_y + w/r + t*(u_y**2 + (v_y + w/r)**2)/2 + (w_y - v/r)**2/2,
             u_y + v_x + t*(u_x*u_y + v_x*(v_y + w/r)) + w_x*(w_y - v/r)])
    eps_t = M(list(mem) + list(bending_shear()))
    assert (eps_t.subs(t, 1) - strains()).applyfunc(expand) == zeros(NE, 1)
    Bsk = eps_t.subs(t, 0).jacobian(D)
    assert (Bsk*m['PhiA'] - Bs).applyfunc(expand) == zeros(NE, DOF)
    return True


def check_rigid_rotation(samples=20, seed=0):
    r"""Zero membrane strains for finite rigid rotations about any axis

    The displacement ``(R - I) X0`` of a rotation ``R`` (Rodrigues, random
    angle up to 3 rad about a random axis through a random point) is
    projected on the local basis and differentiated exactly; the
    Green-Lagrange membrane strains vanish to round-off, the von Kármán
    membrane strains of ``cylshell_fsdt_sanders`` are of the order of the
    angle squared.

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
    vals = {u_x: diff(U, x), u_y: diff(U, th)/r, v_x: diff(V, x),
            v_y: diff(V, th)/r, w_x: diff(W, x), w_y: diff(W, th)/r,
            v: V, w: W}
    eps = strains()[:3, :]
    eps_s = M([u_x + w_x**2/2, v_y + w/r + (w_y - v/r)**2/2,
               u_y + v_x + w_x*(w_y - v/r)])
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


def check_tangent():
    r"""``d(B^T sigma)/dD = B^T F B + H(N)``, ``H = Gm^T diag(N, N, N) Gm``,
    and the split ``K_T = (B^T F B + H(N_NL)) + H(N_L)`` of ``fkC_num`` and
    ``fkG_num``, the latter homogeneous of degree one in ``d``"""
    m = build()
    eps, B, H = m['eps'], m['B'], m['H']
    nd = len(D)
    sigma = F*eps
    exact = (B.T*sigma).applyfunc(expand).jacobian(D).applyfunc(expand)
    sub = dict(zip(STRESSES, sigma))
    KT = (B.T*F*B + H.xreplace(sub)).applyfunc(expand)
    assert (exact - KT).applyfunc(expand) == zeros(nd, nd)
    N2 = M([[Nxx, Nxy], [Nxy, Nyy]])
    Gm = m['Gm']
    assert (Gm.T*sympy.diag(N2, N2, N2)*Gm - H).applyfunc(expand) == zeros(nd, nd)
    sigma_L = F*m['eps_L']
    KC = B.T*F*B + H.xreplace(dict(zip(STRESSES, sigma - sigma_L)))
    KG = H.xreplace(dict(zip(STRESSES, sigma_L)))
    assert (KC + KG - KT).applyfunc(expand) == zeros(nd, nd)
    tt = Symbol('tt')
    KGs = KG.subs({dd: tt*dd for dd in D}, simultaneous=True)
    assert (KGs - tt*KG).applyfunc(expand) == zeros(nd, nd)
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


CHECKS = (check_membrane, check_linear_part, check_rigid_rotation,
          check_tangent, check_numerical)


if __name__ == '__main__':
    for check in CHECKS:
        check()
        print(check.__name__, 'verified')
