r"""Flat plate, TSDT with Green-Lagrange membrane strains (``plate_tsdt_gl``)

Derivation of the strains, of their derivatives and of the matrices at an
integration point, with symbolic checks, run ``python plate_tsdt_gl.py``.
The kernels are written by ``write_pyx.py``.

Kinematics
----------
Reddy's third-order shear deformation theory (TSDT) of a flat plate, degrees of
freedom of a term `u, v, w, \phi_x, \phi_y` along the fixed local axes `x,
y, z`, as in ``plate_tsdt_donnell``.

Membrane strains: the complete Green-Lagrange strains of the mid-surface,
`E = (F^T F - I)/2`, `F = [e_x + u_{,x} \;\; e_y + u_{,y}]` (3 x 2), those
of ``plate_clpt_gl``:

.. math::

    \varepsilon_{xx} = u_{,x} + \tfrac{1}{2}(u_{,x}^2 + v_{,x}^2 + w_{,x}^2)

    \varepsilon_{yy} = v_{,y} + \tfrac{1}{2}(u_{,y}^2 + v_{,y}^2 + w_{,y}^2)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} +
    w_{,x} w_{,y}

Bending and transverse shear strains: the linear ones of
``plate_tsdt_donnell``, `c_1 = 4/(3 h^2)`:

.. math::

    \kappa_{xx} = \phi_{x,x} \qquad \kappa_{yy} = \phi_{y,y} \qquad
    \kappa_{xy} = \phi_{x,y} + \phi_{y,x}

    \gamma_{yz} = \phi_y + w_{,y} \qquad \gamma_{xz} = \phi_x + w_{,x}

    \kappa^{(3)}_{xx} = -c_1 (\phi_{x,x} + w_{,xx}) \qquad
    \kappa^{(3)}_{yy} = -c_1 (\phi_{y,y} + w_{,yy}) \qquad
    \kappa^{(3)}_{xy} = -c_1 (\phi_{x,y} + \phi_{y,x} + 2 w_{,xy})

    \gamma^{(2)}_{yz} = -3 c_1 \gamma_{yz} \qquad
    \gamma^{(2)}_{xz} = -3 c_1 \gamma_{xz}

Compared with ``plate_tsdt_donnell`` (von Kármán), the membrane strains add
the quadratic terms of the in-plane gradients; the linear part is exactly
that of ``plate_tsdt_donnell`` (``check_linear_part``, against
``theory/shells/fsdt_tsdt``). For a finite rigid rotation of any angle
about any axis the membrane strains vanish and, with the constant rotations
`\phi_x = -w_{,x}`, `\phi_y = -w_{,y}`, so do the bending, transverse shear
and third-order strains (``check_rigid_rotation``).

At a point the strains depend on the fifteen generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, phix, phiy,
         phix,x, phix,y, phiy,x, phiy,y, w,xx, w,yy, w,xy) = [Phi] c

and the matrices are those of ``plate_clpt_gl`` with the 8 x 8
constitutive matrix of the TSDT: ``[B] = d eps/d d`` (13 x 15, linear in
the first six derivatives), ``fint = int [Phi]^T [B]^T sigma dA``, ``K_T =
int [Phi]^T ([B]^T [F] [B] + [H(N)]) [Phi] dA``, ``[H(N)] = [Gm]^T diag(N,
N, N) [Gm]`` with the six gradients ``beta = (u,x, u,y, v,x, v,y, w,x,
w,y) = [Gm] d``.

"""
import importlib.util
import os

import sympy
from sympy import Matrix as M, Symbol, symbols, expand, zeros, eye

HERE = os.path.dirname(os.path.abspath(__file__))

a, b = symbols('a, b', positive=True)

#: generalized derivatives at a point, physical coordinates
(u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy, phix_x, phix_y, phiy_x,
 phiy_y, w_xx, w_yy, w_xy) = symbols('u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy, '
                   'phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy')
D = (u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy, phix_x, phix_y, phiy_x,
     phiy_y, w_xx, w_yy, w_xy)
#: the derivatives of the non-linear terms of the strains
GRAD = D[:6]
#: names of the generalized derivatives in the kernels
D_NAMES = tuple(s.name for s in D)
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
STRESSES = (Nxx, Nyy, Nxy) + symbols('Mxx, Myy, Mxy, Pxx, Pyy, Pxy, Qy, Qx, Ry, Rx')

#: constitutive matrix of the TSDT, ``[[A, B, E, 0, 0], [B, D, F, 0, 0], [E, F,
#: H, 0, 0], [0, 0, 0, Ats, Dts], [0, 0, 0, Dts, Fts]]``
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


def beta_of_D():
    r"""The six gradients, the entries of ``F - I``"""
    return M([u_x, u_y, v_x, v_y, w_x, w_y])


def membrane(nonlinear=True):
    r"""Green-Lagrange membrane strains of the mid-surface"""
    ux, uy, vx, vy, wx, wy = beta_of_D()
    t = 1 if nonlinear else 0
    return M([ux + t*(ux**2 + vx**2 + wx**2)/2,
              vy + t*(uy**2 + vy**2 + wy**2)/2,
              uy + vx + t*(ux*uy + vx*vy + wx*wy)])


def bending_shear():
    r"""Linear bending, third-order and transverse shear strains of the TSDT"""
    return M([phix_x,
              phiy_y,
              phix_y + phiy_x,
              -c1*(phix_x + w_xx),
              -c1*(phiy_y + w_yy),
              -c1*(phix_y + phiy_x + 2*w_xy),
              phiy + w_y,
              phix + w_x,
              -3*c1*(phiy + w_y),
              -3*c1*(phix + w_x)])


def strains(nonlinear=True):
    r"""``eps = (exx, eyy, gxy, kxx, kyy, kxy, kxx3, kyy3, kxy3, gyz, gxz,
    gyz2, gxz2)``"""
    return M(list(membrane(nonlinear)) + list(bending_shear())).applyfunc(expand)


def basis(side):
    r"""``[Phi]`` of a term of side ``'A'`` or ``'B'``: the 15 x 5 matrix such
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
    P[12, 2] = (2/a)**2*s('f{0}w_xixi')*s('g{0}w')
    P[13, 2] = (2/b)**2*s('f{0}w')*s('g{0}w_etaeta')
    P[14, 2] = (2/a)*(2/b)*s('f{0}w_xi')*s('g{0}w_eta')
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
    vectors ``e_x + u,x`` and ``e_y + u,y``"""
    Fd = M([[1 + u_x, u_y], [v_x, 1 + v_y], [w_x, w_y]])
    E = (Fd.T*Fd - eye(2))/2
    eps = strains()
    assert expand(eps[0] - E[0, 0]) == 0
    assert expand(eps[1] - E[1, 1]) == 0
    assert expand(eps[2] - 2*E[0, 1]) == 0
    return True


def check_linear_part():
    r"""``[B_L] [Phi]`` equal to the linear operator ``B0`` of
    ``plate_tsdt_donnell`` in ``theory/shells/fsdt_tsdt``, and the
    operator of that model at ``bx = -w,x``, ``by = -w,y`` equal to ``[B]``
    without the quadratic terms of the in-plane gradients (von Kármán
    limit)"""
    ft = _fsdt_tsdt()
    m = build()
    BL = m['B'].subs({g: 0 for g in GRAD})
    sub = {ft.a: a, ft.b: b, ft.c1: c1}
    ref = ft.shape_functions('A', 'tsdt', 'plate', 'donnell')
    B0 = ref['B0'].xreplace(sub)
    assert (BL*m['PhiA'] - B0).applyfunc(expand) == zeros(NE, DOF)
    Bvk = ref['B'].xreplace(sub).xreplace({ft.bx: -w_x, ft.by: -w_y})
    t = Symbol('t')
    mem = M([u_x + t*(u_x**2 + v_x**2)/2 + w_x**2/2,
             v_y + t*(u_y**2 + v_y**2)/2 + w_y**2/2,
             u_y + v_x + t*(u_x*u_y + v_x*v_y) + w_x*w_y])
    eps_t = M(list(mem) + list(bending_shear()))
    assert (eps_t.subs(t, 1) - strains()).applyfunc(expand) == zeros(NE, 1)
    assert (eps_t.subs(t, 0).jacobian(D)*m['PhiA'] - Bvk).applyfunc(expand) == \
        zeros(NE, DOF)
    return True


def check_rigid_rotation(samples=20, seed=0):
    r"""All the strains vanish for finite rigid rotations about any axis

    The displacement ``(R - I) (X0 - p)`` of a rotation ``R`` (Rodrigues,
    random angle up to 3 rad about a random axis) has the constant gradients
    of ``R - I``; with the constant rotations ``phix = -w,x``, ``phiy =
    -w,y`` all thirteen strains vanish to round-off, whereas the von Kármán
    membrane strains of ``plate_tsdt_donnell`` are of the order of the angle
    squared.

    """
    import random
    rnd = random.Random(seed)
    worst_gl, worst_vk = 0., 0.
    eps = strains()
    for _ in range(samples):
        nn = [rnd.uniform(-1, 1) for _ in range(3)]
        norm = sum(c**2 for c in nn)**0.5
        n1, n2, n3 = [c/norm for c in nn]
        t = rnd.uniform(-3, 3)
        K = M([[0, -n3, n2], [n3, 0, -n1], [-n2, n1, 0]])
        R = eye(3) + sympy.sin(t)*K + (1 - sympy.cos(t))*K*K
        G = R - eye(3)
        dv = {u_x: G[0, 0], u_y: G[0, 1], v_x: G[1, 0], v_y: G[1, 1],
              w_x: G[2, 0], w_y: G[2, 1], phix: -G[2, 0], phiy: -G[2, 1],
              phix_x: 0, phix_y: 0, phiy_x: 0, phiy_y: 0,
              w_xx: 0, w_yy: 0, w_xy: 0}
        worst_gl = max(worst_gl, max(abs(float(e.xreplace(dv))) for e in eps))
        vk = [G[0, 0] + G[2, 0]**2/2, G[1, 1] + G[2, 1]**2/2,
              G[0, 1] + G[1, 0] + G[2, 0]*G[2, 1]]
        worst_vk = max(worst_vk, max(abs(float(e)) for e in vk))
    assert worst_gl < 1e-12, worst_gl
    assert worst_vk > 1e-2, worst_vk
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
    vals.update({a: 0.7, b: 0.4})
    for s in m['PhiA'].free_symbols | m['PhiB'].free_symbols:
        vals.setdefault(s, rnd.uniform(-1, 1))
    PhiA = m['PhiA'].xreplace(vals)
    PhiB = m['PhiB'].xreplace(vals)
    Fn = F.xreplace(vals)
    vals[c1] = 0.6
    eps, Bn, Hn = [m[k].xreplace({c1: vals[c1]}) for k in ('eps', 'B', 'H')]
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
