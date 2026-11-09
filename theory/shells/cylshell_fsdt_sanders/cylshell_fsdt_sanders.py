r"""Cylindrical shell, FSDT with the Sanders-Koiter kinematics (``cylshell_fsdt_sanders``)

Derivation of the strains, of their derivatives and of the matrices, with
symbolic checks, run ``python cylshell_fsdt_sanders.py``. The kernels are written by
``write_pyx.py``.

Kinematics
----------
Circular cylinder of radius `r`, `x` along the axis, `y = r \theta` the
arc length, `0 \le x \le a`, `0 \le y \le b`, `z` along the outward
normal, shallow through the thickness, `r + z \approx r`,
the domain being mapped onto `-1 \le \xi, \eta \le +1`. Degrees of freedom
of a term: `u, v, w, \phi_x, \phi_y`.

Displacement field of the first-order shear deformation theory (FSDT),
`z` measured from the reference surface:

.. math::

    u(z) = u + z \phi_x

    v(z) = v + z \left(\phi_y + \frac{v}{r}\right)

    w(z) = w

The terms in `v/r` are those of Sanders (1959, 1963): the rotation of the
transverse normal about `x` is `\phi_y + v/r`, and the rotation about
the normal, `\omega_3 = \frac{1}{2}(v_{,x} - u_{,y})`, enters the twist,
such that all the strains vanish for any small rigid-body motion. With
this field, the term `-v/r` of the transverse shear strain `\gamma_{yz}`
cancels, and no such term appears in the cubic terms, as in the
displacement field `(1 + z/r) v + z \phi_y + ...` of Reddy and Liu
(1985), Eqs. (5), (7) and (11).

Strains
-------
`\varepsilon = \varepsilon^{(0)} + z \varepsilon^{(1)}`, `\gamma =
\gamma^{(0)}`, with the rotations `\beta_x`, `\beta_y` of the
non-linear terms of Sanders:

.. math::

    \beta_x = -w_{,x} \qquad \beta_y = -w_{,y} + \frac{v}{r}

    \varepsilon^{(0)} = \{ u_{,x} + \frac{1}{2} \beta_x^2,\
                           v_{,y} + \frac{w}{r} + \frac{1}{2} \beta_y^2,\
                           u_{,y} + v_{,x} + \beta_x \beta_y \}

    \varepsilon^{(1)} = \{ \phi_{x,x},\
                           \phi_{y,y} + \frac{1}{r} v_{,y},\
                           \phi_{x,y} + \phi_{y,x}
                           + \frac{3}{2} \frac{1}{r} v_{,x}
                           - \frac{1}{2} \frac{1}{r} u_{,y} \}

    \gamma^{(0)} = \{ \gamma_{yz}, \gamma_{xz} \}^{(0)}
                 = \{ \phi_y + w_{,y},\ \phi_x + w_{,x} \}

As for the CLPT, ``theory/shells/cylshell_clpt_sanders``, the non-linear
terms omit the term `\frac{1}{2}\omega_3^2` of the Sanders-Koiter theory
for small strains and moderately small rotations (Amabili 2008, Sec.
1.3.8, Eqs. (1.133)-(1.134)).

Constitutive law
----------------
``8 x 8``, ``[[A, B, 0], [B, D, 0], [0, 0, Ats]]``, with the rows and
columns in the order of the strains above, where ``Ats`` is the shear
corrected transverse shear stiffness ``[[A44, A45], [A45, A55]]`` of the
`(yz, xz)` components, set by ``Shell.fsdt_shear_correction``.

Internal force vector and tangent stiffness
-------------------------------------------
At a point the strains depend on the fourteen generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, v, w, phix, phiy, phix,x, phix,y, phiy,x, phiy,y) = [Phi] c

and, with ``[B] = d eps/d d`` (8 x 14, linear in `w_{,x}`,
`w_{,y}`, `v`), the stresses ``sigma = [F] eps`` and the Hessian
``[H(N)] = sum_r N_r d^2 eps_r/d d^2 = [G]^T [N] [G]``, ``[G] = d(beta_x,
beta_y)/d d``, which involves only the membrane stresses ``N = [[Nxx, Nxy],
[Nxy, Nyy]]`` and is independent of ``d``::

    fint = int [Phi]^T [B]^T sigma dA
    K_T  = int [Phi]^T ([B]^T [F] [B] + [H(N)]) [Phi] dA

split as ``fkC_num`` (``NLgeom=1``) ``= [B]^T [F] [B] + [H(N - N_L)]`` and
``fkG_num = [H(N_L + Nbar)]``, homogeneous of degree one in ``c``, with
``N_L`` the membrane stresses of the linear strains and ``Nbar`` a constant
membrane state (``check_tangent``, ``check_numerical``).

Linear kernels
--------------
The kernels integrated analytically, and the mass and aerodynamic kernels
integrated numerically, are written from the operators of a term,
:func:`shape_functions`, whose linear strain operator is ``[B_L] [Phi]``
(``check_linear_part``), see :func:`build_linear`. The mass matrices
integrate the kinetic energy of the displacement field above through the
thickness, `z` measured from the reference surface and spanning the
laminate, from `-h/2 + d` to `h/2 + d` with the offset `d`, as in
:func:`composites.laminated_plate`. The kinetic energy being a polynomial
in `z`, the density enters only through its moments `I_n = \int \rho
z^n dz`, `n = 0, 1, 2`, ``intrho``, ``intrhoz`` and ``intrhoz2`` of
:class:`composites.Laminate`.
The aerodynamic matrices use the piston theory, with the load
`q = \beta w_{,x} + \gamma w` along `w` for a flow along `x`, see
:meth:`.Shell.calc_kA`.

"""
import random

import sympy
from sympy import (Matrix as M, Poly, Rational, Symbol, expand, solve,
                   symbols, zeros)

a, b, r = symbols('a, b, r')
#: the thickness coordinate from the reference surface
z = Symbol('z')

#: generalized derivatives at a point, physical coordinates
D = symbols('u_x, u_y, v_x, v_y, w_x, w_y, v, w, phix, phiy, phix_x, phix_y, phiy_x, phiy_y')
(u_x, u_y, v_x, v_y, w_x, w_y, v, w, phix, phiy, phix_x, phix_y, phiy_x, phiy_y) = D
#: names of the generalized derivatives in the kernels
D_NAMES = tuple(s.name for s in D)
#: field and orders of the derivatives along `\xi` and `\eta` of each
#: generalized derivative
D_DEF = {'u_x': ('u', 1, 0),
         'u_y': ('u', 0, 1),
         'v_x': ('v', 1, 0),
         'v_y': ('v', 0, 1),
         'w_x': ('w', 1, 0),
         'w_y': ('w', 0, 1),
         'v': ('v', 0, 0),
         'w': ('w', 0, 0),
         'phix': ('phix', 0, 0),
         'phiy': ('phiy', 0, 0),
         'phix_x': ('phix', 1, 0),
         'phix_y': ('phix', 0, 1),
         'phiy_x': ('phiy', 1, 0),
         'phiy_y': ('phiy', 0, 1)}
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
DOF = 5
NE = 8

A11, A12, A16, A22, A26, A66 = symbols('A11, A12, A16, A22, A26, A66')
B11, B12, B16, B22, B26, B66 = symbols('B11, B12, B16, B22, B26, B66')
D11, D12, D16, D22, D26, D66 = symbols('D11, D12, D16, D22, D26, D66')
A44, A45, A55 = symbols('A44, A45, A55')
Nxx, Nyy, Nxy = symbols('Nxx, Nyy, Nxy')
STRESSES = (Nxx, Nyy, Nxy) + symbols('Mxx, Myy, Mxy, Qy, Qx')
#: the moments of the density through the thickness, `I_n = \int \rho z^n dz`
MOMENTS = symbols('I0, I1, I2')
#: parameters of the piston theory
aeromu, beta, gamma = symbols('aeromu, beta, gamma')

#: constitutive matrix of the FSDT, ``[[A, B, 0], [B, D, 0], [0, 0, Ats]]``
F = M([[A11, A12, A16, B11, B12, B16, 0, 0],
       [A12, A22, A26, B12, B22, B26, 0, 0],
       [A16, A26, A66, B16, B26, B66, 0, 0],
       [B11, B12, B16, D11, D12, D16, 0, 0],
       [B12, B22, B26, D12, D22, D26, 0, 0],
       [B16, B26, B66, D16, D26, D66, 0, 0],
       [0, 0, 0, 0, 0, 0, A44, A45],
       [0, 0, 0, 0, 0, 0, A45, A55]])


def rotations():
    r"""Rotations of the non-linear terms, ``(beta_x, beta_y)``"""
    return M([-w_x, -w_y + v/r])


def strains(nonlinear=True):
    r"""``eps = (exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz)``"""
    bx, by = rotations()
    t = 1 if nonlinear else 0
    eps = [u_x + t*bx**2/2,
           v_y + w/r + t*by**2/2,
           u_y + v_x + t*bx*by,
           phix_x,
           phiy_y + v_y/r,
           phix_y + phiy_x + Rational(3, 2)*v_x/r - Rational(1, 2)*u_y/r,
           phiy + w_y,
           phix + w_x]
    return M(eps).applyfunc(expand)


def basis(side):
    r"""``[Phi]`` of a term of side ``'A'`` or ``'B'``: the 14 x 5 matrix such
    that ``d = [Phi] c``, with the approximation functions in the notation of
    the kernels, e.g. ``fAu_xi`` for `\partial f_i^u/\partial \xi`"""
    P = zeros(len(D), DOF)
    for p, name in enumerate(D_NAMES):
        field, dxi, deta = D_DEF[name]
        f = Symbol('f{0}{1}{2}'.format(side, field, '_' + 'xi'*dxi if dxi else ''))
        g = Symbol('g{0}{1}{2}'.format(side, field, '_' + 'eta'*deta if deta else ''))
        P[p, FIELDS.index(field)] = (2/a)**dxi*(2/b)**deta*f*g
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
    membrane stresses and is independent of ``d``), ``'PhiA'``, ``'PhiB'``
    and ``'G'`` (``d(beta_x, beta_y)/d D``).

    """
    eps = strains()
    eps_L = strains(nonlinear=False)
    B = eps.jacobian(D)
    H = hessian_stress(STRESSES[:3])
    PhiA, PhiB = basis('A'), basis('B')
    G = rotations().jacobian(D)
    return dict(eps=eps, eps_L=eps_L, B=B, F=F, H=H, PhiA=PhiA, PhiB=PhiB,
                G=G)


def shape_functions(side):
    r"""Operators of a term of side ``'A'`` or ``'B'``, as rows of 5 DOFs

    Returns a dict with ``'B0'``, the linear strain operator, ``'G'``, the
    operator of the rotations ``(beta_x, beta_y)``, ``'uz'``, ``'vz'`` and
    ``'wz'``, the displacements through the thickness, and ``'w'``,
    ``'w_x'`` and ``'w_y'``.

    """
    def sf(field, dxi=0, deta=0):
        # derivatives after an underscore, e.g. fAw_xixi, gAu_eta
        fname = 'f{0}{1}{2}'.format(side, field, '_' + 'xi'*dxi if dxi else '')
        gname = 'g{0}{1}{2}'.format(side, field, '_' + 'eta'*deta if deta else '')
        row = zeros(1, DOF)
        row[0, FIELDS.index(field)] = Symbol(fname)*Symbol(gname)
        return row

    u, v, w = sf('u'), sf('v'), sf('w')
    u_x, u_y = (2/a)*sf('u', 1), (2/b)*sf('u', 0, 1)
    v_x, v_y = (2/a)*sf('v', 1), (2/b)*sf('v', 0, 1)
    w_x, w_y = (2/a)*sf('w', 1), (2/b)*sf('w', 0, 1)
    phix, phiy = sf('phix'), sf('phiy')
    phix_x, phix_y = (2/a)*sf('phix', 1), (2/b)*sf('phix', 0, 1)
    phiy_x, phiy_y = (2/a)*sf('phiy', 1), (2/b)*sf('phiy', 0, 1)

    eps0 = M([u_x,
              v_y + w/r,
              u_y + v_x])
    eps1 = M([phix_x,
              phiy_y + v_y/r,
              phix_y + phiy_x + Rational(3, 2)*v_x/r - Rational(1, 2)*u_y/r])
    gam0 = M([phiy + w_y, phix + w_x])
    B0 = M([eps0, eps1, gam0])
    G = M([-w_x, -w_y + v/r])

    uz = u + z*phix
    vz = v + z*(phiy + v/r)
    wz = w
    return dict(B0=B0, G=G, uz=uz, vz=vz, wz=wz, w=w, w_x=w_x, w_y=w_y)


def thickness_moments(e):
    r"""Integral through the thickness of ``rho*e``, ``e`` a polynomial in
    `z`, with `\int \rho z^n dz` replaced by the moment `I_n`"""
    p = Poly(expand(e), z)
    return sum(coef*MOMENTS[n] for (n, ), coef in p.terms())


def build_linear():
    r"""Integrands of the linear kernels, per unit area of the natural domain

    The integral over the physical domain `x_1 \le x \le x_2`, `y_1 \le y
    \le y_2` is ``intx*inty/4`` times the integral over `\xi` and `\eta`,
    where ``intx = x2 - x1`` and ``inty = y2 - y1``. Returns a dict with
    ``'kC0'`` (linear constitutive stiffness), ``'kG'`` (geometric
    stiffness), ``'kM'`` (mass), ``'kAx'``, ``'kAy'`` (aerodynamic
    stiffness), ``'cA'`` (aerodynamic damping) and ``'F'``.

    """
    SA = shape_functions('A')
    SB = shape_functions('B')
    kC0 = SA['B0'].T*F*SB['B0']
    Nmat = M([[Nxx, Nxy],
              [Nxy, Nyy]])
    kG = SA['G'].T*Nmat*SB['G']
    T = SA['uz'].T*SB['uz'] + SA['vz'].T*SB['vz'] + SA['wz'].T*SB['wz']
    kM = T.applyfunc(thickness_moments)
    kAx = -beta*SA['w_x'].T*SB['w'] - gamma*SA['w'].T*SB['w']
    kAy = -beta*SA['w_y'].T*SB['w']
    cA = -aeromu*SA['w'].T*SB['w']
    return dict(kC0=kC0, kG=kG, kM=kM, kAx=kAx, kAy=kAy, cA=cA, F=F)


def check_linear_part():
    r"""``[B_L] [Phi]`` equal to the linear strain operator ``B0`` of
    :func:`shape_functions`, and ``[G] [Phi]`` to its operator ``G``"""
    m = build()
    BL = m['eps_L'].jacobian(D)
    ref = shape_functions('A')
    assert (BL*m['PhiA'] - ref['B0']).applyfunc(expand) == zeros(NE, DOF)
    assert (m['G']*m['PhiA'] - ref['G']).applyfunc(expand) == zeros(2, DOF)
    # the strains are those of [B_L] plus the squares of the rotations
    bx, by = rotations()
    quad = M([bx**2/2, by**2/2, bx*by] + [0]*(NE - 3))
    assert (m['eps'] - m['eps_L'] - quad).applyfunc(expand) == zeros(NE, 1)
    return True


def check_tangent():
    r"""``d(B^T sigma)/dD = B^T F B + H(N)``, ``H = G^T N G``, and the split
    ``K_T = (B^T F B + H(N_NL)) + H(N_L)`` of ``fkC_num`` and ``fkG_num``,
    the latter homogeneous of degree one in ``d``"""
    m = build()
    eps, B, H = m['eps'], m['B'], m['H']
    nd = len(D)
    sigma = F*eps
    exact = (B.T*sigma).applyfunc(expand).jacobian(D).applyfunc(expand)
    sub = dict(zip(STRESSES, sigma))
    KT = (B.T*F*B + H.xreplace(sub)).applyfunc(expand)
    assert (exact - KT).applyfunc(expand) == zeros(nd, nd)
    N2 = M([[Nxx, Nxy], [Nxy, Nyy]])
    assert (m['G'].T*N2*m['G'] - H).applyfunc(expand) == zeros(nd, nd)
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
    rnd = random.Random(seed)
    m = build()
    vals = {s: rnd.uniform(-1, 1) for s in F.free_symbols}
    vals.update({a: 0.7, b: 0.4, r: 0.9})
    for s in m['PhiA'].free_symbols | m['PhiB'].free_symbols:
        vals.setdefault(s, rnd.uniform(-1, 1))
    PhiA = m['PhiA'].xreplace(vals)
    PhiB = m['PhiB'].xreplace(vals)
    Fn = F.xreplace(vals)
    eps = m['eps'].xreplace(vals)
    Bn = m['B'].xreplace(vals)
    Hn = m['H'].xreplace(vals)
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
    step = 1e-6
    for e in range(DOF):
        dc = zeros(DOF, 1)
        dc[e] = step
        fd = (fint_A(cB + dc) - fint_A(cB - dc))/(2*step)
        for dd in range(DOF):
            assert abs(float(fd[dd] - KAB[dd, e])) < 1e-6*(1 + abs(float(KAB[dd, e])))
    return True


CHECKS = (check_linear_part, check_tangent, check_numerical)


if __name__ == '__main__':
    for check in CHECKS:
        check()
        print(check.__name__, 'verified')
