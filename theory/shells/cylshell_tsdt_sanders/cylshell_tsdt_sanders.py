r"""Cylindrical shell, TSDT with the Sanders-Koiter kinematics (``cylshell_tsdt_sanders``)

Derivation of the strains, of their derivatives and of the matrices, with
symbolic checks, run ``python cylshell_tsdt_sanders.py``. The kernels are written by
``write_pyx.py``.

Kinematics
----------
Circular cylinder of radius `r`, `x` along the axis, `y = r \theta` the
arc length, `0 \le x \le a`, `0 \le y \le b`, `z` along the outward
normal, shallow through the thickness, `r + z \approx r`,
the domain being mapped onto `-1 \le \xi, \eta \le +1`. Degrees of freedom
of a term: `u, v, w, \phi_x, \phi_y`.
    v(z) = v + z \left(\phi_y + \frac{v}{r}\right)

Displacement field of the third-order shear deformation theory (TSDT) of
Reddy (1984), with `c_1 = 4/(3 h^2)`, `z` measured from the reference
surface and the laminate spanning `d - h/2 \le z \le d + h/2`, with the
offset `d`:

.. math::

    u(z) = u + z \phi_x - c_1 \left[(z - d)^3 + d^3\right] (\phi_x + w_{,x})

    v(z) = v + z \left(\phi_y + \frac{v}{r}\right)
           - c_1 \left[(z - d)^3 + d^3\right] (\phi_y + w_{,y})

    w(z) = w

The cubic term is written about the mid-surface of the laminate, such
that the transverse shear strains, `\gamma_{xz} = (\phi_x + w_{,x}) (1 -
3 c_1 (z - d)^2)` and likewise `\gamma_{yz}`, vanish at both faces `z = d
\pm h/2` for any offset; the constant `c_1 d^3` keeps `u` and `v` the
displacements of the reference surface, `z = 0`, and `\phi_x`, `\phi_y`
are the rotations of the normal at the mid-surface, `u_{,z}(d) =
\phi_x` (``check_offset_cubic``). With `d = 0` this is Reddy's field.
The conditions on the cubic leave one free parameter, which only
selects the coordinate where the rotation variable is measured: at the
reference surface it would be singular when `z = 0` is a face, `d = \pm
h/2`. The connections tie the rotations at the reference surface,
`\theta_x = \phi_x - (4 d^2/h^2)(\phi_x + w_{,x})`, see
:mod:`panels.multidomain.connections.kCsdt`.
For the cylinder the field is the shell field of Reddy and Liu (1985),
Eqs. (5)-(9), with the shallow-shell assumption.

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
`\varepsilon = \varepsilon^{(0)} + z \varepsilon^{(1)} + \left[(z - d)^3
+ d^3\right] \varepsilon^{(3)}`, `\gamma = \gamma^{(0)} + (z - d)^2
\gamma^{(2)}`, with the rotations `\beta_x`, `\beta_y` of the
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

    \varepsilon^{(3)} = -c_1 \{ \phi_{x,x} + w_{,xx},\
                                \phi_{y,y} + w_{,yy},\
                                \phi_{x,y} + \phi_{y,x} + 2 w_{,xy} \}

    \gamma^{(0)} = \{ \gamma_{yz}, \gamma_{xz} \}^{(0)}
                 = \{ \phi_y + w_{,y},\ \phi_x + w_{,x} \}

    \gamma^{(2)} = -3 c_1 \gamma^{(0)}

As for the CLPT, ``theory/shells/cylshell_clpt_sanders``, the non-linear
terms omit the term `\frac{1}{2}\omega_3^2` of the Sanders-Koiter theory
for small strains and moderately small rotations (Amabili 2008, Sec.
1.3.8, Eqs. (1.133)-(1.134)).

Constitutive law
----------------
``13 x 13``, ``T^T F_m T``, with the rows and columns in the order of the
strains above, ``F_m = [[A, B, E, 0, 0], [B, D, F, 0, 0], [E, F, H, 0,
0], [0, 0, 0, Abar_ts, Dts], [0, 0, 0, Dts, Fts]]`` the matrix of the
laminate about its mid-surface, i.e. with the integrals of the
stiffness multiplied by `1, \zeta, \zeta^2, \zeta^3, \zeta^4, \zeta^6`,
`\zeta = z - d`, and of the transverse shear stiffness multiplied by
`1`, `\zeta^2` and `\zeta^4`, with no shear correction, and ``T`` the
identity but for the blocks ``T[0:3, 3:6] = d I`` and ``T[0:3, 6:9] =
d^3 I``, because `\varepsilon = (\varepsilon^{(0)} + d \varepsilon^{(1)}
+ d^3 \varepsilon^{(3)}) + \zeta \varepsilon^{(1)} + \zeta^3
\varepsilon^{(3)}`, see :meth:`.Shell._get_lam_ABD`. With `d = 0` it is
the usual matrix of the laminate.

Internal force vector and tangent stiffness
-------------------------------------------
At a point the strains depend on the seventeen generalized derivatives

    d = (u,x, u,y, v,x, v,y, w,x, w,y, v, w, phix, phiy, phix,x, phix,y, phiy,x, phiy,y, w,xx, w,yy, w,xy) = [Phi] c

and, with ``[B] = d eps/d d`` (13 x 17, linear in `w_{,x}`,
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
in `z`, the density enters only through its moments: the field contains
`z^0`, `z`, `z^2` and `z^3`, and the moments are those about the
mid-surface, `J_n = \int \rho \zeta^n d\zeta`, `\zeta = z - d`, `n = 0,
1, 2, 3, 4, 6`, those of the laminate with a zero offset, with the offset
`d` in the coefficients, since in `\zeta` the field has the powers
`\zeta^0`, `\zeta` and `\zeta^3` only.
The aerodynamic matrices use the piston theory, with the load
`q = \beta w_{,x} + \gamma w` along `w` for a flow along `x`, see
:meth:`.Shell.calc_kA`.

"""
import random

import sympy
from sympy import (Matrix as M, Poly, Rational, Symbol, expand, solve,
                   symbols, zeros)

a, b, r = symbols('a, b, r')
#: the constant of the TSDT, 4/(3 h^2), the thickness and the offset
c1, h, d = symbols('c1, h, d')
#: the thickness coordinate from the reference surface
z = Symbol('z')
#: the thickness coordinate from the mid-surface, `\zeta = z - d`
zeta = Symbol('zeta')

#: generalized derivatives at a point, physical coordinates
D = symbols('u_x, u_y, v_x, v_y, w_x, w_y, v, w, phix, phiy, phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy')
(u_x, u_y, v_x, v_y, w_x, w_y, v, w, phix, phiy, phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy) = D
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
         'phiy_y': ('phiy', 0, 1),
         'w_xx': ('w', 2, 0),
         'w_yy': ('w', 0, 2),
         'w_xy': ('w', 1, 1)}
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
Nxx, Nyy, Nxy = symbols('Nxx, Nyy, Nxy')
STRESSES = (Nxx, Nyy, Nxy) + symbols('Mxx, Myy, Mxy, Pxx, Pyy, Pxy, Qy, Qx, Ry, Rx')
#: the moments of the density about the mid-surface, `J_n = \int \rho
#: \zeta^n d\zeta`, `n = 0, 1, 2, 3, 4, 6`
MOMENTS = symbols('J0, J1, J2, J3, J4, J5, J6')
#: parameters of the piston theory
aeromu, beta, gamma = symbols('aeromu, beta, gamma')

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


def rotations():
    r"""Rotations of the non-linear terms, ``(beta_x, beta_y)``"""
    return M([-w_x, -w_y + v/r])


def strains(nonlinear=True):
    r"""``eps = (exx, eyy, gxy, kxx, kyy, kxy, kxx3, kyy3, kxy3, gyz, gxz,
    gyz2, gxz2)``"""
    bx, by = rotations()
    t = 1 if nonlinear else 0
    eps = [u_x + t*bx**2/2,
           v_y + w/r + t*by**2/2,
           u_y + v_x + t*bx*by,
           phix_x,
           phiy_y + v_y/r,
           phix_y + phiy_x + Rational(3, 2)*v_x/r - Rational(1, 2)*u_y/r,
           -c1*(phix_x + w_xx),
           -c1*(phiy_y + w_yy),
           -c1*(phix_y + phiy_x + 2*w_xy),
           phiy + w_y,
           phix + w_x,
           -3*c1*(phiy + w_y),
           -3*c1*(phix + w_x)]
    return M(eps).applyfunc(expand)


def basis(side):
    r"""``[Phi]`` of a term of side ``'A'`` or ``'B'``: the 17 x 5 matrix such
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
    w_xx = (2/a)**2*sf('w', 2)
    w_yy = (2/b)**2*sf('w', 0, 2)
    w_xy = (2/a)*(2/b)*sf('w', 1, 1)
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
    eps3 = -c1*M([phix_x + w_xx, phiy_y + w_yy, phix_y + phiy_x + 2*w_xy])
    gam2 = -3*c1*gam0
    B0 = M([eps0, eps1, eps3, gam0, gam2])
    G = M([-w_x, -w_y + v/r])

    # the cubic term about the mid-surface, z = d, see check_offset_cubic
    cubic = (z - d)**3 + d**3
    uz = u + z*phix - c1*cubic*(phix + w_x)
    vz = v + z*(phiy + v/r) - c1*cubic*(phiy + w_y)
    wz = w
    return dict(B0=B0, G=G, uz=uz, vz=vz, wz=wz, w=w, w_x=w_x, w_y=w_y)


def thickness_moments(e):
    r"""Integral through the thickness of ``rho*e``, ``e`` a polynomial in
    `z`, with `z = \zeta + d` and `\int \rho \zeta^n d\zeta` replaced by the
    moment `J_n` about the mid-surface"""
    p = Poly(expand(e.subs(z, zeta + d)), zeta)
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
    vals.update({a: 0.7, b: 0.4, r: 0.9, c1: 0.6})
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


def check_offset_cubic():
    r"""Cubic `p(z)` of the field `u(z) = u + z \phi_x + p(z) (\phi_x +
    w_{,x})` with the offset `d`

    Solves `p(0) = 0`, `p'(d \pm h/2) = -1` (zero transverse shear at both
    faces) and `p'(d) = 0` (`\phi_x` the rotation at the mid-surface), and
    checks that `p(z) = -c_1 [(z - d)^3 + d^3]`, `c_1 = 4/(3 h^2)`.

    """
    p1, p2, p3 = symbols('p1, p2, p3')
    p = p1*z + p2*z**2 + p3*z**3
    dp = p.diff(z)
    sol = solve([dp.subs(z, d - h/2) + 1, dp.subs(z, d + h/2) + 1,
                 dp.subs(z, d)], [p1, p2, p3], dict=True)[0]
    p = p.subs(sol)
    assert expand(p + Rational(4, 3)/h**2*((z - d)**3 + d**3)) == 0
    return True


CHECKS = (check_linear_part, check_tangent, check_numerical,
          check_offset_cubic)


if __name__ == '__main__':
    for check in CHECKS:
        check()
        print(check.__name__, 'verified')
