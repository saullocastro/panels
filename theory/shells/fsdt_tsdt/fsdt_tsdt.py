r"""Plates and cylindrical shells using the FSDT and Reddy's TSDT

The domain `0 \le x \le a`, `0 \le y \le b` is mapped onto
`-1 \le \xi, \eta \le +1`. For the cylinders, `y = r \theta` is the arc length
along the circumference, such that `\frac{1}{r} (\cdot)_{,\theta} =
(\cdot)_{,y}`. The field variables are `u, v, w, \phi_x, \phi_y`, in this
order, which are the 5 degrees of freedom of each term of the approximation.

The kinematics of the cylinders follow Sanders (1959, 1963) for the FSDT and
Reddy and Liu (1985) for the TSDT, specialised to a circular cylinder,
adopting the shallow shell assumption `R(z) = r + z \approx r`, as in
``theory/shells/cylshell_clpt_sanders`` for the CLPT.

Displacement field, with `c_1 = 4/(3 h^2)` for the third-order shear
deformation theory (TSDT) of Reddy (1984), and `c_1 = 0` for the first-order
shear deformation theory (FSDT), `z` being measured from the reference
surface and the laminate spanning `d - h/2 \le z \le d + h/2`, with the
offset `d`:

.. math::

    u(z) = u + z \phi_x - c_1 \left[(z - d)^3 + d^3\right] (\phi_x + w_{,x})

    v(z) = v + z \left(\phi_y + \boxed{\frac{v}{r}}\right)
           - c_1 \left[(z - d)^3 + d^3\right] (\phi_y + w_{,y})

    w(z) = w

The cubic term of the TSDT is written about the mid-surface of the
laminate, such that the transverse shear strains, `\gamma_{xz} = (\phi_x +
w_{,x}) (1 - 3 c_1 (z - d)^2)` and likewise `\gamma_{yz}`, vanish at both
faces `z = d \pm h/2` for any offset; the constant `c_1 d^3` keeps `u` and
`v` the displacements of the reference surface, `z = 0`, as for the CLPT
and the FSDT, and `\phi_x`, `\phi_y` are the rotations of the normal at the
mid-surface, `u_{,z}(d) = \phi_x`. With `d = 0` this is Reddy's field. The
conditions on the cubic leave one free parameter, which only selects the
coordinate where the rotation variable is measured: at the reference
surface it would be singular when `z = 0` is a face, `d = \pm h/2`, see
:func:`tsdt_offset_cubic` and ``doc/source/tsdt_offset.rst``. The
connections tie the rotations at the reference surface, `\theta_x = \phi_x -
(4 d^2/h^2)(\phi_x + w_{,x})`, see
:mod:`panels.multidomain.connections.kCsdt`.

The strains are, in Voigt notation, `\varepsilon = \varepsilon^{(0)} + z
\varepsilon^{(1)} + \left[(z - d)^3 + d^3\right] \varepsilon^{(3)}` and
`\gamma = \gamma^{(0)} + (z - d)^2 \gamma^{(2)}`, with the rotations
`\beta_x = -w_{,x}` and `\beta_y = -w_{,y} + \boxed{v/r}` of the von Karman
non-linear terms, which for the CLPT are the rotations `\phi_x` and `\phi_y`
of ``theory/shells/cylshell_clpt_sanders``:

.. math::

    \varepsilon^{(0)} = \{ u_{,x} + \frac{1}{2} \beta_x^2,\
                           v_{,y} + \frac{w}{r} + \frac{1}{2} \beta_y^2,\
                           u_{,y} + v_{,x} + \beta_x \beta_y \}

    \varepsilon^{(1)} = \{ \phi_{x,x},\
                           \phi_{y,y} + \boxed{\frac{1}{r} v_{,y}},\
                           \phi_{x,y} + \phi_{y,x}
                           + \boxed{\frac{3}{2} \frac{1}{r} v_{,x}
                           - \frac{1}{2} \frac{1}{r} u_{,y}} \}

    \varepsilon^{(3)} = -c_1 \{ \phi_{x,x} + w_{,xx},\
                                \phi_{y,y} + w_{,yy},\
                                \phi_{x,y} + \phi_{y,x} + 2 w_{,xy} \}

    \gamma^{(0)} = \{ \gamma_{yz}, \gamma_{xz} \}^{(0)}
                 = \{ \phi_y + w_{,y},\ \phi_x + w_{,x} \}

    \gamma^{(2)} = -3 c_1 \gamma^{(0)}

The boxed terms are the corrections of Sanders (1959, 1963), which make all
the strains vanish for any small rigid-body motion: the rotation of the
transverse normal about `x` is `\Phi_y = \phi_y + v/r`, and the rotation about
the normal, `\omega_3 = \frac{1}{2}(v_{,x} - u_{,y})`, enters the twist. With
Sanders' correction of the displacement field, the term `-v/r` of the 3D
transverse shear strain `\gamma_{yz}` cancels, and no correction appears in
the `z^3` terms, as in the displacement field `(1 + z/r) v + z \phi_y + ...`
of Reddy and Liu (1985), Eqs. (5), (7) and (11). As for the CLPT, see
``theory/shells/cylshell_clpt_sanders``, the non-linear terms omit the term
`\frac{1}{2}\omega_3^2` of the Sanders-Koiter theory for small strains and
moderately small rotations (Amabili 2008, Sec. 1.3.8, Eqs.
(1.133)-(1.134)). The models are:

- ``geometry='plate'``: flat plates, `1/r = 0`, with von Karman kinematics;
- ``geometry='cylshell'``, ``kinematics='donnell'``: cylindrical shells with
  the Donnell kinematics, without the boxed terms;
- ``geometry='cylshell'``, ``kinematics='sanders'``: cylindrical shells with
  the Sanders-Koiter kinematics.

The generalized constitutive matrix, with the rows and columns in the order
of the generalized strains above, is:

- FSDT, ``8 x 8``: ``[[A, B, 0], [B, D, 0], [0, 0, Ats]]``, where ``Ats`` is
  the shear corrected transverse shear stiffness ``[[A44, A45], [A45,
  A55]]`` of the `(yz, xz)` components.

- TSDT, ``13 x 13``: ``T^T F_m T``, with ``F_m = [[A, B, E, 0, 0], [B, D,
  F, 0, 0], [E, F, H, 0, 0], [0, 0, 0, Abar_ts, Dts], [0, 0, 0, Dts,
  Fts]]`` the matrix of the laminate about its mid-surface, i.e. with the
  integrals of the stiffness multiplied by `1, \zeta, \zeta^2, \zeta^3,
  \zeta^4, \zeta^6`, `\zeta = z - d`, and of the transverse shear stiffness
  multiplied by `1`, `\zeta^2` and `\zeta^4`, with no shear correction, and
  ``T`` the identity but for the blocks ``T[0:3, 3:6] = d I`` and ``T[0:3,
  6:9] = d^3 I``, because `\varepsilon = (\varepsilon^{(0)} + d
  \varepsilon^{(1)} + d^3 \varepsilon^{(3)}) + \zeta \varepsilon^{(1)} +
  \zeta^3 \varepsilon^{(3)}`, see :meth:`.Shell._get_lam_ABD`. With `d = 0`
  it is the usual matrix of the laminate.

The mass matrices integrate the kinetic energy of the displacement field
above through the thickness, `z` measured from the reference surface and
spanning the laminate, from `-h/2 + d` to `h/2 + d` with the offset `d`,
as in :func:`composites.laminated_plate`. The kinetic energy being a
polynomial in `z`, the density enters only through its moments: for the
FSDT `I_n = \int \rho z^n dz`, `n = 0, 1, 2`, ``intrho``, ``intrhoz`` and
``intrhoz2`` of :class:`composites.Laminate`; for the TSDT, whose field
contains `z^0`, `z`, `z^2` and `z^3`, the moments about the mid-surface
`J_n = \int \rho \zeta^n d\zeta`, `\zeta = z - d`, `n = 0, 1, 2, 3, 4, 6`,
those of the laminate with a zero offset, with the offset `d` in the
coefficients, since in `\zeta` the field has the powers `\zeta^0`,
`\zeta` and `\zeta^3` only.

The aerodynamic matrices use the piston theory, with the load `q = \beta
w_{,x} + \gamma w` along `w` for a flow along `x`, see
:meth:`.Shell.calc_kA`, where the curvature term `\gamma` exists only for the
cylinders.

In the numerically integrated matrices, ``bx`` and ``by`` are the values of
`\beta_x` and `\beta_y` at the integration point, which vanish unless the
geometric non-linearity is considered.

"""
import os

import numpy as np
from sympy import Matrix as M, Poly, Rational, collect, expand, var, zeros

var('a, b, r, h, d, rho, c1, z, bx, by')
var('I0, I1, I2, I3, I4, I5, I6')
MOMENTS = (I0, I1, I2, I3, I4, I5, I6)
# moments of the density about the mid-surface, zeta = z - d
var('zeta, J0, J1, J2, J3, J4, J5, J6')
JMOMENTS = (J0, J1, J2, J3, J4, J5, J6)
var('aeromu, beta, gamma')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('E11, E12, E16, E22, E26, E66')
var('F11, F12, F16, F22, F26, F66')
var('H11, H12, H16, H22, H26, H66')
var('A44, A45, A55, D44, D45, D55, F44, F45, F55')
var('Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Pxx, Pyy, Pxy, Qy, Qx, Ry, Rx')

FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
DOF = len(FIELDS)

MODELS = {
    'plate_fsdt_donnell': ('fsdt', 'plate', 'donnell'),
    'plate_tsdt_donnell': ('tsdt', 'plate', 'donnell'),
    'cylshell_fsdt_donnell': ('fsdt', 'cylshell', 'donnell'),
    'cylshell_fsdt_sanders': ('fsdt', 'cylshell', 'sanders'),
    'cylshell_tsdt_donnell': ('tsdt', 'cylshell', 'donnell'),
    'cylshell_tsdt_sanders': ('tsdt', 'cylshell', 'sanders'),
}


def sym3(p):
    s = globals()
    return M([[s[p + '11'], s[p + '12'], s[p + '16']],
              [s[p + '12'], s[p + '22'], s[p + '26']],
              [s[p + '16'], s[p + '26'], s[p + '66']]])


def constitutive(theory):
    r"""Generalized constitutive matrix, see the module docstring"""
    A, B, D = sym3('A'), sym3('B'), sym3('D')
    Ats = M([[A44, A45], [A45, A55]])
    if theory == 'fsdt':
        F = zeros(8, 8)
        F[0:3, 0:3] = A
        F[0:3, 3:6] = B
        F[3:6, 0:3] = B
        F[3:6, 3:6] = D
        F[6:8, 6:8] = Ats
        return F
    elif theory == 'tsdt':
        E, Fm, H = sym3('E'), sym3('F'), sym3('H')
        Dts = M([[D44, D45], [D45, D55]])
        Fts = M([[F44, F45], [F45, F55]])
        F = zeros(13, 13)
        F[0:3, 0:3] = A
        F[0:3, 3:6] = B
        F[0:3, 6:9] = E
        F[3:6, 0:3] = B
        F[3:6, 3:6] = D
        F[3:6, 6:9] = Fm
        F[6:9, 0:3] = E
        F[6:9, 3:6] = Fm
        F[6:9, 6:9] = H
        F[9:11, 9:11] = Ats
        F[9:11, 11:13] = Dts
        F[11:13, 9:11] = Dts
        F[11:13, 11:13] = Fts
        return F
    raise ValueError(theory)


def stress_names(theory):
    if theory == 'fsdt':
        return (Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qy, Qx)
    return (Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Pxx, Pyy, Pxy, Qy, Qx, Ry, Rx)


def shape_functions(side, theory, geometry='plate', kinematics='donnell'):
    r"""Operators of side ``'A'`` or ``'B'``, as rows of 5 DOFs"""
    def sf(field, dxi=0, deta=0):
        # derivatives after an underscore, e.g. fAw_xixi, gAu_eta
        fname = 'f{0}{1}{2}'.format(side, field,
                                    '_' + 'xi'*dxi if dxi else '')
        gname = 'g{0}{1}{2}'.format(side, field,
                                    '_' + 'eta'*deta if deta else '')
        row = zeros(1, DOF)
        row[0, FIELDS.index(field)] = var(fname)*var(gname)
        return row

    # 1/r and the switch of the Sanders-Koiter terms
    rinv = 1/r if geometry == 'cylshell' else 0
    sk = 1 if kinematics == 'sanders' else 0
    if geometry == 'plate' and sk:
        raise ValueError('flat plates have no Sanders-Koiter terms')

    u, v, w = sf('u'), sf('v'), sf('w')
    u_x, u_y = (2/a)*sf('u', 1), (2/b)*sf('u', 0, 1)
    v_x, v_y = (2/a)*sf('v', 1), (2/b)*sf('v', 0, 1)
    wA_x, wA_y = (2/a)*sf('w', 1), (2/b)*sf('w', 0, 1)
    w_xx = (2/a)**2*sf('w', 2)
    w_yy = (2/b)**2*sf('w', 0, 2)
    w_xy = (2/a)*(2/b)*sf('w', 1, 1)
    phix, phiy = sf('phix'), sf('phiy')
    phix_x, phix_y = (2/a)*sf('phix', 1), (2/b)*sf('phix', 0, 1)
    phiy_x, phiy_y = (2/a)*sf('phiy', 1), (2/b)*sf('phiy', 0, 1)

    # variations of the rotations of the von Karman terms, beta_x = -w,x and
    # beta_y = -w,y + v/r (Sanders)
    dbx = -wA_x
    dby = -wA_y + sk*rinv*v

    eps0 = M([u_x + bx*dbx,
              v_y + rinv*w + by*dby,
              u_y + v_x + bx*dby + by*dbx])
    eps1 = M([phix_x,
              phiy_y + sk*rinv*v_y,
              phix_y + phiy_x
              + sk*rinv*(Rational(3, 2)*v_x - Rational(1, 2)*u_y)])
    gam0 = M([phiy + wA_y, phix + wA_x])
    if theory == 'fsdt':
        B = M([eps0, eps1, gam0])
    else:
        eps3 = -c1*M([phix_x + w_xx, phiy_y + w_yy, phix_y + phiy_x + 2*w_xy])
        gam2 = -3*c1*gam0
        B = M([eps0, eps1, eps3, gam0, gam2])
    B0 = B.subs({bx: 0, by: 0})

    G = M([dbx, dby])

    # displacements through the thickness
    if theory == 'fsdt':
        uz = u + z*phix
        vz = v + z*(phiy + sk*rinv*v)
    else:
        # the cubic term about the mid-surface, z = d, see the docstring
        cubic = (z - d)**3 + d**3
        uz = u + z*phix - c1*cubic*(phix + wA_x)
        vz = v + z*(phiy + sk*rinv*v) - c1*cubic*(phiy + wA_y)
    wz = w

    return dict(B=B, B0=B0, G=G, uz=uz, vz=vz, wz=wz, w=w, w_x=wA_x, w_y=wA_y)


def tsdt_offset_cubic():
    r"""Cubic `p(z)` of the TSDT field `u(z) = u + z \phi_x + p(z) (\phi_x +
    w_{,x})` with the offset `d`

    Solves `p(0) = 0`, `p'(d \pm h/2) = -1` (zero transverse shear at both
    faces) and `p'(d) = 0` (`\phi_x` the rotation at the mid-surface), and
    returns `p(z)`, which is `-c_1 [(z - d)^3 + d^3]`. Without `p'(d) = 0`
    one parameter is free: it only selects the coordinate where the
    rotation variable is measured, see ``doc/source/tsdt_offset.rst``.

    """
    from sympy import solve, symbols
    p1, p2, p3 = symbols('p1, p2, p3')
    p = p1*z + p2*z**2 + p3*z**3
    dp = p.diff(z)
    sol = solve([dp.subs(z, d - h/2) + 1, dp.subs(z, d + h/2) + 1,
                 dp.subs(z, d)], [p1, p2, p3], dict=True)[0]
    p = p.subs(sol)
    assert expand(p + Rational(4, 3)/h**2*((z - d)**3 + d**3)) == 0
    return p


def thickness_moments(e):
    r"""Integral through the thickness of ``rho*e``, ``e`` a polynomial in
    `z`, with `\int \rho z^n dz` replaced by the moment `I_n`"""
    p = Poly(expand(e), z)
    return sum(coef*MOMENTS[n] for (n, ), coef in p.terms())


def thickness_moments_mid(e):
    r"""Integral through the thickness of ``rho*e``, ``e`` a polynomial in
    `z`, with `z = \zeta + d` and `\int \rho \zeta^n d\zeta` replaced by the
    moment `J_n` about the mid-surface"""
    p = Poly(expand(e.subs(z, zeta + d)), zeta)
    return sum(coef*JMOMENTS[n] for (n, ), coef in p.terms())


def build(theory, geometry='plate', kinematics='donnell'):
    r"""Integrands of all the matrices of ``theory``, 'fsdt' or 'tsdt'

    The integrands are given per unit area of the natural domain, such that
    the integral over the physical domain `x_1 \le x \le x_2`,
    `y_1 \le y \le y_2` is ``intx*inty/4`` times the integral over
    `\xi` and `\eta`, where ``intx = x2 - x1`` and ``inty = y2 - y1``.

    """
    SA = shape_functions('A', theory, geometry, kinematics)
    SB = shape_functions('B', theory, geometry, kinematics)
    F = constitutive(theory)

    # Constitutive stiffness matrix, with KCNL
    kC = SA['B'].T*F*SB['B']

    # Linear constitutive stiffness matrix
    kC0 = SA['B0'].T*F*SB['B0']

    # Internal force vector
    fint = SA['B'].T*M(stress_names(theory))

    # Geometric stiffness matrix
    Nmat = M([[Nxx, Nxy],
              [Nxy, Nyy]])
    kG = SA['G'].T*Nmat*SB['G']

    # Mass matrix, integrated through the thickness
    T = (SA['uz'].T*SB['uz'] + SA['vz'].T*SB['vz'] + SA['wz'].T*SB['wz'])
    kM = T.applyfunc(thickness_moments if theory == 'fsdt' else
                     thickness_moments_mid)

    # Aerodynamic and damping matrices using the piston theory, with the load
    # q = beta*w,x + gamma*w along w, where flat plates have no curvature
    # term gamma, see Shell.calc_kA()
    kAx = -beta*SA['w_x'].T*SB['w']
    if geometry == 'cylshell':
        kAx += -gamma*SA['w'].T*SB['w']
    kAy = -beta*SA['w_y'].T*SB['w']
    cA = -aeromu*SA['w'].T*SB['w']

    return dict(kC=kC, kC0=kC0, fint=fint, kG=kG, kM=kM, kAx=kAx, kAy=kAy,
                cA=cA, F=F, B0=SA['B0'])


if __name__ == '__main__':
    from panels.dev.matrixtools import mprint_as_sparse

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    outdir = './output_expressions_python/'
    os.makedirs(outdir, exist_ok=True)
    for model, args in MODELS.items():
        matrices = build(*args)
        for name in ('kC', 'kC0', 'fint', 'kG', 'kM', 'kAx', 'kAy', 'cA'):
            m = matrices[name]
            out = mprint_as_sparse(m, name, '11', print_file=False)
            with open(outdir + 'sympy_%s_%s.txt' % (model, name), 'w') as f:
                f.write(out)
