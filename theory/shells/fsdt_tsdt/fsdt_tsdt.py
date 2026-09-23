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
shear deformation theory (FSDT):

.. math::

    u(z) = u + z \phi_x - c_1 z^3 (\phi_x + w_{,x})

    v(z) = v + z \left(\phi_y + \boxed{\frac{v}{r}}\right)
           - c_1 z^3 (\phi_y + w_{,y})

    w(z) = w

The strains are, in Voigt notation, `\varepsilon = \varepsilon^{(0)} + z
\varepsilon^{(1)} + z^3 \varepsilon^{(3)}` and `\gamma = \gamma^{(0)} + z^2
\gamma^{(2)}`, with the rotations `\beta_x = w_{,x}` and `\beta_y = w_{,y} -
\boxed{v/r}` of the von Karman non-linear terms:

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
the `z^3` terms. The models are:

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

- TSDT, ``13 x 13``: ``[[A, B, E, 0, 0], [B, D, F, 0, 0], [E, F, H, 0, 0],
  [0, 0, 0, Abar_ts, Dtrans], [0, 0, 0, Dtrans, Ftrans]]``, where
  ``Abar_ts``, ``Dtrans`` and ``Ftrans`` are the integrals of the
  transverse shear stiffness multiplied by `1`, `z^2` and `z^4`, with no
  shear correction.

The mass matrices integrate the kinetic energy of the displacement field
above through the thickness, with the same convention for the laminate
offset ``d`` as the CLPT mass matrices of this package, i.e. `z` spans from
`-h/2 - d` to `h/2 - d`.

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
from sympy import Matrix as M, Rational, collect, expand, integrate, var, zeros

var('a, b, r, h, d, rho, c1, z, bx, by')
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
        fname = 'f{0}{1}{2}'.format(side, field, 'xi'*dxi)
        gname = 'g{0}{1}{2}'.format(side, field, 'eta'*deta)
        row = zeros(1, DOF)
        row[0, FIELDS.index(field)] = var(fname)*var(gname)
        return row

    # 1/r and the switch of the Sanders-Koiter terms
    rinv = 1/r if geometry == 'cylshell' else 0
    sk = 1 if kinematics == 'sanders' else 0
    if geometry == 'plate' and sk:
        raise ValueError('flat plates have no Sanders-Koiter terms')

    u, v, w = sf('u'), sf('v'), sf('w')
    ux, uy = (2/a)*sf('u', 1), (2/b)*sf('u', 0, 1)
    vx, vy = (2/a)*sf('v', 1), (2/b)*sf('v', 0, 1)
    wxA, wyA = (2/a)*sf('w', 1), (2/b)*sf('w', 0, 1)
    wxx = (2/a)**2*sf('w', 2)
    wyy = (2/b)**2*sf('w', 0, 2)
    wxy = (2/a)*(2/b)*sf('w', 1, 1)
    phix, phiy = sf('phix'), sf('phiy')
    phixx, phixy = (2/a)*sf('phix', 1), (2/b)*sf('phix', 0, 1)
    phiyx, phiyy = (2/a)*sf('phiy', 1), (2/b)*sf('phiy', 0, 1)

    # variations of the rotations of the von Karman terms, beta_x = w,x and
    # beta_y = w,y - v/r (Sanders)
    dbx = wxA
    dby = wyA - sk*rinv*v

    eps0 = M([ux + bx*dbx,
              vy + rinv*w + by*dby,
              uy + vx + bx*dby + by*dbx])
    eps1 = M([phixx,
              phiyy + sk*rinv*vy,
              phixy + phiyx + sk*rinv*(Rational(3, 2)*vx - Rational(1, 2)*uy)])
    gam0 = M([phiy + wyA, phix + wxA])
    if theory == 'fsdt':
        B = M([eps0, eps1, gam0])
    else:
        eps3 = -c1*M([phixx + wxx, phiyy + wyy, phixy + phiyx + 2*wxy])
        gam2 = -3*c1*gam0
        B = M([eps0, eps1, eps3, gam0, gam2])
    B0 = B.subs({bx: 0, by: 0})

    G = M([dbx, dby])

    # displacements through the thickness
    if theory == 'fsdt':
        uz = u + z*phix
        vz = v + z*(phiy + sk*rinv*v)
    else:
        uz = u + z*phix - c1*z**3*(phix + wxA)
        vz = v + z*(phiy + sk*rinv*v) - c1*z**3*(phiy + wyA)
    wz = w

    return dict(B=B, B0=B0, G=G, uz=uz, vz=vz, wz=wz, w=w, wx=wxA, wy=wyA)


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

    # Constitutive stiffness matrix, with K0L, KL0 and KLL
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
    kM = rho*T.applyfunc(lambda e: integrate(expand(e), (z, -h/2 - d, h/2 - d)))

    # Aerodynamic and damping matrices using the piston theory, with the load
    # q = beta*w,x + gamma*w along w, where flat plates have no curvature
    # term gamma, see Shell.calc_kA()
    kAx = -beta*SA['wx'].T*SB['w']
    if geometry == 'cylshell':
        kAx += -gamma*SA['w'].T*SB['w']
    kAy = -beta*SA['wy'].T*SB['w']
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
