r"""Utilities of the notebooks weller1977_blade_cylinder_buckling.ipynb and
peterson1959_blade_cylinder_buckling.ipynb

Axial buckling, and buckling under external pressure, of cylinders with
many closely spaced stringers, each stringer an equivalent blade (a flat
plate strip normal to the skin), with
the membrane pre-buckling state of the smeared-stiffener theory of Block,
Card and Mikulas (1965). The assembly is that of
:func:`blade_cylinder_utils.build_cylinder` (curved ``cylshell_clpt_sanders``
skin panels between the stringers, ``plate_clpt_gl`` blades, connections
``'SSycte'`` and ``'BFycte'``): the half cylinder for an even number of
stringers, the whole cylinder for an odd one. The twisting stiffness
`D_{66}` of the blades may be scaled, so that the twisting stiffness of a
blade, `4 D_{66} h_b`, equals the torsional stiffness `G J` of the stringer
section of the source.

All quantities are in SI units or in any other consistent set.

"""
import time

import numpy as np
from numpy import pi
from structsolve import lb

from blade_cylinder_utils import build_cylinder, skin_w, classify


def buckling_membrane(R, h, L, E, nu, tb, hb, side, bc, nstiff, mx, ny, nyb,
        D66_factor=1., num_eigvalues=10, eps=1.e-4):
    r"""Axial buckling about the membrane pre-buckling state

    The pre-buckling state is the uniform axial strain ``-eps`` without
    Poisson expansion: the constant stress resultants `N_{xx} = -E h`
    ``eps`` in the skin panels and `-E t_b` ``eps`` in the blades.
    ``bc='CC'``: both ends clamped, `u = v = w = w_{,x} = 0` (skin and
    blades, see :data:`blade_cylinder_utils.END_FLAGS`); ``bc='SS'``: both
    ends with `N_x = v = w = M_x = 0` (the simple support SS3), the rigid
    axial translation removed by `u = 0` at the point `x = L/2`, `\theta =
    0`. ``nstiff`` even: the half cylinder with two symmetry planes through
    the middle of a bay; odd: the whole cylinder. ``D66_factor`` multiplies
    `D_{66}` of the blades (their ``ABDnxny``).

    Returns the critical average force per unit circumference `\bar{N} =
    P/(2 \pi R)`, the modes ``[(lambda, m, n, c), ...]`` with `\lambda > 0`
    in increasing order, `\bar{N} = \lambda \bar{N}_1`, the value
    `\bar{N}_1` of the applied state, the assembly, the skin panels, the
    number of degrees of freedom and the run time, s.
    """
    t0 = time.time()
    full = nstiff % 2 == 1
    if bc == 'CC':
        md, skin, blades = build_cylinder(R, h, L, E, nu, tb=tb, hb=hb,
                side=side, bc='CC', nstiff=nstiff, mx=mx, ny=ny, nyb=nyb,
                full=full)
    else:
        md, skin, blades = build_cylinder(R, h, L, E, nu, tb=tb, hb=hb,
                side=side, bc=('SL', 'SL'), nstiff=nstiff, mx=mx, ny=ny,
                nyb=nyb, full=full, pd_method='null-space')
        skin[0].add_point_pd(L/2, 0., 1., 0., None, None, None, None)
    if D66_factor != 1.:
        for bl in blades:
            ABD = np.array(bl.ABD, dtype=float)
            ABD[5, 5] *= D66_factor
            bl.ABDnxny = ABD
    P1 = 0.
    for p in skin:
        p.Nxx = -E*h*eps
        P1 += E*h*eps*p.b
    for p in blades:
        p.Nxx = -E*tb*eps
        P1 += E*tb*eps*p.b
    if not full:
        P1 *= 2
    N1 = P1/(2*pi*R)
    K = md.reduce(md.calc_kC(silent=True))
    KG = md.reduce(md.calc_kG(silent=True))
    eigvals, eigvecs = lb(K, KG, tol=0, sparse_solver=True, silent=True,
                          num_eigvalues=num_eigvalues)
    eigvecs = md.expand(eigvecs)
    modes = []
    for k in range(len(eigvals)):
        c = eigvecs[:, k].real
        x, th, w = skin_w(md, c, skin, R)
        m, n = classify(x, th, w, 'SS', nmax=max(40, nstiff))
        modes.append((eigvals[k].real, m, n, c))
    modes = sorted((mode for mode in modes if mode[0] > 0),
                   key=lambda mode: mode[0])
    return (modes[0][0]*N1, modes, N1, md, skin, K.shape[0],
            time.time() - t0)


def smeared_ss(R, L, t, E, nu, d, A, z, Io, J, mmax=12, nmax=60):
    r"""Smeared-stiffener critical `\bar{N}_x` of the simply supported
    cylinder (SS3, membrane pre-buckling state), equation (15) of Block,
    Card and Mikulas (1965) without rings, isotropic skin and stringers,
    minimized over `m = 1 \ldots` ``mmax`` and `n = 1 \ldots` ``nmax``

    ``d`` is the stringer spacing, ``A`` the area of a stringer, ``z`` the
    distance from its centroid to the middle surface of the skin (positive
    outside), ``Io`` its moment of inertia about the middle surface of the
    skin and ``J`` its torsional constant. Returns `\bar{N}_x` and `(m,
    n)`.
    """
    G = E/(2*(1 + nu))
    C = E*t/(1 - nu**2)
    D = E*t**3/(12*(1 - nu**2))
    m = np.arange(1, mmax + 1)[:, None]
    n = np.arange(1, nmax + 1)[None, :]
    am, bn = m*pi/L, n/R
    EAd = E*A/d
    A11 = (C + EAd)*am**2 + G*t*bn**2
    A12 = (nu*C + G*t)*am*bn
    A13 = nu*C/R*am + EAd*z*am**3
    A22 = G*t*am**2 + C*bn**2
    A23 = C/R*bn
    A33 = ((D + E*Io/d)*am**4 + (2*D + G*J/d)*am**2*bn**2 + D*bn**4
           + C/R**2)
    det = A11*A22 - A12**2
    N = (A33 + (A12*A23 - A13*A22)/det*A13
         + (A12*A13 - A11*A23)/det*A23)/am**2
    i, j = np.unravel_index(np.argmin(N), N.shape)
    return N[i, j], (int(m[i, 0]), int(n[0, j]))


def buckling_pressure(R, h, L, E, nu, tb, hb, side, nstiff, mx, ny, nyb,
        D66_factor=1., hydrostatic=False, Nx0=0., num_eigvalues=10):
    r"""Buckling under uniform external pressure `q` about the membrane
    pre-buckling state, simply supported (SS3) ends

    The assembly and the end conditions are those of
    :func:`buckling_membrane` with ``bc='SS'``. The pre-buckling state of
    the pressure `q` is the hoop force `N_{yy} = -q R` in the skin panels
    (the blades carry no hoop force) and, with ``hydrostatic=True``, the
    axial force `\bar{N}_x = q R/2` per unit circumference of the closed
    ends, carried by skin and blades under a uniform axial strain, as in
    :func:`buckling_membrane`. ``Nx0`` is a fixed axial force per unit
    circumference (positive in compression), distributed in the same way
    and added to the constitutive stiffness through the attributes
    ``Nxx_cte`` of the panels, so that the eigenvalue is the pressure.

    Returns the critical pressure `q`, the modes ``[(q, m, n, c), ...]``
    with `q > 0` in increasing order, the assembly, the skin panels, the
    number of degrees of freedom and the run time, s.
    """
    t0 = time.time()
    full = nstiff % 2 == 1
    md, skin, blades = build_cylinder(R, h, L, E, nu, tb=tb, hb=hb,
            side=side, bc=('SL', 'SL'), nstiff=nstiff, mx=mx, ny=ny,
            nyb=nyb, full=full, pd_method='null-space')
    skin[0].add_point_pd(L/2, 0., 1., 0., None, None, None, None)
    if D66_factor != 1.:
        for bl in blades:
            ABD = np.array(bl.ABD, dtype=float)
            ABD[5, 5] *= D66_factor
            bl.ABDnxny = ABD
    # uniform axial strain of a unit force per unit circumference
    area = 2*pi*R*h + nstiff*tb*hb
    eps1 = 2*pi*R/(E*area)
    Nx1 = R/2 if hydrostatic else 0.
    for p in skin:
        p.Nyy = -R
        p.Nxx = -E*h*eps1*Nx1
        p.Nxx_cte = -E*h*eps1*Nx0
    for p in blades:
        p.Nxx = -E*tb*eps1*Nx1
        p.Nxx_cte = -E*tb*eps1*Nx0
    K = md.reduce(md.calc_kC(silent=True))
    KG = md.reduce(md.calc_kG(silent=True))
    eigvals, eigvecs = lb(K, KG, tol=0, sparse_solver=True, silent=True,
                          num_eigvalues=num_eigvalues)
    eigvecs = md.expand(eigvecs)
    modes = []
    for k in range(len(eigvals)):
        c = eigvecs[:, k].real
        x, th, w = skin_w(md, c, skin, R)
        m, n = classify(x, th, w, 'SS', nmax=max(40, nstiff))
        modes.append((eigvals[k].real, m, n, c))
    modes = sorted((mode for mode in modes if mode[0] > 0),
                   key=lambda mode: mode[0])
    return (modes[0][0], modes, md, skin, K.shape[0], time.time() - t0)


def smeared_ss_pressure(R, L, t, E, nu, d, A, z, Io, J, hydrostatic=False,
        Nx0=0., mmax=12, nmax=60):
    r"""Smeared-stiffener critical external pressure `q` of the simply
    supported cylinder (SS3, membrane pre-buckling state), equation (15) of
    Block, Card and Mikulas (1965) without rings,

    .. math::

        \left(\frac{m \pi}{L}\right)^2 \bar{N}_x + \left(\frac{n}{R}
        \right)^2 \bar{N}_y = \textrm{right-hand side of (15)}

    with `\bar{N}_y = q R` and `\bar{N}_x =` ``Nx0`` `+ q R/2` for
    ``hydrostatic=True``, ``Nx0`` otherwise (positive in compression),
    minimized over `m = 1 \ldots` ``mmax`` and `n = 1 \ldots` ``nmax``; the
    parameters are those of :func:`smeared_ss`. Returns `q` and `(m, n)`.
    """
    G = E/(2*(1 + nu))
    C = E*t/(1 - nu**2)
    D = E*t**3/(12*(1 - nu**2))
    m = np.arange(1, mmax + 1)[:, None]
    n = np.arange(1, nmax + 1)[None, :]
    am, bn = m*pi/L, n/R
    EAd = E*A/d
    A11 = (C + EAd)*am**2 + G*t*bn**2
    A12 = (nu*C + G*t)*am*bn
    A13 = nu*C/R*am + EAd*z*am**3
    A22 = G*t*am**2 + C*bn**2
    A23 = C/R*bn
    A33 = ((D + E*Io/d)*am**4 + (2*D + G*J/d)*am**2*bn**2 + D*bn**4
           + C/R**2)
    det = A11*A22 - A12**2
    rhs = (A33 + (A12*A23 - A13*A22)/det*A13
           + (A12*A13 - A11*A23)/det*A23)
    hyd = 1. if hydrostatic else 0.
    q = (rhs - am**2*Nx0)/(hyd*am**2*R/2 + bn**2*R)
    i, j = np.unravel_index(np.argmin(q), q.shape)
    return q[i, j], (int(m[i, 0]), int(n[0, j]))
