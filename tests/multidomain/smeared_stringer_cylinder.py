r"""Smeared-stringer buckling of axially compressed cylinders

Model module of the tests: the critical axial load of a cylinder with
closely spaced, eccentric stringers of Block, Card and Mikulas (1965), for
simply supported ends, used as the reference of the discrete blade-stiffened
cylinders of :func:`panels.multidomain.create_cylinder_blade_stiffened`.

Notes
-----
Block, Card and Mikulas [block1965]_ smear the stringers over their spacing
`d` (Eq. (6), p. 7), treat them as beams whose axial strain varies linearly
over the section, `\epsilon_{x_s} = u_{,x} - z w_{,xx}`, with a twisting
energy `G_s J_s w_{,xy}^2/d`, and the shell with the Donnell strains of
Eq. (1), p. 5, `w` positive outwards. The total load per unit circumference
`\bar{N}_x` (positive in compression) is carried by the shell and the
stringers loaded with a uniform normal stress (p. 8), and enters only the
term `\bar{N}_x w_{,x}^2` of the potential energy of the external forces,
Eq. (9), p. 8: the pre-buckling state is a membrane state, the prebuckling
rotations of the edges being neglected. With the displacements of Eq. (14),
p. 11,

.. math::

    u = \bar{u} \cos\frac{m \pi x}{a} \cos\frac{n y}{R} \qquad
    v = \bar{v} \sin\frac{m \pi x}{a} \sin\frac{n y}{R} \qquad
    w = \bar{w} \sin\frac{m \pi x}{a} \cos\frac{n y}{R}

which satisfy `w = M_x = N_x = v = 0` at `x = 0` and `x = a` (classical
simple support, `u` free), and no rings, `\bar{N}_y = 0`, the stability
equation (15), p. 12, gives

.. math::

    \left(\frac{m\pi}{a}\right)^2 \bar{N}_x = A_{33}
        + \frac{A_{12} A_{23} - A_{13} A_{22}}{A_{11} A_{22} - A_{12}^2} A_{13}
        + \frac{A_{12} A_{13} - A_{11} A_{23}}{A_{11} A_{22} - A_{12}^2} A_{23}

with, writing `k_x = m \pi/a` and `k_y = n/R`,

.. math::

    A_{11} = \left(\bar{A}_{11} + \frac{E_s A_s}{d}\right) k_x^2
             + \bar{A}_{66} k_y^2 \qquad
    A_{12} = (\bar{A}_{12} + \bar{A}_{66}) k_x k_y \qquad
    A_{13} = \frac{\bar{A}_{12}}{R} k_x + \frac{E_s A_s}{d} \bar{z}_s k_x^3

    A_{22} = \bar{A}_{66} k_x^2 + \bar{A}_{22} k_y^2 \qquad
    A_{23} = \frac{\bar{A}_{22}}{R} k_y

    A_{33} = \left(\bar{D}_{11} + \frac{E_s I_{o_s}}{d}\right) k_x^4
           + \left(2 \bar{D}_{12} + 4 \bar{D}_{66}
                   + \frac{G_s J_s}{d}\right) k_x^2 k_y^2
           + \bar{D}_{22} k_y^4 + \frac{\bar{A}_{22}}{R^2}

where the constants of the orthotropic shell of Block et al. are written
as the membrane and bending stiffnesses of a specially orthotropic laminate,
`E_x/(1 - \mu_x' \mu_y') = \bar{A}_{11}`, `\mu_y' E_x/(1 - \mu_x' \mu_y') =
\bar{A}_{12}`, `E_y/(1 - \mu_x' \mu_y') = \bar{A}_{22}`, `G_{xy} =
\bar{A}_{66}`, `D_x/(1 - \mu_x \mu_y) = \bar{D}_{11}`, `\mu_y D_x/(1 - \mu_x
\mu_y) = \bar{D}_{12}`, `D_y/(1 - \mu_x \mu_y) = \bar{D}_{22}` and `2 D_{xy}
= 4 \bar{D}_{66}` (Eq. (12), p. 10, `D_{xy} = G t^3/6` for an isotropic
shell). `E_s A_s`, `E_s I_{o_s}` (about the middle surface of the shell),
`G_s J_s` are the stiffnesses of one stringer and `\bar{z}_s` the distance
from its centroid to the middle surface of the shell, positive for external
stringers (p. 14). The critical load is the minimum over the integers `m
\ge 1` and `n \ge 0` (p. 13). For an isotropic shell the equation reduces to
Eq. (18), p. 15, implemented in :func:`nx_cr_isotropic_eq18` to check
:func:`nx_cr`.

References
----------
.. [block1965] Block, D. L., Card, M. F., and Mikulas, M. M., Jr., "Buckling
   of Eccentrically Stiffened Orthotropic Cylinders", NASA TN D-2960, 1965.
   https://ntrs.nasa.gov/citations/19650021133

"""
import numpy as np


def blade_stringer(E, nu, height, thickness, internal):
    r"""Stiffnesses of a rectangular blade stringer standing on the middle
    surface of the shell

    The blade has the given ``height``, measured from the middle surface of
    the shell, and ``thickness``, as the blade domains of
    :func:`panels.multidomain.create_cylinder_blade_stiffened`.

    Parameters
    ----------
    E : float
        Young's modulus of the stringer, force per unit area.
    nu : float
        Poisson's ratio of the stringer, dimensionless.
    height : float
        Height of the blade from the middle surface of the shell, length.
    thickness : float
        Thickness of the blade, length.
    internal : bool
        ``True`` for a stringer inside the cylinder, ``False`` outside.

    Returns
    -------
    stringer : dict
        ``'EA'`` (force), ``'EI0'`` (force times length squared, about the
        middle surface of the shell), ``'GJ'`` (force times length squared)
        and ``'z'`` (length, negative for an internal stringer).

    Notes
    -----
    `A_s = h t_b`, `\bar{z}_s = \mp h/2`, `I_{o_s} = t_b h^3/3` and the
    torsion constant of a thin rectangle `J_s = h t_b^3/3`, with `G_s = E/(2(1
    + \nu))`, which is also the twisting stiffness of the blade modelled as
    a Kirchhoff plate strip, `4 D_{66} h`.

    """
    A = height*thickness
    G = E/(2*(1 + nu))
    z = -height/2 if internal else height/2
    return dict(EA=E*A, EI0=E*thickness*height**3/3, GJ=G*height*thickness**3/3,
                z=z)


def rectangle_stringer(E, nu, height, thickness, t_skin, internal):
    r"""Stiffnesses of a rectangular stringer standing on the surface of the
    shell, as in Card and Jones (1966)

    Parameters
    ----------
    E : float
        Young's modulus of the stringer, force per unit area.
    nu : float
        Poisson's ratio of the stringer, dimensionless.
    height : float
        Height of the stringer above the surface of the shell, length.
    thickness : float
        Thickness of the stringer, length.
    t_skin : float
        Thickness of the shell, length.
    internal : bool
        ``True`` for a stringer inside the cylinder, ``False`` outside.

    Returns
    -------
    stringer : dict
        As :func:`blade_stringer`.

    Notes
    -----
    `A_s = h t_s`, `\bar{z}_s = \pm (t/2 + h/2)`, `I_{o_s} = t_s h^3/12 +
    A_s \bar{z}_s^2` and the torsion constant of the symbol list of Card and
    Jones [card1966]_ (p. 3, after their ref. 9),

    .. math::

        J = t_s^3 h \left(\frac{1}{3} - \frac{64}{\pi^5} \frac{t_s}{h}
            \tanh\frac{\pi h}{2 t_s}\right)

    References
    ----------
    .. [card1966] Card, M. F., and Jones, R. M., "Experimental and
       Theoretical Results for Buckling of Eccentrically Stiffened
       Cylinders", NASA TN D-3639, 1966.

    """
    A = height*thickness
    G = E/(2*(1 + nu))
    zc = t_skin/2 + height/2
    J = thickness**3*height*(1/3 - 64/np.pi**5*thickness/height
                             *np.tanh(np.pi*height/(2*thickness)))
    return dict(EA=E*A, EI0=E*(thickness*height**3/12 + A*zc**2), GJ=G*J,
                z=-zc if internal else zc)


def isotropic_skin(E, nu, t):
    r"""Membrane and bending stiffnesses of an isotropic shell wall

    Parameters
    ----------
    E : float
        Young's modulus, force per unit area.
    nu : float
        Poisson's ratio, dimensionless.
    t : float
        Thickness, length.

    Returns
    -------
    skin : dict
        ``'A11'``, ``'A12'``, ``'A22'``, ``'A66'`` (force per unit length) and
        ``'D11'``, ``'D12'``, ``'D22'``, ``'D66'`` (force times length).

    Notes
    -----
    `\bar{A}_{11} = \bar{A}_{22} = E t/(1 - \nu^2)`, `\bar{A}_{12} = \nu
    \bar{A}_{11}`, `\bar{A}_{66} = E t/(2(1 + \nu))`, and `\bar{D}_{ij} =
    \bar{A}_{ij} t^2/12`, Eq. (12) of Block et al. (1965).

    """
    A11 = E*t/(1 - nu**2)
    A66 = E*t/(2*(1 + nu))
    skin = dict(A11=A11, A12=nu*A11, A22=A11, A66=A66)
    for k in ('11', '12', '22', '66'):
        skin['D' + k] = skin['A' + k]*t**2/12
    return skin


def nx_mn(skin, stringer, d, R, a, m, n):
    r"""Axial load per unit circumference of the mode ``(m, n)``, Eq. (15)
    of Block et al. (1965) with `\bar{N}_y = 0` and no rings

    Parameters
    ----------
    skin : dict
        Stiffnesses of the shell, see :func:`isotropic_skin`.
    stringer : dict
        Stiffnesses of one stringer, see :func:`blade_stringer`.
    d : float
        Stringer spacing, length.
    R : float
        Radius of the middle surface of the shell, length.
    a : float
        Length of the cylinder, length.
    m : int
        Number of axial half waves, ``m >= 1``.
    n : int
        Number of circumferential full waves, ``n >= 0``.

    Returns
    -------
    Nx : float
        `\bar{N}_x`, force per unit length, positive in compression.

    """
    kx = m*np.pi/a
    ky = n/R
    EAd = stringer['EA']/d
    A11 = (skin['A11'] + EAd)*kx**2 + skin['A66']*ky**2
    A12 = (skin['A12'] + skin['A66'])*kx*ky
    A13 = skin['A12']/R*kx + EAd*stringer['z']*kx**3
    A22 = skin['A66']*kx**2 + skin['A22']*ky**2
    A23 = skin['A22']/R*ky
    A33 = ((skin['D11'] + stringer['EI0']/d)*kx**4
           + (2*skin['D12'] + 4*skin['D66'] + stringer['GJ']/d)*kx**2*ky**2
           + skin['D22']*ky**4 + skin['A22']/R**2)
    det = A11*A22 - A12**2
    rhs = (A33 + (A12*A23 - A13*A22)/det*A13
           + (A12*A13 - A11*A23)/det*A23)
    return rhs/kx**2


def nx_cr(skin, stringer, d, R, a, m_max=20, n_max=60):
    r"""Critical axial load of the smeared-stringer cylinder, Eq. (15) of
    Block et al. (1965) minimized over the integers ``m`` and ``n``

    Parameters
    ----------
    skin, stringer, d, R, a
        See :func:`nx_mn`.
    m_max, n_max : int, optional
        Largest ``m`` and ``n`` searched; the minimum must lie inside the
        range (checked).

    Returns
    -------
    Nx, m, n : tuple
        Critical `\bar{N}_x` (force per unit length, positive in
        compression) and its numbers of waves.

    Raises
    ------
    ValueError
        When the minimum is at ``m_max`` or ``n_max``.

    """
    best = (np.inf, None, None)
    for m in range(1, m_max + 1):
        for n in range(0, n_max + 1):
            Nx = nx_mn(skin, stringer, d, R, a, m, n)
            if Nx < best[0]:
                best = (Nx, m, n)
    if best[1] == m_max or best[2] == n_max:
        raise ValueError('minimum at the limit of the search range')
    return best


def nx_cr_isotropic_eq18(E, nu, t, stringer, d, R, a, m, n):
    r"""Axial load of the mode ``(m, n)`` of an isotropic cylinder with
    stringers only, Eq. (18) of Block et al. (1965)

    Parameters
    ----------
    E, nu, t : float
        Young's modulus (force per unit area), Poisson's ratio and thickness
        (length) of the shell.
    stringer, d, R, a, m, n
        See :func:`nx_mn`.

    Returns
    -------
    Nx : float
        `\bar{N}_x`, force per unit length, positive in compression.

    Notes
    -----
    .. math::

        \frac{\bar{N}_x a^2}{\pi^2 D} = m^2 (1 + \beta^2)^2
            + m^2 \frac{E_s I_s}{d D} + m^2 \beta^2 \frac{G_s J_s}{d D}
            + \frac{12 Z^2}{m^2 \pi^4}
              \frac{1 + \bar{S} \Lambda_s}{(1 + \beta^2)^2
              + 2 \bar{S} \beta^2 (1 + \mu) + \bar{S}(1 - \mu^2)}

    with `\Lambda_s = 1 + 2\alpha^2(\beta^2 - \mu) \bar{z}_s/R + \alpha^4
    (1 + \beta^2)^2 (\bar{z}_s/R)^2` (p. 14), `\alpha = m \pi R/a`, `\beta =
    n a/(m \pi R)`, `\bar{S} = E_s A_s/(E t d)`, `Z^2 = a^4 (1 - \mu^2)/(R^2
    t^2)`, `D = E t^3/(12 (1 - \mu^2))`, `I_s` about the centroid of the
    stringer.

    """
    D = E*t**3/(12*(1 - nu**2))
    alpha = m*np.pi*R/a
    beta = n*a/(m*np.pi*R)
    S = stringer['EA']/(E*t*d)
    zR = stringer['z']/R
    EIs = stringer['EI0'] - stringer['EA']*stringer['z']**2
    Z2 = a**4*(1 - nu**2)/(R**2*t**2)
    Lam = (1 + 2*alpha**2*(beta**2 - nu)*zR
           + alpha**4*(1 + beta**2)**2*zR**2)
    rhs = (m**2*(1 + beta**2)**2 + m**2*EIs/(d*D)
           + m**2*beta**2*stringer['GJ']/(d*D)
           + 12*Z2/(m**2*np.pi**4)*(1 + S*Lam)
             /((1 + beta**2)**2 + 2*S*beta**2*(1 + nu) + S*(1 - nu**2)))
    return rhs*np.pi**2*D/a**2
