r"""Flutter of a T-stiffened panel with a debonding defect
(:mod:`panels.multidomain.tstiff2d_1stiff_flutter`)

The function :func:`.tstiff2d_1stiff_flutter`, imported in
:mod:`panels.multidomain`, builds the assembly of 15 domains of a panel
with one T stiffener and runs its frequency analysis with the aerodynamic
matrix of the linear piston theory of the skin.

"""
import numpy as np
from scipy.sparse import csr_matrix

from structsolve import freq, static

from panels.shell import Shell
from panels.multidomain import MultiDomain


def tstiff2d_1stiff_flutter(a, b, ys, bb, bf, defect_a, rho, plyt,
        laminaprop, stack_skin, stack_base, stack_flange,
        air_speed=None, rho_air=None, Mach=None, speed_sound=None, flow='x',
        Nxx_skin=None, Nxx_base=None, Nxx_flange=None, run_static_case=True,
        r=None, m=8, n=8, mb=None, nb=None, mf=None, nf=None,
        num_eigvalues=25, conn_method='null-space'):
    r"""Flutter of T-Stiffened Shell with possible defect at middle

    The panel assembly looks like::

        skin
         _________ _____ _________
        |         |     |         |
        |         |     |         |
        |   p01   | p02 |   p03   |
        |         |     |         |
        |_________|_____|_________|
        |   p04   | p05 |   p06   |      /\  x
        |_________|_____|_________|       |
        |         |     |         |       |
        |         |     |         |       |
        |   p07   | p08 |   p09   |
        |         |     |         |
        |         |     |         |
        |_________|_____|_________|
               loaded edge

                  base            flange
                   _____           _____
                  |     |         |     |
                  |     |         |     |
                  | p10 |         | p11 |
                  |     |         |     |
                  |_____|         |_____|
                  | p12 |         | p13 |
                  |_____|         |_____|
                  |     |         |     |
                  |     |         |     |
                  | p14 |         | p15 |
                  |     |         |     |
                  |     |         |     |
                  |_____|         |_____|
               loaded edge     loaded edge

    The domains and their connections are those of
    :func:`.tstiff2d_1stiff_compression` with ``run_static_case=True``: the
    middle row of domains has the length ``defect_a*a`` of the defect, or
    ``0.33*a`` when ``defect_a = 0``, and its base is not connected to the
    skin when ``defect_a > 0``. The aerodynamic parameters are given to the
    9 skin domains only, and the aerodynamic matrix is the sum of
    :meth:`.Shell.calc_kA` of the skin domains.

    With ``run_static_case=True`` and at least one of ``Nxx_skin``,
    ``Nxx_base`` and ``Nxx_flange`` given, a distributed force along `x` is
    added with :meth:`.Shell.add_distr_load_fixed_x` at the edge `x = 0` of
    those of the domains ``p07``, ``p08``, ``p09``, ``p14`` and ``p15``
    whose ``Nxx`` is not ``None``, and a linear static analysis with
    ``structsolve.static`` gives the pre-buckling state ``c``. The attributes
    ``Nxx`` of all the domains are then set to ``0`` and the geometric
    stiffness matrix is computed at ``c``, or with ``c=None`` when no load
    was given. The reduced matrices, see :meth:`.MultiDomain.reduce`, the sum
    of the constitutive, geometric and aerodynamic stiffness matrices and
    the mass matrix, are passed to ``structsolve.freq`` with ``tol=0`` and
    ``sparse_solver=True``, and the eigenvectors are expanded with
    :meth:`.MultiDomain.expand`.

    Parameters
    ----------

    a : float
        Total length of the assembly (along `x`), length.
    b : float
        Total width of the assembly (along `y`), length.
    ys : float
        Position of the stiffener along `y`, length.
    bb : float
        Stiffener's base width, length.
    bf : float
        Stiffener's flange width, length.
    defect_a : float
        Debonding defect/assembly length ratio, dimensionless.
    rho : float
        Material density, mass per unit volume.
    plyt : float
        Ply thickness, length.
    laminaprop : list or tuple
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13}, G_{23}`.
        The moduli are forces per unit area, `\nu_{12}` is dimensionless.
    stack_skin : list or tuple
        Stacking sequence for the skin, the angle of each ply in degrees.
    stack_base : list or tuple
        Stacking sequence for the stiffener's base, the angles in degrees.
    stack_flange : list or tuple
        Stacking sequence for the stiffener's flange, the angles in degrees.
    air_speed : float, optional
        Airflow speed, length per unit time. Default ``None``.
    rho_air : float, optional
        Air density, mass per unit volume. Default ``None``.
    Mach : float, optional
        Mach number, dimensionless. Default ``None``; it must be given
        and not lower than ``1``, see :meth:`.Shell.calc_kA`.
    speed_sound : float, optional
        Speed of sound, length per unit time, stored in the skin domains.
        Default ``None``.
    flow : "x" or "y", optional
        Direction of airflow. Default ``'x'``.
    Nxx_skin : float or None, optional
        Skin load distributed at the assembly edge at `x=0`, force per unit
        length. Default ``None``, no load.
    Nxx_base : float or None, optional
        Stiffener's base load distributed at the assembly edge at `x=0`,
        force per unit length. Default ``None``, no load.
    Nxx_flange : float or None, optional
        Stiffener's flange load distributed at the assembly edge at `x=0`,
        force per unit length. Default ``None``, no load.
    run_static_case : bool, optional
        If True a static analysis is run before the frequency analysis
        to compute the real membrane stress state along the domain, otherwise
        the geometric stiffness matrix is not included and ``Nxx_skin``,
        ``Nxx_base`` and ``Nxx_flange`` are not used. Default ``True``.
    r : float or None, optional
        Radius of the stiffened panel, length. Default ``None``, a flat
        panel.
    m, n : int, optional
        Number of terms of the approximation function for the skin. Default
        ``8``.
    mb, nb : int, optional
        Number of terms of the approximation function for the stiffener's base.
        ``None``, the default, uses ``m`` and ``n``.
    mf, nf : int, optional
        Number of terms of the approximation function for the stiffener's
        flange. ``None``, the default, uses ``m`` and ``n``.
    num_eigvalues : int, optional
        Number of eigenvalues to be extracted. Default ``25``.
    conn_method : str, optional
        How the connections are imposed, ``'null-space'``, the default, or
        ``'penalty'``, see :class:`.MultiDomain`.

    Returns
    -------
    assy, eigvals, eigvecs : tuple
        The assembly, :class:`.MultiDomain`, the eigenvalues returned by
        ``structsolve.freq``, see there, and the eigenvectors of the
        assembly, one per column, with ``assy.get_size()`` rows.

    Notes
    -----
    For more details about each parameter and the aerodynamic formulation see
    Ref. [1]_ . The eigenvalue problem solved by ``structsolve.freq`` is

    .. math::

        \left([K_r] + [K_{G r}] + [K_{A r}] + \lambda^2 [M_r]\right)
        \{\phi_r\} = \{0\}

    with the reduced constitutive, geometric and aerodynamic stiffness
    matrices and the reduced mass matrix, see :meth:`.MultiDomain.reduce`.
    `[K_A]` is the matrix of the linear piston theory of
    :meth:`.Shell.calc_kA`, of the skin domains only, since the stiffener is
    not exposed to the flow; it is not symmetric, and the general
    eigenvalue solver is used. The sign convention of `[K_A]` is that of
    :meth:`.Shell.calc_kA`: its term in `\beta` is the transpose of that of
    Eq. (21) of [1]_, equivalent to reversing the flow direction, and the
    term in `\gamma` of Eqs. (21) and (29) of [1]_ has the wrong sign, a
    mistake of the paper, the correction of Krumhaar softening the shell. The flutter onset is found by increasing the
    aerodynamic load, e.g. ``air_speed``, until two natural frequencies
    coalesce and the eigenvalues become complex. With
    ``run_static_case=True`` and loads, `[K_G]` is computed from the
    linear membrane stresses of the static solution, see
    :func:`.tstiff2d_1stiff_compression` for the sign of the loads and the
    model of the debonding defect.

    References
    ----------
    .. [1] Castro, S. G. P., Guimarães, T. A. M., Rade, D. A., and Donadon,
       M. V., Flutter of stiffened composite panels considering the
       stiffener's base as a structural element, Composite Structures,
       2016. https://doi.org/10.1016/j.compstruct.2015.12.056

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../tests/multidomain/test_tstiff2d_assembly.py
        :pyobject: test_tstiff2d_1stiff_flutter

    """
    defect = defect_a * a
    has_defect = True if defect > 0 else False
    defect = 0.33*a if defect == 0 else defect # to avoid weird domains
    aup = (a - defect)/2.
    alow = (a - defect)/2.
    bleft = b - ys - bb/2.
    bright = ys - bb/2.
    mb = m if mb is None else mb
    nb = n if nb is None else nb
    mf = m if mf is None else mf
    nf = n if nf is None else nf
    # skin panels
    p01 = Shell(group='skin', Nxx=Nxx_skin, x0=alow+defect, y0=ys+bb/2., a=aup, b=bleft, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, rho=rho, rho_air=rho_air, speed_sound=speed_sound, Mach=Mach, air_speed=air_speed, flow=flow)
    p02 = Shell(group='skin', Nxx=Nxx_skin, x0=alow+defect, y0=ys-bb/2., a=aup, b=bb, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, rho=rho, rho_air=rho_air, speed_sound=speed_sound, Mach=Mach, air_speed=air_speed, flow=flow)
    p03 = Shell(group='skin', Nxx=Nxx_skin, x0=alow+defect, y0=0, a=aup, b=bright, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, rho=rho, rho_air=rho_air, speed_sound=speed_sound, Mach=Mach, air_speed=air_speed, flow=flow)
    # defect
    p04 = Shell(group='skin', Nxx=Nxx_skin, x0=alow, y0=ys+bb/2., a=defect, b=bleft, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, rho=rho, rho_air=rho_air, speed_sound=speed_sound, Mach=Mach, air_speed=air_speed, flow=flow)
    p05 = Shell(group='skin', Nxx=Nxx_skin, x0=alow, y0=ys-bb/2., a=defect, b=bb, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, rho=rho, rho_air=rho_air, speed_sound=speed_sound, Mach=Mach, air_speed=air_speed, flow=flow)
    p06 = Shell(group='skin', Nxx=Nxx_skin, x0=alow, y0=0, a=defect, b=bright, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, rho=rho, rho_air=rho_air, speed_sound=speed_sound, Mach=Mach, air_speed=air_speed, flow=flow)
    #
    p07 = Shell(group='skin', Nxx=Nxx_skin, x0=0, y0=ys+bb/2., a=alow, b=bleft, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, rho=rho, rho_air=rho_air, speed_sound=speed_sound, Mach=Mach, air_speed=air_speed, flow=flow)
    p08 = Shell(group='skin', Nxx=Nxx_skin, x0=0, y0=ys-bb/2., a=alow, b=bb, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, rho=rho, rho_air=rho_air, speed_sound=speed_sound, Mach=Mach, air_speed=air_speed, flow=flow)
    p09 = Shell(group='skin', Nxx=Nxx_skin, x0=0, y0=0, a=alow, b=bright, r=r, m=m, n=n, plyt=plyt, stack=stack_skin, laminaprop=laminaprop, rho=rho, rho_air=rho_air, speed_sound=speed_sound, Mach=Mach, air_speed=air_speed, flow=flow)

    # stiffeners
    p10 = Shell(group='base', Nxx=Nxx_base, x0=alow+defect, y0=ys-bb/2., a=aup, b=bb, r=r, m=mb, n=nb, plyt=plyt, stack=stack_base, laminaprop=laminaprop, rho=rho)
    p11 = Shell(group='flange', Nxx=Nxx_flange, x0=alow+defect, y0=0, a=aup, b=bf, m=mf, n=nf, plyt=plyt, stack=stack_flange, laminaprop=laminaprop, rho=rho)
    # defect
    p12 = Shell(group='base', Nxx=Nxx_base, x0=alow, y0=ys-bb/2., a=defect, b=bb, r=r, m=mb, n=nb, plyt=plyt, stack=stack_base, laminaprop=laminaprop, rho=rho)
    p13 = Shell(group='flange', Nxx=Nxx_flange, x0=alow, y0=0, a=defect, b=bf, m=mf, n=nf, plyt=plyt, stack=stack_flange, laminaprop=laminaprop, rho=rho)
    #
    p14 = Shell(group='base', Nxx=Nxx_base, x0=0, y0=ys-bb/2., a=alow, b=bb, r=r, m=mb, n=nb, plyt=plyt, stack=stack_base, laminaprop=laminaprop, rho=rho)
    p15 = Shell(group='flange', Nxx=Nxx_flange, x0=0, y0=0, a=alow, b=bf, m=mf, n=nf, plyt=plyt, stack=stack_flange, laminaprop=laminaprop, rho=rho)

    # boundary conditions
    p01.x1u = 1 ; p01.x1ur = 1 ; p01.x2u = 0 ; p01.x2ur = 1
    p01.x1v = 1 ; p01.x1vr = 1 ; p01.x2v = 0 ; p01.x2vr = 1
    p01.x1w = 1 ; p01.x1wr = 1 ; p01.x2w = 0 ; p01.x2wr = 1
    p01.y1u = 1 ; p01.y1ur = 1 ; p01.y2u = 1 ; p01.y2ur = 1
    p01.y1v = 1 ; p01.y1vr = 1 ; p01.y2v = 0 ; p01.y2vr = 1
    p01.y1w = 1 ; p01.y1wr = 1 ; p01.y2w = 0 ; p01.y2wr = 1

    p02.x1u = 1 ; p02.x1ur = 1 ; p02.x2u = 0 ; p02.x2ur = 1
    p02.x1v = 1 ; p02.x1vr = 1 ; p02.x2v = 0 ; p02.x2vr = 1
    p02.x1w = 1 ; p02.x1wr = 1 ; p02.x2w = 0 ; p02.x2wr = 1
    p02.y1u = 1 ; p02.y1ur = 1 ; p02.y2u = 1 ; p02.y2ur = 1
    p02.y1v = 1 ; p02.y1vr = 1 ; p02.y2v = 1 ; p02.y2vr = 1
    p02.y1w = 1 ; p02.y1wr = 1 ; p02.y2w = 1 ; p02.y2wr = 1

    p03.x1u = 1 ; p03.x1ur = 1 ; p03.x2u = 0 ; p03.x2ur = 1
    p03.x1v = 1 ; p03.x1vr = 1 ; p03.x2v = 0 ; p03.x2vr = 1
    p03.x1w = 1 ; p03.x1wr = 1 ; p03.x2w = 0 ; p03.x2wr = 1
    p03.y1u = 1 ; p03.y1ur = 1 ; p03.y2u = 1 ; p03.y2ur = 1
    p03.y1v = 0 ; p03.y1vr = 1 ; p03.y2v = 1 ; p03.y2vr = 1
    p03.y1w = 0 ; p03.y1wr = 1 ; p03.y2w = 1 ; p03.y2wr = 1

    p04.x1u = 1 ; p04.x1ur = 1 ; p04.x2u = 1 ; p04.x2ur = 1
    p04.x1v = 1 ; p04.x1vr = 1 ; p04.x2v = 1 ; p04.x2vr = 1
    p04.x1w = 1 ; p04.x1wr = 1 ; p04.x2w = 1 ; p04.x2wr = 1
    p04.y1u = 1 ; p04.y1ur = 1 ; p04.y2u = 1 ; p04.y2ur = 1
    p04.y1v = 1 ; p04.y1vr = 1 ; p04.y2v = 0 ; p04.y2vr = 1
    p04.y1w = 1 ; p04.y1wr = 1 ; p04.y2w = 0 ; p04.y2wr = 1

    p05.x1u = 1 ; p05.x1ur = 1 ; p05.x2u = 1 ; p05.x2ur = 1
    p05.x1v = 1 ; p05.x1vr = 1 ; p05.x2v = 1 ; p05.x2vr = 1
    p05.x1w = 1 ; p05.x1wr = 1 ; p05.x2w = 1 ; p05.x2wr = 1
    p05.y1u = 1 ; p05.y1ur = 1 ; p05.y2u = 1 ; p05.y2ur = 1
    p05.y1v = 1 ; p05.y1vr = 1 ; p05.y2v = 1 ; p05.y2vr = 1
    p05.y1w = 1 ; p05.y1wr = 1 ; p05.y2w = 1 ; p05.y2wr = 1

    p06.x1u = 1 ; p06.x1ur = 1 ; p06.x2u = 1 ; p06.x2ur = 1
    p06.x1v = 1 ; p06.x1vr = 1 ; p06.x2v = 1 ; p06.x2vr = 1
    p06.x1w = 1 ; p06.x1wr = 1 ; p06.x2w = 1 ; p06.x2wr = 1
    p06.y1u = 1 ; p06.y1ur = 1 ; p06.y2u = 1 ; p06.y2ur = 1
    p06.y1v = 0 ; p06.y1vr = 1 ; p06.y2v = 1 ; p06.y2vr = 1
    p06.y1w = 0 ; p06.y1wr = 1 ; p06.y2w = 1 ; p06.y2wr = 1

    p07.x1u = 1 ; p07.x1ur = 1 ; p07.x2u = 1 ; p07.x2ur = 1
    p07.x1v = 0 ; p07.x1vr = 1 ; p07.x2v = 1 ; p07.x2vr = 1
    p07.x1w = 0 ; p07.x1wr = 1 ; p07.x2w = 1 ; p07.x2wr = 1
    p07.y1u = 1 ; p07.y1ur = 1 ; p07.y2u = 1 ; p07.y2ur = 1
    p07.y1v = 1 ; p07.y1vr = 1 ; p07.y2v = 0 ; p07.y2vr = 1
    p07.y1w = 1 ; p07.y1wr = 1 ; p07.y2w = 0 ; p07.y2wr = 1

    p08.x1u = 1 ; p08.x1ur = 1 ; p08.x2u = 1 ; p08.x2ur = 1
    p08.x1v = 0 ; p08.x1vr = 1 ; p08.x2v = 1 ; p08.x2vr = 1
    p08.x1w = 0 ; p08.x1wr = 1 ; p08.x2w = 1 ; p08.x2wr = 1
    p08.y1u = 1 ; p08.y1ur = 1 ; p08.y2u = 1 ; p08.y2ur = 1
    p08.y1v = 1 ; p08.y1vr = 1 ; p08.y2v = 1 ; p08.y2vr = 1
    p08.y1w = 1 ; p08.y1wr = 1 ; p08.y2w = 1 ; p08.y2wr = 1

    p09.x1u = 1 ; p09.x1ur = 1 ; p09.x2u = 1 ; p09.x2ur = 1
    p09.x1v = 0 ; p09.x1vr = 1 ; p09.x2v = 1 ; p09.x2vr = 1
    p09.x1w = 0 ; p09.x1wr = 1 ; p09.x2w = 1 ; p09.x2wr = 1
    p09.y1u = 1 ; p09.y1ur = 1 ; p09.y2u = 1 ; p09.y2ur = 1
    p09.y1v = 0 ; p09.y1vr = 1 ; p09.y2v = 1 ; p09.y2vr = 1
    p09.y1w = 0 ; p09.y1wr = 1 ; p09.y2w = 1 ; p09.y2wr = 1

    # base up
    p10.x1u = 1 ; p10.x1ur = 1 ; p10.x2u = 1 ; p10.x2ur = 1
    p10.x1v = 1 ; p10.x1vr = 1 ; p10.x2v = 1 ; p10.x2vr = 1
    p10.x1w = 1 ; p10.x1wr = 1 ; p10.x2w = 1 ; p10.x2wr = 1
    p10.y1u = 1 ; p10.y1ur = 1 ; p10.y2u = 1 ; p10.y2ur = 1
    p10.y1v = 1 ; p10.y1vr = 1 ; p10.y2v = 1 ; p10.y2vr = 1
    p10.y1w = 1 ; p10.y1wr = 1 ; p10.y2w = 1 ; p10.y2wr = 1

    # flange up
    p11.x1u = 1 ; p11.x1ur = 1 ; p11.x2u = 0 ; p11.x2ur = 1
    p11.x1v = 1 ; p11.x1vr = 1 ; p11.x2v = 0 ; p11.x2vr = 1
    p11.x1w = 1 ; p11.x1wr = 1 ; p11.x2w = 0 ; p11.x2wr = 1
    p11.y1u = 1 ; p11.y1ur = 1 ; p11.y2u = 1 ; p11.y2ur = 1
    p11.y1v = 1 ; p11.y1vr = 1 ; p11.y2v = 1 ; p11.y2vr = 1
    p11.y1w = 1 ; p11.y1wr = 1 ; p11.y2w = 1 ; p11.y2wr = 1

    # base mid
    p12.x1u = 1 ; p12.x1ur = 1 ; p12.x2u = 1 ; p12.x2ur = 1
    p12.x1v = 1 ; p12.x1vr = 1 ; p12.x2v = 1 ; p12.x2vr = 1
    p12.x1w = 1 ; p12.x1wr = 1 ; p12.x2w = 1 ; p12.x2wr = 1
    p12.y1u = 1 ; p12.y1ur = 1 ; p12.y2u = 1 ; p12.y2ur = 1
    p12.y1v = 1 ; p12.y1vr = 1 ; p12.y2v = 1 ; p12.y2vr = 1
    p12.y1w = 1 ; p12.y1wr = 1 ; p12.y2w = 1 ; p12.y2wr = 1

    # flange mid
    p13.x1u = 1 ; p13.x1ur = 1 ; p13.x2u = 1 ; p13.x2ur = 1
    p13.x1v = 1 ; p13.x1vr = 1 ; p13.x2v = 1 ; p13.x2vr = 1
    p13.x1w = 1 ; p13.x1wr = 1 ; p13.x2w = 1 ; p13.x2wr = 1
    p13.y1u = 1 ; p13.y1ur = 1 ; p13.y2u = 1 ; p13.y2ur = 1
    p13.y1v = 1 ; p13.y1vr = 1 ; p13.y2v = 1 ; p13.y2vr = 1
    p13.y1w = 1 ; p13.y1wr = 1 ; p13.y2w = 1 ; p13.y2wr = 1

    # base low
    p14.x1u = 1 ; p14.x1ur = 1 ; p14.x2u = 1 ; p14.x2ur = 1
    p14.x1v = 1 ; p14.x1vr = 1 ; p14.x2v = 1 ; p14.x2vr = 1
    p14.x1w = 1 ; p14.x1wr = 1 ; p14.x2w = 1 ; p14.x2wr = 1
    p14.y1u = 1 ; p14.y1ur = 1 ; p14.y2u = 1 ; p14.y2ur = 1
    p14.y1v = 1 ; p14.y1vr = 1 ; p14.y2v = 1 ; p14.y2vr = 1
    p14.y1w = 1 ; p14.y1wr = 1 ; p14.y2w = 1 ; p14.y2wr = 1

    # flange low
    p15.x1u = 1 ; p15.x1ur = 1 ; p15.x2u = 1 ; p15.x2ur = 1
    p15.x1v = 0 ; p15.x1vr = 1 ; p15.x2v = 1 ; p15.x2vr = 1
    p15.x1w = 0 ; p15.x1wr = 1 ; p15.x2w = 1 ; p15.x2wr = 1
    p15.y1u = 1 ; p15.y1ur = 1 ; p15.y2u = 1 ; p15.y2ur = 1
    p15.y1v = 1 ; p15.y1vr = 1 ; p15.y2v = 1 ; p15.y2vr = 1
    p15.y1w = 1 ; p15.y1wr = 1 ; p15.y2w = 1 ; p15.y2wr = 1

    conn = [
        # skin-skin
        dict(p1=p01, p2=p02, func='SSycte', ycte1=0, ycte2=p02.b),
        dict(p1=p01, p2=p04, func='SSxcte', xcte1=0, xcte2=p04.a),
        dict(p1=p02, p2=p03, func='SSycte', ycte1=0, ycte2=p03.b),
        dict(p1=p02, p2=p05, func='SSxcte', xcte1=0, xcte2=p05.a),
        dict(p1=p03, p2=p06, func='SSxcte', xcte1=0, xcte2=p06.a),
        dict(p1=p04, p2=p05, func='SSycte', ycte1=0, ycte2=p05.b),
        dict(p1=p04, p2=p07, func='SSxcte', xcte1=0, xcte2=p07.a),
        dict(p1=p05, p2=p06, func='SSycte', ycte1=0, ycte2=p06.b),
        dict(p1=p05, p2=p08, func='SSxcte', xcte1=0, xcte2=p08.a),
        dict(p1=p06, p2=p09, func='SSxcte', xcte1=0, xcte2=p09.a),
        dict(p1=p07, p2=p08, func='SSycte', ycte1=0, ycte2=p08.b),
        dict(p1=p08, p2=p09, func='SSycte', ycte1=0, ycte2=p09.b),

        # skin-base
        dict(p1=p02, p2=p10, func='SB'),
        dict(p1=p05, p2=p12, func='SB', has_defect=has_defect), # defect
        dict(p1=p08, p2=p14, func='SB'),

        # base-base
        dict(p1=p10, p2=p12, func='SSxcte', xcte1=0, xcte2=p12.a),
        dict(p1=p12, p2=p14, func='SSxcte', xcte1=0, xcte2=p14.a),

        # base-flange
        dict(p1=p10, p2=p11, func='BFycte', ycte1=p10.b/2., ycte2=0),
        dict(p1=p12, p2=p13, func='BFycte', ycte1=p12.b/2., ycte2=0),
        dict(p1=p14, p2=p15, func='BFycte', ycte1=p14.b/2., ycte2=0),

        # flange-flange
        dict(p1=p11, p2=p13, func='SSxcte', xcte1=0, xcte2=p13.a),
        dict(p1=p13, p2=p15, func='SSxcte', xcte1=0, xcte2=p15.a),
        ]

    panels = [p01, p02, p03, p04, p05, p06, p07, p08, p09,
              p10, p11, p12, p13, p14, p15]
    skin = [p01, p02, p03, p04, p05, p06, p07, p08, p09]

    assy = MultiDomain(panels, conn_method=conn_method)

    size = assy.get_size()

    valid_conn = []
    for connecti in conn:
        if connecti.get('has_defect'): # connecting if there is no defect
            continue
        valid_conn.append(connecti)

    k0 = assy.reduce(assy.calc_kC(valid_conn, silent=True))

    c = None
    if run_static_case:
        if not (Nxx_skin is None and Nxx_base is None and Nxx_flange is None):
            fext = np.zeros(size)
            for p in [p07, p08, p09, p14, p15]:
                if p.Nxx is None:
                    continue
                p.add_distr_load_fixed_x(0, lambda y: p.Nxx, None, None)
                fext[p.col_start: p.col_end] = p.calc_fext(silent=True)

            incs, cs = static(k0, -assy.reduce(fext), silent=True)
            c = assy.expand(cs[0])

        for p in panels:
            p.Nxx = 0.

        kG = assy.reduce(assy.calc_kG(c=c, silent=True))

    else:
        kG = 0.

    kM = assy.reduce(assy.calc_kM(silent=True))

    kA = 0
    for p in skin:
        # TODO the current approach has some hidden settings
        #     a better strategy would be:
        #     - define module aerodynamics
        #     - function calc_kA inside a module piston_theory
        #     - pass piston_theory parameters and compute kA
        kA += p.calc_kA(size=size, row0=p.row_start, col0=p.col_start, silent=True, finalize=False)

    assert np.any(np.isnan(kA.data)) == False
    assert np.any(np.isinf(kA.data)) == False
    kA = assy.reduce(csr_matrix(kA))

    eigvals, eigvecs = freq((k0 + kG + kA), kM, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)

    return assy, eigvals, assy.expand(eigvecs)
