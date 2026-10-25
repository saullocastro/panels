r"""Cylinders made of several domains (:mod:`panels.multidomain.cylinder`)

:func:`.create_cylinder` builds a closed cylinder from ``npanels`` curved
panels connected along their longitudinal edges, and
:func:`.cylinder_compression_lb_Nxx_cte` and
:func:`.cylinder_compression_lb_Nxx_from_static` run linear buckling
analyses of it under axial loads. All of them are imported in
:mod:`panels.multidomain`.

"""
import numpy as np
from scipy.sparse import csr_matrix

from structsolve import lb, static, Analysis

from panels.shell import Shell
from panels.multidomain import MultiDomain


def create_cylinder(height, r, stack, plyt, laminaprop,
        npanels, m=8, n=8, conn_method='null-space'):
    r"""Create a cylinder multidomain assembly

    The multidomain assembly looks like::


        B                             A
         _______ _______ _______ _______
        |       |       |       |       |
        |       |       |       |       |
        |       |       |       |       |
        |  p04  |  p03  |  p02  |  p01  |
        |       |       |       |       |    /\  x
        |       |       |       |       |    |
        |_______|_______|_______|_______|    |
                                             |
                                   y  <-------

    where edges ``A`` and ``B`` are connected to close the cylinder.

    The panels are :class:`.Shell` objects of the group ``'skin'``, each
    with length ``height``, width ``2*pi*r/npanels``, radius ``r``, ``x0 =
    0`` and ``y0`` equal to ``i`` times the width for the ``i``-th panel.
    The boundary condition flags remove `u`, `v` and `w` at the edges `x =
    0` and `x = height` (``x1u = x2u = x1v = x2v = x1w = x2w = 0``) and keep
    all the other ones. Consecutive panels are connected with ``'SSycte'``,
    the edge `y = b` of one to the edge `y = 0` of the next, and the last
    panel is connected to the first one to close the cylinder. The
    connections are returned and not stored in the assembly, which is
    created with ``conn=None``: pass them to :meth:`.MultiDomain.calc_kC`.

    The code of this function is illustrative of how it constructs the
    multidomain assembly:

    .. literalinclude:: ../../panels/multidomain/cylinder.py
        :pyobject: create_cylinder


    Parameters
    ----------

    height : float
        Cylinder height (along `x`), length.
    r : float
        Cylinder radius, length.
    stack : list or tuple
        Stacking sequence for the cylinder, the angle of each ply in
        degrees.
    plyt : float
        Ply thickness, length.
    laminaprop : list or tuple
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13}, G_{23}`.
        The moduli are forces per unit area, `\nu_{12}` is dimensionless.
    npanels : int
        The number of panels the cylinder perimiter.
    m, n : int, optional
        Number of approximation terms for each panel. Default ``8``.
    conn_method : str, optional
        How the connections are imposed, ``'null-space'``, the default, or
        ``'penalty'``, see :class:`.MultiDomain`.

    Returns
    -------
    md, conns : tuple
        A tuple containing the multidomain assembly and the default connectivity
        list of dictionaries.

    Raises
    ------
    ValueError
        When ``npanels < 2``.

    Notes
    -----
    The boundary condition flags remove the terms of the approximation
    functions of `u`, `v` and `w` at the ends, which keep the rotations
    free: the ends are simply supported with the axial displacement
    restrained. The constraints of the last connection, which closes the
    cylinder, are in part linear combinations of those of the other
    connections around the circumference; the null-space method finds
    these redundant constraints and drops them, see
    :meth:`.MultiDomain.get_Z`.

    """
    if npanels < 2:
        raise ValueError('At least two panels are needed')
    skin = []
    perimiter = 2*np.pi*r
    b_skin = perimiter / npanels
    for i in range(npanels):
        y0 = i*b_skin
        panel = Shell(group='skin', x0=0, y0=y0, a=height, b=b_skin,
            r=r, m=m, n=n, plyt=plyt, stack=stack, laminaprop=laminaprop,
            x1u=0, x1ur=1, x2u=0, x2ur=1,
            x1v=0, x1vr=1, x2v=0, x2vr=1,
            x1w=0, x1wr=1, x2w=0, x2wr=1,
            y1u=1, y1ur=1, y2u=1, y2ur=1,
            y1v=1, y1vr=1, y2v=1, y2vr=1,
            y1w=1, y1wr=1, y2w=1, y2wr=1)
        skin.append(panel)
    conns = []
    skin_loop = skin + [skin[0]]
    for i in range(len(skin)):
        if i != len(skin) - 1:
            p01 = skin_loop[i]
            p02 = skin_loop[i+1]
            conns.append(dict(p1=p01, p2=p02, func='SSycte', ycte1=p01.b, ycte2=0))
        else:
            p01 = skin_loop[i+1]
            p02 = skin_loop[i]
            conns.append(dict(p1=p01, p2=p02, func='SSycte', ycte1=0, ycte2=p02.b))
    md = MultiDomain(skin, conn_method=conn_method)

    return md, conns


def cylinder_compression_lb_Nxx_cte(height, r, stack, plyt, laminaprop,
        npanels, Nxxs, m=8, n=8, num_eigvalues=20, conn_method='null-space'):
    r"""Linear buckling analysis with a constant Nxx for each panel

    See :func:`.create_cylinder` for most parameters.

    The cylinder has the simply supported edges SS1 at the bottom, `x = 0`,
    and SS2 at the top, `x = height`, in the notation of Almroth [1]_: the
    top edge is free to move in the axial direction, see the notes.

    The assembly and its connections come from :func:`.create_cylinder`.
    The attribute ``Nxx`` of the ``i``-th panel is set to ``Nxxs[i]`` and
    ``x2u`` to ``1`` for every panel. The constitutive stiffness matrix,
    :meth:`.MultiDomain.calc_kC` with the connections, and the geometric
    stiffness matrix, :meth:`.MultiDomain.calc_kG`, are reduced with
    :meth:`.MultiDomain.reduce` and passed to ``structsolve.lb`` with
    ``tol=0`` and ``sparse_solver=True``. The eigenvectors are expanded with
    :meth:`.MultiDomain.expand`.

    Parameters
    ----------
    height : float
        Cylinder height (along `x`), length.
    r : float
        Cylinder radius, length.
    stack : list or tuple
        Stacking sequence for the cylinder, the angle of each ply in
        degrees.
    plyt : float
        Ply thickness, length.
    laminaprop : list or tuple
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13},
        G_{23}`. The moduli are forces per unit area, `\nu_{12}` is
        dimensionless.
    npanels : int
        The number of panels along the perimeter of the cylinder.
    Nxxs : list
        A Nxx for each panel, force per unit length, with length
        ``npanels``.
    m, n : int, optional
        Number of approximation terms for each panel. Default ``8``.
    num_eigvalues : int, optional
        Number of eigenvalues to be extracted. Default ``20``.
    conn_method : str, optional
        How the connections are imposed, ``'null-space'``, the default, or
        ``'penalty'``, see :class:`.MultiDomain`.


    Returns
    -------
    md, eigvals, eigvecs : tuple
        Assembly, eigenvalues and eigenvectors. ``eigvals`` are the load
        multipliers returned by ``structsolve.lb``, dimensionless, and
        ``eigvecs`` has one eigenvector of the assembly per column, with
        ``md.get_size()`` rows.

    Raises
    ------
    ValueError
        When the length of ``Nxxs`` is not ``npanels``.

    Notes
    -----
    The linear buckling analysis solves

    .. math::

        \left([K_r] + \lambda [K_{G r}]\right) \{\phi_r\} = \{0\}

    with `[K_r]` and `[K_{G r}]` the reduced constitutive and geometric
    stiffness matrices, see :meth:`.MultiDomain.reduce`, the latter computed
    for the membrane stress state `N_{xx}` assumed constant over each panel,
    without the bending and the redistribution of the stresses of the
    pre-buckling state near the restrained ends, see
    :func:`.cylinder_compression_lb_Nxx_from_static`. ``eigvals`` are the
    multipliers `\lambda` of ``Nxxs``, and the modes are `\{\phi\} = [Z]
    \{\phi_r\}`.

    The simply supported edges of Almroth [1]_, p. 5, all have `w = 0` and
    `M_{xx} = 0` (`w_{,xx} = 0`) and differ in the in-plane conditions. With
    the boundary condition flags of the edge `x = 0` of :class:`.Shell`
    (``x2u``, ``x2v``, ``x2w`` and ``x2wr`` at `x = a`), a flag ``0``
    removing the displacement and ``1`` leaving the natural condition:

    ======  ============================  =================================
    Edge    Conditions                    Flags
    ======  ============================  =================================
    SS1     `u = 0`, `v = 0`              ``x1u=0, x1v=0, x1w=0, x1wr=1``
    SS2     `N_{xx} = 0`, `v = 0`         ``x1u=1, x1v=0, x1w=0, x1wr=1``
    SS3     `u = 0`, `N_{xy} = 0`         ``x1u=0, x1v=1, x1w=0, x1wr=1``
    SS4     `N_{xx} = 0`, `N_{xy} = 0`    ``x1u=1, x1v=1, x1w=0, x1wr=1``
    ======  ============================  =================================

    :func:`.create_cylinder` gives SS1 at both edges, and ``x2u = 1``
    makes the top edge SS2. Almroth denotes these cases S1 to S4, and C1 to
    C4 those with `w_{,x} = 0` instead of `w_{,xx} = 0`.

    References
    ----------
    .. [1] Almroth, B. O., Influence of edge conditions on the stability of
       axially compressed cylindrical shells, NASA CR-161, 1965.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../tests/multidomain/test_cylinder.py
        :pyobject: test_cylinder_compression_lb_Nxx_cte

    """
    md, conns = create_cylinder(height=height, r=r, stack=stack, plyt=plyt,
            laminaprop=laminaprop, npanels=npanels, m=m, n=n,
            conn_method=conn_method)
    if len(Nxxs) != npanels:
        raise ValueError('The length of "Nxxs" must be the same as "npanels"')
    for i, p in enumerate(md.panels):
        p.Nxx = Nxxs[i]
        p.x2u = 1

    k0 = md.reduce(md.calc_kC(conns, silent=True))
    kG = md.reduce(md.calc_kG(silent=True))
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)
    return md, eigvals, md.expand(eigvecs)


def cylinder_compression_lb_Nxx_from_static(height, r, stack, plyt, laminaprop,
        npanels, Nxxs, m=8, n=8, num_eigvalues=20, conn_method='null-space'):
    r"""Linear buckling analysis with a Nxx calculated using static analysis

    See :func:`.create_cylinder` for most parameters.

    The cylinder has the simply supported edges SS1 at the bottom and SS2 at
    the top, see :func:`.cylinder_compression_lb_Nxx_cte`: the top edge is
    free to move in the axial direction.

    The assembly and its connections come from :func:`.create_cylinder`.
    Each panel receives a distributed force along `x` at its edge `x =
    height`, constant along `y`, with :meth:`.Shell.add_distr_load_fixed_x`,
    and ``x2u`` is set to ``1``; the loads stay in the panels of the
    returned assembly. A linear static analysis, ``structsolve.static``
    with the reduced :meth:`.MultiDomain.calc_kC` and
    :meth:`.MultiDomain.calc_fext`, gives the pre-buckling state ``c``, with
    which the geometric stiffness matrix :meth:`.MultiDomain.calc_kG` is
    computed, with its default ``silent=False``, so that its log messages
    are printed. The reduced matrices are passed to ``structsolve.lb`` with
    ``tol=0`` and ``sparse_solver=True``, and the eigenvectors are expanded
    with :meth:`.MultiDomain.expand`.

    Parameters
    ----------
    height : float
        Cylinder height (along `x`), length.
    r : float
        Cylinder radius, length.
    stack : list or tuple
        Stacking sequence for the cylinder, the angle of each ply in
        degrees.
    plyt : float
        Ply thickness, length.
    laminaprop : list or tuple
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13},
        G_{23}`. The moduli are forces per unit area, `\nu_{12}` is
        dimensionless.
    npanels : int
        The number of panels along the perimeter of the cylinder.
    Nxxs : list
        A Nxx for each panel, force per unit length, with length
        ``npanels``, the panel ``i`` of the assembly receiving
        ``Nxxs[i]``.
    m, n : int, optional
        Number of approximation terms for each panel. Default ``8``.
    num_eigvalues : int, optional
        Number of eigenvalues to be extracted. Default ``20``.
    conn_method : str, optional
        How the connections are imposed, ``'null-space'``, the default, or
        ``'penalty'``, see :class:`.MultiDomain`.

    Returns
    -------
    md, c, eigvals, eigvecs : tuple
        Assembly, static results, eigenvalues and eigenvectors. ``c`` are
        the Ritz constants of the assembly, 1-D array of size
        ``md.get_size()``, ``eigvals`` the load multipliers returned by
        ``structsolve.lb``, dimensionless, and ``eigvecs`` has one
        eigenvector of the assembly per column.

    Raises
    ------
    ValueError
        When the length of ``Nxxs`` is not ``npanels``.

    Notes
    -----
    The pre-buckling state is the linear static solution `\{\bar{c}\} = [Z]
    \{\bar{c}_r\}`, with `[K_r] \{\bar{c}_r\} = [Z]^\top \{F_{ext}\}`, which includes
    the bending of the shell near the ends, where the radial displacement
    is restrained, and the linear buckling analysis solves

    .. math::

        \left([K_r] + \lambda [K_{G r}(\bar{c})]\right) \{\phi_r\} = \{0\}

    with the reduced geometric stiffness matrix of the linear membrane
    stresses of `\{\bar{c}\}`, see :meth:`.Shell.calc_kG`, which is linear in
    `\{\bar{c}\}`, such that ``eigvals`` are the multipliers `\lambda` of the
    applied loads. For the boundary conditions SS1 and SS2 see
    :func:`.cylinder_compression_lb_Nxx_cte`.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../tests/multidomain/test_cylinder.py
        :pyobject: test_cylinder_compression_lb_Nxx_from_static

    """
    md, conns = create_cylinder(height=height, r=r, stack=stack, plyt=plyt,
            laminaprop=laminaprop, npanels=npanels, m=m, n=n,
            conn_method=conn_method)
    if len(Nxxs) != npanels:
        raise ValueError('The length of "Nxxs" must be the same as "npanels"')

    for i, p in enumerate(md.panels):
        #NOTE the value is bound when the function is created, the load
        #     functions are evaluated after the loop
        p.add_distr_load_fixed_x(p.a, lambda y, Nxx=Nxxs[i]: Nxx, None, None)
        p.x2u = 1

    fext = md.calc_fext(silent=True)

    k0 = md.reduce(md.calc_kC(conns))
    incs, cs = static(k0, md.reduce(fext), silent=True)
    c = md.expand(cs[0])
    kG = md.reduce(md.calc_kG(c=c))

    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)
    eigvecs = md.expand(eigvecs)

    return md, c, eigvals, eigvecs

