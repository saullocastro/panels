r"""Blade-stiffened cylinders made of several domains
(:mod:`panels.multidomain.cylinder_blade_stiffened`)

:func:`.create_cylinder_blade_stiffened` builds a closed cylinder from
``npanels`` curved skin panels, each with a flat blade stiffener, and
:func:`.cylinder_blade_stiffened_compression_lb_Nxx_cte`,
:func:`.cylinder_blade_stiffened_compression_lb_Nxx_from_static` and
:func:`.cylinder_blade_stiffened_compression_lb_pd_from_static` run linear
buckling analyses of it under axial loads or prescribed axial
displacements. All of them are imported in :mod:`panels.multidomain`.

"""
import numpy as np
from scipy.sparse import csr_matrix

from structsolve import lb, static

from panels.shell import Shell
from panels.multidomain import MultiDomain


def create_cylinder_blade_stiffened(height, r, stack, stack_blades,
        width_blades, plyt, laminaprop, npanels, m=8, n=8,
        conn_method='null-space'):
    r"""Create a cylinder multidomain assembly with blade stiffeners

    The multidomain assembly looks like::


        B                               A
         _______ _______ _______ _______
        |       |       |       |       |
        |       |       |       |       |
        |       |       |       |       |
        |  p04  |  p03  |  p02  |  p01  |
        |       |       |       |       |
        |       |       |       |       |
        |_______|_______|_______|_______|

        Blade   Blade   Blade   Blade
        04      03      02      01
         _       _       _       _
        | |     | |     | |     | |
        | |     | |     | |     | |
        | |     | |     | |     | |
        | |     | |     | |     | |
        |_|     |_|     |_|     |_|
                                               x
                                             /\
                                             |
                                             |
                                   y  <-------

    where edges ``A`` and ``B`` are connected to close de cylinder.

    The skin panels are :class:`.Shell` objects of the group ``'skin'``,
    each with length ``height``, width ``2*pi*r/npanels``, radius ``r``,
    ``x0 = 0`` and ``y0`` equal to ``i`` times the width for the ``i``-th
    panel. The ``i``-th blade is a flat :class:`.Shell` (no radius) of the
    group ``'blade_%02d' % i``, with length ``height``, width
    ``width_blades[i]`` and stacking sequence ``stack_blades[i]``. All the
    panels have the boundary condition flags that remove `u`, `v` and `w`
    at the edges `x = 0` and `x = height` (``x1u = x2u = x1v = x2v = x1w =
    x2w = 0``) and keep all the other ones. Consecutive skin panels are
    connected with ``'SSycte'``, the edge `y = b` of one to the edge `y = 0`
    of the next, the last skin panel to the first one, and each skin panel
    to its blade with ``'BFycte'``, at the edge `y = 0` of both. The
    assembly contains the skin panels followed by the blades. The
    connections are returned and not stored in the assembly, which is
    created with ``conn=None``: pass them to :meth:`.MultiDomain.calc_kC`.

    The code of this function is illustrative of how it constructs the
    multidomain assembly:

    .. literalinclude:: ../../panels/multidomain/cylinder_blade_stiffened.py
        :pyobject: create_cylinder_blade_stiffened


    Parameters
    ----------

    height : float
        Cylinder height (along `x`), length.
    r : float
        Cylinder radius, length.
    stack : array-like
        Stacking sequence for the cylinder, the angle of each ply in
        degrees.
    stack_blades : list of array-like
        The stacking sequence for each blade (with length = npanels), the
        angles in degrees.
    width_blades : array-like
        The width for each blade (with length = npanels), length.
    plyt : float
        Ply thickness (assumed unique for the whole structure), length.
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
        When ``npanels < 2``, or when the length of ``stack_blades`` or of
        ``width_blades`` is not ``npanels``.

    Notes
    -----
    The connection ``'BFycte'`` makes the displacements of the blade, whose
    local axes are rotated by 90 degrees about the `x` axis with respect to
    those of the skin, and the rotation about `x` continuous along the line
    `y = 0` of the skin panel, where the blade stands normal to the skin,
    see :meth:`.MultiDomain.get_kC_conn`. The boundary conditions and the
    redundant constraints of the closed skin are those of
    :func:`.create_cylinder`.

    """
    if npanels < 2:
        raise ValueError('At least two panels are needed')
    if len(stack_blades) != npanels:
        raise ValueError('stack_blades must have length = npanels')
    if len(width_blades) != npanels:
        raise ValueError('width_blades must have length = npanels')
    skin = []
    blades = []
    perimiter = 2*np.pi*r
    b_skin = perimiter / npanels
    for i in range(npanels):
        y0 = i*b_skin
        panel = Shell(group='skin', x0=0, y0=y0, a=height, b=b_skin,
                      r=r, m=m, n=n, plyt=plyt,
                      stack=stack, laminaprop=laminaprop,
                      x1u=0, x1ur=1, x2u=0, x2ur=1,
                      x1v=0, x1vr=1, x2v=0, x2vr=1,
                      x1w=0, x1wr=1, x2w=0, x2wr=1,
                      y1u=1, y1ur=1, y2u=1, y2ur=1,
                      y1v=1, y1vr=1, y2v=1, y2vr=1,
                      y1w=1, y1wr=1, y2w=1, y2wr=1)
        skin.append(panel)
    for i, panel in enumerate(skin):
        y0 = i*b_skin
        blade_name = 'blade_%02d' % i
        blade = Shell(group=blade_name, x0=0, y0=y0, a=height, b=width_blades[i],
                      m=m, n=n, plyt=plyt,
                      stack=stack_blades[i], laminaprop=laminaprop,
                      x1u=0, x1ur=1, x2u=0, x2ur=1,
                      x1v=0, x1vr=1, x2v=0, x2vr=1,
                      x1w=0, x1wr=1, x2w=0, x2wr=1,
                      y1u=1, y1ur=1, y2u=1, y2ur=1,
                      y1v=1, y1vr=1, y2v=1, y2vr=1,
                      y1w=1, y1wr=1, y2w=1, y2wr=1)
        blades.append(blade)

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

    for panel, blade in zip(skin, blades):
        conns.append(dict(p1=panel, p2=blade, func='BFycte', ycte1=0, ycte2=0))

    md = MultiDomain(skin + blades, conn_method=conn_method)

    return md, conns


def cylinder_blade_stiffened_compression_lb_Nxx_cte(height, r, stack, stack_blades,
        width_blades, plyt, laminaprop,
        npanels, Nxxs_skin, Nxxs_blade, m=8, n=8, num_eigvalues=20,
        conn_method='null-space'):
    r"""Linear buckling analysis with a constant Nxx for each panel

    See :func:`.create_cylinder_blade_stiffened` for most parameters.

    The assembly and its connections come from
    :func:`.create_cylinder_blade_stiffened`. For every panel ``x2u`` is set
    to ``1`` and the attribute ``Nxx`` to the next value of ``Nxxs_skin``
    for the panels whose group contains ``'skin'``, or of ``Nxxs_blade`` for
    those whose group contains ``'blade'``, in the order of the panels in
    the assembly. The constitutive stiffness matrix,
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
    stack : array-like
        Stacking sequence for the cylinder, the angle of each ply in
        degrees.
    stack_blades : list of array-like
        The stacking sequence for each blade, with length ``npanels``, the
        angles in degrees.
    width_blades : array-like
        The width for each blade, with length ``npanels``, length.
    plyt : float
        Ply thickness, the same for the whole structure, length.
    laminaprop : list or tuple
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13},
        G_{23}`. The moduli are forces per unit area, `\nu_{12}` is
        dimensionless.
    npanels : int
        The number of skin panels along the perimeter of the cylinder.
    Nxxs_skin : list
        A Nxx for each skin panel, force per unit length, with length
        ``npanels``.
    Nxxs_blade : list
        A Nxx for each blade stiffener, force per unit length, with length
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
        When the length of ``Nxxs_skin`` or of ``Nxxs_blade`` is not
        ``npanels``.

    Notes
    -----
    The linear buckling analysis is that of
    :func:`.cylinder_compression_lb_Nxx_cte`, with the membrane stress state
    `N_{xx}` assumed constant over each panel. The values of ``Nxxs_skin``
    and ``Nxxs_blade`` are independent: they represent the pre-buckling
    state of a cylinder with equal axial strains in the skin and in the
    blades only when they are given approximately in the ratio of the
    membrane stiffnesses of the laminates, whereas the static analysis of
    :func:`.cylinder_blade_stiffened_compression_lb_Nxx_from_static` gives
    a compatible pre-buckling state.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../tests/multidomain/test_cylinder_blade_stiffened.py
        :pyobject: test_cylinder_blade_stiffened_compression_lb_Nxx_cte

    """
    md, conns = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n,
            conn_method=conn_method)
    if len(Nxxs_skin) != npanels:
        raise ValueError('The length of "Nxxs_skin" must be the same as "npanels"')
    if len(Nxxs_blade) != npanels:
        raise ValueError('The length of "Nxxs_blade" must be the same as "npanels"')
    i_skin = -1
    i_blade = -1
    for p in md.panels:
        p.x2u = 1
        if 'skin' in p.group:
            i_skin += 1
            p.Nxx = Nxxs_skin[i_skin]
        elif 'blade' in p.group:
            i_blade += 1
            p.Nxx = Nxxs_blade[i_blade]

    k0 = md.reduce(md.calc_kC(conns, silent=True))
    kG = md.reduce(md.calc_kG(silent=True))
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)
    return md, eigvals, md.expand(eigvecs)


def cylinder_blade_stiffened_compression_lb_Nxx_from_static(height, r, stack,
        stack_blades, width_blades, plyt, laminaprop, npanels, Nxxs_skin,
        Nxxs_blade, m=8, n=8,
        num_eigvalues=20, conn_method='null-space'):
    r"""Linear buckling analysis with a Nxx calculated using static analysis

    See :func:`.create_cylinder_blade_stiffened` for most parameters.

    The assembly and its connections come from
    :func:`.create_cylinder_blade_stiffened`. For every panel ``x2u`` is set
    to ``1`` and a distributed force along `x`, constant along `y`, is added
    at its edge `x = height` with :meth:`.Shell.add_distr_load_fixed_x`,
    taken from ``Nxxs_skin`` for the panels whose group contains ``'skin'``
    and from ``Nxxs_blade`` for those whose group contains ``'blade'``; the
    loads stay in the panels of the returned assembly. A linear static
    analysis, ``structsolve.static`` with the reduced
    :meth:`.MultiDomain.calc_kC` and :meth:`.MultiDomain.calc_fext`, gives
    the pre-buckling state ``c``, with which the geometric stiffness matrix
    :meth:`.MultiDomain.calc_kG` is computed. The reduced matrices are
    passed to ``structsolve.lb`` with ``tol=0`` and ``sparse_solver=True``,
    and the eigenvectors are expanded with :meth:`.MultiDomain.expand`.

    Parameters
    ----------
    height : float
        Cylinder height (along `x`), length.
    r : float
        Cylinder radius, length.
    stack : array-like
        Stacking sequence for the cylinder, the angle of each ply in
        degrees.
    stack_blades : list of array-like
        The stacking sequence for each blade, with length ``npanels``, the
        angles in degrees.
    width_blades : array-like
        The width for each blade, with length ``npanels``, length.
    plyt : float
        Ply thickness, the same for the whole structure, length.
    laminaprop : list or tuple
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13},
        G_{23}`. The moduli are forces per unit area, `\nu_{12}` is
        dimensionless.
    npanels : int
        The number of skin panels along the perimeter of the cylinder.
    Nxxs_skin : list
        A Nxx for each skin panel, force per unit length, with length
        ``npanels``.
    Nxxs_blade : list
        A Nxx for each blade stiffener, force per unit length, with length
        ``npanels``. The skin panels and the blades of the assembly
        receive, in their order in ``md.panels``, the successive values of
        ``Nxxs_skin`` and ``Nxxs_blade``.
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
        When the length of ``Nxxs_skin`` or of ``Nxxs_blade`` is not
        ``npanels``.

    Notes
    -----
    The pre-buckling state and the linear buckling analysis are those of
    :func:`.cylinder_compression_lb_Nxx_from_static`: ``c`` is the linear
    static solution of the reduced problem, and ``eigvals`` are the
    multipliers `\lambda` of the applied loads in `([K_r] + \lambda
    [K_{G r}(c)]) \{\phi_r\} = \{0\}`.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../tests/multidomain/test_cylinder_blade_stiffened.py
        :pyobject: test_cylinder_blade_stiffened_compression_lb_Nxx_from_static

    """
    md, conns = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n,
            conn_method=conn_method)
    if len(Nxxs_skin) != npanels:
        raise ValueError('The length of "Nxxs_skin" must be the same as "npanels"')
    if len(Nxxs_blade) != npanels:
        raise ValueError('The length of "Nxxs_blade" must be the same as "npanels"')
    i_skin = -1
    i_blade = -1
    for p in md.panels:
        p.x2u = 1
        if 'skin' in p.group:
            i_skin += 1
            Nxx = Nxxs_skin[i_skin]
        elif 'blade' in p.group:
            i_blade += 1
            Nxx = Nxxs_blade[i_blade]
        #NOTE the value is bound when the function is created, the load
        #     functions are evaluated after the loop
        p.add_distr_load_fixed_x(p.a, lambda y, Nxx=Nxx: Nxx, None, None)

    fext = md.calc_fext(silent=True)

    k0 = md.reduce(md.calc_kC(conns, silent=True))
    incs, cs = static(k0, md.reduce(fext), silent=True)
    c = md.expand(cs[0])
    kG = md.reduce(md.calc_kG(c=c, silent=True))

    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)
    eigvecs = md.expand(eigvecs)

    return md, c, eigvals, eigvecs


def cylinder_blade_stiffened_compression_lb_pd_from_static(height, r, stack,
        stack_blades, width_blades, plyt, laminaprop, npanels, pds_skin,
        pds_blade, m=8, n=8,
        num_eigvalues=20, ku=1.e6, conn_method='null-space'):
    r"""Linear buckling analysis with prescribed displacement

    See :func:`.create_cylinder_blade_stiffened` for most parameters.

    The panels and the connections come from
    :func:`.create_cylinder_blade_stiffened`, and the returned assembly is
    a :class:`.MultiDomain` of these panels, with the connections stored in
    it and ``pd_method='null-space'``, such that the prescribed
    displacements are imposed exactly. For every panel ``x2u`` is set to
    ``1`` and the displacement `u`, constant along `y`, is prescribed at
    its edge `x = height` with :meth:`.Shell.add_distr_pd_fixed_x` and
    ``cte=False``, i.e. multiplied by the load factor, taken from
    ``pds_skin`` for the panels whose group contains ``'skin'`` and from
    ``pds_blade`` for those whose group contains ``'blade'``; the
    prescriptions stay in the panels of the returned assembly. The linear
    buckling analysis is :meth:`.MultiDomain.linear_buckling`, with
    ``tol=0`` and ``sparse_solver=True`` passed to ``structsolve.lb``, and
    the pre-buckling state ``c`` is :meth:`.MultiDomain.linear_static` for
    a load factor of 1, which has `u` equal to the prescribed values at `x
    = height`.

    Parameters
    ----------
    height : float
        Cylinder height (along `x`), length.
    r : float
        Cylinder radius, length.
    stack : array-like
        Stacking sequence for the cylinder, the angle of each ply in
        degrees.
    stack_blades : list of array-like
        The stacking sequence for each blade, with length ``npanels``, the
        angles in degrees.
    width_blades : array-like
        The width for each blade, with length ``npanels``, length.
    plyt : float
        Ply thickness, the same for the whole structure, length.
    laminaprop : list or tuple
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13},
        G_{23}`. The moduli are forces per unit area, `\nu_{12}` is
        dimensionless.
    npanels : int
        The number of skin panels along the perimeter of the cylinder.
    pds_skin : list
        A prescribed axial compression displacement for each skin panel,
        length, with length ``npanels``.
    pds_blade : list
        A prescribed axial compression displacement for each blade stiffener,
        length, with length ``npanels``. The skin panels and the blades of
        the assembly receive, in their order in ``md.panels``, the
        successive values of ``pds_skin`` and ``pds_blade``.
    m, n : int, optional
        Number of approximation terms for each panel. Default ``8``.
    num_eigvalues : int, optional
        Number of eigenvalues to be extracted. Default ``20``.
    ku : float, optional
        Penalty stiffness stored with the prescribed displacements, force
        per unit area, see :meth:`.Shell.add_distr_pd_fixed_x`. With
        ``pd_method='null-space'`` its value does not enter the analysis,
        it must only be neither ``None`` nor zero, which tells that `u` is
        prescribed. Default ``1.e6``.
    conn_method : str, optional
        How the connections are imposed, ``'null-space'``, the default, or
        ``'penalty'``, see :class:`.MultiDomain`.

    Returns
    -------
    md, c, eigvals, eigvecs : tuple
        Assembly, static results, eigenvalues and eigenvectors. ``c`` are
        the Ritz constants of the assembly, 1-D array of size
        ``md.get_size()``, ``eigvals`` the critical load factors, i.e. the
        multipliers of the prescribed displacements, dimensionless, and
        ``eigvecs`` has one eigenvector of the assembly per column.

    Raises
    ------
    ValueError
        When the length of ``pds_skin`` or of ``pds_blade`` is not
        ``npanels``.

    Notes
    -----
    The prescribed displacements are the linear constraints `u(x = height,
    y) = \lambda u_p`, eliminated exactly together with the connections,
    see :ref:`null_space_pd`: the pre-buckling state is `\{c(\lambda)\} =
    \lambda \{c\}`, with `u = u_p` at `x = height` for `\lambda = 1`, and
    the critical end-shortening of each panel is ``eigvals[0]`` times its
    prescribed value. The modes have `u = 0` at `x = height`, the
    displacement being controlled there. See
    :meth:`.MultiDomain.linear_buckling`.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../tests/multidomain/test_cylinder_blade_stiffened.py
        :pyobject: test_cylinder_blade_stiffened_compression_lb_pd_from_static

    """
    md, conns = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n,
            conn_method=conn_method)
    if len(pds_skin) != npanels:
        raise ValueError('The length of "pds_skin" must be the same as "npanels"')
    if len(pds_blade) != npanels:
        raise ValueError('The length of "pds_blade" must be the same as "npanels"')
    #NOTE the prescribed displacements imposed exactly, the connections
    #     stored in the assembly, used by linear_static and linear_buckling
    md = MultiDomain(md.panels, conns, conn_method=conn_method,
                     pd_method='null-space')
    i_skin = -1
    i_blade = -1
    for p in md.panels:
        p.x2u = 1
        if 'skin' in p.group:
            i_skin += 1
            pd = pds_skin[i_skin]
        elif 'blade' in p.group:
            i_blade += 1
            pd = pds_blade[i_blade]
        #NOTE the value is bound when the function is created, the
        #     functions are evaluated after the loop
        p.add_distr_pd_fixed_x(p.a, ku, None, None, lambda y, pd=pd: pd,
                               None, None, cte=False)

    eigvals, eigvecs = md.linear_buckling(num_eigvalues=num_eigvalues,
            silent=True, tol=0, sparse_solver=True, num_eigvalues_print=5)
    c = md.linear_static(lbd=1., silent=True)

    return md, c, eigvals, eigvecs
