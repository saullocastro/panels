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


# default model of each group of domains
_DEFAULT_MODELS = {'skin': 'cylshell_clpt_sanders', 'blade': 'plate_clpt_gl'}
# models accepted for each group
_ALLOWED_MODELS = {
    'skin': ('cylshell_clpt_sanders', 'cylshell_clpt_gl',
             'cylshell_clpt_tl', 'cylshell_clpt_donnell'),
    'blade': ('plate_clpt_gl', 'plate_clpt_tl', 'plate_clpt_donnell'),
    }


def _models(models):
    r"""The model of each group, ``models`` completed with
    ``_DEFAULT_MODELS``, see :func:`.create_cylinder_blade_stiffened`"""
    models = {} if models is None else dict(models)
    unknown = sorted(set(models) - set(_DEFAULT_MODELS))
    if unknown:
        raise ValueError("unknown keys of models %s, the keys are %s"
                         % (unknown, sorted(_DEFAULT_MODELS)))
    out = dict(_DEFAULT_MODELS, **models)
    for group, model in out.items():
        if model not in _ALLOWED_MODELS[group]:
            raise ValueError("models[%r] must be one of %s, got %r"
                             % (group, _ALLOWED_MODELS[group], model))
    return out


def create_cylinder_blade_stiffened(height, r, stack, stack_blades,
        width_blades, plyt, laminaprop, npanels, m=8, n=8,
        conn_method='null-space', models=None, side='internal'):
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
    ``width_blades[i]`` and stacking sequence ``stack_blades[i]``. The
    models are ``models['blade']`` for the blades, by default
    ``'plate_clpt_gl'``, whose geometric stiffness contains the work of the
    axial load of the blade on its bending in its own plane, the radial
    bending of the stringer, and ``models['skin']`` for the skin panels, by
    default ``'cylshell_clpt_sanders'``, the default of a cylinder (see
    Notes). A skin of ``'cylshell_clpt_gl'`` adds to the geometric
    stiffness the work of the axial load on the slopes `u_{,x}` and
    `v_{,x}` of its in-plane displacements, which lowers the general
    instability loads with few circumferential waves. All the
    panels have the boundary condition flags that remove `u`, `v` and `w`
    at the edges `x = 0` and `x = height` (``x1u = x2u = x1v = x2v = x1w =
    x2w = 0``) and keep all the other ones. Consecutive skin panels are
    connected with ``'SSycte'``, the edge `y = b` of one to the edge `y = 0`
    of the next, the last skin panel to the first one, and each skin panel
    to its blade with ``'BFycte'``, at the edge `y = 0` of the skin panel.
    With ``side='internal'`` the edge `y = 0` of the blade is connected to
    the inner surface of the skin (key ``'face'``, ``'lower'``) and the
    blade stands inside the cylinder, with ``side='external'`` its edge `y`
    = ``width_blades[i]`` is connected to the outer surface (``'upper'``),
    and the blade stands outside: the width of a blade is its height from
    the surface of the skin. The
    assembly contains the skin panels followed by the blades. The
    connections are stored in the assembly, ``md.conn``, and also returned.

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
        The width for each blade (with length = npanels), its height from
        the surface of the skin to its tip, length.
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
    models : dict, optional
        Model of the domains of each group: ``'skin'``, the skin panels,
        ``'cylshell_clpt_sanders'`` (default), ``'cylshell_clpt_gl'``,
        ``'cylshell_clpt_tl'`` or ``'cylshell_clpt_donnell'``, and
        ``'blade'``, the blades, ``'plate_clpt_gl'`` (default),
        ``'plate_clpt_tl'`` or ``'plate_clpt_donnell'``, see
        :mod:`panels.models` and the Notes. A missing key takes its
        default.
    side : str, optional
        ``'internal'``, the default, for blades inside the cylinder, or
        ``'external'`` for blades outside.

    Returns
    -------
    md, conns, info : tuple
        The multidomain assembly, the list of connection dictionaries, the
        list stored in the assembly, ``md.conn``, and a dictionary with
        ``'skin'``, the list
        of the skin panels, ``'blades'``, the list of the blades, ``'ws'``,
        the width of a skin panel, ``2*pi*r/npanels``, and ``'models'``,
        the model of each group, ``models`` completed with the defaults.

    Raises
    ------
    ValueError
        When ``models`` has a key other than ``'skin'`` and ``'blade'`` or a
        model not listed above for its key, when ``side`` is neither
        ``'internal'`` nor ``'external'``, when ``npanels < 2``, or when
        the length of ``stack_blades`` or of ``width_blades`` is not
        ``npanels``.

    Notes
    -----
    The connection ``'BFycte'`` makes the displacements of the blade, whose
    local axes are rotated by 90 degrees about the `x` axis with respect to
    those of the skin, and the rotation about `x` continuous along the line
    `y = 0` of the skin panel, where the blade stands normal to the skin,
    see :meth:`.MultiDomain.get_kC_conn`, at the inner or the outer surface
    of the skin, whose points follow the Kirchhoff-Love kinematics of the
    skin with the Sanders-Koiter term `z v/r` whatever the model of the
    skin, see :mod:`panels.multidomain.connections.kCBFycte_face`: the blade
    does not overlap the skin. The boundary conditions and the
    redundant constraints of the closed skin are those of
    :func:`.create_cylinder`. The local `y` axis of the blade points towards
    the axis of the cylinder (`v_2 = -w_1` along the joint): connected at
    its edge `y = 0` (``ycte2 = 0``) the blade stands on the inner side of
    the skin, connected at its edge `y = b` on the outer side.

    The blades carry their share of the axial load as the membrane stress
    resultant `N_{xx}` of the plate, and the general instability of the
    stiffened cylinder bends them in their own plane: the radial
    displacement `w` of the skin at the joint is the in-plane displacement
    `v` of the blade. The work of `N_{xx}` on this deflection is the term
    `\frac{1}{2} N_{xx} v_{,x}^2` of the Green-Lagrange membrane strain

    .. math::

        \varepsilon_{xx} = u_{,x} + \frac{1}{2} \left( u_{,x}^2 + v_{,x}^2
            + w_{,x}^2 \right)

    of the model ``'plate_clpt_gl'``, whose geometric stiffness matrix has
    the blocks `u`-`u` and `v`-`v` besides the block `w`-`w`. The von
    Kármán strains of ``'plate_clpt_donnell'`` keep `w_{,x}^2/2` only: a
    blade of that model is loaded without destabilizing the radial bending
    of the stringer, and the general instability loads come out too high,
    whereas in the smeared-stiffener theory of Block, Card and Mikulas [1]_
    the total load of the skin and of the stringers, `\bar{N}_x`, enters the
    potential energy of the external forces through `\bar{N}_x w_{,x}^2`
    (their Eq. (9), p. 8). Both models have the same linear constitutive
    stiffness, mass and aerodynamic matrices, so that linear static and
    frequency analyses do not depend on this choice.

    The skin models differ in the geometric stiffness only. Under an axial
    load the Sanders skin, ``'cylshell_clpt_sanders'``, keeps the term
    `\frac{1}{2} N_{xx} w_{,x}^2`, as the Donnell skin, whereas
    ``'cylshell_clpt_gl'`` has the complete Green-Lagrange membrane
    strains, with `\frac{1}{2} N_{xx} (u_{,x}^2 + v_{,x}^2 + w_{,x}^2)`,
    see :mod:`panels.models`. The term `v_{,x}^2` relative to `w_{,x}^2`
    is of the order `1/n^2` for a mode of `n` circumferential waves, so that
    the choice matters for general instability modes with few waves and for
    the bending of the stiffened cylinder as a beam, where the Sanders skin
    overestimates the load (about twice the Euler load of a pinned tube,
    against the Euler load with ``'cylshell_clpt_gl'``). For the cylinder 4
    of Card and Jones [2]_, 60 internal blades, simply supported, critical
    mode of one axial half-wave and 6 circumferential waves, the load with
    the skin ``'cylshell_clpt_gl'`` is 1.4 % below that with
    ``'cylshell_clpt_sanders'`` and 4.1 % below their smeared-stringer
    load `\bar{N}_s` (Table I), and 1.6 % higher with
    ``'cylshell_clpt_donnell'``.

    ``'cylshell_clpt_tl'`` has the membrane strains of
    ``'cylshell_clpt_gl'`` and changes of curvature measured with the
    deformed normal, whose linear part is that of
    ``'cylshell_clpt_sanders'``, see :mod:`panels.models`; likewise
    ``'plate_clpt_tl'`` for the blades with the membrane strains of
    ``'plate_clpt_gl'``. Both have the analytical kernels of the GL model.
    The geometric stiffness of a membrane pre-buckling state, the constant
    `N_{xx}` of :func:`.cylinder_blade_stiffened_compression_lb_Nxx_cte`,
    is then that of the GL models, and so are the buckling loads (cylinder
    4 of Card and Jones [2]_: equal to the last digit). With the static
    pre-buckling state of
    :func:`.cylinder_blade_stiffened_compression_lb_Nxx_from_static` and
    :func:`.cylinder_blade_stiffened_compression_lb_pd_from_static` the
    numerical kernels of the TL models add the work of the bending moments
    of the pre-buckling state on the quadratic part of the changes of
    curvature: -8.7e-8 (relative) for the cylinder 4 of Card and Jones,
    -1.6e-4 and -1.8e-4 for the laminated cylinder of the Examples of
    these two functions. In the non-linear analyses all their strains are
    invariant under finite rigid rotations.

    Only the models of the classical laminated plate theory are accepted:
    those of the shear deformation theories have the rotations `\phi_x,
    \phi_y` as further degrees of freedom, a connection between one of
    them and a model of the classical theory is not implemented (a
    ``NotImplementedError`` of :meth:`.MultiDomain.get_kC_conn`), and the
    boundary condition flags set here, ``x1u`` to ``y2wr``, are those of
    the classical theory only.

    References
    ----------
    .. [1] Block, D. L., Card, M. F., and Mikulas, M. M., Jr., "Buckling of
       Eccentrically Stiffened Orthotropic Cylinders", NASA TN D-2960, 1965.
    .. [2] Card, M. F., and Jones, R. M., "Experimental and Theoretical Results
       for Buckling of Eccentrically Stiffened Cylinders", NASA TN D-3639,
       1966.

    Examples
    --------
    A cylinder of 4 skin panels with 4 blades of 20 mm, the groups of the
    domains and the connections:

    >>> from panels.multidomain import create_cylinder_blade_stiffened
    >>> laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    >>> md, conns, info = create_cylinder_blade_stiffened(height=0.5, r=0.25,
    ...     stack=[0, 90, 0], stack_blades=[[0, 90, 0]]*4,
    ...     width_blades=[0.02]*4, plyt=0.125e-3, laminaprop=laminaprop,
    ...     npanels=4)
    >>> print([p.group for p in md.panels])
    ['skin', 'skin', 'skin', 'skin', 'blade_00', 'blade_01', 'blade_02', 'blade_03']
    >>> print(sorted(set(c['func'] for c in conns)), len(conns),
    ...       md.conn is conns)
    ['BFycte', 'SSycte'] 8 True
    >>> print(info['models'])
    {'skin': 'cylshell_clpt_sanders', 'blade': 'plate_clpt_gl'}

    """
    models = _models(models)
    if side not in ('internal', 'external'):
        raise ValueError("side must be 'internal' or 'external', got %r"
                         % (side, ))
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
                      r=r, model=models['skin'], m=m, n=n, plyt=plyt,
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
                      model=models['blade'], m=m, n=n, plyt=plyt,
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
        ycte2 = 0 if side == 'internal' else blade.b
        face = 'lower' if side == 'internal' else 'upper'
        conns.append(dict(p1=panel, p2=blade, func='BFycte', ycte1=0,
                          ycte2=ycte2, face=face))

    md = MultiDomain(skin + blades, conns, conn_method=conn_method)

    return md, conns, dict(skin=skin, blades=blades, ws=b_skin, models=models)


def cylinder_blade_stiffened_compression_lb_Nxx_cte(height, r, stack, stack_blades,
        width_blades, plyt, laminaprop,
        npanels, Nxxs_skin, Nxxs_blade, m=8, n=8, num_eigvalues=20,
        conn_method='null-space', models=None, side='internal'):
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
    models : dict, optional
        Model of the skin panels, key ``'skin'``, and of the blades, key
        ``'blade'``, see :func:`.create_cylinder_blade_stiffened`.
    side : str, optional
        ``'internal'``, the default, for blades inside the cylinder, or
        ``'external'`` for blades outside, see
        :func:`.create_cylinder_blade_stiffened`.

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
        ``npanels``, or when ``models`` or ``side`` is not accepted by
        :func:`.create_cylinder_blade_stiffened`.

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

    Critical axial load of a cylinder of 5 panels, each with a blade, force per unit length:

    >>> from panels.multidomain import (
    ...     cylinder_blade_stiffened_compression_lb_Nxx_cte)
    >>> laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    >>> Nxxs = [-100.]*5
    >>> md, eigvals, eigvecs = cylinder_blade_stiffened_compression_lb_Nxx_cte(
    ...     height=0.5, r=0.25, stack=[0, 45, -45, 90, -45, 45],
    ...     stack_blades=[[0, 90, 0]*4]*5, width_blades=[0.02]*5,
    ...     plyt=0.125e-3, laminaprop=laminaprop, npanels=5,
    ...     Nxxs_skin=Nxxs, Nxxs_blade=Nxxs, m=8, n=12, num_eigvalues=5)
    >>> round(float(Nxxs[0]*eigvals[0]))
    -51578

    """
    md, conns, info = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n,
            conn_method=conn_method, models=models, side=side)
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

    k0 = md.reduce(md.calc_kC(silent=True))
    kG = md.reduce(md.calc_kG(silent=True))
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)
    return md, eigvals, md.expand(eigvecs)


def cylinder_blade_stiffened_compression_lb_Nxx_from_static(height, r, stack,
        stack_blades, width_blades, plyt, laminaprop, npanels, Nxxs_skin,
        Nxxs_blade, m=8, n=8,
        num_eigvalues=20, conn_method='null-space', models=None,
        side='internal'):
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
    models : dict, optional
        Model of the skin panels, key ``'skin'``, and of the blades, key
        ``'blade'``, see :func:`.create_cylinder_blade_stiffened`.
    side : str, optional
        ``'internal'``, the default, for blades inside the cylinder, or
        ``'external'`` for blades outside, see
        :func:`.create_cylinder_blade_stiffened`.

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
        ``npanels``, or when ``models`` or ``side`` is not accepted by
        :func:`.create_cylinder_blade_stiffened`.

    Notes
    -----
    The pre-buckling state and the linear buckling analysis are those of
    :func:`.cylinder_compression_lb_Nxx_from_static`: ``c`` is the linear
    static solution of the reduced problem, and ``eigvals`` are the
    multipliers `\lambda` of the applied loads in `(\boldsymbol{K}_r + \lambda
    \boldsymbol{K}_{G r}(\boldsymbol{c})) \boldsymbol{\phi}_r = \boldsymbol{0}`.

    Examples
    --------

    Critical axial load of a cylinder of 5 panels, each with a blade, force per unit length:

    >>> from panels.multidomain import (
    ...     cylinder_blade_stiffened_compression_lb_Nxx_from_static)
    >>> laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    >>> Nxxs = [-100.]*5
    >>> md, c, eigvals, eigvecs = cylinder_blade_stiffened_compression_lb_Nxx_from_static(
    ...     height=0.5, r=0.25, stack=[0, 45, -45, 90, -45, 45],
    ...     stack_blades=[[0, 90, 0]*4]*5, width_blades=[0.02]*5,
    ...     plyt=0.125e-3, laminaprop=laminaprop, npanels=5,
    ...     Nxxs_skin=Nxxs, Nxxs_blade=Nxxs, m=8, n=12, num_eigvalues=5)
    >>> round(float(Nxxs[0]*eigvals[0]))
    -40322

    """
    md, conns, info = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n,
            conn_method=conn_method, models=models, side=side)
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

    k0 = md.reduce(md.calc_kC(silent=True))
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
        num_eigvalues=20, ku=1.e6, conn_method='null-space',
        models=None, side='internal'):
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
    models : dict, optional
        Model of the skin panels, key ``'skin'``, and of the blades, key
        ``'blade'``, see :func:`.create_cylinder_blade_stiffened`.
    side : str, optional
        ``'internal'``, the default, for blades inside the cylinder, or
        ``'external'`` for blades outside, see
        :func:`.create_cylinder_blade_stiffened`.

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
        ``npanels``, or when ``models`` or ``side`` is not accepted by
        :func:`.create_cylinder_blade_stiffened`.

    Notes
    -----
    The prescribed displacements are the linear constraints `u(x = height,
    y) = \lambda u_p`, eliminated exactly together with the connections,
    see :meth:`.MultiDomain.get_cp`: the pre-buckling state is `\hat{\boldsymbol{c}}(\lambda) =
    \lambda \hat{\boldsymbol{c}}_1`, with `u = u_p` at `x = height` for `\lambda = 1`, and
    the critical end-shortening of each panel is ``eigvals[0]`` times its
    prescribed value. The modes have `u = 0` at `x = height`, the
    displacement being controlled there. See
    :meth:`.MultiDomain.linear_buckling`.

    Examples
    --------

    Critical end-shortening of a cylinder of 5 panels, each with a blade, length:

    >>> from panels.multidomain import (
    ...     cylinder_blade_stiffened_compression_lb_pd_from_static)
    >>> laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    >>> pds = [-1.e-3]*5
    >>> md, c, eigvals, eigvecs = cylinder_blade_stiffened_compression_lb_pd_from_static(
    ...     height=0.5, r=0.25, stack=[0, 45, -45, 90, -45, 45],
    ...     stack_blades=[[0, 90, 0]*4]*5, width_blades=[0.02]*5,
    ...     plyt=0.125e-3, laminaprop=laminaprop, npanels=5,
    ...     pds_skin=pds, pds_blade=pds, m=8, n=12, num_eigvalues=5)
    >>> round(float(pds[0]*eigvals[0]), 7)
    -0.0004825

    """
    md, conns, info = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n,
            conn_method=conn_method, models=models, side=side)
    if len(pds_skin) != npanels:
        raise ValueError('The length of "pds_skin" must be the same as "npanels"')
    if len(pds_blade) != npanels:
        raise ValueError('The length of "pds_blade" must be the same as "npanels"')
    #NOTE an assembly of the same panels and connections with the
    #     prescribed displacements imposed exactly
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
