import numpy as np
from scipy.sparse import csr_matrix

from structsolve import lb, static

from panels.shell import Shell
from panels.multidomain import MultiDomain


def create_cylinder_blade_stiffened(height, r, stack, stack_blades,
        width_blades, plyt, laminaprop, npanels, m=8, n=8):
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


    Parameters
    ----------

    height : float
        Cylinder height (along `x`).
    r : float
        Cylinder radius.
    stack : array-like
        Stacking sequence for the cylinder.
    stack_blades : list of array-like
        The stacking sequence for each blade (with length = npanels).
    width_blades : array-like
        The width for each blade (with length = npanels).
    plyt : float
        Ply thickness (assumed unique for the whole structure).
    laminaprop : list or tuple
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13}, G_{23}`.
    npanels : int
        The number of panels the cylinder perimiter.
    m, n : int, optional
        Number of approximation terms for each panel.

    Returns
    -------
    md, conns : tuple
        A tuple containing the multidomain assembly and the default connectivity
        list of dictionaries.

    Notes
    -----
    The code of this function is illustrative of how it constructs the
    multidomain assembly:

    .. literalinclude:: ../../panels/multidomain/cylinder_blade_stiffened.py
        :pyobject: create_cylinder_blade_stiffened

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

    md = MultiDomain(skin + blades)

    return md, conns


def cylinder_blade_stiffened_compression_lb_Nxx_cte(height, r, stack, stack_blades,
        width_blades, plyt, laminaprop,
        npanels, Nxxs_skin, Nxxs_blade, m=8, n=8, num_eigvalues=20):
    """Linear buckling analysis with a constant Nxx for each panel

    See :func:`.create_cylinder_blade_stiffened` for most parameters.

    Parameters
    ----------
    Nxxs_skin : list
        A Nxx for each skin panel.
    Nxxs_blade : list
        A Nxx for each blade stiffener.
    num_eigvalues : int
        Number of eigenvalues to be extracted.

    Returns
    -------
    md, eigvals, eigvecs : tuple
        Assembly, eigenvalues and eigenvectors.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../tests/multidomain/test_cylinder_blade_stiffened.py
        :pyobject: test_cylinder_blade_stiffened_compression_lb_Nxx_cte

    """
    md, conns = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n)
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

    k0 = md.calc_kC(conns, silent=True)
    kG = md.calc_kG(silent=True)
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)
    return md, eigvals, eigvecs


def cylinder_blade_stiffened_compression_lb_Nxx_from_static(height, r, stack,
        stack_blades, width_blades, plyt, laminaprop, npanels, Nxxs_skin,
        Nxxs_blade, m=8, n=8,
        num_eigvalues=20):
    """Linear buckling analysis with a Nxx calculated using static analysis

    See :func:`.create_cylinder_blade_stiffened` for most parameters.

    Parameters
    ----------
    Nxxs_skin : list
        A Nxx for each skin panel.
    Nxxs_blade : list
        A Nxx for each blade stiffener.
    num_eigvalues : int
        Number of eigenvalues to be extracted.

    Returns
    -------
    md, c, eigvals, eigvecs : tuple
        Assembly, static results, eigenvalues and eigenvectors.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../tests/multidomain/test_cylinder_blade_stiffened.py
        :pyobject: test_cylinder_blade_stiffened_compression_lb_Nxx_from_static

    """
    md, conns = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n)
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
        p.add_distr_load_fixed_x(p.a, lambda y: Nxx, None, None)

    fext = md.calc_fext(silent=True)

    k0 = md.calc_kC(conns, silent=True)
    incs, cs = static(k0, fext, silent=True)
    c = cs[0]
    kG = md.calc_kG(c=c, silent=True)

    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)

    return md, c, eigvals, eigvecs


def cylinder_blade_stiffened_compression_lb_pd_from_static(height, r, stack,
        stack_blades, width_blades, plyt, laminaprop, npanels, pds_skin,
        pds_blade, m=8, n=8,
        num_eigvalues=20, ku=1.e6):
    """Linear buckling analysis with prescribed displacement

    See :func:`.create_cylinder_blade_stiffened` for most parameters.

    Parameters
    ----------
    pds_skin : list
        A prescribed axial compression displacement for each skin panel.
    pds_blade : list
        A prescribed axial compression displacement for each blade stiffener.
    num_eigvalues : int
        Number of eigenvalues to be extracted.
    ku : float
        Penalty stiffness used for prescribing displacements.

    Returns
    -------
    md, c, eigvals, eigvecs : tuple
        Assembly, static results, eigenvalues and eigenvectors.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../tests/multidomain/test_cylinder_blade_stiffened.py
        :pyobject: test_cylinder_blade_stiffened_compression_lb_pd_from_static

    """
    md, conns = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n)
    if len(pds_skin) != npanels:
        raise ValueError('The length of "pds_skin" must be the same as "npanels"')
    if len(pds_blade) != npanels:
        raise ValueError('The length of "pds_blade" must be the same as "npanels"')
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
        p.add_distr_pd_fixed_x(p.a, ku, None, None, lambda y: pd, None, None)

    fext = md.calc_fext(silent=True)

    k0 = md.calc_kC(conns, silent=True)
    incs, cs = static(k0, fext, silent=True)
    c = cs[0]
    kG = md.calc_kG(c=c, silent=True)

    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)

    return md, c, eigvals, eigvecs
