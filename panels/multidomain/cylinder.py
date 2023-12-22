import numpy as np
from scipy.sparse import csr_matrix

from structsolve import lb, static, Analysis

from panels.shell import Shell
from panels.multidomain import MultiDomain


def create_cylinder_assy(height, r, stack, plyt, laminaprop,
        npanels, m=8, n=8):
    r"""Cylinder Assembly

    The panel assembly looks like::


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


    Parameters
    ----------

    height : float
        Cylinder height (along `x`).
    r : float
        Cylinder radius.
    stack : list or tuple
        Stacking sequence for the cylinder.
    plyt : float
        Ply thickness.
    laminaprop : list or tuple
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13}, G_{23}`.
    npanels : int
        The number of panels the cylinder perimiter.
    m, n : int, optional
        Number of approximation terms for each panel.

    Returns
    -------
    assy, conns : tuple
        A tuple containing the assembly and the default connectivity
        list of dictionaries.

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
    assy = MultiDomain(skin)

    return assy, conns


def cylinder_compression_lb_Nxx_cte(height, r, stack, plyt, laminaprop,
        npanels, Nxxs, m=8, n=8, num_eigvalues=20):
    """Linear buckling analysis with a constant Nxx for each panel

    See :func:`.create_cylinder_assy` for most parameters.

    The cylinder has SS1 at the bottom and SS3 at the top. SS3 means that the
    edge is free to move in the axial direction.

    Parameters
    ----------
    Nxxs : list
        A Nxx for each panel.
    num_eigvalues : int
        Number of eigenvalues to be extracted.


    Returns
    -------
    assy, eigvals, eigvecs : tuple
        Assembly, eigenvalues and eigenvectors.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../../../../tests/multidomain/test_cylinder.py
        :pyobject: test_cylinder_compression_lb_Nxx_cte

    """
    assy, conns = create_cylinder_assy(height=height, r=r, stack=stack, plyt=plyt,
            laminaprop=laminaprop, npanels=npanels, m=m, n=n)
    if len(Nxxs) != npanels:
        raise ValueError('The length of "Nxxs" must be the same as "npanels"')
    for i, p in enumerate(assy.panels):
        p.Nxx = Nxxs[i]
        p.x2u = 1

    k0 = assy.calc_kC(conns, silent=True)
    kG = assy.calc_kG(silent=True)
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)
    return assy, eigvals, eigvecs


def cylinder_compression_lb_Nxx_from_static(height, r, stack, plyt, laminaprop,
        npanels, Nxxs, m=8, n=8, num_eigvalues=20):
    """Linear buckling analysis with a Nxx calculated using static analysis

    See :func:`.create_cylinder_assy` for most parameters.

    The cylinder has SS1 at the bottom and SS3 at the top. SS3 means that the
    edge is free to move in the axial direction.

    Parameters
    ----------
    Nxxs : list
        A Nxx for each panel.
    num_eigvalues : int
        Number of eigenvalues to be extracted.

    Returns
    -------
    assy, c, eigvals, eigvecs : tuple
        Assembly, static results, eigenvalues and eigenvectors.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../../../../compmech/panel/assembly/tests/test_cylinder.py
        :pyobject: test_cylinder_compression_lb_Nxx_from_static

    """
    assy, conns = create_cylinder_assy(height=height, r=r, stack=stack, plyt=plyt,
            laminaprop=laminaprop, npanels=npanels, m=m, n=n)
    if len(Nxxs) != npanels:
        raise ValueError('The length of "Nxxs" must be the same as "npanels"')

    for i, p in enumerate(assy.panels):
        p.add_distr_load_fixed_x(p.a, lambda y: Nxxs[i], None, None)
        p.x2u = 1

    fext = assy.calc_fext(silent=True)

    k0 = assy.calc_kC(conns)
    incs, cs = static(k0, fext, silent=True)
    c = cs[0]
    kG = assy.calc_kG(c=c)

    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)

    return assy, c, eigvals, eigvecs


def cylinder_spla(height, r, stack, plyt, laminaprop,
        npanels, Nxx, SPLA, m=8, n=8):
    """Non-linear buckling analysis using a perturbation load as imperfection

    See :func:`.create_cylinder_assy` for most parameters.

    Parameters
    ----------
    Nxx : float
        The applied axial compression load
    SPLA : float
        The single perturbation load used to induce the imperfection. Applied
        at the midle of the cylinder meridian.

    Returns
    -------
    assy, c : tuple
        Assembly, static results.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../../../../compmech/panel/assembly/tests/test_cylinder.py
        :pyobject: test_cylinder_spla

    """
    assy, conns = create_cylinder_assy(height=height, r=r, stack=stack, plyt=plyt,
            laminaprop=laminaprop, npanels=npanels, m=m, n=n)

    for i, p in enumerate(assy.panels):
        p.x2u = 1

    for p in assy.panels:
        p.add_distr_load_fixed_x(p.a, lambda y: p.Nxx, None, None)

    p_spla = assy.panels[0]
    p_spla.add_point_load(p_spla.a/2, p_spla.b/2, 0, 0, -SPLA, cte=True)

    assy.conn = conns
    analysis = Analysis(assy.calc_fext, assy.calc_kC, assy.calc_fint,
            assy.calc_kT)
    analysis.NL_method = 'NR'
    analysis.modified_NR = False
    analysis.line_search = False
    analysis.kT_initial_state = False
    incs, cs = analysis.static(NLgeom=True)

    return assy, incs, cs
