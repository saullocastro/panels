import numpy as np
from scipy.sparse import csr_matrix

from structsolve import lb, static

from panels.shell import Shell
from panels.multidomain import MultiDomain


def create_cylinder_blade_stiffened(height, r, stack, stack_blades,
        width_blades, plyt, laminaprop, npanels, m=8, n=8):
    r"""Cylinder Assembly

    The panel assembly looks like::


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
    assy, conns : tuple
        A tuple containing the assembly and the default connectivity
        list of dictionaries.

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

    assy = MultiDomain(skin + blades)

    return assy, conns


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
    assy, eigvals, eigvecs : tuple
        Assembly, eigenvalues and eigenvectors.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../../../../compmech/panel/assembly/tests/test_cylinder_blade_stiffened.py
        :pyobject: test_cylinder_blade_stiffened_compression_lb_Nxx_cte

    """
    assy, conns = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n)
    if len(Nxxs_skin) != npanels:
        raise ValueError('The length of "Nxxs_skin" must be the same as "npanels"')
    if len(Nxxs_blade) != npanels:
        raise ValueError('The length of "Nxxs_blade" must be the same as "npanels"')
    i_skin = -1
    i_blade = -1
    for p in assy.panels:
        p.x2u = 1
        if 'skin' in p.group:
            i_skin += 1
            p.Nxx = Nxxs_skin[i_skin]
        elif 'blade' in p.group:
            i_blade += 1
            p.Nxx = Nxxs_blade[i_blade]

    k0 = assy.calc_kC(conns, silent=True)
    kG = assy.calc_kG(silent=True)
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)
    return assy, eigvals, eigvecs


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
    assy, c, eigvals, eigvecs : tuple
        Assembly, static results, eigenvalues and eigenvectors.

    Examples
    --------

    The following example is one of the test cases:

    .. literalinclude:: ../../../../../compmech/panel/assembly/tests/test_cylinder.py
        :pyobject: test_cylinder_blade_stiffened_compression_lb_Nxx_from_static

    """
    assy, conns = create_cylinder_blade_stiffened(height=height, r=r,
            stack=stack, stack_blades=stack_blades, width_blades=width_blades,
            plyt=plyt, laminaprop=laminaprop, npanels=npanels, m=m, n=n)
    if len(Nxxs_skin) != npanels:
        raise ValueError('The length of "Nxxs_skin" must be the same as "npanels"')
    if len(Nxxs_blade) != npanels:
        raise ValueError('The length of "Nxxs_blade" must be the same as "npanels"')
    i_skin = -1
    i_blade = -1
    for p in assy.panels:
        p.x2u = 1
        if 'skin' in p.group:
            i_skin += 1
            Nxx = Nxxs_skin[i_skin]
        elif 'blade' in p.group:
            i_blade += 1
            Nxx = Nxxs_blade[i_blade]
        p.add_distr_load_fixed_x(p.a, lambda y: Nxx, None, None)

    fext = assy.calc_fext(silent=True)

    k0 = assy.calc_kC(conns, silent=True)
    incs, cs = static(k0, fext, silent=True)
    c = cs[0]
    kG = assy.calc_kG(c=c, silent=False)

    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=num_eigvalues, num_eigvalues_print=5)

    return assy, c, eigvals, eigvecs
