'''
Test description
================

This test verifies the connection kCBFxcte.
It is built two identical panels using different connections.

panel_BFycte:
    Closed section using the existing BFycte connection.

panel_BFxcte:
    Closed section using the BFxcte connection.
    This model is equivalent the "panel_BFycte", except it used loads, connections and BCs
    in the xcte direction.

The first eigen value of each model are compared in this test and a reference NASTRAN model
"reference_model_BFxcte.dat" is used to double check the results. Compmech models presents an error
less than 1% compared with NASTRAN.
'''

import pytest
import numpy as np
from structsolve import lb, static

from panels import Shell
from panels.multidomain import MultiDomain


@pytest.fixture
def eig_value_panel_BFycte():
    '''
    Closed section using BFYcte connections

    returns
    -------
        First eigenvalue of the assembly.
    '''

    # Properties
    E1 = 127560 # MPa
    E2 = 13030. # MPa
    G12 = 6410. # MPa
    nu12 = 0.3
    ply_thickness = 0.127 # mm

    # Plate dimensions
    aB = 1181.1
    bB = 746.74

    # Spar L
    aL = 1181.1
    bL = 381.0

    #others
    m = 8
    n = 8

    simple_layup = [+45, -45]*20 + [0, 90]*20
    simple_layup += simple_layup[::-1]

    laminaprop = (E1, E2, nu12, G12, G12, G12)

    # skin panels
    B1 = Shell(group='B1', a=aB, b=bB,m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    B2 = Shell(group='B2', a=aB, b=bB,m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # spar
    L1 = Shell(group='L1', a=aL, b=bL, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    L2 = Shell(group='L2', a=aL, b=bL, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # boundary conditions
    B1.x1u = 1 ; B1.x1ur = 1 ; B1.x2u = 1 ; B1.x2ur = 1
    B1.x1v = 1 ; B1.x1vr = 1 ; B1.x2v = 1 ; B1.x2vr = 1
    B1.x1w = 1 ; B1.x1wr = 1 ; B1.x2w = 1 ; B1.x2wr = 1
    B1.y1u = 1 ; B1.y1ur = 1 ; B1.y2u = 1 ; B1.y2ur = 1
    B1.y1v = 1 ; B1.y1vr = 1 ; B1.y2v = 1 ; B1.y2vr = 1
    B1.y1w = 1 ; B1.y1wr = 1 ; B1.y2w = 1 ; B1.y2wr = 1

    B2.x1u = 1 ; B2.x1ur = 1 ; B2.x2u = 1 ; B2.x2ur = 1
    B2.x1v = 1 ; B2.x1vr = 1 ; B2.x2v = 1 ; B2.x2vr = 1
    B2.x1w = 1 ; B2.x1wr = 1 ; B2.x2w = 1 ; B2.x2wr = 1
    B2.y1u = 1 ; B2.y1ur = 1 ; B2.y2u = 1 ; B2.y2ur = 1
    B2.y1v = 1 ; B2.y1vr = 1 ; B2.y2v = 1 ; B2.y2vr = 1
    B2.y1w = 1 ; B2.y1wr = 1 ; B2.y2w = 1 ; B2.y2wr = 1

    L1.x1u = 0 ; L1.x1ur = 0 ; L1.x2u = 1 ; L1.x2ur = 1
    L1.x1v = 0 ; L1.x1vr = 0 ; L1.x2v = 1 ; L1.x2vr = 1
    L1.x1w = 0 ; L1.x1wr = 0 ; L1.x2w = 1 ; L1.x2wr = 1
    L1.y1u = 1 ; L1.y1ur = 1 ; L1.y2u = 1 ; L1.y2ur = 1
    L1.y1v = 1 ; L1.y1vr = 1 ; L1.y2v = 1 ; L1.y2vr = 1
    L1.y1w = 1 ; L1.y1wr = 1 ; L1.y2w = 1 ; L1.y2wr = 1

    L2.x1u = 0 ; L2.x1ur = 0 ; L2.x2u = 1 ; L2.x2ur = 1
    L2.x1v = 0 ; L2.x1vr = 0 ; L2.x2v = 1 ; L2.x2vr = 1
    L2.x1w = 0 ; L2.x1wr = 0 ; L2.x2w = 1 ; L2.x2wr = 1
    L2.y1u = 1 ; L2.y1ur = 1 ; L2.y2u = 1 ; L2.y2ur = 1
    L2.y1v = 1 ; L2.y1vr = 1 ; L2.y2v = 1 ; L2.y2vr = 1
    L2.y1w = 1 ; L2.y1wr = 1 ; L2.y2w = 1 ; L2.y2wr = 1

    # Assembly
    conn = [
        dict(p1=B1, p2=L1, func='BFycte', ycte1=0, ycte2=L1.b), #LB
        dict(p1=B1, p2=L2, func='BFycte', ycte1=B1.b, ycte2=L2.b),
        dict(p1=B2, p2=L1, func='BFycte', ycte1=0, ycte2=0), #LB
        dict(p1=B2, p2=L2, func='BFycte', ycte1=B1.b, ycte2=0),
    ]

    panels = [B1, B2, L1, L2]
    assy = MultiDomain(panels, conn)
    k0 = assy.calc_kC()

    #Static load case
    size = sum([3*p.m*p.n for p in panels])
    fext = np.zeros(size)

    c = None

    L1.add_point_load(L1.a, L1.b/2, 0, 90010, 0)
    L2.add_point_load(L2.a, L1.b/2, 0, 187888, 0)
    fext[L1.col_start: L1.col_end] = L1.calc_fext(silent=True)
    fext[L2.col_start: L2.col_end] = L2.calc_fext(silent=True)

    incs, cs = static(k0, fext, silent=True)

    kG = assy.calc_kG(c=cs[0])

    #Buckling load case
    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
         num_eigvalues=25, num_eigvalues_print=5)

    return eigvals[0]


@pytest.fixture
def eig_value_panel_BFxcte():
    '''
    Closed section using BFxcte connections

    returns
    -------
        First eigenvalue of the assembly.
    '''

    # Properties
    E1 = 127560 # MPa
    E2 = 13030. # MPa
    G12 = 6410. # MPa
    nu12 = 0.3
    ply_thickness = 0.127 # mm

    # Plate dimensions
    bB = 1181.1 # inverted with relation to the BFycte panel
    aB = 746.74

    # Spar L
    bT = 1181.1 # inverted with relation to the BFycte panel
    aT = 381.0

    #others
    m = 8
    n = 8

    simple_layup = [+45, -45]*20 + [90, 0]*20 # 90 and 0 inverted
    simple_layup += simple_layup[::-1]

    laminaprop = (E1, E2, nu12, G12, G12, G12)

    # skin panels
    B1 = Shell(group='B1', a=aB, b=bB,m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    B2 = Shell(group='B2', a=aB, b=bB,m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # spar
    T1 = Shell(group='T1', a=aT, b=bT, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)
    T2 = Shell(group='T2', a=aT, b=bT, m=m, n=n, plyt=ply_thickness, stack=simple_layup, laminaprop=laminaprop)

    # boundary conditions
    B1.x1u = 1 ; B1.x1ur = 1 ; B1.x2u = 1 ; B1.x2ur = 1
    B1.x1v = 1 ; B1.x1vr = 1 ; B1.x2v = 1 ; B1.x2vr = 1
    B1.x1w = 1 ; B1.x1wr = 1 ; B1.x2w = 1 ; B1.x2wr = 1
    B1.y1u = 1 ; B1.y1ur = 1 ; B1.y2u = 1 ; B1.y2ur = 1
    B1.y1v = 1 ; B1.y1vr = 1 ; B1.y2v = 1 ; B1.y2vr = 1
    B1.y1w = 1 ; B1.y1wr = 1 ; B1.y2w = 1 ; B1.y2wr = 1

    B2.x1u = 1 ; B2.x1ur = 1 ; B2.x2u = 1 ; B2.x2ur = 1
    B2.x1v = 1 ; B2.x1vr = 1 ; B2.x2v = 1 ; B2.x2vr = 1
    B2.x1w = 1 ; B2.x1wr = 1 ; B2.x2w = 1 ; B2.x2wr = 1
    B2.y1u = 1 ; B2.y1ur = 1 ; B2.y2u = 1 ; B2.y2ur = 1
    B2.y1v = 1 ; B2.y1vr = 1 ; B2.y2v = 1 ; B2.y2vr = 1
    B2.y1w = 1 ; B2.y1wr = 1 ; B2.y2w = 1 ; B2.y2wr = 1

    T1.x1u = 1 ; T1.x1ur = 1 ; T1.x2u = 1 ; T1.x2ur = 1
    T1.x1v = 1 ; T1.x1vr = 1 ; T1.x2v = 1 ; T1.x2vr = 1
    T1.x1w = 1 ; T1.x1wr = 1 ; T1.x2w = 1 ; T1.x2wr = 1
    T1.y1u = 0 ; T1.y1ur = 0 ; T1.y2u = 1 ; T1.y2ur = 1
    T1.y1v = 0 ; T1.y1vr = 0 ; T1.y2v = 1 ; T1.y2vr = 1
    T1.y1w = 0 ; T1.y1wr = 0 ; T1.y2w = 1 ; T1.y2wr = 1

    T2.x1u = 1 ; T2.x1ur = 1 ; T2.x2u = 1 ; T2.x2ur = 1
    T2.x1v = 1 ; T2.x1vr = 1 ; T2.x2v = 1 ; T2.x2vr = 1
    T2.x1w = 1 ; T2.x1wr = 1 ; T2.x2w = 1 ; T2.x2wr = 1
    T2.y1u = 0 ; T2.y1ur = 0 ; T2.y2u = 1 ; T2.y2ur = 1
    T2.y1v = 0 ; T2.y1vr = 0 ; T2.y2v = 1 ; T2.y2vr = 1
    T2.y1w = 0 ; T2.y1wr = 0 ; T2.y2w = 1 ; T2.y2wr = 1

    # Assembly
    conn = [
        dict(p1=B1, p2=T1, func='BFxcte', xcte1=0, xcte2=T1.a), #TB
        dict(p1=B1, p2=T2, func='BFxcte', xcte1=B1.a, xcte2=T2.a),
        dict(p1=B2, p2=T1, func='BFxcte', xcte1=0, xcte2=0),
        dict(p1=B2, p2=T2, func='BFxcte', xcte1=B2.a, xcte2=0),
    ]

    panels = [B1, B2, T1, T2]
    assy = MultiDomain(panels, conn)
    k0 = assy.calc_kC()

    #Static load case
    size = sum([3*p.m*p.n for p in panels])
    fext = np.zeros(size)

    c = None

    T1.add_point_load(T1.a/2, T1.b, -90010, 0, 0)
    T2.add_point_load(T2.a/2, T1.b, -187888, 0, 0)
    fext[T1.col_start: T1.col_end] = T1.calc_fext(silent=True)
    fext[T2.col_start: T2.col_end] = T2.calc_fext(silent=True)

    incs, cs = static(k0, fext, silent=True)

    kG = assy.calc_kG(c=cs[0])

    #Buckling load case
    eigvals = eigvecs = None
    eigvals, eigvecs = lb(k0, kG, tol=0, sparse_solver=True, silent=True,
         num_eigvalues=25, num_eigvalues_print=5)

    return eigvals[0]


def test_kCBFxte(eig_value_panel_BFycte, eig_value_panel_BFxcte):
    '''
    This test compare the first eigenvalue of the assemblies.
    They cannot present an error higher than 1%.
    '''
    assert np.isclose(eig_value_panel_BFycte, eig_value_panel_BFycte, atol=0.01, rtol=0.01)



