import numpy as np
from scipy.sparse import csr_matrix

from composites import laminated_plate
from structsolve import freq
from structsolve.sparseutils import make_symmetric

from panels import Shell
import panels.multidomain.connections as connections


def test_conn_4panels_kt_kr():
    """Compare result of 4 assembled panels with single-domain results

    The panel assembly looks like::

         _________ _____
        |         |     |
        |         |     |
        |   p01   | p02 |
        |         |     |
        |_________|_____|
        |   p03   | p04 |
        |         |     |
        |         |     |
        |         |     |
        |         |     |
        |         |     |
        |_________|_____|

    """
    print('Testing validity of the default kt and kr values')

    plyt = 1.e-3 * 0.125
    laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    stack=[0, 45, -45, 90, -45, 45, 0]
    lam = laminated_plate(stack=stack, plyt=plyt, laminaprop=laminaprop)

    rho=1.3e3

    r = 10.
    m = 10
    n = 10

    a1 = 1.5
    a2 = 1.5
    a3 = 2.5
    a4 = 2.5
    b1 = 1.5
    b2 = 0.5
    b3 = 1.5
    b4 = 0.5

    A11 = lam.ABD[0, 0]
    A22 = lam.ABD[1, 1]
    D11 = lam.ABD[3, 3]
    D22 = lam.ABD[4, 4]

    p01 = Shell(group='panels', x0=a3, y0=b2, a=a1, b=b1, r=r, m=m, n=n, plyt=plyt, stack=stack, laminaprop=laminaprop, rho=rho)
    p02 = Shell(group='panels', x0=a3, y0=0, a=a2, b=b2, r=r, m=m, n=n, plyt=plyt, stack=stack, laminaprop=laminaprop, rho=rho)
    p03 = Shell(group='panels', x0=0, y0=b2, a=a3, b=b3, r=r, m=m, n=n, plyt=plyt, stack=stack, laminaprop=laminaprop, rho=rho)
    p04 = Shell(group='panels', x0=0, y0=0, a=a4, b=b4, r=r, m=m, n=n, plyt=plyt, stack=stack, laminaprop=laminaprop, rho=rho)

    kt13, kr13 = connections.calc_kt_kr(p01, p03, 'xcte')
    kt24, kr24 = connections.calc_kt_kr(p02, p04, 'xcte')
    kt12, kr12 = connections.calc_kt_kr(p01, p02, 'ycte')
    kt34, kr34 = connections.calc_kt_kr(p03, p04, 'ycte')

    # boundary conditions
    p01.x1u = 1 ; p01.x1ur = 1 ; p01.x2u = 0 ; p01.x2ur = 1
    p01.x1v = 1 ; p01.x1vr = 1 ; p01.x2v = 0 ; p01.x2vr = 1
    p01.x1w = 1 ; p01.x1wr = 1 ; p01.x2w = 0 ; p01.x2wr = 1
    p01.y1u = 1 ; p01.y1ur = 1 ; p01.y2u = 0 ; p01.y2ur = 1
    p01.y1v = 1 ; p01.y1vr = 1 ; p01.y2v = 0 ; p01.y2vr = 1
    p01.y1w = 1 ; p01.y1wr = 1 ; p01.y2w = 0 ; p01.y2wr = 1

    p02.x1u = 1 ; p02.x1ur = 1 ; p02.x2u = 0 ; p02.x2ur = 1
    p02.x1v = 1 ; p02.x1vr = 1 ; p02.x2v = 0 ; p02.x2vr = 1
    p02.x1w = 1 ; p02.x1wr = 1 ; p02.x2w = 0 ; p02.x2wr = 1
    p02.y1u = 0 ; p02.y1ur = 1 ; p02.y2u = 1 ; p02.y2ur = 1
    p02.y1v = 0 ; p02.y1vr = 1 ; p02.y2v = 1 ; p02.y2vr = 1
    p02.y1w = 0 ; p02.y1wr = 1 ; p02.y2w = 1 ; p02.y2wr = 1

    p03.x1u = 0 ; p03.x1ur = 1 ; p03.x2u = 1 ; p03.x2ur = 1
    p03.x1v = 0 ; p03.x1vr = 1 ; p03.x2v = 1 ; p03.x2vr = 1
    p03.x1w = 0 ; p03.x1wr = 1 ; p03.x2w = 1 ; p03.x2wr = 1
    p03.y1u = 1 ; p03.y1ur = 1 ; p03.y2u = 0 ; p03.y2ur = 1
    p03.y1v = 1 ; p03.y1vr = 1 ; p03.y2v = 0 ; p03.y2vr = 1
    p03.y1w = 1 ; p03.y1wr = 1 ; p03.y2w = 0 ; p03.y2wr = 1

    p04.x1u = 0 ; p04.x1ur = 1 ; p04.x2u = 1 ; p04.x2ur = 1
    p04.x1v = 0 ; p04.x1vr = 1 ; p04.x2v = 1 ; p04.x2vr = 1
    p04.x1w = 0 ; p04.x1wr = 1 ; p04.x2w = 1 ; p04.x2wr = 1
    p04.y1u = 0 ; p04.y1ur = 1 ; p04.y2u = 1 ; p04.y2ur = 1
    p04.y1v = 0 ; p04.y1vr = 1 ; p04.y2v = 1 ; p04.y2vr = 1
    p04.y1w = 0 ; p04.y1wr = 1 ; p04.y2w = 1 ; p04.y2wr = 1

    conndict = [
        dict(p1=p01, p2=p02, func='SSycte', ycte1=0, ycte2=p02.b, kt=kt12, kr=kr12),
        dict(p1=p01, p2=p03, func='SSxcte', xcte1=0, xcte2=p03.a, kt=kt13, kr=kr13),
        dict(p1=p02, p2=p04, func='SSxcte', xcte1=0, xcte2=p04.a, kt=kt24, kr=kr24),
        dict(p1=p03, p2=p04, func='SSycte', ycte1=0, ycte2=p04.b, kt=kt34, kr=kr34),
        ]

    panels = [p01, p02, p03, p04]

    size = sum([3*p.m*p.n for p in panels])

    k0 = 0
    kM = 0

    row0 = 0
    col0 = 0
    for p in panels:
        k0 += p.calc_kC(row0=row0, col0=col0, size=size, silent=True, finalize=False)
        kM += p.calc_kM(row0=row0, col0=col0, size=size, silent=True, finalize=False)
        p.row_start = row0
        p.col_start = col0
        row0 += 3*p.m*p.n
        col0 += 3*p.m*p.n
        p.row_end = row0
        p.col_end = col0

    for conn in conndict:
        if conn.get('has_deffect'): # connecting if there is no deffect
            continue
        p1 = conn['p1']
        p2 = conn['p2']
        if conn['func'] == 'SSycte':
            k0 += connections.kCSSycte.fkCSSycte11(
                    conn['kt'], conn['kr'], p1, conn['ycte1'],
                    size, p1.row_start, col0=p1.col_start)
            k0 += connections.kCSSycte.fkCSSycte12(
                    conn['kt'], conn['kr'], p1, p2, conn['ycte1'], conn['ycte2'],
                    size, p1.row_start, col0=p2.col_start)
            k0 += connections.kCSSycte.fkCSSycte22(
                    conn['kt'], conn['kr'], p1, p2, conn['ycte2'],
                    size, p2.row_start, col0=p2.col_start)
        elif conn['func'] == 'SSxcte':
            k0 += connections.kCSSxcte.fkCSSxcte11(
                    conn['kt'], conn['kr'], p1, conn['xcte1'],
                    size, p1.row_start, col0=p1.col_start)
            k0 += connections.kCSSxcte.fkCSSxcte12(
                    conn['kt'], conn['kr'], p1, p2, conn['xcte1'], conn['xcte2'],
                    size, p1.row_start, col0=p2.col_start)
            k0 += connections.kCSSxcte.fkCSSxcte22(
                    conn['kt'], conn['kr'], p1, p2, conn['xcte2'],
                    size, p2.row_start, col0=p2.col_start)

    assert np.any(np.isnan(k0.data)) == False
    assert np.any(np.isinf(k0.data)) == False
    k0 = csr_matrix(make_symmetric(k0))
    assert np.any(np.isnan(kM.data)) == False
    assert np.any(np.isinf(kM.data)) == False
    kM = csr_matrix(make_symmetric(kM))

    eigvals, eigvecs = freq(k0, kM, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=25, num_eigvalues_print=5)

    # Results for single panel
    m = 13
    n = 13
    singlepanel = Shell(a=(a1+a3), b=(b1+b2), r=r, m=m, n=n, plyt=plyt, stack=stack, laminaprop=laminaprop, rho=rho)
    k0 = singlepanel.calc_kC()
    kM = singlepanel.calc_kM()
    sp_eigvals, sp_eigvecs = freq(k0, kM, tol=0, sparse_solver=True, silent=True,
             num_eigvalues=25, num_eigvalues_print=5)

    assert np.isclose(eigvals[0], sp_eigvals[0], atol=0.01, rtol=0.01)


if __name__ == '__main__':
    test_conn_4panels_kt_kr()
