#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np

from panels import Shell


INT = long
DOUBLE = np.float64
cdef int DOF = 3


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil


def fkCSSycte11(double kt, double kr, object p1, double ycte1,
        int size, int row0, int col0):
    r"""
    Penalty approach calculation to skin-skin ycte panel 1 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Shell
        Shell() object
    ycte1 : float
        Dimension value that determines the flag value eta.
        If ycte1 = 0 => eta = -1, if ycte1 = p1.b => eta = 1.
        Where eta=-1 stands for boundary 1 and eta=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCSSycte11 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel p1 position.

    """
    cdef int i1, k1, j1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double x1u1, x1ur1, x2u1, x2ur1
    cdef double x1v1, x1vr1, x2v1, x2vr1
    cdef double x1w1, x1wr1, x2w1, x2wr1
    cdef double y1u1, y1ur1, y2u1, y2ur1
    cdef double y1v1, y1vr1, y2v1, y2vr1
    cdef double y1w1, y1wr1, y2w1, y2wr1

    cdef long [:] kCSSycte11r, kCSSycte11c
    cdef double [:] kCSSycte11v

    cdef double etacte1
    cdef double f1Auf1Bu, f1Avf1Bv, f1Awf1Bw
    cdef double g1Au, g1Bu, g1Av, g1Bv, g1Aw, g1Bw, g1Aweta, g1Bweta

    a1 = p1.a
    b1 = p1.b
    m1 = p1.m
    n1 = p1.n
    x1u1 = p1.x1u ; x1ur1 = p1.x1ur ; x2u1 = p1.x2u ; x2ur1 = p1.x2ur
    x1v1 = p1.x1v ; x1vr1 = p1.x1vr ; x2v1 = p1.x2v ; x2vr1 = p1.x2vr
    x1w1 = p1.x1w ; x1wr1 = p1.x1wr ; x2w1 = p1.x2w ; x2wr1 = p1.x2wr
    y1u1 = p1.y1u ; y1ur1 = p1.y1ur ; y2u1 = p1.y2u ; y2ur1 = p1.y2ur
    y1v1 = p1.y1v ; y1vr1 = p1.y1vr ; y2v1 = p1.y2v ; y2vr1 = p1.y2vr
    y1w1 = p1.y1w ; y1wr1 = p1.y1wr ; y2w1 = p1.y2w ; y2wr1 = p1.y2wr

    etacte1 = 2*ycte1/b1 - 1.

    fdim = 3*m1*n1*m1*n1

    kCSSycte11r = np.zeros((fdim,), dtype=INT)
    kCSSycte11c = np.zeros((fdim,), dtype=INT)
    kCSSycte11v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):

                f1Auf1Bu = integral_ff(i1, k1, x1u1, x1ur1, x2u1, x2ur1, x1u1, x1ur1, x2u1, x2ur1)
                f1Avf1Bv = integral_ff(i1, k1, x1v1, x1vr1, x2v1, x2vr1, x1v1, x1vr1, x2v1, x2vr1)
                f1Awf1Bw = integral_ff(i1, k1, x1w1, x1wr1, x2w1, x2wr1, x1w1, x1wr1, x2w1, x2wr1)

                for j1 in range(n1):
                    g1Au = f(j1, etacte1, y1u1, y1ur1, y2u1, y2ur1)
                    g1Av = f(j1, etacte1, y1v1, y1vr1, y2v1, y2vr1)
                    g1Aw = f(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
                    g1Aweta = fp(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)

                    for l1 in range(n1):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g1Bu = f(l1, etacte1, y1u1, y1ur1, y2u1, y2ur1)
                        g1Bv = f(l1, etacte1, y1v1, y1vr1, y2v1, y2vr1)
                        g1Bw = f(l1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
                        g1Bweta = fp(l1, etacte1, y1w1, y1wr1, y2w1, y2wr1)

                        c += 1
                        kCSSycte11r[c] = row+0
                        kCSSycte11c[c] = col+0
                        kCSSycte11v[c] += 0.5*a1*f1Auf1Bu*g1Au*g1Bu*kt
                        c += 1
                        kCSSycte11r[c] = row+1
                        kCSSycte11c[c] = col+1
                        kCSSycte11v[c] += 0.5*a1*f1Avf1Bv*g1Av*g1Bv*kt
                        c += 1
                        kCSSycte11r[c] = row+2
                        kCSSycte11c[c] = col+2
                        kCSSycte11v[c] += 0.5*a1*kt*(f1Awf1Bw*g1Aw*g1Bw + 4*f1Awf1Bw*g1Aweta*g1Bweta*kr/((b1*b1)*kt))

    kCSSycte11 = coo_matrix((kCSSycte11v, (kCSSycte11r, kCSSycte11c)), shape=(size, size))

    return kCSSycte11


def fkCSSycte12(double kt, double kr, object p1, object p2,
          double ycte1, double ycte2,
          int size, int row0, int col0):
    r"""
    Penalty approach calculation to skin-skin ycte panel 1 and panel 2 coupling position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Shell
        First Shell object
    p2 : Shell
        Second Shell object
    ycte1 : float
        Dimension value that determines the flag value eta.
        If ycte1 = 0 => eta = -1, if ycte1 = p1.b => eta = 1.
        Where eta=-1 stands for boundary 1 and eta=1 stands for boundary 2.
    ycte2 : float
        Dimension value that determines the flag value eta.
        If ycte1 = 0 => eta = -1, if ycte1 = p1.b => eta = 1.
        Where eta=-1 stands for boundary 1 and eta=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCSSycte12 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel 1 and panel 2 coupling position.

    """
    cdef int i1, k2, j1, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, b1, b2
    cdef double x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCSSycte12r, kCSSycte12c
    cdef double [:] kCSSycte12v

    cdef double etacte1, etacte2
    cdef double f1Auf2Bu, f1Avf2Bv, f1Awf2Bw
    cdef double g1Au, g2Bu, g1Av, g2Bv, g1Aw, g2Bw, g1Aweta, g2Bweta

    a1 = p1.a
    b1 = p1.b
    b2 = p2.b
    m1 = p1.m
    n1 = p1.n
    m2 = p2.m
    n2 = p2.n
    x1u1 = p1.x1u ; x1ur1 = p1.x1ur ; x2u1 = p1.x2u ; x2ur1 = p1.x2ur
    x1v1 = p1.x1v ; x1vr1 = p1.x1vr ; x2v1 = p1.x2v ; x2vr1 = p1.x2vr
    x1w1 = p1.x1w ; x1wr1 = p1.x1wr ; x2w1 = p1.x2w ; x2wr1 = p1.x2wr
    y1u1 = p1.y1u ; y1ur1 = p1.y1ur ; y2u1 = p1.y2u ; y2ur1 = p1.y2ur
    y1v1 = p1.y1v ; y1vr1 = p1.y1vr ; y2v1 = p1.y2v ; y2vr1 = p1.y2vr
    y1w1 = p1.y1w ; y1wr1 = p1.y1wr ; y2w1 = p1.y2w ; y2wr1 = p1.y2wr

    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    etacte1 = 2*ycte1/b1 - 1.
    etacte2 = 2*ycte2/b2 - 1.

    fdim = 3*m1*n1*m2*n2

    kCSSycte12r = np.zeros((fdim,), dtype=INT)
    kCSSycte12c = np.zeros((fdim,), dtype=INT)
    kCSSycte12v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k2 in range(m2):

                f1Auf2Bu = integral_ff(i1, k2, x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2)
                f1Avf2Bv = integral_ff(i1, k2, x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2)
                f1Awf2Bw = integral_ff(i1, k2, x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2)

                for j1 in range(n1):

                    g1Au = f(j1, etacte1, y1u1, y1ur1, y2u1, y2ur1)
                    g1Av = f(j1, etacte1, y1v1, y1vr1, y2v1, y2vr1)
                    g1Aw = f(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
                    g1Aweta = fp(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)

                    for l2 in range(n2):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry
                        #if row > col:
                            #continue

                        g2Bu = f(l2, etacte2, y1u2, y1ur2, y2u2, y2ur2)
                        g2Bv = f(l2, etacte2, y1v2, y1vr2, y2v2, y2vr2)
                        g2Bw = f(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
                        g2Bweta = fp(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

                        c += 1
                        kCSSycte12r[c] = row+0
                        kCSSycte12c[c] = col+0
                        kCSSycte12v[c] += -0.5*a1*f1Auf2Bu*g1Au*g2Bu*kt
                        c += 1
                        kCSSycte12r[c] = row+1
                        kCSSycte12c[c] = col+1
                        kCSSycte12v[c] += -0.5*a1*f1Avf2Bv*g1Av*g2Bv*kt
                        c += 1
                        kCSSycte12r[c] = row+2
                        kCSSycte12c[c] = col+2
                        kCSSycte12v[c] += -0.5*a1*kt*(f1Awf2Bw*g1Aw*g2Bw + 4*f1Awf2Bw*g1Aweta*g2Bweta*kr/(b1*b2*kt))

    kCSSycte12 = coo_matrix((kCSSycte12v, (kCSSycte12r, kCSSycte12c)), shape=(size, size))

    return kCSSycte12


def fkCSSycte22(double kt, double kr, object p1, object p2,
          double ycte2,
          int size, int row0, int col0):
    r"""
    Penalty approach calculation to skin-skin ycte panel 2 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Shell
        First Shell object
    p2 : Shell
        Second Shell object
    ycte2 : float
        Dimension value that determines the flag value eta.
        If ycte1 = 0 => eta = -1, if ycte1 = p1.b => eta = 1.
        Where eta=-1 stands for boundary 1 and eta=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the PanelAssemly.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCSSycte22 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel p2 position.

    """
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double a1, b2
    cdef double x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCSSycte22r, kCSSycte22c
    cdef double [:] kCSSycte22v

    cdef double etacte2
    cdef double f2Auf2Bu, f2Avf2Bv, f2Awf2Bw
    cdef double g2Au, g2Bu, g2Av, g2Bv, g2Aw, g2Bw, g2Aweta, g2Bweta

    a1 = p1.a
    b2 = p2.b
    m2 = p2.m
    n2 = p2.n
    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    etacte2 = 2*ycte2/b2 - 1.

    fdim = 3*m2*n2*m2*n2

    kCSSycte22r = np.zeros((fdim,), dtype=INT)
    kCSSycte22c = np.zeros((fdim,), dtype=INT)
    kCSSycte22v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i2 in range(m2):
            for k2 in range(m2):

                f2Auf2Bu = integral_ff(i2, k2, x1u2, x1ur2, x2u2, x2ur2, x1u2, x1ur2, x2u2, x2ur2)
                f2Avf2Bv = integral_ff(i2, k2, x1v2, x1vr2, x2v2, x2vr2, x1v2, x1vr2, x2v2, x2vr2)
                f2Awf2Bw = integral_ff(i2, k2, x1w2, x1wr2, x2w2, x2wr2, x1w2, x1wr2, x2w2, x2wr2)

                for j2 in range(n2):
                    g2Au = f(j2, etacte2, y1u2, y1ur2, y2u2, y2ur2)
                    g2Av = f(j2, etacte2, y1v2, y1vr2, y2v2, y2vr2)
                    g2Aw = f(j2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
                    g2Aweta = fp(j2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

                    for l2 in range(n2):
                        row = row0 + DOF*(j2*m2 + i2)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g2Bu = f(l2, etacte2, y1u2, y1ur2, y2u2, y2ur2)
                        g2Bv = f(l2, etacte2, y1v2, y1vr2, y2v2, y2vr2)
                        g2Bw = f(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
                        g2Bweta = fp(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

                        c += 1
                        kCSSycte22r[c] = row+0
                        kCSSycte22c[c] = col+0
                        kCSSycte22v[c] += 0.5*a1*f2Auf2Bu*g2Au*g2Bu*kt
                        c += 1
                        kCSSycte22r[c] = row+1
                        kCSSycte22c[c] = col+1
                        kCSSycte22v[c] += 0.5*a1*f2Avf2Bv*g2Av*g2Bv*kt
                        c += 1
                        kCSSycte22r[c] = row+2
                        kCSSycte22c[c] = col+2
                        kCSSycte22v[c] += 0.5*a1*kt*(f2Awf2Bw*g2Aw*g2Bw + 4*f2Awf2Bw*g2Aweta*g2Bweta*kr/((b2*b2)*kt))

    kCSSycte22 = coo_matrix((kCSSycte22v, (kCSSycte22r, kCSSycte22c)), shape=(size, size))

    return kCSSycte22
