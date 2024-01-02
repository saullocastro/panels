#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np


INT = long
DOUBLE = np.float64
cdef int DOF = 3


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil


def fkCBFxcte11(double kt, double kr, object p1, double xcte1,
          int size, int row0, int col0):
    r"""
    Penalty approach calculation to base-flange xcte panel 1 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1 : Shell
        Shell() object
    xcte1 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the MultiDomain.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCBFxcte11 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to xcte of panel p1 position.

    """
    cdef int i1, j1, k1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double x1u1, x1ur1, x2u1, x2ur1
    cdef double x1v1, x1vr1, x2v1, x2vr1
    cdef double x1w1, x1wr1, x2w1, x2wr1
    cdef double y1u1, y1ur1, y2u1, y2ur1
    cdef double y1v1, y1vr1, y2v1, y2vr1
    cdef double y1w1, y1wr1, y2w1, y2wr1

    cdef long [:] kCBFxcte11r, kCBFxcte11c
    cdef double [:] kCBFxcte11v

    cdef double xicte1
    cdef double g1Auf1Bu, g1Avf1Bv, g1Awf1Bw
    cdef double f1Au, f1Av, f1Aw, f1Awxi, f1Bu, f1Bv, f1Bw, f1Bwxi

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

    xicte1 = 2*xcte1/a1 - 1.

    fdim = 3*m1*n1*m1*n1

    kCBFxcte11r = np.zeros((fdim,), dtype=INT)
    kCBFxcte11c = np.zeros((fdim,), dtype=INT)
    kCBFxcte11v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for j1 in range(n1):
            for l1 in range(n1):
                g1Auf1Bu = integral_ff(j1, l1, y1u1, y1ur1, y2u1, y2ur1, y1u1, y1ur1, y2u1, y2ur1)
                g1Avf1Bv = integral_ff(j1, l1, y1v1, y1vr1, y2v1, y2vr1, y1v1, y1vr1, y2v1, y2vr1)
                g1Awf1Bw = integral_ff(j1, l1, y1w1, y1wr1, y2w1, y2wr1, y1w1, y1wr1, y2w1, y2wr1)

                for i1 in range(m1):
                    f1Au = f(i1, xicte1, x1u1, x1ur1, x2u1, x2ur1)
                    f1Av = f(i1, xicte1, x1v1, x1vr1, x2v1, x2vr1)
                    f1Aw = f(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)
                    f1Awxi = fp(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)

                    for k1 in range(m1):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        f1Bu = f(k1, xicte1, x1u1, x1ur1, x2u1, x2ur1)
                        f1Bv = f(k1, xicte1, x1v1, x1vr1, x2v1, x2vr1)
                        f1Bw = f(k1, xicte1, x1w1, x1wr1, x2w1, x2wr1)
                        f1Bwxi = fp(k1, xicte1, x1w1, x1wr1, x2w1, x2wr1)

                        c += 1
                        kCBFxcte11r[c] = row+0
                        kCBFxcte11c[c] = col+0
                        kCBFxcte11v[c] += 0.5*b1*f1Au*f1Bu*g1Auf1Bu*kt
                        c += 1
                        kCBFxcte11r[c] = row+1
                        kCBFxcte11c[c] = col+1
                        kCBFxcte11v[c] += 0.5*b1*f1Av*f1Bv*g1Avf1Bv*kt
                        c += 1
                        kCBFxcte11r[c] = row+2
                        kCBFxcte11c[c] = col+2
                        kCBFxcte11v[c] += 0.5*b1*kt*(f1Aw*f1Bw*g1Awf1Bw + 4*f1Awxi*f1Bwxi*g1Awf1Bw*kr/((a1*a1)*kt))

    kCBFxcte11 = coo_matrix((kCBFxcte11v, (kCBFxcte11r, kCBFxcte11c)), shape=(size, size))

    return kCBFxcte11


def fkCBFxcte12(double kt, double kr, object p1, object p2,
          double xcte1, double xcte2,
          int size, int row0, int col0):
    r"""
    Penalty approach calculation to base-flange xcte panel 1 and panel 2 coupling position.

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
    xcte1 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    xcte2 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the MultiDomain.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCBFxcte12 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to xcte of panel 1 and panel 2 coupling position.

    """
    cdef int i1, j1, k2, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, a2, b1, b2
    cdef double x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCBFxcte12r, kCBFxcte12c
    cdef double [:] kCBFxcte12v

    cdef double xicte1, xicte2
    cdef double g1Auf2Bw, g1Avf2Bv, g1Awf2Bu, g1Awf2Bw
    cdef double f1Au, f1Av, f1Aw, f1Awxi, f2Bu, f2Bv, f2Bw, f2Bwxi

    a1 = p1.a
    a2 = p2.a
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

    xicte1 = 2*xcte1/a1 - 1.
    xicte2 = 2*xcte2/a2 - 1.

    fdim = 4*m1*n1*m2*n2

    kCBFxcte12r = np.zeros((fdim,), dtype=INT)
    kCBFxcte12c = np.zeros((fdim,), dtype=INT)
    kCBFxcte12v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for j1 in range(n1):
            for l2 in range(n2):
                g1Auf2Bw = integral_ff(j1, l2, y1u1, y1ur1, y2u1, y2ur1, y1w2, y1wr2, y2w2, y2wr2)
                g1Avf2Bv = integral_ff(j1, l2, y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2)
                g1Awf2Bu = integral_ff(j1, l2, y1w1, y1wr1, y2w1, y2wr1, y1u2, y1ur2, y2u2, y2ur2)
                g1Awf2Bw = integral_ff(j1, l2, y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2)

                for i1 in range(m1):
                    f1Au = f(i1, xicte1, x1u1, x1ur1, x2u1, x2ur1)
                    f1Av = f(i1, xicte1, x1v1, x1vr1, x2v1, x2vr1)
                    f1Aw = f(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)
                    f1Awxi = fp(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)

                    for k2 in range(m2):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        f2Bu = f(k2, xicte2, x1u2, x1ur2, x2u2, x2ur2)
                        f2Bv = f(k2, xicte2, x1v2, x1vr2, x2v2, x2vr2)
                        f2Bw = f(k2, xicte2, x1w2, x1wr2, x2w2, x2wr2)
                        f2Bwxi = fp(k2, xicte2, x1w2, x1wr2, x2w2, x2wr2)

                        c += 1
                        kCBFxcte12r[c] = row+0
                        kCBFxcte12c[c] = col+2
                        kCBFxcte12v[c] += -0.5*b1*f1Au*f2Bw*g1Auf2Bw*kt
                        c += 1
                        kCBFxcte12r[c] = row+1
                        kCBFxcte12c[c] = col+1
                        kCBFxcte12v[c] += -0.5*b1*f1Av*f2Bv*g1Avf2Bv*kt
                        c += 1
                        kCBFxcte12r[c] = row+2
                        kCBFxcte12c[c] = col+0
                        kCBFxcte12v[c] += 0.5*b1*f1Aw*f2Bu*g1Awf2Bu*kt
                        c += 1
                        kCBFxcte12r[c] = row+2
                        kCBFxcte12c[c] = col+2
                        kCBFxcte12v[c] += -2*b1*f1Awxi*f2Bwxi*g1Awf2Bw*kr/(a1*a2)

    kCBFxcte12 = coo_matrix((kCBFxcte12v, (kCBFxcte12r, kCBFxcte12c)), shape=(size, size))

    return kCBFxcte12


def fkCBFxcte22(double kt, double kr, object p1, object p2, double xcte2,
          int size, int row0, int col0):
    r"""
    Penalty approach calculation to base-flange xcte panel 2 position.

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
    xcte2 : float
        Dimension value that determines the flag value xi.
        If xcte1 = 0 => xi = -1, if xcte1 = p1.a => xi = 1.
        Where xi=-1 stands for boundary 1 and xi=1 stands for boundary 2.
    size : int
        Size of assembly stiffness matrix, which are calculated by sum([3*p.m*p.n for p in self.panels]).
        The size of the assembly can be calculated calling the MultiDomain.get_size() method.
    row0 : int
        Row position of constitutive matrix being calculated.
    col0 : int
        Column position of constitutive matrix being calculated.

    Returns
    -------
    kCBFxcte22 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to xcte of panel p2 position.

    """
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double b1, a2
    cdef double x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCBFxcte22r, kCBFxcte22c
    cdef double [:] kCBFxcte22v

    cdef double xicte2
    cdef double g2Auf2Bu, g2Avf2Bv, g2Awf2Bw
    cdef double f2Au, f2Bu, f2Av, f2Bv, f2Aw, f2Bw, f2Awxi, f2Bwxi
    b1 = p1.b
    a2 = p2.a
    m2 = p2.m
    n2 = p2.n
    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    xicte2 = 2*xcte2/a2 - 1.

    fdim = 3*m2*n2*m2*n2

    kCBFxcte22r = np.zeros((fdim,), dtype=INT)
    kCBFxcte22c = np.zeros((fdim,), dtype=INT)
    kCBFxcte22v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for j2 in range(n2):
            for l2 in range(n2):
                g2Auf2Bu = integral_ff(j2, l2, y1u2, y1ur2, y2u2, y2ur2, y1u2, y1ur2, y2u2, y2ur2)
                g2Avf2Bv = integral_ff(j2, l2, y1v2, y1vr2, y2v2, y2vr2, y1v2, y1vr2, y2v2, y2vr2)
                g2Awf2Bw = integral_ff(j2, l2, y1w2, y1wr2, y2w2, y2wr2, y1w2, y1wr2, y2w2, y2wr2)

                for i2 in range(m2):
                    f2Au = f(i2, xicte2, x1u2, x1ur2, x2u2, x2ur2)
                    f2Av = f(i2, xicte2, x1v2, x1vr2, x2v2, x2vr2)
                    f2Aw = f(i2, xicte2, x1w2, x1wr2, x2w2, x2wr2)
                    f2Awxi = fp(i2, xicte2, x1w2, x1wr2, x2w2, x2wr2)

                    for k2 in range(m2):
                        row = row0 + DOF*(j2*m2 + i2)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        f2Bu = f(k2, xicte2, x1u2, x1ur2, x2u2, x2ur2)
                        f2Bv = f(k2, xicte2, x1v2, x1vr2, x2v2, x2vr2)
                        f2Bw = f(k2, xicte2, x1w2, x1wr2, x2w2, x2wr2)
                        f2Bwxi = fp(k2, xicte2, x1w2, x1wr2, x2w2, x2wr2)

                        c += 1
                        kCBFxcte22r[c] = row+0
                        kCBFxcte22c[c] = col+0
                        kCBFxcte22v[c] += 0.5*b1*f2Au*f2Bu*g2Auf2Bu*kt
                        c += 1
                        kCBFxcte22r[c] = row+1
                        kCBFxcte22c[c] = col+1
                        kCBFxcte22v[c] += 0.5*b1*f2Av*f2Bv*g2Avf2Bv*kt
                        c += 1
                        kCBFxcte22r[c] = row+2
                        kCBFxcte22c[c] = col+2
                        kCBFxcte22v[c] += 0.5*b1*kt*(f2Aw*f2Bw*g2Awf2Bw + 4*f2Awxi*f2Bwxi*g2Awf2Bw*kr/((a2*a2)*kt))

    kCBFxcte22 = coo_matrix((kCBFxcte22v, (kCBFxcte22r, kCBFxcte22c)), shape=(size, size))

    return kCBFxcte22
