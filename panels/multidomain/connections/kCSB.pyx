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
    double integral_ffp(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfp(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil


# TODO: explain dsb parameter
def fkCSB11(double kt, double dsb, object p1, int size, int row0, int col0):
    r"""
    Penalty approach calculation to skin-base ycte panel 1 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    dsb : float
        dsb = sum(pA.plyts)/2. + sum(pB.plyts)/2.
    p1 : Panel
        Panel() object
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
    kCSB11 : scipy.sparse.coo_matrix
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

    cdef long [:] kCSB11r, kCSB11c
    cdef double [:] kCSB11v

    cdef double f1Auf1Bu, f1Auf1Bwxi, f1Avf1Bv, f1Avf1Bw, f1Awf1Bv, f1Awf1Bw, f1Awxif1Bu, f1Awxif1Bwxi
    cdef double g1Aug1Bu, g1Aug1Bw, g1Avg1Bv, g1Avg1Bweta, g1Awg1Bu, g1Awg1Bw, g1Awetag1Bv, g1Awetag1Bweta

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

    fdim = 7*m1*n1*m1*n1

    kCSB11r = np.zeros((fdim,), dtype=INT)
    kCSB11c = np.zeros((fdim,), dtype=INT)
    kCSB11v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCSB11
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):
                f1Auf1Bu = integral_ff(i1, k1, x1u1, x1ur1, x2u1, x2ur1, x1u1, x1ur1, x2u1, x2ur1)
                f1Auf1Bwxi = integral_ffp(i1, k1, x1u1, x1ur1, x2u1, x2ur1, x1w1, x1wr1, x2w1, x2wr1)
                f1Avf1Bv = integral_ff(i1, k1, x1v1, x1vr1, x2v1, x2vr1, x1v1, x1vr1, x2v1, x2vr1)
                f1Avf1Bw = integral_ff(i1, k1, x1v1, x1vr1, x2v1, x2vr1, x1w1, x1wr1, x2w1, x2wr1)
                f1Awf1Bv = integral_ff(i1, k1, x1w1, x1wr1, x2w1, x2wr1, x1v1, x1vr1, x2v1, x2vr1)
                f1Awf1Bw = integral_ff(i1, k1, x1w1, x1wr1, x2w1, x2wr1, x1w1, x1wr1, x2w1, x2wr1)
                f1Awxif1Bu = integral_ffp(k1, i1, x1u1, x1ur1, x2u1, x2ur1, x1w1, x1wr1, x2w1, x2wr1)
                f1Awxif1Bwxi = integral_fpfp(i1, k1, x1w1, x1wr1, x2w1, x2wr1, x1w1, x1wr1, x2w1, x2wr1)

                for j1 in range(n1):
                    for l1 in range(n1):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g1Aug1Bu = integral_ff(j1, l1, y1u1, y1ur1, y2u1, y2ur1, y1u1, y1ur1, y2u1, y2ur1)
                        g1Aug1Bw = integral_ff(j1, l1, y1u1, y1ur1, y2u1, y2ur1, y1w1, y1wr1, y2w1, y2wr1)
                        g1Avg1Bv = integral_ff(j1, l1, y1v1, y1vr1, y2v1, y2vr1, y1v1, y1vr1, y2v1, y2vr1)
                        g1Avg1Bweta = integral_ffp(j1, l1, y1v1, y1vr1, y2v1, y2vr1, y1w1, y1wr1, y2w1, y2wr1)
                        g1Awg1Bu = integral_ff(j1, l1, y1w1, y1wr1, y2w1, y2wr1, y1u1, y1ur1, y2u1, y2ur1)
                        g1Awg1Bw = integral_ff(j1, l1, y1w1, y1wr1, y2w1, y2wr1, y1w1, y1wr1, y2w1, y2wr1)
                        g1Awetag1Bv = integral_ffp(l1, j1, y1v1, y1vr1, y2v1, y2vr1, y1w1, y1wr1, y2w1, y2wr1)
                        g1Awetag1Bweta = integral_fpfp(j1, l1, y1w1, y1wr1, y2w1, y2wr1, y1w1, y1wr1, y2w1, y2wr1)

                        c += 1
                        kCSB11r[c] = row+0
                        kCSB11c[c] = col+0
                        kCSB11v[c] += 0.25*a1*b1*f1Auf1Bu*g1Aug1Bu*kt
                        c += 1
                        kCSB11r[c] = row+0
                        kCSB11c[c] = col+2
                        kCSB11v[c] += 0.5*b1*dsb*f1Auf1Bwxi*g1Aug1Bw*kt
                        c += 1
                        kCSB11r[c] = row+1
                        kCSB11c[c] = col+1
                        kCSB11v[c] += 0.25*a1*b1*f1Avf1Bv*g1Avg1Bv*kt
                        c += 1
                        kCSB11r[c] = row+1
                        kCSB11c[c] = col+2
                        kCSB11v[c] += 0.5*a1*dsb*f1Avf1Bw*g1Avg1Bweta*kt
                        c += 1
                        kCSB11r[c] = row+2
                        kCSB11c[c] = col+0
                        kCSB11v[c] += 0.5*b1*dsb*f1Awxif1Bu*g1Awg1Bu*kt
                        c += 1
                        kCSB11r[c] = row+2
                        kCSB11c[c] = col+1
                        kCSB11v[c] += 0.5*a1*dsb*f1Awf1Bv*g1Awetag1Bv*kt
                        c += 1
                        kCSB11r[c] = row+2
                        kCSB11c[c] = col+2
                        kCSB11v[c] += 0.25*a1*b1*kt*(f1Awf1Bw*g1Awg1Bw + 4*(dsb*dsb)*f1Awf1Bw*g1Awetag1Bweta/(b1*b1) + 4*(dsb*dsb)*f1Awxif1Bwxi*g1Awg1Bw/(a1*a1))

    kCSB11 = coo_matrix((kCSB11v, (kCSB11r, kCSB11c)), shape=(size, size))

    return kCSB11


# TODO: explain dsb parameter
def fkCSB12(double kt, double dsb, object p1, object p2, int size, int row0, int col0):
    r"""
    Penalty approach calculation to skin-base ycte panel 1 and panel 2 coupling position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    dsb : float
    p1 : Panel
        First Panel object
    p2 : Panel
        Second Panel object
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
    kCBFycte12 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel 1 and panel 2 coupling position.

    """
    cdef int i1, j1, k2, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, b1
    cdef double x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCSB12r, kCSB12c
    cdef double [:] kCSB12v

    cdef double f1Auf2Bu, f1Avf2Bv, f1Awf2Bw, f1Awxif2Bu, f1Awf2Bv
    cdef double g1Aug2Bu, g1Avg2Bv, g1Awg2Bw, g1Awetag2Bv, g1Awg2Bu

    a1 = p1.a
    b1 = p1.b
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

    fdim = 5*m1*n1*m2*n2

    kCSB12r = np.zeros((fdim,), dtype=INT)
    kCSB12c = np.zeros((fdim,), dtype=INT)
    kCSB12v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCSB12
        c = -1
        for i1 in range(m1):
            for k2 in range(m2):
                f1Auf2Bu = integral_ff(i1, k2, x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2)
                f1Avf2Bv = integral_ff(i1, k2, x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2)
                f1Awf2Bw = integral_ff(i1, k2, x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2)
                f1Awxif2Bu = integral_ffp(k2, i1, x1u2, x1ur2, x2u2, x2ur2, x1w1, x1wr1, x2w1, x2wr1)
                f1Awf2Bv = integral_ff(i1, k2, x1w1, x1wr1, x2w1, x2wr1, x1v2, x1vr2, x2v2, x2vr2)

                for j1 in range(n1):
                    for l2 in range(n2):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        g1Aug2Bu = integral_ff(j1, l2, y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2)
                        g1Avg2Bv = integral_ff(j1, l2, y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2)
                        g1Awg2Bw = integral_ff(j1, l2, y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2)
                        g1Awetag2Bv = integral_ffp(l2, j1, y1v2, y1vr2, y2v2, y2vr2, y1w1, y1wr1, y2w1, y2wr1)
                        g1Awg2Bu = integral_ff(j1, l2, y1w1, y1wr1, y2w1, y2wr1, y1u2, y1ur2, y2u2, y2ur2)

                        c += 1
                        kCSB12r[c] = row+0
                        kCSB12c[c] = col+0
                        kCSB12v[c] += -0.25*a1*b1*f1Auf2Bu*g1Aug2Bu*kt
                        c += 1
                        kCSB12r[c] = row+1
                        kCSB12c[c] = col+1
                        kCSB12v[c] += -0.25*a1*b1*f1Avf2Bv*g1Avg2Bv*kt
                        c += 1
                        kCSB12r[c] = row+2
                        kCSB12c[c] = col+0
                        kCSB12v[c] += -0.5*b1*dsb*f1Awxif2Bu*g1Awg2Bu*kt
                        c += 1
                        kCSB12r[c] = row+2
                        kCSB12c[c] = col+1
                        kCSB12v[c] += -0.5*a1*dsb*f1Awf2Bv*g1Awetag2Bv*kt
                        c += 1
                        kCSB12r[c] = row+2
                        kCSB12c[c] = col+2
                        kCSB12v[c] += -0.25*a1*b1*f1Awf2Bw*g1Awg2Bw*kt

    kCSB12 = coo_matrix((kCSB12v, (kCSB12r, kCSB12c)), shape=(size, size))

    return kCSB12


def fkCSB22(double kt, object p1, object p2, int size, int row0, int col0):
    r"""
    Penalty approach calculation to skin-base ycte panel 2 position.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    p1 : Panel
        First Panel object
    p2 : Panel
        Second Panel object
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
    kCSB22 : scipy.sparse.coo_matrix
        A sparse matrix that adds the penalty stiffness to ycte of panel p2 position.

    """
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double a1, b1
    cdef double x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCSB22r, kCSB22c
    cdef double [:] kCSB22v

    cdef double f2Auf2Bu, f2Avf2Bv, f2Awf2Bw
    cdef double g2Aug2Bu, g2Avg2Bv, g2Awg2Bw

    a1 = p1.a
    b1 = p1.b
    m2 = p2.m
    n2 = p2.n
    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    fdim = 3*m2*n2*m2*n2

    kCSB22r = np.zeros((fdim,), dtype=INT)
    kCSB22c = np.zeros((fdim,), dtype=INT)
    kCSB22v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i2 in range(m2):
            for k2 in range(m2):
                f2Auf2Bu = integral_ff(i2, k2, x1u2, x1ur2, x2u2, x2ur2, x1u2, x1ur2, x2u2, x2ur2)
                f2Avf2Bv = integral_ff(i2, k2, x1v2, x1vr2, x2v2, x2vr2, x1v2, x1vr2, x2v2, x2vr2)
                f2Awf2Bw = integral_ff(i2, k2, x1w2, x1wr2, x2w2, x2wr2, x1w2, x1wr2, x2w2, x2wr2)

                for j2 in range(n2):
                    for l2 in range(n2):
                        row = row0 + DOF*(j2*m2 + i2)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g2Aug2Bu = integral_ff(j2, l2, y1u2, y1ur2, y2u2, y2ur2, y1u2, y1ur2, y2u2, y2ur2)
                        g2Avg2Bv = integral_ff(j2, l2, y1v2, y1vr2, y2v2, y2vr2, y1v2, y1vr2, y2v2, y2vr2)
                        g2Awg2Bw = integral_ff(j2, l2, y1w2, y1wr2, y2w2, y2wr2, y1w2, y1wr2, y2w2, y2wr2)

                        c += 1
                        kCSB22r[c] = row+0
                        kCSB22c[c] = col+0
                        kCSB22v[c] += 0.25*a1*b1*f2Auf2Bu*g2Aug2Bu*kt
                        c += 1
                        kCSB22r[c] = row+1
                        kCSB22c[c] = col+1
                        kCSB22v[c] += 0.25*a1*b1*f2Avf2Bv*g2Avg2Bv*kt
                        c += 1
                        kCSB22r[c] = row+2
                        kCSB22c[c] = col+2
                        kCSB22v[c] += 0.25*a1*b1*f2Awf2Bw*g2Awg2Bw*kt

    kCSB22 = coo_matrix((kCSB22v, (kCSB22r, kCSB22c)), shape=(size, size))

    return kCSB22
