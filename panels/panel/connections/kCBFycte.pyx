#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from __future__ import division

from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np


cdef extern from 'bardell_functions_uv.hpp':
    double fuv(int i, double xi, double xi1t, double xi2t) nogil

cdef extern from 'bardell_functions_w.hpp':
    double fw(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double fw_x(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef int num = 3


def fkCBFycte11(double kt, double kr, object p1, double ycte1,
          int size, int row0, int col0):
    cdef int i1, j1, k1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double x1u1, x2u1
    cdef double x1v1, x2v1
    cdef double x1w1, x1wr1, x2w1, x2wr1
    cdef double y1u1, y2u1
    cdef double y1v1, y2v1
    cdef double y1w1, y1wr1, y2w1, y2wr1

    cdef np.ndarray[cINT, ndim=1] kCBFycte11r, kCBFycte11c
    cdef np.ndarray[cDOUBLE, ndim=1] kCBFycte11v

    cdef double etacte1
    cdef double f1Auf1Bu, f1Avf1Bv, f1Awf1Bw
    cdef double g1Au, g1Av, g1Aw, g1Aweta, g1Bu, g1Bv, g1Bw, g1Bweta

    a1 = p1.a
    b1 = p1.b
    m1 = p1.m
    n1 = p1.n
    x1u1 = p1.x1u ; x2u1 = p1.x2u
    x1v1 = p1.x1v ; x2v1 = p1.x2v
    x1w1 = p1.x1w ; x1wr1 = p1.x1wr ; x2w1 = p1.x2w ; x2wr1 = p1.x2wr
    y1u1 = p1.y1u ; y2u1 = p1.y2u
    y1v1 = p1.y1v ; y2v1 = p1.y2v
    y1w1 = p1.y1w ; y1wr1 = p1.y1wr ; y2w1 = p1.y2w ; y2wr1 = p1.y2wr

    etacte1 = 2*ycte1/b1 - 1.

    fdim = 3*m1*n1*m1*n1

    kCBFycte11r = np.zeros((fdim,), dtype=INT)
    kCBFycte11c = np.zeros((fdim,), dtype=INT)
    kCBFycte11v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):
                #FIXME not working do integration numerically
                f1Auf1Bu = 0 #integral_ff(i1, k1, x1u1, u1rx1, x2u1, u2rx1, x1u1, u1rx1, x2u1, u2rx1)
                f1Avf1Bv = 0 #integral_ff(i1, k1, x1v1, v1rx1, x2v1, v2rx1, x1v1, v1rx1, x2v1, v2rx1)
                f1Awf1Bw = 0 #integral_ff(i1, k1, x1w1, x1wr1, x2w1, x2wr1, x1w1, x1wr1, x2w1, x2wr1)

                for j1 in range(n1):
                    g1Au = fuv(j1, etacte1, y1u1, y2u1)
                    g1Av = fuv(j1, etacte1, y1v1, y2v1)
                    g1Aw = fw(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
                    g1Aweta = fw_x(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)

                    for l1 in range(n1):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g1Bu = fuv(l1, etacte1, y1u1, y2u1)
                        g1Bv = fuv(l1, etacte1, y1v1, y2v1)
                        g1Bw = fw(l1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
                        g1Bweta = fw_x(l1, etacte1, y1w1, y1wr1, y2w1, y2wr1)

                        c += 1
                        kCBFycte11r[c] = row+0
                        kCBFycte11c[c] = col+0
                        kCBFycte11v[c] += 0.5*a1*f1Auf1Bu*g1Au*g1Bu*kt
                        c += 1
                        kCBFycte11r[c] = row+1
                        kCBFycte11c[c] = col+1
                        kCBFycte11v[c] += 0.5*a1*f1Avf1Bv*g1Av*g1Bv*kt
                        c += 1
                        kCBFycte11r[c] = row+2
                        kCBFycte11c[c] = col+2
                        kCBFycte11v[c] += 0.5*a1*kt*(f1Awf1Bw*g1Aw*g1Bw + 4*f1Awf1Bw*g1Aweta*g1Bweta*kr/((b1*b1)*kt))

    kCBFycte11 = coo_matrix((kCBFycte11v, (kCBFycte11r, kCBFycte11c)), shape=(size, size))

    return kCBFycte11


def fkCBFycte12(double kt, double kr, object p1, object p2,
          double ycte1, double ycte2,
          int size, int row0, int col0):
    cdef int i1, j1, k2, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, b1, b2
    cdef double x1u1, x2u1, x1u2, x2u2
    cdef double x1v1, x2v1, x1v2, x2v2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y2u1, y1u2, y2u2
    cdef double y1v1, y2v1, y1v2, y2v2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef np.ndarray[cINT, ndim=1] kCBFycte12r, kCBFycte12c
    cdef np.ndarray[cDOUBLE, ndim=1] kCBFycte12v

    cdef double etacte1, etacte2
    cdef double f1Auf2Bu, f1Avf2Bw, f1Awf2Bv, f1Awf2Bw
    cdef double g1Au, g1Av, g1Aw, g1Aweta, g2Bu, g2Bv, g2Bw, g2Bweta

    a1 = p1.a
    b1 = p1.b
    b2 = p2.b
    m1 = p1.m
    n1 = p1.n
    m2 = p2.m
    n2 = p2.n
    x1u1 = p1.x1u ; x2u1 = p1.x2u
    x1v1 = p1.x1v ; x2v1 = p1.x2v
    x1w1 = p1.x1w ; x1wr1 = p1.x1wr ; x2w1 = p1.x2w ; x2wr1 = p1.x2wr
    y1u1 = p1.y1u ; y2u1 = p1.y2u
    y1v1 = p1.y1v ; y2v1 = p1.y2v
    y1w1 = p1.y1w ; y1wr1 = p1.y1wr ; y2w1 = p1.y2w ; y2wr1 = p1.y2wr

    x1u2 = p2.x1u ; x2u2 = p2.x2u
    x1v2 = p2.x1v ; x2v2 = p2.x2v
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y2u2 = p2.y2u
    y1v2 = p2.y1v ; y2v2 = p2.y2v
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    etacte1 = 2*ycte1/b1 - 1.
    etacte2 = 2*ycte2/b2 - 1.

    fdim = 4*m1*n1*m2*n2

    kCBFycte12r = np.zeros((fdim,), dtype=INT)
    kCBFycte12c = np.zeros((fdim,), dtype=INT)
    kCBFycte12v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k2 in range(m2):
                #FIXME not working do integration numerically
                f1Auf2Bu = 0 #integral_ff(i1, k2, x1u1, u1rx1, x2u1, u2rx1, x1u2, u1rx2, x2u2, u2rx2)
                f1Avf2Bw = 0 #integral_ff(i1, k2, x1v1, v1rx1, x2v1, v2rx1, x1w2, x1wr2, x2w2, x2wr2)
                f1Awf2Bv = 0 #integral_ff(i1, k2, x1w1, x1wr1, x2w1, x2wr1, x1v2, v1rx2, x2v2, v2rx2)
                f1Awf2Bw = 0 #integral_ff(i1, k2, x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2)

                for j1 in range(n1):
                    g1Au = fuv(j1, etacte1, y1u1, y2u1)
                    g1Av = fuv(j1, etacte1, y1v1, y2v1)
                    g1Aw = fw(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
                    g1Aweta = fw_x(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)

                    for l2 in range(n2):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        g2Bu = fuv(l2, etacte2, y1u2, y2u2)
                        g2Bv = fuv(l2, etacte2, y1v2, y2v2)
                        g2Bw = fw(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
                        g2Bweta = fw_x(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

                        c += 1
                        kCBFycte12r[c] = row+0
                        kCBFycte12c[c] = col+0
                        kCBFycte12v[c] += -0.5*a1*f1Auf2Bu*g1Au*g2Bu*kt
                        c += 1
                        kCBFycte12r[c] = row+1
                        kCBFycte12c[c] = col+2
                        kCBFycte12v[c] += -0.5*a1*f1Avf2Bw*g1Av*g2Bw*kt
                        c += 1
                        kCBFycte12r[c] = row+2
                        kCBFycte12c[c] = col+1
                        kCBFycte12v[c] += 0.5*a1*f1Awf2Bv*g1Aw*g2Bv*kt
                        c += 1
                        kCBFycte12r[c] = row+2
                        kCBFycte12c[c] = col+2
                        kCBFycte12v[c] += -2*a1*f1Awf2Bw*g1Aweta*g2Bweta*kr/(b1*b2)

    kCBFycte12 = coo_matrix((kCBFycte12v, (kCBFycte12r, kCBFycte12c)), shape=(size, size))

    return kCBFycte12


def fkCBFycte22(double kt, double kr, object p1, object p2,
          double ycte2,
          int size, int row0, int col0):
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double a1, b2
    cdef double x1u2, x2u2
    cdef double x1v2, x2v2
    cdef double x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u2, y2u2
    cdef double y1v2, y2v2
    cdef double y1w2, y1wr2, y2w2, y2wr2

    cdef np.ndarray[cINT, ndim=1] kCBFycte22r, kCBFycte22c
    cdef np.ndarray[cDOUBLE, ndim=1] kCBFycte22v

    cdef double etacte2
    cdef double f2Auf2Bu, f2Avf2Bv, f2Awf2Bw
    cdef double g2Au, g2Bu, g2Av, g2Bv, g2Aw, g2Bw, g2Aweta, g2Bweta
    a1 = p1.a
    b2 = p2.b
    m2 = p2.m
    n2 = p2.n
    x1u2 = p2.x1u ; x2u2 = p2.x2u
    x1v2 = p2.x1v ; x2v2 = p2.x2v
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y2u2 = p2.y2u
    y1v2 = p2.y1v ; y2v2 = p2.y2v
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    etacte2 = 2*ycte2/b2 - 1.

    fdim = 3*m2*n2*m2*n2

    kCBFycte22r = np.zeros((fdim,), dtype=INT)
    kCBFycte22c = np.zeros((fdim,), dtype=INT)
    kCBFycte22v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i2 in range(m2):
            for k2 in range(m2):
                #FIXME not working do integration numerically
                f2Auf2Bu = 0 #integral_ff(i2, k2, x1u2, u1rx2, x2u2, u2rx2, x1u2, u1rx2, x2u2, u2rx2)
                f2Avf2Bv = 0 #integral_ff(i2, k2, x1v2, v1rx2, x2v2, v2rx2, x1v2, v1rx2, x2v2, v2rx2)
                f2Awf2Bw = 0 #integral_ff(i2, k2, x1w2, x1wr2, x2w2, x2wr2, x1w2, x1wr2, x2w2, x2wr2)

                for j2 in range(n2):
                    g2Au = fuv(j2, etacte2, y1u2, y2u2)
                    g2Av = fuv(j2, etacte2, y1v2, y2v2)
                    g2Aw = fw(j2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
                    g2Aweta = fw_x(j2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

                    for l2 in range(n2):
                        row = row0 + num*(j2*m2 + i2)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g2Bu = fuv(l2, etacte2, y1u2, y2u2)
                        g2Bv = fuv(l2, etacte2, y1v2, y2v2)
                        g2Bw = fw(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
                        g2Bweta = fw_x(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

                        c += 1
                        kCBFycte22r[c] = row+0
                        kCBFycte22c[c] = col+0
                        kCBFycte22v[c] += 0.5*a1*f2Auf2Bu*g2Au*g2Bu*kt
                        c += 1
                        kCBFycte22r[c] = row+1
                        kCBFycte22c[c] = col+1
                        kCBFycte22v[c] += 0.5*a1*f2Avf2Bv*g2Av*g2Bv*kt
                        c += 1
                        kCBFycte22r[c] = row+2
                        kCBFycte22c[c] = col+2
                        kCBFycte22v[c] += 0.5*a1*kt*(f2Awf2Bw*g2Aw*g2Bw + 4*f2Awf2Bw*g2Aweta*g2Bweta*kr/((b2*b2)*kt))

    kCBFycte22 = coo_matrix((kCBFycte22v, (kCBFycte22r, kCBFycte22c)), shape=(size, size))

    return kCBFycte22
