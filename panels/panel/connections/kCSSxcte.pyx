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


def fkCSSxcte11(double kt, double kr, object p1,
                double xcte1,
                int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double u1tx1, u1rx1, u2tx1, u2rx1
    cdef double v1tx1, v1rx1, v2tx1, v2rx1
    cdef double w1tx1, w1rx1, w2tx1, w2rx1
    cdef double u1ty1, u1ry1, u2ty1, u2ry1
    cdef double v1ty1, v1ry1, v2ty1, v2ry1
    cdef double w1ty1, w1ry1, w2ty1, w2ry1

    cdef np.ndarray[cINT, ndim=1] kCSSxcte11r, kCSSxcte11c
    cdef np.ndarray[cDOUBLE, ndim=1] kCSSxcte11v

    cdef double xicte1
    cdef double f1Au, f1Bu, f1Av, f1Bv, f1Aw, f1Bw, f1Awxi, f1Bwxi
    cdef double g1Aug1Bu, g1Avg1Bv, g1Awg1Bw

    a1 = p1.a
    b1 = p1.b
    m1 = p1.m
    n1 = p1.n
    u1tx1 = p1.u1tx ; u1rx1 = p1.u1rx ; u2tx1 = p1.u2tx ; u2rx1 = p1.u2rx
    v1tx1 = p1.v1tx ; v1rx1 = p1.v1rx ; v2tx1 = p1.v2tx ; v2rx1 = p1.v2rx
    w1tx1 = p1.w1tx ; w1rx1 = p1.w1rx ; w2tx1 = p1.w2tx ; w2rx1 = p1.w2rx
    u1ty1 = p1.u1ty ; u1ry1 = p1.u1ry ; u2ty1 = p1.u2ty ; u2ry1 = p1.u2ry
    v1ty1 = p1.v1ty ; v1ry1 = p1.v1ry ; v2ty1 = p1.v2ty ; v2ry1 = p1.v2ry
    w1ty1 = p1.w1ty ; w1ry1 = p1.w1ry ; w2ty1 = p1.w2ty ; w2ry1 = p1.w2ry

    xicte1 = 2*xcte1/a1 - 1.

    fdim = 3*m1*n1*m1*n1

    kCSSxcte11r = np.zeros((fdim,), dtype=INT)
    kCSSxcte11c = np.zeros((fdim,), dtype=INT)
    kCSSxcte11v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for j1 in range(n1):
            for l1 in range(n1):
                #FIXME do numerically
                g1Aug1Bu = 0 #integral_ff(j1, l1, u1ty1, u1ry1, u2ty1, u2ry1, u1ty1, u1ry1, u2ty1, u2ry1)
                g1Avg1Bv = 0 #integral_ff(j1, l1, v1ty1, v1ry1, v2ty1, v2ry1, v1ty1, v1ry1, v2ty1, v2ry1)
                g1Awg1Bw = 0 #integral_ff(j1, l1, w1ty1, w1ry1, w2ty1, w2ry1, w1ty1, w1ry1, w2ty1, w2ry1)

                for i1 in range(m1):
                    f1Au = fuv(i1, xicte1, u1tx1, u1rx1, u2tx1, u2rx1)
                    f1Av = fuv(i1, xicte1, v1tx1, v1rx1, v2tx1, v2rx1)
                    f1Aw = fw(i1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)
                    f1Awxi = fw_x(i1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)

                    for k1 in range(m1):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        f1Bu = fuv(k1, xicte1, u1tx1, u1rx1, u2tx1, u2rx1)
                        f1Bv = fuv(k1, xicte1, v1tx1, v1rx1, v2tx1, v2rx1)
                        f1Bw = fw(k1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)
                        f1Bwxi = fw_x(k1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)

                        c += 1
                        kCSSxcte11r[c] = row+0
                        kCSSxcte11c[c] = col+0
                        kCSSxcte11v[c] += 0.5*b1*f1Au*f1Bu*g1Aug1Bu*kt
                        c += 1
                        kCSSxcte11r[c] = row+1
                        kCSSxcte11c[c] = col+1
                        kCSSxcte11v[c] += 0.5*b1*f1Av*f1Bv*g1Avg1Bv*kt
                        c += 1
                        kCSSxcte11r[c] = row+2
                        kCSSxcte11c[c] = col+2
                        kCSSxcte11v[c] += 0.5*b1*kt*(f1Aw*f1Bw*g1Awg1Bw + 4*f1Awxi*f1Bwxi*g1Awg1Bw*kr/((a1*a1)*kt))

    kCSSxcte11 = coo_matrix((kCSSxcte11v, (kCSSxcte11r, kCSSxcte11c)), shape=(size, size))

    return kCSSxcte11


def fkCSSxcte12(double kt, double kr, object p1, object p2,
                double xcte1, double xcte2,
                int size, int row0, int col0):
    cdef int i1, k2, j1, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, a2, b1, b2
    cdef double u1tx1, u1rx1, u2tx1, u2rx1, u1tx2, u1rx2, u2tx2, u2rx2
    cdef double v1tx1, v1rx1, v2tx1, v2rx1, v1tx2, v1rx2, v2tx2, v2rx2
    cdef double w1tx1, w1rx1, w2tx1, w2rx1, w1tx2, w1rx2, w2tx2, w2rx2
    cdef double u1ty1, u1ry1, u2ty1, u2ry1, u1ty2, u1ry2, u2ty2, u2ry2
    cdef double v1ty1, v1ry1, v2ty1, v2ry1, v1ty2, v1ry2, v2ty2, v2ry2
    cdef double w1ty1, w1ry1, w2ty1, w2ry1, w1ty2, w1ry2, w2ty2, w2ry2

    cdef np.ndarray[cINT, ndim=1] kCSSxcte12r, kCSSxcte12c
    cdef np.ndarray[cDOUBLE, ndim=1] kCSSxcte12v

    cdef double xicte1, xicte2
    cdef double g1Aug2Bu, g1Avg2Bv, g1Awg2Bw
    cdef double f1Au, f2Bu, f1Av, f2Bv, f1Aw, f2Bw, f1Awxi, f2Bwxi

    a1 = p1.a
    b1 = p1.b
    a2 = p2.a
    b2 = p2.b
    m1 = p1.m
    n1 = p1.n
    m2 = p2.m
    n2 = p2.n
    u1tx1 = p1.u1tx ; u1rx1 = p1.u1rx ; u2tx1 = p1.u2tx ; u2rx1 = p1.u2rx
    v1tx1 = p1.v1tx ; v1rx1 = p1.v1rx ; v2tx1 = p1.v2tx ; v2rx1 = p1.v2rx
    w1tx1 = p1.w1tx ; w1rx1 = p1.w1rx ; w2tx1 = p1.w2tx ; w2rx1 = p1.w2rx
    u1ty1 = p1.u1ty ; u1ry1 = p1.u1ry ; u2ty1 = p1.u2ty ; u2ry1 = p1.u2ry
    v1ty1 = p1.v1ty ; v1ry1 = p1.v1ry ; v2ty1 = p1.v2ty ; v2ry1 = p1.v2ry
    w1ty1 = p1.w1ty ; w1ry1 = p1.w1ry ; w2ty1 = p1.w2ty ; w2ry1 = p1.w2ry

    u1tx2 = p2.u1tx ; u1rx2 = p2.u1rx ; u2tx2 = p2.u2tx ; u2rx2 = p2.u2rx
    v1tx2 = p2.v1tx ; v1rx2 = p2.v1rx ; v2tx2 = p2.v2tx ; v2rx2 = p2.v2rx
    w1tx2 = p2.w1tx ; w1rx2 = p2.w1rx ; w2tx2 = p2.w2tx ; w2rx2 = p2.w2rx
    u1ty2 = p2.u1ty ; u1ry2 = p2.u1ry ; u2ty2 = p2.u2ty ; u2ry2 = p2.u2ry
    v1ty2 = p2.v1ty ; v1ry2 = p2.v1ry ; v2ty2 = p2.v2ty ; v2ry2 = p2.v2ry
    w1ty2 = p2.w1ty ; w1ry2 = p2.w1ry ; w2ty2 = p2.w2ty ; w2ry2 = p2.w2ry

    xicte1 = 2*xcte1/a1 - 1.
    xicte2 = 2*xcte2/a2 - 1.

    fdim = 3*m1*n1*m2*n2

    kCSSxcte12r = np.zeros((fdim,), dtype=INT)
    kCSSxcte12c = np.zeros((fdim,), dtype=INT)
    kCSSxcte12v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for j1 in range(n1):
            for l2 in range(n2):
                #FIXME do numerically
                g1Aug2Bu = 0 #integral_ff(j1, l2, u1ty1, u1ry1, u2ty1, u2ry1, u1ty2, u1ry2, u2ty2, u2ry2)
                g1Avg2Bv = 0 #integral_ff(j1, l2, v1ty1, v1ry1, v2ty1, v2ry1, v1ty2, v1ry2, v2ty2, v2ry2)
                g1Awg2Bw = 0 #integral_ff(j1, l2, w1ty1, w1ry1, w2ty1, w2ry1, w1ty2, w1ry2, w2ty2, w2ry2)

                for i1 in range(m1):
                    f1Au = fuv(i1, xicte1, u1tx1, u1rx1, u2tx1, u2rx1)
                    f1Av = fuv(i1, xicte1, v1tx1, v1rx1, v2tx1, v2rx1)
                    f1Aw = fw(i1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)
                    f1Awxi = fw_x(i1, xicte1, w1tx1, w1rx1, w2tx1, w2rx1)

                    for k2 in range(m2):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry
                        #if row > col:
                            #continue

                        f2Bu = fuv(k2, xicte2, u1tx2, u1rx2, u2tx2, u2rx2)
                        f2Bv = fuv(k2, xicte2, v1tx2, v1rx2, v2tx2, v2rx2)
                        f2Bw = fw(k2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)
                        f2Bwxi = fw_x(k2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)

                        c += 1
                        kCSSxcte12r[c] = row+0
                        kCSSxcte12c[c] = col+0
                        kCSSxcte12v[c] += -0.5*b1*f1Au*f2Bu*g1Aug2Bu*kt
                        c += 1
                        kCSSxcte12r[c] = row+1
                        kCSSxcte12c[c] = col+1
                        kCSSxcte12v[c] += -0.5*b1*f1Av*f2Bv*g1Avg2Bv*kt
                        c += 1
                        kCSSxcte12r[c] = row+2
                        kCSSxcte12c[c] = col+2
                        kCSSxcte12v[c] += -0.5*b1*kt*(f1Aw*f2Bw*g1Awg2Bw + 4*f1Awxi*f2Bwxi*g1Awg2Bw*kr/(a1*a2*kt))

    kCSSxcte12 = coo_matrix((kCSSxcte12v, (kCSSxcte12r, kCSSxcte12c)), shape=(size, size))

    return kCSSxcte12


def fkCSSxcte22(double kt, double kr, object p1, object p2,
                double xcte2,
                int size, int row0, int col0):
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double b1, a2, b2
    cdef double u1tx2, u1rx2, u2tx2, u2rx2
    cdef double v1tx2, v1rx2, v2tx2, v2rx2
    cdef double w1tx2, w1rx2, w2tx2, w2rx2
    cdef double u1ty2, u1ry2, u2ty2, u2ry2
    cdef double v1ty2, v1ry2, v2ty2, v2ry2
    cdef double w1ty2, w1ry2, w2ty2, w2ry2

    cdef np.ndarray[cINT, ndim=1] kCSSxcte22r, kCSSxcte22c
    cdef np.ndarray[cDOUBLE, ndim=1] kCSSxcte22v

    cdef double xicte2
    cdef double f2Au, f2Bu, f2Av, f2Bv, f2Aw, f2Bw, f2Awxi, f2Bwxi
    cdef double g2Aug2Bu, g2Avg2Bv, g2Awg2Bw

    b1 = p1.b
    a2 = p2.a
    b2 = p2.b
    m2 = p2.m
    n2 = p2.n
    u1tx2 = p2.u1tx ; u1rx2 = p2.u1rx ; u2tx2 = p2.u2tx ; u2rx2 = p2.u2rx
    v1tx2 = p2.v1tx ; v1rx2 = p2.v1rx ; v2tx2 = p2.v2tx ; v2rx2 = p2.v2rx
    w1tx2 = p2.w1tx ; w1rx2 = p2.w1rx ; w2tx2 = p2.w2tx ; w2rx2 = p2.w2rx
    u1ty2 = p2.u1ty ; u1ry2 = p2.u1ry ; u2ty2 = p2.u2ty ; u2ry2 = p2.u2ry
    v1ty2 = p2.v1ty ; v1ry2 = p2.v1ry ; v2ty2 = p2.v2ty ; v2ry2 = p2.v2ry
    w1ty2 = p2.w1ty ; w1ry2 = p2.w1ry ; w2ty2 = p2.w2ty ; w2ry2 = p2.w2ry

    xicte2 = 2*xcte2/a2 - 1.

    fdim = 3*m2*n2*m2*n2

    kCSSxcte22r = np.zeros((fdim,), dtype=INT)
    kCSSxcte22c = np.zeros((fdim,), dtype=INT)
    kCSSxcte22v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for j2 in range(n2):
            for l2 in range(n2):
                #FIXME do numerically
                g2Aug2Bu = 0 #integral_ff(j2, l2, u1ty2, u1ry2, u2ty2, u2ry2, u1ty2, u1ry2, u2ty2, u2ry2)
                g2Avg2Bv = 0 #integral_ff(j2, l2, v1ty2, v1ry2, v2ty2, v2ry2, v1ty2, v1ry2, v2ty2, v2ry2)
                g2Awg2Bw = 0 #integral_ff(j2, l2, w1ty2, w1ry2, w2ty2, w2ry2, w1ty2, w1ry2, w2ty2, w2ry2)

                for i2 in range(m2):
                    f2Au = fuv(i2, xicte2, u1tx2, u1rx2, u2tx2, u2rx2)
                    f2Av = fuv(i2, xicte2, v1tx2, v1rx2, v2tx2, v2rx2)
                    f2Aw = fw(i2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)
                    f2Awxi = fw_x(i2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)

                    for k2 in range(m2):
                        row = row0 + num*(j2*m2 + i2)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        f2Bu = fuv(k2, xicte2, u1tx2, u1rx2, u2tx2, u2rx2)
                        f2Bv = fuv(k2, xicte2, v1tx2, v1rx2, v2tx2, v2rx2)
                        f2Bw = fw(k2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)
                        f2Bwxi = fw_x(k2, xicte2, w1tx2, w1rx2, w2tx2, w2rx2)

                        c += 1
                        kCSSxcte22r[c] = row+0
                        kCSSxcte22c[c] = col+0
                        kCSSxcte22v[c] += 0.5*b1*f2Au*f2Bu*g2Aug2Bu*kt
                        c += 1
                        kCSSxcte22r[c] = row+1
                        kCSSxcte22c[c] = col+1
                        kCSSxcte22v[c] += 0.5*b1*f2Av*f2Bv*g2Avg2Bv*kt
                        c += 1
                        kCSSxcte22r[c] = row+2
                        kCSSxcte22c[c] = col+2
                        kCSSxcte22v[c] += 0.5*b1*kt*(f2Aw*f2Bw*g2Awg2Bw + 4*f2Awxi*f2Bwxi*g2Awg2Bw*kr/((a2*a2)*kt))

    kCSSxcte22 = coo_matrix((kCSSxcte22v, (kCSSxcte22r, kCSSxcte22c)), shape=(size, size))

    return kCSSxcte22
