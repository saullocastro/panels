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


def fkCSSycte11(double kt, double kr, object p1, double ycte1,
        int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double u1tx1, u2tx1
    cdef double v1tx1, v2tx1
    cdef double w1tx1, w1rx1, w2tx1, w2rx1
    cdef double u1ty1, u2ty1
    cdef double v1ty1, v2ty1
    cdef double w1ty1, w1ry1, w2ty1, w2ry1

    cdef np.ndarray[cINT, ndim=1] kCSSycte11r, kCSSycte11c
    cdef np.ndarray[cDOUBLE, ndim=1] kCSSycte11v

    cdef double etacte1
    cdef double f1Auf1Bu, f1Avf1Bv, f1Awf1Bw
    cdef double g1Au, g1Bu, g1Av, g1Bv, g1Aw, g1Bw, g1Aweta, g1Bweta

    a1 = p1.a
    b1 = p1.b
    m1 = p1.m
    n1 = p1.n
    u1tx1 = p1.u1tx ; u2tx1 = p1.u2tx
    v1tx1 = p1.v1tx ; v2tx1 = p1.v2tx
    w1tx1 = p1.w1tx ; w1rx1 = p1.w1rx ; w2tx1 = p1.w2tx ; w2rx1 = p1.w2rx
    u1ty1 = p1.u1ty ; u2ty1 = p1.u2ty
    v1ty1 = p1.v1ty ; v2ty1 = p1.v2ty
    w1ty1 = p1.w1ty ; w1ry1 = p1.w1ry ; w2ty1 = p1.w2ty ; w2ry1 = p1.w2ry

    etacte1 = 2*ycte1/b1 - 1.

    fdim = 3*m1*n1*m1*n1

    kCSSycte11r = np.zeros((fdim,), dtype=INT)
    kCSSycte11c = np.zeros((fdim,), dtype=INT)
    kCSSycte11v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):

                #FIXME not working
                f1Auf1Bu = 0 #integral_ff(i1, k1, u1tx1, u1rx1, u2tx1, u2rx1, u1tx1, u1rx1, u2tx1, u2rx1)
                f1Avf1Bv = 0 #integral_ff(i1, k1, v1tx1, v1rx1, v2tx1, v2rx1, v1tx1, v1rx1, v2tx1, v2rx1)
                f1Awf1Bw = 0 #integral_ff(i1, k1, w1tx1, w1rx1, w2tx1, w2rx1, w1tx1, w1rx1, w2tx1, w2rx1)

                for j1 in range(n1):
                    g1Au = fuv(j1, etacte1, u1ty1, u2ty1)
                    g1Av = fuv(j1, etacte1, v1ty1, v2ty1)
                    g1Aw = fw(j1, etacte1, w1ty1, w1ry1, w2ty1, w2ry1)
                    g1Aweta = fw_x(j1, etacte1, w1ty1, w1ry1, w2ty1, w2ry1)

                    for l1 in range(n1):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g1Bu = fuv(l1, etacte1, u1ty1, u2ty1)
                        g1Bv = fuv(l1, etacte1, v1ty1, v2ty1)
                        g1Bw = fw(l1, etacte1, w1ty1, w1ry1, w2ty1, w2ry1)
                        g1Bweta = fw_x(l1, etacte1, w1ty1, w1ry1, w2ty1, w2ry1)

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
    cdef int i1, k2, j1, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, b1, b2
    cdef double u1tx1, u2tx1, u1tx2, u2tx2
    cdef double v1tx1, v2tx1, v1tx2, v2tx2
    cdef double w1tx1, w1rx1, w2tx1, w2rx1, w1tx2, w1rx2, w2tx2, w2rx2
    cdef double u1ty1, u2ty1, u1ty2, u2ty2
    cdef double v1ty1, v2ty1, v1ty2, v2ty2
    cdef double w1ty1, w1ry1, w2ty1, w2ry1, w1ty2, w1ry2, w2ty2, w2ry2

    cdef np.ndarray[cINT, ndim=1] kCSSycte12r, kCSSycte12c
    cdef np.ndarray[cDOUBLE, ndim=1] kCSSycte12v

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
    u1tx1 = p1.u1tx ; u2tx1 = p1.u2tx
    v1tx1 = p1.v1tx ; v2tx1 = p1.v2tx
    w1tx1 = p1.w1tx ; w1rx1 = p1.w1rx ; w2tx1 = p1.w2tx ; w2rx1 = p1.w2rx
    u1ty1 = p1.u1ty ; u2ty1 = p1.u2ty
    v1ty1 = p1.v1ty ; v2ty1 = p1.v2ty
    w1ty1 = p1.w1ty ; w1ry1 = p1.w1ry ; w2ty1 = p1.w2ty ; w2ry1 = p1.w2ry

    u1tx2 = p2.u1tx ; u2tx2 = p2.u2tx
    v1tx2 = p2.v1tx ; v2tx2 = p2.v2tx
    w1tx2 = p2.w1tx ; w1rx2 = p2.w1rx ; w2tx2 = p2.w2tx ; w2rx2 = p2.w2rx
    u1ty2 = p2.u1ty ; u2ty2 = p2.u2ty
    v1ty2 = p2.v1ty ; v2ty2 = p2.v2ty
    w1ty2 = p2.w1ty ; w1ry2 = p2.w1ry ; w2ty2 = p2.w2ty ; w2ry2 = p2.w2ry

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

                #FIXME not working
                f1Auf2Bu = 0 #integral_ff(i1, k2, u1tx1, u1rx1, u2tx1, u2rx1, u1tx2, u1rx2, u2tx2, u2rx2)
                f1Avf2Bv = 0 #integral_ff(i1, k2, v1tx1, v1rx1, v2tx1, v2rx1, v1tx2, v1rx2, v2tx2, v2rx2)
                f1Awf2Bw = 0 #integral_ff(i1, k2, w1tx1, w1rx1, w2tx1, w2rx1, w1tx2, w1rx2, w2tx2, w2rx2)

                for j1 in range(n1):

                    g1Au = fuv(j1, etacte1, u1ty1, u2ty1)
                    g1Av = fuv(j1, etacte1, v1ty1, v2ty1)
                    g1Aw = fw(j1, etacte1, w1ty1, w1ry1, w2ty1, w2ry1)
                    g1Aweta = fw_x(j1, etacte1, w1ty1, w1ry1, w2ty1, w2ry1)

                    for l2 in range(n2):
                        row = row0 + num*(j1*m1 + i1)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry
                        #if row > col:
                            #continue

                        g2Bu = fuv(l2, etacte2, u1ty2, u2ty2)
                        g2Bv = fuv(l2, etacte2, v1ty2, v2ty2)
                        g2Bw = fw(l2, etacte2, w1ty2, w1ry2, w2ty2, w2ry2)
                        g2Bweta = fw_x(l2, etacte2, w1ty2, w1ry2, w2ty2, w2ry2)

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
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double a1, b2
    cdef double u1tx2, u2tx2
    cdef double v1tx2, v2tx2
    cdef double w1tx2, w1rx2, w2tx2, w2rx2
    cdef double u1ty2, u2ty2
    cdef double v1ty2, v2ty2
    cdef double w1ty2, w1ry2, w2ty2, w2ry2

    cdef np.ndarray[cINT, ndim=1] kCSSycte22r, kCSSycte22c
    cdef np.ndarray[cDOUBLE, ndim=1] kCSSycte22v

    cdef double etacte2
    cdef double f2Auf2Bu, f2Avf2Bv, f2Awf2Bw
    cdef double g2Au, g2Bu, g2Av, g2Bv, g2Aw, g2Bw, g2Aweta, g2Bweta

    a1 = p1.a
    b2 = p2.b
    m2 = p2.m
    n2 = p2.n
    u1tx2 = p2.u1tx ; u2tx2 = p2.u2tx
    v1tx2 = p2.v1tx ; v2tx2 = p2.v2tx
    w1tx2 = p2.w1tx ; w1rx2 = p2.w1rx ; w2tx2 = p2.w2tx ; w2rx2 = p2.w2rx
    u1ty2 = p2.u1ty ; u2ty2 = p2.u2ty
    v1ty2 = p2.v1ty ; v2ty2 = p2.v2ty
    w1ty2 = p2.w1ty ; w1ry2 = p2.w1ry ; w2ty2 = p2.w2ty ; w2ry2 = p2.w2ry

    etacte2 = 2*ycte2/b2 - 1.

    fdim = 3*m2*n2*m2*n2

    kCSSycte22r = np.zeros((fdim,), dtype=INT)
    kCSSycte22c = np.zeros((fdim,), dtype=INT)
    kCSSycte22v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        c = -1
        for i2 in range(m2):
            for k2 in range(m2):

                #FIXME not working
                f2Auf2Bu = 0 #integral_ff(i2, k2, u1tx2, u1rx2, u2tx2, u2rx2, u1tx2, u1rx2, u2tx2, u2rx2)
                f2Avf2Bv = 0 #integral_ff(i2, k2, v1tx2, v1rx2, v2tx2, v2rx2, v1tx2, v1rx2, v2tx2, v2rx2)
                f2Awf2Bw = 0 #integral_ff(i2, k2, w1tx2, w1rx2, w2tx2, w2rx2, w1tx2, w1rx2, w2tx2, w2rx2)

                for j2 in range(n2):
                    g2Au = fuv(j2, etacte2, u1ty2, u2ty2)
                    g2Av = fuv(j2, etacte2, v1ty2, v2ty2)
                    g2Aw = fw(j2, etacte2, w1ty2, w1ry2, w2ty2, w2ry2)
                    g2Aweta = fw_x(j2, etacte2, w1ty2, w1ry2, w2ty2, w2ry2)

                    for l2 in range(n2):
                        row = row0 + num*(j2*m2 + i2)
                        col = col0 + num*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g2Bu = fuv(l2, etacte2, u1ty2, u2ty2)
                        g2Bv = fuv(l2, etacte2, v1ty2, v2ty2)
                        g2Bw = fw(l2, etacte2, w1ty2, w1ry2, w2ty2, w2ry2)
                        g2Bweta = fw_x(l2, etacte2, w1ty2, w1ry2, w2ty2, w2ry2)

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
