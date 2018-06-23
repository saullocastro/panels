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


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double calc_f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_fx(int i, double xi, double xi1t, double xi1r,
                   double xi2t, double xi2r) nogil

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef int num = 3
cdef int num1 = 3


def fkCss(double kt, double kr, double ys, double a, double b, int m, int n,
          double u1tx, double u1rx, double u2tx, double u2rx,
          double v1tx, double v1rx, double v2tx, double v2rx,
          double w1tx, double w1rx, double w2tx, double w2rx,
          double u1ty, double u1ry, double u2ty, double u2ry,
          double v1ty, double v1ry, double v2ty, double v2ry,
          double w1ty, double w1ry, double w2ty, double w2ry,
          int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef np.ndarray[cINT, ndim=1] kCssr, kCssc
    cdef np.ndarray[cDOUBLE, ndim=1] kCssv

    cdef double fAufBu, fAvfBv, fAwfBw
    cdef double gAu, gBu, gAv, gBv, gAw, gBw, gAweta, gBweta

    eta = 2*ys/b - 1.

    fdim = 3*m*n*m*n

    kCssr = np.zeros((fdim,), dtype=INT)
    kCssc = np.zeros((fdim,), dtype=INT)
    kCssv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCss
        c = -1
        for i in range(m):
            for k in range(m):
                fAufBu = integral_ff(i, k, u1tx, u1rx, u2tx, u2rx, u1tx, u1rx, u2tx, u2rx)
                fAvfBv = integral_ff(i, k, v1tx, v1rx, v2tx, v2rx, v1tx, v1rx, v2tx, v2rx)
                fAwfBw = integral_ff(i, k, w1tx, w1rx, w2tx, w2rx, w1tx, w1rx, w2tx, w2rx)

                for j in range(n):
                    gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fx(j, eta, w1ty, w1ry, w2ty, w2ry)

                    for l in range(n):
                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = calc_f(l, eta, u1ty, u1ry, u2ty, u2ry)
                        gBv = calc_f(l, eta, v1ty, v1ry, v2ty, v2ry)
                        gBw = calc_f(l, eta, w1ty, w1ry, w2ty, w2ry)
                        gBweta = calc_fx(l, eta, w1ty, w1ry, w2ty, w2ry)

                        c += 1
                        kCssr[c] = row+0
                        kCssc[c] = col+0
                        kCssv[c] += 0.5*a*fAufBu*gAu*gBu*kt
                        c += 1
                        kCssr[c] = row+1
                        kCssc[c] = col+1
                        kCssv[c] += 0.5*a*fAvfBv*gAv*gBv*kt
                        c += 1
                        kCssr[c] = row+2
                        kCssc[c] = col+2
                        kCssv[c] += 0.5*a*(fAwfBw*gAw*gBw*kt + 4*fAwfBw*gAweta*gBweta*kr/(b*b))

    kCss = coo_matrix((kCssv, (kCssr, kCssc)), shape=(size, size))

    return kCss


def fkCsf(double kt, double kr, double ys, double a, double b, double bf,
          int m, int n, int m1, int n1,
          double u1tx, double u1rx, double u2tx, double u2rx,
          double v1tx, double v1rx, double v2tx, double v2rx,
          double w1tx, double w1rx, double w2tx, double w2rx,
          double u1ty, double u1ry, double u2ty, double u2ry,
          double v1ty, double v1ry, double v2ty, double v2ry,
          double w1ty, double w1ry, double w2ty, double w2ry,
          double u1txf, double u1rxf, double u2txf, double u2rxf,
          double v1txf, double v1rxf, double v2txf, double v2rxf,
          double w1txf, double w1rxf, double w2txf, double w2rxf,
          double u1tyf, double u1ryf, double u2tyf, double u2ryf,
          double v1tyf, double v1ryf, double v2tyf, double v2ryf,
          double w1tyf, double w1ryf, double w2tyf, double w2ryf,
          int size, int row0, int col0):
    cdef int i, j, k1, l1, c, row, col
    cdef double eta

    cdef np.ndarray[cINT, ndim=1] kCsfr, kCsfc
    cdef np.ndarray[cDOUBLE, ndim=1] kCsfv

    cdef double fAurBu, fAvrBw, fAwrBv, fAwrBw
    cdef double gAu, gAv, gAw, gAweta, sBu, sBv, sBw, sBweta

    eta = 2*ys/b - 1.

    fdim = 4*m*n*m1*n1

    kCsfr = np.zeros((fdim,), dtype=INT)
    kCsfc = np.zeros((fdim,), dtype=INT)
    kCsfv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCsf
        c = -1
        for i in range(m):
            for k1 in range(m1):
                fAurBu = integral_ff(i, k1, u1tx, u1rx, u2tx, u2rx, u1txf, u1rxf, u2txf, u2rxf)
                fAvrBw = integral_ff(i, k1, v1tx, v1rx, v2tx, v2rx, w1txf, w1rxf, w2txf, w2rxf)
                fAwrBv = integral_ff(i, k1, w1tx, w1rx, w2tx, w2rx, v1txf, v1rxf, v2txf, v2rxf)
                fAwrBw = integral_ff(i, k1, w1tx, w1rx, w2tx, w2rx, w1txf, w1rxf, w2txf, w2rxf)

                for j in range(n):
                    gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fx(j, eta, w1ty, w1ry, w2ty, w2ry)

                    for l1 in range(n1):
                        row = row0 + num1*(j*m + i)
                        col = col0 + num1*(l1*m1 + k1)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        sBu = calc_f(l1, -1., u1tyf, u1ryf, u2tyf, u2ryf)
                        sBv = calc_f(l1, -1., v1tyf, v1ryf, v2tyf, v2ryf)
                        sBw = calc_f(l1, -1., w1tyf, w1ryf, w2tyf, w2ryf)
                        sBweta = calc_fx(l1, -1., w1tyf, w1ryf, w2tyf, w2ryf)

                        c += 1
                        kCsfr[c] = row+0
                        kCsfc[c] = col+0
                        kCsfv[c] += -0.5*a*fAurBu*gAu*kt*sBu
                        c += 1
                        kCsfr[c] = row+1
                        kCsfc[c] = col+2
                        kCsfv[c] += -0.5*a*fAvrBw*gAv*kt*sBw
                        c += 1
                        kCsfr[c] = row+2
                        kCsfc[c] = col+1
                        kCsfv[c] += 0.5*a*fAwrBv*gAw*kt*sBv
                        c += 1
                        kCsfr[c] = row+2
                        kCsfc[c] = col+2
                        kCsfv[c] += -2*a*fAwrBw*gAweta*kr*sBweta/(b*bf)

    kCsf = coo_matrix((kCsfv, (kCsfr, kCsfc)), shape=(size, size))

    return kCsf


def fkCff(double kt, double kr, double a, double bf, int m1, int n1,
          double u1txf, double u1rxf, double u2txf, double u2rxf,
          double v1txf, double v1rxf, double v2txf, double v2rxf,
          double w1txf, double w1rxf, double w2txf, double w2rxf,
          double u1tyf, double u1ryf, double u2tyf, double u2ryf,
          double v1tyf, double v1ryf, double v2tyf, double v2ryf,
          double w1tyf, double w1ryf, double w2tyf, double w2ryf,
          int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col

    cdef np.ndarray[cINT, ndim=1] kCffr, kCffc
    cdef np.ndarray[cDOUBLE, ndim=1] kCffv

    cdef double rAurBu, rAvrBv, rAwrBw
    cdef double sAu, sBu, sAv, sBv, sAw, sBw, sAweta, sBweta

    fdim = 3*m1*n1*m1*n1

    kCffr = np.zeros((fdim,), dtype=INT)
    kCffc = np.zeros((fdim,), dtype=INT)
    kCffv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:

        # kCff
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):
                rAurBu = integral_ff(i1, k1, u1txf, u1rxf, u2txf, u2rxf, u1txf, u1rxf, u2txf, u2rxf)
                rAvrBv = integral_ff(i1, k1, v1txf, v1rxf, v2txf, v2rxf, v1txf, v1rxf, v2txf, v2rxf)
                rAwrBw = integral_ff(i1, k1, w1txf, w1rxf, w2txf, w2rxf, w1txf, w1rxf, w2txf, w2rxf)

                for j1 in range(n1):
                    sAu = calc_f(j1, -1., u1tyf, u1ryf, u2tyf, u2ryf)
                    sAv = calc_f(j1, -1., v1tyf, v1ryf, v2tyf, v2ryf)
                    sAw = calc_f(j1, -1., w1tyf, w1ryf, w2tyf, w2ryf)
                    sAweta = calc_fx(j1, -1., w1tyf, w1ryf, w2tyf, w2ryf)

                    for l1 in range(n1):
                        row = row0 + num1*(j1*m1 + i1)
                        col = col0 + num1*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        sBu = calc_f(l1, -1., u1tyf, u1ryf, u2tyf, u2ryf)
                        sBv = calc_f(l1, -1., v1tyf, v1ryf, v2tyf, v2ryf)
                        sBw = calc_f(l1, -1., w1tyf, w1ryf, w2tyf, w2ryf)
                        sBweta = calc_fx(l1, -1., w1tyf, w1ryf, w2tyf, w2ryf)

                        c += 1
                        kCffr[c] = row+0
                        kCffc[c] = col+0
                        kCffv[c] += 0.5*a*kt*rAurBu*sAu*sBu
                        c += 1
                        kCffr[c] = row+1
                        kCffc[c] = col+1
                        kCffv[c] += 0.5*a*kt*rAvrBv*sAv*sBv
                        c += 1
                        kCffr[c] = row+2
                        kCffc[c] = col+2
                        kCffv[c] += 0.5*a*(kt*rAwrBw*sAw*sBw + 4*kr*rAwrBw*sAweta*sBweta/(bf*bf))

    kCff = coo_matrix((kCffv, (kCffr, kCffc)), shape=(size, size))

    return kCff

