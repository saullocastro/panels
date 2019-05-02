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
cdef int num1 = 3


def fkCss(double kt, double kr, double ys, double a, double b, int m, int n,
          int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef np.ndarray[cINT, ndim=1] kCssr, kCssc
    cdef np.ndarray[cDOUBLE, ndim=1] kCssv

    cdef double fAufBu, fAvfBv, fAwfBw
    cdef double gAu, gBu, gAv, gBv, gAw, gBw, gAweta, gBweta

    cdef double x1u, x2u
    cdef double x1v, x2v
    cdef double x1w, x1wr, x2w, x2wr,
    cdef double y1u, y2u
    cdef double y1v, y2v
    cdef double y1w, y1wr, y2w, y2wr,

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
                #FIXME integrate numerically
                #NOTE this is not working
                fAufBu = 0#integral_ff(i, k, x1u, x2u, x1u, x2u)
                fAvfBv = 0#integral_ff(i, k, x1v, x2v, x1v, x2v)
                fAwfBw = 0#integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAu = fuv(j, eta, y1u, y2u)
                    gAv = fuv(j, eta, y1v, y2v)
                    gAw = calc_f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = calc_fx(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):
                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = fuv(l, eta, y1u, y2u)
                        gBv = fuv(l, eta, y1v, y2v)
                        gBw = calc_f(l, eta, y1w, y1wr, y2w, y2wr)
                        gBweta = calc_fx(l, eta, y1w, y1wr, y2w, y2wr)

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
          int size, int row0, int col0):
    cdef int i, j, k1, l1, c, row, col
    cdef double eta

    cdef np.ndarray[cINT, ndim=1] kCsfr, kCsfc
    cdef np.ndarray[cDOUBLE, ndim=1] kCsfv

    cdef double fAurBu, fAvrBw, fAwrBv, fAwrBw
    cdef double gAu, gAv, gAw, gAweta, sBu, sBv, sBw, sBweta

    cdef double x1u, x2u
    cdef double x1v, x2v
    cdef double x1w, x1wr, x2w, x2wr,
    cdef double y1u, y2u
    cdef double y1v, y2v
    cdef double y1w, y1wr, y2w, y2wr,
    cdef double x1uf, x2uf
    cdef double x1vf, x2vf
    cdef double x1wf, x1wrf, x2wf, x2wrf,
    cdef double y1uf, y2uf
    cdef double y1vf, y2vf
    cdef double y1wf, y1wrf, y2wf, y2wrf,

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
                #FIXME do the integration numerically
                #NOTE this is not working
                fAurBu = 0#integral_ff(i, k1, x1u, x2u, x1uf, x2uf)
                fAvrBw = 0#integral_ff(i, k1, x1v, x2v, x1wf, x1wrf, x2wf, x2wrf)
                fAwrBv = 0#integral_ff(i, k1, x1w, x1wr, x2w, x2wr, x1vf, x2vf)
                fAwrBw = 0#integral_ff(i, k1, x1w, x1wr, x2w, x2wr, x1wf, x1wrf, x2wf, x2wrf)

                for j in range(n):
                    gAu = fuv(j, eta, y1u, y2u)
                    gAv = fuv(j, eta, y1v, y2v)
                    gAw = calc_f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = calc_fx(j, eta, y1w, y1wr, y2w, y2wr)

                    for l1 in range(n1):
                        row = row0 + num1*(j*m + i)
                        col = col0 + num1*(l1*m1 + k1)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        sBu = fuv(l1, -1., y1uf, y2uf)
                        sBv = fuv(l1, -1., y1vf, y2vf)
                        sBw = calc_f(l1, -1., y1wf, y1wrf, y2wf, y2wrf)
                        sBweta = calc_fx(l1, -1., y1wf, y1wrf, y2wf, y2wrf)

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
          int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col

    cdef np.ndarray[cINT, ndim=1] kCffr, kCffc
    cdef np.ndarray[cDOUBLE, ndim=1] kCffv

    cdef double rAurBu, rAvrBv, rAwrBw
    cdef double sAu, sBu, sAv, sBv, sAw, sBw, sAweta, sBweta

    cdef double x1uf, x2uf
    cdef double x1vf, x2vf
    cdef double x1wf, x1wrf, x2wf, x2wrf,
    cdef double y1uf, y2uf
    cdef double y1vf, y2vf
    cdef double y1wf, y1wrf, y2wf, y2wrf,

    fdim = 3*m1*n1*m1*n1

    kCffr = np.zeros((fdim,), dtype=INT)
    kCffc = np.zeros((fdim,), dtype=INT)
    kCffv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:

        # kCff
        c = -1
        for i1 in range(m1):
            for k1 in range(m1):
                #FIXME not working do integration numerically
                rAurBu = 0 #integral_ff(i1, k1, x1uf, u1rxf, x2uf, u2rxf, x1uf, u1rxf, x2uf, u2rxf)
                rAvrBv = 0 #integral_ff(i1, k1, x1vf, v1rxf, x2vf, v2rxf, x1vf, v1rxf, x2vf, v2rxf)
                rAwrBw = 0 #integral_ff(i1, k1, x1wf, x1wrf, x2wf, x2wrf, x1wf, x1wrf, x2wf, x2wrf)

                for j1 in range(n1):
                    sAu = fuv(j1, -1., y1uf, y2uf)
                    sAv = fuv(j1, -1., y1vf, y2vf)
                    sAw = calc_f(j1, -1., y1wf, y1wrf, y2wf, y2wrf)
                    sAweta = calc_fx(j1, -1., y1wf, y1wrf, y2wf, y2wrf)

                    for l1 in range(n1):
                        row = row0 + num1*(j1*m1 + i1)
                        col = col0 + num1*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        sBu = fuv(l1, -1., y1uf, y2uf)
                        sBv = fuv(l1, -1., y1vf, y2vf)
                        sBw = calc_f(l1, -1., y1wf, y1wrf, y2wf, y2wrf)
                        sBweta = calc_fx(l1, -1., y1wf, y1wrf, y2wf, y2wrf)

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

