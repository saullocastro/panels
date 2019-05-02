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


def fkCf(object bay, object s, int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef double fAuxifBuxi, fAuxifBwxixi, fAuxifBwxi, fAwxixifBuxi
    cdef double fAwxifBuxi, fAwxifBwxi, fAwxifBwxixi, fAwxixifBwxi
    cdef double fAwxixifBwxixi
    cdef double gAu, gBu, gAw, gBw, gAweta, gBweta
    cdef double a, b, ys, bf, dbf, E1, F1, S1, Jxx
    cdef int m, n

    cdef np.ndarray[cINT, ndim=1] kCfr, kCfc
    cdef np.ndarray[cDOUBLE, ndim=1] kCfv

    cdef double x1u, x2u
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y2u
    cdef double y1w, y1wr, y2w, y2wr

    a = bay.a
    b = bay.b
    m = bay.m
    n = bay.n
    x1u = bay.x1u; x2u = bay.x2u
    x1w = bay.x1w; x1wr = bay.x1wr; x2w = bay.x2w; x2wr = bay.x2wr
    y1u = bay.y1u; y2u = bay.y2u
    y1w = bay.y1w; y1wr = bay.y1wr; y2w = bay.y2w; y2wr = bay.y2wr

    ys = s.ys
    bf = s.bf
    dbf = s.dbf
    E1 = s.E1
    F1 = s.F1
    S1 = s.S1
    Jxx = s.Jxx

    eta = 2*ys/b - 1.

    fdim = 4*m*n*m*n

    kCfr = np.zeros((fdim,), dtype=INT)
    kCfc = np.zeros((fdim,), dtype=INT)
    kCfv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kCf
        c = -1
        for i in range(m):
            for k in range(m):

                #FIXME do the numerical integrations
                #NOTE THIS IS NOT WORKING
                fAuxifBuxi = 0 #integral_fxfx(i, k, x1u, x2u, x1u, x2u)
                fAuxifBwxixi = 0 #integral_fxfxx(i, k, x1u, x2u, x1w, x1wr, x2w, x2wr)
                fAuxifBwxi = 0 #integral_fxfx(i, k, x1u, x2u, x1w, x1wr, x2w, x2wr)
                fAwxixifBuxi = 0 #integral_fxfxx(k, i, x1u, x2u, x1w, x1wr, x2w, x2wr)
                fAwxifBuxi = 0 #integral_fxfx(i, k, x1w, x1wr, x2w, x2wr, x1u, x2u)
                fAwxifBwxi = 0 #integral_fxfx(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxixi = 0 #integral_fxfxx(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxixifBwxi = 0 #integral_fxfxx(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxixifBwxixi = 0 #integral_fxxfxx(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAu = fuv(j, eta, y1u, y2u)
                    gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fw_x(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = fuv(l, eta, y1u, y2u)
                        gBw = fw(l, eta, y1w, y1wr, y2w, y2wr)
                        gBweta = fw_x(l, eta, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kCfr[c] = row+0
                        kCfc[c] = col+0
                        kCfv[c] += 2*E1*bf*fAuxifBuxi*gAu*gBu/a
                        c += 1
                        kCfr[c] = row+0
                        kCfc[c] = col+2
                        kCfv[c] += 0.5*a*bf*(8*E1*df*fAuxifBwxixi*gAu*gBw/(a*a*a) - 8*S1*fAuxifBwxi*gAu*gBweta/((a*a)*b))
                        c += 1
                        kCfr[c] = row+2
                        kCfc[c] = col+0
                        kCfv[c] += bf*gBu*(4*E1*df*fAwxixifBuxi*gAw/(a*a) - 4*S1*fAwxifBuxi*gAweta/(a*b))
                        c += 1
                        kCfr[c] = row+2
                        kCfc[c] = col+2
                        kCfv[c] += 0.5*a*bf*(-4*gBweta*(-4*Jxx*fAwxifBwxi*gAweta/(a*b) + 4*S1*df*fAwxixifBwxi*gAw/(a*a))/(a*b) - 4*gBw*(4*S1*df*fAwxifBwxixi*gAweta/(a*b) + fAwxixifBwxixi*gAw*(-4*E1*(df*df) - 4*F1)/(a*a))/(a*a))

    kCf = coo_matrix((kCfv, (kCfr, kCfc)), shape=(size, size))

    return kCf


def fkGf(double ys, double Fx, double a, double b, double bf, int m, int n,
         double x1w, double x1wr, double x2w, double x2wr,
         double y1w, double y1wr, double y2w, double y2wr,
         int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef np.ndarray[cINT, ndim=1] kGfr, kGfc
    cdef np.ndarray[cDOUBLE, ndim=1] kGfv

    cdef double fAwxifBwxi, gAw, gBw

    eta = 2*ys/b - 1.

    fdim = 1*m*n*m*n

    kGfr = np.zeros((fdim,), dtype=INT)
    kGfc = np.zeros((fdim,), dtype=INT)
    kGfv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kGf
        c = -1
        for i in range(m):
            for k in range(m):
                fAwxifBwxi = integral_fxfx(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):
                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBw = fw(l, eta, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kGfr[c] = row+2
                        kGfc[c] = col+2
                        kGfv[c] += 2*Fx*fAwxifBwxi*gAw*gBw/a

    kGf = coo_matrix((kGfv, (kGfr, kGfc)), shape=(size, size))

    return kGf


def fkMf(object bay, object s,
        #double ys, double mu, double h, double hb, double hf, double a,
         #double b, double bf, double df,
         #int m, int n,
         int size, int row0, int col0):
    cdef double fAufBu, fAufBwxi, fAvfBv, fAvfBw, fAwxifBu, fAwfBv, fAwfBw
    cdef double fAwxifBwxi
    cdef double gAu, gBu, gAv, gBv, gAw, gBw, gAweta, gBweta

    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef np.ndarray[cINT, ndim=1] kMfr, kMfc
    cdef np.ndarray[cDOUBLE, ndim=1] kMfv

    cdef double x1u, x2u
    cdef double x1v, x2v
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y2u
    cdef double y1v, y2v
    cdef double y1w, y1wr, y2w, y2wr

    eta = 2*ys/b - 1.

    fdim = 7*m*n*m*n

    kMfr = np.zeros((fdim,), dtype=INT)
    kMfc = np.zeros((fdim,), dtype=INT)
    kMfv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kMf
        c = -1
        for i in range(m):
            for k in range(m):

                #FIXME do the integrations
                #NOTE this is not working
                fAufBu = 0 #integral_ff(i, k, x1u, x2u, x1u, x2u)
                fAufBwxi = 0 #integral_ffx(i, k, x1u, x2u, x1w, x1wr, x2w, x2wr)
                fAvfBv = 0 #integral_ff(i, k, x1v, x2v, v2rx, x1v, x2v, v2rx)
                fAvfBw = 0 #integral_ff(i, k, x1v, x2v, v2rx, x1w, x1wr, x2w, x2wr)
                fAwxifBu = 0 #integral_ffx(k, i, x1u, x2u, x1w, x1wr, x2w, x2wr)
                fAwfBv = 0 #integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1v, x2v, v2rx)
                fAwfBw = 0 #integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = 0 #integral_fxfx(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):

                    gAu = fuv(j, eta, y1u, y2u)
                    gAv = fuv(j, eta, y1v, y2v)
                    gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fw_x(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):

                        row = row0 + num*(j*m + i)
                        col = col0 + num*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = fuv(l, eta, y1u, y2u)
                        gBv = fuv(l, eta, y1v, y2v)
                        gBw = fw(l, eta, y1w, y1wr, y2w, y2wr)
                        gBweta = fw_x(l, eta, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kMfr[c] = row+0
                        kMfc[c] = col+0
                        kMfv[c] += 0.5*a*bf*fAufBu*gAu*gBu*hf*mu
                        c += 1
                        kMfr[c] = row+0
                        kMfc[c] = col+2
                        kMfv[c] += 2*bf*df*fAufBwxi*gAu*gBw*hf*mu
                        c += 1
                        kMfr[c] = row+1
                        kMfc[c] = col+1
                        kMfv[c] += 0.5*a*bf*fAvfBv*gAv*gBv*hf*mu
                        c += 1
                        kMfr[c] = row+1
                        kMfc[c] = col+2
                        kMfv[c] += 2*a*bf*df*fAvfBw*gAv*gBweta*hf*mu/b
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+0
                        kMfv[c] += 2*bf*df*fAwxifBu*gAw*gBu*hf*mu
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+1
                        kMfv[c] += 2*a*bf*df*fAwfBv*gAweta*gBv*hf*mu/b
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+2
                        kMfv[c] += 0.166666666666667*bf*hf*mu*((a*a)*fAwfBw*(3*(b*b)*gAw*gBw + gAweta*gBweta*(4*(bf*bf) + 6*bf*(h + 2*hb) + 3*(h + 2*hb)**2)) + (b*b)*fAwxifBwxi*gAw*gBw*(4*(bf*bf) + 6*bf*(h + 2*hb) + 3*(h + 2*hb)**2))/(a*(b*b))

    kMf = coo_matrix((kMfv, (kMfr, kMfc)), shape=(size, size))

    return kMf

