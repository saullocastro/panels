#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np

DOUBLE = np.float64
INT = long
cdef int DOF = 3


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fppfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil


def fkCf(double ys, double a, double b, double bf, double df, double E1, double F1,
         double S1, double Jxx, int m, int n,
         double x1u, double x1ur, double x2u, double x2ur,
         double x1w, double x1wr, double x2w, double x2wr,
         double y1u, double y1ur, double y2u, double y2ur,
         double y1w, double y1wr, double y2w, double y2wr,
         int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef double fAuxifBuxi, fAuxifBwxixi, fAuxifBwxi, fAwxixifBuxi
    cdef double fAwxifBuxi, fAwxifBwxi, fAwxifBwxixi, fAwxixifBwxi
    cdef double fAwxixifBwxixi
    cdef double gAu, gBu, gAw, gBw, gAweta, gBweta

    cdef long [:] k0fr, k0fc
    cdef double [:] k0fv

    eta = 2*ys/b - 1.

    fdim = 4*m*n*m*n

    k0fr = np.zeros((fdim,), dtype=INT)
    k0fc = np.zeros((fdim,), dtype=INT)
    k0fv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # k0f
        c = -1
        for i in range(m):
            for k in range(m):
                fAuxifBuxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAuxifBwxixi = integral_fpfpp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAuxifBwxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwxixifBuxi = integral_fpfpp(k, i, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwxifBuxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1u, x1ur, x2u, x2ur)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxixi = integral_fpfpp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxixifBwxi = integral_fpfpp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxixifBwxixi = integral_fppfpp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = f(l, eta, y1u, y1ur, y2u, y2ur)
                        gBw = f(l, eta, y1w, y1wr, y2w, y2wr)
                        gBweta = fp(l, eta, y1w, y1wr, y2w, y2wr)

                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+0
                        k0fv[c] += 2*E1*bf*fAuxifBuxi*gAu*gBu/a
                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+2
                        k0fv[c] += 0.5*a*bf*(8*E1*df*fAuxifBwxixi*gAu*gBw/(a*a*a) - 8*S1*fAuxifBwxi*gAu*gBweta/((a*a)*b))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+0
                        k0fv[c] += bf*gBu*(4*E1*df*fAwxixifBuxi*gAw/(a*a) - 4*S1*fAwxifBuxi*gAweta/(a*b))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+2
                        k0fv[c] += 0.5*a*bf*(-4*gBweta*(-4*Jxx*fAwxifBwxi*gAweta/(a*b) + 4*S1*df*fAwxixifBwxi*gAw/(a*a))/(a*b) - 4*gBw*(4*S1*df*fAwxifBwxixi*gAweta/(a*b) + fAwxixifBwxixi*gAw*(-4*E1*(df*df) - 4*F1)/(a*a))/(a*a))

    k0f = coo_matrix((k0fv, (k0fr, k0fc)), shape=(size, size))

    return k0f


def fkGf(double ys, double Fx, double a, double b, double bf, int m, int n,
          double x1w, double x1wr, double x2w, double x2wr,
          double y1w, double y1wr, double y2w, double y2wr,
          int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef long [:] kGfr, kGfc
    cdef double [:] kGfv

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
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):
                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBw = f(l, eta, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kGfr[c] = row+2
                        kGfc[c] = col+2
                        kGfv[c] += 2*Fx*fAwxifBwxi*gAw*gBw/a

    kGf = coo_matrix((kGfv, (kGfr, kGfc)), shape=(size, size))

    return kGf


def fkMf(double ys, double rho, double h, double hb, double hf, double a,
         double b, double bf, double df,
         int m, int n,
         double x1u, double x1ur, double x2u, double x2ur,
         double x1v, double x1vr, double x2v, double x2vr,
         double x1w, double x1wr, double x2w, double x2wr,
         double y1u, double y1ur, double y2u, double y2ur,
         double y1v, double y1vr, double y2v, double y2vr,
         double y1w, double y1wr, double y2w, double y2wr,
         int size, int row0, int col0):
    cdef double fAufBu, fAufBwxi, fAvfBv, fAvfBw, fAwxifBu, fAwfBv, fAwfBw
    cdef double fAwxifBwxi
    cdef double gAu, gBu, gAv, gBv, gAw, gBw, gAweta, gBweta

    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef long [:] kMfr, kMfc
    cdef double [:] kMfv

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

                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBwxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBw = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwfBv = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):

                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = f(l, eta, y1u, y1ur, y2u, y2ur)
                        gBv = f(l, eta, y1v, y1vr, y2v, y2vr)
                        gBw = f(l, eta, y1w, y1wr, y2w, y2wr)
                        gBweta = fp(l, eta, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kMfr[c] = row+0
                        kMfc[c] = col+0
                        kMfv[c] += 0.5*a*bf*fAufBu*gAu*gBu*hf*rho
                        c += 1
                        kMfr[c] = row+0
                        kMfc[c] = col+2
                        kMfv[c] += 2*bf*df*fAufBwxi*gAu*gBw*hf*rho
                        c += 1
                        kMfr[c] = row+1
                        kMfc[c] = col+1
                        kMfv[c] += 0.5*a*bf*fAvfBv*gAv*gBv*hf*rho
                        c += 1
                        kMfr[c] = row+1
                        kMfc[c] = col+2
                        kMfv[c] += 2*a*bf*df*fAvfBw*gAv*gBweta*hf*rho/b
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+0
                        kMfv[c] += 2*bf*df*fAwxifBu*gAw*gBu*hf*rho
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+1
                        kMfv[c] += 2*a*bf*df*fAwfBv*gAweta*gBv*hf*rho/b
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+2
                        kMfv[c] += 0.166666666666667*bf*hf*rho*((a*a)*fAwfBw*(3*(b*b)*gAw*gBw + gAweta*gBweta*(4*(bf*bf) + 6*bf*(h + 2*hb) + 3*(h + 2*hb)**2)) + (b*b)*fAwxifBwxi*gAw*gBw*(4*(bf*bf) + 6*bf*(h + 2*hb) + 3*(h + 2*hb)**2))/(a*(b*b))

    kMf = coo_matrix((kMfv, (kMfr, kMfc)), shape=(size, size))

    return kMf
