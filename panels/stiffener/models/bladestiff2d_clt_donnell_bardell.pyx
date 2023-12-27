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
cdef int DOF1 = 3


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil


def fkCss(double kt, double kr, double ys, double a, double b, int m, int n,
          double x1u, double x1ur, double x2u, double x2ur,
          double x1v, double x1vr, double x2v, double x2vr,
          double x1w, double x1wr, double x2w, double x2wr,
          double y1u, double y1ur, double y2u, double y2ur,
          double y1v, double y1vr, double y2v, double y2vr,
          double y1w, double y1wr, double y2w, double y2wr,
          int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef long [:] kCssr, kCssc
    cdef double [:] kCssv

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
                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

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
          double x1u, double x1ur, double x2u, double x2ur,
          double x1v, double x1vr, double x2v, double x2vr,
          double x1w, double x1wr, double x2w, double x2wr,
          double y1u, double y1ur, double y2u, double y2ur,
          double y1v, double y1vr, double y2v, double y2vr,
          double y1w, double y1wr, double y2w, double y2wr,
          double x1uf, double x1urf, double x2uf, double x2urf,
          double x1vf, double x1vrf, double x2vf, double x2vrf,
          double x1wf, double x1wrf, double x2wf, double x2wrf,
          double y1uf, double y1urf, double y2uf, double y2urf,
          double y1vf, double y1vrf, double y2vf, double y2vrf,
          double y1wf, double y1wrf, double y2wf, double y2wrf,
          int size, int row0, int col0):
    cdef int i, j, k1, l1, c, row, col
    cdef double eta

    cdef long [:] kCsfr, kCsfc
    cdef double [:] kCsfv

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
                fAurBu = integral_ff(i, k1, x1u, x1ur, x2u, x2ur, x1uf, x1urf, x2uf, x2urf)
                fAvrBw = integral_ff(i, k1, x1v, x1vr, x2v, x2vr, x1wf, x1wrf, x2wf, x2wrf)
                fAwrBv = integral_ff(i, k1, x1w, x1wr, x2w, x2wr, x1vf, x1vrf, x2vf, x2vrf)
                fAwrBw = integral_ff(i, k1, x1w, x1wr, x2w, x2wr, x1wf, x1wrf, x2wf, x2wrf)

                for j in range(n):
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                    for l1 in range(n1):
                        row = row0 + DOF1*(j*m + i)
                        col = col0 + DOF1*(l1*m1 + k1)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        sBu = f(l1, -1., y1uf, y1urf, y2uf, y2urf)
                        sBv = f(l1, -1., y1vf, y1vrf, y2vf, y2vrf)
                        sBw = f(l1, -1., y1wf, y1wrf, y2wf, y2wrf)
                        sBweta = fp(l1, -1., y1wf, y1wrf, y2wf, y2wrf)

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
          double x1uf, double x1urf, double x2uf, double x2urf,
          double x1vf, double x1vrf, double x2vf, double x2vrf,
          double x1wf, double x1wrf, double x2wf, double x2wrf,
          double y1uf, double y1urf, double y2uf, double y2urf,
          double y1vf, double y1vrf, double y2vf, double y2vrf,
          double y1wf, double y1wrf, double y2wf, double y2wrf,
          int size, int row0, int col0):
    cdef int i1, k1, j1, l1, c, row, col

    cdef long [:] kCffr, kCffc
    cdef double [:] kCffv

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
                rAurBu = integral_ff(i1, k1, x1uf, x1urf, x2uf, x2urf, x1uf, x1urf, x2uf, x2urf)
                rAvrBv = integral_ff(i1, k1, x1vf, x1vrf, x2vf, x2vrf, x1vf, x1vrf, x2vf, x2vrf)
                rAwrBw = integral_ff(i1, k1, x1wf, x1wrf, x2wf, x2wrf, x1wf, x1wrf, x2wf, x2wrf)

                for j1 in range(n1):
                    sAu = f(j1, -1., y1uf, y1urf, y2uf, y2urf)
                    sAv = f(j1, -1., y1vf, y1vrf, y2vf, y2vrf)
                    sAw = f(j1, -1., y1wf, y1wrf, y2wf, y2wrf)
                    sAweta = fp(j1, -1., y1wf, y1wrf, y2wf, y2wrf)

                    for l1 in range(n1):
                        row = row0 + DOF1*(j1*m1 + i1)
                        col = col0 + DOF1*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        sBu = f(l1, -1., y1uf, y1urf, y2uf, y2urf)
                        sBv = f(l1, -1., y1vf, y1vrf, y2vf, y2vrf)
                        sBw = f(l1, -1., y1wf, y1wrf, y2wf, y2wrf)
                        sBweta = fp(l1, -1., y1wf, y1wrf, y2wf, y2wrf)

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

