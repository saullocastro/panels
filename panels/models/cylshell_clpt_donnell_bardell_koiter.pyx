#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from __future__ import division

from libc.stdlib cimport malloc, free

from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np


cdef extern from 'bardell_functions.hpp':
    double calc_f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double calc_fx(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil
    double calc_fxx(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil

cdef extern from 'legendre_gauss_quadrature.hpp':
    void leggauss_quad(int n, double *points, double* weights) nogil


ctypedef np.double_t cDOUBLE
DOUBLE = np.float64
ctypedef np.int64_t cINT
INT = np.int64

cdef int DOF = 3


def strain(double xi, double eta, object shell,
        np.ndarray[cDOUBLE, ndim=1] c,
        np.ndarray[cDOUBLE, ndim=1] ejkj,
        int NLgeom=False):
    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    cdef int i, j, m, n, ind
    cdef double wxi, weta, a, b, r
    s = shell
    a = s.a
    b = s.b
    m = s.m
    n = s.n
    r = s.r
    wxi = 0
    weta = 0
    if NLgeom == 1:
        for j in range(n):
            gAueta = calc_fx(j, eta, s.u1ty, s.u1ry, s.u2ty, s.u2ry)
            gAv = calc_f(j, eta, s.v1ty, s.v1ry, s.v2ty, s.v2ry)
            gAw = calc_f(j, eta, s.w1ty, s.w1ry, s.w2ty, s.w2ry)
            gAweta = calc_fx(j, eta, s.w1ty, s.w1ry, s.w2ty, s.w2ry)
            for i in range(m):
                fAu = calc_f(i, xi, s.u1tx, s.u1rx, s.u2tx, s.u2rx)
                fAvxi = calc_fx(i, xi, s.v1tx, s.v1rx, s.v2tx, s.v2rx)
                fAw = calc_f(i, xi, s.w1tx, s.w1rx, s.w2tx, s.w2rx)
                fAwxi = calc_fx(i, xi, s.w1tx, s.w1rx, s.w2tx, s.w2rx)
                ind = DOF*(j*m + i)
                wxi += c[ind+2]*fAwxi*gAw
                weta += c[ind+2]*fAw*gAweta
    for i in range(6):
        ejkj[i] = 0
    for j in range(n):
        gAu = calc_f(j, eta, s.u1ty, s.u1ry, s.u2ty, s.u2ry)
        gAv = calc_f(j, eta, s.v1ty, s.v1ry, s.v2ty, s.v2ry)
        gAw = calc_f(j, eta, s.w1ty, s.w1ry, s.w2ty, s.w2ry)
        gAueta = calc_fx(j, eta, s.u1ty, s.u1ry, s.u2ty, s.u2ry)
        gAveta = calc_fx(j, eta, s.v1ty, s.v1ry, s.v2ty, s.v2ry)
        gAweta = calc_fx(j, eta, s.w1ty, s.w1ry, s.w2ty, s.w2ry)
        gAwetaeta = calc_fxx(j, eta, s.w1ty, s.w1ry, s.w2ty, s.w2ry)
        for i in range(m):
            fAu = calc_f(i, xi, s.u1tx, s.u1rx, s.u2tx, s.u2rx)
            fAv = calc_f(i, xi, s.v1tx, s.v1rx, s.v2tx, s.v2rx)
            fAw = calc_f(i, xi, s.w1tx, s.w1rx, s.w2tx, s.w2rx)
            fAuxi = calc_fx(i, xi, s.u1tx, s.u1rx, s.u2tx, s.u2rx)
            fAvxi = calc_fx(i, xi, s.v1tx, s.v1rx, s.v2tx, s.v2rx)
            fAwxi = calc_fx(i, xi, s.w1tx, s.w1rx, s.w2tx, s.w2rx)
            fAwxixi = calc_fxx(i, xi, s.w1tx, s.w1rx, s.w2tx, s.w2rx)
            ind = DOF*(j*m + i)
            ejkj[0] += c[ind+0]*(2/a)*fAuxi*gAu + 0.5*c[ind+2]*(2/a)*fAwxi*gAw*(2/a)*wxi
            ejkj[1] += c[ind+1]*(2/b)*fAv*gAveta + 1/r*c[ind+2]*fAw*gAw + 0.5*c[ind+2]*(2/b)*fAw*gAweta*(2/b)*weta
            ejkj[2] += c[ind+0]*(2/b)*fAu*gAueta + c[ind+1]*(2/a)*fAvxi*gAv + c[ind+2]*(2/b)*weta*(2/a)*fAwxi*gAw + c[ind+2]*(2/a)*wxi*(2/b)*fAw*gAweta
            ejkj[3] += -c[ind+2]*(2/a*2/a)*fAwxixi*gAw
            ejkj[4] += -c[ind+2]*(2/b*2/b)*fAw*gAwetaeta
            ejkj[5] += -2*c[ind+2]*(2/a)*fAwxi*(2/b)*gAweta


def integrand_eAkA(
        double xi, double eta,
        np.ndarray[cDOUBLE, ndim=1] c,
        object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double a, b, r, wxi, weta
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, j, iA, iB, jA, jB, ind1, ind2
    cdef double xi1, xi2, eta1, eta2

    cdef np.ndarray[cDOUBLE, ndim=2] eAkA

    cdef double fAu, fAv, fAw, fAuxi, fAvxi, fAwxi, fAwxixi
    cdef double gAu, gAv, gAw, gAueta, gAveta, gAweta, gAwetaeta
    cdef double fBu, fBv, fBw, fBuxi, fBvxi, fBwxi, fBwxixi
    cdef double gBu, gBv, gBw, gBueta, gBveta, gBweta, gBwetaeta
    cdef double fCu, fCv, fCw, fCuxi, fCvxi, fCwxi, fCwxixi
    cdef double gCu, gCv, gCw, gCueta, gCveta, gCweta, gCwetaeta

    cdef np.ndarray[cDOUBLE, ndim=1] xis, etas, weights_xi, weights_eta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    r = shell.r
    m = shell.m
    n = shell.n
    u1tx = shell.u1tx; u1rx = shell.u1rx; u2tx = shell.u2tx; u2rx = shell.u2rx
    v1tx = shell.v1tx; v1rx = shell.v1rx; v2tx = shell.v2tx; v2rx = shell.v2rx
    w1tx = shell.w1tx; w1rx = shell.w1rx; w2tx = shell.w2tx; w2rx = shell.w2rx
    u1ty = shell.u1ty; u1ry = shell.u1ry; u2ty = shell.u2ty; u2ry = shell.u2ry
    v1ty = shell.v1ty; v1ry = shell.v1ry; v2ty = shell.v2ty; v2ry = shell.v2ry
    w1ty = shell.w1ty; w1ry = shell.w1ry; w2ty = shell.w2ty; w2ry = shell.w2ry

    eAkA = np.zeros((6, DOF*m*n), dtype=DOUBLE)

    with nogil:
        wxi = 0
        weta = 0
        if NLgeom == 1:
            for iA in range(m):
                fAu = calc_f(iA, xi, u1tx, u1rx, u2tx, u2rx)
                fAvxi = calc_fx(iA, xi, v1tx, v1rx, v2tx, v2rx)
                fAw = calc_f(iA, xi, w1tx, w1rx, w2tx, w2rx)
                fAwxi = calc_fx(iA, xi, w1tx, w1rx, w2tx, w2rx)
                for jA in range(n):
                    gAueta = calc_fx(jA, eta, u1ty, u1ry, u2ty, u2ry)
                    gAv = calc_f(jA, eta, v1ty, v1ry, v2ty, v2ry)
                    gAw = calc_f(jA, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fx(jA, eta, w1ty, w1ry, w2ty, w2ry)

                    ind1 = col0 + DOF*(jA*m + iA)

                    wxi += c[ind1+2]*fAwxi*gAw
                    weta += c[ind1+2]*fAw*gAweta

        # computing eA

        for iA in range(m):
            fAu = calc_f(iA, xi, u1tx, u1rx, u2tx, u2rx)
            fAuxi = calc_fx(iA, xi, u1tx, u1rx, u2tx, u2rx)
            fAv = calc_f(iA, xi, v1tx, v1rx, v2tx, v2rx)
            fAvxi = calc_fx(iA, xi, v1tx, v1rx, v2tx, v2rx)
            fAw = calc_f(iA, xi, w1tx, w1rx, w2tx, w2rx)
            fAwxi = calc_fx(iA, xi, w1tx, w1rx, w2tx, w2rx)
            fAwxixi = calc_fxx(iA, xi, w1tx, w1rx, w2tx, w2rx)

            for jA in range(n):
                gAu = calc_f(jA, eta, u1ty, u1ry, u2ty, u2ry)
                gAueta = calc_fx(jA, eta, u1ty, u1ry, u2ty, u2ry)
                gAv = calc_f(jA, eta, v1ty, v1ry, v2ty, v2ry)
                gAveta = calc_fx(jA, eta, v1ty, v1ry, v2ty, v2ry)
                gAw = calc_f(jA, eta, w1ty, w1ry, w2ty, w2ry)
                gAweta = calc_fx(jA, eta, w1ty, w1ry, w2ty, w2ry)
                gAwetaeta = calc_fxx(jA, eta, w1ty, w1ry, w2ty, w2ry)

                ind1 = DOF*(jA*m + iA)

                eAkA[0, ind1+0] = 2*fAuxi*gAu/a
                eAkA[0, ind1+2] = 4*fAwxi*gAw*wxi/(a*a)
                eAkA[1, ind1+1] = 2*fAv*gAveta/b
                eAkA[1, ind1+2] = fAw*gAw/r + 4*fAw*gAweta*weta/(b*b)
                eAkA[2, ind1+0] = 2*fAu*gAueta/b
                eAkA[2, ind1+1] = 2*fAvxi*gAv/a
                eAkA[2, ind1+2] = 4*(fAw*gAweta*wxi + fAwxi*gAw*weta)/(a*b)
                eAkA[3, ind1+2] = -4*fAwxixi*gAw/(a*a)
                eAkA[4, ind1+2] = -4*fAw*gAwetaeta/(b*b)
                eAkA[5, ind1+2] = -8*fAwxi*gAweta/(a*b)
    return eAkA


def integrand_eAB(double xi, double eta,
        object shell,
        int size, int row0, int col0, int nx, int ny):
    cdef double a, b, r
    cdef int m, n
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, j, iA, iB, jA, jB, ind1, ind2

    cdef np.ndarray[cDOUBLE, ndim=3] eAB

    cdef double fAw, fAuxi, fAvxi, fAwxi
    cdef double gAw, gAueta, gAveta, gAweta
    cdef double fBw, fBuxi, fBvxi, fBwxi
    cdef double gBw, gBueta, gBveta, gBweta

    cdef np.ndarray[cDOUBLE, ndim=1] xis, etas, weights_xi, weights_eta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    r = shell.r
    m = shell.m
    n = shell.n
    w1tx = shell.w1tx; w1rx = shell.w1rx; w2tx = shell.w2tx; w2rx = shell.w2rx
    w1ty = shell.w1ty; w1ry = shell.w1ry; w2ty = shell.w2ty; w2ry = shell.w2ry

    eAB = np.zeros((3, DOF*m*n, DOF*m*n), dtype=DOUBLE)

    with nogil:
        for iA in range(m):
            fAw = calc_f(iA, xi, w1tx, w1rx, w2tx, w2rx)
            fAwxi = calc_fx(iA, xi, w1tx, w1rx, w2tx, w2rx)

            for jA in range(n):
                gAw = calc_f(jA, eta, w1ty, w1ry, w2ty, w2ry)
                gAweta = calc_fx(jA, eta, w1ty, w1ry, w2ty, w2ry)

                ind1 = DOF*(jA*m + iA)

                for iB in range(m):
                    fBw = calc_f(iB, xi, w1tx, w1rx, w2tx, w2rx)
                    fBwxi = calc_fx(iB, xi, w1tx, w1rx, w2tx, w2rx)

                    for jB in range(n):
                        gBw = calc_f(jB, eta, w1ty, w1ry, w2ty, w2ry)
                        gBweta = calc_fx(jB, eta, w1ty, w1ry, w2ty, w2ry)

                        ind2 = DOF*(jB*m + iB)

                        eAB[0, ind1+2, ind2+2] = 4*fAwxi*fBwxi*gAw*gBw/(a*a)
                        eAB[1, ind1+2, ind2+2] = 4*fAw*fBw*gAweta*gBweta/(b*b)
                        eAB[2, ind1+2, ind2+2] = 4*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta)/(a*b)

    return eAB
