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

cdef int num = 3


def fkC_num(np.ndarray[cDOUBLE, ndim=1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double a, b
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, j, k, l, c, row, col, ptx, pty
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66

    cdef np.ndarray[cINT, ndim=1] kCr, kCc
    cdef np.ndarray[cDOUBLE, ndim=1] kCv

    cdef double fAu, fAuxi, fAv, fAvxi, fAw, fAwxi, fAwxixi
    cdef double fBu, fBuxi, fBv, fBvxi, fBw, fBwxi, fBwxixi
    cdef double gAu, gAueta, gAv, gAveta, gAw, gAweta, gAwetaeta
    cdef double gBu, gBueta, gBv, gBveta, gBw, gBweta, gBwetaeta
    cdef double xi, eta, weight
    cdef double wxi, weta

    cdef np.ndarray[cDOUBLE, ndim=1] xis, etas, weights_xi, weights_eta

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef np.ndarray[cDOUBLE, ndim=4] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.asarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = np.ascontiguousarray(Finput)
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        Finput = np.ascontiguousarray(Finput)
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    u1tx = shell.u1tx; u1rx = shell.u1rx; u2tx = shell.u2tx; u2rx = shell.u2rx
    v1tx = shell.v1tx; v1rx = shell.v1rx; v2tx = shell.v2tx; v2rx = shell.v2rx
    w1tx = shell.w1tx; w1rx = shell.w1rx; w2tx = shell.w2tx; w2rx = shell.w2rx
    u1ty = shell.u1ty; u1ry = shell.u1ry; u2ty = shell.u2ty; u2ry = shell.u2ry
    v1ty = shell.v1ty; v1ry = shell.v1ry; v2ty = shell.v2ty; v2ry = shell.v2ry
    w1ty = shell.w1ty; w1ry = shell.w1ry; w2ty = shell.w2ty; w2ry = shell.w2ry

    fdim = 9*m*m*n*n

    xis = np.zeros(nx, dtype=DOUBLE)
    weights_xi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weights_eta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weights_xi[0])
    leggauss_quad(ny, &etas[0], &weights_eta[0])

    kCr = np.zeros((fdim,), dtype=INT)
    kCc = np.zeros((fdim,), dtype=INT)
    kCv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                weight = weights_xi[ptx]*weights_eta[pty]

                wxi = 0
                weta = 0
                if NLgeom == 1:
                    for j in range(n):
                        #TODO put these in a lookup vector
                        gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                        gAweta = calc_fx(j, eta, w1ty, w1ry, w2ty, w2ry)
                        for i in range(m):
                            #TODO put these in a lookup vector
                            fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                            fAwxi = calc_fx(i, xi, w1tx, w1rx, w2tx, w2rx)

                            col = col0 + num*(j*m + i)

                            wxi += cs[col+2]*fAwxi*gAw
                            weta += cs[col+2]*fAw*gAweta

                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            #TODO could assume symmetry
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                A11 = F[0*6 + 0]
                A12 = F[0*6 + 1]
                A16 = F[0*6 + 2]
                A22 = F[1*6 + 1]
                A26 = F[1*6 + 2]
                A66 = F[2*6 + 2]

                B11 = F[0*6 + 3]
                B12 = F[0*6 + 4]
                B16 = F[0*6 + 5]
                B22 = F[1*6 + 4]
                B26 = F[1*6 + 5]
                B66 = F[2*6 + 5]

                D11 = F[3*6 + 3]
                D12 = F[3*6 + 4]
                D16 = F[3*6 + 5]
                D22 = F[4*6 + 4]
                D26 = F[4*6 + 5]
                D66 = F[5*6 + 5]

                # kC
                c = -1
                for i in range(m):
                    fAu = calc_f(i, xi, u1tx, u1rx, u2tx, u2rx)
                    fAuxi = calc_fx(i, xi, u1tx, u1rx, u2tx, u2rx)
                    fAv = calc_f(i, xi, v1tx, v1rx, v2tx, v2rx)
                    fAvxi = calc_fx(i, xi, v1tx, v1rx, v2tx, v2rx)
                    fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                    fAwxi = calc_fx(i, xi, w1tx, w1rx, w2tx, w2rx)
                    fAwxixi = calc_fxx(i, xi, w1tx, w1rx, w2tx, w2rx)

                    for k in range(m):
                        fBu = calc_f(k, xi, u1tx, u1rx, u2tx, u2rx)
                        fBuxi = calc_fx(k, xi, u1tx, u1rx, u2tx, u2rx)
                        fBv = calc_f(k, xi, v1tx, v1rx, v2tx, v2rx)
                        fBvxi = calc_fx(k, xi, v1tx, v1rx, v2tx, v2rx)
                        fBw = calc_f(k, xi, w1tx, w1rx, w2tx, w2rx)
                        fBwxi = calc_fx(k, xi, w1tx, w1rx, w2tx, w2rx)
                        fBwxixi = calc_fxx(k, xi, w1tx, w1rx, w2tx, w2rx)

                        for j in range(n):
                            gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                            gAueta = calc_fx(j, eta, u1ty, u1ry, u2ty, u2ry)
                            gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                            gAveta = calc_fx(j, eta, v1ty, v1ry, v2ty, v2ry)
                            gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                            gAweta = calc_fx(j, eta, w1ty, w1ry, w2ty, w2ry)
                            gAwetaeta = calc_fxx(j, eta, w1ty, w1ry, w2ty, w2ry)

                            for l in range(n):

                                row = row0 + num*(j*m + i)
                                col = col0 + num*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = calc_f(l, eta, u1ty, u1ry, u2ty, u2ry)
                                gBueta = calc_fx(l, eta, u1ty, u1ry, u2ty, u2ry)
                                gBv = calc_f(l, eta, v1ty, v1ry, v2ty, v2ry)
                                gBveta = calc_fx(l, eta, v1ty, v1ry, v2ty, v2ry)
                                gBw = calc_f(l, eta, w1ty, w1ry, w2ty, w2ry)
                                gBweta = calc_fx(l, eta, w1ty, w1ry, w2ty, w2ry)
                                gBwetaeta = calc_fxx(l, eta, w1ty, w1ry, w2ty, w2ry)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(A11*b*fAuxi*fBuxi*gAu*gBu/a + A16*(fAu*fBuxi*gAueta*gBu + fAuxi*fBu*gAu*gBueta) + A66*a*fAu*fBu*gAueta*gBueta/b)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(A12*fAuxi*fBv*gAu*gBveta + A16*b*fAuxi*fBvxi*gAu*gBv/a + A26*a*fAu*fBv*gAueta*gBveta/b + A66*fAu*fBvxi*gAueta*gBv)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(2*A11*b*fAuxi*fBwxi*gAu*gBw*wxi/(a*a) + 2*A12*fAuxi*fBw*gAu*gBweta*weta/b + 2*A16*(fAu*fBwxi*gAueta*gBw*wxi + fAuxi*gAu*(fBw*gBweta*wxi + fBwxi*gBw*weta))/a + 2*A26*a*fAu*fBw*gAueta*gBweta*weta/(b*b) + 2*A66*fAu*gAueta*(fBw*gBweta*wxi + fBwxi*gBw*weta)/b - 2*B11*b*fAuxi*fBwxixi*gAu*gBw/(a*a) - 2*B12*fAuxi*fBw*gAu*gBwetaeta/b - 2*B16*(fAu*fBwxixi*gAueta*gBw + 2*fAuxi*fBwxi*gAu*gBweta)/a - 2*B26*a*fAu*fBw*gAueta*gBwetaeta/(b*b) - 4*B66*fAu*fBwxi*gAueta*gBweta/b)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(A12*fAv*fBuxi*gAveta*gBu + A16*b*fAvxi*fBuxi*gAv*gBu/a + A26*a*fAv*fBu*gAveta*gBueta/b + A66*fAvxi*fBu*gAv*gBueta)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(A22*a*fAv*fBv*gAveta*gBveta/b + A26*(fAv*fBvxi*gAveta*gBv + fAvxi*fBv*gAv*gBveta) + A66*b*fAvxi*fBvxi*gAv*gBv/a)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(2*A12*fAv*fBwxi*gAveta*gBw*wxi/a + 2*A16*b*fAvxi*fBwxi*gAv*gBw*wxi/(a*a) + 2*A22*a*fAv*fBw*gAveta*gBweta*weta/(b*b) + 2*A26*(fAv*gAveta*(fBw*gBweta*wxi + fBwxi*gBw*weta) + fAvxi*fBw*gAv*gBweta*weta)/b + 2*A66*fAvxi*gAv*(fBw*gBweta*wxi + fBwxi*gBw*weta)/a - 2*B12*fAv*fBwxixi*gAveta*gBw/a - 2*B16*b*fAvxi*fBwxixi*gAv*gBw/(a*a) - 2*B22*a*fAv*fBw*gAveta*gBwetaeta/(b*b) - 2*B26*(2*fAv*fBwxi*gAveta*gBweta + fAvxi*fBw*gAv*gBwetaeta)/b - 4*B66*fAvxi*fBwxi*gAv*gBweta/a)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(2*A11*b*fAwxi*fBuxi*gAw*gBu*wxi/(a*a) + 2*A12*fAw*fBuxi*gAweta*gBu*weta/b + 2*A16*(fAw*fBuxi*gAweta*gBu*wxi + fAwxi*gAw*(fBu*gBueta*wxi + fBuxi*gBu*weta))/a + 2*A26*a*fAw*fBu*gAweta*gBueta*weta/(b*b) + 2*A66*fBu*gBueta*(fAw*gAweta*wxi + fAwxi*gAw*weta)/b - 2*B11*b*fAwxixi*fBuxi*gAw*gBu/(a*a) - 2*B12*fAw*fBuxi*gAwetaeta*gBu/b - 2*B16*(2*fAwxi*fBuxi*gAweta*gBu + fAwxixi*fBu*gAw*gBueta)/a - 2*B26*a*fAw*fBu*gAwetaeta*gBueta/(b*b) - 4*B66*fAwxi*fBu*gAweta*gBueta/b)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(2*A12*fAwxi*fBv*gAw*gBveta*wxi/a + 2*A16*b*fAwxi*fBvxi*gAw*gBv*wxi/(a*a) + 2*A22*a*fAw*fBv*gAweta*gBveta*weta/(b*b) + 2*A26*(fAw*gAweta*(fBv*gBveta*wxi + fBvxi*gBv*weta) + fAwxi*fBv*gAw*gBveta*weta)/b + 2*A66*fBvxi*gBv*(fAw*gAweta*wxi + fAwxi*gAw*weta)/a - 2*B12*fAwxixi*fBv*gAw*gBveta/a - 2*B16*b*fAwxixi*fBvxi*gAw*gBv/(a*a) - 2*B22*a*fAw*fBv*gAwetaeta*gBveta/(b*b) - 2*B26*(fAw*fBvxi*gAwetaeta*gBv + 2*fAwxi*fBv*gAweta*gBveta)/b - 4*B66*fAwxi*fBvxi*gAweta*gBv/a)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(4*A11*b*fAwxi*fBwxi*gAw*gBw*(wxi*wxi)/(a*a*a) + 4*A12*weta*wxi*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta)/(a*b) + 4*A16*wxi*(fAw*fBwxi*gAweta*gBw*wxi + fAwxi*gAw*(fBw*gBweta*wxi + 2*fBwxi*gBw*weta))/(a*a) + 4*A22*a*fAw*fBw*gAweta*gBweta*(weta*weta)/(b*b*b) + 4*A26*weta*(fAw*gAweta*(2*fBw*gBweta*wxi + fBwxi*gBw*weta) + fAwxi*fBw*gAw*gBweta*weta)/(b*b) + 4*A66*(fAw*gAweta*wxi + fAwxi*gAw*weta)*(fBw*gBweta*wxi + fBwxi*gBw*weta)/(a*b) - 4*B11*b*gAw*gBw*wxi*(fAwxi*fBwxixi + fAwxixi*fBwxi)/(a*a*a) - 4*B12*(fAw*gBw*(fBwxi*gAwetaeta*wxi + fBwxixi*gAweta*weta) + fBw*gAw*(fAwxi*gBwetaeta*wxi + fAwxixi*gBweta*weta))/(a*b) - 4*B16*(fAw*fBwxixi*gAweta*gBw*wxi + fAwxi*(2*fBwxi*gAw*gBweta*wxi + 2*fBwxi*gAweta*gBw*wxi + fBwxixi*gAw*gBw*weta) + fAwxixi*gAw*(fBw*gBweta*wxi + fBwxi*gBw*weta))/(a*a) - 4*B22*a*fAw*fBw*weta*(gAweta*gBwetaeta + gAwetaeta*gBweta)/(b*b*b) - 4*B26*(fAw*(fBw*wxi*(gAweta*gBwetaeta + gAwetaeta*gBweta) + fBwxi*weta*(2*gAweta*gBweta + gAwetaeta*gBw)) + fAwxi*fBw*weta*(gAw*gBwetaeta + 2*gAweta*gBweta))/(b*b) - 8*B66*(fAw*fBwxi*gAweta*gBweta*wxi + fAwxi*(fBw*gAweta*gBweta*wxi + fBwxi*weta*(gAw*gBweta + gAweta*gBw)))/(a*b) + 4*D11*b*fAwxixi*fBwxixi*gAw*gBw/(a*a*a) + 4*D12*(fAw*fBwxixi*gAwetaeta*gBw + fAwxixi*fBw*gAw*gBwetaeta)/(a*b) + 8*D16*(fAwxi*fBwxixi*gAweta*gBw + fAwxixi*fBwxi*gAw*gBweta)/(a*a) + 4*D22*a*fAw*fBw*gAwetaeta*gBwetaeta/(b*b*b) + 8*D26*(fAw*fBwxi*gAwetaeta*gBweta + fAwxi*fBw*gAweta*gBwetaeta)/(b*b) + 16*D66*fAwxi*fBwxi*gAweta*gBweta/(a*b))

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC


def fkG_num(np.ndarray[cDOUBLE, ndim=1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double a, b
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col, ptx, pty
    cdef double xi, eta, x, y, weight

    cdef np.ndarray[cINT, ndim=1] kGr, kGc
    cdef np.ndarray[cDOUBLE, ndim=1] kGv

    cdef double fAu, fAv, fAw, fAuxi, fAvxi, fAwxi, fAwxixi
    cdef double gAu, gAv, gAw, gAueta, gAveta, gAweta, gAwetaeta
    cdef double gBw, gBweta, fBw, fBwxi

    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double wxi, weta, Nxx, Nyy, Nxy

    cdef np.ndarray[cDOUBLE, ndim=1] xis, etas, weights_xi, weights_eta

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef np.ndarray[cDOUBLE, ndim=4] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    u1tx = shell.u1tx; u1rx = shell.u1rx; u2tx = shell.u2tx; u2rx = shell.u2rx
    v1tx = shell.v1tx; v1rx = shell.v1rx; v2tx = shell.v2tx; v2rx = shell.v2rx
    w1tx = shell.w1tx; w1rx = shell.w1rx; w2tx = shell.w2tx; w2rx = shell.w2rx
    u1ty = shell.u1ty; u1ry = shell.u1ry; u2ty = shell.u2ty; u2ry = shell.u2ry
    v1ty = shell.v1ty; v1ry = shell.v1ry; v2ty = shell.v2ty; v2ry = shell.v2ry
    w1ty = shell.w1ty; w1ry = shell.w1ry; w2ty = shell.w2ty; w2ry = shell.w2ry

    fdim = 1*m*m*n*n

    xis = np.zeros(nx, dtype=DOUBLE)
    weights_xi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weights_eta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weights_xi[0])
    leggauss_quad(ny, &etas[0], &weights_eta[0])

    kGr = np.zeros((fdim,), dtype=INT)
    kGc = np.zeros((fdim,), dtype=INT)
    kGv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                weight = weights_xi[ptx]*weights_eta[pty]

                # Reading laminate constitutive data
                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            #TODO could assume symmetry
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                A11 = F[0*6 + 0]
                A12 = F[0*6 + 1]
                A16 = F[0*6 + 2]
                A22 = F[1*6 + 1]
                A26 = F[1*6 + 2]
                A66 = F[2*6 + 2]

                B11 = F[0*6 + 3]
                B12 = F[0*6 + 4]
                B16 = F[0*6 + 5]
                B22 = F[1*6 + 4]
                B26 = F[1*6 + 5]
                B66 = F[2*6 + 5]

                wxi = 0
                weta = 0
                if NLgeom == 1:
                    for j in range(n):
                        #TODO put these in a lookup vector
                        gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                        gAweta = calc_fx(j, eta, w1ty, w1ry, w2ty, w2ry)
                        for i in range(m):
                            fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                            fAwxi = calc_fx(i, xi, w1tx, w1rx, w2tx, w2rx)

                            col = col0 + num*(j*m + i)

                            wxi += cs[col+2]*fAwxi*gAw
                            weta += cs[col+2]*fAw*gAweta

                # Calculating strain components
                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.
                for j in range(n):
                    #TODO put these in a lookup vector
                    gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAueta = calc_fx(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAveta = calc_fx(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAweta = calc_fx(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAwetaeta = calc_fxx(j, eta, w1ty, w1ry, w2ty, w2ry)

                    for i in range(m):
                        fAu = calc_f(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAv = calc_f(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAuxi = calc_fx(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAvxi = calc_fx(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAwxi = calc_fx(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxixi = calc_fxx(i, xi, w1tx, w1rx, w2tx, w2rx)

                        col = col0 + num*(j*m + i)

                        exx += cs[col+0]*(2/a)*fAuxi*gAu + 0.5*cs[col+2]*(2/a)*fAwxi*gAw*(2/a)*wxi
                        eyy += cs[col+1]*(2/b)*fAv*gAveta + 0.5*cs[col+2]*(2/b)*fAw*gAweta*(2/b)*weta
                        gxy += cs[col+0]*(2/b)*fAu*gAueta + cs[col+1]*(2/a)*fAvxi*gAv + cs[col+2]*(2/b)*weta*(2/a)*fAwxi*gAw + cs[col+2]*(2/a)*wxi*(2/b)*fAw*gAweta
                        kxx += -cs[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAwetaeta
                        kxy += -2*cs[col+2]*(2/a)*fAwxi*(2/b)*gAweta

                # Calculating membrane stress components
                Nxx = A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy

                # computing kG

                c = -1
                for j in range(n):
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fx(j, eta, w1ty, w1ry, w2ty, w2ry)

                    for l in range(n):
                        gBw = calc_f(l, eta, w1ty, w1ry, w2ty, w2ry)
                        gBweta = calc_fx(l, eta, w1ty, w1ry, w2ty, w2ry)

                        for i in range(m):
                            fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                            fAwxi = calc_fx(i, xi, w1tx, w1rx, w2tx, w2rx)

                            for k in range(m):
                                fBw = calc_f(k, xi, w1tx, w1rx, w2tx, w2rx)
                                fBwxi = calc_fx(k, xi, w1tx, w1rx, w2tx, w2rx)

                                row = row0 + num*(j*m + i)
                                col = col0 + num*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(Nxx*b*fAwxi*fBwxi*gAw*gBw/a + Nxy*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta) + Nyy*a*fAw*fBw*gAweta*gBweta/b)

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG


def calc_fint(np.ndarray[cDOUBLE, ndim=1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    cdef double a, b
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, j, c, col, ptx, pty
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double Nxx, Nyy, Nxy, Mxx, Myy, Mxy
    cdef double exx, eyy, gxy, kxx, kyy, kxy

    cdef double xi, eta, weight
    cdef double wxi, weta

    cdef double fAu, fAuxi, fAv, fAvxi, fAw, fAwxi, fAwxixi
    cdef double gAu, gAueta, gAv, gAveta, gAw, gAweta, gAwetaeta

    cdef np.ndarray[cDOUBLE, ndim=1] xis, etas, weights_xi, weights_eta, fint

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef np.ndarray[cDOUBLE, ndim=4] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    u1tx = shell.u1tx; u1rx = shell.u1rx; u2tx = shell.u2tx; u2rx = shell.u2rx
    v1tx = shell.v1tx; v1rx = shell.v1rx; v2tx = shell.v2tx; v2rx = shell.v2rx
    w1tx = shell.w1tx; w1rx = shell.w1rx; w2tx = shell.w2tx; w2rx = shell.w2rx
    u1ty = shell.u1ty; u1ry = shell.u1ry; u2ty = shell.u2ty; u2ry = shell.u2ry
    v1ty = shell.v1ty; v1ry = shell.v1ry; v2ty = shell.v2ty; v2ry = shell.v2ry
    w1ty = shell.w1ty; w1ry = shell.w1ry; w2ty = shell.w2ty; w2ry = shell.w2ry

    xis = np.zeros(nx, dtype=DOUBLE)
    weights_xi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weights_eta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weights_xi[0])
    leggauss_quad(ny, &etas[0], &weights_eta[0])

    fint = np.zeros(size, dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                weight = weights_xi[ptx]*weights_eta[pty]

                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            #TODO could assume symmetry
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                A11 = F[0*6 + 0]
                A12 = F[0*6 + 1]
                A16 = F[0*6 + 2]
                A22 = F[1*6 + 1]
                A26 = F[1*6 + 2]
                A66 = F[2*6 + 2]

                B11 = F[0*6 + 3]
                B12 = F[0*6 + 4]
                B16 = F[0*6 + 5]
                B22 = F[1*6 + 4]
                B26 = F[1*6 + 5]
                B66 = F[2*6 + 5]

                D11 = F[3*6 + 3]
                D12 = F[3*6 + 4]
                D16 = F[3*6 + 5]
                D22 = F[4*6 + 4]
                D26 = F[4*6 + 5]
                D66 = F[5*6 + 5]

                wxi = 0
                weta = 0
                for j in range(n):
                    #TODO save in buffer
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fx(j, eta, w1ty, w1ry, w2ty, w2ry)
                    for i in range(m):
                        #TODO save in buffer
                        fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxi = calc_fx(i, xi, w1tx, w1rx, w2tx, w2rx)

                        col = col0 + num*(j*m + i)

                        wxi += cs[col+2]*fAwxi*gAw
                        weta += cs[col+2]*fAw*gAweta

                # current strain state
                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.

                for j in range(n):
                    #TODO save in buffer
                    gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAueta = calc_fx(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAveta = calc_fx(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fx(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAwetaeta = calc_fxx(j, eta, w1ty, w1ry, w2ty, w2ry)

                    for i in range(m):
                        #TODO save in buffer
                        fAu = calc_f(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAuxi = calc_fx(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAv = calc_f(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAvxi = calc_fx(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxi = calc_fx(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxixi = calc_fxx(i, xi, w1tx, w1rx, w2tx, w2rx)

                        col = col0 + num*(j*m + i)

                        exx += cs[col+0]*(2/a)*fAuxi*gAu + 0.5*cs[col+2]*(2/a)*fAwxi*gAw*(2/a)*wxi
                        eyy += cs[col+1]*(2/b)*fAv*gAveta + 0.5*cs[col+2]*(2/b)*fAw*gAweta*(2/b)*weta
                        gxy += cs[col+0]*(2/b)*fAu*gAueta + cs[col+1]*(2/a)*fAvxi*gAv + cs[col+2]*(2/b)*weta*(2/a)*fAwxi*gAw + cs[col+2]*(2/a)*wxi*(2/b)*fAw*gAweta
                        kxx += -cs[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAwetaeta
                        kxy += -2*cs[col+2]*(2/a*2/b)*fAwxi*gAweta

                # current stress state
                Nxx = A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy
                Mxx = B11*exx + B12*eyy + B16*gxy + D11*kxx + D12*kyy + D16*kxy
                Myy = B12*exx + B22*eyy + B26*gxy + D12*kxx + D22*kyy + D26*kxy
                Mxy = B16*exx + B26*eyy + B66*gxy + D16*kxx + D26*kyy + D66*kxy

                for j in range(n):
                    gAu = calc_f(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAueta = calc_fx(j, eta, u1ty, u1ry, u2ty, u2ry)
                    gAv = calc_f(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAveta = calc_fx(j, eta, v1ty, v1ry, v2ty, v2ry)
                    gAw = calc_f(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAweta = calc_fx(j, eta, w1ty, w1ry, w2ty, w2ry)
                    gAwetaeta = calc_fxx(j, eta, w1ty, w1ry, w2ty, w2ry)
                    for i in range(m):
                        fAu = calc_f(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAuxi = calc_fx(i, xi, u1tx, u1rx, u2tx, u2rx)
                        fAv = calc_f(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAvxi = calc_fx(i, xi, v1tx, v1rx, v2tx, v2rx)
                        fAw = calc_f(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxi = calc_fx(i, xi, w1tx, w1rx, w2tx, w2rx)
                        fAwxixi = calc_fxx(i, xi, w1tx, w1rx, w2tx, w2rx)

                        col = col0 + num*(j*m + i)

                        fint[col+0] += weight*( 0.25*a*b*(2*Nxx*fAuxi*gAu/a + 2*Nxy*fAu*gAueta/b) )
                        fint[col+1] += weight*( 0.25*a*b*(2*Nxy*fAvxi*gAv/a + 2*Nyy*fAv*gAveta/b) )
                        fint[col+2] += weight*( 0.25*a*b*(-4*Mxx*fAwxixi*gAw/(a*a) - 8*Mxy*fAwxi*gAweta/(a*b) - 4*Myy*fAw*gAwetaeta/(b*b) + 4*Nxx*fAwxi*gAw*wxi/(a*a) + 4*Nxy*(fAw*gAweta*wxi + fAwxi*gAw*weta)/(a*b) + 4*Nyy*fAw*gAweta*weta/(b*b)) )

    return fint
