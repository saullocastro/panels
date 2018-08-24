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

cdef int num = 3


def fkoiter(
        double LAMBDA, double NxxHAT,
        np.ndarray[cDOUBLE, ndim=1] c0,
        np.ndarray[cDOUBLE, ndim=1] cA,
        np.ndarray[cDOUBLE, ndim=1] cB,
        np.ndarray[cDOUBLE, ndim=1] cC,
        np.ndarray[cDOUBLE, ndim=1] cD,
        object Finput, object shell,
        int nx, int ny, int NLgeom=1):
    cdef double phi2 = 0
    cdef double phi2o = 0
    cdef double phi3 = 0
    cdef double phi4 = 0
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, lx, ly, r, intx, inty
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, iA, iB, iC, iD
    cdef int j, jA, jB, jC, jD
    cdef int rA, rB, rC, rD
    cdef int sA, sB, sC, sD
    cdef int col, ptx, pty
    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef double Aij, Bij, Dij
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66

    cdef double N[3]
    cdef double eA[3]
    cdef double eB[3]
    cdef double eC[3]
    cdef double eD[3]
    cdef double kA[3]
    cdef double kB[3]
    cdef double kC[3]
    cdef double kD[3]
    cdef double eAB[3]
    cdef double eAC[3]
    cdef double eAD[3]
    cdef double eBC[3]
    cdef double eBD[3]
    cdef double eCD[3]
    cdef double *suxi
    cdef double *sueta
    cdef double *svxi
    cdef double *sveta
    cdef double *sw
    cdef double *swxi
    cdef double *sweta
    cdef double *swxixi
    cdef double *swetaeta
    cdef double *swxieta

    cdef double fAu, fAuxi, fAv, fAvxi, fAw, fAwxi, fAwxixi
    cdef double fBu, fBuxi, fBv, fBvxi, fBw, fBwxi, fBwxixi
    cdef double gAu, gAueta, gAv, gAveta, gAw, gAweta, gAwetaeta
    cdef double gBu, gBueta, gBv, gBveta, gBw, gBweta, gBwetaeta
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
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
    a = lx = shell.a
    b = ly = shell.b
    r = shell.r
    m = shell.m
    n = shell.n
    x1 = shell.x1
    x2 = shell.x2
    y1 = shell.y1
    y2 = shell.y2
    u1tx = shell.u1tx; u1rx = shell.u1rx; u2tx = shell.u2tx; u2rx = shell.u2rx
    v1tx = shell.v1tx; v1rx = shell.v1rx; v2tx = shell.v2tx; v2rx = shell.v2rx
    w1tx = shell.w1tx; w1rx = shell.w1rx; w2tx = shell.w2tx; w2rx = shell.w2rx
    u1ty = shell.u1ty; u1ry = shell.u1ry; u2ty = shell.u2ty; u2ry = shell.u2ry
    v1ty = shell.v1ty; v1ry = shell.v1ry; v2ty = shell.v2ty; v2ry = shell.v2ry
    w1ty = shell.w1ty; w1ry = shell.w1ry; w2ty = shell.w2ty; w2ry = shell.w2ry

    suxi = <double *>malloc(m * n * sizeof(double))
    sueta = <double *>malloc(m * n * sizeof(double))
    svxi = <double *>malloc(m * n * sizeof(double))
    sveta = <double *>malloc(m * n * sizeof(double))
    sw = <double *>malloc(m * n * sizeof(double))
    swxi = <double *>malloc(m * n * sizeof(double))
    sweta = <double *>malloc(m * n * sizeof(double))
    swxixi = <double *>malloc(m * n * sizeof(double))
    swetaeta = <double *>malloc(m * n * sizeof(double))
    swxieta = <double *>malloc(m * n * sizeof(double))

    xis = np.zeros(nx, dtype=DOUBLE)
    weights_xi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weights_eta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weights_xi[0])
    leggauss_quad(ny, &etas[0], &weights_eta[0])

    xi1 = -1
    xi2 = +1
    if x1 != -1 and x2 != -1:
        xinf = 0
        xsup = shell.a
        xi1 = (x1 - xinf)/(xsup - xinf)*2 - 1
        xi2 = (x2 - xinf)/(xsup - xinf)*2 - 1
    else:
        x1 = 0
        x2 = shell.a

    eta1 = -1
    eta2 = +1
    if y1 != -1 and y2 != -1:
        yinf = 0
        ysup = shell.b
        eta1 = (y1 - yinf)/(ysup - yinf)*2 - 1
        eta2 = (y2 - yinf)/(ysup - yinf)*2 - 1
    else:
        y1 = 0
        y2 = shell.b

    intx = x2 - x1
    inty = y2 - y1

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

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

                            col = num*(j*m + i)

                            wxi += c0[col+2]*fAwxi*gAw
                            weta += c0[col+2]*fAw*gAweta

                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
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

                        col = num*(j*m + i)

                        exx += c0[col+0]*(2/a)*fAuxi*gAu + 0.5*c0[col+2]*(2/a)*fAwxi*gAw*(2/a)*wxi
                        eyy += c0[col+1]*(2/b)*fAv*gAveta + 1/r*c0[col+2]*fAw*gAw + 0.5*c0[col+2]*(2/b)*fAw*gAweta*(2/b)*weta
                        gxy += c0[col+0]*(2/b)*fAu*gAueta + c0[col+1]*(2/a)*fAvxi*gAv + c0[col+2]*(2/b)*weta*(2/a)*fAwxi*gAw + c0[col+2]*(2/a)*wxi*(2/b)*fAw*gAweta
                        kxx += -c0[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -c0[col+2]*(2/b*2/b)*fAw*gAwetaeta
                        kxy += -2*c0[col+2]*(2/a)*fAwxi*(2/b)*gAweta

                # Calculating membrane stress components
                N[0] = A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                N[1] = A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                N[2] = A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy

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

                        sA = jA*m + iA

                        suxi[sA] = fAuxi*gAu
                        sueta[sA] = fAu*gAueta

                        svxi[sA] = fAvxi*gAv
                        sveta[sA] = fAv*gAveta

                        sw[sA] = fAw*gAw
                        swxi[sA] = fAwxi*gAw
                        sweta[sA] = fAw*gAweta

                        swxixi[sA] = fAwxixi*gAw
                        swetaeta[sA] = fAw*gAwetaeta
                        swxieta[sA] = fAwxi*gAweta

                for iA in range(m):
                    for jA in range(n):
                        rA = num*(jA*m + iA)
                        sA = jA*m + iA

                        eA[0] = 2/lx*suxi[sA]*cA[rA+0] + 4/(lx*lx)*wxi*swxi[sA]*cA[rA+2]
                        eA[1] = 2/ly*sveta[sA]*cA[rA+1] + 1/r*sw[sA]*cA[rA+2] + 4/(ly*ly)*weta*sweta[sA]*cA[rA+2]
                        eA[2] = 2/ly*sueta[sA]*cA[rA+0] + 2/lx*svxi[sA]*cA[rA+1] + 2/(lx*ly)*(wxi*sweta[sA] + weta*swxi[sA])*cA[rA+2]

                        kA[0] = -4/(lx*lx) * swxixi[sA]*cA[rA+2]
                        kA[1] = -4/(ly*ly) * swetaeta[sA]*cA[rA+2]
                        kA[2] = -2*4/(lx*ly) * swxieta[sA]*cA[rA+2]

                        for iB in range(m):
                            for jB in range(n):
                                rB = num*(jB*m + iB)
                                sB = jB*m + iB

                                eB[0] = 2/lx*suxi[sB]*cB[rB+0] + 4/(lx*lx)*wxi*swxi[sB]*cB[rB+2]
                                eB[1] = 2/ly*sveta[sB]*cB[rB+1] + 1/r*sw[sB]*cB[rB+2] + 4/(ly*ly)*weta*sweta[sB]*cB[rB+2]
                                eB[2] = 2/ly*sueta[sB]*cB[rB+0] + 2/lx*svxi[sB]*cB[rB+1] + 2/(lx*ly)*(wxi*sweta[sB] + weta*swxi[sB])*cB[rB+2]

                                kB[0] = -4/(lx*lx) * swxixi[sB]*cB[rB+2]
                                kB[1] = -4/(ly*ly) * swetaeta[sB]*cB[rB+2]
                                kB[2] = -2*4/(lx*ly) * swxieta[sB]*cB[rB+2]

                                eAB[0] = 4/(lx*lx) * swxi[sA]*cA[rA+2] * swxi[sB]*cB[rB+2]
                                eAB[1] = 4/(ly*ly) * sweta[sA]*cA[rA+2] * sweta[sB]*cB[rB+2]
                                eAB[2] = 4/(lx*ly) * (sweta[sA]*cA[rA+2] * swxi[sB]*cB[rB+2] +
                                                      swxi[sA]*cA[rA+2] * sweta[sB]*cB[rB+2])

                                for i in range(3):
                                    for j in range(3):
                                        Aij = F[i*6 + j]
                                        Bij = F[i*6 + (j+3)]
                                        Dij = F[(i+3)*6 + (j+3)]

                                        phi2 += weight*(lx*ly)/4 * (
                                                    N[i]*eAB[i]
                                                    + eA[i]*(Aij*eB[j] + Bij*kB[j])
                                                    + kA[i]*(Bij*eB[j] + Dij*kB[j])
                                                    - LAMBDA * NxxHAT * 4/(lx*lx) * swxi[sA]*cA[rA+2] * swxi[sB]*cB[rB+2]
                                                )

                                        phi2o += weight*(lx*ly)/4 * (
                                                    - NxxHAT * 4/(lx*lx) * swxi[sA]*cA[rA+2] * swxi[sB]*cB[rB+2]
                                                )

                                for iC in range(m):
                                    for jC in range(n):
                                        rC = num*(jC*m + iC)
                                        sC = jC*m + iC

                                        eC[0] = 2/lx*suxi[sC]*cC[rC+0] + 4/(lx*lx)*wxi*swxi[sC]*cC[rC+2]
                                        eC[1] = 2/ly*sveta[sC]*cC[rC+1] + 1/r*sw[sC]*cC[rC+2] + 4/(ly*ly)*weta*sweta[sC]*cC[rC+2]
                                        eC[2] = 2/ly*sueta[sC]*cC[rC+0] + 2/lx*svxi[sC]*cC[rC+1] + 2/(lx*ly)*(wxi*sweta[sC] + weta*swxi[sC])*cC[rC+2]

                                        kC[0] = -4/(lx*lx) * swxixi[sC]*cC[rC+2]
                                        kC[1] = -4/(ly*ly) * swetaeta[sC]*cC[rC+2]
                                        kC[2] = -2*4/(lx*ly) * swxieta[sC]*cC[rC+2]

                                        eAC[0] = 4/(lx*lx) * swxi[sA]*cA[rA+2] * swxi[sC]*cC[rC+2]
                                        eAC[1] = 4/(ly*ly) * sweta[sA]*cA[rA+2] * sweta[sC]*cC[rC+2]
                                        eAC[2] = 4/(lx*ly) * (sweta[sA]*cA[rA+2] * swxi[sC]*cC[rC+2] +
                                                              swxi[sA]*cA[rA+2] * sweta[sC]*cC[rC+2])

                                        eBC[0] = 4/(lx*lx) * swxi[sB]*cB[rB+2] * swxi[sC]*cC[rC+2]
                                        eBC[1] = 4/(ly*ly) * sweta[sB]*cB[rB+2] * sweta[sC]*cC[rC+2]
                                        eBC[2] = 4/(lx*ly) * (sweta[sB]*cB[rB+2] * swxi[sC]*cC[rC+2] +
                                                              swxi[sB]*cB[rB+2] * sweta[sC]*cC[rC+2])

                                        for i in range(3):
                                            for j in range(3):
                                                Aij = F[i*6 + j]
                                                Bij = F[i*6 + (j+3)]
                                                Dij = F[(i+3)*6 + (j+3)]

                                                phi3 += weight*(lx*ly)/4 * (
                                                            eAB[i]*(Aij*eC[j] + Bij*kC[j])
                                                            + Aij*eBC[j]*eA[i]
                                                            + eAC[i]*(Aij*eB[j] + Bij*kB[j]) )

                                        for iD in range(m):
                                            for jD in range(n):
                                                rD = num*(jD*m + iD)
                                                sD = jD*m + iD

                                                eD[0] = 2/lx*suxi[sD]*cD[rD+0] + 4/(lx*lx)*wxi*swxi[sD]*cD[rD+2]
                                                eD[1] = 2/ly*sveta[sD]*cD[rD+1] + 1/r*sw[sD]*cD[rD+2] + 4/(ly*ly)*weta*sweta[sD]*cD[rD+2]
                                                eD[2] = 2/ly*sueta[sD]*cD[rD+0] + 2/lx*svxi[sD]*cD[rD+1] + 2/(lx*ly)*(wxi*sweta[sD] + weta*swxi[sD])*cD[rD+2]

                                                kD[0] = -4/(lx*lx) * swxixi[sD]*cD[rD+2]
                                                kD[1] = -4/(ly*ly) * swetaeta[sD]*cD[rD+2]
                                                kD[2] = -2*4/(lx*ly) * swxieta[sD]*cD[rD+2]

                                                eAD[0] = 4/(lx*lx) * swxi[sA]*cA[rA+2] * swxi[sD]*cD[rD+2]
                                                eAD[1] = 4/(ly*ly) * sweta[sA]*cA[rA+2] * sweta[sD]*cD[rD+2]
                                                eAD[2] = 4/(lx*ly) * (sweta[sA]*cA[rA+2] * swxi[sD]*cD[rD+2] +
                                                                      swxi[sA]*cA[rA+2] * sweta[sD]*cD[rD+2])

                                                eBD[0] = 4/(lx*lx) * swxi[sB]*cB[rB+2] * swxi[sD]*cD[rD+2]
                                                eBD[1] = 4/(ly*ly) * sweta[sB]*cB[rB+2] * sweta[sD]*cD[rD+2]
                                                eBD[2] = 4/(lx*ly) * (sweta[sB]*cB[rB+2] * swxi[sD]*cD[rD+2] +
                                                                      swxi[sB]*cB[rB+2] * sweta[sD]*cD[rD+2])

                                                eCD[0] = 4/(lx*lx) * swxi[sC]*cC[rC+2] * swxi[sD]*cD[rD+2]
                                                eCD[1] = 4/(ly*ly) * sweta[sC]*cC[rC+2] * sweta[sD]*cD[rD+2]
                                                eCD[2] = 4/(lx*ly) * (sweta[sC]*cC[rC+2] * swxi[sD]*cD[rD+2] +
                                                                      swxi[sC]*cC[rC+2] * sweta[sD]*cD[rD+2])

                                                for i in range(3):
                                                    for j in range(3):
                                                        Aij = F[i*6 + j]
                                                        Bij = F[i*6 + (j+3)]
                                                        Dij = F[(i+3)*6 + (j+3)]

                                                        phi4 += weight*(lx*ly)/4 * Aij * (
                                                                    eCD[j]*eAB[i] + eBC[j]*eAD[i] + eBD[j]*eAC[i] )

    free(suxi)
    free(sueta)
    free(svxi)
    free(sveta)
    free(sw)
    free(swxi)
    free(sweta)
    free(swxixi)
    free(swetaeta)
    free(swxieta)

    return dict(phi2=phi2, phi2o=phi2o, phi3=phi3, phi4=phi4)


def fphi2Matrix(
        double LAMBDA, double NxxHAT,
        np.ndarray[cDOUBLE, ndim=1] c0, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, r, intx, inty
    cdef int m, n
    cdef double u1tx, u1rx, u2tx, u2rx
    cdef double v1tx, v1rx, v2tx, v2rx
    cdef double w1tx, w1rx, w2tx, w2rx
    cdef double u1ty, u1ry, u2ty, u2ry
    cdef double v1ty, v1ry, v2ty, v2ry
    cdef double w1ty, w1ry, w2ty, w2ry

    cdef int i, k, j, l, c, row, col, ptx, pty
    cdef double xi, eta, x, y, weight
    cdef double xi1, xi2, eta1, eta2

    cdef np.ndarray[cINT, ndim=1] phi2r, phi2c
    cdef np.ndarray[cDOUBLE, ndim=1] phi2v

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
    r = shell.r
    m = shell.m
    n = shell.n
    x1 = shell.x1
    x2 = shell.x2
    y1 = shell.y1
    y2 = shell.y2
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

    phi2r = np.zeros((fdim,), dtype=INT)
    phi2c = np.zeros((fdim,), dtype=INT)
    phi2v = np.zeros((fdim,), dtype=DOUBLE)

    xi1 = -1
    xi2 = +1
    if x1 != -1 and x2 != -1:
        xinf = 0
        xsup = shell.a
        xi1 = (x1 - xinf)/(xsup - xinf)*2 - 1
        xi2 = (x2 - xinf)/(xsup - xinf)*2 - 1
    else:
        x1 = 0
        x2 = shell.a

    eta1 = -1
    eta2 = +1
    if y1 != -1 and y2 != -1:
        yinf = 0
        ysup = shell.b
        eta1 = (y1 - yinf)/(ysup - yinf)*2 - 1
        eta2 = (y2 - yinf)/(ysup - yinf)*2 - 1
    else:
        y1 = 0
        y2 = shell.b

    intx = x2 - x1
    inty = y2 - y1

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

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

                            wxi += c0[col+2]*fAwxi*gAw
                            weta += c0[col+2]*fAw*gAweta

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

                        exx += c0[col+0]*(2/a)*fAuxi*gAu + 0.5*c0[col+2]*(2/a)*fAwxi*gAw*(2/a)*wxi
                        eyy += c0[col+1]*(2/b)*fAv*gAveta + 1/r*c0[col+2]*fAw*gAw + 0.5*c0[col+2]*(2/b)*fAw*gAweta*(2/b)*weta
                        gxy += c0[col+0]*(2/b)*fAu*gAueta + c0[col+1]*(2/a)*fAvxi*gAv + c0[col+2]*(2/b)*weta*(2/a)*fAwxi*gAw + c0[col+2]*(2/a)*wxi*(2/b)*fAw*gAweta
                        kxx += -c0[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -c0[col+2]*(2/b*2/b)*fAw*gAwetaeta
                        kxy += -2*c0[col+2]*(2/a)*fAwxi*(2/b)*gAweta

                # Calculating membrane stress components
                Nxx = A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy

                # computing phi2Matrix

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
                                    phi2r[c] = row+2
                                    phi2c[c] = col+2
                                phi2v[c] += weight*(
                                        Nxx*fAwxi*fBwxi*gAw*gBw*intx*inty/(a*a)
                                        + Nxy*intx*inty*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta)/(a*b)
                                        + Nyy*fAw*fBw*gAweta*gBweta*intx*inty/(b*b)

                                        - (a*b)/4 * LAMBDA * NxxHAT * (4/a**2) * fAwxi*gAw*fBwxi*gBw
                                        )

    phi2Matrix = coo_matrix((phi2v, (phi2r, phi2c)), shape=(size, size))

    return phi2Matrix
