#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef extern from 'legendre_gauss_quadrature.hpp':
    void leggauss_quad(int n, double *points, double* weights) nogil


DOUBLE = np.float64
INT = long

cdef int DOF = 3


def fkC_num(double [::1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, row, col, ptx, pty
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66

    cdef long [::1] kCr, kCc
    cdef double [::1] kCv

    cdef double fAu, fAuxi, fAv, fAvxi, fAw, fAwxi, fAwxixi
    cdef double fBu, fBuxi, fBv, fBvxi, fBw, fBwxi, fBwxixi
    cdef double gAu, gAueta, gAv, gAveta, gAw, gAweta, gAwetaeta
    cdef double gBu, gBueta, gBv, gBveta, gBw, gBweta, gBwetaeta
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double wxi, weta

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef double [:, :, :, ::1] Fnxny

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
    x1 = shell.x1
    x2 = shell.x2
    y1 = shell.y1
    y2 = shell.y2
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

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
                        gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                        gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                        for i in range(m):
                            #TODO put these in a lookup vector
                            fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                            fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                            col = col0 + DOF*(j*m + i)

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
                    fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                    fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                    fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                    fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    fAwxixi = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                    for k in range(m):
                        fBu = f(k, xi, x1u, x1ur, x2u, x2ur)
                        fBuxi = fp(k, xi, x1u, x1ur, x2u, x2ur)
                        fBv = f(k, xi, x1v, x1vr, x2v, x2vr)
                        fBvxi = fp(k, xi, x1v, x1vr, x2v, x2vr)
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)
                        fBwxi = fp(k, xi, x1w, x1wr, x2w, x2wr)
                        fBwxixi = fpp(k, xi, x1w, x1wr, x2w, x2wr)

                        for j in range(n):
                            gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                            gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                            gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                            gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                            gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                            gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                            gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = f(l, eta, y1u, y1ur, y2u, y2ur)
                                gBueta = fp(l, eta, y1u, y1ur, y2u, y2ur)
                                gBv = f(l, eta, y1v, y1vr, y2v, y2vr)
                                gBveta = fp(l, eta, y1v, y1vr, y2v, y2vr)
                                gBw = f(l, eta, y1w, y1wr, y2w, y2wr)
                                gBweta = fp(l, eta, y1w, y1wr, y2w, y2wr)
                                gBwetaeta = fpp(l, eta, y1w, y1wr, y2w, y2wr)

                         ### THIS IS CALCULATES K_C NOT K_C_0

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*( intx*inty*(a*fBu*gBueta*(A16*b*fAuxi*gAu + A66*a*fAu*gAueta) + b*fBuxi*gBu*(A11*b*fAuxi*gAu + A16*a*fAu*gAueta))/((a*a)*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*( intx*inty*(a*fBv*gBveta*(A12*b*fAuxi*gAu + A26*a*fAu*gAueta) + b*fBvxi*gBv*(A16*b*fAuxi*gAu + A66*a*fAu*gAueta))/((a*a)*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*( 2*intx*inty*((a*a)*fBw*(gBweta*weta*(A12*b*fAuxi*gAu + A26*a*fAu*gAueta) - gBwetaeta*(B12*b*fAuxi*gAu + B26*a*fAu*gAueta)) + a*b*(-2*fBwxi*gBweta*(B16*b*fAuxi*gAu + B66*a*fAu*gAueta) + (fBw*gBweta*wxi + fBwxi*gBw*weta)*(A16*b*fAuxi*gAu + A66*a*fAu*gAueta)) + (b*b)*gBw*(fBwxi*wxi*(A11*b*fAuxi*gAu + A16*a*fAu*gAueta) - fBwxixi*(B11*b*fAuxi*gAu + B16*a*fAu*gAueta)))/((a*a*a)*(b*b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*( intx*inty*(a*fBu*gBueta*(A26*a*fAv*gAveta + A66*b*fAvxi*gAv) + b*fBuxi*gBu*(A12*a*fAv*gAveta + A16*b*fAvxi*gAv))/((a*a)*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*( intx*inty*(a*fBv*gBveta*(A22*a*fAv*gAveta + A26*b*fAvxi*gAv) + b*fBvxi*gBv*(A26*a*fAv*gAveta + A66*b*fAvxi*gAv))/((a*a)*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*( 2*intx*inty*((a*a)*fBw*(gBweta*weta*(A22*a*fAv*gAveta + A26*b*fAvxi*gAv) - gBwetaeta*(B22*a*fAv*gAveta + B26*b*fAvxi*gAv)) + a*b*(-2*fBwxi*gBweta*(B26*a*fAv*gAveta + B66*b*fAvxi*gAv) + (fBw*gBweta*wxi + fBwxi*gBw*weta)*(A26*a*fAv*gAveta + A66*b*fAvxi*gAv)) + (b*b)*gBw*(fBwxi*wxi*(A12*a*fAv*gAveta + A16*b*fAvxi*gAv) - fBwxixi*(B12*a*fAv*gAveta + B16*b*fAvxi*gAv)))/((a*a*a)*(b*b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*( 2*intx*inty*(a*fBu*gBueta*((a*a)*fAw*(A26*gAweta*weta - B26*gAwetaeta) + a*b*(A66*(fAw*gAweta*wxi + fAwxi*gAw*weta) - 2*B66*fAwxi*gAweta) + (b*b)*gAw*(A16*fAwxi*wxi - B16*fAwxixi)) + b*fBuxi*gBu*((a*a)*fAw*(A12*gAweta*weta - B12*gAwetaeta) + a*b*(A16*(fAw*gAweta*wxi + fAwxi*gAw*weta) - 2*B16*fAwxi*gAweta) + (b*b)*gAw*(A11*fAwxi*wxi - B11*fAwxixi)))/((a*a*a)*(b*b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*( 2*intx*inty*(a*fBv*gBveta*((a*a)*fAw*(A22*gAweta*weta - B22*gAwetaeta) + a*b*(A26*(fAw*gAweta*wxi + fAwxi*gAw*weta) - 2*B26*fAwxi*gAweta) + (b*b)*gAw*(A12*fAwxi*wxi - B12*fAwxixi)) + b*fBvxi*gBv*((a*a)*fAw*(A26*gAweta*weta - B26*gAwetaeta) + a*b*(A66*(fAw*gAweta*wxi + fAwxi*gAw*weta) - 2*B66*fAwxi*gAweta) + (b*b)*gAw*(A16*fAwxi*wxi - B16*fAwxixi)))/((a*a*a)*(b*b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*( 4*intx*inty*((a*a)*fBw*(gBweta*weta*((a*a)*fAw*(A22*gAweta*weta - B22*gAwetaeta) + a*b*(A26*(fAw*gAweta*wxi + fAwxi*gAw*weta) - 2*B26*fAwxi*gAweta) + (b*b)*gAw*(A12*fAwxi*wxi - B12*fAwxixi)) - gBwetaeta*((a*a)*fAw*(B22*gAweta*weta - D22*gAwetaeta) + a*b*(B26*(fAw*gAweta*wxi + fAwxi*gAw*weta) - 2*D26*fAwxi*gAweta) + (b*b)*gAw*(B12*fAwxi*wxi - D12*fAwxixi))) + a*b*(-2*fBwxi*gBweta*((a*a)*fAw*(B26*gAweta*weta - D26*gAwetaeta) + a*b*(B66*(fAw*gAweta*wxi + fAwxi*gAw*weta) - 2*D66*fAwxi*gAweta) + (b*b)*gAw*(B16*fAwxi*wxi - D16*fAwxixi)) + (fBw*gBweta*wxi + fBwxi*gBw*weta)*((a*a)*fAw*(A26*gAweta*weta - B26*gAwetaeta) + a*b*(A66*(fAw*gAweta*wxi + fAwxi*gAw*weta) - 2*B66*fAwxi*gAweta) + (b*b)*gAw*(A16*fAwxi*wxi - B16*fAwxixi))) + (b*b)*gBw*(fBwxi*wxi*((a*a)*fAw*(A12*gAweta*weta - B12*gAwetaeta) + a*b*(A16*(fAw*gAweta*wxi + fAwxi*gAw*weta) - 2*B16*fAwxi*gAweta) + (b*b)*gAw*(A11*fAwxi*wxi - B11*fAwxixi)) - fBwxixi*((a*a)*fAw*(B12*gAweta*weta - D12*gAwetaeta) + a*b*(B16*(fAw*gAweta*wxi + fAwxi*gAw*weta) - 2*D16*fAwxi*gAweta) + (b*b)*gAw*(B11*fAwxi*wxi - D11*fAwxixi))))/((a*a*a*a)*(b*b*b*b)) )

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC


def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny, int NLgeom=0,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, k, j, l, c, row, col, ptx, pty
    cdef double xi, eta, x, y, weight
    cdef double xi1, xi2, eta1, eta2

    cdef long [::1] kGr, kGc
    cdef double [::1] kGv

    cdef double fAu, fAv, fAw, fAuxi, fAvxi, fAwxi, fAwxixi
    cdef double gAu, gAv, gAw, gAueta, gAveta, gAweta, gAwetaeta
    cdef double gBw, gBweta, fBw, fBwxi

    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double wxi, weta, Nxx, Nyy, Nxy

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef double [:, :, :, ::1] Fnxny

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
    x1 = shell.x1
    x2 = shell.x2
    y1 = shell.y1
    y2 = shell.y2
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

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
                        gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                        gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                        for i in range(m):
                            fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                            fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                            col = col0 + DOF*(j*m + i)

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
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)

                    for i in range(m):
                        fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                        fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

                        exx += cs[col+0]*(2/a)*fAuxi*gAu + 0.5*cs[col+2]*(2/a)*fAwxi*gAw*(2/a)*wxi
                        eyy += cs[col+1]*(2/b)*fAv*gAveta + 0.5*cs[col+2]*(2/b)*fAw*gAweta*(2/b)*weta
                        gxy += cs[col+0]*(2/b)*fAu*gAueta + cs[col+1]*(2/a)*fAvxi*gAv + cs[col+2]*(2/b)*weta*(2/a)*fAwxi*gAw + cs[col+2]*(2/a)*wxi*(2/b)*fAw*gAweta
                        kxx += -cs[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAwetaeta
                        kxy += -2*cs[col+2]*(2/a)*fAwxi*(2/b)*gAweta

                # Calculating membrane stress components
                Nxx = Nxx0 + A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = Nyy0 + A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = Nxy0 + A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy

                # computing kG
                c = -1
                for j in range(n):
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):
                        gBw = f(l, eta, y1w, y1wr, y2w, y2wr)
                        gBweta = fp(l, eta, y1w, y1wr, y2w, y2wr)

                        for i in range(m):
                            fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                            fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                            for k in range(m):
                                fBw = f(k, xi, x1w, x1wr, x2w, x2wr)
                                fBwxi = fp(k, xi, x1w, x1wr, x2w, x2wr)

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*( intx*inty*(a*fBw*gBweta*(Nxy*b*fAwxi*gAw + Nyy*a*fAw*gAweta) + b*fBwxi*gBw*(Nxx*b*fAwxi*gAw + Nxy*a*fAw*gAweta))/((a*a)*(b*b)) )

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG


def fkM_num(object shell, double offset, object hrho_input, int size,
        int row0, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kMr, kMc
    cdef double [::1] kMv

    cdef double fAu, fAv, fAw, fAwxi
    cdef double fBu, fBv, fBw, fBwxi
    cdef double gAu, gAv, gAw, gAweta
    cdef double gBu, gBv, gBw, gBweta
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # F as 4-D matrix, must be [nx, ny, 2], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double h, rho, d
    cdef double [:, :, ::1] hrho_nxny

    cdef int one_hrho_each_point = 0

    d = offset

    hrho_input = np.asarray(hrho_input, dtype=DOUBLE)
    if hrho_input.shape == (nx, ny, 2):
        hrho_nxny = np.ascontiguousarray(hrho_input)
        one_hrho_each_point = 1
    elif hrho_input.shape == (2,):
        # creating dummy 4-D array that is not used
        hrho_nxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        hrho_input = np.ascontiguousarray(hrho_input)
        h, rho = hrho_input
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    x1 = shell.x1
    x2 = shell.x2
    y1 = shell.y1
    y2 = shell.y2
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    fdim = 7*m*m*n*n

    xis = np.zeros(nx, dtype=DOUBLE)
    weights_xi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weights_eta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weights_xi[0])
    leggauss_quad(ny, &etas[0], &weights_eta[0])

    kMr = np.zeros((fdim,), dtype=INT)
    kMc = np.zeros((fdim,), dtype=INT)
    kMv = np.zeros((fdim,), dtype=DOUBLE)

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

                if one_hrho_each_point == 1:
                    h = hrho_nxny[ptx, pty, 0]
                    rho = hrho_nxny[ptx, pty, 1]

                # kM
                c = -1
                for i in range(m):
                    fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                    fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                    fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                    fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                    for k in range(m):
                        fBu = f(k, xi, x1u, x1ur, x2u, x2ur)
                        fBv = f(k, xi, x1v, x1vr, x2v, x2vr)
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)
                        fBwxi = fp(k, xi, x1w, x1wr, x2w, x2wr)

                        for j in range(n):
                            gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                            gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                            gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                            gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = f(l, eta, y1u, y1ur, y2u, y2ur)
                                gBv = f(l, eta, y1v, y1vr, y2v, y2vr)
                                gBw = f(l, eta, y1w, y1wr, y2w, y2wr)
                                gBweta = fp(l, eta, y1w, y1wr, y2w, y2wr)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+0
                                kMv[c] += weight*( 0.25*fAu*fBu*gAu*gBu*h*intx*inty*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+2
                                kMv[c] += weight*( 0.5*d*fAu*fBwxi*gAu*gBw*h*intx*inty*rho/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+1
                                kMv[c] += weight*( 0.25*fAv*fBv*gAv*gBv*h*intx*inty*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+2
                                kMv[c] += weight*( 0.5*d*fAv*fBw*gAv*gBweta*h*intx*inty*rho/b )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+0
                                kMv[c] += weight*( 0.5*d*fAwxi*fBu*gAw*gBu*h*intx*inty*rho/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+1
                                kMv[c] += weight*( 0.5*d*fAw*fBv*gAweta*gBv*h*intx*inty*rho/b )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+2
                                kMv[c] += weight*( 0.25*h*intx*inty*rho*(fAw*fBw*gAw*gBw + 4*fAw*fBw*gAweta*gBweta*((d*d) + 0.0833333333333333*(h*h))/(b*b) + 4*fAwxi*fBwxi*gAw*gBw*((d*d) + 0.0833333333333333*(h*h))/(a*a)) )

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM



def fkAx_num(object shell, int size, int row0, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, beta, gamma, intx, inty
    cdef int m, n
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kAr, kAc
    cdef double [::1] kAv

    cdef double fAw, fAwxi, fBw, gAw, gBw
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    x1 = shell.x1
    x2 = shell.x2
    y1 = shell.y1
    y2 = shell.y2
    beta = shell.beta
    gamma = shell.gamma
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    fdim = 1*m*m*n*n

    xis = np.zeros(nx, dtype=DOUBLE)
    weights_xi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weights_eta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weights_xi[0])
    leggauss_quad(ny, &etas[0], &weights_eta[0])

    kAr = np.zeros((fdim,), dtype=INT)
    kAc = np.zeros((fdim,), dtype=INT)
    kAv = np.zeros((fdim,), dtype=DOUBLE)

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

                # kAx
                c = -1
                for i in range(m):
                    fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                    fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                    for k in range(m):
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)

                        for j in range(n):
                            gAw = f(j, eta, y1w, y1wr, y2w, y2wr)

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBw = f(l, eta, y1w, y1wr, y2w, y2wr)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kAr[c] = row+2
                                    kAc[c] = col+2
                                kAv[c] += weight*( 0.25*intx*inty*(fAw*fBw*gAw*gBw*gamma - 2*beta*fAwxi*fBw*gAw*gBw/a) )

    kAx = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kAx


def fkAy_num(object shell, int size, int row0, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, beta, intx, inty
    cdef int m, n
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kAr, kAc
    cdef double [::1] kAv

    cdef double fAw, fBw, gAweta, gBw
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    x1 = shell.x1
    x2 = shell.x2
    y1 = shell.y1
    y2 = shell.y2
    beta = shell.beta
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    fdim = 1*m*m*n*n

    xis = np.zeros(nx, dtype=DOUBLE)
    weights_xi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weights_eta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weights_xi[0])
    leggauss_quad(ny, &etas[0], &weights_eta[0])

    kAr = np.zeros((fdim,), dtype=INT)
    kAc = np.zeros((fdim,), dtype=INT)
    kAv = np.zeros((fdim,), dtype=DOUBLE)

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

                # kAy
                c = -1
                for i in range(m):
                    fAw = f(i, xi, x1w, x1wr, x2w, x2wr)

                    for k in range(m):
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)

                        for j in range(n):
                            gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBw = f(l, eta, y1w, y1wr, y2w, y2wr)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kAr[c] = row+2
                                    kAc[c] = col+2
                                kAv[c] += weight*( -0.5*beta*fAw*fBw*gAweta*gBw*intx*inty/b )

    kAy = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kAy


def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, col, ptx, pty
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double Nxx, Nyy, Nxy, Mxx, Myy, Mxy
    cdef double exx, eyy, gxy, kxx, kyy, kxy

    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double wxi, weta

    cdef double fAu, fAuxi, fAv, fAvxi, fAw, fAwxi, fAwxixi
    cdef double gAu, gAueta, gAv, gAveta, gAw, gAweta, gAwetaeta

    cdef double [::1] xis, etas, weights_xi, weights_eta, fint

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef double [:, :, :, ::1] Fnxny

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
    x1 = shell.x1
    x2 = shell.x2
    y1 = shell.y1
    y2 = shell.y2
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    xis = np.zeros(nx, dtype=DOUBLE)
    weights_xi = np.zeros(nx, dtype=DOUBLE)
    etas = np.zeros(ny, dtype=DOUBLE)
    weights_eta = np.zeros(ny, dtype=DOUBLE)

    leggauss_quad(nx, &xis[0], &weights_xi[0])
    leggauss_quad(ny, &etas[0], &weights_eta[0])

    fint = np.zeros(size, dtype=DOUBLE)

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
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    for i in range(m):
                        #TODO save in buffer
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

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
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)

                    for i in range(m):
                        #TODO save in buffer
                        fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                        fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

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
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)

                    for i in range(m):
                        fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                        fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

                        fint[col+0] += weight*( intx*inty*(Nxx*b*fAuxi*gAu + Nxy*a*fAu*gAueta)/(2*a*b) )
                        fint[col+1] += weight*( intx*inty*(Nxy*b*fAvxi*gAv + Nyy*a*fAv*gAveta)/(2*a*b) )
                        fint[col+2] += weight*( intx*inty*((a*a)*fAw*(-Myy*gAwetaeta + Nyy*gAweta*weta) + a*b*(-2*Mxy*fAwxi*gAweta + Nxy*(fAw*gAweta*wxi + fAwxi*gAw*weta)) + (b*b)*gAw*(-Mxx*fAwxixi + Nxx*fAwxi*wxi))/((a*a)*(b*b)) )

    return fint

