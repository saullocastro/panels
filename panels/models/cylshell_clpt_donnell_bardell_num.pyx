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
    cdef double a, b, r, intx, inty
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
    r = shell.r
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

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*( A11*fAuxi*fBuxi*gAu*gBu*intx*inty/(a*a) + A16*intx*inty*(fAu*fBuxi*gAueta*gBu + fAuxi*fBu*gAu*gBueta)/(a*b) + A66*fAu*fBu*gAueta*gBueta*intx*inty/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*( A12*fAuxi*fBv*gAu*gBveta*intx*inty/(a*b) + A16*fAuxi*fBvxi*gAu*gBv*intx*inty/(a*a) + A26*fAu*fBv*gAueta*gBveta*intx*inty/(b*b) + A66*fAu*fBvxi*gAueta*gBv*intx*inty/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*( 2*A11*fAuxi*fBwxi*gAu*gBw*intx*inty*wxi/(a*a*a) + 0.5*A12*fAuxi*fBw*gAu*intx*inty*(gBw/r + 4*gBweta*weta/(b*b))/a + 2*A16*intx*inty*(fAu*fBwxi*gAueta*gBw*wxi + fAuxi*gAu*(fBw*gBweta*wxi + fBwxi*gBw*weta))/((a*a)*b) + 0.5*A26*fAu*fBw*gAueta*intx*inty*((b*b)*gBw + 4*gBweta*r*weta)/((b*b*b)*r) + 2*A66*fAu*gAueta*intx*inty*(fBw*gBweta*wxi + fBwxi*gBw*weta)/(a*(b*b)) - 2*B11*fAuxi*fBwxixi*gAu*gBw*intx*inty/(a*a*a) - 2*B12*fAuxi*fBw*gAu*gBwetaeta*intx*inty/(a*(b*b)) - 2*B16*intx*inty*(fAu*fBwxixi*gAueta*gBw + 2*fAuxi*fBwxi*gAu*gBweta)/((a*a)*b) - 2*B26*fAu*fBw*gAueta*gBwetaeta*intx*inty/(b*b*b) - 4*B66*fAu*fBwxi*gAueta*gBweta*intx*inty/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*( A12*fAv*fBuxi*gAveta*gBu*intx*inty/(a*b) + A16*fAvxi*fBuxi*gAv*gBu*intx*inty/(a*a) + A26*fAv*fBu*gAveta*gBueta*intx*inty/(b*b) + A66*fAvxi*fBu*gAv*gBueta*intx*inty/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*( A22*fAv*fBv*gAveta*gBveta*intx*inty/(b*b) + A26*intx*inty*(fAv*fBvxi*gAveta*gBv + fAvxi*fBv*gAv*gBveta)/(a*b) + A66*fAvxi*fBvxi*gAv*gBv*intx*inty/(a*a) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*( 2*A12*fAv*fBwxi*gAveta*gBw*intx*inty*wxi/((a*a)*b) + 2*A16*fAvxi*fBwxi*gAv*gBw*intx*inty*wxi/(a*a*a) + 0.5*A22*fAv*fBw*gAveta*intx*inty*((b*b)*gBw + 4*gBweta*r*weta)/((b*b*b)*r) + 0.5*A26*intx*inty*((b*b)*fAvxi*fBw*gAv*gBw + 4*r*(fAv*gAveta*(fBw*gBweta*wxi + fBwxi*gBw*weta) + fAvxi*fBw*gAv*gBweta*weta))/(a*(b*b)*r) + 2*A66*fAvxi*gAv*intx*inty*(fBw*gBweta*wxi + fBwxi*gBw*weta)/((a*a)*b) - 2*B12*fAv*fBwxixi*gAveta*gBw*intx*inty/((a*a)*b) - 2*B16*fAvxi*fBwxixi*gAv*gBw*intx*inty/(a*a*a) - 2*B22*fAv*fBw*gAveta*gBwetaeta*intx*inty/(b*b*b) - 2*B26*intx*inty*(2*fAv*fBwxi*gAveta*gBweta + fAvxi*fBw*gAv*gBwetaeta)/(a*(b*b)) - 4*B66*fAvxi*fBwxi*gAv*gBweta*intx*inty/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*( 2*A11*fAwxi*fBuxi*gAw*gBu*intx*inty*wxi/(a*a*a) + 0.5*A12*fAw*fBuxi*gBu*intx*inty*(gAw/r + 4*gAweta*weta/(b*b))/a + 2*A16*intx*inty*(fAw*fBuxi*gAweta*gBu*wxi + fAwxi*gAw*(fBu*gBueta*wxi + fBuxi*gBu*weta))/((a*a)*b) + 0.5*A26*fAw*fBu*gBueta*intx*inty*((b*b)*gAw + 4*gAweta*r*weta)/((b*b*b)*r) + 2*A66*fBu*gBueta*intx*inty*(fAw*gAweta*wxi + fAwxi*gAw*weta)/(a*(b*b)) - 2*B11*fAwxixi*fBuxi*gAw*gBu*intx*inty/(a*a*a) - 2*B12*fAw*fBuxi*gAwetaeta*gBu*intx*inty/(a*(b*b)) - 2*B16*intx*inty*(2*fAwxi*fBuxi*gAweta*gBu + fAwxixi*fBu*gAw*gBueta)/((a*a)*b) - 2*B26*fAw*fBu*gAwetaeta*gBueta*intx*inty/(b*b*b) - 4*B66*fAwxi*fBu*gAweta*gBueta*intx*inty/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*( 2*A12*fAwxi*fBv*gAw*gBveta*intx*inty*wxi/((a*a)*b) + 2*A16*fAwxi*fBvxi*gAw*gBv*intx*inty*wxi/(a*a*a) + 0.5*A22*fAw*fBv*gBveta*intx*inty*((b*b)*gAw + 4*gAweta*r*weta)/((b*b*b)*r) + 0.5*A26*intx*inty*((b*b)*fAw*fBvxi*gAw*gBv + 4*r*(fAw*gAweta*(fBv*gBveta*wxi + fBvxi*gBv*weta) + fAwxi*fBv*gAw*gBveta*weta))/(a*(b*b)*r) + 2*A66*fBvxi*gBv*intx*inty*(fAw*gAweta*wxi + fAwxi*gAw*weta)/((a*a)*b) - 2*B12*fAwxixi*fBv*gAw*gBveta*intx*inty/((a*a)*b) - 2*B16*fAwxixi*fBvxi*gAw*gBv*intx*inty/(a*a*a) - 2*B22*fAw*fBv*gAwetaeta*gBveta*intx*inty/(b*b*b) - 2*B26*intx*inty*(fAw*fBvxi*gAwetaeta*gBv + 2*fAwxi*fBv*gAweta*gBveta)/(a*(b*b)) - 4*B66*fAwxi*fBvxi*gAweta*gBv*intx*inty/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*( 4*A11*fAwxi*fBwxi*gAw*gBw*intx*inty*(wxi*wxi)/(a*a*a*a) + A12*intx*inty*wxi*((b*b)*gAw*gBw*(fAw*fBwxi + fAwxi*fBw) + r*weta*(4*fAw*fBwxi*gAweta*gBw + 4*fAwxi*fBw*gAw*gBweta))/((a*a)*(b*b)*r) + 4*A16*intx*inty*wxi*(fAw*fBwxi*gAweta*gBw*wxi + fAwxi*gAw*(fBw*gBweta*wxi + 2*fBwxi*gBw*weta))/((a*a*a)*b) + 0.25*A22*fAw*fBw*intx*inty*((b*b)*gAw + 4*gAweta*r*weta)*((b*b)*gBw + 4*gBweta*r*weta)/((b*b*b*b)*(r*r)) + A26*intx*inty*((b*b)*(fAw*(fBw*gAw*gBweta*wxi + fBw*gAweta*gBw*wxi + fBwxi*gAw*gBw*weta) + fAwxi*fBw*gAw*gBw*weta) + 4*r*weta*(fAw*gAweta*(2*fBw*gBweta*wxi + fBwxi*gBw*weta) + fAwxi*fBw*gAw*gBweta*weta))/(a*(b*b*b)*r) + 4*A66*intx*inty*(fAw*gAweta*wxi + fAwxi*gAw*weta)*(fBw*gBweta*wxi + fBwxi*gBw*weta)/((a*a)*(b*b)) - 4*B11*gAw*gBw*intx*inty*wxi*(fAwxi*fBwxixi + fAwxixi*fBwxi)/(a*a*a*a) - B12*intx*inty*((b*b)*gAw*gBw*(fAw*fBwxixi + fAwxixi*fBw) + 4*r*(fAw*gBw*(fBwxi*gAwetaeta*wxi + fBwxixi*gAweta*weta) + fBw*gAw*(fAwxi*gBwetaeta*wxi + fAwxixi*gBweta*weta)))/((a*a)*(b*b)*r) - 4*B16*intx*inty*(fAw*fBwxixi*gAweta*gBw*wxi + fAwxi*(2*fBwxi*gAw*gBweta*wxi + 2*fBwxi*gAweta*gBw*wxi + fBwxixi*gAw*gBw*weta) + fAwxixi*gAw*(fBw*gBweta*wxi + fBwxi*gBw*weta))/((a*a*a)*b) - B22*fAw*fBw*intx*inty*((b*b)*(gAw*gBwetaeta + gAwetaeta*gBw) + r*weta*(4*gAweta*gBwetaeta + 4*gAwetaeta*gBweta))/((b*b*b*b)*r) - 2*B26*intx*inty*((b*b)*(fAw*fBwxi*gAw*gBweta + fAwxi*fBw*gAweta*gBw) + 2*r*(fAw*(fBw*gAweta*gBwetaeta*wxi + fBw*gAwetaeta*gBweta*wxi + 2*fBwxi*gAweta*gBweta*weta + fBwxi*gAwetaeta*gBw*weta) + fAwxi*fBw*weta*(gAw*gBwetaeta + 2*gAweta*gBweta)))/(a*(b*b*b)*r) - 8*B66*intx*inty*(fAw*fBwxi*gAweta*gBweta*wxi + fAwxi*(fBw*gAweta*gBweta*wxi + fBwxi*weta*(gAw*gBweta + gAweta*gBw)))/((a*a)*(b*b)) + 4*D11*fAwxixi*fBwxixi*gAw*gBw*intx*inty/(a*a*a*a) + 4*D12*intx*inty*(fAw*fBwxixi*gAwetaeta*gBw + fAwxixi*fBw*gAw*gBwetaeta)/((a*a)*(b*b)) + 8*D16*intx*inty*(fAwxi*fBwxixi*gAweta*gBw + fAwxixi*fBwxi*gAw*gBweta)/((a*a*a)*b) + 4*D22*fAw*fBw*gAwetaeta*gBwetaeta*intx*inty/(b*b*b*b) + 8*D26*intx*inty*(fAw*fBwxi*gAwetaeta*gBweta + fAwxi*fBw*gAweta*gBwetaeta)/(a*(b*b*b)) + 16*D66*fAwxi*fBwxi*gAweta*gBweta*intx*inty/((a*a)*(b*b)) )

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC


def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny, int NLgeom=0,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, r, intx, inty
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
    r = shell.r
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
                        eyy += cs[col+1]*(2/b)*fAv*gAveta + 1/r*cs[col+2]*fAw*gAw + 0.5*cs[col+2]*(2/b)*fAw*gAweta*(2/b)*weta
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
                                kGv[c] += weight*(intx*inty/4)*(
                                          Nxx*(2/a)*fAwxi*gAw*(2/a)*fBwxi*gBw
                                        + Nxy*((2/b)*fAw*gAweta*(2/a)*fBwxi*gBw + (2/a)*fAwxi*gAw*(2/b)*fBw*gBweta)
                                        + Nyy*(2/b)*fAw*gAweta*(2/b)*fBw*gBweta )

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG


def fkM_num(object shell, double offset, object hrho_input, int size,
        int row0, int col0, int nx, int ny):
    from .plate_clpt_donnell_bardell_num import fkM_num as plate_fkM_num
    return plate_fkM_num(shell, offset, hrho_input, size, row0, col0, nx, ny)


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
    from . plate_clpt_donnell_bardell_num import fkAy_num as plate_fkAy_num
    return plate_fkAy_num(shell, size, row0, col0, nx, ny)


def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, r, intx, inty
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, c, col, ptx, pty
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
    r = shell.r
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
                        eyy += cs[col+1]*(2/b)*fAv*gAveta + 1./r*cs[col+2]*fAw*gAw + 0.5*cs[col+2]*(2/b)*fAw*gAweta*(2/b)*weta
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

                        fint[col+0] += weight*( 0.25*intx*inty*(2*Nxx*fAuxi*gAu/a + 2*Nxy*fAu*gAueta/b) )
                        fint[col+1] += weight*( 0.25*intx*inty*(2*Nxy*fAvxi*gAv/a + 2*Nyy*fAv*gAveta/b) )
                        fint[col+2] += weight*( 0.25*intx*inty*(-4*Mxx*fAwxixi*gAw/(a*a) - 8*Mxy*fAwxi*gAweta/(a*b) - 4*Myy*fAw*gAwetaeta/(b*b) + 4*Nxx*fAwxi*gAw*wxi/(a*a) + 4*Nxy*(fAw*gAweta*wxi + fAwxi*gAw*weta)/(a*b) + Nyy*(fAw*gAw/r + 4*fAw*gAweta*weta/(b*b))) )

    return fint

