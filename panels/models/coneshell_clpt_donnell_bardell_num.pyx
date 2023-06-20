#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np


cdef extern from 'math.h':
    double sin(double a) nogil
    double cos(double a) nogil

cdef extern from 'bardell_functions_uv.hpp':
    double fuv(int i, double xi, double xi1t, double xi2t) nogil
    double fuv_x(int i, double xi, double xi1t, double xi2t) nogil

cdef extern from 'bardell_functions_w.hpp':
    double fw(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double fw_x(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil
    double fw_xx(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil

cdef extern from 'legendre_gauss_quadrature.hpp':
    void leggauss_quad(int n, double *points, double* weights) nogil


DOUBLE = np.float64
INT = np.int64

cdef int num = 3


def fkC_num(double [::1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double a, b, ra, rb, sina, cosa, x, r, rbot, bbot, alpharad
    cdef int m, n
    cdef double x1u, x2u
    cdef double x1v, x2v
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y2u
    cdef double y1v, y2v
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
    bbot = shell.b
    ra = shell.ra
    rb = shell.rb
    alpharad = shell.alpharad
    m = shell.m
    n = shell.n
    x1u = shell.x1u; x2u = shell.x2u
    x1v = shell.x1v; x2v = shell.x2v
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y2u = shell.y2u
    y1v = shell.y1v; y2v = shell.y2v
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

    rbot = shell.r

    with nogil:
        sina = sin(alpharad)
        cosa = cos(alpharad)

        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                weight = weights_xi[ptx]*weights_eta[pty]

                x = a*(xi + 1)/2

                r = rbot - sina*(x)

                b = r*bbot/rbot

                wxi = 0
                weta = 0
                if NLgeom == 1:
                    for j in range(n):
                        #TODO put these in a lookup vector
                        gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)
                        gAweta = fw_x(j, eta, y1w, y1wr, y2w, y2wr)
                        for i in range(m):
                            #TODO put these in a lookup vector
                            fAw = fw(i, xi, x1w, x1wr, x2w, x2wr)
                            fAwxi = fw_x(i, xi, x1w, x1wr, x2w, x2wr)

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
                    fAu = fuv(i, xi, x1u, x2u)
                    fAuxi = fuv_x(i, xi, x1u, x2u)
                    fAv = fuv(i, xi, x1v, x2v)
                    fAvxi = fuv_x(i, xi, x1v, x2v)
                    fAw = fw(i, xi, x1w, x1wr, x2w, x2wr)
                    fAwxi = fw_x(i, xi, x1w, x1wr, x2w, x2wr)
                    fAwxixi = fw_xx(i, xi, x1w, x1wr, x2w, x2wr)

                    for k in range(m):
                        fBu = fuv(k, xi, x1u, x2u)
                        fBuxi = fuv_x(k, xi, x1u, x2u)
                        fBv = fuv(k, xi, x1v, x2v)
                        fBvxi = fuv_x(k, xi, x1v, x2v)
                        fBw = fw(k, xi, x1w, x1wr, x2w, x2wr)
                        fBwxi = fw_x(k, xi, x1w, x1wr, x2w, x2wr)
                        fBwxixi = fw_xx(k, xi, x1w, x1wr, x2w, x2wr)

                        for j in range(n):
                            gAu = fuv(j, eta, y1u, y2u)
                            gAueta = fuv_x(j, eta, y1u, y2u)
                            gAv = fuv(j, eta, y1v, y2v)
                            gAveta = fuv_x(j, eta, y1v, y2v)
                            gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)
                            gAweta = fw_x(j, eta, y1w, y1wr, y2w, y2wr)
                            gAwetaeta = fw_xx(j, eta, y1w, y1wr, y2w, y2wr)

                            for l in range(n):

                                row = row0 + num*(j*m + i)
                                col = col0 + num*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = fuv(l, eta, y1u, y2u)
                                gBueta = fuv_x(l, eta, y1u, y2u)
                                gBv = fuv(l, eta, y1v, y2v)
                                gBveta = fuv_x(l, eta, y1v, y2v)
                                gBw = fw(l, eta, y1w, y1wr, y2w, y2wr)
                                gBweta = fw_x(l, eta, y1w, y1wr, y2w, y2wr)
                                gBwetaeta = fw_xx(l, eta, y1w, y1wr, y2w, y2wr)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(A11*b*fAuxi*fBuxi*gAu*gBu/a + A12*(0.5*b*fAu*fBuxi*gAu*gBu*sina/r + 0.5*b*fAuxi*fBu*gAu*gBu*sina/r) + A16*(fAu*fBuxi*gAueta*gBu + fAuxi*fBu*gAu*gBueta) + 0.25*A22*a*b*fAu*fBu*gAu*gBu*(sina*sina)/(r*r) + A26*(0.5*a*fAu*fBu*gAu*gBueta*sina/r + 0.5*a*fAu*fBu*gAueta*gBu*sina/r) + A66*a*fAu*fBu*gAueta*gBueta/b)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(A12*fAuxi*fBv*gAu*gBveta + A16*(-0.5*b*fAuxi*fBv*gAu*gBv*sina/r + b*fAuxi*fBvxi*gAu*gBv/a) + 0.5*A22*a*fAu*fBv*gAu*gBveta*sina/r + A26*(-0.25*a*b*fAu*fBv*gAu*gBv*(sina*sina)/(r*r) + a*fAu*fBv*gAueta*gBveta/b + 0.5*b*fAu*fBvxi*gAu*gBv*sina/r) + A66*(-0.5*a*fAu*fBv*gAueta*gBv*sina/r + fAu*fBvxi*gAueta*gBv))
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(0.5*A12*b*cosa*fAuxi*fBw*gAu*gBw/r + 0.25*A22*a*b*cosa*fAu*fBw*gAu*gBw*sina/(r*r) + 0.5*A26*a*cosa*fAu*fBw*gAueta*gBw/r - 2*B11*b*fAuxi*fBwxixi*gAu*gBw/(a*a) + B12*(-2*fAuxi*fBw*gAu*gBwetaeta/b - b*fAu*fBwxixi*gAu*gBw*sina/(a*r) - b*fAuxi*fBwxi*gAu*gBw*sina/(a*r)) + B16*(fAuxi*fBw*gAu*gBweta*sina/r - 2*fAu*fBwxixi*gAueta*gBw/a - 4*fAuxi*fBwxi*gAu*gBweta/a) + B22*(-a*fAu*fBw*gAu*gBwetaeta*sina/(b*r) - 0.5*b*fAu*fBwxi*gAu*gBw*(sina*sina)/(r*r)) + B26*(0.5*a*fAu*fBw*gAu*gBweta*(sina*sina)/(r*r) - 2*a*fAu*fBw*gAueta*gBwetaeta/(b*b) - 2*fAu*fBwxi*gAu*gBweta*sina/r - fAu*fBwxi*gAueta*gBw*sina/r) + B66*(a*fAu*fBw*gAueta*gBweta*sina/(b*r) - 4*fAu*fBwxi*gAueta*gBweta/b))
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(A12*fAv*fBuxi*gAveta*gBu + A16*(-0.5*b*fAv*fBuxi*gAv*gBu*sina/r + b*fAvxi*fBuxi*gAv*gBu/a) + 0.5*A22*a*fAv*fBu*gAveta*gBu*sina/r + A26*(-0.25*a*b*fAv*fBu*gAv*gBu*(sina*sina)/(r*r) + a*fAv*fBu*gAveta*gBueta/b + 0.5*b*fAvxi*fBu*gAv*gBu*sina/r) + A66*(-0.5*a*fAv*fBu*gAv*gBueta*sina/r + fAvxi*fBu*gAv*gBueta))
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(A22*a*fAv*fBv*gAveta*gBveta/b + A26*(-0.5*a*fAv*fBv*gAv*gBveta*sina/r - 0.5*a*fAv*fBv*gAveta*gBv*sina/r + fAv*fBvxi*gAveta*gBv + fAvxi*fBv*gAv*gBveta) + A66*(0.25*a*b*fAv*fBv*gAv*gBv*(sina*sina)/(r*r) - 0.5*b*fAv*fBvxi*gAv*gBv*sina/r - 0.5*b*fAvxi*fBv*gAv*gBv*sina/r + b*fAvxi*fBvxi*gAv*gBv/a))
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(0.5*A22*a*cosa*fAv*fBw*gAveta*gBw/r + A26*(-0.25*a*b*cosa*fAv*fBw*gAv*gBw*sina/(r*r) + 0.5*b*cosa*fAvxi*fBw*gAv*gBw/r) - 2*B12*fAv*fBwxixi*gAveta*gBw/a + B16*(b*fAv*fBwxixi*gAv*gBw*sina/(a*r) - 2*b*fAvxi*fBwxixi*gAv*gBw/(a*a)) + B22*(-2*a*fAv*fBw*gAveta*gBwetaeta/(b*b) - fAv*fBwxi*gAveta*gBw*sina/r) + B26*(a*fAv*fBw*gAv*gBwetaeta*sina/(b*r) + a*fAv*fBw*gAveta*gBweta*sina/(b*r) + 0.5*b*fAv*fBwxi*gAv*gBw*(sina*sina)/(r*r) - 4*fAv*fBwxi*gAveta*gBweta/b - 2*fAvxi*fBw*gAv*gBwetaeta/b - b*fAvxi*fBwxi*gAv*gBw*sina/(a*r)) + B66*(-0.5*a*fAv*fBw*gAv*gBweta*(sina*sina)/(r*r) + 2*fAv*fBwxi*gAv*gBweta*sina/r + fAvxi*fBw*gAv*gBweta*sina/r - 4*fAvxi*fBwxi*gAv*gBweta/a))
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(0.5*A12*b*cosa*fAw*fBuxi*gAw*gBu/r + 0.25*A22*a*b*cosa*fAw*fBu*gAw*gBu*sina/(r*r) + 0.5*A26*a*cosa*fAw*fBu*gAw*gBueta/r - 2*B11*b*fAwxixi*fBuxi*gAw*gBu/(a*a) + B12*(-2*fAw*fBuxi*gAwetaeta*gBu/b - b*fAwxi*fBuxi*gAw*gBu*sina/(a*r) - b*fAwxixi*fBu*gAw*gBu*sina/(a*r)) + B16*(fAw*fBuxi*gAweta*gBu*sina/r - 4*fAwxi*fBuxi*gAweta*gBu/a - 2*fAwxixi*fBu*gAw*gBueta/a) + B22*(-a*fAw*fBu*gAwetaeta*gBu*sina/(b*r) - 0.5*b*fAwxi*fBu*gAw*gBu*(sina*sina)/(r*r)) + B26*(0.5*a*fAw*fBu*gAweta*gBu*(sina*sina)/(r*r) - 2*a*fAw*fBu*gAwetaeta*gBueta/(b*b) - fAwxi*fBu*gAw*gBueta*sina/r - 2*fAwxi*fBu*gAweta*gBu*sina/r) + B66*(a*fAw*fBu*gAweta*gBueta*sina/(b*r) - 4*fAwxi*fBu*gAweta*gBueta/b))
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(0.5*A22*a*cosa*fAw*fBv*gAw*gBveta/r + A26*(-0.25*a*b*cosa*fAw*fBv*gAw*gBv*sina/(r*r) + 0.5*b*cosa*fAw*fBvxi*gAw*gBv/r) - 2*B12*fAwxixi*fBv*gAw*gBveta/a + B16*(b*fAwxixi*fBv*gAw*gBv*sina/(a*r) - 2*b*fAwxixi*fBvxi*gAw*gBv/(a*a)) + B22*(-2*a*fAw*fBv*gAwetaeta*gBveta/(b*b) - fAwxi*fBv*gAw*gBveta*sina/r) + B26*(a*fAw*fBv*gAweta*gBveta*sina/(b*r) + a*fAw*fBv*gAwetaeta*gBv*sina/(b*r) + 0.5*b*fAwxi*fBv*gAw*gBv*(sina*sina)/(r*r) - 2*fAw*fBvxi*gAwetaeta*gBv/b - 4*fAwxi*fBv*gAweta*gBveta/b - b*fAwxi*fBvxi*gAw*gBv*sina/(a*r)) + B66*(-0.5*a*fAw*fBv*gAweta*gBv*(sina*sina)/(r*r) + fAw*fBvxi*gAweta*gBv*sina/r + 2*fAwxi*fBv*gAweta*gBv*sina/r - 4*fAwxi*fBvxi*gAweta*gBv/a))
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(0.25*A22*a*b*(cosa*cosa)*fAw*fBw*gAw*gBw/(r*r) + B12*(-b*cosa*fAw*fBwxixi*gAw*gBw/(a*r) - b*cosa*fAwxixi*fBw*gAw*gBw/(a*r)) + B22*(-a*cosa*fAw*fBw*gAw*gBwetaeta/(b*r) - a*cosa*fAw*fBw*gAwetaeta*gBw/(b*r) - 0.5*b*cosa*fAw*fBwxi*gAw*gBw*sina/(r*r) - 0.5*b*cosa*fAwxi*fBw*gAw*gBw*sina/(r*r)) + B26*(0.5*a*cosa*fAw*fBw*gAw*gBweta*sina/(r*r) + 0.5*a*cosa*fAw*fBw*gAweta*gBw*sina/(r*r) - 2*cosa*fAw*fBwxi*gAw*gBweta/r - 2*cosa*fAwxi*fBw*gAweta*gBw/r) + 4*D11*b*fAwxixi*fBwxixi*gAw*gBw/(a*a*a) + D12*(4*fAw*fBwxixi*gAwetaeta*gBw/(a*b) + 4*fAwxixi*fBw*gAw*gBwetaeta/(a*b) + 2*b*fAwxi*fBwxixi*gAw*gBw*sina/((a*a)*r) + 2*b*fAwxixi*fBwxi*gAw*gBw*sina/((a*a)*r)) + D16*(-2*fAw*fBwxixi*gAweta*gBw*sina/(a*r) - 2*fAwxixi*fBw*gAw*gBweta*sina/(a*r) + 8*fAwxi*fBwxixi*gAweta*gBw/(a*a) + 8*fAwxixi*fBwxi*gAw*gBweta/(a*a)) + D22*(4*a*fAw*fBw*gAwetaeta*gBwetaeta/(b*b*b) + 2*fAw*fBwxi*gAwetaeta*gBw*sina/(b*r) + 2*fAwxi*fBw*gAw*gBwetaeta*sina/(b*r) + b*fAwxi*fBwxi*gAw*gBw*(sina*sina)/(a*(r*r))) + D26*(-2*a*fAw*fBw*gAweta*gBwetaeta*sina/((b*b)*r) - 2*a*fAw*fBw*gAwetaeta*gBweta*sina/((b*b)*r) - fAw*fBwxi*gAweta*gBw*(sina*sina)/(r*r) - fAwxi*fBw*gAw*gBweta*(sina*sina)/(r*r) + 8*fAw*fBwxi*gAwetaeta*gBweta/(b*b) + 8*fAwxi*fBw*gAweta*gBwetaeta/(b*b) + 4*fAwxi*fBwxi*gAw*gBweta*sina/(a*r) + 4*fAwxi*fBwxi*gAweta*gBw*sina/(a*r)) + D66*(a*fAw*fBw*gAweta*gBweta*(sina*sina)/(b*(r*r)) - 4*fAw*fBwxi*gAweta*gBweta*sina/(b*r) - 4*fAwxi*fBw*gAweta*gBweta*sina/(b*r) + 16*fAwxi*fBwxi*gAweta*gBweta/(a*b)))

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC


def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double a, b, r
    cdef int m, n
    cdef double x1u, x2u
    cdef double x1v, x2v
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y2u
    cdef double y1v, y2v
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, k, j, l, c, row, col, ptx, pty
    cdef double xi, eta, x, y, weight

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

    raise NotImplementedError('coneshell not ready yet')

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
    x1u = shell.x1u; x2u = shell.x2u
    x1v = shell.x1v; x2v = shell.x2v
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y2u = shell.y2u
    y1v = shell.y1v; y2v = shell.y2v
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
                        gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)
                        gAweta = fw_x(j, eta, y1w, y1wr, y2w, y2wr)
                        for i in range(m):
                            fAw = fw(i, xi, x1w, x1wr, x2w, x2wr)
                            fAwxi = fw_x(i, xi, x1w, x1wr, x2w, x2wr)

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
                    gAu = fuv(j, eta, y1u, y2u)
                    gAv = fuv(j, eta, y1v, y2v)
                    gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)
                    gAueta = fuv_x(j, eta, y1u, y2u)
                    gAveta = fuv_x(j, eta, y1v, y2v)
                    gAweta = fw_x(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fw_xx(j, eta, y1w, y1wr, y2w, y2wr)

                    for i in range(m):
                        fAu = fuv(i, xi, x1u, x2u)
                        fAv = fuv(i, xi, x1v, x2v)
                        fAw = fw(i, xi, x1w, x1wr, x2w, x2wr)
                        fAuxi = fuv_x(i, xi, x1u, x2u)
                        fAvxi = fuv_x(i, xi, x1v, x2v)
                        fAwxi = fw_x(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fw_xx(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + num*(j*m + i)

                        exx += cs[col+0]*(2/a)*fAuxi*gAu + 0.5*cs[col+2]*(2/a)*fAwxi*gAw*(2/a)*wxi
                        eyy += cs[col+1]*(2/b)*fAv*gAveta + cosa*1/r*cs[col+2]*fAw*gAw + 0.5*cs[col+2]*(2/b)*fAw*gAweta*(2/b)*weta
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
                    gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fw_x(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):
                        gBw = fw(l, eta, y1w, y1wr, y2w, y2wr)
                        gBweta = fw_x(l, eta, y1w, y1wr, y2w, y2wr)

                        for i in range(m):
                            fAw = fw(i, xi, x1w, x1wr, x2w, x2wr)
                            fAwxi = fw_x(i, xi, x1w, x1wr, x2w, x2wr)

                            for k in range(m):
                                fBw = fw(k, xi, x1w, x1wr, x2w, x2wr)
                                fBwxi = fw_x(k, xi, x1w, x1wr, x2w, x2wr)

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


def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    cdef double a, b, r
    cdef int m, n
    cdef double x1u, x2u
    cdef double x1v, x2v
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y2u
    cdef double y1v, y2v
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, c, col, ptx, pty
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double Nxx, Nyy, Nxy, Mxx, Myy, Mxy
    cdef double exx, eyy, gxy, kxx, kyy, kxy

    cdef double xi, eta, weight
    cdef double wxi, weta

    cdef double fAu, fAuxi, fAv, fAvxi, fAw, fAwxi, fAwxixi
    cdef double fBu, fBuxi, fBv, fBvxi, fBw, fBwxi, fBwxixi
    cdef double gAu, gAueta, gAv, gAveta, gAw, gAweta, gAwetaeta
    cdef double gBu, gBueta, gBv, gBveta, gBw, gBweta, gBwetaeta

    cdef double [::1] xis, etas, weights_xi, weights_eta, fint

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    raise NotImplementedError('coneshell not ready yet')

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
    x1u = shell.x1u; x2u = shell.x2u
    x1v = shell.x1v; x2v = shell.x2v
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y2u = shell.y2u
    y1v = shell.y1v; y2v = shell.y2v
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

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
                    gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fw_x(j, eta, y1w, y1wr, y2w, y2wr)
                    for i in range(m):
                        #TODO save in buffer
                        fAw = fw(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fw_x(i, xi, x1w, x1wr, x2w, x2wr)

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
                    gAu = fuv(j, eta, y1u, y2u)
                    gAueta = fuv_x(j, eta, y1u, y2u)
                    gAv = fuv(j, eta, y1v, y2v)
                    gAveta = fuv_x(j, eta, y1v, y2v)
                    gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fw_x(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fw_xx(j, eta, y1w, y1wr, y2w, y2wr)

                    for i in range(m):
                        #TODO save in buffer
                        fAu = fuv(i, xi, x1u, x2u)
                        fAuxi = fuv_x(i, xi, x1u, x2u)
                        fAv = fuv(i, xi, x1v, x2v)
                        fAvxi = fuv_x(i, xi, x1v, x2v)
                        fAw = fw(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fw_x(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fw_xx(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + num*(j*m + i)

                        exx += cs[col+0]*(2/a)*fAuxi*gAu + 0.5*cs[col+2]*(2/a)*fAwxi*gAw*(2/a)*wxi
                        eyy += cs[col+1]*(2/b)*fAv*gAveta + cosa*1./r*cs[col+2]*fAw*gAw + 0.5*cs[col+2]*(2/b)*fAw*gAweta*(2/b)*weta
                        gxy += cs[col+0]*(2/b)*fAu*gAueta + cs[col+1]*(2/a)*fAvxi*gAv
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
                    gAu = fuv(j, eta, y1u, y2u)
                    gAueta = fuv_x(j, eta, y1u, y2u)
                    gAv = fuv(j, eta, y1v, y2v)
                    gAveta = fuv_x(j, eta, y1v, y2v)
                    gAw = fw(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fw_x(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fw_xx(j, eta, y1w, y1wr, y2w, y2wr)
                    for i in range(m):
                        fAu = fuv(i, xi, x1u, x2u)
                        fAuxi = fuv_x(i, xi, x1u, x2u)
                        fAv = fuv(i, xi, x1v, x2v)
                        fAvxi = fuv_x(i, xi, x1v, x2v)
                        fAw = fw(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fw_x(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fw_xx(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + num*(j*m + i)

                        fint[col+0] += weight*( 0.25*a*b * ((2/a)*fAuxi*gAu*Nxx + (2/b)*fAu*gAueta*Nxy) )
                        fint[col+1] += weight*( 0.25*a*b * ((2/b)*fAv*gAveta*Nyy + (2/a)*fAvxi*gAv*Nxy) )
                        fint[col+2] += weight*( 0.25*a*b * ((2/a)*fAwxi*gAw*(2/a)*wxi*Nxx + 1./r*fAw*gAw*Nyy + (2/b)*fAw*gAweta*(2/b)*weta*Nyy + (2/a*2/b)*(fAwxi*gAw*weta + wxi*fAw*gAweta)*Nxy - (2/a*2/a)*fAwxixi*gAw*Mxx - (2/b*2/b)*fAw*gAwetaeta*Myy -2*(2/a*2/b)*fAwxi*gAweta*Mxy) )

    return fint
