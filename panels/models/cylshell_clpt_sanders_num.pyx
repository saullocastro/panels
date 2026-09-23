#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Numerically integrated matrices of the cylindrical shell using the CLPT with
the Sanders-Koiter kinematics

See the kinematic equations in
``theory/shells/cylshell_clpt_sanders/cylshell_clpt_sanders.py``, from where
the integrands herein have been generated. In short, with ``y`` the arc
length along the circumference, the rotations of the normal are::

    phix = -w,x
    phiy = -w,y + v/r

and the strains are::

    exx = u,x + phix**2/2
    eyy = v,y + w/r + phiy**2/2
    gxy = u,y + v,x + phix*phiy
    kxx = -w,xx
    kyy = -w,yy + v,y/r
    kxy = -2*w,xy + 3/2*v,x/r - 1/2*u,y/r

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

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
    cdef double phix, phiy, NxxNL, NyyNL, NxyNL

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
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    fdim = 9*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kCr = np.zeros((fdim,), dtype=INT)
    kCc = np.zeros((fdim,), dtype=INT)
    kCv = np.zeros((fdim,), dtype=DOUBLE)

    xinf = 0
    xsup = shell.a
    xi1 = (x1 - xinf)/(xsup - xinf)*2 - 1
    xi2 = (x2 - xinf)/(xsup - xinf)*2 - 1

    yinf = 0
    ysup = shell.b
    eta1 = (y1 - yinf)/(ysup - yinf)*2 - 1
    eta2 = (y2 - yinf)/(ysup - yinf)*2 - 1

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

                # rotations of the normal, phix = -w,x and phiy = -w,y + v/r
                phix = 0
                phiy = 0
                if NLgeom == 1:
                    for j in range(n):
                        #TODO put these in a lookup vector
                        gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                        gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                        gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                        for i in range(m):
                            #TODO put these in a lookup vector
                            fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                            fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                            fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                            col = col0 + DOF*(j*m + i)

                            phix += -(2/a)*cs[col+2]*fAwxi*gAw
                            phiy += -(2/b)*cs[col+2]*fAw*gAweta + cs[col+1]*fAv*gAv/r

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

                # Membrane stress carried by the nonlinear strain
                # eps_NL = {phix^2/2, phiy^2/2, phix*phiy}. With it, KGNL = KG(N_NL)
                # is collected in kC such that
                #     KT = K0 + K0L + KL0 + KLL + KGNL (fkC_num) + KG(N0 + N_L) (fkG_num)
                # is the exact Jacobian of calc_fint, and fkG_num stays
                # homogeneous of degree one in cs, as linear buckling requires.
                # phix = phiy = 0 when NLgeom == 0, then KGNL vanishes
                NxxNL = A11*0.5*phix*phix + A12*0.5*phiy*phiy + A16*phix*phiy
                NyyNL = A12*0.5*phix*phix + A22*0.5*phiy*phiy + A26*phix*phiy
                NxyNL = A16*0.5*phix*phix + A26*0.5*phiy*phiy + A66*phix*phiy

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
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAuxi*fBuxi*gAu*gBu/(a*a) + 4*A16*(fAu*fBuxi*gAueta*gBu + fAuxi*fBu*gAu*gBueta)/(a*b) + 4*A66*fAu*fBu*gAueta*gBueta/(b*b) - 2*B16*(fAu*fBuxi*gAueta*gBu + fAuxi*fBu*gAu*gBueta)/(a*b*r) - 4*B66*fAu*fBu*gAueta*gBueta/((b*b)*r) + D66*fAu*fBu*gAueta*gBueta/((b*b)*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 2*A12*fAuxi*fBv*gAu*(b*gBv*phiy + 2*gBveta*r)/(a*b*r) + 2*A16*fAuxi*gAu*gBv*(a*fBv*phix + 2*fBvxi*r)/((a*a)*r) + 2*A26*fAu*fBv*gAueta*(b*gBv*phiy + 2*gBveta*r)/((b*b)*r) + 2*A66*fAu*gAueta*gBv*(a*fBv*phix + 2*fBvxi*r)/(a*b*r) + 4*B12*fAuxi*fBv*gAu*gBveta/(a*b*r) + 6*B16*fAuxi*fBvxi*gAu*gBv/((a*a)*r) + B26*fAu*fBv*gAueta*(-b*gBv*phiy + 2*gBveta*r)/((b*b)*(r*r)) + B66*fAu*gAueta*gBv*(-a*fBv*phix + 4*fBvxi*r)/(a*b*(r*r)) - 2*D26*fAu*fBv*gAueta*gBveta/((b*b)*(r*r)) - 3*D66*fAu*fBvxi*gAueta*gBv/(a*b*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( -4*A11*fAuxi*fBwxi*gAu*gBw*phix/(a*a) - 2*A12*fAuxi*fBw*gAu*(-b*gBw + 2*gBweta*phiy*r)/(a*b*r) - 4*A16*(a*fAu*fBwxi*gAueta*gBw*phix + a*fAuxi*fBw*gAu*gBweta*phix + b*fAuxi*fBwxi*gAu*gBw*phiy)/((a*a)*b) - 2*A26*fAu*fBw*gAueta*(-b*gBw + 2*gBweta*phiy*r)/((b*b)*r) - 4*A66*fAu*gAueta*(a*fBw*gBweta*phix + b*fBwxi*gBw*phiy)/(a*(b*b)) - 8*B11*fAuxi*fBwxixi*gAu*gBw/(a*a*a) - 8*B12*fAuxi*fBw*gAu*gBwetaeta/(a*(b*b)) - 2*B16*(-a*fAu*fBwxi*gAueta*gBw*phix + 4*fAu*fBwxixi*gAueta*gBw*r + 8*fAuxi*fBwxi*gAu*gBweta*r)/((a*a)*b*r) - B26*fAu*fBw*gAueta*((b*b)*gBw - 2*b*gBweta*phiy*r + 8*gBwetaeta*(r*r))/((b*b*b)*(r*r)) - 2*B66*fAu*gAueta*(-a*fBw*gBweta*phix - b*fBwxi*gBw*phiy + 8*fBwxi*gBweta*r)/(a*(b*b)*r) + 4*D16*fAu*fBwxixi*gAueta*gBw/((a*a)*b*r) + 4*D26*fAu*fBw*gAueta*gBwetaeta/((b*b*b)*r) + 8*D66*fAu*fBwxi*gAueta*gBweta/(a*(b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 2*A12*fAv*fBuxi*gBu*(b*gAv*phiy + 2*gAveta*r)/(a*b*r) + 2*A16*fBuxi*gAv*gBu*(a*fAv*phix + 2*fAvxi*r)/((a*a)*r) + 2*A26*fAv*fBu*gBueta*(b*gAv*phiy + 2*gAveta*r)/((b*b)*r) + 2*A66*fBu*gAv*gBueta*(a*fAv*phix + 2*fAvxi*r)/(a*b*r) + 4*B12*fAv*fBuxi*gAveta*gBu/(a*b*r) + 6*B16*fAvxi*fBuxi*gAv*gBu/((a*a)*r) + B26*fAv*fBu*gBueta*(-b*gAv*phiy + 2*gAveta*r)/((b*b)*(r*r)) + B66*fBu*gAv*gBueta*(-a*fAv*phix + 4*fAvxi*r)/(a*b*(r*r)) - 2*D26*fAv*fBu*gAveta*gBueta/((b*b)*(r*r)) - 3*D66*fAvxi*fBu*gAv*gBueta/(a*b*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( A22*fAv*fBv*(b*gAv*phiy + 2*gAveta*r)*(b*gBv*phiy + 2*gBveta*r)/((b*b)*(r*r)) + 2*A26*(a*b*fAv*fBv*gAv*gBv*phix*phiy + a*fAv*fBv*gAv*gBveta*phix*r + a*fAv*fBv*gAveta*gBv*phix*r + b*fAv*fBvxi*gAv*gBv*phiy*r + b*fAvxi*fBv*gAv*gBv*phiy*r + 2*fAv*fBvxi*gAveta*gBv*(r*r) + 2*fAvxi*fBv*gAv*gBveta*(r*r))/(a*b*(r*r)) + A66*gAv*gBv*(a*fAv*phix + 2*fAvxi*r)*(a*fBv*phix + 2*fBvxi*r)/((a*a)*(r*r)) + 2*B22*fAv*fBv*(b*gAv*gBveta*phiy + b*gAveta*gBv*phiy + 4*gAveta*gBveta*r)/((b*b)*(r*r)) + B26*(2*a*fAv*fBv*gAv*gBveta*phix + 2*a*fAv*fBv*gAveta*gBv*phix + 3*b*fAv*fBvxi*gAv*gBv*phiy + 3*b*fAvxi*fBv*gAv*gBv*phiy + 10*fAv*fBvxi*gAveta*gBv*r + 10*fAvxi*fBv*gAv*gBveta*r)/(a*b*(r*r)) + 3*B66*gAv*gBv*(a*fAv*fBvxi*phix + a*fAvxi*fBv*phix + 4*fAvxi*fBvxi*r)/((a*a)*(r*r)) + 4*D22*fAv*fBv*gAveta*gBveta/((b*b)*(r*r)) + 6*D26*(fAv*fBvxi*gAveta*gBv + fAvxi*fBv*gAv*gBveta)/(a*b*(r*r)) + 9*D66*fAvxi*fBvxi*gAv*gBv/((a*a)*(r*r)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( NyyNL*fAv*fBv*gAv*gBv/(r*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( -2*A12*fAv*fBwxi*gBw*phix*(b*gAv*phiy + 2*gAveta*r)/(a*b*r) - 2*A16*fBwxi*gAv*gBw*phix*(a*fAv*phix + 2*fAvxi*r)/((a*a)*r) - A22*fAv*fBw*(-b*gBw + 2*gBweta*phiy*r)*(b*gAv*phiy + 2*gAveta*r)/((b*b)*(r*r)) - A26*(-a*(b*b)*fAv*fBw*gAv*gBw*phix + 4*a*b*fAv*fBw*gAv*gBweta*phix*phiy*r + 4*a*fAv*fBw*gAveta*gBweta*phix*(r*r) + 2*(b*b)*fAv*fBwxi*gAv*gBw*(phiy*phiy)*r - 2*(b*b)*fAvxi*fBw*gAv*gBw*r + 4*b*fAv*fBwxi*gAveta*gBw*phiy*(r*r) + 4*b*fAvxi*fBw*gAv*gBweta*phiy*(r*r))/(a*(b*b)*(r*r)) - 2*A66*gAv*(a*fAv*phix + 2*fAvxi*r)*(a*fBw*gBweta*phix + b*fBwxi*gBw*phiy)/((a*a)*b*r) - 4*B12*fAv*gBw*(a*fBwxi*gAveta*phix + b*fBwxixi*gAv*phiy + 2*fBwxixi*gAveta*r)/((a*a)*b*r) - 2*B16*gAv*gBw*(2*a*fAv*fBwxixi*phix + 3*a*fAvxi*fBwxi*phix + 4*fAvxi*fBwxixi*r)/((a*a*a)*r) - 2*B22*fAv*fBw*(-(b*b)*gAveta*gBw + 2*b*gAv*gBwetaeta*phiy*r + 2*b*gAveta*gBweta*phiy*r + 4*gAveta*gBwetaeta*(r*r))/((b*b*b)*(r*r)) - B26*(4*a*fAv*fBw*gAv*gBwetaeta*phix*r + 4*a*fAv*fBw*gAveta*gBweta*phix*r - 3*(b*b)*fAvxi*fBw*gAv*gBw + 8*b*fAv*fBwxi*gAv*gBweta*phiy*r + 4*b*fAv*fBwxi*gAveta*gBw*phiy*r + 6*b*fAvxi*fBw*gAv*gBweta*phiy*r + 16*fAv*fBwxi*gAveta*gBweta*(r*r) + 8*fAvxi*fBw*gAv*gBwetaeta*(r*r))/(a*(b*b)*(r*r)) - 2*B66*gAv*(4*a*fAv*fBwxi*gBweta*phix + 3*a*fAvxi*fBw*gBweta*phix + 3*b*fAvxi*fBwxi*gBw*phiy + 8*fAvxi*fBwxi*gBweta*r)/((a*a)*b*r) - 8*D12*fAv*fBwxixi*gAveta*gBw/((a*a)*b*r) - 12*D16*fAvxi*fBwxixi*gAv*gBw/((a*a*a)*r) - 8*D22*fAv*fBw*gAveta*gBwetaeta/((b*b*b)*r) - 4*D26*(4*fAv*fBwxi*gAveta*gBweta + 3*fAvxi*fBw*gAv*gBwetaeta)/(a*(b*b)*r) - 24*D66*fAvxi*fBwxi*gAv*gBweta/((a*a)*b*r) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( -2*NxyNL*fAv*fBwxi*gAv*gBw/(a*r) - 2*NyyNL*fAv*fBw*gAv*gBweta/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( -4*A11*fAwxi*fBuxi*gAw*gBu*phix/(a*a) - 2*A12*fAw*fBuxi*gBu*(-b*gAw + 2*gAweta*phiy*r)/(a*b*r) - 4*A16*(a*fAw*fBuxi*gAweta*gBu*phix + a*fAwxi*fBu*gAw*gBueta*phix + b*fAwxi*fBuxi*gAw*gBu*phiy)/((a*a)*b) - 2*A26*fAw*fBu*gBueta*(-b*gAw + 2*gAweta*phiy*r)/((b*b)*r) - 4*A66*fBu*gBueta*(a*fAw*gAweta*phix + b*fAwxi*gAw*phiy)/(a*(b*b)) - 8*B11*fAwxixi*fBuxi*gAw*gBu/(a*a*a) - 8*B12*fAw*fBuxi*gAwetaeta*gBu/(a*(b*b)) - 2*B16*(-a*fAwxi*fBu*gAw*gBueta*phix + 8*fAwxi*fBuxi*gAweta*gBu*r + 4*fAwxixi*fBu*gAw*gBueta*r)/((a*a)*b*r) - B26*fAw*fBu*gBueta*((b*b)*gAw - 2*b*gAweta*phiy*r + 8*gAwetaeta*(r*r))/((b*b*b)*(r*r)) - 2*B66*fBu*gBueta*(-a*fAw*gAweta*phix - b*fAwxi*gAw*phiy + 8*fAwxi*gAweta*r)/(a*(b*b)*r) + 4*D16*fAwxixi*fBu*gAw*gBueta/((a*a)*b*r) + 4*D26*fAw*fBu*gAwetaeta*gBueta/((b*b*b)*r) + 8*D66*fAwxi*fBu*gAweta*gBueta/(a*(b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( -2*A12*fAwxi*fBv*gAw*phix*(b*gBv*phiy + 2*gBveta*r)/(a*b*r) - 2*A16*fAwxi*gAw*gBv*phix*(a*fBv*phix + 2*fBvxi*r)/((a*a)*r) - A22*fAw*fBv*(-b*gAw + 2*gAweta*phiy*r)*(b*gBv*phiy + 2*gBveta*r)/((b*b)*(r*r)) - A26*(-a*(b*b)*fAw*fBv*gAw*gBv*phix + 4*a*b*fAw*fBv*gAweta*gBv*phix*phiy*r + 4*a*fAw*fBv*gAweta*gBveta*phix*(r*r) - 2*(b*b)*fAw*fBvxi*gAw*gBv*r + 2*(b*b)*fAwxi*fBv*gAw*gBv*(phiy*phiy)*r + 4*b*fAw*fBvxi*gAweta*gBv*phiy*(r*r) + 4*b*fAwxi*fBv*gAw*gBveta*phiy*(r*r))/(a*(b*b)*(r*r)) - 2*A66*gBv*(a*fBv*phix + 2*fBvxi*r)*(a*fAw*gAweta*phix + b*fAwxi*gAw*phiy)/((a*a)*b*r) - 4*B12*fBv*gAw*(a*fAwxi*gBveta*phix + b*fAwxixi*gBv*phiy + 2*fAwxixi*gBveta*r)/((a*a)*b*r) - 2*B16*gAw*gBv*(3*a*fAwxi*fBvxi*phix + 2*a*fAwxixi*fBv*phix + 4*fAwxixi*fBvxi*r)/((a*a*a)*r) - 2*B22*fAw*fBv*(-(b*b)*gAw*gBveta + 2*b*gAweta*gBveta*phiy*r + 2*b*gAwetaeta*gBv*phiy*r + 4*gAwetaeta*gBveta*(r*r))/((b*b*b)*(r*r)) - B26*(4*a*fAw*fBv*gAweta*gBveta*phix*r + 4*a*fAw*fBv*gAwetaeta*gBv*phix*r - 3*(b*b)*fAw*fBvxi*gAw*gBv + 6*b*fAw*fBvxi*gAweta*gBv*phiy*r + 4*b*fAwxi*fBv*gAw*gBveta*phiy*r + 8*b*fAwxi*fBv*gAweta*gBv*phiy*r + 8*fAw*fBvxi*gAwetaeta*gBv*(r*r) + 16*fAwxi*fBv*gAweta*gBveta*(r*r))/(a*(b*b)*(r*r)) - 2*B66*gBv*(3*a*fAw*fBvxi*gAweta*phix + 4*a*fAwxi*fBv*gAweta*phix + 3*b*fAwxi*fBvxi*gAw*phiy + 8*fAwxi*fBvxi*gAweta*r)/((a*a)*b*r) - 8*D12*fAwxixi*fBv*gAw*gBveta/((a*a)*b*r) - 12*D16*fAwxixi*fBvxi*gAw*gBv/((a*a*a)*r) - 8*D22*fAw*fBv*gAwetaeta*gBveta/((b*b*b)*r) - 4*D26*(3*fAw*fBvxi*gAwetaeta*gBv + 4*fAwxi*fBv*gAweta*gBveta)/(a*(b*b)*r) - 24*D66*fAwxi*fBvxi*gAweta*gBv/((a*a)*b*r) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( -2*NxyNL*fAwxi*fBv*gAw*gBv/(a*r) - 2*NyyNL*fAw*fBv*gAweta*gBv/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAwxi*fBwxi*gAw*gBw*(phix*phix)/(a*a) + 2*A12*phix*(-b*fAw*fBwxi*gAw*gBw - b*fAwxi*fBw*gAw*gBw + 2*fAw*fBwxi*gAweta*gBw*phiy*r + 2*fAwxi*fBw*gAw*gBweta*phiy*r)/(a*b*r) + 4*A16*phix*(a*fAw*fBwxi*gAweta*gBw*phix + a*fAwxi*fBw*gAw*gBweta*phix + 2*b*fAwxi*fBwxi*gAw*gBw*phiy)/((a*a)*b) + A22*fAw*fBw*(-b*gAw + 2*gAweta*phiy*r)*(-b*gBw + 2*gBweta*phiy*r)/((b*b)*(r*r)) + 2*A26*(-a*b*fAw*fBw*gAw*gBweta*phix - a*b*fAw*fBw*gAweta*gBw*phix + 4*a*fAw*fBw*gAweta*gBweta*phix*phiy*r - (b*b)*fAw*fBwxi*gAw*gBw*phiy - (b*b)*fAwxi*fBw*gAw*gBw*phiy + 2*b*fAw*fBwxi*gAweta*gBw*(phiy*phiy)*r + 2*b*fAwxi*fBw*gAw*gBweta*(phiy*phiy)*r)/(a*(b*b)*r) + 4*A66*(a*fAw*gAweta*phix + b*fAwxi*gAw*phiy)*(a*fBw*gBweta*phix + b*fBwxi*gBw*phiy)/((a*a)*(b*b)) + 8*B11*gAw*gBw*phix*(fAwxi*fBwxixi + fAwxixi*fBwxi)/(a*a*a) + 4*B12*(2*a*fAw*fBwxi*gAwetaeta*gBw*phix*r + 2*a*fAwxi*fBw*gAw*gBwetaeta*phix*r - (b*b)*fAw*fBwxixi*gAw*gBw - (b*b)*fAwxixi*fBw*gAw*gBw + 2*b*fAw*fBwxixi*gAweta*gBw*phiy*r + 2*b*fAwxixi*fBw*gAw*gBweta*phiy*r)/((a*a)*(b*b)*r) + 8*B16*(a*fAw*fBwxixi*gAweta*gBw*phix + 2*a*fAwxi*fBwxi*gAw*gBweta*phix + 2*a*fAwxi*fBwxi*gAweta*gBw*phix + a*fAwxixi*fBw*gAw*gBweta*phix + b*fAwxi*fBwxixi*gAw*gBw*phiy + b*fAwxixi*fBwxi*gAw*gBw*phiy)/((a*a*a)*b) + 4*B22*fAw*fBw*(-b*gAw*gBwetaeta - b*gAwetaeta*gBw + 2*gAweta*gBwetaeta*phiy*r + 2*gAwetaeta*gBweta*phiy*r)/((b*b*b)*r) + 8*B26*(a*fAw*fBw*gAweta*gBwetaeta*phix*r + a*fAw*fBw*gAwetaeta*gBweta*phix*r - (b*b)*fAw*fBwxi*gAw*gBweta - (b*b)*fAwxi*fBw*gAweta*gBw + 2*b*fAw*fBwxi*gAweta*gBweta*phiy*r + b*fAw*fBwxi*gAwetaeta*gBw*phiy*r + b*fAwxi*fBw*gAw*gBwetaeta*phiy*r + 2*b*fAwxi*fBw*gAweta*gBweta*phiy*r)/(a*(b*b*b)*r) + 16*B66*(a*fAw*fBwxi*gAweta*gBweta*phix + a*fAwxi*fBw*gAweta*gBweta*phix + b*fAwxi*fBwxi*gAw*gBweta*phiy + b*fAwxi*fBwxi*gAweta*gBw*phiy)/((a*a)*(b*b)) + 16*D11*fAwxixi*fBwxixi*gAw*gBw/(a*a*a*a) + 16*D12*(fAw*fBwxixi*gAwetaeta*gBw + fAwxixi*fBw*gAw*gBwetaeta)/((a*a)*(b*b)) + 32*D16*(fAwxi*fBwxixi*gAweta*gBw + fAwxixi*fBwxi*gAw*gBweta)/((a*a*a)*b) + 16*D22*fAw*fBw*gAwetaeta*gBwetaeta/(b*b*b*b) + 32*D26*(fAw*fBwxi*gAwetaeta*gBweta + fAwxi*fBw*gAweta*gBwetaeta)/(a*(b*b*b)) + 64*D66*fAwxi*fBwxi*gAweta*gBweta/((a*a)*(b*b)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( 4*NxxNL*fAwxi*fBwxi*gAw*gBw/(a*a) + 4*NxyNL*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta)/(a*b) + 4*NyyNL*fAw*fBw*gAweta*gBweta/(b*b) )

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC


def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    """Geometric stiffness matrix of the linear membrane stress state

    The membrane stress used here comes from the *linear* part of the strain
    evaluated at ``cs``, superposed with the constant stress state
    ``(Nxx0, Nyy0, Nxy0)``. The result is therefore homogeneous of degree one
    in ``cs``, which is what the linear buckling eigenvalue problem requires.

    With Sanders' kinematics the rotation ``phiy = -w,y + v/r`` depends on
    ``v``, such that this matrix couples the ``v`` and ``w`` DOFs.

    There is deliberately no switch to include the stress of the non-linear
    strain here. That contribution, ``KGNL``, is collected by
    :func:`.fkC_num` instead, so that

        KT = K0 + K0L + KL0 + KLL + KGNL   (fkC_num)
             + KG(N0 + N_L)                (fkG_num)

    is the exact Jacobian of :func:`.calc_fint` while this function stays
    homogeneous of degree one in ``cs``. Adding ``KGNL`` here would
    double-count it in the tangent stiffness matrix and break linear
    buckling. See the comments in :func:`.fkC_num` and :func:`.calc_fint`.

    """
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
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef long [::1] kGr, kGc
    cdef double [::1] kGv

    cdef double fAu, fAuxi, fAv, fAvxi, fAw, fAwxi, fAwxixi
    cdef double fBv, fBw, fBwxi
    cdef double gAu, gAueta, gAv, gAveta, gAw, gAweta, gAwetaeta
    cdef double gBv, gBw, gBweta

    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double Nxx, Nyy, Nxy

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
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    fdim = 4*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kGr = np.zeros((fdim,), dtype=INT)
    kGc = np.zeros((fdim,), dtype=INT)
    kGv = np.zeros((fdim,), dtype=DOUBLE)

    xinf = 0
    xsup = shell.a
    xi1 = (x1 - xinf)/(xsup - xinf)*2 - 1
    xi2 = (x2 - xinf)/(xsup - xinf)*2 - 1

    yinf = 0
    ysup = shell.b
    eta1 = (y1 - yinf)/(ysup - yinf)*2 - 1
    eta2 = (y2 - yinf)/(ysup - yinf)*2 - 1

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

                # Calculating the linear strain components. The stress of the
                # nonlinear strain enters KT through KGNL in fkC_num, such that
                # kG is homogeneous of degree one in cs
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

                        exx += cs[col+0]*(2/a)*fAuxi*gAu
                        eyy += cs[col+1]*(2/b)*fAv*gAveta + 1/r*cs[col+2]*fAw*gAw
                        gxy += cs[col+0]*(2/b)*fAu*gAueta + cs[col+1]*(2/a)*fAvxi*gAv
                        kxx += -cs[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAwetaeta + 1/r*cs[col+1]*(2/b)*fAv*gAveta
                        kxy += (-2*cs[col+2]*(2/a)*fAwxi*(2/b)*gAweta
                                + 1.5/r*cs[col+1]*(2/a)*fAvxi*gAv - 0.5/r*cs[col+0]*(2/b)*fAu*gAueta)

                # Calculating membrane stress components
                Nxx = Nxx0 + A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = Nyy0 + A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = Nxy0 + A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy

                # kG
                c = -1
                for i in range(m):
                    fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                    fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                    fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                    for k in range(m):
                        fBv = f(k, xi, x1v, x1vr, x2v, x2vr)
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)
                        fBwxi = fp(k, xi, x1w, x1wr, x2w, x2wr)

                        for j in range(n):
                            gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                            gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                            gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBv = f(l, eta, y1v, y1vr, y2v, y2vr)
                                gBw = f(l, eta, y1w, y1wr, y2w, y2wr)
                                gBweta = fp(l, eta, y1w, y1wr, y2w, y2wr)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+1
                                kGv[c] += weight*(intx*inty/4)*( Nyy*fAv*fBv*gAv*gBv/(r*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+2
                                kGv[c] += weight*(intx*inty/4)*( -2*Nxy*fAv*fBwxi*gAv*gBw/(a*r) - 2*Nyy*fAv*fBw*gAv*gBweta/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+1
                                kGv[c] += weight*(intx*inty/4)*( -2*Nxy*fAwxi*fBv*gAw*gBv/(a*r) - 2*Nyy*fAw*fBv*gAweta*gBv/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(intx*inty/4)*( 4*Nxx*fAwxi*fBwxi*gAw*gBw/(a*a) + 4*Nxy*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta)/(a*b) + 4*Nyy*fAw*fBw*gAweta*gBweta/(b*b) )

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG


def fkM_num(object shell, double offset, object hrho_input, int size,
        int row0, int col0, int nx, int ny):
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

    cdef long [::1] kMr, kMc
    cdef double [::1] kMv

    cdef double fAu, fAv, fAw, fAwxi
    cdef double fBu, fBv, fBw, fBwxi
    cdef double gAu, gAv, gAw, gAweta
    cdef double gBu, gBv, gBw, gBweta
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # hrho as 3-D matrix, must be [nx, ny, 2], when there is one pair (h, rho)
    # for each of the nx * ny integration points
    cdef double h, rho, d
    cdef double [:, :, ::1] hrho_nxny

    cdef int one_hrho_each_point = 0

    d = offset

    hrho_input = np.asarray(hrho_input, dtype=DOUBLE)
    if hrho_input.shape == (nx, ny, 2):
        hrho_nxny = np.ascontiguousarray(hrho_input)
        one_hrho_each_point = 1
    elif hrho_input.shape == (2,):
        # creating dummy 3-D array that is not used
        hrho_nxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        # using a constant (h, rho) for all the integration domain
        hrho_input = np.ascontiguousarray(hrho_input)
        h, rho = hrho_input
    else:
        raise ValueError('Invalid shape for hrho_input!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    r = shell.r
    m = shell.m
    n = shell.n
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    fdim = 7*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kMr = np.zeros((fdim,), dtype=INT)
    kMc = np.zeros((fdim,), dtype=INT)
    kMv = np.zeros((fdim,), dtype=DOUBLE)

    xinf = 0
    xsup = shell.a
    xi1 = (x1 - xinf)/(xsup - xinf)*2 - 1
    xi2 = (x2 - xinf)/(xsup - xinf)*2 - 1

    yinf = 0
    ysup = shell.b
    eta1 = (y1 - yinf)/(ysup - yinf)*2 - 1
    eta2 = (y2 - yinf)/(ysup - yinf)*2 - 1

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
                                kMv[c] += weight*(intx*inty/4)*( fAu*fBu*gAu*gBu*h*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( 2*d*fAu*fBwxi*gAu*gBw*h*rho/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( 0.08333333333333333*fAv*fBv*gAv*gBv*h*rho*(12*(d*d) - 24*d*r + (h*h) + 12*(r*r))/(r*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( 0.16666666666666666*fAv*fBw*gAv*gBweta*h*rho*(-12*(d*d) + 12*d*r - (h*h))/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( 2*d*fAwxi*fBu*gAw*gBu*h*rho/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( 0.16666666666666666*fAw*fBv*gAweta*gBv*h*rho*(-12*(d*d) + 12*d*r - (h*h))/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( 0.3333333333333333*h*rho*(3*(a*a)*(b*b)*fAw*fBw*gAw*gBw + 12*(a*a)*(d*d)*fAw*fBw*gAweta*gBweta + (a*a)*fAw*fBw*gAweta*gBweta*(h*h) + 12*(b*b)*(d*d)*fAwxi*fBwxi*gAw*gBw + (b*b)*fAwxi*fBwxi*gAw*gBw*(h*h))/((a*a)*(b*b)) )

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


#NOTE the aerodynamic matrices depend only on w and are therefore the same as
#     for the Donnell kinematics
def fkAx_num(object shell, int size, int row0, int col0, int nx, int ny):
    from .cylshell_clpt_donnell_num import fkAx_num as donnell_fkAx_num
    return donnell_fkAx_num(shell, size, row0, col0, nx, ny)


def fkAy_num(object shell, int size, int row0, int col0, int nx, int ny):
    from .cylshell_clpt_donnell_num import fkAy_num as donnell_fkAy_num
    return donnell_fkAy_num(shell, size, row0, col0, nx, ny)


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
    cdef double phix, phiy

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
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    fint = np.zeros(size, dtype=DOUBLE)

    xinf = 0
    xsup = shell.a
    xi1 = (x1 - xinf)/(xsup - xinf)*2 - 1
    xi2 = (x2 - xinf)/(xsup - xinf)*2 - 1

    yinf = 0
    ysup = shell.b
    eta1 = (y1 - yinf)/(ysup - yinf)*2 - 1
    eta2 = (y2 - yinf)/(ysup - yinf)*2 - 1

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

                # rotations of the normal, phix = -w,x and phiy = -w,y + v/r
                phix = 0
                phiy = 0
                for j in range(n):
                    #TODO save in buffer
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    for i in range(m):
                        #TODO save in buffer
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

                        phix += -(2/a)*cs[col+2]*fAwxi*gAw
                        phiy += -(2/b)*cs[col+2]*fAw*gAweta + cs[col+1]*fAv*gAv/r

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

                        exx += cs[col+0]*(2/a)*fAuxi*gAu
                        eyy += cs[col+1]*(2/b)*fAv*gAveta + 1/r*cs[col+2]*fAw*gAw
                        gxy += cs[col+0]*(2/b)*fAu*gAueta + cs[col+1]*(2/a)*fAvxi*gAv
                        kxx += -cs[col+2]*(2/a*2/a)*fAwxixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAwetaeta + 1/r*cs[col+1]*(2/b)*fAv*gAveta
                        kxy += (-2*cs[col+2]*(2/a)*fAwxi*(2/b)*gAweta
                                + 1.5/r*cs[col+1]*(2/a)*fAvxi*gAv - 0.5/r*cs[col+0]*(2/b)*fAu*gAueta)

                # nonlinear strain eps_NL = {phix^2/2, phiy^2/2, phix*phiy}
                exx += 0.5*phix*phix
                eyy += 0.5*phiy*phiy
                gxy += phix*phiy

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

                        fint[col+0] += weight*(intx*inty/4)*( -Mxy*fAu*gAueta/(b*r) + 2*Nxx*fAuxi*gAu/a + 2*Nxy*fAu*gAueta/b )
                        fint[col+1] += weight*(intx*inty/4)*( 3*Mxy*fAvxi*gAv/(a*r) + 2*Myy*fAv*gAveta/(b*r) + Nxy*gAv*(a*fAv*phix + 2*fAvxi*r)/(a*r) + Nyy*fAv*(b*gAv*phiy + 2*gAveta*r)/(b*r) )
                        fint[col+2] += weight*(intx*inty/4)*( -4*Mxx*fAwxixi*gAw/(a*a) - 8*Mxy*fAwxi*gAweta/(a*b) - 4*Myy*fAw*gAwetaeta/(b*b) - 2*Nxx*fAwxi*gAw*phix/a - 2*Nxy*(a*fAw*gAweta*phix + b*fAwxi*gAw*phiy)/(a*b) - Nyy*fAw*(-b*gAw + 2*gAweta*phiy*r)/(b*r) )

    return fint
