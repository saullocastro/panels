#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
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
    cdef double [:, ::1] gtab, ftab
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

    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw, fAw_xi, fAw_xixi
    cdef double fBu, fBu_xi, fBv, fBv_xi, fBw, fBw_xi, fBw_xixi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw, gAw_eta, gAw_etaeta
    cdef double gBu, gBu_eta, gBv, gBv_eta, gBw, gBw_eta, gBw_etaeta
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

    gtab = np.zeros((7, n), dtype=DOUBLE)
    ftab = np.zeros((7, m), dtype=DOUBLE)

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
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[1, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[2, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[3, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[4, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[5, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[1, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[2, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[3, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[4, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[5, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                if NLgeom == 1:
                    for j in range(n):
                        gAv = gtab[0, j]
                        gAw = gtab[1, j]
                        gAw_eta = gtab[2, j]
                        for i in range(m):
                            fAv = ftab[0, i]
                            fAw = ftab[1, i]
                            fAw_xi = ftab[2, i]

                            col = col0 + DOF*(j*m + i)

                            phix += -(2/a)*cs[col+2]*fAw_xi*gAw
                            phiy += -(2/b)*cs[col+2]*fAw*gAw_eta + cs[col+1]*fAv*gAv/r

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
                #     KT = KC0 + KCNL + KGNL (fkC_num) + KG(Nbar + N_L) (fkG_num)
                # is the exact Jacobian of calc_fint, and fkG_num stays
                # homogeneous of degree one in cs, as linear buckling requires.
                # phix = phiy = 0 when NLgeom == 0, then KGNL vanishes
                NxxNL = A11*0.5*phix*phix + A12*0.5*phiy*phiy + A16*phix*phiy
                NyyNL = A12*0.5*phix*phix + A22*0.5*phiy*phiy + A26*phix*phiy
                NxyNL = A16*0.5*phix*phix + A26*0.5*phiy*phiy + A66*phix*phiy

                # kC
                c = -1
                for i in range(m):
                    fAu = ftab[3, i]
                    fAu_xi = ftab[4, i]
                    fAv = ftab[0, i]
                    fAv_xi = ftab[5, i]
                    fAw = ftab[1, i]
                    fAw_xi = ftab[2, i]
                    fAw_xixi = ftab[6, i]

                    for k in range(m):
                        fBu = ftab[3, k]
                        fBu_xi = ftab[4, k]
                        fBv = ftab[0, k]
                        fBv_xi = ftab[5, k]
                        fBw = ftab[1, k]
                        fBw_xi = ftab[2, k]
                        fBw_xixi = ftab[6, k]

                        for j in range(n):
                            gAu = gtab[3, j]
                            gAu_eta = gtab[4, j]
                            gAv = gtab[0, j]
                            gAv_eta = gtab[5, j]
                            gAw = gtab[1, j]
                            gAw_eta = gtab[2, j]
                            gAw_etaeta = gtab[6, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = gtab[3, l]
                                gBu_eta = gtab[4, l]
                                gBv = gtab[0, l]
                                gBv_eta = gtab[5, l]
                                gBw = gtab[1, l]
                                gBw_eta = gtab[2, l]
                                gBw_etaeta = gtab[6, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAu_xi*fBu_xi*gAu*gBu/(a*a) + 4*A16*(fAu*fBu_xi*gAu_eta*gBu + fAu_xi*fBu*gAu*gBu_eta)/(a*b) + 4*A66*fAu*fBu*gAu_eta*gBu_eta/(b*b) - 2*B16*(fAu*fBu_xi*gAu_eta*gBu + fAu_xi*fBu*gAu*gBu_eta)/(a*b*r) - 4*B66*fAu*fBu*gAu_eta*gBu_eta/((b*b)*r) + D66*fAu*fBu*gAu_eta*gBu_eta/((b*b)*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 2*A12*fAu_xi*fBv*gAu*(b*gBv*phiy + 2*gBv_eta*r)/(a*b*r) + 2*A16*fAu_xi*gAu*gBv*(a*fBv*phix + 2*fBv_xi*r)/((a*a)*r) + 2*A26*fAu*fBv*gAu_eta*(b*gBv*phiy + 2*gBv_eta*r)/((b*b)*r) + 2*A66*fAu*gAu_eta*gBv*(a*fBv*phix + 2*fBv_xi*r)/(a*b*r) + 4*B12*fAu_xi*fBv*gAu*gBv_eta/(a*b*r) + 6*B16*fAu_xi*fBv_xi*gAu*gBv/((a*a)*r) + B26*fAu*fBv*gAu_eta*(-b*gBv*phiy + 2*gBv_eta*r)/((b*b)*(r*r)) + B66*fAu*gAu_eta*gBv*(-a*fBv*phix + 4*fBv_xi*r)/(a*b*(r*r)) - 2*D26*fAu*fBv*gAu_eta*gBv_eta/((b*b)*(r*r)) - 3*D66*fAu*fBv_xi*gAu_eta*gBv/(a*b*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( -4*A11*fAu_xi*fBw_xi*gAu*gBw*phix/(a*a) - 2*A12*fAu_xi*fBw*gAu*(-b*gBw + 2*gBw_eta*phiy*r)/(a*b*r) - 4*A16*(a*fAu*fBw_xi*gAu_eta*gBw*phix + a*fAu_xi*fBw*gAu*gBw_eta*phix + b*fAu_xi*fBw_xi*gAu*gBw*phiy)/((a*a)*b) - 2*A26*fAu*fBw*gAu_eta*(-b*gBw + 2*gBw_eta*phiy*r)/((b*b)*r) - 4*A66*fAu*gAu_eta*(a*fBw*gBw_eta*phix + b*fBw_xi*gBw*phiy)/(a*(b*b)) - 8*B11*fAu_xi*fBw_xixi*gAu*gBw/(a*a*a) - 8*B12*fAu_xi*fBw*gAu*gBw_etaeta/(a*(b*b)) - 2*B16*(-a*fAu*fBw_xi*gAu_eta*gBw*phix + 4*fAu*fBw_xixi*gAu_eta*gBw*r + 8*fAu_xi*fBw_xi*gAu*gBw_eta*r)/((a*a)*b*r) - B26*fAu*fBw*gAu_eta*((b*b)*gBw - 2*b*gBw_eta*phiy*r + 8*gBw_etaeta*(r*r))/((b*b*b)*(r*r)) - 2*B66*fAu*gAu_eta*(-a*fBw*gBw_eta*phix - b*fBw_xi*gBw*phiy + 8*fBw_xi*gBw_eta*r)/(a*(b*b)*r) + 4*D16*fAu*fBw_xixi*gAu_eta*gBw/((a*a)*b*r) + 4*D26*fAu*fBw*gAu_eta*gBw_etaeta/((b*b*b)*r) + 8*D66*fAu*fBw_xi*gAu_eta*gBw_eta/(a*(b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 2*A12*fAv*fBu_xi*gBu*(b*gAv*phiy + 2*gAv_eta*r)/(a*b*r) + 2*A16*fBu_xi*gAv*gBu*(a*fAv*phix + 2*fAv_xi*r)/((a*a)*r) + 2*A26*fAv*fBu*gBu_eta*(b*gAv*phiy + 2*gAv_eta*r)/((b*b)*r) + 2*A66*fBu*gAv*gBu_eta*(a*fAv*phix + 2*fAv_xi*r)/(a*b*r) + 4*B12*fAv*fBu_xi*gAv_eta*gBu/(a*b*r) + 6*B16*fAv_xi*fBu_xi*gAv*gBu/((a*a)*r) + B26*fAv*fBu*gBu_eta*(-b*gAv*phiy + 2*gAv_eta*r)/((b*b)*(r*r)) + B66*fBu*gAv*gBu_eta*(-a*fAv*phix + 4*fAv_xi*r)/(a*b*(r*r)) - 2*D26*fAv*fBu*gAv_eta*gBu_eta/((b*b)*(r*r)) - 3*D66*fAv_xi*fBu*gAv*gBu_eta/(a*b*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( A22*fAv*fBv*(b*gAv*phiy + 2*gAv_eta*r)*(b*gBv*phiy + 2*gBv_eta*r)/((b*b)*(r*r)) + 2*A26*(a*b*fAv*fBv*gAv*gBv*phix*phiy + a*fAv*fBv*gAv*gBv_eta*phix*r + a*fAv*fBv*gAv_eta*gBv*phix*r + b*fAv*fBv_xi*gAv*gBv*phiy*r + b*fAv_xi*fBv*gAv*gBv*phiy*r + 2*fAv*fBv_xi*gAv_eta*gBv*(r*r) + 2*fAv_xi*fBv*gAv*gBv_eta*(r*r))/(a*b*(r*r)) + A66*gAv*gBv*(a*fAv*phix + 2*fAv_xi*r)*(a*fBv*phix + 2*fBv_xi*r)/((a*a)*(r*r)) + 2*B22*fAv*fBv*(b*gAv*gBv_eta*phiy + b*gAv_eta*gBv*phiy + 4*gAv_eta*gBv_eta*r)/((b*b)*(r*r)) + B26*(2*a*fAv*fBv*gAv*gBv_eta*phix + 2*a*fAv*fBv*gAv_eta*gBv*phix + 3*b*fAv*fBv_xi*gAv*gBv*phiy + 3*b*fAv_xi*fBv*gAv*gBv*phiy + 10*fAv*fBv_xi*gAv_eta*gBv*r + 10*fAv_xi*fBv*gAv*gBv_eta*r)/(a*b*(r*r)) + 3*B66*gAv*gBv*(a*fAv*fBv_xi*phix + a*fAv_xi*fBv*phix + 4*fAv_xi*fBv_xi*r)/((a*a)*(r*r)) + 4*D22*fAv*fBv*gAv_eta*gBv_eta/((b*b)*(r*r)) + 6*D26*(fAv*fBv_xi*gAv_eta*gBv + fAv_xi*fBv*gAv*gBv_eta)/(a*b*(r*r)) + 9*D66*fAv_xi*fBv_xi*gAv*gBv/((a*a)*(r*r)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( NyyNL*fAv*fBv*gAv*gBv/(r*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( -2*A12*fAv*fBw_xi*gBw*phix*(b*gAv*phiy + 2*gAv_eta*r)/(a*b*r) - 2*A16*fBw_xi*gAv*gBw*phix*(a*fAv*phix + 2*fAv_xi*r)/((a*a)*r) - A22*fAv*fBw*(-b*gBw + 2*gBw_eta*phiy*r)*(b*gAv*phiy + 2*gAv_eta*r)/((b*b)*(r*r)) - A26*(-a*(b*b)*fAv*fBw*gAv*gBw*phix + 4*a*b*fAv*fBw*gAv*gBw_eta*phix*phiy*r + 4*a*fAv*fBw*gAv_eta*gBw_eta*phix*(r*r) + 2*(b*b)*fAv*fBw_xi*gAv*gBw*(phiy*phiy)*r - 2*(b*b)*fAv_xi*fBw*gAv*gBw*r + 4*b*fAv*fBw_xi*gAv_eta*gBw*phiy*(r*r) + 4*b*fAv_xi*fBw*gAv*gBw_eta*phiy*(r*r))/(a*(b*b)*(r*r)) - 2*A66*gAv*(a*fAv*phix + 2*fAv_xi*r)*(a*fBw*gBw_eta*phix + b*fBw_xi*gBw*phiy)/((a*a)*b*r) - 4*B12*fAv*gBw*(a*fBw_xi*gAv_eta*phix + b*fBw_xixi*gAv*phiy + 2*fBw_xixi*gAv_eta*r)/((a*a)*b*r) - 2*B16*gAv*gBw*(2*a*fAv*fBw_xixi*phix + 3*a*fAv_xi*fBw_xi*phix + 4*fAv_xi*fBw_xixi*r)/((a*a*a)*r) - 2*B22*fAv*fBw*(-(b*b)*gAv_eta*gBw + 2*b*gAv*gBw_etaeta*phiy*r + 2*b*gAv_eta*gBw_eta*phiy*r + 4*gAv_eta*gBw_etaeta*(r*r))/((b*b*b)*(r*r)) - B26*(4*a*fAv*fBw*gAv*gBw_etaeta*phix*r + 4*a*fAv*fBw*gAv_eta*gBw_eta*phix*r - 3*(b*b)*fAv_xi*fBw*gAv*gBw + 8*b*fAv*fBw_xi*gAv*gBw_eta*phiy*r + 4*b*fAv*fBw_xi*gAv_eta*gBw*phiy*r + 6*b*fAv_xi*fBw*gAv*gBw_eta*phiy*r + 16*fAv*fBw_xi*gAv_eta*gBw_eta*(r*r) + 8*fAv_xi*fBw*gAv*gBw_etaeta*(r*r))/(a*(b*b)*(r*r)) - 2*B66*gAv*(4*a*fAv*fBw_xi*gBw_eta*phix + 3*a*fAv_xi*fBw*gBw_eta*phix + 3*b*fAv_xi*fBw_xi*gBw*phiy + 8*fAv_xi*fBw_xi*gBw_eta*r)/((a*a)*b*r) - 8*D12*fAv*fBw_xixi*gAv_eta*gBw/((a*a)*b*r) - 12*D16*fAv_xi*fBw_xixi*gAv*gBw/((a*a*a)*r) - 8*D22*fAv*fBw*gAv_eta*gBw_etaeta/((b*b*b)*r) - 4*D26*(4*fAv*fBw_xi*gAv_eta*gBw_eta + 3*fAv_xi*fBw*gAv*gBw_etaeta)/(a*(b*b)*r) - 24*D66*fAv_xi*fBw_xi*gAv*gBw_eta/((a*a)*b*r) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( -2*NxyNL*fAv*fBw_xi*gAv*gBw/(a*r) - 2*NyyNL*fAv*fBw*gAv*gBw_eta/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( -4*A11*fAw_xi*fBu_xi*gAw*gBu*phix/(a*a) - 2*A12*fAw*fBu_xi*gBu*(-b*gAw + 2*gAw_eta*phiy*r)/(a*b*r) - 4*A16*(a*fAw*fBu_xi*gAw_eta*gBu*phix + a*fAw_xi*fBu*gAw*gBu_eta*phix + b*fAw_xi*fBu_xi*gAw*gBu*phiy)/((a*a)*b) - 2*A26*fAw*fBu*gBu_eta*(-b*gAw + 2*gAw_eta*phiy*r)/((b*b)*r) - 4*A66*fBu*gBu_eta*(a*fAw*gAw_eta*phix + b*fAw_xi*gAw*phiy)/(a*(b*b)) - 8*B11*fAw_xixi*fBu_xi*gAw*gBu/(a*a*a) - 8*B12*fAw*fBu_xi*gAw_etaeta*gBu/(a*(b*b)) - 2*B16*(-a*fAw_xi*fBu*gAw*gBu_eta*phix + 8*fAw_xi*fBu_xi*gAw_eta*gBu*r + 4*fAw_xixi*fBu*gAw*gBu_eta*r)/((a*a)*b*r) - B26*fAw*fBu*gBu_eta*((b*b)*gAw - 2*b*gAw_eta*phiy*r + 8*gAw_etaeta*(r*r))/((b*b*b)*(r*r)) - 2*B66*fBu*gBu_eta*(-a*fAw*gAw_eta*phix - b*fAw_xi*gAw*phiy + 8*fAw_xi*gAw_eta*r)/(a*(b*b)*r) + 4*D16*fAw_xixi*fBu*gAw*gBu_eta/((a*a)*b*r) + 4*D26*fAw*fBu*gAw_etaeta*gBu_eta/((b*b*b)*r) + 8*D66*fAw_xi*fBu*gAw_eta*gBu_eta/(a*(b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( -2*A12*fAw_xi*fBv*gAw*phix*(b*gBv*phiy + 2*gBv_eta*r)/(a*b*r) - 2*A16*fAw_xi*gAw*gBv*phix*(a*fBv*phix + 2*fBv_xi*r)/((a*a)*r) - A22*fAw*fBv*(-b*gAw + 2*gAw_eta*phiy*r)*(b*gBv*phiy + 2*gBv_eta*r)/((b*b)*(r*r)) - A26*(-a*(b*b)*fAw*fBv*gAw*gBv*phix + 4*a*b*fAw*fBv*gAw_eta*gBv*phix*phiy*r + 4*a*fAw*fBv*gAw_eta*gBv_eta*phix*(r*r) - 2*(b*b)*fAw*fBv_xi*gAw*gBv*r + 2*(b*b)*fAw_xi*fBv*gAw*gBv*(phiy*phiy)*r + 4*b*fAw*fBv_xi*gAw_eta*gBv*phiy*(r*r) + 4*b*fAw_xi*fBv*gAw*gBv_eta*phiy*(r*r))/(a*(b*b)*(r*r)) - 2*A66*gBv*(a*fBv*phix + 2*fBv_xi*r)*(a*fAw*gAw_eta*phix + b*fAw_xi*gAw*phiy)/((a*a)*b*r) - 4*B12*fBv*gAw*(a*fAw_xi*gBv_eta*phix + b*fAw_xixi*gBv*phiy + 2*fAw_xixi*gBv_eta*r)/((a*a)*b*r) - 2*B16*gAw*gBv*(3*a*fAw_xi*fBv_xi*phix + 2*a*fAw_xixi*fBv*phix + 4*fAw_xixi*fBv_xi*r)/((a*a*a)*r) - 2*B22*fAw*fBv*(-(b*b)*gAw*gBv_eta + 2*b*gAw_eta*gBv_eta*phiy*r + 2*b*gAw_etaeta*gBv*phiy*r + 4*gAw_etaeta*gBv_eta*(r*r))/((b*b*b)*(r*r)) - B26*(4*a*fAw*fBv*gAw_eta*gBv_eta*phix*r + 4*a*fAw*fBv*gAw_etaeta*gBv*phix*r - 3*(b*b)*fAw*fBv_xi*gAw*gBv + 6*b*fAw*fBv_xi*gAw_eta*gBv*phiy*r + 4*b*fAw_xi*fBv*gAw*gBv_eta*phiy*r + 8*b*fAw_xi*fBv*gAw_eta*gBv*phiy*r + 8*fAw*fBv_xi*gAw_etaeta*gBv*(r*r) + 16*fAw_xi*fBv*gAw_eta*gBv_eta*(r*r))/(a*(b*b)*(r*r)) - 2*B66*gBv*(3*a*fAw*fBv_xi*gAw_eta*phix + 4*a*fAw_xi*fBv*gAw_eta*phix + 3*b*fAw_xi*fBv_xi*gAw*phiy + 8*fAw_xi*fBv_xi*gAw_eta*r)/((a*a)*b*r) - 8*D12*fAw_xixi*fBv*gAw*gBv_eta/((a*a)*b*r) - 12*D16*fAw_xixi*fBv_xi*gAw*gBv/((a*a*a)*r) - 8*D22*fAw*fBv*gAw_etaeta*gBv_eta/((b*b*b)*r) - 4*D26*(3*fAw*fBv_xi*gAw_etaeta*gBv + 4*fAw_xi*fBv*gAw_eta*gBv_eta)/(a*(b*b)*r) - 24*D66*fAw_xi*fBv_xi*gAw_eta*gBv/((a*a)*b*r) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( -2*NxyNL*fAw_xi*fBv*gAw*gBv/(a*r) - 2*NyyNL*fAw*fBv*gAw_eta*gBv/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAw_xi*fBw_xi*gAw*gBw*(phix*phix)/(a*a) + 2*A12*phix*(-b*fAw*fBw_xi*gAw*gBw - b*fAw_xi*fBw*gAw*gBw + 2*fAw*fBw_xi*gAw_eta*gBw*phiy*r + 2*fAw_xi*fBw*gAw*gBw_eta*phiy*r)/(a*b*r) + 4*A16*phix*(a*fAw*fBw_xi*gAw_eta*gBw*phix + a*fAw_xi*fBw*gAw*gBw_eta*phix + 2*b*fAw_xi*fBw_xi*gAw*gBw*phiy)/((a*a)*b) + A22*fAw*fBw*(-b*gAw + 2*gAw_eta*phiy*r)*(-b*gBw + 2*gBw_eta*phiy*r)/((b*b)*(r*r)) + 2*A26*(-a*b*fAw*fBw*gAw*gBw_eta*phix - a*b*fAw*fBw*gAw_eta*gBw*phix + 4*a*fAw*fBw*gAw_eta*gBw_eta*phix*phiy*r - (b*b)*fAw*fBw_xi*gAw*gBw*phiy - (b*b)*fAw_xi*fBw*gAw*gBw*phiy + 2*b*fAw*fBw_xi*gAw_eta*gBw*(phiy*phiy)*r + 2*b*fAw_xi*fBw*gAw*gBw_eta*(phiy*phiy)*r)/(a*(b*b)*r) + 4*A66*(a*fAw*gAw_eta*phix + b*fAw_xi*gAw*phiy)*(a*fBw*gBw_eta*phix + b*fBw_xi*gBw*phiy)/((a*a)*(b*b)) + 8*B11*gAw*gBw*phix*(fAw_xi*fBw_xixi + fAw_xixi*fBw_xi)/(a*a*a) + 4*B12*(2*a*fAw*fBw_xi*gAw_etaeta*gBw*phix*r + 2*a*fAw_xi*fBw*gAw*gBw_etaeta*phix*r - (b*b)*fAw*fBw_xixi*gAw*gBw - (b*b)*fAw_xixi*fBw*gAw*gBw + 2*b*fAw*fBw_xixi*gAw_eta*gBw*phiy*r + 2*b*fAw_xixi*fBw*gAw*gBw_eta*phiy*r)/((a*a)*(b*b)*r) + 8*B16*(a*fAw*fBw_xixi*gAw_eta*gBw*phix + 2*a*fAw_xi*fBw_xi*gAw*gBw_eta*phix + 2*a*fAw_xi*fBw_xi*gAw_eta*gBw*phix + a*fAw_xixi*fBw*gAw*gBw_eta*phix + b*fAw_xi*fBw_xixi*gAw*gBw*phiy + b*fAw_xixi*fBw_xi*gAw*gBw*phiy)/((a*a*a)*b) + 4*B22*fAw*fBw*(-b*gAw*gBw_etaeta - b*gAw_etaeta*gBw + 2*gAw_eta*gBw_etaeta*phiy*r + 2*gAw_etaeta*gBw_eta*phiy*r)/((b*b*b)*r) + 8*B26*(a*fAw*fBw*gAw_eta*gBw_etaeta*phix*r + a*fAw*fBw*gAw_etaeta*gBw_eta*phix*r - (b*b)*fAw*fBw_xi*gAw*gBw_eta - (b*b)*fAw_xi*fBw*gAw_eta*gBw + 2*b*fAw*fBw_xi*gAw_eta*gBw_eta*phiy*r + b*fAw*fBw_xi*gAw_etaeta*gBw*phiy*r + b*fAw_xi*fBw*gAw*gBw_etaeta*phiy*r + 2*b*fAw_xi*fBw*gAw_eta*gBw_eta*phiy*r)/(a*(b*b*b)*r) + 16*B66*(a*fAw*fBw_xi*gAw_eta*gBw_eta*phix + a*fAw_xi*fBw*gAw_eta*gBw_eta*phix + b*fAw_xi*fBw_xi*gAw*gBw_eta*phiy + b*fAw_xi*fBw_xi*gAw_eta*gBw*phiy)/((a*a)*(b*b)) + 16*D11*fAw_xixi*fBw_xixi*gAw*gBw/(a*a*a*a) + 16*D12*(fAw*fBw_xixi*gAw_etaeta*gBw + fAw_xixi*fBw*gAw*gBw_etaeta)/((a*a)*(b*b)) + 32*D16*(fAw_xi*fBw_xixi*gAw_eta*gBw + fAw_xixi*fBw_xi*gAw*gBw_eta)/((a*a*a)*b) + 16*D22*fAw*fBw*gAw_etaeta*gBw_etaeta/(b*b*b*b) + 32*D26*(fAw*fBw_xi*gAw_etaeta*gBw_eta + fAw_xi*fBw*gAw_eta*gBw_etaeta)/(a*(b*b*b)) + 64*D66*fAw_xi*fBw_xi*gAw_eta*gBw_eta/((a*a)*(b*b)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( 4*NxxNL*fAw_xi*fBw_xi*gAw*gBw/(a*a) + 4*NxyNL*(fAw*fBw_xi*gAw_eta*gBw + fAw_xi*fBw*gAw*gBw_eta)/(a*b) + 4*NyyNL*fAw*fBw*gAw_eta*gBw_eta/(b*b) )

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

        KT = KC0 + KCNL + KGNL   (fkC_num)
             + KG(Nbar + N_L)    (fkG_num)

    is the exact Jacobian of :func:`.calc_fint` while this function stays
    homogeneous of degree one in ``cs``. Adding ``KGNL`` here would
    double-count it in the tangent stiffness matrix and break linear
    buckling. See the comments in :func:`.fkC_num` and :func:`.calc_fint`.

    """
    cdef double [:, ::1] gtab, ftab
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

    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw, fAw_xi, fAw_xixi
    cdef double fBv, fBw, fBw_xi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw, gAw_eta, gAw_etaeta
    cdef double gBv, gBw, gBw_eta

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

    gtab = np.zeros((7, n), dtype=DOUBLE)
    ftab = np.zeros((7, m), dtype=DOUBLE)

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
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[2, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[3, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[4, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[5, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[2, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[3, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[4, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[5, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAu = gtab[0, j]
                    gAv = gtab[1, j]
                    gAw = gtab[2, j]
                    gAu_eta = gtab[3, j]
                    gAv_eta = gtab[4, j]
                    gAw_eta = gtab[5, j]
                    gAw_etaeta = gtab[6, j]

                    for i in range(m):
                        fAu = ftab[0, i]
                        fAv = ftab[1, i]
                        fAw = ftab[2, i]
                        fAu_xi = ftab[3, i]
                        fAv_xi = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        fAw_xixi = ftab[6, i]

                        col = col0 + DOF*(j*m + i)

                        exx += cs[col+0]*(2/a)*fAu_xi*gAu
                        eyy += cs[col+1]*(2/b)*fAv*gAv_eta + 1/r*cs[col+2]*fAw*gAw
                        gxy += cs[col+0]*(2/b)*fAu*gAu_eta + cs[col+1]*(2/a)*fAv_xi*gAv
                        kxx += -cs[col+2]*(2/a*2/a)*fAw_xixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAw_etaeta + 1/r*cs[col+1]*(2/b)*fAv*gAv_eta
                        kxy += (-2*cs[col+2]*(2/a)*fAw_xi*(2/b)*gAw_eta
                                + 1.5/r*cs[col+1]*(2/a)*fAv_xi*gAv - 0.5/r*cs[col+0]*(2/b)*fAu*gAu_eta)

                # Calculating membrane stress components
                Nxx = Nxx0 + A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = Nyy0 + A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = Nxy0 + A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy

                # kG
                c = -1
                for i in range(m):
                    fAv = ftab[1, i]
                    fAw = ftab[2, i]
                    fAw_xi = ftab[5, i]

                    for k in range(m):
                        fBv = ftab[1, k]
                        fBw = ftab[2, k]
                        fBw_xi = ftab[5, k]

                        for j in range(n):
                            gAv = gtab[1, j]
                            gAw = gtab[2, j]
                            gAw_eta = gtab[5, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBv = gtab[1, l]
                                gBw = gtab[2, l]
                                gBw_eta = gtab[5, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+1
                                kGv[c] += weight*(intx*inty/4)*( Nyy*fAv*fBv*gAv*gBv/(r*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+2
                                kGv[c] += weight*(intx*inty/4)*( -2*Nxy*fAv*fBw_xi*gAv*gBw/(a*r) - 2*Nyy*fAv*fBw*gAv*gBw_eta/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+1
                                kGv[c] += weight*(intx*inty/4)*( -2*Nxy*fAw_xi*fBv*gAw*gBv/(a*r) - 2*Nyy*fAw*fBv*gAw_eta*gBv/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(intx*inty/4)*( 4*Nxx*fAw_xi*fBw_xi*gAw*gBw/(a*a) + 4*Nxy*(fAw*fBw_xi*gAw_eta*gBw + fAw_xi*fBw*gAw*gBw_eta)/(a*b) + 4*Nyy*fAw*fBw*gAw_eta*gBw_eta/(b*b) )

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG


def fkM_num(object shell, object I_input, int size,
        int row0, int col0, int nx, int ny):
    cdef double [:, ::1] ftab, gtab
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

    cdef double fAu, fAv, fAw, fAw_xi
    cdef double fBu, fBv, fBw, fBw_xi
    cdef double gAu, gAv, gAw, gAw_eta
    cdef double gBu, gBv, gBw, gBw_eta
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # moments of the density through the thickness, I0, I1, I2, as a 3-D
    # matrix [nx, ny, 3] with one set for each of the nx * ny integration
    # points, or as a 1-D array [3] for the whole integration domain
    cdef double I0, I1, I2
    cdef double [:, :, ::1] I_nxny

    cdef int one_set_each_point = 0

    I_input = np.asarray(I_input, dtype=DOUBLE)
    if I_input.shape == (nx, ny, 3):
        I_nxny = np.ascontiguousarray(I_input)
        one_set_each_point = 1
    elif I_input.shape == (3, ):
        # creating dummy 3-D array that is not used
        I_nxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        # using constant moments for all the integration domain
        I0, I1, I2 = I_input
    else:
        raise ValueError('Invalid shape for I_input, it must be (nx, ny, 3) or (3, )')

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

    ftab = np.zeros((4, m), dtype=DOUBLE)
    gtab = np.zeros((4, n), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

                if one_set_each_point == 1:
                    I0 = I_nxny[ptx, pty, 0]
                    I1 = I_nxny[ptx, pty, 1]
                    I2 = I_nxny[ptx, pty, 2]

                # kM
                c = -1
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[2, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[3, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[2, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[3, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)

                for i in range(m):
                    fAu = ftab[0, i]
                    fAv = ftab[1, i]
                    fAw = ftab[2, i]
                    fAw_xi = ftab[3, i]

                    for k in range(m):
                        fBu = ftab[0, k]
                        fBv = ftab[1, k]
                        fBw = ftab[2, k]
                        fBw_xi = ftab[3, k]

                        for j in range(n):
                            gAu = gtab[0, j]
                            gAv = gtab[1, j]
                            gAw = gtab[2, j]
                            gAw_eta = gtab[3, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = gtab[0, l]
                                gBv = gtab[1, l]
                                gBw = gtab[2, l]
                                gBw_eta = gtab[3, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( I0*fAu*fBu*gAu*gBu )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( -2*I1*fAu*fBw_xi*gAu*gBw/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( fAv*fBv*gAv*gBv*(I0*(r*r) + 2*I1*r + I2)/(r*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( -2*fAv*fBw*gAv*gBw_eta*(I1*r + I2)/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( -2*I1*fAw_xi*fBu*gAw*gBu/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( -2*fAw*fBv*gAw_eta*gBv*(I1*r + I2)/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( (I0*(a*a)*(b*b)*fAw*fBw*gAw*gBw + 4*I2*(a*a)*fAw*fBw*gAw_eta*gBw_eta + 4*I2*(b*b)*fAw_xi*fBw_xi*gAw*gBw)/((a*a)*(b*b)) )

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
    cdef double [:, ::1] gtab, ftab
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

    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw, fAw_xi, fAw_xixi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw, gAw_eta, gAw_etaeta

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

    gtab = np.zeros((7, n), dtype=DOUBLE)
    ftab = np.zeros((7, m), dtype=DOUBLE)

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
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[1, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[2, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[3, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[4, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[5, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[1, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[2, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[3, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[4, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[5, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAv = gtab[0, j]
                    gAw = gtab[1, j]
                    gAw_eta = gtab[2, j]
                    for i in range(m):
                        fAv = ftab[0, i]
                        fAw = ftab[1, i]
                        fAw_xi = ftab[2, i]

                        col = col0 + DOF*(j*m + i)

                        phix += -(2/a)*cs[col+2]*fAw_xi*gAw
                        phiy += -(2/b)*cs[col+2]*fAw*gAw_eta + cs[col+1]*fAv*gAv/r

                # current strain state
                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.

                for j in range(n):
                    gAu = gtab[3, j]
                    gAu_eta = gtab[4, j]
                    gAv = gtab[0, j]
                    gAv_eta = gtab[5, j]
                    gAw = gtab[1, j]
                    gAw_eta = gtab[2, j]
                    gAw_etaeta = gtab[6, j]

                    for i in range(m):
                        fAu = ftab[3, i]
                        fAu_xi = ftab[4, i]
                        fAv = ftab[0, i]
                        fAv_xi = ftab[5, i]
                        fAw = ftab[1, i]
                        fAw_xi = ftab[2, i]
                        fAw_xixi = ftab[6, i]

                        col = col0 + DOF*(j*m + i)

                        exx += cs[col+0]*(2/a)*fAu_xi*gAu
                        eyy += cs[col+1]*(2/b)*fAv*gAv_eta + 1/r*cs[col+2]*fAw*gAw
                        gxy += cs[col+0]*(2/b)*fAu*gAu_eta + cs[col+1]*(2/a)*fAv_xi*gAv
                        kxx += -cs[col+2]*(2/a*2/a)*fAw_xixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAw_etaeta + 1/r*cs[col+1]*(2/b)*fAv*gAv_eta
                        kxy += (-2*cs[col+2]*(2/a)*fAw_xi*(2/b)*gAw_eta
                                + 1.5/r*cs[col+1]*(2/a)*fAv_xi*gAv - 0.5/r*cs[col+0]*(2/b)*fAu*gAu_eta)

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
                    gAu = gtab[3, j]
                    gAu_eta = gtab[4, j]
                    gAv = gtab[0, j]
                    gAv_eta = gtab[5, j]
                    gAw = gtab[1, j]
                    gAw_eta = gtab[2, j]
                    gAw_etaeta = gtab[6, j]
                    for i in range(m):
                        fAu = ftab[3, i]
                        fAu_xi = ftab[4, i]
                        fAv = ftab[0, i]
                        fAv_xi = ftab[5, i]
                        fAw = ftab[1, i]
                        fAw_xi = ftab[2, i]
                        fAw_xixi = ftab[6, i]

                        col = col0 + DOF*(j*m + i)

                        fint[col+0] += weight*(intx*inty/4)*( -Mxy*fAu*gAu_eta/(b*r) + 2*Nxx*fAu_xi*gAu/a + 2*Nxy*fAu*gAu_eta/b )
                        fint[col+1] += weight*(intx*inty/4)*( 3*Mxy*fAv_xi*gAv/(a*r) + 2*Myy*fAv*gAv_eta/(b*r) + Nxy*gAv*(a*fAv*phix + 2*fAv_xi*r)/(a*r) + Nyy*fAv*(b*gAv*phiy + 2*gAv_eta*r)/(b*r) )
                        fint[col+2] += weight*(intx*inty/4)*( -4*Mxx*fAw_xixi*gAw/(a*a) - 8*Mxy*fAw_xi*gAw_eta/(a*b) - 4*Myy*fAw*gAw_etaeta/(b*b) - 2*Nxx*fAw_xi*gAw*phix/a - 2*Nxy*(a*fAw*gAw_eta*phix + b*fAw_xi*gAw*phiy)/(a*b) - Nyy*fAw*(-b*gAw + 2*gAw_eta*phiy*r)/(b*r) )

    return fint


# BEGIN follower pressure, generated by theory/shells/follower_pressure/write_pyx.py


def calc_fext_follower(double [::1] cs, double [:, ::1] pnxny,
        double px1, double px2, double py1, double py2, object shell,
        int size, int col0, int quadratic=0, int reference=1):
    r"""Force vector of a follower pressure in the configuration ``cs``

    Integrates ``F_p = int p (Nu ax + Nv ay + Nw az) dA`` over the patch
    ``px1 <= x <= px2``, ``py1 <= y <= py2``, with ``nx x ny`` Gauss-Legendre
    points, ``(nx, ny) = pnxny.shape``, being ``pnxny[ptx, pty]`` the pressure,
    already multiplied by the load factor, at each point. The area vector
    ``(ax, ay, az)`` of the deformed mid-surface is, see
    ``theory/shells/follower_pressure``::

        ax = qf*v_x*w_y - qf*v_y*w_x - qf*v_x*vv/r - qf*w_x*ww/r - w_x
        ay = -qf*u_x*w_y + qf*u_y*w_x + qf*u_x*vv/r - w_y + vv/r
        az = qf*u_x*v_y - qf*u_y*v_x + qf*u_x*ww/r + ref + u_x + v_y + ww/r

    with ``qf = 1`` for ``quadratic=1`` and ``qf = 0`` otherwise, and ``ref =
    1`` for ``reference=1``, which includes the dead part ``F_p(0)``, and
    ``ref = 0`` otherwise.

    """
    cdef double [:, ::1] gtab, ftab
    cdef double a, b, pintx, pinty, qf
    cdef double r
    cdef int m, n, nx, ny
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, col, row, ptx, pty
    cdef double xi, eta, weight
    cdef double u_x, u_y, v_x, v_y, vv, ww, w_x, w_y
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double ref, ax, ay, az
    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw, fAw_xi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw, gAw_eta
    cdef double [::1] fext

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    nx = pnxny.shape[0]
    ny = pnxny.shape[1]
    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)
    pintx = px2 - px1
    pinty = py2 - py1
    qf = 1. if quadratic else 0.
    ref = 1. if reference else 0.

    fext = np.zeros(size, dtype=DOUBLE)

    gtab = np.zeros((6, n), dtype=DOUBLE)
    ftab = np.zeros((6, m), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                # Gauss point of the patch, in the natural coordinates of
                # the approximation functions, which span 0 <= x <= a and
                # 0 <= y <= b
                xi = 2*(px1 + (xis[ptx] + 1)/2*pintx)/a - 1
                eta = 2*(py1 + (etas[pty] + 1)/2*pinty)/b - 1
                weight = (weights_xi[ptx]*weights_eta[pty]*pintx*pinty/4
                          *pnxny[ptx, pty])

                # displacement gradients at the Gauss point
                u_x = u_y = v_x = v_y = vv = ww = w_x = w_y = 0.
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_eta = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_eta = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_eta = gtab[5, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_xi = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        col = col0 + DOF*(j*m + i)
                        u_x += cs[col+0]*(2/a)*fAu_xi*gAu
                        u_y += cs[col+0]*(2/b)*fAu*gAu_eta
                        v_x += cs[col+1]*(2/a)*fAv_xi*gAv
                        v_y += cs[col+1]*(2/b)*fAv*gAv_eta
                        vv += cs[col+1]*fAv*gAv
                        ww += cs[col+2]*fAw*gAw
                        w_x += cs[col+2]*(2/a)*fAw_xi*gAw
                        w_y += cs[col+2]*(2/b)*fAw*gAw_eta

                ax = qf*v_x*w_y - qf*v_y*w_x - qf*v_x*vv/r - qf*w_x*ww/r - w_x
                ay = -qf*u_x*w_y + qf*u_y*w_x + qf*u_x*vv/r - w_y + vv/r
                az = qf*u_x*v_y - qf*u_y*v_x + qf*u_x*ww/r + ref + u_x + v_y + ww/r

                for j in range(n):
                    gAu = gtab[0, j]
                    gAv = gtab[2, j]
                    gAw = gtab[4, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAv = ftab[2, i]
                        fAw = ftab[4, i]
                        col = col0 + DOF*(j*m + i)
                        fext[col+0] += weight*fAu*gAu*ax
                        fext[col+1] += weight*fAv*gAv*ay
                        fext[col+2] += weight*fAw*gAw*az

    return fext


def fkCfollower_num(double [::1] cs, double [:, ::1] pnxny,
        double px1, double px2, double py1, double py2, object shell,
        int size, int row0, int col0, int quadratic=0):
    r"""Load stiffness ``-dF_p/dc`` of a follower pressure

    See :func:`.calc_fext_follower` for the parameters. The matrix is
    unsymmetric in general, so it is assembled in full, not as an upper
    triangle, and must not be symmetrized. With ``quadratic=0`` it does not
    depend on ``cs``.

    """
    cdef double [:, ::1] gtab, ftab
    cdef double a, b, pintx, pinty, qf
    cdef double r
    cdef int m, n, nx, ny
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, col, row, ptx, pty
    cdef double xi, eta, weight
    cdef double u_x, u_y, v_x, v_y, vv, ww, w_x, w_y
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef long [::1] kCr, kCc
    cdef double [::1] kCv
    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw, fAw_xi, fBu, fBu_xi, fBv, fBv_xi, fBw, fBw_xi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw, gAw_eta, gBu, gBu_eta, gBv, gBv_eta, gBw, gBw_eta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    nx = pnxny.shape[0]
    ny = pnxny.shape[1]
    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)
    pintx = px2 - px1
    pinty = py2 - py1
    qf = 1. if quadratic else 0.

    fdim = 8*m*m*n*n
    kCr = np.zeros((fdim,), dtype=INT)
    kCc = np.zeros((fdim,), dtype=INT)
    kCv = np.zeros((fdim,), dtype=DOUBLE)

    gtab = np.zeros((6, n), dtype=DOUBLE)
    ftab = np.zeros((6, m), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                # Gauss point of the patch, in the natural coordinates of
                # the approximation functions, which span 0 <= x <= a and
                # 0 <= y <= b
                xi = 2*(px1 + (xis[ptx] + 1)/2*pintx)/a - 1
                eta = 2*(py1 + (etas[pty] + 1)/2*pinty)/b - 1
                weight = (weights_xi[ptx]*weights_eta[pty]*pintx*pinty/4
                          *pnxny[ptx, pty])

                # displacement gradients at the Gauss point, only used by the
                # quadratic area vector
                u_x = u_y = v_x = v_y = vv = ww = w_x = w_y = 0.
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)

                if quadratic == 1:
                    for j in range(n):
                        gAu = gtab[0, j]
                        gAu_eta = gtab[1, j]
                        gAv = gtab[2, j]
                        gAv_eta = gtab[3, j]
                        gAw = gtab[4, j]
                        gAw_eta = gtab[5, j]
                        for i in range(m):
                            fAu = ftab[0, i]
                            fAu_xi = ftab[1, i]
                            fAv = ftab[2, i]
                            fAv_xi = ftab[3, i]
                            fAw = ftab[4, i]
                            fAw_xi = ftab[5, i]
                            col = col0 + DOF*(j*m + i)
                            u_x += cs[col+0]*(2/a)*fAu_xi*gAu
                            u_y += cs[col+0]*(2/b)*fAu*gAu_eta
                            v_x += cs[col+1]*(2/a)*fAv_xi*gAv
                            v_y += cs[col+1]*(2/b)*fAv*gAv_eta
                            vv += cs[col+1]*fAv*gAv
                            ww += cs[col+2]*fAw*gAw
                            w_x += cs[col+2]*(2/a)*fAw_xi*gAw
                            w_y += cs[col+2]*(2/b)*fAw*gAw_eta

                c = -1
                for i in range(m):
                    fAu = ftab[0, i]
                    fAu_xi = ftab[1, i]
                    fAv = ftab[2, i]
                    fAv_xi = ftab[3, i]
                    fAw = ftab[4, i]
                    fAw_xi = ftab[5, i]
                    for k in range(m):
                        fBu = ftab[0, k]
                        fBu_xi = ftab[1, k]
                        fBv = ftab[2, k]
                        fBv_xi = ftab[3, k]
                        fBw = ftab[4, k]
                        fBw_xi = ftab[5, k]
                        for j in range(n):
                            gAu = gtab[0, j]
                            gAu_eta = gtab[1, j]
                            gAv = gtab[2, j]
                            gAv_eta = gtab[3, j]
                            gAw = gtab[4, j]
                            gAw_eta = gtab[5, j]
                            for l in range(n):
                                gBu = gtab[0, l]
                                gBu_eta = gtab[1, l]
                                gBv = gtab[2, l]
                                gBv_eta = gtab[3, l]
                                gBw = gtab[4, l]
                                gBw_eta = gtab[5, l]
                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*( fAu*fBv*gAu*gBv*qf*v_x/r + 2*fAu*fBv*gAu*gBv_eta*qf*w_x/b - 2*fAu*fBv_xi*gAu*gBv*qf*w_y/a + 2*fAu*fBv_xi*gAu*gBv*qf*vv/(a*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*( fAu*fBw*gAu*gBw*qf*w_x/r - 2*fAu*fBw*gAu*gBw_eta*qf*v_x/b + 2*fAu*fBw_xi*gAu*gBw*qf*v_y/a + 2*fAu*fBw_xi*gAu*gBw*qf*ww/(a*r) + 2*fAu*fBw_xi*gAu*gBw/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*( -2*fAv*fBu*gAv*gBu_eta*qf*w_x/b + 2*fAv*fBu_xi*gAv*gBu*qf*w_y/a - 2*fAv*fBu_xi*gAv*gBu*qf*vv/(a*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*( -fAv*fBv*gAv*gBv*qf*u_x/r - fAv*fBv*gAv*gBv/r )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*( 2*fAv*fBw*gAv*gBw_eta*qf*u_x/b + 2*fAv*fBw*gAv*gBw_eta/b - 2*fAv*fBw_xi*gAv*gBw*qf*u_y/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*( 2*fAw*fBu*gAw*gBu_eta*qf*v_x/b - 2*fAw*fBu_xi*gAw*gBu*qf*v_y/a - 2*fAw*fBu_xi*gAw*gBu*qf*ww/(a*r) - 2*fAw*fBu_xi*gAw*gBu/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*( -2*fAw*fBv*gAw*gBv_eta*qf*u_x/b - 2*fAw*fBv*gAw*gBv_eta/b + 2*fAw*fBv_xi*gAw*gBv*qf*u_y/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*( -fAw*fBw*gAw*gBw*qf*u_x/r - fAw*fBw*gAw*gBw/r )

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC
# END follower pressure


# BEGIN tangent stiffness product, generated by theory/shells/tangent_product/write_pyx.py


def calc_kTv(double [::1] cs, double [::1] vs, object Finput, object shell,
        int size, int col0, int nx, int ny, double Nxx_cte=0.,
        double Nyy_cte=0., double Nxy_cte=0.):
    r"""Product of the tangent stiffness matrix by a vector, without assembly

    Returns `\boldsymbol{K}_T(\boldsymbol{c}) \boldsymbol{v}`, with `\boldsymbol{K}_T` the
    tangent stiffness matrix ``fkC_num(cs, ..., NLgeom=1) + fkG_num(cs, ...)``
    at ``cs``, i.e. the exact Jacobian of :func:`calc_fint`, plus the geometric
    stiffness matrix of the constant membrane stress state ``Nxx_cte``,
    ``Nyy_cte`` and ``Nxy_cte``, multiplied by ``vs``. At each integration
    point, see ``theory/shells/tangent_product``::

        beta = [G] cs,   eps = ([B_L] + [B_NL]/2) cs,   deps = ([B_L] + [B_NL]) vs
        kTv += ([B_L] + [B_NL])^T [F] deps + [G]^T [N] [G] vs

    with ``[N]`` the membrane stresses of ``[F] eps``, plus the constant
    ones. The cost is about that of :func:`calc_fint`.

    Parameters
    ----------
    cs, vs : array-like
        Ritz constants of the configuration and the vector, of size ``size``.
    Finput : array-like
        Constitutive matrix ``(6, 6)``, or one for each integration
        point ``(nx, ny, 6, 6)``.
    shell : :class:`.Shell`
        The shell.
    size, col0 : int
        Size of the vectors and offset of the shell in them.
    nx, ny : int
        Number of integration points.
    Nxx_cte, Nyy_cte, Nxy_cte : float, optional
        Constant membrane stress state.

    Returns
    -------
    kTv : np.ndarray
        The product, of size ``size``.

    """
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty, r
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double bx, by, dbx, dby, Nxx, Nyy, Nxy
    cdef double e[6]
    cdef double de[6]
    cdef double s[6]
    cdef double ds[6]
    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw, fAw_xi
    cdef double fAw_xixi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw, gAw_eta
    cdef double gAw_etaeta

    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta, kTv

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one
    # constitutive matrix [6, 6] for each of the nx * ny integration points
    cdef double F[36]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    if cs.shape[0] != size or vs.shape[0] != size:
        raise ValueError('cs and vs must have the size %d' % size)
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
    r = shell.r
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

    kTv = np.zeros(size, dtype=DOUBLE)
    # basis functions at the integration point
    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)

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

                # the integrands are per unit area of the natural domain
                weight = weights_xi[ptx]*weights_eta[pty]*intx*inty/4

                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)

                # rotations of cs and vs
                bx = 0.
                by = 0.
                dbx = 0.
                dby = 0.
                for j in range(n):
                    gAv = gtab[2, j]
                    gAw = gtab[4, j]
                    gAw_eta = gtab[5, j]
                    for i in range(m):
                        fAv = ftab[2, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        col = col0 + DOF*(j*m + i)
                        bx += -2*fAw_xi*gAw/a*cs[col+2]
                        by += fAv*gAv/r*cs[col+1] - 2*fAw*gAw_eta/b*cs[col+2]
                        dbx += -2*fAw_xi*gAw/a*vs[col+2]
                        dby += fAv*gAv/r*vs[col+1] - 2*fAw*gAw_eta/b*vs[col+2]

                # strains of cs, eps = ([B_L] + [B_NL]/2) cs, and strain
                # increment, deps = ([B_L] + [B_NL]) vs
                for i in range(6):
                    e[i] = 0.
                    de[i] = 0.
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_eta = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_eta = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_eta = gtab[5, j]
                    gAw_etaeta = gtab[6, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_xi = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        fAw_xixi = ftab[6, i]
                        col = col0 + DOF*(j*m + i)
                        e[0] += 2*fAu_xi*gAu/a*cs[col+0] - bx*fAw_xi*gAw/a*cs[col+2]
                        e[1] += (0.5*by*fAv*gAv/r + 2*fAv*gAv_eta/b)*cs[col+1] + (fAw*gAw/r - by*fAw*gAw_eta/b)*cs[col+2]
                        e[2] += 2*fAu*gAu_eta/b*cs[col+0] + (0.5*bx*fAv*gAv/r + 2*fAv_xi*gAv/a)*cs[col+1] + (-bx*fAw*gAw_eta/b - by*fAw_xi*gAw/a)*cs[col+2]
                        e[3] += -4*fAw_xixi*gAw/(a*a)*cs[col+2]
                        e[4] += 2*fAv*gAv_eta/(b*r)*cs[col+1] - 4*fAw*gAw_etaeta/(b*b)*cs[col+2]
                        e[5] += -fAu*gAu_eta/(b*r)*cs[col+0] + 3*fAv_xi*gAv/(a*r)*cs[col+1] - 8*fAw_xi*gAw_eta/(a*b)*cs[col+2]
                        de[0] += 2*fAu_xi*gAu/a*vs[col+0] - 2*bx*fAw_xi*gAw/a*vs[col+2]
                        de[1] += (by*fAv*gAv/r + 2*fAv*gAv_eta/b)*vs[col+1] + (fAw*gAw/r - 2*by*fAw*gAw_eta/b)*vs[col+2]
                        de[2] += 2*fAu*gAu_eta/b*vs[col+0] + (bx*fAv*gAv/r + 2*fAv_xi*gAv/a)*vs[col+1] + (-2*bx*fAw*gAw_eta/b - 2*by*fAw_xi*gAw/a)*vs[col+2]
                        de[3] += -4*fAw_xixi*gAw/(a*a)*vs[col+2]
                        de[4] += 2*fAv*gAv_eta/(b*r)*vs[col+1] - 4*fAw*gAw_etaeta/(b*b)*vs[col+2]
                        de[5] += -fAu*gAu_eta/(b*r)*vs[col+0] + 3*fAv_xi*gAv/(a*r)*vs[col+1] - 8*fAw_xi*gAw_eta/(a*b)*vs[col+2]

                # stresses
                for i in range(6):
                    s[i] = 0.
                    ds[i] = 0.
                    for j in range(6):
                        s[i] += F[i*6 + j]*e[j]
                        ds[i] += F[i*6 + j]*de[j]
                Nxx = s[0] + Nxx_cte
                Nyy = s[1] + Nyy_cte
                Nxy = s[2] + Nxy_cte

                # ([B_L] + [B_NL])^T ds + [G]^T [N] [G] vs
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_eta = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_eta = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_eta = gtab[5, j]
                    gAw_etaeta = gtab[6, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_xi = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        fAw_xixi = ftab[6, i]
                        col = col0 + DOF*(j*m + i)
                        kTv[col+0] += weight*(2*ds[2]*fAu*gAu_eta/b - ds[5]*fAu*gAu_eta/(b*r) + 2*ds[0]*fAu_xi*gAu/a)
                        kTv[col+1] += weight*(Nxy*dbx*fAv*gAv/r + Nyy*dby*fAv*gAv/r + ds[1]*(by*fAv*gAv/r + 2*fAv*gAv_eta/b) + ds[2]*(bx*fAv*gAv/r + 2*fAv_xi*gAv/a) + 2*ds[4]*fAv*gAv_eta/(b*r) + 3*ds[5]*fAv_xi*gAv/(a*r))
                        kTv[col+2] += weight*(-2*Nxx*dbx*fAw_xi*gAw/a + Nxy*(-2*dbx*fAw*gAw_eta/b - 2*dby*fAw_xi*gAw/a) - 2*Nyy*dby*fAw*gAw_eta/b + ds[1]*(fAw*gAw/r - 2*by*fAw*gAw_eta/b) + ds[2]*(-2*bx*fAw*gAw_eta/b - 2*by*fAw_xi*gAw/a) - 4*ds[4]*fAw*gAw_etaeta/(b*b) - 2*bx*ds[0]*fAw_xi*gAw/a - 8*ds[5]*fAw_xi*gAw_eta/(a*b) - 4*ds[3]*fAw_xixi*gAw/(a*a))

    return np.asarray(kTv)

# END tangent stiffness product
