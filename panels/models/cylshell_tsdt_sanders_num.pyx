#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Cylindrical shells using the Reddy's third-order shear deformation theory
(TSDT) with the Sanders-Koiter kinematics

Numerically integrated matrices. See the kinematic equations in
``theory/shells/fsdt_tsdt/fsdt_tsdt.py``, from where the integrands herein
have been generated. The degrees of freedom of each term of the
approximation are ``u, v, w, phix, phiy``, and the constitutive matrix
``Finput`` must be ``13 x 13``, with the rows and columns in the order of
the generalized strains::

    exx, eyy, gxy, kxx, kyy, kxy, kxx3, kyy3, kxy3, gyz, gxz, gyz2, gxz2

which is the order of the generalized stresses::

    Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Pxx, Pyy, Pxy, Qy, Qx, Ry, Rx

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 5
cdef int NE = 13

def fkC_num(double [::1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double [:, ::1] gtab, ftab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double r
    cdef double a, b, intx, inty, h, c1
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double x1phix, x1phixr, x2phix, x2phixr
    cdef double x1phiy, x1phiyr, x2phiy, x2phiyr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double y1phix, y1phixr, y2phix, y2phixr
    cdef double y1phiy, y1phiyr, y2phiy, y2phiyr

    cdef int i, j, k, l, c, row, col, ptx, pty
    cdef double A11, A12, A16, B11, B12
    cdef double B16, E11, E12, E16, A22
    cdef double A26, B22, B26, E22, E26
    cdef double A66, B66, E66, D11, D12
    cdef double D16, F11, F12, F16, D22
    cdef double D26, F22, F26, D66, F66
    cdef double H11, H12, H16, H22, H26
    cdef double H66, A44, A45, D44, D45
    cdef double A55, D55, F44, F45, F55

    cdef long [::1] kCr, kCc
    cdef double [::1] kCv

    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw
    cdef double fAw_xi, fAw_xixi, fAphix, fAphix_xi, fAphiy
    cdef double fAphiy_xi
    cdef double fBu, fBu_xi, fBv, fBv_xi, fBw
    cdef double fBw_xi, fBw_xixi, fBphix, fBphix_xi, fBphiy
    cdef double fBphiy_xi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw
    cdef double gAw_eta, gAw_etaeta, gAphix, gAphix_eta, gAphiy
    cdef double gAphiy_eta
    cdef double gBu, gBu_eta, gBv, gBv_eta, gBw
    cdef double gBw_eta, gBw_etaeta, gBphix, gBphix_eta, gBphiy
    cdef double gBphiy_eta
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double bx, by, NxxNL, NyyNL, NxyNL

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # F as 4-D matrix, must be [nx, ny, 13, 13], when there is one
    # constitutive matrix [13, 13] for each of the nx * ny integration points
    cdef double F[169]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, NE, NE):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (NE, NE):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        for i in range(NE):
            for j in range(NE):
                F[i*NE + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    #NOTE the traction-free faces are assumed at z = +-h/2
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    x1phix = shell.x1phix; x1phixr = shell.x1phixr; x2phix = shell.x2phix; x2phixr = shell.x2phixr
    x1phiy = shell.x1phiy; x1phiyr = shell.x1phiyr; x2phiy = shell.x2phiy; x2phiyr = shell.x2phiyr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    y1phix = shell.y1phix; y1phixr = shell.y1phixr; y2phix = shell.y2phix; y2phixr = shell.y2phixr
    y1phiy = shell.y1phiy; y1phiyr = shell.y1phiyr; y2phiy = shell.y2phiy; y2phiyr = shell.y2phiyr

    fdim = 25*m*m*n*n

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

    gtab = np.zeros((11, n), dtype=DOUBLE)
    ftab = np.zeros((11, m), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

                bx = 0
                by = 0
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[1, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[4, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[5, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[9, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[10, j] = fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[1, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[4, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[5, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[9, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                if NLgeom == 1:
                    for j in range(n):
                        gAw = gtab[0, j]
                        gAw_eta = gtab[1, j]
                        gAv = gtab[2, j]
                        for i in range(m):
                            fAw = ftab[0, i]
                            fAw_xi = ftab[1, i]
                            fAv = ftab[2, i]

                            col = col0 + DOF*(j*m + i)

                            bx -= (2/a)*cs[col+2]*fAw_xi*gAw
                            by -= (2/b)*cs[col+2]*fAw*gAw_eta
                            by += cs[col+1]*fAv*gAv/r

                if one_F_each_point == 1:
                    for i in range(NE):
                        for j in range(NE):
                            #TODO could assume symmetry
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

                A11 = F[0*NE + 0]
                A12 = F[0*NE + 1]
                A16 = F[0*NE + 2]
                B11 = F[0*NE + 3]
                B12 = F[0*NE + 4]
                B16 = F[0*NE + 5]
                E11 = F[0*NE + 6]
                E12 = F[0*NE + 7]
                E16 = F[0*NE + 8]
                A22 = F[1*NE + 1]
                A26 = F[1*NE + 2]
                B22 = F[1*NE + 4]
                B26 = F[1*NE + 5]
                E22 = F[1*NE + 7]
                E26 = F[1*NE + 8]
                A66 = F[2*NE + 2]
                B66 = F[2*NE + 5]
                E66 = F[2*NE + 8]
                D11 = F[3*NE + 3]
                D12 = F[3*NE + 4]
                D16 = F[3*NE + 5]
                F11 = F[3*NE + 6]
                F12 = F[3*NE + 7]
                F16 = F[3*NE + 8]
                D22 = F[4*NE + 4]
                D26 = F[4*NE + 5]
                F22 = F[4*NE + 7]
                F26 = F[4*NE + 8]
                D66 = F[5*NE + 5]
                F66 = F[5*NE + 8]
                H11 = F[6*NE + 6]
                H12 = F[6*NE + 7]
                H16 = F[6*NE + 8]
                H22 = F[7*NE + 7]
                H26 = F[7*NE + 8]
                H66 = F[8*NE + 8]
                A44 = F[9*NE + 9]
                A45 = F[9*NE + 10]
                D44 = F[9*NE + 11]
                D45 = F[9*NE + 12]
                A55 = F[10*NE + 10]
                D55 = F[10*NE + 12]
                F44 = F[11*NE + 11]
                F45 = F[11*NE + 12]
                F55 = F[12*NE + 12]

                # Membrane stress carried by the nonlinear strain
                # eps_NL = {bx^2/2, by^2/2, bx*by}. With it, KGNL = KG(N_NL)
                # is collected in kC such that
                #     KT = KC0 + KCNL + KGNL (fkC_num) + KG(Nbar + N_L) (fkG_num)
                # is the exact Jacobian of calc_fint, and fkG_num stays
                # homogeneous of degree one in cs, as linear buckling requires.
                # bx = by = 0 when NLgeom == 0, then KGNL vanishes
                NxxNL = A11*0.5*bx*bx + A12*0.5*by*by + A16*bx*by
                NyyNL = A12*0.5*bx*bx + A22*0.5*by*by + A26*bx*by
                NxyNL = A16*0.5*bx*bx + A26*0.5*by*by + A66*bx*by

                # kC
                c = -1
                for i in range(m):
                    fAu = ftab[3, i]
                    fAu_xi = ftab[4, i]
                    fAv = ftab[2, i]
                    fAv_xi = ftab[5, i]
                    fAw = ftab[0, i]
                    fAw_xi = ftab[1, i]
                    fAw_xixi = ftab[6, i]
                    fAphix = ftab[7, i]
                    fAphix_xi = ftab[8, i]
                    fAphiy = ftab[9, i]
                    fAphiy_xi = ftab[10, i]

                    for k in range(m):
                        fBu = ftab[3, k]
                        fBu_xi = ftab[4, k]
                        fBv = ftab[2, k]
                        fBv_xi = ftab[5, k]
                        fBw = ftab[0, k]
                        fBw_xi = ftab[1, k]
                        fBw_xixi = ftab[6, k]
                        fBphix = ftab[7, k]
                        fBphix_xi = ftab[8, k]
                        fBphiy = ftab[9, k]
                        fBphiy_xi = ftab[10, k]

                        for j in range(n):
                            gAu = gtab[3, j]
                            gAu_eta = gtab[4, j]
                            gAv = gtab[2, j]
                            gAv_eta = gtab[5, j]
                            gAw = gtab[0, j]
                            gAw_eta = gtab[1, j]
                            gAw_etaeta = gtab[6, j]
                            gAphix = gtab[7, j]
                            gAphix_eta = gtab[8, j]
                            gAphiy = gtab[9, j]
                            gAphiy_eta = gtab[10, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = gtab[3, l]
                                gBu_eta = gtab[4, l]
                                gBv = gtab[2, l]
                                gBv_eta = gtab[5, l]
                                gBw = gtab[0, l]
                                gBw_eta = gtab[1, l]
                                gBw_etaeta = gtab[6, l]
                                gBphix = gtab[7, l]
                                gBphix_eta = gtab[8, l]
                                gBphiy = gtab[9, l]
                                gBphiy_eta = gtab[10, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAu_xi*fBu_xi*gAu*gBu/(a*a) + A16*(4*fAu*fBu_xi*gAu_eta*gBu/(a*b) + 4*fAu_xi*fBu*gAu*gBu_eta/(a*b)) + 4*A66*fAu*fBu*gAu_eta*gBu_eta/(b*b) + B16*(-2*fAu*fBu_xi*gAu_eta*gBu/(a*b*r) - 2*fAu_xi*fBu*gAu*gBu_eta/(a*b*r)) - 4*B66*fAu*fBu*gAu_eta*gBu_eta/((b*b)*r) + D66*fAu*fBu*gAu_eta*gBu_eta/((b*b)*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( A12*(2*by*fAu_xi*fBv*gAu*gBv/(a*r) + 4*fAu_xi*fBv*gAu*gBv_eta/(a*b)) + A16*(2*bx*fAu_xi*fBv*gAu*gBv/(a*r) + 4*fAu_xi*fBv_xi*gAu*gBv/(a*a)) + A26*(2*by*fAu*fBv*gAu_eta*gBv/(b*r) + 4*fAu*fBv*gAu_eta*gBv_eta/(b*b)) + A66*(2*bx*fAu*fBv*gAu_eta*gBv/(b*r) + 4*fAu*fBv_xi*gAu_eta*gBv/(a*b)) + 4*B12*fAu_xi*fBv*gAu*gBv_eta/(a*b*r) + 6*B16*fAu_xi*fBv_xi*gAu*gBv/((a*a)*r) + B26*(-by*fAu*fBv*gAu_eta*gBv/(b*(r*r)) + 2*fAu*fBv*gAu_eta*gBv_eta/((b*b)*r)) + B66*(-bx*fAu*fBv*gAu_eta*gBv/(b*(r*r)) + 4*fAu*fBv_xi*gAu_eta*gBv/(a*b*r)) - 2*D26*fAu*fBv*gAu_eta*gBv_eta/((b*b)*(r*r)) - 3*D66*fAu*fBv_xi*gAu_eta*gBv/(a*b*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( -4*A11*bx*fAu_xi*fBw_xi*gAu*gBw/(a*a) + A12*(2*fAu_xi*fBw*gAu*gBw/(a*r) - 4*by*fAu_xi*fBw*gAu*gBw_eta/(a*b)) + A16*(-4*bx*fAu*fBw_xi*gAu_eta*gBw/(a*b) - 4*bx*fAu_xi*fBw*gAu*gBw_eta/(a*b) - 4*by*fAu_xi*fBw_xi*gAu*gBw/(a*a)) + A26*(2*fAu*fBw*gAu_eta*gBw/(b*r) - 4*by*fAu*fBw*gAu_eta*gBw_eta/(b*b)) + A66*(-4*bx*fAu*fBw*gAu_eta*gBw_eta/(b*b) - 4*by*fAu*fBw_xi*gAu_eta*gBw/(a*b)) + 2*B16*bx*fAu*fBw_xi*gAu_eta*gBw/(a*b*r) + B26*(-fAu*fBw*gAu_eta*gBw/(b*(r*r)) + 2*by*fAu*fBw*gAu_eta*gBw_eta/((b*b)*r)) + B66*(2*bx*fAu*fBw*gAu_eta*gBw_eta/((b*b)*r) + 2*by*fAu*fBw_xi*gAu_eta*gBw/(a*b*r)) - 8*E11*c1*fAu_xi*fBw_xixi*gAu*gBw/(a*a*a) - 8*E12*c1*fAu_xi*fBw*gAu*gBw_etaeta/(a*(b*b)) + E16*(-8*c1*fAu*fBw_xixi*gAu_eta*gBw/((a*a)*b) - 16*c1*fAu_xi*fBw_xi*gAu*gBw_eta/((a*a)*b)) - 8*E26*c1*fAu*fBw*gAu_eta*gBw_etaeta/(b*b*b) - 16*E66*c1*fAu*fBw_xi*gAu_eta*gBw_eta/(a*(b*b)) + 4*F16*c1*fAu*fBw_xixi*gAu_eta*gBw/((a*a)*b*r) + 4*F26*c1*fAu*fBw*gAu_eta*gBw_etaeta/((b*b*b)*r) + 8*F66*c1*fAu*fBw_xi*gAu_eta*gBw_eta/(a*(b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 4*B11*fAu_xi*fBphix_xi*gAu*gBphix/(a*a) + B16*(4*fAu*fBphix_xi*gAu_eta*gBphix/(a*b) + 4*fAu_xi*fBphix*gAu*gBphix_eta/(a*b)) + 4*B66*fAu*fBphix*gAu_eta*gBphix_eta/(b*b) - 2*D16*fAu*fBphix_xi*gAu_eta*gBphix/(a*b*r) - 2*D66*fAu*fBphix*gAu_eta*gBphix_eta/((b*b)*r) - 4*E11*c1*fAu_xi*fBphix_xi*gAu*gBphix/(a*a) + E16*(-4*c1*fAu*fBphix_xi*gAu_eta*gBphix/(a*b) - 4*c1*fAu_xi*fBphix*gAu*gBphix_eta/(a*b)) - 4*E66*c1*fAu*fBphix*gAu_eta*gBphix_eta/(b*b) + 2*F16*c1*fAu*fBphix_xi*gAu_eta*gBphix/(a*b*r) + 2*F66*c1*fAu*fBphix*gAu_eta*gBphix_eta/((b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAu_xi*fBphiy*gAu*gBphiy_eta/(a*b) + 4*B16*fAu_xi*fBphiy_xi*gAu*gBphiy/(a*a) + 4*B26*fAu*fBphiy*gAu_eta*gBphiy_eta/(b*b) + 4*B66*fAu*fBphiy_xi*gAu_eta*gBphiy/(a*b) - 2*D26*fAu*fBphiy*gAu_eta*gBphiy_eta/((b*b)*r) - 2*D66*fAu*fBphiy_xi*gAu_eta*gBphiy/(a*b*r) - 4*E12*c1*fAu_xi*fBphiy*gAu*gBphiy_eta/(a*b) - 4*E16*c1*fAu_xi*fBphiy_xi*gAu*gBphiy/(a*a) - 4*E26*c1*fAu*fBphiy*gAu_eta*gBphiy_eta/(b*b) - 4*E66*c1*fAu*fBphiy_xi*gAu_eta*gBphiy/(a*b) + 2*F26*c1*fAu*fBphiy*gAu_eta*gBphiy_eta/((b*b)*r) + 2*F66*c1*fAu*fBphiy_xi*gAu_eta*gBphiy/(a*b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( A12*(2*by*fAv*fBu_xi*gAv*gBu/(a*r) + 4*fAv*fBu_xi*gAv_eta*gBu/(a*b)) + A16*(2*bx*fAv*fBu_xi*gAv*gBu/(a*r) + 4*fAv_xi*fBu_xi*gAv*gBu/(a*a)) + A26*(2*by*fAv*fBu*gAv*gBu_eta/(b*r) + 4*fAv*fBu*gAv_eta*gBu_eta/(b*b)) + A66*(2*bx*fAv*fBu*gAv*gBu_eta/(b*r) + 4*fAv_xi*fBu*gAv*gBu_eta/(a*b)) + 4*B12*fAv*fBu_xi*gAv_eta*gBu/(a*b*r) + 6*B16*fAv_xi*fBu_xi*gAv*gBu/((a*a)*r) + B26*(-by*fAv*fBu*gAv*gBu_eta/(b*(r*r)) + 2*fAv*fBu*gAv_eta*gBu_eta/((b*b)*r)) + B66*(-bx*fAv*fBu*gAv*gBu_eta/(b*(r*r)) + 4*fAv_xi*fBu*gAv*gBu_eta/(a*b*r)) - 2*D26*fAv*fBu*gAv_eta*gBu_eta/((b*b)*(r*r)) - 3*D66*fAv_xi*fBu*gAv*gBu_eta/(a*b*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( A22*((by*by)*fAv*fBv*gAv*gBv/(r*r) + 2*by*fAv*fBv*gAv*gBv_eta/(b*r) + 2*by*fAv*fBv*gAv_eta*gBv/(b*r) + 4*fAv*fBv*gAv_eta*gBv_eta/(b*b)) + A26*(2*bx*by*fAv*fBv*gAv*gBv/(r*r) + 2*bx*fAv*fBv*gAv*gBv_eta/(b*r) + 2*bx*fAv*fBv*gAv_eta*gBv/(b*r) + 2*by*fAv*fBv_xi*gAv*gBv/(a*r) + 2*by*fAv_xi*fBv*gAv*gBv/(a*r) + 4*fAv*fBv_xi*gAv_eta*gBv/(a*b) + 4*fAv_xi*fBv*gAv*gBv_eta/(a*b)) + A66*((bx*bx)*fAv*fBv*gAv*gBv/(r*r) + 2*bx*fAv*fBv_xi*gAv*gBv/(a*r) + 2*bx*fAv_xi*fBv*gAv*gBv/(a*r) + 4*fAv_xi*fBv_xi*gAv*gBv/(a*a)) + B22*(2*by*fAv*fBv*gAv*gBv_eta/(b*(r*r)) + 2*by*fAv*fBv*gAv_eta*gBv/(b*(r*r)) + 8*fAv*fBv*gAv_eta*gBv_eta/((b*b)*r)) + B26*(2*bx*fAv*fBv*gAv*gBv_eta/(b*(r*r)) + 2*bx*fAv*fBv*gAv_eta*gBv/(b*(r*r)) + 3*by*fAv*fBv_xi*gAv*gBv/(a*(r*r)) + 3*by*fAv_xi*fBv*gAv*gBv/(a*(r*r)) + 10*fAv*fBv_xi*gAv_eta*gBv/(a*b*r) + 10*fAv_xi*fBv*gAv*gBv_eta/(a*b*r)) + B66*(3*bx*fAv*fBv_xi*gAv*gBv/(a*(r*r)) + 3*bx*fAv_xi*fBv*gAv*gBv/(a*(r*r)) + 12*fAv_xi*fBv_xi*gAv*gBv/((a*a)*r)) + 4*D22*fAv*fBv*gAv_eta*gBv_eta/((b*b)*(r*r)) + D26*(6*fAv*fBv_xi*gAv_eta*gBv/(a*b*(r*r)) + 6*fAv_xi*fBv*gAv*gBv_eta/(a*b*(r*r))) + 9*D66*fAv_xi*fBv_xi*gAv*gBv/((a*a)*(r*r)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( NyyNL*fAv*fBv*gAv*gBv/(r*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( A12*(-2*bx*by*fAv*fBw_xi*gAv*gBw/(a*r) - 4*bx*fAv*fBw_xi*gAv_eta*gBw/(a*b)) + A16*(-2*(bx*bx)*fAv*fBw_xi*gAv*gBw/(a*r) - 4*bx*fAv_xi*fBw_xi*gAv*gBw/(a*a)) + A22*(by*fAv*fBw*gAv*gBw/(r*r) - 2*(by*by)*fAv*fBw*gAv*gBw_eta/(b*r) + 2*fAv*fBw*gAv_eta*gBw/(b*r) - 4*by*fAv*fBw*gAv_eta*gBw_eta/(b*b)) + A26*(bx*fAv*fBw*gAv*gBw/(r*r) - 4*bx*by*fAv*fBw*gAv*gBw_eta/(b*r) - 4*bx*fAv*fBw*gAv_eta*gBw_eta/(b*b) - 2*(by*by)*fAv*fBw_xi*gAv*gBw/(a*r) + 2*fAv_xi*fBw*gAv*gBw/(a*r) - 4*by*fAv*fBw_xi*gAv_eta*gBw/(a*b) - 4*by*fAv_xi*fBw*gAv*gBw_eta/(a*b)) + A66*(-2*(bx*bx)*fAv*fBw*gAv*gBw_eta/(b*r) - 2*bx*by*fAv*fBw_xi*gAv*gBw/(a*r) - 4*bx*fAv_xi*fBw*gAv*gBw_eta/(a*b) - 4*by*fAv_xi*fBw_xi*gAv*gBw/(a*a)) - 4*B12*bx*fAv*fBw_xi*gAv_eta*gBw/(a*b*r) - 6*B16*bx*fAv_xi*fBw_xi*gAv*gBw/((a*a)*r) + B22*(2*fAv*fBw*gAv_eta*gBw/(b*(r*r)) - 4*by*fAv*fBw*gAv_eta*gBw_eta/((b*b)*r)) + B26*(-4*bx*fAv*fBw*gAv_eta*gBw_eta/((b*b)*r) + 3*fAv_xi*fBw*gAv*gBw/(a*(r*r)) - 4*by*fAv*fBw_xi*gAv_eta*gBw/(a*b*r) - 6*by*fAv_xi*fBw*gAv*gBw_eta/(a*b*r)) + B66*(-6*bx*fAv_xi*fBw*gAv*gBw_eta/(a*b*r) - 6*by*fAv_xi*fBw_xi*gAv*gBw/((a*a)*r)) + E12*(-4*by*c1*fAv*fBw_xixi*gAv*gBw/((a*a)*r) - 8*c1*fAv*fBw_xixi*gAv_eta*gBw/((a*a)*b)) + E16*(-4*bx*c1*fAv*fBw_xixi*gAv*gBw/((a*a)*r) - 8*c1*fAv_xi*fBw_xixi*gAv*gBw/(a*a*a)) + E22*(-4*by*c1*fAv*fBw*gAv*gBw_etaeta/((b*b)*r) - 8*c1*fAv*fBw*gAv_eta*gBw_etaeta/(b*b*b)) + E26*(-4*bx*c1*fAv*fBw*gAv*gBw_etaeta/((b*b)*r) - 8*by*c1*fAv*fBw_xi*gAv*gBw_eta/(a*b*r) - 16*c1*fAv*fBw_xi*gAv_eta*gBw_eta/(a*(b*b)) - 8*c1*fAv_xi*fBw*gAv*gBw_etaeta/(a*(b*b))) + E66*(-8*bx*c1*fAv*fBw_xi*gAv*gBw_eta/(a*b*r) - 16*c1*fAv_xi*fBw_xi*gAv*gBw_eta/((a*a)*b)) - 8*F12*c1*fAv*fBw_xixi*gAv_eta*gBw/((a*a)*b*r) - 12*F16*c1*fAv_xi*fBw_xixi*gAv*gBw/((a*a*a)*r) - 8*F22*c1*fAv*fBw*gAv_eta*gBw_etaeta/((b*b*b)*r) + F26*(-16*c1*fAv*fBw_xi*gAv_eta*gBw_eta/(a*(b*b)*r) - 12*c1*fAv_xi*fBw*gAv*gBw_etaeta/(a*(b*b)*r)) - 24*F66*c1*fAv_xi*fBw_xi*gAv*gBw_eta/((a*a)*b*r) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( -2*NxyNL*fAv*fBw_xi*gAv*gBw/(a*r) - 2*NyyNL*fAv*fBw*gAv*gBw_eta/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( B12*(2*by*fAv*fBphix_xi*gAv*gBphix/(a*r) + 4*fAv*fBphix_xi*gAv_eta*gBphix/(a*b)) + B16*(2*bx*fAv*fBphix_xi*gAv*gBphix/(a*r) + 4*fAv_xi*fBphix_xi*gAv*gBphix/(a*a)) + B26*(2*by*fAv*fBphix*gAv*gBphix_eta/(b*r) + 4*fAv*fBphix*gAv_eta*gBphix_eta/(b*b)) + B66*(2*bx*fAv*fBphix*gAv*gBphix_eta/(b*r) + 4*fAv_xi*fBphix*gAv*gBphix_eta/(a*b)) + 4*D12*fAv*fBphix_xi*gAv_eta*gBphix/(a*b*r) + 6*D16*fAv_xi*fBphix_xi*gAv*gBphix/((a*a)*r) + 4*D26*fAv*fBphix*gAv_eta*gBphix_eta/((b*b)*r) + 6*D66*fAv_xi*fBphix*gAv*gBphix_eta/(a*b*r) + E12*(-2*by*c1*fAv*fBphix_xi*gAv*gBphix/(a*r) - 4*c1*fAv*fBphix_xi*gAv_eta*gBphix/(a*b)) + E16*(-2*bx*c1*fAv*fBphix_xi*gAv*gBphix/(a*r) - 4*c1*fAv_xi*fBphix_xi*gAv*gBphix/(a*a)) + E26*(-2*by*c1*fAv*fBphix*gAv*gBphix_eta/(b*r) - 4*c1*fAv*fBphix*gAv_eta*gBphix_eta/(b*b)) + E66*(-2*bx*c1*fAv*fBphix*gAv*gBphix_eta/(b*r) - 4*c1*fAv_xi*fBphix*gAv*gBphix_eta/(a*b)) - 4*F12*c1*fAv*fBphix_xi*gAv_eta*gBphix/(a*b*r) - 6*F16*c1*fAv_xi*fBphix_xi*gAv*gBphix/((a*a)*r) - 4*F26*c1*fAv*fBphix*gAv_eta*gBphix_eta/((b*b)*r) - 6*F66*c1*fAv_xi*fBphix*gAv*gBphix_eta/(a*b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( B22*(2*by*fAv*fBphiy*gAv*gBphiy_eta/(b*r) + 4*fAv*fBphiy*gAv_eta*gBphiy_eta/(b*b)) + B26*(2*bx*fAv*fBphiy*gAv*gBphiy_eta/(b*r) + 2*by*fAv*fBphiy_xi*gAv*gBphiy/(a*r) + 4*fAv*fBphiy_xi*gAv_eta*gBphiy/(a*b) + 4*fAv_xi*fBphiy*gAv*gBphiy_eta/(a*b)) + B66*(2*bx*fAv*fBphiy_xi*gAv*gBphiy/(a*r) + 4*fAv_xi*fBphiy_xi*gAv*gBphiy/(a*a)) + 4*D22*fAv*fBphiy*gAv_eta*gBphiy_eta/((b*b)*r) + D26*(4*fAv*fBphiy_xi*gAv_eta*gBphiy/(a*b*r) + 6*fAv_xi*fBphiy*gAv*gBphiy_eta/(a*b*r)) + 6*D66*fAv_xi*fBphiy_xi*gAv*gBphiy/((a*a)*r) + E22*(-2*by*c1*fAv*fBphiy*gAv*gBphiy_eta/(b*r) - 4*c1*fAv*fBphiy*gAv_eta*gBphiy_eta/(b*b)) + E26*(-2*bx*c1*fAv*fBphiy*gAv*gBphiy_eta/(b*r) - 2*by*c1*fAv*fBphiy_xi*gAv*gBphiy/(a*r) - 4*c1*fAv*fBphiy_xi*gAv_eta*gBphiy/(a*b) - 4*c1*fAv_xi*fBphiy*gAv*gBphiy_eta/(a*b)) + E66*(-2*bx*c1*fAv*fBphiy_xi*gAv*gBphiy/(a*r) - 4*c1*fAv_xi*fBphiy_xi*gAv*gBphiy/(a*a)) - 4*F22*c1*fAv*fBphiy*gAv_eta*gBphiy_eta/((b*b)*r) + F26*(-4*c1*fAv*fBphiy_xi*gAv_eta*gBphiy/(a*b*r) - 6*c1*fAv_xi*fBphiy*gAv*gBphiy_eta/(a*b*r)) - 6*F66*c1*fAv_xi*fBphiy_xi*gAv*gBphiy/((a*a)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( -4*A11*bx*fAw_xi*fBu_xi*gAw*gBu/(a*a) + A12*(2*fAw*fBu_xi*gAw*gBu/(a*r) - 4*by*fAw*fBu_xi*gAw_eta*gBu/(a*b)) + A16*(-4*bx*fAw*fBu_xi*gAw_eta*gBu/(a*b) - 4*bx*fAw_xi*fBu*gAw*gBu_eta/(a*b) - 4*by*fAw_xi*fBu_xi*gAw*gBu/(a*a)) + A26*(2*fAw*fBu*gAw*gBu_eta/(b*r) - 4*by*fAw*fBu*gAw_eta*gBu_eta/(b*b)) + A66*(-4*bx*fAw*fBu*gAw_eta*gBu_eta/(b*b) - 4*by*fAw_xi*fBu*gAw*gBu_eta/(a*b)) + 2*B16*bx*fAw_xi*fBu*gAw*gBu_eta/(a*b*r) + B26*(-fAw*fBu*gAw*gBu_eta/(b*(r*r)) + 2*by*fAw*fBu*gAw_eta*gBu_eta/((b*b)*r)) + B66*(2*bx*fAw*fBu*gAw_eta*gBu_eta/((b*b)*r) + 2*by*fAw_xi*fBu*gAw*gBu_eta/(a*b*r)) - 8*E11*c1*fAw_xixi*fBu_xi*gAw*gBu/(a*a*a) - 8*E12*c1*fAw*fBu_xi*gAw_etaeta*gBu/(a*(b*b)) + E16*(-16*c1*fAw_xi*fBu_xi*gAw_eta*gBu/((a*a)*b) - 8*c1*fAw_xixi*fBu*gAw*gBu_eta/((a*a)*b)) - 8*E26*c1*fAw*fBu*gAw_etaeta*gBu_eta/(b*b*b) - 16*E66*c1*fAw_xi*fBu*gAw_eta*gBu_eta/(a*(b*b)) + 4*F16*c1*fAw_xixi*fBu*gAw*gBu_eta/((a*a)*b*r) + 4*F26*c1*fAw*fBu*gAw_etaeta*gBu_eta/((b*b*b)*r) + 8*F66*c1*fAw_xi*fBu*gAw_eta*gBu_eta/(a*(b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( A12*(-2*bx*by*fAw_xi*fBv*gAw*gBv/(a*r) - 4*bx*fAw_xi*fBv*gAw*gBv_eta/(a*b)) + A16*(-2*(bx*bx)*fAw_xi*fBv*gAw*gBv/(a*r) - 4*bx*fAw_xi*fBv_xi*gAw*gBv/(a*a)) + A22*(by*fAw*fBv*gAw*gBv/(r*r) - 2*(by*by)*fAw*fBv*gAw_eta*gBv/(b*r) + 2*fAw*fBv*gAw*gBv_eta/(b*r) - 4*by*fAw*fBv*gAw_eta*gBv_eta/(b*b)) + A26*(bx*fAw*fBv*gAw*gBv/(r*r) - 4*bx*by*fAw*fBv*gAw_eta*gBv/(b*r) - 4*bx*fAw*fBv*gAw_eta*gBv_eta/(b*b) - 2*(by*by)*fAw_xi*fBv*gAw*gBv/(a*r) + 2*fAw*fBv_xi*gAw*gBv/(a*r) - 4*by*fAw*fBv_xi*gAw_eta*gBv/(a*b) - 4*by*fAw_xi*fBv*gAw*gBv_eta/(a*b)) + A66*(-2*(bx*bx)*fAw*fBv*gAw_eta*gBv/(b*r) - 2*bx*by*fAw_xi*fBv*gAw*gBv/(a*r) - 4*bx*fAw*fBv_xi*gAw_eta*gBv/(a*b) - 4*by*fAw_xi*fBv_xi*gAw*gBv/(a*a)) - 4*B12*bx*fAw_xi*fBv*gAw*gBv_eta/(a*b*r) - 6*B16*bx*fAw_xi*fBv_xi*gAw*gBv/((a*a)*r) + B22*(2*fAw*fBv*gAw*gBv_eta/(b*(r*r)) - 4*by*fAw*fBv*gAw_eta*gBv_eta/((b*b)*r)) + B26*(-4*bx*fAw*fBv*gAw_eta*gBv_eta/((b*b)*r) + 3*fAw*fBv_xi*gAw*gBv/(a*(r*r)) - 6*by*fAw*fBv_xi*gAw_eta*gBv/(a*b*r) - 4*by*fAw_xi*fBv*gAw*gBv_eta/(a*b*r)) + B66*(-6*bx*fAw*fBv_xi*gAw_eta*gBv/(a*b*r) - 6*by*fAw_xi*fBv_xi*gAw*gBv/((a*a)*r)) + E12*(-4*by*c1*fAw_xixi*fBv*gAw*gBv/((a*a)*r) - 8*c1*fAw_xixi*fBv*gAw*gBv_eta/((a*a)*b)) + E16*(-4*bx*c1*fAw_xixi*fBv*gAw*gBv/((a*a)*r) - 8*c1*fAw_xixi*fBv_xi*gAw*gBv/(a*a*a)) + E22*(-4*by*c1*fAw*fBv*gAw_etaeta*gBv/((b*b)*r) - 8*c1*fAw*fBv*gAw_etaeta*gBv_eta/(b*b*b)) + E26*(-4*bx*c1*fAw*fBv*gAw_etaeta*gBv/((b*b)*r) - 8*by*c1*fAw_xi*fBv*gAw_eta*gBv/(a*b*r) - 8*c1*fAw*fBv_xi*gAw_etaeta*gBv/(a*(b*b)) - 16*c1*fAw_xi*fBv*gAw_eta*gBv_eta/(a*(b*b))) + E66*(-8*bx*c1*fAw_xi*fBv*gAw_eta*gBv/(a*b*r) - 16*c1*fAw_xi*fBv_xi*gAw_eta*gBv/((a*a)*b)) - 8*F12*c1*fAw_xixi*fBv*gAw*gBv_eta/((a*a)*b*r) - 12*F16*c1*fAw_xixi*fBv_xi*gAw*gBv/((a*a*a)*r) - 8*F22*c1*fAw*fBv*gAw_etaeta*gBv_eta/((b*b*b)*r) + F26*(-12*c1*fAw*fBv_xi*gAw_etaeta*gBv/(a*(b*b)*r) - 16*c1*fAw_xi*fBv*gAw_eta*gBv_eta/(a*(b*b)*r)) - 24*F66*c1*fAw_xi*fBv_xi*gAw_eta*gBv/((a*a)*b*r) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( -2*NxyNL*fAw_xi*fBv*gAw*gBv/(a*r) - 2*NyyNL*fAw*fBv*gAw_eta*gBv/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*(bx*bx)*fAw_xi*fBw_xi*gAw*gBw/(a*a) + A12*(-2*bx*fAw*fBw_xi*gAw*gBw/(a*r) - 2*bx*fAw_xi*fBw*gAw*gBw/(a*r) + 4*bx*by*fAw*fBw_xi*gAw_eta*gBw/(a*b) + 4*bx*by*fAw_xi*fBw*gAw*gBw_eta/(a*b)) + A16*(4*(bx*bx)*fAw*fBw_xi*gAw_eta*gBw/(a*b) + 4*(bx*bx)*fAw_xi*fBw*gAw*gBw_eta/(a*b) + 8*bx*by*fAw_xi*fBw_xi*gAw*gBw/(a*a)) + A22*(fAw*fBw*gAw*gBw/(r*r) - 2*by*fAw*fBw*gAw*gBw_eta/(b*r) - 2*by*fAw*fBw*gAw_eta*gBw/(b*r) + 4*(by*by)*fAw*fBw*gAw_eta*gBw_eta/(b*b)) + A26*(-2*bx*fAw*fBw*gAw*gBw_eta/(b*r) - 2*bx*fAw*fBw*gAw_eta*gBw/(b*r) + 8*bx*by*fAw*fBw*gAw_eta*gBw_eta/(b*b) - 2*by*fAw*fBw_xi*gAw*gBw/(a*r) - 2*by*fAw_xi*fBw*gAw*gBw/(a*r) + 4*(by*by)*fAw*fBw_xi*gAw_eta*gBw/(a*b) + 4*(by*by)*fAw_xi*fBw*gAw*gBw_eta/(a*b)) + 4*A44*fAw*fBw*gAw_eta*gBw_eta/(b*b) + A45*(4*fAw*fBw_xi*gAw_eta*gBw/(a*b) + 4*fAw_xi*fBw*gAw*gBw_eta/(a*b)) + 4*A55*fAw_xi*fBw_xi*gAw*gBw/(a*a) + A66*(4*(bx*bx)*fAw*fBw*gAw_eta*gBw_eta/(b*b) + 4*bx*by*fAw*fBw_xi*gAw_eta*gBw/(a*b) + 4*bx*by*fAw_xi*fBw*gAw*gBw_eta/(a*b) + 4*(by*by)*fAw_xi*fBw_xi*gAw*gBw/(a*a)) - 24*D44*c1*fAw*fBw*gAw_eta*gBw_eta/(b*b) + D45*(-24*c1*fAw*fBw_xi*gAw_eta*gBw/(a*b) - 24*c1*fAw_xi*fBw*gAw*gBw_eta/(a*b)) - 24*D55*c1*fAw_xi*fBw_xi*gAw*gBw/(a*a) + E11*(8*bx*c1*fAw_xi*fBw_xixi*gAw*gBw/(a*a*a) + 8*bx*c1*fAw_xixi*fBw_xi*gAw*gBw/(a*a*a)) + E12*(8*bx*c1*fAw*fBw_xi*gAw_etaeta*gBw/(a*(b*b)) + 8*bx*c1*fAw_xi*fBw*gAw*gBw_etaeta/(a*(b*b)) - 4*c1*fAw*fBw_xixi*gAw*gBw/((a*a)*r) - 4*c1*fAw_xixi*fBw*gAw*gBw/((a*a)*r) + 8*by*c1*fAw*fBw_xixi*gAw_eta*gBw/((a*a)*b) + 8*by*c1*fAw_xixi*fBw*gAw*gBw_eta/((a*a)*b)) + E16*(8*bx*c1*fAw*fBw_xixi*gAw_eta*gBw/((a*a)*b) + 16*bx*c1*fAw_xi*fBw_xi*gAw*gBw_eta/((a*a)*b) + 16*bx*c1*fAw_xi*fBw_xi*gAw_eta*gBw/((a*a)*b) + 8*bx*c1*fAw_xixi*fBw*gAw*gBw_eta/((a*a)*b) + 8*by*c1*fAw_xi*fBw_xixi*gAw*gBw/(a*a*a) + 8*by*c1*fAw_xixi*fBw_xi*gAw*gBw/(a*a*a)) + E22*(-4*c1*fAw*fBw*gAw*gBw_etaeta/((b*b)*r) - 4*c1*fAw*fBw*gAw_etaeta*gBw/((b*b)*r) + 8*by*c1*fAw*fBw*gAw_eta*gBw_etaeta/(b*b*b) + 8*by*c1*fAw*fBw*gAw_etaeta*gBw_eta/(b*b*b)) + E26*(8*bx*c1*fAw*fBw*gAw_eta*gBw_etaeta/(b*b*b) + 8*bx*c1*fAw*fBw*gAw_etaeta*gBw_eta/(b*b*b) - 8*c1*fAw*fBw_xi*gAw*gBw_eta/(a*b*r) - 8*c1*fAw_xi*fBw*gAw_eta*gBw/(a*b*r) + 16*by*c1*fAw*fBw_xi*gAw_eta*gBw_eta/(a*(b*b)) + 8*by*c1*fAw*fBw_xi*gAw_etaeta*gBw/(a*(b*b)) + 8*by*c1*fAw_xi*fBw*gAw*gBw_etaeta/(a*(b*b)) + 16*by*c1*fAw_xi*fBw*gAw_eta*gBw_eta/(a*(b*b))) + E66*(16*bx*c1*fAw*fBw_xi*gAw_eta*gBw_eta/(a*(b*b)) + 16*bx*c1*fAw_xi*fBw*gAw_eta*gBw_eta/(a*(b*b)) + 16*by*c1*fAw_xi*fBw_xi*gAw*gBw_eta/((a*a)*b) + 16*by*c1*fAw_xi*fBw_xi*gAw_eta*gBw/((a*a)*b)) + 36*F44*(c1*c1)*fAw*fBw*gAw_eta*gBw_eta/(b*b) + F45*(36*(c1*c1)*fAw*fBw_xi*gAw_eta*gBw/(a*b) + 36*(c1*c1)*fAw_xi*fBw*gAw*gBw_eta/(a*b)) + 36*F55*(c1*c1)*fAw_xi*fBw_xi*gAw*gBw/(a*a) + 16*H11*(c1*c1)*fAw_xixi*fBw_xixi*gAw*gBw/(a*a*a*a) + H12*(16*(c1*c1)*fAw*fBw_xixi*gAw_etaeta*gBw/((a*a)*(b*b)) + 16*(c1*c1)*fAw_xixi*fBw*gAw*gBw_etaeta/((a*a)*(b*b))) + H16*(32*(c1*c1)*fAw_xi*fBw_xixi*gAw_eta*gBw/((a*a*a)*b) + 32*(c1*c1)*fAw_xixi*fBw_xi*gAw*gBw_eta/((a*a*a)*b)) + 16*H22*(c1*c1)*fAw*fBw*gAw_etaeta*gBw_etaeta/(b*b*b*b) + H26*(32*(c1*c1)*fAw*fBw_xi*gAw_etaeta*gBw_eta/(a*(b*b*b)) + 32*(c1*c1)*fAw_xi*fBw*gAw_eta*gBw_etaeta/(a*(b*b*b))) + 64*H66*(c1*c1)*fAw_xi*fBw_xi*gAw_eta*gBw_eta/((a*a)*(b*b)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( 4*NxxNL*fAw_xi*fBw_xi*gAw*gBw/(a*a) + 4*NxyNL*(fAw*fBw_xi*gAw_eta*gBw + fAw_xi*fBw*gAw*gBw_eta)/(a*b) + 4*NyyNL*fAw*fBw*gAw_eta*gBw_eta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 2*A45*fAw*fBphix*gAw_eta*gBphix/b + 2*A55*fAw_xi*fBphix*gAw*gBphix/a - 4*B11*bx*fAw_xi*fBphix_xi*gAw*gBphix/(a*a) + B12*(2*fAw*fBphix_xi*gAw*gBphix/(a*r) - 4*by*fAw*fBphix_xi*gAw_eta*gBphix/(a*b)) + B16*(-4*bx*fAw*fBphix_xi*gAw_eta*gBphix/(a*b) - 4*bx*fAw_xi*fBphix*gAw*gBphix_eta/(a*b) - 4*by*fAw_xi*fBphix_xi*gAw*gBphix/(a*a)) + B26*(2*fAw*fBphix*gAw*gBphix_eta/(b*r) - 4*by*fAw*fBphix*gAw_eta*gBphix_eta/(b*b)) + B66*(-4*bx*fAw*fBphix*gAw_eta*gBphix_eta/(b*b) - 4*by*fAw_xi*fBphix*gAw*gBphix_eta/(a*b)) - 12*D45*c1*fAw*fBphix*gAw_eta*gBphix/b - 12*D55*c1*fAw_xi*fBphix*gAw*gBphix/a + 4*E11*bx*c1*fAw_xi*fBphix_xi*gAw*gBphix/(a*a) + E12*(-2*c1*fAw*fBphix_xi*gAw*gBphix/(a*r) + 4*by*c1*fAw*fBphix_xi*gAw_eta*gBphix/(a*b)) + E16*(4*bx*c1*fAw*fBphix_xi*gAw_eta*gBphix/(a*b) + 4*bx*c1*fAw_xi*fBphix*gAw*gBphix_eta/(a*b) + 4*by*c1*fAw_xi*fBphix_xi*gAw*gBphix/(a*a)) + E26*(-2*c1*fAw*fBphix*gAw*gBphix_eta/(b*r) + 4*by*c1*fAw*fBphix*gAw_eta*gBphix_eta/(b*b)) + E66*(4*bx*c1*fAw*fBphix*gAw_eta*gBphix_eta/(b*b) + 4*by*c1*fAw_xi*fBphix*gAw*gBphix_eta/(a*b)) - 8*F11*c1*fAw_xixi*fBphix_xi*gAw*gBphix/(a*a*a) - 8*F12*c1*fAw*fBphix_xi*gAw_etaeta*gBphix/(a*(b*b)) + F16*(-16*c1*fAw_xi*fBphix_xi*gAw_eta*gBphix/((a*a)*b) - 8*c1*fAw_xixi*fBphix*gAw*gBphix_eta/((a*a)*b)) - 8*F26*c1*fAw*fBphix*gAw_etaeta*gBphix_eta/(b*b*b) + 18*F45*(c1*c1)*fAw*fBphix*gAw_eta*gBphix/b + 18*F55*(c1*c1)*fAw_xi*fBphix*gAw*gBphix/a - 16*F66*c1*fAw_xi*fBphix*gAw_eta*gBphix_eta/(a*(b*b)) + 8*H11*(c1*c1)*fAw_xixi*fBphix_xi*gAw*gBphix/(a*a*a) + 8*H12*(c1*c1)*fAw*fBphix_xi*gAw_etaeta*gBphix/(a*(b*b)) + H16*(16*(c1*c1)*fAw_xi*fBphix_xi*gAw_eta*gBphix/((a*a)*b) + 8*(c1*c1)*fAw_xixi*fBphix*gAw*gBphix_eta/((a*a)*b)) + 8*H26*(c1*c1)*fAw*fBphix*gAw_etaeta*gBphix_eta/(b*b*b) + 16*H66*(c1*c1)*fAw_xi*fBphix*gAw_eta*gBphix_eta/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 2*A44*fAw*fBphiy*gAw_eta*gBphiy/b + 2*A45*fAw_xi*fBphiy*gAw*gBphiy/a - 4*B12*bx*fAw_xi*fBphiy*gAw*gBphiy_eta/(a*b) - 4*B16*bx*fAw_xi*fBphiy_xi*gAw*gBphiy/(a*a) + B22*(2*fAw*fBphiy*gAw*gBphiy_eta/(b*r) - 4*by*fAw*fBphiy*gAw_eta*gBphiy_eta/(b*b)) + B26*(-4*bx*fAw*fBphiy*gAw_eta*gBphiy_eta/(b*b) + 2*fAw*fBphiy_xi*gAw*gBphiy/(a*r) - 4*by*fAw*fBphiy_xi*gAw_eta*gBphiy/(a*b) - 4*by*fAw_xi*fBphiy*gAw*gBphiy_eta/(a*b)) + B66*(-4*bx*fAw*fBphiy_xi*gAw_eta*gBphiy/(a*b) - 4*by*fAw_xi*fBphiy_xi*gAw*gBphiy/(a*a)) - 12*D44*c1*fAw*fBphiy*gAw_eta*gBphiy/b - 12*D45*c1*fAw_xi*fBphiy*gAw*gBphiy/a + 4*E12*bx*c1*fAw_xi*fBphiy*gAw*gBphiy_eta/(a*b) + 4*E16*bx*c1*fAw_xi*fBphiy_xi*gAw*gBphiy/(a*a) + E22*(-2*c1*fAw*fBphiy*gAw*gBphiy_eta/(b*r) + 4*by*c1*fAw*fBphiy*gAw_eta*gBphiy_eta/(b*b)) + E26*(4*bx*c1*fAw*fBphiy*gAw_eta*gBphiy_eta/(b*b) - 2*c1*fAw*fBphiy_xi*gAw*gBphiy/(a*r) + 4*by*c1*fAw*fBphiy_xi*gAw_eta*gBphiy/(a*b) + 4*by*c1*fAw_xi*fBphiy*gAw*gBphiy_eta/(a*b)) + E66*(4*bx*c1*fAw*fBphiy_xi*gAw_eta*gBphiy/(a*b) + 4*by*c1*fAw_xi*fBphiy_xi*gAw*gBphiy/(a*a)) - 8*F12*c1*fAw_xixi*fBphiy*gAw*gBphiy_eta/((a*a)*b) - 8*F16*c1*fAw_xixi*fBphiy_xi*gAw*gBphiy/(a*a*a) - 8*F22*c1*fAw*fBphiy*gAw_etaeta*gBphiy_eta/(b*b*b) + F26*(-8*c1*fAw*fBphiy_xi*gAw_etaeta*gBphiy/(a*(b*b)) - 16*c1*fAw_xi*fBphiy*gAw_eta*gBphiy_eta/(a*(b*b))) + 18*F44*(c1*c1)*fAw*fBphiy*gAw_eta*gBphiy/b + 18*F45*(c1*c1)*fAw_xi*fBphiy*gAw*gBphiy/a - 16*F66*c1*fAw_xi*fBphiy_xi*gAw_eta*gBphiy/((a*a)*b) + 8*H12*(c1*c1)*fAw_xixi*fBphiy*gAw*gBphiy_eta/((a*a)*b) + 8*H16*(c1*c1)*fAw_xixi*fBphiy_xi*gAw*gBphiy/(a*a*a) + 8*H22*(c1*c1)*fAw*fBphiy*gAw_etaeta*gBphiy_eta/(b*b*b) + H26*(8*(c1*c1)*fAw*fBphiy_xi*gAw_etaeta*gBphiy/(a*(b*b)) + 16*(c1*c1)*fAw_xi*fBphiy*gAw_eta*gBphiy_eta/(a*(b*b))) + 16*H66*(c1*c1)*fAw_xi*fBphiy_xi*gAw_eta*gBphiy/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*B11*fAphix_xi*fBu_xi*gAphix*gBu/(a*a) + B16*(4*fAphix*fBu_xi*gAphix_eta*gBu/(a*b) + 4*fAphix_xi*fBu*gAphix*gBu_eta/(a*b)) + 4*B66*fAphix*fBu*gAphix_eta*gBu_eta/(b*b) - 2*D16*fAphix_xi*fBu*gAphix*gBu_eta/(a*b*r) - 2*D66*fAphix*fBu*gAphix_eta*gBu_eta/((b*b)*r) - 4*E11*c1*fAphix_xi*fBu_xi*gAphix*gBu/(a*a) + E16*(-4*c1*fAphix*fBu_xi*gAphix_eta*gBu/(a*b) - 4*c1*fAphix_xi*fBu*gAphix*gBu_eta/(a*b)) - 4*E66*c1*fAphix*fBu*gAphix_eta*gBu_eta/(b*b) + 2*F16*c1*fAphix_xi*fBu*gAphix*gBu_eta/(a*b*r) + 2*F66*c1*fAphix*fBu*gAphix_eta*gBu_eta/((b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( B12*(2*by*fAphix_xi*fBv*gAphix*gBv/(a*r) + 4*fAphix_xi*fBv*gAphix*gBv_eta/(a*b)) + B16*(2*bx*fAphix_xi*fBv*gAphix*gBv/(a*r) + 4*fAphix_xi*fBv_xi*gAphix*gBv/(a*a)) + B26*(2*by*fAphix*fBv*gAphix_eta*gBv/(b*r) + 4*fAphix*fBv*gAphix_eta*gBv_eta/(b*b)) + B66*(2*bx*fAphix*fBv*gAphix_eta*gBv/(b*r) + 4*fAphix*fBv_xi*gAphix_eta*gBv/(a*b)) + 4*D12*fAphix_xi*fBv*gAphix*gBv_eta/(a*b*r) + 6*D16*fAphix_xi*fBv_xi*gAphix*gBv/((a*a)*r) + 4*D26*fAphix*fBv*gAphix_eta*gBv_eta/((b*b)*r) + 6*D66*fAphix*fBv_xi*gAphix_eta*gBv/(a*b*r) + E12*(-2*by*c1*fAphix_xi*fBv*gAphix*gBv/(a*r) - 4*c1*fAphix_xi*fBv*gAphix*gBv_eta/(a*b)) + E16*(-2*bx*c1*fAphix_xi*fBv*gAphix*gBv/(a*r) - 4*c1*fAphix_xi*fBv_xi*gAphix*gBv/(a*a)) + E26*(-2*by*c1*fAphix*fBv*gAphix_eta*gBv/(b*r) - 4*c1*fAphix*fBv*gAphix_eta*gBv_eta/(b*b)) + E66*(-2*bx*c1*fAphix*fBv*gAphix_eta*gBv/(b*r) - 4*c1*fAphix*fBv_xi*gAphix_eta*gBv/(a*b)) - 4*F12*c1*fAphix_xi*fBv*gAphix*gBv_eta/(a*b*r) - 6*F16*c1*fAphix_xi*fBv_xi*gAphix*gBv/((a*a)*r) - 4*F26*c1*fAphix*fBv*gAphix_eta*gBv_eta/((b*b)*r) - 6*F66*c1*fAphix*fBv_xi*gAphix_eta*gBv/(a*b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 2*A45*fAphix*fBw*gAphix*gBw_eta/b + 2*A55*fAphix*fBw_xi*gAphix*gBw/a - 4*B11*bx*fAphix_xi*fBw_xi*gAphix*gBw/(a*a) + B12*(2*fAphix_xi*fBw*gAphix*gBw/(a*r) - 4*by*fAphix_xi*fBw*gAphix*gBw_eta/(a*b)) + B16*(-4*bx*fAphix*fBw_xi*gAphix_eta*gBw/(a*b) - 4*bx*fAphix_xi*fBw*gAphix*gBw_eta/(a*b) - 4*by*fAphix_xi*fBw_xi*gAphix*gBw/(a*a)) + B26*(2*fAphix*fBw*gAphix_eta*gBw/(b*r) - 4*by*fAphix*fBw*gAphix_eta*gBw_eta/(b*b)) + B66*(-4*bx*fAphix*fBw*gAphix_eta*gBw_eta/(b*b) - 4*by*fAphix*fBw_xi*gAphix_eta*gBw/(a*b)) - 12*D45*c1*fAphix*fBw*gAphix*gBw_eta/b - 12*D55*c1*fAphix*fBw_xi*gAphix*gBw/a + 4*E11*bx*c1*fAphix_xi*fBw_xi*gAphix*gBw/(a*a) + E12*(-2*c1*fAphix_xi*fBw*gAphix*gBw/(a*r) + 4*by*c1*fAphix_xi*fBw*gAphix*gBw_eta/(a*b)) + E16*(4*bx*c1*fAphix*fBw_xi*gAphix_eta*gBw/(a*b) + 4*bx*c1*fAphix_xi*fBw*gAphix*gBw_eta/(a*b) + 4*by*c1*fAphix_xi*fBw_xi*gAphix*gBw/(a*a)) + E26*(-2*c1*fAphix*fBw*gAphix_eta*gBw/(b*r) + 4*by*c1*fAphix*fBw*gAphix_eta*gBw_eta/(b*b)) + E66*(4*bx*c1*fAphix*fBw*gAphix_eta*gBw_eta/(b*b) + 4*by*c1*fAphix*fBw_xi*gAphix_eta*gBw/(a*b)) - 8*F11*c1*fAphix_xi*fBw_xixi*gAphix*gBw/(a*a*a) - 8*F12*c1*fAphix_xi*fBw*gAphix*gBw_etaeta/(a*(b*b)) + F16*(-8*c1*fAphix*fBw_xixi*gAphix_eta*gBw/((a*a)*b) - 16*c1*fAphix_xi*fBw_xi*gAphix*gBw_eta/((a*a)*b)) - 8*F26*c1*fAphix*fBw*gAphix_eta*gBw_etaeta/(b*b*b) + 18*F45*(c1*c1)*fAphix*fBw*gAphix*gBw_eta/b + 18*F55*(c1*c1)*fAphix*fBw_xi*gAphix*gBw/a - 16*F66*c1*fAphix*fBw_xi*gAphix_eta*gBw_eta/(a*(b*b)) + 8*H11*(c1*c1)*fAphix_xi*fBw_xixi*gAphix*gBw/(a*a*a) + 8*H12*(c1*c1)*fAphix_xi*fBw*gAphix*gBw_etaeta/(a*(b*b)) + H16*(8*(c1*c1)*fAphix*fBw_xixi*gAphix_eta*gBw/((a*a)*b) + 16*(c1*c1)*fAphix_xi*fBw_xi*gAphix*gBw_eta/((a*a)*b)) + 8*H26*(c1*c1)*fAphix*fBw*gAphix_eta*gBw_etaeta/(b*b*b) + 16*H66*(c1*c1)*fAphix*fBw_xi*gAphix_eta*gBw_eta/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( A55*fAphix*fBphix*gAphix*gBphix + 4*D11*fAphix_xi*fBphix_xi*gAphix*gBphix/(a*a) + D16*(4*fAphix*fBphix_xi*gAphix_eta*gBphix/(a*b) + 4*fAphix_xi*fBphix*gAphix*gBphix_eta/(a*b)) - 6*D55*c1*fAphix*fBphix*gAphix*gBphix + 4*D66*fAphix*fBphix*gAphix_eta*gBphix_eta/(b*b) - 8*F11*c1*fAphix_xi*fBphix_xi*gAphix*gBphix/(a*a) + F16*(-8*c1*fAphix*fBphix_xi*gAphix_eta*gBphix/(a*b) - 8*c1*fAphix_xi*fBphix*gAphix*gBphix_eta/(a*b)) + 9*F55*(c1*c1)*fAphix*fBphix*gAphix*gBphix - 8*F66*c1*fAphix*fBphix*gAphix_eta*gBphix_eta/(b*b) + 4*H11*(c1*c1)*fAphix_xi*fBphix_xi*gAphix*gBphix/(a*a) + H16*(4*(c1*c1)*fAphix*fBphix_xi*gAphix_eta*gBphix/(a*b) + 4*(c1*c1)*fAphix_xi*fBphix*gAphix*gBphix_eta/(a*b)) + 4*H66*(c1*c1)*fAphix*fBphix*gAphix_eta*gBphix_eta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( A45*fAphix*fBphiy*gAphix*gBphiy + 4*D12*fAphix_xi*fBphiy*gAphix*gBphiy_eta/(a*b) + 4*D16*fAphix_xi*fBphiy_xi*gAphix*gBphiy/(a*a) + 4*D26*fAphix*fBphiy*gAphix_eta*gBphiy_eta/(b*b) - 6*D45*c1*fAphix*fBphiy*gAphix*gBphiy + 4*D66*fAphix*fBphiy_xi*gAphix_eta*gBphiy/(a*b) - 8*F12*c1*fAphix_xi*fBphiy*gAphix*gBphiy_eta/(a*b) - 8*F16*c1*fAphix_xi*fBphiy_xi*gAphix*gBphiy/(a*a) - 8*F26*c1*fAphix*fBphiy*gAphix_eta*gBphiy_eta/(b*b) + 9*F45*(c1*c1)*fAphix*fBphiy*gAphix*gBphiy - 8*F66*c1*fAphix*fBphiy_xi*gAphix_eta*gBphiy/(a*b) + 4*H12*(c1*c1)*fAphix_xi*fBphiy*gAphix*gBphiy_eta/(a*b) + 4*H16*(c1*c1)*fAphix_xi*fBphiy_xi*gAphix*gBphiy/(a*a) + 4*H26*(c1*c1)*fAphix*fBphiy*gAphix_eta*gBphiy_eta/(b*b) + 4*H66*(c1*c1)*fAphix*fBphiy_xi*gAphix_eta*gBphiy/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAphiy*fBu_xi*gAphiy_eta*gBu/(a*b) + 4*B16*fAphiy_xi*fBu_xi*gAphiy*gBu/(a*a) + 4*B26*fAphiy*fBu*gAphiy_eta*gBu_eta/(b*b) + 4*B66*fAphiy_xi*fBu*gAphiy*gBu_eta/(a*b) - 2*D26*fAphiy*fBu*gAphiy_eta*gBu_eta/((b*b)*r) - 2*D66*fAphiy_xi*fBu*gAphiy*gBu_eta/(a*b*r) - 4*E12*c1*fAphiy*fBu_xi*gAphiy_eta*gBu/(a*b) - 4*E16*c1*fAphiy_xi*fBu_xi*gAphiy*gBu/(a*a) - 4*E26*c1*fAphiy*fBu*gAphiy_eta*gBu_eta/(b*b) - 4*E66*c1*fAphiy_xi*fBu*gAphiy*gBu_eta/(a*b) + 2*F26*c1*fAphiy*fBu*gAphiy_eta*gBu_eta/((b*b)*r) + 2*F66*c1*fAphiy_xi*fBu*gAphiy*gBu_eta/(a*b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( B22*(2*by*fAphiy*fBv*gAphiy_eta*gBv/(b*r) + 4*fAphiy*fBv*gAphiy_eta*gBv_eta/(b*b)) + B26*(2*bx*fAphiy*fBv*gAphiy_eta*gBv/(b*r) + 2*by*fAphiy_xi*fBv*gAphiy*gBv/(a*r) + 4*fAphiy*fBv_xi*gAphiy_eta*gBv/(a*b) + 4*fAphiy_xi*fBv*gAphiy*gBv_eta/(a*b)) + B66*(2*bx*fAphiy_xi*fBv*gAphiy*gBv/(a*r) + 4*fAphiy_xi*fBv_xi*gAphiy*gBv/(a*a)) + 4*D22*fAphiy*fBv*gAphiy_eta*gBv_eta/((b*b)*r) + D26*(6*fAphiy*fBv_xi*gAphiy_eta*gBv/(a*b*r) + 4*fAphiy_xi*fBv*gAphiy*gBv_eta/(a*b*r)) + 6*D66*fAphiy_xi*fBv_xi*gAphiy*gBv/((a*a)*r) + E22*(-2*by*c1*fAphiy*fBv*gAphiy_eta*gBv/(b*r) - 4*c1*fAphiy*fBv*gAphiy_eta*gBv_eta/(b*b)) + E26*(-2*bx*c1*fAphiy*fBv*gAphiy_eta*gBv/(b*r) - 2*by*c1*fAphiy_xi*fBv*gAphiy*gBv/(a*r) - 4*c1*fAphiy*fBv_xi*gAphiy_eta*gBv/(a*b) - 4*c1*fAphiy_xi*fBv*gAphiy*gBv_eta/(a*b)) + E66*(-2*bx*c1*fAphiy_xi*fBv*gAphiy*gBv/(a*r) - 4*c1*fAphiy_xi*fBv_xi*gAphiy*gBv/(a*a)) - 4*F22*c1*fAphiy*fBv*gAphiy_eta*gBv_eta/((b*b)*r) + F26*(-6*c1*fAphiy*fBv_xi*gAphiy_eta*gBv/(a*b*r) - 4*c1*fAphiy_xi*fBv*gAphiy*gBv_eta/(a*b*r)) - 6*F66*c1*fAphiy_xi*fBv_xi*gAphiy*gBv/((a*a)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 2*A44*fAphiy*fBw*gAphiy*gBw_eta/b + 2*A45*fAphiy*fBw_xi*gAphiy*gBw/a - 4*B12*bx*fAphiy*fBw_xi*gAphiy_eta*gBw/(a*b) - 4*B16*bx*fAphiy_xi*fBw_xi*gAphiy*gBw/(a*a) + B22*(2*fAphiy*fBw*gAphiy_eta*gBw/(b*r) - 4*by*fAphiy*fBw*gAphiy_eta*gBw_eta/(b*b)) + B26*(-4*bx*fAphiy*fBw*gAphiy_eta*gBw_eta/(b*b) + 2*fAphiy_xi*fBw*gAphiy*gBw/(a*r) - 4*by*fAphiy*fBw_xi*gAphiy_eta*gBw/(a*b) - 4*by*fAphiy_xi*fBw*gAphiy*gBw_eta/(a*b)) + B66*(-4*bx*fAphiy_xi*fBw*gAphiy*gBw_eta/(a*b) - 4*by*fAphiy_xi*fBw_xi*gAphiy*gBw/(a*a)) - 12*D44*c1*fAphiy*fBw*gAphiy*gBw_eta/b - 12*D45*c1*fAphiy*fBw_xi*gAphiy*gBw/a + 4*E12*bx*c1*fAphiy*fBw_xi*gAphiy_eta*gBw/(a*b) + 4*E16*bx*c1*fAphiy_xi*fBw_xi*gAphiy*gBw/(a*a) + E22*(-2*c1*fAphiy*fBw*gAphiy_eta*gBw/(b*r) + 4*by*c1*fAphiy*fBw*gAphiy_eta*gBw_eta/(b*b)) + E26*(4*bx*c1*fAphiy*fBw*gAphiy_eta*gBw_eta/(b*b) - 2*c1*fAphiy_xi*fBw*gAphiy*gBw/(a*r) + 4*by*c1*fAphiy*fBw_xi*gAphiy_eta*gBw/(a*b) + 4*by*c1*fAphiy_xi*fBw*gAphiy*gBw_eta/(a*b)) + E66*(4*bx*c1*fAphiy_xi*fBw*gAphiy*gBw_eta/(a*b) + 4*by*c1*fAphiy_xi*fBw_xi*gAphiy*gBw/(a*a)) - 8*F12*c1*fAphiy*fBw_xixi*gAphiy_eta*gBw/((a*a)*b) - 8*F16*c1*fAphiy_xi*fBw_xixi*gAphiy*gBw/(a*a*a) - 8*F22*c1*fAphiy*fBw*gAphiy_eta*gBw_etaeta/(b*b*b) + F26*(-16*c1*fAphiy*fBw_xi*gAphiy_eta*gBw_eta/(a*(b*b)) - 8*c1*fAphiy_xi*fBw*gAphiy*gBw_etaeta/(a*(b*b))) + 18*F44*(c1*c1)*fAphiy*fBw*gAphiy*gBw_eta/b + 18*F45*(c1*c1)*fAphiy*fBw_xi*gAphiy*gBw/a - 16*F66*c1*fAphiy_xi*fBw_xi*gAphiy*gBw_eta/((a*a)*b) + 8*H12*(c1*c1)*fAphiy*fBw_xixi*gAphiy_eta*gBw/((a*a)*b) + 8*H16*(c1*c1)*fAphiy_xi*fBw_xixi*gAphiy*gBw/(a*a*a) + 8*H22*(c1*c1)*fAphiy*fBw*gAphiy_eta*gBw_etaeta/(b*b*b) + H26*(16*(c1*c1)*fAphiy*fBw_xi*gAphiy_eta*gBw_eta/(a*(b*b)) + 8*(c1*c1)*fAphiy_xi*fBw*gAphiy*gBw_etaeta/(a*(b*b))) + 16*H66*(c1*c1)*fAphiy_xi*fBw_xi*gAphiy*gBw_eta/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( A45*fAphiy*fBphix*gAphiy*gBphix + 4*D12*fAphiy*fBphix_xi*gAphiy_eta*gBphix/(a*b) + 4*D16*fAphiy_xi*fBphix_xi*gAphiy*gBphix/(a*a) + 4*D26*fAphiy*fBphix*gAphiy_eta*gBphix_eta/(b*b) - 6*D45*c1*fAphiy*fBphix*gAphiy*gBphix + 4*D66*fAphiy_xi*fBphix*gAphiy*gBphix_eta/(a*b) - 8*F12*c1*fAphiy*fBphix_xi*gAphiy_eta*gBphix/(a*b) - 8*F16*c1*fAphiy_xi*fBphix_xi*gAphiy*gBphix/(a*a) - 8*F26*c1*fAphiy*fBphix*gAphiy_eta*gBphix_eta/(b*b) + 9*F45*(c1*c1)*fAphiy*fBphix*gAphiy*gBphix - 8*F66*c1*fAphiy_xi*fBphix*gAphiy*gBphix_eta/(a*b) + 4*H12*(c1*c1)*fAphiy*fBphix_xi*gAphiy_eta*gBphix/(a*b) + 4*H16*(c1*c1)*fAphiy_xi*fBphix_xi*gAphiy*gBphix/(a*a) + 4*H26*(c1*c1)*fAphiy*fBphix*gAphiy_eta*gBphix_eta/(b*b) + 4*H66*(c1*c1)*fAphiy_xi*fBphix*gAphiy*gBphix_eta/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( A44*fAphiy*fBphiy*gAphiy*gBphiy + 4*D22*fAphiy*fBphiy*gAphiy_eta*gBphiy_eta/(b*b) + D26*(4*fAphiy*fBphiy_xi*gAphiy_eta*gBphiy/(a*b) + 4*fAphiy_xi*fBphiy*gAphiy*gBphiy_eta/(a*b)) - 6*D44*c1*fAphiy*fBphiy*gAphiy*gBphiy + 4*D66*fAphiy_xi*fBphiy_xi*gAphiy*gBphiy/(a*a) - 8*F22*c1*fAphiy*fBphiy*gAphiy_eta*gBphiy_eta/(b*b) + F26*(-8*c1*fAphiy*fBphiy_xi*gAphiy_eta*gBphiy/(a*b) - 8*c1*fAphiy_xi*fBphiy*gAphiy*gBphiy_eta/(a*b)) + 9*F44*(c1*c1)*fAphiy*fBphiy*gAphiy*gBphiy - 8*F66*c1*fAphiy_xi*fBphiy_xi*gAphiy*gBphiy/(a*a) + 4*H22*(c1*c1)*fAphiy*fBphiy*gAphiy_eta*gBphiy_eta/(b*b) + H26*(4*(c1*c1)*fAphiy*fBphiy_xi*gAphiy_eta*gBphiy/(a*b) + 4*(c1*c1)*fAphiy_xi*fBphiy*gAphiy*gBphiy_eta/(a*b)) + 4*H66*(c1*c1)*fAphiy_xi*fBphiy_xi*gAphiy*gBphiy/(a*a) )

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

    There is deliberately no switch to include the stress of the non-linear
    strain here. That contribution, ``KGNL``, is collected by
    :func:`.fkC_num` instead, so that

        KT = KC0 + KCNL + KGNL   (fkC_num)
             + KG(Nbar + N_L)    (fkG_num)

    is the exact Jacobian of :func:`.calc_fint` while this function stays
    homogeneous of degree one in ``cs``.

    """
    cdef double [:, ::1] gtab, ftab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double r
    cdef double a, b, intx, inty, h, c1
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double x1phix, x1phixr, x2phix, x2phixr
    cdef double x1phiy, x1phiyr, x2phiy, x2phiyr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double y1phix, y1phixr, y2phix, y2phixr
    cdef double y1phiy, y1phiyr, y2phiy, y2phiyr

    cdef int i, j, k, l, c, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef long [::1] kGr, kGc
    cdef double [::1] kGv

    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw
    cdef double fAw_xi, fAw_xixi, fAphix, fAphix_xi, fAphiy
    cdef double fAphiy_xi
    cdef double fBv, fBw, fBw_xi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw
    cdef double gAw_eta, gAw_etaeta, gAphix, gAphix_eta, gAphiy
    cdef double gAphiy_eta
    cdef double gBv, gBw, gBw_eta

    cdef double e[13]
    cdef double Nxx, Nyy, Nxy

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # F as 4-D matrix, must be [nx, ny, 13, 13], when there is one
    # constitutive matrix [13, 13] for each of the nx * ny integration points
    cdef double F[169]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, NE, NE):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (NE, NE):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        for i in range(NE):
            for j in range(NE):
                F[i*NE + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    #NOTE the traction-free faces are assumed at z = +-h/2
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    x1phix = shell.x1phix; x1phixr = shell.x1phixr; x2phix = shell.x2phix; x2phixr = shell.x2phixr
    x1phiy = shell.x1phiy; x1phiyr = shell.x1phiyr; x2phiy = shell.x2phiy; x2phiyr = shell.x2phiyr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    y1phix = shell.y1phix; y1phixr = shell.y1phixr; y2phix = shell.y2phix; y2phixr = shell.y2phixr
    y1phiy = shell.y1phiy; y1phiyr = shell.y1phiyr; y2phiy = shell.y2phiy; y2phiyr = shell.y2phiyr

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

    gtab = np.zeros((11, n), dtype=DOUBLE)
    ftab = np.zeros((11, m), dtype=DOUBLE)

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
                    for i in range(NE):
                        for j in range(NE):
                            #TODO could assume symmetry
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

                # Calculating the linear generalized strains. The stress of
                # the nonlinear strain enters KT through KGNL in fkC_num, such
                # that kG is homogeneous of degree one in cs
                for i in range(NE):
                    e[i] = 0.
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[9, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[10, j] = fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[9, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_eta = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_eta = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_eta = gtab[5, j]
                    gAw_etaeta = gtab[6, j]
                    gAphix = gtab[7, j]
                    gAphix_eta = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_eta = gtab[10, j]

                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_xi = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        fAw_xixi = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_xi = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_xi = ftab[10, i]

                        col = col0 + DOF*(j*m + i)

                        e[0] += 2*cs[col+0]*fAu_xi*gAu/a
                        e[1] += cs[col+2]*fAw*gAw/r + 2*cs[col+1]*fAv*gAv_eta/b
                        e[2] += 2*cs[col+0]*fAu*gAu_eta/b + 2*cs[col+1]*fAv_xi*gAv/a
                        e[3] += 2*cs[col+3]*fAphix_xi*gAphix/a
                        e[4] += 2*cs[col+1]*fAv*gAv_eta/(b*r) + 2*cs[col+4]*fAphiy*gAphiy_eta/b
                        e[5] += -cs[col+0]*fAu*gAu_eta/(b*r) + 2*cs[col+3]*fAphix*gAphix_eta/b + 3*cs[col+1]*fAv_xi*gAv/(a*r) + 2*cs[col+4]*fAphiy_xi*gAphiy/a
                        e[6] += -2*c1*cs[col+3]*fAphix_xi*gAphix/a - 4*c1*cs[col+2]*fAw_xixi*gAw/(a*a)
                        e[7] += -2*c1*cs[col+4]*fAphiy*gAphiy_eta/b - 4*c1*cs[col+2]*fAw*gAw_etaeta/(b*b)
                        e[8] += -2*c1*cs[col+3]*fAphix*gAphix_eta/b - 2*c1*cs[col+4]*fAphiy_xi*gAphiy/a - 8*c1*cs[col+2]*fAw_xi*gAw_eta/(a*b)
                        e[9] += cs[col+4]*fAphiy*gAphiy + 2*cs[col+2]*fAw*gAw_eta/b
                        e[10] += cs[col+3]*fAphix*gAphix + 2*cs[col+2]*fAw_xi*gAw/a
                        e[11] += -3*c1*cs[col+4]*fAphiy*gAphiy - 6*c1*cs[col+2]*fAw*gAw_eta/b
                        e[12] += -3*c1*cs[col+3]*fAphix*gAphix - 6*c1*cs[col+2]*fAw_xi*gAw/a

                # Calculating membrane stress components
                Nxx = Nxx0
                Nyy = Nyy0
                Nxy = Nxy0
                for i in range(NE):
                    Nxx += F[0*NE + i]*e[i]
                    Nyy += F[1*NE + i]*e[i]
                    Nxy += F[2*NE + i]*e[i]

                # kG
                c = -1
                for i in range(m):
                    fAv = ftab[2, i]
                    fAw = ftab[4, i]
                    fAw_xi = ftab[5, i]

                    for k in range(m):
                        fBv = ftab[2, k]
                        fBw = ftab[4, k]
                        fBw_xi = ftab[5, k]

                        for j in range(n):
                            gAv = gtab[2, j]
                            gAw = gtab[4, j]
                            gAw_eta = gtab[5, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBv = gtab[2, l]
                                gBw = gtab[4, l]
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
    cdef double r
    cdef double a, b, intx, inty, c1
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double x1phix, x1phixr, x2phix, x2phixr
    cdef double x1phiy, x1phiyr, x2phiy, x2phiyr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double y1phix, y1phixr, y2phix, y2phixr
    cdef double y1phiy, y1phiyr, y2phiy, y2phiyr

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kMr, kMc
    cdef double [::1] kMv

    cdef double fAu, fAv, fAw, fAw_xi, fAphix
    cdef double fAphiy
    cdef double fBu, fBv, fBw, fBw_xi, fBphix
    cdef double fBphiy
    cdef double gAu, gAv, gAw, gAw_eta, gAphix
    cdef double gAphiy
    cdef double gBu, gBv, gBw, gBw_eta, gBphix
    cdef double gBphiy
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # moments of the density through the thickness, J0, J1, J2, J3, J4, J6, h, as a 3-D
    # matrix [nx, ny, 7] with one set for each of the nx * ny integration
    # points, or as a 1-D array [7] for the whole integration domain
    cdef double J0, J1, J2, J3, J4, J6, h, d
    cdef double [:, :, ::1] I_nxny

    cdef int one_set_each_point = 0

    I_input = np.asarray(I_input, dtype=DOUBLE)
    if I_input.shape == (nx, ny, 7):
        I_nxny = np.ascontiguousarray(I_input)
        one_set_each_point = 1
    elif I_input.shape == (7, ):
        # creating dummy 3-D array that is not used
        I_nxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        # using constant moments for all the integration domain
        J0, J1, J2, J3, J4, J6, h = I_input
    else:
        raise ValueError('Invalid shape for I_input, it must be (nx, ny, 7) or (7, )')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    d = shell.offset
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    x1phix = shell.x1phix; x1phixr = shell.x1phixr; x2phix = shell.x2phix; x2phixr = shell.x2phixr
    x1phiy = shell.x1phiy; x1phiyr = shell.x1phiyr; x2phiy = shell.x2phiy; x2phiyr = shell.x2phiyr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    y1phix = shell.y1phix; y1phixr = shell.y1phixr; y2phix = shell.y2phix; y2phixr = shell.y2phixr
    y1phiy = shell.y1phiy; y1phiyr = shell.y1phiyr; y2phiy = shell.y2phiy; y2phiyr = shell.y2phiyr

    fdim = 17*m*m*n*n

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

    ftab = np.zeros((6, m), dtype=DOUBLE)
    gtab = np.zeros((6, n), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

                if one_set_each_point == 1:
                    J0 = I_nxny[ptx, pty, 0]
                    J1 = I_nxny[ptx, pty, 1]
                    J2 = I_nxny[ptx, pty, 2]
                    J3 = I_nxny[ptx, pty, 3]
                    J4 = I_nxny[ptx, pty, 4]
                    J6 = I_nxny[ptx, pty, 5]
                    h = I_nxny[ptx, pty, 6]
                c1 = 4./(3.*h*h)

                # kM
                c = -1
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[2, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[3, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[4, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[5, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[2, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[3, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[4, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[5, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                for i in range(m):
                    fAu = ftab[0, i]
                    fAv = ftab[1, i]
                    fAw = ftab[2, i]
                    fAw_xi = ftab[3, i]
                    fAphix = ftab[4, i]
                    fAphiy = ftab[5, i]

                    for k in range(m):
                        fBu = ftab[0, k]
                        fBv = ftab[1, k]
                        fBw = ftab[2, k]
                        fBw_xi = ftab[3, k]
                        fBphix = ftab[4, k]
                        fBphiy = ftab[5, k]

                        for j in range(n):
                            gAu = gtab[0, j]
                            gAv = gtab[1, j]
                            gAw = gtab[2, j]
                            gAw_eta = gtab[3, j]
                            gAphix = gtab[4, j]
                            gAphiy = gtab[5, j]

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
                                gBphix = gtab[4, l]
                                gBphiy = gtab[5, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( J0*fAu*fBu*gAu*gBu )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( -2*J0*c1*(d*d*d)*fAu*fBw_xi*gAu*gBw/a - 2*J3*c1*fAu*fBw_xi*gAu*gBw/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( -J0*c1*(d*d*d)*fAu*fBphix*gAu*gBphix + J0*d*fAu*fBphix*gAu*gBphix + J1*fAu*fBphix*gAu*gBphix - J3*c1*fAu*fBphix*gAu*gBphix )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( J0*(d*d)*fAv*fBv*gAv*gBv/(r*r) + 2*J0*d*fAv*fBv*gAv*gBv/r + J0*fAv*fBv*gAv*gBv + 2*J1*d*fAv*fBv*gAv*gBv/(r*r) + 2*J1*fAv*fBv*gAv*gBv/r + J2*fAv*fBv*gAv*gBv/(r*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( -2*J0*c1*(d*d*d*d)*fAv*fBw*gAv*gBw_eta/(b*r) - 2*J0*c1*(d*d*d)*fAv*fBw*gAv*gBw_eta/b - 2*J1*c1*(d*d*d)*fAv*fBw*gAv*gBw_eta/(b*r) - 2*J3*c1*d*fAv*fBw*gAv*gBw_eta/(b*r) - 2*J3*c1*fAv*fBw*gAv*gBw_eta/b - 2*J4*c1*fAv*fBw*gAv*gBw_eta/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( -J0*c1*(d*d*d*d)*fAv*fBphiy*gAv*gBphiy/r - J0*c1*(d*d*d)*fAv*fBphiy*gAv*gBphiy + J0*(d*d)*fAv*fBphiy*gAv*gBphiy/r + J0*d*fAv*fBphiy*gAv*gBphiy - J1*c1*(d*d*d)*fAv*fBphiy*gAv*gBphiy/r + 2*J1*d*fAv*fBphiy*gAv*gBphiy/r + J1*fAv*fBphiy*gAv*gBphiy + J2*fAv*fBphiy*gAv*gBphiy/r - J3*c1*d*fAv*fBphiy*gAv*gBphiy/r - J3*c1*fAv*fBphiy*gAv*gBphiy - J4*c1*fAv*fBphiy*gAv*gBphiy/r )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( -2*J0*c1*(d*d*d)*fAw_xi*fBu*gAw*gBu/a - 2*J3*c1*fAw_xi*fBu*gAw*gBu/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( -2*J0*c1*(d*d*d*d)*fAw*fBv*gAw_eta*gBv/(b*r) - 2*J0*c1*(d*d*d)*fAw*fBv*gAw_eta*gBv/b - 2*J1*c1*(d*d*d)*fAw*fBv*gAw_eta*gBv/(b*r) - 2*J3*c1*d*fAw*fBv*gAw_eta*gBv/(b*r) - 2*J3*c1*fAw*fBv*gAw_eta*gBv/b - 2*J4*c1*fAw*fBv*gAw_eta*gBv/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( J0*fAw*fBw*gAw*gBw + 4*J0*(c1*c1)*(d*d*d*d*d*d)*fAw*fBw*gAw_eta*gBw_eta/(b*b) + 4*J0*(c1*c1)*(d*d*d*d*d*d)*fAw_xi*fBw_xi*gAw*gBw/(a*a) + 8*J3*(c1*c1)*(d*d*d)*fAw*fBw*gAw_eta*gBw_eta/(b*b) + 8*J3*(c1*c1)*(d*d*d)*fAw_xi*fBw_xi*gAw*gBw/(a*a) + 4*J6*(c1*c1)*fAw*fBw*gAw_eta*gBw_eta/(b*b) + 4*J6*(c1*c1)*fAw_xi*fBw_xi*gAw*gBw/(a*a) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( 2*J0*(c1*c1)*(d*d*d*d*d*d)*fAw_xi*fBphix*gAw*gBphix/a - 2*J0*c1*(d*d*d*d)*fAw_xi*fBphix*gAw*gBphix/a - 2*J1*c1*(d*d*d)*fAw_xi*fBphix*gAw*gBphix/a + 4*J3*(c1*c1)*(d*d*d)*fAw_xi*fBphix*gAw*gBphix/a - 2*J3*c1*d*fAw_xi*fBphix*gAw*gBphix/a - 2*J4*c1*fAw_xi*fBphix*gAw*gBphix/a + 2*J6*(c1*c1)*fAw_xi*fBphix*gAw*gBphix/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( 2*J0*(c1*c1)*(d*d*d*d*d*d)*fAw*fBphiy*gAw_eta*gBphiy/b - 2*J0*c1*(d*d*d*d)*fAw*fBphiy*gAw_eta*gBphiy/b - 2*J1*c1*(d*d*d)*fAw*fBphiy*gAw_eta*gBphiy/b + 4*J3*(c1*c1)*(d*d*d)*fAw*fBphiy*gAw_eta*gBphiy/b - 2*J3*c1*d*fAw*fBphiy*gAw_eta*gBphiy/b - 2*J4*c1*fAw*fBphiy*gAw_eta*gBphiy/b + 2*J6*(c1*c1)*fAw*fBphiy*gAw_eta*gBphiy/b )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( -J0*c1*(d*d*d)*fAphix*fBu*gAphix*gBu + J0*d*fAphix*fBu*gAphix*gBu + J1*fAphix*fBu*gAphix*gBu - J3*c1*fAphix*fBu*gAphix*gBu )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( 2*J0*(c1*c1)*(d*d*d*d*d*d)*fAphix*fBw_xi*gAphix*gBw/a - 2*J0*c1*(d*d*d*d)*fAphix*fBw_xi*gAphix*gBw/a - 2*J1*c1*(d*d*d)*fAphix*fBw_xi*gAphix*gBw/a + 4*J3*(c1*c1)*(d*d*d)*fAphix*fBw_xi*gAphix*gBw/a - 2*J3*c1*d*fAphix*fBw_xi*gAphix*gBw/a - 2*J4*c1*fAphix*fBw_xi*gAphix*gBw/a + 2*J6*(c1*c1)*fAphix*fBw_xi*gAphix*gBw/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( J0*(c1*c1)*(d*d*d*d*d*d)*fAphix*fBphix*gAphix*gBphix - 2*J0*c1*(d*d*d*d)*fAphix*fBphix*gAphix*gBphix + J0*(d*d)*fAphix*fBphix*gAphix*gBphix - 2*J1*c1*(d*d*d)*fAphix*fBphix*gAphix*gBphix + 2*J1*d*fAphix*fBphix*gAphix*gBphix + J2*fAphix*fBphix*gAphix*gBphix + 2*J3*(c1*c1)*(d*d*d)*fAphix*fBphix*gAphix*gBphix - 2*J3*c1*d*fAphix*fBphix*gAphix*gBphix - 2*J4*c1*fAphix*fBphix*gAphix*gBphix + J6*(c1*c1)*fAphix*fBphix*gAphix*gBphix )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( -J0*c1*(d*d*d*d)*fAphiy*fBv*gAphiy*gBv/r - J0*c1*(d*d*d)*fAphiy*fBv*gAphiy*gBv + J0*(d*d)*fAphiy*fBv*gAphiy*gBv/r + J0*d*fAphiy*fBv*gAphiy*gBv - J1*c1*(d*d*d)*fAphiy*fBv*gAphiy*gBv/r + 2*J1*d*fAphiy*fBv*gAphiy*gBv/r + J1*fAphiy*fBv*gAphiy*gBv + J2*fAphiy*fBv*gAphiy*gBv/r - J3*c1*d*fAphiy*fBv*gAphiy*gBv/r - J3*c1*fAphiy*fBv*gAphiy*gBv - J4*c1*fAphiy*fBv*gAphiy*gBv/r )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( 2*J0*(c1*c1)*(d*d*d*d*d*d)*fAphiy*fBw*gAphiy*gBw_eta/b - 2*J0*c1*(d*d*d*d)*fAphiy*fBw*gAphiy*gBw_eta/b - 2*J1*c1*(d*d*d)*fAphiy*fBw*gAphiy*gBw_eta/b + 4*J3*(c1*c1)*(d*d*d)*fAphiy*fBw*gAphiy*gBw_eta/b - 2*J3*c1*d*fAphiy*fBw*gAphiy*gBw_eta/b - 2*J4*c1*fAphiy*fBw*gAphiy*gBw_eta/b + 2*J6*(c1*c1)*fAphiy*fBw*gAphiy*gBw_eta/b )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( J0*(c1*c1)*(d*d*d*d*d*d)*fAphiy*fBphiy*gAphiy*gBphiy - 2*J0*c1*(d*d*d*d)*fAphiy*fBphiy*gAphiy*gBphiy + J0*(d*d)*fAphiy*fBphiy*gAphiy*gBphiy - 2*J1*c1*(d*d*d)*fAphiy*fBphiy*gAphiy*gBphiy + 2*J1*d*fAphiy*fBphiy*gAphiy*gBphiy + J2*fAphiy*fBphiy*gAphiy*gBphiy + 2*J3*(c1*c1)*(d*d*d)*fAphiy*fBphiy*gAphiy*gBphiy - 2*J3*c1*d*fAphiy*fBphiy*gAphiy*gBphiy - 2*J4*c1*fAphiy*fBphiy*gAphiy*gBphiy + J6*(c1*c1)*fAphiy*fBphiy*gAphiy*gBphiy )

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


def fkAx_num(object shell, int size, int row0, int col0, int nx, int ny):
    cdef double [:, ::1] ftab, gtab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double r
    cdef double a, b, beta, intx, inty, gamma
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double x1phix, x1phixr, x2phix, x2phixr
    cdef double x1phiy, x1phiyr, x2phiy, x2phiyr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double y1phix, y1phixr, y2phix, y2phixr
    cdef double y1phiy, y1phiyr, y2phiy, y2phiyr

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kAr, kAc
    cdef double [::1] kAv

    cdef double fAw, fAw_xi
    cdef double fBw
    cdef double gAw
    cdef double gBw
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

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
    beta = shell.beta
    gamma = shell.gamma
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    x1phix = shell.x1phix; x1phixr = shell.x1phixr; x2phix = shell.x2phix; x2phixr = shell.x2phixr
    x1phiy = shell.x1phiy; x1phiyr = shell.x1phiyr; x2phiy = shell.x2phiy; x2phiyr = shell.x2phiyr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    y1phix = shell.y1phix; y1phixr = shell.y1phixr; y2phix = shell.y2phix; y2phixr = shell.y2phixr
    y1phiy = shell.y1phiy; y1phiyr = shell.y1phiyr; y2phiy = shell.y2phiy; y2phiyr = shell.y2phiyr

    fdim = 1*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kAr = np.zeros((fdim,), dtype=INT)
    kAc = np.zeros((fdim,), dtype=INT)
    kAv = np.zeros((fdim,), dtype=DOUBLE)

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

    ftab = np.zeros((2, m), dtype=DOUBLE)
    gtab = np.zeros((1, n), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

                # kA
                c = -1
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[1, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1w, y1wr, y2w, y2wr)

                for i in range(m):
                    fAw = ftab[0, i]
                    fAw_xi = ftab[1, i]

                    for k in range(m):
                        fBw = ftab[0, k]

                        for j in range(n):
                            gAw = gtab[0, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                gBw = gtab[0, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kAr[c] = row+2
                                    kAc[c] = col+2
                                kAv[c] += weight*(intx*inty/4)*( -fAw*fBw*gAw*gBw*gamma - 2*beta*fAw_xi*fBw*gAw*gBw/a )

    kA = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kA


def fkAy_num(object shell, int size, int row0, int col0, int nx, int ny):
    cdef double [:, ::1] ftab, gtab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double r
    cdef double a, b, beta, intx, inty
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double x1phix, x1phixr, x2phix, x2phixr
    cdef double x1phiy, x1phiyr, x2phiy, x2phiyr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double y1phix, y1phixr, y2phix, y2phixr
    cdef double y1phiy, y1phiyr, y2phiy, y2phiyr

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kAr, kAc
    cdef double [::1] kAv

    cdef double fAw
    cdef double fBw
    cdef double gAw_eta
    cdef double gBw
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

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
    beta = shell.beta
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    x1phix = shell.x1phix; x1phixr = shell.x1phixr; x2phix = shell.x2phix; x2phixr = shell.x2phixr
    x1phiy = shell.x1phiy; x1phiyr = shell.x1phiyr; x2phiy = shell.x2phiy; x2phiyr = shell.x2phiyr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    y1phix = shell.y1phix; y1phixr = shell.y1phixr; y2phix = shell.y2phix; y2phixr = shell.y2phixr
    y1phiy = shell.y1phiy; y1phiyr = shell.y1phiyr; y2phiy = shell.y2phiy; y2phiyr = shell.y2phiyr

    fdim = 1*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kAr = np.zeros((fdim,), dtype=INT)
    kAc = np.zeros((fdim,), dtype=INT)
    kAv = np.zeros((fdim,), dtype=DOUBLE)

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

    ftab = np.zeros((1, m), dtype=DOUBLE)
    gtab = np.zeros((2, n), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

                # kA
                c = -1
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[1, j] = f(j, eta, y1w, y1wr, y2w, y2wr)

                for i in range(m):
                    fAw = ftab[0, i]

                    for k in range(m):
                        fBw = ftab[0, k]

                        for j in range(n):
                            gAw_eta = gtab[0, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                gBw = gtab[1, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kAr[c] = row+2
                                    kAc[c] = col+2
                                kAv[c] += weight*(intx*inty/4)*( -2*beta*fAw*fBw*gAw_eta*gBw/b )

    kA = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kA


def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    cdef double [:, ::1] gtab, ftab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double r
    cdef double a, b, intx, inty, h, c1
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double x1phix, x1phixr, x2phix, x2phixr
    cdef double x1phiy, x1phiyr, x2phiy, x2phiyr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double y1phix, y1phixr, y2phix, y2phixr
    cdef double y1phiy, y1phiyr, y2phiy, y2phiyr

    cdef int i, j, c, col, ptx, pty
    cdef double e[13]
    cdef double s[13]
    cdef double Nxx, Nyy, Nxy, Mxx, Myy
    cdef double Mxy, Pxx, Pyy, Pxy, Qy
    cdef double Qx, Ry, Rx

    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double bx, by

    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw
    cdef double fAw_xi, fAw_xixi, fAphix, fAphix_xi, fAphiy
    cdef double fAphiy_xi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw
    cdef double gAw_eta, gAw_etaeta, gAphix, gAphix_eta, gAphiy
    cdef double gAphiy_eta

    cdef double [::1] xis, etas, weights_xi, weights_eta, fint

    # F as 4-D matrix, must be [nx, ny, 13, 13], when there is one
    # constitutive matrix [13, 13] for each of the nx * ny integration points
    cdef double F[169]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, NE, NE):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (NE, NE):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        for i in range(NE):
            for j in range(NE):
                F[i*NE + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    #NOTE the traction-free faces are assumed at z = +-h/2
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    x1phix = shell.x1phix; x1phixr = shell.x1phixr; x2phix = shell.x2phix; x2phixr = shell.x2phixr
    x1phiy = shell.x1phiy; x1phiyr = shell.x1phiyr; x2phiy = shell.x2phiy; x2phiyr = shell.x2phiyr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    y1phix = shell.y1phix; y1phixr = shell.y1phixr; y2phix = shell.y2phix; y2phixr = shell.y2phixr
    y1phiy = shell.y1phiy; y1phiyr = shell.y1phiyr; y2phiy = shell.y2phiy; y2phiyr = shell.y2phiyr

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

    gtab = np.zeros((11, n), dtype=DOUBLE)
    ftab = np.zeros((11, m), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

                if one_F_each_point == 1:
                    for i in range(NE):
                        for j in range(NE):
                            #TODO could assume symmetry
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

                bx = 0
                by = 0
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[1, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[4, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[5, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[9, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[10, j] = fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[1, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[4, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[5, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[9, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                for j in range(n):
                    gAw = gtab[0, j]
                    gAw_eta = gtab[1, j]
                    gAv = gtab[2, j]
                    for i in range(m):
                        fAw = ftab[0, i]
                        fAw_xi = ftab[1, i]
                        fAv = ftab[2, i]

                        col = col0 + DOF*(j*m + i)

                        bx -= (2/a)*cs[col+2]*fAw_xi*gAw
                        by -= (2/b)*cs[col+2]*fAw*gAw_eta
                        by += cs[col+1]*fAv*gAv/r

                # current generalized strain state
                for i in range(NE):
                    e[i] = 0.
                for j in range(n):
                    gAu = gtab[3, j]
                    gAu_eta = gtab[4, j]
                    gAv = gtab[2, j]
                    gAv_eta = gtab[5, j]
                    gAw = gtab[0, j]
                    gAw_eta = gtab[1, j]
                    gAw_etaeta = gtab[6, j]
                    gAphix = gtab[7, j]
                    gAphix_eta = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_eta = gtab[10, j]

                    for i in range(m):
                        fAu = ftab[3, i]
                        fAu_xi = ftab[4, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[5, i]
                        fAw = ftab[0, i]
                        fAw_xi = ftab[1, i]
                        fAw_xixi = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_xi = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_xi = ftab[10, i]

                        col = col0 + DOF*(j*m + i)

                        e[0] += 2*cs[col+0]*fAu_xi*gAu/a
                        e[1] += cs[col+2]*fAw*gAw/r + 2*cs[col+1]*fAv*gAv_eta/b
                        e[2] += 2*cs[col+0]*fAu*gAu_eta/b + 2*cs[col+1]*fAv_xi*gAv/a
                        e[3] += 2*cs[col+3]*fAphix_xi*gAphix/a
                        e[4] += 2*cs[col+1]*fAv*gAv_eta/(b*r) + 2*cs[col+4]*fAphiy*gAphiy_eta/b
                        e[5] += -cs[col+0]*fAu*gAu_eta/(b*r) + 2*cs[col+3]*fAphix*gAphix_eta/b + 3*cs[col+1]*fAv_xi*gAv/(a*r) + 2*cs[col+4]*fAphiy_xi*gAphiy/a
                        e[6] += -2*c1*cs[col+3]*fAphix_xi*gAphix/a - 4*c1*cs[col+2]*fAw_xixi*gAw/(a*a)
                        e[7] += -2*c1*cs[col+4]*fAphiy*gAphiy_eta/b - 4*c1*cs[col+2]*fAw*gAw_etaeta/(b*b)
                        e[8] += -2*c1*cs[col+3]*fAphix*gAphix_eta/b - 2*c1*cs[col+4]*fAphiy_xi*gAphiy/a - 8*c1*cs[col+2]*fAw_xi*gAw_eta/(a*b)
                        e[9] += cs[col+4]*fAphiy*gAphiy + 2*cs[col+2]*fAw*gAw_eta/b
                        e[10] += cs[col+3]*fAphix*gAphix + 2*cs[col+2]*fAw_xi*gAw/a
                        e[11] += -3*c1*cs[col+4]*fAphiy*gAphiy - 6*c1*cs[col+2]*fAw*gAw_eta/b
                        e[12] += -3*c1*cs[col+3]*fAphix*gAphix - 6*c1*cs[col+2]*fAw_xi*gAw/a

                # nonlinear strain eps_NL = {bx^2/2, by^2/2, bx*by}
                e[0] += 0.5*bx*bx
                e[1] += 0.5*by*by
                e[2] += bx*by

                # current generalized stress state
                for i in range(NE):
                    s[i] = 0.
                    for j in range(NE):
                        s[i] += F[i*NE + j]*e[j]
                Nxx = s[0]
                Nyy = s[1]
                Nxy = s[2]
                Mxx = s[3]
                Myy = s[4]
                Mxy = s[5]
                Pxx = s[6]
                Pyy = s[7]
                Pxy = s[8]
                Qy = s[9]
                Qx = s[10]
                Ry = s[11]
                Rx = s[12]

                for j in range(n):
                    gAu = gtab[3, j]
                    gAu_eta = gtab[4, j]
                    gAv = gtab[2, j]
                    gAv_eta = gtab[5, j]
                    gAw = gtab[0, j]
                    gAw_eta = gtab[1, j]
                    gAw_etaeta = gtab[6, j]
                    gAphix = gtab[7, j]
                    gAphix_eta = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_eta = gtab[10, j]
                    for i in range(m):
                        fAu = ftab[3, i]
                        fAu_xi = ftab[4, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[5, i]
                        fAw = ftab[0, i]
                        fAw_xi = ftab[1, i]
                        fAw_xixi = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_xi = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_xi = ftab[10, i]

                        col = col0 + DOF*(j*m + i)

                        fint[col+0] += weight*(intx*inty/4)*( -Mxy*fAu*gAu_eta/(b*r) + 2*Nxx*fAu_xi*gAu/a + 2*Nxy*fAu*gAu_eta/b )
                        fint[col+1] += weight*(intx*inty/4)*( 3*Mxy*fAv_xi*gAv/(a*r) + 2*Myy*fAv*gAv_eta/(b*r) + Nxy*gAv*(a*bx*fAv + 2*fAv_xi*r)/(a*r) + Nyy*fAv*(b*by*gAv + 2*gAv_eta*r)/(b*r) )
                        fint[col+2] += weight*(intx*inty/4)*( -2*Nxx*bx*fAw_xi*gAw/a - 2*Nxy*(a*bx*fAw*gAw_eta + b*by*fAw_xi*gAw)/(a*b) - Nyy*fAw*(-b*gAw + 2*by*gAw_eta*r)/(b*r) - 4*Pxx*c1*fAw_xixi*gAw/(a*a) - 8*Pxy*c1*fAw_xi*gAw_eta/(a*b) - 4*Pyy*c1*fAw*gAw_etaeta/(b*b) + 2*Qx*fAw_xi*gAw/a + 2*Qy*fAw*gAw_eta/b - 6*Rx*c1*fAw_xi*gAw/a - 6*Ry*c1*fAw*gAw_eta/b )
                        fint[col+3] += weight*(intx*inty/4)*( 2*Mxx*fAphix_xi*gAphix/a + 2*Mxy*fAphix*gAphix_eta/b - 2*Pxx*c1*fAphix_xi*gAphix/a - 2*Pxy*c1*fAphix*gAphix_eta/b + Qx*fAphix*gAphix - 3*Rx*c1*fAphix*gAphix )
                        fint[col+4] += weight*(intx*inty/4)*( 2*Mxy*fAphiy_xi*gAphiy/a + 2*Myy*fAphiy*gAphiy_eta/b - 2*Pxy*c1*fAphiy_xi*gAphiy/a - 2*Pyy*c1*fAphiy*gAphiy_eta/b + Qy*fAphiy*gAphiy - 3*Ry*c1*fAphiy*gAphiy )

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
        Constitutive matrix ``(13, 13)``, or one for each integration
        point ``(nx, ny, 13, 13)``.
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
    cdef double a, b, intx, inty, r, h, c1
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double x1phix, x1phixr, x2phix, x2phixr
    cdef double x1phiy, x1phiyr, x2phiy, x2phiyr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double y1phix, y1phixr, y2phix, y2phixr
    cdef double y1phiy, y1phiyr, y2phiy, y2phiyr

    cdef int i, j, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double bx, by, dbx, dby, Nxx, Nyy, Nxy
    cdef double e[13]
    cdef double de[13]
    cdef double s[13]
    cdef double ds[13]
    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw, fAw_xi
    cdef double fAw_xixi, fAphix, fAphix_xi, fAphiy, fAphiy_xi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw, gAw_eta
    cdef double gAw_etaeta, gAphix, gAphix_eta, gAphiy, gAphiy_eta

    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta, kTv

    # F as 4-D matrix, must be [nx, ny, 13, 13], when there is one
    # constitutive matrix [13, 13] for each of the nx * ny integration points
    cdef double F[169]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    if cs.shape[0] != size or vs.shape[0] != size:
        raise ValueError('cs and vs must have the size %d' % size)
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 13, 13):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (13, 13):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        for i in range(13):
            for j in range(13):
                F[i*13 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    #NOTE the traction-free faces are assumed at z = +-h/2
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    x1phix = shell.x1phix; x1phixr = shell.x1phixr; x2phix = shell.x2phix; x2phixr = shell.x2phixr
    x1phiy = shell.x1phiy; x1phiyr = shell.x1phiyr; x2phiy = shell.x2phiy; x2phiyr = shell.x2phiyr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    y1phix = shell.y1phix; y1phixr = shell.y1phixr; y2phix = shell.y2phix; y2phixr = shell.y2phixr
    y1phiy = shell.y1phiy; y1phiyr = shell.y1phiyr; y2phiy = shell.y2phiy; y2phiyr = shell.y2phiyr

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kTv = np.zeros(size, dtype=DOUBLE)
    # basis functions at the integration point
    ftab = np.zeros((11, m), dtype=DOUBLE)
    gtab = np.zeros((11, n), dtype=DOUBLE)

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
                    for i in range(13):
                        for j in range(13):
                            F[i*13 + j] = Fnxny[ptx, pty, i, j]

                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[9, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[9, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[10, j] = fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

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
                for i in range(13):
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
                    gAphix = gtab[7, j]
                    gAphix_eta = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_eta = gtab[10, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_xi = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        fAw_xixi = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_xi = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_xi = ftab[10, i]
                        col = col0 + DOF*(j*m + i)
                        e[0] += 2*fAu_xi*gAu/a*cs[col+0] - bx*fAw_xi*gAw/a*cs[col+2]
                        e[1] += (0.5*by*fAv*gAv/r + 2*fAv*gAv_eta/b)*cs[col+1] + (fAw*gAw/r - by*fAw*gAw_eta/b)*cs[col+2]
                        e[2] += 2*fAu*gAu_eta/b*cs[col+0] + (0.5*bx*fAv*gAv/r + 2*fAv_xi*gAv/a)*cs[col+1] + (-bx*fAw*gAw_eta/b - by*fAw_xi*gAw/a)*cs[col+2]
                        e[3] += 2*fAphix_xi*gAphix/a*cs[col+3]
                        e[4] += 2*fAv*gAv_eta/(b*r)*cs[col+1] + 2*fAphiy*gAphiy_eta/b*cs[col+4]
                        e[5] += -fAu*gAu_eta/(b*r)*cs[col+0] + 3*fAv_xi*gAv/(a*r)*cs[col+1] + 2*fAphix*gAphix_eta/b*cs[col+3] + 2*fAphiy_xi*gAphiy/a*cs[col+4]
                        e[6] += -4*c1*fAw_xixi*gAw/(a*a)*cs[col+2] - 2*c1*fAphix_xi*gAphix/a*cs[col+3]
                        e[7] += -4*c1*fAw*gAw_etaeta/(b*b)*cs[col+2] - 2*c1*fAphiy*gAphiy_eta/b*cs[col+4]
                        e[8] += -8*c1*fAw_xi*gAw_eta/(a*b)*cs[col+2] - 2*c1*fAphix*gAphix_eta/b*cs[col+3] - 2*c1*fAphiy_xi*gAphiy/a*cs[col+4]
                        e[9] += 2*fAw*gAw_eta/b*cs[col+2] + fAphiy*gAphiy*cs[col+4]
                        e[10] += 2*fAw_xi*gAw/a*cs[col+2] + fAphix*gAphix*cs[col+3]
                        e[11] += -6*c1*fAw*gAw_eta/b*cs[col+2] - 3*c1*fAphiy*gAphiy*cs[col+4]
                        e[12] += -6*c1*fAw_xi*gAw/a*cs[col+2] - 3*c1*fAphix*gAphix*cs[col+3]
                        de[0] += 2*fAu_xi*gAu/a*vs[col+0] - 2*bx*fAw_xi*gAw/a*vs[col+2]
                        de[1] += (by*fAv*gAv/r + 2*fAv*gAv_eta/b)*vs[col+1] + (fAw*gAw/r - 2*by*fAw*gAw_eta/b)*vs[col+2]
                        de[2] += 2*fAu*gAu_eta/b*vs[col+0] + (bx*fAv*gAv/r + 2*fAv_xi*gAv/a)*vs[col+1] + (-2*bx*fAw*gAw_eta/b - 2*by*fAw_xi*gAw/a)*vs[col+2]
                        de[3] += 2*fAphix_xi*gAphix/a*vs[col+3]
                        de[4] += 2*fAv*gAv_eta/(b*r)*vs[col+1] + 2*fAphiy*gAphiy_eta/b*vs[col+4]
                        de[5] += -fAu*gAu_eta/(b*r)*vs[col+0] + 3*fAv_xi*gAv/(a*r)*vs[col+1] + 2*fAphix*gAphix_eta/b*vs[col+3] + 2*fAphiy_xi*gAphiy/a*vs[col+4]
                        de[6] += -4*c1*fAw_xixi*gAw/(a*a)*vs[col+2] - 2*c1*fAphix_xi*gAphix/a*vs[col+3]
                        de[7] += -4*c1*fAw*gAw_etaeta/(b*b)*vs[col+2] - 2*c1*fAphiy*gAphiy_eta/b*vs[col+4]
                        de[8] += -8*c1*fAw_xi*gAw_eta/(a*b)*vs[col+2] - 2*c1*fAphix*gAphix_eta/b*vs[col+3] - 2*c1*fAphiy_xi*gAphiy/a*vs[col+4]
                        de[9] += 2*fAw*gAw_eta/b*vs[col+2] + fAphiy*gAphiy*vs[col+4]
                        de[10] += 2*fAw_xi*gAw/a*vs[col+2] + fAphix*gAphix*vs[col+3]
                        de[11] += -6*c1*fAw*gAw_eta/b*vs[col+2] - 3*c1*fAphiy*gAphiy*vs[col+4]
                        de[12] += -6*c1*fAw_xi*gAw/a*vs[col+2] - 3*c1*fAphix*gAphix*vs[col+3]

                # stresses
                for i in range(13):
                    s[i] = 0.
                    ds[i] = 0.
                    for j in range(13):
                        s[i] += F[i*13 + j]*e[j]
                        ds[i] += F[i*13 + j]*de[j]
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
                    gAphix = gtab[7, j]
                    gAphix_eta = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_eta = gtab[10, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_xi = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        fAw_xixi = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_xi = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_xi = ftab[10, i]
                        col = col0 + DOF*(j*m + i)
                        kTv[col+0] += weight*(2*ds[2]*fAu*gAu_eta/b - ds[5]*fAu*gAu_eta/(b*r) + 2*ds[0]*fAu_xi*gAu/a)
                        kTv[col+1] += weight*(Nxy*dbx*fAv*gAv/r + Nyy*dby*fAv*gAv/r + ds[1]*(by*fAv*gAv/r + 2*fAv*gAv_eta/b) + ds[2]*(bx*fAv*gAv/r + 2*fAv_xi*gAv/a) + 2*ds[4]*fAv*gAv_eta/(b*r) + 3*ds[5]*fAv_xi*gAv/(a*r))
                        kTv[col+2] += weight*(-2*Nxx*dbx*fAw_xi*gAw/a + Nxy*(-2*dbx*fAw*gAw_eta/b - 2*dby*fAw_xi*gAw/a) - 2*Nyy*dby*fAw*gAw_eta/b + ds[1]*(fAw*gAw/r - 2*by*fAw*gAw_eta/b) + ds[2]*(-2*bx*fAw*gAw_eta/b - 2*by*fAw_xi*gAw/a) - 6*c1*ds[11]*fAw*gAw_eta/b + 2*ds[9]*fAw*gAw_eta/b - 4*c1*ds[7]*fAw*gAw_etaeta/(b*b) - 2*bx*ds[0]*fAw_xi*gAw/a - 6*c1*ds[12]*fAw_xi*gAw/a + 2*ds[10]*fAw_xi*gAw/a - 8*c1*ds[8]*fAw_xi*gAw_eta/(a*b) - 4*c1*ds[6]*fAw_xixi*gAw/(a*a))
                        kTv[col+3] += weight*(-3*c1*ds[12]*fAphix*gAphix + ds[10]*fAphix*gAphix - 2*c1*ds[8]*fAphix*gAphix_eta/b + 2*ds[5]*fAphix*gAphix_eta/b - 2*c1*ds[6]*fAphix_xi*gAphix/a + 2*ds[3]*fAphix_xi*gAphix/a)
                        kTv[col+4] += weight*(-3*c1*ds[11]*fAphiy*gAphiy + ds[9]*fAphiy*gAphiy - 2*c1*ds[7]*fAphiy*gAphiy_eta/b + 2*ds[4]*fAphiy*gAphiy_eta/b - 2*c1*ds[8]*fAphiy_xi*gAphiy/a + 2*ds[5]*fAphiy_xi*gAphiy/a)

    return np.asarray(kTv)

# END tangent stiffness product
