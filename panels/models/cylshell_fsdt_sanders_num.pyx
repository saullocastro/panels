#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Cylindrical shells using the first-order shear deformation theory (FSDT)
with the Sanders-Koiter kinematics

Numerically integrated matrices. See the kinematic equations in
``theory/shells/fsdt_tsdt/fsdt_tsdt.py``, from where the integrands herein
have been generated. The degrees of freedom of each term of the
approximation are ``u, v, w, phix, phiy``, and the constitutive matrix
``Finput`` must be ``8 x 8``, with the rows and columns in the order of
the generalized strains::

    exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz

which is the order of the generalized stresses::

    Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qy, Qx

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
cdef int NE = 8

def fkC_num(double [::1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double [:, ::1] gtab, ftab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double r
    cdef double a, b, intx, inty
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
    cdef double B16, A22, A26, B22, B26
    cdef double A66, B66, D11, D12, D16
    cdef double D22, D26, D66, A44, A45
    cdef double A55

    cdef long [::1] kCr, kCc
    cdef double [::1] kCv

    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw
    cdef double fAw_xi, fAphix, fAphix_xi, fAphiy, fAphiy_xi
    cdef double fBu, fBu_xi, fBv, fBv_xi, fBw
    cdef double fBw_xi, fBphix, fBphix_xi, fBphiy, fBphiy_xi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw
    cdef double gAw_eta, gAphix, gAphix_eta, gAphiy, gAphiy_eta
    cdef double gBu, gBu_eta, gBv, gBv_eta, gBw
    cdef double gBw_eta, gBphix, gBphix_eta, gBphiy, gBphiy_eta
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double bx, by, NxxNL, NyyNL, NxyNL

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # F as 4-D matrix, must be [nx, ny, 8, 8], when there is one
    # constitutive matrix [8, 8] for each of the nx * ny integration points
    cdef double F[64]
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

    gtab = np.zeros((10, n), dtype=DOUBLE)
    ftab = np.zeros((10, m), dtype=DOUBLE)

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
                    gtab[6, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[7, j] = fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[9, j] = fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[1, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[4, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[5, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[7, i] = fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[9, i] = fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

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

                            bx += (2/a)*cs[col+2]*fAw_xi*gAw
                            by += (2/b)*cs[col+2]*fAw*gAw_eta
                            by -= cs[col+1]*fAv*gAv/r

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
                A22 = F[1*NE + 1]
                A26 = F[1*NE + 2]
                B22 = F[1*NE + 4]
                B26 = F[1*NE + 5]
                A66 = F[2*NE + 2]
                B66 = F[2*NE + 5]
                D11 = F[3*NE + 3]
                D12 = F[3*NE + 4]
                D16 = F[3*NE + 5]
                D22 = F[4*NE + 4]
                D26 = F[4*NE + 5]
                D66 = F[5*NE + 5]
                A44 = F[6*NE + 6]
                A45 = F[6*NE + 7]
                A55 = F[7*NE + 7]

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
                    fAphix = ftab[6, i]
                    fAphix_xi = ftab[7, i]
                    fAphiy = ftab[8, i]
                    fAphiy_xi = ftab[9, i]

                    for k in range(m):
                        fBu = ftab[3, k]
                        fBu_xi = ftab[4, k]
                        fBv = ftab[2, k]
                        fBv_xi = ftab[5, k]
                        fBw = ftab[0, k]
                        fBw_xi = ftab[1, k]
                        fBphix = ftab[6, k]
                        fBphix_xi = ftab[7, k]
                        fBphiy = ftab[8, k]
                        fBphiy_xi = ftab[9, k]

                        for j in range(n):
                            gAu = gtab[3, j]
                            gAu_eta = gtab[4, j]
                            gAv = gtab[2, j]
                            gAv_eta = gtab[5, j]
                            gAw = gtab[0, j]
                            gAw_eta = gtab[1, j]
                            gAphix = gtab[6, j]
                            gAphix_eta = gtab[7, j]
                            gAphiy = gtab[8, j]
                            gAphiy_eta = gtab[9, j]

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
                                gBphix = gtab[6, l]
                                gBphix_eta = gtab[7, l]
                                gBphiy = gtab[8, l]
                                gBphiy_eta = gtab[9, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAu_xi*fBu_xi*gAu*gBu/(a*a) + 4*A16*(fAu*fBu_xi*gAu_eta*gBu + fAu_xi*fBu*gAu*gBu_eta)/(a*b) + 4*A66*fAu*fBu*gAu_eta*gBu_eta/(b*b) - 2*B16*(fAu*fBu_xi*gAu_eta*gBu + fAu_xi*fBu*gAu*gBu_eta)/(a*b*r) - 4*B66*fAu*fBu*gAu_eta*gBu_eta/((b*b)*r) + D66*fAu*fBu*gAu_eta*gBu_eta/((b*b)*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 2*A12*fAu_xi*fBv*gAu*(-b*by*gBv + 2*gBv_eta*r)/(a*b*r) + 2*A16*fAu_xi*gAu*gBv*(-a*bx*fBv + 2*fBv_xi*r)/((a*a)*r) + 2*A26*fAu*fBv*gAu_eta*(-b*by*gBv + 2*gBv_eta*r)/((b*b)*r) + 2*A66*fAu*gAu_eta*gBv*(-a*bx*fBv + 2*fBv_xi*r)/(a*b*r) + 4*B12*fAu_xi*fBv*gAu*gBv_eta/(a*b*r) + 6*B16*fAu_xi*fBv_xi*gAu*gBv/((a*a)*r) + B26*fAu*fBv*gAu_eta*(b*by*gBv + 2*gBv_eta*r)/((b*b)*(r*r)) + B66*fAu*gAu_eta*gBv*(a*bx*fBv + 4*fBv_xi*r)/(a*b*(r*r)) - 2*D26*fAu*fBv*gAu_eta*gBv_eta/((b*b)*(r*r)) - 3*D66*fAu*fBv_xi*gAu_eta*gBv/(a*b*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*bx*fAu_xi*fBw_xi*gAu*gBw/(a*a) + 2*A12*fAu_xi*fBw*gAu*(b*gBw + 2*by*gBw_eta*r)/(a*b*r) + 4*A16*(a*bx*fAu*fBw_xi*gAu_eta*gBw + a*bx*fAu_xi*fBw*gAu*gBw_eta + b*by*fAu_xi*fBw_xi*gAu*gBw)/((a*a)*b) + 2*A26*fAu*fBw*gAu_eta*(b*gBw + 2*by*gBw_eta*r)/((b*b)*r) + 4*A66*fAu*gAu_eta*(a*bx*fBw*gBw_eta + b*by*fBw_xi*gBw)/(a*(b*b)) - 2*B16*bx*fAu*fBw_xi*gAu_eta*gBw/(a*b*r) - B26*fAu*fBw*gAu_eta*(b*gBw + 2*by*gBw_eta*r)/((b*b)*(r*r)) - 2*B66*fAu*gAu_eta*(a*bx*fBw*gBw_eta + b*by*fBw_xi*gBw)/(a*(b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 4*B11*fAu_xi*fBphix_xi*gAu*gBphix/(a*a) + 4*B16*(fAu*fBphix_xi*gAu_eta*gBphix + fAu_xi*fBphix*gAu*gBphix_eta)/(a*b) + 4*B66*fAu*fBphix*gAu_eta*gBphix_eta/(b*b) - 2*D16*fAu*fBphix_xi*gAu_eta*gBphix/(a*b*r) - 2*D66*fAu*fBphix*gAu_eta*gBphix_eta/((b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAu_xi*fBphiy*gAu*gBphiy_eta/(a*b) + 4*B16*fAu_xi*fBphiy_xi*gAu*gBphiy/(a*a) + 4*B26*fAu*fBphiy*gAu_eta*gBphiy_eta/(b*b) + 4*B66*fAu*fBphiy_xi*gAu_eta*gBphiy/(a*b) - 2*D26*fAu*fBphiy*gAu_eta*gBphiy_eta/((b*b)*r) - 2*D66*fAu*fBphiy_xi*gAu_eta*gBphiy/(a*b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 2*A12*fAv*fBu_xi*gBu*(-b*by*gAv + 2*gAv_eta*r)/(a*b*r) + 2*A16*fBu_xi*gAv*gBu*(-a*bx*fAv + 2*fAv_xi*r)/((a*a)*r) + 2*A26*fAv*fBu*gBu_eta*(-b*by*gAv + 2*gAv_eta*r)/((b*b)*r) + 2*A66*fBu*gAv*gBu_eta*(-a*bx*fAv + 2*fAv_xi*r)/(a*b*r) + 4*B12*fAv*fBu_xi*gAv_eta*gBu/(a*b*r) + 6*B16*fAv_xi*fBu_xi*gAv*gBu/((a*a)*r) + B26*fAv*fBu*gBu_eta*(b*by*gAv + 2*gAv_eta*r)/((b*b)*(r*r)) + B66*fBu*gAv*gBu_eta*(a*bx*fAv + 4*fAv_xi*r)/(a*b*(r*r)) - 2*D26*fAv*fBu*gAv_eta*gBu_eta/((b*b)*(r*r)) - 3*D66*fAv_xi*fBu*gAv*gBu_eta/(a*b*(r*r)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( A22*fAv*fBv*(-b*by*gAv + 2*gAv_eta*r)*(-b*by*gBv + 2*gBv_eta*r)/((b*b)*(r*r)) + 2*A26*(a*b*bx*by*fAv*fBv*gAv*gBv - a*bx*fAv*fBv*gAv*gBv_eta*r - a*bx*fAv*fBv*gAv_eta*gBv*r - b*by*fAv*fBv_xi*gAv*gBv*r - b*by*fAv_xi*fBv*gAv*gBv*r + 2*fAv*fBv_xi*gAv_eta*gBv*(r*r) + 2*fAv_xi*fBv*gAv*gBv_eta*(r*r))/(a*b*(r*r)) + A66*gAv*gBv*(-a*bx*fAv + 2*fAv_xi*r)*(-a*bx*fBv + 2*fBv_xi*r)/((a*a)*(r*r)) + 2*B22*fAv*fBv*(-b*by*gAv*gBv_eta - b*by*gAv_eta*gBv + 4*gAv_eta*gBv_eta*r)/((b*b)*(r*r)) + B26*(-2*a*bx*fAv*fBv*gAv*gBv_eta - 2*a*bx*fAv*fBv*gAv_eta*gBv - 3*b*by*fAv*fBv_xi*gAv*gBv - 3*b*by*fAv_xi*fBv*gAv*gBv + 10*fAv*fBv_xi*gAv_eta*gBv*r + 10*fAv_xi*fBv*gAv*gBv_eta*r)/(a*b*(r*r)) + 3*B66*gAv*gBv*(-a*bx*fAv*fBv_xi - a*bx*fAv_xi*fBv + 4*fAv_xi*fBv_xi*r)/((a*a)*(r*r)) + 4*D22*fAv*fBv*gAv_eta*gBv_eta/((b*b)*(r*r)) + 6*D26*(fAv*fBv_xi*gAv_eta*gBv + fAv_xi*fBv*gAv*gBv_eta)/(a*b*(r*r)) + 9*D66*fAv_xi*fBv_xi*gAv*gBv/((a*a)*(r*r)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( NyyNL*fAv*fBv*gAv*gBv/(r*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 2*A12*bx*fAv*fBw_xi*gBw*(-b*by*gAv + 2*gAv_eta*r)/(a*b*r) + 2*A16*bx*fBw_xi*gAv*gBw*(-a*bx*fAv + 2*fAv_xi*r)/((a*a)*r) + A22*fAv*fBw*(b*gBw + 2*by*gBw_eta*r)*(-b*by*gAv + 2*gAv_eta*r)/((b*b)*(r*r)) + A26*(-a*(b*b)*bx*fAv*fBw*gAv*gBw - 4*a*b*bx*by*fAv*fBw*gAv*gBw_eta*r + 4*a*bx*fAv*fBw*gAv_eta*gBw_eta*(r*r) - 2*(b*b)*(by*by)*fAv*fBw_xi*gAv*gBw*r + 2*(b*b)*fAv_xi*fBw*gAv*gBw*r + 4*b*by*fAv*fBw_xi*gAv_eta*gBw*(r*r) + 4*b*by*fAv_xi*fBw*gAv*gBw_eta*(r*r))/(a*(b*b)*(r*r)) + 2*A66*gAv*(-a*bx*fAv + 2*fAv_xi*r)*(a*bx*fBw*gBw_eta + b*by*fBw_xi*gBw)/((a*a)*b*r) + 4*B12*bx*fAv*fBw_xi*gAv_eta*gBw/(a*b*r) + 6*B16*bx*fAv_xi*fBw_xi*gAv*gBw/((a*a)*r) + 2*B22*fAv*fBw*gAv_eta*(b*gBw + 2*by*gBw_eta*r)/((b*b)*(r*r)) + B26*(4*a*bx*fAv*fBw*gAv_eta*gBw_eta*r + 3*(b*b)*fAv_xi*fBw*gAv*gBw + 4*b*by*fAv*fBw_xi*gAv_eta*gBw*r + 6*b*by*fAv_xi*fBw*gAv*gBw_eta*r)/(a*(b*b)*(r*r)) + 6*B66*fAv_xi*gAv*(a*bx*fBw*gBw_eta + b*by*fBw_xi*gBw)/((a*a)*b*r) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( -2*NxyNL*fAv*fBw_xi*gAv*gBw/(a*r) - 2*NyyNL*fAv*fBw*gAv*gBw_eta/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 2*B12*fAv*fBphix_xi*gBphix*(-b*by*gAv + 2*gAv_eta*r)/(a*b*r) + 2*B16*fBphix_xi*gAv*gBphix*(-a*bx*fAv + 2*fAv_xi*r)/((a*a)*r) + 2*B26*fAv*fBphix*gBphix_eta*(-b*by*gAv + 2*gAv_eta*r)/((b*b)*r) + 2*B66*fBphix*gAv*gBphix_eta*(-a*bx*fAv + 2*fAv_xi*r)/(a*b*r) + 4*D12*fAv*fBphix_xi*gAv_eta*gBphix/(a*b*r) + 6*D16*fAv_xi*fBphix_xi*gAv*gBphix/((a*a)*r) + 4*D26*fAv*fBphix*gAv_eta*gBphix_eta/((b*b)*r) + 6*D66*fAv_xi*fBphix*gAv*gBphix_eta/(a*b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 2*B22*fAv*fBphiy*gBphiy_eta*(-b*by*gAv + 2*gAv_eta*r)/((b*b)*r) + 2*B26*(-a*bx*fAv*fBphiy*gAv*gBphiy_eta - b*by*fAv*fBphiy_xi*gAv*gBphiy + 2*fAv*fBphiy_xi*gAv_eta*gBphiy*r + 2*fAv_xi*fBphiy*gAv*gBphiy_eta*r)/(a*b*r) + 2*B66*fBphiy_xi*gAv*gBphiy*(-a*bx*fAv + 2*fAv_xi*r)/((a*a)*r) + 4*D22*fAv*fBphiy*gAv_eta*gBphiy_eta/((b*b)*r) + 2*D26*(2*fAv*fBphiy_xi*gAv_eta*gBphiy + 3*fAv_xi*fBphiy*gAv*gBphiy_eta)/(a*b*r) + 6*D66*fAv_xi*fBphiy_xi*gAv*gBphiy/((a*a)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*bx*fAw_xi*fBu_xi*gAw*gBu/(a*a) + 2*A12*fAw*fBu_xi*gBu*(b*gAw + 2*by*gAw_eta*r)/(a*b*r) + 4*A16*(a*bx*fAw*fBu_xi*gAw_eta*gBu + a*bx*fAw_xi*fBu*gAw*gBu_eta + b*by*fAw_xi*fBu_xi*gAw*gBu)/((a*a)*b) + 2*A26*fAw*fBu*gBu_eta*(b*gAw + 2*by*gAw_eta*r)/((b*b)*r) + 4*A66*fBu*gBu_eta*(a*bx*fAw*gAw_eta + b*by*fAw_xi*gAw)/(a*(b*b)) - 2*B16*bx*fAw_xi*fBu*gAw*gBu_eta/(a*b*r) - B26*fAw*fBu*gBu_eta*(b*gAw + 2*by*gAw_eta*r)/((b*b)*(r*r)) - 2*B66*fBu*gBu_eta*(a*bx*fAw*gAw_eta + b*by*fAw_xi*gAw)/(a*(b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 2*A12*bx*fAw_xi*fBv*gAw*(-b*by*gBv + 2*gBv_eta*r)/(a*b*r) + 2*A16*bx*fAw_xi*gAw*gBv*(-a*bx*fBv + 2*fBv_xi*r)/((a*a)*r) + A22*fAw*fBv*(b*gAw + 2*by*gAw_eta*r)*(-b*by*gBv + 2*gBv_eta*r)/((b*b)*(r*r)) + A26*(-a*(b*b)*bx*fAw*fBv*gAw*gBv - 4*a*b*bx*by*fAw*fBv*gAw_eta*gBv*r + 4*a*bx*fAw*fBv*gAw_eta*gBv_eta*(r*r) - 2*(b*b)*(by*by)*fAw_xi*fBv*gAw*gBv*r + 2*(b*b)*fAw*fBv_xi*gAw*gBv*r + 4*b*by*fAw*fBv_xi*gAw_eta*gBv*(r*r) + 4*b*by*fAw_xi*fBv*gAw*gBv_eta*(r*r))/(a*(b*b)*(r*r)) + 2*A66*gBv*(-a*bx*fBv + 2*fBv_xi*r)*(a*bx*fAw*gAw_eta + b*by*fAw_xi*gAw)/((a*a)*b*r) + 4*B12*bx*fAw_xi*fBv*gAw*gBv_eta/(a*b*r) + 6*B16*bx*fAw_xi*fBv_xi*gAw*gBv/((a*a)*r) + 2*B22*fAw*fBv*gBv_eta*(b*gAw + 2*by*gAw_eta*r)/((b*b)*(r*r)) + B26*(4*a*bx*fAw*fBv*gAw_eta*gBv_eta*r + 3*(b*b)*fAw*fBv_xi*gAw*gBv + 6*b*by*fAw*fBv_xi*gAw_eta*gBv*r + 4*b*by*fAw_xi*fBv*gAw*gBv_eta*r)/(a*(b*b)*(r*r)) + 6*B66*fBv_xi*gBv*(a*bx*fAw*gAw_eta + b*by*fAw_xi*gAw)/((a*a)*b*r) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( -2*NxyNL*fAw_xi*fBv*gAw*gBv/(a*r) - 2*NyyNL*fAw*fBv*gAw_eta*gBv/(b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*(bx*bx)*fAw_xi*fBw_xi*gAw*gBw/(a*a) + 2*A12*bx*(b*fAw*fBw_xi*gAw*gBw + b*fAw_xi*fBw*gAw*gBw + 2*by*fAw*fBw_xi*gAw_eta*gBw*r + 2*by*fAw_xi*fBw*gAw*gBw_eta*r)/(a*b*r) + 4*A16*bx*(a*bx*fAw*fBw_xi*gAw_eta*gBw + a*bx*fAw_xi*fBw*gAw*gBw_eta + 2*b*by*fAw_xi*fBw_xi*gAw*gBw)/((a*a)*b) + A22*fAw*fBw*(b*gAw + 2*by*gAw_eta*r)*(b*gBw + 2*by*gBw_eta*r)/((b*b)*(r*r)) + 2*A26*(a*b*bx*fAw*fBw*gAw*gBw_eta + a*b*bx*fAw*fBw*gAw_eta*gBw + 4*a*bx*by*fAw*fBw*gAw_eta*gBw_eta*r + (b*b)*by*fAw*fBw_xi*gAw*gBw + (b*b)*by*fAw_xi*fBw*gAw*gBw + 2*b*(by*by)*fAw*fBw_xi*gAw_eta*gBw*r + 2*b*(by*by)*fAw_xi*fBw*gAw*gBw_eta*r)/(a*(b*b)*r) + 4*A44*fAw*fBw*gAw_eta*gBw_eta/(b*b) + 4*A45*(fAw*fBw_xi*gAw_eta*gBw + fAw_xi*fBw*gAw*gBw_eta)/(a*b) + 4*A55*fAw_xi*fBw_xi*gAw*gBw/(a*a) + 4*A66*(a*bx*fAw*gAw_eta + b*by*fAw_xi*gAw)*(a*bx*fBw*gBw_eta + b*by*fBw_xi*gBw)/((a*a)*(b*b)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( 4*NxxNL*fAw_xi*fBw_xi*gAw*gBw/(a*a) + 4*NxyNL*(fAw*fBw_xi*gAw_eta*gBw + fAw_xi*fBw*gAw*gBw_eta)/(a*b) + 4*NyyNL*fAw*fBw*gAw_eta*gBw_eta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 2*A45*fAw*fBphix*gAw_eta*gBphix/b + 2*A55*fAw_xi*fBphix*gAw*gBphix/a + 4*B11*bx*fAw_xi*fBphix_xi*gAw*gBphix/(a*a) + 2*B12*fAw*fBphix_xi*gBphix*(b*gAw + 2*by*gAw_eta*r)/(a*b*r) + 4*B16*(a*bx*fAw*fBphix_xi*gAw_eta*gBphix + a*bx*fAw_xi*fBphix*gAw*gBphix_eta + b*by*fAw_xi*fBphix_xi*gAw*gBphix)/((a*a)*b) + 2*B26*fAw*fBphix*gBphix_eta*(b*gAw + 2*by*gAw_eta*r)/((b*b)*r) + 4*B66*fBphix*gBphix_eta*(a*bx*fAw*gAw_eta + b*by*fAw_xi*gAw)/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 2*A44*fAw*fBphiy*gAw_eta*gBphiy/b + 2*A45*fAw_xi*fBphiy*gAw*gBphiy/a + 4*B12*bx*fAw_xi*fBphiy*gAw*gBphiy_eta/(a*b) + 4*B16*bx*fAw_xi*fBphiy_xi*gAw*gBphiy/(a*a) + 2*B22*fAw*fBphiy*gBphiy_eta*(b*gAw + 2*by*gAw_eta*r)/((b*b)*r) + 2*B26*(2*a*bx*fAw*fBphiy*gAw_eta*gBphiy_eta*r + (b*b)*fAw*fBphiy_xi*gAw*gBphiy + 2*b*by*fAw*fBphiy_xi*gAw_eta*gBphiy*r + 2*b*by*fAw_xi*fBphiy*gAw*gBphiy_eta*r)/(a*(b*b)*r) + 4*B66*fBphiy_xi*gBphiy*(a*bx*fAw*gAw_eta + b*by*fAw_xi*gAw)/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*B11*fAphix_xi*fBu_xi*gAphix*gBu/(a*a) + 4*B16*(fAphix*fBu_xi*gAphix_eta*gBu + fAphix_xi*fBu*gAphix*gBu_eta)/(a*b) + 4*B66*fAphix*fBu*gAphix_eta*gBu_eta/(b*b) - 2*D16*fAphix_xi*fBu*gAphix*gBu_eta/(a*b*r) - 2*D66*fAphix*fBu*gAphix_eta*gBu_eta/((b*b)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 2*B12*fAphix_xi*fBv*gAphix*(-b*by*gBv + 2*gBv_eta*r)/(a*b*r) + 2*B16*fAphix_xi*gAphix*gBv*(-a*bx*fBv + 2*fBv_xi*r)/((a*a)*r) + 2*B26*fAphix*fBv*gAphix_eta*(-b*by*gBv + 2*gBv_eta*r)/((b*b)*r) + 2*B66*fAphix*gAphix_eta*gBv*(-a*bx*fBv + 2*fBv_xi*r)/(a*b*r) + 4*D12*fAphix_xi*fBv*gAphix*gBv_eta/(a*b*r) + 6*D16*fAphix_xi*fBv_xi*gAphix*gBv/((a*a)*r) + 4*D26*fAphix*fBv*gAphix_eta*gBv_eta/((b*b)*r) + 6*D66*fAphix*fBv_xi*gAphix_eta*gBv/(a*b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 2*A45*fAphix*fBw*gAphix*gBw_eta/b + 2*A55*fAphix*fBw_xi*gAphix*gBw/a + 4*B11*bx*fAphix_xi*fBw_xi*gAphix*gBw/(a*a) + 2*B12*fAphix_xi*fBw*gAphix*(b*gBw + 2*by*gBw_eta*r)/(a*b*r) + 4*B16*(a*bx*fAphix*fBw_xi*gAphix_eta*gBw + a*bx*fAphix_xi*fBw*gAphix*gBw_eta + b*by*fAphix_xi*fBw_xi*gAphix*gBw)/((a*a)*b) + 2*B26*fAphix*fBw*gAphix_eta*(b*gBw + 2*by*gBw_eta*r)/((b*b)*r) + 4*B66*fAphix*gAphix_eta*(a*bx*fBw*gBw_eta + b*by*fBw_xi*gBw)/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( A55*fAphix*fBphix*gAphix*gBphix + 4*D11*fAphix_xi*fBphix_xi*gAphix*gBphix/(a*a) + 4*D16*(fAphix*fBphix_xi*gAphix_eta*gBphix + fAphix_xi*fBphix*gAphix*gBphix_eta)/(a*b) + 4*D66*fAphix*fBphix*gAphix_eta*gBphix_eta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( A45*fAphix*fBphiy*gAphix*gBphiy + 4*D12*fAphix_xi*fBphiy*gAphix*gBphiy_eta/(a*b) + 4*D16*fAphix_xi*fBphiy_xi*gAphix*gBphiy/(a*a) + 4*D26*fAphix*fBphiy*gAphix_eta*gBphiy_eta/(b*b) + 4*D66*fAphix*fBphiy_xi*gAphix_eta*gBphiy/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAphiy*fBu_xi*gAphiy_eta*gBu/(a*b) + 4*B16*fAphiy_xi*fBu_xi*gAphiy*gBu/(a*a) + 4*B26*fAphiy*fBu*gAphiy_eta*gBu_eta/(b*b) + 4*B66*fAphiy_xi*fBu*gAphiy*gBu_eta/(a*b) - 2*D26*fAphiy*fBu*gAphiy_eta*gBu_eta/((b*b)*r) - 2*D66*fAphiy_xi*fBu*gAphiy*gBu_eta/(a*b*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 2*B22*fAphiy*fBv*gAphiy_eta*(-b*by*gBv + 2*gBv_eta*r)/((b*b)*r) + 2*B26*(-a*bx*fAphiy*fBv*gAphiy_eta*gBv - b*by*fAphiy_xi*fBv*gAphiy*gBv + 2*fAphiy*fBv_xi*gAphiy_eta*gBv*r + 2*fAphiy_xi*fBv*gAphiy*gBv_eta*r)/(a*b*r) + 2*B66*fAphiy_xi*gAphiy*gBv*(-a*bx*fBv + 2*fBv_xi*r)/((a*a)*r) + 4*D22*fAphiy*fBv*gAphiy_eta*gBv_eta/((b*b)*r) + 2*D26*(3*fAphiy*fBv_xi*gAphiy_eta*gBv + 2*fAphiy_xi*fBv*gAphiy*gBv_eta)/(a*b*r) + 6*D66*fAphiy_xi*fBv_xi*gAphiy*gBv/((a*a)*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 2*A44*fAphiy*fBw*gAphiy*gBw_eta/b + 2*A45*fAphiy*fBw_xi*gAphiy*gBw/a + 4*B12*bx*fAphiy*fBw_xi*gAphiy_eta*gBw/(a*b) + 4*B16*bx*fAphiy_xi*fBw_xi*gAphiy*gBw/(a*a) + 2*B22*fAphiy*fBw*gAphiy_eta*(b*gBw + 2*by*gBw_eta*r)/((b*b)*r) + 2*B26*(2*a*bx*fAphiy*fBw*gAphiy_eta*gBw_eta*r + (b*b)*fAphiy_xi*fBw*gAphiy*gBw + 2*b*by*fAphiy*fBw_xi*gAphiy_eta*gBw*r + 2*b*by*fAphiy_xi*fBw*gAphiy*gBw_eta*r)/(a*(b*b)*r) + 4*B66*fAphiy_xi*gAphiy*(a*bx*fBw*gBw_eta + b*by*fBw_xi*gBw)/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( A45*fAphiy*fBphix*gAphiy*gBphix + 4*D12*fAphiy*fBphix_xi*gAphiy_eta*gBphix/(a*b) + 4*D16*fAphiy_xi*fBphix_xi*gAphiy*gBphix/(a*a) + 4*D26*fAphiy*fBphix*gAphiy_eta*gBphix_eta/(b*b) + 4*D66*fAphiy_xi*fBphix*gAphiy*gBphix_eta/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( A44*fAphiy*fBphiy*gAphiy*gBphiy + 4*D22*fAphiy*fBphiy*gAphiy_eta*gBphiy_eta/(b*b) + 4*D26*(fAphiy*fBphiy_xi*gAphiy_eta*gBphiy + fAphiy_xi*fBphiy*gAphiy*gBphiy_eta)/(a*b) + 4*D66*fAphiy_xi*fBphiy_xi*gAphiy*gBphiy/(a*a) )

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
    cdef double a, b, intx, inty
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

    cdef double e[8]
    cdef double Nxx, Nyy, Nxy

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # F as 4-D matrix, must be [nx, ny, 8, 8], when there is one
    # constitutive matrix [8, 8] for each of the nx * ny integration points
    cdef double F[64]
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
                        e[6] += cs[col+4]*fAphiy*gAphiy + 2*cs[col+2]*fAw*gAw_eta/b
                        e[7] += cs[col+3]*fAphix*gAphix + 2*cs[col+2]*fAw_xi*gAw/a

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
    cdef double a, b, intx, inty
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

    cdef double fAu, fAv, fAw, fAphix, fAphiy
    cdef double fBu, fBv, fBw, fBphix, fBphiy
    cdef double gAu, gAv, gAw, gAphix, gAphiy
    cdef double gBu, gBv, gBw, gBphix, gBphiy
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
    x1phix = shell.x1phix; x1phixr = shell.x1phixr; x2phix = shell.x2phix; x2phixr = shell.x2phixr
    x1phiy = shell.x1phiy; x1phiyr = shell.x1phiyr; x2phiy = shell.x2phiy; x2phiyr = shell.x2phiyr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    y1phix = shell.y1phix; y1phixr = shell.y1phixr; y2phix = shell.y2phix; y2phixr = shell.y2phixr
    y1phiy = shell.y1phiy; y1phiyr = shell.y1phiyr; y2phiy = shell.y2phiy; y2phiyr = shell.y2phiyr

    fdim = 9*m*m*n*n

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

    ftab = np.zeros((5, m), dtype=DOUBLE)
    gtab = np.zeros((5, n), dtype=DOUBLE)

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
                    ftab[3, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[4, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[2, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[3, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[4, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                for i in range(m):
                    fAu = ftab[0, i]
                    fAv = ftab[1, i]
                    fAw = ftab[2, i]
                    fAphix = ftab[3, i]
                    fAphiy = ftab[4, i]

                    for k in range(m):
                        fBu = ftab[0, k]
                        fBv = ftab[1, k]
                        fBw = ftab[2, k]
                        fBphix = ftab[3, k]
                        fBphiy = ftab[4, k]

                        for j in range(n):
                            gAu = gtab[0, j]
                            gAv = gtab[1, j]
                            gAw = gtab[2, j]
                            gAphix = gtab[3, j]
                            gAphiy = gtab[4, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = gtab[0, l]
                                gBv = gtab[1, l]
                                gBw = gtab[2, l]
                                gBphix = gtab[3, l]
                                gBphiy = gtab[4, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( I0*fAu*fBu*gAu*gBu )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( I1*fAu*fBphix*gAu*gBphix )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( I0*fAv*fBv*gAv*gBv + 2*I1*fAv*fBv*gAv*gBv/r + I2*fAv*fBv*gAv*gBv/(r*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( I1*fAv*fBphiy*gAv*gBphiy + I2*fAv*fBphiy*gAv*gBphiy/r )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( I0*fAw*fBw*gAw*gBw )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( I1*fAphix*fBu*gAphix*gBu )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( I2*fAphix*fBphix*gAphix*gBphix )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( I1*fAphiy*fBv*gAphiy*gBv + I2*fAphiy*fBv*gAphiy*gBv/r )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( I2*fAphiy*fBphiy*gAphiy*gBphiy )

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
    cdef double a, b, intx, inty
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
    cdef double e[8]
    cdef double s[8]
    cdef double Nxx, Nyy, Nxy, Mxx, Myy
    cdef double Mxy, Qy, Qx

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

    # F as 4-D matrix, must be [nx, ny, 8, 8], when there is one
    # constitutive matrix [8, 8] for each of the nx * ny integration points
    cdef double F[64]
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

                        bx += (2/a)*cs[col+2]*fAw_xi*gAw
                        by += (2/b)*cs[col+2]*fAw*gAw_eta
                        by -= cs[col+1]*fAv*gAv/r

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
                        e[6] += cs[col+4]*fAphiy*gAphiy + 2*cs[col+2]*fAw*gAw_eta/b
                        e[7] += cs[col+3]*fAphix*gAphix + 2*cs[col+2]*fAw_xi*gAw/a

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
                Qy = s[6]
                Qx = s[7]

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
                        fint[col+1] += weight*(intx*inty/4)*( 3*Mxy*fAv_xi*gAv/(a*r) + 2*Myy*fAv*gAv_eta/(b*r) + Nxy*gAv*(-a*bx*fAv + 2*fAv_xi*r)/(a*r) + Nyy*fAv*(-b*by*gAv + 2*gAv_eta*r)/(b*r) )
                        fint[col+2] += weight*(intx*inty/4)*( 2*Nxx*bx*fAw_xi*gAw/a + 2*Nxy*(a*bx*fAw*gAw_eta + b*by*fAw_xi*gAw)/(a*b) + Nyy*fAw*(b*gAw + 2*by*gAw_eta*r)/(b*r) + 2*Qx*fAw_xi*gAw/a + 2*Qy*fAw*gAw_eta/b )
                        fint[col+3] += weight*(intx*inty/4)*( 2*Mxx*fAphix_xi*gAphix/a + 2*Mxy*fAphix*gAphix_eta/b + Qx*fAphix*gAphix )
                        fint[col+4] += weight*(intx*inty/4)*( 2*Mxy*fAphiy_xi*gAphiy/a + 2*Myy*fAphiy*gAphiy_eta/b + Qy*fAphiy*gAphiy )

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
        Constitutive matrix ``(8, 8)``, or one for each integration
        point ``(nx, ny, 8, 8)``.
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
    cdef double e[8]
    cdef double de[8]
    cdef double s[8]
    cdef double ds[8]
    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw, fAw_xi
    cdef double fAphix, fAphix_xi, fAphiy, fAphiy_xi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw, gAw_eta
    cdef double gAphix, gAphix_eta, gAphiy, gAphiy_eta

    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta, kTv

    # F as 4-D matrix, must be [nx, ny, 8, 8], when there is one
    # constitutive matrix [8, 8] for each of the nx * ny integration points
    cdef double F[64]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    if cs.shape[0] != size or vs.shape[0] != size:
        raise ValueError('cs and vs must have the size %d' % size)
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 8, 8):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (8, 8):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        for i in range(8):
            for j in range(8):
                F[i*8 + j] = Finput[i, j]
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
    ftab = np.zeros((10, m), dtype=DOUBLE)
    gtab = np.zeros((10, n), dtype=DOUBLE)

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
                    for i in range(8):
                        for j in range(8):
                            F[i*8 + j] = Fnxny[ptx, pty, i, j]

                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[7, i] = fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[9, i] = fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[7, j] = fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[9, j] = fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

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
                        bx += 2*fAw_xi*gAw/a*cs[col+2]
                        by += -fAv*gAv/r*cs[col+1] + 2*fAw*gAw_eta/b*cs[col+2]
                        dbx += 2*fAw_xi*gAw/a*vs[col+2]
                        dby += -fAv*gAv/r*vs[col+1] + 2*fAw*gAw_eta/b*vs[col+2]

                # strains of cs, eps = ([B_L] + [B_NL]/2) cs, and strain
                # increment, deps = ([B_L] + [B_NL]) vs
                for i in range(8):
                    e[i] = 0.
                    de[i] = 0.
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_eta = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_eta = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_eta = gtab[5, j]
                    gAphix = gtab[6, j]
                    gAphix_eta = gtab[7, j]
                    gAphiy = gtab[8, j]
                    gAphiy_eta = gtab[9, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_xi = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        fAphix = ftab[6, i]
                        fAphix_xi = ftab[7, i]
                        fAphiy = ftab[8, i]
                        fAphiy_xi = ftab[9, i]
                        col = col0 + DOF*(j*m + i)
                        e[0] += 2*fAu_xi*gAu/a*cs[col+0] + bx*fAw_xi*gAw/a*cs[col+2]
                        e[1] += (-0.5*by*fAv*gAv/r + 2*fAv*gAv_eta/b)*cs[col+1] + (fAw*gAw/r + by*fAw*gAw_eta/b)*cs[col+2]
                        e[2] += 2*fAu*gAu_eta/b*cs[col+0] + (-0.5*bx*fAv*gAv/r + 2*fAv_xi*gAv/a)*cs[col+1] + (bx*fAw*gAw_eta/b + by*fAw_xi*gAw/a)*cs[col+2]
                        e[3] += 2*fAphix_xi*gAphix/a*cs[col+3]
                        e[4] += 2*fAv*gAv_eta/(b*r)*cs[col+1] + 2*fAphiy*gAphiy_eta/b*cs[col+4]
                        e[5] += -fAu*gAu_eta/(b*r)*cs[col+0] + 3*fAv_xi*gAv/(a*r)*cs[col+1] + 2*fAphix*gAphix_eta/b*cs[col+3] + 2*fAphiy_xi*gAphiy/a*cs[col+4]
                        e[6] += 2*fAw*gAw_eta/b*cs[col+2] + fAphiy*gAphiy*cs[col+4]
                        e[7] += 2*fAw_xi*gAw/a*cs[col+2] + fAphix*gAphix*cs[col+3]
                        de[0] += 2*fAu_xi*gAu/a*vs[col+0] + 2*bx*fAw_xi*gAw/a*vs[col+2]
                        de[1] += (-by*fAv*gAv/r + 2*fAv*gAv_eta/b)*vs[col+1] + (fAw*gAw/r + 2*by*fAw*gAw_eta/b)*vs[col+2]
                        de[2] += 2*fAu*gAu_eta/b*vs[col+0] + (-bx*fAv*gAv/r + 2*fAv_xi*gAv/a)*vs[col+1] + (2*bx*fAw*gAw_eta/b + 2*by*fAw_xi*gAw/a)*vs[col+2]
                        de[3] += 2*fAphix_xi*gAphix/a*vs[col+3]
                        de[4] += 2*fAv*gAv_eta/(b*r)*vs[col+1] + 2*fAphiy*gAphiy_eta/b*vs[col+4]
                        de[5] += -fAu*gAu_eta/(b*r)*vs[col+0] + 3*fAv_xi*gAv/(a*r)*vs[col+1] + 2*fAphix*gAphix_eta/b*vs[col+3] + 2*fAphiy_xi*gAphiy/a*vs[col+4]
                        de[6] += 2*fAw*gAw_eta/b*vs[col+2] + fAphiy*gAphiy*vs[col+4]
                        de[7] += 2*fAw_xi*gAw/a*vs[col+2] + fAphix*gAphix*vs[col+3]

                # stresses
                for i in range(8):
                    s[i] = 0.
                    ds[i] = 0.
                    for j in range(8):
                        s[i] += F[i*8 + j]*e[j]
                        ds[i] += F[i*8 + j]*de[j]
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
                    gAphix = gtab[6, j]
                    gAphix_eta = gtab[7, j]
                    gAphiy = gtab[8, j]
                    gAphiy_eta = gtab[9, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_xi = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        fAphix = ftab[6, i]
                        fAphix_xi = ftab[7, i]
                        fAphiy = ftab[8, i]
                        fAphiy_xi = ftab[9, i]
                        col = col0 + DOF*(j*m + i)
                        kTv[col+0] += weight*(2*ds[2]*fAu*gAu_eta/b - ds[5]*fAu*gAu_eta/(b*r) + 2*ds[0]*fAu_xi*gAu/a)
                        kTv[col+1] += weight*(-Nxy*dbx*fAv*gAv/r - Nyy*dby*fAv*gAv/r + ds[1]*(-by*fAv*gAv/r + 2*fAv*gAv_eta/b) + ds[2]*(-bx*fAv*gAv/r + 2*fAv_xi*gAv/a) + 2*ds[4]*fAv*gAv_eta/(b*r) + 3*ds[5]*fAv_xi*gAv/(a*r))
                        kTv[col+2] += weight*(2*Nxx*dbx*fAw_xi*gAw/a + Nxy*(2*dbx*fAw*gAw_eta/b + 2*dby*fAw_xi*gAw/a) + 2*Nyy*dby*fAw*gAw_eta/b + ds[1]*(fAw*gAw/r + 2*by*fAw*gAw_eta/b) + ds[2]*(2*bx*fAw*gAw_eta/b + 2*by*fAw_xi*gAw/a) + 2*ds[6]*fAw*gAw_eta/b + 2*bx*ds[0]*fAw_xi*gAw/a + 2*ds[7]*fAw_xi*gAw/a)
                        kTv[col+3] += weight*(ds[7]*fAphix*gAphix + 2*ds[5]*fAphix*gAphix_eta/b + 2*ds[3]*fAphix_xi*gAphix/a)
                        kTv[col+4] += weight*(ds[6]*fAphiy*gAphiy + 2*ds[4]*fAphiy*gAphiy_eta/b + 2*ds[5]*fAphiy_xi*gAphiy/a)

    return np.asarray(kTv)

# END tangent stiffness product
