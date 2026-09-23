#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Flat plates using the first-order shear deformation theory (FSDT) with von
Karman kinematics

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
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
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

    cdef double fAu, fAuxi, fAv, fAvxi, fAw
    cdef double fAwxi, fAphix, fAphixxi, fAphiy, fAphiyxi
    cdef double fBu, fBuxi, fBv, fBvxi, fBw
    cdef double fBwxi, fBphix, fBphixxi, fBphiy, fBphiyxi
    cdef double gAu, gAueta, gAv, gAveta, gAw
    cdef double gAweta, gAphix, gAphixeta, gAphiy, gAphiyeta
    cdef double gBu, gBueta, gBv, gBveta, gBw
    cdef double gBweta, gBphix, gBphixeta, gBphiy, gBphiyeta
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

                            bx += (2/a)*cs[col+2]*fAwxi*gAw
                            by += (2/b)*cs[col+2]*fAw*gAweta

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
                #     KT = K0 + K0L + KL0 + KLL + KGNL (fkC_num) + KG(N0 + N_L) (fkG_num)
                # is the exact Jacobian of calc_fint, and fkG_num stays
                # homogeneous of degree one in cs, as linear buckling requires.
                # bx = by = 0 when NLgeom == 0, then KGNL vanishes
                NxxNL = A11*0.5*bx*bx + A12*0.5*by*by + A16*bx*by
                NyyNL = A12*0.5*bx*bx + A22*0.5*by*by + A26*bx*by
                NxyNL = A16*0.5*bx*bx + A26*0.5*by*by + A66*bx*by

                # kC
                c = -1
                for i in range(m):
                    fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                    fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                    fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                    fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    fAphix = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    fAphixxi = fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    fAphiy = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    fAphiyxi = fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                    for k in range(m):
                        fBu = f(k, xi, x1u, x1ur, x2u, x2ur)
                        fBuxi = fp(k, xi, x1u, x1ur, x2u, x2ur)
                        fBv = f(k, xi, x1v, x1vr, x2v, x2vr)
                        fBvxi = fp(k, xi, x1v, x1vr, x2v, x2vr)
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)
                        fBwxi = fp(k, xi, x1w, x1wr, x2w, x2wr)
                        fBphix = f(k, xi, x1phix, x1phixr, x2phix, x2phixr)
                        fBphixxi = fp(k, xi, x1phix, x1phixr, x2phix, x2phixr)
                        fBphiy = f(k, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                        fBphiyxi = fp(k, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                        for j in range(n):
                            gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                            gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                            gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                            gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                            gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                            gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                            gAphix = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                            gAphixeta = fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                            gAphiy = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                            gAphiyeta = fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

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
                                gBphix = f(l, eta, y1phix, y1phixr, y2phix, y2phixr)
                                gBphixeta = fp(l, eta, y1phix, y1phixr, y2phix, y2phixr)
                                gBphiy = f(l, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                                gBphiyeta = fp(l, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAuxi*fBuxi*gAu*gBu/(a*a) + 4*A16*(fAu*fBuxi*gAueta*gBu + fAuxi*fBu*gAu*gBueta)/(a*b) + 4*A66*fAu*fBu*gAueta*gBueta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 4*A12*fAuxi*fBv*gAu*gBveta/(a*b) + 4*A16*fAuxi*fBvxi*gAu*gBv/(a*a) + 4*A26*fAu*fBv*gAueta*gBveta/(b*b) + 4*A66*fAu*fBvxi*gAueta*gBv/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*bx*fAuxi*fBwxi*gAu*gBw/(a*a) + 4*A12*by*fAuxi*fBw*gAu*gBweta/(a*b) + 4*A16*(a*bx*fAu*fBwxi*gAueta*gBw + a*bx*fAuxi*fBw*gAu*gBweta + b*by*fAuxi*fBwxi*gAu*gBw)/((a*a)*b) + 4*A26*by*fAu*fBw*gAueta*gBweta/(b*b) + 4*A66*fAu*gAueta*(a*bx*fBw*gBweta + b*by*fBwxi*gBw)/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 4*B11*fAuxi*fBphixxi*gAu*gBphix/(a*a) + 4*B16*(fAu*fBphixxi*gAueta*gBphix + fAuxi*fBphix*gAu*gBphixeta)/(a*b) + 4*B66*fAu*fBphix*gAueta*gBphixeta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAuxi*fBphiy*gAu*gBphiyeta/(a*b) + 4*B16*fAuxi*fBphiyxi*gAu*gBphiy/(a*a) + 4*B26*fAu*fBphiy*gAueta*gBphiyeta/(b*b) + 4*B66*fAu*fBphiyxi*gAueta*gBphiy/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*A12*fAv*fBuxi*gAveta*gBu/(a*b) + 4*A16*fAvxi*fBuxi*gAv*gBu/(a*a) + 4*A26*fAv*fBu*gAveta*gBueta/(b*b) + 4*A66*fAvxi*fBu*gAv*gBueta/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 4*A22*fAv*fBv*gAveta*gBveta/(b*b) + 4*A26*(fAv*fBvxi*gAveta*gBv + fAvxi*fBv*gAv*gBveta)/(a*b) + 4*A66*fAvxi*fBvxi*gAv*gBv/(a*a) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A12*bx*fAv*fBwxi*gAveta*gBw/(a*b) + 4*A16*bx*fAvxi*fBwxi*gAv*gBw/(a*a) + 4*A22*by*fAv*fBw*gAveta*gBweta/(b*b) + 4*A26*(a*bx*fAv*fBw*gAveta*gBweta + b*by*fAv*fBwxi*gAveta*gBw + b*by*fAvxi*fBw*gAv*gBweta)/(a*(b*b)) + 4*A66*fAvxi*gAv*(a*bx*fBw*gBweta + b*by*fBwxi*gBw)/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAv*fBphixxi*gAveta*gBphix/(a*b) + 4*B16*fAvxi*fBphixxi*gAv*gBphix/(a*a) + 4*B26*fAv*fBphix*gAveta*gBphixeta/(b*b) + 4*B66*fAvxi*fBphix*gAv*gBphixeta/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 4*B22*fAv*fBphiy*gAveta*gBphiyeta/(b*b) + 4*B26*(fAv*fBphiyxi*gAveta*gBphiy + fAvxi*fBphiy*gAv*gBphiyeta)/(a*b) + 4*B66*fAvxi*fBphiyxi*gAv*gBphiy/(a*a) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*bx*fAwxi*fBuxi*gAw*gBu/(a*a) + 4*A12*by*fAw*fBuxi*gAweta*gBu/(a*b) + 4*A16*(a*bx*fAw*fBuxi*gAweta*gBu + a*bx*fAwxi*fBu*gAw*gBueta + b*by*fAwxi*fBuxi*gAw*gBu)/((a*a)*b) + 4*A26*by*fAw*fBu*gAweta*gBueta/(b*b) + 4*A66*fBu*gBueta*(a*bx*fAw*gAweta + b*by*fAwxi*gAw)/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 4*A12*bx*fAwxi*fBv*gAw*gBveta/(a*b) + 4*A16*bx*fAwxi*fBvxi*gAw*gBv/(a*a) + 4*A22*by*fAw*fBv*gAweta*gBveta/(b*b) + 4*A26*(a*bx*fAw*fBv*gAweta*gBveta + b*by*fAw*fBvxi*gAweta*gBv + b*by*fAwxi*fBv*gAw*gBveta)/(a*(b*b)) + 4*A66*fBvxi*gBv*(a*bx*fAw*gAweta + b*by*fAwxi*gAw)/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*(bx*bx)*fAwxi*fBwxi*gAw*gBw/(a*a) + 4*A12*bx*by*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta)/(a*b) + 4*A16*bx*(a*bx*fAw*fBwxi*gAweta*gBw + a*bx*fAwxi*fBw*gAw*gBweta + 2*b*by*fAwxi*fBwxi*gAw*gBw)/((a*a)*b) + 4*A22*(by*by)*fAw*fBw*gAweta*gBweta/(b*b) + 4*A26*by*(2*a*bx*fAw*fBw*gAweta*gBweta + b*by*fAw*fBwxi*gAweta*gBw + b*by*fAwxi*fBw*gAw*gBweta)/(a*(b*b)) + 4*A44*fAw*fBw*gAweta*gBweta/(b*b) + 4*A45*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta)/(a*b) + 4*A55*fAwxi*fBwxi*gAw*gBw/(a*a) + 4*A66*(a*bx*fAw*gAweta + b*by*fAwxi*gAw)*(a*bx*fBw*gBweta + b*by*fBwxi*gBw)/((a*a)*(b*b)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( 4*NxxNL*fAwxi*fBwxi*gAw*gBw/(a*a) + 4*NxyNL*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta)/(a*b) + 4*NyyNL*fAw*fBw*gAweta*gBweta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 2*A45*fAw*fBphix*gAweta*gBphix/b + 2*A55*fAwxi*fBphix*gAw*gBphix/a + 4*B11*bx*fAwxi*fBphixxi*gAw*gBphix/(a*a) + 4*B12*by*fAw*fBphixxi*gAweta*gBphix/(a*b) + 4*B16*(a*bx*fAw*fBphixxi*gAweta*gBphix + a*bx*fAwxi*fBphix*gAw*gBphixeta + b*by*fAwxi*fBphixxi*gAw*gBphix)/((a*a)*b) + 4*B26*by*fAw*fBphix*gAweta*gBphixeta/(b*b) + 4*B66*fBphix*gBphixeta*(a*bx*fAw*gAweta + b*by*fAwxi*gAw)/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 2*A44*fAw*fBphiy*gAweta*gBphiy/b + 2*A45*fAwxi*fBphiy*gAw*gBphiy/a + 4*B12*bx*fAwxi*fBphiy*gAw*gBphiyeta/(a*b) + 4*B16*bx*fAwxi*fBphiyxi*gAw*gBphiy/(a*a) + 4*B22*by*fAw*fBphiy*gAweta*gBphiyeta/(b*b) + 4*B26*(a*bx*fAw*fBphiy*gAweta*gBphiyeta + b*by*fAw*fBphiyxi*gAweta*gBphiy + b*by*fAwxi*fBphiy*gAw*gBphiyeta)/(a*(b*b)) + 4*B66*fBphiyxi*gBphiy*(a*bx*fAw*gAweta + b*by*fAwxi*gAw)/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*B11*fAphixxi*fBuxi*gAphix*gBu/(a*a) + 4*B16*(fAphix*fBuxi*gAphixeta*gBu + fAphixxi*fBu*gAphix*gBueta)/(a*b) + 4*B66*fAphix*fBu*gAphixeta*gBueta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAphixxi*fBv*gAphix*gBveta/(a*b) + 4*B16*fAphixxi*fBvxi*gAphix*gBv/(a*a) + 4*B26*fAphix*fBv*gAphixeta*gBveta/(b*b) + 4*B66*fAphix*fBvxi*gAphixeta*gBv/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 2*A45*fAphix*fBw*gAphix*gBweta/b + 2*A55*fAphix*fBwxi*gAphix*gBw/a + 4*B11*bx*fAphixxi*fBwxi*gAphix*gBw/(a*a) + 4*B12*by*fAphixxi*fBw*gAphix*gBweta/(a*b) + 4*B16*(a*bx*fAphix*fBwxi*gAphixeta*gBw + a*bx*fAphixxi*fBw*gAphix*gBweta + b*by*fAphixxi*fBwxi*gAphix*gBw)/((a*a)*b) + 4*B26*by*fAphix*fBw*gAphixeta*gBweta/(b*b) + 4*B66*fAphix*gAphixeta*(a*bx*fBw*gBweta + b*by*fBwxi*gBw)/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( A55*fAphix*fBphix*gAphix*gBphix + 4*D11*fAphixxi*fBphixxi*gAphix*gBphix/(a*a) + 4*D16*(fAphix*fBphixxi*gAphixeta*gBphix + fAphixxi*fBphix*gAphix*gBphixeta)/(a*b) + 4*D66*fAphix*fBphix*gAphixeta*gBphixeta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( A45*fAphix*fBphiy*gAphix*gBphiy + 4*D12*fAphixxi*fBphiy*gAphix*gBphiyeta/(a*b) + 4*D16*fAphixxi*fBphiyxi*gAphix*gBphiy/(a*a) + 4*D26*fAphix*fBphiy*gAphixeta*gBphiyeta/(b*b) + 4*D66*fAphix*fBphiyxi*gAphixeta*gBphiy/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAphiy*fBuxi*gAphiyeta*gBu/(a*b) + 4*B16*fAphiyxi*fBuxi*gAphiy*gBu/(a*a) + 4*B26*fAphiy*fBu*gAphiyeta*gBueta/(b*b) + 4*B66*fAphiyxi*fBu*gAphiy*gBueta/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 4*B22*fAphiy*fBv*gAphiyeta*gBveta/(b*b) + 4*B26*(fAphiy*fBvxi*gAphiyeta*gBv + fAphiyxi*fBv*gAphiy*gBveta)/(a*b) + 4*B66*fAphiyxi*fBvxi*gAphiy*gBv/(a*a) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 2*A44*fAphiy*fBw*gAphiy*gBweta/b + 2*A45*fAphiy*fBwxi*gAphiy*gBw/a + 4*B12*bx*fAphiy*fBwxi*gAphiyeta*gBw/(a*b) + 4*B16*bx*fAphiyxi*fBwxi*gAphiy*gBw/(a*a) + 4*B22*by*fAphiy*fBw*gAphiyeta*gBweta/(b*b) + 4*B26*(a*bx*fAphiy*fBw*gAphiyeta*gBweta + b*by*fAphiy*fBwxi*gAphiyeta*gBw + b*by*fAphiyxi*fBw*gAphiy*gBweta)/(a*(b*b)) + 4*B66*fAphiyxi*gAphiy*(a*bx*fBw*gBweta + b*by*fBwxi*gBw)/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( A45*fAphiy*fBphix*gAphiy*gBphix + 4*D12*fAphiy*fBphixxi*gAphiyeta*gBphix/(a*b) + 4*D16*fAphiyxi*fBphixxi*gAphiy*gBphix/(a*a) + 4*D26*fAphiy*fBphix*gAphiyeta*gBphixeta/(b*b) + 4*D66*fAphiyxi*fBphix*gAphiy*gBphixeta/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( A44*fAphiy*fBphiy*gAphiy*gBphiy + 4*D22*fAphiy*fBphiy*gAphiyeta*gBphiyeta/(b*b) + 4*D26*(fAphiy*fBphiyxi*gAphiyeta*gBphiy + fAphiyxi*fBphiy*gAphiy*gBphiyeta)/(a*b) + 4*D66*fAphiyxi*fBphiyxi*gAphiy*gBphiy/(a*a) )

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

        KT = K0 + K0L + KL0 + KLL + KGNL   (fkC_num)
             + KG(N0 + N_L)                (fkG_num)

    is the exact Jacobian of :func:`.calc_fint` while this function stays
    homogeneous of degree one in ``cs``.

    """
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
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

    cdef double fAu, fAuxi, fAv, fAvxi, fAw
    cdef double fAwxi, fAwxixi, fAphix, fAphixxi, fAphiy
    cdef double fAphiyxi
    cdef double fBw, fBwxi
    cdef double gAu, gAueta, gAv, gAveta, gAw
    cdef double gAweta, gAwetaeta, gAphix, gAphixeta, gAphiy
    cdef double gAphiyeta
    cdef double gBw, gBweta

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

    fdim = 1*m*m*n*n

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
                    for i in range(NE):
                        for j in range(NE):
                            #TODO could assume symmetry
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

                # Calculating the linear generalized strains. The stress of
                # the nonlinear strain enters KT through KGNL in fkC_num, such
                # that kG is homogeneous of degree one in cs
                for i in range(NE):
                    e[i] = 0.
                for j in range(n):
                    #TODO put these in a lookup vector
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAphix = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gAphixeta = fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gAphiy = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gAphiyeta = fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                    for i in range(m):
                        fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                        fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fpp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAphix = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                        fAphixxi = fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                        fAphiy = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                        fAphiyxi = fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                        col = col0 + DOF*(j*m + i)

                        e[0] += 2*cs[col+0]*fAuxi*gAu/a
                        e[1] += 2*cs[col+1]*fAv*gAveta/b
                        e[2] += 2*cs[col+0]*fAu*gAueta/b + 2*cs[col+1]*fAvxi*gAv/a
                        e[3] += 2*cs[col+3]*fAphixxi*gAphix/a
                        e[4] += 2*cs[col+4]*fAphiy*gAphiyeta/b
                        e[5] += 2*cs[col+3]*fAphix*gAphixeta/b + 2*cs[col+4]*fAphiyxi*gAphiy/a
                        e[6] += cs[col+4]*fAphiy*gAphiy + 2*cs[col+2]*fAw*gAweta/b
                        e[7] += cs[col+3]*fAphix*gAphix + 2*cs[col+2]*fAwxi*gAw/a

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
                    fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                    fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                    for k in range(m):
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)
                        fBwxi = fp(k, xi, x1w, x1wr, x2w, x2wr)

                        for j in range(n):
                            gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                            gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBw = f(l, eta, y1w, y1wr, y2w, y2wr)
                                gBweta = fp(l, eta, y1w, y1wr, y2w, y2wr)

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
    m = shell.m
    n = shell.n
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
                    fAphix = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    fAphiy = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                    for k in range(m):
                        fBu = f(k, xi, x1u, x1ur, x2u, x2ur)
                        fBv = f(k, xi, x1v, x1vr, x2v, x2vr)
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)
                        fBphix = f(k, xi, x1phix, x1phixr, x2phix, x2phixr)
                        fBphiy = f(k, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                        for j in range(n):
                            gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                            gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                            gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                            gAphix = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                            gAphiy = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = f(l, eta, y1u, y1ur, y2u, y2ur)
                                gBv = f(l, eta, y1v, y1vr, y2v, y2vr)
                                gBw = f(l, eta, y1w, y1wr, y2w, y2wr)
                                gBphix = f(l, eta, y1phix, y1phixr, y2phix, y2phixr)
                                gBphiy = f(l, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( fAu*fBu*gAu*gBu*h*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( -d*fAu*fBphix*gAu*gBphix*h*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( fAv*fBv*gAv*gBv*h*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( -d*fAv*fBphiy*gAv*gBphiy*h*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( fAw*fBw*gAw*gBw*h*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( -d*fAphix*fBu*gAphix*gBu*h*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( (d*d)*fAphix*fBphix*gAphix*gBphix*h*rho + 0.08333333333333333*fAphix*fBphix*gAphix*gBphix*(h*h*h)*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( -d*fAphiy*fBv*gAphiy*gBv*h*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( (d*d)*fAphiy*fBphiy*gAphiy*gBphiy*h*rho + 0.08333333333333333*fAphiy*fBphiy*gAphiy*gBphiy*(h*h*h)*rho )

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


def fkAx_num(object shell, int size, int row0, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
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

    cdef double fAwxi
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
                for i in range(m):
                    fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                    for k in range(m):
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)

                        for j in range(n):
                            gAw = f(j, eta, y1w, y1wr, y2w, y2wr)

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                gBw = f(l, eta, y1w, y1wr, y2w, y2wr)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kAr[c] = row+2
                                    kAc[c] = col+2
                                kAv[c] += weight*(intx*inty/4)*( -2*beta*fAwxi*fBw*gAw*gBw/a )

    kA = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kA


def fkAy_num(object shell, int size, int row0, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
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
    cdef double gAweta
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
                for i in range(m):
                    fAw = f(i, xi, x1w, x1wr, x2w, x2wr)

                    for k in range(m):
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)

                        for j in range(n):
                            gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                gBw = f(l, eta, y1w, y1wr, y2w, y2wr)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kAr[c] = row+2
                                    kAc[c] = col+2
                                kAv[c] += weight*(intx*inty/4)*( -2*beta*fAw*fBw*gAweta*gBw/b )

    kA = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kA


def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
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

    cdef double fAu, fAuxi, fAv, fAvxi, fAw
    cdef double fAwxi, fAwxixi, fAphix, fAphixxi, fAphiy
    cdef double fAphiyxi
    cdef double gAu, gAueta, gAv, gAveta, gAw
    cdef double gAweta, gAwetaeta, gAphix, gAphixeta, gAphiy
    cdef double gAphiyeta

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
                for j in range(n):
                    #TODO put these in a lookup vector
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    for i in range(m):
                        #TODO put these in a lookup vector
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

                        bx += (2/a)*cs[col+2]*fAwxi*gAw
                        by += (2/b)*cs[col+2]*fAw*gAweta

                # current generalized strain state
                for i in range(NE):
                    e[i] = 0.
                for j in range(n):
                    #TODO save in buffer
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAphix = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gAphixeta = fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gAphiy = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gAphiyeta = fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                    for i in range(m):
                        #TODO save in buffer
                        fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                        fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fpp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAphix = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                        fAphixxi = fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                        fAphiy = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                        fAphiyxi = fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                        col = col0 + DOF*(j*m + i)

                        e[0] += 2*cs[col+0]*fAuxi*gAu/a
                        e[1] += 2*cs[col+1]*fAv*gAveta/b
                        e[2] += 2*cs[col+0]*fAu*gAueta/b + 2*cs[col+1]*fAvxi*gAv/a
                        e[3] += 2*cs[col+3]*fAphixxi*gAphix/a
                        e[4] += 2*cs[col+4]*fAphiy*gAphiyeta/b
                        e[5] += 2*cs[col+3]*fAphix*gAphixeta/b + 2*cs[col+4]*fAphiyxi*gAphiy/a
                        e[6] += cs[col+4]*fAphiy*gAphiy + 2*cs[col+2]*fAw*gAweta/b
                        e[7] += cs[col+3]*fAphix*gAphix + 2*cs[col+2]*fAwxi*gAw/a

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
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAphix = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gAphixeta = fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gAphiy = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gAphiyeta = fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    for i in range(m):
                        fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                        fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fpp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAphix = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                        fAphixxi = fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                        fAphiy = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                        fAphiyxi = fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                        col = col0 + DOF*(j*m + i)

                        fint[col+0] += weight*(intx*inty/4)*( 2*Nxx*fAuxi*gAu/a + 2*Nxy*fAu*gAueta/b )
                        fint[col+1] += weight*(intx*inty/4)*( 2*Nxy*fAvxi*gAv/a + 2*Nyy*fAv*gAveta/b )
                        fint[col+2] += weight*(intx*inty/4)*( 2*Nxx*bx*fAwxi*gAw/a + 2*Nxy*(a*bx*fAw*gAweta + b*by*fAwxi*gAw)/(a*b) + 2*Nyy*by*fAw*gAweta/b + 2*Qx*fAwxi*gAw/a + 2*Qy*fAw*gAweta/b )
                        fint[col+3] += weight*(intx*inty/4)*( 2*Mxx*fAphixxi*gAphix/a + 2*Mxy*fAphix*gAphixeta/b + Qx*fAphix*gAphix )
                        fint[col+4] += weight*(intx*inty/4)*( 2*Mxy*fAphiyxi*gAphiy/a + 2*Myy*fAphiy*gAphiyeta/b + Qy*fAphiy*gAphiy )

    return fint
