#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Numerically integrated matrices of flat plates using the Reddy's third-order shear deformation theory (TSDT) with von
Karman kinematics

See the kinematic equations in
``theory/shells/plate_fsdt_tsdt_donnell/plate_fsdt_tsdt_donnell.py``, from
where the integrands herein have been generated. The degrees of freedom of
each term of the approximation are ``u, v, w, phix, phiy``, and the
constitutive matrix ``Finput`` must be ``13 x 13``, with the rows and
columns in the order of the generalized strains::

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
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
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

    cdef double fAu, fAuxi, fAv, fAvxi, fAw
    cdef double fAwxi, fAwxixi, fAphix, fAphixxi, fAphiy
    cdef double fAphiyxi
    cdef double fBu, fBuxi, fBv, fBvxi, fBw
    cdef double fBwxi, fBwxixi, fBphix, fBphixxi, fBphiy
    cdef double fBphiyxi
    cdef double gAu, gAueta, gAv, gAveta, gAw
    cdef double gAweta, gAwetaeta, gAphix, gAphixeta, gAphiy
    cdef double gAphiyeta
    cdef double gBu, gBueta, gBv, gBveta, gBw
    cdef double gBweta, gBwetaeta, gBphix, gBphixeta, gBphiy
    cdef double gBphiyeta
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double wx, wy, NxxNL, NyyNL, NxyNL

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

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

                wx = 0
                wy = 0
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

                            wx += (2/a)*cs[col+2]*fAwxi*gAw
                            wy += (2/b)*cs[col+2]*fAw*gAweta

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
                # eps_NL = {w,x^2/2, w,y^2/2, w,x*w,y}. With it, KGNL = KG(N_NL)
                # is collected in kC such that
                #     KT = K0 + K0L + KL0 + KLL + KGNL (fkC_num) + KG(N0 + N_L) (fkG_num)
                # is the exact Jacobian of calc_fint, and fkG_num stays
                # homogeneous of degree one in cs, as linear buckling requires.
                # wx = wy = 0 when NLgeom == 0, then KGNL vanishes
                NxxNL = A11*0.5*wx*wx + A12*0.5*wy*wy + A16*wx*wy
                NyyNL = A12*0.5*wx*wx + A22*0.5*wy*wy + A26*wx*wy
                NxyNL = A16*0.5*wx*wx + A26*0.5*wy*wy + A66*wx*wy

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
                        fBwxixi = fpp(k, xi, x1w, x1wr, x2w, x2wr)
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
                            gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)
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
                                gBwetaeta = fpp(l, eta, y1w, y1wr, y2w, y2wr)
                                gBphix = f(l, eta, y1phix, y1phixr, y2phix, y2phixr)
                                gBphixeta = fp(l, eta, y1phix, y1phixr, y2phix, y2phixr)
                                gBphiy = f(l, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                                gBphiyeta = fp(l, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAuxi*fBuxi*gAu*gBu/(a*a) + A16*(4*fAu*fBuxi*gAueta*gBu/(a*b) + 4*fAuxi*fBu*gAu*gBueta/(a*b)) + 4*A66*fAu*fBu*gAueta*gBueta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 4*A12*fAuxi*fBv*gAu*gBveta/(a*b) + 4*A16*fAuxi*fBvxi*gAu*gBv/(a*a) + 4*A26*fAu*fBv*gAueta*gBveta/(b*b) + 4*A66*fAu*fBvxi*gAueta*gBv/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAuxi*fBwxi*gAu*gBw*wx/(a*a) + 4*A12*fAuxi*fBw*gAu*gBweta*wy/(a*b) + A16*(4*fAu*fBwxi*gAueta*gBw*wx/(a*b) + 4*fAuxi*fBw*gAu*gBweta*wx/(a*b) + 4*fAuxi*fBwxi*gAu*gBw*wy/(a*a)) + 4*A26*fAu*fBw*gAueta*gBweta*wy/(b*b) + A66*(4*fAu*fBw*gAueta*gBweta*wx/(b*b) + 4*fAu*fBwxi*gAueta*gBw*wy/(a*b)) - 8*E11*c1*fAuxi*fBwxixi*gAu*gBw/(a*a*a) - 8*E12*c1*fAuxi*fBw*gAu*gBwetaeta/(a*(b*b)) + E16*(-8*c1*fAu*fBwxixi*gAueta*gBw/((a*a)*b) - 16*c1*fAuxi*fBwxi*gAu*gBweta/((a*a)*b)) - 8*E26*c1*fAu*fBw*gAueta*gBwetaeta/(b*b*b) - 16*E66*c1*fAu*fBwxi*gAueta*gBweta/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 4*B11*fAuxi*fBphixxi*gAu*gBphix/(a*a) + B16*(4*fAu*fBphixxi*gAueta*gBphix/(a*b) + 4*fAuxi*fBphix*gAu*gBphixeta/(a*b)) + 4*B66*fAu*fBphix*gAueta*gBphixeta/(b*b) - 4*E11*c1*fAuxi*fBphixxi*gAu*gBphix/(a*a) + E16*(-4*c1*fAu*fBphixxi*gAueta*gBphix/(a*b) - 4*c1*fAuxi*fBphix*gAu*gBphixeta/(a*b)) - 4*E66*c1*fAu*fBphix*gAueta*gBphixeta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAuxi*fBphiy*gAu*gBphiyeta/(a*b) + 4*B16*fAuxi*fBphiyxi*gAu*gBphiy/(a*a) + 4*B26*fAu*fBphiy*gAueta*gBphiyeta/(b*b) + 4*B66*fAu*fBphiyxi*gAueta*gBphiy/(a*b) - 4*E12*c1*fAuxi*fBphiy*gAu*gBphiyeta/(a*b) - 4*E16*c1*fAuxi*fBphiyxi*gAu*gBphiy/(a*a) - 4*E26*c1*fAu*fBphiy*gAueta*gBphiyeta/(b*b) - 4*E66*c1*fAu*fBphiyxi*gAueta*gBphiy/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*A12*fAv*fBuxi*gAveta*gBu/(a*b) + 4*A16*fAvxi*fBuxi*gAv*gBu/(a*a) + 4*A26*fAv*fBu*gAveta*gBueta/(b*b) + 4*A66*fAvxi*fBu*gAv*gBueta/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 4*A22*fAv*fBv*gAveta*gBveta/(b*b) + A26*(4*fAv*fBvxi*gAveta*gBv/(a*b) + 4*fAvxi*fBv*gAv*gBveta/(a*b)) + 4*A66*fAvxi*fBvxi*gAv*gBv/(a*a) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A12*fAv*fBwxi*gAveta*gBw*wx/(a*b) + 4*A16*fAvxi*fBwxi*gAv*gBw*wx/(a*a) + 4*A22*fAv*fBw*gAveta*gBweta*wy/(b*b) + A26*(4*fAv*fBw*gAveta*gBweta*wx/(b*b) + 4*fAv*fBwxi*gAveta*gBw*wy/(a*b) + 4*fAvxi*fBw*gAv*gBweta*wy/(a*b)) + A66*(4*fAvxi*fBw*gAv*gBweta*wx/(a*b) + 4*fAvxi*fBwxi*gAv*gBw*wy/(a*a)) - 8*E12*c1*fAv*fBwxixi*gAveta*gBw/((a*a)*b) - 8*E16*c1*fAvxi*fBwxixi*gAv*gBw/(a*a*a) - 8*E22*c1*fAv*fBw*gAveta*gBwetaeta/(b*b*b) + E26*(-16*c1*fAv*fBwxi*gAveta*gBweta/(a*(b*b)) - 8*c1*fAvxi*fBw*gAv*gBwetaeta/(a*(b*b))) - 16*E66*c1*fAvxi*fBwxi*gAv*gBweta/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAv*fBphixxi*gAveta*gBphix/(a*b) + 4*B16*fAvxi*fBphixxi*gAv*gBphix/(a*a) + 4*B26*fAv*fBphix*gAveta*gBphixeta/(b*b) + 4*B66*fAvxi*fBphix*gAv*gBphixeta/(a*b) - 4*E12*c1*fAv*fBphixxi*gAveta*gBphix/(a*b) - 4*E16*c1*fAvxi*fBphixxi*gAv*gBphix/(a*a) - 4*E26*c1*fAv*fBphix*gAveta*gBphixeta/(b*b) - 4*E66*c1*fAvxi*fBphix*gAv*gBphixeta/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 4*B22*fAv*fBphiy*gAveta*gBphiyeta/(b*b) + B26*(4*fAv*fBphiyxi*gAveta*gBphiy/(a*b) + 4*fAvxi*fBphiy*gAv*gBphiyeta/(a*b)) + 4*B66*fAvxi*fBphiyxi*gAv*gBphiy/(a*a) - 4*E22*c1*fAv*fBphiy*gAveta*gBphiyeta/(b*b) + E26*(-4*c1*fAv*fBphiyxi*gAveta*gBphiy/(a*b) - 4*c1*fAvxi*fBphiy*gAv*gBphiyeta/(a*b)) - 4*E66*c1*fAvxi*fBphiyxi*gAv*gBphiy/(a*a) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAwxi*fBuxi*gAw*gBu*wx/(a*a) + 4*A12*fAw*fBuxi*gAweta*gBu*wy/(a*b) + A16*(4*fAw*fBuxi*gAweta*gBu*wx/(a*b) + 4*fAwxi*fBu*gAw*gBueta*wx/(a*b) + 4*fAwxi*fBuxi*gAw*gBu*wy/(a*a)) + 4*A26*fAw*fBu*gAweta*gBueta*wy/(b*b) + A66*(4*fAw*fBu*gAweta*gBueta*wx/(b*b) + 4*fAwxi*fBu*gAw*gBueta*wy/(a*b)) - 8*E11*c1*fAwxixi*fBuxi*gAw*gBu/(a*a*a) - 8*E12*c1*fAw*fBuxi*gAwetaeta*gBu/(a*(b*b)) + E16*(-16*c1*fAwxi*fBuxi*gAweta*gBu/((a*a)*b) - 8*c1*fAwxixi*fBu*gAw*gBueta/((a*a)*b)) - 8*E26*c1*fAw*fBu*gAwetaeta*gBueta/(b*b*b) - 16*E66*c1*fAwxi*fBu*gAweta*gBueta/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 4*A12*fAwxi*fBv*gAw*gBveta*wx/(a*b) + 4*A16*fAwxi*fBvxi*gAw*gBv*wx/(a*a) + 4*A22*fAw*fBv*gAweta*gBveta*wy/(b*b) + A26*(4*fAw*fBv*gAweta*gBveta*wx/(b*b) + 4*fAw*fBvxi*gAweta*gBv*wy/(a*b) + 4*fAwxi*fBv*gAw*gBveta*wy/(a*b)) + A66*(4*fAw*fBvxi*gAweta*gBv*wx/(a*b) + 4*fAwxi*fBvxi*gAw*gBv*wy/(a*a)) - 8*E12*c1*fAwxixi*fBv*gAw*gBveta/((a*a)*b) - 8*E16*c1*fAwxixi*fBvxi*gAw*gBv/(a*a*a) - 8*E22*c1*fAw*fBv*gAwetaeta*gBveta/(b*b*b) + E26*(-8*c1*fAw*fBvxi*gAwetaeta*gBv/(a*(b*b)) - 16*c1*fAwxi*fBv*gAweta*gBveta/(a*(b*b))) - 16*E66*c1*fAwxi*fBvxi*gAweta*gBv/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 4*A11*fAwxi*fBwxi*gAw*gBw*(wx*wx)/(a*a) + A12*(4*fAw*fBwxi*gAweta*gBw*wx*wy/(a*b) + 4*fAwxi*fBw*gAw*gBweta*wx*wy/(a*b)) + A16*(4*fAw*fBwxi*gAweta*gBw*(wx*wx)/(a*b) + 4*fAwxi*fBw*gAw*gBweta*(wx*wx)/(a*b) + 8*fAwxi*fBwxi*gAw*gBw*wx*wy/(a*a)) + 4*A22*fAw*fBw*gAweta*gBweta*(wy*wy)/(b*b) + A26*(8*fAw*fBw*gAweta*gBweta*wx*wy/(b*b) + 4*fAw*fBwxi*gAweta*gBw*(wy*wy)/(a*b) + 4*fAwxi*fBw*gAw*gBweta*(wy*wy)/(a*b)) + 4*A44*fAw*fBw*gAweta*gBweta/(b*b) + A45*(4*fAw*fBwxi*gAweta*gBw/(a*b) + 4*fAwxi*fBw*gAw*gBweta/(a*b)) + 4*A55*fAwxi*fBwxi*gAw*gBw/(a*a) + A66*(4*fAw*fBw*gAweta*gBweta*(wx*wx)/(b*b) + 4*fAw*fBwxi*gAweta*gBw*wx*wy/(a*b) + 4*fAwxi*fBw*gAw*gBweta*wx*wy/(a*b) + 4*fAwxi*fBwxi*gAw*gBw*(wy*wy)/(a*a)) - 24*D44*c1*fAw*fBw*gAweta*gBweta/(b*b) + D45*(-24*c1*fAw*fBwxi*gAweta*gBw/(a*b) - 24*c1*fAwxi*fBw*gAw*gBweta/(a*b)) - 24*D55*c1*fAwxi*fBwxi*gAw*gBw/(a*a) + E11*(-8*c1*fAwxi*fBwxixi*gAw*gBw*wx/(a*a*a) - 8*c1*fAwxixi*fBwxi*gAw*gBw*wx/(a*a*a)) + E12*(-8*c1*fAw*fBwxi*gAwetaeta*gBw*wx/(a*(b*b)) - 8*c1*fAwxi*fBw*gAw*gBwetaeta*wx/(a*(b*b)) - 8*c1*fAw*fBwxixi*gAweta*gBw*wy/((a*a)*b) - 8*c1*fAwxixi*fBw*gAw*gBweta*wy/((a*a)*b)) + E16*(-8*c1*fAw*fBwxixi*gAweta*gBw*wx/((a*a)*b) - 16*c1*fAwxi*fBwxi*gAw*gBweta*wx/((a*a)*b) - 16*c1*fAwxi*fBwxi*gAweta*gBw*wx/((a*a)*b) - 8*c1*fAwxixi*fBw*gAw*gBweta*wx/((a*a)*b) - 8*c1*fAwxi*fBwxixi*gAw*gBw*wy/(a*a*a) - 8*c1*fAwxixi*fBwxi*gAw*gBw*wy/(a*a*a)) + E22*(-8*c1*fAw*fBw*gAweta*gBwetaeta*wy/(b*b*b) - 8*c1*fAw*fBw*gAwetaeta*gBweta*wy/(b*b*b)) + E26*(-8*c1*fAw*fBw*gAweta*gBwetaeta*wx/(b*b*b) - 8*c1*fAw*fBw*gAwetaeta*gBweta*wx/(b*b*b) - 16*c1*fAw*fBwxi*gAweta*gBweta*wy/(a*(b*b)) - 8*c1*fAw*fBwxi*gAwetaeta*gBw*wy/(a*(b*b)) - 8*c1*fAwxi*fBw*gAw*gBwetaeta*wy/(a*(b*b)) - 16*c1*fAwxi*fBw*gAweta*gBweta*wy/(a*(b*b))) + E66*(-16*c1*fAw*fBwxi*gAweta*gBweta*wx/(a*(b*b)) - 16*c1*fAwxi*fBw*gAweta*gBweta*wx/(a*(b*b)) - 16*c1*fAwxi*fBwxi*gAw*gBweta*wy/((a*a)*b) - 16*c1*fAwxi*fBwxi*gAweta*gBw*wy/((a*a)*b)) + 36*F44*(c1*c1)*fAw*fBw*gAweta*gBweta/(b*b) + F45*(36*(c1*c1)*fAw*fBwxi*gAweta*gBw/(a*b) + 36*(c1*c1)*fAwxi*fBw*gAw*gBweta/(a*b)) + 36*F55*(c1*c1)*fAwxi*fBwxi*gAw*gBw/(a*a) + 16*H11*(c1*c1)*fAwxixi*fBwxixi*gAw*gBw/(a*a*a*a) + H12*(16*(c1*c1)*fAw*fBwxixi*gAwetaeta*gBw/((a*a)*(b*b)) + 16*(c1*c1)*fAwxixi*fBw*gAw*gBwetaeta/((a*a)*(b*b))) + H16*(32*(c1*c1)*fAwxi*fBwxixi*gAweta*gBw/((a*a*a)*b) + 32*(c1*c1)*fAwxixi*fBwxi*gAw*gBweta/((a*a*a)*b)) + 16*H22*(c1*c1)*fAw*fBw*gAwetaeta*gBwetaeta/(b*b*b*b) + H26*(32*(c1*c1)*fAw*fBwxi*gAwetaeta*gBweta/(a*(b*b*b)) + 32*(c1*c1)*fAwxi*fBw*gAweta*gBwetaeta/(a*(b*b*b))) + 64*H66*(c1*c1)*fAwxi*fBwxi*gAweta*gBweta/((a*a)*(b*b)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*( 4*NxxNL*fAwxi*fBwxi*gAw*gBw/(a*a) + 4*NxyNL*(fAw*fBwxi*gAweta*gBw + fAwxi*fBw*gAw*gBweta)/(a*b) + 4*NyyNL*fAw*fBw*gAweta*gBweta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( 2*A45*fAw*fBphix*gAweta*gBphix/b + 2*A55*fAwxi*fBphix*gAw*gBphix/a + 4*B11*fAwxi*fBphixxi*gAw*gBphix*wx/(a*a) + 4*B12*fAw*fBphixxi*gAweta*gBphix*wy/(a*b) + B16*(4*fAw*fBphixxi*gAweta*gBphix*wx/(a*b) + 4*fAwxi*fBphix*gAw*gBphixeta*wx/(a*b) + 4*fAwxi*fBphixxi*gAw*gBphix*wy/(a*a)) + 4*B26*fAw*fBphix*gAweta*gBphixeta*wy/(b*b) + B66*(4*fAw*fBphix*gAweta*gBphixeta*wx/(b*b) + 4*fAwxi*fBphix*gAw*gBphixeta*wy/(a*b)) - 12*D45*c1*fAw*fBphix*gAweta*gBphix/b - 12*D55*c1*fAwxi*fBphix*gAw*gBphix/a - 4*E11*c1*fAwxi*fBphixxi*gAw*gBphix*wx/(a*a) - 4*E12*c1*fAw*fBphixxi*gAweta*gBphix*wy/(a*b) + E16*(-4*c1*fAw*fBphixxi*gAweta*gBphix*wx/(a*b) - 4*c1*fAwxi*fBphix*gAw*gBphixeta*wx/(a*b) - 4*c1*fAwxi*fBphixxi*gAw*gBphix*wy/(a*a)) - 4*E26*c1*fAw*fBphix*gAweta*gBphixeta*wy/(b*b) + E66*(-4*c1*fAw*fBphix*gAweta*gBphixeta*wx/(b*b) - 4*c1*fAwxi*fBphix*gAw*gBphixeta*wy/(a*b)) - 8*F11*c1*fAwxixi*fBphixxi*gAw*gBphix/(a*a*a) - 8*F12*c1*fAw*fBphixxi*gAwetaeta*gBphix/(a*(b*b)) + F16*(-16*c1*fAwxi*fBphixxi*gAweta*gBphix/((a*a)*b) - 8*c1*fAwxixi*fBphix*gAw*gBphixeta/((a*a)*b)) - 8*F26*c1*fAw*fBphix*gAwetaeta*gBphixeta/(b*b*b) + 18*F45*(c1*c1)*fAw*fBphix*gAweta*gBphix/b + 18*F55*(c1*c1)*fAwxi*fBphix*gAw*gBphix/a - 16*F66*c1*fAwxi*fBphix*gAweta*gBphixeta/(a*(b*b)) + 8*H11*(c1*c1)*fAwxixi*fBphixxi*gAw*gBphix/(a*a*a) + 8*H12*(c1*c1)*fAw*fBphixxi*gAwetaeta*gBphix/(a*(b*b)) + H16*(16*(c1*c1)*fAwxi*fBphixxi*gAweta*gBphix/((a*a)*b) + 8*(c1*c1)*fAwxixi*fBphix*gAw*gBphixeta/((a*a)*b)) + 8*H26*(c1*c1)*fAw*fBphix*gAwetaeta*gBphixeta/(b*b*b) + 16*H66*(c1*c1)*fAwxi*fBphix*gAweta*gBphixeta/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( 2*A44*fAw*fBphiy*gAweta*gBphiy/b + 2*A45*fAwxi*fBphiy*gAw*gBphiy/a + 4*B12*fAwxi*fBphiy*gAw*gBphiyeta*wx/(a*b) + 4*B16*fAwxi*fBphiyxi*gAw*gBphiy*wx/(a*a) + 4*B22*fAw*fBphiy*gAweta*gBphiyeta*wy/(b*b) + B26*(4*fAw*fBphiy*gAweta*gBphiyeta*wx/(b*b) + 4*fAw*fBphiyxi*gAweta*gBphiy*wy/(a*b) + 4*fAwxi*fBphiy*gAw*gBphiyeta*wy/(a*b)) + B66*(4*fAw*fBphiyxi*gAweta*gBphiy*wx/(a*b) + 4*fAwxi*fBphiyxi*gAw*gBphiy*wy/(a*a)) - 12*D44*c1*fAw*fBphiy*gAweta*gBphiy/b - 12*D45*c1*fAwxi*fBphiy*gAw*gBphiy/a - 4*E12*c1*fAwxi*fBphiy*gAw*gBphiyeta*wx/(a*b) - 4*E16*c1*fAwxi*fBphiyxi*gAw*gBphiy*wx/(a*a) - 4*E22*c1*fAw*fBphiy*gAweta*gBphiyeta*wy/(b*b) + E26*(-4*c1*fAw*fBphiy*gAweta*gBphiyeta*wx/(b*b) - 4*c1*fAw*fBphiyxi*gAweta*gBphiy*wy/(a*b) - 4*c1*fAwxi*fBphiy*gAw*gBphiyeta*wy/(a*b)) + E66*(-4*c1*fAw*fBphiyxi*gAweta*gBphiy*wx/(a*b) - 4*c1*fAwxi*fBphiyxi*gAw*gBphiy*wy/(a*a)) - 8*F12*c1*fAwxixi*fBphiy*gAw*gBphiyeta/((a*a)*b) - 8*F16*c1*fAwxixi*fBphiyxi*gAw*gBphiy/(a*a*a) - 8*F22*c1*fAw*fBphiy*gAwetaeta*gBphiyeta/(b*b*b) + F26*(-8*c1*fAw*fBphiyxi*gAwetaeta*gBphiy/(a*(b*b)) - 16*c1*fAwxi*fBphiy*gAweta*gBphiyeta/(a*(b*b))) + 18*F44*(c1*c1)*fAw*fBphiy*gAweta*gBphiy/b + 18*F45*(c1*c1)*fAwxi*fBphiy*gAw*gBphiy/a - 16*F66*c1*fAwxi*fBphiyxi*gAweta*gBphiy/((a*a)*b) + 8*H12*(c1*c1)*fAwxixi*fBphiy*gAw*gBphiyeta/((a*a)*b) + 8*H16*(c1*c1)*fAwxixi*fBphiyxi*gAw*gBphiy/(a*a*a) + 8*H22*(c1*c1)*fAw*fBphiy*gAwetaeta*gBphiyeta/(b*b*b) + H26*(8*(c1*c1)*fAw*fBphiyxi*gAwetaeta*gBphiy/(a*(b*b)) + 16*(c1*c1)*fAwxi*fBphiy*gAweta*gBphiyeta/(a*(b*b))) + 16*H66*(c1*c1)*fAwxi*fBphiyxi*gAweta*gBphiy/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*B11*fAphixxi*fBuxi*gAphix*gBu/(a*a) + B16*(4*fAphix*fBuxi*gAphixeta*gBu/(a*b) + 4*fAphixxi*fBu*gAphix*gBueta/(a*b)) + 4*B66*fAphix*fBu*gAphixeta*gBueta/(b*b) - 4*E11*c1*fAphixxi*fBuxi*gAphix*gBu/(a*a) + E16*(-4*c1*fAphix*fBuxi*gAphixeta*gBu/(a*b) - 4*c1*fAphixxi*fBu*gAphix*gBueta/(a*b)) - 4*E66*c1*fAphix*fBu*gAphixeta*gBueta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAphixxi*fBv*gAphix*gBveta/(a*b) + 4*B16*fAphixxi*fBvxi*gAphix*gBv/(a*a) + 4*B26*fAphix*fBv*gAphixeta*gBveta/(b*b) + 4*B66*fAphix*fBvxi*gAphixeta*gBv/(a*b) - 4*E12*c1*fAphixxi*fBv*gAphix*gBveta/(a*b) - 4*E16*c1*fAphixxi*fBvxi*gAphix*gBv/(a*a) - 4*E26*c1*fAphix*fBv*gAphixeta*gBveta/(b*b) - 4*E66*c1*fAphix*fBvxi*gAphixeta*gBv/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 2*A45*fAphix*fBw*gAphix*gBweta/b + 2*A55*fAphix*fBwxi*gAphix*gBw/a + 4*B11*fAphixxi*fBwxi*gAphix*gBw*wx/(a*a) + 4*B12*fAphixxi*fBw*gAphix*gBweta*wy/(a*b) + B16*(4*fAphix*fBwxi*gAphixeta*gBw*wx/(a*b) + 4*fAphixxi*fBw*gAphix*gBweta*wx/(a*b) + 4*fAphixxi*fBwxi*gAphix*gBw*wy/(a*a)) + 4*B26*fAphix*fBw*gAphixeta*gBweta*wy/(b*b) + B66*(4*fAphix*fBw*gAphixeta*gBweta*wx/(b*b) + 4*fAphix*fBwxi*gAphixeta*gBw*wy/(a*b)) - 12*D45*c1*fAphix*fBw*gAphix*gBweta/b - 12*D55*c1*fAphix*fBwxi*gAphix*gBw/a - 4*E11*c1*fAphixxi*fBwxi*gAphix*gBw*wx/(a*a) - 4*E12*c1*fAphixxi*fBw*gAphix*gBweta*wy/(a*b) + E16*(-4*c1*fAphix*fBwxi*gAphixeta*gBw*wx/(a*b) - 4*c1*fAphixxi*fBw*gAphix*gBweta*wx/(a*b) - 4*c1*fAphixxi*fBwxi*gAphix*gBw*wy/(a*a)) - 4*E26*c1*fAphix*fBw*gAphixeta*gBweta*wy/(b*b) + E66*(-4*c1*fAphix*fBw*gAphixeta*gBweta*wx/(b*b) - 4*c1*fAphix*fBwxi*gAphixeta*gBw*wy/(a*b)) - 8*F11*c1*fAphixxi*fBwxixi*gAphix*gBw/(a*a*a) - 8*F12*c1*fAphixxi*fBw*gAphix*gBwetaeta/(a*(b*b)) + F16*(-8*c1*fAphix*fBwxixi*gAphixeta*gBw/((a*a)*b) - 16*c1*fAphixxi*fBwxi*gAphix*gBweta/((a*a)*b)) - 8*F26*c1*fAphix*fBw*gAphixeta*gBwetaeta/(b*b*b) + 18*F45*(c1*c1)*fAphix*fBw*gAphix*gBweta/b + 18*F55*(c1*c1)*fAphix*fBwxi*gAphix*gBw/a - 16*F66*c1*fAphix*fBwxi*gAphixeta*gBweta/(a*(b*b)) + 8*H11*(c1*c1)*fAphixxi*fBwxixi*gAphix*gBw/(a*a*a) + 8*H12*(c1*c1)*fAphixxi*fBw*gAphix*gBwetaeta/(a*(b*b)) + H16*(8*(c1*c1)*fAphix*fBwxixi*gAphixeta*gBw/((a*a)*b) + 16*(c1*c1)*fAphixxi*fBwxi*gAphix*gBweta/((a*a)*b)) + 8*H26*(c1*c1)*fAphix*fBw*gAphixeta*gBwetaeta/(b*b*b) + 16*H66*(c1*c1)*fAphix*fBwxi*gAphixeta*gBweta/(a*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( A55*fAphix*fBphix*gAphix*gBphix + 4*D11*fAphixxi*fBphixxi*gAphix*gBphix/(a*a) + D16*(4*fAphix*fBphixxi*gAphixeta*gBphix/(a*b) + 4*fAphixxi*fBphix*gAphix*gBphixeta/(a*b)) - 6*D55*c1*fAphix*fBphix*gAphix*gBphix + 4*D66*fAphix*fBphix*gAphixeta*gBphixeta/(b*b) - 8*F11*c1*fAphixxi*fBphixxi*gAphix*gBphix/(a*a) + F16*(-8*c1*fAphix*fBphixxi*gAphixeta*gBphix/(a*b) - 8*c1*fAphixxi*fBphix*gAphix*gBphixeta/(a*b)) + 9*F55*(c1*c1)*fAphix*fBphix*gAphix*gBphix - 8*F66*c1*fAphix*fBphix*gAphixeta*gBphixeta/(b*b) + 4*H11*(c1*c1)*fAphixxi*fBphixxi*gAphix*gBphix/(a*a) + H16*(4*(c1*c1)*fAphix*fBphixxi*gAphixeta*gBphix/(a*b) + 4*(c1*c1)*fAphixxi*fBphix*gAphix*gBphixeta/(a*b)) + 4*H66*(c1*c1)*fAphix*fBphix*gAphixeta*gBphixeta/(b*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( A45*fAphix*fBphiy*gAphix*gBphiy + 4*D12*fAphixxi*fBphiy*gAphix*gBphiyeta/(a*b) + 4*D16*fAphixxi*fBphiyxi*gAphix*gBphiy/(a*a) + 4*D26*fAphix*fBphiy*gAphixeta*gBphiyeta/(b*b) - 6*D45*c1*fAphix*fBphiy*gAphix*gBphiy + 4*D66*fAphix*fBphiyxi*gAphixeta*gBphiy/(a*b) - 8*F12*c1*fAphixxi*fBphiy*gAphix*gBphiyeta/(a*b) - 8*F16*c1*fAphixxi*fBphiyxi*gAphix*gBphiy/(a*a) - 8*F26*c1*fAphix*fBphiy*gAphixeta*gBphiyeta/(b*b) + 9*F45*(c1*c1)*fAphix*fBphiy*gAphix*gBphiy - 8*F66*c1*fAphix*fBphiyxi*gAphixeta*gBphiy/(a*b) + 4*H12*(c1*c1)*fAphixxi*fBphiy*gAphix*gBphiyeta/(a*b) + 4*H16*(c1*c1)*fAphixxi*fBphiyxi*gAphix*gBphiy/(a*a) + 4*H26*(c1*c1)*fAphix*fBphiy*gAphixeta*gBphiyeta/(b*b) + 4*H66*(c1*c1)*fAphix*fBphiyxi*gAphixeta*gBphiy/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(intx*inty/4)*( 4*B12*fAphiy*fBuxi*gAphiyeta*gBu/(a*b) + 4*B16*fAphiyxi*fBuxi*gAphiy*gBu/(a*a) + 4*B26*fAphiy*fBu*gAphiyeta*gBueta/(b*b) + 4*B66*fAphiyxi*fBu*gAphiy*gBueta/(a*b) - 4*E12*c1*fAphiy*fBuxi*gAphiyeta*gBu/(a*b) - 4*E16*c1*fAphiyxi*fBuxi*gAphiy*gBu/(a*a) - 4*E26*c1*fAphiy*fBu*gAphiyeta*gBueta/(b*b) - 4*E66*c1*fAphiyxi*fBu*gAphiy*gBueta/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(intx*inty/4)*( 4*B22*fAphiy*fBv*gAphiyeta*gBveta/(b*b) + B26*(4*fAphiy*fBvxi*gAphiyeta*gBv/(a*b) + 4*fAphiyxi*fBv*gAphiy*gBveta/(a*b)) + 4*B66*fAphiyxi*fBvxi*gAphiy*gBv/(a*a) - 4*E22*c1*fAphiy*fBv*gAphiyeta*gBveta/(b*b) + E26*(-4*c1*fAphiy*fBvxi*gAphiyeta*gBv/(a*b) - 4*c1*fAphiyxi*fBv*gAphiy*gBveta/(a*b)) - 4*E66*c1*fAphiyxi*fBvxi*gAphiy*gBv/(a*a) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(intx*inty/4)*( 2*A44*fAphiy*fBw*gAphiy*gBweta/b + 2*A45*fAphiy*fBwxi*gAphiy*gBw/a + 4*B12*fAphiy*fBwxi*gAphiyeta*gBw*wx/(a*b) + 4*B16*fAphiyxi*fBwxi*gAphiy*gBw*wx/(a*a) + 4*B22*fAphiy*fBw*gAphiyeta*gBweta*wy/(b*b) + B26*(4*fAphiy*fBw*gAphiyeta*gBweta*wx/(b*b) + 4*fAphiy*fBwxi*gAphiyeta*gBw*wy/(a*b) + 4*fAphiyxi*fBw*gAphiy*gBweta*wy/(a*b)) + B66*(4*fAphiyxi*fBw*gAphiy*gBweta*wx/(a*b) + 4*fAphiyxi*fBwxi*gAphiy*gBw*wy/(a*a)) - 12*D44*c1*fAphiy*fBw*gAphiy*gBweta/b - 12*D45*c1*fAphiy*fBwxi*gAphiy*gBw/a - 4*E12*c1*fAphiy*fBwxi*gAphiyeta*gBw*wx/(a*b) - 4*E16*c1*fAphiyxi*fBwxi*gAphiy*gBw*wx/(a*a) - 4*E22*c1*fAphiy*fBw*gAphiyeta*gBweta*wy/(b*b) + E26*(-4*c1*fAphiy*fBw*gAphiyeta*gBweta*wx/(b*b) - 4*c1*fAphiy*fBwxi*gAphiyeta*gBw*wy/(a*b) - 4*c1*fAphiyxi*fBw*gAphiy*gBweta*wy/(a*b)) + E66*(-4*c1*fAphiyxi*fBw*gAphiy*gBweta*wx/(a*b) - 4*c1*fAphiyxi*fBwxi*gAphiy*gBw*wy/(a*a)) - 8*F12*c1*fAphiy*fBwxixi*gAphiyeta*gBw/((a*a)*b) - 8*F16*c1*fAphiyxi*fBwxixi*gAphiy*gBw/(a*a*a) - 8*F22*c1*fAphiy*fBw*gAphiyeta*gBwetaeta/(b*b*b) + F26*(-16*c1*fAphiy*fBwxi*gAphiyeta*gBweta/(a*(b*b)) - 8*c1*fAphiyxi*fBw*gAphiy*gBwetaeta/(a*(b*b))) + 18*F44*(c1*c1)*fAphiy*fBw*gAphiy*gBweta/b + 18*F45*(c1*c1)*fAphiy*fBwxi*gAphiy*gBw/a - 16*F66*c1*fAphiyxi*fBwxi*gAphiy*gBweta/((a*a)*b) + 8*H12*(c1*c1)*fAphiy*fBwxixi*gAphiyeta*gBw/((a*a)*b) + 8*H16*(c1*c1)*fAphiyxi*fBwxixi*gAphiy*gBw/(a*a*a) + 8*H22*(c1*c1)*fAphiy*fBw*gAphiyeta*gBwetaeta/(b*b*b) + H26*(16*(c1*c1)*fAphiy*fBwxi*gAphiyeta*gBweta/(a*(b*b)) + 8*(c1*c1)*fAphiyxi*fBw*gAphiy*gBwetaeta/(a*(b*b))) + 16*H66*(c1*c1)*fAphiyxi*fBwxi*gAphiy*gBweta/((a*a)*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(intx*inty/4)*( A45*fAphiy*fBphix*gAphiy*gBphix + 4*D12*fAphiy*fBphixxi*gAphiyeta*gBphix/(a*b) + 4*D16*fAphiyxi*fBphixxi*gAphiy*gBphix/(a*a) + 4*D26*fAphiy*fBphix*gAphiyeta*gBphixeta/(b*b) - 6*D45*c1*fAphiy*fBphix*gAphiy*gBphix + 4*D66*fAphiyxi*fBphix*gAphiy*gBphixeta/(a*b) - 8*F12*c1*fAphiy*fBphixxi*gAphiyeta*gBphix/(a*b) - 8*F16*c1*fAphiyxi*fBphixxi*gAphiy*gBphix/(a*a) - 8*F26*c1*fAphiy*fBphix*gAphiyeta*gBphixeta/(b*b) + 9*F45*(c1*c1)*fAphiy*fBphix*gAphiy*gBphix - 8*F66*c1*fAphiyxi*fBphix*gAphiy*gBphixeta/(a*b) + 4*H12*(c1*c1)*fAphiy*fBphixxi*gAphiyeta*gBphix/(a*b) + 4*H16*(c1*c1)*fAphiyxi*fBphixxi*gAphiy*gBphix/(a*a) + 4*H26*(c1*c1)*fAphiy*fBphix*gAphiyeta*gBphixeta/(b*b) + 4*H66*(c1*c1)*fAphiyxi*fBphix*gAphiy*gBphixeta/(a*b) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(intx*inty/4)*( A44*fAphiy*fBphiy*gAphiy*gBphiy + 4*D22*fAphiy*fBphiy*gAphiyeta*gBphiyeta/(b*b) + D26*(4*fAphiy*fBphiyxi*gAphiyeta*gBphiy/(a*b) + 4*fAphiyxi*fBphiy*gAphiy*gBphiyeta/(a*b)) - 6*D44*c1*fAphiy*fBphiy*gAphiy*gBphiy + 4*D66*fAphiyxi*fBphiyxi*gAphiy*gBphiy/(a*a) - 8*F22*c1*fAphiy*fBphiy*gAphiyeta*gBphiyeta/(b*b) + F26*(-8*c1*fAphiy*fBphiyxi*gAphiyeta*gBphiy/(a*b) - 8*c1*fAphiyxi*fBphiy*gAphiy*gBphiyeta/(a*b)) + 9*F44*(c1*c1)*fAphiy*fBphiy*gAphiy*gBphiy - 8*F66*c1*fAphiyxi*fBphiyxi*gAphiy*gBphiy/(a*a) + 4*H22*(c1*c1)*fAphiy*fBphiy*gAphiyeta*gBphiyeta/(b*b) + H26*(4*(c1*c1)*fAphiy*fBphiyxi*gAphiyeta*gBphiy/(a*b) + 4*(c1*c1)*fAphiyxi*fBphiy*gAphiy*gBphiyeta/(a*b)) + 4*H66*(c1*c1)*fAphiyxi*fBphiyxi*gAphiy*gBphiy/(a*a) )

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

    cdef double fAu, fAuxi, fAv, fAvxi, fAw
    cdef double fAwxi, fAwxixi, fAphix, fAphixxi, fAphiy
    cdef double fAphiyxi
    cdef double fBw, fBwxi
    cdef double gAu, gAueta, gAv, gAveta, gAw
    cdef double gAweta, gAwetaeta, gAphix, gAphixeta, gAphiy
    cdef double gAphiyeta
    cdef double gBw, gBweta

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
                        e[6] += -2*c1*cs[col+3]*fAphixxi*gAphix/a - 4*c1*cs[col+2]*fAwxixi*gAw/(a*a)
                        e[7] += -2*c1*cs[col+4]*fAphiy*gAphiyeta/b - 4*c1*cs[col+2]*fAw*gAwetaeta/(b*b)
                        e[8] += -2*c1*cs[col+3]*fAphix*gAphixeta/b - 2*c1*cs[col+4]*fAphiyxi*gAphiy/a - 8*c1*cs[col+2]*fAwxi*gAweta/(a*b)
                        e[9] += cs[col+4]*fAphiy*gAphiy + 2*cs[col+2]*fAw*gAweta/b
                        e[10] += cs[col+3]*fAphix*gAphix + 2*cs[col+2]*fAwxi*gAw/a
                        e[11] += -3*c1*cs[col+4]*fAphiy*gAphiy - 6*c1*cs[col+2]*fAw*gAweta/b
                        e[12] += -3*c1*cs[col+3]*fAphix*gAphix - 6*c1*cs[col+2]*fAwxi*gAw/a

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

    cdef double fAu, fAv, fAw, fAwxi, fAphix
    cdef double fAphiy
    cdef double fBu, fBv, fBw, fBwxi, fBphix
    cdef double fBphiy
    cdef double gAu, gAv, gAw, gAweta, gAphix
    cdef double gAphiy
    cdef double gBu, gBv, gBw, gBweta, gBphix
    cdef double gBphiy
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
                c1 = 4./(3.*h*h)

                # kM
                c = -1
                for i in range(m):
                    fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                    fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                    fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                    fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    fAphix = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    fAphiy = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                    for k in range(m):
                        fBu = f(k, xi, x1u, x1ur, x2u, x2ur)
                        fBv = f(k, xi, x1v, x1vr, x2v, x2vr)
                        fBw = f(k, xi, x1w, x1wr, x2w, x2wr)
                        fBwxi = fp(k, xi, x1w, x1wr, x2w, x2wr)
                        fBphix = f(k, xi, x1phix, x1phixr, x2phix, x2phixr)
                        fBphiy = f(k, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)

                        for j in range(n):
                            gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                            gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                            gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                            gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
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
                                gBweta = fp(l, eta, y1w, y1wr, y2w, y2wr)
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
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( 2*c1*(d*d*d)*fAu*fBwxi*gAu*gBw*h*rho/a + 0.5*c1*d*fAu*fBwxi*gAu*gBw*(h*h*h)*rho/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( c1*(d*d*d)*fAu*fBphix*gAu*gBphix*h*rho + d*(0.25*c1*fAu*fBphix*gAu*gBphix*(h*h*h)*rho - fAu*fBphix*gAu*gBphix*h*rho) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( fAv*fBv*gAv*gBv*h*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( 2*c1*(d*d*d)*fAv*fBw*gAv*gBweta*h*rho/b + 0.5*c1*d*fAv*fBw*gAv*gBweta*(h*h*h)*rho/b )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( c1*(d*d*d)*fAv*fBphiy*gAv*gBphiy*h*rho + d*(0.25*c1*fAv*fBphiy*gAv*gBphiy*(h*h*h)*rho - fAv*fBphiy*gAv*gBphiy*h*rho) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( 2*c1*(d*d*d)*fAwxi*fBu*gAw*gBu*h*rho/a + 0.5*c1*d*fAwxi*fBu*gAw*gBu*(h*h*h)*rho/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( 2*c1*(d*d*d)*fAw*fBv*gAweta*gBv*h*rho/b + 0.5*c1*d*fAw*fBv*gAweta*gBv*(h*h*h)*rho/b )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( (d*d*d*d*d*d)*(4*(c1*c1)*fAw*fBw*gAweta*gBweta*h*rho/(b*b) + 4*(c1*c1)*fAwxi*fBwxi*gAw*gBw*h*rho/(a*a)) + (d*d*d*d)*(5*(c1*c1)*fAw*fBw*gAweta*gBweta*(h*h*h)*rho/(b*b) + 5*(c1*c1)*fAwxi*fBwxi*gAw*gBw*(h*h*h)*rho/(a*a)) + (d*d)*(0.75*(c1*c1)*fAw*fBw*gAweta*gBweta*(h*h*h*h*h)*rho/(b*b) + 0.75*(c1*c1)*fAwxi*fBwxi*gAw*gBw*(h*h*h*h*h)*rho/(a*a)) + fAw*fBw*gAw*gBw*h*rho + 0.008928571428571428*(c1*c1)*fAw*fBw*gAweta*gBweta*(h*h*h*h*h*h*h)*rho/(b*b) + 0.008928571428571428*(c1*c1)*fAwxi*fBwxi*gAw*gBw*(h*h*h*h*h*h*h)*rho/(a*a) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( (d*d*d*d)*(2.5*(c1*c1)*fAwxi*fBphix*gAw*gBphix*(h*h*h)*rho/a - 2*c1*fAwxi*fBphix*gAw*gBphix*h*rho/a) + (d*d)*(0.375*(c1*c1)*fAwxi*fBphix*gAw*gBphix*(h*h*h*h*h)*rho/a - c1*fAwxi*fBphix*gAw*gBphix*(h*h*h)*rho/a) + 2*(c1*c1)*(d*d*d*d*d*d)*fAwxi*fBphix*gAw*gBphix*h*rho/a + 0.004464285714285714*(c1*c1)*fAwxi*fBphix*gAw*gBphix*(h*h*h*h*h*h*h)*rho/a - 0.025*c1*fAwxi*fBphix*gAw*gBphix*(h*h*h*h*h)*rho/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( (d*d*d*d)*(2.5*(c1*c1)*fAw*fBphiy*gAweta*gBphiy*(h*h*h)*rho/b - 2*c1*fAw*fBphiy*gAweta*gBphiy*h*rho/b) + (d*d)*(0.375*(c1*c1)*fAw*fBphiy*gAweta*gBphiy*(h*h*h*h*h)*rho/b - c1*fAw*fBphiy*gAweta*gBphiy*(h*h*h)*rho/b) + 2*(c1*c1)*(d*d*d*d*d*d)*fAw*fBphiy*gAweta*gBphiy*h*rho/b + 0.004464285714285714*(c1*c1)*fAw*fBphiy*gAweta*gBphiy*(h*h*h*h*h*h*h)*rho/b - 0.025*c1*fAw*fBphiy*gAweta*gBphiy*(h*h*h*h*h)*rho/b )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( c1*(d*d*d)*fAphix*fBu*gAphix*gBu*h*rho + d*(0.25*c1*fAphix*fBu*gAphix*gBu*(h*h*h)*rho - fAphix*fBu*gAphix*gBu*h*rho) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( (d*d*d*d)*(2.5*(c1*c1)*fAphix*fBwxi*gAphix*gBw*(h*h*h)*rho/a - 2*c1*fAphix*fBwxi*gAphix*gBw*h*rho/a) + (d*d)*(0.375*(c1*c1)*fAphix*fBwxi*gAphix*gBw*(h*h*h*h*h)*rho/a - c1*fAphix*fBwxi*gAphix*gBw*(h*h*h)*rho/a) + 2*(c1*c1)*(d*d*d*d*d*d)*fAphix*fBwxi*gAphix*gBw*h*rho/a + 0.004464285714285714*(c1*c1)*fAphix*fBwxi*gAphix*gBw*(h*h*h*h*h*h*h)*rho/a - 0.025*c1*fAphix*fBwxi*gAphix*gBw*(h*h*h*h*h)*rho/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( (c1*c1)*(d*d*d*d*d*d)*fAphix*fBphix*gAphix*gBphix*h*rho + 0.002232142857142857*(c1*c1)*fAphix*fBphix*gAphix*gBphix*(h*h*h*h*h*h*h)*rho - 0.025*c1*fAphix*fBphix*gAphix*gBphix*(h*h*h*h*h)*rho + (d*d*d*d)*(1.25*(c1*c1)*fAphix*fBphix*gAphix*gBphix*(h*h*h)*rho - 2*c1*fAphix*fBphix*gAphix*gBphix*h*rho) + (d*d)*(0.1875*(c1*c1)*fAphix*fBphix*gAphix*gBphix*(h*h*h*h*h)*rho - c1*fAphix*fBphix*gAphix*gBphix*(h*h*h)*rho + fAphix*fBphix*gAphix*gBphix*h*rho) + 0.08333333333333333*fAphix*fBphix*gAphix*gBphix*(h*h*h)*rho )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( c1*(d*d*d)*fAphiy*fBv*gAphiy*gBv*h*rho + d*(0.25*c1*fAphiy*fBv*gAphiy*gBv*(h*h*h)*rho - fAphiy*fBv*gAphiy*gBv*h*rho) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( (d*d*d*d)*(2.5*(c1*c1)*fAphiy*fBw*gAphiy*gBweta*(h*h*h)*rho/b - 2*c1*fAphiy*fBw*gAphiy*gBweta*h*rho/b) + (d*d)*(0.375*(c1*c1)*fAphiy*fBw*gAphiy*gBweta*(h*h*h*h*h)*rho/b - c1*fAphiy*fBw*gAphiy*gBweta*(h*h*h)*rho/b) + 2*(c1*c1)*(d*d*d*d*d*d)*fAphiy*fBw*gAphiy*gBweta*h*rho/b + 0.004464285714285714*(c1*c1)*fAphiy*fBw*gAphiy*gBweta*(h*h*h*h*h*h*h)*rho/b - 0.025*c1*fAphiy*fBw*gAphiy*gBweta*(h*h*h*h*h)*rho/b )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( (c1*c1)*(d*d*d*d*d*d)*fAphiy*fBphiy*gAphiy*gBphiy*h*rho + 0.002232142857142857*(c1*c1)*fAphiy*fBphiy*gAphiy*gBphiy*(h*h*h*h*h*h*h)*rho - 0.025*c1*fAphiy*fBphiy*gAphiy*gBphiy*(h*h*h*h*h)*rho + (d*d*d*d)*(1.25*(c1*c1)*fAphiy*fBphiy*gAphiy*gBphiy*(h*h*h)*rho - 2*c1*fAphiy*fBphiy*gAphiy*gBphiy*h*rho) + (d*d)*(0.1875*(c1*c1)*fAphiy*fBphiy*gAphiy*gBphiy*(h*h*h*h*h)*rho - c1*fAphiy*fBphiy*gAphiy*gBphiy*(h*h*h)*rho + fAphiy*fBphiy*gAphiy*gBphiy*h*rho) + 0.08333333333333333*fAphiy*fBphiy*gAphiy*gBphiy*(h*h*h)*rho )

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
    cdef double wx, wy

    cdef double fAu, fAuxi, fAv, fAvxi, fAw
    cdef double fAwxi, fAwxixi, fAphix, fAphixxi, fAphiy
    cdef double fAphiyxi
    cdef double gAu, gAueta, gAv, gAveta, gAw
    cdef double gAweta, gAwetaeta, gAphix, gAphixeta, gAphiy
    cdef double gAphiyeta

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

                wx = 0
                wy = 0
                for j in range(n):
                    #TODO save in buffer
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    for i in range(m):
                        #TODO save in buffer
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

                        wx += (2/a)*cs[col+2]*fAwxi*gAw
                        wy += (2/b)*cs[col+2]*fAw*gAweta

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
                        e[6] += -2*c1*cs[col+3]*fAphixxi*gAphix/a - 4*c1*cs[col+2]*fAwxixi*gAw/(a*a)
                        e[7] += -2*c1*cs[col+4]*fAphiy*gAphiyeta/b - 4*c1*cs[col+2]*fAw*gAwetaeta/(b*b)
                        e[8] += -2*c1*cs[col+3]*fAphix*gAphixeta/b - 2*c1*cs[col+4]*fAphiyxi*gAphiy/a - 8*c1*cs[col+2]*fAwxi*gAweta/(a*b)
                        e[9] += cs[col+4]*fAphiy*gAphiy + 2*cs[col+2]*fAw*gAweta/b
                        e[10] += cs[col+3]*fAphix*gAphix + 2*cs[col+2]*fAwxi*gAw/a
                        e[11] += -3*c1*cs[col+4]*fAphiy*gAphiy - 6*c1*cs[col+2]*fAw*gAweta/b
                        e[12] += -3*c1*cs[col+3]*fAphix*gAphix - 6*c1*cs[col+2]*fAwxi*gAw/a

                # nonlinear strain eps_NL = {w,x^2/2, w,y^2/2, w,x*w,y}
                e[0] += 0.5*wx*wx
                e[1] += 0.5*wy*wy
                e[2] += wx*wy

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
                        fint[col+2] += weight*(intx*inty/4)*( 2*Nxx*fAwxi*gAw*wx/a + 2*Nxy*(a*fAw*gAweta*wx + b*fAwxi*gAw*wy)/(a*b) + 2*Nyy*fAw*gAweta*wy/b - 4*Pxx*c1*fAwxixi*gAw/(a*a) - 8*Pxy*c1*fAwxi*gAweta/(a*b) - 4*Pyy*c1*fAw*gAwetaeta/(b*b) + 2*Qx*fAwxi*gAw/a + 2*Qy*fAw*gAweta/b - 6*Rx*c1*fAwxi*gAw/a - 6*Ry*c1*fAw*gAweta/b )
                        fint[col+3] += weight*(intx*inty/4)*( 2*Mxx*fAphixxi*gAphix/a + 2*Mxy*fAphix*gAphixeta/b - 2*Pxx*c1*fAphixxi*gAphix/a - 2*Pxy*c1*fAphix*gAphixeta/b + Qx*fAphix*gAphix - 3*Rx*c1*fAphix*gAphix )
                        fint[col+4] += weight*(intx*inty/4)*( 2*Mxy*fAphiyxi*gAphiy/a + 2*Myy*fAphiy*gAphiyeta/b - 2*Pxy*c1*fAphiyxi*gAphiy/a - 2*Pyy*c1*fAphiy*gAphiyeta/b + Qy*fAphiy*gAphiy - 3*Ry*c1*fAphiy*gAphiy )

    return fint
