#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
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

    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw, fAw_xi, fAw_xixi
    cdef double fBu, fBu_xi, fBv, fBv_xi, fBw, fBw_xi, fBw_xixi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw, gAw_eta, gAw_etaeta
    cdef double gBu, gBu_eta, gBv, gBv_eta, gBw, gBw_eta, gBw_etaeta
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double w_xi, w_eta
    cdef double w_x, w_y, NxxNL, NyyNL, NxyNL

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

                w_xi = 0
                w_eta = 0
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[1, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[2, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[3, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[4, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[5, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[1, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[2, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[3, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[4, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[5, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                if NLgeom == 1:
                    for j in range(n):
                        gAw = gtab[0, j]
                        gAw_eta = gtab[1, j]
                        for i in range(m):
                            fAw = ftab[0, i]
                            fAw_xi = ftab[1, i]

                            col = col0 + DOF*(j*m + i)

                            w_xi += cs[col+2]*fAw_xi*gAw
                            w_eta += cs[col+2]*fAw*gAw_eta

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
                # eps_NL = {w,x^2/2, w,y^2/2, w,x*w,y}. With it, KGNL = KG(N_NL)
                # is collected in kC such that
                #     KT = KC0 + KCNL + KGNL (fkC_num) + KG(Nbar + N_L) (fkG_num)
                # is the exact Jacobian of calc_fint, and fkG_num stays
                # homogeneous of degree one in cs, as linear buckling requires.
                # w_xi = w_eta = 0 when NLgeom == 0, then KGNL vanishes
                w_x = (2/a)*w_xi
                w_y = (2/b)*w_eta
                NxxNL = A11*0.5*w_x*w_x + A12*0.5*w_y*w_y + A16*w_x*w_y
                NyyNL = A12*0.5*w_x*w_x + A22*0.5*w_y*w_y + A26*w_x*w_y
                NxyNL = A16*0.5*w_x*w_x + A26*0.5*w_y*w_y + A66*w_x*w_y

                # kC
                c = -1
                for i in range(m):
                    fAu = ftab[2, i]
                    fAu_xi = ftab[3, i]
                    fAv = ftab[4, i]
                    fAv_xi = ftab[5, i]
                    fAw = ftab[0, i]
                    fAw_xi = ftab[1, i]
                    fAw_xixi = ftab[6, i]

                    for k in range(m):
                        fBu = ftab[2, k]
                        fBu_xi = ftab[3, k]
                        fBv = ftab[4, k]
                        fBv_xi = ftab[5, k]
                        fBw = ftab[0, k]
                        fBw_xi = ftab[1, k]
                        fBw_xixi = ftab[6, k]

                        for j in range(n):
                            gAu = gtab[2, j]
                            gAu_eta = gtab[3, j]
                            gAv = gtab[4, j]
                            gAv_eta = gtab[5, j]
                            gAw = gtab[0, j]
                            gAw_eta = gtab[1, j]
                            gAw_etaeta = gtab[6, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = gtab[2, l]
                                gBu_eta = gtab[3, l]
                                gBv = gtab[4, l]
                                gBv_eta = gtab[5, l]
                                gBw = gtab[0, l]
                                gBw_eta = gtab[1, l]
                                gBw_etaeta = gtab[6, l]

                         ### THIS IS CALCULATES K_C NOT K_C_0

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*( intx*inty*(a*fBu*gBu_eta*(A16*b*fAu_xi*gAu + A66*a*fAu*gAu_eta) + b*fBu_xi*gBu*(A11*b*fAu_xi*gAu + A16*a*fAu*gAu_eta))/((a*a)*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*( intx*inty*(a*fBv*gBv_eta*(A12*b*fAu_xi*gAu + A26*a*fAu*gAu_eta) + b*fBv_xi*gBv*(A16*b*fAu_xi*gAu + A66*a*fAu*gAu_eta))/((a*a)*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*( 2*intx*inty*((a*a)*fBw*(gBw_eta*w_eta*(A12*b*fAu_xi*gAu + A26*a*fAu*gAu_eta) - gBw_etaeta*(B12*b*fAu_xi*gAu + B26*a*fAu*gAu_eta)) + a*b*(-2*fBw_xi*gBw_eta*(B16*b*fAu_xi*gAu + B66*a*fAu*gAu_eta) + (fBw*gBw_eta*w_xi + fBw_xi*gBw*w_eta)*(A16*b*fAu_xi*gAu + A66*a*fAu*gAu_eta)) + (b*b)*gBw*(fBw_xi*w_xi*(A11*b*fAu_xi*gAu + A16*a*fAu*gAu_eta) - fBw_xixi*(B11*b*fAu_xi*gAu + B16*a*fAu*gAu_eta)))/((a*a*a)*(b*b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*( intx*inty*(a*fBu*gBu_eta*(A26*a*fAv*gAv_eta + A66*b*fAv_xi*gAv) + b*fBu_xi*gBu*(A12*a*fAv*gAv_eta + A16*b*fAv_xi*gAv))/((a*a)*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*( intx*inty*(a*fBv*gBv_eta*(A22*a*fAv*gAv_eta + A26*b*fAv_xi*gAv) + b*fBv_xi*gBv*(A26*a*fAv*gAv_eta + A66*b*fAv_xi*gAv))/((a*a)*(b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*( 2*intx*inty*((a*a)*fBw*(gBw_eta*w_eta*(A22*a*fAv*gAv_eta + A26*b*fAv_xi*gAv) - gBw_etaeta*(B22*a*fAv*gAv_eta + B26*b*fAv_xi*gAv)) + a*b*(-2*fBw_xi*gBw_eta*(B26*a*fAv*gAv_eta + B66*b*fAv_xi*gAv) + (fBw*gBw_eta*w_xi + fBw_xi*gBw*w_eta)*(A26*a*fAv*gAv_eta + A66*b*fAv_xi*gAv)) + (b*b)*gBw*(fBw_xi*w_xi*(A12*a*fAv*gAv_eta + A16*b*fAv_xi*gAv) - fBw_xixi*(B12*a*fAv*gAv_eta + B16*b*fAv_xi*gAv)))/((a*a*a)*(b*b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*( 2*intx*inty*(a*fBu*gBu_eta*((a*a)*fAw*(A26*gAw_eta*w_eta - B26*gAw_etaeta) + a*b*(A66*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta) - 2*B66*fAw_xi*gAw_eta) + (b*b)*gAw*(A16*fAw_xi*w_xi - B16*fAw_xixi)) + b*fBu_xi*gBu*((a*a)*fAw*(A12*gAw_eta*w_eta - B12*gAw_etaeta) + a*b*(A16*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta) - 2*B16*fAw_xi*gAw_eta) + (b*b)*gAw*(A11*fAw_xi*w_xi - B11*fAw_xixi)))/((a*a*a)*(b*b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*( 2*intx*inty*(a*fBv*gBv_eta*((a*a)*fAw*(A22*gAw_eta*w_eta - B22*gAw_etaeta) + a*b*(A26*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta) - 2*B26*fAw_xi*gAw_eta) + (b*b)*gAw*(A12*fAw_xi*w_xi - B12*fAw_xixi)) + b*fBv_xi*gBv*((a*a)*fAw*(A26*gAw_eta*w_eta - B26*gAw_etaeta) + a*b*(A66*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta) - 2*B66*fAw_xi*gAw_eta) + (b*b)*gAw*(A16*fAw_xi*w_xi - B16*fAw_xixi)))/((a*a*a)*(b*b*b)) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*( 4*intx*inty*((a*a)*fBw*(gBw_eta*w_eta*((a*a)*fAw*(A22*gAw_eta*w_eta - B22*gAw_etaeta) + a*b*(A26*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta) - 2*B26*fAw_xi*gAw_eta) + (b*b)*gAw*(A12*fAw_xi*w_xi - B12*fAw_xixi)) - gBw_etaeta*((a*a)*fAw*(B22*gAw_eta*w_eta - D22*gAw_etaeta) + a*b*(B26*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta) - 2*D26*fAw_xi*gAw_eta) + (b*b)*gAw*(B12*fAw_xi*w_xi - D12*fAw_xixi))) + a*b*(-2*fBw_xi*gBw_eta*((a*a)*fAw*(B26*gAw_eta*w_eta - D26*gAw_etaeta) + a*b*(B66*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta) - 2*D66*fAw_xi*gAw_eta) + (b*b)*gAw*(B16*fAw_xi*w_xi - D16*fAw_xixi)) + (fBw*gBw_eta*w_xi + fBw_xi*gBw*w_eta)*((a*a)*fAw*(A26*gAw_eta*w_eta - B26*gAw_etaeta) + a*b*(A66*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta) - 2*B66*fAw_xi*gAw_eta) + (b*b)*gAw*(A16*fAw_xi*w_xi - B16*fAw_xixi))) + (b*b)*gBw*(fBw_xi*w_xi*((a*a)*fAw*(A12*gAw_eta*w_eta - B12*gAw_etaeta) + a*b*(A16*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta) - 2*B16*fAw_xi*gAw_eta) + (b*b)*gAw*(A11*fAw_xi*w_xi - B11*fAw_xixi)) - fBw_xixi*((a*a)*fAw*(B12*gAw_eta*w_eta - D12*gAw_etaeta) + a*b*(B16*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta) - 2*D16*fAw_xi*gAw_eta) + (b*b)*gAw*(B11*fAw_xi*w_xi - D11*fAw_xixi))))/((a*a*a*a)*(b*b*b*b)) )
                                # KGNL
                                kCv[c] += weight*(intx*inty/4)*(
                                          NxxNL*(2/a)*fAw_xi*gAw*(2/a)*fBw_xi*gBw
                                        + NxyNL*((2/b)*fAw*gAw_eta*(2/a)*fBw_xi*gBw + (2/a)*fAw_xi*gAw*(2/b)*fBw*gBw_eta)
                                        + NyyNL*(2/b)*fAw*gAw_eta*(2/b)*fBw*gBw_eta )

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
    homogeneous of degree one in ``cs``. Adding ``KGNL`` here would
    double-count it in the tangent stiffness matrix and break linear
    buckling. See the comments in :func:`.fkC_num` and :func:`.calc_fint`.

    """
    cdef double [:, ::1] gtab, ftab
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

    cdef double fAu, fAv, fAw, fAu_xi, fAv_xi, fAw_xi, fAw_xixi
    cdef double gAu, gAv, gAw, gAu_eta, gAv_eta, gAw_eta, gAw_etaeta
    cdef double gBw, gBw_eta, fBw, fBw_xi

    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double w_xi, w_eta, Nxx, Nyy, Nxy

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
                        eyy += cs[col+1]*(2/b)*fAv*gAv_eta
                        gxy += cs[col+0]*(2/b)*fAu*gAu_eta + cs[col+1]*(2/a)*fAv_xi*gAv
                        kxx += -cs[col+2]*(2/a*2/a)*fAw_xixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAw_etaeta
                        kxy += -2*cs[col+2]*(2/a)*fAw_xi*(2/b)*gAw_eta

                # Calculating membrane stress components
                Nxx = Nxx0 + A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = Nyy0 + A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = Nxy0 + A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy

                # computing kG
                c = -1
                for j in range(n):
                    gAw = gtab[2, j]
                    gAw_eta = gtab[5, j]

                    for l in range(n):
                        gBw = gtab[2, l]
                        gBw_eta = gtab[5, l]

                        for i in range(m):
                            fAw = ftab[2, i]
                            fAw_xi = ftab[5, i]

                            for k in range(m):
                                fBw = ftab[2, k]
                                fBw_xi = ftab[5, k]

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*( intx*inty*(a*fBw*gBw_eta*(Nxy*b*fAw_xi*gAw + Nyy*a*fAw*gAw_eta) + b*fBw_xi*gBw*(Nxx*b*fAw_xi*gAw + Nxy*a*fAw*gAw_eta))/((a*a)*(b*b)) )

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG


def fkM_num(object shell, object I_input, int size,
        int row0, int col0, int nx, int ny):
    cdef double [:, ::1] ftab, gtab
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
                                kMv[c] += weight*( 0.25*I0*fAu*fBu*gAu*gBu*intx*inty )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+2
                                kMv[c] += weight*( -0.5*I1*fAu*fBw_xi*gAu*gBw*intx*inty/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+1
                                kMv[c] += weight*( 0.25*I0*fAv*fBv*gAv*gBv*intx*inty )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+2
                                kMv[c] += weight*( -0.5*I1*fAv*fBw*gAv*gBw_eta*intx*inty/b )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+0
                                kMv[c] += weight*( -0.5*I1*fAw_xi*fBu*gAw*gBu*intx*inty/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+1
                                kMv[c] += weight*( -0.5*I1*fAw*fBv*gAw_eta*gBv*intx*inty/b )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+2
                                kMv[c] += weight*( 0.25*I0*fAw*fBw*gAw*gBw*intx*inty + I2*fAw*fBw*gAw_eta*gBw_eta*intx*inty/(b*b) + I2*fAw_xi*fBw_xi*gAw*gBw*intx*inty/(a*a) )

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM



def fkAx_num(object shell, int size, int row0, int col0, int nx, int ny):
    cdef double [:, ::1] ftab, gtab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, beta, intx, inty
    cdef int m, n
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kAr, kAc
    cdef double [::1] kAv

    cdef double fAw, fAw_xi, fBw, gAw, gBw
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
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

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

                # kAx
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
                                #NOTE flat plates have no curvature term
                                #      gamma, see Shell.calc_kA()
                                kAv[c] += weight*( 0.25*intx*inty*(-2*beta*fAw_xi*fBw*gAw*gBw/a) )

    kAx = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kAx


def fkAy_num(object shell, int size, int row0, int col0, int nx, int ny):
    cdef double [:, ::1] ftab, gtab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, beta, intx, inty
    cdef int m, n
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kAr, kAc
    cdef double [::1] kAv

    cdef double fAw, fBw, gAw_eta, gBw
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
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

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

                # kAy
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
                                kAv[c] += weight*( -0.5*beta*fAw*fBw*gAw_eta*gBw*intx*inty/b )

    kAy = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kAy


def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    cdef double [:, ::1] gtab, ftab
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
    cdef double w_xi, w_eta

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

                w_xi = 0
                w_eta = 0
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[1, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[2, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[3, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[4, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[5, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[1, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[2, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[3, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[4, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[5, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAw = gtab[0, j]
                    gAw_eta = gtab[1, j]
                    for i in range(m):
                        fAw = ftab[0, i]
                        fAw_xi = ftab[1, i]

                        col = col0 + DOF*(j*m + i)

                        w_xi += cs[col+2]*fAw_xi*gAw
                        w_eta += cs[col+2]*fAw*gAw_eta

                # current strain state
                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.

                for j in range(n):
                    gAu = gtab[2, j]
                    gAu_eta = gtab[3, j]
                    gAv = gtab[4, j]
                    gAv_eta = gtab[5, j]
                    gAw = gtab[0, j]
                    gAw_eta = gtab[1, j]
                    gAw_etaeta = gtab[6, j]

                    for i in range(m):
                        fAu = ftab[2, i]
                        fAu_xi = ftab[3, i]
                        fAv = ftab[4, i]
                        fAv_xi = ftab[5, i]
                        fAw = ftab[0, i]
                        fAw_xi = ftab[1, i]
                        fAw_xixi = ftab[6, i]

                        col = col0 + DOF*(j*m + i)

                        exx += cs[col+0]*(2/a)*fAu_xi*gAu + 0.5*cs[col+2]*(2/a)*fAw_xi*gAw*(2/a)*w_xi
                        eyy += cs[col+1]*(2/b)*fAv*gAv_eta + 0.5*cs[col+2]*(2/b)*fAw*gAw_eta*(2/b)*w_eta
                        gxy += cs[col+0]*(2/b)*fAu*gAu_eta + cs[col+1]*(2/a)*fAv_xi*gAv + cs[col+2]*(2/a)*fAw_xi*gAw*(2/b)*w_eta
                        kxx += -cs[col+2]*(2/a*2/a)*fAw_xixi*gAw
                        kyy += -cs[col+2]*(2/b*2/b)*fAw*gAw_etaeta
                        kxy += -2*cs[col+2]*(2/a*2/b)*fAw_xi*gAw_eta

                # current stress state
                Nxx = A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy
                Mxx = B11*exx + B12*eyy + B16*gxy + D11*kxx + D12*kyy + D16*kxy
                Myy = B12*exx + B22*eyy + B26*gxy + D12*kxx + D22*kyy + D26*kxy
                Mxy = B16*exx + B26*eyy + B66*gxy + D16*kxx + D26*kyy + D66*kxy

                for j in range(n):
                    gAu = gtab[2, j]
                    gAu_eta = gtab[3, j]
                    gAv = gtab[4, j]
                    gAv_eta = gtab[5, j]
                    gAw = gtab[0, j]
                    gAw_eta = gtab[1, j]
                    gAw_etaeta = gtab[6, j]

                    for i in range(m):
                        fAu = ftab[2, i]
                        fAu_xi = ftab[3, i]
                        fAv = ftab[4, i]
                        fAv_xi = ftab[5, i]
                        fAw = ftab[0, i]
                        fAw_xi = ftab[1, i]
                        fAw_xixi = ftab[6, i]

                        col = col0 + DOF*(j*m + i)

                        fint[col+0] += weight*( intx*inty*(Nxx*b*fAu_xi*gAu + Nxy*a*fAu*gAu_eta)/(2*a*b) )
                        fint[col+1] += weight*( intx*inty*(Nxy*b*fAv_xi*gAv + Nyy*a*fAv*gAv_eta)/(2*a*b) )
                        fint[col+2] += weight*( intx*inty*((a*a)*fAw*(-Myy*gAw_etaeta + Nyy*gAw_eta*w_eta) + a*b*(-2*Mxy*fAw_xi*gAw_eta + Nxy*(fAw*gAw_eta*w_xi + fAw_xi*gAw*w_eta)) + (b*b)*gAw*(-Mxx*fAw_xixi + Nxx*fAw_xi*w_xi))/((a*a)*(b*b)) )

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

        ax = qf*v_x*w_y - qf*v_y*w_x - w_x
        ay = -qf*u_x*w_y + qf*u_y*w_x - w_y
        az = qf*u_x*v_y - qf*u_y*v_x + ref + u_x + v_y

    with ``qf = 1`` for ``quadratic=1`` and ``qf = 0`` otherwise, and ``ref =
    1`` for ``reference=1``, which includes the dead part ``F_p(0)``, and
    ``ref = 0`` otherwise.

    """
    cdef double [:, ::1] gtab, ftab
    cdef double a, b, pintx, pinty, qf
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
                u_x = u_y = v_x = v_y = w_x = w_y = 0.
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
                        w_x += cs[col+2]*(2/a)*fAw_xi*gAw
                        w_y += cs[col+2]*(2/b)*fAw*gAw_eta

                ax = qf*v_x*w_y - qf*v_y*w_x - w_x
                ay = -qf*u_x*w_y + qf*u_y*w_x - w_y
                az = qf*u_x*v_y - qf*u_y*v_x + ref + u_x + v_y

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

    fdim = 6*m*m*n*n
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
                u_x = u_y = v_x = v_y = w_x = w_y = 0.
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
                                kCv[c] += weight*( 2*fAu*fBv*gAu*gBv_eta*qf*w_x/b - 2*fAu*fBv_xi*gAu*gBv*qf*w_y/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*( -2*fAu*fBw*gAu*gBw_eta*qf*v_x/b + 2*fAu*fBw_xi*gAu*gBw*qf*v_y/a + 2*fAu*fBw_xi*gAu*gBw/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*( -2*fAv*fBu*gAv*gBu_eta*qf*w_x/b + 2*fAv*fBu_xi*gAv*gBu*qf*w_y/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*( 2*fAv*fBw*gAv*gBw_eta*qf*u_x/b + 2*fAv*fBw*gAv*gBw_eta/b - 2*fAv*fBw_xi*gAv*gBw*qf*u_y/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*( 2*fAw*fBu*gAw*gBu_eta*qf*v_x/b - 2*fAw*fBu_xi*gAw*gBu*qf*v_y/a - 2*fAw*fBu_xi*gAw*gBu/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*( -2*fAw*fBv*gAw*gBv_eta*qf*u_x/b - 2*fAw*fBv*gAw*gBv_eta/b + 2*fAw*fBv_xi*gAw*gBv*qf*u_y/a )

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
    cdef double a, b, intx, inty
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
                    gAw = gtab[4, j]
                    gAw_eta = gtab[5, j]
                    for i in range(m):
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        col = col0 + DOF*(j*m + i)
                        bx += 2*fAw_xi*gAw/a*cs[col+2]
                        by += 2*fAw*gAw_eta/b*cs[col+2]
                        dbx += 2*fAw_xi*gAw/a*vs[col+2]
                        dby += 2*fAw*gAw_eta/b*vs[col+2]

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
                        e[0] += 2*fAu_xi*gAu/a*cs[col+0] + bx*fAw_xi*gAw/a*cs[col+2]
                        e[1] += 2*fAv*gAv_eta/b*cs[col+1] + by*fAw*gAw_eta/b*cs[col+2]
                        e[2] += 2*fAu*gAu_eta/b*cs[col+0] + 2*fAv_xi*gAv/a*cs[col+1] + (bx*fAw*gAw_eta/b + by*fAw_xi*gAw/a)*cs[col+2]
                        e[3] += -4*fAw_xixi*gAw/(a*a)*cs[col+2]
                        e[4] += -4*fAw*gAw_etaeta/(b*b)*cs[col+2]
                        e[5] += -8*fAw_xi*gAw_eta/(a*b)*cs[col+2]
                        de[0] += 2*fAu_xi*gAu/a*vs[col+0] + 2*bx*fAw_xi*gAw/a*vs[col+2]
                        de[1] += 2*fAv*gAv_eta/b*vs[col+1] + 2*by*fAw*gAw_eta/b*vs[col+2]
                        de[2] += 2*fAu*gAu_eta/b*vs[col+0] + 2*fAv_xi*gAv/a*vs[col+1] + (2*bx*fAw*gAw_eta/b + 2*by*fAw_xi*gAw/a)*vs[col+2]
                        de[3] += -4*fAw_xixi*gAw/(a*a)*vs[col+2]
                        de[4] += -4*fAw*gAw_etaeta/(b*b)*vs[col+2]
                        de[5] += -8*fAw_xi*gAw_eta/(a*b)*vs[col+2]

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
                        kTv[col+0] += weight*(2*ds[2]*fAu*gAu_eta/b + 2*ds[0]*fAu_xi*gAu/a)
                        kTv[col+1] += weight*(2*ds[1]*fAv*gAv_eta/b + 2*ds[2]*fAv_xi*gAv/a)
                        kTv[col+2] += weight*(2*Nxx*dbx*fAw_xi*gAw/a + Nxy*(2*dbx*fAw*gAw_eta/b + 2*dby*fAw_xi*gAw/a) + 2*Nyy*dby*fAw*gAw_eta/b + ds[2]*(2*bx*fAw*gAw_eta/b + 2*by*fAw_xi*gAw/a) + 2*by*ds[1]*fAw*gAw_eta/b - 4*ds[4]*fAw*gAw_etaeta/(b*b) + 2*bx*ds[0]*fAw_xi*gAw/a - 8*ds[5]*fAw_xi*gAw_eta/(a*b) - 4*ds[3]*fAw_xixi*gAw/(a*a))

    return np.asarray(kTv)

# END tangent stiffness product
