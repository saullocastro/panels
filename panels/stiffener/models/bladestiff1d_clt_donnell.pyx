#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE


cdef int DOF = 3

cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fppfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil


def fkCf(double ys, double a, double b, double bf, double df, double E1, double F1,
         double S1, double Jxx, int m, int n,
         double x1u, double x1ur, double x2u, double x2ur,
         double x1w, double x1wr, double x2w, double x2wr,
         double y1u, double y1ur, double y2u, double y2ur,
         double y1w, double y1wr, double y2w, double y2wr,
         int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef double fAuxifBuxi, fAuxifBwxixi, fAuxifBwxi, fAwxixifBuxi
    cdef double fAwxifBuxi, fAwxifBwxi, fAwxifBwxixi, fAwxixifBwxi
    cdef double fAwxixifBwxixi
    cdef double gAu, gBu, gAw, gBw, gAweta, gBweta

    cdef long [:] k0fr, k0fc
    cdef double [:] k0fv

    eta = 2*ys/b - 1.

    fdim = 4*m*n*m*n

    k0fr = np.zeros((fdim,), dtype=INT)
    k0fc = np.zeros((fdim,), dtype=INT)
    k0fv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # k0f
        c = -1
        for i in range(m):
            for k in range(m):
                fAuxifBuxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAuxifBwxixi = integral_fpfpp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAuxifBwxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwxixifBuxi = integral_fpfpp(k, i, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwxifBuxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1u, x1ur, x2u, x2ur)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxixi = integral_fpfpp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxixifBwxi = integral_fpfpp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxixifBwxixi = integral_fppfpp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = f(l, eta, y1u, y1ur, y2u, y2ur)
                        gBw = f(l, eta, y1w, y1wr, y2w, y2wr)
                        gBweta = fp(l, eta, y1w, y1wr, y2w, y2wr)

                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+0
                        k0fv[c] += 2*E1*bf*fAuxifBuxi*gAu*gBu/a
                        c += 1
                        k0fr[c] = row+0
                        k0fc[c] = col+2
                        k0fv[c] += 0.5*a*bf*(8*E1*df*fAuxifBwxixi*gAu*gBw/(a*a*a) - 8*S1*fAuxifBwxi*gAu*gBweta/((a*a)*b))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+0
                        k0fv[c] += bf*gBu*(4*E1*df*fAwxixifBuxi*gAw/(a*a) - 4*S1*fAwxifBuxi*gAweta/(a*b))
                        c += 1
                        k0fr[c] = row+2
                        k0fc[c] = col+2
                        k0fv[c] += 0.5*a*bf*(-4*gBweta*(-4*Jxx*fAwxifBwxi*gAweta/(a*b) + 4*S1*df*fAwxixifBwxi*gAw/(a*a))/(a*b) - 4*gBw*(4*S1*df*fAwxifBwxixi*gAweta/(a*b) + fAwxixifBwxixi*gAw*(-4*E1*(df*df) - 4*F1)/(a*a))/(a*a))

    k0f = coo_matrix((k0fv, (k0fr, k0fc)), shape=(size, size))

    return k0f


def fkGf(double ys, double Fx, double a, double b, double bf, int m, int n,
          double x1w, double x1wr, double x2w, double x2wr,
          double y1w, double y1wr, double y2w, double y2wr,
          int size, int row0, int col0):
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef long [:] kGfr, kGfc
    cdef double [:] kGfv

    cdef double fAwxifBwxi, gAw, gBw

    eta = 2*ys/b - 1.

    fdim = 1*m*n*m*n

    kGfr = np.zeros((fdim,), dtype=INT)
    kGfc = np.zeros((fdim,), dtype=INT)
    kGfv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kGf
        c = -1
        for i in range(m):
            for k in range(m):
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):
                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBw = f(l, eta, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kGfr[c] = row+2
                        kGfc[c] = col+2
                        kGfv[c] += 2*Fx*fAwxifBwxi*gAw*gBw/a

    kGf = coo_matrix((kGfv, (kGfr, kGfc)), shape=(size, size))

    return kGf


def fkMf(double ys, double rho, double h, double hb, double hf, double a,
         double b, double bf, double df,
         int m, int n,
         double x1u, double x1ur, double x2u, double x2ur,
         double x1v, double x1vr, double x2v, double x2vr,
         double x1w, double x1wr, double x2w, double x2wr,
         double y1u, double y1ur, double y2u, double y2ur,
         double y1v, double y1vr, double y2v, double y2vr,
         double y1w, double y1wr, double y2w, double y2wr,
         int size, int row0, int col0):
    r"""Mass matrix of the 1D flange of the blade stiffener

    The flange, of height `b_f` and thickness `h_f`, is a beam along `x` at
    `y = y_s`, below the panel, whose centroid is at the distance `d_f = b_f/2
    + h_b + h/2` from the mid-surface of the panel, `h` being the thickness of
    the panel and `h_b` the thickness of the stiffener's base. With the
    rotations `\phi_x = -w_{,x}` and `\phi_y = -w_{,y}` of the panel, the
    mass matrix is:

    .. math::
        [M_{sf}] = \rho h_f b_f \int_x [S(x, y_s)]^T [k_{mf}] [S(x, y_s)] dx

    where `[S]` gives `\{u, v, w, \phi_x, \phi_y\}^T` and

    .. math::
        [k_{mf}] = \begin{bmatrix}
                   1 & 0 & 0 & -d_f & 0 \\
                   0 & 1 & 0 & 0 & -d_f \\
                   0 & 0 & 1 & 0 & 0 \\
                   -d_f & 0 & 0 & k_{mf44} & 0 \\
                   0 & -d_f & 0 & 0 & k_{mf55}
                   \end{bmatrix}, \quad
        k_{mf44} = k_{mf55} = \frac{b_f^2}{12} + d_f^2
                 = \frac{b_f^2}{3} + \frac{b_f (h + 2 h_b)}{2}
                   + \frac{(h + 2 h_b)^2}{4}

    obtained integrating the kinetic energy through the height of the
    flange, see ``theory/stiffener/mass_matrix_1D_stiffeners.py`` and
    ``theory/multidomain_panels/bladestiff1d_clt_donnell/``.

    .. note:: Correction with respect to Eq. (26) of Castro et al. (2016)
              [castro2016FlutterPanel]_, whose matrix `[k_{mf}]` has `-2
              d_f` in the terms coupling the displacements `u, v` and the
              rotations `\phi_x, \phi_y`. The in-plane displacements of the
              flange at its centroid are `u - d_f \phi_x` and `v - d_f
              \phi_y`, and the term `-2 d_f u \phi_x` of `(u - d_f
              \phi_x)^2` is shared by the two symmetric entries `(1, 4)` and
              `(4, 1)` of `[k_{mf}]`, which are then `-d_f`, consistent with
              the term `d_f^2` of `k_{mf44}` and with the offset of the mass
              matrix of the shells. Up to version 0.7.1 this kernel used `-2
              d_f`, doubling the inertial coupling between the flange and the
              panel. Eq. (27) of the same paper, for the stiffener's base, has
              the same factor, `-2 d_b`, but the base is a :class:`.Shell`
              with offset whose mass matrix uses `-d_b`.

    Parameters
    ----------
    ys : float
        Coordinate `y` of the stiffener.
    rho : float
        Density of the flange.
    h : float
        Thickness of the panel.
    hb, hf : float
        Thicknesses of the stiffener's base and of the flange.
    a, b : float
        Dimensions of the panel.
    bf, df : float
        Height of the flange and distance `d_f` of its centroid to the
        mid-surface of the panel.
    m, n : int
        Number of terms along `x` and `y`.
    x1u, ..., y2wr : float
        Boundary conditions of the panel.
    size, row0, col0 : int
        Size of the global matrix and position of this matrix in it.

    Returns
    -------
    kMf : :class:`scipy.sparse.coo_matrix`
        Upper triangle of the mass matrix of the flange.

    """
    cdef double fAufBu, fAufBwxi, fAvfBv, fAvfBw, fAwxifBu, fAwfBv, fAwfBw
    cdef double fAwxifBwxi
    cdef double gAu, gBu, gAv, gBv, gAw, gBw, gAweta, gBweta

    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef long [:] kMfr, kMfc
    cdef double [:] kMfv

    eta = 2*ys/b - 1.

    fdim = 7*m*n*m*n

    kMfr = np.zeros((fdim,), dtype=INT)
    kMfc = np.zeros((fdim,), dtype=INT)
    kMfv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kMf
        c = -1
        for i in range(m):
            for k in range(m):

                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBwxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBw = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwfBv = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):

                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)

                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = f(l, eta, y1u, y1ur, y2u, y2ur)
                        gBv = f(l, eta, y1v, y1vr, y2v, y2vr)
                        gBw = f(l, eta, y1w, y1wr, y2w, y2wr)
                        gBweta = fp(l, eta, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kMfr[c] = row+0
                        kMfc[c] = col+0
                        kMfv[c] += 0.5*a*bf*fAufBu*gAu*gBu*hf*rho
                        c += 1
                        kMfr[c] = row+0
                        kMfc[c] = col+2
                        kMfv[c] += bf*df*fAufBwxi*gAu*gBw*hf*rho
                        c += 1
                        kMfr[c] = row+1
                        kMfc[c] = col+1
                        kMfv[c] += 0.5*a*bf*fAvfBv*gAv*gBv*hf*rho
                        c += 1
                        kMfr[c] = row+1
                        kMfc[c] = col+2
                        kMfv[c] += a*bf*df*fAvfBw*gAv*gBweta*hf*rho/b
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+0
                        kMfv[c] += bf*df*fAwxifBu*gAw*gBu*hf*rho
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+1
                        kMfv[c] += a*bf*df*fAwfBv*gAweta*gBv*hf*rho/b
                        c += 1
                        kMfr[c] = row+2
                        kMfc[c] = col+2
                        kMfv[c] += 0.166666666666667*bf*hf*rho*((a*a)*fAwfBw*(3*(b*b)*gAw*gBw + gAweta*gBweta*(4*(bf*bf) + 6*bf*(h + 2*hb) + 3*(h + 2*hb)**2)) + (b*b)*fAwxifBwxi*gAw*gBw*(4*(bf*bf) + 6*bf*(h + 2*hb) + 3*(h + 2*hb)**2))/(a*(b*b))

    kMf = coo_matrix((kMfv, (kMfr, kMfc)), shape=(size, size))

    return kMf
