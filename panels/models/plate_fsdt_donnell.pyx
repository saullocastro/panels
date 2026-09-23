#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Analytical matrices of flat plates using the first-order shear deformation theory (FSDT) with von Karman
kinematics, integrated over the full domain

See the kinematic equations in
``theory/shells/plate_fsdt_tsdt_donnell/plate_fsdt_tsdt_donnell.py``, from
where the integrands herein have been generated. The degrees of freedom of
each term of the approximation are ``u, v, w, phix, phiy``.

"""
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE


cdef int DOF = 5
cdef int NE = 8


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


def fk0(object shell, int size, int row0, int col0):
    cdef double a, b
    cdef double [:, ::1] F
    cdef int m, n
    cdef double A11, A12, A16, B11, B12
    cdef double B16, A22, A26, B22, B26
    cdef double A66, B66, D11, D12, D16
    cdef double D22, D26, D66, A44, A45
    cdef double A55
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

    cdef int i, j, k, l, c, row, col

    cdef long [:] k0r, k0c
    cdef double [:] k0v

    cdef double fAphixfBphix, fAphixfBphixxi, fAphixfBphiy, fAphixfBphiyxi, fAphixfBu
    cdef double fAphixfBuxi, fAphixfBv, fAphixfBvxi, fAphixfBw, fAphixfBwxi
    cdef double fAphixxifBphix, fAphixxifBphixxi, fAphixxifBphiy, fAphixxifBphiyxi, fAphixxifBu
    cdef double fAphixxifBuxi, fAphixxifBv, fAphixxifBvxi, fAphiyfBphix, fAphiyfBphixxi
    cdef double fAphiyfBphiy, fAphiyfBphiyxi, fAphiyfBu, fAphiyfBuxi, fAphiyfBv
    cdef double fAphiyfBvxi, fAphiyfBw, fAphiyfBwxi, fAphiyxifBphix, fAphiyxifBphixxi
    cdef double fAphiyxifBphiy, fAphiyxifBphiyxi, fAphiyxifBu, fAphiyxifBuxi, fAphiyxifBv
    cdef double fAphiyxifBvxi, fAufBphix, fAufBphixxi, fAufBphiy, fAufBphiyxi
    cdef double fAufBu, fAufBuxi, fAufBv, fAufBvxi, fAuxifBphix
    cdef double fAuxifBphixxi, fAuxifBphiy, fAuxifBphiyxi, fAuxifBu, fAuxifBuxi
    cdef double fAuxifBv, fAuxifBvxi, fAvfBphix, fAvfBphixxi, fAvfBphiy
    cdef double fAvfBphiyxi, fAvfBu, fAvfBuxi, fAvfBv, fAvfBvxi
    cdef double fAvxifBphix, fAvxifBphixxi, fAvxifBphiy, fAvxifBphiyxi, fAvxifBu
    cdef double fAvxifBuxi, fAvxifBv, fAvxifBvxi, fAwfBphix, fAwfBphiy
    cdef double fAwfBw, fAwfBwxi, fAwxifBphix, fAwxifBphiy, fAwxifBw
    cdef double fAwxifBwxi
    cdef double gAphixetagBphix, gAphixetagBphixeta, gAphixetagBphiy, gAphixetagBphiyeta, gAphixetagBu
    cdef double gAphixetagBueta, gAphixetagBv, gAphixetagBveta, gAphixgBphix, gAphixgBphixeta
    cdef double gAphixgBphiy, gAphixgBphiyeta, gAphixgBu, gAphixgBueta, gAphixgBv
    cdef double gAphixgBveta, gAphixgBw, gAphixgBweta, gAphiyetagBphix, gAphiyetagBphixeta
    cdef double gAphiyetagBphiy, gAphiyetagBphiyeta, gAphiyetagBu, gAphiyetagBueta, gAphiyetagBv
    cdef double gAphiyetagBveta, gAphiygBphix, gAphiygBphixeta, gAphiygBphiy, gAphiygBphiyeta
    cdef double gAphiygBu, gAphiygBueta, gAphiygBv, gAphiygBveta, gAphiygBw
    cdef double gAphiygBweta, gAuetagBphix, gAuetagBphixeta, gAuetagBphiy, gAuetagBphiyeta
    cdef double gAuetagBu, gAuetagBueta, gAuetagBv, gAuetagBveta, gAugBphix
    cdef double gAugBphixeta, gAugBphiy, gAugBphiyeta, gAugBu, gAugBueta
    cdef double gAugBv, gAugBveta, gAvetagBphix, gAvetagBphixeta, gAvetagBphiy
    cdef double gAvetagBphiyeta, gAvetagBu, gAvetagBueta, gAvetagBv, gAvetagBveta
    cdef double gAvgBphix, gAvgBphixeta, gAvgBphiy, gAvgBphiyeta, gAvgBu
    cdef double gAvgBueta, gAvgBv, gAvgBveta, gAwetagBphix, gAwetagBphiy
    cdef double gAwetagBw, gAwetagBweta, gAwgBphix, gAwgBphiy, gAwgBw
    cdef double gAwgBweta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    F = shell.ABD

    if F.shape[0] != NE or F.shape[1] != NE:
        raise ValueError('shell.ABD must be a %d x %d matrix' % (NE, NE))
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

    fdim = 21*m*m*n*n

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        A11 = F[0,0]
        A12 = F[0,1]
        A16 = F[0,2]
        B11 = F[0,3]
        B12 = F[0,4]
        B16 = F[0,5]
        A22 = F[1,1]
        A26 = F[1,2]
        B22 = F[1,4]
        B26 = F[1,5]
        A66 = F[2,2]
        B66 = F[2,5]
        D11 = F[3,3]
        D12 = F[3,4]
        D16 = F[3,5]
        D22 = F[4,4]
        D26 = F[4,5]
        D66 = F[5,5]
        A44 = F[6,6]
        A45 = F[6,7]
        A55 = F[7,7]

        # k0
        c = -1
        for i in range(m):
            for k in range(m):

                fAphixfBphix = integral_ff(i, k, x1phix, x1phixr, x2phix, x2phixr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixfBphixxi = integral_ffp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixfBphiy = integral_ff(i, k, x1phix, x1phixr, x2phix, x2phixr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphixfBphiyxi = integral_ffp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphixfBu = integral_ff(i, k, x1phix, x1phixr, x2phix, x2phixr, x1u, x1ur, x2u, x2ur)
                fAphixfBuxi = integral_ffp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1u, x1ur, x2u, x2ur)
                fAphixfBv = integral_ff(i, k, x1phix, x1phixr, x2phix, x2phixr, x1v, x1vr, x2v, x2vr)
                fAphixfBvxi = integral_ffp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1v, x1vr, x2v, x2vr)
                fAphixfBw = integral_ff(i, k, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
                fAphixfBwxi = integral_ffp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
                fAphixxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBphixxi = integral_fpfp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBphiy = integral_ffp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBphiyxi = integral_fpfp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphixxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBuxi = integral_fpfp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1u, x1ur, x2u, x2ur)
                fAphixxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBvxi = integral_fpfp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1v, x1vr, x2v, x2vr)
                fAphiyfBphix = integral_ff(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phix, x1phixr, x2phix, x2phixr)
                fAphiyfBphixxi = integral_ffp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phix, x1phixr, x2phix, x2phixr)
                fAphiyfBphiy = integral_ff(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyfBphiyxi = integral_ffp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyfBu = integral_ff(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1u, x1ur, x2u, x2ur)
                fAphiyfBuxi = integral_ffp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1u, x1ur, x2u, x2ur)
                fAphiyfBv = integral_ff(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1v, x1vr, x2v, x2vr)
                fAphiyfBvxi = integral_ffp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1v, x1vr, x2v, x2vr)
                fAphiyfBw = integral_ff(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1w, x1wr, x2w, x2wr)
                fAphiyfBwxi = integral_ffp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1w, x1wr, x2w, x2wr)
                fAphiyxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBphixxi = integral_fpfp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phix, x1phixr, x2phix, x2phixr)
                fAphiyxifBphiy = integral_ffp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBphiyxi = integral_fpfp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBuxi = integral_fpfp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1u, x1ur, x2u, x2ur)
                fAphiyxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBvxi = integral_fpfp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1v, x1vr, x2v, x2vr)
                fAufBphix = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1phix, x1phixr, x2phix, x2phixr)
                fAufBphixxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1phix, x1phixr, x2phix, x2phixr)
                fAufBphiy = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAufBphiyxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBuxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBv = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAufBvxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAuxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1u, x1ur, x2u, x2ur)
                fAuxifBphixxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1phix, x1phixr, x2phix, x2phixr)
                fAuxifBphiy = integral_ffp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1u, x1ur, x2u, x2ur)
                fAuxifBphiyxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAuxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAuxifBuxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAuxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAuxifBvxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAvfBphix = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1phix, x1phixr, x2phix, x2phixr)
                fAvfBphixxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1phix, x1phixr, x2phix, x2phixr)
                fAvfBphiy = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAvfBphiyxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAvfBu = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAvfBuxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBvxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1v, x1vr, x2v, x2vr)
                fAvxifBphixxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1phix, x1phixr, x2phix, x2phixr)
                fAvxifBphiy = integral_ffp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1v, x1vr, x2v, x2vr)
                fAvxifBphiyxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAvxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAvxifBuxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAvxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvxifBvxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAwfBphix = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1phix, x1phixr, x2phix, x2phixr)
                fAwfBphiy = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwfBwxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
                fAwxifBphiy = integral_ffp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1w, x1wr, x2w, x2wr)
                fAwxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAphixetagBphix = integral_ffp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphixetagBphixeta = integral_fpfp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphixetagBphiy = integral_ffp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphixetagBphiyeta = integral_fpfp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphixetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1phix, y1phixr, y2phix, y2phixr)
                        gAphixetagBueta = integral_fpfp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1u, y1ur, y2u, y2ur)
                        gAphixetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphixetagBveta = integral_fpfp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1v, y1vr, y2v, y2vr)
                        gAphixgBphix = integral_ff(j, l, y1phix, y1phixr, y2phix, y2phixr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphixgBphixeta = integral_ffp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphixgBphiy = integral_ff(j, l, y1phix, y1phixr, y2phix, y2phixr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphixgBphiyeta = integral_ffp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphixgBu = integral_ff(j, l, y1phix, y1phixr, y2phix, y2phixr, y1u, y1ur, y2u, y2ur)
                        gAphixgBueta = integral_ffp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1u, y1ur, y2u, y2ur)
                        gAphixgBv = integral_ff(j, l, y1phix, y1phixr, y2phix, y2phixr, y1v, y1vr, y2v, y2vr)
                        gAphixgBveta = integral_ffp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1v, y1vr, y2v, y2vr)
                        gAphixgBw = integral_ff(j, l, y1phix, y1phixr, y2phix, y2phixr, y1w, y1wr, y2w, y2wr)
                        gAphixgBweta = integral_ffp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1w, y1wr, y2w, y2wr)
                        gAphiyetagBphix = integral_ffp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBphixeta = integral_fpfp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphiyetagBphiy = integral_ffp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBphiyeta = integral_fpfp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBueta = integral_fpfp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1u, y1ur, y2u, y2ur)
                        gAphiyetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBveta = integral_fpfp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1v, y1vr, y2v, y2vr)
                        gAphiygBphix = integral_ff(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphiygBphixeta = integral_ffp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphiygBphiy = integral_ff(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiygBphiyeta = integral_ffp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiygBu = integral_ff(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1u, y1ur, y2u, y2ur)
                        gAphiygBueta = integral_ffp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1u, y1ur, y2u, y2ur)
                        gAphiygBv = integral_ff(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1v, y1vr, y2v, y2vr)
                        gAphiygBveta = integral_ffp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1v, y1vr, y2v, y2vr)
                        gAphiygBw = integral_ff(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
                        gAphiygBweta = integral_ffp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
                        gAuetagBphix = integral_ffp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1u, y1ur, y2u, y2ur)
                        gAuetagBphixeta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1phix, y1phixr, y2phix, y2phixr)
                        gAuetagBphiy = integral_ffp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1u, y1ur, y2u, y2ur)
                        gAuetagBphiyeta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAuetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAuetagBueta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAuetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAuetagBveta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAugBphix = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1phix, y1phixr, y2phix, y2phixr)
                        gAugBphixeta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1phix, y1phixr, y2phix, y2phixr)
                        gAugBphiy = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAugBphiyeta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAugBu = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAugBueta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAugBv = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAugBveta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAvetagBphix = integral_ffp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1v, y1vr, y2v, y2vr)
                        gAvetagBphixeta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1phix, y1phixr, y2phix, y2phixr)
                        gAvetagBphiy = integral_ffp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1v, y1vr, y2v, y2vr)
                        gAvetagBphiyeta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAvetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAvetagBueta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAvetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvetagBveta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBphix = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1phix, y1phixr, y2phix, y2phixr)
                        gAvgBphixeta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1phix, y1phixr, y2phix, y2phixr)
                        gAvgBphiy = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAvgBphiyeta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAvgBu = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAvgBueta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAvgBv = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBveta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAwetagBphix = integral_ffp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1w, y1wr, y2w, y2wr)
                        gAwetagBphiy = integral_ffp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
                        gAwetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetagBweta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBphix = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1phix, y1phixr, y2phix, y2phixr)
                        gAwgBphiy = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBweta = integral_ffp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+0
                        k0v[c] += A11*b*fAuxifBuxi*gAugBu/a + A16*fAufBuxi*gAuetagBu + A16*fAuxifBu*gAugBueta + A66*a*fAufBu*gAuetagBueta/b
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += A12*fAuxifBv*gAugBveta + A16*b*fAuxifBvxi*gAugBv/a + A26*a*fAufBv*gAuetagBveta/b + A66*fAufBvxi*gAuetagBv
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+3
                        k0v[c] += B11*b*fAuxifBphixxi*gAugBphix/a + B16*fAufBphixxi*gAuetagBphix + B16*fAuxifBphix*gAugBphixeta + B66*a*fAufBphix*gAuetagBphixeta/b
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += B12*fAuxifBphiy*gAugBphiyeta + B16*b*fAuxifBphiyxi*gAugBphiy/a + B26*a*fAufBphiy*gAuetagBphiyeta/b + B66*fAufBphiyxi*gAuetagBphiy
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += A12*fAvfBuxi*gAvetagBu + A16*b*fAvxifBuxi*gAvgBu/a + A26*a*fAvfBu*gAvetagBueta/b + A66*fAvxifBu*gAvgBueta
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += A22*a*fAvfBv*gAvetagBveta/b + A26*fAvfBvxi*gAvetagBv + A26*fAvxifBv*gAvgBveta + A66*b*fAvxifBvxi*gAvgBv/a
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+3
                        k0v[c] += B12*fAvfBphixxi*gAvetagBphix + B16*b*fAvxifBphixxi*gAvgBphix/a + B26*a*fAvfBphix*gAvetagBphixeta/b + B66*fAvxifBphix*gAvgBphixeta
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+4
                        k0v[c] += B22*a*fAvfBphiy*gAvetagBphiyeta/b + B26*fAvfBphiyxi*gAvetagBphiy + B26*fAvxifBphiy*gAvgBphiyeta + B66*b*fAvxifBphiyxi*gAvgBphiy/a
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += A44*a*fAwfBw*gAwetagBweta/b + A45*fAwfBwxi*gAwetagBw + A45*fAwxifBw*gAwgBweta + A55*b*fAwxifBwxi*gAwgBw/a
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += 0.5*A45*a*fAwfBphix*gAwetagBphix + 0.5*A55*b*fAwxifBphix*gAwgBphix
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += 0.5*A44*a*fAwfBphiy*gAwetagBphiy + 0.5*A45*b*fAwxifBphiy*gAwgBphiy
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+0
                        k0v[c] += B11*b*fAphixxifBuxi*gAphixgBu/a + B16*fAphixfBuxi*gAphixetagBu + B16*fAphixxifBu*gAphixgBueta + B66*a*fAphixfBu*gAphixetagBueta/b
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+1
                        k0v[c] += B12*fAphixxifBv*gAphixgBveta + B16*b*fAphixxifBvxi*gAphixgBv/a + B26*a*fAphixfBv*gAphixetagBveta/b + B66*fAphixfBvxi*gAphixetagBv
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += 0.5*A45*a*fAphixfBw*gAphixgBweta + 0.5*A55*b*fAphixfBwxi*gAphixgBw
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+3
                        k0v[c] += 0.25*A55*a*b*fAphixfBphix*gAphixgBphix + D11*b*fAphixxifBphixxi*gAphixgBphix/a + D16*fAphixfBphixxi*gAphixetagBphix + D16*fAphixxifBphix*gAphixgBphixeta + D66*a*fAphixfBphix*gAphixetagBphixeta/b
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+4
                        k0v[c] += 0.25*A45*a*b*fAphixfBphiy*gAphixgBphiy + D12*fAphixxifBphiy*gAphixgBphiyeta + D16*b*fAphixxifBphiyxi*gAphixgBphiy/a + D26*a*fAphixfBphiy*gAphixetagBphiyeta/b + D66*fAphixfBphiyxi*gAphixetagBphiy
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += B12*fAphiyfBuxi*gAphiyetagBu + B16*b*fAphiyxifBuxi*gAphiygBu/a + B26*a*fAphiyfBu*gAphiyetagBueta/b + B66*fAphiyxifBu*gAphiygBueta
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+1
                        k0v[c] += B22*a*fAphiyfBv*gAphiyetagBveta/b + B26*fAphiyfBvxi*gAphiyetagBv + B26*fAphiyxifBv*gAphiygBveta + B66*b*fAphiyxifBvxi*gAphiygBv/a
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += 0.5*A44*a*fAphiyfBw*gAphiygBweta + 0.5*A45*b*fAphiyfBwxi*gAphiygBw
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+3
                        k0v[c] += 0.25*A45*a*b*fAphiyfBphix*gAphiygBphix + D12*fAphiyfBphixxi*gAphiyetagBphix + D16*b*fAphiyxifBphixxi*gAphiygBphix/a + D26*a*fAphiyfBphix*gAphiyetagBphixeta/b + D66*fAphiyxifBphix*gAphiygBphixeta
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+4
                        k0v[c] += 0.25*A44*a*b*fAphiyfBphiy*gAphiygBphiy + D22*a*fAphiyfBphiy*gAphiyetagBphiyeta/b + D26*fAphiyfBphiyxi*gAphiyetagBphiy + D26*fAphiyxifBphiy*gAphiygBphiyeta + D66*b*fAphiyxifBphiyxi*gAphiygBphiy/a

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fkG0(double Nxx, double Nyy, double Nxy, object shell,
         int size, int row0, int col0):
    cdef double a, b
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

    cdef int i, j, k, l, c, row, col

    cdef long [:] kG0r, kG0c
    cdef double [:] kG0v

    cdef double fAwfBw, fAwfBwxi, fAwxifBw, fAwxifBwxi
    cdef double gAwetagBw, gAwetagBweta, gAwgBw, gAwgBweta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n

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

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kG0
        c = -1
        for i in range(m):
            for k in range(m):

                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwfBwxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAwetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetagBweta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBweta = integral_ffp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*fAwfBwxi*gAwetagBw + Nxy*fAwxifBw*gAwgBweta + Nyy*a*fAwfBw*gAwetagBweta/b

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


def fkM(object shell, double d, int size, int row0, int col0):
    cdef double a, b, rho, h
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

    cdef int i, j, k, l, c, row, col

    cdef long [:] kMr, kMc
    cdef double [:] kMv

    cdef double fAphixfBphix, fAphixfBu, fAphiyfBphiy, fAphiyfBv, fAufBphix
    cdef double fAufBu, fAvfBphiy, fAvfBv, fAwfBw
    cdef double gAphixgBphix, gAphixgBu, gAphiygBphiy, gAphiygBv, gAugBphix
    cdef double gAugBu, gAvgBphiy, gAvgBv, gAwgBw

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    rho = shell.rho
    h = sum(shell.plyts)
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

    kMr = np.zeros((fdim,), dtype=INT)
    kMc = np.zeros((fdim,), dtype=INT)
    kMv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kM
        c = -1
        for i in range(m):
            for k in range(m):

                fAphixfBphix = integral_ff(i, k, x1phix, x1phixr, x2phix, x2phixr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixfBu = integral_ff(i, k, x1phix, x1phixr, x2phix, x2phixr, x1u, x1ur, x2u, x2ur)
                fAphiyfBphiy = integral_ff(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyfBv = integral_ff(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1v, x1vr, x2v, x2vr)
                fAufBphix = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1phix, x1phixr, x2phix, x2phixr)
                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAvfBphiy = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAphixgBphix = integral_ff(j, l, y1phix, y1phixr, y2phix, y2phixr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphixgBu = integral_ff(j, l, y1phix, y1phixr, y2phix, y2phixr, y1u, y1ur, y2u, y2ur)
                        gAphiygBphiy = integral_ff(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiygBv = integral_ff(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1v, y1vr, y2v, y2vr)
                        gAugBphix = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1phix, y1phixr, y2phix, y2phixr)
                        gAugBu = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAvgBphiy = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAvgBv = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kMr[c] = row+0
                        kMc[c] = col+0
                        kMv[c] += 0.25*a*b*fAufBu*gAugBu*h*rho
                        c += 1
                        kMr[c] = row+0
                        kMc[c] = col+3
                        kMv[c] += -0.25*a*b*d*fAufBphix*gAugBphix*h*rho
                        c += 1
                        kMr[c] = row+1
                        kMc[c] = col+1
                        kMv[c] += 0.25*a*b*fAvfBv*gAvgBv*h*rho
                        c += 1
                        kMr[c] = row+1
                        kMc[c] = col+4
                        kMv[c] += -0.25*a*b*d*fAvfBphiy*gAvgBphiy*h*rho
                        c += 1
                        kMr[c] = row+2
                        kMc[c] = col+2
                        kMv[c] += 0.25*a*b*fAwfBw*gAwgBw*h*rho
                        c += 1
                        kMr[c] = row+3
                        kMc[c] = col+0
                        kMv[c] += -0.25*a*b*d*fAphixfBu*gAphixgBu*h*rho
                        c += 1
                        kMr[c] = row+3
                        kMc[c] = col+3
                        kMv[c] += 0.25*a*b*(d*d)*fAphixfBphix*gAphixgBphix*h*rho + 0.020833333333333332*a*b*fAphixfBphix*gAphixgBphix*(h*h*h)*rho
                        c += 1
                        kMr[c] = row+4
                        kMc[c] = col+1
                        kMv[c] += -0.25*a*b*d*fAphiyfBv*gAphiygBv*h*rho
                        c += 1
                        kMr[c] = row+4
                        kMc[c] = col+4
                        kMv[c] += 0.25*a*b*(d*d)*fAphiyfBphiy*gAphiygBphiy*h*rho + 0.020833333333333332*a*b*fAphiyfBphiy*gAphiygBphiy*(h*h*h)*rho

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


def fkAx(double beta, double gamma, object shell,
         int size, int row0, int col0):
    cdef double a, b
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

    cdef int i, j, k, l, c, row, col

    cdef long [:] kAxr, kAxc
    cdef double [:] kAxv

    cdef double fAwxifBw
    cdef double gAwgBw

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n

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

    kAxr = np.zeros((fdim,), dtype=INT)
    kAxc = np.zeros((fdim,), dtype=INT)
    kAxv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kAx
        c = -1
        for i in range(m):
            for k in range(m):

                fAwxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kAxr[c] = row+2
                        kAxc[c] = col+2
                        kAxv[c] += -0.5*b*beta*fAwxifBw*gAwgBw

    kAx = coo_matrix((kAxv, (kAxr, kAxc)), shape=(size, size))

    return kAx


def fkAy(double beta, object shell, int size, int row0, int col0):
    cdef double a, b
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

    cdef int i, j, k, l, c, row, col

    cdef long [:] kAyr, kAyc
    cdef double [:] kAyv

    cdef double fAwfBw
    cdef double gAwetagBw

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n

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

    kAyr = np.zeros((fdim,), dtype=INT)
    kAyc = np.zeros((fdim,), dtype=INT)
    kAyv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kAy
        c = -1
        for i in range(m):
            for k in range(m):

                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        gAwetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kAyr[c] = row+2
                        kAyc[c] = col+2
                        kAyv[c] += -0.5*a*beta*fAwfBw*gAwetagBw

    kAy = coo_matrix((kAyv, (kAyr, kAyc)), shape=(size, size))

    return kAy


def fcA(double aeromu, object shell, int size, int row0, int col0):
    cdef double a, b
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

    cdef int i, j, k, l, c, row, col

    cdef long [:] cAr, cAc
    cdef double [:] cAv

    cdef double fAwfBw
    cdef double gAwgBw

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n

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

    cAr = np.zeros((fdim,), dtype=INT)
    cAc = np.zeros((fdim,), dtype=INT)
    cAv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # cA
        c = -1
        for i in range(m):
            for k in range(m):

                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        cAr[c] = row+2
                        cAc[c] = col+2
                        cAv[c] += -0.25*a*aeromu*b*fAwfBw*gAwgBw

    cA = coo_matrix((cAv, (cAr, cAc)), shape=(size, size))

    return cA
