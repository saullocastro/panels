#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Cylindrical shells using the Reddy's third-order shear deformation theory
(TSDT) with the Donnell kinematics

Analytical matrices, integrated over the full domain. See the kinematic
equations in ``theory/shells/fsdt_tsdt/fsdt_tsdt.py``, from where the
integrands herein have been generated. The degrees of freedom of each term
of the approximation are ``u, v, w, phix, phiy``.

"""
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE


cdef int DOF = 5
cdef int NE = 13


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
    cdef double r
    cdef double a, b, h, c1
    cdef double [:, ::1] F
    cdef int m, n
    cdef double A11, A12, A16, B11, B12
    cdef double B16, E11, E12, E16, A22
    cdef double A26, B22, B26, E22, E26
    cdef double A66, B66, E66, D11, D12
    cdef double D16, F11, F12, F16, D22
    cdef double D26, F22, F26, D66, F66
    cdef double H11, H12, H16, H22, H26
    cdef double H66, A44, A45, D44, D45
    cdef double A55, D55, F44, F45, F55
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
    cdef double fAphixfBwxixi, fAphixxifBphix, fAphixxifBphixxi, fAphixxifBphiy, fAphixxifBphiyxi
    cdef double fAphixxifBu, fAphixxifBuxi, fAphixxifBv, fAphixxifBvxi, fAphixxifBw
    cdef double fAphixxifBwxi, fAphixxifBwxixi, fAphiyfBphix, fAphiyfBphixxi, fAphiyfBphiy
    cdef double fAphiyfBphiyxi, fAphiyfBu, fAphiyfBuxi, fAphiyfBv, fAphiyfBvxi
    cdef double fAphiyfBw, fAphiyfBwxi, fAphiyfBwxixi, fAphiyxifBphix, fAphiyxifBphixxi
    cdef double fAphiyxifBphiy, fAphiyxifBphiyxi, fAphiyxifBu, fAphiyxifBuxi, fAphiyxifBv
    cdef double fAphiyxifBvxi, fAphiyxifBw, fAphiyxifBwxi, fAphiyxifBwxixi, fAufBphix
    cdef double fAufBphixxi, fAufBphiy, fAufBphiyxi, fAufBu, fAufBuxi
    cdef double fAufBv, fAufBvxi, fAufBw, fAufBwxi, fAufBwxixi
    cdef double fAuxifBphix, fAuxifBphixxi, fAuxifBphiy, fAuxifBphiyxi, fAuxifBu
    cdef double fAuxifBuxi, fAuxifBv, fAuxifBvxi, fAuxifBw, fAuxifBwxi
    cdef double fAuxifBwxixi, fAvfBphix, fAvfBphixxi, fAvfBphiy, fAvfBphiyxi
    cdef double fAvfBu, fAvfBuxi, fAvfBv, fAvfBvxi, fAvfBw
    cdef double fAvfBwxi, fAvfBwxixi, fAvxifBphix, fAvxifBphixxi, fAvxifBphiy
    cdef double fAvxifBphiyxi, fAvxifBu, fAvxifBuxi, fAvxifBv, fAvxifBvxi
    cdef double fAvxifBw, fAvxifBwxi, fAvxifBwxixi, fAwfBphix, fAwfBphixxi
    cdef double fAwfBphiy, fAwfBphiyxi, fAwfBu, fAwfBuxi, fAwfBv
    cdef double fAwfBvxi, fAwfBw, fAwfBwxi, fAwfBwxixi, fAwxifBphix
    cdef double fAwxifBphixxi, fAwxifBphiy, fAwxifBphiyxi, fAwxifBu, fAwxifBuxi
    cdef double fAwxifBv, fAwxifBvxi, fAwxifBw, fAwxifBwxi, fAwxifBwxixi
    cdef double fAwxixifBphix, fAwxixifBphixxi, fAwxixifBphiy, fAwxixifBphiyxi, fAwxixifBu
    cdef double fAwxixifBuxi, fAwxixifBv, fAwxixifBvxi, fAwxixifBw, fAwxixifBwxi
    cdef double fAwxixifBwxixi
    cdef double gAphixetagBphix, gAphixetagBphixeta, gAphixetagBphiy, gAphixetagBphiyeta, gAphixetagBu
    cdef double gAphixetagBueta, gAphixetagBv, gAphixetagBveta, gAphixetagBw, gAphixetagBweta
    cdef double gAphixetagBwetaeta, gAphixgBphix, gAphixgBphixeta, gAphixgBphiy, gAphixgBphiyeta
    cdef double gAphixgBu, gAphixgBueta, gAphixgBv, gAphixgBveta, gAphixgBw
    cdef double gAphixgBweta, gAphixgBwetaeta, gAphiyetagBphix, gAphiyetagBphixeta, gAphiyetagBphiy
    cdef double gAphiyetagBphiyeta, gAphiyetagBu, gAphiyetagBueta, gAphiyetagBv, gAphiyetagBveta
    cdef double gAphiyetagBw, gAphiyetagBweta, gAphiyetagBwetaeta, gAphiygBphix, gAphiygBphixeta
    cdef double gAphiygBphiy, gAphiygBphiyeta, gAphiygBu, gAphiygBueta, gAphiygBv
    cdef double gAphiygBveta, gAphiygBw, gAphiygBweta, gAphiygBwetaeta, gAuetagBphix
    cdef double gAuetagBphixeta, gAuetagBphiy, gAuetagBphiyeta, gAuetagBu, gAuetagBueta
    cdef double gAuetagBv, gAuetagBveta, gAuetagBw, gAuetagBweta, gAuetagBwetaeta
    cdef double gAugBphix, gAugBphixeta, gAugBphiy, gAugBphiyeta, gAugBu
    cdef double gAugBueta, gAugBv, gAugBveta, gAugBw, gAugBweta
    cdef double gAugBwetaeta, gAvetagBphix, gAvetagBphixeta, gAvetagBphiy, gAvetagBphiyeta
    cdef double gAvetagBu, gAvetagBueta, gAvetagBv, gAvetagBveta, gAvetagBw
    cdef double gAvetagBweta, gAvetagBwetaeta, gAvgBphix, gAvgBphixeta, gAvgBphiy
    cdef double gAvgBphiyeta, gAvgBu, gAvgBueta, gAvgBv, gAvgBveta
    cdef double gAvgBw, gAvgBweta, gAvgBwetaeta, gAwetaetagBphix, gAwetaetagBphixeta
    cdef double gAwetaetagBphiy, gAwetaetagBphiyeta, gAwetaetagBu, gAwetaetagBueta, gAwetaetagBv
    cdef double gAwetaetagBveta, gAwetaetagBw, gAwetaetagBweta, gAwetaetagBwetaeta, gAwetagBphix
    cdef double gAwetagBphixeta, gAwetagBphiy, gAwetagBphiyeta, gAwetagBu, gAwetagBueta
    cdef double gAwetagBv, gAwetagBveta, gAwetagBw, gAwetagBweta, gAwetagBwetaeta
    cdef double gAwgBphix, gAwgBphixeta, gAwgBphiy, gAwgBphiyeta, gAwgBu
    cdef double gAwgBueta, gAwgBv, gAwgBveta, gAwgBw, gAwgBweta
    cdef double gAwgBwetaeta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    F = shell.ABD
    h = sum(shell.plyts)
    c1 = 4./(3.*h*h)
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

    fdim = 25*m*m*n*n

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
        E11 = F[0,6]
        E12 = F[0,7]
        E16 = F[0,8]
        A22 = F[1,1]
        A26 = F[1,2]
        B22 = F[1,4]
        B26 = F[1,5]
        E22 = F[1,7]
        E26 = F[1,8]
        A66 = F[2,2]
        B66 = F[2,5]
        E66 = F[2,8]
        D11 = F[3,3]
        D12 = F[3,4]
        D16 = F[3,5]
        F11 = F[3,6]
        F12 = F[3,7]
        F16 = F[3,8]
        D22 = F[4,4]
        D26 = F[4,5]
        F22 = F[4,7]
        F26 = F[4,8]
        D66 = F[5,5]
        F66 = F[5,8]
        H11 = F[6,6]
        H12 = F[6,7]
        H16 = F[6,8]
        H22 = F[7,7]
        H26 = F[7,8]
        H66 = F[8,8]
        A44 = F[9,9]
        A45 = F[9,10]
        D44 = F[9,11]
        D45 = F[9,12]
        A55 = F[10,10]
        D55 = F[10,12]
        F44 = F[11,11]
        F45 = F[11,12]
        F55 = F[12,12]

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
                fAphixfBwxixi = integral_ffpp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
                fAphixxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBphixxi = integral_fpfp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBphiy = integral_ffp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBphiyxi = integral_fpfp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphixxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBuxi = integral_fpfp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1u, x1ur, x2u, x2ur)
                fAphixxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBvxi = integral_fpfp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1v, x1vr, x2v, x2vr)
                fAphixxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1phix, x1phixr, x2phix, x2phixr)
                fAphixxifBwxi = integral_fpfp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
                fAphixxifBwxixi = integral_fpfpp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
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
                fAphiyfBwxixi = integral_ffpp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1w, x1wr, x2w, x2wr)
                fAphiyxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBphixxi = integral_fpfp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phix, x1phixr, x2phix, x2phixr)
                fAphiyxifBphiy = integral_ffp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBphiyxi = integral_fpfp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBuxi = integral_fpfp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1u, x1ur, x2u, x2ur)
                fAphiyxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBvxi = integral_fpfp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1v, x1vr, x2v, x2vr)
                fAphiyxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyxifBwxi = integral_fpfp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1w, x1wr, x2w, x2wr)
                fAphiyxifBwxixi = integral_fpfpp(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1w, x1wr, x2w, x2wr)
                fAufBphix = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1phix, x1phixr, x2phix, x2phixr)
                fAufBphixxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1phix, x1phixr, x2phix, x2phixr)
                fAufBphiy = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAufBphiyxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBuxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBv = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAufBvxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAufBw = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAufBwxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAufBwxixi = integral_ffpp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAuxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1u, x1ur, x2u, x2ur)
                fAuxifBphixxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1phix, x1phixr, x2phix, x2phixr)
                fAuxifBphiy = integral_ffp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1u, x1ur, x2u, x2ur)
                fAuxifBphiyxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAuxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAuxifBuxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAuxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAuxifBvxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAuxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1u, x1ur, x2u, x2ur)
                fAuxifBwxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAuxifBwxixi = integral_fpfpp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAvfBphix = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1phix, x1phixr, x2phix, x2phixr)
                fAvfBphixxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1phix, x1phixr, x2phix, x2phixr)
                fAvfBphiy = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAvfBphiyxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAvfBu = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAvfBuxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBvxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBw = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvfBwxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvfBwxixi = integral_ffpp(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1v, x1vr, x2v, x2vr)
                fAvxifBphixxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1phix, x1phixr, x2phix, x2phixr)
                fAvxifBphiy = integral_ffp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1v, x1vr, x2v, x2vr)
                fAvxifBphiyxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAvxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAvxifBuxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAvxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvxifBvxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAvxifBwxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvxifBwxixi = integral_fpfpp(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwfBphix = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1phix, x1phixr, x2phix, x2phixr)
                fAwfBphixxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1phix, x1phixr, x2phix, x2phixr)
                fAwfBphiy = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAwfBphiyxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAwfBu = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1u, x1ur, x2u, x2ur)
                fAwfBuxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1u, x1ur, x2u, x2ur)
                fAwfBv = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBvxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwfBwxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwfBwxixi = integral_ffpp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
                fAwxifBphixxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1phix, x1phixr, x2phix, x2phixr)
                fAwxifBphiy = integral_ffp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1w, x1wr, x2w, x2wr)
                fAwxifBphiyxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAwxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwxifBuxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1u, x1ur, x2u, x2ur)
                fAwxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwxifBvxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxixi = integral_fpfpp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxixifBphix = integral_ffpp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
                fAwxixifBphixxi = integral_fpfpp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
                fAwxixifBphiy = integral_ffpp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1w, x1wr, x2w, x2wr)
                fAwxixifBphiyxi = integral_fpfpp(k, i, x1phiy, x1phiyr, x2phiy, x2phiyr, x1w, x1wr, x2w, x2wr)
                fAwxixifBu = integral_ffpp(k, i, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwxixifBuxi = integral_fpfpp(k, i, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwxixifBv = integral_ffpp(k, i, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwxixifBvxi = integral_fpfpp(k, i, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwxixifBw = integral_ffpp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxixifBwxi = integral_fpfpp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxixifBwxixi = integral_fppfpp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

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
                        gAphixetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphixetagBweta = integral_fpfp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1w, y1wr, y2w, y2wr)
                        gAphixetagBwetaeta = integral_fpfpp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1w, y1wr, y2w, y2wr)
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
                        gAphixgBwetaeta = integral_ffpp(j, l, y1phix, y1phixr, y2phix, y2phixr, y1w, y1wr, y2w, y2wr)
                        gAphiyetagBphix = integral_ffp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBphixeta = integral_fpfp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphiyetagBphiy = integral_ffp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBphiyeta = integral_fpfp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBueta = integral_fpfp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1u, y1ur, y2u, y2ur)
                        gAphiyetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBveta = integral_fpfp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1v, y1vr, y2v, y2vr)
                        gAphiyetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiyetagBweta = integral_fpfp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
                        gAphiyetagBwetaeta = integral_fpfpp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
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
                        gAphiygBwetaeta = integral_ffpp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
                        gAuetagBphix = integral_ffp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1u, y1ur, y2u, y2ur)
                        gAuetagBphixeta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1phix, y1phixr, y2phix, y2phixr)
                        gAuetagBphiy = integral_ffp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1u, y1ur, y2u, y2ur)
                        gAuetagBphiyeta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAuetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAuetagBueta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAuetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAuetagBveta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAuetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1u, y1ur, y2u, y2ur)
                        gAuetagBweta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAuetagBwetaeta = integral_fpfpp(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAugBphix = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1phix, y1phixr, y2phix, y2phixr)
                        gAugBphixeta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1phix, y1phixr, y2phix, y2phixr)
                        gAugBphiy = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAugBphiyeta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAugBu = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAugBueta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAugBv = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAugBveta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAugBw = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAugBweta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAugBwetaeta = integral_ffpp(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAvetagBphix = integral_ffp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1v, y1vr, y2v, y2vr)
                        gAvetagBphixeta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1phix, y1phixr, y2phix, y2phixr)
                        gAvetagBphiy = integral_ffp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1v, y1vr, y2v, y2vr)
                        gAvetagBphiyeta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAvetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAvetagBueta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAvetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvetagBveta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1v, y1vr, y2v, y2vr)
                        gAvetagBweta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAvetagBwetaeta = integral_fpfpp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAvgBphix = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1phix, y1phixr, y2phix, y2phixr)
                        gAvgBphixeta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1phix, y1phixr, y2phix, y2phixr)
                        gAvgBphiy = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAvgBphiyeta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAvgBu = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAvgBueta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAvgBv = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBveta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBw = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAvgBweta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAvgBwetaeta = integral_ffpp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBphix = integral_ffpp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBphixeta = integral_fpfpp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBphiy = integral_ffpp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBphiyeta = integral_fpfpp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBu = integral_ffpp(l, j, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBueta = integral_fpfpp(l, j, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBv = integral_ffpp(l, j, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBveta = integral_fpfpp(l, j, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBw = integral_ffpp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBweta = integral_fpfpp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBwetaeta = integral_fppfpp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetagBphix = integral_ffp(l, j, y1phix, y1phixr, y2phix, y2phixr, y1w, y1wr, y2w, y2wr)
                        gAwetagBphixeta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1phix, y1phixr, y2phix, y2phixr)
                        gAwetagBphiy = integral_ffp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
                        gAwetagBphiyeta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAwetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAwetagBueta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1u, y1ur, y2u, y2ur)
                        gAwetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetagBveta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1v, y1vr, y2v, y2vr)
                        gAwetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetagBweta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetagBwetaeta = integral_fpfpp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBphix = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1phix, y1phixr, y2phix, y2phixr)
                        gAwgBphixeta = integral_ffp(j, l, y1w, y1wr, y2w, y2wr, y1phix, y1phixr, y2phix, y2phixr)
                        gAwgBphiy = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAwgBphiyeta = integral_ffp(j, l, y1w, y1wr, y2w, y2wr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAwgBu = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1u, y1ur, y2u, y2ur)
                        gAwgBueta = integral_ffp(j, l, y1w, y1wr, y2w, y2wr, y1u, y1ur, y2u, y2ur)
                        gAwgBv = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1v, y1vr, y2v, y2vr)
                        gAwgBveta = integral_ffp(j, l, y1w, y1wr, y2w, y2wr, y1v, y1vr, y2v, y2vr)
                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBweta = integral_ffp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBwetaeta = integral_ffpp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

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
                        k0c[c] = col+2
                        k0v[c] += 0.5*A12*b*fAuxifBw*gAugBw/r + 0.5*A26*a*fAufBw*gAuetagBw/r - 2*E11*b*c1*fAuxifBwxixi*gAugBw/(a*a) - 2*E12*c1*fAuxifBw*gAugBwetaeta/b - 2*E16*c1*fAufBwxixi*gAuetagBw/a - 4*E16*c1*fAuxifBwxi*gAugBweta/a - 2*E26*a*c1*fAufBw*gAuetagBwetaeta/(b*b) - 4*E66*c1*fAufBwxi*gAuetagBweta/b
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+3
                        k0v[c] += B11*b*fAuxifBphixxi*gAugBphix/a + B16*fAufBphixxi*gAuetagBphix + B16*fAuxifBphix*gAugBphixeta + B66*a*fAufBphix*gAuetagBphixeta/b - E11*b*c1*fAuxifBphixxi*gAugBphix/a - E16*c1*fAufBphixxi*gAuetagBphix - E16*c1*fAuxifBphix*gAugBphixeta - E66*a*c1*fAufBphix*gAuetagBphixeta/b
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+4
                        k0v[c] += B12*fAuxifBphiy*gAugBphiyeta + B16*b*fAuxifBphiyxi*gAugBphiy/a + B26*a*fAufBphiy*gAuetagBphiyeta/b + B66*fAufBphiyxi*gAuetagBphiy - E12*c1*fAuxifBphiy*gAugBphiyeta - E16*b*c1*fAuxifBphiyxi*gAugBphiy/a - E26*a*c1*fAufBphiy*gAuetagBphiyeta/b - E66*c1*fAufBphiyxi*gAuetagBphiy
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
                        k0c[c] = col+2
                        k0v[c] += 0.5*A22*a*fAvfBw*gAvetagBw/r + 0.5*A26*b*fAvxifBw*gAvgBw/r - 2*E12*c1*fAvfBwxixi*gAvetagBw/a - 2*E16*b*c1*fAvxifBwxixi*gAvgBw/(a*a) - 2*E22*a*c1*fAvfBw*gAvetagBwetaeta/(b*b) - 4*E26*c1*fAvfBwxi*gAvetagBweta/b - 2*E26*c1*fAvxifBw*gAvgBwetaeta/b - 4*E66*c1*fAvxifBwxi*gAvgBweta/a
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+3
                        k0v[c] += B12*fAvfBphixxi*gAvetagBphix + B16*b*fAvxifBphixxi*gAvgBphix/a + B26*a*fAvfBphix*gAvetagBphixeta/b + B66*fAvxifBphix*gAvgBphixeta - E12*c1*fAvfBphixxi*gAvetagBphix - E16*b*c1*fAvxifBphixxi*gAvgBphix/a - E26*a*c1*fAvfBphix*gAvetagBphixeta/b - E66*c1*fAvxifBphix*gAvgBphixeta
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+4
                        k0v[c] += B22*a*fAvfBphiy*gAvetagBphiyeta/b + B26*fAvfBphiyxi*gAvetagBphiy + B26*fAvxifBphiy*gAvgBphiyeta + B66*b*fAvxifBphiyxi*gAvgBphiy/a - E22*a*c1*fAvfBphiy*gAvetagBphiyeta/b - E26*c1*fAvfBphiyxi*gAvetagBphiy - E26*c1*fAvxifBphiy*gAvgBphiyeta - E66*b*c1*fAvxifBphiyxi*gAvgBphiy/a
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += 0.5*A12*b*fAwfBuxi*gAwgBu/r + 0.5*A26*a*fAwfBu*gAwgBueta/r - 2*E11*b*c1*fAwxixifBuxi*gAwgBu/(a*a) - 2*E12*c1*fAwfBuxi*gAwetaetagBu/b - 4*E16*c1*fAwxifBuxi*gAwetagBu/a - 2*E16*c1*fAwxixifBu*gAwgBueta/a - 2*E26*a*c1*fAwfBu*gAwetaetagBueta/(b*b) - 4*E66*c1*fAwxifBu*gAwetagBueta/b
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += 0.5*A22*a*fAwfBv*gAwgBveta/r + 0.5*A26*b*fAwfBvxi*gAwgBv/r - 2*E12*c1*fAwxixifBv*gAwgBveta/a - 2*E16*b*c1*fAwxixifBvxi*gAwgBv/(a*a) - 2*E22*a*c1*fAwfBv*gAwetaetagBveta/(b*b) - 2*E26*c1*fAwfBvxi*gAwetaetagBv/b - 4*E26*c1*fAwxifBv*gAwetagBveta/b - 4*E66*c1*fAwxifBvxi*gAwetagBv/a
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += 0.25*A22*a*b*fAwfBw*gAwgBw/(r*r) + A44*a*fAwfBw*gAwetagBweta/b + A45*fAwfBwxi*gAwetagBw + A45*fAwxifBw*gAwgBweta + A55*b*fAwxifBwxi*gAwgBw/a - 6*D44*a*c1*fAwfBw*gAwetagBweta/b - 6*D45*c1*fAwfBwxi*gAwetagBw - 6*D45*c1*fAwxifBw*gAwgBweta - 6*D55*b*c1*fAwxifBwxi*gAwgBw/a - E12*b*c1*fAwfBwxixi*gAwgBw/(a*r) - E12*b*c1*fAwxixifBw*gAwgBw/(a*r) - E22*a*c1*fAwfBw*gAwetaetagBw/(b*r) - E22*a*c1*fAwfBw*gAwgBwetaeta/(b*r) - 2*E26*c1*fAwfBwxi*gAwgBweta/r - 2*E26*c1*fAwxifBw*gAwetagBw/r + 9*F44*a*(c1*c1)*fAwfBw*gAwetagBweta/b + 9*F45*(c1*c1)*fAwfBwxi*gAwetagBw + 9*F45*(c1*c1)*fAwxifBw*gAwgBweta + 9*F55*b*(c1*c1)*fAwxifBwxi*gAwgBw/a + 4*H11*b*(c1*c1)*fAwxixifBwxixi*gAwgBw/(a*a*a) + 4*H12*(c1*c1)*fAwfBwxixi*gAwetaetagBw/(a*b) + 4*H12*(c1*c1)*fAwxixifBw*gAwgBwetaeta/(a*b) + 8*H16*(c1*c1)*fAwxifBwxixi*gAwetagBw/(a*a) + 8*H16*(c1*c1)*fAwxixifBwxi*gAwgBweta/(a*a) + 4*H22*a*(c1*c1)*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 8*H26*(c1*c1)*fAwfBwxi*gAwetaetagBweta/(b*b) + 8*H26*(c1*c1)*fAwxifBw*gAwetagBwetaeta/(b*b) + 16*H66*(c1*c1)*fAwxifBwxi*gAwetagBweta/(a*b)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+3
                        k0v[c] += 0.5*A45*a*fAwfBphix*gAwetagBphix + 0.5*A55*b*fAwxifBphix*gAwgBphix + 0.5*B12*b*fAwfBphixxi*gAwgBphix/r + 0.5*B26*a*fAwfBphix*gAwgBphixeta/r - 3*D45*a*c1*fAwfBphix*gAwetagBphix - 3*D55*b*c1*fAwxifBphix*gAwgBphix - 0.5*E12*b*c1*fAwfBphixxi*gAwgBphix/r - 0.5*E26*a*c1*fAwfBphix*gAwgBphixeta/r - 2*F11*b*c1*fAwxixifBphixxi*gAwgBphix/(a*a) - 2*F12*c1*fAwfBphixxi*gAwetaetagBphix/b - 4*F16*c1*fAwxifBphixxi*gAwetagBphix/a - 2*F16*c1*fAwxixifBphix*gAwgBphixeta/a - 2*F26*a*c1*fAwfBphix*gAwetaetagBphixeta/(b*b) + 4.5*F45*a*(c1*c1)*fAwfBphix*gAwetagBphix + 4.5*F55*b*(c1*c1)*fAwxifBphix*gAwgBphix - 4*F66*c1*fAwxifBphix*gAwetagBphixeta/b + 2*H11*b*(c1*c1)*fAwxixifBphixxi*gAwgBphix/(a*a) + 2*H12*(c1*c1)*fAwfBphixxi*gAwetaetagBphix/b + 4*H16*(c1*c1)*fAwxifBphixxi*gAwetagBphix/a + 2*H16*(c1*c1)*fAwxixifBphix*gAwgBphixeta/a + 2*H26*a*(c1*c1)*fAwfBphix*gAwetaetagBphixeta/(b*b) + 4*H66*(c1*c1)*fAwxifBphix*gAwetagBphixeta/b
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+4
                        k0v[c] += 0.5*A44*a*fAwfBphiy*gAwetagBphiy + 0.5*A45*b*fAwxifBphiy*gAwgBphiy + 0.5*B22*a*fAwfBphiy*gAwgBphiyeta/r + 0.5*B26*b*fAwfBphiyxi*gAwgBphiy/r - 3*D44*a*c1*fAwfBphiy*gAwetagBphiy - 3*D45*b*c1*fAwxifBphiy*gAwgBphiy - 0.5*E22*a*c1*fAwfBphiy*gAwgBphiyeta/r - 0.5*E26*b*c1*fAwfBphiyxi*gAwgBphiy/r - 2*F12*c1*fAwxixifBphiy*gAwgBphiyeta/a - 2*F16*b*c1*fAwxixifBphiyxi*gAwgBphiy/(a*a) - 2*F22*a*c1*fAwfBphiy*gAwetaetagBphiyeta/(b*b) - 2*F26*c1*fAwfBphiyxi*gAwetaetagBphiy/b - 4*F26*c1*fAwxifBphiy*gAwetagBphiyeta/b + 4.5*F44*a*(c1*c1)*fAwfBphiy*gAwetagBphiy + 4.5*F45*b*(c1*c1)*fAwxifBphiy*gAwgBphiy - 4*F66*c1*fAwxifBphiyxi*gAwetagBphiy/a + 2*H12*(c1*c1)*fAwxixifBphiy*gAwgBphiyeta/a + 2*H16*b*(c1*c1)*fAwxixifBphiyxi*gAwgBphiy/(a*a) + 2*H22*a*(c1*c1)*fAwfBphiy*gAwetaetagBphiyeta/(b*b) + 2*H26*(c1*c1)*fAwfBphiyxi*gAwetaetagBphiy/b + 4*H26*(c1*c1)*fAwxifBphiy*gAwetagBphiyeta/b + 4*H66*(c1*c1)*fAwxifBphiyxi*gAwetagBphiy/a
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+0
                        k0v[c] += B11*b*fAphixxifBuxi*gAphixgBu/a + B16*fAphixfBuxi*gAphixetagBu + B16*fAphixxifBu*gAphixgBueta + B66*a*fAphixfBu*gAphixetagBueta/b - E11*b*c1*fAphixxifBuxi*gAphixgBu/a - E16*c1*fAphixfBuxi*gAphixetagBu - E16*c1*fAphixxifBu*gAphixgBueta - E66*a*c1*fAphixfBu*gAphixetagBueta/b
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+1
                        k0v[c] += B12*fAphixxifBv*gAphixgBveta + B16*b*fAphixxifBvxi*gAphixgBv/a + B26*a*fAphixfBv*gAphixetagBveta/b + B66*fAphixfBvxi*gAphixetagBv - E12*c1*fAphixxifBv*gAphixgBveta - E16*b*c1*fAphixxifBvxi*gAphixgBv/a - E26*a*c1*fAphixfBv*gAphixetagBveta/b - E66*c1*fAphixfBvxi*gAphixetagBv
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+2
                        k0v[c] += 0.5*A45*a*fAphixfBw*gAphixgBweta + 0.5*A55*b*fAphixfBwxi*gAphixgBw + 0.5*B12*b*fAphixxifBw*gAphixgBw/r + 0.5*B26*a*fAphixfBw*gAphixetagBw/r - 3*D45*a*c1*fAphixfBw*gAphixgBweta - 3*D55*b*c1*fAphixfBwxi*gAphixgBw - 0.5*E12*b*c1*fAphixxifBw*gAphixgBw/r - 0.5*E26*a*c1*fAphixfBw*gAphixetagBw/r - 2*F11*b*c1*fAphixxifBwxixi*gAphixgBw/(a*a) - 2*F12*c1*fAphixxifBw*gAphixgBwetaeta/b - 2*F16*c1*fAphixfBwxixi*gAphixetagBw/a - 4*F16*c1*fAphixxifBwxi*gAphixgBweta/a - 2*F26*a*c1*fAphixfBw*gAphixetagBwetaeta/(b*b) + 4.5*F45*a*(c1*c1)*fAphixfBw*gAphixgBweta + 4.5*F55*b*(c1*c1)*fAphixfBwxi*gAphixgBw - 4*F66*c1*fAphixfBwxi*gAphixetagBweta/b + 2*H11*b*(c1*c1)*fAphixxifBwxixi*gAphixgBw/(a*a) + 2*H12*(c1*c1)*fAphixxifBw*gAphixgBwetaeta/b + 2*H16*(c1*c1)*fAphixfBwxixi*gAphixetagBw/a + 4*H16*(c1*c1)*fAphixxifBwxi*gAphixgBweta/a + 2*H26*a*(c1*c1)*fAphixfBw*gAphixetagBwetaeta/(b*b) + 4*H66*(c1*c1)*fAphixfBwxi*gAphixetagBweta/b
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+3
                        k0v[c] += 0.25*A55*a*b*fAphixfBphix*gAphixgBphix + D11*b*fAphixxifBphixxi*gAphixgBphix/a + D16*fAphixfBphixxi*gAphixetagBphix + D16*fAphixxifBphix*gAphixgBphixeta - 1.5*D55*a*b*c1*fAphixfBphix*gAphixgBphix + D66*a*fAphixfBphix*gAphixetagBphixeta/b - 2*F11*b*c1*fAphixxifBphixxi*gAphixgBphix/a - 2*F16*c1*fAphixfBphixxi*gAphixetagBphix - 2*F16*c1*fAphixxifBphix*gAphixgBphixeta + 2.25*F55*a*b*(c1*c1)*fAphixfBphix*gAphixgBphix - 2*F66*a*c1*fAphixfBphix*gAphixetagBphixeta/b + H11*b*(c1*c1)*fAphixxifBphixxi*gAphixgBphix/a + H16*(c1*c1)*fAphixfBphixxi*gAphixetagBphix + H16*(c1*c1)*fAphixxifBphix*gAphixgBphixeta + H66*a*(c1*c1)*fAphixfBphix*gAphixetagBphixeta/b
                        c += 1
                        k0r[c] = row+3
                        k0c[c] = col+4
                        k0v[c] += 0.25*A45*a*b*fAphixfBphiy*gAphixgBphiy + D12*fAphixxifBphiy*gAphixgBphiyeta + D16*b*fAphixxifBphiyxi*gAphixgBphiy/a + D26*a*fAphixfBphiy*gAphixetagBphiyeta/b - 1.5*D45*a*b*c1*fAphixfBphiy*gAphixgBphiy + D66*fAphixfBphiyxi*gAphixetagBphiy - 2*F12*c1*fAphixxifBphiy*gAphixgBphiyeta - 2*F16*b*c1*fAphixxifBphiyxi*gAphixgBphiy/a - 2*F26*a*c1*fAphixfBphiy*gAphixetagBphiyeta/b + 2.25*F45*a*b*(c1*c1)*fAphixfBphiy*gAphixgBphiy - 2*F66*c1*fAphixfBphiyxi*gAphixetagBphiy + H12*(c1*c1)*fAphixxifBphiy*gAphixgBphiyeta + H16*b*(c1*c1)*fAphixxifBphiyxi*gAphixgBphiy/a + H26*a*(c1*c1)*fAphixfBphiy*gAphixetagBphiyeta/b + H66*(c1*c1)*fAphixfBphiyxi*gAphixetagBphiy
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+0
                        k0v[c] += B12*fAphiyfBuxi*gAphiyetagBu + B16*b*fAphiyxifBuxi*gAphiygBu/a + B26*a*fAphiyfBu*gAphiyetagBueta/b + B66*fAphiyxifBu*gAphiygBueta - E12*c1*fAphiyfBuxi*gAphiyetagBu - E16*b*c1*fAphiyxifBuxi*gAphiygBu/a - E26*a*c1*fAphiyfBu*gAphiyetagBueta/b - E66*c1*fAphiyxifBu*gAphiygBueta
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+1
                        k0v[c] += B22*a*fAphiyfBv*gAphiyetagBveta/b + B26*fAphiyfBvxi*gAphiyetagBv + B26*fAphiyxifBv*gAphiygBveta + B66*b*fAphiyxifBvxi*gAphiygBv/a - E22*a*c1*fAphiyfBv*gAphiyetagBveta/b - E26*c1*fAphiyfBvxi*gAphiyetagBv - E26*c1*fAphiyxifBv*gAphiygBveta - E66*b*c1*fAphiyxifBvxi*gAphiygBv/a
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+2
                        k0v[c] += 0.5*A44*a*fAphiyfBw*gAphiygBweta + 0.5*A45*b*fAphiyfBwxi*gAphiygBw + 0.5*B22*a*fAphiyfBw*gAphiyetagBw/r + 0.5*B26*b*fAphiyxifBw*gAphiygBw/r - 3*D44*a*c1*fAphiyfBw*gAphiygBweta - 3*D45*b*c1*fAphiyfBwxi*gAphiygBw - 0.5*E22*a*c1*fAphiyfBw*gAphiyetagBw/r - 0.5*E26*b*c1*fAphiyxifBw*gAphiygBw/r - 2*F12*c1*fAphiyfBwxixi*gAphiyetagBw/a - 2*F16*b*c1*fAphiyxifBwxixi*gAphiygBw/(a*a) - 2*F22*a*c1*fAphiyfBw*gAphiyetagBwetaeta/(b*b) - 4*F26*c1*fAphiyfBwxi*gAphiyetagBweta/b - 2*F26*c1*fAphiyxifBw*gAphiygBwetaeta/b + 4.5*F44*a*(c1*c1)*fAphiyfBw*gAphiygBweta + 4.5*F45*b*(c1*c1)*fAphiyfBwxi*gAphiygBw - 4*F66*c1*fAphiyxifBwxi*gAphiygBweta/a + 2*H12*(c1*c1)*fAphiyfBwxixi*gAphiyetagBw/a + 2*H16*b*(c1*c1)*fAphiyxifBwxixi*gAphiygBw/(a*a) + 2*H22*a*(c1*c1)*fAphiyfBw*gAphiyetagBwetaeta/(b*b) + 4*H26*(c1*c1)*fAphiyfBwxi*gAphiyetagBweta/b + 2*H26*(c1*c1)*fAphiyxifBw*gAphiygBwetaeta/b + 4*H66*(c1*c1)*fAphiyxifBwxi*gAphiygBweta/a
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+3
                        k0v[c] += 0.25*A45*a*b*fAphiyfBphix*gAphiygBphix + D12*fAphiyfBphixxi*gAphiyetagBphix + D16*b*fAphiyxifBphixxi*gAphiygBphix/a + D26*a*fAphiyfBphix*gAphiyetagBphixeta/b - 1.5*D45*a*b*c1*fAphiyfBphix*gAphiygBphix + D66*fAphiyxifBphix*gAphiygBphixeta - 2*F12*c1*fAphiyfBphixxi*gAphiyetagBphix - 2*F16*b*c1*fAphiyxifBphixxi*gAphiygBphix/a - 2*F26*a*c1*fAphiyfBphix*gAphiyetagBphixeta/b + 2.25*F45*a*b*(c1*c1)*fAphiyfBphix*gAphiygBphix - 2*F66*c1*fAphiyxifBphix*gAphiygBphixeta + H12*(c1*c1)*fAphiyfBphixxi*gAphiyetagBphix + H16*b*(c1*c1)*fAphiyxifBphixxi*gAphiygBphix/a + H26*a*(c1*c1)*fAphiyfBphix*gAphiyetagBphixeta/b + H66*(c1*c1)*fAphiyxifBphix*gAphiygBphixeta
                        c += 1
                        k0r[c] = row+4
                        k0c[c] = col+4
                        k0v[c] += 0.25*A44*a*b*fAphiyfBphiy*gAphiygBphiy + D22*a*fAphiyfBphiy*gAphiyetagBphiyeta/b + D26*fAphiyfBphiyxi*gAphiyetagBphiy + D26*fAphiyxifBphiy*gAphiygBphiyeta - 1.5*D44*a*b*c1*fAphiyfBphiy*gAphiygBphiy + D66*b*fAphiyxifBphiyxi*gAphiygBphiy/a - 2*F22*a*c1*fAphiyfBphiy*gAphiyetagBphiyeta/b - 2*F26*c1*fAphiyfBphiyxi*gAphiyetagBphiy - 2*F26*c1*fAphiyxifBphiy*gAphiygBphiyeta + 2.25*F44*a*b*(c1*c1)*fAphiyfBphiy*gAphiygBphiy - 2*F66*b*c1*fAphiyxifBphiyxi*gAphiygBphiy/a + H22*a*(c1*c1)*fAphiyfBphiy*gAphiyetagBphiyeta/b + H26*(c1*c1)*fAphiyfBphiyxi*gAphiyetagBphiy + H26*(c1*c1)*fAphiyxifBphiy*gAphiygBphiyeta + H66*b*(c1*c1)*fAphiyxifBphiyxi*gAphiygBphiy/a

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fkG0(double Nxx, double Nyy, double Nxy, object shell,
         int size, int row0, int col0):
    cdef double r
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
    r = shell.r

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
    cdef double r
    cdef double a, b, rho, h, c1
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

    cdef double fAphixfBphix, fAphixfBu, fAphixfBwxi, fAphiyfBphiy, fAphiyfBv
    cdef double fAphiyfBw, fAufBphix, fAufBu, fAufBwxi, fAvfBphiy
    cdef double fAvfBv, fAvfBw, fAwfBphiy, fAwfBv, fAwfBw
    cdef double fAwxifBphix, fAwxifBu, fAwxifBwxi
    cdef double gAphixgBphix, gAphixgBu, gAphixgBw, gAphiygBphiy, gAphiygBv
    cdef double gAphiygBweta, gAugBphix, gAugBu, gAugBw, gAvgBphiy
    cdef double gAvgBv, gAvgBweta, gAwetagBphiy, gAwetagBv, gAwetagBweta
    cdef double gAwgBphix, gAwgBu, gAwgBw

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    rho = shell.rho
    h = sum(shell.plyts)
    c1 = 4./(3.*h*h)
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
                fAphixfBwxi = integral_ffp(i, k, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
                fAphiyfBphiy = integral_ff(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAphiyfBv = integral_ff(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1v, x1vr, x2v, x2vr)
                fAphiyfBw = integral_ff(i, k, x1phiy, x1phiyr, x2phiy, x2phiyr, x1w, x1wr, x2w, x2wr)
                fAufBphix = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1phix, x1phixr, x2phix, x2phixr)
                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBwxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAvfBphiy = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBw = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwfBphiy = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1phiy, x1phiyr, x2phiy, x2phiyr)
                fAwfBv = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBphix = integral_ffp(k, i, x1phix, x1phixr, x2phix, x2phixr, x1w, x1wr, x2w, x2wr)
                fAwxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAphixgBphix = integral_ff(j, l, y1phix, y1phixr, y2phix, y2phixr, y1phix, y1phixr, y2phix, y2phixr)
                        gAphixgBu = integral_ff(j, l, y1phix, y1phixr, y2phix, y2phixr, y1u, y1ur, y2u, y2ur)
                        gAphixgBw = integral_ff(j, l, y1phix, y1phixr, y2phix, y2phixr, y1w, y1wr, y2w, y2wr)
                        gAphiygBphiy = integral_ff(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAphiygBv = integral_ff(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1v, y1vr, y2v, y2vr)
                        gAphiygBweta = integral_ffp(j, l, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
                        gAugBphix = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1phix, y1phixr, y2phix, y2phixr)
                        gAugBu = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAugBw = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAvgBphiy = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1phiy, y1phiyr, y2phiy, y2phiyr)
                        gAvgBv = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBweta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetagBphiy = integral_ffp(l, j, y1phiy, y1phiyr, y2phiy, y2phiyr, y1w, y1wr, y2w, y2wr)
                        gAwetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetagBweta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBphix = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1phix, y1phixr, y2phix, y2phixr)
                        gAwgBu = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1u, y1ur, y2u, y2ur)
                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kMr[c] = row+0
                        kMc[c] = col+0
                        kMv[c] += 0.25*a*b*fAufBu*gAugBu*h*rho
                        c += 1
                        kMr[c] = row+0
                        kMc[c] = col+2
                        kMv[c] += 0.5*b*c1*(d*d*d)*fAufBwxi*gAugBw*h*rho + 0.125*b*c1*d*fAufBwxi*gAugBw*(h*h*h)*rho
                        c += 1
                        kMr[c] = row+0
                        kMc[c] = col+3
                        kMv[c] += 0.25*a*b*c1*(d*d*d)*fAufBphix*gAugBphix*h*rho + 0.0625*a*b*c1*d*fAufBphix*gAugBphix*(h*h*h)*rho - 0.25*a*b*d*fAufBphix*gAugBphix*h*rho
                        c += 1
                        kMr[c] = row+1
                        kMc[c] = col+1
                        kMv[c] += 0.25*a*b*fAvfBv*gAvgBv*h*rho
                        c += 1
                        kMr[c] = row+1
                        kMc[c] = col+2
                        kMv[c] += 0.5*a*c1*(d*d*d)*fAvfBw*gAvgBweta*h*rho + 0.125*a*c1*d*fAvfBw*gAvgBweta*(h*h*h)*rho
                        c += 1
                        kMr[c] = row+1
                        kMc[c] = col+4
                        kMv[c] += 0.25*a*b*c1*(d*d*d)*fAvfBphiy*gAvgBphiy*h*rho + 0.0625*a*b*c1*d*fAvfBphiy*gAvgBphiy*(h*h*h)*rho - 0.25*a*b*d*fAvfBphiy*gAvgBphiy*h*rho
                        c += 1
                        kMr[c] = row+2
                        kMc[c] = col+0
                        kMv[c] += 0.5*b*c1*(d*d*d)*fAwxifBu*gAwgBu*h*rho + 0.125*b*c1*d*fAwxifBu*gAwgBu*(h*h*h)*rho
                        c += 1
                        kMr[c] = row+2
                        kMc[c] = col+1
                        kMv[c] += 0.5*a*c1*(d*d*d)*fAwfBv*gAwetagBv*h*rho + 0.125*a*c1*d*fAwfBv*gAwetagBv*(h*h*h)*rho
                        c += 1
                        kMr[c] = row+2
                        kMc[c] = col+2
                        kMv[c] += 0.25*a*b*fAwfBw*gAwgBw*h*rho + a*(c1*c1)*(d*d*d*d*d*d)*fAwfBw*gAwetagBweta*h*rho/b + 1.25*a*(c1*c1)*(d*d*d*d)*fAwfBw*gAwetagBweta*(h*h*h)*rho/b + 0.1875*a*(c1*c1)*(d*d)*fAwfBw*gAwetagBweta*(h*h*h*h*h)*rho/b + 0.002232142857142857*a*(c1*c1)*fAwfBw*gAwetagBweta*(h*h*h*h*h*h*h)*rho/b + b*(c1*c1)*(d*d*d*d*d*d)*fAwxifBwxi*gAwgBw*h*rho/a + 1.25*b*(c1*c1)*(d*d*d*d)*fAwxifBwxi*gAwgBw*(h*h*h)*rho/a + 0.1875*b*(c1*c1)*(d*d)*fAwxifBwxi*gAwgBw*(h*h*h*h*h)*rho/a + 0.002232142857142857*b*(c1*c1)*fAwxifBwxi*gAwgBw*(h*h*h*h*h*h*h)*rho/a
                        c += 1
                        kMr[c] = row+2
                        kMc[c] = col+3
                        kMv[c] += 0.5*b*(c1*c1)*(d*d*d*d*d*d)*fAwxifBphix*gAwgBphix*h*rho + 0.625*b*(c1*c1)*(d*d*d*d)*fAwxifBphix*gAwgBphix*(h*h*h)*rho + 0.09375*b*(c1*c1)*(d*d)*fAwxifBphix*gAwgBphix*(h*h*h*h*h)*rho + 0.0011160714285714285*b*(c1*c1)*fAwxifBphix*gAwgBphix*(h*h*h*h*h*h*h)*rho - 0.5*b*c1*(d*d*d*d)*fAwxifBphix*gAwgBphix*h*rho - 0.25*b*c1*(d*d)*fAwxifBphix*gAwgBphix*(h*h*h)*rho - 0.00625*b*c1*fAwxifBphix*gAwgBphix*(h*h*h*h*h)*rho
                        c += 1
                        kMr[c] = row+2
                        kMc[c] = col+4
                        kMv[c] += 0.5*a*(c1*c1)*(d*d*d*d*d*d)*fAwfBphiy*gAwetagBphiy*h*rho + 0.625*a*(c1*c1)*(d*d*d*d)*fAwfBphiy*gAwetagBphiy*(h*h*h)*rho + 0.09375*a*(c1*c1)*(d*d)*fAwfBphiy*gAwetagBphiy*(h*h*h*h*h)*rho + 0.0011160714285714285*a*(c1*c1)*fAwfBphiy*gAwetagBphiy*(h*h*h*h*h*h*h)*rho - 0.5*a*c1*(d*d*d*d)*fAwfBphiy*gAwetagBphiy*h*rho - 0.25*a*c1*(d*d)*fAwfBphiy*gAwetagBphiy*(h*h*h)*rho - 0.00625*a*c1*fAwfBphiy*gAwetagBphiy*(h*h*h*h*h)*rho
                        c += 1
                        kMr[c] = row+3
                        kMc[c] = col+0
                        kMv[c] += 0.25*a*b*c1*(d*d*d)*fAphixfBu*gAphixgBu*h*rho + 0.0625*a*b*c1*d*fAphixfBu*gAphixgBu*(h*h*h)*rho - 0.25*a*b*d*fAphixfBu*gAphixgBu*h*rho
                        c += 1
                        kMr[c] = row+3
                        kMc[c] = col+2
                        kMv[c] += 0.5*b*(c1*c1)*(d*d*d*d*d*d)*fAphixfBwxi*gAphixgBw*h*rho + 0.625*b*(c1*c1)*(d*d*d*d)*fAphixfBwxi*gAphixgBw*(h*h*h)*rho + 0.09375*b*(c1*c1)*(d*d)*fAphixfBwxi*gAphixgBw*(h*h*h*h*h)*rho + 0.0011160714285714285*b*(c1*c1)*fAphixfBwxi*gAphixgBw*(h*h*h*h*h*h*h)*rho - 0.5*b*c1*(d*d*d*d)*fAphixfBwxi*gAphixgBw*h*rho - 0.25*b*c1*(d*d)*fAphixfBwxi*gAphixgBw*(h*h*h)*rho - 0.00625*b*c1*fAphixfBwxi*gAphixgBw*(h*h*h*h*h)*rho
                        c += 1
                        kMr[c] = row+3
                        kMc[c] = col+3
                        kMv[c] += 0.25*a*b*(c1*c1)*(d*d*d*d*d*d)*fAphixfBphix*gAphixgBphix*h*rho + 0.3125*a*b*(c1*c1)*(d*d*d*d)*fAphixfBphix*gAphixgBphix*(h*h*h)*rho + 0.046875*a*b*(c1*c1)*(d*d)*fAphixfBphix*gAphixgBphix*(h*h*h*h*h)*rho + 0.0005580357142857143*a*b*(c1*c1)*fAphixfBphix*gAphixgBphix*(h*h*h*h*h*h*h)*rho - 0.5*a*b*c1*(d*d*d*d)*fAphixfBphix*gAphixgBphix*h*rho - 0.25*a*b*c1*(d*d)*fAphixfBphix*gAphixgBphix*(h*h*h)*rho - 0.00625*a*b*c1*fAphixfBphix*gAphixgBphix*(h*h*h*h*h)*rho + 0.25*a*b*(d*d)*fAphixfBphix*gAphixgBphix*h*rho + 0.020833333333333332*a*b*fAphixfBphix*gAphixgBphix*(h*h*h)*rho
                        c += 1
                        kMr[c] = row+4
                        kMc[c] = col+1
                        kMv[c] += 0.25*a*b*c1*(d*d*d)*fAphiyfBv*gAphiygBv*h*rho + 0.0625*a*b*c1*d*fAphiyfBv*gAphiygBv*(h*h*h)*rho - 0.25*a*b*d*fAphiyfBv*gAphiygBv*h*rho
                        c += 1
                        kMr[c] = row+4
                        kMc[c] = col+2
                        kMv[c] += 0.5*a*(c1*c1)*(d*d*d*d*d*d)*fAphiyfBw*gAphiygBweta*h*rho + 0.625*a*(c1*c1)*(d*d*d*d)*fAphiyfBw*gAphiygBweta*(h*h*h)*rho + 0.09375*a*(c1*c1)*(d*d)*fAphiyfBw*gAphiygBweta*(h*h*h*h*h)*rho + 0.0011160714285714285*a*(c1*c1)*fAphiyfBw*gAphiygBweta*(h*h*h*h*h*h*h)*rho - 0.5*a*c1*(d*d*d*d)*fAphiyfBw*gAphiygBweta*h*rho - 0.25*a*c1*(d*d)*fAphiyfBw*gAphiygBweta*(h*h*h)*rho - 0.00625*a*c1*fAphiyfBw*gAphiygBweta*(h*h*h*h*h)*rho
                        c += 1
                        kMr[c] = row+4
                        kMc[c] = col+4
                        kMv[c] += 0.25*a*b*(c1*c1)*(d*d*d*d*d*d)*fAphiyfBphiy*gAphiygBphiy*h*rho + 0.3125*a*b*(c1*c1)*(d*d*d*d)*fAphiyfBphiy*gAphiygBphiy*(h*h*h)*rho + 0.046875*a*b*(c1*c1)*(d*d)*fAphiyfBphiy*gAphiygBphiy*(h*h*h*h*h)*rho + 0.0005580357142857143*a*b*(c1*c1)*fAphiyfBphiy*gAphiygBphiy*(h*h*h*h*h*h*h)*rho - 0.5*a*b*c1*(d*d*d*d)*fAphiyfBphiy*gAphiygBphiy*h*rho - 0.25*a*b*c1*(d*d)*fAphiyfBphiy*gAphiygBphiy*(h*h*h)*rho - 0.00625*a*b*c1*fAphiyfBphiy*gAphiygBphiy*(h*h*h*h*h)*rho + 0.25*a*b*(d*d)*fAphiyfBphiy*gAphiygBphiy*h*rho + 0.020833333333333332*a*b*fAphiyfBphiy*gAphiygBphiy*(h*h*h)*rho

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


def fkAx(double beta, double gamma, object shell,
         int size, int row0, int col0):
    cdef double r
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

    cdef double fAwfBw, fAwxifBw
    cdef double gAwgBw

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

                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kAxr[c] = row+2
                        kAxc[c] = col+2
                        kAxv[c] += -0.25*a*b*fAwfBw*gAwgBw*gamma - 0.5*b*beta*fAwxifBw*gAwgBw

    kAx = coo_matrix((kAxv, (kAxr, kAxc)), shape=(size, size))

    return kAx


def fkAy(double beta, object shell, int size, int row0, int col0):
    cdef double r
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
    r = shell.r

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
    cdef double r
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
    r = shell.r

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
