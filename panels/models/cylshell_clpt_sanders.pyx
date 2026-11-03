#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Cylindrical shells using the classical laminated plate theory (CLPT)
with the Sanders-Koiter kinematics

Matrices integrated analytically over the whole domain: the linear
constitutive stiffness :func:`fk0`, the geometric stiffness :func:`fkG0`,
the mass :func:`fkM`, the aerodynamic stiffness :func:`fkAx`, :func:`fkAy`
and the aerodynamic damping :func:`fcA`, as
:class:`scipy.sparse.coo_matrix`.

The degrees of freedom of each term of the approximation are
``u, v, w``; the degree of freedom ``k`` of the term ``(i, j)``, ``0 <=
i < m``, ``0 <= j < n``, is at the row ``row0 + 3*(j*m + i) + k``
of the matrices.

Notes
-----
For the cylinders `y = r \theta` is the arc length along the
circumference, `w` is positive outwards and `r + z \approx r`. With the rotations `\beta_x = -w_{,x}` and `\beta_y = -w_{,y} + \frac{v}{r}` of the
non-linear terms, the strains of the Sanders-Koiter kinematics, see
:mod:`panels.models`, are:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} \beta_x^2 \qquad
    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \beta_y^2 \qquad
    \gamma_{xy} = u_{,y} + v_{,x} + \beta_x \beta_y

    \kappa_{xx} = -w_{,xx} \qquad
    \kappa_{yy} = -w_{,yy} + \frac{v_{,y}}{r} \qquad
    \kappa_{xy} = -2 w_{,xy} + \frac{3}{2} \frac{v_{,x}}{r} - \frac{1}{2} \frac{u_{,y}}{r}

with the generalized stresses `N_{xx}, N_{yy}, N_{xy}, M_{xx}, M_{yy},
M_{xy}` given by the ``6 x 6`` matrix ``ABD`` of the CLPT.

The integrals over the whole domain are evaluated in closed form with the
integrals of Bardell's functions and their derivatives along `\xi = 2x/a -
1` and `\eta = 2y/b - 1`, which requires the constitutive matrix and the
moments of the density to be constant over the domain.

"""
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


def fk0(object shell, int size, int row0, int col0):
    r"""Linear constitutive stiffness matrix, integrated analytically

    Integrates the linear constitutive stiffness matrix of the shell over
    its whole domain for the constitutive matrix ``shell.ABD``, constant
    over the domain.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    k0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell` or ``shell.ABD`` is not a
        ``6 x 6`` matrix.

    Notes
    -----
    With the linear part `\boldsymbol{\varepsilon}_L =
    \boldsymbol{B}_L \boldsymbol{c}` of the generalized strains of the kinematics of the module,
    and the constitutive matrix `\boldsymbol{F}` (``shell.ABD``):

    .. math::

        \boldsymbol{K}_{C0} = \int_0^b \int_0^a \boldsymbol{B}_L^\top
        \boldsymbol{F} \boldsymbol{B}_L \, dx \, dy

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\xi` and `\eta`, with `x = a(\xi +
    1)/2`, `y = b(\eta + 1)/2` and `dx\,dy = (ab/4)\,d\xi\,d\eta`.

    """
    cdef double a, b, r
    cdef double [:, ::1] F
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, row, col
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66

    cdef long [:] k0r, k0c
    cdef double [:] k0v

    cdef double fAufBu, fAufBuxi, fAufBv, fAufBvxi, fAufBw, fAufBwxi
    cdef double fAufBwxixi, fAuxifBu, fAuxifBuxi, fAuxifBv, fAuxifBvxi, fAuxifBw
    cdef double fAuxifBwxi, fAuxifBwxixi, fAvfBu, fAvfBuxi, fAvfBv, fAvfBvxi
    cdef double fAvfBw, fAvfBwxi, fAvfBwxixi, fAvxifBu, fAvxifBuxi, fAvxifBv
    cdef double fAvxifBvxi, fAvxifBw, fAvxifBwxi, fAvxifBwxixi, fAwfBu, fAwfBuxi
    cdef double fAwfBv, fAwfBvxi, fAwfBw, fAwfBwxi, fAwfBwxixi, fAwxifBu
    cdef double fAwxifBuxi, fAwxifBv, fAwxifBvxi, fAwxifBw, fAwxifBwxi, fAwxifBwxixi
    cdef double fAwxixifBu, fAwxixifBuxi, fAwxixifBv, fAwxixifBvxi, fAwxixifBw, fAwxixifBwxi
    cdef double fAwxixifBwxixi
    cdef double gAuetagBu, gAuetagBueta, gAuetagBv, gAuetagBveta, gAuetagBw, gAuetagBweta
    cdef double gAuetagBwetaeta, gAugBu, gAugBueta, gAugBv, gAugBveta, gAugBw
    cdef double gAugBweta, gAugBwetaeta, gAvetagBu, gAvetagBueta, gAvetagBv, gAvetagBveta
    cdef double gAvetagBw, gAvetagBweta, gAvetagBwetaeta, gAvgBu, gAvgBueta, gAvgBv
    cdef double gAvgBveta, gAvgBw, gAvgBweta, gAvgBwetaeta, gAwetaetagBu, gAwetaetagBueta
    cdef double gAwetaetagBv, gAwetaetagBveta, gAwetaetagBw, gAwetaetagBweta, gAwetaetagBwetaeta, gAwetagBu
    cdef double gAwetagBueta, gAwetagBv, gAwetagBveta, gAwetagBw, gAwetagBweta, gAwetagBwetaeta
    cdef double gAwgBu, gAwgBueta, gAwgBv, gAwgBveta, gAwgBw, gAwgBweta
    cdef double gAwgBwetaeta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    r = shell.r
    F = shell.ABD
    if F.shape[0] != 6 or F.shape[1] != 6:
        raise ValueError('shell.ABD must be a 6 x 6 matrix')
    m = shell.m
    n = shell.n
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    fdim = 9*m*m*n*n

    k0r = np.zeros((fdim,), dtype=INT)
    k0c = np.zeros((fdim,), dtype=INT)
    k0v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        A11 = F[0,0]
        A12 = F[0,1]
        A16 = F[0,2]
        A22 = F[1,1]
        A26 = F[1,2]
        A66 = F[2,2]

        B11 = F[0,3]
        B12 = F[0,4]
        B16 = F[0,5]
        B22 = F[1,4]
        B26 = F[1,5]
        B66 = F[2,5]

        D11 = F[3,3]
        D12 = F[3,4]
        D16 = F[3,5]
        D22 = F[4,4]
        D26 = F[4,5]
        D66 = F[5,5]

        # k0
        c = -1
        for i in range(m):
            for k in range(m):

                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBuxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBv = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAufBvxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAufBw = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAufBwxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAufBwxixi = integral_ffpp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAuxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAuxifBuxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAuxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAuxifBvxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAuxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1u, x1ur, x2u, x2ur)
                fAuxifBwxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAuxifBwxixi = integral_fpfpp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAvfBu = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAvfBuxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBvxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBw = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvfBwxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvfBwxixi = integral_ffpp(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr)
                fAvxifBuxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1u, x1ur, x2u, x2ur)
                fAvxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvxifBvxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAvxifBwxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvxifBwxixi = integral_fpfpp(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwfBu = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1u, x1ur, x2u, x2ur)
                fAwfBuxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1u, x1ur, x2u, x2ur)
                fAwfBv = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBvxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwfBwxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwfBwxixi = integral_ffpp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwxifBuxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1u, x1ur, x2u, x2ur)
                fAwxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwxifBvxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxixi = integral_fpfpp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
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

                        gAuetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAuetagBueta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAuetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAuetagBveta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAuetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1u, y1ur, y2u, y2ur)
                        gAuetagBweta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAuetagBwetaeta = integral_fpfpp(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAugBu = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAugBueta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAugBv = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAugBveta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAugBw = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAugBweta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAugBwetaeta = integral_ffpp(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAvetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr)
                        gAvetagBueta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAvetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvetagBveta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1v, y1vr, y2v, y2vr)
                        gAvetagBweta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAvetagBwetaeta = integral_fpfpp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAvgBu = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAvgBueta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1u, y1ur, y2u, y2ur)
                        gAvgBv = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBveta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBw = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAvgBweta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAvgBwetaeta = integral_ffpp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBu = integral_ffpp(l, j, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBueta = integral_fpfpp(l, j, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBv = integral_ffpp(l, j, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBveta = integral_fpfpp(l, j, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBw = integral_ffpp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBweta = integral_fpfpp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetaetagBwetaeta = integral_fppfpp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAwetagBueta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1u, y1ur, y2u, y2ur)
                        gAwetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetagBveta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1v, y1vr, y2v, y2vr)
                        gAwetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetagBweta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetagBwetaeta = integral_fpfpp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
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
                        k0v[c] += A11*b*fAuxifBuxi*gAugBu/a + A16*fAufBuxi*gAuetagBu + A16*fAuxifBu*gAugBueta + A66*a*fAufBu*gAuetagBueta/b - 0.5*B16*fAufBuxi*gAuetagBu/r - 0.5*B16*fAuxifBu*gAugBueta/r - B66*a*fAufBu*gAuetagBueta/(b*r) + 0.25*D66*a*fAufBu*gAuetagBueta/(b*(r*r))
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+1
                        k0v[c] += A12*fAuxifBv*gAugBveta + A16*b*fAuxifBvxi*gAugBv/a + A26*a*fAufBv*gAuetagBveta/b + A66*fAufBvxi*gAuetagBv + B12*fAuxifBv*gAugBveta/r + 1.5*B16*b*fAuxifBvxi*gAugBv/(a*r) + 0.5*B26*a*fAufBv*gAuetagBveta/(b*r) + B66*fAufBvxi*gAuetagBv/r - 0.5*D26*a*fAufBv*gAuetagBveta/(b*(r*r)) - 0.75*D66*fAufBvxi*gAuetagBv/(r*r)
                        c += 1
                        k0r[c] = row+0
                        k0c[c] = col+2
                        k0v[c] += 0.5*A12*b*fAuxifBw*gAugBw/r + 0.5*A26*a*fAufBw*gAuetagBw/r - 2*B11*b*fAuxifBwxixi*gAugBw/(a*a) - 2*B12*fAuxifBw*gAugBwetaeta/b - 2*B16*fAufBwxixi*gAuetagBw/a - 4*B16*fAuxifBwxi*gAugBweta/a - 0.25*B26*a*fAufBw*gAuetagBw/(r*r) - 2*B26*a*fAufBw*gAuetagBwetaeta/(b*b) - 4*B66*fAufBwxi*gAuetagBweta/b + D16*fAufBwxixi*gAuetagBw/(a*r) + D26*a*fAufBw*gAuetagBwetaeta/((b*b)*r) + 2*D66*fAufBwxi*gAuetagBweta/(b*r)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+0
                        k0v[c] += A12*fAvfBuxi*gAvetagBu + A16*b*fAvxifBuxi*gAvgBu/a + A26*a*fAvfBu*gAvetagBueta/b + A66*fAvxifBu*gAvgBueta + B12*fAvfBuxi*gAvetagBu/r + 1.5*B16*b*fAvxifBuxi*gAvgBu/(a*r) + 0.5*B26*a*fAvfBu*gAvetagBueta/(b*r) + B66*fAvxifBu*gAvgBueta/r - 0.5*D26*a*fAvfBu*gAvetagBueta/(b*(r*r)) - 0.75*D66*fAvxifBu*gAvgBueta/(r*r)
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+1
                        k0v[c] += A22*a*fAvfBv*gAvetagBveta/b + A26*fAvfBvxi*gAvetagBv + A26*fAvxifBv*gAvgBveta + A66*b*fAvxifBvxi*gAvgBv/a + 2*B22*a*fAvfBv*gAvetagBveta/(b*r) + 2.5*B26*fAvfBvxi*gAvetagBv/r + 2.5*B26*fAvxifBv*gAvgBveta/r + 3*B66*b*fAvxifBvxi*gAvgBv/(a*r) + D22*a*fAvfBv*gAvetagBveta/(b*(r*r)) + 1.5*D26*fAvfBvxi*gAvetagBv/(r*r) + 1.5*D26*fAvxifBv*gAvgBveta/(r*r) + 2.25*D66*b*fAvxifBvxi*gAvgBv/(a*(r*r))
                        c += 1
                        k0r[c] = row+1
                        k0c[c] = col+2
                        k0v[c] += 0.5*A22*a*fAvfBw*gAvetagBw/r + 0.5*A26*b*fAvxifBw*gAvgBw/r - 2*B12*fAvfBwxixi*gAvetagBw/a - 2*B16*b*fAvxifBwxixi*gAvgBw/(a*a) + 0.5*B22*a*fAvfBw*gAvetagBw/(r*r) - 2*B22*a*fAvfBw*gAvetagBwetaeta/(b*b) + 0.75*B26*b*fAvxifBw*gAvgBw/(r*r) - 4*B26*fAvfBwxi*gAvetagBweta/b - 2*B26*fAvxifBw*gAvgBwetaeta/b - 4*B66*fAvxifBwxi*gAvgBweta/a - 2*D12*fAvfBwxixi*gAvetagBw/(a*r) - 3*D16*b*fAvxifBwxixi*gAvgBw/((a*a)*r) - 2*D22*a*fAvfBw*gAvetagBwetaeta/((b*b)*r) - 4*D26*fAvfBwxi*gAvetagBweta/(b*r) - 3*D26*fAvxifBw*gAvgBwetaeta/(b*r) - 6*D66*fAvxifBwxi*gAvgBweta/(a*r)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+0
                        k0v[c] += 0.5*A12*b*fAwfBuxi*gAwgBu/r + 0.5*A26*a*fAwfBu*gAwgBueta/r - 2*B11*b*fAwxixifBuxi*gAwgBu/(a*a) - 2*B12*fAwfBuxi*gAwetaetagBu/b - 4*B16*fAwxifBuxi*gAwetagBu/a - 2*B16*fAwxixifBu*gAwgBueta/a - 0.25*B26*a*fAwfBu*gAwgBueta/(r*r) - 2*B26*a*fAwfBu*gAwetaetagBueta/(b*b) - 4*B66*fAwxifBu*gAwetagBueta/b + D16*fAwxixifBu*gAwgBueta/(a*r) + D26*a*fAwfBu*gAwetaetagBueta/((b*b)*r) + 2*D66*fAwxifBu*gAwetagBueta/(b*r)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+1
                        k0v[c] += 0.5*A22*a*fAwfBv*gAwgBveta/r + 0.5*A26*b*fAwfBvxi*gAwgBv/r - 2*B12*fAwxixifBv*gAwgBveta/a - 2*B16*b*fAwxixifBvxi*gAwgBv/(a*a) + 0.5*B22*a*fAwfBv*gAwgBveta/(r*r) - 2*B22*a*fAwfBv*gAwetaetagBveta/(b*b) + 0.75*B26*b*fAwfBvxi*gAwgBv/(r*r) - 2*B26*fAwfBvxi*gAwetaetagBv/b - 4*B26*fAwxifBv*gAwetagBveta/b - 4*B66*fAwxifBvxi*gAwetagBv/a - 2*D12*fAwxixifBv*gAwgBveta/(a*r) - 3*D16*b*fAwxixifBvxi*gAwgBv/((a*a)*r) - 2*D22*a*fAwfBv*gAwetaetagBveta/((b*b)*r) - 3*D26*fAwfBvxi*gAwetaetagBv/(b*r) - 4*D26*fAwxifBv*gAwetagBveta/(b*r) - 6*D66*fAwxifBvxi*gAwetagBv/(a*r)
                        c += 1
                        k0r[c] = row+2
                        k0c[c] = col+2
                        k0v[c] += 0.25*A22*a*b*fAwfBw*gAwgBw/(r*r) - B12*b*fAwfBwxixi*gAwgBw/(a*r) - B12*b*fAwxixifBw*gAwgBw/(a*r) - B22*a*fAwfBw*gAwetaetagBw/(b*r) - B22*a*fAwfBw*gAwgBwetaeta/(b*r) - 2*B26*fAwfBwxi*gAwgBweta/r - 2*B26*fAwxifBw*gAwetagBw/r + 4*D11*b*fAwxixifBwxixi*gAwgBw/(a*a*a) + 4*D12*fAwfBwxixi*gAwetaetagBw/(a*b) + 4*D12*fAwxixifBw*gAwgBwetaeta/(a*b) + 8*D16*fAwxifBwxixi*gAwetagBw/(a*a) + 8*D16*fAwxixifBwxi*gAwgBweta/(a*a) + 4*D22*a*fAwfBw*gAwetaetagBwetaeta/(b*b*b) + 8*D26*fAwfBwxi*gAwetaetagBweta/(b*b) + 8*D26*fAwxifBw*gAwetagBwetaeta/(b*b) + 16*D66*fAwxifBwxi*gAwetagBweta/(a*b)

    k0 = coo_matrix((k0v, (k0r, k0c)), shape=(size, size))

    return k0


def fkG0(double Nxx, double Nyy, double Nxy, object shell,
         int size, int row0, int col0):
    r"""Geometric stiffness matrix of a constant membrane stress state

    Integrates, analytically over the whole domain, the geometric
    stiffness matrix of the membrane stress resultants ``Nxx``, ``Nyy``
    and ``Nxy``, constant over the domain.

    Parameters
    ----------
    Nxx, Nyy, Nxy : float
        Membrane stress resultants, force per unit length, tension positive.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    kG0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With `\boldsymbol{N} = \begin{bmatrix} N_{xx} & N_{xy} \\ N_{xy} &
    N_{yy} \end{bmatrix}` and `\boldsymbol{G} \boldsymbol{c} = \{\beta_x, \beta_y\}^\top` the
    rotations of the non-linear terms of the membrane strains:

    .. math::

        \boldsymbol{K}_{G} = \int_0^b \int_0^a \boldsymbol{G}^\top
        \boldsymbol{N} \boldsymbol{G} \, dx \, dy

    which is the Hessian of the work of the constant stress resultants on
    the non-linear part of the membrane strains of the kinematics of the module.

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\xi` and `\eta`, with `x = a(\xi +
    1)/2`, `y = b(\eta + 1)/2` and `dx\,dy = (ab/4)\,d\xi\,d\eta`.

    """
    cdef double a, b, r
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, k, j, l, c, row, col

    cdef long [:] kG0r, kG0c
    cdef double [:] kG0v

    cdef double fAvfBv, fAvfBw, fAvfBwxi, fAwfBv, fAwfBw, fAwfBwxi
    cdef double fAwxifBv, fAwxifBw, fAwxifBwxi
    cdef double gAvgBv, gAvgBw, gAvgBweta, gAwetagBv, gAwetagBw, gAwetagBweta
    cdef double gAwgBv, gAwgBw, gAwgBweta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    r = shell.r
    m = shell.m
    n = shell.n
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    fdim = 4*m*m*n*n

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kG0
        c = -1
        for i in range(m):
            for k in range(m):

                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBw = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvfBwxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwfBv = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwfBwxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAvgBv = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBw = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAvgBweta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetagBweta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBv = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1v, y1vr, y2v, y2vr)
                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBweta = integral_ffp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kG0r[c] = row+1
                        kG0c[c] = col+1
                        kG0v[c] += 0.25*Nyy*a*b*fAvfBv*gAvgBv/(r*r)
                        c += 1
                        kG0r[c] = row+1
                        kG0c[c] = col+2
                        kG0v[c] += -0.5*Nxy*b*fAvfBwxi*gAvgBw/r - 0.5*Nyy*a*fAvfBw*gAvgBweta/r
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+1
                        kG0v[c] += -0.5*Nxy*b*fAwxifBv*gAwgBv/r - 0.5*Nyy*a*fAwfBv*gAwetagBv/r
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*fAwfBwxi*gAwetagBw + Nxy*fAwxifBw*gAwgBweta + Nyy*a*fAwfBw*gAwetagBweta/b

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0


def fkM(object shell, int size, int row0, int col0):
    r"""Mass matrix, integrated analytically

    Integrates the consistent mass matrix of the shell over its whole
    domain, with the moments of the density of the laminate
    ``shell.lam``, constant over the domain.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    kM : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With the displacement field through the thickness `\{u(z), v(z),
    w(z)\}^\top = \boldsymbol{S}(z) \boldsymbol{c}` of the kinematics of the module, `z` measured
    from the reference surface:

    .. math::

        \boldsymbol{M} = \int_0^b \int_0^a \int_{d - h/2}^{d + h/2} \rho \,
        \boldsymbol{S}(z)^\top \boldsymbol{S}(z) \, dz \, dx \, dy

    with the offset `d` (``shell.offset``). The integral through the
    thickness gives the moments `I_n = \int \rho z^n dz`, `n = 0, 1, 2`,
    ``intrho``, ``intrhoz`` and ``intrhoz2`` of ``shell.lam``.

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\xi` and `\eta`, with `x = a(\xi +
    1)/2`, `y = b(\eta + 1)/2` and `dx\,dy = (ab/4)\,d\xi\,d\eta`.

    """
    cdef double a, b, r, I0, I1, I2
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, k, j, l, c, row, col

    cdef long [:] kMr, kMc
    cdef double [:] kMv

    cdef double fAufBu, fAufBwxi, fAvfBv, fAvfBw, fAwfBv, fAwfBw
    cdef double fAwxifBu, fAwxifBwxi
    cdef double gAugBu, gAugBw, gAvgBv, gAvgBweta, gAwetagBv, gAwetagBweta
    cdef double gAwgBu, gAwgBw

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    r = shell.r
    I0 = shell.lam.intrho
    I1 = shell.lam.intrhoz
    I2 = shell.lam.intrhoz2
    m = shell.m
    n = shell.n
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    fdim = 7*m*m*n*n

    kMr = np.zeros((fdim,), dtype=INT)
    kMc = np.zeros((fdim,), dtype=INT)
    kMv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kM
        c = -1
        for i in range(m):
            for k in range(m):

                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBwxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBw = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwfBv = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAugBu = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAugBw = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, y2wr)
                        gAvgBv = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBweta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetagBweta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBu = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1u, y1ur, y2u, y2ur)
                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kMr[c] = row+0
                        kMc[c] = col+0
                        kMv[c] += 0.25*I0*a*b*fAufBu*gAugBu
                        c += 1
                        kMr[c] = row+0
                        kMc[c] = col+2
                        kMv[c] += -0.5*I1*b*fAufBwxi*gAugBw
                        c += 1
                        kMr[c] = row+1
                        kMc[c] = col+1
                        kMv[c] += 0.25*I0*a*b*fAvfBv*gAvgBv + 0.5*I1*a*b*fAvfBv*gAvgBv/r + 0.25*I2*a*b*fAvfBv*gAvgBv/(r*r)
                        c += 1
                        kMr[c] = row+1
                        kMc[c] = col+2
                        kMv[c] += -0.5*I1*a*fAvfBw*gAvgBweta - 0.5*I2*a*fAvfBw*gAvgBweta/r
                        c += 1
                        kMr[c] = row+2
                        kMc[c] = col+0
                        kMv[c] += -0.5*I1*b*fAwxifBu*gAwgBu
                        c += 1
                        kMr[c] = row+2
                        kMc[c] = col+1
                        kMv[c] += -0.5*I1*a*fAwfBv*gAwetagBv - 0.5*I2*a*fAwfBv*gAwetagBv/r
                        c += 1
                        kMr[c] = row+2
                        kMc[c] = col+2
                        kMv[c] += 0.25*I0*a*b*fAwfBw*gAwgBw + I2*a*fAwfBw*gAwetagBweta/b + I2*b*fAwxifBwxi*gAwgBw/a

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


#NOTE the aerodynamic matrices depend only on w and are therefore the same as
#     for the Donnell kinematics
def fkAx(double beta, double gamma, object shell,
         int size, int row0, int col0):
    r"""The aerodynamic stiffness matrix of the flow along x, those of the Donnell kinematics

    The aerodynamic matrices depend only on `w` and are the same as for
    the Donnell kinematics: this function returns
    :func:`panels.models.cylshell_clpt_donnell.fkAx` with the same parameters.

    Parameters
    ----------
    beta : float
        Aerodynamic pressure parameter `\beta` of the piston theory, a
        pressure, see :meth:`.Shell.calc_kA`.
    gamma : float
        Curvature correction `\gamma` of the piston theory, a pressure per
        unit length.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    kAx : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. The matrix is assembled in full.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    See :func:`panels.models.cylshell_clpt_donnell.fkAx` for the integral.

    """
    from .cylshell_clpt_donnell import fkAx as donnell_fkAx
    return donnell_fkAx(beta, gamma, shell, size, row0, col0)


def fkAy(double beta, object shell, int size, int row0, int col0):
    r"""The aerodynamic stiffness matrix of the flow along y, those of the Donnell kinematics

    The aerodynamic matrices depend only on `w` and are the same as for
    the Donnell kinematics: this function returns
    :func:`panels.models.cylshell_clpt_donnell.fkAy` with the same parameters.

    Parameters
    ----------
    beta : float
        Aerodynamic pressure parameter `\beta` of the piston theory, a
        pressure, see :meth:`.Shell.calc_kA`.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    kAy : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. The matrix is assembled in full.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    See :func:`panels.models.cylshell_clpt_donnell.fkAy` for the integral.

    """
    from .cylshell_clpt_donnell import fkAy as donnell_fkAy
    return donnell_fkAy(beta, shell, size, row0, col0)


def fcA(double aeromu, object shell, int size, int row0, int col0):
    r"""The aerodynamic damping matrix, those of the Donnell kinematics

    The aerodynamic matrices depend only on `w` and are the same as for
    the Donnell kinematics: this function returns
    :func:`panels.models.cylshell_clpt_donnell.fcA` with the same parameters.

    Parameters
    ----------
    aeromu : float
        Aerodynamic damping coefficient `\mu` of the piston theory, a
        pressure per unit velocity.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    cA : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    See :func:`panels.models.cylshell_clpt_donnell.fcA` for the integral.

    """
    from .cylshell_clpt_donnell import fcA as donnell_fcA
    return donnell_fcA(aeromu, shell, size, row0, col0)
