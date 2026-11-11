#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Flat plates, prototype B of the total-Lagrangian third-order shear
deformation theory (TSDT)

Numerically integrated kernels of the model ``'plate_tsdt_tlB_proto'``: the
constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the internal force vector :func:`calc_fint`, the product
of the tangent stiffness matrix by a vector :func:`calc_kTv`. The mass and
aerodynamic matrices, ``fkM_num``, ``fkAx_num`` and ``fkAy_num``, are those
of :mod:`panels.models.plate_tsdt_donnell_num`. The constitutive matrix ``Finput`` is the
``13 x 13`` matrix ``ABD`` of :class:`.Shell` for this model.

The degrees of freedom of each term of the approximation are
``u, v, w, phix, phiy``; the degree of freedom ``k`` of the term ``(i, j)``,
``0 <= i < m``, ``0 <= j < n``, is at the row ``row0 + 5*(j*m + i) + k``
of the matrices.

Notes
-----
Prototype of the comparison of two total-Lagrangian TSDT. The deformed
position of the point at `z` is `\mathbf{X} + z \mathbf{d} + z^3
\mathbf{q}`, `\mathbf{q} = -c_1 (\mathbf{d} - \mathbf{N})`, with the
deformed mid-surface `\mathbf{X}`, the second-order director `\mathbf{d}` of
:mod:`panels.models.plate_fsdt_tl_num`, the normal `\mathbf{N} =
\mathbf{X}_{,x} \times \mathbf{X}_{,y}` and `c_1 = 4/(3 h^2)`, whose
linear part is the field of Reddy [1]_. The strains are the coefficients of
`z^0`, `z`, `z^3` of the in-plane and of `z^0`, `z^2` of the transverse
shear Green-Lagrange strains: those of
:mod:`panels.models.plate_fsdt_tl_num` and

.. math::

    \kappa^{(3)}_{xx} = -c_1 (\kappa_{xx} + \mathbf{X}_{,xx} \cdot
    \mathbf{N}) \qquad
    \kappa^{(3)}_{yy} = -c_1 (\kappa_{yy} + \mathbf{X}_{,yy} \cdot
    \mathbf{N}) \qquad
    \kappa^{(3)}_{xy} = -c_1 (\kappa_{xy} + 2 \mathbf{X}_{,xy} \cdot
    \mathbf{N})

    \gamma^{(2)}_{\alpha} = -3 c_1 \gamma_\alpha

The linear part is that of :mod:`panels.models.plate_tsdt_donnell`.

The numerical integration over the domain of
:meth:`.Shell.integration_limits` admits a constitutive matrix and moments
of the density varying over the domain, a partial domain and the
non-linear terms of the Ritz constants.

References
----------
.. [1] Reddy, J. N., "Mechanics of Laminated Composite Plates and Shells:
   Theory and Analysis", 2nd ed., CRC Press, 2003.
   https://doi.org/10.1201/b12409

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
#NOTE the linear kernels, independent of the non-linear strains, are those
#     of plate_tsdt_donnell
from .plate_tsdt_donnell_num import fkM_num, fkAx_num, fkAy_num


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 5
cdef int NE = 13


def fkC_num(double [::1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    r"""Constitutive stiffness matrix, integrated numerically

    With ``NLgeom=0``, the linear constitutive stiffness matrix
    ``KC0``; with ``NLgeom=1``, the matrix ``KC0 + KCNL + KGNL`` at the
    configuration ``cs``, which, added to :func:`fkG_num` at the same
    ``cs``, gives the tangent stiffness matrix.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements (the rotations in radians).
    Finput : array-like
        Constitutive matrix of the generalized strains of the model, shape
        ``(13, 13)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 13, 13)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.
    NLgeom : int, optional
        ``1`` to include the non-linear terms of ``cs``, default ``0``.

    Returns
    -------
    kC : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape or ``shell`` is not a
        :class:`.Shell`.

    Notes
    -----
    With the generalized derivatives at the integration point `\boldsymbol{d}
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the rotations and their gradients), the strains
    `\boldsymbol{\varepsilon}(\boldsymbol{d})`, cubic, `\boldsymbol{B} =
    \partial \boldsymbol{\varepsilon} / \partial \boldsymbol{d}`, the
    stresses `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}`, `\boldsymbol{H}(\boldsymbol{\sigma},
    \boldsymbol{d}) = \sum_r \sigma_r \, \partial^2 \varepsilon_r /
    \partial \boldsymbol{d}^2`, linear in `\boldsymbol{d}`, and
    `\boldsymbol{H}_0(\boldsymbol{\sigma}) =
    \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{0})`:

    .. math::

        \boldsymbol{K}_C = \int \boldsymbol{\Phi}^\top \left[\boldsymbol{B}^\top
        \boldsymbol{F} \boldsymbol{B} + \boldsymbol{H}(\boldsymbol{\sigma},
        \boldsymbol{d}) - \boldsymbol{H}_0(\boldsymbol{\sigma}_L)\right]
        \boldsymbol{\Phi} \, dA

    with `\boldsymbol{\sigma}_L` the stresses of the linear strains, whose
    term `\boldsymbol{H}_0(\boldsymbol{\sigma}_L)` is assembled by
    :func:`fkG_num`, such that ``fkC_num(NLgeom=1) + fkG_num`` is the exact
    Jacobian of :func:`calc_fint`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \le x \le x_2`, `y_1 \le y \le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

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

    cdef int i, j, k, l, c, p, q, r, s, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy
    cdef double phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy, u_xx
    cdef double u_yy, u_xy, v_xx, v_yy, v_xy
    cdef double e[13]
    cdef double eL[13]
    cdef double sg[13]
    cdef double sgL[13]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAw_xx
    cdef double fAphix, fAphix_x, fAphiy, fAphiy_x, fAu_xx, fAv_xx, gAu
    cdef double gAu_y, gAv, gAv_y, gAw, gAw_y, gAw_yy, gAphix
    cdef double gAphix_y, gAphiy, gAphiy_y, gAu_yy, gAv_yy, fBu, fBu_x
    cdef double fBv, fBv_x, fBw, fBw_x, fBw_xx, fBphix, fBphix_x
    cdef double fBphiy, fBphiy_x, fBu_xx, fBv_xx, gBu, gBu_y, gBv
    cdef double gBv_y, gBw, gBw_y, gBw_yy, gBphix, gBphix_y, gBphiy
    cdef double gBphiy_y, gBu_yy, gBv_yy
    cdef double Bm[273]
    cdef double FB[273]
    cdef double K[441]
    cdef double TA[105]
    cdef double pA[21]
    cdef double pB[21]

    cdef long [::1] kCr, kCc
    cdef double [::1] kCv

    # F as 4-D matrix, must be [nx, ny, NE, NE], when there is one F[NE, NE]
    # for each of the nx * ny integration points
    cdef double F[13 * 13]
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

    ftab = np.zeros((13, m), dtype=DOUBLE)
    gtab = np.zeros((13, n), dtype=DOUBLE)

    fdim = 25*m*m*n*n
    kCr = np.zeros((fdim,), dtype=INT)
    kCc = np.zeros((fdim,), dtype=INT)
    kCv = np.zeros((fdim,), dtype=DOUBLE)

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
                    for i in range(NE):
                        for j in range(NE):
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = (2/a)*fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[9, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[11, i] = (2/a)*(2/a)*fpp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[12, i] = (2/a)*(2/a)*fpp(i, xi, x1v, x1vr, x2v, x2vr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[9, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[10, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[11, j] = (2/b)*(2/b)*fpp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[12, j] = (2/b)*(2/b)*fpp(j, eta, y1v, y1vr, y2v, y2vr)

                # generalized derivatives at the point, zero when NLgeom == 0
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w_x = 0.
                w_y = 0.
                phix = 0.
                phiy = 0.
                phix_x = 0.
                phix_y = 0.
                phiy_x = 0.
                phiy_y = 0.
                w_xx = 0.
                w_yy = 0.
                w_xy = 0.
                u_xx = 0.
                u_yy = 0.
                u_xy = 0.
                v_xx = 0.
                v_yy = 0.
                v_xy = 0.
                if NLgeom == 1:
                    for j in range(n):
                        gAu = gtab[0, j]
                        gAu_y = gtab[1, j]
                        gAv = gtab[2, j]
                        gAv_y = gtab[3, j]
                        gAw = gtab[4, j]
                        gAw_y = gtab[5, j]
                        gAw_yy = gtab[6, j]
                        gAphix = gtab[7, j]
                        gAphix_y = gtab[8, j]
                        gAphiy = gtab[9, j]
                        gAphiy_y = gtab[10, j]
                        gAu_yy = gtab[11, j]
                        gAv_yy = gtab[12, j]
                        for i in range(m):
                            fAu = ftab[0, i]
                            fAu_x = ftab[1, i]
                            fAv = ftab[2, i]
                            fAv_x = ftab[3, i]
                            fAw = ftab[4, i]
                            fAw_x = ftab[5, i]
                            fAw_xx = ftab[6, i]
                            fAphix = ftab[7, i]
                            fAphix_x = ftab[8, i]
                            fAphiy = ftab[9, i]
                            fAphiy_x = ftab[10, i]
                            fAu_xx = ftab[11, i]
                            fAv_xx = ftab[12, i]
                            col = col0 + DOF*(j*m + i)
                            u_x += fAu_x*gAu*cs[col+0]
                            u_y += fAu*gAu_y*cs[col+0]
                            v_x += fAv_x*gAv*cs[col+1]
                            v_y += fAv*gAv_y*cs[col+1]
                            w_x += fAw_x*gAw*cs[col+2]
                            w_y += fAw*gAw_y*cs[col+2]
                            phix += fAphix*gAphix*cs[col+3]
                            phiy += fAphiy*gAphiy*cs[col+4]
                            phix_x += fAphix_x*gAphix*cs[col+3]
                            phix_y += fAphix*gAphix_y*cs[col+3]
                            phiy_x += fAphiy_x*gAphiy*cs[col+4]
                            phiy_y += fAphiy*gAphiy_y*cs[col+4]
                            w_xx += fAw_xx*gAw*cs[col+2]
                            w_yy += fAw*gAw_yy*cs[col+2]
                            w_xy += fAw_x*gAw_y*cs[col+2]
                            u_xx += fAu_xx*gAu*cs[col+0]
                            u_yy += fAu*gAu_yy*cs[col+0]
                            u_xy += fAu_x*gAu_y*cs[col+0]
                            v_xx += fAv_xx*gAv*cs[col+1]
                            v_yy += fAv*gAv_yy*cs[col+1]
                            v_xy += fAv_x*gAv_y*cs[col+1]

                # [B] = d eps/d d at the point
                for p in range(273):
                    Bm[p] = 0.
                Bm[0] = u_x + 1
                Bm[2] = v_x
                Bm[4] = w_x
                Bm[22] = u_y
                Bm[24] = v_y + 1
                Bm[26] = w_y
                Bm[42] = u_y
                Bm[43] = u_x + 1
                Bm[44] = v_y + 1
                Bm[45] = v_x
                Bm[46] = w_y
                Bm[47] = w_x
                Bm[63] = phix_x
                Bm[65] = phiy_x
                Bm[67] = -phix*phix_x - phiy*phiy_x
                Bm[69] = -phix_x*w_x
                Bm[70] = -phiy_x*w_x
                Bm[71] = -phix*w_x + u_x + 1
                Bm[73] = -phiy*w_x + v_x
                Bm[85] = phix_y
                Bm[87] = phiy_y
                Bm[89] = -phix*phix_y - phiy*phiy_y
                Bm[90] = -phix_y*w_y
                Bm[91] = -phiy_y*w_y
                Bm[93] = -phix*w_y + u_y
                Bm[95] = -phiy*w_y + v_y + 1
                Bm[105] = phix_y
                Bm[106] = phix_x
                Bm[107] = phiy_y
                Bm[108] = phiy_x
                Bm[109] = -phix*phix_y - phiy*phiy_y
                Bm[110] = -phix*phix_x - phiy*phiy_x
                Bm[111] = -phix_x*w_y - phix_y*w_x
                Bm[112] = -phiy_x*w_y - phiy_y*w_x
                Bm[113] = -phix*w_y + u_y
                Bm[114] = -phix*w_x + u_x + 1
                Bm[115] = -phiy*w_y + v_y + 1
                Bm[116] = -phiy*w_x + v_x
                Bm[126] = -c1*phix_x + c1*v_xx*w_y - c1*v_y*w_xx - c1*w_xx
                Bm[127] = c1*v_x*w_xx - c1*v_xx*w_x
                Bm[128] = -c1*phiy_x - c1*u_xx*w_y + c1*u_y*w_xx
                Bm[129] = -c1*u_x*w_xx + c1*u_xx*w_x - c1*w_xx
                Bm[130] = c1*phix*phix_x + c1*phiy*phiy_x + c1*u_xx*v_y + c1*u_xx - c1*u_y*v_xx
                Bm[131] = c1*u_x*v_xx - c1*u_xx*v_x + c1*v_xx
                Bm[132] = c1*phix_x*w_x
                Bm[133] = c1*phiy_x*w_x
                Bm[134] = c1*phix*w_x - c1*u_x - c1
                Bm[136] = c1*phiy*w_x - c1*v_x
                Bm[138] = -c1*u_x*v_y - c1*u_x + c1*u_y*v_x - c1*v_y - c1
                Bm[141] = -c1*v_x*w_y + c1*v_y*w_x + c1*w_x
                Bm[144] = c1*u_x*w_y - c1*u_y*w_x + c1*w_y
                Bm[147] = -c1*v_y*w_yy + c1*v_yy*w_y - c1*w_yy
                Bm[148] = -c1*phix_y + c1*v_x*w_yy - c1*v_yy*w_x
                Bm[149] = c1*u_y*w_yy - c1*u_yy*w_y
                Bm[150] = -c1*phiy_y - c1*u_x*w_yy + c1*u_yy*w_x - c1*w_yy
                Bm[151] = -c1*u_y*v_yy + c1*u_yy*v_y + c1*u_yy
                Bm[152] = c1*phix*phix_y + c1*phiy*phiy_y + c1*u_x*v_yy - c1*u_yy*v_x + c1*v_yy
                Bm[153] = c1*phix_y*w_y
                Bm[154] = c1*phiy_y*w_y
                Bm[156] = c1*phix*w_y - c1*u_y
                Bm[158] = c1*phiy*w_y - c1*v_y - c1
                Bm[160] = -c1*u_x*v_y - c1*u_x + c1*u_y*v_x - c1*v_y - c1
                Bm[163] = -c1*v_x*w_y + c1*v_y*w_x + c1*w_x
                Bm[166] = c1*u_x*w_y - c1*u_y*w_x + c1*w_y
                Bm[168] = -c1*phix_y + 2*c1*v_xy*w_y - 2*c1*v_y*w_xy - 2*c1*w_xy
                Bm[169] = -c1*phix_x + 2*c1*v_x*w_xy - 2*c1*v_xy*w_x
                Bm[170] = -c1*phiy_y - 2*c1*u_xy*w_y + 2*c1*u_y*w_xy
                Bm[171] = -c1*phiy_x - 2*c1*u_x*w_xy + 2*c1*u_xy*w_x - 2*c1*w_xy
                Bm[172] = c1*phix*phix_y + c1*phiy*phiy_y + 2*c1*u_xy*v_y + 2*c1*u_xy - 2*c1*u_y*v_xy
                Bm[173] = c1*phix*phix_x + c1*phiy*phiy_x + 2*c1*u_x*v_xy - 2*c1*u_xy*v_x + 2*c1*v_xy
                Bm[174] = c1*phix_x*w_y + c1*phix_y*w_x
                Bm[175] = c1*phiy_x*w_y + c1*phiy_y*w_x
                Bm[176] = c1*phix*w_y - c1*u_y
                Bm[177] = c1*phix*w_x - c1*u_x - c1
                Bm[178] = c1*phiy*w_y - c1*v_y - c1
                Bm[179] = c1*phiy*w_x - c1*v_x
                Bm[182] = -2*c1*u_x*v_y - 2*c1*u_x + 2*c1*u_y*v_x - 2*c1*v_y - 2*c1
                Bm[185] = -2*c1*v_x*w_y + 2*c1*v_y*w_x + 2*c1*w_x
                Bm[188] = 2*c1*u_x*w_y - 2*c1*u_y*w_x + 2*c1*w_y
                Bm[190] = phix
                Bm[192] = phiy
                Bm[194] = -0.5*(phix*phix) - 0.5*(phiy*phiy) + 1
                Bm[195] = -phix*w_y + u_y
                Bm[196] = -phiy*w_y + v_y + 1
                Bm[210] = phix
                Bm[212] = phiy
                Bm[214] = -0.5*(phix*phix) - 0.5*(phiy*phiy) + 1
                Bm[216] = -phix*w_x + u_x + 1
                Bm[217] = -phiy*w_x + v_x
                Bm[232] = -3*c1*phix
                Bm[234] = -3*c1*phiy
                Bm[236] = 1.5*c1*(phix*phix) + 1.5*c1*(phiy*phiy) - 3*c1
                Bm[237] = 3*c1*phix*w_y - 3*c1*u_y
                Bm[238] = 3*c1*phiy*w_y - 3*c1*v_y - 3*c1
                Bm[252] = -3*c1*phix
                Bm[254] = -3*c1*phiy
                Bm[256] = 1.5*c1*(phix*phix) + 1.5*c1*(phiy*phiy) - 3*c1
                Bm[258] = 3*c1*phix*w_x - 3*c1*u_x - 3*c1
                Bm[259] = 3*c1*phiy*w_x - 3*c1*v_x

                # K = [B]^T [F] [B]
                for r in range(NE):
                    for q in range(21):
                        FB[r*21 + q] = 0.
                        for s in range(NE):
                            FB[r*21 + q] += F[r*NE + s]*Bm[s*21 + q]
                for p in range(21):
                    for q in range(21):
                        K[p*21 + q] = 0.
                        for r in range(NE):
                            K[p*21 + q] += Bm[r*21 + p]*FB[r*21 + q]

                # stresses of the strains and of the linear strains
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x + phiy_x*v_x
                e[4] = -phix*phix_y*w_y + phix_y*u_y - phiy*phiy_y*w_y + phiy_y*v_y + phiy_y
                e[5] = -phix*phix_x*w_y - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - phiy*phiy_x*w_y - phiy*phiy_y*w_x + phiy_x*v_y + phiy_x + phiy_y*v_x
                e[6] = c1*phix*phix_x*w_x - c1*phix_x*u_x - c1*phix_x + c1*phiy*phiy_x*w_x - c1*phiy_x*v_x + c1*u_x*v_xx*w_y - c1*u_x*v_y*w_xx - c1*u_x*w_xx - c1*u_xx*v_x*w_y + c1*u_xx*v_y*w_x + c1*u_xx*w_x + c1*u_y*v_x*w_xx - c1*u_y*v_xx*w_x + c1*v_xx*w_y - c1*v_y*w_xx - c1*w_xx
                e[7] = c1*phix*phix_y*w_y - c1*phix_y*u_y + c1*phiy*phiy_y*w_y - c1*phiy_y*v_y - c1*phiy_y - c1*u_x*v_y*w_yy + c1*u_x*v_yy*w_y - c1*u_x*w_yy + c1*u_y*v_x*w_yy - c1*u_y*v_yy*w_x - c1*u_yy*v_x*w_y + c1*u_yy*v_y*w_x + c1*u_yy*w_x - c1*v_y*w_yy + c1*v_yy*w_y - c1*w_yy
                e[8] = c1*phix*phix_x*w_y + c1*phix*phix_y*w_x - c1*phix_x*u_y - c1*phix_y*u_x - c1*phix_y + c1*phiy*phiy_x*w_y + c1*phiy*phiy_y*w_x - c1*phiy_x*v_y - c1*phiy_x - c1*phiy_y*v_x + 2*c1*u_x*v_xy*w_y - 2*c1*u_x*v_y*w_xy - 2*c1*u_x*w_xy - 2*c1*u_xy*v_x*w_y + 2*c1*u_xy*v_y*w_x + 2*c1*u_xy*w_x + 2*c1*u_y*v_x*w_xy - 2*c1*u_y*v_xy*w_x + 2*c1*v_xy*w_y - 2*c1*v_y*w_xy - 2*c1*w_xy
                e[9] = -0.5*(phix*phix)*w_y + phix*u_y - 0.5*(phiy*phiy)*w_y + phiy*v_y + phiy + w_y
                e[10] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x + w_x
                e[11] = 1.5*c1*(phix*phix)*w_y - 3*c1*phix*u_y + 1.5*c1*(phiy*phiy)*w_y - 3*c1*phiy*v_y - 3*c1*phiy - 3*c1*w_y
                e[12] = 1.5*c1*(phix*phix)*w_x - 3*c1*phix*u_x - 3*c1*phix + 1.5*c1*(phiy*phiy)*w_x - 3*c1*phiy*v_x - 3*c1*w_x
                eL[0] = u_x
                eL[1] = v_y
                eL[2] = u_y + v_x
                eL[3] = phix_x
                eL[4] = phiy_y
                eL[5] = phix_y + phiy_x
                eL[6] = -c1*phix_x - c1*w_xx
                eL[7] = -c1*phiy_y - c1*w_yy
                eL[8] = -c1*phix_y - c1*phiy_x - 2*c1*w_xy
                eL[9] = phiy + w_y
                eL[10] = phix + w_x
                eL[11] = -3*c1*phiy - 3*c1*w_y
                eL[12] = -3*c1*phix - 3*c1*w_x
                for r in range(NE):
                    sg[r] = 0.
                    for s in range(NE):
                        sg[r] += F[r*NE + s]*e[s]
                for r in range(NE):
                    sgL[r] = 0.
                    for s in range(NE):
                        sgL[r] += F[r*NE + s]*eL[s]
                # K += [H(sigma, d)] - [H0(sigma_L)], the second term being
                # assembled by fkG_num, which is homogeneous of degree one in
                # cs as linear buckling requires
                K[0] += -sgL[0] + sg[0]
                K[1] += -sgL[2] + sg[2]
                K[3] += -c1*sg[6]*w_xx - c1*sg[7]*w_yy - 2*c1*sg[8]*w_xy
                K[5] += c1*sg[6]*v_xx + c1*sg[7]*v_yy + 2*c1*sg[8]*v_xy
                K[6] += 3*c1*sgL[12] - 3*c1*sg[12] - sgL[10] + sg[10]
                K[8] += c1*sgL[6] - c1*sg[6] - sgL[3] + sg[3]
                K[9] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[12] += c1*sgL[6] - c1*sg[6]*v_y - c1*sg[6]
                K[13] += c1*sgL[7] - c1*sg[7]*v_y - c1*sg[7]
                K[14] += 2*c1*sgL[8] - 2*c1*sg[8]*v_y - 2*c1*sg[8]
                K[18] += c1*sg[6]*w_y
                K[19] += c1*sg[7]*w_y
                K[20] += 2*c1*sg[8]*w_y
                K[21] += -sgL[2] + sg[2]
                K[22] += -sgL[1] + sg[1]
                K[23] += c1*sg[6]*w_xx + c1*sg[7]*w_yy + 2*c1*sg[8]*w_xy
                K[25] += -c1*sg[6]*v_xx - c1*sg[7]*v_yy - 2*c1*sg[8]*v_xy
                K[27] += 3*c1*sgL[11] - 3*c1*sg[11] - sgL[9] + sg[9]
                K[29] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[30] += c1*sgL[7] - c1*sg[7] - sgL[4] + sg[4]
                K[33] += c1*sg[6]*v_x
                K[34] += c1*sg[7]*v_x
                K[35] += 2*c1*sg[8]*v_x
                K[39] += -c1*sg[6]*w_x
                K[40] += -c1*sg[7]*w_x
                K[41] += -2*c1*sg[8]*w_x
                K[43] += c1*sg[6]*w_xx + c1*sg[7]*w_yy + 2*c1*sg[8]*w_xy
                K[44] += -sgL[0] + sg[0]
                K[45] += -sgL[2] + sg[2]
                K[47] += -c1*sg[6]*u_xx - c1*sg[7]*u_yy - 2*c1*sg[8]*u_xy
                K[49] += 3*c1*sgL[12] - 3*c1*sg[12] - sgL[10] + sg[10]
                K[52] += c1*sgL[6] - c1*sg[6] - sgL[3] + sg[3]
                K[53] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[54] += c1*sg[6]*u_y
                K[55] += c1*sg[7]*u_y
                K[56] += 2*c1*sg[8]*u_y
                K[57] += -c1*sg[6]*w_y
                K[58] += -c1*sg[7]*w_y
                K[59] += -2*c1*sg[8]*w_y
                K[63] += -c1*sg[6]*w_xx - c1*sg[7]*w_yy - 2*c1*sg[8]*w_xy
                K[65] += -sgL[2] + sg[2]
                K[66] += -sgL[1] + sg[1]
                K[67] += c1*sg[6]*u_xx + c1*sg[7]*u_yy + 2*c1*sg[8]*u_xy
                K[70] += 3*c1*sgL[11] - 3*c1*sg[11] - sgL[9] + sg[9]
                K[73] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[74] += c1*sgL[7] - c1*sg[7] - sgL[4] + sg[4]
                K[75] += c1*sgL[6] - c1*sg[6]*u_x - c1*sg[6]
                K[76] += c1*sgL[7] - c1*sg[7]*u_x - c1*sg[7]
                K[77] += 2*c1*sgL[8] - 2*c1*sg[8]*u_x - 2*c1*sg[8]
                K[78] += c1*sg[6]*w_x
                K[79] += c1*sg[7]*w_x
                K[80] += 2*c1*sg[8]*w_x
                K[85] += -c1*sg[6]*v_xx - c1*sg[7]*v_yy - 2*c1*sg[8]*v_xy
                K[87] += c1*sg[6]*u_xx + c1*sg[7]*u_yy + 2*c1*sg[8]*u_xy
                K[88] += -sgL[0] + sg[0]
                K[89] += -sgL[2] + sg[2]
                K[90] += 3*c1*phix*sg[12] + c1*phix_x*sg[6] + c1*phix_y*sg[8] - phix*sg[10] - phix_x*sg[3] - phix_y*sg[5]
                K[91] += 3*c1*phiy*sg[12] + c1*phiy_x*sg[6] + c1*phiy_y*sg[8] - phiy*sg[10] - phiy_x*sg[3] - phiy_y*sg[5]
                K[92] += c1*phix*sg[6] - phix*sg[3]
                K[93] += c1*phix*sg[8] - phix*sg[5]
                K[94] += c1*phiy*sg[6] - phiy*sg[3]
                K[95] += c1*phiy*sg[8] - phiy*sg[5]
                K[99] += -c1*sgL[6] + c1*sg[6]*v_y + c1*sg[6]
                K[100] += -c1*sgL[7] + c1*sg[7]*v_y + c1*sg[7]
                K[101] += -2*c1*sgL[8] + 2*c1*sg[8]*v_y + 2*c1*sg[8]
                K[102] += -c1*sg[6]*u_y
                K[103] += -c1*sg[7]*u_y
                K[104] += -2*c1*sg[8]*u_y
                K[105] += c1*sg[6]*v_xx + c1*sg[7]*v_yy + 2*c1*sg[8]*v_xy
                K[107] += -c1*sg[6]*u_xx - c1*sg[7]*u_yy - 2*c1*sg[8]*u_xy
                K[109] += -sgL[2] + sg[2]
                K[110] += -sgL[1] + sg[1]
                K[111] += 3*c1*phix*sg[11] + c1*phix_x*sg[8] + c1*phix_y*sg[7] - phix*sg[9] - phix_x*sg[5] - phix_y*sg[4]
                K[112] += 3*c1*phiy*sg[11] + c1*phiy_x*sg[8] + c1*phiy_y*sg[7] - phiy*sg[9] - phiy_x*sg[5] - phiy_y*sg[4]
                K[113] += c1*phix*sg[8] - phix*sg[5]
                K[114] += c1*phix*sg[7] - phix*sg[4]
                K[115] += c1*phiy*sg[8] - phiy*sg[5]
                K[116] += c1*phiy*sg[7] - phiy*sg[4]
                K[120] += -c1*sg[6]*v_x
                K[121] += -c1*sg[7]*v_x
                K[122] += -2*c1*sg[8]*v_x
                K[123] += -c1*sgL[6] + c1*sg[6]*u_x + c1*sg[6]
                K[124] += -c1*sgL[7] + c1*sg[7]*u_x + c1*sg[7]
                K[125] += -2*c1*sgL[8] + 2*c1*sg[8]*u_x + 2*c1*sg[8]
                K[126] += 3*c1*sgL[12] - 3*c1*sg[12] - sgL[10] + sg[10]
                K[127] += 3*c1*sgL[11] - 3*c1*sg[11] - sgL[9] + sg[9]
                K[130] += 3*c1*phix*sg[12] + c1*phix_x*sg[6] + c1*phix_y*sg[8] - phix*sg[10] - phix_x*sg[3] - phix_y*sg[5]
                K[131] += 3*c1*phix*sg[11] + c1*phix_x*sg[8] + c1*phix_y*sg[7] - phix*sg[9] - phix_x*sg[5] - phix_y*sg[4]
                K[132] += 3*c1*sg[11]*w_y + 3*c1*sg[12]*w_x - sg[10]*w_x - sg[9]*w_y
                K[134] += c1*sg[6]*w_x + c1*sg[8]*w_y - sg[3]*w_x - sg[5]*w_y
                K[135] += c1*sg[7]*w_y + c1*sg[8]*w_x - sg[4]*w_y - sg[5]*w_x
                K[149] += 3*c1*sgL[12] - 3*c1*sg[12] - sgL[10] + sg[10]
                K[150] += 3*c1*sgL[11] - 3*c1*sg[11] - sgL[9] + sg[9]
                K[151] += 3*c1*phiy*sg[12] + c1*phiy_x*sg[6] + c1*phiy_y*sg[8] - phiy*sg[10] - phiy_x*sg[3] - phiy_y*sg[5]
                K[152] += 3*c1*phiy*sg[11] + c1*phiy_x*sg[8] + c1*phiy_y*sg[7] - phiy*sg[9] - phiy_x*sg[5] - phiy_y*sg[4]
                K[154] += 3*c1*sg[11]*w_y + 3*c1*sg[12]*w_x - sg[10]*w_x - sg[9]*w_y
                K[157] += c1*sg[6]*w_x + c1*sg[8]*w_y - sg[3]*w_x - sg[5]*w_y
                K[158] += c1*sg[7]*w_y + c1*sg[8]*w_x - sg[4]*w_y - sg[5]*w_x
                K[168] += c1*sgL[6] - c1*sg[6] - sgL[3] + sg[3]
                K[169] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[172] += c1*phix*sg[6] - phix*sg[3]
                K[173] += c1*phix*sg[8] - phix*sg[5]
                K[174] += c1*sg[6]*w_x + c1*sg[8]*w_y - sg[3]*w_x - sg[5]*w_y
                K[189] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[190] += c1*sgL[7] - c1*sg[7] - sgL[4] + sg[4]
                K[193] += c1*phix*sg[8] - phix*sg[5]
                K[194] += c1*phix*sg[7] - phix*sg[4]
                K[195] += c1*sg[7]*w_y + c1*sg[8]*w_x - sg[4]*w_y - sg[5]*w_x
                K[212] += c1*sgL[6] - c1*sg[6] - sgL[3] + sg[3]
                K[213] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[214] += c1*phiy*sg[6] - phiy*sg[3]
                K[215] += c1*phiy*sg[8] - phiy*sg[5]
                K[217] += c1*sg[6]*w_x + c1*sg[8]*w_y - sg[3]*w_x - sg[5]*w_y
                K[233] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[234] += c1*sgL[7] - c1*sg[7] - sgL[4] + sg[4]
                K[235] += c1*phiy*sg[8] - phiy*sg[5]
                K[236] += c1*phiy*sg[7] - phiy*sg[4]
                K[238] += c1*sg[7]*w_y + c1*sg[8]*w_x - sg[4]*w_y - sg[5]*w_x
                K[252] += c1*sgL[6] - c1*sg[6]*v_y - c1*sg[6]
                K[253] += c1*sg[6]*v_x
                K[254] += c1*sg[6]*u_y
                K[255] += c1*sgL[6] - c1*sg[6]*u_x - c1*sg[6]
                K[273] += c1*sgL[7] - c1*sg[7]*v_y - c1*sg[7]
                K[274] += c1*sg[7]*v_x
                K[275] += c1*sg[7]*u_y
                K[276] += c1*sgL[7] - c1*sg[7]*u_x - c1*sg[7]
                K[294] += 2*c1*sgL[8] - 2*c1*sg[8]*v_y - 2*c1*sg[8]
                K[295] += 2*c1*sg[8]*v_x
                K[296] += 2*c1*sg[8]*u_y
                K[297] += 2*c1*sgL[8] - 2*c1*sg[8]*u_x - 2*c1*sg[8]
                K[317] += -c1*sg[6]*w_y
                K[318] += c1*sg[6]*w_x
                K[319] += -c1*sgL[6] + c1*sg[6]*v_y + c1*sg[6]
                K[320] += -c1*sg[6]*v_x
                K[338] += -c1*sg[7]*w_y
                K[339] += c1*sg[7]*w_x
                K[340] += -c1*sgL[7] + c1*sg[7]*v_y + c1*sg[7]
                K[341] += -c1*sg[7]*v_x
                K[359] += -2*c1*sg[8]*w_y
                K[360] += 2*c1*sg[8]*w_x
                K[361] += -2*c1*sgL[8] + 2*c1*sg[8]*v_y + 2*c1*sg[8]
                K[362] += -2*c1*sg[8]*v_x
                K[378] += c1*sg[6]*w_y
                K[379] += -c1*sg[6]*w_x
                K[382] += -c1*sg[6]*u_y
                K[383] += -c1*sgL[6] + c1*sg[6]*u_x + c1*sg[6]
                K[399] += c1*sg[7]*w_y
                K[400] += -c1*sg[7]*w_x
                K[403] += -c1*sg[7]*u_y
                K[404] += -c1*sgL[7] + c1*sg[7]*u_x + c1*sg[7]
                K[420] += 2*c1*sg[8]*w_y
                K[421] += -2*c1*sg[8]*w_x
                K[424] += -2*c1*sg[8]*u_y
                K[425] += -2*c1*sgL[8] + 2*c1*sg[8]*u_x + 2*c1*sg[8]

                c = -1
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAw_yy = gtab[6, j]
                    gAphix = gtab[7, j]
                    gAphix_y = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_y = gtab[10, j]
                    gAu_yy = gtab[11, j]
                    gAv_yy = gtab[12, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_x = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_x = ftab[10, i]
                        fAu_xx = ftab[11, i]
                        fAv_xx = ftab[12, i]
                        row = row0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAphix*gAphix
                        pA[7] = fAphiy*gAphiy
                        pA[8] = fAphix_x*gAphix
                        pA[9] = fAphix*gAphix_y
                        pA[10] = fAphiy_x*gAphiy
                        pA[11] = fAphiy*gAphiy_y
                        pA[12] = fAw_xx*gAw
                        pA[13] = fAw*gAw_yy
                        pA[14] = fAw_x*gAw_y
                        pA[15] = fAu_xx*gAu
                        pA[16] = fAu*gAu_yy
                        pA[17] = fAu_x*gAu_y
                        pA[18] = fAv_xx*gAv
                        pA[19] = fAv*gAv_yy
                        pA[20] = fAv_x*gAv_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[21] + pA[15]*K[315] + pA[16]*K[336] + pA[17]*K[357]
                        TA[1] = pA[0]*K[1] + pA[1]*K[22] + pA[15]*K[316] + pA[16]*K[337] + pA[17]*K[358]
                        TA[2] = pA[0]*K[2] + pA[1]*K[23] + pA[15]*K[317] + pA[16]*K[338] + pA[17]*K[359]
                        TA[3] = pA[0]*K[3] + pA[1]*K[24] + pA[15]*K[318] + pA[16]*K[339] + pA[17]*K[360]
                        TA[4] = pA[0]*K[4] + pA[1]*K[25] + pA[15]*K[319] + pA[16]*K[340] + pA[17]*K[361]
                        TA[5] = pA[0]*K[5] + pA[1]*K[26] + pA[15]*K[320] + pA[16]*K[341] + pA[17]*K[362]
                        TA[6] = pA[0]*K[6] + pA[1]*K[27] + pA[15]*K[321] + pA[16]*K[342] + pA[17]*K[363]
                        TA[7] = pA[0]*K[7] + pA[1]*K[28] + pA[15]*K[322] + pA[16]*K[343] + pA[17]*K[364]
                        TA[8] = pA[0]*K[8] + pA[1]*K[29] + pA[15]*K[323] + pA[16]*K[344] + pA[17]*K[365]
                        TA[9] = pA[0]*K[9] + pA[1]*K[30] + pA[15]*K[324] + pA[16]*K[345] + pA[17]*K[366]
                        TA[10] = pA[0]*K[10] + pA[1]*K[31] + pA[15]*K[325] + pA[16]*K[346] + pA[17]*K[367]
                        TA[11] = pA[0]*K[11] + pA[1]*K[32] + pA[15]*K[326] + pA[16]*K[347] + pA[17]*K[368]
                        TA[12] = pA[0]*K[12] + pA[1]*K[33] + pA[15]*K[327] + pA[16]*K[348] + pA[17]*K[369]
                        TA[13] = pA[0]*K[13] + pA[1]*K[34] + pA[15]*K[328] + pA[16]*K[349] + pA[17]*K[370]
                        TA[14] = pA[0]*K[14] + pA[1]*K[35] + pA[15]*K[329] + pA[16]*K[350] + pA[17]*K[371]
                        TA[15] = pA[0]*K[15] + pA[1]*K[36] + pA[15]*K[330] + pA[16]*K[351] + pA[17]*K[372]
                        TA[16] = pA[0]*K[16] + pA[1]*K[37] + pA[15]*K[331] + pA[16]*K[352] + pA[17]*K[373]
                        TA[17] = pA[0]*K[17] + pA[1]*K[38] + pA[15]*K[332] + pA[16]*K[353] + pA[17]*K[374]
                        TA[18] = pA[0]*K[18] + pA[1]*K[39] + pA[15]*K[333] + pA[16]*K[354] + pA[17]*K[375]
                        TA[19] = pA[0]*K[19] + pA[1]*K[40] + pA[15]*K[334] + pA[16]*K[355] + pA[17]*K[376]
                        TA[20] = pA[0]*K[20] + pA[1]*K[41] + pA[15]*K[335] + pA[16]*K[356] + pA[17]*K[377]
                        TA[21] = pA[2]*K[42] + pA[3]*K[63] + pA[18]*K[378] + pA[19]*K[399] + pA[20]*K[420]
                        TA[22] = pA[2]*K[43] + pA[3]*K[64] + pA[18]*K[379] + pA[19]*K[400] + pA[20]*K[421]
                        TA[23] = pA[2]*K[44] + pA[3]*K[65] + pA[18]*K[380] + pA[19]*K[401] + pA[20]*K[422]
                        TA[24] = pA[2]*K[45] + pA[3]*K[66] + pA[18]*K[381] + pA[19]*K[402] + pA[20]*K[423]
                        TA[25] = pA[2]*K[46] + pA[3]*K[67] + pA[18]*K[382] + pA[19]*K[403] + pA[20]*K[424]
                        TA[26] = pA[2]*K[47] + pA[3]*K[68] + pA[18]*K[383] + pA[19]*K[404] + pA[20]*K[425]
                        TA[27] = pA[2]*K[48] + pA[3]*K[69] + pA[18]*K[384] + pA[19]*K[405] + pA[20]*K[426]
                        TA[28] = pA[2]*K[49] + pA[3]*K[70] + pA[18]*K[385] + pA[19]*K[406] + pA[20]*K[427]
                        TA[29] = pA[2]*K[50] + pA[3]*K[71] + pA[18]*K[386] + pA[19]*K[407] + pA[20]*K[428]
                        TA[30] = pA[2]*K[51] + pA[3]*K[72] + pA[18]*K[387] + pA[19]*K[408] + pA[20]*K[429]
                        TA[31] = pA[2]*K[52] + pA[3]*K[73] + pA[18]*K[388] + pA[19]*K[409] + pA[20]*K[430]
                        TA[32] = pA[2]*K[53] + pA[3]*K[74] + pA[18]*K[389] + pA[19]*K[410] + pA[20]*K[431]
                        TA[33] = pA[2]*K[54] + pA[3]*K[75] + pA[18]*K[390] + pA[19]*K[411] + pA[20]*K[432]
                        TA[34] = pA[2]*K[55] + pA[3]*K[76] + pA[18]*K[391] + pA[19]*K[412] + pA[20]*K[433]
                        TA[35] = pA[2]*K[56] + pA[3]*K[77] + pA[18]*K[392] + pA[19]*K[413] + pA[20]*K[434]
                        TA[36] = pA[2]*K[57] + pA[3]*K[78] + pA[18]*K[393] + pA[19]*K[414] + pA[20]*K[435]
                        TA[37] = pA[2]*K[58] + pA[3]*K[79] + pA[18]*K[394] + pA[19]*K[415] + pA[20]*K[436]
                        TA[38] = pA[2]*K[59] + pA[3]*K[80] + pA[18]*K[395] + pA[19]*K[416] + pA[20]*K[437]
                        TA[39] = pA[2]*K[60] + pA[3]*K[81] + pA[18]*K[396] + pA[19]*K[417] + pA[20]*K[438]
                        TA[40] = pA[2]*K[61] + pA[3]*K[82] + pA[18]*K[397] + pA[19]*K[418] + pA[20]*K[439]
                        TA[41] = pA[2]*K[62] + pA[3]*K[83] + pA[18]*K[398] + pA[19]*K[419] + pA[20]*K[440]
                        TA[42] = pA[4]*K[84] + pA[5]*K[105] + pA[12]*K[252] + pA[13]*K[273] + pA[14]*K[294]
                        TA[43] = pA[4]*K[85] + pA[5]*K[106] + pA[12]*K[253] + pA[13]*K[274] + pA[14]*K[295]
                        TA[44] = pA[4]*K[86] + pA[5]*K[107] + pA[12]*K[254] + pA[13]*K[275] + pA[14]*K[296]
                        TA[45] = pA[4]*K[87] + pA[5]*K[108] + pA[12]*K[255] + pA[13]*K[276] + pA[14]*K[297]
                        TA[46] = pA[4]*K[88] + pA[5]*K[109] + pA[12]*K[256] + pA[13]*K[277] + pA[14]*K[298]
                        TA[47] = pA[4]*K[89] + pA[5]*K[110] + pA[12]*K[257] + pA[13]*K[278] + pA[14]*K[299]
                        TA[48] = pA[4]*K[90] + pA[5]*K[111] + pA[12]*K[258] + pA[13]*K[279] + pA[14]*K[300]
                        TA[49] = pA[4]*K[91] + pA[5]*K[112] + pA[12]*K[259] + pA[13]*K[280] + pA[14]*K[301]
                        TA[50] = pA[4]*K[92] + pA[5]*K[113] + pA[12]*K[260] + pA[13]*K[281] + pA[14]*K[302]
                        TA[51] = pA[4]*K[93] + pA[5]*K[114] + pA[12]*K[261] + pA[13]*K[282] + pA[14]*K[303]
                        TA[52] = pA[4]*K[94] + pA[5]*K[115] + pA[12]*K[262] + pA[13]*K[283] + pA[14]*K[304]
                        TA[53] = pA[4]*K[95] + pA[5]*K[116] + pA[12]*K[263] + pA[13]*K[284] + pA[14]*K[305]
                        TA[54] = pA[4]*K[96] + pA[5]*K[117] + pA[12]*K[264] + pA[13]*K[285] + pA[14]*K[306]
                        TA[55] = pA[4]*K[97] + pA[5]*K[118] + pA[12]*K[265] + pA[13]*K[286] + pA[14]*K[307]
                        TA[56] = pA[4]*K[98] + pA[5]*K[119] + pA[12]*K[266] + pA[13]*K[287] + pA[14]*K[308]
                        TA[57] = pA[4]*K[99] + pA[5]*K[120] + pA[12]*K[267] + pA[13]*K[288] + pA[14]*K[309]
                        TA[58] = pA[4]*K[100] + pA[5]*K[121] + pA[12]*K[268] + pA[13]*K[289] + pA[14]*K[310]
                        TA[59] = pA[4]*K[101] + pA[5]*K[122] + pA[12]*K[269] + pA[13]*K[290] + pA[14]*K[311]
                        TA[60] = pA[4]*K[102] + pA[5]*K[123] + pA[12]*K[270] + pA[13]*K[291] + pA[14]*K[312]
                        TA[61] = pA[4]*K[103] + pA[5]*K[124] + pA[12]*K[271] + pA[13]*K[292] + pA[14]*K[313]
                        TA[62] = pA[4]*K[104] + pA[5]*K[125] + pA[12]*K[272] + pA[13]*K[293] + pA[14]*K[314]
                        TA[63] = pA[6]*K[126] + pA[8]*K[168] + pA[9]*K[189]
                        TA[64] = pA[6]*K[127] + pA[8]*K[169] + pA[9]*K[190]
                        TA[65] = pA[6]*K[128] + pA[8]*K[170] + pA[9]*K[191]
                        TA[66] = pA[6]*K[129] + pA[8]*K[171] + pA[9]*K[192]
                        TA[67] = pA[6]*K[130] + pA[8]*K[172] + pA[9]*K[193]
                        TA[68] = pA[6]*K[131] + pA[8]*K[173] + pA[9]*K[194]
                        TA[69] = pA[6]*K[132] + pA[8]*K[174] + pA[9]*K[195]
                        TA[70] = pA[6]*K[133] + pA[8]*K[175] + pA[9]*K[196]
                        TA[71] = pA[6]*K[134] + pA[8]*K[176] + pA[9]*K[197]
                        TA[72] = pA[6]*K[135] + pA[8]*K[177] + pA[9]*K[198]
                        TA[73] = pA[6]*K[136] + pA[8]*K[178] + pA[9]*K[199]
                        TA[74] = pA[6]*K[137] + pA[8]*K[179] + pA[9]*K[200]
                        TA[75] = pA[6]*K[138] + pA[8]*K[180] + pA[9]*K[201]
                        TA[76] = pA[6]*K[139] + pA[8]*K[181] + pA[9]*K[202]
                        TA[77] = pA[6]*K[140] + pA[8]*K[182] + pA[9]*K[203]
                        TA[78] = pA[6]*K[141] + pA[8]*K[183] + pA[9]*K[204]
                        TA[79] = pA[6]*K[142] + pA[8]*K[184] + pA[9]*K[205]
                        TA[80] = pA[6]*K[143] + pA[8]*K[185] + pA[9]*K[206]
                        TA[81] = pA[6]*K[144] + pA[8]*K[186] + pA[9]*K[207]
                        TA[82] = pA[6]*K[145] + pA[8]*K[187] + pA[9]*K[208]
                        TA[83] = pA[6]*K[146] + pA[8]*K[188] + pA[9]*K[209]
                        TA[84] = pA[7]*K[147] + pA[10]*K[210] + pA[11]*K[231]
                        TA[85] = pA[7]*K[148] + pA[10]*K[211] + pA[11]*K[232]
                        TA[86] = pA[7]*K[149] + pA[10]*K[212] + pA[11]*K[233]
                        TA[87] = pA[7]*K[150] + pA[10]*K[213] + pA[11]*K[234]
                        TA[88] = pA[7]*K[151] + pA[10]*K[214] + pA[11]*K[235]
                        TA[89] = pA[7]*K[152] + pA[10]*K[215] + pA[11]*K[236]
                        TA[90] = pA[7]*K[153] + pA[10]*K[216] + pA[11]*K[237]
                        TA[91] = pA[7]*K[154] + pA[10]*K[217] + pA[11]*K[238]
                        TA[92] = pA[7]*K[155] + pA[10]*K[218] + pA[11]*K[239]
                        TA[93] = pA[7]*K[156] + pA[10]*K[219] + pA[11]*K[240]
                        TA[94] = pA[7]*K[157] + pA[10]*K[220] + pA[11]*K[241]
                        TA[95] = pA[7]*K[158] + pA[10]*K[221] + pA[11]*K[242]
                        TA[96] = pA[7]*K[159] + pA[10]*K[222] + pA[11]*K[243]
                        TA[97] = pA[7]*K[160] + pA[10]*K[223] + pA[11]*K[244]
                        TA[98] = pA[7]*K[161] + pA[10]*K[224] + pA[11]*K[245]
                        TA[99] = pA[7]*K[162] + pA[10]*K[225] + pA[11]*K[246]
                        TA[100] = pA[7]*K[163] + pA[10]*K[226] + pA[11]*K[247]
                        TA[101] = pA[7]*K[164] + pA[10]*K[227] + pA[11]*K[248]
                        TA[102] = pA[7]*K[165] + pA[10]*K[228] + pA[11]*K[249]
                        TA[103] = pA[7]*K[166] + pA[10]*K[229] + pA[11]*K[250]
                        TA[104] = pA[7]*K[167] + pA[10]*K[230] + pA[11]*K[251]
                        for l in range(n):
                            gBu = gtab[0, l]
                            gBu_y = gtab[1, l]
                            gBv = gtab[2, l]
                            gBv_y = gtab[3, l]
                            gBw = gtab[4, l]
                            gBw_y = gtab[5, l]
                            gBw_yy = gtab[6, l]
                            gBphix = gtab[7, l]
                            gBphix_y = gtab[8, l]
                            gBphiy = gtab[9, l]
                            gBphiy_y = gtab[10, l]
                            gBu_yy = gtab[11, l]
                            gBv_yy = gtab[12, l]
                            for k in range(m):
                                col = col0 + DOF*(l*m + k)
                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue
                                fBu = ftab[0, k]
                                fBu_x = ftab[1, k]
                                fBv = ftab[2, k]
                                fBv_x = ftab[3, k]
                                fBw = ftab[4, k]
                                fBw_x = ftab[5, k]
                                fBw_xx = ftab[6, k]
                                fBphix = ftab[7, k]
                                fBphix_x = ftab[8, k]
                                fBphiy = ftab[9, k]
                                fBphiy_x = ftab[10, k]
                                fBu_xx = ftab[11, k]
                                fBv_xx = ftab[12, k]
                                pB[0] = fBu_x*gBu
                                pB[1] = fBu*gBu_y
                                pB[2] = fBv_x*gBv
                                pB[3] = fBv*gBv_y
                                pB[4] = fBw_x*gBw
                                pB[5] = fBw*gBw_y
                                pB[6] = fBphix*gBphix
                                pB[7] = fBphiy*gBphiy
                                pB[8] = fBphix_x*gBphix
                                pB[9] = fBphix*gBphix_y
                                pB[10] = fBphiy_x*gBphiy
                                pB[11] = fBphiy*gBphiy_y
                                pB[12] = fBw_xx*gBw
                                pB[13] = fBw*gBw_yy
                                pB[14] = fBw_x*gBw_y
                                pB[15] = fBu_xx*gBu
                                pB[16] = fBu*gBu_yy
                                pB[17] = fBu_x*gBu_y
                                pB[18] = fBv_xx*gBv
                                pB[19] = fBv*gBv_yy
                                pB[20] = fBv_x*gBv_y
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1] + TA[15]*pB[15] + TA[16]*pB[16] + TA[17]*pB[17])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[2]*pB[2] + TA[3]*pB[3] + TA[18]*pB[18] + TA[19]*pB[19] + TA[20]*pB[20])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[4]*pB[4] + TA[5]*pB[5] + TA[12]*pB[12] + TA[13]*pB[13] + TA[14]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[6]*pB[6] + TA[8]*pB[8] + TA[9]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[7]*pB[7] + TA[10]*pB[10] + TA[11]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[21]*pB[0] + TA[22]*pB[1] + TA[36]*pB[15] + TA[37]*pB[16] + TA[38]*pB[17])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[23]*pB[2] + TA[24]*pB[3] + TA[39]*pB[18] + TA[40]*pB[19] + TA[41]*pB[20])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[25]*pB[4] + TA[26]*pB[5] + TA[33]*pB[12] + TA[34]*pB[13] + TA[35]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[27]*pB[6] + TA[29]*pB[8] + TA[30]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[28]*pB[7] + TA[31]*pB[10] + TA[32]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[42]*pB[0] + TA[43]*pB[1] + TA[57]*pB[15] + TA[58]*pB[16] + TA[59]*pB[17])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[44]*pB[2] + TA[45]*pB[3] + TA[60]*pB[18] + TA[61]*pB[19] + TA[62]*pB[20])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[46]*pB[4] + TA[47]*pB[5] + TA[54]*pB[12] + TA[55]*pB[13] + TA[56]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[48]*pB[6] + TA[50]*pB[8] + TA[51]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[49]*pB[7] + TA[52]*pB[10] + TA[53]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[63]*pB[0] + TA[64]*pB[1] + TA[78]*pB[15] + TA[79]*pB[16] + TA[80]*pB[17])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[65]*pB[2] + TA[66]*pB[3] + TA[81]*pB[18] + TA[82]*pB[19] + TA[83]*pB[20])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[67]*pB[4] + TA[68]*pB[5] + TA[75]*pB[12] + TA[76]*pB[13] + TA[77]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[69]*pB[6] + TA[71]*pB[8] + TA[72]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[70]*pB[7] + TA[73]*pB[10] + TA[74]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[84]*pB[0] + TA[85]*pB[1] + TA[99]*pB[15] + TA[100]*pB[16] + TA[101]*pB[17])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[86]*pB[2] + TA[87]*pB[3] + TA[102]*pB[18] + TA[103]*pB[19] + TA[104]*pB[20])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[88]*pB[4] + TA[89]*pB[5] + TA[96]*pB[12] + TA[97]*pB[13] + TA[98]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[90]*pB[6] + TA[92]*pB[8] + TA[93]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[91]*pB[7] + TA[94]*pB[10] + TA[95]*pB[11])

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC


def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    r"""Geometric stiffness matrix of the linear stress state

    Integrates numerically the geometric stiffness matrix of the stress
    resultants of the linear strains at ``cs``, superposed to the constant
    membrane state ``Nxx0``, ``Nyy0``, ``Nxy0``: the membrane stresses give
    one block per local gradient of ``u``, ``v`` and ``w``, as for the
    Green-Lagrange membrane strains, and the moments and transverse shear
    forces couple the in-plane gradients with the rotations. The matrix is
    homogeneous of degree one in ``cs``, as the linear buckling eigenvalue
    problem requires.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements (the rotations in radians).
    Finput : array-like
        Constitutive matrix of the generalized strains of the model, shape
        ``(13, 13)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 13, 13)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.
    Nxx0, Nyy0, Nxy0 : float, optional
        Constant membrane stress resultants, force per unit length, default
        ``0``.

    Returns
    -------
    kG : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape or ``shell`` is not a
        :class:`.Shell`.

    Notes
    -----
    With the generalized derivatives at the integration point `\boldsymbol{d}
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the rotations and their gradients), the strains
    `\boldsymbol{\varepsilon}(\boldsymbol{d})`, cubic, `\boldsymbol{B} =
    \partial \boldsymbol{\varepsilon} / \partial \boldsymbol{d}`, the
    stresses `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}`, `\boldsymbol{H}(\boldsymbol{\sigma},
    \boldsymbol{d}) = \sum_r \sigma_r \, \partial^2 \varepsilon_r /
    \partial \boldsymbol{d}^2`, linear in `\boldsymbol{d}`, and
    `\boldsymbol{H}_0(\boldsymbol{\sigma}) =
    \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{0})`:

    .. math::

        \boldsymbol{K}_G = \int \boldsymbol{\Phi}^\top
        \boldsymbol{H}_0(\boldsymbol{\sigma}) \boldsymbol{\Phi} \, dA

    with `\boldsymbol{\sigma} = \{N_{xx0}, N_{yy0}, N_{xy0}, 0, \ldots\}^\top +
    \boldsymbol{\sigma}_L`, `\boldsymbol{\sigma}_L` the stresses of the
    linear strains at ``cs``. The rest of the tangent stiffness matrix is
    assembled by :func:`fkC_num`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \le x \le x_2`, `y_1 \le y \le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

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

    cdef int i, j, k, l, c, p, q, r, s, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy
    cdef double phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy
    cdef double e[13]
    cdef double sg[13]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAw_xx
    cdef double fAphix, fAphix_x, fAphiy, fAphiy_x, fAu_xx, fAv_xx, gAu
    cdef double gAu_y, gAv, gAv_y, gAw, gAw_y, gAw_yy, gAphix
    cdef double gAphix_y, gAphiy, gAphiy_y, gAu_yy, gAv_yy, fBu, fBu_x
    cdef double fBv, fBv_x, fBw, fBw_x, fBw_xx, fBphix, fBphix_x
    cdef double fBphiy, fBphiy_x, fBu_xx, fBv_xx, gBu, gBu_y, gBv
    cdef double gBv_y, gBw, gBw_y, gBw_yy, gBphix, gBphix_y, gBphiy
    cdef double gBphiy_y, gBu_yy, gBv_yy
    cdef double K[441]
    cdef double TA[105]
    cdef double pA[21]
    cdef double pB[21]

    cdef long [::1] kGr, kGc
    cdef double [::1] kGv

    # F as 4-D matrix, must be [nx, ny, NE, NE], when there is one F[NE, NE]
    # for each of the nx * ny integration points
    cdef double F[13 * 13]
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

    ftab = np.zeros((13, m), dtype=DOUBLE)
    gtab = np.zeros((13, n), dtype=DOUBLE)

    fdim = 11*m*m*n*n
    kGr = np.zeros((fdim,), dtype=INT)
    kGc = np.zeros((fdim,), dtype=INT)
    kGv = np.zeros((fdim,), dtype=DOUBLE)

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
                    for i in range(NE):
                        for j in range(NE):
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = (2/a)*fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[9, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[11, i] = (2/a)*(2/a)*fpp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[12, i] = (2/a)*(2/a)*fpp(i, xi, x1v, x1vr, x2v, x2vr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[9, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[10, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[11, j] = (2/b)*(2/b)*fpp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[12, j] = (2/b)*(2/b)*fpp(j, eta, y1v, y1vr, y2v, y2vr)

                # derivatives of the linear strains at the point
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w_x = 0.
                w_y = 0.
                phix = 0.
                phiy = 0.
                phix_x = 0.
                phix_y = 0.
                phiy_x = 0.
                phiy_y = 0.
                w_xx = 0.
                w_yy = 0.
                w_xy = 0.
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAw_yy = gtab[6, j]
                    gAphix = gtab[7, j]
                    gAphix_y = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_y = gtab[10, j]
                    gAu_yy = gtab[11, j]
                    gAv_yy = gtab[12, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_x = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_x = ftab[10, i]
                        fAu_xx = ftab[11, i]
                        fAv_xx = ftab[12, i]
                        col = col0 + DOF*(j*m + i)
                        u_x += fAu_x*gAu*cs[col+0]
                        u_y += fAu*gAu_y*cs[col+0]
                        v_x += fAv_x*gAv*cs[col+1]
                        v_y += fAv*gAv_y*cs[col+1]
                        w_x += fAw_x*gAw*cs[col+2]
                        w_y += fAw*gAw_y*cs[col+2]
                        phix += fAphix*gAphix*cs[col+3]
                        phiy += fAphiy*gAphiy*cs[col+4]
                        phix_x += fAphix_x*gAphix*cs[col+3]
                        phix_y += fAphix*gAphix_y*cs[col+3]
                        phiy_x += fAphiy_x*gAphiy*cs[col+4]
                        phiy_y += fAphiy*gAphiy_y*cs[col+4]
                        w_xx += fAw_xx*gAw*cs[col+2]
                        w_yy += fAw*gAw_yy*cs[col+2]
                        w_xy += fAw_x*gAw_y*cs[col+2]

                # linear strains and stresses
                e[0] = u_x
                e[1] = v_y
                e[2] = u_y + v_x
                e[3] = phix_x
                e[4] = phiy_y
                e[5] = phix_y + phiy_x
                e[6] = -c1*phix_x - c1*w_xx
                e[7] = -c1*phiy_y - c1*w_yy
                e[8] = -c1*phix_y - c1*phiy_x - 2*c1*w_xy
                e[9] = phiy + w_y
                e[10] = phix + w_x
                e[11] = -3*c1*phiy - 3*c1*w_y
                e[12] = -3*c1*phix - 3*c1*w_x
                for r in range(NE):
                    sg[r] = 0.
                    for s in range(NE):
                        sg[r] += F[r*NE + s]*e[s]
                sg[0] += Nxx0
                sg[1] += Nyy0
                sg[2] += Nxy0

                # K = [H0(sigma)]
                for p in range(441):
                    K[p] = 0.
                K[0] = sg[0]
                K[1] = sg[2]
                K[6] = -3*c1*sg[12] + sg[10]
                K[8] = -c1*sg[6] + sg[3]
                K[9] = -c1*sg[8] + sg[5]
                K[12] = -c1*sg[6]
                K[13] = -c1*sg[7]
                K[14] = -2*c1*sg[8]
                K[21] = sg[2]
                K[22] = sg[1]
                K[27] = -3*c1*sg[11] + sg[9]
                K[29] = -c1*sg[8] + sg[5]
                K[30] = -c1*sg[7] + sg[4]
                K[44] = sg[0]
                K[45] = sg[2]
                K[49] = -3*c1*sg[12] + sg[10]
                K[52] = -c1*sg[6] + sg[3]
                K[53] = -c1*sg[8] + sg[5]
                K[65] = sg[2]
                K[66] = sg[1]
                K[70] = -3*c1*sg[11] + sg[9]
                K[73] = -c1*sg[8] + sg[5]
                K[74] = -c1*sg[7] + sg[4]
                K[75] = -c1*sg[6]
                K[76] = -c1*sg[7]
                K[77] = -2*c1*sg[8]
                K[88] = sg[0]
                K[89] = sg[2]
                K[99] = c1*sg[6]
                K[100] = c1*sg[7]
                K[101] = 2*c1*sg[8]
                K[109] = sg[2]
                K[110] = sg[1]
                K[123] = c1*sg[6]
                K[124] = c1*sg[7]
                K[125] = 2*c1*sg[8]
                K[126] = -3*c1*sg[12] + sg[10]
                K[127] = -3*c1*sg[11] + sg[9]
                K[149] = -3*c1*sg[12] + sg[10]
                K[150] = -3*c1*sg[11] + sg[9]
                K[168] = -c1*sg[6] + sg[3]
                K[169] = -c1*sg[8] + sg[5]
                K[189] = -c1*sg[8] + sg[5]
                K[190] = -c1*sg[7] + sg[4]
                K[212] = -c1*sg[6] + sg[3]
                K[213] = -c1*sg[8] + sg[5]
                K[233] = -c1*sg[8] + sg[5]
                K[234] = -c1*sg[7] + sg[4]
                K[252] = -c1*sg[6]
                K[255] = -c1*sg[6]
                K[273] = -c1*sg[7]
                K[276] = -c1*sg[7]
                K[294] = -2*c1*sg[8]
                K[297] = -2*c1*sg[8]
                K[319] = c1*sg[6]
                K[340] = c1*sg[7]
                K[361] = 2*c1*sg[8]
                K[383] = c1*sg[6]
                K[404] = c1*sg[7]
                K[425] = 2*c1*sg[8]

                c = -1
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAw_yy = gtab[6, j]
                    gAphix = gtab[7, j]
                    gAphix_y = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_y = gtab[10, j]
                    gAu_yy = gtab[11, j]
                    gAv_yy = gtab[12, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_x = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_x = ftab[10, i]
                        fAu_xx = ftab[11, i]
                        fAv_xx = ftab[12, i]
                        row = row0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAphix*gAphix
                        pA[7] = fAphiy*gAphiy
                        pA[8] = fAphix_x*gAphix
                        pA[9] = fAphix*gAphix_y
                        pA[10] = fAphiy_x*gAphiy
                        pA[11] = fAphiy*gAphiy_y
                        pA[12] = fAw_xx*gAw
                        pA[13] = fAw*gAw_yy
                        pA[14] = fAw_x*gAw_y
                        pA[15] = fAu_xx*gAu
                        pA[16] = fAu*gAu_yy
                        pA[17] = fAu_x*gAu_y
                        pA[18] = fAv_xx*gAv
                        pA[19] = fAv*gAv_yy
                        pA[20] = fAv_x*gAv_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[21]
                        TA[1] = pA[0]*K[1] + pA[1]*K[22]
                        TA[4] = pA[15]*K[319] + pA[16]*K[340] + pA[17]*K[361]
                        TA[6] = pA[0]*K[6] + pA[1]*K[27]
                        TA[8] = pA[0]*K[8] + pA[1]*K[29]
                        TA[9] = pA[0]*K[9] + pA[1]*K[30]
                        TA[12] = pA[0]*K[12]
                        TA[13] = pA[0]*K[13]
                        TA[14] = pA[0]*K[14]
                        TA[23] = pA[2]*K[44] + pA[3]*K[65]
                        TA[24] = pA[2]*K[45] + pA[3]*K[66]
                        TA[26] = pA[18]*K[383] + pA[19]*K[404] + pA[20]*K[425]
                        TA[28] = pA[2]*K[49] + pA[3]*K[70]
                        TA[31] = pA[2]*K[52] + pA[3]*K[73]
                        TA[32] = pA[2]*K[53] + pA[3]*K[74]
                        TA[33] = pA[3]*K[75]
                        TA[34] = pA[3]*K[76]
                        TA[35] = pA[3]*K[77]
                        TA[42] = pA[12]*K[252] + pA[13]*K[273] + pA[14]*K[294]
                        TA[45] = pA[12]*K[255] + pA[13]*K[276] + pA[14]*K[297]
                        TA[46] = pA[4]*K[88] + pA[5]*K[109]
                        TA[47] = pA[4]*K[89] + pA[5]*K[110]
                        TA[57] = pA[4]*K[99]
                        TA[58] = pA[4]*K[100]
                        TA[59] = pA[4]*K[101]
                        TA[60] = pA[5]*K[123]
                        TA[61] = pA[5]*K[124]
                        TA[62] = pA[5]*K[125]
                        TA[63] = pA[6]*K[126] + pA[8]*K[168] + pA[9]*K[189]
                        TA[64] = pA[6]*K[127] + pA[8]*K[169] + pA[9]*K[190]
                        TA[86] = pA[7]*K[149] + pA[10]*K[212] + pA[11]*K[233]
                        TA[87] = pA[7]*K[150] + pA[10]*K[213] + pA[11]*K[234]
                        for l in range(n):
                            gBu = gtab[0, l]
                            gBu_y = gtab[1, l]
                            gBv = gtab[2, l]
                            gBv_y = gtab[3, l]
                            gBw = gtab[4, l]
                            gBw_y = gtab[5, l]
                            gBw_yy = gtab[6, l]
                            gBphix = gtab[7, l]
                            gBphix_y = gtab[8, l]
                            gBphiy = gtab[9, l]
                            gBphiy_y = gtab[10, l]
                            gBu_yy = gtab[11, l]
                            gBv_yy = gtab[12, l]
                            for k in range(m):
                                col = col0 + DOF*(l*m + k)
                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue
                                fBu = ftab[0, k]
                                fBu_x = ftab[1, k]
                                fBv = ftab[2, k]
                                fBv_x = ftab[3, k]
                                fBw = ftab[4, k]
                                fBw_x = ftab[5, k]
                                fBw_xx = ftab[6, k]
                                fBphix = ftab[7, k]
                                fBphix_x = ftab[8, k]
                                fBphiy = ftab[9, k]
                                fBphiy_x = ftab[10, k]
                                fBu_xx = ftab[11, k]
                                fBv_xx = ftab[12, k]
                                pB[0] = fBu_x*gBu
                                pB[1] = fBu*gBu_y
                                pB[2] = fBv_x*gBv
                                pB[3] = fBv*gBv_y
                                pB[4] = fBw_x*gBw
                                pB[5] = fBw*gBw_y
                                pB[6] = fBphix*gBphix
                                pB[7] = fBphiy*gBphiy
                                pB[8] = fBphix_x*gBphix
                                pB[9] = fBphix*gBphix_y
                                pB[10] = fBphiy_x*gBphiy
                                pB[11] = fBphiy*gBphiy_y
                                pB[12] = fBw_xx*gBw
                                pB[13] = fBw*gBw_yy
                                pB[14] = fBw_x*gBw_y
                                pB[15] = fBu_xx*gBu
                                pB[16] = fBu*gBu_yy
                                pB[17] = fBu_x*gBu_y
                                pB[18] = fBv_xx*gBv
                                pB[19] = fBv*gBv_yy
                                pB[20] = fBv_x*gBv_y
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[4]*pB[4] + TA[12]*pB[12] + TA[13]*pB[13] + TA[14]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+3
                                kGv[c] += weight*(TA[6]*pB[6] + TA[8]*pB[8] + TA[9]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[23]*pB[2] + TA[24]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[26]*pB[5] + TA[33]*pB[12] + TA[34]*pB[13] + TA[35]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+4
                                kGv[c] += weight*(TA[28]*pB[7] + TA[31]*pB[10] + TA[32]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[42]*pB[0] + TA[57]*pB[15] + TA[58]*pB[16] + TA[59]*pB[17])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[45]*pB[3] + TA[60]*pB[18] + TA[61]*pB[19] + TA[62]*pB[20])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[46]*pB[4] + TA[47]*pB[5])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+3
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[63]*pB[0] + TA[64]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+4
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[86]*pB[2] + TA[87]*pB[3])

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG


def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    r"""Internal force vector, integrated numerically

    Evaluates the generalized strains and stress resultants at
    ``cs`` at each integration point and integrates the internal force
    vector, whose Jacobian is ``fkC_num(NLgeom=1) + fkG_num``.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements (the rotations in radians).
    Finput : array-like
        Constitutive matrix of the generalized strains of the model, shape
        ``(13, 13)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 13, 13)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Size of the vector, the number of degrees of freedom of the assembly.
    col0 : int
        Position of the vector where the terms of this shell start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    fint : numpy.ndarray
        Internal force vector of size ``size``.

    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape or ``shell`` is not a
        :class:`.Shell`.

    Notes
    -----
    With the generalized strains of the kinematics of the module,
    `\boldsymbol{\varepsilon}(\boldsymbol{c})`, their variation `\delta
    \boldsymbol{\varepsilon} = \boldsymbol{B}(\boldsymbol{c}) \delta
    \boldsymbol{c}` and the stress resultants `\boldsymbol{\sigma} =
    \boldsymbol{F} \boldsymbol{\varepsilon}`:

    .. math::

        \boldsymbol{f}_{int} = \int \boldsymbol{B}(\boldsymbol{c})^\top
        \boldsymbol{\sigma} \, dA

    the gradient of the strain energy `\frac{1}{2} \int
    \boldsymbol{\varepsilon}^\top \boldsymbol{F} \boldsymbol{\varepsilon}
    \, dA`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \le x \le x_2`, `y_1 \le y \le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

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

    cdef int i, j, k, l, c, p, q, r, s, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy
    cdef double phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy, u_xx
    cdef double u_yy, u_xy, v_xx, v_yy, v_xy
    cdef double e[13]
    cdef double sg[13]
    cdef double S[21]
    cdef double pA[21]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAw_xx
    cdef double fAphix, fAphix_x, fAphiy, fAphiy_x, fAu_xx, fAv_xx, gAu
    cdef double gAu_y, gAv, gAv_y, gAw, gAw_y, gAw_yy, gAphix
    cdef double gAphix_y, gAphiy, gAphiy_y, gAu_yy, gAv_yy

    cdef double [::1] fint

    # F as 4-D matrix, must be [nx, ny, NE, NE], when there is one F[NE, NE]
    # for each of the nx * ny integration points
    cdef double F[13 * 13]
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

    ftab = np.zeros((13, m), dtype=DOUBLE)
    gtab = np.zeros((13, n), dtype=DOUBLE)

    fint = np.zeros(size, dtype=DOUBLE)

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
                    for i in range(NE):
                        for j in range(NE):
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = (2/a)*fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[9, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[11, i] = (2/a)*(2/a)*fpp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[12, i] = (2/a)*(2/a)*fpp(i, xi, x1v, x1vr, x2v, x2vr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[9, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[10, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[11, j] = (2/b)*(2/b)*fpp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[12, j] = (2/b)*(2/b)*fpp(j, eta, y1v, y1vr, y2v, y2vr)

                # generalized derivatives at the point
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w_x = 0.
                w_y = 0.
                phix = 0.
                phiy = 0.
                phix_x = 0.
                phix_y = 0.
                phiy_x = 0.
                phiy_y = 0.
                w_xx = 0.
                w_yy = 0.
                w_xy = 0.
                u_xx = 0.
                u_yy = 0.
                u_xy = 0.
                v_xx = 0.
                v_yy = 0.
                v_xy = 0.
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAw_yy = gtab[6, j]
                    gAphix = gtab[7, j]
                    gAphix_y = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_y = gtab[10, j]
                    gAu_yy = gtab[11, j]
                    gAv_yy = gtab[12, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_x = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_x = ftab[10, i]
                        fAu_xx = ftab[11, i]
                        fAv_xx = ftab[12, i]
                        col = col0 + DOF*(j*m + i)
                        u_x += fAu_x*gAu*cs[col+0]
                        u_y += fAu*gAu_y*cs[col+0]
                        v_x += fAv_x*gAv*cs[col+1]
                        v_y += fAv*gAv_y*cs[col+1]
                        w_x += fAw_x*gAw*cs[col+2]
                        w_y += fAw*gAw_y*cs[col+2]
                        phix += fAphix*gAphix*cs[col+3]
                        phiy += fAphiy*gAphiy*cs[col+4]
                        phix_x += fAphix_x*gAphix*cs[col+3]
                        phix_y += fAphix*gAphix_y*cs[col+3]
                        phiy_x += fAphiy_x*gAphiy*cs[col+4]
                        phiy_y += fAphiy*gAphiy_y*cs[col+4]
                        w_xx += fAw_xx*gAw*cs[col+2]
                        w_yy += fAw*gAw_yy*cs[col+2]
                        w_xy += fAw_x*gAw_y*cs[col+2]
                        u_xx += fAu_xx*gAu*cs[col+0]
                        u_yy += fAu*gAu_yy*cs[col+0]
                        u_xy += fAu_x*gAu_y*cs[col+0]
                        v_xx += fAv_xx*gAv*cs[col+1]
                        v_yy += fAv*gAv_yy*cs[col+1]
                        v_xy += fAv_x*gAv_y*cs[col+1]

                # strains, stresses and S = [B]^T sigma
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x + phiy_x*v_x
                e[4] = -phix*phix_y*w_y + phix_y*u_y - phiy*phiy_y*w_y + phiy_y*v_y + phiy_y
                e[5] = -phix*phix_x*w_y - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - phiy*phiy_x*w_y - phiy*phiy_y*w_x + phiy_x*v_y + phiy_x + phiy_y*v_x
                e[6] = c1*phix*phix_x*w_x - c1*phix_x*u_x - c1*phix_x + c1*phiy*phiy_x*w_x - c1*phiy_x*v_x + c1*u_x*v_xx*w_y - c1*u_x*v_y*w_xx - c1*u_x*w_xx - c1*u_xx*v_x*w_y + c1*u_xx*v_y*w_x + c1*u_xx*w_x + c1*u_y*v_x*w_xx - c1*u_y*v_xx*w_x + c1*v_xx*w_y - c1*v_y*w_xx - c1*w_xx
                e[7] = c1*phix*phix_y*w_y - c1*phix_y*u_y + c1*phiy*phiy_y*w_y - c1*phiy_y*v_y - c1*phiy_y - c1*u_x*v_y*w_yy + c1*u_x*v_yy*w_y - c1*u_x*w_yy + c1*u_y*v_x*w_yy - c1*u_y*v_yy*w_x - c1*u_yy*v_x*w_y + c1*u_yy*v_y*w_x + c1*u_yy*w_x - c1*v_y*w_yy + c1*v_yy*w_y - c1*w_yy
                e[8] = c1*phix*phix_x*w_y + c1*phix*phix_y*w_x - c1*phix_x*u_y - c1*phix_y*u_x - c1*phix_y + c1*phiy*phiy_x*w_y + c1*phiy*phiy_y*w_x - c1*phiy_x*v_y - c1*phiy_x - c1*phiy_y*v_x + 2*c1*u_x*v_xy*w_y - 2*c1*u_x*v_y*w_xy - 2*c1*u_x*w_xy - 2*c1*u_xy*v_x*w_y + 2*c1*u_xy*v_y*w_x + 2*c1*u_xy*w_x + 2*c1*u_y*v_x*w_xy - 2*c1*u_y*v_xy*w_x + 2*c1*v_xy*w_y - 2*c1*v_y*w_xy - 2*c1*w_xy
                e[9] = -0.5*(phix*phix)*w_y + phix*u_y - 0.5*(phiy*phiy)*w_y + phiy*v_y + phiy + w_y
                e[10] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x + w_x
                e[11] = 1.5*c1*(phix*phix)*w_y - 3*c1*phix*u_y + 1.5*c1*(phiy*phiy)*w_y - 3*c1*phiy*v_y - 3*c1*phiy - 3*c1*w_y
                e[12] = 1.5*c1*(phix*phix)*w_x - 3*c1*phix*u_x - 3*c1*phix + 1.5*c1*(phiy*phiy)*w_x - 3*c1*phiy*v_x - 3*c1*w_x
                for r in range(NE):
                    sg[r] = 0.
                    for s in range(NE):
                        sg[r] += F[r*NE + s]*e[s]
                S[0] = -3*c1*phix*sg[12] - c1*phix_x*sg[6] - c1*phix_y*sg[8] + c1*sg[6]*v_xx*w_y - c1*sg[6]*v_y*w_xx - c1*sg[6]*w_xx - c1*sg[7]*v_y*w_yy + c1*sg[7]*v_yy*w_y - c1*sg[7]*w_yy + 2*c1*sg[8]*v_xy*w_y - 2*c1*sg[8]*v_y*w_xy - 2*c1*sg[8]*w_xy + phix*sg[10] + phix_x*sg[3] + phix_y*sg[5] + sg[0]*u_x + sg[0] + sg[2]*u_y
                S[1] = -3*c1*phix*sg[11] - c1*phix_x*sg[8] - c1*phix_y*sg[7] + c1*sg[6]*v_x*w_xx - c1*sg[6]*v_xx*w_x + c1*sg[7]*v_x*w_yy - c1*sg[7]*v_yy*w_x + 2*c1*sg[8]*v_x*w_xy - 2*c1*sg[8]*v_xy*w_x + phix*sg[9] + phix_x*sg[5] + phix_y*sg[4] + sg[1]*u_y + sg[2]*u_x + sg[2]
                S[2] = -3*c1*phiy*sg[12] - c1*phiy_x*sg[6] - c1*phiy_y*sg[8] - c1*sg[6]*u_xx*w_y + c1*sg[6]*u_y*w_xx + c1*sg[7]*u_y*w_yy - c1*sg[7]*u_yy*w_y - 2*c1*sg[8]*u_xy*w_y + 2*c1*sg[8]*u_y*w_xy + phiy*sg[10] + phiy_x*sg[3] + phiy_y*sg[5] + sg[0]*v_x + sg[2]*v_y + sg[2]
                S[3] = -3*c1*phiy*sg[11] - c1*phiy_x*sg[8] - c1*phiy_y*sg[7] - c1*sg[6]*u_x*w_xx + c1*sg[6]*u_xx*w_x - c1*sg[6]*w_xx - c1*sg[7]*u_x*w_yy + c1*sg[7]*u_yy*w_x - c1*sg[7]*w_yy - 2*c1*sg[8]*u_x*w_xy + 2*c1*sg[8]*u_xy*w_x - 2*c1*sg[8]*w_xy + phiy*sg[9] + phiy_x*sg[5] + phiy_y*sg[4] + sg[1]*v_y + sg[1] + sg[2]*v_x
                S[4] = 1.5*c1*(phix*phix)*sg[12] + c1*phix*phix_x*sg[6] + c1*phix*phix_y*sg[8] + 1.5*c1*(phiy*phiy)*sg[12] + c1*phiy*phiy_x*sg[6] + c1*phiy*phiy_y*sg[8] - 3*c1*sg[12] + c1*sg[6]*u_xx*v_y + c1*sg[6]*u_xx - c1*sg[6]*u_y*v_xx - c1*sg[7]*u_y*v_yy + c1*sg[7]*u_yy*v_y + c1*sg[7]*u_yy + 2*c1*sg[8]*u_xy*v_y + 2*c1*sg[8]*u_xy - 2*c1*sg[8]*u_y*v_xy - 0.5*(phix*phix)*sg[10] - phix*phix_x*sg[3] - phix*phix_y*sg[5] - 0.5*(phiy*phiy)*sg[10] - phiy*phiy_x*sg[3] - phiy*phiy_y*sg[5] + sg[0]*w_x + sg[10] + sg[2]*w_y
                S[5] = 1.5*c1*(phix*phix)*sg[11] + c1*phix*phix_x*sg[8] + c1*phix*phix_y*sg[7] + 1.5*c1*(phiy*phiy)*sg[11] + c1*phiy*phiy_x*sg[8] + c1*phiy*phiy_y*sg[7] - 3*c1*sg[11] + c1*sg[6]*u_x*v_xx - c1*sg[6]*u_xx*v_x + c1*sg[6]*v_xx + c1*sg[7]*u_x*v_yy - c1*sg[7]*u_yy*v_x + c1*sg[7]*v_yy + 2*c1*sg[8]*u_x*v_xy - 2*c1*sg[8]*u_xy*v_x + 2*c1*sg[8]*v_xy - 0.5*(phix*phix)*sg[9] - phix*phix_x*sg[5] - phix*phix_y*sg[4] - 0.5*(phiy*phiy)*sg[9] - phiy*phiy_x*sg[5] - phiy*phiy_y*sg[4] + sg[1]*w_y + sg[2]*w_x + sg[9]
                S[6] = 3*c1*phix*sg[11]*w_y + 3*c1*phix*sg[12]*w_x + c1*phix_x*sg[6]*w_x + c1*phix_x*sg[8]*w_y + c1*phix_y*sg[7]*w_y + c1*phix_y*sg[8]*w_x - 3*c1*sg[11]*u_y - 3*c1*sg[12]*u_x - 3*c1*sg[12] - phix*sg[10]*w_x - phix*sg[9]*w_y - phix_x*sg[3]*w_x - phix_x*sg[5]*w_y - phix_y*sg[4]*w_y - phix_y*sg[5]*w_x + sg[10]*u_x + sg[10] + sg[9]*u_y
                S[7] = 3*c1*phiy*sg[11]*w_y + 3*c1*phiy*sg[12]*w_x + c1*phiy_x*sg[6]*w_x + c1*phiy_x*sg[8]*w_y + c1*phiy_y*sg[7]*w_y + c1*phiy_y*sg[8]*w_x - 3*c1*sg[11]*v_y - 3*c1*sg[11] - 3*c1*sg[12]*v_x - phiy*sg[10]*w_x - phiy*sg[9]*w_y - phiy_x*sg[3]*w_x - phiy_x*sg[5]*w_y - phiy_y*sg[4]*w_y - phiy_y*sg[5]*w_x + sg[10]*v_x + sg[9]*v_y + sg[9]
                S[8] = c1*phix*sg[6]*w_x + c1*phix*sg[8]*w_y - c1*sg[6]*u_x - c1*sg[6] - c1*sg[8]*u_y - phix*sg[3]*w_x - phix*sg[5]*w_y + sg[3]*u_x + sg[3] + sg[5]*u_y
                S[9] = c1*phix*sg[7]*w_y + c1*phix*sg[8]*w_x - c1*sg[7]*u_y - c1*sg[8]*u_x - c1*sg[8] - phix*sg[4]*w_y - phix*sg[5]*w_x + sg[4]*u_y + sg[5]*u_x + sg[5]
                S[10] = c1*phiy*sg[6]*w_x + c1*phiy*sg[8]*w_y - c1*sg[6]*v_x - c1*sg[8]*v_y - c1*sg[8] - phiy*sg[3]*w_x - phiy*sg[5]*w_y + sg[3]*v_x + sg[5]*v_y + sg[5]
                S[11] = c1*phiy*sg[7]*w_y + c1*phiy*sg[8]*w_x - c1*sg[7]*v_y - c1*sg[7] - c1*sg[8]*v_x - phiy*sg[4]*w_y - phiy*sg[5]*w_x + sg[4]*v_y + sg[4] + sg[5]*v_x
                S[12] = -c1*sg[6]*u_x*v_y - c1*sg[6]*u_x + c1*sg[6]*u_y*v_x - c1*sg[6]*v_y - c1*sg[6]
                S[13] = -c1*sg[7]*u_x*v_y - c1*sg[7]*u_x + c1*sg[7]*u_y*v_x - c1*sg[7]*v_y - c1*sg[7]
                S[14] = -2*c1*sg[8]*u_x*v_y - 2*c1*sg[8]*u_x + 2*c1*sg[8]*u_y*v_x - 2*c1*sg[8]*v_y - 2*c1*sg[8]
                S[15] = -c1*sg[6]*v_x*w_y + c1*sg[6]*v_y*w_x + c1*sg[6]*w_x
                S[16] = -c1*sg[7]*v_x*w_y + c1*sg[7]*v_y*w_x + c1*sg[7]*w_x
                S[17] = -2*c1*sg[8]*v_x*w_y + 2*c1*sg[8]*v_y*w_x + 2*c1*sg[8]*w_x
                S[18] = c1*sg[6]*u_x*w_y - c1*sg[6]*u_y*w_x + c1*sg[6]*w_y
                S[19] = c1*sg[7]*u_x*w_y - c1*sg[7]*u_y*w_x + c1*sg[7]*w_y
                S[20] = 2*c1*sg[8]*u_x*w_y - 2*c1*sg[8]*u_y*w_x + 2*c1*sg[8]*w_y

                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAw_yy = gtab[6, j]
                    gAphix = gtab[7, j]
                    gAphix_y = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_y = gtab[10, j]
                    gAu_yy = gtab[11, j]
                    gAv_yy = gtab[12, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_x = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_x = ftab[10, i]
                        fAu_xx = ftab[11, i]
                        fAv_xx = ftab[12, i]
                        col = col0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAphix*gAphix
                        pA[7] = fAphiy*gAphiy
                        pA[8] = fAphix_x*gAphix
                        pA[9] = fAphix*gAphix_y
                        pA[10] = fAphiy_x*gAphiy
                        pA[11] = fAphiy*gAphiy_y
                        pA[12] = fAw_xx*gAw
                        pA[13] = fAw*gAw_yy
                        pA[14] = fAw_x*gAw_y
                        pA[15] = fAu_xx*gAu
                        pA[16] = fAu*gAu_yy
                        pA[17] = fAu_x*gAu_y
                        pA[18] = fAv_xx*gAv
                        pA[19] = fAv*gAv_yy
                        pA[20] = fAv_x*gAv_y
                        fint[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1] + pA[15]*S[15] + pA[16]*S[16] + pA[17]*S[17])
                        fint[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[18]*S[18] + pA[19]*S[19] + pA[20]*S[20])
                        fint[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[12]*S[12] + pA[13]*S[13] + pA[14]*S[14])
                        fint[col+3] += weight*(pA[6]*S[6] + pA[8]*S[8] + pA[9]*S[9])
                        fint[col+4] += weight*(pA[7]*S[7] + pA[10]*S[10] + pA[11]*S[11])

    return np.asarray(fint)


def calc_kTv(double [::1] cs, double [::1] vs, object Finput, object shell,
        int size, int col0, int nx, int ny, double Nxx_cte=0.,
        double Nyy_cte=0., double Nxy_cte=0.):
    r"""Product of the tangent stiffness matrix by a vector, without assembly

    Returns ``KT(cs) vs``, with ``KT`` the tangent stiffness matrix
    ``fkC_num(cs, ..., NLgeom=1) + fkG_num(cs, ...)`` at ``cs``, the exact
    Jacobian of :func:`calc_fint`, plus the geometric stiffness matrix of
    the constant membrane stress state ``Nxx_cte``, ``Nyy_cte`` and
    ``Nxy_cte``. The cost is about that of :func:`calc_fint`.

    Parameters
    ----------
    cs, vs : array-like
        Ritz constants of the configuration and the vector, 1-D of size
        ``size``.
    Finput : array-like
        Constitutive matrix of the generalized strains of the model, shape
        ``(13, 13)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 13, 13)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size, col0 : int
        Size of the vectors and position where the terms of this shell
        start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.
    Nxx_cte, Nyy_cte, Nxy_cte : float, optional
        Constant membrane stress resultants, force per unit length, default
        ``0``.

    Returns
    -------
    kTv : numpy.ndarray
        The product, of size ``size``.

    Raises
    ------
    ValueError
        If ``cs`` or ``vs`` do not have the size ``size``, ``Finput`` has an
        invalid shape or ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With the generalized derivatives at the integration point `\boldsymbol{d}
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the rotations and their gradients), the strains
    `\boldsymbol{\varepsilon}(\boldsymbol{d})`, cubic, `\boldsymbol{B} =
    \partial \boldsymbol{\varepsilon} / \partial \boldsymbol{d}`, the
    stresses `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}`, `\boldsymbol{H}(\boldsymbol{\sigma},
    \boldsymbol{d}) = \sum_r \sigma_r \, \partial^2 \varepsilon_r /
    \partial \boldsymbol{d}^2`, linear in `\boldsymbol{d}`, and
    `\boldsymbol{H}_0(\boldsymbol{\sigma}) =
    \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{0})`:

    .. math::

        \boldsymbol{K}_T \boldsymbol{v} = \int \boldsymbol{\Phi}^\top
        \left[\boldsymbol{B}^\top \boldsymbol{F} \boldsymbol{B} +
        \boldsymbol{H}(\boldsymbol{\sigma} + \bar{\boldsymbol{\sigma}},
        \boldsymbol{d})\right] \boldsymbol{\Phi} \boldsymbol{v} \, dA

    with `\bar{\boldsymbol{\sigma}}` the constant membrane stresses,
    evaluated without forming `\boldsymbol{K}_T`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \le x \le x_2`, `y_1 \le y \le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

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

    cdef int i, j, k, l, c, p, q, r, s, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, phix, phiy
    cdef double phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy, u_xx
    cdef double u_yy, u_xy, v_xx, v_yy, v_xy
    cdef double du_x, du_y, dv_x, dv_y, dw_x, dw_y, dphix, dphiy
    cdef double dphix_x, dphix_y, dphiy_x, dphiy_y, dw_xx, dw_yy, dw_xy, du_xx
    cdef double du_yy, du_xy, dv_xx, dv_yy, dv_xy
    cdef double e[13]
    cdef double de[13]
    cdef double sg[13]
    cdef double dsg[13]
    cdef double S[21]
    cdef double pA[21]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAw_xx
    cdef double fAphix, fAphix_x, fAphiy, fAphiy_x, fAu_xx, fAv_xx, gAu
    cdef double gAu_y, gAv, gAv_y, gAw, gAw_y, gAw_yy, gAphix
    cdef double gAphix_y, gAphiy, gAphiy_y, gAu_yy, gAv_yy

    cdef double [::1] kTv

    if cs.shape[0] != size or vs.shape[0] != size:
        raise ValueError('cs and vs must have the size %d' % size)

    # F as 4-D matrix, must be [nx, ny, NE, NE], when there is one F[NE, NE]
    # for each of the nx * ny integration points
    cdef double F[13 * 13]
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

    ftab = np.zeros((13, m), dtype=DOUBLE)
    gtab = np.zeros((13, n), dtype=DOUBLE)

    kTv = np.zeros(size, dtype=DOUBLE)

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
                    for i in range(NE):
                        for j in range(NE):
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = (2/a)*fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[9, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[11, i] = (2/a)*(2/a)*fpp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[12, i] = (2/a)*(2/a)*fpp(i, xi, x1v, x1vr, x2v, x2vr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[9, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[10, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[11, j] = (2/b)*(2/b)*fpp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[12, j] = (2/b)*(2/b)*fpp(j, eta, y1v, y1vr, y2v, y2vr)

                # generalized derivatives of cs and vs at the point
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w_x = 0.
                w_y = 0.
                phix = 0.
                phiy = 0.
                phix_x = 0.
                phix_y = 0.
                phiy_x = 0.
                phiy_y = 0.
                w_xx = 0.
                w_yy = 0.
                w_xy = 0.
                u_xx = 0.
                u_yy = 0.
                u_xy = 0.
                v_xx = 0.
                v_yy = 0.
                v_xy = 0.
                du_x = 0.
                du_y = 0.
                dv_x = 0.
                dv_y = 0.
                dw_x = 0.
                dw_y = 0.
                dphix = 0.
                dphiy = 0.
                dphix_x = 0.
                dphix_y = 0.
                dphiy_x = 0.
                dphiy_y = 0.
                dw_xx = 0.
                dw_yy = 0.
                dw_xy = 0.
                du_xx = 0.
                du_yy = 0.
                du_xy = 0.
                dv_xx = 0.
                dv_yy = 0.
                dv_xy = 0.
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAw_yy = gtab[6, j]
                    gAphix = gtab[7, j]
                    gAphix_y = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_y = gtab[10, j]
                    gAu_yy = gtab[11, j]
                    gAv_yy = gtab[12, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_x = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_x = ftab[10, i]
                        fAu_xx = ftab[11, i]
                        fAv_xx = ftab[12, i]
                        col = col0 + DOF*(j*m + i)
                        u_x += fAu_x*gAu*cs[col+0]
                        u_y += fAu*gAu_y*cs[col+0]
                        v_x += fAv_x*gAv*cs[col+1]
                        v_y += fAv*gAv_y*cs[col+1]
                        w_x += fAw_x*gAw*cs[col+2]
                        w_y += fAw*gAw_y*cs[col+2]
                        phix += fAphix*gAphix*cs[col+3]
                        phiy += fAphiy*gAphiy*cs[col+4]
                        phix_x += fAphix_x*gAphix*cs[col+3]
                        phix_y += fAphix*gAphix_y*cs[col+3]
                        phiy_x += fAphiy_x*gAphiy*cs[col+4]
                        phiy_y += fAphiy*gAphiy_y*cs[col+4]
                        w_xx += fAw_xx*gAw*cs[col+2]
                        w_yy += fAw*gAw_yy*cs[col+2]
                        w_xy += fAw_x*gAw_y*cs[col+2]
                        u_xx += fAu_xx*gAu*cs[col+0]
                        u_yy += fAu*gAu_yy*cs[col+0]
                        u_xy += fAu_x*gAu_y*cs[col+0]
                        v_xx += fAv_xx*gAv*cs[col+1]
                        v_yy += fAv*gAv_yy*cs[col+1]
                        v_xy += fAv_x*gAv_y*cs[col+1]
                        du_x += fAu_x*gAu*vs[col+0]
                        du_y += fAu*gAu_y*vs[col+0]
                        dv_x += fAv_x*gAv*vs[col+1]
                        dv_y += fAv*gAv_y*vs[col+1]
                        dw_x += fAw_x*gAw*vs[col+2]
                        dw_y += fAw*gAw_y*vs[col+2]
                        dphix += fAphix*gAphix*vs[col+3]
                        dphiy += fAphiy*gAphiy*vs[col+4]
                        dphix_x += fAphix_x*gAphix*vs[col+3]
                        dphix_y += fAphix*gAphix_y*vs[col+3]
                        dphiy_x += fAphiy_x*gAphiy*vs[col+4]
                        dphiy_y += fAphiy*gAphiy_y*vs[col+4]
                        dw_xx += fAw_xx*gAw*vs[col+2]
                        dw_yy += fAw*gAw_yy*vs[col+2]
                        dw_xy += fAw_x*gAw_y*vs[col+2]
                        du_xx += fAu_xx*gAu*vs[col+0]
                        du_yy += fAu*gAu_yy*vs[col+0]
                        du_xy += fAu_x*gAu_y*vs[col+0]
                        dv_xx += fAv_xx*gAv*vs[col+1]
                        dv_yy += fAv*gAv_yy*vs[col+1]
                        dv_xy += fAv_x*gAv_y*vs[col+1]

                # strains, strain increments, stresses
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x + phiy_x*v_x
                e[4] = -phix*phix_y*w_y + phix_y*u_y - phiy*phiy_y*w_y + phiy_y*v_y + phiy_y
                e[5] = -phix*phix_x*w_y - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - phiy*phiy_x*w_y - phiy*phiy_y*w_x + phiy_x*v_y + phiy_x + phiy_y*v_x
                e[6] = c1*phix*phix_x*w_x - c1*phix_x*u_x - c1*phix_x + c1*phiy*phiy_x*w_x - c1*phiy_x*v_x + c1*u_x*v_xx*w_y - c1*u_x*v_y*w_xx - c1*u_x*w_xx - c1*u_xx*v_x*w_y + c1*u_xx*v_y*w_x + c1*u_xx*w_x + c1*u_y*v_x*w_xx - c1*u_y*v_xx*w_x + c1*v_xx*w_y - c1*v_y*w_xx - c1*w_xx
                e[7] = c1*phix*phix_y*w_y - c1*phix_y*u_y + c1*phiy*phiy_y*w_y - c1*phiy_y*v_y - c1*phiy_y - c1*u_x*v_y*w_yy + c1*u_x*v_yy*w_y - c1*u_x*w_yy + c1*u_y*v_x*w_yy - c1*u_y*v_yy*w_x - c1*u_yy*v_x*w_y + c1*u_yy*v_y*w_x + c1*u_yy*w_x - c1*v_y*w_yy + c1*v_yy*w_y - c1*w_yy
                e[8] = c1*phix*phix_x*w_y + c1*phix*phix_y*w_x - c1*phix_x*u_y - c1*phix_y*u_x - c1*phix_y + c1*phiy*phiy_x*w_y + c1*phiy*phiy_y*w_x - c1*phiy_x*v_y - c1*phiy_x - c1*phiy_y*v_x + 2*c1*u_x*v_xy*w_y - 2*c1*u_x*v_y*w_xy - 2*c1*u_x*w_xy - 2*c1*u_xy*v_x*w_y + 2*c1*u_xy*v_y*w_x + 2*c1*u_xy*w_x + 2*c1*u_y*v_x*w_xy - 2*c1*u_y*v_xy*w_x + 2*c1*v_xy*w_y - 2*c1*v_y*w_xy - 2*c1*w_xy
                e[9] = -0.5*(phix*phix)*w_y + phix*u_y - 0.5*(phiy*phiy)*w_y + phiy*v_y + phiy + w_y
                e[10] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x + w_x
                e[11] = 1.5*c1*(phix*phix)*w_y - 3*c1*phix*u_y + 1.5*c1*(phiy*phiy)*w_y - 3*c1*phiy*v_y - 3*c1*phiy - 3*c1*w_y
                e[12] = 1.5*c1*(phix*phix)*w_x - 3*c1*phix*u_x - 3*c1*phix + 1.5*c1*(phiy*phiy)*w_x - 3*c1*phiy*v_x - 3*c1*w_x
                de[0] = du_x*u_x + du_x + dv_x*v_x + dw_x*w_x
                de[1] = du_y*u_y + dv_y*v_y + dv_y + dw_y*w_y
                de[2] = du_x*u_y + du_y*u_x + du_y + dv_x*v_y + dv_x + dv_y*v_x + dw_x*w_y + dw_y*w_x
                de[3] = -dphix*phix_x*w_x - dphix_x*phix*w_x + dphix_x*u_x + dphix_x - dphiy*phiy_x*w_x - dphiy_x*phiy*w_x + dphiy_x*v_x + du_x*phix_x + dv_x*phiy_x - dw_x*phix*phix_x - dw_x*phiy*phiy_x
                de[4] = -dphix*phix_y*w_y - dphix_y*phix*w_y + dphix_y*u_y - dphiy*phiy_y*w_y - dphiy_y*phiy*w_y + dphiy_y*v_y + dphiy_y + du_y*phix_y + dv_y*phiy_y - dw_y*phix*phix_y - dw_y*phiy*phiy_y
                de[5] = -dphix*phix_x*w_y - dphix*phix_y*w_x - dphix_x*phix*w_y + dphix_x*u_y - dphix_y*phix*w_x + dphix_y*u_x + dphix_y - dphiy*phiy_x*w_y - dphiy*phiy_y*w_x - dphiy_x*phiy*w_y + dphiy_x*v_y + dphiy_x - dphiy_y*phiy*w_x + dphiy_y*v_x + du_x*phix_y + du_y*phix_x + dv_x*phiy_y + dv_y*phiy_x - dw_x*phix*phix_y - dw_x*phiy*phiy_y - dw_y*phix*phix_x - dw_y*phiy*phiy_x
                de[6] = c1*dphix*phix_x*w_x + c1*dphix_x*phix*w_x - c1*dphix_x*u_x - c1*dphix_x + c1*dphiy*phiy_x*w_x + c1*dphiy_x*phiy*w_x - c1*dphiy_x*v_x - c1*du_x*phix_x + c1*du_x*v_xx*w_y - c1*du_x*v_y*w_xx - c1*du_x*w_xx - c1*du_xx*v_x*w_y + c1*du_xx*v_y*w_x + c1*du_xx*w_x + c1*du_y*v_x*w_xx - c1*du_y*v_xx*w_x - c1*dv_x*phiy_x - c1*dv_x*u_xx*w_y + c1*dv_x*u_y*w_xx + c1*dv_xx*u_x*w_y - c1*dv_xx*u_y*w_x + c1*dv_xx*w_y - c1*dv_y*u_x*w_xx + c1*dv_y*u_xx*w_x - c1*dv_y*w_xx + c1*dw_x*phix*phix_x + c1*dw_x*phiy*phiy_x + c1*dw_x*u_xx*v_y + c1*dw_x*u_xx - c1*dw_x*u_y*v_xx - c1*dw_xx*u_x*v_y - c1*dw_xx*u_x + c1*dw_xx*u_y*v_x - c1*dw_xx*v_y - c1*dw_xx + c1*dw_y*u_x*v_xx - c1*dw_y*u_xx*v_x + c1*dw_y*v_xx
                de[7] = c1*dphix*phix_y*w_y + c1*dphix_y*phix*w_y - c1*dphix_y*u_y + c1*dphiy*phiy_y*w_y + c1*dphiy_y*phiy*w_y - c1*dphiy_y*v_y - c1*dphiy_y - c1*du_x*v_y*w_yy + c1*du_x*v_yy*w_y - c1*du_x*w_yy - c1*du_y*phix_y + c1*du_y*v_x*w_yy - c1*du_y*v_yy*w_x - c1*du_yy*v_x*w_y + c1*du_yy*v_y*w_x + c1*du_yy*w_x + c1*dv_x*u_y*w_yy - c1*dv_x*u_yy*w_y - c1*dv_y*phiy_y - c1*dv_y*u_x*w_yy + c1*dv_y*u_yy*w_x - c1*dv_y*w_yy + c1*dv_yy*u_x*w_y - c1*dv_yy*u_y*w_x + c1*dv_yy*w_y - c1*dw_x*u_y*v_yy + c1*dw_x*u_yy*v_y + c1*dw_x*u_yy + c1*dw_y*phix*phix_y + c1*dw_y*phiy*phiy_y + c1*dw_y*u_x*v_yy - c1*dw_y*u_yy*v_x + c1*dw_y*v_yy - c1*dw_yy*u_x*v_y - c1*dw_yy*u_x + c1*dw_yy*u_y*v_x - c1*dw_yy*v_y - c1*dw_yy
                de[8] = c1*dphix*phix_x*w_y + c1*dphix*phix_y*w_x + c1*dphix_x*phix*w_y - c1*dphix_x*u_y + c1*dphix_y*phix*w_x - c1*dphix_y*u_x - c1*dphix_y + c1*dphiy*phiy_x*w_y + c1*dphiy*phiy_y*w_x + c1*dphiy_x*phiy*w_y - c1*dphiy_x*v_y - c1*dphiy_x + c1*dphiy_y*phiy*w_x - c1*dphiy_y*v_x - c1*du_x*phix_y + 2*c1*du_x*v_xy*w_y - 2*c1*du_x*v_y*w_xy - 2*c1*du_x*w_xy - 2*c1*du_xy*v_x*w_y + 2*c1*du_xy*v_y*w_x + 2*c1*du_xy*w_x - c1*du_y*phix_x + 2*c1*du_y*v_x*w_xy - 2*c1*du_y*v_xy*w_x - c1*dv_x*phiy_y - 2*c1*dv_x*u_xy*w_y + 2*c1*dv_x*u_y*w_xy + 2*c1*dv_xy*u_x*w_y - 2*c1*dv_xy*u_y*w_x + 2*c1*dv_xy*w_y - c1*dv_y*phiy_x - 2*c1*dv_y*u_x*w_xy + 2*c1*dv_y*u_xy*w_x - 2*c1*dv_y*w_xy + c1*dw_x*phix*phix_y + c1*dw_x*phiy*phiy_y + 2*c1*dw_x*u_xy*v_y + 2*c1*dw_x*u_xy - 2*c1*dw_x*u_y*v_xy - 2*c1*dw_xy*u_x*v_y - 2*c1*dw_xy*u_x + 2*c1*dw_xy*u_y*v_x - 2*c1*dw_xy*v_y - 2*c1*dw_xy + c1*dw_y*phix*phix_x + c1*dw_y*phiy*phiy_x + 2*c1*dw_y*u_x*v_xy - 2*c1*dw_y*u_xy*v_x + 2*c1*dw_y*v_xy
                de[9] = -dphix*phix*w_y + dphix*u_y - dphiy*phiy*w_y + dphiy*v_y + dphiy + du_y*phix + dv_y*phiy - 0.5*dw_y*(phix*phix) - 0.5*dw_y*(phiy*phiy) + dw_y
                de[10] = -dphix*phix*w_x + dphix*u_x + dphix - dphiy*phiy*w_x + dphiy*v_x + du_x*phix + dv_x*phiy - 0.5*dw_x*(phix*phix) - 0.5*dw_x*(phiy*phiy) + dw_x
                de[11] = 3*c1*dphix*phix*w_y - 3*c1*dphix*u_y + 3*c1*dphiy*phiy*w_y - 3*c1*dphiy*v_y - 3*c1*dphiy - 3*c1*du_y*phix - 3*c1*dv_y*phiy + 1.5*c1*dw_y*(phix*phix) + 1.5*c1*dw_y*(phiy*phiy) - 3*c1*dw_y
                de[12] = 3*c1*dphix*phix*w_x - 3*c1*dphix*u_x - 3*c1*dphix + 3*c1*dphiy*phiy*w_x - 3*c1*dphiy*v_x - 3*c1*du_x*phix - 3*c1*dv_x*phiy + 1.5*c1*dw_x*(phix*phix) + 1.5*c1*dw_x*(phiy*phiy) - 3*c1*dw_x
                for r in range(NE):
                    sg[r] = 0.
                    for s in range(NE):
                        sg[r] += F[r*NE + s]*e[s]
                for r in range(NE):
                    dsg[r] = 0.
                    for s in range(NE):
                        dsg[r] += F[r*NE + s]*de[s]
                sg[0] += Nxx_cte
                sg[1] += Nyy_cte
                sg[2] += Nxy_cte

                # S = [B]^T dsigma + [H(sigma, d)] dv
                S[0] = -3*c1*dphix*sg[12] - c1*dphix_x*sg[6] - c1*dphix_y*sg[8] - 3*c1*dsg[12]*phix - c1*dsg[6]*phix_x + c1*dsg[6]*v_xx*w_y - c1*dsg[6]*v_y*w_xx - c1*dsg[6]*w_xx - c1*dsg[7]*v_y*w_yy + c1*dsg[7]*v_yy*w_y - c1*dsg[7]*w_yy - c1*dsg[8]*phix_y + 2*c1*dsg[8]*v_xy*w_y - 2*c1*dsg[8]*v_y*w_xy - 2*c1*dsg[8]*w_xy + c1*dv_xx*sg[6]*w_y + 2*c1*dv_xy*sg[8]*w_y - c1*dv_y*sg[6]*w_xx - c1*dv_y*sg[7]*w_yy - 2*c1*dv_y*sg[8]*w_xy + c1*dv_yy*sg[7]*w_y - c1*dw_xx*sg[6]*v_y - c1*dw_xx*sg[6] - 2*c1*dw_xy*sg[8]*v_y - 2*c1*dw_xy*sg[8] + c1*dw_y*sg[6]*v_xx + c1*dw_y*sg[7]*v_yy + 2*c1*dw_y*sg[8]*v_xy - c1*dw_yy*sg[7]*v_y - c1*dw_yy*sg[7] + dphix*sg[10] + dphix_x*sg[3] + dphix_y*sg[5] + dsg[0]*u_x + dsg[0] + dsg[10]*phix + dsg[2]*u_y + dsg[3]*phix_x + dsg[5]*phix_y + du_x*sg[0] + du_y*sg[2]
                S[1] = -3*c1*dphix*sg[11] - c1*dphix_x*sg[8] - c1*dphix_y*sg[7] - 3*c1*dsg[11]*phix + c1*dsg[6]*v_x*w_xx - c1*dsg[6]*v_xx*w_x - c1*dsg[7]*phix_y + c1*dsg[7]*v_x*w_yy - c1*dsg[7]*v_yy*w_x - c1*dsg[8]*phix_x + 2*c1*dsg[8]*v_x*w_xy - 2*c1*dsg[8]*v_xy*w_x + c1*dv_x*sg[6]*w_xx + c1*dv_x*sg[7]*w_yy + 2*c1*dv_x*sg[8]*w_xy - c1*dv_xx*sg[6]*w_x - 2*c1*dv_xy*sg[8]*w_x - c1*dv_yy*sg[7]*w_x - c1*dw_x*sg[6]*v_xx - c1*dw_x*sg[7]*v_yy - 2*c1*dw_x*sg[8]*v_xy + c1*dw_xx*sg[6]*v_x + 2*c1*dw_xy*sg[8]*v_x + c1*dw_yy*sg[7]*v_x + dphix*sg[9] + dphix_x*sg[5] + dphix_y*sg[4] + dsg[1]*u_y + dsg[2]*u_x + dsg[2] + dsg[4]*phix_y + dsg[5]*phix_x + dsg[9]*phix + du_x*sg[2] + du_y*sg[1]
                S[2] = -3*c1*dphiy*sg[12] - c1*dphiy_x*sg[6] - c1*dphiy_y*sg[8] - 3*c1*dsg[12]*phiy - c1*dsg[6]*phiy_x - c1*dsg[6]*u_xx*w_y + c1*dsg[6]*u_y*w_xx + c1*dsg[7]*u_y*w_yy - c1*dsg[7]*u_yy*w_y - c1*dsg[8]*phiy_y - 2*c1*dsg[8]*u_xy*w_y + 2*c1*dsg[8]*u_y*w_xy - c1*du_xx*sg[6]*w_y - 2*c1*du_xy*sg[8]*w_y + c1*du_y*sg[6]*w_xx + c1*du_y*sg[7]*w_yy + 2*c1*du_y*sg[8]*w_xy - c1*du_yy*sg[7]*w_y + c1*dw_xx*sg[6]*u_y + 2*c1*dw_xy*sg[8]*u_y - c1*dw_y*sg[6]*u_xx - c1*dw_y*sg[7]*u_yy - 2*c1*dw_y*sg[8]*u_xy + c1*dw_yy*sg[7]*u_y + dphiy*sg[10] + dphiy_x*sg[3] + dphiy_y*sg[5] + dsg[0]*v_x + dsg[10]*phiy + dsg[2]*v_y + dsg[2] + dsg[3]*phiy_x + dsg[5]*phiy_y + dv_x*sg[0] + dv_y*sg[2]
                S[3] = -3*c1*dphiy*sg[11] - c1*dphiy_x*sg[8] - c1*dphiy_y*sg[7] - 3*c1*dsg[11]*phiy - c1*dsg[6]*u_x*w_xx + c1*dsg[6]*u_xx*w_x - c1*dsg[6]*w_xx - c1*dsg[7]*phiy_y - c1*dsg[7]*u_x*w_yy + c1*dsg[7]*u_yy*w_x - c1*dsg[7]*w_yy - c1*dsg[8]*phiy_x - 2*c1*dsg[8]*u_x*w_xy + 2*c1*dsg[8]*u_xy*w_x - 2*c1*dsg[8]*w_xy - c1*du_x*sg[6]*w_xx - c1*du_x*sg[7]*w_yy - 2*c1*du_x*sg[8]*w_xy + c1*du_xx*sg[6]*w_x + 2*c1*du_xy*sg[8]*w_x + c1*du_yy*sg[7]*w_x + c1*dw_x*sg[6]*u_xx + c1*dw_x*sg[7]*u_yy + 2*c1*dw_x*sg[8]*u_xy - c1*dw_xx*sg[6]*u_x - c1*dw_xx*sg[6] - 2*c1*dw_xy*sg[8]*u_x - 2*c1*dw_xy*sg[8] - c1*dw_yy*sg[7]*u_x - c1*dw_yy*sg[7] + dphiy*sg[9] + dphiy_x*sg[5] + dphiy_y*sg[4] + dsg[1]*v_y + dsg[1] + dsg[2]*v_x + dsg[4]*phiy_y + dsg[5]*phiy_x + dsg[9]*phiy + dv_x*sg[2] + dv_y*sg[1]
                S[4] = 3*c1*dphix*phix*sg[12] + c1*dphix*phix_x*sg[6] + c1*dphix*phix_y*sg[8] + c1*dphix_x*phix*sg[6] + c1*dphix_y*phix*sg[8] + 3*c1*dphiy*phiy*sg[12] + c1*dphiy*phiy_x*sg[6] + c1*dphiy*phiy_y*sg[8] + c1*dphiy_x*phiy*sg[6] + c1*dphiy_y*phiy*sg[8] + 1.5*c1*dsg[12]*(phix*phix) + 1.5*c1*dsg[12]*(phiy*phiy) - 3*c1*dsg[12] + c1*dsg[6]*phix*phix_x + c1*dsg[6]*phiy*phiy_x + c1*dsg[6]*u_xx*v_y + c1*dsg[6]*u_xx - c1*dsg[6]*u_y*v_xx - c1*dsg[7]*u_y*v_yy + c1*dsg[7]*u_yy*v_y + c1*dsg[7]*u_yy + c1*dsg[8]*phix*phix_y + c1*dsg[8]*phiy*phiy_y + 2*c1*dsg[8]*u_xy*v_y + 2*c1*dsg[8]*u_xy - 2*c1*dsg[8]*u_y*v_xy + c1*du_xx*sg[6]*v_y + c1*du_xx*sg[6] + 2*c1*du_xy*sg[8]*v_y + 2*c1*du_xy*sg[8] - c1*du_y*sg[6]*v_xx - c1*du_y*sg[7]*v_yy - 2*c1*du_y*sg[8]*v_xy + c1*du_yy*sg[7]*v_y + c1*du_yy*sg[7] - c1*dv_xx*sg[6]*u_y - 2*c1*dv_xy*sg[8]*u_y + c1*dv_y*sg[6]*u_xx + c1*dv_y*sg[7]*u_yy + 2*c1*dv_y*sg[8]*u_xy - c1*dv_yy*sg[7]*u_y - dphix*phix*sg[10] - dphix*phix_x*sg[3] - dphix*phix_y*sg[5] - dphix_x*phix*sg[3] - dphix_y*phix*sg[5] - dphiy*phiy*sg[10] - dphiy*phiy_x*sg[3] - dphiy*phiy_y*sg[5] - dphiy_x*phiy*sg[3] - dphiy_y*phiy*sg[5] + dsg[0]*w_x - 0.5*dsg[10]*(phix*phix) - 0.5*dsg[10]*(phiy*phiy) + dsg[10] + dsg[2]*w_y - dsg[3]*phix*phix_x - dsg[3]*phiy*phiy_x - dsg[5]*phix*phix_y - dsg[5]*phiy*phiy_y + dw_x*sg[0] + dw_y*sg[2]
                S[5] = 3*c1*dphix*phix*sg[11] + c1*dphix*phix_x*sg[8] + c1*dphix*phix_y*sg[7] + c1*dphix_x*phix*sg[8] + c1*dphix_y*phix*sg[7] + 3*c1*dphiy*phiy*sg[11] + c1*dphiy*phiy_x*sg[8] + c1*dphiy*phiy_y*sg[7] + c1*dphiy_x*phiy*sg[8] + c1*dphiy_y*phiy*sg[7] + 1.5*c1*dsg[11]*(phix*phix) + 1.5*c1*dsg[11]*(phiy*phiy) - 3*c1*dsg[11] + c1*dsg[6]*u_x*v_xx - c1*dsg[6]*u_xx*v_x + c1*dsg[6]*v_xx + c1*dsg[7]*phix*phix_y + c1*dsg[7]*phiy*phiy_y + c1*dsg[7]*u_x*v_yy - c1*dsg[7]*u_yy*v_x + c1*dsg[7]*v_yy + c1*dsg[8]*phix*phix_x + c1*dsg[8]*phiy*phiy_x + 2*c1*dsg[8]*u_x*v_xy - 2*c1*dsg[8]*u_xy*v_x + 2*c1*dsg[8]*v_xy + c1*du_x*sg[6]*v_xx + c1*du_x*sg[7]*v_yy + 2*c1*du_x*sg[8]*v_xy - c1*du_xx*sg[6]*v_x - 2*c1*du_xy*sg[8]*v_x - c1*du_yy*sg[7]*v_x - c1*dv_x*sg[6]*u_xx - c1*dv_x*sg[7]*u_yy - 2*c1*dv_x*sg[8]*u_xy + c1*dv_xx*sg[6]*u_x + c1*dv_xx*sg[6] + 2*c1*dv_xy*sg[8]*u_x + 2*c1*dv_xy*sg[8] + c1*dv_yy*sg[7]*u_x + c1*dv_yy*sg[7] - dphix*phix*sg[9] - dphix*phix_x*sg[5] - dphix*phix_y*sg[4] - dphix_x*phix*sg[5] - dphix_y*phix*sg[4] - dphiy*phiy*sg[9] - dphiy*phiy_x*sg[5] - dphiy*phiy_y*sg[4] - dphiy_x*phiy*sg[5] - dphiy_y*phiy*sg[4] + dsg[1]*w_y + dsg[2]*w_x - dsg[4]*phix*phix_y - dsg[4]*phiy*phiy_y - dsg[5]*phix*phix_x - dsg[5]*phiy*phiy_x - 0.5*dsg[9]*(phix*phix) - 0.5*dsg[9]*(phiy*phiy) + dsg[9] + dw_x*sg[2] + dw_y*sg[1]
                S[6] = 3*c1*dphix*sg[11]*w_y + 3*c1*dphix*sg[12]*w_x + c1*dphix_x*sg[6]*w_x + c1*dphix_x*sg[8]*w_y + c1*dphix_y*sg[7]*w_y + c1*dphix_y*sg[8]*w_x + 3*c1*dsg[11]*phix*w_y - 3*c1*dsg[11]*u_y + 3*c1*dsg[12]*phix*w_x - 3*c1*dsg[12]*u_x - 3*c1*dsg[12] + c1*dsg[6]*phix_x*w_x + c1*dsg[7]*phix_y*w_y + c1*dsg[8]*phix_x*w_y + c1*dsg[8]*phix_y*w_x - 3*c1*du_x*sg[12] - 3*c1*du_y*sg[11] + 3*c1*dw_x*phix*sg[12] + c1*dw_x*phix_x*sg[6] + c1*dw_x*phix_y*sg[8] + 3*c1*dw_y*phix*sg[11] + c1*dw_y*phix_x*sg[8] + c1*dw_y*phix_y*sg[7] - dphix*sg[10]*w_x - dphix*sg[9]*w_y - dphix_x*sg[3]*w_x - dphix_x*sg[5]*w_y - dphix_y*sg[4]*w_y - dphix_y*sg[5]*w_x - dsg[10]*phix*w_x + dsg[10]*u_x + dsg[10] - dsg[3]*phix_x*w_x - dsg[4]*phix_y*w_y - dsg[5]*phix_x*w_y - dsg[5]*phix_y*w_x - dsg[9]*phix*w_y + dsg[9]*u_y + du_x*sg[10] + du_y*sg[9] - dw_x*phix*sg[10] - dw_x*phix_x*sg[3] - dw_x*phix_y*sg[5] - dw_y*phix*sg[9] - dw_y*phix_x*sg[5] - dw_y*phix_y*sg[4]
                S[7] = 3*c1*dphiy*sg[11]*w_y + 3*c1*dphiy*sg[12]*w_x + c1*dphiy_x*sg[6]*w_x + c1*dphiy_x*sg[8]*w_y + c1*dphiy_y*sg[7]*w_y + c1*dphiy_y*sg[8]*w_x + 3*c1*dsg[11]*phiy*w_y - 3*c1*dsg[11]*v_y - 3*c1*dsg[11] + 3*c1*dsg[12]*phiy*w_x - 3*c1*dsg[12]*v_x + c1*dsg[6]*phiy_x*w_x + c1*dsg[7]*phiy_y*w_y + c1*dsg[8]*phiy_x*w_y + c1*dsg[8]*phiy_y*w_x - 3*c1*dv_x*sg[12] - 3*c1*dv_y*sg[11] + 3*c1*dw_x*phiy*sg[12] + c1*dw_x*phiy_x*sg[6] + c1*dw_x*phiy_y*sg[8] + 3*c1*dw_y*phiy*sg[11] + c1*dw_y*phiy_x*sg[8] + c1*dw_y*phiy_y*sg[7] - dphiy*sg[10]*w_x - dphiy*sg[9]*w_y - dphiy_x*sg[3]*w_x - dphiy_x*sg[5]*w_y - dphiy_y*sg[4]*w_y - dphiy_y*sg[5]*w_x - dsg[10]*phiy*w_x + dsg[10]*v_x - dsg[3]*phiy_x*w_x - dsg[4]*phiy_y*w_y - dsg[5]*phiy_x*w_y - dsg[5]*phiy_y*w_x - dsg[9]*phiy*w_y + dsg[9]*v_y + dsg[9] + dv_x*sg[10] + dv_y*sg[9] - dw_x*phiy*sg[10] - dw_x*phiy_x*sg[3] - dw_x*phiy_y*sg[5] - dw_y*phiy*sg[9] - dw_y*phiy_x*sg[5] - dw_y*phiy_y*sg[4]
                S[8] = c1*dphix*sg[6]*w_x + c1*dphix*sg[8]*w_y + c1*dsg[6]*phix*w_x - c1*dsg[6]*u_x - c1*dsg[6] + c1*dsg[8]*phix*w_y - c1*dsg[8]*u_y - c1*du_x*sg[6] - c1*du_y*sg[8] + c1*dw_x*phix*sg[6] + c1*dw_y*phix*sg[8] - dphix*sg[3]*w_x - dphix*sg[5]*w_y - dsg[3]*phix*w_x + dsg[3]*u_x + dsg[3] - dsg[5]*phix*w_y + dsg[5]*u_y + du_x*sg[3] + du_y*sg[5] - dw_x*phix*sg[3] - dw_y*phix*sg[5]
                S[9] = c1*dphix*sg[7]*w_y + c1*dphix*sg[8]*w_x + c1*dsg[7]*phix*w_y - c1*dsg[7]*u_y + c1*dsg[8]*phix*w_x - c1*dsg[8]*u_x - c1*dsg[8] - c1*du_x*sg[8] - c1*du_y*sg[7] + c1*dw_x*phix*sg[8] + c1*dw_y*phix*sg[7] - dphix*sg[4]*w_y - dphix*sg[5]*w_x - dsg[4]*phix*w_y + dsg[4]*u_y - dsg[5]*phix*w_x + dsg[5]*u_x + dsg[5] + du_x*sg[5] + du_y*sg[4] - dw_x*phix*sg[5] - dw_y*phix*sg[4]
                S[10] = c1*dphiy*sg[6]*w_x + c1*dphiy*sg[8]*w_y + c1*dsg[6]*phiy*w_x - c1*dsg[6]*v_x + c1*dsg[8]*phiy*w_y - c1*dsg[8]*v_y - c1*dsg[8] - c1*dv_x*sg[6] - c1*dv_y*sg[8] + c1*dw_x*phiy*sg[6] + c1*dw_y*phiy*sg[8] - dphiy*sg[3]*w_x - dphiy*sg[5]*w_y - dsg[3]*phiy*w_x + dsg[3]*v_x - dsg[5]*phiy*w_y + dsg[5]*v_y + dsg[5] + dv_x*sg[3] + dv_y*sg[5] - dw_x*phiy*sg[3] - dw_y*phiy*sg[5]
                S[11] = c1*dphiy*sg[7]*w_y + c1*dphiy*sg[8]*w_x + c1*dsg[7]*phiy*w_y - c1*dsg[7]*v_y - c1*dsg[7] + c1*dsg[8]*phiy*w_x - c1*dsg[8]*v_x - c1*dv_x*sg[8] - c1*dv_y*sg[7] + c1*dw_x*phiy*sg[8] + c1*dw_y*phiy*sg[7] - dphiy*sg[4]*w_y - dphiy*sg[5]*w_x - dsg[4]*phiy*w_y + dsg[4]*v_y + dsg[4] - dsg[5]*phiy*w_x + dsg[5]*v_x + dv_x*sg[5] + dv_y*sg[4] - dw_x*phiy*sg[5] - dw_y*phiy*sg[4]
                S[12] = -c1*dsg[6]*u_x*v_y - c1*dsg[6]*u_x + c1*dsg[6]*u_y*v_x - c1*dsg[6]*v_y - c1*dsg[6] - c1*du_x*sg[6]*v_y - c1*du_x*sg[6] + c1*du_y*sg[6]*v_x + c1*dv_x*sg[6]*u_y - c1*dv_y*sg[6]*u_x - c1*dv_y*sg[6]
                S[13] = -c1*dsg[7]*u_x*v_y - c1*dsg[7]*u_x + c1*dsg[7]*u_y*v_x - c1*dsg[7]*v_y - c1*dsg[7] - c1*du_x*sg[7]*v_y - c1*du_x*sg[7] + c1*du_y*sg[7]*v_x + c1*dv_x*sg[7]*u_y - c1*dv_y*sg[7]*u_x - c1*dv_y*sg[7]
                S[14] = -2*c1*dsg[8]*u_x*v_y - 2*c1*dsg[8]*u_x + 2*c1*dsg[8]*u_y*v_x - 2*c1*dsg[8]*v_y - 2*c1*dsg[8] - 2*c1*du_x*sg[8]*v_y - 2*c1*du_x*sg[8] + 2*c1*du_y*sg[8]*v_x + 2*c1*dv_x*sg[8]*u_y - 2*c1*dv_y*sg[8]*u_x - 2*c1*dv_y*sg[8]
                S[15] = -c1*dsg[6]*v_x*w_y + c1*dsg[6]*v_y*w_x + c1*dsg[6]*w_x - c1*dv_x*sg[6]*w_y + c1*dv_y*sg[6]*w_x + c1*dw_x*sg[6]*v_y + c1*dw_x*sg[6] - c1*dw_y*sg[6]*v_x
                S[16] = -c1*dsg[7]*v_x*w_y + c1*dsg[7]*v_y*w_x + c1*dsg[7]*w_x - c1*dv_x*sg[7]*w_y + c1*dv_y*sg[7]*w_x + c1*dw_x*sg[7]*v_y + c1*dw_x*sg[7] - c1*dw_y*sg[7]*v_x
                S[17] = -2*c1*dsg[8]*v_x*w_y + 2*c1*dsg[8]*v_y*w_x + 2*c1*dsg[8]*w_x - 2*c1*dv_x*sg[8]*w_y + 2*c1*dv_y*sg[8]*w_x + 2*c1*dw_x*sg[8]*v_y + 2*c1*dw_x*sg[8] - 2*c1*dw_y*sg[8]*v_x
                S[18] = c1*dsg[6]*u_x*w_y - c1*dsg[6]*u_y*w_x + c1*dsg[6]*w_y + c1*du_x*sg[6]*w_y - c1*du_y*sg[6]*w_x - c1*dw_x*sg[6]*u_y + c1*dw_y*sg[6]*u_x + c1*dw_y*sg[6]
                S[19] = c1*dsg[7]*u_x*w_y - c1*dsg[7]*u_y*w_x + c1*dsg[7]*w_y + c1*du_x*sg[7]*w_y - c1*du_y*sg[7]*w_x - c1*dw_x*sg[7]*u_y + c1*dw_y*sg[7]*u_x + c1*dw_y*sg[7]
                S[20] = 2*c1*dsg[8]*u_x*w_y - 2*c1*dsg[8]*u_y*w_x + 2*c1*dsg[8]*w_y + 2*c1*du_x*sg[8]*w_y - 2*c1*du_y*sg[8]*w_x - 2*c1*dw_x*sg[8]*u_y + 2*c1*dw_y*sg[8]*u_x + 2*c1*dw_y*sg[8]

                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAw_yy = gtab[6, j]
                    gAphix = gtab[7, j]
                    gAphix_y = gtab[8, j]
                    gAphiy = gtab[9, j]
                    gAphiy_y = gtab[10, j]
                    gAu_yy = gtab[11, j]
                    gAv_yy = gtab[12, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        fAphix = ftab[7, i]
                        fAphix_x = ftab[8, i]
                        fAphiy = ftab[9, i]
                        fAphiy_x = ftab[10, i]
                        fAu_xx = ftab[11, i]
                        fAv_xx = ftab[12, i]
                        col = col0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAphix*gAphix
                        pA[7] = fAphiy*gAphiy
                        pA[8] = fAphix_x*gAphix
                        pA[9] = fAphix*gAphix_y
                        pA[10] = fAphiy_x*gAphiy
                        pA[11] = fAphiy*gAphiy_y
                        pA[12] = fAw_xx*gAw
                        pA[13] = fAw*gAw_yy
                        pA[14] = fAw_x*gAw_y
                        pA[15] = fAu_xx*gAu
                        pA[16] = fAu*gAu_yy
                        pA[17] = fAu_x*gAu_y
                        pA[18] = fAv_xx*gAv
                        pA[19] = fAv*gAv_yy
                        pA[20] = fAv_x*gAv_y
                        kTv[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1] + pA[15]*S[15] + pA[16]*S[16] + pA[17]*S[17])
                        kTv[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[18]*S[18] + pA[19]*S[19] + pA[20]*S[20])
                        kTv[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[12]*S[12] + pA[13]*S[13] + pA[14]*S[14])
                        kTv[col+3] += weight*(pA[6]*S[6] + pA[8]*S[8] + pA[9]*S[9])
                        kTv[col+4] += weight*(pA[7]*S[7] + pA[10]*S[10] + pA[11]*S[11])

    return np.asarray(kTv)
