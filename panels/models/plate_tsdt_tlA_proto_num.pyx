#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Flat plates, prototype A of the total-Lagrangian third-order shear
deformation theory (TSDT)

Numerically integrated kernels of the model ``'plate_tsdt_tlA_proto'``: the
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
Prototype of the comparison of two total-Lagrangian TSDT. The first-order
part, the membrane strains `\varepsilon`, the curvatures `\kappa` and the
transverse shear strains `\gamma`, is that of
:mod:`panels.models.plate_fsdt_tl_num` (second-order director
`\mathbf{d}`, deformed mid-surface `\mathbf{X}`); the third-order strains
of Reddy [1]_ are kept linear, as in :mod:`panels.models.plate_tsdt_gl`:

.. math::

    \kappa^{(3)}_{xx} = -c_1 (\phi_{x,x} + w_{,xx}) \qquad
    \kappa^{(3)}_{yy} = -c_1 (\phi_{y,y} + w_{,yy}) \qquad
    \kappa^{(3)}_{xy} = -c_1 (\phi_{x,y} + \phi_{y,x} + 2 w_{,xy})

    \gamma^{(2)}_{\alpha} = -3 c_1 (\phi_\alpha + w_{,\alpha})

with `c_1 = 4/(3 h^2)`. The linear part is that of
:mod:`panels.models.plate_tsdt_donnell`.

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
    cdef double phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy
    cdef double e[13]
    cdef double eL[13]
    cdef double sg[13]
    cdef double sgL[13]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAw_xx
    cdef double fAphix, fAphix_x, fAphiy, fAphiy_x, gAu, gAu_y, gAv
    cdef double gAv_y, gAw, gAw_y, gAw_yy, gAphix, gAphix_y, gAphiy
    cdef double gAphiy_y, fBu, fBu_x, fBv, fBv_x, fBw, fBw_x
    cdef double fBw_xx, fBphix, fBphix_x, fBphiy, fBphiy_x, gBu, gBu_y
    cdef double gBv, gBv_y, gBw, gBw_y, gBw_yy, gBphix, gBphix_y
    cdef double gBphiy, gBphiy_y
    cdef double Bm[195]
    cdef double FB[195]
    cdef double K[225]
    cdef double TA[75]
    cdef double pA[15]
    cdef double pB[15]

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

    ftab = np.zeros((11, m), dtype=DOUBLE)
    gtab = np.zeros((11, n), dtype=DOUBLE)

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

                # [B] = d eps/d d at the point
                for p in range(195):
                    Bm[p] = 0.
                Bm[0] = u_x + 1
                Bm[2] = v_x
                Bm[4] = w_x
                Bm[16] = u_y
                Bm[18] = v_y + 1
                Bm[20] = w_y
                Bm[30] = u_y
                Bm[31] = u_x + 1
                Bm[32] = v_y + 1
                Bm[33] = v_x
                Bm[34] = w_y
                Bm[35] = w_x
                Bm[45] = phix_x
                Bm[47] = phiy_x
                Bm[49] = -phix*phix_x - phiy*phiy_x
                Bm[51] = -phix_x*w_x
                Bm[52] = -phiy_x*w_x
                Bm[53] = -phix*w_x + u_x + 1
                Bm[55] = -phiy*w_x + v_x
                Bm[61] = phix_y
                Bm[63] = phiy_y
                Bm[65] = -phix*phix_y - phiy*phiy_y
                Bm[66] = -phix_y*w_y
                Bm[67] = -phiy_y*w_y
                Bm[69] = -phix*w_y + u_y
                Bm[71] = -phiy*w_y + v_y + 1
                Bm[75] = phix_y
                Bm[76] = phix_x
                Bm[77] = phiy_y
                Bm[78] = phiy_x
                Bm[79] = -phix*phix_y - phiy*phiy_y
                Bm[80] = -phix*phix_x - phiy*phiy_x
                Bm[81] = -phix_x*w_y - phix_y*w_x
                Bm[82] = -phiy_x*w_y - phiy_y*w_x
                Bm[83] = -phix*w_y + u_y
                Bm[84] = -phix*w_x + u_x + 1
                Bm[85] = -phiy*w_y + v_y + 1
                Bm[86] = -phiy*w_x + v_x
                Bm[98] = -c1
                Bm[102] = -c1
                Bm[116] = -c1
                Bm[118] = -c1
                Bm[129] = -c1
                Bm[130] = -c1
                Bm[134] = -2*c1
                Bm[136] = phix
                Bm[138] = phiy
                Bm[140] = -0.5*(phix*phix) - 0.5*(phiy*phiy) + 1
                Bm[141] = -phix*w_y + u_y
                Bm[142] = -phiy*w_y + v_y + 1
                Bm[150] = phix
                Bm[152] = phiy
                Bm[154] = -0.5*(phix*phix) - 0.5*(phiy*phiy) + 1
                Bm[156] = -phix*w_x + u_x + 1
                Bm[157] = -phiy*w_x + v_x
                Bm[170] = -3*c1
                Bm[172] = -3*c1
                Bm[184] = -3*c1
                Bm[186] = -3*c1

                # K = [B]^T [F] [B]
                for r in range(NE):
                    for q in range(15):
                        FB[r*15 + q] = 0.
                        for s in range(NE):
                            FB[r*15 + q] += F[r*NE + s]*Bm[s*15 + q]
                for p in range(15):
                    for q in range(15):
                        K[p*15 + q] = 0.
                        for r in range(NE):
                            K[p*15 + q] += Bm[r*15 + p]*FB[r*15 + q]

                # stresses of the strains and of the linear strains
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x + phiy_x*v_x
                e[4] = -phix*phix_y*w_y + phix_y*u_y - phiy*phiy_y*w_y + phiy_y*v_y + phiy_y
                e[5] = -phix*phix_x*w_y - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - phiy*phiy_x*w_y - phiy*phiy_y*w_x + phiy_x*v_y + phiy_x + phiy_y*v_x
                e[6] = -c1*phix_x - c1*w_xx
                e[7] = -c1*phiy_y - c1*w_yy
                e[8] = -c1*phix_y - c1*phiy_x - 2*c1*w_xy
                e[9] = -0.5*(phix*phix)*w_y + phix*u_y - 0.5*(phiy*phiy)*w_y + phiy*v_y + phiy + w_y
                e[10] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x + w_x
                e[11] = -3*c1*phiy - 3*c1*w_y
                e[12] = -3*c1*phix - 3*c1*w_x
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
                K[6] += -sgL[10] + sg[10]
                K[8] += -sgL[3] + sg[3]
                K[9] += -sgL[5] + sg[5]
                K[15] += -sgL[2] + sg[2]
                K[16] += -sgL[1] + sg[1]
                K[21] += -sgL[9] + sg[9]
                K[23] += -sgL[5] + sg[5]
                K[24] += -sgL[4] + sg[4]
                K[32] += -sgL[0] + sg[0]
                K[33] += -sgL[2] + sg[2]
                K[37] += -sgL[10] + sg[10]
                K[40] += -sgL[3] + sg[3]
                K[41] += -sgL[5] + sg[5]
                K[47] += -sgL[2] + sg[2]
                K[48] += -sgL[1] + sg[1]
                K[52] += -sgL[9] + sg[9]
                K[55] += -sgL[5] + sg[5]
                K[56] += -sgL[4] + sg[4]
                K[64] += -sgL[0] + sg[0]
                K[65] += -sgL[2] + sg[2]
                K[66] += -phix*sg[10] - phix_x*sg[3] - phix_y*sg[5]
                K[67] += -phiy*sg[10] - phiy_x*sg[3] - phiy_y*sg[5]
                K[68] += -phix*sg[3]
                K[69] += -phix*sg[5]
                K[70] += -phiy*sg[3]
                K[71] += -phiy*sg[5]
                K[79] += -sgL[2] + sg[2]
                K[80] += -sgL[1] + sg[1]
                K[81] += -phix*sg[9] - phix_x*sg[5] - phix_y*sg[4]
                K[82] += -phiy*sg[9] - phiy_x*sg[5] - phiy_y*sg[4]
                K[83] += -phix*sg[5]
                K[84] += -phix*sg[4]
                K[85] += -phiy*sg[5]
                K[86] += -phiy*sg[4]
                K[90] += -sgL[10] + sg[10]
                K[91] += -sgL[9] + sg[9]
                K[94] += -phix*sg[10] - phix_x*sg[3] - phix_y*sg[5]
                K[95] += -phix*sg[9] - phix_x*sg[5] - phix_y*sg[4]
                K[96] += -sg[10]*w_x - sg[9]*w_y
                K[98] += -sg[3]*w_x - sg[5]*w_y
                K[99] += -sg[4]*w_y - sg[5]*w_x
                K[107] += -sgL[10] + sg[10]
                K[108] += -sgL[9] + sg[9]
                K[109] += -phiy*sg[10] - phiy_x*sg[3] - phiy_y*sg[5]
                K[110] += -phiy*sg[9] - phiy_x*sg[5] - phiy_y*sg[4]
                K[112] += -sg[10]*w_x - sg[9]*w_y
                K[115] += -sg[3]*w_x - sg[5]*w_y
                K[116] += -sg[4]*w_y - sg[5]*w_x
                K[120] += -sgL[3] + sg[3]
                K[121] += -sgL[5] + sg[5]
                K[124] += -phix*sg[3]
                K[125] += -phix*sg[5]
                K[126] += -sg[3]*w_x - sg[5]*w_y
                K[135] += -sgL[5] + sg[5]
                K[136] += -sgL[4] + sg[4]
                K[139] += -phix*sg[5]
                K[140] += -phix*sg[4]
                K[141] += -sg[4]*w_y - sg[5]*w_x
                K[152] += -sgL[3] + sg[3]
                K[153] += -sgL[5] + sg[5]
                K[154] += -phiy*sg[3]
                K[155] += -phiy*sg[5]
                K[157] += -sg[3]*w_x - sg[5]*w_y
                K[167] += -sgL[5] + sg[5]
                K[168] += -sgL[4] + sg[4]
                K[169] += -phiy*sg[5]
                K[170] += -phiy*sg[4]
                K[172] += -sg[4]*w_y - sg[5]*w_x

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
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[15]
                        TA[1] = pA[0]*K[1] + pA[1]*K[16]
                        TA[2] = pA[0]*K[2] + pA[1]*K[17]
                        TA[3] = pA[0]*K[3] + pA[1]*K[18]
                        TA[4] = pA[0]*K[4] + pA[1]*K[19]
                        TA[5] = pA[0]*K[5] + pA[1]*K[20]
                        TA[6] = pA[0]*K[6] + pA[1]*K[21]
                        TA[7] = pA[0]*K[7] + pA[1]*K[22]
                        TA[8] = pA[0]*K[8] + pA[1]*K[23]
                        TA[9] = pA[0]*K[9] + pA[1]*K[24]
                        TA[10] = pA[0]*K[10] + pA[1]*K[25]
                        TA[11] = pA[0]*K[11] + pA[1]*K[26]
                        TA[12] = pA[0]*K[12] + pA[1]*K[27]
                        TA[13] = pA[0]*K[13] + pA[1]*K[28]
                        TA[14] = pA[0]*K[14] + pA[1]*K[29]
                        TA[15] = pA[2]*K[30] + pA[3]*K[45]
                        TA[16] = pA[2]*K[31] + pA[3]*K[46]
                        TA[17] = pA[2]*K[32] + pA[3]*K[47]
                        TA[18] = pA[2]*K[33] + pA[3]*K[48]
                        TA[19] = pA[2]*K[34] + pA[3]*K[49]
                        TA[20] = pA[2]*K[35] + pA[3]*K[50]
                        TA[21] = pA[2]*K[36] + pA[3]*K[51]
                        TA[22] = pA[2]*K[37] + pA[3]*K[52]
                        TA[23] = pA[2]*K[38] + pA[3]*K[53]
                        TA[24] = pA[2]*K[39] + pA[3]*K[54]
                        TA[25] = pA[2]*K[40] + pA[3]*K[55]
                        TA[26] = pA[2]*K[41] + pA[3]*K[56]
                        TA[27] = pA[2]*K[42] + pA[3]*K[57]
                        TA[28] = pA[2]*K[43] + pA[3]*K[58]
                        TA[29] = pA[2]*K[44] + pA[3]*K[59]
                        TA[30] = pA[4]*K[60] + pA[5]*K[75] + pA[12]*K[180] + pA[13]*K[195] + pA[14]*K[210]
                        TA[31] = pA[4]*K[61] + pA[5]*K[76] + pA[12]*K[181] + pA[13]*K[196] + pA[14]*K[211]
                        TA[32] = pA[4]*K[62] + pA[5]*K[77] + pA[12]*K[182] + pA[13]*K[197] + pA[14]*K[212]
                        TA[33] = pA[4]*K[63] + pA[5]*K[78] + pA[12]*K[183] + pA[13]*K[198] + pA[14]*K[213]
                        TA[34] = pA[4]*K[64] + pA[5]*K[79] + pA[12]*K[184] + pA[13]*K[199] + pA[14]*K[214]
                        TA[35] = pA[4]*K[65] + pA[5]*K[80] + pA[12]*K[185] + pA[13]*K[200] + pA[14]*K[215]
                        TA[36] = pA[4]*K[66] + pA[5]*K[81] + pA[12]*K[186] + pA[13]*K[201] + pA[14]*K[216]
                        TA[37] = pA[4]*K[67] + pA[5]*K[82] + pA[12]*K[187] + pA[13]*K[202] + pA[14]*K[217]
                        TA[38] = pA[4]*K[68] + pA[5]*K[83] + pA[12]*K[188] + pA[13]*K[203] + pA[14]*K[218]
                        TA[39] = pA[4]*K[69] + pA[5]*K[84] + pA[12]*K[189] + pA[13]*K[204] + pA[14]*K[219]
                        TA[40] = pA[4]*K[70] + pA[5]*K[85] + pA[12]*K[190] + pA[13]*K[205] + pA[14]*K[220]
                        TA[41] = pA[4]*K[71] + pA[5]*K[86] + pA[12]*K[191] + pA[13]*K[206] + pA[14]*K[221]
                        TA[42] = pA[4]*K[72] + pA[5]*K[87] + pA[12]*K[192] + pA[13]*K[207] + pA[14]*K[222]
                        TA[43] = pA[4]*K[73] + pA[5]*K[88] + pA[12]*K[193] + pA[13]*K[208] + pA[14]*K[223]
                        TA[44] = pA[4]*K[74] + pA[5]*K[89] + pA[12]*K[194] + pA[13]*K[209] + pA[14]*K[224]
                        TA[45] = pA[6]*K[90] + pA[8]*K[120] + pA[9]*K[135]
                        TA[46] = pA[6]*K[91] + pA[8]*K[121] + pA[9]*K[136]
                        TA[47] = pA[6]*K[92] + pA[8]*K[122] + pA[9]*K[137]
                        TA[48] = pA[6]*K[93] + pA[8]*K[123] + pA[9]*K[138]
                        TA[49] = pA[6]*K[94] + pA[8]*K[124] + pA[9]*K[139]
                        TA[50] = pA[6]*K[95] + pA[8]*K[125] + pA[9]*K[140]
                        TA[51] = pA[6]*K[96] + pA[8]*K[126] + pA[9]*K[141]
                        TA[52] = pA[6]*K[97] + pA[8]*K[127] + pA[9]*K[142]
                        TA[53] = pA[6]*K[98] + pA[8]*K[128] + pA[9]*K[143]
                        TA[54] = pA[6]*K[99] + pA[8]*K[129] + pA[9]*K[144]
                        TA[55] = pA[6]*K[100] + pA[8]*K[130] + pA[9]*K[145]
                        TA[56] = pA[6]*K[101] + pA[8]*K[131] + pA[9]*K[146]
                        TA[57] = pA[6]*K[102] + pA[8]*K[132] + pA[9]*K[147]
                        TA[58] = pA[6]*K[103] + pA[8]*K[133] + pA[9]*K[148]
                        TA[59] = pA[6]*K[104] + pA[8]*K[134] + pA[9]*K[149]
                        TA[60] = pA[7]*K[105] + pA[10]*K[150] + pA[11]*K[165]
                        TA[61] = pA[7]*K[106] + pA[10]*K[151] + pA[11]*K[166]
                        TA[62] = pA[7]*K[107] + pA[10]*K[152] + pA[11]*K[167]
                        TA[63] = pA[7]*K[108] + pA[10]*K[153] + pA[11]*K[168]
                        TA[64] = pA[7]*K[109] + pA[10]*K[154] + pA[11]*K[169]
                        TA[65] = pA[7]*K[110] + pA[10]*K[155] + pA[11]*K[170]
                        TA[66] = pA[7]*K[111] + pA[10]*K[156] + pA[11]*K[171]
                        TA[67] = pA[7]*K[112] + pA[10]*K[157] + pA[11]*K[172]
                        TA[68] = pA[7]*K[113] + pA[10]*K[158] + pA[11]*K[173]
                        TA[69] = pA[7]*K[114] + pA[10]*K[159] + pA[11]*K[174]
                        TA[70] = pA[7]*K[115] + pA[10]*K[160] + pA[11]*K[175]
                        TA[71] = pA[7]*K[116] + pA[10]*K[161] + pA[11]*K[176]
                        TA[72] = pA[7]*K[117] + pA[10]*K[162] + pA[11]*K[177]
                        TA[73] = pA[7]*K[118] + pA[10]*K[163] + pA[11]*K[178]
                        TA[74] = pA[7]*K[119] + pA[10]*K[164] + pA[11]*K[179]
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
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[2]*pB[2] + TA[3]*pB[3])
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
                                kCv[c] += weight*(TA[15]*pB[0] + TA[16]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[17]*pB[2] + TA[18]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[19]*pB[4] + TA[20]*pB[5] + TA[27]*pB[12] + TA[28]*pB[13] + TA[29]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[21]*pB[6] + TA[23]*pB[8] + TA[24]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[22]*pB[7] + TA[25]*pB[10] + TA[26]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[30]*pB[0] + TA[31]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[32]*pB[2] + TA[33]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[34]*pB[4] + TA[35]*pB[5] + TA[42]*pB[12] + TA[43]*pB[13] + TA[44]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[36]*pB[6] + TA[38]*pB[8] + TA[39]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[37]*pB[7] + TA[40]*pB[10] + TA[41]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[45]*pB[0] + TA[46]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[47]*pB[2] + TA[48]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[49]*pB[4] + TA[50]*pB[5] + TA[57]*pB[12] + TA[58]*pB[13] + TA[59]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[51]*pB[6] + TA[53]*pB[8] + TA[54]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[52]*pB[7] + TA[55]*pB[10] + TA[56]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[60]*pB[0] + TA[61]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[62]*pB[2] + TA[63]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[64]*pB[4] + TA[65]*pB[5] + TA[72]*pB[12] + TA[73]*pB[13] + TA[74]*pB[14])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[66]*pB[6] + TA[68]*pB[8] + TA[69]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[67]*pB[7] + TA[70]*pB[10] + TA[71]*pB[11])

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
    cdef double fAphix, fAphix_x, fAphiy, fAphiy_x, gAu, gAu_y, gAv
    cdef double gAv_y, gAw, gAw_y, gAw_yy, gAphix, gAphix_y, gAphiy
    cdef double gAphiy_y, fBu, fBu_x, fBv, fBv_x, fBw, fBw_x
    cdef double fBw_xx, fBphix, fBphix_x, fBphiy, fBphiy_x, gBu, gBu_y
    cdef double gBv, gBv_y, gBw, gBw_y, gBw_yy, gBphix, gBphix_y
    cdef double gBphiy, gBphiy_y
    cdef double K[225]
    cdef double TA[75]
    cdef double pA[15]
    cdef double pB[15]

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

    ftab = np.zeros((11, m), dtype=DOUBLE)
    gtab = np.zeros((11, n), dtype=DOUBLE)

    fdim = 7*m*m*n*n
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
                for p in range(225):
                    K[p] = 0.
                K[0] = sg[0]
                K[1] = sg[2]
                K[6] = sg[10]
                K[8] = sg[3]
                K[9] = sg[5]
                K[15] = sg[2]
                K[16] = sg[1]
                K[21] = sg[9]
                K[23] = sg[5]
                K[24] = sg[4]
                K[32] = sg[0]
                K[33] = sg[2]
                K[37] = sg[10]
                K[40] = sg[3]
                K[41] = sg[5]
                K[47] = sg[2]
                K[48] = sg[1]
                K[52] = sg[9]
                K[55] = sg[5]
                K[56] = sg[4]
                K[64] = sg[0]
                K[65] = sg[2]
                K[79] = sg[2]
                K[80] = sg[1]
                K[90] = sg[10]
                K[91] = sg[9]
                K[107] = sg[10]
                K[108] = sg[9]
                K[120] = sg[3]
                K[121] = sg[5]
                K[135] = sg[5]
                K[136] = sg[4]
                K[152] = sg[3]
                K[153] = sg[5]
                K[167] = sg[5]
                K[168] = sg[4]

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
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[15]
                        TA[1] = pA[0]*K[1] + pA[1]*K[16]
                        TA[6] = pA[0]*K[6] + pA[1]*K[21]
                        TA[8] = pA[0]*K[8] + pA[1]*K[23]
                        TA[9] = pA[0]*K[9] + pA[1]*K[24]
                        TA[17] = pA[2]*K[32] + pA[3]*K[47]
                        TA[18] = pA[2]*K[33] + pA[3]*K[48]
                        TA[22] = pA[2]*K[37] + pA[3]*K[52]
                        TA[25] = pA[2]*K[40] + pA[3]*K[55]
                        TA[26] = pA[2]*K[41] + pA[3]*K[56]
                        TA[34] = pA[4]*K[64] + pA[5]*K[79]
                        TA[35] = pA[4]*K[65] + pA[5]*K[80]
                        TA[45] = pA[6]*K[90] + pA[8]*K[120] + pA[9]*K[135]
                        TA[46] = pA[6]*K[91] + pA[8]*K[121] + pA[9]*K[136]
                        TA[62] = pA[7]*K[107] + pA[10]*K[152] + pA[11]*K[167]
                        TA[63] = pA[7]*K[108] + pA[10]*K[153] + pA[11]*K[168]
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
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+3
                                kGv[c] += weight*(TA[6]*pB[6] + TA[8]*pB[8] + TA[9]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[17]*pB[2] + TA[18]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+4
                                kGv[c] += weight*(TA[22]*pB[7] + TA[25]*pB[10] + TA[26]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[34]*pB[4] + TA[35]*pB[5])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+3
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[45]*pB[0] + TA[46]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+4
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[62]*pB[2] + TA[63]*pB[3])

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
    cdef double phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy
    cdef double e[13]
    cdef double sg[13]
    cdef double S[15]
    cdef double pA[15]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAw_xx
    cdef double fAphix, fAphix_x, fAphiy, fAphiy_x, gAu, gAu_y, gAv
    cdef double gAv_y, gAw, gAw_y, gAw_yy, gAphix, gAphix_y, gAphiy
    cdef double gAphiy_y

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

    ftab = np.zeros((11, m), dtype=DOUBLE)
    gtab = np.zeros((11, n), dtype=DOUBLE)

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

                # strains, stresses and S = [B]^T sigma
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x + phiy_x*v_x
                e[4] = -phix*phix_y*w_y + phix_y*u_y - phiy*phiy_y*w_y + phiy_y*v_y + phiy_y
                e[5] = -phix*phix_x*w_y - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - phiy*phiy_x*w_y - phiy*phiy_y*w_x + phiy_x*v_y + phiy_x + phiy_y*v_x
                e[6] = -c1*phix_x - c1*w_xx
                e[7] = -c1*phiy_y - c1*w_yy
                e[8] = -c1*phix_y - c1*phiy_x - 2*c1*w_xy
                e[9] = -0.5*(phix*phix)*w_y + phix*u_y - 0.5*(phiy*phiy)*w_y + phiy*v_y + phiy + w_y
                e[10] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x + w_x
                e[11] = -3*c1*phiy - 3*c1*w_y
                e[12] = -3*c1*phix - 3*c1*w_x
                for r in range(NE):
                    sg[r] = 0.
                    for s in range(NE):
                        sg[r] += F[r*NE + s]*e[s]
                S[0] = phix*sg[10] + phix_x*sg[3] + phix_y*sg[5] + sg[0]*u_x + sg[0] + sg[2]*u_y
                S[1] = phix*sg[9] + phix_x*sg[5] + phix_y*sg[4] + sg[1]*u_y + sg[2]*u_x + sg[2]
                S[2] = phiy*sg[10] + phiy_x*sg[3] + phiy_y*sg[5] + sg[0]*v_x + sg[2]*v_y + sg[2]
                S[3] = phiy*sg[9] + phiy_x*sg[5] + phiy_y*sg[4] + sg[1]*v_y + sg[1] + sg[2]*v_x
                S[4] = -3*c1*sg[12] - 0.5*(phix*phix)*sg[10] - phix*phix_x*sg[3] - phix*phix_y*sg[5] - 0.5*(phiy*phiy)*sg[10] - phiy*phiy_x*sg[3] - phiy*phiy_y*sg[5] + sg[0]*w_x + sg[10] + sg[2]*w_y
                S[5] = -3*c1*sg[11] - 0.5*(phix*phix)*sg[9] - phix*phix_x*sg[5] - phix*phix_y*sg[4] - 0.5*(phiy*phiy)*sg[9] - phiy*phiy_x*sg[5] - phiy*phiy_y*sg[4] + sg[1]*w_y + sg[2]*w_x + sg[9]
                S[6] = -3*c1*sg[12] - phix*sg[10]*w_x - phix*sg[9]*w_y - phix_x*sg[3]*w_x - phix_x*sg[5]*w_y - phix_y*sg[4]*w_y - phix_y*sg[5]*w_x + sg[10]*u_x + sg[10] + sg[9]*u_y
                S[7] = -3*c1*sg[11] - phiy*sg[10]*w_x - phiy*sg[9]*w_y - phiy_x*sg[3]*w_x - phiy_x*sg[5]*w_y - phiy_y*sg[4]*w_y - phiy_y*sg[5]*w_x + sg[10]*v_x + sg[9]*v_y + sg[9]
                S[8] = -c1*sg[6] - phix*sg[3]*w_x - phix*sg[5]*w_y + sg[3]*u_x + sg[3] + sg[5]*u_y
                S[9] = -c1*sg[8] - phix*sg[4]*w_y - phix*sg[5]*w_x + sg[4]*u_y + sg[5]*u_x + sg[5]
                S[10] = -c1*sg[8] - phiy*sg[3]*w_x - phiy*sg[5]*w_y + sg[3]*v_x + sg[5]*v_y + sg[5]
                S[11] = -c1*sg[7] - phiy*sg[4]*w_y - phiy*sg[5]*w_x + sg[4]*v_y + sg[4] + sg[5]*v_x
                S[12] = -c1*sg[6]
                S[13] = -c1*sg[7]
                S[14] = -2*c1*sg[8]

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
                        fint[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        fint[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3])
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
    cdef double phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy
    cdef double du_x, du_y, dv_x, dv_y, dw_x, dw_y, dphix, dphiy
    cdef double dphix_x, dphix_y, dphiy_x, dphiy_y, dw_xx, dw_yy, dw_xy
    cdef double e[13]
    cdef double de[13]
    cdef double sg[13]
    cdef double dsg[13]
    cdef double S[15]
    cdef double pA[15]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAw_xx
    cdef double fAphix, fAphix_x, fAphiy, fAphiy_x, gAu, gAu_y, gAv
    cdef double gAv_y, gAw, gAw_y, gAw_yy, gAphix, gAphix_y, gAphiy
    cdef double gAphiy_y

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

    ftab = np.zeros((11, m), dtype=DOUBLE)
    gtab = np.zeros((11, n), dtype=DOUBLE)

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

                # strains, strain increments, stresses
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x + phiy_x*v_x
                e[4] = -phix*phix_y*w_y + phix_y*u_y - phiy*phiy_y*w_y + phiy_y*v_y + phiy_y
                e[5] = -phix*phix_x*w_y - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - phiy*phiy_x*w_y - phiy*phiy_y*w_x + phiy_x*v_y + phiy_x + phiy_y*v_x
                e[6] = -c1*phix_x - c1*w_xx
                e[7] = -c1*phiy_y - c1*w_yy
                e[8] = -c1*phix_y - c1*phiy_x - 2*c1*w_xy
                e[9] = -0.5*(phix*phix)*w_y + phix*u_y - 0.5*(phiy*phiy)*w_y + phiy*v_y + phiy + w_y
                e[10] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x + w_x
                e[11] = -3*c1*phiy - 3*c1*w_y
                e[12] = -3*c1*phix - 3*c1*w_x
                de[0] = du_x*u_x + du_x + dv_x*v_x + dw_x*w_x
                de[1] = du_y*u_y + dv_y*v_y + dv_y + dw_y*w_y
                de[2] = du_x*u_y + du_y*u_x + du_y + dv_x*v_y + dv_x + dv_y*v_x + dw_x*w_y + dw_y*w_x
                de[3] = -dphix*phix_x*w_x - dphix_x*phix*w_x + dphix_x*u_x + dphix_x - dphiy*phiy_x*w_x - dphiy_x*phiy*w_x + dphiy_x*v_x + du_x*phix_x + dv_x*phiy_x - dw_x*phix*phix_x - dw_x*phiy*phiy_x
                de[4] = -dphix*phix_y*w_y - dphix_y*phix*w_y + dphix_y*u_y - dphiy*phiy_y*w_y - dphiy_y*phiy*w_y + dphiy_y*v_y + dphiy_y + du_y*phix_y + dv_y*phiy_y - dw_y*phix*phix_y - dw_y*phiy*phiy_y
                de[5] = -dphix*phix_x*w_y - dphix*phix_y*w_x - dphix_x*phix*w_y + dphix_x*u_y - dphix_y*phix*w_x + dphix_y*u_x + dphix_y - dphiy*phiy_x*w_y - dphiy*phiy_y*w_x - dphiy_x*phiy*w_y + dphiy_x*v_y + dphiy_x - dphiy_y*phiy*w_x + dphiy_y*v_x + du_x*phix_y + du_y*phix_x + dv_x*phiy_y + dv_y*phiy_x - dw_x*phix*phix_y - dw_x*phiy*phiy_y - dw_y*phix*phix_x - dw_y*phiy*phiy_x
                de[6] = -c1*dphix_x - c1*dw_xx
                de[7] = -c1*dphiy_y - c1*dw_yy
                de[8] = -c1*dphix_y - c1*dphiy_x - 2*c1*dw_xy
                de[9] = -dphix*phix*w_y + dphix*u_y - dphiy*phiy*w_y + dphiy*v_y + dphiy + du_y*phix + dv_y*phiy - 0.5*dw_y*(phix*phix) - 0.5*dw_y*(phiy*phiy) + dw_y
                de[10] = -dphix*phix*w_x + dphix*u_x + dphix - dphiy*phiy*w_x + dphiy*v_x + du_x*phix + dv_x*phiy - 0.5*dw_x*(phix*phix) - 0.5*dw_x*(phiy*phiy) + dw_x
                de[11] = -3*c1*dphiy - 3*c1*dw_y
                de[12] = -3*c1*dphix - 3*c1*dw_x
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
                S[0] = dphix*sg[10] + dphix_x*sg[3] + dphix_y*sg[5] + dsg[0]*u_x + dsg[0] + dsg[10]*phix + dsg[2]*u_y + dsg[3]*phix_x + dsg[5]*phix_y + du_x*sg[0] + du_y*sg[2]
                S[1] = dphix*sg[9] + dphix_x*sg[5] + dphix_y*sg[4] + dsg[1]*u_y + dsg[2]*u_x + dsg[2] + dsg[4]*phix_y + dsg[5]*phix_x + dsg[9]*phix + du_x*sg[2] + du_y*sg[1]
                S[2] = dphiy*sg[10] + dphiy_x*sg[3] + dphiy_y*sg[5] + dsg[0]*v_x + dsg[10]*phiy + dsg[2]*v_y + dsg[2] + dsg[3]*phiy_x + dsg[5]*phiy_y + dv_x*sg[0] + dv_y*sg[2]
                S[3] = dphiy*sg[9] + dphiy_x*sg[5] + dphiy_y*sg[4] + dsg[1]*v_y + dsg[1] + dsg[2]*v_x + dsg[4]*phiy_y + dsg[5]*phiy_x + dsg[9]*phiy + dv_x*sg[2] + dv_y*sg[1]
                S[4] = -3*c1*dsg[12] - dphix*phix*sg[10] - dphix*phix_x*sg[3] - dphix*phix_y*sg[5] - dphix_x*phix*sg[3] - dphix_y*phix*sg[5] - dphiy*phiy*sg[10] - dphiy*phiy_x*sg[3] - dphiy*phiy_y*sg[5] - dphiy_x*phiy*sg[3] - dphiy_y*phiy*sg[5] + dsg[0]*w_x - 0.5*dsg[10]*(phix*phix) - 0.5*dsg[10]*(phiy*phiy) + dsg[10] + dsg[2]*w_y - dsg[3]*phix*phix_x - dsg[3]*phiy*phiy_x - dsg[5]*phix*phix_y - dsg[5]*phiy*phiy_y + dw_x*sg[0] + dw_y*sg[2]
                S[5] = -3*c1*dsg[11] - dphix*phix*sg[9] - dphix*phix_x*sg[5] - dphix*phix_y*sg[4] - dphix_x*phix*sg[5] - dphix_y*phix*sg[4] - dphiy*phiy*sg[9] - dphiy*phiy_x*sg[5] - dphiy*phiy_y*sg[4] - dphiy_x*phiy*sg[5] - dphiy_y*phiy*sg[4] + dsg[1]*w_y + dsg[2]*w_x - dsg[4]*phix*phix_y - dsg[4]*phiy*phiy_y - dsg[5]*phix*phix_x - dsg[5]*phiy*phiy_x - 0.5*dsg[9]*(phix*phix) - 0.5*dsg[9]*(phiy*phiy) + dsg[9] + dw_x*sg[2] + dw_y*sg[1]
                S[6] = -3*c1*dsg[12] - dphix*sg[10]*w_x - dphix*sg[9]*w_y - dphix_x*sg[3]*w_x - dphix_x*sg[5]*w_y - dphix_y*sg[4]*w_y - dphix_y*sg[5]*w_x - dsg[10]*phix*w_x + dsg[10]*u_x + dsg[10] - dsg[3]*phix_x*w_x - dsg[4]*phix_y*w_y - dsg[5]*phix_x*w_y - dsg[5]*phix_y*w_x - dsg[9]*phix*w_y + dsg[9]*u_y + du_x*sg[10] + du_y*sg[9] - dw_x*phix*sg[10] - dw_x*phix_x*sg[3] - dw_x*phix_y*sg[5] - dw_y*phix*sg[9] - dw_y*phix_x*sg[5] - dw_y*phix_y*sg[4]
                S[7] = -3*c1*dsg[11] - dphiy*sg[10]*w_x - dphiy*sg[9]*w_y - dphiy_x*sg[3]*w_x - dphiy_x*sg[5]*w_y - dphiy_y*sg[4]*w_y - dphiy_y*sg[5]*w_x - dsg[10]*phiy*w_x + dsg[10]*v_x - dsg[3]*phiy_x*w_x - dsg[4]*phiy_y*w_y - dsg[5]*phiy_x*w_y - dsg[5]*phiy_y*w_x - dsg[9]*phiy*w_y + dsg[9]*v_y + dsg[9] + dv_x*sg[10] + dv_y*sg[9] - dw_x*phiy*sg[10] - dw_x*phiy_x*sg[3] - dw_x*phiy_y*sg[5] - dw_y*phiy*sg[9] - dw_y*phiy_x*sg[5] - dw_y*phiy_y*sg[4]
                S[8] = -c1*dsg[6] - dphix*sg[3]*w_x - dphix*sg[5]*w_y - dsg[3]*phix*w_x + dsg[3]*u_x + dsg[3] - dsg[5]*phix*w_y + dsg[5]*u_y + du_x*sg[3] + du_y*sg[5] - dw_x*phix*sg[3] - dw_y*phix*sg[5]
                S[9] = -c1*dsg[8] - dphix*sg[4]*w_y - dphix*sg[5]*w_x - dsg[4]*phix*w_y + dsg[4]*u_y - dsg[5]*phix*w_x + dsg[5]*u_x + dsg[5] + du_x*sg[5] + du_y*sg[4] - dw_x*phix*sg[5] - dw_y*phix*sg[4]
                S[10] = -c1*dsg[8] - dphiy*sg[3]*w_x - dphiy*sg[5]*w_y - dsg[3]*phiy*w_x + dsg[3]*v_x - dsg[5]*phiy*w_y + dsg[5]*v_y + dsg[5] + dv_x*sg[3] + dv_y*sg[5] - dw_x*phiy*sg[3] - dw_y*phiy*sg[5]
                S[11] = -c1*dsg[7] - dphiy*sg[4]*w_y - dphiy*sg[5]*w_x - dsg[4]*phiy*w_y + dsg[4]*v_y + dsg[4] - dsg[5]*phiy*w_x + dsg[5]*v_x + dv_x*sg[5] + dv_y*sg[4] - dw_x*phiy*sg[5] - dw_y*phiy*sg[4]
                S[12] = -c1*dsg[6]
                S[13] = -c1*dsg[7]
                S[14] = -2*c1*dsg[8]

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
                        kTv[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        kTv[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3])
                        kTv[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[12]*S[12] + pA[13]*S[13] + pA[14]*S[14])
                        kTv[col+3] += weight*(pA[6]*S[6] + pA[8]*S[8] + pA[9]*S[9])
                        kTv[col+4] += weight*(pA[7]*S[7] + pA[10]*S[10] + pA[11]*S[11])

    return np.asarray(kTv)
