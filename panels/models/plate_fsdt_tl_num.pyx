#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Flat plates using the total-Lagrangian first-order shear deformation theory
(FSDT) with a second-order director

Numerically integrated kernels of the model ``'plate_fsdt_tl'``: the
constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the internal force vector :func:`calc_fint`, the product
of the tangent stiffness matrix by a vector :func:`calc_kTv`, and the force
vector and load stiffness matrix of a follower pressure,
:func:`calc_fext_follower` and :func:`fkCfollower_num`. The mass and
aerodynamic matrices, ``fkM_num``, ``fkAx_num`` and ``fkAy_num``, are those
of :mod:`panels.models.plate_fsdt_donnell_num`. The constitutive matrix ``Finput`` is the
``8 x 8`` matrix ``ABD`` of :class:`.Shell` for this model.

The degrees of freedom of each term of the approximation are
``u, v, w, phix, phiy``; the degree of freedom ``k`` of the term ``(i, j)``,
``0 <= i < m``, ``0 <= j < n``, is at the row ``row0 + 5*(j*m + i) + k``
of the matrices.

Notes
-----
The displacement field is that of the moderate rotation theory of Schmidt
and Reddy [1]_ and Palmerio et al. [2]_, linear through the thickness,
with the transverse normal strain of [2]_, Eq. (20), `E^{(0)}_{33} =
u^{(1)}_3 + (\phi_x^2 + \phi_y^2)/2`, set to zero (inextensible normal):
the deformed position of the point at `z` is `\mathbf{X} + z \mathbf{d}`,

.. math::

    \mathbf{X} = (x + u, \; y + v, \; w) \qquad
    \mathbf{d} = \left(\phi_x, \; \phi_y, \; 1 - \frac{1}{2}(\phi_x^2 +
    \phi_y^2)\right)

with the second-order director `\mathbf{d}`. The strains are the
coefficients of `z^0` and `z` of the Green-Lagrange strains of this field:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} (u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2) \qquad
    \varepsilon_{yy} = v_{,y} + \frac{1}{2} (u_{,y}^2 + v_{,y}^2 +
    w_{,y}^2)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} + w_{,x}
    w_{,y}

    \kappa_{xx} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,x} \qquad
    \kappa_{yy} = \mathbf{X}_{,y} \cdot \mathbf{d}_{,y} \qquad
    \kappa_{xy} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,y} + \mathbf{X}_{,y}
    \cdot \mathbf{d}_{,x}

    \gamma_{yz} = \mathbf{X}_{,y} \cdot \mathbf{d} \qquad
    \gamma_{xz} = \mathbf{X}_{,x} \cdot \mathbf{d}

cubic polynomials of the generalized derivatives. The membrane strains are
those of :mod:`panels.models.plate_fsdt_gl` and the linear part is that of
:mod:`panels.models.plate_fsdt_donnell` [3]_. To the order `\epsilon^3` of
the moderate rotation theory, with strains of order `\epsilon^2` and
rotations of the normal of order `\epsilon`, the strains are those of
[2]_, Eq. (20); the terms of higher order make them the exact strains of
the field above, so that a finite rigid rotation `t` of a bent state
changes the curvatures by `O(t^4)`, against `O(t^2)` for the linear
curvatures of :mod:`panels.models.plate_fsdt_gl`.

The numerical integration over the domain of
:meth:`.Shell.integration_limits` admits a constitutive matrix and moments
of the density varying over the domain, a partial domain and the
non-linear terms of the Ritz constants.

References
----------
.. [1] Schmidt, R., and Reddy, J. N., "A Refined Small Strain and Moderate
   Rotation Theory of Elastic Anisotropic Shells", Journal of Applied
   Mechanics, Vol. 55, No. 3, 1988, pp. 611-617.
   https://doi.org/10.1115/1.3125837
.. [2] Palmerio, A. F., Reddy, J. N., and Schmidt, R., "On a moderate
   rotation theory of laminated anisotropic shells - Part 1. Theory",
   International Journal of Non-Linear Mechanics, Vol. 25, No. 6, 1990,
   pp. 687-700. https://doi.org/10.1016/0020-7462(90)90007-V
.. [3] Reddy, J. N., "Mechanics of Laminated Composite Plates and Shells:
   Theory and Analysis", 2nd ed., CRC Press, 2003.
   https://doi.org/10.1201/b12409

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
#NOTE the linear kernels, independent of the non-linear strains, are those
#     of plate_fsdt_donnell
from .plate_fsdt_donnell_num import fkM_num, fkAx_num, fkAy_num


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 5
cdef int NE = 8


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
        ``(8, 8)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 8, 8)`` with one matrix per integration point.
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
    cdef double phix_x, phix_y, phiy_x, phiy_y
    cdef double e[8]
    cdef double eL[8]
    cdef double sg[8]
    cdef double sgL[8]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAphix
    cdef double fAphix_x, fAphiy, fAphiy_x, gAu, gAu_y, gAv, gAv_y
    cdef double gAw, gAw_y, gAphix, gAphix_y, gAphiy, gAphiy_y, fBu
    cdef double fBu_x, fBv, fBv_x, fBw, fBw_x, fBphix, fBphix_x
    cdef double fBphiy, fBphiy_x, gBu, gBu_y, gBv, gBv_y, gBw
    cdef double gBw_y, gBphix, gBphix_y, gBphiy, gBphiy_y
    cdef double Bm[96]
    cdef double FB[96]
    cdef double K[144]
    cdef double TA[60]
    cdef double pA[12]
    cdef double pB[12]

    cdef long [::1] kCr, kCc
    cdef double [::1] kCv

    # F as 4-D matrix, must be [nx, ny, NE, NE], when there is one F[NE, NE]
    # for each of the nx * ny integration points
    cdef double F[8 * 8]
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

    ftab = np.zeros((10, m), dtype=DOUBLE)
    gtab = np.zeros((10, n), dtype=DOUBLE)

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
                    ftab[6, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[9, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[7, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[9, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

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
                if NLgeom == 1:
                    for j in range(n):
                        gAu = gtab[0, j]
                        gAu_y = gtab[1, j]
                        gAv = gtab[2, j]
                        gAv_y = gtab[3, j]
                        gAw = gtab[4, j]
                        gAw_y = gtab[5, j]
                        gAphix = gtab[6, j]
                        gAphix_y = gtab[7, j]
                        gAphiy = gtab[8, j]
                        gAphiy_y = gtab[9, j]
                        for i in range(m):
                            fAu = ftab[0, i]
                            fAu_x = ftab[1, i]
                            fAv = ftab[2, i]
                            fAv_x = ftab[3, i]
                            fAw = ftab[4, i]
                            fAw_x = ftab[5, i]
                            fAphix = ftab[6, i]
                            fAphix_x = ftab[7, i]
                            fAphiy = ftab[8, i]
                            fAphiy_x = ftab[9, i]
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

                # [B] = d eps/d d at the point
                for p in range(96):
                    Bm[p] = 0.
                Bm[0] = u_x + 1
                Bm[2] = v_x
                Bm[4] = w_x
                Bm[13] = u_y
                Bm[15] = v_y + 1
                Bm[17] = w_y
                Bm[24] = u_y
                Bm[25] = u_x + 1
                Bm[26] = v_y + 1
                Bm[27] = v_x
                Bm[28] = w_y
                Bm[29] = w_x
                Bm[36] = phix_x
                Bm[38] = phiy_x
                Bm[40] = -phix*phix_x - phiy*phiy_x
                Bm[42] = -phix_x*w_x
                Bm[43] = -phiy_x*w_x
                Bm[44] = -phix*w_x + u_x + 1
                Bm[46] = -phiy*w_x + v_x
                Bm[49] = phix_y
                Bm[51] = phiy_y
                Bm[53] = -phix*phix_y - phiy*phiy_y
                Bm[54] = -phix_y*w_y
                Bm[55] = -phiy_y*w_y
                Bm[57] = -phix*w_y + u_y
                Bm[59] = -phiy*w_y + v_y + 1
                Bm[60] = phix_y
                Bm[61] = phix_x
                Bm[62] = phiy_y
                Bm[63] = phiy_x
                Bm[64] = -phix*phix_y - phiy*phiy_y
                Bm[65] = -phix*phix_x - phiy*phiy_x
                Bm[66] = -phix_x*w_y - phix_y*w_x
                Bm[67] = -phiy_x*w_y - phiy_y*w_x
                Bm[68] = -phix*w_y + u_y
                Bm[69] = -phix*w_x + u_x + 1
                Bm[70] = -phiy*w_y + v_y + 1
                Bm[71] = -phiy*w_x + v_x
                Bm[73] = phix
                Bm[75] = phiy
                Bm[77] = -0.5*(phix*phix) - 0.5*(phiy*phiy) + 1
                Bm[78] = -phix*w_y + u_y
                Bm[79] = -phiy*w_y + v_y + 1
                Bm[84] = phix
                Bm[86] = phiy
                Bm[88] = -0.5*(phix*phix) - 0.5*(phiy*phiy) + 1
                Bm[90] = -phix*w_x + u_x + 1
                Bm[91] = -phiy*w_x + v_x

                # K = [B]^T [F] [B]
                for r in range(NE):
                    for q in range(12):
                        FB[r*12 + q] = 0.
                        for s in range(NE):
                            FB[r*12 + q] += F[r*NE + s]*Bm[s*12 + q]
                for p in range(12):
                    for q in range(12):
                        K[p*12 + q] = 0.
                        for r in range(NE):
                            K[p*12 + q] += Bm[r*12 + p]*FB[r*12 + q]

                # stresses of the strains and of the linear strains
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x + phiy_x*v_x
                e[4] = -phix*phix_y*w_y + phix_y*u_y - phiy*phiy_y*w_y + phiy_y*v_y + phiy_y
                e[5] = -phix*phix_x*w_y - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - phiy*phiy_x*w_y - phiy*phiy_y*w_x + phiy_x*v_y + phiy_x + phiy_y*v_x
                e[6] = -0.5*(phix*phix)*w_y + phix*u_y - 0.5*(phiy*phiy)*w_y + phiy*v_y + phiy + w_y
                e[7] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x + w_x
                eL[0] = u_x
                eL[1] = v_y
                eL[2] = u_y + v_x
                eL[3] = phix_x
                eL[4] = phiy_y
                eL[5] = phix_y + phiy_x
                eL[6] = phiy + w_y
                eL[7] = phix + w_x
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
                K[6] += -sgL[7] + sg[7]
                K[8] += -sgL[3] + sg[3]
                K[9] += -sgL[5] + sg[5]
                K[12] += -sgL[2] + sg[2]
                K[13] += -sgL[1] + sg[1]
                K[18] += -sgL[6] + sg[6]
                K[20] += -sgL[5] + sg[5]
                K[21] += -sgL[4] + sg[4]
                K[26] += -sgL[0] + sg[0]
                K[27] += -sgL[2] + sg[2]
                K[31] += -sgL[7] + sg[7]
                K[34] += -sgL[3] + sg[3]
                K[35] += -sgL[5] + sg[5]
                K[38] += -sgL[2] + sg[2]
                K[39] += -sgL[1] + sg[1]
                K[43] += -sgL[6] + sg[6]
                K[46] += -sgL[5] + sg[5]
                K[47] += -sgL[4] + sg[4]
                K[52] += -sgL[0] + sg[0]
                K[53] += -sgL[2] + sg[2]
                K[54] += -phix*sg[7] - phix_x*sg[3] - phix_y*sg[5]
                K[55] += -phiy*sg[7] - phiy_x*sg[3] - phiy_y*sg[5]
                K[56] += -phix*sg[3]
                K[57] += -phix*sg[5]
                K[58] += -phiy*sg[3]
                K[59] += -phiy*sg[5]
                K[64] += -sgL[2] + sg[2]
                K[65] += -sgL[1] + sg[1]
                K[66] += -phix*sg[6] - phix_x*sg[5] - phix_y*sg[4]
                K[67] += -phiy*sg[6] - phiy_x*sg[5] - phiy_y*sg[4]
                K[68] += -phix*sg[5]
                K[69] += -phix*sg[4]
                K[70] += -phiy*sg[5]
                K[71] += -phiy*sg[4]
                K[72] += -sgL[7] + sg[7]
                K[73] += -sgL[6] + sg[6]
                K[76] += -phix*sg[7] - phix_x*sg[3] - phix_y*sg[5]
                K[77] += -phix*sg[6] - phix_x*sg[5] - phix_y*sg[4]
                K[78] += -sg[6]*w_y - sg[7]*w_x
                K[80] += -sg[3]*w_x - sg[5]*w_y
                K[81] += -sg[4]*w_y - sg[5]*w_x
                K[86] += -sgL[7] + sg[7]
                K[87] += -sgL[6] + sg[6]
                K[88] += -phiy*sg[7] - phiy_x*sg[3] - phiy_y*sg[5]
                K[89] += -phiy*sg[6] - phiy_x*sg[5] - phiy_y*sg[4]
                K[91] += -sg[6]*w_y - sg[7]*w_x
                K[94] += -sg[3]*w_x - sg[5]*w_y
                K[95] += -sg[4]*w_y - sg[5]*w_x
                K[96] += -sgL[3] + sg[3]
                K[97] += -sgL[5] + sg[5]
                K[100] += -phix*sg[3]
                K[101] += -phix*sg[5]
                K[102] += -sg[3]*w_x - sg[5]*w_y
                K[108] += -sgL[5] + sg[5]
                K[109] += -sgL[4] + sg[4]
                K[112] += -phix*sg[5]
                K[113] += -phix*sg[4]
                K[114] += -sg[4]*w_y - sg[5]*w_x
                K[122] += -sgL[3] + sg[3]
                K[123] += -sgL[5] + sg[5]
                K[124] += -phiy*sg[3]
                K[125] += -phiy*sg[5]
                K[127] += -sg[3]*w_x - sg[5]*w_y
                K[134] += -sgL[5] + sg[5]
                K[135] += -sgL[4] + sg[4]
                K[136] += -phiy*sg[5]
                K[137] += -phiy*sg[4]
                K[139] += -sg[4]*w_y - sg[5]*w_x

                c = -1
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAphix = gtab[6, j]
                    gAphix_y = gtab[7, j]
                    gAphiy = gtab[8, j]
                    gAphiy_y = gtab[9, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAphix = ftab[6, i]
                        fAphix_x = ftab[7, i]
                        fAphiy = ftab[8, i]
                        fAphiy_x = ftab[9, i]
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
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[12]
                        TA[1] = pA[0]*K[1] + pA[1]*K[13]
                        TA[2] = pA[0]*K[2] + pA[1]*K[14]
                        TA[3] = pA[0]*K[3] + pA[1]*K[15]
                        TA[4] = pA[0]*K[4] + pA[1]*K[16]
                        TA[5] = pA[0]*K[5] + pA[1]*K[17]
                        TA[6] = pA[0]*K[6] + pA[1]*K[18]
                        TA[7] = pA[0]*K[7] + pA[1]*K[19]
                        TA[8] = pA[0]*K[8] + pA[1]*K[20]
                        TA[9] = pA[0]*K[9] + pA[1]*K[21]
                        TA[10] = pA[0]*K[10] + pA[1]*K[22]
                        TA[11] = pA[0]*K[11] + pA[1]*K[23]
                        TA[12] = pA[2]*K[24] + pA[3]*K[36]
                        TA[13] = pA[2]*K[25] + pA[3]*K[37]
                        TA[14] = pA[2]*K[26] + pA[3]*K[38]
                        TA[15] = pA[2]*K[27] + pA[3]*K[39]
                        TA[16] = pA[2]*K[28] + pA[3]*K[40]
                        TA[17] = pA[2]*K[29] + pA[3]*K[41]
                        TA[18] = pA[2]*K[30] + pA[3]*K[42]
                        TA[19] = pA[2]*K[31] + pA[3]*K[43]
                        TA[20] = pA[2]*K[32] + pA[3]*K[44]
                        TA[21] = pA[2]*K[33] + pA[3]*K[45]
                        TA[22] = pA[2]*K[34] + pA[3]*K[46]
                        TA[23] = pA[2]*K[35] + pA[3]*K[47]
                        TA[24] = pA[4]*K[48] + pA[5]*K[60]
                        TA[25] = pA[4]*K[49] + pA[5]*K[61]
                        TA[26] = pA[4]*K[50] + pA[5]*K[62]
                        TA[27] = pA[4]*K[51] + pA[5]*K[63]
                        TA[28] = pA[4]*K[52] + pA[5]*K[64]
                        TA[29] = pA[4]*K[53] + pA[5]*K[65]
                        TA[30] = pA[4]*K[54] + pA[5]*K[66]
                        TA[31] = pA[4]*K[55] + pA[5]*K[67]
                        TA[32] = pA[4]*K[56] + pA[5]*K[68]
                        TA[33] = pA[4]*K[57] + pA[5]*K[69]
                        TA[34] = pA[4]*K[58] + pA[5]*K[70]
                        TA[35] = pA[4]*K[59] + pA[5]*K[71]
                        TA[36] = pA[6]*K[72] + pA[8]*K[96] + pA[9]*K[108]
                        TA[37] = pA[6]*K[73] + pA[8]*K[97] + pA[9]*K[109]
                        TA[38] = pA[6]*K[74] + pA[8]*K[98] + pA[9]*K[110]
                        TA[39] = pA[6]*K[75] + pA[8]*K[99] + pA[9]*K[111]
                        TA[40] = pA[6]*K[76] + pA[8]*K[100] + pA[9]*K[112]
                        TA[41] = pA[6]*K[77] + pA[8]*K[101] + pA[9]*K[113]
                        TA[42] = pA[6]*K[78] + pA[8]*K[102] + pA[9]*K[114]
                        TA[43] = pA[6]*K[79] + pA[8]*K[103] + pA[9]*K[115]
                        TA[44] = pA[6]*K[80] + pA[8]*K[104] + pA[9]*K[116]
                        TA[45] = pA[6]*K[81] + pA[8]*K[105] + pA[9]*K[117]
                        TA[46] = pA[6]*K[82] + pA[8]*K[106] + pA[9]*K[118]
                        TA[47] = pA[6]*K[83] + pA[8]*K[107] + pA[9]*K[119]
                        TA[48] = pA[7]*K[84] + pA[10]*K[120] + pA[11]*K[132]
                        TA[49] = pA[7]*K[85] + pA[10]*K[121] + pA[11]*K[133]
                        TA[50] = pA[7]*K[86] + pA[10]*K[122] + pA[11]*K[134]
                        TA[51] = pA[7]*K[87] + pA[10]*K[123] + pA[11]*K[135]
                        TA[52] = pA[7]*K[88] + pA[10]*K[124] + pA[11]*K[136]
                        TA[53] = pA[7]*K[89] + pA[10]*K[125] + pA[11]*K[137]
                        TA[54] = pA[7]*K[90] + pA[10]*K[126] + pA[11]*K[138]
                        TA[55] = pA[7]*K[91] + pA[10]*K[127] + pA[11]*K[139]
                        TA[56] = pA[7]*K[92] + pA[10]*K[128] + pA[11]*K[140]
                        TA[57] = pA[7]*K[93] + pA[10]*K[129] + pA[11]*K[141]
                        TA[58] = pA[7]*K[94] + pA[10]*K[130] + pA[11]*K[142]
                        TA[59] = pA[7]*K[95] + pA[10]*K[131] + pA[11]*K[143]
                        for l in range(n):
                            gBu = gtab[0, l]
                            gBu_y = gtab[1, l]
                            gBv = gtab[2, l]
                            gBv_y = gtab[3, l]
                            gBw = gtab[4, l]
                            gBw_y = gtab[5, l]
                            gBphix = gtab[6, l]
                            gBphix_y = gtab[7, l]
                            gBphiy = gtab[8, l]
                            gBphiy_y = gtab[9, l]
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
                                fBphix = ftab[6, k]
                                fBphix_x = ftab[7, k]
                                fBphiy = ftab[8, k]
                                fBphiy_x = ftab[9, k]
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
                                kCv[c] += weight*(TA[4]*pB[4] + TA[5]*pB[5])
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
                                kCv[c] += weight*(TA[12]*pB[0] + TA[13]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[14]*pB[2] + TA[15]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[16]*pB[4] + TA[17]*pB[5])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[18]*pB[6] + TA[20]*pB[8] + TA[21]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[19]*pB[7] + TA[22]*pB[10] + TA[23]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[24]*pB[0] + TA[25]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[26]*pB[2] + TA[27]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[28]*pB[4] + TA[29]*pB[5])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[30]*pB[6] + TA[32]*pB[8] + TA[33]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[31]*pB[7] + TA[34]*pB[10] + TA[35]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[36]*pB[0] + TA[37]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[38]*pB[2] + TA[39]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[40]*pB[4] + TA[41]*pB[5])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[42]*pB[6] + TA[44]*pB[8] + TA[45]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[43]*pB[7] + TA[46]*pB[10] + TA[47]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[48]*pB[0] + TA[49]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[50]*pB[2] + TA[51]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[52]*pB[4] + TA[53]*pB[5])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[54]*pB[6] + TA[56]*pB[8] + TA[57]*pB[9])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[55]*pB[7] + TA[58]*pB[10] + TA[59]*pB[11])

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
        ``(8, 8)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 8, 8)`` with one matrix per integration point.
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
    cdef double phix_x, phix_y, phiy_x, phiy_y
    cdef double e[8]
    cdef double sg[8]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAphix
    cdef double fAphix_x, fAphiy, fAphiy_x, gAu, gAu_y, gAv, gAv_y
    cdef double gAw, gAw_y, gAphix, gAphix_y, gAphiy, gAphiy_y, fBu
    cdef double fBu_x, fBv, fBv_x, fBw, fBw_x, fBphix, fBphix_x
    cdef double fBphiy, fBphiy_x, gBu, gBu_y, gBv, gBv_y, gBw
    cdef double gBw_y, gBphix, gBphix_y, gBphiy, gBphiy_y
    cdef double K[144]
    cdef double TA[60]
    cdef double pA[12]
    cdef double pB[12]

    cdef long [::1] kGr, kGc
    cdef double [::1] kGv

    # F as 4-D matrix, must be [nx, ny, NE, NE], when there is one F[NE, NE]
    # for each of the nx * ny integration points
    cdef double F[8 * 8]
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

    ftab = np.zeros((10, m), dtype=DOUBLE)
    gtab = np.zeros((10, n), dtype=DOUBLE)

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
                    ftab[6, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[9, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[7, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[9, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

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
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAphix = gtab[6, j]
                    gAphix_y = gtab[7, j]
                    gAphiy = gtab[8, j]
                    gAphiy_y = gtab[9, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAphix = ftab[6, i]
                        fAphix_x = ftab[7, i]
                        fAphiy = ftab[8, i]
                        fAphiy_x = ftab[9, i]
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

                # linear strains and stresses
                e[0] = u_x
                e[1] = v_y
                e[2] = u_y + v_x
                e[3] = phix_x
                e[4] = phiy_y
                e[5] = phix_y + phiy_x
                e[6] = phiy + w_y
                e[7] = phix + w_x
                for r in range(NE):
                    sg[r] = 0.
                    for s in range(NE):
                        sg[r] += F[r*NE + s]*e[s]
                sg[0] += Nxx0
                sg[1] += Nyy0
                sg[2] += Nxy0

                # K = [H0(sigma)]
                for p in range(144):
                    K[p] = 0.
                K[0] = sg[0]
                K[1] = sg[2]
                K[6] = sg[7]
                K[8] = sg[3]
                K[9] = sg[5]
                K[12] = sg[2]
                K[13] = sg[1]
                K[18] = sg[6]
                K[20] = sg[5]
                K[21] = sg[4]
                K[26] = sg[0]
                K[27] = sg[2]
                K[31] = sg[7]
                K[34] = sg[3]
                K[35] = sg[5]
                K[38] = sg[2]
                K[39] = sg[1]
                K[43] = sg[6]
                K[46] = sg[5]
                K[47] = sg[4]
                K[52] = sg[0]
                K[53] = sg[2]
                K[64] = sg[2]
                K[65] = sg[1]
                K[72] = sg[7]
                K[73] = sg[6]
                K[86] = sg[7]
                K[87] = sg[6]
                K[96] = sg[3]
                K[97] = sg[5]
                K[108] = sg[5]
                K[109] = sg[4]
                K[122] = sg[3]
                K[123] = sg[5]
                K[134] = sg[5]
                K[135] = sg[4]

                c = -1
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAphix = gtab[6, j]
                    gAphix_y = gtab[7, j]
                    gAphiy = gtab[8, j]
                    gAphiy_y = gtab[9, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAphix = ftab[6, i]
                        fAphix_x = ftab[7, i]
                        fAphiy = ftab[8, i]
                        fAphiy_x = ftab[9, i]
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
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[12]
                        TA[1] = pA[0]*K[1] + pA[1]*K[13]
                        TA[6] = pA[0]*K[6] + pA[1]*K[18]
                        TA[8] = pA[0]*K[8] + pA[1]*K[20]
                        TA[9] = pA[0]*K[9] + pA[1]*K[21]
                        TA[14] = pA[2]*K[26] + pA[3]*K[38]
                        TA[15] = pA[2]*K[27] + pA[3]*K[39]
                        TA[19] = pA[2]*K[31] + pA[3]*K[43]
                        TA[22] = pA[2]*K[34] + pA[3]*K[46]
                        TA[23] = pA[2]*K[35] + pA[3]*K[47]
                        TA[28] = pA[4]*K[52] + pA[5]*K[64]
                        TA[29] = pA[4]*K[53] + pA[5]*K[65]
                        TA[36] = pA[6]*K[72] + pA[8]*K[96] + pA[9]*K[108]
                        TA[37] = pA[6]*K[73] + pA[8]*K[97] + pA[9]*K[109]
                        TA[50] = pA[7]*K[86] + pA[10]*K[122] + pA[11]*K[134]
                        TA[51] = pA[7]*K[87] + pA[10]*K[123] + pA[11]*K[135]
                        for l in range(n):
                            gBu = gtab[0, l]
                            gBu_y = gtab[1, l]
                            gBv = gtab[2, l]
                            gBv_y = gtab[3, l]
                            gBw = gtab[4, l]
                            gBw_y = gtab[5, l]
                            gBphix = gtab[6, l]
                            gBphix_y = gtab[7, l]
                            gBphiy = gtab[8, l]
                            gBphiy_y = gtab[9, l]
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
                                fBphix = ftab[6, k]
                                fBphix_x = ftab[7, k]
                                fBphiy = ftab[8, k]
                                fBphiy_x = ftab[9, k]
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
                                kGv[c] += weight*(TA[14]*pB[2] + TA[15]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+4
                                kGv[c] += weight*(TA[19]*pB[7] + TA[22]*pB[10] + TA[23]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[28]*pB[4] + TA[29]*pB[5])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+3
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[36]*pB[0] + TA[37]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+4
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[50]*pB[2] + TA[51]*pB[3])

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
        ``(8, 8)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 8, 8)`` with one matrix per integration point.
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
    cdef double phix_x, phix_y, phiy_x, phiy_y
    cdef double e[8]
    cdef double sg[8]
    cdef double S[12]
    cdef double pA[12]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAphix
    cdef double fAphix_x, fAphiy, fAphiy_x, gAu, gAu_y, gAv, gAv_y
    cdef double gAw, gAw_y, gAphix, gAphix_y, gAphiy, gAphiy_y

    cdef double [::1] fint

    # F as 4-D matrix, must be [nx, ny, NE, NE], when there is one F[NE, NE]
    # for each of the nx * ny integration points
    cdef double F[8 * 8]
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

    ftab = np.zeros((10, m), dtype=DOUBLE)
    gtab = np.zeros((10, n), dtype=DOUBLE)

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
                    ftab[6, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[9, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[7, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[9, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

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
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAphix = gtab[6, j]
                    gAphix_y = gtab[7, j]
                    gAphiy = gtab[8, j]
                    gAphiy_y = gtab[9, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAphix = ftab[6, i]
                        fAphix_x = ftab[7, i]
                        fAphiy = ftab[8, i]
                        fAphiy_x = ftab[9, i]
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

                # strains, stresses and S = [B]^T sigma
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x + phiy_x*v_x
                e[4] = -phix*phix_y*w_y + phix_y*u_y - phiy*phiy_y*w_y + phiy_y*v_y + phiy_y
                e[5] = -phix*phix_x*w_y - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - phiy*phiy_x*w_y - phiy*phiy_y*w_x + phiy_x*v_y + phiy_x + phiy_y*v_x
                e[6] = -0.5*(phix*phix)*w_y + phix*u_y - 0.5*(phiy*phiy)*w_y + phiy*v_y + phiy + w_y
                e[7] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x + w_x
                for r in range(NE):
                    sg[r] = 0.
                    for s in range(NE):
                        sg[r] += F[r*NE + s]*e[s]
                S[0] = phix*sg[7] + phix_x*sg[3] + phix_y*sg[5] + sg[0]*u_x + sg[0] + sg[2]*u_y
                S[1] = phix*sg[6] + phix_x*sg[5] + phix_y*sg[4] + sg[1]*u_y + sg[2]*u_x + sg[2]
                S[2] = phiy*sg[7] + phiy_x*sg[3] + phiy_y*sg[5] + sg[0]*v_x + sg[2]*v_y + sg[2]
                S[3] = phiy*sg[6] + phiy_x*sg[5] + phiy_y*sg[4] + sg[1]*v_y + sg[1] + sg[2]*v_x
                S[4] = -0.5*(phix*phix)*sg[7] - phix*phix_x*sg[3] - phix*phix_y*sg[5] - 0.5*(phiy*phiy)*sg[7] - phiy*phiy_x*sg[3] - phiy*phiy_y*sg[5] + sg[0]*w_x + sg[2]*w_y + sg[7]
                S[5] = -0.5*(phix*phix)*sg[6] - phix*phix_x*sg[5] - phix*phix_y*sg[4] - 0.5*(phiy*phiy)*sg[6] - phiy*phiy_x*sg[5] - phiy*phiy_y*sg[4] + sg[1]*w_y + sg[2]*w_x + sg[6]
                S[6] = -phix*sg[6]*w_y - phix*sg[7]*w_x - phix_x*sg[3]*w_x - phix_x*sg[5]*w_y - phix_y*sg[4]*w_y - phix_y*sg[5]*w_x + sg[6]*u_y + sg[7]*u_x + sg[7]
                S[7] = -phiy*sg[6]*w_y - phiy*sg[7]*w_x - phiy_x*sg[3]*w_x - phiy_x*sg[5]*w_y - phiy_y*sg[4]*w_y - phiy_y*sg[5]*w_x + sg[6]*v_y + sg[6] + sg[7]*v_x
                S[8] = -phix*sg[3]*w_x - phix*sg[5]*w_y + sg[3]*u_x + sg[3] + sg[5]*u_y
                S[9] = -phix*sg[4]*w_y - phix*sg[5]*w_x + sg[4]*u_y + sg[5]*u_x + sg[5]
                S[10] = -phiy*sg[3]*w_x - phiy*sg[5]*w_y + sg[3]*v_x + sg[5]*v_y + sg[5]
                S[11] = -phiy*sg[4]*w_y - phiy*sg[5]*w_x + sg[4]*v_y + sg[4] + sg[5]*v_x

                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAphix = gtab[6, j]
                    gAphix_y = gtab[7, j]
                    gAphiy = gtab[8, j]
                    gAphiy_y = gtab[9, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAphix = ftab[6, i]
                        fAphix_x = ftab[7, i]
                        fAphiy = ftab[8, i]
                        fAphiy_x = ftab[9, i]
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
                        fint[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        fint[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3])
                        fint[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5])
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
        ``(8, 8)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 8, 8)`` with one matrix per integration point.
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
    cdef double phix_x, phix_y, phiy_x, phiy_y
    cdef double du_x, du_y, dv_x, dv_y, dw_x, dw_y, dphix, dphiy
    cdef double dphix_x, dphix_y, dphiy_x, dphiy_y
    cdef double e[8]
    cdef double de[8]
    cdef double sg[8]
    cdef double dsg[8]
    cdef double S[12]
    cdef double pA[12]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAphix
    cdef double fAphix_x, fAphiy, fAphiy_x, gAu, gAu_y, gAv, gAv_y
    cdef double gAw, gAw_y, gAphix, gAphix_y, gAphiy, gAphiy_y

    cdef double [::1] kTv

    if cs.shape[0] != size or vs.shape[0] != size:
        raise ValueError('cs and vs must have the size %d' % size)

    # F as 4-D matrix, must be [nx, ny, NE, NE], when there is one F[NE, NE]
    # for each of the nx * ny integration points
    cdef double F[8 * 8]
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

    ftab = np.zeros((10, m), dtype=DOUBLE)
    gtab = np.zeros((10, n), dtype=DOUBLE)

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
                    ftab[6, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[9, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[7, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[8, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[9, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

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
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAphix = gtab[6, j]
                    gAphix_y = gtab[7, j]
                    gAphiy = gtab[8, j]
                    gAphiy_y = gtab[9, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAphix = ftab[6, i]
                        fAphix_x = ftab[7, i]
                        fAphiy = ftab[8, i]
                        fAphiy_x = ftab[9, i]
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

                # strains, strain increments, stresses
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x + phiy_x*v_x
                e[4] = -phix*phix_y*w_y + phix_y*u_y - phiy*phiy_y*w_y + phiy_y*v_y + phiy_y
                e[5] = -phix*phix_x*w_y - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - phiy*phiy_x*w_y - phiy*phiy_y*w_x + phiy_x*v_y + phiy_x + phiy_y*v_x
                e[6] = -0.5*(phix*phix)*w_y + phix*u_y - 0.5*(phiy*phiy)*w_y + phiy*v_y + phiy + w_y
                e[7] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x + w_x
                de[0] = du_x*u_x + du_x + dv_x*v_x + dw_x*w_x
                de[1] = du_y*u_y + dv_y*v_y + dv_y + dw_y*w_y
                de[2] = du_x*u_y + du_y*u_x + du_y + dv_x*v_y + dv_x + dv_y*v_x + dw_x*w_y + dw_y*w_x
                de[3] = -dphix*phix_x*w_x - dphix_x*phix*w_x + dphix_x*u_x + dphix_x - dphiy*phiy_x*w_x - dphiy_x*phiy*w_x + dphiy_x*v_x + du_x*phix_x + dv_x*phiy_x - dw_x*phix*phix_x - dw_x*phiy*phiy_x
                de[4] = -dphix*phix_y*w_y - dphix_y*phix*w_y + dphix_y*u_y - dphiy*phiy_y*w_y - dphiy_y*phiy*w_y + dphiy_y*v_y + dphiy_y + du_y*phix_y + dv_y*phiy_y - dw_y*phix*phix_y - dw_y*phiy*phiy_y
                de[5] = -dphix*phix_x*w_y - dphix*phix_y*w_x - dphix_x*phix*w_y + dphix_x*u_y - dphix_y*phix*w_x + dphix_y*u_x + dphix_y - dphiy*phiy_x*w_y - dphiy*phiy_y*w_x - dphiy_x*phiy*w_y + dphiy_x*v_y + dphiy_x - dphiy_y*phiy*w_x + dphiy_y*v_x + du_x*phix_y + du_y*phix_x + dv_x*phiy_y + dv_y*phiy_x - dw_x*phix*phix_y - dw_x*phiy*phiy_y - dw_y*phix*phix_x - dw_y*phiy*phiy_x
                de[6] = -dphix*phix*w_y + dphix*u_y - dphiy*phiy*w_y + dphiy*v_y + dphiy + du_y*phix + dv_y*phiy - 0.5*dw_y*(phix*phix) - 0.5*dw_y*(phiy*phiy) + dw_y
                de[7] = -dphix*phix*w_x + dphix*u_x + dphix - dphiy*phiy*w_x + dphiy*v_x + du_x*phix + dv_x*phiy - 0.5*dw_x*(phix*phix) - 0.5*dw_x*(phiy*phiy) + dw_x
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
                S[0] = dphix*sg[7] + dphix_x*sg[3] + dphix_y*sg[5] + dsg[0]*u_x + dsg[0] + dsg[2]*u_y + dsg[3]*phix_x + dsg[5]*phix_y + dsg[7]*phix + du_x*sg[0] + du_y*sg[2]
                S[1] = dphix*sg[6] + dphix_x*sg[5] + dphix_y*sg[4] + dsg[1]*u_y + dsg[2]*u_x + dsg[2] + dsg[4]*phix_y + dsg[5]*phix_x + dsg[6]*phix + du_x*sg[2] + du_y*sg[1]
                S[2] = dphiy*sg[7] + dphiy_x*sg[3] + dphiy_y*sg[5] + dsg[0]*v_x + dsg[2]*v_y + dsg[2] + dsg[3]*phiy_x + dsg[5]*phiy_y + dsg[7]*phiy + dv_x*sg[0] + dv_y*sg[2]
                S[3] = dphiy*sg[6] + dphiy_x*sg[5] + dphiy_y*sg[4] + dsg[1]*v_y + dsg[1] + dsg[2]*v_x + dsg[4]*phiy_y + dsg[5]*phiy_x + dsg[6]*phiy + dv_x*sg[2] + dv_y*sg[1]
                S[4] = -dphix*phix*sg[7] - dphix*phix_x*sg[3] - dphix*phix_y*sg[5] - dphix_x*phix*sg[3] - dphix_y*phix*sg[5] - dphiy*phiy*sg[7] - dphiy*phiy_x*sg[3] - dphiy*phiy_y*sg[5] - dphiy_x*phiy*sg[3] - dphiy_y*phiy*sg[5] + dsg[0]*w_x + dsg[2]*w_y - dsg[3]*phix*phix_x - dsg[3]*phiy*phiy_x - dsg[5]*phix*phix_y - dsg[5]*phiy*phiy_y - 0.5*dsg[7]*(phix*phix) - 0.5*dsg[7]*(phiy*phiy) + dsg[7] + dw_x*sg[0] + dw_y*sg[2]
                S[5] = -dphix*phix*sg[6] - dphix*phix_x*sg[5] - dphix*phix_y*sg[4] - dphix_x*phix*sg[5] - dphix_y*phix*sg[4] - dphiy*phiy*sg[6] - dphiy*phiy_x*sg[5] - dphiy*phiy_y*sg[4] - dphiy_x*phiy*sg[5] - dphiy_y*phiy*sg[4] + dsg[1]*w_y + dsg[2]*w_x - dsg[4]*phix*phix_y - dsg[4]*phiy*phiy_y - dsg[5]*phix*phix_x - dsg[5]*phiy*phiy_x - 0.5*dsg[6]*(phix*phix) - 0.5*dsg[6]*(phiy*phiy) + dsg[6] + dw_x*sg[2] + dw_y*sg[1]
                S[6] = -dphix*sg[6]*w_y - dphix*sg[7]*w_x - dphix_x*sg[3]*w_x - dphix_x*sg[5]*w_y - dphix_y*sg[4]*w_y - dphix_y*sg[5]*w_x - dsg[3]*phix_x*w_x - dsg[4]*phix_y*w_y - dsg[5]*phix_x*w_y - dsg[5]*phix_y*w_x - dsg[6]*phix*w_y + dsg[6]*u_y - dsg[7]*phix*w_x + dsg[7]*u_x + dsg[7] + du_x*sg[7] + du_y*sg[6] - dw_x*phix*sg[7] - dw_x*phix_x*sg[3] - dw_x*phix_y*sg[5] - dw_y*phix*sg[6] - dw_y*phix_x*sg[5] - dw_y*phix_y*sg[4]
                S[7] = -dphiy*sg[6]*w_y - dphiy*sg[7]*w_x - dphiy_x*sg[3]*w_x - dphiy_x*sg[5]*w_y - dphiy_y*sg[4]*w_y - dphiy_y*sg[5]*w_x - dsg[3]*phiy_x*w_x - dsg[4]*phiy_y*w_y - dsg[5]*phiy_x*w_y - dsg[5]*phiy_y*w_x - dsg[6]*phiy*w_y + dsg[6]*v_y + dsg[6] - dsg[7]*phiy*w_x + dsg[7]*v_x + dv_x*sg[7] + dv_y*sg[6] - dw_x*phiy*sg[7] - dw_x*phiy_x*sg[3] - dw_x*phiy_y*sg[5] - dw_y*phiy*sg[6] - dw_y*phiy_x*sg[5] - dw_y*phiy_y*sg[4]
                S[8] = -dphix*sg[3]*w_x - dphix*sg[5]*w_y - dsg[3]*phix*w_x + dsg[3]*u_x + dsg[3] - dsg[5]*phix*w_y + dsg[5]*u_y + du_x*sg[3] + du_y*sg[5] - dw_x*phix*sg[3] - dw_y*phix*sg[5]
                S[9] = -dphix*sg[4]*w_y - dphix*sg[5]*w_x - dsg[4]*phix*w_y + dsg[4]*u_y - dsg[5]*phix*w_x + dsg[5]*u_x + dsg[5] + du_x*sg[5] + du_y*sg[4] - dw_x*phix*sg[5] - dw_y*phix*sg[4]
                S[10] = -dphiy*sg[3]*w_x - dphiy*sg[5]*w_y - dsg[3]*phiy*w_x + dsg[3]*v_x - dsg[5]*phiy*w_y + dsg[5]*v_y + dsg[5] + dv_x*sg[3] + dv_y*sg[5] - dw_x*phiy*sg[3] - dw_y*phiy*sg[5]
                S[11] = -dphiy*sg[4]*w_y - dphiy*sg[5]*w_x - dsg[4]*phiy*w_y + dsg[4]*v_y + dsg[4] - dsg[5]*phiy*w_x + dsg[5]*v_x + dv_x*sg[5] + dv_y*sg[4] - dw_x*phiy*sg[5] - dw_y*phiy*sg[4]

                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAphix = gtab[6, j]
                    gAphix_y = gtab[7, j]
                    gAphiy = gtab[8, j]
                    gAphiy_y = gtab[9, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAphix = ftab[6, i]
                        fAphix_x = ftab[7, i]
                        fAphiy = ftab[8, i]
                        fAphiy_x = ftab[9, i]
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
                        kTv[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        kTv[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3])
                        kTv[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5])
                        kTv[col+3] += weight*(pA[6]*S[6] + pA[8]*S[8] + pA[9]*S[9])
                        kTv[col+4] += weight*(pA[7]*S[7] + pA[10]*S[10] + pA[11]*S[11])

    return np.asarray(kTv)


# BEGIN follower pressure, generated by theory/shells/follower_pressure/write_pyx.py


def calc_fext_follower(double [::1] cs, double [:, ::1] pnxny,
        double px1, double px2, double py1, double py2, object shell,
        int size, int col0, int quadratic=0, int reference=1):
    r"""Force vector of a follower pressure in the configuration ``cs``

    Integrates, over the patch ``px1 <= x <= px2``, ``py1 <= y <= py2``
    with ``nx x ny`` Gauss-Legendre points, ``(nx, ny) = pnxny.shape``,
    the force vector of a pressure that stays normal to the deformed
    mid-surface and acts on its deformed area. Only ``u, v, w`` enter.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements (the rotations in radians).
    pnxny : array-like
        Pressure at each Gauss-Legendre point of the patch, already
        multiplied by the load factor, shape ``(nx, ny)``, force per unit
        area, positive along `z`.
    px1, px2, py1, py2 : float
        Limits of the loaded patch along `x` and `y`, lengths.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Size of the vector, the number of degrees of freedom of the assembly.
    col0 : int
        Position of the vector where the terms of this shell start.
    quadratic : int, optional
        ``1`` keeps the part of the area vector quadratic in the
        displacement gradients, default ``0``.
    reference : int, optional
        ``1`` includes the dead part ``F_p(0)``, the load in the undeformed
        configuration, default ``1``.

    Returns
    -------
    fext : numpy.ndarray
        Force vector of size ``size``.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With `\boldsymbol{X} = \boldsymbol{X}_0 + \boldsymbol{u}` the
    deformed mid-surface, the virtual work of the pressure `p` is `\delta W_p
    = \int p \, \boldsymbol{a} \cdot \delta \boldsymbol{u} \, dx \, dy`, with
    the area vector per unit undeformed area `\boldsymbol{a} =
    \boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y} = \boldsymbol{a}_0 +
    \boldsymbol{a}_1 + \boldsymbol{a}_2`, of order 0, 1 and 2 in the
    displacement gradients, in the local basis of the undeformed
    mid-surface. The force vector is

    .. math::

        \boldsymbol{F}_p = \int p \left[(\boldsymbol{S}^u)^\top a_x +
        (\boldsymbol{S}^v)^\top a_y + (\boldsymbol{S}^w)^\top a_z\right]
        dx \, dy

    with `\boldsymbol{S}^u`, `\boldsymbol{S}^v`, `\boldsymbol{S}^w` the rows
    of the approximation functions of `u`, `v`, `w`, and the area vector
    `r_{ef} \boldsymbol{a}_0 + \boldsymbol{a}_1 + q_f \boldsymbol{a}_2` of
    the kinematics of the module::

            ax = qf*v_x*w_y - qf*v_y*w_x - w_x
        ay = -qf*u_x*w_y + qf*u_y*w_x - w_y
        az = qf*u_x*v_y - qf*u_y*v_x + ref + u_x + v_y

    with ``qf = 1`` for ``quadratic=1`` and ``qf = 0`` otherwise, and ``ref =
    1`` for ``reference=1`` and ``ref = 0`` otherwise. The part
    `\boldsymbol{a}_2` is of the order of the terms that the strains
    neglect.

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
    r"""Load stiffness matrix of a follower pressure

    Returns ``-dF_p/dc`` at ``cs``, ``F_p`` the force vector of
    :func:`calc_fext_follower`. The matrix is unsymmetric in general, so
    it is assembled in full, not as an upper triangle, and must not be
    symmetrized. With ``quadratic=0`` it does not depend on ``cs``.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements (the rotations in radians).
    pnxny : array-like
        Pressure at each Gauss-Legendre point of the patch, already
        multiplied by the load factor, shape ``(nx, ny)``, force per unit
        area, positive along `z`.
    px1, px2, py1, py2 : float
        Limits of the loaded patch along `x` and `y`, lengths.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.
    quadratic : int, optional
        ``1`` keeps the part of the area vector quadratic in the
        displacement gradients, default ``0``.

    Returns
    -------
    kC : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. The matrix is assembled in full.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With the area vector `\boldsymbol{a}(\boldsymbol{c})` of
    :func:`calc_fext_follower`:

    .. math::

        \boldsymbol{K}_F = -\frac{\partial \boldsymbol{F}_p}{\partial
        \boldsymbol{c}} = -\int p \left[(\boldsymbol{S}^u)^\top
        \frac{\partial a_x}{\partial \boldsymbol{c}} + (\boldsymbol{S}^v)^\top
        \frac{\partial a_y}{\partial \boldsymbol{c}} +
        (\boldsymbol{S}^w)^\top \frac{\partial a_z}{\partial
        \boldsymbol{c}}\right] dx \, dy

    added to the tangent stiffness matrix. For the area vector of first
    order, `\boldsymbol{K}_F` is symmetric when `p` is uniform over the whole
    integration domain and, on each edge, either `w = 0` or the displacement
    normal to the edge is zero; otherwise, e.g. for a free edge, a patch or a
    varying `p`, the load is not conservative and `\boldsymbol{K}_F` is
    unsymmetric.

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
