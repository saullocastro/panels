#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Cylindrical shells using the total-Lagrangian third-order shear deformation
theory (TSDT) with a second-order director

Numerically integrated kernels of the model ``'cylshell_tsdt_tl'``: the
constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the internal force vector :func:`calc_fint`, the product
of the tangent stiffness matrix by a vector :func:`calc_kTv`, and the force
vector and load stiffness matrix of a follower pressure,
:func:`calc_fext_follower` and :func:`fkCfollower_num`. The mass and
aerodynamic matrices, ``fkM_num``, ``fkAx_num`` and ``fkAy_num``, are those
of :mod:`panels.models.cylshell_tsdt_sanders_num`. The constitutive matrix ``Finput`` is the
``13 x 13`` matrix ``ABD`` of :class:`.Shell` for this model.

The degrees of freedom of each term of the approximation are
``u, v, w, phix, phiy``; the degree of freedom ``k`` of the term ``(i, j)``,
``0 <= i < m``, ``0 <= j < n``, is at the row ``row0 + 5*(j*m + i) + k``
of the matrices.

Notes
-----
The displacement field of Reddy [1]_, cubic through the thickness, is
written for finite rotations as in :mod:`panels.models.plate_tsdt_tl_num`,
with the deformed mid-surface `\mathbf{X}`, the second-order director
`\mathbf{d}` of :mod:`panels.models.cylshell_fsdt_tl_num`, after the
moderate rotation theory of Schmidt and Reddy [2]_ and Palmerio et al.
[3]_, and the normal of the deformed mid-surface, not normalized,
`\mathbf{N} = \mathbf{X}_{,x} \times \mathbf{X}_{,y}`, in the local basis
`(\mathbf{e}_x, \mathbf{e}_y, \mathbf{e}_z)`, `\mathbf{e}_{y,y} =
-\mathbf{e}_z/r`, `\mathbf{e}_{z,y} = \mathbf{e}_y/r`. The deformed position
of the point at `z` is

.. math::

    \mathbf{X} + z \mathbf{d} + z^3 \mathbf{q} \qquad
    \mathbf{q} = -c_1 (\mathbf{d} - \mathbf{N}) \qquad
    c_1 = \frac{4}{3 h^2}

    \mathbf{X}_{,x} = (1 + u_{,x}, \; v_{,x}, \; w_{,x}) \qquad
    \mathbf{X}_{,y} = (u_{,y}, \; 1 + v_{,y} + w/r, \; w_{,y} - v/r)

    \mathbf{d} = \left(\psi_x, \; \psi_y, \; 1 - \frac{1}{2}(\psi_x^2 +
    \psi_y^2)\right) \qquad \psi_x = \phi_x \qquad \psi_y = \phi_y + v/r

The membrane strains, the curvatures `\kappa` and the transverse shear
strains `\gamma` are those of :mod:`panels.models.cylshell_fsdt_tl_num`,
and

.. math::

    \kappa^{(3)}_{xx} = \mathbf{X}_{,x} \cdot \mathbf{q}_{,x} \qquad
    \kappa^{(3)}_{yy} = \mathbf{X}_{,y} \cdot \mathbf{q}_{,y} -
    \frac{c_1}{r} (\varepsilon_{xx} + \varepsilon_{yy}) \qquad
    \kappa^{(3)}_{xy} = \mathbf{X}_{,x} \cdot \mathbf{q}_{,y} +
    \mathbf{X}_{,y} \cdot \mathbf{q}_{,x}

    \gamma^{(2)}_{yz} = -3 c_1 \gamma_{yz} \qquad
    \gamma^{(2)}_{xz} = -3 c_1 \gamma_{xz}

polynomials of degree three of the generalized derivatives, among which
the second derivatives of `u` and `v`. Without the term `-c_1
(\varepsilon_{xx} + \varepsilon_{yy})/r`, the third-order strains are the
coefficients of `z^3` and `z^2` of the Green-Lagrange strains of the field
above; the term removes from the linear part the stretch of `\mathbf{N}`
times the curvature `1/r`, so that the linear part is that of
:mod:`panels.models.cylshell_tsdt_sanders` [1]_. The membrane strains are
those of :mod:`panels.models.cylshell_tsdt_gl`. The transverse shear strain
`\gamma_\alpha (1 - 4 z^2/h^2)` vanishes at the faces `z = \pm h/2` for
any deformation, and a finite rigid rotation `t` changes the strains by
`O(t^4)`, from the second-order director.

The numerical integration over the domain of
:meth:`.Shell.integration_limits` admits a constitutive matrix and moments
of the density varying over the domain, a partial domain and the
non-linear terms of the Ritz constants.

References
----------
.. [1] Reddy, J. N., "Mechanics of Laminated Composite Plates and Shells:
   Theory and Analysis", 2nd ed., CRC Press, 2003.
   https://doi.org/10.1201/b12409
.. [2] Schmidt, R., and Reddy, J. N., "A Refined Small Strain and Moderate
   Rotation Theory of Elastic Anisotropic Shells", Journal of Applied
   Mechanics, Vol. 55, No. 3, 1988, pp. 611-617.
   https://doi.org/10.1115/1.3125837
.. [3] Palmerio, A. F., Reddy, J. N., and Schmidt, R., "On a moderate
   rotation theory of laminated anisotropic shells - Part 1. Theory",
   International Journal of Non-Linear Mechanics, Vol. 25, No. 6, 1990,
   pp. 687-700. https://doi.org/10.1016/0020-7462(90)90007-V

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
#NOTE the linear kernels, independent of the non-linear strains, are those
#     of cylshell_tsdt_sanders
from .cylshell_tsdt_sanders_num import fkM_num, fkAx_num, fkAy_num


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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the first and second derivatives of `u`, `v`, `w`, the values of `v` and `w`, the rotations and their gradients), the strains
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
    cdef double h, c1
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w
    cdef double phix, phiy, phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy
    cdef double w_xy, u_xx, u_yy, u_xy, v_xx, v_yy, v_xy
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
    cdef double Bm[299]
    cdef double FB[299]
    cdef double K[529]
    cdef double TA[115]
    cdef double pA[23]
    cdef double pB[23]

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
    #NOTE the traction-free faces are assumed at z = +-h/2 about the
    #     mid-surface
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    radius = shell.r
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
                v = 0.
                w = 0.
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
                            v += fAv*gAv*cs[col+1]
                            w += fAw*gAw*cs[col+2]
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
                for p in range(299):
                    Bm[p] = 0.
                Bm[0] = u_x + 1
                Bm[2] = v_x
                Bm[4] = w_x
                Bm[24] = u_y
                Bm[26] = v_y + 1 + w/radius
                Bm[28] = w_y - v/radius
                Bm[29] = -w_y/radius + v/(radius*radius)
                Bm[30] = v_y/radius + 1/radius + w/(radius*radius)
                Bm[46] = u_y
                Bm[47] = u_x + 1
                Bm[48] = v_y + 1 + w/radius
                Bm[49] = v_x
                Bm[50] = w_y - v/radius
                Bm[51] = w_x
                Bm[52] = -w_x/radius
                Bm[53] = v_x/radius
                Bm[69] = phix_x
                Bm[71] = -phiy*w_x/radius + phiy_x + 2*v_x/radius - v*w_x/(radius*radius)
                Bm[73] = -phix*phix_x - phiy*phiy_x - phiy*v_x/radius - phiy_x*v/radius - v*v_x/(radius*radius)
                Bm[75] = -phiy_x*w_x/radius - v_x*w_x/(radius*radius)
                Bm[77] = -phix_x*w_x
                Bm[78] = -phiy_x*w_x - v_x*w_x/radius
                Bm[79] = -phix*w_x + u_x + 1
                Bm[81] = -phiy*w_x + v_x - v*w_x/radius
                Bm[93] = phix_y - u_y/radius
                Bm[95] = -0.5*(phix*phix)/radius - 0.5*(phiy*phiy)/radius - phiy*w_y/radius + phiy_y + v_y/radius + 1/radius - v*w_y/(radius*radius) + 0.5*(v*v)/(radius*radius*radius)
                Bm[97] = -phix*phix_y - phiy*phiy_y - phiy*v_y/radius - phiy/radius - phiy_y*v/radius - w_y/radius - v*v_y/(radius*radius)
                Bm[98] = phix*phix_y/radius + phiy*phiy_y/radius - phiy*w/(radius*radius*radius) - phiy_y*w_y/radius + 2*phiy_y*v/(radius*radius) - v_y*w_y/(radius*radius) + v*v_y/(radius*radius*radius) - v*w/(radius*radius*radius*radius)
                Bm[99] = -0.5*(phix*phix)/(radius*radius) - 0.5*(phiy*phiy)/(radius*radius) - phiy*v/(radius*radius*radius) + phiy_y/radius - w/(radius*radius*radius) - 0.5*(v*v)/(radius*radius*radius*radius)
                Bm[100] = -phix*v_y/radius - phix/radius - phix*w/(radius*radius) - phix_y*w_y + phix_y*v/radius
                Bm[101] = -phiy*v_y/radius - phiy/radius - phiy*w/(radius*radius) - phiy_y*w_y + phiy_y*v/radius - v_y*w_y/radius - w_y/radius - v*w/(radius*radius*radius)
                Bm[103] = -phix*w_y + phix*v/radius + u_y
                Bm[105] = -phiy*w_y + phiy*v/radius + v_y + 1 - v*w_y/radius + w/radius + (v*v)/(radius*radius)
                Bm[115] = phix_y - 0.5*u_y/radius
                Bm[116] = phix_x - 0.5*u_x/radius - 0.5/radius
                Bm[117] = -0.5*(phix*phix)/radius - 0.5*(phiy*phiy)/radius - phiy*w_y/radius + phiy_y + 1.5*v_y/radius + 1.5/radius - v*w_y/(radius*radius) + 0.5*w/(radius*radius) + 0.5*(v*v)/(radius*radius*radius)
                Bm[118] = -phiy*w_x/radius + phiy_x + 1.5*v_x/radius - v*w_x/(radius*radius)
                Bm[119] = -phix*phix_y - phiy*phiy_y - phiy*v_y/radius - phiy/radius - phiy_y*v/radius - 0.5*w_y/radius - v*v_y/(radius*radius) - 0.5*v/(radius*radius)
                Bm[120] = -phix*phix_x - phiy*phiy_x - phiy*v_x/radius - phiy_x*v/radius - 0.5*w_x/radius - v*v_x/(radius*radius)
                Bm[121] = phix*phix_x/radius + phiy*phiy_x/radius - phiy_x*w_y/radius + 2*phiy_x*v/(radius*radius) - phiy_y*w_x/radius - v_x*w_y/(radius*radius) - v_y*w_x/(radius*radius) - 0.5*w_x/(radius*radius) + v*v_x/(radius*radius*radius)
                Bm[122] = phiy_x/radius + 0.5*v_x/(radius*radius)
                Bm[123] = -phix*v_x/radius - phix_x*w_y + phix_x*v/radius - phix_y*w_x
                Bm[124] = -phiy*v_x/radius - phiy_x*w_y + phiy_x*v/radius - phiy_y*w_x - v_x*w_y/radius - v_y*w_x/radius - w_x/radius
                Bm[125] = -phix*w_y + phix*v/radius + u_y
                Bm[126] = -phix*w_x + u_x + 1
                Bm[127] = -phiy*w_y + phiy*v/radius + v_y + 1 - v*w_y/radius + w/radius + (v*v)/(radius*radius)
                Bm[128] = -phiy*w_x + v_x - v*w_x/radius
                Bm[138] = -c1*phix_x + c1*v_xx*w_y - c1*v_y*w_xx - c1*w_xx - c1*v*v_xx/radius - c1*w*w_xx/radius
                Bm[139] = c1*v_x*w_xx - c1*v_xx*w_x
                Bm[140] = c1*phiy*w_x/radius - c1*phiy_x - c1*u_xx*w_y + c1*u_y*w_xx + c1*u_xx*v/radius - 2*c1*v_x/radius + c1*v*w_x/(radius*radius)
                Bm[141] = -c1*u_x*w_xx + c1*u_xx*w_x - c1*w_xx
                Bm[142] = c1*phix*phix_x + c1*phiy*phiy_x + c1*phiy*v_x/radius + c1*phiy_x*v/radius + c1*u_xx*v_y + c1*u_xx - c1*u_y*v_xx + c1*u_xx*w/radius + c1*v*v_x/(radius*radius)
                Bm[143] = c1*u_x*v_xx - c1*u_xx*v_x + c1*v_xx
                Bm[144] = c1*phiy_x*w_x/radius - c1*u_x*v_xx/radius + c1*u_xx*v_x/radius - c1*v_xx/radius + c1*v_x*w_x/(radius*radius)
                Bm[145] = -c1*u_x*w_xx/radius + c1*u_xx*w_x/radius - c1*w_xx/radius
                Bm[146] = c1*phix_x*w_x
                Bm[147] = c1*phiy_x*w_x + c1*v_x*w_x/radius
                Bm[148] = c1*phix*w_x - c1*u_x - c1
                Bm[150] = c1*phiy*w_x - c1*v_x + c1*v*w_x/radius
                Bm[152] = -c1*u_x*v_y - c1*u_x + c1*u_y*v_x - c1*v_y - c1 - c1*u_x*w/radius - c1*w/radius
                Bm[155] = -c1*v_x*w_y + c1*v_y*w_x + c1*w_x + c1*v*v_x/radius + c1*w*w_x/radius
                Bm[158] = c1*u_x*w_y - c1*u_y*w_x + c1*w_y - c1*u_x*v/radius - c1*v/radius
                Bm[161] = -c1*v_y*w_yy + c1*v_yy*w_y - c1*w_yy - c1*u_x/radius - c1*v*v_yy/radius + 2*c1*(v_y*v_y)/radius + 3*c1*v_y/radius - c1*w*w_yy/radius + 2*c1*(w_y*w_y)/radius - 3*c1*v*w_y/(radius*radius) + 3*c1*v_y*w/(radius*radius) + 2*c1*w/(radius*radius) + c1*(v*v)/(radius*radius*radius) + c1*(w*w)/(radius*radius*radius)
                Bm[162] = -c1*phix_y + c1*v_x*w_yy - c1*v_yy*w_x - c1*u_y/radius - 2*c1*v_x*v_y/radius - c1*v_x/radius - 2*c1*w_x*w_y/radius + c1*v*w_x/(radius*radius) - c1*v_x*w/(radius*radius)
                Bm[163] = c1*u_y*w_yy - c1*u_yy*w_y - 2*c1*u_y*v_y/radius - c1*u_y/radius + c1*u_yy*v/radius - c1*v_x/radius - c1*u_y*w/(radius*radius)
                Bm[164] = 0.5*c1*(phix*phix)/radius + 0.5*c1*(phiy*phiy)/radius + c1*phiy*w_y/radius - c1*phiy_y - c1*u_x*w_yy + c1*u_yy*w_x - c1*w_yy + 4*c1*u_x*v_y/radius + 3*c1*u_x/radius - 2*c1*u_y*v_x/radius + c1*v_y/radius + 3*c1*u_x*w/(radius*radius) + c1*v*w_y/(radius*radius) + c1*w/(radius*radius) - 0.5*c1*(v*v)/(radius*radius*radius)
                Bm[165] = -c1*u_y*v_yy + c1*u_yy*v_y + c1*u_yy - 2*c1*u_y*w_y/radius + c1*u_yy*w/radius - c1*w_x/radius + c1*u_y*v/(radius*radius)
                Bm[166] = c1*phix*phix_y + c1*phiy*phiy_y + c1*phiy*v_y/radius + c1*phiy/radius + c1*phiy_y*v/radius + c1*u_x*v_yy - c1*u_yy*v_x + c1*v_yy + 4*c1*u_x*w_y/radius - 2*c1*u_y*w_x/radius + 3*c1*w_y/radius - 3*c1*u_x*v/(radius*radius) + c1*v*v_y/(radius*radius) - c1*v/(radius*radius)
                Bm[167] = -c1*phix*phix_y/radius - c1*phiy*phiy_y/radius + c1*phiy*w/(radius*radius*radius) + c1*phiy_y*w_y/radius - 2*c1*phiy_y*v/(radius*radius) - c1*u_x*v_yy/radius + c1*u_yy*v_x/radius - c1*v_yy/radius - 3*c1*u_x*w_y/(radius*radius) + c1*u_y*w_x/(radius*radius) + c1*v_y*w_y/(radius*radius) - c1*w_y/(radius*radius) + 2*c1*u_x*v/(radius*radius*radius) - c1*v*v_y/(radius*radius*radius) + c1*v*w/(radius*radius*radius*radius)
                Bm[168] = 0.5*c1*(phix*phix)/(radius*radius) + 0.5*c1*(phiy*phiy)/(radius*radius) + c1*phiy*v/(radius*radius*radius) - c1*phiy_y/radius - c1*u_x*w_yy/radius + c1*u_yy*w_x/radius - c1*w_yy/radius + 3*c1*u_x*v_y/(radius*radius) + 2*c1*u_x/(radius*radius) - c1*u_y*v_x/(radius*radius) + c1*v_y/(radius*radius) + 2*c1*u_x*w/(radius*radius*radius) + c1*w/(radius*radius*radius) + 0.5*c1*(v*v)/(radius*radius*radius*radius)
                Bm[169] = c1*phix*v_y/radius + c1*phix/radius + c1*phix*w/(radius*radius) + c1*phix_y*w_y - c1*phix_y*v/radius
                Bm[170] = c1*phiy*v_y/radius + c1*phiy/radius + c1*phiy*w/(radius*radius) + c1*phiy_y*w_y - c1*phiy_y*v/radius + c1*v_y*w_y/radius + c1*w_y/radius + c1*v*w/(radius*radius*radius)
                Bm[172] = c1*phix*w_y - c1*phix*v/radius - c1*u_y
                Bm[174] = c1*phiy*w_y - c1*phiy*v/radius - c1*v_y - c1 + c1*v*w_y/radius - c1*w/radius - c1*(v*v)/(radius*radius)
                Bm[176] = -c1*u_x*v_y - c1*u_x + c1*u_y*v_x - c1*v_y - c1 - c1*u_x*w/radius - c1*w/radius
                Bm[179] = -c1*v_x*w_y + c1*v_y*w_x + c1*w_x + c1*v*v_x/radius + c1*w*w_x/radius
                Bm[182] = c1*u_x*w_y - c1*u_y*w_x + c1*w_y - c1*u_x*v/radius - c1*v/radius
                Bm[184] = -c1*phix_y + 2*c1*v_xy*w_y - 2*c1*v_y*w_xy - 2*c1*w_xy - 2*c1*v*v_xy/radius + 2*c1*v_x*v_y/radius + 2*c1*v_x/radius - 2*c1*w*w_xy/radius + 2*c1*w_x*w_y/radius - 2*c1*v*w_x/(radius*radius) + 2*c1*v_x*w/(radius*radius)
                Bm[185] = -c1*phix_x + 2*c1*v_x*w_xy - 2*c1*v_xy*w_x - 2*c1*(v_x*v_x)/radius - 2*c1*(w_x*w_x)/radius
                Bm[186] = 0.5*c1*(phix*phix)/radius + 0.5*c1*(phiy*phiy)/radius + c1*phiy*w_y/radius - c1*phiy_y - 2*c1*u_xy*w_y + 2*c1*u_y*w_xy + 2*c1*u_x*v_y/radius + 2*c1*u_x/radius + 2*c1*u_xy*v/radius - 4*c1*u_y*v_x/radius + 2*c1*u_x*w/(radius*radius) + c1*v*w_y/(radius*radius) + c1*w/(radius*radius) - 0.5*c1*(v*v)/(radius*radius*radius)
                Bm[187] = c1*phiy*w_x/radius - c1*phiy_x - 2*c1*u_x*w_xy + 2*c1*u_xy*w_x - 2*c1*w_xy + 2*c1*u_x*v_x/radius + c1*v*w_x/(radius*radius)
                Bm[188] = c1*phix*phix_y + c1*phiy*phiy_y + c1*phiy*v_y/radius + c1*phiy/radius + c1*phiy_y*v/radius + 2*c1*u_xy*v_y + 2*c1*u_xy - 2*c1*u_y*v_xy + 2*c1*u_x*w_y/radius + 2*c1*u_xy*w/radius - 4*c1*u_y*w_x/radius + 2*c1*w_y/radius - 2*c1*u_x*v/(radius*radius) + c1*v*v_y/(radius*radius) - c1*v/(radius*radius)
                Bm[189] = c1*phix*phix_x + c1*phiy*phiy_x + c1*phiy*v_x/radius + c1*phiy_x*v/radius + 2*c1*u_x*v_xy - 2*c1*u_xy*v_x + 2*c1*v_xy + 2*c1*u_x*w_x/radius + 2*c1*w_x/radius + c1*v*v_x/(radius*radius)
                Bm[190] = -c1*phix*phix_x/radius - c1*phiy*phiy_x/radius + c1*phiy_x*w_y/radius - 2*c1*phiy_x*v/(radius*radius) + c1*phiy_y*w_x/radius - 2*c1*u_x*v_xy/radius + 2*c1*u_xy*v_x/radius - 2*c1*v_xy/radius - 2*c1*u_x*w_x/(radius*radius) + c1*v_x*w_y/(radius*radius) + c1*v_y*w_x/(radius*radius) - c1*w_x/(radius*radius) - c1*v*v_x/(radius*radius*radius)
                Bm[191] = -c1*phiy_x/radius - 2*c1*u_x*w_xy/radius + 2*c1*u_xy*w_x/radius - 2*c1*w_xy/radius + 2*c1*u_x*v_x/(radius*radius) + c1*v_x/(radius*radius)
                Bm[192] = c1*phix*v_x/radius + c1*phix_x*w_y - c1*phix_x*v/radius + c1*phix_y*w_x
                Bm[193] = c1*phiy*v_x/radius + c1*phiy_x*w_y - c1*phiy_x*v/radius + c1*phiy_y*w_x + c1*v_x*w_y/radius + c1*v_y*w_x/radius + c1*w_x/radius
                Bm[194] = c1*phix*w_y - c1*phix*v/radius - c1*u_y
                Bm[195] = c1*phix*w_x - c1*u_x - c1
                Bm[196] = c1*phiy*w_y - c1*phiy*v/radius - c1*v_y - c1 + c1*v*w_y/radius - c1*w/radius - c1*(v*v)/(radius*radius)
                Bm[197] = c1*phiy*w_x - c1*v_x + c1*v*w_x/radius
                Bm[200] = -2*c1*u_x*v_y - 2*c1*u_x + 2*c1*u_y*v_x - 2*c1*v_y - 2*c1 - 2*c1*u_x*w/radius - 2*c1*w/radius
                Bm[203] = -2*c1*v_x*w_y + 2*c1*v_y*w_x + 2*c1*w_x + 2*c1*v*v_x/radius + 2*c1*w*w_x/radius
                Bm[206] = 2*c1*u_x*w_y - 2*c1*u_y*w_x + 2*c1*w_y - 2*c1*u_x*v/radius - 2*c1*v/radius
                Bm[208] = phix
                Bm[210] = phiy + v/radius
                Bm[212] = -0.5*(phix*phix) - 0.5*(phiy*phiy) - phiy*v/radius + 1 - 0.5*(v*v)/(radius*radius)
                Bm[213] = 0.5*(phix*phix)/radius + 0.5*(phiy*phiy)/radius - phiy*w_y/radius + 2*phiy*v/(radius*radius) + v_y/radius - v*w_y/(radius*radius) + w/(radius*radius) + 1.5*(v*v)/(radius*radius*radius)
                Bm[214] = phiy/radius + v/(radius*radius)
                Bm[215] = -phix*w_y + phix*v/radius + u_y
                Bm[216] = -phiy*w_y + phiy*v/radius + v_y + 1 - v*w_y/radius + w/radius + (v*v)/(radius*radius)
                Bm[230] = phix
                Bm[232] = phiy + v/radius
                Bm[234] = -0.5*(phix*phix) - 0.5*(phiy*phiy) - phiy*v/radius + 1 - 0.5*(v*v)/(radius*radius)
                Bm[236] = -phiy*w_x/radius + v_x/radius - v*w_x/(radius*radius)
                Bm[238] = -phix*w_x + u_x + 1
                Bm[239] = -phiy*w_x + v_x - v*w_x/radius
                Bm[254] = -3*c1*phix
                Bm[256] = -3*c1*phiy - 3*c1*v/radius
                Bm[258] = 1.5*c1*(phix*phix) + 1.5*c1*(phiy*phiy) + 3*c1*phiy*v/radius - 3*c1 + 1.5*c1*(v*v)/(radius*radius)
                Bm[259] = -1.5*c1*(phix*phix)/radius - 1.5*c1*(phiy*phiy)/radius + 3*c1*phiy*w_y/radius - 6*c1*phiy*v/(radius*radius) - 3*c1*v_y/radius + 3*c1*v*w_y/(radius*radius) - 3*c1*w/(radius*radius) - 4.5*c1*(v*v)/(radius*radius*radius)
                Bm[260] = -3*c1*phiy/radius - 3*c1*v/(radius*radius)
                Bm[261] = 3*c1*phix*w_y - 3*c1*phix*v/radius - 3*c1*u_y
                Bm[262] = 3*c1*phiy*w_y - 3*c1*phiy*v/radius - 3*c1*v_y - 3*c1 + 3*c1*v*w_y/radius - 3*c1*w/radius - 3*c1*(v*v)/(radius*radius)
                Bm[276] = -3*c1*phix
                Bm[278] = -3*c1*phiy - 3*c1*v/radius
                Bm[280] = 1.5*c1*(phix*phix) + 1.5*c1*(phiy*phiy) + 3*c1*phiy*v/radius - 3*c1 + 1.5*c1*(v*v)/(radius*radius)
                Bm[282] = 3*c1*phiy*w_x/radius - 3*c1*v_x/radius + 3*c1*v*w_x/(radius*radius)
                Bm[284] = 3*c1*phix*w_x - 3*c1*u_x - 3*c1
                Bm[285] = 3*c1*phiy*w_x - 3*c1*v_x + 3*c1*v*w_x/radius

                # K = [B]^T [F] [B]
                for r in range(NE):
                    for q in range(23):
                        FB[r*23 + q] = 0.
                        for s in range(NE):
                            FB[r*23 + q] += F[r*NE + s]*Bm[s*23 + q]
                for p in range(23):
                    for q in range(23):
                        K[p*23 + q] = 0.
                        for r in range(NE):
                            K[p*23 + q] += Bm[r*23 + p]*FB[r*23 + q]

                # stresses of the strains and of the linear strains
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x - phiy*v_x*w_x/radius + phiy_x*v_x - phiy_x*v*w_x/radius + (v_x*v_x)/radius - v*v_x*w_x/(radius*radius)
                e[4] = -0.5*(phix*phix)*v_y/radius - 0.5*(phix*phix)/radius - 0.5*(phix*phix)*w/(radius*radius) - phix*phix_y*w_y + phix*phix_y*v/radius + phix_y*u_y - 0.5*(phiy*phiy)*v_y/radius - 0.5*(phiy*phiy)/radius - 0.5*(phiy*phiy)*w/(radius*radius) - phiy*phiy_y*w_y + phiy*phiy_y*v/radius - phiy*v_y*w_y/radius - phiy*w_y/radius - phiy*v*w/(radius*radius*radius) + phiy_y*v_y + phiy_y - phiy_y*v*w_y/radius + phiy_y*w/radius + phiy_y*(v*v)/(radius*radius) - 0.5*(u_y*u_y)/radius + 0.5*(v_y*v_y)/radius + v_y/radius - 0.5*(w_y*w_y)/radius - v*v_y*w_y/(radius*radius) + 0.5*(v*v)*v_y/(radius*radius*radius) - 0.5*(w*w)/(radius*radius*radius) - 0.5*(v*v)*w/(radius*radius*radius*radius)
                e[5] = -0.5*(phix*phix)*v_x/radius - phix*phix_x*w_y + phix*phix_x*v/radius - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - 0.5*(phiy*phiy)*v_x/radius - phiy*phiy_x*w_y + phiy*phiy_x*v/radius - phiy*phiy_y*w_x - phiy*v_x*w_y/radius - phiy*v_y*w_x/radius - phiy*w_x/radius + phiy_x*v_y + phiy_x - phiy_x*v*w_y/radius + phiy_x*w/radius + phiy_x*(v*v)/(radius*radius) + phiy_y*v_x - phiy_y*v*w_x/radius - 0.5*u_x*u_y/radius - 0.5*u_y/radius + 1.5*v_x*v_y/radius + 1.5*v_x/radius - 0.5*w_x*w_y/radius - v*v_x*w_y/(radius*radius) - v*v_y*w_x/(radius*radius) - 0.5*v*w_x/(radius*radius) + 0.5*v_x*w/(radius*radius) + 0.5*(v*v)*v_x/(radius*radius*radius)
                e[6] = c1*phix*phix_x*w_x - c1*phix_x*u_x - c1*phix_x + c1*phiy*phiy_x*w_x + c1*phiy*v_x*w_x/radius - c1*phiy_x*v_x + c1*phiy_x*v*w_x/radius + c1*u_x*v_xx*w_y - c1*u_x*v_y*w_xx - c1*u_x*w_xx - c1*u_xx*v_x*w_y + c1*u_xx*v_y*w_x + c1*u_xx*w_x + c1*u_y*v_x*w_xx - c1*u_y*v_xx*w_x + c1*v_xx*w_y - c1*v_y*w_xx - c1*w_xx - c1*u_x*v*v_xx/radius - c1*u_x*w*w_xx/radius + c1*u_xx*v*v_x/radius + c1*u_xx*w*w_x/radius - c1*v*v_xx/radius - c1*(v_x*v_x)/radius - c1*w*w_xx/radius + c1*v*v_x*w_x/(radius*radius)
                e[7] = 0.5*c1*(phix*phix)*v_y/radius + 0.5*c1*(phix*phix)/radius + 0.5*c1*(phix*phix)*w/(radius*radius) + c1*phix*phix_y*w_y - c1*phix*phix_y*v/radius - c1*phix_y*u_y + 0.5*c1*(phiy*phiy)*v_y/radius + 0.5*c1*(phiy*phiy)/radius + 0.5*c1*(phiy*phiy)*w/(radius*radius) + c1*phiy*phiy_y*w_y - c1*phiy*phiy_y*v/radius + c1*phiy*v_y*w_y/radius + c1*phiy*w_y/radius + c1*phiy*v*w/(radius*radius*radius) - c1*phiy_y*v_y - c1*phiy_y + c1*phiy_y*v*w_y/radius - c1*phiy_y*w/radius - c1*phiy_y*(v*v)/(radius*radius) - c1*u_x*v_y*w_yy + c1*u_x*v_yy*w_y - c1*u_x*w_yy + c1*u_y*v_x*w_yy - c1*u_y*v_yy*w_x - c1*u_yy*v_x*w_y + c1*u_yy*v_y*w_x + c1*u_yy*w_x - c1*v_y*w_yy + c1*v_yy*w_y - c1*w_yy - 0.5*c1*(u_x*u_x)/radius - c1*u_x*v*v_yy/radius + 2*c1*u_x*(v_y*v_y)/radius + 3*c1*u_x*v_y/radius - c1*u_x*w*w_yy/radius + 2*c1*u_x*(w_y*w_y)/radius - 0.5*c1*(u_y*u_y)/radius - 2*c1*u_y*v_x*v_y/radius - c1*u_y*v_x/radius - 2*c1*u_y*w_x*w_y/radius + c1*u_yy*v*v_x/radius + c1*u_yy*w*w_x/radius - c1*v*v_yy/radius - 0.5*c1*(v_x*v_x)/radius + 0.5*c1*(v_y*v_y)/radius - c1*w*w_yy/radius - 0.5*c1*(w_x*w_x)/radius + 1.5*c1*(w_y*w_y)/radius - 3*c1*u_x*v*w_y/(radius*radius) + 3*c1*u_x*v_y*w/(radius*radius) + 2*c1*u_x*w/(radius*radius) + c1*u_y*v*w_x/(radius*radius) - c1*u_y*v_x*w/(radius*radius) + c1*v*v_y*w_y/(radius*radius) - c1*v*w_y/(radius*radius) + c1*v_y*w/(radius*radius) + c1*u_x*(v*v)/(radius*radius*radius) + c1*u_x*(w*w)/(radius*radius*radius) - 0.5*c1*(v*v)*v_y/(radius*radius*radius) + 0.5*c1*(w*w)/(radius*radius*radius) + 0.5*c1*(v*v)*w/(radius*radius*radius*radius)
                e[8] = 0.5*c1*(phix*phix)*v_x/radius + c1*phix*phix_x*w_y - c1*phix*phix_x*v/radius + c1*phix*phix_y*w_x - c1*phix_x*u_y - c1*phix_y*u_x - c1*phix_y + 0.5*c1*(phiy*phiy)*v_x/radius + c1*phiy*phiy_x*w_y - c1*phiy*phiy_x*v/radius + c1*phiy*phiy_y*w_x + c1*phiy*v_x*w_y/radius + c1*phiy*v_y*w_x/radius + c1*phiy*w_x/radius - c1*phiy_x*v_y - c1*phiy_x + c1*phiy_x*v*w_y/radius - c1*phiy_x*w/radius - c1*phiy_x*(v*v)/(radius*radius) - c1*phiy_y*v_x + c1*phiy_y*v*w_x/radius + 2*c1*u_x*v_xy*w_y - 2*c1*u_x*v_y*w_xy - 2*c1*u_x*w_xy - 2*c1*u_xy*v_x*w_y + 2*c1*u_xy*v_y*w_x + 2*c1*u_xy*w_x + 2*c1*u_y*v_x*w_xy - 2*c1*u_y*v_xy*w_x + 2*c1*v_xy*w_y - 2*c1*v_y*w_xy - 2*c1*w_xy - 2*c1*u_x*v*v_xy/radius + 2*c1*u_x*v_x*v_y/radius + 2*c1*u_x*v_x/radius - 2*c1*u_x*w*w_xy/radius + 2*c1*u_x*w_x*w_y/radius + 2*c1*u_xy*v*v_x/radius + 2*c1*u_xy*w*w_x/radius - 2*c1*u_y*(v_x*v_x)/radius - 2*c1*u_y*(w_x*w_x)/radius - 2*c1*v*v_xy/radius - 2*c1*w*w_xy/radius + 2*c1*w_x*w_y/radius - 2*c1*u_x*v*w_x/(radius*radius) + 2*c1*u_x*v_x*w/(radius*radius) + c1*v*v_x*w_y/(radius*radius) + c1*v*v_y*w_x/(radius*radius) - c1*v*w_x/(radius*radius) + c1*v_x*w/(radius*radius) - 0.5*c1*(v*v)*v_x/(radius*radius*radius)
                e[9] = -0.5*(phix*phix)*w_y + 0.5*(phix*phix)*v/radius + phix*u_y - 0.5*(phiy*phiy)*w_y + 0.5*(phiy*phiy)*v/radius + phiy*v_y + phiy - phiy*v*w_y/radius + phiy*w/radius + phiy*(v*v)/(radius*radius) + w_y + v*v_y/radius - 0.5*(v*v)*w_y/(radius*radius) + v*w/(radius*radius) + 0.5*(v*v*v)/(radius*radius*radius)
                e[10] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x - phiy*v*w_x/radius + w_x + v*v_x/radius - 0.5*(v*v)*w_x/(radius*radius)
                e[11] = 1.5*c1*(phix*phix)*w_y - 1.5*c1*(phix*phix)*v/radius - 3*c1*phix*u_y + 1.5*c1*(phiy*phiy)*w_y - 1.5*c1*(phiy*phiy)*v/radius - 3*c1*phiy*v_y - 3*c1*phiy + 3*c1*phiy*v*w_y/radius - 3*c1*phiy*w/radius - 3*c1*phiy*(v*v)/(radius*radius) - 3*c1*w_y - 3*c1*v*v_y/radius + 1.5*c1*(v*v)*w_y/(radius*radius) - 3*c1*v*w/(radius*radius) - 1.5*c1*(v*v*v)/(radius*radius*radius)
                e[12] = 1.5*c1*(phix*phix)*w_x - 3*c1*phix*u_x - 3*c1*phix + 1.5*c1*(phiy*phiy)*w_x - 3*c1*phiy*v_x + 3*c1*phiy*v*w_x/radius - 3*c1*w_x - 3*c1*v*v_x/radius + 1.5*c1*(v*v)*w_x/(radius*radius)
                eL[0] = u_x
                eL[1] = v_y + w/radius
                eL[2] = u_y + v_x
                eL[3] = phix_x
                eL[4] = phiy_y + v_y/radius
                eL[5] = phix_y + phiy_x - 0.5*u_y/radius + 1.5*v_x/radius
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
                K[0] += c1*sgL[7]/radius - c1*sg[7]/radius - sgL[0] + sg[0]
                K[1] += -sgL[2] + sg[2] + 0.5*sgL[5]/radius - 0.5*sg[5]/radius
                K[2] += -2*c1*sgL[8]/radius + 2*c1*sg[8]*v_y/radius + 2*c1*sg[8]/radius + 2*c1*sg[8]*w/(radius*radius)
                K[3] += -c1*sg[6]*w_xx - c1*sg[7]*w_yy - 2*c1*sg[8]*w_xy - 3*c1*sgL[7]/radius + 4*c1*sg[7]*v_y/radius + 3*c1*sg[7]/radius + 2*c1*sg[8]*v_x/radius + 3*c1*sg[7]*w/(radius*radius)
                K[4] += 2*c1*sg[8]*w_y/radius - 2*c1*sg[8]*v/(radius*radius)
                K[5] += c1*sg[6]*v_xx + c1*sg[7]*v_yy + 2*c1*sg[8]*v_xy + 4*c1*sg[7]*w_y/radius + 2*c1*sg[8]*w_x/radius - 3*c1*sg[7]*v/(radius*radius)
                K[6] += -c1*sg[6]*v_xx/radius - c1*sg[7]*v_yy/radius - 2*c1*sg[8]*v_xy/radius - 3*c1*sg[7]*w_y/(radius*radius) - 2*c1*sg[8]*w_x/(radius*radius) + 2*c1*sg[7]*v/(radius*radius*radius)
                K[7] += -c1*sg[6]*w_xx/radius - c1*sg[7]*w_yy/radius - 2*c1*sg[8]*w_xy/radius - 2*c1*sgL[7]/(radius*radius) + 3*c1*sg[7]*v_y/(radius*radius) + 2*c1*sg[7]/(radius*radius) + 2*c1*sg[8]*v_x/(radius*radius) + 2*c1*sg[7]*w/(radius*radius*radius)
                K[8] += 3*c1*sgL[12] - 3*c1*sg[12] - sgL[10] + sg[10]
                K[10] += c1*sgL[6] - c1*sg[6] - sgL[3] + sg[3]
                K[11] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[14] += c1*sgL[6] - c1*sg[6]*v_y - c1*sg[6] - c1*sg[6]*w/radius
                K[15] += c1*sgL[7] - c1*sg[7]*v_y - c1*sg[7] - c1*sg[7]*w/radius
                K[16] += 2*c1*sgL[8] - 2*c1*sg[8]*v_y - 2*c1*sg[8] - 2*c1*sg[8]*w/radius
                K[20] += c1*sg[6]*w_y - c1*sg[6]*v/radius
                K[21] += c1*sg[7]*w_y - c1*sg[7]*v/radius
                K[22] += 2*c1*sg[8]*w_y - 2*c1*sg[8]*v/radius
                K[23] += -sgL[2] + sg[2] + 0.5*sgL[5]/radius - 0.5*sg[5]/radius
                K[24] += c1*sgL[7]/radius - c1*sg[7]/radius - sgL[1] + sg[1] + sgL[4]/radius - sg[4]/radius
                K[25] += c1*sg[6]*w_xx + c1*sg[7]*w_yy + 2*c1*sg[8]*w_xy + c1*sgL[7]/radius - 2*c1*sg[7]*v_y/radius - c1*sg[7]/radius - 4*c1*sg[8]*v_x/radius - c1*sg[7]*w/(radius*radius)
                K[26] += -2*c1*sg[7]*v_x/radius
                K[27] += -c1*sg[6]*v_xx - c1*sg[7]*v_yy - 2*c1*sg[8]*v_xy - 2*c1*sg[7]*w_y/radius - 4*c1*sg[8]*w_x/radius + c1*sg[7]*v/(radius*radius)
                K[28] += -2*c1*sg[7]*w_x/radius
                K[29] += c1*sg[7]*w_x/(radius*radius)
                K[30] += -c1*sg[7]*v_x/(radius*radius)
                K[31] += 3*c1*sgL[11] - 3*c1*sg[11] - sgL[9] + sg[9]
                K[33] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[34] += c1*sgL[7] - c1*sg[7] - sgL[4] + sg[4]
                K[37] += c1*sg[6]*v_x
                K[38] += c1*sg[7]*v_x
                K[39] += 2*c1*sg[8]*v_x
                K[43] += -c1*sg[6]*w_x
                K[44] += -c1*sg[7]*w_x
                K[45] += -2*c1*sg[8]*w_x
                K[46] += -2*c1*sgL[8]/radius + 2*c1*sg[8]*v_y/radius + 2*c1*sg[8]/radius + 2*c1*sg[8]*w/(radius*radius)
                K[47] += c1*sg[6]*w_xx + c1*sg[7]*w_yy + 2*c1*sg[8]*w_xy + c1*sgL[7]/radius - 2*c1*sg[7]*v_y/radius - c1*sg[7]/radius - 4*c1*sg[8]*v_x/radius - c1*sg[7]*w/(radius*radius)
                K[48] += 2*c1*sgL[6]/radius + c1*sgL[7]/radius - 2*c1*sg[6]/radius - c1*sg[7]/radius - 4*c1*sg[8]*u_y/radius - sgL[0] + sg[0] - 2*sgL[3]/radius + 2*sg[3]/radius
                K[49] += -2*c1*sg[7]*u_y/radius + 2*c1*sg[8]*u_x/radius - sgL[2] + sg[2] - 1.5*sgL[5]/radius + 1.5*sg[5]/radius
                K[50] += c1*phiy*sg[6]/radius + c1*sg[6]*v/(radius*radius) - phiy*sg[3]/radius - sg[3]*v/(radius*radius)
                K[51] += c1*phiy*sg[8]/radius - c1*sg[6]*u_xx - c1*sg[7]*u_yy - 2*c1*sg[8]*u_xy + c1*sg[8]*v/(radius*radius) - phiy*sg[5]/radius - sg[5]*v/(radius*radius)
                K[52] += 3*c1*sgL[12]/radius - 3*c1*sg[12]/radius + c1*sg[6]*u_xx/radius + c1*sg[7]*u_yy/radius + 2*c1*sg[8]*u_xy/radius + c1*sg[6]*w_x/(radius*radius) + c1*sg[8]*w_y/(radius*radius) - c1*sg[8]*v/(radius*radius*radius) - sgL[10]/radius + sg[10]/radius - sg[3]*w_x/(radius*radius) - sg[5]*w_y/(radius*radius) + sg[5]*v/(radius*radius*radius)
                K[53] += -c1*sgL[8]/(radius*radius) - c1*sg[7]*u_y/(radius*radius) + 2*c1*sg[8]*u_x/(radius*radius) + c1*sg[8]/(radius*radius) - sgL[2]/radius + sg[2]/radius - 0.5*sgL[5]/(radius*radius) + 0.5*sg[5]/(radius*radius)
                K[54] += c1*phix*sg[8]/radius - phix*sg[5]/radius
                K[55] += c1*phiy*sg[8]/radius + 3*c1*sgL[12] - 3*c1*sg[12] + c1*sg[6]*w_x/radius + c1*sg[8]*w_y/radius - phiy*sg[5]/radius - sgL[10] + sg[10] - sg[3]*w_x/radius - sg[5]*w_y/radius
                K[58] += c1*sgL[6] - c1*sg[6] - sgL[3] + sg[3]
                K[59] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[60] += c1*sg[6]*u_y
                K[61] += c1*sg[7]*u_y
                K[62] += 2*c1*sg[8]*u_y
                K[63] += -c1*sg[6]*w_y + c1*sg[6]*v/radius
                K[64] += -c1*sg[7]*w_y + c1*sg[7]*v/radius
                K[65] += -2*c1*sg[8]*w_y + 2*c1*sg[8]*v/radius
                K[69] += -c1*sg[6]*w_xx - c1*sg[7]*w_yy - 2*c1*sg[8]*w_xy - 3*c1*sgL[7]/radius + 4*c1*sg[7]*v_y/radius + 3*c1*sg[7]/radius + 2*c1*sg[8]*v_x/radius + 3*c1*sg[7]*w/(radius*radius)
                K[70] += -2*c1*sg[7]*v_x/radius
                K[71] += -2*c1*sg[7]*u_y/radius + 2*c1*sg[8]*u_x/radius - sgL[2] + sg[2] - 1.5*sgL[5]/radius + 1.5*sg[5]/radius
                K[72] += -c1*sgL[7]/radius + 4*c1*sg[7]*u_x/radius + c1*sg[7]/radius - sgL[1] + sg[1] - sgL[4]/radius + sg[4]/radius
                K[73] += c1*phiy*sg[8]/radius + c1*sg[6]*u_xx + c1*sg[7]*u_yy + 2*c1*sg[8]*u_xy + c1*sg[8]*v/(radius*radius) - phiy*sg[5]/radius - sg[5]*v/(radius*radius)
                K[74] += c1*phiy*sg[7]/radius + c1*sg[7]*v/(radius*radius) - phiy*sg[4]/radius - sg[4]*v/(radius*radius)
                K[75] += 3*c1*sgL[11]/radius - 3*c1*sg[11]/radius + c1*sg[7]*w_y/(radius*radius) + c1*sg[8]*w_x/(radius*radius) - c1*sg[7]*v/(radius*radius*radius) - sgL[9]/radius + sg[9]/radius - sg[4]*w_y/(radius*radius) - sg[5]*w_x/(radius*radius) + sg[4]*v/(radius*radius*radius)
                K[76] += -c1*sgL[7]/(radius*radius) + 3*c1*sg[7]*u_x/(radius*radius) + c1*sg[7]/(radius*radius) - sgL[1]/radius + sg[1]/radius
                K[77] += c1*phix*sg[7]/radius - phix*sg[4]/radius
                K[78] += c1*phiy*sg[7]/radius + 3*c1*sgL[11] - 3*c1*sg[11] + c1*sg[7]*w_y/radius + c1*sg[8]*w_x/radius - phiy*sg[4]/radius - sgL[9] + sg[9] - sg[4]*w_y/radius - sg[5]*w_x/radius
                K[81] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[82] += c1*sgL[7] - c1*sg[7] - sgL[4] + sg[4]
                K[83] += c1*sgL[6] - c1*sg[6]*u_x - c1*sg[6]
                K[84] += c1*sgL[7] - c1*sg[7]*u_x - c1*sg[7]
                K[85] += 2*c1*sgL[8] - 2*c1*sg[8]*u_x - 2*c1*sg[8]
                K[86] += c1*sg[6]*w_x
                K[87] += c1*sg[7]*w_x
                K[88] += 2*c1*sg[8]*w_x
                K[92] += 2*c1*sg[8]*w_y/radius - 2*c1*sg[8]*v/(radius*radius)
                K[93] += -c1*sg[6]*v_xx - c1*sg[7]*v_yy - 2*c1*sg[8]*v_xy - 2*c1*sg[7]*w_y/radius - 4*c1*sg[8]*w_x/radius + c1*sg[7]*v/(radius*radius)
                K[94] += c1*phiy*sg[6]/radius + c1*sg[6]*v/(radius*radius) - phiy*sg[3]/radius - sg[3]*v/(radius*radius)
                K[95] += c1*phiy*sg[8]/radius + c1*sg[6]*u_xx + c1*sg[7]*u_yy + 2*c1*sg[8]*u_xy + c1*sg[8]*v/(radius*radius) - phiy*sg[5]/radius - sg[5]*v/(radius*radius)
                K[96] += c1*sgL[7]/radius - c1*sg[7]/radius - 4*c1*sg[8]*u_y/radius - sgL[0] + sg[0]
                K[97] += -2*c1*sgL[8]/radius - 2*c1*sg[7]*u_y/radius + 2*c1*sg[8]*u_x/radius + 2*c1*sg[8]/radius - sgL[2] + sg[2] + 0.5*sgL[5]/radius - 0.5*sg[5]/radius
                K[98] += 3*c1*phiy*sg[12]/radius + c1*phiy_x*sg[6]/radius + c1*phiy_y*sg[8]/radius + c1*sgL[8]/(radius*radius) + 3*c1*sg[12]*v/(radius*radius) + c1*sg[6]*v_x/(radius*radius) + c1*sg[7]*u_y/(radius*radius) - 2*c1*sg[8]*u_x/(radius*radius) + c1*sg[8]*v_y/(radius*radius) - c1*sg[8]/(radius*radius) - phiy*sg[10]/radius - phiy_x*sg[3]/radius - phiy_y*sg[5]/radius + sgL[2]/radius - sg[2]/radius + 0.5*sgL[5]/(radius*radius) - sg[10]*v/(radius*radius) - sg[3]*v_x/(radius*radius) - sg[5]*v_y/(radius*radius) - 0.5*sg[5]/(radius*radius)
                K[99] += c1*sg[6]*u_xx/radius + c1*sg[7]*u_yy/radius + 2*c1*sg[8]*u_xy/radius
                K[100] += 3*c1*phix*sg[12] + c1*phix_x*sg[6] + c1*phix_y*sg[8] - phix*sg[10] - phix_x*sg[3] - phix_y*sg[5]
                K[101] += 3*c1*phiy*sg[12] + c1*phiy_x*sg[6] + c1*phiy_y*sg[8] - c1*sgL[8]/radius + 3*c1*sg[12]*v/radius + c1*sg[6]*v_x/radius + c1*sg[8]*v_y/radius + c1*sg[8]/radius - phiy*sg[10] - phiy_x*sg[3] - phiy_y*sg[5] + sgL[5]/radius - sg[10]*v/radius - sg[3]*v_x/radius - sg[5]*v_y/radius - sg[5]/radius
                K[102] += c1*phix*sg[6] - phix*sg[3]
                K[103] += c1*phix*sg[8] - phix*sg[5]
                K[104] += c1*phiy*sg[6] + c1*sg[6]*v/radius - phiy*sg[3] - sg[3]*v/radius
                K[105] += c1*phiy*sg[8] + c1*sg[8]*v/radius - phiy*sg[5] - sg[5]*v/radius
                K[109] += -c1*sgL[6] + c1*sg[6]*v_y + c1*sg[6] + c1*sg[6]*w/radius
                K[110] += -c1*sgL[7] + c1*sg[7]*v_y + c1*sg[7] + c1*sg[7]*w/radius
                K[111] += -2*c1*sgL[8] + 2*c1*sg[8]*v_y + 2*c1*sg[8] + 2*c1*sg[8]*w/radius
                K[112] += -c1*sg[6]*u_y
                K[113] += -c1*sg[7]*u_y
                K[114] += -2*c1*sg[8]*u_y
                K[115] += c1*sg[6]*v_xx + c1*sg[7]*v_yy + 2*c1*sg[8]*v_xy + 4*c1*sg[7]*w_y/radius + 2*c1*sg[8]*w_x/radius - 3*c1*sg[7]*v/(radius*radius)
                K[116] += -2*c1*sg[7]*w_x/radius
                K[117] += c1*phiy*sg[8]/radius - c1*sg[6]*u_xx - c1*sg[7]*u_yy - 2*c1*sg[8]*u_xy + c1*sg[8]*v/(radius*radius) - phiy*sg[5]/radius - sg[5]*v/(radius*radius)
                K[118] += c1*phiy*sg[7]/radius + c1*sg[7]*v/(radius*radius) - phiy*sg[4]/radius - sg[4]*v/(radius*radius)
                K[119] += -2*c1*sgL[8]/radius - 2*c1*sg[7]*u_y/radius + 2*c1*sg[8]*u_x/radius + 2*c1*sg[8]/radius - sgL[2] + sg[2] + 0.5*sgL[5]/radius - 0.5*sg[5]/radius
                K[120] += -3*c1*sgL[7]/radius + 4*c1*sg[7]*u_x/radius + 3*c1*sg[7]/radius - sgL[1] + sg[1] + sgL[4]/radius - sg[4]/radius
                K[121] += 3*c1*phiy*sg[11]/radius + c1*phiy_x*sg[8]/radius + c1*phiy_y*sg[7]/radius + c1*sgL[7]/(radius*radius) + 3*c1*sg[11]*v/(radius*radius) - 3*c1*sg[7]*u_x/(radius*radius) + c1*sg[7]*v_y/(radius*radius) - c1*sg[7]/(radius*radius) + c1*sg[8]*v_x/(radius*radius) - phiy*sg[9]/radius - phiy_x*sg[5]/radius - phiy_y*sg[4]/radius + sgL[1]/radius - sg[1]/radius - sg[4]*v_y/(radius*radius) - sg[5]*v_x/(radius*radius) - sg[9]*v/(radius*radius)
                K[123] += 3*c1*phix*sg[11] + c1*phix_x*sg[8] + c1*phix_y*sg[7] - phix*sg[9] - phix_x*sg[5] - phix_y*sg[4]
                K[124] += 3*c1*phiy*sg[11] + c1*phiy_x*sg[8] + c1*phiy_y*sg[7] - c1*sgL[7]/radius + 3*c1*sg[11]*v/radius + c1*sg[7]*v_y/radius + c1*sg[7]/radius + c1*sg[8]*v_x/radius - phiy*sg[9] - phiy_x*sg[5] - phiy_y*sg[4] + sgL[4]/radius - sg[4]*v_y/radius - sg[4]/radius - sg[5]*v_x/radius - sg[9]*v/radius
                K[125] += c1*phix*sg[8] - phix*sg[5]
                K[126] += c1*phix*sg[7] - phix*sg[4]
                K[127] += c1*phiy*sg[8] + c1*sg[8]*v/radius - phiy*sg[5] - sg[5]*v/radius
                K[128] += c1*phiy*sg[7] + c1*sg[7]*v/radius - phiy*sg[4] - sg[4]*v/radius
                K[132] += -c1*sg[6]*v_x
                K[133] += -c1*sg[7]*v_x
                K[134] += -2*c1*sg[8]*v_x
                K[135] += -c1*sgL[6] + c1*sg[6]*u_x + c1*sg[6]
                K[136] += -c1*sgL[7] + c1*sg[7]*u_x + c1*sg[7]
                K[137] += -2*c1*sgL[8] + 2*c1*sg[8]*u_x + 2*c1*sg[8]
                K[138] += -c1*sg[6]*v_xx/radius - c1*sg[7]*v_yy/radius - 2*c1*sg[8]*v_xy/radius - 3*c1*sg[7]*w_y/(radius*radius) - 2*c1*sg[8]*w_x/(radius*radius) + 2*c1*sg[7]*v/(radius*radius*radius)
                K[139] += c1*sg[7]*w_x/(radius*radius)
                K[140] += 3*c1*sgL[12]/radius - 3*c1*sg[12]/radius + c1*sg[6]*u_xx/radius + c1*sg[7]*u_yy/radius + 2*c1*sg[8]*u_xy/radius + c1*sg[6]*w_x/(radius*radius) + c1*sg[8]*w_y/(radius*radius) - c1*sg[8]*v/(radius*radius*radius) - sgL[10]/radius + sg[10]/radius - sg[3]*w_x/(radius*radius) - sg[5]*w_y/(radius*radius) + sg[5]*v/(radius*radius*radius)
                K[141] += 3*c1*sgL[11]/radius - 3*c1*sg[11]/radius + c1*sg[7]*w_y/(radius*radius) + c1*sg[8]*w_x/(radius*radius) - c1*sg[7]*v/(radius*radius*radius) - sgL[9]/radius + sg[9]/radius - sg[4]*w_y/(radius*radius) - sg[5]*w_x/(radius*radius) + sg[4]*v/(radius*radius*radius)
                K[142] += 3*c1*phiy*sg[12]/radius + c1*phiy_x*sg[6]/radius + c1*phiy_y*sg[8]/radius + c1*sgL[8]/(radius*radius) + 3*c1*sg[12]*v/(radius*radius) + c1*sg[6]*v_x/(radius*radius) + c1*sg[7]*u_y/(radius*radius) - 2*c1*sg[8]*u_x/(radius*radius) + c1*sg[8]*v_y/(radius*radius) - c1*sg[8]/(radius*radius) - phiy*sg[10]/radius - phiy_x*sg[3]/radius - phiy_y*sg[5]/radius + sgL[2]/radius - sg[2]/radius + 0.5*sgL[5]/(radius*radius) - sg[10]*v/(radius*radius) - sg[3]*v_x/(radius*radius) - sg[5]*v_y/(radius*radius) - 0.5*sg[5]/(radius*radius)
                K[143] += 3*c1*phiy*sg[11]/radius + c1*phiy_x*sg[8]/radius + c1*phiy_y*sg[7]/radius + c1*sgL[7]/(radius*radius) + 3*c1*sg[11]*v/(radius*radius) - 3*c1*sg[7]*u_x/(radius*radius) + c1*sg[7]*v_y/(radius*radius) - c1*sg[7]/(radius*radius) + c1*sg[8]*v_x/(radius*radius) - phiy*sg[9]/radius - phiy_x*sg[5]/radius - phiy_y*sg[4]/radius + sgL[1]/radius - sg[1]/radius - sg[4]*v_y/(radius*radius) - sg[5]*v_x/(radius*radius) - sg[9]*v/(radius*radius)
                K[144] += -6*c1*phiy*sg[11]/(radius*radius) - 2*c1*phiy_x*sg[8]/(radius*radius) - 2*c1*phiy_y*sg[7]/(radius*radius) + 3*c1*sg[11]*w_y/(radius*radius) + 3*c1*sg[12]*w_x/(radius*radius) - 9*c1*sg[11]*v/(radius*radius*radius) + 2*c1*sg[7]*u_x/(radius*radius*radius) - c1*sg[7]*v_y/(radius*radius*radius) - c1*sg[8]*v_x/(radius*radius*radius) + c1*sg[7]*w/(radius*radius*radius*radius) + 2*phiy*sg[9]/(radius*radius) + 2*phiy_x*sg[5]/(radius*radius) + 2*phiy_y*sg[4]/(radius*radius) - sgL[1]/(radius*radius) - sg[10]*w_x/(radius*radius) + sg[1]/(radius*radius) - sg[9]*w_y/(radius*radius) + sg[4]*v_y/(radius*radius*radius) + sg[5]*v_x/(radius*radius*radius) + 3*sg[9]*v/(radius*radius*radius) - sg[4]*w/(radius*radius*radius*radius)
                K[145] += c1*phiy*sg[7]/(radius*radius*radius) + 3*c1*sgL[11]/(radius*radius) - 3*c1*sg[11]/(radius*radius) + c1*sg[7]*v/(radius*radius*radius*radius) - phiy*sg[4]/(radius*radius*radius) - sgL[9]/(radius*radius) + sg[9]/(radius*radius) - sg[4]*v/(radius*radius*radius*radius)
                K[146] += -3*c1*phix*sg[11]/radius - c1*phix_x*sg[8]/radius - c1*phix_y*sg[7]/radius + phix*sg[9]/radius + phix_x*sg[5]/radius + phix_y*sg[4]/radius
                K[147] += -3*c1*phiy*sg[11]/radius - c1*phiy_x*sg[8]/radius - c1*phiy_y*sg[7]/radius + 3*c1*sg[11]*w_y/radius + 3*c1*sg[12]*w_x/radius - 6*c1*sg[11]*v/(radius*radius) + c1*sg[7]*w/(radius*radius*radius) + phiy*sg[9]/radius + phiy_x*sg[5]/radius + phiy_y*sg[4]/radius - sg[10]*w_x/radius - sg[9]*w_y/radius + 2*sg[9]*v/(radius*radius) - sg[4]*w/(radius*radius*radius)
                K[148] += -c1*phix*sg[8]/radius + phix*sg[5]/radius
                K[149] += -c1*phix*sg[7]/radius + phix*sg[4]/radius
                K[150] += -c1*phiy*sg[8]/radius + c1*sg[6]*w_x/radius + c1*sg[8]*w_y/radius - 2*c1*sg[8]*v/(radius*radius) + phiy*sg[5]/radius - sg[3]*w_x/radius - sg[5]*w_y/radius + 2*sg[5]*v/(radius*radius)
                K[151] += -c1*phiy*sg[7]/radius + c1*sg[7]*w_y/radius + c1*sg[8]*w_x/radius - 2*c1*sg[7]*v/(radius*radius) + phiy*sg[4]/radius - sg[4]*w_y/radius - sg[5]*w_x/radius + 2*sg[4]*v/(radius*radius)
                K[155] += c1*sg[6]*v_x/radius
                K[156] += c1*sg[7]*v_x/radius
                K[157] += 2*c1*sg[8]*v_x/radius
                K[158] += c1*sgL[6]/radius - c1*sg[6]*u_x/radius - c1*sg[6]/radius
                K[159] += c1*sgL[7]/radius - c1*sg[7]*u_x/radius - c1*sg[7]/radius
                K[160] += 2*c1*sgL[8]/radius - 2*c1*sg[8]*u_x/radius - 2*c1*sg[8]/radius
                K[161] += -c1*sg[6]*w_xx/radius - c1*sg[7]*w_yy/radius - 2*c1*sg[8]*w_xy/radius - 2*c1*sgL[7]/(radius*radius) + 3*c1*sg[7]*v_y/(radius*radius) + 2*c1*sg[7]/(radius*radius) + 2*c1*sg[8]*v_x/(radius*radius) + 2*c1*sg[7]*w/(radius*radius*radius)
                K[162] += -c1*sg[7]*v_x/(radius*radius)
                K[163] += -c1*sgL[8]/(radius*radius) - c1*sg[7]*u_y/(radius*radius) + 2*c1*sg[8]*u_x/(radius*radius) + c1*sg[8]/(radius*radius) - sgL[2]/radius + sg[2]/radius - 0.5*sgL[5]/(radius*radius) + 0.5*sg[5]/(radius*radius)
                K[164] += -c1*sgL[7]/(radius*radius) + 3*c1*sg[7]*u_x/(radius*radius) + c1*sg[7]/(radius*radius) - sgL[1]/radius + sg[1]/radius
                K[165] += c1*sg[6]*u_xx/radius + c1*sg[7]*u_yy/radius + 2*c1*sg[8]*u_xy/radius
                K[167] += c1*phiy*sg[7]/(radius*radius*radius) + 3*c1*sgL[11]/(radius*radius) - 3*c1*sg[11]/(radius*radius) + c1*sg[7]*v/(radius*radius*radius*radius) - phiy*sg[4]/(radius*radius*radius) - sgL[9]/(radius*radius) + sg[9]/(radius*radius) - sg[4]*v/(radius*radius*radius*radius)
                K[168] += -c1*sgL[7]/(radius*radius*radius) + 2*c1*sg[7]*u_x/(radius*radius*radius) + c1*sg[7]/(radius*radius*radius) - sgL[1]/(radius*radius) + sg[1]/(radius*radius) + sgL[4]/(radius*radius*radius) - sg[4]/(radius*radius*radius)
                K[169] += c1*phix*sg[7]/(radius*radius) - phix*sg[4]/(radius*radius)
                K[170] += c1*phiy*sg[7]/(radius*radius) + 3*c1*sgL[11]/radius - 3*c1*sg[11]/radius + c1*sg[7]*v/(radius*radius*radius) - phiy*sg[4]/(radius*radius) - sgL[9]/radius + sg[9]/radius - sg[4]*v/(radius*radius*radius)
                K[173] += c1*sgL[8]/radius - c1*sg[8]/radius - sgL[5]/radius + sg[5]/radius
                K[174] += c1*sgL[7]/radius - c1*sg[7]/radius - sgL[4]/radius + sg[4]/radius
                K[175] += c1*sgL[6]/radius - c1*sg[6]*u_x/radius - c1*sg[6]/radius
                K[176] += c1*sgL[7]/radius - c1*sg[7]*u_x/radius - c1*sg[7]/radius
                K[177] += 2*c1*sgL[8]/radius - 2*c1*sg[8]*u_x/radius - 2*c1*sg[8]/radius
                K[178] += c1*sg[6]*w_x/radius
                K[179] += c1*sg[7]*w_x/radius
                K[180] += 2*c1*sg[8]*w_x/radius
                K[184] += 3*c1*sgL[12] - 3*c1*sg[12] - sgL[10] + sg[10]
                K[185] += 3*c1*sgL[11] - 3*c1*sg[11] - sgL[9] + sg[9]
                K[186] += c1*phix*sg[8]/radius - phix*sg[5]/radius
                K[187] += c1*phix*sg[7]/radius - phix*sg[4]/radius
                K[188] += 3*c1*phix*sg[12] + c1*phix_x*sg[6] + c1*phix_y*sg[8] - phix*sg[10] - phix_x*sg[3] - phix_y*sg[5]
                K[189] += 3*c1*phix*sg[11] + c1*phix_x*sg[8] + c1*phix_y*sg[7] - phix*sg[9] - phix_x*sg[5] - phix_y*sg[4]
                K[190] += -3*c1*phix*sg[11]/radius - c1*phix_x*sg[8]/radius - c1*phix_y*sg[7]/radius + phix*sg[9]/radius + phix_x*sg[5]/radius + phix_y*sg[4]/radius
                K[191] += c1*phix*sg[7]/(radius*radius) - phix*sg[4]/(radius*radius)
                K[192] += 3*c1*sg[11]*w_y + 3*c1*sg[12]*w_x - c1*sgL[7]/radius - 3*c1*sg[11]*v/radius + c1*sg[7]*v_y/radius + c1*sg[7]/radius + c1*sg[8]*v_x/radius + c1*sg[7]*w/(radius*radius) - sg[10]*w_x - sg[9]*w_y + sgL[4]/radius - sg[4]*v_y/radius - sg[4]/radius - sg[5]*v_x/radius + sg[9]*v/radius - sg[4]*w/(radius*radius)
                K[194] += c1*sg[6]*w_x + c1*sg[8]*w_y - c1*sg[8]*v/radius - sg[3]*w_x - sg[5]*w_y + sg[5]*v/radius
                K[195] += c1*sg[7]*w_y + c1*sg[8]*w_x - c1*sg[7]*v/radius - sg[4]*w_y - sg[5]*w_x + sg[4]*v/radius
                K[209] += c1*phiy*sg[8]/radius + 3*c1*sgL[12] - 3*c1*sg[12] + c1*sg[6]*w_x/radius + c1*sg[8]*w_y/radius - phiy*sg[5]/radius - sgL[10] + sg[10] - sg[3]*w_x/radius - sg[5]*w_y/radius
                K[210] += c1*phiy*sg[7]/radius + 3*c1*sgL[11] - 3*c1*sg[11] + c1*sg[7]*w_y/radius + c1*sg[8]*w_x/radius - phiy*sg[4]/radius - sgL[9] + sg[9] - sg[4]*w_y/radius - sg[5]*w_x/radius
                K[211] += 3*c1*phiy*sg[12] + c1*phiy_x*sg[6] + c1*phiy_y*sg[8] - c1*sgL[8]/radius + 3*c1*sg[12]*v/radius + c1*sg[6]*v_x/radius + c1*sg[8]*v_y/radius + c1*sg[8]/radius - phiy*sg[10] - phiy_x*sg[3] - phiy_y*sg[5] + sgL[5]/radius - sg[10]*v/radius - sg[3]*v_x/radius - sg[5]*v_y/radius - sg[5]/radius
                K[212] += 3*c1*phiy*sg[11] + c1*phiy_x*sg[8] + c1*phiy_y*sg[7] - c1*sgL[7]/radius + 3*c1*sg[11]*v/radius + c1*sg[7]*v_y/radius + c1*sg[7]/radius + c1*sg[8]*v_x/radius - phiy*sg[9] - phiy_x*sg[5] - phiy_y*sg[4] + sgL[4]/radius - sg[4]*v_y/radius - sg[4]/radius - sg[5]*v_x/radius - sg[9]*v/radius
                K[213] += -3*c1*phiy*sg[11]/radius - c1*phiy_x*sg[8]/radius - c1*phiy_y*sg[7]/radius + 3*c1*sg[11]*w_y/radius + 3*c1*sg[12]*w_x/radius - 6*c1*sg[11]*v/(radius*radius) + c1*sg[7]*w/(radius*radius*radius) + phiy*sg[9]/radius + phiy_x*sg[5]/radius + phiy_y*sg[4]/radius - sg[10]*w_x/radius - sg[9]*w_y/radius + 2*sg[9]*v/(radius*radius) - sg[4]*w/(radius*radius*radius)
                K[214] += c1*phiy*sg[7]/(radius*radius) + 3*c1*sgL[11]/radius - 3*c1*sg[11]/radius + c1*sg[7]*v/(radius*radius*radius) - phiy*sg[4]/(radius*radius) - sgL[9]/radius + sg[9]/radius - sg[4]*v/(radius*radius*radius)
                K[216] += 3*c1*sg[11]*w_y + 3*c1*sg[12]*w_x - c1*sgL[7]/radius - 3*c1*sg[11]*v/radius + c1*sg[7]*v_y/radius + c1*sg[7]/radius + c1*sg[8]*v_x/radius + c1*sg[7]*w/(radius*radius) - sg[10]*w_x - sg[9]*w_y + sgL[4]/radius - sg[4]*v_y/radius - sg[4]/radius - sg[5]*v_x/radius + sg[9]*v/radius - sg[4]*w/(radius*radius)
                K[219] += c1*sg[6]*w_x + c1*sg[8]*w_y - c1*sg[8]*v/radius - sg[3]*w_x - sg[5]*w_y + sg[5]*v/radius
                K[220] += c1*sg[7]*w_y + c1*sg[8]*w_x - c1*sg[7]*v/radius - sg[4]*w_y - sg[5]*w_x + sg[4]*v/radius
                K[230] += c1*sgL[6] - c1*sg[6] - sgL[3] + sg[3]
                K[231] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[234] += c1*phix*sg[6] - phix*sg[3]
                K[235] += c1*phix*sg[8] - phix*sg[5]
                K[236] += -c1*phix*sg[8]/radius + phix*sg[5]/radius
                K[238] += c1*sg[6]*w_x + c1*sg[8]*w_y - c1*sg[8]*v/radius - sg[3]*w_x - sg[5]*w_y + sg[5]*v/radius
                K[253] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[254] += c1*sgL[7] - c1*sg[7] - sgL[4] + sg[4]
                K[257] += c1*phix*sg[8] - phix*sg[5]
                K[258] += c1*phix*sg[7] - phix*sg[4]
                K[259] += -c1*phix*sg[7]/radius + phix*sg[4]/radius
                K[261] += c1*sg[7]*w_y + c1*sg[8]*w_x - c1*sg[7]*v/radius - sg[4]*w_y - sg[5]*w_x + sg[4]*v/radius
                K[278] += c1*sgL[6] - c1*sg[6] - sgL[3] + sg[3]
                K[279] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[280] += c1*phiy*sg[6] + c1*sg[6]*v/radius - phiy*sg[3] - sg[3]*v/radius
                K[281] += c1*phiy*sg[8] + c1*sg[8]*v/radius - phiy*sg[5] - sg[5]*v/radius
                K[282] += -c1*phiy*sg[8]/radius + c1*sg[6]*w_x/radius + c1*sg[8]*w_y/radius - 2*c1*sg[8]*v/(radius*radius) + phiy*sg[5]/radius - sg[3]*w_x/radius - sg[5]*w_y/radius + 2*sg[5]*v/(radius*radius)
                K[283] += c1*sgL[8]/radius - c1*sg[8]/radius - sgL[5]/radius + sg[5]/radius
                K[285] += c1*sg[6]*w_x + c1*sg[8]*w_y - c1*sg[8]*v/radius - sg[3]*w_x - sg[5]*w_y + sg[5]*v/radius
                K[301] += c1*sgL[8] - c1*sg[8] - sgL[5] + sg[5]
                K[302] += c1*sgL[7] - c1*sg[7] - sgL[4] + sg[4]
                K[303] += c1*phiy*sg[8] + c1*sg[8]*v/radius - phiy*sg[5] - sg[5]*v/radius
                K[304] += c1*phiy*sg[7] + c1*sg[7]*v/radius - phiy*sg[4] - sg[4]*v/radius
                K[305] += -c1*phiy*sg[7]/radius + c1*sg[7]*w_y/radius + c1*sg[8]*w_x/radius - 2*c1*sg[7]*v/(radius*radius) + phiy*sg[4]/radius - sg[4]*w_y/radius - sg[5]*w_x/radius + 2*sg[4]*v/(radius*radius)
                K[306] += c1*sgL[7]/radius - c1*sg[7]/radius - sgL[4]/radius + sg[4]/radius
                K[308] += c1*sg[7]*w_y + c1*sg[8]*w_x - c1*sg[7]*v/radius - sg[4]*w_y - sg[5]*w_x + sg[4]*v/radius
                K[322] += c1*sgL[6] - c1*sg[6]*v_y - c1*sg[6] - c1*sg[6]*w/radius
                K[323] += c1*sg[6]*v_x
                K[324] += c1*sg[6]*u_y
                K[325] += c1*sgL[6] - c1*sg[6]*u_x - c1*sg[6]
                K[329] += c1*sgL[6]/radius - c1*sg[6]*u_x/radius - c1*sg[6]/radius
                K[345] += c1*sgL[7] - c1*sg[7]*v_y - c1*sg[7] - c1*sg[7]*w/radius
                K[346] += c1*sg[7]*v_x
                K[347] += c1*sg[7]*u_y
                K[348] += c1*sgL[7] - c1*sg[7]*u_x - c1*sg[7]
                K[352] += c1*sgL[7]/radius - c1*sg[7]*u_x/radius - c1*sg[7]/radius
                K[368] += 2*c1*sgL[8] - 2*c1*sg[8]*v_y - 2*c1*sg[8] - 2*c1*sg[8]*w/radius
                K[369] += 2*c1*sg[8]*v_x
                K[370] += 2*c1*sg[8]*u_y
                K[371] += 2*c1*sgL[8] - 2*c1*sg[8]*u_x - 2*c1*sg[8]
                K[375] += 2*c1*sgL[8]/radius - 2*c1*sg[8]*u_x/radius - 2*c1*sg[8]/radius
                K[393] += -c1*sg[6]*w_y + c1*sg[6]*v/radius
                K[394] += c1*sg[6]*w_x
                K[395] += -c1*sgL[6] + c1*sg[6]*v_y + c1*sg[6] + c1*sg[6]*w/radius
                K[396] += -c1*sg[6]*v_x
                K[397] += c1*sg[6]*v_x/radius
                K[398] += c1*sg[6]*w_x/radius
                K[416] += -c1*sg[7]*w_y + c1*sg[7]*v/radius
                K[417] += c1*sg[7]*w_x
                K[418] += -c1*sgL[7] + c1*sg[7]*v_y + c1*sg[7] + c1*sg[7]*w/radius
                K[419] += -c1*sg[7]*v_x
                K[420] += c1*sg[7]*v_x/radius
                K[421] += c1*sg[7]*w_x/radius
                K[439] += -2*c1*sg[8]*w_y + 2*c1*sg[8]*v/radius
                K[440] += 2*c1*sg[8]*w_x
                K[441] += -2*c1*sgL[8] + 2*c1*sg[8]*v_y + 2*c1*sg[8] + 2*c1*sg[8]*w/radius
                K[442] += -2*c1*sg[8]*v_x
                K[443] += 2*c1*sg[8]*v_x/radius
                K[444] += 2*c1*sg[8]*w_x/radius
                K[460] += c1*sg[6]*w_y - c1*sg[6]*v/radius
                K[461] += -c1*sg[6]*w_x
                K[464] += -c1*sg[6]*u_y
                K[465] += -c1*sgL[6] + c1*sg[6]*u_x + c1*sg[6]
                K[466] += c1*sgL[6]/radius - c1*sg[6]*u_x/radius - c1*sg[6]/radius
                K[483] += c1*sg[7]*w_y - c1*sg[7]*v/radius
                K[484] += -c1*sg[7]*w_x
                K[487] += -c1*sg[7]*u_y
                K[488] += -c1*sgL[7] + c1*sg[7]*u_x + c1*sg[7]
                K[489] += c1*sgL[7]/radius - c1*sg[7]*u_x/radius - c1*sg[7]/radius
                K[506] += 2*c1*sg[8]*w_y - 2*c1*sg[8]*v/radius
                K[507] += -2*c1*sg[8]*w_x
                K[510] += -2*c1*sg[8]*u_y
                K[511] += -2*c1*sgL[8] + 2*c1*sg[8]*u_x + 2*c1*sg[8]
                K[512] += 2*c1*sgL[8]/radius - 2*c1*sg[8]*u_x/radius - 2*c1*sg[8]/radius

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
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAphix*gAphix
                        pA[9] = fAphiy*gAphiy
                        pA[10] = fAphix_x*gAphix
                        pA[11] = fAphix*gAphix_y
                        pA[12] = fAphiy_x*gAphiy
                        pA[13] = fAphiy*gAphiy_y
                        pA[14] = fAw_xx*gAw
                        pA[15] = fAw*gAw_yy
                        pA[16] = fAw_x*gAw_y
                        pA[17] = fAu_xx*gAu
                        pA[18] = fAu*gAu_yy
                        pA[19] = fAu_x*gAu_y
                        pA[20] = fAv_xx*gAv
                        pA[21] = fAv*gAv_yy
                        pA[22] = fAv_x*gAv_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[23] + pA[17]*K[391] + pA[18]*K[414] + pA[19]*K[437]
                        TA[1] = pA[0]*K[1] + pA[1]*K[24] + pA[17]*K[392] + pA[18]*K[415] + pA[19]*K[438]
                        TA[2] = pA[0]*K[2] + pA[1]*K[25] + pA[17]*K[393] + pA[18]*K[416] + pA[19]*K[439]
                        TA[3] = pA[0]*K[3] + pA[1]*K[26] + pA[17]*K[394] + pA[18]*K[417] + pA[19]*K[440]
                        TA[4] = pA[0]*K[4] + pA[1]*K[27] + pA[17]*K[395] + pA[18]*K[418] + pA[19]*K[441]
                        TA[5] = pA[0]*K[5] + pA[1]*K[28] + pA[17]*K[396] + pA[18]*K[419] + pA[19]*K[442]
                        TA[6] = pA[0]*K[6] + pA[1]*K[29] + pA[17]*K[397] + pA[18]*K[420] + pA[19]*K[443]
                        TA[7] = pA[0]*K[7] + pA[1]*K[30] + pA[17]*K[398] + pA[18]*K[421] + pA[19]*K[444]
                        TA[8] = pA[0]*K[8] + pA[1]*K[31] + pA[17]*K[399] + pA[18]*K[422] + pA[19]*K[445]
                        TA[9] = pA[0]*K[9] + pA[1]*K[32] + pA[17]*K[400] + pA[18]*K[423] + pA[19]*K[446]
                        TA[10] = pA[0]*K[10] + pA[1]*K[33] + pA[17]*K[401] + pA[18]*K[424] + pA[19]*K[447]
                        TA[11] = pA[0]*K[11] + pA[1]*K[34] + pA[17]*K[402] + pA[18]*K[425] + pA[19]*K[448]
                        TA[12] = pA[0]*K[12] + pA[1]*K[35] + pA[17]*K[403] + pA[18]*K[426] + pA[19]*K[449]
                        TA[13] = pA[0]*K[13] + pA[1]*K[36] + pA[17]*K[404] + pA[18]*K[427] + pA[19]*K[450]
                        TA[14] = pA[0]*K[14] + pA[1]*K[37] + pA[17]*K[405] + pA[18]*K[428] + pA[19]*K[451]
                        TA[15] = pA[0]*K[15] + pA[1]*K[38] + pA[17]*K[406] + pA[18]*K[429] + pA[19]*K[452]
                        TA[16] = pA[0]*K[16] + pA[1]*K[39] + pA[17]*K[407] + pA[18]*K[430] + pA[19]*K[453]
                        TA[17] = pA[0]*K[17] + pA[1]*K[40] + pA[17]*K[408] + pA[18]*K[431] + pA[19]*K[454]
                        TA[18] = pA[0]*K[18] + pA[1]*K[41] + pA[17]*K[409] + pA[18]*K[432] + pA[19]*K[455]
                        TA[19] = pA[0]*K[19] + pA[1]*K[42] + pA[17]*K[410] + pA[18]*K[433] + pA[19]*K[456]
                        TA[20] = pA[0]*K[20] + pA[1]*K[43] + pA[17]*K[411] + pA[18]*K[434] + pA[19]*K[457]
                        TA[21] = pA[0]*K[21] + pA[1]*K[44] + pA[17]*K[412] + pA[18]*K[435] + pA[19]*K[458]
                        TA[22] = pA[0]*K[22] + pA[1]*K[45] + pA[17]*K[413] + pA[18]*K[436] + pA[19]*K[459]
                        TA[23] = pA[2]*K[46] + pA[3]*K[69] + pA[6]*K[138] + pA[20]*K[460] + pA[21]*K[483] + pA[22]*K[506]
                        TA[24] = pA[2]*K[47] + pA[3]*K[70] + pA[6]*K[139] + pA[20]*K[461] + pA[21]*K[484] + pA[22]*K[507]
                        TA[25] = pA[2]*K[48] + pA[3]*K[71] + pA[6]*K[140] + pA[20]*K[462] + pA[21]*K[485] + pA[22]*K[508]
                        TA[26] = pA[2]*K[49] + pA[3]*K[72] + pA[6]*K[141] + pA[20]*K[463] + pA[21]*K[486] + pA[22]*K[509]
                        TA[27] = pA[2]*K[50] + pA[3]*K[73] + pA[6]*K[142] + pA[20]*K[464] + pA[21]*K[487] + pA[22]*K[510]
                        TA[28] = pA[2]*K[51] + pA[3]*K[74] + pA[6]*K[143] + pA[20]*K[465] + pA[21]*K[488] + pA[22]*K[511]
                        TA[29] = pA[2]*K[52] + pA[3]*K[75] + pA[6]*K[144] + pA[20]*K[466] + pA[21]*K[489] + pA[22]*K[512]
                        TA[30] = pA[2]*K[53] + pA[3]*K[76] + pA[6]*K[145] + pA[20]*K[467] + pA[21]*K[490] + pA[22]*K[513]
                        TA[31] = pA[2]*K[54] + pA[3]*K[77] + pA[6]*K[146] + pA[20]*K[468] + pA[21]*K[491] + pA[22]*K[514]
                        TA[32] = pA[2]*K[55] + pA[3]*K[78] + pA[6]*K[147] + pA[20]*K[469] + pA[21]*K[492] + pA[22]*K[515]
                        TA[33] = pA[2]*K[56] + pA[3]*K[79] + pA[6]*K[148] + pA[20]*K[470] + pA[21]*K[493] + pA[22]*K[516]
                        TA[34] = pA[2]*K[57] + pA[3]*K[80] + pA[6]*K[149] + pA[20]*K[471] + pA[21]*K[494] + pA[22]*K[517]
                        TA[35] = pA[2]*K[58] + pA[3]*K[81] + pA[6]*K[150] + pA[20]*K[472] + pA[21]*K[495] + pA[22]*K[518]
                        TA[36] = pA[2]*K[59] + pA[3]*K[82] + pA[6]*K[151] + pA[20]*K[473] + pA[21]*K[496] + pA[22]*K[519]
                        TA[37] = pA[2]*K[60] + pA[3]*K[83] + pA[6]*K[152] + pA[20]*K[474] + pA[21]*K[497] + pA[22]*K[520]
                        TA[38] = pA[2]*K[61] + pA[3]*K[84] + pA[6]*K[153] + pA[20]*K[475] + pA[21]*K[498] + pA[22]*K[521]
                        TA[39] = pA[2]*K[62] + pA[3]*K[85] + pA[6]*K[154] + pA[20]*K[476] + pA[21]*K[499] + pA[22]*K[522]
                        TA[40] = pA[2]*K[63] + pA[3]*K[86] + pA[6]*K[155] + pA[20]*K[477] + pA[21]*K[500] + pA[22]*K[523]
                        TA[41] = pA[2]*K[64] + pA[3]*K[87] + pA[6]*K[156] + pA[20]*K[478] + pA[21]*K[501] + pA[22]*K[524]
                        TA[42] = pA[2]*K[65] + pA[3]*K[88] + pA[6]*K[157] + pA[20]*K[479] + pA[21]*K[502] + pA[22]*K[525]
                        TA[43] = pA[2]*K[66] + pA[3]*K[89] + pA[6]*K[158] + pA[20]*K[480] + pA[21]*K[503] + pA[22]*K[526]
                        TA[44] = pA[2]*K[67] + pA[3]*K[90] + pA[6]*K[159] + pA[20]*K[481] + pA[21]*K[504] + pA[22]*K[527]
                        TA[45] = pA[2]*K[68] + pA[3]*K[91] + pA[6]*K[160] + pA[20]*K[482] + pA[21]*K[505] + pA[22]*K[528]
                        TA[46] = pA[4]*K[92] + pA[5]*K[115] + pA[7]*K[161] + pA[14]*K[322] + pA[15]*K[345] + pA[16]*K[368]
                        TA[47] = pA[4]*K[93] + pA[5]*K[116] + pA[7]*K[162] + pA[14]*K[323] + pA[15]*K[346] + pA[16]*K[369]
                        TA[48] = pA[4]*K[94] + pA[5]*K[117] + pA[7]*K[163] + pA[14]*K[324] + pA[15]*K[347] + pA[16]*K[370]
                        TA[49] = pA[4]*K[95] + pA[5]*K[118] + pA[7]*K[164] + pA[14]*K[325] + pA[15]*K[348] + pA[16]*K[371]
                        TA[50] = pA[4]*K[96] + pA[5]*K[119] + pA[7]*K[165] + pA[14]*K[326] + pA[15]*K[349] + pA[16]*K[372]
                        TA[51] = pA[4]*K[97] + pA[5]*K[120] + pA[7]*K[166] + pA[14]*K[327] + pA[15]*K[350] + pA[16]*K[373]
                        TA[52] = pA[4]*K[98] + pA[5]*K[121] + pA[7]*K[167] + pA[14]*K[328] + pA[15]*K[351] + pA[16]*K[374]
                        TA[53] = pA[4]*K[99] + pA[5]*K[122] + pA[7]*K[168] + pA[14]*K[329] + pA[15]*K[352] + pA[16]*K[375]
                        TA[54] = pA[4]*K[100] + pA[5]*K[123] + pA[7]*K[169] + pA[14]*K[330] + pA[15]*K[353] + pA[16]*K[376]
                        TA[55] = pA[4]*K[101] + pA[5]*K[124] + pA[7]*K[170] + pA[14]*K[331] + pA[15]*K[354] + pA[16]*K[377]
                        TA[56] = pA[4]*K[102] + pA[5]*K[125] + pA[7]*K[171] + pA[14]*K[332] + pA[15]*K[355] + pA[16]*K[378]
                        TA[57] = pA[4]*K[103] + pA[5]*K[126] + pA[7]*K[172] + pA[14]*K[333] + pA[15]*K[356] + pA[16]*K[379]
                        TA[58] = pA[4]*K[104] + pA[5]*K[127] + pA[7]*K[173] + pA[14]*K[334] + pA[15]*K[357] + pA[16]*K[380]
                        TA[59] = pA[4]*K[105] + pA[5]*K[128] + pA[7]*K[174] + pA[14]*K[335] + pA[15]*K[358] + pA[16]*K[381]
                        TA[60] = pA[4]*K[106] + pA[5]*K[129] + pA[7]*K[175] + pA[14]*K[336] + pA[15]*K[359] + pA[16]*K[382]
                        TA[61] = pA[4]*K[107] + pA[5]*K[130] + pA[7]*K[176] + pA[14]*K[337] + pA[15]*K[360] + pA[16]*K[383]
                        TA[62] = pA[4]*K[108] + pA[5]*K[131] + pA[7]*K[177] + pA[14]*K[338] + pA[15]*K[361] + pA[16]*K[384]
                        TA[63] = pA[4]*K[109] + pA[5]*K[132] + pA[7]*K[178] + pA[14]*K[339] + pA[15]*K[362] + pA[16]*K[385]
                        TA[64] = pA[4]*K[110] + pA[5]*K[133] + pA[7]*K[179] + pA[14]*K[340] + pA[15]*K[363] + pA[16]*K[386]
                        TA[65] = pA[4]*K[111] + pA[5]*K[134] + pA[7]*K[180] + pA[14]*K[341] + pA[15]*K[364] + pA[16]*K[387]
                        TA[66] = pA[4]*K[112] + pA[5]*K[135] + pA[7]*K[181] + pA[14]*K[342] + pA[15]*K[365] + pA[16]*K[388]
                        TA[67] = pA[4]*K[113] + pA[5]*K[136] + pA[7]*K[182] + pA[14]*K[343] + pA[15]*K[366] + pA[16]*K[389]
                        TA[68] = pA[4]*K[114] + pA[5]*K[137] + pA[7]*K[183] + pA[14]*K[344] + pA[15]*K[367] + pA[16]*K[390]
                        TA[69] = pA[8]*K[184] + pA[10]*K[230] + pA[11]*K[253]
                        TA[70] = pA[8]*K[185] + pA[10]*K[231] + pA[11]*K[254]
                        TA[71] = pA[8]*K[186] + pA[10]*K[232] + pA[11]*K[255]
                        TA[72] = pA[8]*K[187] + pA[10]*K[233] + pA[11]*K[256]
                        TA[73] = pA[8]*K[188] + pA[10]*K[234] + pA[11]*K[257]
                        TA[74] = pA[8]*K[189] + pA[10]*K[235] + pA[11]*K[258]
                        TA[75] = pA[8]*K[190] + pA[10]*K[236] + pA[11]*K[259]
                        TA[76] = pA[8]*K[191] + pA[10]*K[237] + pA[11]*K[260]
                        TA[77] = pA[8]*K[192] + pA[10]*K[238] + pA[11]*K[261]
                        TA[78] = pA[8]*K[193] + pA[10]*K[239] + pA[11]*K[262]
                        TA[79] = pA[8]*K[194] + pA[10]*K[240] + pA[11]*K[263]
                        TA[80] = pA[8]*K[195] + pA[10]*K[241] + pA[11]*K[264]
                        TA[81] = pA[8]*K[196] + pA[10]*K[242] + pA[11]*K[265]
                        TA[82] = pA[8]*K[197] + pA[10]*K[243] + pA[11]*K[266]
                        TA[83] = pA[8]*K[198] + pA[10]*K[244] + pA[11]*K[267]
                        TA[84] = pA[8]*K[199] + pA[10]*K[245] + pA[11]*K[268]
                        TA[85] = pA[8]*K[200] + pA[10]*K[246] + pA[11]*K[269]
                        TA[86] = pA[8]*K[201] + pA[10]*K[247] + pA[11]*K[270]
                        TA[87] = pA[8]*K[202] + pA[10]*K[248] + pA[11]*K[271]
                        TA[88] = pA[8]*K[203] + pA[10]*K[249] + pA[11]*K[272]
                        TA[89] = pA[8]*K[204] + pA[10]*K[250] + pA[11]*K[273]
                        TA[90] = pA[8]*K[205] + pA[10]*K[251] + pA[11]*K[274]
                        TA[91] = pA[8]*K[206] + pA[10]*K[252] + pA[11]*K[275]
                        TA[92] = pA[9]*K[207] + pA[12]*K[276] + pA[13]*K[299]
                        TA[93] = pA[9]*K[208] + pA[12]*K[277] + pA[13]*K[300]
                        TA[94] = pA[9]*K[209] + pA[12]*K[278] + pA[13]*K[301]
                        TA[95] = pA[9]*K[210] + pA[12]*K[279] + pA[13]*K[302]
                        TA[96] = pA[9]*K[211] + pA[12]*K[280] + pA[13]*K[303]
                        TA[97] = pA[9]*K[212] + pA[12]*K[281] + pA[13]*K[304]
                        TA[98] = pA[9]*K[213] + pA[12]*K[282] + pA[13]*K[305]
                        TA[99] = pA[9]*K[214] + pA[12]*K[283] + pA[13]*K[306]
                        TA[100] = pA[9]*K[215] + pA[12]*K[284] + pA[13]*K[307]
                        TA[101] = pA[9]*K[216] + pA[12]*K[285] + pA[13]*K[308]
                        TA[102] = pA[9]*K[217] + pA[12]*K[286] + pA[13]*K[309]
                        TA[103] = pA[9]*K[218] + pA[12]*K[287] + pA[13]*K[310]
                        TA[104] = pA[9]*K[219] + pA[12]*K[288] + pA[13]*K[311]
                        TA[105] = pA[9]*K[220] + pA[12]*K[289] + pA[13]*K[312]
                        TA[106] = pA[9]*K[221] + pA[12]*K[290] + pA[13]*K[313]
                        TA[107] = pA[9]*K[222] + pA[12]*K[291] + pA[13]*K[314]
                        TA[108] = pA[9]*K[223] + pA[12]*K[292] + pA[13]*K[315]
                        TA[109] = pA[9]*K[224] + pA[12]*K[293] + pA[13]*K[316]
                        TA[110] = pA[9]*K[225] + pA[12]*K[294] + pA[13]*K[317]
                        TA[111] = pA[9]*K[226] + pA[12]*K[295] + pA[13]*K[318]
                        TA[112] = pA[9]*K[227] + pA[12]*K[296] + pA[13]*K[319]
                        TA[113] = pA[9]*K[228] + pA[12]*K[297] + pA[13]*K[320]
                        TA[114] = pA[9]*K[229] + pA[12]*K[298] + pA[13]*K[321]
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
                                pB[6] = fBv*gBv
                                pB[7] = fBw*gBw
                                pB[8] = fBphix*gBphix
                                pB[9] = fBphiy*gBphiy
                                pB[10] = fBphix_x*gBphix
                                pB[11] = fBphix*gBphix_y
                                pB[12] = fBphiy_x*gBphiy
                                pB[13] = fBphiy*gBphiy_y
                                pB[14] = fBw_xx*gBw
                                pB[15] = fBw*gBw_yy
                                pB[16] = fBw_x*gBw_y
                                pB[17] = fBu_xx*gBu
                                pB[18] = fBu*gBu_yy
                                pB[19] = fBu_x*gBu_y
                                pB[20] = fBv_xx*gBv
                                pB[21] = fBv*gBv_yy
                                pB[22] = fBv_x*gBv_y
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1] + TA[17]*pB[17] + TA[18]*pB[18] + TA[19]*pB[19])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[2]*pB[2] + TA[3]*pB[3] + TA[6]*pB[6] + TA[20]*pB[20] + TA[21]*pB[21] + TA[22]*pB[22])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[4]*pB[4] + TA[5]*pB[5] + TA[7]*pB[7] + TA[14]*pB[14] + TA[15]*pB[15] + TA[16]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[8]*pB[8] + TA[10]*pB[10] + TA[11]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[9]*pB[9] + TA[12]*pB[12] + TA[13]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[23]*pB[0] + TA[24]*pB[1] + TA[40]*pB[17] + TA[41]*pB[18] + TA[42]*pB[19])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[25]*pB[2] + TA[26]*pB[3] + TA[29]*pB[6] + TA[43]*pB[20] + TA[44]*pB[21] + TA[45]*pB[22])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[27]*pB[4] + TA[28]*pB[5] + TA[30]*pB[7] + TA[37]*pB[14] + TA[38]*pB[15] + TA[39]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[31]*pB[8] + TA[33]*pB[10] + TA[34]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[32]*pB[9] + TA[35]*pB[12] + TA[36]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[46]*pB[0] + TA[47]*pB[1] + TA[63]*pB[17] + TA[64]*pB[18] + TA[65]*pB[19])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[48]*pB[2] + TA[49]*pB[3] + TA[52]*pB[6] + TA[66]*pB[20] + TA[67]*pB[21] + TA[68]*pB[22])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[50]*pB[4] + TA[51]*pB[5] + TA[53]*pB[7] + TA[60]*pB[14] + TA[61]*pB[15] + TA[62]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[54]*pB[8] + TA[56]*pB[10] + TA[57]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[55]*pB[9] + TA[58]*pB[12] + TA[59]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[69]*pB[0] + TA[70]*pB[1] + TA[86]*pB[17] + TA[87]*pB[18] + TA[88]*pB[19])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[71]*pB[2] + TA[72]*pB[3] + TA[75]*pB[6] + TA[89]*pB[20] + TA[90]*pB[21] + TA[91]*pB[22])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[73]*pB[4] + TA[74]*pB[5] + TA[76]*pB[7] + TA[83]*pB[14] + TA[84]*pB[15] + TA[85]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[77]*pB[8] + TA[79]*pB[10] + TA[80]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[78]*pB[9] + TA[81]*pB[12] + TA[82]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[92]*pB[0] + TA[93]*pB[1] + TA[109]*pB[17] + TA[110]*pB[18] + TA[111]*pB[19])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[94]*pB[2] + TA[95]*pB[3] + TA[98]*pB[6] + TA[112]*pB[20] + TA[113]*pB[21] + TA[114]*pB[22])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[96]*pB[4] + TA[97]*pB[5] + TA[99]*pB[7] + TA[106]*pB[14] + TA[107]*pB[15] + TA[108]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[100]*pB[8] + TA[102]*pB[10] + TA[103]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[101]*pB[9] + TA[104]*pB[12] + TA[105]*pB[13])

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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the first and second derivatives of `u`, `v`, `w`, the values of `v` and `w`, the rotations and their gradients), the strains
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
    cdef double h, c1
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, w, phix
    cdef double phiy, phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy, w_xy
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
    cdef double K[529]
    cdef double TA[115]
    cdef double pA[23]
    cdef double pB[23]

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
    #NOTE the traction-free faces are assumed at z = +-h/2 about the
    #     mid-surface
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    radius = shell.r
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

    fdim = 17*m*m*n*n
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
                w = 0.
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
                        w += fAw*gAw*cs[col+2]
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
                e[1] = v_y + w/radius
                e[2] = u_y + v_x
                e[3] = phix_x
                e[4] = phiy_y + v_y/radius
                e[5] = phix_y + phiy_x - 0.5*u_y/radius + 1.5*v_x/radius
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
                for p in range(529):
                    K[p] = 0.
                K[0] = -c1*sg[7]/radius + sg[0]
                K[1] = sg[2] - 0.5*sg[5]/radius
                K[2] = 2*c1*sg[8]/radius
                K[3] = 3*c1*sg[7]/radius
                K[7] = 2*c1*sg[7]/(radius*radius)
                K[8] = -3*c1*sg[12] + sg[10]
                K[10] = -c1*sg[6] + sg[3]
                K[11] = -c1*sg[8] + sg[5]
                K[14] = -c1*sg[6]
                K[15] = -c1*sg[7]
                K[16] = -2*c1*sg[8]
                K[23] = sg[2] - 0.5*sg[5]/radius
                K[24] = -c1*sg[7]/radius + sg[1] - sg[4]/radius
                K[25] = -c1*sg[7]/radius
                K[31] = -3*c1*sg[11] + sg[9]
                K[33] = -c1*sg[8] + sg[5]
                K[34] = -c1*sg[7] + sg[4]
                K[46] = 2*c1*sg[8]/radius
                K[47] = -c1*sg[7]/radius
                K[48] = -2*c1*sg[6]/radius - c1*sg[7]/radius + sg[0] + 2*sg[3]/radius
                K[49] = sg[2] + 1.5*sg[5]/radius
                K[52] = -3*c1*sg[12]/radius + sg[10]/radius
                K[53] = c1*sg[8]/(radius*radius) + sg[2]/radius + 0.5*sg[5]/(radius*radius)
                K[55] = -3*c1*sg[12] + sg[10]
                K[58] = -c1*sg[6] + sg[3]
                K[59] = -c1*sg[8] + sg[5]
                K[69] = 3*c1*sg[7]/radius
                K[71] = sg[2] + 1.5*sg[5]/radius
                K[72] = c1*sg[7]/radius + sg[1] + sg[4]/radius
                K[75] = -3*c1*sg[11]/radius + sg[9]/radius
                K[76] = c1*sg[7]/(radius*radius) + sg[1]/radius
                K[78] = -3*c1*sg[11] + sg[9]
                K[81] = -c1*sg[8] + sg[5]
                K[82] = -c1*sg[7] + sg[4]
                K[83] = -c1*sg[6]
                K[84] = -c1*sg[7]
                K[85] = -2*c1*sg[8]
                K[96] = -c1*sg[7]/radius + sg[0]
                K[97] = 2*c1*sg[8]/radius + sg[2] - 0.5*sg[5]/radius
                K[98] = -c1*sg[8]/(radius*radius) - sg[2]/radius - 0.5*sg[5]/(radius*radius)
                K[101] = c1*sg[8]/radius - sg[5]/radius
                K[109] = c1*sg[6]
                K[110] = c1*sg[7]
                K[111] = 2*c1*sg[8]
                K[119] = 2*c1*sg[8]/radius + sg[2] - 0.5*sg[5]/radius
                K[120] = 3*c1*sg[7]/radius + sg[1] - sg[4]/radius
                K[121] = -c1*sg[7]/(radius*radius) - sg[1]/radius
                K[124] = c1*sg[7]/radius - sg[4]/radius
                K[135] = c1*sg[6]
                K[136] = c1*sg[7]
                K[137] = 2*c1*sg[8]
                K[140] = -3*c1*sg[12]/radius + sg[10]/radius
                K[141] = -3*c1*sg[11]/radius + sg[9]/radius
                K[142] = -c1*sg[8]/(radius*radius) - sg[2]/radius - 0.5*sg[5]/(radius*radius)
                K[143] = -c1*sg[7]/(radius*radius) - sg[1]/radius
                K[144] = sg[1]/(radius*radius)
                K[145] = -3*c1*sg[11]/(radius*radius) + sg[9]/(radius*radius)
                K[158] = -c1*sg[6]/radius
                K[159] = -c1*sg[7]/radius
                K[160] = -2*c1*sg[8]/radius
                K[161] = 2*c1*sg[7]/(radius*radius)
                K[163] = c1*sg[8]/(radius*radius) + sg[2]/radius + 0.5*sg[5]/(radius*radius)
                K[164] = c1*sg[7]/(radius*radius) + sg[1]/radius
                K[167] = -3*c1*sg[11]/(radius*radius) + sg[9]/(radius*radius)
                K[168] = c1*sg[7]/(radius*radius*radius) + sg[1]/(radius*radius) - sg[4]/(radius*radius*radius)
                K[170] = -3*c1*sg[11]/radius + sg[9]/radius
                K[173] = -c1*sg[8]/radius + sg[5]/radius
                K[174] = -c1*sg[7]/radius + sg[4]/radius
                K[175] = -c1*sg[6]/radius
                K[176] = -c1*sg[7]/radius
                K[177] = -2*c1*sg[8]/radius
                K[184] = -3*c1*sg[12] + sg[10]
                K[185] = -3*c1*sg[11] + sg[9]
                K[192] = c1*sg[7]/radius - sg[4]/radius
                K[209] = -3*c1*sg[12] + sg[10]
                K[210] = -3*c1*sg[11] + sg[9]
                K[211] = c1*sg[8]/radius - sg[5]/radius
                K[212] = c1*sg[7]/radius - sg[4]/radius
                K[214] = -3*c1*sg[11]/radius + sg[9]/radius
                K[216] = c1*sg[7]/radius - sg[4]/radius
                K[230] = -c1*sg[6] + sg[3]
                K[231] = -c1*sg[8] + sg[5]
                K[253] = -c1*sg[8] + sg[5]
                K[254] = -c1*sg[7] + sg[4]
                K[278] = -c1*sg[6] + sg[3]
                K[279] = -c1*sg[8] + sg[5]
                K[283] = -c1*sg[8]/radius + sg[5]/radius
                K[301] = -c1*sg[8] + sg[5]
                K[302] = -c1*sg[7] + sg[4]
                K[306] = -c1*sg[7]/radius + sg[4]/radius
                K[322] = -c1*sg[6]
                K[325] = -c1*sg[6]
                K[329] = -c1*sg[6]/radius
                K[345] = -c1*sg[7]
                K[348] = -c1*sg[7]
                K[352] = -c1*sg[7]/radius
                K[368] = -2*c1*sg[8]
                K[371] = -2*c1*sg[8]
                K[375] = -2*c1*sg[8]/radius
                K[395] = c1*sg[6]
                K[418] = c1*sg[7]
                K[441] = 2*c1*sg[8]
                K[465] = c1*sg[6]
                K[466] = -c1*sg[6]/radius
                K[488] = c1*sg[7]
                K[489] = -c1*sg[7]/radius
                K[511] = 2*c1*sg[8]
                K[512] = -2*c1*sg[8]/radius

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
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAphix*gAphix
                        pA[9] = fAphiy*gAphiy
                        pA[10] = fAphix_x*gAphix
                        pA[11] = fAphix*gAphix_y
                        pA[12] = fAphiy_x*gAphiy
                        pA[13] = fAphiy*gAphiy_y
                        pA[14] = fAw_xx*gAw
                        pA[15] = fAw*gAw_yy
                        pA[16] = fAw_x*gAw_y
                        pA[17] = fAu_xx*gAu
                        pA[18] = fAu*gAu_yy
                        pA[19] = fAu_x*gAu_y
                        pA[20] = fAv_xx*gAv
                        pA[21] = fAv*gAv_yy
                        pA[22] = fAv_x*gAv_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[23]
                        TA[1] = pA[0]*K[1] + pA[1]*K[24]
                        TA[2] = pA[0]*K[2] + pA[1]*K[25]
                        TA[3] = pA[0]*K[3]
                        TA[4] = pA[17]*K[395] + pA[18]*K[418] + pA[19]*K[441]
                        TA[7] = pA[0]*K[7]
                        TA[8] = pA[0]*K[8] + pA[1]*K[31]
                        TA[10] = pA[0]*K[10] + pA[1]*K[33]
                        TA[11] = pA[0]*K[11] + pA[1]*K[34]
                        TA[14] = pA[0]*K[14]
                        TA[15] = pA[0]*K[15]
                        TA[16] = pA[0]*K[16]
                        TA[23] = pA[2]*K[46] + pA[3]*K[69]
                        TA[24] = pA[2]*K[47]
                        TA[25] = pA[2]*K[48] + pA[3]*K[71] + pA[6]*K[140]
                        TA[26] = pA[2]*K[49] + pA[3]*K[72] + pA[6]*K[141]
                        TA[27] = pA[6]*K[142]
                        TA[28] = pA[6]*K[143] + pA[20]*K[465] + pA[21]*K[488] + pA[22]*K[511]
                        TA[29] = pA[2]*K[52] + pA[3]*K[75] + pA[6]*K[144] + pA[20]*K[466] + pA[21]*K[489] + pA[22]*K[512]
                        TA[30] = pA[2]*K[53] + pA[3]*K[76] + pA[6]*K[145]
                        TA[32] = pA[2]*K[55] + pA[3]*K[78]
                        TA[35] = pA[2]*K[58] + pA[3]*K[81]
                        TA[36] = pA[2]*K[59] + pA[3]*K[82]
                        TA[37] = pA[3]*K[83]
                        TA[38] = pA[3]*K[84]
                        TA[39] = pA[3]*K[85]
                        TA[43] = pA[6]*K[158]
                        TA[44] = pA[6]*K[159]
                        TA[45] = pA[6]*K[160]
                        TA[46] = pA[7]*K[161] + pA[14]*K[322] + pA[15]*K[345] + pA[16]*K[368]
                        TA[48] = pA[7]*K[163]
                        TA[49] = pA[7]*K[164] + pA[14]*K[325] + pA[15]*K[348] + pA[16]*K[371]
                        TA[50] = pA[4]*K[96] + pA[5]*K[119]
                        TA[51] = pA[4]*K[97] + pA[5]*K[120]
                        TA[52] = pA[4]*K[98] + pA[5]*K[121] + pA[7]*K[167]
                        TA[53] = pA[7]*K[168] + pA[14]*K[329] + pA[15]*K[352] + pA[16]*K[375]
                        TA[55] = pA[4]*K[101] + pA[5]*K[124] + pA[7]*K[170]
                        TA[58] = pA[7]*K[173]
                        TA[59] = pA[7]*K[174]
                        TA[60] = pA[7]*K[175]
                        TA[61] = pA[7]*K[176]
                        TA[62] = pA[7]*K[177]
                        TA[63] = pA[4]*K[109]
                        TA[64] = pA[4]*K[110]
                        TA[65] = pA[4]*K[111]
                        TA[66] = pA[5]*K[135]
                        TA[67] = pA[5]*K[136]
                        TA[68] = pA[5]*K[137]
                        TA[69] = pA[8]*K[184] + pA[10]*K[230] + pA[11]*K[253]
                        TA[70] = pA[8]*K[185] + pA[10]*K[231] + pA[11]*K[254]
                        TA[77] = pA[8]*K[192]
                        TA[94] = pA[9]*K[209] + pA[12]*K[278] + pA[13]*K[301]
                        TA[95] = pA[9]*K[210] + pA[12]*K[279] + pA[13]*K[302]
                        TA[96] = pA[9]*K[211]
                        TA[97] = pA[9]*K[212]
                        TA[99] = pA[9]*K[214] + pA[12]*K[283] + pA[13]*K[306]
                        TA[101] = pA[9]*K[216]
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
                                pB[6] = fBv*gBv
                                pB[7] = fBw*gBw
                                pB[8] = fBphix*gBphix
                                pB[9] = fBphiy*gBphiy
                                pB[10] = fBphix_x*gBphix
                                pB[11] = fBphix*gBphix_y
                                pB[12] = fBphiy_x*gBphiy
                                pB[13] = fBphiy*gBphiy_y
                                pB[14] = fBw_xx*gBw
                                pB[15] = fBw*gBw_yy
                                pB[16] = fBw_x*gBw_y
                                pB[17] = fBu_xx*gBu
                                pB[18] = fBu*gBu_yy
                                pB[19] = fBu_x*gBu_y
                                pB[20] = fBv_xx*gBv
                                pB[21] = fBv*gBv_yy
                                pB[22] = fBv_x*gBv_y
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[2]*pB[2] + TA[3]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[4]*pB[4] + TA[7]*pB[7] + TA[14]*pB[14] + TA[15]*pB[15] + TA[16]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+3
                                kGv[c] += weight*(TA[8]*pB[8] + TA[10]*pB[10] + TA[11]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[23]*pB[0] + TA[24]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[25]*pB[2] + TA[26]*pB[3] + TA[29]*pB[6] + TA[43]*pB[20] + TA[44]*pB[21] + TA[45]*pB[22])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[27]*pB[4] + TA[28]*pB[5] + TA[30]*pB[7] + TA[37]*pB[14] + TA[38]*pB[15] + TA[39]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+4
                                kGv[c] += weight*(TA[32]*pB[9] + TA[35]*pB[12] + TA[36]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[46]*pB[0] + TA[63]*pB[17] + TA[64]*pB[18] + TA[65]*pB[19])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[48]*pB[2] + TA[49]*pB[3] + TA[52]*pB[6] + TA[66]*pB[20] + TA[67]*pB[21] + TA[68]*pB[22])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[50]*pB[4] + TA[51]*pB[5] + TA[53]*pB[7] + TA[60]*pB[14] + TA[61]*pB[15] + TA[62]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+4
                                kGv[c] += weight*(TA[55]*pB[9] + TA[58]*pB[12] + TA[59]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+3
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[69]*pB[0] + TA[70]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+3
                                    kGc[c] = col+3
                                kGv[c] += weight*(TA[77]*pB[8])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+4
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[94]*pB[2] + TA[95]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+4
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[96]*pB[4] + TA[97]*pB[5] + TA[99]*pB[7])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+4
                                    kGc[c] = col+4
                                kGv[c] += weight*(TA[101]*pB[9])

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
    cdef double h, c1
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w
    cdef double phix, phiy, phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy
    cdef double w_xy, u_xx, u_yy, u_xy, v_xx, v_yy, v_xy
    cdef double e[13]
    cdef double sg[13]
    cdef double S[23]
    cdef double pA[23]
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
    #NOTE the traction-free faces are assumed at z = +-h/2 about the
    #     mid-surface
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    radius = shell.r
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
                v = 0.
                w = 0.
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
                        v += fAv*gAv*cs[col+1]
                        w += fAw*gAw*cs[col+2]
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
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x - phiy*v_x*w_x/radius + phiy_x*v_x - phiy_x*v*w_x/radius + (v_x*v_x)/radius - v*v_x*w_x/(radius*radius)
                e[4] = -0.5*(phix*phix)*v_y/radius - 0.5*(phix*phix)/radius - 0.5*(phix*phix)*w/(radius*radius) - phix*phix_y*w_y + phix*phix_y*v/radius + phix_y*u_y - 0.5*(phiy*phiy)*v_y/radius - 0.5*(phiy*phiy)/radius - 0.5*(phiy*phiy)*w/(radius*radius) - phiy*phiy_y*w_y + phiy*phiy_y*v/radius - phiy*v_y*w_y/radius - phiy*w_y/radius - phiy*v*w/(radius*radius*radius) + phiy_y*v_y + phiy_y - phiy_y*v*w_y/radius + phiy_y*w/radius + phiy_y*(v*v)/(radius*radius) - 0.5*(u_y*u_y)/radius + 0.5*(v_y*v_y)/radius + v_y/radius - 0.5*(w_y*w_y)/radius - v*v_y*w_y/(radius*radius) + 0.5*(v*v)*v_y/(radius*radius*radius) - 0.5*(w*w)/(radius*radius*radius) - 0.5*(v*v)*w/(radius*radius*radius*radius)
                e[5] = -0.5*(phix*phix)*v_x/radius - phix*phix_x*w_y + phix*phix_x*v/radius - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - 0.5*(phiy*phiy)*v_x/radius - phiy*phiy_x*w_y + phiy*phiy_x*v/radius - phiy*phiy_y*w_x - phiy*v_x*w_y/radius - phiy*v_y*w_x/radius - phiy*w_x/radius + phiy_x*v_y + phiy_x - phiy_x*v*w_y/radius + phiy_x*w/radius + phiy_x*(v*v)/(radius*radius) + phiy_y*v_x - phiy_y*v*w_x/radius - 0.5*u_x*u_y/radius - 0.5*u_y/radius + 1.5*v_x*v_y/radius + 1.5*v_x/radius - 0.5*w_x*w_y/radius - v*v_x*w_y/(radius*radius) - v*v_y*w_x/(radius*radius) - 0.5*v*w_x/(radius*radius) + 0.5*v_x*w/(radius*radius) + 0.5*(v*v)*v_x/(radius*radius*radius)
                e[6] = c1*phix*phix_x*w_x - c1*phix_x*u_x - c1*phix_x + c1*phiy*phiy_x*w_x + c1*phiy*v_x*w_x/radius - c1*phiy_x*v_x + c1*phiy_x*v*w_x/radius + c1*u_x*v_xx*w_y - c1*u_x*v_y*w_xx - c1*u_x*w_xx - c1*u_xx*v_x*w_y + c1*u_xx*v_y*w_x + c1*u_xx*w_x + c1*u_y*v_x*w_xx - c1*u_y*v_xx*w_x + c1*v_xx*w_y - c1*v_y*w_xx - c1*w_xx - c1*u_x*v*v_xx/radius - c1*u_x*w*w_xx/radius + c1*u_xx*v*v_x/radius + c1*u_xx*w*w_x/radius - c1*v*v_xx/radius - c1*(v_x*v_x)/radius - c1*w*w_xx/radius + c1*v*v_x*w_x/(radius*radius)
                e[7] = 0.5*c1*(phix*phix)*v_y/radius + 0.5*c1*(phix*phix)/radius + 0.5*c1*(phix*phix)*w/(radius*radius) + c1*phix*phix_y*w_y - c1*phix*phix_y*v/radius - c1*phix_y*u_y + 0.5*c1*(phiy*phiy)*v_y/radius + 0.5*c1*(phiy*phiy)/radius + 0.5*c1*(phiy*phiy)*w/(radius*radius) + c1*phiy*phiy_y*w_y - c1*phiy*phiy_y*v/radius + c1*phiy*v_y*w_y/radius + c1*phiy*w_y/radius + c1*phiy*v*w/(radius*radius*radius) - c1*phiy_y*v_y - c1*phiy_y + c1*phiy_y*v*w_y/radius - c1*phiy_y*w/radius - c1*phiy_y*(v*v)/(radius*radius) - c1*u_x*v_y*w_yy + c1*u_x*v_yy*w_y - c1*u_x*w_yy + c1*u_y*v_x*w_yy - c1*u_y*v_yy*w_x - c1*u_yy*v_x*w_y + c1*u_yy*v_y*w_x + c1*u_yy*w_x - c1*v_y*w_yy + c1*v_yy*w_y - c1*w_yy - 0.5*c1*(u_x*u_x)/radius - c1*u_x*v*v_yy/radius + 2*c1*u_x*(v_y*v_y)/radius + 3*c1*u_x*v_y/radius - c1*u_x*w*w_yy/radius + 2*c1*u_x*(w_y*w_y)/radius - 0.5*c1*(u_y*u_y)/radius - 2*c1*u_y*v_x*v_y/radius - c1*u_y*v_x/radius - 2*c1*u_y*w_x*w_y/radius + c1*u_yy*v*v_x/radius + c1*u_yy*w*w_x/radius - c1*v*v_yy/radius - 0.5*c1*(v_x*v_x)/radius + 0.5*c1*(v_y*v_y)/radius - c1*w*w_yy/radius - 0.5*c1*(w_x*w_x)/radius + 1.5*c1*(w_y*w_y)/radius - 3*c1*u_x*v*w_y/(radius*radius) + 3*c1*u_x*v_y*w/(radius*radius) + 2*c1*u_x*w/(radius*radius) + c1*u_y*v*w_x/(radius*radius) - c1*u_y*v_x*w/(radius*radius) + c1*v*v_y*w_y/(radius*radius) - c1*v*w_y/(radius*radius) + c1*v_y*w/(radius*radius) + c1*u_x*(v*v)/(radius*radius*radius) + c1*u_x*(w*w)/(radius*radius*radius) - 0.5*c1*(v*v)*v_y/(radius*radius*radius) + 0.5*c1*(w*w)/(radius*radius*radius) + 0.5*c1*(v*v)*w/(radius*radius*radius*radius)
                e[8] = 0.5*c1*(phix*phix)*v_x/radius + c1*phix*phix_x*w_y - c1*phix*phix_x*v/radius + c1*phix*phix_y*w_x - c1*phix_x*u_y - c1*phix_y*u_x - c1*phix_y + 0.5*c1*(phiy*phiy)*v_x/radius + c1*phiy*phiy_x*w_y - c1*phiy*phiy_x*v/radius + c1*phiy*phiy_y*w_x + c1*phiy*v_x*w_y/radius + c1*phiy*v_y*w_x/radius + c1*phiy*w_x/radius - c1*phiy_x*v_y - c1*phiy_x + c1*phiy_x*v*w_y/radius - c1*phiy_x*w/radius - c1*phiy_x*(v*v)/(radius*radius) - c1*phiy_y*v_x + c1*phiy_y*v*w_x/radius + 2*c1*u_x*v_xy*w_y - 2*c1*u_x*v_y*w_xy - 2*c1*u_x*w_xy - 2*c1*u_xy*v_x*w_y + 2*c1*u_xy*v_y*w_x + 2*c1*u_xy*w_x + 2*c1*u_y*v_x*w_xy - 2*c1*u_y*v_xy*w_x + 2*c1*v_xy*w_y - 2*c1*v_y*w_xy - 2*c1*w_xy - 2*c1*u_x*v*v_xy/radius + 2*c1*u_x*v_x*v_y/radius + 2*c1*u_x*v_x/radius - 2*c1*u_x*w*w_xy/radius + 2*c1*u_x*w_x*w_y/radius + 2*c1*u_xy*v*v_x/radius + 2*c1*u_xy*w*w_x/radius - 2*c1*u_y*(v_x*v_x)/radius - 2*c1*u_y*(w_x*w_x)/radius - 2*c1*v*v_xy/radius - 2*c1*w*w_xy/radius + 2*c1*w_x*w_y/radius - 2*c1*u_x*v*w_x/(radius*radius) + 2*c1*u_x*v_x*w/(radius*radius) + c1*v*v_x*w_y/(radius*radius) + c1*v*v_y*w_x/(radius*radius) - c1*v*w_x/(radius*radius) + c1*v_x*w/(radius*radius) - 0.5*c1*(v*v)*v_x/(radius*radius*radius)
                e[9] = -0.5*(phix*phix)*w_y + 0.5*(phix*phix)*v/radius + phix*u_y - 0.5*(phiy*phiy)*w_y + 0.5*(phiy*phiy)*v/radius + phiy*v_y + phiy - phiy*v*w_y/radius + phiy*w/radius + phiy*(v*v)/(radius*radius) + w_y + v*v_y/radius - 0.5*(v*v)*w_y/(radius*radius) + v*w/(radius*radius) + 0.5*(v*v*v)/(radius*radius*radius)
                e[10] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x - phiy*v*w_x/radius + w_x + v*v_x/radius - 0.5*(v*v)*w_x/(radius*radius)
                e[11] = 1.5*c1*(phix*phix)*w_y - 1.5*c1*(phix*phix)*v/radius - 3*c1*phix*u_y + 1.5*c1*(phiy*phiy)*w_y - 1.5*c1*(phiy*phiy)*v/radius - 3*c1*phiy*v_y - 3*c1*phiy + 3*c1*phiy*v*w_y/radius - 3*c1*phiy*w/radius - 3*c1*phiy*(v*v)/(radius*radius) - 3*c1*w_y - 3*c1*v*v_y/radius + 1.5*c1*(v*v)*w_y/(radius*radius) - 3*c1*v*w/(radius*radius) - 1.5*c1*(v*v*v)/(radius*radius*radius)
                e[12] = 1.5*c1*(phix*phix)*w_x - 3*c1*phix*u_x - 3*c1*phix + 1.5*c1*(phiy*phiy)*w_x - 3*c1*phiy*v_x + 3*c1*phiy*v*w_x/radius - 3*c1*w_x - 3*c1*v*v_x/radius + 1.5*c1*(v*v)*w_x/(radius*radius)
                for r in range(NE):
                    sg[r] = 0.
                    for s in range(NE):
                        sg[r] += F[r*NE + s]*e[s]
                S[0] = -3*c1*phix*sg[12] - c1*phix_x*sg[6] - c1*phix_y*sg[8] + c1*sg[6]*v_xx*w_y - c1*sg[6]*v_y*w_xx - c1*sg[6]*w_xx - c1*sg[7]*v_y*w_yy + c1*sg[7]*v_yy*w_y - c1*sg[7]*w_yy + 2*c1*sg[8]*v_xy*w_y - 2*c1*sg[8]*v_y*w_xy - 2*c1*sg[8]*w_xy - c1*sg[6]*v*v_xx/radius - c1*sg[6]*w*w_xx/radius - c1*sg[7]*u_x/radius - c1*sg[7]*v*v_yy/radius + 2*c1*sg[7]*(v_y*v_y)/radius + 3*c1*sg[7]*v_y/radius - c1*sg[7]*w*w_yy/radius + 2*c1*sg[7]*(w_y*w_y)/radius - 2*c1*sg[8]*v*v_xy/radius + 2*c1*sg[8]*v_x*v_y/radius + 2*c1*sg[8]*v_x/radius - 2*c1*sg[8]*w*w_xy/radius + 2*c1*sg[8]*w_x*w_y/radius - 3*c1*sg[7]*v*w_y/(radius*radius) + 3*c1*sg[7]*v_y*w/(radius*radius) + 2*c1*sg[7]*w/(radius*radius) - 2*c1*sg[8]*v*w_x/(radius*radius) + 2*c1*sg[8]*v_x*w/(radius*radius) + c1*sg[7]*(v*v)/(radius*radius*radius) + c1*sg[7]*(w*w)/(radius*radius*radius) + phix*sg[10] + phix_x*sg[3] + phix_y*sg[5] + sg[0]*u_x + sg[0] + sg[2]*u_y - 0.5*sg[5]*u_y/radius
                S[1] = -3*c1*phix*sg[11] - c1*phix_x*sg[8] - c1*phix_y*sg[7] + c1*sg[6]*v_x*w_xx - c1*sg[6]*v_xx*w_x + c1*sg[7]*v_x*w_yy - c1*sg[7]*v_yy*w_x + 2*c1*sg[8]*v_x*w_xy - 2*c1*sg[8]*v_xy*w_x - c1*sg[7]*u_y/radius - 2*c1*sg[7]*v_x*v_y/radius - c1*sg[7]*v_x/radius - 2*c1*sg[7]*w_x*w_y/radius - 2*c1*sg[8]*(v_x*v_x)/radius - 2*c1*sg[8]*(w_x*w_x)/radius + c1*sg[7]*v*w_x/(radius*radius) - c1*sg[7]*v_x*w/(radius*radius) + phix*sg[9] + phix_x*sg[5] + phix_y*sg[4] + sg[1]*u_y + sg[2]*u_x + sg[2] - sg[4]*u_y/radius - 0.5*sg[5]*u_x/radius - 0.5*sg[5]/radius
                S[2] = 0.5*c1*(phix*phix)*sg[8]/radius + 0.5*c1*(phiy*phiy)*sg[8]/radius - 3*c1*phiy*sg[12] + c1*phiy*sg[6]*w_x/radius + c1*phiy*sg[8]*w_y/radius - c1*phiy_x*sg[6] - c1*phiy_y*sg[8] - c1*sg[6]*u_xx*w_y + c1*sg[6]*u_y*w_xx + c1*sg[7]*u_y*w_yy - c1*sg[7]*u_yy*w_y - 2*c1*sg[8]*u_xy*w_y + 2*c1*sg[8]*u_y*w_xy - 3*c1*sg[12]*v/radius + c1*sg[6]*u_xx*v/radius - 2*c1*sg[6]*v_x/radius - 2*c1*sg[7]*u_y*v_y/radius - c1*sg[7]*u_y/radius + c1*sg[7]*u_yy*v/radius - c1*sg[7]*v_x/radius + 2*c1*sg[8]*u_x*v_y/radius + 2*c1*sg[8]*u_x/radius + 2*c1*sg[8]*u_xy*v/radius - 4*c1*sg[8]*u_y*v_x/radius + c1*sg[6]*v*w_x/(radius*radius) - c1*sg[7]*u_y*w/(radius*radius) + 2*c1*sg[8]*u_x*w/(radius*radius) + c1*sg[8]*v*w_y/(radius*radius) + c1*sg[8]*w/(radius*radius) - 0.5*c1*sg[8]*(v*v)/(radius*radius*radius) - 0.5*(phix*phix)*sg[5]/radius - 0.5*(phiy*phiy)*sg[5]/radius + phiy*sg[10] - phiy*sg[3]*w_x/radius - phiy*sg[5]*w_y/radius + phiy_x*sg[3] + phiy_y*sg[5] + sg[0]*v_x + sg[2]*v_y + sg[2] + sg[10]*v/radius + sg[2]*w/radius + 2*sg[3]*v_x/radius + 1.5*sg[5]*v_y/radius + 1.5*sg[5]/radius - sg[3]*v*w_x/(radius*radius) - sg[5]*v*w_y/(radius*radius) + 0.5*sg[5]*w/(radius*radius) + 0.5*sg[5]*(v*v)/(radius*radius*radius)
                S[3] = 0.5*c1*(phix*phix)*sg[7]/radius + 0.5*c1*(phiy*phiy)*sg[7]/radius - 3*c1*phiy*sg[11] + c1*phiy*sg[7]*w_y/radius + c1*phiy*sg[8]*w_x/radius - c1*phiy_x*sg[8] - c1*phiy_y*sg[7] - c1*sg[6]*u_x*w_xx + c1*sg[6]*u_xx*w_x - c1*sg[6]*w_xx - c1*sg[7]*u_x*w_yy + c1*sg[7]*u_yy*w_x - c1*sg[7]*w_yy - 2*c1*sg[8]*u_x*w_xy + 2*c1*sg[8]*u_xy*w_x - 2*c1*sg[8]*w_xy - 3*c1*sg[11]*v/radius + 4*c1*sg[7]*u_x*v_y/radius + 3*c1*sg[7]*u_x/radius - 2*c1*sg[7]*u_y*v_x/radius + c1*sg[7]*v_y/radius + 2*c1*sg[8]*u_x*v_x/radius + 3*c1*sg[7]*u_x*w/(radius*radius) + c1*sg[7]*v*w_y/(radius*radius) + c1*sg[7]*w/(radius*radius) + c1*sg[8]*v*w_x/(radius*radius) - 0.5*c1*sg[7]*(v*v)/(radius*radius*radius) - 0.5*(phix*phix)*sg[4]/radius - 0.5*(phiy*phiy)*sg[4]/radius + phiy*sg[9] - phiy*sg[4]*w_y/radius - phiy*sg[5]*w_x/radius + phiy_x*sg[5] + phiy_y*sg[4] + sg[1]*v_y + sg[1] + sg[2]*v_x + sg[1]*w/radius + sg[4]*v_y/radius + sg[4]/radius + 1.5*sg[5]*v_x/radius + sg[9]*v/radius - sg[4]*v*w_y/(radius*radius) - sg[5]*v*w_x/(radius*radius) + 0.5*sg[4]*(v*v)/(radius*radius*radius)
                S[4] = 1.5*c1*(phix*phix)*sg[12] + c1*phix*phix_x*sg[6] + c1*phix*phix_y*sg[8] + 1.5*c1*(phiy*phiy)*sg[12] + c1*phiy*phiy_x*sg[6] + c1*phiy*phiy_y*sg[8] + 3*c1*phiy*sg[12]*v/radius + c1*phiy*sg[6]*v_x/radius + c1*phiy*sg[8]*v_y/radius + c1*phiy*sg[8]/radius + c1*phiy_x*sg[6]*v/radius + c1*phiy_y*sg[8]*v/radius - 3*c1*sg[12] + c1*sg[6]*u_xx*v_y + c1*sg[6]*u_xx - c1*sg[6]*u_y*v_xx - c1*sg[7]*u_y*v_yy + c1*sg[7]*u_yy*v_y + c1*sg[7]*u_yy + 2*c1*sg[8]*u_xy*v_y + 2*c1*sg[8]*u_xy - 2*c1*sg[8]*u_y*v_xy + c1*sg[6]*u_xx*w/radius - 2*c1*sg[7]*u_y*w_y/radius + c1*sg[7]*u_yy*w/radius - c1*sg[7]*w_x/radius + 2*c1*sg[8]*u_x*w_y/radius + 2*c1*sg[8]*u_xy*w/radius - 4*c1*sg[8]*u_y*w_x/radius + 2*c1*sg[8]*w_y/radius + 1.5*c1*sg[12]*(v*v)/(radius*radius) + c1*sg[6]*v*v_x/(radius*radius) + c1*sg[7]*u_y*v/(radius*radius) - 2*c1*sg[8]*u_x*v/(radius*radius) + c1*sg[8]*v*v_y/(radius*radius) - c1*sg[8]*v/(radius*radius) - 0.5*(phix*phix)*sg[10] - phix*phix_x*sg[3] - phix*phix_y*sg[5] - 0.5*(phiy*phiy)*sg[10] - phiy*phiy_x*sg[3] - phiy*phiy_y*sg[5] - phiy*sg[10]*v/radius - phiy*sg[3]*v_x/radius - phiy*sg[5]*v_y/radius - phiy*sg[5]/radius - phiy_x*sg[3]*v/radius - phiy_y*sg[5]*v/radius + sg[0]*w_x + sg[10] + sg[2]*w_y - sg[2]*v/radius - 0.5*sg[5]*w_y/radius - 0.5*sg[10]*(v*v)/(radius*radius) - sg[3]*v*v_x/(radius*radius) - sg[5]*v*v_y/(radius*radius) - 0.5*sg[5]*v/(radius*radius)
                S[5] = 1.5*c1*(phix*phix)*sg[11] + c1*phix*phix_x*sg[8] + c1*phix*phix_y*sg[7] + 1.5*c1*(phiy*phiy)*sg[11] + c1*phiy*phiy_x*sg[8] + c1*phiy*phiy_y*sg[7] + 3*c1*phiy*sg[11]*v/radius + c1*phiy*sg[7]*v_y/radius + c1*phiy*sg[7]/radius + c1*phiy*sg[8]*v_x/radius + c1*phiy_x*sg[8]*v/radius + c1*phiy_y*sg[7]*v/radius - 3*c1*sg[11] + c1*sg[6]*u_x*v_xx - c1*sg[6]*u_xx*v_x + c1*sg[6]*v_xx + c1*sg[7]*u_x*v_yy - c1*sg[7]*u_yy*v_x + c1*sg[7]*v_yy + 2*c1*sg[8]*u_x*v_xy - 2*c1*sg[8]*u_xy*v_x + 2*c1*sg[8]*v_xy + 4*c1*sg[7]*u_x*w_y/radius - 2*c1*sg[7]*u_y*w_x/radius + 3*c1*sg[7]*w_y/radius + 2*c1*sg[8]*u_x*w_x/radius + 2*c1*sg[8]*w_x/radius + 1.5*c1*sg[11]*(v*v)/(radius*radius) - 3*c1*sg[7]*u_x*v/(radius*radius) + c1*sg[7]*v*v_y/(radius*radius) - c1*sg[7]*v/(radius*radius) + c1*sg[8]*v*v_x/(radius*radius) - 0.5*(phix*phix)*sg[9] - phix*phix_x*sg[5] - phix*phix_y*sg[4] - 0.5*(phiy*phiy)*sg[9] - phiy*phiy_x*sg[5] - phiy*phiy_y*sg[4] - phiy*sg[4]*v_y/radius - phiy*sg[4]/radius - phiy*sg[5]*v_x/radius - phiy*sg[9]*v/radius - phiy_x*sg[5]*v/radius - phiy_y*sg[4]*v/radius + sg[1]*w_y + sg[2]*w_x + sg[9] - sg[1]*v/radius - sg[4]*w_y/radius - 0.5*sg[5]*w_x/radius - sg[4]*v*v_y/(radius*radius) - sg[5]*v*v_x/(radius*radius) - 0.5*sg[9]*(v*v)/(radius*radius)
                S[6] = -1.5*c1*(phix*phix)*sg[11]/radius - c1*phix*phix_x*sg[8]/radius - c1*phix*phix_y*sg[7]/radius - 1.5*c1*(phiy*phiy)*sg[11]/radius - c1*phiy*phiy_x*sg[8]/radius - c1*phiy*phiy_y*sg[7]/radius + 3*c1*phiy*sg[11]*w_y/radius + 3*c1*phiy*sg[12]*w_x/radius - 6*c1*phiy*sg[11]*v/(radius*radius) + c1*phiy*sg[7]*w/(radius*radius*radius) + c1*phiy_x*sg[6]*w_x/radius + c1*phiy_x*sg[8]*w_y/radius - 2*c1*phiy_x*sg[8]*v/(radius*radius) + c1*phiy_y*sg[7]*w_y/radius + c1*phiy_y*sg[8]*w_x/radius - 2*c1*phiy_y*sg[7]*v/(radius*radius) - 3*c1*sg[11]*v_y/radius - 3*c1*sg[12]*v_x/radius - c1*sg[6]*u_x*v_xx/radius + c1*sg[6]*u_xx*v_x/radius - c1*sg[6]*v_xx/radius - c1*sg[7]*u_x*v_yy/radius + c1*sg[7]*u_yy*v_x/radius - c1*sg[7]*v_yy/radius - 2*c1*sg[8]*u_x*v_xy/radius + 2*c1*sg[8]*u_xy*v_x/radius - 2*c1*sg[8]*v_xy/radius + 3*c1*sg[11]*v*w_y/(radius*radius) - 3*c1*sg[11]*w/(radius*radius) + 3*c1*sg[12]*v*w_x/(radius*radius) + c1*sg[6]*v_x*w_x/(radius*radius) - 3*c1*sg[7]*u_x*w_y/(radius*radius) + c1*sg[7]*u_y*w_x/(radius*radius) + c1*sg[7]*v_y*w_y/(radius*radius) - c1*sg[7]*w_y/(radius*radius) - 2*c1*sg[8]*u_x*w_x/(radius*radius) + c1*sg[8]*v_x*w_y/(radius*radius) + c1*sg[8]*v_y*w_x/(radius*radius) - c1*sg[8]*w_x/(radius*radius) - 4.5*c1*sg[11]*(v*v)/(radius*radius*radius) + 2*c1*sg[7]*u_x*v/(radius*radius*radius) - c1*sg[7]*v*v_y/(radius*radius*radius) - c1*sg[8]*v*v_x/(radius*radius*radius) + c1*sg[7]*v*w/(radius*radius*radius*radius) + 0.5*(phix*phix)*sg[9]/radius + phix*phix_x*sg[5]/radius + phix*phix_y*sg[4]/radius + 0.5*(phiy*phiy)*sg[9]/radius + phiy*phiy_x*sg[5]/radius + phiy*phiy_y*sg[4]/radius - phiy*sg[10]*w_x/radius - phiy*sg[9]*w_y/radius + 2*phiy*sg[9]*v/(radius*radius) - phiy*sg[4]*w/(radius*radius*radius) - phiy_x*sg[3]*w_x/radius - phiy_x*sg[5]*w_y/radius + 2*phiy_x*sg[5]*v/(radius*radius) - phiy_y*sg[4]*w_y/radius - phiy_y*sg[5]*w_x/radius + 2*phiy_y*sg[4]*v/(radius*radius) + sg[10]*v_x/radius - sg[1]*w_y/radius - sg[2]*w_x/radius + sg[9]*v_y/radius - sg[10]*v*w_x/(radius*radius) + sg[1]*v/(radius*radius) - sg[3]*v_x*w_x/(radius*radius) - sg[4]*v_y*w_y/(radius*radius) - sg[5]*v_x*w_y/(radius*radius) - sg[5]*v_y*w_x/(radius*radius) - 0.5*sg[5]*w_x/(radius*radius) - sg[9]*v*w_y/(radius*radius) + sg[9]*w/(radius*radius) + sg[4]*v*v_y/(radius*radius*radius) + sg[5]*v*v_x/(radius*radius*radius) + 1.5*sg[9]*(v*v)/(radius*radius*radius) - sg[4]*v*w/(radius*radius*radius*radius)
                S[7] = 0.5*c1*(phix*phix)*sg[7]/(radius*radius) + 0.5*c1*(phiy*phiy)*sg[7]/(radius*radius) - 3*c1*phiy*sg[11]/radius + c1*phiy*sg[7]*v/(radius*radius*radius) - c1*phiy_x*sg[8]/radius - c1*phiy_y*sg[7]/radius - c1*sg[6]*u_x*w_xx/radius + c1*sg[6]*u_xx*w_x/radius - c1*sg[6]*w_xx/radius - c1*sg[7]*u_x*w_yy/radius + c1*sg[7]*u_yy*w_x/radius - c1*sg[7]*w_yy/radius - 2*c1*sg[8]*u_x*w_xy/radius + 2*c1*sg[8]*u_xy*w_x/radius - 2*c1*sg[8]*w_xy/radius - 3*c1*sg[11]*v/(radius*radius) + 3*c1*sg[7]*u_x*v_y/(radius*radius) + 2*c1*sg[7]*u_x/(radius*radius) - c1*sg[7]*u_y*v_x/(radius*radius) + c1*sg[7]*v_y/(radius*radius) + 2*c1*sg[8]*u_x*v_x/(radius*radius) + c1*sg[8]*v_x/(radius*radius) + 2*c1*sg[7]*u_x*w/(radius*radius*radius) + c1*sg[7]*w/(radius*radius*radius) + 0.5*c1*sg[7]*(v*v)/(radius*radius*radius*radius) - 0.5*(phix*phix)*sg[4]/(radius*radius) - 0.5*(phiy*phiy)*sg[4]/(radius*radius) + phiy*sg[9]/radius - phiy*sg[4]*v/(radius*radius*radius) + phiy_x*sg[5]/radius + phiy_y*sg[4]/radius + sg[1]*v_y/radius + sg[1]/radius + sg[2]*v_x/radius + sg[1]*w/(radius*radius) + 0.5*sg[5]*v_x/(radius*radius) + sg[9]*v/(radius*radius) - sg[4]*w/(radius*radius*radius) - 0.5*sg[4]*(v*v)/(radius*radius*radius*radius)
                S[8] = 3*c1*phix*sg[11]*w_y + 3*c1*phix*sg[12]*w_x - 3*c1*phix*sg[11]*v/radius + c1*phix*sg[7]*v_y/radius + c1*phix*sg[7]/radius + c1*phix*sg[8]*v_x/radius + c1*phix*sg[7]*w/(radius*radius) + c1*phix_x*sg[6]*w_x + c1*phix_x*sg[8]*w_y - c1*phix_x*sg[8]*v/radius + c1*phix_y*sg[7]*w_y + c1*phix_y*sg[8]*w_x - c1*phix_y*sg[7]*v/radius - 3*c1*sg[11]*u_y - 3*c1*sg[12]*u_x - 3*c1*sg[12] - phix*sg[10]*w_x - phix*sg[9]*w_y - phix*sg[4]*v_y/radius - phix*sg[4]/radius - phix*sg[5]*v_x/radius + phix*sg[9]*v/radius - phix*sg[4]*w/(radius*radius) - phix_x*sg[3]*w_x - phix_x*sg[5]*w_y + phix_x*sg[5]*v/radius - phix_y*sg[4]*w_y - phix_y*sg[5]*w_x + phix_y*sg[4]*v/radius + sg[10]*u_x + sg[10] + sg[9]*u_y
                S[9] = 3*c1*phiy*sg[11]*w_y + 3*c1*phiy*sg[12]*w_x - 3*c1*phiy*sg[11]*v/radius + c1*phiy*sg[7]*v_y/radius + c1*phiy*sg[7]/radius + c1*phiy*sg[8]*v_x/radius + c1*phiy*sg[7]*w/(radius*radius) + c1*phiy_x*sg[6]*w_x + c1*phiy_x*sg[8]*w_y - c1*phiy_x*sg[8]*v/radius + c1*phiy_y*sg[7]*w_y + c1*phiy_y*sg[8]*w_x - c1*phiy_y*sg[7]*v/radius - 3*c1*sg[11]*v_y - 3*c1*sg[11] - 3*c1*sg[12]*v_x + 3*c1*sg[11]*v*w_y/radius - 3*c1*sg[11]*w/radius + 3*c1*sg[12]*v*w_x/radius + c1*sg[6]*v_x*w_x/radius + c1*sg[7]*v_y*w_y/radius + c1*sg[7]*w_y/radius + c1*sg[8]*v_x*w_y/radius + c1*sg[8]*v_y*w_x/radius + c1*sg[8]*w_x/radius - 3*c1*sg[11]*(v*v)/(radius*radius) + c1*sg[7]*v*w/(radius*radius*radius) - phiy*sg[10]*w_x - phiy*sg[9]*w_y - phiy*sg[4]*v_y/radius - phiy*sg[4]/radius - phiy*sg[5]*v_x/radius + phiy*sg[9]*v/radius - phiy*sg[4]*w/(radius*radius) - phiy_x*sg[3]*w_x - phiy_x*sg[5]*w_y + phiy_x*sg[5]*v/radius - phiy_y*sg[4]*w_y - phiy_y*sg[5]*w_x + phiy_y*sg[4]*v/radius + sg[10]*v_x + sg[9]*v_y + sg[9] - sg[10]*v*w_x/radius - sg[3]*v_x*w_x/radius - sg[4]*v_y*w_y/radius - sg[4]*w_y/radius - sg[5]*v_x*w_y/radius - sg[5]*v_y*w_x/radius - sg[5]*w_x/radius - sg[9]*v*w_y/radius + sg[9]*w/radius + sg[9]*(v*v)/(radius*radius) - sg[4]*v*w/(radius*radius*radius)
                S[10] = c1*phix*sg[6]*w_x + c1*phix*sg[8]*w_y - c1*phix*sg[8]*v/radius - c1*sg[6]*u_x - c1*sg[6] - c1*sg[8]*u_y - phix*sg[3]*w_x - phix*sg[5]*w_y + phix*sg[5]*v/radius + sg[3]*u_x + sg[3] + sg[5]*u_y
                S[11] = c1*phix*sg[7]*w_y + c1*phix*sg[8]*w_x - c1*phix*sg[7]*v/radius - c1*sg[7]*u_y - c1*sg[8]*u_x - c1*sg[8] - phix*sg[4]*w_y - phix*sg[5]*w_x + phix*sg[4]*v/radius + sg[4]*u_y + sg[5]*u_x + sg[5]
                S[12] = c1*phiy*sg[6]*w_x + c1*phiy*sg[8]*w_y - c1*phiy*sg[8]*v/radius - c1*sg[6]*v_x - c1*sg[8]*v_y - c1*sg[8] + c1*sg[6]*v*w_x/radius + c1*sg[8]*v*w_y/radius - c1*sg[8]*w/radius - c1*sg[8]*(v*v)/(radius*radius) - phiy*sg[3]*w_x - phiy*sg[5]*w_y + phiy*sg[5]*v/radius + sg[3]*v_x + sg[5]*v_y + sg[5] - sg[3]*v*w_x/radius - sg[5]*v*w_y/radius + sg[5]*w/radius + sg[5]*(v*v)/(radius*radius)
                S[13] = c1*phiy*sg[7]*w_y + c1*phiy*sg[8]*w_x - c1*phiy*sg[7]*v/radius - c1*sg[7]*v_y - c1*sg[7] - c1*sg[8]*v_x + c1*sg[7]*v*w_y/radius - c1*sg[7]*w/radius + c1*sg[8]*v*w_x/radius - c1*sg[7]*(v*v)/(radius*radius) - phiy*sg[4]*w_y - phiy*sg[5]*w_x + phiy*sg[4]*v/radius + sg[4]*v_y + sg[4] + sg[5]*v_x - sg[4]*v*w_y/radius + sg[4]*w/radius - sg[5]*v*w_x/radius + sg[4]*(v*v)/(radius*radius)
                S[14] = -c1*sg[6]*u_x*v_y - c1*sg[6]*u_x + c1*sg[6]*u_y*v_x - c1*sg[6]*v_y - c1*sg[6] - c1*sg[6]*u_x*w/radius - c1*sg[6]*w/radius
                S[15] = -c1*sg[7]*u_x*v_y - c1*sg[7]*u_x + c1*sg[7]*u_y*v_x - c1*sg[7]*v_y - c1*sg[7] - c1*sg[7]*u_x*w/radius - c1*sg[7]*w/radius
                S[16] = -2*c1*sg[8]*u_x*v_y - 2*c1*sg[8]*u_x + 2*c1*sg[8]*u_y*v_x - 2*c1*sg[8]*v_y - 2*c1*sg[8] - 2*c1*sg[8]*u_x*w/radius - 2*c1*sg[8]*w/radius
                S[17] = -c1*sg[6]*v_x*w_y + c1*sg[6]*v_y*w_x + c1*sg[6]*w_x + c1*sg[6]*v*v_x/radius + c1*sg[6]*w*w_x/radius
                S[18] = -c1*sg[7]*v_x*w_y + c1*sg[7]*v_y*w_x + c1*sg[7]*w_x + c1*sg[7]*v*v_x/radius + c1*sg[7]*w*w_x/radius
                S[19] = -2*c1*sg[8]*v_x*w_y + 2*c1*sg[8]*v_y*w_x + 2*c1*sg[8]*w_x + 2*c1*sg[8]*v*v_x/radius + 2*c1*sg[8]*w*w_x/radius
                S[20] = c1*sg[6]*u_x*w_y - c1*sg[6]*u_y*w_x + c1*sg[6]*w_y - c1*sg[6]*u_x*v/radius - c1*sg[6]*v/radius
                S[21] = c1*sg[7]*u_x*w_y - c1*sg[7]*u_y*w_x + c1*sg[7]*w_y - c1*sg[7]*u_x*v/radius - c1*sg[7]*v/radius
                S[22] = 2*c1*sg[8]*u_x*w_y - 2*c1*sg[8]*u_y*w_x + 2*c1*sg[8]*w_y - 2*c1*sg[8]*u_x*v/radius - 2*c1*sg[8]*v/radius

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
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAphix*gAphix
                        pA[9] = fAphiy*gAphiy
                        pA[10] = fAphix_x*gAphix
                        pA[11] = fAphix*gAphix_y
                        pA[12] = fAphiy_x*gAphiy
                        pA[13] = fAphiy*gAphiy_y
                        pA[14] = fAw_xx*gAw
                        pA[15] = fAw*gAw_yy
                        pA[16] = fAw_x*gAw_y
                        pA[17] = fAu_xx*gAu
                        pA[18] = fAu*gAu_yy
                        pA[19] = fAu_x*gAu_y
                        pA[20] = fAv_xx*gAv
                        pA[21] = fAv*gAv_yy
                        pA[22] = fAv_x*gAv_y
                        fint[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1] + pA[17]*S[17] + pA[18]*S[18] + pA[19]*S[19])
                        fint[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[6]*S[6] + pA[20]*S[20] + pA[21]*S[21] + pA[22]*S[22])
                        fint[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[7]*S[7] + pA[14]*S[14] + pA[15]*S[15] + pA[16]*S[16])
                        fint[col+3] += weight*(pA[8]*S[8] + pA[10]*S[10] + pA[11]*S[11])
                        fint[col+4] += weight*(pA[9]*S[9] + pA[12]*S[12] + pA[13]*S[13])

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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the first and second derivatives of `u`, `v`, `w`, the values of `v` and `w`, the rotations and their gradients), the strains
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
    cdef double h, c1
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w
    cdef double phix, phiy, phix_x, phix_y, phiy_x, phiy_y, w_xx, w_yy
    cdef double w_xy, u_xx, u_yy, u_xy, v_xx, v_yy, v_xy
    cdef double du_x, du_y, dv_x, dv_y, dw_x, dw_y, dv, dw
    cdef double dphix, dphiy, dphix_x, dphix_y, dphiy_x, dphiy_y, dw_xx, dw_yy
    cdef double dw_xy, du_xx, du_yy, du_xy, dv_xx, dv_yy, dv_xy
    cdef double e[13]
    cdef double de[13]
    cdef double sg[13]
    cdef double dsg[13]
    cdef double S[23]
    cdef double pA[23]
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
    #NOTE the traction-free faces are assumed at z = +-h/2 about the
    #     mid-surface
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    radius = shell.r
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
                v = 0.
                w = 0.
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
                dv = 0.
                dw = 0.
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
                        v += fAv*gAv*cs[col+1]
                        w += fAw*gAw*cs[col+2]
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
                        dv += fAv*gAv*vs[col+1]
                        dw += fAw*gAw*vs[col+2]
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
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x - phiy*v_x*w_x/radius + phiy_x*v_x - phiy_x*v*w_x/radius + (v_x*v_x)/radius - v*v_x*w_x/(radius*radius)
                e[4] = -0.5*(phix*phix)*v_y/radius - 0.5*(phix*phix)/radius - 0.5*(phix*phix)*w/(radius*radius) - phix*phix_y*w_y + phix*phix_y*v/radius + phix_y*u_y - 0.5*(phiy*phiy)*v_y/radius - 0.5*(phiy*phiy)/radius - 0.5*(phiy*phiy)*w/(radius*radius) - phiy*phiy_y*w_y + phiy*phiy_y*v/radius - phiy*v_y*w_y/radius - phiy*w_y/radius - phiy*v*w/(radius*radius*radius) + phiy_y*v_y + phiy_y - phiy_y*v*w_y/radius + phiy_y*w/radius + phiy_y*(v*v)/(radius*radius) - 0.5*(u_y*u_y)/radius + 0.5*(v_y*v_y)/radius + v_y/radius - 0.5*(w_y*w_y)/radius - v*v_y*w_y/(radius*radius) + 0.5*(v*v)*v_y/(radius*radius*radius) - 0.5*(w*w)/(radius*radius*radius) - 0.5*(v*v)*w/(radius*radius*radius*radius)
                e[5] = -0.5*(phix*phix)*v_x/radius - phix*phix_x*w_y + phix*phix_x*v/radius - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - 0.5*(phiy*phiy)*v_x/radius - phiy*phiy_x*w_y + phiy*phiy_x*v/radius - phiy*phiy_y*w_x - phiy*v_x*w_y/radius - phiy*v_y*w_x/radius - phiy*w_x/radius + phiy_x*v_y + phiy_x - phiy_x*v*w_y/radius + phiy_x*w/radius + phiy_x*(v*v)/(radius*radius) + phiy_y*v_x - phiy_y*v*w_x/radius - 0.5*u_x*u_y/radius - 0.5*u_y/radius + 1.5*v_x*v_y/radius + 1.5*v_x/radius - 0.5*w_x*w_y/radius - v*v_x*w_y/(radius*radius) - v*v_y*w_x/(radius*radius) - 0.5*v*w_x/(radius*radius) + 0.5*v_x*w/(radius*radius) + 0.5*(v*v)*v_x/(radius*radius*radius)
                e[6] = c1*phix*phix_x*w_x - c1*phix_x*u_x - c1*phix_x + c1*phiy*phiy_x*w_x + c1*phiy*v_x*w_x/radius - c1*phiy_x*v_x + c1*phiy_x*v*w_x/radius + c1*u_x*v_xx*w_y - c1*u_x*v_y*w_xx - c1*u_x*w_xx - c1*u_xx*v_x*w_y + c1*u_xx*v_y*w_x + c1*u_xx*w_x + c1*u_y*v_x*w_xx - c1*u_y*v_xx*w_x + c1*v_xx*w_y - c1*v_y*w_xx - c1*w_xx - c1*u_x*v*v_xx/radius - c1*u_x*w*w_xx/radius + c1*u_xx*v*v_x/radius + c1*u_xx*w*w_x/radius - c1*v*v_xx/radius - c1*(v_x*v_x)/radius - c1*w*w_xx/radius + c1*v*v_x*w_x/(radius*radius)
                e[7] = 0.5*c1*(phix*phix)*v_y/radius + 0.5*c1*(phix*phix)/radius + 0.5*c1*(phix*phix)*w/(radius*radius) + c1*phix*phix_y*w_y - c1*phix*phix_y*v/radius - c1*phix_y*u_y + 0.5*c1*(phiy*phiy)*v_y/radius + 0.5*c1*(phiy*phiy)/radius + 0.5*c1*(phiy*phiy)*w/(radius*radius) + c1*phiy*phiy_y*w_y - c1*phiy*phiy_y*v/radius + c1*phiy*v_y*w_y/radius + c1*phiy*w_y/radius + c1*phiy*v*w/(radius*radius*radius) - c1*phiy_y*v_y - c1*phiy_y + c1*phiy_y*v*w_y/radius - c1*phiy_y*w/radius - c1*phiy_y*(v*v)/(radius*radius) - c1*u_x*v_y*w_yy + c1*u_x*v_yy*w_y - c1*u_x*w_yy + c1*u_y*v_x*w_yy - c1*u_y*v_yy*w_x - c1*u_yy*v_x*w_y + c1*u_yy*v_y*w_x + c1*u_yy*w_x - c1*v_y*w_yy + c1*v_yy*w_y - c1*w_yy - 0.5*c1*(u_x*u_x)/radius - c1*u_x*v*v_yy/radius + 2*c1*u_x*(v_y*v_y)/radius + 3*c1*u_x*v_y/radius - c1*u_x*w*w_yy/radius + 2*c1*u_x*(w_y*w_y)/radius - 0.5*c1*(u_y*u_y)/radius - 2*c1*u_y*v_x*v_y/radius - c1*u_y*v_x/radius - 2*c1*u_y*w_x*w_y/radius + c1*u_yy*v*v_x/radius + c1*u_yy*w*w_x/radius - c1*v*v_yy/radius - 0.5*c1*(v_x*v_x)/radius + 0.5*c1*(v_y*v_y)/radius - c1*w*w_yy/radius - 0.5*c1*(w_x*w_x)/radius + 1.5*c1*(w_y*w_y)/radius - 3*c1*u_x*v*w_y/(radius*radius) + 3*c1*u_x*v_y*w/(radius*radius) + 2*c1*u_x*w/(radius*radius) + c1*u_y*v*w_x/(radius*radius) - c1*u_y*v_x*w/(radius*radius) + c1*v*v_y*w_y/(radius*radius) - c1*v*w_y/(radius*radius) + c1*v_y*w/(radius*radius) + c1*u_x*(v*v)/(radius*radius*radius) + c1*u_x*(w*w)/(radius*radius*radius) - 0.5*c1*(v*v)*v_y/(radius*radius*radius) + 0.5*c1*(w*w)/(radius*radius*radius) + 0.5*c1*(v*v)*w/(radius*radius*radius*radius)
                e[8] = 0.5*c1*(phix*phix)*v_x/radius + c1*phix*phix_x*w_y - c1*phix*phix_x*v/radius + c1*phix*phix_y*w_x - c1*phix_x*u_y - c1*phix_y*u_x - c1*phix_y + 0.5*c1*(phiy*phiy)*v_x/radius + c1*phiy*phiy_x*w_y - c1*phiy*phiy_x*v/radius + c1*phiy*phiy_y*w_x + c1*phiy*v_x*w_y/radius + c1*phiy*v_y*w_x/radius + c1*phiy*w_x/radius - c1*phiy_x*v_y - c1*phiy_x + c1*phiy_x*v*w_y/radius - c1*phiy_x*w/radius - c1*phiy_x*(v*v)/(radius*radius) - c1*phiy_y*v_x + c1*phiy_y*v*w_x/radius + 2*c1*u_x*v_xy*w_y - 2*c1*u_x*v_y*w_xy - 2*c1*u_x*w_xy - 2*c1*u_xy*v_x*w_y + 2*c1*u_xy*v_y*w_x + 2*c1*u_xy*w_x + 2*c1*u_y*v_x*w_xy - 2*c1*u_y*v_xy*w_x + 2*c1*v_xy*w_y - 2*c1*v_y*w_xy - 2*c1*w_xy - 2*c1*u_x*v*v_xy/radius + 2*c1*u_x*v_x*v_y/radius + 2*c1*u_x*v_x/radius - 2*c1*u_x*w*w_xy/radius + 2*c1*u_x*w_x*w_y/radius + 2*c1*u_xy*v*v_x/radius + 2*c1*u_xy*w*w_x/radius - 2*c1*u_y*(v_x*v_x)/radius - 2*c1*u_y*(w_x*w_x)/radius - 2*c1*v*v_xy/radius - 2*c1*w*w_xy/radius + 2*c1*w_x*w_y/radius - 2*c1*u_x*v*w_x/(radius*radius) + 2*c1*u_x*v_x*w/(radius*radius) + c1*v*v_x*w_y/(radius*radius) + c1*v*v_y*w_x/(radius*radius) - c1*v*w_x/(radius*radius) + c1*v_x*w/(radius*radius) - 0.5*c1*(v*v)*v_x/(radius*radius*radius)
                e[9] = -0.5*(phix*phix)*w_y + 0.5*(phix*phix)*v/radius + phix*u_y - 0.5*(phiy*phiy)*w_y + 0.5*(phiy*phiy)*v/radius + phiy*v_y + phiy - phiy*v*w_y/radius + phiy*w/radius + phiy*(v*v)/(radius*radius) + w_y + v*v_y/radius - 0.5*(v*v)*w_y/(radius*radius) + v*w/(radius*radius) + 0.5*(v*v*v)/(radius*radius*radius)
                e[10] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x - phiy*v*w_x/radius + w_x + v*v_x/radius - 0.5*(v*v)*w_x/(radius*radius)
                e[11] = 1.5*c1*(phix*phix)*w_y - 1.5*c1*(phix*phix)*v/radius - 3*c1*phix*u_y + 1.5*c1*(phiy*phiy)*w_y - 1.5*c1*(phiy*phiy)*v/radius - 3*c1*phiy*v_y - 3*c1*phiy + 3*c1*phiy*v*w_y/radius - 3*c1*phiy*w/radius - 3*c1*phiy*(v*v)/(radius*radius) - 3*c1*w_y - 3*c1*v*v_y/radius + 1.5*c1*(v*v)*w_y/(radius*radius) - 3*c1*v*w/(radius*radius) - 1.5*c1*(v*v*v)/(radius*radius*radius)
                e[12] = 1.5*c1*(phix*phix)*w_x - 3*c1*phix*u_x - 3*c1*phix + 1.5*c1*(phiy*phiy)*w_x - 3*c1*phiy*v_x + 3*c1*phiy*v*w_x/radius - 3*c1*w_x - 3*c1*v*v_x/radius + 1.5*c1*(v*v)*w_x/(radius*radius)
                de[0] = du_x*u_x + du_x + dv_x*v_x + dw_x*w_x
                de[1] = du_y*u_y - dv*w_y/radius + dv*v/(radius*radius) + dv_y*v_y + dv_y + dv_y*w/radius + dw*v_y/radius + dw/radius + dw*w/(radius*radius) + dw_y*w_y - dw_y*v/radius
                de[2] = du_x*u_y + du_y*u_x + du_y - dv*w_x/radius + dv_x*v_y + dv_x + dv_x*w/radius + dv_y*v_x + dw*v_x/radius + dw_x*w_y - dw_x*v/radius + dw_y*w_x
                de[3] = -dphix*phix_x*w_x - dphix_x*phix*w_x + dphix_x*u_x + dphix_x - dphiy*phiy_x*w_x - dphiy*v_x*w_x/radius - dphiy_x*phiy*w_x + dphiy_x*v_x - dphiy_x*v*w_x/radius + du_x*phix_x - dv*phiy_x*w_x/radius - dv*v_x*w_x/(radius*radius) - dv_x*phiy*w_x/radius + dv_x*phiy_x + 2*dv_x*v_x/radius - dv_x*v*w_x/(radius*radius) - dw_x*phix*phix_x - dw_x*phiy*phiy_x - dw_x*phiy*v_x/radius - dw_x*phiy_x*v/radius - dw_x*v*v_x/(radius*radius)
                de[4] = -dphix*phix*v_y/radius - dphix*phix/radius - dphix*phix*w/(radius*radius) - dphix*phix_y*w_y + dphix*phix_y*v/radius - dphix_y*phix*w_y + dphix_y*phix*v/radius + dphix_y*u_y - dphiy*phiy*v_y/radius - dphiy*phiy/radius - dphiy*phiy*w/(radius*radius) - dphiy*phiy_y*w_y + dphiy*phiy_y*v/radius - dphiy*v_y*w_y/radius - dphiy*w_y/radius - dphiy*v*w/(radius*radius*radius) - dphiy_y*phiy*w_y + dphiy_y*phiy*v/radius + dphiy_y*v_y + dphiy_y - dphiy_y*v*w_y/radius + dphiy_y*w/radius + dphiy_y*(v*v)/(radius*radius) + du_y*phix_y - du_y*u_y/radius + dv*phix*phix_y/radius + dv*phiy*phiy_y/radius - dv*phiy*w/(radius*radius*radius) - dv*phiy_y*w_y/radius + 2*dv*phiy_y*v/(radius*radius) - dv*v_y*w_y/(radius*radius) + dv*v*v_y/(radius*radius*radius) - dv*v*w/(radius*radius*radius*radius) - 0.5*dv_y*(phix*phix)/radius - 0.5*dv_y*(phiy*phiy)/radius - dv_y*phiy*w_y/radius + dv_y*phiy_y + dv_y*v_y/radius + dv_y/radius - dv_y*v*w_y/(radius*radius) + 0.5*dv_y*(v*v)/(radius*radius*radius) - 0.5*dw*(phix*phix)/(radius*radius) - 0.5*dw*(phiy*phiy)/(radius*radius) - dw*phiy*v/(radius*radius*radius) + dw*phiy_y/radius - dw*w/(radius*radius*radius) - 0.5*dw*(v*v)/(radius*radius*radius*radius) - dw_y*phix*phix_y - dw_y*phiy*phiy_y - dw_y*phiy*v_y/radius - dw_y*phiy/radius - dw_y*phiy_y*v/radius - dw_y*w_y/radius - dw_y*v*v_y/(radius*radius)
                de[5] = -dphix*phix*v_x/radius - dphix*phix_x*w_y + dphix*phix_x*v/radius - dphix*phix_y*w_x - dphix_x*phix*w_y + dphix_x*phix*v/radius + dphix_x*u_y - dphix_y*phix*w_x + dphix_y*u_x + dphix_y - dphiy*phiy*v_x/radius - dphiy*phiy_x*w_y + dphiy*phiy_x*v/radius - dphiy*phiy_y*w_x - dphiy*v_x*w_y/radius - dphiy*v_y*w_x/radius - dphiy*w_x/radius - dphiy_x*phiy*w_y + dphiy_x*phiy*v/radius + dphiy_x*v_y + dphiy_x - dphiy_x*v*w_y/radius + dphiy_x*w/radius + dphiy_x*(v*v)/(radius*radius) - dphiy_y*phiy*w_x + dphiy_y*v_x - dphiy_y*v*w_x/radius + du_x*phix_y - 0.5*du_x*u_y/radius + du_y*phix_x - 0.5*du_y*u_x/radius - 0.5*du_y/radius + dv*phix*phix_x/radius + dv*phiy*phiy_x/radius - dv*phiy_x*w_y/radius + 2*dv*phiy_x*v/(radius*radius) - dv*phiy_y*w_x/radius - dv*v_x*w_y/(radius*radius) - dv*v_y*w_x/(radius*radius) - 0.5*dv*w_x/(radius*radius) + dv*v*v_x/(radius*radius*radius) - 0.5*dv_x*(phix*phix)/radius - 0.5*dv_x*(phiy*phiy)/radius - dv_x*phiy*w_y/radius + dv_x*phiy_y + 1.5*dv_x*v_y/radius + 1.5*dv_x/radius - dv_x*v*w_y/(radius*radius) + 0.5*dv_x*w/(radius*radius) + 0.5*dv_x*(v*v)/(radius*radius*radius) - dv_y*phiy*w_x/radius + dv_y*phiy_x + 1.5*dv_y*v_x/radius - dv_y*v*w_x/(radius*radius) + dw*phiy_x/radius + 0.5*dw*v_x/(radius*radius) - dw_x*phix*phix_y - dw_x*phiy*phiy_y - dw_x*phiy*v_y/radius - dw_x*phiy/radius - dw_x*phiy_y*v/radius - 0.5*dw_x*w_y/radius - dw_x*v*v_y/(radius*radius) - 0.5*dw_x*v/(radius*radius) - dw_y*phix*phix_x - dw_y*phiy*phiy_x - dw_y*phiy*v_x/radius - dw_y*phiy_x*v/radius - 0.5*dw_y*w_x/radius - dw_y*v*v_x/(radius*radius)
                de[6] = c1*dphix*phix_x*w_x + c1*dphix_x*phix*w_x - c1*dphix_x*u_x - c1*dphix_x + c1*dphiy*phiy_x*w_x + c1*dphiy*v_x*w_x/radius + c1*dphiy_x*phiy*w_x - c1*dphiy_x*v_x + c1*dphiy_x*v*w_x/radius - c1*du_x*phix_x + c1*du_x*v_xx*w_y - c1*du_x*v_y*w_xx - c1*du_x*w_xx - c1*du_x*v*v_xx/radius - c1*du_x*w*w_xx/radius - c1*du_xx*v_x*w_y + c1*du_xx*v_y*w_x + c1*du_xx*w_x + c1*du_xx*v*v_x/radius + c1*du_xx*w*w_x/radius + c1*du_y*v_x*w_xx - c1*du_y*v_xx*w_x + c1*dv*phiy_x*w_x/radius - c1*dv*u_x*v_xx/radius + c1*dv*u_xx*v_x/radius - c1*dv*v_xx/radius + c1*dv*v_x*w_x/(radius*radius) + c1*dv_x*phiy*w_x/radius - c1*dv_x*phiy_x - c1*dv_x*u_xx*w_y + c1*dv_x*u_y*w_xx + c1*dv_x*u_xx*v/radius - 2*c1*dv_x*v_x/radius + c1*dv_x*v*w_x/(radius*radius) + c1*dv_xx*u_x*w_y - c1*dv_xx*u_y*w_x + c1*dv_xx*w_y - c1*dv_xx*u_x*v/radius - c1*dv_xx*v/radius - c1*dv_y*u_x*w_xx + c1*dv_y*u_xx*w_x - c1*dv_y*w_xx - c1*dw*u_x*w_xx/radius + c1*dw*u_xx*w_x/radius - c1*dw*w_xx/radius + c1*dw_x*phix*phix_x + c1*dw_x*phiy*phiy_x + c1*dw_x*phiy*v_x/radius + c1*dw_x*phiy_x*v/radius + c1*dw_x*u_xx*v_y + c1*dw_x*u_xx - c1*dw_x*u_y*v_xx + c1*dw_x*u_xx*w/radius + c1*dw_x*v*v_x/(radius*radius) - c1*dw_xx*u_x*v_y - c1*dw_xx*u_x + c1*dw_xx*u_y*v_x - c1*dw_xx*v_y - c1*dw_xx - c1*dw_xx*u_x*w/radius - c1*dw_xx*w/radius + c1*dw_y*u_x*v_xx - c1*dw_y*u_xx*v_x + c1*dw_y*v_xx
                de[7] = c1*dphix*phix*v_y/radius + c1*dphix*phix/radius + c1*dphix*phix*w/(radius*radius) + c1*dphix*phix_y*w_y - c1*dphix*phix_y*v/radius + c1*dphix_y*phix*w_y - c1*dphix_y*phix*v/radius - c1*dphix_y*u_y + c1*dphiy*phiy*v_y/radius + c1*dphiy*phiy/radius + c1*dphiy*phiy*w/(radius*radius) + c1*dphiy*phiy_y*w_y - c1*dphiy*phiy_y*v/radius + c1*dphiy*v_y*w_y/radius + c1*dphiy*w_y/radius + c1*dphiy*v*w/(radius*radius*radius) + c1*dphiy_y*phiy*w_y - c1*dphiy_y*phiy*v/radius - c1*dphiy_y*v_y - c1*dphiy_y + c1*dphiy_y*v*w_y/radius - c1*dphiy_y*w/radius - c1*dphiy_y*(v*v)/(radius*radius) - c1*du_x*v_y*w_yy + c1*du_x*v_yy*w_y - c1*du_x*w_yy - c1*du_x*u_x/radius - c1*du_x*v*v_yy/radius + 2*c1*du_x*(v_y*v_y)/radius + 3*c1*du_x*v_y/radius - c1*du_x*w*w_yy/radius + 2*c1*du_x*(w_y*w_y)/radius - 3*c1*du_x*v*w_y/(radius*radius) + 3*c1*du_x*v_y*w/(radius*radius) + 2*c1*du_x*w/(radius*radius) + c1*du_x*(v*v)/(radius*radius*radius) + c1*du_x*(w*w)/(radius*radius*radius) - c1*du_y*phix_y + c1*du_y*v_x*w_yy - c1*du_y*v_yy*w_x - c1*du_y*u_y/radius - 2*c1*du_y*v_x*v_y/radius - c1*du_y*v_x/radius - 2*c1*du_y*w_x*w_y/radius + c1*du_y*v*w_x/(radius*radius) - c1*du_y*v_x*w/(radius*radius) - c1*du_yy*v_x*w_y + c1*du_yy*v_y*w_x + c1*du_yy*w_x + c1*du_yy*v*v_x/radius + c1*du_yy*w*w_x/radius - c1*dv*phix*phix_y/radius - c1*dv*phiy*phiy_y/radius + c1*dv*phiy*w/(radius*radius*radius) + c1*dv*phiy_y*w_y/radius - 2*c1*dv*phiy_y*v/(radius*radius) - c1*dv*u_x*v_yy/radius + c1*dv*u_yy*v_x/radius - c1*dv*v_yy/radius - 3*c1*dv*u_x*w_y/(radius*radius) + c1*dv*u_y*w_x/(radius*radius) + c1*dv*v_y*w_y/(radius*radius) - c1*dv*w_y/(radius*radius) + 2*c1*dv*u_x*v/(radius*radius*radius) - c1*dv*v*v_y/(radius*radius*radius) + c1*dv*v*w/(radius*radius*radius*radius) + c1*dv_x*u_y*w_yy - c1*dv_x*u_yy*w_y - 2*c1*dv_x*u_y*v_y/radius - c1*dv_x*u_y/radius + c1*dv_x*u_yy*v/radius - c1*dv_x*v_x/radius - c1*dv_x*u_y*w/(radius*radius) + 0.5*c1*dv_y*(phix*phix)/radius + 0.5*c1*dv_y*(phiy*phiy)/radius + c1*dv_y*phiy*w_y/radius - c1*dv_y*phiy_y - c1*dv_y*u_x*w_yy + c1*dv_y*u_yy*w_x - c1*dv_y*w_yy + 4*c1*dv_y*u_x*v_y/radius + 3*c1*dv_y*u_x/radius - 2*c1*dv_y*u_y*v_x/radius + c1*dv_y*v_y/radius + 3*c1*dv_y*u_x*w/(radius*radius) + c1*dv_y*v*w_y/(radius*radius) + c1*dv_y*w/(radius*radius) - 0.5*c1*dv_y*(v*v)/(radius*radius*radius) + c1*dv_yy*u_x*w_y - c1*dv_yy*u_y*w_x + c1*dv_yy*w_y - c1*dv_yy*u_x*v/radius - c1*dv_yy*v/radius + 0.5*c1*dw*(phix*phix)/(radius*radius) + 0.5*c1*dw*(phiy*phiy)/(radius*radius) + c1*dw*phiy*v/(radius*radius*radius) - c1*dw*phiy_y/radius - c1*dw*u_x*w_yy/radius + c1*dw*u_yy*w_x/radius - c1*dw*w_yy/radius + 3*c1*dw*u_x*v_y/(radius*radius) + 2*c1*dw*u_x/(radius*radius) - c1*dw*u_y*v_x/(radius*radius) + c1*dw*v_y/(radius*radius) + 2*c1*dw*u_x*w/(radius*radius*radius) + c1*dw*w/(radius*radius*radius) + 0.5*c1*dw*(v*v)/(radius*radius*radius*radius) - c1*dw_x*u_y*v_yy + c1*dw_x*u_yy*v_y + c1*dw_x*u_yy - 2*c1*dw_x*u_y*w_y/radius + c1*dw_x*u_yy*w/radius - c1*dw_x*w_x/radius + c1*dw_x*u_y*v/(radius*radius) + c1*dw_y*phix*phix_y + c1*dw_y*phiy*phiy_y + c1*dw_y*phiy*v_y/radius + c1*dw_y*phiy/radius + c1*dw_y*phiy_y*v/radius + c1*dw_y*u_x*v_yy - c1*dw_y*u_yy*v_x + c1*dw_y*v_yy + 4*c1*dw_y*u_x*w_y/radius - 2*c1*dw_y*u_y*w_x/radius + 3*c1*dw_y*w_y/radius - 3*c1*dw_y*u_x*v/(radius*radius) + c1*dw_y*v*v_y/(radius*radius) - c1*dw_y*v/(radius*radius) - c1*dw_yy*u_x*v_y - c1*dw_yy*u_x + c1*dw_yy*u_y*v_x - c1*dw_yy*v_y - c1*dw_yy - c1*dw_yy*u_x*w/radius - c1*dw_yy*w/radius
                de[8] = c1*dphix*phix*v_x/radius + c1*dphix*phix_x*w_y - c1*dphix*phix_x*v/radius + c1*dphix*phix_y*w_x + c1*dphix_x*phix*w_y - c1*dphix_x*phix*v/radius - c1*dphix_x*u_y + c1*dphix_y*phix*w_x - c1*dphix_y*u_x - c1*dphix_y + c1*dphiy*phiy*v_x/radius + c1*dphiy*phiy_x*w_y - c1*dphiy*phiy_x*v/radius + c1*dphiy*phiy_y*w_x + c1*dphiy*v_x*w_y/radius + c1*dphiy*v_y*w_x/radius + c1*dphiy*w_x/radius + c1*dphiy_x*phiy*w_y - c1*dphiy_x*phiy*v/radius - c1*dphiy_x*v_y - c1*dphiy_x + c1*dphiy_x*v*w_y/radius - c1*dphiy_x*w/radius - c1*dphiy_x*(v*v)/(radius*radius) + c1*dphiy_y*phiy*w_x - c1*dphiy_y*v_x + c1*dphiy_y*v*w_x/radius - c1*du_x*phix_y + 2*c1*du_x*v_xy*w_y - 2*c1*du_x*v_y*w_xy - 2*c1*du_x*w_xy - 2*c1*du_x*v*v_xy/radius + 2*c1*du_x*v_x*v_y/radius + 2*c1*du_x*v_x/radius - 2*c1*du_x*w*w_xy/radius + 2*c1*du_x*w_x*w_y/radius - 2*c1*du_x*v*w_x/(radius*radius) + 2*c1*du_x*v_x*w/(radius*radius) - 2*c1*du_xy*v_x*w_y + 2*c1*du_xy*v_y*w_x + 2*c1*du_xy*w_x + 2*c1*du_xy*v*v_x/radius + 2*c1*du_xy*w*w_x/radius - c1*du_y*phix_x + 2*c1*du_y*v_x*w_xy - 2*c1*du_y*v_xy*w_x - 2*c1*du_y*(v_x*v_x)/radius - 2*c1*du_y*(w_x*w_x)/radius - c1*dv*phix*phix_x/radius - c1*dv*phiy*phiy_x/radius + c1*dv*phiy_x*w_y/radius - 2*c1*dv*phiy_x*v/(radius*radius) + c1*dv*phiy_y*w_x/radius - 2*c1*dv*u_x*v_xy/radius + 2*c1*dv*u_xy*v_x/radius - 2*c1*dv*v_xy/radius - 2*c1*dv*u_x*w_x/(radius*radius) + c1*dv*v_x*w_y/(radius*radius) + c1*dv*v_y*w_x/(radius*radius) - c1*dv*w_x/(radius*radius) - c1*dv*v*v_x/(radius*radius*radius) + 0.5*c1*dv_x*(phix*phix)/radius + 0.5*c1*dv_x*(phiy*phiy)/radius + c1*dv_x*phiy*w_y/radius - c1*dv_x*phiy_y - 2*c1*dv_x*u_xy*w_y + 2*c1*dv_x*u_y*w_xy + 2*c1*dv_x*u_x*v_y/radius + 2*c1*dv_x*u_x/radius + 2*c1*dv_x*u_xy*v/radius - 4*c1*dv_x*u_y*v_x/radius + 2*c1*dv_x*u_x*w/(radius*radius) + c1*dv_x*v*w_y/(radius*radius) + c1*dv_x*w/(radius*radius) - 0.5*c1*dv_x*(v*v)/(radius*radius*radius) + 2*c1*dv_xy*u_x*w_y - 2*c1*dv_xy*u_y*w_x + 2*c1*dv_xy*w_y - 2*c1*dv_xy*u_x*v/radius - 2*c1*dv_xy*v/radius + c1*dv_y*phiy*w_x/radius - c1*dv_y*phiy_x - 2*c1*dv_y*u_x*w_xy + 2*c1*dv_y*u_xy*w_x - 2*c1*dv_y*w_xy + 2*c1*dv_y*u_x*v_x/radius + c1*dv_y*v*w_x/(radius*radius) - c1*dw*phiy_x/radius - 2*c1*dw*u_x*w_xy/radius + 2*c1*dw*u_xy*w_x/radius - 2*c1*dw*w_xy/radius + 2*c1*dw*u_x*v_x/(radius*radius) + c1*dw*v_x/(radius*radius) + c1*dw_x*phix*phix_y + c1*dw_x*phiy*phiy_y + c1*dw_x*phiy*v_y/radius + c1*dw_x*phiy/radius + c1*dw_x*phiy_y*v/radius + 2*c1*dw_x*u_xy*v_y + 2*c1*dw_x*u_xy - 2*c1*dw_x*u_y*v_xy + 2*c1*dw_x*u_x*w_y/radius + 2*c1*dw_x*u_xy*w/radius - 4*c1*dw_x*u_y*w_x/radius + 2*c1*dw_x*w_y/radius - 2*c1*dw_x*u_x*v/(radius*radius) + c1*dw_x*v*v_y/(radius*radius) - c1*dw_x*v/(radius*radius) - 2*c1*dw_xy*u_x*v_y - 2*c1*dw_xy*u_x + 2*c1*dw_xy*u_y*v_x - 2*c1*dw_xy*v_y - 2*c1*dw_xy - 2*c1*dw_xy*u_x*w/radius - 2*c1*dw_xy*w/radius + c1*dw_y*phix*phix_x + c1*dw_y*phiy*phiy_x + c1*dw_y*phiy*v_x/radius + c1*dw_y*phiy_x*v/radius + 2*c1*dw_y*u_x*v_xy - 2*c1*dw_y*u_xy*v_x + 2*c1*dw_y*v_xy + 2*c1*dw_y*u_x*w_x/radius + 2*c1*dw_y*w_x/radius + c1*dw_y*v*v_x/(radius*radius)
                de[9] = -dphix*phix*w_y + dphix*phix*v/radius + dphix*u_y - dphiy*phiy*w_y + dphiy*phiy*v/radius + dphiy*v_y + dphiy - dphiy*v*w_y/radius + dphiy*w/radius + dphiy*(v*v)/(radius*radius) + du_y*phix + 0.5*dv*(phix*phix)/radius + 0.5*dv*(phiy*phiy)/radius - dv*phiy*w_y/radius + 2*dv*phiy*v/(radius*radius) + dv*v_y/radius - dv*v*w_y/(radius*radius) + dv*w/(radius*radius) + 1.5*dv*(v*v)/(radius*radius*radius) + dv_y*phiy + dv_y*v/radius + dw*phiy/radius + dw*v/(radius*radius) - 0.5*dw_y*(phix*phix) - 0.5*dw_y*(phiy*phiy) - dw_y*phiy*v/radius + dw_y - 0.5*dw_y*(v*v)/(radius*radius)
                de[10] = -dphix*phix*w_x + dphix*u_x + dphix - dphiy*phiy*w_x + dphiy*v_x - dphiy*v*w_x/radius + du_x*phix - dv*phiy*w_x/radius + dv*v_x/radius - dv*v*w_x/(radius*radius) + dv_x*phiy + dv_x*v/radius - 0.5*dw_x*(phix*phix) - 0.5*dw_x*(phiy*phiy) - dw_x*phiy*v/radius + dw_x - 0.5*dw_x*(v*v)/(radius*radius)
                de[11] = 3*c1*dphix*phix*w_y - 3*c1*dphix*phix*v/radius - 3*c1*dphix*u_y + 3*c1*dphiy*phiy*w_y - 3*c1*dphiy*phiy*v/radius - 3*c1*dphiy*v_y - 3*c1*dphiy + 3*c1*dphiy*v*w_y/radius - 3*c1*dphiy*w/radius - 3*c1*dphiy*(v*v)/(radius*radius) - 3*c1*du_y*phix - 1.5*c1*dv*(phix*phix)/radius - 1.5*c1*dv*(phiy*phiy)/radius + 3*c1*dv*phiy*w_y/radius - 6*c1*dv*phiy*v/(radius*radius) - 3*c1*dv*v_y/radius + 3*c1*dv*v*w_y/(radius*radius) - 3*c1*dv*w/(radius*radius) - 4.5*c1*dv*(v*v)/(radius*radius*radius) - 3*c1*dv_y*phiy - 3*c1*dv_y*v/radius - 3*c1*dw*phiy/radius - 3*c1*dw*v/(radius*radius) + 1.5*c1*dw_y*(phix*phix) + 1.5*c1*dw_y*(phiy*phiy) + 3*c1*dw_y*phiy*v/radius - 3*c1*dw_y + 1.5*c1*dw_y*(v*v)/(radius*radius)
                de[12] = 3*c1*dphix*phix*w_x - 3*c1*dphix*u_x - 3*c1*dphix + 3*c1*dphiy*phiy*w_x - 3*c1*dphiy*v_x + 3*c1*dphiy*v*w_x/radius - 3*c1*du_x*phix + 3*c1*dv*phiy*w_x/radius - 3*c1*dv*v_x/radius + 3*c1*dv*v*w_x/(radius*radius) - 3*c1*dv_x*phiy - 3*c1*dv_x*v/radius + 1.5*c1*dw_x*(phix*phix) + 1.5*c1*dw_x*(phiy*phiy) + 3*c1*dw_x*phiy*v/radius - 3*c1*dw_x + 1.5*c1*dw_x*(v*v)/(radius*radius)
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
                S[0] = -3*c1*dphix*sg[12] - c1*dphix_x*sg[6] - c1*dphix_y*sg[8] - 3*c1*dsg[12]*phix - c1*dsg[6]*phix_x + c1*dsg[6]*v_xx*w_y - c1*dsg[6]*v_y*w_xx - c1*dsg[6]*w_xx - c1*dsg[6]*v*v_xx/radius - c1*dsg[6]*w*w_xx/radius - c1*dsg[7]*v_y*w_yy + c1*dsg[7]*v_yy*w_y - c1*dsg[7]*w_yy - c1*dsg[7]*u_x/radius - c1*dsg[7]*v*v_yy/radius + 2*c1*dsg[7]*(v_y*v_y)/radius + 3*c1*dsg[7]*v_y/radius - c1*dsg[7]*w*w_yy/radius + 2*c1*dsg[7]*(w_y*w_y)/radius - 3*c1*dsg[7]*v*w_y/(radius*radius) + 3*c1*dsg[7]*v_y*w/(radius*radius) + 2*c1*dsg[7]*w/(radius*radius) + c1*dsg[7]*(v*v)/(radius*radius*radius) + c1*dsg[7]*(w*w)/(radius*radius*radius) - c1*dsg[8]*phix_y + 2*c1*dsg[8]*v_xy*w_y - 2*c1*dsg[8]*v_y*w_xy - 2*c1*dsg[8]*w_xy - 2*c1*dsg[8]*v*v_xy/radius + 2*c1*dsg[8]*v_x*v_y/radius + 2*c1*dsg[8]*v_x/radius - 2*c1*dsg[8]*w*w_xy/radius + 2*c1*dsg[8]*w_x*w_y/radius - 2*c1*dsg[8]*v*w_x/(radius*radius) + 2*c1*dsg[8]*v_x*w/(radius*radius) - c1*du_x*sg[7]/radius - c1*dv*sg[6]*v_xx/radius - c1*dv*sg[7]*v_yy/radius - 2*c1*dv*sg[8]*v_xy/radius - 3*c1*dv*sg[7]*w_y/(radius*radius) - 2*c1*dv*sg[8]*w_x/(radius*radius) + 2*c1*dv*sg[7]*v/(radius*radius*radius) + 2*c1*dv_x*sg[8]*v_y/radius + 2*c1*dv_x*sg[8]/radius + 2*c1*dv_x*sg[8]*w/(radius*radius) + c1*dv_xx*sg[6]*w_y - c1*dv_xx*sg[6]*v/radius + 2*c1*dv_xy*sg[8]*w_y - 2*c1*dv_xy*sg[8]*v/radius - c1*dv_y*sg[6]*w_xx - c1*dv_y*sg[7]*w_yy - 2*c1*dv_y*sg[8]*w_xy + 4*c1*dv_y*sg[7]*v_y/radius + 3*c1*dv_y*sg[7]/radius + 2*c1*dv_y*sg[8]*v_x/radius + 3*c1*dv_y*sg[7]*w/(radius*radius) + c1*dv_yy*sg[7]*w_y - c1*dv_yy*sg[7]*v/radius - c1*dw*sg[6]*w_xx/radius - c1*dw*sg[7]*w_yy/radius - 2*c1*dw*sg[8]*w_xy/radius + 3*c1*dw*sg[7]*v_y/(radius*radius) + 2*c1*dw*sg[7]/(radius*radius) + 2*c1*dw*sg[8]*v_x/(radius*radius) + 2*c1*dw*sg[7]*w/(radius*radius*radius) + 2*c1*dw_x*sg[8]*w_y/radius - 2*c1*dw_x*sg[8]*v/(radius*radius) - c1*dw_xx*sg[6]*v_y - c1*dw_xx*sg[6] - c1*dw_xx*sg[6]*w/radius - 2*c1*dw_xy*sg[8]*v_y - 2*c1*dw_xy*sg[8] - 2*c1*dw_xy*sg[8]*w/radius + c1*dw_y*sg[6]*v_xx + c1*dw_y*sg[7]*v_yy + 2*c1*dw_y*sg[8]*v_xy + 4*c1*dw_y*sg[7]*w_y/radius + 2*c1*dw_y*sg[8]*w_x/radius - 3*c1*dw_y*sg[7]*v/(radius*radius) - c1*dw_yy*sg[7]*v_y - c1*dw_yy*sg[7] - c1*dw_yy*sg[7]*w/radius + dphix*sg[10] + dphix_x*sg[3] + dphix_y*sg[5] + dsg[0]*u_x + dsg[0] + dsg[10]*phix + dsg[2]*u_y + dsg[3]*phix_x + dsg[5]*phix_y - 0.5*dsg[5]*u_y/radius + du_x*sg[0] + du_y*sg[2] - 0.5*du_y*sg[5]/radius
                S[1] = -3*c1*dphix*sg[11] - c1*dphix_x*sg[8] - c1*dphix_y*sg[7] - 3*c1*dsg[11]*phix + c1*dsg[6]*v_x*w_xx - c1*dsg[6]*v_xx*w_x - c1*dsg[7]*phix_y + c1*dsg[7]*v_x*w_yy - c1*dsg[7]*v_yy*w_x - c1*dsg[7]*u_y/radius - 2*c1*dsg[7]*v_x*v_y/radius - c1*dsg[7]*v_x/radius - 2*c1*dsg[7]*w_x*w_y/radius + c1*dsg[7]*v*w_x/(radius*radius) - c1*dsg[7]*v_x*w/(radius*radius) - c1*dsg[8]*phix_x + 2*c1*dsg[8]*v_x*w_xy - 2*c1*dsg[8]*v_xy*w_x - 2*c1*dsg[8]*(v_x*v_x)/radius - 2*c1*dsg[8]*(w_x*w_x)/radius - c1*du_y*sg[7]/radius + c1*dv*sg[7]*w_x/(radius*radius) + c1*dv_x*sg[6]*w_xx + c1*dv_x*sg[7]*w_yy + 2*c1*dv_x*sg[8]*w_xy - 2*c1*dv_x*sg[7]*v_y/radius - c1*dv_x*sg[7]/radius - 4*c1*dv_x*sg[8]*v_x/radius - c1*dv_x*sg[7]*w/(radius*radius) - c1*dv_xx*sg[6]*w_x - 2*c1*dv_xy*sg[8]*w_x - 2*c1*dv_y*sg[7]*v_x/radius - c1*dv_yy*sg[7]*w_x - c1*dw*sg[7]*v_x/(radius*radius) - c1*dw_x*sg[6]*v_xx - c1*dw_x*sg[7]*v_yy - 2*c1*dw_x*sg[8]*v_xy - 2*c1*dw_x*sg[7]*w_y/radius - 4*c1*dw_x*sg[8]*w_x/radius + c1*dw_x*sg[7]*v/(radius*radius) + c1*dw_xx*sg[6]*v_x + 2*c1*dw_xy*sg[8]*v_x - 2*c1*dw_y*sg[7]*w_x/radius + c1*dw_yy*sg[7]*v_x + dphix*sg[9] + dphix_x*sg[5] + dphix_y*sg[4] + dsg[1]*u_y + dsg[2]*u_x + dsg[2] + dsg[4]*phix_y - dsg[4]*u_y/radius + dsg[5]*phix_x - 0.5*dsg[5]*u_x/radius - 0.5*dsg[5]/radius + dsg[9]*phix + du_x*sg[2] - 0.5*du_x*sg[5]/radius + du_y*sg[1] - du_y*sg[4]/radius
                S[2] = c1*dphix*phix*sg[8]/radius + c1*dphiy*phiy*sg[8]/radius - 3*c1*dphiy*sg[12] + c1*dphiy*sg[6]*w_x/radius + c1*dphiy*sg[8]*w_y/radius - c1*dphiy_x*sg[6] - c1*dphiy_y*sg[8] - 3*c1*dsg[12]*phiy - 3*c1*dsg[12]*v/radius + c1*dsg[6]*phiy*w_x/radius - c1*dsg[6]*phiy_x - c1*dsg[6]*u_xx*w_y + c1*dsg[6]*u_y*w_xx + c1*dsg[6]*u_xx*v/radius - 2*c1*dsg[6]*v_x/radius + c1*dsg[6]*v*w_x/(radius*radius) + c1*dsg[7]*u_y*w_yy - c1*dsg[7]*u_yy*w_y - 2*c1*dsg[7]*u_y*v_y/radius - c1*dsg[7]*u_y/radius + c1*dsg[7]*u_yy*v/radius - c1*dsg[7]*v_x/radius - c1*dsg[7]*u_y*w/(radius*radius) + 0.5*c1*dsg[8]*(phix*phix)/radius + 0.5*c1*dsg[8]*(phiy*phiy)/radius + c1*dsg[8]*phiy*w_y/radius - c1*dsg[8]*phiy_y - 2*c1*dsg[8]*u_xy*w_y + 2*c1*dsg[8]*u_y*w_xy + 2*c1*dsg[8]*u_x*v_y/radius + 2*c1*dsg[8]*u_x/radius + 2*c1*dsg[8]*u_xy*v/radius - 4*c1*dsg[8]*u_y*v_x/radius + 2*c1*dsg[8]*u_x*w/(radius*radius) + c1*dsg[8]*v*w_y/(radius*radius) + c1*dsg[8]*w/(radius*radius) - 0.5*c1*dsg[8]*(v*v)/(radius*radius*radius) + 2*c1*du_x*sg[8]*v_y/radius + 2*c1*du_x*sg[8]/radius + 2*c1*du_x*sg[8]*w/(radius*radius) - c1*du_xx*sg[6]*w_y + c1*du_xx*sg[6]*v/radius - 2*c1*du_xy*sg[8]*w_y + 2*c1*du_xy*sg[8]*v/radius + c1*du_y*sg[6]*w_xx + c1*du_y*sg[7]*w_yy + 2*c1*du_y*sg[8]*w_xy - 2*c1*du_y*sg[7]*v_y/radius - c1*du_y*sg[7]/radius - 4*c1*du_y*sg[8]*v_x/radius - c1*du_y*sg[7]*w/(radius*radius) - c1*du_yy*sg[7]*w_y + c1*du_yy*sg[7]*v/radius - 3*c1*dv*sg[12]/radius + c1*dv*sg[6]*u_xx/radius + c1*dv*sg[7]*u_yy/radius + 2*c1*dv*sg[8]*u_xy/radius + c1*dv*sg[6]*w_x/(radius*radius) + c1*dv*sg[8]*w_y/(radius*radius) - c1*dv*sg[8]*v/(radius*radius*radius) - 2*c1*dv_x*sg[6]/radius - c1*dv_x*sg[7]/radius - 4*c1*dv_x*sg[8]*u_y/radius - 2*c1*dv_y*sg[7]*u_y/radius + 2*c1*dv_y*sg[8]*u_x/radius - c1*dw*sg[7]*u_y/(radius*radius) + 2*c1*dw*sg[8]*u_x/(radius*radius) + c1*dw*sg[8]/(radius*radius) + c1*dw_x*phiy*sg[6]/radius + c1*dw_x*sg[6]*v/(radius*radius) + c1*dw_xx*sg[6]*u_y + 2*c1*dw_xy*sg[8]*u_y + c1*dw_y*phiy*sg[8]/radius - c1*dw_y*sg[6]*u_xx - c1*dw_y*sg[7]*u_yy - 2*c1*dw_y*sg[8]*u_xy + c1*dw_y*sg[8]*v/(radius*radius) + c1*dw_yy*sg[7]*u_y - dphix*phix*sg[5]/radius - dphiy*phiy*sg[5]/radius + dphiy*sg[10] - dphiy*sg[3]*w_x/radius - dphiy*sg[5]*w_y/radius + dphiy_x*sg[3] + dphiy_y*sg[5] + dsg[0]*v_x + dsg[10]*phiy + dsg[10]*v/radius + dsg[2]*v_y + dsg[2] + dsg[2]*w/radius - dsg[3]*phiy*w_x/radius + dsg[3]*phiy_x + 2*dsg[3]*v_x/radius - dsg[3]*v*w_x/(radius*radius) - 0.5*dsg[5]*(phix*phix)/radius - 0.5*dsg[5]*(phiy*phiy)/radius - dsg[5]*phiy*w_y/radius + dsg[5]*phiy_y + 1.5*dsg[5]*v_y/radius + 1.5*dsg[5]/radius - dsg[5]*v*w_y/(radius*radius) + 0.5*dsg[5]*w/(radius*radius) + 0.5*dsg[5]*(v*v)/(radius*radius*radius) + dv*sg[10]/radius - dv*sg[3]*w_x/(radius*radius) - dv*sg[5]*w_y/(radius*radius) + dv*sg[5]*v/(radius*radius*radius) + dv_x*sg[0] + 2*dv_x*sg[3]/radius + dv_y*sg[2] + 1.5*dv_y*sg[5]/radius + dw*sg[2]/radius + 0.5*dw*sg[5]/(radius*radius) - dw_x*phiy*sg[3]/radius - dw_x*sg[3]*v/(radius*radius) - dw_y*phiy*sg[5]/radius - dw_y*sg[5]*v/(radius*radius)
                S[3] = c1*dphix*phix*sg[7]/radius + c1*dphiy*phiy*sg[7]/radius - 3*c1*dphiy*sg[11] + c1*dphiy*sg[7]*w_y/radius + c1*dphiy*sg[8]*w_x/radius - c1*dphiy_x*sg[8] - c1*dphiy_y*sg[7] - 3*c1*dsg[11]*phiy - 3*c1*dsg[11]*v/radius - c1*dsg[6]*u_x*w_xx + c1*dsg[6]*u_xx*w_x - c1*dsg[6]*w_xx + 0.5*c1*dsg[7]*(phix*phix)/radius + 0.5*c1*dsg[7]*(phiy*phiy)/radius + c1*dsg[7]*phiy*w_y/radius - c1*dsg[7]*phiy_y - c1*dsg[7]*u_x*w_yy + c1*dsg[7]*u_yy*w_x - c1*dsg[7]*w_yy + 4*c1*dsg[7]*u_x*v_y/radius + 3*c1*dsg[7]*u_x/radius - 2*c1*dsg[7]*u_y*v_x/radius + c1*dsg[7]*v_y/radius + 3*c1*dsg[7]*u_x*w/(radius*radius) + c1*dsg[7]*v*w_y/(radius*radius) + c1*dsg[7]*w/(radius*radius) - 0.5*c1*dsg[7]*(v*v)/(radius*radius*radius) + c1*dsg[8]*phiy*w_x/radius - c1*dsg[8]*phiy_x - 2*c1*dsg[8]*u_x*w_xy + 2*c1*dsg[8]*u_xy*w_x - 2*c1*dsg[8]*w_xy + 2*c1*dsg[8]*u_x*v_x/radius + c1*dsg[8]*v*w_x/(radius*radius) - c1*du_x*sg[6]*w_xx - c1*du_x*sg[7]*w_yy - 2*c1*du_x*sg[8]*w_xy + 4*c1*du_x*sg[7]*v_y/radius + 3*c1*du_x*sg[7]/radius + 2*c1*du_x*sg[8]*v_x/radius + 3*c1*du_x*sg[7]*w/(radius*radius) + c1*du_xx*sg[6]*w_x + 2*c1*du_xy*sg[8]*w_x - 2*c1*du_y*sg[7]*v_x/radius + c1*du_yy*sg[7]*w_x - 3*c1*dv*sg[11]/radius + c1*dv*sg[7]*w_y/(radius*radius) + c1*dv*sg[8]*w_x/(radius*radius) - c1*dv*sg[7]*v/(radius*radius*radius) - 2*c1*dv_x*sg[7]*u_y/radius + 2*c1*dv_x*sg[8]*u_x/radius + 4*c1*dv_y*sg[7]*u_x/radius + c1*dv_y*sg[7]/radius + 3*c1*dw*sg[7]*u_x/(radius*radius) + c1*dw*sg[7]/(radius*radius) + c1*dw_x*phiy*sg[8]/radius + c1*dw_x*sg[6]*u_xx + c1*dw_x*sg[7]*u_yy + 2*c1*dw_x*sg[8]*u_xy + c1*dw_x*sg[8]*v/(radius*radius) - c1*dw_xx*sg[6]*u_x - c1*dw_xx*sg[6] - 2*c1*dw_xy*sg[8]*u_x - 2*c1*dw_xy*sg[8] + c1*dw_y*phiy*sg[7]/radius + c1*dw_y*sg[7]*v/(radius*radius) - c1*dw_yy*sg[7]*u_x - c1*dw_yy*sg[7] - dphix*phix*sg[4]/radius - dphiy*phiy*sg[4]/radius + dphiy*sg[9] - dphiy*sg[4]*w_y/radius - dphiy*sg[5]*w_x/radius + dphiy_x*sg[5] + dphiy_y*sg[4] + dsg[1]*v_y + dsg[1] + dsg[1]*w/radius + dsg[2]*v_x - 0.5*dsg[4]*(phix*phix)/radius - 0.5*dsg[4]*(phiy*phiy)/radius - dsg[4]*phiy*w_y/radius + dsg[4]*phiy_y + dsg[4]*v_y/radius + dsg[4]/radius - dsg[4]*v*w_y/(radius*radius) + 0.5*dsg[4]*(v*v)/(radius*radius*radius) - dsg[5]*phiy*w_x/radius + dsg[5]*phiy_x + 1.5*dsg[5]*v_x/radius - dsg[5]*v*w_x/(radius*radius) + dsg[9]*phiy + dsg[9]*v/radius + dv*sg[9]/radius - dv*sg[4]*w_y/(radius*radius) - dv*sg[5]*w_x/(radius*radius) + dv*sg[4]*v/(radius*radius*radius) + dv_x*sg[2] + 1.5*dv_x*sg[5]/radius + dv_y*sg[1] + dv_y*sg[4]/radius + dw*sg[1]/radius - dw_x*phiy*sg[5]/radius - dw_x*sg[5]*v/(radius*radius) - dw_y*phiy*sg[4]/radius - dw_y*sg[4]*v/(radius*radius)
                S[4] = 3*c1*dphix*phix*sg[12] + c1*dphix*phix_x*sg[6] + c1*dphix*phix_y*sg[8] + c1*dphix_x*phix*sg[6] + c1*dphix_y*phix*sg[8] + 3*c1*dphiy*phiy*sg[12] + c1*dphiy*phiy_x*sg[6] + c1*dphiy*phiy_y*sg[8] + 3*c1*dphiy*sg[12]*v/radius + c1*dphiy*sg[6]*v_x/radius + c1*dphiy*sg[8]*v_y/radius + c1*dphiy*sg[8]/radius + c1*dphiy_x*phiy*sg[6] + c1*dphiy_x*sg[6]*v/radius + c1*dphiy_y*phiy*sg[8] + c1*dphiy_y*sg[8]*v/radius + 1.5*c1*dsg[12]*(phix*phix) + 1.5*c1*dsg[12]*(phiy*phiy) + 3*c1*dsg[12]*phiy*v/radius - 3*c1*dsg[12] + 1.5*c1*dsg[12]*(v*v)/(radius*radius) + c1*dsg[6]*phix*phix_x + c1*dsg[6]*phiy*phiy_x + c1*dsg[6]*phiy*v_x/radius + c1*dsg[6]*phiy_x*v/radius + c1*dsg[6]*u_xx*v_y + c1*dsg[6]*u_xx - c1*dsg[6]*u_y*v_xx + c1*dsg[6]*u_xx*w/radius + c1*dsg[6]*v*v_x/(radius*radius) - c1*dsg[7]*u_y*v_yy + c1*dsg[7]*u_yy*v_y + c1*dsg[7]*u_yy - 2*c1*dsg[7]*u_y*w_y/radius + c1*dsg[7]*u_yy*w/radius - c1*dsg[7]*w_x/radius + c1*dsg[7]*u_y*v/(radius*radius) + c1*dsg[8]*phix*phix_y + c1*dsg[8]*phiy*phiy_y + c1*dsg[8]*phiy*v_y/radius + c1*dsg[8]*phiy/radius + c1*dsg[8]*phiy_y*v/radius + 2*c1*dsg[8]*u_xy*v_y + 2*c1*dsg[8]*u_xy - 2*c1*dsg[8]*u_y*v_xy + 2*c1*dsg[8]*u_x*w_y/radius + 2*c1*dsg[8]*u_xy*w/radius - 4*c1*dsg[8]*u_y*w_x/radius + 2*c1*dsg[8]*w_y/radius - 2*c1*dsg[8]*u_x*v/(radius*radius) + c1*dsg[8]*v*v_y/(radius*radius) - c1*dsg[8]*v/(radius*radius) + 2*c1*du_x*sg[8]*w_y/radius - 2*c1*du_x*sg[8]*v/(radius*radius) + c1*du_xx*sg[6]*v_y + c1*du_xx*sg[6] + c1*du_xx*sg[6]*w/radius + 2*c1*du_xy*sg[8]*v_y + 2*c1*du_xy*sg[8] + 2*c1*du_xy*sg[8]*w/radius - c1*du_y*sg[6]*v_xx - c1*du_y*sg[7]*v_yy - 2*c1*du_y*sg[8]*v_xy - 2*c1*du_y*sg[7]*w_y/radius - 4*c1*du_y*sg[8]*w_x/radius + c1*du_y*sg[7]*v/(radius*radius) + c1*du_yy*sg[7]*v_y + c1*du_yy*sg[7] + c1*du_yy*sg[7]*w/radius + 3*c1*dv*phiy*sg[12]/radius + c1*dv*phiy_x*sg[6]/radius + c1*dv*phiy_y*sg[8]/radius + 3*c1*dv*sg[12]*v/(radius*radius) + c1*dv*sg[6]*v_x/(radius*radius) + c1*dv*sg[7]*u_y/(radius*radius) - 2*c1*dv*sg[8]*u_x/(radius*radius) + c1*dv*sg[8]*v_y/(radius*radius) - c1*dv*sg[8]/(radius*radius) + c1*dv_x*phiy*sg[6]/radius + c1*dv_x*sg[6]*v/(radius*radius) - c1*dv_xx*sg[6]*u_y - 2*c1*dv_xy*sg[8]*u_y + c1*dv_y*phiy*sg[8]/radius + c1*dv_y*sg[6]*u_xx + c1*dv_y*sg[7]*u_yy + 2*c1*dv_y*sg[8]*u_xy + c1*dv_y*sg[8]*v/(radius*radius) - c1*dv_yy*sg[7]*u_y + c1*dw*sg[6]*u_xx/radius + c1*dw*sg[7]*u_yy/radius + 2*c1*dw*sg[8]*u_xy/radius - c1*dw_x*sg[7]/radius - 4*c1*dw_x*sg[8]*u_y/radius - 2*c1*dw_y*sg[7]*u_y/radius + 2*c1*dw_y*sg[8]*u_x/radius + 2*c1*dw_y*sg[8]/radius - dphix*phix*sg[10] - dphix*phix_x*sg[3] - dphix*phix_y*sg[5] - dphix_x*phix*sg[3] - dphix_y*phix*sg[5] - dphiy*phiy*sg[10] - dphiy*phiy_x*sg[3] - dphiy*phiy_y*sg[5] - dphiy*sg[10]*v/radius - dphiy*sg[3]*v_x/radius - dphiy*sg[5]*v_y/radius - dphiy*sg[5]/radius - dphiy_x*phiy*sg[3] - dphiy_x*sg[3]*v/radius - dphiy_y*phiy*sg[5] - dphiy_y*sg[5]*v/radius + dsg[0]*w_x - 0.5*dsg[10]*(phix*phix) - 0.5*dsg[10]*(phiy*phiy) - dsg[10]*phiy*v/radius + dsg[10] - 0.5*dsg[10]*(v*v)/(radius*radius) + dsg[2]*w_y - dsg[2]*v/radius - dsg[3]*phix*phix_x - dsg[3]*phiy*phiy_x - dsg[3]*phiy*v_x/radius - dsg[3]*phiy_x*v/radius - dsg[3]*v*v_x/(radius*radius) - dsg[5]*phix*phix_y - dsg[5]*phiy*phiy_y - dsg[5]*phiy*v_y/radius - dsg[5]*phiy/radius - dsg[5]*phiy_y*v/radius - 0.5*dsg[5]*w_y/radius - dsg[5]*v*v_y/(radius*radius) - 0.5*dsg[5]*v/(radius*radius) - dv*phiy*sg[10]/radius - dv*phiy_x*sg[3]/radius - dv*phiy_y*sg[5]/radius - dv*sg[2]/radius - dv*sg[10]*v/(radius*radius) - dv*sg[3]*v_x/(radius*radius) - dv*sg[5]*v_y/(radius*radius) - 0.5*dv*sg[5]/(radius*radius) - dv_x*phiy*sg[3]/radius - dv_x*sg[3]*v/(radius*radius) - dv_y*phiy*sg[5]/radius - dv_y*sg[5]*v/(radius*radius) + dw_x*sg[0] + dw_y*sg[2] - 0.5*dw_y*sg[5]/radius
                S[5] = 3*c1*dphix*phix*sg[11] + c1*dphix*phix_x*sg[8] + c1*dphix*phix_y*sg[7] + c1*dphix_x*phix*sg[8] + c1*dphix_y*phix*sg[7] + 3*c1*dphiy*phiy*sg[11] + c1*dphiy*phiy_x*sg[8] + c1*dphiy*phiy_y*sg[7] + 3*c1*dphiy*sg[11]*v/radius + c1*dphiy*sg[7]*v_y/radius + c1*dphiy*sg[7]/radius + c1*dphiy*sg[8]*v_x/radius + c1*dphiy_x*phiy*sg[8] + c1*dphiy_x*sg[8]*v/radius + c1*dphiy_y*phiy*sg[7] + c1*dphiy_y*sg[7]*v/radius + 1.5*c1*dsg[11]*(phix*phix) + 1.5*c1*dsg[11]*(phiy*phiy) + 3*c1*dsg[11]*phiy*v/radius - 3*c1*dsg[11] + 1.5*c1*dsg[11]*(v*v)/(radius*radius) + c1*dsg[6]*u_x*v_xx - c1*dsg[6]*u_xx*v_x + c1*dsg[6]*v_xx + c1*dsg[7]*phix*phix_y + c1*dsg[7]*phiy*phiy_y + c1*dsg[7]*phiy*v_y/radius + c1*dsg[7]*phiy/radius + c1*dsg[7]*phiy_y*v/radius + c1*dsg[7]*u_x*v_yy - c1*dsg[7]*u_yy*v_x + c1*dsg[7]*v_yy + 4*c1*dsg[7]*u_x*w_y/radius - 2*c1*dsg[7]*u_y*w_x/radius + 3*c1*dsg[7]*w_y/radius - 3*c1*dsg[7]*u_x*v/(radius*radius) + c1*dsg[7]*v*v_y/(radius*radius) - c1*dsg[7]*v/(radius*radius) + c1*dsg[8]*phix*phix_x + c1*dsg[8]*phiy*phiy_x + c1*dsg[8]*phiy*v_x/radius + c1*dsg[8]*phiy_x*v/radius + 2*c1*dsg[8]*u_x*v_xy - 2*c1*dsg[8]*u_xy*v_x + 2*c1*dsg[8]*v_xy + 2*c1*dsg[8]*u_x*w_x/radius + 2*c1*dsg[8]*w_x/radius + c1*dsg[8]*v*v_x/(radius*radius) + c1*du_x*sg[6]*v_xx + c1*du_x*sg[7]*v_yy + 2*c1*du_x*sg[8]*v_xy + 4*c1*du_x*sg[7]*w_y/radius + 2*c1*du_x*sg[8]*w_x/radius - 3*c1*du_x*sg[7]*v/(radius*radius) - c1*du_xx*sg[6]*v_x - 2*c1*du_xy*sg[8]*v_x - 2*c1*du_y*sg[7]*w_x/radius - c1*du_yy*sg[7]*v_x + 3*c1*dv*phiy*sg[11]/radius + c1*dv*phiy_x*sg[8]/radius + c1*dv*phiy_y*sg[7]/radius + 3*c1*dv*sg[11]*v/(radius*radius) - 3*c1*dv*sg[7]*u_x/(radius*radius) + c1*dv*sg[7]*v_y/(radius*radius) - c1*dv*sg[7]/(radius*radius) + c1*dv*sg[8]*v_x/(radius*radius) + c1*dv_x*phiy*sg[8]/radius - c1*dv_x*sg[6]*u_xx - c1*dv_x*sg[7]*u_yy - 2*c1*dv_x*sg[8]*u_xy + c1*dv_x*sg[8]*v/(radius*radius) + c1*dv_xx*sg[6]*u_x + c1*dv_xx*sg[6] + 2*c1*dv_xy*sg[8]*u_x + 2*c1*dv_xy*sg[8] + c1*dv_y*phiy*sg[7]/radius + c1*dv_y*sg[7]*v/(radius*radius) + c1*dv_yy*sg[7]*u_x + c1*dv_yy*sg[7] - 2*c1*dw_x*sg[7]*u_y/radius + 2*c1*dw_x*sg[8]*u_x/radius + 2*c1*dw_x*sg[8]/radius + 4*c1*dw_y*sg[7]*u_x/radius + 3*c1*dw_y*sg[7]/radius - dphix*phix*sg[9] - dphix*phix_x*sg[5] - dphix*phix_y*sg[4] - dphix_x*phix*sg[5] - dphix_y*phix*sg[4] - dphiy*phiy*sg[9] - dphiy*phiy_x*sg[5] - dphiy*phiy_y*sg[4] - dphiy*sg[4]*v_y/radius - dphiy*sg[4]/radius - dphiy*sg[5]*v_x/radius - dphiy*sg[9]*v/radius - dphiy_x*phiy*sg[5] - dphiy_x*sg[5]*v/radius - dphiy_y*phiy*sg[4] - dphiy_y*sg[4]*v/radius + dsg[1]*w_y - dsg[1]*v/radius + dsg[2]*w_x - dsg[4]*phix*phix_y - dsg[4]*phiy*phiy_y - dsg[4]*phiy*v_y/radius - dsg[4]*phiy/radius - dsg[4]*phiy_y*v/radius - dsg[4]*w_y/radius - dsg[4]*v*v_y/(radius*radius) - dsg[5]*phix*phix_x - dsg[5]*phiy*phiy_x - dsg[5]*phiy*v_x/radius - dsg[5]*phiy_x*v/radius - 0.5*dsg[5]*w_x/radius - dsg[5]*v*v_x/(radius*radius) - 0.5*dsg[9]*(phix*phix) - 0.5*dsg[9]*(phiy*phiy) - dsg[9]*phiy*v/radius + dsg[9] - 0.5*dsg[9]*(v*v)/(radius*radius) - dv*phiy*sg[9]/radius - dv*phiy_x*sg[5]/radius - dv*phiy_y*sg[4]/radius - dv*sg[1]/radius - dv*sg[4]*v_y/(radius*radius) - dv*sg[5]*v_x/(radius*radius) - dv*sg[9]*v/(radius*radius) - dv_x*phiy*sg[5]/radius - dv_x*sg[5]*v/(radius*radius) - dv_y*phiy*sg[4]/radius - dv_y*sg[4]*v/(radius*radius) + dw_x*sg[2] - 0.5*dw_x*sg[5]/radius + dw_y*sg[1] - dw_y*sg[4]/radius
                S[6] = -3*c1*dphix*phix*sg[11]/radius - c1*dphix*phix_x*sg[8]/radius - c1*dphix*phix_y*sg[7]/radius - c1*dphix_x*phix*sg[8]/radius - c1*dphix_y*phix*sg[7]/radius - 3*c1*dphiy*phiy*sg[11]/radius - c1*dphiy*phiy_x*sg[8]/radius - c1*dphiy*phiy_y*sg[7]/radius + 3*c1*dphiy*sg[11]*w_y/radius + 3*c1*dphiy*sg[12]*w_x/radius - 6*c1*dphiy*sg[11]*v/(radius*radius) + c1*dphiy*sg[7]*w/(radius*radius*radius) - c1*dphiy_x*phiy*sg[8]/radius + c1*dphiy_x*sg[6]*w_x/radius + c1*dphiy_x*sg[8]*w_y/radius - 2*c1*dphiy_x*sg[8]*v/(radius*radius) - c1*dphiy_y*phiy*sg[7]/radius + c1*dphiy_y*sg[7]*w_y/radius + c1*dphiy_y*sg[8]*w_x/radius - 2*c1*dphiy_y*sg[7]*v/(radius*radius) - 1.5*c1*dsg[11]*(phix*phix)/radius - 1.5*c1*dsg[11]*(phiy*phiy)/radius + 3*c1*dsg[11]*phiy*w_y/radius - 6*c1*dsg[11]*phiy*v/(radius*radius) - 3*c1*dsg[11]*v_y/radius + 3*c1*dsg[11]*v*w_y/(radius*radius) - 3*c1*dsg[11]*w/(radius*radius) - 4.5*c1*dsg[11]*(v*v)/(radius*radius*radius) + 3*c1*dsg[12]*phiy*w_x/radius - 3*c1*dsg[12]*v_x/radius + 3*c1*dsg[12]*v*w_x/(radius*radius) + c1*dsg[6]*phiy_x*w_x/radius - c1*dsg[6]*u_x*v_xx/radius + c1*dsg[6]*u_xx*v_x/radius - c1*dsg[6]*v_xx/radius + c1*dsg[6]*v_x*w_x/(radius*radius) - c1*dsg[7]*phix*phix_y/radius - c1*dsg[7]*phiy*phiy_y/radius + c1*dsg[7]*phiy*w/(radius*radius*radius) + c1*dsg[7]*phiy_y*w_y/radius - 2*c1*dsg[7]*phiy_y*v/(radius*radius) - c1*dsg[7]*u_x*v_yy/radius + c1*dsg[7]*u_yy*v_x/radius - c1*dsg[7]*v_yy/radius - 3*c1*dsg[7]*u_x*w_y/(radius*radius) + c1*dsg[7]*u_y*w_x/(radius*radius) + c1*dsg[7]*v_y*w_y/(radius*radius) - c1*dsg[7]*w_y/(radius*radius) + 2*c1*dsg[7]*u_x*v/(radius*radius*radius) - c1*dsg[7]*v*v_y/(radius*radius*radius) + c1*dsg[7]*v*w/(radius*radius*radius*radius) - c1*dsg[8]*phix*phix_x/radius - c1*dsg[8]*phiy*phiy_x/radius + c1*dsg[8]*phiy_x*w_y/radius - 2*c1*dsg[8]*phiy_x*v/(radius*radius) + c1*dsg[8]*phiy_y*w_x/radius - 2*c1*dsg[8]*u_x*v_xy/radius + 2*c1*dsg[8]*u_xy*v_x/radius - 2*c1*dsg[8]*v_xy/radius - 2*c1*dsg[8]*u_x*w_x/(radius*radius) + c1*dsg[8]*v_x*w_y/(radius*radius) + c1*dsg[8]*v_y*w_x/(radius*radius) - c1*dsg[8]*w_x/(radius*radius) - c1*dsg[8]*v*v_x/(radius*radius*radius) - c1*du_x*sg[6]*v_xx/radius - c1*du_x*sg[7]*v_yy/radius - 2*c1*du_x*sg[8]*v_xy/radius - 3*c1*du_x*sg[7]*w_y/(radius*radius) - 2*c1*du_x*sg[8]*w_x/(radius*radius) + 2*c1*du_x*sg[7]*v/(radius*radius*radius) + c1*du_xx*sg[6]*v_x/radius + 2*c1*du_xy*sg[8]*v_x/radius + c1*du_y*sg[7]*w_x/(radius*radius) + c1*du_yy*sg[7]*v_x/radius - 6*c1*dv*phiy*sg[11]/(radius*radius) - 2*c1*dv*phiy_x*sg[8]/(radius*radius) - 2*c1*dv*phiy_y*sg[7]/(radius*radius) + 3*c1*dv*sg[11]*w_y/(radius*radius) + 3*c1*dv*sg[12]*w_x/(radius*radius) - 9*c1*dv*sg[11]*v/(radius*radius*radius) + 2*c1*dv*sg[7]*u_x/(radius*radius*radius) - c1*dv*sg[7]*v_y/(radius*radius*radius) - c1*dv*sg[8]*v_x/(radius*radius*radius) + c1*dv*sg[7]*w/(radius*radius*radius*radius) - 3*c1*dv_x*sg[12]/radius + c1*dv_x*sg[6]*u_xx/radius + c1*dv_x*sg[7]*u_yy/radius + 2*c1*dv_x*sg[8]*u_xy/radius + c1*dv_x*sg[6]*w_x/(radius*radius) + c1*dv_x*sg[8]*w_y/(radius*radius) - c1*dv_x*sg[8]*v/(radius*radius*radius) - c1*dv_xx*sg[6]*u_x/radius - c1*dv_xx*sg[6]/radius - 2*c1*dv_xy*sg[8]*u_x/radius - 2*c1*dv_xy*sg[8]/radius - 3*c1*dv_y*sg[11]/radius + c1*dv_y*sg[7]*w_y/(radius*radius) + c1*dv_y*sg[8]*w_x/(radius*radius) - c1*dv_y*sg[7]*v/(radius*radius*radius) - c1*dv_yy*sg[7]*u_x/radius - c1*dv_yy*sg[7]/radius + c1*dw*phiy*sg[7]/(radius*radius*radius) - 3*c1*dw*sg[11]/(radius*radius) + c1*dw*sg[7]*v/(radius*radius*radius*radius) + 3*c1*dw_x*phiy*sg[12]/radius + c1*dw_x*phiy_x*sg[6]/radius + c1*dw_x*phiy_y*sg[8]/radius + 3*c1*dw_x*sg[12]*v/(radius*radius) + c1*dw_x*sg[6]*v_x/(radius*radius) + c1*dw_x*sg[7]*u_y/(radius*radius) - 2*c1*dw_x*sg[8]*u_x/(radius*radius) + c1*dw_x*sg[8]*v_y/(radius*radius) - c1*dw_x*sg[8]/(radius*radius) + 3*c1*dw_y*phiy*sg[11]/radius + c1*dw_y*phiy_x*sg[8]/radius + c1*dw_y*phiy_y*sg[7]/radius + 3*c1*dw_y*sg[11]*v/(radius*radius) - 3*c1*dw_y*sg[7]*u_x/(radius*radius) + c1*dw_y*sg[7]*v_y/(radius*radius) - c1*dw_y*sg[7]/(radius*radius) + c1*dw_y*sg[8]*v_x/(radius*radius) + dphix*phix*sg[9]/radius + dphix*phix_x*sg[5]/radius + dphix*phix_y*sg[4]/radius + dphix_x*phix*sg[5]/radius + dphix_y*phix*sg[4]/radius + dphiy*phiy*sg[9]/radius + dphiy*phiy_x*sg[5]/radius + dphiy*phiy_y*sg[4]/radius - dphiy*sg[10]*w_x/radius - dphiy*sg[9]*w_y/radius + 2*dphiy*sg[9]*v/(radius*radius) - dphiy*sg[4]*w/(radius*radius*radius) + dphiy_x*phiy*sg[5]/radius - dphiy_x*sg[3]*w_x/radius - dphiy_x*sg[5]*w_y/radius + 2*dphiy_x*sg[5]*v/(radius*radius) + dphiy_y*phiy*sg[4]/radius - dphiy_y*sg[4]*w_y/radius - dphiy_y*sg[5]*w_x/radius + 2*dphiy_y*sg[4]*v/(radius*radius) - dsg[10]*phiy*w_x/radius + dsg[10]*v_x/radius - dsg[10]*v*w_x/(radius*radius) - dsg[1]*w_y/radius + dsg[1]*v/(radius*radius) - dsg[2]*w_x/radius - dsg[3]*phiy_x*w_x/radius - dsg[3]*v_x*w_x/(radius*radius) + dsg[4]*phix*phix_y/radius + dsg[4]*phiy*phiy_y/radius - dsg[4]*phiy*w/(radius*radius*radius) - dsg[4]*phiy_y*w_y/radius + 2*dsg[4]*phiy_y*v/(radius*radius) - dsg[4]*v_y*w_y/(radius*radius) + dsg[4]*v*v_y/(radius*radius*radius) - dsg[4]*v*w/(radius*radius*radius*radius) + dsg[5]*phix*phix_x/radius + dsg[5]*phiy*phiy_x/radius - dsg[5]*phiy_x*w_y/radius + 2*dsg[5]*phiy_x*v/(radius*radius) - dsg[5]*phiy_y*w_x/radius - dsg[5]*v_x*w_y/(radius*radius) - dsg[5]*v_y*w_x/(radius*radius) - 0.5*dsg[5]*w_x/(radius*radius) + dsg[5]*v*v_x/(radius*radius*radius) + 0.5*dsg[9]*(phix*phix)/radius + 0.5*dsg[9]*(phiy*phiy)/radius - dsg[9]*phiy*w_y/radius + 2*dsg[9]*phiy*v/(radius*radius) + dsg[9]*v_y/radius - dsg[9]*v*w_y/(radius*radius) + dsg[9]*w/(radius*radius) + 1.5*dsg[9]*(v*v)/(radius*radius*radius) + 2*dv*phiy*sg[9]/(radius*radius) + 2*dv*phiy_x*sg[5]/(radius*radius) + 2*dv*phiy_y*sg[4]/(radius*radius) - dv*sg[10]*w_x/(radius*radius) + dv*sg[1]/(radius*radius) - dv*sg[9]*w_y/(radius*radius) + dv*sg[4]*v_y/(radius*radius*radius) + dv*sg[5]*v_x/(radius*radius*radius) + 3*dv*sg[9]*v/(radius*radius*radius) - dv*sg[4]*w/(radius*radius*radius*radius) + dv_x*sg[10]/radius - dv_x*sg[3]*w_x/(radius*radius) - dv_x*sg[5]*w_y/(radius*radius) + dv_x*sg[5]*v/(radius*radius*radius) + dv_y*sg[9]/radius - dv_y*sg[4]*w_y/(radius*radius) - dv_y*sg[5]*w_x/(radius*radius) + dv_y*sg[4]*v/(radius*radius*radius) - dw*phiy*sg[4]/(radius*radius*radius) + dw*sg[9]/(radius*radius) - dw*sg[4]*v/(radius*radius*radius*radius) - dw_x*phiy*sg[10]/radius - dw_x*phiy_x*sg[3]/radius - dw_x*phiy_y*sg[5]/radius - dw_x*sg[2]/radius - dw_x*sg[10]*v/(radius*radius) - dw_x*sg[3]*v_x/(radius*radius) - dw_x*sg[5]*v_y/(radius*radius) - 0.5*dw_x*sg[5]/(radius*radius) - dw_y*phiy*sg[9]/radius - dw_y*phiy_x*sg[5]/radius - dw_y*phiy_y*sg[4]/radius - dw_y*sg[1]/radius - dw_y*sg[4]*v_y/(radius*radius) - dw_y*sg[5]*v_x/(radius*radius) - dw_y*sg[9]*v/(radius*radius)
                S[7] = c1*dphix*phix*sg[7]/(radius*radius) + c1*dphiy*phiy*sg[7]/(radius*radius) - 3*c1*dphiy*sg[11]/radius + c1*dphiy*sg[7]*v/(radius*radius*radius) - c1*dphiy_x*sg[8]/radius - c1*dphiy_y*sg[7]/radius - 3*c1*dsg[11]*phiy/radius - 3*c1*dsg[11]*v/(radius*radius) - c1*dsg[6]*u_x*w_xx/radius + c1*dsg[6]*u_xx*w_x/radius - c1*dsg[6]*w_xx/radius + 0.5*c1*dsg[7]*(phix*phix)/(radius*radius) + 0.5*c1*dsg[7]*(phiy*phiy)/(radius*radius) + c1*dsg[7]*phiy*v/(radius*radius*radius) - c1*dsg[7]*phiy_y/radius - c1*dsg[7]*u_x*w_yy/radius + c1*dsg[7]*u_yy*w_x/radius - c1*dsg[7]*w_yy/radius + 3*c1*dsg[7]*u_x*v_y/(radius*radius) + 2*c1*dsg[7]*u_x/(radius*radius) - c1*dsg[7]*u_y*v_x/(radius*radius) + c1*dsg[7]*v_y/(radius*radius) + 2*c1*dsg[7]*u_x*w/(radius*radius*radius) + c1*dsg[7]*w/(radius*radius*radius) + 0.5*c1*dsg[7]*(v*v)/(radius*radius*radius*radius) - c1*dsg[8]*phiy_x/radius - 2*c1*dsg[8]*u_x*w_xy/radius + 2*c1*dsg[8]*u_xy*w_x/radius - 2*c1*dsg[8]*w_xy/radius + 2*c1*dsg[8]*u_x*v_x/(radius*radius) + c1*dsg[8]*v_x/(radius*radius) - c1*du_x*sg[6]*w_xx/radius - c1*du_x*sg[7]*w_yy/radius - 2*c1*du_x*sg[8]*w_xy/radius + 3*c1*du_x*sg[7]*v_y/(radius*radius) + 2*c1*du_x*sg[7]/(radius*radius) + 2*c1*du_x*sg[8]*v_x/(radius*radius) + 2*c1*du_x*sg[7]*w/(radius*radius*radius) + c1*du_xx*sg[6]*w_x/radius + 2*c1*du_xy*sg[8]*w_x/radius - c1*du_y*sg[7]*v_x/(radius*radius) + c1*du_yy*sg[7]*w_x/radius + c1*dv*phiy*sg[7]/(radius*radius*radius) - 3*c1*dv*sg[11]/(radius*radius) + c1*dv*sg[7]*v/(radius*radius*radius*radius) - c1*dv_x*sg[7]*u_y/(radius*radius) + 2*c1*dv_x*sg[8]*u_x/(radius*radius) + c1*dv_x*sg[8]/(radius*radius) + 3*c1*dv_y*sg[7]*u_x/(radius*radius) + c1*dv_y*sg[7]/(radius*radius) + 2*c1*dw*sg[7]*u_x/(radius*radius*radius) + c1*dw*sg[7]/(radius*radius*radius) + c1*dw_x*sg[6]*u_xx/radius + c1*dw_x*sg[7]*u_yy/radius + 2*c1*dw_x*sg[8]*u_xy/radius - c1*dw_xx*sg[6]*u_x/radius - c1*dw_xx*sg[6]/radius - 2*c1*dw_xy*sg[8]*u_x/radius - 2*c1*dw_xy*sg[8]/radius - c1*dw_yy*sg[7]*u_x/radius - c1*dw_yy*sg[7]/radius - dphix*phix*sg[4]/(radius*radius) - dphiy*phiy*sg[4]/(radius*radius) + dphiy*sg[9]/radius - dphiy*sg[4]*v/(radius*radius*radius) + dphiy_x*sg[5]/radius + dphiy_y*sg[4]/radius + dsg[1]*v_y/radius + dsg[1]/radius + dsg[1]*w/(radius*radius) + dsg[2]*v_x/radius - 0.5*dsg[4]*(phix*phix)/(radius*radius) - 0.5*dsg[4]*(phiy*phiy)/(radius*radius) - dsg[4]*phiy*v/(radius*radius*radius) + dsg[4]*phiy_y/radius - dsg[4]*w/(radius*radius*radius) - 0.5*dsg[4]*(v*v)/(radius*radius*radius*radius) + dsg[5]*phiy_x/radius + 0.5*dsg[5]*v_x/(radius*radius) + dsg[9]*phiy/radius + dsg[9]*v/(radius*radius) - dv*phiy*sg[4]/(radius*radius*radius) + dv*sg[9]/(radius*radius) - dv*sg[4]*v/(radius*radius*radius*radius) + dv_x*sg[2]/radius + 0.5*dv_x*sg[5]/(radius*radius) + dv_y*sg[1]/radius + dw*sg[1]/(radius*radius) - dw*sg[4]/(radius*radius*radius)
                S[8] = 3*c1*dphix*sg[11]*w_y + 3*c1*dphix*sg[12]*w_x - 3*c1*dphix*sg[11]*v/radius + c1*dphix*sg[7]*v_y/radius + c1*dphix*sg[7]/radius + c1*dphix*sg[8]*v_x/radius + c1*dphix*sg[7]*w/(radius*radius) + c1*dphix_x*sg[6]*w_x + c1*dphix_x*sg[8]*w_y - c1*dphix_x*sg[8]*v/radius + c1*dphix_y*sg[7]*w_y + c1*dphix_y*sg[8]*w_x - c1*dphix_y*sg[7]*v/radius + 3*c1*dsg[11]*phix*w_y - 3*c1*dsg[11]*phix*v/radius - 3*c1*dsg[11]*u_y + 3*c1*dsg[12]*phix*w_x - 3*c1*dsg[12]*u_x - 3*c1*dsg[12] + c1*dsg[6]*phix_x*w_x + c1*dsg[7]*phix*v_y/radius + c1*dsg[7]*phix/radius + c1*dsg[7]*phix*w/(radius*radius) + c1*dsg[7]*phix_y*w_y - c1*dsg[7]*phix_y*v/radius + c1*dsg[8]*phix*v_x/radius + c1*dsg[8]*phix_x*w_y - c1*dsg[8]*phix_x*v/radius + c1*dsg[8]*phix_y*w_x - 3*c1*du_x*sg[12] - 3*c1*du_y*sg[11] - 3*c1*dv*phix*sg[11]/radius - c1*dv*phix_x*sg[8]/radius - c1*dv*phix_y*sg[7]/radius + c1*dv_x*phix*sg[8]/radius + c1*dv_y*phix*sg[7]/radius + c1*dw*phix*sg[7]/(radius*radius) + 3*c1*dw_x*phix*sg[12] + c1*dw_x*phix_x*sg[6] + c1*dw_x*phix_y*sg[8] + 3*c1*dw_y*phix*sg[11] + c1*dw_y*phix_x*sg[8] + c1*dw_y*phix_y*sg[7] - dphix*sg[10]*w_x - dphix*sg[9]*w_y - dphix*sg[4]*v_y/radius - dphix*sg[4]/radius - dphix*sg[5]*v_x/radius + dphix*sg[9]*v/radius - dphix*sg[4]*w/(radius*radius) - dphix_x*sg[3]*w_x - dphix_x*sg[5]*w_y + dphix_x*sg[5]*v/radius - dphix_y*sg[4]*w_y - dphix_y*sg[5]*w_x + dphix_y*sg[4]*v/radius - dsg[10]*phix*w_x + dsg[10]*u_x + dsg[10] - dsg[3]*phix_x*w_x - dsg[4]*phix*v_y/radius - dsg[4]*phix/radius - dsg[4]*phix*w/(radius*radius) - dsg[4]*phix_y*w_y + dsg[4]*phix_y*v/radius - dsg[5]*phix*v_x/radius - dsg[5]*phix_x*w_y + dsg[5]*phix_x*v/radius - dsg[5]*phix_y*w_x - dsg[9]*phix*w_y + dsg[9]*phix*v/radius + dsg[9]*u_y + du_x*sg[10] + du_y*sg[9] + dv*phix*sg[9]/radius + dv*phix_x*sg[5]/radius + dv*phix_y*sg[4]/radius - dv_x*phix*sg[5]/radius - dv_y*phix*sg[4]/radius - dw*phix*sg[4]/(radius*radius) - dw_x*phix*sg[10] - dw_x*phix_x*sg[3] - dw_x*phix_y*sg[5] - dw_y*phix*sg[9] - dw_y*phix_x*sg[5] - dw_y*phix_y*sg[4]
                S[9] = 3*c1*dphiy*sg[11]*w_y + 3*c1*dphiy*sg[12]*w_x - 3*c1*dphiy*sg[11]*v/radius + c1*dphiy*sg[7]*v_y/radius + c1*dphiy*sg[7]/radius + c1*dphiy*sg[8]*v_x/radius + c1*dphiy*sg[7]*w/(radius*radius) + c1*dphiy_x*sg[6]*w_x + c1*dphiy_x*sg[8]*w_y - c1*dphiy_x*sg[8]*v/radius + c1*dphiy_y*sg[7]*w_y + c1*dphiy_y*sg[8]*w_x - c1*dphiy_y*sg[7]*v/radius + 3*c1*dsg[11]*phiy*w_y - 3*c1*dsg[11]*phiy*v/radius - 3*c1*dsg[11]*v_y - 3*c1*dsg[11] + 3*c1*dsg[11]*v*w_y/radius - 3*c1*dsg[11]*w/radius - 3*c1*dsg[11]*(v*v)/(radius*radius) + 3*c1*dsg[12]*phiy*w_x - 3*c1*dsg[12]*v_x + 3*c1*dsg[12]*v*w_x/radius + c1*dsg[6]*phiy_x*w_x + c1*dsg[6]*v_x*w_x/radius + c1*dsg[7]*phiy*v_y/radius + c1*dsg[7]*phiy/radius + c1*dsg[7]*phiy*w/(radius*radius) + c1*dsg[7]*phiy_y*w_y - c1*dsg[7]*phiy_y*v/radius + c1*dsg[7]*v_y*w_y/radius + c1*dsg[7]*w_y/radius + c1*dsg[7]*v*w/(radius*radius*radius) + c1*dsg[8]*phiy*v_x/radius + c1*dsg[8]*phiy_x*w_y - c1*dsg[8]*phiy_x*v/radius + c1*dsg[8]*phiy_y*w_x + c1*dsg[8]*v_x*w_y/radius + c1*dsg[8]*v_y*w_x/radius + c1*dsg[8]*w_x/radius - 3*c1*dv*phiy*sg[11]/radius - c1*dv*phiy_x*sg[8]/radius - c1*dv*phiy_y*sg[7]/radius + 3*c1*dv*sg[11]*w_y/radius + 3*c1*dv*sg[12]*w_x/radius - 6*c1*dv*sg[11]*v/(radius*radius) + c1*dv*sg[7]*w/(radius*radius*radius) + c1*dv_x*phiy*sg[8]/radius - 3*c1*dv_x*sg[12] + c1*dv_x*sg[6]*w_x/radius + c1*dv_x*sg[8]*w_y/radius + c1*dv_y*phiy*sg[7]/radius - 3*c1*dv_y*sg[11] + c1*dv_y*sg[7]*w_y/radius + c1*dv_y*sg[8]*w_x/radius + c1*dw*phiy*sg[7]/(radius*radius) - 3*c1*dw*sg[11]/radius + c1*dw*sg[7]*v/(radius*radius*radius) + 3*c1*dw_x*phiy*sg[12] + c1*dw_x*phiy_x*sg[6] + c1*dw_x*phiy_y*sg[8] + 3*c1*dw_x*sg[12]*v/radius + c1*dw_x*sg[6]*v_x/radius + c1*dw_x*sg[8]*v_y/radius + c1*dw_x*sg[8]/radius + 3*c1*dw_y*phiy*sg[11] + c1*dw_y*phiy_x*sg[8] + c1*dw_y*phiy_y*sg[7] + 3*c1*dw_y*sg[11]*v/radius + c1*dw_y*sg[7]*v_y/radius + c1*dw_y*sg[7]/radius + c1*dw_y*sg[8]*v_x/radius - dphiy*sg[10]*w_x - dphiy*sg[9]*w_y - dphiy*sg[4]*v_y/radius - dphiy*sg[4]/radius - dphiy*sg[5]*v_x/radius + dphiy*sg[9]*v/radius - dphiy*sg[4]*w/(radius*radius) - dphiy_x*sg[3]*w_x - dphiy_x*sg[5]*w_y + dphiy_x*sg[5]*v/radius - dphiy_y*sg[4]*w_y - dphiy_y*sg[5]*w_x + dphiy_y*sg[4]*v/radius - dsg[10]*phiy*w_x + dsg[10]*v_x - dsg[10]*v*w_x/radius - dsg[3]*phiy_x*w_x - dsg[3]*v_x*w_x/radius - dsg[4]*phiy*v_y/radius - dsg[4]*phiy/radius - dsg[4]*phiy*w/(radius*radius) - dsg[4]*phiy_y*w_y + dsg[4]*phiy_y*v/radius - dsg[4]*v_y*w_y/radius - dsg[4]*w_y/radius - dsg[4]*v*w/(radius*radius*radius) - dsg[5]*phiy*v_x/radius - dsg[5]*phiy_x*w_y + dsg[5]*phiy_x*v/radius - dsg[5]*phiy_y*w_x - dsg[5]*v_x*w_y/radius - dsg[5]*v_y*w_x/radius - dsg[5]*w_x/radius - dsg[9]*phiy*w_y + dsg[9]*phiy*v/radius + dsg[9]*v_y + dsg[9] - dsg[9]*v*w_y/radius + dsg[9]*w/radius + dsg[9]*(v*v)/(radius*radius) + dv*phiy*sg[9]/radius + dv*phiy_x*sg[5]/radius + dv*phiy_y*sg[4]/radius - dv*sg[10]*w_x/radius - dv*sg[9]*w_y/radius + 2*dv*sg[9]*v/(radius*radius) - dv*sg[4]*w/(radius*radius*radius) - dv_x*phiy*sg[5]/radius + dv_x*sg[10] - dv_x*sg[3]*w_x/radius - dv_x*sg[5]*w_y/radius - dv_y*phiy*sg[4]/radius + dv_y*sg[9] - dv_y*sg[4]*w_y/radius - dv_y*sg[5]*w_x/radius - dw*phiy*sg[4]/(radius*radius) + dw*sg[9]/radius - dw*sg[4]*v/(radius*radius*radius) - dw_x*phiy*sg[10] - dw_x*phiy_x*sg[3] - dw_x*phiy_y*sg[5] - dw_x*sg[10]*v/radius - dw_x*sg[3]*v_x/radius - dw_x*sg[5]*v_y/radius - dw_x*sg[5]/radius - dw_y*phiy*sg[9] - dw_y*phiy_x*sg[5] - dw_y*phiy_y*sg[4] - dw_y*sg[4]*v_y/radius - dw_y*sg[4]/radius - dw_y*sg[5]*v_x/radius - dw_y*sg[9]*v/radius
                S[10] = c1*dphix*sg[6]*w_x + c1*dphix*sg[8]*w_y - c1*dphix*sg[8]*v/radius + c1*dsg[6]*phix*w_x - c1*dsg[6]*u_x - c1*dsg[6] + c1*dsg[8]*phix*w_y - c1*dsg[8]*phix*v/radius - c1*dsg[8]*u_y - c1*du_x*sg[6] - c1*du_y*sg[8] - c1*dv*phix*sg[8]/radius + c1*dw_x*phix*sg[6] + c1*dw_y*phix*sg[8] - dphix*sg[3]*w_x - dphix*sg[5]*w_y + dphix*sg[5]*v/radius - dsg[3]*phix*w_x + dsg[3]*u_x + dsg[3] - dsg[5]*phix*w_y + dsg[5]*phix*v/radius + dsg[5]*u_y + du_x*sg[3] + du_y*sg[5] + dv*phix*sg[5]/radius - dw_x*phix*sg[3] - dw_y*phix*sg[5]
                S[11] = c1*dphix*sg[7]*w_y + c1*dphix*sg[8]*w_x - c1*dphix*sg[7]*v/radius + c1*dsg[7]*phix*w_y - c1*dsg[7]*phix*v/radius - c1*dsg[7]*u_y + c1*dsg[8]*phix*w_x - c1*dsg[8]*u_x - c1*dsg[8] - c1*du_x*sg[8] - c1*du_y*sg[7] - c1*dv*phix*sg[7]/radius + c1*dw_x*phix*sg[8] + c1*dw_y*phix*sg[7] - dphix*sg[4]*w_y - dphix*sg[5]*w_x + dphix*sg[4]*v/radius - dsg[4]*phix*w_y + dsg[4]*phix*v/radius + dsg[4]*u_y - dsg[5]*phix*w_x + dsg[5]*u_x + dsg[5] + du_x*sg[5] + du_y*sg[4] + dv*phix*sg[4]/radius - dw_x*phix*sg[5] - dw_y*phix*sg[4]
                S[12] = c1*dphiy*sg[6]*w_x + c1*dphiy*sg[8]*w_y - c1*dphiy*sg[8]*v/radius + c1*dsg[6]*phiy*w_x - c1*dsg[6]*v_x + c1*dsg[6]*v*w_x/radius + c1*dsg[8]*phiy*w_y - c1*dsg[8]*phiy*v/radius - c1*dsg[8]*v_y - c1*dsg[8] + c1*dsg[8]*v*w_y/radius - c1*dsg[8]*w/radius - c1*dsg[8]*(v*v)/(radius*radius) - c1*dv*phiy*sg[8]/radius + c1*dv*sg[6]*w_x/radius + c1*dv*sg[8]*w_y/radius - 2*c1*dv*sg[8]*v/(radius*radius) - c1*dv_x*sg[6] - c1*dv_y*sg[8] - c1*dw*sg[8]/radius + c1*dw_x*phiy*sg[6] + c1*dw_x*sg[6]*v/radius + c1*dw_y*phiy*sg[8] + c1*dw_y*sg[8]*v/radius - dphiy*sg[3]*w_x - dphiy*sg[5]*w_y + dphiy*sg[5]*v/radius - dsg[3]*phiy*w_x + dsg[3]*v_x - dsg[3]*v*w_x/radius - dsg[5]*phiy*w_y + dsg[5]*phiy*v/radius + dsg[5]*v_y + dsg[5] - dsg[5]*v*w_y/radius + dsg[5]*w/radius + dsg[5]*(v*v)/(radius*radius) + dv*phiy*sg[5]/radius - dv*sg[3]*w_x/radius - dv*sg[5]*w_y/radius + 2*dv*sg[5]*v/(radius*radius) + dv_x*sg[3] + dv_y*sg[5] + dw*sg[5]/radius - dw_x*phiy*sg[3] - dw_x*sg[3]*v/radius - dw_y*phiy*sg[5] - dw_y*sg[5]*v/radius
                S[13] = c1*dphiy*sg[7]*w_y + c1*dphiy*sg[8]*w_x - c1*dphiy*sg[7]*v/radius + c1*dsg[7]*phiy*w_y - c1*dsg[7]*phiy*v/radius - c1*dsg[7]*v_y - c1*dsg[7] + c1*dsg[7]*v*w_y/radius - c1*dsg[7]*w/radius - c1*dsg[7]*(v*v)/(radius*radius) + c1*dsg[8]*phiy*w_x - c1*dsg[8]*v_x + c1*dsg[8]*v*w_x/radius - c1*dv*phiy*sg[7]/radius + c1*dv*sg[7]*w_y/radius + c1*dv*sg[8]*w_x/radius - 2*c1*dv*sg[7]*v/(radius*radius) - c1*dv_x*sg[8] - c1*dv_y*sg[7] - c1*dw*sg[7]/radius + c1*dw_x*phiy*sg[8] + c1*dw_x*sg[8]*v/radius + c1*dw_y*phiy*sg[7] + c1*dw_y*sg[7]*v/radius - dphiy*sg[4]*w_y - dphiy*sg[5]*w_x + dphiy*sg[4]*v/radius - dsg[4]*phiy*w_y + dsg[4]*phiy*v/radius + dsg[4]*v_y + dsg[4] - dsg[4]*v*w_y/radius + dsg[4]*w/radius + dsg[4]*(v*v)/(radius*radius) - dsg[5]*phiy*w_x + dsg[5]*v_x - dsg[5]*v*w_x/radius + dv*phiy*sg[4]/radius - dv*sg[4]*w_y/radius - dv*sg[5]*w_x/radius + 2*dv*sg[4]*v/(radius*radius) + dv_x*sg[5] + dv_y*sg[4] + dw*sg[4]/radius - dw_x*phiy*sg[5] - dw_x*sg[5]*v/radius - dw_y*phiy*sg[4] - dw_y*sg[4]*v/radius
                S[14] = -c1*dsg[6]*u_x*v_y - c1*dsg[6]*u_x + c1*dsg[6]*u_y*v_x - c1*dsg[6]*v_y - c1*dsg[6] - c1*dsg[6]*u_x*w/radius - c1*dsg[6]*w/radius - c1*du_x*sg[6]*v_y - c1*du_x*sg[6] - c1*du_x*sg[6]*w/radius + c1*du_y*sg[6]*v_x + c1*dv_x*sg[6]*u_y - c1*dv_y*sg[6]*u_x - c1*dv_y*sg[6] - c1*dw*sg[6]*u_x/radius - c1*dw*sg[6]/radius
                S[15] = -c1*dsg[7]*u_x*v_y - c1*dsg[7]*u_x + c1*dsg[7]*u_y*v_x - c1*dsg[7]*v_y - c1*dsg[7] - c1*dsg[7]*u_x*w/radius - c1*dsg[7]*w/radius - c1*du_x*sg[7]*v_y - c1*du_x*sg[7] - c1*du_x*sg[7]*w/radius + c1*du_y*sg[7]*v_x + c1*dv_x*sg[7]*u_y - c1*dv_y*sg[7]*u_x - c1*dv_y*sg[7] - c1*dw*sg[7]*u_x/radius - c1*dw*sg[7]/radius
                S[16] = -2*c1*dsg[8]*u_x*v_y - 2*c1*dsg[8]*u_x + 2*c1*dsg[8]*u_y*v_x - 2*c1*dsg[8]*v_y - 2*c1*dsg[8] - 2*c1*dsg[8]*u_x*w/radius - 2*c1*dsg[8]*w/radius - 2*c1*du_x*sg[8]*v_y - 2*c1*du_x*sg[8] - 2*c1*du_x*sg[8]*w/radius + 2*c1*du_y*sg[8]*v_x + 2*c1*dv_x*sg[8]*u_y - 2*c1*dv_y*sg[8]*u_x - 2*c1*dv_y*sg[8] - 2*c1*dw*sg[8]*u_x/radius - 2*c1*dw*sg[8]/radius
                S[17] = -c1*dsg[6]*v_x*w_y + c1*dsg[6]*v_y*w_x + c1*dsg[6]*w_x + c1*dsg[6]*v*v_x/radius + c1*dsg[6]*w*w_x/radius + c1*dv*sg[6]*v_x/radius - c1*dv_x*sg[6]*w_y + c1*dv_x*sg[6]*v/radius + c1*dv_y*sg[6]*w_x + c1*dw*sg[6]*w_x/radius + c1*dw_x*sg[6]*v_y + c1*dw_x*sg[6] + c1*dw_x*sg[6]*w/radius - c1*dw_y*sg[6]*v_x
                S[18] = -c1*dsg[7]*v_x*w_y + c1*dsg[7]*v_y*w_x + c1*dsg[7]*w_x + c1*dsg[7]*v*v_x/radius + c1*dsg[7]*w*w_x/radius + c1*dv*sg[7]*v_x/radius - c1*dv_x*sg[7]*w_y + c1*dv_x*sg[7]*v/radius + c1*dv_y*sg[7]*w_x + c1*dw*sg[7]*w_x/radius + c1*dw_x*sg[7]*v_y + c1*dw_x*sg[7] + c1*dw_x*sg[7]*w/radius - c1*dw_y*sg[7]*v_x
                S[19] = -2*c1*dsg[8]*v_x*w_y + 2*c1*dsg[8]*v_y*w_x + 2*c1*dsg[8]*w_x + 2*c1*dsg[8]*v*v_x/radius + 2*c1*dsg[8]*w*w_x/radius + 2*c1*dv*sg[8]*v_x/radius - 2*c1*dv_x*sg[8]*w_y + 2*c1*dv_x*sg[8]*v/radius + 2*c1*dv_y*sg[8]*w_x + 2*c1*dw*sg[8]*w_x/radius + 2*c1*dw_x*sg[8]*v_y + 2*c1*dw_x*sg[8] + 2*c1*dw_x*sg[8]*w/radius - 2*c1*dw_y*sg[8]*v_x
                S[20] = c1*dsg[6]*u_x*w_y - c1*dsg[6]*u_y*w_x + c1*dsg[6]*w_y - c1*dsg[6]*u_x*v/radius - c1*dsg[6]*v/radius + c1*du_x*sg[6]*w_y - c1*du_x*sg[6]*v/radius - c1*du_y*sg[6]*w_x - c1*dv*sg[6]*u_x/radius - c1*dv*sg[6]/radius - c1*dw_x*sg[6]*u_y + c1*dw_y*sg[6]*u_x + c1*dw_y*sg[6]
                S[21] = c1*dsg[7]*u_x*w_y - c1*dsg[7]*u_y*w_x + c1*dsg[7]*w_y - c1*dsg[7]*u_x*v/radius - c1*dsg[7]*v/radius + c1*du_x*sg[7]*w_y - c1*du_x*sg[7]*v/radius - c1*du_y*sg[7]*w_x - c1*dv*sg[7]*u_x/radius - c1*dv*sg[7]/radius - c1*dw_x*sg[7]*u_y + c1*dw_y*sg[7]*u_x + c1*dw_y*sg[7]
                S[22] = 2*c1*dsg[8]*u_x*w_y - 2*c1*dsg[8]*u_y*w_x + 2*c1*dsg[8]*w_y - 2*c1*dsg[8]*u_x*v/radius - 2*c1*dsg[8]*v/radius + 2*c1*du_x*sg[8]*w_y - 2*c1*du_x*sg[8]*v/radius - 2*c1*du_y*sg[8]*w_x - 2*c1*dv*sg[8]*u_x/radius - 2*c1*dv*sg[8]/radius - 2*c1*dw_x*sg[8]*u_y + 2*c1*dw_y*sg[8]*u_x + 2*c1*dw_y*sg[8]

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
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAphix*gAphix
                        pA[9] = fAphiy*gAphiy
                        pA[10] = fAphix_x*gAphix
                        pA[11] = fAphix*gAphix_y
                        pA[12] = fAphiy_x*gAphiy
                        pA[13] = fAphiy*gAphiy_y
                        pA[14] = fAw_xx*gAw
                        pA[15] = fAw*gAw_yy
                        pA[16] = fAw_x*gAw_y
                        pA[17] = fAu_xx*gAu
                        pA[18] = fAu*gAu_yy
                        pA[19] = fAu_x*gAu_y
                        pA[20] = fAv_xx*gAv
                        pA[21] = fAv*gAv_yy
                        pA[22] = fAv_x*gAv_y
                        kTv[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1] + pA[17]*S[17] + pA[18]*S[18] + pA[19]*S[19])
                        kTv[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[6]*S[6] + pA[20]*S[20] + pA[21]*S[21] + pA[22]*S[22])
                        kTv[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[7]*S[7] + pA[14]*S[14] + pA[15]*S[15] + pA[16]*S[16])
                        kTv[col+3] += weight*(pA[8]*S[8] + pA[10]*S[10] + pA[11]*S[11])
                        kTv[col+4] += weight*(pA[9]*S[9] + pA[12]*S[12] + pA[13]*S[13])

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

            ax = qf*v_x*w_y - qf*v_y*w_x - qf*v_x*vv/r - qf*w_x*ww/r - w_x
        ay = -qf*u_x*w_y + qf*u_y*w_x + qf*u_x*vv/r - w_y + vv/r
        az = qf*u_x*v_y - qf*u_y*v_x + qf*u_x*ww/r + ref + u_x + v_y + ww/r

    with ``qf = 1`` for ``quadratic=1`` and ``qf = 0`` otherwise, and ``ref =
    1`` for ``reference=1`` and ``ref = 0`` otherwise. The part
    `\boldsymbol{a}_2` is of the order of the terms that the strains
    neglect.

    """
    cdef double [:, ::1] gtab, ftab
    cdef double a, b, pintx, pinty, qf
    cdef double r
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
    r = shell.r
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
                u_x = u_y = v_x = v_y = vv = ww = w_x = w_y = 0.
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
                        vv += cs[col+1]*fAv*gAv
                        ww += cs[col+2]*fAw*gAw
                        w_x += cs[col+2]*(2/a)*fAw_xi*gAw
                        w_y += cs[col+2]*(2/b)*fAw*gAw_eta

                ax = qf*v_x*w_y - qf*v_y*w_x - qf*v_x*vv/r - qf*w_x*ww/r - w_x
                ay = -qf*u_x*w_y + qf*u_y*w_x + qf*u_x*vv/r - w_y + vv/r
                az = qf*u_x*v_y - qf*u_y*v_x + qf*u_x*ww/r + ref + u_x + v_y + ww/r

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
    cdef double r
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
    r = shell.r
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

    fdim = 8*m*m*n*n
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
                u_x = u_y = v_x = v_y = vv = ww = w_x = w_y = 0.
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
                            vv += cs[col+1]*fAv*gAv
                            ww += cs[col+2]*fAw*gAw
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
                                kCv[c] += weight*( fAu*fBv*gAu*gBv*qf*v_x/r + 2*fAu*fBv*gAu*gBv_eta*qf*w_x/b - 2*fAu*fBv_xi*gAu*gBv*qf*w_y/a + 2*fAu*fBv_xi*gAu*gBv*qf*vv/(a*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*( fAu*fBw*gAu*gBw*qf*w_x/r - 2*fAu*fBw*gAu*gBw_eta*qf*v_x/b + 2*fAu*fBw_xi*gAu*gBw*qf*v_y/a + 2*fAu*fBw_xi*gAu*gBw*qf*ww/(a*r) + 2*fAu*fBw_xi*gAu*gBw/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*( -2*fAv*fBu*gAv*gBu_eta*qf*w_x/b + 2*fAv*fBu_xi*gAv*gBu*qf*w_y/a - 2*fAv*fBu_xi*gAv*gBu*qf*vv/(a*r) )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*( -fAv*fBv*gAv*gBv*qf*u_x/r - fAv*fBv*gAv*gBv/r )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*( 2*fAv*fBw*gAv*gBw_eta*qf*u_x/b + 2*fAv*fBw*gAv*gBw_eta/b - 2*fAv*fBw_xi*gAv*gBw*qf*u_y/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*( 2*fAw*fBu*gAw*gBu_eta*qf*v_x/b - 2*fAw*fBu_xi*gAw*gBu*qf*v_y/a - 2*fAw*fBu_xi*gAw*gBu*qf*ww/(a*r) - 2*fAw*fBu_xi*gAw*gBu/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*( -2*fAw*fBv*gAw*gBv_eta*qf*u_x/b - 2*fAw*fBv*gAw*gBv_eta/b + 2*fAw*fBv_xi*gAw*gBv*qf*u_y/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*( -fAw*fBw*gAw*gBw*qf*u_x/r - fAw*fBw*gAw*gBw/r )

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC
# END follower pressure
