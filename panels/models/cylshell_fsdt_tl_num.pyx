#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Cylindrical shells using the total-Lagrangian first-order shear deformation
theory (FSDT) with a second-order director

Numerically integrated kernels of the model ``'cylshell_fsdt_tl'``: the
constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the internal force vector :func:`calc_fint`, the product
of the tangent stiffness matrix by a vector :func:`calc_kTv`, and the force
vector and load stiffness matrix of a follower pressure,
:func:`calc_fext_follower` and :func:`fkCfollower_num`. The mass and
aerodynamic matrices, ``fkM_num``, ``fkAx_num`` and ``fkAy_num``, are those
of :mod:`panels.models.cylshell_fsdt_sanders_num`. The constitutive matrix ``Finput`` is the
``8 x 8`` matrix ``ABD`` of :class:`.Shell` for this model.

The degrees of freedom of each term of the approximation are
``u, v, w, phix, phiy``; the degree of freedom ``k`` of the term ``(i, j)``,
``0 <= i < m``, ``0 <= j < n``, is at the row ``row0 + 5*(j*m + i) + k``
of the matrices.

Notes
-----
The displacement field is that of the moderate rotation theory of Schmidt
and Reddy [1]_ and Palmerio et al. [2]_, linear through the thickness, with
the transverse normal strain of [2]_, Eq. (20), set to zero (inextensible
normal). In the local basis `(\mathbf{e}_x, \mathbf{e}_y,
\mathbf{e}_z)`, `\mathbf{e}_{y,y} = -\mathbf{e}_z/r`, `\mathbf{e}_{z,y} =
\mathbf{e}_y/r`, the deformed position of the point at `z` is
`\mathbf{X} + z \mathbf{d}`, with

.. math::

    \mathbf{X}_{,x} = (1 + u_{,x}, \; v_{,x}, \; w_{,x}) \qquad
    \mathbf{X}_{,y} = (u_{,y}, \; 1 + v_{,y} + w/r, \; w_{,y} - v/r)

    \mathbf{d} = \left(\psi_x, \; \psi_y, \; 1 - \frac{1}{2}(\psi_x^2 +
    \psi_y^2)\right) \qquad \psi_x = \phi_x \qquad \psi_y = \phi_y + v/r

the second-order director, whose components `\psi_x`, `\psi_y` are the
rotations of the normal of the transverse shear strains of [2]_, Eq. (20),
`\gamma_{yz} = \psi_y + w_{,y} - v/r = \phi_y + w_{,y}`. The strains are

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} (u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2)

    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \left(u_{,y}^2 +
    (v_{,y} + w/r)^2 + (w_{,y} - v/r)^2\right)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} (v_{,y} + w/r)
    + w_{,x} (w_{,y} - v/r)

    \kappa_{xx} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,x} \qquad
    \kappa_{yy} = \mathbf{X}_{,y} \cdot \mathbf{d}_{,y} - \frac{1}{r} -
    \frac{\varepsilon_{yy}}{r}

    \kappa_{xy} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,y} + \mathbf{X}_{,y}
    \cdot \mathbf{d}_{,x} - \frac{\gamma_{xy}}{2 r}

    \gamma_{yz} = \mathbf{X}_{,y} \cdot \mathbf{d} \qquad
    \gamma_{xz} = \mathbf{X}_{,x} \cdot \mathbf{d}

cubic polynomials of the generalized derivatives. The membrane strains are
those of :mod:`panels.models.cylshell_fsdt_gl`. Without the terms
`-\varepsilon_{yy}/r` and `-\gamma_{xy}/(2r)`, the curvatures are the
coefficients of `z` of the Green-Lagrange strains of the field above; the
two terms make the linear part that of
:mod:`panels.models.cylshell_fsdt_sanders` [3]_, as for
:mod:`panels.models.cylshell_clpt_tl`. To the order `\epsilon^3` of the
moderate rotation theory, with strains of order `\epsilon^2` and rotations
of the normal of order `\epsilon`, the strains without these two terms are
those of [2]_, Eq. (20). A finite rigid rotation `t` changes the curvatures
and the transverse shear strains by `O(t^4)`, against `O(t^2)` for the
linear curvatures of :mod:`panels.models.cylshell_fsdt_gl`.

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
#     of cylshell_fsdt_sanders
from .cylshell_fsdt_sanders_num import fkM_num, fkAx_num, fkAy_num


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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the values of `v` and `w`, the rotations and their gradients), the strains
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
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w
    cdef double phix, phiy, phix_x, phix_y, phiy_x, phiy_y
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
    cdef double Bm[112]
    cdef double FB[112]
    cdef double K[196]
    cdef double TA[70]
    cdef double pA[14]
    cdef double pB[14]

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
                v = 0.
                w = 0.
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
                            v += fAv*gAv*cs[col+1]
                            w += fAw*gAw*cs[col+2]
                            phix += fAphix*gAphix*cs[col+3]
                            phiy += fAphiy*gAphiy*cs[col+4]
                            phix_x += fAphix_x*gAphix*cs[col+3]
                            phix_y += fAphix*gAphix_y*cs[col+3]
                            phiy_x += fAphiy_x*gAphiy*cs[col+4]
                            phiy_y += fAphiy*gAphiy_y*cs[col+4]

                # [B] = d eps/d d at the point
                for p in range(112):
                    Bm[p] = 0.
                Bm[0] = u_x + 1
                Bm[2] = v_x
                Bm[4] = w_x
                Bm[15] = u_y
                Bm[17] = v_y + 1 + w/radius
                Bm[19] = w_y - v/radius
                Bm[20] = -w_y/radius + v/(radius*radius)
                Bm[21] = v_y/radius + 1/radius + w/(radius*radius)
                Bm[28] = u_y
                Bm[29] = u_x + 1
                Bm[30] = v_y + 1 + w/radius
                Bm[31] = v_x
                Bm[32] = w_y - v/radius
                Bm[33] = w_x
                Bm[34] = -w_x/radius
                Bm[35] = v_x/radius
                Bm[42] = phix_x
                Bm[44] = -phiy*w_x/radius + phiy_x + 2*v_x/radius - v*w_x/(radius*radius)
                Bm[46] = -phix*phix_x - phiy*phiy_x - phiy*v_x/radius - phiy_x*v/radius - v*v_x/(radius*radius)
                Bm[48] = -phiy_x*w_x/radius - v_x*w_x/(radius*radius)
                Bm[50] = -phix_x*w_x
                Bm[51] = -phiy_x*w_x - v_x*w_x/radius
                Bm[52] = -phix*w_x + u_x + 1
                Bm[54] = -phiy*w_x + v_x - v*w_x/radius
                Bm[57] = phix_y - u_y/radius
                Bm[59] = -0.5*(phix*phix)/radius - 0.5*(phiy*phiy)/radius - phiy*w_y/radius + phiy_y + v_y/radius + 1/radius - v*w_y/(radius*radius) + 0.5*(v*v)/(radius*radius*radius)
                Bm[61] = -phix*phix_y - phiy*phiy_y - phiy*v_y/radius - phiy/radius - phiy_y*v/radius - w_y/radius - v*v_y/(radius*radius)
                Bm[62] = phix*phix_y/radius + phiy*phiy_y/radius - phiy*w/(radius*radius*radius) - phiy_y*w_y/radius + 2*phiy_y*v/(radius*radius) - v_y*w_y/(radius*radius) + v*v_y/(radius*radius*radius) - v*w/(radius*radius*radius*radius)
                Bm[63] = -0.5*(phix*phix)/(radius*radius) - 0.5*(phiy*phiy)/(radius*radius) - phiy*v/(radius*radius*radius) + phiy_y/radius - w/(radius*radius*radius) - 0.5*(v*v)/(radius*radius*radius*radius)
                Bm[64] = -phix*v_y/radius - phix/radius - phix*w/(radius*radius) - phix_y*w_y + phix_y*v/radius
                Bm[65] = -phiy*v_y/radius - phiy/radius - phiy*w/(radius*radius) - phiy_y*w_y + phiy_y*v/radius - v_y*w_y/radius - w_y/radius - v*w/(radius*radius*radius)
                Bm[67] = -phix*w_y + phix*v/radius + u_y
                Bm[69] = -phiy*w_y + phiy*v/radius + v_y + 1 - v*w_y/radius + w/radius + (v*v)/(radius*radius)
                Bm[70] = phix_y - 0.5*u_y/radius
                Bm[71] = phix_x - 0.5*u_x/radius - 0.5/radius
                Bm[72] = -0.5*(phix*phix)/radius - 0.5*(phiy*phiy)/radius - phiy*w_y/radius + phiy_y + 1.5*v_y/radius + 1.5/radius - v*w_y/(radius*radius) + 0.5*w/(radius*radius) + 0.5*(v*v)/(radius*radius*radius)
                Bm[73] = -phiy*w_x/radius + phiy_x + 1.5*v_x/radius - v*w_x/(radius*radius)
                Bm[74] = -phix*phix_y - phiy*phiy_y - phiy*v_y/radius - phiy/radius - phiy_y*v/radius - 0.5*w_y/radius - v*v_y/(radius*radius) - 0.5*v/(radius*radius)
                Bm[75] = -phix*phix_x - phiy*phiy_x - phiy*v_x/radius - phiy_x*v/radius - 0.5*w_x/radius - v*v_x/(radius*radius)
                Bm[76] = phix*phix_x/radius + phiy*phiy_x/radius - phiy_x*w_y/radius + 2*phiy_x*v/(radius*radius) - phiy_y*w_x/radius - v_x*w_y/(radius*radius) - v_y*w_x/(radius*radius) - 0.5*w_x/(radius*radius) + v*v_x/(radius*radius*radius)
                Bm[77] = phiy_x/radius + 0.5*v_x/(radius*radius)
                Bm[78] = -phix*v_x/radius - phix_x*w_y + phix_x*v/radius - phix_y*w_x
                Bm[79] = -phiy*v_x/radius - phiy_x*w_y + phiy_x*v/radius - phiy_y*w_x - v_x*w_y/radius - v_y*w_x/radius - w_x/radius
                Bm[80] = -phix*w_y + phix*v/radius + u_y
                Bm[81] = -phix*w_x + u_x + 1
                Bm[82] = -phiy*w_y + phiy*v/radius + v_y + 1 - v*w_y/radius + w/radius + (v*v)/(radius*radius)
                Bm[83] = -phiy*w_x + v_x - v*w_x/radius
                Bm[85] = phix
                Bm[87] = phiy + v/radius
                Bm[89] = -0.5*(phix*phix) - 0.5*(phiy*phiy) - phiy*v/radius + 1 - 0.5*(v*v)/(radius*radius)
                Bm[90] = 0.5*(phix*phix)/radius + 0.5*(phiy*phiy)/radius - phiy*w_y/radius + 2*phiy*v/(radius*radius) + v_y/radius - v*w_y/(radius*radius) + w/(radius*radius) + 1.5*(v*v)/(radius*radius*radius)
                Bm[91] = phiy/radius + v/(radius*radius)
                Bm[92] = -phix*w_y + phix*v/radius + u_y
                Bm[93] = -phiy*w_y + phiy*v/radius + v_y + 1 - v*w_y/radius + w/radius + (v*v)/(radius*radius)
                Bm[98] = phix
                Bm[100] = phiy + v/radius
                Bm[102] = -0.5*(phix*phix) - 0.5*(phiy*phiy) - phiy*v/radius + 1 - 0.5*(v*v)/(radius*radius)
                Bm[104] = -phiy*w_x/radius + v_x/radius - v*w_x/(radius*radius)
                Bm[106] = -phix*w_x + u_x + 1
                Bm[107] = -phiy*w_x + v_x - v*w_x/radius

                # K = [B]^T [F] [B]
                for r in range(NE):
                    for q in range(14):
                        FB[r*14 + q] = 0.
                        for s in range(NE):
                            FB[r*14 + q] += F[r*NE + s]*Bm[s*14 + q]
                for p in range(14):
                    for q in range(14):
                        K[p*14 + q] = 0.
                        for r in range(NE):
                            K[p*14 + q] += Bm[r*14 + p]*FB[r*14 + q]

                # stresses of the strains and of the linear strains
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x - phiy*v_x*w_x/radius + phiy_x*v_x - phiy_x*v*w_x/radius + (v_x*v_x)/radius - v*v_x*w_x/(radius*radius)
                e[4] = -0.5*(phix*phix)*v_y/radius - 0.5*(phix*phix)/radius - 0.5*(phix*phix)*w/(radius*radius) - phix*phix_y*w_y + phix*phix_y*v/radius + phix_y*u_y - 0.5*(phiy*phiy)*v_y/radius - 0.5*(phiy*phiy)/radius - 0.5*(phiy*phiy)*w/(radius*radius) - phiy*phiy_y*w_y + phiy*phiy_y*v/radius - phiy*v_y*w_y/radius - phiy*w_y/radius - phiy*v*w/(radius*radius*radius) + phiy_y*v_y + phiy_y - phiy_y*v*w_y/radius + phiy_y*w/radius + phiy_y*(v*v)/(radius*radius) - 0.5*(u_y*u_y)/radius + 0.5*(v_y*v_y)/radius + v_y/radius - 0.5*(w_y*w_y)/radius - v*v_y*w_y/(radius*radius) + 0.5*(v*v)*v_y/(radius*radius*radius) - 0.5*(w*w)/(radius*radius*radius) - 0.5*(v*v)*w/(radius*radius*radius*radius)
                e[5] = -0.5*(phix*phix)*v_x/radius - phix*phix_x*w_y + phix*phix_x*v/radius - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - 0.5*(phiy*phiy)*v_x/radius - phiy*phiy_x*w_y + phiy*phiy_x*v/radius - phiy*phiy_y*w_x - phiy*v_x*w_y/radius - phiy*v_y*w_x/radius - phiy*w_x/radius + phiy_x*v_y + phiy_x - phiy_x*v*w_y/radius + phiy_x*w/radius + phiy_x*(v*v)/(radius*radius) + phiy_y*v_x - phiy_y*v*w_x/radius - 0.5*u_x*u_y/radius - 0.5*u_y/radius + 1.5*v_x*v_y/radius + 1.5*v_x/radius - 0.5*w_x*w_y/radius - v*v_x*w_y/(radius*radius) - v*v_y*w_x/(radius*radius) - 0.5*v*w_x/(radius*radius) + 0.5*v_x*w/(radius*radius) + 0.5*(v*v)*v_x/(radius*radius*radius)
                e[6] = -0.5*(phix*phix)*w_y + 0.5*(phix*phix)*v/radius + phix*u_y - 0.5*(phiy*phiy)*w_y + 0.5*(phiy*phiy)*v/radius + phiy*v_y + phiy - phiy*v*w_y/radius + phiy*w/radius + phiy*(v*v)/(radius*radius) + w_y + v*v_y/radius - 0.5*(v*v)*w_y/(radius*radius) + v*w/(radius*radius) + 0.5*(v*v*v)/(radius*radius*radius)
                e[7] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x - phiy*v*w_x/radius + w_x + v*v_x/radius - 0.5*(v*v)*w_x/(radius*radius)
                eL[0] = u_x
                eL[1] = v_y + w/radius
                eL[2] = u_y + v_x
                eL[3] = phix_x
                eL[4] = phiy_y + v_y/radius
                eL[5] = phix_y + phiy_x - 0.5*u_y/radius + 1.5*v_x/radius
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
                K[1] += -sgL[2] + sg[2] + 0.5*sgL[5]/radius - 0.5*sg[5]/radius
                K[8] += -sgL[7] + sg[7]
                K[10] += -sgL[3] + sg[3]
                K[11] += -sgL[5] + sg[5]
                K[14] += -sgL[2] + sg[2] + 0.5*sgL[5]/radius - 0.5*sg[5]/radius
                K[15] += -sgL[1] + sg[1] + sgL[4]/radius - sg[4]/radius
                K[22] += -sgL[6] + sg[6]
                K[24] += -sgL[5] + sg[5]
                K[25] += -sgL[4] + sg[4]
                K[30] += -sgL[0] + sg[0] - 2*sgL[3]/radius + 2*sg[3]/radius
                K[31] += -sgL[2] + sg[2] - 1.5*sgL[5]/radius + 1.5*sg[5]/radius
                K[32] += -phiy*sg[3]/radius - sg[3]*v/(radius*radius)
                K[33] += -phiy*sg[5]/radius - sg[5]*v/(radius*radius)
                K[34] += -sgL[7]/radius + sg[7]/radius - sg[3]*w_x/(radius*radius) - sg[5]*w_y/(radius*radius) + sg[5]*v/(radius*radius*radius)
                K[35] += -sgL[2]/radius + sg[2]/radius - 0.5*sgL[5]/(radius*radius) + 0.5*sg[5]/(radius*radius)
                K[36] += -phix*sg[5]/radius
                K[37] += -phiy*sg[5]/radius - sgL[7] + sg[7] - sg[3]*w_x/radius - sg[5]*w_y/radius
                K[40] += -sgL[3] + sg[3]
                K[41] += -sgL[5] + sg[5]
                K[44] += -sgL[2] + sg[2] - 1.5*sgL[5]/radius + 1.5*sg[5]/radius
                K[45] += -sgL[1] + sg[1] - sgL[4]/radius + sg[4]/radius
                K[46] += -phiy*sg[5]/radius - sg[5]*v/(radius*radius)
                K[47] += -phiy*sg[4]/radius - sg[4]*v/(radius*radius)
                K[48] += -sgL[6]/radius + sg[6]/radius - sg[4]*w_y/(radius*radius) - sg[5]*w_x/(radius*radius) + sg[4]*v/(radius*radius*radius)
                K[49] += -sgL[1]/radius + sg[1]/radius
                K[50] += -phix*sg[4]/radius
                K[51] += -phiy*sg[4]/radius - sgL[6] + sg[6] - sg[4]*w_y/radius - sg[5]*w_x/radius
                K[54] += -sgL[5] + sg[5]
                K[55] += -sgL[4] + sg[4]
                K[58] += -phiy*sg[3]/radius - sg[3]*v/(radius*radius)
                K[59] += -phiy*sg[5]/radius - sg[5]*v/(radius*radius)
                K[60] += -sgL[0] + sg[0]
                K[61] += -sgL[2] + sg[2] + 0.5*sgL[5]/radius - 0.5*sg[5]/radius
                K[62] += -phiy*sg[7]/radius - phiy_x*sg[3]/radius - phiy_y*sg[5]/radius + sgL[2]/radius - sg[2]/radius + 0.5*sgL[5]/(radius*radius) - sg[3]*v_x/(radius*radius) - sg[5]*v_y/(radius*radius) - 0.5*sg[5]/(radius*radius) - sg[7]*v/(radius*radius)
                K[64] += -phix*sg[7] - phix_x*sg[3] - phix_y*sg[5]
                K[65] += -phiy*sg[7] - phiy_x*sg[3] - phiy_y*sg[5] + sgL[5]/radius - sg[3]*v_x/radius - sg[5]*v_y/radius - sg[5]/radius - sg[7]*v/radius
                K[66] += -phix*sg[3]
                K[67] += -phix*sg[5]
                K[68] += -phiy*sg[3] - sg[3]*v/radius
                K[69] += -phiy*sg[5] - sg[5]*v/radius
                K[72] += -phiy*sg[5]/radius - sg[5]*v/(radius*radius)
                K[73] += -phiy*sg[4]/radius - sg[4]*v/(radius*radius)
                K[74] += -sgL[2] + sg[2] + 0.5*sgL[5]/radius - 0.5*sg[5]/radius
                K[75] += -sgL[1] + sg[1] + sgL[4]/radius - sg[4]/radius
                K[76] += -phiy*sg[6]/radius - phiy_x*sg[5]/radius - phiy_y*sg[4]/radius + sgL[1]/radius - sg[1]/radius - sg[4]*v_y/(radius*radius) - sg[5]*v_x/(radius*radius) - sg[6]*v/(radius*radius)
                K[78] += -phix*sg[6] - phix_x*sg[5] - phix_y*sg[4]
                K[79] += -phiy*sg[6] - phiy_x*sg[5] - phiy_y*sg[4] + sgL[4]/radius - sg[4]*v_y/radius - sg[4]/radius - sg[5]*v_x/radius - sg[6]*v/radius
                K[80] += -phix*sg[5]
                K[81] += -phix*sg[4]
                K[82] += -phiy*sg[5] - sg[5]*v/radius
                K[83] += -phiy*sg[4] - sg[4]*v/radius
                K[86] += -sgL[7]/radius + sg[7]/radius - sg[3]*w_x/(radius*radius) - sg[5]*w_y/(radius*radius) + sg[5]*v/(radius*radius*radius)
                K[87] += -sgL[6]/radius + sg[6]/radius - sg[4]*w_y/(radius*radius) - sg[5]*w_x/(radius*radius) + sg[4]*v/(radius*radius*radius)
                K[88] += -phiy*sg[7]/radius - phiy_x*sg[3]/radius - phiy_y*sg[5]/radius + sgL[2]/radius - sg[2]/radius + 0.5*sgL[5]/(radius*radius) - sg[3]*v_x/(radius*radius) - sg[5]*v_y/(radius*radius) - 0.5*sg[5]/(radius*radius) - sg[7]*v/(radius*radius)
                K[89] += -phiy*sg[6]/radius - phiy_x*sg[5]/radius - phiy_y*sg[4]/radius + sgL[1]/radius - sg[1]/radius - sg[4]*v_y/(radius*radius) - sg[5]*v_x/(radius*radius) - sg[6]*v/(radius*radius)
                K[90] += 2*phiy*sg[6]/(radius*radius) + 2*phiy_x*sg[5]/(radius*radius) + 2*phiy_y*sg[4]/(radius*radius) - sgL[1]/(radius*radius) + sg[1]/(radius*radius) - sg[6]*w_y/(radius*radius) - sg[7]*w_x/(radius*radius) + sg[4]*v_y/(radius*radius*radius) + sg[5]*v_x/(radius*radius*radius) + 3*sg[6]*v/(radius*radius*radius) - sg[4]*w/(radius*radius*radius*radius)
                K[91] += -phiy*sg[4]/(radius*radius*radius) - sgL[6]/(radius*radius) + sg[6]/(radius*radius) - sg[4]*v/(radius*radius*radius*radius)
                K[92] += phix*sg[6]/radius + phix_x*sg[5]/radius + phix_y*sg[4]/radius
                K[93] += phiy*sg[6]/radius + phiy_x*sg[5]/radius + phiy_y*sg[4]/radius - sg[6]*w_y/radius - sg[7]*w_x/radius + 2*sg[6]*v/(radius*radius) - sg[4]*w/(radius*radius*radius)
                K[94] += phix*sg[5]/radius
                K[95] += phix*sg[4]/radius
                K[96] += phiy*sg[5]/radius - sg[3]*w_x/radius - sg[5]*w_y/radius + 2*sg[5]*v/(radius*radius)
                K[97] += phiy*sg[4]/radius - sg[4]*w_y/radius - sg[5]*w_x/radius + 2*sg[4]*v/(radius*radius)
                K[100] += -sgL[2]/radius + sg[2]/radius - 0.5*sgL[5]/(radius*radius) + 0.5*sg[5]/(radius*radius)
                K[101] += -sgL[1]/radius + sg[1]/radius
                K[104] += -phiy*sg[4]/(radius*radius*radius) - sgL[6]/(radius*radius) + sg[6]/(radius*radius) - sg[4]*v/(radius*radius*radius*radius)
                K[105] += -sgL[1]/(radius*radius) + sg[1]/(radius*radius) + sgL[4]/(radius*radius*radius) - sg[4]/(radius*radius*radius)
                K[106] += -phix*sg[4]/(radius*radius)
                K[107] += -phiy*sg[4]/(radius*radius) - sgL[6]/radius + sg[6]/radius - sg[4]*v/(radius*radius*radius)
                K[110] += -sgL[5]/radius + sg[5]/radius
                K[111] += -sgL[4]/radius + sg[4]/radius
                K[112] += -sgL[7] + sg[7]
                K[113] += -sgL[6] + sg[6]
                K[114] += -phix*sg[5]/radius
                K[115] += -phix*sg[4]/radius
                K[116] += -phix*sg[7] - phix_x*sg[3] - phix_y*sg[5]
                K[117] += -phix*sg[6] - phix_x*sg[5] - phix_y*sg[4]
                K[118] += phix*sg[6]/radius + phix_x*sg[5]/radius + phix_y*sg[4]/radius
                K[119] += -phix*sg[4]/(radius*radius)
                K[120] += -sg[6]*w_y - sg[7]*w_x + sgL[4]/radius - sg[4]*v_y/radius - sg[4]/radius - sg[5]*v_x/radius + sg[6]*v/radius - sg[4]*w/(radius*radius)
                K[122] += -sg[3]*w_x - sg[5]*w_y + sg[5]*v/radius
                K[123] += -sg[4]*w_y - sg[5]*w_x + sg[4]*v/radius
                K[128] += -phiy*sg[5]/radius - sgL[7] + sg[7] - sg[3]*w_x/radius - sg[5]*w_y/radius
                K[129] += -phiy*sg[4]/radius - sgL[6] + sg[6] - sg[4]*w_y/radius - sg[5]*w_x/radius
                K[130] += -phiy*sg[7] - phiy_x*sg[3] - phiy_y*sg[5] + sgL[5]/radius - sg[3]*v_x/radius - sg[5]*v_y/radius - sg[5]/radius - sg[7]*v/radius
                K[131] += -phiy*sg[6] - phiy_x*sg[5] - phiy_y*sg[4] + sgL[4]/radius - sg[4]*v_y/radius - sg[4]/radius - sg[5]*v_x/radius - sg[6]*v/radius
                K[132] += phiy*sg[6]/radius + phiy_x*sg[5]/radius + phiy_y*sg[4]/radius - sg[6]*w_y/radius - sg[7]*w_x/radius + 2*sg[6]*v/(radius*radius) - sg[4]*w/(radius*radius*radius)
                K[133] += -phiy*sg[4]/(radius*radius) - sgL[6]/radius + sg[6]/radius - sg[4]*v/(radius*radius*radius)
                K[135] += -sg[6]*w_y - sg[7]*w_x + sgL[4]/radius - sg[4]*v_y/radius - sg[4]/radius - sg[5]*v_x/radius + sg[6]*v/radius - sg[4]*w/(radius*radius)
                K[138] += -sg[3]*w_x - sg[5]*w_y + sg[5]*v/radius
                K[139] += -sg[4]*w_y - sg[5]*w_x + sg[4]*v/radius
                K[140] += -sgL[3] + sg[3]
                K[141] += -sgL[5] + sg[5]
                K[144] += -phix*sg[3]
                K[145] += -phix*sg[5]
                K[146] += phix*sg[5]/radius
                K[148] += -sg[3]*w_x - sg[5]*w_y + sg[5]*v/radius
                K[154] += -sgL[5] + sg[5]
                K[155] += -sgL[4] + sg[4]
                K[158] += -phix*sg[5]
                K[159] += -phix*sg[4]
                K[160] += phix*sg[4]/radius
                K[162] += -sg[4]*w_y - sg[5]*w_x + sg[4]*v/radius
                K[170] += -sgL[3] + sg[3]
                K[171] += -sgL[5] + sg[5]
                K[172] += -phiy*sg[3] - sg[3]*v/radius
                K[173] += -phiy*sg[5] - sg[5]*v/radius
                K[174] += phiy*sg[5]/radius - sg[3]*w_x/radius - sg[5]*w_y/radius + 2*sg[5]*v/(radius*radius)
                K[175] += -sgL[5]/radius + sg[5]/radius
                K[177] += -sg[3]*w_x - sg[5]*w_y + sg[5]*v/radius
                K[184] += -sgL[5] + sg[5]
                K[185] += -sgL[4] + sg[4]
                K[186] += -phiy*sg[5] - sg[5]*v/radius
                K[187] += -phiy*sg[4] - sg[4]*v/radius
                K[188] += phiy*sg[4]/radius - sg[4]*w_y/radius - sg[5]*w_x/radius + 2*sg[4]*v/(radius*radius)
                K[189] += -sgL[4]/radius + sg[4]/radius
                K[191] += -sg[4]*w_y - sg[5]*w_x + sg[4]*v/radius

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
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAphix*gAphix
                        pA[9] = fAphiy*gAphiy
                        pA[10] = fAphix_x*gAphix
                        pA[11] = fAphix*gAphix_y
                        pA[12] = fAphiy_x*gAphiy
                        pA[13] = fAphiy*gAphiy_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[14]
                        TA[1] = pA[0]*K[1] + pA[1]*K[15]
                        TA[2] = pA[0]*K[2] + pA[1]*K[16]
                        TA[3] = pA[0]*K[3] + pA[1]*K[17]
                        TA[4] = pA[0]*K[4] + pA[1]*K[18]
                        TA[5] = pA[0]*K[5] + pA[1]*K[19]
                        TA[6] = pA[0]*K[6] + pA[1]*K[20]
                        TA[7] = pA[0]*K[7] + pA[1]*K[21]
                        TA[8] = pA[0]*K[8] + pA[1]*K[22]
                        TA[9] = pA[0]*K[9] + pA[1]*K[23]
                        TA[10] = pA[0]*K[10] + pA[1]*K[24]
                        TA[11] = pA[0]*K[11] + pA[1]*K[25]
                        TA[12] = pA[0]*K[12] + pA[1]*K[26]
                        TA[13] = pA[0]*K[13] + pA[1]*K[27]
                        TA[14] = pA[2]*K[28] + pA[3]*K[42] + pA[6]*K[84]
                        TA[15] = pA[2]*K[29] + pA[3]*K[43] + pA[6]*K[85]
                        TA[16] = pA[2]*K[30] + pA[3]*K[44] + pA[6]*K[86]
                        TA[17] = pA[2]*K[31] + pA[3]*K[45] + pA[6]*K[87]
                        TA[18] = pA[2]*K[32] + pA[3]*K[46] + pA[6]*K[88]
                        TA[19] = pA[2]*K[33] + pA[3]*K[47] + pA[6]*K[89]
                        TA[20] = pA[2]*K[34] + pA[3]*K[48] + pA[6]*K[90]
                        TA[21] = pA[2]*K[35] + pA[3]*K[49] + pA[6]*K[91]
                        TA[22] = pA[2]*K[36] + pA[3]*K[50] + pA[6]*K[92]
                        TA[23] = pA[2]*K[37] + pA[3]*K[51] + pA[6]*K[93]
                        TA[24] = pA[2]*K[38] + pA[3]*K[52] + pA[6]*K[94]
                        TA[25] = pA[2]*K[39] + pA[3]*K[53] + pA[6]*K[95]
                        TA[26] = pA[2]*K[40] + pA[3]*K[54] + pA[6]*K[96]
                        TA[27] = pA[2]*K[41] + pA[3]*K[55] + pA[6]*K[97]
                        TA[28] = pA[4]*K[56] + pA[5]*K[70] + pA[7]*K[98]
                        TA[29] = pA[4]*K[57] + pA[5]*K[71] + pA[7]*K[99]
                        TA[30] = pA[4]*K[58] + pA[5]*K[72] + pA[7]*K[100]
                        TA[31] = pA[4]*K[59] + pA[5]*K[73] + pA[7]*K[101]
                        TA[32] = pA[4]*K[60] + pA[5]*K[74] + pA[7]*K[102]
                        TA[33] = pA[4]*K[61] + pA[5]*K[75] + pA[7]*K[103]
                        TA[34] = pA[4]*K[62] + pA[5]*K[76] + pA[7]*K[104]
                        TA[35] = pA[4]*K[63] + pA[5]*K[77] + pA[7]*K[105]
                        TA[36] = pA[4]*K[64] + pA[5]*K[78] + pA[7]*K[106]
                        TA[37] = pA[4]*K[65] + pA[5]*K[79] + pA[7]*K[107]
                        TA[38] = pA[4]*K[66] + pA[5]*K[80] + pA[7]*K[108]
                        TA[39] = pA[4]*K[67] + pA[5]*K[81] + pA[7]*K[109]
                        TA[40] = pA[4]*K[68] + pA[5]*K[82] + pA[7]*K[110]
                        TA[41] = pA[4]*K[69] + pA[5]*K[83] + pA[7]*K[111]
                        TA[42] = pA[8]*K[112] + pA[10]*K[140] + pA[11]*K[154]
                        TA[43] = pA[8]*K[113] + pA[10]*K[141] + pA[11]*K[155]
                        TA[44] = pA[8]*K[114] + pA[10]*K[142] + pA[11]*K[156]
                        TA[45] = pA[8]*K[115] + pA[10]*K[143] + pA[11]*K[157]
                        TA[46] = pA[8]*K[116] + pA[10]*K[144] + pA[11]*K[158]
                        TA[47] = pA[8]*K[117] + pA[10]*K[145] + pA[11]*K[159]
                        TA[48] = pA[8]*K[118] + pA[10]*K[146] + pA[11]*K[160]
                        TA[49] = pA[8]*K[119] + pA[10]*K[147] + pA[11]*K[161]
                        TA[50] = pA[8]*K[120] + pA[10]*K[148] + pA[11]*K[162]
                        TA[51] = pA[8]*K[121] + pA[10]*K[149] + pA[11]*K[163]
                        TA[52] = pA[8]*K[122] + pA[10]*K[150] + pA[11]*K[164]
                        TA[53] = pA[8]*K[123] + pA[10]*K[151] + pA[11]*K[165]
                        TA[54] = pA[8]*K[124] + pA[10]*K[152] + pA[11]*K[166]
                        TA[55] = pA[8]*K[125] + pA[10]*K[153] + pA[11]*K[167]
                        TA[56] = pA[9]*K[126] + pA[12]*K[168] + pA[13]*K[182]
                        TA[57] = pA[9]*K[127] + pA[12]*K[169] + pA[13]*K[183]
                        TA[58] = pA[9]*K[128] + pA[12]*K[170] + pA[13]*K[184]
                        TA[59] = pA[9]*K[129] + pA[12]*K[171] + pA[13]*K[185]
                        TA[60] = pA[9]*K[130] + pA[12]*K[172] + pA[13]*K[186]
                        TA[61] = pA[9]*K[131] + pA[12]*K[173] + pA[13]*K[187]
                        TA[62] = pA[9]*K[132] + pA[12]*K[174] + pA[13]*K[188]
                        TA[63] = pA[9]*K[133] + pA[12]*K[175] + pA[13]*K[189]
                        TA[64] = pA[9]*K[134] + pA[12]*K[176] + pA[13]*K[190]
                        TA[65] = pA[9]*K[135] + pA[12]*K[177] + pA[13]*K[191]
                        TA[66] = pA[9]*K[136] + pA[12]*K[178] + pA[13]*K[192]
                        TA[67] = pA[9]*K[137] + pA[12]*K[179] + pA[13]*K[193]
                        TA[68] = pA[9]*K[138] + pA[12]*K[180] + pA[13]*K[194]
                        TA[69] = pA[9]*K[139] + pA[12]*K[181] + pA[13]*K[195]
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
                                pB[6] = fBv*gBv
                                pB[7] = fBw*gBw
                                pB[8] = fBphix*gBphix
                                pB[9] = fBphiy*gBphiy
                                pB[10] = fBphix_x*gBphix
                                pB[11] = fBphix*gBphix_y
                                pB[12] = fBphiy_x*gBphiy
                                pB[13] = fBphiy*gBphiy_y
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[2]*pB[2] + TA[3]*pB[3] + TA[6]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[4]*pB[4] + TA[5]*pB[5] + TA[7]*pB[7])
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
                                kCv[c] += weight*(TA[14]*pB[0] + TA[15]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[16]*pB[2] + TA[17]*pB[3] + TA[20]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[18]*pB[4] + TA[19]*pB[5] + TA[21]*pB[7])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[22]*pB[8] + TA[24]*pB[10] + TA[25]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[23]*pB[9] + TA[26]*pB[12] + TA[27]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[28]*pB[0] + TA[29]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[30]*pB[2] + TA[31]*pB[3] + TA[34]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[32]*pB[4] + TA[33]*pB[5] + TA[35]*pB[7])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[36]*pB[8] + TA[38]*pB[10] + TA[39]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[37]*pB[9] + TA[40]*pB[12] + TA[41]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[42]*pB[0] + TA[43]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[44]*pB[2] + TA[45]*pB[3] + TA[48]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[46]*pB[4] + TA[47]*pB[5] + TA[49]*pB[7])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[50]*pB[8] + TA[52]*pB[10] + TA[53]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[51]*pB[9] + TA[54]*pB[12] + TA[55]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[56]*pB[0] + TA[57]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[58]*pB[2] + TA[59]*pB[3] + TA[62]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[60]*pB[4] + TA[61]*pB[5] + TA[63]*pB[7])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[64]*pB[8] + TA[66]*pB[10] + TA[67]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[65]*pB[9] + TA[68]*pB[12] + TA[69]*pB[13])

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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the values of `v` and `w`, the rotations and their gradients), the strains
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
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, w, phix
    cdef double phiy, phix_x, phix_y, phiy_x, phiy_y
    cdef double e[8]
    cdef double sg[8]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAphix
    cdef double fAphix_x, fAphiy, fAphiy_x, gAu, gAu_y, gAv, gAv_y
    cdef double gAw, gAw_y, gAphix, gAphix_y, gAphiy, gAphiy_y, fBu
    cdef double fBu_x, fBv, fBv_x, fBw, fBw_x, fBphix, fBphix_x
    cdef double fBphiy, fBphiy_x, gBu, gBu_y, gBv, gBv_y, gBw
    cdef double gBw_y, gBphix, gBphix_y, gBphiy, gBphiy_y
    cdef double K[196]
    cdef double TA[70]
    cdef double pA[14]
    cdef double pB[14]

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

    ftab = np.zeros((10, m), dtype=DOUBLE)
    gtab = np.zeros((10, n), dtype=DOUBLE)

    fdim = 13*m*m*n*n
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
                w = 0.
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
                        w += fAw*gAw*cs[col+2]
                        phix += fAphix*gAphix*cs[col+3]
                        phiy += fAphiy*gAphiy*cs[col+4]
                        phix_x += fAphix_x*gAphix*cs[col+3]
                        phix_y += fAphix*gAphix_y*cs[col+3]
                        phiy_x += fAphiy_x*gAphiy*cs[col+4]
                        phiy_y += fAphiy*gAphiy_y*cs[col+4]

                # linear strains and stresses
                e[0] = u_x
                e[1] = v_y + w/radius
                e[2] = u_y + v_x
                e[3] = phix_x
                e[4] = phiy_y + v_y/radius
                e[5] = phix_y + phiy_x - 0.5*u_y/radius + 1.5*v_x/radius
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
                for p in range(196):
                    K[p] = 0.
                K[0] = sg[0]
                K[1] = sg[2] - 0.5*sg[5]/radius
                K[8] = sg[7]
                K[10] = sg[3]
                K[11] = sg[5]
                K[14] = sg[2] - 0.5*sg[5]/radius
                K[15] = sg[1] - sg[4]/radius
                K[22] = sg[6]
                K[24] = sg[5]
                K[25] = sg[4]
                K[30] = sg[0] + 2*sg[3]/radius
                K[31] = sg[2] + 1.5*sg[5]/radius
                K[34] = sg[7]/radius
                K[35] = sg[2]/radius + 0.5*sg[5]/(radius*radius)
                K[37] = sg[7]
                K[40] = sg[3]
                K[41] = sg[5]
                K[44] = sg[2] + 1.5*sg[5]/radius
                K[45] = sg[1] + sg[4]/radius
                K[48] = sg[6]/radius
                K[49] = sg[1]/radius
                K[51] = sg[6]
                K[54] = sg[5]
                K[55] = sg[4]
                K[60] = sg[0]
                K[61] = sg[2] - 0.5*sg[5]/radius
                K[62] = -sg[2]/radius - 0.5*sg[5]/(radius*radius)
                K[65] = -sg[5]/radius
                K[74] = sg[2] - 0.5*sg[5]/radius
                K[75] = sg[1] - sg[4]/radius
                K[76] = -sg[1]/radius
                K[79] = -sg[4]/radius
                K[86] = sg[7]/radius
                K[87] = sg[6]/radius
                K[88] = -sg[2]/radius - 0.5*sg[5]/(radius*radius)
                K[89] = -sg[1]/radius
                K[90] = sg[1]/(radius*radius)
                K[91] = sg[6]/(radius*radius)
                K[100] = sg[2]/radius + 0.5*sg[5]/(radius*radius)
                K[101] = sg[1]/radius
                K[104] = sg[6]/(radius*radius)
                K[105] = sg[1]/(radius*radius) - sg[4]/(radius*radius*radius)
                K[107] = sg[6]/radius
                K[110] = sg[5]/radius
                K[111] = sg[4]/radius
                K[112] = sg[7]
                K[113] = sg[6]
                K[120] = -sg[4]/radius
                K[128] = sg[7]
                K[129] = sg[6]
                K[130] = -sg[5]/radius
                K[131] = -sg[4]/radius
                K[133] = sg[6]/radius
                K[135] = -sg[4]/radius
                K[140] = sg[3]
                K[141] = sg[5]
                K[154] = sg[5]
                K[155] = sg[4]
                K[170] = sg[3]
                K[171] = sg[5]
                K[175] = sg[5]/radius
                K[184] = sg[5]
                K[185] = sg[4]
                K[189] = sg[4]/radius

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
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAphix*gAphix
                        pA[9] = fAphiy*gAphiy
                        pA[10] = fAphix_x*gAphix
                        pA[11] = fAphix*gAphix_y
                        pA[12] = fAphiy_x*gAphiy
                        pA[13] = fAphiy*gAphiy_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[14]
                        TA[1] = pA[0]*K[1] + pA[1]*K[15]
                        TA[8] = pA[0]*K[8] + pA[1]*K[22]
                        TA[10] = pA[0]*K[10] + pA[1]*K[24]
                        TA[11] = pA[0]*K[11] + pA[1]*K[25]
                        TA[16] = pA[2]*K[30] + pA[3]*K[44] + pA[6]*K[86]
                        TA[17] = pA[2]*K[31] + pA[3]*K[45] + pA[6]*K[87]
                        TA[18] = pA[6]*K[88]
                        TA[19] = pA[6]*K[89]
                        TA[20] = pA[2]*K[34] + pA[3]*K[48] + pA[6]*K[90]
                        TA[21] = pA[2]*K[35] + pA[3]*K[49] + pA[6]*K[91]
                        TA[23] = pA[2]*K[37] + pA[3]*K[51]
                        TA[26] = pA[2]*K[40] + pA[3]*K[54]
                        TA[27] = pA[2]*K[41] + pA[3]*K[55]
                        TA[30] = pA[7]*K[100]
                        TA[31] = pA[7]*K[101]
                        TA[32] = pA[4]*K[60] + pA[5]*K[74]
                        TA[33] = pA[4]*K[61] + pA[5]*K[75]
                        TA[34] = pA[4]*K[62] + pA[5]*K[76] + pA[7]*K[104]
                        TA[35] = pA[7]*K[105]
                        TA[37] = pA[4]*K[65] + pA[5]*K[79] + pA[7]*K[107]
                        TA[40] = pA[7]*K[110]
                        TA[41] = pA[7]*K[111]
                        TA[42] = pA[8]*K[112] + pA[10]*K[140] + pA[11]*K[154]
                        TA[43] = pA[8]*K[113] + pA[10]*K[141] + pA[11]*K[155]
                        TA[50] = pA[8]*K[120]
                        TA[58] = pA[9]*K[128] + pA[12]*K[170] + pA[13]*K[184]
                        TA[59] = pA[9]*K[129] + pA[12]*K[171] + pA[13]*K[185]
                        TA[60] = pA[9]*K[130]
                        TA[61] = pA[9]*K[131]
                        TA[63] = pA[9]*K[133] + pA[12]*K[175] + pA[13]*K[189]
                        TA[65] = pA[9]*K[135]
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
                                pB[6] = fBv*gBv
                                pB[7] = fBw*gBw
                                pB[8] = fBphix*gBphix
                                pB[9] = fBphiy*gBphiy
                                pB[10] = fBphix_x*gBphix
                                pB[11] = fBphix*gBphix_y
                                pB[12] = fBphiy_x*gBphiy
                                pB[13] = fBphiy*gBphiy_y
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+3
                                kGv[c] += weight*(TA[8]*pB[8] + TA[10]*pB[10] + TA[11]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[16]*pB[2] + TA[17]*pB[3] + TA[20]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[18]*pB[4] + TA[19]*pB[5] + TA[21]*pB[7])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+4
                                kGv[c] += weight*(TA[23]*pB[9] + TA[26]*pB[12] + TA[27]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[30]*pB[2] + TA[31]*pB[3] + TA[34]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[32]*pB[4] + TA[33]*pB[5] + TA[35]*pB[7])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+4
                                kGv[c] += weight*(TA[37]*pB[9] + TA[40]*pB[12] + TA[41]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+3
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[42]*pB[0] + TA[43]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+3
                                    kGc[c] = col+3
                                kGv[c] += weight*(TA[50]*pB[8])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+4
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[58]*pB[2] + TA[59]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+4
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[60]*pB[4] + TA[61]*pB[5] + TA[63]*pB[7])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+4
                                    kGc[c] = col+4
                                kGv[c] += weight*(TA[65]*pB[9])

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
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w
    cdef double phix, phiy, phix_x, phix_y, phiy_x, phiy_y
    cdef double e[8]
    cdef double sg[8]
    cdef double S[14]
    cdef double pA[14]
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
                v = 0.
                w = 0.
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
                        v += fAv*gAv*cs[col+1]
                        w += fAw*gAw*cs[col+2]
                        phix += fAphix*gAphix*cs[col+3]
                        phiy += fAphiy*gAphiy*cs[col+4]
                        phix_x += fAphix_x*gAphix*cs[col+3]
                        phix_y += fAphix*gAphix_y*cs[col+3]
                        phiy_x += fAphiy_x*gAphiy*cs[col+4]
                        phiy_y += fAphiy*gAphiy_y*cs[col+4]

                # strains, stresses and S = [B]^T sigma
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x - phiy*v_x*w_x/radius + phiy_x*v_x - phiy_x*v*w_x/radius + (v_x*v_x)/radius - v*v_x*w_x/(radius*radius)
                e[4] = -0.5*(phix*phix)*v_y/radius - 0.5*(phix*phix)/radius - 0.5*(phix*phix)*w/(radius*radius) - phix*phix_y*w_y + phix*phix_y*v/radius + phix_y*u_y - 0.5*(phiy*phiy)*v_y/radius - 0.5*(phiy*phiy)/radius - 0.5*(phiy*phiy)*w/(radius*radius) - phiy*phiy_y*w_y + phiy*phiy_y*v/radius - phiy*v_y*w_y/radius - phiy*w_y/radius - phiy*v*w/(radius*radius*radius) + phiy_y*v_y + phiy_y - phiy_y*v*w_y/radius + phiy_y*w/radius + phiy_y*(v*v)/(radius*radius) - 0.5*(u_y*u_y)/radius + 0.5*(v_y*v_y)/radius + v_y/radius - 0.5*(w_y*w_y)/radius - v*v_y*w_y/(radius*radius) + 0.5*(v*v)*v_y/(radius*radius*radius) - 0.5*(w*w)/(radius*radius*radius) - 0.5*(v*v)*w/(radius*radius*radius*radius)
                e[5] = -0.5*(phix*phix)*v_x/radius - phix*phix_x*w_y + phix*phix_x*v/radius - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - 0.5*(phiy*phiy)*v_x/radius - phiy*phiy_x*w_y + phiy*phiy_x*v/radius - phiy*phiy_y*w_x - phiy*v_x*w_y/radius - phiy*v_y*w_x/radius - phiy*w_x/radius + phiy_x*v_y + phiy_x - phiy_x*v*w_y/radius + phiy_x*w/radius + phiy_x*(v*v)/(radius*radius) + phiy_y*v_x - phiy_y*v*w_x/radius - 0.5*u_x*u_y/radius - 0.5*u_y/radius + 1.5*v_x*v_y/radius + 1.5*v_x/radius - 0.5*w_x*w_y/radius - v*v_x*w_y/(radius*radius) - v*v_y*w_x/(radius*radius) - 0.5*v*w_x/(radius*radius) + 0.5*v_x*w/(radius*radius) + 0.5*(v*v)*v_x/(radius*radius*radius)
                e[6] = -0.5*(phix*phix)*w_y + 0.5*(phix*phix)*v/radius + phix*u_y - 0.5*(phiy*phiy)*w_y + 0.5*(phiy*phiy)*v/radius + phiy*v_y + phiy - phiy*v*w_y/radius + phiy*w/radius + phiy*(v*v)/(radius*radius) + w_y + v*v_y/radius - 0.5*(v*v)*w_y/(radius*radius) + v*w/(radius*radius) + 0.5*(v*v*v)/(radius*radius*radius)
                e[7] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x - phiy*v*w_x/radius + w_x + v*v_x/radius - 0.5*(v*v)*w_x/(radius*radius)
                for r in range(NE):
                    sg[r] = 0.
                    for s in range(NE):
                        sg[r] += F[r*NE + s]*e[s]
                S[0] = phix*sg[7] + phix_x*sg[3] + phix_y*sg[5] + sg[0]*u_x + sg[0] + sg[2]*u_y - 0.5*sg[5]*u_y/radius
                S[1] = phix*sg[6] + phix_x*sg[5] + phix_y*sg[4] + sg[1]*u_y + sg[2]*u_x + sg[2] - sg[4]*u_y/radius - 0.5*sg[5]*u_x/radius - 0.5*sg[5]/radius
                S[2] = -0.5*(phix*phix)*sg[5]/radius - 0.5*(phiy*phiy)*sg[5]/radius + phiy*sg[7] - phiy*sg[3]*w_x/radius - phiy*sg[5]*w_y/radius + phiy_x*sg[3] + phiy_y*sg[5] + sg[0]*v_x + sg[2]*v_y + sg[2] + sg[2]*w/radius + 2*sg[3]*v_x/radius + 1.5*sg[5]*v_y/radius + 1.5*sg[5]/radius + sg[7]*v/radius - sg[3]*v*w_x/(radius*radius) - sg[5]*v*w_y/(radius*radius) + 0.5*sg[5]*w/(radius*radius) + 0.5*sg[5]*(v*v)/(radius*radius*radius)
                S[3] = -0.5*(phix*phix)*sg[4]/radius - 0.5*(phiy*phiy)*sg[4]/radius + phiy*sg[6] - phiy*sg[4]*w_y/radius - phiy*sg[5]*w_x/radius + phiy_x*sg[5] + phiy_y*sg[4] + sg[1]*v_y + sg[1] + sg[2]*v_x + sg[1]*w/radius + sg[4]*v_y/radius + sg[4]/radius + 1.5*sg[5]*v_x/radius + sg[6]*v/radius - sg[4]*v*w_y/(radius*radius) - sg[5]*v*w_x/(radius*radius) + 0.5*sg[4]*(v*v)/(radius*radius*radius)
                S[4] = -0.5*(phix*phix)*sg[7] - phix*phix_x*sg[3] - phix*phix_y*sg[5] - 0.5*(phiy*phiy)*sg[7] - phiy*phiy_x*sg[3] - phiy*phiy_y*sg[5] - phiy*sg[3]*v_x/radius - phiy*sg[5]*v_y/radius - phiy*sg[5]/radius - phiy*sg[7]*v/radius - phiy_x*sg[3]*v/radius - phiy_y*sg[5]*v/radius + sg[0]*w_x + sg[2]*w_y + sg[7] - sg[2]*v/radius - 0.5*sg[5]*w_y/radius - sg[3]*v*v_x/(radius*radius) - sg[5]*v*v_y/(radius*radius) - 0.5*sg[5]*v/(radius*radius) - 0.5*sg[7]*(v*v)/(radius*radius)
                S[5] = -0.5*(phix*phix)*sg[6] - phix*phix_x*sg[5] - phix*phix_y*sg[4] - 0.5*(phiy*phiy)*sg[6] - phiy*phiy_x*sg[5] - phiy*phiy_y*sg[4] - phiy*sg[4]*v_y/radius - phiy*sg[4]/radius - phiy*sg[5]*v_x/radius - phiy*sg[6]*v/radius - phiy_x*sg[5]*v/radius - phiy_y*sg[4]*v/radius + sg[1]*w_y + sg[2]*w_x + sg[6] - sg[1]*v/radius - sg[4]*w_y/radius - 0.5*sg[5]*w_x/radius - sg[4]*v*v_y/(radius*radius) - sg[5]*v*v_x/(radius*radius) - 0.5*sg[6]*(v*v)/(radius*radius)
                S[6] = 0.5*(phix*phix)*sg[6]/radius + phix*phix_x*sg[5]/radius + phix*phix_y*sg[4]/radius + 0.5*(phiy*phiy)*sg[6]/radius + phiy*phiy_x*sg[5]/radius + phiy*phiy_y*sg[4]/radius - phiy*sg[6]*w_y/radius - phiy*sg[7]*w_x/radius + 2*phiy*sg[6]*v/(radius*radius) - phiy*sg[4]*w/(radius*radius*radius) - phiy_x*sg[3]*w_x/radius - phiy_x*sg[5]*w_y/radius + 2*phiy_x*sg[5]*v/(radius*radius) - phiy_y*sg[4]*w_y/radius - phiy_y*sg[5]*w_x/radius + 2*phiy_y*sg[4]*v/(radius*radius) - sg[1]*w_y/radius - sg[2]*w_x/radius + sg[6]*v_y/radius + sg[7]*v_x/radius + sg[1]*v/(radius*radius) - sg[3]*v_x*w_x/(radius*radius) - sg[4]*v_y*w_y/(radius*radius) - sg[5]*v_x*w_y/(radius*radius) - sg[5]*v_y*w_x/(radius*radius) - 0.5*sg[5]*w_x/(radius*radius) - sg[6]*v*w_y/(radius*radius) + sg[6]*w/(radius*radius) - sg[7]*v*w_x/(radius*radius) + sg[4]*v*v_y/(radius*radius*radius) + sg[5]*v*v_x/(radius*radius*radius) + 1.5*sg[6]*(v*v)/(radius*radius*radius) - sg[4]*v*w/(radius*radius*radius*radius)
                S[7] = -0.5*(phix*phix)*sg[4]/(radius*radius) - 0.5*(phiy*phiy)*sg[4]/(radius*radius) + phiy*sg[6]/radius - phiy*sg[4]*v/(radius*radius*radius) + phiy_x*sg[5]/radius + phiy_y*sg[4]/radius + sg[1]*v_y/radius + sg[1]/radius + sg[2]*v_x/radius + sg[1]*w/(radius*radius) + 0.5*sg[5]*v_x/(radius*radius) + sg[6]*v/(radius*radius) - sg[4]*w/(radius*radius*radius) - 0.5*sg[4]*(v*v)/(radius*radius*radius*radius)
                S[8] = -phix*sg[6]*w_y - phix*sg[7]*w_x - phix*sg[4]*v_y/radius - phix*sg[4]/radius - phix*sg[5]*v_x/radius + phix*sg[6]*v/radius - phix*sg[4]*w/(radius*radius) - phix_x*sg[3]*w_x - phix_x*sg[5]*w_y + phix_x*sg[5]*v/radius - phix_y*sg[4]*w_y - phix_y*sg[5]*w_x + phix_y*sg[4]*v/radius + sg[6]*u_y + sg[7]*u_x + sg[7]
                S[9] = -phiy*sg[6]*w_y - phiy*sg[7]*w_x - phiy*sg[4]*v_y/radius - phiy*sg[4]/radius - phiy*sg[5]*v_x/radius + phiy*sg[6]*v/radius - phiy*sg[4]*w/(radius*radius) - phiy_x*sg[3]*w_x - phiy_x*sg[5]*w_y + phiy_x*sg[5]*v/radius - phiy_y*sg[4]*w_y - phiy_y*sg[5]*w_x + phiy_y*sg[4]*v/radius + sg[6]*v_y + sg[6] + sg[7]*v_x - sg[3]*v_x*w_x/radius - sg[4]*v_y*w_y/radius - sg[4]*w_y/radius - sg[5]*v_x*w_y/radius - sg[5]*v_y*w_x/radius - sg[5]*w_x/radius - sg[6]*v*w_y/radius + sg[6]*w/radius - sg[7]*v*w_x/radius + sg[6]*(v*v)/(radius*radius) - sg[4]*v*w/(radius*radius*radius)
                S[10] = -phix*sg[3]*w_x - phix*sg[5]*w_y + phix*sg[5]*v/radius + sg[3]*u_x + sg[3] + sg[5]*u_y
                S[11] = -phix*sg[4]*w_y - phix*sg[5]*w_x + phix*sg[4]*v/radius + sg[4]*u_y + sg[5]*u_x + sg[5]
                S[12] = -phiy*sg[3]*w_x - phiy*sg[5]*w_y + phiy*sg[5]*v/radius + sg[3]*v_x + sg[5]*v_y + sg[5] - sg[3]*v*w_x/radius - sg[5]*v*w_y/radius + sg[5]*w/radius + sg[5]*(v*v)/(radius*radius)
                S[13] = -phiy*sg[4]*w_y - phiy*sg[5]*w_x + phiy*sg[4]*v/radius + sg[4]*v_y + sg[4] + sg[5]*v_x - sg[4]*v*w_y/radius + sg[4]*w/radius - sg[5]*v*w_x/radius + sg[4]*(v*v)/(radius*radius)

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
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAphix*gAphix
                        pA[9] = fAphiy*gAphiy
                        pA[10] = fAphix_x*gAphix
                        pA[11] = fAphix*gAphix_y
                        pA[12] = fAphiy_x*gAphiy
                        pA[13] = fAphiy*gAphiy_y
                        fint[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        fint[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[6]*S[6])
                        fint[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[7]*S[7])
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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the values of `v` and `w`, the rotations and their gradients), the strains
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
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w
    cdef double phix, phiy, phix_x, phix_y, phiy_x, phiy_y
    cdef double du_x, du_y, dv_x, dv_y, dw_x, dw_y, dv, dw
    cdef double dphix, dphiy, dphix_x, dphix_y, dphiy_x, dphiy_y
    cdef double e[8]
    cdef double de[8]
    cdef double sg[8]
    cdef double dsg[8]
    cdef double S[14]
    cdef double pA[14]
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
                v = 0.
                w = 0.
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
                dv = 0.
                dw = 0.
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
                        v += fAv*gAv*cs[col+1]
                        w += fAw*gAw*cs[col+2]
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
                        dv += fAv*gAv*vs[col+1]
                        dw += fAw*gAw*vs[col+2]
                        dphix += fAphix*gAphix*vs[col+3]
                        dphiy += fAphiy*gAphiy*vs[col+4]
                        dphix_x += fAphix_x*gAphix*vs[col+3]
                        dphix_y += fAphix*gAphix_y*vs[col+3]
                        dphiy_x += fAphiy_x*gAphiy*vs[col+4]
                        dphiy_y += fAphiy*gAphiy_y*vs[col+4]

                # strains, strain increments, stresses
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
                e[3] = -phix*phix_x*w_x + phix_x*u_x + phix_x - phiy*phiy_x*w_x - phiy*v_x*w_x/radius + phiy_x*v_x - phiy_x*v*w_x/radius + (v_x*v_x)/radius - v*v_x*w_x/(radius*radius)
                e[4] = -0.5*(phix*phix)*v_y/radius - 0.5*(phix*phix)/radius - 0.5*(phix*phix)*w/(radius*radius) - phix*phix_y*w_y + phix*phix_y*v/radius + phix_y*u_y - 0.5*(phiy*phiy)*v_y/radius - 0.5*(phiy*phiy)/radius - 0.5*(phiy*phiy)*w/(radius*radius) - phiy*phiy_y*w_y + phiy*phiy_y*v/radius - phiy*v_y*w_y/radius - phiy*w_y/radius - phiy*v*w/(radius*radius*radius) + phiy_y*v_y + phiy_y - phiy_y*v*w_y/radius + phiy_y*w/radius + phiy_y*(v*v)/(radius*radius) - 0.5*(u_y*u_y)/radius + 0.5*(v_y*v_y)/radius + v_y/radius - 0.5*(w_y*w_y)/radius - v*v_y*w_y/(radius*radius) + 0.5*(v*v)*v_y/(radius*radius*radius) - 0.5*(w*w)/(radius*radius*radius) - 0.5*(v*v)*w/(radius*radius*radius*radius)
                e[5] = -0.5*(phix*phix)*v_x/radius - phix*phix_x*w_y + phix*phix_x*v/radius - phix*phix_y*w_x + phix_x*u_y + phix_y*u_x + phix_y - 0.5*(phiy*phiy)*v_x/radius - phiy*phiy_x*w_y + phiy*phiy_x*v/radius - phiy*phiy_y*w_x - phiy*v_x*w_y/radius - phiy*v_y*w_x/radius - phiy*w_x/radius + phiy_x*v_y + phiy_x - phiy_x*v*w_y/radius + phiy_x*w/radius + phiy_x*(v*v)/(radius*radius) + phiy_y*v_x - phiy_y*v*w_x/radius - 0.5*u_x*u_y/radius - 0.5*u_y/radius + 1.5*v_x*v_y/radius + 1.5*v_x/radius - 0.5*w_x*w_y/radius - v*v_x*w_y/(radius*radius) - v*v_y*w_x/(radius*radius) - 0.5*v*w_x/(radius*radius) + 0.5*v_x*w/(radius*radius) + 0.5*(v*v)*v_x/(radius*radius*radius)
                e[6] = -0.5*(phix*phix)*w_y + 0.5*(phix*phix)*v/radius + phix*u_y - 0.5*(phiy*phiy)*w_y + 0.5*(phiy*phiy)*v/radius + phiy*v_y + phiy - phiy*v*w_y/radius + phiy*w/radius + phiy*(v*v)/(radius*radius) + w_y + v*v_y/radius - 0.5*(v*v)*w_y/(radius*radius) + v*w/(radius*radius) + 0.5*(v*v*v)/(radius*radius*radius)
                e[7] = -0.5*(phix*phix)*w_x + phix*u_x + phix - 0.5*(phiy*phiy)*w_x + phiy*v_x - phiy*v*w_x/radius + w_x + v*v_x/radius - 0.5*(v*v)*w_x/(radius*radius)
                de[0] = du_x*u_x + du_x + dv_x*v_x + dw_x*w_x
                de[1] = du_y*u_y - dv*w_y/radius + dv*v/(radius*radius) + dv_y*v_y + dv_y + dv_y*w/radius + dw*v_y/radius + dw/radius + dw*w/(radius*radius) + dw_y*w_y - dw_y*v/radius
                de[2] = du_x*u_y + du_y*u_x + du_y - dv*w_x/radius + dv_x*v_y + dv_x + dv_x*w/radius + dv_y*v_x + dw*v_x/radius + dw_x*w_y - dw_x*v/radius + dw_y*w_x
                de[3] = -dphix*phix_x*w_x - dphix_x*phix*w_x + dphix_x*u_x + dphix_x - dphiy*phiy_x*w_x - dphiy*v_x*w_x/radius - dphiy_x*phiy*w_x + dphiy_x*v_x - dphiy_x*v*w_x/radius + du_x*phix_x - dv*phiy_x*w_x/radius - dv*v_x*w_x/(radius*radius) - dv_x*phiy*w_x/radius + dv_x*phiy_x + 2*dv_x*v_x/radius - dv_x*v*w_x/(radius*radius) - dw_x*phix*phix_x - dw_x*phiy*phiy_x - dw_x*phiy*v_x/radius - dw_x*phiy_x*v/radius - dw_x*v*v_x/(radius*radius)
                de[4] = -dphix*phix*v_y/radius - dphix*phix/radius - dphix*phix*w/(radius*radius) - dphix*phix_y*w_y + dphix*phix_y*v/radius - dphix_y*phix*w_y + dphix_y*phix*v/radius + dphix_y*u_y - dphiy*phiy*v_y/radius - dphiy*phiy/radius - dphiy*phiy*w/(radius*radius) - dphiy*phiy_y*w_y + dphiy*phiy_y*v/radius - dphiy*v_y*w_y/radius - dphiy*w_y/radius - dphiy*v*w/(radius*radius*radius) - dphiy_y*phiy*w_y + dphiy_y*phiy*v/radius + dphiy_y*v_y + dphiy_y - dphiy_y*v*w_y/radius + dphiy_y*w/radius + dphiy_y*(v*v)/(radius*radius) + du_y*phix_y - du_y*u_y/radius + dv*phix*phix_y/radius + dv*phiy*phiy_y/radius - dv*phiy*w/(radius*radius*radius) - dv*phiy_y*w_y/radius + 2*dv*phiy_y*v/(radius*radius) - dv*v_y*w_y/(radius*radius) + dv*v*v_y/(radius*radius*radius) - dv*v*w/(radius*radius*radius*radius) - 0.5*dv_y*(phix*phix)/radius - 0.5*dv_y*(phiy*phiy)/radius - dv_y*phiy*w_y/radius + dv_y*phiy_y + dv_y*v_y/radius + dv_y/radius - dv_y*v*w_y/(radius*radius) + 0.5*dv_y*(v*v)/(radius*radius*radius) - 0.5*dw*(phix*phix)/(radius*radius) - 0.5*dw*(phiy*phiy)/(radius*radius) - dw*phiy*v/(radius*radius*radius) + dw*phiy_y/radius - dw*w/(radius*radius*radius) - 0.5*dw*(v*v)/(radius*radius*radius*radius) - dw_y*phix*phix_y - dw_y*phiy*phiy_y - dw_y*phiy*v_y/radius - dw_y*phiy/radius - dw_y*phiy_y*v/radius - dw_y*w_y/radius - dw_y*v*v_y/(radius*radius)
                de[5] = -dphix*phix*v_x/radius - dphix*phix_x*w_y + dphix*phix_x*v/radius - dphix*phix_y*w_x - dphix_x*phix*w_y + dphix_x*phix*v/radius + dphix_x*u_y - dphix_y*phix*w_x + dphix_y*u_x + dphix_y - dphiy*phiy*v_x/radius - dphiy*phiy_x*w_y + dphiy*phiy_x*v/radius - dphiy*phiy_y*w_x - dphiy*v_x*w_y/radius - dphiy*v_y*w_x/radius - dphiy*w_x/radius - dphiy_x*phiy*w_y + dphiy_x*phiy*v/radius + dphiy_x*v_y + dphiy_x - dphiy_x*v*w_y/radius + dphiy_x*w/radius + dphiy_x*(v*v)/(radius*radius) - dphiy_y*phiy*w_x + dphiy_y*v_x - dphiy_y*v*w_x/radius + du_x*phix_y - 0.5*du_x*u_y/radius + du_y*phix_x - 0.5*du_y*u_x/radius - 0.5*du_y/radius + dv*phix*phix_x/radius + dv*phiy*phiy_x/radius - dv*phiy_x*w_y/radius + 2*dv*phiy_x*v/(radius*radius) - dv*phiy_y*w_x/radius - dv*v_x*w_y/(radius*radius) - dv*v_y*w_x/(radius*radius) - 0.5*dv*w_x/(radius*radius) + dv*v*v_x/(radius*radius*radius) - 0.5*dv_x*(phix*phix)/radius - 0.5*dv_x*(phiy*phiy)/radius - dv_x*phiy*w_y/radius + dv_x*phiy_y + 1.5*dv_x*v_y/radius + 1.5*dv_x/radius - dv_x*v*w_y/(radius*radius) + 0.5*dv_x*w/(radius*radius) + 0.5*dv_x*(v*v)/(radius*radius*radius) - dv_y*phiy*w_x/radius + dv_y*phiy_x + 1.5*dv_y*v_x/radius - dv_y*v*w_x/(radius*radius) + dw*phiy_x/radius + 0.5*dw*v_x/(radius*radius) - dw_x*phix*phix_y - dw_x*phiy*phiy_y - dw_x*phiy*v_y/radius - dw_x*phiy/radius - dw_x*phiy_y*v/radius - 0.5*dw_x*w_y/radius - dw_x*v*v_y/(radius*radius) - 0.5*dw_x*v/(radius*radius) - dw_y*phix*phix_x - dw_y*phiy*phiy_x - dw_y*phiy*v_x/radius - dw_y*phiy_x*v/radius - 0.5*dw_y*w_x/radius - dw_y*v*v_x/(radius*radius)
                de[6] = -dphix*phix*w_y + dphix*phix*v/radius + dphix*u_y - dphiy*phiy*w_y + dphiy*phiy*v/radius + dphiy*v_y + dphiy - dphiy*v*w_y/radius + dphiy*w/radius + dphiy*(v*v)/(radius*radius) + du_y*phix + 0.5*dv*(phix*phix)/radius + 0.5*dv*(phiy*phiy)/radius - dv*phiy*w_y/radius + 2*dv*phiy*v/(radius*radius) + dv*v_y/radius - dv*v*w_y/(radius*radius) + dv*w/(radius*radius) + 1.5*dv*(v*v)/(radius*radius*radius) + dv_y*phiy + dv_y*v/radius + dw*phiy/radius + dw*v/(radius*radius) - 0.5*dw_y*(phix*phix) - 0.5*dw_y*(phiy*phiy) - dw_y*phiy*v/radius + dw_y - 0.5*dw_y*(v*v)/(radius*radius)
                de[7] = -dphix*phix*w_x + dphix*u_x + dphix - dphiy*phiy*w_x + dphiy*v_x - dphiy*v*w_x/radius + du_x*phix - dv*phiy*w_x/radius + dv*v_x/radius - dv*v*w_x/(radius*radius) + dv_x*phiy + dv_x*v/radius - 0.5*dw_x*(phix*phix) - 0.5*dw_x*(phiy*phiy) - dw_x*phiy*v/radius + dw_x - 0.5*dw_x*(v*v)/(radius*radius)
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
                S[0] = dphix*sg[7] + dphix_x*sg[3] + dphix_y*sg[5] + dsg[0]*u_x + dsg[0] + dsg[2]*u_y + dsg[3]*phix_x + dsg[5]*phix_y - 0.5*dsg[5]*u_y/radius + dsg[7]*phix + du_x*sg[0] + du_y*sg[2] - 0.5*du_y*sg[5]/radius
                S[1] = dphix*sg[6] + dphix_x*sg[5] + dphix_y*sg[4] + dsg[1]*u_y + dsg[2]*u_x + dsg[2] + dsg[4]*phix_y - dsg[4]*u_y/radius + dsg[5]*phix_x - 0.5*dsg[5]*u_x/radius - 0.5*dsg[5]/radius + dsg[6]*phix + du_x*sg[2] - 0.5*du_x*sg[5]/radius + du_y*sg[1] - du_y*sg[4]/radius
                S[2] = -dphix*phix*sg[5]/radius - dphiy*phiy*sg[5]/radius + dphiy*sg[7] - dphiy*sg[3]*w_x/radius - dphiy*sg[5]*w_y/radius + dphiy_x*sg[3] + dphiy_y*sg[5] + dsg[0]*v_x + dsg[2]*v_y + dsg[2] + dsg[2]*w/radius - dsg[3]*phiy*w_x/radius + dsg[3]*phiy_x + 2*dsg[3]*v_x/radius - dsg[3]*v*w_x/(radius*radius) - 0.5*dsg[5]*(phix*phix)/radius - 0.5*dsg[5]*(phiy*phiy)/radius - dsg[5]*phiy*w_y/radius + dsg[5]*phiy_y + 1.5*dsg[5]*v_y/radius + 1.5*dsg[5]/radius - dsg[5]*v*w_y/(radius*radius) + 0.5*dsg[5]*w/(radius*radius) + 0.5*dsg[5]*(v*v)/(radius*radius*radius) + dsg[7]*phiy + dsg[7]*v/radius + dv*sg[7]/radius - dv*sg[3]*w_x/(radius*radius) - dv*sg[5]*w_y/(radius*radius) + dv*sg[5]*v/(radius*radius*radius) + dv_x*sg[0] + 2*dv_x*sg[3]/radius + dv_y*sg[2] + 1.5*dv_y*sg[5]/radius + dw*sg[2]/radius + 0.5*dw*sg[5]/(radius*radius) - dw_x*phiy*sg[3]/radius - dw_x*sg[3]*v/(radius*radius) - dw_y*phiy*sg[5]/radius - dw_y*sg[5]*v/(radius*radius)
                S[3] = -dphix*phix*sg[4]/radius - dphiy*phiy*sg[4]/radius + dphiy*sg[6] - dphiy*sg[4]*w_y/radius - dphiy*sg[5]*w_x/radius + dphiy_x*sg[5] + dphiy_y*sg[4] + dsg[1]*v_y + dsg[1] + dsg[1]*w/radius + dsg[2]*v_x - 0.5*dsg[4]*(phix*phix)/radius - 0.5*dsg[4]*(phiy*phiy)/radius - dsg[4]*phiy*w_y/radius + dsg[4]*phiy_y + dsg[4]*v_y/radius + dsg[4]/radius - dsg[4]*v*w_y/(radius*radius) + 0.5*dsg[4]*(v*v)/(radius*radius*radius) - dsg[5]*phiy*w_x/radius + dsg[5]*phiy_x + 1.5*dsg[5]*v_x/radius - dsg[5]*v*w_x/(radius*radius) + dsg[6]*phiy + dsg[6]*v/radius + dv*sg[6]/radius - dv*sg[4]*w_y/(radius*radius) - dv*sg[5]*w_x/(radius*radius) + dv*sg[4]*v/(radius*radius*radius) + dv_x*sg[2] + 1.5*dv_x*sg[5]/radius + dv_y*sg[1] + dv_y*sg[4]/radius + dw*sg[1]/radius - dw_x*phiy*sg[5]/radius - dw_x*sg[5]*v/(radius*radius) - dw_y*phiy*sg[4]/radius - dw_y*sg[4]*v/(radius*radius)
                S[4] = -dphix*phix*sg[7] - dphix*phix_x*sg[3] - dphix*phix_y*sg[5] - dphix_x*phix*sg[3] - dphix_y*phix*sg[5] - dphiy*phiy*sg[7] - dphiy*phiy_x*sg[3] - dphiy*phiy_y*sg[5] - dphiy*sg[3]*v_x/radius - dphiy*sg[5]*v_y/radius - dphiy*sg[5]/radius - dphiy*sg[7]*v/radius - dphiy_x*phiy*sg[3] - dphiy_x*sg[3]*v/radius - dphiy_y*phiy*sg[5] - dphiy_y*sg[5]*v/radius + dsg[0]*w_x + dsg[2]*w_y - dsg[2]*v/radius - dsg[3]*phix*phix_x - dsg[3]*phiy*phiy_x - dsg[3]*phiy*v_x/radius - dsg[3]*phiy_x*v/radius - dsg[3]*v*v_x/(radius*radius) - dsg[5]*phix*phix_y - dsg[5]*phiy*phiy_y - dsg[5]*phiy*v_y/radius - dsg[5]*phiy/radius - dsg[5]*phiy_y*v/radius - 0.5*dsg[5]*w_y/radius - dsg[5]*v*v_y/(radius*radius) - 0.5*dsg[5]*v/(radius*radius) - 0.5*dsg[7]*(phix*phix) - 0.5*dsg[7]*(phiy*phiy) - dsg[7]*phiy*v/radius + dsg[7] - 0.5*dsg[7]*(v*v)/(radius*radius) - dv*phiy*sg[7]/radius - dv*phiy_x*sg[3]/radius - dv*phiy_y*sg[5]/radius - dv*sg[2]/radius - dv*sg[3]*v_x/(radius*radius) - dv*sg[5]*v_y/(radius*radius) - 0.5*dv*sg[5]/(radius*radius) - dv*sg[7]*v/(radius*radius) - dv_x*phiy*sg[3]/radius - dv_x*sg[3]*v/(radius*radius) - dv_y*phiy*sg[5]/radius - dv_y*sg[5]*v/(radius*radius) + dw_x*sg[0] + dw_y*sg[2] - 0.5*dw_y*sg[5]/radius
                S[5] = -dphix*phix*sg[6] - dphix*phix_x*sg[5] - dphix*phix_y*sg[4] - dphix_x*phix*sg[5] - dphix_y*phix*sg[4] - dphiy*phiy*sg[6] - dphiy*phiy_x*sg[5] - dphiy*phiy_y*sg[4] - dphiy*sg[4]*v_y/radius - dphiy*sg[4]/radius - dphiy*sg[5]*v_x/radius - dphiy*sg[6]*v/radius - dphiy_x*phiy*sg[5] - dphiy_x*sg[5]*v/radius - dphiy_y*phiy*sg[4] - dphiy_y*sg[4]*v/radius + dsg[1]*w_y - dsg[1]*v/radius + dsg[2]*w_x - dsg[4]*phix*phix_y - dsg[4]*phiy*phiy_y - dsg[4]*phiy*v_y/radius - dsg[4]*phiy/radius - dsg[4]*phiy_y*v/radius - dsg[4]*w_y/radius - dsg[4]*v*v_y/(radius*radius) - dsg[5]*phix*phix_x - dsg[5]*phiy*phiy_x - dsg[5]*phiy*v_x/radius - dsg[5]*phiy_x*v/radius - 0.5*dsg[5]*w_x/radius - dsg[5]*v*v_x/(radius*radius) - 0.5*dsg[6]*(phix*phix) - 0.5*dsg[6]*(phiy*phiy) - dsg[6]*phiy*v/radius + dsg[6] - 0.5*dsg[6]*(v*v)/(radius*radius) - dv*phiy*sg[6]/radius - dv*phiy_x*sg[5]/radius - dv*phiy_y*sg[4]/radius - dv*sg[1]/radius - dv*sg[4]*v_y/(radius*radius) - dv*sg[5]*v_x/(radius*radius) - dv*sg[6]*v/(radius*radius) - dv_x*phiy*sg[5]/radius - dv_x*sg[5]*v/(radius*radius) - dv_y*phiy*sg[4]/radius - dv_y*sg[4]*v/(radius*radius) + dw_x*sg[2] - 0.5*dw_x*sg[5]/radius + dw_y*sg[1] - dw_y*sg[4]/radius
                S[6] = dphix*phix*sg[6]/radius + dphix*phix_x*sg[5]/radius + dphix*phix_y*sg[4]/radius + dphix_x*phix*sg[5]/radius + dphix_y*phix*sg[4]/radius + dphiy*phiy*sg[6]/radius + dphiy*phiy_x*sg[5]/radius + dphiy*phiy_y*sg[4]/radius - dphiy*sg[6]*w_y/radius - dphiy*sg[7]*w_x/radius + 2*dphiy*sg[6]*v/(radius*radius) - dphiy*sg[4]*w/(radius*radius*radius) + dphiy_x*phiy*sg[5]/radius - dphiy_x*sg[3]*w_x/radius - dphiy_x*sg[5]*w_y/radius + 2*dphiy_x*sg[5]*v/(radius*radius) + dphiy_y*phiy*sg[4]/radius - dphiy_y*sg[4]*w_y/radius - dphiy_y*sg[5]*w_x/radius + 2*dphiy_y*sg[4]*v/(radius*radius) - dsg[1]*w_y/radius + dsg[1]*v/(radius*radius) - dsg[2]*w_x/radius - dsg[3]*phiy_x*w_x/radius - dsg[3]*v_x*w_x/(radius*radius) + dsg[4]*phix*phix_y/radius + dsg[4]*phiy*phiy_y/radius - dsg[4]*phiy*w/(radius*radius*radius) - dsg[4]*phiy_y*w_y/radius + 2*dsg[4]*phiy_y*v/(radius*radius) - dsg[4]*v_y*w_y/(radius*radius) + dsg[4]*v*v_y/(radius*radius*radius) - dsg[4]*v*w/(radius*radius*radius*radius) + dsg[5]*phix*phix_x/radius + dsg[5]*phiy*phiy_x/radius - dsg[5]*phiy_x*w_y/radius + 2*dsg[5]*phiy_x*v/(radius*radius) - dsg[5]*phiy_y*w_x/radius - dsg[5]*v_x*w_y/(radius*radius) - dsg[5]*v_y*w_x/(radius*radius) - 0.5*dsg[5]*w_x/(radius*radius) + dsg[5]*v*v_x/(radius*radius*radius) + 0.5*dsg[6]*(phix*phix)/radius + 0.5*dsg[6]*(phiy*phiy)/radius - dsg[6]*phiy*w_y/radius + 2*dsg[6]*phiy*v/(radius*radius) + dsg[6]*v_y/radius - dsg[6]*v*w_y/(radius*radius) + dsg[6]*w/(radius*radius) + 1.5*dsg[6]*(v*v)/(radius*radius*radius) - dsg[7]*phiy*w_x/radius + dsg[7]*v_x/radius - dsg[7]*v*w_x/(radius*radius) + 2*dv*phiy*sg[6]/(radius*radius) + 2*dv*phiy_x*sg[5]/(radius*radius) + 2*dv*phiy_y*sg[4]/(radius*radius) + dv*sg[1]/(radius*radius) - dv*sg[6]*w_y/(radius*radius) - dv*sg[7]*w_x/(radius*radius) + dv*sg[4]*v_y/(radius*radius*radius) + dv*sg[5]*v_x/(radius*radius*radius) + 3*dv*sg[6]*v/(radius*radius*radius) - dv*sg[4]*w/(radius*radius*radius*radius) + dv_x*sg[7]/radius - dv_x*sg[3]*w_x/(radius*radius) - dv_x*sg[5]*w_y/(radius*radius) + dv_x*sg[5]*v/(radius*radius*radius) + dv_y*sg[6]/radius - dv_y*sg[4]*w_y/(radius*radius) - dv_y*sg[5]*w_x/(radius*radius) + dv_y*sg[4]*v/(radius*radius*radius) - dw*phiy*sg[4]/(radius*radius*radius) + dw*sg[6]/(radius*radius) - dw*sg[4]*v/(radius*radius*radius*radius) - dw_x*phiy*sg[7]/radius - dw_x*phiy_x*sg[3]/radius - dw_x*phiy_y*sg[5]/radius - dw_x*sg[2]/radius - dw_x*sg[3]*v_x/(radius*radius) - dw_x*sg[5]*v_y/(radius*radius) - 0.5*dw_x*sg[5]/(radius*radius) - dw_x*sg[7]*v/(radius*radius) - dw_y*phiy*sg[6]/radius - dw_y*phiy_x*sg[5]/radius - dw_y*phiy_y*sg[4]/radius - dw_y*sg[1]/radius - dw_y*sg[4]*v_y/(radius*radius) - dw_y*sg[5]*v_x/(radius*radius) - dw_y*sg[6]*v/(radius*radius)
                S[7] = -dphix*phix*sg[4]/(radius*radius) - dphiy*phiy*sg[4]/(radius*radius) + dphiy*sg[6]/radius - dphiy*sg[4]*v/(radius*radius*radius) + dphiy_x*sg[5]/radius + dphiy_y*sg[4]/radius + dsg[1]*v_y/radius + dsg[1]/radius + dsg[1]*w/(radius*radius) + dsg[2]*v_x/radius - 0.5*dsg[4]*(phix*phix)/(radius*radius) - 0.5*dsg[4]*(phiy*phiy)/(radius*radius) - dsg[4]*phiy*v/(radius*radius*radius) + dsg[4]*phiy_y/radius - dsg[4]*w/(radius*radius*radius) - 0.5*dsg[4]*(v*v)/(radius*radius*radius*radius) + dsg[5]*phiy_x/radius + 0.5*dsg[5]*v_x/(radius*radius) + dsg[6]*phiy/radius + dsg[6]*v/(radius*radius) - dv*phiy*sg[4]/(radius*radius*radius) + dv*sg[6]/(radius*radius) - dv*sg[4]*v/(radius*radius*radius*radius) + dv_x*sg[2]/radius + 0.5*dv_x*sg[5]/(radius*radius) + dv_y*sg[1]/radius + dw*sg[1]/(radius*radius) - dw*sg[4]/(radius*radius*radius)
                S[8] = -dphix*sg[6]*w_y - dphix*sg[7]*w_x - dphix*sg[4]*v_y/radius - dphix*sg[4]/radius - dphix*sg[5]*v_x/radius + dphix*sg[6]*v/radius - dphix*sg[4]*w/(radius*radius) - dphix_x*sg[3]*w_x - dphix_x*sg[5]*w_y + dphix_x*sg[5]*v/radius - dphix_y*sg[4]*w_y - dphix_y*sg[5]*w_x + dphix_y*sg[4]*v/radius - dsg[3]*phix_x*w_x - dsg[4]*phix*v_y/radius - dsg[4]*phix/radius - dsg[4]*phix*w/(radius*radius) - dsg[4]*phix_y*w_y + dsg[4]*phix_y*v/radius - dsg[5]*phix*v_x/radius - dsg[5]*phix_x*w_y + dsg[5]*phix_x*v/radius - dsg[5]*phix_y*w_x - dsg[6]*phix*w_y + dsg[6]*phix*v/radius + dsg[6]*u_y - dsg[7]*phix*w_x + dsg[7]*u_x + dsg[7] + du_x*sg[7] + du_y*sg[6] + dv*phix*sg[6]/radius + dv*phix_x*sg[5]/radius + dv*phix_y*sg[4]/radius - dv_x*phix*sg[5]/radius - dv_y*phix*sg[4]/radius - dw*phix*sg[4]/(radius*radius) - dw_x*phix*sg[7] - dw_x*phix_x*sg[3] - dw_x*phix_y*sg[5] - dw_y*phix*sg[6] - dw_y*phix_x*sg[5] - dw_y*phix_y*sg[4]
                S[9] = -dphiy*sg[6]*w_y - dphiy*sg[7]*w_x - dphiy*sg[4]*v_y/radius - dphiy*sg[4]/radius - dphiy*sg[5]*v_x/radius + dphiy*sg[6]*v/radius - dphiy*sg[4]*w/(radius*radius) - dphiy_x*sg[3]*w_x - dphiy_x*sg[5]*w_y + dphiy_x*sg[5]*v/radius - dphiy_y*sg[4]*w_y - dphiy_y*sg[5]*w_x + dphiy_y*sg[4]*v/radius - dsg[3]*phiy_x*w_x - dsg[3]*v_x*w_x/radius - dsg[4]*phiy*v_y/radius - dsg[4]*phiy/radius - dsg[4]*phiy*w/(radius*radius) - dsg[4]*phiy_y*w_y + dsg[4]*phiy_y*v/radius - dsg[4]*v_y*w_y/radius - dsg[4]*w_y/radius - dsg[4]*v*w/(radius*radius*radius) - dsg[5]*phiy*v_x/radius - dsg[5]*phiy_x*w_y + dsg[5]*phiy_x*v/radius - dsg[5]*phiy_y*w_x - dsg[5]*v_x*w_y/radius - dsg[5]*v_y*w_x/radius - dsg[5]*w_x/radius - dsg[6]*phiy*w_y + dsg[6]*phiy*v/radius + dsg[6]*v_y + dsg[6] - dsg[6]*v*w_y/radius + dsg[6]*w/radius + dsg[6]*(v*v)/(radius*radius) - dsg[7]*phiy*w_x + dsg[7]*v_x - dsg[7]*v*w_x/radius + dv*phiy*sg[6]/radius + dv*phiy_x*sg[5]/radius + dv*phiy_y*sg[4]/radius - dv*sg[6]*w_y/radius - dv*sg[7]*w_x/radius + 2*dv*sg[6]*v/(radius*radius) - dv*sg[4]*w/(radius*radius*radius) - dv_x*phiy*sg[5]/radius + dv_x*sg[7] - dv_x*sg[3]*w_x/radius - dv_x*sg[5]*w_y/radius - dv_y*phiy*sg[4]/radius + dv_y*sg[6] - dv_y*sg[4]*w_y/radius - dv_y*sg[5]*w_x/radius - dw*phiy*sg[4]/(radius*radius) + dw*sg[6]/radius - dw*sg[4]*v/(radius*radius*radius) - dw_x*phiy*sg[7] - dw_x*phiy_x*sg[3] - dw_x*phiy_y*sg[5] - dw_x*sg[3]*v_x/radius - dw_x*sg[5]*v_y/radius - dw_x*sg[5]/radius - dw_x*sg[7]*v/radius - dw_y*phiy*sg[6] - dw_y*phiy_x*sg[5] - dw_y*phiy_y*sg[4] - dw_y*sg[4]*v_y/radius - dw_y*sg[4]/radius - dw_y*sg[5]*v_x/radius - dw_y*sg[6]*v/radius
                S[10] = -dphix*sg[3]*w_x - dphix*sg[5]*w_y + dphix*sg[5]*v/radius - dsg[3]*phix*w_x + dsg[3]*u_x + dsg[3] - dsg[5]*phix*w_y + dsg[5]*phix*v/radius + dsg[5]*u_y + du_x*sg[3] + du_y*sg[5] + dv*phix*sg[5]/radius - dw_x*phix*sg[3] - dw_y*phix*sg[5]
                S[11] = -dphix*sg[4]*w_y - dphix*sg[5]*w_x + dphix*sg[4]*v/radius - dsg[4]*phix*w_y + dsg[4]*phix*v/radius + dsg[4]*u_y - dsg[5]*phix*w_x + dsg[5]*u_x + dsg[5] + du_x*sg[5] + du_y*sg[4] + dv*phix*sg[4]/radius - dw_x*phix*sg[5] - dw_y*phix*sg[4]
                S[12] = -dphiy*sg[3]*w_x - dphiy*sg[5]*w_y + dphiy*sg[5]*v/radius - dsg[3]*phiy*w_x + dsg[3]*v_x - dsg[3]*v*w_x/radius - dsg[5]*phiy*w_y + dsg[5]*phiy*v/radius + dsg[5]*v_y + dsg[5] - dsg[5]*v*w_y/radius + dsg[5]*w/radius + dsg[5]*(v*v)/(radius*radius) + dv*phiy*sg[5]/radius - dv*sg[3]*w_x/radius - dv*sg[5]*w_y/radius + 2*dv*sg[5]*v/(radius*radius) + dv_x*sg[3] + dv_y*sg[5] + dw*sg[5]/radius - dw_x*phiy*sg[3] - dw_x*sg[3]*v/radius - dw_y*phiy*sg[5] - dw_y*sg[5]*v/radius
                S[13] = -dphiy*sg[4]*w_y - dphiy*sg[5]*w_x + dphiy*sg[4]*v/radius - dsg[4]*phiy*w_y + dsg[4]*phiy*v/radius + dsg[4]*v_y + dsg[4] - dsg[4]*v*w_y/radius + dsg[4]*w/radius + dsg[4]*(v*v)/(radius*radius) - dsg[5]*phiy*w_x + dsg[5]*v_x - dsg[5]*v*w_x/radius + dv*phiy*sg[4]/radius - dv*sg[4]*w_y/radius - dv*sg[5]*w_x/radius + 2*dv*sg[4]*v/(radius*radius) + dv_x*sg[5] + dv_y*sg[4] + dw*sg[4]/radius - dw_x*phiy*sg[5] - dw_x*sg[5]*v/radius - dw_y*phiy*sg[4] - dw_y*sg[4]*v/radius

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
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAphix*gAphix
                        pA[9] = fAphiy*gAphiy
                        pA[10] = fAphix_x*gAphix
                        pA[11] = fAphix*gAphix_y
                        pA[12] = fAphiy_x*gAphiy
                        pA[13] = fAphiy*gAphiy_y
                        kTv[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        kTv[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[6]*S[6])
                        kTv[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[7]*S[7])
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
