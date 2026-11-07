#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Cylindrical shells using the classical laminated plate theory (CLPT), total
Lagrangian: Green-Lagrange membrane strains and changes of curvature
measured with the deformed normal

Numerically integrated kernels of the model ``'cylshell_clpt_tl'``: the
constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the internal force vector :func:`calc_fint`, the product
of the tangent stiffness matrix by a vector :func:`calc_kTv`, and the force
vector and load stiffness matrix of a follower pressure,
:func:`calc_fext_follower` and :func:`fkCfollower_num`. The mass and
aerodynamic matrices, ``fkM_num``, ``fkAx_num`` and ``fkAy_num``, are those
of :mod:`panels.models.cylshell_clpt_sanders_num`. The constitutive matrix
``Finput`` is the ``6 x 6`` matrix ``ABD`` of the CLPT.

The degrees of freedom of each term of the approximation are
``u, v, w``; the degree of freedom ``k`` of the term ``(i, j)``, ``0 <=
i < m``, ``0 <= j < n``, is at the row ``row0 + 3*(j*m + i) + k``
of the matrices.

Notes
-----
The membrane strains are the Green-Lagrange strains of the mid-surface,
those of :mod:`panels.models.cylshell_clpt_gl`, which vanish for any rigid
motion:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} (u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2)

    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \left(u_{,y}^2 +
    \left(v_{,y} + \frac{w}{r}\right)^2 + \left(w_{,y} -
    \frac{v}{r}\right)^2\right)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} \left(v_{,y} +
    \frac{w}{r}\right) + w_{,x} \left(w_{,y} - \frac{v}{r}\right)

with `y` the arc length along the circumference and `w` positive outwards,
the membrane strains of the Flügge-Lur'e-Byrne non-linear theory as given
by Amabili [1]_, Eqs. (1.118)-(1.120). The changes of curvature are measured
with the area vector `\boldsymbol{a} = \boldsymbol{X}_{,x} \times
\boldsymbol{X}_{,y}` of the deformed mid-surface `\boldsymbol{X}`, the
normal times the area stretch `|\boldsymbol{a}| = 1 + \varepsilon_{xx} +
\varepsilon_{yy} + \ldots`:

.. math::

    \kappa_{xx} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,xx} \qquad
    \kappa_{xy} = -2 \boldsymbol{a} \cdot \boldsymbol{X}_{,xy} -
    \frac{\gamma_{xy}}{2 r}

    \kappa_{yy} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,yy} - \frac{1}{r} -
    \frac{\varepsilon_{xx} + 2 \varepsilon_{yy}}{r}

with, in the local basis `(\boldsymbol{e}_x, \boldsymbol{e}_y,
\boldsymbol{e}_z)` of the undeformed cylinder:

.. math::

    \boldsymbol{X}_{,x} = (1 + u_{,x},\; v_{,x},\; w_{,x}) \qquad
    \boldsymbol{X}_{,y} = \left(u_{,y},\; 1 + v_{,y} + \frac{w}{r},\;
    w_{,y} - \frac{v}{r}\right)

    \boldsymbol{X}_{,xx} = (u_{,xx},\; v_{,xx},\; w_{,xx}) \qquad
    \boldsymbol{X}_{,xy} = \left(u_{,xy},\; v_{,xy} + \frac{w_{,x}}{r},\;
    w_{,xy} - \frac{v_{,x}}{r}\right)

    \boldsymbol{X}_{,yy} = \left(u_{,yy},\; v_{,yy} + \frac{2 w_{,y}}{r} -
    \frac{v}{r^2},\; w_{,yy} - \frac{2 v_{,y}}{r} - \frac{w}{r^2} -
    \frac{1}{r}\right)

`-\boldsymbol{a} \cdot \boldsymbol{X}_{,\alpha\beta} + (1 + \varepsilon_{xx}
+ \varepsilon_{yy}) b^0_{\alpha\beta}`, with `b^0_{yy} = -1/r` the only
non-zero component of the second fundamental form of the undeformed
cylinder, is the change of the second fundamental form `b_{\alpha\beta} =
\boldsymbol{n} \cdot \boldsymbol{X}_{,\alpha\beta}` [3]_, Sec. 3.16, p.
182, times the area stretch, to first order in the reference term. To first
order, the area vector is `(1 + \theta_{xx} + \theta_{yy}) \boldsymbol{e}_z
- \varphi_x \boldsymbol{e}_x - \varphi_y \boldsymbol{e}_y`, with the linear
membrane strains `\theta_{xx} = u_{,x}`, `\theta_{yy} = v_{,y} + w/r`,
`\theta_{xy} = (u_{,y} + v_{,x})/2` and the rotations of the normal
`\varphi_x = w_{,x}`, `\varphi_y = w_{,y} - v/r`; the stretch cancels with
that of the reference term, and the linear changes of the second
fundamental form are

.. math::

    \delta b_{xx} = w_{,xx} \qquad
    \delta b_{xy} = w_{,xy} - \frac{v_{,x}}{r} \qquad
    \delta b_{yy} = w_{,yy} - \frac{2 v_{,y}}{r} - \frac{w}{r^2}

whereas Koiter's linear changes of curvature [3]_, Eq. (3.16.8), p. 183,
are for the cylinder ([3]_, Eq. (3.17.3), p. 202)

.. math::

    \rho_{xx} = w_{,xx} \qquad
    \rho_{xy} = w_{,xy} - \frac{3}{4} \frac{v_{,x}}{r} + \frac{1}{4}
    \frac{u_{,y}}{r} \qquad
    \rho_{yy} = w_{,yy} - \frac{v_{,y}}{r}

so that `\delta b_{yy} = \rho_{yy} - \theta_{yy}/r` and `\delta b_{xy} =
\rho_{xy} - \theta_{xy}/(2 r)`: the second fundamental form sees the strain
part of `v_{,x}` and the circumferential strain, where `\rho_{\alpha\beta}`
keeps only the rotation about the normal. The terms
`-\varepsilon_{yy}/r` and `-\gamma_{xy}/(2 r)` remove these differences, so
that the linear part is `\kappa_{xx} = -\rho_{xx}`, `\kappa_{yy} =
-\rho_{yy}`, `\kappa_{xy} = -2 \rho_{xy}`, the linear changes of curvature
of Sanders [2]_, those of :mod:`panels.models.cylshell_clpt_sanders`;
changes of curvature that differ by terms of the order of the membrane
strain divided by the radius are equivalent within the accuracy of the
classical theory of thin shells ([3]_, p. 202). The six strains are
invariant under any rigid rotation of
any deformed state, cubic polynomials in the displacements, see
:mod:`panels.models`. The integrands are polynomials of degree `6p - 2`
along `x` and `6p` along `y` for approximation functions of degree `p`,
integrated exactly by `3p + 1` Gauss-Legendre points, the default of
:class:`.Shell` for this model.

The numerical integration over the domain of
:meth:`.Shell.integration_limits` admits a constitutive matrix and moments
of the density varying over the domain, a partial domain and the
non-linear terms of the Ritz constants.

References
----------
.. [1] Amabili, M., "Nonlinear Vibrations and Stability of Shells and
   Plates", Cambridge University Press, 2008.
   https://doi.org/10.1017/CBO9780511619694
.. [2] Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
   Applied Mathematics, 21(1), 21-36, 1963.
   https://doi.org/10.1090/qam/147023
.. [3] van der Heijden, A. M. A. (ed.), "W. T. Koiter's Elastic Stability
   of Solids and Structures", Cambridge University Press, 2008.
   https://doi.org/10.1017/CBO9780511546174

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
#NOTE the linear kernels, independent of the non-linear strains, are those
#     of the Sanders-Koiter cylinder
from .cylshell_clpt_sanders_num import fkM_num, fkAx_num, fkAy_num


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 3


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
        ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        6, 6)`` with one matrix per integration point.
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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients and second derivatives
    of `u`, `v`, `w` and the values of `v` and `w`), the strains
    `\boldsymbol{\varepsilon}(\boldsymbol{d})`,
    `\boldsymbol{B} = \partial \boldsymbol{\varepsilon} / \partial
    \boldsymbol{d}`, the stresses `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{H}(\boldsymbol{\sigma},
    \boldsymbol{d}) = \sum_r \sigma_r \, \partial^2 \varepsilon_r / \partial
    \boldsymbol{d}^2`:

    .. math::

        \boldsymbol{K}_C = \int \boldsymbol{\Phi}^\top \left[\boldsymbol{B}^\top
        \boldsymbol{F} \boldsymbol{B} + \boldsymbol{H}(\boldsymbol{\sigma},
        \boldsymbol{d}) - \boldsymbol{H}(\boldsymbol{\sigma}_L,
        \boldsymbol{0})\right] \boldsymbol{\Phi} \, dA

    with `\boldsymbol{\sigma}_L` the stresses of the linear strains, whose
    term `\boldsymbol{H}(\boldsymbol{\sigma}_L, \boldsymbol{0})` is assembled
    by :func:`fkG_num`, such that ``fkC_num(NLgeom=1) + fkG_num`` is the exact
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
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, p, q, r, s, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w
    cdef double u_xx, u_yy, u_xy, v_xx, v_yy, v_xy, w_xx, w_yy
    cdef double w_xy
    cdef double e[6]
    cdef double eL[6]
    cdef double sg[6]
    cdef double sgL[6]
    cdef double fAu, fAu_x, fAu_xx, fAv, fAv_x, fAv_xx, fAw
    cdef double fAw_x, fAw_xx, gAu, gAu_y, gAu_yy, gAv, gAv_y
    cdef double gAv_yy, gAw, gAw_y, gAw_yy, fBu, fBu_x, fBu_xx
    cdef double fBv, fBv_x, fBv_xx, fBw, fBw_x, fBw_xx, gBu
    cdef double gBu_y, gBu_yy, gBv, gBv_y, gBv_yy, gBw, gBw_y
    cdef double gBw_yy
    cdef double Bm[102]
    cdef double FB[102]
    cdef double K[289]
    cdef double TA[51]
    cdef double pA[17]
    cdef double pB[17]

    cdef long [::1] kCr, kCc
    cdef double [::1] kCv

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
    radius = shell.r
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

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

    ftab = np.zeros((9, m), dtype=DOUBLE)
    gtab = np.zeros((9, n), dtype=DOUBLE)

    fdim = 9*m*m*n*n
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
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = (2/a)*fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = (2/a)*(2/a)*fpp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[3, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[5, i] = (2/a)*(2/a)*fpp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[8, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = (2/b)*(2/b)*fpp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[3, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[5, j] = (2/b)*(2/b)*fpp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[6, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[8, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

                # generalized derivatives at the point, zero when NLgeom == 0
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w_x = 0.
                w_y = 0.
                v = 0.
                w = 0.
                u_xx = 0.
                u_yy = 0.
                u_xy = 0.
                v_xx = 0.
                v_yy = 0.
                v_xy = 0.
                w_xx = 0.
                w_yy = 0.
                w_xy = 0.
                if NLgeom == 1:
                    for j in range(n):
                        gAu = gtab[0, j]
                        gAu_y = gtab[1, j]
                        gAu_yy = gtab[2, j]
                        gAv = gtab[3, j]
                        gAv_y = gtab[4, j]
                        gAv_yy = gtab[5, j]
                        gAw = gtab[6, j]
                        gAw_y = gtab[7, j]
                        gAw_yy = gtab[8, j]
                        for i in range(m):
                            fAu = ftab[0, i]
                            fAu_x = ftab[1, i]
                            fAu_xx = ftab[2, i]
                            fAv = ftab[3, i]
                            fAv_x = ftab[4, i]
                            fAv_xx = ftab[5, i]
                            fAw = ftab[6, i]
                            fAw_x = ftab[7, i]
                            fAw_xx = ftab[8, i]
                            col = col0 + DOF*(j*m + i)
                            u_x += fAu_x*gAu*cs[col+0]
                            u_y += fAu*gAu_y*cs[col+0]
                            v_x += fAv_x*gAv*cs[col+1]
                            v_y += fAv*gAv_y*cs[col+1]
                            w_x += fAw_x*gAw*cs[col+2]
                            w_y += fAw*gAw_y*cs[col+2]
                            v += fAv*gAv*cs[col+1]
                            w += fAw*gAw*cs[col+2]
                            u_xx += fAu_xx*gAu*cs[col+0]
                            u_yy += fAu*gAu_yy*cs[col+0]
                            u_xy += fAu_x*gAu_y*cs[col+0]
                            v_xx += fAv_xx*gAv*cs[col+1]
                            v_yy += fAv*gAv_yy*cs[col+1]
                            v_xy += fAv_x*gAv_y*cs[col+1]
                            w_xx += fAw_xx*gAw*cs[col+2]
                            w_yy += fAw*gAw_yy*cs[col+2]
                            w_xy += fAw_x*gAw_y*cs[col+2]

                # [B] = d eps/d d at the point
                for p in range(102):
                    Bm[p] = 0.
                Bm[0] = u_x + 1
                Bm[2] = v_x
                Bm[4] = w_x
                Bm[18] = u_y
                Bm[20] = v_y + 1 + w/radius
                Bm[22] = w_y - v/radius
                Bm[23] = -w_y/radius + v/(radius*radius)
                Bm[24] = v_y/radius + 1/radius + w/(radius*radius)
                Bm[34] = u_y
                Bm[35] = u_x + 1
                Bm[36] = v_y + 1 + w/radius
                Bm[37] = v_x
                Bm[38] = w_y - v/radius
                Bm[39] = w_x
                Bm[40] = -w_x/radius
                Bm[41] = v_x/radius
                Bm[51] = v_xx*w_y - v_y*w_xx - w_xx - v*v_xx/radius - w*w_xx/radius
                Bm[52] = v_x*w_xx - v_xx*w_x
                Bm[53] = -u_xx*w_y + u_y*w_xx + u_xx*v/radius
                Bm[54] = -u_x*w_xx + u_xx*w_x - w_xx
                Bm[55] = u_xx*v_y + u_xx - u_y*v_xx + u_xx*w/radius
                Bm[56] = u_x*v_xx - u_xx*v_x + v_xx
                Bm[57] = -u_x*v_xx/radius + u_xx*v_x/radius - v_xx/radius
                Bm[58] = -u_x*w_xx/radius + u_xx*w_x/radius - w_xx/radius
                Bm[59] = -v_x*w_y + v_y*w_x + w_x + v*v_x/radius + w*w_x/radius
                Bm[62] = u_x*w_y - u_y*w_x + w_y - u_x*v/radius - v/radius
                Bm[65] = -u_x*v_y - u_x + u_y*v_x - v_y - 1 - u_x*w/radius - w/radius
                Bm[68] = -v_y*w_yy + v_yy*w_y - w_yy - u_x/radius - v*v_yy/radius + 2*(v_y*v_y)/radius + 3*v_y/radius - w*w_yy/radius + 2*(w_y*w_y)/radius - 3*v*w_y/(radius*radius) + 3*v_y*w/(radius*radius) + 2*w/(radius*radius) + (v*v)/(radius*radius*radius) + (w*w)/(radius*radius*radius)
                Bm[69] = v_x*w_yy - v_yy*w_x - 2*u_y/radius - 2*v_x*v_y/radius - v_x/radius - 2*w_x*w_y/radius + v*w_x/(radius*radius) - v_x*w/(radius*radius)
                Bm[70] = u_y*w_yy - u_yy*w_y - 2*u_y*v_y/radius - u_y/radius + u_yy*v/radius - v_x/radius - u_y*w/(radius*radius)
                Bm[71] = -u_x*w_yy + u_yy*w_x - w_yy + 4*u_x*v_y/radius + 3*u_x/radius - 2*u_y*v_x/radius + 2*v_y/radius + 1/radius + 3*u_x*w/(radius*radius) + w/(radius*radius)
                Bm[72] = -u_y*v_yy + u_yy*v_y + u_yy - 2*u_y*w_y/radius + u_yy*w/radius - w_x/radius + u_y*v/(radius*radius)
                Bm[73] = u_x*v_yy - u_yy*v_x + v_yy + 4*u_x*w_y/radius - 2*u_y*w_x/radius + 2*w_y/radius - 3*u_x*v/(radius*radius) - v/(radius*radius)
                Bm[74] = -u_x*v_yy/radius + u_yy*v_x/radius - v_yy/radius - 3*u_x*w_y/(radius*radius) + u_y*w_x/(radius*radius) - w_y/(radius*radius) + 2*u_x*v/(radius*radius*radius)
                Bm[75] = -u_x*w_yy/radius + u_yy*w_x/radius - w_yy/radius + 3*u_x*v_y/(radius*radius) + 2*u_x/(radius*radius) - u_y*v_x/(radius*radius) + v_y/(radius*radius) + 2*u_x*w/(radius*radius*radius)
                Bm[77] = -v_x*w_y + v_y*w_x + w_x + v*v_x/radius + w*w_x/radius
                Bm[80] = u_x*w_y - u_y*w_x + w_y - u_x*v/radius - v/radius
                Bm[83] = -u_x*v_y - u_x + u_y*v_x - v_y - 1 - u_x*w/radius - w/radius
                Bm[85] = 2*v_xy*w_y - 2*v_y*w_xy - 2*w_xy - 0.5*u_y/radius - 2*v*v_xy/radius + 2*v_x*v_y/radius + 2*v_x/radius - 2*w*w_xy/radius + 2*w_x*w_y/radius - 2*v*w_x/(radius*radius) + 2*v_x*w/(radius*radius)
                Bm[86] = 2*v_x*w_xy - 2*v_xy*w_x - 0.5*u_x/radius - 2*(v_x*v_x)/radius - 2*(w_x*w_x)/radius - 0.5/radius
                Bm[87] = -2*u_xy*w_y + 2*u_y*w_xy + 2*u_x*v_y/radius + 2*u_x/radius + 2*u_xy*v/radius - 4*u_y*v_x/radius + 1.5*v_y/radius + 1.5/radius + 2*u_x*w/(radius*radius) + 1.5*w/(radius*radius)
                Bm[88] = -2*u_x*w_xy + 2*u_xy*w_x - 2*w_xy + 2*u_x*v_x/radius + 1.5*v_x/radius
                Bm[89] = 2*u_xy*v_y + 2*u_xy - 2*u_y*v_xy + 2*u_x*w_y/radius + 2*u_xy*w/radius - 4*u_y*w_x/radius + 1.5*w_y/radius - 2*u_x*v/(radius*radius) - 1.5*v/(radius*radius)
                Bm[90] = 2*u_x*v_xy - 2*u_xy*v_x + 2*v_xy + 2*u_x*w_x/radius + 1.5*w_x/radius
                Bm[91] = -2*u_x*v_xy/radius + 2*u_xy*v_x/radius - 2*v_xy/radius - 2*u_x*w_x/(radius*radius) - 1.5*w_x/(radius*radius)
                Bm[92] = -2*u_x*w_xy/radius + 2*u_xy*w_x/radius - 2*w_xy/radius + 2*u_x*v_x/(radius*radius) + 1.5*v_x/(radius*radius)
                Bm[95] = -2*v_x*w_y + 2*v_y*w_x + 2*w_x + 2*v*v_x/radius + 2*w*w_x/radius
                Bm[98] = 2*u_x*w_y - 2*u_y*w_x + 2*w_y - 2*u_x*v/radius - 2*v/radius
                Bm[101] = -2*u_x*v_y - 2*u_x + 2*u_y*v_x - 2*v_y - 2 - 2*u_x*w/radius - 2*w/radius

                # K = [B]^T [ABD] [B]
                for r in range(6):
                    for q in range(17):
                        FB[r*17 + q] = 0.
                        for s in range(6):
                            FB[r*17 + q] += F[r*6 + s]*Bm[s*17 + q]
                for p in range(17):
                    for q in range(17):
                        K[p*17 + q] = 0.
                        for r in range(6):
                            K[p*17 + q] += Bm[r*17 + p]*FB[r*17 + q]

                # stresses of the strains and of the linear strains
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
                e[3] = u_x*v_xx*w_y - u_x*v_y*w_xx - u_x*w_xx - u_xx*v_x*w_y + u_xx*v_y*w_x + u_xx*w_x + u_y*v_x*w_xx - u_y*v_xx*w_x + v_xx*w_y - v_y*w_xx - w_xx - u_x*v*v_xx/radius - u_x*w*w_xx/radius + u_xx*v*v_x/radius + u_xx*w*w_x/radius - v*v_xx/radius - w*w_xx/radius
                e[4] = -u_x*v_y*w_yy + u_x*v_yy*w_y - u_x*w_yy + u_y*v_x*w_yy - u_y*v_yy*w_x - u_yy*v_x*w_y + u_yy*v_y*w_x + u_yy*w_x - v_y*w_yy + v_yy*w_y - w_yy - 0.5*(u_x*u_x)/radius - u_x*v*v_yy/radius + 2*u_x*(v_y*v_y)/radius + 3*u_x*v_y/radius - u_x*w*w_yy/radius + 2*u_x*(w_y*w_y)/radius - (u_y*u_y)/radius - 2*u_y*v_x*v_y/radius - u_y*v_x/radius - 2*u_y*w_x*w_y/radius + u_yy*v*v_x/radius + u_yy*w*w_x/radius - v*v_yy/radius - 0.5*(v_x*v_x)/radius + (v_y*v_y)/radius + v_y/radius - w*w_yy/radius - 0.5*(w_x*w_x)/radius + (w_y*w_y)/radius - 3*u_x*v*w_y/(radius*radius) + 3*u_x*v_y*w/(radius*radius) + 2*u_x*w/(radius*radius) + u_y*v*w_x/(radius*radius) - u_y*v_x*w/(radius*radius) - v*w_y/(radius*radius) + v_y*w/(radius*radius) + u_x*(v*v)/(radius*radius*radius) + u_x*(w*w)/(radius*radius*radius)
                e[5] = 2*u_x*v_xy*w_y - 2*u_x*v_y*w_xy - 2*u_x*w_xy - 2*u_xy*v_x*w_y + 2*u_xy*v_y*w_x + 2*u_xy*w_x + 2*u_y*v_x*w_xy - 2*u_y*v_xy*w_x + 2*v_xy*w_y - 2*v_y*w_xy - 2*w_xy - 0.5*u_x*u_y/radius - 2*u_x*v*v_xy/radius + 2*u_x*v_x*v_y/radius + 2*u_x*v_x/radius - 2*u_x*w*w_xy/radius + 2*u_x*w_x*w_y/radius + 2*u_xy*v*v_x/radius + 2*u_xy*w*w_x/radius - 2*u_y*(v_x*v_x)/radius - 2*u_y*(w_x*w_x)/radius - 0.5*u_y/radius - 2*v*v_xy/radius + 1.5*v_x*v_y/radius + 1.5*v_x/radius - 2*w*w_xy/radius + 1.5*w_x*w_y/radius - 2*u_x*v*w_x/(radius*radius) + 2*u_x*v_x*w/(radius*radius) - 1.5*v*w_x/(radius*radius) + 1.5*v_x*w/(radius*radius)
                eL[0] = u_x
                eL[1] = v_y + w/radius
                eL[2] = u_y + v_x
                eL[3] = -w_xx
                eL[4] = -w_yy + v_y/radius
                eL[5] = -2*w_xy - 0.5*u_y/radius + 1.5*v_x/radius
                for r in range(6):
                    sg[r] = 0.
                    for s in range(6):
                        sg[r] += F[r*6 + s]*e[s]
                for r in range(6):
                    sgL[r] = 0.
                    for s in range(6):
                        sgL[r] += F[r*6 + s]*eL[s]
                # K += [H(sigma, d)] - [H0(sigma_L)], the second term being
                # assembled by fkG_num, which is homogeneous of degree one
                # in cs as linear buckling requires
                K[0] += -sgL[0] + sg[0] + sgL[4]/radius - sg[4]/radius
                K[1] += -sgL[2] + sg[2] + 0.5*sgL[5]/radius - 0.5*sg[5]/radius
                K[2] += -2*sgL[5]/radius + 2*sg[5]*v_y/radius + 2*sg[5]/radius + 2*sg[5]*w/(radius*radius)
                K[3] += -sg[3]*w_xx - sg[4]*w_yy - 2*sg[5]*w_xy - 3*sgL[4]/radius + 4*sg[4]*v_y/radius + 3*sg[4]/radius + 2*sg[5]*v_x/radius + 3*sg[4]*w/(radius*radius)
                K[4] += 2*sg[5]*w_y/radius - 2*sg[5]*v/(radius*radius)
                K[5] += sg[3]*v_xx + sg[4]*v_yy + 2*sg[5]*v_xy + 4*sg[4]*w_y/radius + 2*sg[5]*w_x/radius - 3*sg[4]*v/(radius*radius)
                K[6] += -sg[3]*v_xx/radius - sg[4]*v_yy/radius - 2*sg[5]*v_xy/radius - 3*sg[4]*w_y/(radius*radius) - 2*sg[5]*w_x/(radius*radius) + 2*sg[4]*v/(radius*radius*radius)
                K[7] += -sg[3]*w_xx/radius - sg[4]*w_yy/radius - 2*sg[5]*w_xy/radius - 2*sgL[4]/(radius*radius) + 3*sg[4]*v_y/(radius*radius) + 2*sg[4]/(radius*radius) + 2*sg[5]*v_x/(radius*radius) + 2*sg[4]*w/(radius*radius*radius)
                K[11] += sg[3]*w_y - sg[3]*v/radius
                K[12] += sg[4]*w_y - sg[4]*v/radius
                K[13] += 2*sg[5]*w_y - 2*sg[5]*v/radius
                K[14] += sgL[3] - sg[3]*v_y - sg[3] - sg[3]*w/radius
                K[15] += sgL[4] - sg[4]*v_y - sg[4] - sg[4]*w/radius
                K[16] += 2*sgL[5] - 2*sg[5]*v_y - 2*sg[5] - 2*sg[5]*w/radius
                K[17] += -sgL[2] + sg[2] + 0.5*sgL[5]/radius - 0.5*sg[5]/radius
                K[18] += -sgL[1] + sg[1] + 2*sgL[4]/radius - 2*sg[4]/radius
                K[19] += sg[3]*w_xx + sg[4]*w_yy + 2*sg[5]*w_xy + sgL[4]/radius - 2*sg[4]*v_y/radius - sg[4]/radius - 4*sg[5]*v_x/radius - sg[4]*w/(radius*radius)
                K[20] += -2*sg[4]*v_x/radius
                K[21] += -sg[3]*v_xx - sg[4]*v_yy - 2*sg[5]*v_xy - 2*sg[4]*w_y/radius - 4*sg[5]*w_x/radius + sg[4]*v/(radius*radius)
                K[22] += -2*sg[4]*w_x/radius
                K[23] += sg[4]*w_x/(radius*radius)
                K[24] += -sg[4]*v_x/(radius*radius)
                K[28] += -sg[3]*w_x
                K[29] += -sg[4]*w_x
                K[30] += -2*sg[5]*w_x
                K[31] += sg[3]*v_x
                K[32] += sg[4]*v_x
                K[33] += 2*sg[5]*v_x
                K[34] += -2*sgL[5]/radius + 2*sg[5]*v_y/radius + 2*sg[5]/radius + 2*sg[5]*w/(radius*radius)
                K[35] += sg[3]*w_xx + sg[4]*w_yy + 2*sg[5]*w_xy + sgL[4]/radius - 2*sg[4]*v_y/radius - sg[4]/radius - 4*sg[5]*v_x/radius - sg[4]*w/(radius*radius)
                K[36] += -sgL[0] + sg[0] + sgL[4]/radius - sg[4]/radius - 4*sg[5]*u_y/radius
                K[37] += -sgL[2] + sg[2] - 1.5*sgL[5]/radius - 2*sg[4]*u_y/radius + 2*sg[5]*u_x/radius + 1.5*sg[5]/radius
                K[39] += -sg[3]*u_xx - sg[4]*u_yy - 2*sg[5]*u_xy
                K[40] += sg[3]*u_xx/radius + sg[4]*u_yy/radius + 2*sg[5]*u_xy/radius
                K[41] += -sgL[2]/radius + sg[2]/radius - 1.5*sgL[5]/(radius*radius) - sg[4]*u_y/(radius*radius) + 2*sg[5]*u_x/(radius*radius) + 1.5*sg[5]/(radius*radius)
                K[42] += -sg[3]*w_y + sg[3]*v/radius
                K[43] += -sg[4]*w_y + sg[4]*v/radius
                K[44] += -2*sg[5]*w_y + 2*sg[5]*v/radius
                K[48] += sg[3]*u_y
                K[49] += sg[4]*u_y
                K[50] += 2*sg[5]*u_y
                K[51] += -sg[3]*w_xx - sg[4]*w_yy - 2*sg[5]*w_xy - 3*sgL[4]/radius + 4*sg[4]*v_y/radius + 3*sg[4]/radius + 2*sg[5]*v_x/radius + 3*sg[4]*w/(radius*radius)
                K[52] += -2*sg[4]*v_x/radius
                K[53] += -sgL[2] + sg[2] - 1.5*sgL[5]/radius - 2*sg[4]*u_y/radius + 2*sg[5]*u_x/radius + 1.5*sg[5]/radius
                K[54] += -sgL[1] + sg[1] - 2*sgL[4]/radius + 4*sg[4]*u_x/radius + 2*sg[4]/radius
                K[55] += sg[3]*u_xx + sg[4]*u_yy + 2*sg[5]*u_xy
                K[58] += -sgL[1]/radius + sg[1]/radius - sgL[4]/(radius*radius) + 3*sg[4]*u_x/(radius*radius) + sg[4]/(radius*radius)
                K[59] += sg[3]*w_x
                K[60] += sg[4]*w_x
                K[61] += 2*sg[5]*w_x
                K[65] += sgL[3] - sg[3]*u_x - sg[3]
                K[66] += sgL[4] - sg[4]*u_x - sg[4]
                K[67] += 2*sgL[5] - 2*sg[5]*u_x - 2*sg[5]
                K[68] += 2*sg[5]*w_y/radius - 2*sg[5]*v/(radius*radius)
                K[69] += -sg[3]*v_xx - sg[4]*v_yy - 2*sg[5]*v_xy - 2*sg[4]*w_y/radius - 4*sg[5]*w_x/radius + sg[4]*v/(radius*radius)
                K[71] += sg[3]*u_xx + sg[4]*u_yy + 2*sg[5]*u_xy
                K[72] += -sgL[0] + sg[0] + sgL[4]/radius - sg[4]/radius - 4*sg[5]*u_y/radius
                K[73] += -sgL[2] + sg[2] - 1.5*sgL[5]/radius - 2*sg[4]*u_y/radius + 2*sg[5]*u_x/radius + 1.5*sg[5]/radius
                K[74] += sgL[2]/radius - sg[2]/radius + 1.5*sgL[5]/(radius*radius) + sg[4]*u_y/(radius*radius) - 2*sg[5]*u_x/(radius*radius) - 1.5*sg[5]/(radius*radius)
                K[75] += sg[3]*u_xx/radius + sg[4]*u_yy/radius + 2*sg[5]*u_xy/radius
                K[76] += -sgL[3] + sg[3]*v_y + sg[3] + sg[3]*w/radius
                K[77] += -sgL[4] + sg[4]*v_y + sg[4] + sg[4]*w/radius
                K[78] += -2*sgL[5] + 2*sg[5]*v_y + 2*sg[5] + 2*sg[5]*w/radius
                K[79] += -sg[3]*u_y
                K[80] += -sg[4]*u_y
                K[81] += -2*sg[5]*u_y
                K[85] += sg[3]*v_xx + sg[4]*v_yy + 2*sg[5]*v_xy + 4*sg[4]*w_y/radius + 2*sg[5]*w_x/radius - 3*sg[4]*v/(radius*radius)
                K[86] += -2*sg[4]*w_x/radius
                K[87] += -sg[3]*u_xx - sg[4]*u_yy - 2*sg[5]*u_xy
                K[89] += -sgL[2] + sg[2] - 1.5*sgL[5]/radius - 2*sg[4]*u_y/radius + 2*sg[5]*u_x/radius + 1.5*sg[5]/radius
                K[90] += -sgL[1] + sg[1] - 2*sgL[4]/radius + 4*sg[4]*u_x/radius + 2*sg[4]/radius
                K[91] += sgL[1]/radius - sg[1]/radius + sgL[4]/(radius*radius) - 3*sg[4]*u_x/(radius*radius) - sg[4]/(radius*radius)
                K[93] += -sg[3]*v_x
                K[94] += -sg[4]*v_x
                K[95] += -2*sg[5]*v_x
                K[96] += -sgL[3] + sg[3]*u_x + sg[3]
                K[97] += -sgL[4] + sg[4]*u_x + sg[4]
                K[98] += -2*sgL[5] + 2*sg[5]*u_x + 2*sg[5]
                K[102] += -sg[3]*v_xx/radius - sg[4]*v_yy/radius - 2*sg[5]*v_xy/radius - 3*sg[4]*w_y/(radius*radius) - 2*sg[5]*w_x/(radius*radius) + 2*sg[4]*v/(radius*radius*radius)
                K[103] += sg[4]*w_x/(radius*radius)
                K[104] += sg[3]*u_xx/radius + sg[4]*u_yy/radius + 2*sg[5]*u_xy/radius
                K[106] += sgL[2]/radius - sg[2]/radius + 1.5*sgL[5]/(radius*radius) + sg[4]*u_y/(radius*radius) - 2*sg[5]*u_x/(radius*radius) - 1.5*sg[5]/(radius*radius)
                K[107] += sgL[1]/radius - sg[1]/radius + sgL[4]/(radius*radius) - 3*sg[4]*u_x/(radius*radius) - sg[4]/(radius*radius)
                K[108] += -sgL[1]/(radius*radius) + sg[1]/(radius*radius) + 2*sg[4]*u_x/(radius*radius*radius)
                K[110] += sg[3]*v_x/radius
                K[111] += sg[4]*v_x/radius
                K[112] += 2*sg[5]*v_x/radius
                K[113] += sgL[3]/radius - sg[3]*u_x/radius - sg[3]/radius
                K[114] += sgL[4]/radius - sg[4]*u_x/radius - sg[4]/radius
                K[115] += 2*sgL[5]/radius - 2*sg[5]*u_x/radius - 2*sg[5]/radius
                K[119] += -sg[3]*w_xx/radius - sg[4]*w_yy/radius - 2*sg[5]*w_xy/radius - 2*sgL[4]/(radius*radius) + 3*sg[4]*v_y/(radius*radius) + 2*sg[4]/(radius*radius) + 2*sg[5]*v_x/(radius*radius) + 2*sg[4]*w/(radius*radius*radius)
                K[120] += -sg[4]*v_x/(radius*radius)
                K[121] += -sgL[2]/radius + sg[2]/radius - 1.5*sgL[5]/(radius*radius) - sg[4]*u_y/(radius*radius) + 2*sg[5]*u_x/(radius*radius) + 1.5*sg[5]/(radius*radius)
                K[122] += -sgL[1]/radius + sg[1]/radius - sgL[4]/(radius*radius) + 3*sg[4]*u_x/(radius*radius) + sg[4]/(radius*radius)
                K[123] += sg[3]*u_xx/radius + sg[4]*u_yy/radius + 2*sg[5]*u_xy/radius
                K[126] += -sgL[1]/(radius*radius) + sg[1]/(radius*radius) + 2*sg[4]*u_x/(radius*radius*radius)
                K[127] += sg[3]*w_x/radius
                K[128] += sg[4]*w_x/radius
                K[129] += 2*sg[5]*w_x/radius
                K[133] += sgL[3]/radius - sg[3]*u_x/radius - sg[3]/radius
                K[134] += sgL[4]/radius - sg[4]*u_x/radius - sg[4]/radius
                K[135] += 2*sgL[5]/radius - 2*sg[5]*u_x/radius - 2*sg[5]/radius
                K[138] += -sg[3]*w_y + sg[3]*v/radius
                K[139] += sg[3]*w_x
                K[140] += -sgL[3] + sg[3]*v_y + sg[3] + sg[3]*w/radius
                K[141] += -sg[3]*v_x
                K[142] += sg[3]*v_x/radius
                K[143] += sg[3]*w_x/radius
                K[155] += -sg[4]*w_y + sg[4]*v/radius
                K[156] += sg[4]*w_x
                K[157] += -sgL[4] + sg[4]*v_y + sg[4] + sg[4]*w/radius
                K[158] += -sg[4]*v_x
                K[159] += sg[4]*v_x/radius
                K[160] += sg[4]*w_x/radius
                K[172] += -2*sg[5]*w_y + 2*sg[5]*v/radius
                K[173] += 2*sg[5]*w_x
                K[174] += -2*sgL[5] + 2*sg[5]*v_y + 2*sg[5] + 2*sg[5]*w/radius
                K[175] += -2*sg[5]*v_x
                K[176] += 2*sg[5]*v_x/radius
                K[177] += 2*sg[5]*w_x/radius
                K[187] += sg[3]*w_y - sg[3]*v/radius
                K[188] += -sg[3]*w_x
                K[191] += -sg[3]*u_y
                K[192] += -sgL[3] + sg[3]*u_x + sg[3]
                K[193] += sgL[3]/radius - sg[3]*u_x/radius - sg[3]/radius
                K[204] += sg[4]*w_y - sg[4]*v/radius
                K[205] += -sg[4]*w_x
                K[208] += -sg[4]*u_y
                K[209] += -sgL[4] + sg[4]*u_x + sg[4]
                K[210] += sgL[4]/radius - sg[4]*u_x/radius - sg[4]/radius
                K[221] += 2*sg[5]*w_y - 2*sg[5]*v/radius
                K[222] += -2*sg[5]*w_x
                K[225] += -2*sg[5]*u_y
                K[226] += -2*sgL[5] + 2*sg[5]*u_x + 2*sg[5]
                K[227] += 2*sgL[5]/radius - 2*sg[5]*u_x/radius - 2*sg[5]/radius
                K[238] += sgL[3] - sg[3]*v_y - sg[3] - sg[3]*w/radius
                K[239] += sg[3]*v_x
                K[240] += sg[3]*u_y
                K[241] += sgL[3] - sg[3]*u_x - sg[3]
                K[245] += sgL[3]/radius - sg[3]*u_x/radius - sg[3]/radius
                K[255] += sgL[4] - sg[4]*v_y - sg[4] - sg[4]*w/radius
                K[256] += sg[4]*v_x
                K[257] += sg[4]*u_y
                K[258] += sgL[4] - sg[4]*u_x - sg[4]
                K[262] += sgL[4]/radius - sg[4]*u_x/radius - sg[4]/radius
                K[272] += 2*sgL[5] - 2*sg[5]*v_y - 2*sg[5] - 2*sg[5]*w/radius
                K[273] += 2*sg[5]*v_x
                K[274] += 2*sg[5]*u_y
                K[275] += 2*sgL[5] - 2*sg[5]*u_x - 2*sg[5]
                K[279] += 2*sgL[5]/radius - 2*sg[5]*u_x/radius - 2*sg[5]/radius

                c = -1
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAu_yy = gtab[2, j]
                    gAv = gtab[3, j]
                    gAv_y = gtab[4, j]
                    gAv_yy = gtab[5, j]
                    gAw = gtab[6, j]
                    gAw_y = gtab[7, j]
                    gAw_yy = gtab[8, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAu_xx = ftab[2, i]
                        fAv = ftab[3, i]
                        fAv_x = ftab[4, i]
                        fAv_xx = ftab[5, i]
                        fAw = ftab[6, i]
                        fAw_x = ftab[7, i]
                        fAw_xx = ftab[8, i]
                        row = row0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAu_xx*gAu
                        pA[9] = fAu*gAu_yy
                        pA[10] = fAu_x*gAu_y
                        pA[11] = fAv_xx*gAv
                        pA[12] = fAv*gAv_yy
                        pA[13] = fAv_x*gAv_y
                        pA[14] = fAw_xx*gAw
                        pA[15] = fAw*gAw_yy
                        pA[16] = fAw_x*gAw_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[17] + pA[8]*K[136] + pA[9]*K[153] + pA[10]*K[170]
                        TA[1] = pA[0]*K[1] + pA[1]*K[18] + pA[8]*K[137] + pA[9]*K[154] + pA[10]*K[171]
                        TA[2] = pA[0]*K[2] + pA[1]*K[19] + pA[8]*K[138] + pA[9]*K[155] + pA[10]*K[172]
                        TA[3] = pA[0]*K[3] + pA[1]*K[20] + pA[8]*K[139] + pA[9]*K[156] + pA[10]*K[173]
                        TA[4] = pA[0]*K[4] + pA[1]*K[21] + pA[8]*K[140] + pA[9]*K[157] + pA[10]*K[174]
                        TA[5] = pA[0]*K[5] + pA[1]*K[22] + pA[8]*K[141] + pA[9]*K[158] + pA[10]*K[175]
                        TA[6] = pA[0]*K[6] + pA[1]*K[23] + pA[8]*K[142] + pA[9]*K[159] + pA[10]*K[176]
                        TA[7] = pA[0]*K[7] + pA[1]*K[24] + pA[8]*K[143] + pA[9]*K[160] + pA[10]*K[177]
                        TA[8] = pA[0]*K[8] + pA[1]*K[25] + pA[8]*K[144] + pA[9]*K[161] + pA[10]*K[178]
                        TA[9] = pA[0]*K[9] + pA[1]*K[26] + pA[8]*K[145] + pA[9]*K[162] + pA[10]*K[179]
                        TA[10] = pA[0]*K[10] + pA[1]*K[27] + pA[8]*K[146] + pA[9]*K[163] + pA[10]*K[180]
                        TA[11] = pA[0]*K[11] + pA[1]*K[28] + pA[8]*K[147] + pA[9]*K[164] + pA[10]*K[181]
                        TA[12] = pA[0]*K[12] + pA[1]*K[29] + pA[8]*K[148] + pA[9]*K[165] + pA[10]*K[182]
                        TA[13] = pA[0]*K[13] + pA[1]*K[30] + pA[8]*K[149] + pA[9]*K[166] + pA[10]*K[183]
                        TA[14] = pA[0]*K[14] + pA[1]*K[31] + pA[8]*K[150] + pA[9]*K[167] + pA[10]*K[184]
                        TA[15] = pA[0]*K[15] + pA[1]*K[32] + pA[8]*K[151] + pA[9]*K[168] + pA[10]*K[185]
                        TA[16] = pA[0]*K[16] + pA[1]*K[33] + pA[8]*K[152] + pA[9]*K[169] + pA[10]*K[186]
                        TA[17] = pA[2]*K[34] + pA[3]*K[51] + pA[6]*K[102] + pA[11]*K[187] + pA[12]*K[204] + pA[13]*K[221]
                        TA[18] = pA[2]*K[35] + pA[3]*K[52] + pA[6]*K[103] + pA[11]*K[188] + pA[12]*K[205] + pA[13]*K[222]
                        TA[19] = pA[2]*K[36] + pA[3]*K[53] + pA[6]*K[104] + pA[11]*K[189] + pA[12]*K[206] + pA[13]*K[223]
                        TA[20] = pA[2]*K[37] + pA[3]*K[54] + pA[6]*K[105] + pA[11]*K[190] + pA[12]*K[207] + pA[13]*K[224]
                        TA[21] = pA[2]*K[38] + pA[3]*K[55] + pA[6]*K[106] + pA[11]*K[191] + pA[12]*K[208] + pA[13]*K[225]
                        TA[22] = pA[2]*K[39] + pA[3]*K[56] + pA[6]*K[107] + pA[11]*K[192] + pA[12]*K[209] + pA[13]*K[226]
                        TA[23] = pA[2]*K[40] + pA[3]*K[57] + pA[6]*K[108] + pA[11]*K[193] + pA[12]*K[210] + pA[13]*K[227]
                        TA[24] = pA[2]*K[41] + pA[3]*K[58] + pA[6]*K[109] + pA[11]*K[194] + pA[12]*K[211] + pA[13]*K[228]
                        TA[25] = pA[2]*K[42] + pA[3]*K[59] + pA[6]*K[110] + pA[11]*K[195] + pA[12]*K[212] + pA[13]*K[229]
                        TA[26] = pA[2]*K[43] + pA[3]*K[60] + pA[6]*K[111] + pA[11]*K[196] + pA[12]*K[213] + pA[13]*K[230]
                        TA[27] = pA[2]*K[44] + pA[3]*K[61] + pA[6]*K[112] + pA[11]*K[197] + pA[12]*K[214] + pA[13]*K[231]
                        TA[28] = pA[2]*K[45] + pA[3]*K[62] + pA[6]*K[113] + pA[11]*K[198] + pA[12]*K[215] + pA[13]*K[232]
                        TA[29] = pA[2]*K[46] + pA[3]*K[63] + pA[6]*K[114] + pA[11]*K[199] + pA[12]*K[216] + pA[13]*K[233]
                        TA[30] = pA[2]*K[47] + pA[3]*K[64] + pA[6]*K[115] + pA[11]*K[200] + pA[12]*K[217] + pA[13]*K[234]
                        TA[31] = pA[2]*K[48] + pA[3]*K[65] + pA[6]*K[116] + pA[11]*K[201] + pA[12]*K[218] + pA[13]*K[235]
                        TA[32] = pA[2]*K[49] + pA[3]*K[66] + pA[6]*K[117] + pA[11]*K[202] + pA[12]*K[219] + pA[13]*K[236]
                        TA[33] = pA[2]*K[50] + pA[3]*K[67] + pA[6]*K[118] + pA[11]*K[203] + pA[12]*K[220] + pA[13]*K[237]
                        TA[34] = pA[4]*K[68] + pA[5]*K[85] + pA[7]*K[119] + pA[14]*K[238] + pA[15]*K[255] + pA[16]*K[272]
                        TA[35] = pA[4]*K[69] + pA[5]*K[86] + pA[7]*K[120] + pA[14]*K[239] + pA[15]*K[256] + pA[16]*K[273]
                        TA[36] = pA[4]*K[70] + pA[5]*K[87] + pA[7]*K[121] + pA[14]*K[240] + pA[15]*K[257] + pA[16]*K[274]
                        TA[37] = pA[4]*K[71] + pA[5]*K[88] + pA[7]*K[122] + pA[14]*K[241] + pA[15]*K[258] + pA[16]*K[275]
                        TA[38] = pA[4]*K[72] + pA[5]*K[89] + pA[7]*K[123] + pA[14]*K[242] + pA[15]*K[259] + pA[16]*K[276]
                        TA[39] = pA[4]*K[73] + pA[5]*K[90] + pA[7]*K[124] + pA[14]*K[243] + pA[15]*K[260] + pA[16]*K[277]
                        TA[40] = pA[4]*K[74] + pA[5]*K[91] + pA[7]*K[125] + pA[14]*K[244] + pA[15]*K[261] + pA[16]*K[278]
                        TA[41] = pA[4]*K[75] + pA[5]*K[92] + pA[7]*K[126] + pA[14]*K[245] + pA[15]*K[262] + pA[16]*K[279]
                        TA[42] = pA[4]*K[76] + pA[5]*K[93] + pA[7]*K[127] + pA[14]*K[246] + pA[15]*K[263] + pA[16]*K[280]
                        TA[43] = pA[4]*K[77] + pA[5]*K[94] + pA[7]*K[128] + pA[14]*K[247] + pA[15]*K[264] + pA[16]*K[281]
                        TA[44] = pA[4]*K[78] + pA[5]*K[95] + pA[7]*K[129] + pA[14]*K[248] + pA[15]*K[265] + pA[16]*K[282]
                        TA[45] = pA[4]*K[79] + pA[5]*K[96] + pA[7]*K[130] + pA[14]*K[249] + pA[15]*K[266] + pA[16]*K[283]
                        TA[46] = pA[4]*K[80] + pA[5]*K[97] + pA[7]*K[131] + pA[14]*K[250] + pA[15]*K[267] + pA[16]*K[284]
                        TA[47] = pA[4]*K[81] + pA[5]*K[98] + pA[7]*K[132] + pA[14]*K[251] + pA[15]*K[268] + pA[16]*K[285]
                        TA[48] = pA[4]*K[82] + pA[5]*K[99] + pA[7]*K[133] + pA[14]*K[252] + pA[15]*K[269] + pA[16]*K[286]
                        TA[49] = pA[4]*K[83] + pA[5]*K[100] + pA[7]*K[134] + pA[14]*K[253] + pA[15]*K[270] + pA[16]*K[287]
                        TA[50] = pA[4]*K[84] + pA[5]*K[101] + pA[7]*K[135] + pA[14]*K[254] + pA[15]*K[271] + pA[16]*K[288]
                        for l in range(n):
                            gBu = gtab[0, l]
                            gBu_y = gtab[1, l]
                            gBu_yy = gtab[2, l]
                            gBv = gtab[3, l]
                            gBv_y = gtab[4, l]
                            gBv_yy = gtab[5, l]
                            gBw = gtab[6, l]
                            gBw_y = gtab[7, l]
                            gBw_yy = gtab[8, l]
                            for k in range(m):
                                col = col0 + DOF*(l*m + k)
                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue
                                fBu = ftab[0, k]
                                fBu_x = ftab[1, k]
                                fBu_xx = ftab[2, k]
                                fBv = ftab[3, k]
                                fBv_x = ftab[4, k]
                                fBv_xx = ftab[5, k]
                                fBw = ftab[6, k]
                                fBw_x = ftab[7, k]
                                fBw_xx = ftab[8, k]
                                pB[0] = fBu_x*gBu
                                pB[1] = fBu*gBu_y
                                pB[2] = fBv_x*gBv
                                pB[3] = fBv*gBv_y
                                pB[4] = fBw_x*gBw
                                pB[5] = fBw*gBw_y
                                pB[6] = fBv*gBv
                                pB[7] = fBw*gBw
                                pB[8] = fBu_xx*gBu
                                pB[9] = fBu*gBu_yy
                                pB[10] = fBu_x*gBu_y
                                pB[11] = fBv_xx*gBv
                                pB[12] = fBv*gBv_yy
                                pB[13] = fBv_x*gBv_y
                                pB[14] = fBw_xx*gBw
                                pB[15] = fBw*gBw_yy
                                pB[16] = fBw_x*gBw_y
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1] + TA[8]*pB[8] + TA[9]*pB[9] + TA[10]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[2]*pB[2] + TA[3]*pB[3] + TA[6]*pB[6] + TA[11]*pB[11] + TA[12]*pB[12] + TA[13]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[4]*pB[4] + TA[5]*pB[5] + TA[7]*pB[7] + TA[14]*pB[14] + TA[15]*pB[15] + TA[16]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[17]*pB[0] + TA[18]*pB[1] + TA[25]*pB[8] + TA[26]*pB[9] + TA[27]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[19]*pB[2] + TA[20]*pB[3] + TA[23]*pB[6] + TA[28]*pB[11] + TA[29]*pB[12] + TA[30]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[21]*pB[4] + TA[22]*pB[5] + TA[24]*pB[7] + TA[31]*pB[14] + TA[32]*pB[15] + TA[33]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[34]*pB[0] + TA[35]*pB[1] + TA[42]*pB[8] + TA[43]*pB[9] + TA[44]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[36]*pB[2] + TA[37]*pB[3] + TA[40]*pB[6] + TA[45]*pB[11] + TA[46]*pB[12] + TA[47]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[38]*pB[4] + TA[39]*pB[5] + TA[41]*pB[7] + TA[48]*pB[14] + TA[49]*pB[15] + TA[50]*pB[16])

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC


def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    r"""Geometric stiffness matrix of the linear stress state

    Integrates numerically the geometric stiffness matrix of the stresses
    of the linear strains at ``cs``, superposed to the constant membrane
    state ``Nxx0``, ``Nyy0``, ``Nxy0``. Besides the blocks of the membrane
    stresses of :mod:`panels.models.cylshell_clpt_gl`, the moments couple
    the first derivatives and the values of ``u``, ``v`` and ``w`` with
    their second derivatives. The matrix is homogeneous of degree one in
    ``cs``, as the linear buckling eigenvalue problem requires.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements (the rotations in radians).
    Finput : array-like
        Constitutive matrix of the generalized strains of the model, shape
        ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        6, 6)`` with one matrix per integration point.
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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients and second derivatives
    of `u`, `v`, `w` and the values of `v` and `w`), the strains
    `\boldsymbol{\varepsilon}(\boldsymbol{d})`,
    `\boldsymbol{B} = \partial \boldsymbol{\varepsilon} / \partial
    \boldsymbol{d}`, the stresses `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{H}(\boldsymbol{\sigma},
    \boldsymbol{d}) = \sum_r \sigma_r \, \partial^2 \varepsilon_r / \partial
    \boldsymbol{d}^2`:

    .. math::

        \boldsymbol{K}_G = \int \boldsymbol{\Phi}^\top
        \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{0}) \boldsymbol{\Phi}
        \, dA

    with `\boldsymbol{\sigma} = \{N_{xx0}, N_{yy0}, N_{xy0}, 0, 0, 0\}^\top +
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
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, p, q, r, s, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w, w_xx, w_yy, w_xy
    cdef double e[6]
    cdef double sg[6]
    cdef double fAu, fAu_x, fAu_xx, fAv, fAv_x, fAv_xx, fAw
    cdef double fAw_x, fAw_xx, gAu, gAu_y, gAu_yy, gAv, gAv_y
    cdef double gAv_yy, gAw, gAw_y, gAw_yy, fBu, fBu_x, fBu_xx
    cdef double fBv, fBv_x, fBv_xx, fBw, fBw_x, fBw_xx, gBu
    cdef double gBu_y, gBu_yy, gBv, gBv_y, gBv_yy, gBw, gBw_y
    cdef double gBw_yy
    cdef double K[289]
    cdef double TA[51]
    cdef double pA[17]
    cdef double pB[17]

    cdef long [::1] kGr, kGc
    cdef double [::1] kGv

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
    radius = shell.r
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

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

    ftab = np.zeros((9, m), dtype=DOUBLE)
    gtab = np.zeros((9, n), dtype=DOUBLE)

    fdim = 9*m*m*n*n
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
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = (2/a)*fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = (2/a)*(2/a)*fpp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[3, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[5, i] = (2/a)*(2/a)*fpp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[8, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = (2/b)*(2/b)*fpp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[3, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[5, j] = (2/b)*(2/b)*fpp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[6, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[8, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

                # derivatives of the linear strains at the point
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w = 0.
                w_xx = 0.
                w_yy = 0.
                w_xy = 0.
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAu_yy = gtab[2, j]
                    gAv = gtab[3, j]
                    gAv_y = gtab[4, j]
                    gAv_yy = gtab[5, j]
                    gAw = gtab[6, j]
                    gAw_y = gtab[7, j]
                    gAw_yy = gtab[8, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAu_xx = ftab[2, i]
                        fAv = ftab[3, i]
                        fAv_x = ftab[4, i]
                        fAv_xx = ftab[5, i]
                        fAw = ftab[6, i]
                        fAw_x = ftab[7, i]
                        fAw_xx = ftab[8, i]
                        col = col0 + DOF*(j*m + i)
                        u_x += fAu_x*gAu*cs[col+0]
                        u_y += fAu*gAu_y*cs[col+0]
                        v_x += fAv_x*gAv*cs[col+1]
                        v_y += fAv*gAv_y*cs[col+1]
                        w += fAw*gAw*cs[col+2]
                        w_xx += fAw_xx*gAw*cs[col+2]
                        w_yy += fAw*gAw_yy*cs[col+2]
                        w_xy += fAw_x*gAw_y*cs[col+2]

                # linear strains and stresses
                e[0] = u_x
                e[1] = v_y + w/radius
                e[2] = u_y + v_x
                e[3] = -w_xx
                e[4] = -w_yy + v_y/radius
                e[5] = -2*w_xy - 0.5*u_y/radius + 1.5*v_x/radius
                for r in range(6):
                    sg[r] = 0.
                    for s in range(6):
                        sg[r] += F[r*6 + s]*e[s]
                sg[0] += Nxx0
                sg[1] += Nyy0
                sg[2] += Nxy0

                # K = [H0(sigma)]
                K[0] = sg[0] - sg[4]/radius
                K[1] = sg[2] - 0.5*sg[5]/radius
                K[2] = 2*sg[5]/radius
                K[3] = 3*sg[4]/radius
                K[7] = 2*sg[4]/(radius*radius)
                K[14] = -sg[3]
                K[15] = -sg[4]
                K[16] = -2*sg[5]
                K[17] = sg[2] - 0.5*sg[5]/radius
                K[18] = sg[1] - 2*sg[4]/radius
                K[19] = -sg[4]/radius
                K[34] = 2*sg[5]/radius
                K[35] = -sg[4]/radius
                K[36] = sg[0] - sg[4]/radius
                K[37] = sg[2] + 1.5*sg[5]/radius
                K[41] = sg[2]/radius + 1.5*sg[5]/(radius*radius)
                K[51] = 3*sg[4]/radius
                K[53] = sg[2] + 1.5*sg[5]/radius
                K[54] = sg[1] + 2*sg[4]/radius
                K[58] = sg[1]/radius + sg[4]/(radius*radius)
                K[65] = -sg[3]
                K[66] = -sg[4]
                K[67] = -2*sg[5]
                K[72] = sg[0] - sg[4]/radius
                K[73] = sg[2] + 1.5*sg[5]/radius
                K[74] = -sg[2]/radius - 1.5*sg[5]/(radius*radius)
                K[76] = sg[3]
                K[77] = sg[4]
                K[78] = 2*sg[5]
                K[89] = sg[2] + 1.5*sg[5]/radius
                K[90] = sg[1] + 2*sg[4]/radius
                K[91] = -sg[1]/radius - sg[4]/(radius*radius)
                K[96] = sg[3]
                K[97] = sg[4]
                K[98] = 2*sg[5]
                K[106] = -sg[2]/radius - 1.5*sg[5]/(radius*radius)
                K[107] = -sg[1]/radius - sg[4]/(radius*radius)
                K[108] = sg[1]/(radius*radius)
                K[113] = -sg[3]/radius
                K[114] = -sg[4]/radius
                K[115] = -2*sg[5]/radius
                K[119] = 2*sg[4]/(radius*radius)
                K[121] = sg[2]/radius + 1.5*sg[5]/(radius*radius)
                K[122] = sg[1]/radius + sg[4]/(radius*radius)
                K[126] = sg[1]/(radius*radius)
                K[133] = -sg[3]/radius
                K[134] = -sg[4]/radius
                K[135] = -2*sg[5]/radius
                K[140] = sg[3]
                K[157] = sg[4]
                K[174] = 2*sg[5]
                K[192] = sg[3]
                K[193] = -sg[3]/radius
                K[209] = sg[4]
                K[210] = -sg[4]/radius
                K[226] = 2*sg[5]
                K[227] = -2*sg[5]/radius
                K[238] = -sg[3]
                K[241] = -sg[3]
                K[245] = -sg[3]/radius
                K[255] = -sg[4]
                K[258] = -sg[4]
                K[262] = -sg[4]/radius
                K[272] = -2*sg[5]
                K[275] = -2*sg[5]
                K[279] = -2*sg[5]/radius

                c = -1
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAu_yy = gtab[2, j]
                    gAv = gtab[3, j]
                    gAv_y = gtab[4, j]
                    gAv_yy = gtab[5, j]
                    gAw = gtab[6, j]
                    gAw_y = gtab[7, j]
                    gAw_yy = gtab[8, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAu_xx = ftab[2, i]
                        fAv = ftab[3, i]
                        fAv_x = ftab[4, i]
                        fAv_xx = ftab[5, i]
                        fAw = ftab[6, i]
                        fAw_x = ftab[7, i]
                        fAw_xx = ftab[8, i]
                        row = row0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAu_xx*gAu
                        pA[9] = fAu*gAu_yy
                        pA[10] = fAu_x*gAu_y
                        pA[11] = fAv_xx*gAv
                        pA[12] = fAv*gAv_yy
                        pA[13] = fAv_x*gAv_y
                        pA[14] = fAw_xx*gAw
                        pA[15] = fAw*gAw_yy
                        pA[16] = fAw_x*gAw_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[17]
                        TA[1] = pA[0]*K[1] + pA[1]*K[18]
                        TA[2] = pA[0]*K[2] + pA[1]*K[19]
                        TA[3] = pA[0]*K[3]
                        TA[4] = pA[8]*K[140] + pA[9]*K[157] + pA[10]*K[174]
                        TA[7] = pA[0]*K[7]
                        TA[14] = pA[0]*K[14]
                        TA[15] = pA[0]*K[15]
                        TA[16] = pA[0]*K[16]
                        TA[17] = pA[2]*K[34] + pA[3]*K[51]
                        TA[18] = pA[2]*K[35]
                        TA[19] = pA[2]*K[36] + pA[3]*K[53]
                        TA[20] = pA[2]*K[37] + pA[3]*K[54]
                        TA[21] = pA[6]*K[106]
                        TA[22] = pA[6]*K[107] + pA[11]*K[192] + pA[12]*K[209] + pA[13]*K[226]
                        TA[23] = pA[6]*K[108] + pA[11]*K[193] + pA[12]*K[210] + pA[13]*K[227]
                        TA[24] = pA[2]*K[41] + pA[3]*K[58]
                        TA[28] = pA[6]*K[113]
                        TA[29] = pA[6]*K[114]
                        TA[30] = pA[6]*K[115]
                        TA[31] = pA[3]*K[65]
                        TA[32] = pA[3]*K[66]
                        TA[33] = pA[3]*K[67]
                        TA[34] = pA[7]*K[119] + pA[14]*K[238] + pA[15]*K[255] + pA[16]*K[272]
                        TA[36] = pA[7]*K[121]
                        TA[37] = pA[7]*K[122] + pA[14]*K[241] + pA[15]*K[258] + pA[16]*K[275]
                        TA[38] = pA[4]*K[72] + pA[5]*K[89]
                        TA[39] = pA[4]*K[73] + pA[5]*K[90]
                        TA[40] = pA[4]*K[74] + pA[5]*K[91]
                        TA[41] = pA[7]*K[126] + pA[14]*K[245] + pA[15]*K[262] + pA[16]*K[279]
                        TA[42] = pA[4]*K[76]
                        TA[43] = pA[4]*K[77]
                        TA[44] = pA[4]*K[78]
                        TA[45] = pA[5]*K[96]
                        TA[46] = pA[5]*K[97]
                        TA[47] = pA[5]*K[98]
                        TA[48] = pA[7]*K[133]
                        TA[49] = pA[7]*K[134]
                        TA[50] = pA[7]*K[135]
                        for l in range(n):
                            gBu = gtab[0, l]
                            gBu_y = gtab[1, l]
                            gBu_yy = gtab[2, l]
                            gBv = gtab[3, l]
                            gBv_y = gtab[4, l]
                            gBv_yy = gtab[5, l]
                            gBw = gtab[6, l]
                            gBw_y = gtab[7, l]
                            gBw_yy = gtab[8, l]
                            for k in range(m):
                                col = col0 + DOF*(l*m + k)
                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue
                                fBu = ftab[0, k]
                                fBu_x = ftab[1, k]
                                fBu_xx = ftab[2, k]
                                fBv = ftab[3, k]
                                fBv_x = ftab[4, k]
                                fBv_xx = ftab[5, k]
                                fBw = ftab[6, k]
                                fBw_x = ftab[7, k]
                                fBw_xx = ftab[8, k]
                                pB[0] = fBu_x*gBu
                                pB[1] = fBu*gBu_y
                                pB[2] = fBv_x*gBv
                                pB[3] = fBv*gBv_y
                                pB[4] = fBw_x*gBw
                                pB[5] = fBw*gBw_y
                                pB[6] = fBv*gBv
                                pB[7] = fBw*gBw
                                pB[8] = fBu_xx*gBu
                                pB[9] = fBu*gBu_yy
                                pB[10] = fBu_x*gBu_y
                                pB[11] = fBv_xx*gBv
                                pB[12] = fBv*gBv_yy
                                pB[13] = fBv_x*gBv_y
                                pB[14] = fBw_xx*gBw
                                pB[15] = fBw*gBw_yy
                                pB[16] = fBw_x*gBw_y
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
                                    kGr[c] = row+1
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[17]*pB[0] + TA[18]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[19]*pB[2] + TA[20]*pB[3] + TA[23]*pB[6] + TA[28]*pB[11] + TA[29]*pB[12] + TA[30]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[21]*pB[4] + TA[22]*pB[5] + TA[24]*pB[7] + TA[31]*pB[14] + TA[32]*pB[15] + TA[33]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[34]*pB[0] + TA[42]*pB[8] + TA[43]*pB[9] + TA[44]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[36]*pB[2] + TA[37]*pB[3] + TA[40]*pB[6] + TA[45]*pB[11] + TA[46]*pB[12] + TA[47]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[38]*pB[4] + TA[39]*pB[5] + TA[41]*pB[7] + TA[48]*pB[14] + TA[49]*pB[15] + TA[50]*pB[16])

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
        ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        6, 6)`` with one matrix per integration point.
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
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, p, q, r, s, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w, u_xx, u_yy, u_xy, v_xx, v_yy, v_xy, w_xx, w_yy, w_xy
    cdef double e[6]
    cdef double sg[6]
    cdef double S[17]
    cdef double pA[17]
    cdef double fAu, fAu_x, fAu_xx, fAv, fAv_x, fAv_xx, fAw
    cdef double fAw_x, fAw_xx, gAu, gAu_y, gAu_yy, gAv, gAv_y
    cdef double gAv_yy, gAw, gAw_y, gAw_yy

    cdef double [::1] fint

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
    radius = shell.r
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

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

    ftab = np.zeros((9, m), dtype=DOUBLE)
    gtab = np.zeros((9, n), dtype=DOUBLE)

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
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = (2/a)*fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = (2/a)*(2/a)*fpp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[3, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[5, i] = (2/a)*(2/a)*fpp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[8, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = (2/b)*(2/b)*fpp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[3, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[5, j] = (2/b)*(2/b)*fpp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[6, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[8, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

                # generalized derivatives at the point
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w_x = 0.
                w_y = 0.
                v = 0.
                w = 0.
                u_xx = 0.
                u_yy = 0.
                u_xy = 0.
                v_xx = 0.
                v_yy = 0.
                v_xy = 0.
                w_xx = 0.
                w_yy = 0.
                w_xy = 0.
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAu_yy = gtab[2, j]
                    gAv = gtab[3, j]
                    gAv_y = gtab[4, j]
                    gAv_yy = gtab[5, j]
                    gAw = gtab[6, j]
                    gAw_y = gtab[7, j]
                    gAw_yy = gtab[8, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAu_xx = ftab[2, i]
                        fAv = ftab[3, i]
                        fAv_x = ftab[4, i]
                        fAv_xx = ftab[5, i]
                        fAw = ftab[6, i]
                        fAw_x = ftab[7, i]
                        fAw_xx = ftab[8, i]
                        col = col0 + DOF*(j*m + i)
                        u_x += fAu_x*gAu*cs[col+0]
                        u_y += fAu*gAu_y*cs[col+0]
                        v_x += fAv_x*gAv*cs[col+1]
                        v_y += fAv*gAv_y*cs[col+1]
                        w_x += fAw_x*gAw*cs[col+2]
                        w_y += fAw*gAw_y*cs[col+2]
                        v += fAv*gAv*cs[col+1]
                        w += fAw*gAw*cs[col+2]
                        u_xx += fAu_xx*gAu*cs[col+0]
                        u_yy += fAu*gAu_yy*cs[col+0]
                        u_xy += fAu_x*gAu_y*cs[col+0]
                        v_xx += fAv_xx*gAv*cs[col+1]
                        v_yy += fAv*gAv_yy*cs[col+1]
                        v_xy += fAv_x*gAv_y*cs[col+1]
                        w_xx += fAw_xx*gAw*cs[col+2]
                        w_yy += fAw*gAw_yy*cs[col+2]
                        w_xy += fAw_x*gAw_y*cs[col+2]

                # strains, stresses and S = [B]^T sigma
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
                e[3] = u_x*v_xx*w_y - u_x*v_y*w_xx - u_x*w_xx - u_xx*v_x*w_y + u_xx*v_y*w_x + u_xx*w_x + u_y*v_x*w_xx - u_y*v_xx*w_x + v_xx*w_y - v_y*w_xx - w_xx - u_x*v*v_xx/radius - u_x*w*w_xx/radius + u_xx*v*v_x/radius + u_xx*w*w_x/radius - v*v_xx/radius - w*w_xx/radius
                e[4] = -u_x*v_y*w_yy + u_x*v_yy*w_y - u_x*w_yy + u_y*v_x*w_yy - u_y*v_yy*w_x - u_yy*v_x*w_y + u_yy*v_y*w_x + u_yy*w_x - v_y*w_yy + v_yy*w_y - w_yy - 0.5*(u_x*u_x)/radius - u_x*v*v_yy/radius + 2*u_x*(v_y*v_y)/radius + 3*u_x*v_y/radius - u_x*w*w_yy/radius + 2*u_x*(w_y*w_y)/radius - (u_y*u_y)/radius - 2*u_y*v_x*v_y/radius - u_y*v_x/radius - 2*u_y*w_x*w_y/radius + u_yy*v*v_x/radius + u_yy*w*w_x/radius - v*v_yy/radius - 0.5*(v_x*v_x)/radius + (v_y*v_y)/radius + v_y/radius - w*w_yy/radius - 0.5*(w_x*w_x)/radius + (w_y*w_y)/radius - 3*u_x*v*w_y/(radius*radius) + 3*u_x*v_y*w/(radius*radius) + 2*u_x*w/(radius*radius) + u_y*v*w_x/(radius*radius) - u_y*v_x*w/(radius*radius) - v*w_y/(radius*radius) + v_y*w/(radius*radius) + u_x*(v*v)/(radius*radius*radius) + u_x*(w*w)/(radius*radius*radius)
                e[5] = 2*u_x*v_xy*w_y - 2*u_x*v_y*w_xy - 2*u_x*w_xy - 2*u_xy*v_x*w_y + 2*u_xy*v_y*w_x + 2*u_xy*w_x + 2*u_y*v_x*w_xy - 2*u_y*v_xy*w_x + 2*v_xy*w_y - 2*v_y*w_xy - 2*w_xy - 0.5*u_x*u_y/radius - 2*u_x*v*v_xy/radius + 2*u_x*v_x*v_y/radius + 2*u_x*v_x/radius - 2*u_x*w*w_xy/radius + 2*u_x*w_x*w_y/radius + 2*u_xy*v*v_x/radius + 2*u_xy*w*w_x/radius - 2*u_y*(v_x*v_x)/radius - 2*u_y*(w_x*w_x)/radius - 0.5*u_y/radius - 2*v*v_xy/radius + 1.5*v_x*v_y/radius + 1.5*v_x/radius - 2*w*w_xy/radius + 1.5*w_x*w_y/radius - 2*u_x*v*w_x/(radius*radius) + 2*u_x*v_x*w/(radius*radius) - 1.5*v*w_x/(radius*radius) + 1.5*v_x*w/(radius*radius)
                for r in range(6):
                    sg[r] = 0.
                    for s in range(6):
                        sg[r] += F[r*6 + s]*e[s]
                S[0] = sg[0]*u_x + sg[0] + sg[2]*u_y + sg[3]*v_xx*w_y - sg[3]*v_y*w_xx - sg[3]*w_xx - sg[4]*v_y*w_yy + sg[4]*v_yy*w_y - sg[4]*w_yy + 2*sg[5]*v_xy*w_y - 2*sg[5]*v_y*w_xy - 2*sg[5]*w_xy - sg[3]*v*v_xx/radius - sg[3]*w*w_xx/radius - sg[4]*u_x/radius - sg[4]*v*v_yy/radius + 2*sg[4]*(v_y*v_y)/radius + 3*sg[4]*v_y/radius - sg[4]*w*w_yy/radius + 2*sg[4]*(w_y*w_y)/radius - 0.5*sg[5]*u_y/radius - 2*sg[5]*v*v_xy/radius + 2*sg[5]*v_x*v_y/radius + 2*sg[5]*v_x/radius - 2*sg[5]*w*w_xy/radius + 2*sg[5]*w_x*w_y/radius - 3*sg[4]*v*w_y/(radius*radius) + 3*sg[4]*v_y*w/(radius*radius) + 2*sg[4]*w/(radius*radius) - 2*sg[5]*v*w_x/(radius*radius) + 2*sg[5]*v_x*w/(radius*radius) + sg[4]*(v*v)/(radius*radius*radius) + sg[4]*(w*w)/(radius*radius*radius)
                S[1] = sg[1]*u_y + sg[2]*u_x + sg[2] + sg[3]*v_x*w_xx - sg[3]*v_xx*w_x + sg[4]*v_x*w_yy - sg[4]*v_yy*w_x + 2*sg[5]*v_x*w_xy - 2*sg[5]*v_xy*w_x - 2*sg[4]*u_y/radius - 2*sg[4]*v_x*v_y/radius - sg[4]*v_x/radius - 2*sg[4]*w_x*w_y/radius - 0.5*sg[5]*u_x/radius - 2*sg[5]*(v_x*v_x)/radius - 2*sg[5]*(w_x*w_x)/radius - 0.5*sg[5]/radius + sg[4]*v*w_x/(radius*radius) - sg[4]*v_x*w/(radius*radius)
                S[2] = sg[0]*v_x + sg[2]*v_y + sg[2] - sg[3]*u_xx*w_y + sg[3]*u_y*w_xx + sg[4]*u_y*w_yy - sg[4]*u_yy*w_y - 2*sg[5]*u_xy*w_y + 2*sg[5]*u_y*w_xy + sg[2]*w/radius + sg[3]*u_xx*v/radius - 2*sg[4]*u_y*v_y/radius - sg[4]*u_y/radius + sg[4]*u_yy*v/radius - sg[4]*v_x/radius + 2*sg[5]*u_x*v_y/radius + 2*sg[5]*u_x/radius + 2*sg[5]*u_xy*v/radius - 4*sg[5]*u_y*v_x/radius + 1.5*sg[5]*v_y/radius + 1.5*sg[5]/radius - sg[4]*u_y*w/(radius*radius) + 2*sg[5]*u_x*w/(radius*radius) + 1.5*sg[5]*w/(radius*radius)
                S[3] = sg[1]*v_y + sg[1] + sg[2]*v_x - sg[3]*u_x*w_xx + sg[3]*u_xx*w_x - sg[3]*w_xx - sg[4]*u_x*w_yy + sg[4]*u_yy*w_x - sg[4]*w_yy - 2*sg[5]*u_x*w_xy + 2*sg[5]*u_xy*w_x - 2*sg[5]*w_xy + sg[1]*w/radius + 4*sg[4]*u_x*v_y/radius + 3*sg[4]*u_x/radius - 2*sg[4]*u_y*v_x/radius + 2*sg[4]*v_y/radius + sg[4]/radius + 2*sg[5]*u_x*v_x/radius + 1.5*sg[5]*v_x/radius + 3*sg[4]*u_x*w/(radius*radius) + sg[4]*w/(radius*radius)
                S[4] = sg[0]*w_x + sg[2]*w_y + sg[3]*u_xx*v_y + sg[3]*u_xx - sg[3]*u_y*v_xx - sg[4]*u_y*v_yy + sg[4]*u_yy*v_y + sg[4]*u_yy + 2*sg[5]*u_xy*v_y + 2*sg[5]*u_xy - 2*sg[5]*u_y*v_xy - sg[2]*v/radius + sg[3]*u_xx*w/radius - 2*sg[4]*u_y*w_y/radius + sg[4]*u_yy*w/radius - sg[4]*w_x/radius + 2*sg[5]*u_x*w_y/radius + 2*sg[5]*u_xy*w/radius - 4*sg[5]*u_y*w_x/radius + 1.5*sg[5]*w_y/radius + sg[4]*u_y*v/(radius*radius) - 2*sg[5]*u_x*v/(radius*radius) - 1.5*sg[5]*v/(radius*radius)
                S[5] = sg[1]*w_y + sg[2]*w_x + sg[3]*u_x*v_xx - sg[3]*u_xx*v_x + sg[3]*v_xx + sg[4]*u_x*v_yy - sg[4]*u_yy*v_x + sg[4]*v_yy + 2*sg[5]*u_x*v_xy - 2*sg[5]*u_xy*v_x + 2*sg[5]*v_xy - sg[1]*v/radius + 4*sg[4]*u_x*w_y/radius - 2*sg[4]*u_y*w_x/radius + 2*sg[4]*w_y/radius + 2*sg[5]*u_x*w_x/radius + 1.5*sg[5]*w_x/radius - 3*sg[4]*u_x*v/(radius*radius) - sg[4]*v/(radius*radius)
                S[6] = -sg[1]*w_y/radius - sg[2]*w_x/radius - sg[3]*u_x*v_xx/radius + sg[3]*u_xx*v_x/radius - sg[3]*v_xx/radius - sg[4]*u_x*v_yy/radius + sg[4]*u_yy*v_x/radius - sg[4]*v_yy/radius - 2*sg[5]*u_x*v_xy/radius + 2*sg[5]*u_xy*v_x/radius - 2*sg[5]*v_xy/radius + sg[1]*v/(radius*radius) - 3*sg[4]*u_x*w_y/(radius*radius) + sg[4]*u_y*w_x/(radius*radius) - sg[4]*w_y/(radius*radius) - 2*sg[5]*u_x*w_x/(radius*radius) - 1.5*sg[5]*w_x/(radius*radius) + 2*sg[4]*u_x*v/(radius*radius*radius)
                S[7] = sg[1]*v_y/radius + sg[1]/radius + sg[2]*v_x/radius - sg[3]*u_x*w_xx/radius + sg[3]*u_xx*w_x/radius - sg[3]*w_xx/radius - sg[4]*u_x*w_yy/radius + sg[4]*u_yy*w_x/radius - sg[4]*w_yy/radius - 2*sg[5]*u_x*w_xy/radius + 2*sg[5]*u_xy*w_x/radius - 2*sg[5]*w_xy/radius + sg[1]*w/(radius*radius) + 3*sg[4]*u_x*v_y/(radius*radius) + 2*sg[4]*u_x/(radius*radius) - sg[4]*u_y*v_x/(radius*radius) + sg[4]*v_y/(radius*radius) + 2*sg[5]*u_x*v_x/(radius*radius) + 1.5*sg[5]*v_x/(radius*radius) + 2*sg[4]*u_x*w/(radius*radius*radius)
                S[8] = -sg[3]*v_x*w_y + sg[3]*v_y*w_x + sg[3]*w_x + sg[3]*v*v_x/radius + sg[3]*w*w_x/radius
                S[9] = -sg[4]*v_x*w_y + sg[4]*v_y*w_x + sg[4]*w_x + sg[4]*v*v_x/radius + sg[4]*w*w_x/radius
                S[10] = -2*sg[5]*v_x*w_y + 2*sg[5]*v_y*w_x + 2*sg[5]*w_x + 2*sg[5]*v*v_x/radius + 2*sg[5]*w*w_x/radius
                S[11] = sg[3]*u_x*w_y - sg[3]*u_y*w_x + sg[3]*w_y - sg[3]*u_x*v/radius - sg[3]*v/radius
                S[12] = sg[4]*u_x*w_y - sg[4]*u_y*w_x + sg[4]*w_y - sg[4]*u_x*v/radius - sg[4]*v/radius
                S[13] = 2*sg[5]*u_x*w_y - 2*sg[5]*u_y*w_x + 2*sg[5]*w_y - 2*sg[5]*u_x*v/radius - 2*sg[5]*v/radius
                S[14] = -sg[3]*u_x*v_y - sg[3]*u_x + sg[3]*u_y*v_x - sg[3]*v_y - sg[3] - sg[3]*u_x*w/radius - sg[3]*w/radius
                S[15] = -sg[4]*u_x*v_y - sg[4]*u_x + sg[4]*u_y*v_x - sg[4]*v_y - sg[4] - sg[4]*u_x*w/radius - sg[4]*w/radius
                S[16] = -2*sg[5]*u_x*v_y - 2*sg[5]*u_x + 2*sg[5]*u_y*v_x - 2*sg[5]*v_y - 2*sg[5] - 2*sg[5]*u_x*w/radius - 2*sg[5]*w/radius

                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAu_yy = gtab[2, j]
                    gAv = gtab[3, j]
                    gAv_y = gtab[4, j]
                    gAv_yy = gtab[5, j]
                    gAw = gtab[6, j]
                    gAw_y = gtab[7, j]
                    gAw_yy = gtab[8, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAu_xx = ftab[2, i]
                        fAv = ftab[3, i]
                        fAv_x = ftab[4, i]
                        fAv_xx = ftab[5, i]
                        fAw = ftab[6, i]
                        fAw_x = ftab[7, i]
                        fAw_xx = ftab[8, i]
                        col = col0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAu_xx*gAu
                        pA[9] = fAu*gAu_yy
                        pA[10] = fAu_x*gAu_y
                        pA[11] = fAv_xx*gAv
                        pA[12] = fAv*gAv_yy
                        pA[13] = fAv_x*gAv_y
                        pA[14] = fAw_xx*gAw
                        pA[15] = fAw*gAw_yy
                        pA[16] = fAw_x*gAw_y
                        fint[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1] + pA[8]*S[8] + pA[9]*S[9] + pA[10]*S[10])
                        fint[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[6]*S[6] + pA[11]*S[11] + pA[12]*S[12] + pA[13]*S[13])
                        fint[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[7]*S[7] + pA[14]*S[14] + pA[15]*S[15] + pA[16]*S[16])

    return fint


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
        ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        6, 6)`` with one matrix per integration point.
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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients and second derivatives
    of `u`, `v`, `w` and the values of `v` and `w`), the strains
    `\boldsymbol{\varepsilon}(\boldsymbol{d})`,
    `\boldsymbol{B} = \partial \boldsymbol{\varepsilon} / \partial
    \boldsymbol{d}`, the stresses `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{H}(\boldsymbol{\sigma},
    \boldsymbol{d}) = \sum_r \sigma_r \, \partial^2 \varepsilon_r / \partial
    \boldsymbol{d}^2`, and
    `\bar{\boldsymbol{\sigma}}` the constant membrane stresses:

    .. math::

        \boldsymbol{K}_T \boldsymbol{v} = \int \boldsymbol{\Phi}^\top
        \left[\boldsymbol{B}^\top \boldsymbol{F} \boldsymbol{B} +
        \boldsymbol{H}(\boldsymbol{\sigma} + \bar{\boldsymbol{\sigma}},
        \boldsymbol{d})\right] \boldsymbol{\Phi} \boldsymbol{v} \, dA

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
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, k, l, c, p, q, r, s, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w
    cdef double u_xx, u_yy, u_xy, v_xx, v_yy, v_xy, w_xx, w_yy
    cdef double w_xy
    cdef double du_x, du_y, dv_x, dv_y, dw_x, dw_y, dv, dw
    cdef double du_xx, du_yy, du_xy, dv_xx, dv_yy, dv_xy, dw_xx, dw_yy
    cdef double dw_xy
    cdef double e[6]
    cdef double de[6]
    cdef double sg[6]
    cdef double dsg[6]
    cdef double S[17]
    cdef double pA[17]
    cdef double fAu, fAu_x, fAu_xx, fAv, fAv_x, fAv_xx, fAw
    cdef double fAw_x, fAw_xx, gAu, gAu_y, gAu_yy, gAv, gAv_y
    cdef double gAv_yy, gAw, gAw_y, gAw_yy

    cdef double [::1] kTv

    if cs.shape[0] != size or vs.shape[0] != size:
        raise ValueError('cs and vs must have the size %d' % size)

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
    radius = shell.r
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

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

    ftab = np.zeros((9, m), dtype=DOUBLE)
    gtab = np.zeros((9, n), dtype=DOUBLE)

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
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = (2/a)*fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = (2/a)*(2/a)*fpp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[3, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[5, i] = (2/a)*(2/a)*fpp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[6, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[8, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = (2/b)*(2/b)*fpp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[3, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[5, j] = (2/b)*(2/b)*fpp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[6, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[7, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[8, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

                # generalized derivatives of cs and vs at the point
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w_x = 0.
                w_y = 0.
                v = 0.
                w = 0.
                u_xx = 0.
                u_yy = 0.
                u_xy = 0.
                v_xx = 0.
                v_yy = 0.
                v_xy = 0.
                w_xx = 0.
                w_yy = 0.
                w_xy = 0.
                du_x = 0.
                du_y = 0.
                dv_x = 0.
                dv_y = 0.
                dw_x = 0.
                dw_y = 0.
                dv = 0.
                dw = 0.
                du_xx = 0.
                du_yy = 0.
                du_xy = 0.
                dv_xx = 0.
                dv_yy = 0.
                dv_xy = 0.
                dw_xx = 0.
                dw_yy = 0.
                dw_xy = 0.
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAu_yy = gtab[2, j]
                    gAv = gtab[3, j]
                    gAv_y = gtab[4, j]
                    gAv_yy = gtab[5, j]
                    gAw = gtab[6, j]
                    gAw_y = gtab[7, j]
                    gAw_yy = gtab[8, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAu_xx = ftab[2, i]
                        fAv = ftab[3, i]
                        fAv_x = ftab[4, i]
                        fAv_xx = ftab[5, i]
                        fAw = ftab[6, i]
                        fAw_x = ftab[7, i]
                        fAw_xx = ftab[8, i]
                        col = col0 + DOF*(j*m + i)
                        u_x += fAu_x*gAu*cs[col+0]
                        u_y += fAu*gAu_y*cs[col+0]
                        v_x += fAv_x*gAv*cs[col+1]
                        v_y += fAv*gAv_y*cs[col+1]
                        w_x += fAw_x*gAw*cs[col+2]
                        w_y += fAw*gAw_y*cs[col+2]
                        v += fAv*gAv*cs[col+1]
                        w += fAw*gAw*cs[col+2]
                        u_xx += fAu_xx*gAu*cs[col+0]
                        u_yy += fAu*gAu_yy*cs[col+0]
                        u_xy += fAu_x*gAu_y*cs[col+0]
                        v_xx += fAv_xx*gAv*cs[col+1]
                        v_yy += fAv*gAv_yy*cs[col+1]
                        v_xy += fAv_x*gAv_y*cs[col+1]
                        w_xx += fAw_xx*gAw*cs[col+2]
                        w_yy += fAw*gAw_yy*cs[col+2]
                        w_xy += fAw_x*gAw_y*cs[col+2]
                        du_x += fAu_x*gAu*vs[col+0]
                        du_y += fAu*gAu_y*vs[col+0]
                        dv_x += fAv_x*gAv*vs[col+1]
                        dv_y += fAv*gAv_y*vs[col+1]
                        dw_x += fAw_x*gAw*vs[col+2]
                        dw_y += fAw*gAw_y*vs[col+2]
                        dv += fAv*gAv*vs[col+1]
                        dw += fAw*gAw*vs[col+2]
                        du_xx += fAu_xx*gAu*vs[col+0]
                        du_yy += fAu*gAu_yy*vs[col+0]
                        du_xy += fAu_x*gAu_y*vs[col+0]
                        dv_xx += fAv_xx*gAv*vs[col+1]
                        dv_yy += fAv*gAv_yy*vs[col+1]
                        dv_xy += fAv_x*gAv_y*vs[col+1]
                        dw_xx += fAw_xx*gAw*vs[col+2]
                        dw_yy += fAw*gAw_yy*vs[col+2]
                        dw_xy += fAw_x*gAw_y*vs[col+2]

                # strains, strain increments, stresses
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
                e[3] = u_x*v_xx*w_y - u_x*v_y*w_xx - u_x*w_xx - u_xx*v_x*w_y + u_xx*v_y*w_x + u_xx*w_x + u_y*v_x*w_xx - u_y*v_xx*w_x + v_xx*w_y - v_y*w_xx - w_xx - u_x*v*v_xx/radius - u_x*w*w_xx/radius + u_xx*v*v_x/radius + u_xx*w*w_x/radius - v*v_xx/radius - w*w_xx/radius
                e[4] = -u_x*v_y*w_yy + u_x*v_yy*w_y - u_x*w_yy + u_y*v_x*w_yy - u_y*v_yy*w_x - u_yy*v_x*w_y + u_yy*v_y*w_x + u_yy*w_x - v_y*w_yy + v_yy*w_y - w_yy - 0.5*(u_x*u_x)/radius - u_x*v*v_yy/radius + 2*u_x*(v_y*v_y)/radius + 3*u_x*v_y/radius - u_x*w*w_yy/radius + 2*u_x*(w_y*w_y)/radius - (u_y*u_y)/radius - 2*u_y*v_x*v_y/radius - u_y*v_x/radius - 2*u_y*w_x*w_y/radius + u_yy*v*v_x/radius + u_yy*w*w_x/radius - v*v_yy/radius - 0.5*(v_x*v_x)/radius + (v_y*v_y)/radius + v_y/radius - w*w_yy/radius - 0.5*(w_x*w_x)/radius + (w_y*w_y)/radius - 3*u_x*v*w_y/(radius*radius) + 3*u_x*v_y*w/(radius*radius) + 2*u_x*w/(radius*radius) + u_y*v*w_x/(radius*radius) - u_y*v_x*w/(radius*radius) - v*w_y/(radius*radius) + v_y*w/(radius*radius) + u_x*(v*v)/(radius*radius*radius) + u_x*(w*w)/(radius*radius*radius)
                e[5] = 2*u_x*v_xy*w_y - 2*u_x*v_y*w_xy - 2*u_x*w_xy - 2*u_xy*v_x*w_y + 2*u_xy*v_y*w_x + 2*u_xy*w_x + 2*u_y*v_x*w_xy - 2*u_y*v_xy*w_x + 2*v_xy*w_y - 2*v_y*w_xy - 2*w_xy - 0.5*u_x*u_y/radius - 2*u_x*v*v_xy/radius + 2*u_x*v_x*v_y/radius + 2*u_x*v_x/radius - 2*u_x*w*w_xy/radius + 2*u_x*w_x*w_y/radius + 2*u_xy*v*v_x/radius + 2*u_xy*w*w_x/radius - 2*u_y*(v_x*v_x)/radius - 2*u_y*(w_x*w_x)/radius - 0.5*u_y/radius - 2*v*v_xy/radius + 1.5*v_x*v_y/radius + 1.5*v_x/radius - 2*w*w_xy/radius + 1.5*w_x*w_y/radius - 2*u_x*v*w_x/(radius*radius) + 2*u_x*v_x*w/(radius*radius) - 1.5*v*w_x/(radius*radius) + 1.5*v_x*w/(radius*radius)
                de[0] = du_x*u_x + du_x + dv_x*v_x + dw_x*w_x
                de[1] = du_y*u_y - dv*w_y/radius + dv*v/(radius*radius) + dv_y*v_y + dv_y + dv_y*w/radius + dw*v_y/radius + dw/radius + dw*w/(radius*radius) + dw_y*w_y - dw_y*v/radius
                de[2] = du_x*u_y + du_y*u_x + du_y - dv*w_x/radius + dv_x*v_y + dv_x + dv_x*w/radius + dv_y*v_x + dw*v_x/radius + dw_x*w_y - dw_x*v/radius + dw_y*w_x
                de[3] = du_x*v_xx*w_y - du_x*v_y*w_xx - du_x*w_xx - du_x*v*v_xx/radius - du_x*w*w_xx/radius - du_xx*v_x*w_y + du_xx*v_y*w_x + du_xx*w_x + du_xx*v*v_x/radius + du_xx*w*w_x/radius + du_y*v_x*w_xx - du_y*v_xx*w_x - dv*u_x*v_xx/radius + dv*u_xx*v_x/radius - dv*v_xx/radius - dv_x*u_xx*w_y + dv_x*u_y*w_xx + dv_x*u_xx*v/radius + dv_xx*u_x*w_y - dv_xx*u_y*w_x + dv_xx*w_y - dv_xx*u_x*v/radius - dv_xx*v/radius - dv_y*u_x*w_xx + dv_y*u_xx*w_x - dv_y*w_xx - dw*u_x*w_xx/radius + dw*u_xx*w_x/radius - dw*w_xx/radius + dw_x*u_xx*v_y + dw_x*u_xx - dw_x*u_y*v_xx + dw_x*u_xx*w/radius - dw_xx*u_x*v_y - dw_xx*u_x + dw_xx*u_y*v_x - dw_xx*v_y - dw_xx - dw_xx*u_x*w/radius - dw_xx*w/radius + dw_y*u_x*v_xx - dw_y*u_xx*v_x + dw_y*v_xx
                de[4] = -du_x*v_y*w_yy + du_x*v_yy*w_y - du_x*w_yy - du_x*u_x/radius - du_x*v*v_yy/radius + 2*du_x*(v_y*v_y)/radius + 3*du_x*v_y/radius - du_x*w*w_yy/radius + 2*du_x*(w_y*w_y)/radius - 3*du_x*v*w_y/(radius*radius) + 3*du_x*v_y*w/(radius*radius) + 2*du_x*w/(radius*radius) + du_x*(v*v)/(radius*radius*radius) + du_x*(w*w)/(radius*radius*radius) + du_y*v_x*w_yy - du_y*v_yy*w_x - 2*du_y*u_y/radius - 2*du_y*v_x*v_y/radius - du_y*v_x/radius - 2*du_y*w_x*w_y/radius + du_y*v*w_x/(radius*radius) - du_y*v_x*w/(radius*radius) - du_yy*v_x*w_y + du_yy*v_y*w_x + du_yy*w_x + du_yy*v*v_x/radius + du_yy*w*w_x/radius - dv*u_x*v_yy/radius + dv*u_yy*v_x/radius - dv*v_yy/radius - 3*dv*u_x*w_y/(radius*radius) + dv*u_y*w_x/(radius*radius) - dv*w_y/(radius*radius) + 2*dv*u_x*v/(radius*radius*radius) + dv_x*u_y*w_yy - dv_x*u_yy*w_y - 2*dv_x*u_y*v_y/radius - dv_x*u_y/radius + dv_x*u_yy*v/radius - dv_x*v_x/radius - dv_x*u_y*w/(radius*radius) - dv_y*u_x*w_yy + dv_y*u_yy*w_x - dv_y*w_yy + 4*dv_y*u_x*v_y/radius + 3*dv_y*u_x/radius - 2*dv_y*u_y*v_x/radius + 2*dv_y*v_y/radius + dv_y/radius + 3*dv_y*u_x*w/(radius*radius) + dv_y*w/(radius*radius) + dv_yy*u_x*w_y - dv_yy*u_y*w_x + dv_yy*w_y - dv_yy*u_x*v/radius - dv_yy*v/radius - dw*u_x*w_yy/radius + dw*u_yy*w_x/radius - dw*w_yy/radius + 3*dw*u_x*v_y/(radius*radius) + 2*dw*u_x/(radius*radius) - dw*u_y*v_x/(radius*radius) + dw*v_y/(radius*radius) + 2*dw*u_x*w/(radius*radius*radius) - dw_x*u_y*v_yy + dw_x*u_yy*v_y + dw_x*u_yy - 2*dw_x*u_y*w_y/radius + dw_x*u_yy*w/radius - dw_x*w_x/radius + dw_x*u_y*v/(radius*radius) + dw_y*u_x*v_yy - dw_y*u_yy*v_x + dw_y*v_yy + 4*dw_y*u_x*w_y/radius - 2*dw_y*u_y*w_x/radius + 2*dw_y*w_y/radius - 3*dw_y*u_x*v/(radius*radius) - dw_y*v/(radius*radius) - dw_yy*u_x*v_y - dw_yy*u_x + dw_yy*u_y*v_x - dw_yy*v_y - dw_yy - dw_yy*u_x*w/radius - dw_yy*w/radius
                de[5] = 2*du_x*v_xy*w_y - 2*du_x*v_y*w_xy - 2*du_x*w_xy - 0.5*du_x*u_y/radius - 2*du_x*v*v_xy/radius + 2*du_x*v_x*v_y/radius + 2*du_x*v_x/radius - 2*du_x*w*w_xy/radius + 2*du_x*w_x*w_y/radius - 2*du_x*v*w_x/(radius*radius) + 2*du_x*v_x*w/(radius*radius) - 2*du_xy*v_x*w_y + 2*du_xy*v_y*w_x + 2*du_xy*w_x + 2*du_xy*v*v_x/radius + 2*du_xy*w*w_x/radius + 2*du_y*v_x*w_xy - 2*du_y*v_xy*w_x - 0.5*du_y*u_x/radius - 2*du_y*(v_x*v_x)/radius - 2*du_y*(w_x*w_x)/radius - 0.5*du_y/radius - 2*dv*u_x*v_xy/radius + 2*dv*u_xy*v_x/radius - 2*dv*v_xy/radius - 2*dv*u_x*w_x/(radius*radius) - 1.5*dv*w_x/(radius*radius) - 2*dv_x*u_xy*w_y + 2*dv_x*u_y*w_xy + 2*dv_x*u_x*v_y/radius + 2*dv_x*u_x/radius + 2*dv_x*u_xy*v/radius - 4*dv_x*u_y*v_x/radius + 1.5*dv_x*v_y/radius + 1.5*dv_x/radius + 2*dv_x*u_x*w/(radius*radius) + 1.5*dv_x*w/(radius*radius) + 2*dv_xy*u_x*w_y - 2*dv_xy*u_y*w_x + 2*dv_xy*w_y - 2*dv_xy*u_x*v/radius - 2*dv_xy*v/radius - 2*dv_y*u_x*w_xy + 2*dv_y*u_xy*w_x - 2*dv_y*w_xy + 2*dv_y*u_x*v_x/radius + 1.5*dv_y*v_x/radius - 2*dw*u_x*w_xy/radius + 2*dw*u_xy*w_x/radius - 2*dw*w_xy/radius + 2*dw*u_x*v_x/(radius*radius) + 1.5*dw*v_x/(radius*radius) + 2*dw_x*u_xy*v_y + 2*dw_x*u_xy - 2*dw_x*u_y*v_xy + 2*dw_x*u_x*w_y/radius + 2*dw_x*u_xy*w/radius - 4*dw_x*u_y*w_x/radius + 1.5*dw_x*w_y/radius - 2*dw_x*u_x*v/(radius*radius) - 1.5*dw_x*v/(radius*radius) - 2*dw_xy*u_x*v_y - 2*dw_xy*u_x + 2*dw_xy*u_y*v_x - 2*dw_xy*v_y - 2*dw_xy - 2*dw_xy*u_x*w/radius - 2*dw_xy*w/radius + 2*dw_y*u_x*v_xy - 2*dw_y*u_xy*v_x + 2*dw_y*v_xy + 2*dw_y*u_x*w_x/radius + 1.5*dw_y*w_x/radius
                for r in range(6):
                    sg[r] = 0.
                    for s in range(6):
                        sg[r] += F[r*6 + s]*e[s]
                for r in range(6):
                    dsg[r] = 0.
                    for s in range(6):
                        dsg[r] += F[r*6 + s]*de[s]
                sg[0] += Nxx_cte
                sg[1] += Nyy_cte
                sg[2] += Nxy_cte

                # S = [B]^T dsigma + [H(sigma, d)] dv
                S[0] = dsg[0]*u_x + dsg[0] + dsg[2]*u_y + dsg[3]*v_xx*w_y - dsg[3]*v_y*w_xx - dsg[3]*w_xx - dsg[3]*v*v_xx/radius - dsg[3]*w*w_xx/radius - dsg[4]*v_y*w_yy + dsg[4]*v_yy*w_y - dsg[4]*w_yy - dsg[4]*u_x/radius - dsg[4]*v*v_yy/radius + 2*dsg[4]*(v_y*v_y)/radius + 3*dsg[4]*v_y/radius - dsg[4]*w*w_yy/radius + 2*dsg[4]*(w_y*w_y)/radius - 3*dsg[4]*v*w_y/(radius*radius) + 3*dsg[4]*v_y*w/(radius*radius) + 2*dsg[4]*w/(radius*radius) + dsg[4]*(v*v)/(radius*radius*radius) + dsg[4]*(w*w)/(radius*radius*radius) + 2*dsg[5]*v_xy*w_y - 2*dsg[5]*v_y*w_xy - 2*dsg[5]*w_xy - 0.5*dsg[5]*u_y/radius - 2*dsg[5]*v*v_xy/radius + 2*dsg[5]*v_x*v_y/radius + 2*dsg[5]*v_x/radius - 2*dsg[5]*w*w_xy/radius + 2*dsg[5]*w_x*w_y/radius - 2*dsg[5]*v*w_x/(radius*radius) + 2*dsg[5]*v_x*w/(radius*radius) + du_x*sg[0] - du_x*sg[4]/radius + du_y*sg[2] - 0.5*du_y*sg[5]/radius - dv*sg[3]*v_xx/radius - dv*sg[4]*v_yy/radius - 2*dv*sg[5]*v_xy/radius - 3*dv*sg[4]*w_y/(radius*radius) - 2*dv*sg[5]*w_x/(radius*radius) + 2*dv*sg[4]*v/(radius*radius*radius) + 2*dv_x*sg[5]*v_y/radius + 2*dv_x*sg[5]/radius + 2*dv_x*sg[5]*w/(radius*radius) + dv_xx*sg[3]*w_y - dv_xx*sg[3]*v/radius + 2*dv_xy*sg[5]*w_y - 2*dv_xy*sg[5]*v/radius - dv_y*sg[3]*w_xx - dv_y*sg[4]*w_yy - 2*dv_y*sg[5]*w_xy + 4*dv_y*sg[4]*v_y/radius + 3*dv_y*sg[4]/radius + 2*dv_y*sg[5]*v_x/radius + 3*dv_y*sg[4]*w/(radius*radius) + dv_yy*sg[4]*w_y - dv_yy*sg[4]*v/radius - dw*sg[3]*w_xx/radius - dw*sg[4]*w_yy/radius - 2*dw*sg[5]*w_xy/radius + 3*dw*sg[4]*v_y/(radius*radius) + 2*dw*sg[4]/(radius*radius) + 2*dw*sg[5]*v_x/(radius*radius) + 2*dw*sg[4]*w/(radius*radius*radius) + 2*dw_x*sg[5]*w_y/radius - 2*dw_x*sg[5]*v/(radius*radius) - dw_xx*sg[3]*v_y - dw_xx*sg[3] - dw_xx*sg[3]*w/radius - 2*dw_xy*sg[5]*v_y - 2*dw_xy*sg[5] - 2*dw_xy*sg[5]*w/radius + dw_y*sg[3]*v_xx + dw_y*sg[4]*v_yy + 2*dw_y*sg[5]*v_xy + 4*dw_y*sg[4]*w_y/radius + 2*dw_y*sg[5]*w_x/radius - 3*dw_y*sg[4]*v/(radius*radius) - dw_yy*sg[4]*v_y - dw_yy*sg[4] - dw_yy*sg[4]*w/radius
                S[1] = dsg[1]*u_y + dsg[2]*u_x + dsg[2] + dsg[3]*v_x*w_xx - dsg[3]*v_xx*w_x + dsg[4]*v_x*w_yy - dsg[4]*v_yy*w_x - 2*dsg[4]*u_y/radius - 2*dsg[4]*v_x*v_y/radius - dsg[4]*v_x/radius - 2*dsg[4]*w_x*w_y/radius + dsg[4]*v*w_x/(radius*radius) - dsg[4]*v_x*w/(radius*radius) + 2*dsg[5]*v_x*w_xy - 2*dsg[5]*v_xy*w_x - 0.5*dsg[5]*u_x/radius - 2*dsg[5]*(v_x*v_x)/radius - 2*dsg[5]*(w_x*w_x)/radius - 0.5*dsg[5]/radius + du_x*sg[2] - 0.5*du_x*sg[5]/radius + du_y*sg[1] - 2*du_y*sg[4]/radius + dv*sg[4]*w_x/(radius*radius) + dv_x*sg[3]*w_xx + dv_x*sg[4]*w_yy + 2*dv_x*sg[5]*w_xy - 2*dv_x*sg[4]*v_y/radius - dv_x*sg[4]/radius - 4*dv_x*sg[5]*v_x/radius - dv_x*sg[4]*w/(radius*radius) - dv_xx*sg[3]*w_x - 2*dv_xy*sg[5]*w_x - 2*dv_y*sg[4]*v_x/radius - dv_yy*sg[4]*w_x - dw*sg[4]*v_x/(radius*radius) - dw_x*sg[3]*v_xx - dw_x*sg[4]*v_yy - 2*dw_x*sg[5]*v_xy - 2*dw_x*sg[4]*w_y/radius - 4*dw_x*sg[5]*w_x/radius + dw_x*sg[4]*v/(radius*radius) + dw_xx*sg[3]*v_x + 2*dw_xy*sg[5]*v_x - 2*dw_y*sg[4]*w_x/radius + dw_yy*sg[4]*v_x
                S[2] = dsg[0]*v_x + dsg[2]*v_y + dsg[2] + dsg[2]*w/radius - dsg[3]*u_xx*w_y + dsg[3]*u_y*w_xx + dsg[3]*u_xx*v/radius + dsg[4]*u_y*w_yy - dsg[4]*u_yy*w_y - 2*dsg[4]*u_y*v_y/radius - dsg[4]*u_y/radius + dsg[4]*u_yy*v/radius - dsg[4]*v_x/radius - dsg[4]*u_y*w/(radius*radius) - 2*dsg[5]*u_xy*w_y + 2*dsg[5]*u_y*w_xy + 2*dsg[5]*u_x*v_y/radius + 2*dsg[5]*u_x/radius + 2*dsg[5]*u_xy*v/radius - 4*dsg[5]*u_y*v_x/radius + 1.5*dsg[5]*v_y/radius + 1.5*dsg[5]/radius + 2*dsg[5]*u_x*w/(radius*radius) + 1.5*dsg[5]*w/(radius*radius) + 2*du_x*sg[5]*v_y/radius + 2*du_x*sg[5]/radius + 2*du_x*sg[5]*w/(radius*radius) - du_xx*sg[3]*w_y + du_xx*sg[3]*v/radius - 2*du_xy*sg[5]*w_y + 2*du_xy*sg[5]*v/radius + du_y*sg[3]*w_xx + du_y*sg[4]*w_yy + 2*du_y*sg[5]*w_xy - 2*du_y*sg[4]*v_y/radius - du_y*sg[4]/radius - 4*du_y*sg[5]*v_x/radius - du_y*sg[4]*w/(radius*radius) - du_yy*sg[4]*w_y + du_yy*sg[4]*v/radius + dv*sg[3]*u_xx/radius + dv*sg[4]*u_yy/radius + 2*dv*sg[5]*u_xy/radius + dv_x*sg[0] - dv_x*sg[4]/radius - 4*dv_x*sg[5]*u_y/radius + dv_y*sg[2] - 2*dv_y*sg[4]*u_y/radius + 2*dv_y*sg[5]*u_x/radius + 1.5*dv_y*sg[5]/radius + dw*sg[2]/radius - dw*sg[4]*u_y/(radius*radius) + 2*dw*sg[5]*u_x/(radius*radius) + 1.5*dw*sg[5]/(radius*radius) + dw_xx*sg[3]*u_y + 2*dw_xy*sg[5]*u_y - dw_y*sg[3]*u_xx - dw_y*sg[4]*u_yy - 2*dw_y*sg[5]*u_xy + dw_yy*sg[4]*u_y
                S[3] = dsg[1]*v_y + dsg[1] + dsg[1]*w/radius + dsg[2]*v_x - dsg[3]*u_x*w_xx + dsg[3]*u_xx*w_x - dsg[3]*w_xx - dsg[4]*u_x*w_yy + dsg[4]*u_yy*w_x - dsg[4]*w_yy + 4*dsg[4]*u_x*v_y/radius + 3*dsg[4]*u_x/radius - 2*dsg[4]*u_y*v_x/radius + 2*dsg[4]*v_y/radius + dsg[4]/radius + 3*dsg[4]*u_x*w/(radius*radius) + dsg[4]*w/(radius*radius) - 2*dsg[5]*u_x*w_xy + 2*dsg[5]*u_xy*w_x - 2*dsg[5]*w_xy + 2*dsg[5]*u_x*v_x/radius + 1.5*dsg[5]*v_x/radius - du_x*sg[3]*w_xx - du_x*sg[4]*w_yy - 2*du_x*sg[5]*w_xy + 4*du_x*sg[4]*v_y/radius + 3*du_x*sg[4]/radius + 2*du_x*sg[5]*v_x/radius + 3*du_x*sg[4]*w/(radius*radius) + du_xx*sg[3]*w_x + 2*du_xy*sg[5]*w_x - 2*du_y*sg[4]*v_x/radius + du_yy*sg[4]*w_x + dv_x*sg[2] - 2*dv_x*sg[4]*u_y/radius + 2*dv_x*sg[5]*u_x/radius + 1.5*dv_x*sg[5]/radius + dv_y*sg[1] + 4*dv_y*sg[4]*u_x/radius + 2*dv_y*sg[4]/radius + dw*sg[1]/radius + 3*dw*sg[4]*u_x/(radius*radius) + dw*sg[4]/(radius*radius) + dw_x*sg[3]*u_xx + dw_x*sg[4]*u_yy + 2*dw_x*sg[5]*u_xy - dw_xx*sg[3]*u_x - dw_xx*sg[3] - 2*dw_xy*sg[5]*u_x - 2*dw_xy*sg[5] - dw_yy*sg[4]*u_x - dw_yy*sg[4]
                S[4] = dsg[0]*w_x + dsg[2]*w_y - dsg[2]*v/radius + dsg[3]*u_xx*v_y + dsg[3]*u_xx - dsg[3]*u_y*v_xx + dsg[3]*u_xx*w/radius - dsg[4]*u_y*v_yy + dsg[4]*u_yy*v_y + dsg[4]*u_yy - 2*dsg[4]*u_y*w_y/radius + dsg[4]*u_yy*w/radius - dsg[4]*w_x/radius + dsg[4]*u_y*v/(radius*radius) + 2*dsg[5]*u_xy*v_y + 2*dsg[5]*u_xy - 2*dsg[5]*u_y*v_xy + 2*dsg[5]*u_x*w_y/radius + 2*dsg[5]*u_xy*w/radius - 4*dsg[5]*u_y*w_x/radius + 1.5*dsg[5]*w_y/radius - 2*dsg[5]*u_x*v/(radius*radius) - 1.5*dsg[5]*v/(radius*radius) + 2*du_x*sg[5]*w_y/radius - 2*du_x*sg[5]*v/(radius*radius) + du_xx*sg[3]*v_y + du_xx*sg[3] + du_xx*sg[3]*w/radius + 2*du_xy*sg[5]*v_y + 2*du_xy*sg[5] + 2*du_xy*sg[5]*w/radius - du_y*sg[3]*v_xx - du_y*sg[4]*v_yy - 2*du_y*sg[5]*v_xy - 2*du_y*sg[4]*w_y/radius - 4*du_y*sg[5]*w_x/radius + du_y*sg[4]*v/(radius*radius) + du_yy*sg[4]*v_y + du_yy*sg[4] + du_yy*sg[4]*w/radius - dv*sg[2]/radius + dv*sg[4]*u_y/(radius*radius) - 2*dv*sg[5]*u_x/(radius*radius) - 1.5*dv*sg[5]/(radius*radius) - dv_xx*sg[3]*u_y - 2*dv_xy*sg[5]*u_y + dv_y*sg[3]*u_xx + dv_y*sg[4]*u_yy + 2*dv_y*sg[5]*u_xy - dv_yy*sg[4]*u_y + dw*sg[3]*u_xx/radius + dw*sg[4]*u_yy/radius + 2*dw*sg[5]*u_xy/radius + dw_x*sg[0] - dw_x*sg[4]/radius - 4*dw_x*sg[5]*u_y/radius + dw_y*sg[2] - 2*dw_y*sg[4]*u_y/radius + 2*dw_y*sg[5]*u_x/radius + 1.5*dw_y*sg[5]/radius
                S[5] = dsg[1]*w_y - dsg[1]*v/radius + dsg[2]*w_x + dsg[3]*u_x*v_xx - dsg[3]*u_xx*v_x + dsg[3]*v_xx + dsg[4]*u_x*v_yy - dsg[4]*u_yy*v_x + dsg[4]*v_yy + 4*dsg[4]*u_x*w_y/radius - 2*dsg[4]*u_y*w_x/radius + 2*dsg[4]*w_y/radius - 3*dsg[4]*u_x*v/(radius*radius) - dsg[4]*v/(radius*radius) + 2*dsg[5]*u_x*v_xy - 2*dsg[5]*u_xy*v_x + 2*dsg[5]*v_xy + 2*dsg[5]*u_x*w_x/radius + 1.5*dsg[5]*w_x/radius + du_x*sg[3]*v_xx + du_x*sg[4]*v_yy + 2*du_x*sg[5]*v_xy + 4*du_x*sg[4]*w_y/radius + 2*du_x*sg[5]*w_x/radius - 3*du_x*sg[4]*v/(radius*radius) - du_xx*sg[3]*v_x - 2*du_xy*sg[5]*v_x - 2*du_y*sg[4]*w_x/radius - du_yy*sg[4]*v_x - dv*sg[1]/radius - 3*dv*sg[4]*u_x/(radius*radius) - dv*sg[4]/(radius*radius) - dv_x*sg[3]*u_xx - dv_x*sg[4]*u_yy - 2*dv_x*sg[5]*u_xy + dv_xx*sg[3]*u_x + dv_xx*sg[3] + 2*dv_xy*sg[5]*u_x + 2*dv_xy*sg[5] + dv_yy*sg[4]*u_x + dv_yy*sg[4] + dw_x*sg[2] - 2*dw_x*sg[4]*u_y/radius + 2*dw_x*sg[5]*u_x/radius + 1.5*dw_x*sg[5]/radius + dw_y*sg[1] + 4*dw_y*sg[4]*u_x/radius + 2*dw_y*sg[4]/radius
                S[6] = -dsg[1]*w_y/radius + dsg[1]*v/(radius*radius) - dsg[2]*w_x/radius - dsg[3]*u_x*v_xx/radius + dsg[3]*u_xx*v_x/radius - dsg[3]*v_xx/radius - dsg[4]*u_x*v_yy/radius + dsg[4]*u_yy*v_x/radius - dsg[4]*v_yy/radius - 3*dsg[4]*u_x*w_y/(radius*radius) + dsg[4]*u_y*w_x/(radius*radius) - dsg[4]*w_y/(radius*radius) + 2*dsg[4]*u_x*v/(radius*radius*radius) - 2*dsg[5]*u_x*v_xy/radius + 2*dsg[5]*u_xy*v_x/radius - 2*dsg[5]*v_xy/radius - 2*dsg[5]*u_x*w_x/(radius*radius) - 1.5*dsg[5]*w_x/(radius*radius) - du_x*sg[3]*v_xx/radius - du_x*sg[4]*v_yy/radius - 2*du_x*sg[5]*v_xy/radius - 3*du_x*sg[4]*w_y/(radius*radius) - 2*du_x*sg[5]*w_x/(radius*radius) + 2*du_x*sg[4]*v/(radius*radius*radius) + du_xx*sg[3]*v_x/radius + 2*du_xy*sg[5]*v_x/radius + du_y*sg[4]*w_x/(radius*radius) + du_yy*sg[4]*v_x/radius + dv*sg[1]/(radius*radius) + 2*dv*sg[4]*u_x/(radius*radius*radius) + dv_x*sg[3]*u_xx/radius + dv_x*sg[4]*u_yy/radius + 2*dv_x*sg[5]*u_xy/radius - dv_xx*sg[3]*u_x/radius - dv_xx*sg[3]/radius - 2*dv_xy*sg[5]*u_x/radius - 2*dv_xy*sg[5]/radius - dv_yy*sg[4]*u_x/radius - dv_yy*sg[4]/radius - dw_x*sg[2]/radius + dw_x*sg[4]*u_y/(radius*radius) - 2*dw_x*sg[5]*u_x/(radius*radius) - 1.5*dw_x*sg[5]/(radius*radius) - dw_y*sg[1]/radius - 3*dw_y*sg[4]*u_x/(radius*radius) - dw_y*sg[4]/(radius*radius)
                S[7] = dsg[1]*v_y/radius + dsg[1]/radius + dsg[1]*w/(radius*radius) + dsg[2]*v_x/radius - dsg[3]*u_x*w_xx/radius + dsg[3]*u_xx*w_x/radius - dsg[3]*w_xx/radius - dsg[4]*u_x*w_yy/radius + dsg[4]*u_yy*w_x/radius - dsg[4]*w_yy/radius + 3*dsg[4]*u_x*v_y/(radius*radius) + 2*dsg[4]*u_x/(radius*radius) - dsg[4]*u_y*v_x/(radius*radius) + dsg[4]*v_y/(radius*radius) + 2*dsg[4]*u_x*w/(radius*radius*radius) - 2*dsg[5]*u_x*w_xy/radius + 2*dsg[5]*u_xy*w_x/radius - 2*dsg[5]*w_xy/radius + 2*dsg[5]*u_x*v_x/(radius*radius) + 1.5*dsg[5]*v_x/(radius*radius) - du_x*sg[3]*w_xx/radius - du_x*sg[4]*w_yy/radius - 2*du_x*sg[5]*w_xy/radius + 3*du_x*sg[4]*v_y/(radius*radius) + 2*du_x*sg[4]/(radius*radius) + 2*du_x*sg[5]*v_x/(radius*radius) + 2*du_x*sg[4]*w/(radius*radius*radius) + du_xx*sg[3]*w_x/radius + 2*du_xy*sg[5]*w_x/radius - du_y*sg[4]*v_x/(radius*radius) + du_yy*sg[4]*w_x/radius + dv_x*sg[2]/radius - dv_x*sg[4]*u_y/(radius*radius) + 2*dv_x*sg[5]*u_x/(radius*radius) + 1.5*dv_x*sg[5]/(radius*radius) + dv_y*sg[1]/radius + 3*dv_y*sg[4]*u_x/(radius*radius) + dv_y*sg[4]/(radius*radius) + dw*sg[1]/(radius*radius) + 2*dw*sg[4]*u_x/(radius*radius*radius) + dw_x*sg[3]*u_xx/radius + dw_x*sg[4]*u_yy/radius + 2*dw_x*sg[5]*u_xy/radius - dw_xx*sg[3]*u_x/radius - dw_xx*sg[3]/radius - 2*dw_xy*sg[5]*u_x/radius - 2*dw_xy*sg[5]/radius - dw_yy*sg[4]*u_x/radius - dw_yy*sg[4]/radius
                S[8] = -dsg[3]*v_x*w_y + dsg[3]*v_y*w_x + dsg[3]*w_x + dsg[3]*v*v_x/radius + dsg[3]*w*w_x/radius + dv*sg[3]*v_x/radius - dv_x*sg[3]*w_y + dv_x*sg[3]*v/radius + dv_y*sg[3]*w_x + dw*sg[3]*w_x/radius + dw_x*sg[3]*v_y + dw_x*sg[3] + dw_x*sg[3]*w/radius - dw_y*sg[3]*v_x
                S[9] = -dsg[4]*v_x*w_y + dsg[4]*v_y*w_x + dsg[4]*w_x + dsg[4]*v*v_x/radius + dsg[4]*w*w_x/radius + dv*sg[4]*v_x/radius - dv_x*sg[4]*w_y + dv_x*sg[4]*v/radius + dv_y*sg[4]*w_x + dw*sg[4]*w_x/radius + dw_x*sg[4]*v_y + dw_x*sg[4] + dw_x*sg[4]*w/radius - dw_y*sg[4]*v_x
                S[10] = -2*dsg[5]*v_x*w_y + 2*dsg[5]*v_y*w_x + 2*dsg[5]*w_x + 2*dsg[5]*v*v_x/radius + 2*dsg[5]*w*w_x/radius + 2*dv*sg[5]*v_x/radius - 2*dv_x*sg[5]*w_y + 2*dv_x*sg[5]*v/radius + 2*dv_y*sg[5]*w_x + 2*dw*sg[5]*w_x/radius + 2*dw_x*sg[5]*v_y + 2*dw_x*sg[5] + 2*dw_x*sg[5]*w/radius - 2*dw_y*sg[5]*v_x
                S[11] = dsg[3]*u_x*w_y - dsg[3]*u_y*w_x + dsg[3]*w_y - dsg[3]*u_x*v/radius - dsg[3]*v/radius + du_x*sg[3]*w_y - du_x*sg[3]*v/radius - du_y*sg[3]*w_x - dv*sg[3]*u_x/radius - dv*sg[3]/radius - dw_x*sg[3]*u_y + dw_y*sg[3]*u_x + dw_y*sg[3]
                S[12] = dsg[4]*u_x*w_y - dsg[4]*u_y*w_x + dsg[4]*w_y - dsg[4]*u_x*v/radius - dsg[4]*v/radius + du_x*sg[4]*w_y - du_x*sg[4]*v/radius - du_y*sg[4]*w_x - dv*sg[4]*u_x/radius - dv*sg[4]/radius - dw_x*sg[4]*u_y + dw_y*sg[4]*u_x + dw_y*sg[4]
                S[13] = 2*dsg[5]*u_x*w_y - 2*dsg[5]*u_y*w_x + 2*dsg[5]*w_y - 2*dsg[5]*u_x*v/radius - 2*dsg[5]*v/radius + 2*du_x*sg[5]*w_y - 2*du_x*sg[5]*v/radius - 2*du_y*sg[5]*w_x - 2*dv*sg[5]*u_x/radius - 2*dv*sg[5]/radius - 2*dw_x*sg[5]*u_y + 2*dw_y*sg[5]*u_x + 2*dw_y*sg[5]
                S[14] = -dsg[3]*u_x*v_y - dsg[3]*u_x + dsg[3]*u_y*v_x - dsg[3]*v_y - dsg[3] - dsg[3]*u_x*w/radius - dsg[3]*w/radius - du_x*sg[3]*v_y - du_x*sg[3] - du_x*sg[3]*w/radius + du_y*sg[3]*v_x + dv_x*sg[3]*u_y - dv_y*sg[3]*u_x - dv_y*sg[3] - dw*sg[3]*u_x/radius - dw*sg[3]/radius
                S[15] = -dsg[4]*u_x*v_y - dsg[4]*u_x + dsg[4]*u_y*v_x - dsg[4]*v_y - dsg[4] - dsg[4]*u_x*w/radius - dsg[4]*w/radius - du_x*sg[4]*v_y - du_x*sg[4] - du_x*sg[4]*w/radius + du_y*sg[4]*v_x + dv_x*sg[4]*u_y - dv_y*sg[4]*u_x - dv_y*sg[4] - dw*sg[4]*u_x/radius - dw*sg[4]/radius
                S[16] = -2*dsg[5]*u_x*v_y - 2*dsg[5]*u_x + 2*dsg[5]*u_y*v_x - 2*dsg[5]*v_y - 2*dsg[5] - 2*dsg[5]*u_x*w/radius - 2*dsg[5]*w/radius - 2*du_x*sg[5]*v_y - 2*du_x*sg[5] - 2*du_x*sg[5]*w/radius + 2*du_y*sg[5]*v_x + 2*dv_x*sg[5]*u_y - 2*dv_y*sg[5]*u_x - 2*dv_y*sg[5] - 2*dw*sg[5]*u_x/radius - 2*dw*sg[5]/radius

                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAu_yy = gtab[2, j]
                    gAv = gtab[3, j]
                    gAv_y = gtab[4, j]
                    gAv_yy = gtab[5, j]
                    gAw = gtab[6, j]
                    gAw_y = gtab[7, j]
                    gAw_yy = gtab[8, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAu_xx = ftab[2, i]
                        fAv = ftab[3, i]
                        fAv_x = ftab[4, i]
                        fAv_xx = ftab[5, i]
                        fAw = ftab[6, i]
                        fAw_x = ftab[7, i]
                        fAw_xx = ftab[8, i]
                        col = col0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAu_xx*gAu
                        pA[9] = fAu*gAu_yy
                        pA[10] = fAu_x*gAu_y
                        pA[11] = fAv_xx*gAv
                        pA[12] = fAv*gAv_yy
                        pA[13] = fAv_x*gAv_y
                        pA[14] = fAw_xx*gAw
                        pA[15] = fAw*gAw_yy
                        pA[16] = fAw_x*gAw_y
                        kTv[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1] + pA[8]*S[8] + pA[9]*S[9] + pA[10]*S[10])
                        kTv[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[6]*S[6] + pA[11]*S[11] + pA[12]*S[12] + pA[13]*S[13])
                        kTv[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[7]*S[7] + pA[14]*S[14] + pA[15]*S[15] + pA[16]*S[16])

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
