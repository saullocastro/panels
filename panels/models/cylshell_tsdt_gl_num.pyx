#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Cylindrical shells using the third-order shear deformation theory (TSDT)
with Green-Lagrange membrane strains

Numerically integrated kernels of the model ``'cylshell_tsdt_gl'``: the
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
The membrane strains are the Green-Lagrange strains of the mid-surface,
those of :mod:`panels.models.cylshell_clpt_gl`, which vanish for any rigid
motion, and the bending and transverse shear strains are the linear ones
of :mod:`panels.models.cylshell_tsdt_sanders`:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} (u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2)

    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \left(u_{,y}^2 +
    \left(v_{,y} + \frac{w}{r}\right)^2 + \left(w_{,y} -
    \frac{v}{r}\right)^2\right)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} \left(v_{,y} +
    \frac{w}{r}\right) + w_{,x} \left(w_{,y} - \frac{v}{r}\right)

    \kappa_{xx} = \phi_{x,x} \qquad \kappa_{yy} = \phi_{y,y} +
    \frac{v_{,y}}{r} \qquad \kappa_{xy} = \phi_{x,y} + \phi_{y,x} +
    \frac{3}{2} \frac{v_{,x}}{r} - \frac{1}{2} \frac{u_{,y}}{r}

    \gamma_{yz} = \phi_y + w_{,y} \qquad \gamma_{xz} = \phi_x + w_{,x}

    \kappa^{(3)}_{xx} = -c_1 (\phi_{x,x} + w_{,xx}) \qquad
    \kappa^{(3)}_{yy} = -c_1 (\phi_{y,y} + w_{,yy}) \qquad
    \kappa^{(3)}_{xy} = -c_1 (\phi_{x,y} + \phi_{y,x} + 2 w_{,xy})

    \gamma^{(2)}_{yz} = -3 c_1 \gamma_{yz} \qquad
    \gamma^{(2)}_{xz} = -3 c_1 \gamma_{xz}

with `y` the arc length along the circumference and `w` positive outwards.
The membrane strains are those of the Flügge-Lur'e-Byrne non-linear theory
as given by Amabili [1]_, Sec. 1.3.6, Eqs. (1.118)-(1.120); besides the
non-linear terms of :mod:`panels.models.cylshell_tsdt_sanders`, the squares
of the rotations `-w_{,x}` and `-w_{,y} + v/r`, they contain the quadratic
terms of the in-plane gradients, among which those of the rotation about
the normal, `(v_{,x} - u_{,y})/2`, kept by Sanders [2]_. The bending,
third-order and transverse shear strains are those of the third-order shear
deformation theory of Reddy and Liu [3]_ with the Sanders-Koiter terms,
`c_1 = 4/(3 h^2)` [4]_, see :mod:`panels.models`.

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
.. [3] Reddy, J. N., and Liu, C. F., "A higher-order shear deformation
   theory of laminated elastic shells", International Journal of
   Engineering Science, 23(3), 319-330, 1985.
   https://doi.org/10.1016/0020-7225(85)90051-5
.. [4] Reddy, J. N., "A simple higher-order theory for laminated
   composite plates", Journal of Applied Mechanics, 51(4), 745-752, 1984.
   https://doi.org/10.1115/1.3167719

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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the values of `v` and `w`, the rotations and their gradients and the second derivatives of `w`), the strains
    `\boldsymbol{\varepsilon}(\boldsymbol{d})`, `\boldsymbol{B} =
    \partial \boldsymbol{\varepsilon} / \partial \boldsymbol{d}`, the
    stresses `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{H}(\boldsymbol{\sigma}) =
    \sum_r \sigma_r \, \partial^2 \varepsilon_r / \partial
    \boldsymbol{d}^2`, which involves only the membrane stresses and is
    independent of `\boldsymbol{d}`, the membrane strains being quadratic:

    .. math::

        \boldsymbol{K}_C = \int \boldsymbol{\Phi}^\top \left[\boldsymbol{B}^\top
        \boldsymbol{F} \boldsymbol{B} + \boldsymbol{H}(\boldsymbol{\sigma} -
        \boldsymbol{\sigma}_L)\right] \boldsymbol{\Phi} \, dA

    with `\boldsymbol{\sigma}_L` the stresses of the linear strains, whose
    term `\boldsymbol{H}(\boldsymbol{\sigma}_L)` is assembled by
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
    cdef double w_xy
    cdef double e[13]
    cdef double eL[13]
    cdef double sg[13]
    cdef double sgL[13]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAphix
    cdef double fAphix_x, fAphiy, fAphiy_x, fAw_xx, gAu, gAu_y, gAv
    cdef double gAv_y, gAw, gAw_y, gAphix, gAphix_y, gAphiy, gAphiy_y
    cdef double gAw_yy, fBu, fBu_x, fBv, fBv_x, fBw, fBw_x
    cdef double fBphix, fBphix_x, fBphiy, fBphiy_x, fBw_xx, gBu, gBu_y
    cdef double gBv, gBv_y, gBw, gBw_y, gBphix, gBphix_y, gBphiy
    cdef double gBphiy_y, gBw_yy
    cdef double Bm[221]
    cdef double FB[221]
    cdef double K[289]
    cdef double TA[85]
    cdef double pA[17]
    cdef double pB[17]

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
                    ftab[6, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[9, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
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
                    gtab[10, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

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
                        gAw_yy = gtab[10, j]
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
                            fAw_xx = ftab[10, i]
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

                # [B] = d eps/d d at the point
                for p in range(221):
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
                Bm[61] = 1
                Bm[71] = 1/radius
                Bm[81] = 1
                Bm[86] = -0.5/radius
                Bm[87] = 1.5/radius
                Bm[96] = 1
                Bm[97] = 1
                Bm[112] = -c1
                Bm[116] = -c1
                Bm[132] = -c1
                Bm[134] = -c1
                Bm[147] = -c1
                Bm[148] = -c1
                Bm[152] = -2*c1
                Bm[158] = 1
                Bm[162] = 1
                Bm[174] = 1
                Bm[178] = 1
                Bm[192] = -3*c1
                Bm[196] = -3*c1
                Bm[208] = -3*c1
                Bm[212] = -3*c1

                # K = [B]^T [F] [B]
                for r in range(NE):
                    for q in range(17):
                        FB[r*17 + q] = 0.
                        for s in range(NE):
                            FB[r*17 + q] += F[r*NE + s]*Bm[s*17 + q]
                for p in range(17):
                    for q in range(17):
                        K[p*17 + q] = 0.
                        for r in range(NE):
                            K[p*17 + q] += Bm[r*17 + p]*FB[r*17 + q]

                # stresses of the strains and of the linear strains
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
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
                # K += [H(sigma - sigma_L)], [H(sigma_L)] being assembled by
                # fkG_num, which is homogeneous of degree one in cs as linear
                # buckling requires
                K[0] += -sgL[0] + sg[0]
                K[1] += -sgL[2] + sg[2]
                K[17] += -sgL[2] + sg[2]
                K[18] += -sgL[1] + sg[1]
                K[36] += -sgL[0] + sg[0]
                K[37] += -sgL[2] + sg[2]
                K[41] += -sgL[2]/radius + sg[2]/radius
                K[53] += -sgL[2] + sg[2]
                K[54] += -sgL[1] + sg[1]
                K[58] += -sgL[1]/radius + sg[1]/radius
                K[72] += -sgL[0] + sg[0]
                K[73] += -sgL[2] + sg[2]
                K[74] += sgL[2]/radius - sg[2]/radius
                K[89] += -sgL[2] + sg[2]
                K[90] += -sgL[1] + sg[1]
                K[91] += sgL[1]/radius - sg[1]/radius
                K[106] += sgL[2]/radius - sg[2]/radius
                K[107] += sgL[1]/radius - sg[1]/radius
                K[108] += -sgL[1]/(radius*radius) + sg[1]/(radius*radius)
                K[121] += -sgL[2]/radius + sg[2]/radius
                K[122] += -sgL[1]/radius + sg[1]/radius
                K[126] += -sgL[1]/(radius*radius) + sg[1]/(radius*radius)

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
                    gAw_yy = gtab[10, j]
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
                        fAw_xx = ftab[10, i]
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
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[17]
                        TA[1] = pA[0]*K[1] + pA[1]*K[18]
                        TA[2] = pA[0]*K[2] + pA[1]*K[19]
                        TA[3] = pA[0]*K[3] + pA[1]*K[20]
                        TA[4] = pA[0]*K[4] + pA[1]*K[21]
                        TA[5] = pA[0]*K[5] + pA[1]*K[22]
                        TA[6] = pA[0]*K[6] + pA[1]*K[23]
                        TA[7] = pA[0]*K[7] + pA[1]*K[24]
                        TA[8] = pA[0]*K[8] + pA[1]*K[25]
                        TA[9] = pA[0]*K[9] + pA[1]*K[26]
                        TA[10] = pA[0]*K[10] + pA[1]*K[27]
                        TA[11] = pA[0]*K[11] + pA[1]*K[28]
                        TA[12] = pA[0]*K[12] + pA[1]*K[29]
                        TA[13] = pA[0]*K[13] + pA[1]*K[30]
                        TA[14] = pA[0]*K[14] + pA[1]*K[31]
                        TA[15] = pA[0]*K[15] + pA[1]*K[32]
                        TA[16] = pA[0]*K[16] + pA[1]*K[33]
                        TA[17] = pA[2]*K[34] + pA[3]*K[51] + pA[6]*K[102]
                        TA[18] = pA[2]*K[35] + pA[3]*K[52] + pA[6]*K[103]
                        TA[19] = pA[2]*K[36] + pA[3]*K[53] + pA[6]*K[104]
                        TA[20] = pA[2]*K[37] + pA[3]*K[54] + pA[6]*K[105]
                        TA[21] = pA[2]*K[38] + pA[3]*K[55] + pA[6]*K[106]
                        TA[22] = pA[2]*K[39] + pA[3]*K[56] + pA[6]*K[107]
                        TA[23] = pA[2]*K[40] + pA[3]*K[57] + pA[6]*K[108]
                        TA[24] = pA[2]*K[41] + pA[3]*K[58] + pA[6]*K[109]
                        TA[25] = pA[2]*K[42] + pA[3]*K[59] + pA[6]*K[110]
                        TA[26] = pA[2]*K[43] + pA[3]*K[60] + pA[6]*K[111]
                        TA[27] = pA[2]*K[44] + pA[3]*K[61] + pA[6]*K[112]
                        TA[28] = pA[2]*K[45] + pA[3]*K[62] + pA[6]*K[113]
                        TA[29] = pA[2]*K[46] + pA[3]*K[63] + pA[6]*K[114]
                        TA[30] = pA[2]*K[47] + pA[3]*K[64] + pA[6]*K[115]
                        TA[31] = pA[2]*K[48] + pA[3]*K[65] + pA[6]*K[116]
                        TA[32] = pA[2]*K[49] + pA[3]*K[66] + pA[6]*K[117]
                        TA[33] = pA[2]*K[50] + pA[3]*K[67] + pA[6]*K[118]
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
                        TA[51] = pA[8]*K[136] + pA[10]*K[170] + pA[11]*K[187]
                        TA[52] = pA[8]*K[137] + pA[10]*K[171] + pA[11]*K[188]
                        TA[53] = pA[8]*K[138] + pA[10]*K[172] + pA[11]*K[189]
                        TA[54] = pA[8]*K[139] + pA[10]*K[173] + pA[11]*K[190]
                        TA[55] = pA[8]*K[140] + pA[10]*K[174] + pA[11]*K[191]
                        TA[56] = pA[8]*K[141] + pA[10]*K[175] + pA[11]*K[192]
                        TA[57] = pA[8]*K[142] + pA[10]*K[176] + pA[11]*K[193]
                        TA[58] = pA[8]*K[143] + pA[10]*K[177] + pA[11]*K[194]
                        TA[59] = pA[8]*K[144] + pA[10]*K[178] + pA[11]*K[195]
                        TA[60] = pA[8]*K[145] + pA[10]*K[179] + pA[11]*K[196]
                        TA[61] = pA[8]*K[146] + pA[10]*K[180] + pA[11]*K[197]
                        TA[62] = pA[8]*K[147] + pA[10]*K[181] + pA[11]*K[198]
                        TA[63] = pA[8]*K[148] + pA[10]*K[182] + pA[11]*K[199]
                        TA[64] = pA[8]*K[149] + pA[10]*K[183] + pA[11]*K[200]
                        TA[65] = pA[8]*K[150] + pA[10]*K[184] + pA[11]*K[201]
                        TA[66] = pA[8]*K[151] + pA[10]*K[185] + pA[11]*K[202]
                        TA[67] = pA[8]*K[152] + pA[10]*K[186] + pA[11]*K[203]
                        TA[68] = pA[9]*K[153] + pA[12]*K[204] + pA[13]*K[221]
                        TA[69] = pA[9]*K[154] + pA[12]*K[205] + pA[13]*K[222]
                        TA[70] = pA[9]*K[155] + pA[12]*K[206] + pA[13]*K[223]
                        TA[71] = pA[9]*K[156] + pA[12]*K[207] + pA[13]*K[224]
                        TA[72] = pA[9]*K[157] + pA[12]*K[208] + pA[13]*K[225]
                        TA[73] = pA[9]*K[158] + pA[12]*K[209] + pA[13]*K[226]
                        TA[74] = pA[9]*K[159] + pA[12]*K[210] + pA[13]*K[227]
                        TA[75] = pA[9]*K[160] + pA[12]*K[211] + pA[13]*K[228]
                        TA[76] = pA[9]*K[161] + pA[12]*K[212] + pA[13]*K[229]
                        TA[77] = pA[9]*K[162] + pA[12]*K[213] + pA[13]*K[230]
                        TA[78] = pA[9]*K[163] + pA[12]*K[214] + pA[13]*K[231]
                        TA[79] = pA[9]*K[164] + pA[12]*K[215] + pA[13]*K[232]
                        TA[80] = pA[9]*K[165] + pA[12]*K[216] + pA[13]*K[233]
                        TA[81] = pA[9]*K[166] + pA[12]*K[217] + pA[13]*K[234]
                        TA[82] = pA[9]*K[167] + pA[12]*K[218] + pA[13]*K[235]
                        TA[83] = pA[9]*K[168] + pA[12]*K[219] + pA[13]*K[236]
                        TA[84] = pA[9]*K[169] + pA[12]*K[220] + pA[13]*K[237]
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
                            gBw_yy = gtab[10, l]
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
                                fBw_xx = ftab[10, k]
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
                                kCv[c] += weight*(TA[17]*pB[0] + TA[18]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[19]*pB[2] + TA[20]*pB[3] + TA[23]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[21]*pB[4] + TA[22]*pB[5] + TA[24]*pB[7] + TA[31]*pB[14] + TA[32]*pB[15] + TA[33]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[25]*pB[8] + TA[27]*pB[10] + TA[28]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[26]*pB[9] + TA[29]*pB[12] + TA[30]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[34]*pB[0] + TA[35]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[36]*pB[2] + TA[37]*pB[3] + TA[40]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[38]*pB[4] + TA[39]*pB[5] + TA[41]*pB[7] + TA[48]*pB[14] + TA[49]*pB[15] + TA[50]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[42]*pB[8] + TA[44]*pB[10] + TA[45]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[43]*pB[9] + TA[46]*pB[12] + TA[47]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[51]*pB[0] + TA[52]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[53]*pB[2] + TA[54]*pB[3] + TA[57]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[55]*pB[4] + TA[56]*pB[5] + TA[58]*pB[7] + TA[65]*pB[14] + TA[66]*pB[15] + TA[67]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[59]*pB[8] + TA[61]*pB[10] + TA[62]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[60]*pB[9] + TA[63]*pB[12] + TA[64]*pB[13])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[68]*pB[0] + TA[69]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[70]*pB[2] + TA[71]*pB[3] + TA[74]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[72]*pB[4] + TA[73]*pB[5] + TA[75]*pB[7] + TA[82]*pB[14] + TA[83]*pB[15] + TA[84]*pB[16])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[76]*pB[8] + TA[78]*pB[10] + TA[79]*pB[11])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[77]*pB[9] + TA[80]*pB[12] + TA[81]*pB[13])

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC


def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    r"""Geometric stiffness matrix of the linear stress state

    Integrates numerically the geometric stiffness matrix of the membrane
    stresses of the linear strains at ``cs``, superposed to the constant
    membrane state ``Nxx0``, ``Nyy0``, ``Nxy0``, with one block per local
    gradient of ``u``, ``v`` and ``w``. The matrix is homogeneous of degree
    one in ``cs``, as the linear buckling eigenvalue problem requires.

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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the values of `v` and `w`, the rotations and their gradients and the second derivatives of `w`), the strains
    `\boldsymbol{\varepsilon}(\boldsymbol{d})`, `\boldsymbol{B} =
    \partial \boldsymbol{\varepsilon} / \partial \boldsymbol{d}`, the
    stresses `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{H}(\boldsymbol{\sigma}) =
    \sum_r \sigma_r \, \partial^2 \varepsilon_r / \partial
    \boldsymbol{d}^2`, which involves only the membrane stresses and is
    independent of `\boldsymbol{d}`, the membrane strains being quadratic:

    .. math::

        \boldsymbol{K}_G = \int \boldsymbol{\Phi}^\top
        \boldsymbol{H}(\boldsymbol{\sigma}) \boldsymbol{\Phi} \, dA

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
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAphix
    cdef double fAphix_x, fAphiy, fAphiy_x, fAw_xx, gAu, gAu_y, gAv
    cdef double gAv_y, gAw, gAw_y, gAphix, gAphix_y, gAphiy, gAphiy_y
    cdef double gAw_yy, fBu, fBu_x, fBv, fBv_x, fBw, fBw_x
    cdef double fBphix, fBphix_x, fBphiy, fBphiy_x, fBw_xx, gBu, gBu_y
    cdef double gBv, gBv_y, gBw, gBw_y, gBphix, gBphix_y, gBphiy
    cdef double gBphiy_y, gBw_yy
    cdef double K[289]
    cdef double TA[85]
    cdef double pA[17]
    cdef double pB[17]

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

    ftab = np.zeros((11, m), dtype=DOUBLE)
    gtab = np.zeros((11, n), dtype=DOUBLE)

    fdim = 5*m*m*n*n
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
                    ftab[10, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
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
                    gtab[10, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

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
                    gAphix = gtab[6, j]
                    gAphix_y = gtab[7, j]
                    gAphiy = gtab[8, j]
                    gAphiy_y = gtab[9, j]
                    gAw_yy = gtab[10, j]
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
                        fAw_xx = ftab[10, i]
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

                # K = [H(sigma)]
                for p in range(289):
                    K[p] = 0.
                K[0] = sg[0]
                K[1] = sg[2]
                K[17] = sg[2]
                K[18] = sg[1]
                K[36] = sg[0]
                K[37] = sg[2]
                K[41] = sg[2]/radius
                K[53] = sg[2]
                K[54] = sg[1]
                K[58] = sg[1]/radius
                K[72] = sg[0]
                K[73] = sg[2]
                K[74] = -sg[2]/radius
                K[89] = sg[2]
                K[90] = sg[1]
                K[91] = -sg[1]/radius
                K[106] = -sg[2]/radius
                K[107] = -sg[1]/radius
                K[108] = sg[1]/(radius*radius)
                K[121] = sg[2]/radius
                K[122] = sg[1]/radius
                K[126] = sg[1]/(radius*radius)

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
                    gAw_yy = gtab[10, j]
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
                        fAw_xx = ftab[10, i]
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
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[17]
                        TA[1] = pA[0]*K[1] + pA[1]*K[18]
                        TA[19] = pA[2]*K[36] + pA[3]*K[53]
                        TA[20] = pA[2]*K[37] + pA[3]*K[54]
                        TA[21] = pA[6]*K[106]
                        TA[22] = pA[6]*K[107]
                        TA[23] = pA[6]*K[108]
                        TA[24] = pA[2]*K[41] + pA[3]*K[58]
                        TA[36] = pA[7]*K[121]
                        TA[37] = pA[7]*K[122]
                        TA[38] = pA[4]*K[72] + pA[5]*K[89]
                        TA[39] = pA[4]*K[73] + pA[5]*K[90]
                        TA[40] = pA[4]*K[74] + pA[5]*K[91]
                        TA[41] = pA[7]*K[126]
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
                            gBw_yy = gtab[10, l]
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
                                fBw_xx = ftab[10, k]
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
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[19]*pB[2] + TA[20]*pB[3] + TA[23]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[21]*pB[4] + TA[22]*pB[5] + TA[24]*pB[7])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[36]*pB[2] + TA[37]*pB[3] + TA[40]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[38]*pB[4] + TA[39]*pB[5] + TA[41]*pB[7])

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
    cdef double w_xy
    cdef double e[13]
    cdef double sg[13]
    cdef double S[17]
    cdef double pA[17]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAphix
    cdef double fAphix_x, fAphiy, fAphiy_x, fAw_xx, gAu, gAu_y, gAv
    cdef double gAv_y, gAw, gAw_y, gAphix, gAphix_y, gAphiy, gAphiy_y
    cdef double gAw_yy

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
                    ftab[6, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[9, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
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
                    gtab[10, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

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
                    gAw_yy = gtab[10, j]
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
                        fAw_xx = ftab[10, i]
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

                # strains, stresses and S = [B]^T sigma
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
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
                S[0] = sg[0]*u_x + sg[0] + sg[2]*u_y
                S[1] = sg[1]*u_y + sg[2]*u_x + sg[2] - 0.5*sg[5]/radius
                S[2] = sg[0]*v_x + sg[2]*v_y + sg[2] + sg[2]*w/radius + 1.5*sg[5]/radius
                S[3] = sg[1]*v_y + sg[1] + sg[2]*v_x + sg[1]*w/radius + sg[4]/radius
                S[4] = -3*c1*sg[12] + sg[0]*w_x + sg[10] + sg[2]*w_y - sg[2]*v/radius
                S[5] = -3*c1*sg[11] + sg[1]*w_y + sg[2]*w_x + sg[9] - sg[1]*v/radius
                S[6] = -sg[1]*w_y/radius - sg[2]*w_x/radius + sg[1]*v/(radius*radius)
                S[7] = sg[1]*v_y/radius + sg[1]/radius + sg[2]*v_x/radius + sg[1]*w/(radius*radius)
                S[8] = -3*c1*sg[12] + sg[10]
                S[9] = -3*c1*sg[11] + sg[9]
                S[10] = -c1*sg[6] + sg[3]
                S[11] = -c1*sg[8] + sg[5]
                S[12] = -c1*sg[8] + sg[5]
                S[13] = -c1*sg[7] + sg[4]
                S[14] = -c1*sg[6]
                S[15] = -c1*sg[7]
                S[16] = -2*c1*sg[8]

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
                    gAw_yy = gtab[10, j]
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
                        fAw_xx = ftab[10, i]
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
                        fint[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        fint[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[6]*S[6])
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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the values of `v` and `w`, the rotations and their gradients and the second derivatives of `w`), the strains
    `\boldsymbol{\varepsilon}(\boldsymbol{d})`, `\boldsymbol{B} =
    \partial \boldsymbol{\varepsilon} / \partial \boldsymbol{d}`, the
    stresses `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{H}(\boldsymbol{\sigma}) =
    \sum_r \sigma_r \, \partial^2 \varepsilon_r / \partial
    \boldsymbol{d}^2`, which involves only the membrane stresses and is
    independent of `\boldsymbol{d}`, the membrane strains being quadratic:

    .. math::

        \boldsymbol{K}_T \boldsymbol{v} = \int \boldsymbol{\Phi}^\top
        \left[\boldsymbol{B}^\top \boldsymbol{F} \boldsymbol{B} +
        \boldsymbol{H}(\boldsymbol{\sigma} + \bar{\boldsymbol{\sigma}})
        \right] \boldsymbol{\Phi} \boldsymbol{v} \, dA

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
    cdef double w_xy
    cdef double du_x, du_y, dv_x, dv_y, dw_x, dw_y, dv, dw
    cdef double dphix, dphiy, dphix_x, dphix_y, dphiy_x, dphiy_y, dw_xx, dw_yy
    cdef double dw_xy
    cdef double e[13]
    cdef double de[13]
    cdef double sg[13]
    cdef double dsg[13]
    cdef double S[17]
    cdef double pA[17]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAphix
    cdef double fAphix_x, fAphiy, fAphiy_x, fAw_xx, gAu, gAu_y, gAv
    cdef double gAv_y, gAw, gAw_y, gAphix, gAphix_y, gAphiy, gAphiy_y
    cdef double gAw_yy

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
                    ftab[6, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[7, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[8, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[9, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[10, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
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
                    gtab[10, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

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
                    gAw_yy = gtab[10, j]
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
                        fAw_xx = ftab[10, i]
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

                # strains, strain increments, stresses
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
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
                de[0] = du_x*u_x + du_x + dv_x*v_x + dw_x*w_x
                de[1] = du_y*u_y - dv*w_y/radius + dv*v/(radius*radius) + dv_y*v_y + dv_y + dv_y*w/radius + dw*v_y/radius + dw/radius + dw*w/(radius*radius) + dw_y*w_y - dw_y*v/radius
                de[2] = du_x*u_y + du_y*u_x + du_y - dv*w_x/radius + dv_x*v_y + dv_x + dv_x*w/radius + dv_y*v_x + dw*v_x/radius + dw_x*w_y - dw_x*v/radius + dw_y*w_x
                de[3] = dphix_x
                de[4] = dphiy_y + dv_y/radius
                de[5] = dphix_y + dphiy_x - 0.5*du_y/radius + 1.5*dv_x/radius
                de[6] = -c1*dphix_x - c1*dw_xx
                de[7] = -c1*dphiy_y - c1*dw_yy
                de[8] = -c1*dphix_y - c1*dphiy_x - 2*c1*dw_xy
                de[9] = dphiy + dw_y
                de[10] = dphix + dw_x
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

                # S = [B]^T dsigma + [H(sigma)] dv
                S[0] = dsg[0]*u_x + dsg[0] + dsg[2]*u_y + du_x*sg[0] + du_y*sg[2]
                S[1] = dsg[1]*u_y + dsg[2]*u_x + dsg[2] - 0.5*dsg[5]/radius + du_x*sg[2] + du_y*sg[1]
                S[2] = dsg[0]*v_x + dsg[2]*v_y + dsg[2] + dsg[2]*w/radius + 1.5*dsg[5]/radius + dv_x*sg[0] + dv_y*sg[2] + dw*sg[2]/radius
                S[3] = dsg[1]*v_y + dsg[1] + dsg[1]*w/radius + dsg[2]*v_x + dsg[4]/radius + dv_x*sg[2] + dv_y*sg[1] + dw*sg[1]/radius
                S[4] = -3*c1*dsg[12] + dsg[0]*w_x + dsg[10] + dsg[2]*w_y - dsg[2]*v/radius - dv*sg[2]/radius + dw_x*sg[0] + dw_y*sg[2]
                S[5] = -3*c1*dsg[11] + dsg[1]*w_y - dsg[1]*v/radius + dsg[2]*w_x + dsg[9] - dv*sg[1]/radius + dw_x*sg[2] + dw_y*sg[1]
                S[6] = -dsg[1]*w_y/radius + dsg[1]*v/(radius*radius) - dsg[2]*w_x/radius + dv*sg[1]/(radius*radius) - dw_x*sg[2]/radius - dw_y*sg[1]/radius
                S[7] = dsg[1]*v_y/radius + dsg[1]/radius + dsg[1]*w/(radius*radius) + dsg[2]*v_x/radius + dv_x*sg[2]/radius + dv_y*sg[1]/radius + dw*sg[1]/(radius*radius)
                S[8] = -3*c1*dsg[12] + dsg[10]
                S[9] = -3*c1*dsg[11] + dsg[9]
                S[10] = -c1*dsg[6] + dsg[3]
                S[11] = -c1*dsg[8] + dsg[5]
                S[12] = -c1*dsg[8] + dsg[5]
                S[13] = -c1*dsg[7] + dsg[4]
                S[14] = -c1*dsg[6]
                S[15] = -c1*dsg[7]
                S[16] = -2*c1*dsg[8]

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
                    gAw_yy = gtab[10, j]
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
                        fAw_xx = ftab[10, i]
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
                        kTv[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        kTv[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[6]*S[6])
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
