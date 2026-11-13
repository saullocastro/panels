#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Cylindrical shells using the first-order shear deformation theory (FSDT)
with the Donnell kinematics

Numerically integrated kernels of the model ``'cylshell_fsdt_donnell'``: the
constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the internal force vector :func:`calc_fint`, the product
of the tangent stiffness matrix by a vector :func:`calc_kTv`, the mass
:func:`fkM_num`, the aerodynamic stiffness :func:`fkAx_num` and
:func:`fkAy_num`, and the force vector and load stiffness matrix of a
follower pressure, :func:`calc_fext_follower` and
:func:`fkCfollower_num`. The constitutive matrix ``Finput`` is the
``8 x 8`` matrix ``ABD`` of :class:`.Shell` for this model, with the
rows and columns in the order of the generalized strains::

    exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz

which is the order of the generalized stresses::

    Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qy, Qx

The degrees of freedom of each term of the approximation are
``u, v, w, phix, phiy``; the degree of freedom ``k`` of the term ``(i, j)``,
``0 <= i < m``, ``0 <= j < n``, is at the row ``row0 + 5*(j*m + i) + k``
of the matrices.

Notes
-----
The generalized strains are, with the rotations `\beta_x = -w_{,x}`
and `\beta_y = -w_{,y}` of the non-linear terms of Donnell:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} \beta_x^2 \qquad
    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \beta_y^2 \qquad
    \gamma_{xy} = u_{,y} + v_{,x} + \beta_x \beta_y

    \kappa_{xx} = \phi_{x,x} \qquad \kappa_{yy} = \phi_{y,y} \qquad
    \kappa_{xy} = \phi_{x,y} + \phi_{y,x}

    \gamma_{yz} = \phi_y + w_{,y} \qquad \gamma_{xz} = \phi_x + w_{,x}

With `y` the arc length along the circumference and `w` positive outwards.
The bending and transverse shear strains are those of the first-order shear
deformation theory (FSDT) [1]_, the in-plane strains being
`\varepsilon^{(0)} + z \kappa`.

The numerical integration over the domain of
:meth:`.Shell.integration_limits` admits a constitutive matrix and moments
of the density varying over the domain, a partial domain and the
non-linear terms of the Ritz constants.

References
----------
.. [1] Reddy, J. N., and Liu, C. F., "A higher-order shear deformation
   theory of laminated elastic shells", International Journal of
   Engineering Science, 23(3), 319-330, 1985.
   https://doi.org/10.1016/0020-7225(85)90051-5

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE


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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the value of `w`, the rotations and their gradients), the strains
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
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, w, phix
    cdef double phiy, phix_x, phix_y, phiy_x, phiy_y
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
    cdef double Bm[104]
    cdef double FB[104]
    cdef double K[169]
    cdef double TA[65]
    cdef double pA[13]
    cdef double pB[13]

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
                            w += fAw*gAw*cs[col+2]
                            phix += fAphix*gAphix*cs[col+3]
                            phiy += fAphiy*gAphiy*cs[col+4]
                            phix_x += fAphix_x*gAphix*cs[col+3]
                            phix_y += fAphix*gAphix_y*cs[col+3]
                            phiy_x += fAphiy_x*gAphiy*cs[col+4]
                            phiy_y += fAphiy*gAphiy_y*cs[col+4]

                # [B] = d eps/d d at the point
                for p in range(104):
                    Bm[p] = 0.
                Bm[0] = 1
                Bm[4] = w_x
                Bm[16] = 1
                Bm[18] = w_y
                Bm[19] = 1/radius
                Bm[27] = 1
                Bm[28] = 1
                Bm[30] = w_y
                Bm[31] = w_x
                Bm[48] = 1
                Bm[64] = 1
                Bm[75] = 1
                Bm[76] = 1
                Bm[83] = 1
                Bm[86] = 1
                Bm[95] = 1
                Bm[98] = 1

                # K = [B]^T [F] [B]
                for r in range(NE):
                    for q in range(13):
                        FB[r*13 + q] = 0.
                        for s in range(NE):
                            FB[r*13 + q] += F[r*NE + s]*Bm[s*13 + q]
                for p in range(13):
                    for q in range(13):
                        K[p*13 + q] = 0.
                        for r in range(NE):
                            K[p*13 + q] += Bm[r*13 + p]*FB[r*13 + q]

                # stresses of the strains and of the linear strains
                e[0] = u_x + 0.5*(w_x*w_x)
                e[1] = v_y + 0.5*(w_y*w_y) + w/radius
                e[2] = u_y + v_x + w_x*w_y
                e[3] = phix_x
                e[4] = phiy_y
                e[5] = phix_y + phiy_x
                e[6] = phiy + w_y
                e[7] = phix + w_x
                eL[0] = u_x
                eL[1] = v_y + w/radius
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
                # K += [H(sigma - sigma_L)], [H(sigma_L)] being assembled by
                # fkG_num, which is homogeneous of degree one in cs as linear
                # buckling requires
                K[56] += -sgL[0] + sg[0]
                K[57] += -sgL[2] + sg[2]
                K[69] += -sgL[2] + sg[2]
                K[70] += -sgL[1] + sg[1]

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
                        pA[6] = fAw*gAw
                        pA[7] = fAphix*gAphix
                        pA[8] = fAphiy*gAphiy
                        pA[9] = fAphix_x*gAphix
                        pA[10] = fAphix*gAphix_y
                        pA[11] = fAphiy_x*gAphiy
                        pA[12] = fAphiy*gAphiy_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[13]
                        TA[1] = pA[0]*K[1] + pA[1]*K[14]
                        TA[2] = pA[0]*K[2] + pA[1]*K[15]
                        TA[3] = pA[0]*K[3] + pA[1]*K[16]
                        TA[4] = pA[0]*K[4] + pA[1]*K[17]
                        TA[5] = pA[0]*K[5] + pA[1]*K[18]
                        TA[6] = pA[0]*K[6] + pA[1]*K[19]
                        TA[7] = pA[0]*K[7] + pA[1]*K[20]
                        TA[8] = pA[0]*K[8] + pA[1]*K[21]
                        TA[9] = pA[0]*K[9] + pA[1]*K[22]
                        TA[10] = pA[0]*K[10] + pA[1]*K[23]
                        TA[11] = pA[0]*K[11] + pA[1]*K[24]
                        TA[12] = pA[0]*K[12] + pA[1]*K[25]
                        TA[13] = pA[2]*K[26] + pA[3]*K[39]
                        TA[14] = pA[2]*K[27] + pA[3]*K[40]
                        TA[15] = pA[2]*K[28] + pA[3]*K[41]
                        TA[16] = pA[2]*K[29] + pA[3]*K[42]
                        TA[17] = pA[2]*K[30] + pA[3]*K[43]
                        TA[18] = pA[2]*K[31] + pA[3]*K[44]
                        TA[19] = pA[2]*K[32] + pA[3]*K[45]
                        TA[20] = pA[2]*K[33] + pA[3]*K[46]
                        TA[21] = pA[2]*K[34] + pA[3]*K[47]
                        TA[22] = pA[2]*K[35] + pA[3]*K[48]
                        TA[23] = pA[2]*K[36] + pA[3]*K[49]
                        TA[24] = pA[2]*K[37] + pA[3]*K[50]
                        TA[25] = pA[2]*K[38] + pA[3]*K[51]
                        TA[26] = pA[4]*K[52] + pA[5]*K[65] + pA[6]*K[78]
                        TA[27] = pA[4]*K[53] + pA[5]*K[66] + pA[6]*K[79]
                        TA[28] = pA[4]*K[54] + pA[5]*K[67] + pA[6]*K[80]
                        TA[29] = pA[4]*K[55] + pA[5]*K[68] + pA[6]*K[81]
                        TA[30] = pA[4]*K[56] + pA[5]*K[69] + pA[6]*K[82]
                        TA[31] = pA[4]*K[57] + pA[5]*K[70] + pA[6]*K[83]
                        TA[32] = pA[4]*K[58] + pA[5]*K[71] + pA[6]*K[84]
                        TA[33] = pA[4]*K[59] + pA[5]*K[72] + pA[6]*K[85]
                        TA[34] = pA[4]*K[60] + pA[5]*K[73] + pA[6]*K[86]
                        TA[35] = pA[4]*K[61] + pA[5]*K[74] + pA[6]*K[87]
                        TA[36] = pA[4]*K[62] + pA[5]*K[75] + pA[6]*K[88]
                        TA[37] = pA[4]*K[63] + pA[5]*K[76] + pA[6]*K[89]
                        TA[38] = pA[4]*K[64] + pA[5]*K[77] + pA[6]*K[90]
                        TA[39] = pA[7]*K[91] + pA[9]*K[117] + pA[10]*K[130]
                        TA[40] = pA[7]*K[92] + pA[9]*K[118] + pA[10]*K[131]
                        TA[41] = pA[7]*K[93] + pA[9]*K[119] + pA[10]*K[132]
                        TA[42] = pA[7]*K[94] + pA[9]*K[120] + pA[10]*K[133]
                        TA[43] = pA[7]*K[95] + pA[9]*K[121] + pA[10]*K[134]
                        TA[44] = pA[7]*K[96] + pA[9]*K[122] + pA[10]*K[135]
                        TA[45] = pA[7]*K[97] + pA[9]*K[123] + pA[10]*K[136]
                        TA[46] = pA[7]*K[98] + pA[9]*K[124] + pA[10]*K[137]
                        TA[47] = pA[7]*K[99] + pA[9]*K[125] + pA[10]*K[138]
                        TA[48] = pA[7]*K[100] + pA[9]*K[126] + pA[10]*K[139]
                        TA[49] = pA[7]*K[101] + pA[9]*K[127] + pA[10]*K[140]
                        TA[50] = pA[7]*K[102] + pA[9]*K[128] + pA[10]*K[141]
                        TA[51] = pA[7]*K[103] + pA[9]*K[129] + pA[10]*K[142]
                        TA[52] = pA[8]*K[104] + pA[11]*K[143] + pA[12]*K[156]
                        TA[53] = pA[8]*K[105] + pA[11]*K[144] + pA[12]*K[157]
                        TA[54] = pA[8]*K[106] + pA[11]*K[145] + pA[12]*K[158]
                        TA[55] = pA[8]*K[107] + pA[11]*K[146] + pA[12]*K[159]
                        TA[56] = pA[8]*K[108] + pA[11]*K[147] + pA[12]*K[160]
                        TA[57] = pA[8]*K[109] + pA[11]*K[148] + pA[12]*K[161]
                        TA[58] = pA[8]*K[110] + pA[11]*K[149] + pA[12]*K[162]
                        TA[59] = pA[8]*K[111] + pA[11]*K[150] + pA[12]*K[163]
                        TA[60] = pA[8]*K[112] + pA[11]*K[151] + pA[12]*K[164]
                        TA[61] = pA[8]*K[113] + pA[11]*K[152] + pA[12]*K[165]
                        TA[62] = pA[8]*K[114] + pA[11]*K[153] + pA[12]*K[166]
                        TA[63] = pA[8]*K[115] + pA[11]*K[154] + pA[12]*K[167]
                        TA[64] = pA[8]*K[116] + pA[11]*K[155] + pA[12]*K[168]
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
                                pB[6] = fBw*gBw
                                pB[7] = fBphix*gBphix
                                pB[8] = fBphiy*gBphiy
                                pB[9] = fBphix_x*gBphix
                                pB[10] = fBphix*gBphix_y
                                pB[11] = fBphiy_x*gBphiy
                                pB[12] = fBphiy*gBphiy_y
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
                                kCv[c] += weight*(TA[4]*pB[4] + TA[5]*pB[5] + TA[6]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[7]*pB[7] + TA[9]*pB[9] + TA[10]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[8]*pB[8] + TA[11]*pB[11] + TA[12]*pB[12])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[13]*pB[0] + TA[14]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[15]*pB[2] + TA[16]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[17]*pB[4] + TA[18]*pB[5] + TA[19]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[20]*pB[7] + TA[22]*pB[9] + TA[23]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[21]*pB[8] + TA[24]*pB[11] + TA[25]*pB[12])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[26]*pB[0] + TA[27]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[28]*pB[2] + TA[29]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[30]*pB[4] + TA[31]*pB[5] + TA[32]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[33]*pB[7] + TA[35]*pB[9] + TA[36]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[34]*pB[8] + TA[37]*pB[11] + TA[38]*pB[12])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[39]*pB[0] + TA[40]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[41]*pB[2] + TA[42]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[43]*pB[4] + TA[44]*pB[5] + TA[45]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[46]*pB[7] + TA[48]*pB[9] + TA[49]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+3
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[47]*pB[8] + TA[50]*pB[11] + TA[51]*pB[12])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[52]*pB[0] + TA[53]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[54]*pB[2] + TA[55]*pB[3])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[56]*pB[4] + TA[57]*pB[5] + TA[58]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+3
                                kCv[c] += weight*(TA[59]*pB[7] + TA[61]*pB[9] + TA[62]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+4
                                    kCc[c] = col+4
                                kCv[c] += weight*(TA[60]*pB[8] + TA[63]*pB[11] + TA[64]*pB[12])

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC


def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    r"""Geometric stiffness matrix of the linear stress state

    Integrates numerically the geometric stiffness matrix of the membrane
    stresses of the linear strains at ``cs``, superposed to the constant
    membrane state ``Nxx0``, ``Nyy0``, ``Nxy0``, with the rotations
    `\beta_x` and `\beta_y` of the non-linear terms of the membrane
    strains. The matrix is homogeneous of degree one in ``cs``, as the
    linear buckling eigenvalue problem requires.

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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the value of `w`, the rotations and their gradients), the strains
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
    cdef double K[169]
    cdef double TA[65]
    cdef double pA[13]
    cdef double pB[13]

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

    fdim = 1*m*m*n*n
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

                # K = [H(sigma)]
                for p in range(169):
                    K[p] = 0.
                K[56] = sg[0]
                K[57] = sg[2]
                K[69] = sg[2]
                K[70] = sg[1]

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
                        pA[6] = fAw*gAw
                        pA[7] = fAphix*gAphix
                        pA[8] = fAphiy*gAphiy
                        pA[9] = fAphix_x*gAphix
                        pA[10] = fAphix*gAphix_y
                        pA[11] = fAphiy_x*gAphiy
                        pA[12] = fAphiy*gAphiy_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[30] = pA[4]*K[56] + pA[5]*K[69]
                        TA[31] = pA[4]*K[57] + pA[5]*K[70]
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
                                pB[6] = fBw*gBw
                                pB[7] = fBphix*gBphix
                                pB[8] = fBphiy*gBphiy
                                pB[9] = fBphix_x*gBphix
                                pB[10] = fBphix*gBphix_y
                                pB[11] = fBphiy_x*gBphiy
                                pB[12] = fBphiy*gBphiy_y
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[30]*pB[4] + TA[31]*pB[5])

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

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, w, phix
    cdef double phiy, phix_x, phix_y, phiy_x, phiy_y
    cdef double e[8]
    cdef double sg[8]
    cdef double S[13]
    cdef double pA[13]
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

                # strains, stresses and S = [B]^T sigma
                e[0] = u_x + 0.5*(w_x*w_x)
                e[1] = v_y + 0.5*(w_y*w_y) + w/radius
                e[2] = u_y + v_x + w_x*w_y
                e[3] = phix_x
                e[4] = phiy_y
                e[5] = phix_y + phiy_x
                e[6] = phiy + w_y
                e[7] = phix + w_x
                for r in range(NE):
                    sg[r] = 0.
                    for s in range(NE):
                        sg[r] += F[r*NE + s]*e[s]
                S[0] = sg[0]
                S[1] = sg[2]
                S[2] = sg[2]
                S[3] = sg[1]
                S[4] = sg[0]*w_x + sg[2]*w_y + sg[7]
                S[5] = sg[1]*w_y + sg[2]*w_x + sg[6]
                S[6] = sg[1]/radius
                S[7] = sg[7]
                S[8] = sg[6]
                S[9] = sg[3]
                S[10] = sg[5]
                S[11] = sg[5]
                S[12] = sg[4]

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
                        pA[6] = fAw*gAw
                        pA[7] = fAphix*gAphix
                        pA[8] = fAphiy*gAphiy
                        pA[9] = fAphix_x*gAphix
                        pA[10] = fAphix*gAphix_y
                        pA[11] = fAphiy_x*gAphiy
                        pA[12] = fAphiy*gAphiy_y
                        fint[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        fint[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3])
                        fint[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[6]*S[6])
                        fint[col+3] += weight*(pA[7]*S[7] + pA[9]*S[9] + pA[10]*S[10])
                        fint[col+4] += weight*(pA[8]*S[8] + pA[11]*S[11] + pA[12]*S[12])

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
    = \boldsymbol{\Phi} \boldsymbol{c}` (the gradients of `u`, `v`, `w`, the value of `w`, the rotations and their gradients), the strains
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
    cdef double radius

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, w, phix
    cdef double phiy, phix_x, phix_y, phiy_x, phiy_y
    cdef double du_x, du_y, dv_x, dv_y, dw_x, dw_y, dw, dphix
    cdef double dphiy, dphix_x, dphix_y, dphiy_x, dphiy_y
    cdef double e[8]
    cdef double de[8]
    cdef double sg[8]
    cdef double dsg[8]
    cdef double S[13]
    cdef double pA[13]
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
                        dw += fAw*gAw*vs[col+2]
                        dphix += fAphix*gAphix*vs[col+3]
                        dphiy += fAphiy*gAphiy*vs[col+4]
                        dphix_x += fAphix_x*gAphix*vs[col+3]
                        dphix_y += fAphix*gAphix_y*vs[col+3]
                        dphiy_x += fAphiy_x*gAphiy*vs[col+4]
                        dphiy_y += fAphiy*gAphiy_y*vs[col+4]

                # strains, strain increments, stresses
                e[0] = u_x + 0.5*(w_x*w_x)
                e[1] = v_y + 0.5*(w_y*w_y) + w/radius
                e[2] = u_y + v_x + w_x*w_y
                e[3] = phix_x
                e[4] = phiy_y
                e[5] = phix_y + phiy_x
                e[6] = phiy + w_y
                e[7] = phix + w_x
                de[0] = du_x + dw_x*w_x
                de[1] = dv_y + dw/radius + dw_y*w_y
                de[2] = du_y + dv_x + dw_x*w_y + dw_y*w_x
                de[3] = dphix_x
                de[4] = dphiy_y
                de[5] = dphix_y + dphiy_x
                de[6] = dphiy + dw_y
                de[7] = dphix + dw_x
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
                S[0] = dsg[0]
                S[1] = dsg[2]
                S[2] = dsg[2]
                S[3] = dsg[1]
                S[4] = dsg[0]*w_x + dsg[2]*w_y + dsg[7] + dw_x*sg[0] + dw_y*sg[2]
                S[5] = dsg[1]*w_y + dsg[2]*w_x + dsg[6] + dw_x*sg[2] + dw_y*sg[1]
                S[6] = dsg[1]/radius
                S[7] = dsg[7]
                S[8] = dsg[6]
                S[9] = dsg[3]
                S[10] = dsg[5]
                S[11] = dsg[5]
                S[12] = dsg[4]

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
                        pA[6] = fAw*gAw
                        pA[7] = fAphix*gAphix
                        pA[8] = fAphiy*gAphiy
                        pA[9] = fAphix_x*gAphix
                        pA[10] = fAphix*gAphix_y
                        pA[11] = fAphiy_x*gAphiy
                        pA[12] = fAphiy*gAphiy_y
                        kTv[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        kTv[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3])
                        kTv[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[6]*S[6])
                        kTv[col+3] += weight*(pA[7]*S[7] + pA[9]*S[9] + pA[10]*S[10])
                        kTv[col+4] += weight*(pA[8]*S[8] + pA[11]*S[11] + pA[12]*S[12])

    return np.asarray(kTv)


def fkM_num(object shell, object I_input, int size,
        int row0, int col0, int nx, int ny):
    r"""Mass matrix, integrated numerically

    Integrates numerically the consistent mass matrix of the shell, with
    the moments of the density given for the whole domain or at each
    integration point, see :meth:`.Shell.calc_kM`.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    I_input : array-like
        Moments of the density through the thickness, and for the TSDT the
        thickness ``h``, ``I0, I1, I2``, shape ``(3,)`` for the whole domain or ``(nx, ny, 3)`` with one set per
        integration point.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    kM : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``I_input`` has an invalid shape or ``shell`` is not a
        :class:`.Shell`.

    Notes
    -----
    With the displacement field through the thickness `\{u(z), v(z),
    w(z)\}^\top = \boldsymbol{S}(z) \boldsymbol{c}` of the kinematics of the module:

    .. math::

        \boldsymbol{M} = \int \int_{d - h/2}^{d + h/2} \rho \,
        \boldsymbol{S}(z)^\top \boldsymbol{S}(z) \, dz \, dA

    with the offset `d`, where the integral through the thickness gives the moments of ``I_input``, `I_n = \int \rho z^n dz`, `n = 0, 1,
    2`, for the FSDT, and for the TSDT the moments `J_n`, `n = 0, 1, 2, 3, 4,
    6`, about the mid-surface and the thickness `h`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \le x \le x_2`, `y_1 \le y \le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
    cdef double [:, ::1] ftab, gtab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double r
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

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kMr, kMc
    cdef double [::1] kMv

    cdef double fAu, fAv, fAw, fAphix, fAphiy
    cdef double fBu, fBv, fBw, fBphix, fBphiy
    cdef double gAu, gAv, gAw, gAphix, gAphiy
    cdef double gBu, gBv, gBw, gBphix, gBphiy
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # moments of the density through the thickness, I0, I1, I2, as a 3-D
    # matrix [nx, ny, 3] with one set for each of the nx * ny integration
    # points, or as a 1-D array [3] for the whole integration domain
    cdef double I0, I1, I2
    cdef double [:, :, ::1] I_nxny

    cdef int one_set_each_point = 0

    I_input = np.asarray(I_input, dtype=DOUBLE)
    if I_input.shape == (nx, ny, 3):
        I_nxny = np.ascontiguousarray(I_input)
        one_set_each_point = 1
    elif I_input.shape == (3, ):
        # creating dummy 3-D array that is not used
        I_nxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        # using constant moments for all the integration domain
        I0, I1, I2 = I_input
    else:
        raise ValueError('Invalid shape for I_input, it must be (nx, ny, 3) or (3, )')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
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

    fdim = 9*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kMr = np.zeros((fdim,), dtype=INT)
    kMc = np.zeros((fdim,), dtype=INT)
    kMv = np.zeros((fdim,), dtype=DOUBLE)

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

    ftab = np.zeros((5, m), dtype=DOUBLE)
    gtab = np.zeros((5, n), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

                if one_set_each_point == 1:
                    I0 = I_nxny[ptx, pty, 0]
                    I1 = I_nxny[ptx, pty, 1]
                    I2 = I_nxny[ptx, pty, 2]

                # kM
                c = -1
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[2, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[3, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[4, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[2, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[3, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[4, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                for i in range(m):
                    fAu = ftab[0, i]
                    fAv = ftab[1, i]
                    fAw = ftab[2, i]
                    fAphix = ftab[3, i]
                    fAphiy = ftab[4, i]

                    for k in range(m):
                        fBu = ftab[0, k]
                        fBv = ftab[1, k]
                        fBw = ftab[2, k]
                        fBphix = ftab[3, k]
                        fBphiy = ftab[4, k]

                        for j in range(n):
                            gAu = gtab[0, j]
                            gAv = gtab[1, j]
                            gAw = gtab[2, j]
                            gAphix = gtab[3, j]
                            gAphiy = gtab[4, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                #NOTE symmetry assumption True if no follower forces are used
                                if row > col:
                                    continue

                                gBu = gtab[0, l]
                                gBv = gtab[1, l]
                                gBw = gtab[2, l]
                                gBphix = gtab[3, l]
                                gBphiy = gtab[4, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( I0*fAu*fBu*gAu*gBu )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+0
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( I1*fAu*fBphix*gAu*gBphix )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( I0*fAv*fBv*gAv*gBv )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+1
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( I1*fAv*fBphiy*gAv*gBphiy )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+2
                                    kMc[c] = col+2
                                kMv[c] += weight*(intx*inty/4)*( I0*fAw*fBw*gAw*gBw )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+0
                                kMv[c] += weight*(intx*inty/4)*( I1*fAphix*fBu*gAphix*gBu )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+3
                                    kMc[c] = col+3
                                kMv[c] += weight*(intx*inty/4)*( I2*fAphix*fBphix*gAphix*gBphix )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+1
                                kMv[c] += weight*(intx*inty/4)*( I1*fAphiy*fBv*gAphiy*gBv )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kMr[c] = row+4
                                    kMc[c] = col+4
                                kMv[c] += weight*(intx*inty/4)*( I2*fAphiy*fBphiy*gAphiy*gBphiy )

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


def fkAx_num(object shell, int size, int row0, int col0, int nx, int ny):
    r"""Aerodynamic stiffness matrix of the piston theory, integrated numerically

    ``fkAx_num`` gives the matrix of the flow along `x` and ``fkAy_num``
    that of the flow along `y`, with the parameters ``shell.beta`` and,
    for the flow along `x`, ``shell.gamma`` of :meth:`.Shell.calc_kA`.

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
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    kA : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. The matrix is assembled in full.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With `\boldsymbol{S}^w` the row of the approximation functions of
    `w`, the matrices added to the stiffness matrix are:

    .. math::

        \boldsymbol{K}_{A_x} = -\int \left[\beta (\boldsymbol{S}^w_{,x})^\top
        \boldsymbol{S}^w + \gamma (\boldsymbol{S}^w)^\top
        \boldsymbol{S}^w\right] dA \qquad
        \boldsymbol{K}_{A_y} = -\beta \int (\boldsymbol{S}^w_{,y})^\top
        \boldsymbol{S}^w \, dA

    `\gamma = 0` for the flat plates; they are not symmetric.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \le x \le x_2`, `y_1 \le y \le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
    cdef double [:, ::1] ftab, gtab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double r
    cdef double a, b, beta, intx, inty, gamma
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

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kAr, kAc
    cdef double [::1] kAv

    cdef double fAw, fAw_xi
    cdef double fBw
    cdef double gAw
    cdef double gBw
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    beta = shell.beta
    gamma = shell.gamma
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

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kAr = np.zeros((fdim,), dtype=INT)
    kAc = np.zeros((fdim,), dtype=INT)
    kAv = np.zeros((fdim,), dtype=DOUBLE)

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

    ftab = np.zeros((2, m), dtype=DOUBLE)
    gtab = np.zeros((1, n), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

                # kA
                c = -1
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[1, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1w, y1wr, y2w, y2wr)

                for i in range(m):
                    fAw = ftab[0, i]
                    fAw_xi = ftab[1, i]

                    for k in range(m):
                        fBw = ftab[0, k]

                        for j in range(n):
                            gAw = gtab[0, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                gBw = gtab[0, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kAr[c] = row+2
                                    kAc[c] = col+2
                                kAv[c] += weight*(intx*inty/4)*( -fAw*fBw*gAw*gBw*gamma - 2*beta*fAw_xi*fBw*gAw*gBw/a )

    kA = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kA


def fkAy_num(object shell, int size, int row0, int col0, int nx, int ny):
    r"""Aerodynamic stiffness matrix of the piston theory, integrated numerically

    ``fkAx_num`` gives the matrix of the flow along `x` and ``fkAy_num``
    that of the flow along `y`, with the parameters ``shell.beta`` and,
    for the flow along `x`, ``shell.gamma`` of :meth:`.Shell.calc_kA`.

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
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    kA : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. The matrix is assembled in full.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With `\boldsymbol{S}^w` the row of the approximation functions of
    `w`, the matrices added to the stiffness matrix are:

    .. math::

        \boldsymbol{K}_{A_x} = -\int \left[\beta (\boldsymbol{S}^w_{,x})^\top
        \boldsymbol{S}^w + \gamma (\boldsymbol{S}^w)^\top
        \boldsymbol{S}^w\right] dA \qquad
        \boldsymbol{K}_{A_y} = -\beta \int (\boldsymbol{S}^w_{,y})^\top
        \boldsymbol{S}^w \, dA

    `\gamma = 0` for the flat plates; they are not symmetric.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \le x \le x_2`, `y_1 \le y \le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
    cdef double [:, ::1] ftab, gtab
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double r
    cdef double a, b, beta, intx, inty
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

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kAr, kAc
    cdef double [::1] kAv

    cdef double fAw
    cdef double fBw
    cdef double gAw_eta
    cdef double gBw
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    r = shell.r
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    beta = shell.beta
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

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kAr = np.zeros((fdim,), dtype=INT)
    kAc = np.zeros((fdim,), dtype=INT)
    kAv = np.zeros((fdim,), dtype=DOUBLE)

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

    ftab = np.zeros((1, m), dtype=DOUBLE)
    gtab = np.zeros((2, n), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]

                # kA
                c = -1
                # approximation functions evaluated once, see panels/dev/tabulate.py
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[1, j] = f(j, eta, y1w, y1wr, y2w, y2wr)

                for i in range(m):
                    fAw = ftab[0, i]

                    for k in range(m):
                        fBw = ftab[0, k]

                        for j in range(n):
                            gAw_eta = gtab[0, j]

                            for l in range(n):

                                row = row0 + DOF*(j*m + i)
                                col = col0 + DOF*(l*m + k)

                                gBw = gtab[1, l]

                                c += 1
                                if ptx == 0 and pty == 0:
                                    kAr[c] = row+2
                                    kAc[c] = col+2
                                kAv[c] += weight*(intx*inty/4)*( -2*beta*fAw*fBw*gAw_eta*gBw/b )

    kA = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kA


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

            ax = qf*v_x*w_y - qf*v_y*w_x - qf*w_x*ww/r - w_x
        ay = -qf*u_x*w_y + qf*u_y*w_x - w_y
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
                u_x = u_y = v_x = v_y = ww = w_x = w_y = 0.
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
                        ww += cs[col+2]*fAw*gAw
                        w_x += cs[col+2]*(2/a)*fAw_xi*gAw
                        w_y += cs[col+2]*(2/b)*fAw*gAw_eta

                ax = qf*v_x*w_y - qf*v_y*w_x - qf*w_x*ww/r - w_x
                ay = -qf*u_x*w_y + qf*u_y*w_x - w_y
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

    fdim = 7*m*m*n*n
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
                u_x = u_y = v_x = v_y = ww = w_x = w_y = 0.
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
                                kCv[c] += weight*( 2*fAu*fBv*gAu*gBv_eta*qf*w_x/b - 2*fAu*fBv_xi*gAu*gBv*qf*w_y/a )
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+0
                                    kCc[c] = col+2
                                kCv[c] += weight*( fAu*fBw*gAu*gBw*qf*w_x/r - 2*fAu*fBw*gAu*gBw_eta*qf*v_x/b + 2*fAu*fBw_xi*gAu*gBw*qf*v_y/a + 2*fAu*fBw_xi*gAu*gBw*qf*ww/(a*r) + 2*fAu*fBw_xi*gAu*gBw/a )
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
