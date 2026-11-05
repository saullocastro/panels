#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Cylindrical shells using the classical laminated plate theory (CLPT) with
Green-Lagrange membrane strains

Numerically integrated kernels of the model ``'cylshell_clpt_gl'``: the
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
which vanish for any rigid motion, with the changes of curvature of the
Sanders-Koiter kinematics of :mod:`panels.models.cylshell_clpt_sanders`:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} (u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2)

    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \left(u_{,y}^2 +
    \left(v_{,y} + \frac{w}{r}\right)^2 + \left(w_{,y} -
    \frac{v}{r}\right)^2\right)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} \left(v_{,y} +
    \frac{w}{r}\right) + w_{,x} \left(w_{,y} - \frac{v}{r}\right)

    \kappa_{xx} = -w_{,xx} \qquad \kappa_{yy} = -w_{,yy} + \frac{v_{,y}}{r}
    \qquad \kappa_{xy} = -2 w_{,xy} + \frac{3}{2} \frac{v_{,x}}{r} -
    \frac{1}{2} \frac{u_{,y}}{r}

with `y` the arc length along the circumference and `w` positive outwards,
the membrane strains of the Flügge-Lur'e-Byrne non-linear theory as
given by Amabili [1]_, Sec. 1.3.6, Eqs. (1.118)-(1.120). They contain the
non-linear terms of :mod:`panels.models.cylshell_clpt_sanders` and also
the quadratic terms of the in-plane gradients, among which those of the
rotation about the normal, `(v_{,x} - u_{,y})/2`, which Sanders [2]_ keeps
and :mod:`panels.models.cylshell_clpt_sanders` omits; under an axial
membrane force `N_{xx}` the geometric stiffness has the term `N_{xx}
v_{,x}^2` besides `N_{xx} w_{,x}^2`, which about halves the column
buckling load of a long tube compared with the Sanders kinematics. The changes of curvature are
linear, measured in the fixed local basis.

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
    With the generalized strains of the kinematics of the module,
    `\boldsymbol{\varepsilon} = (\boldsymbol{B}_L + \frac{1}{2}
    \boldsymbol{B}_{NL}(\boldsymbol{c})) \boldsymbol{c}`, their variation
    `\delta \boldsymbol{\varepsilon} = \boldsymbol{B} \delta \boldsymbol{c}`,
    `\boldsymbol{B} = \boldsymbol{B}_L + \boldsymbol{B}_{NL}(\boldsymbol{c})`,
    and the constitutive matrix `\boldsymbol{F}`:

    .. math::

        \boldsymbol{K}_C = \int \boldsymbol{B}^\top \boldsymbol{F}
        \boldsymbol{B} \, dA + \int \boldsymbol{G}^\top
        \boldsymbol{N}_{NL} \boldsymbol{G} \, dA

    with `\boldsymbol{G} \boldsymbol{c} = \{u_{,x}, u_{,y}, v_{,x}, v_{,y} + w/r,
    w_{,x}, w_{,y} - v/r\}^\top` the gradients of the displacements
    in the local basis, with one
    block `\boldsymbol{N}` per displacement, and `\boldsymbol{N}_{NL}` the membrane stress resultants of the
    non-linear part of the strains, `\frac{1}{2} \boldsymbol{B}_{NL}
    \boldsymbol{c}`. With ``NLgeom=0``, `\boldsymbol{B}_{NL} = \boldsymbol{0}`
    and `\boldsymbol{K}_C = \boldsymbol{K}_{C0}`. The membrane stress of the
    linear strains enters :func:`fkG_num`, such that ``fkC_num(NLgeom=1) +
    fkG_num`` is the exact Jacobian of :func:`calc_fint`.

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
    cdef double A11, A12, A16, A22, A26, A66
    cdef double NxxNL, NyyNL, NxyNL
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAw_xx
    cdef double gAu, gAu_y, gAv, gAv_y, gAw, gAw_y, gAw_yy
    cdef double fBu, fBu_x, fBv, fBv_x, fBw, fBw_x, fBw_xx
    cdef double gBu, gBu_y, gBv, gBv_y, gBw, gBw_y, gBw_yy
    cdef double Bm[66]
    cdef double FB[66]
    cdef double K[121]
    cdef double TA[33]
    cdef double pA[11]
    cdef double pB[11]

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

    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)

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
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

                # displacement gradients at the point, zero when NLgeom == 0
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w_x = 0.
                w_y = 0.
                v = 0.
                w = 0.
                if NLgeom == 1:
                    for j in range(n):
                        gAu = gtab[0, j]
                        gAu_y = gtab[1, j]
                        gAv = gtab[2, j]
                        gAv_y = gtab[3, j]
                        gAw = gtab[4, j]
                        gAw_y = gtab[5, j]
                        gAw_yy = gtab[6, j]
                        for i in range(m):
                            fAu = ftab[0, i]
                            fAu_x = ftab[1, i]
                            fAv = ftab[2, i]
                            fAv_x = ftab[3, i]
                            fAw = ftab[4, i]
                            fAw_x = ftab[5, i]
                            fAw_xx = ftab[6, i]
                            col = col0 + DOF*(j*m + i)
                            u_x += fAu_x*gAu*cs[col+0]
                            u_y += fAu*gAu_y*cs[col+0]
                            v_x += fAv_x*gAv*cs[col+1]
                            v_y += fAv*gAv_y*cs[col+1]
                            w_x += fAw_x*gAw*cs[col+2]
                            w_y += fAw*gAw_y*cs[col+2]
                            v += fAv*gAv*cs[col+1]
                            w += fAw*gAw*cs[col+2]

                A11 = F[0*6 + 0]
                A12 = F[0*6 + 1]
                A16 = F[0*6 + 2]
                A22 = F[1*6 + 1]
                A26 = F[1*6 + 2]
                A66 = F[2*6 + 2]

                # [B] = d eps/d d at the point
                for p in range(66):
                    Bm[p] = 0.
                Bm[0] = u_x + 1
                Bm[2] = v_x
                Bm[4] = w_x
                Bm[12] = u_y
                Bm[14] = v_y + 1 + w/radius
                Bm[16] = w_y - v/radius
                Bm[17] = -w_y/radius + v/(radius*radius)
                Bm[18] = v_y/radius + 1/radius + w/(radius*radius)
                Bm[22] = u_y
                Bm[23] = u_x + 1
                Bm[24] = v_y + 1 + w/radius
                Bm[25] = v_x
                Bm[26] = w_y - v/radius
                Bm[27] = w_x
                Bm[28] = -w_x/radius
                Bm[29] = v_x/radius
                Bm[41] = -1
                Bm[47] = 1/radius
                Bm[53] = -1
                Bm[56] = -0.5/radius
                Bm[57] = 1.5/radius
                Bm[65] = -2

                # K = [B]^T [ABD] [B] + [H(N_NL)]
                for r in range(6):
                    for q in range(11):
                        FB[r*11 + q] = 0.
                        for s in range(6):
                            FB[r*11 + q] += F[r*6 + s]*Bm[s*11 + q]
                for p in range(11):
                    for q in range(11):
                        K[p*11 + q] = 0.
                        for r in range(6):
                            K[p*11 + q] += Bm[r*11 + p]*FB[r*11 + q]
                # membrane stresses of the quadratic strains, whose
                # geometric stiffness KGNL is collected here such that
                # fkG_num is homogeneous of degree one in cs, as linear
                # buckling requires
                NxxNL = 0.5*A11*(u_x*u_x) + 0.5*A11*(v_x*v_x) + 0.5*A11*(w_x*w_x) + 0.5*A12*(u_y*u_y) + 0.5*A12*(v_y*v_y) + 0.5*A12*(w_y*w_y) - A12*v*w_y/radius + A12*v_y*w/radius + 0.5*A12*(v*v)/(radius*radius) + 0.5*A12*(w*w)/(radius*radius) + A16*u_x*u_y + A16*v_x*v_y + A16*w_x*w_y - A16*v*w_x/radius + A16*v_x*w/radius
                NyyNL = 0.5*A12*(u_x*u_x) + 0.5*A12*(v_x*v_x) + 0.5*A12*(w_x*w_x) + 0.5*A22*(u_y*u_y) + 0.5*A22*(v_y*v_y) + 0.5*A22*(w_y*w_y) - A22*v*w_y/radius + A22*v_y*w/radius + 0.5*A22*(v*v)/(radius*radius) + 0.5*A22*(w*w)/(radius*radius) + A26*u_x*u_y + A26*v_x*v_y + A26*w_x*w_y - A26*v*w_x/radius + A26*v_x*w/radius
                NxyNL = 0.5*A16*(u_x*u_x) + 0.5*A16*(v_x*v_x) + 0.5*A16*(w_x*w_x) + 0.5*A26*(u_y*u_y) + 0.5*A26*(v_y*v_y) + 0.5*A26*(w_y*w_y) - A26*v*w_y/radius + A26*v_y*w/radius + 0.5*A26*(v*v)/(radius*radius) + 0.5*A26*(w*w)/(radius*radius) + A66*u_x*u_y + A66*v_x*v_y + A66*w_x*w_y - A66*v*w_x/radius + A66*v_x*w/radius
                K[0] += NxxNL
                K[1] += NxyNL
                K[11] += NxyNL
                K[12] += NyyNL
                K[24] += NxxNL
                K[25] += NxyNL
                K[29] += NxyNL/radius
                K[35] += NxyNL
                K[36] += NyyNL
                K[40] += NyyNL/radius
                K[48] += NxxNL
                K[49] += NxyNL
                K[50] += -NxyNL/radius
                K[59] += NxyNL
                K[60] += NyyNL
                K[61] += -NyyNL/radius
                K[70] += -NxyNL/radius
                K[71] += -NyyNL/radius
                K[72] += NyyNL/(radius*radius)
                K[79] += NxyNL/radius
                K[80] += NyyNL/radius
                K[84] += NyyNL/(radius*radius)

                c = -1
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAw_yy = gtab[6, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        row = row0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAw_xx*gAw
                        pA[9] = fAw*gAw_yy
                        pA[10] = fAw_x*gAw_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[11]
                        TA[1] = pA[0]*K[1] + pA[1]*K[12]
                        TA[2] = pA[0]*K[2] + pA[1]*K[13]
                        TA[3] = pA[0]*K[3] + pA[1]*K[14]
                        TA[4] = pA[0]*K[4] + pA[1]*K[15]
                        TA[5] = pA[0]*K[5] + pA[1]*K[16]
                        TA[6] = pA[0]*K[6] + pA[1]*K[17]
                        TA[7] = pA[0]*K[7] + pA[1]*K[18]
                        TA[8] = pA[0]*K[8] + pA[1]*K[19]
                        TA[9] = pA[0]*K[9] + pA[1]*K[20]
                        TA[10] = pA[0]*K[10] + pA[1]*K[21]
                        TA[11] = pA[2]*K[22] + pA[3]*K[33] + pA[6]*K[66]
                        TA[12] = pA[2]*K[23] + pA[3]*K[34] + pA[6]*K[67]
                        TA[13] = pA[2]*K[24] + pA[3]*K[35] + pA[6]*K[68]
                        TA[14] = pA[2]*K[25] + pA[3]*K[36] + pA[6]*K[69]
                        TA[15] = pA[2]*K[26] + pA[3]*K[37] + pA[6]*K[70]
                        TA[16] = pA[2]*K[27] + pA[3]*K[38] + pA[6]*K[71]
                        TA[17] = pA[2]*K[28] + pA[3]*K[39] + pA[6]*K[72]
                        TA[18] = pA[2]*K[29] + pA[3]*K[40] + pA[6]*K[73]
                        TA[19] = pA[2]*K[30] + pA[3]*K[41] + pA[6]*K[74]
                        TA[20] = pA[2]*K[31] + pA[3]*K[42] + pA[6]*K[75]
                        TA[21] = pA[2]*K[32] + pA[3]*K[43] + pA[6]*K[76]
                        TA[22] = pA[4]*K[44] + pA[5]*K[55] + pA[7]*K[77] + pA[8]*K[88] + pA[9]*K[99] + pA[10]*K[110]
                        TA[23] = pA[4]*K[45] + pA[5]*K[56] + pA[7]*K[78] + pA[8]*K[89] + pA[9]*K[100] + pA[10]*K[111]
                        TA[24] = pA[4]*K[46] + pA[5]*K[57] + pA[7]*K[79] + pA[8]*K[90] + pA[9]*K[101] + pA[10]*K[112]
                        TA[25] = pA[4]*K[47] + pA[5]*K[58] + pA[7]*K[80] + pA[8]*K[91] + pA[9]*K[102] + pA[10]*K[113]
                        TA[26] = pA[4]*K[48] + pA[5]*K[59] + pA[7]*K[81] + pA[8]*K[92] + pA[9]*K[103] + pA[10]*K[114]
                        TA[27] = pA[4]*K[49] + pA[5]*K[60] + pA[7]*K[82] + pA[8]*K[93] + pA[9]*K[104] + pA[10]*K[115]
                        TA[28] = pA[4]*K[50] + pA[5]*K[61] + pA[7]*K[83] + pA[8]*K[94] + pA[9]*K[105] + pA[10]*K[116]
                        TA[29] = pA[4]*K[51] + pA[5]*K[62] + pA[7]*K[84] + pA[8]*K[95] + pA[9]*K[106] + pA[10]*K[117]
                        TA[30] = pA[4]*K[52] + pA[5]*K[63] + pA[7]*K[85] + pA[8]*K[96] + pA[9]*K[107] + pA[10]*K[118]
                        TA[31] = pA[4]*K[53] + pA[5]*K[64] + pA[7]*K[86] + pA[8]*K[97] + pA[9]*K[108] + pA[10]*K[119]
                        TA[32] = pA[4]*K[54] + pA[5]*K[65] + pA[7]*K[87] + pA[8]*K[98] + pA[9]*K[109] + pA[10]*K[120]
                        for l in range(n):
                            gBu = gtab[0, l]
                            gBu_y = gtab[1, l]
                            gBv = gtab[2, l]
                            gBv_y = gtab[3, l]
                            gBw = gtab[4, l]
                            gBw_y = gtab[5, l]
                            gBw_yy = gtab[6, l]
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
                                pB[0] = fBu_x*gBu
                                pB[1] = fBu*gBu_y
                                pB[2] = fBv_x*gBv
                                pB[3] = fBv*gBv_y
                                pB[4] = fBw_x*gBw
                                pB[5] = fBw*gBw_y
                                pB[6] = fBv*gBv
                                pB[7] = fBw*gBw
                                pB[8] = fBw_xx*gBw
                                pB[9] = fBw*gBw_yy
                                pB[10] = fBw_x*gBw_y
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
                                kCv[c] += weight*(TA[4]*pB[4] + TA[5]*pB[5] + TA[7]*pB[7] + TA[8]*pB[8] + TA[9]*pB[9] + TA[10]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[11]*pB[0] + TA[12]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[13]*pB[2] + TA[14]*pB[3] + TA[17]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+1
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[15]*pB[4] + TA[16]*pB[5] + TA[18]*pB[7] + TA[19]*pB[8] + TA[20]*pB[9] + TA[21]*pB[10])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+0
                                kCv[c] += weight*(TA[22]*pB[0] + TA[23]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+1
                                kCv[c] += weight*(TA[24]*pB[2] + TA[25]*pB[3] + TA[28]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kCr[c] = row+2
                                    kCc[c] = col+2
                                kCv[c] += weight*(TA[26]*pB[4] + TA[27]*pB[5] + TA[29]*pB[7] + TA[30]*pB[8] + TA[31]*pB[9] + TA[32]*pB[10])

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC


def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    r"""Geometric stiffness matrix of the linear membrane stress state

    Integrates numerically the geometric stiffness matrix of the membrane
    stress resultants of the linear strains at ``cs``, superposed to
    the constant state ``Nxx0``, ``Nyy0``, ``Nxy0``. The matrix is
    homogeneous of degree one in ``cs``, as the linear buckling
    eigenvalue problem requires; the stress of the non-linear strains
    enters :func:`fkC_num`.

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
    With `\boldsymbol{G} \boldsymbol{c} = \{u_{,x}, u_{,y}, v_{,x}, v_{,y} + w/r,
    w_{,x}, w_{,y} - v/r\}^\top` the gradients of the displacements
    in the local basis, with one
    block `\boldsymbol{N}` per displacement, and the membrane stress resultants
    `\boldsymbol{N} = \boldsymbol{N}_0 + \boldsymbol{N}_L`,
    `\boldsymbol{N}_L` those of `\boldsymbol{F} \boldsymbol{B}_L
    \boldsymbol{c}`, the linear generalized strains of the kinematics of the module:

    .. math::

        \boldsymbol{K}_G = \int \boldsymbol{G}^\top \boldsymbol{N}
        \boldsymbol{G} \, dA, \qquad \boldsymbol{N} = \begin{bmatrix} N_{xx}
        & N_{xy} \\ N_{xy} & N_{yy} \end{bmatrix}

    The tangent stiffness matrix `\boldsymbol{K}_{C0} +
    \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL} +
    \boldsymbol{K}_G(\boldsymbol{N}_0 + \boldsymbol{N}_L)`, the first three terms from :func:`fkC_num` and the
    last from this function, is
    the exact Jacobian of :func:`calc_fint` when `\boldsymbol{N}_0 =
    \boldsymbol{0}`.

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

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w, w_xx, w_yy, w_xy
    cdef double Nxx, Nyy, Nxy
    cdef double e[6]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAw_xx
    cdef double gAu, gAu_y, gAv, gAv_y, gAw, gAw_y, gAw_yy
    cdef double fBu, fBu_x, fBv, fBv_x, fBw, fBw_x, fBw_xx
    cdef double gBu, gBu_y, gBv, gBv_y, gBw, gBw_y, gBw_yy
    cdef double K[121]
    cdef double TA[33]
    cdef double pA[11]
    cdef double pB[11]

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

    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)

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
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = (2/a)*fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

                # generalized derivatives at the point
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w_x = 0.
                w_y = 0.
                v = 0.
                w = 0.
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
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        col = col0 + DOF*(j*m + i)
                        u_x += fAu_x*gAu*cs[col+0]
                        u_y += fAu*gAu_y*cs[col+0]
                        v_x += fAv_x*gAv*cs[col+1]
                        v_y += fAv*gAv_y*cs[col+1]
                        w_x += fAw_x*gAw*cs[col+2]
                        w_y += fAw*gAw_y*cs[col+2]
                        v += fAv*gAv*cs[col+1]
                        w += fAw*gAw*cs[col+2]
                        w_xx += fAw_xx*gAw*cs[col+2]
                        w_yy += fAw*gAw_yy*cs[col+2]
                        w_xy += fAw_x*gAw_y*cs[col+2]

                # linear strains and membrane stresses
                e[0] = u_x
                e[1] = v_y + w/radius
                e[2] = u_y + v_x
                e[3] = -w_xx
                e[4] = -w_yy + v_y/radius
                e[5] = -2*w_xy - 0.5*u_y/radius + 1.5*v_x/radius
                Nxx = Nxx0
                Nyy = Nyy0
                Nxy = Nxy0
                for s in range(6):
                    Nxx += F[0*6 + s]*e[s]
                    Nyy += F[1*6 + s]*e[s]
                    Nxy += F[2*6 + s]*e[s]

                # K = [H(N)]
                K[0] = Nxx
                K[1] = Nxy
                K[11] = Nxy
                K[12] = Nyy
                K[24] = Nxx
                K[25] = Nxy
                K[29] = Nxy/radius
                K[35] = Nxy
                K[36] = Nyy
                K[40] = Nyy/radius
                K[48] = Nxx
                K[49] = Nxy
                K[50] = -Nxy/radius
                K[59] = Nxy
                K[60] = Nyy
                K[61] = -Nyy/radius
                K[70] = -Nxy/radius
                K[71] = -Nyy/radius
                K[72] = Nyy/(radius*radius)
                K[79] = Nxy/radius
                K[80] = Nyy/radius
                K[84] = Nyy/(radius*radius)

                c = -1
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAw_yy = gtab[6, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        row = row0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAw_xx*gAw
                        pA[9] = fAw*gAw_yy
                        pA[10] = fAw_x*gAw_y
                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]
                        TA[0] = pA[0]*K[0] + pA[1]*K[11]
                        TA[1] = pA[0]*K[1] + pA[1]*K[12]
                        TA[13] = pA[2]*K[24] + pA[3]*K[35]
                        TA[14] = pA[2]*K[25] + pA[3]*K[36]
                        TA[15] = pA[6]*K[70]
                        TA[16] = pA[6]*K[71]
                        TA[17] = pA[6]*K[72]
                        TA[18] = pA[2]*K[29] + pA[3]*K[40]
                        TA[24] = pA[7]*K[79]
                        TA[25] = pA[7]*K[80]
                        TA[26] = pA[4]*K[48] + pA[5]*K[59]
                        TA[27] = pA[4]*K[49] + pA[5]*K[60]
                        TA[28] = pA[4]*K[50] + pA[5]*K[61]
                        TA[29] = pA[7]*K[84]
                        for l in range(n):
                            gBu = gtab[0, l]
                            gBu_y = gtab[1, l]
                            gBv = gtab[2, l]
                            gBv_y = gtab[3, l]
                            gBw = gtab[4, l]
                            gBw_y = gtab[5, l]
                            gBw_yy = gtab[6, l]
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
                                pB[0] = fBu_x*gBu
                                pB[1] = fBu*gBu_y
                                pB[2] = fBv_x*gBv
                                pB[3] = fBv*gBv_y
                                pB[4] = fBw_x*gBw
                                pB[5] = fBw*gBw_y
                                pB[6] = fBv*gBv
                                pB[7] = fBw*gBw
                                pB[8] = fBw_xx*gBw
                                pB[9] = fBw*gBw_yy
                                pB[10] = fBw_x*gBw_y
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+0
                                    kGc[c] = col+0
                                kGv[c] += weight*(TA[0]*pB[0] + TA[1]*pB[1])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[13]*pB[2] + TA[14]*pB[3] + TA[17]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+1
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[15]*pB[4] + TA[16]*pB[5] + TA[18]*pB[7])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+1
                                kGv[c] += weight*(TA[24]*pB[2] + TA[25]*pB[3] + TA[28]*pB[6])
                                c += 1
                                if ptx == 0 and pty == 0:
                                    kGr[c] = row+2
                                    kGc[c] = col+2
                                kGv[c] += weight*(TA[26]*pB[4] + TA[27]*pB[5] + TA[29]*pB[7])

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

    cdef double u_x, u_y, v_x, v_y, w_x, w_y, v, w, w_xx, w_yy, w_xy
    cdef double e[6]
    cdef double sg[6]
    cdef double S[11]
    cdef double pA[11]
    cdef double fAu, fAu_x, fAv, fAv_x, fAw, fAw_x, fAw_xx
    cdef double gAu, gAu_y, gAv, gAv_y, gAw, gAw_y, gAw_yy

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

    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)

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
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)

                # generalized derivatives at the point
                u_x = 0.
                u_y = 0.
                v_x = 0.
                v_y = 0.
                w_x = 0.
                w_y = 0.
                v = 0.
                w = 0.
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
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        col = col0 + DOF*(j*m + i)
                        u_x += fAu_x*gAu*cs[col+0]
                        u_y += fAu*gAu_y*cs[col+0]
                        v_x += fAv_x*gAv*cs[col+1]
                        v_y += fAv*gAv_y*cs[col+1]
                        w_x += fAw_x*gAw*cs[col+2]
                        w_y += fAw*gAw_y*cs[col+2]
                        v += fAv*gAv*cs[col+1]
                        w += fAw*gAw*cs[col+2]
                        w_xx += fAw_xx*gAw*cs[col+2]
                        w_yy += fAw*gAw_yy*cs[col+2]
                        w_xy += fAw_x*gAw_y*cs[col+2]

                # strains, stresses and S = [B]^T sigma
                e[0] = 0.5*(u_x*u_x) + u_x + 0.5*(v_x*v_x) + 0.5*(w_x*w_x)
                e[1] = 0.5*(u_y*u_y) + 0.5*(v_y*v_y) + v_y + 0.5*(w_y*w_y) - v*w_y/radius + v_y*w/radius + w/radius + 0.5*(v*v)/(radius*radius) + 0.5*(w*w)/(radius*radius)
                e[2] = u_x*u_y + u_y + v_x*v_y + v_x + w_x*w_y - v*w_x/radius + v_x*w/radius
                e[3] = -w_xx
                e[4] = -w_yy + v_y/radius
                e[5] = -2*w_xy - 0.5*u_y/radius + 1.5*v_x/radius
                for r in range(6):
                    sg[r] = 0.
                    for s in range(6):
                        sg[r] += F[r*6 + s]*e[s]
                S[0] = sg[0]*u_x + sg[0] + sg[2]*u_y
                S[1] = sg[1]*u_y + sg[2]*u_x + sg[2] - 0.5*sg[5]/radius
                S[2] = sg[0]*v_x + sg[2]*v_y + sg[2] + sg[2]*w/radius + 1.5*sg[5]/radius
                S[3] = sg[1]*v_y + sg[1] + sg[2]*v_x + sg[1]*w/radius + sg[4]/radius
                S[4] = sg[0]*w_x + sg[2]*w_y - sg[2]*v/radius
                S[5] = sg[1]*w_y + sg[2]*w_x - sg[1]*v/radius
                S[6] = -sg[1]*w_y/radius - sg[2]*w_x/radius + sg[1]*v/(radius*radius)
                S[7] = sg[1]*v_y/radius + sg[1]/radius + sg[2]*v_x/radius + sg[1]*w/(radius*radius)
                S[8] = -sg[3]
                S[9] = -sg[4]
                S[10] = -2*sg[5]

                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_y = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_y = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_y = gtab[5, j]
                    gAw_yy = gtab[6, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_x = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_x = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_x = ftab[5, i]
                        fAw_xx = ftab[6, i]
                        col = col0 + DOF*(j*m + i)
                        pA[0] = fAu_x*gAu
                        pA[1] = fAu*gAu_y
                        pA[2] = fAv_x*gAv
                        pA[3] = fAv*gAv_y
                        pA[4] = fAw_x*gAw
                        pA[5] = fAw*gAw_y
                        pA[6] = fAv*gAv
                        pA[7] = fAw*gAw
                        pA[8] = fAw_xx*gAw
                        pA[9] = fAw*gAw_yy
                        pA[10] = fAw_x*gAw_y
                        fint[col+0] += weight*(pA[0]*S[0] + pA[1]*S[1])
                        fint[col+1] += weight*(pA[2]*S[2] + pA[3]*S[3] + pA[6]*S[6])
                        fint[col+2] += weight*(pA[4]*S[4] + pA[5]*S[5] + pA[7]*S[7] + pA[8]*S[8] + pA[9]*S[9] + pA[10]*S[10])

    return fint


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


# BEGIN tangent stiffness product, generated by theory/shells/tangent_product/write_pyx.py


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
    At each integration point, with the gradients `\boldsymbol{G}
    \boldsymbol{c}` of the non-linear terms of the strains, the strains
    `\boldsymbol{\varepsilon} = (\boldsymbol{B}_L + \frac{1}{2}
    \boldsymbol{B}_{NL}) \boldsymbol{c}` and the membrane stress resultants
    `\boldsymbol{N}` of `\boldsymbol{F} \boldsymbol{\varepsilon}`, plus the
    constant ones:

    .. math::

        \boldsymbol{K}_T \boldsymbol{v} = \int \left[(\boldsymbol{B}_L +
        \boldsymbol{B}_{NL})^\top \boldsymbol{F} (\boldsymbol{B}_L +
        \boldsymbol{B}_{NL}) \boldsymbol{v} + \boldsymbol{G}^\top
        \boldsymbol{N} \boldsymbol{G} \boldsymbol{v}\right] dA

    evaluated without forming `\boldsymbol{K}_T`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \le x \le x_2`, `y_1 \le y \le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty, r
    cdef int m, n
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr

    cdef int i, j, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double b0, b1, b2, b3, b4, b5, db0, db1, db2, db3, db4, db5, Nxx, Nyy, Nxy
    cdef double e[6]
    cdef double de[6]
    cdef double s[6]
    cdef double ds[6]
    cdef double fAu, fAu_xi, fAv, fAv_xi, fAw, fAw_xi
    cdef double fAw_xixi
    cdef double gAu, gAu_eta, gAv, gAv_eta, gAw, gAw_eta
    cdef double gAw_etaeta

    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta, kTv

    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one
    # constitutive matrix [6, 6] for each of the nx * ny integration points
    cdef double F[36]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    if cs.shape[0] != size or vs.shape[0] != size:
        raise ValueError('cs and vs must have the size %d' % size)
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
    r = shell.r
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kTv = np.zeros(size, dtype=DOUBLE)
    # basis functions at the integration point
    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)

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

                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = fpp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = fpp(j, eta, y1w, y1wr, y2w, y2wr)

                # rotations of cs and vs
                b0 = 0.
                b1 = 0.
                b2 = 0.
                b3 = 0.
                b4 = 0.
                b5 = 0.
                db0 = 0.
                db1 = 0.
                db2 = 0.
                db3 = 0.
                db4 = 0.
                db5 = 0.
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
                        b0 += 2*fAu_xi*gAu/a*cs[col+0]
                        b1 += 2*fAu*gAu_eta/b*cs[col+0]
                        b2 += 2*fAv_xi*gAv/a*cs[col+1]
                        b3 += 2*fAv*gAv_eta/b*cs[col+1] + fAw*gAw/r*cs[col+2]
                        b4 += 2*fAw_xi*gAw/a*cs[col+2]
                        b5 += -fAv*gAv/r*cs[col+1] + 2*fAw*gAw_eta/b*cs[col+2]
                        db0 += 2*fAu_xi*gAu/a*vs[col+0]
                        db1 += 2*fAu*gAu_eta/b*vs[col+0]
                        db2 += 2*fAv_xi*gAv/a*vs[col+1]
                        db3 += 2*fAv*gAv_eta/b*vs[col+1] + fAw*gAw/r*vs[col+2]
                        db4 += 2*fAw_xi*gAw/a*vs[col+2]
                        db5 += -fAv*gAv/r*vs[col+1] + 2*fAw*gAw_eta/b*vs[col+2]

                # strains of cs, eps = ([B_L] + [B_NL]/2) cs, and strain
                # increment, deps = ([B_L] + [B_NL]) vs
                for i in range(6):
                    e[i] = 0.
                    de[i] = 0.
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_eta = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_eta = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_eta = gtab[5, j]
                    gAw_etaeta = gtab[6, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_xi = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        fAw_xixi = ftab[6, i]
                        col = col0 + DOF*(j*m + i)
                        e[0] += (b0*fAu_xi*gAu/a + 2*fAu_xi*gAu/a)*cs[col+0] + b2*fAv_xi*gAv/a*cs[col+1] + b4*fAw_xi*gAw/a*cs[col+2]
                        e[1] += b1*fAu*gAu_eta/b*cs[col+0] + (-0.5*b5*fAv*gAv/r + b3*fAv*gAv_eta/b + 2*fAv*gAv_eta/b)*cs[col+1] + (0.5*b3*fAw*gAw/r + fAw*gAw/r + b5*fAw*gAw_eta/b)*cs[col+2]
                        e[2] += (b0*fAu*gAu_eta/b + 2*fAu*gAu_eta/b + b1*fAu_xi*gAu/a)*cs[col+0] + (-0.5*b4*fAv*gAv/r + b2*fAv*gAv_eta/b + b3*fAv_xi*gAv/a + 2*fAv_xi*gAv/a)*cs[col+1] + (0.5*b2*fAw*gAw/r + b4*fAw*gAw_eta/b + b5*fAw_xi*gAw/a)*cs[col+2]
                        e[3] += -4*fAw_xixi*gAw/(a*a)*cs[col+2]
                        e[4] += 2*fAv*gAv_eta/(b*r)*cs[col+1] - 4*fAw*gAw_etaeta/(b*b)*cs[col+2]
                        e[5] += -fAu*gAu_eta/(b*r)*cs[col+0] + 3*fAv_xi*gAv/(a*r)*cs[col+1] - 8*fAw_xi*gAw_eta/(a*b)*cs[col+2]
                        de[0] += (2*b0*fAu_xi*gAu/a + 2*fAu_xi*gAu/a)*vs[col+0] + 2*b2*fAv_xi*gAv/a*vs[col+1] + 2*b4*fAw_xi*gAw/a*vs[col+2]
                        de[1] += 2*b1*fAu*gAu_eta/b*vs[col+0] + (-b5*fAv*gAv/r + 2*b3*fAv*gAv_eta/b + 2*fAv*gAv_eta/b)*vs[col+1] + (b3*fAw*gAw/r + fAw*gAw/r + 2*b5*fAw*gAw_eta/b)*vs[col+2]
                        de[2] += (2*b0*fAu*gAu_eta/b + 2*fAu*gAu_eta/b + 2*b1*fAu_xi*gAu/a)*vs[col+0] + (-b4*fAv*gAv/r + 2*b2*fAv*gAv_eta/b + 2*b3*fAv_xi*gAv/a + 2*fAv_xi*gAv/a)*vs[col+1] + (b2*fAw*gAw/r + 2*b4*fAw*gAw_eta/b + 2*b5*fAw_xi*gAw/a)*vs[col+2]
                        de[3] += -4*fAw_xixi*gAw/(a*a)*vs[col+2]
                        de[4] += 2*fAv*gAv_eta/(b*r)*vs[col+1] - 4*fAw*gAw_etaeta/(b*b)*vs[col+2]
                        de[5] += -fAu*gAu_eta/(b*r)*vs[col+0] + 3*fAv_xi*gAv/(a*r)*vs[col+1] - 8*fAw_xi*gAw_eta/(a*b)*vs[col+2]

                # stresses
                for i in range(6):
                    s[i] = 0.
                    ds[i] = 0.
                    for j in range(6):
                        s[i] += F[i*6 + j]*e[j]
                        ds[i] += F[i*6 + j]*de[j]
                Nxx = s[0] + Nxx_cte
                Nyy = s[1] + Nyy_cte
                Nxy = s[2] + Nxy_cte

                # ([B_L] + [B_NL])^T ds + [G]^T [N] [G] vs
                for j in range(n):
                    gAu = gtab[0, j]
                    gAu_eta = gtab[1, j]
                    gAv = gtab[2, j]
                    gAv_eta = gtab[3, j]
                    gAw = gtab[4, j]
                    gAw_eta = gtab[5, j]
                    gAw_etaeta = gtab[6, j]
                    for i in range(m):
                        fAu = ftab[0, i]
                        fAu_xi = ftab[1, i]
                        fAv = ftab[2, i]
                        fAv_xi = ftab[3, i]
                        fAw = ftab[4, i]
                        fAw_xi = ftab[5, i]
                        fAw_xixi = ftab[6, i]
                        col = col0 + DOF*(j*m + i)
                        kTv[col+0] += weight*(2*Nxx*db0*fAu_xi*gAu/a + Nxy*(2*db0*fAu*gAu_eta/b + 2*db1*fAu_xi*gAu/a) + 2*Nyy*db1*fAu*gAu_eta/b + ds[0]*(2*b0*fAu_xi*gAu/a + 2*fAu_xi*gAu/a) + ds[2]*(2*b0*fAu*gAu_eta/b + 2*fAu*gAu_eta/b + 2*b1*fAu_xi*gAu/a) + 2*b1*ds[1]*fAu*gAu_eta/b - ds[5]*fAu*gAu_eta/(b*r))
                        kTv[col+1] += weight*(2*Nxx*db2*fAv_xi*gAv/a + Nxy*(-db4*fAv*gAv/r + 2*db2*fAv*gAv_eta/b + 2*db3*fAv_xi*gAv/a) + Nyy*(-db5*fAv*gAv/r + 2*db3*fAv*gAv_eta/b) + ds[1]*(-b5*fAv*gAv/r + 2*b3*fAv*gAv_eta/b + 2*fAv*gAv_eta/b) + ds[2]*(-b4*fAv*gAv/r + 2*b2*fAv*gAv_eta/b + 2*b3*fAv_xi*gAv/a + 2*fAv_xi*gAv/a) + 2*ds[4]*fAv*gAv_eta/(b*r) + 2*b2*ds[0]*fAv_xi*gAv/a + 3*ds[5]*fAv_xi*gAv/(a*r))
                        kTv[col+2] += weight*(2*Nxx*db4*fAw_xi*gAw/a + Nxy*(db2*fAw*gAw/r + 2*db4*fAw*gAw_eta/b + 2*db5*fAw_xi*gAw/a) + Nyy*(db3*fAw*gAw/r + 2*db5*fAw*gAw_eta/b) + ds[1]*(b3*fAw*gAw/r + fAw*gAw/r + 2*b5*fAw*gAw_eta/b) + ds[2]*(b2*fAw*gAw/r + 2*b4*fAw*gAw_eta/b + 2*b5*fAw_xi*gAw/a) - 4*ds[4]*fAw*gAw_etaeta/(b*b) + 2*b4*ds[0]*fAw_xi*gAw/a - 8*ds[5]*fAw_xi*gAw_eta/(a*b) - 4*ds[3]*fAw_xixi*gAw/(a*a))

    return np.asarray(kTv)

# END tangent stiffness product
