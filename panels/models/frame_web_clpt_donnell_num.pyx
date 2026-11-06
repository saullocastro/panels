#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Web of a ring frame, an annular-sector flat plate, using the classical
laminated plate theory (CLPT) with von Kármán strains in polar coordinates

Numerically integrated kernels of the model ``'frame_web_clpt_donnell'``:
the constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the mass :func:`fkM_num`, the internal force vector
:func:`calc_fint`, and the fields :func:`fuvw`, :func:`fstrain` and
:func:`fg`, the last one that of :mod:`panels.models.clpt_field`. The
degree of freedom ``k`` of the term ``(i, j)``, ``0 <= i < m``, ``0 <= j <
n``, is at the row ``row0 + 3*(j*m + i) + k`` of the matrices.

The web lies in the plane normal to the axis of a cylinder of radius ``r``,
``shell.r``, the outer radius of the web, between the radii ``r`` and ``r -
a``: ``a``, ``shell.a``, is the height of the web and ``b``, ``shell.b``,
the arc length of its outer edge. The coordinate `x = r - \rho` runs from the
outer edge inwards, `y = r \theta` along the outer edge, with the polar
coordinates `(\rho, \theta)` of the plane; ``u``, ``v``, ``w`` are the
displacements along `\boldsymbol{e}_x = -\boldsymbol{e}_\rho`,
`\boldsymbol{e}_y = \boldsymbol{e}_\theta` and `\boldsymbol{e}_z =
\boldsymbol{e}_x \times \boldsymbol{e}_y`, the local axes of the flange of
the connection ``'BFxcte'`` whose base is the cylinder.

Notes
-----
With `s = r/\rho`, the von Kármán strains in polar coordinates of Amabili
[1]_, Eqs. (1.175)-(1.180), are, in these coordinates:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} w_{,x}^2 \qquad
    \varepsilon_{yy} = s v_{,y} - \frac{u}{\rho} + \frac{1}{2} s^2 w_{,y}^2

    \gamma_{xy} = s u_{,y} + v_{,x} + \frac{v}{\rho} + s w_{,x} w_{,y}

    \kappa_{xx} = -w_{,xx} \qquad
    \kappa_{yy} = \frac{w_{,x}}{\rho} - s^2 w_{,yy} \qquad
    \kappa_{xy} = -2 s \left(w_{,xy} + \frac{w_{,y}}{\rho}\right)

and the element of area is `dA = (\rho/r)\, dx\, dy`. The integrands are
rational functions of `x`, integrated with the Gauss-Legendre rule over the
domain of :meth:`.Shell.integration_limits`.

References
----------
.. [1] Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
   Cambridge University Press, 2008, Sec. 1.4.
   https://doi.org/10.1017/CBO9780511619694

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
from .clpt_field import fg


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 3
cdef int[3] P0 = [0, 3, 6]
cdef int[3] P1 = [3, 6, 12]


def fkC_num(double [::1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    r"""Constitutive stiffness matrix, integrated numerically

    With ``NLgeom=0``, the linear constitutive stiffness matrix ``KC0``;
    with ``NLgeom=1``, the matrix ``KC0 + KCNL + KGNL`` at the configuration
    ``cs``, which, added to :func:`fkG_num` at the same ``cs``, gives the
    tangent stiffness matrix.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements.
    Finput : array-like
        Constitutive matrix ``ABD`` of the classical laminated plate theory,
        shape ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 6, 6)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The web, which gives the height ``a``, the arc length ``b`` of the
        outer edge, the outer radius ``r``, the numbers of terms ``m`` and
        ``n`` and the boundary flags of the approximation functions of each
        field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this web start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.
    NLgeom : int, optional
        ``1`` to include the non-linear terms of ``cs``, default ``0``.

    Returns
    -------
    kC : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper triangle is
        assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape, ``shell`` is not a
        :class:`.Shell` or its outer radius ``r`` is not larger than its
        height ``a``.

    Notes
    -----
    With the generalized derivatives `\boldsymbol{d} = \boldsymbol{\Phi}
    \boldsymbol{c}` of the module, `\boldsymbol{B} = \partial
    \boldsymbol{\varepsilon}/\partial \boldsymbol{d}` and the membrane
    stress resultants `\boldsymbol{N}_{NL}` of the quadratic part of the
    strains:

    .. math::

        \boldsymbol{K}_C = \int \boldsymbol{\Phi}^\top \left(
        \boldsymbol{B}^\top \boldsymbol{F} \boldsymbol{B} +
        \boldsymbol{H}(\boldsymbol{N}_{NL}) \right) \boldsymbol{\Phi} \, dA

    with `\boldsymbol{H}(\boldsymbol{N}) = \boldsymbol{G}^\top
    \boldsymbol{N} \boldsymbol{G}`, `\boldsymbol{G} \boldsymbol{d} =
    \{w_{,x}, s w_{,y}\}^\top`, `dA = (\rho/r)\, dx\, dy`, see the notes of
    the module.

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, intx, inty, xi, eta, xp, rho, s, irho, weight
    cdef int m, n, i, j, k, l, c, p, q, r, e, f2, row, col, ptx, pty
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double pA[12]
    cdef double pB[12]
    cdef double dv[12]
    cdef double K[144]
    cdef double TA[36]
    cdef double Bm[72]
    cdef double FB[72]
    cdef double w_x, w_y, eps0, eps1, eps2, NxxNL, NyyNL, NxyNL

    cdef long [::1] kr, kc
    cdef double [::1] kv
    cdef double F[36]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')
    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    R = shell.r
    m = shell.m
    n = shell.n
    if not R > a:
        raise ValueError('the outer radius r must be larger than the height a '
                         'of the web, got r={0} and a={1}'.format(R, a))
    #NOTE physical limits of the integration domain, resolved and validated
    #     by Shell.integration_limits()
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)
    xi1 = x1/a*2 - 1
    xi2 = x2/a*2 - 1
    eta1 = y1/b*2 - 1
    eta2 = y2/b*2 - 1
    intx = x2 - x1
    inty = y2 - y1
    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)

    fdim = 9*m*m*n*n
    kr = np.zeros((fdim,), dtype=INT)
    kc = np.zeros((fdim,), dtype=INT)
    kv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = (xis[ptx] + 1)/2*(xi2 - xi1) + xi1
                eta = (etas[pty] + 1)/2*(eta2 - eta1) + eta1
                xp = (xi + 1)*a/2
                rho = R - xp
                s = R/rho
                irho = 1./rho
                # per unit area of the natural domain, times dA/(dx dy)
                weight = weights_xi[ptx]*weights_eta[pty]*intx*inty/4*rho/R
                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]
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

                for p in range(72):
                    Bm[p] = 0.
                w_x = 0.
                w_y = 0.
                if NLgeom == 1:
                    for p in range(12):
                        dv[p] = 0.
                    for j in range(n):
                        for i in range(m):
                            col = col0 + 3*(j*m + i)
                            pA[0] = ftab[0, i]*gtab[0, j]
                            pA[1] = ftab[1, i]*gtab[0, j]
                            pA[2] = ftab[0, i]*gtab[1, j]
                            pA[3] = ftab[2, i]*gtab[2, j]
                            pA[4] = ftab[3, i]*gtab[2, j]
                            pA[5] = ftab[2, i]*gtab[3, j]
                            pA[6] = ftab[4, i]*gtab[4, j]
                            pA[7] = ftab[5, i]*gtab[4, j]
                            pA[8] = ftab[4, i]*gtab[5, j]
                            pA[9] = ftab[6, i]*gtab[4, j]
                            pA[10] = ftab[4, i]*gtab[6, j]
                            pA[11] = ftab[5, i]*gtab[5, j]
                            for p in range(12):
                                dv[p] += pA[p]*cs[col + (0 if p < 3 else (1 if p < 6 else 2))]
                    w_x = dv[7]
                    w_y = dv[8]
                Bm[1] = 1.
                Bm[7] = w_x
                Bm[12] = -irho
                Bm[17] = s
                Bm[20] = (s)*(s)*w_y
                Bm[26] = s
                Bm[27] = irho
                Bm[28] = 1.
                Bm[31] = s*w_y
                Bm[32] = s*w_x
                Bm[45] = -1.
                Bm[55] = irho
                Bm[58] = -(s)*(s)
                Bm[68] = -2.*irho*s
                Bm[71] = -2.*s
                # K = [B]^T [ABD] [B] + [H(N_NL)]
                for p in range(6):
                    for q in range(12):
                        FB[p*12 + q] = 0.
                        for r in range(6):
                            FB[p*12 + q] += F[p*6 + r]*Bm[r*12 + q]
                for p in range(12):
                    for q in range(12):
                        K[p*12 + q] = 0.
                        for r in range(6):
                            K[p*12 + q] += Bm[r*12 + p]*FB[r*12 + q]
                eps0 = 0.5*w_x*w_x
                eps1 = 0.5*s*s*w_y*w_y
                eps2 = s*w_x*w_y
                NxxNL = F[0]*eps0 + F[1]*eps1 + F[2]*eps2
                NyyNL = F[6]*eps0 + F[7]*eps1 + F[8]*eps2
                NxyNL = F[12]*eps0 + F[13]*eps1 + F[14]*eps2
                K[91] += NxxNL
                K[92] += NxyNL*s
                K[103] += NxyNL*s
                K[104] += NyyNL*(s)*(s)
                c = -1
                for j in range(n):
                    for i in range(m):
                        row = row0 + 3*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[2, j]
                        pA[4] = ftab[3, i]*gtab[2, j]
                        pA[5] = ftab[2, i]*gtab[3, j]
                        pA[6] = ftab[4, i]*gtab[4, j]
                        pA[7] = ftab[5, i]*gtab[4, j]
                        pA[8] = ftab[4, i]*gtab[5, j]
                        pA[9] = ftab[6, i]*gtab[4, j]
                        pA[10] = ftab[4, i]*gtab[6, j]
                        pA[11] = ftab[5, i]*gtab[5, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(3):
                            for q in range(12):
                                TA[e*12 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*12 + q] += pA[p]*K[p*12 + q]
                        for l in range(n):
                            for k in range(m):
                                col = col0 + 3*(l*m + k)
                                #NOTE upper triangle, the caller symmetrizes
                                if row > col:
                                    continue
                                pB[0] = ftab[0, k]*gtab[0, l]
                                pB[1] = ftab[1, k]*gtab[0, l]
                                pB[2] = ftab[0, k]*gtab[1, l]
                                pB[3] = ftab[2, k]*gtab[2, l]
                                pB[4] = ftab[3, k]*gtab[2, l]
                                pB[5] = ftab[2, k]*gtab[3, l]
                                pB[6] = ftab[4, k]*gtab[4, l]
                                pB[7] = ftab[5, k]*gtab[4, l]
                                pB[8] = ftab[4, k]*gtab[5, l]
                                pB[9] = ftab[6, k]*gtab[4, l]
                                pB[10] = ftab[4, k]*gtab[6, l]
                                pB[11] = ftab[5, k]*gtab[5, l]
                                for e in range(3):
                                    for f2 in range(3):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*12 + q]*pB[q]

    return coo_matrix((kv, (kr, kc)), shape=(size, size))

def fkG_num(double [::1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny,
        double Nxx0=0, double Nyy0=0, double Nxy0=0):
    r"""Geometric stiffness matrix of the linear membrane stress state

    Integrates numerically the geometric stiffness matrix of the membrane
    stress resultants of the linear strains at ``cs``, superposed to the
    constant state ``Nxx0``, ``Nyy0``, ``Nxy0``. The matrix is homogeneous
    of degree one in ``cs``, as the linear buckling eigenvalue problem
    requires; the stress of the non-linear strains enters :func:`fkC_num`.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements.
    Finput : array-like
        Constitutive matrix ``ABD`` of the classical laminated plate theory,
        shape ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 6, 6)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The web, which gives the height ``a``, the arc length ``b`` of the
        outer edge, the outer radius ``r``, the numbers of terms ``m`` and
        ``n`` and the boundary flags of the approximation functions of each
        field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this web start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.
    Nxx0, Nyy0, Nxy0 : float, optional
        Constant membrane stress resultants along the local axes, `N_{xx}
        = N_{\rho\rho}`, `N_{yy} = N_{\theta\theta}` and `N_{xy} =
        -N_{\rho\theta}`, force per unit length, default ``0``.

    Returns
    -------
    kG : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper triangle is
        assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape, ``shell`` is not a
        :class:`.Shell` or its outer radius ``r`` is not larger than its
        height ``a``.

    Notes
    -----
    With `\boldsymbol{N} = \boldsymbol{N}_0 + \boldsymbol{N}_L`,
    `\boldsymbol{N}_L` the membrane stress resultants of the linear strains
    of ``cs``:

    .. math::

        \boldsymbol{K}_G = \int \boldsymbol{\Phi}^\top \boldsymbol{G}^\top
        \begin{bmatrix} N_{xx} & N_{xy} \\ N_{xy} & N_{yy}
        \end{bmatrix} \boldsymbol{G} \boldsymbol{\Phi} \, dA, \qquad
        \boldsymbol{G} \boldsymbol{d} = \{w_{,x}, s w_{,y}\}^\top

    the variation of the work of the membrane stresses on the quadratic
    terms of the von Kármán strains, see the notes of the module.

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, intx, inty, xi, eta, xp, rho, s, irho, weight
    cdef int m, n, i, j, k, l, c, p, q, r, e, f2, row, col, ptx, pty
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double pA[12]
    cdef double pB[12]
    cdef double dv[12]
    cdef double K[144]
    cdef double TA[36]
    cdef double eps0, eps1, eps2, eps3, eps4, eps5, Nxx, Nyy, Nxy

    cdef long [::1] kr, kc
    cdef double [::1] kv
    cdef double F[36]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')
    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    R = shell.r
    m = shell.m
    n = shell.n
    if not R > a:
        raise ValueError('the outer radius r must be larger than the height a '
                         'of the web, got r={0} and a={1}'.format(R, a))
    #NOTE physical limits of the integration domain, resolved and validated
    #     by Shell.integration_limits()
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)
    xi1 = x1/a*2 - 1
    xi2 = x2/a*2 - 1
    eta1 = y1/b*2 - 1
    eta2 = y2/b*2 - 1
    intx = x2 - x1
    inty = y2 - y1
    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)

    fdim = 9*m*m*n*n
    kr = np.zeros((fdim,), dtype=INT)
    kc = np.zeros((fdim,), dtype=INT)
    kv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = (xis[ptx] + 1)/2*(xi2 - xi1) + xi1
                eta = (etas[pty] + 1)/2*(eta2 - eta1) + eta1
                xp = (xi + 1)*a/2
                rho = R - xp
                s = R/rho
                irho = 1./rho
                # per unit area of the natural domain, times dA/(dx dy)
                weight = weights_xi[ptx]*weights_eta[pty]*intx*inty/4*rho/R
                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]
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

                for p in range(12):
                    dv[p] = 0.
                for j in range(n):
                    for i in range(m):
                        col = col0 + 3*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[2, j]
                        pA[4] = ftab[3, i]*gtab[2, j]
                        pA[5] = ftab[2, i]*gtab[3, j]
                        pA[6] = ftab[4, i]*gtab[4, j]
                        pA[7] = ftab[5, i]*gtab[4, j]
                        pA[8] = ftab[4, i]*gtab[5, j]
                        pA[9] = ftab[6, i]*gtab[4, j]
                        pA[10] = ftab[4, i]*gtab[6, j]
                        pA[11] = ftab[5, i]*gtab[5, j]
                        for p in range(12):
                            dv[p] += pA[p]*cs[col + (0 if p < 3 else (1 if p < 6 else 2))]
                eps0 = dv[1]
                eps1 = -dv[0]*irho + dv[5]*s
                eps2 = dv[2]*s + dv[3]*irho + dv[4]
                eps3 = -dv[9]
                eps4 = -dv[10]*(s)*(s) + dv[7]*irho
                eps5 = -2.*dv[11]*s - 2.*dv[8]*irho*s
                Nxx = F[0]*eps0 + F[1]*eps1 + F[2]*eps2 + F[3]*eps3 + F[4]*eps4 + F[5]*eps5
                Nyy = F[6]*eps0 + F[7]*eps1 + F[8]*eps2 + F[9]*eps3 + F[10]*eps4 + F[11]*eps5
                Nxy = F[12]*eps0 + F[13]*eps1 + F[14]*eps2 + F[15]*eps3 + F[16]*eps4 + F[17]*eps5
                Nxx = Nxx + Nxx0
                Nyy = Nyy + Nyy0
                Nxy = Nxy + Nxy0
                for p in range(144):
                    K[p] = 0.
                K[91] += Nxx
                K[92] += Nxy*s
                K[103] += Nxy*s
                K[104] += Nyy*(s)*(s)
                c = -1
                for j in range(n):
                    for i in range(m):
                        row = row0 + 3*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[2, j]
                        pA[4] = ftab[3, i]*gtab[2, j]
                        pA[5] = ftab[2, i]*gtab[3, j]
                        pA[6] = ftab[4, i]*gtab[4, j]
                        pA[7] = ftab[5, i]*gtab[4, j]
                        pA[8] = ftab[4, i]*gtab[5, j]
                        pA[9] = ftab[6, i]*gtab[4, j]
                        pA[10] = ftab[4, i]*gtab[6, j]
                        pA[11] = ftab[5, i]*gtab[5, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(3):
                            for q in range(12):
                                TA[e*12 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*12 + q] += pA[p]*K[p*12 + q]
                        for l in range(n):
                            for k in range(m):
                                col = col0 + 3*(l*m + k)
                                #NOTE upper triangle, the caller symmetrizes
                                if row > col:
                                    continue
                                pB[0] = ftab[0, k]*gtab[0, l]
                                pB[1] = ftab[1, k]*gtab[0, l]
                                pB[2] = ftab[0, k]*gtab[1, l]
                                pB[3] = ftab[2, k]*gtab[2, l]
                                pB[4] = ftab[3, k]*gtab[2, l]
                                pB[5] = ftab[2, k]*gtab[3, l]
                                pB[6] = ftab[4, k]*gtab[4, l]
                                pB[7] = ftab[5, k]*gtab[4, l]
                                pB[8] = ftab[4, k]*gtab[5, l]
                                pB[9] = ftab[6, k]*gtab[4, l]
                                pB[10] = ftab[4, k]*gtab[6, l]
                                pB[11] = ftab[5, k]*gtab[5, l]
                                for e in range(3):
                                    for f2 in range(3):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*12 + q]*pB[q]

    return coo_matrix((kv, (kr, kc)), shape=(size, size))

def fkM_num(object shell, object I_input, int size,
        int row0, int col0, int nx, int ny):
    r"""Mass matrix, integrated numerically

    Parameters
    ----------
    shell : :class:`.Shell`
        The web, which gives the height ``a``, the arc length ``b`` of the
        outer edge, the outer radius ``r``, the numbers of terms ``m`` and
        ``n`` and the boundary flags of the approximation functions of each
        field.
    I_input : array-like
        Moments of the density through the thickness, ``I0, I1, I2``, shape
        ``(3,)`` for the whole domain or ``(nx, ny, 3)`` with one set per
        integration point.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this web start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    kM : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper triangle is
        assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``I_input`` has an invalid shape, ``shell`` is not a
        :class:`.Shell` or its outer radius ``r`` is not larger than its
        height ``a``.

    Notes
    -----
    With the displacement through the thickness of the Kirchhoff-Love
    kinematics, `u(z) = u - z w_{,x}`, `v(z) = v - z s w_{,y}`, `w(z) =
    w`, and the moments `I_n = \int \rho_m z^n dz`, `n = 0, 1, 2`, of the
    density `\rho_m`:

    .. math::

        \boldsymbol{M} = \int \boldsymbol{\Phi}^\top \boldsymbol{J}^\top
        \begin{bmatrix} I_0 \boldsymbol{I}_3 & I_1 \boldsymbol{E} \\
        I_1 \boldsymbol{E}^\top & I_2 \boldsymbol{I}_2 \end{bmatrix}
        \boldsymbol{J} \boldsymbol{\Phi} \, dA, \qquad
        \boldsymbol{J} \boldsymbol{d} = \{u, v, w, -w_{,x}, -s
        w_{,y}\}^\top

    with `\boldsymbol{E}` the first two columns of `\boldsymbol{I}_3` and
    `dA = (\rho/r)\, dx\, dy`.

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, intx, inty, xi, eta, xp, rho, s, irho, weight
    cdef int m, n, i, j, k, l, c, p, q, r, e, f2, row, col, ptx, pty
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double pA[12]
    cdef double pB[12]
    cdef double dv[12]
    cdef double K[144]
    cdef double TA[36]
    cdef double I0, I1, I2
    cdef double [:, :, ::1] Inxny
    cdef int one_I_each_point = 0
    I_input = np.ascontiguousarray(I_input, dtype=DOUBLE)
    if I_input.shape == (nx, ny, 3):
        Inxny = I_input
        one_I_each_point = 1
    elif I_input.shape == (3,):
        Inxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        I0 = I_input[0]
        I1 = I_input[1]
        I2 = I_input[2]
    else:
        raise ValueError('Invalid shape for I_input!')

    cdef long [::1] kr, kc
    cdef double [::1] kv
    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    R = shell.r
    m = shell.m
    n = shell.n
    if not R > a:
        raise ValueError('the outer radius r must be larger than the height a '
                         'of the web, got r={0} and a={1}'.format(R, a))
    #NOTE physical limits of the integration domain, resolved and validated
    #     by Shell.integration_limits()
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)
    xi1 = x1/a*2 - 1
    xi2 = x2/a*2 - 1
    eta1 = y1/b*2 - 1
    eta2 = y2/b*2 - 1
    intx = x2 - x1
    inty = y2 - y1
    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)

    fdim = 9*m*m*n*n
    kr = np.zeros((fdim,), dtype=INT)
    kc = np.zeros((fdim,), dtype=INT)
    kv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = (xis[ptx] + 1)/2*(xi2 - xi1) + xi1
                eta = (etas[pty] + 1)/2*(eta2 - eta1) + eta1
                xp = (xi + 1)*a/2
                rho = R - xp
                s = R/rho
                irho = 1./rho
                # per unit area of the natural domain, times dA/(dx dy)
                weight = weights_xi[ptx]*weights_eta[pty]*intx*inty/4*rho/R
                if one_I_each_point == 1:
                    I0 = Inxny[ptx, pty, 0]
                    I1 = Inxny[ptx, pty, 1]
                    I2 = Inxny[ptx, pty, 2]
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

                for p in range(144):
                    K[p] = 0.
                K[0] = I0
                K[7] = -I1
                K[39] = I0
                K[44] = -I1*s
                K[78] = I0
                K[84] = -I1
                K[91] = I2
                K[99] = -I1*s
                K[104] = I2*(s)*(s)
                c = -1
                for j in range(n):
                    for i in range(m):
                        row = row0 + 3*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[2, j]
                        pA[4] = ftab[3, i]*gtab[2, j]
                        pA[5] = ftab[2, i]*gtab[3, j]
                        pA[6] = ftab[4, i]*gtab[4, j]
                        pA[7] = ftab[5, i]*gtab[4, j]
                        pA[8] = ftab[4, i]*gtab[5, j]
                        pA[9] = ftab[6, i]*gtab[4, j]
                        pA[10] = ftab[4, i]*gtab[6, j]
                        pA[11] = ftab[5, i]*gtab[5, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(3):
                            for q in range(12):
                                TA[e*12 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*12 + q] += pA[p]*K[p*12 + q]
                        for l in range(n):
                            for k in range(m):
                                col = col0 + 3*(l*m + k)
                                #NOTE upper triangle, the caller symmetrizes
                                if row > col:
                                    continue
                                pB[0] = ftab[0, k]*gtab[0, l]
                                pB[1] = ftab[1, k]*gtab[0, l]
                                pB[2] = ftab[0, k]*gtab[1, l]
                                pB[3] = ftab[2, k]*gtab[2, l]
                                pB[4] = ftab[3, k]*gtab[2, l]
                                pB[5] = ftab[2, k]*gtab[3, l]
                                pB[6] = ftab[4, k]*gtab[4, l]
                                pB[7] = ftab[5, k]*gtab[4, l]
                                pB[8] = ftab[4, k]*gtab[5, l]
                                pB[9] = ftab[6, k]*gtab[4, l]
                                pB[10] = ftab[4, k]*gtab[6, l]
                                pB[11] = ftab[5, k]*gtab[5, l]
                                for e in range(3):
                                    for f2 in range(3):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*12 + q]*pB[q]

    return coo_matrix((kv, (kr, kc)), shape=(size, size))

def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    r"""Internal force vector, integrated numerically

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements.
    Finput : array-like
        Constitutive matrix ``ABD`` of the classical laminated plate theory,
        shape ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 6, 6)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The web, which gives the height ``a``, the arc length ``b`` of the
        outer edge, the outer radius ``r``, the numbers of terms ``m`` and
        ``n`` and the boundary flags of the approximation functions of each
        field.
    size : int
        Size of the vector, the number of degrees of freedom of the
        assembly.
    col0 : int
        Entry of the vector where the terms of this web start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    fint : np.ndarray
        Internal force vector, 1-D of size ``size``.

    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape, ``shell`` is not a
        :class:`.Shell` or its outer radius ``r`` is not larger than its
        height ``a``.

    Notes
    -----
    With the von Kármán strains `\boldsymbol{\varepsilon}(\boldsymbol{d})`
    of the module, `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{B} = \partial
    \boldsymbol{\varepsilon}/\partial \boldsymbol{d}`:

    .. math::

        \boldsymbol{f}_{int} = \int \boldsymbol{\Phi}^\top
        \boldsymbol{B}^\top \boldsymbol{\sigma} \, dA

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, intx, inty, xi, eta, xp, rho, s, irho, weight
    cdef int m, n, i, j, k, l, c, p, q, r, e, f2, row, col, ptx, pty
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double pA[12]
    cdef double pB[12]
    cdef double dv[12]
    cdef double K[144]
    cdef double TA[36]
    cdef double Bm[72]
    cdef double bs[12]
    cdef double w_x, w_y, eps0, eps1, eps2, eps3, eps4, eps5
    cdef double Nxx, Nyy, Nxy, Mxx, Myy, Mxy
    cdef double [::1] fint
    cdef int row0 = col0
    cdef double F[36]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')
    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    R = shell.r
    m = shell.m
    n = shell.n
    if not R > a:
        raise ValueError('the outer radius r must be larger than the height a '
                         'of the web, got r={0} and a={1}'.format(R, a))
    #NOTE physical limits of the integration domain, resolved and validated
    #     by Shell.integration_limits()
    x1, x2, y1, y2 = shell.integration_limits()
    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr
    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)
    xi1 = x1/a*2 - 1
    xi2 = x2/a*2 - 1
    eta1 = y1/b*2 - 1
    eta2 = y2/b*2 - 1
    intx = x2 - x1
    inty = y2 - y1
    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)

    fint = np.zeros(size, dtype=DOUBLE)

    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = (xis[ptx] + 1)/2*(xi2 - xi1) + xi1
                eta = (etas[pty] + 1)/2*(eta2 - eta1) + eta1
                xp = (xi + 1)*a/2
                rho = R - xp
                s = R/rho
                irho = 1./rho
                # per unit area of the natural domain, times dA/(dx dy)
                weight = weights_xi[ptx]*weights_eta[pty]*intx*inty/4*rho/R
                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]
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

                for p in range(12):
                    dv[p] = 0.
                for j in range(n):
                    for i in range(m):
                        col = col0 + 3*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[2, j]
                        pA[4] = ftab[3, i]*gtab[2, j]
                        pA[5] = ftab[2, i]*gtab[3, j]
                        pA[6] = ftab[4, i]*gtab[4, j]
                        pA[7] = ftab[5, i]*gtab[4, j]
                        pA[8] = ftab[4, i]*gtab[5, j]
                        pA[9] = ftab[6, i]*gtab[4, j]
                        pA[10] = ftab[4, i]*gtab[6, j]
                        pA[11] = ftab[5, i]*gtab[5, j]
                        for p in range(12):
                            dv[p] += pA[p]*cs[col + (0 if p < 3 else (1 if p < 6 else 2))]
                w_x = dv[7]
                w_y = dv[8]
                eps0 = dv[1] + (dv[7])*(dv[7])/2.
                eps1 = -dv[0]*irho + dv[5]*s + (dv[8])*(dv[8])*(s)*(s)/2.
                eps2 = dv[2]*s + dv[3]*irho + dv[4] + dv[7]*dv[8]*s
                eps3 = -dv[9]
                eps4 = -dv[10]*(s)*(s) + dv[7]*irho
                eps5 = -2.*dv[11]*s - 2.*dv[8]*irho*s
                Nxx = F[0]*eps0 + F[1]*eps1 + F[2]*eps2 + F[3]*eps3 + F[4]*eps4 + F[5]*eps5
                Nyy = F[6]*eps0 + F[7]*eps1 + F[8]*eps2 + F[9]*eps3 + F[10]*eps4 + F[11]*eps5
                Nxy = F[12]*eps0 + F[13]*eps1 + F[14]*eps2 + F[15]*eps3 + F[16]*eps4 + F[17]*eps5
                Mxx = F[18]*eps0 + F[19]*eps1 + F[20]*eps2 + F[21]*eps3 + F[22]*eps4 + F[23]*eps5
                Myy = F[24]*eps0 + F[25]*eps1 + F[26]*eps2 + F[27]*eps3 + F[28]*eps4 + F[29]*eps5
                Mxy = F[30]*eps0 + F[31]*eps1 + F[32]*eps2 + F[33]*eps3 + F[34]*eps4 + F[35]*eps5
                for p in range(72):
                    Bm[p] = 0.
                Bm[1] = 1.
                Bm[7] = w_x
                Bm[12] = -irho
                Bm[17] = s
                Bm[20] = (s)*(s)*w_y
                Bm[26] = s
                Bm[27] = irho
                Bm[28] = 1.
                Bm[31] = s*w_y
                Bm[32] = s*w_x
                Bm[45] = -1.
                Bm[55] = irho
                Bm[58] = -(s)*(s)
                Bm[68] = -2.*irho*s
                Bm[71] = -2.*s
                for p in range(12):
                    bs[p] = (Bm[p]*Nxx + Bm[12 + p]*Nyy + Bm[24 + p]*Nxy
                             + Bm[36 + p]*Mxx + Bm[48 + p]*Myy
                             + Bm[60 + p]*Mxy)
                for j in range(n):
                    for i in range(m):
                        row = row0 + 3*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[2, j]
                        pA[4] = ftab[3, i]*gtab[2, j]
                        pA[5] = ftab[2, i]*gtab[3, j]
                        pA[6] = ftab[4, i]*gtab[4, j]
                        pA[7] = ftab[5, i]*gtab[4, j]
                        pA[8] = ftab[4, i]*gtab[5, j]
                        pA[9] = ftab[6, i]*gtab[4, j]
                        pA[10] = ftab[4, i]*gtab[6, j]
                        pA[11] = ftab[5, i]*gtab[5, j]
                        for e in range(3):
                            for p in range(P0[e], P1[e]):
                                fint[row + e] += weight*pA[p]*bs[p]
    return np.asarray(fint)

def _point_derivatives(double [::1] c, object shell, double xp, double yp):
    r"""Generalized derivatives `\boldsymbol{d}` of the web at a point

    Parameters
    ----------
    c : np.ndarray
        Ritz constants of the web, 1-D array of size ``3*m*n``.
    shell : :class:`.Shell`
        The web.
    xp, yp : float
        Coordinates `x` and `y` of the point, lengths.

    Returns
    -------
    d : np.ndarray
        ``(u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy)``, 1-D
        array of size 12.

    """
    cdef int m, n, i, j, col, p, col0 = 0
    cdef double a, b, xi, eta
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    cdef double pA[12]
    cdef double dv[12]
    cdef double [:, ::1] ftab, gtab
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
    xi = 2*xp/a - 1
    eta = 2*yp/b - 1
    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)
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
    for p in range(12):
        dv[p] = 0.
    for j in range(n):
        for i in range(m):
            col = col0 + 3*(j*m + i)
            pA[0] = ftab[0, i]*gtab[0, j]
            pA[1] = ftab[1, i]*gtab[0, j]
            pA[2] = ftab[0, i]*gtab[1, j]
            pA[3] = ftab[2, i]*gtab[2, j]
            pA[4] = ftab[3, i]*gtab[2, j]
            pA[5] = ftab[2, i]*gtab[3, j]
            pA[6] = ftab[4, i]*gtab[4, j]
            pA[7] = ftab[5, i]*gtab[4, j]
            pA[8] = ftab[4, i]*gtab[5, j]
            pA[9] = ftab[6, i]*gtab[4, j]
            pA[10] = ftab[4, i]*gtab[6, j]
            pA[11] = ftab[5, i]*gtab[5, j]
            for p in range(12):
                dv[p] += pA[p]*c[col + (0 if p < 3 else (1 if p < 6 else 2))]
    return np.array([dv[p] for p in range(12)])


def fuvw(double [::1] c, object s, double [::1] xs, double [::1] ys,
         int num_cores=4):
    r"""Displacements and rotations of the web at the points ``xs``, ``ys``

    Parameters
    ----------
    c : np.ndarray
        Ritz constants of the web ``s``, 1-D array of size ``3*s.m*s.n``,
        see :meth:`.Shell.get_size`.
    s : :class:`.Shell`
        The web, whose dimensions, outer radius, numbers of terms and
        boundary flags are read.
    xs, ys : np.ndarray
        Coordinates `x` and `y` of the points, lengths, 1-D arrays of the
        same size.
    num_cores : int, optional
        Not used, the points are evaluated serially; kept for the interface
        of :mod:`panels.models.clpt_field`.

    Returns
    -------
    us, vs, ws, phixs, phiys : np.ndarray
        The displacements ``u``, ``v``, ``w`` along the local axes, lengths,
        and the rotations of the normal ``phix = -w,x`` and ``phiy = -(r/rho)
        w,y``, dimensionless, 1-D arrays of the size of ``xs``.

    Notes
    -----
    With `\rho = r - x` the radius of the point, `-(r/\rho) w_{,y} =
    -w_{,\theta}/\rho` is the derivative of `w` along the arc.

    """
    out = np.zeros((5, xs.shape[0]), dtype=DOUBLE)
    for k in range(xs.shape[0]):
        d = _point_derivatives(c, s, xs[k], ys[k])
        out[:, k] = (d[0], d[3], d[6], -d[7], -s.r/(s.r - xs[k])*d[8])
    return tuple(out)


def fstrain(double [::1] c, object s, double [::1] xs, double [::1] ys,
            int num_cores=4, int NLgeom=0):
    r"""Generalized strains of the web at the points ``xs``, ``ys``

    Parameters
    ----------
    c : np.ndarray
        Ritz constants of the web ``s``, 1-D array of size ``3*s.m*s.n``,
        see :meth:`.Shell.get_size`.
    s : :class:`.Shell`
        The web, whose dimensions, outer radius, numbers of terms and
        boundary flags are read.
    xs, ys : np.ndarray
        Coordinates `x` and `y` of the points, lengths, 1-D arrays of the
        same size.
    num_cores : int, optional
        Not used, the points are evaluated serially; kept for the interface
        of :mod:`panels.models.clpt_field`.
    NLgeom : int, optional
        ``1`` to include the quadratic terms of the membrane strains, ``0``
        otherwise. Default ``0``.

    Returns
    -------
    exxs, eyys, gxys, kxxs, kyys, kxys : np.ndarray
        The membrane strains, dimensionless, and the changes of curvature,
        per unit length, along the local axes, 1-D arrays of the size of
        ``xs``.

    Notes
    -----
    The von Kármán strains in polar coordinates given in the notes of the
    module, with the quadratic terms multiplied by ``NLgeom``.

    """
    cdef double rho, s_, R
    R = s.r
    out = np.zeros((6, xs.shape[0]), dtype=DOUBLE)
    for k in range(xs.shape[0]):
        d = _point_derivatives(c, s, xs[k], ys[k])
        rho = R - xs[k]
        s_ = R/rho
        u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy = d
        out[0, k] = u_x + NLgeom*0.5*w_x*w_x
        out[1, k] = s_*v_y - u/rho + NLgeom*0.5*s_*s_*w_y*w_y
        out[2, k] = s_*u_y + v_x + v/rho + NLgeom*s_*w_x*w_y
        out[3, k] = -w_xx
        out[4, k] = w_x/rho - s_*s_*w_yy
        out[5, k] = -2*s_*(w_xy + w_y/rho)
    return tuple(out)

def fkAx_num(*args, **kwargs):
    r"""Not available for the web of a ring frame

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('fkAx_num is not implemented for the model '
                              '"frame_web_clpt_donnell"')

def fkAy_num(*args, **kwargs):
    r"""Not available for the web of a ring frame

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('fkAy_num is not implemented for the model '
                              '"frame_web_clpt_donnell"')
