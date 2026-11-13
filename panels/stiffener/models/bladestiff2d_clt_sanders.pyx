#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: profile=False
#cython: infer_types=False
r"""
Kernels of the connection between a cylindrical skin with Sanders'
kinematics and the flange of the blade stiffener modelled as a plate

Penalty stiffness matrices of the connection between the skin (or the base)
of a curved bay at `y = y_s` and the edge `\eta_f = -1` of the flange of
:class:`.BladeStiff2D`, a plate normal to the skin: the block of the skin
:func:`fkCss`, the coupling block :func:`fkCsf` and the block of the flange
:func:`fkCff`, as :class:`scipy.sparse.coo_matrix`. The degrees of freedom
of each term are ``u, v, w`` of the skin and ``u_f, v_f, w_f`` of the
flange, in its own axes. The module is written by
``theory/multidomain_panels/bladestiff2d_clt_sanders/bladestiff2d_clt_sanders.py``.

Notes
-----
The penalty energy of the connection is

.. math::

    U = \frac{1}{2} \int_0^a k_t \left[(u - u_f)^2 + (v - w_f)^2 + (w +
    v_f)^2\right] + k_r \left(w_{,y} - \frac{v}{r} - w_{f,y}\right)^2 \, dx

with `u, v, w` and `w_{,y}` of the skin at `y = y_s`, and `u_f, v_f, w_f`
and `w_{f,y}` of the flange at its edge `\eta_f = -1`, the local axes of
the flange being rotated by 90 degrees about `x`. The rotation of the
normal of the skin about `x` is that of Sanders [1]_, `-\phi_y = w_{,y} -
v/r`, as in the kinematics of ``cylshell_clpt_sanders``; with Donnell's
`w_{,y}`, the kernels of ``bladestiff2d_clt_donnell``, a rigid rotation of
the cylinder about its axis would strain the connection. Each kernel gives
the Hessian of one part of `U`.

References
----------
.. [1] Sanders, J. L., Nonlinear theories for thin shells, Quarterly of
   Applied Mathematics, 21(1), 21-36, 1963. https://doi.org/10.1090/qam/147023

"""
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE


cdef int DOF = 3
cdef int DOF1 = 3

cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil


def fkCss(double kt, double kr, double ys, double a, double b, double r,
          int m, int n,
          double x1u, double x1ur, double x2u, double x2ur,
          double x1v, double x1vr, double x2v, double x2vr,
          double x1w, double x1wr, double x2w, double x2wr,
          double y1u, double y1ur, double y2u, double y2ur,
          double y1v, double y1vr, double y2v, double y2vr,
          double y1w, double y1wr, double y2w, double y2wr,
          int size, int row0, int col0):
    r"""Penalty stiffness matrix of the connection, block of the skin

    Block of the Ritz constants of the skin of the penalty stiffness
    matrix that connects the cylindrical skin at `y = y_s` to the flange.

    Parameters
    ----------
    kt, kr : float
        Translational and rotational penalty constants of the connection,
        force per unit area and force per unit length, see
        :func:`.calc_kt_kr`.
    ys : float
        Coordinate `y` of the stiffener on the skin, arc length.
    a, b : float
        Dimensions of the skin along `x` and `y`, lengths, `b` being an arc
        length.
    r : float
        Radius of the skin, length.
    m, n : int
        Numbers of terms of the approximation of the skin along `x` and `y`.
    x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr : float
        Boundary flags of the approximation functions of the skin: the
        translation (``t``) and rotation (``r``) flags at the edges ``1`` and
        ``2`` along ``x`` and ``y`` of each field ``u``, ``v``, ``w`` used by
        the kernel, see :class:`.Shell`.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the block starts.

    Returns
    -------
    kCss : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Notes
    -----
    Hessian of the penalty energy of the module with respect to the
    Ritz constants of the skin:

    .. math::

        \boldsymbol{K}_{ss} = \int_0^a k_t \left[(\boldsymbol{S}^u)^\top
        \boldsymbol{S}^u + (\boldsymbol{S}^v)^\top \boldsymbol{S}^v +
        (\boldsymbol{S}^w)^\top \boldsymbol{S}^w\right] + k_r
        \boldsymbol{S}_\phi^\top \boldsymbol{S}_\phi \, dx,
        \qquad \boldsymbol{S}_\phi = \boldsymbol{S}^w_{,y} -
        \frac{1}{r} \boldsymbol{S}^v

    with the rows `\boldsymbol{S}` of the approximation functions of the skin
    at `y = y_s` and `\boldsymbol{S}_\phi` that of Sanders' rotation `w_{,y}
    - v/r`.

    The integral along `x` is evaluated in closed form with the
    integrals of Bardell's functions, `dx = (a/2) d\xi`, and the functions
    along `y` of the skin are evaluated at `\eta_s = 2 y_s/b - 1`.

    """
    cdef double [:, ::1] gtab
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef long [:] kCssr, kCssc
    cdef double [:] kCssv

    cdef double fAufBu, fAvfBv, fAvfBw, fAwfBv, fAwfBw
    cdef double gAu, gBu, gAv, gBv, gAw, gBw, gAw_eta, gBw_eta

    eta = 2*ys/b - 1.

    fdim = 5*m*n*m*n

    kCssr = np.zeros((fdim,), dtype=INT)
    kCssc = np.zeros((fdim,), dtype=INT)
    kCssv = np.zeros((fdim,), dtype=DOUBLE)

    gtab = np.zeros((4, n), dtype=DOUBLE)

    with nogil:
        # kCss
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j in range(n):
            gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
            gtab[1, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
            gtab[2, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
            gtab[3, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)

        for i in range(m):
            for k in range(m):
                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBw = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwfBv = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    gAu = gtab[0, j]
                    gAv = gtab[1, j]
                    gAw = gtab[2, j]
                    gAw_eta = gtab[3, j]

                    for l in range(n):
                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = gtab[0, l]
                        gBv = gtab[1, l]
                        gBw = gtab[2, l]
                        gBw_eta = gtab[3, l]

                        c += 1
                        kCssr[c] = row+0
                        kCssc[c] = col+0
                        kCssv[c] += a*fAufBu*gAu*gBu*kt/2
                        c += 1
                        kCssr[c] = row+1
                        kCssc[c] = col+1
                        kCssv[c] += a*fAvfBv*gAv*gBv*kr/(2*(r*r)) + a*fAvfBv*gAv*gBv*kt/2
                        c += 1
                        kCssr[c] = row+1
                        kCssc[c] = col+2
                        kCssv[c] += -a*fAvfBw*gAv*gBw_eta*kr/(b*r)
                        c += 1
                        kCssr[c] = row+2
                        kCssc[c] = col+1
                        kCssv[c] += -a*fAwfBv*gAw_eta*gBv*kr/(b*r)
                        c += 1
                        kCssr[c] = row+2
                        kCssc[c] = col+2
                        kCssv[c] += a*fAwfBw*gAw*gBw*kt/2 + 2*a*fAwfBw*gAw_eta*gBw_eta*kr/(b*b)

    kCss = coo_matrix((kCssv, (kCssr, kCssc)), shape=(size, size))

    return kCss


def fkCsf(double kt, double kr, double ys, double a, double b, double r,
          double bf, int m, int n, int m1, int n1,
          double x1u, double x1ur, double x2u, double x2ur,
          double x1v, double x1vr, double x2v, double x2vr,
          double x1w, double x1wr, double x2w, double x2wr,
          double y1u, double y1ur, double y2u, double y2ur,
          double y1v, double y1vr, double y2v, double y2vr,
          double y1w, double y1wr, double y2w, double y2wr,
          double x1uf, double x1urf, double x2uf, double x2urf,
          double x1vf, double x1vrf, double x2vf, double x2vrf,
          double x1wf, double x1wrf, double x2wf, double x2wrf,
          double y1uf, double y1urf, double y2uf, double y2urf,
          double y1vf, double y1vrf, double y2vf, double y2vrf,
          double y1wf, double y1wrf, double y2wf, double y2wrf,
          int size, int row0, int col0):
    r"""Penalty stiffness matrix of the connection, coupling block

    Block coupling the Ritz constants of the cylindrical skin (rows) and of
    the flange (columns) of the penalty stiffness matrix of the connection,
    assembled in full.

    Parameters
    ----------
    kt, kr : float
        Translational and rotational penalty constants of the connection,
        force per unit area and force per unit length, see
        :func:`.calc_kt_kr`.
    ys : float
        Coordinate `y` of the stiffener on the skin, arc length.
    a, b : float
        Dimensions of the skin along `x` and `y`, lengths, `b` being an arc
        length.
    r : float
        Radius of the skin, length.
    bf : float
        Height of the flange, its width `b_f` along its own `y`, a length.
    m, n : int
        Numbers of terms of the approximation of the skin along `x` and `y`.
    m1, n1 : int
        Numbers of terms of the approximation of the flange.
    x1u, x1ur, x2u, x2ur, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr, y1u, y1ur, y2u, y2ur, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr : float
        Boundary flags of the approximation functions of the skin: the
        translation (``t``) and rotation (``r``) flags at the edges ``1`` and
        ``2`` along ``x`` and ``y`` of each field ``u``, ``v``, ``w`` used by
        the kernel, see :class:`.Shell`.
    x1uf, x1urf, x2uf, x2urf, x1vf, x1vrf, x2vf, x2vrf, x1wf, x1wrf, x2wf, x2wrf, y1uf, y1urf, y2uf, y2urf, y1vf, y1vrf, y2vf, y2vrf, y1wf, y1wrf, y2wf, y2wrf : float
        Boundary flags of the approximation functions of the flange, in the
        same order as those of the skin.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the block starts.

    Returns
    -------
    kCsf : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. The block is assembled in full.

    Notes
    -----
    Mixed second derivative of the penalty energy of the module:

    .. math::

        \boldsymbol{K}_{sf} = -\int_0^a k_t \left[(\boldsymbol{S}^u)^\top
        \boldsymbol{S}^{u_f} + (\boldsymbol{S}^v)^\top \boldsymbol{S}^{w_f} -
        (\boldsymbol{S}^w)^\top \boldsymbol{S}^{v_f}\right] + k_r
        \boldsymbol{S}_\phi^\top \boldsymbol{S}^{w_f}_{,y} \, dx,
        \qquad \boldsymbol{S}_\phi = \boldsymbol{S}^w_{,y} -
        \frac{1}{r} \boldsymbol{S}^v

    with the rows `\boldsymbol{S}` of the approximation functions of the skin
    at `y = y_s` and those of the flange at its edge `\eta_f = -1`, and
    `\boldsymbol{S}_\phi` that of Sanders' rotation `w_{,y} - v/r` of the
    skin.

    The integral along `x` is evaluated in closed form with the
    integrals of Bardell's functions, `dx = (a/2) d\xi`, and the functions
    along `y` of the skin are evaluated at `\eta_s = 2 y_s/b - 1`.

    """
    cdef double [:, ::1] gtab, gtabf
    cdef int i, j, k1, l1, c, row, col
    cdef double eta

    cdef long [:] kCsfr, kCsfc
    cdef double [:] kCsfv

    cdef double fAurBu, fAvrBw, fAwrBv, fAwrBw
    cdef double gAu, gAv, gAw, gAw_eta, sBu, sBv, sBw, sBw_eta

    eta = 2*ys/b - 1.

    fdim = 4*m*n*m1*n1

    kCsfr = np.zeros((fdim,), dtype=INT)
    kCsfc = np.zeros((fdim,), dtype=INT)
    kCsfv = np.zeros((fdim,), dtype=DOUBLE)

    gtab = np.zeros((4, n), dtype=DOUBLE)
    gtabf = np.zeros((4, n1), dtype=DOUBLE)

    with nogil:
        # kCsf
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j in range(n):
            gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
            gtab[1, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
            gtab[2, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
            gtab[3, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)
        for l1 in range(n1):
            gtabf[0, l1] = f(l1, -1., y1uf, y1urf, y2uf, y2urf)
            gtabf[1, l1] = f(l1, -1., y1vf, y1vrf, y2vf, y2vrf)
            gtabf[2, l1] = f(l1, -1., y1wf, y1wrf, y2wf, y2wrf)
            gtabf[3, l1] = fp(l1, -1., y1wf, y1wrf, y2wf, y2wrf)

        for i in range(m):
            for k1 in range(m1):
                fAurBu = integral_ff(i, k1, x1u, x1ur, x2u, x2ur, x1uf, x1urf, x2uf, x2urf)
                fAvrBw = integral_ff(i, k1, x1v, x1vr, x2v, x2vr, x1wf, x1wrf, x2wf, x2wrf)
                fAwrBv = integral_ff(i, k1, x1w, x1wr, x2w, x2wr, x1vf, x1vrf, x2vf, x2vrf)
                fAwrBw = integral_ff(i, k1, x1w, x1wr, x2w, x2wr, x1wf, x1wrf, x2wf, x2wrf)

                for j in range(n):
                    gAu = gtab[0, j]
                    gAv = gtab[1, j]
                    gAw = gtab[2, j]
                    gAw_eta = gtab[3, j]

                    for l1 in range(n1):
                        row = row0 + DOF1*(j*m + i)
                        col = col0 + DOF1*(l1*m1 + k1)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        sBu = gtabf[0, l1]
                        sBv = gtabf[1, l1]
                        sBw = gtabf[2, l1]
                        sBw_eta = gtabf[3, l1]

                        c += 1
                        kCsfr[c] = row+0
                        kCsfc[c] = col+0
                        kCsfv[c] += -a*fAurBu*gAu*kt*sBu/2
                        c += 1
                        kCsfr[c] = row+1
                        kCsfc[c] = col+2
                        kCsfv[c] += -a*fAvrBw*gAv*kt*sBw/2 + a*fAvrBw*gAv*kr*sBw_eta/(bf*r)
                        c += 1
                        kCsfr[c] = row+2
                        kCsfc[c] = col+1
                        kCsfv[c] += a*fAwrBv*gAw*kt*sBv/2
                        c += 1
                        kCsfr[c] = row+2
                        kCsfc[c] = col+2
                        kCsfv[c] += -2*a*fAwrBw*gAw_eta*kr*sBw_eta/(b*bf)

    kCsf = coo_matrix((kCsfv, (kCsfr, kCsfc)), shape=(size, size))

    return kCsf


def fkCff(double kt, double kr, double a, double bf, int m1, int n1,
          double x1uf, double x1urf, double x2uf, double x2urf,
          double x1vf, double x1vrf, double x2vf, double x2vrf,
          double x1wf, double x1wrf, double x2wf, double x2wrf,
          double y1uf, double y1urf, double y2uf, double y2urf,
          double y1vf, double y1vrf, double y2vf, double y2vrf,
          double y1wf, double y1wrf, double y2wf, double y2wrf,
          int size, int row0, int col0):
    r"""Penalty stiffness matrix of the connection, block of the flange

    Block of the Ritz constants of the flange of the penalty stiffness
    matrix that connects its edge `\eta_f = -1` to the skin.

    Parameters
    ----------
    kt, kr : float
        Translational and rotational penalty constants of the connection,
        force per unit area and force per unit length, see
        :func:`.calc_kt_kr`.
    a : float
        Length of the skin and of the flange along `x`.
    bf : float
        Height of the flange, its width `b_f` along its own `y`, a length.
    m1, n1 : int
        Numbers of terms of the approximation of the flange.
    x1uf, x1urf, x2uf, x2urf, x1vf, x1vrf, x2vf, x2vrf, x1wf, x1wrf, x2wf, x2wrf, y1uf, y1urf, y2uf, y2urf, y1vf, y1vrf, y2vf, y2vrf, y1wf, y1wrf, y2wf, y2wrf : float
        Boundary flags of the approximation functions of the flange, in the
        same order as those of the skin.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the block starts.

    Returns
    -------
    kCff : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Notes
    -----
    Hessian of the penalty energy of the module with respect to the
    Ritz constants of the flange, a plate, the same as that of
    ``bladestiff2d_clt_donnell``:

    .. math::

        \boldsymbol{K}_{ff} = \int_0^a k_t \left[(\boldsymbol{S}^{u_f})^\top
        \boldsymbol{S}^{u_f} + (\boldsymbol{S}^{v_f})^\top \boldsymbol{S}^{v_f}
        + (\boldsymbol{S}^{w_f})^\top \boldsymbol{S}^{w_f}\right] + k_r
        (\boldsymbol{S}^{w_f}_{,y})^\top \boldsymbol{S}^{w_f}_{,y} \, dx

    with the rows `\boldsymbol{S}` of the approximation functions of the flange
    at its edge `\eta_f = -1`, `w_{f,y} = (2/b_f) w_{f,\eta}`.

    The integral along `x` is evaluated in closed form with the
    integrals of Bardell's functions, `dx = (a/2) d\xi`,
    with `a` the length of the flange.

    """
    cdef double [:, ::1] gtabf
    cdef int i1, k1, j1, l1, c, row, col

    cdef long [:] kCffr, kCffc
    cdef double [:] kCffv

    cdef double rAurBu, rAvrBv, rAwrBw
    cdef double sAu, sBu, sAv, sBv, sAw, sBw, sAw_eta, sBw_eta

    fdim = 3*m1*n1*m1*n1

    kCffr = np.zeros((fdim,), dtype=INT)
    kCffc = np.zeros((fdim,), dtype=INT)
    kCffv = np.zeros((fdim,), dtype=DOUBLE)

    gtabf = np.zeros((4, n1), dtype=DOUBLE)

    with nogil:

        # kCff
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j1 in range(n1):
            gtabf[0, j1] = f(j1, -1., y1uf, y1urf, y2uf, y2urf)
            gtabf[1, j1] = f(j1, -1., y1vf, y1vrf, y2vf, y2vrf)
            gtabf[2, j1] = f(j1, -1., y1wf, y1wrf, y2wf, y2wrf)
            gtabf[3, j1] = fp(j1, -1., y1wf, y1wrf, y2wf, y2wrf)

        for i1 in range(m1):
            for k1 in range(m1):
                rAurBu = integral_ff(i1, k1, x1uf, x1urf, x2uf, x2urf, x1uf, x1urf, x2uf, x2urf)
                rAvrBv = integral_ff(i1, k1, x1vf, x1vrf, x2vf, x2vrf, x1vf, x1vrf, x2vf, x2vrf)
                rAwrBw = integral_ff(i1, k1, x1wf, x1wrf, x2wf, x2wrf, x1wf, x1wrf, x2wf, x2wrf)

                for j1 in range(n1):
                    sAu = gtabf[0, j1]
                    sAv = gtabf[1, j1]
                    sAw = gtabf[2, j1]
                    sAw_eta = gtabf[3, j1]

                    for l1 in range(n1):
                        row = row0 + DOF1*(j1*m1 + i1)
                        col = col0 + DOF1*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        sBu = gtabf[0, l1]
                        sBv = gtabf[1, l1]
                        sBw = gtabf[2, l1]
                        sBw_eta = gtabf[3, l1]

                        c += 1
                        kCffr[c] = row+0
                        kCffc[c] = col+0
                        kCffv[c] += a*kt*rAurBu*sAu*sBu/2
                        c += 1
                        kCffr[c] = row+1
                        kCffc[c] = col+1
                        kCffv[c] += a*kt*rAvrBv*sAv*sBv/2
                        c += 1
                        kCffr[c] = row+2
                        kCffc[c] = col+2
                        kCffv[c] += a*kt*rAwrBw*sAw*sBw/2 + 2*a*kr*rAwrBw*sAw_eta*sBw_eta/(bf*bf)

    kCff = coo_matrix((kCffv, (kCffr, kCffc)), shape=(size, size))

    return kCff
