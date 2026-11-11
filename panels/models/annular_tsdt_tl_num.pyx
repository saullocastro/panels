#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Annular-sector flat plate, using Reddy's third-order shear deformation
theory (TSDT), total-Lagrangian, in polar coordinates

Numerically integrated kernels of the model ``'annular_tsdt_tl'``:
the constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the mass :func:`fkM_num`, the internal force vector
:func:`calc_fint`, the product of the tangent stiffness by a vector
:func:`calc_kTv` and the strains :func:`fstrain`; the displacements and
rotations :func:`fuvw`, the approximation functions :func:`fg` and the
names :func:`strain_names`, :func:`stress_names` are those of
:mod:`panels.models.fsdt_tsdt_field`. The degree of freedom ``k`` of
the term ``(i, j)``, ``0 <= i < m``, ``0 <= j < n``, is at the row
``row0 + 5*(j*m + i) + k`` of the matrices, in the order ``u, v, w,
phix, phiy``.

The annular sector lies in a plane, between the outer radius ``r``,
``shell.r``, and the inner radius ``r - a``: ``a``, ``shell.a``, is its
radial width and ``b``, ``shell.b``, the arc length of its outer edge. The
coordinate `x = r - \rho` runs from the outer edge inwards, `y = r \theta`
along the outer edge, with the polar coordinates `(\rho, \theta)` of the
plane; ``u``, ``v``, ``w`` are the displacements along
`\boldsymbol{e}_x = -\boldsymbol{e}_\rho`, `\boldsymbol{e}_y =
\boldsymbol{e}_\theta` and `\boldsymbol{e}_z = \boldsymbol{e}_x \times
\boldsymbol{e}_y`, and ``phix``, ``phiy`` the rotations of the normal at
the mid-surface of the laminate, `\phi_x = u_{,z}` and `\phi_y = v_{,z}`
there. These are the local axes of the flange of the connection
``'BFxcte'`` whose base is a cylinder: the web of a ring frame of a
stiffened cylinder is an application, with ``a`` the height of the web,
``b`` the arc length of its outer edge and ``r`` its outer radius. The
laminate spans `d - h/2 \le z \le d + h/2` about the reference plane, `d`
being the offset ``shell.offset``, which enters through the constitutive
matrix ``shell.ABD`` of the TSDT and the mass matrix.

Notes
-----
The displacement field of Reddy's third-order shear deformation theory
[1]_ (TSDT), with the cubic term about the mid-surface of the laminate,
`c_1 = 4/(3 h^2)`, as in ``annular_tsdt_donnell``, is written for finite
rotations as in ``plate_tsdt_tl``: the deformed position of a point is

.. math::

    \boldsymbol{r}(z) = \boldsymbol{X} + z \boldsymbol{d} + p(z)
    \boldsymbol{q}, \qquad p(z) = (z - d)^3 + d^3, \qquad \boldsymbol{q} =
    -c_1 (\boldsymbol{d} - \boldsymbol{N})

with the position `\boldsymbol{X}` of the deformed reference plane, the
second-order director of Schmidt and Reddy [2]_, Eq. (16), and Palmerio,
Reddy and Schmidt [3]_, Eqs. (16) and (20), `\boldsymbol{d} = \phi_x
\boldsymbol{e}_x + \phi_y \boldsymbol{e}_y + (1 - (\phi_x^2 + \phi_y^2)/2)
\boldsymbol{e}_z`, as in ``annular_fsdt_tl``, and the normal of the deformed
reference plane, not normalized, `\boldsymbol{N} = \boldsymbol{g}_1 \times
\boldsymbol{g}_2`. The linear part of `\boldsymbol{d} - \boldsymbol{N}` is
Reddy's vector `(\phi_x + w_{,x}, \phi_y + s w_{,y}, \cdot)`, `s =
r/\rho`.

With, per unit length of the reference lines and in the local basis,
which turns with `\theta`, the tangent vectors `\boldsymbol{g}_1 =
\boldsymbol{X}_{,x} = \{1 + u_{,x}, v_{,x}, w_{,x}\}^\top` and
`\boldsymbol{g}_2 = s \boldsymbol{X}_{,y} = \{s u_{,y} + v/\rho, 1 + s
v_{,y} - u/\rho, s w_{,y}\}^\top`, and, for a vector `\boldsymbol{V} =
\{V_1, V_2, V_3\}^\top`, `\boldsymbol{V}_{,x} = \{V_{1,x}, V_{2,x},
V_{3,x}\}^\top` and `s \boldsymbol{V}_{,y} = \{s V_{1,y} + V_2/\rho, s
V_{2,y} - V_1/\rho, s V_{3,y}\}^\top`, the strains are the coefficients of
`1`, `z`, `p(z)` of the Green-Lagrange strains of `\boldsymbol{r}(z)`, and
of `(z - d)^2` in the transverse shear strains:

.. math::

    \varepsilon_{xx} = \frac{1}{2} \left(\boldsymbol{g}_1 \cdot
    \boldsymbol{g}_1 - 1\right) \qquad
    \varepsilon_{yy} = \frac{1}{2} \left(\boldsymbol{g}_2 \cdot
    \boldsymbol{g}_2 - 1\right) \qquad
    \gamma_{xy} = \boldsymbol{g}_1 \cdot \boldsymbol{g}_2

    \kappa_{xx} = \boldsymbol{g}_1 \cdot \boldsymbol{d}_{,x} \qquad
    \kappa_{yy} = \boldsymbol{g}_2 \cdot s \boldsymbol{d}_{,y} \qquad
    \kappa_{xy} = \boldsymbol{g}_1 \cdot s \boldsymbol{d}_{,y} +
    \boldsymbol{g}_2 \cdot \boldsymbol{d}_{,x}

    \kappa^{(3)}_{xx} = \boldsymbol{g}_1 \cdot \boldsymbol{q}_{,x} \qquad
    \kappa^{(3)}_{yy} = \boldsymbol{g}_2 \cdot s \boldsymbol{q}_{,y} \qquad
    \kappa^{(3)}_{xy} = \boldsymbol{g}_1 \cdot s \boldsymbol{q}_{,y} +
    \boldsymbol{g}_2 \cdot \boldsymbol{q}_{,x}

    \gamma_{yz} = \boldsymbol{g}_2 \cdot \boldsymbol{d} \qquad
    \gamma_{xz} = \boldsymbol{g}_1 \cdot \boldsymbol{d} \qquad
    \gamma^{(2)}_{yz} = -3 c_1 \gamma_{yz} \qquad
    \gamma^{(2)}_{xz} = -3 c_1 \gamma_{xz}

the last two because `\boldsymbol{g}_\alpha \cdot \boldsymbol{N} = 0`, with
`\boldsymbol{\varepsilon}(z) = \boldsymbol{\varepsilon}^{(0)} + z
\boldsymbol{\kappa} + p(z) \boldsymbol{\kappa}^{(3)}` and
`\boldsymbol{\gamma}(z) = \boldsymbol{\gamma} + (z - d)^2
\boldsymbol{\gamma}^{(2)}`, in the order of the rows of ``shell.ABD``, the
constitutive matrix of the TSDT with the offset `d`, see
:meth:`.Shell._get_lam_ABD`. The transverse shear strains `\boldsymbol{\gamma}
(1 - 4 (z - d)^2/h^2)` vanish at both faces for any deformation. Dropped,
as in the linear TSDT: the terms in `z^2`, `z p`, `p^2` of the in-plane
strains, the other powers of the transverse shear strains and the
transverse normal strain. The membrane strains are those of
``annular_tsdt_gl``, the Green-Lagrange strains of Amabili [4]_, Eqs.
(1.166)-(1.168); `\boldsymbol{\kappa}` and `\boldsymbol{\gamma}` are those
of ``annular_fsdt_tl``; the linear part is that of
``annular_tsdt_donnell``, and for `r \to \infty` the strains are those of
``plate_tsdt_tl``. A finite rigid rotation `t` changes the strains by
`O(t^4)` or less, from the second-order director. The second derivatives
of the strains with respect to the generalized derivatives are linear in
them. The mass matrix is that of the linear kinematics, as in
``annular_tsdt_gl``.

The element of area is `dA = (\rho/r)\, dx\, dy`. The integrands are
rational functions of `x`, integrated with the Gauss-Legendre rule over the
domain of :meth:`.Shell.integration_limits`.

References
----------
.. [1] Reddy, J. N., A simple higher-order theory for laminated composite
   plates, J. Appl. Mech. 51 (1984) 745-752.
   https://doi.org/10.1115/1.3167719
.. [2] Schmidt, R., and Reddy, J. N., A refined small strain and moderate
   rotation theory of elastic anisotropic shells, Journal of Applied
   Mechanics, Vol. 55, No. 3, 1988, pp. 611-617, Eq. (16).
   https://doi.org/10.1115/1.3125837
.. [3] Palmerio, A. F., Reddy, J. N., and Schmidt, R., On a moderate
   rotation theory of laminated anisotropic shells - Part 1. Theory,
   International Journal of Non-Linear Mechanics, Vol. 25, No. 6, 1990,
   pp. 687-700, Eqs. (16) and (20).
   https://doi.org/10.1016/0020-7462(90)90007-V
.. [4] Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
   Cambridge University Press, 2008, Sec. 1.4.
   https://doi.org/10.1017/CBO9780511619694

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
from .fsdt_tsdt_field import fg, fuvw, strain_names, stress_names


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 5
cdef int[5] P0 = [0, 6, 12, 18, 21]
cdef int[5] P1 = [6, 12, 18, 21, 24]
#: degree of freedom of each generalized derivative
cdef int[24] FLD = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]


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
        units of the displacements and rotations.
    Finput : array-like
        Constitutive matrix of the TSDT, shape ``(13, 13)`` for the
        whole domain, e.g. ``shell.ABD``, or ``(nx, ny, 13, 13)`` with
        one matrix per integration point, rows and columns in the order of
        the strains of the module.
    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the
        thickness ``lam.h``, the numbers of terms ``m`` and ``n`` and the
        boundary flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this panel start.
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
        radial width ``a``.

    Notes
    -----
    With the generalized derivatives `\boldsymbol{d} = \boldsymbol{\Phi}
    \boldsymbol{c}` of the module, `\boldsymbol{B} = \partial
    \boldsymbol{\varepsilon}/\partial \boldsymbol{d}` at `\boldsymbol{d}`,
    the stress resultants `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{\sigma}_L` of the linear
    strains:

    .. math::

        \boldsymbol{K}_C = \int \boldsymbol{\Phi}^\top \left(
        \boldsymbol{B}^\top \boldsymbol{F} \boldsymbol{B} +
        \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{d}) -
        \boldsymbol{H}(\boldsymbol{\sigma}_L, \boldsymbol{0}) \right)
        \boldsymbol{\Phi} \, dA, \qquad
        \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{d}) = \sum_r
        \sigma_r \frac{\partial^2 \varepsilon_r}{\partial \boldsymbol{d}
        \partial \boldsymbol{d}}

    with `dA = (\rho/r)\, dx\, dy`, see the notes of the module. All the
    stresses enter `\boldsymbol{H}`, linear in `\boldsymbol{d}`.
    With ``NLgeom=0``, `\boldsymbol{d} = \boldsymbol{0}` and the matrix is
    `\int \boldsymbol{\Phi}^\top \boldsymbol{B}_0^\top \boldsymbol{F}
    \boldsymbol{B}_0 \boldsymbol{\Phi} \, dA`.

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, h, c1, intx, inty, xi, eta, xp, rho, s, irho, weight
    cdef int m, n, i, j, k, l, c, p, q, r, e, f2, row, col, ptx, pty
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
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double pA[24]
    cdef double pB[24]
    cdef double dv[24]
    cdef double K[576]
    cdef double TA[120]
    cdef double Bm[312]
    cdef double FB[312]
    cdef double eps[13]
    cdef double epsL[13]
    cdef double sig[13]
    cdef double sigL[13]

    cdef long [::1] kr, kc
    cdef double [::1] kv
    cdef double F[169]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 13, 13):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (13, 13):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(13):
            for j in range(13):
                F[i*13 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')
    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    R = shell.r
    m = shell.m
    n = shell.n
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    if not R > a:
        raise ValueError('the outer radius r must be larger than the radial '
                         'width a, got r={0} and a={1}'.format(R, a))
    #NOTE physical limits of the integration domain, resolved and validated
    #     by Shell.integration_limits()
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
    xi1 = x1/a*2 - 1
    xi2 = x2/a*2 - 1
    eta1 = y1/b*2 - 1
    eta2 = y2/b*2 - 1
    intx = x2 - x1
    inty = y2 - y1
    ftab = np.zeros((15, m), dtype=DOUBLE)
    gtab = np.zeros((15, n), dtype=DOUBLE)

    fdim = 25*m*m*n*n
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
                    for i in range(13):
                        for j in range(13):
                            F[i*13 + j] = Fnxny[ptx, pty, i, j]
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
                    ftab[9, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[10, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[11, i] = (2/a)*(2/a)*fpp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[12, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[13, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[14, i] = (2/a)*(2/a)*fpp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
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
                    gtab[9, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[10, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[11, j] = (2/b)*(2/b)*fpp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[12, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[13, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[14, j] = (2/b)*(2/b)*fpp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                for p in range(312):
                    Bm[p] = 0.
                for p in range(24):
                    dv[p] = 0.
                if NLgeom == 1:
                    for p in range(24):
                        dv[p] = 0.
                    for j in range(n):
                        for i in range(m):
                            col = col0 + 5*(j*m + i)
                            pA[0] = ftab[0, i]*gtab[0, j]
                            pA[1] = ftab[1, i]*gtab[0, j]
                            pA[2] = ftab[0, i]*gtab[1, j]
                            pA[3] = ftab[2, i]*gtab[0, j]
                            pA[4] = ftab[0, i]*gtab[2, j]
                            pA[5] = ftab[1, i]*gtab[1, j]
                            pA[6] = ftab[3, i]*gtab[3, j]
                            pA[7] = ftab[4, i]*gtab[3, j]
                            pA[8] = ftab[3, i]*gtab[4, j]
                            pA[9] = ftab[5, i]*gtab[3, j]
                            pA[10] = ftab[3, i]*gtab[5, j]
                            pA[11] = ftab[4, i]*gtab[4, j]
                            pA[12] = ftab[6, i]*gtab[6, j]
                            pA[13] = ftab[7, i]*gtab[6, j]
                            pA[14] = ftab[6, i]*gtab[7, j]
                            pA[15] = ftab[8, i]*gtab[6, j]
                            pA[16] = ftab[6, i]*gtab[8, j]
                            pA[17] = ftab[7, i]*gtab[7, j]
                            pA[18] = ftab[9, i]*gtab[9, j]
                            pA[19] = ftab[10, i]*gtab[9, j]
                            pA[20] = ftab[9, i]*gtab[10, j]
                            pA[21] = ftab[12, i]*gtab[12, j]
                            pA[22] = ftab[13, i]*gtab[12, j]
                            pA[23] = ftab[12, i]*gtab[13, j]
                            for p in range(24):
                                dv[p] += pA[p]*cs[col + FLD[p]]
                Bm[1] = dv[1] + 1.
                Bm[7] = dv[7]
                Bm[13] = dv[13]
                Bm[24] = dv[0]*(irho)*(irho) - dv[8]*irho*s - irho
                Bm[26] = dv[2]*(s)*(s) + dv[6]*irho*s
                Bm[30] = dv[2]*irho*s + dv[6]*(irho)*(irho)
                Bm[32] = -dv[0]*irho*s + dv[8]*(s)*(s) + s
                Bm[38] = dv[14]*(s)*(s)
                Bm[48] = -dv[7]*irho
                Bm[49] = dv[2]*s + dv[6]*irho
                Bm[50] = dv[1]*s + s
                Bm[54] = dv[1]*irho + irho
                Bm[55] = -dv[0]*irho + dv[8]*s + 1.
                Bm[56] = dv[7]*s
                Bm[61] = dv[14]*s
                Bm[62] = dv[13]*s
                Bm[73] = dv[19]
                Bm[79] = dv[22]
                Bm[85] = -dv[18]*dv[19] - dv[21]*dv[22]
                Bm[90] = -dv[13]*dv[19]
                Bm[91] = -dv[13]*dv[18] + dv[1] + 1.
                Bm[93] = -dv[13]*dv[22]
                Bm[94] = -dv[13]*dv[21] + dv[7]
                Bm[96] = dv[18]*(irho)*(irho) - dv[23]*irho*s
                Bm[98] = dv[20]*(s)*(s) + dv[21]*irho*s
                Bm[102] = dv[20]*irho*s + dv[21]*(irho)*(irho)
                Bm[104] = -dv[18]*irho*s + dv[23]*(s)*(s)
                Bm[110] = -dv[18]*dv[20]*(s)*(s) - dv[21]*dv[23]*(s)*(s)
                Bm[114] = dv[0]*(irho)*(irho) - dv[14]*dv[20]*(s)*(s) - dv[8]*irho*s - irho
                Bm[116] = -dv[14]*dv[18]*(s)*(s) + dv[2]*(s)*(s) + dv[6]*irho*s
                Bm[117] = -dv[14]*dv[23]*(s)*(s) + dv[2]*irho*s + dv[6]*(irho)*(irho)
                Bm[119] = -dv[0]*irho*s - dv[14]*dv[21]*(s)*(s) + dv[8]*(s)*(s) + s
                Bm[120] = -dv[22]*irho
                Bm[121] = dv[20]*s + dv[21]*irho
                Bm[122] = dv[19]*s
                Bm[126] = dv[19]*irho
                Bm[127] = -dv[18]*irho + dv[23]*s
                Bm[128] = dv[22]*s
                Bm[133] = -dv[18]*dv[20]*s - dv[21]*dv[23]*s
                Bm[134] = -dv[18]*dv[19]*s - dv[21]*dv[22]*s
                Bm[138] = -dv[13]*dv[20]*s - dv[14]*dv[19]*s - dv[7]*irho
                Bm[139] = -dv[14]*dv[18]*s + dv[2]*s + dv[6]*irho
                Bm[140] = -dv[13]*dv[18]*s + dv[1]*s + s
                Bm[141] = -dv[13]*dv[23]*s - dv[14]*dv[22]*s + dv[1]*irho + irho
                Bm[142] = -dv[0]*irho - dv[14]*dv[21]*s + dv[8]*s + 1.
                Bm[143] = -dv[13]*dv[21]*s + dv[7]*s
                Bm[144] = -c1*dv[13]*dv[3]*irho + c1*dv[15]*dv[1]*irho + c1*dv[15]*irho
                Bm[145] = c1*dv[0]*dv[15]*irho + c1*dv[14]*dv[9]*s - c1*dv[15]*dv[8]*s - c1*dv[15] - c1*dv[19]
                Bm[146] = -c1*dv[13]*dv[9]*s + c1*dv[15]*dv[7]*s
                Bm[147] = -c1*dv[0]*dv[13]*irho + c1*dv[13]*dv[8]*s + c1*dv[13] - c1*dv[14]*dv[7]*s
                Bm[150] = -c1*dv[13]*dv[9]*irho + c1*dv[15]*dv[7]*irho
                Bm[151] = -c1*dv[14]*dv[3]*s + c1*dv[15]*dv[2]*s + c1*dv[15]*dv[6]*irho - c1*dv[22]
                Bm[152] = c1*dv[13]*dv[3]*s - c1*dv[15]*dv[1]*s - c1*dv[15]*s
                Bm[153] = -c1*dv[13]*dv[2]*s - c1*dv[13]*dv[6]*irho + c1*dv[14]*dv[1]*s + c1*dv[14]*s
                Bm[157] = -c1*dv[0]*dv[3]*irho + c1*dv[18]*dv[19] + c1*dv[21]*dv[22] - c1*dv[2]*dv[9]*s + c1*dv[3]*dv[8]*s + c1*dv[3] - c1*dv[6]*dv[9]*irho
                Bm[158] = c1*dv[1]*dv[9]*s - c1*dv[3]*dv[7]*s + c1*dv[9]*s
                Bm[159] = c1*dv[0]*dv[1]*irho + c1*dv[0]*irho - c1*dv[1]*dv[8]*s - c1*dv[1] + c1*dv[2]*dv[7]*s + c1*dv[6]*dv[7]*irho - c1*dv[8]*s - c1
                Bm[162] = c1*dv[13]*dv[19]
                Bm[163] = c1*dv[13]*dv[18] - c1*dv[1] - c1
                Bm[165] = c1*dv[13]*dv[22]
                Bm[166] = c1*dv[13]*dv[21] - c1*dv[7]
                Bm[168] = 2.*c1*dv[0]*dv[13]*(irho)*(irho)*(irho) - c1*dv[13]*dv[4]*irho*(s)*(s) - 3.*c1*dv[13]*dv[8]*(irho)*(irho)*s - 2.*c1*dv[13]*(irho)*(irho) + c1*dv[14]*dv[7]*(irho)*(irho)*s + c1*dv[16]*dv[1]*irho*(s)*(s) + c1*dv[16]*irho*(s)*(s) - c1*dv[18]*(irho)*(irho) + c1*dv[23]*irho*s
                Bm[169] = c1*dv[0]*dv[16]*irho*(s)*(s) + c1*dv[10]*dv[14]*(s)*(s)*(s) - 2.*c1*dv[14]*dv[2]*irho*(s)*(s) - c1*dv[14]*dv[6]*(irho)*(irho)*s - c1*dv[16]*dv[8]*(s)*(s)*(s) - c1*dv[16]*(s)*(s)
                Bm[170] = -c1*dv[10]*dv[13]*(s)*(s)*(s) + 4.*c1*dv[13]*dv[2]*irho*(s)*(s) + 3.*c1*dv[13]*dv[6]*(irho)*(irho)*s - 2.*c1*dv[14]*dv[1]*irho*(s)*(s) - 2.*c1*dv[14]*irho*(s)*(s) + c1*dv[16]*dv[7]*(s)*(s)*(s) - c1*dv[20]*(s)*(s) - c1*dv[21]*irho*s
                Bm[172] = -c1*dv[0]*dv[13]*irho*(s)*(s) + c1*dv[13]*dv[8]*(s)*(s)*(s) + c1*dv[13]*(s)*(s) - c1*dv[14]*dv[7]*(s)*(s)*(s)
                Bm[174] = -c1*dv[10]*dv[13]*irho*(s)*(s) + 3.*c1*dv[13]*dv[2]*(irho)*(irho)*s + 2.*c1*dv[13]*dv[6]*(irho)*(irho)*(irho) - c1*dv[14]*dv[1]*(irho)*(irho)*s - c1*dv[14]*(irho)*(irho)*s + c1*dv[16]*dv[7]*irho*(s)*(s) - c1*dv[20]*irho*s - c1*dv[21]*(irho)*(irho)
                Bm[175] = c1*dv[0]*dv[14]*(irho)*(irho)*s - c1*dv[14]*dv[4]*(s)*(s)*(s) - 2.*c1*dv[14]*dv[8]*irho*(s)*(s) - c1*dv[14]*irho*s + c1*dv[16]*dv[2]*(s)*(s)*(s) + c1*dv[16]*dv[6]*irho*(s)*(s)
                Bm[176] = -3.*c1*dv[0]*dv[13]*(irho)*(irho)*s + c1*dv[13]*dv[4]*(s)*(s)*(s) + 4.*c1*dv[13]*dv[8]*irho*(s)*(s) + 3.*c1*dv[13]*irho*s - 2.*c1*dv[14]*dv[7]*irho*(s)*(s) - c1*dv[16]*dv[1]*(s)*(s)*(s) - c1*dv[16]*(s)*(s)*(s) + c1*dv[18]*irho*s - c1*dv[23]*(s)*(s)
                Bm[178] = -c1*dv[13]*dv[2]*(s)*(s)*(s) - c1*dv[13]*dv[6]*irho*(s)*(s) + c1*dv[14]*dv[1]*(s)*(s)*(s) + c1*dv[14]*(s)*(s)*(s)
                Bm[181] = c1*(dv[0])*(dv[0])*(irho)*(irho)*(irho) - c1*dv[0]*dv[4]*irho*(s)*(s) - 3.*c1*dv[0]*dv[8]*(irho)*(irho)*s - 2.*c1*dv[0]*(irho)*(irho) - c1*dv[10]*dv[2]*(s)*(s)*(s) - c1*dv[10]*dv[6]*irho*(s)*(s) + 2.*c1*(dv[2])*(dv[2])*irho*(s)*(s) + 3.*c1*dv[2]*dv[6]*(irho)*(irho)*s + c1*dv[4]*dv[8]*(s)*(s)*(s) + c1*dv[4]*(s)*(s) + c1*(dv[6])*(dv[6])*(irho)*(irho)*(irho) + 2.*c1*(dv[8])*(dv[8])*irho*(s)*(s) + 3.*c1*dv[8]*irho*s + c1*irho
                Bm[182] = c1*dv[0]*dv[7]*(irho)*(irho)*s + c1*dv[10]*dv[1]*(s)*(s)*(s) + c1*dv[10]*(s)*(s)*(s) + c1*dv[18]*dv[20]*(s)*(s) - 2.*c1*dv[1]*dv[2]*irho*(s)*(s) - c1*dv[1]*dv[6]*(irho)*(irho)*s + c1*dv[21]*dv[23]*(s)*(s) - 2.*c1*dv[2]*irho*(s)*(s) - c1*dv[4]*dv[7]*(s)*(s)*(s) - c1*dv[6]*(irho)*(irho)*s - 2.*c1*dv[7]*dv[8]*irho*(s)*(s) - c1*dv[7]*irho*s
                Bm[184] = c1*dv[0]*dv[1]*irho*(s)*(s) + c1*dv[0]*irho*(s)*(s) - c1*dv[1]*dv[8]*(s)*(s)*(s) - c1*dv[1]*(s)*(s) + c1*dv[2]*dv[7]*(s)*(s)*(s) + c1*dv[6]*dv[7]*irho*(s)*(s) - c1*dv[8]*(s)*(s)*(s) - c1*(s)*(s)
                Bm[186] = -c1*dv[0]*(irho)*(irho) + c1*dv[14]*dv[20]*(s)*(s) + c1*dv[8]*irho*s + c1*irho
                Bm[188] = c1*dv[14]*dv[18]*(s)*(s) - c1*dv[2]*(s)*(s) - c1*dv[6]*irho*s
                Bm[189] = c1*dv[14]*dv[23]*(s)*(s) - c1*dv[2]*irho*s - c1*dv[6]*(irho)*(irho)
                Bm[191] = c1*dv[0]*irho*s + c1*dv[14]*dv[21]*(s)*(s) - c1*dv[8]*(s)*(s) - c1*s
                Bm[192] = -2.*c1*dv[13]*dv[5]*irho*s - 2.*c1*dv[13]*dv[7]*(irho)*(irho) + 2.*c1*dv[17]*dv[1]*irho*s + 2.*c1*dv[17]*irho*s + c1*dv[22]*irho
                Bm[193] = 2.*c1*dv[0]*dv[17]*irho*s + 2.*c1*dv[11]*dv[14]*(s)*(s) + 2.*c1*dv[13]*dv[2]*irho*s + 2.*c1*dv[13]*dv[6]*(irho)*(irho) - 4.*c1*dv[14]*dv[1]*irho*s - 4.*c1*dv[14]*irho*s - 2.*c1*dv[17]*dv[8]*(s)*(s) - 2.*c1*dv[17]*s - c1*dv[20]*s - c1*dv[21]*irho
                Bm[194] = -2.*c1*dv[11]*dv[13]*(s)*(s) + 2.*c1*dv[13]*dv[1]*irho*s + 2.*c1*dv[13]*irho*s + 2.*c1*dv[17]*dv[7]*(s)*(s) - c1*dv[19]*s
                Bm[197] = -2.*c1*dv[0]*dv[13]*irho*s + 2.*c1*dv[13]*dv[8]*(s)*(s) + 2.*c1*dv[13]*s - 2.*c1*dv[14]*dv[7]*(s)*(s)
                Bm[198] = -2.*c1*dv[11]*dv[13]*irho*s + 2.*c1*dv[13]*dv[1]*(irho)*(irho) + 2.*c1*dv[13]*(irho)*(irho) + 2.*c1*dv[17]*dv[7]*irho*s - c1*dv[19]*irho
                Bm[199] = -2.*c1*dv[0]*dv[13]*(irho)*(irho) + 2.*c1*dv[13]*dv[8]*irho*s + 2.*c1*dv[13]*irho - 2.*c1*dv[14]*dv[5]*(s)*(s) - 4.*c1*dv[14]*dv[7]*irho*s + 2.*c1*dv[17]*dv[2]*(s)*(s) + 2.*c1*dv[17]*dv[6]*irho*s + c1*dv[18]*irho - c1*dv[23]*s
                Bm[200] = 2.*c1*dv[13]*dv[5]*(s)*(s) + 2.*c1*dv[13]*dv[7]*irho*s - 2.*c1*dv[17]*dv[1]*(s)*(s) - 2.*c1*dv[17]*(s)*(s) - c1*dv[22]*s
                Bm[203] = -2.*c1*dv[13]*dv[2]*(s)*(s) - 2.*c1*dv[13]*dv[6]*irho*s + 2.*c1*dv[14]*dv[1]*(s)*(s) + 2.*c1*dv[14]*(s)*(s)
                Bm[205] = -2.*c1*dv[0]*dv[5]*irho*s - 2.*c1*dv[0]*dv[7]*(irho)*(irho) - 2.*c1*dv[11]*dv[2]*(s)*(s) - 2.*c1*dv[11]*dv[6]*irho*s + c1*dv[18]*dv[20]*s + 2.*c1*dv[1]*dv[2]*irho*s + 2.*c1*dv[1]*dv[6]*(irho)*(irho) + c1*dv[21]*dv[23]*s + 2.*c1*dv[2]*irho*s + 2.*c1*dv[5]*dv[8]*(s)*(s) + 2.*c1*dv[5]*s + 2.*c1*dv[6]*(irho)*(irho) + 2.*c1*dv[7]*dv[8]*irho*s + 2.*c1*dv[7]*irho
                Bm[206] = 2.*c1*dv[11]*dv[1]*(s)*(s) + 2.*c1*dv[11]*(s)*(s) + c1*dv[18]*dv[19]*s - 2.*c1*(dv[1])*(dv[1])*irho*s - 4.*c1*dv[1]*irho*s + c1*dv[21]*dv[22]*s - 2.*c1*dv[5]*dv[7]*(s)*(s) - 2.*c1*(dv[7])*(dv[7])*irho*s - 2.*c1*irho*s
                Bm[209] = 2.*c1*dv[0]*dv[1]*irho*s + 2.*c1*dv[0]*irho*s - 2.*c1*dv[1]*dv[8]*(s)*(s) - 2.*c1*dv[1]*s + 2.*c1*dv[2]*dv[7]*(s)*(s) + 2.*c1*dv[6]*dv[7]*irho*s - 2.*c1*dv[8]*(s)*(s) - 2.*c1*s
                Bm[210] = c1*dv[13]*dv[20]*s + c1*dv[14]*dv[19]*s + c1*dv[7]*irho
                Bm[211] = c1*dv[14]*dv[18]*s - c1*dv[2]*s - c1*dv[6]*irho
                Bm[212] = c1*dv[13]*dv[18]*s - c1*dv[1]*s - c1*s
                Bm[213] = c1*dv[13]*dv[23]*s + c1*dv[14]*dv[22]*s - c1*dv[1]*irho - c1*irho
                Bm[214] = c1*dv[0]*irho + c1*dv[14]*dv[21]*s - c1*dv[8]*s - c1
                Bm[215] = c1*dv[13]*dv[21]*s - c1*dv[7]*s
                Bm[216] = -dv[21]*irho
                Bm[218] = dv[18]*s
                Bm[222] = dv[18]*irho
                Bm[224] = dv[21]*s
                Bm[230] = -(dv[18])*(dv[18])*s/2. - (dv[21])*(dv[21])*s/2. + s
                Bm[234] = -dv[14]*dv[18]*s + dv[2]*s + dv[6]*irho
                Bm[237] = -dv[0]*irho - dv[14]*dv[21]*s + dv[8]*s + 1.
                Bm[241] = dv[18]
                Bm[247] = dv[21]
                Bm[253] = -(dv[18])*(dv[18])/2. - (dv[21])*(dv[21])/2. + 1.
                Bm[258] = -dv[13]*dv[18] + dv[1] + 1.
                Bm[261] = -dv[13]*dv[21] + dv[7]
                Bm[264] = 3.*c1*dv[21]*irho
                Bm[266] = -3.*c1*dv[18]*s
                Bm[270] = -3.*c1*dv[18]*irho
                Bm[272] = -3.*c1*dv[21]*s
                Bm[278] = 3.*c1*(dv[18])*(dv[18])*s/2. + 3.*c1*(dv[21])*(dv[21])*s/2. - 3.*c1*s
                Bm[282] = 3.*c1*dv[14]*dv[18]*s - 3.*c1*dv[2]*s - 3.*c1*dv[6]*irho
                Bm[285] = 3.*c1*dv[0]*irho + 3.*c1*dv[14]*dv[21]*s - 3.*c1*dv[8]*s - 3.*c1
                Bm[289] = -3.*c1*dv[18]
                Bm[295] = -3.*c1*dv[21]
                Bm[301] = 3.*c1*(dv[18])*(dv[18])/2. + 3.*c1*(dv[21])*(dv[21])/2. - 3.*c1
                Bm[306] = 3.*c1*dv[13]*dv[18] - 3.*c1*dv[1] - 3.*c1
                Bm[309] = 3.*c1*dv[13]*dv[21] - 3.*c1*dv[7]
                # K = [B]^T [F] [B]
                for p in range(13):
                    for q in range(24):
                        FB[p*24 + q] = 0.
                        for r in range(13):
                            FB[p*24 + q] += F[p*13 + r]*Bm[r*24 + q]
                for p in range(24):
                    for q in range(24):
                        K[p*24 + q] = 0.
                        for r in range(13):
                            K[p*24 + q] += Bm[r*24 + p]*FB[r*24 + q]
                if NLgeom == 1:
                    # + [H(sigma, d)] - [H(sigma_L, 0)]
                    eps[0] = (dv[13])*(dv[13])/2. + (dv[1])*(dv[1])/2. + dv[1] + (dv[7])*(dv[7])/2.
                    eps[1] = (dv[0])*(dv[0])*(irho)*(irho)/2. - dv[0]*dv[8]*irho*s - dv[0]*irho + (dv[14])*(dv[14])*(s)*(s)/2. + (dv[2])*(dv[2])*(s)*(s)/2. + dv[2]*dv[6]*irho*s + (dv[6])*(dv[6])*(irho)*(irho)/2. + (dv[8])*(dv[8])*(s)*(s)/2. + dv[8]*s
                    eps[2] = -dv[0]*dv[7]*irho + dv[13]*dv[14]*s + dv[1]*dv[2]*s + dv[1]*dv[6]*irho + dv[2]*s + dv[6]*irho + dv[7]*dv[8]*s + dv[7]
                    eps[3] = -dv[13]*dv[18]*dv[19] - dv[13]*dv[21]*dv[22] + dv[19]*dv[1] + dv[19] + dv[22]*dv[7]
                    eps[4] = dv[0]*dv[18]*(irho)*(irho) - dv[0]*dv[23]*irho*s - dv[14]*dv[18]*dv[20]*(s)*(s) - dv[14]*dv[21]*dv[23]*(s)*(s) - dv[18]*dv[8]*irho*s - dv[18]*irho + dv[20]*dv[2]*(s)*(s) + dv[20]*dv[6]*irho*s + dv[21]*dv[2]*irho*s + dv[21]*dv[6]*(irho)*(irho) + dv[23]*dv[8]*(s)*(s) + dv[23]*s
                    eps[5] = -dv[0]*dv[22]*irho - dv[13]*dv[18]*dv[20]*s - dv[13]*dv[21]*dv[23]*s - dv[14]*dv[18]*dv[19]*s - dv[14]*dv[21]*dv[22]*s - dv[18]*dv[7]*irho + dv[19]*dv[2]*s + dv[19]*dv[6]*irho + dv[1]*dv[20]*s + dv[1]*dv[21]*irho + dv[20]*s + dv[21]*irho + dv[22]*dv[8]*s + dv[22] + dv[23]*dv[7]*s
                    eps[6] = -c1*dv[0]*dv[13]*dv[3]*irho + c1*dv[0]*dv[15]*dv[1]*irho + c1*dv[0]*dv[15]*irho + c1*dv[13]*dv[18]*dv[19] + c1*dv[13]*dv[21]*dv[22] - c1*dv[13]*dv[2]*dv[9]*s + c1*dv[13]*dv[3]*dv[8]*s + c1*dv[13]*dv[3] - c1*dv[13]*dv[6]*dv[9]*irho + c1*dv[14]*dv[1]*dv[9]*s - c1*dv[14]*dv[3]*dv[7]*s + c1*dv[14]*dv[9]*s - c1*dv[15]*dv[1]*dv[8]*s - c1*dv[15]*dv[1] + c1*dv[15]*dv[2]*dv[7]*s + c1*dv[15]*dv[6]*dv[7]*irho - c1*dv[15]*dv[8]*s - c1*dv[15] - c1*dv[19]*dv[1] - c1*dv[19] - c1*dv[22]*dv[7]
                    eps[7] = c1*(dv[0])*(dv[0])*dv[13]*(irho)*(irho)*(irho) - c1*dv[0]*dv[13]*dv[4]*irho*(s)*(s) - 3.*c1*dv[0]*dv[13]*dv[8]*(irho)*(irho)*s - 2.*c1*dv[0]*dv[13]*(irho)*(irho) + c1*dv[0]*dv[14]*dv[7]*(irho)*(irho)*s + c1*dv[0]*dv[16]*dv[1]*irho*(s)*(s) + c1*dv[0]*dv[16]*irho*(s)*(s) - c1*dv[0]*dv[18]*(irho)*(irho) + c1*dv[0]*dv[23]*irho*s - c1*dv[10]*dv[13]*dv[2]*(s)*(s)*(s) - c1*dv[10]*dv[13]*dv[6]*irho*(s)*(s) + c1*dv[10]*dv[14]*dv[1]*(s)*(s)*(s) + c1*dv[10]*dv[14]*(s)*(s)*(s) + 2.*c1*dv[13]*(dv[2])*(dv[2])*irho*(s)*(s) + 3.*c1*dv[13]*dv[2]*dv[6]*(irho)*(irho)*s + c1*dv[13]*dv[4]*dv[8]*(s)*(s)*(s) + c1*dv[13]*dv[4]*(s)*(s) + c1*dv[13]*(dv[6])*(dv[6])*(irho)*(irho)*(irho) + 2.*c1*dv[13]*(dv[8])*(dv[8])*irho*(s)*(s) + 3.*c1*dv[13]*dv[8]*irho*s + c1*dv[13]*irho + c1*dv[14]*dv[18]*dv[20]*(s)*(s) - 2.*c1*dv[14]*dv[1]*dv[2]*irho*(s)*(s) - c1*dv[14]*dv[1]*dv[6]*(irho)*(irho)*s + c1*dv[14]*dv[21]*dv[23]*(s)*(s) - 2.*c1*dv[14]*dv[2]*irho*(s)*(s) - c1*dv[14]*dv[4]*dv[7]*(s)*(s)*(s) - c1*dv[14]*dv[6]*(irho)*(irho)*s - 2.*c1*dv[14]*dv[7]*dv[8]*irho*(s)*(s) - c1*dv[14]*dv[7]*irho*s - c1*dv[16]*dv[1]*dv[8]*(s)*(s)*(s) - c1*dv[16]*dv[1]*(s)*(s) + c1*dv[16]*dv[2]*dv[7]*(s)*(s)*(s) + c1*dv[16]*dv[6]*dv[7]*irho*(s)*(s) - c1*dv[16]*dv[8]*(s)*(s)*(s) - c1*dv[16]*(s)*(s) + c1*dv[18]*dv[8]*irho*s + c1*dv[18]*irho - c1*dv[20]*dv[2]*(s)*(s) - c1*dv[20]*dv[6]*irho*s - c1*dv[21]*dv[2]*irho*s - c1*dv[21]*dv[6]*(irho)*(irho) - c1*dv[23]*dv[8]*(s)*(s) - c1*dv[23]*s
                    eps[8] = -2.*c1*dv[0]*dv[13]*dv[5]*irho*s - 2.*c1*dv[0]*dv[13]*dv[7]*(irho)*(irho) + 2.*c1*dv[0]*dv[17]*dv[1]*irho*s + 2.*c1*dv[0]*dv[17]*irho*s + c1*dv[0]*dv[22]*irho - 2.*c1*dv[11]*dv[13]*dv[2]*(s)*(s) - 2.*c1*dv[11]*dv[13]*dv[6]*irho*s + 2.*c1*dv[11]*dv[14]*dv[1]*(s)*(s) + 2.*c1*dv[11]*dv[14]*(s)*(s) + c1*dv[13]*dv[18]*dv[20]*s + 2.*c1*dv[13]*dv[1]*dv[2]*irho*s + 2.*c1*dv[13]*dv[1]*dv[6]*(irho)*(irho) + c1*dv[13]*dv[21]*dv[23]*s + 2.*c1*dv[13]*dv[2]*irho*s + 2.*c1*dv[13]*dv[5]*dv[8]*(s)*(s) + 2.*c1*dv[13]*dv[5]*s + 2.*c1*dv[13]*dv[6]*(irho)*(irho) + 2.*c1*dv[13]*dv[7]*dv[8]*irho*s + 2.*c1*dv[13]*dv[7]*irho + c1*dv[14]*dv[18]*dv[19]*s - 2.*c1*dv[14]*(dv[1])*(dv[1])*irho*s - 4.*c1*dv[14]*dv[1]*irho*s + c1*dv[14]*dv[21]*dv[22]*s - 2.*c1*dv[14]*dv[5]*dv[7]*(s)*(s) - 2.*c1*dv[14]*(dv[7])*(dv[7])*irho*s - 2.*c1*dv[14]*irho*s - 2.*c1*dv[17]*dv[1]*dv[8]*(s)*(s) - 2.*c1*dv[17]*dv[1]*s + 2.*c1*dv[17]*dv[2]*dv[7]*(s)*(s) + 2.*c1*dv[17]*dv[6]*dv[7]*irho*s - 2.*c1*dv[17]*dv[8]*(s)*(s) - 2.*c1*dv[17]*s + c1*dv[18]*dv[7]*irho - c1*dv[19]*dv[2]*s - c1*dv[19]*dv[6]*irho - c1*dv[1]*dv[20]*s - c1*dv[1]*dv[21]*irho - c1*dv[20]*s - c1*dv[21]*irho - c1*dv[22]*dv[8]*s - c1*dv[22] - c1*dv[23]*dv[7]*s
                    eps[9] = -dv[0]*dv[21]*irho - dv[14]*(dv[18])*(dv[18])*s/2. - dv[14]*(dv[21])*(dv[21])*s/2. + dv[14]*s + dv[18]*dv[2]*s + dv[18]*dv[6]*irho + dv[21]*dv[8]*s + dv[21]
                    eps[10] = -dv[13]*(dv[18])*(dv[18])/2. - dv[13]*(dv[21])*(dv[21])/2. + dv[13] + dv[18]*dv[1] + dv[18] + dv[21]*dv[7]
                    eps[11] = 3.*c1*dv[0]*dv[21]*irho + 3.*c1*dv[14]*(dv[18])*(dv[18])*s/2. + 3.*c1*dv[14]*(dv[21])*(dv[21])*s/2. - 3.*c1*dv[14]*s - 3.*c1*dv[18]*dv[2]*s - 3.*c1*dv[18]*dv[6]*irho - 3.*c1*dv[21]*dv[8]*s - 3.*c1*dv[21]
                    eps[12] = 3.*c1*dv[13]*(dv[18])*(dv[18])/2. + 3.*c1*dv[13]*(dv[21])*(dv[21])/2. - 3.*c1*dv[13] - 3.*c1*dv[18]*dv[1] - 3.*c1*dv[18] - 3.*c1*dv[21]*dv[7]
                    epsL[0] = dv[1]
                    epsL[1] = -dv[0]*irho + dv[8]*s
                    epsL[2] = dv[2]*s + dv[6]*irho + dv[7]
                    epsL[3] = dv[19]
                    epsL[4] = -dv[18]*irho + dv[23]*s
                    epsL[5] = dv[20]*s + dv[21]*irho + dv[22]
                    epsL[6] = -c1*dv[15] - c1*dv[19]
                    epsL[7] = c1*dv[13]*irho - c1*dv[16]*(s)*(s) + c1*dv[18]*irho - c1*dv[23]*s
                    epsL[8] = -2.*c1*dv[14]*irho*s - 2.*c1*dv[17]*s - c1*dv[20]*s - c1*dv[21]*irho - c1*dv[22]
                    epsL[9] = dv[14]*s + dv[21]
                    epsL[10] = dv[13] + dv[18]
                    epsL[11] = -3.*c1*dv[14]*s - 3.*c1*dv[21]
                    epsL[12] = -3.*c1*dv[13] - 3.*c1*dv[18]
                    for r in range(13):
                        sig[r] = 0.
                        for q in range(13):
                            sig[r] += F[r*13 + q]*eps[q]
                    for r in range(13):
                        sigL[r] = 0.
                        for q in range(13):
                            sigL[r] += F[r*13 + q]*epsL[q]
                    K[0] += 2.*c1*dv[13]*(irho)*(irho)*(irho)*sig[7] + (irho)*(irho)*sig[1]
                    K[1] += c1*dv[15]*irho*sig[6] + c1*dv[16]*irho*(s)*(s)*sig[7] + 2.*c1*dv[17]*irho*s*sig[8]
                    K[3] += -c1*dv[13]*irho*sig[6]
                    K[4] += -c1*dv[13]*irho*(s)*(s)*sig[7]
                    K[5] += -2.*c1*dv[13]*irho*s*sig[8]
                    K[7] += -2.*c1*dv[13]*(irho)*(irho)*sig[8] + c1*dv[14]*(irho)*(irho)*s*sig[7] - irho*sig[2]
                    K[8] += -3.*c1*dv[13]*(irho)*(irho)*s*sig[7] - irho*s*sig[1]
                    K[13] += 2.*c1*dv[0]*(irho)*(irho)*(irho)*sig[7] - c1*dv[3]*irho*sig[6] - c1*dv[4]*irho*(s)*(s)*sig[7] - 2.*c1*dv[5]*irho*s*sig[8] - 2.*c1*dv[7]*(irho)*(irho)*sig[8] - 3.*c1*dv[8]*(irho)*(irho)*s*sig[7] - 2.*c1*(irho)*(irho)*sig[7]
                    K[14] += c1*dv[7]*(irho)*(irho)*s*sig[7]
                    K[15] += c1*dv[1]*irho*sig[6] + c1*irho*sig[6]
                    K[16] += c1*dv[1]*irho*(s)*(s)*sig[7] + c1*irho*(s)*(s)*sig[7]
                    K[17] += 2.*c1*dv[1]*irho*s*sig[8] + 2.*c1*irho*s*sig[8]
                    K[18] += -c1*(irho)*(irho)*sig[7] + (irho)*(irho)*sig[4]
                    K[21] += 3.*c1*irho*sig[11] - irho*sig[9]
                    K[22] += c1*irho*sig[8] - irho*sig[5]
                    K[23] += c1*irho*s*sig[7] - irho*s*sig[4]
                    K[24] += c1*dv[15]*irho*sig[6] + c1*dv[16]*irho*(s)*(s)*sig[7] + 2.*c1*dv[17]*irho*s*sig[8]
                    K[25] += -4.*c1*dv[14]*irho*s*sig[8] + sig[0]
                    K[26] += 2.*c1*dv[13]*irho*s*sig[8] - 2.*c1*dv[14]*irho*(s)*(s)*sig[7] + s*sig[2]
                    K[30] += 2.*c1*dv[13]*(irho)*(irho)*sig[8] - c1*dv[14]*(irho)*(irho)*s*sig[7] + irho*sig[2]
                    K[32] += -c1*dv[15]*s*sig[6] - c1*dv[16]*(s)*(s)*(s)*sig[7] - 2.*c1*dv[17]*(s)*(s)*sig[8]
                    K[33] += c1*dv[14]*s*sig[6]
                    K[34] += c1*dv[14]*(s)*(s)*(s)*sig[7]
                    K[35] += 2.*c1*dv[14]*(s)*(s)*sig[8]
                    K[37] += 2.*c1*dv[2]*irho*s*sig[8] + 2.*c1*dv[6]*(irho)*(irho)*sig[8]
                    K[38] += c1*dv[10]*(s)*(s)*(s)*sig[7] + 2.*c1*dv[11]*(s)*(s)*sig[8] - 4.*c1*dv[1]*irho*s*sig[8] - 2.*c1*dv[2]*irho*(s)*(s)*sig[7] - c1*dv[6]*(irho)*(irho)*s*sig[7] + c1*dv[9]*s*sig[6] - 4.*c1*irho*s*sig[8]
                    K[39] += c1*dv[0]*irho*sig[6] - c1*dv[8]*s*sig[6] - c1*sig[6]
                    K[40] += c1*dv[0]*irho*(s)*(s)*sig[7] - c1*dv[8]*(s)*(s)*(s)*sig[7] - c1*(s)*(s)*sig[7]
                    K[41] += 2.*c1*dv[0]*irho*s*sig[8] - 2.*c1*dv[8]*(s)*(s)*sig[8] - 2.*c1*s*sig[8]
                    K[42] += -3.*c1*sig[12] + sig[10]
                    K[43] += -c1*sig[6] + sig[3]
                    K[44] += -c1*s*sig[8] + s*sig[5]
                    K[45] += -c1*irho*sig[8] + irho*sig[5]
                    K[49] += 2.*c1*dv[13]*irho*s*sig[8] - 2.*c1*dv[14]*irho*(s)*(s)*sig[7] + s*sig[2]
                    K[50] += 4.*c1*dv[13]*irho*(s)*(s)*sig[7] + (s)*(s)*sig[1]
                    K[54] += 3.*c1*dv[13]*(irho)*(irho)*s*sig[7] + irho*s*sig[1]
                    K[55] += c1*dv[15]*s*sig[6] + c1*dv[16]*(s)*(s)*(s)*sig[7] + 2.*c1*dv[17]*(s)*(s)*sig[8]
                    K[57] += -c1*dv[13]*s*sig[6]
                    K[58] += -c1*dv[13]*(s)*(s)*(s)*sig[7]
                    K[59] += -2.*c1*dv[13]*(s)*(s)*sig[8]
                    K[61] += -c1*dv[10]*(s)*(s)*(s)*sig[7] - 2.*c1*dv[11]*(s)*(s)*sig[8] + 2.*c1*dv[1]*irho*s*sig[8] + 4.*c1*dv[2]*irho*(s)*(s)*sig[7] + 3.*c1*dv[6]*(irho)*(irho)*s*sig[7] - c1*dv[9]*s*sig[6] + 2.*c1*irho*s*sig[8]
                    K[62] += -2.*c1*dv[1]*irho*(s)*(s)*sig[7] - 2.*c1*irho*(s)*(s)*sig[7]
                    K[63] += c1*dv[7]*s*sig[6]
                    K[64] += c1*dv[7]*(s)*(s)*(s)*sig[7]
                    K[65] += 2.*c1*dv[7]*(s)*(s)*sig[8]
                    K[66] += -3.*c1*s*sig[11] + s*sig[9]
                    K[67] += -c1*s*sig[8] + s*sig[5]
                    K[68] += -c1*(s)*(s)*sig[7] + (s)*(s)*sig[4]
                    K[69] += -c1*irho*s*sig[7] + irho*s*sig[4]
                    K[72] += -c1*dv[13]*irho*sig[6]
                    K[79] += -c1*dv[14]*s*sig[6]
                    K[80] += c1*dv[13]*s*sig[6]
                    K[85] += -c1*dv[0]*irho*sig[6] + c1*dv[8]*s*sig[6] + c1*sig[6]
                    K[86] += -c1*dv[7]*s*sig[6]
                    K[96] += -c1*dv[13]*irho*(s)*(s)*sig[7]
                    K[103] += -c1*dv[14]*(s)*(s)*(s)*sig[7]
                    K[104] += c1*dv[13]*(s)*(s)*(s)*sig[7]
                    K[109] += -c1*dv[0]*irho*(s)*(s)*sig[7] + c1*dv[8]*(s)*(s)*(s)*sig[7] + c1*(s)*(s)*sig[7]
                    K[110] += -c1*dv[7]*(s)*(s)*(s)*sig[7]
                    K[120] += -2.*c1*dv[13]*irho*s*sig[8]
                    K[127] += -2.*c1*dv[14]*(s)*(s)*sig[8]
                    K[128] += 2.*c1*dv[13]*(s)*(s)*sig[8]
                    K[133] += -2.*c1*dv[0]*irho*s*sig[8] + 2.*c1*dv[8]*(s)*(s)*sig[8] + 2.*c1*s*sig[8]
                    K[134] += -2.*c1*dv[7]*(s)*(s)*sig[8]
                    K[145] += 2.*c1*dv[13]*(irho)*(irho)*sig[8] - c1*dv[14]*(irho)*(irho)*s*sig[7] + irho*sig[2]
                    K[146] += 3.*c1*dv[13]*(irho)*(irho)*s*sig[7] + irho*s*sig[1]
                    K[150] += 2.*c1*dv[13]*(irho)*(irho)*(irho)*sig[7] + (irho)*(irho)*sig[1]
                    K[151] += c1*dv[15]*irho*sig[6] + c1*dv[16]*irho*(s)*(s)*sig[7] + 2.*c1*dv[17]*irho*s*sig[8]
                    K[153] += -c1*dv[13]*irho*sig[6]
                    K[154] += -c1*dv[13]*irho*(s)*(s)*sig[7]
                    K[155] += -2.*c1*dv[13]*irho*s*sig[8]
                    K[157] += -c1*dv[10]*irho*(s)*(s)*sig[7] - 2.*c1*dv[11]*irho*s*sig[8] + 2.*c1*dv[1]*(irho)*(irho)*sig[8] + 3.*c1*dv[2]*(irho)*(irho)*s*sig[7] + 2.*c1*dv[6]*(irho)*(irho)*(irho)*sig[7] - c1*dv[9]*irho*sig[6] + 2.*c1*(irho)*(irho)*sig[8]
                    K[158] += -c1*dv[1]*(irho)*(irho)*s*sig[7] - c1*(irho)*(irho)*s*sig[7]
                    K[159] += c1*dv[7]*irho*sig[6]
                    K[160] += c1*dv[7]*irho*(s)*(s)*sig[7]
                    K[161] += 2.*c1*dv[7]*irho*s*sig[8]
                    K[162] += -3.*c1*irho*sig[11] + irho*sig[9]
                    K[163] += -c1*irho*sig[8] + irho*sig[5]
                    K[164] += -c1*irho*s*sig[7] + irho*s*sig[4]
                    K[165] += -c1*(irho)*(irho)*sig[7] + (irho)*(irho)*sig[4]
                    K[168] += -2.*c1*dv[13]*(irho)*(irho)*sig[8] + c1*dv[14]*(irho)*(irho)*s*sig[7] - irho*sig[2]
                    K[170] += c1*dv[15]*s*sig[6] + c1*dv[16]*(s)*(s)*(s)*sig[7] + 2.*c1*dv[17]*(s)*(s)*sig[8]
                    K[171] += -c1*dv[14]*s*sig[6]
                    K[172] += -c1*dv[14]*(s)*(s)*(s)*sig[7]
                    K[173] += -2.*c1*dv[14]*(s)*(s)*sig[8]
                    K[174] += c1*dv[15]*irho*sig[6] + c1*dv[16]*irho*(s)*(s)*sig[7] + 2.*c1*dv[17]*irho*s*sig[8]
                    K[175] += -4.*c1*dv[14]*irho*s*sig[8] + sig[0]
                    K[176] += 2.*c1*dv[13]*irho*s*sig[8] - 2.*c1*dv[14]*irho*(s)*(s)*sig[7] + s*sig[2]
                    K[181] += -2.*c1*dv[0]*(irho)*(irho)*sig[8] + 2.*c1*dv[8]*irho*s*sig[8] + 2.*c1*irho*sig[8]
                    K[182] += c1*dv[0]*(irho)*(irho)*s*sig[7] - c1*dv[3]*s*sig[6] - c1*dv[4]*(s)*(s)*(s)*sig[7] - 2.*c1*dv[5]*(s)*(s)*sig[8] - 4.*c1*dv[7]*irho*s*sig[8] - 2.*c1*dv[8]*irho*(s)*(s)*sig[7] - c1*irho*s*sig[7]
                    K[183] += c1*dv[2]*s*sig[6] + c1*dv[6]*irho*sig[6]
                    K[184] += c1*dv[2]*(s)*(s)*(s)*sig[7] + c1*dv[6]*irho*(s)*(s)*sig[7]
                    K[185] += 2.*c1*dv[2]*(s)*(s)*sig[8] + 2.*c1*dv[6]*irho*s*sig[8]
                    K[186] += c1*irho*sig[8] - irho*sig[5]
                    K[189] += -3.*c1*sig[12] + sig[10]
                    K[190] += -c1*sig[6] + sig[3]
                    K[191] += -c1*s*sig[8] + s*sig[5]
                    K[192] += -3.*c1*dv[13]*(irho)*(irho)*s*sig[7] - irho*s*sig[1]
                    K[193] += -c1*dv[15]*s*sig[6] - c1*dv[16]*(s)*(s)*(s)*sig[7] - 2.*c1*dv[17]*(s)*(s)*sig[8]
                    K[195] += c1*dv[13]*s*sig[6]
                    K[196] += c1*dv[13]*(s)*(s)*(s)*sig[7]
                    K[197] += 2.*c1*dv[13]*(s)*(s)*sig[8]
                    K[199] += 2.*c1*dv[13]*irho*s*sig[8] - 2.*c1*dv[14]*irho*(s)*(s)*sig[7] + s*sig[2]
                    K[200] += 4.*c1*dv[13]*irho*(s)*(s)*sig[7] + (s)*(s)*sig[1]
                    K[205] += -3.*c1*dv[0]*(irho)*(irho)*s*sig[7] + c1*dv[3]*s*sig[6] + c1*dv[4]*(s)*(s)*(s)*sig[7] + 2.*c1*dv[5]*(s)*(s)*sig[8] + 2.*c1*dv[7]*irho*s*sig[8] + 4.*c1*dv[8]*irho*(s)*(s)*sig[7] + 3.*c1*irho*s*sig[7]
                    K[206] += -2.*c1*dv[7]*irho*(s)*(s)*sig[7]
                    K[207] += -c1*dv[1]*s*sig[6] - c1*s*sig[6]
                    K[208] += -c1*dv[1]*(s)*(s)*(s)*sig[7] - c1*(s)*(s)*(s)*sig[7]
                    K[209] += -2.*c1*dv[1]*(s)*(s)*sig[8] - 2.*c1*(s)*(s)*sig[8]
                    K[210] += c1*irho*s*sig[7] - irho*s*sig[4]
                    K[213] += -3.*c1*s*sig[11] + s*sig[9]
                    K[214] += -c1*s*sig[8] + s*sig[5]
                    K[215] += -c1*(s)*(s)*sig[7] + (s)*(s)*sig[4]
                    K[217] += c1*dv[14]*s*sig[6]
                    K[218] += -c1*dv[13]*s*sig[6]
                    K[222] += -c1*dv[13]*irho*sig[6]
                    K[229] += -c1*dv[2]*s*sig[6] - c1*dv[6]*irho*sig[6]
                    K[230] += c1*dv[1]*s*sig[6] + c1*s*sig[6]
                    K[241] += c1*dv[14]*(s)*(s)*(s)*sig[7]
                    K[242] += -c1*dv[13]*(s)*(s)*(s)*sig[7]
                    K[246] += -c1*dv[13]*irho*(s)*(s)*sig[7]
                    K[253] += -c1*dv[2]*(s)*(s)*(s)*sig[7] - c1*dv[6]*irho*(s)*(s)*sig[7]
                    K[254] += c1*dv[1]*(s)*(s)*(s)*sig[7] + c1*(s)*(s)*(s)*sig[7]
                    K[265] += 2.*c1*dv[14]*(s)*(s)*sig[8]
                    K[266] += -2.*c1*dv[13]*(s)*(s)*sig[8]
                    K[270] += -2.*c1*dv[13]*irho*s*sig[8]
                    K[277] += -2.*c1*dv[2]*(s)*(s)*sig[8] - 2.*c1*dv[6]*irho*s*sig[8]
                    K[278] += 2.*c1*dv[1]*(s)*(s)*sig[8] + 2.*c1*(s)*(s)*sig[8]
                    K[312] += 2.*c1*dv[0]*(irho)*(irho)*(irho)*sig[7] - c1*dv[3]*irho*sig[6] - c1*dv[4]*irho*(s)*(s)*sig[7] - 2.*c1*dv[5]*irho*s*sig[8] - 2.*c1*dv[7]*(irho)*(irho)*sig[8] - 3.*c1*dv[8]*(irho)*(irho)*s*sig[7] - 2.*c1*(irho)*(irho)*sig[7]
                    K[313] += 2.*c1*dv[2]*irho*s*sig[8] + 2.*c1*dv[6]*(irho)*(irho)*sig[8]
                    K[314] += -c1*dv[10]*(s)*(s)*(s)*sig[7] - 2.*c1*dv[11]*(s)*(s)*sig[8] + 2.*c1*dv[1]*irho*s*sig[8] + 4.*c1*dv[2]*irho*(s)*(s)*sig[7] + 3.*c1*dv[6]*(irho)*(irho)*s*sig[7] - c1*dv[9]*s*sig[6] + 2.*c1*irho*s*sig[8]
                    K[315] += -c1*dv[0]*irho*sig[6] + c1*dv[8]*s*sig[6] + c1*sig[6]
                    K[316] += -c1*dv[0]*irho*(s)*(s)*sig[7] + c1*dv[8]*(s)*(s)*(s)*sig[7] + c1*(s)*(s)*sig[7]
                    K[317] += -2.*c1*dv[0]*irho*s*sig[8] + 2.*c1*dv[8]*(s)*(s)*sig[8] + 2.*c1*s*sig[8]
                    K[318] += -c1*dv[10]*irho*(s)*(s)*sig[7] - 2.*c1*dv[11]*irho*s*sig[8] + 2.*c1*dv[1]*(irho)*(irho)*sig[8] + 3.*c1*dv[2]*(irho)*(irho)*s*sig[7] + 2.*c1*dv[6]*(irho)*(irho)*(irho)*sig[7] - c1*dv[9]*irho*sig[6] + 2.*c1*(irho)*(irho)*sig[8]
                    K[319] += -2.*c1*dv[0]*(irho)*(irho)*sig[8] + 2.*c1*dv[8]*irho*s*sig[8] + 2.*c1*irho*sig[8]
                    K[320] += -3.*c1*dv[0]*(irho)*(irho)*s*sig[7] + c1*dv[3]*s*sig[6] + c1*dv[4]*(s)*(s)*(s)*sig[7] + 2.*c1*dv[5]*(s)*(s)*sig[8] + 2.*c1*dv[7]*irho*s*sig[8] + 4.*c1*dv[8]*irho*(s)*(s)*sig[7] + 3.*c1*irho*s*sig[7]
                    K[321] += -c1*dv[2]*s*sig[6] - c1*dv[6]*irho*sig[6]
                    K[322] += -c1*dv[2]*(s)*(s)*(s)*sig[7] - c1*dv[6]*irho*(s)*(s)*sig[7]
                    K[323] += -2.*c1*dv[2]*(s)*(s)*sig[8] - 2.*c1*dv[6]*irho*s*sig[8]
                    K[325] += sig[0]
                    K[326] += s*sig[2]
                    K[330] += 3.*c1*dv[18]*sig[12] + c1*dv[19]*sig[6] + c1*dv[20]*s*sig[8] - dv[18]*sig[10] - dv[19]*sig[3] - dv[20]*s*sig[5]
                    K[331] += c1*dv[18]*sig[6] - dv[18]*sig[3]
                    K[332] += c1*dv[18]*s*sig[8] - dv[18]*s*sig[5]
                    K[333] += 3.*c1*dv[21]*sig[12] + c1*dv[22]*sig[6] + c1*dv[23]*s*sig[8] - dv[21]*sig[10] - dv[22]*sig[3] - dv[23]*s*sig[5]
                    K[334] += c1*dv[21]*sig[6] - dv[21]*sig[3]
                    K[335] += c1*dv[21]*s*sig[8] - dv[21]*s*sig[5]
                    K[336] += c1*dv[7]*(irho)*(irho)*s*sig[7]
                    K[337] += c1*dv[10]*(s)*(s)*(s)*sig[7] + 2.*c1*dv[11]*(s)*(s)*sig[8] - 4.*c1*dv[1]*irho*s*sig[8] - 2.*c1*dv[2]*irho*(s)*(s)*sig[7] - c1*dv[6]*(irho)*(irho)*s*sig[7] + c1*dv[9]*s*sig[6] - 4.*c1*irho*s*sig[8]
                    K[338] += -2.*c1*dv[1]*irho*(s)*(s)*sig[7] - 2.*c1*irho*(s)*(s)*sig[7]
                    K[339] += -c1*dv[7]*s*sig[6]
                    K[340] += -c1*dv[7]*(s)*(s)*(s)*sig[7]
                    K[341] += -2.*c1*dv[7]*(s)*(s)*sig[8]
                    K[342] += -c1*dv[1]*(irho)*(irho)*s*sig[7] - c1*(irho)*(irho)*s*sig[7]
                    K[343] += c1*dv[0]*(irho)*(irho)*s*sig[7] - c1*dv[3]*s*sig[6] - c1*dv[4]*(s)*(s)*(s)*sig[7] - 2.*c1*dv[5]*(s)*(s)*sig[8] - 4.*c1*dv[7]*irho*s*sig[8] - 2.*c1*dv[8]*irho*(s)*(s)*sig[7] - c1*irho*s*sig[7]
                    K[344] += -2.*c1*dv[7]*irho*(s)*(s)*sig[7]
                    K[345] += c1*dv[1]*s*sig[6] + c1*s*sig[6]
                    K[346] += c1*dv[1]*(s)*(s)*(s)*sig[7] + c1*(s)*(s)*(s)*sig[7]
                    K[347] += 2.*c1*dv[1]*(s)*(s)*sig[8] + 2.*c1*(s)*(s)*sig[8]
                    K[349] += s*sig[2]
                    K[350] += (s)*(s)*sig[1]
                    K[354] += 3.*c1*dv[18]*s*sig[11] + c1*dv[19]*s*sig[8] + c1*dv[20]*(s)*(s)*sig[7] - dv[18]*s*sig[9] - dv[19]*s*sig[5] - dv[20]*(s)*(s)*sig[4]
                    K[355] += c1*dv[18]*s*sig[8] - dv[18]*s*sig[5]
                    K[356] += c1*dv[18]*(s)*(s)*sig[7] - dv[18]*(s)*(s)*sig[4]
                    K[357] += 3.*c1*dv[21]*s*sig[11] + c1*dv[22]*s*sig[8] + c1*dv[23]*(s)*(s)*sig[7] - dv[21]*s*sig[9] - dv[22]*s*sig[5] - dv[23]*(s)*(s)*sig[4]
                    K[358] += c1*dv[21]*s*sig[8] - dv[21]*s*sig[5]
                    K[359] += c1*dv[21]*(s)*(s)*sig[7] - dv[21]*(s)*(s)*sig[4]
                    K[360] += c1*dv[1]*irho*sig[6] + c1*irho*sig[6]
                    K[361] += c1*dv[0]*irho*sig[6] - c1*dv[8]*s*sig[6] - c1*sig[6]
                    K[362] += c1*dv[7]*s*sig[6]
                    K[366] += c1*dv[7]*irho*sig[6]
                    K[367] += c1*dv[2]*s*sig[6] + c1*dv[6]*irho*sig[6]
                    K[368] += -c1*dv[1]*s*sig[6] - c1*s*sig[6]
                    K[384] += c1*dv[1]*irho*(s)*(s)*sig[7] + c1*irho*(s)*(s)*sig[7]
                    K[385] += c1*dv[0]*irho*(s)*(s)*sig[7] - c1*dv[8]*(s)*(s)*(s)*sig[7] - c1*(s)*(s)*sig[7]
                    K[386] += c1*dv[7]*(s)*(s)*(s)*sig[7]
                    K[390] += c1*dv[7]*irho*(s)*(s)*sig[7]
                    K[391] += c1*dv[2]*(s)*(s)*(s)*sig[7] + c1*dv[6]*irho*(s)*(s)*sig[7]
                    K[392] += -c1*dv[1]*(s)*(s)*(s)*sig[7] - c1*(s)*(s)*(s)*sig[7]
                    K[408] += 2.*c1*dv[1]*irho*s*sig[8] + 2.*c1*irho*s*sig[8]
                    K[409] += 2.*c1*dv[0]*irho*s*sig[8] - 2.*c1*dv[8]*(s)*(s)*sig[8] - 2.*c1*s*sig[8]
                    K[410] += 2.*c1*dv[7]*(s)*(s)*sig[8]
                    K[414] += 2.*c1*dv[7]*irho*s*sig[8]
                    K[415] += 2.*c1*dv[2]*(s)*(s)*sig[8] + 2.*c1*dv[6]*irho*s*sig[8]
                    K[416] += -2.*c1*dv[1]*(s)*(s)*sig[8] - 2.*c1*(s)*(s)*sig[8]
                    K[432] += -c1*(irho)*(irho)*sig[7] + (irho)*(irho)*sig[4]
                    K[433] += -3.*c1*sig[12] + sig[10]
                    K[434] += -3.*c1*s*sig[11] + s*sig[9]
                    K[438] += -3.*c1*irho*sig[11] + irho*sig[9]
                    K[439] += c1*irho*sig[8] - irho*sig[5]
                    K[440] += c1*irho*s*sig[7] - irho*s*sig[4]
                    K[445] += 3.*c1*dv[18]*sig[12] + c1*dv[19]*sig[6] + c1*dv[20]*s*sig[8] - dv[18]*sig[10] - dv[19]*sig[3] - dv[20]*s*sig[5]
                    K[446] += 3.*c1*dv[18]*s*sig[11] + c1*dv[19]*s*sig[8] + c1*dv[20]*(s)*(s)*sig[7] - dv[18]*s*sig[9] - dv[19]*s*sig[5] - dv[20]*(s)*(s)*sig[4]
                    K[450] += 3.*c1*dv[13]*sig[12] + 3.*c1*dv[14]*s*sig[11] - dv[13]*sig[10] - dv[14]*s*sig[9]
                    K[451] += c1*dv[13]*sig[6] + c1*dv[14]*s*sig[8] - dv[13]*sig[3] - dv[14]*s*sig[5]
                    K[452] += c1*dv[13]*s*sig[8] + c1*dv[14]*(s)*(s)*sig[7] - dv[13]*s*sig[5] - dv[14]*(s)*(s)*sig[4]
                    K[457] += -c1*sig[6] + sig[3]
                    K[458] += -c1*s*sig[8] + s*sig[5]
                    K[462] += -c1*irho*sig[8] + irho*sig[5]
                    K[469] += c1*dv[18]*sig[6] - dv[18]*sig[3]
                    K[470] += c1*dv[18]*s*sig[8] - dv[18]*s*sig[5]
                    K[474] += c1*dv[13]*sig[6] + c1*dv[14]*s*sig[8] - dv[13]*sig[3] - dv[14]*s*sig[5]
                    K[481] += -c1*s*sig[8] + s*sig[5]
                    K[482] += -c1*(s)*(s)*sig[7] + (s)*(s)*sig[4]
                    K[486] += -c1*irho*s*sig[7] + irho*s*sig[4]
                    K[493] += c1*dv[18]*s*sig[8] - dv[18]*s*sig[5]
                    K[494] += c1*dv[18]*(s)*(s)*sig[7] - dv[18]*(s)*(s)*sig[4]
                    K[498] += c1*dv[13]*s*sig[8] + c1*dv[14]*(s)*(s)*sig[7] - dv[13]*s*sig[5] - dv[14]*(s)*(s)*sig[4]
                    K[504] += 3.*c1*irho*sig[11] - irho*sig[9]
                    K[505] += -c1*irho*sig[8] + irho*sig[5]
                    K[506] += -c1*irho*s*sig[7] + irho*s*sig[4]
                    K[510] += -c1*(irho)*(irho)*sig[7] + (irho)*(irho)*sig[4]
                    K[511] += -3.*c1*sig[12] + sig[10]
                    K[512] += -3.*c1*s*sig[11] + s*sig[9]
                    K[517] += 3.*c1*dv[21]*sig[12] + c1*dv[22]*sig[6] + c1*dv[23]*s*sig[8] - dv[21]*sig[10] - dv[22]*sig[3] - dv[23]*s*sig[5]
                    K[518] += 3.*c1*dv[21]*s*sig[11] + c1*dv[22]*s*sig[8] + c1*dv[23]*(s)*(s)*sig[7] - dv[21]*s*sig[9] - dv[22]*s*sig[5] - dv[23]*(s)*(s)*sig[4]
                    K[525] += 3.*c1*dv[13]*sig[12] + 3.*c1*dv[14]*s*sig[11] - dv[13]*sig[10] - dv[14]*s*sig[9]
                    K[526] += c1*dv[13]*sig[6] + c1*dv[14]*s*sig[8] - dv[13]*sig[3] - dv[14]*s*sig[5]
                    K[527] += c1*dv[13]*s*sig[8] + c1*dv[14]*(s)*(s)*sig[7] - dv[13]*s*sig[5] - dv[14]*(s)*(s)*sig[4]
                    K[528] += c1*irho*sig[8] - irho*sig[5]
                    K[535] += -c1*sig[6] + sig[3]
                    K[536] += -c1*s*sig[8] + s*sig[5]
                    K[541] += c1*dv[21]*sig[6] - dv[21]*sig[3]
                    K[542] += c1*dv[21]*s*sig[8] - dv[21]*s*sig[5]
                    K[549] += c1*dv[13]*sig[6] + c1*dv[14]*s*sig[8] - dv[13]*sig[3] - dv[14]*s*sig[5]
                    K[552] += c1*irho*s*sig[7] - irho*s*sig[4]
                    K[559] += -c1*s*sig[8] + s*sig[5]
                    K[560] += -c1*(s)*(s)*sig[7] + (s)*(s)*sig[4]
                    K[565] += c1*dv[21]*s*sig[8] - dv[21]*s*sig[5]
                    K[566] += c1*dv[21]*(s)*(s)*sig[7] - dv[21]*(s)*(s)*sig[4]
                    K[573] += c1*dv[13]*s*sig[8] + c1*dv[14]*(s)*(s)*sig[7] - dv[13]*s*sig[5] - dv[14]*(s)*(s)*sig[4]
                    K[0] -= (irho)*(irho)*sigL[1]
                    K[7] -= -irho*sigL[2]
                    K[8] -= -irho*s*sigL[1]
                    K[13] -= -2.*c1*(irho)*(irho)*sigL[7]
                    K[15] -= c1*irho*sigL[6]
                    K[16] -= c1*irho*(s)*(s)*sigL[7]
                    K[17] -= 2.*c1*irho*s*sigL[8]
                    K[18] -= -c1*(irho)*(irho)*sigL[7] + (irho)*(irho)*sigL[4]
                    K[21] -= 3.*c1*irho*sigL[11] - irho*sigL[9]
                    K[22] -= c1*irho*sigL[8] - irho*sigL[5]
                    K[23] -= c1*irho*s*sigL[7] - irho*s*sigL[4]
                    K[25] -= sigL[0]
                    K[26] -= s*sigL[2]
                    K[30] -= irho*sigL[2]
                    K[38] -= -4.*c1*irho*s*sigL[8]
                    K[39] -= -c1*sigL[6]
                    K[40] -= -c1*(s)*(s)*sigL[7]
                    K[41] -= -2.*c1*s*sigL[8]
                    K[42] -= -3.*c1*sigL[12] + sigL[10]
                    K[43] -= -c1*sigL[6] + sigL[3]
                    K[44] -= -c1*s*sigL[8] + s*sigL[5]
                    K[45] -= -c1*irho*sigL[8] + irho*sigL[5]
                    K[49] -= s*sigL[2]
                    K[50] -= (s)*(s)*sigL[1]
                    K[54] -= irho*s*sigL[1]
                    K[61] -= 2.*c1*irho*s*sigL[8]
                    K[62] -= -2.*c1*irho*(s)*(s)*sigL[7]
                    K[66] -= -3.*c1*s*sigL[11] + s*sigL[9]
                    K[67] -= -c1*s*sigL[8] + s*sigL[5]
                    K[68] -= -c1*(s)*(s)*sigL[7] + (s)*(s)*sigL[4]
                    K[69] -= -c1*irho*s*sigL[7] + irho*s*sigL[4]
                    K[85] -= c1*sigL[6]
                    K[109] -= c1*(s)*(s)*sigL[7]
                    K[133] -= 2.*c1*s*sigL[8]
                    K[145] -= irho*sigL[2]
                    K[146] -= irho*s*sigL[1]
                    K[150] -= (irho)*(irho)*sigL[1]
                    K[157] -= 2.*c1*(irho)*(irho)*sigL[8]
                    K[158] -= -c1*(irho)*(irho)*s*sigL[7]
                    K[162] -= -3.*c1*irho*sigL[11] + irho*sigL[9]
                    K[163] -= -c1*irho*sigL[8] + irho*sigL[5]
                    K[164] -= -c1*irho*s*sigL[7] + irho*s*sigL[4]
                    K[165] -= -c1*(irho)*(irho)*sigL[7] + (irho)*(irho)*sigL[4]
                    K[168] -= -irho*sigL[2]
                    K[175] -= sigL[0]
                    K[176] -= s*sigL[2]
                    K[181] -= 2.*c1*irho*sigL[8]
                    K[182] -= -c1*irho*s*sigL[7]
                    K[186] -= c1*irho*sigL[8] - irho*sigL[5]
                    K[189] -= -3.*c1*sigL[12] + sigL[10]
                    K[190] -= -c1*sigL[6] + sigL[3]
                    K[191] -= -c1*s*sigL[8] + s*sigL[5]
                    K[192] -= -irho*s*sigL[1]
                    K[199] -= s*sigL[2]
                    K[200] -= (s)*(s)*sigL[1]
                    K[205] -= 3.*c1*irho*s*sigL[7]
                    K[207] -= -c1*s*sigL[6]
                    K[208] -= -c1*(s)*(s)*(s)*sigL[7]
                    K[209] -= -2.*c1*(s)*(s)*sigL[8]
                    K[210] -= c1*irho*s*sigL[7] - irho*s*sigL[4]
                    K[213] -= -3.*c1*s*sigL[11] + s*sigL[9]
                    K[214] -= -c1*s*sigL[8] + s*sigL[5]
                    K[215] -= -c1*(s)*(s)*sigL[7] + (s)*(s)*sigL[4]
                    K[230] -= c1*s*sigL[6]
                    K[254] -= c1*(s)*(s)*(s)*sigL[7]
                    K[278] -= 2.*c1*(s)*(s)*sigL[8]
                    K[312] -= -2.*c1*(irho)*(irho)*sigL[7]
                    K[314] -= 2.*c1*irho*s*sigL[8]
                    K[315] -= c1*sigL[6]
                    K[316] -= c1*(s)*(s)*sigL[7]
                    K[317] -= 2.*c1*s*sigL[8]
                    K[318] -= 2.*c1*(irho)*(irho)*sigL[8]
                    K[319] -= 2.*c1*irho*sigL[8]
                    K[320] -= 3.*c1*irho*s*sigL[7]
                    K[325] -= sigL[0]
                    K[326] -= s*sigL[2]
                    K[337] -= -4.*c1*irho*s*sigL[8]
                    K[338] -= -2.*c1*irho*(s)*(s)*sigL[7]
                    K[342] -= -c1*(irho)*(irho)*s*sigL[7]
                    K[343] -= -c1*irho*s*sigL[7]
                    K[345] -= c1*s*sigL[6]
                    K[346] -= c1*(s)*(s)*(s)*sigL[7]
                    K[347] -= 2.*c1*(s)*(s)*sigL[8]
                    K[349] -= s*sigL[2]
                    K[350] -= (s)*(s)*sigL[1]
                    K[360] -= c1*irho*sigL[6]
                    K[361] -= -c1*sigL[6]
                    K[368] -= -c1*s*sigL[6]
                    K[384] -= c1*irho*(s)*(s)*sigL[7]
                    K[385] -= -c1*(s)*(s)*sigL[7]
                    K[392] -= -c1*(s)*(s)*(s)*sigL[7]
                    K[408] -= 2.*c1*irho*s*sigL[8]
                    K[409] -= -2.*c1*s*sigL[8]
                    K[416] -= -2.*c1*(s)*(s)*sigL[8]
                    K[432] -= -c1*(irho)*(irho)*sigL[7] + (irho)*(irho)*sigL[4]
                    K[433] -= -3.*c1*sigL[12] + sigL[10]
                    K[434] -= -3.*c1*s*sigL[11] + s*sigL[9]
                    K[438] -= -3.*c1*irho*sigL[11] + irho*sigL[9]
                    K[439] -= c1*irho*sigL[8] - irho*sigL[5]
                    K[440] -= c1*irho*s*sigL[7] - irho*s*sigL[4]
                    K[457] -= -c1*sigL[6] + sigL[3]
                    K[458] -= -c1*s*sigL[8] + s*sigL[5]
                    K[462] -= -c1*irho*sigL[8] + irho*sigL[5]
                    K[481] -= -c1*s*sigL[8] + s*sigL[5]
                    K[482] -= -c1*(s)*(s)*sigL[7] + (s)*(s)*sigL[4]
                    K[486] -= -c1*irho*s*sigL[7] + irho*s*sigL[4]
                    K[504] -= 3.*c1*irho*sigL[11] - irho*sigL[9]
                    K[505] -= -c1*irho*sigL[8] + irho*sigL[5]
                    K[506] -= -c1*irho*s*sigL[7] + irho*s*sigL[4]
                    K[510] -= -c1*(irho)*(irho)*sigL[7] + (irho)*(irho)*sigL[4]
                    K[511] -= -3.*c1*sigL[12] + sigL[10]
                    K[512] -= -3.*c1*s*sigL[11] + s*sigL[9]
                    K[528] -= c1*irho*sigL[8] - irho*sigL[5]
                    K[535] -= -c1*sigL[6] + sigL[3]
                    K[536] -= -c1*s*sigL[8] + s*sigL[5]
                    K[552] -= c1*irho*s*sigL[7] - irho*s*sigL[4]
                    K[559] -= -c1*s*sigL[8] + s*sigL[5]
                    K[560] -= -c1*(s)*(s)*sigL[7] + (s)*(s)*sigL[4]
                c = -1
                for j in range(n):
                    for i in range(m):
                        row = row0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[0, j]
                        pA[4] = ftab[0, i]*gtab[2, j]
                        pA[5] = ftab[1, i]*gtab[1, j]
                        pA[6] = ftab[3, i]*gtab[3, j]
                        pA[7] = ftab[4, i]*gtab[3, j]
                        pA[8] = ftab[3, i]*gtab[4, j]
                        pA[9] = ftab[5, i]*gtab[3, j]
                        pA[10] = ftab[3, i]*gtab[5, j]
                        pA[11] = ftab[4, i]*gtab[4, j]
                        pA[12] = ftab[6, i]*gtab[6, j]
                        pA[13] = ftab[7, i]*gtab[6, j]
                        pA[14] = ftab[6, i]*gtab[7, j]
                        pA[15] = ftab[8, i]*gtab[6, j]
                        pA[16] = ftab[6, i]*gtab[8, j]
                        pA[17] = ftab[7, i]*gtab[7, j]
                        pA[18] = ftab[9, i]*gtab[9, j]
                        pA[19] = ftab[10, i]*gtab[9, j]
                        pA[20] = ftab[9, i]*gtab[10, j]
                        pA[21] = ftab[12, i]*gtab[12, j]
                        pA[22] = ftab[13, i]*gtab[12, j]
                        pA[23] = ftab[12, i]*gtab[13, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(5):
                            for q in range(24):
                                TA[e*24 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*24 + q] += pA[p]*K[p*24 + q]
                        for l in range(n):
                            for k in range(m):
                                col = col0 + 5*(l*m + k)
                                #NOTE upper triangle, the caller symmetrizes
                                if row > col:
                                    continue
                                pB[0] = ftab[0, k]*gtab[0, l]
                                pB[1] = ftab[1, k]*gtab[0, l]
                                pB[2] = ftab[0, k]*gtab[1, l]
                                pB[3] = ftab[2, k]*gtab[0, l]
                                pB[4] = ftab[0, k]*gtab[2, l]
                                pB[5] = ftab[1, k]*gtab[1, l]
                                pB[6] = ftab[3, k]*gtab[3, l]
                                pB[7] = ftab[4, k]*gtab[3, l]
                                pB[8] = ftab[3, k]*gtab[4, l]
                                pB[9] = ftab[5, k]*gtab[3, l]
                                pB[10] = ftab[3, k]*gtab[5, l]
                                pB[11] = ftab[4, k]*gtab[4, l]
                                pB[12] = ftab[6, k]*gtab[6, l]
                                pB[13] = ftab[7, k]*gtab[6, l]
                                pB[14] = ftab[6, k]*gtab[7, l]
                                pB[15] = ftab[8, k]*gtab[6, l]
                                pB[16] = ftab[6, k]*gtab[8, l]
                                pB[17] = ftab[7, k]*gtab[7, l]
                                pB[18] = ftab[9, k]*gtab[9, l]
                                pB[19] = ftab[10, k]*gtab[9, l]
                                pB[20] = ftab[9, k]*gtab[10, l]
                                pB[21] = ftab[12, k]*gtab[12, l]
                                pB[22] = ftab[13, k]*gtab[12, l]
                                pB[23] = ftab[12, k]*gtab[13, l]
                                for e in range(5):
                                    for f2 in range(5):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*24 + q]*pB[q]

    return coo_matrix((kv, (kr, kc)), shape=(size, size))

def fkG_num(double [::1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny,
        double Nxx0=0, double Nyy0=0, double Nxy0=0):
    r"""Geometric stiffness matrix of the linear stress state

    Integrates numerically the geometric stiffness matrix of the stress
    resultants of the linear strains at ``cs``, superposed to the constant
    membrane state ``Nxx0``, ``Nyy0``, ``Nxy0``. The matrix is homogeneous
    of degree one in ``cs``, as the linear buckling eigenvalue problem
    requires; the remaining terms of the tangent stiffness enter
    :func:`fkC_num`.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements and rotations.
    Finput : array-like
        Constitutive matrix of the TSDT, shape ``(13, 13)`` for the
        whole domain, e.g. ``shell.ABD``, or ``(nx, ny, 13, 13)`` with
        one matrix per integration point, rows and columns in the order of
        the strains of the module.
    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the
        thickness ``lam.h``, the numbers of terms ``m`` and ``n`` and the
        boundary flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this panel start.
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
        radial width ``a``.

    Notes
    -----
    With `\boldsymbol{\sigma} = \boldsymbol{\sigma}_L +
    \boldsymbol{\sigma}_0`, `\boldsymbol{\sigma}_L` the stress resultants
    of the linear strains of ``cs`` and `\boldsymbol{\sigma}_0` the
    constant membrane stress resultants `N_{xx0}, N_{yy0}, N_{xy0}`:

    .. math::

        \boldsymbol{K}_G = \int \boldsymbol{\Phi}^\top
        \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{0})
        \boldsymbol{\Phi} \, dA

    with `\boldsymbol{H}` of :func:`fkC_num`, the variation of the work of
    the stresses on the second-order terms of the strains, see the notes of
    the module.

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, h, c1, intx, inty, xi, eta, xp, rho, s, irho, weight
    cdef int m, n, i, j, k, l, c, p, q, r, e, f2, row, col, ptx, pty
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
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double pA[24]
    cdef double pB[24]
    cdef double dv[24]
    cdef double K[576]
    cdef double TA[120]
    cdef double eps[13]
    cdef double epsL[13]
    cdef double sig[13]
    cdef double sigL[13]

    cdef long [::1] kr, kc
    cdef double [::1] kv
    cdef double F[169]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 13, 13):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (13, 13):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(13):
            for j in range(13):
                F[i*13 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')
    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    R = shell.r
    m = shell.m
    n = shell.n
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    if not R > a:
        raise ValueError('the outer radius r must be larger than the radial '
                         'width a, got r={0} and a={1}'.format(R, a))
    #NOTE physical limits of the integration domain, resolved and validated
    #     by Shell.integration_limits()
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
    xi1 = x1/a*2 - 1
    xi2 = x2/a*2 - 1
    eta1 = y1/b*2 - 1
    eta2 = y2/b*2 - 1
    intx = x2 - x1
    inty = y2 - y1
    ftab = np.zeros((15, m), dtype=DOUBLE)
    gtab = np.zeros((15, n), dtype=DOUBLE)

    fdim = 25*m*m*n*n
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
                    for i in range(13):
                        for j in range(13):
                            F[i*13 + j] = Fnxny[ptx, pty, i, j]
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
                    ftab[9, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[10, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[11, i] = (2/a)*(2/a)*fpp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[12, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[13, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[14, i] = (2/a)*(2/a)*fpp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
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
                    gtab[9, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[10, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[11, j] = (2/b)*(2/b)*fpp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[12, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[13, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[14, j] = (2/b)*(2/b)*fpp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                for p in range(24):
                    dv[p] = 0.
                for j in range(n):
                    for i in range(m):
                        col = col0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[0, j]
                        pA[4] = ftab[0, i]*gtab[2, j]
                        pA[5] = ftab[1, i]*gtab[1, j]
                        pA[6] = ftab[3, i]*gtab[3, j]
                        pA[7] = ftab[4, i]*gtab[3, j]
                        pA[8] = ftab[3, i]*gtab[4, j]
                        pA[9] = ftab[5, i]*gtab[3, j]
                        pA[10] = ftab[3, i]*gtab[5, j]
                        pA[11] = ftab[4, i]*gtab[4, j]
                        pA[12] = ftab[6, i]*gtab[6, j]
                        pA[13] = ftab[7, i]*gtab[6, j]
                        pA[14] = ftab[6, i]*gtab[7, j]
                        pA[15] = ftab[8, i]*gtab[6, j]
                        pA[16] = ftab[6, i]*gtab[8, j]
                        pA[17] = ftab[7, i]*gtab[7, j]
                        pA[18] = ftab[9, i]*gtab[9, j]
                        pA[19] = ftab[10, i]*gtab[9, j]
                        pA[20] = ftab[9, i]*gtab[10, j]
                        pA[21] = ftab[12, i]*gtab[12, j]
                        pA[22] = ftab[13, i]*gtab[12, j]
                        pA[23] = ftab[12, i]*gtab[13, j]
                        for p in range(24):
                            dv[p] += pA[p]*cs[col + FLD[p]]
                epsL[0] = dv[1]
                epsL[1] = -dv[0]*irho + dv[8]*s
                epsL[2] = dv[2]*s + dv[6]*irho + dv[7]
                epsL[3] = dv[19]
                epsL[4] = -dv[18]*irho + dv[23]*s
                epsL[5] = dv[20]*s + dv[21]*irho + dv[22]
                epsL[6] = -c1*dv[15] - c1*dv[19]
                epsL[7] = c1*dv[13]*irho - c1*dv[16]*(s)*(s) + c1*dv[18]*irho - c1*dv[23]*s
                epsL[8] = -2.*c1*dv[14]*irho*s - 2.*c1*dv[17]*s - c1*dv[20]*s - c1*dv[21]*irho - c1*dv[22]
                epsL[9] = dv[14]*s + dv[21]
                epsL[10] = dv[13] + dv[18]
                epsL[11] = -3.*c1*dv[14]*s - 3.*c1*dv[21]
                epsL[12] = -3.*c1*dv[13] - 3.*c1*dv[18]
                for r in range(13):
                    sigL[r] = 0.
                    for q in range(13):
                        sigL[r] += F[r*13 + q]*epsL[q]
                sigL[0] = sigL[0] + Nxx0
                sigL[1] = sigL[1] + Nyy0
                sigL[2] = sigL[2] + Nxy0
                for p in range(576):
                    K[p] = 0.
                K[0] += (irho)*(irho)*sigL[1]
                K[7] += -irho*sigL[2]
                K[8] += -irho*s*sigL[1]
                K[13] += -2.*c1*(irho)*(irho)*sigL[7]
                K[15] += c1*irho*sigL[6]
                K[16] += c1*irho*(s)*(s)*sigL[7]
                K[17] += 2.*c1*irho*s*sigL[8]
                K[18] += -c1*(irho)*(irho)*sigL[7] + (irho)*(irho)*sigL[4]
                K[21] += 3.*c1*irho*sigL[11] - irho*sigL[9]
                K[22] += c1*irho*sigL[8] - irho*sigL[5]
                K[23] += c1*irho*s*sigL[7] - irho*s*sigL[4]
                K[25] += sigL[0]
                K[26] += s*sigL[2]
                K[30] += irho*sigL[2]
                K[38] += -4.*c1*irho*s*sigL[8]
                K[39] += -c1*sigL[6]
                K[40] += -c1*(s)*(s)*sigL[7]
                K[41] += -2.*c1*s*sigL[8]
                K[42] += -3.*c1*sigL[12] + sigL[10]
                K[43] += -c1*sigL[6] + sigL[3]
                K[44] += -c1*s*sigL[8] + s*sigL[5]
                K[45] += -c1*irho*sigL[8] + irho*sigL[5]
                K[49] += s*sigL[2]
                K[50] += (s)*(s)*sigL[1]
                K[54] += irho*s*sigL[1]
                K[61] += 2.*c1*irho*s*sigL[8]
                K[62] += -2.*c1*irho*(s)*(s)*sigL[7]
                K[66] += -3.*c1*s*sigL[11] + s*sigL[9]
                K[67] += -c1*s*sigL[8] + s*sigL[5]
                K[68] += -c1*(s)*(s)*sigL[7] + (s)*(s)*sigL[4]
                K[69] += -c1*irho*s*sigL[7] + irho*s*sigL[4]
                K[85] += c1*sigL[6]
                K[109] += c1*(s)*(s)*sigL[7]
                K[133] += 2.*c1*s*sigL[8]
                K[145] += irho*sigL[2]
                K[146] += irho*s*sigL[1]
                K[150] += (irho)*(irho)*sigL[1]
                K[157] += 2.*c1*(irho)*(irho)*sigL[8]
                K[158] += -c1*(irho)*(irho)*s*sigL[7]
                K[162] += -3.*c1*irho*sigL[11] + irho*sigL[9]
                K[163] += -c1*irho*sigL[8] + irho*sigL[5]
                K[164] += -c1*irho*s*sigL[7] + irho*s*sigL[4]
                K[165] += -c1*(irho)*(irho)*sigL[7] + (irho)*(irho)*sigL[4]
                K[168] += -irho*sigL[2]
                K[175] += sigL[0]
                K[176] += s*sigL[2]
                K[181] += 2.*c1*irho*sigL[8]
                K[182] += -c1*irho*s*sigL[7]
                K[186] += c1*irho*sigL[8] - irho*sigL[5]
                K[189] += -3.*c1*sigL[12] + sigL[10]
                K[190] += -c1*sigL[6] + sigL[3]
                K[191] += -c1*s*sigL[8] + s*sigL[5]
                K[192] += -irho*s*sigL[1]
                K[199] += s*sigL[2]
                K[200] += (s)*(s)*sigL[1]
                K[205] += 3.*c1*irho*s*sigL[7]
                K[207] += -c1*s*sigL[6]
                K[208] += -c1*(s)*(s)*(s)*sigL[7]
                K[209] += -2.*c1*(s)*(s)*sigL[8]
                K[210] += c1*irho*s*sigL[7] - irho*s*sigL[4]
                K[213] += -3.*c1*s*sigL[11] + s*sigL[9]
                K[214] += -c1*s*sigL[8] + s*sigL[5]
                K[215] += -c1*(s)*(s)*sigL[7] + (s)*(s)*sigL[4]
                K[230] += c1*s*sigL[6]
                K[254] += c1*(s)*(s)*(s)*sigL[7]
                K[278] += 2.*c1*(s)*(s)*sigL[8]
                K[312] += -2.*c1*(irho)*(irho)*sigL[7]
                K[314] += 2.*c1*irho*s*sigL[8]
                K[315] += c1*sigL[6]
                K[316] += c1*(s)*(s)*sigL[7]
                K[317] += 2.*c1*s*sigL[8]
                K[318] += 2.*c1*(irho)*(irho)*sigL[8]
                K[319] += 2.*c1*irho*sigL[8]
                K[320] += 3.*c1*irho*s*sigL[7]
                K[325] += sigL[0]
                K[326] += s*sigL[2]
                K[337] += -4.*c1*irho*s*sigL[8]
                K[338] += -2.*c1*irho*(s)*(s)*sigL[7]
                K[342] += -c1*(irho)*(irho)*s*sigL[7]
                K[343] += -c1*irho*s*sigL[7]
                K[345] += c1*s*sigL[6]
                K[346] += c1*(s)*(s)*(s)*sigL[7]
                K[347] += 2.*c1*(s)*(s)*sigL[8]
                K[349] += s*sigL[2]
                K[350] += (s)*(s)*sigL[1]
                K[360] += c1*irho*sigL[6]
                K[361] += -c1*sigL[6]
                K[368] += -c1*s*sigL[6]
                K[384] += c1*irho*(s)*(s)*sigL[7]
                K[385] += -c1*(s)*(s)*sigL[7]
                K[392] += -c1*(s)*(s)*(s)*sigL[7]
                K[408] += 2.*c1*irho*s*sigL[8]
                K[409] += -2.*c1*s*sigL[8]
                K[416] += -2.*c1*(s)*(s)*sigL[8]
                K[432] += -c1*(irho)*(irho)*sigL[7] + (irho)*(irho)*sigL[4]
                K[433] += -3.*c1*sigL[12] + sigL[10]
                K[434] += -3.*c1*s*sigL[11] + s*sigL[9]
                K[438] += -3.*c1*irho*sigL[11] + irho*sigL[9]
                K[439] += c1*irho*sigL[8] - irho*sigL[5]
                K[440] += c1*irho*s*sigL[7] - irho*s*sigL[4]
                K[457] += -c1*sigL[6] + sigL[3]
                K[458] += -c1*s*sigL[8] + s*sigL[5]
                K[462] += -c1*irho*sigL[8] + irho*sigL[5]
                K[481] += -c1*s*sigL[8] + s*sigL[5]
                K[482] += -c1*(s)*(s)*sigL[7] + (s)*(s)*sigL[4]
                K[486] += -c1*irho*s*sigL[7] + irho*s*sigL[4]
                K[504] += 3.*c1*irho*sigL[11] - irho*sigL[9]
                K[505] += -c1*irho*sigL[8] + irho*sigL[5]
                K[506] += -c1*irho*s*sigL[7] + irho*s*sigL[4]
                K[510] += -c1*(irho)*(irho)*sigL[7] + (irho)*(irho)*sigL[4]
                K[511] += -3.*c1*sigL[12] + sigL[10]
                K[512] += -3.*c1*s*sigL[11] + s*sigL[9]
                K[528] += c1*irho*sigL[8] - irho*sigL[5]
                K[535] += -c1*sigL[6] + sigL[3]
                K[536] += -c1*s*sigL[8] + s*sigL[5]
                K[552] += c1*irho*s*sigL[7] - irho*s*sigL[4]
                K[559] += -c1*s*sigL[8] + s*sigL[5]
                K[560] += -c1*(s)*(s)*sigL[7] + (s)*(s)*sigL[4]
                c = -1
                for j in range(n):
                    for i in range(m):
                        row = row0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[0, j]
                        pA[4] = ftab[0, i]*gtab[2, j]
                        pA[5] = ftab[1, i]*gtab[1, j]
                        pA[6] = ftab[3, i]*gtab[3, j]
                        pA[7] = ftab[4, i]*gtab[3, j]
                        pA[8] = ftab[3, i]*gtab[4, j]
                        pA[9] = ftab[5, i]*gtab[3, j]
                        pA[10] = ftab[3, i]*gtab[5, j]
                        pA[11] = ftab[4, i]*gtab[4, j]
                        pA[12] = ftab[6, i]*gtab[6, j]
                        pA[13] = ftab[7, i]*gtab[6, j]
                        pA[14] = ftab[6, i]*gtab[7, j]
                        pA[15] = ftab[8, i]*gtab[6, j]
                        pA[16] = ftab[6, i]*gtab[8, j]
                        pA[17] = ftab[7, i]*gtab[7, j]
                        pA[18] = ftab[9, i]*gtab[9, j]
                        pA[19] = ftab[10, i]*gtab[9, j]
                        pA[20] = ftab[9, i]*gtab[10, j]
                        pA[21] = ftab[12, i]*gtab[12, j]
                        pA[22] = ftab[13, i]*gtab[12, j]
                        pA[23] = ftab[12, i]*gtab[13, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(5):
                            for q in range(24):
                                TA[e*24 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*24 + q] += pA[p]*K[p*24 + q]
                        for l in range(n):
                            for k in range(m):
                                col = col0 + 5*(l*m + k)
                                #NOTE upper triangle, the caller symmetrizes
                                if row > col:
                                    continue
                                pB[0] = ftab[0, k]*gtab[0, l]
                                pB[1] = ftab[1, k]*gtab[0, l]
                                pB[2] = ftab[0, k]*gtab[1, l]
                                pB[3] = ftab[2, k]*gtab[0, l]
                                pB[4] = ftab[0, k]*gtab[2, l]
                                pB[5] = ftab[1, k]*gtab[1, l]
                                pB[6] = ftab[3, k]*gtab[3, l]
                                pB[7] = ftab[4, k]*gtab[3, l]
                                pB[8] = ftab[3, k]*gtab[4, l]
                                pB[9] = ftab[5, k]*gtab[3, l]
                                pB[10] = ftab[3, k]*gtab[5, l]
                                pB[11] = ftab[4, k]*gtab[4, l]
                                pB[12] = ftab[6, k]*gtab[6, l]
                                pB[13] = ftab[7, k]*gtab[6, l]
                                pB[14] = ftab[6, k]*gtab[7, l]
                                pB[15] = ftab[8, k]*gtab[6, l]
                                pB[16] = ftab[6, k]*gtab[8, l]
                                pB[17] = ftab[7, k]*gtab[7, l]
                                pB[18] = ftab[9, k]*gtab[9, l]
                                pB[19] = ftab[10, k]*gtab[9, l]
                                pB[20] = ftab[9, k]*gtab[10, l]
                                pB[21] = ftab[12, k]*gtab[12, l]
                                pB[22] = ftab[13, k]*gtab[12, l]
                                pB[23] = ftab[12, k]*gtab[13, l]
                                for e in range(5):
                                    for f2 in range(5):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*24 + q]*pB[q]

    return coo_matrix((kv, (kr, kc)), shape=(size, size))

def fkM_num(object shell, object I_input, int size,
        int row0, int col0, int nx, int ny):
    r"""Mass matrix, integrated numerically

    Integrates numerically the consistent mass matrix of the panel, with
    the moments of the density given for the whole domain or at each
    integration point.

    Parameters
    ----------
    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the
        thickness ``lam.h``, the numbers of terms ``m`` and ``n`` and the
        boundary flags of the approximation functions of each field.
    I_input : array-like
        Moments of the density about the mid-surface of the laminate and
        its thickness, ``J0, J1, J2, J3, J4, J6, h``, shape ``(7,)`` for the
        whole domain or ``(nx, ny, 7)`` with one set per integration point,
        see :meth:`.Shell.calc_kM`.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this panel start.
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
        radial width ``a``.

    Notes
    -----
    With the displacement field through the thickness of the module,
    `\{u(z), v(z), w(z)\}^\top = \boldsymbol{J}(z) \boldsymbol{d}`, and the
    density `\rho_m`:

    .. math::

        \boldsymbol{M} = \int \boldsymbol{\Phi}^\top \left(
        \int_{d - h/2}^{d + h/2} \rho_m \boldsymbol{J}(z)^\top
        \boldsymbol{J}(z) \, dz \right) \boldsymbol{\Phi} \, dA

    with `dA = (\rho/r)\, dx\, dy`. In `\zeta = z - d` the field has the
    powers `\zeta^0`, `\zeta` and `\zeta^3`: the integral through the
    thickness is written with the moments `J_n = \int \rho_m \zeta^n
    d\zeta`, `n = 0, 1, 2, 3, 4, 6`, the offset `d` (``shell.offset``) and
    `c_1 = 4/(3 h^2)`.

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, h, c1, intx, inty, xi, eta, xp, rho, s, irho, weight
    cdef int m, n, i, j, k, l, c, p, q, r, e, f2, row, col, ptx, pty
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
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double pA[24]
    cdef double pB[24]
    cdef double dv[24]
    cdef double K[576]
    cdef double TA[120]
    cdef double J0, J1, J2, J3, J4, J6, d
    cdef double [:, :, ::1] Inxny
    cdef int one_I_each_point = 0

    cdef long [::1] kr, kc
    cdef double [::1] kv
    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    R = shell.r
    m = shell.m
    n = shell.n
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    if not R > a:
        raise ValueError('the outer radius r must be larger than the radial '
                         'width a, got r={0} and a={1}'.format(R, a))
    #NOTE physical limits of the integration domain, resolved and validated
    #     by Shell.integration_limits()
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
    xi1 = x1/a*2 - 1
    xi2 = x2/a*2 - 1
    eta1 = y1/b*2 - 1
    eta2 = y2/b*2 - 1
    intx = x2 - x1
    inty = y2 - y1
    I_input = np.ascontiguousarray(I_input, dtype=DOUBLE)
    if I_input.shape == (nx, ny, 7):
        Inxny = I_input
        one_I_each_point = 1
    elif I_input.shape == (7,):
        Inxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        J0, J1, J2, J3, J4, J6, h = I_input
    else:
        raise ValueError('Invalid shape for I_input!')
    d = shell.offset
    ftab = np.zeros((15, m), dtype=DOUBLE)
    gtab = np.zeros((15, n), dtype=DOUBLE)

    fdim = 25*m*m*n*n
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
                    J0 = Inxny[ptx, pty, 0]
                    J1 = Inxny[ptx, pty, 1]
                    J2 = Inxny[ptx, pty, 2]
                    J3 = Inxny[ptx, pty, 3]
                    J4 = Inxny[ptx, pty, 4]
                    J6 = Inxny[ptx, pty, 5]
                    h = Inxny[ptx, pty, 6]
                c1 = 4./(3.*h*h)
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
                    ftab[9, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[10, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[11, i] = (2/a)*(2/a)*fpp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[12, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[13, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[14, i] = (2/a)*(2/a)*fpp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
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
                    gtab[9, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[10, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[11, j] = (2/b)*(2/b)*fpp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[12, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[13, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[14, j] = (2/b)*(2/b)*fpp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                for p in range(576):
                    K[p] = 0.
                K[0] = J0
                K[13] = -J0*c1*(d)*(d)*(d) - J3*c1
                K[18] = -J0*c1*(d)*(d)*(d) + J0*d + J1 - J3*c1
                K[150] = J0
                K[158] = -J0*c1*(d)*(d)*(d)*s - J3*c1*s
                K[165] = -J0*c1*(d)*(d)*(d) + J0*d + J1 - J3*c1
                K[300] = J0
                K[312] = -J0*c1*(d)*(d)*(d) - J3*c1
                K[325] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d) + 2.*J3*(c1)*(c1)*(d)*(d)*(d) + J6*(c1)*(c1)
                K[330] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d) - J0*c1*(d)*(d)*(d)*(d) - J1*c1*(d)*(d)*(d) + 2.*J3*(c1)*(c1)*(d)*(d)*(d) - J3*c1*d - J4*c1 + J6*(c1)*(c1)
                K[342] = -J0*c1*(d)*(d)*(d)*s - J3*c1*s
                K[350] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d)*(s)*(s) + 2.*J3*(c1)*(c1)*(d)*(d)*(d)*(s)*(s) + J6*(c1)*(c1)*(s)*(s)
                K[357] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d)*s - J0*c1*(d)*(d)*(d)*(d)*s - J1*c1*(d)*(d)*(d)*s + 2.*J3*(c1)*(c1)*(d)*(d)*(d)*s - J3*c1*d*s - J4*c1*s + J6*(c1)*(c1)*s
                K[432] = -J0*c1*(d)*(d)*(d) + J0*d + J1 - J3*c1
                K[445] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d) - J0*c1*(d)*(d)*(d)*(d) - J1*c1*(d)*(d)*(d) + 2.*J3*(c1)*(c1)*(d)*(d)*(d) - J3*c1*d - J4*c1 + J6*(c1)*(c1)
                K[450] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d) - 2.*J0*c1*(d)*(d)*(d)*(d) + J0*(d)*(d) - 2.*J1*c1*(d)*(d)*(d) + 2.*J1*d + J2 + 2.*J3*(c1)*(c1)*(d)*(d)*(d) - 2.*J3*c1*d - 2.*J4*c1 + J6*(c1)*(c1)
                K[510] = -J0*c1*(d)*(d)*(d) + J0*d + J1 - J3*c1
                K[518] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d)*s - J0*c1*(d)*(d)*(d)*(d)*s - J1*c1*(d)*(d)*(d)*s + 2.*J3*(c1)*(c1)*(d)*(d)*(d)*s - J3*c1*d*s - J4*c1*s + J6*(c1)*(c1)*s
                K[525] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d) - 2.*J0*c1*(d)*(d)*(d)*(d) + J0*(d)*(d) - 2.*J1*c1*(d)*(d)*(d) + 2.*J1*d + J2 + 2.*J3*(c1)*(c1)*(d)*(d)*(d) - 2.*J3*c1*d - 2.*J4*c1 + J6*(c1)*(c1)
                c = -1
                for j in range(n):
                    for i in range(m):
                        row = row0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[0, j]
                        pA[4] = ftab[0, i]*gtab[2, j]
                        pA[5] = ftab[1, i]*gtab[1, j]
                        pA[6] = ftab[3, i]*gtab[3, j]
                        pA[7] = ftab[4, i]*gtab[3, j]
                        pA[8] = ftab[3, i]*gtab[4, j]
                        pA[9] = ftab[5, i]*gtab[3, j]
                        pA[10] = ftab[3, i]*gtab[5, j]
                        pA[11] = ftab[4, i]*gtab[4, j]
                        pA[12] = ftab[6, i]*gtab[6, j]
                        pA[13] = ftab[7, i]*gtab[6, j]
                        pA[14] = ftab[6, i]*gtab[7, j]
                        pA[15] = ftab[8, i]*gtab[6, j]
                        pA[16] = ftab[6, i]*gtab[8, j]
                        pA[17] = ftab[7, i]*gtab[7, j]
                        pA[18] = ftab[9, i]*gtab[9, j]
                        pA[19] = ftab[10, i]*gtab[9, j]
                        pA[20] = ftab[9, i]*gtab[10, j]
                        pA[21] = ftab[12, i]*gtab[12, j]
                        pA[22] = ftab[13, i]*gtab[12, j]
                        pA[23] = ftab[12, i]*gtab[13, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(5):
                            for q in range(24):
                                TA[e*24 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*24 + q] += pA[p]*K[p*24 + q]
                        for l in range(n):
                            for k in range(m):
                                col = col0 + 5*(l*m + k)
                                #NOTE upper triangle, the caller symmetrizes
                                if row > col:
                                    continue
                                pB[0] = ftab[0, k]*gtab[0, l]
                                pB[1] = ftab[1, k]*gtab[0, l]
                                pB[2] = ftab[0, k]*gtab[1, l]
                                pB[3] = ftab[2, k]*gtab[0, l]
                                pB[4] = ftab[0, k]*gtab[2, l]
                                pB[5] = ftab[1, k]*gtab[1, l]
                                pB[6] = ftab[3, k]*gtab[3, l]
                                pB[7] = ftab[4, k]*gtab[3, l]
                                pB[8] = ftab[3, k]*gtab[4, l]
                                pB[9] = ftab[5, k]*gtab[3, l]
                                pB[10] = ftab[3, k]*gtab[5, l]
                                pB[11] = ftab[4, k]*gtab[4, l]
                                pB[12] = ftab[6, k]*gtab[6, l]
                                pB[13] = ftab[7, k]*gtab[6, l]
                                pB[14] = ftab[6, k]*gtab[7, l]
                                pB[15] = ftab[8, k]*gtab[6, l]
                                pB[16] = ftab[6, k]*gtab[8, l]
                                pB[17] = ftab[7, k]*gtab[7, l]
                                pB[18] = ftab[9, k]*gtab[9, l]
                                pB[19] = ftab[10, k]*gtab[9, l]
                                pB[20] = ftab[9, k]*gtab[10, l]
                                pB[21] = ftab[12, k]*gtab[12, l]
                                pB[22] = ftab[13, k]*gtab[12, l]
                                pB[23] = ftab[12, k]*gtab[13, l]
                                for e in range(5):
                                    for f2 in range(5):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*24 + q]*pB[q]

    return coo_matrix((kv, (kr, kc)), shape=(size, size))

def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    r"""Internal force vector, integrated numerically

    Integrates numerically the internal force vector of the configuration
    ``cs``, the gradient of the strain energy with respect to the Ritz
    constants.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements and rotations.
    Finput : array-like
        Constitutive matrix of the TSDT, shape ``(13, 13)`` for the
        whole domain, e.g. ``shell.ABD``, or ``(nx, ny, 13, 13)`` with
        one matrix per integration point, rows and columns in the order of
        the strains of the module.
    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the
        thickness ``lam.h``, the numbers of terms ``m`` and ``n`` and the
        boundary flags of the approximation functions of each field.
    size : int
        Size of the vector, the number of degrees of freedom of the
        assembly.
    col0 : int
        Entry of the vector where the terms of this panel start.
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
        radial width ``a``.

    Notes
    -----
    With the strains `\boldsymbol{\varepsilon}(\boldsymbol{d})` of the
    module, `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{B} = \partial
    \boldsymbol{\varepsilon}/\partial \boldsymbol{d}`:

    .. math::

        \boldsymbol{f}_{int} = \int \boldsymbol{\Phi}^\top
        \boldsymbol{B}^\top \boldsymbol{\sigma} \, dA

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, h, c1, intx, inty, xi, eta, xp, rho, s, irho, weight
    cdef int m, n, i, j, k, l, c, p, q, r, e, f2, row, col, ptx, pty
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
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double pA[24]
    cdef double pB[24]
    cdef double dv[24]
    cdef double K[576]
    cdef double TA[120]
    cdef double Bm[312]
    cdef double bs[24]
    cdef double eps[13]
    cdef double sig[13]
    cdef double [::1] fint
    cdef int row0 = col0
    cdef double F[169]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 13, 13):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (13, 13):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(13):
            for j in range(13):
                F[i*13 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')
    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    R = shell.r
    m = shell.m
    n = shell.n
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    if not R > a:
        raise ValueError('the outer radius r must be larger than the radial '
                         'width a, got r={0} and a={1}'.format(R, a))
    #NOTE physical limits of the integration domain, resolved and validated
    #     by Shell.integration_limits()
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
    xi1 = x1/a*2 - 1
    xi2 = x2/a*2 - 1
    eta1 = y1/b*2 - 1
    eta2 = y2/b*2 - 1
    intx = x2 - x1
    inty = y2 - y1
    ftab = np.zeros((15, m), dtype=DOUBLE)
    gtab = np.zeros((15, n), dtype=DOUBLE)

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
                    for i in range(13):
                        for j in range(13):
                            F[i*13 + j] = Fnxny[ptx, pty, i, j]
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
                    ftab[9, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[10, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[11, i] = (2/a)*(2/a)*fpp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[12, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[13, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[14, i] = (2/a)*(2/a)*fpp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
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
                    gtab[9, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[10, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[11, j] = (2/b)*(2/b)*fpp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[12, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[13, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[14, j] = (2/b)*(2/b)*fpp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                for p in range(24):
                    dv[p] = 0.
                for j in range(n):
                    for i in range(m):
                        col = col0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[0, j]
                        pA[4] = ftab[0, i]*gtab[2, j]
                        pA[5] = ftab[1, i]*gtab[1, j]
                        pA[6] = ftab[3, i]*gtab[3, j]
                        pA[7] = ftab[4, i]*gtab[3, j]
                        pA[8] = ftab[3, i]*gtab[4, j]
                        pA[9] = ftab[5, i]*gtab[3, j]
                        pA[10] = ftab[3, i]*gtab[5, j]
                        pA[11] = ftab[4, i]*gtab[4, j]
                        pA[12] = ftab[6, i]*gtab[6, j]
                        pA[13] = ftab[7, i]*gtab[6, j]
                        pA[14] = ftab[6, i]*gtab[7, j]
                        pA[15] = ftab[8, i]*gtab[6, j]
                        pA[16] = ftab[6, i]*gtab[8, j]
                        pA[17] = ftab[7, i]*gtab[7, j]
                        pA[18] = ftab[9, i]*gtab[9, j]
                        pA[19] = ftab[10, i]*gtab[9, j]
                        pA[20] = ftab[9, i]*gtab[10, j]
                        pA[21] = ftab[12, i]*gtab[12, j]
                        pA[22] = ftab[13, i]*gtab[12, j]
                        pA[23] = ftab[12, i]*gtab[13, j]
                        for p in range(24):
                            dv[p] += pA[p]*cs[col + FLD[p]]
                eps[0] = (dv[13])*(dv[13])/2. + (dv[1])*(dv[1])/2. + dv[1] + (dv[7])*(dv[7])/2.
                eps[1] = (dv[0])*(dv[0])*(irho)*(irho)/2. - dv[0]*dv[8]*irho*s - dv[0]*irho + (dv[14])*(dv[14])*(s)*(s)/2. + (dv[2])*(dv[2])*(s)*(s)/2. + dv[2]*dv[6]*irho*s + (dv[6])*(dv[6])*(irho)*(irho)/2. + (dv[8])*(dv[8])*(s)*(s)/2. + dv[8]*s
                eps[2] = -dv[0]*dv[7]*irho + dv[13]*dv[14]*s + dv[1]*dv[2]*s + dv[1]*dv[6]*irho + dv[2]*s + dv[6]*irho + dv[7]*dv[8]*s + dv[7]
                eps[3] = -dv[13]*dv[18]*dv[19] - dv[13]*dv[21]*dv[22] + dv[19]*dv[1] + dv[19] + dv[22]*dv[7]
                eps[4] = dv[0]*dv[18]*(irho)*(irho) - dv[0]*dv[23]*irho*s - dv[14]*dv[18]*dv[20]*(s)*(s) - dv[14]*dv[21]*dv[23]*(s)*(s) - dv[18]*dv[8]*irho*s - dv[18]*irho + dv[20]*dv[2]*(s)*(s) + dv[20]*dv[6]*irho*s + dv[21]*dv[2]*irho*s + dv[21]*dv[6]*(irho)*(irho) + dv[23]*dv[8]*(s)*(s) + dv[23]*s
                eps[5] = -dv[0]*dv[22]*irho - dv[13]*dv[18]*dv[20]*s - dv[13]*dv[21]*dv[23]*s - dv[14]*dv[18]*dv[19]*s - dv[14]*dv[21]*dv[22]*s - dv[18]*dv[7]*irho + dv[19]*dv[2]*s + dv[19]*dv[6]*irho + dv[1]*dv[20]*s + dv[1]*dv[21]*irho + dv[20]*s + dv[21]*irho + dv[22]*dv[8]*s + dv[22] + dv[23]*dv[7]*s
                eps[6] = -c1*dv[0]*dv[13]*dv[3]*irho + c1*dv[0]*dv[15]*dv[1]*irho + c1*dv[0]*dv[15]*irho + c1*dv[13]*dv[18]*dv[19] + c1*dv[13]*dv[21]*dv[22] - c1*dv[13]*dv[2]*dv[9]*s + c1*dv[13]*dv[3]*dv[8]*s + c1*dv[13]*dv[3] - c1*dv[13]*dv[6]*dv[9]*irho + c1*dv[14]*dv[1]*dv[9]*s - c1*dv[14]*dv[3]*dv[7]*s + c1*dv[14]*dv[9]*s - c1*dv[15]*dv[1]*dv[8]*s - c1*dv[15]*dv[1] + c1*dv[15]*dv[2]*dv[7]*s + c1*dv[15]*dv[6]*dv[7]*irho - c1*dv[15]*dv[8]*s - c1*dv[15] - c1*dv[19]*dv[1] - c1*dv[19] - c1*dv[22]*dv[7]
                eps[7] = c1*(dv[0])*(dv[0])*dv[13]*(irho)*(irho)*(irho) - c1*dv[0]*dv[13]*dv[4]*irho*(s)*(s) - 3.*c1*dv[0]*dv[13]*dv[8]*(irho)*(irho)*s - 2.*c1*dv[0]*dv[13]*(irho)*(irho) + c1*dv[0]*dv[14]*dv[7]*(irho)*(irho)*s + c1*dv[0]*dv[16]*dv[1]*irho*(s)*(s) + c1*dv[0]*dv[16]*irho*(s)*(s) - c1*dv[0]*dv[18]*(irho)*(irho) + c1*dv[0]*dv[23]*irho*s - c1*dv[10]*dv[13]*dv[2]*(s)*(s)*(s) - c1*dv[10]*dv[13]*dv[6]*irho*(s)*(s) + c1*dv[10]*dv[14]*dv[1]*(s)*(s)*(s) + c1*dv[10]*dv[14]*(s)*(s)*(s) + 2.*c1*dv[13]*(dv[2])*(dv[2])*irho*(s)*(s) + 3.*c1*dv[13]*dv[2]*dv[6]*(irho)*(irho)*s + c1*dv[13]*dv[4]*dv[8]*(s)*(s)*(s) + c1*dv[13]*dv[4]*(s)*(s) + c1*dv[13]*(dv[6])*(dv[6])*(irho)*(irho)*(irho) + 2.*c1*dv[13]*(dv[8])*(dv[8])*irho*(s)*(s) + 3.*c1*dv[13]*dv[8]*irho*s + c1*dv[13]*irho + c1*dv[14]*dv[18]*dv[20]*(s)*(s) - 2.*c1*dv[14]*dv[1]*dv[2]*irho*(s)*(s) - c1*dv[14]*dv[1]*dv[6]*(irho)*(irho)*s + c1*dv[14]*dv[21]*dv[23]*(s)*(s) - 2.*c1*dv[14]*dv[2]*irho*(s)*(s) - c1*dv[14]*dv[4]*dv[7]*(s)*(s)*(s) - c1*dv[14]*dv[6]*(irho)*(irho)*s - 2.*c1*dv[14]*dv[7]*dv[8]*irho*(s)*(s) - c1*dv[14]*dv[7]*irho*s - c1*dv[16]*dv[1]*dv[8]*(s)*(s)*(s) - c1*dv[16]*dv[1]*(s)*(s) + c1*dv[16]*dv[2]*dv[7]*(s)*(s)*(s) + c1*dv[16]*dv[6]*dv[7]*irho*(s)*(s) - c1*dv[16]*dv[8]*(s)*(s)*(s) - c1*dv[16]*(s)*(s) + c1*dv[18]*dv[8]*irho*s + c1*dv[18]*irho - c1*dv[20]*dv[2]*(s)*(s) - c1*dv[20]*dv[6]*irho*s - c1*dv[21]*dv[2]*irho*s - c1*dv[21]*dv[6]*(irho)*(irho) - c1*dv[23]*dv[8]*(s)*(s) - c1*dv[23]*s
                eps[8] = -2.*c1*dv[0]*dv[13]*dv[5]*irho*s - 2.*c1*dv[0]*dv[13]*dv[7]*(irho)*(irho) + 2.*c1*dv[0]*dv[17]*dv[1]*irho*s + 2.*c1*dv[0]*dv[17]*irho*s + c1*dv[0]*dv[22]*irho - 2.*c1*dv[11]*dv[13]*dv[2]*(s)*(s) - 2.*c1*dv[11]*dv[13]*dv[6]*irho*s + 2.*c1*dv[11]*dv[14]*dv[1]*(s)*(s) + 2.*c1*dv[11]*dv[14]*(s)*(s) + c1*dv[13]*dv[18]*dv[20]*s + 2.*c1*dv[13]*dv[1]*dv[2]*irho*s + 2.*c1*dv[13]*dv[1]*dv[6]*(irho)*(irho) + c1*dv[13]*dv[21]*dv[23]*s + 2.*c1*dv[13]*dv[2]*irho*s + 2.*c1*dv[13]*dv[5]*dv[8]*(s)*(s) + 2.*c1*dv[13]*dv[5]*s + 2.*c1*dv[13]*dv[6]*(irho)*(irho) + 2.*c1*dv[13]*dv[7]*dv[8]*irho*s + 2.*c1*dv[13]*dv[7]*irho + c1*dv[14]*dv[18]*dv[19]*s - 2.*c1*dv[14]*(dv[1])*(dv[1])*irho*s - 4.*c1*dv[14]*dv[1]*irho*s + c1*dv[14]*dv[21]*dv[22]*s - 2.*c1*dv[14]*dv[5]*dv[7]*(s)*(s) - 2.*c1*dv[14]*(dv[7])*(dv[7])*irho*s - 2.*c1*dv[14]*irho*s - 2.*c1*dv[17]*dv[1]*dv[8]*(s)*(s) - 2.*c1*dv[17]*dv[1]*s + 2.*c1*dv[17]*dv[2]*dv[7]*(s)*(s) + 2.*c1*dv[17]*dv[6]*dv[7]*irho*s - 2.*c1*dv[17]*dv[8]*(s)*(s) - 2.*c1*dv[17]*s + c1*dv[18]*dv[7]*irho - c1*dv[19]*dv[2]*s - c1*dv[19]*dv[6]*irho - c1*dv[1]*dv[20]*s - c1*dv[1]*dv[21]*irho - c1*dv[20]*s - c1*dv[21]*irho - c1*dv[22]*dv[8]*s - c1*dv[22] - c1*dv[23]*dv[7]*s
                eps[9] = -dv[0]*dv[21]*irho - dv[14]*(dv[18])*(dv[18])*s/2. - dv[14]*(dv[21])*(dv[21])*s/2. + dv[14]*s + dv[18]*dv[2]*s + dv[18]*dv[6]*irho + dv[21]*dv[8]*s + dv[21]
                eps[10] = -dv[13]*(dv[18])*(dv[18])/2. - dv[13]*(dv[21])*(dv[21])/2. + dv[13] + dv[18]*dv[1] + dv[18] + dv[21]*dv[7]
                eps[11] = 3.*c1*dv[0]*dv[21]*irho + 3.*c1*dv[14]*(dv[18])*(dv[18])*s/2. + 3.*c1*dv[14]*(dv[21])*(dv[21])*s/2. - 3.*c1*dv[14]*s - 3.*c1*dv[18]*dv[2]*s - 3.*c1*dv[18]*dv[6]*irho - 3.*c1*dv[21]*dv[8]*s - 3.*c1*dv[21]
                eps[12] = 3.*c1*dv[13]*(dv[18])*(dv[18])/2. + 3.*c1*dv[13]*(dv[21])*(dv[21])/2. - 3.*c1*dv[13] - 3.*c1*dv[18]*dv[1] - 3.*c1*dv[18] - 3.*c1*dv[21]*dv[7]
                for r in range(13):
                    sig[r] = 0.
                    for q in range(13):
                        sig[r] += F[r*13 + q]*eps[q]
                for p in range(312):
                    Bm[p] = 0.
                Bm[1] = dv[1] + 1.
                Bm[7] = dv[7]
                Bm[13] = dv[13]
                Bm[24] = dv[0]*(irho)*(irho) - dv[8]*irho*s - irho
                Bm[26] = dv[2]*(s)*(s) + dv[6]*irho*s
                Bm[30] = dv[2]*irho*s + dv[6]*(irho)*(irho)
                Bm[32] = -dv[0]*irho*s + dv[8]*(s)*(s) + s
                Bm[38] = dv[14]*(s)*(s)
                Bm[48] = -dv[7]*irho
                Bm[49] = dv[2]*s + dv[6]*irho
                Bm[50] = dv[1]*s + s
                Bm[54] = dv[1]*irho + irho
                Bm[55] = -dv[0]*irho + dv[8]*s + 1.
                Bm[56] = dv[7]*s
                Bm[61] = dv[14]*s
                Bm[62] = dv[13]*s
                Bm[73] = dv[19]
                Bm[79] = dv[22]
                Bm[85] = -dv[18]*dv[19] - dv[21]*dv[22]
                Bm[90] = -dv[13]*dv[19]
                Bm[91] = -dv[13]*dv[18] + dv[1] + 1.
                Bm[93] = -dv[13]*dv[22]
                Bm[94] = -dv[13]*dv[21] + dv[7]
                Bm[96] = dv[18]*(irho)*(irho) - dv[23]*irho*s
                Bm[98] = dv[20]*(s)*(s) + dv[21]*irho*s
                Bm[102] = dv[20]*irho*s + dv[21]*(irho)*(irho)
                Bm[104] = -dv[18]*irho*s + dv[23]*(s)*(s)
                Bm[110] = -dv[18]*dv[20]*(s)*(s) - dv[21]*dv[23]*(s)*(s)
                Bm[114] = dv[0]*(irho)*(irho) - dv[14]*dv[20]*(s)*(s) - dv[8]*irho*s - irho
                Bm[116] = -dv[14]*dv[18]*(s)*(s) + dv[2]*(s)*(s) + dv[6]*irho*s
                Bm[117] = -dv[14]*dv[23]*(s)*(s) + dv[2]*irho*s + dv[6]*(irho)*(irho)
                Bm[119] = -dv[0]*irho*s - dv[14]*dv[21]*(s)*(s) + dv[8]*(s)*(s) + s
                Bm[120] = -dv[22]*irho
                Bm[121] = dv[20]*s + dv[21]*irho
                Bm[122] = dv[19]*s
                Bm[126] = dv[19]*irho
                Bm[127] = -dv[18]*irho + dv[23]*s
                Bm[128] = dv[22]*s
                Bm[133] = -dv[18]*dv[20]*s - dv[21]*dv[23]*s
                Bm[134] = -dv[18]*dv[19]*s - dv[21]*dv[22]*s
                Bm[138] = -dv[13]*dv[20]*s - dv[14]*dv[19]*s - dv[7]*irho
                Bm[139] = -dv[14]*dv[18]*s + dv[2]*s + dv[6]*irho
                Bm[140] = -dv[13]*dv[18]*s + dv[1]*s + s
                Bm[141] = -dv[13]*dv[23]*s - dv[14]*dv[22]*s + dv[1]*irho + irho
                Bm[142] = -dv[0]*irho - dv[14]*dv[21]*s + dv[8]*s + 1.
                Bm[143] = -dv[13]*dv[21]*s + dv[7]*s
                Bm[144] = -c1*dv[13]*dv[3]*irho + c1*dv[15]*dv[1]*irho + c1*dv[15]*irho
                Bm[145] = c1*dv[0]*dv[15]*irho + c1*dv[14]*dv[9]*s - c1*dv[15]*dv[8]*s - c1*dv[15] - c1*dv[19]
                Bm[146] = -c1*dv[13]*dv[9]*s + c1*dv[15]*dv[7]*s
                Bm[147] = -c1*dv[0]*dv[13]*irho + c1*dv[13]*dv[8]*s + c1*dv[13] - c1*dv[14]*dv[7]*s
                Bm[150] = -c1*dv[13]*dv[9]*irho + c1*dv[15]*dv[7]*irho
                Bm[151] = -c1*dv[14]*dv[3]*s + c1*dv[15]*dv[2]*s + c1*dv[15]*dv[6]*irho - c1*dv[22]
                Bm[152] = c1*dv[13]*dv[3]*s - c1*dv[15]*dv[1]*s - c1*dv[15]*s
                Bm[153] = -c1*dv[13]*dv[2]*s - c1*dv[13]*dv[6]*irho + c1*dv[14]*dv[1]*s + c1*dv[14]*s
                Bm[157] = -c1*dv[0]*dv[3]*irho + c1*dv[18]*dv[19] + c1*dv[21]*dv[22] - c1*dv[2]*dv[9]*s + c1*dv[3]*dv[8]*s + c1*dv[3] - c1*dv[6]*dv[9]*irho
                Bm[158] = c1*dv[1]*dv[9]*s - c1*dv[3]*dv[7]*s + c1*dv[9]*s
                Bm[159] = c1*dv[0]*dv[1]*irho + c1*dv[0]*irho - c1*dv[1]*dv[8]*s - c1*dv[1] + c1*dv[2]*dv[7]*s + c1*dv[6]*dv[7]*irho - c1*dv[8]*s - c1
                Bm[162] = c1*dv[13]*dv[19]
                Bm[163] = c1*dv[13]*dv[18] - c1*dv[1] - c1
                Bm[165] = c1*dv[13]*dv[22]
                Bm[166] = c1*dv[13]*dv[21] - c1*dv[7]
                Bm[168] = 2.*c1*dv[0]*dv[13]*(irho)*(irho)*(irho) - c1*dv[13]*dv[4]*irho*(s)*(s) - 3.*c1*dv[13]*dv[8]*(irho)*(irho)*s - 2.*c1*dv[13]*(irho)*(irho) + c1*dv[14]*dv[7]*(irho)*(irho)*s + c1*dv[16]*dv[1]*irho*(s)*(s) + c1*dv[16]*irho*(s)*(s) - c1*dv[18]*(irho)*(irho) + c1*dv[23]*irho*s
                Bm[169] = c1*dv[0]*dv[16]*irho*(s)*(s) + c1*dv[10]*dv[14]*(s)*(s)*(s) - 2.*c1*dv[14]*dv[2]*irho*(s)*(s) - c1*dv[14]*dv[6]*(irho)*(irho)*s - c1*dv[16]*dv[8]*(s)*(s)*(s) - c1*dv[16]*(s)*(s)
                Bm[170] = -c1*dv[10]*dv[13]*(s)*(s)*(s) + 4.*c1*dv[13]*dv[2]*irho*(s)*(s) + 3.*c1*dv[13]*dv[6]*(irho)*(irho)*s - 2.*c1*dv[14]*dv[1]*irho*(s)*(s) - 2.*c1*dv[14]*irho*(s)*(s) + c1*dv[16]*dv[7]*(s)*(s)*(s) - c1*dv[20]*(s)*(s) - c1*dv[21]*irho*s
                Bm[172] = -c1*dv[0]*dv[13]*irho*(s)*(s) + c1*dv[13]*dv[8]*(s)*(s)*(s) + c1*dv[13]*(s)*(s) - c1*dv[14]*dv[7]*(s)*(s)*(s)
                Bm[174] = -c1*dv[10]*dv[13]*irho*(s)*(s) + 3.*c1*dv[13]*dv[2]*(irho)*(irho)*s + 2.*c1*dv[13]*dv[6]*(irho)*(irho)*(irho) - c1*dv[14]*dv[1]*(irho)*(irho)*s - c1*dv[14]*(irho)*(irho)*s + c1*dv[16]*dv[7]*irho*(s)*(s) - c1*dv[20]*irho*s - c1*dv[21]*(irho)*(irho)
                Bm[175] = c1*dv[0]*dv[14]*(irho)*(irho)*s - c1*dv[14]*dv[4]*(s)*(s)*(s) - 2.*c1*dv[14]*dv[8]*irho*(s)*(s) - c1*dv[14]*irho*s + c1*dv[16]*dv[2]*(s)*(s)*(s) + c1*dv[16]*dv[6]*irho*(s)*(s)
                Bm[176] = -3.*c1*dv[0]*dv[13]*(irho)*(irho)*s + c1*dv[13]*dv[4]*(s)*(s)*(s) + 4.*c1*dv[13]*dv[8]*irho*(s)*(s) + 3.*c1*dv[13]*irho*s - 2.*c1*dv[14]*dv[7]*irho*(s)*(s) - c1*dv[16]*dv[1]*(s)*(s)*(s) - c1*dv[16]*(s)*(s)*(s) + c1*dv[18]*irho*s - c1*dv[23]*(s)*(s)
                Bm[178] = -c1*dv[13]*dv[2]*(s)*(s)*(s) - c1*dv[13]*dv[6]*irho*(s)*(s) + c1*dv[14]*dv[1]*(s)*(s)*(s) + c1*dv[14]*(s)*(s)*(s)
                Bm[181] = c1*(dv[0])*(dv[0])*(irho)*(irho)*(irho) - c1*dv[0]*dv[4]*irho*(s)*(s) - 3.*c1*dv[0]*dv[8]*(irho)*(irho)*s - 2.*c1*dv[0]*(irho)*(irho) - c1*dv[10]*dv[2]*(s)*(s)*(s) - c1*dv[10]*dv[6]*irho*(s)*(s) + 2.*c1*(dv[2])*(dv[2])*irho*(s)*(s) + 3.*c1*dv[2]*dv[6]*(irho)*(irho)*s + c1*dv[4]*dv[8]*(s)*(s)*(s) + c1*dv[4]*(s)*(s) + c1*(dv[6])*(dv[6])*(irho)*(irho)*(irho) + 2.*c1*(dv[8])*(dv[8])*irho*(s)*(s) + 3.*c1*dv[8]*irho*s + c1*irho
                Bm[182] = c1*dv[0]*dv[7]*(irho)*(irho)*s + c1*dv[10]*dv[1]*(s)*(s)*(s) + c1*dv[10]*(s)*(s)*(s) + c1*dv[18]*dv[20]*(s)*(s) - 2.*c1*dv[1]*dv[2]*irho*(s)*(s) - c1*dv[1]*dv[6]*(irho)*(irho)*s + c1*dv[21]*dv[23]*(s)*(s) - 2.*c1*dv[2]*irho*(s)*(s) - c1*dv[4]*dv[7]*(s)*(s)*(s) - c1*dv[6]*(irho)*(irho)*s - 2.*c1*dv[7]*dv[8]*irho*(s)*(s) - c1*dv[7]*irho*s
                Bm[184] = c1*dv[0]*dv[1]*irho*(s)*(s) + c1*dv[0]*irho*(s)*(s) - c1*dv[1]*dv[8]*(s)*(s)*(s) - c1*dv[1]*(s)*(s) + c1*dv[2]*dv[7]*(s)*(s)*(s) + c1*dv[6]*dv[7]*irho*(s)*(s) - c1*dv[8]*(s)*(s)*(s) - c1*(s)*(s)
                Bm[186] = -c1*dv[0]*(irho)*(irho) + c1*dv[14]*dv[20]*(s)*(s) + c1*dv[8]*irho*s + c1*irho
                Bm[188] = c1*dv[14]*dv[18]*(s)*(s) - c1*dv[2]*(s)*(s) - c1*dv[6]*irho*s
                Bm[189] = c1*dv[14]*dv[23]*(s)*(s) - c1*dv[2]*irho*s - c1*dv[6]*(irho)*(irho)
                Bm[191] = c1*dv[0]*irho*s + c1*dv[14]*dv[21]*(s)*(s) - c1*dv[8]*(s)*(s) - c1*s
                Bm[192] = -2.*c1*dv[13]*dv[5]*irho*s - 2.*c1*dv[13]*dv[7]*(irho)*(irho) + 2.*c1*dv[17]*dv[1]*irho*s + 2.*c1*dv[17]*irho*s + c1*dv[22]*irho
                Bm[193] = 2.*c1*dv[0]*dv[17]*irho*s + 2.*c1*dv[11]*dv[14]*(s)*(s) + 2.*c1*dv[13]*dv[2]*irho*s + 2.*c1*dv[13]*dv[6]*(irho)*(irho) - 4.*c1*dv[14]*dv[1]*irho*s - 4.*c1*dv[14]*irho*s - 2.*c1*dv[17]*dv[8]*(s)*(s) - 2.*c1*dv[17]*s - c1*dv[20]*s - c1*dv[21]*irho
                Bm[194] = -2.*c1*dv[11]*dv[13]*(s)*(s) + 2.*c1*dv[13]*dv[1]*irho*s + 2.*c1*dv[13]*irho*s + 2.*c1*dv[17]*dv[7]*(s)*(s) - c1*dv[19]*s
                Bm[197] = -2.*c1*dv[0]*dv[13]*irho*s + 2.*c1*dv[13]*dv[8]*(s)*(s) + 2.*c1*dv[13]*s - 2.*c1*dv[14]*dv[7]*(s)*(s)
                Bm[198] = -2.*c1*dv[11]*dv[13]*irho*s + 2.*c1*dv[13]*dv[1]*(irho)*(irho) + 2.*c1*dv[13]*(irho)*(irho) + 2.*c1*dv[17]*dv[7]*irho*s - c1*dv[19]*irho
                Bm[199] = -2.*c1*dv[0]*dv[13]*(irho)*(irho) + 2.*c1*dv[13]*dv[8]*irho*s + 2.*c1*dv[13]*irho - 2.*c1*dv[14]*dv[5]*(s)*(s) - 4.*c1*dv[14]*dv[7]*irho*s + 2.*c1*dv[17]*dv[2]*(s)*(s) + 2.*c1*dv[17]*dv[6]*irho*s + c1*dv[18]*irho - c1*dv[23]*s
                Bm[200] = 2.*c1*dv[13]*dv[5]*(s)*(s) + 2.*c1*dv[13]*dv[7]*irho*s - 2.*c1*dv[17]*dv[1]*(s)*(s) - 2.*c1*dv[17]*(s)*(s) - c1*dv[22]*s
                Bm[203] = -2.*c1*dv[13]*dv[2]*(s)*(s) - 2.*c1*dv[13]*dv[6]*irho*s + 2.*c1*dv[14]*dv[1]*(s)*(s) + 2.*c1*dv[14]*(s)*(s)
                Bm[205] = -2.*c1*dv[0]*dv[5]*irho*s - 2.*c1*dv[0]*dv[7]*(irho)*(irho) - 2.*c1*dv[11]*dv[2]*(s)*(s) - 2.*c1*dv[11]*dv[6]*irho*s + c1*dv[18]*dv[20]*s + 2.*c1*dv[1]*dv[2]*irho*s + 2.*c1*dv[1]*dv[6]*(irho)*(irho) + c1*dv[21]*dv[23]*s + 2.*c1*dv[2]*irho*s + 2.*c1*dv[5]*dv[8]*(s)*(s) + 2.*c1*dv[5]*s + 2.*c1*dv[6]*(irho)*(irho) + 2.*c1*dv[7]*dv[8]*irho*s + 2.*c1*dv[7]*irho
                Bm[206] = 2.*c1*dv[11]*dv[1]*(s)*(s) + 2.*c1*dv[11]*(s)*(s) + c1*dv[18]*dv[19]*s - 2.*c1*(dv[1])*(dv[1])*irho*s - 4.*c1*dv[1]*irho*s + c1*dv[21]*dv[22]*s - 2.*c1*dv[5]*dv[7]*(s)*(s) - 2.*c1*(dv[7])*(dv[7])*irho*s - 2.*c1*irho*s
                Bm[209] = 2.*c1*dv[0]*dv[1]*irho*s + 2.*c1*dv[0]*irho*s - 2.*c1*dv[1]*dv[8]*(s)*(s) - 2.*c1*dv[1]*s + 2.*c1*dv[2]*dv[7]*(s)*(s) + 2.*c1*dv[6]*dv[7]*irho*s - 2.*c1*dv[8]*(s)*(s) - 2.*c1*s
                Bm[210] = c1*dv[13]*dv[20]*s + c1*dv[14]*dv[19]*s + c1*dv[7]*irho
                Bm[211] = c1*dv[14]*dv[18]*s - c1*dv[2]*s - c1*dv[6]*irho
                Bm[212] = c1*dv[13]*dv[18]*s - c1*dv[1]*s - c1*s
                Bm[213] = c1*dv[13]*dv[23]*s + c1*dv[14]*dv[22]*s - c1*dv[1]*irho - c1*irho
                Bm[214] = c1*dv[0]*irho + c1*dv[14]*dv[21]*s - c1*dv[8]*s - c1
                Bm[215] = c1*dv[13]*dv[21]*s - c1*dv[7]*s
                Bm[216] = -dv[21]*irho
                Bm[218] = dv[18]*s
                Bm[222] = dv[18]*irho
                Bm[224] = dv[21]*s
                Bm[230] = -(dv[18])*(dv[18])*s/2. - (dv[21])*(dv[21])*s/2. + s
                Bm[234] = -dv[14]*dv[18]*s + dv[2]*s + dv[6]*irho
                Bm[237] = -dv[0]*irho - dv[14]*dv[21]*s + dv[8]*s + 1.
                Bm[241] = dv[18]
                Bm[247] = dv[21]
                Bm[253] = -(dv[18])*(dv[18])/2. - (dv[21])*(dv[21])/2. + 1.
                Bm[258] = -dv[13]*dv[18] + dv[1] + 1.
                Bm[261] = -dv[13]*dv[21] + dv[7]
                Bm[264] = 3.*c1*dv[21]*irho
                Bm[266] = -3.*c1*dv[18]*s
                Bm[270] = -3.*c1*dv[18]*irho
                Bm[272] = -3.*c1*dv[21]*s
                Bm[278] = 3.*c1*(dv[18])*(dv[18])*s/2. + 3.*c1*(dv[21])*(dv[21])*s/2. - 3.*c1*s
                Bm[282] = 3.*c1*dv[14]*dv[18]*s - 3.*c1*dv[2]*s - 3.*c1*dv[6]*irho
                Bm[285] = 3.*c1*dv[0]*irho + 3.*c1*dv[14]*dv[21]*s - 3.*c1*dv[8]*s - 3.*c1
                Bm[289] = -3.*c1*dv[18]
                Bm[295] = -3.*c1*dv[21]
                Bm[301] = 3.*c1*(dv[18])*(dv[18])/2. + 3.*c1*(dv[21])*(dv[21])/2. - 3.*c1
                Bm[306] = 3.*c1*dv[13]*dv[18] - 3.*c1*dv[1] - 3.*c1
                Bm[309] = 3.*c1*dv[13]*dv[21] - 3.*c1*dv[7]
                for p in range(24):
                    bs[p] = 0.
                    for r in range(13):
                        bs[p] += Bm[r*24 + p]*sig[r]
                for j in range(n):
                    for i in range(m):
                        row = row0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[0, j]
                        pA[4] = ftab[0, i]*gtab[2, j]
                        pA[5] = ftab[1, i]*gtab[1, j]
                        pA[6] = ftab[3, i]*gtab[3, j]
                        pA[7] = ftab[4, i]*gtab[3, j]
                        pA[8] = ftab[3, i]*gtab[4, j]
                        pA[9] = ftab[5, i]*gtab[3, j]
                        pA[10] = ftab[3, i]*gtab[5, j]
                        pA[11] = ftab[4, i]*gtab[4, j]
                        pA[12] = ftab[6, i]*gtab[6, j]
                        pA[13] = ftab[7, i]*gtab[6, j]
                        pA[14] = ftab[6, i]*gtab[7, j]
                        pA[15] = ftab[8, i]*gtab[6, j]
                        pA[16] = ftab[6, i]*gtab[8, j]
                        pA[17] = ftab[7, i]*gtab[7, j]
                        pA[18] = ftab[9, i]*gtab[9, j]
                        pA[19] = ftab[10, i]*gtab[9, j]
                        pA[20] = ftab[9, i]*gtab[10, j]
                        pA[21] = ftab[12, i]*gtab[12, j]
                        pA[22] = ftab[13, i]*gtab[12, j]
                        pA[23] = ftab[12, i]*gtab[13, j]
                        for e in range(5):
                            for p in range(P0[e], P1[e]):
                                fint[row + e] += weight*pA[p]*bs[p]
    return np.asarray(fint)

def calc_kTv(double [::1] cs, double [::1] vs, object Finput, object shell,
        int size, int col0, int nx, int ny, double Nxx_cte=0.,
        double Nyy_cte=0., double Nxy_cte=0.):
    r"""Product of the tangent stiffness matrix by a vector, without assembly

    Returns ``KT(cs) vs``, with ``KT`` the tangent stiffness matrix
    ``fkC_num(cs, ..., NLgeom=1) + fkG_num(cs, ...)`` at ``cs``, the exact
    Jacobian of :func:`calc_fint`, plus the geometric stiffness matrix of
    the constant membrane stress state ``Nxx_cte``, ``Nyy_cte`` and
    ``Nxy_cte``.

    Parameters
    ----------
    cs, vs : array-like
        Ritz constants of the configuration and the vector, 1-D of size
        ``size``.
    Finput : array-like
        Constitutive matrix of the TSDT, shape ``(13, 13)`` for the
        whole domain, e.g. ``shell.ABD``, or ``(nx, ny, 13, 13)`` with
        one matrix per integration point, rows and columns in the order of
        the strains of the module.
    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the
        thickness ``lam.h``, the numbers of terms ``m`` and ``n`` and the
        boundary flags of the approximation functions of each field.
    size, col0 : int
        Size of the vectors and position where the terms of this panel
        start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.
    Nxx_cte, Nyy_cte, Nxy_cte : float, optional
        Constant membrane stress resultants along the local axes, force per
        unit length, default ``0``.

    Returns
    -------
    kTv : numpy.ndarray
        The product, of size ``size``.

    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape, ``shell`` is not a
        :class:`.Shell` or its outer radius ``r`` is not larger than its
        radial width ``a``.

    Notes
    -----
    At each integration point, with `\boldsymbol{B} = \partial
    \boldsymbol{\varepsilon}/\partial \boldsymbol{d}` at `\boldsymbol{d} =
    \boldsymbol{\Phi} \boldsymbol{c}` and the stress resultants
    `\boldsymbol{\sigma}` of `\boldsymbol{F} \boldsymbol{\varepsilon}`,
    plus the constant membrane ones `\boldsymbol{\sigma}_0`:

    .. math::

        \boldsymbol{K}_T \boldsymbol{v} = \int \boldsymbol{\Phi}^\top \left(
        \boldsymbol{B}^\top \boldsymbol{F} \boldsymbol{B} +
        \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{d}) +
        \boldsymbol{H}(\boldsymbol{\sigma}_0, \boldsymbol{0}) \right)
        \boldsymbol{\Phi} \boldsymbol{v} \, dA

    evaluated without forming `\boldsymbol{K}_T`, with
    `\boldsymbol{H}` of :func:`fkC_num`.

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, h, c1, intx, inty, xi, eta, xp, rho, s, irho, weight
    cdef int m, n, i, j, k, l, c, p, q, r, e, f2, row, col, ptx, pty
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
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double pA[24]
    cdef double pB[24]
    cdef double dv[24]
    cdef double K[576]
    cdef double TA[120]
    cdef double Bm[312]
    cdef double bs[24]
    cdef double dq[24]
    cdef double Bq[13]
    cdef double FBq[13]
    cdef double eps[13]
    cdef double sig[13]
    cdef double [::1] kTv
    cdef int row0 = col0
    cdef double F[169]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 13, 13):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (13, 13):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(13):
            for j in range(13):
                F[i*13 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')
    if cs.shape[0] != size or vs.shape[0] != size:
        raise ValueError('cs and vs must have the size size')
    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    R = shell.r
    m = shell.m
    n = shell.n
    h = shell.lam.h
    c1 = 4./(3.*h*h)
    if not R > a:
        raise ValueError('the outer radius r must be larger than the radial '
                         'width a, got r={0} and a={1}'.format(R, a))
    #NOTE physical limits of the integration domain, resolved and validated
    #     by Shell.integration_limits()
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
    xi1 = x1/a*2 - 1
    xi2 = x2/a*2 - 1
    eta1 = y1/b*2 - 1
    eta2 = y2/b*2 - 1
    intx = x2 - x1
    inty = y2 - y1
    ftab = np.zeros((15, m), dtype=DOUBLE)
    gtab = np.zeros((15, n), dtype=DOUBLE)

    kTv = np.zeros(size, dtype=DOUBLE)

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
                    for i in range(13):
                        for j in range(13):
                            F[i*13 + j] = Fnxny[ptx, pty, i, j]
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
                    ftab[9, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[10, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[11, i] = (2/a)*(2/a)*fpp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
                    ftab[12, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[13, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
                    ftab[14, i] = (2/a)*(2/a)*fpp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
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
                    gtab[9, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[10, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[11, j] = (2/b)*(2/b)*fpp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
                    gtab[12, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[13, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
                    gtab[14, j] = (2/b)*(2/b)*fpp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)

                for p in range(24):
                    dq[p] = 0.
                for j in range(n):
                    for i in range(m):
                        col = col0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[0, j]
                        pA[4] = ftab[0, i]*gtab[2, j]
                        pA[5] = ftab[1, i]*gtab[1, j]
                        pA[6] = ftab[3, i]*gtab[3, j]
                        pA[7] = ftab[4, i]*gtab[3, j]
                        pA[8] = ftab[3, i]*gtab[4, j]
                        pA[9] = ftab[5, i]*gtab[3, j]
                        pA[10] = ftab[3, i]*gtab[5, j]
                        pA[11] = ftab[4, i]*gtab[4, j]
                        pA[12] = ftab[6, i]*gtab[6, j]
                        pA[13] = ftab[7, i]*gtab[6, j]
                        pA[14] = ftab[6, i]*gtab[7, j]
                        pA[15] = ftab[8, i]*gtab[6, j]
                        pA[16] = ftab[6, i]*gtab[8, j]
                        pA[17] = ftab[7, i]*gtab[7, j]
                        pA[18] = ftab[9, i]*gtab[9, j]
                        pA[19] = ftab[10, i]*gtab[9, j]
                        pA[20] = ftab[9, i]*gtab[10, j]
                        pA[21] = ftab[12, i]*gtab[12, j]
                        pA[22] = ftab[13, i]*gtab[12, j]
                        pA[23] = ftab[12, i]*gtab[13, j]
                        for p in range(24):
                            dq[p] += pA[p]*vs[col + FLD[p]]
                for p in range(24):
                    dv[p] = 0.
                for j in range(n):
                    for i in range(m):
                        col = col0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[0, j]
                        pA[4] = ftab[0, i]*gtab[2, j]
                        pA[5] = ftab[1, i]*gtab[1, j]
                        pA[6] = ftab[3, i]*gtab[3, j]
                        pA[7] = ftab[4, i]*gtab[3, j]
                        pA[8] = ftab[3, i]*gtab[4, j]
                        pA[9] = ftab[5, i]*gtab[3, j]
                        pA[10] = ftab[3, i]*gtab[5, j]
                        pA[11] = ftab[4, i]*gtab[4, j]
                        pA[12] = ftab[6, i]*gtab[6, j]
                        pA[13] = ftab[7, i]*gtab[6, j]
                        pA[14] = ftab[6, i]*gtab[7, j]
                        pA[15] = ftab[8, i]*gtab[6, j]
                        pA[16] = ftab[6, i]*gtab[8, j]
                        pA[17] = ftab[7, i]*gtab[7, j]
                        pA[18] = ftab[9, i]*gtab[9, j]
                        pA[19] = ftab[10, i]*gtab[9, j]
                        pA[20] = ftab[9, i]*gtab[10, j]
                        pA[21] = ftab[12, i]*gtab[12, j]
                        pA[22] = ftab[13, i]*gtab[12, j]
                        pA[23] = ftab[12, i]*gtab[13, j]
                        for p in range(24):
                            dv[p] += pA[p]*cs[col + FLD[p]]
                eps[0] = (dv[13])*(dv[13])/2. + (dv[1])*(dv[1])/2. + dv[1] + (dv[7])*(dv[7])/2.
                eps[1] = (dv[0])*(dv[0])*(irho)*(irho)/2. - dv[0]*dv[8]*irho*s - dv[0]*irho + (dv[14])*(dv[14])*(s)*(s)/2. + (dv[2])*(dv[2])*(s)*(s)/2. + dv[2]*dv[6]*irho*s + (dv[6])*(dv[6])*(irho)*(irho)/2. + (dv[8])*(dv[8])*(s)*(s)/2. + dv[8]*s
                eps[2] = -dv[0]*dv[7]*irho + dv[13]*dv[14]*s + dv[1]*dv[2]*s + dv[1]*dv[6]*irho + dv[2]*s + dv[6]*irho + dv[7]*dv[8]*s + dv[7]
                eps[3] = -dv[13]*dv[18]*dv[19] - dv[13]*dv[21]*dv[22] + dv[19]*dv[1] + dv[19] + dv[22]*dv[7]
                eps[4] = dv[0]*dv[18]*(irho)*(irho) - dv[0]*dv[23]*irho*s - dv[14]*dv[18]*dv[20]*(s)*(s) - dv[14]*dv[21]*dv[23]*(s)*(s) - dv[18]*dv[8]*irho*s - dv[18]*irho + dv[20]*dv[2]*(s)*(s) + dv[20]*dv[6]*irho*s + dv[21]*dv[2]*irho*s + dv[21]*dv[6]*(irho)*(irho) + dv[23]*dv[8]*(s)*(s) + dv[23]*s
                eps[5] = -dv[0]*dv[22]*irho - dv[13]*dv[18]*dv[20]*s - dv[13]*dv[21]*dv[23]*s - dv[14]*dv[18]*dv[19]*s - dv[14]*dv[21]*dv[22]*s - dv[18]*dv[7]*irho + dv[19]*dv[2]*s + dv[19]*dv[6]*irho + dv[1]*dv[20]*s + dv[1]*dv[21]*irho + dv[20]*s + dv[21]*irho + dv[22]*dv[8]*s + dv[22] + dv[23]*dv[7]*s
                eps[6] = -c1*dv[0]*dv[13]*dv[3]*irho + c1*dv[0]*dv[15]*dv[1]*irho + c1*dv[0]*dv[15]*irho + c1*dv[13]*dv[18]*dv[19] + c1*dv[13]*dv[21]*dv[22] - c1*dv[13]*dv[2]*dv[9]*s + c1*dv[13]*dv[3]*dv[8]*s + c1*dv[13]*dv[3] - c1*dv[13]*dv[6]*dv[9]*irho + c1*dv[14]*dv[1]*dv[9]*s - c1*dv[14]*dv[3]*dv[7]*s + c1*dv[14]*dv[9]*s - c1*dv[15]*dv[1]*dv[8]*s - c1*dv[15]*dv[1] + c1*dv[15]*dv[2]*dv[7]*s + c1*dv[15]*dv[6]*dv[7]*irho - c1*dv[15]*dv[8]*s - c1*dv[15] - c1*dv[19]*dv[1] - c1*dv[19] - c1*dv[22]*dv[7]
                eps[7] = c1*(dv[0])*(dv[0])*dv[13]*(irho)*(irho)*(irho) - c1*dv[0]*dv[13]*dv[4]*irho*(s)*(s) - 3.*c1*dv[0]*dv[13]*dv[8]*(irho)*(irho)*s - 2.*c1*dv[0]*dv[13]*(irho)*(irho) + c1*dv[0]*dv[14]*dv[7]*(irho)*(irho)*s + c1*dv[0]*dv[16]*dv[1]*irho*(s)*(s) + c1*dv[0]*dv[16]*irho*(s)*(s) - c1*dv[0]*dv[18]*(irho)*(irho) + c1*dv[0]*dv[23]*irho*s - c1*dv[10]*dv[13]*dv[2]*(s)*(s)*(s) - c1*dv[10]*dv[13]*dv[6]*irho*(s)*(s) + c1*dv[10]*dv[14]*dv[1]*(s)*(s)*(s) + c1*dv[10]*dv[14]*(s)*(s)*(s) + 2.*c1*dv[13]*(dv[2])*(dv[2])*irho*(s)*(s) + 3.*c1*dv[13]*dv[2]*dv[6]*(irho)*(irho)*s + c1*dv[13]*dv[4]*dv[8]*(s)*(s)*(s) + c1*dv[13]*dv[4]*(s)*(s) + c1*dv[13]*(dv[6])*(dv[6])*(irho)*(irho)*(irho) + 2.*c1*dv[13]*(dv[8])*(dv[8])*irho*(s)*(s) + 3.*c1*dv[13]*dv[8]*irho*s + c1*dv[13]*irho + c1*dv[14]*dv[18]*dv[20]*(s)*(s) - 2.*c1*dv[14]*dv[1]*dv[2]*irho*(s)*(s) - c1*dv[14]*dv[1]*dv[6]*(irho)*(irho)*s + c1*dv[14]*dv[21]*dv[23]*(s)*(s) - 2.*c1*dv[14]*dv[2]*irho*(s)*(s) - c1*dv[14]*dv[4]*dv[7]*(s)*(s)*(s) - c1*dv[14]*dv[6]*(irho)*(irho)*s - 2.*c1*dv[14]*dv[7]*dv[8]*irho*(s)*(s) - c1*dv[14]*dv[7]*irho*s - c1*dv[16]*dv[1]*dv[8]*(s)*(s)*(s) - c1*dv[16]*dv[1]*(s)*(s) + c1*dv[16]*dv[2]*dv[7]*(s)*(s)*(s) + c1*dv[16]*dv[6]*dv[7]*irho*(s)*(s) - c1*dv[16]*dv[8]*(s)*(s)*(s) - c1*dv[16]*(s)*(s) + c1*dv[18]*dv[8]*irho*s + c1*dv[18]*irho - c1*dv[20]*dv[2]*(s)*(s) - c1*dv[20]*dv[6]*irho*s - c1*dv[21]*dv[2]*irho*s - c1*dv[21]*dv[6]*(irho)*(irho) - c1*dv[23]*dv[8]*(s)*(s) - c1*dv[23]*s
                eps[8] = -2.*c1*dv[0]*dv[13]*dv[5]*irho*s - 2.*c1*dv[0]*dv[13]*dv[7]*(irho)*(irho) + 2.*c1*dv[0]*dv[17]*dv[1]*irho*s + 2.*c1*dv[0]*dv[17]*irho*s + c1*dv[0]*dv[22]*irho - 2.*c1*dv[11]*dv[13]*dv[2]*(s)*(s) - 2.*c1*dv[11]*dv[13]*dv[6]*irho*s + 2.*c1*dv[11]*dv[14]*dv[1]*(s)*(s) + 2.*c1*dv[11]*dv[14]*(s)*(s) + c1*dv[13]*dv[18]*dv[20]*s + 2.*c1*dv[13]*dv[1]*dv[2]*irho*s + 2.*c1*dv[13]*dv[1]*dv[6]*(irho)*(irho) + c1*dv[13]*dv[21]*dv[23]*s + 2.*c1*dv[13]*dv[2]*irho*s + 2.*c1*dv[13]*dv[5]*dv[8]*(s)*(s) + 2.*c1*dv[13]*dv[5]*s + 2.*c1*dv[13]*dv[6]*(irho)*(irho) + 2.*c1*dv[13]*dv[7]*dv[8]*irho*s + 2.*c1*dv[13]*dv[7]*irho + c1*dv[14]*dv[18]*dv[19]*s - 2.*c1*dv[14]*(dv[1])*(dv[1])*irho*s - 4.*c1*dv[14]*dv[1]*irho*s + c1*dv[14]*dv[21]*dv[22]*s - 2.*c1*dv[14]*dv[5]*dv[7]*(s)*(s) - 2.*c1*dv[14]*(dv[7])*(dv[7])*irho*s - 2.*c1*dv[14]*irho*s - 2.*c1*dv[17]*dv[1]*dv[8]*(s)*(s) - 2.*c1*dv[17]*dv[1]*s + 2.*c1*dv[17]*dv[2]*dv[7]*(s)*(s) + 2.*c1*dv[17]*dv[6]*dv[7]*irho*s - 2.*c1*dv[17]*dv[8]*(s)*(s) - 2.*c1*dv[17]*s + c1*dv[18]*dv[7]*irho - c1*dv[19]*dv[2]*s - c1*dv[19]*dv[6]*irho - c1*dv[1]*dv[20]*s - c1*dv[1]*dv[21]*irho - c1*dv[20]*s - c1*dv[21]*irho - c1*dv[22]*dv[8]*s - c1*dv[22] - c1*dv[23]*dv[7]*s
                eps[9] = -dv[0]*dv[21]*irho - dv[14]*(dv[18])*(dv[18])*s/2. - dv[14]*(dv[21])*(dv[21])*s/2. + dv[14]*s + dv[18]*dv[2]*s + dv[18]*dv[6]*irho + dv[21]*dv[8]*s + dv[21]
                eps[10] = -dv[13]*(dv[18])*(dv[18])/2. - dv[13]*(dv[21])*(dv[21])/2. + dv[13] + dv[18]*dv[1] + dv[18] + dv[21]*dv[7]
                eps[11] = 3.*c1*dv[0]*dv[21]*irho + 3.*c1*dv[14]*(dv[18])*(dv[18])*s/2. + 3.*c1*dv[14]*(dv[21])*(dv[21])*s/2. - 3.*c1*dv[14]*s - 3.*c1*dv[18]*dv[2]*s - 3.*c1*dv[18]*dv[6]*irho - 3.*c1*dv[21]*dv[8]*s - 3.*c1*dv[21]
                eps[12] = 3.*c1*dv[13]*(dv[18])*(dv[18])/2. + 3.*c1*dv[13]*(dv[21])*(dv[21])/2. - 3.*c1*dv[13] - 3.*c1*dv[18]*dv[1] - 3.*c1*dv[18] - 3.*c1*dv[21]*dv[7]
                for r in range(13):
                    sig[r] = 0.
                    for q in range(13):
                        sig[r] += F[r*13 + q]*eps[q]
                for p in range(312):
                    Bm[p] = 0.
                Bm[1] = dv[1] + 1.
                Bm[7] = dv[7]
                Bm[13] = dv[13]
                Bm[24] = dv[0]*(irho)*(irho) - dv[8]*irho*s - irho
                Bm[26] = dv[2]*(s)*(s) + dv[6]*irho*s
                Bm[30] = dv[2]*irho*s + dv[6]*(irho)*(irho)
                Bm[32] = -dv[0]*irho*s + dv[8]*(s)*(s) + s
                Bm[38] = dv[14]*(s)*(s)
                Bm[48] = -dv[7]*irho
                Bm[49] = dv[2]*s + dv[6]*irho
                Bm[50] = dv[1]*s + s
                Bm[54] = dv[1]*irho + irho
                Bm[55] = -dv[0]*irho + dv[8]*s + 1.
                Bm[56] = dv[7]*s
                Bm[61] = dv[14]*s
                Bm[62] = dv[13]*s
                Bm[73] = dv[19]
                Bm[79] = dv[22]
                Bm[85] = -dv[18]*dv[19] - dv[21]*dv[22]
                Bm[90] = -dv[13]*dv[19]
                Bm[91] = -dv[13]*dv[18] + dv[1] + 1.
                Bm[93] = -dv[13]*dv[22]
                Bm[94] = -dv[13]*dv[21] + dv[7]
                Bm[96] = dv[18]*(irho)*(irho) - dv[23]*irho*s
                Bm[98] = dv[20]*(s)*(s) + dv[21]*irho*s
                Bm[102] = dv[20]*irho*s + dv[21]*(irho)*(irho)
                Bm[104] = -dv[18]*irho*s + dv[23]*(s)*(s)
                Bm[110] = -dv[18]*dv[20]*(s)*(s) - dv[21]*dv[23]*(s)*(s)
                Bm[114] = dv[0]*(irho)*(irho) - dv[14]*dv[20]*(s)*(s) - dv[8]*irho*s - irho
                Bm[116] = -dv[14]*dv[18]*(s)*(s) + dv[2]*(s)*(s) + dv[6]*irho*s
                Bm[117] = -dv[14]*dv[23]*(s)*(s) + dv[2]*irho*s + dv[6]*(irho)*(irho)
                Bm[119] = -dv[0]*irho*s - dv[14]*dv[21]*(s)*(s) + dv[8]*(s)*(s) + s
                Bm[120] = -dv[22]*irho
                Bm[121] = dv[20]*s + dv[21]*irho
                Bm[122] = dv[19]*s
                Bm[126] = dv[19]*irho
                Bm[127] = -dv[18]*irho + dv[23]*s
                Bm[128] = dv[22]*s
                Bm[133] = -dv[18]*dv[20]*s - dv[21]*dv[23]*s
                Bm[134] = -dv[18]*dv[19]*s - dv[21]*dv[22]*s
                Bm[138] = -dv[13]*dv[20]*s - dv[14]*dv[19]*s - dv[7]*irho
                Bm[139] = -dv[14]*dv[18]*s + dv[2]*s + dv[6]*irho
                Bm[140] = -dv[13]*dv[18]*s + dv[1]*s + s
                Bm[141] = -dv[13]*dv[23]*s - dv[14]*dv[22]*s + dv[1]*irho + irho
                Bm[142] = -dv[0]*irho - dv[14]*dv[21]*s + dv[8]*s + 1.
                Bm[143] = -dv[13]*dv[21]*s + dv[7]*s
                Bm[144] = -c1*dv[13]*dv[3]*irho + c1*dv[15]*dv[1]*irho + c1*dv[15]*irho
                Bm[145] = c1*dv[0]*dv[15]*irho + c1*dv[14]*dv[9]*s - c1*dv[15]*dv[8]*s - c1*dv[15] - c1*dv[19]
                Bm[146] = -c1*dv[13]*dv[9]*s + c1*dv[15]*dv[7]*s
                Bm[147] = -c1*dv[0]*dv[13]*irho + c1*dv[13]*dv[8]*s + c1*dv[13] - c1*dv[14]*dv[7]*s
                Bm[150] = -c1*dv[13]*dv[9]*irho + c1*dv[15]*dv[7]*irho
                Bm[151] = -c1*dv[14]*dv[3]*s + c1*dv[15]*dv[2]*s + c1*dv[15]*dv[6]*irho - c1*dv[22]
                Bm[152] = c1*dv[13]*dv[3]*s - c1*dv[15]*dv[1]*s - c1*dv[15]*s
                Bm[153] = -c1*dv[13]*dv[2]*s - c1*dv[13]*dv[6]*irho + c1*dv[14]*dv[1]*s + c1*dv[14]*s
                Bm[157] = -c1*dv[0]*dv[3]*irho + c1*dv[18]*dv[19] + c1*dv[21]*dv[22] - c1*dv[2]*dv[9]*s + c1*dv[3]*dv[8]*s + c1*dv[3] - c1*dv[6]*dv[9]*irho
                Bm[158] = c1*dv[1]*dv[9]*s - c1*dv[3]*dv[7]*s + c1*dv[9]*s
                Bm[159] = c1*dv[0]*dv[1]*irho + c1*dv[0]*irho - c1*dv[1]*dv[8]*s - c1*dv[1] + c1*dv[2]*dv[7]*s + c1*dv[6]*dv[7]*irho - c1*dv[8]*s - c1
                Bm[162] = c1*dv[13]*dv[19]
                Bm[163] = c1*dv[13]*dv[18] - c1*dv[1] - c1
                Bm[165] = c1*dv[13]*dv[22]
                Bm[166] = c1*dv[13]*dv[21] - c1*dv[7]
                Bm[168] = 2.*c1*dv[0]*dv[13]*(irho)*(irho)*(irho) - c1*dv[13]*dv[4]*irho*(s)*(s) - 3.*c1*dv[13]*dv[8]*(irho)*(irho)*s - 2.*c1*dv[13]*(irho)*(irho) + c1*dv[14]*dv[7]*(irho)*(irho)*s + c1*dv[16]*dv[1]*irho*(s)*(s) + c1*dv[16]*irho*(s)*(s) - c1*dv[18]*(irho)*(irho) + c1*dv[23]*irho*s
                Bm[169] = c1*dv[0]*dv[16]*irho*(s)*(s) + c1*dv[10]*dv[14]*(s)*(s)*(s) - 2.*c1*dv[14]*dv[2]*irho*(s)*(s) - c1*dv[14]*dv[6]*(irho)*(irho)*s - c1*dv[16]*dv[8]*(s)*(s)*(s) - c1*dv[16]*(s)*(s)
                Bm[170] = -c1*dv[10]*dv[13]*(s)*(s)*(s) + 4.*c1*dv[13]*dv[2]*irho*(s)*(s) + 3.*c1*dv[13]*dv[6]*(irho)*(irho)*s - 2.*c1*dv[14]*dv[1]*irho*(s)*(s) - 2.*c1*dv[14]*irho*(s)*(s) + c1*dv[16]*dv[7]*(s)*(s)*(s) - c1*dv[20]*(s)*(s) - c1*dv[21]*irho*s
                Bm[172] = -c1*dv[0]*dv[13]*irho*(s)*(s) + c1*dv[13]*dv[8]*(s)*(s)*(s) + c1*dv[13]*(s)*(s) - c1*dv[14]*dv[7]*(s)*(s)*(s)
                Bm[174] = -c1*dv[10]*dv[13]*irho*(s)*(s) + 3.*c1*dv[13]*dv[2]*(irho)*(irho)*s + 2.*c1*dv[13]*dv[6]*(irho)*(irho)*(irho) - c1*dv[14]*dv[1]*(irho)*(irho)*s - c1*dv[14]*(irho)*(irho)*s + c1*dv[16]*dv[7]*irho*(s)*(s) - c1*dv[20]*irho*s - c1*dv[21]*(irho)*(irho)
                Bm[175] = c1*dv[0]*dv[14]*(irho)*(irho)*s - c1*dv[14]*dv[4]*(s)*(s)*(s) - 2.*c1*dv[14]*dv[8]*irho*(s)*(s) - c1*dv[14]*irho*s + c1*dv[16]*dv[2]*(s)*(s)*(s) + c1*dv[16]*dv[6]*irho*(s)*(s)
                Bm[176] = -3.*c1*dv[0]*dv[13]*(irho)*(irho)*s + c1*dv[13]*dv[4]*(s)*(s)*(s) + 4.*c1*dv[13]*dv[8]*irho*(s)*(s) + 3.*c1*dv[13]*irho*s - 2.*c1*dv[14]*dv[7]*irho*(s)*(s) - c1*dv[16]*dv[1]*(s)*(s)*(s) - c1*dv[16]*(s)*(s)*(s) + c1*dv[18]*irho*s - c1*dv[23]*(s)*(s)
                Bm[178] = -c1*dv[13]*dv[2]*(s)*(s)*(s) - c1*dv[13]*dv[6]*irho*(s)*(s) + c1*dv[14]*dv[1]*(s)*(s)*(s) + c1*dv[14]*(s)*(s)*(s)
                Bm[181] = c1*(dv[0])*(dv[0])*(irho)*(irho)*(irho) - c1*dv[0]*dv[4]*irho*(s)*(s) - 3.*c1*dv[0]*dv[8]*(irho)*(irho)*s - 2.*c1*dv[0]*(irho)*(irho) - c1*dv[10]*dv[2]*(s)*(s)*(s) - c1*dv[10]*dv[6]*irho*(s)*(s) + 2.*c1*(dv[2])*(dv[2])*irho*(s)*(s) + 3.*c1*dv[2]*dv[6]*(irho)*(irho)*s + c1*dv[4]*dv[8]*(s)*(s)*(s) + c1*dv[4]*(s)*(s) + c1*(dv[6])*(dv[6])*(irho)*(irho)*(irho) + 2.*c1*(dv[8])*(dv[8])*irho*(s)*(s) + 3.*c1*dv[8]*irho*s + c1*irho
                Bm[182] = c1*dv[0]*dv[7]*(irho)*(irho)*s + c1*dv[10]*dv[1]*(s)*(s)*(s) + c1*dv[10]*(s)*(s)*(s) + c1*dv[18]*dv[20]*(s)*(s) - 2.*c1*dv[1]*dv[2]*irho*(s)*(s) - c1*dv[1]*dv[6]*(irho)*(irho)*s + c1*dv[21]*dv[23]*(s)*(s) - 2.*c1*dv[2]*irho*(s)*(s) - c1*dv[4]*dv[7]*(s)*(s)*(s) - c1*dv[6]*(irho)*(irho)*s - 2.*c1*dv[7]*dv[8]*irho*(s)*(s) - c1*dv[7]*irho*s
                Bm[184] = c1*dv[0]*dv[1]*irho*(s)*(s) + c1*dv[0]*irho*(s)*(s) - c1*dv[1]*dv[8]*(s)*(s)*(s) - c1*dv[1]*(s)*(s) + c1*dv[2]*dv[7]*(s)*(s)*(s) + c1*dv[6]*dv[7]*irho*(s)*(s) - c1*dv[8]*(s)*(s)*(s) - c1*(s)*(s)
                Bm[186] = -c1*dv[0]*(irho)*(irho) + c1*dv[14]*dv[20]*(s)*(s) + c1*dv[8]*irho*s + c1*irho
                Bm[188] = c1*dv[14]*dv[18]*(s)*(s) - c1*dv[2]*(s)*(s) - c1*dv[6]*irho*s
                Bm[189] = c1*dv[14]*dv[23]*(s)*(s) - c1*dv[2]*irho*s - c1*dv[6]*(irho)*(irho)
                Bm[191] = c1*dv[0]*irho*s + c1*dv[14]*dv[21]*(s)*(s) - c1*dv[8]*(s)*(s) - c1*s
                Bm[192] = -2.*c1*dv[13]*dv[5]*irho*s - 2.*c1*dv[13]*dv[7]*(irho)*(irho) + 2.*c1*dv[17]*dv[1]*irho*s + 2.*c1*dv[17]*irho*s + c1*dv[22]*irho
                Bm[193] = 2.*c1*dv[0]*dv[17]*irho*s + 2.*c1*dv[11]*dv[14]*(s)*(s) + 2.*c1*dv[13]*dv[2]*irho*s + 2.*c1*dv[13]*dv[6]*(irho)*(irho) - 4.*c1*dv[14]*dv[1]*irho*s - 4.*c1*dv[14]*irho*s - 2.*c1*dv[17]*dv[8]*(s)*(s) - 2.*c1*dv[17]*s - c1*dv[20]*s - c1*dv[21]*irho
                Bm[194] = -2.*c1*dv[11]*dv[13]*(s)*(s) + 2.*c1*dv[13]*dv[1]*irho*s + 2.*c1*dv[13]*irho*s + 2.*c1*dv[17]*dv[7]*(s)*(s) - c1*dv[19]*s
                Bm[197] = -2.*c1*dv[0]*dv[13]*irho*s + 2.*c1*dv[13]*dv[8]*(s)*(s) + 2.*c1*dv[13]*s - 2.*c1*dv[14]*dv[7]*(s)*(s)
                Bm[198] = -2.*c1*dv[11]*dv[13]*irho*s + 2.*c1*dv[13]*dv[1]*(irho)*(irho) + 2.*c1*dv[13]*(irho)*(irho) + 2.*c1*dv[17]*dv[7]*irho*s - c1*dv[19]*irho
                Bm[199] = -2.*c1*dv[0]*dv[13]*(irho)*(irho) + 2.*c1*dv[13]*dv[8]*irho*s + 2.*c1*dv[13]*irho - 2.*c1*dv[14]*dv[5]*(s)*(s) - 4.*c1*dv[14]*dv[7]*irho*s + 2.*c1*dv[17]*dv[2]*(s)*(s) + 2.*c1*dv[17]*dv[6]*irho*s + c1*dv[18]*irho - c1*dv[23]*s
                Bm[200] = 2.*c1*dv[13]*dv[5]*(s)*(s) + 2.*c1*dv[13]*dv[7]*irho*s - 2.*c1*dv[17]*dv[1]*(s)*(s) - 2.*c1*dv[17]*(s)*(s) - c1*dv[22]*s
                Bm[203] = -2.*c1*dv[13]*dv[2]*(s)*(s) - 2.*c1*dv[13]*dv[6]*irho*s + 2.*c1*dv[14]*dv[1]*(s)*(s) + 2.*c1*dv[14]*(s)*(s)
                Bm[205] = -2.*c1*dv[0]*dv[5]*irho*s - 2.*c1*dv[0]*dv[7]*(irho)*(irho) - 2.*c1*dv[11]*dv[2]*(s)*(s) - 2.*c1*dv[11]*dv[6]*irho*s + c1*dv[18]*dv[20]*s + 2.*c1*dv[1]*dv[2]*irho*s + 2.*c1*dv[1]*dv[6]*(irho)*(irho) + c1*dv[21]*dv[23]*s + 2.*c1*dv[2]*irho*s + 2.*c1*dv[5]*dv[8]*(s)*(s) + 2.*c1*dv[5]*s + 2.*c1*dv[6]*(irho)*(irho) + 2.*c1*dv[7]*dv[8]*irho*s + 2.*c1*dv[7]*irho
                Bm[206] = 2.*c1*dv[11]*dv[1]*(s)*(s) + 2.*c1*dv[11]*(s)*(s) + c1*dv[18]*dv[19]*s - 2.*c1*(dv[1])*(dv[1])*irho*s - 4.*c1*dv[1]*irho*s + c1*dv[21]*dv[22]*s - 2.*c1*dv[5]*dv[7]*(s)*(s) - 2.*c1*(dv[7])*(dv[7])*irho*s - 2.*c1*irho*s
                Bm[209] = 2.*c1*dv[0]*dv[1]*irho*s + 2.*c1*dv[0]*irho*s - 2.*c1*dv[1]*dv[8]*(s)*(s) - 2.*c1*dv[1]*s + 2.*c1*dv[2]*dv[7]*(s)*(s) + 2.*c1*dv[6]*dv[7]*irho*s - 2.*c1*dv[8]*(s)*(s) - 2.*c1*s
                Bm[210] = c1*dv[13]*dv[20]*s + c1*dv[14]*dv[19]*s + c1*dv[7]*irho
                Bm[211] = c1*dv[14]*dv[18]*s - c1*dv[2]*s - c1*dv[6]*irho
                Bm[212] = c1*dv[13]*dv[18]*s - c1*dv[1]*s - c1*s
                Bm[213] = c1*dv[13]*dv[23]*s + c1*dv[14]*dv[22]*s - c1*dv[1]*irho - c1*irho
                Bm[214] = c1*dv[0]*irho + c1*dv[14]*dv[21]*s - c1*dv[8]*s - c1
                Bm[215] = c1*dv[13]*dv[21]*s - c1*dv[7]*s
                Bm[216] = -dv[21]*irho
                Bm[218] = dv[18]*s
                Bm[222] = dv[18]*irho
                Bm[224] = dv[21]*s
                Bm[230] = -(dv[18])*(dv[18])*s/2. - (dv[21])*(dv[21])*s/2. + s
                Bm[234] = -dv[14]*dv[18]*s + dv[2]*s + dv[6]*irho
                Bm[237] = -dv[0]*irho - dv[14]*dv[21]*s + dv[8]*s + 1.
                Bm[241] = dv[18]
                Bm[247] = dv[21]
                Bm[253] = -(dv[18])*(dv[18])/2. - (dv[21])*(dv[21])/2. + 1.
                Bm[258] = -dv[13]*dv[18] + dv[1] + 1.
                Bm[261] = -dv[13]*dv[21] + dv[7]
                Bm[264] = 3.*c1*dv[21]*irho
                Bm[266] = -3.*c1*dv[18]*s
                Bm[270] = -3.*c1*dv[18]*irho
                Bm[272] = -3.*c1*dv[21]*s
                Bm[278] = 3.*c1*(dv[18])*(dv[18])*s/2. + 3.*c1*(dv[21])*(dv[21])*s/2. - 3.*c1*s
                Bm[282] = 3.*c1*dv[14]*dv[18]*s - 3.*c1*dv[2]*s - 3.*c1*dv[6]*irho
                Bm[285] = 3.*c1*dv[0]*irho + 3.*c1*dv[14]*dv[21]*s - 3.*c1*dv[8]*s - 3.*c1
                Bm[289] = -3.*c1*dv[18]
                Bm[295] = -3.*c1*dv[21]
                Bm[301] = 3.*c1*(dv[18])*(dv[18])/2. + 3.*c1*(dv[21])*(dv[21])/2. - 3.*c1
                Bm[306] = 3.*c1*dv[13]*dv[18] - 3.*c1*dv[1] - 3.*c1
                Bm[309] = 3.*c1*dv[13]*dv[21] - 3.*c1*dv[7]
                for r in range(13):
                    Bq[r] = 0.
                    for p in range(24):
                        Bq[r] += Bm[r*24 + p]*dq[p]
                for r in range(13):
                    FBq[r] = 0.
                    for q in range(13):
                        FBq[r] += F[r*13 + q]*Bq[q]
                for p in range(24):
                    bs[p] = 0.
                    for r in range(13):
                        bs[p] += Bm[r*24 + p]*FBq[r]
                bs[0] += -Nxy_cte*dq[7]*irho + Nyy_cte*dq[0]*(irho)*(irho) - Nyy_cte*dq[8]*irho*s + 2.*c1*dq[0]*dv[13]*(irho)*(irho)*(irho)*sig[7] + 2.*c1*dq[13]*dv[0]*(irho)*(irho)*(irho)*sig[7] - c1*dq[13]*dv[3]*irho*sig[6] - c1*dq[13]*dv[4]*irho*(s)*(s)*sig[7] - 2.*c1*dq[13]*dv[5]*irho*s*sig[8] - 2.*c1*dq[13]*dv[7]*(irho)*(irho)*sig[8] - 3.*c1*dq[13]*dv[8]*(irho)*(irho)*s*sig[7] - 2.*c1*dq[13]*(irho)*(irho)*sig[7] + c1*dq[14]*dv[7]*(irho)*(irho)*s*sig[7] + c1*dq[15]*dv[1]*irho*sig[6] + c1*dq[15]*irho*sig[6] + c1*dq[16]*dv[1]*irho*(s)*(s)*sig[7] + c1*dq[16]*irho*(s)*(s)*sig[7] + 2.*c1*dq[17]*dv[1]*irho*s*sig[8] + 2.*c1*dq[17]*irho*s*sig[8] - c1*dq[18]*(irho)*(irho)*sig[7] + c1*dq[1]*dv[15]*irho*sig[6] + c1*dq[1]*dv[16]*irho*(s)*(s)*sig[7] + 2.*c1*dq[1]*dv[17]*irho*s*sig[8] + 3.*c1*dq[21]*irho*sig[11] + c1*dq[22]*irho*sig[8] + c1*dq[23]*irho*s*sig[7] - c1*dq[3]*dv[13]*irho*sig[6] - c1*dq[4]*dv[13]*irho*(s)*(s)*sig[7] - 2.*c1*dq[5]*dv[13]*irho*s*sig[8] - 2.*c1*dq[7]*dv[13]*(irho)*(irho)*sig[8] + c1*dq[7]*dv[14]*(irho)*(irho)*s*sig[7] - 3.*c1*dq[8]*dv[13]*(irho)*(irho)*s*sig[7] + dq[0]*(irho)*(irho)*sig[1] + dq[18]*(irho)*(irho)*sig[4] - dq[21]*irho*sig[9] - dq[22]*irho*sig[5] - dq[23]*irho*s*sig[4] - dq[7]*irho*sig[2] - dq[8]*irho*s*sig[1]
                bs[1] += Nxx_cte*dq[1] + Nxy_cte*dq[2]*s + Nxy_cte*dq[6]*irho + c1*dq[0]*dv[15]*irho*sig[6] + c1*dq[0]*dv[16]*irho*(s)*(s)*sig[7] + 2.*c1*dq[0]*dv[17]*irho*s*sig[8] + c1*dq[10]*dv[14]*(s)*(s)*(s)*sig[7] + 2.*c1*dq[11]*dv[14]*(s)*(s)*sig[8] + 2.*c1*dq[13]*dv[2]*irho*s*sig[8] + 2.*c1*dq[13]*dv[6]*(irho)*(irho)*sig[8] + c1*dq[14]*dv[10]*(s)*(s)*(s)*sig[7] + 2.*c1*dq[14]*dv[11]*(s)*(s)*sig[8] - 4.*c1*dq[14]*dv[1]*irho*s*sig[8] - 2.*c1*dq[14]*dv[2]*irho*(s)*(s)*sig[7] - c1*dq[14]*dv[6]*(irho)*(irho)*s*sig[7] + c1*dq[14]*dv[9]*s*sig[6] - 4.*c1*dq[14]*irho*s*sig[8] + c1*dq[15]*dv[0]*irho*sig[6] - c1*dq[15]*dv[8]*s*sig[6] - c1*dq[15]*sig[6] + c1*dq[16]*dv[0]*irho*(s)*(s)*sig[7] - c1*dq[16]*dv[8]*(s)*(s)*(s)*sig[7] - c1*dq[16]*(s)*(s)*sig[7] + 2.*c1*dq[17]*dv[0]*irho*s*sig[8] - 2.*c1*dq[17]*dv[8]*(s)*(s)*sig[8] - 2.*c1*dq[17]*s*sig[8] - 3.*c1*dq[18]*sig[12] - c1*dq[19]*sig[6] - 4.*c1*dq[1]*dv[14]*irho*s*sig[8] - c1*dq[20]*s*sig[8] - c1*dq[21]*irho*sig[8] + 2.*c1*dq[2]*dv[13]*irho*s*sig[8] - 2.*c1*dq[2]*dv[14]*irho*(s)*(s)*sig[7] + 2.*c1*dq[6]*dv[13]*(irho)*(irho)*sig[8] - c1*dq[6]*dv[14]*(irho)*(irho)*s*sig[7] - c1*dq[8]*dv[15]*s*sig[6] - c1*dq[8]*dv[16]*(s)*(s)*(s)*sig[7] - 2.*c1*dq[8]*dv[17]*(s)*(s)*sig[8] + c1*dq[9]*dv[14]*s*sig[6] + dq[18]*sig[10] + dq[19]*sig[3] + dq[1]*sig[0] + dq[20]*s*sig[5] + dq[21]*irho*sig[5] + dq[2]*s*sig[2] + dq[6]*irho*sig[2]
                bs[2] += Nxy_cte*dq[1]*s + Nyy_cte*dq[2]*(s)*(s) + Nyy_cte*dq[6]*irho*s - c1*dq[10]*dv[13]*(s)*(s)*(s)*sig[7] - 2.*c1*dq[11]*dv[13]*(s)*(s)*sig[8] - c1*dq[13]*dv[10]*(s)*(s)*(s)*sig[7] - 2.*c1*dq[13]*dv[11]*(s)*(s)*sig[8] + 2.*c1*dq[13]*dv[1]*irho*s*sig[8] + 4.*c1*dq[13]*dv[2]*irho*(s)*(s)*sig[7] + 3.*c1*dq[13]*dv[6]*(irho)*(irho)*s*sig[7] - c1*dq[13]*dv[9]*s*sig[6] + 2.*c1*dq[13]*irho*s*sig[8] - 2.*c1*dq[14]*dv[1]*irho*(s)*(s)*sig[7] - 2.*c1*dq[14]*irho*(s)*(s)*sig[7] + c1*dq[15]*dv[7]*s*sig[6] + c1*dq[16]*dv[7]*(s)*(s)*(s)*sig[7] + 2.*c1*dq[17]*dv[7]*(s)*(s)*sig[8] - 3.*c1*dq[18]*s*sig[11] - c1*dq[19]*s*sig[8] + 2.*c1*dq[1]*dv[13]*irho*s*sig[8] - 2.*c1*dq[1]*dv[14]*irho*(s)*(s)*sig[7] - c1*dq[20]*(s)*(s)*sig[7] - c1*dq[21]*irho*s*sig[7] + 4.*c1*dq[2]*dv[13]*irho*(s)*(s)*sig[7] + 3.*c1*dq[6]*dv[13]*(irho)*(irho)*s*sig[7] + c1*dq[7]*dv[15]*s*sig[6] + c1*dq[7]*dv[16]*(s)*(s)*(s)*sig[7] + 2.*c1*dq[7]*dv[17]*(s)*(s)*sig[8] - c1*dq[9]*dv[13]*s*sig[6] + dq[18]*s*sig[9] + dq[19]*s*sig[5] + dq[1]*s*sig[2] + dq[20]*(s)*(s)*sig[4] + dq[21]*irho*s*sig[4] + dq[2]*(s)*(s)*sig[1] + dq[6]*irho*s*sig[1]
                bs[3] += -c1*dq[0]*dv[13]*irho*sig[6] - c1*dq[13]*dv[0]*irho*sig[6] + c1*dq[13]*dv[8]*s*sig[6] + c1*dq[13]*sig[6] - c1*dq[14]*dv[7]*s*sig[6] - c1*dq[7]*dv[14]*s*sig[6] + c1*dq[8]*dv[13]*s*sig[6]
                bs[4] += -c1*dq[0]*dv[13]*irho*(s)*(s)*sig[7] - c1*dq[13]*dv[0]*irho*(s)*(s)*sig[7] + c1*dq[13]*dv[8]*(s)*(s)*(s)*sig[7] + c1*dq[13]*(s)*(s)*sig[7] - c1*dq[14]*dv[7]*(s)*(s)*(s)*sig[7] - c1*dq[7]*dv[14]*(s)*(s)*(s)*sig[7] + c1*dq[8]*dv[13]*(s)*(s)*(s)*sig[7]
                bs[5] += -2.*c1*dq[0]*dv[13]*irho*s*sig[8] - 2.*c1*dq[13]*dv[0]*irho*s*sig[8] + 2.*c1*dq[13]*dv[8]*(s)*(s)*sig[8] + 2.*c1*dq[13]*s*sig[8] - 2.*c1*dq[14]*dv[7]*(s)*(s)*sig[8] - 2.*c1*dq[7]*dv[14]*(s)*(s)*sig[8] + 2.*c1*dq[8]*dv[13]*(s)*(s)*sig[8]
                bs[6] += Nxy_cte*dq[1]*irho + Nyy_cte*dq[2]*irho*s + Nyy_cte*dq[6]*(irho)*(irho) - c1*dq[10]*dv[13]*irho*(s)*(s)*sig[7] - 2.*c1*dq[11]*dv[13]*irho*s*sig[8] - c1*dq[13]*dv[10]*irho*(s)*(s)*sig[7] - 2.*c1*dq[13]*dv[11]*irho*s*sig[8] + 2.*c1*dq[13]*dv[1]*(irho)*(irho)*sig[8] + 3.*c1*dq[13]*dv[2]*(irho)*(irho)*s*sig[7] + 2.*c1*dq[13]*dv[6]*(irho)*(irho)*(irho)*sig[7] - c1*dq[13]*dv[9]*irho*sig[6] + 2.*c1*dq[13]*(irho)*(irho)*sig[8] - c1*dq[14]*dv[1]*(irho)*(irho)*s*sig[7] - c1*dq[14]*(irho)*(irho)*s*sig[7] + c1*dq[15]*dv[7]*irho*sig[6] + c1*dq[16]*dv[7]*irho*(s)*(s)*sig[7] + 2.*c1*dq[17]*dv[7]*irho*s*sig[8] - 3.*c1*dq[18]*irho*sig[11] - c1*dq[19]*irho*sig[8] + 2.*c1*dq[1]*dv[13]*(irho)*(irho)*sig[8] - c1*dq[1]*dv[14]*(irho)*(irho)*s*sig[7] - c1*dq[20]*irho*s*sig[7] - c1*dq[21]*(irho)*(irho)*sig[7] + 3.*c1*dq[2]*dv[13]*(irho)*(irho)*s*sig[7] + 2.*c1*dq[6]*dv[13]*(irho)*(irho)*(irho)*sig[7] + c1*dq[7]*dv[15]*irho*sig[6] + c1*dq[7]*dv[16]*irho*(s)*(s)*sig[7] + 2.*c1*dq[7]*dv[17]*irho*s*sig[8] - c1*dq[9]*dv[13]*irho*sig[6] + dq[18]*irho*sig[9] + dq[19]*irho*sig[5] + dq[1]*irho*sig[2] + dq[20]*irho*s*sig[4] + dq[21]*(irho)*(irho)*sig[4] + dq[2]*irho*s*sig[1] + dq[6]*(irho)*(irho)*sig[1]
                bs[7] += Nxx_cte*dq[7] - Nxy_cte*dq[0]*irho + Nxy_cte*dq[8]*s - 2.*c1*dq[0]*dv[13]*(irho)*(irho)*sig[8] + c1*dq[0]*dv[14]*(irho)*(irho)*s*sig[7] - 2.*c1*dq[13]*dv[0]*(irho)*(irho)*sig[8] + 2.*c1*dq[13]*dv[8]*irho*s*sig[8] + 2.*c1*dq[13]*irho*sig[8] + c1*dq[14]*dv[0]*(irho)*(irho)*s*sig[7] - c1*dq[14]*dv[3]*s*sig[6] - c1*dq[14]*dv[4]*(s)*(s)*(s)*sig[7] - 2.*c1*dq[14]*dv[5]*(s)*(s)*sig[8] - 4.*c1*dq[14]*dv[7]*irho*s*sig[8] - 2.*c1*dq[14]*dv[8]*irho*(s)*(s)*sig[7] - c1*dq[14]*irho*s*sig[7] + c1*dq[15]*dv[2]*s*sig[6] + c1*dq[15]*dv[6]*irho*sig[6] + c1*dq[16]*dv[2]*(s)*(s)*(s)*sig[7] + c1*dq[16]*dv[6]*irho*(s)*(s)*sig[7] + 2.*c1*dq[17]*dv[2]*(s)*(s)*sig[8] + 2.*c1*dq[17]*dv[6]*irho*s*sig[8] + c1*dq[18]*irho*sig[8] - 3.*c1*dq[21]*sig[12] - c1*dq[22]*sig[6] - c1*dq[23]*s*sig[8] + c1*dq[2]*dv[15]*s*sig[6] + c1*dq[2]*dv[16]*(s)*(s)*(s)*sig[7] + 2.*c1*dq[2]*dv[17]*(s)*(s)*sig[8] - c1*dq[3]*dv[14]*s*sig[6] - c1*dq[4]*dv[14]*(s)*(s)*(s)*sig[7] - 2.*c1*dq[5]*dv[14]*(s)*(s)*sig[8] + c1*dq[6]*dv[15]*irho*sig[6] + c1*dq[6]*dv[16]*irho*(s)*(s)*sig[7] + 2.*c1*dq[6]*dv[17]*irho*s*sig[8] - 4.*c1*dq[7]*dv[14]*irho*s*sig[8] + 2.*c1*dq[8]*dv[13]*irho*s*sig[8] - 2.*c1*dq[8]*dv[14]*irho*(s)*(s)*sig[7] - dq[0]*irho*sig[2] - dq[18]*irho*sig[5] + dq[21]*sig[10] + dq[22]*sig[3] + dq[23]*s*sig[5] + dq[7]*sig[0] + dq[8]*s*sig[2]
                bs[8] += Nxy_cte*dq[7]*s - Nyy_cte*dq[0]*irho*s + Nyy_cte*dq[8]*(s)*(s) - 3.*c1*dq[0]*dv[13]*(irho)*(irho)*s*sig[7] - 3.*c1*dq[13]*dv[0]*(irho)*(irho)*s*sig[7] + c1*dq[13]*dv[3]*s*sig[6] + c1*dq[13]*dv[4]*(s)*(s)*(s)*sig[7] + 2.*c1*dq[13]*dv[5]*(s)*(s)*sig[8] + 2.*c1*dq[13]*dv[7]*irho*s*sig[8] + 4.*c1*dq[13]*dv[8]*irho*(s)*(s)*sig[7] + 3.*c1*dq[13]*irho*s*sig[7] - 2.*c1*dq[14]*dv[7]*irho*(s)*(s)*sig[7] - c1*dq[15]*dv[1]*s*sig[6] - c1*dq[15]*s*sig[6] - c1*dq[16]*dv[1]*(s)*(s)*(s)*sig[7] - c1*dq[16]*(s)*(s)*(s)*sig[7] - 2.*c1*dq[17]*dv[1]*(s)*(s)*sig[8] - 2.*c1*dq[17]*(s)*(s)*sig[8] + c1*dq[18]*irho*s*sig[7] - c1*dq[1]*dv[15]*s*sig[6] - c1*dq[1]*dv[16]*(s)*(s)*(s)*sig[7] - 2.*c1*dq[1]*dv[17]*(s)*(s)*sig[8] - 3.*c1*dq[21]*s*sig[11] - c1*dq[22]*s*sig[8] - c1*dq[23]*(s)*(s)*sig[7] + c1*dq[3]*dv[13]*s*sig[6] + c1*dq[4]*dv[13]*(s)*(s)*(s)*sig[7] + 2.*c1*dq[5]*dv[13]*(s)*(s)*sig[8] + 2.*c1*dq[7]*dv[13]*irho*s*sig[8] - 2.*c1*dq[7]*dv[14]*irho*(s)*(s)*sig[7] + 4.*c1*dq[8]*dv[13]*irho*(s)*(s)*sig[7] - dq[0]*irho*s*sig[1] - dq[18]*irho*s*sig[4] + dq[21]*s*sig[9] + dq[22]*s*sig[5] + dq[23]*(s)*(s)*sig[4] + dq[7]*s*sig[2] + dq[8]*(s)*(s)*sig[1]
                bs[9] += -c1*dq[13]*dv[2]*s*sig[6] - c1*dq[13]*dv[6]*irho*sig[6] + c1*dq[14]*dv[1]*s*sig[6] + c1*dq[14]*s*sig[6] + c1*dq[1]*dv[14]*s*sig[6] - c1*dq[2]*dv[13]*s*sig[6] - c1*dq[6]*dv[13]*irho*sig[6]
                bs[10] += -c1*dq[13]*dv[2]*(s)*(s)*(s)*sig[7] - c1*dq[13]*dv[6]*irho*(s)*(s)*sig[7] + c1*dq[14]*dv[1]*(s)*(s)*(s)*sig[7] + c1*dq[14]*(s)*(s)*(s)*sig[7] + c1*dq[1]*dv[14]*(s)*(s)*(s)*sig[7] - c1*dq[2]*dv[13]*(s)*(s)*(s)*sig[7] - c1*dq[6]*dv[13]*irho*(s)*(s)*sig[7]
                bs[11] += -2.*c1*dq[13]*dv[2]*(s)*(s)*sig[8] - 2.*c1*dq[13]*dv[6]*irho*s*sig[8] + 2.*c1*dq[14]*dv[1]*(s)*(s)*sig[8] + 2.*c1*dq[14]*(s)*(s)*sig[8] + 2.*c1*dq[1]*dv[14]*(s)*(s)*sig[8] - 2.*c1*dq[2]*dv[13]*(s)*(s)*sig[8] - 2.*c1*dq[6]*dv[13]*irho*s*sig[8]
                bs[13] += Nxx_cte*dq[13] + Nxy_cte*dq[14]*s + 2.*c1*dq[0]*dv[0]*(irho)*(irho)*(irho)*sig[7] - c1*dq[0]*dv[3]*irho*sig[6] - c1*dq[0]*dv[4]*irho*(s)*(s)*sig[7] - 2.*c1*dq[0]*dv[5]*irho*s*sig[8] - 2.*c1*dq[0]*dv[7]*(irho)*(irho)*sig[8] - 3.*c1*dq[0]*dv[8]*(irho)*(irho)*s*sig[7] - 2.*c1*dq[0]*(irho)*(irho)*sig[7] - c1*dq[10]*dv[2]*(s)*(s)*(s)*sig[7] - c1*dq[10]*dv[6]*irho*(s)*(s)*sig[7] - 2.*c1*dq[11]*dv[2]*(s)*(s)*sig[8] - 2.*c1*dq[11]*dv[6]*irho*s*sig[8] + 3.*c1*dq[18]*dv[18]*sig[12] + c1*dq[18]*dv[19]*sig[6] + c1*dq[18]*dv[20]*s*sig[8] + c1*dq[19]*dv[18]*sig[6] + 2.*c1*dq[1]*dv[2]*irho*s*sig[8] + 2.*c1*dq[1]*dv[6]*(irho)*(irho)*sig[8] + c1*dq[20]*dv[18]*s*sig[8] + 3.*c1*dq[21]*dv[21]*sig[12] + c1*dq[21]*dv[22]*sig[6] + c1*dq[21]*dv[23]*s*sig[8] + c1*dq[22]*dv[21]*sig[6] + c1*dq[23]*dv[21]*s*sig[8] - c1*dq[2]*dv[10]*(s)*(s)*(s)*sig[7] - 2.*c1*dq[2]*dv[11]*(s)*(s)*sig[8] + 2.*c1*dq[2]*dv[1]*irho*s*sig[8] + 4.*c1*dq[2]*dv[2]*irho*(s)*(s)*sig[7] + 3.*c1*dq[2]*dv[6]*(irho)*(irho)*s*sig[7] - c1*dq[2]*dv[9]*s*sig[6] + 2.*c1*dq[2]*irho*s*sig[8] - c1*dq[3]*dv[0]*irho*sig[6] + c1*dq[3]*dv[8]*s*sig[6] + c1*dq[3]*sig[6] - c1*dq[4]*dv[0]*irho*(s)*(s)*sig[7] + c1*dq[4]*dv[8]*(s)*(s)*(s)*sig[7] + c1*dq[4]*(s)*(s)*sig[7] - 2.*c1*dq[5]*dv[0]*irho*s*sig[8] + 2.*c1*dq[5]*dv[8]*(s)*(s)*sig[8] + 2.*c1*dq[5]*s*sig[8] - c1*dq[6]*dv[10]*irho*(s)*(s)*sig[7] - 2.*c1*dq[6]*dv[11]*irho*s*sig[8] + 2.*c1*dq[6]*dv[1]*(irho)*(irho)*sig[8] + 3.*c1*dq[6]*dv[2]*(irho)*(irho)*s*sig[7] + 2.*c1*dq[6]*dv[6]*(irho)*(irho)*(irho)*sig[7] - c1*dq[6]*dv[9]*irho*sig[6] + 2.*c1*dq[6]*(irho)*(irho)*sig[8] - 2.*c1*dq[7]*dv[0]*(irho)*(irho)*sig[8] + 2.*c1*dq[7]*dv[8]*irho*s*sig[8] + 2.*c1*dq[7]*irho*sig[8] - 3.*c1*dq[8]*dv[0]*(irho)*(irho)*s*sig[7] + c1*dq[8]*dv[3]*s*sig[6] + c1*dq[8]*dv[4]*(s)*(s)*(s)*sig[7] + 2.*c1*dq[8]*dv[5]*(s)*(s)*sig[8] + 2.*c1*dq[8]*dv[7]*irho*s*sig[8] + 4.*c1*dq[8]*dv[8]*irho*(s)*(s)*sig[7] + 3.*c1*dq[8]*irho*s*sig[7] - c1*dq[9]*dv[2]*s*sig[6] - c1*dq[9]*dv[6]*irho*sig[6] + dq[13]*sig[0] + dq[14]*s*sig[2] - dq[18]*dv[18]*sig[10] - dq[18]*dv[19]*sig[3] - dq[18]*dv[20]*s*sig[5] - dq[19]*dv[18]*sig[3] - dq[20]*dv[18]*s*sig[5] - dq[21]*dv[21]*sig[10] - dq[21]*dv[22]*sig[3] - dq[21]*dv[23]*s*sig[5] - dq[22]*dv[21]*sig[3] - dq[23]*dv[21]*s*sig[5]
                bs[14] += Nxy_cte*dq[13]*s + Nyy_cte*dq[14]*(s)*(s) + c1*dq[0]*dv[7]*(irho)*(irho)*s*sig[7] + c1*dq[10]*dv[1]*(s)*(s)*(s)*sig[7] + c1*dq[10]*(s)*(s)*(s)*sig[7] + 2.*c1*dq[11]*dv[1]*(s)*(s)*sig[8] + 2.*c1*dq[11]*(s)*(s)*sig[8] + 3.*c1*dq[18]*dv[18]*s*sig[11] + c1*dq[18]*dv[19]*s*sig[8] + c1*dq[18]*dv[20]*(s)*(s)*sig[7] + c1*dq[19]*dv[18]*s*sig[8] + c1*dq[1]*dv[10]*(s)*(s)*(s)*sig[7] + 2.*c1*dq[1]*dv[11]*(s)*(s)*sig[8] - 4.*c1*dq[1]*dv[1]*irho*s*sig[8] - 2.*c1*dq[1]*dv[2]*irho*(s)*(s)*sig[7] - c1*dq[1]*dv[6]*(irho)*(irho)*s*sig[7] + c1*dq[1]*dv[9]*s*sig[6] - 4.*c1*dq[1]*irho*s*sig[8] + c1*dq[20]*dv[18]*(s)*(s)*sig[7] + 3.*c1*dq[21]*dv[21]*s*sig[11] + c1*dq[21]*dv[22]*s*sig[8] + c1*dq[21]*dv[23]*(s)*(s)*sig[7] + c1*dq[22]*dv[21]*s*sig[8] + c1*dq[23]*dv[21]*(s)*(s)*sig[7] - 2.*c1*dq[2]*dv[1]*irho*(s)*(s)*sig[7] - 2.*c1*dq[2]*irho*(s)*(s)*sig[7] - c1*dq[3]*dv[7]*s*sig[6] - c1*dq[4]*dv[7]*(s)*(s)*(s)*sig[7] - 2.*c1*dq[5]*dv[7]*(s)*(s)*sig[8] - c1*dq[6]*dv[1]*(irho)*(irho)*s*sig[7] - c1*dq[6]*(irho)*(irho)*s*sig[7] + c1*dq[7]*dv[0]*(irho)*(irho)*s*sig[7] - c1*dq[7]*dv[3]*s*sig[6] - c1*dq[7]*dv[4]*(s)*(s)*(s)*sig[7] - 2.*c1*dq[7]*dv[5]*(s)*(s)*sig[8] - 4.*c1*dq[7]*dv[7]*irho*s*sig[8] - 2.*c1*dq[7]*dv[8]*irho*(s)*(s)*sig[7] - c1*dq[7]*irho*s*sig[7] - 2.*c1*dq[8]*dv[7]*irho*(s)*(s)*sig[7] + c1*dq[9]*dv[1]*s*sig[6] + c1*dq[9]*s*sig[6] + dq[13]*s*sig[2] + dq[14]*(s)*(s)*sig[1] - dq[18]*dv[18]*s*sig[9] - dq[18]*dv[19]*s*sig[5] - dq[18]*dv[20]*(s)*(s)*sig[4] - dq[19]*dv[18]*s*sig[5] - dq[20]*dv[18]*(s)*(s)*sig[4] - dq[21]*dv[21]*s*sig[9] - dq[21]*dv[22]*s*sig[5] - dq[21]*dv[23]*(s)*(s)*sig[4] - dq[22]*dv[21]*s*sig[5] - dq[23]*dv[21]*(s)*(s)*sig[4]
                bs[15] += c1*dq[0]*dv[1]*irho*sig[6] + c1*dq[0]*irho*sig[6] + c1*dq[1]*dv[0]*irho*sig[6] - c1*dq[1]*dv[8]*s*sig[6] - c1*dq[1]*sig[6] + c1*dq[2]*dv[7]*s*sig[6] + c1*dq[6]*dv[7]*irho*sig[6] + c1*dq[7]*dv[2]*s*sig[6] + c1*dq[7]*dv[6]*irho*sig[6] - c1*dq[8]*dv[1]*s*sig[6] - c1*dq[8]*s*sig[6]
                bs[16] += c1*dq[0]*dv[1]*irho*(s)*(s)*sig[7] + c1*dq[0]*irho*(s)*(s)*sig[7] + c1*dq[1]*dv[0]*irho*(s)*(s)*sig[7] - c1*dq[1]*dv[8]*(s)*(s)*(s)*sig[7] - c1*dq[1]*(s)*(s)*sig[7] + c1*dq[2]*dv[7]*(s)*(s)*(s)*sig[7] + c1*dq[6]*dv[7]*irho*(s)*(s)*sig[7] + c1*dq[7]*dv[2]*(s)*(s)*(s)*sig[7] + c1*dq[7]*dv[6]*irho*(s)*(s)*sig[7] - c1*dq[8]*dv[1]*(s)*(s)*(s)*sig[7] - c1*dq[8]*(s)*(s)*(s)*sig[7]
                bs[17] += 2.*c1*dq[0]*dv[1]*irho*s*sig[8] + 2.*c1*dq[0]*irho*s*sig[8] + 2.*c1*dq[1]*dv[0]*irho*s*sig[8] - 2.*c1*dq[1]*dv[8]*(s)*(s)*sig[8] - 2.*c1*dq[1]*s*sig[8] + 2.*c1*dq[2]*dv[7]*(s)*(s)*sig[8] + 2.*c1*dq[6]*dv[7]*irho*s*sig[8] + 2.*c1*dq[7]*dv[2]*(s)*(s)*sig[8] + 2.*c1*dq[7]*dv[6]*irho*s*sig[8] - 2.*c1*dq[8]*dv[1]*(s)*(s)*sig[8] - 2.*c1*dq[8]*(s)*(s)*sig[8]
                bs[18] += -c1*dq[0]*(irho)*(irho)*sig[7] + 3.*c1*dq[13]*dv[18]*sig[12] + c1*dq[13]*dv[19]*sig[6] + c1*dq[13]*dv[20]*s*sig[8] + 3.*c1*dq[14]*dv[18]*s*sig[11] + c1*dq[14]*dv[19]*s*sig[8] + c1*dq[14]*dv[20]*(s)*(s)*sig[7] + 3.*c1*dq[18]*dv[13]*sig[12] + 3.*c1*dq[18]*dv[14]*s*sig[11] + c1*dq[19]*dv[13]*sig[6] + c1*dq[19]*dv[14]*s*sig[8] - 3.*c1*dq[1]*sig[12] + c1*dq[20]*dv[13]*s*sig[8] + c1*dq[20]*dv[14]*(s)*(s)*sig[7] - 3.*c1*dq[2]*s*sig[11] - 3.*c1*dq[6]*irho*sig[11] + c1*dq[7]*irho*sig[8] + c1*dq[8]*irho*s*sig[7] + dq[0]*(irho)*(irho)*sig[4] - dq[13]*dv[18]*sig[10] - dq[13]*dv[19]*sig[3] - dq[13]*dv[20]*s*sig[5] - dq[14]*dv[18]*s*sig[9] - dq[14]*dv[19]*s*sig[5] - dq[14]*dv[20]*(s)*(s)*sig[4] - dq[18]*dv[13]*sig[10] - dq[18]*dv[14]*s*sig[9] - dq[19]*dv[13]*sig[3] - dq[19]*dv[14]*s*sig[5] + dq[1]*sig[10] - dq[20]*dv[13]*s*sig[5] - dq[20]*dv[14]*(s)*(s)*sig[4] + dq[2]*s*sig[9] + dq[6]*irho*sig[9] - dq[7]*irho*sig[5] - dq[8]*irho*s*sig[4]
                bs[19] += c1*dq[13]*dv[18]*sig[6] + c1*dq[14]*dv[18]*s*sig[8] + c1*dq[18]*dv[13]*sig[6] + c1*dq[18]*dv[14]*s*sig[8] - c1*dq[1]*sig[6] - c1*dq[2]*s*sig[8] - c1*dq[6]*irho*sig[8] - dq[13]*dv[18]*sig[3] - dq[14]*dv[18]*s*sig[5] - dq[18]*dv[13]*sig[3] - dq[18]*dv[14]*s*sig[5] + dq[1]*sig[3] + dq[2]*s*sig[5] + dq[6]*irho*sig[5]
                bs[20] += c1*dq[13]*dv[18]*s*sig[8] + c1*dq[14]*dv[18]*(s)*(s)*sig[7] + c1*dq[18]*dv[13]*s*sig[8] + c1*dq[18]*dv[14]*(s)*(s)*sig[7] - c1*dq[1]*s*sig[8] - c1*dq[2]*(s)*(s)*sig[7] - c1*dq[6]*irho*s*sig[7] - dq[13]*dv[18]*s*sig[5] - dq[14]*dv[18]*(s)*(s)*sig[4] - dq[18]*dv[13]*s*sig[5] - dq[18]*dv[14]*(s)*(s)*sig[4] + dq[1]*s*sig[5] + dq[2]*(s)*(s)*sig[4] + dq[6]*irho*s*sig[4]
                bs[21] += 3.*c1*dq[0]*irho*sig[11] + 3.*c1*dq[13]*dv[21]*sig[12] + c1*dq[13]*dv[22]*sig[6] + c1*dq[13]*dv[23]*s*sig[8] + 3.*c1*dq[14]*dv[21]*s*sig[11] + c1*dq[14]*dv[22]*s*sig[8] + c1*dq[14]*dv[23]*(s)*(s)*sig[7] - c1*dq[1]*irho*sig[8] + 3.*c1*dq[21]*dv[13]*sig[12] + 3.*c1*dq[21]*dv[14]*s*sig[11] + c1*dq[22]*dv[13]*sig[6] + c1*dq[22]*dv[14]*s*sig[8] + c1*dq[23]*dv[13]*s*sig[8] + c1*dq[23]*dv[14]*(s)*(s)*sig[7] - c1*dq[2]*irho*s*sig[7] - c1*dq[6]*(irho)*(irho)*sig[7] - 3.*c1*dq[7]*sig[12] - 3.*c1*dq[8]*s*sig[11] - dq[0]*irho*sig[9] - dq[13]*dv[21]*sig[10] - dq[13]*dv[22]*sig[3] - dq[13]*dv[23]*s*sig[5] - dq[14]*dv[21]*s*sig[9] - dq[14]*dv[22]*s*sig[5] - dq[14]*dv[23]*(s)*(s)*sig[4] + dq[1]*irho*sig[5] - dq[21]*dv[13]*sig[10] - dq[21]*dv[14]*s*sig[9] - dq[22]*dv[13]*sig[3] - dq[22]*dv[14]*s*sig[5] - dq[23]*dv[13]*s*sig[5] - dq[23]*dv[14]*(s)*(s)*sig[4] + dq[2]*irho*s*sig[4] + dq[6]*(irho)*(irho)*sig[4] + dq[7]*sig[10] + dq[8]*s*sig[9]
                bs[22] += c1*dq[0]*irho*sig[8] + c1*dq[13]*dv[21]*sig[6] + c1*dq[14]*dv[21]*s*sig[8] + c1*dq[21]*dv[13]*sig[6] + c1*dq[21]*dv[14]*s*sig[8] - c1*dq[7]*sig[6] - c1*dq[8]*s*sig[8] - dq[0]*irho*sig[5] - dq[13]*dv[21]*sig[3] - dq[14]*dv[21]*s*sig[5] - dq[21]*dv[13]*sig[3] - dq[21]*dv[14]*s*sig[5] + dq[7]*sig[3] + dq[8]*s*sig[5]
                bs[23] += c1*dq[0]*irho*s*sig[7] + c1*dq[13]*dv[21]*s*sig[8] + c1*dq[14]*dv[21]*(s)*(s)*sig[7] + c1*dq[21]*dv[13]*s*sig[8] + c1*dq[21]*dv[14]*(s)*(s)*sig[7] - c1*dq[7]*s*sig[8] - c1*dq[8]*(s)*(s)*sig[7] - dq[0]*irho*s*sig[4] - dq[13]*dv[21]*s*sig[5] - dq[14]*dv[21]*(s)*(s)*sig[4] - dq[21]*dv[13]*s*sig[5] - dq[21]*dv[14]*(s)*(s)*sig[4] + dq[7]*s*sig[5] + dq[8]*(s)*(s)*sig[4]
                for j in range(n):
                    for i in range(m):
                        row = row0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[2, i]*gtab[0, j]
                        pA[4] = ftab[0, i]*gtab[2, j]
                        pA[5] = ftab[1, i]*gtab[1, j]
                        pA[6] = ftab[3, i]*gtab[3, j]
                        pA[7] = ftab[4, i]*gtab[3, j]
                        pA[8] = ftab[3, i]*gtab[4, j]
                        pA[9] = ftab[5, i]*gtab[3, j]
                        pA[10] = ftab[3, i]*gtab[5, j]
                        pA[11] = ftab[4, i]*gtab[4, j]
                        pA[12] = ftab[6, i]*gtab[6, j]
                        pA[13] = ftab[7, i]*gtab[6, j]
                        pA[14] = ftab[6, i]*gtab[7, j]
                        pA[15] = ftab[8, i]*gtab[6, j]
                        pA[16] = ftab[6, i]*gtab[8, j]
                        pA[17] = ftab[7, i]*gtab[7, j]
                        pA[18] = ftab[9, i]*gtab[9, j]
                        pA[19] = ftab[10, i]*gtab[9, j]
                        pA[20] = ftab[9, i]*gtab[10, j]
                        pA[21] = ftab[12, i]*gtab[12, j]
                        pA[22] = ftab[13, i]*gtab[12, j]
                        pA[23] = ftab[12, i]*gtab[13, j]
                        for e in range(5):
                            for p in range(P0[e], P1[e]):
                                kTv[row + e] += weight*pA[p]*bs[p]
    return np.asarray(kTv)

def _point_derivatives(double [::1] c, object shell, double xp, double yp):
    r"""Generalized derivatives `\boldsymbol{d}` of the panel at a point

    Parameters
    ----------
    c : np.ndarray
        Ritz constants of the panel, 1-D array of size ``5*m*n``.
    shell : :class:`.Shell`
        The annular sector.
    xp, yp : float
        Coordinates `x` and `y` of the point, lengths.

    Returns
    -------
    d : np.ndarray
        ``(u, u_x, u_y, u_xx, u_yy, u_xy, v, v_x, v_y, v_xx, v_yy, v_xy, w, w_x, w_y, w_xx, w_yy, w_xy, phix, phix_x, phix_y, phiy, phiy_x, phiy_y)``, 1-D array of size 24.

    Notes
    -----
    `\boldsymbol{d} = \boldsymbol{\Phi}(x, y) \boldsymbol{c}`,
    with the products of the approximation functions of Bardell along `x`
    and `y` and of their derivatives, see the notes of the module.

    """
    cdef int m, n, i, j, col, p, col0 = 0
    cdef double a, b, xi, eta
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
    cdef double pA[24]
    cdef double dv[24]
    cdef double [:, ::1] ftab, gtab
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
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
    xi = 2*xp/a - 1
    eta = 2*yp/b - 1
    ftab = np.zeros((15, m), dtype=DOUBLE)
    gtab = np.zeros((15, n), dtype=DOUBLE)
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
        ftab[9, i] = f(i, xi, x1phix, x1phixr, x2phix, x2phixr)
        ftab[10, i] = (2/a)*fp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
        ftab[11, i] = (2/a)*(2/a)*fpp(i, xi, x1phix, x1phixr, x2phix, x2phixr)
        ftab[12, i] = f(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
        ftab[13, i] = (2/a)*fp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
        ftab[14, i] = (2/a)*(2/a)*fpp(i, xi, x1phiy, x1phiyr, x2phiy, x2phiyr)
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
        gtab[9, j] = f(j, eta, y1phix, y1phixr, y2phix, y2phixr)
        gtab[10, j] = (2/b)*fp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
        gtab[11, j] = (2/b)*(2/b)*fpp(j, eta, y1phix, y1phixr, y2phix, y2phixr)
        gtab[12, j] = f(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
        gtab[13, j] = (2/b)*fp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
        gtab[14, j] = (2/b)*(2/b)*fpp(j, eta, y1phiy, y1phiyr, y2phiy, y2phiyr)
    for p in range(24):
        dv[p] = 0.
    for j in range(n):
        for i in range(m):
            col = col0 + 5*(j*m + i)
            pA[0] = ftab[0, i]*gtab[0, j]
            pA[1] = ftab[1, i]*gtab[0, j]
            pA[2] = ftab[0, i]*gtab[1, j]
            pA[3] = ftab[2, i]*gtab[0, j]
            pA[4] = ftab[0, i]*gtab[2, j]
            pA[5] = ftab[1, i]*gtab[1, j]
            pA[6] = ftab[3, i]*gtab[3, j]
            pA[7] = ftab[4, i]*gtab[3, j]
            pA[8] = ftab[3, i]*gtab[4, j]
            pA[9] = ftab[5, i]*gtab[3, j]
            pA[10] = ftab[3, i]*gtab[5, j]
            pA[11] = ftab[4, i]*gtab[4, j]
            pA[12] = ftab[6, i]*gtab[6, j]
            pA[13] = ftab[7, i]*gtab[6, j]
            pA[14] = ftab[6, i]*gtab[7, j]
            pA[15] = ftab[8, i]*gtab[6, j]
            pA[16] = ftab[6, i]*gtab[8, j]
            pA[17] = ftab[7, i]*gtab[7, j]
            pA[18] = ftab[9, i]*gtab[9, j]
            pA[19] = ftab[10, i]*gtab[9, j]
            pA[20] = ftab[9, i]*gtab[10, j]
            pA[21] = ftab[12, i]*gtab[12, j]
            pA[22] = ftab[13, i]*gtab[12, j]
            pA[23] = ftab[12, i]*gtab[13, j]
            for p in range(24):
                dv[p] += pA[p]*c[col + FLD[p]]
    return np.array([dv[p] for p in range(24)])


def fstrain(double [::1] c, object s, double [::1] xs, double [::1] ys,
            int num_cores=4, int NLgeom=0):
    r"""Generalized strains of the panel at the points ``xs``, ``ys``

    Evaluates the thirteen generalized strains of the configuration ``c``
    at each point, serially.

    Parameters
    ----------
    c : np.ndarray
        Ritz constants of the panel ``s``, 1-D array of size ``5*s.m*s.n``,
        see :meth:`.Shell.get_size`.
    s : :class:`.Shell`
        The annular sector, whose dimensions, outer radius, thickness,
        numbers of terms and boundary flags are read.
    xs, ys : np.ndarray
        Coordinates `x` and `y` of the points, lengths, 1-D arrays of the
        same size.
    num_cores : int, optional
        Not used, the points are evaluated serially; kept for the interface
        of :mod:`panels.models.fsdt_tsdt_field`.
    NLgeom : int, optional
        ``1`` to include the non-linear terms of the strains, ``0``
        otherwise. Default ``0``.

    Returns
    -------
    strains : tuple of np.ndarray
        The 13 generalized strains along the local axes, in the order of
        :func:`.fsdt_tsdt_field.strain_names`, 1-D arrays of the size of
        ``xs``.

    Notes
    -----
    The strains given in the notes of the module, with the non-linear terms
    multiplied by ``NLgeom``.

    """
    cdef double R, s_, irho, h, c1
    R = s.r
    h = s.lam.h
    c1 = 4./(3.*h*h)
    out = np.zeros((13, xs.shape[0]), dtype=DOUBLE)
    for k in range(xs.shape[0]):
        dv = _point_derivatives(c, s, xs[k], ys[k])
        irho = 1./(R - xs[k])
        s_ = R*irho
        out[0, k] = dv[1] + NLgeom*((dv[13])*(dv[13])/2. + (dv[1])*(dv[1])/2. + (dv[7])*(dv[7])/2.)
        out[1, k] = -dv[0]*irho + dv[8]*s_ + NLgeom*((dv[0])*(dv[0])*(irho)*(irho)/2. - dv[0]*dv[8]*irho*s_ + (dv[14])*(dv[14])*(s_)*(s_)/2. + (dv[2])*(dv[2])*(s_)*(s_)/2. + dv[2]*dv[6]*irho*s_ + (dv[6])*(dv[6])*(irho)*(irho)/2. + (dv[8])*(dv[8])*(s_)*(s_)/2.)
        out[2, k] = dv[2]*s_ + dv[6]*irho + dv[7] + NLgeom*(-dv[0]*dv[7]*irho + dv[13]*dv[14]*s_ + dv[1]*dv[2]*s_ + dv[1]*dv[6]*irho + dv[7]*dv[8]*s_)
        out[3, k] = dv[19] + NLgeom*(-dv[13]*dv[18]*dv[19] - dv[13]*dv[21]*dv[22] + dv[19]*dv[1] + dv[22]*dv[7])
        out[4, k] = -dv[18]*irho + dv[23]*s_ + NLgeom*(dv[0]*dv[18]*(irho)*(irho) - dv[0]*dv[23]*irho*s_ - dv[14]*dv[18]*dv[20]*(s_)*(s_) - dv[14]*dv[21]*dv[23]*(s_)*(s_) - dv[18]*dv[8]*irho*s_ + dv[20]*dv[2]*(s_)*(s_) + dv[20]*dv[6]*irho*s_ + dv[21]*dv[2]*irho*s_ + dv[21]*dv[6]*(irho)*(irho) + dv[23]*dv[8]*(s_)*(s_))
        out[5, k] = dv[20]*s_ + dv[21]*irho + dv[22] + NLgeom*(-dv[0]*dv[22]*irho - dv[13]*dv[18]*dv[20]*s_ - dv[13]*dv[21]*dv[23]*s_ - dv[14]*dv[18]*dv[19]*s_ - dv[14]*dv[21]*dv[22]*s_ - dv[18]*dv[7]*irho + dv[19]*dv[2]*s_ + dv[19]*dv[6]*irho + dv[1]*dv[20]*s_ + dv[1]*dv[21]*irho + dv[22]*dv[8]*s_ + dv[23]*dv[7]*s_)
        out[6, k] = -c1*dv[15] - c1*dv[19] + NLgeom*(-c1*dv[0]*dv[13]*dv[3]*irho + c1*dv[0]*dv[15]*dv[1]*irho + c1*dv[0]*dv[15]*irho + c1*dv[13]*dv[18]*dv[19] + c1*dv[13]*dv[21]*dv[22] - c1*dv[13]*dv[2]*dv[9]*s_ + c1*dv[13]*dv[3]*dv[8]*s_ + c1*dv[13]*dv[3] - c1*dv[13]*dv[6]*dv[9]*irho + c1*dv[14]*dv[1]*dv[9]*s_ - c1*dv[14]*dv[3]*dv[7]*s_ + c1*dv[14]*dv[9]*s_ - c1*dv[15]*dv[1]*dv[8]*s_ - c1*dv[15]*dv[1] + c1*dv[15]*dv[2]*dv[7]*s_ + c1*dv[15]*dv[6]*dv[7]*irho - c1*dv[15]*dv[8]*s_ - c1*dv[19]*dv[1] - c1*dv[22]*dv[7])
        out[7, k] = c1*dv[13]*irho - c1*dv[16]*(s_)*(s_) + c1*dv[18]*irho - c1*dv[23]*s_ + NLgeom*(c1*(dv[0])*(dv[0])*dv[13]*(irho)*(irho)*(irho) - c1*dv[0]*dv[13]*dv[4]*irho*(s_)*(s_) - 3.*c1*dv[0]*dv[13]*dv[8]*(irho)*(irho)*s_ - 2.*c1*dv[0]*dv[13]*(irho)*(irho) + c1*dv[0]*dv[14]*dv[7]*(irho)*(irho)*s_ + c1*dv[0]*dv[16]*dv[1]*irho*(s_)*(s_) + c1*dv[0]*dv[16]*irho*(s_)*(s_) - c1*dv[0]*dv[18]*(irho)*(irho) + c1*dv[0]*dv[23]*irho*s_ - c1*dv[10]*dv[13]*dv[2]*(s_)*(s_)*(s_) - c1*dv[10]*dv[13]*dv[6]*irho*(s_)*(s_) + c1*dv[10]*dv[14]*dv[1]*(s_)*(s_)*(s_) + c1*dv[10]*dv[14]*(s_)*(s_)*(s_) + 2.*c1*dv[13]*(dv[2])*(dv[2])*irho*(s_)*(s_) + 3.*c1*dv[13]*dv[2]*dv[6]*(irho)*(irho)*s_ + c1*dv[13]*dv[4]*dv[8]*(s_)*(s_)*(s_) + c1*dv[13]*dv[4]*(s_)*(s_) + c1*dv[13]*(dv[6])*(dv[6])*(irho)*(irho)*(irho) + 2.*c1*dv[13]*(dv[8])*(dv[8])*irho*(s_)*(s_) + 3.*c1*dv[13]*dv[8]*irho*s_ + c1*dv[14]*dv[18]*dv[20]*(s_)*(s_) - 2.*c1*dv[14]*dv[1]*dv[2]*irho*(s_)*(s_) - c1*dv[14]*dv[1]*dv[6]*(irho)*(irho)*s_ + c1*dv[14]*dv[21]*dv[23]*(s_)*(s_) - 2.*c1*dv[14]*dv[2]*irho*(s_)*(s_) - c1*dv[14]*dv[4]*dv[7]*(s_)*(s_)*(s_) - c1*dv[14]*dv[6]*(irho)*(irho)*s_ - 2.*c1*dv[14]*dv[7]*dv[8]*irho*(s_)*(s_) - c1*dv[14]*dv[7]*irho*s_ - c1*dv[16]*dv[1]*dv[8]*(s_)*(s_)*(s_) - c1*dv[16]*dv[1]*(s_)*(s_) + c1*dv[16]*dv[2]*dv[7]*(s_)*(s_)*(s_) + c1*dv[16]*dv[6]*dv[7]*irho*(s_)*(s_) - c1*dv[16]*dv[8]*(s_)*(s_)*(s_) + c1*dv[18]*dv[8]*irho*s_ - c1*dv[20]*dv[2]*(s_)*(s_) - c1*dv[20]*dv[6]*irho*s_ - c1*dv[21]*dv[2]*irho*s_ - c1*dv[21]*dv[6]*(irho)*(irho) - c1*dv[23]*dv[8]*(s_)*(s_))
        out[8, k] = -2.*c1*dv[14]*irho*s_ - 2.*c1*dv[17]*s_ - c1*dv[20]*s_ - c1*dv[21]*irho - c1*dv[22] + NLgeom*(-2.*c1*dv[0]*dv[13]*dv[5]*irho*s_ - 2.*c1*dv[0]*dv[13]*dv[7]*(irho)*(irho) + 2.*c1*dv[0]*dv[17]*dv[1]*irho*s_ + 2.*c1*dv[0]*dv[17]*irho*s_ + c1*dv[0]*dv[22]*irho - 2.*c1*dv[11]*dv[13]*dv[2]*(s_)*(s_) - 2.*c1*dv[11]*dv[13]*dv[6]*irho*s_ + 2.*c1*dv[11]*dv[14]*dv[1]*(s_)*(s_) + 2.*c1*dv[11]*dv[14]*(s_)*(s_) + c1*dv[13]*dv[18]*dv[20]*s_ + 2.*c1*dv[13]*dv[1]*dv[2]*irho*s_ + 2.*c1*dv[13]*dv[1]*dv[6]*(irho)*(irho) + c1*dv[13]*dv[21]*dv[23]*s_ + 2.*c1*dv[13]*dv[2]*irho*s_ + 2.*c1*dv[13]*dv[5]*dv[8]*(s_)*(s_) + 2.*c1*dv[13]*dv[5]*s_ + 2.*c1*dv[13]*dv[6]*(irho)*(irho) + 2.*c1*dv[13]*dv[7]*dv[8]*irho*s_ + 2.*c1*dv[13]*dv[7]*irho + c1*dv[14]*dv[18]*dv[19]*s_ - 2.*c1*dv[14]*(dv[1])*(dv[1])*irho*s_ - 4.*c1*dv[14]*dv[1]*irho*s_ + c1*dv[14]*dv[21]*dv[22]*s_ - 2.*c1*dv[14]*dv[5]*dv[7]*(s_)*(s_) - 2.*c1*dv[14]*(dv[7])*(dv[7])*irho*s_ - 2.*c1*dv[17]*dv[1]*dv[8]*(s_)*(s_) - 2.*c1*dv[17]*dv[1]*s_ + 2.*c1*dv[17]*dv[2]*dv[7]*(s_)*(s_) + 2.*c1*dv[17]*dv[6]*dv[7]*irho*s_ - 2.*c1*dv[17]*dv[8]*(s_)*(s_) + c1*dv[18]*dv[7]*irho - c1*dv[19]*dv[2]*s_ - c1*dv[19]*dv[6]*irho - c1*dv[1]*dv[20]*s_ - c1*dv[1]*dv[21]*irho - c1*dv[22]*dv[8]*s_ - c1*dv[23]*dv[7]*s_)
        out[9, k] = dv[14]*s_ + dv[21] + NLgeom*(-dv[0]*dv[21]*irho - dv[14]*(dv[18])*(dv[18])*s_/2. - dv[14]*(dv[21])*(dv[21])*s_/2. + dv[18]*dv[2]*s_ + dv[18]*dv[6]*irho + dv[21]*dv[8]*s_)
        out[10, k] = dv[13] + dv[18] + NLgeom*(-dv[13]*(dv[18])*(dv[18])/2. - dv[13]*(dv[21])*(dv[21])/2. + dv[18]*dv[1] + dv[21]*dv[7])
        out[11, k] = -3.*c1*dv[14]*s_ - 3.*c1*dv[21] + NLgeom*(3.*c1*dv[0]*dv[21]*irho + 3.*c1*dv[14]*(dv[18])*(dv[18])*s_/2. + 3.*c1*dv[14]*(dv[21])*(dv[21])*s_/2. - 3.*c1*dv[18]*dv[2]*s_ - 3.*c1*dv[18]*dv[6]*irho - 3.*c1*dv[21]*dv[8]*s_)
        out[12, k] = -3.*c1*dv[13] - 3.*c1*dv[18] + NLgeom*(3.*c1*dv[13]*(dv[18])*(dv[18])/2. + 3.*c1*dv[13]*(dv[21])*(dv[21])/2. - 3.*c1*dv[18]*dv[1] - 3.*c1*dv[21]*dv[7])
    return tuple(out)

def fkAx_num(*args, **kwargs):
    r"""Not available for the annular sector

    The function exists for the interface of the models with the piston
    theory, see :meth:`.Shell.calc_kA`.

    Parameters
    ----------
    *args : object, optional
        Positional arguments, ignored.
    **kwargs : object, optional
        Keyword arguments, ignored.

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('fkAx_num is not implemented for the model '
                              '"annular_tsdt_tl"')

def fkAy_num(*args, **kwargs):
    r"""Not available for the annular sector

    The function exists for the interface of the models with the piston
    theory, see :meth:`.Shell.calc_kA`.

    Parameters
    ----------
    *args : object, optional
        Positional arguments, ignored.
    **kwargs : object, optional
        Keyword arguments, ignored.

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('fkAy_num is not implemented for the model '
                              '"annular_tsdt_tl"')
