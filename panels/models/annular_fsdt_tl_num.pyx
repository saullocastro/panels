#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Annular-sector flat plate, using the total-Lagrangian first-order shear
deformation theory (FSDT) with a second-order director, in polar coordinates

Numerically integrated kernels of the model ``'annular_fsdt_tl'``:
the constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the mass :func:`fkM_num`, the internal force vector
:func:`calc_fint`, the product of the tangent stiffness by a vector
:func:`calc_kTv`, and the fields :func:`fuvw`, :func:`fstrain` and
:func:`fg`, the last one that of :mod:`panels.models.fsdt_tsdt_field`. The
degree of freedom ``k`` of the term ``(i, j)``, ``0 <= i < m``, ``0 <= j <
n``, is at the row ``row0 + 5*(j*m + i) + k`` of the matrices.

The annular sector lies in a plane, between the outer radius ``r``,
``shell.r``, and the inner radius ``r - a``: ``a``, ``shell.a``, is its
radial width and ``b``, ``shell.b``, the arc length of its outer edge. The
coordinate `x = r - \rho` runs from the outer edge inwards, `y = r \theta`
along the outer edge, with the polar coordinates `(\rho, \theta)` of the
plane; ``u``, ``v``, ``w`` are the displacements along
`\boldsymbol{e}_x = -\boldsymbol{e}_\rho`, `\boldsymbol{e}_y =
\boldsymbol{e}_\theta` and `\boldsymbol{e}_z = \boldsymbol{e}_x \times
\boldsymbol{e}_y`, and ``phix``, ``phiy`` the rotations of the normal.
These are the local axes of the flange of the connection ``'BFxcte'``
whose base is a cylinder of radius ``r``: the web of a ring
frame of a stiffened cylinder is an application, with ``a`` the height of
the web, ``b`` the arc length of its outer edge and ``r`` its outer radius.
The laminate spans `d - h/2 \le z \le d + h/2` about the reference plane,
`d` being the offset ``shell.offset``, which enters through the matrix
``ABD`` and the moments of the density. The matrix ``shell.ABD`` of the
first-order shear deformation theory has the transverse shear stiffness in
its last two rows and columns, with the shear correction of
:class:`.Shell`.

Notes
-----
The deformed position of a point is `\boldsymbol{X} + z \boldsymbol{d}`,
with the position `\boldsymbol{X}` of the deformed reference plane and the
second-order director of Schmidt and Reddy [1]_, Eq. (16), and Palmerio,
Reddy and Schmidt [2]_, Eqs. (16) and (20), with an inextensible normal,
`\boldsymbol{d} = \phi_x \boldsymbol{e}_x + \phi_y \boldsymbol{e}_y + (1
- (\phi_x^2 + \phi_y^2)/2) \boldsymbol{e}_z`, as in ``plate_fsdt_tl``. With
`s = r/\rho` and, per unit length of the reference lines and in the local
basis, which turns with `\theta`,

.. math::

    \boldsymbol{g}_1 = \boldsymbol{X}_{,x} = \begin{Bmatrix} 1 + u_{,x} \\
    v_{,x} \\ w_{,x} \end{Bmatrix} \qquad
    \boldsymbol{g}_2 = s \boldsymbol{X}_{,y} = \begin{Bmatrix} s u_{,y} +
    v/\rho \\ 1 + s v_{,y} - u/\rho \\ s w_{,y} \end{Bmatrix}

    \boldsymbol{d}_{,x} = \begin{Bmatrix} \phi_{x,x} \\ \phi_{y,x} \\
    -(\phi_x \phi_{x,x} + \phi_y \phi_{y,x}) \end{Bmatrix} \qquad
    s \boldsymbol{d}_{,y} = \begin{Bmatrix} s \phi_{x,y} + \phi_y/\rho \\
    s \phi_{y,y} - \phi_x/\rho \\ -s (\phi_x \phi_{x,y} + \phi_y
    \phi_{y,y}) \end{Bmatrix}

the strains are the coefficients of `z^0` and `z^1` of the Green-Lagrange
strains of `\boldsymbol{X} + z \boldsymbol{d}`:

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

    \gamma_{yz} = \boldsymbol{g}_2 \cdot \boldsymbol{d} \qquad
    \gamma_{xz} = \boldsymbol{g}_1 \cdot \boldsymbol{d}

The membrane strains are the Green-Lagrange strains of Amabili [3]_, Eqs.
(1.166)-(1.168); the linear part is that of ``annular_fsdt_donnell``, and
for `r \to \infty` the strains are those of ``plate_fsdt_tl``. A finite
rigid rotation `t` changes the strains by `O(t^4)`, from the second-order
director. The second derivatives of the strains with respect to the
generalized derivatives are linear in them. The mass matrix is that of
the linear kinematics, `u(z) = u + z \phi_x`, `v(z) = v + z \phi_y`, `w(z)
= w`.

The element of area is `dA = (\rho/r)\, dx\, dy`. The integrands are
rational functions of `x`, integrated with the Gauss-Legendre rule over the
domain of :meth:`.Shell.integration_limits`.

References
----------
.. [1] Schmidt, R., and Reddy, J. N., A refined small strain and moderate
   rotation theory of elastic anisotropic shells, Journal of Applied
   Mechanics, Vol. 55, No. 3, 1988, pp. 611-617, Eq. (16).
   https://doi.org/10.1115/1.3125837
.. [2] Palmerio, A. F., Reddy, J. N., and Schmidt, R., On a moderate
   rotation theory of laminated anisotropic shells - Part 1. Theory,
   International Journal of Non-Linear Mechanics, Vol. 25, No. 6, 1990,
   pp. 687-700, Eqs. (16) and (20).
   https://doi.org/10.1016/0020-7462(90)90007-V
.. [3] Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
   Cambridge University Press, 2008, Sec. 1.4.
   https://doi.org/10.1017/CBO9780511619694

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
from .fsdt_tsdt_field import fg


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 5
cdef int[5] P0 = [0, 3, 6, 9, 12]
cdef int[5] P1 = [3, 6, 9, 12, 15]
#: degree of freedom of each generalized derivative
cdef int[15] FLD = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]


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
        Constitutive matrix of the first-order shear deformation theory,
        ``[[A, B, 0], [B, D, 0], [0, 0, A_s]]`` with the transverse shear
        stiffness ``A_s``, shape ``(8, 8)`` for the whole domain, e.g.
        ``shell.ABD``, or ``(nx, ny, 8, 8)`` with one matrix per integration
        point.
    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the numbers
        of terms ``m`` and ``n`` and the boundary flags of the approximation
        functions of each field.
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

    with `dA = (\rho/r)\, dx\, dy`, see the notes of the module. With
    ``NLgeom=0``, `\boldsymbol{d} = \boldsymbol{0}` and the matrix is
    `\int \boldsymbol{\Phi}^\top \boldsymbol{B}_0^\top \boldsymbol{F}
    \boldsymbol{B}_0 \boldsymbol{\Phi} \, dA`.

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, intx, inty, xi, eta, xp, rho, s, irho, weight
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
    cdef double pA[15]
    cdef double pB[15]
    cdef double dv[15]
    cdef double K[225]
    cdef double TA[75]
    cdef double Bm[120]
    cdef double FB[120]
    cdef double eps0, eps1, eps2, eps3, eps4, eps5, eps6, eps7
    cdef double epsL0, epsL1, epsL2, epsL3, epsL4, epsL5, epsL6, epsL7
    cdef double Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qy, Qx
    cdef double NxxL, NyyL, NxyL, MxxL, MyyL, MxyL, QyL, QxL

    cdef long [::1] kr, kc
    cdef double [::1] kv
    cdef double F[64]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 8, 8):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (8, 8):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(8):
            for j in range(8):
                F[i*8 + j] = Finput[i, j]
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
                    for i in range(8):
                        for j in range(8):
                            F[i*8 + j] = Fnxny[ptx, pty, i, j]
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

                for p in range(120):
                    Bm[p] = 0.
                for p in range(15):
                    dv[p] = 0.
                if NLgeom == 1:
                    for p in range(15):
                        dv[p] = 0.
                    for j in range(n):
                        for i in range(m):
                            col = col0 + 5*(j*m + i)
                            pA[0] = ftab[0, i]*gtab[0, j]
                            pA[1] = ftab[1, i]*gtab[0, j]
                            pA[2] = ftab[0, i]*gtab[1, j]
                            pA[3] = ftab[3, i]*gtab[3, j]
                            pA[4] = ftab[4, i]*gtab[3, j]
                            pA[5] = ftab[3, i]*gtab[4, j]
                            pA[6] = ftab[6, i]*gtab[6, j]
                            pA[7] = ftab[7, i]*gtab[6, j]
                            pA[8] = ftab[6, i]*gtab[7, j]
                            pA[9] = ftab[9, i]*gtab[9, j]
                            pA[10] = ftab[10, i]*gtab[9, j]
                            pA[11] = ftab[9, i]*gtab[10, j]
                            pA[12] = ftab[12, i]*gtab[12, j]
                            pA[13] = ftab[13, i]*gtab[12, j]
                            pA[14] = ftab[12, i]*gtab[13, j]
                            for p in range(15):
                                dv[p] += pA[p]*cs[col + FLD[p]]
                Bm[1] = dv[1] + 1.
                Bm[4] = dv[4]
                Bm[7] = dv[7]
                Bm[15] = dv[0]*(irho)*(irho) - dv[5]*irho*s - irho
                Bm[17] = dv[2]*(s)*(s) + dv[3]*irho*s
                Bm[18] = dv[2]*irho*s + dv[3]*(irho)*(irho)
                Bm[20] = -dv[0]*irho*s + dv[5]*(s)*(s) + s
                Bm[23] = dv[8]*(s)*(s)
                Bm[30] = -dv[4]*irho
                Bm[31] = dv[2]*s + dv[3]*irho
                Bm[32] = dv[1]*s + s
                Bm[33] = dv[1]*irho + irho
                Bm[34] = -dv[0]*irho + dv[5]*s + 1.
                Bm[35] = dv[4]*s
                Bm[37] = dv[8]*s
                Bm[38] = dv[7]*s
                Bm[46] = dv[10]
                Bm[49] = dv[13]
                Bm[52] = -dv[10]*dv[9] - dv[12]*dv[13]
                Bm[54] = -dv[10]*dv[7]
                Bm[55] = dv[1] - dv[7]*dv[9] + 1.
                Bm[57] = -dv[13]*dv[7]
                Bm[58] = -dv[12]*dv[7] + dv[4]
                Bm[60] = -dv[14]*irho*s + dv[9]*(irho)*(irho)
                Bm[62] = dv[11]*(s)*(s) + dv[12]*irho*s
                Bm[63] = dv[11]*irho*s + dv[12]*(irho)*(irho)
                Bm[65] = dv[14]*(s)*(s) - dv[9]*irho*s
                Bm[68] = -dv[11]*dv[9]*(s)*(s) - dv[12]*dv[14]*(s)*(s)
                Bm[69] = dv[0]*(irho)*(irho) - dv[11]*dv[8]*(s)*(s) - dv[5]*irho*s - irho
                Bm[71] = dv[2]*(s)*(s) + dv[3]*irho*s - dv[8]*dv[9]*(s)*(s)
                Bm[72] = -dv[14]*dv[8]*(s)*(s) + dv[2]*irho*s + dv[3]*(irho)*(irho)
                Bm[74] = -dv[0]*irho*s - dv[12]*dv[8]*(s)*(s) + dv[5]*(s)*(s) + s
                Bm[75] = -dv[13]*irho
                Bm[76] = dv[11]*s + dv[12]*irho
                Bm[77] = dv[10]*s
                Bm[78] = dv[10]*irho
                Bm[79] = dv[14]*s - dv[9]*irho
                Bm[80] = dv[13]*s
                Bm[82] = -dv[11]*dv[9]*s - dv[12]*dv[14]*s
                Bm[83] = -dv[10]*dv[9]*s - dv[12]*dv[13]*s
                Bm[84] = -dv[10]*dv[8]*s - dv[11]*dv[7]*s - dv[4]*irho
                Bm[85] = dv[2]*s + dv[3]*irho - dv[8]*dv[9]*s
                Bm[86] = dv[1]*s - dv[7]*dv[9]*s + s
                Bm[87] = -dv[13]*dv[8]*s - dv[14]*dv[7]*s + dv[1]*irho + irho
                Bm[88] = -dv[0]*irho - dv[12]*dv[8]*s + dv[5]*s + 1.
                Bm[89] = -dv[12]*dv[7]*s + dv[4]*s
                Bm[90] = -dv[12]*irho
                Bm[92] = dv[9]*s
                Bm[93] = dv[9]*irho
                Bm[95] = dv[12]*s
                Bm[98] = -(dv[12])*(dv[12])*s/2. - (dv[9])*(dv[9])*s/2. + s
                Bm[99] = dv[2]*s + dv[3]*irho - dv[8]*dv[9]*s
                Bm[102] = -dv[0]*irho - dv[12]*dv[8]*s + dv[5]*s + 1.
                Bm[106] = dv[9]
                Bm[109] = dv[12]
                Bm[112] = -(dv[12])*(dv[12])/2. - (dv[9])*(dv[9])/2. + 1.
                Bm[114] = dv[1] - dv[7]*dv[9] + 1.
                Bm[117] = -dv[12]*dv[7] + dv[4]
                # K = [B]^T [F] [B]
                for p in range(8):
                    for q in range(15):
                        FB[p*15 + q] = 0.
                        for r in range(8):
                            FB[p*15 + q] += F[p*8 + r]*Bm[r*15 + q]
                for p in range(15):
                    for q in range(15):
                        K[p*15 + q] = 0.
                        for r in range(8):
                            K[p*15 + q] += Bm[r*15 + p]*FB[r*15 + q]
                if NLgeom == 1:
                    # + [H(sigma, d)] - [H(sigma_L, 0)]
                    eps0 = (dv[1])*(dv[1])/2. + dv[1] + (dv[4])*(dv[4])/2. + (dv[7])*(dv[7])/2.
                    eps1 = (dv[0])*(dv[0])*(irho)*(irho)/2. - dv[0]*dv[5]*irho*s - dv[0]*irho + (dv[2])*(dv[2])*(s)*(s)/2. + dv[2]*dv[3]*irho*s + (dv[3])*(dv[3])*(irho)*(irho)/2. + (dv[5])*(dv[5])*(s)*(s)/2. + dv[5]*s + (dv[8])*(dv[8])*(s)*(s)/2.
                    eps2 = -dv[0]*dv[4]*irho + dv[1]*dv[2]*s + dv[1]*dv[3]*irho + dv[2]*s + dv[3]*irho + dv[4]*dv[5]*s + dv[4] + dv[7]*dv[8]*s
                    eps3 = dv[10]*dv[1] - dv[10]*dv[7]*dv[9] + dv[10] - dv[12]*dv[13]*dv[7] + dv[13]*dv[4]
                    eps4 = -dv[0]*dv[14]*irho*s + dv[0]*dv[9]*(irho)*(irho) + dv[11]*dv[2]*(s)*(s) + dv[11]*dv[3]*irho*s - dv[11]*dv[8]*dv[9]*(s)*(s) - dv[12]*dv[14]*dv[8]*(s)*(s) + dv[12]*dv[2]*irho*s + dv[12]*dv[3]*(irho)*(irho) + dv[14]*dv[5]*(s)*(s) + dv[14]*s - dv[5]*dv[9]*irho*s - dv[9]*irho
                    eps5 = -dv[0]*dv[13]*irho + dv[10]*dv[2]*s + dv[10]*dv[3]*irho - dv[10]*dv[8]*dv[9]*s + dv[11]*dv[1]*s - dv[11]*dv[7]*dv[9]*s + dv[11]*s - dv[12]*dv[13]*dv[8]*s - dv[12]*dv[14]*dv[7]*s + dv[12]*dv[1]*irho + dv[12]*irho + dv[13]*dv[5]*s + dv[13] + dv[14]*dv[4]*s - dv[4]*dv[9]*irho
                    eps6 = -dv[0]*dv[12]*irho - (dv[12])*(dv[12])*dv[8]*s/2. + dv[12]*dv[5]*s + dv[12] + dv[2]*dv[9]*s + dv[3]*dv[9]*irho - dv[8]*(dv[9])*(dv[9])*s/2. + dv[8]*s
                    eps7 = -(dv[12])*(dv[12])*dv[7]/2. + dv[12]*dv[4] + dv[1]*dv[9] - dv[7]*(dv[9])*(dv[9])/2. + dv[7] + dv[9]
                    epsL0 = dv[1]
                    epsL1 = -dv[0]*irho + dv[5]*s
                    epsL2 = dv[2]*s + dv[3]*irho + dv[4]
                    epsL3 = dv[10]
                    epsL4 = dv[14]*s - dv[9]*irho
                    epsL5 = dv[11]*s + dv[12]*irho + dv[13]
                    epsL6 = dv[12] + dv[8]*s
                    epsL7 = dv[7] + dv[9]
                    Nxx = F[0]*eps0 + F[1]*eps1 + F[2]*eps2 + F[3]*eps3 + F[4]*eps4 + F[5]*eps5 + F[6]*eps6 + F[7]*eps7
                    Nyy = F[8]*eps0 + F[9]*eps1 + F[10]*eps2 + F[11]*eps3 + F[12]*eps4 + F[13]*eps5 + F[14]*eps6 + F[15]*eps7
                    Nxy = F[16]*eps0 + F[17]*eps1 + F[18]*eps2 + F[19]*eps3 + F[20]*eps4 + F[21]*eps5 + F[22]*eps6 + F[23]*eps7
                    Mxx = F[24]*eps0 + F[25]*eps1 + F[26]*eps2 + F[27]*eps3 + F[28]*eps4 + F[29]*eps5 + F[30]*eps6 + F[31]*eps7
                    Myy = F[32]*eps0 + F[33]*eps1 + F[34]*eps2 + F[35]*eps3 + F[36]*eps4 + F[37]*eps5 + F[38]*eps6 + F[39]*eps7
                    Mxy = F[40]*eps0 + F[41]*eps1 + F[42]*eps2 + F[43]*eps3 + F[44]*eps4 + F[45]*eps5 + F[46]*eps6 + F[47]*eps7
                    Qy = F[48]*eps0 + F[49]*eps1 + F[50]*eps2 + F[51]*eps3 + F[52]*eps4 + F[53]*eps5 + F[54]*eps6 + F[55]*eps7
                    Qx = F[56]*eps0 + F[57]*eps1 + F[58]*eps2 + F[59]*eps3 + F[60]*eps4 + F[61]*eps5 + F[62]*eps6 + F[63]*eps7
                    NxxL = F[0]*epsL0 + F[1]*epsL1 + F[2]*epsL2 + F[3]*epsL3 + F[4]*epsL4 + F[5]*epsL5 + F[6]*epsL6 + F[7]*epsL7
                    NyyL = F[8]*epsL0 + F[9]*epsL1 + F[10]*epsL2 + F[11]*epsL3 + F[12]*epsL4 + F[13]*epsL5 + F[14]*epsL6 + F[15]*epsL7
                    NxyL = F[16]*epsL0 + F[17]*epsL1 + F[18]*epsL2 + F[19]*epsL3 + F[20]*epsL4 + F[21]*epsL5 + F[22]*epsL6 + F[23]*epsL7
                    MxxL = F[24]*epsL0 + F[25]*epsL1 + F[26]*epsL2 + F[27]*epsL3 + F[28]*epsL4 + F[29]*epsL5 + F[30]*epsL6 + F[31]*epsL7
                    MyyL = F[32]*epsL0 + F[33]*epsL1 + F[34]*epsL2 + F[35]*epsL3 + F[36]*epsL4 + F[37]*epsL5 + F[38]*epsL6 + F[39]*epsL7
                    MxyL = F[40]*epsL0 + F[41]*epsL1 + F[42]*epsL2 + F[43]*epsL3 + F[44]*epsL4 + F[45]*epsL5 + F[46]*epsL6 + F[47]*epsL7
                    QyL = F[48]*epsL0 + F[49]*epsL1 + F[50]*epsL2 + F[51]*epsL3 + F[52]*epsL4 + F[53]*epsL5 + F[54]*epsL6 + F[55]*epsL7
                    QxL = F[56]*epsL0 + F[57]*epsL1 + F[58]*epsL2 + F[59]*epsL3 + F[60]*epsL4 + F[61]*epsL5 + F[62]*epsL6 + F[63]*epsL7
                    K[0] += Nyy*(irho)*(irho)
                    K[4] += -Nxy*irho
                    K[5] += -Nyy*irho*s
                    K[9] += Myy*(irho)*(irho)
                    K[12] += -Qy*irho
                    K[13] += -Mxy*irho
                    K[14] += -Myy*irho*s
                    K[16] += Nxx
                    K[17] += Nxy*s
                    K[18] += Nxy*irho
                    K[24] += Qx
                    K[25] += Mxx
                    K[26] += Mxy*s
                    K[27] += Mxy*irho
                    K[31] += Nxy*s
                    K[32] += Nyy*(s)*(s)
                    K[33] += Nyy*irho*s
                    K[39] += Qy*s
                    K[40] += Mxy*s
                    K[41] += Myy*(s)*(s)
                    K[42] += Myy*irho*s
                    K[46] += Nxy*irho
                    K[47] += Nyy*irho*s
                    K[48] += Nyy*(irho)*(irho)
                    K[54] += Qy*irho
                    K[55] += Mxy*irho
                    K[56] += Myy*irho*s
                    K[57] += Myy*(irho)*(irho)
                    K[60] += -Nxy*irho
                    K[64] += Nxx
                    K[65] += Nxy*s
                    K[69] += -Mxy*irho
                    K[72] += Qx
                    K[73] += Mxx
                    K[74] += Mxy*s
                    K[75] += -Nyy*irho*s
                    K[79] += Nxy*s
                    K[80] += Nyy*(s)*(s)
                    K[84] += -Myy*irho*s
                    K[87] += Qy*s
                    K[88] += Mxy*s
                    K[89] += Myy*(s)*(s)
                    K[112] += Nxx
                    K[113] += Nxy*s
                    K[114] += -Mxx*dv[10] - Mxy*dv[11]*s - Qx*dv[9]
                    K[115] += -Mxx*dv[9]
                    K[116] += -Mxy*dv[9]*s
                    K[117] += -Mxx*dv[13] - Mxy*dv[14]*s - Qx*dv[12]
                    K[118] += -Mxx*dv[12]
                    K[119] += -Mxy*dv[12]*s
                    K[127] += Nxy*s
                    K[128] += Nyy*(s)*(s)
                    K[129] += -Mxy*dv[10]*s - Myy*dv[11]*(s)*(s) - Qy*dv[9]*s
                    K[130] += -Mxy*dv[9]*s
                    K[131] += -Myy*dv[9]*(s)*(s)
                    K[132] += -Mxy*dv[13]*s - Myy*dv[14]*(s)*(s) - Qy*dv[12]*s
                    K[133] += -Mxy*dv[12]*s
                    K[134] += -Myy*dv[12]*(s)*(s)
                    K[135] += Myy*(irho)*(irho)
                    K[136] += Qx
                    K[137] += Qy*s
                    K[138] += Qy*irho
                    K[139] += -Mxy*irho
                    K[140] += -Myy*irho*s
                    K[142] += -Mxx*dv[10] - Mxy*dv[11]*s - Qx*dv[9]
                    K[143] += -Mxy*dv[10]*s - Myy*dv[11]*(s)*(s) - Qy*dv[9]*s
                    K[144] += -Qx*dv[7] - Qy*dv[8]*s
                    K[145] += -Mxx*dv[7] - Mxy*dv[8]*s
                    K[146] += -Mxy*dv[7]*s - Myy*dv[8]*(s)*(s)
                    K[151] += Mxx
                    K[152] += Mxy*s
                    K[153] += Mxy*irho
                    K[157] += -Mxx*dv[9]
                    K[158] += -Mxy*dv[9]*s
                    K[159] += -Mxx*dv[7] - Mxy*dv[8]*s
                    K[166] += Mxy*s
                    K[167] += Myy*(s)*(s)
                    K[168] += Myy*irho*s
                    K[172] += -Mxy*dv[9]*s
                    K[173] += -Myy*dv[9]*(s)*(s)
                    K[174] += -Mxy*dv[7]*s - Myy*dv[8]*(s)*(s)
                    K[180] += -Qy*irho
                    K[181] += Mxy*irho
                    K[182] += Myy*irho*s
                    K[183] += Myy*(irho)*(irho)
                    K[184] += Qx
                    K[185] += Qy*s
                    K[187] += -Mxx*dv[13] - Mxy*dv[14]*s - Qx*dv[12]
                    K[188] += -Mxy*dv[13]*s - Myy*dv[14]*(s)*(s) - Qy*dv[12]*s
                    K[192] += -Qx*dv[7] - Qy*dv[8]*s
                    K[193] += -Mxx*dv[7] - Mxy*dv[8]*s
                    K[194] += -Mxy*dv[7]*s - Myy*dv[8]*(s)*(s)
                    K[195] += -Mxy*irho
                    K[199] += Mxx
                    K[200] += Mxy*s
                    K[202] += -Mxx*dv[12]
                    K[203] += -Mxy*dv[12]*s
                    K[207] += -Mxx*dv[7] - Mxy*dv[8]*s
                    K[210] += -Myy*irho*s
                    K[214] += Mxy*s
                    K[215] += Myy*(s)*(s)
                    K[217] += -Mxy*dv[12]*s
                    K[218] += -Myy*dv[12]*(s)*(s)
                    K[222] += -Mxy*dv[7]*s - Myy*dv[8]*(s)*(s)
                    K[0] -= NyyL*(irho)*(irho)
                    K[4] -= -NxyL*irho
                    K[5] -= -NyyL*irho*s
                    K[9] -= MyyL*(irho)*(irho)
                    K[12] -= -QyL*irho
                    K[13] -= -MxyL*irho
                    K[14] -= -MyyL*irho*s
                    K[16] -= NxxL
                    K[17] -= NxyL*s
                    K[18] -= NxyL*irho
                    K[24] -= QxL
                    K[25] -= MxxL
                    K[26] -= MxyL*s
                    K[27] -= MxyL*irho
                    K[31] -= NxyL*s
                    K[32] -= NyyL*(s)*(s)
                    K[33] -= NyyL*irho*s
                    K[39] -= QyL*s
                    K[40] -= MxyL*s
                    K[41] -= MyyL*(s)*(s)
                    K[42] -= MyyL*irho*s
                    K[46] -= NxyL*irho
                    K[47] -= NyyL*irho*s
                    K[48] -= NyyL*(irho)*(irho)
                    K[54] -= QyL*irho
                    K[55] -= MxyL*irho
                    K[56] -= MyyL*irho*s
                    K[57] -= MyyL*(irho)*(irho)
                    K[60] -= -NxyL*irho
                    K[64] -= NxxL
                    K[65] -= NxyL*s
                    K[69] -= -MxyL*irho
                    K[72] -= QxL
                    K[73] -= MxxL
                    K[74] -= MxyL*s
                    K[75] -= -NyyL*irho*s
                    K[79] -= NxyL*s
                    K[80] -= NyyL*(s)*(s)
                    K[84] -= -MyyL*irho*s
                    K[87] -= QyL*s
                    K[88] -= MxyL*s
                    K[89] -= MyyL*(s)*(s)
                    K[112] -= NxxL
                    K[113] -= NxyL*s
                    K[127] -= NxyL*s
                    K[128] -= NyyL*(s)*(s)
                    K[135] -= MyyL*(irho)*(irho)
                    K[136] -= QxL
                    K[137] -= QyL*s
                    K[138] -= QyL*irho
                    K[139] -= -MxyL*irho
                    K[140] -= -MyyL*irho*s
                    K[151] -= MxxL
                    K[152] -= MxyL*s
                    K[153] -= MxyL*irho
                    K[166] -= MxyL*s
                    K[167] -= MyyL*(s)*(s)
                    K[168] -= MyyL*irho*s
                    K[180] -= -QyL*irho
                    K[181] -= MxyL*irho
                    K[182] -= MyyL*irho*s
                    K[183] -= MyyL*(irho)*(irho)
                    K[184] -= QxL
                    K[185] -= QyL*s
                    K[195] -= -MxyL*irho
                    K[199] -= MxxL
                    K[200] -= MxyL*s
                    K[210] -= -MyyL*irho*s
                    K[214] -= MxyL*s
                    K[215] -= MyyL*(s)*(s)
                c = -1
                for j in range(n):
                    for i in range(m):
                        row = row0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[3, i]*gtab[3, j]
                        pA[4] = ftab[4, i]*gtab[3, j]
                        pA[5] = ftab[3, i]*gtab[4, j]
                        pA[6] = ftab[6, i]*gtab[6, j]
                        pA[7] = ftab[7, i]*gtab[6, j]
                        pA[8] = ftab[6, i]*gtab[7, j]
                        pA[9] = ftab[9, i]*gtab[9, j]
                        pA[10] = ftab[10, i]*gtab[9, j]
                        pA[11] = ftab[9, i]*gtab[10, j]
                        pA[12] = ftab[12, i]*gtab[12, j]
                        pA[13] = ftab[13, i]*gtab[12, j]
                        pA[14] = ftab[12, i]*gtab[13, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(5):
                            for q in range(15):
                                TA[e*15 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*15 + q] += pA[p]*K[p*15 + q]
                        for l in range(n):
                            for k in range(m):
                                col = col0 + 5*(l*m + k)
                                #NOTE upper triangle, the caller symmetrizes
                                if row > col:
                                    continue
                                pB[0] = ftab[0, k]*gtab[0, l]
                                pB[1] = ftab[1, k]*gtab[0, l]
                                pB[2] = ftab[0, k]*gtab[1, l]
                                pB[3] = ftab[3, k]*gtab[3, l]
                                pB[4] = ftab[4, k]*gtab[3, l]
                                pB[5] = ftab[3, k]*gtab[4, l]
                                pB[6] = ftab[6, k]*gtab[6, l]
                                pB[7] = ftab[7, k]*gtab[6, l]
                                pB[8] = ftab[6, k]*gtab[7, l]
                                pB[9] = ftab[9, k]*gtab[9, l]
                                pB[10] = ftab[10, k]*gtab[9, l]
                                pB[11] = ftab[9, k]*gtab[10, l]
                                pB[12] = ftab[12, k]*gtab[12, l]
                                pB[13] = ftab[13, k]*gtab[12, l]
                                pB[14] = ftab[12, k]*gtab[13, l]
                                for e in range(5):
                                    for f2 in range(5):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*15 + q]*pB[q]

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
        units of the displacements.
    Finput : array-like
        Constitutive matrix of the first-order shear deformation theory,
        ``[[A, B, 0], [B, D, 0], [0, 0, A_s]]`` with the transverse shear
        stiffness ``A_s``, shape ``(8, 8)`` for the whole domain, e.g.
        ``shell.ABD``, or ``(nx, ny, 8, 8)`` with one matrix per integration
        point.
    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the numbers
        of terms ``m`` and ``n`` and the boundary flags of the approximation
        functions of each field.
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
    of the linear strains of ``cs`` and `\boldsymbol{\sigma}_0 = \{N_{xx0},
    N_{yy0}, N_{xy0}, 0, 0, 0, 0, 0\}^\top`:

    .. math::

        \boldsymbol{K}_G = \int \boldsymbol{\Phi}^\top
        \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{0})
        \boldsymbol{\Phi} \, dA

    with `\boldsymbol{H}` of :func:`fkC_num`, the variation of the work of
    the stresses on the second-order terms of the strains, see the notes of
    the module.

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, intx, inty, xi, eta, xp, rho, s, irho, weight
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
    cdef double pA[15]
    cdef double pB[15]
    cdef double dv[15]
    cdef double K[225]
    cdef double TA[75]
    cdef double epsL0, epsL1, epsL2, epsL3, epsL4, epsL5, epsL6, epsL7
    cdef double NxxL, NyyL, NxyL, MxxL, MyyL, MxyL, QyL, QxL

    cdef long [::1] kr, kc
    cdef double [::1] kv
    cdef double F[64]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 8, 8):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (8, 8):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(8):
            for j in range(8):
                F[i*8 + j] = Finput[i, j]
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
                    for i in range(8):
                        for j in range(8):
                            F[i*8 + j] = Fnxny[ptx, pty, i, j]
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

                for p in range(15):
                    dv[p] = 0.
                for j in range(n):
                    for i in range(m):
                        col = col0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[3, i]*gtab[3, j]
                        pA[4] = ftab[4, i]*gtab[3, j]
                        pA[5] = ftab[3, i]*gtab[4, j]
                        pA[6] = ftab[6, i]*gtab[6, j]
                        pA[7] = ftab[7, i]*gtab[6, j]
                        pA[8] = ftab[6, i]*gtab[7, j]
                        pA[9] = ftab[9, i]*gtab[9, j]
                        pA[10] = ftab[10, i]*gtab[9, j]
                        pA[11] = ftab[9, i]*gtab[10, j]
                        pA[12] = ftab[12, i]*gtab[12, j]
                        pA[13] = ftab[13, i]*gtab[12, j]
                        pA[14] = ftab[12, i]*gtab[13, j]
                        for p in range(15):
                            dv[p] += pA[p]*cs[col + FLD[p]]
                epsL0 = dv[1]
                epsL1 = -dv[0]*irho + dv[5]*s
                epsL2 = dv[2]*s + dv[3]*irho + dv[4]
                epsL3 = dv[10]
                epsL4 = dv[14]*s - dv[9]*irho
                epsL5 = dv[11]*s + dv[12]*irho + dv[13]
                epsL6 = dv[12] + dv[8]*s
                epsL7 = dv[7] + dv[9]
                NxxL = F[0]*epsL0 + F[1]*epsL1 + F[2]*epsL2 + F[3]*epsL3 + F[4]*epsL4 + F[5]*epsL5 + F[6]*epsL6 + F[7]*epsL7
                NyyL = F[8]*epsL0 + F[9]*epsL1 + F[10]*epsL2 + F[11]*epsL3 + F[12]*epsL4 + F[13]*epsL5 + F[14]*epsL6 + F[15]*epsL7
                NxyL = F[16]*epsL0 + F[17]*epsL1 + F[18]*epsL2 + F[19]*epsL3 + F[20]*epsL4 + F[21]*epsL5 + F[22]*epsL6 + F[23]*epsL7
                MxxL = F[24]*epsL0 + F[25]*epsL1 + F[26]*epsL2 + F[27]*epsL3 + F[28]*epsL4 + F[29]*epsL5 + F[30]*epsL6 + F[31]*epsL7
                MyyL = F[32]*epsL0 + F[33]*epsL1 + F[34]*epsL2 + F[35]*epsL3 + F[36]*epsL4 + F[37]*epsL5 + F[38]*epsL6 + F[39]*epsL7
                MxyL = F[40]*epsL0 + F[41]*epsL1 + F[42]*epsL2 + F[43]*epsL3 + F[44]*epsL4 + F[45]*epsL5 + F[46]*epsL6 + F[47]*epsL7
                QyL = F[48]*epsL0 + F[49]*epsL1 + F[50]*epsL2 + F[51]*epsL3 + F[52]*epsL4 + F[53]*epsL5 + F[54]*epsL6 + F[55]*epsL7
                QxL = F[56]*epsL0 + F[57]*epsL1 + F[58]*epsL2 + F[59]*epsL3 + F[60]*epsL4 + F[61]*epsL5 + F[62]*epsL6 + F[63]*epsL7
                NxxL = NxxL + Nxx0
                NyyL = NyyL + Nyy0
                NxyL = NxyL + Nxy0
                for p in range(225):
                    K[p] = 0.
                K[0] += NyyL*(irho)*(irho)
                K[4] += -NxyL*irho
                K[5] += -NyyL*irho*s
                K[9] += MyyL*(irho)*(irho)
                K[12] += -QyL*irho
                K[13] += -MxyL*irho
                K[14] += -MyyL*irho*s
                K[16] += NxxL
                K[17] += NxyL*s
                K[18] += NxyL*irho
                K[24] += QxL
                K[25] += MxxL
                K[26] += MxyL*s
                K[27] += MxyL*irho
                K[31] += NxyL*s
                K[32] += NyyL*(s)*(s)
                K[33] += NyyL*irho*s
                K[39] += QyL*s
                K[40] += MxyL*s
                K[41] += MyyL*(s)*(s)
                K[42] += MyyL*irho*s
                K[46] += NxyL*irho
                K[47] += NyyL*irho*s
                K[48] += NyyL*(irho)*(irho)
                K[54] += QyL*irho
                K[55] += MxyL*irho
                K[56] += MyyL*irho*s
                K[57] += MyyL*(irho)*(irho)
                K[60] += -NxyL*irho
                K[64] += NxxL
                K[65] += NxyL*s
                K[69] += -MxyL*irho
                K[72] += QxL
                K[73] += MxxL
                K[74] += MxyL*s
                K[75] += -NyyL*irho*s
                K[79] += NxyL*s
                K[80] += NyyL*(s)*(s)
                K[84] += -MyyL*irho*s
                K[87] += QyL*s
                K[88] += MxyL*s
                K[89] += MyyL*(s)*(s)
                K[112] += NxxL
                K[113] += NxyL*s
                K[127] += NxyL*s
                K[128] += NyyL*(s)*(s)
                K[135] += MyyL*(irho)*(irho)
                K[136] += QxL
                K[137] += QyL*s
                K[138] += QyL*irho
                K[139] += -MxyL*irho
                K[140] += -MyyL*irho*s
                K[151] += MxxL
                K[152] += MxyL*s
                K[153] += MxyL*irho
                K[166] += MxyL*s
                K[167] += MyyL*(s)*(s)
                K[168] += MyyL*irho*s
                K[180] += -QyL*irho
                K[181] += MxyL*irho
                K[182] += MyyL*irho*s
                K[183] += MyyL*(irho)*(irho)
                K[184] += QxL
                K[185] += QyL*s
                K[195] += -MxyL*irho
                K[199] += MxxL
                K[200] += MxyL*s
                K[210] += -MyyL*irho*s
                K[214] += MxyL*s
                K[215] += MyyL*(s)*(s)
                c = -1
                for j in range(n):
                    for i in range(m):
                        row = row0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[3, i]*gtab[3, j]
                        pA[4] = ftab[4, i]*gtab[3, j]
                        pA[5] = ftab[3, i]*gtab[4, j]
                        pA[6] = ftab[6, i]*gtab[6, j]
                        pA[7] = ftab[7, i]*gtab[6, j]
                        pA[8] = ftab[6, i]*gtab[7, j]
                        pA[9] = ftab[9, i]*gtab[9, j]
                        pA[10] = ftab[10, i]*gtab[9, j]
                        pA[11] = ftab[9, i]*gtab[10, j]
                        pA[12] = ftab[12, i]*gtab[12, j]
                        pA[13] = ftab[13, i]*gtab[12, j]
                        pA[14] = ftab[12, i]*gtab[13, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(5):
                            for q in range(15):
                                TA[e*15 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*15 + q] += pA[p]*K[p*15 + q]
                        for l in range(n):
                            for k in range(m):
                                col = col0 + 5*(l*m + k)
                                #NOTE upper triangle, the caller symmetrizes
                                if row > col:
                                    continue
                                pB[0] = ftab[0, k]*gtab[0, l]
                                pB[1] = ftab[1, k]*gtab[0, l]
                                pB[2] = ftab[0, k]*gtab[1, l]
                                pB[3] = ftab[3, k]*gtab[3, l]
                                pB[4] = ftab[4, k]*gtab[3, l]
                                pB[5] = ftab[3, k]*gtab[4, l]
                                pB[6] = ftab[6, k]*gtab[6, l]
                                pB[7] = ftab[7, k]*gtab[6, l]
                                pB[8] = ftab[6, k]*gtab[7, l]
                                pB[9] = ftab[9, k]*gtab[9, l]
                                pB[10] = ftab[10, k]*gtab[9, l]
                                pB[11] = ftab[9, k]*gtab[10, l]
                                pB[12] = ftab[12, k]*gtab[12, l]
                                pB[13] = ftab[13, k]*gtab[12, l]
                                pB[14] = ftab[12, k]*gtab[13, l]
                                for e in range(5):
                                    for f2 in range(5):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*15 + q]*pB[q]

    return coo_matrix((kv, (kr, kc)), shape=(size, size))

def fkM_num(object shell, object I_input, int size,
        int row0, int col0, int nx, int ny):
    r"""Mass matrix, integrated numerically

    Parameters
    ----------
    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the numbers
        of terms ``m`` and ``n`` and the boundary flags of the approximation
        functions of each field.
    I_input : array-like
        Moments of the density through the thickness, ``I0, I1, I2``, shape
        ``(3,)`` for the whole domain or ``(nx, ny, 3)`` with one set per
        integration point.
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
    With the displacement through the thickness of the first-order shear
    deformation theory, `u(z) = u + z \phi_x`, `v(z) = v + z \phi_y`, `w(z)
    = w`, `z` measured from the reference plane, and the moments `I_n = \int
    \rho_m z^n dz`, `n = 0, 1, 2`, of the density `\rho_m` over `d - h/2
    \le z \le d + h/2`, with the offset `d` (``shell.offset``):

    .. math::

        \boldsymbol{M} = \int \boldsymbol{\Phi}^\top \boldsymbol{J}^\top
        \begin{bmatrix} I_0 \boldsymbol{I}_3 & I_1 \boldsymbol{E} \\
        I_1 \boldsymbol{E}^\top & I_2 \boldsymbol{I}_2 \end{bmatrix}
        \boldsymbol{J} \boldsymbol{\Phi} \, dA, \qquad
        \boldsymbol{J} \boldsymbol{d} = \{u, v, w, \phi_x,
        \phi_y\}^\top

    with `\boldsymbol{E}` the first two columns of `\boldsymbol{I}_3` and
    `dA = (\rho/r)\, dx\, dy`.

    """
    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, intx, inty, xi, eta, xp, rho, s, irho, weight
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
    cdef double pA[15]
    cdef double pB[15]
    cdef double dv[15]
    cdef double K[225]
    cdef double TA[75]
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
                if one_I_each_point == 1:
                    I0 = Inxny[ptx, pty, 0]
                    I1 = Inxny[ptx, pty, 1]
                    I2 = Inxny[ptx, pty, 2]
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

                for p in range(225):
                    K[p] = 0.
                K[0] = I0
                K[9] = I1
                K[48] = I0
                K[57] = I1
                K[96] = I0
                K[135] = I1
                K[144] = I2
                K[183] = I1
                K[192] = I2
                c = -1
                for j in range(n):
                    for i in range(m):
                        row = row0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[3, i]*gtab[3, j]
                        pA[4] = ftab[4, i]*gtab[3, j]
                        pA[5] = ftab[3, i]*gtab[4, j]
                        pA[6] = ftab[6, i]*gtab[6, j]
                        pA[7] = ftab[7, i]*gtab[6, j]
                        pA[8] = ftab[6, i]*gtab[7, j]
                        pA[9] = ftab[9, i]*gtab[9, j]
                        pA[10] = ftab[10, i]*gtab[9, j]
                        pA[11] = ftab[9, i]*gtab[10, j]
                        pA[12] = ftab[12, i]*gtab[12, j]
                        pA[13] = ftab[13, i]*gtab[12, j]
                        pA[14] = ftab[12, i]*gtab[13, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(5):
                            for q in range(15):
                                TA[e*15 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*15 + q] += pA[p]*K[p*15 + q]
                        for l in range(n):
                            for k in range(m):
                                col = col0 + 5*(l*m + k)
                                #NOTE upper triangle, the caller symmetrizes
                                if row > col:
                                    continue
                                pB[0] = ftab[0, k]*gtab[0, l]
                                pB[1] = ftab[1, k]*gtab[0, l]
                                pB[2] = ftab[0, k]*gtab[1, l]
                                pB[3] = ftab[3, k]*gtab[3, l]
                                pB[4] = ftab[4, k]*gtab[3, l]
                                pB[5] = ftab[3, k]*gtab[4, l]
                                pB[6] = ftab[6, k]*gtab[6, l]
                                pB[7] = ftab[7, k]*gtab[6, l]
                                pB[8] = ftab[6, k]*gtab[7, l]
                                pB[9] = ftab[9, k]*gtab[9, l]
                                pB[10] = ftab[10, k]*gtab[9, l]
                                pB[11] = ftab[9, k]*gtab[10, l]
                                pB[12] = ftab[12, k]*gtab[12, l]
                                pB[13] = ftab[13, k]*gtab[12, l]
                                pB[14] = ftab[12, k]*gtab[13, l]
                                for e in range(5):
                                    for f2 in range(5):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*15 + q]*pB[q]

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
        Constitutive matrix of the first-order shear deformation theory,
        ``[[A, B, 0], [B, D, 0], [0, 0, A_s]]`` with the transverse shear
        stiffness ``A_s``, shape ``(8, 8)`` for the whole domain, e.g.
        ``shell.ABD``, or ``(nx, ny, 8, 8)`` with one matrix per integration
        point.
    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the numbers
        of terms ``m`` and ``n`` and the boundary flags of the approximation
        functions of each field.
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
    cdef double a, b, R, intx, inty, xi, eta, xp, rho, s, irho, weight
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
    cdef double pA[15]
    cdef double pB[15]
    cdef double dv[15]
    cdef double K[225]
    cdef double TA[75]
    cdef double Bm[120]
    cdef double bs[15]
    cdef double eps0, eps1, eps2, eps3, eps4, eps5, eps6, eps7
    cdef double Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qy, Qx
    cdef double [::1] fint
    cdef int row0 = col0
    cdef double F[64]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 8, 8):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (8, 8):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(8):
            for j in range(8):
                F[i*8 + j] = Finput[i, j]
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
                    for i in range(8):
                        for j in range(8):
                            F[i*8 + j] = Fnxny[ptx, pty, i, j]
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

                for p in range(15):
                    dv[p] = 0.
                for j in range(n):
                    for i in range(m):
                        col = col0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[3, i]*gtab[3, j]
                        pA[4] = ftab[4, i]*gtab[3, j]
                        pA[5] = ftab[3, i]*gtab[4, j]
                        pA[6] = ftab[6, i]*gtab[6, j]
                        pA[7] = ftab[7, i]*gtab[6, j]
                        pA[8] = ftab[6, i]*gtab[7, j]
                        pA[9] = ftab[9, i]*gtab[9, j]
                        pA[10] = ftab[10, i]*gtab[9, j]
                        pA[11] = ftab[9, i]*gtab[10, j]
                        pA[12] = ftab[12, i]*gtab[12, j]
                        pA[13] = ftab[13, i]*gtab[12, j]
                        pA[14] = ftab[12, i]*gtab[13, j]
                        for p in range(15):
                            dv[p] += pA[p]*cs[col + FLD[p]]
                eps0 = (dv[1])*(dv[1])/2. + dv[1] + (dv[4])*(dv[4])/2. + (dv[7])*(dv[7])/2.
                eps1 = (dv[0])*(dv[0])*(irho)*(irho)/2. - dv[0]*dv[5]*irho*s - dv[0]*irho + (dv[2])*(dv[2])*(s)*(s)/2. + dv[2]*dv[3]*irho*s + (dv[3])*(dv[3])*(irho)*(irho)/2. + (dv[5])*(dv[5])*(s)*(s)/2. + dv[5]*s + (dv[8])*(dv[8])*(s)*(s)/2.
                eps2 = -dv[0]*dv[4]*irho + dv[1]*dv[2]*s + dv[1]*dv[3]*irho + dv[2]*s + dv[3]*irho + dv[4]*dv[5]*s + dv[4] + dv[7]*dv[8]*s
                eps3 = dv[10]*dv[1] - dv[10]*dv[7]*dv[9] + dv[10] - dv[12]*dv[13]*dv[7] + dv[13]*dv[4]
                eps4 = -dv[0]*dv[14]*irho*s + dv[0]*dv[9]*(irho)*(irho) + dv[11]*dv[2]*(s)*(s) + dv[11]*dv[3]*irho*s - dv[11]*dv[8]*dv[9]*(s)*(s) - dv[12]*dv[14]*dv[8]*(s)*(s) + dv[12]*dv[2]*irho*s + dv[12]*dv[3]*(irho)*(irho) + dv[14]*dv[5]*(s)*(s) + dv[14]*s - dv[5]*dv[9]*irho*s - dv[9]*irho
                eps5 = -dv[0]*dv[13]*irho + dv[10]*dv[2]*s + dv[10]*dv[3]*irho - dv[10]*dv[8]*dv[9]*s + dv[11]*dv[1]*s - dv[11]*dv[7]*dv[9]*s + dv[11]*s - dv[12]*dv[13]*dv[8]*s - dv[12]*dv[14]*dv[7]*s + dv[12]*dv[1]*irho + dv[12]*irho + dv[13]*dv[5]*s + dv[13] + dv[14]*dv[4]*s - dv[4]*dv[9]*irho
                eps6 = -dv[0]*dv[12]*irho - (dv[12])*(dv[12])*dv[8]*s/2. + dv[12]*dv[5]*s + dv[12] + dv[2]*dv[9]*s + dv[3]*dv[9]*irho - dv[8]*(dv[9])*(dv[9])*s/2. + dv[8]*s
                eps7 = -(dv[12])*(dv[12])*dv[7]/2. + dv[12]*dv[4] + dv[1]*dv[9] - dv[7]*(dv[9])*(dv[9])/2. + dv[7] + dv[9]
                Nxx = F[0]*eps0 + F[1]*eps1 + F[2]*eps2 + F[3]*eps3 + F[4]*eps4 + F[5]*eps5 + F[6]*eps6 + F[7]*eps7
                Nyy = F[8]*eps0 + F[9]*eps1 + F[10]*eps2 + F[11]*eps3 + F[12]*eps4 + F[13]*eps5 + F[14]*eps6 + F[15]*eps7
                Nxy = F[16]*eps0 + F[17]*eps1 + F[18]*eps2 + F[19]*eps3 + F[20]*eps4 + F[21]*eps5 + F[22]*eps6 + F[23]*eps7
                Mxx = F[24]*eps0 + F[25]*eps1 + F[26]*eps2 + F[27]*eps3 + F[28]*eps4 + F[29]*eps5 + F[30]*eps6 + F[31]*eps7
                Myy = F[32]*eps0 + F[33]*eps1 + F[34]*eps2 + F[35]*eps3 + F[36]*eps4 + F[37]*eps5 + F[38]*eps6 + F[39]*eps7
                Mxy = F[40]*eps0 + F[41]*eps1 + F[42]*eps2 + F[43]*eps3 + F[44]*eps4 + F[45]*eps5 + F[46]*eps6 + F[47]*eps7
                Qy = F[48]*eps0 + F[49]*eps1 + F[50]*eps2 + F[51]*eps3 + F[52]*eps4 + F[53]*eps5 + F[54]*eps6 + F[55]*eps7
                Qx = F[56]*eps0 + F[57]*eps1 + F[58]*eps2 + F[59]*eps3 + F[60]*eps4 + F[61]*eps5 + F[62]*eps6 + F[63]*eps7
                for p in range(120):
                    Bm[p] = 0.
                Bm[1] = dv[1] + 1.
                Bm[4] = dv[4]
                Bm[7] = dv[7]
                Bm[15] = dv[0]*(irho)*(irho) - dv[5]*irho*s - irho
                Bm[17] = dv[2]*(s)*(s) + dv[3]*irho*s
                Bm[18] = dv[2]*irho*s + dv[3]*(irho)*(irho)
                Bm[20] = -dv[0]*irho*s + dv[5]*(s)*(s) + s
                Bm[23] = dv[8]*(s)*(s)
                Bm[30] = -dv[4]*irho
                Bm[31] = dv[2]*s + dv[3]*irho
                Bm[32] = dv[1]*s + s
                Bm[33] = dv[1]*irho + irho
                Bm[34] = -dv[0]*irho + dv[5]*s + 1.
                Bm[35] = dv[4]*s
                Bm[37] = dv[8]*s
                Bm[38] = dv[7]*s
                Bm[46] = dv[10]
                Bm[49] = dv[13]
                Bm[52] = -dv[10]*dv[9] - dv[12]*dv[13]
                Bm[54] = -dv[10]*dv[7]
                Bm[55] = dv[1] - dv[7]*dv[9] + 1.
                Bm[57] = -dv[13]*dv[7]
                Bm[58] = -dv[12]*dv[7] + dv[4]
                Bm[60] = -dv[14]*irho*s + dv[9]*(irho)*(irho)
                Bm[62] = dv[11]*(s)*(s) + dv[12]*irho*s
                Bm[63] = dv[11]*irho*s + dv[12]*(irho)*(irho)
                Bm[65] = dv[14]*(s)*(s) - dv[9]*irho*s
                Bm[68] = -dv[11]*dv[9]*(s)*(s) - dv[12]*dv[14]*(s)*(s)
                Bm[69] = dv[0]*(irho)*(irho) - dv[11]*dv[8]*(s)*(s) - dv[5]*irho*s - irho
                Bm[71] = dv[2]*(s)*(s) + dv[3]*irho*s - dv[8]*dv[9]*(s)*(s)
                Bm[72] = -dv[14]*dv[8]*(s)*(s) + dv[2]*irho*s + dv[3]*(irho)*(irho)
                Bm[74] = -dv[0]*irho*s - dv[12]*dv[8]*(s)*(s) + dv[5]*(s)*(s) + s
                Bm[75] = -dv[13]*irho
                Bm[76] = dv[11]*s + dv[12]*irho
                Bm[77] = dv[10]*s
                Bm[78] = dv[10]*irho
                Bm[79] = dv[14]*s - dv[9]*irho
                Bm[80] = dv[13]*s
                Bm[82] = -dv[11]*dv[9]*s - dv[12]*dv[14]*s
                Bm[83] = -dv[10]*dv[9]*s - dv[12]*dv[13]*s
                Bm[84] = -dv[10]*dv[8]*s - dv[11]*dv[7]*s - dv[4]*irho
                Bm[85] = dv[2]*s + dv[3]*irho - dv[8]*dv[9]*s
                Bm[86] = dv[1]*s - dv[7]*dv[9]*s + s
                Bm[87] = -dv[13]*dv[8]*s - dv[14]*dv[7]*s + dv[1]*irho + irho
                Bm[88] = -dv[0]*irho - dv[12]*dv[8]*s + dv[5]*s + 1.
                Bm[89] = -dv[12]*dv[7]*s + dv[4]*s
                Bm[90] = -dv[12]*irho
                Bm[92] = dv[9]*s
                Bm[93] = dv[9]*irho
                Bm[95] = dv[12]*s
                Bm[98] = -(dv[12])*(dv[12])*s/2. - (dv[9])*(dv[9])*s/2. + s
                Bm[99] = dv[2]*s + dv[3]*irho - dv[8]*dv[9]*s
                Bm[102] = -dv[0]*irho - dv[12]*dv[8]*s + dv[5]*s + 1.
                Bm[106] = dv[9]
                Bm[109] = dv[12]
                Bm[112] = -(dv[12])*(dv[12])/2. - (dv[9])*(dv[9])/2. + 1.
                Bm[114] = dv[1] - dv[7]*dv[9] + 1.
                Bm[117] = -dv[12]*dv[7] + dv[4]
                for p in range(15):
                    bs[p] = (Bm[0 + p]*Nxx
                             + Bm[15 + p]*Nyy
                             + Bm[30 + p]*Nxy
                             + Bm[45 + p]*Mxx
                             + Bm[60 + p]*Myy
                             + Bm[75 + p]*Mxy
                             + Bm[90 + p]*Qy
                             + Bm[105 + p]*Qx)
                for j in range(n):
                    for i in range(m):
                        row = row0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[3, i]*gtab[3, j]
                        pA[4] = ftab[4, i]*gtab[3, j]
                        pA[5] = ftab[3, i]*gtab[4, j]
                        pA[6] = ftab[6, i]*gtab[6, j]
                        pA[7] = ftab[7, i]*gtab[6, j]
                        pA[8] = ftab[6, i]*gtab[7, j]
                        pA[9] = ftab[9, i]*gtab[9, j]
                        pA[10] = ftab[10, i]*gtab[9, j]
                        pA[11] = ftab[9, i]*gtab[10, j]
                        pA[12] = ftab[12, i]*gtab[12, j]
                        pA[13] = ftab[13, i]*gtab[12, j]
                        pA[14] = ftab[12, i]*gtab[13, j]
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
        Constitutive matrix of the first-order shear deformation theory,
        ``[[A, B, 0], [B, D, 0], [0, 0, A_s]]`` with the transverse shear
        stiffness ``A_s``, shape ``(8, 8)`` for the whole domain, e.g.
        ``shell.ABD``, or ``(nx, ny, 8, 8)`` with one matrix per integration
        point.
    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the numbers
        of terms ``m`` and ``n`` and the boundary flags of the approximation
        functions of each field.
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
    cdef double a, b, R, intx, inty, xi, eta, xp, rho, s, irho, weight
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
    cdef double pA[15]
    cdef double pB[15]
    cdef double dv[15]
    cdef double K[225]
    cdef double TA[75]
    cdef double Bm[120]
    cdef double bs[15]
    cdef double dq[15]
    cdef double Bq[8]
    cdef double FBq[8]
    cdef double eps0, eps1, eps2, eps3, eps4, eps5, eps6, eps7
    cdef double Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qy, Qx
    cdef double [::1] kTv
    cdef int row0 = col0
    cdef double F[64]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 8, 8):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (8, 8):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(8):
            for j in range(8):
                F[i*8 + j] = Finput[i, j]
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
                    for i in range(8):
                        for j in range(8):
                            F[i*8 + j] = Fnxny[ptx, pty, i, j]
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

                for p in range(15):
                    dq[p] = 0.
                for j in range(n):
                    for i in range(m):
                        col = col0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[3, i]*gtab[3, j]
                        pA[4] = ftab[4, i]*gtab[3, j]
                        pA[5] = ftab[3, i]*gtab[4, j]
                        pA[6] = ftab[6, i]*gtab[6, j]
                        pA[7] = ftab[7, i]*gtab[6, j]
                        pA[8] = ftab[6, i]*gtab[7, j]
                        pA[9] = ftab[9, i]*gtab[9, j]
                        pA[10] = ftab[10, i]*gtab[9, j]
                        pA[11] = ftab[9, i]*gtab[10, j]
                        pA[12] = ftab[12, i]*gtab[12, j]
                        pA[13] = ftab[13, i]*gtab[12, j]
                        pA[14] = ftab[12, i]*gtab[13, j]
                        for p in range(15):
                            dq[p] += pA[p]*vs[col + FLD[p]]
                for p in range(15):
                    dv[p] = 0.
                for j in range(n):
                    for i in range(m):
                        col = col0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[3, i]*gtab[3, j]
                        pA[4] = ftab[4, i]*gtab[3, j]
                        pA[5] = ftab[3, i]*gtab[4, j]
                        pA[6] = ftab[6, i]*gtab[6, j]
                        pA[7] = ftab[7, i]*gtab[6, j]
                        pA[8] = ftab[6, i]*gtab[7, j]
                        pA[9] = ftab[9, i]*gtab[9, j]
                        pA[10] = ftab[10, i]*gtab[9, j]
                        pA[11] = ftab[9, i]*gtab[10, j]
                        pA[12] = ftab[12, i]*gtab[12, j]
                        pA[13] = ftab[13, i]*gtab[12, j]
                        pA[14] = ftab[12, i]*gtab[13, j]
                        for p in range(15):
                            dv[p] += pA[p]*cs[col + FLD[p]]
                eps0 = (dv[1])*(dv[1])/2. + dv[1] + (dv[4])*(dv[4])/2. + (dv[7])*(dv[7])/2.
                eps1 = (dv[0])*(dv[0])*(irho)*(irho)/2. - dv[0]*dv[5]*irho*s - dv[0]*irho + (dv[2])*(dv[2])*(s)*(s)/2. + dv[2]*dv[3]*irho*s + (dv[3])*(dv[3])*(irho)*(irho)/2. + (dv[5])*(dv[5])*(s)*(s)/2. + dv[5]*s + (dv[8])*(dv[8])*(s)*(s)/2.
                eps2 = -dv[0]*dv[4]*irho + dv[1]*dv[2]*s + dv[1]*dv[3]*irho + dv[2]*s + dv[3]*irho + dv[4]*dv[5]*s + dv[4] + dv[7]*dv[8]*s
                eps3 = dv[10]*dv[1] - dv[10]*dv[7]*dv[9] + dv[10] - dv[12]*dv[13]*dv[7] + dv[13]*dv[4]
                eps4 = -dv[0]*dv[14]*irho*s + dv[0]*dv[9]*(irho)*(irho) + dv[11]*dv[2]*(s)*(s) + dv[11]*dv[3]*irho*s - dv[11]*dv[8]*dv[9]*(s)*(s) - dv[12]*dv[14]*dv[8]*(s)*(s) + dv[12]*dv[2]*irho*s + dv[12]*dv[3]*(irho)*(irho) + dv[14]*dv[5]*(s)*(s) + dv[14]*s - dv[5]*dv[9]*irho*s - dv[9]*irho
                eps5 = -dv[0]*dv[13]*irho + dv[10]*dv[2]*s + dv[10]*dv[3]*irho - dv[10]*dv[8]*dv[9]*s + dv[11]*dv[1]*s - dv[11]*dv[7]*dv[9]*s + dv[11]*s - dv[12]*dv[13]*dv[8]*s - dv[12]*dv[14]*dv[7]*s + dv[12]*dv[1]*irho + dv[12]*irho + dv[13]*dv[5]*s + dv[13] + dv[14]*dv[4]*s - dv[4]*dv[9]*irho
                eps6 = -dv[0]*dv[12]*irho - (dv[12])*(dv[12])*dv[8]*s/2. + dv[12]*dv[5]*s + dv[12] + dv[2]*dv[9]*s + dv[3]*dv[9]*irho - dv[8]*(dv[9])*(dv[9])*s/2. + dv[8]*s
                eps7 = -(dv[12])*(dv[12])*dv[7]/2. + dv[12]*dv[4] + dv[1]*dv[9] - dv[7]*(dv[9])*(dv[9])/2. + dv[7] + dv[9]
                Nxx = F[0]*eps0 + F[1]*eps1 + F[2]*eps2 + F[3]*eps3 + F[4]*eps4 + F[5]*eps5 + F[6]*eps6 + F[7]*eps7
                Nyy = F[8]*eps0 + F[9]*eps1 + F[10]*eps2 + F[11]*eps3 + F[12]*eps4 + F[13]*eps5 + F[14]*eps6 + F[15]*eps7
                Nxy = F[16]*eps0 + F[17]*eps1 + F[18]*eps2 + F[19]*eps3 + F[20]*eps4 + F[21]*eps5 + F[22]*eps6 + F[23]*eps7
                Mxx = F[24]*eps0 + F[25]*eps1 + F[26]*eps2 + F[27]*eps3 + F[28]*eps4 + F[29]*eps5 + F[30]*eps6 + F[31]*eps7
                Myy = F[32]*eps0 + F[33]*eps1 + F[34]*eps2 + F[35]*eps3 + F[36]*eps4 + F[37]*eps5 + F[38]*eps6 + F[39]*eps7
                Mxy = F[40]*eps0 + F[41]*eps1 + F[42]*eps2 + F[43]*eps3 + F[44]*eps4 + F[45]*eps5 + F[46]*eps6 + F[47]*eps7
                Qy = F[48]*eps0 + F[49]*eps1 + F[50]*eps2 + F[51]*eps3 + F[52]*eps4 + F[53]*eps5 + F[54]*eps6 + F[55]*eps7
                Qx = F[56]*eps0 + F[57]*eps1 + F[58]*eps2 + F[59]*eps3 + F[60]*eps4 + F[61]*eps5 + F[62]*eps6 + F[63]*eps7
                for p in range(120):
                    Bm[p] = 0.
                Bm[1] = dv[1] + 1.
                Bm[4] = dv[4]
                Bm[7] = dv[7]
                Bm[15] = dv[0]*(irho)*(irho) - dv[5]*irho*s - irho
                Bm[17] = dv[2]*(s)*(s) + dv[3]*irho*s
                Bm[18] = dv[2]*irho*s + dv[3]*(irho)*(irho)
                Bm[20] = -dv[0]*irho*s + dv[5]*(s)*(s) + s
                Bm[23] = dv[8]*(s)*(s)
                Bm[30] = -dv[4]*irho
                Bm[31] = dv[2]*s + dv[3]*irho
                Bm[32] = dv[1]*s + s
                Bm[33] = dv[1]*irho + irho
                Bm[34] = -dv[0]*irho + dv[5]*s + 1.
                Bm[35] = dv[4]*s
                Bm[37] = dv[8]*s
                Bm[38] = dv[7]*s
                Bm[46] = dv[10]
                Bm[49] = dv[13]
                Bm[52] = -dv[10]*dv[9] - dv[12]*dv[13]
                Bm[54] = -dv[10]*dv[7]
                Bm[55] = dv[1] - dv[7]*dv[9] + 1.
                Bm[57] = -dv[13]*dv[7]
                Bm[58] = -dv[12]*dv[7] + dv[4]
                Bm[60] = -dv[14]*irho*s + dv[9]*(irho)*(irho)
                Bm[62] = dv[11]*(s)*(s) + dv[12]*irho*s
                Bm[63] = dv[11]*irho*s + dv[12]*(irho)*(irho)
                Bm[65] = dv[14]*(s)*(s) - dv[9]*irho*s
                Bm[68] = -dv[11]*dv[9]*(s)*(s) - dv[12]*dv[14]*(s)*(s)
                Bm[69] = dv[0]*(irho)*(irho) - dv[11]*dv[8]*(s)*(s) - dv[5]*irho*s - irho
                Bm[71] = dv[2]*(s)*(s) + dv[3]*irho*s - dv[8]*dv[9]*(s)*(s)
                Bm[72] = -dv[14]*dv[8]*(s)*(s) + dv[2]*irho*s + dv[3]*(irho)*(irho)
                Bm[74] = -dv[0]*irho*s - dv[12]*dv[8]*(s)*(s) + dv[5]*(s)*(s) + s
                Bm[75] = -dv[13]*irho
                Bm[76] = dv[11]*s + dv[12]*irho
                Bm[77] = dv[10]*s
                Bm[78] = dv[10]*irho
                Bm[79] = dv[14]*s - dv[9]*irho
                Bm[80] = dv[13]*s
                Bm[82] = -dv[11]*dv[9]*s - dv[12]*dv[14]*s
                Bm[83] = -dv[10]*dv[9]*s - dv[12]*dv[13]*s
                Bm[84] = -dv[10]*dv[8]*s - dv[11]*dv[7]*s - dv[4]*irho
                Bm[85] = dv[2]*s + dv[3]*irho - dv[8]*dv[9]*s
                Bm[86] = dv[1]*s - dv[7]*dv[9]*s + s
                Bm[87] = -dv[13]*dv[8]*s - dv[14]*dv[7]*s + dv[1]*irho + irho
                Bm[88] = -dv[0]*irho - dv[12]*dv[8]*s + dv[5]*s + 1.
                Bm[89] = -dv[12]*dv[7]*s + dv[4]*s
                Bm[90] = -dv[12]*irho
                Bm[92] = dv[9]*s
                Bm[93] = dv[9]*irho
                Bm[95] = dv[12]*s
                Bm[98] = -(dv[12])*(dv[12])*s/2. - (dv[9])*(dv[9])*s/2. + s
                Bm[99] = dv[2]*s + dv[3]*irho - dv[8]*dv[9]*s
                Bm[102] = -dv[0]*irho - dv[12]*dv[8]*s + dv[5]*s + 1.
                Bm[106] = dv[9]
                Bm[109] = dv[12]
                Bm[112] = -(dv[12])*(dv[12])/2. - (dv[9])*(dv[9])/2. + 1.
                Bm[114] = dv[1] - dv[7]*dv[9] + 1.
                Bm[117] = -dv[12]*dv[7] + dv[4]
                for r in range(8):
                    Bq[r] = 0.
                    for p in range(15):
                        Bq[r] += Bm[r*15 + p]*dq[p]
                for r in range(8):
                    FBq[r] = 0.
                    for q in range(8):
                        FBq[r] += F[r*8 + q]*Bq[q]
                for p in range(15):
                    bs[p] = 0.
                    for r in range(8):
                        bs[p] += Bm[r*15 + p]*FBq[r]
                bs[0] += -Mxy*dq[13]*irho - Myy*dq[14]*irho*s + Myy*dq[9]*(irho)*(irho) - Nxy*dq[4]*irho - Nxy_cte*dq[4]*irho + Nyy*dq[0]*(irho)*(irho) - Nyy*dq[5]*irho*s + Nyy_cte*dq[0]*(irho)*(irho) - Nyy_cte*dq[5]*irho*s - Qy*dq[12]*irho
                bs[1] += Mxx*dq[10] + Mxy*dq[11]*s + Mxy*dq[12]*irho + Nxx*dq[1] + Nxx_cte*dq[1] + Nxy*dq[2]*s + Nxy*dq[3]*irho + Nxy_cte*dq[2]*s + Nxy_cte*dq[3]*irho + Qx*dq[9]
                bs[2] += Mxy*dq[10]*s + Myy*dq[11]*(s)*(s) + Myy*dq[12]*irho*s + Nxy*dq[1]*s + Nxy_cte*dq[1]*s + Nyy*dq[2]*(s)*(s) + Nyy*dq[3]*irho*s + Nyy_cte*dq[2]*(s)*(s) + Nyy_cte*dq[3]*irho*s + Qy*dq[9]*s
                bs[3] += Mxy*dq[10]*irho + Myy*dq[11]*irho*s + Myy*dq[12]*(irho)*(irho) + Nxy*dq[1]*irho + Nxy_cte*dq[1]*irho + Nyy*dq[2]*irho*s + Nyy*dq[3]*(irho)*(irho) + Nyy_cte*dq[2]*irho*s + Nyy_cte*dq[3]*(irho)*(irho) + Qy*dq[9]*irho
                bs[4] += Mxx*dq[13] + Mxy*dq[14]*s - Mxy*dq[9]*irho + Nxx*dq[4] + Nxx_cte*dq[4] - Nxy*dq[0]*irho + Nxy*dq[5]*s - Nxy_cte*dq[0]*irho + Nxy_cte*dq[5]*s + Qx*dq[12]
                bs[5] += Mxy*dq[13]*s + Myy*dq[14]*(s)*(s) - Myy*dq[9]*irho*s + Nxy*dq[4]*s + Nxy_cte*dq[4]*s - Nyy*dq[0]*irho*s + Nyy*dq[5]*(s)*(s) - Nyy_cte*dq[0]*irho*s + Nyy_cte*dq[5]*(s)*(s) + Qy*dq[12]*s
                bs[7] += -Mxx*dq[10]*dv[9] - Mxx*dq[12]*dv[13] - Mxx*dq[13]*dv[12] - Mxx*dq[9]*dv[10] - Mxy*dq[11]*dv[9]*s - Mxy*dq[12]*dv[14]*s - Mxy*dq[14]*dv[12]*s - Mxy*dq[9]*dv[11]*s + Nxx*dq[7] + Nxx_cte*dq[7] + Nxy*dq[8]*s + Nxy_cte*dq[8]*s - Qx*dq[12]*dv[12] - Qx*dq[9]*dv[9]
                bs[8] += -Mxy*dq[10]*dv[9]*s - Mxy*dq[12]*dv[13]*s - Mxy*dq[13]*dv[12]*s - Mxy*dq[9]*dv[10]*s - Myy*dq[11]*dv[9]*(s)*(s) - Myy*dq[12]*dv[14]*(s)*(s) - Myy*dq[14]*dv[12]*(s)*(s) - Myy*dq[9]*dv[11]*(s)*(s) + Nxy*dq[7]*s + Nxy_cte*dq[7]*s + Nyy*dq[8]*(s)*(s) + Nyy_cte*dq[8]*(s)*(s) - Qy*dq[12]*dv[12]*s - Qy*dq[9]*dv[9]*s
                bs[9] += -Mxx*dq[10]*dv[7] - Mxx*dq[7]*dv[10] - Mxy*dq[10]*dv[8]*s - Mxy*dq[11]*dv[7]*s - Mxy*dq[4]*irho - Mxy*dq[7]*dv[11]*s - Mxy*dq[8]*dv[10]*s + Myy*dq[0]*(irho)*(irho) - Myy*dq[11]*dv[8]*(s)*(s) - Myy*dq[5]*irho*s - Myy*dq[8]*dv[11]*(s)*(s) + Qx*dq[1] - Qx*dq[7]*dv[9] - Qx*dq[9]*dv[7] + Qy*dq[2]*s + Qy*dq[3]*irho - Qy*dq[8]*dv[9]*s - Qy*dq[9]*dv[8]*s
                bs[10] += Mxx*dq[1] - Mxx*dq[7]*dv[9] - Mxx*dq[9]*dv[7] + Mxy*dq[2]*s + Mxy*dq[3]*irho - Mxy*dq[8]*dv[9]*s - Mxy*dq[9]*dv[8]*s
                bs[11] += Mxy*dq[1]*s - Mxy*dq[7]*dv[9]*s - Mxy*dq[9]*dv[7]*s + Myy*dq[2]*(s)*(s) + Myy*dq[3]*irho*s - Myy*dq[8]*dv[9]*(s)*(s) - Myy*dq[9]*dv[8]*(s)*(s)
                bs[12] += -Mxx*dq[13]*dv[7] - Mxx*dq[7]*dv[13] - Mxy*dq[13]*dv[8]*s - Mxy*dq[14]*dv[7]*s + Mxy*dq[1]*irho - Mxy*dq[7]*dv[14]*s - Mxy*dq[8]*dv[13]*s - Myy*dq[14]*dv[8]*(s)*(s) + Myy*dq[2]*irho*s + Myy*dq[3]*(irho)*(irho) - Myy*dq[8]*dv[14]*(s)*(s) - Qx*dq[12]*dv[7] + Qx*dq[4] - Qx*dq[7]*dv[12] - Qy*dq[0]*irho - Qy*dq[12]*dv[8]*s + Qy*dq[5]*s - Qy*dq[8]*dv[12]*s
                bs[13] += -Mxx*dq[12]*dv[7] + Mxx*dq[4] - Mxx*dq[7]*dv[12] - Mxy*dq[0]*irho - Mxy*dq[12]*dv[8]*s + Mxy*dq[5]*s - Mxy*dq[8]*dv[12]*s
                bs[14] += -Mxy*dq[12]*dv[7]*s + Mxy*dq[4]*s - Mxy*dq[7]*dv[12]*s - Myy*dq[0]*irho*s - Myy*dq[12]*dv[8]*(s)*(s) + Myy*dq[5]*(s)*(s) - Myy*dq[8]*dv[12]*(s)*(s)
                for j in range(n):
                    for i in range(m):
                        row = row0 + 5*(j*m + i)
                        pA[0] = ftab[0, i]*gtab[0, j]
                        pA[1] = ftab[1, i]*gtab[0, j]
                        pA[2] = ftab[0, i]*gtab[1, j]
                        pA[3] = ftab[3, i]*gtab[3, j]
                        pA[4] = ftab[4, i]*gtab[3, j]
                        pA[5] = ftab[3, i]*gtab[4, j]
                        pA[6] = ftab[6, i]*gtab[6, j]
                        pA[7] = ftab[7, i]*gtab[6, j]
                        pA[8] = ftab[6, i]*gtab[7, j]
                        pA[9] = ftab[9, i]*gtab[9, j]
                        pA[10] = ftab[10, i]*gtab[9, j]
                        pA[11] = ftab[9, i]*gtab[10, j]
                        pA[12] = ftab[12, i]*gtab[12, j]
                        pA[13] = ftab[13, i]*gtab[12, j]
                        pA[14] = ftab[12, i]*gtab[13, j]
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
        ``(u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, phix, phix_x, phix_y, phiy, phiy_x, phiy_y)``, 1-D array of size 15.

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
    cdef double pA[15]
    cdef double dv[15]
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
    for p in range(15):
        dv[p] = 0.
    for j in range(n):
        for i in range(m):
            col = col0 + 5*(j*m + i)
            pA[0] = ftab[0, i]*gtab[0, j]
            pA[1] = ftab[1, i]*gtab[0, j]
            pA[2] = ftab[0, i]*gtab[1, j]
            pA[3] = ftab[3, i]*gtab[3, j]
            pA[4] = ftab[4, i]*gtab[3, j]
            pA[5] = ftab[3, i]*gtab[4, j]
            pA[6] = ftab[6, i]*gtab[6, j]
            pA[7] = ftab[7, i]*gtab[6, j]
            pA[8] = ftab[6, i]*gtab[7, j]
            pA[9] = ftab[9, i]*gtab[9, j]
            pA[10] = ftab[10, i]*gtab[9, j]
            pA[11] = ftab[9, i]*gtab[10, j]
            pA[12] = ftab[12, i]*gtab[12, j]
            pA[13] = ftab[13, i]*gtab[12, j]
            pA[14] = ftab[12, i]*gtab[13, j]
            for p in range(15):
                dv[p] += pA[p]*c[col + FLD[p]]
    return np.array([dv[p] for p in range(15)])


def fuvw(double [::1] c, object s, double [::1] xs, double [::1] ys,
         int num_cores=4):
    r"""Displacements and rotations of the panel at the points ``xs``, ``ys``

    Parameters
    ----------
    c : np.ndarray
        Ritz constants of the panel ``s``, 1-D array of size ``5*s.m*s.n``,
        see :meth:`.Shell.get_size`.
    s : :class:`.Shell`
        The annular sector, whose dimensions, outer radius, numbers of terms
        and boundary flags are read.
    xs, ys : np.ndarray
        Coordinates `x` and `y` of the points, lengths, 1-D arrays of the
        same size.
    num_cores : int, optional
        Not used, the points are evaluated serially; kept for the interface
        of :mod:`panels.models.fsdt_tsdt_field`.

    Returns
    -------
    us, vs, ws, phixs, phiys : np.ndarray
        The displacements ``u``, ``v``, ``w`` along the local axes, lengths,
        and the rotations of the normal ``phix`` and ``phiy``, the
        derivatives of ``u`` and ``v`` along ``z``, dimensionless, 1-D arrays
        of the size of ``xs``.

    """
    out = np.zeros((5, xs.shape[0]), dtype=DOUBLE)
    for k in range(xs.shape[0]):
        d = _point_derivatives(c, s, xs[k], ys[k])
        out[:, k] = (d[0], d[3], d[6], d[9], d[12])
    return tuple(out)


def fstrain(double [::1] c, object s, double [::1] xs, double [::1] ys,
            int num_cores=4, int NLgeom=0):
    r"""Generalized strains of the panel at the points ``xs``, ``ys``

    Parameters
    ----------
    c : np.ndarray
        Ritz constants of the panel ``s``, 1-D array of size ``5*s.m*s.n``,
        see :meth:`.Shell.get_size`.
    s : :class:`.Shell`
        The annular sector, whose dimensions, outer radius, numbers of terms
        and boundary flags are read.
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
    exxs, eyys, gxys, kxxs, kyys, kxys, gyzs, gxzs : np.ndarray
        The membrane strains, dimensionless, the changes of curvature, per
        unit length, and the transverse shear strains, dimensionless, along
        the local axes, 1-D arrays of the size of ``xs``, in the order of
        :func:`strain_names`.

    Notes
    -----
    The strains given in the notes of the module, with the non-linear terms
    multiplied by ``NLgeom``.

    """
    cdef double R, s_, irho
    R = s.r
    out = np.zeros((8, xs.shape[0]), dtype=DOUBLE)
    for k in range(xs.shape[0]):
        dv = _point_derivatives(c, s, xs[k], ys[k])
        irho = 1./(R - xs[k])
        s_ = R*irho
        out[0, k] = dv[1] + NLgeom*((dv[1])*(dv[1])/2. + (dv[4])*(dv[4])/2. + (dv[7])*(dv[7])/2.)
        out[1, k] = -dv[0]*irho + dv[5]*s_ + NLgeom*((dv[0])*(dv[0])*(irho)*(irho)/2. - dv[0]*dv[5]*irho*s_ + (dv[2])*(dv[2])*(s_)*(s_)/2. + dv[2]*dv[3]*irho*s_ + (dv[3])*(dv[3])*(irho)*(irho)/2. + (dv[5])*(dv[5])*(s_)*(s_)/2. + (dv[8])*(dv[8])*(s_)*(s_)/2.)
        out[2, k] = dv[2]*s_ + dv[3]*irho + dv[4] + NLgeom*(-dv[0]*dv[4]*irho + dv[1]*dv[2]*s_ + dv[1]*dv[3]*irho + dv[4]*dv[5]*s_ + dv[7]*dv[8]*s_)
        out[3, k] = dv[10] + NLgeom*(dv[10]*dv[1] - dv[10]*dv[7]*dv[9] - dv[12]*dv[13]*dv[7] + dv[13]*dv[4])
        out[4, k] = dv[14]*s_ - dv[9]*irho + NLgeom*(-dv[0]*dv[14]*irho*s_ + dv[0]*dv[9]*(irho)*(irho) + dv[11]*dv[2]*(s_)*(s_) + dv[11]*dv[3]*irho*s_ - dv[11]*dv[8]*dv[9]*(s_)*(s_) - dv[12]*dv[14]*dv[8]*(s_)*(s_) + dv[12]*dv[2]*irho*s_ + dv[12]*dv[3]*(irho)*(irho) + dv[14]*dv[5]*(s_)*(s_) - dv[5]*dv[9]*irho*s_)
        out[5, k] = dv[11]*s_ + dv[12]*irho + dv[13] + NLgeom*(-dv[0]*dv[13]*irho + dv[10]*dv[2]*s_ + dv[10]*dv[3]*irho - dv[10]*dv[8]*dv[9]*s_ + dv[11]*dv[1]*s_ - dv[11]*dv[7]*dv[9]*s_ - dv[12]*dv[13]*dv[8]*s_ - dv[12]*dv[14]*dv[7]*s_ + dv[12]*dv[1]*irho + dv[13]*dv[5]*s_ + dv[14]*dv[4]*s_ - dv[4]*dv[9]*irho)
        out[6, k] = dv[12] + dv[8]*s_ + NLgeom*(-dv[0]*dv[12]*irho - (dv[12])*(dv[12])*dv[8]*s_/2. + dv[12]*dv[5]*s_ + dv[2]*dv[9]*s_ + dv[3]*dv[9]*irho - dv[8]*(dv[9])*(dv[9])*s_/2.)
        out[7, k] = dv[7] + dv[9] + NLgeom*(-(dv[12])*(dv[12])*dv[7]/2. + dv[12]*dv[4] + dv[1]*dv[9] - dv[7]*(dv[9])*(dv[9])/2.)
    return tuple(out)


def strain_names(object s):
    r"""Names of the generalized strains returned by :func:`fstrain`

    Parameters
    ----------
    s : :class:`.Shell`
        The annular sector, not read: the names are those of the model.

    Returns
    -------
    names : tuple of str
        ``exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz``, in the order of the rows and columns of
        ``Shell.ABD``.

    """
    return ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz')


def stress_names(object s):
    r"""Names of the generalized stresses, conjugate to :func:`strain_names`

    Parameters
    ----------
    s : :class:`.Shell`
        The annular sector, not read: the names are those of the model.

    Returns
    -------
    names : tuple of str
        ``Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qy, Qx``, the products of ``Shell.ABD`` by the strains of
        :func:`strain_names`, used as keys of ``Shell.fields`` by
        :meth:`.Shell.stress`.

    """
    return ('Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qy', 'Qx')

def fkAx_num(*args, **kwargs):
    r"""Not available for the annular sector

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('fkAx_num is not implemented for the model '
                              '"annular_fsdt_tl"')

def fkAy_num(*args, **kwargs):
    r"""Not available for the annular sector

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('fkAy_num is not implemented for the model '
                              '"annular_fsdt_tl"')
