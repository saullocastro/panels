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
theory (TSDT) with von Kármán strains in polar coordinates

Numerically integrated kernels of the model ``'annular_tsdt_donnell'``:
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
The displacement field is that of Reddy's third-order shear deformation
theory [1]_ (TSDT), with the cubic term about the mid-surface of the
laminate, `c_1 = 4/(3 h^2)`, as in :mod:`panels.models.plate_tsdt_donnell`,
and the derivative `s w_{,y}` of `w` along the arc, `s = r/\rho`:

.. math::

    u(z) = u + z \phi_x - c_1 \left[(z - d)^3 + d^3\right]
    (\phi_x + w_{,x})

    v(z) = v + z \phi_y - c_1 \left[(z - d)^3 + d^3\right]
    (\phi_y + s w_{,y}) \qquad w(z) = w

With the linear membrane operator in polar coordinates
`\mathcal{L}(f, g) = \{f_{,x},\ s g_{,y} - f/\rho,\ s f_{,y} + g_{,x} +
g/\rho\}^\top`, the in-plane strains at `z` are
`\mathcal{L}(u(z), v(z))` plus the von Kármán terms of Amabili [2]_, Eqs.
(1.175)-(1.177), at the reference plane, and the transverse shear strains
`\{\gamma_{yz}, \gamma_{xz}\} = \{v_{,z} + s w_{,y},\ u_{,z} + w_{,x}\}`:

.. math::

    \boldsymbol{\varepsilon}^{(0)} = \mathcal{L}(u, v) + \left\{
    \tfrac{1}{2} w_{,x}^2,\ \tfrac{1}{2} s^2 w_{,y}^2,\ s w_{,x} w_{,y}
    \right\}^\top \qquad
    \boldsymbol{\varepsilon}^{(1)} = \mathcal{L}(\phi_x, \phi_y)

    \boldsymbol{\varepsilon}^{(3)} = -c_1 \mathcal{L}(\phi_x + w_{,x},
    \phi_y + s w_{,y}) \qquad
    \boldsymbol{\gamma}^{(0)} = \{\phi_y + s w_{,y},\ \phi_x + w_{,x}\}^\top
    \qquad \boldsymbol{\gamma}^{(2)} = -3 c_1 \boldsymbol{\gamma}^{(0)}

with `\boldsymbol{\varepsilon}(z) = \boldsymbol{\varepsilon}^{(0)} + z
\boldsymbol{\varepsilon}^{(1)} + [(z - d)^3 + d^3]
\boldsymbol{\varepsilon}^{(3)}` and `\boldsymbol{\gamma}(z) =
\boldsymbol{\gamma}^{(0)} + (z - d)^2 \boldsymbol{\gamma}^{(2)}`, in the
order of the rows of ``shell.ABD``, the constitutive matrix of the TSDT
with the offset `d`, see :meth:`.Shell._get_lam_ABD`. Since `\partial
s/\partial x = s/\rho`, `\mathcal{L}` applied to `s w_{,y}` gives `s
w_{,xy} + s w_{,y}/\rho` and `s^2 w_{,yy}`. With `\phi_x = -w_{,x}`,
`\phi_y = -s w_{,y}` the strains are those of
``annular_clpt_donnell``; for `r \to \infty` those of
``plate_tsdt_donnell``. The second derivatives of the strains with respect
to the generalized derivatives are constant and only on `(w_{,x},
w_{,y})`.

The element of area is `dA = (\rho/r)\, dx\, dy`. The integrands are
rational functions of `x`, integrated with the Gauss-Legendre rule over the
domain of :meth:`.Shell.integration_limits`.

References
----------
.. [1] Reddy, J. N., A simple higher-order theory for laminated composite
   plates, J. Appl. Mech. 51 (1984) 745-752.
   https://doi.org/10.1115/1.3167719
.. [2] Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
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
cdef int[5] P0 = [0, 3, 6, 12, 15]
cdef int[5] P1 = [3, 6, 12, 15, 18]
#: degree of freedom of each generalized derivative
cdef int[18] FLD = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4]


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

    with `dA = (\rho/r)\, dx\, dy`, see the notes of the module. Only the
    membrane stresses enter `\boldsymbol{H}`. With ``NLgeom=0``,
    `\boldsymbol{d} = \boldsymbol{0}` and the matrix is `\int
    \boldsymbol{\Phi}^\top \boldsymbol{B}_0^\top \boldsymbol{F}
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
    cdef double pA[18]
    cdef double pB[18]
    cdef double dv[18]
    cdef double K[324]
    cdef double TA[90]
    cdef double Bm[234]
    cdef double FB[234]
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

                for p in range(234):
                    Bm[p] = 0.
                for p in range(18):
                    dv[p] = 0.
                if NLgeom == 1:
                    for p in range(18):
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
                            pA[9] = ftab[8, i]*gtab[6, j]
                            pA[10] = ftab[6, i]*gtab[8, j]
                            pA[11] = ftab[7, i]*gtab[7, j]
                            pA[12] = ftab[9, i]*gtab[9, j]
                            pA[13] = ftab[10, i]*gtab[9, j]
                            pA[14] = ftab[9, i]*gtab[10, j]
                            pA[15] = ftab[12, i]*gtab[12, j]
                            pA[16] = ftab[13, i]*gtab[12, j]
                            pA[17] = ftab[12, i]*gtab[13, j]
                            for p in range(18):
                                dv[p] += pA[p]*cs[col + FLD[p]]
                Bm[1] = 1.
                Bm[7] = dv[7]
                Bm[18] = -irho
                Bm[23] = s
                Bm[26] = dv[8]*(s)*(s)
                Bm[38] = s
                Bm[39] = irho
                Bm[40] = 1.
                Bm[43] = dv[8]*s
                Bm[44] = dv[7]*s
                Bm[67] = 1.
                Bm[84] = -irho
                Bm[89] = s
                Bm[104] = s
                Bm[105] = irho
                Bm[106] = 1.
                Bm[117] = -c1
                Bm[121] = -c1
                Bm[133] = c1*irho
                Bm[136] = -c1*(s)*(s)
                Bm[138] = c1*irho
                Bm[143] = -c1*s
                Bm[152] = -2.*c1*irho*s
                Bm[155] = -2.*c1*s
                Bm[158] = -c1*s
                Bm[159] = -c1*irho
                Bm[160] = -c1
                Bm[170] = s
                Bm[177] = 1.
                Bm[187] = 1.
                Bm[192] = 1.
                Bm[206] = -3.*c1*s
                Bm[213] = -3.*c1
                Bm[223] = -3.*c1
                Bm[228] = -3.*c1
                # K = [B]^T [F] [B]
                for p in range(13):
                    for q in range(18):
                        FB[p*18 + q] = 0.
                        for r in range(13):
                            FB[p*18 + q] += F[p*13 + r]*Bm[r*18 + q]
                for p in range(18):
                    for q in range(18):
                        K[p*18 + q] = 0.
                        for r in range(13):
                            K[p*18 + q] += Bm[r*18 + p]*FB[r*18 + q]
                if NLgeom == 1:
                    # + [H(sigma, d)] - [H(sigma_L, 0)]
                    eps[0] = dv[1] + (dv[7])*(dv[7])/2.
                    eps[1] = -dv[0]*irho + dv[5]*s + (dv[8])*(dv[8])*(s)*(s)/2.
                    eps[2] = dv[2]*s + dv[3]*irho + dv[4] + dv[7]*dv[8]*s
                    eps[3] = dv[13]
                    eps[4] = -dv[12]*irho + dv[17]*s
                    eps[5] = dv[14]*s + dv[15]*irho + dv[16]
                    eps[6] = -c1*dv[13] - c1*dv[9]
                    eps[7] = -c1*dv[10]*(s)*(s) + c1*dv[12]*irho - c1*dv[17]*s + c1*dv[7]*irho
                    eps[8] = -2.*c1*dv[11]*s - c1*dv[14]*s - c1*dv[15]*irho - c1*dv[16] - 2.*c1*dv[8]*irho*s
                    eps[9] = dv[15] + dv[8]*s
                    eps[10] = dv[12] + dv[7]
                    eps[11] = -3.*c1*dv[15] - 3.*c1*dv[8]*s
                    eps[12] = -3.*c1*dv[12] - 3.*c1*dv[7]
                    epsL[0] = dv[1]
                    epsL[1] = -dv[0]*irho + dv[5]*s
                    epsL[2] = dv[2]*s + dv[3]*irho + dv[4]
                    epsL[3] = dv[13]
                    epsL[4] = -dv[12]*irho + dv[17]*s
                    epsL[5] = dv[14]*s + dv[15]*irho + dv[16]
                    epsL[6] = -c1*dv[13] - c1*dv[9]
                    epsL[7] = -c1*dv[10]*(s)*(s) + c1*dv[12]*irho - c1*dv[17]*s + c1*dv[7]*irho
                    epsL[8] = -2.*c1*dv[11]*s - c1*dv[14]*s - c1*dv[15]*irho - c1*dv[16] - 2.*c1*dv[8]*irho*s
                    epsL[9] = dv[15] + dv[8]*s
                    epsL[10] = dv[12] + dv[7]
                    epsL[11] = -3.*c1*dv[15] - 3.*c1*dv[8]*s
                    epsL[12] = -3.*c1*dv[12] - 3.*c1*dv[7]
                    for r in range(3):
                        sig[r] = 0.
                        for q in range(13):
                            sig[r] += F[r*13 + q]*eps[q]
                    for r in range(3):
                        sigL[r] = 0.
                        for q in range(13):
                            sigL[r] += F[r*13 + q]*epsL[q]
                    K[133] += sig[0]
                    K[134] += s*sig[2]
                    K[151] += s*sig[2]
                    K[152] += (s)*(s)*sig[1]
                    K[133] -= sigL[0]
                    K[134] -= s*sigL[2]
                    K[151] -= s*sigL[2]
                    K[152] -= (s)*(s)*sigL[1]
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
                        pA[9] = ftab[8, i]*gtab[6, j]
                        pA[10] = ftab[6, i]*gtab[8, j]
                        pA[11] = ftab[7, i]*gtab[7, j]
                        pA[12] = ftab[9, i]*gtab[9, j]
                        pA[13] = ftab[10, i]*gtab[9, j]
                        pA[14] = ftab[9, i]*gtab[10, j]
                        pA[15] = ftab[12, i]*gtab[12, j]
                        pA[16] = ftab[13, i]*gtab[12, j]
                        pA[17] = ftab[12, i]*gtab[13, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(5):
                            for q in range(18):
                                TA[e*18 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*18 + q] += pA[p]*K[p*18 + q]
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
                                pB[9] = ftab[8, k]*gtab[6, l]
                                pB[10] = ftab[6, k]*gtab[8, l]
                                pB[11] = ftab[7, k]*gtab[7, l]
                                pB[12] = ftab[9, k]*gtab[9, l]
                                pB[13] = ftab[10, k]*gtab[9, l]
                                pB[14] = ftab[9, k]*gtab[10, l]
                                pB[15] = ftab[12, k]*gtab[12, l]
                                pB[16] = ftab[13, k]*gtab[12, l]
                                pB[17] = ftab[12, k]*gtab[13, l]
                                for e in range(5):
                                    for f2 in range(5):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*18 + q]*pB[q]

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
    cdef double pA[18]
    cdef double pB[18]
    cdef double dv[18]
    cdef double K[324]
    cdef double TA[90]
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

                for p in range(18):
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
                        pA[9] = ftab[8, i]*gtab[6, j]
                        pA[10] = ftab[6, i]*gtab[8, j]
                        pA[11] = ftab[7, i]*gtab[7, j]
                        pA[12] = ftab[9, i]*gtab[9, j]
                        pA[13] = ftab[10, i]*gtab[9, j]
                        pA[14] = ftab[9, i]*gtab[10, j]
                        pA[15] = ftab[12, i]*gtab[12, j]
                        pA[16] = ftab[13, i]*gtab[12, j]
                        pA[17] = ftab[12, i]*gtab[13, j]
                        for p in range(18):
                            dv[p] += pA[p]*cs[col + FLD[p]]
                epsL[0] = dv[1]
                epsL[1] = -dv[0]*irho + dv[5]*s
                epsL[2] = dv[2]*s + dv[3]*irho + dv[4]
                epsL[3] = dv[13]
                epsL[4] = -dv[12]*irho + dv[17]*s
                epsL[5] = dv[14]*s + dv[15]*irho + dv[16]
                epsL[6] = -c1*dv[13] - c1*dv[9]
                epsL[7] = -c1*dv[10]*(s)*(s) + c1*dv[12]*irho - c1*dv[17]*s + c1*dv[7]*irho
                epsL[8] = -2.*c1*dv[11]*s - c1*dv[14]*s - c1*dv[15]*irho - c1*dv[16] - 2.*c1*dv[8]*irho*s
                epsL[9] = dv[15] + dv[8]*s
                epsL[10] = dv[12] + dv[7]
                epsL[11] = -3.*c1*dv[15] - 3.*c1*dv[8]*s
                epsL[12] = -3.*c1*dv[12] - 3.*c1*dv[7]
                for r in range(3):
                    sigL[r] = 0.
                    for q in range(13):
                        sigL[r] += F[r*13 + q]*epsL[q]
                sigL[0] = sigL[0] + Nxx0
                sigL[1] = sigL[1] + Nyy0
                sigL[2] = sigL[2] + Nxy0
                for p in range(324):
                    K[p] = 0.
                K[133] += sigL[0]
                K[134] += s*sigL[2]
                K[151] += s*sigL[2]
                K[152] += (s)*(s)*sigL[1]
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
                        pA[9] = ftab[8, i]*gtab[6, j]
                        pA[10] = ftab[6, i]*gtab[8, j]
                        pA[11] = ftab[7, i]*gtab[7, j]
                        pA[12] = ftab[9, i]*gtab[9, j]
                        pA[13] = ftab[10, i]*gtab[9, j]
                        pA[14] = ftab[9, i]*gtab[10, j]
                        pA[15] = ftab[12, i]*gtab[12, j]
                        pA[16] = ftab[13, i]*gtab[12, j]
                        pA[17] = ftab[12, i]*gtab[13, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(5):
                            for q in range(18):
                                TA[e*18 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*18 + q] += pA[p]*K[p*18 + q]
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
                                pB[9] = ftab[8, k]*gtab[6, l]
                                pB[10] = ftab[6, k]*gtab[8, l]
                                pB[11] = ftab[7, k]*gtab[7, l]
                                pB[12] = ftab[9, k]*gtab[9, l]
                                pB[13] = ftab[10, k]*gtab[9, l]
                                pB[14] = ftab[9, k]*gtab[10, l]
                                pB[15] = ftab[12, k]*gtab[12, l]
                                pB[16] = ftab[13, k]*gtab[12, l]
                                pB[17] = ftab[12, k]*gtab[13, l]
                                for e in range(5):
                                    for f2 in range(5):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*18 + q]*pB[q]

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
    cdef double pA[18]
    cdef double pB[18]
    cdef double dv[18]
    cdef double K[324]
    cdef double TA[90]
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

                for p in range(324):
                    K[p] = 0.
                K[0] = J0
                K[7] = -J0*c1*(d)*(d)*(d) - J3*c1
                K[12] = -J0*c1*(d)*(d)*(d) + J0*d + J1 - J3*c1
                K[57] = J0
                K[62] = -J0*c1*(d)*(d)*(d)*s - J3*c1*s
                K[69] = -J0*c1*(d)*(d)*(d) + J0*d + J1 - J3*c1
                K[114] = J0
                K[126] = -J0*c1*(d)*(d)*(d) - J3*c1
                K[133] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d) + 2.*J3*(c1)*(c1)*(d)*(d)*(d) + J6*(c1)*(c1)
                K[138] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d) - J0*c1*(d)*(d)*(d)*(d) - J1*c1*(d)*(d)*(d) + 2.*J3*(c1)*(c1)*(d)*(d)*(d) - J3*c1*d - J4*c1 + J6*(c1)*(c1)
                K[147] = -J0*c1*(d)*(d)*(d)*s - J3*c1*s
                K[152] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d)*(s)*(s) + 2.*J3*(c1)*(c1)*(d)*(d)*(d)*(s)*(s) + J6*(c1)*(c1)*(s)*(s)
                K[159] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d)*s - J0*c1*(d)*(d)*(d)*(d)*s - J1*c1*(d)*(d)*(d)*s + 2.*J3*(c1)*(c1)*(d)*(d)*(d)*s - J3*c1*d*s - J4*c1*s + J6*(c1)*(c1)*s
                K[216] = -J0*c1*(d)*(d)*(d) + J0*d + J1 - J3*c1
                K[223] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d) - J0*c1*(d)*(d)*(d)*(d) - J1*c1*(d)*(d)*(d) + 2.*J3*(c1)*(c1)*(d)*(d)*(d) - J3*c1*d - J4*c1 + J6*(c1)*(c1)
                K[228] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d) - 2.*J0*c1*(d)*(d)*(d)*(d) + J0*(d)*(d) - 2.*J1*c1*(d)*(d)*(d) + 2.*J1*d + J2 + 2.*J3*(c1)*(c1)*(d)*(d)*(d) - 2.*J3*c1*d - 2.*J4*c1 + J6*(c1)*(c1)
                K[273] = -J0*c1*(d)*(d)*(d) + J0*d + J1 - J3*c1
                K[278] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d)*s - J0*c1*(d)*(d)*(d)*(d)*s - J1*c1*(d)*(d)*(d)*s + 2.*J3*(c1)*(c1)*(d)*(d)*(d)*s - J3*c1*d*s - J4*c1*s + J6*(c1)*(c1)*s
                K[285] = J0*(c1)*(c1)*(d)*(d)*(d)*(d)*(d)*(d) - 2.*J0*c1*(d)*(d)*(d)*(d) + J0*(d)*(d) - 2.*J1*c1*(d)*(d)*(d) + 2.*J1*d + J2 + 2.*J3*(c1)*(c1)*(d)*(d)*(d) - 2.*J3*c1*d - 2.*J4*c1 + J6*(c1)*(c1)
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
                        pA[9] = ftab[8, i]*gtab[6, j]
                        pA[10] = ftab[6, i]*gtab[8, j]
                        pA[11] = ftab[7, i]*gtab[7, j]
                        pA[12] = ftab[9, i]*gtab[9, j]
                        pA[13] = ftab[10, i]*gtab[9, j]
                        pA[14] = ftab[9, i]*gtab[10, j]
                        pA[15] = ftab[12, i]*gtab[12, j]
                        pA[16] = ftab[13, i]*gtab[12, j]
                        pA[17] = ftab[12, i]*gtab[13, j]
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range(5):
                            for q in range(18):
                                TA[e*18 + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*18 + q] += pA[p]*K[p*18 + q]
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
                                pB[9] = ftab[8, k]*gtab[6, l]
                                pB[10] = ftab[6, k]*gtab[8, l]
                                pB[11] = ftab[7, k]*gtab[7, l]
                                pB[12] = ftab[9, k]*gtab[9, l]
                                pB[13] = ftab[10, k]*gtab[9, l]
                                pB[14] = ftab[9, k]*gtab[10, l]
                                pB[15] = ftab[12, k]*gtab[12, l]
                                pB[16] = ftab[13, k]*gtab[12, l]
                                pB[17] = ftab[12, k]*gtab[13, l]
                                for e in range(5):
                                    for f2 in range(5):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*18 + q]*pB[q]

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
    cdef double pA[18]
    cdef double pB[18]
    cdef double dv[18]
    cdef double K[324]
    cdef double TA[90]
    cdef double Bm[234]
    cdef double bs[18]
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

                for p in range(18):
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
                        pA[9] = ftab[8, i]*gtab[6, j]
                        pA[10] = ftab[6, i]*gtab[8, j]
                        pA[11] = ftab[7, i]*gtab[7, j]
                        pA[12] = ftab[9, i]*gtab[9, j]
                        pA[13] = ftab[10, i]*gtab[9, j]
                        pA[14] = ftab[9, i]*gtab[10, j]
                        pA[15] = ftab[12, i]*gtab[12, j]
                        pA[16] = ftab[13, i]*gtab[12, j]
                        pA[17] = ftab[12, i]*gtab[13, j]
                        for p in range(18):
                            dv[p] += pA[p]*cs[col + FLD[p]]
                eps[0] = dv[1] + (dv[7])*(dv[7])/2.
                eps[1] = -dv[0]*irho + dv[5]*s + (dv[8])*(dv[8])*(s)*(s)/2.
                eps[2] = dv[2]*s + dv[3]*irho + dv[4] + dv[7]*dv[8]*s
                eps[3] = dv[13]
                eps[4] = -dv[12]*irho + dv[17]*s
                eps[5] = dv[14]*s + dv[15]*irho + dv[16]
                eps[6] = -c1*dv[13] - c1*dv[9]
                eps[7] = -c1*dv[10]*(s)*(s) + c1*dv[12]*irho - c1*dv[17]*s + c1*dv[7]*irho
                eps[8] = -2.*c1*dv[11]*s - c1*dv[14]*s - c1*dv[15]*irho - c1*dv[16] - 2.*c1*dv[8]*irho*s
                eps[9] = dv[15] + dv[8]*s
                eps[10] = dv[12] + dv[7]
                eps[11] = -3.*c1*dv[15] - 3.*c1*dv[8]*s
                eps[12] = -3.*c1*dv[12] - 3.*c1*dv[7]
                for r in range(13):
                    sig[r] = 0.
                    for q in range(13):
                        sig[r] += F[r*13 + q]*eps[q]
                for p in range(234):
                    Bm[p] = 0.
                Bm[1] = 1.
                Bm[7] = dv[7]
                Bm[18] = -irho
                Bm[23] = s
                Bm[26] = dv[8]*(s)*(s)
                Bm[38] = s
                Bm[39] = irho
                Bm[40] = 1.
                Bm[43] = dv[8]*s
                Bm[44] = dv[7]*s
                Bm[67] = 1.
                Bm[84] = -irho
                Bm[89] = s
                Bm[104] = s
                Bm[105] = irho
                Bm[106] = 1.
                Bm[117] = -c1
                Bm[121] = -c1
                Bm[133] = c1*irho
                Bm[136] = -c1*(s)*(s)
                Bm[138] = c1*irho
                Bm[143] = -c1*s
                Bm[152] = -2.*c1*irho*s
                Bm[155] = -2.*c1*s
                Bm[158] = -c1*s
                Bm[159] = -c1*irho
                Bm[160] = -c1
                Bm[170] = s
                Bm[177] = 1.
                Bm[187] = 1.
                Bm[192] = 1.
                Bm[206] = -3.*c1*s
                Bm[213] = -3.*c1
                Bm[223] = -3.*c1
                Bm[228] = -3.*c1
                for p in range(18):
                    bs[p] = 0.
                    for r in range(13):
                        bs[p] += Bm[r*18 + p]*sig[r]
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
                        pA[9] = ftab[8, i]*gtab[6, j]
                        pA[10] = ftab[6, i]*gtab[8, j]
                        pA[11] = ftab[7, i]*gtab[7, j]
                        pA[12] = ftab[9, i]*gtab[9, j]
                        pA[13] = ftab[10, i]*gtab[9, j]
                        pA[14] = ftab[9, i]*gtab[10, j]
                        pA[15] = ftab[12, i]*gtab[12, j]
                        pA[16] = ftab[13, i]*gtab[12, j]
                        pA[17] = ftab[12, i]*gtab[13, j]
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
    cdef double pA[18]
    cdef double pB[18]
    cdef double dv[18]
    cdef double K[324]
    cdef double TA[90]
    cdef double Bm[234]
    cdef double bs[18]
    cdef double dq[18]
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

                for p in range(18):
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
                        pA[9] = ftab[8, i]*gtab[6, j]
                        pA[10] = ftab[6, i]*gtab[8, j]
                        pA[11] = ftab[7, i]*gtab[7, j]
                        pA[12] = ftab[9, i]*gtab[9, j]
                        pA[13] = ftab[10, i]*gtab[9, j]
                        pA[14] = ftab[9, i]*gtab[10, j]
                        pA[15] = ftab[12, i]*gtab[12, j]
                        pA[16] = ftab[13, i]*gtab[12, j]
                        pA[17] = ftab[12, i]*gtab[13, j]
                        for p in range(18):
                            dq[p] += pA[p]*vs[col + FLD[p]]
                for p in range(18):
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
                        pA[9] = ftab[8, i]*gtab[6, j]
                        pA[10] = ftab[6, i]*gtab[8, j]
                        pA[11] = ftab[7, i]*gtab[7, j]
                        pA[12] = ftab[9, i]*gtab[9, j]
                        pA[13] = ftab[10, i]*gtab[9, j]
                        pA[14] = ftab[9, i]*gtab[10, j]
                        pA[15] = ftab[12, i]*gtab[12, j]
                        pA[16] = ftab[13, i]*gtab[12, j]
                        pA[17] = ftab[12, i]*gtab[13, j]
                        for p in range(18):
                            dv[p] += pA[p]*cs[col + FLD[p]]
                eps[0] = dv[1] + (dv[7])*(dv[7])/2.
                eps[1] = -dv[0]*irho + dv[5]*s + (dv[8])*(dv[8])*(s)*(s)/2.
                eps[2] = dv[2]*s + dv[3]*irho + dv[4] + dv[7]*dv[8]*s
                eps[3] = dv[13]
                eps[4] = -dv[12]*irho + dv[17]*s
                eps[5] = dv[14]*s + dv[15]*irho + dv[16]
                eps[6] = -c1*dv[13] - c1*dv[9]
                eps[7] = -c1*dv[10]*(s)*(s) + c1*dv[12]*irho - c1*dv[17]*s + c1*dv[7]*irho
                eps[8] = -2.*c1*dv[11]*s - c1*dv[14]*s - c1*dv[15]*irho - c1*dv[16] - 2.*c1*dv[8]*irho*s
                eps[9] = dv[15] + dv[8]*s
                eps[10] = dv[12] + dv[7]
                eps[11] = -3.*c1*dv[15] - 3.*c1*dv[8]*s
                eps[12] = -3.*c1*dv[12] - 3.*c1*dv[7]
                for r in range(3):
                    sig[r] = 0.
                    for q in range(13):
                        sig[r] += F[r*13 + q]*eps[q]
                for p in range(234):
                    Bm[p] = 0.
                Bm[1] = 1.
                Bm[7] = dv[7]
                Bm[18] = -irho
                Bm[23] = s
                Bm[26] = dv[8]*(s)*(s)
                Bm[38] = s
                Bm[39] = irho
                Bm[40] = 1.
                Bm[43] = dv[8]*s
                Bm[44] = dv[7]*s
                Bm[67] = 1.
                Bm[84] = -irho
                Bm[89] = s
                Bm[104] = s
                Bm[105] = irho
                Bm[106] = 1.
                Bm[117] = -c1
                Bm[121] = -c1
                Bm[133] = c1*irho
                Bm[136] = -c1*(s)*(s)
                Bm[138] = c1*irho
                Bm[143] = -c1*s
                Bm[152] = -2.*c1*irho*s
                Bm[155] = -2.*c1*s
                Bm[158] = -c1*s
                Bm[159] = -c1*irho
                Bm[160] = -c1
                Bm[170] = s
                Bm[177] = 1.
                Bm[187] = 1.
                Bm[192] = 1.
                Bm[206] = -3.*c1*s
                Bm[213] = -3.*c1
                Bm[223] = -3.*c1
                Bm[228] = -3.*c1
                for r in range(13):
                    Bq[r] = 0.
                    for p in range(18):
                        Bq[r] += Bm[r*18 + p]*dq[p]
                for r in range(13):
                    FBq[r] = 0.
                    for q in range(13):
                        FBq[r] += F[r*13 + q]*Bq[q]
                for p in range(18):
                    bs[p] = 0.
                    for r in range(13):
                        bs[p] += Bm[r*18 + p]*FBq[r]
                bs[7] += Nxx_cte*dq[7] + Nxy_cte*dq[8]*s + dq[7]*sig[0] + dq[8]*s*sig[2]
                bs[8] += Nxy_cte*dq[7]*s + Nyy_cte*dq[8]*(s)*(s) + dq[7]*s*sig[2] + dq[8]*(s)*(s)*sig[1]
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
                        pA[9] = ftab[8, i]*gtab[6, j]
                        pA[10] = ftab[6, i]*gtab[8, j]
                        pA[11] = ftab[7, i]*gtab[7, j]
                        pA[12] = ftab[9, i]*gtab[9, j]
                        pA[13] = ftab[10, i]*gtab[9, j]
                        pA[14] = ftab[9, i]*gtab[10, j]
                        pA[15] = ftab[12, i]*gtab[12, j]
                        pA[16] = ftab[13, i]*gtab[12, j]
                        pA[17] = ftab[12, i]*gtab[13, j]
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
        ``(u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy, phix, phix_x, phix_y, phiy, phiy_x, phiy_y)``, 1-D array of size 18.

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
    cdef double pA[18]
    cdef double dv[18]
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
    for p in range(18):
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
            pA[9] = ftab[8, i]*gtab[6, j]
            pA[10] = ftab[6, i]*gtab[8, j]
            pA[11] = ftab[7, i]*gtab[7, j]
            pA[12] = ftab[9, i]*gtab[9, j]
            pA[13] = ftab[10, i]*gtab[9, j]
            pA[14] = ftab[9, i]*gtab[10, j]
            pA[15] = ftab[12, i]*gtab[12, j]
            pA[16] = ftab[13, i]*gtab[12, j]
            pA[17] = ftab[12, i]*gtab[13, j]
            for p in range(18):
                dv[p] += pA[p]*c[col + FLD[p]]
    return np.array([dv[p] for p in range(18)])


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
        out[0, k] = dv[1] + NLgeom*((dv[7])*(dv[7])/2.)
        out[1, k] = -dv[0]*irho + dv[5]*s_ + NLgeom*((dv[8])*(dv[8])*(s_)*(s_)/2.)
        out[2, k] = dv[2]*s_ + dv[3]*irho + dv[4] + NLgeom*(dv[7]*dv[8]*s_)
        out[3, k] = dv[13]
        out[4, k] = -dv[12]*irho + dv[17]*s_
        out[5, k] = dv[14]*s_ + dv[15]*irho + dv[16]
        out[6, k] = -c1*dv[13] - c1*dv[9]
        out[7, k] = -c1*dv[10]*(s_)*(s_) + c1*dv[12]*irho - c1*dv[17]*s_ + c1*dv[7]*irho
        out[8, k] = -2.*c1*dv[11]*s_ - c1*dv[14]*s_ - c1*dv[15]*irho - c1*dv[16] - 2.*c1*dv[8]*irho*s_
        out[9, k] = dv[15] + dv[8]*s_
        out[10, k] = dv[12] + dv[7]
        out[11, k] = -3.*c1*dv[15] - 3.*c1*dv[8]*s_
        out[12, k] = -3.*c1*dv[12] - 3.*c1*dv[7]
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
                              '"annular_tsdt_donnell"')

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
                              '"annular_tsdt_donnell"')
