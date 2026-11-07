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

Analytical kernels of the model ``'cylshell_tsdt_gl'``, integrated over the whole
domain. The linear constitutive stiffness ``fk0``, the mass ``fkM`` and the
aerodynamic ``fkAx``, ``fkAy`` and ``fcA`` matrices are those of
:mod:`panels.models.cylshell_tsdt_sanders`. The geometric stiffness matrix :func:`fkG0` of
the constant membrane stress state ``Nxx, Nyy, Nxy`` has one block per
local gradient of ``u``, ``v`` and ``w``.

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

The integrals over the whole domain are evaluated in closed form with the
integrals of Bardell's functions and their derivatives along `\xi = 2x/a -
1` and `\eta = 2y/b - 1`, which requires the constitutive matrix and the
moments of the density to be constant over the domain.

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

from panels import INT, DOUBLE
#NOTE the linear kernels, independent of the non-linear strains, are those
#     of cylshell_tsdt_sanders
from .cylshell_tsdt_sanders import fk0, fkM, fkAx, fkAy, fcA


cdef int DOF = 5


cdef extern from 'bardell.hpp':

    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil


def fkG0(double Nxx, double Nyy, double Nxy, object shell,
         int size, int row0, int col0):
    r"""Geometric stiffness matrix of a constant membrane stress state

    Integrates, analytically over the whole domain, the geometric
    stiffness matrix of the membrane stress resultants ``Nxx``, ``Nyy``
    and ``Nxy``, constant over the domain.

    Parameters
    ----------
    Nxx, Nyy, Nxy : float
        Membrane stress resultants, force per unit length, tension positive.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    kG0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With `\boldsymbol{N} = \begin{bmatrix} N_{xx} & N_{xy} \\ N_{xy} &
    N_{yy} \end{bmatrix}` and `\boldsymbol{G} \boldsymbol{c} = \{u_{,x}, u_{,y}, v_{,x}, v_{,y} + w/r,
    w_{,x}, w_{,y} - v/r\}^\top` the gradients of the displacements
    in the local basis, with one
    block `\boldsymbol{N}` per displacement:

    .. math::

        \boldsymbol{K}_{G} = \int_0^b \int_0^a \boldsymbol{G}^\top
        \boldsymbol{N} \boldsymbol{G} \, dx \, dy

    which is the Hessian of the work of the constant stress resultants on
    the non-linear part of the membrane strains of the kinematics of the module.

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\xi` and `\eta`, with `x = a(\xi +
    1)/2`, `y = b(\eta + 1)/2` and `dx\,dy = (ab/4)\,d\xi\,d\eta`.

    """
    cdef double a, b, radius
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

    cdef int i, j, k, l, c, row, col

    cdef long [:] kG0r, kG0c
    cdef double [:] kG0v

    cdef double fAufBu, fAufBuxi, fAuxifBu, fAuxifBuxi
    cdef double fAvfBv, fAvfBvxi, fAvfBw, fAvfBwxi
    cdef double fAvxifBv, fAvxifBvxi, fAvxifBw, fAwfBv
    cdef double fAwfBvxi, fAwfBw, fAwfBwxi, fAwxifBv
    cdef double fAwxifBw, fAwxifBwxi
    cdef double gAuetagBu, gAuetagBueta, gAugBu, gAugBueta
    cdef double gAvetagBv, gAvetagBveta, gAvetagBw, gAvgBv
    cdef double gAvgBveta, gAvgBw, gAvgBweta, gAwetagBv
    cdef double gAwetagBw, gAwetagBweta, gAwgBv, gAwgBveta
    cdef double gAwgBw, gAwgBweta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    radius = shell.r
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

    fdim = 5*m*m*n*n

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kG0
        c = -1
        for i in range(m):
            for k in range(m):

                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAufBuxi = integral_ffp(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAuxifBu = integral_ffp(k, i, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAuxifBuxi = integral_fpfp(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBvxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvfBw = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvfBwxi = integral_ffp(i, k, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAvxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvxifBvxi = integral_fpfp(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAvxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBv = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBvxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwfBwxi = integral_ffp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBv = integral_ffp(k, i, x1v, x1vr, x2v, x2vr, x1w, x1wr, x2w, x2wr)
                fAwxifBw = integral_ffp(k, i, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)
                fAwxifBwxi = integral_fpfp(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):
                    for l in range(n):

                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gAuetagBu = integral_ffp(l, j, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAuetagBueta = integral_fpfp(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAugBu = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAugBueta = integral_ffp(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                        gAvetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvetagBveta = integral_fpfp(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1v, y1vr, y2v, y2vr)
                        gAvgBv = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBveta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                        gAvgBw = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAvgBweta = integral_ffp(j, l, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetagBv = integral_ffp(l, j, y1v, y1vr, y2v, y2vr, y1w, y1wr, y2w, y2wr)
                        gAwetagBw = integral_ffp(l, j, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwetagBweta = integral_fpfp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBv = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1v, y1vr, y2v, y2vr)
                        gAwgBveta = integral_ffp(j, l, y1w, y1wr, y2w, y2wr, y1v, y1vr, y2v, y2vr)
                        gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)
                        gAwgBweta = integral_ffp(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                        c += 1
                        kG0r[c] = row+0
                        kG0c[c] = col+0
                        kG0v[c] += Nxx*b*fAuxifBuxi*gAugBu/a + Nxy*fAufBuxi*gAuetagBu + Nxy*fAuxifBu*gAugBueta + Nyy*a*fAufBu*gAuetagBueta/b
                        c += 1
                        kG0r[c] = row+1
                        kG0c[c] = col+1
                        kG0v[c] += Nxx*b*fAvxifBvxi*gAvgBv/a + Nxy*fAvfBvxi*gAvetagBv + Nxy*fAvxifBv*gAvgBveta + 0.25*Nyy*a*b*fAvfBv*gAvgBv/(radius*radius) + Nyy*a*fAvfBv*gAvetagBveta/b
                        c += 1
                        kG0r[c] = row+1
                        kG0c[c] = col+2
                        kG0v[c] += 0.5*(-Nxy*b*fAvfBwxi*gAvgBw + Nxy*b*fAvxifBw*gAvgBw + Nyy*a*fAvfBw*gAvetagBw - Nyy*a*fAvfBw*gAvgBweta)/radius
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+1
                        kG0v[c] += 0.5*(Nxy*b*fAwfBvxi*gAwgBv - Nxy*b*fAwxifBv*gAwgBv - Nyy*a*fAwfBv*gAwetagBv + Nyy*a*fAwfBv*gAwgBveta)/radius
                        c += 1
                        kG0r[c] = row+2
                        kG0c[c] = col+2
                        kG0v[c] += Nxx*b*fAwxifBwxi*gAwgBw/a + Nxy*fAwfBwxi*gAwetagBw + Nxy*fAwxifBw*gAwgBweta + 0.25*Nyy*a*b*fAwfBw*gAwgBw/(radius*radius) + Nyy*a*fAwfBw*gAwetagBweta/b

    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))

    return kG0
