#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Kernels of the base-flange connection ``'BFxcte'``

Penalty stiffness matrices of the connection along `y` between the edge
`x_1 =` ``xcte1`` of the stiffener's base, or skin, ``p1``, and the
edge `x_2 =` ``xcte2`` of the flange ``p2``, normal to ``p1``, the
local axes of the flange being those of ``p1`` rotated by 90 degrees about
the common axis `y`. For the models based on the classical laminated
plate theory (CLPT), with 3 DOFs `u, v, w` per term, :func:`.fkCBFxcte11`
gives the block of the base, :func:`.fkCBFxcte12` the coupling block,
rows of the base and columns of the flange, and :func:`.fkCBFxcte22` the
block of the flange. For the models based on shear deformation theories,
e.g. ``'plate_fsdt_donnell'`` or ``'cylshell_tsdt_sanders'``, with 5 DOFs
`u, v, w, \phi_x, \phi_y` per term, the same blocks are given by
:func:`.fkCBFxcte11_sdt`, :func:`.fkCBFxcte12_sdt` and
:func:`.fkCBFxcte22_sdt`. The blocks of a single panel are returned as
their upper triangle. With the default ``conn_method='null-space'`` of
:class:`.MultiDomain` the same quantities are imposed exactly, see
:mod:`panels.multidomain.connections.nullspace`.

Notes
-----
With the local axes of the flange rotated by 90 degrees about the common
axis `y`, the penalty energy is:

.. math::

    U = \frac{1}{2} \int_0^{b_1} k_t \left[ (u_1 - w_2)^2 + (v_1 - v_2)^2 +
        (w_1 + u_2)^2 \right] + k_r \left( \omega_1 - \omega_2 \right)^2 dy

where `\omega_i` is the rotation of the normal of each panel about `y`,
positive with the right-hand rule in its own local axes. For the models
based on the CLPT, `\omega_i = -w_{i,x}`, also with the kinematics of
Sanders-Koiter, whose rotation about `y` has no term of the curvature.

For the models based on shear deformation theories, `\omega_i =
\theta_{x,i}`, the rotation at the reference surface, `z = 0`, `\theta_x =
\phi_x - (4 d^2/h^2)(\phi_x + w_{,x})` with the offset `d` and the
thickness `h` of each panel for the TSDT, `\theta_x = \phi_x` otherwise,
also with the kinematics of Sanders-Koiter, see :func:`.kCsdt._rotation_x`,
and the rotation penalty is `k_r (\theta_{x,1} - \theta_{x,2})^2`. For the
TSDT, the derivative `w_{,x}` is not penalized on its own: at a T-joint
only the rotation of the normals is common to the two panels. The rotation
`\phi_y` of each panel is a drilling rotation of the other and is not
penalized either.

Each field of a panel is approximated as `q = \boldsymbol{N}_q
\boldsymbol{c}`, `q = u, v, w`, with `\boldsymbol{N}_q` the products `f_i(\xi)
g_j(\eta)` of the Legendre-based (Bardell) approximation functions of the
field, with its boundary flags, along `\xi = 2x/a - 1` and `\eta = 2y/b -
1` [1]_, and `\boldsymbol{c}` the Ritz constants of the panel, those of the
term `(i, j)` at ``3*(j*m + i)`` plus ``0``, ``1``, ``2`` for `u, v, w`. The
blocks of the connection matrix are the second derivatives of `U` with
respect to the Ritz constants of the two panels, `\boldsymbol{K}_{11}`,
`\boldsymbol{K}_{12}` and `\boldsymbol{K}_{22}`; `\boldsymbol{K}_{21} =
\boldsymbol{K}_{12}^\top` is not computed and the blocks of a single panel
are returned as their upper triangle, the matrix of the assembly being
symmetrized afterwards. Along the connection `\int_0^{b_1} dy = (b_1/2)
\int_{-1}^{1} d\eta`, computed with the closed-form integrals of the products
of Bardell's functions, the two edges sharing the natural coordinate `\eta`;
across it the functions are evaluated at the natural coordinate of the
connection in each panel, and `\partial/\partial x = (2/a)
\partial/\partial \xi`. For the models with 5 DOFs per term the
Ritz constants of the term `(i, j)` are at ``5*(j*m + i)`` plus ``0`` to
``4`` for `u, v, w, \phi_x, \phi_y`. The expressions of the kernels were derived
symbolically with SymPy.

References
----------
.. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
   models to Address Linear Buckling and Vibration of Stiffened Composite
   Panels with Debonding Defect, Composite Structures, 160, 232-247, 2017.
   https://doi.org/10.1016/j.compstruct.2016.10.026

"""
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE


cdef int DOF = 3

cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil


def fkCBFxcte11(double kt, double kr, object p1, double xcte1,
          int size, int row0, int col0):
    r"""
    Block of the base ``p1`` of the base-flange connection ``'BFxcte'``

    The penalty terms of the connection that involve only the base, or skin, ``p1``, for the
    models with 3 DOFs per term: the products of the approximation functions of
    ``p1`` are integrated along `y` over the length of the edge of ``p1``
    and evaluated at ``xcte1`` across it.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Base, or skin.
    xcte1 : float
        Coordinate `x_1` of the connection in ``p1``, length, between ``0``
        and ``p1.a``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0 : int
        First row of the block in the matrix of the assembly, the
        ``row_start`` of the panel of the rows.
    col0 : int
        First column of the block in the matrix of the assembly, the
        ``col_start`` of the panel of the columns.

    Returns
    -------
    kCBFxcte11 : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p1``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFxcte`, see its Notes:

    .. math::

        \boldsymbol{K}_{11} = \frac{b_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q1}^\top \boldsymbol{N}_{q1}
        + k_r \boldsymbol{N}_{\omega 1}^\top \boldsymbol{N}_{\omega 1} \, d\eta

    with the functions of ``p1`` at `\xi_1 = 2 x_{cte1}/a_1 - 1`,
    and
    `\boldsymbol{N}_{\omega 1} = -(2/a_1) \boldsymbol{N}_{w1,\xi}`, the rotation
    `\omega_1 = -w_{1,x}` about `y`.

    """
    cdef double [:, ::1] ftab1
    cdef int i1, j1, k1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double x1u1, x1ur1, x2u1, x2ur1
    cdef double x1v1, x1vr1, x2v1, x2vr1
    cdef double x1w1, x1wr1, x2w1, x2wr1
    cdef double y1u1, y1ur1, y2u1, y2ur1
    cdef double y1v1, y1vr1, y2v1, y2vr1
    cdef double y1w1, y1wr1, y2w1, y2wr1

    cdef long [:] kCBFxcte11r, kCBFxcte11c
    cdef double [:] kCBFxcte11v

    cdef double xicte1
    cdef double g1Auf1Bu, g1Avf1Bv, g1Awf1Bw
    cdef double f1Au, f1Av, f1Aw, f1Aw_xi, f1Bu, f1Bv, f1Bw, f1Bw_xi

    a1 = p1.a
    b1 = p1.b
    m1 = p1.m
    n1 = p1.n
    x1u1 = p1.x1u ; x1ur1 = p1.x1ur ; x2u1 = p1.x2u ; x2ur1 = p1.x2ur
    x1v1 = p1.x1v ; x1vr1 = p1.x1vr ; x2v1 = p1.x2v ; x2vr1 = p1.x2vr
    x1w1 = p1.x1w ; x1wr1 = p1.x1wr ; x2w1 = p1.x2w ; x2wr1 = p1.x2wr
    y1u1 = p1.y1u ; y1ur1 = p1.y1ur ; y2u1 = p1.y2u ; y2ur1 = p1.y2ur
    y1v1 = p1.y1v ; y1vr1 = p1.y1vr ; y2v1 = p1.y2v ; y2vr1 = p1.y2vr
    y1w1 = p1.y1w ; y1wr1 = p1.y1wr ; y2w1 = p1.y2w ; y2wr1 = p1.y2wr

    xicte1 = 2*xcte1/a1 - 1.

    fdim = 3*m1*n1*m1*n1

    kCBFxcte11r = np.zeros((fdim,), dtype=INT)
    kCBFxcte11c = np.zeros((fdim,), dtype=INT)
    kCBFxcte11v = np.zeros((fdim,), dtype=DOUBLE)

    ftab1 = np.zeros((4, m1), dtype=DOUBLE)

    with nogil:
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for i1 in range(m1):
            ftab1[0, i1] = f(i1, xicte1, x1u1, x1ur1, x2u1, x2ur1)
            ftab1[1, i1] = f(i1, xicte1, x1v1, x1vr1, x2v1, x2vr1)
            ftab1[2, i1] = f(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)
            ftab1[3, i1] = fp(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)

        for j1 in range(n1):
            for l1 in range(n1):
                g1Auf1Bu = integral_ff(j1, l1, y1u1, y1ur1, y2u1, y2ur1, y1u1, y1ur1, y2u1, y2ur1)
                g1Avf1Bv = integral_ff(j1, l1, y1v1, y1vr1, y2v1, y2vr1, y1v1, y1vr1, y2v1, y2vr1)
                g1Awf1Bw = integral_ff(j1, l1, y1w1, y1wr1, y2w1, y2wr1, y1w1, y1wr1, y2w1, y2wr1)

                for i1 in range(m1):
                    f1Au = ftab1[0, i1]
                    f1Av = ftab1[1, i1]
                    f1Aw = ftab1[2, i1]
                    f1Aw_xi = ftab1[3, i1]

                    for k1 in range(m1):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        f1Bu = ftab1[0, k1]
                        f1Bv = ftab1[1, k1]
                        f1Bw = ftab1[2, k1]
                        f1Bw_xi = ftab1[3, k1]

                        c += 1
                        kCBFxcte11r[c] = row+0
                        kCBFxcte11c[c] = col+0
                        kCBFxcte11v[c] += 0.5*b1*f1Au*f1Bu*g1Auf1Bu*kt
                        c += 1
                        kCBFxcte11r[c] = row+1
                        kCBFxcte11c[c] = col+1
                        kCBFxcte11v[c] += 0.5*b1*f1Av*f1Bv*g1Avf1Bv*kt
                        c += 1
                        kCBFxcte11r[c] = row+2
                        kCBFxcte11c[c] = col+2
                        kCBFxcte11v[c] += 0.5*b1*kt*(f1Aw*f1Bw*g1Awf1Bw + 4*f1Aw_xi*f1Bw_xi*g1Awf1Bw*kr/((a1*a1)*kt))

    kCBFxcte11 = coo_matrix((kCBFxcte11v, (kCBFxcte11r, kCBFxcte11c)), shape=(size, size))

    return kCBFxcte11


def fkCBFxcte12(double kt, double kr, object p1, object p2,
          double xcte1, double xcte2,
          int size, int row0, int col0):
    r"""
    Coupling block of the base-flange connection ``'BFxcte'``

    The penalty terms of the connection that couple the base ``p1``
    (rows) and the flange ``p2`` (columns), for the models with 3 DOFs per term:
    the products of the approximation functions are integrated along `y`
    over the length of the edge of ``p1`` and evaluated at ``xcte1`` in
    ``p1`` and ``xcte2`` in ``p2`` across it. The whole block is
    returned.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Base, or skin.
    p2 : :class:`.Shell`
        Flange, whose edge has the length of the edge of ``p1``.
    xcte1 : float
        Coordinate `x_1` of the connection in ``p1``, length, between ``0``
        and ``p1.a``.
    xcte2 : float
        Coordinate `x_2` of the connection in ``p2``, length, between ``0``
        and ``p2.a``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0 : int
        First row of the block in the matrix of the assembly, the
        ``row_start`` of the panel of the rows.
    col0 : int
        First column of the block in the matrix of the assembly, the
        ``col_start`` of the panel of the columns.

    Returns
    -------
    kCBFxcte12 : scipy.sparse.coo_matrix
        Coupling block, rows of ``p1`` and columns of ``p2``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFxcte`, see its Notes:

    .. math::

        \boldsymbol{K}_{12} = -\frac{b_1}{2} \int_{-1}^{1} k_t \left(
        \boldsymbol{N}_{u1}^\top \boldsymbol{N}_{w2} + \boldsymbol{N}_{v1}^\top \boldsymbol{N}_{v2}
        - \boldsymbol{N}_{w1}^\top \boldsymbol{N}_{u2} \right)
        + k_r \boldsymbol{N}_{\omega 1}^\top \boldsymbol{N}_{\omega 2} \, d\eta

    with the functions of ``p1`` at `\xi_1 = 2 x_{cte1}/a_1 - 1` and
    those of ``p2`` at `\xi_2 = 2 x_{cte2}/a_2 - 1`, and
    `\boldsymbol{N}_{\omega p} = -(2/a_p) \boldsymbol{N}_{wp,\xi}`, the
    rotation `\omega_p = -w_{p,x}` of each panel about `y`.

    """
    cdef double [:, ::1] ftab1, ftab2
    cdef int i1, j1, k2, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, a2, b1, b2
    cdef double x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCBFxcte12r, kCBFxcte12c
    cdef double [:] kCBFxcte12v

    cdef double xicte1, xicte2
    cdef double g1Auf2Bw, g1Avf2Bv, g1Awf2Bu, g1Awf2Bw
    cdef double f1Au, f1Av, f1Aw, f1Aw_xi, f2Bu, f2Bv, f2Bw, f2Bw_xi

    a1 = p1.a
    a2 = p2.a
    b1 = p1.b
    b2 = p2.b
    m1 = p1.m
    n1 = p1.n
    m2 = p2.m
    n2 = p2.n
    x1u1 = p1.x1u ; x1ur1 = p1.x1ur ; x2u1 = p1.x2u ; x2ur1 = p1.x2ur
    x1v1 = p1.x1v ; x1vr1 = p1.x1vr ; x2v1 = p1.x2v ; x2vr1 = p1.x2vr
    x1w1 = p1.x1w ; x1wr1 = p1.x1wr ; x2w1 = p1.x2w ; x2wr1 = p1.x2wr
    y1u1 = p1.y1u ; y1ur1 = p1.y1ur ; y2u1 = p1.y2u ; y2ur1 = p1.y2ur
    y1v1 = p1.y1v ; y1vr1 = p1.y1vr ; y2v1 = p1.y2v ; y2vr1 = p1.y2vr
    y1w1 = p1.y1w ; y1wr1 = p1.y1wr ; y2w1 = p1.y2w ; y2wr1 = p1.y2wr

    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    xicte1 = 2*xcte1/a1 - 1.
    xicte2 = 2*xcte2/a2 - 1.

    fdim = 4*m1*n1*m2*n2

    kCBFxcte12r = np.zeros((fdim,), dtype=INT)
    kCBFxcte12c = np.zeros((fdim,), dtype=INT)
    kCBFxcte12v = np.zeros((fdim,), dtype=DOUBLE)

    ftab1 = np.zeros((4, m1), dtype=DOUBLE)
    ftab2 = np.zeros((4, m2), dtype=DOUBLE)

    with nogil:
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for i1 in range(m1):
            ftab1[0, i1] = f(i1, xicte1, x1u1, x1ur1, x2u1, x2ur1)
            ftab1[1, i1] = f(i1, xicte1, x1v1, x1vr1, x2v1, x2vr1)
            ftab1[2, i1] = f(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)
            ftab1[3, i1] = fp(i1, xicte1, x1w1, x1wr1, x2w1, x2wr1)
        for k2 in range(m2):
            ftab2[0, k2] = f(k2, xicte2, x1u2, x1ur2, x2u2, x2ur2)
            ftab2[1, k2] = f(k2, xicte2, x1v2, x1vr2, x2v2, x2vr2)
            ftab2[2, k2] = f(k2, xicte2, x1w2, x1wr2, x2w2, x2wr2)
            ftab2[3, k2] = fp(k2, xicte2, x1w2, x1wr2, x2w2, x2wr2)

        for j1 in range(n1):
            for l2 in range(n2):
                g1Auf2Bw = integral_ff(j1, l2, y1u1, y1ur1, y2u1, y2ur1, y1w2, y1wr2, y2w2, y2wr2)
                g1Avf2Bv = integral_ff(j1, l2, y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2)
                g1Awf2Bu = integral_ff(j1, l2, y1w1, y1wr1, y2w1, y2wr1, y1u2, y1ur2, y2u2, y2ur2)
                g1Awf2Bw = integral_ff(j1, l2, y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2)

                for i1 in range(m1):
                    f1Au = ftab1[0, i1]
                    f1Av = ftab1[1, i1]
                    f1Aw = ftab1[2, i1]
                    f1Aw_xi = ftab1[3, i1]

                    for k2 in range(m2):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        f2Bu = ftab2[0, k2]
                        f2Bv = ftab2[1, k2]
                        f2Bw = ftab2[2, k2]
                        f2Bw_xi = ftab2[3, k2]

                        c += 1
                        kCBFxcte12r[c] = row+0
                        kCBFxcte12c[c] = col+2
                        kCBFxcte12v[c] += -0.5*b1*f1Au*f2Bw*g1Auf2Bw*kt
                        c += 1
                        kCBFxcte12r[c] = row+1
                        kCBFxcte12c[c] = col+1
                        kCBFxcte12v[c] += -0.5*b1*f1Av*f2Bv*g1Avf2Bv*kt
                        c += 1
                        kCBFxcte12r[c] = row+2
                        kCBFxcte12c[c] = col+0
                        kCBFxcte12v[c] += 0.5*b1*f1Aw*f2Bu*g1Awf2Bu*kt
                        c += 1
                        kCBFxcte12r[c] = row+2
                        kCBFxcte12c[c] = col+2
                        kCBFxcte12v[c] += -2*b1*f1Aw_xi*f2Bw_xi*g1Awf2Bw*kr/(a1*a2)

    kCBFxcte12 = coo_matrix((kCBFxcte12v, (kCBFxcte12r, kCBFxcte12c)), shape=(size, size))

    return kCBFxcte12


def fkCBFxcte22(double kt, double kr, object p1, object p2, double xcte2,
          int size, int row0, int col0):
    r"""
    Block of the flange ``p2`` of the base-flange connection ``'BFxcte'``

    The penalty terms of the connection that involve only the flange ``p2``, for the
    models with 3 DOFs per term: the products of the approximation functions of
    ``p2`` are integrated along `y` over the length of the edge of ``p1``
    and evaluated at ``xcte2`` across it.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Base, or skin, which gives the length of the edge.
    p2 : :class:`.Shell`
        Flange.
    xcte2 : float
        Coordinate `x_2` of the connection in ``p2``, length, between ``0``
        and ``p2.a``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0 : int
        First row of the block in the matrix of the assembly, the
        ``row_start`` of the panel of the rows.
    col0 : int
        First column of the block in the matrix of the assembly, the
        ``col_start`` of the panel of the columns.

    Returns
    -------
    kCBFxcte22 : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p2``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFxcte`, see its Notes:

    .. math::

        \boldsymbol{K}_{22} = \frac{b_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q2}^\top \boldsymbol{N}_{q2}
        + k_r \boldsymbol{N}_{\omega 2}^\top \boldsymbol{N}_{\omega 2} \, d\eta

    with the functions of ``p2`` at `\xi_2 = 2 x_{cte2}/a_2 - 1`,
    and
    `\boldsymbol{N}_{\omega 2} = -(2/a_2) \boldsymbol{N}_{w2,\xi}`, the rotation
    `\omega_2 = -w_{2,x}` about `y`.

    """
    cdef double [:, ::1] ftab2
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double b1, a2
    cdef double x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCBFxcte22r, kCBFxcte22c
    cdef double [:] kCBFxcte22v

    cdef double xicte2
    cdef double g2Auf2Bu, g2Avf2Bv, g2Awf2Bw
    cdef double f2Au, f2Bu, f2Av, f2Bv, f2Aw, f2Bw, f2Aw_xi, f2Bw_xi
    b1 = p1.b
    a2 = p2.a
    m2 = p2.m
    n2 = p2.n
    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    xicte2 = 2*xcte2/a2 - 1.

    fdim = 3*m2*n2*m2*n2

    kCBFxcte22r = np.zeros((fdim,), dtype=INT)
    kCBFxcte22c = np.zeros((fdim,), dtype=INT)
    kCBFxcte22v = np.zeros((fdim,), dtype=DOUBLE)

    ftab2 = np.zeros((4, m2), dtype=DOUBLE)

    with nogil:
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for i2 in range(m2):
            ftab2[0, i2] = f(i2, xicte2, x1u2, x1ur2, x2u2, x2ur2)
            ftab2[1, i2] = f(i2, xicte2, x1v2, x1vr2, x2v2, x2vr2)
            ftab2[2, i2] = f(i2, xicte2, x1w2, x1wr2, x2w2, x2wr2)
            ftab2[3, i2] = fp(i2, xicte2, x1w2, x1wr2, x2w2, x2wr2)

        for j2 in range(n2):
            for l2 in range(n2):
                g2Auf2Bu = integral_ff(j2, l2, y1u2, y1ur2, y2u2, y2ur2, y1u2, y1ur2, y2u2, y2ur2)
                g2Avf2Bv = integral_ff(j2, l2, y1v2, y1vr2, y2v2, y2vr2, y1v2, y1vr2, y2v2, y2vr2)
                g2Awf2Bw = integral_ff(j2, l2, y1w2, y1wr2, y2w2, y2wr2, y1w2, y1wr2, y2w2, y2wr2)

                for i2 in range(m2):
                    f2Au = ftab2[0, i2]
                    f2Av = ftab2[1, i2]
                    f2Aw = ftab2[2, i2]
                    f2Aw_xi = ftab2[3, i2]

                    for k2 in range(m2):
                        row = row0 + DOF*(j2*m2 + i2)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        f2Bu = ftab2[0, k2]
                        f2Bv = ftab2[1, k2]
                        f2Bw = ftab2[2, k2]
                        f2Bw_xi = ftab2[3, k2]

                        c += 1
                        kCBFxcte22r[c] = row+0
                        kCBFxcte22c[c] = col+0
                        kCBFxcte22v[c] += 0.5*b1*f2Au*f2Bu*g2Auf2Bu*kt
                        c += 1
                        kCBFxcte22r[c] = row+1
                        kCBFxcte22c[c] = col+1
                        kCBFxcte22v[c] += 0.5*b1*f2Av*f2Bv*g2Avf2Bv*kt
                        c += 1
                        kCBFxcte22r[c] = row+2
                        kCBFxcte22c[c] = col+2
                        kCBFxcte22v[c] += 0.5*b1*kt*(f2Aw*f2Bw*g2Awf2Bw + 4*f2Aw_xi*f2Bw_xi*g2Awf2Bw*kr/((a2*a2)*kt))

    kCBFxcte22 = coo_matrix((kCBFxcte22v, (kCBFxcte22r, kCBFxcte22c)), shape=(size, size))

    return kCBFxcte22


# ----------------------------------------------------------------------------
# Models based on shear deformation theories (FSDT and TSDT), 5 DOFs per term
# ----------------------------------------------------------------------------

cdef int DOF_SDT = 5
U, V, W, PHIX, PHIY = range(5)
FIELDS_SDT = ('u', 'v', 'w', 'phix', 'phiy')


def _check_sdt(*panels):
    r"""
    Check that the panels are models based on the FSDT or TSDT

    Parameters
    ----------
    *panels : :class:`.Shell`
        Panels whose attribute ``model`` is checked.

    Raises
    ------
    ValueError
        When the model of a panel is ``None`` or contains neither ``'fsdt'``
        nor ``'tsdt'``.

    """
    for p in panels:
        if p.model is None or not ('fsdt' in p.model or 'tsdt' in p.model):
            raise ValueError("Expected a model based on shear deformation "
                             "theories, got model '{0}'".format(p.model))


def _alpha(object p):
    r"""
    Factor `\alpha = 4 d^2/h^2` of the rotation at the reference surface

    Parameters
    ----------
    p : :class:`.Shell`
        Panel of a model based on the FSDT or TSDT, with its ``offset`` and
        ply thicknesses ``plyts``.

    Returns
    -------
    alpha : float
        `4 d^2/h^2` for a TSDT model, with the offset `d` and the thickness
        `h`, the sum of the ply thicknesses; ``0.`` for the FSDT.

    Notes
    -----
    With the TSDT field of :mod:`panels.multidomain.connections.kCsdt`, the
    rotation of the normal at the reference surface `z = 0` is `\theta_x =
    \phi_x - 3 c_1 d^2 (\phi_x + w_{,x}) = (1 - \alpha) \phi_x - \alpha
    w_{,x}`, with `c_1 = 4/(3 h^2)`, see :func:`.kCsdt._rotation_x`.

    """
    if 'tsdt' not in p.model:
        return 0.
    return 4.*p.offset**2/sum(p.plyts)**2


def _product_terms(double coeff, tA, tB):
    r"""
    Penalty terms of the product of two quantities

    Parameters
    ----------
    coeff : float
        Factor of the product, e.g. ``kr`` or ``-kr``.
    tA, tB : list
        Quantities `q_A` and `q_B` as lists of ``(dof, der, a)``, the DOF
        numbered ``0`` to ``4`` for `u, v, w, \phi_x, \phi_y`, ``der`` the order
        of the derivative across the connection, ``0`` or ``1``, and ``a`` the
        factor.

    Returns
    -------
    terms : list
        One ``(dofA, dofB, coeff*aA*aB, derA, derB)`` per pair of terms, as
        read by :func:`._block_sdt`.

    Notes
    -----
    With `q_A = \sum_s a_s \partial^{d_s} q_s` and `q_B = \sum_t a_t
    \partial^{d_t} q_t`, the product `\mathrm{coeff} \, q_A q_B` expands into
    the terms `\mathrm{coeff} \, a_s a_t \, \partial^{d_s} q_s \, \partial^{d_t}
    q_t`, each integrated by :func:`._block_sdt`.

    """
    return [(dA, dB, coeff*cA*cB, derA, derB)
            for dA, derA, cA in tA for dB, derB, cB in tB]


def _rotation_ref(object p):
    r"""
    Rotation of the normal about `y` at the reference surface

    Parameters
    ----------
    p : :class:`.Shell`
        Panel of a model based on the FSDT or TSDT.

    Returns
    -------
    terms : list
        The rotation as a list of ``(dof, der, a)``, see
        :func:`._product_terms`: `\phi_x` and, for a TSDT panel with an offset,
        the derivative of `w` across the connection.

    Notes
    -----
    `\theta_x = (1 - \alpha) \phi_x - \alpha w_{,x}`, with `\alpha` of
    :func:`._alpha`, see :func:`.kCsdt._rotation_x`; `w_{,x}` is the derivative
    across the connection.

    """
    a = _alpha(p)
    terms = [(PHIX, 0, 1. - a)]
    if a != 0:
        terms.append((W, 1, -a))
    return terms


def _flags_sdt(object p, fields, str direction):
    r"""
    Boundary flags of the approximation functions of some fields

    Parameters
    ----------
    p : :class:`.Shell`
        Panel.
    fields : list of int
        Fields, numbered ``0`` to ``4`` for `u, v, w, \phi_x, \phi_y`.
    direction : str
        ``'x'`` or ``'y'``.

    Returns
    -------
    flags : np.ndarray
        Shape ``(len(fields), 4)``, for each field the flags ``(1t, 1r, 2t,
        2r)`` of the edges 1 and 2 along ``direction``, e.g. ``p.x1u``,
        ``p.x1ur``, ``p.x2u``, ``p.x2ur`` for ``'x'`` and `u`.

    """
    out = np.zeros((len(fields), 4), dtype=DOUBLE)
    for t, field in enumerate(fields):
        name = FIELDS_SDT[field]
        out[t, 0] = getattr(p, direction + '1' + name)
        out[t, 1] = getattr(p, direction + '1' + name + 'r')
        out[t, 2] = getattr(p, direction + '2' + name)
        out[t, 3] = getattr(p, direction + '2' + name + 'r')
    return out


cdef void _terms_sdt(int nt, long [::1] dofA, long [::1] dofB,
        long [::1] derA, long [::1] derB,
        double [::1] coeff, double [:, ::1] alA, double [:, ::1] alB,
        double [:, ::1] acA, double [:, ::1] acB, int nalA, int nalB,
        int nacA, int nacB, double cteA, double cteB, double jac,
        int along_x, int upper, int row0, int col0,
        double [::1] gAs, double [::1] gBs,
        long [::1] r, long [::1] c, double [::1] v) noexcept nogil:
    #NOTE the terms computed here are described in _block_sdt(); ``al*`` are
    #     the flags of the functions along the connection and ``ac*`` those
    #     of the functions across it, ``nal*`` and ``nac*`` their numbers;
    #     the functions across it are evaluated once per term ``t`` in
    #     ``gAs`` and ``gBs``, see panels/dev/tabulate.py
    cdef int t, iA, iB, jA, jB, row, col, pos, mA, mB
    cdef double I, gA, gB
    pos = 0
    mA = nalA if along_x else nacA
    mB = nalB if along_x else nacB
    for t in range(nt):
        #NOTE the first derivative across the connection, with respect to
        #     the natural coordinate, when derA[t] or derB[t] is 1
        for jA in range(nacA):
            if derA[t] == 1:
                gAs[jA] = fp(jA, cteA, acA[t, 0], acA[t, 1], acA[t, 2], acA[t, 3])
            else:
                gAs[jA] = f(jA, cteA, acA[t, 0], acA[t, 1], acA[t, 2], acA[t, 3])
        for jB in range(nacB):
            if derB[t] == 1:
                gBs[jB] = fp(jB, cteB, acB[t, 0], acB[t, 1], acB[t, 2], acB[t, 3])
            else:
                gBs[jB] = f(jB, cteB, acB[t, 0], acB[t, 1], acB[t, 2], acB[t, 3])
        for iA in range(nalA):
            for iB in range(nalB):
                I = jac*integral_ff(iA, iB, alA[t, 0], alA[t, 1], alA[t, 2],
                        alA[t, 3], alB[t, 0], alB[t, 1], alB[t, 2], alB[t, 3])
                for jA in range(nacA):
                    gA = gAs[jA]
                    for jB in range(nacB):
                        gB = gBs[jB]
                        if along_x:
                            row = row0 + DOF_SDT*(jA*mA + iA) + dofA[t]
                            col = col0 + DOF_SDT*(jB*mB + iB) + dofB[t]
                        else:
                            row = row0 + DOF_SDT*(iA*mA + jA) + dofA[t]
                            col = col0 + DOF_SDT*(iB*mB + jB) + dofB[t]
                        if upper and row > col:
                            v[pos] = 0.
                        else:
                            v[pos] = coeff[t]*I*gA*gB
                        r[pos] = row
                        c[pos] = col
                        pos += 1


def _block_sdt(terms, object pA, object pB, double cteA, double cteB,
        str along, int upper, int size, int row0, int col0):
    r"""
    Block of the connection matrix between ``pA`` and ``pB``

    Each element ``(dofA, dofB, coeff)`` or ``(dofA, dofB, coeff, derA,
    derB)`` of ``terms`` adds the penalty term ``coeff`` times the integral of
    the product of the field ``dofA`` of ``pA``, or its first derivative across
    the connection when ``derA`` is ``1``, and the field ``dofB`` of ``pB``, or
    its derivative when ``derB`` is ``1``, with the DOFs `u, v, w, \phi_x,
    \phi_y` numbered from 0 to 4. The approximation functions along the
    connection are integrated over the length of ``pA``, and those across it
    are evaluated at the natural coordinates ``cteA`` and ``cteB``. The term
    `(i, j)` of a panel, with `i` along `x` and `j` along `y`, has its Ritz
    constants at ``5*(j*m + i)``.

    Parameters
    ----------
    terms : list of tuple
        Penalty terms, see above and :func:`._product_terms`.
    pA, pB : :class:`.Shell`
        Panels of the rows and of the columns, models with 5 DOFs per term.
    cteA, cteB : float
        Natural coordinates of the connection across it, in ``pA`` and
        ``pB``, between ``-1`` and ``1``.
    along : str
        Direction of the connection, ``'x'`` or ``'y'``.
    upper : int
        ``1`` to keep only the upper triangle, for the blocks of a single
        panel, ``0`` to keep the whole block.
    size : int
        Size of the stiffness matrix of the assembly.
    row0, col0 : int
        First row and first column of the block in the matrix of the assembly.

    Returns
    -------
    kC : scipy.sparse.coo_matrix
        The block, shape ``(size, size)``.

    Notes
    -----
    For a connection along `x`, each term adds

    .. math::

        \mathrm{coeff} \left(\frac{2}{b_A}\right)^{d_A}
        \left(\frac{2}{b_B}\right)^{d_B} \frac{a_A}{2} \int_{-1}^{1}
        f^A_i(\xi) f^B_k(\xi) \, d\xi \; \partial_\eta^{d_A} g^A_j(\eta_A) \,
        \partial_\eta^{d_B} g^B_l(\eta_B)

    to the entry of the Ritz constants `(i, j)` of ``pA`` and `(k, l)` of
    ``pB``, with `\eta_A`, `\eta_B` given by ``cteA``, ``cteB``, `d_A`, `d_B`
    the orders of the derivatives across the connection and `f`, `g` the
    Legendre-based (Bardell) functions of each field with its boundary flags
    [1]_; the integral along the connection is computed in closed form. For a
    connection along `y` the roles of `x` and `y` are exchanged.

    References
    ----------
    .. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
       models to Address Linear Buckling and Vibration of Stiffened Composite
       Panels with Debonding Defect, Composite Structures, 160, 232-247, 2017.
       https://doi.org/10.1016/j.compstruct.2016.10.026

    """
    cdef int nt, nalA, nalB, nacA, nacB, n
    cdef long [::1] dofA, dofB, derA, derB, r, c
    cdef double [::1] coeff, v, gAs, gBs
    cdef double [:, ::1] alA, alB, acA, acB
    cdef double jac
    cdef int along_x = along == 'x'
    across = 'y' if along_x else 'x'
    nt = len(terms)
    dofA = np.array([t[0] for t in terms], dtype=INT)
    dofB = np.array([t[1] for t in terms], dtype=INT)
    derA = np.array([t[3] if len(t) > 3 else 0 for t in terms], dtype=INT)
    derB = np.array([t[4] if len(t) > 3 else 0 for t in terms], dtype=INT)
    #NOTE d/dx = (2/a) d/dxi and d/dy = (2/b) d/deta across the connection
    LA = pA.b if along == 'x' else pA.a
    LB = pB.b if along == 'x' else pB.a
    coeff = np.array([t[2]*(2./LA)**(t[3] if len(t) > 3 else 0)
                      *(2./LB)**(t[4] if len(t) > 3 else 0) for t in terms],
                     dtype=DOUBLE)
    alA = _flags_sdt(pA, [t[0] for t in terms], along)
    alB = _flags_sdt(pB, [t[1] for t in terms], along)
    acA = _flags_sdt(pA, [t[0] for t in terms], across)
    acB = _flags_sdt(pB, [t[1] for t in terms], across)
    if along == 'x':
        nalA, nalB, nacA, nacB = pA.m, pB.m, pA.n, pB.n
        #NOTE the connection spans the length of pA
        jac = pA.a/2.
    else:
        nalA, nalB, nacA, nacB = pA.n, pB.n, pA.m, pB.m
        jac = pA.b/2.
    n = nt*nalA*nalB*nacA*nacB
    r = np.zeros(n, dtype=INT)
    c = np.zeros(n, dtype=INT)
    v = np.zeros(n, dtype=DOUBLE)
    gAs = np.zeros(nacA, dtype=DOUBLE)
    gBs = np.zeros(nacB, dtype=DOUBLE)
    with nogil:
        _terms_sdt(nt, dofA, dofB, derA, derB, coeff, alA, alB, acA, acB, nalA, nalB,
                   nacA, nacB, cteA, cteB, jac, along_x, upper, row0,
                   col0, gAs, gBs, r, c, v)
    return coo_matrix((v, (r, c)), shape=(size, size))


def fkCBFxcte11_sdt(double kt, double kr, object p1, double xcte1,
          int size, int row0, int col0):
    r"""
    Block of the base of ``'BFxcte'``, models with 5 DOFs per term

    The penalty terms of the connection that involve only the base ``p1``,
    for the models based on the FSDT and TSDT: the differences of `u, v, w` with ``kt`` and the rotation of
    the normal about `y` at the reference surface with ``kr``, see
    :func:`._rotation_ref`. The products of the approximation functions are
    integrated along `y` over the length of the edge of ``p1`` and evaluated at
    ``xcte1`` across it.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Base, or skin.
    xcte1 : float
        Coordinate `x_1` of the connection in ``p1``, length, between ``0``
        and ``p1.a``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0 : int
        First row of the block in the matrix of the assembly, the
        ``row_start`` of the panel of the rows.
    col0 : int
        First column of the block in the matrix of the assembly, the
        ``col_start`` of the panel of the columns.

    Returns
    -------
    kCBFxcte11_sdt : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p1``, shape ``(size, size)``.

    Raises
    ------
    ValueError
        When a panel is not a model based on the FSDT or TSDT.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFxcte`, see its Notes:

    .. math::

        \boldsymbol{K}_{11} = \frac{b_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q1}^\top \boldsymbol{N}_{q1}
        + k_r \boldsymbol{N}_{\theta 1}^\top \boldsymbol{N}_{\theta 1} \, d\eta

    with the functions of ``p1`` at `\xi_1 = 2 x_{cte1}/a_1 - 1`,
    and
    `\boldsymbol{N}_{\theta 1} = (1 - \alpha_1) \boldsymbol{N}_{\phi_x 1} - \alpha_1
    (2/a_1) \boldsymbol{N}_{w1,\xi}`, the rotation `\theta_x` of the panel at the reference
    surface, with `\alpha` of :func:`._alpha`.
    The Ritz constants of the term `(i, j)` are at ``5*(j*m + i)``.

    """
    _check_sdt(p1)
    terms = ([(U, U, kt), (V, V, kt), (W, W, kt)]
             + _product_terms(kr, _rotation_ref(p1), _rotation_ref(p1)))
    cte = 2*xcte1/p1.a - 1.
    return _block_sdt(terms, p1, p1, cte, cte, 'y', 1, size, row0, col0)


def fkCBFxcte12_sdt(double kt, double kr, object p1, object p2,
          double xcte1, double xcte2,
          int size, int row0, int col0):
    r"""
    Coupling block of ``'BFxcte'``, models with 5 DOFs per term

    The penalty terms of the connection that couple the base ``p1`` (rows) and the flange ``p2`` (columns),
    for the models based on the FSDT and TSDT: the differences of `u, v, w` with ``kt`` and the rotation of
    the normal about `y` at the reference surface with ``kr``, see
    :func:`._rotation_ref`. The products of the approximation functions are
    integrated along `y` over the length of the edge of ``p1`` and evaluated at
    ``xcte1`` in ``p1`` and ``xcte2`` in ``p2`` across it. The whole block is returned.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Base, or skin.
    p2 : :class:`.Shell`
        Flange.
    xcte1 : float
        Coordinate `x_1` of the connection in ``p1``, length, between ``0``
        and ``p1.a``.
    xcte2 : float
        Coordinate `x_2` of the connection in ``p2``, length, between ``0``
        and ``p2.a``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0 : int
        First row of the block in the matrix of the assembly, the
        ``row_start`` of the panel of the rows.
    col0 : int
        First column of the block in the matrix of the assembly, the
        ``col_start`` of the panel of the columns.

    Returns
    -------
    kCBFxcte12_sdt : scipy.sparse.coo_matrix
        Coupling block, rows of ``p1`` and columns of ``p2``, shape ``(size, size)``.

    Raises
    ------
    ValueError
        When a panel is not a model based on the FSDT or TSDT.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFxcte`, see its Notes:

    .. math::

        \boldsymbol{K}_{12} = -\frac{b_1}{2} \int_{-1}^{1} k_t \left(
        \boldsymbol{N}_{u1}^\top \boldsymbol{N}_{w2} + \boldsymbol{N}_{v1}^\top \boldsymbol{N}_{v2}
        - \boldsymbol{N}_{w1}^\top \boldsymbol{N}_{u2} \right)
        + k_r \boldsymbol{N}_{\theta 1}^\top \boldsymbol{N}_{\theta 2} \, d\eta

    with the functions of ``p1`` at `\xi_1 = 2 x_{cte1}/a_1 - 1` and
    those of ``p2`` at `\xi_2 = 2 x_{cte2}/a_2 - 1`, and
    `\boldsymbol{N}_{\theta p} = (1 - \alpha_p) \boldsymbol{N}_{\phi_x p} - \alpha_p
    (2/a_p) \boldsymbol{N}_{wp,\xi}`, the rotation `\theta_x` of each panel at the reference
    surface, with `\alpha` of :func:`._alpha`.
    The Ritz constants of the term `(i, j)` are at ``5*(j*m + i)``.

    """
    _check_sdt(p1, p2)
    terms = ([(U, W, -kt), (V, V, -kt), (W, U, kt)]
             + _product_terms(-kr, _rotation_ref(p1), _rotation_ref(p2)))
    cte1 = 2*xcte1/p1.a - 1.
    cte2 = 2*xcte2/p2.a - 1.
    return _block_sdt(terms, p1, p2, cte1, cte2, 'y', 0, size, row0, col0)


def fkCBFxcte22_sdt(double kt, double kr, object p1, object p2,
          double xcte2,
          int size, int row0, int col0):
    r"""
    Block of the flange of ``'BFxcte'``, models with 5 DOFs per term

    The penalty terms of the connection that involve only the flange ``p2``,
    for the models based on the FSDT and TSDT: the differences of `u, v, w` with ``kt`` and the rotation of
    the normal about `y` at the reference surface with ``kr``, see
    :func:`._rotation_ref`. The products of the approximation functions are
    integrated along `y` over the length of the edge of ``p2`` and evaluated at
    ``xcte2`` across it.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Base, or skin.
    p2 : :class:`.Shell`
        Flange.
    xcte2 : float
        Coordinate `x_2` of the connection in ``p2``, length, between ``0``
        and ``p2.a``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0 : int
        First row of the block in the matrix of the assembly, the
        ``row_start`` of the panel of the rows.
    col0 : int
        First column of the block in the matrix of the assembly, the
        ``col_start`` of the panel of the columns.

    Returns
    -------
    kCBFxcte22_sdt : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p2``, shape ``(size, size)``.

    Raises
    ------
    ValueError
        When a panel is not a model based on the FSDT or TSDT.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFxcte`, see its Notes:

    .. math::

        \boldsymbol{K}_{22} = \frac{b_2}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q2}^\top \boldsymbol{N}_{q2}
        + k_r \boldsymbol{N}_{\theta 2}^\top \boldsymbol{N}_{\theta 2} \, d\eta

    with the functions of ``p2`` at `\xi_2 = 2 x_{cte2}/a_2 - 1`,
    and
    `\boldsymbol{N}_{\theta 2} = (1 - \alpha_2) \boldsymbol{N}_{\phi_x 2} - \alpha_2
    (2/a_2) \boldsymbol{N}_{w2,\xi}`, the rotation `\theta_x` of the panel at the reference
    surface, with `\alpha` of :func:`._alpha`.
    The Ritz constants of the term `(i, j)` are at ``5*(j*m + i)``.

    """
    _check_sdt(p1, p2)
    terms = ([(U, U, kt), (V, V, kt), (W, W, kt)]
             + _product_terms(kr, _rotation_ref(p2), _rotation_ref(p2)))
    cte = 2*xcte2/p2.a - 1.
    return _block_sdt(terms, p2, p2, cte, cte, 'y', 1, size, row0, col0)
