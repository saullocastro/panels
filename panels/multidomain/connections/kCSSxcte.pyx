#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Kernels of the skin-skin connection ``'SSxcte'``

Penalty stiffness matrices of the connection along `y` between the edge
`x_1 =` ``xcte1`` of ``p1`` and the edge `x_2 =` ``xcte2`` of
``p2``, two domains with the same axes, for the models based on the classical
laminated plate theory (CLPT), with 3 DOFs `u, v, w` per term:
:func:`.fkCSSxcte11` gives the block of ``p1``, :func:`.fkCSSxcte12` the
coupling block, rows of ``p1`` and columns of ``p2``, and
:func:`.fkCSSxcte22` the block of ``p2``. The two edges have the same
length, that of the edge of ``p1``. The models based on shear deformation
theories, with 5 DOFs per term, use :func:`.kCsdt.fkCSSxcte_sdt`. With the
default ``conn_method='null-space'`` of :class:`.MultiDomain` the same
quantities are imposed exactly, see
:mod:`panels.multidomain.connections.nullspace`.

Notes
-----
The penalty energy of the connection is

.. math::

    U = \frac{1}{2} \int_0^{b_1} k_t \left[ (u_1 - u_2)^2 + (v_1 - v_2)^2
        + (w_1 - w_2)^2 \right] + k_r \left( w_{1,x} - w_{2,x}
        \right)^2 dy

with the fields of ``p1`` at `x_1 = x_{cte1}` and those of ``p2``
at `x_2 = x_{cte2}`. The rotation penalty acts on the slope normal
to the edge, the tangential slope `w_{,y}` being continuous when `w` is.

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
\partial/\partial \xi`. The expressions of the kernels were derived
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
    double f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r,
                    double xi2t, double xi2r) nogil


def fkCSSxcte11(double kt, double kr, object p1, double xcte1, int size, int row0, int col0):
    r"""
    Block of ``p1`` of the skin-skin connection ``'SSxcte'``

    The penalty terms of the connection that involve only ``p1``, for
    the models with 3 DOFs per term: the products of the approximation functions
    of ``p1`` are integrated along `y` over the length of its edge and
    evaluated at ``xcte1`` across it.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        First panel of the connection.
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
    kCSSxcte11 : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p11``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCSSxcte`, see its Notes:

    .. math::

        \boldsymbol{K}_{11} = \frac{b_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q1}^\top \boldsymbol{N}_{q1}
        + k_r \frac{4}{a_1^2} \boldsymbol{N}_{w1,\xi}^\top \boldsymbol{N}_{w1,\xi}
        \, d\eta

    with the functions of ``p1`` at `\xi_1 = 2 x_{cte1}/a_1 - 1`.

    """
    cdef double [:, ::1] ftab1
    cdef int i1, k1, j1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double x1u1, x1ur1, x2u1, x2ur1
    cdef double x1v1, x1vr1, x2v1, x2vr1
    cdef double x1w1, x1wr1, x2w1, x2wr1
    cdef double y1u1, y1ur1, y2u1, y2ur1
    cdef double y1v1, y1vr1, y2v1, y2vr1
    cdef double y1w1, y1wr1, y2w1, y2wr1

    cdef long [:] kCSSxcte11r, kCSSxcte11c
    cdef double [:] kCSSxcte11v

    cdef double xicte1
    cdef double f1Au, f1Bu, f1Av, f1Bv, f1Aw, f1Bw, f1Aw_xi, f1Bw_xi
    cdef double g1Aug1Bu, g1Avg1Bv, g1Awg1Bw

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

    # Converting to natural CS
    xicte1 = 2*xcte1/a1 - 1.

    fdim = 3*m1*n1*m1*n1

    kCSSxcte11r = np.zeros((fdim,), dtype=INT)
    kCSSxcte11c = np.zeros((fdim,), dtype=INT)
    kCSSxcte11v = np.zeros((fdim,), dtype=DOUBLE)

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
                g1Aug1Bu = integral_ff(j1, l1, y1u1, y1ur1, y2u1, y2ur1, y1u1, y1ur1, y2u1, y2ur1)
                g1Avg1Bv = integral_ff(j1, l1, y1v1, y1vr1, y2v1, y2vr1, y1v1, y1vr1, y2v1, y2vr1)
                g1Awg1Bw = integral_ff(j1, l1, y1w1, y1wr1, y2w1, y2wr1, y1w1, y1wr1, y2w1, y2wr1)

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
                        kCSSxcte11r[c] = row+0
                        kCSSxcte11c[c] = col+0
                        kCSSxcte11v[c] += 0.5*b1*f1Au*f1Bu*g1Aug1Bu*kt
                        c += 1
                        kCSSxcte11r[c] = row+1
                        kCSSxcte11c[c] = col+1
                        kCSSxcte11v[c] += 0.5*b1*f1Av*f1Bv*g1Avg1Bv*kt
                        c += 1
                        kCSSxcte11r[c] = row+2
                        kCSSxcte11c[c] = col+2
                        kCSSxcte11v[c] += 0.5*b1*kt*(f1Aw*f1Bw*g1Awg1Bw + 4*f1Aw_xi*f1Bw_xi*g1Awg1Bw*kr/((a1*a1)*kt))

    kCSSxcte11 = coo_matrix((kCSSxcte11v, (kCSSxcte11r, kCSSxcte11c)), shape=(size, size))

    return kCSSxcte11


def fkCSSxcte12(double kt, double kr, object p1, object p2,
                double xcte1, double xcte2,
                int size, int row0, int col0):
    r"""
    Coupling block of the skin-skin connection ``'SSxcte'``

    The penalty terms of the connection that couple ``p1`` (rows) and
    ``p2`` (columns), for the models with 3 DOFs per term: the products of the
    approximation functions are integrated along `y` over the length of the
    edge of ``p1`` and evaluated at ``xcte1`` in ``p1`` and ``xcte2`` in
    ``p2`` across it. The whole block is returned.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        First panel of the connection.
    p2 : :class:`.Shell`
        Second panel of the connection, whose edge has the length of the
        edge of ``p1``.
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
    kCSSxcte12 : scipy.sparse.coo_matrix
        Coupling block, rows of ``p1`` and columns of ``p2``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCSSxcte`, see its Notes:

    .. math::

        \boldsymbol{K}_{12} = -\frac{b_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q1}^\top \boldsymbol{N}_{q2}
        + k_r \frac{4}{a_1 a_2} \boldsymbol{N}_{w1,\xi}^\top \boldsymbol{N}_{w2,\xi}
        \, d\eta

    with the functions of ``p1`` at `\xi_1 = 2 x_{cte1}/a_1 - 1` and
    those of ``p2`` at `\xi_2 = 2 x_{cte2}/a_2 - 1`.

    """
    cdef double [:, ::1] ftab1, ftab2
    cdef int i1, k2, j1, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, a2, b1, b2
    cdef double x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCSSxcte12r, kCSSxcte12c
    cdef double [:] kCSSxcte12v

    cdef double xicte1, xicte2
    cdef double g1Aug2Bu, g1Avg2Bv, g1Awg2Bw
    cdef double f1Au, f2Bu, f1Av, f2Bv, f1Aw, f2Bw, f1Aw_xi, f2Bw_xi

    a1 = p1.a
    b1 = p1.b
    a2 = p2.a
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

    fdim = 3*m1*n1*m2*n2

    kCSSxcte12r = np.zeros((fdim,), dtype=INT)
    kCSSxcte12c = np.zeros((fdim,), dtype=INT)
    kCSSxcte12v = np.zeros((fdim,), dtype=DOUBLE)

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
                g1Aug2Bu = integral_ff(j1, l2, y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2)
                g1Avg2Bv = integral_ff(j1, l2, y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2)
                g1Awg2Bw = integral_ff(j1, l2, y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2)

                for i1 in range(m1):
                    f1Au = ftab1[0, i1]
                    f1Av = ftab1[1, i1]
                    f1Aw = ftab1[2, i1]
                    f1Aw_xi = ftab1[3, i1]

                    for k2 in range(m2):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry
                        #if row > col:
                            #continue

                        f2Bu = ftab2[0, k2]
                        f2Bv = ftab2[1, k2]
                        f2Bw = ftab2[2, k2]
                        f2Bw_xi = ftab2[3, k2]

                        c += 1
                        kCSSxcte12r[c] = row+0
                        kCSSxcte12c[c] = col+0
                        kCSSxcte12v[c] += -0.5*b1*f1Au*f2Bu*g1Aug2Bu*kt
                        c += 1
                        kCSSxcte12r[c] = row+1
                        kCSSxcte12c[c] = col+1
                        kCSSxcte12v[c] += -0.5*b1*f1Av*f2Bv*g1Avg2Bv*kt
                        c += 1
                        kCSSxcte12r[c] = row+2
                        kCSSxcte12c[c] = col+2
                        kCSSxcte12v[c] += -0.5*b1*kt*(f1Aw*f2Bw*g1Awg2Bw + 4*f1Aw_xi*f2Bw_xi*g1Awg2Bw*kr/(a1*a2*kt))

    kCSSxcte12 = coo_matrix((kCSSxcte12v, (kCSSxcte12r, kCSSxcte12c)), shape=(size, size))

    return kCSSxcte12


def fkCSSxcte22(double kt, double kr, object p1, object p2,
                double xcte2,
                int size, int row0, int col0):
    r"""
    Block of ``p2`` of the skin-skin connection ``'SSxcte'``

    The penalty terms of the connection that involve only ``p2``, for
    the models with 3 DOFs per term: the products of the approximation functions
    of ``p2`` are integrated along `y` over the length of the edge of ``p1``
    and evaluated at ``xcte2`` across it.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        First panel of the connection, which gives the length of the edge.
    p2 : :class:`.Shell`
        Second panel of the connection.
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
    kCSSxcte22 : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p22``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCSSxcte`, see its Notes:

    .. math::

        \boldsymbol{K}_{22} = \frac{b_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q2}^\top \boldsymbol{N}_{q2}
        + k_r \frac{4}{a_2^2} \boldsymbol{N}_{w2,\xi}^\top \boldsymbol{N}_{w2,\xi}
        \, d\eta

    with the functions of ``p2`` at `\xi_2 = 2 x_{cte2}/a_2 - 1`.

    """
    cdef double [:, ::1] ftab2
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double b1, a2, b2
    cdef double x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCSSxcte22r, kCSSxcte22c
    cdef double [:] kCSSxcte22v

    cdef double xicte2
    cdef double f2Au, f2Bu, f2Av, f2Bv, f2Aw, f2Bw, f2Aw_xi, f2Bw_xi
    cdef double g2Aug2Bu, g2Avg2Bv, g2Awg2Bw

    b1 = p1.b
    a2 = p2.a
    b2 = p2.b
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

    kCSSxcte22r = np.zeros((fdim,), dtype=INT)
    kCSSxcte22c = np.zeros((fdim,), dtype=INT)
    kCSSxcte22v = np.zeros((fdim,), dtype=DOUBLE)

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
                g2Aug2Bu = integral_ff(j2, l2, y1u2, y1ur2, y2u2, y2ur2, y1u2, y1ur2, y2u2, y2ur2)
                g2Avg2Bv = integral_ff(j2, l2, y1v2, y1vr2, y2v2, y2vr2, y1v2, y1vr2, y2v2, y2vr2)
                g2Awg2Bw = integral_ff(j2, l2, y1w2, y1wr2, y2w2, y2wr2, y1w2, y1wr2, y2w2, y2wr2)

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
                        kCSSxcte22r[c] = row+0
                        kCSSxcte22c[c] = col+0
                        kCSSxcte22v[c] += 0.5*b1*f2Au*f2Bu*g2Aug2Bu*kt
                        c += 1
                        kCSSxcte22r[c] = row+1
                        kCSSxcte22c[c] = col+1
                        kCSSxcte22v[c] += 0.5*b1*f2Av*f2Bv*g2Avg2Bv*kt
                        c += 1
                        kCSSxcte22r[c] = row+2
                        kCSSxcte22c[c] = col+2
                        kCSSxcte22v[c] += 0.5*b1*kt*(f2Aw*f2Bw*g2Awg2Bw + 4*f2Aw_xi*f2Bw_xi*g2Awg2Bw*kr/((a2*a2)*kt))

    kCSSxcte22 = coo_matrix((kCSSxcte22v, (kCSSxcte22r, kCSSxcte22c)), shape=(size, size))

    return kCSSxcte22
