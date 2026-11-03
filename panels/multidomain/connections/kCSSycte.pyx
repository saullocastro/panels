#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Kernels of the skin-skin connection ``'SSycte'``

Penalty stiffness matrices of the connection along `x` between the edge
`y_1 =` ``ycte1`` of ``p1`` and the edge `y_2 =` ``ycte2`` of
``p2``, two domains with the same axes, for the models based on the classical
laminated plate theory (CLPT), with 3 DOFs `u, v, w` per term:
:func:`.fkCSSycte11` gives the block of ``p1``, :func:`.fkCSSycte12` the
coupling block, rows of ``p1`` and columns of ``p2``, and
:func:`.fkCSSycte22` the block of ``p2``. The two edges have the same
length, that of the edge of ``p1``. The models based on shear deformation
theories, with 5 DOFs per term, use :func:`.kCsdt.fkCSSycte_sdt`. With the
default ``conn_method='null-space'`` of :class:`.MultiDomain` the same
quantities are imposed exactly, see
:mod:`panels.multidomain.connections.nullspace`.

Notes
-----
The penalty energy of the connection is

.. math::

    U = \frac{1}{2} \int_0^{a_1} k_t \left[ (u_1 - u_2)^2 + (v_1 - v_2)^2
        + (w_1 - w_2)^2 \right] + k_r \left( w_{1,y} - w_{2,y}
        \right)^2 dx

with the fields of ``p1`` at `y_1 = y_{cte1}` and those of ``p2``
at `y_2 = y_{cte2}`. The rotation penalty acts on the slope normal
to the edge, the tangential slope `w_{,x}` being continuous when `w` is.

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
symmetrized afterwards. Along the connection `\int_0^{a_1} dx = (a_1/2)
\int_{-1}^{1} d\xi`, computed with the closed-form integrals of the products
of Bardell's functions, the two edges sharing the natural coordinate `\xi`;
across it the functions are evaluated at the natural coordinate of the
connection in each panel, and `\partial/\partial y = (2/b)
\partial/\partial \eta`. The expressions of the kernels were derived
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


def fkCSSycte11(double kt, double kr, object p1, double ycte1,
        int size, int row0, int col0):
    r"""
    Block of ``p1`` of the skin-skin connection ``'SSycte'``

    The penalty terms of the connection that involve only ``p1``, for
    the models with 3 DOFs per term: the products of the approximation functions
    of ``p1`` are integrated along `x` over the length of its edge and
    evaluated at ``ycte1`` across it.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        First panel of the connection.
    ycte1 : float
        Coordinate `y_1` of the connection in ``p1``, length, between ``0``
        and ``p1.b``.
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
    kCSSycte11 : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p11``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCSSycte`, see its Notes:

    .. math::

        \boldsymbol{K}_{11} = \frac{a_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q1}^\top \boldsymbol{N}_{q1}
        + k_r \frac{4}{b_1^2} \boldsymbol{N}_{w1,\eta}^\top \boldsymbol{N}_{w1,\eta}
        \, d\xi

    with the functions of ``p1`` at `\eta_1 = 2 y_{cte1}/b_1 - 1`.

    """
    cdef double [:, ::1] gtab1
    cdef int i1, k1, j1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double x1u1, x1ur1, x2u1, x2ur1
    cdef double x1v1, x1vr1, x2v1, x2vr1
    cdef double x1w1, x1wr1, x2w1, x2wr1
    cdef double y1u1, y1ur1, y2u1, y2ur1
    cdef double y1v1, y1vr1, y2v1, y2vr1
    cdef double y1w1, y1wr1, y2w1, y2wr1

    cdef long [:] kCSSycte11r, kCSSycte11c
    cdef double [:] kCSSycte11v

    cdef double etacte1
    cdef double f1Auf1Bu, f1Avf1Bv, f1Awf1Bw
    cdef double g1Au, g1Bu, g1Av, g1Bv, g1Aw, g1Bw, g1Aw_eta, g1Bw_eta

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

    etacte1 = 2*ycte1/b1 - 1.

    fdim = 3*m1*n1*m1*n1

    kCSSycte11r = np.zeros((fdim,), dtype=INT)
    kCSSycte11c = np.zeros((fdim,), dtype=INT)
    kCSSycte11v = np.zeros((fdim,), dtype=DOUBLE)

    gtab1 = np.zeros((4, n1), dtype=DOUBLE)

    with nogil:
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j1 in range(n1):
            gtab1[0, j1] = f(j1, etacte1, y1u1, y1ur1, y2u1, y2ur1)
            gtab1[1, j1] = f(j1, etacte1, y1v1, y1vr1, y2v1, y2vr1)
            gtab1[2, j1] = f(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
            gtab1[3, j1] = fp(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)

        for i1 in range(m1):
            for k1 in range(m1):

                f1Auf1Bu = integral_ff(i1, k1, x1u1, x1ur1, x2u1, x2ur1, x1u1, x1ur1, x2u1, x2ur1)
                f1Avf1Bv = integral_ff(i1, k1, x1v1, x1vr1, x2v1, x2vr1, x1v1, x1vr1, x2v1, x2vr1)
                f1Awf1Bw = integral_ff(i1, k1, x1w1, x1wr1, x2w1, x2wr1, x1w1, x1wr1, x2w1, x2wr1)

                for j1 in range(n1):
                    g1Au = gtab1[0, j1]
                    g1Av = gtab1[1, j1]
                    g1Aw = gtab1[2, j1]
                    g1Aw_eta = gtab1[3, j1]

                    for l1 in range(n1):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g1Bu = gtab1[0, l1]
                        g1Bv = gtab1[1, l1]
                        g1Bw = gtab1[2, l1]
                        g1Bw_eta = gtab1[3, l1]

                        c += 1
                        kCSSycte11r[c] = row+0
                        kCSSycte11c[c] = col+0
                        kCSSycte11v[c] += 0.5*a1*f1Auf1Bu*g1Au*g1Bu*kt
                        c += 1
                        kCSSycte11r[c] = row+1
                        kCSSycte11c[c] = col+1
                        kCSSycte11v[c] += 0.5*a1*f1Avf1Bv*g1Av*g1Bv*kt
                        c += 1
                        kCSSycte11r[c] = row+2
                        kCSSycte11c[c] = col+2
                        kCSSycte11v[c] += 0.5*a1*kt*(f1Awf1Bw*g1Aw*g1Bw + 4*f1Awf1Bw*g1Aw_eta*g1Bw_eta*kr/((b1*b1)*kt))

    kCSSycte11 = coo_matrix((kCSSycte11v, (kCSSycte11r, kCSSycte11c)), shape=(size, size))

    return kCSSycte11


def fkCSSycte12(double kt, double kr, object p1, object p2,
          double ycte1, double ycte2,
          int size, int row0, int col0):
    r"""
    Coupling block of the skin-skin connection ``'SSycte'``

    The penalty terms of the connection that couple ``p1`` (rows) and
    ``p2`` (columns), for the models with 3 DOFs per term: the products of the
    approximation functions are integrated along `x` over the length of the
    edge of ``p1`` and evaluated at ``ycte1`` in ``p1`` and ``ycte2`` in
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
    ycte1 : float
        Coordinate `y_1` of the connection in ``p1``, length, between ``0``
        and ``p1.b``.
    ycte2 : float
        Coordinate `y_2` of the connection in ``p2``, length, between ``0``
        and ``p2.b``.
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
    kCSSycte12 : scipy.sparse.coo_matrix
        Coupling block, rows of ``p1`` and columns of ``p2``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCSSycte`, see its Notes:

    .. math::

        \boldsymbol{K}_{12} = -\frac{a_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q1}^\top \boldsymbol{N}_{q2}
        + k_r \frac{4}{b_1 b_2} \boldsymbol{N}_{w1,\eta}^\top \boldsymbol{N}_{w2,\eta}
        \, d\xi

    with the functions of ``p1`` at `\eta_1 = 2 y_{cte1}/b_1 - 1` and
    those of ``p2`` at `\eta_2 = 2 y_{cte2}/b_2 - 1`.

    """
    cdef double [:, ::1] gtab1, gtab2
    cdef int i1, k2, j1, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, b1, b2
    cdef double x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCSSycte12r, kCSSycte12c
    cdef double [:] kCSSycte12v

    cdef double etacte1, etacte2
    cdef double f1Auf2Bu, f1Avf2Bv, f1Awf2Bw
    cdef double g1Au, g2Bu, g1Av, g2Bv, g1Aw, g2Bw, g1Aw_eta, g2Bw_eta

    a1 = p1.a
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

    etacte1 = 2*ycte1/b1 - 1.
    etacte2 = 2*ycte2/b2 - 1.

    fdim = 3*m1*n1*m2*n2

    kCSSycte12r = np.zeros((fdim,), dtype=INT)
    kCSSycte12c = np.zeros((fdim,), dtype=INT)
    kCSSycte12v = np.zeros((fdim,), dtype=DOUBLE)

    gtab1 = np.zeros((4, n1), dtype=DOUBLE)
    gtab2 = np.zeros((4, n2), dtype=DOUBLE)

    with nogil:
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j1 in range(n1):
            gtab1[0, j1] = f(j1, etacte1, y1u1, y1ur1, y2u1, y2ur1)
            gtab1[1, j1] = f(j1, etacte1, y1v1, y1vr1, y2v1, y2vr1)
            gtab1[2, j1] = f(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
            gtab1[3, j1] = fp(j1, etacte1, y1w1, y1wr1, y2w1, y2wr1)
        for l2 in range(n2):
            gtab2[0, l2] = f(l2, etacte2, y1u2, y1ur2, y2u2, y2ur2)
            gtab2[1, l2] = f(l2, etacte2, y1v2, y1vr2, y2v2, y2vr2)
            gtab2[2, l2] = f(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
            gtab2[3, l2] = fp(l2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

        for i1 in range(m1):
            for k2 in range(m2):

                f1Auf2Bu = integral_ff(i1, k2, x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2)
                f1Avf2Bv = integral_ff(i1, k2, x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2)
                f1Awf2Bw = integral_ff(i1, k2, x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2)

                for j1 in range(n1):

                    g1Au = gtab1[0, j1]
                    g1Av = gtab1[1, j1]
                    g1Aw = gtab1[2, j1]
                    g1Aw_eta = gtab1[3, j1]

                    for l2 in range(n2):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry
                        #if row > col:
                            #continue

                        g2Bu = gtab2[0, l2]
                        g2Bv = gtab2[1, l2]
                        g2Bw = gtab2[2, l2]
                        g2Bw_eta = gtab2[3, l2]

                        c += 1
                        kCSSycte12r[c] = row+0
                        kCSSycte12c[c] = col+0
                        kCSSycte12v[c] += -0.5*a1*f1Auf2Bu*g1Au*g2Bu*kt
                        c += 1
                        kCSSycte12r[c] = row+1
                        kCSSycte12c[c] = col+1
                        kCSSycte12v[c] += -0.5*a1*f1Avf2Bv*g1Av*g2Bv*kt
                        c += 1
                        kCSSycte12r[c] = row+2
                        kCSSycte12c[c] = col+2
                        kCSSycte12v[c] += -0.5*a1*kt*(f1Awf2Bw*g1Aw*g2Bw + 4*f1Awf2Bw*g1Aw_eta*g2Bw_eta*kr/(b1*b2*kt))

    kCSSycte12 = coo_matrix((kCSSycte12v, (kCSSycte12r, kCSSycte12c)), shape=(size, size))

    return kCSSycte12


def fkCSSycte22(double kt, double kr, object p1, object p2,
          double ycte2,
          int size, int row0, int col0):
    r"""
    Block of ``p2`` of the skin-skin connection ``'SSycte'``

    The penalty terms of the connection that involve only ``p2``, for
    the models with 3 DOFs per term: the products of the approximation functions
    of ``p2`` are integrated along `x` over the length of the edge of ``p1``
    and evaluated at ``ycte2`` across it.

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
    ycte2 : float
        Coordinate `y_2` of the connection in ``p2``, length, between ``0``
        and ``p2.b``.
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
    kCSSycte22 : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p22``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCSSycte`, see its Notes:

    .. math::

        \boldsymbol{K}_{22} = \frac{a_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q2}^\top \boldsymbol{N}_{q2}
        + k_r \frac{4}{b_2^2} \boldsymbol{N}_{w2,\eta}^\top \boldsymbol{N}_{w2,\eta}
        \, d\xi

    with the functions of ``p2`` at `\eta_2 = 2 y_{cte2}/b_2 - 1`.

    """
    cdef double [:, ::1] gtab2
    cdef int i2, k2, j2, l2, c, row, col
    cdef int m2, n2
    cdef double a1, b2
    cdef double x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCSSycte22r, kCSSycte22c
    cdef double [:] kCSSycte22v

    cdef double etacte2
    cdef double f2Auf2Bu, f2Avf2Bv, f2Awf2Bw
    cdef double g2Au, g2Bu, g2Av, g2Bv, g2Aw, g2Bw, g2Aw_eta, g2Bw_eta

    a1 = p1.a
    b2 = p2.b
    m2 = p2.m
    n2 = p2.n
    x1u2 = p2.x1u ; x1ur2 = p2.x1ur ; x2u2 = p2.x2u ; x2ur2 = p2.x2ur
    x1v2 = p2.x1v ; x1vr2 = p2.x1vr ; x2v2 = p2.x2v ; x2vr2 = p2.x2vr
    x1w2 = p2.x1w ; x1wr2 = p2.x1wr ; x2w2 = p2.x2w ; x2wr2 = p2.x2wr
    y1u2 = p2.y1u ; y1ur2 = p2.y1ur ; y2u2 = p2.y2u ; y2ur2 = p2.y2ur
    y1v2 = p2.y1v ; y1vr2 = p2.y1vr ; y2v2 = p2.y2v ; y2vr2 = p2.y2vr
    y1w2 = p2.y1w ; y1wr2 = p2.y1wr ; y2w2 = p2.y2w ; y2wr2 = p2.y2wr

    etacte2 = 2*ycte2/b2 - 1.

    fdim = 3*m2*n2*m2*n2

    kCSSycte22r = np.zeros((fdim,), dtype=INT)
    kCSSycte22c = np.zeros((fdim,), dtype=INT)
    kCSSycte22v = np.zeros((fdim,), dtype=DOUBLE)

    gtab2 = np.zeros((4, n2), dtype=DOUBLE)

    with nogil:
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j2 in range(n2):
            gtab2[0, j2] = f(j2, etacte2, y1u2, y1ur2, y2u2, y2ur2)
            gtab2[1, j2] = f(j2, etacte2, y1v2, y1vr2, y2v2, y2vr2)
            gtab2[2, j2] = f(j2, etacte2, y1w2, y1wr2, y2w2, y2wr2)
            gtab2[3, j2] = fp(j2, etacte2, y1w2, y1wr2, y2w2, y2wr2)

        for i2 in range(m2):
            for k2 in range(m2):

                f2Auf2Bu = integral_ff(i2, k2, x1u2, x1ur2, x2u2, x2ur2, x1u2, x1ur2, x2u2, x2ur2)
                f2Avf2Bv = integral_ff(i2, k2, x1v2, x1vr2, x2v2, x2vr2, x1v2, x1vr2, x2v2, x2vr2)
                f2Awf2Bw = integral_ff(i2, k2, x1w2, x1wr2, x2w2, x2wr2, x1w2, x1wr2, x2w2, x2wr2)

                for j2 in range(n2):
                    g2Au = gtab2[0, j2]
                    g2Av = gtab2[1, j2]
                    g2Aw = gtab2[2, j2]
                    g2Aw_eta = gtab2[3, j2]

                    for l2 in range(n2):
                        row = row0 + DOF*(j2*m2 + i2)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry
                        if row > col:
                            continue

                        g2Bu = gtab2[0, l2]
                        g2Bv = gtab2[1, l2]
                        g2Bw = gtab2[2, l2]
                        g2Bw_eta = gtab2[3, l2]

                        c += 1
                        kCSSycte22r[c] = row+0
                        kCSSycte22c[c] = col+0
                        kCSSycte22v[c] += 0.5*a1*f2Auf2Bu*g2Au*g2Bu*kt
                        c += 1
                        kCSSycte22r[c] = row+1
                        kCSSycte22c[c] = col+1
                        kCSSycte22v[c] += 0.5*a1*f2Avf2Bv*g2Av*g2Bv*kt
                        c += 1
                        kCSSycte22r[c] = row+2
                        kCSSycte22c[c] = col+2
                        kCSSycte22v[c] += 0.5*a1*kt*(f2Awf2Bw*g2Aw*g2Bw + 4*f2Awf2Bw*g2Aw_eta*g2Bw_eta*kr/((b2*b2)*kt))

    kCSSycte22 = coo_matrix((kCSSycte22v, (kCSSycte22r, kCSSycte22c)), shape=(size, size))

    return kCSSycte22
