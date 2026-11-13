#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Kernels of the base-flange connection ``'BFycte'``

Penalty stiffness matrices of the connection along `x` between the edge
`y_1 =` ``ycte1`` of the stiffener's base, or skin, ``p1``, and the
edge `y_2 =` ``ycte2`` of the flange ``p2``, normal to ``p1``, the
local axes of the flange being those of ``p1`` rotated by 90 degrees about
the common axis `x`. For the models based on the classical laminated
plate theory (CLPT), with 3 DOFs `u, v, w` per term, :func:`.fkCBFycte11`
gives the block of the base, :func:`.fkCBFycte12` the coupling block,
rows of the base and columns of the flange, and :func:`.fkCBFycte22` the
block of the flange. For the models based on shear deformation theories,
e.g. ``'plate_fsdt_donnell'`` or ``'cylshell_tsdt_sanders'``, with 5 DOFs
`u, v, w, \phi_x, \phi_y` per term, the same blocks are given by
:func:`.fkCBFycte11_sdt`, :func:`.fkCBFycte12_sdt` and
:func:`.fkCBFycte22_sdt`. The blocks of a single panel are returned as
their upper triangle. With the default ``conn_method='null-space'`` of
:class:`.MultiDomain` the same quantities are imposed exactly, see
:mod:`panels.multidomain.connections.nullspace`. These kernels connect the
reference surface of the base; with the key ``'face'`` the flange is tied
to a face of the base instead, see
:mod:`panels.multidomain.connections.kCBFycte_face`.

Notes
-----
With the local axes of the flange rotated by 90 degrees about the common
axis `x`, the penalty energy is:

.. math::

    U = \frac{1}{2} \int_0^{a_1} k_t \left[ (u_1 - u_2)^2 + (v_1 - w_2)^2 +
        (w_1 + v_2)^2 \right] + k_r \left( \omega_1 - \omega_2 \right)^2 dx

where `\omega_i` is the rotation of the normal of each panel about `x`,
positive with the right-hand rule in its own local axes. With the kinematics
of Donnell and for flat plates, `\omega_i = w_{i,y}`. With the kinematics of
Sanders-Koiter (``'cylshell_clpt_sanders'``), the rotation of the normal also
includes the rigid-body rotation of the cross-section, `\omega_i = w_{i,y} -
v_i/r_i`, consistent with `\phi_y = -w_{,y} + v/r` of the model. The
arguments ``rinv1`` and ``rinv2`` are `1/r_1` and `1/r_2` for panels with
the kinematics of Sanders-Koiter and zero otherwise, see
:meth:`.MultiDomain.get_kC_conn`.

Without the term `v_1/r_1`, a rigid-body rotation of a stiffened cylinder
about its axis, `v_1 = r_1 \theta`, is penalized by the
rotation penalty, since the Sanders-Koiter skin rotates by `\theta` while the
flange must follow with `w_{2,y} = \theta`. With the Donnell kinematics the
term `v/r` is neglected in the rotations and in the curvatures of the shell
alike, and the kernels are unchanged.

For the models based on shear deformation theories, the rotation of the
normal about `x` is `\omega_i = -\theta_{y,i}`, the rotation at the
reference surface, `z = 0`, `\theta_y = \Phi_y - (4 d^2/h^2)(\phi_y +
w_{,y})` with the offset `d` and the thickness `h` of each panel for the
TSDT, `\theta_y = \Phi_y` otherwise, where `\Phi_y = \phi_y + v/r` for the
kinematics of Sanders-Koiter and `\Phi_y = \phi_y` otherwise, see
:func:`.kCsdt._rotation_y`. The rotation penalty is `k_r (\theta_{y,1} -
\theta_{y,2})^2`, with ``rinv1`` and ``rinv2`` as for the CLPT. For the
TSDT, the derivative `w_{,y}` is not penalized on its own: at a T-joint
only the rotation of the normals is common to the two panels. The rotation
`\phi_x` of each panel is a drilling rotation of the other and is not
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
symmetrized afterwards. Along the connection `\int_0^{a_1} dx = (a_1/2)
\int_{-1}^{1} d\xi`, computed with the closed-form integrals of the products
of Bardell's functions, the two edges sharing the natural coordinate `\xi`;
across it the functions are evaluated at the natural coordinate of the
connection in each panel, and `\partial/\partial y = (2/b)
\partial/\partial \eta`. For the models with 5 DOFs per term the
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


def fkCBFycte11(double kt, double kr, object p1, double ycte1,
          int size, int row0, int col0, double rinv1=0.):
    r"""
    Block of the base ``p1`` of the base-flange connection ``'BFycte'``

    The penalty terms of the connection that involve only the base, or skin, ``p1``, for the
    models with 3 DOFs per term: the products of the approximation functions of
    ``p1`` are integrated along `x` over the length of the edge of ``p1``
    and evaluated at ``ycte1`` across it.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Base, or skin.
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
    rinv1 : float, optional
        `1/r_1` when ``p1`` has the kinematics of Sanders-Koiter, zero
        otherwise. Default ``0.``.

    Returns
    -------
    kCBFycte11 : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p1``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFycte`, see its Notes:

    .. math::

        \boldsymbol{K}_{11} = \frac{a_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q1}^\top \boldsymbol{N}_{q1}
        + k_r \boldsymbol{N}_{\omega 1}^\top \boldsymbol{N}_{\omega 1} \, d\xi

    with the functions of ``p1`` at `\eta_1 = 2 y_{cte1}/b_1 - 1`,
    and
    `\boldsymbol{N}_{\omega 1} = (2/b_1) \boldsymbol{N}_{w1,\eta} - r_1^{-1}
    \boldsymbol{N}_{v1}`, the rotation `\omega_1 = w_{1,y} - v_1/r_1` about `x`, with
    `r_1^{-1}` given by ``rinv1``.

    """
    cdef double [:, ::1] gtab1
    cdef int i1, j1, k1, l1, c, row, col
    cdef int m1, n1
    cdef double a1, b1
    cdef double x1u1, x1ur1, x2u1, x2ur1
    cdef double x1v1, x1vr1, x2v1, x2vr1
    cdef double x1w1, x1wr1, x2w1, x2wr1
    cdef double y1u1, y1ur1, y2u1, y2ur1
    cdef double y1v1, y1vr1, y2v1, y2vr1
    cdef double y1w1, y1wr1, y2w1, y2wr1

    cdef long [:] kCBFycte11r, kCBFycte11c
    cdef double [:] kCBFycte11v

    cdef double etacte1
    cdef double f1Auf1Bu, f1Avf1Bv, f1Awf1Bw, f1Avf1Bw, f1Awf1Bv
    cdef double g1Au, g1Av, g1Aw, g1Aw_eta, g1Bu, g1Bv, g1Bw, g1Bw_eta

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

    fdim = 5*m1*n1*m1*n1

    kCBFycte11r = np.zeros((fdim,), dtype=INT)
    kCBFycte11c = np.zeros((fdim,), dtype=INT)
    kCBFycte11v = np.zeros((fdim,), dtype=DOUBLE)

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
                f1Avf1Bw = integral_ff(i1, k1, x1v1, x1vr1, x2v1, x2vr1, x1w1, x1wr1, x2w1, x2wr1)
                f1Awf1Bv = integral_ff(i1, k1, x1w1, x1wr1, x2w1, x2wr1, x1v1, x1vr1, x2v1, x2vr1)

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
                        kCBFycte11r[c] = row+0
                        kCBFycte11c[c] = col+0
                        kCBFycte11v[c] += 0.5*a1*f1Auf1Bu*g1Au*g1Bu*kt
                        c += 1
                        kCBFycte11r[c] = row+1
                        kCBFycte11c[c] = col+1
                        kCBFycte11v[c] += 0.5*a1*f1Avf1Bv*g1Av*g1Bv*kt
                        c += 1
                        kCBFycte11r[c] = row+2
                        kCBFycte11c[c] = col+2
                        kCBFycte11v[c] += 0.5*a1*kt*(f1Awf1Bw*g1Aw*g1Bw + 4*f1Awf1Bw*g1Aw_eta*g1Bw_eta*kr/((b1*b1)*kt))
                        if rinv1 != 0:
                            #NOTE Sanders-Koiter rotation w1,y - v1/r1
                            c += 1
                            kCBFycte11r[c] = row+1
                            kCBFycte11c[c] = col+1
                            kCBFycte11v[c] += 0.5*a1*kr*rinv1*rinv1*f1Avf1Bv*g1Av*g1Bv
                            c += 1
                            kCBFycte11r[c] = row+1
                            kCBFycte11c[c] = col+2
                            kCBFycte11v[c] += -a1*kr*rinv1*f1Avf1Bw*g1Av*g1Bw_eta/b1
                            c += 1
                            kCBFycte11r[c] = row+2
                            kCBFycte11c[c] = col+1
                            kCBFycte11v[c] += -a1*kr*rinv1*f1Awf1Bv*g1Aw_eta*g1Bv/b1

    kCBFycte11 = coo_matrix((kCBFycte11v, (kCBFycte11r, kCBFycte11c)), shape=(size, size))

    return kCBFycte11


def fkCBFycte12(double kt, double kr, object p1, object p2,
          double ycte1, double ycte2,
          int size, int row0, int col0, double rinv1=0., double rinv2=0.):
    r"""
    Coupling block of the base-flange connection ``'BFycte'``

    The penalty terms of the connection that couple the base ``p1``
    (rows) and the flange ``p2`` (columns), for the models with 3 DOFs per term:
    the products of the approximation functions are integrated along `x`
    over the length of the edge of ``p1`` and evaluated at ``ycte1`` in
    ``p1`` and ``ycte2`` in ``p2`` across it. The whole block is
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
    rinv1 : float, optional
        `1/r_1` when ``p1`` has the kinematics of Sanders-Koiter, zero
        otherwise. Default ``0.``.
    rinv2 : float, optional
        `1/r_2` when ``p2`` has the kinematics of Sanders-Koiter, zero
        otherwise. Default ``0.``.

    Returns
    -------
    kCBFycte12 : scipy.sparse.coo_matrix
        Coupling block, rows of ``p1`` and columns of ``p2``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFycte`, see its Notes:

    .. math::

        \boldsymbol{K}_{12} = -\frac{a_1}{2} \int_{-1}^{1} k_t \left(
        \boldsymbol{N}_{u1}^\top \boldsymbol{N}_{u2} + \boldsymbol{N}_{v1}^\top \boldsymbol{N}_{w2}
        - \boldsymbol{N}_{w1}^\top \boldsymbol{N}_{v2} \right)
        + k_r \boldsymbol{N}_{\omega 1}^\top \boldsymbol{N}_{\omega 2} \, d\xi

    with the functions of ``p1`` at `\eta_1 = 2 y_{cte1}/b_1 - 1` and
    those of ``p2`` at `\eta_2 = 2 y_{cte2}/b_2 - 1`, and
    `\boldsymbol{N}_{\omega p} = (2/b_p) \boldsymbol{N}_{wp,\eta} - r_p^{-1}
    \boldsymbol{N}_{vp}`, the rotation `\omega_p = w_{p,y} - v_p/r_p` of each panel about
    `x`, with `r_p^{-1}` given by ``rinv1`` and ``rinv2``.

    """
    cdef double [:, ::1] gtab1, gtab2
    cdef int i1, j1, k2, l2, c, row, col
    cdef int m1, n1, m2, n2
    cdef double a1, b1, b2
    cdef double x1u1, x1ur1, x2u1, x2ur1, x1u2, x1ur2, x2u2, x2ur2
    cdef double x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2
    cdef double x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2
    cdef double y1u1, y1ur1, y2u1, y2ur1, y1u2, y1ur2, y2u2, y2ur2
    cdef double y1v1, y1vr1, y2v1, y2vr1, y1v2, y1vr2, y2v2, y2vr2
    cdef double y1w1, y1wr1, y2w1, y2wr1, y1w2, y1wr2, y2w2, y2wr2

    cdef long [:] kCBFycte12r, kCBFycte12c
    cdef double [:] kCBFycte12v

    cdef double etacte1, etacte2
    cdef double f1Auf2Bu, f1Avf2Bw, f1Awf2Bv, f1Awf2Bw, f1Avf2Bv
    cdef double g1Au, g1Av, g1Aw, g1Aw_eta, g2Bu, g2Bv, g2Bw, g2Bw_eta

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

    fdim = 7*m1*n1*m2*n2

    kCBFycte12r = np.zeros((fdim,), dtype=INT)
    kCBFycte12c = np.zeros((fdim,), dtype=INT)
    kCBFycte12v = np.zeros((fdim,), dtype=DOUBLE)

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
                f1Avf2Bw = integral_ff(i1, k2, x1v1, x1vr1, x2v1, x2vr1, x1w2, x1wr2, x2w2, x2wr2)
                f1Awf2Bv = integral_ff(i1, k2, x1w1, x1wr1, x2w1, x2wr1, x1v2, x1vr2, x2v2, x2vr2)
                f1Awf2Bw = integral_ff(i1, k2, x1w1, x1wr1, x2w1, x2wr1, x1w2, x1wr2, x2w2, x2wr2)
                f1Avf2Bv = integral_ff(i1, k2, x1v1, x1vr1, x2v1, x2vr1, x1v2, x1vr2, x2v2, x2vr2)

                for j1 in range(n1):
                    g1Au = gtab1[0, j1]
                    g1Av = gtab1[1, j1]
                    g1Aw = gtab1[2, j1]
                    g1Aw_eta = gtab1[3, j1]

                    for l2 in range(n2):
                        row = row0 + DOF*(j1*m1 + i1)
                        col = col0 + DOF*(l2*m2 + k2)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        g2Bu = gtab2[0, l2]
                        g2Bv = gtab2[1, l2]
                        g2Bw = gtab2[2, l2]
                        g2Bw_eta = gtab2[3, l2]

                        c += 1
                        kCBFycte12r[c] = row+0
                        kCBFycte12c[c] = col+0
                        kCBFycte12v[c] += -0.5*a1*f1Auf2Bu*g1Au*g2Bu*kt
                        c += 1
                        kCBFycte12r[c] = row+1
                        kCBFycte12c[c] = col+2
                        kCBFycte12v[c] += -0.5*a1*f1Avf2Bw*g1Av*g2Bw*kt
                        c += 1
                        kCBFycte12r[c] = row+2
                        kCBFycte12c[c] = col+1
                        kCBFycte12v[c] += 0.5*a1*f1Awf2Bv*g1Aw*g2Bv*kt
                        c += 1
                        kCBFycte12r[c] = row+2
                        kCBFycte12c[c] = col+2
                        kCBFycte12v[c] += -2*a1*f1Awf2Bw*g1Aw_eta*g2Bw_eta*kr/(b1*b2)
                        if rinv1 != 0 or rinv2 != 0:
                            #NOTE Sanders-Koiter rotations w1,y - v1/r1 and
                            #     w2,y - v2/r2
                            c += 1
                            kCBFycte12r[c] = row+1
                            kCBFycte12c[c] = col+1
                            kCBFycte12v[c] += -0.5*a1*kr*rinv1*rinv2*f1Avf2Bv*g1Av*g2Bv
                            c += 1
                            kCBFycte12r[c] = row+1
                            kCBFycte12c[c] = col+2
                            kCBFycte12v[c] += a1*kr*rinv1*f1Avf2Bw*g1Av*g2Bw_eta/b2
                            c += 1
                            kCBFycte12r[c] = row+2
                            kCBFycte12c[c] = col+1
                            kCBFycte12v[c] += a1*kr*rinv2*f1Awf2Bv*g1Aw_eta*g2Bv/b1

    kCBFycte12 = coo_matrix((kCBFycte12v, (kCBFycte12r, kCBFycte12c)), shape=(size, size))

    return kCBFycte12


def fkCBFycte22(double kt, double kr, object p1, object p2,
          double ycte2,
          int size, int row0, int col0, double rinv2=0.):
    r"""
    Block of the flange ``p2`` of the base-flange connection ``'BFycte'``

    The penalty terms of the connection that involve only the flange ``p2``, for the
    models with 3 DOFs per term: the products of the approximation functions of
    ``p2`` are integrated along `x` over the length of the edge of ``p1``
    and evaluated at ``ycte2`` across it.

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
    rinv2 : float, optional
        `1/r_2` when ``p2`` has the kinematics of Sanders-Koiter, zero
        otherwise. Default ``0.``.

    Returns
    -------
    kCBFycte22 : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p2``, shape ``(size, size)``.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFycte`, see its Notes:

    .. math::

        \boldsymbol{K}_{22} = \frac{a_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q2}^\top \boldsymbol{N}_{q2}
        + k_r \boldsymbol{N}_{\omega 2}^\top \boldsymbol{N}_{\omega 2} \, d\xi

    with the functions of ``p2`` at `\eta_2 = 2 y_{cte2}/b_2 - 1`,
    and
    `\boldsymbol{N}_{\omega 2} = (2/b_2) \boldsymbol{N}_{w2,\eta} - r_2^{-1}
    \boldsymbol{N}_{v2}`, the rotation `\omega_2 = w_{2,y} - v_2/r_2` about `x`, with
    `r_2^{-1}` given by ``rinv2``.

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

    cdef long [:] kCBFycte22r, kCBFycte22c
    cdef double [:] kCBFycte22v

    cdef double etacte2
    cdef double f2Auf2Bu, f2Avf2Bv, f2Awf2Bw, f2Avf2Bw, f2Awf2Bv
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

    fdim = 5*m2*n2*m2*n2

    kCBFycte22r = np.zeros((fdim,), dtype=INT)
    kCBFycte22c = np.zeros((fdim,), dtype=INT)
    kCBFycte22v = np.zeros((fdim,), dtype=DOUBLE)

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
                f2Avf2Bw = integral_ff(i2, k2, x1v2, x1vr2, x2v2, x2vr2, x1w2, x1wr2, x2w2, x2wr2)
                f2Awf2Bv = integral_ff(i2, k2, x1w2, x1wr2, x2w2, x2wr2, x1v2, x1vr2, x2v2, x2vr2)

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
                        kCBFycte22r[c] = row+0
                        kCBFycte22c[c] = col+0
                        kCBFycte22v[c] += 0.5*a1*f2Auf2Bu*g2Au*g2Bu*kt
                        c += 1
                        kCBFycte22r[c] = row+1
                        kCBFycte22c[c] = col+1
                        kCBFycte22v[c] += 0.5*a1*f2Avf2Bv*g2Av*g2Bv*kt
                        c += 1
                        kCBFycte22r[c] = row+2
                        kCBFycte22c[c] = col+2
                        kCBFycte22v[c] += 0.5*a1*kt*(f2Awf2Bw*g2Aw*g2Bw + 4*f2Awf2Bw*g2Aw_eta*g2Bw_eta*kr/((b2*b2)*kt))
                        if rinv2 != 0:
                            #NOTE Sanders-Koiter rotation w2,y - v2/r2
                            c += 1
                            kCBFycte22r[c] = row+1
                            kCBFycte22c[c] = col+1
                            kCBFycte22v[c] += 0.5*a1*kr*rinv2*rinv2*f2Avf2Bv*g2Av*g2Bv
                            c += 1
                            kCBFycte22r[c] = row+1
                            kCBFycte22c[c] = col+2
                            kCBFycte22v[c] += -a1*kr*rinv2*f2Avf2Bw*g2Av*g2Bw_eta/b2
                            c += 1
                            kCBFycte22r[c] = row+2
                            kCBFycte22c[c] = col+1
                            kCBFycte22v[c] += -a1*kr*rinv2*f2Awf2Bv*g2Aw_eta*g2Bv/b2

    kCBFycte22 = coo_matrix((kCBFycte22v, (kCBFycte22r, kCBFycte22c)), shape=(size, size))

    return kCBFycte22


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


def _rotation_ref(object p, double rinv):
    r"""
    Rotation of the normal about `x` at the reference surface

    Parameters
    ----------
    p : :class:`.Shell`
        Panel of a model based on the FSDT or TSDT.
    rinv : float
        `1/r` when ``p`` has the kinematics of Sanders-Koiter, zero otherwise.

    Returns
    -------
    terms : list
        The rotation as a list of ``(dof, der, a)``, see
        :func:`._product_terms`: `\phi_y`, `v` when ``rinv`` is not zero and,
        for a TSDT panel with an offset, the derivative of `w` across the
        connection.

    Notes
    -----
    `\theta_y = (1 - \alpha) \phi_y + v/r - \alpha w_{,y}`, with `\alpha` of
    :func:`._alpha`, see :func:`.kCsdt._rotation_y`; `w_{,y}` is the derivative
    across the connection and `v/r` the term of the Sanders-Koiter kinematics.

    """
    a = _alpha(p)
    terms = [(PHIY, 0, 1. - a)]
    if rinv != 0:
        terms.append((V, 0, rinv))
    if a != 0:
        terms.append((W, 1, -a))
    return terms


def _rotation_terms(double coeff, object pA, object pB, double rinvA,
        double rinvB):
    r"""
    Penalty terms of the product of the rotations of two panels

    Parameters
    ----------
    coeff : float
        Factor of the product, ``kr`` or ``-kr``.
    pA, pB : :class:`.Shell`
        Panels of the rows and of the columns.
    rinvA, rinvB : float
        `1/r` of ``pA`` and ``pB`` with the kinematics of Sanders-Koiter, zero
        otherwise.

    Returns
    -------
    terms : list
        The terms of `\mathrm{coeff} \, \theta_{y,A} \theta_{y,B}`, see
        :func:`._rotation_ref` and :func:`._product_terms`.

    """
    return _product_terms(coeff, _rotation_ref(pA, rinvA), _rotation_ref(pB, rinvB))


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


def fkCBFycte11_sdt(double kt, double kr, object p1, double ycte1,
          int size, int row0, int col0, double rinv1=0.):
    r"""
    Block of the base of ``'BFycte'``, models with 5 DOFs per term

    The penalty terms of the connection that involve only the base ``p1``,
    for the models based on the FSDT and TSDT: the differences of `u, v, w` with ``kt`` and the rotation of
    the normal about `x` at the reference surface with ``kr``, see
    :func:`._rotation_ref`. The products of the approximation functions are
    integrated along `x` over the length of the edge of ``p1`` and evaluated at
    ``ycte1`` across it.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Base, or skin.
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
    rinv1 : float, optional
        `1/r_1` when ``p1`` has the kinematics of Sanders-Koiter, zero
        otherwise. Default ``0.``.

    Returns
    -------
    kCBFycte11_sdt : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p1``, shape ``(size, size)``.

    Raises
    ------
    ValueError
        When a panel is not a model based on the FSDT or TSDT.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFycte`, see its Notes:

    .. math::

        \boldsymbol{K}_{11} = \frac{a_1}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q1}^\top \boldsymbol{N}_{q1}
        + k_r \boldsymbol{N}_{\theta 1}^\top \boldsymbol{N}_{\theta 1} \, d\xi

    with the functions of ``p1`` at `\eta_1 = 2 y_{cte1}/b_1 - 1`,
    and
    `\boldsymbol{N}_{\theta 1} = (1 - \alpha_1) \boldsymbol{N}_{\phi_y 1} + r_1^{-1}
    \boldsymbol{N}_{v1} - \alpha_1 (2/b_1) \boldsymbol{N}_{w1,\eta}`, the rotation `\theta_y` of the panel
    at the reference surface, with `\alpha` of :func:`._alpha` and `r^{-1}`
    given by ``rinv1`` and ``rinv2``.
    The Ritz constants of the term `(i, j)` are at ``5*(j*m + i)``.

    """
    _check_sdt(p1)
    terms = [(U, U, kt), (V, V, kt), (W, W, kt)] + _rotation_terms(kr, p1, p1, rinv1, rinv1)
    cte = 2*ycte1/p1.b - 1.
    return _block_sdt(terms, p1, p1, cte, cte, 'x', 1, size, row0, col0)


def fkCBFycte12_sdt(double kt, double kr, object p1, object p2,
          double ycte1, double ycte2,
          int size, int row0, int col0, double rinv1=0., double rinv2=0.):
    r"""
    Coupling block of ``'BFycte'``, models with 5 DOFs per term

    The penalty terms of the connection that couple the base ``p1`` (rows) and the flange ``p2`` (columns),
    for the models based on the FSDT and TSDT: the differences of `u, v, w` with ``kt`` and the rotation of
    the normal about `x` at the reference surface with ``kr``, see
    :func:`._rotation_ref`. The products of the approximation functions are
    integrated along `x` over the length of the edge of ``p1`` and evaluated at
    ``ycte1`` in ``p1`` and ``ycte2`` in ``p2`` across it. The whole block is returned.

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
    rinv1 : float, optional
        `1/r_1` when ``p1`` has the kinematics of Sanders-Koiter, zero
        otherwise. Default ``0.``.
    rinv2 : float, optional
        `1/r_2` when ``p2`` has the kinematics of Sanders-Koiter, zero
        otherwise. Default ``0.``.

    Returns
    -------
    kCBFycte12_sdt : scipy.sparse.coo_matrix
        Coupling block, rows of ``p1`` and columns of ``p2``, shape ``(size, size)``.

    Raises
    ------
    ValueError
        When a panel is not a model based on the FSDT or TSDT.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFycte`, see its Notes:

    .. math::

        \boldsymbol{K}_{12} = -\frac{a_1}{2} \int_{-1}^{1} k_t \left(
        \boldsymbol{N}_{u1}^\top \boldsymbol{N}_{u2} + \boldsymbol{N}_{v1}^\top \boldsymbol{N}_{w2}
        - \boldsymbol{N}_{w1}^\top \boldsymbol{N}_{v2} \right)
        + k_r \boldsymbol{N}_{\theta 1}^\top \boldsymbol{N}_{\theta 2} \, d\xi

    with the functions of ``p1`` at `\eta_1 = 2 y_{cte1}/b_1 - 1` and
    those of ``p2`` at `\eta_2 = 2 y_{cte2}/b_2 - 1`, and
    `\boldsymbol{N}_{\theta p} = (1 - \alpha_p) \boldsymbol{N}_{\phi_y p} + r_p^{-1}
    \boldsymbol{N}_{vp} - \alpha_p (2/b_p) \boldsymbol{N}_{wp,\eta}`, the rotation `\theta_y` of each panel
    at the reference surface, with `\alpha` of :func:`._alpha` and `r^{-1}`
    given by ``rinv1`` and ``rinv2``.
    The Ritz constants of the term `(i, j)` are at ``5*(j*m + i)``.

    """
    _check_sdt(p1, p2)
    terms = ([(U, U, -kt), (V, W, -kt), (W, V, kt)]
             + _rotation_terms(-kr, p1, p2, rinv1, rinv2))
    cte1 = 2*ycte1/p1.b - 1.
    cte2 = 2*ycte2/p2.b - 1.
    return _block_sdt(terms, p1, p2, cte1, cte2, 'x', 0, size, row0, col0)


def fkCBFycte22_sdt(double kt, double kr, object p1, object p2,
          double ycte2,
          int size, int row0, int col0, double rinv2=0.):
    r"""
    Block of the flange of ``'BFycte'``, models with 5 DOFs per term

    The penalty terms of the connection that involve only the flange ``p2``,
    for the models based on the FSDT and TSDT: the differences of `u, v, w` with ``kt`` and the rotation of
    the normal about `x` at the reference surface with ``kr``, see
    :func:`._rotation_ref`. The products of the approximation functions are
    integrated along `x` over the length of the edge of ``p2`` and evaluated at
    ``ycte2`` across it.

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
    rinv2 : float, optional
        `1/r_2` when ``p2`` has the kinematics of Sanders-Koiter, zero
        otherwise. Default ``0.``.

    Returns
    -------
    kCBFycte22_sdt : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p2``, shape ``(size, size)``.

    Raises
    ------
    ValueError
        When a panel is not a model based on the FSDT or TSDT.

    Notes
    -----
    The block of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFycte`, see its Notes:

    .. math::

        \boldsymbol{K}_{22} = \frac{a_2}{2} \int_{-1}^{1} k_t
        \sum_{q = u, v, w} \boldsymbol{N}_{q2}^\top \boldsymbol{N}_{q2}
        + k_r \boldsymbol{N}_{\theta 2}^\top \boldsymbol{N}_{\theta 2} \, d\xi

    with the functions of ``p2`` at `\eta_2 = 2 y_{cte2}/b_2 - 1`,
    and
    `\boldsymbol{N}_{\theta 2} = (1 - \alpha_2) \boldsymbol{N}_{\phi_y 2} + r_2^{-1}
    \boldsymbol{N}_{v2} - \alpha_2 (2/b_2) \boldsymbol{N}_{w2,\eta}`, the rotation `\theta_y` of the panel
    at the reference surface, with `\alpha` of :func:`._alpha` and `r^{-1}`
    given by ``rinv1`` and ``rinv2``.
    The Ritz constants of the term `(i, j)` are at ``5*(j*m + i)``.

    """
    _check_sdt(p1, p2)
    terms = [(U, U, kt), (V, V, kt), (W, W, kt)] + _rotation_terms(kr, p2, p2, rinv2, rinv2)
    cte = 2*ycte2/p2.b - 1.
    return _block_sdt(terms, p2, p2, cte, cte, 'x', 1, size, row0, col0)
