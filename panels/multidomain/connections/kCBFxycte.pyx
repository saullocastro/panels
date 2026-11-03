#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Kernels of the connection ``'BFxycte'``

Penalty stiffness matrices of the connection between the edge `x_1 =`
``xcte1`` of ``p1``, along `y_1`, and the edge `y_2 =` ``ycte2`` of ``p2``,
along `x_2`, two domains meeting at an angle along a common straight line,
e.g. a spar (``p1``), whose edge at the end of a bay is `x_1 = a_1`, joined
to a rib (``p2``), whose edges `y_2 = 0` and `y_2 = b_2` run along the
height of the wing-box. The two edges are the same line, `y_1 = x_2`: the
points `(x_{cte1}, 0)` of ``p1`` and `(0, y_{cte2})` of ``p2`` coincide and
the edges have the same length, `b_1 = a_2`. ``p1`` must be flat, for its
edge to be a straight line; ``p2`` may be a cylindrical shell, whose edges
`y_2 =` constant are straight, see :func:`.check_xycte`. The angle
`\theta` of the connection, the key ``'angle'`` in degrees, default ``90``,
is the angle from the normal of ``p1`` to the normal of ``p2`` about the
common axis, with the right-hand rule. For `\theta = 90` degrees the frame
of ``p2`` is that of ``p1`` with the axes renamed cyclically, the case of a
spar and a rib connected to the skins with ``'BFycte'`` and ``'BFxcte'``;
for `\theta = 0` or 180 degrees the two domains are coplanar, a skin-skin
connection between an edge `x =` constant and an edge `y =` constant.

The blocks 11 and 22 are those of :func:`.fkCBFxcte11` and
:func:`.fkCBFycte22` (:func:`.fkCBFxcte11_sdt` and :func:`.fkCBFycte22_sdt`
for the models with 5 DOFs per term), given by :func:`.fkCBFxycte11`,
:func:`.fkCBFxycte22`, :func:`.fkCBFxycte11_sdt` and
:func:`.fkCBFxycte22_sdt`; the coupling block :func:`.fkCBFxycte12`, rows
of ``p1`` and columns of ``p2``, serves the models with 3 and with 5 DOFs
per term and is computed from the quantities of :func:`.xycte_components`,
which are also the constraints of the null-space method, see
:mod:`panels.multidomain.connections.nullspace`. Both panels must have the
same number of DOFs per term.

Notes
-----
.. rubric:: Frames

The common axis is `\boldsymbol{s} = \boldsymbol{e}_{y1} =
\boldsymbol{e}_{x2}`. With the angle `\theta` from the normal
`\boldsymbol{e}_{z1}` of ``p1`` to the normal `\boldsymbol{e}_{z2}` of
``p2``, about `\boldsymbol{s}` with the right-hand rule:

.. math::

    \boldsymbol{e}_{x2} = \boldsymbol{e}_{y1}, \qquad
    \boldsymbol{e}_{y2} = -\cos\theta \, \boldsymbol{e}_{x1}
                          + \sin\theta \, \boldsymbol{e}_{z1}, \qquad
    \boldsymbol{e}_{z2} = \sin\theta \, \boldsymbol{e}_{x1}
                          + \cos\theta \, \boldsymbol{e}_{z1}

For `\theta = 90^\circ`, `(\boldsymbol{e}_{x2}, \boldsymbol{e}_{y2},
\boldsymbol{e}_{z2}) = (\boldsymbol{e}_{y1}, \boldsymbol{e}_{z1},
\boldsymbol{e}_{x1})`: a skin `(X, Y, Z)`, the spars `(X, -Z, Y)` and the
ribs `(-Z, Y, X)`.

.. rubric:: Penalty energy

The displacement of a point of the edge is the same vector in both frames,
`u_1 \boldsymbol{e}_{x1} + v_1 \boldsymbol{e}_{y1} + w_1 \boldsymbol{e}_{z1}
= u_2 \boldsymbol{e}_{x2} + v_2 \boldsymbol{e}_{y2} + w_2
\boldsymbol{e}_{z2}`, and the rotation of the normal about `\boldsymbol{s}`
is common to both domains, `\omega_1 = \omega_2`, with the right-hand rule
about `\boldsymbol{s}`:

.. math::

    U = \frac{1}{2} \int_0^{b_1} k_t \left[ \left(u_1 + \cos\theta \, v_2
        - \sin\theta \, w_2\right)^2 + \left(v_1 - u_2\right)^2
        + \left(w_1 - \sin\theta \, v_2 - \cos\theta \, w_2\right)^2
        \right] + k_r \left(\omega_1 - \omega_2\right)^2 dy_1

which for `\theta = 90^\circ` is `k_t [(u_1 - w_2)^2 + (v_1 - u_2)^2 +
(w_1 - v_2)^2]`. The rotation about `\boldsymbol{s}` is the rotation about
`y` of ``p1``, `\omega_1 = \theta_{x1}`, as in ``'BFxcte'``, and the
rotation about `x` of ``p2``, `\omega_2 = -\theta_{y2}`, as in
``'BFycte'``. For the models based on the classical laminated plate theory
(CLPT), with 3 DOFs `u, v, w` per term, `\omega_1 = -w_{1,x}` and
`\omega_2 = w_{2,y} - v_2/r_2`, the term `v_2/r_2` present only when ``p2``
has the Sanders-Koiter kinematics. For the models based on shear
deformation theories, with 5 DOFs `u, v, w, \phi_x, \phi_y` per term, the
rotations at the reference surface of :func:`.kCsdt._rotation_x` and
:func:`.kCsdt._rotation_y`, with the offset of each panel for the TSDT; the
derivatives of `w` normal to the edge and the drilling rotations are not
penalized, as for ``'BFxcte'`` and ``'BFycte'``.

The quantities only of ``p1`` are those of ``'BFxcte'`` and only of ``p2``
those of ``'BFycte'``, and since the displacements of ``p2`` enter through an
orthogonal matrix, `k_t (u_2^2 + v_2^2 + w_2^2)` whatever the angle, the
blocks 11 and 22 are those of these connections. The coupling block
integrates the products of the functions of ``p1`` along `y_1` and of
``p2`` along `x_2` over the common natural coordinate, `\eta_1 = \xi_2`,
with the closed-form integrals of the Legendre-based (Bardell) functions
[1]_, and evaluates the functions across the edge at `\xi_1 = 2
x_{cte1}/a_1 - 1` and `\eta_2 = 2 y_{cte2}/b_2 - 1`. The expressions were
derived symbolically with SymPy.

References
----------
.. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
   models to Address Linear Buckling and Vibration of Stiffened Composite
   Panels with Debonding Defect, Composite Structures, 160, 232-247, 2017.
   https://doi.org/10.1016/j.compstruct.2016.10.026

"""
from scipy.sparse import coo_matrix
import numpy as np

__all__ = ['fkCBFxycte11', 'fkCBFxycte12', 'fkCBFxycte22', 'fkCBFxycte11_sdt',
           'fkCBFxycte22_sdt']

from panels import INT, DOUBLE
from .kCBFxcte import fkCBFxcte11, fkCBFxcte11_sdt
from .kCBFycte import fkCBFycte22, fkCBFycte22_sdt
from . import kCsdt


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil


U, V, W, PHIX, PHIY = range(5)
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')


def _cos_sin(double angle):
    r"""
    Cosine and sine of an angle in degrees, without round-off at 90 degrees

    Parameters
    ----------
    angle : float
        Angle in degrees.

    Returns
    -------
    c, s : float
        Cosine and sine of ``angle``, each set to zero when its absolute value
        is below ``1e-14``, as for the multiples of 90 degrees.

    """
    t = np.radians(angle)
    c, s = np.cos(t), np.sin(t)
    c = 0. if abs(c) < 1e-14 else c
    s = 0. if abs(s) < 1e-14 else s
    return c, s


def check_xycte(p1, p2):
    r"""
    Check that the edges of a ``'BFxycte'`` connection can coincide

    The edge `x_1 =` constant of ``p1`` must be a straight line, the edges must
    have the same length and both panels the same number of DOFs per term.
    Nothing is returned.

    Parameters
    ----------
    p1 : :class:`.Shell`
        Panel with the edge `x_1 =` constant.
    p2 : :class:`.Shell`
        Panel with the edge `y_2 =` constant.

    Raises
    ------
    ValueError
        When the edge `x_1 =` constant of ``p1`` is not a straight line,
        ``p1`` being a cylindrical shell, a model whose name does not start
        with ``'plate'`` or, without a model, a radius different from zero;
        when the edges do not have the same length, ``p1.b`` different from
        ``p2.a``.
    NotImplementedError
        When ``p1`` and ``p2`` have a different number of DOFs per term.

    """
    if p1.model is not None:
        curved = not p1.model.startswith('plate')
    else:
        curved = p1.r is not None and p1.r != 0 and not np.isinf(p1.r)
    if curved:
        raise ValueError("The edge x = xcte1 of p1 of a 'BFxycte' connection "
                         "must be a straight line: p1 must be flat, got model "
                         "'{0}' and r={1}".format(p1.model, p1.r))
    if not np.isclose(p1.b, p2.a):
        raise ValueError("The edges of a 'BFxycte' connection must have the "
                         "same length, b of p1 and a of p2, got {0} and {1}"
                         .format(p1.b, p2.a))
    if kCsdt._dofs(p1) != kCsdt._dofs(p2):
        raise NotImplementedError(
            "Connection 'BFxycte' between models with a different number of "
            "DOFs per term, got models '{0}' and '{1}'".format(p1.model,
            p2.model))


def xycte_components(p1, p2, double angle=90.):
    r"""
    Quantities of the connection ``'BFxycte'``

    The quantities of each panel that must be equal along the common edge, as
    linear combinations of the fields and of their derivatives across the
    edge: three displacements, penalized with ``kt``, and the rotation about
    the common axis, penalized with ``kr``. They give the coupling block of
    :func:`.fkCBFxycte12` and the constraints of the null-space method, see
    :func:`.nullspace.constraint_matrix`.

    Parameters
    ----------
    p1, p2 : :class:`.Shell`
        The panels with the edges `x_1 =` constant and `y_2 =` constant.
    angle : float, optional
        Angle `\theta` of the connection in degrees, see
        :mod:`panels.multidomain.connections.kCBFxycte`. Default ``90.``.

    Returns
    -------
    components : list
        Each element ``(key, ((p1, terms1), (p2, terms2)))``, where ``key``
        is ``'kt'`` or ``'kr'``, the penalty constant of the quantity, and
        each ``terms`` a list of ``(field, der_x, der_y, coeff)``, the fields
        numbered ``0`` to ``4`` for `u, v, w, \phi_x, \phi_y`; the quantity
        of each panel is the sum of the terms and the two must be equal. The
        terms with a zero coefficient are dropped.

    Notes
    -----
    Each quantity of a panel is `\sum \alpha \, \partial_x^{d_x}
    \partial_y^{d_y} q`, with `\alpha` given by ``coeff``. With `c =
    \cos\theta` and `s = \sin\theta`, `u_1 = -c v_2 + s w_2`, `v_1 = u_2`, `w_1
    = s v_2 + c w_2` and `\theta_{x1} = -\theta_{y2}`, the rotations of the
    normal at the reference surface of :func:`.kCsdt._rotation_x` and
    :func:`.kCsdt._rotation_y`, the latter with the term `v_2/r_2` of the
    Sanders-Koiter kinematics. The derivatives are only across the edge,
    along `x` for ``p1`` and along `y` for ``p2``.

    """
    c, s = _cos_sin(angle)
    drop = lambda terms: [t for t in terms if t[3] != 0]
    rot2 = [(f, dx, dy, -a) for f, dx, dy, a in kCsdt._rotation_y(p2)]
    return [
        ('kt', ((p1, [(U, 0, 0, 1.)]),
                (p2, drop([(V, 0, 0, -c), (W, 0, 0, s)])))),
        ('kt', ((p1, [(V, 0, 0, 1.)]), (p2, [(U, 0, 0, 1.)]))),
        ('kt', ((p1, [(W, 0, 0, 1.)]),
                (p2, drop([(V, 0, 0, s), (W, 0, 0, c)])))),
        ('kr', ((p1, drop(kCsdt._rotation_x(p1))), (p2, drop(rot2)))),
        ]


def _flags(object p, fields, str direction):
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
        name = FIELDS[field]
        out[t, 0] = getattr(p, direction + '1' + name)
        out[t, 1] = getattr(p, direction + '1' + name + 'r')
        out[t, 2] = getattr(p, direction + '2' + name)
        out[t, 3] = getattr(p, direction + '2' + name + 'r')
    return out


cdef void _terms_xy(int nt, long [::1] fld1, long [::1] fld2,
        long [::1] der1, long [::1] der2, double [::1] coeff,
        double [:, ::1] al1, double [:, ::1] al2,
        double [:, ::1] ac1, double [:, ::1] ac2,
        int m1, int n1, int m2, int n2, double xicte1, double etacte2,
        double jac, int dof, int row0, int col0,
        double [::1] fs1, double [::1] gs2,
        long [::1] r, long [::1] c, double [::1] v) noexcept nogil:
    #NOTE ``al*`` are the flags of the functions along the edge, `y` of p1
    #     and `x` of p2, and ``ac*`` those across it, `x` of p1 and `y` of
    #     p2, evaluated once per term in ``fs1`` and ``gs2``
    cdef int t, i1, j1, i2, j2, pos
    cdef double I, f1, g2
    pos = 0
    for t in range(nt):
        for i1 in range(m1):
            if der1[t] == 1:
                fs1[i1] = fp(i1, xicte1, ac1[t, 0], ac1[t, 1], ac1[t, 2], ac1[t, 3])
            else:
                fs1[i1] = f(i1, xicte1, ac1[t, 0], ac1[t, 1], ac1[t, 2], ac1[t, 3])
        for j2 in range(n2):
            if der2[t] == 1:
                gs2[j2] = fp(j2, etacte2, ac2[t, 0], ac2[t, 1], ac2[t, 2], ac2[t, 3])
            else:
                gs2[j2] = f(j2, etacte2, ac2[t, 0], ac2[t, 1], ac2[t, 2], ac2[t, 3])
        for j1 in range(n1):
            for i2 in range(m2):
                #NOTE eta1 = xi2 along the common edge
                I = jac*integral_ff(j1, i2, al1[t, 0], al1[t, 1], al1[t, 2],
                        al1[t, 3], al2[t, 0], al2[t, 1], al2[t, 2], al2[t, 3])
                for i1 in range(m1):
                    f1 = fs1[i1]
                    for j2 in range(n2):
                        g2 = gs2[j2]
                        r[pos] = row0 + dof*(j1*m1 + i1) + fld1[t]
                        c[pos] = col0 + dof*(j2*m2 + i2) + fld2[t]
                        v[pos] = coeff[t]*I*f1*g2
                        pos += 1


def _block12(terms, object p1, object p2, double xcte1, double ycte2,
             int size, int row0, int col0):
    r"""
    Coupling block of ``p1`` (rows) and ``p2`` (columns)

    Each element ``(field1, der1, field2, der2, coeff)`` of ``terms`` adds
    ``coeff`` times the integral along the common edge of the product of the
    field ``field1`` of ``p1``, or its derivative along `x` when ``der1`` is
    ``1``, at `x_1 =` ``xcte1``, and the field ``field2`` of ``p2``, or its
    derivative along `y` when ``der2`` is ``1``, at `y_2 =` ``ycte2``. The
    term `(i, j)` of a panel, with `i` along `x` and `j` along `y`, has its
    Ritz constants at ``dofs*(j*m + i)``, with the number of DOFs per term
    ``dofs`` of the model of ``p1``.

    Parameters
    ----------
    terms : list of tuple
        Penalty terms, see above and :func:`._terms12`.
    p1, p2 : :class:`.Shell`
        Panels with the edges `x_1 =` constant and `y_2 =` constant.
    xcte1, ycte2 : float
        Coordinates of the edges, `x_1` in ``p1`` and `y_2` in ``p2``, length.
    size : int
        Size of the stiffness matrix of the assembly.
    row0, col0 : int
        First row of ``p1`` and first column of ``p2`` in the matrix of the
        assembly.

    Returns
    -------
    kC : scipy.sparse.coo_matrix
        The whole coupling block, shape ``(size, size)``.

    Notes
    -----
    Each term adds, to the entry of the Ritz constants `(i_1, j_1)` of ``p1``
    and `(i_2, j_2)` of ``p2``,

    .. math::

        \mathrm{coeff} \left(\frac{2}{a_1}\right)^{d_1}
        \left(\frac{2}{b_2}\right)^{d_2} \frac{b_1}{2} \int_{-1}^{1}
        g^1_{j_1}(\eta_1) f^2_{i_2}(\xi_2) \, d\eta_1 \;
        \partial_\xi^{d_1} f^1_{i_1}(\xi_1) \, \partial_\eta^{d_2}
        g^2_{j_2}(\eta_2)

    with `\eta_1 = \xi_2` along the common edge of length `b_1 = a_2`, `\xi_1 =
    2 x_{cte1}/a_1 - 1`, `\eta_2 = 2 y_{cte2}/b_2 - 1`, `d_1`, `d_2` given by
    ``der1``, ``der2``, and `f`, `g` the Legendre-based (Bardell) functions of
    each field with its boundary flags [1]_, the integral computed in closed
    form.

    References
    ----------
    .. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
       models to Address Linear Buckling and Vibration of Stiffened Composite
       Panels with Debonding Defect, Composite Structures, 160, 232-247, 2017.
       https://doi.org/10.1016/j.compstruct.2016.10.026

    """
    cdef int nt, n, dof, m1, n1, m2, n2
    cdef double xicte1, etacte2, jac
    cdef long [::1] fld1, fld2, der1, der2, r, c
    cdef double [::1] coeff, v, fs1, gs2
    cdef double [:, ::1] al1, al2, ac1, ac2
    dof = kCsdt._dofs(p1)
    nt = len(terms)
    fld1 = np.array([t[0] for t in terms], dtype=INT)
    der1 = np.array([t[1] for t in terms], dtype=INT)
    fld2 = np.array([t[2] for t in terms], dtype=INT)
    der2 = np.array([t[3] for t in terms], dtype=INT)
    #NOTE d/dx = (2/a1) d/dxi of p1 and d/dy = (2/b2) d/deta of p2
    coeff = np.array([t[4]*(2./p1.a)**t[1]*(2./p2.b)**t[3] for t in terms],
                     dtype=DOUBLE)
    al1 = _flags(p1, [t[0] for t in terms], 'y')
    ac1 = _flags(p1, [t[0] for t in terms], 'x')
    al2 = _flags(p2, [t[2] for t in terms], 'x')
    ac2 = _flags(p2, [t[2] for t in terms], 'y')
    m1, n1, m2, n2 = p1.m, p1.n, p2.m, p2.n
    xicte1 = 2*xcte1/p1.a - 1.
    etacte2 = 2*ycte2/p2.b - 1.
    #NOTE the edge spans the length b1 = a2
    jac = p1.b/2.
    n = nt*m1*n1*m2*n2
    r = np.zeros(n, dtype=INT)
    c = np.zeros(n, dtype=INT)
    v = np.zeros(n, dtype=DOUBLE)
    fs1 = np.zeros(m1, dtype=DOUBLE)
    gs2 = np.zeros(n2, dtype=DOUBLE)
    with nogil:
        _terms_xy(nt, fld1, fld2, der1, der2, coeff, al1, al2, ac1, ac2,
                  m1, n1, m2, n2, xicte1, etacte2, jac, dof, row0, col0,
                  fs1, gs2, r, c, v)
    return coo_matrix((v, (r, c)), shape=(size, size))


def _terms12(double kt, double kr, object p1, object p2, double angle):
    r"""
    Penalty terms of the coupling block of ``'BFxycte'``

    Parameters
    ----------
    kt, kr : float
        Translation and rotation penalty stiffnesses; a quantity whose penalty
        is zero adds no term.
    p1, p2 : :class:`.Shell`
        Panels with the edges `x_1 =` constant and `y_2 =` constant.
    angle : float
        Angle of the connection in degrees.

    Returns
    -------
    terms : list of tuple
        The terms ``(field1, der1, field2, der2, coeff)`` read by
        :func:`._block12`.

    Notes
    -----
    For each quantity `q` of :func:`.xycte_components`, with the penalty `k`,
    the coupling block of `\frac{1}{2} k \int (q_1 - q_2)^2` is `-k \int
    q_1^\top q_2`; with `q_p = \sum \alpha \, \partial^{d} q_p`, each pair of
    terms of ``p1`` and ``p2`` gives the coefficient `-k \alpha_1 \alpha_2`,
    the derivative of ``p1`` being along `x` and that of ``p2`` along `y`.

    """
    penalty = dict(kt=kt, kr=kr)
    terms = []
    for key, ((_, terms1), (_, terms2)) in xycte_components(p1, p2, angle):
        k = penalty[key]
        if k == 0:
            continue
        for f1, dx1, dy1, a1 in terms1:
            for f2, dx2, dy2, a2 in terms2:
                terms.append((f1, dx1, f2, dy2, -k*a1*a2))
    return terms


def fkCBFxycte11(double kt, double kr, object p1, double xcte1,
          int size, int row0, int col0):
    r"""
    Block of ``p1`` of the connection ``'BFxycte'``

    The quantities of ``p1`` are those of ``'BFxcte'``, `u_1, v_1, w_1` and the
    rotation about the common axis, and the block is that of
    :func:`.fkCBFxcte11`, integrated along the edge of ``p1``, for the models
    with 3 DOFs per term.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Panel with the edge `x_1 =` ``xcte1``, a model with 3 DOFs per term.
    xcte1 : float
        Coordinate `x_1` of the connection in ``p1``, length, between ``0``
        and ``p1.a``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0, col0 : int
        First row and first column of the block in the matrix of the assembly.

    Returns
    -------
    kCBFxycte11 : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p1``, shape ``(size, size)``.

    Notes
    -----
    The rotation of ``p1`` about the common axis is `\omega_1 = -w_{1,x}`, see
    the Notes of :mod:`panels.multidomain.connections.kCBFxycte` and of
    :func:`.fkCBFxcte11`.

    """
    return fkCBFxcte11(kt, kr, p1, xcte1, size, row0, col0)


def fkCBFxycte12(double kt, double kr, object p1, object p2,
          double xcte1, double ycte2, int size, int row0, int col0,
          double angle=90.):
    r"""
    Coupling block of the connection ``'BFxycte'``

    The penalty terms that couple ``p1`` (rows) and ``p2`` (columns), for the
    models with 3 and with 5 DOFs per term, from the quantities of
    :func:`.xycte_components`; the products of the approximation functions are
    integrated along the common edge and evaluated at ``xcte1`` in ``p1`` and
    ``ycte2`` in ``p2`` across it. The edges are checked with
    :func:`.check_xycte`. The whole block is returned.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Panel with the edge `x_1 =` ``xcte1``.
    p2 : :class:`.Shell`
        Panel with the edge `y_2 =` ``ycte2``, with ``p2.a`` equal to
        ``p1.b``.
    xcte1 : float
        Coordinate `x_1` of the connection in ``p1``, length, between ``0``
        and ``p1.a``.
    ycte2 : float
        Coordinate `y_2` of the connection in ``p2``, length, between ``0``
        and ``p2.b``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0 : int
        First row of ``p1`` in the matrix of the assembly.
    col0 : int
        First column of ``p2`` in the matrix of the assembly.
    angle : float, optional
        Angle `\theta` from the normal of ``p1`` to the normal of ``p2``
        about the common axis, in degrees. Default ``90.``.

    Returns
    -------
    kCBFxycte12 : scipy.sparse.coo_matrix
        Coupling block, rows of ``p1`` and columns of ``p2``, shape ``(size,
        size)``.

    Raises
    ------
    ValueError
        When the edges cannot coincide, see :func:`.check_xycte`.
    NotImplementedError
        When ``p1`` and ``p2`` have a different number of DOFs per term.

    Notes
    -----
    The block is `-k \int q_1^\top q_2 \, dy_1` for each quantity `q` of
    :func:`.xycte_components`, with its penalty `k`, ``kt`` or ``kr``, which
    serves the models with 3 and with 5 DOFs per term, with the term `v_2/r_2`
    of the Sanders-Koiter kinematics of ``p2`` when present; see
    :func:`._terms12`, :func:`._block12` and the Notes of
    :mod:`panels.multidomain.connections.kCBFxycte`.

    """
    check_xycte(p1, p2)
    return _block12(_terms12(kt, kr, p1, p2, angle), p1, p2, xcte1, ycte2,
                    size, row0, col0)


def fkCBFxycte22(double kt, double kr, object p2, double ycte2,
          int size, int row0, int col0, double rinv2=0.):
    r"""
    Block of ``p2`` of the connection ``'BFxycte'``

    The quantities of ``p2`` are those of the flange of ``'BFycte'``, `u_2,
    v_2, w_2` and the rotation about the common axis, and the block is that of
    :func:`.fkCBFycte22` with ``p2`` as both arguments, integrated over the
    length ``p2.a`` of the edge, for the models with 3 DOFs per term.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p2 : :class:`.Shell`
        Panel with the edge `y_2 =` ``ycte2``, a model with 3 DOFs per term.
    ycte2 : float
        Coordinate `y_2` of the connection in ``p2``, length, between ``0``
        and ``p2.b``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0, col0 : int
        First row and first column of the block in the matrix of the assembly.
    rinv2 : float, optional
        `1/r_2` when ``p2`` has the kinematics of Sanders-Koiter, zero
        otherwise. Default ``0.``.

    Returns
    -------
    kCBFxycte22 : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p2``, shape ``(size, size)``.

    Notes
    -----
    The rotation of ``p2`` about the common axis is `\omega_2 = w_{2,y} -
    v_2/r_2`, with `1/r_2` given by ``rinv2``, see the Notes of
    :mod:`panels.multidomain.connections.kCBFxycte` and of
    :func:`.fkCBFycte22`.

    """
    return fkCBFycte22(kt, kr, p2, p2, ycte2, size, row0, col0, rinv2=rinv2)


def fkCBFxycte11_sdt(double kt, double kr, object p1, double xcte1,
          int size, int row0, int col0):
    r"""
    Block of ``p1`` of ``'BFxycte'``, models with 5 DOFs per term

    The block of :func:`.fkCBFxcte11_sdt`, integrated along the edge of ``p1``,
    for the models based on the FSDT and TSDT.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Panel with the edge `x_1 =` ``xcte1``, a model with 5 DOFs per term.
    xcte1 : float
        Coordinate `x_1` of the connection in ``p1``, length, between ``0``
        and ``p1.a``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0, col0 : int
        First row and first column of the block in the matrix of the assembly.

    Returns
    -------
    kCBFxycte11_sdt : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p1``, shape ``(size, size)``.

    Raises
    ------
    ValueError
        When ``p1`` is not a model based on the FSDT or TSDT.

    Notes
    -----
    The rotation of ``p1`` about the common axis is `\theta_{x1}`, the rotation
    at the reference surface of :func:`.kCsdt._rotation_x`, see the Notes of
    :mod:`panels.multidomain.connections.kCBFxycte` and of
    :func:`.fkCBFxcte11_sdt`.

    """
    return fkCBFxcte11_sdt(kt, kr, p1, xcte1, size, row0, col0)


def fkCBFxycte22_sdt(double kt, double kr, object p2, double ycte2,
          int size, int row0, int col0, double rinv2=0.):
    r"""
    Block of ``p2`` of ``'BFxycte'``, models with 5 DOFs per term

    The block of :func:`.fkCBFycte22_sdt` with ``p2`` as both arguments,
    integrated over the length ``p2.a`` of the edge, for the models based on
    the FSDT and TSDT.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p2 : :class:`.Shell`
        Panel with the edge `y_2 =` ``ycte2``, a model with 5 DOFs per term.
    ycte2 : float
        Coordinate `y_2` of the connection in ``p2``, length, between ``0``
        and ``p2.b``.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    row0, col0 : int
        First row and first column of the block in the matrix of the assembly.
    rinv2 : float, optional
        `1/r_2` when ``p2`` has the kinematics of Sanders-Koiter, zero
        otherwise. Default ``0.``.

    Returns
    -------
    kCBFxycte22_sdt : scipy.sparse.coo_matrix
        Upper triangle of the block of ``p2``, shape ``(size, size)``.

    Raises
    ------
    ValueError
        When ``p2`` is not a model based on the FSDT or TSDT.

    Notes
    -----
    The rotation of ``p2`` about the common axis is `-\theta_{y2}`, the
    rotation at the reference surface of :func:`.kCsdt._rotation_y` with the
    term `v_2/r_2` given by ``rinv2``, see the Notes of
    :mod:`panels.multidomain.connections.kCBFxycte` and of
    :func:`.fkCBFycte22_sdt`.

    """
    return fkCBFycte22_sdt(kt, kr, p2, p2, ycte2, size, row0, col0,
                           rinv2=rinv2)
