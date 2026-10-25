r"""
Null-space method for the connections

Imposes the connections of a :class:`.MultiDomain` assembly exactly, with
the default ``conn_method='null-space'``, see :ref:`null_space` for the
usage and a comparison with the penalty method. The quantities penalized by
the kernels of :mod:`panels.multidomain.connections` become linear
constraints on the Ritz constants, built by :func:`.constraint_matrix`, one
row per quantity and point, and eliminated with a sparse basis of their null
space, built by :func:`.null_space_basis`. The matrices and vectors of the
assembly are reduced to the independent Ritz constants and the reduced
problem is solved with the solvers of ``structsolve`` unchanged, see
:meth:`.MultiDomain.get_Z`, :meth:`.MultiDomain.reduce`,
:meth:`.MultiDomain.expand` and :meth:`.MultiDomain.get_reduced_functions`.

The quantities of each connection are those of its penalty kernel, see
:ref:`null_space_quantities`. The connections available are listed in
:data:`NULL_SPACE_FUNCS`, the damaged connection ``'SB_TSL'`` remains a
penalty stiffness.

With ``MultiDomain(..., pd_method='null-space')`` the prescribed
displacements of the panels are constraints with a right-hand side, built
by :func:`.pd_constraint_matrix` and eliminated after the connections by
:class:`.Elimination`, which also gives the particular solutions, see
:ref:`null_space_pd`.

Notes
-----
The connections are imposed as the penalty kernels of
:mod:`panels.multidomain.connections` would with penalty constants going to
infinity. Each penalty kernel adds the energy `\frac{1}{2} k_k \int
(q_k^{(1)} - q_k^{(2)})^2` for the quantities `q_k` that must be continuous
across the connection, e.g. `u, v, w` and the rotation of the normal. With
Bardell's polynomial functions `q_k^{(1)} - q_k^{(2)}` is a polynomial along
the connection, and it vanishes everywhere when it vanishes at as many
Gauss-Legendre points as its number of coefficients. The connections are
then the linear constraints:

.. math::

    \boldsymbol{B} \boldsymbol{c} = \boldsymbol{0}

built by :func:`.constraint_matrix`, one row per quantity and point. They
are eliminated with a sparse basis `\boldsymbol{Z}` of the null space of `\boldsymbol{B}`, built by
:func:`.null_space_basis`:

.. math::

    \boldsymbol{c} = \boldsymbol{Z} \boldsymbol{c}_r, \qquad \boldsymbol{K}_r = \boldsymbol{Z}^\top \boldsymbol{K} \boldsymbol{Z}, \qquad
    \boldsymbol{F}_r = \boldsymbol{Z}^\top \boldsymbol{F}

The vectors with zero penalty energy are exactly those that satisfy the
constraints. Lagrange multipliers would add one unknown per constraint and
give an indefinite system, whose geometric stiffness and mass matrices have
zero rows and columns for the multipliers, giving infinite eigenvalues that
the shift-and-invert eigenvalue solvers of ``structsolve`` do not handle.
The null-space method removes unknowns instead: the reduced stiffness matrix
is symmetric and positive definite whenever the connected assembly has no
rigid-body motion, like the stiffness matrix of the penalty method. The
constraints are linear in `\boldsymbol{c}`, so `\boldsymbol{Z}` does not change during a
non-linear analysis, where `\boldsymbol{F}_{int, r} = \boldsymbol{Z}^\top \boldsymbol{F}_{int}(\boldsymbol{Z}
\boldsymbol{c}_r)` and `\boldsymbol{K}_{T r} = \boldsymbol{Z}^\top \boldsymbol{K}_T(\boldsymbol{Z} \boldsymbol{c}_r) \boldsymbol{Z}`.

The damaged connection ``'SB_TSL'`` is a physical stiffness of the
interface, not a constraint, and therefore remains a penalty stiffness.

With ``MultiDomain(..., pd_method='null-space')`` the prescribed
displacements of the panels are the inhomogeneous constraints `\boldsymbol{B}_d \boldsymbol{c}
= \boldsymbol{d}_0 + \lambda \boldsymbol{d}_1`, built by :func:`.pd_constraint_matrix` and
eliminated after the connections by :class:`.Elimination`, which also gives
the particular solutions `\boldsymbol{c} = \boldsymbol{Z} \boldsymbol{c}_r + \boldsymbol{c}_{p0} + \lambda
\boldsymbol{c}_{p1}`, see :ref:`null_space_pd`.

"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import qr, solve_triangular
from scipy.sparse import coo_matrix, csr_matrix, identity

from panels import DOUBLE
from panels.bardell import calc_vec_f, calc_vec_fp
from panels.logger import warn
from . import kCsdt


#: Connections that can be imposed with the null-space method
NULL_SPACE_FUNCS = ('SSxcte', 'SSycte', 'BFxcte', 'BFycte', 'SB')

U, V, W, PHIX, PHIY = range(5)
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')


_dofs = kCsdt._dofs
_rinv_sanders = kCsdt._rinv_sanders


def _length(p, direction):
    return p.a if direction == 'x' else p.b


def _num(p, direction):
    return p.m if direction == 'x' else p.n


def _eval(p, field, der, direction, xis):
    r"""Approximation functions of ``field`` along ``direction``, or their
    first derivative, at the natural coordinates ``xis``, with shape
    ``(len(xis), num)``

    The values of the fifth and following functions are set to zero at the
    ends ``xi = -1`` and ``xi = 1``.

    Notes
    -----
    The derivative along `x` is `(2/L) \partial/\partial \xi`, with `\xi =
    2 x/L - 1` and `L` the length of the panel along ``direction``.
    The fifth and following functions of Bardell vanish together with their
    slope at `\xi = \pm 1`; setting them to zero there removes their
    round-off, which would otherwise give spurious constraints once the rows
    of `\boldsymbol{B}` are normalized, see :meth:`.Elimination.add`.

    """
    flags = tuple(float(getattr(p, direction + edge + FIELDS[field] + r))
                  for edge, r in (('1', ''), ('1', 'r'), ('2', ''), ('2', 'r')))
    num = _num(p, direction)
    out = np.zeros((len(xis), num), dtype=DOUBLE)
    for k, xi in enumerate(xis):
        if der == 0:
            out[k] = calc_vec_f(xi, *flags)[:num]
        else:
            out[k] = calc_vec_fp(xi, *flags)[:num]*(2/_length(p, direction))
        #NOTE the functions of order 5 and higher vanish together with their
        #     slope at the ends, but not exactly in floating point. The
        #     round-off would give a row of B when all the other functions
        #     are removed by the boundary condition flags, e.g. with v fixed
        #     at the connected edge of both panels, which the normalization
        #     of the rows would turn into a spurious constraint
        if abs(xi) == 1.:
            out[k, 4:] = 0.
    return out


def _clpt_components(func, p1, p2):
    r"""Quantities of the edge connections between models with 3 DOFs, each
    one ``((p1, terms1), (p2, terms2))`` with ``terms`` a list of ``(field,
    der_x, der_y, coeff)``

    The quantities of ``'SB'``, the same for all the models, are those of
    :func:`._sb_components`.

    Notes
    -----
    The constraints are those of the penalty kernels of the classical
    laminated plate theory, with the fields of ``p1`` on the left side and
    those of ``p2`` on the right side:

    - ``'SSxcte'``: `u, v, w` and `w_{,x}` continuous;
    - ``'SSycte'``: `u, v, w` and `w_{,y}` continuous;
    - ``'BFycte'``: `u_1 = u_2`, `v_1 = w_2`, `w_1 = -v_2` and `w_{1,y} -
      v_1/r_1 = w_{2,y} - v_2/r_2`, the flange being normal to the base and
      the terms `v/r` present only with the Sanders-Koiter kinematics, see
      :mod:`panels.multidomain.connections.kCBFycte`;
    - ``'BFxcte'``: `u_1 = w_2`, `v_1 = v_2`, `w_1 = -u_2` and `w_{1,x} =
      w_{2,x}`, see :mod:`panels.multidomain.connections.kCBFxcte`.

    """
    if func == 'SSxcte':
        rot = [(W, 1, 0, 1.)]
        return [((p1, [(f, 0, 0, 1.)]), (p2, [(f, 0, 0, 1.)]))
                for f in (U, V, W)] + [((p1, rot), (p2, rot))]
    if func == 'SSycte':
        rot = [(W, 0, 1, 1.)]
        return [((p1, [(f, 0, 0, 1.)]), (p2, [(f, 0, 0, 1.)]))
                for f in (U, V, W)] + [((p1, rot), (p2, rot))]
    if func == 'BFycte':
        #NOTE rotation about x, w,y - v/r with Sanders-Koiter, see kCBFycte
        rot1 = [(W, 0, 1, 1.), (V, 0, 0, -_rinv_sanders(p1))]
        rot2 = [(W, 0, 1, 1.), (V, 0, 0, -_rinv_sanders(p2))]
        return [((p1, [(U, 0, 0, 1.)]), (p2, [(U, 0, 0, 1.)])),
                ((p1, [(V, 0, 0, 1.)]), (p2, [(W, 0, 0, 1.)])),
                ((p1, [(W, 0, 0, 1.)]), (p2, [(V, 0, 0, -1.)])),
                ((p1, rot1), (p2, rot2))]
    if func == 'BFxcte':
        rot = [(W, 1, 0, 1.)]
        return [((p1, [(U, 0, 0, 1.)]), (p2, [(W, 0, 0, 1.)])),
                ((p1, [(V, 0, 0, 1.)]), (p2, [(V, 0, 0, 1.)])),
                ((p1, [(W, 0, 0, 1.)]), (p2, [(U, 0, 0, -1.)])),
                ((p1, rot), (p2, rot))]
    raise NotImplementedError(func)


def _sb_components(p1, p2, kr_sb):
    r"""Quantities of ``'SB'`` for all the models, see
    :func:`.kCsdt._sb_components`, the rotations of the models with 5 DOFs
    per term being connected only when ``kr_sb`` is not zero

    Notes
    -----
    `u(z), v(z)` at the lower face `z_1 = d_1 - h_1/2` of ``p1`` equal to
    those at the upper face `z_2 = d_2 + h_2/2` of ``p2``, each panel with its
    own kinematics, see :func:`.kCsdt._surface_terms`, and `w_1 = w_2`. For
    the classical laminated plate theory, `u_1 - z_1 w_{1,x} = u_2 - z_2
    w_{2,x}` and `(1 + z_1/r_1) v_1 - z_1 w_{1,y} = (1 + z_2/r_2) v_2 - z_2
    w_{2,y}`, the terms `1/r` present only with the Sanders-Koiter
    kinematics. A rigid rotation about the axis of the cylinder, `v = r
    \Omega`, then gives no separation when the radius of each panel is that
    of its reference surface, `r_1 + z_1 = r_2 + z_2`. With ``kr_sb`` the
    rotations of the normal at the reference surfaces of the FSDT and TSDT,
    :func:`.kCsdt._rotation_x` and :func:`.kCsdt._rotation_y`, are also
    equal. ``kr_sb`` is not used for the classical laminated plate theory,
    whose rotations would add redundant constraints or, with the
    Sanders-Koiter kinematics, over-constrain the connection, `(z_1 - z_2)
    w_{,y} = 0`, see :func:`.kCsdt._sb_components`.

    """
    return [sides for _, sides in kCsdt._sb_components(p1, p2, 1., kr_sb)]


def _sdt_components(func, p1, p2):
    r"""Quantities of the edge connections between models with 5 DOFs, see
    :func:`._clpt_components`

    Notes
    -----
    The constraints are those of the penalty kernels of the shear
    deformation theories: for ``'SSxcte'`` and ``'SSycte'``, `u, v, w,
    \theta_x, \theta_y` and, for the TSDT, the derivative of `w` normal to
    the edge, see :func:`.kCsdt._edge_components`; for ``'BFycte'``, `u_1 =
    u_2`, `v_1 = w_2`, `w_1 = -v_2` and `\theta_{y1} = \theta_{y2}`; for
    ``'BFxcte'``, `u_1 = w_2`, `v_1 = v_2`, `w_1 = -u_2` and `\theta_{x1} =
    \theta_{x2}`; for ``'SB'`` see :func:`._sb_components`. The
    rotations `\theta_x`, `\theta_y` are those of the normal at the
    reference surface, :func:`.kCsdt._rotation_x` and
    :func:`.kCsdt._rotation_y`, with the offset of each panel for the TSDT.

    """
    if func in ('SSxcte', 'SSycte'):
        comps = kCsdt._edge_components(1., 1., p1, p2, func[2])
        return [sides for _, sides in comps]
    if func == 'BFycte':
        return [((p1, [(U, 0, 0, 1.)]), (p2, [(U, 0, 0, 1.)])),
                ((p1, [(V, 0, 0, 1.)]), (p2, [(W, 0, 0, 1.)])),
                ((p1, [(W, 0, 0, 1.)]), (p2, [(V, 0, 0, -1.)])),
                ((p1, kCsdt._rotation_y(p1)), (p2, kCsdt._rotation_y(p2)))]
    if func == 'BFxcte':
        return [((p1, [(U, 0, 0, 1.)]), (p2, [(W, 0, 0, 1.)])),
                ((p1, [(V, 0, 0, 1.)]), (p2, [(V, 0, 0, 1.)])),
                ((p1, [(W, 0, 0, 1.)]), (p2, [(U, 0, 0, -1.)])),
                ((p1, kCsdt._rotation_x(p1)), (p2, kCsdt._rotation_x(p2)))]
    raise NotImplementedError(func)


def _points(p1, p2, direction, ctes):
    r"""Natural coordinates of the points of ``p1`` and ``p2`` along
    ``direction``: the constant coordinates of the connection, or the
    Gauss-Legendre points along it, ``max(m1, m2, 4)`` or ``max(n1, n2,
    4)``, the same for both panels

    Raises a ``ValueError`` when the two panels do not have the same length
    along ``direction``.

    Notes
    -----
    The first 4 functions of Bardell are cubic, the term `i \ge 4` has the
    order `i`, and the derivatives only reduce the order, so the difference
    `q_k^{(1)} - q_k^{(2)}` of a quantity along the connection is a
    polynomial of `\xi` (or `\eta`) with at most ``max(num, 4)``
    coefficients. It vanishes everywhere when it vanishes at as many distinct points. The
    two panels share the natural coordinate along the connection, which
    requires the same length.

    """
    if ctes is not None:
        return [np.array([2*ctes[0]/_length(p1, direction) - 1]),
                np.array([2*ctes[1]/_length(p2, direction) - 1])]
    if not np.isclose(_length(p1, direction), _length(p2, direction)):
        raise ValueError('The panels must have the same length along {0}, '
                         'got {1} and {2}'.format(direction,
                         _length(p1, direction), _length(p2, direction)))
    #NOTE the first 4 functions of Bardell are cubic, the term i >= 4 has
    #     order i, and the derivatives only reduce the order
    num = max(_num(p1, direction), _num(p2, direction), 4)
    xis = leggauss(num)[0]
    return [xis, xis]


def constraint_matrix(connecti, size):
    r"""Rows of `\boldsymbol{B}` of a single connection

    The quantities of the connection, those of its penalty kernel, are
    evaluated at the points of the connection: the constant coordinates
    ``xcte1``, ``xcte2`` or ``ycte1``, ``ycte2`` across it and, along it,
    as many Gauss-Legendre points as the coefficients of the polynomials,
    ``max(m, 4)`` or ``max(n, 4)`` of the panel with the most terms; for
    ``'SB'`` these points are used along both `x` and `y`. Each quantity
    gives one row per point, the terms of ``p1`` with a positive sign and
    those of ``p2`` with a negative sign, in the columns starting at
    ``col_start`` of each panel. The edge connections of the models with 3
    DOFs per term use the quantities of the classical laminated plate
    theory, those of the models with 5 DOFs the quantities of
    :mod:`.kCsdt`; ``'SB'`` uses for all the models the displacements at the
    faces in contact, the quantities of the penalty matrix of
    :func:`.skin_base.fkCSB`, see :func:`._sb_components`, and also
    connects the rotations of the FSDT and TSDT when ``connecti['kr']`` is
    given and not zero.

    Parameters
    ----------
    connecti : dict
        A connection, as in :meth:`.MultiDomain.get_kC_conn`, with ``'func'``
        one of :data:`NULL_SPACE_FUNCS`. The coordinates of the connection,
        ``xcte1`` or ``ycte1``, refer to ``p1``, and ``xcte2`` or ``ycte2``
        to ``p2``. For ``'BFxcte'`` and ``'BFycte'``, ``p1`` is the base and
        ``p2`` the flange; for ``'SB'``, ``p1`` is the panel on the
        positive side of the interface along the `z` axis of both panels,
        and ``p2`` the one on the negative side, see
        :meth:`.MultiDomain.get_kC_conn`.
    size : int
        Size of the assembly.

    Returns
    -------
    B : scipy.sparse.csr_matrix
        The constraints ``B @ c = 0``, with ``size`` columns and one row per
        quantity and point. The rows are not normalized.

    Raises
    ------
    ValueError
        When ``p1`` and ``p2`` are the same object; for ``'SB'``, when the
        panels do not have the same dimensions ``a`` and ``b``.
    NotImplementedError
        When the models of ``p1`` and ``p2`` have a different number of
        DOFs per term; when ``connecti['func']`` is not one of
        :data:`NULL_SPACE_FUNCS`.

    See Also
    --------
    null_space_basis

    Notes
    -----
    Each row is `q_k^{(1)} - q_k^{(2)}` at one point, where the quantity
    `q_k` is a linear combination of the fields and derivatives of a panel,
    see :func:`._clpt_components`, :func:`._sdt_components` and
    :func:`._sb_components`, such that
    `\boldsymbol{B} \boldsymbol{c} = \boldsymbol{0}` holds exactly when the penalty energy
    `\frac{1}{2} k_k \int (q_k^{(1)} - q_k^{(2)})^2` of the connection
    vanishes for any `k_k > 0`, see :func:`._points`. For ``'SB'`` the
    differences are polynomials of `\xi` and `\eta`, and the tensor grid of
    the Gauss-Legendre points along `x` and `y` gives as many points as
    their coefficients. The approximation functions are products `f(\xi)
    g(\eta)`, so each block is the Kronecker product of the values along
    `x` and along `y`.

    """
    func = connecti['func']
    p1, p2 = connecti['p1'], connecti['p2']
    if p1 is p2:
        raise ValueError('A connection must be between two different panels')
    dofs1, dofs2 = _dofs(p1), _dofs(p2)
    if dofs1 != dofs2:
        raise NotImplementedError(
            "Connection '{0}' between models with a different number of "
            "DOFs per term, got models '{1}' and '{2}'".format(func,
            p1.model, p2.model))
    if func in ('SSxcte', 'BFxcte'):
        ctes = dict(x=(connecti['xcte1'], connecti['xcte2']), y=None)
    elif func in ('SSycte', 'BFycte'):
        ctes = dict(x=None, y=(connecti['ycte1'], connecti['ycte2']))
    elif func == 'SB':
        if not (np.isclose(p1.a, p2.a) and np.isclose(p1.b, p2.b)):
            raise ValueError('The panels of a "SB" connection must have the '
                             'same dimensions')
        ctes = dict(x=None, y=None)
    else:
        raise NotImplementedError(
            "Connection '{0}' is not available with the null-space method, "
            "the available ones are {1}".format(func, NULL_SPACE_FUNCS))
    if func == 'SB':
        comps = _sb_components(p1, p2, connecti.get('kr', 0.))
    elif dofs1 == 3:
        comps = _clpt_components(func, p1, p2)
    else:
        comps = _sdt_components(func, p1, p2)
    xis = _points(p1, p2, 'x', ctes['x'])
    etas = _points(p1, p2, 'y', ctes['y'])
    npts = len(xis[0])*len(etas[0])

    rows, cols, vals = [], [], []
    row0 = 0
    for sides in comps:
        for iside, (p, terms) in enumerate(sides):
            sign = 1. if iside == 0 else -1.
            for field, der_x, der_y, coeff in terms:
                if coeff == 0:
                    continue
                F = _eval(p, field, der_x, 'x', xis[iside])
                G = _eval(p, field, der_y, 'y', etas[iside])
                #NOTE Ritz constant of term (i, j) at dofs*(j*m + i), the
                #     row of point (xi_k, eta_l) is l*len(xis) + k
                block = sign*coeff*np.kron(G, F)
                r, c = np.nonzero(block)
                rows.append(row0 + r)
                cols.append(p.col_start + _dofs(p)*c + field)
                vals.append(block[r, c])
        row0 += npts
    B = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(row0, size))
    B.sum_duplicates()
    return B.tocsr()


#: Components that can be prescribed exactly, in the order of the values
#: stored by :meth:`.Shell.add_point_pd`, :meth:`.Shell.add_distr_pd_fixed_x`
#: and :meth:`.Shell.add_distr_pd_fixed_y`
PD_COMPONENTS = FIELDS


def _pd_terms(p, comp, z=0.):
    r"""Terms ``(field, der_x, der_y, coeff)`` of a prescribed component,
    the displacements `u` and `v` at the thickness coordinate ``z``, `w`
    and the rotations of the normal at the reference surface

    The terms with a zero coefficient are dropped.

    Notes
    -----
    The in-plane displacements at the coordinate `z`, measured from the
    reference surface, are those of the displacement field of the model:

    .. math::

        u(z) = u + z \Phi_x + p(z) \left(\phi_x + w_{,x}\right) \qquad
        v(z) = v + z \Phi_y + p(z) \left(\phi_y + w_{,y}\right)

    with `\Phi_x = -w_{,x}`, `\Phi_y = -w_{,y}` and `p(z) = 0` for the
    classical laminated plate theory, `\Phi_x = \phi_x`, `\Phi_y = \phi_y`
    for the shear deformation theories, with `p(z) = 0` for the FSDT and
    `p(z) = -c_1 [(z - d)^3 + d^3]`, `c_1 = 4/(3 h^2)`, for the TSDT, see
    :func:`.kCsdt._surface_terms`, and the term `v/r` added to `\Phi_y` with
    the Sanders-Koiter kinematics. For `z = 0` they are the fields `u` and
    `v`.

    The rotations are `\phi_x = -w_{,x}` and `\phi_y = -w_{,y}` for the
    classical laminated plate theory, the rows of ``fg`` of the field module
    of the model, and for the shear deformation theories the rotations at
    the reference surface `z = 0`, `\theta_x` and `\theta_y` of
    :func:`.kCsdt._rotation_x` and :func:`.kCsdt._rotation_y`, the
    rotations also tied by the connections:

    .. math::

        \theta_x = \phi_x - \frac{4 d^2}{h^2} \left(\phi_x + w_{,x}\right)
        \qquad
        \theta_y = \Phi_y - \frac{4 d^2}{h^2} \left(\phi_y + w_{,y}\right)

    with the offset `d` and the thickness `h` of the TSDT panel, which are
    the rows ``phix`` and ``phiy`` of ``fg``, `\phi_x` and `\Phi_y`, for
    the FSDT and for the TSDT with `d = 0`. `\Phi_y = \phi_y + v/r` with the
    Sanders-Koiter kinematics, and likewise `v/r` is added to `-w_{,y}` for
    the classical laminated plate theory.

    """
    if comp == W or (comp in (U, V) and z == 0):
        return [(comp, 0, 0, 1.)]
    if comp in (U, V):
        #NOTE the operator of the skin-base connection 'SB', all the models
        tu, tv = kCsdt._surface_terms(p, z)
        terms = tu if comp == U else tv
        return [term for term in terms if term[3] != 0]
    #NOTE for the TSDT with an offset, phix and phiy are the rotations at the
    #     mid-surface, the rotation at z = 0 includes the transverse shear;
    #     for the CLPT -w,x and -w,y + v/r
    rotation = kCsdt._rotation_x if comp == PHIX else kCsdt._rotation_y
    return [term for term in rotation(p) if term[3] != 0]


def pd_rows(p, comp, xis, etas, z=0., col0=0, size=None):
    r"""Rows of a prescribed component at the points of a grid

    One row per point ``(xis[k], etas[l])``, ordered as ``l*len(xis) + k``,
    whose product with the Ritz constants gives the component at the point,
    see :func:`._pd_terms`.

    Parameters
    ----------
    p : :class:`.Shell`
        Panel.
    comp : int
        Index of the component in :data:`PD_COMPONENTS`.
    xis, etas : array-like
        Natural coordinates of the points along `x` and `y`, between ``-1``
        and ``1``.
    z : float, optional
        Thickness coordinate, measured from the reference surface, of the
        in-plane displacements ``'u'`` and ``'v'``, a length; not used by
        the other components. Default ``0.``, the reference surface.
    col0 : int, optional
        Column of the first Ritz constant of ``p``. Default ``0``.
    size : int or None, optional
        Number of columns. Default ``None``, ``col0`` plus the number of
        Ritz constants of ``p``.

    Returns
    -------
    B : scipy.sparse.csr_matrix
        The rows, with the shape ``(len(xis)*len(etas), size)``.

    """
    if size is None:
        size = col0 + _dofs(p)*p.m*p.n
    rows, cols, vals = [], [], []
    for field, der_x, der_y, coeff in _pd_terms(p, comp, z):
        F = _eval(p, field, der_x, 'x', xis)
        G = _eval(p, field, der_y, 'y', etas)
        block = coeff*np.kron(G, F)
        r, c = np.nonzero(block)
        rows.append(r)
        cols.append(col0 + _dofs(p)*c + field)
        vals.append(block[r, c])
    B = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(xis)*len(etas), size))
    B.sum_duplicates()
    return B.tocsr()


def pd_points(p, x, y):
    r"""Points of a prescribed displacement of ``p``

    A point ``(x, y)``, or the line ``x`` (``y=None``) or ``y`` (``x=None``)
    with as many Gauss-Legendre points as the coefficients of the
    polynomials along it, ``max(n, 4)`` or ``max(m, 4)``, see
    :func:`._points`.

    Parameters
    ----------
    p : :class:`.Shell`
        Panel with the prescribed displacement.
    x : float or None
        Coordinate `x` of the point or of the line, length, between ``0``
        and ``p.a``. ``None`` for a line along `x` at the constant ``y``.
    y : float or None
        Coordinate `y` of the point or of the line, length, between ``0``
        and ``p.b``. ``None`` for a line along `y` at the constant ``x``.

    Returns
    -------
    xis, etas : np.ndarray
        Natural coordinates of the points, the rows of the constraints being
        ordered as ``l*len(xis) + k`` for the point ``(xis[k], etas[l])``.
    s : np.ndarray or None
        Coordinate along the line of each row, where the prescribed function
        is evaluated, ``None`` for a point. Length, 1-D array with the size
        of the Gauss-Legendre points.

    Raises
    ------
    ValueError
        When both ``x`` and ``y`` are ``None``; when ``x`` is out of
        ``[0, p.a]`` or ``y`` out of ``[0, p.b]``, with a relative tolerance
        of ``1e-12``.

    Notes
    -----
    Along a line, the prescribed component of the panel is a polynomial of
    the coordinate along the line with at most as many coefficients as the
    points, see :func:`._points`. Imposing the prescribed function at the
    Gauss-Legendre points therefore makes the component equal to the
    polynomial interpolation of the prescribed function at these points,
    which is the prescribed function itself when it is such a polynomial
    and the boundary condition flags do not remove the needed terms.

    """
    if x is None and y is None:
        raise ValueError('x and y cannot be None when defining displacements')
    for name, value, length in (('x', x, p.a), ('y', y, p.b)):
        if value is not None and not (-1e-12*length <= value
                                      <= length*(1 + 1e-12)):
            raise ValueError('The prescribed displacement at {0}={1} is out '
                             'of the panel, 0 <= {0} <= {2}'.format(name,
                             value, length))
    if x is not None and y is not None:
        return np.array([2*x/p.a - 1]), np.array([2*y/p.b - 1]), None
    if y is None:
        etas = leggauss(max(p.n, 4))[0]
        return np.array([2*x/p.a - 1]), etas, (etas + 1)*p.b/2
    xis = leggauss(max(p.m, 4))[0]
    return xis, np.array([2*y/p.b - 1]), (xis + 1)*p.a/2


def pd_constraint_matrix(p, comp, x, y, size, z=0.):
    r"""Rows `\boldsymbol{B}` of a prescribed component at a point or along a line

    The component is evaluated at the points of :func:`.pd_points`, one row
    per point, see :func:`.pd_rows`. The in-plane displacements ``'u'`` and
    ``'v'`` are those at the thickness coordinate ``z``, which for ``z``
    different from zero combine `u` or `v` with a rotation and a slope of
    `w`. The rotations ``'phix'`` and ``'phiy'`` are those of the normal at
    the reference surface `z = 0`, see :func:`._pd_terms`, derived from `w`
    for the models with 3 DOFs per term and, for those with 5 DOFs, the
    fields ``phix`` and ``phiy`` plus, for the TSDT with an offset, the
    terms of the transverse shear. The rows are not normalized.

    Parameters
    ----------
    p : :class:`.Shell`
        Panel, with ``col_start`` defined.
    comp : int
        Index of the component in :data:`PD_COMPONENTS`.
    x, y : float or None
        The point, or the line, see :func:`.pd_points`.
    size : int
        Size of the assembly.
    z : float, optional
        Thickness coordinate of ``'u'`` and ``'v'``, measured from the
        reference surface, a length. Default ``0.``.

    Returns
    -------
    B : scipy.sparse.csr_matrix
        The rows of ``B @ c = d``, one per point of :func:`.pd_points`.

    Notes
    -----
    The in-plane displacements at `z` are, see :func:`._pd_terms`,

    .. math::

        u(z) = u + \left(z - c_1 g\right) \phi_x - c_1 g w_{,x} \qquad
        v(z) = \left(1 + \frac{z}{r}\right) v + \left(z - c_1 g\right)
        \phi_y - c_1 g w_{,y}

    with `g = (z - d)^3 + d^3`, `c_1 = 4/(3 h^2)` for the TSDT and `c_1 =
    0` for the FSDT, the term `z/r` only with the Sanders-Koiter
    kinematics, and `u(z) = u - z w_{,x}`, `v(z) = (1 + z/r) v - z w_{,y}`
    for the classical laminated plate theory.

    The rotations ``'phix'`` and ``'phiy'`` are `-w_{,x}` and `-w_{,y}` for
    the models with 3 DOFs per term and, for those with 5 DOFs, the
    rotations at the reference surface

    .. math::

        \theta_x = \phi_x - \frac{4 d^2}{h^2} \left(\phi_x + w_{,x}\right)
        \qquad
        \theta_y = \phi_y - \frac{4 d^2}{h^2} \left(\phi_y + w_{,y}\right)

    with the offset `d` and the thickness `h` of a TSDT panel, `\theta =
    \phi` for the FSDT and for `d = 0`, see :func:`.kCsdt._rotation_x` and
    :func:`.kCsdt._rotation_y`; the term `v/r` is added to ``'phiy'`` for
    the models with the Sanders-Koiter kinematics.

    """
    xis, etas, _ = pd_points(p, x, y)
    return pd_rows(p, comp, xis, etas, z, p.col_start, size)


def pd_entries(p):
    r"""Prescribed components of a panel, ``(cte, x, y, comp, value, z)`` in
    the order of :func:`.pd_blocks`, the constant ones first

    The constant prescriptions are read from ``p.point_pds_values`` and
    ``p.distr_pds_values``, those multiplied by the load factor from
    ``p.point_pds_values_inc`` and ``p.distr_pds_values_inc``, the points
    before the lines in each group. Each stored prescription gives one entry
    per component whose value is not ``None``.

    Parameters
    ----------
    p : :class:`.Shell`
        Panel with the prescribed displacements.

    Yields
    ------
    cte : bool
        ``True`` for a constant prescription, ``False`` for one multiplied
        by the load factor.
    x, y : float or None
        The point, or the line, see :func:`.pd_points`.
    comp : int
        Index of the component in :data:`PD_COMPONENTS`.
    value : float or callable
        Prescribed value, a function of the coordinate along the line for a
        line, see :func:`.pd_rhs`.
    z : float
        Thickness coordinate of the in-plane displacements, see
        :meth:`.Shell.add_point_pd`.

    """
    for cte, points, lines in ((True, p.point_pds_values, p.distr_pds_values),
            (False, p.point_pds_values_inc, p.distr_pds_values_inc)):
        for values in points + lines:
            x, y = values[:2]
            z = values[7]
            for comp, value in enumerate(values[2:7]):
                if value is not None:
                    yield cte, x, y, comp, value, z


def pd_blocks(p, size):
    r"""Prescribed displacements of a panel imposed exactly

    One block per prescription and component, see
    :meth:`.Shell.add_point_pd`, :meth:`.Shell.add_distr_pd_fixed_x` and
    :meth:`.Shell.add_distr_pd_fixed_y`, the constant ones first.

    Parameters
    ----------
    p : :class:`.Shell`
        Panel with the prescribed displacements, with ``col_start`` defined.
    size : int
        Size of the assembly.

    Returns
    -------
    blocks : list of dict
        With the keys ``'panel'``, ``'x'``, ``'y'``, ``'z'``, ``'comp'``
        (index in :data:`PD_COMPONENTS`), ``'value'`` (float or function),
        ``'cte'`` and ``'B'``, see :func:`.pd_constraint_matrix`.

    """
    return [dict(panel=p, x=x, y=y, z=z, comp=comp, value=value, cte=cte,
                 B=pd_constraint_matrix(p, comp, x, y, size, z))
            for cte, x, y, comp, value, z in pd_entries(p)]


def pd_rhs(block):
    r"""Prescribed values of a block of :func:`.pd_blocks` at its points

    For a point the value is used as given; for a line the function
    ``block['value']`` is called at the coordinate along the line of each
    point, see :func:`.pd_points`.

    Parameters
    ----------
    block : dict
        A block of :func:`.pd_blocks`.

    Returns
    -------
    d : np.ndarray
        1-D array with one value per row of ``block['B']``.

    """
    _, _, s = pd_points(block['panel'], block['x'], block['y'])
    value = block['value']
    if s is None:
        return np.array([value], dtype=DOUBLE)
    return np.array([value(si) for si in s], dtype=DOUBLE)


class Elimination(object):
    r"""Elimination of linear constraints, one block of rows at a time

    The constraints of each block, rows ``B`` with right-hand sides ``d``,
    are written in the independent Ritz constants left by the previous
    blocks and eliminated with a QR decomposition with column pivoting, see
    :meth:`.add`. The attribute ``Z`` holds the basis of the independent
    Ritz constants left, initially the identity. The blocks added with
    ``rhs=True``, e.g. the prescribed displacements, keep the data needed to
    compute particular solutions for any right-hand sides, see
    :meth:`.particular`, without repeating the decompositions.

    Parameters
    ----------
    size : int
        Size of the assembly.
    tol : float
        Tolerance to find the rank of the constraints, see
        :func:`.null_space_basis`.

    Notes
    -----
    The constraints of each block, `\boldsymbol{B} \boldsymbol{c} = \boldsymbol{d}`, are written in the
    independent Ritz constants left by the previous blocks, `\boldsymbol{c} = \boldsymbol{Z}
    \boldsymbol{c}_r + \boldsymbol{c}_p`, where `\boldsymbol{Z}` is a basis of the null space of the
    homogeneous constraints of the previous blocks and `\boldsymbol{c}_p` a
    particular solution of all the previous blocks. Eliminating the blocks
    one at a time keeps `\boldsymbol{Z}` sparse, each block involving only the Ritz
    constants of its panels.

    """
    def __init__(self, size, tol):
        self.size = size
        self.tol = tol
        self.Z = identity(size, dtype=DOUBLE, format='csr')
        self.min_kept = np.inf
        self.max_dropped = 0.
        #: data of the blocks added with ``rhs=True``, in order
        self.steps = []

    def add(self, B, rhs=False, info=None):
        r"""Eliminate the constraints of a block

        The rows of ``B`` are normalized to a unit norm and written in the
        current independent Ritz constants, ``B @ Z``. The columns with no
        entry larger than ``tol`` are skipped, the others are decomposed by
        a QR decomposition with column pivoting, and the constants of the
        ``rank`` pivots larger than ``tol`` are eliminated. The attribute
        ``Z`` is replaced by the basis of the remaining constants, and
        ``min_kept`` and ``max_dropped`` are updated with the smallest pivot
        kept and the largest pivot dropped. With ``rhs=True`` the data of
        the block are appended to the attribute ``steps``, also when nothing
        is eliminated. When no column is left, ``Z`` is not changed.

        Parameters
        ----------
        B : sparse matrix
            Rows of the constraints, with ``size`` columns.
        rhs : bool, optional
            Whether the constraints may have a non-null right-hand side.
            The homogeneous blocks must all be added before the first block
            with ``rhs=True``. Default ``False``.
        info : object, optional
            Description of the block, stored with its data. Default
            ``None``.

        Raises
        ------
        RuntimeError
            When ``rhs=False`` and a block with ``rhs=True`` was already
            added.

        Notes
        -----
        With the normalized rows `\boldsymbol{B}`, `\bar{\boldsymbol{B}} = \boldsymbol{B} \boldsymbol{Z}` and the columns
        `J` of `\bar{\boldsymbol{B}}` kept, the QR decomposition with column pivoting
        `\bar{\boldsymbol{B}}_J \boldsymbol{P} = \boldsymbol{Q} \boldsymbol{R}`, with the upper triangular factor `\boldsymbol{R}`,
        not the residual, reveals the rank `r`, the number of diagonal terms
        of `\boldsymbol{R}` larger than ``tol``. The constants
        `\boldsymbol{c}_e` of the first `r` pivot columns are eliminated in favour of
        the remaining ones, `\boldsymbol{c}_k`:

        .. math::

            \boldsymbol{R}_{11} \boldsymbol{c}_e + \boldsymbol{R}_{12} \boldsymbol{c}_k = \boldsymbol{Q}_1^\top \boldsymbol{d}, \qquad
            \boldsymbol{c}_e = -\boldsymbol{R}_{11}^{-1} \boldsymbol{R}_{12} \boldsymbol{c}_k
                      + \boldsymbol{R}_{11}^{-1} \boldsymbol{Q}_1^\top \boldsymbol{d}

        where `\boldsymbol{R}_{11}` is the leading `r \times r` triangle of `\boldsymbol{R}` and
        `\boldsymbol{Q}_1` the first `r` columns of `\boldsymbol{Q}`. The homogeneous part gives
        the new basis `\boldsymbol{Z} \boldsymbol{S}`, `\boldsymbol{S}` mapping the remaining constants to
        all the current ones, the terms of `-\boldsymbol{R}_{11}^{-1} \boldsymbol{R}_{12}` smaller
        than ``1e-14`` times its largest one being dropped to keep `\boldsymbol{Z}`
        sparse. The column pivoting reveals the rank and chooses the
        eliminated constants such that `\boldsymbol{R}_{11}` is well conditioned. The
        rows are normalized only once,
        before being multiplied by `\boldsymbol{Z}`, so that a redundant row has only
        round-off errors, compared with the absolute tolerance ``tol``. The
        homogeneous blocks come first so that the particular solution,
        built in :meth:`.particular` with the basis of each block with
        ``rhs=True``, satisfies all the homogeneous constraints.

        """
        Z = self.Z
        tol = self.tol
        B = csr_matrix(B)
        #NOTE the rows are normalized only once, so that a redundant row,
        #     after written in the reduced Ritz constants, has only round-off
        #     errors that are compared with the absolute tolerance
        norms = np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
        B = csr_matrix(B.multiply(1./np.where(norms > 0, norms, 1.)[:, None]))
        Bz = (B @ Z).toarray()
        step = None
        if rhs:
            step = dict(info=info, B=B, norms=norms, rank=0)
            self.steps.append(step)
        elif self.steps:
            raise RuntimeError('The homogeneous constraints must be added '
                               'before the ones with a right-hand side')
        J = np.flatnonzero(np.abs(Bz).max(axis=0, initial=0) > tol)
        if J.size == 0:
            return
        Q, R, P = qr(Bz[:, J], mode='economic', pivoting=True)
        d = np.abs(np.diag(R))
        rank = int(np.sum(d > tol))
        if rank < d.size:
            self.max_dropped = max(self.max_dropped, d[rank])
        if rank == 0:
            return
        self.min_kept = min(self.min_kept, d[rank - 1])
        eliminated = J[P[:rank]]
        kept = J[P[rank:]]
        #NOTE R11 c_eliminated + R12 c_kept = Q1^T d
        X = -solve_triangular(R[:rank, :rank], R[:rank, rank:])
        X[np.abs(X) < 1e-14*np.abs(X).max(initial=0.)] = 0.
        if step is not None:
            #NOTE particular solution with c_kept = 0, see particular()
            step.update(rank=rank, Q1=Q[:, :rank], R11=R[:rank, :rank],
                        Bze=Bz[:, eliminated], Ze=Z[:, eliminated].tocsc())
        nr = Z.shape[1]
        remaining = np.setdiff1d(np.arange(nr), eliminated)
        new_index = -np.ones(nr, dtype=int)
        new_index[remaining] = np.arange(remaining.size)
        re, ce = np.nonzero(X)
        S = coo_matrix((np.concatenate((np.ones(remaining.size), X[re, ce])),
                        (np.concatenate((remaining, eliminated[re])),
                         np.concatenate((np.arange(remaining.size),
                                         new_index[kept[ce]])))),
                       shape=(nr, remaining.size))
        self.Z = (Z @ S.tocsr()).tocsr()

    def finish(self):
        r"""Basis `\boldsymbol{Z}` of the null space of all the constraints added

        A warning is issued when a pivot was dropped and the smallest pivot
        kept is less than ``1e4`` times the largest pivot dropped. The
        explicit zeros of the attribute ``Z`` are removed, in place.

        Returns
        -------
        Z : scipy.sparse.csr_matrix
            The attribute ``Z``, with shape ``(size, size - rank)``, ``rank``
            being the number of constraints eliminated.

        Notes
        -----
        The independent constraints have pivots many orders of magnitude
        larger than the round-off left in the redundant ones; a gap smaller
        than ``1e4`` means that the separation between independent and
        redundant constraints, and therefore the rank and `\boldsymbol{Z}`, depend on
        ``tol``.

        """
        if self.max_dropped > 0 and self.min_kept < 1e4*self.max_dropped:
            warn('Null-space method: the rank of the constraints is not well '
                 'defined, the smallest independent pivot is {0:.1e} and the '
                 'largest redundant one {1:.1e}, see the parameter "tol"'
                 .format(self.min_kept, self.max_dropped))
        self.Z.eliminate_zeros()
        return self.Z

    def particular(self, ds, rhs_tol=1.e-8, describe=None):
        r"""Particular solution of the blocks added with ``rhs=True``

        The right-hand sides are divided by the norms of the rows of their
        blocks, as the rows were, and the blocks are solved in the order
        they were added, each one with the Ritz constants that it
        eliminated, using the decompositions stored by :meth:`.add`. The
        residual of each block, including its redundant rows, is compared
        with ``rhs_tol`` times the largest normalized right-hand side. When
        all the right-hand sides are zero a vector of zeros is returned.

        Parameters
        ----------
        ds : list of array-like
            Right-hand side ``d`` of each block added with ``rhs=True``,
            in the order they were added.
        rhs_tol : float, optional
            Relative tolerance of the consistency of the right-hand sides.
            The constraints dropped as redundant, e.g. at the corner where a
            prescribed edge meets a connection or another prescribed edge,
            must be satisfied by the particular solution within ``rhs_tol``
            times the largest prescribed value, both for the normalized
            rows. Default ``1.e-8``.
        describe : callable, optional
            ``describe(i, info, res, atol)`` returns the message of the
            error raised when the block ``i`` is not consistent, ``res``
            being the residuals of its normalized rows. Default ``None``, a
            generic message.

        Returns
        -------
        c_p : np.ndarray
            Vector with ``size`` elements satisfying ``B @ c_p = d`` for all
            the blocks, and ``B @ c_p = 0`` for the homogeneous ones.

        Raises
        ------
        ValueError
            When the number of right-hand sides is not the number of blocks
            added with ``rhs=True``; when the right-hand sides are not
            consistent.

        Notes
        -----
        For the block `i`, with the normalized rows `\boldsymbol{B}_i` and right-hand
        side `\bar{\boldsymbol{d}}_i`, the residual of the current particular solution
        is `\boldsymbol{\delta} = \bar{\boldsymbol{d}}_i - \boldsymbol{B}_i \boldsymbol{c}_p`, and

        .. math::

            \boldsymbol{y} = \boldsymbol{R}_{11}^{-1} \boldsymbol{Q}_1^\top \boldsymbol{\delta}, \qquad
            \boldsymbol{c}_p \leftarrow \boldsymbol{c}_p + \boldsymbol{Z}_e \boldsymbol{y}

        where `\boldsymbol{Z}_e` are the columns of the basis `\boldsymbol{Z}` of the eliminated
        constants, both stored by :meth:`.add`. The columns of `\boldsymbol{Z}`
        satisfy the homogeneous constraints of the previous blocks, which
        are therefore not changed, and the remaining constants of the block
        are set to zero. The residual `\boldsymbol{\delta} - \boldsymbol{B}_i \boldsymbol{Z}_e \boldsymbol{y}` is
        zero when `\boldsymbol{\delta}` is in the range of `\boldsymbol{Q}_1`, i.e. when the
        right-hand side is consistent with the redundant rows.

        """
        cp = np.zeros(self.size, dtype=DOUBLE)
        if len(ds) != len(self.steps):
            raise ValueError('Expected {0} right-hand sides, got {1}'.format(
                             len(self.steps), len(ds)))
        dns = [np.asarray(d, dtype=DOUBLE)/np.where(step['norms'] > 0,
               step['norms'], 1.) for d, step in zip(ds, self.steps)]
        scale = max([np.abs(dn).max(initial=0.) for dn in dns], default=0.)
        if scale == 0:
            return cp
        atol = rhs_tol*scale
        for i, (step, dn) in enumerate(zip(self.steps, dns)):
            dt = dn - step['B'] @ cp
            if step['rank'] > 0:
                y = solve_triangular(step['R11'], step['Q1'].T @ dt)
                res = dt - step['Bze'] @ y
                cp += step['Ze'] @ y
            else:
                res = dt
            if np.abs(res).max(initial=0.) > atol:
                if describe is None:
                    msg = ('The right-hand side of the constrained block {0} '
                           'is not consistent with the previous constraints'
                           .format(i))
                else:
                    msg = describe(i, step['info'], res, atol)
                raise ValueError(msg)
        return cp


def null_space_basis(Bs, size, tol=1.e-9):
    r"""Sparse basis `\boldsymbol{Z}` of the null space of the constraints

    The constraints of ``Bs`` are eliminated one connection at a time, in
    the order given, with :meth:`.Elimination.add`, and the basis is
    returned by :meth:`.Elimination.finish`, which warns when the rank of
    the constraints is not well defined.

    Parameters
    ----------
    Bs : list of sparse matrices
        The constraints of each connection, see :func:`.constraint_matrix`.
    size : int
        Size of the assembly.
    tol : float, optional
        Tolerance to find the rank of the constraints. Each row is
        normalized to a unit norm, and a constraint is independent when its
        pivot in the QR decomposition is larger than ``tol``. A warning is
        issued when the smallest pivot kept is less than ``1e4`` times the
        largest pivot dropped. Default ``1.e-9``.

    Returns
    -------
    Z : scipy.sparse.csr_matrix
        Matrix with shape ``(size, size - rank)``, such that ``c = Z @ c_r``
        satisfies all the constraints for any ``c_r``.

    Notes
    -----
    The basis satisfies `\boldsymbol{B} \boldsymbol{Z} = \boldsymbol{0}` for all the constraints, and the
    matrices and vectors of the assembly are reduced as `\boldsymbol{K}_r = \boldsymbol{Z}^\top \boldsymbol{K}
    \boldsymbol{Z}` and `\boldsymbol{F}_r = \boldsymbol{Z}^\top \boldsymbol{F}`, see
    :mod:`panels.multidomain.connections.nullspace`. The independent
    constraints have pivots many orders of magnitude larger than the
    round-off errors left in the redundant ones, see
    :meth:`.Elimination.finish`. Redundant constraints happen e.g. at a
    corner shared by four domains, around a closed cylinder, or where a
    boundary condition flag already removes the term that a constraint would
    remove. The fifth and following functions of Bardell vanish together
    with their slope at the ends of the domain, so the constraints of an edge
    involve only the first four functions across it, and `\boldsymbol{Z}` has very few
    non-zero terms, an elimination similar to the assembly of the
    hierarchical elements of the p-version of the finite element method.

    """
    elim = Elimination(size, tol)
    for B in Bs:
        elim.add(B)
    return elim.finish()
