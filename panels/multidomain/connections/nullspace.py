r"""
Null-space method for the connections

Imposes the connections of a :class:`.MultiDomain` assembly exactly, with
the default ``conn_method='null-space'``, as the penalty kernels of :mod:`panels.multidomain.connections` would with penalty
constants going to infinity, see :ref:`null_space` for the theory, the usage
and a comparison with the penalty method.

Each penalty kernel adds the energy `\frac{1}{2} k_k \int (q_k^{(1)} -
q_k^{(2)})^2` for the quantities `q_k` that must be continuous across the
connection, e.g. `u, v, w` and the rotation of the normal. With Bardell's
polynomial functions `q_k^{(1)} - q_k^{(2)}` is a polynomial along the
connection, and it vanishes everywhere when it vanishes at as many
Gauss-Legendre points as its number of coefficients. The connections are
then the linear constraints:

.. math::

    [B] \{c\} = \{0\}

built by :func:`.constraint_matrix`, one row per quantity and point. They
are eliminated with a sparse basis `[T]` of the null space of `[B]`, built by
:func:`.null_space_basis`:

.. math::

    \{c\} = [T] \{c_r\}, \qquad [K_r] = [T]^T [K] [T], \qquad
    \{f_r\} = [T]^T \{f\}

and the reduced problem is solved with the solvers of ``structsolve``
unchanged, see :meth:`.MultiDomain.get_T`, :meth:`.MultiDomain.reduce`,
:meth:`.MultiDomain.expand` and :meth:`.MultiDomain.get_reduced_functions`.

The quantities of each connection are those of its penalty kernel, see
:ref:`null_space_quantities`. The connections available are listed in
:data:`NULL_SPACE_FUNCS`, the damaged connection ``'SB_TSL'`` is a physical
stiffness of the interface and remains a penalty stiffness.

"""
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import qr, solve_triangular
from scipy.sparse import coo_matrix, csr_matrix, identity

from panels import DOUBLE
import panels.modelDB as modelDB
from panels.bardell import calc_vec_f, calc_vec_fp
from panels.logger import warn
from . import kCsdt


#: Connections that can be imposed with the null-space method
NULL_SPACE_FUNCS = ('SSxcte', 'SSycte', 'BFxcte', 'BFycte', 'SB')

U, V, W, PHIX, PHIY = range(5)
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')


def _dofs(p):
    if p.model is None:
        return 3
    return modelDB.db[p.model]['dofs']


def _rinv_sanders(p):
    if p.model is not None and 'sanders' in p.model:
        return 1./p.r
    return 0.


def _length(p, direction):
    return p.a if direction == 'x' else p.b


def _num(p, direction):
    return p.m if direction == 'x' else p.n


def _eval(p, field, der, direction, xis):
    r"""Approximation functions of ``field`` along ``direction``, or their
    first derivative, at the natural coordinates ``xis``, with shape
    ``(len(xis), num)``"""
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


def _clpt_components(func, p1, p2, dsb):
    r"""Quantities of the connections between models with 3 DOFs, each one
    ``((p1, terms1), (p2, terms2))`` with ``terms`` a list of ``(field,
    der_x, der_y, coeff)``"""
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
    if func == 'SB':
        #NOTE u1 + dsb*w1,x = u2, v1 + dsb*w1,y = v2, w1 = w2, see kCSB.pyx
        return [((p1, [(U, 0, 0, 1.), (W, 1, 0, dsb)]), (p2, [(U, 0, 0, 1.)])),
                ((p1, [(V, 0, 0, 1.), (W, 0, 1, dsb)]), (p2, [(V, 0, 0, 1.)])),
                ((p1, [(W, 0, 0, 1.)]), (p2, [(W, 0, 0, 1.)]))]
    raise NotImplementedError(func)


def _sdt_components(func, p1, p2, kr_sb):
    r"""Quantities of the connections between models with 5 DOFs, see
    :func:`._clpt_components`"""
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
                ((p1, [(PHIX, 0, 0, 1.)]), (p2, [(PHIX, 0, 0, 1.)]))]
    if func == 'SB':
        tu1, tv1 = kCsdt._surface_terms(p1, -sum(p1.plyts)/2.)
        tu2, tv2 = kCsdt._surface_terms(p2, +sum(p2.plyts)/2.)
        comps = [((p1, tu1), (p2, tu2)), ((p1, tv1), (p2, tv2)),
                 ((p1, [(W, 0, 0, 1.)]), (p2, [(W, 0, 0, 1.)]))]
        if kr_sb:
            comps += [((p1, [(PHIX, 0, 0, 1.)]), (p2, [(PHIX, 0, 0, 1.)])),
                      ((p1, kCsdt._rotation_y(p1)), (p2, kCsdt._rotation_y(p2)))]
        return comps
    raise NotImplementedError(func)


def _points(p1, p2, direction, ctes):
    r"""Natural coordinates of the points of ``p1`` and ``p2`` along
    ``direction``: the constant coordinates of the connection, or the
    Gauss-Legendre points along it, as many as the coefficients of the
    polynomials along it"""
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
    r"""Rows of `[B]` of a single connection

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
        The constraints `[B] \{c\} = \{0\}`, with ``size`` columns.

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
    if dofs1 == 3:
        dsb = sum(p1.plyts)/2. + sum(p2.plyts)/2.
        comps = _clpt_components(func, p1, p2, dsb)
    else:
        comps = _sdt_components(func, p1, p2, connecti.get('kr', 0.))
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


def null_space_basis(Bs, size, tol=1.e-9):
    r"""Sparse basis `[T]` of the null space of the constraints

    Parameters
    ----------
    Bs : list of sparse matrices
        The constraints of each connection, see :func:`.constraint_matrix`.
    size : int
        Size of the assembly.
    tol : float, optional
        Tolerance to find the rank of the constraints. Each row is
        normalized to a unit norm, and a constraint is independent when its
        pivot in the QR decomposition is larger than ``tol``. The
        independent constraints have pivots many orders of magnitude larger
        than the round-off errors left in the redundant ones, and a warning
        is issued when this gap is smaller than ``1e4``.

    Returns
    -------
    T : scipy.sparse.csr_matrix
        Matrix with shape ``(size, size - rank)``, such that `\{c\} = [T]
        \{c_r\}` satisfies all the constraints for any `\{c_r\}`.

    """
    T = identity(size, dtype=DOUBLE, format='csr')
    min_kept = np.inf
    max_dropped = 0.
    for B in Bs:
        B = csr_matrix(B)
        #NOTE the rows are normalized only once, so that a redundant row,
        #     after written in the reduced Ritz constants, has only round-off
        #     errors that are compared with the absolute tolerance
        norms = np.sqrt(np.asarray(B.multiply(B).sum(axis=1)).ravel())
        B = csr_matrix(B.multiply(1./np.where(norms > 0, norms, 1.)[:, None]))
        Bt = (B @ T).toarray()
        J = np.flatnonzero(np.abs(Bt).max(axis=0, initial=0) > tol)
        if J.size == 0:
            continue
        _, R, P = qr(Bt[:, J], mode='economic', pivoting=True)
        d = np.abs(np.diag(R))
        rank = int(np.sum(d > tol))
        if rank < d.size:
            max_dropped = max(max_dropped, d[rank])
        if rank == 0:
            continue
        min_kept = min(min_kept, d[rank - 1])
        eliminated = J[P[:rank]]
        kept = J[P[rank:]]
        #NOTE R11 c_eliminated + R12 c_kept = 0
        X = -solve_triangular(R[:rank, :rank], R[:rank, rank:])
        X[np.abs(X) < 1e-14*np.abs(X).max(initial=0.)] = 0.
        nr = T.shape[1]
        remaining = np.setdiff1d(np.arange(nr), eliminated)
        new_index = -np.ones(nr, dtype=int)
        new_index[remaining] = np.arange(remaining.size)
        re, ce = np.nonzero(X)
        S = coo_matrix((np.concatenate((np.ones(remaining.size), X[re, ce])),
                        (np.concatenate((remaining, eliminated[re])),
                         np.concatenate((np.arange(remaining.size),
                                         new_index[kept[ce]])))),
                       shape=(nr, remaining.size))
        T = (T @ S.tocsr()).tocsr()
    if max_dropped > 0 and min_kept < 1e4*max_dropped:
        warn('Null-space method: the rank of the constraints is not well '
             'defined, the smallest independent pivot is {0:.1e} and the '
             'largest redundant one {1:.1e}, see the parameter "tol"'
             .format(min_kept, max_dropped))
    T.eliminate_zeros()
    return T
