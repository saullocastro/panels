r"""
Skin-base connection ``'SB'`` of every model

Penalty stiffness matrix of the connection ``'SB'`` between two panels
covering the same area, for the models based on the classical laminated
plate theory (CLPT), with 3 DOFs per term, and on the shear deformation
theories (FSDT and TSDT), with 5 DOFs per term, Donnell or Sanders-Koiter,
see :func:`.fkCSB`. The two panels share the axes `x, y, z`, `z` being
normal to their reference surfaces; ``p_top``, the panel ``p1`` of the
connection, is on the positive side of the interface along `z`, and
``p_bot``, the panel ``p2``, on the negative side, see
:meth:`.MultiDomain.get_kC_conn`.

The connection imposes, at the interface,

.. math::

    u_1(z_1) = u_2(z_2), \qquad v_1(z_1) = v_2(z_2), \qquad w_1 = w_2

with `z_1 = d_1 - h_1/2` the lower face of ``p1`` and `z_2 = d_2 + h_2/2`
the upper face of ``p2``, each measured from the reference surface of its
panel, with the offsets `d_1`, `d_2` and the thicknesses `h_1`, `h_2`, see
:func:`.kCsdt._face`. Each panel has its own kinematics, see
:func:`.kCsdt._surface_terms`: `u(z) = u - z w_{,x}` and `v(z) = (1 +
z/r) v - z w_{,y}` for the CLPT, with the slope of each panel, and the
fields of the FSDT and TSDT with the rotations `\phi_x, \phi_y`, the term
`z v/r` being present only with the Sanders-Koiter kinematics. For the FSDT
and TSDT, with the key ``'kr'`` of the connection, the rotations of the
normal at the reference surfaces, :func:`.kCsdt._rotation_x` and
:func:`.kCsdt._rotation_y`, are also tied; the rotations of the CLPT follow
from `w`, tied by `w_1 = w_2`, see :func:`.kCsdt._sb_components`.

The same quantities, :func:`.kCsdt._sb_components`, give the constraints
of the null-space method, :func:`.nullspace.constraint_matrix`, and the
same displacement jump is the separation of the cohesive connection
``'SB_TSL'``, see :meth:`.MultiDomain._tsl_operators`.

The functions ``fkCSB11``, ``fkCSB12`` and ``fkCSB22`` of earlier versions,
hand-written kernels of the blocks of the CLPT matrix with the form
`u_1 + d_{sb} w_{1,x} = u_2` of the thesis of D'Souza (2024), which uses the
slope of ``p1`` only, are deprecated: they return the blocks of
:func:`.fkCSB` and ignore their distance arguments.

"""
import warnings

import numpy as np

from scipy.sparse import coo_matrix, triu

from panels import INT, DOUBLE
from . import kCsdt


def fkCSB(kt, p_top, p_bot, size, kr=0.):
    r"""Skin-base connection over the area of two panels, every model

    The differences of the in-plane displacements `u(z)` and `v(z)` between
    the lower face of ``p_top`` and the upper face of ``p_bot``, see
    :func:`.kCsdt._face` and :func:`.kCsdt._surface_terms`, and the
    difference of `w` are penalized with ``kt``; for the FSDT and TSDT,
    when ``kr`` is not zero, the differences of the rotations of the normal
    at the reference surface, :func:`.kCsdt._rotation_x` and
    :func:`.kCsdt._rotation_y`, with ``kr``.
    The products of the approximation functions are integrated exactly
    along `x` and `y`. The terms are placed at ``row_start`` and
    ``col_start`` of each panel, with the number of DOFs per term of their
    model.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, per unit area, with units of
        force/length**3, e.g. from :func:`.calc_kt_kr` with
        ``'bot-top'``.
    p_top, p_bot : :class:`.Shell`
        Panels on the positive and negative sides of the interface along
        their common `z` axis, ``p1`` and ``p2`` of the connection,
        covering the same area, with models of the same number of DOFs per
        term.
    size : int
        Size of the assembly.
    kr : float, optional
        Rotation penalty stiffness, per unit area, with units of
        force/length, used only by the models with 5 DOFs per term. Default
        ``0.``, only the displacements at the interface are connected.

    Returns
    -------
    kCSB : scipy.sparse.coo_matrix
        Upper triangle of the connection matrix, shape ``(size, size)``.

    Notes
    -----
    The penalty energy, integrated over the area of the connection, is

    .. math::

        U = \frac{1}{2} \int_x \int_y k_t \left[ (u_1(z_1) - u_2(z_2))^2
            + (v_1(z_1) - v_2(z_2))^2 + (w_1 - w_2)^2 \right] + k_r \left[
            (\theta_{x1} - \theta_{x2})^2 + (\theta_{y1} - \theta_{y2})^2
            \right] dy dx

    with `z_1 = d_1 - h_1/2` and `z_2 = d_2 + h_2/2`, see
    :mod:`panels.multidomain.connections.skin_base`. Each quantity is a
    linear combination `\{B\} \{c\}` of the Ritz constants, and the
    matrix is `k \int \{B\}^T \{B\}` for each quantity, see
    :func:`.kCsdt._connection_matrix`. For the CLPT,

    .. math::

        u_1(z_1) - u_2(z_2) = u_1 - z_1 w_{1,x} - u_2 + z_2 w_{2,x}, \qquad
        v_1(z_1) - v_2(z_2) = \left(1 + \frac{z_1}{r_1}\right) v_1 - z_1
        w_{1,y} - \left(1 + \frac{z_2}{r_2}\right) v_2 + z_2 w_{2,y}

    with `1/r = 0` for the Donnell kinematics and for plates. The term with
    `k_r` is present only for the FSDT and TSDT.

    """
    comps = kCsdt._sb_components(p_top, p_bot, kt, kr)
    return kCsdt._connection_matrix(comps, dict(x=None, y=None), size)


def _block(kt, p_top, p_bot, iP, iQ, size, row0, col0, upper):
    r"""Block ``(iP, iQ)`` of :func:`.fkCSB` placed at ``row0``, ``col0``,
    ``iP`` and ``iQ`` being ``0`` for ``p_top`` and ``1`` for ``p_bot``"""
    rows, cols, vals = [], [], []
    for penalty, sides in kCsdt._sb_components(p_top, p_bot, kt):
        pP, termsP = sides[iP]
        pQ, termsQ = sides[iQ]
        sign = 1 if iP == iQ else -1
        for termP in termsP:
            for termQ in termsQ:
                coeff = sign*penalty*termP[3]*termQ[3]
                if coeff == 0:
                    continue
                X = kCsdt._factor(pP, termP, None, pQ, termQ, None, 'x')
                Y = kCsdt._factor(pP, termP, None, pQ, termQ, None, 'y')
                block = coeff*np.kron(Y, X)
                r = row0 + kCsdt._dofs(pP)*np.arange(block.shape[0]) + termP[0]
                c = col0 + kCsdt._dofs(pQ)*np.arange(block.shape[1]) + termQ[0]
                R, C = np.meshgrid(r, c, indexing='ij')
                rows.append(R.ravel())
                cols.append(C.ravel())
                vals.append(block.ravel())
    kC = coo_matrix((np.concatenate(vals), (np.concatenate(rows).astype(INT),
                     np.concatenate(cols).astype(INT))), shape=(size, size),
                    dtype=DOUBLE)
    kC.sum_duplicates()
    return triu(kC, format='coo') if upper else kC


def _deprecated(name):
    warnings.warn('{0}() is deprecated and will be removed, use fkCSB(kt, '
                  'p_top, p_bot, size), which builds the whole matrix of the '
                  'connection; the distance arguments are ignored, the '
                  'interface being the faces of the panels'.format(name),
                  DeprecationWarning, stacklevel=3)


def fkCSB11(kt, dsb, p1, size, row0, col0, dt=0., rinv1=0.):
    r"""Deprecated, the block ``p1``-``p1`` of :func:`.fkCSB`

    Returns the upper triangle of the block of ``p1`` of the matrix of
    :func:`.fkCSB`, placed at ``row0`` and ``col0``, with the interface at
    the lower face of ``p1`` and its own kinematics. The arguments ``dsb``,
    ``dt`` and ``rinv1`` are ignored. A ``DeprecationWarning`` is issued.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force/length**3.
    dsb, dt, rinv1 : float
        Ignored.
    p1 : :class:`.Shell`
        Panel on the positive side of the interface along `z`.
    size : int
        Size of the assembly.
    row0, col0 : int
        Position of the block.

    Returns
    -------
    kCSB11 : scipy.sparse.coo_matrix
        Shape ``(size, size)``.

    """
    _deprecated('fkCSB11')
    return _block(kt, p1, p1, 0, 0, size, row0, col0, upper=True)


def fkCSB12(kt, dsb, p1, p2, size, row0, col0, dt=0., db=0., rinv1=0.,
            rinv2=0.):
    r"""Deprecated, the block ``p1``-``p2`` of :func:`.fkCSB`

    Returns the whole block coupling ``p1`` and ``p2`` of the matrix of
    :func:`.fkCSB`, placed at ``row0`` and ``col0``. The arguments ``dsb``,
    ``dt``, ``db``, ``rinv1`` and ``rinv2`` are ignored. A
    ``DeprecationWarning`` is issued.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force/length**3.
    dsb, dt, db, rinv1, rinv2 : float
        Ignored.
    p1, p2 : :class:`.Shell`
        Panels on the positive and negative sides of the interface along
        `z`.
    size : int
        Size of the assembly.
    row0, col0 : int
        Position of the block.

    Returns
    -------
    kCSB12 : scipy.sparse.coo_matrix
        Shape ``(size, size)``.

    """
    _deprecated('fkCSB12')
    return _block(kt, p1, p2, 0, 1, size, row0, col0, upper=False)


def fkCSB22(kt, p1, p2, size, row0, col0, db=0., rinv2=0.):
    r"""Deprecated, the block ``p2``-``p2`` of :func:`.fkCSB`

    Returns the upper triangle of the block of ``p2`` of the matrix of
    :func:`.fkCSB`, placed at ``row0`` and ``col0``, with the interface at
    the upper face of ``p2`` and its own kinematics. The arguments ``db``
    and ``rinv2`` are ignored. A ``DeprecationWarning`` is issued.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force/length**3.
    p1, p2 : :class:`.Shell`
        Panels on the positive and negative sides of the interface along
        `z`.
    db, rinv2 : float
        Ignored.
    size : int
        Size of the assembly.
    row0, col0 : int
        Position of the block.

    Returns
    -------
    kCSB22 : scipy.sparse.coo_matrix
        Shape ``(size, size)``.

    """
    _deprecated('fkCSB22')
    return _block(kt, p1, p2, 1, 1, size, row0, col0, upper=True)
