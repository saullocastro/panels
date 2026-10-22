r"""
Connections of the models based on shear deformation theories

The plate and cylindrical shell models based on the first-order (FSDT) and
third-order (TSDT) shear deformation theories, e.g. ``'plate_fsdt_donnell'``
or ``'cylshell_tsdt_sanders'``, have 5 DOFs per term, `u, v, w, \phi_x,
\phi_y`. The penalty stiffness matrices below connect two domains of these
models:

- :func:`.fkCSSxcte_sdt` and :func:`.fkCSSycte_sdt`, between two skins along
  an edge, penalize the difference of `u, v, w` with ``kt`` and of the
  rotations of the normal at the reference surface, :func:`._rotation_x`
  and :func:`._rotation_y`, the latter with the term `v/r` of the
  Sanders-Koiter kinematics, with ``kr``. For the TSDT the derivative of
  `w` normal to the edge is penalized with ``kr`` too.
- :func:`.fkCSB_sdt`, between two skins connected over an area, penalizes
  with ``kt`` the difference of the displacements `u, v, w` at the
  interface, the lower face of the panel on the positive side of the
  interface along the common `z` axis, ``p_top`` (``p1`` of the
  connection), and the upper face of the panel on the negative side,
  ``p_bot`` (``p2``), see :meth:`.MultiDomain.get_kC_conn`. When ``kr`` is
  given the difference of the rotations is also penalized.

All the functions return the upper triangle of the connection matrix, with
the shape ``(size, size)`` of the assembly, as the Cython kernels of the
models with 3 DOFs, see :func:`structsolve.sparseutils.finalize_symmetric_matrix`.

The matrices are computed with the integrals of :ref:`Bardell's functions
<theory_func_bardell>`.

Notes
-----
The models have the displacement field

.. math::

    u(z) = u + z \phi_x - c_1 \left[(z - d)^3 + d^3\right] \left(\phi_x +
    w_{,x}\right) \qquad
    v(z) = v + z \Phi_y - c_1 \left[(z - d)^3 + d^3\right] \left(\phi_y +
    w_{,y}\right) \qquad
    w(z) = w

with `z` measured from the reference surface, `d` the offset of the
laminate, `c_1 = 0` for the first-order (FSDT) and `c_1 = 4/(3 h^2)` for
the third-order shear deformation theory (TSDT), see :mod:`panels.models`,
and where `\Phi_y = \phi_y + v/r` is the rotation of the normal about `x`
with the Sanders-Koiter kinematics, `\Phi_y = \phi_y` otherwise. The
penalty stiffness matrices enforce the compatibility between the
displacement fields of two domains, with the penalty energy `\frac{1}{2}
\sum_k k_k \int (q_k^{(1)} - q_k^{(2)})^2` of the quantities `q_k` of
:func:`._connection_matrix`:

- :func:`.fkCSSxcte_sdt` and :func:`.fkCSSycte_sdt` penalize the
  difference of `u, v, w` with ``kt`` and of the rotations of the normal
  at the reference surface `z = 0`, `\theta_x` and `\theta_y`, with
  ``kr``. For the TSDT the normal derivative of `w`, which also enters the
  displacement field, is penalized with ``kr`` too, see
  :func:`._edge_components`.
- :func:`.fkCSB_sdt` penalizes with ``kt`` the difference of the
  displacements `u(z), v(z), w(z)` at the interface, see
  :func:`._surface_terms`. The laminates may then rotate independently,
  like in a layerwise theory. When ``kr`` is given the difference of the
  rotations `\phi_x, \Phi_y` is also penalized, and for the FSDT two panels
  connected with a high ``kr`` behave like a single laminate with both
  stacking sequences, provided that the shear correction factor is a
  constant, e.g. ``Shell.fsdt_shear_correction = 5/6``; the default,
  ``'rohwer'``, depends on the stacking sequence of each laminate. The TSDT
  laminates have zero transverse shear strains at their faces, including
  the interface, and are therefore stiffer than a single TSDT laminate. The
  interface is the face of each laminate in its own coordinate `z`, at `z
  = d \mp h/2` for the TSDT; for the FSDT, as for the kernels of the models
  with 3 DOFs, the offsets are not considered and the faces are at `z =
  \mp h/2`.

.. rubric:: Rotations at the reference surface

The connected panels share the reference surface, `z = 0`, where `u` and
`v` are the displacements of both panels even when their offsets differ,
and the rotations tied by ``kr`` are those of the normal at the same `z =
0`, `\theta_x = \partial u/\partial z|_{z=0}` and `\theta_y = \partial
v/\partial z|_{z=0}`:

.. math::

    \theta_x = \phi_x - \frac{4 d^2}{h^2} \left(\phi_x + w_{,x}\right)
    \qquad
    \theta_y = \Phi_y - \frac{4 d^2}{h^2} \left(\phi_y + w_{,y}\right)

computed with the offset `d` and the thickness `h` of each panel. For the
FSDT, and for the TSDT with `d = 0`, `\theta_x = \phi_x` and `\theta_y =
\Phi_y`: the rotation of the FSDT is constant through the thickness. The
rotations `\phi_x`, `\phi_y` of the TSDT are those at the mid-surface,
`z = d`, which differs from panel to panel; when the reference surface is a
face, `d = \pm h/2`, the transverse shear vanishes there and `\theta_x =
-w_{,x}`, `\theta_y = \Phi_y - \phi_y - w_{,y}`, i.e. `-w_{,y}` plus the
term `v/r` of Sanders-Koiter.

The integrals of Bardell's functions are exact for the polynomial
approximations.

"""
import numpy as np
from scipy.sparse import coo_matrix, triu

from panels import INT, DOUBLE
from panels.bardell import (calc_vec_f, calc_vec_fp, calc_integral_ff,
        calc_integral_ffp, calc_integral_fpfp)


DOF = 5
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
U, V, W, PHIX, PHIY = range(DOF)


def _check_sdt(*panels):
    for p in panels:
        if p.model is None or not ('fsdt' in p.model or 'tsdt' in p.model):
            raise ValueError("Expected a model based on shear deformation "
                             "theories, got model '{0}'".format(p.model))


def _rinv_sanders(p):
    r"""`1/r` of a panel with the Sanders-Koiter kinematics, zero otherwise"""
    return 1./p.r if 'sanders' in p.model else 0.


def _rotation_y(p):
    r"""Terms of the rotation of the normal about `x` at the reference
    surface, `\theta_y = \partial v/\partial z|_{z=0}`, with the term `v/r`
    of the Sanders-Koiter kinematics

    Notes
    -----
    `\Phi_y = \phi_y + v/r` with the Sanders-Koiter kinematics, whose
    rotation of the normal about `x` includes the rigid-body rotation `v/r`
    of the cross-section, and `\Phi_y = \phi_y` otherwise, the term `v/r`
    being neglected with the Donnell kinematics and absent for plates. With
    the TSDT field, `\theta_y = \Phi_y - 3 c_1 d^2 (\phi_y + w_{,y}) = (1 -
    \alpha) \phi_y + v/r - \alpha w_{,y}`, see :func:`._rotation_x`.

    """
    a = _alpha(p)
    terms = [(PHIY, 0, 0, 1. - a)]
    rinv = _rinv_sanders(p)
    if rinv != 0:
        terms.append((V, 0, 0, rinv))
    if a != 0:
        terms.append((W, 0, 1, -a))
    return terms


def _is_tsdt(p):
    return 'tsdt' in p.model


def _alpha(p):
    r"""`4 d^2/h^2` of a TSDT panel, with its offset `d` and thickness `h`,
    zero for the FSDT, see :func:`._rotation_x`"""
    if not _is_tsdt(p):
        return 0.
    return 4.*p.offset**2/sum(p.plyts)**2


def _rotation_x(p):
    r"""Terms of the rotation of the normal about `y` at the reference
    surface, `\theta_x = \partial u/\partial z|_{z=0}`

    Notes
    -----
    With the TSDT field of :mod:`panels.multidomain.connections.kCsdt`,
    `\theta_x = \phi_x - 3 c_1 d^2 (\phi_x + w_{,x}) = (1 - \alpha) \phi_x -
    \alpha w_{,x}`, `\alpha = 4 d^2/h^2`, see :func:`._alpha`; `\theta_x =
    \phi_x` for the FSDT and for `d = 0`.

    """
    a = _alpha(p)
    terms = [(PHIX, 0, 0, 1. - a)]
    if a != 0:
        terms.append((W, 1, 0, -a))
    return terms


def _flags(p, field, direction):
    return tuple(float(getattr(p, direction + edge + FIELDS[field] + r))
                 for edge, r in (('1', ''), ('1', 'r'), ('2', ''), ('2', 'r')))


def _length(p, direction):
    return p.a if direction == 'x' else p.b


def _num(p, direction):
    return p.m if direction == 'x' else p.n


def _point(p, field, der, direction, coord):
    r"""Approximation functions of ``field``, or their first derivative,
    at the physical coordinate ``coord`` along ``direction``"""
    L = _length(p, direction)
    xi = 2*coord/L - 1.
    flags = _flags(p, field, direction)
    if der == 0:
        vec = calc_vec_f(xi, *flags)
    else:
        vec = calc_vec_fp(xi, *flags)*(2/L)
    return vec[:_num(p, direction)]


def _integral(pP, fieldP, derP, pQ, fieldQ, derQ, direction):
    r"""Integral along ``direction`` of the products of the approximation
    functions of ``pP`` and ``pQ``, or of their first derivatives, which must
    span the same length

    Notes
    -----
    With the natural coordinate `\xi = 2 x/L - 1` shared by the two panels,
    `\int_0^L f_P f_Q dx = \frac{L}{2} \int_{-1}^{1} f_P f_Q d\xi` and each
    derivative is `\partial/\partial x = (2/L) \partial/\partial \xi`. The
    integrals along `\xi` are the closed-form integrals of Bardell's
    functions, ``calc_integral_ff``, ``calc_integral_ffp`` and
    ``calc_integral_fpfp`` of :mod:`panels.bardell`.

    """
    LP = _length(pP, direction)
    LQ = _length(pQ, direction)
    numP = _num(pP, direction)
    numQ = _num(pQ, direction)
    flP = _flags(pP, fieldP, direction)
    flQ = _flags(pQ, fieldQ, direction)
    out = np.zeros((numP, numQ), dtype=DOUBLE)
    for i in range(numP):
        for k in range(numQ):
            if derP == 0 and derQ == 0:
                out[i, k] = calc_integral_ff(i, k, *flP, *flQ)
            elif derP == 0 and derQ == 1:
                out[i, k] = calc_integral_ffp(i, k, *flP, *flQ)
            elif derP == 1 and derQ == 0:
                out[i, k] = calc_integral_ffp(k, i, *flQ, *flP)
            else:
                out[i, k] = calc_integral_fpfp(i, k, *flP, *flQ)
    return out*(LP/2)*(2/LP)**derP*(2/LQ)**derQ


def _factor(pP, termP, cteP, pQ, termQ, cteQ, direction):
    r"""Factor of the term pair along ``direction``

    The constant coordinates ``cteP`` and ``cteQ`` of the connection in
    ``pP`` and ``pQ`` are ``None`` for an integral along ``direction``.

    """
    fieldP, derP = termP[0], termP[1 if direction == 'x' else 2]
    fieldQ, derQ = termQ[0], termQ[1 if direction == 'x' else 2]
    if cteP is None:
        return _integral(pP, fieldP, derP, pQ, fieldQ, derQ, direction)
    return np.outer(_point(pP, fieldP, derP, direction, cteP),
                    _point(pQ, fieldQ, derQ, direction, cteQ))


def _connection_matrix(components, ctes, size):
    r"""Penalty stiffness matrix of a connection

    Each penalized quantity of a panel is a linear combination of terms
    ``(field, der_x, der_y, coeff)``, with ``der_x`` and ``der_y`` the orders
    of the derivatives along `x` and `y`. The products of the terms of the
    two panels are integrated along the directions where ``ctes`` is
    ``None`` and evaluated at the constant coordinates otherwise, with a
    positive sign within a panel and a negative sign between the two
    panels. A penalty equal to zero adds no term. The upper triangle of the
    matrix is returned, a :class:`scipy.sparse.coo_matrix` with the shape
    ``(size, size)``.

    Parameters
    ----------
    components : list
        Each element is ``(penalty, ((p1, terms1), (p2, terms2)))``.
    ctes : dict
        With keys ``'x'`` and ``'y'``, each either ``None`` for an integral
        along that direction or a tuple with the constant coordinates of the
        connection in ``p1`` and ``p2``.
    size : int
        Size of the assembly.

    Notes
    -----
    The penalty energy is `\frac{1}{2} \sum_k k_k \int (q_k^1 - q_k^2)^2`,
    where each quantity `q_k^p` of panel `p` is a linear combination of terms
    ``(field, der_x, der_y, coeff)``, `q_k^p = \sum_t \alpha_t \{N_t\}
    \{c^p\}`, where `\alpha_t` is ``coeff``, `\{N_t\}` the derivative of
    order ``der_x`` along `x` and ``der_y`` along `y` of the approximation
    functions of ``field`` and `\{c^p\}` the Ritz constants of panel `p`.
    The second derivatives of the energy with respect to `\{c^P\}` and
    `\{c^Q\}` give the blocks

    .. math::

        [K_{PQ}] = \pm k_k \sum_{t \in P} \sum_{s \in Q} \alpha_t \alpha_s
        \int \{N_t\}^T \{N_s\}

    with the sign `+` for `P = Q` and `-` for `P \neq Q`. The approximation
    functions are products `f(x) g(y)`, so each integral is the Kronecker
    product of the factors along `x` and `y` of :func:`._factor`, the
    integrals of :func:`._integral` or the products of the functions at the
    constant coordinates of the connection.

    """
    rows = []
    cols = []
    vals = []
    ctes = {d: (None, None) if c is None else c for d, c in ctes.items()}
    for penalty, sides in components:
        if penalty == 0:
            continue
        for iP, (pP, termsP) in enumerate(sides):
            for iQ, (pQ, termsQ) in enumerate(sides):
                sign = 1 if iP == iQ else -1
                for termP in termsP:
                    for termQ in termsQ:
                        coeff = sign*penalty*termP[3]*termQ[3]
                        if coeff == 0:
                            continue
                        X = _factor(pP, termP, ctes['x'][iP],
                                    pQ, termQ, ctes['x'][iQ], 'x')
                        Y = _factor(pP, termP, ctes['y'][iP],
                                    pQ, termQ, ctes['y'][iQ], 'y')
                        #NOTE Ritz constant of term (i, j) at DOF*(j*m + i)
                        block = coeff*np.kron(Y, X)
                        r = pP.row_start + DOF*np.arange(block.shape[0]) + termP[0]
                        c = pQ.col_start + DOF*np.arange(block.shape[1]) + termQ[0]
                        R, C = np.meshgrid(r, c, indexing='ij')
                        rows.append(R.ravel())
                        cols.append(C.ravel())
                        vals.append(block.ravel())
    if not rows:
        return coo_matrix((size, size), dtype=DOUBLE)
    kC = coo_matrix((np.concatenate(vals), (np.concatenate(rows).astype(INT),
                     np.concatenate(cols).astype(INT))), shape=(size, size))
    kC.sum_duplicates()
    return triu(kC, format='coo')


def _edge_components(kt, kr, p1, p2, normal):
    r"""Components penalized along an edge, ``normal`` is the direction of
    the derivative of `w` that enters the TSDT displacement field

    The differences of `u, v, w` are penalized with ``kt`` and those of the
    rotations at the reference surface of :func:`._rotation_x` and
    :func:`._rotation_y` with ``kr``; when ``p1`` or ``p2`` is a TSDT
    model, the difference of the derivative of `w` along ``normal`` is
    penalized with ``kr`` too.

    Notes
    -----
    The TSDT displacement field contains `w_{,x}` in `u(z)` and `w_{,y}` in
    `v(z)`, see :mod:`panels.multidomain.connections.kCsdt`. Along the edge,
    the tangential derivative of `w` is continuous when `w` is, so only the
    derivative normal to the edge needs to be penalized for the continuity
    of `u(z)` and `v(z)` through the thickness.

    """
    comps = [
        (kt, ((p1, [(U, 0, 0, 1.)]), (p2, [(U, 0, 0, 1.)]))),
        (kt, ((p1, [(V, 0, 0, 1.)]), (p2, [(V, 0, 0, 1.)]))),
        (kt, ((p1, [(W, 0, 0, 1.)]), (p2, [(W, 0, 0, 1.)]))),
        (kr, ((p1, _rotation_x(p1)), (p2, _rotation_x(p2)))),
        (kr, ((p1, _rotation_y(p1)), (p2, _rotation_y(p2)))),
        ]
    if _is_tsdt(p1) or _is_tsdt(p2):
        dx, dy = (1, 0) if normal == 'x' else (0, 1)
        comps.append((kr, ((p1, [(W, dx, dy, 1.)]), (p2, [(W, dx, dy, 1.)]))))
    return comps


def fkCSSxcte_sdt(kt, kr, p1, p2, xcte1, xcte2, size):
    r"""Skin-skin connection along `x_1 = x_{cte1}` and `x_2 = x_{cte2}`

    The differences of `u, v, w` between the two panels are penalized with
    ``kt``, those of the rotations of the normal at the reference surface,
    :func:`._rotation_x` and :func:`._rotation_y`, with ``kr`` and, when one
    of the panels is a TSDT model, that of `w_{,x}` with ``kr`` too. The
    products of the approximation functions are integrated along `y` and
    evaluated at ``xcte1`` in ``p1`` and ``xcte2`` in ``p2``. The terms are
    placed at ``row_start`` and ``col_start`` of each panel. A penalty
    equal to zero adds no term. A ``ValueError`` is raised by
    ``_check_sdt`` when a panel is not a model based on the FSDT or TSDT.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1, p2 : :class:`.Shell`
        Panels being connected, with the same width ``b``, models based on
        the FSDT or TSDT.
    xcte1, xcte2 : float
        Coordinates of the connection in each panel, length.
    size : int
        Size of the assembly.

    Returns
    -------
    kCSSxcte : scipy.sparse.coo_matrix
        Upper triangle of the connection matrix, shape ``(size, size)``.

    Notes
    -----
    The penalty energy is

    .. math::

        U = \frac{1}{2} \int_y k_t \left[ (u_1 - u_2)^2 + (v_1 - v_2)^2 +
            (w_1 - w_2)^2 \right] + k_r \left[ (\theta_{x1} - \theta_{x2})^2 +
            (\theta_{y1} - \theta_{y2})^2 + (w_{1,x} - w_{2,x})^2 \right] dy

    with the fields of ``p1`` at `x_1 = x_{cte1}` and those of ``p2`` at
    `x_2 = x_{cte2}`, the rotations at the reference surface `\theta_x` and
    `\theta_y` of :func:`._rotation_x` and :func:`._rotation_y`, computed
    with the offset and the thickness of each panel, and the term `w_{,x}`
    only when one of the panels is a TSDT model, see
    :func:`._edge_components`. The displacements `u, v, w` and the rotations
    `\theta_x, \theta_y` are those of the common reference surface, `z =
    0`, also for panels with different offsets.

    """
    _check_sdt(p1, p2)
    comps = _edge_components(kt, kr, p1, p2, 'x')
    return _connection_matrix(comps, dict(x=(xcte1, xcte2), y=None), size)


def fkCSSycte_sdt(kt, kr, p1, p2, ycte1, ycte2, size):
    r"""Skin-skin connection along `y_1 = y_{cte1}` and `y_2 = y_{cte2}`

    The differences of `u, v, w` between the two panels are penalized with
    ``kt``, those of the rotations of the normal at the reference surface,
    :func:`._rotation_x` and :func:`._rotation_y`, with ``kr`` and, when one
    of the panels is a TSDT model, that of `w_{,y}` with ``kr`` too. The
    products of the approximation functions are integrated along `x` and
    evaluated at ``ycte1`` in ``p1`` and ``ycte2`` in ``p2``. The terms are
    placed at ``row_start`` and ``col_start`` of each panel. A penalty
    equal to zero adds no term. A ``ValueError`` is raised by
    ``_check_sdt`` when a panel is not a model based on the FSDT or TSDT.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1, p2 : :class:`.Shell`
        Panels being connected, with the same length ``a``, models based on
        the FSDT or TSDT.
    ycte1, ycte2 : float
        Coordinates of the connection in each panel, length.
    size : int
        Size of the assembly.

    Returns
    -------
    kCSSycte : scipy.sparse.coo_matrix
        Upper triangle of the connection matrix, shape ``(size, size)``.

    Notes
    -----
    The penalty energy is that of :func:`.fkCSSxcte_sdt`, integrated along
    `x` with the fields of ``p1`` at `y_1 = y_{cte1}` and those of ``p2`` at
    `y_2 = y_{cte2}`, and with `w_{,y}` in place of `w_{,x}`.

    """
    _check_sdt(p1, p2)
    comps = _edge_components(kt, kr, p1, p2, 'y')
    return _connection_matrix(comps, dict(x=None, y=(ycte1, ycte2)), size)


def _surface_terms(p, z):
    r"""Terms of `u(z)` and `v(z)` of panel ``p`` at the coordinate `z`
    measured from its reference surface

    Notes
    -----
    From the displacement field of :mod:`panels.multidomain.connections.kCsdt`:

    .. math::

        u(z) = u + (z - c_1 g) \phi_x - c_1 g w_{,x}, \qquad
        v(z) = (1 + z/r) v + (z - c_1 g) \phi_y - c_1 g w_{,y}

    with `g = (z - d)^3 + d^3`, the offset `d` of ``p``, `c_1 = 4/(3 h^2)`
    for the TSDT, `h` being the sum of the ply thicknesses of ``p``, `c_1 =
    0` for the FSDT, and the term `z v/r` only with the Sanders-Koiter
    kinematics.

    """
    c1 = 4/(3*sum(p.plyts)**2) if _is_tsdt(p) else 0.
    d = p.offset if _is_tsdt(p) else 0.
    g = (z - d)**3 + d**3
    cphi = z - c1*g
    cw = -c1*g
    tu = [(U, 0, 0, 1.), (PHIX, 0, 0, cphi), (W, 1, 0, cw)]
    #NOTE v(z) = v + z*(phiy + v/r) - c1*z**3*(phiy + w,y) with Sanders
    tv = [(V, 0, 0, 1. + z*_rinv_sanders(p)), (PHIY, 0, 0, cphi), (W, 0, 1, cw)]
    return tu, tv


def _face(p, side):
    r"""Coordinate `z`, from the reference surface of ``p``, of its lower
    face, ``side = -1``, or upper face, ``side = +1``

    `z = d + side \, h/2` for the TSDT, with the offset `d` of ``p``, and
    `z = side \, h/2` for the FSDT, whose offset is not considered, as for
    the models with 3 DOFs.

    """
    d = p.offset if _is_tsdt(p) else 0.
    return d + side*sum(p.plyts)/2.


def fkCSB_sdt(kt, p_top, p_bot, size, kr=0.):
    r"""Skin-base connection over the area of two panels

    The differences of the in-plane displacements `u` and `v` between the
    lower face of ``p_top`` and the upper face of ``p_bot``, see
    :func:`._face`, and the difference of `w` are penalized with ``kt``; the
    differences of the rotations at the reference surface of
    :func:`._rotation_x` and :func:`._rotation_y`, with the term `v/r` of
    the Sanders-Koiter kinematics, with ``kr``. The products of the
    approximation functions
    are integrated along `x` and `y`. The terms are placed at ``row_start``
    and ``col_start`` of each panel. A penalty equal to zero adds no term. A
    ``ValueError`` is raised by ``_check_sdt`` when a panel is not a model
    based on the FSDT or TSDT.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, per unit area, with units of
        force/length**3.
    p_top, p_bot : :class:`.Shell`
        Panels on the positive and negative sides of the interface along
        their common `z` axis, ``p1`` and ``p2`` of the connection, covering
        the same area, models based on the FSDT or TSDT.
    size : int
        Size of the assembly.
    kr : float, optional
        Rotation penalty stiffness, per unit area, with units of
        force/length, for instance ``kt*h**2`` with ``h`` the thickness of the
        laminates. With ``kr = 0`` only the displacements at the interface
        are connected. Default ``0.``.

    Returns
    -------
    kCSB : scipy.sparse.coo_matrix
        Upper triangle of the connection matrix, shape ``(size, size)``.

    Notes
    -----
    The penalty energy, integrated over the area of the connection, is

    .. math::

        U = \frac{1}{2} \int_x \int_y k_t \left[ (u_t - u_b)^2 + (v_t -
            v_b)^2 + (w_t - w_b)^2 \right] + k_r \left[ (\theta_{xt} -
            \theta_{xb})^2 + (\theta_{yt} - \theta_{yb})^2 \right] dy dx

    where `u_t, v_t` are `u(z), v(z)` of ``p_top`` at its lower face and
    `u_b, v_b` those of ``p_bot`` at its upper face, see :func:`._face` and
    :func:`._surface_terms`, and `\theta_x`, `\theta_y` are given by
    :func:`._rotation_x` and :func:`._rotation_y`.
    With ``kr = 0`` the two laminates may rotate independently at the
    interface, see :mod:`panels.multidomain.connections.kCsdt`.

    """
    _check_sdt(p_top, p_bot)
    tu_top, tv_top = _surface_terms(p_top, _face(p_top, -1))
    tu_bot, tv_bot = _surface_terms(p_bot, _face(p_bot, +1))
    comps = [
        (kt, ((p_top, tu_top), (p_bot, tu_bot))),
        (kt, ((p_top, tv_top), (p_bot, tv_bot))),
        (kt, ((p_top, [(W, 0, 0, 1.)]), (p_bot, [(W, 0, 0, 1.)]))),
        (kr, ((p_top, _rotation_x(p_top)), (p_bot, _rotation_x(p_bot)))),
        (kr, ((p_top, _rotation_y(p_top)), (p_bot, _rotation_y(p_bot)))),
        ]
    return _connection_matrix(comps, dict(x=None, y=None), size)
