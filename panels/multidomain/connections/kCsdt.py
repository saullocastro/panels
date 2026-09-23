r"""
Connections of the models based on shear deformation theories

The models ``'plate_fsdt_donnell'`` and ``'plate_tsdt_donnell'`` have 5 DOFs
per term, `u, v, w, \phi_x, \phi_y`, and the displacement field

.. math::

    u(z) = u + z \phi_x - c_1 z^3 \left(\phi_x + w_{,x}\right) \qquad
    v(z) = v + z \phi_y - c_1 z^3 \left(\phi_y + w_{,y}\right) \qquad
    w(z) = w

with `c_1 = 0` for the first-order (FSDT) and `c_1 = 4/(3 h^2)` for the
third-order shear deformation theory (TSDT). The penalty stiffness matrices
below enforce the compatibility between the displacement fields of two
domains:

- :func:`.fkCSSxcte_sdt` and :func:`.fkCSSycte_sdt`, between two skins along
  an edge, penalize the difference of `u, v, w` with ``kt`` and of
  `\phi_x, \phi_y` with ``kr``. For the TSDT the normal derivative of `w`,
  which also enters the displacement field, is penalized with ``kr`` too.
- :func:`.fkCSB_sdt`, between two skins connected over an area, penalizes
  with ``kt`` the difference of the displacements `u(z), v(z), w(z)` at the
  interface, where the top panel has `z = -h_{top}/2` and the bottom panel
  `z = +h_{bot}/2`. The laminates may then rotate independently, like in a
  layerwise theory. When ``kr`` is given the difference of the rotations
  `\phi_x, \phi_y` is also penalized, and for the FSDT two panels connected
  with a high ``kr`` behave like a single laminate with both stacking
  sequences, provided that the shear correction factor is a constant, e.g.
  the default ``Shell.fsdt_shear_correction = 5/6``. The TSDT laminates have
  zero transverse shear strains at their faces, including the interface,
  and are therefore stiffer than a single TSDT laminate. The offsets of the
  laminates are not considered, as in the kernels of the models with 3 DOFs.

All the functions return the upper triangle of the connection matrix, with
the shape ``(size, size)`` of the assembly, as the Cython kernels of the
models with 3 DOFs, see :func:`structsolve.sparseutils.finalize_symmetric_matrix`.

The matrices are computed with the integrals of :ref:`Bardell's functions
<theory_func_bardell>`, which are exact for the polynomial approximations.

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
        if p.model not in ('plate_fsdt_donnell', 'plate_tsdt_donnell'):
            raise ValueError("Expected a model based on shear deformation "
                             "theories, got model '{0}'".format(p.model))


def _is_tsdt(p):
    return 'tsdt' in p.model


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
    span the same length"""
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

    The penalty energy is `\frac{1}{2} \sum_k k_k \int (q_k^1 - q_k^2)^2`,
    where each quantity `q_k^p` of panel `p` is a linear combination of terms
    ``(field, der_x, der_y, coeff)``, with ``der_x`` and ``der_y`` the orders
    of the derivatives along `x` and `y`.

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
    the derivative of `w` that enters the TSDT displacement field"""
    comps = [
        (kt, ((p1, [(U, 0, 0, 1.)]), (p2, [(U, 0, 0, 1.)]))),
        (kt, ((p1, [(V, 0, 0, 1.)]), (p2, [(V, 0, 0, 1.)]))),
        (kt, ((p1, [(W, 0, 0, 1.)]), (p2, [(W, 0, 0, 1.)]))),
        (kr, ((p1, [(PHIX, 0, 0, 1.)]), (p2, [(PHIX, 0, 0, 1.)]))),
        (kr, ((p1, [(PHIY, 0, 0, 1.)]), (p2, [(PHIY, 0, 0, 1.)]))),
        ]
    if _is_tsdt(p1) or _is_tsdt(p2):
        dx, dy = (1, 0) if normal == 'x' else (0, 1)
        comps.append((kr, ((p1, [(W, dx, dy, 1.)]), (p2, [(W, dx, dy, 1.)]))))
    return comps


def fkCSSxcte_sdt(kt, kr, p1, p2, xcte1, xcte2, size):
    r"""Skin-skin connection along `x_1 = x_{cte1}` and `x_2 = x_{cte2}`

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1, p2 : :class:`.Shell`
        Panels being connected, with the same width ``b``.
    xcte1, xcte2 : float
        Coordinates of the connection in each panel.
    size : int
        Size of the assembly.

    Returns
    -------
    kCSSxcte : scipy.sparse.coo_matrix
        Upper triangle of the connection matrix.

    """
    _check_sdt(p1, p2)
    comps = _edge_components(kt, kr, p1, p2, 'x')
    return _connection_matrix(comps, dict(x=(xcte1, xcte2), y=None), size)


def fkCSSycte_sdt(kt, kr, p1, p2, ycte1, ycte2, size):
    r"""Skin-skin connection along `y_1 = y_{cte1}` and `y_2 = y_{cte2}`

    Parameters
    ----------
    kt : float
        Translation penalty stiffness.
    kr : float
        Rotation penalty stiffness.
    p1, p2 : :class:`.Shell`
        Panels being connected, with the same length ``a``.
    ycte1, ycte2 : float
        Coordinates of the connection in each panel.
    size : int
        Size of the assembly.

    Returns
    -------
    kCSSycte : scipy.sparse.coo_matrix
        Upper triangle of the connection matrix.

    """
    _check_sdt(p1, p2)
    comps = _edge_components(kt, kr, p1, p2, 'y')
    return _connection_matrix(comps, dict(x=None, y=(ycte1, ycte2)), size)


def _surface_terms(p, z):
    r"""Terms of `u(z)` and `v(z)` of panel ``p`` at the distance `z` from
    its mid-surface"""
    c1 = 4/(3*sum(p.plyts)**2) if _is_tsdt(p) else 0.
    cphi = z - c1*z**3
    cw = -c1*z**3
    tu = [(U, 0, 0, 1.), (PHIX, 0, 0, cphi), (W, 1, 0, cw)]
    tv = [(V, 0, 0, 1.), (PHIY, 0, 0, cphi), (W, 0, 1, cw)]
    return tu, tv


def fkCSB_sdt(kt, p_top, p_bot, size, kr=0.):
    r"""Skin-base connection over the area of two panels

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, per unit area.
    p_top, p_bot : :class:`.Shell`
        Top and bottom panels, covering the same area.
    size : int
        Size of the assembly.
    kr : float, optional
        Rotation penalty stiffness, per unit area, with units of
        force/length, for instance ``kt*h**2`` with ``h`` the thickness of the
        laminates. With ``kr = 0`` only the displacements at the interface
        are connected.

    Returns
    -------
    kCSB : scipy.sparse.coo_matrix
        Upper triangle of the connection matrix.

    """
    _check_sdt(p_top, p_bot)
    tu_top, tv_top = _surface_terms(p_top, -sum(p_top.plyts)/2.)
    tu_bot, tv_bot = _surface_terms(p_bot, +sum(p_bot.plyts)/2.)
    comps = [
        (kt, ((p_top, tu_top), (p_bot, tu_bot))),
        (kt, ((p_top, tv_top), (p_bot, tv_bot))),
        (kt, ((p_top, [(W, 0, 0, 1.)]), (p_bot, [(W, 0, 0, 1.)]))),
        (kr, ((p_top, [(PHIX, 0, 0, 1.)]), (p_bot, [(PHIX, 0, 0, 1.)]))),
        (kr, ((p_top, [(PHIY, 0, 0, 1.)]), (p_bot, [(PHIY, 0, 0, 1.)]))),
        ]
    return _connection_matrix(comps, dict(x=None, y=None), size)
