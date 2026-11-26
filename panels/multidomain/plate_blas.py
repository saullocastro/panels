r"""
Plate kernels as tabulated operators and matrix products
(:mod:`panels.multidomain.plate_blas`)

.. currentmodule:: panels.multidomain.plate_blas

The tangent stiffness matrices, the internal force vector and the product of
the tangent by a vector of the domains of the models ``'plate_clpt_donnell'``
and ``'plate_fsdt_donnell'``, i.e. what the numerical Cython kernels of
:mod:`panels.models.plate_clpt_donnell_num` and
:mod:`panels.models.plate_fsdt_donnell_num` return to
:meth:`.Shell.calc_kC` and :meth:`.Shell.calc_kG` with ``NLgeom=True``, to
:meth:`.Shell.calc_fint` and to :meth:`.Shell.calc_kTv`, written as dense
matrix products (BLAS GEMM) with operators tabulated once per domain. It is
selected by ``MultiDomain(..., plate_kernels='blas')``, see
:class:`.MultiDomain`, for the domains of :func:`is_eligible`; the Cython
kernels are used for every other domain and call.

The results are those of the Cython kernels to the round-off: same
Gauss-Legendre points, same constitutive matrix and same split of the
tangent between ``kC`` and ``kG``; the matrices are returned with the same
pattern, the upper triangle by pairs of terms, see :meth:`PlateOperators.kC`.

Notes
-----
Let `\boldsymbol{d}_p` be the vector of the generalized derivatives of the
displacement fields at the integration point `p`, linear in the Ritz
constants `\boldsymbol{c}` of the domain, `\boldsymbol{d}_p =
\boldsymbol{D}_p \boldsymbol{c}`:

- CLPT, 9 entries: `(u_{,x}, u_{,y}, v_{,x}, v_{,y}, w_{,x}, w_{,y},
  w_{,xx}, w_{,yy}, w_{,xy})`;
- FSDT, 12 entries: `(u_{,x}, u_{,y}, v_{,x}, v_{,y}, w_{,x}, w_{,y},
  \phi_x, \phi_y, \phi_{x,x}, \phi_{x,y}, \phi_{y,x}, \phi_{y,y})`.

A row of `\boldsymbol{D}_p` is the product `f_i(\xi_p) g_j(\eta_p)` of
Bardell's functions of :mod:`panels.bardell`, or of their derivatives, of
the field that it differentiates, scaled by `2/a` and `2/b` per derivative,
so that `\boldsymbol{D}_p` depends only on `m, n, a, b`, the points and the
boundary flags, and is tabulated once. The generalized strains of the von
Kármán kinematics of :mod:`panels.models` are, with `\boldsymbol{\varepsilon}
= (\varepsilon_{xx}, \varepsilon_{yy}, \gamma_{xy}, \kappa_{xx},
\kappa_{yy}, \kappa_{xy})` for the CLPT, `\kappa = (-w_{,xx}, -w_{,yy},
-2 w_{,xy})`, and `(\varepsilon_{xx}, \varepsilon_{yy}, \gamma_{xy},
\phi_{x,x}, \phi_{y,y}, \phi_{x,y} + \phi_{y,x}, \gamma_{yz} = \phi_y +
w_{,y}, \gamma_{xz} = \phi_x + w_{,x})` for the FSDT,

.. math::

    \boldsymbol{\varepsilon}_p = \boldsymbol{\varepsilon}_{L,p} +
    \boldsymbol{\varepsilon}_{NL,p} \qquad
    \boldsymbol{\varepsilon}_{NL,p} = \left(\tfrac{1}{2} w_{,x}^2,
    \tfrac{1}{2} w_{,y}^2, w_{,x} w_{,y}, 0, \dots\right)

linear in `\boldsymbol{d}_p` for `\boldsymbol{\varepsilon}_{L,p}`, and
their variation is `\delta \boldsymbol{\varepsilon}_p = \boldsymbol{B}_p
\delta \boldsymbol{d}_p`, where the small matrix `\boldsymbol{B}_p` (6 x 9
or 8 x 12) is constant apart from the entries `w_{,x}` and `w_{,y}` of the
membrane strains. With the constitutive matrix `\boldsymbol{F}`, the
stresses `\boldsymbol{\sigma}_p = \boldsymbol{F} \boldsymbol{\varepsilon}_p`
and the weights `w_p` of the points (times `ab/4`):

.. math::

    \boldsymbol{F}_{int} = \sum_p w_p \boldsymbol{D}_p^\top
    \boldsymbol{B}_p^\top \boldsymbol{\sigma}_p \qquad
    \boldsymbol{K} = \sum_p w_p \boldsymbol{D}_p^\top \left(
    \boldsymbol{B}_p^\top \boldsymbol{F} \boldsymbol{B}_p +
    \boldsymbol{G}_p \right) \boldsymbol{D}_p = \boldsymbol{D}^\top
    \boldsymbol{H} \boldsymbol{D}

where `\boldsymbol{G}_p` holds the membrane stresses `N_{xx}, N_{xy},
N_{yy}` in the rows and columns of `w_{,x}, w_{,y}`, `\boldsymbol{D}` is
`\boldsymbol{D}_p` of all the points stacked, and `\boldsymbol{H}` is
block diagonal with the blocks `w_p (\boldsymbol{B}_p^\top \boldsymbol{F}
\boldsymbol{B}_p + \boldsymbol{G}_p)`. The kernels split the tangent as:

- ``kC``: `\boldsymbol{B}_p^\top \boldsymbol{F} \boldsymbol{B}_p` and
  `\boldsymbol{G}_p` of the stresses of `\boldsymbol{\varepsilon}_{NL,p}`,
  i.e. `\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}`;
- ``kG``: `\boldsymbol{G}_p` of the stresses of
  `\boldsymbol{\varepsilon}_{L,p}`, homogeneous of degree one in
  `\boldsymbol{c}`;

and the product `\boldsymbol{K}_T \boldsymbol{v}` is
`\boldsymbol{D}^\top (\boldsymbol{H}_T (\boldsymbol{D} \boldsymbol{v}))`
with both parts in `\boldsymbol{H}_T`. Since each row of `\boldsymbol{D}_p`
differentiates one field only, `\boldsymbol{D}` is kept by field, and the
block of the fields `f, g` of `\boldsymbol{K}` is the GEMM
`\boldsymbol{D}_f^\top (\boldsymbol{H}_{fg} \boldsymbol{D}_g)`, of inner
dimension the number of points times the derivatives of `f`, against the
loops over the pairs of terms of the Cython kernels; the blocks with a null
`\boldsymbol{H}_{fg}` are skipped and `\boldsymbol{K}_{gf} =
\boldsymbol{K}_{fg}^\top`, `\boldsymbol{H}` being symmetric.

The constitutive matrix is the one that the Cython kernels read: for the
CLPT, ``kC``, ``kG`` and ``fint`` read the upper entries of the blocks
``A``, ``B`` and ``D`` only, while ``kTv`` and the FSDT kernels read the
whole matrix.

"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.special import roots_legendre

from panels import bardell


#: Models whose domains can use the operators of this module
MODELS = ('plate_clpt_donnell', 'plate_fsdt_donnell')

#: Fields of each model, in the order of the degrees of freedom of a term
_FIELDS = {'plate_clpt_donnell': ('u', 'v', 'w'),
           'plate_fsdt_donnell': ('u', 'v', 'w', 'phix', 'phiy')}

#: Generalized derivatives ``(field, x function, y function)``
_SPEC = {
    'plate_clpt_donnell': [
        ('u', 'fp', 'g'), ('u', 'f', 'gp'), ('v', 'fp', 'g'),
        ('v', 'f', 'gp'), ('w', 'fp', 'g'), ('w', 'f', 'gp'),
        ('w', 'fpp', 'g'), ('w', 'f', 'gpp'), ('w', 'fp', 'gp')],
    'plate_fsdt_donnell': [
        ('u', 'fp', 'g'), ('u', 'f', 'gp'), ('v', 'fp', 'g'),
        ('v', 'f', 'gp'), ('w', 'fp', 'g'), ('w', 'f', 'gp'),
        ('phix', 'f', 'g'), ('phiy', 'f', 'g'), ('phix', 'fp', 'g'),
        ('phix', 'f', 'gp'), ('phiy', 'fp', 'g'), ('phiy', 'f', 'gp')]}

# indices of w_{,x} and w_{,y} in the generalized derivatives
_WX, _WY = 4, 5


def _flags(p, fields):
    r"""Boundary flags of the approximation functions of ``fields``"""
    out = []
    for f in fields:
        for d in 'xy':
            out.extend(float(getattr(p, '%s%s%s%s' % (d, e, f, r)))
                       for e, r in (('1', ''), ('1', 'r'), ('2', ''),
                                    ('2', 'r')))
    return tuple(out)


def is_eligible(p):
    r"""Whether the domain ``p`` can use :class:`PlateOperators`

    The attributes of the domain are checked against the conditions listed
    below, without building the operators. A domain for which
    :meth:`.Shell.is_partial_domain` raises ``ValueError`` is not eligible.

    Parameters
    ----------
    p : :class:`.Shell`
        A domain of a :class:`.MultiDomain`.

    Returns
    -------
    eligible : bool
        ``True`` when all of the following hold, ``False`` otherwise:

        - ``p.model`` is ``'plate_clpt_donnell'`` or
          ``'plate_fsdt_donnell'``;
        - the constitutive matrix is constant over the domain: the
          attribute ``ABDnxny`` is ``None`` or a single ``(ne, ne)``
          matrix;
        - the integration domain is the whole domain,
          :meth:`.Shell.is_partial_domain` is ``False``;
        - no constant membrane stresses: ``Nxx``, ``Nyy``, ``Nxy``,
          ``Nxx_cte``, ``Nyy_cte`` and ``Nxy_cte`` are all zero;
        - no follower pressure load, :meth:`.Shell.has_follower_loads`;
        - ``row_start`` and ``col_start`` are set and equal, as in a
          :class:`.MultiDomain`.

    """
    if p.model not in MODELS:
        return False
    if p.row_start is None or p.col_start is None:
        return False
    if p.row_start != p.col_start:
        return False
    if any((p.Nxx, p.Nyy, p.Nxy, p.Nxx_cte, p.Nyy_cte, p.Nxy_cte)):
        return False
    if p.has_follower_loads():
        return False
    if p.ABDnxny is not None and np.ndim(p.ABDnxny) != 2:
        return False
    try:
        if p.is_partial_domain():
            return False
    except ValueError:
        return False
    return True


def operators_key(p):
    r"""Everything the tabulated operators of the domain ``p`` depend on

    The key is stored in the attribute ``key`` of :class:`PlateOperators`
    when the operators are built; operators whose key differs from the
    current key of the domain must be built again.

    Parameters
    ----------
    p : :class:`.Shell`
        A domain of the model ``'plate_clpt_donnell'`` or
        ``'plate_fsdt_donnell'``, with ``row_start`` set.

    Returns
    -------
    key : tuple
        The model, ``m``, ``n``, ``a``, ``b``, the numbers of points ``nx``
        and ``ny``, the boundary flags of the fields of the model and
        ``row_start``. Two domains with the same key have the same
        operators.

    """
    nx, ny = p.nx, p.ny
    return (p.model, int(p.m), int(p.n), float(p.a), float(p.b), int(nx),
            int(ny), _flags(p, _FIELDS[p.model]), int(p.row_start))


def _tab(fun, xs, flags, m):
    return np.array([np.asarray(fun(x, *flags))[:m] for x in xs])


class PlateOperators(object):
    r"""Tabulated generalized-derivative operators of one domain

    Built from the current attributes of the domain, see
    :func:`operators_key`; the constitutive matrix is read at each call.
    The domain must satisfy :func:`is_eligible`.

    Parameters
    ----------
    p : :class:`.Shell`
        The domain, of the model ``'plate_clpt_donnell'`` or
        ``'plate_fsdt_donnell'``.

    Attributes
    ----------
    key : tuple
        :func:`operators_key` of the domain when the operators were built.
    ndof : int
        Number of Ritz constants of the domain, ``dofs*m*n``.
    npts : int
        Number of integration points, ``nx*ny``.

    Raises
    ------
    ValueError
        When the model of ``p`` is not one of :data:`MODELS`.

    Notes
    -----
    For each field `f` of the model, the operator `\boldsymbol{D}_f` of
    shape ``(k_f, npts, m*n)`` holds, for each of the `k_f` generalized
    derivatives of the field and each integration point `p`, the products
    `f_i(\xi_p) g_j(\eta_p)` of the approximation functions, or of their
    derivatives scaled by `2/a` and `2/b` per derivative, in the column
    ``j*m + i`` of the term `(i, j)`. The points are the tensor product of
    the ``nx`` and ``ny`` Gauss-Legendre points, `\eta` varying fastest,
    with the weights `w_p = w_{\xi} w_{\eta} ab/4`.

    """
    def __init__(self, p):
        if p.model not in MODELS:
            raise ValueError('model {0!r} not in {1}'.format(p.model, MODELS))
        self.key = operators_key(p)
        self.model = p.model
        self.clpt = p.model == 'plate_clpt_donnell'
        m, n, a, b = p.m, p.n, p.a, p.b
        nx, ny = p.nx, p.ny
        xis, wx = roots_legendre(nx)
        etas, wy = roots_legendre(ny)
        #NOTE points ordered (ptx, pty), pty fastest, as the weights
        self.w = (np.outer(wx, wy)*a*b/4).ravel()
        self.npts = nx*ny
        self.fields = _FIELDS[p.model]
        self.DOF = len(self.fields)
        self.nmn = m*n
        self.ndof = self.DOF*self.nmn
        self.row0 = p.row_start
        spec = _SPEC[p.model]
        self.nd = len(spec)

        tab = {}
        for f in self.fields:
            fx = tuple(float(getattr(p, 'x%s%s%s' % (e, f, r)))
                       for e, r in (('1', ''), ('1', 'r'), ('2', ''),
                                    ('2', 'r')))
            fy = tuple(float(getattr(p, 'y%s%s%s' % (e, f, r)))
                       for e, r in (('1', ''), ('1', 'r'), ('2', ''),
                                    ('2', 'r')))
            tab[f] = dict(
                f=_tab(bardell.calc_vec_f, xis, fx, m),
                fp=_tab(bardell.calc_vec_fp, xis, fx, m)*(2/a),
                fpp=_tab(bardell.calc_vec_fpp, xis, fx, m)*(2/a)**2,
                g=_tab(bardell.calc_vec_f, etas, fy, n),
                gp=_tab(bardell.calc_vec_fp, etas, fy, n)*(2/b),
                gpp=_tab(bardell.calc_vec_fpp, etas, fy, n)*(2/b)**2)

        def op(f, kx, ky):
            # (npts, m*n), column j*m + i, as the terms of the kernels
            A = tab[f][kx]
            B = tab[f][ky]
            return np.einsum('xi,yj->xyji', A, B).reshape(self.npts,
                                                          self.nmn)

        #NOTE per field: the indices of its derivatives in d_p and the
        #     operators (k_f, npts, m*n) acting on its m*n constants
        self.fder = [[k for k, s in enumerate(spec) if s[0] == f]
                     for f in self.fields]
        self.Df = [np.ascontiguousarray(np.stack([op(*spec[k]) for k in ks]))
                   for ks in self.fder]

        # pattern of the kernels: upper triangle by pairs of terms
        idx = np.arange(self.ndof)
        r = np.repeat(idx, self.ndof)
        cc = np.tile(idx, self.ndof)
        upper = (r // self.DOF) <= (cc // self.DOF)
        self._flat_upper = np.flatnonzero(upper)
        self._rows_upper = self.row0 + r[upper]
        self._cols_upper = self.row0 + cc[upper]
        iw = self.fields.index('w')
        ww = upper & (r % self.DOF == iw) & (cc % self.DOF == iw)
        self._flat_ww = np.flatnonzero(ww)
        self._rows_ww = self.row0 + r[ww]
        self._cols_ww = self.row0 + cc[ww]

    # ------------------------------------------------------------------
    def _local(self, c):
        return np.asarray(c[self.row0:self.row0 + self.ndof], dtype=float)

    def _derivatives(self, cl):
        r"""``d`` of shape ``(npts, nd)`` of the local constants ``cl``

        Notes
        -----
        `\boldsymbol{d}_p = \boldsymbol{D}_p \boldsymbol{c}`, the
        generalized derivatives of the displacement fields at each
        integration point, computed field by field with the constants of
        the field.

        """
        d = np.empty((self.npts, self.nd))
        for fi, ks in enumerate(self.fder):
            cf = cl[fi::self.DOF]
            d[:, ks] = np.einsum('kpj,j->pk', self.Df[fi], cf)
        return d

    def _transpose(self, g):
        r"""``D^T g`` for ``g`` of shape ``(npts, nd)``, local vector

        Notes
        -----
        `\sum_p \boldsymbol{D}_p^\top \boldsymbol{g}_p`, computed field by
        field: the entries of a field receive the product of the transpose
        of its operator by the entries of `\boldsymbol{g}_p` of its
        derivatives.

        """
        out = np.empty(self.ndof)
        for fi, ks in enumerate(self.fder):
            A = self.Df[fi].reshape(-1, self.nmn)
            out[fi::self.DOF] = A.T @ g[:, ks].T.ravel()
        return out

    def _state(self, d):
        r"""``B`` (npts, ne, nd), linear and non-linear strains

        Notes
        -----
        From the generalized derivatives ``d`` at the points:
        `\boldsymbol{\varepsilon}_L`, linear in ``d``;
        `\boldsymbol{\varepsilon}_{NL} = (\tfrac{1}{2} w_{,x}^2,
        \tfrac{1}{2} w_{,y}^2, w_{,x} w_{,y}, 0, \dots)`; and
        `\boldsymbol{B}_p` such that `\delta \boldsymbol{\varepsilon}_p =
        \boldsymbol{B}_p \delta \boldsymbol{d}_p`, constant apart from the
        entries `w_{,x}` and `w_{,y}` in the rows of the membrane strains.

        """
        npts = self.npts
        wx, wy = d[:, _WX], d[:, _WY]
        if self.clpt:
            B = np.zeros((npts, 6, self.nd))
            B[:, 3, 6] = -1.
            B[:, 4, 7] = -1.
            B[:, 5, 8] = -2.
            eL = np.stack((d[:, 0], d[:, 3], d[:, 1] + d[:, 2], -d[:, 6],
                           -d[:, 7], -2*d[:, 8]), axis=1)
        else:
            B = np.zeros((npts, 8, self.nd))
            B[:, 3, 8] = 1.
            B[:, 4, 11] = 1.
            B[:, 5, 9] = 1.
            B[:, 5, 10] = 1.
            B[:, 6, 5] = 1.
            B[:, 6, 7] = 1.
            B[:, 7, 4] = 1.
            B[:, 7, 6] = 1.
            eL = np.stack((d[:, 0], d[:, 3], d[:, 1] + d[:, 2], d[:, 8],
                           d[:, 11], d[:, 9] + d[:, 10], d[:, 7] + d[:, 5],
                           d[:, 6] + d[:, 4]), axis=1)
        B[:, 0, 0] = 1.
        B[:, 0, 4] = wx
        B[:, 1, 3] = 1.
        B[:, 1, 5] = wy
        B[:, 2, 1] = 1.
        B[:, 2, 2] = 1.
        B[:, 2, 4] = wy
        B[:, 2, 5] = wx
        eNL = np.zeros_like(eL)
        eNL[:, 0] = 0.5*wx*wx
        eNL[:, 1] = 0.5*wy*wy
        eNL[:, 2] = wx*wy
        return B, eL, eNL

    def _H(self, d, F, part):
        r"""Weighted per-point matrices ``(npts, nd, nd)`` of ``part``,
        ``'kC'``, ``'kG'`` or ``'kT'``

        Notes
        -----
        `\boldsymbol{H}_p = w_p (\boldsymbol{B}_p^\top \boldsymbol{F}
        \boldsymbol{B}_p + \boldsymbol{G}_p)`, where `\boldsymbol{G}_p`
        holds the membrane stresses `N_{xx}, N_{xy}, N_{yy}`, the first
        three rows of `\boldsymbol{F}` times the strains, in the rows and
        columns of `w_{,x}, w_{,y}`:

        - ``'kC'``: `\boldsymbol{B}_p^\top \boldsymbol{F} \boldsymbol{B}_p`
          and `\boldsymbol{G}_p` of the stresses of
          `\boldsymbol{\varepsilon}_{NL,p}`;
        - ``'kG'``: `\boldsymbol{G}_p` of the stresses of
          `\boldsymbol{\varepsilon}_{L,p}` only;
        - ``'kT'``: `\boldsymbol{B}_p^\top \boldsymbol{F} \boldsymbol{B}_p`
          and `\boldsymbol{G}_p` of the stresses of the total strains.

        """
        B, eL, eNL = self._state(d)
        H = np.zeros((self.npts, self.nd, self.nd))
        if part in ('kC', 'kT'):
            FB = np.einsum('rs,psq->prq', F, B)
            H += np.einsum('pra,prq->paq', B, FB)
        if part == 'kC':
            N = eNL @ F[:3, :].T
        elif part == 'kG':
            N = eL @ F[:3, :].T
        else:
            N = (eL + eNL) @ F[:3, :].T
        H[:, _WX, _WX] += N[:, 0]
        H[:, _WX, _WY] += N[:, 2]
        H[:, _WY, _WX] += N[:, 2]
        H[:, _WY, _WY] += N[:, 1]
        return H*self.w[:, None, None]

    def _dense(self, H):
        r"""Local dense matrix ``D^T H D``, by blocks of fields

        Notes
        -----
        The block of the fields `f, g` is `\boldsymbol{K}_{fg} =
        \boldsymbol{D}_f^\top (\boldsymbol{H}_{fg} \boldsymbol{D}_g)`, with
        `\boldsymbol{H}_{fg}` the rows of the derivatives of `f` and the
        columns of those of `g` of the per-point matrices. Only the blocks
        with `g \ge f` are computed, `\boldsymbol{K}_{gf} =
        \boldsymbol{K}_{fg}^\top` for a symmetric `\boldsymbol{H}`, and
        the blocks with a null `\boldsymbol{H}_{fg}` are skipped.

        """
        nf, nmn, DOF = len(self.fields), self.nmn, self.DOF
        K = np.zeros((self.ndof, self.ndof))
        for f in range(nf):
            A = self.Df[f].reshape(-1, nmn)
            for g in range(f, nf):
                Hfg = H[:, self.fder[f]][:, :, self.fder[g]]
                if not np.any(Hfg):
                    continue
                # E_a = sum_b H_ab D_b, a derivative of f, b of g
                E = np.einsum('pab,bpj->apj', Hfg, self.Df[g])
                Kfg = A.T @ E.reshape(-1, nmn)
                K[f::DOF, g::DOF] = Kfg
                if g != f:
                    K[g::DOF, f::DOF] = Kfg.T
        return K

    # ------------------------------------------------------------------
    def kC(self, c, F, size):
        r"""Constitutive part of the tangent at ``c``

        As ``fkC_num`` of the numerical kernels of the model,
        :mod:`panels.models.plate_clpt_donnell_num` or
        :mod:`panels.models.plate_fsdt_donnell_num`, with ``NLgeom=1``.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, 1-D of size ``size``.
        F : np.ndarray
            Constitutive matrix ``(ne, ne)`` of the domain, see
            :func:`constitutive_matrix`.
        size : int
            Size of the assembly.

        Returns
        -------
        kC : scipy.sparse.coo_matrix
            Shape ``(size, size)``, with the entries of the block of the
            domain whose term of the row is not after the term of the column,
            i.e. ``(r - row0)//dofs <= (c - row0)//dofs``, as the Cython
            kernels: the caller symmetrizes it, e.g. with
            ``structsolve.sparseutils.finalize_symmetric_matrix``.

        Notes
        -----
        `\boldsymbol{D}^\top \boldsymbol{H} \boldsymbol{D}` with the blocks
        `w_p (\boldsymbol{B}_p^\top \boldsymbol{F} \boldsymbol{B}_p +
        \boldsymbol{G}_p)` of `\boldsymbol{H}`, where `\boldsymbol{G}_p`
        holds the membrane stresses of the non-linear strains
        `\boldsymbol{\varepsilon}_{NL,p}` at ``c``.

        """
        H = self._H(self._derivatives(self._local(c)), F, 'kC')
        K = self._dense(H).ravel()
        return coo_matrix((K[self._flat_upper], (self._rows_upper,
                          self._cols_upper)), shape=(size, size))

    def kG(self, c, F, size):
        r"""Geometric stiffness of the linear membrane stresses at ``c``

        As ``fkG_num`` of the numerical kernels of the model, without
        constant membrane stresses.

        The pattern is that of :meth:`kC`, restricted to the entries of `w`
        with `w`, the only ones returned.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, 1-D of size ``size``.
        F : np.ndarray
            Constitutive matrix ``(ne, ne)`` of the domain, see
            :func:`constitutive_matrix`.
        size : int
            Size of the assembly.

        Returns
        -------
        kG : scipy.sparse.coo_matrix
            Shape ``(size, size)``.

        Notes
        -----
        `\boldsymbol{D}^\top \boldsymbol{H} \boldsymbol{D}` with the blocks
        `w_p \boldsymbol{G}_p` of `\boldsymbol{H}`, where `\boldsymbol{G}_p`
        holds the membrane stresses `N_{xx}, N_{xy}, N_{yy}` of the linear
        strains `\boldsymbol{\varepsilon}_{L,p}` at ``c`` in the rows and
        columns of `w_{,x}, w_{,y}`; it is homogeneous of degree one in
        ``c``.

        """
        H = self._H(self._derivatives(self._local(c)), F, 'kG')
        K = self._dense(H).ravel()
        return coo_matrix((K[self._flat_ww], (self._rows_ww, self._cols_ww)),
                          shape=(size, size))

    def fint(self, c, F, size):
        r"""Internal force vector at ``c``

        As ``calc_fint`` of the numerical kernels of the model.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, 1-D of size ``size``.
        F : np.ndarray
            Constitutive matrix ``(ne, ne)`` of the domain, see
            :func:`constitutive_matrix`.
        size : int
            Size of the assembly.

        Returns
        -------
        fint : np.ndarray
            1-D of size ``size``, null outside the domain.

        Notes
        -----
        `\boldsymbol{F}_{int} = \sum_p w_p \boldsymbol{D}_p^\top
        \boldsymbol{B}_p^\top \boldsymbol{\sigma}_p`, with the stresses
        `\boldsymbol{\sigma}_p = \boldsymbol{F} (\boldsymbol{\varepsilon}_{L,p}
        + \boldsymbol{\varepsilon}_{NL,p})` at ``c``.

        """
        d = self._derivatives(self._local(c))
        B, eL, eNL = self._state(d)
        sig = (eL + eNL) @ F.T
        g = np.einsum('pra,pr->pa', B, sig)*self.w[:, None]
        out = np.zeros(size)
        out[self.row0:self.row0 + self.ndof] = self._transpose(g)
        return out

    def kTv(self, c, v, F, size):
        r"""Product of the tangent at ``c`` by ``v``

        As ``calc_kTv`` of the numerical kernels of the model, without
        constant membrane stresses.

        Parameters
        ----------
        c, v : np.ndarray
            Ritz constants of the assembly and the vector, 1-D of size
            ``size``.
        F : np.ndarray
            Constitutive matrix ``(ne, ne)`` of the domain, see
            :func:`constitutive_matrix`.
        size : int
            Size of the assembly.

        Returns
        -------
        kTv : np.ndarray
            1-D of size ``size``, null outside the domain.

        Notes
        -----
        `\boldsymbol{K}_T \boldsymbol{v} = \boldsymbol{D}^\top
        (\boldsymbol{H}_T (\boldsymbol{D} \boldsymbol{v}))`, where the
        blocks of `\boldsymbol{H}_T` are `w_p (\boldsymbol{B}_p^\top
        \boldsymbol{F} \boldsymbol{B}_p + \boldsymbol{G}_p)` with
        `\boldsymbol{G}_p` of the membrane stresses of the total strains at
        ``c``; the matrix `\boldsymbol{K}_T` is not formed.

        """
        H = self._H(self._derivatives(self._local(c)), F, 'kT')
        dv = self._derivatives(self._local(v))
        g = np.einsum('pab,pb->pa', H, dv)
        out = np.zeros(size)
        out[self.row0:self.row0 + self.ndof] = self._transpose(g)
        return out


def constitutive_matrix(p, symmetric_blocks):
    r"""Constant constitutive matrix of the domain as the kernels read it

    The matrix of the domain is copied; with ``symmetric_blocks=True`` the
    entries below the diagonal of the blocks ``A``, ``B`` and ``D`` are
    replaced by those above it, and every other entry is set to zero.

    Parameters
    ----------
    p : :class:`.Shell`
        An eligible domain, see :func:`is_eligible`.
    symmetric_blocks : bool
        ``True`` to build the blocks ``A``, ``B`` and ``D`` from their upper
        entries, as the CLPT kernels of ``kC``, ``kG`` and ``fint`` read
        them; ``False`` for the matrix as given.

    Returns
    -------
    F : np.ndarray
        Shape ``(6, 6)`` (CLPT) or ``(8, 8)`` (FSDT).

    Notes
    -----
    `\boldsymbol{F}` relates the generalized strains to the stress
    resultants, `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}`, with the membrane, coupling and bending
    blocks `\boldsymbol{A}`, `\boldsymbol{B}` and `\boldsymbol{D}` in its
    first six rows and columns. With ``symmetric_blocks=True`` each of
    these blocks is `\boldsymbol{X}_u + \boldsymbol{X}_s^\top`, with
    `\boldsymbol{X}_u` its upper triangle and `\boldsymbol{X}_s` its
    strictly upper triangle, and the block `\boldsymbol{B}` is placed in
    both off-diagonal positions.

    """
    F, _, _ = p._get_ABDnxny(None, None, None)
    F = np.array(F, dtype=float)
    if symmetric_blocks:
        out = np.zeros_like(F)
        for i, j in ((0, 0), (0, 3), (3, 3)):
            X = np.triu(F[i:i+3, j:j+3])
            X = X + np.triu(X, 1).T
            out[i:i+3, j:j+3] = X
            out[j:j+3, i:i+3] = X
        F = out
    return F
