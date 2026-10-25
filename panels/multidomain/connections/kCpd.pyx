#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE
from panels import modelDB as _modelDB


def _dofs(p):
    r"""Number of degrees of freedom per term of the model of ``p``

    Parameters
    ----------
    p : :class:`.Shell`
        Panel whose attribute ``model`` is read.

    Returns
    -------
    dofs : int
        ``3`` for the CLPT models and ``5`` for the FSDT and TSDT models,
        whose Ritz constants of the term ``(i, j)`` start at ``dofs*(j*m +
        i)`` with ``u, v, w`` at the offsets ``0, 1, 2``, see
        :mod:`panels.modelDB`.

    Raises
    ------
    ValueError
        If ``p.model`` is not a model of :mod:`panels.modelDB`.

    """
    model = getattr(p, 'model', None)
    if model not in _modelDB.db:
        raise ValueError('{0!r} is not a valid model, see '
                         'panels.modelDB'.format(model))
    return _modelDB.db[model]['dofs']


def _fkC_z(ks, p, xp, yp, z, size, row0, col0):
    r"""Penalty stiffness of the prescribed `u`, `v` and `w` at the
    thickness coordinate ``z``, at the point ``(xp, yp)`` or along the line
    ``xp`` (``yp=None``) or ``yp`` (``xp=None``)

    The rows of each component are those of
    :func:`.nullspace.pd_rows`, combinations of the fields, of the
    rotations and of the slopes of `w` for ``z`` different from zero. Along
    a line the integral is computed with ``max(num, 4)`` Gauss-Legendre
    points, ``num`` the number of terms along the line, exact for the
    products of the polynomials of Bardell's functions. Returns the upper
    triangle, as the kernels with ``z = 0``.

    """
    from numpy.polynomial.legendre import leggauss
    from scipy.sparse import triu
    from panels.multidomain.connections import nullspace
    if xp is not None and yp is not None:
        xis = np.array([2*xp/p.a - 1])
        etas = np.array([2*yp/p.b - 1])
        weights = np.ones(1)
    elif yp is None:
        xis = np.array([2*xp/p.a - 1])
        etas, weights = leggauss(max(p.n, 4))
        weights = weights*p.b/2
    else:
        xis, weights = leggauss(max(p.m, 4))
        etas = np.array([2*yp/p.b - 1])
        weights = weights*p.a/2
    K = coo_matrix((size, size))
    for comp, k in enumerate(ks):
        if k == 0:
            continue
        Br = nullspace.pd_rows(p, comp, xis, etas, z, row0, size)
        Bc = nullspace.pd_rows(p, comp, xis, etas, z, col0, size)
        K = K + k*(Br.T.multiply(weights[None, :]) @ Bc)
    return triu(coo_matrix(K), format='coo')


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j,
            double x1t, double x1r, double x2t, double x2r,
            double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) nogil


def fkCpd(double ku, double kv, double kw, object p, double xp, double yp,
          int size, int row0, int col0, double z=0.):
    r"""
    Penalty for prescribed displacement at a point.

    The in-plane displacements are those at the thickness coordinate ``z``,
    the one given to :meth:`.Shell.add_point_pd`.

    Parameters
    ----------
    ku, kv, kw : float
        Translation penalty stiffnesses.
        Use this to specify which disp is being provided. The corresponding stiffness is non zero - rest are zero
    p : :class:`.Shell`
        Panel with the prescribed displacement, of any model of
        :mod:`panels.modelDB`, the stride of its Ritz constants being the
        number of degrees of freedom per term of the model.
    xp, yp : float
        Positions x and y of the prescribed displacement. This is in physical (x,y) coordinates
    size : int
        Size of the stiffness matrix of the assembly, the sum of
        ``dofs*p.m*p.n`` of its panels, with ``dofs`` the number of degrees
        of freedom per term of the model of each panel, ``3`` or ``5``, see
        :mod:`panels.modelDB`; see :meth:`.MultiDomain.get_size` and
        :meth:`.Shell.get_size`.
    row0 : int
        First row of the panel in the matrix of the assembly,
        ``p.row_start``.
    col0 : int
        First column of the panel in the matrix of the assembly,
        ``p.col_start``.
    z : float, optional
        Thickness coordinate, measured from the reference surface, of the
        penalized `u` and `v`, a length. Default ``0.``, the reference
        surface.

    Returns
    -------
    kCpd : scipy.sparse.coo_matrix
        A sparse matrix with the penalty stiffness for prescribed point
        displacement, of shape ``(size, size)``, with the upper triangle only,
        to be symmetrized with
        :func:`structsolve.sparseutils.finalize_symmetric_matrix`.

    Raises
    ------
    ValueError
        If ``p.model`` is not a model of :mod:`panels.modelDB`.

    Notes
    -----
    The penalty energy is `\frac{1}{2} \sum_k k_k (\{g_k\} \{\bar{c}\} -
    u_{kp})^2`, with `\{g_k\}` the row of the component `k` of `u, v, w`
    at the point, whose second derivative gives the stiffness `k_k
    \{g_k\}^\top \{g_k\}`. For ``z`` different from zero the rows of `u(z)`
    and `v(z)` combine `u` or `v`, a rotation and a slope of `w`, see
    :func:`.nullspace.pd_constraint_matrix`, and the stiffness has terms
    between these fields.

    """
    if z != 0:
        _dofs(p)
        return _fkC_z((ku, kv, kw), p, xp, yp, z, size, row0, col0)
    cdef double [:, ::1] gtab, ftab
    cdef int i, k, j, l, c, row, col, dofs
    cdef int m, n # no of terms in x,y
    cdef double a, b, xip, etap
    
    # BCs
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    # x1, x2 are the switch cases for the BCs

    cdef long [:] kCpdr, kCpdc # Contains the positions (rows and cols) of the sparse matrix whose values are in kCpdv
    cdef double [:] kCpdv

    cdef double fAu, fBu, gAu, gBu
    cdef double fAv, fBv, gAv, gBv
    cdef double fAw, fBw, gAw, gBw

    #NOTE stride of the Ritz constants, 3 or 5 DOFs per term, with u, v, w
    #     at the offsets 0, 1, 2 of each term
    dofs = _dofs(p)
    a = p.a
    b = p.b
    m = p.m
    n = p.n
    x1u = p.x1u ; x1ur = p.x1ur ; x2u = p.x2u ; x2ur = p.x2ur
    x1v = p.x1v ; x1vr = p.x1vr ; x2v = p.x2v ; x2vr = p.x2vr
    x1w = p.x1w ; x1wr = p.x1wr ; x2w = p.x2w ; x2wr = p.x2wr
    y1u = p.y1u ; y1ur = p.y1ur ; y2u = p.y2u ; y2ur = p.y2ur
    y1v = p.y1v ; y1vr = p.y1vr ; y2v = p.y2v ; y2vr = p.y2vr
    y1w = p.y1w ; y1wr = p.y1wr ; y2w = p.y2w ; y2wr = p.y2wr

    fdim = 3*m*n*m*n # 3 bec the matrix (from the sym expressions) has only 1 sparse term. Might be better to make it a var and then change it

    kCpdr = np.zeros((fdim,), dtype=INT)
    kCpdc = np.zeros((fdim,), dtype=INT)
    kCpdv = np.zeros((fdim,), dtype=DOUBLE)

    # A and B are the left and right terms when you have multiple terms
    # For eg: S1p*S2p, then the first term is A and 2nd is B so that shape functions are now independent


    # i,j controls f and g for A (so starting from the left term)
    # k,l ...  for B (... right)
    
    # i, j are for the rows (see the line where the row is being updated) - check var_descr.py CHECK 
    # k, l are for the cols
    
    gtab = np.zeros((3, n), dtype=DOUBLE)
    ftab = np.zeros((3, m), dtype=DOUBLE)

    with nogil:
        xip = 2*xp/a - 1.
        etap = 2*yp/b - 1.
        
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j in range(n):
            gtab[0, j] = f(j, etap, y1u, y1ur, y2u, y2ur)
            gtab[1, j] = f(j, etap, y1v, y1vr, y2v, y2vr)
            gtab[2, j] = f(j, etap, y1w, y1wr, y2w, y2wr)
        for i in range(m):
            ftab[0, i] = f(i, xip, x1u, x1ur, x2u, x2ur)
            ftab[1, i] = f(i, xip, x1v, x1vr, x2v, x2vr)
            ftab[2, i] = f(i, xip, x1w, x1wr, x2w, x2wr)

        for j in range(n):
            gAu = gtab[0, j] # the point location should be in natural coords
            gAv = gtab[1, j]
            gAw = gtab[2, j]
            # Generates a single shape function, based on the bardell function number j 
                # So its actually gAw_j
    
            for l in range(n):
                gBu = gtab[0, l]
                gBv = gtab[1, l]
                gBw = gtab[2, l]
        
                for i in range(m):
                    fAu = ftab[0, i]
                    fAv = ftab[1, i]
                    fAw = ftab[2, i]
            
                    for k in range(m):
                        row = row0 + dofs*(j*m + i) # row0 = starting row of this submatrix in the global matrix
                        col = col0 + dofs*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        fBu = ftab[0, k]
                        fBv = ftab[1, k]
                        fBw = ftab[2, k]
            
                        c += 1
                        kCpdr[c] = row+0
                        kCpdc[c] = col+0
                        kCpdv[c] += ku*fAu*fBu*gAu*gBu
                        c += 1
                        kCpdr[c] = row+1
                        kCpdc[c] = col+1
                        kCpdv[c] += kv*fAv*fBv*gAv*gBv
                        c += 1
                        kCpdr[c] = row+2
                        kCpdc[c] = col+2
                        kCpdv[c] += kw*fAw*fBw*gAw*gBw


    kCpd = coo_matrix((kCpdv, (kCpdr, kCpdc)), shape=(size, size))

    return kCpd


def fkCld_xcte(double ku, double kv, double kw, object p, double xp,
               int size, int row0, int col0, double z=0.):
    r"""
    Penalty for prescribed line displacement with a fixed x coordinate.

    The in-plane displacements are those at the thickness coordinate ``z``,
    the one given to :meth:`.Shell.add_distr_pd_fixed_x`.

    Parameters
    ----------
    ku, kv, kw : float
        Translation penalty stiffnesses.
    p : :class:`.Shell`
        Panel with the prescribed displacement, of any model of
        :mod:`panels.modelDB`, the stride of its Ritz constants being the
        number of degrees of freedom per term of the model.
    xp : float
        Position x of the prescribed line displacement. This is in physical (x,y) coordinates
    size : int
        Size of the stiffness matrix of the assembly, the sum of
        ``dofs*p.m*p.n`` of its panels, with ``dofs`` the number of degrees
        of freedom per term of the model of each panel, ``3`` or ``5``, see
        :mod:`panels.modelDB`; see :meth:`.MultiDomain.get_size` and
        :meth:`.Shell.get_size`.
    row0 : int
        First row of the panel in the matrix of the assembly,
        ``p.row_start``.
    col0 : int
        First column of the panel in the matrix of the assembly,
        ``p.col_start``.
    z : float, optional
        Thickness coordinate, measured from the reference surface, of the
        penalized `u` and `v`, a length. Default ``0.``, the reference
        surface.

    Returns
    -------
    kCld_xcte : scipy.sparse.coo_matrix
        A sparse matrix with the penalty stiffness for prescribed line
        displacement with constant x (so integral over y), of shape
        ``(size, size)``, with the upper triangle only, to be symmetrized
        with
        :func:`structsolve.sparseutils.finalize_symmetric_matrix`.

    Raises
    ------
    ValueError
        If ``p.model`` is not a model of :mod:`panels.modelDB`.

    Notes
    -----
    The penalty energy is that of :func:`.fkCpd` integrated along `y`,
    `\frac{1}{2} \sum_k \int_0^b k_k (\{g_k\} \{\bar{c}\} - g_k(y))^2 dy`,
    with the integrals of Bardell's functions along `y`; for ``z``
    different from zero the integrals are computed with ``max(n, 4)``
    Gauss-Legendre points, exact for these polynomials.

    """
    if z != 0:
        _dofs(p)
        return _fkC_z((ku, kv, kw), p, xp, None, z, size, row0, col0)
    cdef double [:, ::1] ftab
    cdef int i, k, j, l, c, row, col, dofs
    cdef int m, n
    cdef double a, b, xip
    
    # BCs
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    # x1, x2 are the limits of x

    cdef long [:] kCld_xcter, kCld_xctec # Contains the positions (rows and cols) of the sparse matrix whose values are in kCpdv
    cdef double [:] kCld_xctev
    
    cdef double fAu, fBu, gAugBu
    cdef double fAv, fBv, gAvgBv
    cdef double fAw, fBw, gAwgBw

    #NOTE stride of the Ritz constants, 3 or 5 DOFs per term, with u, v, w
    #     at the offsets 0, 1, 2 of each term
    dofs = _dofs(p)
    a = p.a
    b = p.b
    m = p.m
    n = p.n
    x1u = p.x1u ; x1ur = p.x1ur ; x2u = p.x2u ; x2ur = p.x2ur
    x1v = p.x1v ; x1vr = p.x1vr ; x2v = p.x2v ; x2vr = p.x2vr
    x1w = p.x1w ; x1wr = p.x1wr ; x2w = p.x2w ; x2wr = p.x2wr
    y1u = p.y1u ; y1ur = p.y1ur ; y2u = p.y2u ; y2ur = p.y2ur
    y1v = p.y1v ; y1vr = p.y1vr ; y2v = p.y2v ; y2vr = p.y2vr
    y1w = p.y1w ; y1wr = p.y1wr ; y2w = p.y2w ; y2wr = p.y2wr

    fdim = 3*m*n*m*n # 1 bec the matrix has only 1 sparse term. Might be better to make it a var and then change it

    kCld_xcter = np.zeros((fdim,), dtype=INT)
    kCld_xctec = np.zeros((fdim,), dtype=INT)
    kCld_xctev = np.zeros((fdim,), dtype=DOUBLE)

    # A and B are the left and right terms when you have multiple terms
    # For eg: S1p*S2p, then the first term is A and 2nd is B so that shape functions are now independent


    # i,j controls f and g for A (so starting from the left term)
    # l,m ...  for B (... right)
    # i, j are for the rows (see the line where the row is being updated) - WHY THOUHG?? CHECK 
    # k, l are for the cols
    
    ftab = np.zeros((3, m), dtype=DOUBLE)

    with nogil:
        xip = 2*xp/a - 1.
        
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for i in range(m):
            ftab[0, i] = f(i, xip, x1u, x1ur, x2u, x2ur)
            ftab[1, i] = f(i, xip, x1v, x1vr, x2v, x2vr)
            ftab[2, i] = f(i, xip, x1w, x1wr, x2w, x2wr)

        for j in range(n):    
            for l in range(n):
                # j and l are for y coord i.e. g - since thats being integrated, loop it outside
                
                # this already integrates it so extra integration needed
                gAugBu = integral_ff(j, l, y1u, y1ur, y2u, y2ur, y1u, y1ur, y2u, y2ur)
                gAvgBv = integral_ff(j, l, y1v, y1vr, y2v, y2vr, y1v, y1vr, y2v, y2vr)
                gAwgBw = integral_ff(j, l, y1w, y1wr, y2w, y2wr, y1w, y1wr, y2w, y2wr)

                for i in range(m):
                    fAu = ftab[0, i]
                    fAv = ftab[1, i]
                    fAw = ftab[2, i]
            
                    for k in range(m):
                        row = row0 + dofs*(j*m + i)
                        col = col0 + dofs*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        fBu = ftab[0, k]
                        fBv = ftab[1, k]
                        fBw = ftab[2, k]
            
                        c += 1
                        kCld_xcter[c] = row+0
                        kCld_xctec[c] = col+0
                        kCld_xctev[c] += ku*b*fAu*fBu*gAugBu/2.
                        c += 1
                        kCld_xcter[c] = row+1
                        kCld_xctec[c] = col+1
                        kCld_xctev[c] += kv*b*fAv*fBv*gAvgBv/2.
                        c += 1
                        kCld_xcter[c] = row+2
                        kCld_xctec[c] = col+2
                        kCld_xctev[c] += kw*b*fAw*fBw*gAwgBw/2.

    kCld_xcte = coo_matrix((kCld_xctev, (kCld_xcter, kCld_xctec)), shape=(size, size))
    # Creates a sparse matrix by putting ..v and the corresponding row and col given by ..r ..c

    return kCld_xcte


def fkCld_ycte(double ku, double kv, double kw, object p, double yp,
               int size, int row0, int col0, double z=0.):
    r"""
    Penalty for prescribed line displacement with a fixed y coordinate.

    The in-plane displacements are those at the thickness coordinate ``z``,
    the one given to :meth:`.Shell.add_distr_pd_fixed_y`.

    Parameters
    ----------
    ku, kv, kw : float
        Translation penalty stiffnesses.
    p : :class:`.Shell`
        Panel with the prescribed displacement, of any model of
        :mod:`panels.modelDB`, the stride of its Ritz constants being the
        number of degrees of freedom per term of the model.
    yp : float
        Position y of the prescribed line displacement. This is in physical (x,y) coordinates
    size : int
        Size of the stiffness matrix of the assembly, the sum of
        ``dofs*p.m*p.n`` of its panels, with ``dofs`` the number of degrees
        of freedom per term of the model of each panel, ``3`` or ``5``, see
        :mod:`panels.modelDB`; see :meth:`.MultiDomain.get_size` and
        :meth:`.Shell.get_size`.
    row0 : int
        First row of the panel in the matrix of the assembly,
        ``p.row_start``.
    col0 : int
        First column of the panel in the matrix of the assembly,
        ``p.col_start``.
    z : float, optional
        Thickness coordinate, measured from the reference surface, of the
        penalized `u` and `v`, a length. Default ``0.``, the reference
        surface.

    Returns
    -------
    kCld_ycte : scipy.sparse.coo_matrix
        A sparse matrix with the penalty stiffness for prescribed line
        displacement with constant y (so integral over x), of shape
        ``(size, size)``, with the upper triangle only, to be symmetrized
        with
        :func:`structsolve.sparseutils.finalize_symmetric_matrix`.

    Raises
    ------
    ValueError
        If ``p.model`` is not a model of :mod:`panels.modelDB`.

    Notes
    -----
    The penalty energy is that of :func:`.fkCld_xcte` integrated along `x`,
    with ``max(m, 4)`` Gauss-Legendre points for ``z`` different from
    zero.

    """
    if z != 0:
        _dofs(p)
        return _fkC_z((ku, kv, kw), p, None, yp, z, size, row0, col0)
    cdef double [:, ::1] gtab
    cdef int i, k, j, l, c, row, col, dofs
    cdef int m, n
    cdef double a, b, etap
    
    # BCs
    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr
    # x1, x2 are the limits of x

    cdef long [:] kCld_ycter, kCld_yctec # Contains the positions (rows and cols) of the sparse matrix whose values are in kCpdv
    cdef double [:] kCld_yctev
    
    cdef double fAufBu, gAu, gBu
    cdef double fAvfBv, gAv, gBv
    cdef double fAwfBw, gAw, gBw

    #NOTE stride of the Ritz constants, 3 or 5 DOFs per term, with u, v, w
    #     at the offsets 0, 1, 2 of each term
    dofs = _dofs(p)
    a = p.a
    b = p.b
    m = p.m
    n = p.n
    x1u = p.x1u ; x1ur = p.x1ur ; x2u = p.x2u ; x2ur = p.x2ur
    x1v = p.x1v ; x1vr = p.x1vr ; x2v = p.x2v ; x2vr = p.x2vr
    x1w = p.x1w ; x1wr = p.x1wr ; x2w = p.x2w ; x2wr = p.x2wr
    y1u = p.y1u ; y1ur = p.y1ur ; y2u = p.y2u ; y2ur = p.y2ur
    y1v = p.y1v ; y1vr = p.y1vr ; y2v = p.y2v ; y2vr = p.y2vr
    y1w = p.y1w ; y1wr = p.y1wr ; y2w = p.y2w ; y2wr = p.y2wr

    fdim = 3*m*n*m*n # 1 bec the matrix has only 1 sparse term. Might be better to make it a var and then change it

    kCld_ycter = np.zeros((fdim,), dtype=INT)
    kCld_yctec = np.zeros((fdim,), dtype=INT)
    kCld_yctev = np.zeros((fdim,), dtype=DOUBLE)

    # A and B are the left and right terms when you have multiple terms
    # For eg: S1p*S2p, then the first term is A and 2nd is B so that shape functions are now independent


    # i,j controls f and g for A (so starting from the left term)
    # l,m ...  for B (... right)
    # i, j are for the rows (see the line where the row is being updated) - WHY THOUHG?? CHECK 
    # k, l are for the cols
    
    gtab = np.zeros((3, n), dtype=DOUBLE)

    with nogil:
        etap = 2*yp/b - 1.
        
        c = -1


        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j in range(n):
            gtab[0, j] = f(j, etap, y1u, y1ur, y2u, y2ur)
            gtab[1, j] = f(j, etap, y1v, y1vr, y2v, y2vr)
            gtab[2, j] = f(j, etap, y1w, y1wr, y2w, y2wr)

        for i in range(m):
            for k in range(m):
                # i and k are for x coord i.e. f - since thats being integrated, loop it outside
                fAufBu = integral_ff(i, k, x1u, x1ur, x2u, x2ur, x1u, x1ur, x2u, x2ur)
                fAvfBv = integral_ff(i, k, x1v, x1vr, x2v, x2vr, x1v, x1vr, x2v, x2vr)
                fAwfBw = integral_ff(i, k, x1w, x1wr, x2w, x2wr, x1w, x1wr, x2w, x2wr)

                for j in range(n):    
                    gAu = gtab[0, j]
                    gAv = gtab[1, j]
                    gAw = gtab[2, j]

                    for l in range(n):
            
                        row = row0 + dofs*(j*m + i)
                        col = col0 + dofs*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = gtab[0, l]
                        gBv = gtab[1, l]
                        gBw = gtab[2, l]
            
                        c += 1
                        kCld_ycter[c] = row+0
                        kCld_yctec[c] = col+0
                        kCld_yctev[c] += ku*a*fAufBu*gAu*gBu/2.
                        c += 1
                        kCld_ycter[c] = row+1
                        kCld_yctec[c] = col+1
                        kCld_yctev[c] += kv*a*fAvfBv*gAv*gBv/2.
                        c += 1
                        kCld_ycter[c] = row+2
                        kCld_yctec[c] = col+2
                        kCld_yctev[c] += kw*a*fAwfBw*gAw*gBw/2.

    kCld_ycte = coo_matrix((kCld_yctev, (kCld_ycter, kCld_yctec)), shape=(size, size))

    return kCld_ycte
