import numpy as np
from scipy.sparse.linalg import spsolve

from .logger import msg
from .sparseutils import remove_null_cols


def solve(a, b, silent=False, **kwargs):
    """Wrapper for spsolve removing null columns

    The null columns of matrix ``a`` is removed such and the linear system of
    equations is solved. The corresponding values of the solution ``x`` where
    the columns are null will also be null values.

    Parameters
    ----------
    a : ndarray or sparse matrix
        A square matrix that will be converted to CSR form in the solution.
    b : scipy sparse matrix
        The matrix or vector representing the right hand side of the equation.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    kwargs : keyword arguments, optional
        Other arguments directly passed to :func:`spsolve`.

    Returns
    -------
    x : ndarray or sparse matrix
        The solution of the sparse linear equation.
        If ``b`` is a vector, then ``x`` is a vector of size ``a.shape[1]``.
        If ``b`` is a sparse matrix, then ``x`` is a matrix of size
        ``(a.shape[1], b.shape[1])``.

    """
    a, used_cols = remove_null_cols(a, silent=silent)
    px = spsolve(a, b[used_cols], **kwargs)
    x = np.zeros(b.shape[0], dtype=b.dtype)
    x[used_cols] = px

    return x


def static(K, fext, silent=False):
    """Static Analyses

    Parameters
    ----------

    K : sparse_matrix
        Stiffness matrix. Should include initial stress stiffness matrix,
        aerodynamic matrix and so forth when applicable.
    fext : array-like
        Vector of external loads.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.

    """
    increments = []
    cs = []

    NLgeom=False
    if NLgeom:
        raise NotImplementedError('Independent static function not ready for NLgeom')
    else:
        msg('Started Linear Static Analysis', silent=silent)
        c = solve(K, fext, silent=silent)
        increments.append(1.)
        cs.append(c)
        msg('Finished Linear Static Analysis', silent=silent)

    return increments, cs
