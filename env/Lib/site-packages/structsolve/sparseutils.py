import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

from .logger import msg


def remove_null_cols(*args, **kwargs):
    """Remove null rows and cols of a symmetric, square sparse matrix.

    Parameters
    ----------
    args : list of sparse matrices
        The first matrix in this list will be used to extract the columns
        to be removed from all the other matrices. Use :class:`csr_matrix` to
        obtain a better performance.

    Returns
    -------
    out : list of sparse matrices and removed columns
        A list with the reduced matrices in the same order of ``args`` plus
        an array containing the removed columns at the last position.

    """
    silent = kwargs.get('silent', False)
    args = list(args)
    msg('Removing null columns...', level=3, silent=silent)
    num_cols = args[0].shape[1]

    if isinstance(args[0], csr_matrix):
        m = args[0]
    else:
        m = csr_matrix(args[0])
    rows, cols = m.nonzero()
    used_cols = np.unique(cols)

    for i, arg in enumerate(args):
        if isinstance(arg, csr_matrix):
            m = arg
        else:
            m = csr_matrix(arg)
        m = m[used_cols, :]
        #NOTE below, converting to csc_matrix seems to require more time than
        #     the "slow" column slicing for csr_matrix
        m = m[:, used_cols]
        args[i] = m
    args.append(used_cols)
    msg('{} columns removed'.format(num_cols - used_cols.shape[0]),
            level=4, silent=silent)
    msg('finished!', level=3, silent=silent)

    return args


def make_symmetric(m):
    """Returns a new coo_matrix which is symmetric

    Convenient function to populate a sparse matrix symmetricaly. Only the
    upper triagle of matrix ``m`` has to be defined.

    The rows, cols and values are evaluated such that where ``rows > cols``
    the values will be ignored and recreated from the region where ``cols >
    rows``, in order to obtain a symmetric matrix.

    Parameters
    ----------
    m : array or sparse matrix
        A square matrix with the upper triangle defined.

    Returns
    -------
    m_sym : coo_matrix
        The symmetric sparse matrix.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    triu = c >= r
    r = r[triu]
    c = c[triu]
    v = v[triu]
    pos = r.shape[0]
    r = np.concatenate((r, r*0))
    c = np.concatenate((c, c*0))
    v = np.concatenate((v, v*0))
    triu_no_diag = np.where(c > r)[0]
    r[triu_no_diag + pos] = c[triu_no_diag]
    c[triu_no_diag + pos] = r[triu_no_diag]
    v[triu_no_diag + pos] = v[triu_no_diag]

    return coo_matrix((v, (r, c)), shape=m.shape, dtype=m.dtype)


def make_skew_symmetric(m):
    """Returns a new coo_matrix which is skew-symmetric

    Convenient function to populate a sparse matrix skew-symmetricaly, where
    the off-diagonal elements below the diagonal are negative when compared to
    the terms above the diagonal. Only the upper triagle of matrix ``m`` has
    to be defined.

    The rows, cols and values are evaluated such that where ``rows > cols``
    the values will be ignored and recreated from the region where ``cols >
    rows``, in order to obtain a symmetric matrix.

    Parameters
    ----------
    m : array or sparse matrix
        A square matrix with the upper triangle defined.

    Returns
    -------
    m_sym : coo_matrix
        The symmetric sparse matrix.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    triu = c >= r
    r = r[triu]
    c = c[triu]
    v = v[triu]
    pos = r.shape[0]
    r = np.concatenate((r, r*0))
    c = np.concatenate((c, c*0))
    v = np.concatenate((v, v*0))
    triu_no_diag = np.where(c > r)[0]
    r[triu_no_diag + pos] = c[triu_no_diag]
    c[triu_no_diag + pos] = r[triu_no_diag]
    v[triu_no_diag + pos] = -v[triu_no_diag]

    return coo_matrix((v, (r, c)), shape=m.shape, dtype=m.dtype)


def is_symmetric(m):
    """Check if a sparse matrix is symmetric

    Parameters
    ----------
    m : array or sparse matrix
        A square matrix.

    Returns
    -------
    check : bool
        The check result.

    """
    if m.shape[0] != m.shape[1]:
        raise ValueError('m must be a square matrix')

    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)

    r, c, v = m.row, m.col, m.data
    tril_no_diag = r > c
    triu_no_diag = c > r

    if triu_no_diag.sum() != tril_no_diag.sum():
        return False

    rl = r[tril_no_diag]
    cl = c[tril_no_diag]
    vl = v[tril_no_diag]
    ru = r[triu_no_diag]
    cu = c[triu_no_diag]
    vu = v[triu_no_diag]

    sortl = np.lexsort((cl, rl))
    sortu = np.lexsort((ru, cu))
    vl = vl[sortl]
    vu = vu[sortu]

    check = np.allclose(vl, vu)

    return check


def finalize_symmetric_matrix(M):
    """ Check for nan and inf valus and makes M symmetric
    """
    assert np.any(np.isnan(M.data)) == False
    assert np.any(np.isinf(M.data)) == False
    return csr_matrix(make_symmetric(M))
