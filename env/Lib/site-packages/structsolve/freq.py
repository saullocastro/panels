import numpy as np
import scipy
from scipy.sparse.linalg import eigs
from scipy.linalg import eig

from .logger import msg, warn
from .sparseutils import remove_null_cols


def freq(K, M, tol=0, sparse_solver=True,
        silent=False, sort=True, num_eigvalues=25,
        num_eigvalues_print=5):
    """Frequency Analysis

    Calculate the eigenvalues (`\lambda^2`) and mass-normalized eigenvectors
    solving the following eigenvalue problem::

        [K] + lambda**2 * [M] = 0

    Parameters
    ----------
    K : sparse_matrix
        Stiffness matrix. Should include initial stress stiffness matrix,
        aerodynamic matrix and so forth when applicable.
    M : sparse_matrix
        Mass matrix.
    tol : float, optional
        A tolerance value passed to ``scipy.sparse.linalg.eigs``.
    sparse_solver : bool, optional
        Tells if solver :func:`scipy.linalg.eig` or
        :func:`scipy.sparse.linalg.eigs` should be used.

        .. note:: It is recommended ``nparse_solver=False``, because it
                  was verified that the sparse solver becomes unstable
                  for some cases, though the sparse solver is faster.
    silent : bool, optional
        A boolean to tell whether the log messages should be printed.
    sort : bool, optional
        Sort the output eigenvalues and eigenmodes.
    num_eigvalues : int, optional
        Number of calculated eigenvalues.
    num_eigvalues_print : int, optional
        Number of eigenvalues to print.

    Returns
    -------
    The extracted eigenvalues are stored in the ``eigvals`` parameter and
    the `i^{th}` eigenvector in the ``eigvecs[:, i-1]`` parameter. The
    eigenvectors are mass-normalized.


    """
    msg('Running frequency analysis...', silent=silent)

    msg('Eigenvalue solver... ', level=2, silent=silent)

    k = min(num_eigvalues, M.shape[0]-2)
    if sparse_solver:
        msg('eigs() solver...', level=3, silent=silent)
        sizebkp = M.shape[0]
        Keff, Meff, used_cols = remove_null_cols(K, M, silent=silent,
                level=3)
        #NOTE Looking for better performance with symmetric matrices, I tried
        #     using sparseutils.sparse.is_symmetric and eigsh, but it seems not
        #     to improve speed (I did not try passing only half of the sparse
        #     matrices to the solver)
        eigvals, peigvecs = eigs(A=Keff, M=Meff, k=k, which='LM', tol=tol,
                                 sigma=-1.)
        #NOTE eigs solves: [K] {u} = eigval [M] {u}
        #     therefore we must correct he sign of lambda^2 here:
        lambda2 = -eigvals
        eigvecs = np.zeros((sizebkp, k), dtype=peigvecs.dtype)
        eigvecs[used_cols, :] = peigvecs
    else:
        msg('eig() solver...', level=3, silent=silent)
        if isinstance(M, scipy.sparse.spmatrix):
            Meff = M.toarray()
        else:
            Meff = np.asarray(M)
        if isinstance(K, scipy.sparse.spmatrix):
            Keff = K.toarray()
        else:
            Keff = np.asarray(K)
        sizebkp = Meff.shape[0]
        col_sum = Meff.sum(axis=0)
        check = col_sum != 0
        used_cols = np.arange(Meff.shape[0])[check]
        Meff = Meff[:, check][check, :]
        Keff = Keff[:, check][check, :]

        #TODO did not try using eigh when input is symmetric to see if there
        #     will be speed improvements
        # for effiency reasons, solving:
        #    [M]{u} = (-1/lambda2)[K]{u}
        #    [M]{u} = eigval [K]{u}
        eigvals, peigvecs = eig(a=Meff, b=Keff)
        lambda2 = -1./eigvals
        eigvecs = np.zeros((sizebkp, Keff.shape[0]),
                           dtype=peigvecs.dtype)
        eigvecs[check, :] = peigvecs

    msg('finished!', level=3, silent=silent)

    if sort:
        omegan = np.sqrt(-lambda2)
        sort_ind = np.lexsort((np.round(omegan.imag, 1),
                               np.round(omegan.real, 1)))
        omegan = omegan[sort_ind]
        eigvecs = eigvecs[:, sort_ind]

        higher_zero = omegan.real > 1e-6

        omegan = omegan[higher_zero]
        eigvecs = eigvecs[:, higher_zero]

    msg('finished!', level=2, silent=silent)

    msg('first {0} eigenvalues:'.format(num_eigvalues_print), level=1,
        silent=silent)
    for lambda2i in lambda2[:num_eigvalues_print]:
        msg('lambda**2: %1.5f, natural frequency: %1.5f rad/s' % (lambda2i, (-lambda2i)**0.5), level=2, silent=silent)

    return lambda2, eigvecs
