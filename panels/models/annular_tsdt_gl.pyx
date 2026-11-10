#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Annular-sector flat plate, using Reddy's third-order shear deformation
theory (TSDT) with Green-Lagrange membrane strains in polar coordinates

Matrices of the whole domain of the model ``'annular_tsdt_gl'``,
called by :class:`.Shell` when the constitutive matrix and the density are
constant: :func:`fk0`, :func:`fkG0` and :func:`fkM`. The strains have
coefficients in `1/\rho`, with `\rho = r - x` the radius of the point, and
the matrices are integrated with the Gauss-Legendre rule of
:mod:`panels.models.annular_tsdt_gl_num`, with ``shell.nx`` and
``shell.ny`` points; the piston-theory matrices ``fkAx``, ``fkAy`` and
``fcA`` are not available.

Notes
-----
See :mod:`panels.models.annular_tsdt_gl_num`.

"""
import numpy as np

from panels import DOUBLE
from .annular_tsdt_gl_num import fkC_num, fkG_num, fkM_num


def fk0(object shell, int size, int row0, int col0):
    r"""Linear constitutive stiffness matrix of the whole panel

    Parameters
    ----------
    shell : :class:`.Shell`
        The annular sector, with its constitutive matrix ``shell.ABD`` and
        its numbers of integration points ``shell.nx`` and ``shell.ny``.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the terms of this panel start.

    Returns
    -------
    kC0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``, upper triangle.

    Notes
    -----
    :func:`.annular_tsdt_gl_num.fkC_num` with ``NLgeom=0``.

    """
    return fkC_num(np.zeros(size, dtype=DOUBLE), shell.ABD, shell, size,
                   row0, col0, shell.nx, shell.ny, 0)


def fkG0(double Nxx, double Nyy, double Nxy, object shell,
         int size, int row0, int col0):
    r"""Geometric stiffness matrix of a constant membrane stress state

    Parameters
    ----------
    Nxx, Nyy, Nxy : float
        Membrane stress resultants along the local axes, `N_{xx} =
        N_{\rho\rho}`, `N_{yy} = N_{\theta\theta}`, `N_{xy} =
        -N_{\rho\theta}`, force per unit length.
    shell : :class:`.Shell`
        The annular sector, with its numbers of integration points
        ``shell.nx`` and ``shell.ny``.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the terms of this panel start.

    Returns
    -------
    kG0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``, upper triangle.

    Notes
    -----
    :func:`.annular_tsdt_gl_num.fkG_num` with zero Ritz constants.

    """
    return fkG_num(np.zeros(size, dtype=DOUBLE), shell.ABD, shell, size,
                   row0, col0, shell.nx, shell.ny, Nxx, Nyy, Nxy)


def fkM(object shell, int size, int row0, int col0):
    r"""Mass matrix of the whole panel

    Parameters
    ----------
    shell : :class:`.Shell`
        The annular sector, with its laminate ``shell.lam`` and its numbers
        of integration points ``shell.nx`` and ``shell.ny``.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the terms of this panel start.

    Returns
    -------
    kM : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``, upper triangle.

    Notes
    -----
    :func:`.annular_tsdt_gl_num.fkM_num` with the moments ``intrho``,
    ``intrhoz``, ``intrhoz2``, ``intrhoz3``, ``intrhoz4``, ``intrhoz6`` of
    the laminate about its mid-surface, :meth:`.Shell._get_lam_mid`, and the
    thickness of ``shell.lam``; the offset ``shell.offset`` enters the
    kernel.

    """
    mid = shell._get_lam_mid()
    I_input = np.array([mid.intrho, mid.intrhoz, mid.intrhoz2, mid.intrhoz3,
                        mid.intrhoz4, mid.intrhoz6, shell.lam.h], dtype=DOUBLE)
    return fkM_num(shell, I_input, size, row0, col0, shell.nx, shell.ny)

def fkAx(*args, **kwargs):
    r"""Not available for the annular sector

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('fkAx is not implemented for the model '
                              '"annular_tsdt_gl"')

def fkAy(*args, **kwargs):
    r"""Not available for the annular sector

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('fkAy is not implemented for the model '
                              '"annular_tsdt_gl"')

def fcA(*args, **kwargs):
    r"""Not available for the annular sector

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('fcA is not implemented for the model '
                              '"annular_tsdt_gl"')
