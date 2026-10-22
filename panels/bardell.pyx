#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: profile=False
#cython: infer_types=False
r"""Bardell's hierarchical functions (:mod:`panels.bardell`)

Python access to the C++ implementation of Bardell's functions used by the
kernels: the values and the first and second derivatives of the 30 terms at
a natural coordinate `-1 \le \xi \le 1`, and the integrals over
`-1 \le \xi \le 1` of the products of two terms or of their derivatives.
The boundary conditions enter through the flags that multiply the first
four terms.

"""
import numpy as np

NUM = 30 # maximum number of terms currently implemented
DOUBLE = np.float64


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double x1t, double x1r, double x2t, double x2r) nogil
    double fp(int i, double xi, double x1t, double x1r, double x2t, double x2r) nogil
    double fpp(int i, double xi, double x1t, double x1r, double x2t, double x2r) nogil
    void vec_f(double *f, double xi, double x1t, double x1r, double x2t, double x2r) nogil
    void vec_fp(double *f, double xi, double x1t, double x1r, double x2t, double x2r) nogil
    void vec_fpp(double *f, double xi, double x1t, double x1r, double x2t, double x2r) nogil


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fppfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil


def calc_f(int i, double xi, double x1t, double x1r, double x2t, double x2r):
    r"""Value of the term ``i`` of Bardell's functions at ``xi``

    Bardell's hierarchical functions are defined over the natural
    coordinate `-1 \le \xi \le 1`.

    Parameters
    ----------
    i : int
        Index of the term, ``0`` to ``29``; a larger index gives ``0.``.
    xi : float
        Natural coordinate `\xi`, dimensionless.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.

    Returns
    -------
    f : float
        The value, dimensionless.

    See Also
    --------
    calc_vec_f

    """
    return f(i, xi, x1t, x1r, x2t, x2r)
def calc_fp(int i, double xi, double x1t, double x1r, double x2t, double x2r):
    r"""First derivative of the term ``i`` of Bardell's functions at ``xi``

    Bardell's hierarchical functions are defined over the natural
    coordinate `-1 \le \xi \le 1`. The derivative is with respect to `\xi`.

    Parameters
    ----------
    i : int
        Index of the term, ``0`` to ``29``; a larger index gives ``0.``.
    xi : float
        Natural coordinate `\xi`, dimensionless.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.

    Returns
    -------
    fp : float
        The value, dimensionless.

    See Also
    --------
    calc_vec_fp

    """
    return fp(i, xi, x1t, x1r, x2t, x2r)
def calc_fpp(int i, double xi, double x1t, double x1r, double x2t, double x2r):
    r"""Second derivative of the term ``i`` of Bardell's functions at ``xi``

    Bardell's hierarchical functions are defined over the natural
    coordinate `-1 \le \xi \le 1`. The derivative is with respect to `\xi`.

    Parameters
    ----------
    i : int
        Index of the term, ``0`` to ``29``; a larger index gives ``0.``.
    xi : float
        Natural coordinate `\xi`, dimensionless.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.

    Returns
    -------
    fpp : float
        The value, dimensionless.

    See Also
    --------
    calc_vec_fpp

    """
    return fpp(i, xi, x1t, x1r, x2t, x2r)
def calc_vec_f(double xi, double x1t, double x1r, double x2t, double x2r):
    r"""Values of the 30 terms of Bardell's functions at ``xi``

    See :func:`calc_f`.

    Parameters
    ----------
    xi : float
        Natural coordinate `\xi`, `-1 \le \xi \le 1`, dimensionless.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.

    Returns
    -------
    out : np.ndarray
        1-D array of size ``30``, the entry ``i`` being the term ``i``.

    """
    cdef double [::1] out = np.zeros(NUM, dtype=DOUBLE)
    vec_f(&out[0], xi, x1t, x1r, x2t, x2r)
    return np.asarray(out)
def calc_vec_fp(double xi, double x1t, double x1r, double x2t, double x2r):
    r"""First derivatives of the 30 terms of Bardell's functions at ``xi``

    The derivatives are with respect to the natural coordinate `\xi`, see
    :func:`calc_fp`.

    Parameters
    ----------
    xi : float
        Natural coordinate `\xi`, `-1 \le \xi \le 1`, dimensionless.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.

    Returns
    -------
    out : np.ndarray
        1-D array of size ``30``, the entry ``i`` being the term ``i``.

    """
    cdef double [::1] out = np.zeros(NUM, dtype=DOUBLE)
    vec_fp(&out[0], xi, x1t, x1r, x2t, x2r)
    return np.asarray(out)
def calc_vec_fpp(double xi, double x1t, double x1r, double x2t, double x2r):
    r"""Second derivatives of the 30 terms of Bardell's functions at ``xi``

    The derivatives are with respect to the natural coordinate `\xi`, see
    :func:`calc_fpp`.

    Parameters
    ----------
    xi : float
        Natural coordinate `\xi`, `-1 \le \xi \le 1`, dimensionless.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.

    Returns
    -------
    out : np.ndarray
        1-D array of size ``30``, the entry ``i`` being the term ``i``.

    """
    cdef double [::1] out = np.zeros(NUM, dtype=DOUBLE)
    vec_fpp(&out[0], xi, x1t, x1r, x2t, x2r)
    return np.asarray(out)


def calc_integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                         double y1t, double y1r, double y2t, double y2r):
    r"""Integral of ``f_i*f_j`` over `-1 \le \xi \le 1`

    ``f_i`` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and ``f_j`` the term ``j`` with the flags
    ``y1t, y1r, y2t, y2r``; the primes are derivatives with respect to the
    natural coordinate `\xi`. The values are tabulated in closed form.

    Parameters
    ----------
    i, j : int
        Indices of the terms, ``0`` to ``29``; a larger index gives ``0.``.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.
    y1t, y1r, y2t, y2r : float
        The same flags for the function of the term ``j``. Dimensionless.

    Returns
    -------
    integral : float
        The value of the integral, dimensionless.

    """
    return integral_ff(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_ffp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                          double y1t, double y1r, double y2t, double y2r):
    r"""Integral of ``f_i*f_j'`` over `-1 \le \xi \le 1`

    ``f_i`` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and ``f_j`` the term ``j`` with the flags
    ``y1t, y1r, y2t, y2r``; the primes are derivatives with respect to the
    natural coordinate `\xi`. The values are tabulated in closed form.

    Parameters
    ----------
    i, j : int
        Indices of the terms, ``0`` to ``29``; a larger index gives ``0.``.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.
    y1t, y1r, y2t, y2r : float
        The same flags for the function of the term ``j``. Dimensionless.

    Returns
    -------
    integral : float
        The value of the integral, dimensionless.

    """
    return integral_ffp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_ffpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                           double y1t, double y1r, double y2t, double y2r):
    r"""Integral of ``f_i*f_j''`` over `-1 \le \xi \le 1`

    ``f_i`` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and ``f_j`` the term ``j`` with the flags
    ``y1t, y1r, y2t, y2r``; the primes are derivatives with respect to the
    natural coordinate `\xi`. The values are tabulated in closed form.

    Parameters
    ----------
    i, j : int
        Indices of the terms, ``0`` to ``29``; a larger index gives ``0.``.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.
    y1t, y1r, y2t, y2r : float
        The same flags for the function of the term ``j``. Dimensionless.

    Returns
    -------
    integral : float
        The value of the integral, dimensionless.

    """
    return integral_ffpp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_fpfp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                           double y1t, double y1r, double y2t, double y2r):
    r"""Integral of ``f_i'*f_j'`` over `-1 \le \xi \le 1`

    ``f_i`` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and ``f_j`` the term ``j`` with the flags
    ``y1t, y1r, y2t, y2r``; the primes are derivatives with respect to the
    natural coordinate `\xi`. The values are tabulated in closed form.

    Parameters
    ----------
    i, j : int
        Indices of the terms, ``0`` to ``29``; a larger index gives ``0.``.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.
    y1t, y1r, y2t, y2r : float
        The same flags for the function of the term ``j``. Dimensionless.

    Returns
    -------
    integral : float
        The value of the integral, dimensionless.

    """
    return integral_fpfp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_fpfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                            double y1t, double y1r, double y2t, double y2r):
    r"""Integral of ``f_i'*f_j''`` over `-1 \le \xi \le 1`

    ``f_i`` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and ``f_j`` the term ``j`` with the flags
    ``y1t, y1r, y2t, y2r``; the primes are derivatives with respect to the
    natural coordinate `\xi`. The values are tabulated in closed form.

    Parameters
    ----------
    i, j : int
        Indices of the terms, ``0`` to ``29``; a larger index gives ``0.``.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.
    y1t, y1r, y2t, y2r : float
        The same flags for the function of the term ``j``. Dimensionless.

    Returns
    -------
    integral : float
        The value of the integral, dimensionless.

    """
    return integral_fpfpp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_fppfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                             double y1t, double y1r, double y2t, double y2r):
    r"""Integral of ``f_i''*f_j''`` over `-1 \le \xi \le 1`

    ``f_i`` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and ``f_j`` the term ``j`` with the flags
    ``y1t, y1r, y2t, y2r``; the primes are derivatives with respect to the
    natural coordinate `\xi`. The values are tabulated in closed form.

    Parameters
    ----------
    i, j : int
        Indices of the terms, ``0`` to ``29``; a larger index gives ``0.``.
    x1t, x1r, x2t, x2r : float
        Flags of the boundary conditions, multiplying the functions of the
        terms ``0`` and ``1`` (value and derivative at `\xi = -1`) and ``2``
        and ``3`` (value and derivative at `\xi = +1`); ``0.`` removes and
        ``1.`` keeps that term. Dimensionless.
    y1t, y1r, y2t, y2r : float
        The same flags for the function of the term ``j``. Dimensionless.

    Returns
    -------
    integral : float
        The value of the integral, dimensionless.

    """
    return integral_fppfpp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)

