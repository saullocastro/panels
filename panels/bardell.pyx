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

Notes
-----
The functions are those applied by Bardell [1]_ in the hierarchical finite
element method, and used as approximation functions of the Ritz method of
panels [2]_. The first four terms are the cubic Hermite polynomials,
multiplied by the flags ``x1t, x1r, x2t, x2r``:

.. math::

    P_0 = x_{1t} \left(\frac{1}{2} - \frac{3}{4}\xi + \frac{1}{4}\xi^3\right)
    \qquad
    P_1 = x_{1r} \frac{1}{8}\left(1 - \xi - \xi^2 + \xi^3\right)

    P_2 = x_{2t} \left(\frac{1}{2} + \frac{3}{4}\xi - \frac{1}{4}\xi^3\right)
    \qquad
    P_3 = x_{2r} \frac{1}{8}\left(-1 - \xi + \xi^2 + \xi^3\right)

such that, for unit flags, `P_0(-1) = 1` and `P_1'(-1) = 1/2` are the only
non-zero values and first derivatives of `P_0` and `P_1` at the ends, and
`P_2(+1) = 1`, `P_3'(+1) = 1/2` those of `P_2` and `P_3`. The terms
`i \ge 4` are the Legendre-based polynomials in Rodrigues form

.. math::

    P_i = \sum_{n=0}^{\lfloor i/2 \rfloor} \frac{(-1)^n (2i - 2n - 5)!!}
          {2^n \, n! \, (i - 2n)!} \xi^{i - 2n}

with `(-1)!! = 1`, e.g. `P_4 = (\xi^2 - 1)^2/8`, which vanish together with
their first derivative at both ends, `\xi = \pm 1`. The set is therefore
hierarchical: adding terms does not change the lower ones, and only the
first four terms carry the value and the derivative at the ends. A flag
``0.`` removes its term, which makes the value (``x1t``, ``x2t``) or the
first derivative (``x1r``, ``x2r``) of every function zero at that end, an
essential boundary condition imposed exactly; any non-zero flag keeps the
term, only scaling it. For example, ``x1t = 0, x1r = 1`` gives a simply
supported end at `\xi = -1` for `w`, ``x1t = x1r = 0`` a clamped end and
``x1t = x1r = 1`` a free end.

References
----------
.. [1] Bardell, N. S., "Free vibration analysis of a flat plate using the
   hierarchical finite element method", Journal of Sound and Vibration,
   1991. https://doi.org/10.1016/0022-460X(91)90855-E
.. [2] Castro, S. G. P., and Donadon, M. V., "Assembly of Semi-Analytical
   models to Address Linear Buckling and Vibration of Stiffened Composite
   Panels with Debonding Defect", Composite Structures, 2017.
   https://doi.org/10.1016/j.compstruct.2016.10.026

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
    r"""Integral of `P_i P_j` over `-1 \le \xi \le 1`

    `P_i` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and `P_j` the term ``j`` with the flags
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

    Notes
    -----
    The integrals of the six functions ``calc_integral_*`` are those of the
    closed-form structural matrices, which separate the two directions of
    the approximation functions, see :mod:`panels.bardell`. With `x = (\xi +
    1) a/2`, an integral over the whole length `0 \le x \le a` of a product
    of the functions and of their derivatives with respect to `x` is

    .. math::

        \int_0^a \frac{d^p P_i}{dx^p} \frac{d^q P_j}{dx^q} dx =
        \left(\frac{2}{a}\right)^{p + q} \frac{a}{2} \int_{-1}^{1}
        \frac{d^p P_i}{d\xi^p} \frac{d^q P_j}{d\xi^q} d\xi

    The functions being polynomials, the integrals over `-1 \le \xi \le 1`
    are exact.

    """
    return integral_ff(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_ffp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                          double y1t, double y1r, double y2t, double y2r):
    r"""Integral of `P_i P_j'` over `-1 \le \xi \le 1`

    `P_i` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and `P_j` the term ``j`` with the flags
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

    Notes
    -----
    See the notes of :func:`calc_integral_ff`.

    """
    return integral_ffp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_ffpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                           double y1t, double y1r, double y2t, double y2r):
    r"""Integral of `P_i P_j''` over `-1 \le \xi \le 1`

    `P_i` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and `P_j` the term ``j`` with the flags
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

    Notes
    -----
    See the notes of :func:`calc_integral_ff`.

    """
    return integral_ffpp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_fpfp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                           double y1t, double y1r, double y2t, double y2r):
    r"""Integral of `P_i' P_j'` over `-1 \le \xi \le 1`

    `P_i` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and `P_j` the term ``j`` with the flags
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

    Notes
    -----
    See the notes of :func:`calc_integral_ff`.

    """
    return integral_fpfp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_fpfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                            double y1t, double y1r, double y2t, double y2r):
    r"""Integral of `P_i' P_j''` over `-1 \le \xi \le 1`

    `P_i` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and `P_j` the term ``j`` with the flags
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

    Notes
    -----
    See the notes of :func:`calc_integral_ff`.

    """
    return integral_fpfpp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)
def calc_integral_fppfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                             double y1t, double y1r, double y2t, double y2r):
    r"""Integral of `P_i'' P_j''` over `-1 \le \xi \le 1`

    `P_i` is the term ``i`` of Bardell's functions with the flags
    ``x1t, x1r, x2t, x2r`` and `P_j` the term ``j`` with the flags
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

    Notes
    -----
    See the notes of :func:`calc_integral_ff`.

    """
    return integral_fppfpp(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r)

