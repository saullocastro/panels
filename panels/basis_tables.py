r"""
Tables of the approximation functions kept per shell across calls
(:mod:`panels.basis_tables`)
==================================================================

.. currentmodule:: panels.basis_tables

The numerically integrated kernels of :mod:`panels.models` evaluate the
approximation functions of Bardell, and their derivatives, once per
integration point and term, in tables such as ``ftab`` along `x` and
``gtab`` along `y`. The integration points of a shell do not change from
one call to the next, nor do the numbers of terms or the boundary flags,
such that these tables are also the same from one call to the next. The
kernels therefore keep them on the shell, one table per kernel, filled at
the first call and only read afterwards, for every point of the integration
domain: a table along `x` has the shape ``(nx, rows, m)`` and one along `y`
the shape ``(ny, rows, n)``.

Each table is stored with the key of all the values it depends on: the
numbers of terms and of integration points, the Gauss-Legendre abscissae,
the limits of the integration domain in natural coordinates, the boundary
flags of the fields of its rows and the dimensions entering its rows, e.g.
``2/a`` of a derivative. A kernel whose key differs from the stored one,
e.g. after a change of ``m``, of a boundary flag or of
:meth:`.Shell.integration_limits`, fills a new table and replaces the stored
one, such that a shell keeps at most one table per kernel and table name.

The kernels call :func:`lookup_table` and :func:`store_table`, inserted by
:func:`panels.dev.tabulate.tabulate_basis`; :func:`clear_tables` empties the
tables of a shell, e.g. to release their memory.

Notes
-----
The values in a stored table are computed by the same code of the kernel,
at the same coordinates, as in a kernel that fills its tables at every
call: the matrices and vectors are the same, bit for bit.

The tables are created and stored while the kernel holds the GIL, before
and after its ``nogil`` block. A table is filled only while it is new and
not yet stored, and a stored table is never modified, only replaced: a
kernel running in another thread keeps reading the table it started with.

"""
import numpy as np

from panels import DOUBLE


def _normalized(value):
    """A hashable value of an item of a key, exact to the bit"""
    if isinstance(value, bool):
        return ('b', value)
    if isinstance(value, int):
        return ('i', value)
    if isinstance(value, float):
        # float.hex distinguishes -0. from 0. and compares nan equal to nan
        return ('f', value.hex())
    arr = np.asarray(value)
    return ('a', arr.dtype.str, arr.shape, arr.tobytes())


def lookup_table(shell, module, function, table, key, shape):
    r"""Table of approximation functions of a kernel, stored or new

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell whose tables are looked up.
    module, function, table : str
        Names of the module of the kernel, of the kernel and of the table,
        e.g. ``'panels.models.plate_clpt_donnell_num'``, ``'fkC_num'`` and
        ``'ftab'``, which identify the code that fills the table.
    key : tuple
        The values the table depends on, numbers or arrays, see the module
        docstring.
    shape : tuple of int
        Shape of the table, ``(points, rows, terms)``.

    Returns
    -------
    tab : numpy.ndarray
        The stored table when it has the same key, otherwise a new,
        uninitialized array of shape ``shape`` and dtype ``DOUBLE``, which
        the kernel fills and stores with :func:`store_table`.
    new : int
        ``1`` when ``tab`` is new and must be filled, ``0`` when it is the
        stored table.

    """
    tables = getattr(shell, '_basis_tables', None)
    if tables is not None:
        stored = tables.get((module, function, table))
        if (stored is not None and stored[1].shape == tuple(shape)
                and stored[0] == tuple(_normalized(v) for v in key)):
            return stored[1], 0
    return np.empty(shape, dtype=DOUBLE), 1


def store_table(shell, module, function, table, key, tab):
    r"""Store a table filled by a kernel, replacing the stored one

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell that keeps the table. Nothing is stored for an object
        without the attribute ``_basis_tables``.
    module, function, table : str
        As in :func:`lookup_table`.
    key : tuple
        As in :func:`lookup_table`.
    tab : numpy.ndarray
        The filled table, which is not modified afterwards.

    """
    try:
        tables = shell._basis_tables
    except AttributeError:
        tables = None
    if tables is None:
        try:
            shell._basis_tables = tables = {}
        except AttributeError:
            return
    # replacing the entry, never modifying a stored table, keeps the tables
    # read by kernels running in other threads valid
    tables[(module, function, table)] = (tuple(_normalized(v) for v in key),
                                         tab)


def clear_tables(shell):
    r"""Remove the tables of the approximation functions of a shell

    The kernels fill them again at their next call, with the same values.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell.

    """
    try:
        shell._basis_tables = None
    except AttributeError:
        pass
