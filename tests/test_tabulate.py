r"""The approximation functions of the numerical kernels are tabulated

See :mod:`panels.dev.tabulate`. Here the rewriting itself is checked, also
the one that keeps the tables on the shell across calls, against the
expected lines of code, and that no kernel of the package evaluates the
approximation functions in its loops over the terms.
"""
import os
import re
import subprocess

import pytest

from panels.dev.tabulate import tabulate_basis, keep_tables, MARK, IMPORT

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

KERNEL = '''def fk(object p, int size, int nx, int ny):
    cdef int i, j, k, l, ptx, pty
    cdef double xi, eta, fA, fB, gA, gB_eta
    cdef double [::1] xis, etas, weights_xi, weights_eta
    m = p.m
    n = p.n
    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                for i in range(m):
                    fA = f(i, xi, x1u, x1ur, x2u, x2ur)
                    for k in range(m):
                        fB = fp(k, xi, x1w, x1wr, x2w, x2wr)
                        for j in range(n):
                            gA = f(j, eta, y1u, y1ur, y2u, y2ur)
                            for l in range(n):
                                gB_eta = fp(l, eta, y1w, y1wr, y2w, y2wr)
                                out += fA*fB*gA*gB_eta
'''

LINE_KERNEL = '''def fkline(object p1, double xcte1, int size):
    cdef int i1, k1
    cdef double xicte1, f1A, f1B
    m1 = p1.m
    xicte1 = 2*xcte1/p1.a - 1.
    with nogil:
        for i1 in range(m1):
            f1A = f(i1, xicte1, x1u1, x1ur1, x2u1, x2ur1)
            for k1 in range(m1):
                f1B = f(k1, xicte1, x1u1, x1ur1, x2u1, x2ur1)
'''


def test_kernel_rewritten():
    out = tabulate_basis(KERNEL)
    assert MARK in out
    assert '    cdef double [:, ::1] ftab, gtab' in out
    assert '    ftab = np.zeros((2, m), dtype=DOUBLE)' in out
    assert '    gtab = np.zeros((2, n), dtype=DOUBLE)' in out
    for line in ('fA = ftab[0, i]', 'fB = ftab[1, k]', 'gA = gtab[0, j]',
                 'gB_eta = gtab[1, l]',
                 'ftab[1, i] = fp(i, xi, x1w, x1wr, x2w, x2wr)',
                 'gtab[1, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)'):
        assert line in out, line
    # the tables are filled in the loop over the points, after the
    # coordinates and before the loops over the terms
    lines = out.split('\n')
    fill = lines.index(' '*16 + MARK)
    assert lines[fill - 1].strip() == 'eta = etas[pty]'
    assert lines[fill + 1].strip() == 'for i in range(m):'
    # already tabulated: unchanged
    assert tabulate_basis(out) == out


def test_line_kernel_tables_filled_once():
    out = tabulate_basis(LINE_KERNEL)
    assert 'f1A = ftab1[0, i1]' in out and 'f1B = ftab1[0, k1]' in out
    lines = out.split('\n')
    fill = lines.index(' '*8 + MARK)
    assert lines[fill - 1].strip() == 'with nogil:'


def test_unexpected_form_raises():
    with pytest.raises(ValueError, match='expected form'):
        tabulate_basis(KERNEL.replace('f(j, eta, y1u, y1ur, y2u, y2ur)',
                                      'f(j, eta, y1u, y1ur, y2v, y2ur)'))
    # a coordinate changed after the tables are filled
    with pytest.raises(ValueError, match='assigned after'):
        tabulate_basis(KERNEL.replace('out += fA*fB*gA*gB_eta',
                                      'xi = 0.'))


SHELL_KERNEL = '''from panels import INT, DOUBLE


def fk(object shell, int size, int nx, int ny):
    cdef int i, j, k, l, ptx, pty
    cdef double xi, eta, fA, fB, gA, gB_eta, a
    cdef double [::1] xis, etas, weights_xi, weights_eta
    m = shell.m
    n = shell.n
    a = shell.a
    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                xi = (xi + 1)/2*(xi2 - xi1) + xi1
                eta = etas[pty]
                for i in range(m):
                    fA = f(i, xi, x1u, x1ur, x2u, x2ur)
                    for k in range(m):
                        fB = fp(k, xi, x1w, x1wr, x2w, x2wr)
                        for j in range(n):
                            gA = f(j, eta, y1u, y1ur, y2u, y2ur)
                            for l in range(n):
                                gB_eta = fp(l, eta, y1w, y1wr, y2w, y2wr)
                                out += fA*fB*gA*gB_eta
    return out
'''


def test_tables_kept_across_calls():
    """A kernel with a shell keeps its tables on the shell, see
    :func:`panels.dev.tabulate.keep_tables`: one row per point of the loop
    that the coordinate depends on, the key of every value they depend on,
    filled only when new, stored after the nogil block"""
    out = tabulate_basis(SHELL_KERNEL)
    assert IMPORT in out
    for line in (
            '    cdef double [:, :, ::1] ftab',
            '    cdef int ftab_new',
            '    ftab_key = (nx, m, x1u, x1ur, x2u, x2ur, x1w, x1wr, x2w, '
            'x2wr, xis, xi2, xi1)',
            "    ftab, ftab_new = lookup_table(shell, __name__, 'fk', 'ftab', "
            "ftab_key, (nx, 2, m))",
            '    gtab_key = (ny, n, y1u, y1ur, y2u, y2ur, y1w, y1wr, y2w, '
            'y2wr, etas)',
            '                if ftab_new:',
            '                        ftab[ptx, 1, i] = fp(i, xi, x1w, x1wr, '
            'x2w, x2wr)',
            '                        gtab[pty, 0, j] = f(j, eta, y1u, y1ur, '
            'y2u, y2ur)',
            '                    fA = ftab[ptx, 0, i]',
            '                                gB_eta = gtab[pty, 1, l]',
            '    if ftab_new:',
            "        store_table(shell, __name__, 'fk', 'ftab', ftab_key, "
            "ftab)"):
        assert line in out.split('\n'), line
    assert 'np.zeros' not in out
    # the store after the nogil block, before the return
    lines = out.split('\n')
    assert lines.index('    return out') > lines.index(
        "        store_table(shell, __name__, 'fk', 'gtab', gtab_key, gtab)")
    # already rewritten: unchanged
    assert tabulate_basis(out) == out
    assert keep_tables(out) == out


def test_tables_not_kept():
    """Tables of a kernel without a shell, e.g. of the connections, or
    filled outside the loops over the points are filled at every call"""
    assert keep_tables(tabulate_basis(KERNEL)) == tabulate_basis(KERNEL)
    out = tabulate_basis(LINE_KERNEL)
    assert keep_tables(out.replace('object p1', 'object shell')) == \
        out.replace('object p1', 'object shell')


def test_unknown_dependence_raises():
    """The key of a table is not established: a coordinate assigned
    conditionally, depending on the previous point, or a value modified in
    the nogil block"""
    with pytest.raises(ValueError, match='conditionally'):
        tabulate_basis(SHELL_KERNEL.replace(
            '                xi = (xi + 1)/2*(xi2 - xi1) + xi1',
            '                if pty > 0:\n'
            '                    xi = (xi + 1)/2*(xi2 - xi1) + xi1'))
    with pytest.raises(ValueError, match='previous point'):
        tabulate_basis(SHELL_KERNEL.replace(
            '                xi = xis[ptx]\n', ''))
    with pytest.raises(ValueError, match='modified in the nogil'):
        tabulate_basis(SHELL_KERNEL.replace(
            '                                out += fA*fB*gA*gB_eta',
            '                                x1u = 0.'))


@pytest.mark.slow
def test_no_kernel_evaluates_the_functions_in_its_loops():
    """Every call of an approximation function of the package fills a
    table, except in the field outputs, which evaluate the fields once per
    point of the grid"""
    try:
        files = subprocess.check_output(['git', 'ls-files', '*.pyx'],
                                        cwd=ROOT, text=True).split()
    except (OSError, subprocess.CalledProcessError):
        files = []
    if not files:
        pytest.skip('the sources are not in a git repository')
    call = re.compile(r'^\s*[^#]*\b\w+ = (?:fpp|fp|f)\(', flags=re.M)
    for path in files:
        if path.endswith('_field.pyx'):
            continue
        with open(os.path.join(ROOT, path)) as f:
            text = f.read()
        found = call.findall(text)
        assert not found, (path, found[:2])
