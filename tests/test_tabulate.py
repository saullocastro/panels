r"""The approximation functions of the numerical kernels are tabulated

See :mod:`panels.dev.tabulate` and ``doc/source/tabulation.rst``. The
results of the tabulated kernels are those of the previous implementation,
bit for bit; here the rewriting itself is checked, and that no kernel of
the package evaluates the approximation functions in its loops over the
terms any more.
"""
import os
import re
import subprocess

import pytest

from panels.dev.tabulate import tabulate_basis, MARK

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
