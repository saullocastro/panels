"""Expressions of the matrices of the Sanders cylinder

The expressions are written to ``sanders_exprs.py``, which is read by
``write_pyx.py`` to write ``panels/models/cylshell_clpt_sanders{,_num}.pyx``
"""
import os
import re
import sys
import time

import numpy as np
import sympy
from sympy import Add, Mul, Symbol, Rational, Float, expand, collect, factor
from sympy.printing.str import StrPrinter

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'theory', 'shells', 'cylshell_clpt_sanders'))

import cylshell_clpt_sanders as th
from panels.dev.matrixtools import pow2mult

ABD = th.ABD


class Printer(StrPrinter):
    def _print_Float(self, e):
        return repr(float(e))


def floatify(e):
    reps = {q: Float(q) for q in e.atoms(Rational) if not q.is_Integer}
    return e.xreplace(reps)


def pstr(e):
    return pow2mult(Printer().doprint(floatify(e)))


def simp(e, vars=ABD):
    e = expand(e)
    return collect(e, vars, func=factor)


BASIS = re.compile(r'^([fg])([AB])([uvw])((?:xi|eta)*)$')


def basis_info(name):
    mo = BASIS.match(name)
    if mo is None:
        return None
    fg, side, field, der = mo.groups()
    order = der.count('xi') if fg == 'f' else der.count('eta')
    return fg, side, field, order


# integration of products of Bardell functions
KIND = {(0, 0): 'ff', (0, 1): 'ffp', (0, 2): 'ffpp',
        (1, 1): 'fpfp', (1, 2): 'fpfpp', (2, 2): 'fppfpp'}


def integral_name(fg, pA, dA, pB, dB):
    d = 'xi' if fg == 'f' else 'eta'
    return '{0}A{1}{2}{0}B{3}{4}'.format(fg, pA, d*dA, pB, d*dB)


def integral_call(name):
    fg = name[0]
    d = 'xi' if fg == 'f' else 'eta'
    mo = re.match(r'^{0}A([uvw])((?:{1})*){0}B([uvw])((?:{1})*)$'.format(fg, d), name)
    pA, dA, pB, dB = mo.groups()
    dA = dA.count(d)
    dB = dB.count(d)
    i, k = ('i', 'k') if fg == 'f' else ('j', 'l')
    xy = 'x' if fg == 'f' else 'y'
    bc = lambda p: '{0}1{1}, {0}1{1}r, {0}2{1}, {0}2{1}r'.format(xy, p)
    if dA <= dB:
        return 'integral_{0}({1}, {2}, {3}, {4})'.format(KIND[(dA, dB)], i, k, bc(pA), bc(pB))
    else:
        return 'integral_{0}({2}, {1}, {4}, {3})'.format(KIND[(dB, dA)], i, k, bc(pA), bc(pB))


def to_integrals(e):
    """Replace fA*fB*gA*gB by the symbols of their integrals"""
    e = expand(e)
    out = 0
    for term in Add.make_args(e):
        coeff = 1
        f = {}
        g = {}
        for fac in Mul.make_args(term):
            if fac.is_Symbol and basis_info(fac.name):
                fg, side, field, order = basis_info(fac.name)
                (f if fg == 'f' else g)[side] = (field, order)
            else:
                coeff *= fac
        assert len(f) == 2 and len(g) == 2, term
        fi = Symbol(integral_name('f', *f['A'], *f['B']))
        gi = Symbol(integral_name('g', *g['A'], *g['B']))
        out += coeff*fi*gi
    return out


def entries(m):
    return [(i, j, v) for (i, j), v in np.ndenumerate(np.array(m)) if v != 0]


def cdefs(names, per_line=6):
    names = list(names)
    out = []
    for i in range(0, len(names), per_line):
        out.append('    cdef double ' + ', '.join(names[i:i + per_line]))
    return '\n'.join(out)


BC_BLOCK = """    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr"""

BC_CDEF = """    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr"""

ABD_CDEF = """    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66"""


def ordered_integrals(names):
    return sorted(names, key=lambda s: (s[:2], s))


def analytical_function(header, cdef_extra, prelude, name, m, nnz_note=''):
    """Analytical matrix with the loop structure of cylshell_clpt_donnell"""
    ent = entries(m)
    ent = [(i, j, to_integrals(v)) for i, j, v in ent]
    fnames = set()
    gnames = set()
    for _, _, v in ent:
        for s in v.free_symbols:
            if re.match(r'^fA[uvw]', s.name):
                fnames.add(s.name)
            elif re.match(r'^gA[uvw]', s.name):
                gnames.add(s.name)
    fnames = ordered_integrals(fnames)
    gnames = ordered_integrals(gnames)
    lines = []
    lines.append(header)
    lines.append(cdef_extra)
    lines.append('')
    lines.append('    cdef long [:] {0}r, {0}c'.format(name))
    lines.append('    cdef double [:] {0}v'.format(name))
    lines.append('')
    lines.append(cdefs(fnames))
    lines.append(cdefs(gnames))
    lines.append('')
    lines.append(prelude)
    lines.append('')
    lines.append('    fdim = {0}*m*m*n*n'.format(len(ent)))
    lines.append('')
    lines.append('    {0}r = np.zeros((fdim,), dtype=INT)'.format(name))
    lines.append('    {0}c = np.zeros((fdim,), dtype=INT)'.format(name))
    lines.append('    {0}v = np.zeros((fdim,), dtype=DOUBLE)'.format(name))
    lines.append('')
    lines.append('    with nogil:')
    if 'A11' in prelude or name == 'k0':
        lines.append(ABD_READ_ANALYTICAL)
    lines.append('        # {0}'.format(name))
    lines.append('        c = -1')
    lines.append('        for i in range(m):')
    lines.append('            for k in range(m):')
    lines.append('')
    for s in fnames:
        lines.append('                {0} = {1}'.format(s, integral_call(s)))
    lines.append('')
    lines.append('                for j in range(n):')
    lines.append('                    for l in range(n):')
    lines.append('')
    lines.append('                        row = row0 + DOF*(j*m + i)')
    lines.append('                        col = col0 + DOF*(l*m + k)')
    lines.append('')
    lines.append('                        #NOTE symmetry')
    lines.append('                        if row > col:')
    lines.append('                            continue')
    lines.append('')
    for s in gnames:
        lines.append('                        {0} = {1}'.format(s, integral_call(s)))
    lines.append('')
    for i, j, v in ent:
        lines.append('                        c += 1')
        lines.append('                        {0}r[c] = row+{1}'.format(name, i))
        lines.append('                        {0}c[c] = col+{1}'.format(name, j))
        lines.append('                        {0}v[c] += {1}'.format(name, pstr(v)))
    lines.append('')
    lines.append('    {0} = coo_matrix(({0}v, ({0}r, {0}c)), shape=(size, size))'.format(name))
    lines.append('')
    lines.append('    return {0}'.format(name))
    return '\n'.join(lines)


ABD_READ_ANALYTICAL = """        A11 = F[0,0]
        A12 = F[0,1]
        A16 = F[0,2]
        A22 = F[1,1]
        A26 = F[1,2]
        A66 = F[2,2]

        B11 = F[0,3]
        B12 = F[0,4]
        B16 = F[0,5]
        B22 = F[1,4]
        B26 = F[1,5]
        B66 = F[2,5]

        D11 = F[3,3]
        D12 = F[3,4]
        D16 = F[3,5]
        D22 = F[4,4]
        D26 = F[4,5]
        D66 = F[5,5]
"""


def main():
    t0 = time.time()
    mats = th.build()
    print('built', time.time() - t0)
    var = sympy.var
    a, b = var('a, b')
    # analytical: full domain, intx = a and inty = b
    area = a*b/4

    kC0 = mats['kC0'].applyfunc(lambda e: simp(area*e))
    kG = mats['kG'].applyfunc(lambda e: simp(area*e, (th.Nxx, th.Nyy, th.Nxy)))
    kM = mats['kM'].applyfunc(lambda e: expand(area*e))
    print('analytical simplified', time.time() - t0)

    out = {}
    out['k0'] = analytical_function(
        'def fk0(object shell, int size, int row0, int col0):',
        '\n'.join(['    cdef double a, b, r',
                   '    cdef double [:, ::1] F',
                   '    cdef int m, n',
                   BC_CDEF, '',
                   '    cdef int i, j, k, l, c, row, col',
                   ABD_CDEF]),
        '\n'.join(["    if not 'Shell' in shell.__class__.__name__:",
                   "        raise ValueError('a Shell object must be given as input')",
                   '    a = shell.a',
                   '    b = shell.b',
                   '    r = shell.r',
                   '    F = shell.lam.ABD',
                   '    m = shell.m',
                   '    n = shell.n',
                   BC_BLOCK]),
        'k0', kC0)
    out['kG0'] = analytical_function(
        'def fkG0(double Nxx, double Nyy, double Nxy, object shell,\n         int size, int row0, int col0):',
        '\n'.join(['    cdef double a, b, r',
                   '    cdef int m, n',
                   BC_CDEF, '',
                   '    cdef int i, k, j, l, c, row, col']),
        '\n'.join(["    if not 'Shell' in shell.__class__.__name__:",
                   "        raise ValueError('a Shell object must be given as input')",
                   '    a = shell.a',
                   '    b = shell.b',
                   '    r = shell.r',
                   '    m = shell.m',
                   '    n = shell.n',
                   BC_BLOCK]),
        'kG0', kG)
    out['kM'] = analytical_function(
        'def fkM(object shell, double d, int size, int row0, int col0):',
        '\n'.join(['    cdef double a, b, r, rho, h',
                   '    cdef int m, n',
                   BC_CDEF, '',
                   '    cdef int i, k, j, l, c, row, col']),
        '\n'.join(["    if not 'Shell' in shell.__class__.__name__:",
                   "        raise ValueError('a Shell object must be given as input')",
                   '    a = shell.a',
                   '    b = shell.b',
                   '    r = shell.r',
                   '    rho = shell.rho',
                   '    h = sum(shell.plyts)',
                   '    m = shell.m',
                   '    n = shell.n',
                   BC_BLOCK]),
        'kM', kM)
    print('analytical printed', time.time() - t0)

    # numerical
    num = {}
    num['kC'] = entries(mats['kC'].applyfunc(simp))
    print('kC simplified', time.time() - t0)
    NxxNL, NyyNL, NxyNL = var('NxxNL, NyyNL, NxyNL')
    kGNL = mats['kG'].subs({th.Nxx: NxxNL, th.Nyy: NyyNL, th.Nxy: NxyNL},
                           simultaneous=True)
    num['kGNL'] = {(i, j): v for i, j, v in entries(kGNL.applyfunc(
        lambda e: simp(e, (NxxNL, NyyNL, NxyNL))))}
    num['kG'] = entries(mats['kG'].applyfunc(lambda e: simp(e, (th.Nxx, th.Nyy, th.Nxy))))
    num['kM'] = entries(mats['kM'].applyfunc(expand).applyfunc(factor))
    num['fint'] = entries(mats['fint'].applyfunc(
        lambda e: simp(e, (th.Nxx, th.Nyy, th.Nxy, th.Mxx, th.Myy, th.Mxy))))
    print('numerical simplified', time.time() - t0)

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sanders_exprs.py'), 'w') as f:
        f.write('ANALYTICAL = %r\n' % out)
        numstr = {}
        numstr['kC'] = [(i, j, pstr(v)) for i, j, v in num['kC']]
        numstr['kGNL'] = {k: pstr(v) for k, v in num['kGNL'].items()}
        numstr['kG'] = [(i, j, pstr(v)) for i, j, v in num['kG']]
        numstr['kM'] = [(i, j, pstr(v)) for i, j, v in num['kM']]
        numstr['fint'] = [(i, j, pstr(v)) for i, j, v in num['fint']]
        f.write('NUMERICAL = %r\n' % numstr)
    print('done', time.time() - t0)


if __name__ == '__main__':
    main()
