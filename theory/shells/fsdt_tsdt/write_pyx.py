"""Generate the kernels of the models based on the FSDT and TSDT

Writes ``panels/models/<model>.pyx`` and ``panels/models/<model>_num.pyx``
for each model of ``fsdt_tsdt.MODELS``, or for the models given as
arguments::

    python write_pyx.py [model ...]

"""
import os
import re
import sys
import textwrap
import time

import numpy as np
import sympy
from sympy import Add, Mul, Symbol, Rational, Float, expand, collect, factor
from sympy.printing.str import StrPrinter

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'theory', 'shells', 'fsdt_tsdt'))

import fsdt_tsdt as th
from panels.dev.matrixtools import pow2mult

FIELDS = th.FIELDS
DOF = th.DOF
FIELD_RE = '(u|v|w|phix|phiy)'


class Printer(StrPrinter):
    def _print_Float(self, e):
        return repr(float(e))


def pstr(e):
    reps = {q: Float(q) for q in e.atoms(Rational) if not q.is_Integer}
    return pow2mult(Printer().doprint(e.xreplace(reps)))


def simp(e, vars, do_factor=True):
    e = expand(e)
    if do_factor:
        return collect(e, vars, func=factor)
    return collect(e, vars)


BASIS = re.compile(r'^([fg])([AB])' + FIELD_RE + r'((?:xi|eta)*)$')


def basis_info(name):
    mo = BASIS.match(name)
    if mo is None:
        return None
    fg, side, field, der = mo.groups()
    order = der.count('xi') if fg == 'f' else der.count('eta')
    return fg, side, field, order


KIND = {(0, 0): 'ff', (0, 1): 'ffp', (0, 2): 'ffpp',
        (1, 1): 'fpfp', (1, 2): 'fpfpp', (2, 2): 'fppfpp'}


def integral_name(fg, pA, dA, pB, dB):
    d = 'xi' if fg == 'f' else 'eta'
    return '{0}A{1}{2}{0}B{3}{4}'.format(fg, pA, d*dA, pB, d*dB)


def integral_call(name):
    fg = name[0]
    d = 'xi' if fg == 'f' else 'eta'
    mo = re.match(r'^{0}A{2}((?:{1})*){0}B{2}((?:{1})*)$'.format(fg, d, FIELD_RE), name)
    pA, dA, pB, dB = mo.groups()
    dA = dA.count(d)
    dB = dB.count(d)
    i, k = ('i', 'k') if fg == 'f' else ('j', 'l')
    xy = 'x' if fg == 'f' else 'y'
    bc = lambda p: '{0}1{1}, {0}1{1}r, {0}2{1}, {0}2{1}r'.format(xy, p)
    if dA <= dB:
        return 'integral_{0}({1}, {2}, {3}, {4})'.format(KIND[(dA, dB)], i, k, bc(pA), bc(pB))
    return 'integral_{0}({2}, {1}, {4}, {3})'.format(KIND[(dB, dA)], i, k, bc(pA), bc(pB))


def to_integrals(e):
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
        out += (coeff*Symbol(integral_name('f', *f['A'], *f['B']))
                *Symbol(integral_name('g', *g['A'], *g['B'])))
    return out


def entries(m):
    return [(i, j, v) for (i, j), v in np.ndenumerate(np.array(m)) if v != 0]


def cdefs(names, indent=4, per_line=5):
    names = list(names)
    out = []
    for i in range(0, len(names), per_line):
        out.append(' '*indent + 'cdef double ' + ', '.join(names[i:i + per_line]))
    return '\n'.join(out)


HEADER = """#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
"""

BC_CDEF = '\n'.join('    cdef double {0}1{1}, {0}1{1}r, {0}2{1}, {0}2{1}r'.format(xy, p)
                    for xy in 'xy' for p in FIELDS)
BC_BLOCK = '\n'.join('    {0}1{1} = shell.{0}1{1}; {0}1{1}r = shell.{0}1{1}r; {0}2{1} = shell.{0}2{1}; {0}2{1}r = shell.{0}2{1}r'.format(xy, p)
                     for xy in 'xy' for p in FIELDS)

LIMITS = """    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()"""

MAPPING = """    xinf = 0
    xsup = shell.a
    xi1 = (x1 - xinf)/(xsup - xinf)*2 - 1
    xi2 = (x2 - xinf)/(xsup - xinf)*2 - 1

    yinf = 0
    ysup = shell.b
    eta1 = (y1 - yinf)/(ysup - yinf)*2 - 1
    eta2 = (y2 - yinf)/(ysup - yinf)*2 - 1

    intx = x2 - x1
    inty = y2 - y1"""

POINT = """        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]
"""


def finput(NE):
    return """    # F as 4-D matrix, must be [nx, ny, {NE}, {NE}], when there is one
    # constitutive matrix [{NE}, {NE}] for each of the nx * ny integration points
    cdef double F[{NE2}]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, NE, NE):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (NE, NE):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        for i in range(NE):
            for j in range(NE):
                F[i*NE + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')""".format(NE=NE, NE2=NE*NE)


def const_names(Fsym):
    """Symbol name -> (i, j) of its first appearance in the upper triangle"""
    out = {}
    NE = Fsym.shape[0]
    for i in range(NE):
        for j in range(i, NE):
            v = Fsym[i, j]
            if v != 0 and v.name not in out:
                out[v.name] = (i, j)
    return out


def const_read(Fsym, indent, flat=True, names=None):
    s = ' '*indent
    cn = const_names(Fsym)
    lines = []
    for name, (i, j) in cn.items():
        if names is not None and name not in names:
            continue
        if flat:
            lines.append(s + '{0} = F[{1}*NE + {2}]'.format(name, i, j))
        else:
            lines.append(s + '{0} = F[{1},{2}]'.format(name, i, j))
    return '\n'.join(lines)


def ordered_integrals(names):
    return sorted(names, key=lambda s: (s[:2], s))


def analytical_function(signature, cdef_extra, prelude, name, m, symmetric,
                        read=''):
    ent = [(i, j, to_integrals(v)) for i, j, v in entries(m)]
    fnames, gnames = set(), set()
    for _, _, v in ent:
        for s in v.free_symbols:
            if s.name.startswith('fA'):
                fnames.add(s.name)
            elif s.name.startswith('gA'):
                gnames.add(s.name)
    fnames = ordered_integrals(fnames)
    gnames = ordered_integrals(gnames)
    L = [signature, cdef_extra, BC_CDEF, '',
         '    cdef int i, j, k, l, c, row, col', '',
         '    cdef long [:] {0}r, {0}c'.format(name),
         '    cdef double [:] {0}v'.format(name), '',
         cdefs(fnames), cdefs(gnames), '',
         "    if not 'Shell' in shell.__class__.__name__:",
         "        raise ValueError('a Shell object must be given as input')",
         '    a = shell.a', '    b = shell.b', '    m = shell.m', '    n = shell.n',
         prelude, BC_BLOCK, '',
         '    fdim = {0}*m*m*n*n'.format(len(ent)), '',
         '    {0}r = np.zeros((fdim,), dtype=INT)'.format(name),
         '    {0}c = np.zeros((fdim,), dtype=INT)'.format(name),
         '    {0}v = np.zeros((fdim,), dtype=DOUBLE)'.format(name), '',
         '    with nogil:']
    if read:
        L.append(read)
        L.append('')
    L += ['        # {0}'.format(name), '        c = -1',
          '        for i in range(m):', '            for k in range(m):', '']
    L += ['                {0} = {1}'.format(s, integral_call(s)) for s in fnames]
    L += ['', '                for j in range(n):', '                    for l in range(n):', '',
          '                        row = row0 + DOF*(j*m + i)',
          '                        col = col0 + DOF*(l*m + k)', '']
    if symmetric:
        L += ['                        #NOTE symmetry', '                        if row > col:',
              '                            continue', '']
    L += ['                        {0} = {1}'.format(s, integral_call(s)) for s in gnames]
    L.append('')
    for i, j, v in ent:
        L += ['                        c += 1',
              '                        {0}r[c] = row+{1}'.format(name, i),
              '                        {0}c[c] = col+{1}'.format(name, j),
              '                        {0}v[c] += {1}'.format(name, pstr(v))]
    L += ['', '    {0} = coo_matrix(({0}v, ({0}r, {0}c)), shape=(size, size))'.format(name),
          '', '    return {0}'.format(name)]
    return '\n'.join(L)


def field_names(fg):
    d = 'xi' if fg == 'f' else 'eta'
    out = []
    for p in FIELDS:
        out += [p, p + d]
        if p == 'w':
            out.append(p + d + d)
    return out


F_NAMES = field_names('f')
G_NAMES = field_names('g')


def basis_line(fg, side, name, indent):
    mo = re.match(FIELD_RE + '(.*)$', name)
    field, der = mo.groups()
    if fg == 'f':
        order = der.count('xi'); idx = 'i' if side == 'A' else 'k'; var, xy = 'xi', 'x'
    else:
        order = der.count('eta'); idx = 'j' if side == 'A' else 'l'; var, xy = 'eta', 'y'
    func = ['f', 'fp', 'fpp'][order]
    return '{0}{1}{2}{3} = {4}({5}, {6}, {7}1{8}, {7}1{8}r, {7}2{8}, {7}2{8}r)'.format(
        ' '*indent, fg, side, name, func, idx, var, xy, field)


def used(names, text, fg, side):
    return [n for n in names if re.search(r'\b{0}{1}{2}\b'.format(fg, side, n), text)]


def basis_loops(mname, exprs, entry_lines, symmetric=True):
    text = ' '.join(exprs)
    fA, fB = used(F_NAMES, text, 'f', 'A'), used(F_NAMES, text, 'f', 'B')
    gA, gB = used(G_NAMES, text, 'g', 'A'), used(G_NAMES, text, 'g', 'B')
    L = ['                # {0}'.format(mname), '                c = -1',
         '                for i in range(m):']
    L += [basis_line('f', 'A', n, 20) for n in fA]
    L += ['', '                    for k in range(m):']
    L += [basis_line('f', 'B', n, 24) for n in fB]
    L += ['', '                        for j in range(n):']
    L += [basis_line('g', 'A', n, 28) for n in gA]
    L += ['', '                            for l in range(n):', '',
          '                                row = row0 + DOF*(j*m + i)',
          '                                col = col0 + DOF*(l*m + k)', '']
    if symmetric:
        L += ['                                #NOTE symmetry assumption True if no follower forces are used',
              '                                if row > col:', '                                    continue', '']
    L += [basis_line('g', 'B', n, 32) for n in gB]
    L.append('')
    L += entry_lines
    names = (['fA' + n for n in fA] + ['fB' + n for n in fB]
             + ['gA' + n for n in gA] + ['gB' + n for n in gB])
    return '\n'.join(L), names


def entry(mname, i, j, expr, extra=None):
    L = ['                                c += 1',
         '                                if ptx == 0 and pty == 0:',
         '                                    {0}r[c] = row+{1}'.format(mname, i),
         '                                    {0}c[c] = col+{1}'.format(mname, j),
         '                                {0}v[c] += weight*(intx*inty/4)*( {1} )'.format(mname, expr)]
    if extra is not None:
        L += ['                                # KGNL',
              '                                {0}v[c] += weight*(intx*inty/4)*( {1} )'.format(mname, extra)]
    return L


def basis_cdefs(names):
    out = []
    for fg, NAMES in (('f', F_NAMES), ('g', G_NAMES)):
        for side in 'AB':
            ns = ['{0}{1}{2}'.format(fg, side, n) for n in NAMES
                  if '{0}{1}{2}'.format(fg, side, n) in names]
            out.append(cdefs(ns))
    return '\n'.join(o for o in out if o)


ALL_A = (['fA' + n for n in F_NAMES] + ['gA' + n for n in G_NAMES])


def eval_A_lines(indent_g, indent_f):
    """Evaluate all the side A basis functions, g outside and f inside"""
    g = [basis_line('g', 'A', n, indent_g) for n in G_NAMES]
    f = [basis_line('f', 'A', n, indent_f) for n in F_NAMES]
    return g, f


def strain_lines(B0, indent):
    """e[k] += B0[k, :] @ cs[col:col+DOF] for the side A basis at the point"""
    cs = [Symbol('cs{0}'.format(d)) for d in range(DOF)]
    L = []
    for k in range(B0.shape[0]):
        e = sum(B0[k, d]*cs[d] for d in range(DOF))
        e = expand(e)
        if e == 0:
            continue
        s = pstr(e)
        for d in range(DOF):
            s = re.sub(r'\bcs{0}\b'.format(d), 'cs[col+{0}]'.format(d), s)
        L.append(' '*indent + 'e[{0}] += {1}'.format(k, s))
    return L


DESCRIPTION = {
    ('plate', 'donnell'): 'flat plates using the {title} with von Karman kinematics',
    ('cylshell', 'donnell'): ('cylindrical shells using the {title} with the '
                              'Donnell kinematics'),
    ('cylshell', 'sanders'): ('cylindrical shells using the {title} with the '
                              'Sanders-Koiter kinematics'),
}


def nl_block(sanders, indent, cond):
    r"""Rotations ``bx = w,x`` and ``by = w,y - v/r`` (Sanders) of the von
    Karman terms at the integration point"""
    s = ' '*indent
    t = ' '*(indent + 4) if cond else s
    L = [s + 'bx = 0', s + 'by = 0']
    if cond:
        L.append(s + 'if NLgeom == 1:')
    L += [t + 'for j in range(n):',
          t + '    #TODO put these in a lookup vector',
          t + '    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)',
          t + '    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)']
    if sanders:
        L.append(t + '    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)')
    L += [t + '    for i in range(m):',
          t + '        #TODO put these in a lookup vector',
          t + '        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)',
          t + '        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)']
    if sanders:
        L.append(t + '        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)')
    L += ['',
          t + '        col = col0 + DOF*(j*m + i)',
          '',
          t + '        bx += (2/a)*cs[col+2]*fAwxi*gAw',
          t + '        by += (2/b)*cs[col+2]*fAw*gAweta']
    if sanders:
        L.append(t + '        by -= cs[col+1]*fAv*gAv/r')
    return '\n'.join(L)


def with_radius(text, cyl):
    r"""Reads the radius ``r`` in every function of the cylinders"""
    if not cyl:
        return text
    n = text.count('    n = shell.n\n')
    text = text.replace('    cdef double a, b', '    cdef double r\n    cdef double a, b')
    text = text.replace('    n = shell.n\n', '    n = shell.n\n    r = shell.r\n')
    assert n > 0 and text.count('    cdef double r\n') == n
    return text


def generate(model):
    t0 = time.time()
    theory, geometry, kinematics = th.MODELS[model]
    cyl = geometry == 'cylshell'
    sanders = kinematics == 'sanders'
    mats = th.build(theory, geometry, kinematics)
    Fsym = mats['F']
    NE = Fsym.shape[0]
    cvars = tuple(sympy.Symbol(n) for n in const_names(Fsym))
    a, b = sympy.symbols('a, b')
    area = a*b/4
    do_factor = theory == 'fsdt'

    mod = model
    title = {'fsdt': 'first-order shear deformation theory (FSDT)',
             'tsdt': "Reddy's third-order shear deformation theory (TSDT)"}[theory]
    desc = DESCRIPTION[(geometry, kinematics)].format(title=title)
    c1_line = '    c1 = 4./(3.*h*h)' if theory == 'tsdt' else ''
    c1_cdef = ', h, c1' if theory == 'tsdt' else ''
    c1_read = '    h = sum(shell.plyts)\n' + c1_line if theory == 'tsdt' else ''

    kC0 = mats['kC0'].applyfunc(lambda e: simp(area*e, cvars, do_factor))
    kG = mats['kG'].applyfunc(lambda e: simp(area*e, (th.Nxx, th.Nyy, th.Nxy)))
    kM = mats['kM'].applyfunc(lambda e: expand(area*e))
    kAx = mats['kAx'].applyfunc(lambda e: expand(area*e))
    kAy = mats['kAy'].applyfunc(lambda e: expand(area*e))
    cA = mats['cA'].applyfunc(lambda e: expand(area*e))
    print(model, 'analytical simplified', time.time() - t0)

    # --------------------------------------------------------------- analytical
    A = [HEADER + '''r"""
{desc}

Analytical matrices, integrated over the full domain. See the kinematic
equations in ``theory/shells/fsdt_tsdt/fsdt_tsdt.py``, from where the
integrands herein have been generated. The degrees of freedom of each term
of the approximation are ``u, v, w, phix, phiy``.

"""
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE


cdef int DOF = {DOF}
cdef int NE = {NE}


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

'''.format(desc=textwrap.fill(desc[0].upper() + desc[1:], 76), DOF=DOF, NE=NE)]
    k0_cdef = '\n'.join(['    cdef double a, b' + c1_cdef,
                         '    cdef double [:, ::1] F',
                         '    cdef int m, n',
                         cdefs([v.name for v in cvars])])
    k0_pre = '\n'.join(['    F = shell.ABD', c1_read,
                        "    if F.shape[0] != NE or F.shape[1] != NE:",
                        "        raise ValueError('shell.ABD must be a %d x %d matrix' % (NE, NE))"])
    A.append(analytical_function('def fk0(object shell, int size, int row0, int col0):',
                                 k0_cdef, k0_pre, 'k0', kC0, True,
                                 read=const_read(Fsym, 8, flat=False)))
    A.append('\n')
    A.append(analytical_function(
        'def fkG0(double Nxx, double Nyy, double Nxy, object shell,\n         int size, int row0, int col0):',
        '    cdef double a, b\n    cdef int m, n', '', 'kG0', kG, True))
    A.append('\n')
    A.append(analytical_function(
        'def fkM(object shell, double d, int size, int row0, int col0):',
        '    cdef double a, b, rho, h' + (', c1' if theory == 'tsdt' else '') + '\n    cdef int m, n',
        '    rho = shell.rho\n    h = sum(shell.plyts)' + ('\n' + c1_line if theory == 'tsdt' else ''),
        'kM', kM, True))
    A.append('\n')
    A.append(analytical_function(
        'def fkAx(double beta, double gamma, object shell,\n         int size, int row0, int col0):',
        '    cdef double a, b\n    cdef int m, n', '', 'kAx', kAx, False))
    A.append('\n')
    A.append(analytical_function(
        'def fkAy(double beta, object shell, int size, int row0, int col0):',
        '    cdef double a, b\n    cdef int m, n', '', 'kAy', kAy, False))
    A.append('\n')
    A.append(analytical_function(
        'def fcA(double aeromu, object shell, int size, int row0, int col0):',
        '    cdef double a, b\n    cdef int m, n', '', 'cA', cA, True))
    A.append('')
    with open(os.path.join(REPO, 'panels', 'models', mod + '.pyx'), 'w', newline='\n') as f:
        f.write(with_radius('\n'.join(A), cyl))
    print(model, 'analytical written', time.time() - t0)

    # --------------------------------------------------------------- numerical
    kC = entries(mats['kC'].applyfunc(lambda e: simp(e, cvars, do_factor)))
    print(model, 'kC simplified', time.time() - t0)
    NxxNL, NyyNL, NxyNL = sympy.symbols('NxxNL, NyyNL, NxyNL')
    kGNL = {(i, j): pstr(v) for i, j, v in entries(mats['kG'].subs(
        {th.Nxx: NxxNL, th.Nyy: NyyNL, th.Nxy: NxyNL}, simultaneous=True).applyfunc(
        lambda e: simp(e, (NxxNL, NyyNL, NxyNL))))}
    kGn = [(i, j, pstr(v)) for i, j, v in entries(mats['kG'].applyfunc(
        lambda e: simp(e, (th.Nxx, th.Nyy, th.Nxy))))]
    kMn = [(i, j, pstr(v)) for i, j, v in entries(mats['kM'].applyfunc(
        lambda e: collect(expand(e), (sympy.Symbol('d'),))))]
    kAxn = [(i, j, pstr(v)) for i, j, v in entries(mats['kAx'].applyfunc(expand))]
    kAyn = [(i, j, pstr(v)) for i, j, v in entries(mats['kAy'].applyfunc(expand))]
    snames = th.stress_names(theory)
    fintn = [(i, pstr(v)) for i, j, v in entries(mats['fint'].applyfunc(
        lambda e: simp(e, snames)))]
    kCn = [(i, j, pstr(v)) for i, j, v in kC]
    print(model, 'numerical simplified', time.time() - t0)

    B0 = mats['B0']
    ne_names = ('exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz' if theory == 'fsdt' else
                'exx, eyy, gxy, kxx, kyy, kxy, kxx3, kyy3, kxy3, gyz, gxz, gyz2, gxz2')
    sn = ', '.join(s.name for s in snames)

    NUM = [HEADER + '''r"""
{desc}

Numerically integrated matrices. See the kinematic equations in
``theory/shells/fsdt_tsdt/fsdt_tsdt.py``, from where the integrands herein
have been generated. The degrees of freedom of each term of the
approximation are ``u, v, w, phix, phiy``, and the constitutive matrix
``Finput`` must be ``{NE} x {NE}``, with the rows and columns in the order of
the generalized strains::

    {ne_names}

which is the order of the generalized stresses::

    {sn}

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = {DOF}
cdef int NE = {NE}
'''.format(desc=textwrap.fill(desc[0].upper() + desc[1:], 76), DOF=DOF, NE=NE,
           ne_names=ne_names, sn=sn)]

    c1n_cdef = ', h, c1' if theory == 'tsdt' else ''
    c1n_read = ('    #NOTE the traction-free faces are assumed at z = +-h/2\n'
                '    h = shell.lam.h\n    c1 = 4./(3.*h*h)') if theory == 'tsdt' else ''

    # fkC_num
    L = []
    for i, j, v in kCn:
        L += entry('kC', i, j, v, kGNL.get((i, j)))
    loops, names = basis_loops('kC', [v for _, _, v in kCn] + list(kGNL.values()), L)
    NUM.append('''
def fkC_num(double [::1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty{c1}
    cdef int m, n
{BC_CDEF}

    cdef int i, j, k, l, c, row, col, ptx, pty
{cc}

    cdef long [::1] kCr, kCc
    cdef double [::1] kCv

{basis}
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double bx, by, NxxNL, NyyNL, NxyNL

    cdef double [::1] xis, etas, weights_xi, weights_eta

{FINPUT}

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
{c1read}
{LIMITS}
{BC_BLOCK}

    fdim = {nnz}*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kCr = np.zeros((fdim,), dtype=INT)
    kCc = np.zeros((fdim,), dtype=INT)
    kCv = np.zeros((fdim,), dtype=DOUBLE)

{MAPPING}

    with nogil:
{POINT}
{nl}

                if one_F_each_point == 1:
                    for i in range(NE):
                        for j in range(NE):
                            #TODO could assume symmetry
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

{read}

                # Membrane stress carried by the nonlinear strain
                # eps_NL = {{bx^2/2, by^2/2, bx*by}}. With it, KGNL = KG(N_NL)
                # is collected in kC such that
                #     KT = K0 + K0L + KL0 + KLL + KGNL (fkC_num) + KG(N0 + N_L) (fkG_num)
                # is the exact Jacobian of calc_fint, and fkG_num stays
                # homogeneous of degree one in cs, as linear buckling requires.
                # bx = by = 0 when NLgeom == 0, then KGNL vanishes
                NxxNL = A11*0.5*bx*bx + A12*0.5*by*by + A16*bx*by
                NyyNL = A12*0.5*bx*bx + A22*0.5*by*by + A26*bx*by
                NxyNL = A16*0.5*bx*bx + A26*0.5*by*by + A66*bx*by

{loops}

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC
'''.format(c1=c1n_cdef, BC_CDEF=BC_CDEF, cc=cdefs([v.name for v in cvars]),
           basis=basis_cdefs(set(names) | {'fAw', 'fAwxi', 'gAw', 'gAweta'}
                             | ({'fAv', 'gAv'} if sanders else set())),
           FINPUT=finput(NE), c1read=c1n_read, LIMITS=LIMITS, BC_BLOCK=BC_BLOCK,
           nnz=len(kCn), MAPPING=MAPPING, POINT=POINT, nl=nl_block(sanders, 16, True),
           read=const_read(Fsym, 16), loops=loops))

    # fkG_num
    L = []
    for i, j, v in kGn:
        L += entry('kG', i, j, v)
    loops, names = basis_loops('kG', [v for _, _, v in kGn], L)
    gl, fl = eval_A_lines(20, 24)
    NUM.append('''

def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    """Geometric stiffness matrix of the linear membrane stress state

    The membrane stress used here comes from the *linear* part of the strain
    evaluated at ``cs``, superposed with the constant stress state
    ``(Nxx0, Nyy0, Nxy0)``. The result is therefore homogeneous of degree one
    in ``cs``, which is what the linear buckling eigenvalue problem requires.

    There is deliberately no switch to include the stress of the non-linear
    strain here. That contribution, ``KGNL``, is collected by
    :func:`.fkC_num` instead, so that

        KT = K0 + K0L + KL0 + KLL + KGNL   (fkC_num)
             + KG(N0 + N_L)                (fkG_num)

    is the exact Jacobian of :func:`.calc_fint` while this function stays
    homogeneous of degree one in ``cs``.

    """
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty{c1}
    cdef int m, n
{BC_CDEF}

    cdef int i, j, k, l, c, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef long [::1] kGr, kGc
    cdef double [::1] kGv

{basis}

    cdef double e[{NE}]
    cdef double Nxx, Nyy, Nxy

    cdef double [::1] xis, etas, weights_xi, weights_eta

{FINPUT}

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
{c1read}
{LIMITS}
{BC_BLOCK}

    fdim = {nnz}*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kGr = np.zeros((fdim,), dtype=INT)
    kGc = np.zeros((fdim,), dtype=INT)
    kGv = np.zeros((fdim,), dtype=DOUBLE)

{MAPPING}

    with nogil:
{POINT}
                # Reading laminate constitutive data
                if one_F_each_point == 1:
                    for i in range(NE):
                        for j in range(NE):
                            #TODO could assume symmetry
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

                # Calculating the linear generalized strains. The stress of
                # the nonlinear strain enters KT through KGNL in fkC_num, such
                # that kG is homogeneous of degree one in cs
                for i in range(NE):
                    e[i] = 0.
                for j in range(n):
                    #TODO put these in a lookup vector
{gl}

                    for i in range(m):
{fl}

                        col = col0 + DOF*(j*m + i)

{strains}

                # Calculating membrane stress components
                Nxx = Nxx0
                Nyy = Nyy0
                Nxy = Nxy0
                for i in range(NE):
                    Nxx += F[0*NE + i]*e[i]
                    Nyy += F[1*NE + i]*e[i]
                    Nxy += F[2*NE + i]*e[i]

{loops}

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG
'''.format(c1=c1n_cdef, BC_CDEF=BC_CDEF,
           basis=basis_cdefs(set(names) | set(ALL_A)), FINPUT=finput(NE),
           c1read=c1n_read, LIMITS=LIMITS, BC_BLOCK=BC_BLOCK, nnz=len(kGn),
           MAPPING=MAPPING, POINT=POINT, gl='\n'.join(gl), fl='\n'.join(fl),
           strains='\n'.join(strain_lines(B0, 24)), loops=loops, NE=NE))

    # fkM_num
    L = []
    for i, j, v in kMn:
        L += entry('kM', i, j, v)
    loops, names = basis_loops('kM', [v for _, _, v in kMn], L)
    NUM.append('''

def fkM_num(object shell, double offset, object hrho_input, int size,
        int row0, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty{c1}
    cdef int m, n
{BC_CDEF}

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kMr, kMc
    cdef double [::1] kMv

{basis}
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # hrho as 3-D matrix, must be [nx, ny, 2], when there is one pair (h, rho)
    # for each of the nx * ny integration points
    cdef double h, rho, d
    cdef double [:, :, ::1] hrho_nxny

    cdef int one_hrho_each_point = 0

    d = offset

    hrho_input = np.asarray(hrho_input, dtype=DOUBLE)
    if hrho_input.shape == (nx, ny, 2):
        hrho_nxny = np.ascontiguousarray(hrho_input)
        one_hrho_each_point = 1
    elif hrho_input.shape == (2,):
        # creating dummy 3-D array that is not used
        hrho_nxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        # using a constant (h, rho) for all the integration domain
        hrho_input = np.ascontiguousarray(hrho_input)
        h, rho = hrho_input
    else:
        raise ValueError('Invalid shape for hrho_input!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
{LIMITS}
{BC_BLOCK}

    fdim = {nnz}*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kMr = np.zeros((fdim,), dtype=INT)
    kMc = np.zeros((fdim,), dtype=INT)
    kMv = np.zeros((fdim,), dtype=DOUBLE)

{MAPPING}

    with nogil:
{POINT}
                if one_hrho_each_point == 1:
                    h = hrho_nxny[ptx, pty, 0]
                    rho = hrho_nxny[ptx, pty, 1]
{c1pt}
{loops}

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM
'''.format(c1=', c1' if theory == 'tsdt' else '', BC_CDEF=BC_CDEF,
           basis=basis_cdefs(set(names)), LIMITS=LIMITS, BC_BLOCK=BC_BLOCK,
           nnz=len(kMn), MAPPING=MAPPING, POINT=POINT, loops=loops,
           c1pt=('                c1 = 4./(3.*h*h)\n' if theory == 'tsdt' else '')))

    # fkAx_num and fkAy_num
    for fname, ents, beta_gamma in (
            ('fkAx_num', kAxn, 'beta = shell.beta' + ('\n    gamma = shell.gamma' if cyl else '')),
            ('fkAy_num', kAyn, 'beta = shell.beta')):
        L = []
        for i, j, v in ents:
            L += entry('kA', i, j, v)
        loops, names = basis_loops('kA', [v for _, _, v in ents], L, symmetric=False)
        NUM.append('''

def {fname}(object shell, int size, int row0, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, beta, intx, inty{gcdef}
    cdef int m, n
{BC_CDEF}

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kAr, kAc
    cdef double [::1] kAv

{basis}
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
{LIMITS}
    {bg}
{BC_BLOCK}

    fdim = {nnz}*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kAr = np.zeros((fdim,), dtype=INT)
    kAc = np.zeros((fdim,), dtype=INT)
    kAv = np.zeros((fdim,), dtype=DOUBLE)

{MAPPING}

    with nogil:
{POINT}
{loops}

    kA = coo_matrix((kAv, (kAr, kAc)), shape=(size, size))

    return kA
'''.format(fname=fname, BC_CDEF=BC_CDEF, basis=basis_cdefs(set(names)),
           LIMITS=LIMITS, bg=beta_gamma, BC_BLOCK=BC_BLOCK, nnz=len(ents),
           gcdef=', gamma' if cyl and fname == 'fkAx_num' else '',
           MAPPING=MAPPING, POINT=POINT, loops=loops))

    # calc_fint
    gl1, fl1 = eval_A_lines(20, 24)
    fint_lines = ['                        fint[col+{0}] += weight*(intx*inty/4)*( {1} )'.format(i, v)
                  for i, v in fintn]
    s_assign = '\n'.join('                {0} = s[{1}]'.format(s.name, k) for k, s in enumerate(snames))
    NUM.append('''

def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty{c1}
    cdef int m, n
{BC_CDEF}

    cdef int i, j, c, col, ptx, pty
    cdef double e[{NE}]
    cdef double s[{NE}]
{sc}

    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double bx, by

{basis}

    cdef double [::1] xis, etas, weights_xi, weights_eta, fint

{FINPUT}

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
{c1read}
{LIMITS}
{BC_BLOCK}

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    fint = np.zeros(size, dtype=DOUBLE)

{MAPPING}

    with nogil:
{POINT}
                if one_F_each_point == 1:
                    for i in range(NE):
                        for j in range(NE):
                            #TODO could assume symmetry
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

{nl}

                # current generalized strain state
                for i in range(NE):
                    e[i] = 0.
                for j in range(n):
                    #TODO save in buffer
{gl}

                    for i in range(m):
                        #TODO save in buffer
{fl}

                        col = col0 + DOF*(j*m + i)

{strains}

                # nonlinear strain eps_NL = {{bx^2/2, by^2/2, bx*by}}
                e[0] += 0.5*bx*bx
                e[1] += 0.5*by*by
                e[2] += bx*by

                # current generalized stress state
                for i in range(NE):
                    s[i] = 0.
                    for j in range(NE):
                        s[i] += F[i*NE + j]*e[j]
{sassign}

                for j in range(n):
{gl}
                    for i in range(m):
{fl}

                        col = col0 + DOF*(j*m + i)

{FINT}

    return fint
'''.format(c1=c1n_cdef, BC_CDEF=BC_CDEF, sc=cdefs([s.name for s in snames]),
           basis=basis_cdefs(set(ALL_A)), FINPUT=finput(NE), c1read=c1n_read,
           LIMITS=LIMITS, BC_BLOCK=BC_BLOCK, MAPPING=MAPPING, POINT=POINT,
           gl='\n'.join(gl1), fl='\n'.join(fl1),
           strains='\n'.join(strain_lines(B0, 24)), sassign=s_assign,
           FINT='\n'.join(fint_lines), NE=NE, nl=nl_block(sanders, 16, False)))

    with open(os.path.join(REPO, 'panels', 'models', mod + '_num.pyx'), 'w', newline='\n') as f:
        f.write(with_radius(''.join(NUM), cyl))
    print(model, 'numerical written', time.time() - t0)


if __name__ == '__main__':
    for model in sys.argv[1:] or th.MODELS:
        generate(model)
