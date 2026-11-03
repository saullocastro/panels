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
from panels.dev.tabulate import tabulate_basis
#NOTE the follower-pressure kernels, appended to the numerical modules
import importlib.util
_spec = importlib.util.spec_from_file_location('follower_write_pyx',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                 'follower_pressure', 'write_pyx.py'))
_follower = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_follower)
follower_block = _follower.pyx_block
#NOTE the kernel of the product of the tangent stiffness matrix by a vector
_spec = importlib.util.spec_from_file_location('tangent_write_pyx',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                 'tangent_product', 'write_pyx.py'))
_tangent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tangent)
tangent_block = _tangent.pyx_block

FIELDS = th.FIELDS
#NOTE the moments of the density of composites.Laminate
LAM_MOMENTS = dict(I0='intrho', I1='intrhoz', I2='intrhoz2', I3='intrhoz3',
                   I4='intrhoz4', I6='intrhoz6',
                   J0='intrho', J1='intrhoz', J2='intrhoz2', J3='intrhoz3',
                   J4='intrhoz4', J6='intrhoz6')
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


BASIS = re.compile(r'^([fg])([AB])' + FIELD_RE + r'((?:_(?:xi|eta)+)?)$')


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
#cython: initializedcheck=False
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
    d = '_xi' if fg == 'f' else '_eta'
    out = []
    for p in FIELDS:
        out += [p, p + d]
        if p == 'w':
            out.append(p + d + d[1:])
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
    ('plate', 'donnell'): 'flat plates using the {title} with von Kármán kinematics',
    ('cylshell', 'donnell'): ('cylindrical shells using the {title} with the '
                              'Donnell kinematics'),
    ('cylshell', 'sanders'): ('cylindrical shells using the {title} with the '
                              'Sanders-Koiter kinematics'),
}


def nl_block(sanders, indent, cond):
    r"""Rotations ``bx = -w,x`` and ``by = -w,y + v/r`` (Sanders) of the von
    Karman terms at the integration point"""
    s = ' '*indent
    t = ' '*(indent + 4) if cond else s
    L = [s + 'bx = 0', s + 'by = 0']
    if cond:
        L.append(s + 'if NLgeom == 1:')
    L += [t + 'for j in range(n):',
          t + '    #TODO put these in a lookup vector',
          t + '    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)',
          t + '    gAw_eta = fp(j, eta, y1w, y1wr, y2w, y2wr)']
    if sanders:
        L.append(t + '    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)')
    L += [t + '    for i in range(m):',
          t + '        #TODO put these in a lookup vector',
          t + '        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)',
          t + '        fAw_xi = fp(i, xi, x1w, x1wr, x2w, x2wr)']
    if sanders:
        L.append(t + '        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)')
    L += ['',
          t + '        col = col0 + DOF*(j*m + i)',
          '',
          t + '        bx -= (2/a)*cs[col+2]*fAw_xi*gAw',
          t + '        by -= (2/b)*cs[col+2]*fAw*gAw_eta']
    if sanders:
        L.append(t + '        by += cs[col+1]*fAv*gAv/r')
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
    # the moments of the density through the thickness used by the theory,
    # about the mid-surface for the TSDT, see fsdt_tsdt.py
    moment_names = [s.name for s in th.MOMENTS + th.JMOMENTS
                    if any(s in e.free_symbols for e in mats['kM'])]
    assert moment_names == (['I0', 'I1', 'I2'] if theory == 'fsdt' else
                            ['J0', 'J1', 'J2', 'J3', 'J4', 'J6']), moment_names
    kAx = mats['kAx'].applyfunc(lambda e: expand(area*e))
    kAy = mats['kAy'].applyfunc(lambda e: expand(area*e))
    cA = mats['cA'].applyfunc(lambda e: expand(area*e))
    print(model, 'analytical simplified', time.time() - t0)

    # --------------------------------------------------------------- analytical
    A = [HEADER + '''r"""
{desc}

Matrices integrated analytically over the whole domain: the linear
constitutive stiffness :func:`fk0`, the geometric stiffness :func:`fkG0`,
the mass :func:`fkM`, the aerodynamic stiffness :func:`fkAx`, :func:`fkAy`
and the aerodynamic damping :func:`fcA`, as
:class:`scipy.sparse.coo_matrix`.
The constitutive matrix ``shell.ABD`` is ``{NE} x {NE}``.

The degrees of freedom of each term of the approximation are
``u, v, w, phix, phiy``; the degree of freedom ``k`` of the term ``(i, j)``, ``0 <=
i < m``, ``0 <= j < n``, is at the row ``row0 + {DOF}*(j*m + i) + k``
of the matrices.

Notes
-----
The displacement field through the thickness and the generalized strains
are those of :mod:`panels.models.fsdt_tsdt_field`, with `c_1 = 4/(3 h^2)`
for the TSDT and `c_1 = 0` for the FSDT, `1/r = 0` for the plates, and the
Sanders-Koiter terms only for the models ``*_sanders``. The rotations of
the non-linear terms are `\\beta_x = -w_{{,x}}` and `\\beta_y = -w_{{,y}} + s
v/r`, `s = 1` for the Sanders-Koiter and `s = 0` for the Donnell
kinematics.

The integrals over the whole domain are evaluated in closed form with the
integrals of Bardell's functions and their derivatives along `\\xi = 2x/a -
1` and `\\eta = 2y/b - 1`, which requires the constitutive matrix and the
moments of the density to be constant over the domain.

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
    #NOTE the kernels read the property Shell.ABD, and fkM calls
    #     Shell._get_lam_mid, with C variables already set, which must not
    #     call matrix-matrix BLAS products, see the note in
    #     Shell._get_lam_ABD and tests/tests_shell/test_kernel_python_calls.py
    k0_pre = '\n'.join(['    F = shell.ABD', c1_read,
                        "    if F.shape[0] != NE or F.shape[1] != NE:",
                        "        raise ValueError('shell.ABD must be a %d x %d matrix' % (NE, NE))"])
    A.append(analytical_function(r'''def fk0(object shell, int size, int row0, int col0):
    r"""Linear constitutive stiffness matrix, integrated analytically

    Integrates the linear constitutive stiffness matrix of the shell over
    its whole domain for the constitutive matrix ``shell.ABD``, constant
    over the domain.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    k0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell` or ``shell.ABD`` is not a
        ``NE x NE`` matrix.

    Notes
    -----
    With the linear part `\boldsymbol{\varepsilon}_L =
    \boldsymbol{B}_L \boldsymbol{c}` of the generalized strains of the kinematics of the module,
    and the constitutive matrix `\boldsymbol{F}` (``shell.ABD``):

    .. math::

        \boldsymbol{K}_{C0} = \int_0^b \int_0^a \boldsymbol{B}_L^\top
        \boldsymbol{F} \boldsymbol{B}_L \, dx \, dy

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\xi` and `\eta`, with `x = a(\xi +
    1)/2`, `y = b(\eta + 1)/2` and `dx\,dy = (ab/4)\,d\xi\,d\eta`.

    """''',
                                 k0_cdef, k0_pre, 'k0', kC0, True,
                                 read=const_read(Fsym, 8, flat=False)))
    A.append('\n')
    A.append(analytical_function(
        r'''def fkG0(double Nxx, double Nyy, double Nxy, object shell,
         int size, int row0, int col0):
    r"""Geometric stiffness matrix of a constant membrane stress state

    Integrates, analytically over the whole domain, the geometric
    stiffness matrix of the membrane stress resultants ``Nxx``, ``Nyy``
    and ``Nxy``, constant over the domain.

    Parameters
    ----------
    Nxx, Nyy, Nxy : float
        Membrane stress resultants, force per unit length, tension positive.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    kG0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With `\boldsymbol{N} = \begin{bmatrix} N_{xx} & N_{xy} \\ N_{xy} &
    N_{yy} \end{bmatrix}` and `\boldsymbol{G} \boldsymbol{c} = \{\beta_x, \beta_y\}^\top` the
    rotations of the non-linear terms of the membrane strains:

    .. math::

        \boldsymbol{K}_{G} = \int_0^b \int_0^a \boldsymbol{G}^\top
        \boldsymbol{N} \boldsymbol{G} \, dx \, dy

    which is the Hessian of the work of the constant stress resultants on
    the non-linear part of the membrane strains of the kinematics of the module.

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\xi` and `\eta`, with `x = a(\xi +
    1)/2`, `y = b(\eta + 1)/2` and `dx\,dy = (ab/4)\,d\xi\,d\eta`.

    """''',
        '    cdef double a, b\n    cdef int m, n', '', 'kG0', kG, True))
    A.append('\n')
    A.append(analytical_function(
        r'''def fkM(object shell, int size, int row0, int col0):
    r"""Mass matrix, integrated analytically

    Integrates the consistent mass matrix of the shell over its whole
    domain, with the moments of the density of the laminate
    ``shell.lam``, constant over the domain.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    kM : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With the displacement field through the thickness `\{u(z), v(z),
    w(z)\}^\top = \boldsymbol{S}(z) \boldsymbol{c}` of the kinematics of the module, `z` measured
    from the reference surface:

    .. math::

        \boldsymbol{M} = \int_0^b \int_0^a \int_{d - h/2}^{d + h/2} \rho \,
        \boldsymbol{S}(z)^\top \boldsymbol{S}(z) \, dz \, dx \, dy

    with the offset `d` (``shell.offset``). The integral through the
    thickness gives the moments `I_n = \int \rho z^n dz`, `n = 0, 1, 2`, of
    ``shell.lam`` for the FSDT, and for the TSDT the moments `J_n`, `n = 0,
    1, 2, 3, 4, 6`, about the mid-surface, of :meth:`.Shell._get_lam_mid`,
    with `d` and the thickness `h` of ``shell.lam``.

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\xi` and `\eta`, with `x = a(\xi +
    1)/2`, `y = b(\eta + 1)/2` and `dx\,dy = (ab/4)\,d\xi\,d\eta`.

    """''',
        '    cdef double a, b, ' + ', '.join(moment_names) + (', d, h, c1' if theory == 'tsdt' else '') + '\n    cdef int m, n',
        ('    #NOTE the moments of the density about the mid-surface, those of\n'
         '    #     the laminate with a zero offset, and the offset d\n'
         '    lam_mid = shell._get_lam_mid()\n' if theory == 'tsdt' else '')
        + '\n'.join('    {0} = {1}.{2}'.format(name, 'lam_mid' if theory == 'tsdt' else 'shell.lam',
                                             LAM_MOMENTS[name])
                    for name in moment_names)
        + ('\n    d = shell.offset\n    h = shell.lam.h\n' + c1_line if theory == 'tsdt' else ''),
        'kM', kM, True))
    A.append('\n')
    A.append(analytical_function(
        r'''def fkAx(double beta, double gamma, object shell,
         int size, int row0, int col0):
    r"""Aerodynamic stiffness matrix of the piston theory, flow along x

    Integrates, analytically over the whole domain, the aerodynamic
    stiffness matrix of a supersonic flow along `x`.

    Parameters
    ----------
    beta : float
        Aerodynamic pressure parameter `\beta` of the piston theory, a
        pressure, see :meth:`.Shell.calc_kA`.
    gamma : float
        Curvature correction `\gamma` of the piston theory for the
        cylinders, a pressure per unit length; ignored by the flat plates.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    kAx : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. The matrix is assembled in full.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With `\boldsymbol{S}^w` the row of the approximation functions of
    `w`, `w = \boldsymbol{S}^w \boldsymbol{c}`:

    .. math::

        \boldsymbol{K}_{A_x} = -\int_0^b \int_0^a \left[\beta
        (\boldsymbol{S}^w_{,x})^\top \boldsymbol{S}^w + \gamma
        (\boldsymbol{S}^w)^\top \boldsymbol{S}^w\right] dx \, dy

    added to the stiffness matrix, see :meth:`.Shell.calc_kA` for the
    convention of signs; the matrix is not symmetric.

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\xi` and `\eta`, with `x = a(\xi +
    1)/2`, `y = b(\eta + 1)/2` and `dx\,dy = (ab/4)\,d\xi\,d\eta`.

    """''',
        '    cdef double a, b\n    cdef int m, n', '', 'kAx', kAx, False))
    A.append('\n')
    A.append(analytical_function(
        r'''def fkAy(double beta, object shell, int size, int row0, int col0):
    r"""Aerodynamic stiffness matrix of the piston theory, flow along y

    Integrates, analytically over the whole domain, the aerodynamic
    stiffness matrix of a supersonic flow along `y`.

    Parameters
    ----------
    beta : float
        Aerodynamic pressure parameter `\beta` of the piston theory, a
        pressure, see :meth:`.Shell.calc_kA`.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    kAy : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. The matrix is assembled in full.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With `\boldsymbol{S}^w` the row of the approximation functions of
    `w`, `w = \boldsymbol{S}^w \boldsymbol{c}`:

    .. math::

        \boldsymbol{K}_{A_y} = -\beta \int_0^b \int_0^a
        (\boldsymbol{S}^w_{,y})^\top \boldsymbol{S}^w \, dx \, dy

    added to the stiffness matrix, see :meth:`.Shell.calc_kA`; the matrix is
    not symmetric.

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\xi` and `\eta`, with `x = a(\xi +
    1)/2`, `y = b(\eta + 1)/2` and `dx\,dy = (ab/4)\,d\xi\,d\eta`.

    """''',
        '    cdef double a, b\n    cdef int m, n', '', 'kAy', kAy, False))
    A.append('\n')
    A.append(analytical_function(
        r'''def fcA(double aeromu, object shell, int size, int row0, int col0):
    r"""Aerodynamic damping matrix of the piston theory

    Integrates, analytically over the whole domain, the aerodynamic
    damping matrix, without the factor `i` that
    :meth:`.Shell.calc_cA` applies.

    Parameters
    ----------
    aeromu : float
        Aerodynamic damping coefficient `\mu` of the piston theory, a
        pressure per unit velocity.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.

    Returns
    -------
    cA : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With `\boldsymbol{S}^w` the row of the approximation functions of
    `w`:

    .. math::

        \boldsymbol{C}_A = -\mu \int_0^b \int_0^a (\boldsymbol{S}^w)^\top
        \boldsymbol{S}^w \, dx \, dy

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\xi` and `\eta`, with `x = a(\xi +
    1)/2`, `y = b(\eta + 1)/2` and `dx\,dy = (ab/4)\,d\xi\,d\eta`.

    """''',
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
        lambda e: collect(expand(e), th.MOMENTS)))]
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

Matrices and vectors integrated numerically with the Gauss-Legendre
quadrature: the constitutive stiffness :func:`fkC_num`, the geometric
stiffness :func:`fkG_num`, the mass :func:`fkM_num`, the aerodynamic
stiffness :func:`fkAx_num`, :func:`fkAy_num`, the internal force vector
:func:`calc_fint`, the product of the tangent stiffness matrix by a vector
:func:`calc_kTv`, and the force vector and load stiffness matrix of a
follower pressure, :func:`calc_fext_follower` and :func:`fkCfollower_num`.
The constitutive matrix ``Finput`` is ``{NE} x {NE}``, with the
rows and columns in the order of the generalized strains::

    {ne_names}

which is the order of the generalized stresses::

    {sn}

The degrees of freedom of each term of the approximation are
``u, v, w, phix, phiy``; the degree of freedom ``k`` of the term ``(i, j)``, ``0 <=
i < m``, ``0 <= j < n``, is at the row ``row0 + {DOF}*(j*m + i) + k``
of the matrices.

Notes
-----
The displacement field through the thickness and the generalized strains
are those of :mod:`panels.models.fsdt_tsdt_field`, with `c_1 = 4/(3 h^2)`
for the TSDT and `c_1 = 0` for the FSDT, `1/r = 0` for the plates, and the
Sanders-Koiter terms only for the models ``*_sanders``. The rotations of
the non-linear terms are `\\beta_x = -w_{{,x}}` and `\\beta_y = -w_{{,y}} + s
v/r`, `s = 1` for the Sanders-Koiter and `s = 0` for the Donnell
kinematics.

The numerical integration over the domain of
:meth:`.Shell.integration_limits` admits a constitutive matrix and moments
of the density varying over the domain, a partial domain and the
non-linear terms of the Ritz constants.

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
    r"""Constitutive stiffness matrix, integrated numerically

    With ``NLgeom=0``, the linear constitutive stiffness matrix
    ``KC0``; with ``NLgeom=1``, the matrix ``KC0 + KCNL + KGNL`` at the
    configuration ``cs``, which, added to :func:`fkG_num` at the same
    ``cs``, gives the tangent stiffness matrix.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements (the rotations in radians).
    Finput : array-like
        Constitutive matrix of the generalized strains of the model, shape
        ``(NE, NE)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        NE, NE)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.
    NLgeom : int, optional
        ``1`` to include the non-linear terms of ``cs``, default ``0``.

    Returns
    -------
    kC : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape or ``shell`` is not a
        :class:`.Shell`.

    Notes
    -----
    With the generalized strains of the kinematics of the module,
    `\\boldsymbol{{\\varepsilon}} = (\\boldsymbol{{B}}_L + \\frac{{1}}{{2}}
    \\boldsymbol{{B}}_{{NL}}(\\boldsymbol{{c}})) \\boldsymbol{{c}}`, their variation
    `\\delta \\boldsymbol{{\\varepsilon}} = \\boldsymbol{{B}} \\delta \\boldsymbol{{c}}`,
    `\\boldsymbol{{B}} = \\boldsymbol{{B}}_L + \\boldsymbol{{B}}_{{NL}}(\\boldsymbol{{c}})`,
    and the constitutive matrix `\\boldsymbol{{F}}`:

    .. math::

        \\boldsymbol{{K}}_C = \\int \\boldsymbol{{B}}^\\top \\boldsymbol{{F}}
        \\boldsymbol{{B}} \\, dA + \\int \\boldsymbol{{G}}^\\top
        \\boldsymbol{{N}}_{{NL}} \\boldsymbol{{G}} \\, dA

    with `\\boldsymbol{{G}} \\boldsymbol{{c}} = \\{{\\beta_x, \\beta_y\\}}^\\top` the
    rotations of the non-linear terms of the membrane strains, and `\\boldsymbol{{N}}_{{NL}}` the membrane stress resultants of the
    non-linear part of the strains, `\\frac{{1}}{{2}} \\boldsymbol{{B}}_{{NL}}
    \\boldsymbol{{c}}`. With ``NLgeom=0``, `\\boldsymbol{{B}}_{{NL}} = \\boldsymbol{{0}}`
    and `\\boldsymbol{{K}}_C = \\boldsymbol{{K}}_{{C0}}`. The membrane stress of the
    linear strains enters :func:`fkG_num`, such that ``fkC_num(NLgeom=1) +
    fkG_num`` is the exact Jacobian of :func:`calc_fint`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \\le x \\le x_2`, `y_1 \\le y \\le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
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
                #     KT = KC0 + KCNL + KGNL (fkC_num) + KG(Nbar + N_L) (fkG_num)
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
           basis=basis_cdefs(set(names) | {'fAw', 'fAw_xi', 'gAw', 'gAw_eta'}
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
    r"""Geometric stiffness matrix of the linear membrane stress state

    Integrates numerically the geometric stiffness matrix of the membrane
    stress resultants of the linear strains at ``cs``, superposed to
    the constant state ``Nxx0``, ``Nyy0``, ``Nxy0``. The matrix is
    homogeneous of degree one in ``cs``, as the linear buckling
    eigenvalue problem requires; the stress of the non-linear strains
    enters :func:`fkC_num`.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements (the rotations in radians).
    Finput : array-like
        Constitutive matrix of the generalized strains of the model, shape
        ``(NE, NE)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        NE, NE)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.
    Nxx0, Nyy0, Nxy0 : float, optional
        Constant membrane stress resultants, force per unit length, default
        ``0``.

    Returns
    -------
    kG : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape or ``shell`` is not a
        :class:`.Shell`.

    Notes
    -----
    With `\\boldsymbol{{G}} \\boldsymbol{{c}} = \\{{\\beta_x, \\beta_y\\}}^\\top` the
    rotations of the non-linear terms of the membrane strains, and the membrane stress resultants
    `\\boldsymbol{{N}} = \\boldsymbol{{N}}_0 + \\boldsymbol{{N}}_L`,
    `\\boldsymbol{{N}}_L` those of `\\boldsymbol{{F}} \\boldsymbol{{B}}_L
    \\boldsymbol{{c}}`, the linear generalized strains of the kinematics of the module:

    .. math::

        \\boldsymbol{{K}}_G = \\int \\boldsymbol{{G}}^\\top \\boldsymbol{{N}}
        \\boldsymbol{{G}} \\, dA, \\qquad \\boldsymbol{{N}} = \\begin{{bmatrix}} N_{{xx}}
        & N_{{xy}} \\\\ N_{{xy}} & N_{{yy}} \\end{{bmatrix}}

    The tangent stiffness matrix `\\boldsymbol{{K}}_{{C0}} +
    \\boldsymbol{{K}}_{{CNL}} + \\boldsymbol{{K}}_{{GNL}} +
    \\boldsymbol{{K}}_G(\\boldsymbol{{N}}_0 + \\boldsymbol{{N}}_L)`, the first three terms from :func:`fkC_num` and the
    last from this function, is
    the exact Jacobian of :func:`calc_fint` when `\\boldsymbol{{N}}_0 =
    \\boldsymbol{{0}}`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \\le x \\le x_2`, `y_1 \\le y \\le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

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

    # fkM_num, the thickness h of the TSDT gives c1 at each point
    input_names = moment_names + (['h'] if theory == 'tsdt' else [])
    L = []
    for i, j, v in kMn:
        L += entry('kM', i, j, v)
    loops, names = basis_loops('kM', [v for _, _, v in kMn], L)
    NUM.append('''

def fkM_num(object shell, object I_input, int size,
        int row0, int col0, int nx, int ny):
    r"""Mass matrix, integrated numerically

    Integrates numerically the consistent mass matrix of the shell, with
    the moments of the density given for the whole domain or at each
    integration point, see :meth:`.Shell.calc_kM`.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    I_input : array-like
        Moments of the density through the thickness, and for the TSDT the
        thickness ``h``, ``{inames}``, shape ``({K},)`` for the whole domain or ``(nx, ny, {K})`` with one set per
        integration point.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    kM : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``I_input`` has an invalid shape or ``shell`` is not a
        :class:`.Shell`.

    Notes
    -----
    With the displacement field through the thickness `\\{{u(z), v(z),
    w(z)\\}}^\\top = \\boldsymbol{{S}}(z) \\boldsymbol{{c}}` of the kinematics of the module:

    .. math::

        \\boldsymbol{{M}} = \\int \\int_{{d - h/2}}^{{d + h/2}} \\rho \\,
        \\boldsymbol{{S}}(z)^\\top \\boldsymbol{{S}}(z) \\, dz \\, dA

    with the offset `d`, where the integral through the thickness gives the moments of ``I_input``, `I_n = \\int \\rho z^n dz`, `n = 0, 1,
    2`, for the FSDT, and for the TSDT the moments `J_n`, `n = 0, 1, 2, 3, 4,
    6`, about the mid-surface and the thickness `h`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \\le x \\le x_2`, `y_1 \\le y \\le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
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

    # moments of the density through the thickness, {inames}, as a 3-D
    # matrix [nx, ny, {K}] with one set for each of the nx * ny integration
    # points, or as a 1-D array [{K}] for the whole integration domain
    cdef double {icdef}{dcdef}
    cdef double [:, :, ::1] I_nxny

    cdef int one_set_each_point = 0

    I_input = np.asarray(I_input, dtype=DOUBLE)
    if I_input.shape == (nx, ny, {K}):
        I_nxny = np.ascontiguousarray(I_input)
        one_set_each_point = 1
    elif I_input.shape == ({K}, ):
        # creating dummy 3-D array that is not used
        I_nxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        # using constant moments for all the integration domain
        {icdef} = I_input
    else:
        raise ValueError('Invalid shape for I_input, it must be (nx, ny, {K}) or ({K}, )')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
{dread}{LIMITS}
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
                if one_set_each_point == 1:
{iread}
{c1pt}
{loops}

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM
'''.format(c1=', c1' if theory == 'tsdt' else '', BC_CDEF=BC_CDEF,
           basis=basis_cdefs(set(names)), LIMITS=LIMITS, BC_BLOCK=BC_BLOCK,
           nnz=len(kMn), MAPPING=MAPPING, POINT=POINT, loops=loops,
           K=len(input_names), inames=', '.join(input_names),
           icdef=', '.join(input_names),
           dcdef=(', d' if theory == 'tsdt' else ''),
           dread=('    d = shell.offset\n' if theory == 'tsdt' else ''),
           iread='\n'.join('                    {0} = I_nxny[ptx, pty, {1}]'.format(name, i)
                           for i, name in enumerate(input_names)),
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
    r"""Aerodynamic stiffness matrix of the piston theory, integrated numerically

    ``fkAx_num`` gives the matrix of the flow along `x` and ``fkAy_num``
    that of the flow along `y`, with the parameters ``shell.beta`` and,
    for the flow along `x`, ``shell.gamma`` of :meth:`.Shell.calc_kA`.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this shell start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    kA : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. The matrix is assembled in full.

    Raises
    ------
    ValueError
        If ``shell`` is not a :class:`.Shell`.

    Notes
    -----
    With `\\boldsymbol{{S}}^w` the row of the approximation functions of
    `w`, the matrices added to the stiffness matrix are:

    .. math::

        \\boldsymbol{{K}}_{{A_x}} = -\\int \\left[\\beta (\\boldsymbol{{S}}^w_{{,x}})^\\top
        \\boldsymbol{{S}}^w + \\gamma (\\boldsymbol{{S}}^w)^\\top
        \\boldsymbol{{S}}^w\\right] dA \\qquad
        \\boldsymbol{{K}}_{{A_y}} = -\\beta \\int (\\boldsymbol{{S}}^w_{{,y}})^\\top
        \\boldsymbol{{S}}^w \\, dA

    `\\gamma = 0` for the flat plates; they are not symmetric.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \\le x \\le x_2`, `y_1 \\le y \\le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
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
    r"""Internal force vector, integrated numerically

    Evaluates the generalized strains and stress resultants at
    ``cs`` at each integration point and integrates the internal force
    vector, whose Jacobian is ``fkC_num(NLgeom=1) + fkG_num``.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements (the rotations in radians).
    Finput : array-like
        Constitutive matrix of the generalized strains of the model, shape
        ``(NE, NE)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        NE, NE)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
    size : int
        Size of the vector, the number of degrees of freedom of the assembly.
    col0 : int
        Position of the vector where the terms of this shell start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    fint : numpy.ndarray
        Internal force vector of size ``size``.

    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape or ``shell`` is not a
        :class:`.Shell`.

    Notes
    -----
    With the generalized strains of the kinematics of the module,
    `\\boldsymbol{{\\varepsilon}}(\\boldsymbol{{c}})`, their variation `\\delta
    \\boldsymbol{{\\varepsilon}} = \\boldsymbol{{B}}(\\boldsymbol{{c}}) \\delta
    \\boldsymbol{{c}}` and the stress resultants `\\boldsymbol{{\\sigma}} =
    \\boldsymbol{{F}} \\boldsymbol{{\\varepsilon}}`:

    .. math::

        \\boldsymbol{{f}}_{{int}} = \\int \\boldsymbol{{B}}(\\boldsymbol{{c}})^\\top
        \\boldsymbol{{\\sigma}} \\, dA

    the gradient of the strain energy `\\frac{{1}}{{2}} \\int
    \\boldsymbol{{\\varepsilon}}^\\top \\boldsymbol{{F}} \\boldsymbol{{\\varepsilon}}
    \\, dA`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \\le x \\le x_2`, `y_1 \\le y \\le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
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
        #NOTE the follower-pressure kernels, see
        #     theory/shells/follower_pressure/write_pyx.py
        #NOTE the approximation functions evaluated once per integration
        #     point, see panels/dev/tabulate.py
        f.write(tabulate_basis(with_radius(''.join(NUM), cyl)) + '\n\n'
                + follower_block(model)
                + '\n\n' + tangent_block(model))
    print(model, 'numerical written', time.time() - t0)


if __name__ == '__main__':
    for model in sys.argv[1:] or th.MODELS:
        generate(model)
