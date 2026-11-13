"""Generate the kernels of the model ``plate_clpt_gl``

Writes ``panels/models/plate_clpt_gl.pyx`` (analytical matrices) and
``panels/models/plate_clpt_gl_num.pyx`` (numerically integrated matrices,
internal force vector, follower-pressure kernels and product of the tangent
stiffness matrix by a vector) from the derivation of ``plate_clpt_gl.py``::

    python write_pyx.py

The linear kernels are those of ``plate_clpt_donnell``, whose strains are
the same once the quadratic terms are dropped, and are imported from its
modules: ``fk0``, ``fkM``, ``fkAx``, ``fkAy``, ``fcA``, ``fkM_num``,
``fkAx_num`` and ``fkAy_num``. The geometric stiffness matrices, ``fkG0``
and ``fkG_num``, have ``u-u`` and ``v-v`` blocks besides the ``w-w`` block
of ``plate_clpt_donnell``.

Kernel layout
-------------
At each integration point, the nine generalized derivatives ``d = [Phi] c``
(``plate_clpt_gl.D``), the 6 x 9 matrix ``[B] = d eps/d d`` and the 9 x 9
matrix ``K = [B]^T [ABD] [B] + [H(N)]`` are evaluated once; the entry of the
pair of terms ``A = (i, j)``, ``B = (k, l)`` and the degrees of freedom ``d``
and ``e`` is then ``sum_pq Phi_A[p, d] K[p, q] Phi_B[q, e]``, computed as
``TA[d, q] = sum_p Phi_A[p, d] K[p, q]`` in the loop over the terms ``A`` and
``sum_q TA[d, q] Phi_B[q, e]`` in the loop over the terms ``B``. The
approximation functions are tabulated once per integration point, with the
derivatives already in the physical coordinates, ``(2/a) f'(xi)``,
``(2/a)^2 f''(xi)``, named ``fAu_x``, ``fAw_xx``, ``gAw_yy`` and so on.

The helpers take a :class:`Spec`, the derivation of a model and the rows of
its tables, ``SPEC`` (this model) by default; ``plate_clpt_tl/write_pyx.py``
uses them with its own derivation.

"""
import importlib.util
import os
import re
import sys
import textwrap

import numpy as np
import sympy
from sympy import Add, Float, Mul, Rational, Symbol, expand, zeros
from sympy.printing.str import StrPrinter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import plate_clpt_gl as th
from panels.dev.matrixtools import pow2mult


def _load(name, *path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, *path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#NOTE the follower-pressure kernels and the kernel of the product of the
#     tangent stiffness matrix by a vector, appended to the numerical module
follower_block = _load('follower_write_pyx', '..', 'follower_pressure',
                       'write_pyx.py').pyx_block
tangent_block = _load('tangent_write_pyx', '..', 'tangent_product',
                      'write_pyx.py').pyx_block


# Integrals of the approximation functions of the analytical kernels
# ------------------------------------------------------------------
FIELD_RE = '(u|v|w)'
BASIS = re.compile(r'^([fg])([AB])' + FIELD_RE + r'((?:_(?:xi|eta)+)?)$')
KIND = {(0, 0): 'ff', (0, 1): 'ffp', (0, 2): 'ffpp',
        (1, 1): 'fpfp', (1, 2): 'fpfpp', (2, 2): 'fppfpp'}


def basis_info(name):
    mo = BASIS.match(name)
    if mo is None:
        return None
    fg, side, field, der = mo.groups()
    order = der.count('xi') if fg == 'f' else der.count('eta')
    return fg, side, field, order


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
    r"""Products of the basis functions of a term ``A`` and a term ``B``
    replaced by the one-dimensional integrals of their products"""
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


def ordered_integrals(names):
    return sorted(names, key=lambda s: (s[:2], s))


MODEL = 'plate_clpt_gl'
DOF = th.DOF
NE = th.NE
ND = len(th.D)
FIELDS = th.FIELDS

#: rows of the tables of the approximation functions, along x and y
F_ROWS = ['u', 'u_x', 'v', 'v_x', 'w', 'w_x', 'w_xx']
G_ROWS = ['u', 'u_y', 'v', 'v_y', 'w', 'w_y', 'w_yy']


class Printer(StrPrinter):
    def _print_Float(self, e):
        return repr(float(e))


def pstr(e):
    reps = {q: Float(q) for q in e.atoms(Rational) if not q.is_Integer}
    return pow2mult(Printer().doprint(e.xreplace(reps)))


def physical(e):
    r"""Expression of the basis functions in the physical derivatives,
    ``(2/a) fAu_xi -> fAu_x``, ``(2/b)**2 gAw_etaeta -> gAw_yy``"""
    rename = {}
    for s in e.free_symbols:
        mo = re.match(r'^([fg])([AB])(u|v|w)((?:_(?:xi|eta)+)?)$', s.name)
        if mo is None:
            continue
        fg, side, field, der = mo.groups()
        order = der.count('xi') + der.count('eta')
        new = '{0}{1}{2}{3}'.format(fg, side, field,
                                    '' if order == 0 else '_' + ('x' if fg == 'f' else 'y')*order)
        rename[s] = Symbol(new)
    out = e.xreplace(rename).subs({th.a: 2, th.b: 2})
    return expand(out)


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
BC_BLOCK = '\n'.join('    {0}1{1} = shell.{0}1{1}; {0}1{1}r = shell.{0}1{1}r; '
                     '{0}2{1} = shell.{0}2{1}; {0}2{1}r = shell.{0}2{1}r'.format(xy, p)
                     for xy in 'xy' for p in FIELDS)

FINPUT_TEMPLATE = """    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()
""" + BC_BLOCK + """

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    xinf = 0
    xsup = shell.a
    xi1 = (x1 - xinf)/(xsup - xinf)*2 - 1
    xi2 = (x2 - xinf)/(xsup - xinf)*2 - 1

    yinf = 0
    ysup = shell.b
    eta1 = (y1 - yinf)/(ysup - yinf)*2 - 1
    eta2 = (y2 - yinf)/(ysup - yinf)*2 - 1

    intx = x2 - x1
    inty = y2 - y1

    ftab = np.zeros((NF, m), dtype=DOUBLE)
    gtab = np.zeros((NG, n), dtype=DOUBLE)"""

COMMON_CDEF = """    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, intx, inty
    cdef int m, n
""" + BC_CDEF + """

    cdef int i, j, k, l, c, p, q, r, s, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
"""

POINT_HEAD = """    with nogil:
        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                # the integrands are per unit area of the natural domain
                weight = weights_xi[ptx]*weights_eta[pty]*intx*inty/4

                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
"""


class Spec(object):
    r"""Derivation of a model and rows of the tables of the approximation
    functions along ``x`` (``f_rows``) and ``y`` (``g_rows``), e.g.
    ``'w_xx'`` for `(2/a)^2 f''(\xi)` of ``w``

    ``th`` is the module of the derivation, with ``build()``, ``D``,
    ``D_NAMES``, ``DOF`` and ``F``.

    """
    def __init__(self, th, f_rows, g_rows):
        self.th = th
        self.F_ROWS = list(f_rows)
        self.G_ROWS = list(g_rows)
        self.ND = len(th.D)
        self.DOF = th.DOF
        self.D_NAMES = th.D_NAMES
        self.M = th.build()
        self.PHI_A = self.M['PhiA'].applyfunc(physical)
        self.PHI_B = self.M['PhiB'].applyfunc(physical)
        #: the generalized derivatives of each degree of freedom
        self.DSET = [[p for p in range(self.ND) if self.PHI_A[p, d] != 0]
                     for d in range(self.DOF)]
        self.DOF_OF = {p: d for d in range(self.DOF) for p in self.DSET[d]}


def table_lines(sp):
    r"""Tabulation of the rows of ``sp`` at the integration point"""
    L = []
    for fg, rows, idx, coord, ab in (('f', sp.F_ROWS, 'i', 'xi', 'a'),
                                     ('g', sp.G_ROWS, 'j', 'eta', 'b')):
        xy = 'x' if fg == 'f' else 'y'
        L.append('                for {0} in range({1}):'.format(idx, 'm' if fg == 'f' else 'n'))
        for k, row in enumerate(rows):
            field = row[0]
            order = row.count(xy) if '_' in row else 0
            func = ('f', 'fp', 'fpp')[order]
            L.append('                    {0}tab[{1}, {2}] = {3}{4}({2}, {5}, {6}1{7}, '
                     '{6}1{7}r, {6}2{7}, {6}2{7}r)'.format(
                         fg, k, idx, '(2/{0})*'.format(ab)*order, func, coord, xy, field))
    return '\n'.join(L) + '\n'


def finput(sp):
    r"""Reading of the input and allocation of the tables of ``sp``"""
    return FINPUT_TEMPLATE.replace('(NF, m)', '(%d, m)' % len(sp.F_ROWS)).replace(
        '(NG, n)', '(%d, n)' % len(sp.G_ROWS))


def point(sp):
    r"""Loop over the integration points and tabulation of ``sp``"""
    return POINT_HEAD + table_lines(sp)


SPEC = Spec(th, F_ROWS, G_ROWS)
M = SPEC.M
PHI_A = SPEC.PHI_A
PHI_B = SPEC.PHI_B
DSET = SPEC.DSET
DOF_OF = SPEC.DOF_OF
D_NAMES = SPEC.D_NAMES
FINPUT = finput(SPEC)
POINT = point(SPEC)



def basis_read(side, fg, indent, sp=None):
    sp = SPEC if sp is None else sp
    ind = ' '*indent
    rows = sp.F_ROWS if fg == 'f' else sp.G_ROWS
    idx = {('A', 'f'): 'i', ('A', 'g'): 'j', ('B', 'f'): 'k', ('B', 'g'): 'l'}[(side, fg)]
    tab = 'ftab' if fg == 'f' else 'gtab'
    return [ind + '{0}{1}{2} = {3}[{4}, {5}]'.format(fg, side, nm, tab, k, idx)
            for k, nm in enumerate(rows)]


def basis_cdef(sides='AB', sp=None):
    sp = SPEC if sp is None else sp
    names = []
    for side in sides:
        names += ['f{0}{1}'.format(side, nm) for nm in sp.F_ROWS]
        names += ['g{0}{1}'.format(side, nm) for nm in sp.G_ROWS]
    return '\n'.join('    cdef double ' + ', '.join(names[i:i + 7])
                     for i in range(0, len(names), 7))


def phi_lines(side, indent, sp=None):
    r"""``pA[p] = Phi_A[p, d]``, the derivative ``p`` of the term"""
    sp = SPEC if sp is None else sp
    P = sp.PHI_A if side == 'A' else sp.PHI_B
    return ['{0}p{1}[{2}] = {3}'.format(' '*indent, side, p, pstr(P[p, sp.DOF_OF[p]]))
            for p in range(sp.ND)]


def derivative_lines(indent, names=None, sp=None, vec='cs', prefix=''):
    r"""``d[p] += Phi_A[p, d] cs[col+d]``, the derivatives at the point, or
    those of the vector ``vec``, named with ``prefix``"""
    sp = SPEC if sp is None else sp
    names = range(sp.ND) if names is None else names
    return ['{0}{1}{2} += {3}*{4}[col+{5}]'.format(' '*indent, prefix, sp.D_NAMES[p],
            pstr(sp.PHI_A[p, sp.DOF_OF[p]]), vec, sp.DOF_OF[p]) for p in names]


def assembly(mname, pattern, blocks, symmetric=True, sp=None):
    r"""Loops over the pairs of terms, ``mname`` the name of the matrix,
    ``pattern[p][q]`` the non-zero entries of ``K`` and ``blocks`` the pairs
    of degrees of freedom ``(d, e)`` of the matrix"""
    sp = SPEC if sp is None else sp
    DOF, ND, DSET = sp.DOF, sp.ND, sp.DSET
    T = {}
    for d in range(DOF):
        for q in range(ND):
            ps = [p for p in DSET[d] if pattern[p][q]]
            if ps:
                T[(d, q)] = ' + '.join('pA[{0}]*K[{1}]'.format(p, p*ND + q) for p in ps)
    L = ['                c = -1',
         '                for j in range(n):']
    L += basis_read('A', 'g', 20, sp)
    L += ['                    for i in range(m):']
    L += basis_read('A', 'f', 24, sp)
    L += ['                        row = row0 + DOF*(j*m + i)']
    L += phi_lines('A', 24, sp)
    L += ['                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]']
    used_d = sorted({d for d, e in blocks})
    for (d, q), expr in sorted(T.items()):
        if d in used_d:
            L.append('                        TA[{0}] = {1}'.format(d*ND + q, expr))
    L += ['                        for l in range(n):']
    L += basis_read('B', 'g', 28, sp)
    L += ['                            for k in range(m):',
          '                                col = col0 + DOF*(l*m + k)']
    if symmetric:
        L += ['                                #NOTE symmetry assumption True if no follower forces are used',
              '                                if row > col:',
              '                                    continue']
    L += basis_read('B', 'f', 32, sp)
    L += phi_lines('B', 32, sp)
    for d, e in blocks:
        terms = ['TA[{0}]*pB[{1}]'.format(d*ND + q, q) for q in DSET[e] if (d, q) in T]
        if not terms:
            continue
        L += ['                                c += 1',
              '                                if ptx == 0 and pty == 0:',
              '                                    {0}r[c] = row+{1}'.format(mname, d),
              '                                    {0}c[c] = col+{1}'.format(mname, e),
              '                                {0}v[c] += weight*({1})'.format(mname, ' + '.join(terms))]
    return '\n'.join(L)


def matrix_lines(name, Mx, indent, op='='):
    sp = ' '*indent
    out = []
    rows, cols = Mx.shape
    for p in range(rows):
        for q in range(cols):
            if Mx[p, q] != 0:
                out.append('{0}{1}[{2}] {3} {4}'.format(sp, name, p*cols + q, op, pstr(Mx[p, q])))
    return out


def const_read(names, indent):
    F = th.F
    out = []
    seen = set()
    for i in range(6):
        for j in range(i, 6):
            nm = F[i, j].name
            if nm in names and nm not in seen:
                seen.add(nm)
                out.append('{0}{1} = F[{2}*6 + {3}]'.format(' '*indent, nm, i, j))
    return out


def gradient_loop(indent, names, sp=None, extra=()):
    r"""Loop over the terms summing the derivatives ``names`` at the point;
    ``extra`` holds ``(vec, prefix)`` of further vectors to differentiate"""
    ind = ' '*indent
    L = [ind + 'for j in range(n):']
    L += basis_read('A', 'g', indent + 4, sp)
    L += [ind + '    for i in range(m):']
    L += basis_read('A', 'f', indent + 8, sp)
    L += [ind + '        col = col0 + DOF*(j*m + i)']
    L += derivative_lines(indent + 8, names, sp)
    for vec, prefix in extra:
        L += derivative_lines(indent + 8, names, sp, vec, prefix)
    return L


DESC = ('Flat plates using the classical laminated plate theory (CLPT) with '
        'Green-Lagrange membrane strains')


def numerical_module():
    B = M['B']
    H = M['H']
    eps = M['eps']
    eps_L = M['eps_L']
    N_NL = M['N_NL']
    full = [[True]*ND for _ in range(ND)]
    Hpat = [[H[p, q] != 0 for q in range(ND)] for p in range(ND)]
    blocks_all = [(d, e) for d in range(DOF) for e in range(DOF)]
    #NOTE the pairs of degrees of freedom coupled by [H(N)], the diagonal
    #     ones u-u, v-v, w-w for the plate
    blocks_G = [(d, e) for d in range(DOF) for e in range(DOF)
                if any(Hpat[p][q] for p in DSET[d] for q in DSET[e])]
    NxxNL, NyyNL, NxyNL = sympy.symbols('NxxNL, NyyNL, NxyNL')
    H_NL = H.xreplace({th.Nxx: NxxNL, th.Nyy: NyyNL, th.Nxy: NxyNL})
    #NOTE the derivatives of the non-linear terms of the strains, the six
    #     gradients for the plate
    grads = [D_NAMES.index(g.name) for g in th.GRAD]
    sym_names = ', '.join(D_NAMES)

    out = [HEADER + 'r"""\n' + textwrap.fill(DESC, 76) + '''

Numerically integrated kernels of the model ``'plate_clpt_gl'``: the
constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the internal force vector :func:`calc_fint`, the product
of the tangent stiffness matrix by a vector :func:`calc_kTv`, and the force
vector and load stiffness matrix of a follower pressure,
:func:`calc_fext_follower` and :func:`fkCfollower_num`. The mass and
aerodynamic matrices, ``fkM_num``, ``fkAx_num`` and ``fkAy_num``, are those
of :mod:`panels.models.plate_clpt_donnell_num`. The constitutive matrix
``Finput`` is the ``6 x 6`` matrix ``ABD`` of the CLPT.

The degrees of freedom of each term of the approximation are
``u, v, w``; the degree of freedom ``k`` of the term ``(i, j)``, ``0 <=
i < m``, ``0 <= j < n``, is at the row ``row0 + 3*(j*m + i) + k``
of the matrices.

Notes
-----
The membrane strains are the Green-Lagrange strains, which vanish for
any rigid rotation:

.. math::

    \\varepsilon_{xx} = u_{,x} + \\frac{1}{2} (u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2) \\qquad
    \\varepsilon_{yy} = v_{,y} + \\frac{1}{2} (u_{,y}^2 + v_{,y}^2 +
    w_{,y}^2)

    \\gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} + w_{,x}
    w_{,y}

and the changes of curvature are those of ``plate_clpt_donnell``,
`\\kappa_{xx} = -w_{,xx}`, `\\kappa_{yy} = -w_{,yy}`, `\\kappa_{xy} = -2
w_{,xy}`, see :mod:`panels.models`.

The numerical integration over the domain of
:meth:`.Shell.integration_limits` admits a constitutive matrix and moments
of the density varying over the domain, a partial domain and the
non-linear terms of the Ritz constants.

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
#NOTE the linear kernels, independent of the strains, are those of the von
#     Karman plate
from .plate_clpt_donnell_num import fkM_num, fkAx_num, fkAy_num


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 3
''']

    # ------------------------------------------------------------- fkC_num
    out.append('''

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
        ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        6, 6)`` with one matrix per integration point.
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

    with `\\boldsymbol{{G}} \\boldsymbol{{c}} = \\{{u_{{,x}}, u_{{,y}}, v_{{,x}}, v_{{,y}},
    w_{{,x}}, w_{{,y}}\\}}^\\top` the gradients of the displacements, with one
    block `\\boldsymbol{{N}}` per displacement, and `\\boldsymbol{{N}}_{{NL}}` the membrane stress resultants of the
    non-linear part of the strains, `\\frac{{1}}{{2}} \\boldsymbol{{B}}_{{NL}}
    \\boldsymbol{{c}}`. With ``NLgeom=0``, `\\boldsymbol{{B}}_{{NL}} = \\boldsymbol{{0}}`
    and `\\boldsymbol{{K}}_C = \\boldsymbol{{K}}_{{C0}}`. The membrane stress of the
    linear strains enters :func:`fkG_num`, such that ``fkC_num(NLgeom=1) +
    fkG_num`` is the exact Jacobian of :func:`calc_fint`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \\le x \\le x_2`, `y_1 \\le y \\le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
{COMMON}
    cdef double {gr}
    cdef double A11, A12, A16, A22, A26, A66
    cdef double NxxNL, NyyNL, NxyNL
{basis}
    cdef double Bm[{NB}]
    cdef double FB[{NB}]
    cdef double K[{NK}]
    cdef double TA[{NT}]
    cdef double pA[{ND}]
    cdef double pB[{ND}]

    cdef long [::1] kCr, kCc
    cdef double [::1] kCv

{FINPUT}

    fdim = 9*m*m*n*n
    kCr = np.zeros((fdim,), dtype=INT)
    kCc = np.zeros((fdim,), dtype=INT)
    kCv = np.zeros((fdim,), dtype=DOUBLE)

{POINT}
                # displacement gradients at the point, zero when NLgeom == 0
{gzero}
                if NLgeom == 1:
{gloop}

{Aread}

                # [B] = d eps/d d at the point
                for p in range({NB}):
                    Bm[p] = 0.
{Blines}

                # K = [B]^T [ABD] [B] + [H(N_NL)]
                for r in range(6):
                    for q in range({ND}):
                        FB[r*{ND} + q] = 0.
                        for s in range(6):
                            FB[r*{ND} + q] += F[r*6 + s]*Bm[s*{ND} + q]
                for p in range({ND}):
                    for q in range({ND}):
                        K[p*{ND} + q] = 0.
                        for r in range(6):
                            K[p*{ND} + q] += Bm[r*{ND} + p]*FB[r*{ND} + q]
                # membrane stresses of the quadratic strains, whose
                # geometric stiffness KGNL is collected here such that
                # fkG_num is homogeneous of degree one in cs, as linear
                # buckling requires
{NNL}
{HNL}

{loops}

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC
'''.format(COMMON=COMMON_CDEF, gr=', '.join(D_NAMES[p] for p in grads), basis=basis_cdef(),
           NB=NE*ND, NK=ND*ND, NT=DOF*ND, ND=ND, FINPUT=FINPUT, POINT=POINT,
           gzero='\n'.join('                {0} = 0.'.format(D_NAMES[p]) for p in grads),
           gloop='\n'.join(gradient_loop(20, grads)),
           Aread='\n'.join(const_read({'A11', 'A12', 'A16', 'A22', 'A26', 'A66'}, 16)),
           Blines='\n'.join(matrix_lines('Bm', B, 16)),
           NNL='\n'.join('                {0} = {1}'.format(nm, pstr(N_NL[k]))
                         for k, nm in enumerate(('NxxNL', 'NyyNL', 'NxyNL'))),
           HNL='\n'.join(matrix_lines('K', H_NL, 16, op='+=')),
           loops=assembly('kC', full, blocks_all)))

    # ------------------------------------------------------------- fkG_num
    N_L = (th.F[:3, :]*eps_L)
    out.append('''

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
        ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        6, 6)`` with one matrix per integration point.
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
    With `\\boldsymbol{{G}} \\boldsymbol{{c}} = \\{{u_{{,x}}, u_{{,y}}, v_{{,x}}, v_{{,y}},
    w_{{,x}}, w_{{,y}}\\}}^\\top` the gradients of the displacements, with one
    block `\\boldsymbol{{N}}` per displacement, and the membrane stress resultants
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
{COMMON}
    cdef double {dn}
    cdef double Nxx, Nyy, Nxy
    cdef double e[6]
{basis}
    cdef double K[{NK}]
    cdef double TA[{NT}]
    cdef double pA[{ND}]
    cdef double pB[{ND}]

    cdef long [::1] kGr, kGc
    cdef double [::1] kGv

{FINPUT}

    fdim = {NG}*m*m*n*n
    kGr = np.zeros((fdim,), dtype=INT)
    kGc = np.zeros((fdim,), dtype=INT)
    kGv = np.zeros((fdim,), dtype=DOUBLE)

{POINT}
                # generalized derivatives at the point
{dzero}
{dloop}

                # linear strains and membrane stresses
{elines}
                Nxx = Nxx0
                Nyy = Nyy0
                Nxy = Nxy0
                for s in range(6):
                    Nxx += F[0*6 + s]*e[s]
                    Nyy += F[1*6 + s]*e[s]
                    Nxy += F[2*6 + s]*e[s]

                # K = [H(N)]
{Hlines}

{loops}

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG
'''.format(COMMON=COMMON_CDEF, dn=', '.join(D_NAMES), basis=basis_cdef(),
           NK=ND*ND, NT=DOF*ND, ND=ND, FINPUT=FINPUT, POINT=POINT,
           dzero='\n'.join('                {0} = 0.'.format(nm) for nm in D_NAMES),
           dloop='\n'.join(gradient_loop(16, range(ND))),
           elines='\n'.join('                e[{0}] = {1}'.format(k, pstr(eps_L[k])) for k in range(NE)),
           Hlines='\n'.join(matrix_lines('K', H, 16)),
           NG=len(blocks_G), loops=assembly('kG', Hpat, blocks_G)))

    out.append(calc_fint_src())
    text = ''.join(out)
    #NOTE c, p, q, r, s, k, l and parts of the tables are not used in every
    #     kernel, which is harmless
    return text


def calc_fint_src(sp=None):
    r"""Source of ``calc_fint`` of ``sp``, ``int [Phi]^T [B]^T sigma dA``,
    for any strains ``eps(d)`` of the CLPT"""
    sp = SPEC if sp is None else sp
    B, eps = sp.M['B'], sp.M['eps']
    DOF, ND, D_NAMES, NE = sp.DOF, sp.ND, sp.D_NAMES, sp.th.NE
    S = (B.T*Matrix_s()).applyfunc(expand)
    fint_lines = []
    for d in range(DOF):
        terms = ['pA[{0}]*S[{0}]'.format(p) for p in sp.DSET[d]]
        fint_lines.append('                        fint[col+{0}] += weight*({1})'.format(d, ' + '.join(terms)))
    return '''

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
        ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        6, 6)`` with one matrix per integration point.
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
{COMMON}
    cdef double {dn}
    cdef double e[6]
    cdef double sg[6]
    cdef double S[{ND}]
    cdef double pA[{ND}]
{basis}

    cdef double [::1] fint

{FINPUT}

    fint = np.zeros(size, dtype=DOUBLE)

{POINT}
                # generalized derivatives at the point
{dzero}
{dloop}

                # strains, stresses and S = [B]^T sigma
{elines}
                for r in range(6):
                    sg[r] = 0.
                    for s in range(6):
                        sg[r] += F[r*6 + s]*e[s]
{Slines}

                for j in range(n):
{gA}
                    for i in range(m):
{fA}
                        col = col0 + DOF*(j*m + i)
{pA}
{fint}

    return fint
'''.format(COMMON=COMMON_CDEF, dn=', '.join(D_NAMES), ND=ND,
           basis=basis_cdef('A', sp), FINPUT=finput(sp), POINT=point(sp),
           dzero='\n'.join('                {0} = 0.'.format(nm) for nm in D_NAMES),
           dloop='\n'.join(gradient_loop(16, range(ND), sp)),
           elines='\n'.join('                e[{0}] = {1}'.format(k, pstr(eps[k])) for k in range(NE)),
           Slines='\n'.join('                S[{0}] = {1}'.format(p, pstr(S[p])) for p in range(ND)),
           gA='\n'.join(basis_read('A', 'g', 20, sp)), fA='\n'.join(basis_read('A', 'f', 24, sp)),
           pA='\n'.join(phi_lines('A', 24, sp)), fint='\n'.join(fint_lines))


def Matrix_s():
    return sympy.Matrix(sympy.symbols('sg0:6')).xreplace(
        {Symbol('sg%d' % k): Symbol('sg[%d]' % k) for k in range(6)})


def analytical_module():
    r"""``fkG0``, and the other analytical kernels imported from
    ``plate_clpt_donnell``"""
    H = M['H']
    a, b = th.a, th.b
    kG = (a*b/4)*M['PhiA'].T*H*M['PhiB']
    ent = [(i, j, to_integrals(kG[i, j])) for i in range(DOF) for j in range(DOF)
           if kG[i, j] != 0]
    fnames, gnames = set(), set()
    for _, _, v in ent:
        for s in v.free_symbols:
            if s.name.startswith('fA'):
                fnames.add(s.name)
            elif s.name.startswith('gA'):
                gnames.add(s.name)
    fnames = ordered_integrals(fnames)
    gnames = ordered_integrals(gnames)

    def cdefs(names):
        return '\n'.join('    cdef double ' + ', '.join(names[i:i + 4])
                         for i in range(0, len(names), 4))

    L = [HEADER + 'r"""\n' + textwrap.fill(DESC, 76) + '''

Analytical kernels of the model ``'plate_clpt_gl'``, integrated over
the whole domain. The linear constitutive stiffness ``fk0``, the mass
``fkM`` and the aerodynamic ``fkAx``, ``fkAy`` and ``fcA`` matrices are
those of :mod:`panels.models.plate_clpt_donnell`. The geometric stiffness
matrix :func:`fkG0` of the constant membrane stress state ``Nxx, Nyy,
Nxy`` has one block per displacement ``u, v, w``.

Notes
-----
The membrane strains are the Green-Lagrange strains, which vanish for
any rigid rotation:

.. math::

    \\varepsilon_{xx} = u_{,x} + \\frac{1}{2} (u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2) \\qquad
    \\varepsilon_{yy} = v_{,y} + \\frac{1}{2} (u_{,y}^2 + v_{,y}^2 +
    w_{,y}^2)

    \\gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} + w_{,x}
    w_{,y}

and the changes of curvature are those of ``plate_clpt_donnell``,
`\\kappa_{xx} = -w_{,xx}`, `\\kappa_{yy} = -w_{,yy}`, `\\kappa_{xy} = -2
w_{,xy}`, see :mod:`panels.models`.

The integrals over the whole domain are evaluated in closed form with the
integrals of Bardell's functions and their derivatives along `\\xi = 2x/a -
1` and `\\eta = 2y/b - 1`, which requires the constitutive matrix and the
moments of the density to be constant over the domain.

"""
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE
#NOTE the linear kernels, independent of the strains, are those of the von
#     Karman plate
from .plate_clpt_donnell import fk0, fkM, fkAx, fkAy, fcA


cdef int DOF = 3


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil


def fkG0(double Nxx, double Nyy, double Nxy, object shell,
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
    With `\\boldsymbol{N} = \\begin{bmatrix} N_{xx} & N_{xy} \\\\ N_{xy} &
    N_{yy} \\end{bmatrix}` and `\\boldsymbol{G} \\boldsymbol{c} = \\{u_{,x}, u_{,y}, v_{,x}, v_{,y},
    w_{,x}, w_{,y}\\}^\\top` the gradients of the displacements, with one
    block `\\boldsymbol{N}` per displacement:

    .. math::

        \\boldsymbol{K}_{G} = \\int_0^b \\int_0^a \\boldsymbol{G}^\\top
        \\boldsymbol{N} \\boldsymbol{G} \\, dx \\, dy

    which is the Hessian of the work of the constant stress resultants on
    the non-linear part of the membrane strains of the kinematics of the module.

    The integral over the whole domain is evaluated in closed form, each
    entry being a sum of products of one-dimensional integrals of Bardell's
    functions and their derivatives along `\\xi` and `\\eta`, with `x = a(\\xi +
    1)/2`, `y = b(\\eta + 1)/2` and `dx\\,dy = (ab/4)\\,d\\xi\\,d\\eta`.

    """
    cdef double a, b
    cdef int m, n
''' + BC_CDEF + '''

    cdef int i, j, k, l, c, row, col

    cdef long [:] kG0r, kG0c
    cdef double [:] kG0v

''' + cdefs(fnames) + '\n' + cdefs(gnames) + '''

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
''' + BC_BLOCK + '''

    fdim = {0}*m*m*n*n

    kG0r = np.zeros((fdim,), dtype=INT)
    kG0c = np.zeros((fdim,), dtype=INT)
    kG0v = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
        # kG0
        c = -1
        for i in range(m):
            for k in range(m):
'''.format(len(ent))]
    L += ['                {0} = {1}'.format(s, integral_call(s)) for s in fnames]
    L += ['', '                for j in range(n):', '                    for l in range(n):', '',
          '                        row = row0 + DOF*(j*m + i)',
          '                        col = col0 + DOF*(l*m + k)', '',
          '                        #NOTE symmetry', '                        if row > col:',
          '                            continue', '']
    L += ['                        {0} = {1}'.format(s, integral_call(s)) for s in gnames]
    L.append('')
    for i, j, v in ent:
        L += ['                        c += 1',
              '                        kG0r[c] = row+{0}'.format(i),
              '                        kG0c[c] = col+{0}'.format(j),
              '                        kG0v[c] += {0}'.format(pstr(sympy.factor_terms(expand(v))))]
    L += ['', '    kG0 = coo_matrix((kG0v, (kG0r, kG0c)), shape=(size, size))', '',
          '    return kG0', '']
    return '\n'.join(L)


def generate():
    path = os.path.join(REPO, 'panels', 'models', MODEL + '.pyx')
    with open(path, 'w', newline='\n') as f:
        f.write(analytical_module())
    print(MODEL, 'analytical written')
    path = os.path.join(REPO, 'panels', 'models', MODEL + '_num.pyx')
    with open(path, 'w', newline='\n') as f:
        f.write(numerical_module() + '\n\n' + follower_block(MODEL)
                + '\n\n' + tangent_block(MODEL))
    print(MODEL, 'numerical written')


if __name__ == '__main__':
    generate()
