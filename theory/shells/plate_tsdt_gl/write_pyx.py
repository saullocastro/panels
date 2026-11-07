"""Generate the kernels of the model ``plate_tsdt_gl``

Writes ``panels/models/plate_tsdt_gl.pyx`` (analytical geometric
stiffness matrix) and ``panels/models/plate_tsdt_gl_num.pyx``
(numerically integrated matrices, internal force vector, product of the
tangent stiffness matrix by a vector and follower-pressure kernels) from the
derivation of ``plate_tsdt_gl.py``::

    python write_pyx.py

The linear kernels are those of ``plate_tsdt_donnell``, whose strains are
the same once the quadratic terms are dropped (``check_linear_part``), and
are imported from its modules: ``fk0``, ``fkM``, ``fkAx``, ``fkAy``,
``fcA``, ``fkM_num``, ``fkAx_num`` and ``fkAy_num``.

Kernel layout
-------------
At each integration point the generalized derivatives ``d = [Phi] c``
(``plate_tsdt_gl.D``), ``[B] = d eps/d d`` and ``K = [B]^T [F] [B] +
[H]`` are evaluated once; the entry of the pair of terms ``A = (i, j)``,
``B = (k, l)`` and the degrees of freedom ``d`` and ``e`` is ``sum_pq
Phi_A[p, d] K[p, q] Phi_B[q, e]``, computed as ``TA[d, q] = sum_p Phi_A[p,
d] K[p, q]`` in the loop over the terms ``A`` and ``sum_q TA[d, q] Phi_B[q,
e]`` in the loop over the terms ``B``. The approximation functions are
tabulated once per integration point, with the derivatives in the physical
coordinates, ``(2/a) f'(xi)``, named ``fAu_x``, ``gAphiy_y`` and so on. This
is the layout of the kernels of ``plate_clpt_gl`` and ``cylshell_clpt_tl``,
repeated here with the five degrees of freedom and the ``NE x NE``
constitutive matrix of the TSDT. ``calc_kTv`` evaluates ``[B]^T [F] [B] v
+ [H(sigma, d)] v`` at each point. The follower-pressure kernels use the exact area vector of
the plate (``follower_pressure.MODELS``, geometry ``plate``), written by
``follower_pressure/write_pyx.py``.

"""
import importlib.util
import os
import re
import sys
import textwrap
import types

import sympy
from sympy import Add, Float, Matrix, Mul, Rational, Symbol, expand
from sympy.printing.str import StrPrinter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import plate_tsdt_gl as th
from panels.dev.matrixtools import pow2mult


def _load(name, *path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, *path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#NOTE the follower-pressure kernels, appended to the numerical module
follower_block = _load('follower_write_pyx', '..', 'follower_pressure',
                       'write_pyx.py').pyx_block

# Settings of the model
# ---------------------
MODEL = 'plate_tsdt_gl'
#: the model of the linear kernels
LINEAR = 'plate_tsdt_donnell'
#: whether the kernels read the radius
CYL = False
#: rows of the tables of the approximation functions, along x and y
F_ROWS = ['u', 'u_x', 'v', 'v_x', 'w', 'w_x', 'phix', 'phix_x', 'phiy',
          'phiy_x', 'w_xx']
G_ROWS = ['u', 'u_y', 'v', 'v_y', 'w', 'w_y', 'phix', 'phix_y', 'phiy',
          'phiy_y', 'w_yy']
DESC = ('Flat plates using the third-order shear deformation theory '
        '(TSDT) with Green-Lagrange membrane strains')
#: the generalized derivatives in words, for the docstrings
D_WORDS = ('the gradients of `u`, `v`, `w`, the rotations and their '
           'gradients and the second derivatives of `w`')
#: the gradients of the geometric stiffness, for the docstrings
GRAD_TEXT = (r'\{u_{,x}, u_{,y}, v_{,x}, v_{,y},' + '\n    '
             + r'w_{,x}, w_{,y}\}^\top` the gradients of the displacements')

NOTES = r'''Notes
-----
The membrane strains are the Green-Lagrange strains of the mid-surface,
those of :mod:`panels.models.plate_clpt_gl`, which vanish for any rigid
rotation, and the bending and transverse shear strains are the linear ones
of the third-order shear deformation theory [2]_ of
:mod:`panels.models.plate_tsdt_donnell`:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} (u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2) \qquad
    \varepsilon_{yy} = v_{,y} + \frac{1}{2} (u_{,y}^2 + v_{,y}^2 +
    w_{,y}^2)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} + w_{,x}
    w_{,y}

    \kappa_{xx} = \phi_{x,x} \qquad \kappa_{yy} = \phi_{y,y} \qquad
    \kappa_{xy} = \phi_{x,y} + \phi_{y,x}

    \gamma_{yz} = \phi_y + w_{,y} \qquad \gamma_{xz} = \phi_x + w_{,x}

    \kappa^{(3)}_{xx} = -c_1 (\phi_{x,x} + w_{,xx}) \qquad
    \kappa^{(3)}_{yy} = -c_1 (\phi_{y,y} + w_{,yy}) \qquad
    \kappa^{(3)}_{xy} = -c_1 (\phi_{x,y} + \phi_{y,x} + 2 w_{,xy})

    \gamma^{(2)}_{yz} = -3 c_1 \gamma_{yz} \qquad
    \gamma^{(2)}_{xz} = -3 c_1 \gamma_{xz}

with the third-order terms of Reddy's third-order shear deformation theory
(TSDT), `c_1 = 4/(3 h^2)`, as in ``:mod:`panels.models.plate_tsdt_donnell```, the in-plane strains being
`\varepsilon^{(0)} + z \kappa + [(z - d)^3 + d^3] \kappa^{(3)}` and the
transverse shear strains `\gamma + (z - d)^2 \gamma^{(2)}`, see
``theory/shells/fsdt_tsdt``.

Besides the von Kármán terms of :mod:`panels.models.plate_tsdt_donnell`,
the membrane strains contain the quadratic terms of the in-plane
gradients. For a finite rigid rotation the membrane strains vanish and,
with the constant rotations `\phi_x = -w_{,x}`, `\phi_y = -w_{,y}`, so do
the bending, third-order and transverse shear strains, see :mod:`panels.models`.
'''

REFS = r'''
References
----------
.. [1] Reddy, J. N., "Mechanics of Laminated Composite Plates and Shells:
   Theory and Analysis", 2nd ed., CRC Press, 2003.
   https://doi.org/10.1201/b12409
.. [2] Reddy, J. N., "A simple higher-order theory for laminated
   composite plates", Journal of Applied Mechanics, 51(4), 745-752, 1984.
   https://doi.org/10.1115/1.3167719
'''

# Layout of the kernels
# ---------------------
DOF = th.DOF
NE = th.NE
ND = len(th.D)
FIELDS = th.FIELDS
FIELD_RE = '(u|v|w|phix|phiy)'
#: name of the radius in the kernels
RADIUS = Symbol('radius')


class Printer(StrPrinter):
    def _print_Float(self, e):
        return repr(float(e))


def pstr(e):
    reps = {q: Float(q) for q in e.atoms(Rational) if not q.is_Integer}
    return pow2mult(Printer().doprint(e.xreplace(reps)))


BASIS = re.compile(r'^([fg])([AB])' + FIELD_RE + r'((?:_(?:xi|eta)+)?)$')


def basis_info(name):
    mo = BASIS.match(name)
    if mo is None:
        return None
    fg, side, field, der = mo.groups()
    order = der.count('xi') if fg == 'f' else der.count('eta')
    return fg, side, field, order


def physical(e):
    r"""Expression of the basis functions in the physical derivatives,
    ``(2/a) fAu_xi -> fAu_x``, ``(2/b) gAphiy_eta -> gAphiy_y``"""
    rename = {}
    for s in e.free_symbols:
        info = basis_info(s.name)
        if info is None:
            continue
        fg, side, field, order = info
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

FINPUT_TEMPLATE = """    # F as 4-D matrix, must be [nx, ny, NE, NE], when there is one F[NE, NE]
    # for each of the nx * ny integration points
    cdef double F[NE * NE]
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
    #NOTE the traction-free faces are assumed at z = +-h/2 about the
    #     mid-surface
    h = shell.lam.h
    c1 = 4./(3.*h*h)
""" + ('    radius = shell.r\n' if CYL else '') + BC_BLOCK + """

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
    cdef double h, c1
""" + ('    cdef double radius\n' if CYL else '')

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
                    for i in range(NE):
                        for j in range(NE):
                            F[i*NE + j] = Fnxny[ptx, pty, i, j]

                # approximation functions and their derivatives in x and y
"""


def _build():
    out = th.build()
    if CYL:
        out = {k: v.xreplace({th.r: RADIUS}) for k, v in out.items()}
    return out


class Spec(object):
    r"""Derivation of the model at the integration point, with the basis
    functions in the physical derivatives"""
    def __init__(self):
        self.M = _build()
        self.PHI_A = self.M['PhiA'].applyfunc(physical)
        self.PHI_B = self.M['PhiB'].applyfunc(physical)
        #: the generalized derivatives of each degree of freedom
        self.DSET = [[p for p in range(ND) if self.PHI_A[p, d] != 0]
                     for d in range(DOF)]
        self.DOF_OF = {p: d for d in range(DOF) for p in self.DSET[d]}


SP = Spec()
D_NAMES = th.D_NAMES


def table_lines():
    r"""Tabulation of the rows of the tables at the integration point"""
    L = []
    for fg, rows, idx, coord, ab in (('f', F_ROWS, 'i', 'xi', 'a'),
                                     ('g', G_ROWS, 'j', 'eta', 'b')):
        xy = 'x' if fg == 'f' else 'y'
        L.append('                for {0} in range({1}):'.format(idx, 'm' if fg == 'f' else 'n'))
        for k, row in enumerate(rows):
            field, _, der = row.partition('_')
            order = len(der)
            func = ('f', 'fp', 'fpp')[order]
            L.append('                    {0}tab[{1}, {2}] = {3}{4}({2}, {5}, {6}1{7}, '
                     '{6}1{7}r, {6}2{7}, {6}2{7}r)'.format(
                         fg, k, idx, '(2/{0})*'.format(ab)*order, func, coord, xy, field))
    return '\n'.join(L) + '\n'


FINPUT = FINPUT_TEMPLATE.replace('(NF, m)', '(%d, m)' % len(F_ROWS)).replace(
    '(NG, n)', '(%d, n)' % len(G_ROWS)).replace('F[NE * NE]', 'F[%d * %d]' % (NE, NE))
POINT = POINT_HEAD + table_lines()
COMMON = COMMON_CDEF


def basis_read(side, fg, indent):
    ind = ' '*indent
    rows = F_ROWS if fg == 'f' else G_ROWS
    idx = {('A', 'f'): 'i', ('A', 'g'): 'j', ('B', 'f'): 'k', ('B', 'g'): 'l'}[(side, fg)]
    tab = 'ftab' if fg == 'f' else 'gtab'
    return [ind + '{0}{1}{2} = {3}[{4}, {5}]'.format(fg, side, nm, tab, k, idx)
            for k, nm in enumerate(rows)]


def basis_cdef(sides):
    names = []
    for side in sides:
        names += ['f{0}{1}'.format(side, nm) for nm in F_ROWS]
        names += ['g{0}{1}'.format(side, nm) for nm in G_ROWS]
    return '\n'.join('    cdef double ' + ', '.join(names[i:i + 7])
                     for i in range(0, len(names), 7))


def phi_lines(side, indent):
    r"""``pA[p] = Phi_A[p, d]``, the derivative ``p`` of the term"""
    P = SP.PHI_A if side == 'A' else SP.PHI_B
    return ['{0}p{1}[{2}] = {3}'.format(' '*indent, side, p, pstr(P[p, SP.DOF_OF[p]]))
            for p in range(ND)]


def derivative_lines(indent, names, vec='cs', prefix=''):
    r"""``d[p] += Phi_A[p, d] cs[col+d]``, the derivatives at the point, or
    those of the vector ``vec``, named with ``prefix``"""
    return ['{0}{1}{2} += {3}*{4}[col+{5}]'.format(' '*indent, prefix, D_NAMES[p],
            pstr(SP.PHI_A[p, SP.DOF_OF[p]]), vec, SP.DOF_OF[p]) for p in names]


def assembly(mname, pattern, blocks, symmetric=True):
    r"""Loops over the pairs of terms, ``mname`` the name of the matrix,
    ``pattern[p][q]`` the non-zero entries of ``K`` and ``blocks`` the pairs
    of degrees of freedom ``(d, e)`` of the matrix"""
    DSET = SP.DSET
    T = {}
    for d in range(DOF):
        for q in range(ND):
            ps = [p for p in DSET[d] if pattern[p][q]]
            if ps:
                T[(d, q)] = ' + '.join('pA[{0}]*K[{1}]'.format(p, p*ND + q) for p in ps)
    L = ['                c = -1',
         '                for j in range(n):']
    L += basis_read('A', 'g', 20)
    L += ['                    for i in range(m):']
    L += basis_read('A', 'f', 24)
    L += ['                        row = row0 + DOF*(j*m + i)']
    L += phi_lines('A', 24)
    L += ['                        # TA[d, q] = sum_p Phi_A[p, d] K[p, q]']
    used_d = sorted({d for d, e in blocks})
    for (d, q), expr in sorted(T.items()):
        if d in used_d:
            L.append('                        TA[{0}] = {1}'.format(d*ND + q, expr))
    L += ['                        for l in range(n):']
    L += basis_read('B', 'g', 28)
    L += ['                            for k in range(m):',
          '                                col = col0 + DOF*(l*m + k)']
    if symmetric:
        L += ['                                #NOTE symmetry assumption True if no follower forces are used',
              '                                if row > col:',
              '                                    continue']
    L += basis_read('B', 'f', 32)
    L += phi_lines('B', 32)
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


def gradient_loop(indent, names, extra=()):
    r"""Loop over the terms summing the derivatives ``names`` at the point;
    ``extra`` holds ``(vec, prefix)`` of further vectors to differentiate"""
    ind = ' '*indent
    L = [ind + 'for j in range(n):']
    L += basis_read('A', 'g', indent + 4)
    L += [ind + '    for i in range(m):']
    L += basis_read('A', 'f', indent + 8)
    L += [ind + '        col = col0 + DOF*(j*m + i)']
    L += derivative_lines(indent + 8, names)
    for vec, prefix in extra:
        L += derivative_lines(indent + 8, names, vec, prefix)
    return L


def cdefs(names, per=8):
    return '\n'.join('    cdef double ' + ', '.join(names[i:i + per])
                     for i in range(0, len(names), per))


def zero_lines(names, indent=16):
    return '\n'.join('{0}{1} = 0.'.format(' '*indent, nm) for nm in names)


def stresses(e, sg, indent=16):
    sp = ' '*indent
    return '\n'.join([sp + 'for r in range(NE):',
                      sp + '    {0}[r] = 0.'.format(sg),
                      sp + '    for s in range(NE):',
                      sp + '        {0}[r] += F[r*NE + s]*{1}[s]'.format(sg, e)])


def strain_lines(name, e):
    return '\n'.join('                {0}[{1}] = {2}'.format(name, k, pstr(e[k]))
                     for k in range(NE))


SG = [Symbol('sg[%d]' % k) for k in range(NE)]
SGL = [Symbol('sgL[%d]' % k) for k in range(NE)]

FINPUT_DOC = '''    Finput : array-like
        Constitutive matrix of the generalized strains of the model, shape
        ``({0}, {0})`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, {0}, {0})`` with one matrix per integration point.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.'''.format(NE)

NOTES_D = '''    With the generalized derivatives at the integration point `\\boldsymbol{{d}}
    = \\boldsymbol{{\\Phi}} \\boldsymbol{{c}}` ({0}), the strains
    `\\boldsymbol{{\\varepsilon}}(\\boldsymbol{{d}})`, `\\boldsymbol{{B}} =
    \\partial \\boldsymbol{{\\varepsilon}} / \\partial \\boldsymbol{{d}}`, the
    stresses `\\boldsymbol{{\\sigma}} = \\boldsymbol{{F}}
    \\boldsymbol{{\\varepsilon}}` and `\\boldsymbol{{H}}(\\boldsymbol{{\\sigma}}) =
    \\sum_r \\sigma_r \\, \\partial^2 \\varepsilon_r / \\partial
    \\boldsymbol{{d}}^2`, which involves only the membrane stresses and is
    independent of `\\boldsymbol{{d}}`, the membrane strains being quadratic:'''.format(D_WORDS)

LIMITS_DOC = '''    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \\le x \\le x_2`, `y_1 \\le y \\le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.'''


def numerical_module():
    m = SP.M
    B, eps, eps_L, H = m['B'], m['eps'], m['eps_L'], m['H']
    alld = list(range(ND))
    full = [[True]*ND for _ in range(ND)]
    blocks_all = [(d, e) for d in range(DOF) for e in range(DOF)]

    out = [HEADER + 'r"""\n' + textwrap.fill(DESC, 76) + '''

Numerically integrated kernels of the model ``'{0}'``: the
constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the internal force vector :func:`calc_fint`, the product
of the tangent stiffness matrix by a vector :func:`calc_kTv`, and the force
vector and load stiffness matrix of a follower pressure,
:func:`calc_fext_follower` and :func:`fkCfollower_num`. The mass and
aerodynamic matrices, ``fkM_num``, ``fkAx_num`` and ``fkAy_num``, are those
of :mod:`panels.models.{1}_num`. The constitutive matrix ``Finput`` is the
``{2} x {2}`` matrix ``ABD`` of :class:`.Shell` for this model.

The degrees of freedom of each term of the approximation are
``u, v, w, phix, phiy``; the degree of freedom ``k`` of the term ``(i, j)``,
``0 <= i < m``, ``0 <= j < n``, is at the row ``row0 + 5*(j*m + i) + k``
of the matrices.

'''.format(MODEL, LINEAR, NE) + NOTES + '''
The numerical integration over the domain of
:meth:`.Shell.integration_limits` admits a constitutive matrix and moments
of the density varying over the domain, a partial domain and the
non-linear terms of the Ritz constants.
''' + REFS + '''
"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
#NOTE the linear kernels, independent of the non-linear strains, are those
#     of {0}
from .{0}_num import fkM_num, fkAx_num, fkAy_num


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = {1}
cdef int NE = {2}
'''.format(LINEAR, DOF, NE)]

    # ------------------------------------------------------------- fkC_num
    Hd = H.xreplace({s: SG[k] - SGL[k] for k, s in enumerate(th.STRESSES)}).applyfunc(expand)
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
{FDOC}
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
{NOTES_D}

    .. math::

        \\boldsymbol{{K}}_C = \\int \\boldsymbol{{\\Phi}}^\\top \\left[\\boldsymbol{{B}}^\\top
        \\boldsymbol{{F}} \\boldsymbol{{B}} + \\boldsymbol{{H}}(\\boldsymbol{{\\sigma}} -
        \\boldsymbol{{\\sigma}}_L)\\right] \\boldsymbol{{\\Phi}} \\, dA

    with `\\boldsymbol{{\\sigma}}_L` the stresses of the linear strains, whose
    term `\\boldsymbol{{H}}(\\boldsymbol{{\\sigma}}_L)` is assembled by
    :func:`fkG_num`, such that ``fkC_num(NLgeom=1) + fkG_num`` is the exact
    Jacobian of :func:`calc_fint`.

{LDOC}

    """
{COMMON}
{dcdef}
    cdef double e[{NE}]
    cdef double eL[{NE}]
    cdef double sg[{NE}]
    cdef double sgL[{NE}]
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

    fdim = {DOF2}*m*m*n*n
    kCr = np.zeros((fdim,), dtype=INT)
    kCc = np.zeros((fdim,), dtype=INT)
    kCv = np.zeros((fdim,), dtype=DOUBLE)

{POINT}
                # generalized derivatives at the point, zero when NLgeom == 0
{dzero}
                if NLgeom == 1:
{dloop}

                # [B] = d eps/d d at the point
                for p in range({NB}):
                    Bm[p] = 0.
{Blines}

                # K = [B]^T [F] [B]
                for r in range(NE):
                    for q in range({ND}):
                        FB[r*{ND} + q] = 0.
                        for s in range(NE):
                            FB[r*{ND} + q] += F[r*NE + s]*Bm[s*{ND} + q]
                for p in range({ND}):
                    for q in range({ND}):
                        K[p*{ND} + q] = 0.
                        for r in range(NE):
                            K[p*{ND} + q] += Bm[r*{ND} + p]*FB[r*{ND} + q]

                # stresses of the strains and of the linear strains
{elines}
{eLlines}
{sg}
{sgL}
                # K += [H(sigma - sigma_L)], [H(sigma_L)] being assembled by
                # fkG_num, which is homogeneous of degree one in cs as linear
                # buckling requires
{Hlines}

{loops}

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC
'''.format(FDOC=FINPUT_DOC, NOTES_D=NOTES_D, LDOC=LIMITS_DOC, COMMON=COMMON,
           dcdef=cdefs(list(D_NAMES)), basis=basis_cdef('AB'), NE=NE,
           NB=NE*ND, NK=ND*ND, NT=DOF*ND, ND=ND, DOF2=DOF*DOF, FINPUT=FINPUT,
           POINT=POINT, dzero=zero_lines(D_NAMES),
           dloop='\n'.join(gradient_loop(20, alld)),
           Blines='\n'.join(matrix_lines('Bm', B, 16)),
           elines=strain_lines('e', eps), eLlines=strain_lines('eL', eps_L),
           sg=stresses('e', 'sg'), sgL=stresses('eL', 'sgL'),
           Hlines='\n'.join(matrix_lines('K', Hd, 16, op='+=')),
           loops=assembly('kC', full, blocks_all)))

    # ------------------------------------------------------------- fkG_num
    Hs = H.xreplace(dict(zip(th.STRESSES, SG))).applyfunc(expand)
    Hpat = [[H[p, q] != 0 for q in range(ND)] for p in range(ND)]
    blocks = [(d, e) for d in range(DOF) for e in range(DOF)
              if any(Hpat[p][q] for p in SP.DSET[d] for q in SP.DSET[e])]
    lin = [p for p in range(ND) if th.D[p] in set().union(*[x.free_symbols for x in eps_L])]
    out.append('''

def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    r"""Geometric stiffness matrix of the linear stress state

    Integrates numerically the geometric stiffness matrix of the membrane
    stresses of the linear strains at ``cs``, superposed to the constant
    membrane state ``Nxx0``, ``Nyy0``, ``Nxy0``, with one block per local
    gradient of ``u``, ``v`` and ``w``. The matrix is homogeneous of degree
    one in ``cs``, as the linear buckling eigenvalue problem requires.

    Parameters
    ----------
    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements (the rotations in radians).
{FDOC}
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
{NOTES_D}

    .. math::

        \\boldsymbol{{K}}_G = \\int \\boldsymbol{{\\Phi}}^\\top
        \\boldsymbol{{H}}(\\boldsymbol{{\\sigma}}) \\boldsymbol{{\\Phi}} \\, dA

    with `\\boldsymbol{{\\sigma}} = \\{{N_{{xx0}}, N_{{yy0}}, N_{{xy0}}, 0, \\ldots\\}}^\\top +
    \\boldsymbol{{\\sigma}}_L`, `\\boldsymbol{{\\sigma}}_L` the stresses of the
    linear strains at ``cs``. The rest of the tangent stiffness matrix is
    assembled by :func:`fkC_num`.

{LDOC}

    """
{COMMON}
{dcdef}
    cdef double e[{NE}]
    cdef double sg[{NE}]
{basis}
    cdef double K[{NK}]
    cdef double TA[{NT}]
    cdef double pA[{ND}]
    cdef double pB[{ND}]

    cdef long [::1] kGr, kGc
    cdef double [::1] kGv

{FINPUT}

    fdim = {NBLK}*m*m*n*n
    kGr = np.zeros((fdim,), dtype=INT)
    kGc = np.zeros((fdim,), dtype=INT)
    kGv = np.zeros((fdim,), dtype=DOUBLE)

{POINT}
                # derivatives of the linear strains at the point
{dzero}
{dloop}

                # linear strains and stresses
{elines}
{sg}
                sg[0] += Nxx0
                sg[1] += Nyy0
                sg[2] += Nxy0

                # K = [H(sigma)]
                for p in range({NK}):
                    K[p] = 0.
{Hlines}

{loops}

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG
'''.format(FDOC=FINPUT_DOC, NOTES_D=NOTES_D, LDOC=LIMITS_DOC, COMMON=COMMON,
           dcdef=cdefs([D_NAMES[p] for p in lin]), basis=basis_cdef('AB'),
           NE=NE, NK=ND*ND, NT=DOF*ND, ND=ND, FINPUT=FINPUT, POINT=POINT,
           NBLK=len(blocks), dzero=zero_lines([D_NAMES[p] for p in lin]),
           dloop='\n'.join(gradient_loop(16, lin)),
           elines=strain_lines('e', eps_L), sg=stresses('e', 'sg'),
           Hlines='\n'.join(matrix_lines('K', Hs, 16)),
           loops=assembly('kG', Hpat, blocks)))

    # ------------------------------------------------------------- calc_fint
    S = (B.T*Matrix(SG)).applyfunc(expand)
    fint_lines = []
    for d in range(DOF):
        terms = ['pA[{0}]*S[{0}]'.format(p) for p in SP.DSET[d]]
        fint_lines.append('                        fint[col+{0}] += weight*({1})'.format(d, ' + '.join(terms)))
    out.append('''

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
{FDOC}
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

{LDOC}

    """
{COMMON}
{dcdef}
    cdef double e[{NE}]
    cdef double sg[{NE}]
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
{sg}
{Slines}

                for j in range(n):
{gA}
                    for i in range(m):
{fA}
                        col = col0 + DOF*(j*m + i)
{pA}
{fint}

    return np.asarray(fint)
'''.format(FDOC=FINPUT_DOC, LDOC=LIMITS_DOC, COMMON=COMMON,
           dcdef=cdefs(list(D_NAMES)), NE=NE, ND=ND, basis=basis_cdef('A'),
           FINPUT=FINPUT, POINT=POINT, dzero=zero_lines(D_NAMES),
           dloop='\n'.join(gradient_loop(16, alld)),
           elines=strain_lines('e', eps), sg=stresses('e', 'sg'),
           Slines='\n'.join('                S[{0}] = {1}'.format(p, pstr(S[p])) for p in range(ND)),
           gA='\n'.join(basis_read('A', 'g', 20)), fA='\n'.join(basis_read('A', 'f', 24)),
           pA='\n'.join(phi_lines('A', 24)), fint='\n'.join(fint_lines)))

    # ------------------------------------------------------------- calc_kTv
    dv = Matrix([Symbol('d' + nm) for nm in D_NAMES])
    DSG = Matrix([Symbol('dsg[%d]' % k) for k in range(NE)])
    de = (B*dv).applyfunc(expand)
    S = (B.T*DSG + H.xreplace(dict(zip(th.STRESSES, SG)))*dv).applyfunc(expand)
    kTv_lines = []
    for d in range(DOF):
        terms = ['pA[{0}]*S[{0}]'.format(p) for p in SP.DSET[d]]
        kTv_lines.append('                        kTv[col+{0}] += weight*({1})'.format(d, ' + '.join(terms)))
    out.append('''

def calc_kTv(double [::1] cs, double [::1] vs, object Finput, object shell,
        int size, int col0, int nx, int ny, double Nxx_cte=0.,
        double Nyy_cte=0., double Nxy_cte=0.):
    r"""Product of the tangent stiffness matrix by a vector, without assembly

    Returns ``KT(cs) vs``, with ``KT`` the tangent stiffness matrix
    ``fkC_num(cs, ..., NLgeom=1) + fkG_num(cs, ...)`` at ``cs``, the exact
    Jacobian of :func:`calc_fint`, plus the geometric stiffness matrix of
    the constant membrane stress state ``Nxx_cte``, ``Nyy_cte`` and
    ``Nxy_cte``. The cost is about that of :func:`calc_fint`.

    Parameters
    ----------
    cs, vs : array-like
        Ritz constants of the configuration and the vector, 1-D of size
        ``size``.
{FDOC}
    size, col0 : int
        Size of the vectors and position where the terms of this shell
        start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.
    Nxx_cte, Nyy_cte, Nxy_cte : float, optional
        Constant membrane stress resultants, force per unit length, default
        ``0``.

    Returns
    -------
    kTv : numpy.ndarray
        The product, of size ``size``.

    Raises
    ------
    ValueError
        If ``cs`` or ``vs`` do not have the size ``size``, ``Finput`` has an
        invalid shape or ``shell`` is not a :class:`.Shell`.

    Notes
    -----
{NOTES_D}

    .. math::

        \\boldsymbol{{K}}_T \\boldsymbol{{v}} = \\int \\boldsymbol{{\\Phi}}^\\top
        \\left[\\boldsymbol{{B}}^\\top \\boldsymbol{{F}} \\boldsymbol{{B}} +
        \\boldsymbol{{H}}(\\boldsymbol{{\\sigma}} + \\bar{{\\boldsymbol{{\\sigma}}}})
        \\right] \\boldsymbol{{\\Phi}} \\boldsymbol{{v}} \\, dA

    with `\\bar{{\\boldsymbol{{\\sigma}}}}` the constant membrane stresses,
    evaluated without forming `\\boldsymbol{{K}}_T`.

{LDOC}

    """
{COMMON}
{dcdef}
{ddcdef}
    cdef double e[{NE}]
    cdef double de[{NE}]
    cdef double sg[{NE}]
    cdef double dsg[{NE}]
    cdef double S[{ND}]
    cdef double pA[{ND}]
{basis}

    cdef double [::1] kTv

    if cs.shape[0] != size or vs.shape[0] != size:
        raise ValueError('cs and vs must have the size %d' % size)

{FINPUT}

    kTv = np.zeros(size, dtype=DOUBLE)

{POINT}
                # generalized derivatives of cs and vs at the point
{dzero}
{ddzero}
{dloop}

                # strains, strain increments, stresses
{elines}
{delines}
{sg}
{dsg}
                sg[0] += Nxx_cte
                sg[1] += Nyy_cte
                sg[2] += Nxy_cte

                # S = [B]^T dsigma + [H(sigma)] dv
{Slines}

                for j in range(n):
{gA}
                    for i in range(m):
{fA}
                        col = col0 + DOF*(j*m + i)
{pA}
{kTv}

    return np.asarray(kTv)
'''.format(FDOC=FINPUT_DOC, NOTES_D=NOTES_D, LDOC=LIMITS_DOC, COMMON=COMMON,
           dcdef=cdefs(list(D_NAMES)), ddcdef=cdefs(['d' + nm for nm in D_NAMES]),
           NE=NE, ND=ND, basis=basis_cdef('A'), FINPUT=FINPUT, POINT=POINT,
           dzero=zero_lines(D_NAMES), ddzero=zero_lines(['d' + nm for nm in D_NAMES]),
           dloop='\n'.join(gradient_loop(16, alld, extra=(('vs', 'd'),))),
           elines=strain_lines('e', eps), delines=strain_lines('de', de),
           sg=stresses('e', 'sg'), dsg=stresses('de', 'dsg'),
           Slines='\n'.join('                S[{0}] = {1}'.format(p, pstr(S[p])) for p in range(ND)),
           gA='\n'.join(basis_read('A', 'g', 20)), fA='\n'.join(basis_read('A', 'f', 24)),
           pA='\n'.join(phi_lines('A', 24)), kTv='\n'.join(kTv_lines)))
    #NOTE c, p, q, r, s, k, l and parts of the tables are not used in every
    #     kernel, which is harmless
    return ''.join(out)


# Analytical geometric stiffness
# ------------------------------
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


def analytical_module():
    r"""``fkG0``, and the other analytical kernels imported from the linear
    model"""
    mm = th.build()
    H = mm['H']
    a, b = th.a, th.b
    kG = (a*b/4)*mm['PhiA'].T*H*mm['PhiB']
    if CYL:
        kG = kG.xreplace({th.r: RADIUS})
    ent = [(i, j, to_integrals(kG[i, j])) for i in range(DOF) for j in range(DOF)
           if kG[i, j] != 0]
    fnames, gnames, kinds = set(), set(), set()
    for _, _, v in ent:
        for s in v.free_symbols:
            if s.name.startswith('fA'):
                fnames.add(s.name)
            elif s.name.startswith('gA'):
                gnames.add(s.name)
    fnames = sorted(fnames)
    gnames = sorted(gnames)
    for s in fnames + gnames:
        kinds.add(integral_call(s).split('(')[0])

    def cdefs4(names):
        return '\n'.join('    cdef double ' + ', '.join(names[i:i + 4])
                         for i in range(0, len(names), 4))

    L = [HEADER + 'r"""\n' + textwrap.fill(DESC, 76) + '''

Analytical kernels of the model ``'{0}'``, integrated over the whole
domain. The linear constitutive stiffness ``fk0``, the mass ``fkM`` and the
aerodynamic ``fkAx``, ``fkAy`` and ``fcA`` matrices are those of
:mod:`panels.models.{1}`. The geometric stiffness matrix :func:`fkG0` of
the constant membrane stress state ``Nxx, Nyy, Nxy`` has one block per
local gradient of ``u``, ``v`` and ``w``.

'''.format(MODEL, LINEAR) + NOTES + '''
The integrals over the whole domain are evaluated in closed form with the
integrals of Bardell's functions and their derivatives along `\\xi = 2x/a -
1` and `\\eta = 2y/b - 1`, which requires the constitutive matrix and the
moments of the density to be constant over the domain.
''' + REFS + '''
"""
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE
#NOTE the linear kernels, independent of the non-linear strains, are those
#     of {0}
from .{0} import fk0, fkM, fkAx, fkAy, fcA


cdef int DOF = {1}


cdef extern from 'bardell.hpp':
'''.format(LINEAR, DOF)]
    for kind in sorted(kinds):
        L.append('''    double {0}(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil'''.format(kind))
    L.append('''

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
    N_{yy} \\end{bmatrix}` and `\\boldsymbol{G} \\boldsymbol{c} = ''' + GRAD_TEXT + ''', with one
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
    cdef double a, b''' + (', radius' if CYL else '') + '''
    cdef int m, n
''' + BC_CDEF + '''

    cdef int i, j, k, l, c, row, col

    cdef long [:] kG0r, kG0c
    cdef double [:] kG0v

''' + cdefs4(fnames) + '\n' + cdefs4(gnames) + '''

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
''' + ('    radius = shell.r\n' if CYL else '') + '''    m = shell.m
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
'''.format(len(ent)))
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


def generate(outdir=None):
    outdir = outdir or os.path.join(REPO, 'panels', 'models')
    path = os.path.join(outdir, MODEL + '.pyx')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(analytical_module())
    print(MODEL, 'analytical written')
    path = os.path.join(outdir, MODEL + '_num.pyx')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(numerical_module() + '\n\n' + follower_block(MODEL))
    print(MODEL, 'numerical written')


if __name__ == '__main__':
    generate(*sys.argv[1:2])
