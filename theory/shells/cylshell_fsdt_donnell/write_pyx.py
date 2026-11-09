"""Generate the kernels of the model ``cylshell_fsdt_donnell``

Writes ``panels/models/cylshell_fsdt_donnell.pyx`` (matrices integrated analytically)
and ``panels/models/cylshell_fsdt_donnell_num.pyx`` (matrices and vectors integrated
numerically, product of the tangent stiffness matrix by a vector and
follower-pressure kernels) from the derivation of ``cylshell_fsdt_donnell.py``::

    python write_pyx.py [outdir]

Kernel layout
-------------
The non-linear kernels, ``fkC_num``, ``fkG_num``, ``calc_fint`` and
``calc_kTv``, evaluate at each integration point the generalized
derivatives ``d = [Phi] c`` (``cylshell_fsdt_donnell.D``), ``[B] = d eps/d d`` and ``K
= [B]^T [F] [B] + [H]`` once; the entry of the pair of terms ``A = (i,
j)``, ``B = (k, l)`` and the degrees of freedom ``d`` and ``e`` is
``sum_pq Phi_A[p, d] K[p, q] Phi_B[q, e]``, computed as ``TA[d, q] = sum_p
Phi_A[p, d] K[p, q]`` in the loop over the terms ``A`` and ``sum_q TA[d,
q] Phi_B[q, e]`` in the loop over the terms ``B``. The approximation
functions are tabulated once per integration point, with the derivatives
in the physical coordinates, ``(2/a) f'(xi)``, named ``fAu_x``,
``gAphiy_y`` and so on. This is the layout of the kernels of the models
``*_gl`` and ``*_tl``. ``calc_kTv`` evaluates ``[B]^T [F] [B] v + [H(sigma)]
v`` at each point.
The radius is named ``radius`` in these kernels, ``r`` being one of their
loop indices.

The linear kernels, ``fk0``, ``fkG0``, ``fkM``, ``fkAx``, ``fkAy`` and
``fcA`` integrated analytically, and ``fkM_num``, ``fkAx_num`` and
``fkAy_num`` integrated numerically, are written from the expanded
integrands of each entry of :func:`cylshell_fsdt_donnell.build_linear`, the analytical
ones as sums of products of the integrals of Bardell's functions. The
follower-pressure kernels are written by
``../follower_pressure/write_pyx.py``.

"""
import importlib.util
import os
import re
import sys
import textwrap

import numpy as np
import sympy
from sympy import (Add, Float, Matrix, Mul, Rational, Symbol, collect,
                   expand, factor)
from sympy.printing.str import StrPrinter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import cylshell_fsdt_donnell as th
from panels.dev.matrixtools import pow2mult
from panels.dev.tabulate import tabulate_basis


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
MODEL = 'cylshell_fsdt_donnell'
#: whether the kernels read the radius
CYL = True
#: rows of the tables of the approximation functions, along x and y
F_ROWS = ['u', 'u_x', 'v', 'v_x', 'w', 'w_x', 'phix', 'phix_x', 'phiy',
          'phiy_x']
G_ROWS = ['u', 'u_y', 'v', 'v_y', 'w', 'w_y', 'phix', 'phix_y', 'phiy',
          'phiy_y']
DESC = ("Cylindrical shells using the first-order shear deformation theory (FSDT) with the Donnell kinematics")
#: the generalized derivatives in words, for the docstrings
D_WORDS = ('the gradients of `u`, `v`, `w`, the value of `w`, the rotations and their gradients')
#: the generalized strains and stresses, in the order of ``Finput``
NE_NAMES = 'exx, eyy, gxy, kxx, kyy, kxy, gyz, gxz'
STRESS_NAMES = 'Nxx, Nyy, Nxy, Mxx, Myy, Mxy, Qy, Qx'
#: the moments of the density through the thickness read by the mass
#: matrices, as named in the derivation and in :class:`composites.Laminate`
MOMENT_NAMES = ['I0', 'I1', 'I2']
LAM_MOMENTS = dict(I0='intrho', I1='intrhoz', I2='intrhoz2')

NOTES = r'''Notes
-----
The generalized strains are, with the rotations `\beta_x = -w_{,x}`
and `\beta_y = -w_{,y}` of the non-linear terms of Donnell:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} \beta_x^2 \qquad
    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \beta_y^2 \qquad
    \gamma_{xy} = u_{,y} + v_{,x} + \beta_x \beta_y

    \kappa_{xx} = \phi_{x,x} \qquad \kappa_{yy} = \phi_{y,y} \qquad
    \kappa_{xy} = \phi_{x,y} + \phi_{y,x}

    \gamma_{yz} = \phi_y + w_{,y} \qquad \gamma_{xz} = \phi_x + w_{,x}

With `y` the arc length along the circumference and `w` positive outwards.
The bending and transverse shear strains are those of the first-order shear
deformation theory (FSDT) [1]_, the in-plane strains being
`\varepsilon^{(0)} + z \kappa`.
'''

REFS = r'''
References
----------
.. [1] Reddy, J. N., and Liu, C. F., "A higher-order shear deformation
   theory of laminated elastic shells", International Journal of
   Engineering Science, 23(3), 319-330, 1985.
   https://doi.org/10.1016/0020-7225(85)90051-5
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
of the tangent stiffness matrix by a vector :func:`calc_kTv`, the mass
:func:`fkM_num`, the aerodynamic stiffness :func:`fkAx_num` and
:func:`fkAy_num`, and the force vector and load stiffness matrix of a
follower pressure, :func:`calc_fext_follower` and
:func:`fkCfollower_num`. The constitutive matrix ``Finput`` is the
``{1} x {1}`` matrix ``ABD`` of :class:`.Shell` for this model, with the
rows and columns in the order of the generalized strains::

    {2}

which is the order of the generalized stresses::

    {3}

The degrees of freedom of each term of the approximation are
``u, v, w, phix, phiy``; the degree of freedom ``k`` of the term ``(i, j)``,
``0 <= i < m``, ``0 <= j < n``, is at the row ``row0 + 5*(j*m + i) + k``
of the matrices.

'''.format(MODEL, NE, NE_NAMES, STRESS_NAMES) + NOTES + '''
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


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = {0}
cdef int NE = {1}
'''.format(DOF, NE)]

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
    membrane state ``Nxx0``, ``Nyy0``, ``Nxy0``, with the rotations
    `\\beta_x` and `\\beta_y` of the non-linear terms of the membrane
    strains. The matrix is homogeneous of degree one in ``cs``, as the
    linear buckling eigenvalue problem requires.

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


# Linear kernels
# --------------
# written from the expanded integrands of each entry of th.build_linear()


def simp(e, vars, do_factor=True):
    e = expand(e)
    if do_factor:
        return collect(e, vars, func=factor)
    return collect(e, vars)


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


def cdefs5(names, indent=4, per_line=5):
    names = list(names)
    out = []
    for i in range(0, len(names), per_line):
        out.append(' '*indent + 'cdef double ' + ', '.join(names[i:i + per_line]))
    return '\n'.join(out)


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

POINT_LIN = """        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]
"""


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
         cdefs5(fnames), cdefs5(gnames), '',
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
            out.append(cdefs5(ns))
    return '\n'.join(o for o in out if o)



def with_radius(text, cyl):
    r"""Reads the radius ``r`` in every function of the cylinders"""
    if not cyl:
        return text
    n = text.count('    n = shell.n\n')
    text = text.replace('    cdef double a, b', '    cdef double r\n    cdef double a, b')
    text = text.replace('    n = shell.n\n', '    n = shell.n\n    r = shell.r\n')
    assert n > 0 and text.count('    cdef double r\n') == n
    return text


def analytical_module():
    r"""``fk0``, ``fkG0``, ``fkM``, ``fkAx``, ``fkAy`` and ``fcA``"""
    mats = th.build_linear()
    Fsym = mats['F']
    NE = Fsym.shape[0]
    cvars = tuple(sympy.Symbol(n) for n in const_names(Fsym))
    a, b = sympy.symbols('a, b')
    area = a*b/4
    desc = DESC
    moment_names = MOMENT_NAMES

    kC0 = mats['kC0'].applyfunc(lambda e: simp(area*e, cvars))
    kG = mats['kG'].applyfunc(lambda e: simp(area*e, (th.Nxx, th.Nyy, th.Nxy)))
    kM = mats['kM'].applyfunc(lambda e: expand(area*e))
    assert [s.name for s in th.MOMENTS
            if any(s in e.free_symbols for e in mats['kM'])] == moment_names
    kAx = mats['kAx'].applyfunc(lambda e: expand(area*e))
    kAy = mats['kAy'].applyfunc(lambda e: expand(area*e))
    cA = mats['cA'].applyfunc(lambda e: expand(area*e))

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
    k0_cdef = '\n'.join(['    cdef double a, b',
                         '    cdef double [:, ::1] F',
                         '    cdef int m, n',
                         cdefs5([v.name for v in cvars])])
    #NOTE the kernels read the property Shell.ABD, and fkM calls
    #     Shell._get_lam_mid, with C variables already set, which must not
    #     call matrix-matrix BLAS products, see the note in
    #     Shell._get_lam_ABD and tests/tests_shell/test_kernel_python_calls.py
    k0_pre = '\n'.join(['    F = shell.ABD', '',
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
        '    cdef double a, b, ' + ', '.join(moment_names) + '\n    cdef int m, n',
        '\n'.join('    {0} = shell.lam.{1}'.format(name, LAM_MOMENTS[name])
                  for name in moment_names),
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
    return with_radius('\n'.join(A), CYL)


def linear_numerical_kernels():
    r"""``fkM_num``, ``fkAx_num`` and ``fkAy_num``"""
    mats = th.build_linear()
    moment_names = MOMENT_NAMES
    kMn = [(i, j, pstr(v)) for i, j, v in entries(mats['kM'].applyfunc(
        lambda e: collect(expand(e), th.MOMENTS)))]
    kAxn = [(i, j, pstr(v)) for i, j, v in entries(mats['kAx'].applyfunc(expand))]
    kAyn = [(i, j, pstr(v)) for i, j, v in entries(mats['kAy'].applyfunc(expand))]
    NUM = []

    # fkM_num
    input_names = moment_names
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
'''.format(c1='', BC_CDEF=BC_CDEF,
           basis=basis_cdefs(set(names)), LIMITS=LIMITS, BC_BLOCK=BC_BLOCK,
           nnz=len(kMn), MAPPING=MAPPING, POINT=POINT_LIN, loops=loops,
           K=len(input_names), inames=', '.join(input_names),
           icdef=', '.join(input_names),
           dcdef='', dread='',
           iread='\n'.join('                    {0} = I_nxny[ptx, pty, {1}]'.format(name, i)
                           for i, name in enumerate(input_names)),
           c1pt=''))

    # fkAx_num and fkAy_num
    for fname, ents, beta_gamma in (
            ('fkAx_num', kAxn, 'beta = shell.beta\n    gamma = shell.gamma'),
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
           gcdef=', gamma' if fname == 'fkAx_num' else '',
           MAPPING=MAPPING, POINT=POINT_LIN, loops=loops))
    return tabulate_basis(with_radius(''.join(NUM), CYL))


def generate(outdir=None):
    outdir = outdir or os.path.join(REPO, 'panels', 'models')
    path = os.path.join(outdir, MODEL + '.pyx')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(analytical_module())
    print(MODEL, 'analytical written')
    path = os.path.join(outdir, MODEL + '_num.pyx')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(numerical_module() + linear_numerical_kernels() + '\n\n'
                + follower_block(MODEL))
    print(MODEL, 'numerical written')


if __name__ == '__main__':
    generate(*sys.argv[1:2])
