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

"""
import importlib.util
import os
import re
import sys
import textwrap

import numpy as np
import sympy
from sympy import Float, Rational, Symbol, expand, zeros
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
#NOTE the integrals of the approximation functions of the analytical kernels
_fsdt = _load('fsdt_tsdt_write_pyx', '..', 'fsdt_tsdt', 'write_pyx.py')
to_integrals = _fsdt.to_integrals
integral_call = _fsdt.integral_call
ordered_integrals = _fsdt.ordered_integrals

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

FINPUT = """    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
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

    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)"""

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

POINT = """    with nogil:
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
                for i in range(m):
                    ftab[0, i] = f(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[1, i] = (2/a)*fp(i, xi, x1u, x1ur, x2u, x2ur)
                    ftab[2, i] = f(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[3, i] = (2/a)*fp(i, xi, x1v, x1vr, x2v, x2vr)
                    ftab[4, i] = f(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[5, i] = (2/a)*fp(i, xi, x1w, x1wr, x2w, x2wr)
                    ftab[6, i] = (2/a)*(2/a)*fpp(i, xi, x1w, x1wr, x2w, x2wr)
                for j in range(n):
                    gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[1, j] = (2/b)*fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gtab[2, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[3, j] = (2/b)*fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gtab[4, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[5, j] = (2/b)*fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gtab[6, j] = (2/b)*(2/b)*fpp(j, eta, y1w, y1wr, y2w, y2wr)
"""


def basis_read(side, fg, indent):
    sp = ' '*indent
    rows = F_ROWS if fg == 'f' else G_ROWS
    idx = {('A', 'f'): 'i', ('A', 'g'): 'j', ('B', 'f'): 'k', ('B', 'g'): 'l'}[(side, fg)]
    tab = 'ftab' if fg == 'f' else 'gtab'
    return [sp + '{0}{1}{2} = {3}[{4}, {5}]'.format(fg, side, nm, tab, k, idx)
            for k, nm in enumerate(rows)]


def basis_cdef(sides='AB'):
    names = []
    for side in sides:
        names += ['f{0}{1}'.format(side, nm) for nm in F_ROWS]
        names += ['g{0}{1}'.format(side, nm) for nm in G_ROWS]
    return '\n'.join('    cdef double ' + ', '.join(names[i:i + 7])
                     for i in range(0, len(names), 7))


M = th.build()
PHI_A = M['PhiA'].applyfunc(physical)
PHI_B = M['PhiB'].applyfunc(physical)
#: the generalized derivatives of each degree of freedom
DSET = [[p for p in range(ND) if PHI_A[p, d] != 0] for d in range(DOF)]
DOF_OF = {p: d for d in range(DOF) for p in DSET[d]}
D_NAMES = th.D_NAMES


def phi_lines(side, indent):
    r"""``pA[p] = Phi_A[p, d]``, the derivative ``p`` of the term"""
    P = PHI_A if side == 'A' else PHI_B
    return ['{0}p{1}[{2}] = {3}'.format(' '*indent, side, p, pstr(P[p, DOF_OF[p]]))
            for p in range(ND)]


def derivative_lines(indent, names=None):
    r"""``d[p] += Phi_A[p, d] cs[col+d]``, the derivatives at the point"""
    names = range(ND) if names is None else names
    return ['{0}{1} += {2}*cs[col+{3}]'.format(' '*indent, D_NAMES[p],
            pstr(PHI_A[p, DOF_OF[p]]), DOF_OF[p]) for p in names]


def assembly(mname, pattern, blocks, symmetric=True):
    r"""Loops over the pairs of terms, ``mname`` the name of the matrix,
    ``pattern[p][q]`` the non-zero entries of ``K`` and ``blocks`` the pairs
    of degrees of freedom ``(d, e)`` of the matrix"""
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


def gradient_loop(indent, names):
    sp = ' '*indent
    L = [sp + 'for j in range(n):']
    L += basis_read('A', 'g', indent + 4)
    L += [sp + '    for i in range(m):']
    L += basis_read('A', 'f', indent + 8)
    L += [sp + '        col = col0 + DOF*(j*m + i)']
    L += derivative_lines(indent + 8, names)
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
    blocks_diag = [(d, d) for d in range(DOF)]
    NxxNL, NyyNL, NxyNL = sympy.symbols('NxxNL, NyyNL, NxyNL')
    H_NL = H.xreplace({th.Nxx: NxxNL, th.Nyy: NyyNL, th.Nxy: NxyNL})
    grads = list(range(6))
    sym_names = ', '.join(D_NAMES)

    out = [HEADER + 'r"""\n' + textwrap.fill(DESC, 76) + '''

Numerically integrated matrices and internal force vector of the model
``'plate_clpt_gl'``, generated by ``theory/shells/plate_clpt_gl/write_pyx.py``
from the derivation of ``theory/shells/plate_clpt_gl/plate_clpt_gl.py``.
The degrees of freedom of each term of the approximation are ``u, v, w``,
and the constitutive matrix ``Finput`` is the ``6 x 6`` matrix ``ABD`` of the
CLPT. The membrane strains are the Green-Lagrange strains::

    exx = u,x + (u,x^2 + v,x^2 + w,x^2)/2
    eyy = v,y + (u,y^2 + v,y^2 + w,y^2)/2
    gxy = u,y + v,x + u,x u,y + v,x v,y + w,x w,y

and the changes of curvature those of ``plate_clpt_donnell``, ``kxx =
-w,xx``, ``kyy = -w,yy``, ``kxy = -2 w,xy``. The mass and aerodynamic
matrices, ``fkM_num``, ``fkAx_num`` and ``fkAy_num``, are those of
:mod:`panels.models.plate_clpt_donnell_num`.

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
    r"""Constitutive stiffness matrix, ``KC0 + KCNL + KGNL``

    With ``NLgeom=1`` the matrix at the state ``cs`` is
    ``int [Phi]^T ([B]^T [ABD] [B] + [H(N_NL)]) [Phi] dA``, ``N_NL`` the
    membrane stresses of the quadratic strains, such that, with
    :func:`.fkG_num`, ``KT = fkC_num + fkG_num`` is the exact Jacobian of
    :func:`.calc_fint`. With ``NLgeom=0`` it is the linear constitutive
    stiffness matrix ``KC0``.

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
'''.format(COMMON=COMMON_CDEF, gr=', '.join(D_NAMES[:6]), basis=basis_cdef(),
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

    ``int [Phi]^T [H(N)] [Phi] dA``, with ``N = (Nxx0, Nyy0, Nxy0) + N_L``,
    ``N_L`` the membrane stresses of the *linear* strains at ``cs``, and
    ``[H(N)]`` with one block ``[[Nxx, Nxy], [Nxy, Nyy]]`` for each of the
    gradients of ``u``, ``v`` and ``w``. The result is homogeneous of degree
    one in ``cs``, which is what the linear buckling eigenvalue problem
    requires. The stress of the quadratic strains enters the tangent
    stiffness matrix through :func:`.fkC_num`, such that

        KT = KC0 + KCNL + KGNL   (fkC_num)
             + KG(Nbar + N_L)    (fkG_num)

    is the exact Jacobian of :func:`.calc_fint`.

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

    fdim = 3*m*m*n*n
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
           loops=assembly('kG', Hpat, blocks_diag)))

    # ------------------------------------------------------------- calc_fint
    S = (B.T*Matrix_s()).applyfunc(expand)
    fint_lines = []
    for d in range(DOF):
        terms = ['pA[{0}]*S[{0}]'.format(p) for p in DSET[d]]
        fint_lines.append('                        fint[col+{0}] += weight*({1})'.format(d, ' + '.join(terms)))
    out.append('''

def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    r"""Internal force vector ``int [Phi]^T [B]^T sigma dA``

    With the Green-Lagrange membrane strains and the changes of curvature at
    ``cs``, and ``sigma = [ABD] eps``.

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
           basis=basis_cdef('A'), FINPUT=FINPUT, POINT=POINT,
           dzero='\n'.join('                {0} = 0.'.format(nm) for nm in D_NAMES),
           dloop='\n'.join(gradient_loop(16, range(ND))),
           elines='\n'.join('                e[{0}] = {1}'.format(k, pstr(eps[k])) for k in range(NE)),
           Slines='\n'.join('                S[{0}] = {1}'.format(p, pstr(S[p])) for p in range(ND)),
           gA='\n'.join(basis_read('A', 'g', 20)), fA='\n'.join(basis_read('A', 'f', 24)),
           pA='\n'.join(phi_lines('A', 24)), fint='\n'.join(fint_lines)))
    text = ''.join(out)
    #NOTE c, p, q, r, s, k, l and parts of the tables are not used in every
    #     kernel, which is harmless
    return text


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

Analytical matrices of the model ``'plate_clpt_gl'``, integrated over the
full domain, generated by ``theory/shells/plate_clpt_gl/write_pyx.py``. The
linear constitutive stiffness ``fk0``, the mass ``fkM`` and the aerodynamic
``fkAx``, ``fkAy`` and ``fcA`` matrices are those of
:mod:`panels.models.plate_clpt_donnell`. The geometric stiffness matrix
``fkG0`` of the constant membrane stress state ``Nxx, Nyy, Nxy`` has one
block per displacement ``u, v, w``.

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
    r"""Geometric stiffness matrix of the constant membrane stress state
    ``Nxx, Nyy, Nxy``, with the blocks ``u-u``, ``v-v`` and ``w-w``"""
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
