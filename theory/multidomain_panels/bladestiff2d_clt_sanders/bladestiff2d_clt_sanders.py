# Symbolically calculates the matrices of the blade stiffener with a 2D
# formulation for the flange and for the base (panels.stiffener.BladeStiff2D)
# on a curved bay, model 'bladestiff2d_clt_sanders', and writes the kernels of
# the connection to panels/stiffener/models/bladestiff2d_clt_sanders.pyx
#
# Run:  python bladestiff2d_clt_sanders.py
#
# The flange remains a plate (plate_clpt_donnell), of length a and width bf,
# normal to the skin, with its own approximation functions (r along x, s
# along the width of the flange) and degrees of freedom (matrices *f).
#
# The stiffener's base (padup) becomes a cylindrical shell with the
# Sanders-Koiter kinematics of the skin (cylshell_clpt_sanders): it uses the
# approximation functions (f, g) and degrees of freedom of the bay, with the
# offset d of its mid-surface, and is integrated between y1 = ys - bb/2 and
# y2 = ys + bb/2 (matrices *y1y2). The Shell objects BladeStiff2D.base and
# BladeStiff2D.flange calculate these matrices with the kernels of
# panels/models/cylshell_clpt_sanders.pyx and panels/models/plate_clpt_donnell.pyx:
#
#     k0y1y2, k0f: fk0
#     kMy1y2, kMf: fkM
#     kG0f: fkG0 (the base adds no geometric stiffness, see BladeStiff2D.calc_kG)
#
# The rotation of the normal of the skin about x is, with Sanders,
#
#     phiy = -w,y + v/r
#
# The connection between the base (or the skin, when there is no base) at
# y = ys and the edge of the flange at eta_f = -1 is enforced with the
# penalty constants kt and kr, with the flange normal to the skin:
#
#     u = uf,  v = wf,  w = -vf,  -phiy = w,y - v/r = wf,y
#
# Donnell's kinematics (bladestiff2d_clt_donnell) has phiy = -w,y. Under a
# rigid rotation alpha of the cylinder about its axis, v = r alpha, w = 0,
# and the normals of the skin and the flange rotate by alpha: w,y - v/r =
# -alpha = wf,y, whereas w,y = 0; only the Sanders rotation leaves the
# connection free of energy under this rigid motion (check_rigid_rotation).
#
# The penalty energy gives the matrices kCss, kCsf and kCff, whose kernels
# are written to panels/stiffener/models/bladestiff2d_clt_sanders.pyx:
#
#     kCss: fkCss,  kCsf: fkCsf,  kCff: fkCff
#
# In the kernels the functions g and s of the connection are evaluated at
# eta(ys) and eta_f = -1, respectively, and only the functions f and r are
# integrated along x, in closed form with the integrals of Bardell's
# functions, integral_ff, whose names fApfBq (or fAprBq, rAprBq) give the
# functions and fields p, q of the two sides
import os
import re

import numpy as np
from sympy import Matrix as M, var, expand, Poly, simplify, collect, Symbol

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
PYX = os.path.join(REPO, 'panels', 'stiffener', 'models',
                   'bladestiff2d_clt_sanders.pyx')

var('a, b, r, bf, h, hf, d, rho')
var('kt, kr')
var('I0, I1, I2')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')

FIELDS = ('u', 'v', 'w')


def vectors(f, g, AB):
    r"""Shape functions of the term ``AB`` ('A' or 'B') of a domain with
    approximation functions named ``f`` along x and ``g`` along y"""
    s = lambda fname, gname: (var('%s%s%s' % (f, AB, fname))
                              *var('%s%s%s' % (g, AB, gname)))
    return dict(
        u=M([[s('u', 'u'), 0, 0]]),
        v=M([[0, s('v', 'v'), 0]]),
        w=M([[0, 0, s('w', 'w')]]),
        u_xi=M([[s('u_xi', 'u'), 0, 0]]),
        v_xi=M([[0, s('v_xi', 'v'), 0]]),
        w_xi=M([[0, 0, s('w_xi', 'w')]]),
        u_eta=M([[s('u', 'u_eta'), 0, 0]]),
        v_eta=M([[0, s('v', 'v_eta'), 0]]),
        w_eta=M([[0, 0, s('w', 'w_eta')]]),
        w_xixi=M([[0, 0, s('w_xixi', 'w')]]),
        w_xieta=M([[0, 0, s('w_xi', 'w_eta')]]),
        w_etaeta=M([[0, 0, s('w', 'w_etaeta')]]),
        )


sA, sB = vectors('f', 'g', 'A'), vectors('f', 'g', 'B')
sAf, sBf = vectors('r', 's', 'A'), vectors('r', 's', 'B')

F = M([[A11, A12, A16, B11, B12, B16],
       [A12, A22, A26, B12, B22, B26],
       [A16, A26, A66, B16, B26, B66],
       [B11, B12, B16, D11, D12, D16],
       [B12, B22, B26, D12, D22, D26],
       [B16, B26, B66, D16, D26, D66]])


# Base: Sanders' cylindrical shell (linear part), as cylshell_clpt_sanders

def phi(s, a, b, r=None):
    r"""Variations of the rotations ``(phix, phiy)`` of the normal, Sanders
    for ``r`` given, plate otherwise"""
    phix = -(2/a)*s['w_xi']
    phiy = -(2/b)*s['w_eta'] + (s['v']/r if r is not None else 0*s['v'])
    return phix, phiy


def B0(s, a, b, r=None):
    r"""Linear strain-displacement matrix, Sanders' cylindrical shell for
    ``r`` given, plate otherwise"""
    if r is None:
        return M([(2/a)*s['u_xi'],
                  (2/b)*s['v_eta'],
                  (2/b)*s['u_eta'] + (2/a)*s['v_xi'],
                 -(2/a)*(2/a)*s['w_xixi'],
                 -(2/b)*(2/b)*s['w_etaeta'],
                 -(2/a)*(2/b)*2*s['w_xieta']])
    return M([(2/a)*s['u_xi'],
              (2/b)*s['v_eta'] + s['w']/r,
              (2/b)*s['u_eta'] + (2/a)*s['v_xi'],
             -(2/a)*(2/a)*s['w_xixi'],
             -(2/b)*(2/b)*s['w_etaeta'] + (2/b)*s['v_eta']/r,
             -(2/a)*(2/b)*2*s['w_xieta'] + 3/(2*r)*(2/a)*s['v_xi']
             - 1/(2*r)*(2/b)*s['u_eta']])


def g5(s, a, b, r=None):
    r"""``(u, v, w, phix, phiy)`` of the displacement field ``u(z) = u + z
    phix``, ``v(z) = v + z phiy``, ``w(z) = w``"""
    phix, phiy = phi(s, a, b, r)
    return M([s['u'], s['v'], s['w'], phix, phiy])


# the moments I_n = int rho z^n dz of the laminate about the mid-surface of
# the skin, the offset d of the base being included in them
maux = M([[I0,  0,  0, I1,  0],
          [ 0, I0,  0,  0, I1],
          [ 0,  0, I0,  0,  0],
          [I1,  0,  0, I2,  0],
          [ 0, I1,  0,  0, I2]])

# Base of the stiffener (Sanders), and flange (plate)
k0y1y2 = (a*b/4)*B0(sA, a, b, r).T*F*B0(sB, a, b, r)
kMy1y2 = (a*b/4)*g5(sA, a, b, r).T*maux*g5(sB, a, b, r)
k0f = (a*bf/4)*B0(sAf, a, bf).T*F*B0(sBf, a, bf)
kMf = (a*bf/4)*g5(sAf, a, bf).T*maux*g5(sBf, a, bf)


# Penalty connection between the skin (s) and the flange (f)
#
#     U = 1/2 int_0^a [kt (u - uf)^2 + kt (v - wf)^2 + kt (w + vf)^2
#                      + kr (w,y - v/r - wf,y)^2] dx
#
# only the upper triangle of the connectivity matrix is needed, the block
# kCfs = kCsf^T is not written

rotA = -phi(sA, a, b, r)[1]     # w,y - v/r of the skin
rotB = -phi(sB, a, b, r)[1]
rotBf = (2/bf)*sBf['w_eta']     # wf,y of the flange
rotAf = (2/bf)*sAf['w_eta']

kCss = (a/2)*(kt*sA['u'].T*sB['u']
            + kt*sA['v'].T*sB['v']
            + kt*sA['w'].T*sB['w']
            + kr*rotA.T*rotB)

kCsf = (a/2)*(-kt*sA['u'].T*sBf['u']
              -kt*sA['v'].T*sBf['w']
              +kt*sA['w'].T*sBf['v']
              -kr*rotA.T*rotBf)

kCff = (a/2)*(kt*sAf['u'].T*sBf['u']
            + kt*sAf['v'].T*sBf['v']
            + kt*sAf['w'].T*sBf['w']
            + kr*rotAf.T*rotBf)


def check_rigid_rotation():
    r"""The connection energy vanishes under a rigid rotation of the
    cylinder about its axis

    ``v = r alpha``, ``w = 0`` on the skin and, the flange pointing inwards
    along ``-z`` with ``w = -vf``, ``wf = (r - yf) alpha`` along the width
    ``yf`` of the flange, so ``wf,y = -alpha``. The residuals of the four
    constraints vanish with Sanders' rotation; Donnell's ``w,y`` leaves
    ``alpha``.

    """
    alpha, yf = var('alpha, yf')
    u, v, w, w_y = 0, r*alpha, 0, 0
    uf, vf, wf = 0, 0, (r - yf)*alpha
    wf_y = wf.diff(yf)
    res = [u - uf, v - wf.subs(yf, 0), w + vf, (w_y - v/r) - wf_y]
    assert all(simplify(ri) == 0 for ri in res)
    assert simplify(w_y - wf_y) == alpha
    print('check_rigid_rotation: passed')


def check_base_is_cylshell_clpt_sanders():
    r"""The linear stiffness and the mass of the base are those of
    ``cylshell_clpt_sanders`` (``build()`` of
    ``theory/shells/cylshell_clpt_sanders/cylshell_clpt_sanders.py``)"""
    import importlib.util
    path = os.path.join(HERE, '..', '..', 'shells', 'cylshell_clpt_sanders',
                        'cylshell_clpt_sanders.py')
    spec = importlib.util.spec_from_file_location('sanders', path)
    sanders = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sanders)
    ref = sanders.build()
    # the shell integrands are per unit area of the natural domain
    for mine, theirs in ((k0y1y2, ref['kC0']), (kMy1y2, ref['kM'])):
        diff = expand(mine - (a*b/4)*theirs)
        assert diff == M.zeros(*diff.shape)
    print('check_base_is_cylshell_clpt_sanders: passed')


# Kernels of the connection, written to the .pyx

SIDES = {'f': ('f', 'g', ''), 'r': ('r', 's', 'f')}


def integrals(expr):
    r"""Replaces the products of the functions along x of the two sides by
    the names of their integrals, e.g. ``fAv*fBw -> fAvfBw``"""
    xfuncs = [Symbol('%s%s%s' % (f, AB, p)) for f in 'fr' for AB in 'AB'
              for p in FIELDS]
    expr = expand(expr)
    present = [x for x in xfuncs if expr.has(x)]
    if not present:
        return expr, set()
    out = 0
    names = set()
    for monom, coeff in Poly(expr, *present).terms():
        pair = [x for x, e in zip(present, monom) if e == 1]
        assert len(pair) == 2 and sum(monom) == 2
        name = str(pair[0]) + str(pair[1])
        names.add(name)
        out += coeff*Symbol(name)
    return out, names


def integral_call(name):
    r"""``integral_ff`` call of the integral ``name``, e.g. ``fAvrBw``"""
    m = re.fullmatch(r'([fr])A([uvw])([fr])B([uvw])', name)
    fa, pa, fb, pb = m.groups()
    sa, sb = SIDES[fa][2], SIDES[fb][2]

    ia = 'i' if sa == '' else 'i1'
    kb = 'k' if sb == '' else 'k1'
    bc = lambda p, s: ', '.join(t % (p, s) for t in
                                ('x1%s%s', 'x1%sr%s', 'x2%s%s', 'x2%sr%s'))
    return '%s = integral_ff(%s, %s, %s, %s)' % (name, ia, kb, bc(pa, sa),
                                                 bc(pb, sb))


def cexpr(expr):
    r"""Cython source of ``expr``, expanded, the powers written as products, e.g.
    ``b**2 -> (b*b)``"""
    expr = expand(expr)
    return re.sub(r'(\w+)\*\*(\d+)',
                  lambda m: '(' + '*'.join([m.group(1)]*int(m.group(2))) + ')',
                  str(expr))


def entries(m):
    out = []
    names = set()
    for (i, j), mij in np.ndenumerate(np.array(m.tolist(), dtype=object)):
        if mij == 0:
            continue
        e, n = integrals(mij)
        names |= n
        out.append((i, j, cexpr(e)))
    return out, sorted(names)


BC_PARAMS = """          double x1u{s}, double x1ur{s}, double x2u{s}, double x2ur{s},
          double x1v{s}, double x1vr{s}, double x2v{s}, double x2vr{s},
          double x1w{s}, double x1wr{s}, double x2w{s}, double x2wr{s},
          double y1u{s}, double y1ur{s}, double y2u{s}, double y2ur{s},
          double y1v{s}, double y1vr{s}, double y2v{s}, double y2vr{s},
          double y1w{s}, double y1wr{s}, double y2w{s}, double y2wr{s},"""

BC_NAMES = ('x1u{s}, x1ur{s}, x2u{s}, x2ur{s}, x1v{s}, x1vr{s}, x2v{s}, '
            'x2vr{s}, x1w{s}, x1wr{s}, x2w{s}, x2wr{s}, y1u{s}, y1ur{s}, '
            'y2u{s}, y2ur{s}, y1v{s}, y1vr{s}, y2v{s}, y2vr{s}, y1w{s}, '
            'y1wr{s}, y2w{s}, y2wr{s}')

DOC_KT = """    kt, kr : float
        Translational and rotational penalty constants of the connection,
        force per unit area and force per unit length, see
        :func:`.calc_kt_kr`."""

DOC_SKIN = """    ys : float
        Coordinate `y` of the stiffener on the skin, arc length.
    a, b : float
        Dimensions of the skin along `x` and `y`, lengths, `b` being an arc
        length.
    r : float
        Radius of the skin, length."""

DOC_BC_SKIN = """    {names} : float
        Boundary flags of the approximation functions of the skin: the
        translation (``t``) and rotation (``r``) flags at the edges ``1`` and
        ``2`` along ``x`` and ``y`` of each field ``u``, ``v``, ``w`` used by
        the kernel, see :class:`.Shell`.""".format(names=BC_NAMES.format(s=''))

DOC_BC_FLANGE = """    {names} : float
        Boundary flags of the approximation functions of the flange, in the
        same order as those of the skin.""".format(names=BC_NAMES.format(s='f'))

DOC_SIZE = """    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the block starts."""

MODULE_DOC = r'''r"""
Kernels of the connection between a cylindrical skin with Sanders'
kinematics and the flange of the blade stiffener modelled as a plate

Penalty stiffness matrices of the connection between the skin (or the base)
of a curved bay at `y = y_s` and the edge `\eta_f = -1` of the flange of
:class:`.BladeStiff2D`, a plate normal to the skin: the block of the skin
:func:`fkCss`, the coupling block :func:`fkCsf` and the block of the flange
:func:`fkCff`, as :class:`scipy.sparse.coo_matrix`. The degrees of freedom
of each term are ``u, v, w`` of the skin and ``u_f, v_f, w_f`` of the
flange, in its own axes. The module is written by
``theory/multidomain_panels/bladestiff2d_clt_sanders/bladestiff2d_clt_sanders.py``.

Notes
-----
The penalty energy of the connection is

.. math::

    U = \frac{1}{2} \int_0^a k_t \left[(u - u_f)^2 + (v - w_f)^2 + (w +
    v_f)^2\right] + k_r \left(w_{,y} - \frac{v}{r} - w_{f,y}\right)^2 \, dx

with `u, v, w` and `w_{,y}` of the skin at `y = y_s`, and `u_f, v_f, w_f`
and `w_{f,y}` of the flange at its edge `\eta_f = -1`, the local axes of
the flange being rotated by 90 degrees about `x`. The rotation of the
normal of the skin about `x` is that of Sanders [1]_, `-\phi_y = w_{,y} -
v/r`, as in the kinematics of ``cylshell_clpt_sanders``; with Donnell's
`w_{,y}`, the kernels of ``bladestiff2d_clt_donnell``, a rigid rotation of
the cylinder about its axis would strain the connection. Each kernel gives
the Hessian of one part of `U`.

References
----------
.. [1] Sanders, J. L., Nonlinear theories for thin shells, Quarterly of
   Applied Mathematics, 21(1), 21-36, 1963. https://doi.org/10.1090/qam/147023

"""'''

HEADER = """#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: profile=False
#cython: infer_types=False
"""

IMPORTS = """from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE


cdef int DOF = 3
cdef int DOF1 = 3

cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil

cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
"""

GTAB = """            gtab[0, j] = f(j, eta, y1u, y1ur, y2u, y2ur)
            gtab[1, j] = f(j, eta, y1v, y1vr, y2v, y2vr)
            gtab[2, j] = f(j, eta, y1w, y1wr, y2w, y2wr)
            gtab[3, j] = fp(j, eta, y1w, y1wr, y2w, y2wr)"""

GTABF = """            gtabf[0, {l}] = f({l}, -1., y1uf, y1urf, y2uf, y2urf)
            gtabf[1, {l}] = f({l}, -1., y1vf, y1vrf, y2vf, y2vrf)
            gtabf[2, {l}] = f({l}, -1., y1wf, y1wrf, y2wf, y2wrf)
            gtabf[3, {l}] = fp({l}, -1., y1wf, y1wrf, y2wf, y2wrf)"""


def assembly(name, ents, indent):
    sp = ' '*indent
    lines = []
    for i, j, e in ents:
        lines += ['%sc += 1' % sp,
                  '%s%sr[c] = row+%d' % (sp, name, i),
                  '%s%sc[c] = col+%d' % (sp, name, j),
                  '%s%sv[c] += %s' % (sp, name, e)]
    return '\n'.join(lines)


def write_fkCss():
    ents, names = entries(kCss)
    return '''

def fkCss(double kt, double kr, double ys, double a, double b, double r,
          int m, int n,
{bcs}
          int size, int row0, int col0):
    r"""Penalty stiffness matrix of the connection, block of the skin

    Block of the Ritz constants of the skin of the penalty stiffness
    matrix that connects the cylindrical skin at `y = y_s` to the flange.

    Parameters
    ----------
{doc_kt}
{doc_skin}
    m, n : int
        Numbers of terms of the approximation of the skin along `x` and `y`.
{doc_bc}
{doc_size}

    Returns
    -------
    kCss : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Notes
    -----
    Hessian of the penalty energy of the module with respect to the
    Ritz constants of the skin:

    .. math::

        \\boldsymbol{{K}}_{{ss}} = \\int_0^a k_t \\left[(\\boldsymbol{{S}}^u)^\\top
        \\boldsymbol{{S}}^u + (\\boldsymbol{{S}}^v)^\\top \\boldsymbol{{S}}^v +
        (\\boldsymbol{{S}}^w)^\\top \\boldsymbol{{S}}^w\\right] + k_r
        \\boldsymbol{{S}}_\\phi^\\top \\boldsymbol{{S}}_\\phi \\, dx,
        \\qquad \\boldsymbol{{S}}_\\phi = \\boldsymbol{{S}}^w_{{,y}} -
        \\frac{{1}}{{r}} \\boldsymbol{{S}}^v

    with the rows `\\boldsymbol{{S}}` of the approximation functions of the skin
    at `y = y_s` and `\\boldsymbol{{S}}_\\phi` that of Sanders' rotation `w_{{,y}}
    - v/r`.

    The integral along `x` is evaluated in closed form with the
    integrals of Bardell's functions, `dx = (a/2) d\\xi`, and the functions
    along `y` of the skin are evaluated at `\\eta_s = 2 y_s/b - 1`.

    """
    cdef double [:, ::1] gtab
    cdef int i, k, j, l, c, row, col
    cdef double eta

    cdef long [:] kCssr, kCssc
    cdef double [:] kCssv

    cdef double {names}
    cdef double gAu, gBu, gAv, gBv, gAw, gBw, gAw_eta, gBw_eta

    eta = 2*ys/b - 1.

    fdim = {nent}*m*n*m*n

    kCssr = np.zeros((fdim,), dtype=INT)
    kCssc = np.zeros((fdim,), dtype=INT)
    kCssv = np.zeros((fdim,), dtype=DOUBLE)

    gtab = np.zeros((4, n), dtype=DOUBLE)

    with nogil:
        # kCss
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j in range(n):
{gtab}

        for i in range(m):
            for k in range(m):
{ints}

                for j in range(n):
                    gAu = gtab[0, j]
                    gAv = gtab[1, j]
                    gAw = gtab[2, j]
                    gAw_eta = gtab[3, j]

                    for l in range(n):
                        row = row0 + DOF*(j*m + i)
                        col = col0 + DOF*(l*m + k)

                        #NOTE symmetry
                        if row > col:
                            continue

                        gBu = gtab[0, l]
                        gBv = gtab[1, l]
                        gBw = gtab[2, l]
                        gBw_eta = gtab[3, l]

{asm}

    kCss = coo_matrix((kCssv, (kCssr, kCssc)), shape=(size, size))

    return kCss
'''.format(bcs=BC_PARAMS.format(s=''), doc_kt=DOC_KT, doc_skin=DOC_SKIN,
           doc_bc=DOC_BC_SKIN, doc_size=DOC_SIZE, names=', '.join(names),
           nent=len(ents), gtab=GTAB,
           ints='\n'.join(' '*16 + integral_call(nm) for nm in names),
           asm=assembly('kCss', ents, 24))


def write_fkCsf():
    ents, names = entries(kCsf)
    return '''

def fkCsf(double kt, double kr, double ys, double a, double b, double r,
          double bf, int m, int n, int m1, int n1,
{bcs}
{bcsf}
          int size, int row0, int col0):
    r"""Penalty stiffness matrix of the connection, coupling block

    Block coupling the Ritz constants of the cylindrical skin (rows) and of
    the flange (columns) of the penalty stiffness matrix of the connection,
    assembled in full.

    Parameters
    ----------
{doc_kt}
{doc_skin}
    bf : float
        Height of the flange, its width `b_f` along its own `y`, a length.
    m, n : int
        Numbers of terms of the approximation of the skin along `x` and `y`.
    m1, n1 : int
        Numbers of terms of the approximation of the flange.
{doc_bc}
{doc_bcf}
{doc_size}

    Returns
    -------
    kCsf : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. The block is assembled in full.

    Notes
    -----
    Mixed second derivative of the penalty energy of the module:

    .. math::

        \\boldsymbol{{K}}_{{sf}} = -\\int_0^a k_t \\left[(\\boldsymbol{{S}}^u)^\\top
        \\boldsymbol{{S}}^{{u_f}} + (\\boldsymbol{{S}}^v)^\\top \\boldsymbol{{S}}^{{w_f}} -
        (\\boldsymbol{{S}}^w)^\\top \\boldsymbol{{S}}^{{v_f}}\\right] + k_r
        \\boldsymbol{{S}}_\\phi^\\top \\boldsymbol{{S}}^{{w_f}}_{{,y}} \\, dx,
        \\qquad \\boldsymbol{{S}}_\\phi = \\boldsymbol{{S}}^w_{{,y}} -
        \\frac{{1}}{{r}} \\boldsymbol{{S}}^v

    with the rows `\\boldsymbol{{S}}` of the approximation functions of the skin
    at `y = y_s` and those of the flange at its edge `\\eta_f = -1`, and
    `\\boldsymbol{{S}}_\\phi` that of Sanders' rotation `w_{{,y}} - v/r` of the
    skin.

    The integral along `x` is evaluated in closed form with the
    integrals of Bardell's functions, `dx = (a/2) d\\xi`, and the functions
    along `y` of the skin are evaluated at `\\eta_s = 2 y_s/b - 1`.

    """
    cdef double [:, ::1] gtab, gtabf
    cdef int i, j, k1, l1, c, row, col
    cdef double eta

    cdef long [:] kCsfr, kCsfc
    cdef double [:] kCsfv

    cdef double {names}
    cdef double gAu, gAv, gAw, gAw_eta, sBu, sBv, sBw, sBw_eta

    eta = 2*ys/b - 1.

    fdim = {nent}*m*n*m1*n1

    kCsfr = np.zeros((fdim,), dtype=INT)
    kCsfc = np.zeros((fdim,), dtype=INT)
    kCsfv = np.zeros((fdim,), dtype=DOUBLE)

    gtab = np.zeros((4, n), dtype=DOUBLE)
    gtabf = np.zeros((4, n1), dtype=DOUBLE)

    with nogil:
        # kCsf
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j in range(n):
{gtab}
        for l1 in range(n1):
{gtabf}

        for i in range(m):
            for k1 in range(m1):
{ints}

                for j in range(n):
                    gAu = gtab[0, j]
                    gAv = gtab[1, j]
                    gAw = gtab[2, j]
                    gAw_eta = gtab[3, j]

                    for l1 in range(n1):
                        row = row0 + DOF1*(j*m + i)
                        col = col0 + DOF1*(l1*m1 + k1)

                        #NOTE symmetry not applicable here
                        #if row > col:
                            #continue

                        sBu = gtabf[0, l1]
                        sBv = gtabf[1, l1]
                        sBw = gtabf[2, l1]
                        sBw_eta = gtabf[3, l1]

{asm}

    kCsf = coo_matrix((kCsfv, (kCsfr, kCsfc)), shape=(size, size))

    return kCsf
'''.format(bcs=BC_PARAMS.format(s=''), bcsf=BC_PARAMS.format(s='f'),
           doc_kt=DOC_KT, doc_skin=DOC_SKIN, doc_bc=DOC_BC_SKIN,
           doc_bcf=DOC_BC_FLANGE, doc_size=DOC_SIZE, names=', '.join(names),
           nent=len(ents), gtab=GTAB, gtabf=GTABF.format(l='l1'),
           ints='\n'.join(' '*16 + integral_call(nm) for nm in names),
           asm=assembly('kCsf', ents, 24))


def write_fkCff():
    ents, names = entries(kCff)
    return '''

def fkCff(double kt, double kr, double a, double bf, int m1, int n1,
{bcsf}
          int size, int row0, int col0):
    r"""Penalty stiffness matrix of the connection, block of the flange

    Block of the Ritz constants of the flange of the penalty stiffness
    matrix that connects its edge `\\eta_f = -1` to the skin.

    Parameters
    ----------
{doc_kt}
    a : float
        Length of the skin and of the flange along `x`.
    bf : float
        Height of the flange, its width `b_f` along its own `y`, a length.
    m1, n1 : int
        Numbers of terms of the approximation of the flange.
{doc_bcf}
{doc_size}

    Returns
    -------
    kCff : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper
        triangle is assembled; the caller symmetrizes it.

    Notes
    -----
    Hessian of the penalty energy of the module with respect to the
    Ritz constants of the flange, a plate, the same as that of
    ``bladestiff2d_clt_donnell``:

    .. math::

        \\boldsymbol{{K}}_{{ff}} = \\int_0^a k_t \\left[(\\boldsymbol{{S}}^{{u_f}})^\\top
        \\boldsymbol{{S}}^{{u_f}} + (\\boldsymbol{{S}}^{{v_f}})^\\top \\boldsymbol{{S}}^{{v_f}}
        + (\\boldsymbol{{S}}^{{w_f}})^\\top \\boldsymbol{{S}}^{{w_f}}\\right] + k_r
        (\\boldsymbol{{S}}^{{w_f}}_{{,y}})^\\top \\boldsymbol{{S}}^{{w_f}}_{{,y}} \\, dx

    with the rows `\\boldsymbol{{S}}` of the approximation functions of the flange
    at its edge `\\eta_f = -1`, `w_{{f,y}} = (2/b_f) w_{{f,\\eta}}`.

    The integral along `x` is evaluated in closed form with the
    integrals of Bardell's functions, `dx = (a/2) d\\xi`,
    with `a` the length of the flange.

    """
    cdef double [:, ::1] gtabf
    cdef int i1, k1, j1, l1, c, row, col

    cdef long [:] kCffr, kCffc
    cdef double [:] kCffv

    cdef double {names}
    cdef double sAu, sBu, sAv, sBv, sAw, sBw, sAw_eta, sBw_eta

    fdim = {nent}*m1*n1*m1*n1

    kCffr = np.zeros((fdim,), dtype=INT)
    kCffc = np.zeros((fdim,), dtype=INT)
    kCffv = np.zeros((fdim,), dtype=DOUBLE)

    gtabf = np.zeros((4, n1), dtype=DOUBLE)

    with nogil:

        # kCff
        c = -1
        # approximation functions evaluated once, see panels/dev/tabulate.py
        for j1 in range(n1):
{gtabf}

        for i1 in range(m1):
            for k1 in range(m1):
{ints}

                for j1 in range(n1):
                    sAu = gtabf[0, j1]
                    sAv = gtabf[1, j1]
                    sAw = gtabf[2, j1]
                    sAw_eta = gtabf[3, j1]

                    for l1 in range(n1):
                        row = row0 + DOF1*(j1*m1 + i1)
                        col = col0 + DOF1*(l1*m1 + k1)

                        #NOTE symmetry
                        if row > col:
                            continue

                        sBu = gtabf[0, l1]
                        sBv = gtabf[1, l1]
                        sBw = gtabf[2, l1]
                        sBw_eta = gtabf[3, l1]

{asm}

    kCff = coo_matrix((kCffv, (kCffr, kCffc)), shape=(size, size))

    return kCff
'''.format(bcsf=BC_PARAMS.format(s='f'), doc_kt=DOC_KT,
           doc_bcf=DOC_BC_FLANGE, doc_size=DOC_SIZE, names=', '.join(names),
           nent=len(ents), gtabf=GTABF.format(l='j1'),
           ints='\n'.join(' '*16 + integral_call(nm) for nm in names),
           asm=assembly('kCff', ents, 24))


def write_pyx(path=PYX):
    text = (HEADER + MODULE_DOC + '\n' + IMPORTS + write_fkCss()
            + write_fkCsf() + write_fkCff())
    with open(path, 'w', newline='\n') as f:
        f.write(text)
    print('written', path)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, REPO)
    check_rigid_rotation()
    check_base_is_cylshell_clpt_sanders()
    out = sys.argv[1] if len(sys.argv) > 1 else PYX
    write_pyx(out)
