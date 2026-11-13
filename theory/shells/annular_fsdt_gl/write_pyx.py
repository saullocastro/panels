"""Generate the kernels of an annular-sector model of this folder

Writes ``panels/models/<model>_num.pyx`` (numerically integrated matrices,
internal force vector and fields) and ``panels/models/<model>.pyx`` (the
matrices of the whole domain called by :class:`.Shell` for the default
integration) from the derivation of ``<model>.py``, ``<model>`` being the
name of this folder::

    python write_pyx.py [output folder]

Kernel layout
-------------
At each integration point the generalized derivatives ``d = [Phi] c``
(``<model>.D``), the NE x ND matrix ``[B] = d eps/d d`` at ``d`` and the ND x
ND matrix ``K`` of the kernel (``[B]^T [F] [B] + [H(sigma, d)] -
[H(sigma_L, 0)]`` for ``fkC_num``, ``[H(sigma_L + Nbar, 0)]`` for
``fkG_num``, ``[J]^T [I] [J]`` for ``fkM_num``) are evaluated once, with
``[H(sigma, d)] = sum_r sigma_r d^2 eps_r/d d^2``. The entry of the pair of
terms ``A = (i, j)``, ``B = (k, l)`` and the degrees of freedom ``e``,
``f`` is ``sum_pq Phi_A[p, e] K[p, q] Phi_B[q, f]``, computed as ``TA[e, q]
= sum_p Phi_A[p, e] K[p, q]`` in the loop over the terms ``A`` and ``sum_q
TA[e, q] Phi_B[q, f]`` in the loop over the terms ``B``; ``Phi_A[p, e]`` is
zero unless the generalized derivative ``p`` belongs to the degree of
freedom ``e``. The coefficients of the strains depend on the radius ``rho =
R - x`` of the point, so that the integrands are rational functions of
``x``, integrated with the Gauss-Legendre rule.

The script is self-contained and the same in every folder
``annular_fsdt_*``: the model, its strains and the text describing them
come from ``<model>.py``. The degrees of freedom of a term are ``u, v, w,
phix, phiy``, the strains and stress resultants the eight of the first-order
shear deformation theory (FSDT).

"""
import importlib
import os
import sys

import sympy
from sympy import Symbol
from sympy.printing.str import StrPrinter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)

MODEL = os.path.basename(HERE)
th = importlib.import_module(MODEL)

ND = len(th.D)
DOF = th.DOF
NE = th.NE
NF = len(th.FIELDS)
#: field index of each generalized derivative
FLD = [th.FIELDS.index(th.D_OF[d.name][0]) for d in th.D]
assert FLD == sorted(FLD), 'the generalized derivatives must be grouped by field'
#: first and last + 1 generalized derivative of each degree of freedom
P0 = tuple(FLD.index(e) for e in range(DOF))
P1 = tuple(len(FLD) - FLD[::-1].index(e) for e in range(DOF))
#: index of a generalized derivative by name
IDX = {d.name: k for k, d in enumerate(th.D)}

S_ = Symbol('s')
IRHO = Symbol('irho')


class _Printer(StrPrinter):
    def _print_Rational(self, e):
        return repr(float(e))

    def _print_Integer(self, e):
        return '%d.' % int(e)

    def _print_Pow(self, e, rational=False):
        b, p = e.as_base_exp()
        if p.is_Integer and 0 < p <= 4:
            return '*'.join(['(%s)' % self._print(b)]*int(p))
        return StrPrinter._print_Pow(self, e, rational)


def pstr(e, sname='s'):
    """Cython expression, ``R/rho`` and ``1/rho`` as ``s`` (or ``sname``)
    and ``irho``"""
    e = sympy.expand(sympy.sympify(e).subs(th.R, S_*th.rho)
                     .subs(th.rho, 1/IRHO))
    if sname != 's':
        e = e.subs(S_, Symbol(sname))
    return _Printer().doprint(e)


B = th.build()
#: the generalized derivatives as the entries of the array ``dv``
DV = {d: Symbol('dv[%d]' % k) for k, d in enumerate(th.D)}
#: the stresses of the configuration and of its linear strains
SIG_NAMES = tuple(sg.name for sg in th.STRESSES)
SIG = [Symbol(n) for n in SIG_NAMES]
SIGL = [Symbol(n + 'L') for n in SIG_NAMES]
#: names of the strains, conjugate to the stresses
EPS_NAMES = ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz')
assert len(SIG_NAMES) == NE == len(EPS_NAMES)

HEADER = """#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
"""

MODULE_NOTES = r"""
The annular sector lies in a plane, between the outer radius ``r``,
``shell.r``, and the inner radius ``r - a``: ``a``, ``shell.a``, is its
radial width and ``b``, ``shell.b``, the arc length of its outer edge. The
coordinate `x = r - \rho` runs from the outer edge inwards, `y = r \theta`
along the outer edge, with the polar coordinates `(\rho, \theta)` of the
plane; ``u``, ``v``, ``w`` are the displacements along
`\boldsymbol{e}_x = -\boldsymbol{e}_\rho`, `\boldsymbol{e}_y =
\boldsymbol{e}_\theta` and `\boldsymbol{e}_z = \boldsymbol{e}_x \times
\boldsymbol{e}_y`, and ``phix``, ``phiy`` the rotations of the normal.
These are the local axes of the flange of the connection ``'BFxcte'``
whose base is a cylinder of radius ``r``: the web of a ring
frame of a stiffened cylinder is an application, with ``a`` the height of
the web, ``b`` the arc length of its outer edge and ``r`` its outer radius.
The laminate spans `d - h/2 \le z \le d + h/2` about the reference plane,
`d` being the offset ``shell.offset``, which enters through the matrix
``ABD`` and the moments of the density. The matrix ``shell.ABD`` of the
first-order shear deformation theory has the transverse shear stiffness in
its last two rows and columns, with the shear correction of
:class:`.Shell`.

Notes
-----
""" + th.STRAIN_NOTES + r"""
The element of area is `dA = (\rho/r)\, dx\, dy`. The integrands are
rational functions of `x`, integrated with the Gauss-Legendre rule over the
domain of :meth:`.Shell.integration_limits`.

References
----------
""" + th.REFERENCES

SHELL_PARAM = """    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the numbers
        of terms ``m`` and ``n`` and the boundary flags of the approximation
        functions of each field."""

#: boundary flags of each edge direction and field
BC_GROUPS = [['%s%s%s%s' % (xy, k, f, r) for k in '12' for r in ('', 'r')]
             for xy in 'xy' for f in th.FIELDS]
BC_NAMES = [n for g in BC_GROUPS for n in g]
BC_CDEF = '\n'.join('    cdef double ' + ', '.join(g) for g in BC_GROUPS)
BC_READ = '\n'.join('    ' + '; '.join('%s = shell.%s' % (n, n) for n in g)
                    for g in BC_GROUPS)

COMMON_CDEF = """    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, intx, inty, xi, eta, xp, rho, s, irho, weight
    cdef int m, n, i, j, k, l, c, p, q, r, e, f2, row, col, ptx, pty
{bc}
    cdef double [:, ::1] ftab, gtab
    cdef double [::1] xis, etas, weights_xi, weights_eta
    cdef double pA[{nd}]
    cdef double pB[{nd}]
    cdef double dv[{nd}]
    cdef double K[{nd2}]
    cdef double TA[{dofnd}]
""".format(bc=BC_CDEF, nd=ND, nd2=ND*ND, dofnd=DOF*ND)

SETUP = """    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    R = shell.r
    m = shell.m
    n = shell.n
    if not R > a:
        raise ValueError('the outer radius r must be larger than the radial '
                         'width a, got r={{0}} and a={{1}}'.format(R, a))
    #NOTE physical limits of the integration domain, resolved and validated
    #     by Shell.integration_limits()
    x1, x2, y1, y2 = shell.integration_limits()
{bc}
    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)
    xi1 = x1/a*2 - 1
    xi2 = x2/a*2 - 1
    eta1 = y1/b*2 - 1
    eta2 = y2/b*2 - 1
    intx = x2 - x1
    inty = y2 - y1
    ftab = np.zeros(({nt}, m), dtype=DOUBLE)
    gtab = np.zeros(({nt}, n), dtype=DOUBLE)
""".format(bc=BC_READ, nt=3*NF)


def tab_lines(indent, tab, k, coord, L, s1, s2):
    """Rows ``3*field + order`` of the table of the functions of ``coord``"""
    t = ' '*indent
    out = []
    for fi, fld in enumerate(th.FIELDS):
        flags = '%s, %s1%sr, %s2%s, %s2%sr' % (
            '%s1%s' % (s1, fld), s1, fld, s1, fld, s1, fld)
        out.append('%s%s[%d, %s] = f(%s, %s, %s)' % (t, tab, 3*fi, k, k, coord, flags))
        out.append('%s%s[%d, %s] = (2/%s)*fp(%s, %s, %s)' % (t, tab, 3*fi + 1, k, L, k, coord, flags))
        out.append('%s%s[%d, %s] = (2/%s)*(2/%s)*fpp(%s, %s, %s)' % (
            t, tab, 3*fi + 2, k, L, L, k, coord, flags))
    return '\n'.join(out)


TABS = lambda indent: ('%sfor i in range(m):\n%s\n%sfor j in range(n):\n%s' % (
    ' '*indent, tab_lines(indent + 4, 'ftab', 'i', 'xi', 'a', 'x', 'x'),
    ' '*indent, tab_lines(indent + 4, 'gtab', 'j', 'eta', 'b', 'y', 'y')))

POINT = """        for ptx in range(nx):
            for pty in range(ny):
                xi = (xis[ptx] + 1)/2*(xi2 - xi1) + xi1
                eta = (etas[pty] + 1)/2*(eta2 - eta1) + eta1
                xp = (xi + 1)*a/2
                rho = R - xp
                s = R/rho
                irho = 1./rho
                # per unit area of the natural domain, times dA/(dx dy)
                weight = weights_xi[ptx]*weights_eta[pty]*intx*inty/4*rho/R
{extra}
""" + TABS(16) + '\n'


def phi_lines(side, ii, jj, indent):
    """``Phi`` of the term ``(ii, jj)``, the generalized derivatives"""
    t = ' '*indent
    out = []
    for k, d in enumerate(th.D):
        fld, ox, oy = th.D_OF[d.name]
        fi = th.FIELDS.index(fld)
        out.append('%sp%s[%d] = ftab[%d, %s]*gtab[%d, %s]'
                   % (t, side, k, 3*fi + ox, ii, 3*fi + oy, jj))
    return '\n'.join(out)


def dv_lines(indent, vec='cs'):
    """Generalized derivatives ``dv`` at the point from the vector ``vec``"""
    t = ' '*indent
    return '\n'.join([
        t + 'for p in range(%d):' % ND,
        t + '    dv[p] = 0.',
        t + 'for j in range(n):',
        t + '    for i in range(m):',
        t + '        col = col0 + %d*(j*m + i)' % DOF,
        phi_lines('A', 'i', 'j', indent + 8),
        t + '        for p in range(%d):' % ND,
        t + '            dv[p] += pA[p]*%s[col + FLD[p]]' % vec,
    ])


ASSEMBLY = """                c = -1
                for j in range(n):
                    for i in range(m):
                        row = row0 + {dof}*(j*m + i)
{phiA}
                        # TA[e, q] = sum_p Phi_A[p, e] K[p, q]
                        for e in range({dof}):
                            for q in range({nd}):
                                TA[e*{nd} + q] = 0.
                                for p in range(P0[e], P1[e]):
                                    TA[e*{nd} + q] += pA[p]*K[p*{nd} + q]
                        for l in range(n):
                            for k in range(m):
                                col = col0 + {dof}*(l*m + k)
                                #NOTE upper triangle, the caller symmetrizes
                                if row > col:
                                    continue
{phiB}
                                for e in range({dof}):
                                    for f2 in range({dof}):
                                        c += 1
                                        if ptx == 0 and pty == 0:
                                            kr[c] = row + e
                                            kc[c] = col + f2
                                        for q in range(P0[f2], P1[f2]):
                                            kv[c] += weight*TA[e*{nd} + q]*pB[q]
""".format(dof=DOF, nd=ND, phiA=phi_lines('A', 'i', 'j', 24),
           phiB=phi_lines('B', 'k', 'l', 32))


def matrix_assign(name, Mx, indent, ncols):
    t = ' '*indent
    return '\n'.join('%s%s[%d] = %s' % (t, name, i*ncols + j,
                                        pstr(Mx[i, j].subs(DV, simultaneous=True)))
                     for i in range(Mx.shape[0]) for j in range(Mx.shape[1])
                     if Mx[i, j] != 0)


def strain_lines(indent, eps, prefix='eps'):
    t = ' '*indent
    return '\n'.join('%s%s%d = %s' % (t, prefix, r,
                                      pstr(eps[r].subs(DV, simultaneous=True)))
                     for r in range(NE))


def stress_lines(indent, names, eps_prefix='eps', rows=range(NE)):
    t = ' '*indent
    return '\n'.join('%s%s = %s' % (t, names[r], ' + '.join(
        'F[%d]*%s%d' % (r*NE + q, eps_prefix, q) for q in range(NE)))
        for r in rows)


def H_lines(indent, Hm, names, op='+='):
    """Entries of ``[H]`` for the stresses ``names`` at ``dv``"""
    t = ' '*indent
    Hs = Hm.subs(dict(zip(th.STRESSES, names)), simultaneous=True)
    out = []
    for i in range(ND):
        for j in range(ND):
            if Hs[i, j] != 0:
                out.append('%sK[%d] %s %s' % (t, i*ND + j, op,
                           pstr(Hs[i, j].subs(DV, simultaneous=True))))
    return '\n'.join(out)


FINPUT = """    cdef double F[{ne2}]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, {ne}, {ne}):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == ({ne}, {ne}):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range({ne}):
            for j in range({ne}):
                F[i*{ne} + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')
""".format(ne=NE, ne2=NE*NE)

F_POINT = """                if one_F_each_point == 1:
                    for i in range({ne}):
                        for j in range({ne}):
                            F[i*{ne} + j] = Fnxny[ptx, pty, i, j]""".format(ne=NE)

PARAMS_COMMON = """    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this panel start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`."""

FINPUT_PARAM = """    Finput : array-like
        Constitutive matrix of the first-order shear deformation theory,
        ``[[A, B, 0], [B, D, 0], [0, 0, A_s]]`` with the transverse shear
        stiffness ``A_s``, shape ``(8, 8)`` for the whole domain, e.g.
        ``shell.ABD``, or ``(nx, ny, 8, 8)`` with one matrix per integration
        point."""

CS_PARAM = """    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements."""

RAISES = """    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape, ``shell`` is not a
        :class:`.Shell` or its outer radius ``r`` is not larger than its
        radial width ``a``."""

RETURNS = """    Returns
    -------
    {name} : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper triangle is
        assembled; the caller symmetrizes it."""


def fill(doc, kname):
    """Replace the placeholders ``<<key>>`` of a docstring template"""
    for k, v in (('cs', CS_PARAM), ('F', FINPUT_PARAM), ('shell', SHELL_PARAM),
                 ('common', PARAMS_COMMON),
                 ('returns', RETURNS.format(name=kname)), ('raises', RAISES)):
        doc = doc.replace('<<%s>>' % k, v)
    return doc


def kernel(name, signature, doc, cdefs, setup, extra, kform, out='k'):
    return '''
def {name}({signature}):
    r"""{doc}
    """
{common}{cdefs}
    cdef long [::1] kr, kc
    cdef double [::1] kv
{setup}{common_setup}
    fdim = {dof2}*m*m*n*n
    kr = np.zeros((fdim,), dtype=INT)
    kc = np.zeros((fdim,), dtype=INT)
    kv = np.zeros((fdim,), dtype=DOUBLE)

    with nogil:
{point}
{kform}
{assembly}
    return coo_matrix((kv, (kr, kc)), shape=(size, size))
'''.format(name=name, signature=signature, doc=doc, common=COMMON_CDEF,
           cdefs=cdefs, setup=setup, common_setup=SETUP, dof2=DOF*DOF,
           point=POINT.format(extra=extra), kform=kform, assembly=ASSEMBLY)


Bm = B['B']
H0 = B['H0']

#: declarations of the strains and stresses of the kernels
CDEF_EPS = '    cdef double %s\n' % ', '.join('eps%d' % r for r in range(NE))
CDEF_EPSL = '    cdef double %s\n' % ', '.join('epsL%d' % r for r in range(NE))
CDEF_SIG = '    cdef double %s\n' % ', '.join(SIG_NAMES)
CDEF_SIGL = '    cdef double %s\n' % ', '.join(n + 'L' for n in SIG_NAMES)

FKC_DOC = r"""Constitutive stiffness matrix, integrated numerically

    With ``NLgeom=0``, the linear constitutive stiffness matrix ``KC0``;
    with ``NLgeom=1``, the matrix ``KC0 + KCNL + KGNL`` at the configuration
    ``cs``, which, added to :func:`fkG_num` at the same ``cs``, gives the
    tangent stiffness matrix.

    Parameters
    ----------
<<cs>>
<<F>>
<<shell>>
<<common>>
    NLgeom : int, optional
        ``1`` to include the non-linear terms of ``cs``, default ``0``.

<<returns>>

<<raises>>

    Notes
    -----
    With the generalized derivatives `\boldsymbol{d} = \boldsymbol{\Phi}
    \boldsymbol{c}` of the module, `\boldsymbol{B} = \partial
    \boldsymbol{\varepsilon}/\partial \boldsymbol{d}` at `\boldsymbol{d}`,
    the stress resultants `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{\sigma}_L` of the linear
    strains:

    .. math::

        \boldsymbol{K}_C = \int \boldsymbol{\Phi}^\top \left(
        \boldsymbol{B}^\top \boldsymbol{F} \boldsymbol{B} +
        \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{d}) -
        \boldsymbol{H}(\boldsymbol{\sigma}_L, \boldsymbol{0}) \right)
        \boldsymbol{\Phi} \, dA, \qquad
        \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{d}) = \sum_r
        \sigma_r \frac{\partial^2 \varepsilon_r}{\partial \boldsymbol{d}
        \partial \boldsymbol{d}}

    with `dA = (\rho/r)\, dx\, dy`, see the notes of the module. With
    ``NLgeom=0``, `\boldsymbol{d} = \boldsymbol{0}` and the matrix is
    `\int \boldsymbol{\Phi}^\top \boldsymbol{B}_0^\top \boldsymbol{F}
    \boldsymbol{B}_0 \boldsymbol{\Phi} \, dA`.
"""
FKC_DOC = fill(FKC_DOC, 'kC')

FKC_FORM = """                for p in range({nd6}):
                    Bm[p] = 0.
                for p in range({nd}):
                    dv[p] = 0.
                if NLgeom == 1:
{dv}
{Bm}
                # K = [B]^T [F] [B]
                for p in range({ne}):
                    for q in range({nd}):
                        FB[p*{nd} + q] = 0.
                        for r in range({ne}):
                            FB[p*{nd} + q] += F[p*{ne} + r]*Bm[r*{nd} + q]
                for p in range({nd}):
                    for q in range({nd}):
                        K[p*{nd} + q] = 0.
                        for r in range({ne}):
                            K[p*{nd} + q] += Bm[r*{nd} + p]*FB[r*{nd} + q]
                if NLgeom == 1:
                    # + [H(sigma, d)] - [H(sigma_L, 0)]
{eps}
{epsL}
{sig}
{sigL}
{H}
{H0}""".format(nd=ND, nd6=NE*ND, ne=NE, dv=dv_lines(20, 'cs'),
               Bm=matrix_assign('Bm', Bm, 16, ND),
               eps=strain_lines(20, B['eps']),
               epsL=strain_lines(20, B['epsL'], 'epsL'),
               sig=stress_lines(20, [s.name for s in SIG]),
               sigL=stress_lines(20, [s.name for s in SIGL], 'epsL'),
               H=H_lines(20, B['H'], SIG),
               H0=H_lines(20, H0, SIGL, '-='))

FKC = kernel('fkC_num',
             'double [::1] cs, object Finput, object shell,\n'
             '        int size, int row0, int col0, int nx, int ny, int NLgeom=0',
             FKC_DOC,
             '    cdef double Bm[%d]\n    cdef double FB[%d]\n'
             % (NE*ND, NE*ND) + CDEF_EPS + CDEF_EPSL + CDEF_SIG + CDEF_SIGL,
             FINPUT, F_POINT, FKC_FORM)

FKG_DOC = r"""Geometric stiffness matrix of the linear stress state

    Integrates numerically the geometric stiffness matrix of the stress
    resultants of the linear strains at ``cs``, superposed to the constant
    membrane state ``Nxx0``, ``Nyy0``, ``Nxy0``. The matrix is homogeneous
    of degree one in ``cs``, as the linear buckling eigenvalue problem
    requires; the remaining terms of the tangent stiffness enter
    :func:`fkC_num`.

    Parameters
    ----------
<<cs>>
<<F>>
<<shell>>
<<common>>
    Nxx0, Nyy0, Nxy0 : float, optional
        Constant membrane stress resultants along the local axes, `N_{xx}
        = N_{\rho\rho}`, `N_{yy} = N_{\theta\theta}` and `N_{xy} =
        -N_{\rho\theta}`, force per unit length, default ``0``.

<<returns>>

<<raises>>

    Notes
    -----
    With `\boldsymbol{\sigma} = \boldsymbol{\sigma}_L +
    \boldsymbol{\sigma}_0`, `\boldsymbol{\sigma}_L` the stress resultants
    of the linear strains of ``cs`` and `\boldsymbol{\sigma}_0 = \{N_{xx0},
    N_{yy0}, N_{xy0}, 0, 0, 0, 0, 0\}^\top`:

    .. math::

        \boldsymbol{K}_G = \int \boldsymbol{\Phi}^\top
        \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{0})
        \boldsymbol{\Phi} \, dA

    with `\boldsymbol{H}` of :func:`fkC_num`, the variation of the work of
    the stresses on the second-order terms of the strains, see the notes of
    the module.
"""
FKG_DOC = fill(FKG_DOC, 'kG')

FKG_FORM = """{dv}
{epsL}
{sigL}
                NxxL = NxxL + Nxx0
                NyyL = NyyL + Nyy0
                NxyL = NxyL + Nxy0
                for p in range({nd2}):
                    K[p] = 0.
{H}""".format(dv=dv_lines(16, 'cs'), epsL=strain_lines(16, B['epsL'], 'epsL'),
              sigL=stress_lines(16, [s.name for s in SIGL], 'epsL'),
              nd2=ND*ND, H=H_lines(16, H0, SIGL))

FKG = kernel('fkG_num',
             'double [::1] cs, object Finput, object shell,\n'
             '        int size, int row0, int col0, int nx, int ny,\n'
             '        double Nxx0=0, double Nyy0=0, double Nxy0=0',
             FKG_DOC, CDEF_EPSL + CDEF_SIGL, FINPUT, F_POINT, FKG_FORM)

J = B['J']
var_I = sympy.symbols('I0, I1, I2')
MAUX = sympy.Matrix([[var_I[0], 0, 0, var_I[1], 0],
                     [0, var_I[0], 0, 0, var_I[1]],
                     [0, 0, var_I[0], 0, 0],
                     [var_I[1], 0, 0, var_I[2], 0],
                     [0, var_I[1], 0, 0, var_I[2]]])
KM = J.T*MAUX*J

FKM_DOC = r"""Mass matrix, integrated numerically

    Integrates numerically the consistent mass matrix of the panel, with
    the moments of the density given for the whole domain or at each
    integration point.

    Parameters
    ----------
<<shell>>
    I_input : array-like
        Moments of the density through the thickness, ``I0, I1, I2``, shape
        ``(3,)`` for the whole domain or ``(nx, ny, 3)`` with one set per
        integration point.
<<common>>

    Returns
    -------
    kM : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``. Only the upper triangle is
        assembled; the caller symmetrizes it.

    Raises
    ------
    ValueError
        If ``I_input`` has an invalid shape, ``shell`` is not a
        :class:`.Shell` or its outer radius ``r`` is not larger than its
        radial width ``a``.

    Notes
    -----
    With the displacement through the thickness of the first-order shear
    deformation theory, `u(z) = u + z \phi_x`, `v(z) = v + z \phi_y`, `w(z)
    = w`, `z` measured from the reference plane, and the moments `I_n = \int
    \rho_m z^n dz`, `n = 0, 1, 2`, of the density `\rho_m` over `d - h/2
    \le z \le d + h/2`, with the offset `d` (``shell.offset``):

    .. math::

        \boldsymbol{M} = \int \boldsymbol{\Phi}^\top \boldsymbol{J}^\top
        \begin{bmatrix} I_0 \boldsymbol{I}_3 & I_1 \boldsymbol{E} \\
        I_1 \boldsymbol{E}^\top & I_2 \boldsymbol{I}_2 \end{bmatrix}
        \boldsymbol{J} \boldsymbol{\Phi} \, dA, \qquad
        \boldsymbol{J} \boldsymbol{d} = \{u, v, w, \phi_x,
        \phi_y\}^\top

    with `\boldsymbol{E}` the first two columns of `\boldsymbol{I}_3` and
    `dA = (\rho/r)\, dx\, dy`.
"""
FKM_DOC = fill(FKM_DOC, None)

FKM_SETUP = """    cdef double I0, I1, I2
    cdef double [:, :, ::1] Inxny
    cdef int one_I_each_point = 0
    I_input = np.ascontiguousarray(I_input, dtype=DOUBLE)
    if I_input.shape == (nx, ny, 3):
        Inxny = I_input
        one_I_each_point = 1
    elif I_input.shape == (3,):
        Inxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        I0 = I_input[0]
        I1 = I_input[1]
        I2 = I_input[2]
    else:
        raise ValueError('Invalid shape for I_input!')
"""

FKM_EXTRA = """                if one_I_each_point == 1:
                    I0 = Inxny[ptx, pty, 0]
                    I1 = Inxny[ptx, pty, 1]
                    I2 = Inxny[ptx, pty, 2]"""

FKM_FORM = """                for p in range({nd2}):
                    K[p] = 0.
{K}""".format(nd2=ND*ND, K=matrix_assign('K', KM, 16, ND))

FKM = kernel('fkM_num',
             'object shell, object I_input, int size,\n'
             '        int row0, int col0, int nx, int ny',
             FKM_DOC, FKM_SETUP, '', FKM_EXTRA, FKM_FORM)

FINT_DOC = r"""Internal force vector, integrated numerically

    Integrates numerically the internal force vector of the configuration
    ``cs``, the gradient of the strain energy with respect to the Ritz
    constants.

    Parameters
    ----------
<<cs>>
<<F>>
<<shell>>
    size : int
        Size of the vector, the number of degrees of freedom of the
        assembly.
    col0 : int
        Entry of the vector where the terms of this panel start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    fint : np.ndarray
        Internal force vector, 1-D of size ``size``.

<<raises>>

    Notes
    -----
    With the strains `\boldsymbol{\varepsilon}(\boldsymbol{d})` of the
    module, `\boldsymbol{\sigma} = \boldsymbol{F}
    \boldsymbol{\varepsilon}` and `\boldsymbol{B} = \partial
    \boldsymbol{\varepsilon}/\partial \boldsymbol{d}`:

    .. math::

        \boldsymbol{f}_{int} = \int \boldsymbol{\Phi}^\top
        \boldsymbol{B}^\top \boldsymbol{\sigma} \, dA
"""
FINT_DOC = fill(FINT_DOC, None)

FINT = '''
def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    r"""{doc}
    """
{common}    cdef double Bm[{nb}]
    cdef double bs[{nd}]
{cdefs}    cdef double [::1] fint
    cdef int row0 = col0
{finput}{setup}
    fint = np.zeros(size, dtype=DOUBLE)

    with nogil:
{point}
{dv}
{eps}
{sig}
{Bm}
                for p in range({nd}):
                    bs[p] = ({bsig})
                for j in range(n):
                    for i in range(m):
                        row = row0 + {dof}*(j*m + i)
{phiA}
                        for e in range({dof}):
                            for p in range(P0[e], P1[e]):
                                fint[row + e] += weight*pA[p]*bs[p]
    return np.asarray(fint)
'''.format(doc=FINT_DOC, common=COMMON_CDEF, nb=NE*ND, nd=ND,
           cdefs=CDEF_EPS + CDEF_SIG,
           bsig='\n                             + '.join(
               'Bm[%d + p]*%s' % (r*ND, n) for r, n in enumerate(SIG_NAMES)),
           finput=FINPUT, setup=SETUP,
           point=POINT.format(extra=F_POINT), dv=dv_lines(16, 'cs'),
           eps=strain_lines(16, B['eps']),
           sig=stress_lines(16, SIG_NAMES),
           Bm='\n'.join(['                for p in range(%d):' % (NE*ND),
                         '                    Bm[p] = 0.',
                         matrix_assign('Bm', Bm, 16, ND)]),
           dof=DOF, phiA=phi_lines('A', 'i', 'j', 24))

KTV_DOC = r"""Product of the tangent stiffness matrix by a vector, without assembly

    Returns ``KT(cs) vs``, with ``KT`` the tangent stiffness matrix
    ``fkC_num(cs, ..., NLgeom=1) + fkG_num(cs, ...)`` at ``cs``, the exact
    Jacobian of :func:`calc_fint`, plus the geometric stiffness matrix of
    the constant membrane stress state ``Nxx_cte``, ``Nyy_cte`` and
    ``Nxy_cte``.

    Parameters
    ----------
    cs, vs : array-like
        Ritz constants of the configuration and the vector, 1-D of size
        ``size``.
<<F>>
<<shell>>
    size, col0 : int
        Size of the vectors and position where the terms of this panel
        start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.
    Nxx_cte, Nyy_cte, Nxy_cte : float, optional
        Constant membrane stress resultants along the local axes, force per
        unit length, default ``0``.

    Returns
    -------
    kTv : numpy.ndarray
        The product, of size ``size``.

<<raises>>

    Notes
    -----
    At each integration point, with `\boldsymbol{B} = \partial
    \boldsymbol{\varepsilon}/\partial \boldsymbol{d}` at `\boldsymbol{d} =
    \boldsymbol{\Phi} \boldsymbol{c}` and the stress resultants
    `\boldsymbol{\sigma}` of `\boldsymbol{F} \boldsymbol{\varepsilon}`,
    plus the constant membrane ones `\boldsymbol{\sigma}_0`:

    .. math::

        \boldsymbol{K}_T \boldsymbol{v} = \int \boldsymbol{\Phi}^\top \left(
        \boldsymbol{B}^\top \boldsymbol{F} \boldsymbol{B} +
        \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{d}) +
        \boldsymbol{H}(\boldsymbol{\sigma}_0, \boldsymbol{0}) \right)
        \boldsymbol{\Phi} \boldsymbol{v} \, dA

    evaluated without forming `\boldsymbol{K}_T`, with
    `\boldsymbol{H}` of :func:`fkC_num`.
"""
KTV_DOC = fill(KTV_DOC, None)


def _Hq_lines(indent):
    t = ' '*indent
    Hs = (B['H'].subs(DV, simultaneous=True)
          + H0.subs(dict(zip(th.STRESSES, [Symbol('Nxx_cte'), Symbol('Nyy_cte'),
                                           Symbol('Nxy_cte')] + [0]*(NE - 3))),
                    simultaneous=True))
    out = []
    for i in range(ND):
        e = sum(Hs[i, j]*Symbol('dq[%d]' % j) for j in range(ND))
        if e != 0:
            out.append('%sbs[%d] += %s' % (t, i, pstr(e)))
    return '\n'.join(out)


KTV = '''
def calc_kTv(double [::1] cs, double [::1] vs, object Finput, object shell,
        int size, int col0, int nx, int ny, double Nxx_cte=0.,
        double Nyy_cte=0., double Nxy_cte=0.):
    r"""{doc}
    """
{common}    cdef double Bm[{nb}]
    cdef double bs[{nd}]
    cdef double dq[{nd}]
    cdef double Bq[{ne}]
    cdef double FBq[{ne}]
{cdefs}    cdef double [::1] kTv
    cdef int row0 = col0
{finput}    if cs.shape[0] != size or vs.shape[0] != size:
        raise ValueError('cs and vs must have the size size')
{setup}
    kTv = np.zeros(size, dtype=DOUBLE)

    with nogil:
{point}
{dq}
{dv}
{eps}
{sig}
{Bm}
                for r in range({ne}):
                    Bq[r] = 0.
                    for p in range({nd}):
                        Bq[r] += Bm[r*{nd} + p]*dq[p]
                for r in range({ne}):
                    FBq[r] = 0.
                    for q in range({ne}):
                        FBq[r] += F[r*{ne} + q]*Bq[q]
                for p in range({nd}):
                    bs[p] = 0.
                    for r in range({ne}):
                        bs[p] += Bm[r*{nd} + p]*FBq[r]
{Hq}
                for j in range(n):
                    for i in range(m):
                        row = row0 + {dof}*(j*m + i)
{phiA}
                        for e in range({dof}):
                            for p in range(P0[e], P1[e]):
                                kTv[row + e] += weight*pA[p]*bs[p]
    return np.asarray(kTv)
'''.format(doc=KTV_DOC, common=COMMON_CDEF, nb=NE*ND, nd=ND, ne=NE,
           cdefs=CDEF_EPS + CDEF_SIG, finput=FINPUT,
           setup=SETUP, point=POINT.format(extra=F_POINT),
           dq=dv_lines(16, 'vs').replace('dv[', 'dq['), dv=dv_lines(16, 'cs'),
           eps=strain_lines(16, B['eps']),
           sig=stress_lines(16, SIG_NAMES),
           Bm='\n'.join(['                for p in range(%d):' % (NE*ND),
                         '                    Bm[p] = 0.',
                         matrix_assign('Bm', Bm, 16, ND)]),
           dof=DOF, Hq=_Hq_lines(16), phiA=phi_lines('A', 'i', 'j', 24))

FIELD_PARAMS = """    c : np.ndarray
        Ritz constants of the panel ``s``, 1-D array of size ``5*s.m*s.n``,
        see :meth:`.Shell.get_size`.
    s : :class:`.Shell`
        The annular sector, whose dimensions, outer radius, numbers of terms
        and boundary flags are read.
    xs, ys : np.ndarray
        Coordinates `x` and `y` of the points, lengths, 1-D arrays of the
        same size.
    num_cores : int, optional
        Not used, the points are evaluated serially; kept for the interface
        of :mod:`panels.models.fsdt_tsdt_field`."""


def _fstrain_lines(indent):
    t = ' '*indent
    out = []
    for r in range(NE):
        lin = pstr(B['epsL'][r].subs(DV, simultaneous=True), 's_')
        nl = B['eps'][r] - B['epsL'][r]
        line = '%sout[%d, k] = %s' % (t, r, lin)
        if sympy.expand(nl) != 0:
            line += ' + NLgeom*(%s)' % pstr(nl.subs(DV, simultaneous=True), 's_')
        out.append(line)
    return '\n'.join(out)


FIELDS = '''
def _point_derivatives(double [::1] c, object shell, double xp, double yp):
    r"""Generalized derivatives `\\boldsymbol{{d}}` of the panel at a point

    Parameters
    ----------
    c : np.ndarray
        Ritz constants of the panel, 1-D array of size ``5*m*n``.
    shell : :class:`.Shell`
        The annular sector.
    xp, yp : float
        Coordinates `x` and `y` of the point, lengths.

    Returns
    -------
    d : np.ndarray
        ``({dnames})``, 1-D array of size {nd}.

    Notes
    -----
    `\\boldsymbol{{d}} = \\boldsymbol{{\\Phi}}(x, y) \\boldsymbol{{c}}`,
    with the products of the approximation functions of Bardell along `x`
    and `y` and of their derivatives, see the notes of the module.

    """
    cdef int m, n, i, j, col, p, col0 = 0
    cdef double a, b, xi, eta
{bc}
    cdef double pA[{nd}]
    cdef double dv[{nd}]
    cdef double [:, ::1] ftab, gtab
    a = shell.a
    b = shell.b
    m = shell.m
    n = shell.n
{bcread}
    xi = 2*xp/a - 1
    eta = 2*yp/b - 1
    ftab = np.zeros(({nt}, m), dtype=DOUBLE)
    gtab = np.zeros(({nt}, n), dtype=DOUBLE)
{tabs}
{dv}
    return np.array([dv[p] for p in range({nd})])


def fuvw(double [::1] c, object s, double [::1] xs, double [::1] ys,
         int num_cores=4):
    r"""Displacements and rotations of the panel at the points ``xs``, ``ys``

    Evaluates the displacements and the rotations of the normal of the
    configuration ``c`` at each point, serially.

    Parameters
    ----------
{params}

    Returns
    -------
    us, vs, ws, phixs, phiys : np.ndarray
        The displacements ``u``, ``v``, ``w`` along the local axes, lengths,
        and the rotations of the normal ``phix`` and ``phiy``, the
        derivatives of ``u`` and ``v`` along ``z``, dimensionless, 1-D arrays
        of the size of ``xs``.

    """
    out = np.zeros((5, xs.shape[0]), dtype=DOUBLE)
    for k in range(xs.shape[0]):
        d = _point_derivatives(c, s, xs[k], ys[k])
        out[:, k] = (d[{iu}], d[{iv}], d[{iw}], d[{iphix}], d[{iphiy}])
    return tuple(out)


def fstrain(double [::1] c, object s, double [::1] xs, double [::1] ys,
            int num_cores=4, int NLgeom=0):
    r"""Generalized strains of the panel at the points ``xs``, ``ys``

    Evaluates the eight generalized strains of the configuration ``c`` at
    each point, serially.

    Parameters
    ----------
{params}
    NLgeom : int, optional
        ``1`` to include the non-linear terms of the strains, ``0``
        otherwise. Default ``0``.

    Returns
    -------
    exxs, eyys, gxys, kxxs, kyys, kxys, gyzs, gxzs : np.ndarray
        The membrane strains, dimensionless, the changes of curvature, per
        unit length, and the transverse shear strains, dimensionless, along
        the local axes, 1-D arrays of the size of ``xs``, in the order of
        :func:`strain_names`.

    Notes
    -----
    The strains given in the notes of the module, with the non-linear terms
    multiplied by ``NLgeom``.

    """
    cdef double R, s_, irho
    R = s.r
    out = np.zeros(({ne}, xs.shape[0]), dtype=DOUBLE)
    for k in range(xs.shape[0]):
        dv = _point_derivatives(c, s, xs[k], ys[k])
        irho = 1./(R - xs[k])
        s_ = R*irho
{fstrain}
    return tuple(out)


def strain_names(object s):
    r"""Names of the generalized strains returned by :func:`fstrain`

    The names are those of the model, the same for every panel.

    Parameters
    ----------
    s : :class:`.Shell`
        The annular sector, not read: the names are those of the model.

    Returns
    -------
    names : tuple of str
        ``{epsnames}``, in the order of the rows and columns of
        ``Shell.ABD``.

    """
    return {epstuple}


def stress_names(object s):
    r"""Names of the generalized stresses, conjugate to :func:`strain_names`

    The names are those of the model, the same for every panel.

    Parameters
    ----------
    s : :class:`.Shell`
        The annular sector, not read: the names are those of the model.

    Returns
    -------
    names : tuple of str
        ``{signames}``, the products of ``Shell.ABD`` by the strains of
        :func:`strain_names`, used as keys of ``Shell.fields`` by
        :meth:`.Shell.stress`.

    """
    return {sigtuple}
'''.format(bc=BC_CDEF, bcread=BC_READ, nd=ND, tabs=TABS(4), nt=3*NF, ne=NE,
           dnames=', '.join(d.name for d in th.D),
           dv=dv_lines(4, 'c'), params=FIELD_PARAMS,
           iu=IDX['u'], iv=IDX['v'], iw=IDX['w'], iphix=IDX['phix'],
           iphiy=IDX['phiy'], fstrain=_fstrain_lines(8),
           epsnames=', '.join(EPS_NAMES), signames=', '.join(SIG_NAMES),
           epstuple=repr(EPS_NAMES), sigtuple=repr(SIG_NAMES))

NOT_IMPLEMENTED = '''
def {name}(*args, **kwargs):
    r"""Not available for the annular sector

    The function exists for the interface of the models with the piston
    theory, see :meth:`.Shell.calc_kA`.

    Parameters
    ----------
    *args : object, optional
        Positional arguments, ignored.
    **kwargs : object, optional
        Keyword arguments, ignored.

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('{name} is not implemented for the model '
                              '"{model}"')
'''


def numerical_module():
    head = (HEADER + 'r"""\n' + th.TITLE + '\n\nNumerically integrated kernels '
            'of the model ``\'' + MODEL + '\'``:\nthe constitutive stiffness '
            ':func:`fkC_num`, the geometric stiffness\n:func:`fkG_num`, the '
            'mass :func:`fkM_num`, the internal force vector\n:func:`calc_fint`, '
            'the product of the tangent stiffness by a vector\n:func:`calc_kTv`, '
            'and the fields :func:`fuvw`, :func:`fstrain` and\n:func:`fg`, the '
            'last one that of :mod:`panels.models.fsdt_tsdt_field`. The\ndegree of '
            'freedom ``k`` of the term ``(i, j)``, ``0 <= i < m``, ``0 <= j <\n'
            'n``, is at the row ``row0 + 5*(j*m + i) + k`` of the matrices.\n'
            + MODULE_NOTES + '\n"""\n')
    imports = '''from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
from .fsdt_tsdt_field import fg


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = %d
cdef int[%d] P0 = [%s]
cdef int[%d] P1 = [%s]
#: degree of freedom of each generalized derivative
cdef int[%d] FLD = [%s]

''' % (DOF, DOF, ', '.join(map(str, P0)), DOF, ', '.join(map(str, P1)), ND,
       ', '.join(map(str, FLD)))
    na = ''.join(NOT_IMPLEMENTED.format(name=nm, model=MODEL)
                 for nm in ('fkAx_num', 'fkAy_num'))
    return head + imports + FKC + FKG + FKM + FINT + KTV + FIELDS + na


ANALYTICAL = HEADER + '''r"""
{title}

Matrices of the whole domain of the model ``'{model}'``,
called by :class:`.Shell` when the constitutive matrix and the density are
constant: :func:`fk0`, :func:`fkG0` and :func:`fkM`. The strains have
coefficients in `1/\\rho`, with `\\rho = r - x` the radius of the point, and
the matrices are integrated with the Gauss-Legendre rule of
:mod:`panels.models.{model}_num`, with ``shell.nx`` and
``shell.ny`` points; the piston-theory matrices ``fkAx``, ``fkAy`` and
``fcA`` are not available.

Notes
-----
See :mod:`panels.models.{model}_num`.

"""
import numpy as np

from panels import DOUBLE
from .{model}_num import fkC_num, fkG_num, fkM_num


def fk0(object shell, int size, int row0, int col0):
    r"""Linear constitutive stiffness matrix of the whole panel

    Integrates numerically the linear constitutive stiffness matrix over
    the whole domain, with ``shell.nx`` x ``shell.ny`` Gauss-Legendre
    points.

    Parameters
    ----------
    shell : :class:`.Shell`
        The annular sector, with its constitutive matrix ``shell.ABD`` of the
        first-order shear deformation theory and its numbers of integration
        points ``shell.nx`` and ``shell.ny``.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the terms of this panel start.

    Returns
    -------
    kC0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``, upper triangle.

    Notes
    -----
    :func:`.{model}_num.fkC_num` with ``NLgeom=0``.

    """
    return fkC_num(np.zeros(size, dtype=DOUBLE), shell.ABD, shell, size,
                   row0, col0, shell.nx, shell.ny, 0)


def fkG0(double Nxx, double Nyy, double Nxy, object shell,
         int size, int row0, int col0):
    r"""Geometric stiffness matrix of a constant membrane stress state

    Integrates numerically, over the whole domain, the geometric stiffness
    matrix of the membrane stress resultants ``Nxx``, ``Nyy``, ``Nxy``.

    Parameters
    ----------
    Nxx, Nyy, Nxy : float
        Membrane stress resultants along the local axes, `N_{{xx}} =
        N_{{\\rho\\rho}}`, `N_{{yy}} = N_{{\\theta\\theta}}`, `N_{{xy}} =
        -N_{{\\rho\\theta}}`, force per unit length.
    shell : :class:`.Shell`
        The annular sector, with its numbers of integration points
        ``shell.nx`` and ``shell.ny``.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the terms of this panel start.

    Returns
    -------
    kG0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``, upper triangle.

    Notes
    -----
    :func:`.{model}_num.fkG_num` with zero Ritz constants.

    """
    return fkG_num(np.zeros(size, dtype=DOUBLE), shell.ABD, shell, size,
                   row0, col0, shell.nx, shell.ny, Nxx, Nyy, Nxy)


def fkM(object shell, int size, int row0, int col0):
    r"""Mass matrix of the whole panel

    Integrates numerically the consistent mass matrix over the whole
    domain, with the moments of the density of the laminate.

    Parameters
    ----------
    shell : :class:`.Shell`
        The annular sector, with its laminate ``shell.lam`` and its numbers
        of integration points ``shell.nx`` and ``shell.ny``.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the terms of this panel start.

    Returns
    -------
    kM : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``, upper triangle.

    Notes
    -----
    :func:`.{model}_num.fkM_num` with the moments
    ``intrho``, ``intrhoz``, ``intrhoz2`` of ``shell.lam``, which include
    the offset ``shell.offset``.

    """
    lam = shell.lam
    I_input = np.array([lam.intrho, lam.intrhoz, lam.intrhoz2], dtype=DOUBLE)
    return fkM_num(shell, I_input, size, row0, col0, shell.nx, shell.ny)
'''.format(title=th.TITLE, model=MODEL) + ''.join(
    NOT_IMPLEMENTED.format(name=nm, model=MODEL) for nm in ('fkAx', 'fkAy', 'fcA'))


def generate(outdir=None):
    outdir = outdir or os.path.join(REPO, 'panels', 'models')
    for name, src in ((MODEL + '_num.pyx', numerical_module()),
                      (MODEL + '.pyx', ANALYTICAL)):
        with open(os.path.join(outdir, name), 'w', encoding='utf-8',
                  newline='\n') as fh:
            fh.write(src)
        print('written', os.path.join(outdir, name))


if __name__ == '__main__':
    generate(sys.argv[1] if len(sys.argv) > 1 else None)
