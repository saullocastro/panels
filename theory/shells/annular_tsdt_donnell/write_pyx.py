r"""Generate the kernels of an annular-sector TSDT model of this folder

Writes ``panels/models/<model>_num.pyx`` (numerically integrated matrices,
internal force vector and strain field) and ``panels/models/<model>.pyx``
(the matrices of the whole domain called by :class:`.Shell` for the default
integration) from the derivation of ``<model>.py``, ``<model>`` being the
name of this folder::

    python write_pyx.py [output folder]

Kernel layout
-------------
As the writer of the ``annular_clpt_*`` folders, for the five degrees of
freedom ``u, v, w, phix, phiy`` and the thirteen strains of the TSDT. At
each integration point the generalized derivatives ``d = [Phi] c``
(``<model>.D``), the NE x ND matrix ``[B] = d eps/d d`` at ``d`` and the ND
x ND matrix ``K`` of the kernel (``[B]^T [F] [B] + [H(sigma, d)] -
[H(sigma_L, 0)]`` for ``fkC_num``, ``[H(sigma_L + Nbar, 0)]`` for
``fkG_num``, the mass integrand ``KM`` of ``<model>.py`` for ``fkM_num``)
are evaluated once, with ``[H(sigma, d)] = sum_r sigma_r d^2 eps_r/d d^2``.
The entry of the pair of terms ``A = (i, j)``, ``B = (k, l)`` and the
degrees of freedom ``e``, ``f`` is ``sum_pq Phi_A[p, e] K[p, q] Phi_B[q,
f]``, computed as ``TA[e, q] = sum_p Phi_A[p, e] K[p, q]`` in the loop over
the terms ``A`` and ``sum_q TA[e, q] Phi_B[q, f]`` in the loop over the
terms ``B``. The coefficients of the strains depend on the radius ``rho = R
- x`` of the point: the integrands are rational functions of ``x``,
integrated with the Gauss-Legendre rule. The TSDT constant ``c1 = 4/(3
h^2)`` comes from the thickness ``shell.lam.h`` (stiffness) or from the
thickness given with the moments of the density (mass).

The script is self-contained and the same in every folder
``annular_tsdt_*``: the model, its strains and the text describing them
come from ``<model>.py``.

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
NH = th.NH
#: field index of each generalized derivative
FLD = [th.FIELDS.index(th.D_OF[dd.name][0]) for dd in th.D]
assert FLD == sorted(FLD), 'the generalized derivatives must be grouped by field'
#: first and last + 1 generalized derivative of each degree of freedom
P0 = tuple(FLD.index(e) for e in range(DOF))
P1 = tuple(len(FLD) - FLD[::-1].index(e) for e in range(DOF))
#: index of a generalized derivative by name
IDX = {dd.name: k for k, dd in enumerate(th.D)}

S_ = Symbol('s')
IRHO = Symbol('irho')


class _Printer(StrPrinter):
    def _print_Rational(self, e):
        return repr(float(e))

    def _print_Integer(self, e):
        return '%d.' % int(e)

    def _print_Pow(self, e, rational=False):
        b, p = e.as_base_exp()
        if p.is_Integer and 0 < p <= 6:
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
DV = {dd: Symbol('dv[%d]' % k) for k, dd in enumerate(th.D)}
#: the stresses of the configuration and of its linear strains
SIG = [Symbol('sig[%d]' % k) for k in range(NE)]
SIGL = [Symbol('sigL[%d]' % k) for k in range(NE)]

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
\boldsymbol{e}_y`, and ``phix``, ``phiy`` the rotations of the normal at
the mid-surface of the laminate, `\phi_x = u_{,z}` and `\phi_y = v_{,z}`
there. These are the local axes of the flange of the connection
``'BFxcte'`` whose base is a cylinder: the web of a ring frame of a
stiffened cylinder is an application, with ``a`` the height of the web,
``b`` the arc length of its outer edge and ``r`` its outer radius. The
laminate spans `d - h/2 \le z \le d + h/2` about the reference plane, `d`
being the offset ``shell.offset``, which enters through the constitutive
matrix ``shell.ABD`` of the TSDT and the mass matrix.

Notes
-----
""" + th.KINEMATICS_NOTES + '\n' + th.STRAIN_NOTES + r"""
The element of area is `dA = (\rho/r)\, dx\, dy`. The integrands are
rational functions of `x`, integrated with the Gauss-Legendre rule over the
domain of :meth:`.Shell.integration_limits`.

References
----------
""" + th.REFERENCES

SHELL_PARAM = """    shell : :class:`.Shell`
        The annular sector, which gives the radial width ``a``, the arc
        length ``b`` of the outer edge, the outer radius ``r``, the
        thickness ``lam.h``, the numbers of terms ``m`` and ``n`` and the
        boundary flags of the approximation functions of each field."""

#: boundary flags of the approximation functions, one group per edge
#: direction and field, e.g. ``x1phix, x1phixr, x2phix, x2phixr``
BC_GROUPS = [['%s%s%s%s' % (xy, k, f, r) for k in '12' for r in ('', 'r')]
             for xy in 'xy' for f in th.FIELDS]
BC_CDEF = '\n'.join('    cdef double ' + ', '.join(g) for g in BC_GROUPS)
BC_READ = '\n'.join('    ' + '; '.join('%s = shell.%s' % (n, n) for n in g)
                    for g in BC_GROUPS)

COMMON_CDEF = """    cdef double x1, x2, y1, y2, xi1, xi2, eta1, eta2
    cdef double a, b, R, h, c1, intx, inty, xi, eta, xp, rho, s, irho, weight
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
    h = shell.lam.h
    c1 = 4./(3.*h*h)
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
""".format(bc=BC_READ, nt=3*DOF)


def tab_lines(indent, tab, k, coord, L, s1, s2):
    """Rows ``3*field + order`` of the table of the functions of ``coord``"""
    t = ' '*indent
    out = []
    for fi, fld in enumerate(th.FIELDS):
        flags = '%s1%s, %s1%sr, %s2%s, %s2%sr' % (s1, fld, s1, fld, s1, fld,
                                                 s1, fld)
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
    for k, dd in enumerate(th.D):
        fld, ox, oy = th.D_OF[dd.name]
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


def strain_lines(indent, eps, name='eps'):
    t = ' '*indent
    return '\n'.join('%s%s[%d] = %s' % (t, name, r,
                                        pstr(eps[r].subs(DV, simultaneous=True)))
                     for r in range(NE))


def stress_lines(indent, sig='sig', eps='eps', rows=range(NE)):
    """``sig[r] = sum_q F[r, q] eps[q]`` for the first ``len(rows)`` rows"""
    t = ' '*indent
    return '\n'.join([
        t + 'for r in range(%d):' % len(rows),
        t + '    %s[r] = 0.' % sig,
        t + '    for q in range(%d):' % NE,
        t + '        %s[r] += F[r*%d + q]*%s[q]' % (sig, NE, eps)])


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
        Constitutive matrix of the TSDT, shape ``({ne}, {ne})`` for the
        whole domain, e.g. ``shell.ABD``, or ``(nx, ny, {ne}, {ne})`` with
        one matrix per integration point, rows and columns in the order of
        the strains of the module.""".format(ne=NE)

CS_PARAM = """    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements and rotations."""

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


def kernel(name, signature, doc, cdefs, setup, extra, kform):
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
STRAIN_CDEF = ('    cdef double eps[{ne}]\n    cdef double epsL[{ne}]\n'
               '    cdef double sig[{ne}]\n    cdef double sigL[{ne}]\n'
               ).format(ne=NE)

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

    with `dA = (\rho/r)\, dx\, dy`, see the notes of the module. Only the
    membrane stresses enter `\boldsymbol{H}`. With ``NLgeom=0``,
    `\boldsymbol{d} = \boldsymbol{0}` and the matrix is `\int
    \boldsymbol{\Phi}^\top \boldsymbol{B}_0^\top \boldsymbol{F}
    \boldsymbol{B}_0 \boldsymbol{\Phi} \, dA`.
"""
FKC_DOC = fill(FKC_DOC, 'kC')

FKC_FORM = """                for p in range({nbm}):
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
{H0}""".format(nd=ND, ne=NE, nbm=NE*ND, dv=dv_lines(20, 'cs'),
               Bm=matrix_assign('Bm', Bm, 16, ND),
               eps=strain_lines(20, B['eps']),
               epsL=strain_lines(20, B['epsL'], 'epsL'),
               sig=stress_lines(20, 'sig', 'eps', range(NH)),
               sigL=stress_lines(20, 'sigL', 'epsL', range(NH)),
               H=H_lines(20, B['H'], SIG),
               H0=H_lines(20, H0, SIGL, '-='))

FKC = kernel('fkC_num',
             'double [::1] cs, object Finput, object shell,\n'
             '        int size, int row0, int col0, int nx, int ny, int NLgeom=0',
             FKC_DOC,
             '    cdef double Bm[%d]\n    cdef double FB[%d]\n' % (NE*ND, NE*ND)
             + STRAIN_CDEF, FINPUT, F_POINT, FKC_FORM)

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
    of the linear strains of ``cs`` and `\boldsymbol{\sigma}_0` the
    constant membrane stress resultants `N_{xx0}, N_{yy0}, N_{xy0}`:

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
                sigL[0] = sigL[0] + Nxx0
                sigL[1] = sigL[1] + Nyy0
                sigL[2] = sigL[2] + Nxy0
                for p in range({nd2}):
                    K[p] = 0.
{H}""".format(dv=dv_lines(16, 'cs'), epsL=strain_lines(16, B['epsL'], 'epsL'),
              sigL=stress_lines(16, 'sigL', 'epsL', range(NH)),
              nd2=ND*ND, H=H_lines(16, H0, SIGL))

FKG = kernel('fkG_num',
             'double [::1] cs, object Finput, object shell,\n'
             '        int size, int row0, int col0, int nx, int ny,\n'
             '        double Nxx0=0, double Nyy0=0, double Nxy0=0',
             FKG_DOC, STRAIN_CDEF, FINPUT, F_POINT, FKG_FORM)

KM = B['KM']

FKM_DOC = r"""Mass matrix, integrated numerically

    Parameters
    ----------
<<shell>>
    I_input : array-like
        Moments of the density about the mid-surface of the laminate and
        its thickness, ``J0, J1, J2, J3, J4, J6, h``, shape ``(7,)`` for the
        whole domain or ``(nx, ny, 7)`` with one set per integration point,
        see :meth:`.Shell.calc_kM`.
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
    With the displacement field through the thickness of the module,
    `\{u(z), v(z), w(z)\}^\top = \boldsymbol{J}(z) \boldsymbol{d}`, and the
    density `\rho_m`:

    .. math::

        \boldsymbol{M} = \int \boldsymbol{\Phi}^\top \left(
        \int_{d - h/2}^{d + h/2} \rho_m \boldsymbol{J}(z)^\top
        \boldsymbol{J}(z) \, dz \right) \boldsymbol{\Phi} \, dA

    with `dA = (\rho/r)\, dx\, dy`. In `\zeta = z - d` the field has the
    powers `\zeta^0`, `\zeta` and `\zeta^3`: the integral through the
    thickness is written with the moments `J_n = \int \rho_m \zeta^n
    d\zeta`, `n = 0, 1, 2, 3, 4, 6`, the offset `d` (``shell.offset``) and
    `c_1 = 4/(3 h^2)`.
"""
FKM_DOC = fill(FKM_DOC, None)

FKM_CDEF = """    cdef double J0, J1, J2, J3, J4, J6, d
    cdef double [:, :, ::1] Inxny
    cdef int one_I_each_point = 0
"""

FKM_SETUP = """    I_input = np.ascontiguousarray(I_input, dtype=DOUBLE)
    if I_input.shape == (nx, ny, 7):
        Inxny = I_input
        one_I_each_point = 1
    elif I_input.shape == (7,):
        Inxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        J0, J1, J2, J3, J4, J6, h = I_input
    else:
        raise ValueError('Invalid shape for I_input!')
    d = shell.offset
"""

FKM_EXTRA = """                if one_I_each_point == 1:
                    J0 = Inxny[ptx, pty, 0]
                    J1 = Inxny[ptx, pty, 1]
                    J2 = Inxny[ptx, pty, 2]
                    J3 = Inxny[ptx, pty, 3]
                    J4 = Inxny[ptx, pty, 4]
                    J6 = Inxny[ptx, pty, 5]
                    h = Inxny[ptx, pty, 6]
                c1 = 4./(3.*h*h)"""

FKM_FORM = """                for p in range({nd2}):
                    K[p] = 0.
{K}""".format(nd2=ND*ND, K=matrix_assign('K', KM, 16, ND))

#NOTE the mass kernel reads h from I_input, after the common setup, which
#     reads shell.lam.h
FKM = kernel('fkM_num',
             'object shell, object I_input, int size,\n'
             '        int row0, int col0, int nx, int ny',
             FKM_DOC, FKM_CDEF, '', FKM_EXTRA, FKM_FORM).replace(
    "    ftab = np.zeros(", FKM_SETUP + "    ftab = np.zeros(", 1)

FINT_DOC = r"""Internal force vector, integrated numerically

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

BM_LINES = '\n'.join(['                for p in range(%d):' % (NE*ND),
                      '                    Bm[p] = 0.',
                      matrix_assign('Bm', Bm, 16, ND)])

FINT = '''
def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    r"""{doc}
    """
{common}    cdef double Bm[{nb}]
    cdef double bs[{nd}]
    cdef double eps[{ne}]
    cdef double sig[{ne}]
    cdef double [::1] fint
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
                    bs[p] = 0.
                    for r in range({ne}):
                        bs[p] += Bm[r*{nd} + p]*sig[r]
                for j in range(n):
                    for i in range(m):
                        row = row0 + {dof}*(j*m + i)
{phiA}
                        for e in range({dof}):
                            for p in range(P0[e], P1[e]):
                                fint[row + e] += weight*pA[p]*bs[p]
    return np.asarray(fint)
'''.format(doc=FINT_DOC, common=COMMON_CDEF, nb=NE*ND, nd=ND, ne=NE,
           finput=FINPUT, setup=SETUP, point=POINT.format(extra=F_POINT),
           dv=dv_lines(16, 'cs'), eps=strain_lines(16, B['eps']),
           sig=stress_lines(16), Bm=BM_LINES,
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
    cte = [Symbol('Nxx_cte'), Symbol('Nyy_cte'), Symbol('Nxy_cte')] + [0]*(NE - 3)
    Hs = (B['H'].subs(dict(zip(th.STRESSES, SIG)), simultaneous=True)
          .subs(DV, simultaneous=True)
          + H0.subs(dict(zip(th.STRESSES, cte)), simultaneous=True))
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
    cdef double eps[{ne}]
    cdef double sig[{ne}]
    cdef double [::1] kTv
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
           finput=FINPUT, setup=SETUP, point=POINT.format(extra=F_POINT),
           dq=dv_lines(16, 'vs').replace('dv[', 'dq['), dv=dv_lines(16, 'cs'),
           eps=strain_lines(16, B['eps']), sig=stress_lines(16, rows=range(NH)),
           Bm=BM_LINES, dof=DOF, Hq=_Hq_lines(16),
           phiA=phi_lines('A', 'i', 'j', 24))

FIELD_PARAMS = """    c : np.ndarray
        Ritz constants of the panel ``s``, 1-D array of size ``5*s.m*s.n``,
        see :meth:`.Shell.get_size`.
    s : :class:`.Shell`
        The annular sector, whose dimensions, outer radius, thickness,
        numbers of terms and boundary flags are read.
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


def fstrain(double [::1] c, object s, double [::1] xs, double [::1] ys,
            int num_cores=4, int NLgeom=0):
    r"""Generalized strains of the panel at the points ``xs``, ``ys``

    Parameters
    ----------
{params}
    NLgeom : int, optional
        ``1`` to include the non-linear terms of the strains, ``0``
        otherwise. Default ``0``.

    Returns
    -------
    strains : tuple of np.ndarray
        The {ne} generalized strains along the local axes, in the order of
        :func:`.fsdt_tsdt_field.strain_names`, 1-D arrays of the size of
        ``xs``.

    Notes
    -----
    The strains given in the notes of the module, with the non-linear terms
    multiplied by ``NLgeom``.

    """
    cdef double R, s_, irho, h, c1
    R = s.r
    h = s.lam.h
    c1 = 4./(3.*h*h)
    out = np.zeros(({ne}, xs.shape[0]), dtype=DOUBLE)
    for k in range(xs.shape[0]):
        dv = _point_derivatives(c, s, xs[k], ys[k])
        irho = 1./(R - xs[k])
        s_ = R*irho
{fstrain}
    return tuple(out)
'''.format(bc=BC_CDEF, bcread=BC_READ, nd=ND, ne=NE, nt=3*DOF, tabs=TABS(4),
           dnames=', '.join(dd.name for dd in th.D),
           dv=dv_lines(4, 'c'), params=FIELD_PARAMS,
           fstrain=_fstrain_lines(8))

NOT_IMPLEMENTED = '''
def {name}(*args, **kwargs):
    r"""Not available for the annular sector

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
            'the product of the tangent stiffness by a vector\n:func:`calc_kTv` '
            'and the strains :func:`fstrain`; the displacements and\nrotations '
            ':func:`fuvw`, the approximation functions :func:`fg` and the\n'
            'names :func:`strain_names`, :func:`stress_names` are those of\n'
            ':mod:`panels.models.fsdt_tsdt_field`. The degree of freedom ``k`` '
            'of\nthe term ``(i, j)``, ``0 <= i < m``, ``0 <= j < n``, is at the '
            'row\n``row0 + 5*(j*m + i) + k`` of the matrices, in the order '
            '``u, v, w,\nphix, phiy``.\n'
            + MODULE_NOTES + '\n"""\n')
    imports = '''from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
from .fsdt_tsdt_field import fg, fuvw, strain_names, stress_names


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

    Parameters
    ----------
    shell : :class:`.Shell`
        The annular sector, with its constitutive matrix ``shell.ABD`` and
        its numbers of integration points ``shell.nx`` and ``shell.ny``.
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
    :func:`.{model}_num.fkM_num` with the moments ``intrho``,
    ``intrhoz``, ``intrhoz2``, ``intrhoz3``, ``intrhoz4``, ``intrhoz6`` of
    the laminate about its mid-surface, :meth:`.Shell._get_lam_mid`, and the
    thickness of ``shell.lam``; the offset ``shell.offset`` enters the
    kernel.

    """
    mid = shell._get_lam_mid()
    I_input = np.array([mid.intrho, mid.intrhoz, mid.intrhoz2, mid.intrhoz3,
                        mid.intrhoz4, mid.intrhoz6, shell.lam.h], dtype=DOUBLE)
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
