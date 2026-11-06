"""Generate the kernels of the model ``frame_web_clpt_donnell``

Writes ``panels/models/frame_web_clpt_donnell_num.pyx`` (numerically
integrated matrices, internal force vector and fields) and
``panels/models/frame_web_clpt_donnell.pyx`` (the matrices of the whole
domain called by :class:`.Shell` for the default integration) from the
derivation of ``frame_web_clpt_donnell.py``::

    python write_pyx.py [output folder]

Kernel layout
-------------
At each integration point the twelve generalized derivatives ``d = [Phi] c``
(``frame_web_clpt_donnell.D``), the 6 x 12 matrix ``[B] = d eps/d d`` and
the 12 x 12 matrix ``K`` of the kernel (``[B]^T [ABD] [B] + [H(N_NL)]`` for
``fkC_num``, ``[H(N_L + Nbar)]`` for ``fkG_num``, ``[J]^T [I] [J]`` for
``fkM_num``) are evaluated once. The entry of the pair of terms ``A = (i,
j)``, ``B = (k, l)`` and the degrees of freedom ``e``, ``f`` is ``sum_pq
Phi_A[p, e] K[p, q] Phi_B[q, f]``, computed as ``TA[e, q] = sum_p Phi_A[p,
e] K[p, q]`` in the loop over the terms ``A`` and ``sum_q TA[e, q] Phi_B[q,
f]`` in the loop over the terms ``B``; ``Phi_A[p, e]`` is zero unless the
generalized derivative ``p`` belongs to the degree of freedom ``e``. The
coefficients of the strains depend on the radius ``rho = R - x`` of the
point, so that the integrands are rational functions of ``x``, integrated
with the Gauss-Legendre rule.

The script is self-contained: the templates repeat those of
``../plate_clpt_gl/write_pyx.py`` where they apply.

"""
import os
import sys

import sympy
from sympy import Rational, Symbol
from sympy.printing.str import StrPrinter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)

import frame_web_clpt_donnell as th

MODEL = 'frame_web_clpt_donnell'
ND = len(th.D)
DOF = th.DOF
#: first and last + 1 generalized derivative of each degree of freedom
P0 = (0, 3, 6)
P1 = (3, 6, 12)

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


def pstr(e):
    """Cython expression, ``R/rho`` and ``1/rho`` as ``s`` and ``irho``"""
    e = sympy.expand(sympy.sympify(e).subs(th.R, S_*th.rho)
                     .subs(th.rho, 1/IRHO))
    return _Printer().doprint(e)


B = th.build()

HEADER = """#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
"""

REFERENCES = r"""
    References
    ----------
    .. [1] Amabili, M., Nonlinear Vibrations and Stability of Shells and
       Plates, Cambridge University Press, 2008, Sec. 1.4.
       https://doi.org/10.1017/CBO9780511619694
"""

MODULE_NOTES = r"""
The web lies in the plane normal to the axis of a cylinder of radius ``r``,
``shell.r``, the outer radius of the web, between the radii ``r`` and ``r -
a``: ``a``, ``shell.a``, is the height of the web and ``b``, ``shell.b``,
the arc length of its outer edge. The coordinate `x = r - \rho` runs from the
outer edge inwards, `y = r \theta` along the outer edge, with the polar
coordinates `(\rho, \theta)` of the plane; ``u``, ``v``, ``w`` are the
displacements along `\boldsymbol{e}_x = -\boldsymbol{e}_\rho`,
`\boldsymbol{e}_y = \boldsymbol{e}_\theta` and `\boldsymbol{e}_z =
\boldsymbol{e}_x \times \boldsymbol{e}_y`, the local axes of the flange of
the connection ``'BFxcte'`` whose base is the cylinder.

Notes
-----
With `s = r/\rho`, the von Kármán strains in polar coordinates of Amabili
[1]_, Eqs. (1.175)-(1.180), are, in these coordinates:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} w_{,x}^2 \qquad
    \varepsilon_{yy} = s v_{,y} - \frac{u}{\rho} + \frac{1}{2} s^2 w_{,y}^2

    \gamma_{xy} = s u_{,y} + v_{,x} + \frac{v}{\rho} + s w_{,x} w_{,y}

    \kappa_{xx} = -w_{,xx} \qquad
    \kappa_{yy} = \frac{w_{,x}}{\rho} - s^2 w_{,yy} \qquad
    \kappa_{xy} = -2 s \left(w_{,xy} + \frac{w_{,y}}{\rho}\right)

and the element of area is `dA = (\rho/r)\, dx\, dy`. The integrands are
rational functions of `x`, integrated with the Gauss-Legendre rule over the
domain of :meth:`.Shell.integration_limits`.

References
----------
.. [1] Amabili, M., Nonlinear Vibrations and Stability of Shells and Plates,
   Cambridge University Press, 2008, Sec. 1.4.
   https://doi.org/10.1017/CBO9780511619694
"""

SHELL_PARAM = """    shell : :class:`.Shell`
        The web, which gives the height ``a``, the arc length ``b`` of the
        outer edge, the outer radius ``r``, the numbers of terms ``m`` and
        ``n`` and the boundary flags of the approximation functions of each
        field."""

BC_NAMES = ['%s%s%s%s' % (xy, k, f, r) for xy in 'xy' for k in '12'
            for f in 'uvw' for r in ('', 'r')]
BC_CDEF = '\n'.join('    cdef double ' + ', '.join(
    n for n in BC_NAMES if n[0] == xy and n[2] == f)
    for xy in 'xy' for f in 'uvw')
BC_READ = '\n'.join('    ' + '; '.join('%s = shell.%s' % (n, n) for n in BC_NAMES
                                       if n[0] == xy and n[2] == f)
                    for xy in 'xy' for f in 'uvw')

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
        raise ValueError('the outer radius r must be larger than the height a '
                         'of the web, got r={{0}} and a={{1}}'.format(R, a))
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
    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)
""".format(bc=BC_READ)

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


def phi_lines(side, ii, jj, indent):
    """``Phi`` of the term ``(ii, jj)``, the twelve generalized derivatives"""
    t = ' '*indent
    rows = [('0', '0'), ('1', '0'), ('0', '1'), ('2', '2'), ('3', '2'),
            ('2', '3'), ('4', '4'), ('5', '4'), ('4', '5'), ('6', '4'),
            ('4', '6'), ('5', '5')]
    return '\n'.join('%sp%s[%d] = ftab[%s, %s]*gtab[%s, %s]'
                     % (t, side, k, fr, ii, gr, jj)
                     for k, (fr, gr) in enumerate(rows))


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
        t + '            dv[p] += pA[p]*%s[col + (0 if p < 3 else (1 if p < 6 else 2))]' % vec,
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
    return '\n'.join('%s%s[%d] = %s' % (t, name, i*ncols + j, pstr(Mx[i, j]))
                     for i in range(Mx.shape[0]) for j in range(Mx.shape[1])
                     if Mx[i, j] != 0)


def strain_lines(indent, eps, prefix='eps'):
    t = ' '*indent
    rep = {d: Symbol('dv[%d]' % k) for k, d in enumerate(th.D)}
    return '\n'.join('%s%s%d = %s' % (t, prefix, r, pstr(eps[r].subs(rep)))
                     for r in range(6))


def stress_lines(indent, names, eps_prefix='eps', rows=range(6)):
    t = ' '*indent
    return '\n'.join('%s%s = %s' % (t, names[r], ' + '.join(
        'F[%d]*%s%d' % (r*6 + q, eps_prefix, q) for q in range(6)))
        for r in rows)


def H_lines(indent, Nnames, op='+='):
    """Block of ``[H]`` on ``(w,x, w,y)`` for the membrane stresses"""
    t = ' '*indent
    Hs = B['H'].subs({th.Nxx: Symbol(Nnames[0]), th.Nyy: Symbol(Nnames[1]),
                      th.Nxy: Symbol(Nnames[2])})
    out = []
    for i in range(ND):
        for j in range(ND):
            if Hs[i, j] != 0:
                out.append('%sK[%d] %s %s' % (t, i*ND + j, op, pstr(Hs[i, j])))
    return '\n'.join(out)


FINPUT = """    cdef double F[36]
    cdef double [:, :, :, ::1] Fnxny
    cdef int one_F_each_point = 0
    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')
"""

F_POINT = """                if one_F_each_point == 1:
                    for i in range(6):
                        for j in range(6):
                            F[i*6 + j] = Fnxny[ptx, pty, i, j]"""

PARAMS_COMMON = """    size : int
        Number of rows and columns of the matrix, the number of degrees of
        freedom of the assembly.
    row0, col0 : int
        Row and column of the matrix where the terms of this web start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`."""

FINPUT_PARAM = """    Finput : array-like
        Constitutive matrix ``ABD`` of the classical laminated plate theory,
        shape ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx,
        ny, 6, 6)`` with one matrix per integration point."""

CS_PARAM = """    cs : array-like
        Ritz constants of the configuration, 1-D of size ``size``, in the
        units of the displacements."""

RAISES = """    Raises
    ------
    ValueError
        If ``Finput`` has an invalid shape, ``shell`` is not a
        :class:`.Shell` or its outer radius ``r`` is not larger than its
        height ``a``."""

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
BL = B['BL']

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
    \boldsymbol{\varepsilon}/\partial \boldsymbol{d}` and the membrane
    stress resultants `\boldsymbol{N}_{NL}` of the quadratic part of the
    strains:

    .. math::

        \boldsymbol{K}_C = \int \boldsymbol{\Phi}^\top \left(
        \boldsymbol{B}^\top \boldsymbol{F} \boldsymbol{B} +
        \boldsymbol{H}(\boldsymbol{N}_{NL}) \right) \boldsymbol{\Phi} \, dA

    with `\boldsymbol{H}(\boldsymbol{N}) = \boldsymbol{G}^\top
    \boldsymbol{N} \boldsymbol{G}`, `\boldsymbol{G} \boldsymbol{d} =
    \{w_{,x}, s w_{,y}\}^\top`, `dA = (\rho/r)\, dx\, dy`, see the notes of
    the module.
"""
FKC_DOC = fill(FKC_DOC, 'kC')

FKC_FORM = """                for p in range({nb}):
                    Bm[p] = 0.
                w_x = 0.
                w_y = 0.
                if NLgeom == 1:
{dv}
                    w_x = dv[7]
                    w_y = dv[8]
{Bm}
                # K = [B]^T [ABD] [B] + [H(N_NL)]
                for p in range(6):
                    for q in range({nd}):
                        FB[p*{nd} + q] = 0.
                        for r in range(6):
                            FB[p*{nd} + q] += F[p*6 + r]*Bm[r*{nd} + q]
                for p in range({nd}):
                    for q in range({nd}):
                        K[p*{nd} + q] = 0.
                        for r in range(6):
                            K[p*{nd} + q] += Bm[r*{nd} + p]*FB[r*{nd} + q]
                eps0 = 0.5*w_x*w_x
                eps1 = 0.5*s*s*w_y*w_y
                eps2 = s*w_x*w_y
                NxxNL = F[0]*eps0 + F[1]*eps1 + F[2]*eps2
                NyyNL = F[6]*eps0 + F[7]*eps1 + F[8]*eps2
                NxyNL = F[12]*eps0 + F[13]*eps1 + F[14]*eps2
{H}""".format(nb=6*ND, nd=ND, dv=dv_lines(20, 'cs').replace('col0 +', 'col0 +'),
              Bm=matrix_assign('Bm', Bm, 16, ND),
              H=H_lines(16, ('NxxNL', 'NyyNL', 'NxyNL')))

# the derivatives of cs are read with the column of the term, col0 of the web
FKC = kernel('fkC_num',
             'double [::1] cs, object Finput, object shell,\n'
             '        int size, int row0, int col0, int nx, int ny, int NLgeom=0',
             FKC_DOC,
             '    cdef double Bm[%d]\n    cdef double FB[%d]\n'
             '    cdef double w_x, w_y, eps0, eps1, eps2, NxxNL, NyyNL, NxyNL\n'
             % (6*ND, 6*ND), FINPUT, F_POINT, FKC_FORM)

FKG_DOC = r"""Geometric stiffness matrix of the linear membrane stress state

    Integrates numerically the geometric stiffness matrix of the membrane
    stress resultants of the linear strains at ``cs``, superposed to the
    constant state ``Nxx0``, ``Nyy0``, ``Nxy0``. The matrix is homogeneous
    of degree one in ``cs``, as the linear buckling eigenvalue problem
    requires; the stress of the non-linear strains enters :func:`fkC_num`.

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
    With `\boldsymbol{N} = \boldsymbol{N}_0 + \boldsymbol{N}_L`,
    `\boldsymbol{N}_L` the membrane stress resultants of the linear strains
    of ``cs``:

    .. math::

        \boldsymbol{K}_G = \int \boldsymbol{\Phi}^\top \boldsymbol{G}^\top
        \begin{bmatrix} N_{xx} & N_{xy} \\ N_{xy} & N_{yy}
        \end{bmatrix} \boldsymbol{G} \boldsymbol{\Phi} \, dA, \qquad
        \boldsymbol{G} \boldsymbol{d} = \{w_{,x}, s w_{,y}\}^\top

    the variation of the work of the membrane stresses on the quadratic
    terms of the von Kármán strains, see the notes of the module.
"""
FKG_DOC = fill(FKG_DOC, 'kG')

FKG_FORM = """{dv}
{epsL}
{N}
                Nxx = Nxx + Nxx0
                Nyy = Nyy + Nyy0
                Nxy = Nxy + Nxy0
                for p in range({nd2}):
                    K[p] = 0.
{H}""".format(dv=dv_lines(16, 'cs'), epsL=strain_lines(16, B['epsL']),
              N=stress_lines(16, ('Nxx', 'Nyy', 'Nxy'), rows=range(3)),
              nd2=ND*ND, H=H_lines(16, ('Nxx', 'Nyy', 'Nxy')))

FKG = kernel('fkG_num',
             'double [::1] cs, object Finput, object shell,\n'
             '        int size, int row0, int col0, int nx, int ny,\n'
             '        double Nxx0=0, double Nyy0=0, double Nxy0=0',
             FKG_DOC,
             '    cdef double eps0, eps1, eps2, eps3, eps4, eps5, Nxx, Nyy, Nxy\n',
             FINPUT, F_POINT, FKG_FORM)

J = B['J']
var_I = sympy.symbols('I0, I1, I2')
MAUX = sympy.Matrix([[var_I[0], 0, 0, var_I[1], 0],
                     [0, var_I[0], 0, 0, var_I[1]],
                     [0, 0, var_I[0], 0, 0],
                     [var_I[1], 0, 0, var_I[2], 0],
                     [0, var_I[1], 0, 0, var_I[2]]])
KM = J.T*MAUX*J

FKM_DOC = r"""Mass matrix, integrated numerically

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
        height ``a``.

    Notes
    -----
    With the displacement through the thickness of the Kirchhoff-Love
    kinematics, `u(z) = u - z w_{,x}`, `v(z) = v - z s w_{,y}`, `w(z) =
    w`, and the moments `I_n = \int \rho_m z^n dz`, `n = 0, 1, 2`, of the
    density `\rho_m`:

    .. math::

        \boldsymbol{M} = \int \boldsymbol{\Phi}^\top \boldsymbol{J}^\top
        \begin{bmatrix} I_0 \boldsymbol{I}_3 & I_1 \boldsymbol{E} \\
        I_1 \boldsymbol{E}^\top & I_2 \boldsymbol{I}_2 \end{bmatrix}
        \boldsymbol{J} \boldsymbol{\Phi} \, dA, \qquad
        \boldsymbol{J} \boldsymbol{d} = \{u, v, w, -w_{,x}, -s
        w_{,y}\}^\top

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

    Parameters
    ----------
<<cs>>
<<F>>
<<shell>>
    size : int
        Size of the vector, the number of degrees of freedom of the
        assembly.
    col0 : int
        Entry of the vector where the terms of this web start.
    nx, ny : int
        Number of Gauss-Legendre points along `x` and `y`.

    Returns
    -------
    fint : np.ndarray
        Internal force vector, 1-D of size ``size``.

<<raises>>

    Notes
    -----
    With the von Kármán strains `\boldsymbol{\varepsilon}(\boldsymbol{d})`
    of the module, `\boldsymbol{\sigma} = \boldsymbol{F}
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
    cdef double w_x, w_y, eps0, eps1, eps2, eps3, eps4, eps5
    cdef double Nxx, Nyy, Nxy, Mxx, Myy, Mxy
    cdef double [::1] fint
    cdef int row0 = col0
{finput}{setup}
    fint = np.zeros(size, dtype=DOUBLE)

    with nogil:
{point}
{dv}
                w_x = dv[7]
                w_y = dv[8]
{eps}
{sig}
                for p in range({nb}):
                    Bm[p] = 0.
{Bm}
                for p in range({nd}):
                    bs[p] = (Bm[p]*Nxx + Bm[{nd} + p]*Nyy + Bm[{nd2} + p]*Nxy
                             + Bm[{nd3} + p]*Mxx + Bm[{nd4} + p]*Myy
                             + Bm[{nd5} + p]*Mxy)
                for j in range(n):
                    for i in range(m):
                        row = row0 + {dof}*(j*m + i)
{phiA}
                        for e in range({dof}):
                            for p in range(P0[e], P1[e]):
                                fint[row + e] += weight*pA[p]*bs[p]
    return np.asarray(fint)
'''.format(doc=FINT_DOC, common=COMMON_CDEF, nb=6*ND, nd=ND, nd2=2*ND,
           nd3=3*ND, nd4=4*ND, nd5=5*ND, finput=FINPUT, setup=SETUP,
           point=POINT.format(extra=F_POINT), dv=dv_lines(16, 'cs'),
           eps=strain_lines(16, B['eps']),
           sig=stress_lines(16, ('Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy')),
           Bm=matrix_assign('Bm', Bm, 16, ND), dof=DOF,
           phiA=phi_lines('A', 'i', 'j', 24))

FIELD_PARAMS = """    c : np.ndarray
        Ritz constants of the web ``s``, 1-D array of size ``3*s.m*s.n``,
        see :meth:`.Shell.get_size`.
    s : :class:`.Shell`
        The web, whose dimensions, outer radius, numbers of terms and
        boundary flags are read.
    xs, ys : np.ndarray
        Coordinates `x` and `y` of the points, lengths, 1-D arrays of the
        same size.
    num_cores : int, optional
        Not used, the points are evaluated serially; kept for the interface
        of :mod:`panels.models.clpt_field`."""

FIELDS = '''
def _point_derivatives(double [::1] c, object shell, double xp, double yp):
    r"""Generalized derivatives `\\boldsymbol{{d}}` of the web at a point

    Parameters
    ----------
    c : np.ndarray
        Ritz constants of the web, 1-D array of size ``3*m*n``.
    shell : :class:`.Shell`
        The web.
    xp, yp : float
        Coordinates `x` and `y` of the point, lengths.

    Returns
    -------
    d : np.ndarray
        ``(u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy)``, 1-D
        array of size 12.

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
    ftab = np.zeros((7, m), dtype=DOUBLE)
    gtab = np.zeros((7, n), dtype=DOUBLE)
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
{dv}
    return np.array([dv[p] for p in range({nd})])


def fuvw(double [::1] c, object s, double [::1] xs, double [::1] ys,
         int num_cores=4):
    r"""Displacements and rotations of the web at the points ``xs``, ``ys``

    Parameters
    ----------
{params}

    Returns
    -------
    us, vs, ws, phixs, phiys : np.ndarray
        The displacements ``u``, ``v``, ``w`` along the local axes, lengths,
        and the rotations of the normal ``phix = -w,x`` and ``phiy = -(r/rho)
        w,y``, dimensionless, 1-D arrays of the size of ``xs``.

    Notes
    -----
    With `\\rho = r - x` the radius of the point, `-(r/\\rho) w_{{,y}} =
    -w_{{,\\theta}}/\\rho` is the derivative of `w` along the arc.

    """
    out = np.zeros((5, xs.shape[0]), dtype=DOUBLE)
    for k in range(xs.shape[0]):
        d = _point_derivatives(c, s, xs[k], ys[k])
        out[:, k] = (d[0], d[3], d[6], -d[7], -s.r/(s.r - xs[k])*d[8])
    return tuple(out)


def fstrain(double [::1] c, object s, double [::1] xs, double [::1] ys,
            int num_cores=4, int NLgeom=0):
    r"""Generalized strains of the web at the points ``xs``, ``ys``

    Parameters
    ----------
{params}
    NLgeom : int, optional
        ``1`` to include the quadratic terms of the membrane strains, ``0``
        otherwise. Default ``0``.

    Returns
    -------
    exxs, eyys, gxys, kxxs, kyys, kxys : np.ndarray
        The membrane strains, dimensionless, and the changes of curvature,
        per unit length, along the local axes, 1-D arrays of the size of
        ``xs``.

    Notes
    -----
    The von Kármán strains in polar coordinates given in the notes of the
    module, with the quadratic terms multiplied by ``NLgeom``.

    """
    cdef double rho, s_, R
    R = s.r
    out = np.zeros((6, xs.shape[0]), dtype=DOUBLE)
    for k in range(xs.shape[0]):
        d = _point_derivatives(c, s, xs[k], ys[k])
        rho = R - xs[k]
        s_ = R/rho
        u, u_x, u_y, v, v_x, v_y, w, w_x, w_y, w_xx, w_yy, w_xy = d
        out[0, k] = u_x + NLgeom*0.5*w_x*w_x
        out[1, k] = s_*v_y - u/rho + NLgeom*0.5*s_*s_*w_y*w_y
        out[2, k] = s_*u_y + v_x + v/rho + NLgeom*s_*w_x*w_y
        out[3, k] = -w_xx
        out[4, k] = w_x/rho - s_*s_*w_yy
        out[5, k] = -2*s_*(w_xy + w_y/rho)
    return tuple(out)
'''.format(bc=BC_CDEF, bcread=BC_READ.replace('shell.', 'shell.'), nd=ND,
           dv=dv_lines(4, 'c').replace('            dv[p] +=', '            dv[p] +='),
           params=FIELD_PARAMS)

NOT_IMPLEMENTED = '''
def {name}(*args, **kwargs):
    r"""Not available for the web of a ring frame

    Raises
    ------
    NotImplementedError
        Always: the piston theory is not implemented for this model.

    """
    raise NotImplementedError('{name} is not implemented for the model '
                              '"{model}"')
'''


def numerical_module():
    head = HEADER + 'r"""\nWeb of a ring frame, an annular-sector flat plate, using the classical\nlaminated plate theory (CLPT) with von Kármán strains in polar coordinates\n\nNumerically integrated kernels of the model ``\'frame_web_clpt_donnell\'``:\nthe constitutive stiffness :func:`fkC_num`, the geometric stiffness\n:func:`fkG_num`, the mass :func:`fkM_num`, the internal force vector\n:func:`calc_fint`, and the fields :func:`fuvw`, :func:`fstrain` and\n:func:`fg`, the last one that of :mod:`panels.models.clpt_field`. The\ndegree of freedom ``k`` of the term ``(i, j)``, ``0 <= i < m``, ``0 <= j <\nn``, is at the row ``row0 + 3*(j*m + i) + k`` of the matrices.\n' + MODULE_NOTES + '\n"""\n'
    imports = '''from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE
from .clpt_field import fg


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 3
cdef int[3] P0 = [0, 3, 6]
cdef int[3] P1 = [3, 6, 12]

'''
    na = ''.join(NOT_IMPLEMENTED.format(name=nm, model=MODEL)
                 for nm in ('fkAx_num', 'fkAy_num'))
    return head + imports + FKC + FKG + FKM + FINT + FIELDS + na


ANALYTICAL = HEADER + '''r"""
Web of a ring frame, an annular-sector flat plate, using the classical
laminated plate theory (CLPT) with von Kármán strains in polar coordinates

Matrices of the whole domain of the model ``'frame_web_clpt_donnell'``,
called by :class:`.Shell` when the constitutive matrix and the density are
constant: :func:`fk0`, :func:`fkG0` and :func:`fkM`. The strains have
coefficients in `1/\\rho`, with `\\rho = r - x` the radius of the point, and
the matrices are integrated with the Gauss-Legendre rule of
:mod:`panels.models.frame_web_clpt_donnell_num`, with ``shell.nx`` and
``shell.ny`` points; the piston-theory matrices ``fkAx``, ``fkAy`` and
``fcA`` are not available.

Notes
-----
See :mod:`panels.models.frame_web_clpt_donnell_num`.

"""
import numpy as np

from panels import DOUBLE
from .frame_web_clpt_donnell_num import fkC_num, fkG_num, fkM_num


def fk0(object shell, int size, int row0, int col0):
    r"""Linear constitutive stiffness matrix of the whole web

    Parameters
    ----------
    shell : :class:`.Shell`
        The web, with its constitutive matrix ``shell.ABD`` and its numbers
        of integration points ``shell.nx`` and ``shell.ny``.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the terms of this web start.

    Returns
    -------
    kC0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``, upper triangle.

    Notes
    -----
    :func:`.frame_web_clpt_donnell_num.fkC_num` with ``NLgeom=0``.

    """
    return fkC_num(np.zeros(size, dtype=DOUBLE), shell.ABD, shell, size,
                   row0, col0, shell.nx, shell.ny, 0)


def fkG0(double Nxx, double Nyy, double Nxy, object shell,
         int size, int row0, int col0):
    r"""Geometric stiffness matrix of a constant membrane stress state

    Parameters
    ----------
    Nxx, Nyy, Nxy : float
        Membrane stress resultants along the local axes, `N_{xx} =
        N_{\\rho\\rho}`, `N_{yy} = N_{\\theta\\theta}`, `N_{xy} =
        -N_{\\rho\\theta}`, force per unit length.
    shell : :class:`.Shell`
        The web, with its numbers of integration points ``shell.nx`` and
        ``shell.ny``.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the terms of this web start.

    Returns
    -------
    kG0 : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``, upper triangle.

    Notes
    -----
    :func:`.frame_web_clpt_donnell_num.fkG_num` with zero Ritz constants.

    """
    return fkG_num(np.zeros(size, dtype=DOUBLE), shell.ABD, shell, size,
                   row0, col0, shell.nx, shell.ny, Nxx, Nyy, Nxy)


def fkM(object shell, int size, int row0, int col0):
    r"""Mass matrix of the whole web

    Parameters
    ----------
    shell : :class:`.Shell`
        The web, with its laminate ``shell.lam`` and its numbers of
        integration points ``shell.nx`` and ``shell.ny``.
    size : int
        Number of rows and columns of the matrix.
    row0, col0 : int
        Row and column of the matrix where the terms of this web start.

    Returns
    -------
    kM : scipy.sparse.coo_matrix
        Matrix of shape ``(size, size)``, upper triangle.

    Notes
    -----
    :func:`.frame_web_clpt_donnell_num.fkM_num` with the moments
    ``intrho``, ``intrhoz``, ``intrhoz2`` of ``shell.lam``.

    """
    lam = shell.lam
    I_input = np.array([lam.intrho, lam.intrhoz, lam.intrhoz2], dtype=DOUBLE)
    return fkM_num(shell, I_input, size, row0, col0, shell.nx, shell.ny)
''' + ''.join(NOT_IMPLEMENTED.format(name=nm, model=MODEL)
              for nm in ('fkAx', 'fkAy', 'fcA'))


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
