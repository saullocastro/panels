"""Generate the kernels of the model ``cylshell_clpt_tl``

Writes ``panels/models/cylshell_clpt_tl_num.pyx`` (numerically integrated
matrices, internal force vector, follower-pressure kernels and product of the
tangent stiffness matrix by a vector) from the derivation of
``cylshell_clpt_tl.py``::

    python write_pyx.py

The analytical kernels are those of :mod:`panels.models.cylshell_clpt_gl`:
the linear strains are those of ``cylshell_clpt_sanders`` (``fk0``, ``fkM``,
``fkAx``, ``fkAy``, ``fcA``), and the geometric stiffness of a constant
membrane stress state (``fkG0``) is that of the Green-Lagrange membrane
strains, the curvatures only adding terms of the moments. The mass and
aerodynamic matrices of the numerical module, ``fkM_num``, ``fkAx_num`` and
``fkAy_num``, are imported from :mod:`panels.models.cylshell_clpt_sanders_num`.

The kernels use the helpers of ``plate_clpt_gl/write_pyx.py`` (layout,
tables, assembly of the pairs of terms, ``calc_fint``) with the seventeen
generalized derivatives of this model, among which the second derivatives
of ``u`` and ``v`` and the values ``v`` and ``w``, which enter the strains
through the curvature of the reference surface: the tables of the
approximation functions have nine rows along each direction. A private copy
of that generator is loaded and the radius is added to its declarations and
to the reading of the input, named ``radius`` in the kernels, ``r`` being
one of their loop indices. ``fkC_num``, ``fkG_num`` and ``calc_kTv`` are
written here, as in ``plate_clpt_tl/write_pyx.py``: ``[B]`` is quadratic in
the derivatives and the Hessian of the curvatures linear in them, and
``calc_kTv`` evaluates ``[B]^T [ABD] [B] v + [H(sigma, d)] v`` at each point,
since the strains are not of the form assumed by
``theory/shells/tangent_product``. The follower-pressure kernels are those
of the Sanders cylinder, whose area vector is the exact one of the deformed
mid-surface (``follower_pressure.MODELS``).

"""
import importlib.util
import os
import sys
import textwrap
import types

from sympy import Matrix, Symbol, expand

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import cylshell_clpt_tl as th


def _load(name, *path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, *path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MODEL = 'cylshell_clpt_tl'
DOF = th.DOF
NE = th.NE
ND = len(th.D)
#: name of the radius in the kernels
RADIUS = Symbol('radius')

#: rows of the tables of the approximation functions, along x and y
F_ROWS = ['u', 'u_x', 'u_xx', 'v', 'v_x', 'v_xx', 'w', 'w_x', 'w_xx']
G_ROWS = ['u', 'u_y', 'u_yy', 'v', 'v_y', 'v_yy', 'w', 'w_y', 'w_yy']

DESC = ('Cylindrical shells using the classical laminated plate theory '
        '(CLPT), total Lagrangian: Green-Lagrange membrane strains and '
        'changes of curvature measured with the deformed normal')

#NOTE a private copy of the generator of plate_clpt_gl, whose helpers write
#     the tables, the assembly and calc_fint; the radius is added to its
#     declarations and to the reading of the input
gw = _load('plate_clpt_gl_write_pyx_for_cylshell_tl', '..', 'plate_clpt_gl',
           'write_pyx.py')
pstr = gw.pstr
matrix_lines = gw.matrix_lines


def _build():
    return {k: v.xreplace({th.r: RADIUS}) for k, v in th.build().items()}


thc = types.SimpleNamespace(build=_build, D=th.D, D_NAMES=th.D_NAMES,
                            GRAD=th.GRAD, DOF=th.DOF, NE=th.NE, F=th.F,
                            Nxx=th.Nxx, Nyy=th.Nyy, Nxy=th.Nxy, a=th.a,
                            b=th.b, FIELDS=th.FIELDS)
gw.th = thc
gw.COMMON_CDEF = gw.COMMON_CDEF + '    cdef double radius\n'
_lim = '    x1, x2, y1, y2 = shell.integration_limits()\n'
assert gw.FINPUT_TEMPLATE.count(_lim) == 1
gw.FINPUT_TEMPLATE = gw.FINPUT_TEMPLATE.replace(
    _lim, _lim + '    radius = shell.r\n')

SP = gw.Spec(thc, F_ROWS, G_ROWS)
D_NAMES = SP.D_NAMES
FINPUT = gw.finput(SP)
POINT = gw.point(SP)
COMMON = gw.COMMON_CDEF

SG = [Symbol('sg[%d]' % r) for r in range(NE)]
SGL = [Symbol('sgL[%d]' % r) for r in range(NE)]


def cdefs(names, per=8):
    return '\n'.join('    cdef double ' + ', '.join(names[i:i + per])
                     for i in range(0, len(names), per))


def zero_lines(names, indent=16):
    return '\n'.join('{0}{1} = 0.'.format(' '*indent, nm) for nm in names)


def stresses(e, sg, indent=16):
    sp = ' '*indent
    return '\n'.join([sp + 'for r in range(6):',
                      sp + '    {0}[r] = 0.'.format(sg),
                      sp + '    for s in range(6):',
                      sp + '        {0}[r] += F[r*6 + s]*{1}[s]'.format(sg, e)])


NOTES = r'''Notes
-----
The membrane strains are the Green-Lagrange strains of the mid-surface,
those of :mod:`panels.models.cylshell_clpt_gl`, which vanish for any rigid
motion:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} (u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2)

    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \left(u_{,y}^2 +
    \left(v_{,y} + \frac{w}{r}\right)^2 + \left(w_{,y} -
    \frac{v}{r}\right)^2\right)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} \left(v_{,y} +
    \frac{w}{r}\right) + w_{,x} \left(w_{,y} - \frac{v}{r}\right)

with `y` the arc length along the circumference and `w` positive outwards,
the membrane strains of the Flügge-Lur'e-Byrne non-linear theory as given
by Amabili [1]_, Eqs. (1.118)-(1.120). The changes of curvature are measured
with the area vector `\boldsymbol{a} = \boldsymbol{X}_{,x} \times
\boldsymbol{X}_{,y}` of the deformed mid-surface `\boldsymbol{X}`, the
normal times the area stretch `|\boldsymbol{a}| = 1 + \varepsilon_{xx} +
\varepsilon_{yy} + \ldots`:

.. math::

    \kappa_{xx} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,xx} \qquad
    \kappa_{xy} = -2 \boldsymbol{a} \cdot \boldsymbol{X}_{,xy} -
    \frac{\gamma_{xy}}{2 r}

    \kappa_{yy} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,yy} - \frac{1}{r} -
    \frac{\varepsilon_{xx} + 2 \varepsilon_{yy}}{r}

with, in the local basis `(\boldsymbol{e}_x, \boldsymbol{e}_y,
\boldsymbol{e}_z)` of the undeformed cylinder:

.. math::

    \boldsymbol{X}_{,x} = (1 + u_{,x},\; v_{,x},\; w_{,x}) \qquad
    \boldsymbol{X}_{,y} = \left(u_{,y},\; 1 + v_{,y} + \frac{w}{r},\;
    w_{,y} - \frac{v}{r}\right)

    \boldsymbol{X}_{,xx} = (u_{,xx},\; v_{,xx},\; w_{,xx}) \qquad
    \boldsymbol{X}_{,xy} = \left(u_{,xy},\; v_{,xy} + \frac{w_{,x}}{r},\;
    w_{,xy} - \frac{v_{,x}}{r}\right)

    \boldsymbol{X}_{,yy} = \left(u_{,yy},\; v_{,yy} + \frac{2 w_{,y}}{r} -
    \frac{v}{r^2},\; w_{,yy} - \frac{2 v_{,y}}{r} - \frac{w}{r^2} -
    \frac{1}{r}\right)

`-\boldsymbol{a} \cdot \boldsymbol{X}_{,\alpha\beta} + (1 + \varepsilon_{xx}
+ \varepsilon_{yy}) b^0_{\alpha\beta}`, with `b^0_{yy} = -1/r` the only
non-zero component of the second fundamental form of the undeformed
cylinder, is the change of the second fundamental form `b_{\alpha\beta} =
\boldsymbol{n} \cdot \boldsymbol{X}_{,\alpha\beta}` [3]_, Sec. 3.16, times
the area stretch, to first order in the reference term. The terms
`-\varepsilon_{yy}/r` and `-\gamma_{xy}/(2 r)` make the linear part the
linear changes of curvature of Sanders [2]_, those of
:mod:`panels.models.cylshell_clpt_sanders`, with the rotation about the
normal (Koiter's form, [3]_, Eq. (3.16.8)); changes of curvature that
differ by terms of the order of the membrane strain divided by the radius
are equivalent within the accuracy of the classical theory of thin shells
([3]_, p. 202). The six strains are invariant under any rigid rotation of
any deformed state, cubic polynomials in the displacements, see
:mod:`panels.models`. The integrands are polynomials of degree `6p - 2`
along `x` and `6p` along `y` for approximation functions of degree `p`,
integrated exactly by `3p + 1` Gauss-Legendre points, the default of
:class:`.Shell` for this model.
'''

REFS = r'''
References
----------
.. [1] Amabili, M., "Nonlinear Vibrations and Stability of Shells and
   Plates", Cambridge University Press, 2008.
   https://doi.org/10.1017/CBO9780511619694
.. [2] Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
   Applied Mathematics, 21(1), 21-36, 1963.
   https://doi.org/10.1090/qam/147023
.. [3] van der Heijden, A. M. A. (ed.), "W. T. Koiter's Elastic Stability
   of Solids and Structures", Cambridge University Press, 2008.
   https://doi.org/10.1017/CBO9780511546174
'''


def numerical_module():
    m = SP.M
    B, eps, eps_L, H, H0 = m['B'], m['eps'], m['eps_L'], m['H'], m['H0']
    alld = list(range(ND))
    full = [[True]*ND for _ in range(ND)]
    blocks_all = [(d, e) for d in range(DOF) for e in range(DOF)]

    out = [gw.HEADER + 'r"""\n' + textwrap.fill(DESC, 76) + '''

Numerically integrated kernels of the model ``'cylshell_clpt_tl'``: the
constitutive stiffness :func:`fkC_num`, the geometric stiffness
:func:`fkG_num`, the internal force vector :func:`calc_fint`, the product
of the tangent stiffness matrix by a vector :func:`calc_kTv`, and the force
vector and load stiffness matrix of a follower pressure,
:func:`calc_fext_follower` and :func:`fkCfollower_num`. The mass and
aerodynamic matrices, ``fkM_num``, ``fkAx_num`` and ``fkAy_num``, are those
of :mod:`panels.models.cylshell_clpt_sanders_num`. The constitutive matrix
``Finput`` is the ``6 x 6`` matrix ``ABD`` of the CLPT.

The degrees of freedom of each term of the approximation are
``u, v, w``; the degree of freedom ``k`` of the term ``(i, j)``, ``0 <=
i < m``, ``0 <= j < n``, is at the row ``row0 + 3*(j*m + i) + k``
of the matrices.

''' + NOTES + '''
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
#     of the Sanders-Koiter cylinder
from .cylshell_clpt_sanders_num import fkM_num, fkAx_num, fkAy_num


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 3
''']

    # ------------------------------------------------------------- fkC_num
    Hd = (H.xreplace(dict(zip(th.STRESSES, SG)))
          - H0.xreplace(dict(zip(th.STRESSES, SGL)))).applyfunc(expand)
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
    With the generalized derivatives at the integration point `\\boldsymbol{{d}}
    = \\boldsymbol{{\\Phi}} \\boldsymbol{{c}}` (the gradients and second derivatives
    of `u`, `v`, `w` and the values of `v` and `w`), the strains
    `\\boldsymbol{{\\varepsilon}}(\\boldsymbol{{d}})`,
    `\\boldsymbol{{B}} = \\partial \\boldsymbol{{\\varepsilon}} / \\partial
    \\boldsymbol{{d}}`, the stresses `\\boldsymbol{{\\sigma}} = \\boldsymbol{{F}}
    \\boldsymbol{{\\varepsilon}}` and `\\boldsymbol{{H}}(\\boldsymbol{{\\sigma}},
    \\boldsymbol{{d}}) = \\sum_r \\sigma_r \\, \\partial^2 \\varepsilon_r / \\partial
    \\boldsymbol{{d}}^2`:

    .. math::

        \\boldsymbol{{K}}_C = \\int \\boldsymbol{{\\Phi}}^\\top \\left[\\boldsymbol{{B}}^\\top
        \\boldsymbol{{F}} \\boldsymbol{{B}} + \\boldsymbol{{H}}(\\boldsymbol{{\\sigma}},
        \\boldsymbol{{d}}) - \\boldsymbol{{H}}(\\boldsymbol{{\\sigma}}_L,
        \\boldsymbol{{0}})\\right] \\boldsymbol{{\\Phi}} \\, dA

    with `\\boldsymbol{{\\sigma}}_L` the stresses of the linear strains, whose
    term `\\boldsymbol{{H}}(\\boldsymbol{{\\sigma}}_L, \\boldsymbol{{0}})` is assembled
    by :func:`fkG_num`, such that ``fkC_num(NLgeom=1) + fkG_num`` is the exact
    Jacobian of :func:`calc_fint`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \\le x \\le x_2`, `y_1 \\le y \\le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
{COMMON}
{dcdef}
    cdef double e[6]
    cdef double eL[6]
    cdef double sg[6]
    cdef double sgL[6]
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
                # generalized derivatives at the point, zero when NLgeom == 0
{dzero}
                if NLgeom == 1:
{dloop}

                # [B] = d eps/d d at the point
                for p in range({NB}):
                    Bm[p] = 0.
{Blines}

                # K = [B]^T [ABD] [B]
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

                # stresses of the strains and of the linear strains
{elines}
{eLlines}
{sg}
{sgL}
                # K += [H(sigma, d)] - [H0(sigma_L)], the second term being
                # assembled by fkG_num, which is homogeneous of degree one
                # in cs as linear buckling requires
{Hlines}

{loops}

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC
'''.format(COMMON=COMMON, dcdef=cdefs(list(D_NAMES)), basis=gw.basis_cdef('AB', SP),
           NB=NE*ND, NK=ND*ND, NT=DOF*ND, ND=ND, FINPUT=FINPUT, POINT=POINT,
           dzero=zero_lines(D_NAMES),
           dloop='\n'.join(gw.gradient_loop(20, alld, SP)),
           Blines='\n'.join(matrix_lines('Bm', B, 16)),
           elines='\n'.join('                e[{0}] = {1}'.format(k, pstr(eps[k])) for k in range(NE)),
           eLlines='\n'.join('                eL[{0}] = {1}'.format(k, pstr(eps_L[k])) for k in range(NE)),
           sg=stresses('e', 'sg'), sgL=stresses('eL', 'sgL'),
           Hlines='\n'.join(matrix_lines('K', Hd, 16, op='+=')),
           loops=gw.assembly('kC', full, blocks_all, sp=SP)))

    # ------------------------------------------------------------- fkG_num
    H0s = H0.xreplace(dict(zip(th.STRESSES, SG))).applyfunc(expand)
    H0pat = [[H0[p, q] != 0 for q in range(ND)] for p in range(ND)]
    blocks = [(d, e) for d in range(DOF) for e in range(DOF)
              if any(H0pat[p][q] for p in SP.DSET[d] for q in SP.DSET[e])]
    lin = [p for p in range(ND) if th.D[p] in set().union(*[x.free_symbols for x in eps_L])]
    out.append('''

def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    r"""Geometric stiffness matrix of the linear stress state

    Integrates numerically the geometric stiffness matrix of the stresses
    of the linear strains at ``cs``, superposed to the constant membrane
    state ``Nxx0``, ``Nyy0``, ``Nxy0``. Besides the blocks of the membrane
    stresses of :mod:`panels.models.cylshell_clpt_gl`, the moments couple
    the first derivatives and the values of ``u``, ``v`` and ``w`` with
    their second derivatives. The matrix is homogeneous of degree one in
    ``cs``, as the linear buckling eigenvalue problem requires.

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
    With the generalized derivatives at the integration point `\\boldsymbol{{d}}
    = \\boldsymbol{{\\Phi}} \\boldsymbol{{c}}` (the gradients and second derivatives
    of `u`, `v`, `w` and the values of `v` and `w`), the strains
    `\\boldsymbol{{\\varepsilon}}(\\boldsymbol{{d}})`,
    `\\boldsymbol{{B}} = \\partial \\boldsymbol{{\\varepsilon}} / \\partial
    \\boldsymbol{{d}}`, the stresses `\\boldsymbol{{\\sigma}} = \\boldsymbol{{F}}
    \\boldsymbol{{\\varepsilon}}` and `\\boldsymbol{{H}}(\\boldsymbol{{\\sigma}},
    \\boldsymbol{{d}}) = \\sum_r \\sigma_r \\, \\partial^2 \\varepsilon_r / \\partial
    \\boldsymbol{{d}}^2`:

    .. math::

        \\boldsymbol{{K}}_G = \\int \\boldsymbol{{\\Phi}}^\\top
        \\boldsymbol{{H}}(\\boldsymbol{{\\sigma}}, \\boldsymbol{{0}}) \\boldsymbol{{\\Phi}}
        \\, dA

    with `\\boldsymbol{{\\sigma}} = \\{{N_{{xx0}}, N_{{yy0}}, N_{{xy0}}, 0, 0, 0\\}}^\\top +
    \\boldsymbol{{\\sigma}}_L`, `\\boldsymbol{{\\sigma}}_L` the stresses of the
    linear strains at ``cs``. The rest of the tangent stiffness matrix is
    assembled by :func:`fkC_num`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \\le x \\le x_2`, `y_1 \\le y \\le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
{COMMON}
{dcdef}
    cdef double e[6]
    cdef double sg[6]
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

                # K = [H0(sigma)]
{Hlines}

{loops}

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG
'''.format(COMMON=COMMON, dcdef=cdefs([D_NAMES[p] for p in lin]),
           basis=gw.basis_cdef('AB', SP), NK=ND*ND, NT=DOF*ND, ND=ND,
           FINPUT=FINPUT, POINT=POINT, NBLK=len(blocks),
           dzero=zero_lines([D_NAMES[p] for p in lin]),
           dloop='\n'.join(gw.gradient_loop(16, lin, SP)),
           elines='\n'.join('                e[{0}] = {1}'.format(k, pstr(eps_L[k])) for k in range(NE)),
           sg=stresses('e', 'sg'),
           Hlines='\n'.join(matrix_lines('K', H0s, 16)),
           loops=gw.assembly('kG', H0pat, blocks, sp=SP)))

    # ------------------------------------------------------------- calc_fint
    out.append(gw.calc_fint_src(SP))

    # ------------------------------------------------------------- calc_kTv
    out.append(calc_kTv_src())
    text = ''.join(out)
    #NOTE c, p, q, r, s, k, l and parts of the tables are not used in every
    #     kernel, which is harmless
    return text


def calc_kTv_src():
    r"""``calc_kTv``: ``K_T v = int [Phi]^T ([B]^T [ABD] [B] [Phi] v + [H(sigma
    + Nbar, d)] [Phi] v) dA`` at each point"""
    m = SP.M
    B, eps, H = m['B'], m['eps'], m['H']
    dv = Matrix([Symbol('d' + nm) for nm in D_NAMES])
    DSG = Matrix([Symbol('dsg[%d]' % r) for r in range(NE)])
    de = (B*dv).applyfunc(expand)
    S = (B.T*DSG + H.xreplace(dict(zip(th.STRESSES, SG)))*dv).applyfunc(expand)
    kTv_lines = []
    for d in range(DOF):
        terms = ['pA[{0}]*S[{0}]'.format(p) for p in SP.DSET[d]]
        kTv_lines.append('                        kTv[col+{0}] += weight*({1})'.format(d, ' + '.join(terms)))
    alld = list(range(ND))
    return '''

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
    Finput : array-like
        Constitutive matrix of the generalized strains of the model, shape
        ``(6, 6)`` for the whole domain, e.g. ``shell.ABD``, or ``(nx, ny,
        6, 6)`` with one matrix per integration point.
    shell : :class:`.Shell`
        The shell, which gives the dimensions ``a`` and ``b``, the radius ``r``
        of the cylinders, the numbers of terms ``m`` and ``n`` and the boundary
        flags of the approximation functions of each field.
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
    With the generalized derivatives at the integration point `\\boldsymbol{{d}}
    = \\boldsymbol{{\\Phi}} \\boldsymbol{{c}}` (the gradients and second derivatives
    of `u`, `v`, `w` and the values of `v` and `w`), the strains
    `\\boldsymbol{{\\varepsilon}}(\\boldsymbol{{d}})`,
    `\\boldsymbol{{B}} = \\partial \\boldsymbol{{\\varepsilon}} / \\partial
    \\boldsymbol{{d}}`, the stresses `\\boldsymbol{{\\sigma}} = \\boldsymbol{{F}}
    \\boldsymbol{{\\varepsilon}}` and `\\boldsymbol{{H}}(\\boldsymbol{{\\sigma}},
    \\boldsymbol{{d}}) = \\sum_r \\sigma_r \\, \\partial^2 \\varepsilon_r / \\partial
    \\boldsymbol{{d}}^2`, and
    `\\bar{{\\boldsymbol{{\\sigma}}}}` the constant membrane stresses:

    .. math::

        \\boldsymbol{{K}}_T \\boldsymbol{{v}} = \\int \\boldsymbol{{\\Phi}}^\\top
        \\left[\\boldsymbol{{B}}^\\top \\boldsymbol{{F}} \\boldsymbol{{B}} +
        \\boldsymbol{{H}}(\\boldsymbol{{\\sigma}} + \\bar{{\\boldsymbol{{\\sigma}}}},
        \\boldsymbol{{d}})\\right] \\boldsymbol{{\\Phi}} \\boldsymbol{{v}} \\, dA

    evaluated without forming `\\boldsymbol{{K}}_T`.

    The integrals are evaluated with ``nx x ny`` Gauss-Legendre points
    over the integration domain `x_1 \\le x \\le x_2`, `y_1 \\le y \\le y_2` of
    :meth:`.Shell.integration_limits`, the whole shell by default.

    """
{COMMON}
{dcdef}
{ddcdef}
    cdef double e[6]
    cdef double de[6]
    cdef double sg[6]
    cdef double dsg[6]
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

                # S = [B]^T dsigma + [H(sigma, d)] dv
{Slines}

                for j in range(n):
{gA}
                    for i in range(m):
{fA}
                        col = col0 + DOF*(j*m + i)
{pA}
{kTv}

    return np.asarray(kTv)
'''.format(COMMON=COMMON, dcdef=cdefs(list(D_NAMES)),
           ddcdef=cdefs(['d' + nm for nm in D_NAMES]), ND=ND,
           basis=gw.basis_cdef('A', SP), FINPUT=FINPUT, POINT=POINT,
           dzero=zero_lines(D_NAMES), ddzero=zero_lines(['d' + nm for nm in D_NAMES]),
           dloop='\n'.join(gw.gradient_loop(16, alld, SP, extra=(('vs', 'd'),))),
           elines='\n'.join('                e[{0}] = {1}'.format(k, pstr(eps[k])) for k in range(NE)),
           delines='\n'.join('                de[{0}] = {1}'.format(k, pstr(de[k])) for k in range(NE)),
           sg=stresses('e', 'sg'), dsg=stresses('de', 'dsg'),
           Slines='\n'.join('                S[{0}] = {1}'.format(p, pstr(S[p])) for p in range(ND)),
           gA='\n'.join(gw.basis_read('A', 'g', 20, SP)),
           fA='\n'.join(gw.basis_read('A', 'f', 24, SP)),
           pA='\n'.join(gw.phi_lines('A', 24, SP)), kTv='\n'.join(kTv_lines))


def generate():
    path = os.path.join(REPO, 'panels', 'models', MODEL + '_num.pyx')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(numerical_module() + '\n\n' + gw.follower_block(MODEL))
    print(MODEL, 'numerical written')


if __name__ == '__main__':
    generate()
