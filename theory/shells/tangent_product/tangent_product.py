r"""Product of the tangent stiffness matrix by a vector, without assembly

For all the models with a geometrically non-linear formulation, the
generalized strains at a point are, see the theory of each model,

.. math::

    \{\varepsilon(c)\} = [B(\beta/2)] \{c\}, \qquad \{\beta\} = [G] \{c\}

where `[B(\beta)]` is linear in the rotations `\{\beta\}` of the von Kármán
terms, `\{\beta\} = \{w_{,x}, w_{,y}\}` (Donnell, `\{\beta_x, \beta_y\}` with
`\beta_y = w_{,y} - v/r` for Sanders, `\{\phi_x, \phi_y\}` with the sign
convention of ``cylshell_clpt_sanders``), and only the membrane rows of
`[B]` depend on `\{\beta\}`:

.. math::

    \frac{\partial [B]}{\partial \beta_x} = \begin{bmatrix} G_x \\ 0 \\ G_y
    \\ 0 \end{bmatrix}, \qquad \frac{\partial [B]}{\partial \beta_y} =
    \begin{bmatrix} 0 \\ G_y \\ G_x \\ 0 \end{bmatrix}

Using `\beta/2` in `[B]` gives the quadratic terms `\beta_x^2/2`,
`\beta_y^2/2` and `\beta_x \beta_y`. The internal force vector is

.. math::

    \{F_{int}(c)\} = \int [B(\beta)]^T \{\sigma\}\, dA, \qquad \{\sigma\} =
    [F] \{\varepsilon(c)\}

and its directional derivative along `\{v\}`, i.e. the product of the
tangent stiffness matrix `[K_T] = \partial \{F_{int}\}/\partial \{c\}` by
`\{v\}`, is

.. math::

    [K_T] \{v\} = \int \left( [B(\beta)]^T [F] [B(\beta)] \{v\} + [G]^T
    [N] [G] \{v\} \right) dA, \qquad [N] = \begin{bmatrix} N_{xx} & N_{xy}
    \\ N_{xy} & N_{yy} \end{bmatrix}

because `\frac{d}{dt} \{\varepsilon(c + t v)\}|_{t=0} = [B(\beta)] \{v\}`,
the strains being a quadratic form whose bilinear form is `[B(\beta)]`, and
`\frac{d}{dt} [B(\beta(c + t v))]^T|_{t=0} \{\sigma\} = [G]^T [N] [G]
\{v\}` from the derivatives of `[B]` above. These are the two parts that
``fkC_num`` (`K_{C0} + K_{0L} + K_{L0} + K_{LL} + K_{G_{NL}}`) and ``fkG_num``
(`K_G(N_0 + N_L)`) assemble.

At each integration point the product needs only:

1. the rotations `\{\beta\} = [G] \{c\}` and `[G] \{v\}`;
2. the strains `\{\varepsilon\} = [B(\beta/2)] \{c\}` and the strain
   increment `\{\delta \varepsilon\} = [B(\beta)] \{v\}`;
3. the stresses `\{\sigma\} = [F] \{\varepsilon\}`, of which `[N]`, and
   `\{\delta \sigma\} = [F] \{\delta \varepsilon\}`;
4. the sum over the terms of `[B(\beta)]^T \{\delta \sigma\} + [G]^T [N] [G]
   \{v\}`.

Each step is a loop over the terms of the approximation, such that the cost
is about that of :func:`calc_fint`, instead of the loop over the pairs of
terms of the assembly of `[K_T]`. A constant membrane stress state
`N_{xx}^{cte}`, `N_{yy}^{cte}`, `N_{xy}^{cte}`, whose geometric stiffness
:meth:`.Shell.calc_kC` adds to the tangent stiffness matrix, is added to
`[N]`.

:func:`check` verifies with SymPy, for every model, that this product is the
directional derivative of the internal force vector of one term, which
extends to any number of terms since `[B]` and `[G]` are linear in the
degrees of freedom. :mod:`write_pyx` writes the kernels ``calc_kTv``.

"""
import os
import sys

from sympy import Matrix as M, Rational, Symbol, diff, expand, symbols, zeros

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fsdt_tsdt'))
sys.path.insert(0, os.path.join(HERE, '..', 'cylshell_clpt_sanders'))

import fsdt_tsdt
import cylshell_clpt_sanders as clpt_sanders

a, b, r = symbols('a, b, r')
bx, by = symbols('bx, by')
Nxx, Nyy, Nxy = symbols('Nxx, Nyy, Nxy')

#: theory, geometry and kinematics of each model
MODELS = {
    'plate_clpt_donnell': ('clpt', 'plate', 'donnell'),
    'cylshell_clpt_donnell': ('clpt', 'cylshell', 'donnell'),
    'cylshell_clpt_sanders': ('clpt', 'cylshell', 'sanders'),
    'plate_fsdt_donnell': ('fsdt', 'plate', 'donnell'),
    'plate_tsdt_donnell': ('tsdt', 'plate', 'donnell'),
    'cylshell_fsdt_donnell': ('fsdt', 'cylshell', 'donnell'),
    'cylshell_fsdt_sanders': ('fsdt', 'cylshell', 'sanders'),
    'cylshell_tsdt_donnell': ('tsdt', 'cylshell', 'donnell'),
    'cylshell_tsdt_sanders': ('tsdt', 'cylshell', 'sanders'),
}


def _clpt_donnell(geometry):
    r"""`[B(\beta)]` and `[G]` of the CLPT with Donnell kinematics, as in
    ``plate_clpt_donnell.py`` and ``cylshell_clpt_donnell.py`` with
    `\beta_x = (2/a) w_{,\xi}`, `\beta_y = (2/b) w_{,\eta}`"""
    var = {}

    def s(name):
        if name not in var:
            var[name] = Symbol(name)
        return var[name]

    def row(field, fname, gname):
        out = zeros(1, 3)
        out[0, 'uvw'.index(field)] = s(fname)*s(gname)
        return out

    rinv = 1/r if geometry == 'cylshell' else 0
    su, sv, sw = row('u', 'fAu', 'gAu'), row('v', 'fAv', 'gAv'), row('w', 'fAw', 'gAw')
    suxi, sueta = row('u', 'fAuxi', 'gAu'), row('u', 'fAu', 'gAueta')
    svxi, sveta = row('v', 'fAvxi', 'gAv'), row('v', 'fAv', 'gAveta')
    swxi, sweta = row('w', 'fAwxi', 'gAw'), row('w', 'fAw', 'gAweta')
    swxixi = row('w', 'fAwxixi', 'gAw')
    swxieta = row('w', 'fAwxi', 'gAweta')
    swetaeta = row('w', 'fAw', 'gAwetaeta')
    Gx = (2/a)*swxi
    Gy = (2/b)*sweta
    B = M([(2/a)*suxi + bx*Gx,
           (2/b)*sveta + rinv*sw + by*Gy,
           (2/b)*sueta + (2/a)*svxi + bx*Gy + by*Gx,
           -(2/a)**2*swxixi,
           -(2/b)**2*swetaeta,
           -2*(2/a)*(2/b)*swxieta])
    return B, M([Gx, Gy])


def operators(model):
    r"""`[B(\beta)]`, `[G]`, the rotation symbols `(\beta_x, \beta_y)`, the
    fields and the number of generalized strains of ``model``"""
    theory, geometry, kinematics = MODELS[model]
    if theory == 'clpt' and kinematics == 'donnell':
        B, G = _clpt_donnell(geometry)
        rot = (bx, by)
        fields = ('u', 'v', 'w')
    elif theory == 'clpt':
        B, B0, G, g5 = clpt_sanders.shape_functions('A')
        rot = (clpt_sanders.phix, clpt_sanders.phiy)
        fields = ('u', 'v', 'w')
    else:
        S = fsdt_tsdt.shape_functions('A', theory, geometry, kinematics)
        B, G = S['B'], S['G']
        rot = (fsdt_tsdt.bx, fsdt_tsdt.by)
        fields = fsdt_tsdt.FIELDS
    return dict(B=B, G=G, rot=rot, fields=fields, NE=B.shape[0],
                DOF=len(fields), theory=theory, geometry=geometry)


def tangent_product(model):
    r"""Expressions of the kernel ``calc_kTv`` at an integration point

    Returns a dict with, for each degree of freedom ``k`` of a term:

    - ``'beta'``: the two rows of `[G]`, giving the rotations from ``cs`` and
      from ``vs``;
    - ``'eps'``: the columns ``k`` of `[B(\beta/2)]`, the strains from ``cs``;
    - ``'deps'``: the columns ``k`` of `[B(\beta)]`, the strain increment from
      ``vs``;
    - ``'kTv'``: the entry ``k`` of `[B(\beta)]^T \{\delta \sigma\} + [G]^T
      [N] \{\delta \beta\}`, with the symbols ``ds0, ds1, ...`` of
      `\{\delta \sigma\}`, ``Nxx, Nyy, Nxy`` of `[N]` and ``dbx, dby`` of
      `\{\delta \beta\} = [G] \{v\}`;

    with ``bx, by`` the rotations `\{\beta\}` of ``cs``.

    """
    op = operators(model)
    B, G, (rx, ry) = op['B'], op['G'], op['rot']
    NE, DOF = op['NE'], op['DOF']
    rename = {rx: bx, ry: by}
    B = B.xreplace(rename)
    Bhalf = B.xreplace({bx: bx/2, by: by/2})
    ds = M(symbols('ds0:%d' % NE))
    dbx, dby = symbols('dbx, dby')
    Nmat = M([[Nxx, Nxy], [Nxy, Nyy]])
    kTv = B.T*ds + G.T*Nmat*M([dbx, dby])
    return dict(beta=[[G[i, k] for k in range(DOF)] for i in range(2)],
                eps=[[expand(Bhalf[i, k]) for k in range(DOF)]
                     for i in range(NE)],
                deps=[[expand(B[i, k]) for k in range(DOF)]
                      for i in range(NE)],
                kTv=[expand(kTv[k]) for k in range(DOF)],
                op=op)


def check(model):
    r"""Verify that ``kTv`` is the directional derivative of `\{F_{int}\}`

    For one term with the degrees of freedom `\{c\}`, `\{F_{int}\} =
    [B(\beta)]^T [F] [B(\beta/2)] \{c\}`, `\{\beta\} = [G] \{c\}`, with a
    symbolic symmetric `[F]`, is differentiated along `\{v\}`, and compared
    with the expressions of :func:`tangent_product`.

    """
    tp = tangent_product(model)
    op = tp['op']
    NE, DOF = op['NE'], op['DOF']
    B = op['B'].xreplace({op['rot'][0]: bx, op['rot'][1]: by})
    G = op['G']
    F = M(NE, NE, lambda i, j: Symbol('F%d_%d' % (min(i, j), max(i, j))))
    c = M(symbols('c0:%d' % DOF))
    v = M(symbols('v0:%d' % DOF))
    t = Symbol('t')
    ct = c + t*v
    beta = G*ct
    sub = {bx: beta[0], by: beta[1]}
    eps = B.xreplace({bx: bx/2, by: by/2}).xreplace(sub)*ct
    fint = B.xreplace(sub).T*F*eps
    exact = fint.applyfunc(lambda e: diff(e, t).subs(t, 0))

    beta_c = G*c
    sub_c = {bx: beta_c[0], by: beta_c[1]}
    sigma = F*(B.xreplace({bx: bx/2, by: by/2}).xreplace(sub_c)*c)
    deps = B.xreplace(sub_c)*v
    ds = F*deps
    dbeta = G*v
    values = {Symbol('ds%d' % i): ds[i] for i in range(NE)}
    values.update({Nxx: sigma[0], Nyy: sigma[1], Nxy: sigma[2],
                   Symbol('dbx'): dbeta[0], Symbol('dby'): dbeta[1]})
    values.update(sub_c)
    for k in range(DOF):
        got = tp['kTv'][k].xreplace(values)
        assert expand(got - exact[k]) == 0, (model, k)
    return True


if __name__ == '__main__':
    for model in sys.argv[1:] or MODELS:
        check(model)
        print(model, 'K_T v is the directional derivative of fint: verified')
