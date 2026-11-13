r"""Product of the tangent stiffness matrix by a vector, without assembly

For the models of ``MODELS``, the CLPT models with a geometrically
non-linear formulation, the generalized strains at a point are, see the
theory of each model,

.. math::

    \boldsymbol{\varepsilon}(\boldsymbol{c}) = \left( \boldsymbol{B}_L + \tfrac{1}{2} \boldsymbol{B}_{NL} \right) \boldsymbol{c},
    \qquad \boldsymbol{\beta} = \boldsymbol{G} \boldsymbol{c}

where `\boldsymbol{B}_{NL}` is linear in the rotations `\boldsymbol{\beta}` of the von Kármán
terms, `\boldsymbol{\beta} = \{\beta_x, \beta_y\} = \{-w_{,x}, -w_{,y}\}` (Donnell, with
`\beta_y = -w_{,y} + v/r` for Sanders, i.e. the rotations `\{\phi_x, \phi_y\}` of
``cylshell_clpt_sanders`` for the CLPT), and enters only the membrane rows
(for ``plate_clpt_gl`` and ``cylshell_clpt_gl``, with the Green-Lagrange
membrane strains, `\boldsymbol{\beta}` are the six gradients of `u, v, w` and
`\boldsymbol{N}` below has one block per displacement, see :func:`operators`):

.. math::

    \frac{\partial \boldsymbol{B}_{NL}}{\partial \beta_x} = \begin{bmatrix} G_x \\ 0 \\
    G_y \\ 0 \end{bmatrix}, \qquad \frac{\partial \boldsymbol{B}_{NL}}{\partial \beta_y}
    = \begin{bmatrix} 0 \\ G_y \\ G_x \\ 0 \end{bmatrix}

The factor `1/2` of `\boldsymbol{B}_{NL}` gives the quadratic terms `\beta_x^2/2`,
`\beta_y^2/2` and `\beta_x \beta_y`. With the constitutive matrix `\boldsymbol{C}`,
the internal force vector is

.. math::

    \boldsymbol{F}_{int}(\boldsymbol{c}) = \int (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top \boldsymbol{\sigma}\, dA, \qquad
    \boldsymbol{\sigma} = \boldsymbol{C} \boldsymbol{\varepsilon}(\boldsymbol{c})

and its directional derivative along `\boldsymbol{v}`, i.e. the product of the
tangent stiffness matrix `\boldsymbol{K}_T = \partial \boldsymbol{F}_{int}/\partial \boldsymbol{c}` by
`\boldsymbol{v}`, is

.. math::

    \boldsymbol{K}_T \boldsymbol{v} = \int \left( (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top \boldsymbol{C} (\boldsymbol{B}_L + \boldsymbol{B}_{NL})
    \boldsymbol{v} + \boldsymbol{G}^\top \boldsymbol{N} \boldsymbol{G} \boldsymbol{v} \right) dA, \qquad \boldsymbol{N} = \begin{bmatrix}
    N_{xx} & N_{xy} \\ N_{xy} & N_{yy} \end{bmatrix}

because `\frac{d}{dt} \boldsymbol{\varepsilon}(\boldsymbol{c} + t \boldsymbol{v})|_{t=0} = (\boldsymbol{B}_L + \boldsymbol{B}_{NL})
\boldsymbol{v}`, the strains being a quadratic form whose bilinear form is `\boldsymbol{B}_L +
\boldsymbol{B}_{NL}`, and `\frac{d}{dt} \boldsymbol{B}_{NL}(\boldsymbol{\beta}(\boldsymbol{c} + t \boldsymbol{v}))^\top|_{t=0} \boldsymbol{\sigma} =
\boldsymbol{G}^\top \boldsymbol{N} \boldsymbol{G} \boldsymbol{v}` from the derivatives of `\boldsymbol{B}_{NL}` above. These are
the two parts that ``fkC_num`` (`\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL}`) and
``fkG_num`` (`\boldsymbol{K}_G(\bar{\boldsymbol{N}} + \boldsymbol{N}_L)`) assemble.

At each integration point the product needs only:

1. the rotations `\boldsymbol{\beta} = \boldsymbol{G} \boldsymbol{c}` and `\boldsymbol{G} \boldsymbol{v}`;
2. the strains `\boldsymbol{\varepsilon} = (\boldsymbol{B}_L + \tfrac{1}{2} \boldsymbol{B}_{NL}) \boldsymbol{c}`
   and the strain increment `\delta \boldsymbol{\varepsilon} = (\boldsymbol{B}_L + \boldsymbol{B}_{NL})
   \boldsymbol{v}`;
3. the stresses `\boldsymbol{\sigma} = \boldsymbol{C} \boldsymbol{\varepsilon}`, of which `\boldsymbol{N}`, and
   `\delta \boldsymbol{\sigma} = \boldsymbol{C} \delta \boldsymbol{\varepsilon}`;
4. the sum over the terms of `(\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top \delta \boldsymbol{\sigma} +
   \boldsymbol{G}^\top \boldsymbol{N} \boldsymbol{G} \boldsymbol{v}`.

Each step is a loop over the terms of the approximation, such that the cost
is about that of :func:`calc_fint`, instead of the loop over the pairs of
terms of the assembly of `\boldsymbol{K}_T`. A constant membrane stress state
`N_{xx}^{cte}`, `N_{yy}^{cte}`, `N_{xy}^{cte}`, whose geometric stiffness
:meth:`.Shell.calc_kC` adds to the tangent stiffness matrix, is added to
`\boldsymbol{N}`.

:func:`check` verifies with SymPy, for every model, that this product is the
directional derivative of the internal force vector of one term, which
extends to any number of terms since `\boldsymbol{B}` and `\boldsymbol{G}` are linear in the
degrees of freedom. :mod:`write_pyx` writes the kernels ``calc_kTv``.

"""
import os
import sys

from sympy import Matrix as M, Rational, Symbol, diag, diff, expand, symbols, zeros

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'cylshell_clpt_sanders'))
sys.path.insert(0, os.path.join(HERE, '..', 'plate_clpt_gl'))
sys.path.insert(0, os.path.join(HERE, '..', 'cylshell_clpt_gl'))

import cylshell_clpt_sanders as clpt_sanders
import plate_clpt_gl as clpt_gl
import cylshell_clpt_gl as cyl_gl

a, b, r = symbols('a, b, r')
bx, by = symbols('bx, by')
Nxx, Nyy, Nxy = symbols('Nxx, Nyy, Nxy')

#: theory, geometry and kinematics of each model
MODELS = {
    'plate_clpt_donnell': ('clpt', 'plate', 'donnell'),
    'cylshell_clpt_donnell': ('clpt', 'cylshell', 'donnell'),
    'cylshell_clpt_sanders': ('clpt', 'cylshell', 'sanders'),
    'plate_clpt_gl': ('clpt', 'plate', 'gl'),
    'cylshell_clpt_gl': ('clpt', 'cylshell', 'gl'),
}


def _clpt_donnell(geometry):
    r"""`\boldsymbol{B}_L + \boldsymbol{B}_{NL}` and `\boldsymbol{G}` of the CLPT with Donnell kinematics, as in
    ``plate_clpt_donnell.py`` and ``cylshell_clpt_donnell.py`` with
    `\beta_x = -(2/a) w_{,\xi}`, `\beta_y = -(2/b) w_{,\eta}`"""
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
    su_xi, su_eta = row('u', 'fAu_xi', 'gAu'), row('u', 'fAu', 'gAu_eta')
    sv_xi, sv_eta = row('v', 'fAv_xi', 'gAv'), row('v', 'fAv', 'gAv_eta')
    sw_xi, sw_eta = row('w', 'fAw_xi', 'gAw'), row('w', 'fAw', 'gAw_eta')
    sw_xixi = row('w', 'fAw_xixi', 'gAw')
    sw_xieta = row('w', 'fAw_xi', 'gAw_eta')
    sw_etaeta = row('w', 'fAw', 'gAw_etaeta')
    Gx = -(2/a)*sw_xi
    Gy = -(2/b)*sw_eta
    B = M([(2/a)*su_xi + bx*Gx,
           (2/b)*sv_eta + rinv*sw + by*Gy,
           (2/b)*su_eta + (2/a)*sv_xi + bx*Gy + by*Gx,
           -(2/a)**2*sw_xixi,
           -(2/b)**2*sw_etaeta,
           -2*(2/a)*(2/b)*sw_xieta])
    return B, M([Gx, Gy])


def _clpt_gl(geometry):
    r"""`\boldsymbol{B}_L + \boldsymbol{B}_{NL}`, `\boldsymbol{G}` and the symbols of
    `\boldsymbol{\beta}` of the CLPT with Green-Lagrange membrane strains,
    ``plate_clpt_gl.py``, whose `\boldsymbol{\beta}` are the six gradients
    `u_{,x}, u_{,y}, v_{,x}, v_{,y}, w_{,x}, w_{,y}`, or ``cylshell_clpt_gl.py``,
    whose `\boldsymbol{\beta}` are the six local gradients `u_{,x}, u_{,y},
    v_{,x}, v_{,y} + w/r, w_{,x}, w_{,y} - v/r`"""
    if geometry == 'plate':
        m = clpt_gl.build()
        return m['BA'], m['GA'], clpt_gl.GRAD
    m = cyl_gl.build()
    #NOTE the radius of cylshell_clpt_gl.py is a positive symbol
    rs = {cyl_gl.r: r}
    return m['BA'].xreplace(rs), m['GA'].xreplace(rs), cyl_gl.BETA


def operators(model):
    r"""`\boldsymbol{B}_L + \boldsymbol{B}_{NL}`, `\boldsymbol{G}`, the rotation symbols `\boldsymbol{\beta}`, the
    membrane stress matrix `\boldsymbol{N}` of `\boldsymbol{G}^\top \boldsymbol{N} \boldsymbol{G}`, the
    fields and the number of generalized strains of ``model``

    `\boldsymbol{\beta}` has two rotations, `(\beta_x, \beta_y)`, and `\boldsymbol{N}` is
    `[[N_{xx}, N_{xy}], [N_{xy}, N_{yy}]]`, except for ``'plate_clpt_gl'`` and
    ``'cylshell_clpt_gl'``, whose `\boldsymbol{\beta}` are the six gradients
    ``plate_clpt_gl.GRAD`` or ``cylshell_clpt_gl.BETA`` and `\boldsymbol{N}` the
    block-diagonal matrix with one such block per displacement `u, v, w`.

    """
    theory, geometry, kinematics = MODELS[model]
    N2 = M([[Nxx, Nxy], [Nxy, Nyy]])
    Nmat = N2
    if kinematics == 'gl':
        B, G, rot = _clpt_gl(geometry)
        fields = ('u', 'v', 'w')
        Nmat = diag(N2, N2, N2)
    elif theory == 'clpt' and kinematics == 'donnell':
        B, G = _clpt_donnell(geometry)
        rot = (bx, by)
        fields = ('u', 'v', 'w')
    else:
        B, B0, G, g5 = clpt_sanders.shape_functions('A')
        rot = (clpt_sanders.phix, clpt_sanders.phiy)
        fields = ('u', 'v', 'w')
    return dict(B=B, G=G, rot=rot, Nmat=Nmat, fields=fields, NE=B.shape[0],
                DOF=len(fields), theory=theory, geometry=geometry)


def rotation_names(model):
    r"""Names of `\boldsymbol{\beta}` and `\delta \boldsymbol{\beta}` in the kernels, ``bx, by``
    and ``dbx, dby`` for two rotations, else ``b0, b1, ...`` and ``db0,
    db1, ...``"""
    k = len(operators(model)['rot'])
    names = ['bx', 'by'] if k == 2 else ['b%d' % i for i in range(k)]
    return names, ['d' + nm for nm in names]


def tangent_product(model):
    r"""Expressions of the kernel ``calc_kTv`` at an integration point

    Returns a dict with, for each degree of freedom ``k`` of a term:

    - ``'beta'``: the rows of `\boldsymbol{G}`, giving the rotations from ``cs`` and
      from ``vs``;
    - ``'eps'``: the columns ``k`` of `\boldsymbol{B}_L + \tfrac{1}{2} \boldsymbol{B}_{NL}`, the
      strains from ``cs``;
    - ``'deps'``: the columns ``k`` of `\boldsymbol{B}_L + \boldsymbol{B}_{NL}`, the strain
      increment from ``vs``;
    - ``'kTv'``: the entry ``k`` of `(\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top \delta \boldsymbol{\sigma} +
      \boldsymbol{G}^\top \boldsymbol{N} \delta \boldsymbol{\beta}`, with the symbols ``ds0, ds1, ...`` of
      `\delta \boldsymbol{\sigma}`, ``Nxx, Nyy, Nxy`` of `\boldsymbol{N}` and ``dbx, dby`` of
      `\delta \boldsymbol{\beta} = \boldsymbol{G} \boldsymbol{v}`;

    with ``bx, by`` the rotations `\boldsymbol{\beta}` of ``cs``, see
    :func:`rotation_names` for the names of more than two rotations.

    """
    op = operators(model)
    B, G = op['B'], op['G']
    NE, DOF = op['NE'], op['DOF']
    names, dnames = rotation_names(model)
    beta = [Symbol(nm) for nm in names]
    dbeta = [Symbol(nm) for nm in dnames]
    B = B.xreplace(dict(zip(op['rot'], beta)))
    Bhalf = B.xreplace({q: q/2 for q in beta})
    ds = M(symbols('ds0:%d' % NE))
    kTv = B.T*ds + G.T*op['Nmat']*M(dbeta)
    return dict(beta=[[G[i, k] for k in range(DOF)] for i in range(len(beta))],
                eps=[[expand(Bhalf[i, k]) for k in range(DOF)]
                     for i in range(NE)],
                deps=[[expand(B[i, k]) for k in range(DOF)]
                      for i in range(NE)],
                kTv=[expand(kTv[k]) for k in range(DOF)],
                names=names, dnames=dnames, op=op)


def check(model):
    r"""Verify that ``kTv`` is the directional derivative of `\boldsymbol{F}_{int}`

    For one term with the degrees of freedom `\boldsymbol{c}`, `\boldsymbol{F}_{int} = (\boldsymbol{B}_L
    + \boldsymbol{B}_{NL})^\top \boldsymbol{C} (\boldsymbol{B}_L + \tfrac{1}{2} \boldsymbol{B}_{NL}) \boldsymbol{c}`, `\boldsymbol{\beta} =
    \boldsymbol{G} \boldsymbol{c}`, with a symbolic symmetric constitutive matrix `\boldsymbol{C}`, is
    differentiated along `\boldsymbol{v}`, and compared
    with the expressions of :func:`tangent_product`.

    """
    tp = tangent_product(model)
    op = tp['op']
    NE, DOF = op['NE'], op['DOF']
    beta_s = [Symbol(nm) for nm in tp['names']]
    half = {q: q/2 for q in beta_s}
    B = op['B'].xreplace(dict(zip(op['rot'], beta_s)))
    G = op['G']
    F = M(NE, NE, lambda i, j: Symbol('F%d_%d' % (min(i, j), max(i, j))))
    c = M(symbols('c0:%d' % DOF))
    v = M(symbols('v0:%d' % DOF))
    t = Symbol('t')
    ct = c + t*v
    beta = G*ct
    sub = dict(zip(beta_s, beta))
    eps = B.xreplace(half).xreplace(sub)*ct
    fint = B.xreplace(sub).T*F*eps
    exact = fint.applyfunc(lambda e: diff(e, t).subs(t, 0))

    beta_c = G*c
    sub_c = dict(zip(beta_s, beta_c))
    sigma = F*(B.xreplace(half).xreplace(sub_c)*c)
    deps = B.xreplace(sub_c)*v
    ds = F*deps
    dbeta = G*v
    values = {Symbol('ds%d' % i): ds[i] for i in range(NE)}
    values.update({Nxx: sigma[0], Nyy: sigma[1], Nxy: sigma[2]})
    values.update({Symbol(nm): dbeta[i] for i, nm in enumerate(tp['dnames'])})
    values.update(sub_c)
    for k in range(DOF):
        got = tp['kTv'][k].xreplace(values)
        assert expand(got - exact[k]) == 0, (model, k)
    return True


if __name__ == '__main__':
    for model in sys.argv[1:] or MODELS:
        check(model)
        print(model, 'K_T v is the directional derivative of fint: verified')
