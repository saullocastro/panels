r"""
Displacement field of the Ritz approximation (replaces fuvw.nb)

The displacement field of every model is built with Bardell's functions
`f_i(\xi)` along `x` and `g_j(\eta)` along `y`, with `\xi = 2x/a - 1` and
`\eta = 2y/b - 1`, each field having its own boundary flags. With the Ritz
constants of the term `(i, j)` stored at ``DOF*(j*m + i)``, the fields
`\{u, v, w, \phi_x, \phi_y\}` at a point are

.. math::

    \{u, v, w, \phi_x, \phi_y\}^T = \sum_{j=0}^{n-1} \sum_{i=0}^{m-1}
        [g_{ij}] \{c_{ij}\}

where the matrix `[g_{ij}]`, with 5 rows and ``DOF`` columns, is derived
below for:

- ``'clpt_donnell'``: the classical laminated plate theory (CLPT) with
  Donnell's kinematics, ``DOF = 3``, `\phi_x = -w_{,x}` and `\phi_y =
  -w_{,y}`, used by ``'plate_clpt_donnell'`` and ``'cylshell_clpt_donnell'``;
- ``'clpt_sanders'``: the CLPT with the Sanders-Koiter kinematics, ``DOF =
  3``, `\phi_y = -w_{,y} + v/r`, used by ``'cylshell_clpt_sanders'``;
- ``'sdt'``: the first-order and third-order shear deformation theories
  (FSDT and TSDT), ``DOF = 5``, with the rotations `\phi_x, \phi_y` as
  independent fields, used by ``'plate_fsdt_donnell'``,
  ``'plate_tsdt_donnell'``, ``'cylshell_fsdt_donnell'`` and
  ``'cylshell_tsdt_donnell'``;
- ``'sdt_sanders'``: the FSDT and TSDT with the Sanders-Koiter kinematics,
  ``DOF = 5``, whose rotation of the normal about `x` is `\Phi_y = \phi_y +
  v/r`, used by ``'cylshell_fsdt_sanders'`` and ``'cylshell_tsdt_sanders'``.

The displacements through the thickness of the last two are

.. math::

    u(z) = u + z \phi_x - c_1 z^3 (\phi_x + w_{,x}) \qquad
    v(z) = v + z \Phi_y - c_1 z^3 (\phi_y + w_{,y}) \qquad
    w(z) = w

with `c_1 = 0` for the FSDT and `c_1 = 4/(3 h^2)` for the TSDT, and `\Phi_y =
\phi_y` for ``'sdt'``.

The matrices `[g_{ij}]` are those of the function ``fg`` of the field modules
:mod:`panels.models.clpt_field` and :mod:`panels.models.fsdt_tsdt_field`,
whose function ``fuvw`` evaluates the sum above. Running this script prints
and writes the expressions to ``./output_expressions_python/`` and checks
them numerically against ``fg``.

"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sympy import (Matrix as M, Rational, Symbol, diff, factorial,
                   factorial2, lambdify, var, zeros)

var('xi, eta, a, b, r, z, c1')

FIELDS = ('u', 'v', 'w', 'phix', 'phiy')


def bardell(nmax=30):
    r"""Bardell's functions, the first four multiplied by the flags of the
    translation and rotation at `\xi = -1` and `\xi = +1`"""
    t1, r1, t2, r2 = var('t1, r1, t2, r2')
    f = [t1*(Rational(1, 2) - Rational(3, 4)*xi + Rational(1, 4)*xi**3),
         r1*(Rational(1, 8) - Rational(1, 8)*xi - Rational(1, 8)*xi**2 + Rational(1, 8)*xi**3),
         t2*(Rational(1, 2) + Rational(3, 4)*xi - Rational(1, 4)*xi**3),
         r2*(-Rational(1, 8) - Rational(1, 8)*xi + Rational(1, 8)*xi**2 + Rational(1, 8)*xi**3)]
    for k in range(5, nmax + 1):
        f.append(sum((-1)**n*factorial2(2*k - 2*n - 7)
                     /(2**n*factorial(n)*factorial(k - 2*n - 1))*xi**(k - 2*n - 1)
                     for n in range(0, k//2 + 1)))
    return (t1, r1, t2, r2), f


def shape_function_matrix(model):
    r"""Matrix `[g_{ij}]` in terms of the symbols ``f<field>``, ``g<field>``
    of the term `(i, j)` and of their derivatives ``f<field>xi``,
    ``g<field>eta``"""
    f = {k: Symbol('f' + k) for k in FIELDS}
    g = {k: Symbol('g' + k) for k in FIELDS}
    fwxi = Symbol('fwxi')
    gweta = Symbol('gweta')
    if model in ('clpt_donnell', 'clpt_sanders'):
        G = zeros(5, 3)
        G[0, 0] = f['u']*g['u']
        G[1, 1] = f['v']*g['v']
        G[2, 2] = f['w']*g['w']
        G[3, 2] = -(2/a)*fwxi*g['w']
        G[4, 2] = -(2/b)*f['w']*gweta
        if model == 'clpt_sanders':
            G[4, 1] = f['v']*g['v']/r
        return G
    assert model in ('sdt', 'sdt_sanders')
    G = zeros(5, 5)
    for k, name in enumerate(FIELDS):
        G[k, k] = f[name]*g[name]
    if model == 'sdt_sanders':
        # rotation of the normal Phiy = phiy + v/r
        G[4, 1] = f['v']*g['v']/r
    return G


def through_thickness(G, sanders=False):
    r"""`u(z), v(z), w(z)` of the models based on shear deformation theories,
    per Ritz constant of the term `(i, j)`, where the row of ``phiy`` of
    ``G`` is `\Phi_y`"""
    fwxi = Symbol('fwxi')
    gweta = Symbol('gweta')
    wx = M([[0, 0, (2/a)*fwxi*Symbol('gw'), 0, 0]])
    wy = M([[0, 0, (2/b)*Symbol('fw')*gweta, 0, 0]])
    phiy = G[4, :] - G[1, :]/r if sanders else G[4, :]
    uz = G[0, :] + z*G[3, :] - c1*z**3*(G[3, :] + wx)
    vz = G[1, :] + z*G[4, :] - c1*z**3*(phiy + wy)
    return M([uz, vz, G[2, :]])


def check(model, panels_models, num_checks=3, seed=0):
    r"""Compares `[g_{ij}]` with ``fg`` of the field module of panels"""
    from panels.shell import Shell
    from panels import modelDB

    rng = np.random.default_rng(seed)
    flags, f = bardell()
    G = shape_function_matrix(model)
    dof = G.shape[1]
    fnum = [lambdify((xi,) + flags, fi) for fi in f]
    fpnum = [lambdify((xi,) + flags, diff(fi, xi)) for fi in f]
    Gnum = lambdify([Symbol(s) for s in
                     ['f' + k for k in FIELDS] + ['g' + k for k in FIELDS]
                     + ['fwxi', 'gweta', 'a', 'b', 'r']], G)
    worst = 0.
    for name in panels_models:
        for _ in range(num_checks):
            s = Shell(model=name, a=0.7, b=0.4, stack=[0, 90], plyt=1e-3,
                      laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
                      m=6, n=5)
            s.r = 1.3 if 'cylshell' in name else None
            for k in FIELDS:
                for d in 'xy':
                    for e in ('1', '2'):
                        setattr(s, d + e + k, float(rng.integers(0, 2)))
                        setattr(s, d + e + k + 'r', float(rng.integers(0, 2)))
            s._rebuild()
            s._check_r()
            x, y = rng.uniform(0, s.a), rng.uniform(0, s.b)
            g_panels = np.zeros((5, s.get_size()))
            modelDB.db[name]['field'].fg(g_panels, x, y, s)
            ksi, et = 2*x/s.a - 1, 2*y/s.b - 1
            g_sympy = np.zeros_like(g_panels)
            for j in range(s.n):
                for i in range(s.m):
                    vals = {}
                    for k in FIELDS:
                        bx = [getattr(s, 'x' + e + k + t) for e, t in
                              (('1', ''), ('1', 'r'), ('2', ''), ('2', 'r'))]
                        by = [getattr(s, 'y' + e + k + t) for e, t in
                              (('1', ''), ('1', 'r'), ('2', ''), ('2', 'r'))]
                        vals['f' + k] = fnum[i](ksi, *bx)
                        vals['g' + k] = fnum[j](et, *by)
                        if k == 'w':
                            vals['fwxi'] = fpnum[i](ksi, *bx)
                            vals['gweta'] = fpnum[j](et, *by)
                    args = ([vals['f' + k] for k in FIELDS]
                            + [vals['g' + k] for k in FIELDS]
                            + [vals['fwxi'], vals['gweta'], s.a, s.b,
                               s.r if s.r else 1.])
                    col = dof*(j*s.m + i)
                    g_sympy[:, col:col + dof] = np.array(Gnum(*args), dtype=float)
            scale = max(np.abs(g_panels).max(), 1.)
            worst = max(worst, np.abs(g_sympy - g_panels).max()/scale)
    return worst


if __name__ == '__main__':
    outdir = './output_expressions_python/'
    os.makedirs(outdir, exist_ok=True)
    cases = {
        'clpt_donnell': ['plate_clpt_donnell', 'cylshell_clpt_donnell'],
        'clpt_sanders': ['cylshell_clpt_sanders'],
        'sdt': ['plate_fsdt_donnell', 'plate_tsdt_donnell',
                'cylshell_fsdt_donnell', 'cylshell_tsdt_donnell'],
        'sdt_sanders': ['cylshell_fsdt_sanders', 'cylshell_tsdt_sanders'],
        }
    for model, panels_models in cases.items():
        G = shape_function_matrix(model)
        lines = ['# [g_ij] of the %s kinematics, rows u, v, w, phix, phiy' % model]
        for k in range(G.shape[0]):
            for l in range(G.shape[1]):
                if G[k, l] != 0:
                    lines.append('g[%d, col+%d] = %s' % (k, l, G[k, l]))
        if model.startswith('sdt'):
            U = through_thickness(G, sanders=model == 'sdt_sanders')
            lines.append('# u(z), v(z), w(z)')
            for k, name in enumerate(('uz', 'vz', 'wz')):
                for l in range(U.shape[1]):
                    if U[k, l] != 0:
                        lines.append('%s[col+%d] = %s' % (name, l, U[k, l]))
        out = '\n'.join(lines) + '\n'
        print(out)
        with open(outdir + 'fuvw_%s.txt' % model, 'w') as f:
            f.write(out)
        err = check(model, panels_models)
        print('%s: max relative difference to panels fg = %.2e' % (model, err))
        assert err < 1e-12
