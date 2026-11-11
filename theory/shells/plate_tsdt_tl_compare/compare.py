r"""Comparison of the prototypes of the total-Lagrangian TSDT, options A and B

Run from the root of the repository (built in place), ``python
theory/shells/plate_tsdt_tl_compare/compare.py``; prints the tables of the
comparison:

1. internal forces of a finite rigid rotation of a free thick plate,
   ``|fint(c)|/|KC0 c|``;
2. a thin cantilever strip bent out of its plane by a tip force, against the
   inextensible elastica;
3. a thick strip, ``L/h = 5``, against the geometrically exact (Reissner)
   beam with the bending, axial and shear stiffnesses of the strip, the
   shear stiffness calibrated on the linear tip deflection of each model;
4. the thick strip under an oblique tip force (out-of-plane and in-plane
   bending, three-dimensional rotations), model against model;
5. a thick simply supported square plate, ``a/h = 5``, under a uniform
   pressure, model against model;
6. the cost: size of the generated source and of the compiled module, and
   the time of one tangent stiffness matrix and internal force vector.

"""
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from structsolve import Analysis

from panels.shell import Shell
import panels.models as pm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'tests', 'tests_shell'))
from test_fsdt_tl import plate_rigid_rotation
from test_tsdt_tl_proto import thick_plate

A, B = 'plate_tsdt_tlA_proto', 'plate_tsdt_tlB_proto'
TSDT = (A, B, 'plate_tsdt_gl')
ALL = (A, B, 'plate_fsdt_tl', 'plate_tsdt_gl')
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')


def out(*args):
    print(*args, flush=True)


def flags(s, edges, fields, **values):
    for e in edges:
        for f in fields:
            for k, val in values.items():
                setattr(s, e + f + k, val)


# 1. rigid rotation ------------------------------------------------------
def rigid_rotation():
    out('\n1. rigid rotation of a free plate a/h = 6, |fint|/|KC0 c|')
    out('axis, angle: ' + ', '.join(ALL))
    for axis in ((1, 0, 0), (0, 1, 0), (1, 2, 3)):
        for t in (0.2, 0.1):
            row = []
            for model in ALL:
                s = thick_plate(model, free=True)
                c = plate_rigid_rotation(s, axis, t)
                f = s.calc_fint(c)
                f0 = s.calc_kC(c=np.zeros_like(c), NLgeom=True) @ c
                row.append(np.linalg.norm(f)/np.linalg.norm(f0))
            out('{0}, {1}: '.format(axis, t) + ', '.join('%.3e' % r for r in row))


# strips -----------------------------------------------------------------
def strip(model, L, b, h, alpha, inc, dirn=(0., 0., 1.), NLgeom=True, m=14,
          n=4, E=70e9):
    r"""Tip displacement ``(u, v, w)/L``, mean over the tip, of a strip
    clamped at ``x = 0`` under a tip force of constant direction ``dirn``,
    ``alpha = P L^2/EI`` with the out-of-plane ``EI``"""
    EI = E*h**3*b/12
    P = alpha*EI/L**2
    s = Shell(model=model, a=L, b=b, stack=[0], plyt=h, laminaprop=(E, 0.),
              m=m, n=n)
    flags(s, ('x2', 'y1', 'y2'), FIELDS, **{'': 1., 'r': 1.})
    flags(s, ('x1', ), FIELDS, **{'': 0., 'r': 1.})
    if '_tsdt_' in model:
        s.x1wr = 0.
    s._rebuild()
    for y in np.linspace(0, b, 5):
        s.add_point_load(L, y, P*dirn[0]/5, P*dirn[1]/5, P*dirn[2]/5, cte=False)
    an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    an.initialInc = inc
    an.maxInc = inc
    t0 = time.time()
    an.static(NLgeom=NLgeom, silent=True)
    cpu = time.time() - t0
    assert np.isclose(an.increments[-1], 1.)
    f = s.uvw(an.cs[-1], xs=np.full(5, L), ys=np.linspace(0, b, 5))[1]
    return np.array([f['u'].mean(), f['v'].mean(), f['w'].mean()])/L, cpu


def elastica(alpha):
    def shoot(th_tip):
        f = lambda s, y: [y[1], -alpha*np.cos(y[0])]
        return solve_ivp(f, [1., 0.], [th_tip, 0.], rtol=1e-11, atol=1e-13,
                         dense_output=True)
    th = brentq(lambda t: shoot(t).y[0, -1], 1e-9, 1.5)
    s = np.linspace(0, 1, 4001)
    t = shoot(th).sol(s)[0]
    return np.trapezoid(np.sin(t), s), np.trapezoid(1 - np.cos(t), s), th


def reissner(P, L, EI, EA, kGA):
    r"""Tip ``(w, -u)/L`` and rotation of the geometrically exact beam
    (Reissner): ``r' = (1 + e) a1 + g a3``, ``a1 = (cos th, sin th)``, ``a3 =
    (-sin th, cos th)``, ``e = P sin th/EA``, ``g = P cos th/kGA``, ``th' =
    M/EI``, ``M' = -P x'``, clamped at ``s = 0``, ``M(L) = 0``"""
    def rhs(s, y):
        th, M, x, z = y
        e, g = P*np.sin(th)/EA, P*np.cos(th)/kGA
        xp = (1 + e)*np.cos(th) - g*np.sin(th)
        zp = (1 + e)*np.sin(th) + g*np.cos(th)
        return [M/EI, -P*xp, xp, zp]

    def end(M0):
        return solve_ivp(rhs, [0, L], [0., M0, 0., 0.], rtol=1e-11, atol=1e-14)
    M0 = brentq(lambda M0: end(M0).y[1, -1], 1e-6*P*L, 1.0001*P*L)
    y = end(M0).y[:, -1]
    return y[3]/L, (L - y[2])/L, y[0]


def thin_strips():
    out('\n2. thin strip L = 1, b = 0.04, h = 2e-3, nu = 0, 14 x 4 terms, vs elastica')
    for alpha, inc in ((1., 0.1), (3., 0.05)):
        w_ref, u_ref, th = elastica(alpha)
        out('alpha = %g, tip rotation %.3f rad, elastica w = %.7f L, -u = %.7f L'
            % (alpha, th, w_ref, u_ref))
        for model in ALL:
            try:
                (u, v, w), cpu = strip(model, 1., 0.04, 2e-3, alpha, inc)
                out('  %-22s w = %.7f (%+.3f %%), -u = %.7f (%+.3f %%), %.1f s'
                    % (model, w, 100*(w/w_ref - 1), -u, 100*(-u/u_ref - 1), cpu))
            except Exception as ex:
                out('  %-22s failed: %r' % (model, ex))


def thick_strips():
    L, b, h, E = 1., 0.2, 0.2, 70e9
    EI, EA = E*h**3*b/12, E*h*b
    out('\n3. thick strip L = 1, b = h = 0.2 (L/h = 5), nu = 0, 14 x 4 terms, '
        'vs Reissner beam with kGA from the linear tip deflection of the model')
    for alpha, inc in ((1., 0.1), (3., 0.05)):
        P = alpha*EI/L**2
        for model in ALL:
            try:
                (_, _, wl), _ = strip(model, L, b, h, alpha, inc, NLgeom=False)
                kGA = P*L/(wl*L - P*L**3/(3*EI))
                w_ref, u_ref, th = reissner(P, L, EI, EA, kGA)
                (u, v, w), cpu = strip(model, L, b, h, alpha, inc)
                out('  alpha = %g %-22s kGA/(GA) = %.4f, tip rotation %.3f, '
                    'w = %.6f (ref %.6f, %+.3f %%), -u = %.6f (ref %.6f, %+.3f %%), %.1f s'
                    % (alpha, model, kGA/(E/2*b*h), th, w, w_ref, 100*(w/w_ref - 1),
                       -u, u_ref, 100*(-u/u_ref - 1), cpu))
            except Exception as ex:
                out('  alpha = %g %-22s failed: %r' % (alpha, model, ex))


def oblique_strips():
    out('\n4. thick strip b = h = 0.2, tip force along (0, 1, 1)/sqrt(2), '
        'alpha = 2 (out-of-plane EI), 14 x 6 terms, model against model')
    res = {}
    for model in ALL:
        try:
            res[model], cpu = strip(model, 1., 0.2, 0.2, 2., 0.05,
                                    dirn=(0., 2**-0.5, 2**-0.5), n=6)
            out('  %-22s (u, v, w)/L = (%.6f, %.6f, %.6f), %.1f s'
                % ((model,) + tuple(res[model]) + (cpu,)))
        except Exception as ex:
            out('  %-22s failed: %r' % (model, ex))
    if A in res and B in res:
        out('  A - B relative to |B|: %.3e' % (np.linalg.norm(res[A] - res[B])
                                              /np.linalg.norm(res[B])))


def thick_plate_pressure():
    out('\n5. simply supported square plate a = 1, h = 0.2, nu = 0.3, '
        'uniform pressure, 12 x 12 terms, centre w/h')
    E, nu, h = 70e9, 0.3, 0.2
    ps = (0.25e9, 0.5e9, 1.0e9, 1.5e9)
    res = {}
    for model in ALL:
        s = Shell(model=model, a=1., b=1., stack=[0], plyt=h,
                  laminaprop=(E, nu), m=12, n=12)
        flags(s, ('x1', 'x2', 'y1', 'y2'), FIELDS, **{'': 1., 'r': 1.})
        flags(s, ('x1', 'x2', 'y1', 'y2'), ('w',), **{'': 0.})
        s.x1u = 0.
        s.y1v = 0.
        s._rebuild()
        s.add_pressure_load(ps[-1], cte=False)
        an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
        an.initialInc = 1/12
        an.maxInc = 1/12
        t0 = time.time()
        try:
            an.static(NLgeom=True, silent=True)
        except Exception as ex:
            out('  %-22s failed: %r' % (model, ex))
            continue
        cpu = time.time() - t0
        ws = []
        for p in ps:
            k = int(np.argmin(np.abs(np.asarray(an.increments) - p/ps[-1])))
            ws.append(s.uvw(an.cs[k], xs=np.array([0.5]), ys=np.array([0.5]))[1]['w'][0]/h)
        res[model] = np.array(ws)
        out('  %-22s ' % model + ', '.join('p = %.2g: %.5f' % (p, w) for p, w in zip(ps, ws))
            + ', %.1f s' % cpu)
    if B in res:
        for model in res:
            out('  %-22s relative to B: ' % model
                + ', '.join('%+.3f %%' % (100*(x/y - 1)) for x, y in zip(res[model], res[B])))


def cost():
    out('\n6. cost: source and compiled sizes; time of calc_kC(NLgeom) + calc_kG '
        '+ calc_fint, plate 12 x 12 terms, best of 3')
    for model in ALL:
        mod = pm.__dict__[model + '_num']
        so = mod.__file__
        pyx = os.path.join(REPO, 'panels', 'models', model + '_num.pyx')
        lines = sum(1 for _ in open(pyx, encoding='utf-8'))
        s = thick_plate(model, m=12, n=12)
        rng = np.random.default_rng(0)
        c = 0.01*rng.standard_normal(s.get_size())
        best = {}
        for _ in range(3):
            for name, fun in (('kC', lambda: s.calc_kC(c=c, NLgeom=True)),
                              ('kG', lambda: s.calc_kG(c=c, NLgeom=True)),
                              ('fint', lambda: s.calc_fint(c))):
                t0 = time.time()
                fun()
                best[name] = min(best.get(name, 1e9), time.time() - t0)
        out('  %-22s pyx %d lines %d kB, compiled %d kB, kC %.3f s, kG %.3f s, fint %.3f s'
            % (model, lines, os.path.getsize(pyx)//1024, os.path.getsize(so)//1024,
               best['kC'], best['kG'], best['fint']))


if __name__ == '__main__':
    cases = sys.argv[1:] or ['1', '2', '3', '4', '5', '6']
    funcs = {'1': rigid_rotation, '2': thin_strips, '3': thick_strips,
             '4': oblique_strips, '5': thick_plate_pressure, '6': cost}
    for k in cases:
        funcs[k]()
