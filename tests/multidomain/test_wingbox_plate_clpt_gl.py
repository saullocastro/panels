r"""High-AR-t5 wing box under a tip force with ``plate_clpt_gl``

``test_wingbox_linear_equals_von_karman`` verifies that the linear static
solution under the dead tip force is the same with ``plate_clpt_gl``,
``plate_clpt_tl`` and ``plate_clpt_donnell``, 22.68 mm at 25 N, and
``test_wingbox_follower_tip_force_one_increment`` the nonlinear static
solution under the follower tip force of 25 N against it. The module
also provides the model of the wing box of Elalfy, De Breuker and Castro
[1]_ under a follower tip force, :func:`wingbox`, and the tip
z-displacement of their Abaqus result (S4R, static Riks), ``ABAQUS_FIG6``,
the upper panel of Fig. 6 (p. 10) of the preprint, read from the vector
paths of the figure, about +-1 mm, for other modules.

Geometry Table 1 (p. 6) of the preprint: half-span ``L = 1010.0 mm``, chord
``W = 50.5 mm``, height ``H = 12.12 mm``, 8 ribs (9 bays), skins 1.0 mm,
spars 5.0 mm; material Table 2 (p. 6): ``E = 73.00 GPa``, ``nu = 0.33``;
clamped root, tip force (p. 7).

Model: 45 ``plate_clpt_gl`` domains, 6264 degrees of freedom: per bay two
skins (8 x 8 terms) and two spars (8 x 5) at the mid-surfaces, a rib (4 x 6)
at the end of each bay, including the tip; penalty connections with the
default stiffnesses: ``'SSxcte'`` between bays, ``'BFycte'`` spars to
skins, ``'BFxcte'`` ribs to skins (the ribs are not connected to the
spars). The tip force is a follower pressure (``follower='quadratic'``, the
complete area vector) on the upper skin over the last 10 mm of the span,
without imperfection.

The linear tip deflection, identical for ``plate_clpt_donnell``, is
453.52 mm at 500 N (22.68 mm at 25 N, the dead load).

References
----------
.. [1] Elalfy, M. H., De Breuker, R. and Castro, S. G. P., "Beam-like models
   for the post-buckling behaviour of wingbox structures", preprint, Delft
   University of Technology.

"""
import sys
sys.path.append('../..')

import numpy as np
from structsolve import Analysis

from panels.shell import Shell
from panels.multidomain import MultiDomain

E, NU = 73.e9, 0.33
L, W, H = 1.010, 50.5e-3, 12.12e-3
T_SKIN, T_SPAR = 1.e-3, 5.e-3
NBAYS = 9
X_, Y_, Z_ = np.eye(3)

#: Abaqus, Fig. 6 of [1], tip follower load [N] -> tip z-displacement [mm]
ABAQUS_FIG6 = {25: 23.3, 100: 93.2, 200: 189.6, 300: 298.9, 400: 410.0,
               500: 515.5}


def _plate(model, a, b, t, m, n, group, origin, ax, axy):
    p = Shell(a=a, b=b, r=None, stack=[0.], plyt=t, laminaprop=(E, NU),
              model=model, m=m, n=n, group=group)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for dof in ('u', 'v', 'w', 'wr'):
            setattr(p, edge + dof, 1.)
    o = np.asarray(origin, float)
    p.x0, p.y0, p.z0 = o
    p.point_x = o + ax
    p.point_xy = o + axy
    return p


def wingbox(model, P, follower, dx_load=10.e-3):
    r"""The High-AR-t5 box, frames: skins (X, Y, Z), spars (X, -Z, Y), ribs
    (-Z, Y, X), with X along the span and Z up; returns ``md`` and the tip
    domain of the upper skin"""
    a = L/NBAYS
    doms, conn, prev = [], [], None
    for i in range(NBAYS):
        x0 = i*a
        top = _plate(model, a, W, T_SKIN, 8, 8, 'skin_top', (x0, 0., H/2), X_, Y_)
        bot = _plate(model, a, W, T_SKIN, 8, 8, 'skin_bot', (x0, 0., -H/2), X_, Y_)
        spars = [_plate(model, a, H, T_SPAR, 8, 5, 'spar', (x0, y, H/2), X_, -Z_)
                 for y in (0., W)]
        rib = _plate(model, H, W, T_SKIN, 4, 6, 'rib', (x0 + a, 0., H/2), -Z_, Y_)
        conn += [dict(p1=top, p2=spars[0], func='BFycte', ycte1=0., ycte2=0.),
                 dict(p1=bot, p2=spars[0], func='BFycte', ycte1=0., ycte2=H),
                 dict(p1=top, p2=spars[1], func='BFycte', ycte1=W, ycte2=0.),
                 dict(p1=bot, p2=spars[1], func='BFycte', ycte1=W, ycte2=H),
                 dict(p1=top, p2=rib, func='BFxcte', xcte1=a, xcte2=0.),
                 dict(p1=bot, p2=rib, func='BFxcte', xcte1=a, xcte2=H)]
        bay = [top, bot] + spars
        if prev is None:
            for p in bay:
                p.x1u = p.x1v = p.x1w = p.x1wr = 0.
        else:
            conn += [dict(p1=p1, p2=p2, func='SSxcte', xcte1=a, xcte2=0.)
                     for p1, p2 in zip(prev, bay)]
        prev = bay
        doms += bay + [rib]
    top.add_pressure_load(P/(W*dx_load), x1=a - dx_load, follower=follower,
                          cte=False)
    md = MultiDomain(doms, conn, conn_method='penalty')
    md.out_num_cores = 1
    return md, top


def tip_w(md, top, c):
    r"""Mean z-displacement of the tip edge of the upper skin"""
    return md.uvw(c, None, gridx=3, gridy=3, eval_panel=top)['w'][0][:, -1].mean()


def test_wingbox_linear_equals_von_karman():
    r"""Linear static solution of the box under the dead tip force: the same
    for both models, 22.68 mm at 25 N"""
    out = {}
    for model in ('plate_clpt_gl', 'plate_clpt_tl', 'plate_clpt_donnell'):
        md, top = wingbox(model, 25., False)
        an = Analysis(md.calc_fext, md.calc_fint, md.calc_kC, md.calc_kG)
        an.static(NLgeom=False, silent=True)
        out[model] = 1e3*tip_w(md, top, an.cs[0])
    assert np.isclose(out['plate_clpt_gl'], out['plate_clpt_donnell'], rtol=1e-10)
    assert np.isclose(out['plate_clpt_tl'], out['plate_clpt_donnell'], rtol=1e-10)
    assert np.isclose(out['plate_clpt_gl'], 22.68, atol=0.01)


def test_wingbox_follower_tip_force_one_increment():
    r"""Geometrically nonlinear static analysis of the box under the
    follower tip force (``follower='quadratic'``) of 25 N in one increment
    of the Newton-Raphson method: in the linear range the tip deflection is
    that of the linear solution, 453.52/20 mm, within 0.2 %; 22.6753 mm is
    the value of the code"""
    md, top = wingbox('plate_clpt_gl', 25., 'quadratic')
    an = Analysis(md.calc_fext, md.calc_fint, md.calc_kC, md.calc_kG)
    an.initialInc = an.maxInc = 1.
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    w = 1e3*tip_w(md, top, an.cs[-1])
    assert np.isclose(w, 453.52/20, rtol=2e-3)
    assert np.isclose(w, 22.6753, rtol=1e-4)
