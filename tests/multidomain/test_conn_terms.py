r"""Too few terms across an edge connection

A blade flange connected along its edge with 3 terms across its height has
no linear field across the height: the first 3 of Bardell's functions all
have a zero derivative at `\eta = +1`. Its in-plane rigid rotation and its
rotation about the joint are missing, and a T-beam locks. See the Notes of
:class:`.MultiDomain` and :meth:`.MultiDomain.check_conn_terms`.

"""
import itertools
import warnings

import numpy as np
import pytest
from scipy.linalg import eigh

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.multidomain import (ConnectionTermsWarning,
                                            _terms_linear)

E, NU = 73.e9, 0.33


def free(p):
    for e in ('x1', 'x2', 'y1', 'y2'):
        for d in ('u', 'v', 'w'):
            setattr(p, e + d, 1.)
            setattr(p, e + d + 'r', 1.)


def tbeam(n, conn_method='null-space', ycte2=0., flags={}, load='bending',
          L=0.5, b=40e-3, t=1e-3, hb=10e-3, ts=1.5e-3):
    r"""Cantilever T-beam: skin ``b x t`` and a blade ``hb x ts`` with ``n``
    terms across its height, joined at the middle of the skin with
    'BFycte', clamped at `x = 0`, unit tip force or torque

    Returns the tip deflection over the one of Euler-Bernoulli for
    ``load='bending'``, or the twist of the skin at the tip for
    ``load='torsion'``.

    """
    skin = Shell(a=L, b=b, stack=[0.], plyt=t, laminaprop=(E, NU),
                 model='plate_clpt_donnell', m=10, n=8, group='skin')
    blade = Shell(a=L, b=hb, stack=[0.], plyt=ts, laminaprop=(E, NU),
                  model='plate_clpt_donnell', m=10, n=n, group='blade')
    for p in (skin, blade):
        free(p)
        p.x1u = p.x1v = p.x1w = p.x1wr = 0.
    for k, v in flags.items():
        setattr(blade, k, v)
    conn = [dict(p1=skin, p2=blade, func='BFycte', ycte1=b/2, ycte2=ycte2)]
    if load == 'bending':
        skin.add_point_load(L, b/2, 0., 0., 1.)
    else:
        skin.add_point_load(L, b, 0., 0., 1./b)
        skin.add_point_load(L, 0., 0., 0., -1./b)
    md = MultiDomain([skin, blade], conn, conn_method=conn_method)
    c = md.linear_static()
    _, f = skin.uvw(c[skin.col_start:skin.col_end], xs=np.array([L, L, L]),
                    ys=np.array([0., b/2, b]))
    w = f['w'].ravel()
    if load == 'torsion':
        return (w[2] - w[0])/b
    # blade below the skin, from z = 0 to z = -hb
    A1, A2 = b*t, hb*ts
    zc = -A2*hb/2/(A1 + A2)
    I = b*t**3/12 + A1*zc**2 + ts*hb**3/12 + A2*(-hb/2 - zc)**2
    return w[1]/(L**3/(3*E*I))


def test_terms_linear_rule():
    """Terms needed for the linear fields admitted by the flags"""
    # free-free: the linear field needs P0..P3
    assert _terms_linear((True, True, True, True)) == 4
    # a translation removed at one edge, b*(1 + eta) or b*(1 - eta)
    assert _terms_linear((False, True, True, True)) == 4
    assert _terms_linear((True, True, False, True)) == 4
    # a derivative removed: only the constant, P0 + P2
    assert _terms_linear((True, True, True, False)) == 3
    assert _terms_linear((True, False, True, True)) == 3
    # simply supported or clamped at one edge and free at the other, or
    # simply supported at both: no linear field but zero
    assert _terms_linear((False, True, False, True)) == 0
    assert _terms_linear((False, False, True, True)) == 0
    assert _terms_linear((True, True, False, False)) == 0
    for flags in itertools.product((True, False), repeat=4):
        t1, r1, t2, r2 = flags
        if r1 and r2 and (t1 or t2):
            expected = 4
        elif t1 and t2:
            expected = 3
        else:
            expected = 0
        assert _terms_linear(flags) == expected


@pytest.mark.parametrize('n, zero_modes', [(2, 0), (3, 4), (4, 6), (5, 6)])
def test_free_flange_rigid_modes(n, zero_modes):
    """A free plate with 3 terms along y misses 2 of its 6 rigid modes, the
    rotations about z (u linear in y) and about x (w linear in y)"""
    p = Shell(a=0.1, b=3.5e-3, stack=[0.], plyt=1.5e-3, laminaprop=(E, NU),
              model='plate_clpt_donnell', m=8, n=n)
    free(p)
    ev = eigh(p.calc_kC().toarray(), eigvals_only=True)
    assert np.sum(ev < 1e-9*ev.max()) == zero_modes


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
@pytest.mark.parametrize('ycte2', [0., 10e-3])
def test_tbeam_locking(conn_method, ycte2):
    """3 terms across the blade: 137 times too stiff; 4 and 5: Euler-
    Bernoulli within 0.3 %, whichever edge of the blade is joined"""
    with pytest.warns(ConnectionTermsWarning, match=r"n = 3 < 4 for u, v, w"):
        r3 = tbeam(3, conn_method, ycte2)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConnectionTermsWarning)
        r4 = tbeam(4, conn_method, ycte2)
        r5 = tbeam(5, conn_method, ycte2)
    assert r3 < 0.008
    assert r4 == pytest.approx(0.9978, abs=5e-4)
    assert r5 == pytest.approx(r4, rel=1e-6)


def test_tbeam_locking_fields():
    """Which field locks: 4 terms with the derivative at the free edge
    removed, ``y2ur = 0`` (u) locks the bending like 3 terms, ``y2wr = 0``
    (w) the torsion, ``y2vr = 0`` (v) stiffens the bending by a few
    percent; the flags exclude these fields, no warning"""
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConnectionTermsWarning)
        bending = {k: tbeam(4, flags={k: 0.})
                   for k in ('y2ur', 'y2vr', 'y2wr')}
        twist = {k: tbeam(4, flags={k: 0.}, load='torsion')
                 for k in ('y2ur', 'y2wr')}
        twist4 = tbeam(4, load='torsion')
    assert bending['y2ur'] < 0.008
    assert 0.95 < bending['y2vr'] < 0.98
    assert bending['y2wr'] == pytest.approx(0.9978, abs=5e-4)
    assert twist['y2ur'] == pytest.approx(twist4, rel=1e-3)
    assert twist['y2wr'] < 0.06*twist4
    with pytest.warns(ConnectionTermsWarning):
        twist3 = tbeam(3, load='torsion')
    assert twist3 < 0.06*twist4
    assert tbeam(5, load='torsion') == pytest.approx(twist4, rel=1e-5)


def test_check_conn_terms():
    """The issues returned, the warning issued once, and again only when the
    terms change; area connections and released derivatives not checked"""
    a, b = 0.3, 0.1
    p1 = Shell(a=a, b=b, stack=[0.], plyt=1e-3, laminaprop=(E, NU),
               model='plate_clpt_donnell', m=6, n=6, name='skin')
    p2 = Shell(a=a, b=b, stack=[0.], plyt=1e-3, laminaprop=(E, NU),
               model='plate_clpt_donnell', m=3, n=6, name='web')
    free(p1)
    free(p2)
    conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=a, xcte2=0.)]
    with pytest.warns(ConnectionTermsWarning, match=r"'web'.*m = 3 < 4"):
        md = MultiDomain([p1, p2], conn)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConnectionTermsWarning)
        md.get_Z()
        issues = md.check_conn_terms(warn=False)
    assert len(issues) == 1
    p, func, terms, needed, fields = issues[0]
    assert p is p2 and func == 'SSxcte'
    assert (terms, needed, fields) == (3, 4, ('u', 'v', 'w'))
    # only the constant admitted for w: 3 terms enough
    p2.x2ur = p2.x2vr = p2.x2wr = 0.
    assert md.check_conn_terms(warn=False) == []
    # 2 terms: not even the constant
    p2.m = 2
    with pytest.warns(ConnectionTermsWarning, match=r"m = 2 < 3"):
        md.check_conn_terms()
    # the default flags, u = v = w = 0 at the edges, admit no linear field
    p3 = Shell(a=a, b=b, stack=[0.], plyt=1e-3, laminaprop=(E, NU),
               model='plate_clpt_donnell', m=2, n=2)
    p4 = Shell(a=a, b=b, stack=[0.], plyt=1e-3, laminaprop=(E, NU),
               model='plate_clpt_donnell', m=2, n=2)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConnectionTermsWarning)
        MultiDomain([p3, p4], [dict(p1=p3, p2=p4, func='SB')])
        MultiDomain([p3, p4], [dict(p1=p3, p2=p4, func='SSycte', ycte1=b,
                                    ycte2=0.)])
