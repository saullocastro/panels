r"""Natural frequencies of bays with :class:`.BladeStiff2D` stiffeners

The tests verify:

- the lowest natural frequency of a sector of the internally stiffened
  cylinder (shell IS) of Sewall and Naumann (1968), simply supported, against
  the two measured frequencies of the mode `(m, n) = (1, 6)` of their Table
  II and against the :class:`.MultiDomain` assembly of the same sector with
  the blades as plate domains connected with ``'BFycte'``, with the default
  end flags of the flanges and with the circumferential displacement of the
  flange ends freed;
- the lowest natural frequency of a flat bay with one deep blade against the
  :class:`.MultiDomain` assembly of the same bay;
- the default flags of the flange of :class:`.BladeStiff2D`: free edges
  `y_f = 0` and `y_f = b_f`, and ends `x = 0, a` with the flags of the bay
  through the connection at the root, `u_f = u`, `v_f = -w`, `w_f = v`.

Notes
-----
The flange of :class:`.BladeStiff2D` is a plate whose edge `y_f = 0` is
connected to the skin and whose edge `y_f = b_f` is the free tip of the
blade. Both edges are free: in bending of the blade in its own plane the
axial displacement varies linearly over the height, `u_f = u_0 - y_f
v_{f,x}`, so a flag ``0`` of `u_{f,y}` at an edge would impose a shear
strain `\gamma_{xy} = v_{f,x}` there, which more terms only confine to a
thinner layer. With the flags ``0`` of `u_{f,y}` and `v_{f,y}` at both
edges, the sector of Sewall and Naumann below gives 746 Hz with 10 by 8
terms of the skin and 10 by 5 of each flange and 484 Hz with 14 by 12 and
14 by 7, and the :class:`.MultiDomain` sector with the same flags of its
blades 744, 484 and 367 Hz with 12, 14 and 16 terms along `x`, against 138
and 139 Hz with free edges (the count of the eigenvalues of `\mathbf{K} -
\omega^2 \mathbf{M}` below 400 Hz is zero with the flags ``0``, apart from
the rigid axial translation).

"""
import numpy as np
import pytest
from structsolve import freq

from panels.multidomain import MultiDomain
from panels.shell import Shell
from panels.stiffpanelbay import StiffPanelBay

IN = 0.0254
LBF = 4.4482216152605
KEYS = ('u', 'ur', 'v', 'vr', 'w', 'wr')


def _flags(obj, edges, flags):
    for e in edges:
        for k, f in zip(KEYS, flags):
            setattr(obj, e + k, f)


def _isotropic(E, nu):
    G = E/(2*(1 + nu))
    return (E, E, nu, G, G, G)


def _f1(K, M):
    eigvals, _ = freq(K, M, silent=True, num_eigvalues=8)
    f = np.sort(np.sqrt(np.abs(eigvals.real)))/(2*np.pi)
    return f[f > 1.][0]


def test_flange_edges_free():
    """The edges `y_f = 0` and `y_f = b_f` of the flange are free"""
    bay = StiffPanelBay()
    bay.a, bay.b, bay.r = 0.4, 0.2, None
    bay.stack = [0.]
    bay.plyt = 1.e-3
    bay.laminaprop = _isotropic(70.e9, 0.3)
    bay.rho = 2700.
    bay.m, bay.n = 6, 6
    bay.add_panel(0, 0.1)
    bay.add_panel(0.1, 0.2)
    s = bay.add_bladestiff2d(ys=0.1, bf=0.02, fstack=[0.], fplyt=1.e-3,
                             flaminaprop=bay.laminaprop, mf=4, nf=4)
    for e in ('y1', 'y2'):
        for k in KEYS:
            assert getattr(s.flange, e + k) == 1.


def test_flange_end_flags_follow_bay():
    """The end flags of the flange are those of the bay, `u_f` from `u`,
    `v_f` from `w`, `w_f` from `v`, copied when the stiffener is created,
    and a flag of the flange changed afterwards is kept"""
    for bay_x1 in [(0., 0., 0., 0., 0., 1.), (1., 1., 0., 1., 0., 1.),
                   (1., 0., 1., 0., 0., 0.)]:
        bay = StiffPanelBay()
        bay.a, bay.b, bay.r = 0.4, 0.2, None
        bay.stack = [0.]
        bay.plyt = 1.e-3
        bay.laminaprop = _isotropic(70.e9, 0.3)
        bay.rho = 2700.
        bay.m, bay.n = 6, 6
        _flags(bay, ('x1',), bay_x1)
        _flags(bay, ('x2',), bay_x1[::-1])
        bay.add_panel(0, 0.1)
        bay.add_panel(0.1, 0.2)
        s = bay.add_bladestiff2d(ys=0.1, bf=0.02, fstack=[0.], fplyt=1.e-3,
                                 flaminaprop=bay.laminaprop, mf=4, nf=4)
        for e in ('x1', 'x2'):
            for kf, kb in (('u', 'u'), ('v', 'w'), ('w', 'v')):
                for r in ('', 'r'):
                    assert (getattr(s.flange, e + kf + r)
                            == getattr(bay, e + kb + r))
        s.flange.x1w = s.flange.x2w = 1.
        bay.calc_kC(silent=True)
        assert s.flange.x1w == s.flange.x2w == 1.


def _sewall_sector(blades, m, n, mf, nf, blade_x=None):
    r"""Sector `0 \le \theta \le \pi/6` of the shell IS of Sewall and
    Naumann (1968), simply supported, with its 5 stringers; ``blade_x``, the
    end flags of the blades, is required for the :class:`.MultiDomain`
    sector and replaces the default end flags of the :class:`.BladeStiff2D`
    flanges"""
    R = 9.535*IN
    h = 0.0266*IN
    tb = 0.100*IN
    hb = (0.0266/2 + 0.2754)*IN
    L = 24.00*IN
    rho = 2.54e-4*LBF/IN**4
    lam = _isotropic(1.e7*LBF/IN**2, 0.315)
    b = R*np.pi/6
    ys = [(k + 0.5)*b/5 for k in range(5)]
    skin_x = (1., 1., 0., 1., 0., 1.)
    if blades == 'BladeStiff2D':
        bay = StiffPanelBay()
        bay.a, bay.b, bay.r = L, b, R
        bay.stack = [0.]
        bay.plyt = h
        bay.laminaprop = lam
        bay.rho = rho
        bay.m, bay.n = m, n
        _flags(bay, ('x1', 'x2'), skin_x)
        _flags(bay, ('y1', 'y2'), (1., 1., 0., 1., 1., 0.))
        edges = [0.] + ys + [b]
        for y1, y2 in zip(edges[:-1], edges[1:]):
            bay.add_panel(y1, y2)
        for y in ys:
            s = bay.add_bladestiff2d(ys=y, bf=hb, fstack=[0.], fplyt=tb,
                                     flaminaprop=lam, mf=mf, nf=nf)
            if blade_x is not None:
                _flags(s.flange, ('x1', 'x2'), blade_x)
        return _f1(bay.calc_kC(silent=True), bay.calc_kM(silent=True))
    edges = [0.] + ys + [b]
    skin = []
    for i, (y1, y2) in enumerate(zip(edges[:-1], edges[1:])):
        p = Shell(group='skin', x0=0., y0=y1, a=L, b=y2 - y1, r=R, m=m,
                  n=max(4, n//2) if i in (0, 5) else n, plyt=h, stack=[0.],
                  laminaprop=lam, rho=rho, model='cylshell_clpt_sanders')
        _flags(p, ('x1', 'x2'), skin_x)
        _flags(p, ('y1', 'y2'), (1.,)*6)
        skin.append(p)
    skin[0].y1v = skin[0].y1wr = skin[-1].y2v = skin[-1].y2wr = 0.
    conns = [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0.)
             for p1, p2 in zip(skin[:-1], skin[1:])]
    panels = list(skin)
    for k in range(5):
        bl = Shell(group='blade', x0=0., y0=ys[k], a=L, b=hb, m=mf, n=nf,
                   plyt=tb, stack=[0.], laminaprop=lam, rho=rho,
                   model='plate_clpt_donnell')
        _flags(bl, ('x1', 'x2'), blade_x)
        _flags(bl, ('y1', 'y2'), (1.,)*6)
        panels.append(bl)
        conns.append(dict(p1=skin[k + 1], p2=bl, func='BFycte', ycte1=0.,
                          ycte2=0.))
    md = MultiDomain(panels, conns, conn_method='null-space')
    return _f1(md.reduce(md.calc_kC(silent=True)),
               md.reduce(md.calc_kM(silent=True)))


@pytest.mark.slow
def test_sewall1968_internally_stiffened_sector():
    r"""Lowest frequency of the shell IS of Sewall and Naumann (1968), SS

    The sector of `\pi/6` between two planes of symmetry through the middle
    of a bay, with 5 of the 60 stringers as :class:`.BladeStiff2D` flanges,
    against the measured frequencies of the mode `(1, 6)`, 137.0 and 147.0
    Hz (Sewall and Naumann 1968, Table II), and against the
    :class:`.MultiDomain` sector with skin panels between the stringers and
    the blades as plate domains, ``'BFycte'``, with the same end flags of
    the blades, within 1.5 %: with the default end flags of the flanges,
    those of the bay, and with their circumferential displacement `w_f`
    freed by the user.

    Notes
    -----
    Dimensions of Table I (p. 30) of [1]_: radius 9.535 in, skin 0.0266 in,
    stringers 0.100 in wide and 0.2754 in high from the skin surface, 1.00
    in apart, length between the supports 24.00 in, `E = 10^7` psi, `\nu =
    0.315`, density `2.54 \times 10^{-4}` lb-s^2/in^4. The flange is a
    plate of thickness 0.100 in from the middle surface of the skin to the
    tip of the stringer, `h/2 + 0.2754` in, inside the skin. The planes of
    symmetry `\theta = 0` and `\pi/6` have `v = w_{,y} = 0`, which keeps the
    modes with `n` a multiple of 6. The ends `x = 0, L` of the skin have `v
    = w = 0` with `u` free (knife-edge rings). The ends of the flanges take
    by default the flags of the bay through the connection at the root,
    `u_f` free and `v_f = w_f = 0`, the derivatives free: the ring at the
    stringer tips holds them radially, `v_f = 0`, and the default also
    circumferentially, `w_f = 0`; the second model frees `w_f`. The table
    gives the dual resonance of the mode `(1, 6)` as 137.0 and 147.0 Hz
    (p. 31); the models, without the fillets, which stiffen the skin around
    the circumference, lie between them. With the default flags: 140.0 Hz
    with the terms of the test and 139.98 Hz with 14 by 12 terms of the
    skin and 14 by 7 of each flange, against 139.7 Hz for the
    :class:`.MultiDomain` sector with the same flags of the blades. With
    `w_f` free: 138.0 Hz with the terms of the test and 137.96 Hz with 14
    by 12 and 14 by 7, against 139.3 Hz for the :class:`.MultiDomain`
    sector, whose connections are exact (null space), with the terms of the
    test and 139.1 Hz with 16 terms along `x`, 10 across a bay and 8 across
    a blade. With the end flags `u_f = u_{f,x} = v_f = v_{f,x} = w_f = 0`,
    `w_{f,x}` free, which do not follow the bay, the :class:`.BladeStiff2D`
    sector gives 234.2 Hz with the terms of the test.

    References
    ----------
    .. [1] Sewall, J. L., and Naumann, E. C., An experimental and
       analytical vibration study of thin cylindrical shells with and
       without longitudinal stiffeners, NASA TN D-4705, 1968.

    """
    blade_x = (1., 1., 0., 1., 0., 1.)
    f_bay = _sewall_sector('BladeStiff2D', 10, 8, 10, 5)
    f_md = _sewall_sector('MultiDomain', 12, 6, 12, 4, blade_x)
    assert 137.0 <= f_bay <= 147.0
    assert abs(f_bay - f_md) <= 0.015*f_md
    blade_x = (1., 1., 0., 1., 1., 1.)
    f_bay = _sewall_sector('BladeStiff2D', 10, 8, 10, 5, blade_x)
    f_md = _sewall_sector('MultiDomain', 12, 6, 12, 4, blade_x)
    assert 137.0 <= f_bay <= 147.0
    assert abs(f_bay - f_md) <= 0.015*f_md


@pytest.mark.slow
def test_flat_bay_deep_blade_against_multidomain():
    r"""Lowest frequency of a flat bay with one deep blade

    A simply supported aluminium bay, 0.6 x 0.3 m, 2 mm thick, `w = 0` and
    the in-plane displacements free at its edges, with a blade of 18 x 2 mm
    at mid-width as a :class:`.BladeStiff2D` flange whose ends have only
    `v_f = 0` (normal to the skin), the default end flags of the flange on
    this bay, against the
    :class:`.MultiDomain` assembly of two skin plates and a plate blade
    connected with ``'BFycte'``, within 0.5 %: 96.57 and 96.40 Hz with
    these terms (216.5 Hz for the bay with the flags ``0`` of `u_{f,y}` and
    `v_{f,y}` at the edges of the flange).

    """
    a, b, h, bf, tf = 0.6, 0.3, 2.e-3, 18.e-3, 2.e-3
    lam = _isotropic(70.e9, 0.33)
    rho = 2700.
    skin_flags = (1., 1., 1., 1., 0., 1.)
    blade_x = (1., 1., 0., 1., 1., 1.)
    bay = StiffPanelBay()
    bay.a, bay.b, bay.r = a, b, None
    bay.stack = [0.]
    bay.plyt = h
    bay.laminaprop = lam
    bay.rho = rho
    bay.m, bay.n = 10, 10
    _flags(bay, ('x1', 'x2', 'y1', 'y2'), skin_flags)
    bay.add_panel(0, b/2)
    bay.add_panel(b/2, b)
    s = bay.add_bladestiff2d(ys=b/2, bf=bf, fstack=[0.], fplyt=tf,
                             flaminaprop=lam, mf=10, nf=6)
    for e in ('x1', 'x2'):
        assert tuple(getattr(s.flange, e + k) for k in KEYS) == blade_x
    f_bay = _f1(bay.calc_kC(silent=True), bay.calc_kM(silent=True))

    p1 = Shell(group='skin', x0=0, y0=0, a=a, b=b/2, m=10, n=8, plyt=h,
               stack=[0.], laminaprop=lam, rho=rho, model='plate_clpt_donnell')
    p2 = Shell(group='skin', x0=0, y0=b/2, a=a, b=b/2, m=10, n=8, plyt=h,
               stack=[0.], laminaprop=lam, rho=rho, model='plate_clpt_donnell')
    _flags(p1, ('x1', 'x2', 'y1'), skin_flags)
    _flags(p1, ('y2',), (1.,)*6)
    _flags(p2, ('x1', 'x2', 'y2'), skin_flags)
    _flags(p2, ('y1',), (1.,)*6)
    bl = Shell(group='blade', x0=0, y0=b/2, a=a, b=bf, m=10, n=6, plyt=tf,
               stack=[0.], laminaprop=lam, rho=rho, model='plate_clpt_donnell')
    _flags(bl, ('x1', 'x2'), blade_x)
    _flags(bl, ('y1', 'y2'), (1.,)*6)
    conns = [dict(p1=p1, p2=p2, func='SSycte', ycte1=b/2, ycte2=0.),
             dict(p1=p2, p2=bl, func='BFycte', ycte1=0., ycte2=0.)]
    md = MultiDomain([p1, p2, bl], conns, conn_method='null-space')
    f_md = _f1(md.reduce(md.calc_kC(silent=True)),
               md.reduce(md.calc_kM(silent=True)))
    assert abs(f_bay - f_md) <= 0.005*f_md
