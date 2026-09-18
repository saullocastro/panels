import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
from structsolve import freq, lb

from panels.shell import Shell
from panels.stiffener import BladeStiff1D, BladeStiff2D
from panels.stiffpanelbay import StiffPanelBay
from panels.stiffpanelbay.stiffpanelbay import load


laminaprop = (105.e9, 10.5e9, 0.25, 5.25e9, 5.25e9, 5.25e9)
stack = [0, 90, 90, 0, 0]
plyt = 0.134e-3
tskin = len(stack)*plyt
b = 100*tskin

MF, NF = 6, 5


def _bay(stiffener=None, m=8, n=8):
    """Two-panel bay with an optional stiffener at mid-width"""
    bay = StiffPanelBay()
    bay.a = 2*b
    bay.b = b
    bay.r = None
    bay.model = 'plate_clpt_donnell'
    bay.stack = stack
    bay.plyt = plyt
    bay.laminaprop = laminaprop
    bay.rho = 1600.
    bay.m = m
    bay.n = n
    bay.add_panel(0, b/2, Nxx=-1.)
    bay.add_panel(b/2, b, Nxx=-1.)
    if stiffener == '1d':
        bay.add_bladestiff1d(ys=b/2, bf=9*tskin, fstack=stack, fplyt=plyt,
                             flaminaprop=laminaprop)
    elif stiffener == '1d_base':
        bay.add_bladestiff1d(ys=b/2, bb=10*tskin, bstack=stack, bplyt=plyt,
                             blaminaprop=laminaprop, bf=9*tskin, fstack=stack,
                             fplyt=plyt, flaminaprop=laminaprop)
    elif stiffener == '2d':
        bay.add_bladestiff2d(ys=b/2, bf=9*tskin, fstack=stack, fplyt=plyt,
                             flaminaprop=laminaprop, mf=MF, nf=NF)
    elif stiffener == '2d_base':
        bay.add_bladestiff2d(ys=b/2, bb=10*tskin, bstack=stack, bplyt=plyt,
                             blaminaprop=laminaprop, bf=9*tskin, fstack=stack,
                             fplyt=plyt, flaminaprop=laminaprop, mf=MF, nf=NF)
    return bay


def _lb(bay):
    eigvals, eigvecs = lb(bay.calc_kC(silent=True), bay.calc_kG(silent=True),
                          silent=True, num_eigvalues=3)
    return eigvals, eigvecs


def test_get_size_includes_bladestiff2d_flange():
    skin = 3*8*8
    assert _bay().get_size() == skin
    assert _bay('1d').get_size() == skin
    assert _bay('1d_base').get_size() == skin
    assert _bay('2d').get_size() == skin + 3*MF*NF
    assert _bay('2d_base').get_size() == skin + 3*MF*NF


def test_add_stiffener_kwargs_are_set():
    bay = _bay()
    s1 = bay.add_bladestiff1d(ys=b/2, bf=9*tskin, fstack=stack, fplyt=plyt,
                              flaminaprop=laminaprop, Fx=10.)
    s2 = bay.add_bladestiff2d(ys=b/2, bf=9*tskin, fstack=stack, fplyt=plyt,
                              flaminaprop=laminaprop, mf=MF, nf=NF,
                              name='blade')
    assert isinstance(s1, BladeStiff1D) and s1.Fx == 10.
    assert isinstance(s2, BladeStiff2D) and s2.name == 'blade'
    assert bay.stiffeners == [s1, s2]
    assert s1.rho == s2.rho == bay.rho


@pytest.mark.parametrize('ys', [0., b])
def test_stiffener_at_bay_edge(ys):
    bay = _bay()
    s1 = bay.add_bladestiff1d(ys=ys, bf=9*tskin, fstack=stack, fplyt=plyt,
                              flaminaprop=laminaprop)
    s2 = bay.add_bladestiff2d(ys=ys, bf=9*tskin, fstack=stack, fplyt=plyt,
                              flaminaprop=laminaprop, mf=MF, nf=NF)
    for s in (s1, s2):
        assert s.panel1 is s.panel2


@pytest.mark.parametrize('method', ['add_bladestiff1d', 'add_bladestiff2d'])
@pytest.mark.parametrize('kwargs, error, match', [
    (dict(), ValueError, 'bstack or fstack'),
    (dict(bstack=stack, blaminaprop=laminaprop), ValueError, 'bplyt'),
    (dict(bstack=stack, bplyt=plyt), ValueError, 'blaminaprop'),
    (dict(fstack=stack, flaminaprop=laminaprop), ValueError, 'fplyt'),
    (dict(fstack=stack, fplyt=plyt), ValueError, 'flaminaprop'),
    (dict(ys=b/3, fstack=stack, fplyt=plyt, flaminaprop=laminaprop),
        RuntimeError, 'panel1 and panel2'),
])
def test_add_stiffener_invalid_input(method, kwargs, error, match):
    bay = _bay()
    kwargs = dict(kwargs)
    kwargs.setdefault('ys', b/2)
    with pytest.raises(error, match=match):
        getattr(bay, method)(bb=10*tskin, bf=9*tskin, **kwargs)


@pytest.mark.parametrize('method', ['add_bladestiff1d', 'add_bladestiff2d'])
def test_panels_must_come_before_stiffeners(method):
    bay = StiffPanelBay()
    bay.a = 2*b
    bay.b = b
    with pytest.raises(RuntimeError, match='panels must be added'):
        getattr(bay, method)(ys=b/2, bf=9*tskin, fstack=stack, fplyt=plyt,
                             flaminaprop=laminaprop)


def test_rebuild_requires_dimensions():
    bay = StiffPanelBay()
    with pytest.raises(ValueError, match='length a'):
        bay.calc_kC(silent=True)
    bay.a = 1.
    with pytest.raises(ValueError, match='width b'):
        bay.calc_kC(silent=True)


def test_bladestiff2d_increases_buckling_load():
    unstiffened, _ = _lb(_bay())
    flange, _ = _lb(_bay('2d'))
    flange_base, _ = _lb(_bay('2d_base'))
    assert flange[0] > 3*unstiffened[0]
    # the base adds stiffness to the skin
    assert flange_base[0] > flange[0]


def test_bladestiff1d_base_increases_buckling_load():
    flange, _ = _lb(_bay('1d'))
    flange_base, _ = _lb(_bay('1d_base'))
    assert flange_base[0] > flange[0]


def test_bladestiff2d_close_to_bladestiff1d():
    """Both blade formulations model the same stiffener

    The 1D flange is a beam following the skin, whereas the 2D flange is a
    plate connected to the skin by penalties, so the two agree only roughly.

    """
    ev1d, _ = _lb(_bay('1d'))
    ev2d, _ = _lb(_bay('2d'))
    assert np.isclose(ev1d[0], ev2d[0], rtol=0.2)


def _padup_bay():
    bay = _bay()
    bay.add_bladestiff2d(ys=b/2, bb=10*tskin, bstack=stack, bplyt=plyt,
                         blaminaprop=laminaprop)
    return bay


def test_bladestiff2d_without_flange():
    """A base-only BladeStiff2D is a pad-up on the skin"""
    bay = _padup_bay()
    assert bay.bladestiff2ds[0].flange is None
    assert bay.get_size() == 3*8*8
    unstiffened, _ = _lb(_bay())
    padup, _ = _lb(bay)
    assert padup[0] > unstiffened[0]
    c = np.zeros(bay.get_size())
    assert np.all(bay.calc_fext(silent=True) == 0)
    assert bay.calc_kM(silent=True).shape == (bay.get_size(), bay.get_size())
    with pytest.raises(RuntimeError, match='no flange'):
        bay.uvw_stiffener(c, 0)


@pytest.mark.parametrize('stiffener', [None, '1d', '1d_base', '2d', '2d_base'])
def test_calc_kM(stiffener):
    """The mass matrix is symmetric and usable in a frequency analysis"""
    bay = _bay(stiffener)
    kM = bay.calc_kM(silent=True)
    size = bay.get_size()
    assert kM.shape == (size, size)
    assert abs(kM - kM.T).max() <= 1e-12*abs(kM).max()
    eigvals, _ = freq(bay.calc_kC(silent=True), kM, silent=True,
                      num_eigvalues=3)
    assert np.all(np.isfinite(eigvals))


def _skin_kinetic_energy(bay):
    """Kinetic energy measure for a Ritz vector acting on the skin only"""
    kM = bay.calc_kM(silent=True)
    c = np.zeros(bay.get_size())
    c[:3*8*8] = 1.
    return c @ (kM @ c)


def test_calc_kM_bladestiff1d_adds_mass():
    assert _skin_kinetic_energy(_bay('1d')) > _skin_kinetic_energy(_bay())


def test_calc_kM_bladestiff2d_flange_has_mass():
    bay = _bay('2d')
    kM = bay.calc_kM(silent=True)
    assert abs(kM[3*8*8:, 3*8*8:]).max() > 0


def test_calc_kM_bladestiff2d_base_adds_mass():
    """The base of a BladeStiff2D adds the mass of a strip of skin

    The base has the lamination and density of the skin, over a width
    ``bb``, so it must add exactly the mass of that strip of skin.

    """
    unstiffened = _skin_kinetic_energy(_bay())
    padup = _skin_kinetic_energy(_padup_bay())
    assert padup > unstiffened

    strip = _bay()
    strip.add_panel(b/2 - 5*tskin, b/2 + 5*tskin)
    assert np.isclose(padup, _skin_kinetic_energy(strip), rtol=1e-3)


def test_uvw_skin(tmp_path):
    bay = _bay('2d')
    _, eigvecs = _lb(bay)
    c = eigvecs[:, 0]
    u, v, w, phix, phiy = bay.uvw_skin(c, gridx=7, gridy=5)
    assert w.shape == (5, 7)
    assert np.abs(w).max() > 0
    # the skin field only depends on the skin Ritz constants
    c2 = c.copy()
    c2[3*8*8:] = 0.
    assert np.allclose(bay.uvw_skin(c2, gridx=7, gridy=5)[2], w)
    # user-defined points
    xs = np.array([0.25, 0.5])*bay.a
    ys = np.array([0.25, 0.5])*bay.b
    assert bay.uvw_skin(c, xs=xs, ys=ys)[2].shape == (2,)
    with pytest.raises(ValueError, match='full vector'):
        bay.uvw_skin(c[:-1])


def test_uvw_stiffener():
    bay = _bay('2d')
    _, eigvecs = _lb(bay)
    c = eigvecs[:, 0]
    u, v, w, phix, phiy = bay.uvw_stiffener(c, 0, gridx=7, gridy=5)
    assert w.shape == (5, 7)
    assert bay.Ys.max() == pytest.approx(9*tskin)
    # the flange field only depends on the flange Ritz constants
    c2 = np.zeros_like(c)
    c2[3*8*8:] = c[3*8*8:]
    assert np.allclose(bay.uvw_stiffener(c2, 0, gridx=7, gridy=5)[2], w)
    with pytest.raises(ValueError, match='full vector'):
        bay.uvw_stiffener(c[:-1], 0, gridx=7, gridy=5)
    with pytest.raises(ValueError, match='Invalid region'):
        bay.uvw_stiffener(c, 0, region='web')
    with pytest.raises(RuntimeError, match='base of BladeStiff2D'):
        bay.uvw_stiffener(c, 0, region='base')


def test_uvw_stiffener_second_bladestiff2d():
    """The Ritz constants of each flange come after those of the previous"""
    bay = _bay('1d')
    bay.panels = []
    ys = [b/3, 2*b/3]
    for y1, y2 in zip([0.] + ys, ys + [b]):
        bay.add_panel(y1, y2, Nxx=-1.)
    bay.bladestiff1ds = []
    bay.stiffeners = []
    bay.add_bladestiff1d(ys=ys[0], bf=9*tskin, fstack=stack, fplyt=plyt,
                         flaminaprop=laminaprop)
    for y in ys:
        bay.add_bladestiff2d(ys=y, bf=9*tskin, fstack=stack, fplyt=plyt,
                             flaminaprop=laminaprop, mf=MF, nf=NF)
    skin = 3*8*8
    flange = 3*MF*NF
    assert bay.get_size() == skin + 2*flange
    rng = np.random.default_rng(0)
    c = rng.random(bay.get_size())
    w2 = bay.uvw_stiffener(c, 2, gridx=5, gridy=4)[2].copy()
    # only the constants of the second flange matter
    c2 = np.zeros_like(c)
    c2[skin + flange:] = c[skin + flange:]
    assert np.array_equal(bay.uvw_stiffener(c2, 2, gridx=5, gridy=4)[2], w2)
    c1 = c.copy()
    c1[skin + flange:] = 0.
    assert np.all(bay.uvw_stiffener(c1, 2, gridx=5, gridy=4)[2] == 0)


def test_uvw_stiffener_bladestiff1d():
    bay = _bay('1d')
    with pytest.raises(RuntimeError, match='BladeStiff1D'):
        bay.uvw_stiffener(np.zeros(bay.get_size()), 0)


def test_default_field_shape_mismatch():
    with pytest.raises(ValueError, match='same shape'):
        _bay()._default_field(np.zeros(2), 1., np.zeros(3), 1., 2, 2)


def test_plot_skin_and_stiffener(tmp_path):
    bay = _bay('2d')
    _, eigvecs = _lb(bay)
    c = eigvecs[:, 0]
    bay.uvw_skin(c, gridx=4, gridy=3)
    w_before = bay.w.copy()

    fig, ax = plt.subplots()
    out = bay.plot_skin(c, ax=ax, gridx=8, gridy=6, num_levels=5, silent=True)
    assert out is ax
    out = bay.plot_stiffener(c, 0, ax=ax, gridx=8, gridy=6, num_levels=5,
                             silent=True)
    assert out is ax
    plt.close(fig)
    # the fields computed before plotting are restored
    assert np.array_equal(bay.w, w_before)

    for method, args in (('plot_skin', ()), ('plot_stiffener', (0,))):
        filename = tmp_path / (method + '.png')
        ax = getattr(bay, method)(c, *args, vec='u', deform_u=True,
                colorbar=True, cbar_title='u', invert_y=True, title=method,
                clean=False, texts=[dict(x=0.1, y=0.1, s='text')],
                filename=str(filename), dpi=20, gridx=8, gridy=6,
                num_levels=5, silent=True)
        assert filename.exists()
        assert ax.get_title() == method
        with pytest.raises(ValueError, match='not a valid vec'):
            getattr(bay, method)(c, *args, vec='Nxx', silent=True)
        with pytest.raises(ValueError, match='Axes'):
            getattr(bay, method)(c, *args, ax='axes', gridx=8, gridy=6,
                                 silent=True)
    plt.close('all')


def test_save_and_load(tmp_path):
    bay = _bay('2d')
    eigvals, _ = _lb(bay)
    for p in bay.panels:
        p.calc_kC()
    bay.name = str(tmp_path / 'bay')
    bay.save()
    assert bay.kC is None
    assert all(v is None for p in bay.panels for v in p.matrices.values())
    for name in (bay.name, bay.name + '.StiffPanelBay'):
        bay2 = load(name)
        assert len(bay2.panels) == 2
        assert len(bay2.bladestiff2ds) == 1
        assert np.allclose(_lb(bay2)[0], eigvals)


def _shell(bay):
    """Single Shell over the whole bay, with the boundary conditions of the bay"""
    bcs = {e + d + r: getattr(bay, e + d + r) for e in ('x1', 'x2', 'y1', 'y2')
           for d in 'uvw' for r in ('', 'r')}
    return Shell(a=bay.a, b=bay.b, r=bay.r, stack=stack, plyt=plyt,
                 laminaprop=laminaprop, model=bay.model, m=bay.m, n=bay.n,
                 rho=bay.rho, **bcs)


@pytest.mark.parametrize('stiffener', [None, '1d', '2d', '2d_base'])
def test_calc_fext_without_forces(stiffener):
    bay = _bay(stiffener)
    fext = bay.calc_fext(silent=True)
    assert fext.shape == (bay.get_size(),)
    assert np.all(fext == 0)


def test_calc_fext_skin_point_load():
    bay = _bay('2d')
    load_ = (bay.a/3, bay.b/4, 1., 2., 3.)
    bay.forces_skin.append(load_)
    fext = bay.calc_fext(silent=True)
    s = _shell(bay)
    s.add_point_load(*load_)
    skin = 3*8*8
    assert np.allclose(fext[:skin], s.calc_fext())
    assert np.all(fext[skin:] == 0)


def test_calc_fext_flange_point_load():
    bay = _bay('2d')
    flange = bay.bladestiff2ds[0].flange
    load_ = (bay.a/3, flange.b/2, 0., 0., 1.)
    bay.bladestiff2ds[0].forces_flange.append(load_)
    fext = bay.calc_fext(silent=True)
    flange.add_point_load(*load_)
    skin = 3*8*8
    assert np.all(fext[:skin] == 0)
    assert np.allclose(fext[skin:], flange.calc_fext())
    assert np.abs(fext).max() > 0


@pytest.mark.parametrize('flow', ['x', 'y'])
def test_calc_kA_matches_single_shell(flow):
    """Splitting the skin into panels must not change kA"""
    bay = _bay()
    bay.flow = flow
    bay.beta = 1.3
    kA = bay.calc_kA(silent=True)
    s = _shell(bay)
    s.flow = flow
    s.beta = 1.3
    s.gamma = 0.
    kA_s = s.calc_kA(finalize=False)
    from structsolve.sparseutils import make_skew_symmetric
    kA_s = make_skew_symmetric(kA_s)
    assert kA.shape == (bay.get_size(), bay.get_size())
    assert abs(kA - kA_s).max() <= 1e-8*abs(kA_s).max()
    assert abs(kA + kA.T).max() <= 1e-12*abs(kA).max()


def test_calc_kA_from_Mach():
    bay = _bay()
    bay.rho_air = 1.2
    bay.V = 600.
    bay.speed_sound = 340.
    with pytest.raises(ValueError, match='NoneValue'):
        bay.calc_kA(silent=True)
    bay.Mach = 0.8
    with pytest.raises(ValueError, match='>= 1'):
        bay.calc_kA(silent=True)
    bay.Mach = 2.
    kA = bay.calc_kA(silent=True)
    beta = 1.2*600.**2/(2.**2 - 1)**0.5
    assert all(p.beta == beta for p in bay.panels)
    bay2 = _bay()
    bay2.beta = beta
    assert abs(kA - bay2.calc_kA(silent=True)).max() == 0


def test_calc_cA_matches_single_shell():
    bay = _bay('2d')
    bay.beta = 1.
    bay.aeromu = 2.5
    cA = bay.calc_cA(silent=True)
    assert cA.shape == (bay.get_size(), bay.get_size())
    cA_s = _shell(bay).calc_cA(2.5, size=bay.get_size())
    assert abs(cA - cA_s).max() == 0
    assert abs(cA).max() > 0
