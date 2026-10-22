r"""Mass matrices with the density moments `I_n` of the laminate

The kinetic energy `c^T [M] c` of a random Ritz vector is checked against an
independent integration of `\rho (u(z)^2 + v(z)^2 + w^2)` ply by ply, with
`z` from `-h/2 + d` to `h/2 + d`, the convention of the offset `d` of
:func:`composites.laminated_plate`, and the fields of :meth:`.Shell.uvw`.

"""
import sys
sys.path.append('../..')

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest

from panels.shell import Shell
from panels.stiffpanelbay import StiffPanelBay


LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
ALU = (70e9, 70e9, 0.33, 26.3e9, 26.3e9, 26.3e9)
FOAM = (0.1e9, 0.1e9, 0.3, 0.04e9, 0.04e9, 0.04e9)
MODELS = ['plate_clpt_donnell', 'plate_fsdt_donnell', 'plate_tsdt_donnell',
          'cylshell_clpt_donnell', 'cylshell_clpt_sanders',
          'cylshell_fsdt_donnell', 'cylshell_fsdt_sanders',
          'cylshell_tsdt_donnell', 'cylshell_tsdt_sanders']
PLYTS = [0.5e-3, 2e-3, 0.5e-3]
RHOS = [1600., 100., 2700.]


def _shell(model, offset, rhos=RHOS, m=6, n=5):
    s = Shell(model=model, a=0.5, b=0.3, r=(0.4 if 'cyl' in model else None),
              stack=[0, 0, 90], plyts=PLYTS, laminaprops=[LAMINAPROP, FOAM, ALU],
              rhos=rhos, offset=offset, m=m, n=n)
    return s


def _fields(s, c, xs, ys):
    s.uvw(c, xs=xs, ys=ys)
    out = [s.fields[k].ravel().copy() for k in ('u', 'v', 'w', 'phix', 'phiy')]
    dx, dy = 1e-6*s.a, 1e-6*s.b
    s.uvw(c, xs=xs + dx, ys=ys)
    wp = s.fields['w'].ravel().copy()
    s.uvw(c, xs=xs - dx, ys=ys)
    wx = (wp - s.fields['w'].ravel())/(2*dx)
    s.uvw(c, xs=xs, ys=ys + dy)
    wp = s.fields['w'].ravel().copy()
    s.uvw(c, xs=xs, ys=ys - dy)
    wy = (wp - s.fields['w'].ravel())/(2*dy)
    return out + [wx, wy]


def kinetic_energy_quadrature(s, c, plyts, rhos, ngp=20, nz=6):
    r"""`\int_A \int \rho (u(z)^2 + v(z)^2 + w^2) dz dA`, ply by ply"""
    xg, wg = leggauss(ngp)
    X, Y = np.meshgrid((xg + 1)/2*s.a, (xg + 1)/2*s.b, indexing='ij')
    W = np.outer(wg, wg).ravel()*s.a*s.b/4
    u, v, w, phix, phiy, wx, wy = _fields(s, c, X.ravel().copy(), Y.ravel().copy())
    h = sum(plyts)
    c1 = 4/(3*h**2) if '_tsdt_' in s.model else 0.
    #NOTE for the Sanders kinematics phiy of Shell.uvw() includes v/r, which
    #     the cubic term of the TSDT does not have
    phiy3 = phiy - v/s.r if ('sanders' in s.model and c1) else phiy
    zg, wzg = leggauss(nz)
    z0 = -h/2 + s.offset
    total = 0.
    for t, rho in zip(plyts, rhos):
        for zeta, wz in zip(zg, wzg):
            z = z0 + (zeta + 1)/2*t
            uz = u + z*phix - c1*z**3*(phix + wx)
            vz = v + z*phiy - c1*z**3*(phiy3 + wy)
            total += rho*wz*t/2*np.sum(W*(uz**2 + vz**2 + w**2))
        z0 += t
    return total


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('offset', [0., 1.8e-3])
def test_kinetic_energy_ply_by_ply(model, offset):
    s = _shell(model, offset)
    rng = np.random.default_rng(7)
    c = rng.standard_normal(s.get_size())
    kM = s.calc_kM()
    I_nxny = _lam_moments(s)
    kM_num = s.calc_kM(I_nxny=I_nxny)
    ref = kinetic_energy_quadrature(s, c, PLYTS, RHOS)
    assert np.isclose(c @ kM @ c, ref, rtol=1e-6)
    assert np.isclose(c @ kM_num @ c, ref, rtol=1e-6)


def _lam_moments(s):
    lam = s.lam
    moments = [lam.intrho, lam.intrhoz, lam.intrhoz2]
    if '_tsdt_' in s.model:
        moments += [lam.intrhoz3, lam.intrhoz4, lam.intrhoz6]
    out = np.zeros((s.nx, s.ny, len(moments)))
    out[:, :] = moments
    return out


@pytest.mark.parametrize('model', MODELS)
def test_closed_form_equals_numerical(model):
    s = _shell(model, 1.8e-3)
    kM = s.calc_kM().toarray()
    kM_num = s.calc_kM(I_nxny=_lam_moments(s)).toarray()
    assert np.abs(kM).max() > 0
    assert np.abs(kM - kM_num).max() <= 1e-12*np.abs(kM).max()


@pytest.mark.parametrize('model', MODELS)
def test_uniform_density_h_rho_nxny(model):
    r"""A uniform density through ``h_nxny`` and ``rho_nxny`` gives the
    moments of the laminate, at any offset"""
    s = _shell(model, -2.2e-3, rhos=[1600.]*3)
    kM = s.calc_kM().toarray()
    kM_num = s.calc_kM(h_nxny=np.full((s.nx, s.ny), s.lam.h),
                       rho_nxny=np.full((s.nx, s.ny), 1600.)).toarray()
    assert np.abs(kM - kM_num).max() <= 1e-12*np.abs(kM).max()


def test_invalid_moments():
    s = _shell('plate_fsdt_donnell', 0.)
    I_nxny = _lam_moments(s)
    with pytest.raises(ValueError, match='rho_nxny'):
        s.calc_kM(I_nxny=I_nxny, rho_nxny=np.ones((s.nx, s.ny)))
    with pytest.raises(ValueError, match='TSDT'):
        s.calc_kM(I_nxny=I_nxny, h_nxny=np.ones((s.nx, s.ny)))
    with pytest.raises(ValueError, match='shape'):
        s.calc_kM(I_nxny=I_nxny[..., :2])


def _bay():
    bay = StiffPanelBay()
    bay.a = 0.4
    bay.b = 0.3
    bay.r = None
    bay.model = 'plate_clpt_donnell'
    bay.stack = [0, 90, 90, 0]
    bay.plyt = 0.125e-3
    bay.laminaprop = LAMINAPROP
    bay.rho = 1600.
    bay.m = 6
    bay.n = 5
    bay.add_panel(0, 0.15)
    bay.add_panel(0.15, 0.3)
    return bay


def test_stiffener_base_below_the_skin():
    r"""The base of a blade stiffener is below the skin, as the flange: the
    coupling of `u` and `\phi_x = -w_{,x}` of both has the same sign"""
    plain = _bay()
    kM0 = plain.calc_kM(silent=True).toarray()
    base = _bay()
    base.add_bladestiff1d(ys=0.15, bb=0.02, bstack=[0, 90]*4, bplyt=0.125e-3,
                          blaminaprop=LAMINAPROP)
    flange = _bay()
    flange.add_bladestiff1d(ys=0.15, bf=0.02, fstack=[0, 90]*4,
                            fplyt=0.125e-3, flaminaprop=LAMINAPROP)
    dKb = base.calc_kM(silent=True).toarray() - kM0
    dKf = flange.calc_kM(silent=True).toarray() - kM0
    u = np.arange(0, kM0.shape[0], 3)
    w = u + 2
    Kb, Kf = dKb[np.ix_(u, w)], dKf[np.ix_(u, w)]
    assert np.abs(Kb).max() > 0 and np.abs(Kf).max() > 0
    assert np.sum(Kb*Kf) > 0
