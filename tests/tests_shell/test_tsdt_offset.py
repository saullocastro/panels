r"""TSDT with a laminate offset `d`

The cubic term of the TSDT is written about the mid-surface, `-c_1 [(z -
d)^3 + d^3] (\phi + w_{,x})`, such that the transverse shear strains vanish
at both faces `z = d \pm h/2`. The strain energy of :meth:`.Shell.calc_kC`
is checked against the integration ply by ply of the strains through the
thickness, and the natural frequencies of a free homogeneous plate must not
depend on the choice of the reference surface.

"""
import sys
sys.path.append('../..')

import numpy as np
from numpy.polynomial.legendre import leggauss
import pytest
from scipy.linalg import eigh

from panels.shell import Shell


LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.8e9)
FOAM = (0.1e9, 0.1e9, 0.3, 0.04e9, 0.04e9, 0.04e9)
MODELS = ['plate_tsdt_donnell', 'cylshell_tsdt_donnell', 'cylshell_tsdt_sanders']
PLYTS = [0.5e-3, 2e-3, 0.5e-3]


def _shell(model, offset, m=6, n=5):
    return Shell(model=model, a=0.4, b=0.3, r=(0.5 if 'cyl' in model else None),
                 stack=[0, 45, 90], plyts=PLYTS,
                 laminaprops=[LAMINAPROP, FOAM, LAMINAPROP],
                 rhos=[1600., 100., 1600.], offset=offset, m=m, n=n)


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('offset', [0., 1.2e-3, -2.5e-3])
def test_shear_vanishes_at_both_faces(model, offset):
    s = _shell(model, offset)
    c = np.random.default_rng(3).standard_normal(s.get_size())
    xs, ys = np.meshgrid(np.linspace(0.05, 0.35, 4), np.linspace(0.05, 0.25, 3))
    s.strain(c, xs=xs, ys=ys, NLgeom=False)
    f = s.fields
    h = sum(PLYTS)
    for face in (offset - h/2, offset + h/2):
        for g0, g2 in (('gyz', 'gyz2'), ('gxz', 'gxz2')):
            gam = f[g0] + (face - offset)**2*f[g2]
            assert np.abs(gam).max() <= 1e-12*np.abs(f[g0]).max()
    assert np.abs(f['gxz']).max() > 0


def _strain_energy_quadrature(s, c, ngp=16, nz=6):
    r"""`\frac{1}{2} \int_A \int (\varepsilon^\top \bar Q \varepsilon +
    \gamma^\top \bar Q_s \gamma) dz dA`, ply by ply, with the kinematics of the
    docstring of this module"""
    xg, wg = leggauss(ngp)
    X, Y = np.meshgrid((xg + 1)/2*s.a, (xg + 1)/2*s.b, indexing='ij')
    W = np.outer(wg, wg).ravel()*s.a*s.b/4
    s.strain(c, xs=X.ravel().copy(), ys=Y.ravel().copy(), NLgeom=False)
    f = {k: s.fields[k].ravel() for k in ('exx', 'eyy', 'gxy', 'kxx', 'kyy',
         'kxy', 'kxx3', 'kyy3', 'kxy3', 'gyz', 'gxz', 'gyz2', 'gxz2')}
    e0 = np.array([f['exx'], f['eyy'], f['gxy']])
    e1 = np.array([f['kxx'], f['kyy'], f['kxy']])
    e3 = np.array([f['kxx3'], f['kyy3'], f['kxy3']])
    g0 = np.array([f['gyz'], f['gxz']])
    g2 = np.array([f['gyz2'], f['gxz2']])
    d = s.offset
    h = sum(s.plyts)
    zg, wzg = leggauss(nz)
    z0 = d - h/2
    U = 0.
    for ply in s.lam.plies:
        Q = np.array([[ply.q11L, ply.q12L, ply.q16L],
                      [ply.q12L, ply.q22L, ply.q26L],
                      [ply.q16L, ply.q26L, ply.q66L]])
        Qs = np.array([[ply.q44L, ply.q45L], [ply.q45L, ply.q55L]])
        for zeta, wz in zip(zg, wzg):
            z = z0 + (zeta + 1)/2*ply.h
            eps = e0 + z*e1 + ((z - d)**3 + d**3)*e3
            gam = g0 + (z - d)**2*g2
            dens = np.einsum('ip,ij,jp->p', eps, Q, eps) + np.einsum('ip,ij,jp->p', gam, Qs, gam)
            U += 0.5*wz*ply.h/2*np.sum(W*dens)
        z0 += ply.h
    return U


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('offset', [0., 1.2e-3, -2.5e-3])
def test_strain_energy_ply_by_ply(model, offset):
    s = _shell(model, offset)
    c = np.random.default_rng(5).standard_normal(s.get_size())*1e-3
    kC = s.calc_kC()
    U = 0.5*c @ kC @ c
    assert np.isclose(U, _strain_energy_quadrature(s, c), rtol=1e-8)


def test_offset_zero_is_the_laminate():
    s = _shell('plate_tsdt_donnell', 0.)
    lam = s.lam
    ABD = s.ABD
    assert np.array_equal(ABD[0:3, 0:3], lam.A)
    assert np.array_equal(ABD[0:3, 3:6], lam.B)
    assert np.array_equal(ABD[0:3, 6:9], lam.E)
    assert np.array_equal(ABD[6:9, 6:9], lam.H)
    assert np.array_equal(ABD[11:13, 11:13], lam.Fts)
    assert s._get_lam_mid() is lam


def _free_plate_freqs(offset, k=6):
    E, nu, rho, h = 70e9, 0.3, 2700., 0.02
    s = Shell(model='plate_tsdt_donnell', a=0.6, b=0.4, stack=[0.], plyt=h,
              laminaprop=(E, nu), rho=rho, offset=offset, m=8, n=7)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for dof in ('u', 'v', 'w', 'phix', 'phiy'):
            setattr(s, edge + dof, 1.)
            setattr(s, edge + dof + 'r', 1.)
    s._rebuild()
    kC = s.calc_kC().toarray()
    kM = s.calc_kM().toarray()
    w2 = np.sort(eigh(kC, kM, eigvals_only=True))
    # six rigid-body modes
    assert np.abs(w2[:6]).max() <= 1e-6*w2[6]
    return np.sqrt(w2[6:6 + k])


def test_free_plate_frequencies_do_not_depend_on_the_reference_surface():
    r"""A homogeneous free plate is the same structure for any choice of the
    reference surface, whose Ritz spaces coincide for free edges"""
    f0 = _free_plate_freqs(0.)
    for offset in (0.01, -0.013, 0.025):
        assert np.allclose(_free_plate_freqs(offset), f0, rtol=1e-7)
