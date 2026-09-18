r"""Limits of the integration domain, ``Shell.x1, x2, y1, y2``

Up to panels 0.6.9 the full domain was given by the sentinels ``x1 = y1 =
-1`` and ``x2 = y2 = +1``, and a partial domain was recognized only when both
limits of a pair departed from them. Two consequences are checked here not to
come back:

- ``x2 = 1.0`` is a coordinate like any other, it silently meant the full
  domain on a shell with ``a > 1``.
- One limit alone, e.g. ``x1 = 0.2``, integrates ``[0.2, a]``, it was silently
  ignored.

The full domain is now given by ``None``, and the matrices of the full domain
are the same as before, see ``test_full_domain_*``.

"""
import numpy as np
import pytest

from panels.shell import Shell

MODELS = [('plate_clpt_donnell', None), ('cylshell_clpt_donnell', 1.5)]


def make(model, r, a=2.0, b=1.3, **limits):
    s = Shell(a=a, b=b, r=r, stack=[0, 45, 90, -30, 0], plyt=0.4e-3,
              laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
              rho=1600., model=model, m=8, n=7)
    s.nx = 4*s.m
    s.ny = 4*s.n
    s.Nxx = -1.3
    s.Nyy = 0.4
    s.Nxy = 0.2
    for k, v in limits.items():
        setattr(s, k, v)
    return s


def matrices(s, c):
    """calc_kC (with NLgeom), calc_kG, calc_kM and calc_fint"""
    return dict(kC=s.calc_kC(c=c, NLgeom=True).toarray(),
                kG=s.calc_kG(c=c, NLgeom=True).toarray(),
                kM=s.calc_kM().toarray(),
                fint=s.calc_fint(c))


def random_c(s, seed=0):
    return np.random.default_rng(seed).standard_normal(s.get_size())*1e-3


@pytest.mark.parametrize('model, r', MODELS)
@pytest.mark.parametrize('edge', ['x', 'y'])
def test_limit_at_one_is_a_coordinate(model, r, edge):
    """``a = 2.0, x1 = 0.2, x2 = 1.0`` is a partial domain

    A perturbation of 1e-9 of the limit must change the matrices by about
    1e-9, and not by the 34 % that separated the full domain from the partial
    one when ``x2 = 1.0`` was read as the full-domain sentinel.

    """
    lim = dict(x1=0.2, x2=1.0) if edge == 'x' else dict(y1=0.2, y2=1.0)
    s = make(model, r, **lim)
    assert s.is_partial_domain()
    near = make(model, r, **{k: (v + 1e-9 if v == 1.0 else v)
                             for k, v in lim.items()})
    full = make(model, r)
    c = random_c(s)
    got, ref, whole = matrices(s, c), matrices(near, c), matrices(full, c)
    for k in got:
        scale = np.abs(ref[k]).max()
        assert np.abs(got[k] - ref[k]).max() < 1e-6*scale, k
        assert np.abs(got[k] - whole[k]).max() > 1e-2*scale, k


@pytest.mark.parametrize('model, r', MODELS)
@pytest.mark.parametrize('lim, explicit', [
    (dict(x1=0.2), dict(x1=0.2, x2=2.0)),
    (dict(x2=1.6), dict(x1=0., x2=1.6)),
    (dict(y1=0.3), dict(y1=0.3, y2=1.3)),
    (dict(y2=0.9), dict(y1=0., y2=0.9)),
    ])
def test_one_sided_limit(model, r, lim, explicit):
    """A single limit integrates up to the edge of the shell, and the two
    sides of the cut add up to the full domain"""
    one = make(model, r, **lim)
    assert one.is_partial_domain()
    both = make(model, r, **explicit)
    c = random_c(one)
    got, ref = matrices(one, c), matrices(both, c)
    for k in got:
        assert np.array_equal(got[k], ref[k]), k

    name, value = list(lim.items())[0]
    other = {'x1': 'x2', 'x2': 'x1', 'y1': 'y2', 'y2': 'y1'}[name]
    rest = make(model, r, **{other: value})
    full = make(model, r)
    total = {k: got[k] + v for k, v in matrices(rest, c).items()}
    whole = matrices(full, c)
    for k in whole:
        assert np.allclose(total[k], whole[k], rtol=1e-10,
                           atol=1e-10*np.abs(whole[k]).max()), k


@pytest.mark.parametrize('model, r', MODELS)
def test_full_domain_is_the_default(model, r):
    """``None`` and the explicit edges must give the very same matrices,
    through the numerical integration, and the analytical closed-form
    matrices must still be selected"""
    default = make(model, r)
    explicit = make(model, r, x1=0., x2=2.0, y1=0., y2=1.3)
    assert default.integration_limits() == (0., 2.0, 0., 1.3)
    assert not default.is_partial_domain()
    assert not explicit.is_partial_domain()
    c = random_c(default)
    got, ref = matrices(default, c), matrices(explicit, c)
    for k in got:
        assert np.array_equal(got[k], ref[k]), k


@pytest.mark.parametrize('model, r', MODELS)
def test_full_domain_numerical_matches_analytical(model, r):
    """With the default limits the numerical kernels integrate ``xi, eta``
    in ``[-1, +1]``, as the closed-form matrices do"""
    s = make(model, r)
    size = s.get_size()
    zero = np.zeros(size)
    kC_ana = s.calc_kC().toarray()
    kC_num = s.calc_kC(c=zero).toarray()
    assert np.allclose(kC_num, kC_ana, atol=1e-10*np.abs(kC_ana).max())
    kG_ana = s.calc_kG().toarray()
    kG_num = s.calc_kG(c=zero, NLgeom=True).toarray()
    assert np.allclose(kG_num, kG_ana, atol=1e-10*np.abs(kG_ana).max())
    kM_ana = s.calc_kM().toarray()
    kM_num = s.calc_kM(h_nxny=np.full((s.nx, s.ny), s.lam.h)).toarray()
    assert np.allclose(kM_num, kM_ana, atol=1e-10*np.abs(kM_ana).max())


@pytest.mark.parametrize('lim', [dict(x1=-1), dict(y1=-1),
                                 dict(x1=-1, x2=+1), dict(y1=-1, y2=+1)])
def test_old_sentinels_are_rejected(lim):
    """``-1`` was the full-domain sentinel and is now an invalid coordinate"""
    s = make('plate_clpt_donnell', None)
    for k, v in lim.items():
        setattr(s, k, v)
    with pytest.raises(ValueError, match='Use None'):
        s.calc_kC()
    # the constructor builds the shell, so it fails already
    with pytest.raises(ValueError, match='Use None'):
        Shell(a=2.0, b=1.3, stack=[0, 90], plyt=1e-3,
              laminaprop=(1e9, 1e9, 0.3, 1e8, 1e8, 1e8), **lim)


@pytest.mark.parametrize('lim', [dict(x2=2.5), dict(x1=1.0, x2=1.0),
                                 dict(x1=1.2, x2=0.4), dict(y2=1.4),
                                 dict(y1=-0.1)])
def test_invalid_limits_are_rejected(lim):
    s = make('plate_clpt_donnell', None, **lim)
    with pytest.raises(ValueError, match='must satisfy'):
        s.calc_kG()


def test_round_off_at_the_edges_is_absorbed():
    s = make('plate_clpt_donnell', None, x1=-1e-15, x2=2.0*(1 + 1e-14))
    assert s.integration_limits()[:2] == (0., 2.0)
    assert not s.is_partial_domain()


if __name__ == '__main__':
    pytest.main(['-v', __file__])
