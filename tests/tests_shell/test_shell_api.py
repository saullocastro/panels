import sys
sys.path.append('../..')

import numpy as np
import pytest

from panels.shell import Shell, check_c, load


laminaprop = (71e9, 0.33)


def _plate(**kwargs):
    kw = dict(a=1., b=0.5, r=None, stack=[0], plyt=2e-3, laminaprop=laminaprop,
              model='plate_clpt_donnell', m=6, n=6)
    kw.update(kwargs)
    return Shell(**kw)


def test_check_c():
    check_c(np.zeros(4), 4)
    with pytest.raises(TypeError):
        check_c([0., 0., 0., 0.], 4)
    with pytest.raises(ValueError):
        check_c(np.zeros((2, 2)), 4)
    with pytest.raises(ValueError):
        check_c(np.zeros(3), 4)


@pytest.mark.parametrize('kwargs, match', [
    (dict(model='plate_unknown_theory'), 'valid models'),
    (dict(stack=[]), 'stack must be defined'),
    (dict(laminaprop=None), 'laminaprop must be defined'),
    (dict(plyt=None), 'plyt must be defined'),
    (dict(m=31), 'order 31'),
])
def test_invalid_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _plate(**kwargs)


def test_force_orthotropic_laminate():
    stack = [30, -30, 45]
    lp = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    aniso = _plate(stack=stack, laminaprop=lp)
    ortho = _plate(stack=stack, laminaprop=lp, force_orthotropic_laminate=True)
    idx = [(0, 2), (1, 2), (0, 5), (1, 5), (3, 5), (4, 5), (2, 3), (2, 4)]
    assert all(aniso.ABD[i, j] != 0 for i, j in idx)
    assert all(ortho.ABD[i, j] == 0 for i, j in idx)
    assert all(ortho.ABD[j, i] == 0 for i, j in idx)
    # terms that are not coupling terms are kept
    assert np.allclose(np.diag(ortho.ABD), np.diag(aniso.ABD))


def test_distr_load_fixed_y_matches_fixed_x():
    """A line load along x on a square plate is the transpose of one along y

    On a square plate with the same boundary conditions on all edges, the
    work of ``q`` along ``y = b/2`` against the transverse field must equal
    the work of ``q`` along ``x = a/2``, because the displacement field is
    symmetric about the diagonal for a symmetric laminate.

    """
    s1 = _plate(a=1., b=1.)
    s1.add_distr_load_fixed_y(0.5, funcz=lambda x: 1.)
    s2 = _plate(a=1., b=1.)
    s2.add_distr_load_fixed_x(0.5, funcz=lambda y: 1.)
    f1 = s1.calc_fext()
    f2 = s2.calc_fext()
    assert np.isclose(f1.sum(), f2.sum())
    assert np.abs(f1).max() > 0


def test_distr_load_fixed_y_inc_is_scaled():
    s = _plate()
    s.add_distr_load_fixed_y(0.25, funcx=lambda x: 1., cte=False)
    f1 = s.calc_fext(inc=1.)
    f05 = s.calc_fext(inc=0.5)
    assert np.allclose(f05, 0.5*f1)
    assert np.abs(f1).max() > 0


def test_distr_load_fixed_y_requires_a_function():
    with pytest.raises(ValueError):
        _plate().add_distr_load_fixed_y(0.25)


def test_clear_loads():
    s = _plate()
    s.add_point_load(0.5, 0.25, 0, 0, 1.)
    s.add_point_load(0.5, 0.25, 0, 0, 1., cte=False)
    s.add_distr_load_fixed_x(0.5, funcz=lambda y: 1.)
    s.add_distr_load_fixed_y(0.25, funcz=lambda x: 1., cte=False)
    assert np.abs(s.calc_fext()).max() > 0
    s.clear_loads()
    assert np.all(s.calc_fext() == 0)


def test_clear_disps():
    s = _plate()
    s.add_point_pd(0.5, 0.25, 0, 0, 0, 0, 1e6, 1e-3)
    s.add_distr_pd_fixed_x(0.5, kw=1e6, funcw=lambda y: 1e-3)
    s.clear_disps()
    assert s.point_pds == [] and s.point_pds_inc == []
    assert s.distr_pds == [] and s.distr_pds_inc == []


def test_stiffness_point_constraint():
    """The penalty matrix is the outer product of the shape functions"""
    s = _plate()
    size = s.get_size()
    kPC = s.calc_stiffness_point_constraint(0.3, 0.2, u=False, v=False, w=True,
                                            kuvw=1.)
    assert kPC.shape == (size, size)
    # rank one and symmetric positive semi-definite
    kPC = np.asarray(kPC)
    assert np.allclose(kPC, kPC.T)
    assert np.linalg.matrix_rank(kPC) == 1
    assert np.all(np.linalg.eigvalsh(kPC) > -1e-12)

    kall = np.asarray(s.calc_stiffness_point_constraint(0.3, 0.2, phix=True,
                                                        phiy=True))
    assert np.linalg.matrix_rank(kall) == 5


def test_stiffness_point_constraint_fixes_the_point():
    """A w penalty at the point where the load is applied cancels w there"""
    from structsolve import solve
    s = _plate(a=1., b=1.)
    s.add_point_load(0.5, 0.5, 0, 0, 1.)
    kC = s.calc_kC()
    x, y = 0.5, 0.5
    c = solve(kC + s.calc_stiffness_point_constraint(x, y, u=False, v=False,
                                                     kuvw=1e12),
              s.calc_fext(), silent=True)
    s.uvw(c, xs=np.array([x]), ys=np.array([y]))
    w_constrained = s.fields['w'][0]
    c = solve(kC, s.calc_fext(), silent=True)
    s.uvw(c, xs=np.array([x]), ys=np.array([y]))
    w_free = s.fields['w'][0]
    assert abs(w_constrained) < 1e-3*abs(w_free)


@pytest.mark.parametrize('flow', ['x', 'y'])
def test_calc_kA_flow(flow):
    s = _plate()
    s.flow = flow
    s.beta = 1.
    kA = s.calc_kA()
    assert kA.shape == (s.get_size(), s.get_size())
    assert abs(kA).max() > 0


def test_calc_kA_invalid_flow():
    s = _plate()
    s.flow = 'z'
    s.beta = 1.
    with pytest.raises(ValueError, match='Invalid flow'):
        s.calc_kA()


def test_calc_kA_from_Mach():
    s = _plate()
    s.rho_air = 1.2
    s.air_speed = 600.
    with pytest.raises(ValueError, match='NoneValue'):
        s.calc_kA()
    s.Mach = 0.8
    with pytest.raises(ValueError, match='>= 1'):
        s.calc_kA()
    s.Mach = 1.
    s.calc_kA()
    assert s.Mach == 1.0001
    assert np.isclose(s.beta, 1.2*600.**2/(1.0001**2 - 1)**0.5)


def test_calc_cA():
    s = _plate()
    size = s.get_size()
    cA = s.calc_cA(aeromu=1.)
    assert cA.shape == (size, size)
    assert s.matrices['cA'] is cA
    # purely imaginary, symmetric and linear in aeromu
    assert np.all(cA.real.toarray() == 0)
    assert abs(cA - cA.T).max() == 0
    assert abs(cA.imag).max() > 0
    assert abs(s.calc_cA(aeromu=2.) - 2*cA).max() <= 1e-12*abs(cA).max()


@pytest.mark.parametrize('method', ['uvw', 'strain', 'stress'])
def test_fields_of_plate_with_unset_radius(method):
    """An unset radius of a plate is the same as a zero radius"""
    c = np.random.default_rng(0).random(_plate().get_size())
    _, unset = getattr(_plate(r=None), method)(c, gridx=4, gridy=3)
    _, zero = getattr(_plate(r=0.), method)(c, gridx=4, gridy=3)
    assert unset.keys() == zero.keys()
    for k in unset:
        assert np.array_equal(unset[k], zero[k])


def test_save_and_load(tmp_path):
    s = _plate()
    s.calc_kC()
    s.name = str(tmp_path / 'plate')
    s.save()
    for name in (s.name, s.name + '.Shell'):
        s2 = load(name)
        assert s2.a == s.a and s2.b == s.b
        assert s2.stack == s.stack
        assert np.allclose(s2.ABD, s.ABD)
