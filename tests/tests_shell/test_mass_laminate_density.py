r"""Density of the closed-form mass matrix and the per-ply lists of Shell

The closed-form mass matrix ``fkM`` of every model uses the mean density of
the laminate, `\int \rho \, dz / h` = ``lam.intrho/lam.h``, as the numerical
``fkM_num`` of :meth:`.Shell.calc_kM`. The per-ply lists ``plyts``,
``laminaprops`` and ``rhos`` not given by the user are derived again from
``plyt``, ``laminaprop`` and ``rho`` at every rebuild.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest

from panels.shell import Shell, load


LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
STACK = [0, 45, 90, -45, 0]
RHOS = [1600., 1600., 2700., 1600., 1600.]
MODELS = ['plate_clpt_donnell', 'plate_fsdt_donnell', 'plate_tsdt_donnell',
          'cylshell_clpt_donnell', 'cylshell_clpt_sanders',
          'cylshell_fsdt_donnell', 'cylshell_fsdt_sanders',
          'cylshell_tsdt_donnell', 'cylshell_tsdt_sanders']


def _shell(model, **kwargs):
    kw = dict(a=0.5, b=0.3, stack=STACK, plyt=0.2e-3, laminaprop=LAMINAPROP,
              m=6, n=7, model=model)
    if model.startswith('cylshell'):
        kw['r'] = 0.4
    kw.update(kwargs)
    return Shell(**kw)


@pytest.mark.parametrize('model', MODELS)
def test_closed_form_equals_numerical_with_ply_densities(model):
    """With per-ply densities the closed-form kM is the numerical one"""
    s = _shell(model, rhos=RHOS, offset=0.1e-3)
    assert s.rho == 0
    kM = s.calc_kM().toarray()
    #NOTE giving the moments of the laminate at each point forces the
    #     numerical integration
    #NOTE the moments about the mid-surface for the TSDT
    lam = s._get_lam_mid() if '_tsdt_' in model else s.lam
    moments = [lam.intrho, lam.intrhoz, lam.intrhoz2]
    if '_tsdt_' in model:
        moments += [lam.intrhoz3, lam.intrhoz4, lam.intrhoz6]
    I_nxny = np.zeros((s.nx, s.ny, len(moments)))
    I_nxny[:, :] = moments
    kM_num = s.calc_kM(I_nxny=I_nxny).toarray()
    assert np.abs(kM).max() > 0
    assert np.abs(kM - kM_num).max() <= 1e-12*np.abs(kM).max()
    rho_mean = np.dot(RHOS, s.plyts)/sum(s.plyts)
    assert np.isclose(s.lam.intrho/s.lam.h, rho_mean, rtol=1e-14)


@pytest.mark.parametrize('model', MODELS)
def test_rhos_only_equals_uniform_rho(model):
    """A Shell given only ``rhos`` has the mass of the equivalent ``rho``"""
    kM_rhos = _shell(model, rhos=[1600.]*len(STACK)).calc_kM().toarray()
    kM_rho = _shell(model, rho=1600.).calc_kM().toarray()
    assert np.abs(kM_rhos).max() > 0
    assert np.abs(kM_rhos - kM_rho).max() <= 1e-14*np.abs(kM_rho).max()


def test_rho_set_after_construction():
    """``rhos`` derived from ``rho`` follows a later change of ``rho``"""
    s = _shell('plate_clpt_donnell')
    assert s.rhos == [0]*len(STACK)
    assert np.abs(s.calc_kM().toarray()).max() == 0
    s.rho = 1600.
    s.calc_kC()
    assert s.rhos == [1600.]*len(STACK)
    kM_ref = _shell('plate_clpt_donnell', rho=1600.).calc_kM().toarray()
    assert np.abs(s.calc_kM().toarray() - kM_ref).max() <= 1e-14*np.abs(kM_ref).max()


def test_plyt_and_laminaprop_set_after_construction():
    s = _shell('plate_clpt_donnell', rho=1600.)
    s.plyt = 0.3e-3
    s.laminaprop = (70e9, 0.3)
    s.stack = [0, 90]
    s._rebuild()
    assert s.plyts == [0.3e-3]*2
    assert s.laminaprops == [(70e9, 0.3)]*2
    assert s.rhos == [1600.]*2
    assert np.isclose(s.lam.h, 0.6e-3)


def test_user_lists_are_kept():
    """Lists given by the user, also changed in place, are never replaced"""
    s = _shell('plate_clpt_donnell', rhos=list(RHOS))
    kM = s.calc_kM().toarray()
    s.rho = 5000.
    s._rebuild()
    assert s.rhos == RHOS
    assert np.abs(s.calc_kM().toarray() - kM).max() == 0

    s = _shell('plate_clpt_donnell', rho=1600.)
    s.rhos = list(RHOS)
    s.rho = 5000.
    s._rebuild()
    assert s.rhos == RHOS

    s = _shell('plate_clpt_donnell', rho=1600.)
    s.rhos[2] = 2700.
    s.rho = 5000.
    s._rebuild()
    assert s.rhos == RHOS

    s = _shell('plate_clpt_donnell', plyts=[0.1e-3]*len(STACK))
    s.plyt = 0.3e-3
    s._rebuild()
    assert s.plyts == [0.1e-3]*len(STACK)


def test_save_and_load_keep_the_derived_lists(tmp_path):
    s = _shell('plate_clpt_donnell', rho=1600., rhos=None)
    u = _shell('plate_clpt_donnell', rhos=list(RHOS))
    for shell, name in ((s, 'derived'), (u, 'user')):
        shell.save(str(tmp_path / name))
    s2 = load(str(tmp_path / 'derived'))
    u2 = load(str(tmp_path / 'user'))
    assert s2.rhos == [1600.]*len(STACK)
    assert u2.rhos == RHOS
    s2.rho = u2.rho = 2000.
    s2._rebuild()
    u2._rebuild()
    assert s2.rhos == [2000.]*len(STACK)
    assert u2.rhos == RHOS
