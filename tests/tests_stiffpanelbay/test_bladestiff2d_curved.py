r"""Tests of :class:`.BladeStiff2D` on a curved bay

The selection of the stiffener model and of the model of its base from the
skin, and the connection of the Sanders kinematics,
``bladestiff2d_clt_sanders``, under a rigid rotation of the cylinder.

"""
import numpy as np
import pytest
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.bardell import calc_vec_f
from panels.stiffpanelbay import StiffPanelBay
from panels.stiffener.models import (bladestiff2d_clt_donnell,
                                     bladestiff2d_clt_sanders)


def _bay(r, model, m=6, n=6):
    bay = StiffPanelBay()
    bay.a, bay.b, bay.r = 0.4, 0.2, r
    bay.model = model
    bay.stack = [0, 90, 90, 0]
    bay.plyt = 0.125e-3
    bay.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    bay.rho = 1600.
    bay.m, bay.n = m, n
    bay.add_panel(0, 0.1)
    bay.add_panel(0.1, 0.2)
    return bay


def _stiffener(bay, model=None):
    return bay.add_bladestiff2d(ys=0.1, bb=0.02, bstack=[0, 90, 90, 0],
            bplyt=0.125e-3, blaminaprop=bay.laminaprop, bf=0.02,
            fstack=[0, 90, 90, 0], fplyt=0.125e-3,
            flaminaprop=bay.laminaprop, mf=4, nf=4, model=model)


@pytest.mark.parametrize('r, skin, model, expected, base', [
    (None, 'plate_clpt_donnell', None, 'bladestiff2d_clt_donnell',
     'plate_clpt_donnell'),
    (0.5, 'cylshell_clpt_sanders', None, 'bladestiff2d_clt_sanders',
     'cylshell_clpt_sanders'),
    (0.5, None, None, 'bladestiff2d_clt_sanders', 'cylshell_clpt_sanders'),
    (0.5, 'cylshell_clpt_gl', None, 'bladestiff2d_clt_sanders',
     'cylshell_clpt_sanders'),
    (0.5, 'cylshell_clpt_donnell', None, 'bladestiff2d_clt_donnell',
     'cylshell_clpt_donnell'),
    (0.5, 'cylshell_clpt_sanders', 'bladestiff2d_clt_donnell',
     'bladestiff2d_clt_donnell', 'cylshell_clpt_donnell'),
    ])
def test_model_selection(r, skin, model, expected, base):
    """The stiffener model and the model of the base follow the skin

    A curved skin with the Sanders kinematics (the default with a radius,
    or ``cylshell_clpt_gl``) gets ``bladestiff2d_clt_sanders`` and a base
    ``cylshell_clpt_sanders``; a flat skin, a skin
    ``cylshell_clpt_donnell`` or an explicit ``model`` get
    ``bladestiff2d_clt_donnell`` and a Donnell base. The stiffness matrix
    of the bay is assembled.

    """
    bay = _bay(r, skin)
    s = _stiffener(bay, model)
    assert s.model == expected
    assert s.base.model == base
    kC = bay.calc_kC(silent=True)
    assert kC.shape == (bay.get_size(), bay.get_size())
    assert abs(kC - kC.T).max() <= 1e-12*abs(kC).max()


def test_model_errors():
    """An unknown model, and the Sanders model on a flat bay, raise"""
    with pytest.raises(ValueError, match='requires a curved bay'):
        _stiffener(_bay(None, 'plate_clpt_donnell'),
                   'bladestiff2d_clt_sanders')
    with pytest.raises(ValueError, match='models of BladeStiff2D'):
        _stiffener(_bay(0.5, 'cylshell_clpt_sanders'),
                   'bladestiff1d_clt_donnell')


def test_rigid_rotation_of_the_connection():
    r"""A rigid rotation of the cylinder about its axis does not strain the
    connection of ``bladestiff2d_clt_sanders``

    With all the edges free, the Ritz vector of the rigid rotation `\alpha`
    is `v = r \alpha` on the skin and `w_f = (r - y_f) \alpha` on the
    flange, which points inwards. The penalty energy of the kernels of
    ``bladestiff2d_clt_sanders`` vanishes; that of
    ``bladestiff2d_clt_donnell`` is `k_r \alpha^2 a/2`, from the rotation
    `w_{,y} = 0` of the skin against `w_{f,y} = -\alpha` of the flange.

    Notes
    -----
    The rotation of the normal of Sanders' kinematics is `-\phi_y = w_{,y}
    - v/r` (Sanders 1963), which is `-\alpha`, as `w_{f,y}`.

    References
    ----------
    Sanders, J. L., Nonlinear theories for thin shells, Quarterly of
    Applied Mathematics, 21(1), 21-36, 1963.
    https://doi.org/10.1090/qam/147023

    """
    m = n = 6
    m1, n1 = 6, 5
    a, b, r, bf = 0.6, 0.13, 0.25, 0.007
    kt, kr = 1.e9, 1.e3
    ys = 0.37*b
    free = (1., 1., 1., 1.)
    bc = free*6
    xs = np.linspace(-1, 1, 25)

    def fit(nterms, values):
        F = np.array([calc_vec_f(x, *free)[:nterms] for x in xs])
        c = np.linalg.lstsq(F, values, rcond=None)[0]
        assert np.allclose(F @ c, values, rtol=0, atol=1e-12*abs(values).max())
        return c

    one, one1 = fit(m, np.ones_like(xs)), fit(m1, np.ones_like(xs))
    lin = fit(n1, r - (xs + 1)*bf/2)
    size = 3*m*n + 3*m1*n1
    col0 = 3*m*n
    alpha = 1.e-3
    c = np.zeros(size)
    for j in range(n):
        for i in range(m):
            c[3*(j*m + i) + 1] = r*alpha*one[i]*one[j]
    for l1 in range(n1):
        for k1 in range(m1):
            c[col0 + 3*(l1*m1 + k1) + 2] = alpha*one1[k1]*lin[l1]
    energy = {}
    for name, mod, geom in (('sanders', bladestiff2d_clt_sanders, (a, b, r)),
                            ('donnell', bladestiff2d_clt_donnell, (a, b))):
        K = (mod.fkCss(kt, kr, ys, *geom, m, n, *bc, size, 0, 0)
             + mod.fkCsf(kt, kr, ys, *geom, bf, m, n, m1, n1, *bc, *bc, size,
                         0, col0)
             + mod.fkCff(kt, kr, a, bf, m1, n1, *bc, size, col0, col0))
        K = finalize_symmetric_matrix(K)
        energy[name] = 0.5*c @ (K @ c)
    expected = kr*alpha**2*a/2
    assert np.isclose(energy['donnell'], expected, rtol=1e-8)
    assert abs(energy['sanders']) <= 1e-8*expected

