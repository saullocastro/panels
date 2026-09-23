r"""Curvature term of the piston theory for cylindrical shells

For a flow along `x` the aerodynamic load on the shell along `w` is `q =
\beta w_{,x} + \gamma w`, see :meth:`.Shell.calc_kA`. The term `\gamma w`,
with `\gamma = \beta/(2 r \sqrt{M^2 - 1})`, is Krumhaar's correction for the
external flow over a cylinder with `w` positive outwards: an outward
displacement lowers the pressure, pulling the shell outwards.

Reference: Krumhaar, H., "The accuracy of linear piston theory when applied
to cylindrical shells", AIAA J., 1(6), 1448-1449, 1963.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.special import hankel2

from structsolve.sparseutils import finalize_symmetric_matrix

from panels import modelDB
from panels.shell import Shell

CYLINDERS = ['cylshell_clpt_donnell', 'cylshell_clpt_sanders',
             'cylshell_fsdt_donnell', 'cylshell_fsdt_sanders',
             'cylshell_tsdt_donnell', 'cylshell_tsdt_sanders']
PLATES = ['plate_clpt_donnell', 'plate_fsdt_donnell', 'plate_tsdt_donnell']


def make(model):
    s = Shell(model=model, a=0.8, b=0.5, stack=[0, 45, 90], plyt=0.3e-3,
              laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9),
              rho=1600., m=7, n=6)
    if model in CYLINDERS:
        s.r = 0.6
    s.rho_air = 0.5
    s.air_speed = 600.
    s.Mach = 2.
    return s


def test_krumhaar_correction():
    r"""Exact linear potential flow over a cylinder of radius `R` with the
    radial displacement `W e^{i k x}`, positive outwards

    The disturbance potential `\Phi(r) e^{i k x}`, with `\Phi(r) \propto
    H_0^{(2)}(\beta_M k r)` and `\beta_M = \sqrt{M^2 - 1}`, gives the
    pressure `p = -\rho U^2 k^2 W H_0^{(2)}/(\beta_M k H_1^{(2)})` at `r =
    R`, which for short wavelengths becomes `p \approx (\rho U^2/\beta_M)
    (w_{,x} - w/(2 \beta_M R))`, i.e. `-q = \beta w_{,x} - \gamma w` for
    the flow along `x`
    """
    rho_air, U, Mach, R = 1.2, 500., 2.5, 0.8
    betaM = np.sqrt(Mach**2 - 1)
    beta = rho_air*U**2/betaM
    gamma = beta/(2*R*betaM)
    W = 1.
    errors = []
    for k in (100., 200., 400.):
        z = betaM*k*R
        p = -rho_air*U**2*k**2*W*hankel2(0, z)/(betaM*k*hankel2(1, z))
        piston = beta*(1j*k*W) - gamma*W
        errors.append(abs(p - piston)/abs(beta*k*W))
    # the error of the corrected piston theory is of second order
    assert errors[0] < 2e-5
    assert np.allclose(np.array(errors[:-1])/np.array(errors[1:]), 4.,
                       rtol=0.02)


@pytest.mark.parametrize('model', CYLINDERS)
def test_curvature_term(model):
    r"""`[K_A](\gamma) - [K_A](0) = -\gamma \int \{N_w\}^T \{N_w\} dA`, a
    negative semi-definite matrix, the same for the analytical and the
    numerical kernels"""
    s = make(model)
    s.calc_kA()
    beta, gamma = s.beta, s.gamma
    assert np.isclose(gamma, beta/(2*s.r*np.sqrt(s.Mach**2 - 1)))
    size = s.get_size()
    matrices = modelDB.db[model]['matrices']
    matrices_num = modelDB.db[model]['matrices_num']
    #NOTE fcA gives the upper triangle of -aeromu*int(Nw^T Nw)
    NtN = -finalize_symmetric_matrix(matrices.fcA(1., s, size, 0, 0)).toarray()
    kA = matrices.fkAx(beta, gamma, s, size, 0, 0).toarray()
    kA0 = matrices.fkAx(beta, 0., s, size, 0, 0).toarray()
    assert np.allclose(kA - kA0, -gamma*NtN, atol=1e-12*np.abs(kA).max())
    assert np.linalg.eigvalsh(-gamma*(NtN + NtN.T)/2).max() < 1e-12*gamma*np.abs(NtN).max()
    kA_num = matrices_num.fkAx_num(s, size, 0, 0, 2*s.m, 2*s.n).toarray()
    assert np.allclose(kA_num, kA, atol=1e-10*np.abs(kA).max())


@pytest.mark.parametrize('model', PLATES)
def test_plates_have_no_curvature_term(model):
    s = make(model)
    s.calc_kA()
    assert s.gamma == 0.
    kA = s.calc_kA().toarray()
    s.gamma = 1.e5
    assert np.array_equal(s.calc_kA().toarray(), kA)
    assert s.gamma == 0.
    size = s.get_size()
    matrices = modelDB.db[model]['matrices']
    matrices_num = modelDB.db[model]['matrices_num']
    kA_ana = matrices.fkAx(s.beta, 1.e5, s, size, 0, 0).toarray()
    s.gamma = 1.e5
    kA_num = matrices_num.fkAx_num(s, size, 0, 0, 2*s.m, 2*s.n).toarray()
    assert np.allclose(kA_ana, kA, atol=1e-12*np.abs(kA).max())
    assert np.allclose(kA_num, kA, atol=1e-10*np.abs(kA).max())
