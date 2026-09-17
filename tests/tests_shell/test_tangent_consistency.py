"""The tangent stiffness matrix must be the derivative of the internal force

KT = calc_kC(c, NLgeom=True) + calc_kG(c, NLgeom=True), with

    calc_kC -> K0 + K0L + KL0 + KLL + KGNL
    calc_kG -> KG(N0 + N_L)

is what every Newton-Raphson built on the Shell class uses as the Jacobian of
calc_fint. If the two drift apart, the analyses still converge, but linearly
instead of quadratically.

The main check is a directional Taylor test,

    |fint(c + h d) - fint(c) - h KT d| / |h KT d|

which falls proportionally to h for a consistent tangent and plateaus for an
inconsistent one.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest

from panels.shell import Shell

DOF = 3
MODELS = ['plate_clpt_donnell', 'cylshell_clpt_donnell']


def make_shell(model):
    s = Shell()
    s.model = model
    s.a = 0.3
    s.b = 0.2
    s.r = 0.4 if model.startswith('cylshell') else None
    # unsymmetric and unbalanced, such that A16, A26 and B are non-zero
    s.stack = [30, -45, 0, 90]
    s.plyt = 0.125e-3
    E11 = 142.5e9
    E22 = E11/20
    G12 = 0.5*E22
    s.laminaprop = (E11, E22, 0.25, G12, G12, G12)
    s.m = 5
    s.n = 5
    s.nx = 8
    s.ny = 8
    s.Nxx = s.Nyy = s.Nxy = 0.
    s.Nxx_cte = s.Nyy_cte = s.Nxy_cte = 0.
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for field in 'uvw':
            setattr(s, edge + field, 0. if field == 'w' else 1.)
            setattr(s, edge + field + 'r', 1.)
    # removing the in-plane rigid-body motions
    s.x1u = 0.
    s.y1v = 0.
    s._rebuild()
    return s


def random_state(s, rng):
    """In-plane amplitudes of the order of the thickness, and deflections
    well above it, such that the nonlinear terms carry weight"""
    h_shell = len(s.stack)*s.plyt
    c = h_shell*rng.standard_normal(s.get_size())
    c[2::DOF] *= 20
    return c


def make_callables(model):
    s = make_shell(model)

    def fint(c):
        return np.asarray(s.calc_fint(c=np.ascontiguousarray(c)), dtype=float)

    def KT(c):
        return s.calc_kT(c=np.ascontiguousarray(c)).toarray()

    return s, fint, KT


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_tangent_is_derivative_of_fint(model, seed):
    """Taylor test: the error must fall by about ten for each decade of h."""
    s, fint, KT = make_callables(model)
    rng = np.random.default_rng(seed)
    c = random_state(s, rng)
    d = random_state(s, rng)

    f0 = fint(c)
    KTd = KT(c) @ d
    scale = np.linalg.norm(KTd)
    assert scale > 0

    steps = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
    errors = [np.linalg.norm(fint(c + h*d) - f0 - h*KTd)/(h*scale)
              for h in steps]

    # a consistent tangent leaves a second order remainder, so the error
    # falls by ten for every decade of h; an inconsistent one leaves a first
    # order remainder, so the error plateaus instead
    for h, prev, err in zip(steps[1:], errors[:-1], errors[1:]):
        assert err < 0.2*prev, (
            'error did not fall by ten going to h=%.0e: %.3e -> %.3e; the '
            'tangent is not the derivative of fint' % (h, prev, err))

    # error/h is the constant that multiplies the second derivative, so it
    # must not drift with h
    coefficients = [err/h for h, err in zip(steps, errors)]
    spread = max(coefficients)/min(coefficients)
    assert spread < 2., (
        'error/h drifted by a factor %.1f over %.0e..%.0e, so the remainder '
        'is not second order' % (spread, steps[0], steps[-1]))

    assert errors[-1] < 1.e-4, (
        'residual %.3e at h=%.0e is too large for a consistent tangent'
        % (errors[-1], steps[-1]))


@pytest.mark.parametrize('model', MODELS)
def test_tangent_matches_finite_difference_jacobian(model):
    """Every entry of KT, against a central difference of fint."""
    s, fint, KT = make_callables(model)
    rng = np.random.default_rng(7)
    c = random_state(s, rng)
    N = c.shape[0]

    step = 1.e-8
    J = np.empty((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = step
        J[:, j] = (fint(c + e) - fint(c - e))/(2*step)

    K = KT(c)
    err = np.linalg.norm(K - J)/np.linalg.norm(J)
    assert err < 1.e-5, 'KT differs from d(fint)/dc by %.3e' % err


@pytest.mark.parametrize('model', MODELS)
def test_fint_and_tangent_in_the_undeformed_state(model):
    """At c = 0 there is no internal force, and KT reduces to K0."""
    s, fint, KT = make_callables(model)
    c0 = np.zeros(s.get_size())
    assert np.all(fint(c0) == 0)
    K0 = s.calc_kC(c=c0).toarray()
    assert np.abs(K0).max() > 0
    assert np.abs(KT(c0) - K0).max() <= 1.e-12*np.abs(K0).max()


@pytest.mark.parametrize('model', MODELS)
def test_fint_is_linear_for_small_displacements(model):
    """For a state well below the thickness, fint approaches K0 @ c."""
    s, fint, KT = make_callables(model)
    rng = np.random.default_rng(3)
    c = 1.e-6*random_state(s, rng)
    K0 = s.calc_kC(c=np.zeros_like(c)).toarray()
    assert np.linalg.norm(fint(c) - K0 @ c)/np.linalg.norm(K0 @ c) < 1.e-5


@pytest.mark.parametrize('model', MODELS)
def test_newton_raphson_converges_quadratically(model):
    """Newton-Raphson around a known equilibrium must square the error.

    The external force is manufactured as fext = fint(c_eq), such that c_eq is
    an equilibrium state by construction, with deflections well above the
    thickness. Starting nearby, a consistent tangent gives e_k+1 ~ C e_k^2,
    independently of any postbuckling branch selection.
    """
    s, fint, KT = make_callables(model)
    rng = np.random.default_rng(13)
    c_eq = random_state(s, rng)
    # DOFs removed by the boundary conditions carry no stiffness
    used = ~np.isclose(KT(c_eq).diagonal(), 0)
    c_eq[~used] = 0
    fext = fint(c_eq)

    c = c_eq + 1.e-2*random_state(s, rng)
    c[~used] = 0
    errors = [np.linalg.norm(c - c_eq)/np.linalg.norm(c_eq)]
    # stopping well above round-off, which is about 1.e-12 here
    while errors[-1] > 1.e-8 and len(errors) < 10:
        K = KT(c)[np.ix_(used, used)]
        R = fint(c) - fext
        c[used] -= np.linalg.solve(K, R[used])
        errors.append(np.linalg.norm(c - c_eq)/np.linalg.norm(c_eq))

    # from 1.e-2 to below 1.e-8 takes 3 iterations with a quadratic rate, and
    # about 10 or more with a linear rate
    assert len(errors) <= 4, 'too many iterations: %s' % errors
    # the order log(e_k+1)/log(e_k) is 2 for quadratic and 1 for linear rates
    orders = [np.log(e1)/np.log(e0) for e0, e1 in zip(errors[:-1], errors[1:])]
    assert min(orders) > 1.5, 'convergence is not quadratic: %s' % errors


@pytest.mark.parametrize('model', MODELS)
def test_kG_is_homogeneous_of_degree_one(model):
    """kG(2c) = 2 kG(c), required by linear buckling; KGNL belongs to kC."""
    s = make_shell(model)
    rng = np.random.default_rng(5)
    c = random_state(s, rng)
    kG1 = s.calc_kG(c=c, NLgeom=True).toarray()
    kG2 = s.calc_kG(c=2*c, NLgeom=True).toarray()
    assert np.linalg.norm(kG2 - 2*kG1) <= 1.e-10*np.linalg.norm(kG2)


if __name__ == '__main__':
    for model in MODELS:
        s, fint, KT = make_callables(model)
        rng = np.random.default_rng(0)
        c = random_state(s, rng)
        d = random_state(s, rng)
        f0 = fint(c)
        KTd = KT(c) @ d
        print(model)
        prev = None
        for h in (1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6, 1.e-7):
            err = (np.linalg.norm(fint(c + h*d) - f0 - h*KTd)
                   / np.linalg.norm(h*KTd))
            ratio = '' if prev is None else '   (x %.2f)' % (err/prev)
            print('   h %.0e   rel err %.6e%s' % (h, err, ratio))
            prev = err
