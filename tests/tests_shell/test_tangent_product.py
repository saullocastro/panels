r"""The product of the tangent stiffness matrix by a vector, without assembly

:meth:`.Shell.calc_kTv` returns `[K_T(c)] \{v\}` from the kernels
``calc_kTv`` of ``theory/shells/tangent_product``, at the cost of about one
:meth:`.Shell.calc_fint`. It must be the product of the assembled tangent
stiffness matrix of :meth:`.Shell.calc_kT`, i.e. the directional derivative
of :meth:`.Shell.calc_fint`. The checks, for all 9 models, are those of
``test_tangent_consistency.py`` applied to the product:

1. the product equals ``calc_kT(c) @ v`` to round-off, also with a constant
   membrane stress state, one constitutive matrix per integration point, a
   partial integration domain, an offset in a larger vector, and follower
   pressure loads;
2. directional Taylor test of ``calc_fint``, whose remainder must fall as
   `O(h^2)`;
3. the matrix built column by column from the product against the central
   difference of ``calc_fint``;
4. in the undeformed state the product is ``K0 @ v``;
5. Newton-Raphson with the tangent built from the product converges
   quadratically.

"""
import sys
sys.path.append('../..')

import numpy as np
from scipy.linalg import eigh
import pytest

from panels import modelDB
from test_tangent_consistency import MODELS, make_shell, random_state
from test_follower_pressure import add_pressure


def rel(a, b):
    return np.linalg.norm(a - b)/np.linalg.norm(b)


def make_callables(model):
    s = make_shell(model)

    def fint(c):
        return np.asarray(s.calc_fint(c=np.ascontiguousarray(c)), dtype=float)

    def kTv(c, v):
        return s.calc_kTv(c, v)

    def KT(c):
        return s.calc_kT(c=np.ascontiguousarray(c)).toarray()

    return s, fint, kTv, KT


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_product_equals_assembled_tangent(model, seed):
    s, fint, kTv, KT = make_callables(model)
    rng = np.random.default_rng(seed)
    c = random_state(s, rng)
    v = random_state(s, rng)
    assert rel(kTv(c, v), KT(c) @ v) < 1.e-12


@pytest.mark.parametrize('model', MODELS)
def test_product_with_constant_stress_per_point_properties_and_offset(model):
    """``Nxx_cte``, ``Nyy_cte``, ``Nxy_cte``, ``ABDnxny``, a partial domain,
    and the shell placed at ``col0`` of a larger vector"""
    s = make_shell(model)
    s.Nxx_cte, s.Nyy_cte, s.Nxy_cte = -30., 12., 7.
    s.x1, s.x2, s.y2 = 0.04, 0.25, 0.17
    rng = np.random.default_rng(4)
    c = random_state(s, rng)
    v = random_state(s, rng)
    ABDnxny = np.empty((s.nx, s.ny) + s.ABD.shape)
    for i in range(s.nx):
        for j in range(s.ny):
            ABDnxny[i, j] = s.ABD*(1 + 0.1*i - 0.05*j)
    KT = s.calc_kT(c=c, ABDnxny=ABDnxny).toarray()
    assert rel(s.calc_kTv(c, v, ABDnxny=ABDnxny), KT @ v) < 1.e-12

    size = s.get_size()
    col0 = 7
    C = np.zeros(size + 11)
    V = np.zeros(size + 11)
    C[col0:col0 + size] = c
    V[col0:col0 + size] = v
    out = s.calc_kTv(C, V, size=size + 11, col0=col0, ABDnxny=ABDnxny)
    assert np.all(out[:col0] == 0) and np.all(out[col0 + size:] == 0)
    assert rel(out[col0:col0 + size], KT @ v) < 1.e-12

    with pytest.raises(ValueError):
        s.calc_kTv(c, v[:-1])


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('follower', ['linear', 'quadratic'])
def test_product_with_follower_pressure(model, follower):
    """Dead and follower pressures, constant and incremented, at a load
    factor other than one; the follower part is a central difference, exact
    since the follower force is at most quadratic in ``c``"""
    s = make_shell(model)
    add_pressure(s, 'constant', follower, cte=True)
    add_pressure(s, 'patch', follower, cte=False)
    add_pressure(s, 'function', False, cte=False)
    rng = np.random.default_rng(9)
    c = random_state(s, rng)
    for v in (random_state(s, rng), 1.e-9*random_state(s, rng)):
        KT = s.calc_kT(c=c, inc=0.7).toarray()
        assert rel(s.calc_kTv(c, v, inc=0.7), KT @ v) < 1.e-11
    assert not np.any(s.calc_kTv(c, np.zeros_like(c), inc=0.7))


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_product_is_derivative_of_fint(model, seed):
    """Taylor test: the error must fall by about ten for each decade of h."""
    s, fint, kTv, KT = make_callables(model)
    rng = np.random.default_rng(seed)
    c = random_state(s, rng)
    d = random_state(s, rng)

    f0 = fint(c)
    KTd = kTv(c, d)
    scale = np.linalg.norm(KTd)
    assert scale > 0

    steps = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]
    errors = [np.linalg.norm(fint(c + h*d) - f0 - h*KTd)/(h*scale)
              for h in steps]
    for h, prev, err in zip(steps[1:], errors[:-1], errors[1:]):
        assert err < 0.2*prev, (
            'error did not fall by ten going to h=%.0e: %.3e -> %.3e; the '
            'product is not the derivative of fint' % (h, prev, err))
    coefficients = [err/h for h, err in zip(steps, errors)]
    assert max(coefficients)/min(coefficients) < 2.
    assert errors[-1] < 1.e-4


@pytest.mark.parametrize('model', MODELS)
def test_product_matches_finite_difference_jacobian(model):
    """Every column ``kTv(c, e_j)`` against a central difference of fint."""
    s, fint, kTv, KT = make_callables(model)
    rng = np.random.default_rng(7)
    c = random_state(s, rng)
    N = c.shape[0]

    step = 1.e-8
    J = np.empty((N, N))
    K = np.empty((N, N))
    for j in range(N):
        e = np.zeros(N)
        e[j] = step
        J[:, j] = (fint(c + e) - fint(c - e))/(2*step)
        e[j] = 1.
        K[:, j] = kTv(c, e)

    assert np.linalg.norm(K - J)/np.linalg.norm(J) < 1.e-5
    assert np.linalg.norm(K - KT(c))/np.linalg.norm(K) < 1.e-12


@pytest.mark.parametrize('model', MODELS)
def test_product_in_the_undeformed_state(model):
    """At c = 0 the product is K0 @ v."""
    s, fint, kTv, KT = make_callables(model)
    rng = np.random.default_rng(3)
    v = random_state(s, rng)
    c0 = np.zeros(s.get_size())
    K0 = s.calc_kC(c=c0).toarray()
    assert rel(kTv(c0, v), K0 @ v) < 1.e-12


@pytest.mark.parametrize('model', MODELS)
def test_newton_raphson_converges_quadratically(model):
    """Newton-Raphson around a known equilibrium, with the tangent built from
    the product, must square the error; see the test of the same name in
    ``test_tangent_consistency.py``"""
    s, fint, kTv, KT = make_callables(model)

    def KT_from_product(c):
        N = c.shape[0]
        K = np.empty((N, N))
        for j in range(N):
            e = np.zeros(N)
            e[j] = 1.
            K[:, j] = kTv(c, e)
        return K

    for seed in range(13, 33):
        rng = np.random.default_rng(seed)
        c_eq = random_state(s, rng)
        used = ~np.isclose(KT(c_eq).diagonal(), 0)
        c_eq[~used] = 0
        K0 = s.calc_kC(c=np.zeros_like(c_eq)).toarray()[np.ix_(used, used)]
        lam = eigh(KT(c_eq)[np.ix_(used, used)], K0, eigvals_only=True)
        if np.abs(lam).min() > 0.05:
            break
    else:
        raise AssertionError('no regular equilibrium state found')
    fext = fint(c_eq)

    c = c_eq + 1.e-2*random_state(s, rng)
    c[~used] = 0
    errors = [np.linalg.norm(c - c_eq)/np.linalg.norm(c_eq)]
    while errors[-1] > 1.e-8 and len(errors) < 12:
        K = KT_from_product(c)[np.ix_(used, used)]
        R = fint(c) - fext
        c[used] -= np.linalg.solve(K, R[used])
        errors.append(np.linalg.norm(c - c_eq)/np.linalg.norm(c_eq))

    assert len(errors) - 1 <= 6, 'too many iterations: %s' % errors
    floor = 1.e-9 if modelDB.db[model]['dofs'] == 5 else 1.e-12
    orders = [np.log(e2/e1)/np.log(e1/e0)
              for e0, e1, e2 in zip(errors[:-2], errors[1:-1], errors[2:])
              if e2 > floor]
    assert len(orders) >= 1, 'no step above the round-off floor: %s' % errors
    assert min(orders) > 1.5, 'convergence is not quadratic: %s' % errors
