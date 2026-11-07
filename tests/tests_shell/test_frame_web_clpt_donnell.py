r"""Web of a ring frame, ``'frame_web_clpt_donnell'``

Free vibration of annular plates clamped on the outer edge and free on the
inner edge against the exact roots of the frequency equation of Leissa
(1969), Eqs. (1.11)-(1.13) and (1.17), and against Leissa (1969), Table
2.21; linear buckling of a simply supported square web of large radius
against the flat plate, `4 \pi^2 D/b^2` (Timoshenko and Gere 1961, Eq.
(h), p. 353); the tangent stiffness against the derivative of the internal
force vector; and the check of the radii.

Notes
-----
With `w = W(\rho) \cos n\theta`, the solution of the free vibration of a
Kirchhoff plate is `W = A J_n(k\rho) + B Y_n(k\rho) + C I_n(k\rho) + D_n
K_n(k\rho)`, Leissa [1]_, Eq. (1.17), with `k^4 = \rho_m h \omega^2/D`. The
annulus clamped at the outer radius `r` (`W = W' = 0`) and free at the inner
radius `\beta r`, with the bending moment and the Kelvin-Kirchhoff edge
reaction of Leissa's Eqs. (1.11)-(1.13),

.. math::

    M_\rho \propto W'' + \nu \left(\frac{W'}{\rho} - \frac{n^2 W}{\rho^2}
    \right) = 0, \qquad
    V_\rho \propto \frac{d}{d\rho} \left(W'' + \frac{W'}{\rho} -
    \frac{n^2 W}{\rho^2}\right) - (1 - \nu) \frac{n^2}{\rho^2}
    \left(W' - \frac{W}{\rho}\right) = 0

has the frequency parameters `\lambda^2 = k^2 r^2 = \omega r^2 \sqrt{\rho_m
h/D}` at the roots of the `4 \times 4` determinant, computed by
:func:`exact_lambda2`. Leissa's Table 2.21 (`\nu = 1/3`, from Vogel and
Skinner, his ref. 2.42) gives these parameters to within 2.1 %: 19.80
against the exact 19.389 for `\beta = 0.4`, `n = 1`, and within 0.5 % for
the other entries used here.

A half annulus whose radial edges have `w = 0` and a free rotation has the
modes `w = W(\rho) \sin n\theta`, `n = 1, 2, ...`, of the full annulus, the
in-plane and out-of-plane problems being uncoupled for an isotropic plate; a
full annulus is made of two half annuli connected with ``'SSycte'``.

For `r \to \infty` the strains of the model are those of
``'plate_clpt_donnell'``, see
:mod:`panels.models.frame_web_clpt_donnell_num`, the difference being of
the order of `a/r`; a simply supported square plate under a uniaxial
compression `N` buckles at `N = 4 \pi^2 D/b^2`, Timoshenko and Gere [2]_,
p. 353, Eq. (h).

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, 1969, Eqs.
   (1.11)-(1.13) and (1.17), p. 2, and Table 2.21, p. 23.
.. [2] Timoshenko, S. P., and Gere, J. M., Theory of Elastic Stability, 2nd
   ed., McGraw-Hill, 1961, p. 353, Eq. (h).

"""
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import iv, ivp, jv, jvp, kv, kvp, yv, yvp
from structsolve import freq, lb

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.models import frame_web_clpt_donnell_num as fw


E, NU, H, RHO = 70e9, 1/3, 1e-3, 2700.
R = 1.
D = E*H**3/(12*(1 - NU**2))
#: Leissa (1969), Table 2.21, lambda^2 for n = 0, 1, 2 nodal diameters
LEISSA = {0.4: (13.54, 19.80, 31.34), 0.6: (25.60, 28.52, 36.60)}


def _det(lam2, n, beta):
    k = np.sqrt(lam2)
    cols = []
    for f, fp in ((jv, jvp), (yv, yvp), (iv, ivp), (kv, kvp)):
        W0, W1 = f(n, k), k*fp(n, k, 1)
        W, W1i, W2, W3 = (f(n, k*beta), k*fp(n, k*beta, 1),
                          k**2*fp(n, k*beta, 2), k**3*fp(n, k*beta, 3))
        b = beta
        M = W2 + NU*(W1i/b - n**2*W/b**2)
        V = (W3 + W2/b - W1i/b**2 - n**2*W1i/b**2 + 2*n**2*W/b**3
             - (1 - NU)*n**2/b**2*(W1i - W/b))
        col = np.array([W0, W1, M, V])
        cols.append(col/np.abs(col).max())
    return np.linalg.det(np.array(cols).T)


def exact_lambda2(n, beta, guess):
    """Root of the frequency equation of the clamped-free annulus near
    ``guess``"""
    return brentq(_det, 0.9*guess, 1.1*guess, args=(n, beta))


def web(beta, b, m=10, n=12):
    s = Shell(model='frame_web_clpt_donnell', a=R*(1 - beta), b=b, r=R,
              m=m, n=n, plyt=H, stack=[0], laminaprop=(E, NU), rho=RHO)
    for f in 'uvw':
        for e in ('x2', 'y1', 'y2'):
            setattr(s, e + f, 1)
            setattr(s, e + f + 'r', 1)
        setattr(s, 'x1' + f, 0)
        setattr(s, 'x1' + f + 'r', 0)
    return s


def lambda2(K, M, num=8):
    eigvals, _ = freq(K, M, sparse_solver=False, silent=True,
                      num_eigvalues=num)
    return np.sort(np.sqrt(-eigvals.real))*R**2*np.sqrt(RHO*H/D)


@pytest.mark.parametrize('beta', [0.4, 0.6])
def test_exact_roots_against_leissa_table(beta):
    """The roots of the frequency equation against Leissa (1969), Table
    2.21, within 2.1 %"""
    for n, ref in enumerate(LEISSA[beta]):
        assert np.isclose(exact_lambda2(n, beta, ref), ref, rtol=0.022)


@pytest.mark.parametrize('beta', [0.4, 0.6])
def test_half_annulus(beta):
    """Half annulus, `n = 1, 2`, against the exact roots (Leissa 1969,
    Eqs. (1.11)-(1.13) and (1.17))"""
    s = web(beta, np.pi*R)
    s.y1w = 0
    s.y2w = 0
    l2 = lambda2(s.calc_kC(silent=True), s.calc_kM(silent=True))
    ref = [exact_lambda2(n, beta, LEISSA[beta][n]) for n in (1, 2)]
    assert np.allclose(l2[:2], ref, rtol=1e-5)


@pytest.mark.parametrize('beta', [0.4, 0.6])
def test_full_annulus(beta):
    """Two half annuli with ``'SSycte'``, `n = 0, 1, 1, 2, 2`, against the
    exact roots (Leissa 1969, Eqs. (1.11)-(1.13) and (1.17))"""
    p1 = web(beta, np.pi*R)
    p2 = web(beta, np.pi*R)
    conn = [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0),
            dict(p1=p1, p2=p2, func='SSycte', ycte1=0, ycte2=p2.b)]
    md = MultiDomain([p1, p2], conn)
    l2 = lambda2(md.reduce(md.calc_kC(silent=True)),
                 md.reduce(md.calc_kM(silent=True)))
    ref = np.array([exact_lambda2(n, beta, LEISSA[beta][n])
                    for n in (0, 1, 1, 2, 2)])
    assert np.allclose(l2[:5], ref, rtol=1e-5)


@pytest.mark.parametrize('load', ['Nxx', 'Nyy'])
def test_plate_limit_buckling(load):
    """Simply supported square web with `r = 10^4 a` against `4 \\pi^2
    D/b^2` (Timoshenko and Gere 1961, Eq. (h), p. 353)"""
    a = 1.
    s = Shell(model='frame_web_clpt_donnell', a=a, b=a, r=1e4*a, m=10, n=10,
              plyt=H, stack=[0], laminaprop=(E, NU))
    for e in ('x1', 'x2', 'y1', 'y2'):
        setattr(s, e + 'w', 0)
    setattr(s, load, -1.)
    eigvals, _ = lb(s.calc_kC(silent=True), s.calc_kG(silent=True),
                    silent=True, num_eigvalues=4)
    assert np.isclose(eigvals[0], 4*np.pi**2*D/a**2, rtol=3e-4)


def test_tangent_is_derivative_of_fint():
    """``fkC_num(NLgeom=1) + fkG_num`` against central differences of
    ``calc_fint``"""
    s = web(0.5, np.pi/2*R, m=6, n=6)
    size = s.get_size()
    c = np.random.default_rng(1).standard_normal(size)*1e-3
    fint = lambda cc: fw.calc_fint(cc, s.ABD, s, size, 0, s.nx, s.ny)
    kT = (s.calc_kC(c=c, NLgeom=True, silent=True)
          + s.calc_kG(c=c, NLgeom=True, silent=True)).toarray()
    dc = 1e-7
    J = np.column_stack([(fint(c + dc*e) - fint(c - dc*e))/(2*dc)
                         for e in np.eye(size)])
    assert np.abs(kT - J).max() < 1e-8*np.abs(kT).max()


def test_r_not_larger_than_a_raises():
    """An inner radius `r - a \\le 0` is rejected"""
    s = Shell(model='frame_web_clpt_donnell', a=1., b=1., r=1., m=4, n=4,
              plyt=H, stack=[0], laminaprop=(E, NU))
    with pytest.raises(ValueError):
        s.calc_kC(silent=True)
