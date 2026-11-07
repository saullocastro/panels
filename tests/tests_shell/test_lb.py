r"""Linear buckling of simply supported and partially free panels

The first load multiplier of a plate under uniform ``Nxx`` or ``Nyy``, for
the CLPT plate and the CLPT cylindrical shells with a large radius:
simply supported (SSSS) symmetric cross-ply laminate against the closed-form
solution of a plate of rectangular orthotropy (Leissa, 1969, eqs. (9.24),
(10.1) and (10.43)-(10.46)); isotropic plate with one free edge parallel to
the load and the other three simply supported against the exact Lévy-type
solution (Leissa, 1969, eqs. (10.52)-(10.56), with zero frequency).

Notes
-----
**SSSS.** The laminate ``[0, 90, 0, 90]s`` is symmetric with `D_{16} = D_{26}
= 0`, a plate of rectangular orthotropy with `D_x = D_{11}`, `D_y = D_{22}`
and `D_{xy} = D_{12} + 2 D_{66}` (eq. (9.24) of [1]_). The deflection `w =
\sin(m \pi x/a) \sin(n \pi y/b)` (eq. (10.43)) solves eq. (10.1) of [1]_ with
`\omega = 0` when

.. math::

    \lambda \left[N_{xx} \left(\frac{m \pi}{a}\right)^2 + N_{yy}
    \left(\frac{n \pi}{b}\right)^2\right] + D_{11} \left(\frac{m
    \pi}{a}\right)^4 + 2 D_{xy} \left(\frac{m \pi}{a}\right)^2
    \left(\frac{n \pi}{b}\right)^2 + D_{22} \left(\frac{n \pi}{b}\right)^4 = 0

the orthotropic form of eqs. (10.44) and (10.46); the reference is the
lowest `\lambda` over `m, n`: 145.760 (``Nxx = -1``, `m = 2`) and 52.1465
(``Nyy = -1``, `m = 1`). The flat plate decouples the membrane and bending
problems, so the in-plane boundary conditions do not enter.

**One free edge.** For the isotropic plate simply supported at ``x = 0, a``
and ``y = 0``, free at ``y = b``, under `N_{xx} = -N`, eq. (10.52) of [1]_,
`w = Y(y) \sin \alpha x` with `\alpha = m \pi/a`, gives eq. (10.53) with
`\omega = 0` and `N_2 = 0`:

.. math::

    Y'''' - 2 \alpha^2 Y'' + \left(\alpha^4 - \frac{N}{D} \alpha^2\right) Y = 0

whose general solution is eq. (10.54). The simply supported edge gives `Y(0)
= Y''(0) = 0`; the free edge gives a zero bending moment and a zero edge
reaction, eq. (10.56) with `N_n = N_{yy} = 0`:

.. math::

    Y''(b) - \nu \alpha^2 Y(b) = 0, \qquad Y'''(b) - (2 - \nu) \alpha^2 Y'(b)
    = 0

The critical `N` is the lowest root over `N` and `m` of the determinant of
these two conditions applied to the two solutions that satisfy the
conditions at `y = 0`. :func:`levy_ssfs` evaluates the solutions at `y = b`
with the matrix exponential of the first-order form of the equation, which
spans the same solutions as eq. (10.54) whatever the signs of `\psi_m^2` and
`\phi_m^2` of eq. (10.55), and finds the root with Brent's method:
`N/(\pi^2 D/b^2) = 0.6493` for ``a/b = 2``, `\nu = 0.33` (`m = 1`). The
second case is the same plate turned by 90 degrees: free edge ``x = a``,
load ``Nyy``.

With ``m = n = 12`` (SSSS) and ``m = 12``, ``n = 13`` (one free edge) the
three models agree with the references within `2 \times 10^{-10}` (``m = n =
8``: `2 \times 10^{-5}`); the tolerance is `10^{-6}`. The cylinders have ``r =
1e8`` and give the plate values.

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, National
   Aeronautics and Space Administration, Washington, DC, 1969.
   https://ntrs.nasa.gov/citations/19700009156

"""
import sys
sys.path.append('../..')

import pytest
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from structsolve import lb
from panels.shell import Shell
from panels.plot_shell import plot_shell


def lb_ss_orthotropic(s, Nxx, Nyy, mmax=10):
    """Closed-form buckling multiplier of a simply supported plate of
    rectangular orthotropy under constant ``Nxx`` and ``Nyy``"""
    D = s.ABD[3:6, 3:6]
    D11, D22, Dxy = D[0, 0], D[1, 1], D[0, 1] + 2*D[2, 2]
    lambdas = []
    for m in range(1, mmax + 1):
        for n in range(1, mmax + 1):
            alpha = m*np.pi/s.a
            beta = n*np.pi/s.b
            K = D11*alpha**4 + 2*Dxy*alpha**2*beta**2 + D22*beta**4
            G = Nxx*alpha**2 + Nyy*beta**2
            if G < 0:
                lambdas.append(-K/G)
    return min(lambdas)


def levy_ssfs(D, nu, a, b, mmax=5):
    """Exact critical compression ``N`` along ``x`` of an isotropic plate
    simply supported at ``x = 0, a`` and ``y = 0``, free at ``y = b``"""
    Ncrs = []
    for m in range(1, mmax + 1):
        alpha = m*np.pi/a

        def det(N):
            # first-order form of Y'''' = 2 alpha^2 Y'' - (alpha^4 - N alpha^2/D) Y
            A = np.zeros((4, 4))
            A[0, 1] = A[1, 2] = A[2, 3] = 1
            A[3, 0] = -(alpha**4 - N*alpha**2/D)
            A[3, 2] = 2*alpha**2
            Z = expm(A*b)[:, [1, 3]]  # solutions with Y(0) = Y''(0) = 0
            My = Z[2] - nu*alpha**2*Z[0]
            Vy = Z[3] - (2 - nu)*alpha**2*Z[1]
            return My[0]*Vy[1] - My[1]*Vy[0]

        Ns = np.linspace(1e-9*D, 10*D*max(alpha, np.pi/b)**2, 4001)
        dets = np.array([det(N) for N in Ns])
        i = np.where(np.sign(dets[:-1]) != np.sign(dets[1:]))[0][0]
        Ncrs.append(brentq(det, Ns[i], Ns[i+1], xtol=1e-14*Ns[i+1], rtol=1e-14))
    return min(Ncrs)


@pytest.mark.slow
def test_shell_lb():
    for model in ['plate_clpt_donnell',
                  'cylshell_clpt_donnell',
                  'cylshell_clpt_sanders',
                  ]:
        # ssss
        s = Shell()
        s.m = 12
        s.n = 12
        s.stack = [0, 90, 0, 90, 90, 0, 90, 0]
        s.plyt = 0.125e-3
        s.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        s.model = model
        s.a = 2.
        s.b = 1.
        s.r = 1.e8
        s.Nxx = -1
        eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
        assert np.isclose(eigvals[0], lb_ss_orthotropic(s, -1., 0.), rtol=1e-6)

        s.Nxx = 0
        s.Nyy = -1
        eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
        assert np.isclose(eigvals[0], lb_ss_orthotropic(s, 0., -1.), rtol=1e-6)

        # ssfs, isotropic
        E = 70e9
        nu = 0.33
        thickness = 1e-3
        D = E*thickness**3/(12*(1 - nu**2))
        s = Shell()
        s.y2u = 1
        s.y2v = 1
        s.y2w = 1
        s.m = 12
        s.n = 13
        s.stack = [0]
        s.plyt = thickness
        s.laminaprop = (E, nu)
        s.model = model
        s.a = 1.
        s.b = 0.5
        s.r = 1.e8
        s.Nxx = -1
        eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
        assert np.isclose(eigvals[0], levy_ssfs(D, nu, s.a, s.b), rtol=1e-6)

        # free edge x = a, load along y
        s.x2u = 1
        s.x2v = 1
        s.x2w = 1
        s.y2u = 0
        s.y2v = 0
        s.y2w = 0
        s.Nxx = 0
        s.Nyy = -1
        eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
        assert np.isclose(eigvals[0], levy_ssfs(D, nu, s.b, s.a), rtol=1e-6)
        plot_shell(s, eigvecs[:, 0], vec='w')


if __name__ == '__main__':
    test_shell_lb()
