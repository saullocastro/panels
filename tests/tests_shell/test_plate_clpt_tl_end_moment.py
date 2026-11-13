r"""Cantilever strip rolled into a circle by an end moment

Benchmark 3.2 of Sze, Liu and Lo (2004), Fig. 3a and Table 3a: a strip
clamped at ``x = 0``, ``L = 12``, ``b = 1``, ``h = 0.1``, ``E = 1.2e6``,
``nu = 0``, under an end moment ``M`` up to ``M_max = 2 pi M_o``, ``M_o =
EI/L = 25/3``, which bends it into a full circle. The tip displacements of
``'plate_clpt_tl'`` are checked against the cylindrical solution of the
model, :func:`model_arc`, and against the inextensible circular arc of the
benchmark, :func:`inextensible_arc`, tabulated as "Exact" in Table 3a.
panels has no applied moments: the end moment is applied as a conservative
load that depends on the configuration, :class:`EndMoment`.

Notes
-----
The exact solution of the inextensible Euler-Bernoulli beam is the circular
arc of curvature ``M/EI``, p. 1555 of [1]_:

.. math::

    \frac{U}{L} = \frac{\sin\theta}{\theta} - 1, \qquad
    \frac{W}{L} = \frac{1 - \cos\theta}{\theta}, \qquad
    \theta = \frac{M L}{E I} = 2 \pi \frac{M}{M_{max}}

tabulated as "Exact" in Table 3a (p. 1557). (The text of p. 1555 says that
``M_max`` is ``M_o``, a misprint: Fig. 3a, Table 3a and the full circle
are ``2 pi M_o``.)

The end moment is applied as a conservative load of potential

.. math::

    \Pi_{ext} = -\frac{M}{b} \int_0^b \theta(L, y)\, dy, \qquad
    \theta = \operatorname{atan2}(w_{,x}, 1 + u_{,x})

the rotation of the tangent of the deformed axis at the tip, which is the
work-conjugate of the moment of the beam theory for any rotation. Its load
vector ``g(c) = -dPi_ext/dc`` depends on the configuration, and its
derivative, the load stiffness, is subtracted from ``calc_kC``. Through the
callables of ``structsolve.Analysis`` (``structsolve/callbacks.py``), the
residual being ``calc_fint(c, inc) - inc calc_fext()``, the part ``g(c) -
g(0)`` enters ``calc_fint`` and ``calc_fext(c=c)`` returns ``g(c)``, for a
unit load factor.

``'plate_clpt_tl'`` does not give exactly the inextensible arc: its
curvature is measured with the area vector, ``kxx = -a . u,xx``, with
``|a| = lambda mu`` for the stretches ``lambda`` along ``x`` and ``mu``
along ``y``, so that for the cylindrical deformation of the strip, of
rotation ``theta' = k`` per unit length,

.. math::

    \kappa_{xx} = \lambda^2 \mu\, k, \qquad
    \varepsilon_{xx} = \frac{\lambda^2 - 1}{2}, \qquad
    \varepsilon_{yy} = \frac{\mu^2 - 1}{2}

and the stationarity of `\int (D \kappa_{xx}^2 + A \varepsilon_{xx}^2 + A
\varepsilon_{yy}^2)/2\, dA - M k L` (``nu = 0``, ``D = E h^3/12``, ``A = E
h``) gives the arc of :func:`model_arc`: the bending moment shortens the
strip and its width, `\lambda - 1 \approx -(\kappa h)^2/6` and `\mu - 1
\approx -(\kappa h)^2/12`, and the curvature `k = M/(D b \lambda^4 \mu^2)`
is larger than ``M/EI`` by about `5 (\kappa h)^2/6`, 0.23 % at the full
circle (`\kappa h = 0.052`), which turns the tip by 0.014 rad more. The
Ritz solution converges to this arc with ``m``, and differs from the
inextensible one by at most 2.3e-3 L.

References
----------
.. [1] Sze, K.Y., Liu, X.H., Lo, S.H., 2004. Popular benchmark problems for
   geometric nonlinear analysis of shells. Finite Elements in Analysis and
   Design 40, 1551-1569. https://doi.org/10.1016/j.finel.2003.11.001

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from scipy.optimize import fsolve
from structsolve import Analysis

from panels.bardell import calc_vec_f, calc_vec_fp
from panels.shell import Shell

TL = 'plate_clpt_tl'
GL = 'plate_clpt_gl'

L, B, H, E = 12., 1., 0.1, 1.2e6
EI = E*B*H**3/12
M_O = EI/L
M_MAX = 2*np.pi*M_O

# Table 3a of [1], p. 1557, columns "Exact": M/M_max, -U_tip, W_tip
TABLE_3A_EXACT = {0.05: (0.196, 1.870), 0.25: (4.361, 7.639),
                  0.50: (12.000, 7.639), 0.75: (14.546, 2.546),
                  1.00: (12.000, 0.000)}


def inextensible_arc(M):
    r"""Tip displacements ``(U, W)`` of the inextensible beam, p. 1555 of
    [1]"""
    t = M*L/EI
    return L*(np.sin(t)/t - 1), L*(1 - np.cos(t))/t


def model_arc(M):
    r"""Tip displacements ``(U, W)`` of the uniform cylindrical solution of
    ``'plate_clpt_tl'``, the stationary point of the energy of the module
    docstring, per unit width: ``D lambda^4 mu^2 k = M/b``, ``2 D lambda^3
    mu^2 k^2 + A exx lambda = 0``, ``D lambda^4 mu k^2 + A eyy mu = 0``,
    the tip at ``lambda/k (sin kL, 1 - cos kL)``"""
    D, A = E*H**3/12, E*H

    def eqs(x):
        lam, mu, k = x
        exx, eyy = (lam**2 - 1)/2, (mu**2 - 1)/2
        return [D*lam**4*mu**2*k - M/B, 2*D*lam**3*mu**2*k**2 + A*exx*lam,
                D*lam**4*mu*k**2 + A*eyy*mu]

    lam, mu, k = fsolve(eqs, [1., 1., M/EI], xtol=1e-12)
    return lam/k*np.sin(k*L) - L, lam/k*(1 - np.cos(k*L))


class EndMoment:
    r"""Conservative end moment at ``x = a``, of potential ``-M/b int
    theta(a, y) dy``, ``theta = atan2(w,x, 1 + u,x)``, integrated with
    Gauss-Legendre points along ``y``

    ``g(c)`` is ``-dPi/dc``, the load vector, and ``K(c)`` its derivative,
    symmetric; with ``A = 1 + u,x``, ``W = w,x``, ``r^2 = A^2 + W^2``:
    ``dtheta = (A dW - W dA)/r^2``, ``theta_AA = -theta_WW = 2 A W/r^4``,
    ``theta_AW = (W^2 - A^2)/r^4``.
    """
    def __init__(self, s, M, ny=None):
        self.M = M
        self.b = s.b
        ny = ny or s.n + 1
        size = s.get_size()
        eta, wy = np.polynomial.legendre.leggauss(ny)
        self.wy = wy*s.b/2
        fu = calc_vec_fp(1., s.x1u, s.x1ur, s.x2u, s.x2ur)*2/s.a
        fw = calc_vec_fp(1., s.x1w, s.x1wr, s.x2w, s.x2wr)*2/s.a
        self.Ux = np.zeros((ny, size))
        self.Wx = np.zeros((ny, size))
        for k, e in enumerate(eta):
            gu = calc_vec_f(e, s.y1u, s.y1ur, s.y2u, s.y2ur)
            gw = calc_vec_f(e, s.y1w, s.y1wr, s.y2w, s.y2wr)
            for j in range(s.n):
                for i in range(s.m):
                    col = 3*(j*s.m + i)
                    self.Ux[k, col] = fu[i]*gu[j]
                    self.Wx[k, col + 2] = fw[i]*gw[j]

    def _gradients(self, c):
        A = 1 + self.Ux@c
        W = self.Wx@c
        return A, W, A**2 + W**2

    def g(self, c):
        A, W, r2 = self._gradients(c)
        dth = (A/r2)[:, None]*self.Wx - (W/r2)[:, None]*self.Ux
        return self.M/self.b*(self.wy@dth)

    def K(self, c):
        A, W, r2 = self._gradients(c)
        taa = self.wy*2*A*W/r2**2
        taw = self.wy*(W**2 - A**2)/r2**2
        Ux, Wx = self.Ux, self.Wx
        out = ((Ux.T*taa)@Ux - (Wx.T*taa)@Wx
               + (Ux.T*taw)@Wx + (Wx.T*taw)@Ux)
        return self.M/self.b*out


def strip(model, m, n=4):
    r"""The strip of [1], clamped at ``x = 0`` (``u = v = w = w,x = 0``),
    free elsewhere; ``n = 4`` terms along ``y`` contain the linear
    functions"""
    s = Shell(model=model, a=L, b=B, stack=[0], plyt=H, laminaprop=(E, 0.),
              m=m, n=n)
    for edge in ('x2', 'y1', 'y2'):
        for f in 'uvw':
            setattr(s, edge + f, 1.)
            setattr(s, edge + f + 'r', 1.)
    s.x1u = s.x1v = s.x1w = s.x1wr = 0.
    return s


def end_moment_analysis(model, m, inc=0.05):
    r"""Newton-Raphson analysis up to ``M_max`` in increments ``inc``;
    returns ``{M/M_max: (U, W)}`` at the tip, averaged over the width, for
    the converged increments"""
    s = strip(model, m)
    load = EndMoment(s, M_MAX)
    size = s.get_size()
    g0 = load.g(np.zeros(size))

    def calc_fext(inc=1., silent=True, c=None):
        return inc*(g0 if c is None else load.g(c))

    def calc_fint(c, silent=True, inc=1.):
        return s.calc_fint(c) - inc*(load.g(c) - g0)

    def calc_kC(c=None, NLgeom=False, silent=True, inc=1.):
        kC = s.calc_kC(c=c, NLgeom=NLgeom)
        return kC if c is None else kC - inc*load.K(c)

    def calc_kG(c=None, NLgeom=False, silent=True):
        return s.calc_kG(c=c, NLgeom=NLgeom)

    an = Analysis(calc_fext, calc_fint, calc_kC, calc_kG)
    an.initialInc = inc
    an.maxInc = inc
    an.static(NLgeom=True, silent=True)
    out = {}
    for lbd, c in zip(an.increments, an.cs):
        f = s.uvw(c, xs=np.full(3, L), ys=np.linspace(0, B, 3))[1]
        out[round(lbd, 6)] = (f['u'].mean(), f['w'].mean())
    return out


def test_load_stiffness_is_derivative_of_load_vector():
    s = strip(TL, 6)
    load = EndMoment(s, 1.)
    rng = np.random.default_rng(0)
    c = 0.3*rng.standard_normal(s.get_size())
    K = load.K(c)
    step = 1e-6
    FD = np.column_stack([(load.g(c + step*e) - load.g(c - step*e))/(2*step)
                          for e in np.eye(c.size)])
    assert np.abs(K - FD).max() < 1e-7*np.abs(K).max()
    assert np.allclose(K, K.T, rtol=0, atol=1e-14*np.abs(K).max())


@pytest.mark.slow
def test_end_moment_rolls_strip_into_circle():
    r"""``'plate_clpt_tl'``, ``m = 12``, 20 increments of 0.05 ``M_max``:
    at 0.05, 0.25, 0.5, 0.75 and 1 ``M_max`` the tip is that of the
    cylindrical solution of the model, :func:`model_arc`, to 2e-5 L (the
    Ritz error, observed 4e-6 L, is 3e-4 L with ``m = 10`` and 2e-2 L with
    ``m = 8`` at the full circle, the angle `2 \pi` needing polynomials of
    degree 11 or more), and that of the inextensible arc of [1] to 2.5e-3 L
    (observed 2.3e-3 L at the full circle), the stretch of the model of the
    order of `(\kappa h)^2`; the tabulated "Exact" of Table 3a, three
    decimals, within the same tolerance plus their round-off.

    ``'plate_clpt_gl'``, whose curvatures ``-w,xx`` are measured in the fixed
    frame, is already 7 % more flexible in ``W`` at 0.05 ``M_max``, a tip
    rotation of 0.31 rad (bending energy times about `\cos^2\theta`), and
    its Newton-Raphson analysis stops before 0.1 ``M_max``"""
    tl = end_moment_analysis(TL, 12)
    assert max(tl) == 1.
    for f, (mU, W_table) in TABLE_3A_EXACT.items():
        U, W = tl[f]
        Um, Wm = model_arc(f*M_MAX)
        Ue, We = inextensible_arc(f*M_MAX)
        print(f, (-U, W), 'model', (-Um, Wm), 'exact', (-Ue, We))
        assert abs(U - Um) < 2e-5*L and abs(W - Wm) < 2e-5*L
        assert abs(U - Ue) < 2.5e-3*L and abs(W - We) < 2.5e-3*L
        assert abs(-U - mU) < 2.5e-3*L + 5e-4
        assert abs(W - W_table) < 2.5e-3*L + 5e-4
    gl = end_moment_analysis(GL, 12)
    Ue, We = inextensible_arc(0.05*M_MAX)
    print(GL, gl)
    assert gl[0.05][1] > 1.03*We
