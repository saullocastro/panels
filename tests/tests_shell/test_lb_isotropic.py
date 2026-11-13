r"""Linear buckling of an isotropic cylindrical panel under axial compression

``cylshell_clpt_donnell``, a simply supported isotropic cylindrical panel
under uniform ``Nxx``, with the in-plane boundary conditions under which a
double sine series is the exact solution of Donnell's equation: the first
load multiplier against the closed-form critical load of Batdorf (1947),
Appendix D, eq. (D1) and boundary conditions (D3)-(D14). The first mode is
plotted.

Notes
-----
For a curved panel of length `a`, arc width `b`, radius `r` and thickness
`h`, under the axial compression `\sigma_x h = -N_{xx}`, Donnell's equation
(Appendix D of [1]_, p. 21)

.. math::

    D \nabla^8 w + \frac{E h}{r^2} \frac{\partial^4 w}{\partial x^4} +
    \sigma_x h \nabla^4 \frac{\partial^2 w}{\partial x^2} = 0

is solved exactly by `w = w_0 \sin(m \pi x/a) \sin(n \pi y/b)`, eq. (D1),
with `w = 0` and zero moments on every edge, the motion parallel to each edge
prevented and the motion normal to each edge in the plane of the shell free,
eqs. (D3)-(D14) of [1]_: ``u`` free and ``v = 0`` on ``x = 0, a``, ``u = 0``
and ``v`` free on ``y = 0, b``. Substituting, with `\alpha = m \pi/a`,
`\beta = n \pi/b`:

.. math::

    N_{cr} = \min_{m, n} \left[\frac{D (\alpha^2 + \beta^2)^2}{\alpha^2} +
    \frac{E h}{r^2} \frac{\alpha^2}{(\alpha^2 + \beta^2)^2}\right], \qquad
    D = \frac{E h^3}{12 (1 - \nu^2)}

here `N_{cr} = 1.3711 \times 10^6` (`m = 7`, `n = 2`, curvature parameter
`Z = b^2 \sqrt{1 - \nu^2}/(r h) = 59`). The model converges to it: relative
differences `4.9 \times 10^{-3}`, `3.4 \times 10^{-4}`, `2.3 \times 10^{-7}`
and `6 \times 10^{-13}` for ``m = n = 12, 16, 20, 25``; with ``m = n = 20``
the tolerance is `10^{-5}`. (``cylshell_clpt_sanders`` converges to a value
0.10 % lower, a difference of the shell theories.)

References
----------
.. [1] Batdorf, S. B., A Simplified Method of Elastic-Stability Analysis
   for Thin Cylindrical Shells, NACA Report 874, National Advisory
   Committee for Aeronautics, 1947.
   https://ntrs.nasa.gov/citations/19930091945

"""
import sys
sys.path.append('../..')

import pytest
import numpy as np
from structsolve import lb
import matplotlib

from panels.shell import Shell
from panels.plot_shell import plot_shell


@pytest.mark.slow
def test_lb_isotropic():
    s = Shell()
    s.m = 20
    s.n = 20
    thickness = 8e-3
    E = 70e9
    nu = 0.33
    s.a = 2.
    s.b = 1.
    s.r = 2.

    # simply supported, motion parallel to each edge prevented, motion
    # normal to each edge free
    s.x1u = 1
    s.x2u = 1
    s.y1v = 1
    s.y2v = 1

    # compression
    s.Nxx = -1

    s.stack = [0]
    s.plyt = thickness
    s.laminaprop = (E, nu)
    s.model = 'cylshell_clpt_donnell'

    eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
    plot_shell(s, eigvecs[:, 0], vec='w', filename='example_isotropic.png')

    D = E*thickness**3/(12*(1 - nu**2))
    Ncrs = []
    for m in range(1, 80):
        for n in range(1, 40):
            alpha = m*np.pi/s.a
            beta = n*np.pi/s.b
            Ncrs.append(D*(alpha**2 + beta**2)**2/alpha**2
                        + E*thickness/s.r**2*alpha**2/(alpha**2 + beta**2)**2)
    assert np.isclose(eigvals[0], min(Ncrs), rtol=1e-5)
