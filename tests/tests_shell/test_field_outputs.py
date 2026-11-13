r"""Field outputs of a static analysis

``Shell.uvw``, ``Shell.stress`` and ``Shell.strain`` of a symmetric and
balanced laminated panel under an in-plane load distributed along ``x = 0``,
for five models, with in-plane boundary conditions that leave the membrane
state uniform: the extrema of ``Nxx``, ``exx`` and ``eyy`` against the
uniform membrane solution of classical lamination theory (Reddy, 2003), and
``w`` against zero.

Notes
-----
The laminate ``[0, -45, +45, 90]s`` is symmetric, `\boldsymbol{B} =
\boldsymbol{0}`, and balanced, `A_{16} = A_{26} = 0`. The edge ``x = a`` has
``u = 0``, the edge ``y = 0`` has ``v = 0`` and the other in-plane
displacements are free, so that the load `P/b` per unit length applied at
``x = 0`` produces the uniform membrane state of the laminate constitutive
relation `\boldsymbol{N} = \boldsymbol{A} \boldsymbol{\varepsilon}` of the
classical lamination theory [1]_:

.. math::

    N_{xx} = -\frac{P}{b}, \qquad N_{yy} = N_{xy} = 0, \qquad
    \varepsilon_{xx} = a_{11} N_{xx}, \qquad
    \varepsilon_{yy} = a_{12} N_{xx}, \qquad
    \boldsymbol{a} = \boldsymbol{A}^{-1}

with `u = \varepsilon_{xx} (x - a)` and `v = \varepsilon_{yy} y`, linear
fields that the approximation functions represent exactly, and `w = 0`.
`\boldsymbol{A}` is taken from ``Shell.ABD``. All models give the reference
within `10^{-8}` for ``m = n = 4, 7, 10``; the tolerance is `10^{-6}`. The
largest `|w|` is `5 \times 10^{-12}` (``cylshell_clpt_sanders``, radius ``r
= 1e5``), against `|u| \le 4 \times 10^{-5}`; the bound is `10^{-10}`.

References
----------
.. [1] Reddy, J. N., Mechanics of Laminated Composite Plates and Shells:
   Theory and Analysis, 2nd ed., CRC Press, 2003.
   https://doi.org/10.1201/b12409

"""
import sys
sys.path.append('../../..')

import pytest
import numpy as np

from structsolve import static
from panels.shell import Shell


@pytest.mark.slow
def test_panel_field_outputs():
    m = 7
    n = 6
    for model in ['plate_clpt_donnell',
                  'cylshell_clpt_donnell',
                  'cylshell_clpt_sanders',
                  'plate_fsdt_donnell',
                  'plate_tsdt_donnell']:
        print('Testing model %s' % model)
        s = Shell()
        s.model = model
        # u = 0 at x = a, v = 0 at y = 0, other in-plane displacements free
        s.x1u = 1
        s.y1u = 1
        s.y2u = 1
        s.x1v = 1
        s.x2v = 1
        s.y1v = 0
        s.y2v = 1

        s.a = 2.
        s.b = 1.
        s.r = 1.e5
        s.stack = [0, -45, +45, 90, 90, +45, -45, 0]
        s.plyt = 1e-3*0.125
        s.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        s.nx = m
        s.ny = n
        s.m = m
        s.n = n

        P = 1000.
        Nxx = P/s.b
        s.add_distr_load_fixed_x(0, funcx=lambda y: Nxx, funcy=None, funcz=None, cte=False)
        increments, cs = static(s.calc_kC(), s.calc_fext(), silent=True)
        _, fields = s.uvw(cs[0])
        _, fields = s.stress(cs[0])
        _, fields = s.strain(cs[0])

        # uniform membrane state, N = A epsilon
        a_inv = np.linalg.inv(s.ABD[:3, :3])
        Nxx_ref = -Nxx
        exx_ref = a_inv[0, 0]*Nxx_ref
        eyy_ref = a_inv[0, 1]*Nxx_ref
        for name, ref in [('Nxx', Nxx_ref), ('exx', exx_ref), ('eyy', eyy_ref)]:
            assert np.isclose(fields.get(name).min(), ref, rtol=1e-6)
            assert np.isclose(fields.get(name).max(), ref, rtol=1e-6)
        assert np.abs(fields.get('w')).max() < 1e-10


if __name__ == '__main__':
    test_panel_field_outputs()
