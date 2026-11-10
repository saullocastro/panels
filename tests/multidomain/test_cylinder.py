r"""Linear buckling of a cylinder made of multiple domains

The assembly functions :func:`.cylinder_compression_lb_Nxx_cte` and
:func:`.cylinder_compression_lb_Nxx_from_static` (slow) build an isotropic
cylinder of 5 domains connected along the circumference, under axial
compression, with the default Sanders kinematics and penalty connections.
The tests verify the critical load ``-Nxx*eigvals[0]`` against the classical
buckling load of Batdorf (1947), eq. (1), within 1.5 %, and against a
regression value of the same model. ``test_create_cylinder`` checks that
the connections returned by :func:`.create_cylinder` are those stored in
the assembly, ``md.conn``, and close the cylinder.

Notes
-----
The cylinder has length ``height = 0.2``, radius ``r = 0.25`` and thickness
``t = 2e-3`` (aluminium, ``E = 70e9``, ``nu = 0.3``; SI units), curvature
parameter `Z = L^2 \sqrt{1 - \nu^2}/(r t) = 76`. The classical buckling
load of a moderately long cylinder, eq. (1) of [1]_ (p. 5), is

.. math::

    N_{cl} = \sigma_x t = \frac{E t^2}{r \sqrt{3 (1 - \nu^2)}}

here `N_{cl} = 6.7786 \times 10^5`; the minimum of eq. (B9) of [1]_ (p. 18)
over integer `m` and `n` (`m = 3` axial half-waves, `n = 10`
circumferential waves) is `6.7787 \times 10^5`. Runs of this cylinder on
:func:`.create_cylinder` with ``m = 14``, ``n = 12`` and the edges of [1]_
(`u` free at both ends, ``x1u = 1``) separate the differences: Donnell
kinematics with ``conn_method='null-space'``, `+0.003 \%`; the same with
the end `x = 0` of :func:`.create_cylinder` (`u = 0`), `+0.007 \%`; Donnell
with ``'penalty'`` connections, `-0.69 \%`; Sanders with ``'penalty'``,
`-0.82 \%`. The pre-buckling state of
:func:`.cylinder_compression_lb_Nxx_from_static`, with the radial
displacement restrained at the ends, gives `-1.14 \%`. With ``m = 18`` and
``m = 22`` the critical loads change by less than `10^{-5}`. The tolerance
of 1.5 % covers the penalty connections and the Sanders kinematics; the
regression values, from the same runs (``m = 14``, ``n = 12``), are
checked to `10^{-5}`.

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

from panels.multidomain import (create_cylinder,
        cylinder_compression_lb_Nxx_cte,
        cylinder_compression_lb_Nxx_from_static)

E = 70e9
nu = 0.3
thickness = 2e-3
height = 0.2
r = 0.25
# classical buckling load of the cylinder (Notes)
Ncl = E*thickness**2/(r*np.sqrt(3*(1 - nu**2)))


def test_create_cylinder():
    md, conns = create_cylinder(height=height, r=r, stack=[0],
            plyt=thickness, laminaprop=(E, nu), npanels=5, m=4, n=4)
    assert md.conn is conns
    assert md.conn_method == 'null-space'
    assert [c['func'] for c in conns] == ['SSycte']*5
    # consecutive panels, the last joined to the first
    pairs = [(md.panels.index(c['p1']), md.panels.index(c['p2']))
             for c in conns]
    assert pairs == [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]


def test_cylinder_compression_lb_Nxx_cte():
    print('Testing assembly function: cylinder_compression_lb_Nxx_cte')
    npanels = 5
    Nxxs = [-100.]*npanels
    assy, eigvals, eigvecs = cylinder_compression_lb_Nxx_cte(
        height=height,
        r=r,
        plyt=thickness,
        stack=[0],
        laminaprop=(E, nu),
        npanels=npanels,
        Nxxs=Nxxs,
        m=14, n=12,
        num_eigvalues=10, conn_method='penalty')

    assy.plot(eigvecs[:, 0], 'skin', filename='tmp_cylinder_compression_lb_Nxx_cte.png')

    Ncr = -Nxxs[0]*eigvals[0]
    assert np.isclose(Ncr, Ncl, rtol=0.015)
    # regression value
    assert np.isclose(Ncr, 672279., rtol=1e-5)


@pytest.mark.slow
def test_cylinder_compression_lb_Nxx_from_static():
    print('Testing assembly function: cylinder_compression_lb_Nxx_from_static')
    npanels = 5
    Nxxs = [-100.]*npanels
    assy, c, eigvals, eigvecs = cylinder_compression_lb_Nxx_from_static(
        height=height,
        r=r,
        plyt=thickness,
        stack=[0],
        laminaprop=(E, nu),
        npanels=npanels,
        Nxxs=Nxxs,
        m=14, n=12,
        num_eigvalues=10, conn_method='penalty')

    assy.plot(c, 'skin', filename='tmp_cylinder_compression_lb_Nxx_from_static_c.png')
    assy.plot(eigvecs[:, 0], 'skin', filename='tmp_cylinder_compression_lb_Nxx_from_static_eigvec.png')

    Ncr = -Nxxs[0]*eigvals[0]
    assert np.isclose(Ncr, Ncl, rtol=0.015)
    # regression value
    assert np.isclose(Ncr, 670099., rtol=1e-5)
