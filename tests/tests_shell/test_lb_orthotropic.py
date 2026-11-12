r"""Linear buckling of a laminated cylindrical panel under axial compression

``cylshell_clpt_donnell`` and ``cylshell_clpt_sanders``, the half-cylinder
panel of Jaunky and Knight (1999) with the ``[+-70/+-70/70]s`` laminate
under uniform axial compression, boundary conditions 1 (their Fig. 3), with
12 x 12 terms: the first load multiplier against values of the code with
these terms, and Donnell's load above Sanders-Koiter's, as in their Table 2
(`R/t = 600`); the first mode is plotted.

Notes
-----
The panel of [1]_ (Section 3.2, Table 2, Fig. 3) has length ``L = 60 in``,
radius ``R = 6 in``, an arc of `\pi` radians (``b = pi R``) and thickness
``t = 0.01 in``; the laminate ``[+-70/+-70/70]s`` has ten plies of equal
thickness with ``E11 = 13.75 Msi``, ``E22 = 1.03 Msi``, ``G12 = G13 = G23 =
0.42 Msi`` and ``nu12 = 0.25``, the angles measured from the axis. Boundary
conditions 1: ``u = v = w = 0`` at ``x = 0``, ``v = w = 0`` at ``x = L``,
``u = w = 0`` on the straight edges, the rotations free; in panels ``x2u =
1`` and ``y1v = y2v = 1``. The prebuckling state is a uniform ``Nxx``, as in
[1]_. Units: inches and psi, the buckling load in lb/in.

Table 2 of [1]_ gives 31.91 lb/in with Donnell's and 30.79 lb/in with
Sanders-Koiter's theory. With ``m = n = 30``, the largest number of
Bardell terms of panels, the models
give 31.73 (``cylshell_clpt_donnell``, `-0.56 \%`) and 31.25
(``cylshell_clpt_sanders``, `+1.5 \%`), after 32.02 and 31.35 with ``m = n =
18``, and 33.88 and 32.59 with ``m = n = 12``.

Only this row of Tables 2 and 3 of [1]_ is used. The other rows are not
reproduced within 2 % with the same model and boundary conditions (``m = n
= 30``): ``R = 6 in``, ``t = 0.06 in``: Donnell +7.7 %, Sanders-Koiter
+20 % (isotropic: +2.3 %, +5.4 %); ``R = 24 in``, ``t = 0.24 in``: -3.1 %,
-0.5 % (isotropic: +1.0 %, +0.6 %); ``R = 48 in``, ``t = 0.24 in``: -7.9 %
both. For isotropic panels the Donnell model of panels reproduces the exact
solution of Batdorf [2]_ to `10^{-12}` (``test_lb_isotropic.py``), so the
deviations come either from the solutions of [1]_ or from the reading of
their boundary conditions, and these rows are not used as reference values.

References
----------
.. [1] Jaunky, N., Knight, N. F., Jr., An assessment of shell theories for
   buckling of circular cylindrical laminated composite panels loaded in
   axial compression. International Journal of Solids and Structures 36
   (1999) 3799-3820. https://doi.org/10.1016/S0020-7683(98)00177-2
.. [2] Batdorf, S. B., A Simplified Method of Elastic-Stability Analysis
   for Thin Cylindrical Shells, NACA Report 874, National Advisory
   Committee for Aeronautics, 1947.
   https://ntrs.nasa.gov/citations/19930091945

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from structsolve import lb
import matplotlib.pyplot as plt

from panels.shell import Shell
from panels.plot_shell import plot_shell


@pytest.mark.slow
def test_lb_orthotropic():
    # m = n = 12: values of the code with these terms, above those of
    # m = n = 30 given in the Notes, Donnell above Sanders as in Table 2
    loads = {}
    for model, ref in [('cylshell_clpt_donnell', 33.8823),
                       ('cylshell_clpt_sanders', 32.5944)]:
        s = Shell()
        s.m = 12
        s.n = 12
        s.stack = [70, -70, 70, -70, 70, 70, -70, 70, -70, 70]
        thickness = 0.01
        s.plyt = thickness/len(s.stack)
        s.laminaprop = (13.75e6, 1.03e6, 0.25, 0.42e6, 0.42e6, 0.42e6)
        s.model = model
        R = 6.
        s.a = 60.
        s.b = np.pi*R
        s.r = R
        # u free at x = a, v free on the straight edges
        s.x2u = 1
        s.y1v = 1
        s.y2v = 1

        # compression
        s.Nxx = -1

        eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
        loads[model] = eigvals[0]
        assert np.isclose(eigvals[0], ref, rtol=1e-4)
    assert loads['cylshell_clpt_donnell'] > loads['cylshell_clpt_sanders']
    plot_shell(s, eigvecs[:, 0], vec='w', save=False)
    plt.close("all")
