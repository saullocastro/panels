.. _aeroelasticity:

Panel flutter with the piston theory
====================================

The supersonic flutter of plates, cylindrical panels and stiffened panels
is analysed with the aerodynamic load of the linear piston theory, a
pressure on `w` proportional to the slope of `w` along the flow, with the
parameter `\beta`, plus, for a flow along the axis of a cylinder, a
curvature correction proportional to `w`, with the parameter `\gamma`. The
flow is along `x` or `y`, the attribute ``flow``.


Aerodynamic matrices
--------------------

The aerodynamic stiffness matrix `[K_A]` and the aerodynamic damping
matrix are calculated by:

- :meth:`.Shell.calc_kA` and :meth:`.Shell.calc_cA`, for one domain;
- :meth:`.StiffPanelBay.calc_kA` and :meth:`.StiffPanelBay.calc_cA`, the
  sum over the panels of the skin of a bay, the stiffeners not contributing;
- in a :class:`.MultiDomain` assembly, by adding the matrices of the
  domains exposed to the flow, ``calc_kA(size=size, row0=p.row_start,
  col0=p.col_start)``, as :func:`.tstiff2d_1stiff_flutter` does for the
  skin.

The parameters ``beta`` and ``gamma`` are given directly, or calculated
from the Mach number ``Mach``, the air density ``rho_air``, the air speed
and the radius, see :meth:`.Shell.calc_kA` and
:meth:`.StiffPanelBay.calc_kA`; the piston theory requires ``Mach`` `\ge
1`. The matrix `[K_A]` is unsymmetric and is never symmetrized.


Flutter analysis
----------------

`[K_A]` is added to the stiffness matrix of a frequency analysis,

.. math::

    \left([K_C] + [K_G] + [K_A] + \lambda^2 [M]\right) \{\bar{c}\} = \{0\}

solved with :func:`structsolve.freq`, which handles unsymmetric matrices,
where `[K_G]` is the geometric stiffness of an optional prestress and
`\lambda^2 = -\omega^2`. Increasing `\beta`, two natural frequencies
approach each other and coalesce at the flutter onset, where the
eigenvalues become complex. The sign conventions of `[K_A]` and the
damping are given in :meth:`.Shell.calc_kA` and :meth:`.Shell.calc_cA`.

The examples are :doc:`ex_flutter`, for one domain with and without
prestress, and the panel with a T-stiffener of
:doc:`ex_multidomain_tstiff2d`. The aerodynamic matrix of a
:class:`.StiffPanelBay` is the sum of those of its panels, and does not
change when the skin is divided into panels:

.. literalinclude:: ../../tests/tests_stiffpanelbay/test_stiffpanelbay_api.py
    :pyobject: test_calc_kA_matches_single_shell
