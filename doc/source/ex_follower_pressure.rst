.. _ex_follower_pressure:

Follower (hydrostatic) pressure
===============================

A pressure added with ``follower=True`` in :meth:`.Shell.add_pressure_load`
stays normal to the deformed mid-surface and acts on its deformed area. In
the undeformed state it equals the dead load returned by
:meth:`.Shell.calc_fext`. The configuration-dependent part of its force
vector enters :meth:`.Shell.calc_fint`, and its load stiffness
:meth:`.Shell.calc_kCfollower`, which is unsymmetric in general, enters
:meth:`.Shell.calc_kT`, both at the load factor ``inc``. The theory, the
truncation of the area vector and the conditions for a symmetric load
stiffness are derived in :ref:`follower_pressure`.

Linear buckling of rings and long cylinders
-------------------------------------------

The linear buckling problem of a hydrostatic pressure is
``lb(kC0, kG + kCfollower)``, where :func:`structsolve.lb` selects the
solvers of unsymmetric matrices when needed. A long cylinder buckles as a
ring, modelled with a quarter of the circumference and symmetry planes:

.. literalinclude:: ../../tests/tests_shell/test_follower_pressure_validation.py
    :pyobject: ring_shell

.. literalinclude:: ../../tests/tests_shell/test_follower_pressure_validation.py
    :pyobject: ring_buckling_pressure

With the Sanders kinematics the critical pressure of the mode ``n = 2`` is
``3 D/r**3`` for the follower pressure and ``4 D/r**3`` for the dead load:

.. literalinclude:: ../../tests/tests_shell/test_follower_pressure_validation.py
    :pyobject: test_isotropic_ring

The long sandwich cylinders of Han, Kardomateas and Simitses
[han2004SandwichCylinder]_ are compared with their shell formulas and, by
running the module, with their elasticity solution, with the pressure on
the outer face given by ``zp = h/2``:

.. literalinclude:: ../../tests/tests_shell/test_follower_pressure_validation.py
    :pyobject: test_han2004_shell_formulas

Non-linear analyses
-------------------

The callables of the :class:`.Shell` are passed directly to
:class:`structsolve.Analysis`, which passes its load factor to the callables
that accept ``inc``, and uses ``calc_fext(inc=1., c=c)`` as the load vector
of the current configuration in the arc-length methods:

.. literalinclude:: ../../tests/tests_shell/test_follower_pressure.py
    :pyobject: test_newton_raphson_converges_quadratically

Linear static analysis and kinetic criterion
--------------------------------------------

A geometrically linear structure under a follower load solves the
unsymmetric system ``(k0 + kCfollower) c = fext``, which
:class:`structsolve.Analysis` does with ``static(NLgeom=False)``, since
:meth:`.Shell.calc_fext` accepts the configuration ``c``:

.. literalinclude:: ../../tests/tests_shell/test_follower_pressure.py
    :pyobject: test_linear_static_with_follower_load

Non-conservative systems may lose stability by flutter, which only the
kinetic criterion detects: :func:`structsolve.freq` solves the unsymmetric
eigenvalue problem of the tangent stiffness matrix with the load stiffness,
whose eigenvalues become complex at a flutter load. The cantilever cylinder
of [schweizerhof1984Pressure]_ diverges instead, its lowest frequency
vanishing at the critical load of the static criterion:

.. literalinclude:: ../../tests/tests_shell/test_follower_pressure_validation.py
    :pyobject: test_kinetic_criterion_cantilever

Validation notebooks
--------------------

The comparisons with the literature are reproduced, per reference, by the
notebooks of the folder ``notebooks``, which use the models and the data of
``tests/tests_shell/test_follower_pressure_validation.py``:

- ``han2004_sandwich_rings.ipynb``: long sandwich cylinders, Tables 2-4 of
  [han2004SandwichCylinder]_;
- ``kardomateas2003_sandwich_rings.ipynb``: Tables 2-4 of
  [kardomateas2003Sandwich]_;
- ``kardomateas1993_thick_rings.ipynb``: thick orthotropic and isotropic
  rings, Tables 1 and 2 and Figs. 2 and 5 of [kardomateas1993Orthotropic]_;
- ``schweizerhof1984_pressure_loads.ipynb``: Tables 2-4 and Fig. 13 of
  [schweizerhof1984Pressure]_, rings and cantilever cylinders with an
  unsymmetric load stiffness;
- ``nasa_sp8007_external_pressure.ipynb``: Batdorf's equations for lateral
  and hydrostatic pressure of [nasa2020SP8007]_.
