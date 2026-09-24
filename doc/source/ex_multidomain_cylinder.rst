Cylinders modeled as multi-domain assemblies
============================================

A :class:`.MultiDomain` assembly connects :class:`.Shell` domains, see
:mod:`panels.multidomain.connections`. The function
:func:`.create_cylinder` creates a cylinder with panels along its
circumference, connecting the last panel to the first one to close the
cylinder.

The connections are imposed exactly with the default
``conn_method='null-space'``, see :ref:`null_space`, or with penalty
stiffnesses with ``conn_method='penalty'``. The closed cylinder below gives
-47113.0 N/m with the null-space method and -47056.0 N/m with the default
penalty constants, see ``tests/multidomain/test_null_space.py`` and
``tests/multidomain/test_cylinder.py``.

Linear buckling with a constant axial compression ``Nxx`` in each panel,
calculated with :func:`.cylinder_compression_lb_Nxx_cte`:

.. literalinclude:: ../../tests/multidomain/test_cylinder.py
    :pyobject: test_cylinder_compression_lb_Nxx_cte

Linear buckling with the pre-buckling stress state calculated from a static
analysis, with :func:`.cylinder_compression_lb_Nxx_from_static`:

.. literalinclude:: ../../tests/multidomain/test_cylinder.py
    :pyobject: test_cylinder_compression_lb_Nxx_from_static

Blade stiffeners can be added along the length of the cylinder, modeled as
additional domains connected to the skin, see
:func:`.create_cylinder_blade_stiffened`:

.. literalinclude:: ../../tests/multidomain/test_cylinder_blade_stiffened.py
    :pyobject: test_cylinder_blade_stiffened_compression_lb_Nxx_from_static
