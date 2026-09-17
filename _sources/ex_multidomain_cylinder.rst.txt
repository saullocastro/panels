Cylinders modeled as multi-domain assemblies
============================================

A :class:`.MultiDomain` assembly connects :class:`.Shell` domains with penalty
stiffnesses, see :mod:`panels.multidomain.connections`. The function
:func:`.create_cylinder` creates a cylinder with panels along its
circumference, connecting the last panel to the first one to close the
cylinder.

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
