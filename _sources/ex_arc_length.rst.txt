Non-linear static analysis with the arc-length methods
======================================================

The arc-length methods of
`structsolve <https://github.com/saullocastro/structsolve>`_ trace equilibrium
paths that include limit points, where the Newton-Raphson method with load
control fails. The methods of the :class:`.Shell` object are passed directly
to :class:`structsolve.Analysis`, selecting the Riks method with
``NL_method='arc_length_riks'``, or the Crisfield method with
``NL_method='arc_length_crisfield'``.

The example below is the snap-through benchmark of Sabir and Lock (1972): a
shallow cylindrical panel with the straight edges hinged and the curved edges
free, loaded by a point force at its center. With a reference load larger
than the limit load, the analysis passes the limit point, where the load
factor decreases during the snap-through, and continues until the load factor
reaches exactly 1.0. The code is extracted from one of the ``panels`` unit
tests:

.. literalinclude:: ../../tests/tests_shell/test_nonlinear_riks.py
    :pyobject: test_riks_hinged_cylindrical_shell
