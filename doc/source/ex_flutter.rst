Panel flutter analysis
======================

The aerodynamic stiffness matrix of the linear piston theory is calculated
with :meth:`.Shell.calc_kA`, as a function of the aerodynamic parameter
``beta`` of the :class:`.Shell` object. The flutter onset is found by
increasing ``beta`` and solving the eigenvalue problem
`([K_C] + [K_G] + [K_A] + \lambda^2 [M])\{c\} = \{0\}` with
:func:`structsolve.freq`, until two natural frequencies coalesce and the
eigenvalues become complex. The code below is extracted from one of the
``panels`` unit tests, which verifies the critical ``beta`` with and without
pre-stress:

.. literalinclude:: ../../tests/tests_shell/test_aero.py
    :pyobject: test_aero
