Frequency analysis
==================

The mass matrix is calculated with :meth:`.Shell.calc_kM` and the eigenvalue
problem `([K] + \lambda^2 [M])\{c\} = \{0\}` is solved with
:func:`structsolve.freq`, where `\lambda^2 = -\omega_n^2` and `\omega_n` is
the natural frequency in rad/s. The effect of a pre-stress is included by
adding the geometric stiffness matrix to the constitutive stiffness matrix. The
code below is extracted from one of the ``panels`` unit tests:

.. literalinclude:: ../../tests/tests_shell/test_natural_frequencies.py
    :pyobject: test_panel_freq
