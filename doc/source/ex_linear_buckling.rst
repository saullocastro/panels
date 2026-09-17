Linear buckling analysis
========================

The linear buckling eigenvalue problem
`([K_C] + \lambda [K_G])\{c\} = \{0\}` is solved with
:func:`structsolve.lb`, returning the load multipliers `\lambda`.

Constant pre-buckling stress state
----------------------------------

The pre-buckling stress state can be defined by the constant stress resultants
``Nxx``, ``Nyy`` and ``Nxy`` of the :class:`.Shell` object, used by
:meth:`.Shell.calc_kG` to calculate the geometric stiffness matrix
analytically. The example below verifies the critical loads of laminated
plates and shells, simply supported on all edges or with one free edge, under
axial or transverse compression:

.. literalinclude:: ../../tests/tests_shell/test_lb.py
    :pyobject: test_shell_lb

Isotropic materials are defined with ``laminaprop = (E, nu)``, and combined
load cases are defined by more than one stress resultant, e.g. compression
and shear:

.. literalinclude:: ../../tests/tests_shell/test_lb_isotropic.py
    :pyobject: test_lb_isotropic

Pre-buckling stress state from a static analysis
------------------------------------------------

The geometric stiffness matrix can also be calculated from the Ritz constants
of a linear static solution, passed as ``c`` to :meth:`.Shell.calc_kG`, which
then integrates the pre-buckling stress field numerically:

.. literalinclude:: ../../tests/tests_shell/test_lb_num.py
    :pyobject: test_panel_fkG_num
