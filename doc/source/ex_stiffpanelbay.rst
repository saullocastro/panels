Stiffened panel bays
====================

A :class:`.StiffPanelBay` is built by setting the attributes of the bay,
adding the panels of the skin, separated along `y`, and then the blade
stiffeners at the edges between the panels, see :ref:`stiffened_panels`.
The bay below has one stiffener at the middle of its width, with a 1D or a
2D flange and with or without a base:

.. literalinclude:: ../../tests/tests_stiffpanelbay/test_stiffpanelbay_api.py
    :pyobject: _bay

Its linear buckling analysis uses the matrices of the bay as those of a
:class:`.Shell`:

.. literalinclude:: ../../tests/tests_stiffpanelbay/test_stiffpanelbay_api.py
    :pyobject: _lb

A flange, and a base, increase the buckling load of the bay, and the two
formulations of the flange agree roughly:

.. literalinclude:: ../../tests/tests_stiffpanelbay/test_stiffpanelbay_api.py
    :pyobject: test_bladestiff2d_increases_buckling_load

.. literalinclude:: ../../tests/tests_stiffpanelbay/test_stiffpanelbay_api.py
    :pyobject: test_bladestiff2d_close_to_bladestiff1d

Dividing an unstiffened skin into panels does not change its buckling load,
since the panels share the approximation functions of the bay:

.. literalinclude:: ../../tests/tests_stiffpanelbay/test_stiffpanelbay_lb.py
    :pyobject: _bay

.. literalinclude:: ../../tests/tests_stiffpanelbay/test_stiffpanelbay_lb.py
    :pyobject: test_unstiffened_bay_matches_single_shell
