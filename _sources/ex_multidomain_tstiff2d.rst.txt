Panels with a T-stiffener modeled as multi-domain assemblies
============================================================

A flat or curved panel with one T-stiffener is modeled with 2D domains for the
skin, the base and the flange of the stiffener, following Castro and Donadon
(2017) [castro2017Multidomain]_. A debonding defect of length ``defect_a`` is
included at the middle of the panel, where the base is not connected to the
skin.

Linear buckling, with constant stress resultants or with the pre-buckling
stress state from a static analysis, using
:func:`.tstiff2d_1stiff_compression`:

.. literalinclude:: ../../tests/multidomain/test_tstiff2d_assembly.py
    :pyobject: test_tstiff2d_1stiff_compression

Frequency analysis, using :func:`.tstiff2d_1stiff_freq`:

.. literalinclude:: ../../tests/multidomain/test_tstiff2d_assembly.py
    :pyobject: test_tstiff2d_1stiff_freq

Flutter analysis with the piston theory, using
:func:`.tstiff2d_1stiff_flutter`:

.. literalinclude:: ../../tests/multidomain/test_tstiff2d_assembly.py
    :pyobject: test_tstiff2d_1stiff_flutter
