Linear static analysis
======================

A :class:`.Shell` object describes a plate or a cylindrical shell through its
attributes: the geometry, the laminate, the number of terms ``m`` and ``n`` of
the approximation functions and the model, by default
``'plate_clpt_donnell'`` for plates and ``'cylshell_clpt_sanders'`` for
cylindrical shells, see :mod:`panels.models` for the other models.
The boundary conditions are controlled by the flags ``x1u``, ``x1ur``,
``x2u``, ..., ``y2wr``, where ``0`` constrains and ``1`` releases the
translation (e.g. ``x1u``) or the rotation (e.g. ``x1ur``) of each
displacement component at each edge, see
:ref:`Bardell's functions <theory_func_bardell>`. The models based on shear
deformation theories have the rotations as independent fields, with the
analogous flags ``x1phix``, ``x1phixr``, ..., ``y2phiyr``.

The loads are added with :meth:`.Shell.add_point_load`,
:meth:`.Shell.add_distr_load_fixed_x` or :meth:`.Shell.add_distr_load_fixed_y`,
and the linear system `[K_C]\{c\} = \{F_{ext}\}` is solved with
`structsolve <https://github.com/saullocastro/structsolve>`_. The displacement,
strain and stress fields are calculated from the Ritz constants ``c`` with
:meth:`.Shell.uvw`, :meth:`.Shell.strain` and :meth:`.Shell.stress`. The code
below is extracted from one of the ``panels`` unit tests:

.. literalinclude:: ../../tests/tests_shell/test_field_outputs.py
    :pyobject: test_panel_field_outputs
