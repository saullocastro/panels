.. _ex_face_clamp:

Clamped and face-clamped edges
==============================

A clamped edge of the third-order shear deformation theory (TSDT) removes
the slope of `w` normal to the edge together with the rotations, ``x1wr =
0`` and ``x1phix = x1phiy = 0`` on `x = 0`, such that the cross-section
stays planar and unrotated; an edge clamped by its faces, `u = v = 0` at
`z = d \mp h/2` and `w = 0`, leaves the section free to warp and is
prescribed with the keyword ``z`` of :meth:`.Shell.add_distr_pd_fixed_x`,
see :ref:`boundary_conditions`. The square plates below, `h/a = 0.1`, are
clamped at `x = 0, a`, free at `y = 0, b`, and loaded by a uniform
pressure:

.. literalinclude:: ../../tests/multidomain/test_face_clamp.py
    :pyobject: plate

.. literalinclude:: ../../tests/multidomain/test_face_clamp.py
    :pyobject: clamp_by_flags

.. literalinclude:: ../../tests/multidomain/test_face_clamp.py
    :pyobject: face_clamp

For the CLPT and the FSDT the face clamp is the clamp of the flags:

.. literalinclude:: ../../tests/multidomain/test_face_clamp.py
    :pyobject: test_face_clamp_equals_clamp_for_clpt_and_fsdt

For the TSDT, `u(z) = 0` prescribed at a third coordinate gives the planar
clamp of the flags, which the former clamp with the slope of `w` free does
not:

.. literalinclude:: ../../tests/multidomain/test_face_clamp.py
    :pyobject: test_planar_clamp_is_u_zero_at_three_coordinates

The face clamp lies between the planar clamp and the simple support, here
with the central deflections `w D/(q a^4)` = 0.002901 (planar clamp),
0.002953 (face clamp) and 0.013459 (simple support): the displacements of
both faces vanish along the edges, whereas the section warps, with
`\phi_x = w_{,x}/2`:

.. literalinclude:: ../../tests/multidomain/test_face_clamp.py
    :pyobject: test_face_clamp_tsdt_thick_plate

Both clamps are defined by the physical section, and the results do not
depend on the reference surface of the laminate:

.. literalinclude:: ../../tests/multidomain/test_face_clamp.py
    :pyobject: test_face_clamp_does_not_depend_on_the_reference_surface

With the penalty method, the kernels of the prescribed displacements are
given the same ``z``:

.. literalinclude:: ../../tests/multidomain/test_face_clamp.py
    :pyobject: test_face_clamp_penalty_and_null_space_agree
