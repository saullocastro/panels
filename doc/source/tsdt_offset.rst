.. _tsdt_offset:

Third-order shear deformation theory with a laminate offset
===========================================================

This page documents the displacement field of the models based on Reddy's
third-order shear deformation theory (TSDT), ``'plate_tsdt_donnell'``,
``'cylshell_tsdt_donnell'`` and ``'cylshell_tsdt_sanders'``, when the
laminate is offset from the reference surface, ``Shell.offset``, and how
panels with different offsets are connected, see :mod:`panels.models` and
:mod:`panels.multidomain.connections.kCsdt`.

The coordinate `z` is measured from the reference surface, `z = 0`, and the
laminate of thickness `h` occupies `d - h/2 \le z \le d + h/2`, with the
offset `d`, the convention of :func:`composites.laminated_plate`, whose
stiffness matrices are the integrals through the thickness from `z = d -
h/2`.

.. _tsdt_offset_field:

Displacement field
------------------

The in-plane displacements are sought as

.. math::
    :label: tsdt-offset-ansatz

    u(z) = u + z \phi_x + p(z) \left(\phi_x + w_{,x}\right) \qquad
    v(z) = v + z \Phi_y + p(z) \left(\phi_y + w_{,y}\right)

with `p(z)` a cubic polynomial, `\Phi_y = \phi_y + v/r` with the
Sanders-Koiter kinematics and `\Phi_y = \phi_y` otherwise. The conditions
on `p(z)` are:

- `p(0) = 0`, such that `u` and `v` are the displacements of the reference
  surface, as for the classical (CLPT) and the first-order (FSDT)
  theories;
- `p'(d \pm h/2) = -1`, such that the transverse shear strain
  `\gamma_{xz} = u_{,z} + w_{,x} = (\phi_x + w_{,x}) (1 + p'(z))` vanishes
  at both faces.

These three conditions leave one free parameter in the cubic `p(z)`. It
does not change the family of displacement fields: `\gamma_{xz}` is a
quadratic polynomial in `z` vanishing at both faces, therefore
`\gamma_{xz} = \hat{\gamma}_{xz} (1 - 4 (z - d)^2/h^2)` for any choice, and
the integration of `u_{,z} = \gamma_{xz} - w_{,x}` with `u(0) = u` gives

.. math::
    :label: tsdt-offset-family

    u(z) = u - z w_{,x} + \hat{\gamma}_{xz} \left[ z - \frac{4}{3 h^2}
    \left( (z - d)^3 + d^3 \right) \right]

which depends only on `u`, `w` and the amplitude `\hat{\gamma}_{xz}` of
the transverse shear strain. The free parameter only selects which
rotation of the normal is the unknown `\phi_x = u_{,z}(z_\phi) = -w_{,x} +
\hat{\gamma}_{xz} (1 - 4 (z_\phi - d)^2/h^2)`, i.e. the coordinate
`z_\phi` where it is measured:

- at the reference surface, `z_\phi = 0`: `\hat{\gamma}_{xz} = (\phi_x +
  w_{,x})/(1 - 4 d^2/h^2)`, singular for `d = \pm h/2`. When the reference
  surface is a face of the laminate, the transverse shear vanishes there
  and `u_{,z}(0) = -w_{,x}`, which cannot be an independent variable; the
  determinant of the relation, proportional to `(2 d - h)(2 d + h)`,
  vanishes, and the formulation is ill-conditioned when the reference
  surface approaches a face, the common case of a skin and a stiffener
  connected at their common face;
- at the mid-surface, `z_\phi = d`: `\hat{\gamma}_{xz} = \phi_x +
  w_{,x}`, regular for any offset;
- the average through the thickness, `\bar\phi_x = \frac{1}{h}
  \int u_{,z} dz = -w_{,x} + \frac{2}{3} \hat{\gamma}_{xz}`, also regular,
  but not Reddy's variable for `d = 0`.

The rotations at the mid-surface are adopted, which gives

.. math::
    :label: tsdt-offset-p

    p(z) = -c_1 \left[ (z - d)^3 + d^3 \right], \qquad c_1 = \frac{4}{3
    h^2}

Reddy's field for `d = 0`. The conditions are solved with SymPy.

.. _tsdt_offset_matrices:

Strains, stiffness and mass matrices
------------------------------------

With `\zeta = z - d` measured from the mid-surface, the in-plane strains
and the transverse shear strains are

.. math::
    :label: tsdt-offset-strains

    \varepsilon = \left( \varepsilon^{(0)} + d \varepsilon^{(1)} + d^3
    \varepsilon^{(3)} \right) + \zeta \varepsilon^{(1)} + \zeta^3
    \varepsilon^{(3)} \qquad
    \gamma = \gamma^{(0)} + \zeta^2 \gamma^{(2)}

with the generalized strains of the TSDT, see
:mod:`panels.models.fsdt_tsdt_field`. The strain energy is that of the
laminate about its mid-surface, whose matrices `A, B, D, E, F, H` are the
integrals of the stiffness of the plies multiplied by `1, \zeta, \zeta^2,
\zeta^3, \zeta^4, \zeta^6`, computed by :mod:`composites` with a zero
offset, :meth:`.Shell._get_lam_mid`, for the generalized strains `T e`,
see :meth:`.Shell._get_lam_ABD`. In `z` the products of the strains would
require the integrals of `z^5`, which do not appear in `\zeta`. Likewise,
the mass matrix uses the moments of the density about the mid-surface,
`J_n = \int \rho \zeta^n d\zeta`, `n = 0, 1, 2, 3, 4, 6`, see
:meth:`.Shell.calc_kM`.

.. _tsdt_offset_connections:

Connections at the reference surface
------------------------------------

Panels with different offsets share the reference surface, where the
connections of :mod:`panels.multidomain.connections.kCsdt`,
:mod:`panels.multidomain.connections.kCBFycte`,
:mod:`panels.multidomain.connections.kCBFxcte` and of the
:ref:`null-space method <null_space>` tie `u, v, w` and the rotations of
the normal at `z = 0`:

.. math::
    :label: tsdt-offset-theta

    \theta_x = u_{,z}\big|_{z=0} = \phi_x - \frac{4 d^2}{h^2} \left(\phi_x +
    w_{,x}\right) \qquad
    \theta_y = v_{,z}\big|_{z=0} = \Phi_y - \frac{4 d^2}{h^2} \left(\phi_y +
    w_{,y}\right)

computed with the offset `d` and the thickness `h` of each panel, see
:func:`.kCsdt._rotation_x` and :func:`.kCsdt._rotation_y`. For `d = \pm
h/2` they become `\theta_x = -w_{,x}` and `\theta_y = \Phi_y - \phi_y -
w_{,y}`. The rotations `\phi_x, \phi_y` belong to the mid-surfaces, at the
coordinates `z = d` of each panel, and are not continuous across the
connection of panels with different offsets. For the FSDT the rotation is
constant through the thickness, `\theta = \phi`, and for the CLPT it is
`-w_{,x}` and `-w_{,y}` for any offset: their connections are unchanged.

.. _tsdt_offset_prescribed_rotations:

Prescribed rotations
--------------------

The rotations ``phix`` and ``phiy`` prescribed with
:meth:`.Shell.add_point_pd`, :meth:`.Shell.add_distr_pd_fixed_x` and
:meth:`.Shell.add_distr_pd_fixed_y`, imposed with ``pd_method='null-space'``,
see :ref:`null_space_pd`, are the same rotations at the reference surface,
`\theta_x` and `\theta_y` of Eq. :eq:`tsdt-offset-theta`, computed with the
offset and the thickness of the panel, see
:func:`.nullspace.pd_constraint_matrix`. With an offset, the rotations at
the mid-surface `\phi_x` and `\phi_y` differ from the prescribed values by
the terms of the transverse shear strains, and for `d = \pm h/2` the
prescribed rotations are `-w_{,x}` and `\Phi_y - \phi_y - w_{,y}`. For the
FSDT, and for the TSDT with `d = 0`, the prescribed rotations are `\phi_x`
and `\Phi_y`, as before. The penalty method,
``pd_method='penalty'``, does not prescribe rotations.

.. literalinclude:: ../../tests/multidomain/test_tsdt_prescribed_rotations.py
    :pyobject: test_prescribed_rotation_with_offset

.. _tsdt_offset_clamped_edges:

Clamped edges
-------------

A clamped edge keeps the section planar and unrotated, `u(z) = 0` for every
`z`. Eq. :eq:`tsdt-offset-family` is a cubic in `z` with the three
constants `u`, `w_{,x}` and `\hat{\gamma}_{xz} = \phi_x + w_{,x}`, all
zero: `u = \phi_x = w_{,x} = 0`, the flags ``x1u = x1phix = x1wr = 0`` on
`x = 0`, and then `\theta_x = 0` for any offset. The rotation at the
reference surface alone, Eq. :eq:`tsdt-offset-theta`, does not clamp the
edge, see :ref:`boundary_conditions`.

.. _tsdt_offset_example:

Example
-------

Two plates with the offsets `d_1 = 0` and `d_2`, connected along `x =
x_s` with the null-space method and loaded by a pressure, have continuous
rotations at the reference surface, whereas the rotations at the
mid-surfaces jump across the connection:

.. literalinclude:: ../../tests/multidomain/test_tsdt_offset_connections.py
    :pyobject: test_reference_rotation_continuous_with_different_offsets

and two plates with the same offset reproduce the single plate:

.. literalinclude:: ../../tests/multidomain/test_tsdt_offset_connections.py
    :pyobject: test_same_offset_split_plate_equals_single_plate

For a free homogeneous plate, the choice of the reference surface does not
change the structure, and the natural frequencies do not depend on the
offset:

.. literalinclude:: ../../tests/tests_shell/test_tsdt_offset.py
    :pyobject: test_free_plate_frequencies_do_not_depend_on_the_reference_surface
