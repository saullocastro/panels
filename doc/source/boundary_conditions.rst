.. _boundary_conditions:

Boundary conditions
===================

The boundary conditions of a domain are imposed in three ways, from the most
to the least common:

- on whole edges, exactly, with the flags of the approximation functions,
  attributes of :class:`.Shell`, see :ref:`theory_func_bardell`;
- at points or along lines, as prescribed displacements, with the
  null-space or the penalty method of a :class:`.MultiDomain`, see
  :ref:`null_space_pd`, or as penalty springs at points of a single shell,
  :meth:`.Shell.calc_stiffness_point_constraint`;
- between domains, with the connections of a :class:`.MultiDomain`, see
  :ref:`multidomain_overview`.


Flags of the edges
------------------

A flag ``0`` removes the value (e.g. ``x1w``) or the slope normal to the
edge (e.g. ``x1wr``, the derivative with respect to `x` at `x = 0`) of a
field on the whole edge, and a flag ``1`` leaves it free, the natural
boundary condition following from the energy. The edges are `x = 0`
(``x1``), `x = a` (``x2``), `y = 0` (``y1``) and `y = b` (``y2``). For the
models based on the classical laminated plate theory (CLPT), on the edge `x
= 0`, with `v` the displacement tangential to the edge and `u` the normal
one:

========================================= ==========================================
condition                                  flags equal to ``0``, the others ``1``
========================================= ==========================================
simply supported, `u` and `v` restrained    ``x1u``, ``x1v``, ``x1w`` (default)
simply supported, `u` free                  ``x1v``, ``x1w``
clamped                                     ``x1u``, ``x1v``, ``x1w``, ``x1wr``
free                                        none
symmetry plane                              ``x1u``, ``x1vr``, ``x1wr``
========================================= ==========================================

and likewise on the other edges, with the roles of `u` and `v` exchanged on
the edges `y = const`. With the default flags, a :class:`.Shell` is simply
supported on its four edges with the in-plane displacements restrained.

The models based on shear deformation theories, FSDT and TSDT, have the
rotations `\phi_x` and `\phi_y` as independent fields, with the flags
``x1phix``, ``x1phixr``, ..., ``y2phiyr``. By default only the rotation
tangential to each edge is removed, ``x1phiy = x2phiy = 0`` and ``y1phix =
y2phix = 0``, the hard simply supported condition. A clamped edge keeps
the cross-section planar and unrotated, `u(z) = v(z) = 0` through the
thickness, which removes on the edge `x = 0`:

========== ====================================================================
model      flags of the clamped edge `x = 0` equal to ``0``
========== ====================================================================
CLPT       ``x1u``, ``x1v``, ``x1w``, ``x1wr``
FSDT       ``x1u``, ``x1v``, ``x1w``, ``x1phix``, ``x1phiy``
TSDT       ``x1u``, ``x1v``, ``x1w``, ``x1phix``, ``x1phiy``, ``x1wr``
========== ====================================================================

The rotation of the FSDT is constant through the thickness, and its slope
of `w` stays free. The in-plane displacement of the TSDT, `u(z) = u + z
\phi_x + p(z) (\phi_x + w_{,x})` with the cubic `p(z)` of Eq.
:eq:`tsdt-offset-p`, also contains the slope `w_{,x}`, which must be
removed for `u(z)` to vanish at every `z`; the rotation at the reference
surface, `\theta_x` of Eq. :eq:`tsdt-offset-theta`, is then zero for any
offset. The meaning of the default flags is described in :class:`.Shell`.

The three supports used to compare the theories, an edge of constant `x`
being simply supported with the normal displacement free (S), clamped (C)
or free (F), therefore remove, with all the other flags of the edge equal
to ``1``:

========== ================================ ==================================================== =================
model      S                                C                                                    F
========== ================================ ==================================================== =================
CLPT       ``x1v``, ``x1w``                 ``x1u``, ``x1v``, ``x1w``, ``x1wr``                  none
FSDT       ``x1v``, ``x1w``, ``x1phiy``     ``x1u``, ``x1v``, ``x1w``, ``x1phix``, ``x1phiy``    none
TSDT       ``x1v``, ``x1w``, ``x1phiy``     ``x1u``, ``x1v``, ``x1w``, ``x1phix``, ``x1phiy``,   none
                                            ``x1wr``
========== ================================ ==================================================== =================

and, on an edge of constant `y`, the same with `u` and `\phi_x` the
tangential displacement and rotation. With them the shear deformation
theories recover the classical plate theory for thin plates: the natural
frequency parameters `\omega a^2 \sqrt{\rho h/D}`, `D = E h^3/(12 (1 -
\nu^2))`, of isotropic square plates with `h/a = 0.001` and the edges SSSS,
SCSC, SCSS, SCSF, SSSF and SFSF agree for the three models with the
classical solutions tabulated by Leissa [leissa1973Plates]_ to their five
significant digits.
For these thin plates the slope `w_{,x}` of a TSDT clamped edge changes the
frequencies by less than `1.2 \times 10^{-5}`, e.g. 28.95027 with the slope
free and 28.95059 with ``x1wr = 0`` for the plate ``'SCSC'``, against 28.951
of Leissa. For thick plates the difference is large: for the square
isotropic plate with `h/a = 0.1` clamped on four edges, the first frequency
`\omega a^2 \sqrt{\rho h/D}` of the TSDT is 30.18 with the slope free and
32.64 with the slope clamped, close to the 32.62 of the FSDT.

The removed terms leave null rows and columns in the matrices, which the
solvers of `structsolve <https://github.com/saullocastro/structsolve>`_
remove before solving. A free domain, all flags ``1``, has rigid-body
modes, and its stiffness matrix is singular.


Prescribed displacements
------------------------

Displacements different from zero, or zero displacements at points or
along lines inside the domain, are prescribed with
:meth:`.Shell.add_point_pd`, :meth:`.Shell.add_distr_pd_fixed_x` and
:meth:`.Shell.add_distr_pd_fixed_y`, constant (``cte=True``) or proportional
to the load factor (``cte=False``). They are imposed by the
:class:`.MultiDomain` that contains the shell, also an assembly of one shell
without connections, with ``pd_method``:

- ``'null-space'``: exactly, as linear constraints eliminated together with
  the connections, the solution being `\boldsymbol{c} = \boldsymbol{Z}\boldsymbol{c}_r + \boldsymbol{c}_{p0} +
  \lambda \boldsymbol{c}_{p1}`, with the reactions of :meth:`.MultiDomain.reaction_pd`
  and :meth:`.MultiDomain.reaction_resultant`, see :ref:`null_space_pd`;
- ``'penalty'``, the default: with springs between the structure and the
  support, whose stiffness the user adds to the stiffness matrix, e.g. with
  :func:`.fkCpd`, and whose forces enter the external force vector,
  :meth:`.MultiDomain.calc_fext`.

A single :class:`.Shell` can be supported at a point by the penalty springs
of :meth:`.Shell.calc_stiffness_point_constraint`, added by the user to the
constitutive stiffness matrix.

The in-plane displacements `u` and `v` are prescribed at the reference
surface, `z = 0`, or at the thickness coordinate given by the keyword ``z``
of these methods, with the same ``z`` given to the penalty kernels.

Displacement control and load control are equivalent when the prescribed
displacement produces the same membrane state as the load. A plate of
length `a` and thickness `h`, isotropic, with `u = 0` along `x = 0`, `u =
-\delta` along `x = a`, `v = 0` at one corner and the lateral edges free
in-plane, has the uniform membrane state

.. math::
    :label: bc-end-shortening

    u = -\delta \frac{x}{a}, \qquad v = \nu \delta \frac{y}{a}, \qquad
    N_{xx} = -\frac{E h \delta}{a}

that of the line load `N_{xx}` of :meth:`.Shell.add_distr_load_fixed_x`
along `x = a` with `u` free there. A flat plate buckles out of plane, its
modes having zero in-plane components, so that the critical end-shortening
`\delta_{cr}` of :meth:`.MultiDomain.linear_buckling`, prescribed with
``cte=False``, and the critical load of the equivalent `N_{xx}` satisfy
Eq. :eq:`bc-end-shortening`.
For a cylinder, whose modes couple the membrane and bending displacements,
they differ, see :ref:`null_space_pd`. As the loads, see :ref:`loads`, the
prescribed values are split into a constant and an incremented part,
`\bar{\boldsymbol{d}}_0 + \lambda \bar{\boldsymbol{d}}_1`, Eq. :eq:`ns-pd`: a constant
end-shortening `\delta_0` along one edge and an incremented one `\lambda
\delta_1` along the other reach the critical state at `\delta_0 + \lambda
\delta_1 = \delta_{cr}`, whereas a constant and an incremented prescription
of the same component along the same line contradict each other, the
constraints holding for every `\lambda`, and raise an error.


.. _boundary_conditions_face_clamp:

Face-clamped edges
------------------

An edge can be clamped by restraining the displacements of the bottom
and top faces of the laminate, `z = d \mp h/2`, the section being free to
deform in between. This face clamp, `u = v = 0` at both faces and `w = 0`,
is prescribed with the keyword ``z`` of :meth:`.Shell.add_distr_pd_fixed_x`
and :meth:`.Shell.add_distr_pd_fixed_y`; the flags of the edge are left
free, ``1``. With the in-plane displacement at `z`, see
:func:`.nullspace.pd_constraint_matrix`,

.. math::

    u(z) = u + \left(z - c_1 g(z)\right) \phi_x - c_1 g(z) w_{,x}, \qquad
    g(z) = (z - d)^3 + d^3

with `c_1 = 4/(3 h^2)` for the TSDT and `c_1 = 0` for the FSDT, the two
faces give `u = \phi_x = 0` for the FSDT, and `u = w_{,x} = 0` for the CLPT,
`u(z) = u - z w_{,x}`: the face clamp is the clamp of the flags. For the TSDT
the difference of the two faces gives `(h/3)(2 \phi_x - w_{,x}) = 0` and
their sum `u = -(d/2)(1 - 4 d^2/h^2) w_{,x}`, two conditions instead of the
three of the planar clamp; the section warps,

.. math::

    u(z) = \frac{w_{,x}}{2} \zeta \left(1 - \frac{4 \zeta^2}{h^2}\right),
    \qquad \zeta = z - d

which no combination of flags can express. Along the edge `w = 0` gives
`w_{,y} = 0`, and the two faces then give `v = \phi_y = 0`. The edge is
softer than the planar clamp and stiffer than the simple support, and,
being defined by the physical faces, it does not depend on the reference
surface.
