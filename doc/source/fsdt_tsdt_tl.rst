.. _fsdt_tsdt_tl:

Total-Lagrangian shear deformable models
========================================

The models ``'plate_fsdt_tl'`` and ``'cylshell_fsdt_tl'`` keep the complete
Green-Lagrange membrane strains of ``'plate_fsdt_gl'`` and
``'cylshell_fsdt_gl'`` (see :ref:`fsdt_tsdt_gl`) and measure the bending
and transverse shear strains with a director that rotates with the
deformation, as ``'plate_clpt_tl'`` and ``'cylshell_clpt_tl'`` measure the
curvatures with the deformed normal. The degrees of freedom of a term are
those of the first-order shear deformation theory (FSDT), `u, v, w,
\phi_x, \phi_y`.

Moderate rotation kinematics
----------------------------

The displacement field is that of the moderate rotation theory of Schmidt
and Reddy [schmidt1988Refined]_, Eq. (16), and Palmerio, Reddy and Schmidt
[palmerio1990Moderate]_, Eq. (16), linear through the thickness:

.. math::

    \mathbf{u}(z) = \mathbf{u}^{(0)} + z \, \mathbf{u}^{(1)}

with the mid-surface displacements `\mathbf{u}^{(0)} = (u, v, w)` and the
components `u^{(1)}_\alpha` and `u^{(1)}_3` of the change of the normal.
Strains of order `\epsilon^2`, rotations of the normal of order `\epsilon`
and rotations about the normal of order `\epsilon^2`
([palmerio1990Moderate]_, Eqs. (9) and (18)) lead to strains of the
Green-Lagrange tensor kept to `O(\epsilon^3)`, Eq. (20) of
[palmerio1990Moderate]_, among which the transverse normal strain

.. math::

    E^{(0)}_{33} = u^{(1)}_3 + \frac{1}{2} u^{(1)}_\lambda u^{(1)\lambda}

The theory of [schmidt1988Refined]_ keeps `u^{(1)}_3` as an unknown and
does not assume that the normal is inextensible (p. 613 there). The models
of panels set `E^{(0)}_{33} = 0`, so that `u^{(1)}_3 = -(\psi_x^2 +
\psi_y^2)/2`, with `\psi_\alpha = u^{(1)}_\alpha`, and the deformed position
of the point at `z` is `\mathbf{X} + z \mathbf{d}`, with the second-order
director

.. math::
    :label: fsdt-tl-director

    \mathbf{d} = \psi_x \mathbf{e}_x + \psi_y \mathbf{e}_y + \left(1 -
    \frac{1}{2}(\psi_x^2 + \psi_y^2)\right) \mathbf{e}_z, \qquad
    |\mathbf{d}|^2 = 1 + \frac{1}{4}(\psi_x^2 + \psi_y^2)^2

in the local basis of the mid-surface, `\mathbf{e}_{y,y} =
-\mathbf{e}_z/r`, `\mathbf{e}_{z,y} = \mathbf{e}_y/r` for the cylinder and
`1/r = 0` for the plate.

The components `\psi_\alpha` follow from the linear part of the transverse
shear strain of [palmerio1990Moderate]_, Eq. (20), `2 E^{(0)}_{\alpha 3} =
\Phi^{(0)}_{\alpha 3} + u^{(1)}_\alpha + \ldots`, with
`\Phi^{(0)}_{\alpha 3} = u_{3,\alpha} + b^\lambda_\alpha u_\lambda`, Eq.
(15c). For the cylinder, `b^y_y = -1/r` with `w` positive outwards, so
`\Phi^{(0)}_{y3} = w_{,y} - v/r`, and the transverse shear strain of the FSDT
with the Sanders-Koiter kinematics, `\gamma_{yz} = \phi_y + w_{,y}`, gives

.. math::

    \psi_x = \phi_x, \qquad \psi_y = \phi_y + \frac{v}{r}

the rotation of the normal of ``'cylshell_fsdt_sanders'``, whose
tangential displacement through the thickness is `(1 + z/r) v + z \phi_y`.
For the plate, `\psi_\alpha = \phi_\alpha`. With `\psi_y = \phi_y` the
cylinder would not keep a rigid rotation about its axis free of strain,
where `\phi_y = 0` and `v/r` is the rotation.

Strains
-------

With the tangent vectors of the deformed mid-surface in the local basis,

.. math::

    \mathbf{X}_{,x} = (1 + u_{,x}, \; v_{,x}, \; w_{,x}) \qquad
    \mathbf{X}_{,y} = (u_{,y}, \; 1 + v_{,y} + w/r, \; w_{,y} - v/r)

and the derivatives of the director `\mathbf{d}_{,x} = (\psi_{x,x},
\psi_{y,x}, \psi_{z,x})`, `\mathbf{d}_{,y} = (\psi_{x,y}, \psi_{y,y} +
\psi_z/r, \psi_{z,y} - \psi_y/r)`, `\psi_z = 1 - (\psi_x^2 + \psi_y^2)/2`,
the strains are

.. math::
    :label: fsdt-tl-strains

    \varepsilon_{xx} = \tfrac{1}{2}(\mathbf{X}_{,x} \cdot \mathbf{X}_{,x} - 1)
    \qquad \varepsilon_{yy} = \tfrac{1}{2}(\mathbf{X}_{,y} \cdot
    \mathbf{X}_{,y} - 1) \qquad \gamma_{xy} = \mathbf{X}_{,x} \cdot
    \mathbf{X}_{,y}

    \kappa_{xx} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,x} \qquad
    \kappa_{yy} = \mathbf{X}_{,y} \cdot \mathbf{d}_{,y} - \frac{1}{r} -
    \frac{\varepsilon_{yy}}{r} \qquad
    \kappa_{xy} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,y} + \mathbf{X}_{,y}
    \cdot \mathbf{d}_{,x} - \frac{\gamma_{xy}}{2 r}

    \gamma_{yz} = \mathbf{X}_{,y} \cdot \mathbf{d} \qquad
    \gamma_{xz} = \mathbf{X}_{,x} \cdot \mathbf{d}

cubic polynomials of the derivatives of the displacements and rotations.
Without the terms `-\varepsilon_{yy}/r` and `-\gamma_{xy}/(2r)` these are
the coefficients of `z^0` and `z` of the Green-Lagrange strains of the
field `\mathbf{X} + z \mathbf{d}`, with the reference vectors
`\mathbf{e}_x` and `(1 + z/r) \mathbf{e}_y`, and the transverse components
at the mid-surface. The two terms, Koiter's modification of the changes of
curvature [vanderheijden2008Koiter]_, p. 202, also used by
``'cylshell_clpt_tl'`` (see :ref:`cylshell_clpt_tl`), make the linear part
exactly that of ``'cylshell_fsdt_sanders'``, as follows.

Linear part
-----------

With `\psi_y = \phi_y + v/r`, `\psi_z = 1 + O(\psi^2)`, and keeping the
terms linear in the displacements and rotations, the dot products of
:eq:`fsdt-tl-strains` are

.. math::

    \mathbf{X}_{,x} \cdot \mathbf{d}_{,x} \approx \phi_{x,x} \qquad
    \mathbf{X}_{,y} \cdot \mathbf{d}_{,y} - \frac{1}{r} \approx
    \psi_{y,y} + \frac{1}{r}\left(v_{,y} + \frac{w}{r}\right) =
    \phi_{y,y} + \frac{v_{,y}}{r} + \frac{\varepsilon^L_{yy}}{r}

    \mathbf{X}_{,x} \cdot \mathbf{d}_{,y} \approx \phi_{x,y} +
    \frac{v_{,x}}{r} \qquad
    \mathbf{X}_{,y} \cdot \mathbf{d}_{,x} \approx \psi_{y,x} = \phi_{y,x} +
    \frac{v_{,x}}{r}

    \mathbf{X}_{,y} \cdot \mathbf{d} \approx \psi_y + w_{,y} - \frac{v}{r}
    = \phi_y + w_{,y} \qquad
    \mathbf{X}_{,x} \cdot \mathbf{d} \approx \phi_x + w_{,x}

with `\varepsilon^L_{yy} = v_{,y} + w/r` and `\gamma^L_{xy} = u_{,y} +
v_{,x}` the linear membrane strains. In `\mathbf{X}_{,y} \cdot
\mathbf{d}_{,y}` the reference `1/r` and the hoop strain come from `(1 +
v_{,y} + w/r) \psi_z/r`; in `\mathbf{X}_{,x} \cdot \mathbf{d}_{,y}` the term
`v_{,x}/r` comes from `v_{,x} \psi_z/r`. Without the two Koiter terms the
linear curvatures are those of Sanders plus `(0, \varepsilon^L_{yy}/r,
\gamma^L_{xy}/(2r))`; with them

.. math::

    \kappa^L_{xx} = \phi_{x,x} \qquad
    \kappa^L_{yy} = \phi_{y,y} + \frac{v_{,y}}{r} \qquad
    \kappa^L_{xy} = \phi_{x,y} + \phi_{y,x} + \frac{3}{2} \frac{v_{,x}}{r}
    - \frac{1}{2} \frac{u_{,y}}{r}

    \gamma^L_{yz} = \phi_y + w_{,y} \qquad \gamma^L_{xz} = \phi_x + w_{,x}

the linear strains of ``'cylshell_fsdt_sanders'``, whose twist contains the
rotation about the normal, `(v_{,x} - u_{,y})/(2r)`. The symbolic check
``check_linear_part`` of the generator of ``'cylshell_fsdt_tl'`` verifies
each of these steps and the equality of the linear strain-displacement
operator with that of ``'cylshell_fsdt_sanders'``. The Koiter terms use the
complete Green-Lagrange strains, which a rigid rotation does not change.
For the plate, `1/r = 0`, the linear part is that of
``'plate_fsdt_donnell'`` (``check_linear_part`` of ``'plate_fsdt_tl'``).

Dropped terms
-------------

As in every FSDT
model of panels, the coefficient of `z^2` of the in-plane strains is
dropped, and so is the coefficient of `z` of the transverse shear strains,
`\mathbf{d}_{,\alpha} \cdot \mathbf{d} = (\psi_x^2 + \psi_y^2)(\psi_x
\psi_{x,\alpha} + \psi_y \psi_{y,\alpha})/2`, of the fourth order.

Comparison with the moderate rotation theory
--------------------------------------------

With the orders of magnitude of [palmerio1990Moderate]_, the strains
:eq:`fsdt-tl-strains` without the two Koiter terms are those of Eq. (20)
there, with `E^{(0)}_{33} = 0`, to `O(\epsilon^3)`: the differences are of
the fourth order, e.g. the quadratic terms of the in-plane gradients in the
membrane strains, `u_{,x} \phi_{x,x}` in `\kappa_{xx}` and
`u^{(1)}_3 (w_{,x} + \phi_x)/2` in `\gamma_{xz}`. For the plate the
non-linear terms of the curvatures, among them the cubic terms `-w_{,x}
(\phi_x \phi_{x,x} + \phi_y \phi_{y,x})` from the third component of the
director, are all of the fourth order once the normal is inextensible, and
the model differs from ``'plate_fsdt_gl'`` at `O(\epsilon^3)` through the
transverse shear strains only. For the cylinder the curvatures contain
quadratic terms of the order of the linear ones, from `\Phi^{(0)}_{y3}
\Phi^{(1)}_{y3}` with `\Phi^{(1)}_{y3} = u^{(1)}_{3,y} - \psi_y/r`, which the
linear curvatures of ``'cylshell_fsdt_gl'`` lack. The terms beyond
`\epsilon^3` are kept, so that the strains are exact for the field
`\mathbf{X} + z \mathbf{d}`, built from dot products of vectors that a
rigid rotation turns together.

Rigid rotations
---------------

Under a rigid rotation `\boldsymbol{R}` of a deformed state the vectors
`\mathbf{X}_{,\alpha}` turn without change of their dot products, and the
membrane strains do not change for any angle. The director
:eq:`fsdt-tl-director` cannot follow the rotation exactly: with `\psi` the
components of the rotated director, its third component differs from that
of the unit vector by `(\psi_x^2 + \psi_y^2)^2/8 + \ldots`, of the fourth
order in the angle `t`. A rigid rotation of the flat plate, with the
rotations those of the rotated normal, gives zero membrane and bending
strains and transverse shear strains of `O(t^5)`; a rotation of a bent
plate or of the cylinder changes the curvatures and transverse shear
strains by `O(t^4)`, against `O(t^2)`, the factor `\cos t`, for the linear
curvatures of the GL models.

Tangent stiffness
-----------------

At an integration point the strains depend on the generalized derivatives
`\boldsymbol{d} = \boldsymbol{\Phi} \boldsymbol{c}` of the GL models: the
gradients of `u, v, w`, for the cylinder the values of `v` and `w`, the
rotations and their gradients. With `\boldsymbol{B} = \partial
\boldsymbol{\varepsilon}/\partial \boldsymbol{d}`, quadratic in
`\boldsymbol{d}`, and the generalized stresses `\boldsymbol{\sigma} =
\boldsymbol{F}_c \boldsymbol{\varepsilon}`:

.. math::

    \boldsymbol{f}_{int} = \int \boldsymbol{\Phi}^\top \boldsymbol{B}^\top
    \boldsymbol{\sigma} \, dA \qquad
    \boldsymbol{K}_T = \int \boldsymbol{\Phi}^\top \left(\boldsymbol{B}^\top
    \boldsymbol{F}_c \boldsymbol{B} + \boldsymbol{H}(\boldsymbol{\sigma},
    \boldsymbol{d})\right) \boldsymbol{\Phi} \, dA, \qquad
    \boldsymbol{H} = \sum_r \sigma_r \frac{\partial^2 \varepsilon_r}{\partial
    \boldsymbol{d}^2}

`\boldsymbol{H}` is linear in `\boldsymbol{d}` and involves the moments and
transverse shear forces as well as the membrane forces. The geometric
stiffness of linear buckling is `\boldsymbol{H}(\boldsymbol{\sigma}_L,
\boldsymbol{0})`, homogeneous in the prebuckling state; with a constant
membrane state only it is that of the GL models, whose analytical kernels
the two models use for linear statics, vibration and linear buckling. The
integrands of `\boldsymbol{f}_{int}` and `\boldsymbol{K}_T` are polynomials
of degree `6 p` along each direction, `p` the degree of the approximation
functions (the rotations enter without derivative), integrated exactly by
`3 p + 1` Gauss-Legendre points, the default of these models.

Limitations
-----------

- Moderate rotations: the director has the length `(1 + (\psi_x^2 +
  \psi_y^2)^2/4)^{1/2}`, with `\psi_x^2 + \psi_y^2 = \sin^2 t` for a
  normal rotated by `t`: 0.7 % too long at 0.5 rad, 6 % at 1 rad.
- The thickness does not change and the constitutive law is linear in the
  Green-Lagrange strains (small strains).
- The connections of :class:`.MultiDomain` stay linear in the local
  displacements and rotations.
- The third-order shear deformation theory has no total-Lagrangian model.
