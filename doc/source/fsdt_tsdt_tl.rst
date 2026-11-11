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
\phi_x, \phi_y`. The models ``'plate_tsdt_tl'`` and ``'cylshell_tsdt_tl'``
extend them to the third-order shear deformation theory (TSDT) of Reddy
[reddy1984HigherOrder]_, [reddy1985Shells]_, with the same degrees of
freedom, see :ref:`tsdt_tl`.

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

.. _tsdt_tl:

Third-order shear deformation theory
------------------------------------

Reddy's displacement field [reddy1984HigherOrder]_ adds to the field of the
FSDT a cubic term that makes the transverse shear strains vanish at the
faces `z = \pm h/2`:

.. math::

    \mathbf{u}(z) = \mathbf{u} + z (\phi_x, \phi_y, 0) - c_1 z^3 (\phi_x +
    w_{,x}, \; \phi_y + w_{,y}, \; 0), \qquad c_1 = \frac{4}{3 h^2}

For finite rotations, the vector of the cubic term, `(\phi_x + w_{,x},
\phi_y + w_{,y}, 0)`, is read as the linear part of the difference between
the director :eq:`fsdt-tl-director` and the normal of the deformed
mid-surface, not normalized,

.. math::

    \mathbf{N} = \mathbf{X}_{,x} \times \mathbf{X}_{,y}

whose linear part is `(-w_{,x}, -w_{,y} + v/r, 1 + u_{,x} + v_{,y} + w/r)`
in the local basis. The deformed position of the point at `z` of the models
``'plate_tsdt_tl'`` and ``'cylshell_tsdt_tl'`` is

.. math::
    :label: tsdt-tl-field

    \mathbf{r}(z) = \mathbf{X} + z \mathbf{d} + z^3 \mathbf{q}, \qquad
    \mathbf{q} = -c_1 (\mathbf{d} - \mathbf{N})

Its strains are the coefficients of `z^0`, `z` and `z^3` of the in-plane
Green-Lagrange strains and of `z^0` and `z^2` of the transverse shear
strains, the powers of the linear TSDT: the strains
:eq:`fsdt-tl-strains` and

.. math::
    :label: tsdt-tl-strains

    \kappa^{(3)}_{xx} = \mathbf{X}_{,x} \cdot \mathbf{q}_{,x} \qquad
    \kappa^{(3)}_{yy} = \mathbf{X}_{,y} \cdot \mathbf{q}_{,y} -
    \frac{c_1}{r} (\varepsilon_{xx} + \varepsilon_{yy}) \qquad
    \kappa^{(3)}_{xy} = \mathbf{X}_{,x} \cdot \mathbf{q}_{,y} +
    \mathbf{X}_{,y} \cdot \mathbf{q}_{,x}

    \gamma^{(2)}_{yz} = 3 \mathbf{X}_{,y} \cdot \mathbf{q} \qquad
    \gamma^{(2)}_{xz} = 3 \mathbf{X}_{,x} \cdot \mathbf{q}

with the derivatives of the vectors along `y` in the local basis of the
cylinder as for `\mathbf{d}_{,y}`, and `1/r = 0` for the plate. Dropped, as
in the linear TSDT: the coefficients of `z^2`, `z^4` and `z^6` of the
in-plane strains, of the odd powers of the transverse shear strains and the
transverse normal strain. The second derivatives of `u` and `v` enter
through those of `\mathbf{X}`.

Since `\mathbf{X}_{,\alpha} \cdot \mathbf{N} = 0` for any deformation,
`\mathbf{X}_{,\alpha} \cdot \mathbf{N}_{,\beta} = -\mathbf{X}_{,\alpha\beta}
\cdot \mathbf{N}` and, for the plate,

.. math::

    \kappa^{(3)}_{xx} = -c_1 (\kappa_{xx} + \mathbf{X}_{,xx} \cdot
    \mathbf{N}) \qquad
    \kappa^{(3)}_{yy} = -c_1 (\kappa_{yy} + \mathbf{X}_{,yy} \cdot
    \mathbf{N}) \qquad
    \kappa^{(3)}_{xy} = -c_1 (\kappa_{xy} + 2 \mathbf{X}_{,xy} \cdot
    \mathbf{N})

    \gamma^{(2)}_\alpha = -3 c_1 \gamma_\alpha

Reddy's `\phi_{\alpha,\beta}` becomes the curvature of the director and
`w_{,\alpha\beta}` the second fundamental form of the deformed mid-surface,
`\mathbf{X}_{,\alpha\beta} \cdot \mathbf{N}`. The relation `\gamma^{(2)} =
-3 c_1 \gamma` is exact, so that the transverse shear strain through the
thickness, `\gamma_\alpha (1 - 4 z^2/h^2)`, vanishes at the faces for any
deformation: Reddy's traction-free faces hold at finite rotations.

Cylinder
^^^^^^^^

The `z` component of `\mathbf{N}` is `1 + \varepsilon_{xx} +
\varepsilon_{yy}` to first order, the stretch of the area of the
mid-surface. In the plate it is dropped with the coefficient of `z^3` of
the transverse shear strains, whose linear part it is, `c_1 (u_{,x} +
v_{,y})_{,\alpha}`. In the cylinder it enters `\mathbf{q}_{,y}` through the
curvature of the reference, `\mathbf{e}_{z,y} = \mathbf{e}_y/r`, and adds
`c_1 (u_{,x} + v_{,y} + w/r)/r` to the linear part of `\mathbf{X}_{,y}
\cdot \mathbf{q}_{,y}`. The term `-c_1 (\varepsilon_{xx} +
\varepsilon_{yy})/r` of :eq:`tsdt-tl-strains`, with the Green-Lagrange
membrane strains, removes it, as the Koiter terms of :eq:`fsdt-tl-strains`
remove the membrane strains from the linear curvatures; the linear part is
then that of ``'cylshell_tsdt_sanders'``, `\kappa^{(3)}_{xx} = -c_1
(\phi_{x,x} + w_{,xx})`, `\kappa^{(3)}_{yy} = -c_1 (\phi_{y,y} +
w_{,yy})`, `\kappa^{(3)}_{xy} = -c_1 (\phi_{x,y} + \phi_{y,x} + 2
w_{,xy})`, and the linear parts of ``'plate_tsdt_tl'`` and
``'cylshell_tsdt_tl'`` are those of ``'plate_tsdt_donnell'`` and
``'cylshell_tsdt_sanders'``.

Rigid rotations
^^^^^^^^^^^^^^^

The strains :eq:`tsdt-tl-strains` are dot products of vectors that a rigid
rotation turns together, and Green-Lagrange membrane strains; as for the
FSDT, only the second-order director cannot follow the rotation: a rigid
rotation of the flat plate gives zero membrane, bending and third-order
bending strains and transverse shear strains of `O(t^5)`, and a rotation of
a bent plate or of the cylinder changes the other strains by `O(t^4)`. The
linear third-order strains of ``'plate_tsdt_gl'`` and ``'cylshell_tsdt_gl'``
change at the second or the first order in the angle.

Third-order terms linear
^^^^^^^^^^^^^^^^^^^^^^^^

A simpler model keeps the first-order part :eq:`fsdt-tl-strains` and
the linear third-order strains of Reddy, those of the GL models,
`\kappa^{(3)}_{\alpha\beta} = -c_1 (\phi_{\alpha,\beta} + \phi_{\beta,\alpha}
+ 2 w_{,\alpha\beta})/(1 + \delta_{\alpha\beta})` and `\gamma^{(2)}_\alpha =
-3 c_1 (\phi_\alpha + w_{,\alpha})`, as Amabili [amabili2018Nonlinear]_,
Sec. 3.9, does in a model with the nonlinear terms of von Kármán. It has
the same linear part, but its shear strain at the faces is `\mathbf{X}_{,\alpha}
\cdot \mathbf{d} - (\phi_\alpha + w_{,\alpha})`, cubic, not zero, and a
rigid rotation about an oblique axis gives third-order strains of `O(t^2)`.
In large bending the shear strain of the director, `\mathbf{X}_{,x} \cdot
\mathbf{d}`, and the linear one, `\phi_x + w_{,x}`, cannot both vanish:
with `\gamma` and `\gamma_L` these two strains, the transverse shear energy
per unit area of an isotropic layer is `G h (\gamma^2 - \tfrac{2}{3} \gamma
\gamma_L + \tfrac{1}{5} \gamma_L^2)/2`, which penalizes the bending by a
factor of the order of `(L/h)^2`: the model locks. A strip of `L/h = 500`,
clamped at one end under a tip force of constant direction normal to its
plane, `\alpha = P L^2/EI = 1`, whose inextensible elastica has the tip
rotation 0.46 rad, `w = 0.3017 L` and `-u = 0.0564 L`, gives with that model
`w` 25.5 % and `-u` 47 % too small, and at `\alpha = 3` (0.99 rad) 50 % and
79 %; the field :eq:`tsdt-tl-field` gives -0.22 % and -0.47 % at `\alpha =
1`, -1.4 % and -3.8 % at `\alpha = 3`, close to the FSDT model (-0.27 %,
-0.59 %; -1.8 %, -4.7 %), the differences coming from the second-order
director. A thick strip, `L/h = 5`, compared with the geometrically exact
beam of Reissner [reissner1972Beam]_ with the shear stiffness of its linear
deflection, at `\alpha = 3` (0.98 rad): -3.4 % and -7.9 % for the linear
third-order terms, -0.81 % and -0.20 % for :eq:`tsdt-tl-field`. Schmidt and
Reddy [schmidt1988Refined]_, Eqs. (17)-(19), written for an expansion of
any order in the thickness coordinate, have products of the rotation
terms of all orders in the strains, i.e. third-order strains that are not
linear.

The field :eq:`tsdt-tl-field` costs more: 21 generalized derivatives at an
integration point of the plate instead of 15, and for a plate of 12 x 12
terms the constitutive and geometric stiffness matrices take about 1.3 and
1.7 times as long as with the linear third-order terms.

Tangent stiffness
-----------------

At an integration point the strains depend on the generalized derivatives
`\boldsymbol{d} = \boldsymbol{\Phi} \boldsymbol{c}` of the GL models: the
gradients of `u, v, w`, for the cylinder the values of `v` and `w`, the
rotations and their gradients, and for the TSDT the second derivatives
of `u`, `v` and `w`. With `\boldsymbol{B} = \partial
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
- The third-order shear deformation theory keeps the transverse shear
  strain of the director through the thickness, `\gamma_\alpha (1 - 4
  z^2/h^2)`, and drops the higher powers of the thickness coordinate, as the
  linear TSDT.
