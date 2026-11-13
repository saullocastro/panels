.. _fsdt_tsdt_gl:

Shear deformable models with Green-Lagrange membrane strains
============================================================

The models ``'plate_fsdt_gl'``, ``'plate_tsdt_gl'``, ``'cylshell_fsdt_gl'``
and ``'cylshell_tsdt_gl'`` combine the complete Green-Lagrange membrane
strains of ``'plate_clpt_gl'`` and ``'cylshell_clpt_gl'`` (see
:ref:`plate_clpt_gl`) with the first-order (FSDT) and third-order (TSDT)
shear deformation theories of ``'plate_fsdt_donnell'``,
``'plate_tsdt_donnell'``, ``'cylshell_fsdt_sanders'`` and
``'cylshell_tsdt_sanders'``. The degrees of freedom of a term are those of
the shear deformation theories, `u, v, w, \phi_x, \phi_y`.

Strains
-------

The membrane strains are the Green-Lagrange strains of the mid-surface,
`\boldsymbol{E} = (\boldsymbol{F}^\top \boldsymbol{F} - \boldsymbol{I})/2`,
with the columns of `\boldsymbol{F}` the tangent vectors of the deformed
mid-surface in the local basis. For the cylinder of radius `r`, `y` the
arc length and `w` positive outwards:

.. math::
    :label: fsdt-gl-membrane

    \varepsilon_{xx} = u_{,x} + \tfrac{1}{2}\left(u_{,x}^2 + v_{,x}^2 +
    w_{,x}^2\right)

    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \tfrac{1}{2}\left(u_{,y}^2 +
    \left(v_{,y} + \frac{w}{r}\right)^2 + \left(w_{,y} -
    \frac{v}{r}\right)^2\right)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x}\left(v_{,y} +
    \frac{w}{r}\right) + w_{,x}\left(w_{,y} - \frac{v}{r}\right)

those of the Flügge-Lur'e-Byrne non-linear theory as given by Amabili
[amabili2008Nonlinear]_, Eqs. (1.118)-(1.120), and `1/r = 0` for the plates. The
bending and transverse shear strains are the linear ones of the FSDT,
with the terms of Sanders [sanders1963Nonlinear]_ for the cylinders:

.. math::
    :label: fsdt-gl-bending

    \kappa_{xx} = \phi_{x,x} \qquad \kappa_{yy} = \phi_{y,y} +
    \frac{v_{,y}}{r} \qquad \kappa_{xy} = \phi_{x,y} + \phi_{y,x} +
    \frac{3}{2} \frac{v_{,x}}{r} - \frac{1}{2} \frac{u_{,y}}{r}

    \gamma_{yz} = \phi_y + w_{,y} \qquad \gamma_{xz} = \phi_x + w_{,x}

and, for the TSDT of Reddy [reddy1984HigherOrder]_, the third-order terms
`-c_1 (\phi_{x,x} + w_{,xx})`, `-c_1 (\phi_{y,y} + w_{,yy})`, `-c_1
(\phi_{x,y} + \phi_{y,x} + 2 w_{,xy})` of the in-plane strains and
`-3 c_1 \gamma_{yz}`, `-3 c_1 \gamma_{xz}` of the transverse shear strains,
`c_1 = 4/(3 h^2)`, as in :ref:`tsdt_offset`.

Relation with the other models
------------------------------

Without the quadratic terms the strains are those of
``'plate_fsdt_donnell'``, ``'plate_tsdt_donnell'``,
``'cylshell_fsdt_sanders'`` and ``'cylshell_tsdt_sanders'``: the linear
constitutive stiffness, the mass and the aerodynamic matrices are theirs,
and so are linear static solutions and natural frequencies. Their
non-linear terms, the squares of the rotations `\beta_x = -w_{,x}`,
`\beta_y = -w_{,y} + v/r`, are contained in Eq. :eq:`fsdt-gl-membrane`,
which adds the quadratic terms of the in-plane gradients `u_{,x}`,
`u_{,y}`, `v_{,x}` and `v_{,y} + w/r`; the geometric stiffness has blocks
`u-u` and `v-v` (and `v-w` for the cylinders) besides the block `w-w`.
Under an axial membrane force the term `N_{xx} v_{,x}^2` is the one that
gives the Euler load of a tube bending as a beam, as for
``'cylshell_clpt_gl'``.

Rigid rotations
---------------

For a finite rigid rotation `\boldsymbol{R}` the tangent vectors are
rotated without stretch, `\boldsymbol{F}^\top \boldsymbol{F} =
\boldsymbol{I}`, and the membrane strains vanish for any angle and axis.
For the plates the displacement gradients are constant, the first two
columns of `\boldsymbol{R} - \boldsymbol{I}`; with the constant rotations
`\phi_x = -w_{,x}`, `\phi_y = -w_{,y}` the bending, third-order and
transverse shear strains vanish too, and a rigid rotation of any size
stores no energy. For the cylinders the linear bending strains of a finite
rigid rotation about an axis normal to the cylinder axis are of the order
`(1 - \cos t)/r`, as for ``'cylshell_clpt_gl'``. A bending deformation
superposed on a large rotation `t` about an in-plane axis is seen with the
factor `\cos t`, the bending strains being measured in the fixed local
basis.

Tangent stiffness
-----------------

At an integration point the strains `\boldsymbol{\varepsilon}` depend on
the generalized derivatives `\boldsymbol{d} = \boldsymbol{\Phi}
\boldsymbol{c}`: the gradients of `u, v, w`, for the cylinders the values
of `v` and `w`, the rotations and their gradients and, for the TSDT, the
second derivatives of `w`. With `\boldsymbol{B} = \partial
\boldsymbol{\varepsilon}/\partial \boldsymbol{d}`, the generalized
stresses `\boldsymbol{\sigma} = \boldsymbol{F}_c \boldsymbol{\varepsilon}`
(`\boldsymbol{F}_c` the constitutive matrix of the FSDT or TSDT) and the
six local gradients `\boldsymbol{\beta} = \boldsymbol{G}_m \boldsymbol{d}`
of Eq. :eq:`fsdt-gl-membrane`:

.. math::

    \boldsymbol{f}_{int} = \int \boldsymbol{\Phi}^\top \boldsymbol{B}^\top
    \boldsymbol{\sigma} \, dA \qquad
    \boldsymbol{K}_T = \int \boldsymbol{\Phi}^\top \left(\boldsymbol{B}^\top
    \boldsymbol{F}_c \boldsymbol{B} + \boldsymbol{G}_m^\top
    \mathrm{diag}(\boldsymbol{N}, \boldsymbol{N}, \boldsymbol{N})
    \boldsymbol{G}_m\right) \boldsymbol{\Phi} \, dA

with `\boldsymbol{N}` the `2 \times 2` matrix of the membrane stresses.
Only the membrane strains are non-linear and quadratic, so
`\boldsymbol{B}` is linear in `\boldsymbol{d}` and the integrands are
products of four approximation functions, integrated exactly by the
default Gauss-Legendre rule of the linear models.

Shear deformation and total-Lagrangian kinematics
-------------------------------------------------

With independent rotations the total-Lagrangian kinematics of a shell
rotate the director through finite angles: in the degenerate-continuum
shell elements the director is a unit vector defined by two angles, and
the strains are the Green-Lagrange strains of the shell continuum
[crisfield1991Nonlinear]_, Sec. 8.2. The rotations of these models are not finite
rotations of the director; their bending and transverse shear strains stay
linear. The total-Lagrangian FSDT models with the second-order director of
the moderate rotation theory are described in :ref:`fsdt_tsdt_tl`.

