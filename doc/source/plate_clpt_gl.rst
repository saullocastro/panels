.. _plate_clpt_gl:

Large rotations: plates with Green-Lagrange membrane strains
============================================================

The model ``'plate_clpt_gl'`` is the classical laminated plate theory (CLPT)
of ``'plate_clpt_donnell'`` with the complete Green-Lagrange membrane
strains of the mid-surface, instead of the von Kármán strains. It is meant
for assemblies of plates, e.g. a wing box with :class:`.MultiDomain`, that
undergo large rigid rotations while their strains stay small. The kernels
:mod:`panels.models.plate_clpt_gl` and :mod:`panels.models.plate_clpt_gl_num`
are generated from a symbolic derivation with SymPy of the strains below.


Why the von Kármán strains fail in an assembly
----------------------------------------------

The von Kármán strains keep the quadratic terms of `w` only, `w_{,x}^2/2`,
`w_{,y}^2/2` and `w_{,x} w_{,y}`. In a box beam bent about a horizontal axis,
the rotation `\theta` of a cross-section is an out-of-plane rotation of the
skins, `w_{,x} = \theta`, and an in-plane rotation of the spars, `v_{,x} =
\pm \theta`, which enters no strain. A rigid rotation of the box then
stretches the skins by `\theta^2/2` relative to the spars, and the box
stiffens spuriously once its tip deflection reaches the order of its height,
instead of its length. For the wing box of the example below, with a height of
12 mm and a span of 1010 mm, the tip deflection at 25 N is 13.9 mm with
``'plate_clpt_donnell'`` against 22.7 mm of the linear solution, and 54.7 mm at
500 N against 454 mm.


Kinematics
----------

The degrees of freedom of a term are `u, v, w`, the displacements of the
mid-surface along the fixed local axes of the plate, and the strains are

.. math::

    \varepsilon_{xx} = u_{,x} + \tfrac{1}{2}\left(u_{,x}^2 + v_{,x}^2 + w_{,x}^2\right)
    \qquad
    \varepsilon_{yy} = v_{,y} + \tfrac{1}{2}\left(u_{,y}^2 + v_{,y}^2 + w_{,y}^2\right)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} + w_{,x} w_{,y}

    \kappa_{xx} = -w_{,xx} \qquad \kappa_{yy} = -w_{,yy} \qquad \kappa_{xy} = -2 w_{,xy}

the membrane strains being `\boldsymbol{E} = (\boldsymbol{F}^\top
\boldsymbol{F} - \boldsymbol{I})/2` with the deformation gradient of the
mid-surface `\boldsymbol{F} = [\boldsymbol{e}_x + \boldsymbol{u}_{,x} \;\;
\boldsymbol{e}_y + \boldsymbol{u}_{,y}]`. For a rigid rotation
`\boldsymbol{u} = (\boldsymbol{R} - \boldsymbol{I}) \boldsymbol{X}` the
columns of `\boldsymbol{F}` are two columns of `\boldsymbol{R}`, hence
`\boldsymbol{F}^\top \boldsymbol{F} = \boldsymbol{I}`: the membrane strains
vanish for any rotation, in plane or out of plane, whereas the von Kármán
`\varepsilon_{xx}` of an in-plane rotation `\theta` is `\cos\theta - 1`.
Without the quadratic terms of `u` and `v` the strains are those of
``'plate_clpt_donnell'``.


Internal forces and tangent stiffness
-------------------------------------

At a point, the strains depend on the nine generalized derivatives
`\boldsymbol{d} = (u_{,x}, u_{,y}, v_{,x}, v_{,y}, w_{,x}, w_{,y},
w_{,xx}, w_{,yy}, w_{,xy}) = \boldsymbol{\Phi} \boldsymbol{c}`. With
`\boldsymbol{B} = \partial \boldsymbol{\varepsilon}/\partial \boldsymbol{d}`
(6 x 9, linear in the gradients) and the stress resultants
`\boldsymbol{\sigma} = \boldsymbol{ABD}\, \boldsymbol{\varepsilon}`:

.. math::

    \boldsymbol{F}_{int} = \int \boldsymbol{\Phi}^\top \boldsymbol{B}^\top \boldsymbol{\sigma}\, dA
    \qquad
    \boldsymbol{K}_T = \int \boldsymbol{\Phi}^\top \left( \boldsymbol{B}^\top \boldsymbol{ABD}\, \boldsymbol{B}
    + \boldsymbol{H}(\boldsymbol{N}) \right) \boldsymbol{\Phi}\, dA

where `\boldsymbol{H}(\boldsymbol{N})` has the block `[[N_{xx}, N_{xy}],
[N_{xy}, N_{yy}]]` on each pair of gradients `(u_{,x}, u_{,y})`, `(v_{,x},
v_{,y})` and `(w_{,x}, w_{,y})`: the geometric stiffness matrices
:meth:`.Shell.calc_kG` and the analytical one of a constant stress state have
``u-u`` and ``v-v`` blocks besides the ``w-w`` block of the von Kármán plate.
As for the other models, the stress of the quadratic strains enters
:meth:`.Shell.calc_kC` with ``NLgeom=True`` and :meth:`.Shell.calc_kG` stays
homogeneous of degree one in `\boldsymbol{c}`, see
:ref:`tangent_product`, whose product ``calc_kTv`` is generated with the six
gradients in place of the two rotations of the von Kármán terms. The matrix
`\boldsymbol{B}^\top \boldsymbol{ABD}\, \boldsymbol{B} + \boldsymbol{H}` is
formed once per integration point, and each pair of terms costs 27
multiplications.

The integrands are still products of four approximation functions, so the
default numbers of integration points, :func:`.gauss_points`, are unchanged.
The linear constitutive stiffness, the mass and the aerodynamic matrices are
those of ``'plate_clpt_donnell'``: a linear static analysis and the natural
frequencies are identical, and a linear buckling eigenvalue changes only
through the in-plane blocks of the geometric stiffness, not at all for a
symmetric laminate and at the order of the membrane strains otherwise.
:func:`.clpt_field.fstrain` gives the Green-Lagrange membrane strains.

Follower pressures use the exact area vector of the plate,
``follower='quadratic'``, see :ref:`follower_pressure`; the first-order area
vector of ``follower='linear'`` assumes moderate rotations.


Limitations
-----------

- The changes of curvature are measured in the fixed local frame of the
  plate. A plate bent out of plane by a large rotation keeps the bending
  energy of the linear plate, its membrane strains being kept small by the
  shortening `u`: a cantilever strip under a tip force of constant
  direction bent out of plane follows the linear beam, `w = \alpha L/3` at
  `\alpha = P L^2/EI = 1` against `0.302 L` of the elastica, whereas bent in
  its plane, where the bending is carried by the membrane strains, it
  follows the elastica to 0.15 % at `\alpha = 2`. In a box beam the global
  bending is carried by the membrane strains of the skins and the in-plane
  bending of the spars, and the out-of-plane bending of the skins
  contributes little; a local buckle of a skin rotated by `\theta` is seen
  with its bending energy times `\cos^2\theta`. The model
  ``'plate_clpt_tl'`` measures the curvatures with the deformed normal, see
  :ref:`plate_clpt_tl`.
- The connections of :class:`.MultiDomain` are unchanged: they are linear
  in the displacements in the fixed local frames, exact for the continuity
  of the displacements at any rotation, and the rotation continuity
  (`w_{,y}` or `w_{,x}` across a joint) is that of small rotations, with an
  error of order `1 - \cos\theta` at a joint rotated by `\theta`. The
  null-space matrix of the constraints stays constant.
- The strains are small: the constitutive law is linear in the
  Green-Lagrange strains.


Accuracy for box beams
----------------------

For a box beam whose global bending is carried by the membrane strains of
the skins and the in-plane bending of the spars, the model follows the
large rotations of the beam. For the High-AR-t5 wing box of Elalfy, De
Breuker and Castro [elalfyWingboxPreprint]_ under a follower tip force,
modelled with 45 connected domains, the tip deflection is within 4 % of
their Abaqus shell model up to well beyond the buckling of the top skin.
