.. _plate_clpt_tl:

Large rotations: total-Lagrangian Kirchhoff-Love plates
=======================================================

The model ``'plate_clpt_tl'`` is the plate of ``'plate_clpt_gl'``, with the
complete Green-Lagrange membrane strains, whose changes of curvature are
measured with the normal of the deformed mid-surface instead of the fixed
local frame. All its strains are then invariant under any rigid rotation of
any deformed state: a bending deformation carried through a large rotation,
e.g. a local buckle of a skin of a wing box bent by a large rotation, keeps
its bending energy, which ``'plate_clpt_gl'`` multiplies by `\cos^2\theta`.
The derivation, with SymPy, is in
``theory/shells/plate_clpt_tl/plate_clpt_tl.py``, with the note
``plate_clpt_tl.md``, and ``write_pyx.py`` generates
:mod:`panels.models.plate_clpt_tl_num` with the helpers of the generator of
``'plate_clpt_gl'``.


Kinematics
----------

With `\boldsymbol{X} = (x, y, 0) + (u, v, w)` the deformed mid-surface and
its area vector `\boldsymbol{a} = \boldsymbol{X}_{,x} \times
\boldsymbol{X}_{,y}`, the strains are

.. math::

    \varepsilon_{xx} = u_{,x} + \tfrac{1}{2}\left(u_{,x}^2 + v_{,x}^2 + w_{,x}^2\right)
    \qquad
    \varepsilon_{yy} = v_{,y} + \tfrac{1}{2}\left(u_{,y}^2 + v_{,y}^2 + w_{,y}^2\right)

    \gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} v_{,y} + w_{,x} w_{,y}

    \kappa_{xx} = -\boldsymbol{a} \cdot \boldsymbol{u}_{,xx} \qquad
    \kappa_{yy} = -\boldsymbol{a} \cdot \boldsymbol{u}_{,yy} \qquad
    \kappa_{xy} = -2\, \boldsymbol{a} \cdot \boldsymbol{u}_{,xy}

    \boldsymbol{a} = \left( v_{,x} w_{,y} - w_{,x} (1 + v_{,y}),\;
    w_{,x} u_{,y} - (1 + u_{,x}) w_{,y},\;
    (1 + u_{,x})(1 + v_{,y}) - v_{,x} u_{,y} \right)

with `\boldsymbol{u}_{,xx} = (u_{,xx}, v_{,xx}, w_{,xx})` and so on. The
linear part is that of ``'plate_clpt_donnell'``, and the quadratic part of
`\kappa_{xx}` is `w_{,x} u_{,xx} + w_{,y} v_{,xx} - (u_{,x} + v_{,y})
w_{,xx}`. For a rotation `\boldsymbol{X} \to \boldsymbol{R} \boldsymbol{X}`
of a deformed state, `\boldsymbol{a} \to \boldsymbol{R} \boldsymbol{a}` and
`\boldsymbol{u}_{,\alpha\beta} \to \boldsymbol{R}
\boldsymbol{u}_{,\alpha\beta}`, so the curvatures do not change.

The area vector is used instead of the unit normal `\boldsymbol{n} =
\boldsymbol{a}/|\boldsymbol{a}|`, with `|\boldsymbol{a}|^2 =
\det(\boldsymbol{F}^\top \boldsymbol{F}) = (1 + 2 \varepsilon_{xx})(1 + 2
\varepsilon_{yy}) - \gamma_{xy}^2`: `|\boldsymbol{a}| = 1` for any rigid
motion, and `1 + \varepsilon_{xx} + \varepsilon_{yy}` to first order
otherwise. The curvatures are `|\boldsymbol{a}|` times those of the unit
normal, and the bending stiffness is scaled by `1 + 2(\varepsilon_{xx} +
\varepsilon_{yy})`, an error of the order of the membrane strain, as the
other terms neglected by the Kirchhoff-Love kinematics with a constitutive
law linear in the Green-Lagrange strains; in exchange the strains are
cubic polynomials in the displacements, the integrands polynomials, and the
Gauss-Legendre rule is exact.


Internal forces and tangent stiffness
-------------------------------------

At a point, the strains depend on the six gradients and the nine second
derivatives of `u, v, w`, `\boldsymbol{d} = \boldsymbol{\Phi}
\boldsymbol{c}`. With `\boldsymbol{B} = \partial
\boldsymbol{\varepsilon}/\partial \boldsymbol{d}`, quadratic in
`\boldsymbol{d}`, and `\boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{d})
= \sum_r \sigma_r\, \partial^2 \varepsilon_r/\partial \boldsymbol{d}^2`,
linear in `\boldsymbol{d}`:

.. math::

    \boldsymbol{F}_{int} = \int \boldsymbol{\Phi}^\top \boldsymbol{B}^\top \boldsymbol{\sigma}\, dA
    \qquad
    \boldsymbol{K}_T = \int \boldsymbol{\Phi}^\top \left( \boldsymbol{B}^\top \boldsymbol{ABD}\, \boldsymbol{B}
    + \boldsymbol{H}(\boldsymbol{\sigma}, \boldsymbol{d}) \right) \boldsymbol{\Phi}\, dA

:meth:`.Shell.calc_kG` assembles `\boldsymbol{H}(\boldsymbol{\sigma}_L +
\bar{\boldsymbol{N}}, \boldsymbol{0})`, homogeneous of degree one in
`\boldsymbol{c}` as the linear buckling analysis requires: the ``u-u``,
``v-v`` and ``w-w`` blocks of ``'plate_clpt_gl'``, and ``u-w`` and ``v-w``
blocks of the moments of the linear strains. :meth:`.Shell.calc_kC` with
``NLgeom=True`` assembles the rest, so that their sum is the Jacobian of
:meth:`.Shell.calc_fint`. With a constant membrane stress the geometric
stiffness is that of ``'plate_clpt_gl'``, whose analytical kernels the model
uses: linear static solutions and natural frequencies are those of
``'plate_clpt_donnell'``, and linear buckling eigenvalues those of
``'plate_clpt_gl'`` when the prebuckling state has no moments.
:meth:`.Shell.calc_kTv` evaluates `\boldsymbol{B}^\top \boldsymbol{ABD}\,
\boldsymbol{B} \boldsymbol{\Phi} \boldsymbol{v} + \boldsymbol{H}
\boldsymbol{\Phi} \boldsymbol{v}` at each point, the strains not having the
form of :ref:`tangent_product`. Follower pressures use the exact area
vector of the plate, ``follower='quadratic'``, and the connections of
:class:`.MultiDomain` are unchanged.

The integrands are polynomials of degree `6p - 2` along each direction,
for approximation functions of degree `p = \max(m - 1, 3)`, against `4p`
for the other models; the default number of points of this model,
:func:`.gauss_points` with the key ``'integrand_degree'`` of
:mod:`panels.modelDB`, is `3p`, 21 for ``m = 8`` against 15, which
integrates them exactly. The highest-degree terms are products of five or
six of the highest functions and carry little weight: with the 15 points of
``'plate_clpt_gl'`` the internal forces of a state with gradients of order
0.5 differ by 3e-13, so ``nx = gauss_points(m)``, ``ny = gauss_points(n)``
is a cheaper choice in practice. :func:`.clpt_field.fstrain` gives the
curvatures of the deformed normal.


Verification
------------

``tests/tests_shell/test_plate_clpt_tl.py``:

- rigid rotations of 0.1 to 1 rad of a free plate, in plane, out of plane and
  about an oblique axis, give zero strains, curvatures and internal forces;
- a bent plate rotated rigidly by 0.5 and 1 rad keeps its strains and its
  bending energy to 1e-8, whereas the bending energy of ``'plate_clpt_gl'``
  is multiplied by `\cos^2\theta` (0.77 and 0.29);
- the tangent stiffness and :meth:`.Shell.calc_kTv` are the central
  differences of :meth:`.Shell.calc_fint` at a state rotated by 1 rad;
- a cantilever strip, ``L = 1``, ``b = 0.04``, ``t = 2e-3``, under a tip
  force of constant direction bent out of plane follows the inextensible
  elastica to 1e-4 at `P L^2/EI = 1` and `3` (tip rotations of 0.46 and
  1.0 rad), `w = 0.30172 L` and `0.6033 L`, where ``'plate_clpt_gl'`` follows
  the linear beam, `w = L/3` at `P L^2/EI = 1`; bent in its plane it follows
  the elastica as ``'plate_clpt_gl'``;
- for a simply supported plate under pressure with centre deflections of
  1.1, 3.4 and 11 times the thickness, the deflection is smaller than that
  of ``'plate_clpt_gl'`` by 2e-5, 1.1e-4 and 1.3e-3, the order of the
  membrane strain.


Wing box under a follower tip force
-----------------------------------

The High-AR-t5 wing box of :ref:`plate_clpt_gl`, with the same 45 domains
(6264 degrees of freedom) and 20 Newton-Raphson increments, deflects at the
tip (mm):

.. list-table::
    :header-rows: 1

    * - tip force (N)
      - 25
      - 100
      - 200
      - 300
      - 400
      - 500
    * - Abaqus, Fig. 6 [elalfyWingboxPreprint]_
      - 23.3
      - 93.2
      - 189.6
      - 298.9
      - 410.0
      - 515.5
    * - ``'plate_clpt_tl'``
      - 22.68
      - 90.84
      - 184.32
      - 286.16
      - 388.84
      - 488.73
    * - ``'plate_clpt_gl'``
      - 22.68
      - 90.84
      - 184.37
      - 287.24
      - 393.59
      - 499.83

The two models agree to 0.03 % up to 200 N, below the first buckling load,
where the box bends through the membrane strains of the skins and the
in-plane bending of the spars. Above it, ``'plate_clpt_tl'`` is up to
2.2 % stiffer, 5.2 % below Abaqus at 500 N against 3.0 %, which is
consistent with the buckles of the top skin, rotated with the box, keeping
their bending energy instead of losing the factor `\cos^2\theta`. The
remaining difference with Abaqus includes the 2.7 % of the linear range
(mid-surface domains, penalty connections), the perfect geometry and the
rotation continuity of the connections, linear in the displacements of the
fixed frames, which is exact only for small rotations of the joints.

One analysis takes about twice as long as with ``'plate_clpt_gl'`` (288 s
against 146 s on one core of a shared machine), from the 21 x 21
integration points of the skins instead of 15 x 15. With the integration
points of ``'plate_clpt_gl'``, setting ``nx = gauss_points(m)`` and ``ny =
gauss_points(n)`` on every domain, the tip deflections are the same to
0.01 mm at every load, in 138 s.
