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
The kernels :mod:`panels.models.plate_clpt_tl_num` are generated from a
symbolic derivation with SymPy of the strains below, with the helpers of
the generator of ``'plate_clpt_gl'``.

.. note::

    **Integration points: exact by default, a faster option.** By default
    the integrals of ``'plate_clpt_tl'`` are exact, with `3p` Gauss-Legendre
    points along each direction, 21 for ``m = 8``, against the `2p + 1`, 15,
    of ``'plate_clpt_gl'``, see `Internal forces and tangent stiffness`_. A
    tangent stiffness matrix then costs about 2.2 times that of
    ``'plate_clpt_gl'``. The points of ``'plate_clpt_gl'`` do not integrate
    this model exactly but are enough in practice, at about half the cost:

    .. code-block:: python

        from panels.shell import gauss_points

        for p in panels:  # every domain of model 'plate_clpt_tl'
            p.nx = gauss_points(p.m)
            p.ny = gauss_points(p.n)

    With them, the internal forces of a plate with displacement gradients of
    order 0.5 differ by 3e-13, and the tip deflections of the wing box of
    :ref:`plate_clpt_gl` under a follower tip force by less than 0.01 mm up
    to 500 N. Assigning ``None`` to
    ``nx`` and ``ny`` restores the exact default.


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
w_{,xx}`, of the order of the membrane strain times the curvature, so that
for moderate deflections the response differs from that of
``'plate_clpt_gl'`` by a relative amount of the order of the membrane
strain. For a rotation `\boldsymbol{X} \to \boldsymbol{R} \boldsymbol{X}`
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


Accuracy for large rotations
----------------------------

Benchmark 3.2 of [sze2004Benchmarks]_, a cantilever strip rolled into a
full circle by an end moment, has the exact solution of the inextensible
beam, an arc of curvature `M/EI`. ``'plate_clpt_tl'`` rolls the strip into
the circle within about `2 \times 10^{-3} L` of that solution; the
difference is that of the model, not of the Ritz approximation: with the
curvature measured by the area vector, `\kappa_{xx} = \lambda^2 \mu\,
\theta_{,x}` for the stretches `\lambda` and `\mu` along `x` and `y`, the
bending moment shortens the strip by terms of order `(\kappa h)^2`. The
angle `2 \pi` of the full circle requires polynomials of degree 11 or more
along the strip. ``'plate_clpt_gl'``, whose curvatures are measured in the
fixed frame, is too flexible from tip rotations of about 0.3 rad, and
``'plate_clpt_donnell'``, whose membrane strains are linear in `u`, never
rolls the strip. Under a tip force of constant direction a cantilever strip
bent out of plane follows the inextensible elastica, where
``'plate_clpt_gl'`` follows the linear beam, and bent in its plane, the
bending being carried by the membrane strains, it follows the elastica as
``'plate_clpt_gl'`` does.

In the wing box of :ref:`plate_clpt_gl`, below the buckling of the top skin,
the box bends through the membrane strains of the skins and the in-plane
bending of the spars, and ``'plate_clpt_tl'`` and ``'plate_clpt_gl'`` agree;
above it, ``'plate_clpt_tl'`` is a few per cent stiffer, since the buckles
of the top skin, rotated with the box, keep their bending energy instead of
losing the factor `\cos^2\theta`.
