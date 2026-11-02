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


Cantilever strip rolled into a circle by an end moment
------------------------------------------------------

Benchmark 3.2 of [sze2004Benchmarks]_ (Fig. 3a, p. 1556): a strip clamped
at one end, `L = 12`, `b = 1`, `h = 0.1`, `E = 1.2 \times 10^6`, `\nu =
0`, under an end moment `M` up to `M_{max} = 2 \pi M_o`, `M_o = EI/L =
25/3`, which rolls it into a full circle. The exact solution of the
inextensible beam (p. 1555) is the arc of curvature `M/EI`, with the tip at

.. math::

    \frac{U}{L} = \frac{\sin\theta}{\theta} - 1, \qquad
    \frac{W}{L} = \frac{1 - \cos\theta}{\theta}, \qquad
    \theta = \frac{M L}{E I} = 2 \pi \frac{M}{M_{max}}

tabulated as "Exact" in Table 3a (p. 1557). panels has no applied moments;
``tests/tests_shell/test_plate_clpt_tl_end_moment.py`` applies the moment
as a conservative load of potential `-M/b \int_0^b \theta(L, y)\, dy`,
`\theta = \operatorname{atan2}(w_{,x}, 1 + u_{,x})`, with its
configuration-dependent load vector and load stiffness passed to
``structsolve.Analysis`` through ``calc_fext(c=c)``, ``calc_fint(c, inc)``
and ``calc_kC(c, inc)``. With ``m = 12``, ``n = 4`` and 20 Newton-Raphson
increments of `0.05 M_{max}` (4 s), the tip displacements, averaged over
the width, are:

.. list-table::
    :header-rows: 1

    * - `M/M_{max}`
      - 0.05
      - 0.25
      - 0.5
      - 0.75
      - 1
    * - `-U`, exact, Table 3a
      - 0.196
      - 4.361
      - 12.000
      - 14.546
      - 12.000
    * - `-U`, ``'plate_clpt_tl'``
      - 0.1964
      - 4.3619
      - 12.0068
      - 14.5425
      - 11.9725
    * - `-U`, ``'plate_clpt_gl'``
      - 0.2260
      - stops
      - --
      - --
      - --
    * - `-U`, ``'plate_clpt_donnell'``
      - 0.2183
      - 4.4238
      - 7.6697
      - 10.1090
      - 12.1940
    * - `W`, exact, Table 3a
      - 1.870
      - 7.639
      - 7.639
      - 2.546
      - 0.000
    * - `W`, ``'plate_clpt_tl'``
      - 1.8695
      - 7.6398
      - 7.6342
      - 2.5272
      - 0.0002
    * - `W`, ``'plate_clpt_gl'``
      - 1.9994
      - stops
      - --
      - --
      - --
    * - `W`, ``'plate_clpt_donnell'``
      - 1.9821
      - 8.9234
      - 11.7495
      - 13.4893
      - 14.8152

``'plate_clpt_tl'`` rolls the strip into the circle, within 2.3e-3 `L` of
the exact solution (0.03 at the full circle). The difference is that of
the model, not of the Ritz approximation: the curvature measured with the
area vector, `\kappa_{xx} = \lambda^2 \mu\, \theta_{,x}` for the
stretches `\lambda` and `\mu` along `x` and `y`, makes the bending moment
shorten the strip and its width by `(\kappa h)^2/6` and `(\kappa h)^2/12`,
and the curvature larger than `M/EI` by about `5 (\kappa h)^2/6`, 0.23 %
at the full circle (`\kappa h = 0.052`). The uniform cylindrical solution
of these equations, computed in the test, is reproduced to 4e-6 `L`; the
largest difference with it over the five loads converges with the number
of terms `m` along the length as 2.3e-2 `L` (`m = 8`), 2.8e-4 `L` (10),
3.9e-6 `L` (12) and 3.7e-6 `L` (14 and 16), the angle `2 \pi` of the
full circle requiring polynomials of degree 11 or more.
``'plate_clpt_gl'``, whose curvatures `-w_{,xx}` are measured in the fixed
frame, is 7 % more flexible in `W` already at `0.05 M_{max}` (a tip
rotation of 0.31 rad), and its Newton-Raphson analysis stops at `0.079
M_{max}` (0.50 rad, `W = 4.19` against 2.93); ``'plate_clpt_donnell'``,
whose membrane strains are linear in `u`, never rolls the strip: `W` keeps
increasing, to 14.8 at `M_{max}`.


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
