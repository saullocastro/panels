.. _cohesive_zone:

Cohesive zone model of the multi-domain approach
================================================

The area connection ``'SB_TSL'`` of :meth:`.MultiDomain.get_kC_conn` connects
two skins over an area with a traction-separation law (TSL) at the interface,
such that the progressive delamination between them can be simulated. It
originates from the MSc thesis of D'Souza (2024) [nathan2024MSc]_,
*Multi-Domain Semi-Analytical Cohesive Zone Approach*. This page describes the
model as implemented and documents where the implementation departs from the
thesis. Equation numbers of the form "Eq. 5.12" refer to the thesis. Each
deviation states the problem, its consequence on the load-displacement curve
of the double cantilever beam (DCB), and what the implementation does instead.

The implementation is found in:

- the kernels :func:`.fkCSB11_dmg`, :func:`.fkCSB12_dmg` and
  :func:`.fkCSB22_dmg` of ``panels/multidomain/connections/kCSB_dmg.pyx``,
  and :func:`.calc_kw_tsl`;
- the rules of the discretisation of the cohesive domain,
  :func:`.tsl_resolution`, :func:`.tsl_margin` and
  :func:`.tsl_discretization`, see :ref:`cohesive_zone_discretisation`;
- the methods of :class:`.MultiDomain`: :meth:`.MultiDomain.get_kC_conn`,
  :meth:`.MultiDomain.calc_k_dmg`, :meth:`.MultiDomain.correct_separation`,
  :meth:`.MultiDomain.calc_kT_TSL`, :meth:`.MultiDomain.reaction_line_pd_xcte`
  and :meth:`.MultiDomain.force_out_plane_damage`;
- the nonlinear driver ``tests/multidomain/test_dcb_damage.py``.

The model of the thesis is the DCB of its Section 7.3: `L = 65` mm, `b = 25`
mm, `a_0 = 48` mm, arms `[0]_{15}` with ply thickness 0.14 mm of
AS4D/PEKK-FC, `G_{Ic} = 1.12` N/mm, three domains per arm, of which the first
one, with length `a_1 = L - a_0 = 17` mm, carries the cohesive zone. It is
the model of the sections below up to the solution procedure and of
``tests/multidomain/test_dcb_damage.py``. The rules of the discretisation
of :ref:`cohesive_zone_discretisation` are stated for a model in which the
cohesive domain is followed by a perfect bond.


Traction-separation law and damage
----------------------------------

The law implemented is bilinear, with one damage variable `d` that degrades
the three components of the traction:

.. math::
    :label: cz-law

    \tau_\alpha = k_o \left(1 - d\right) \Delta_\alpha, \qquad \alpha = u, v, w

where `k_o` is the penalty stiffness of the pristine interface (key ``'k_o'``
of the connection) and `\Delta_\alpha` are the separations of
Eq. :eq:`cz-jump`. The separation at damage onset is `\Delta_o = \tau_o/k_o`
and the separation at complete failure is `\Delta_f = 2 G_{Ic}/\tau_o`, with
the keys ``'tau_o'`` and ``'G1c'`` of the connection. The damage is driven by
the normal separation after the correction described in
:ref:`cohesive_zone_correction`, `\bar\Delta_w`, and it is irreversible:

.. math::
    :label: cz-damage

    d = \max\left( d^{h}, \hat d(\bar\Delta_w) \right), \qquad
    \hat d(\Delta) = \begin{cases}
        0 & \Delta \le \Delta_o \\
        \dfrac{\Delta_f (\Delta - \Delta_o)}{(\Delta_f - \Delta_o) \Delta} & \Delta_o < \Delta < \Delta_f \\
        1 & \Delta \ge \Delta_f
    \end{cases}

where `d^h` is the damage of the last converged increment, stored in
``MultiDomain.dmg_index`` by :meth:`.MultiDomain.update_TSL_history`. The
damage `\hat d` is computed by :func:`.calc_kw_tsl` and the maximum with the
history, together with the degraded stiffness `k_o (1 - d)` at each
integration point, by :meth:`.MultiDomain.calc_k_dmg`. The area under the
softening branch of the law is `G_{Ic}`.


.. _cohesive_zone_correction:

Correction of the separation field
----------------------------------

Section 6.3.1 of the thesis describes the correction of the separation used
to compute the damage as: start at the point of maximum separation, location
`A`, move in the direction of decreasing separation up to the first
non-positive separation, location `B`, and set the separation to zero beyond
`B`. Setting to zero, for each row `y =` cte, everything from `x = 0` up to
the *last* non-positive separation of the row would instead zero the whole
row, including the fracture process zone, whenever the approximation
functions give a non-positive value between `A` and the edge of the domain,
for instance a small oscillation at `x = a_1`.
:meth:`.MultiDomain.correct_separation` implements the description of the
thesis, for a crack front that advances along `-x`:

.. math::
    :label: cz-correction

    \Delta^{corr}_{i,j} = \begin{cases}
        0 & j \le B_i \\
        \Delta_{i,j} & j > B_i
    \end{cases}
    \qquad
    B_i = \max \left\{ j \le A_i : \Delta_{i,j} \le 0 \right\}, \quad A_i = \arg\max_j \Delta_{i,j}

with `i` the row and `j` the column of the grid of integration points, `j`
increasing with `x`. Rows without non-positive values between `x = 0` and
`A_i` are not modified, and neither are rows without any positive
separation. The negative values of the separation are not only numerical: for
a beam on an elastic foundation the separation behind the crack front
oscillates with a decaying amplitude, and its first negative lobe does not
change when the number of terms of the cohesive domain is increased from 10
to 20.

The corrected separation is used only to compute the damage. The tractions,
the internal force vector and the stiffness matrices use the full separation.


Separations at the interface
----------------------------

The compatibility of the area connection in the thesis, Eqs. 4.57--4.62,
assumes `w^i_{,x} = w^j_{,x}` and `w^i_{,y} = w^j_{,y}` (Eq. 4.61) to write
the in-plane compatibility with the slope of the top panel only, Eq. 4.62:

.. math::
    :label: cz-thesis-jump

    \Delta u^{thesis} = u^t + (d^t + d^b) \, w^t_{,x} - u^b

This holds for a perfect bond, but not inside the fracture process zone,
where the two arms rotate in opposite directions. There,
Eq. :eq:`cz-thesis-jump` gives a spurious tangential separation, penalised
with the same `k^w_{CZ}`, which stiffens the interface exactly where it should
open. The implementation uses the displacement jump between the two surfaces
in contact, with the slope of each panel:

.. math::
    :label: cz-jump

    \begin{aligned}
        \Delta_u &= u^t + d^t w^t_{,x} - u^b + d^b w^b_{,x} \\
        \Delta_v &= v^t + d^t w^t_{,y} - v^b + d^b w^b_{,y} \\
        \Delta_w &= w^t - w^b
    \end{aligned}

where `d^t` and `d^b` are the distances from the reference surfaces of the
top and bottom panels to the interface, `u^t, v^t, w^t` and `u^b, v^b, w^b`
the displacements of their reference surfaces. For cylindrical panels with
the Sanders-Koiter kinematics, `v(z) = v + z (v/r - w_{,y})`, and
`w_{,y}` is replaced by `w_{,y} - v/r` in `\Delta_v`, with the radius `r` of
the reference surface of each panel, see :meth:`.MultiDomain._tsl_operators`
and :mod:`panels.multidomain.connections.kCSB_dmg`. With the thicknesses `h^t`,
`h^b` and the offsets `d_t`, `d_b` of the laminates, ``Shell.offset``, each
laminate occupying `d - h/2 \le z \le d + h/2` in the coordinate `z` of its
panel, `d^t = h^t/2 - d_t` and `d^b = h^b/2 + d_b`, see
:func:`.kCsdt._sb_distances`; without offsets, as in the DCB examples, they
are `h^t/2` and `h^b/2`, the distances from the mid-planes. Top and bottom
refer to the `z` axis, normal to the reference surfaces and common to both
panels, pointing up in the DCB: the top panel
is the panel on the positive side of the interface along `z`, whose face `z =
-d^t` is at the interface, and it is ``p1`` of the connection dictionary; the
bottom panel is the one on the negative side, whose face `z = +d^b` is at the
interface, and it is ``p2``, whatever their order in the assembly, see
:meth:`.MultiDomain.get_kC_conn`. With
these three separations and the degraded stiffness `k^w_{CZ} = k_o (1 - d)`,
the penalty energy of Eq. 5.1 becomes:

.. math::
    :label: cz-energy

    U^{tb}_{pen-AR-dmg} = \frac{1}{2} \int_x \int_y k^w_{CZ} \left( \Delta_u^2 +
    \Delta_v^2 + \Delta_w^2 \right) dx\,dy

and the stiffness matrix of the connection is:

.. math::
    :label: cz-stiffness

    \boldsymbol{K}^{conn}_{dmg} = \int_A k^w_{CZ} \boldsymbol{B}_\Delta^\top \boldsymbol{B}_\Delta \, dA,
    \qquad
    U^{tb}_{pen-AR-dmg} = \frac{1}{2} \boldsymbol{c}^\top \boldsymbol{K}^{conn}_{dmg} \boldsymbol{c}

where `\boldsymbol{B}_\Delta` is the operator that gives the three separations of
Eq. :eq:`cz-jump` from the Ritz constants `\boldsymbol{c}`. For a frozen damage field,
the internal force of the connection is `\boldsymbol{K}^{conn}_{dmg}\boldsymbol{c}`, see
Eq. :eq:`cz-residual`. The integral is evaluated
with a Gauss-Legendre rule of ``nr_x_gauss`` `\times` ``nr_y_gauss`` points
over the domain of the top panel, and `k^w_{CZ}` is given at each of these
points. The terms of the kernels that are new or modified with respect to the
thesis are listed in the docstrings of :func:`.fkCSB11_dmg`,
:func:`.fkCSB12_dmg` and :func:`.fkCSB22_dmg`. The undamaged connection
``'SB'``, :func:`.fkCSB`, imposes the jump of Eq. :eq:`cz-jump` at zero
separation, such that the pristine ``'SB_TSL'``, with `k^w_{CZ} = k_o` at
every point, has the matrix of ``'SB'`` with `k_t = k_o`, as checked by
``tests/multidomain/test_sb_unified.py``.


Energy dissipated by crack formation
------------------------------------

The thesis adds the term `U_{crack}` of Eq. 5.18 to the total potential
energy, Eq. 5.29, with the force vector `\boldsymbol{F}_{crack}` of Eq. 5.28 and the
stiffness matrix `\boldsymbol{K}_{crack}` of Eq. 5.26. This counts the dissipated energy
twice. The degradation of the penalty stiffness `k^w_{CZ} = k_o(1-d)` inside
`\boldsymbol{K}^{conn}_{dmg}`, Eq. 5.5, already makes the traction follow the softening
branch of the traction-separation law, and the area under that branch is
`G_{Ic}`, exactly as in a finite element with a cohesive law. The
implementation has neither `U_{crack}`, `\boldsymbol{F}_{crack}` nor `\boldsymbol{K}_{crack}`, and
the residual is:

.. math::
    :label: cz-residual

    \boldsymbol{R} = \boldsymbol{F}_{int} - \boldsymbol{F}_{ext}, \qquad
    \boldsymbol{F}_{int} = \boldsymbol{F}^{panels}_{int} + \boldsymbol{K}^{conn}_{dmg}(d)\boldsymbol{c} + \boldsymbol{K}^{pen}_{PD}\boldsymbol{c}

where `\boldsymbol{F}^{panels}_{int}` is the internal force vector of the
panels, `d` is the damage at each integration point, Eq. :eq:`cz-damage`, the
largest value between the converged history and the value obtained from the
current separation, and `\boldsymbol{K}^{pen}_{PD}` is the penalty stiffness of the
prescribed displacement. The sudden increase of the slope at damage onset
reported in Section 7.1.4 of the thesis, and the need to drop
`\boldsymbol{K}^2_{crack}`, are consequences of the double counting.


Consistent tangent stiffness
----------------------------

Writing the separations as `\Delta_\alpha = \boldsymbol{B}_\alpha\boldsymbol{c}`, the internal
force vector of the connection and its derivative are:

.. math::
    :label: cz-fint-coh

    \boldsymbol{F}^{coh}_{int} = \int_A k_o (1 - d) \sum_\alpha \boldsymbol{B}_\alpha^\top \Delta_\alpha \, dA = \boldsymbol{K}^{conn}_{dmg}\boldsymbol{c}

.. math::
    :label: cz-tangent

    \frac{\partial \boldsymbol{F}^{coh}_{int}}{\partial \boldsymbol{c}} = \boldsymbol{K}^{conn}_{dmg} + \boldsymbol{K}^{conn}_{\dot d}, \qquad
    \boldsymbol{K}^{conn}_{\dot d} = - \int_A \chi \, k_o \, \hat d'(\Delta_w) \left( \sum_\alpha \boldsymbol{B}_\alpha^\top \Delta_\alpha \right) \boldsymbol{B}_w \, dA

with:

.. math::
    :label: cz-dprime

    \hat d'(\Delta) = \frac{\Delta_f \Delta_o}{(\Delta_f - \Delta_o) \Delta^2}, \qquad
    \chi = \begin{cases}
        1 & \Delta_o < \bar\Delta_w < \Delta_f \text{ and } \hat d(\bar\Delta_w) > d^h \\
        0 & \text{otherwise}
    \end{cases}

The thesis does not derive the tangent of the cohesive zone. With the secant
stiffness `\boldsymbol{K}^{conn}_{dmg}` alone, the iteration matrix misses the change of
the damage with the displacements. It overestimates the stiffness of every
softening point, and the Newton-Raphson iterations converge linearly, slowly,
once the crack grows: in the reference DCB, the first increment after the
first fully damaged point needed more than 130 iterations.

The mask `\chi` selects the points where the damage grows in the current
state. Where it is 1, the correction of :ref:`cohesive_zone_correction` leaves
the separation unchanged, `\bar\Delta_w = \Delta_w` and `\partial \bar\Delta_w
/ \partial \boldsymbol{c} = \boldsymbol{B}_w`. The derivative of the correction itself, a switch
between 0 and `\Delta_w`, is zero almost everywhere. Some properties of
Eq. :eq:`cz-tangent`:

- `\boldsymbol{K}^{conn}_{\dot d}` is not symmetric, because the tangential tractions
  depend on `\Delta_w` through `d`, while `d` does not depend on `\Delta_u`,
  `\Delta_v`. The term `\boldsymbol{B}_w^\top \Delta_w \boldsymbol{B}_w` is symmetric and negative
  semi-definite; it makes the normal tangent stiffness of a softening point
  `k_o (1 - d - \hat d' \Delta_w) = -k_o \Delta_o / (\Delta_f - \Delta_o)`,
  the slope of the descending branch of the bilinear law.
- The iteration matrix becomes `\boldsymbol{K}_C + \boldsymbol{K}_G + \boldsymbol{K}^{conn}_{dmg} +
  \boldsymbol{K}^{conn}_{\dot d} + \boldsymbol{K}^{pen}_{PD}`. Under displacement control the
  prescribed-displacement penalty keeps it regular on the softening branch of
  the DCB. ``structsolve.solve`` uses ``scipy.sparse.linalg.spsolve``, which
  accepts non-symmetric matrices.
- Like `\boldsymbol{K}^{conn}_{dmg}`, `\boldsymbol{K}^{conn}_{\dot d}` is evaluated at every
  iteration, while `\boldsymbol{K}_C + \boldsymbol{K}_G` is refreshed every `n_{NR}` iterations,
  see :ref:`cohesive_zone_solution`.

:meth:`.MultiDomain.calc_kT_TSL` evaluates `\boldsymbol{K}^{conn}_{\dot d}` at the
Gauss-Legendre points of the connection, with `\boldsymbol{B}_\alpha` built once per
panel from the field functions and cached. The nonlinear drivers use it
by default (``consistent_tangent=True``).


Assembly of the cohesive stiffness by matrix products
-----------------------------------------------------

The thesis assembles `\boldsymbol{K}^{conn}_{dmg}` with loops over the terms of both
panels and over the integration points, where the approximation functions are
evaluated inside the loops, ``kCSB_dmg.pyx``. Because `\boldsymbol{K}^{conn}_{dmg}` is
part of the internal force vector, it is needed at every iteration (see
:ref:`cohesive_zone_solution`), and with `60 \times 30` integration points and `10 \times 10`, `8 \times 8` terms these
loops dominate the run time in the elastic range.
The implementation writes the same Gauss-Legendre rule as matrix products.
Let `g = 1, \dots, n_g` be the integration points, with weights `w_g = w_{\xi}
w_{\eta} \, ab/4`, and `\boldsymbol{c}_{tb}` the Ritz constants of the top panel
followed by those of the bottom panel. The operators of Eq. :eq:`cz-jump`
evaluated at all points are the `n_g \times n_{dof}` matrices:

.. math::
    :label: cz-operators

    \begin{aligned}
        \boldsymbol{B}_u &= \left[ \boldsymbol{N}^t_u + d^t \boldsymbol{N}^t_{w,x} \;\; -\boldsymbol{N}^b_u + d^b \boldsymbol{N}^b_{w,x} \right] \\
        \boldsymbol{B}_v &= \left[ \boldsymbol{N}^t_v + d^t \boldsymbol{N}^t_{w,y} \;\; -\boldsymbol{N}^b_v + d^b \boldsymbol{N}^b_{w,y} \right] \\
        \boldsymbol{B}_w &= \left[ \boldsymbol{N}^t_w \;\; -\boldsymbol{N}^b_w \right]
    \end{aligned}

where row `g` of `\boldsymbol{N}^t_u` holds the approximation functions of `u` of the
top panel at point `g`, and so on. The panel functions are the ones of the
field evaluation, ``fuvw``, obtained by evaluating the field for each unit
vector of Ritz constants. With `\text{diag}(\cdot)` the diagonal matrix of a
vector over the points, the secant stiffness is:

.. math::
    :label: cz-K-products

    \boldsymbol{K}^{conn}_{dmg} = \sum_{\alpha = u, v, w} \boldsymbol{B}_\alpha^\top \text{diag}\left( w_g k_g \right) \boldsymbol{B}_\alpha, \qquad k_g = k_o (1 - d_g)

a product that runs on optimised BLAS routines. The integration rule is the
same as in the loops, so the result is the same up to round-off. The
operators depend only on the geometry and on the approximation functions, and
they are computed once and cached. The consistent tangent of
Eq. :eq:`cz-tangent` uses the same operators, restricted to the rows where
`\chi = 1`. Only the upper triangle of Eq. :eq:`cz-K-products` is added to
the connection matrix, which is made symmetric afterwards together with the
other connections.

**Update restricted to the damaged points.** Splitting `k_g = k_o - (k_o -
k_g)` in Eq. :eq:`cz-K-products`:

.. math::
    :label: cz-K-update

    \boldsymbol{K}^{conn}_{dmg} = \boldsymbol{K}^{conn}_{o} - \sum_{\alpha = u, v, w} \boldsymbol{B}_{\alpha,\mathcal{D}}^\top \, \text{diag}\left( w_g (k_o - k_g) \right)_{\mathcal{D}} \, \boldsymbol{B}_{\alpha,\mathcal{D}}, \qquad
    \boldsymbol{K}^{conn}_{o} = k_o \sum_{\alpha = u, v, w} \boldsymbol{B}_\alpha^\top \text{diag}\left( w_g \right) \boldsymbol{B}_\alpha

where `\mathcal{D} = \{ g : k_g \neq k_o \}` is the set of damaged points and
the subscript `\mathcal{D}` keeps only their rows. `\boldsymbol{K}^{conn}_{o}`, the
stiffness of the pristine interface, is computed once for each `k_o` and
cached. Before damage onset `\mathcal{D}` is empty and the assembly reduces
to a copy of `\boldsymbol{K}^{conn}_{o}`. After onset, `\mathcal{D}` holds the fracture
process zone and the cracked region, and the cost of the correction grows
with the number of their points, not with the size of the whole interface.
The correction is exact, not an approximation: it is Eq. :eq:`cz-K-products`
regrouped. Because `k_g` is taken from the stiffness field used by the
kernels, the approach does not depend on how the damage is computed.

The loops of ``kCSB_dmg.pyx`` are used only when the connection has
``use_kernels=True``. Both assemblies integrate over the domain of the top
panel and evaluate the functions of the bottom panel at the natural
coordinates of the top panel, therefore the two panels must have the same
dimensions `a`, `b`, which :class:`.MultiDomain` checks for the connections
``'SB'`` and ``'SB_TSL'``, raising a ``ValueError`` otherwise.

Written as matrix products, one assembly of `\boldsymbol{K}^{conn}_{dmg}` on the DCB model
of the thesis is one to two orders of magnitude faster than with the loops over
the terms and the integration points.


.. _cohesive_zone_load:

Load measured at the loaded edge
--------------------------------

The thesis evaluates the load with two approaches, Section 5.4, and scales
the second one, the area integral of the traction, Eq. 5.38, with the ratio
between both at the first increment, Section 7.1.4. The area integral was
computed with the corrected separation of Section 6.3.1, in which the
separation behind the crack front is set to zero. This removes the
compressive tractions that balance the tensile ones. For a beam on an elastic
foundation the separation behind the peak is physically negative, and its
integral is not small: in a linear solution of the DCB of the thesis the
integral of the traction computed with the corrected separation is more than
an order of magnitude larger than the reaction at the loaded edge. Their
ratio depends on `k_o` and on the damage state, such that a scaling factor
fixed at the first increment distorts the curve after damage onset. A line
integral of the transverse shear force `Q_x` of the loaded arm, evaluated
from the derivatives of the moments of the Ritz field, is a third measure,
less accurate than the reaction. The implementation provides:

- :meth:`.MultiDomain.reaction_line_pd_xcte`, the reaction of the prescribed
  displacement `w_p(y)` imposed with the penalty `k_w` along `x = x_p`:

  .. math::
      :label: cz-reaction

      R = \int_0^b k_w \left( w_p(y) - w(x_p, y) \right) dy

  This is what the load cell measures, and it is used for the
  load-displacement curves.
- :meth:`.MultiDomain.force_out_plane_damage` integrates `\tau = k^w_{CZ}
  \Delta` with the full separation `\Delta`, the same that enters
  Eq. :eq:`cz-residual`. It matches `R` by equilibrium of the loaded arm and
  needs no scaling. The corrected separation is used only to compute the
  damage.

When the cohesive domain carries the whole load, the difference between both
measures is a check of the equilibrium of the solution, and the drivers
report it at every increment. It grows when the basis does not represent the
tractions, or when the tractions ahead of the process zone, its compressive
lobe and the decaying opening beyond it, reach a perfect bond beyond the
cohesive domain, whose tractions are not part of the integral.


Penalty stiffness of the edge connections
-----------------------------------------

The default penalties of :func:`.calc_kt_kr` for an ``'SSxcte'`` connection
between two equal laminates are `k_t = A_{11}/h` and `k_r = D_{11}/h`, from
Castro and Donadon (2017) [castro2017Multidomain]_. With the penalty energy
`\frac{1}{2} k_r (w^i_{,x} - w^j_{,x})^2` per unit width, see
:mod:`panels.multidomain.connections`, the connection behaves as a
rotational spring of stiffness `k_r` per unit width, which adds to a
cantilever loaded at the tip a compliance equal to `3 D_{11}/(k_r a) = 3h/a`
of its bending compliance, about 13% for each connection of the DCB arm (`h
= 2.1` mm, `a = 48` mm). The arms of the reference model have two
connections each, at the crack tip moment, so that the elastic stiffness of
the DCB with the default penalties is lower than that of the beam on an
elastic foundation of Kanninen [kanninen1973AugmentedDCB]_, which the model
approaches when the penalties grow, and the separation at the crack front,
and therefore the damage onset, lags the imposed displacement. The drivers
multiply the default ``'SSxcte'`` penalties by ``edge_penalty_factor``, 100
by default. The library defaults are unchanged.



.. _cohesive_zone_solution:

Solution procedure
------------------

The nonlinear drivers solve Eq. :eq:`cz-residual` under displacement
control with the modified Newton-Raphson method of Section 5.3 of the thesis.
The changes described below modify the iterations but, except for the
convergence criterion, not the equations being solved.

**Secant stiffness of the cohesive zone at every iteration.** The driver of
the thesis refreshed `\boldsymbol{K}^{conn}_{dmg}` only every `n_{NR} = 3` iterations,
together with the tangent stiffness matrix. However, `\boldsymbol{K}^{conn}_{dmg}` is
the secant stiffness of the cohesive zone and it is part of the internal force
vector, Eq. :eq:`cz-residual`. Keeping it frozen makes the residual
inconsistent with the traction-separation law, and a converged state may
carry a damage field computed with the displacements of earlier iterations.
The drivers evaluate `\boldsymbol{K}^{conn}_{dmg}` at every iteration, and only the
tangent stiffness of the panels, `\boldsymbol{K}_C + \boldsymbol{K}_G`, is refreshed every `n_{NR}`
iterations. The iteration matrix of Section 5.3 of the thesis is `\boldsymbol{K}_C +
\boldsymbol{K}_G + \boldsymbol{K}^{conn}_{dmg} + \boldsymbol{K}^{pen}_{PD}`, where the secant stiffness of the
cohesive zone replaces its tangent; the drivers add the damage-rate term
`\boldsymbol{K}^{conn}_{\dot d}` of Eq. :eq:`cz-tangent`.

**Predictor.** Each increment started from the last converged state. Under
displacement control the first residual is then dominated by the jump of the
prescribed displacement, and the first iteration always ended with a ratio of
Eq. :eq:`cz-criterion` close to 1. The increment `n` starts from the
linear extrapolation of the last two converged states:

.. math::

    \boldsymbol{c}_{n}^{0} = \boldsymbol{c}_{n-1} + \frac{w_{p,n} - w_{p,n-1}}{w_{p,n-1} - w_{p,n-2}} \left( \boldsymbol{c}_{n-1} - \boldsymbol{c}_{n-2} \right)

with the state `w_p = 0`, `\boldsymbol{c} = \boldsymbol{0}` used for the second increment
(``predictor=True``). The damage history is only updated after convergence,
so the damage of the predicted state is not stored.

**Reuse of** `\boldsymbol{K}_C + \boldsymbol{K}_G` **in the elastic range.** While no point of the
interface is damaged, the tangent stiffness of the panels is reused for up to
``kT_pan_reuse_steps`` increments, 5 by default, and within an increment it
is refreshed every `n_{NR}` iterations, but not after the first one. The
residual is always evaluated exactly, so the converged solution does not
change; only the number of iterations can increase. After onset, `\boldsymbol{K}_C +
\boldsymbol{K}_G` is evaluated at the start of every increment.

**Why Newton-Raphson fails after the peak.** The residual is continuous but
only piecewise smooth. Its derivative jumps whenever an integration point
crosses `\Delta_o` (from `k_o` to the negative softening slope `-k_o
\Delta_o/(\Delta_f - \Delta_o)`), crosses `\Delta_f` (from the softening slope
to zero), switches between loading (`\chi = 1`) and unloading (`\hat d \le
d^h`), or changes sign and enters or leaves the region zeroed by the
correction of Eq. :eq:`cz-correction`. The consistent tangent of
Eq. :eq:`cz-tangent` is exact only on one side of each kink. Before the peak,
few points cross a kink in one iteration and the quadratic convergence is not
affected. After the peak, the Ritz functions are global, so a Newton
correction moves the whole crack front at once, and many points can cross a
kink in one iteration when the integration points are sparse compared with
the cohesive zone. The iterations without the safeguards below may then
enter an exact cycle at the first openings after the peak, with a period
that is a multiple of the refresh interval `n_{NR}` of `\boldsymbol{K}_C +
\boldsymbol{K}_G`, which no number of iterations can break. Two safeguards
are added to the drivers. Neither changes the equations or the convergence
criterion, so a converged state is the same solution as before, only the
path to it changes.

**Backtracking line search.** With `\delta \boldsymbol{c}` the solution of
`\boldsymbol{K}\delta \boldsymbol{c} = -\boldsymbol{R}(\boldsymbol{c}_i)`, where `\boldsymbol{K}` is the iteration matrix, the
new iterate is `\boldsymbol{c}_{i+1} = \boldsymbol{c}_i + s\delta \boldsymbol{c}`, with the step length
`s` taken as the first of `s = 1, \frac{1}{2}, \frac{1}{4}, \dots,
2^{-n_{ls}}` that satisfies:

.. math::
    :label: cz-armijo

    \left\| \boldsymbol{R}(\boldsymbol{c}_i + s\delta \boldsymbol{c}) \right\|_{\boldsymbol{d}} \le \left(1 - 10^{-4} s\right) \left\| \boldsymbol{R}(\boldsymbol{c}_i) \right\|_{\boldsymbol{d}}, \qquad
    \left\| \boldsymbol{v} \right\|_{\boldsymbol{d}} = \sqrt{\boldsymbol{v}^\top \mathrm{diag}(|\boldsymbol{d}|)^{-1} \boldsymbol{v}}

where the entries `d_j` of the diagonal scaling vector `\boldsymbol{d}`, not
to be confused with the damage `d`, are the diagonal of `\boldsymbol{K}` at the start
of the increment, the scaling of the criterion of Eq. :eq:`cz-criterion`,
which balances the rows of the prescribed-displacement penalty against the
others. The merit function is the residual norm and not an energy, because
the damaged interface has no potential and `\boldsymbol{K}` is not symmetric. When `\boldsymbol{K}`
is the tangent stiffness matrix `\boldsymbol{K}_T = \partial \boldsymbol{R}/\partial
\boldsymbol{c}`, the exact Jacobian of the residual,
`\delta \boldsymbol{c}` is a descent direction of this merit function for any
positive weighting:

.. math::

    \left. \frac{d}{ds} \frac{1}{2} \left\| \boldsymbol{R}(\boldsymbol{c}_i + s\delta \boldsymbol{c}) \right\|_{\boldsymbol{d}}^2 \right|_{s=0} = \boldsymbol{R}^\top \mathrm{diag}(|\boldsymbol{d}|)^{-1} \boldsymbol{K}_T \delta \boldsymbol{c} = - \boldsymbol{R}^\top \mathrm{diag}(|\boldsymbol{d}|)^{-1} \boldsymbol{R} < 0

so a small enough `s` always satisfies Eq. :eq:`cz-armijo`. Between refreshes
of `\boldsymbol{K}_C + \boldsymbol{K}_G` the iteration matrix is not the exact Jacobian and this is
not guaranteed; if no trial satisfies Eq. :eq:`cz-armijo`, the trial with the
smallest norm is taken, a trial whose residual is not finite counting as an
infinite norm. Halving `s` halves the jump of the separations and
thus reduces the number of kinks crossed per iteration. Each trial costs one
evaluation of the residual, which is cheap with the assembly by matrix
products; the trial `s = 1` is the residual that the iteration needs anyway.
The drivers use `n_{ls} = 6` (``line_search_max``).

**Bisection of increments.** If an increment from `w_{p,n-1}` to `w_{p,n}`
does not converge within `n_{max} = 50` iterations (``max_NR_iter``), or
diverges, that is, the ratio of Eq. :eq:`cz-criterion` is not finite or
exceeds `10^3` after any iteration, its iterates are discarded and the first
half, from `w_{p,n-1}` to
`(w_{p,n-1} + w_{p,n})/2`, is solved from the last converged state. This is
applied recursively with a stack of pending targets: when a sub-increment
converges, the driver tries again to reach the next pending target, and when
it fails, the remaining interval is halved again, down to `2^{-6}` of the
original increment (``max_bisections``), after which the analysis is aborted.
Discarding the iterates is consistent because the damage history `d^h` is
only updated with converged states. Every converged sub-increment updates
`d^h` and the states used by the predictor, so a bisected increment follows a
finer loading path, which matters for an irreversible law; the results are
stored only at the prescribed displacements of the original list. The list of
prescribed displacements itself does not change: sub-increments are only
created when an increment fails.

**Effect of the consistent tangent and of the safeguards.** With the
consistent tangent, every opening of a DCB discretised by the rules of
:ref:`cohesive_zone_discretisation` converges in a few iterations, without
line search or bisection. With the secant stiffness as the only tangent of
the interface, the number of iterations grows as the process zone develops
and the iterations after the peak converge only linearly. The line search
breaks the cycles of the coarse integration rules, and the bisection is the
safeguard that completes their analyses.

**Convergence criterion.** The prescribed displacement is imposed with `k_w =
10^6`, such that `\boldsymbol{F}_{ext}` and `\boldsymbol{K}^{pen}_{PD}\boldsymbol{c}` are of the order of
`10^6`--`10^7`, while the physical forces are of the order of `10^2` N. The
denominator of Eq. 5.35 of the thesis, `\max(F(\boldsymbol{F}_{ext}),
F(\boldsymbol{F}_{int}))`, is dominated by the penalty terms, and the tolerance
`10^{-4}` is then not a relative tolerance on the physical forces. The
drivers use:

.. math::
    :label: cz-criterion

    \frac{F(\boldsymbol{R})}{\max\left( F(\boldsymbol{F}_{int} - \boldsymbol{K}^{pen}_{PD}\boldsymbol{c}), F(\boldsymbol{K}^{pen}_{PD}\boldsymbol{c} - \boldsymbol{F}_{ext}) \right)} < \varepsilon

where `F(\cdot) = \|\cdot\|_{\boldsymbol{d}}`, the first term of the
denominator is the internal force vector of the panels and of the
connections, and the second term is the
reaction of the prescribed displacement. The large `k_w` also degrades the
accuracy of the first solution of each increment, whose ratio of
Eq. :eq:`cz-criterion` can be close to 1. An increment is therefore declared
divergent only when the ratio is not finite or exceeds `10^3`, instead of
when it reaches 1. A divergent increment does not abort the analysis: it is
bisected as described above, and the analysis is aborted only when a
sub-increment of `2^{-6}` of the original increment (``max_bisections``)
still does not converge.


.. _cohesive_zone_discretisation:

Discretisation of the cohesive domain
-------------------------------------

The cohesive domain needs a Ritz basis and an integration rule that resolve
two lengths. Ahead of the process zone the opening `w^t - w^b` of two arms
connected by an elastic foundation of stiffness `k_o` decays as
`e^{-\lambda x}`, the solution of beams on an elastic foundation
[kanninen1973AugmentedDCB]_, with the rate:

.. math::
    :label: cz-lambda

    \lambda = \left( \frac{k_o}{2 D_{11}} \right)^{1/4}

and the cohesive zone, where the traction softens, has the length estimated
with Rice's model (Turon et al. 2007 [turon2007MeshSize]_):

.. math::
    :label: cz-lcz

    l_{cz} = 0.88\,\frac{E_2 G_{Ic}}{\tau_o^2}

The Ritz basis of the cohesive domain, `m_{tsl}` terms along `x`, sets the
shape of the response after the peak. With too few terms per decay length
the basis cannot follow the process zone continuously, like a cohesive finite
element mesh that is coarse compared with the cohesive zone
[alfano2001Interface]_ [turon2007MeshSize]_: the elastic energy builds up
while the crack front is held and is released when a group of integration
points fails at once, a discrete snap-back that a displacement-controlled
analysis crosses as a sudden load drop, and the load-displacement curve
becomes a sawtooth. More terms refine the displacement field and remove the
sawtooth. The Gauss-Legendre points along `x`, `n_x`, set the robustness and
a floor of the error: when the crack front crosses several integration
points in one increment, the iterations increase and increments need
bisection, but the integration points refine the damage field without
changing the shape of the curve. The terms along `y`, the Gauss points along
`y` and the terms of the other domains matter little for a specimen that is
uniform across the width. The cohesive domain must contain the crack growth,
the process zone and the region ahead of it where the opening decays: when
the compressive lobe ahead of the process zone reaches the perfect bond
beyond the cohesive domain, the load hardly changes, but the integral of the
traction no longer balances the reaction, see :ref:`cohesive_zone_load`.

A convergence study on a DCB of AS4D/PEKK-FC (Tijs et al. 2022
[tijs2022Interlaminar]_) gives the rules:

.. math::
    :label: cz-rules

    m_{tsl} = 1.25\,\lambda L_{tsl}, \qquad
    n_x = 10\,\frac{L_{tsl}}{l_{cz}}, \qquad
    L_{tsl} \geq \Delta a + 3 l_{cz} + \frac{8}{\lambda}

with `\Delta a` the crack growth up to the last opening, `m_{tsl} \leq 25`,
`n_x \geq 4 m_{tsl}` rounded up to a multiple of 10, `n_{tsl} = 10`, `n_y =
30` and `m = n = 8` in the other domains. The margin `8/\lambda` lets the
opening decay to below `e^{-8}` of its value at the end of the process zone.
The Bardell functions go up to 30 terms, so a
cohesive domain longer than about `30/\lambda` is not resolved; a lower
`k_o` gives a longer decay length `1/\lambda` and fewer terms. The rules are
implemented in :func:`.tsl_resolution`, :func:`.tsl_margin` and
:func:`.tsl_discretization`, and :meth:`.MultiDomain.calc_k_dmg` prints a
warning, the first time a point is damaged, when `m_{tsl} < 1.0\,\lambda
L_{tsl}` or `n_x < 7\,L_{tsl}/l_{cz}`. The Gauss-Legendre points and weights
are computed by :func:`scipy.special.roots_legendre`, without an upper limit
for `n_x`.

The model has been compared with the mode I DCB results of Alfano and
Crisfield (2001) [alfano2001Interface]_, Camanho et al. (2003)
[camanho2003Delamination]_, Turon et al. (2007) [turon2007MeshSize]_,
Krueger (2008) [krueger2008DCB]_, Tijs et al. (2022)
[tijs2022Interlaminar]_, Leciñana et al. (2023) [lecinana2023Fatigue]_ and
Tijs (2023) [tijs2023PhD]_, and with the mixed-mode benchmarks of Krueger
(2012) [krueger2012MMB]_, which ``'SB_TSL'`` cannot represent, since its
damage is driven by the normal separation only. For the unidirectional
specimens the peak loads are within 2.6% of the corrected beam theory or of
a cohesive finite element model with the same law, while the initial
compliance is 15-21% lower than that of the corrected beam theory,
because the CLPT arms have no transverse shear deformation.


.. _cohesive_zone_limits:

Limits of validity
------------------

The following limits come from the model itself, not from its
discretisation:

- **Bilinear law driven by the normal separation.** The law is bilinear,
  :func:`.calc_kw_tsl` accepts no other ``tsl_type``, and the damage of
  Eq. :eq:`cz-damage` is computed from the corrected normal separation
  `\bar\Delta_w` only; the tangential separations are degraded by the same
  damage but do not drive it. Mixed-mode delamination and laws with more
  than one softening branch, such as those that represent an R-curve from
  fibre bridging, are not represented.
- **No interpenetration stiffness.** The stiffness is `k_o (1 - d)` for
  separations of either sign. The interpenetration stiffness of Eq. 5.12 of
  the thesis is not implemented, so a fully damaged point offers no
  resistance when the faces of the crack close. Under an increasing opening
  the damaged points of a DCB have a positive separation and this does not
  matter; it does when the faces of a damaged region come into contact.
- **One cohesive interface per assembly.** The damage history `d^h` is
  stored once per assembly, in ``MultiDomain.dmg_index``, and is read by
  every ``'SB_TSL'`` connection, which restricts an assembly to one
  ``'SB_TSL'`` connection.
- **One crack front along** `-x`. The correction of the separation,
  Eq. :eq:`cz-correction`, assumes in each row `y =` cte a single crack front
  that advances along `-x`.
- **Classical laminated plate theory.** ``'SB_TSL'`` is available only for
  the models of the classical laminated plate theory; the models based on
  shear deformation theories raise a ``NotImplementedError``. The arms have
  no transverse shear deformation, so the compliance of arms with a low
  transverse shear stiffness is underestimated, while the peak loads,
  controlled by the fracture toughness, are affected much less.
- **Matching domains.** The two panels of the connection must have the
  same dimensions `a`, `b`, since the separations are evaluated at the
  integration points of the domain of the top panel.
