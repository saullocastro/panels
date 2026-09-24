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
of the double cantilever beam (DCB), and what the implementation does now.

The implementation is found in:

- the kernels :func:`.fkCSB11_dmg`, :func:`.fkCSB12_dmg` and
  :func:`.fkCSB22_dmg` of ``panels/multidomain/connections/kCSB_dmg.pyx``,
  and :func:`.calc_kw_tsl`;
- the methods of :class:`.MultiDomain`: :meth:`.MultiDomain.get_kC_conn`,
  :meth:`.MultiDomain.calc_k_dmg`, :meth:`.MultiDomain.correct_separation`,
  :meth:`.MultiDomain.calc_kT_TSL`, :meth:`.MultiDomain.reaction_line_pd_xcte`
  and :meth:`.MultiDomain.force_out_plane_damage`;
- the nonlinear driver ``tests/multidomain/test_dcb_damage.py`` and its
  counterpart for the validation notebooks, ``notebooks/dcb_utils.py``.

The reference case is the DCB of Section 7.3 of the thesis: `L = 65` mm,
`b = 25` mm, `a_0 = 48` mm, arms `[0]_{15}` with ply thickness 0.14 mm of
AS4D/PEKK-FC, `G_{Ic} = 1.12` N/mm, three domains per arm, of which the first
one, with length `a_1 = L - a_0 = 17` mm, carries the cohesive zone.


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
`B`. The previous implementation set to zero, for each row `y =` cte,
everything from `x = 0` up to the *last* non-positive separation of the row.
When the approximation functions give a non-positive value between `A` and
the edge of the domain, for instance a small oscillation at `x = a_1`, the
whole row was set to zero, including the fracture process zone.
:meth:`.MultiDomain.correct_separation` now implements the description of the
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

where `d^t` and `d^b` are the distances from the mid-planes of the top and
bottom panels to the interface, `u^t, v^t, w^t` and `u^b, v^b, w^b` their
mid-plane displacements. Top and bottom refer to the `z` axis, normal to the
mid-surfaces and common to both panels, pointing up in the DCB: the top panel
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

    [K^{conn}_{dmg}] = \int_A k^w_{CZ} [B_\Delta]^\top [B_\Delta] \, dA,
    \qquad
    U^{tb}_{pen-AR-dmg} = \frac{1}{2} \{c\}^\top [K^{conn}_{dmg}] \{c\}

where `[B_\Delta]` is the operator that gives the three separations of
Eq. :eq:`cz-jump` from the Ritz constants `\{c\}`. For a frozen damage field,
the internal force of the connection is `[K^{conn}_{dmg}]\{c\}`, see
Eq. :eq:`cz-residual`. The integral is evaluated
with a Gauss-Legendre rule of ``nr_x_gauss`` `\times` ``nr_y_gauss`` points
over the domain of the top panel, and `k^w_{CZ}` is given at each of these
points. The terms of the kernels that are new or modified with respect to the
thesis are listed in the docstrings of :func:`.fkCSB11_dmg`,
:func:`.fkCSB12_dmg` and :func:`.fkCSB22_dmg`. The kernel of the undamaged
connection ``'SB'``, ``kCSB.pyx``, keeps Eq. :eq:`cz-thesis-jump`, which is
adequate for a perfect bond.

The same kernel had the approximation functions in `\eta` of the top panel in
the coupling block built with the flags of the edge `y_1` in place of `y_2`.
This had no effect on the DCB, where all `y` flags are 1.


Energy dissipated by crack formation
------------------------------------

The thesis adds the term `U_{crack}` of Eq. 5.18 to the total potential
energy, Eq. 5.29, with the force vector `\{f_{crack}\}` of Eq. 5.28 and the
stiffness matrix `[K_{crack}]` of Eq. 5.26. This counts the dissipated energy
twice. The degradation of the penalty stiffness `k^w_{CZ} = k_o(1-d)` inside
`[K^{conn}_{dmg}]`, Eq. 5.5, already makes the traction follow the softening
branch of the traction-separation law, and the area under that branch is
`G_{Ic}`, exactly as in a finite element with a cohesive law. The
implementation has neither `U_{crack}`, `\{f_{crack}\}` nor `[K_{crack}]`, and
the residual is:

.. math::
    :label: cz-residual

    \{R\} = \{f_{int}\}_{panels} + [K^{conn}_{dmg}(d)]\{c\} + [K^{pen}_{PD}]\{c\} - \{f_{ext}\}

where `d` is the damage at each integration point, Eq. :eq:`cz-damage`, the
largest value between the converged history and the value obtained from the
current separation, and `[K^{pen}_{PD}]` is the penalty stiffness of the
prescribed displacement. The sudden increase of the slope at damage onset
reported in Section 7.1.4 of the thesis, and the need to drop
`[K^2_{crack}]`, are consequences of the double counting.


Consistent tangent stiffness
----------------------------

Writing the separations as `\Delta_\alpha = [B_\alpha]\{c\}`, the internal
force vector of the connection and its derivative are:

.. math::
    :label: cz-fint-coh

    \{f^{coh}_{int}\} = \int_A k_o (1 - d) \sum_\alpha [B_\alpha]^\top \Delta_\alpha \, dA = [K^{conn}_{dmg}]\{c\}

.. math::
    :label: cz-tangent

    \frac{\partial \{f^{coh}_{int}\}}{\partial \{c\}} = [K^{conn}_{dmg}] + [K^{conn}_{\dot d}], \qquad
    [K^{conn}_{\dot d}] = - \int_A \chi \, k_o \, \hat d'(\Delta_w) \left( \sum_\alpha [B_\alpha]^\top \Delta_\alpha \right) [B_w] \, dA

with:

.. math::
    :label: cz-dprime

    \hat d'(\Delta) = \frac{\Delta_f \Delta_o}{(\Delta_f - \Delta_o) \Delta^2}, \qquad
    \chi = \begin{cases}
        1 & \Delta_o < \bar\Delta_w < \Delta_f \text{ and } \hat d(\bar\Delta_w) > d^h \\
        0 & \text{otherwise}
    \end{cases}

The thesis does not derive the tangent of the cohesive zone. With the secant
stiffness `[K^{conn}_{dmg}]` alone, the iteration matrix misses the change of
the damage with the displacements. It overestimates the stiffness of every
softening point, and the Newton-Raphson iterations converge linearly, slowly,
once the crack grows: in the reference DCB, the first increment after the
first fully damaged point needed more than 130 iterations.

The mask `\chi` selects the points where the damage grows in the current
state. Where it is 1, the correction of :ref:`cohesive_zone_correction` leaves
the separation unchanged, `\bar\Delta_w = \Delta_w` and `\partial \bar\Delta_w
/ \partial \{c\} = [B_w]`. The derivative of the correction itself, a switch
between 0 and `\Delta_w`, is zero almost everywhere. Some properties of
Eq. :eq:`cz-tangent`:

- `[K^{conn}_{\dot d}]` is not symmetric, because the tangential tractions
  depend on `\Delta_w` through `d`, while `d` does not depend on `\Delta_u`,
  `\Delta_v`. The term `[B_w]^\top \Delta_w [B_w]` is symmetric and negative
  semi-definite; it makes the normal tangent stiffness of a softening point
  `k_o (1 - d - \hat d' \Delta_w) = -k_o \Delta_o / (\Delta_f - \Delta_o)`,
  the slope of the descending branch of the bilinear law.
- The iteration matrix becomes `[K_C] + [K_G] + [K^{conn}_{dmg}] +
  [K^{conn}_{\dot d}] + [K^{pen}_{PD}]`. Under displacement control the
  prescribed-displacement penalty keeps it regular on the softening branch of
  the DCB. ``structsolve.solve`` uses ``scipy.sparse.linalg.spsolve``, which
  accepts non-symmetric matrices.
- Like `[K^{conn}_{dmg}]`, `[K^{conn}_{\dot d}]` is evaluated at every
  iteration, while `[K_C] + [K_G]` is refreshed every `n_{NR}` iterations,
  see :ref:`cohesive_zone_solution`.

:meth:`.MultiDomain.calc_kT_TSL` evaluates `[K^{conn}_{\dot d}]` at the
Gauss-Legendre points of the connection, with `[B_\alpha]` built once per
panel from the field functions and cached. The drivers use it by default
(``consistent_tangent=True`` in ``test_dcb_damage.py``, always in
``dcb_utils.solve_dcb``).


Assembly of the cohesive stiffness by matrix products
-----------------------------------------------------

The thesis assembles `[K^{conn}_{dmg}]` with loops over the terms of both
panels and over the integration points, where the approximation functions are
evaluated inside the loops, ``kCSB_dmg.pyx``. Because `[K^{conn}_{dmg}]` is
part of the internal force vector, it is needed at every iteration (see
:ref:`cohesive_zone_solution`), and with `60 \times 30` integration points and `10 \times 10`, `8 \times 8` terms these
loops took about 11 s per assembly, 77% of the run time in the elastic range.
The implementation now writes the same Gauss-Legendre rule as matrix products.
Let `g = 1, \dots, n_g` be the integration points, with weights `w_g = w_{\xi}
w_{\eta} \, ab/4`, and `\{c_{tb}\}` the Ritz constants of the top panel
followed by those of the bottom panel. The operators of Eq. :eq:`cz-jump`
evaluated at all points are the `n_g \times n_{dof}` matrices:

.. math::
    :label: cz-operators

    \begin{aligned}
        {}[B_u] &= \left[ [N^t_u] + d^t [N^t_{w,x}] \;\; -[N^b_u] + d^b [N^b_{w,x}] \right] \\
        [B_v] &= \left[ [N^t_v] + d^t [N^t_{w,y}] \;\; -[N^b_v] + d^b [N^b_{w,y}] \right] \\
        [B_w] &= \left[ [N^t_w] \;\; -[N^b_w] \right]
    \end{aligned}

where row `g` of `[N^t_u]` holds the approximation functions of `u` of the
top panel at point `g`, and so on. The panel functions are the ones of the
field evaluation, ``fuvw``, obtained by evaluating the field for each unit
vector of Ritz constants. With `\text{diag}(\cdot)` the diagonal matrix of a
vector over the points, the secant stiffness is:

.. math::
    :label: cz-K-products

    [K^{conn}_{dmg}] = \sum_{\alpha = u, v, w} [B_\alpha]^\top \text{diag}\left( w_g k_g \right) [B_\alpha], \qquad k_g = k_o (1 - d_g)

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

    [K^{conn}_{dmg}] = [K^{conn}_{o}] - \sum_{\alpha = u, v, w} [B_\alpha]^\top_{\mathcal{D}} \, \text{diag}\left( w_g (k_o - k_g) \right)_{\mathcal{D}} \, [B_\alpha]_{\mathcal{D}}, \qquad
    [K^{conn}_{o}] = k_o \sum_{\alpha = u, v, w} [B_\alpha]^\top \text{diag}\left( w_g \right) [B_\alpha]

where `\mathcal{D} = \{ g : k_g \neq k_o \}` is the set of damaged points and
the subscript `\mathcal{D}` keeps only their rows. `[K^{conn}_{o}]`, the
stiffness of the pristine interface, is computed once for each `k_o` and
cached. Before damage onset `\mathcal{D}` is empty and the assembly reduces
to a copy of `[K^{conn}_{o}]`. After onset, `\mathcal{D}` holds the fracture
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
``'SB'`` and ``'SB_TSL'``, raising a ``ValueError`` otherwise. In the elastic
range of the DCB, the matrix products and
the update of Eq. :eq:`cz-K-update`, together with the changes of the driver
described in :ref:`cohesive_zone_solution`, reduce the time of eight
displacement increments from 341 s to 29 s, with the same loads to a relative
`1.3 \times 10^{-8}`, the order of the convergence tolerance.


Load measured at the loaded edge
--------------------------------

The thesis evaluates the load with two approaches, Section 5.4, and scales
the second one, the area integral of the traction, Eq. 5.38, with the ratio
between both at the first increment, Section 7.1.4. The area integral was
computed with the corrected separation of Section 6.3.1, in which the
separation behind the crack front is set to zero. This removes the
compressive tractions that balance the tensile ones. For a beam on an elastic
foundation the separation behind the peak is physically negative, and its
integral is not small. The table below compares, for a linear elastic
solution with a tip displacement of 1 mm, the reaction at the loaded edge with
the integrals of the traction computed with the full separation `\Delta` and
with the corrected separation `\Delta^{corr}`.

.. list-table:: Out-of-plane force of the DCB for a tip displacement of 1 mm,
                linear solution, default edge penalties.
    :header-rows: 1

    * - `k_o` [N/mm³]
      - Reaction [N]
      - `\int k \Delta \, dA` [N]
      - `\int k \Delta^{corr} \, dA` [N]
      - `\int Q_x \, dy` [N]
    * - `1\times10^4`
      - 25.30
      - 25.31
      - 353.0
      - 24.73
    * - `5\times10^4`
      - 26.18
      - 26.18
      - 523.9
      - 25.59
    * - `2\times10^5`
      - 26.69
      - 26.70
      - 741.2
      - 26.09

The ratio between the corrected integral and the reaction depends on `k_o`
and on the damage state, such that a scaling factor fixed at the first
increment distorts the curve after damage onset. The line integral of `Q_x`
with finite differences of `M_{xx}` is 2% low. The implementation now
provides:

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


Penalty stiffness of the edge connections
-----------------------------------------

The default penalties of :func:`.calc_kt_kr` for an ``'SSxcte'`` connection
between two equal laminates are `k_t = A_{11}/h` and `k_r = D_{11}/h`, from
Castro and Donadon (2017) [castro2017Multidomain]_. With the penalty energy
`k_r (w^i_{,x} - w^j_{,x})^2` the connection behaves as a rotational spring
of stiffness `2 k_r` per unit width, which adds to a cantilever loaded at the
tip a compliance equal to `3h/(2a)` of its bending compliance, about 7% for
each connection of the DCB arm (`h = 2.1` mm, `a = 48` mm). The arms of the
reference model have two connections each, at the crack tip moment. The
table below gives the elastic stiffness of the DCB, `k_o = 2\times10^5`
N/mm³, when the default `k_t` and `k_r` are multiplied by a factor.

.. list-table:: Elastic stiffness of the DCB versus the factor applied to the
                default edge penalties.
    :header-rows: 1

    * - Factor
      - `P/\delta` [N/mm]
      - `\Delta_{max}/\delta`
    * - 1
      - 26.69
      - `4.22\times10^{-4}`
    * - 10
      - 32.54
      - `5.21\times10^{-4}`
    * - 100
      - 33.27
      - `5.35\times10^{-4}`
    * - 1000
      - 33.35
      - `5.36\times10^{-4}`

The converged value agrees with a beam on an elastic foundation, `P/\delta
\approx 33` N/mm. With the default penalties the separation at the crack tip,
and therefore the damage onset, lag the imposed displacement by 20%. This is
independent of the 0.015 mm domain used between the cohesive domain and the
free arm: replacing it by a 5 mm domain gives 27.23 N/mm with the default
penalties and 33.28 N/mm with the factor 100. The drivers multiply the
default ``'SSxcte'`` penalties by ``edge_penalty_factor``, 100 by default. The
library defaults are unchanged.


.. _cohesive_zone_solution:

Solution procedure
------------------

The drivers, ``tests/multidomain/test_dcb_damage.py`` and
``notebooks/dcb_utils.py``, solve Eq. :eq:`cz-residual` under displacement
control with the modified Newton-Raphson method of Section 5.3 of the thesis.
The changes described below modify the iterations but, except for the
convergence criterion, not the equations being solved.

**Secant stiffness of the cohesive zone at every iteration.** The driver of
the thesis refreshed `[K^{conn}_{dmg}]` only every `n_{NR} = 3` iterations,
together with the tangent stiffness matrix. However, `[K^{conn}_{dmg}]` is
the secant stiffness of the cohesive zone and it is part of the internal force
vector, Eq. :eq:`cz-residual`. Keeping it frozen makes the residual
inconsistent with the traction-separation law, and a converged state may
carry a damage field computed with the displacements of earlier iterations.
The drivers now evaluate `[K^{conn}_{dmg}]` at every iteration, and only the
tangent stiffness of the panels, `[K_C] + [K_G]`, is refreshed every `n_{NR}`
iterations. The iteration matrix of Section 5.3 of the thesis is `[K_C] +
[K_G] + [K^{conn}_{dmg}] + [K^{pen}_{PD}]`, where the secant stiffness of the
cohesive zone replaces its tangent; the drivers add the damage-rate term
`[K^{conn}_{\dot d}]` of Eq. :eq:`cz-tangent`.

**Predictor.** Each increment started from the last converged state. Under
displacement control the first residual is then dominated by the jump of the
prescribed displacement, and the first iteration always ended with a ratio of
Eq. :eq:`cz-criterion` close to 1. The increment `n` now starts from the
linear extrapolation of the last two converged states:

.. math::

    \{c\}_{n}^{0} = \{c\}_{n-1} + \frac{w_{p,n} - w_{p,n-1}}{w_{p,n-1} - w_{p,n-2}} \left( \{c\}_{n-1} - \{c\}_{n-2} \right)

with the state `w_p = 0`, `\{c\} = 0` used for the second increment
(``predictor=True``). The damage history is only updated after convergence,
so the damage of the predicted state is not stored.

**Reuse of** `[K_C] + [K_G]` **in the elastic range.** While no point of the
interface is damaged, the tangent stiffness of the panels is reused for up to
``kT_pan_reuse_steps`` increments, 5 by default, and within an increment it
is refreshed every `n_{NR}` iterations, but not after the first one. The
residual is always evaluated exactly, so the converged solution does not
change; only the number of iterations can increase. After onset, `[K_C] +
[K_G]` is evaluated at the start of every increment.

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
correction moves the whole crack front at once. In the DCB with `k_o = 5
\times 10^4` N/mm³ and `\tau_o = 87` MPa, at `w_p = 6.8` mm, every correction
changed `\{c\}` by 3--20%, the normal separation of points at the crack front
swung between `-0.0018` and `+0.0039` mm, about `3 \Delta_o`, and 14--110
points switched between loading and unloading at every iteration. The
iterations entered an exact cycle of period 6, with the ratio of
Eq. :eq:`cz-criterion` repeating 0.998, 0.999, 0.994, 0.995, 0.988, 0.687,
which no number of iterations can break. Two safeguards are added to the
drivers. Neither changes the equations or the convergence criterion, so a
converged state is the same solution as before, only the path to it changes.

**Backtracking line search.** With `\{\delta c\}` the solution of
`[K]\{\delta c\} = -\{R(\{c_i\})\}`, where `[K]` is the iteration matrix, the
new iterate is `\{c_{i+1}\} = \{c_i\} + s\{\delta c\}`, with the step length
`s` taken as the first of `s = 1, \frac{1}{2}, \frac{1}{4}, \dots,
2^{-n_{ls}}` that satisfies:

.. math::
    :label: cz-armijo

    \left\| \{R(\{c_i\} + s\{\delta c\})\} \right\|_D \le \left(1 - 10^{-4} s\right) \left\| \{R(\{c_i\})\} \right\|_D, \qquad
    \left\| \{v\} \right\|_D = \sqrt{\{v\}^\top [D]^{-1} \{v\}}

where `[D]` holds the absolute values of the diagonal of `[K]` at the start
of the increment, the scaling of the criterion of Eq. :eq:`cz-criterion`,
which balances the rows of the prescribed-displacement penalty against the
others. The merit function is the residual norm and not an energy, because
the damaged interface has no potential and `[K]` is not symmetric. When `[K]`
is the exact Jacobian `[J]` of the residual, `\{\delta c\}` is a descent
direction of this merit function for any positive weighting:

.. math::

    \left. \frac{d}{ds} \frac{1}{2} \left\| \{R(\{c_i\} + s\{\delta c\})\} \right\|_D^2 \right|_{s=0} = \{R\}^\top [D]^{-1} [J] \{\delta c\} = - \{R\}^\top [D]^{-1} \{R\} < 0

so a small enough `s` always satisfies Eq. :eq:`cz-armijo`. Between refreshes
of `[K_C] + [K_G]` the iteration matrix is not the exact Jacobian and this is
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

**Convergence criterion.** The prescribed displacement is imposed with `k_w =
10^6`, such that `\{f_{ext}\}` and `[K^{pen}_{PD}]\{c\}` are of the order of
`10^6`--`10^7`, while the physical forces are of the order of `10^2` N. The
denominator of Eq. 5.35 of the thesis, `\max(F(\{f_{ext}\}),
F(\{f_{int}\}))`, is dominated by the penalty terms, and the tolerance
`10^{-4}` is then not a relative tolerance on the physical forces. The
drivers use:

.. math::
    :label: cz-criterion

    \frac{F(\{R\})}{\max\left( F(\{f_{int}\}), F([K^{pen}_{PD}]\{c\} - \{f_{ext}\}) \right)} < \varepsilon

where `F(\cdot) = \|\cdot\|_D` and the second term of the denominator is the
reaction of the prescribed displacement. The large `k_w` also degrades the
accuracy of the first solution of each increment, whose ratio of
Eq. :eq:`cz-criterion` can be close to 1. An increment is therefore declared
divergent only when the ratio is not finite or exceeds `10^3`, instead of
when it reaches 1. A divergent increment does not abort the analysis: it is
bisected as described above, and the analysis is aborted only when a
sub-increment of `2^{-6}` of the original increment (``max_bisections``)
still does not converge.


Verification
------------

The implementation is verified by the tests of
``tests/multidomain/test_sb_tsl.py``:

- The kernels :func:`.fkCSB11_dmg`, :func:`.fkCSB12_dmg` and
  :func:`.fkCSB22_dmg` were verified by comparing `\{c\}^\top [K] \{c\}` with
  twice the Gauss-Legendre integral of the penalty energy of
  Eq. :eq:`cz-energy`, evaluated from the displacement field,
  for random `\{c\}`, random boundary flags, panels with different numbers of
  terms and thicknesses, and a random `k^w_{CZ}` field: the relative
  difference is below `10^{-15}`.
- The assembly by matrix products and the loops of ``kCSB_dmg.pyx`` agree to
  a relative `10^{-12}` for random boundary flags, both orders of the panels
  in the assembly and a damage field with pristine, softening and failed
  points.
- :meth:`.MultiDomain.calc_kT_TSL` was verified against central finite
  differences of `[K^{conn}_{dmg}(\{c\})]\{c\}`, for random states in the
  softening range, with and without points where `d^h > \hat d`: the relative
  error of `[K^{conn}_{dmg}] + [K^{conn}_{\dot d}]` is below `2 \times
  10^{-9}`, against 9--62% for `[K^{conn}_{dmg}]` alone.
- A ``'SB_TSL'`` connection between panels of different dimensions raises a
  ``ValueError``, with the matrix products and with the kernels.

In the nonlinear analyses, the area integral of the traction,
:meth:`.MultiDomain.force_out_plane_damage`, must match the reaction of the
prescribed displacement, :meth:`.MultiDomain.reaction_line_pd_xcte`, which
the drivers report at every increment.


Convergence study
-----------------

The discretisation of the cohesive domain was studied on the reference DCB
(length 65 mm, width 25 mm, precrack 48 mm, arms of 15 plies of 0.14 mm,
`E_1 = 133.15` GPa, `E_2 = 10.95` GPa, `\nu_{12} = 0.316`, `G_{12} = 5.19`
GPa, `G_{Ic} = 1.12` N/mm), with the driver described in
:ref:`cohesive_zone_solution` and two sets of cohesive parameters:

- case A: `k_o = 5 \times 10^4` N/mm³, `\tau_o = 87` MPa;
- case B: `k_o = 2 \times 10^5` N/mm³, `\tau_o = 74.2` MPa.

The cohesive domain is 17 mm long, the free arms are modelled with `8 \times
8` terms, and the prescribed displacement goes up to `w_p = 8` mm in 38
increments, of 0.2 mm after `w_p = 5` mm. The parameters varied are the
number of terms along `x` of the cohesive domain, `m_{tsl}`, and the number of
Gauss-Legendre points along `x` of the connection, `n_x`, in a full grid
`m_{tsl} \in \{10, 15, 20, 25\} \times n_x \in \{60, 120, 180\}`. Separately,
from the reference `m_{tsl} = 15`, `n_x = 120`, the study varied the number
of terms along `y` of the cohesive domain (`n_{tsl} = 10 \to 15`), the number
of Gauss points along `y` (`n_y = 30 \to 60`), the number of terms of the
other domains (`m = n = 8 \to 12`) and the increment of `w_p` after 5 mm
(halved). The metrics are:

- the peak load `P_{max}` and the prescribed displacement at the peak,
  `w_p(P_{max})`;
- the loads at `w_p = 6`, 7 and 8 mm, after the peak;
- the coefficient of variation of the load drop per increment after the peak,
  `CV = \mathrm{std}(\Delta P)/|\mathrm{mean}(\Delta P)|`, which is small for
  a smooth softening and large when the crack front advances in jumps and the
  curve has plateaus;
- the number of increments that did not converge and were bisected, and the
  total number of Newton-Raphson iterations;
- the wall time. The runs shared 16 cores, 8 to 11 at a time, so the wall
  times only indicate the relative cost.

The study is reproduced by ``notebooks/convergence_study.py``, and the metrics
and load curves of all runs are stored in
``notebooks/results/convergence_study.json``.

.. list-table:: Convergence study, terms `m_{tsl}` and Gauss points `n_x`
                along `x` of the cohesive domain. Loads in N, displacements
                in mm, wall time in min. The runs B, `n_x = 60`, with
                `m_{tsl} = 10` and 20 were aborted at `w_p = 7.2` and 7.8 mm,
                when an increment still did not converge after the last
                bisection, and the run with `m_{tsl} = 25` was interrupted at
                `w_p = 7.8` mm.
    :header-rows: 1

    * - case
      - `m_{tsl}`
      - `n_x`
      - `P_{max}`
      - `w_p(P_{max})`
      - `P(6)`
      - `P(7)`
      - `P(8)`
      - `CV`
      - bisections
      - iterations
      - time
    * - A
      - 10
      - 60
      - 166.96
      - 5.80
      - 161.97
      - 150.68
      - 141.10
      - 0.81
      - 5
      - 394
      - 32
    * - A
      - 10
      - 120
      - 167.16
      - 5.80
      - 161.95
      - 150.76
      - 141.20
      - 0.73
      - 0
      - 134
      - 9
    * - A
      - 10
      - 180
      - 167.08
      - 5.80
      - 162.00
      - 150.76
      - 141.19
      - 0.73
      - 0
      - 135
      - 9
    * - A
      - 15
      - 60
      - 165.92
      - 5.60
      - 162.53
      - 150.52
      - 140.70
      - 0.32
      - 4
      - 375
      - 56
    * - A
      - 15
      - 120
      - 165.96
      - 5.60
      - 162.70
      - 150.61
      - 140.70
      - 0.27
      - 0
      - 137
      - 26
    * - A
      - 15
      - 180
      - 165.96
      - 5.60
      - 162.74
      - 150.61
      - 140.60
      - 0.26
      - 0
      - 138
      - 27
    * - A
      - 20
      - 60
      - 165.91
      - 5.60
      - 162.45
      - 150.49
      - 140.69
      - 0.32
      - 2
      - 266
      - 91
    * - A
      - 20
      - 120
      - 165.95
      - 5.60
      - 162.64
      - 150.58
      - 140.69
      - 0.26
      - 0
      - 139
      - 59
    * - A
      - 20
      - 180
      - 165.95
      - 5.60
      - 162.68
      - 150.57
      - 140.61
      - 0.26
      - 0
      - 140
      - 59
    * - A
      - 25
      - 60
      - 165.91
      - 5.60
      - 162.44
      - 150.49
      - 140.69
      - 0.32
      - 1
      - 203
      - 158
    * - A
      - 25
      - 120
      - 165.95
      - 5.60
      - 162.63
      - 150.58
      - 140.70
      - 0.26
      - 0
      - 137
      - 117
    * - A
      - 25
      - 180
      - 165.95
      - 5.60
      - 162.67
      - 150.58
      - 140.62
      - 0.26
      - 0
      - 136
      - 118
    * - B
      - 10
      - 60
      - 168.12
      - 5.80
      - 161.41
      - 150.10
      - --
      - 1.04
      - 6
      - 469
      - 43
    * - B
      - 10
      - 120
      - 168.17
      - 5.80
      - 161.19
      - 150.41
      - 141.41
      - 1.08
      - 0
      - 137
      - 9
    * - B
      - 10
      - 180
      - 168.16
      - 5.80
      - 161.30
      - 150.37
      - 141.44
      - 1.07
      - 0
      - 138
      - 9
    * - B
      - 15
      - 60
      - 165.33
      - 5.80
      - 163.15
      - 150.49
      - 140.40
      - 0.27
      - 7
      - 542
      - 85
    * - B
      - 15
      - 120
      - 165.36
      - 5.80
      - 162.76
      - 150.42
      - 140.75
      - 0.16
      - 0
      - 132
      - 27
    * - B
      - 15
      - 180
      - 165.34
      - 5.80
      - 162.70
      - 150.43
      - 140.70
      - 0.16
      - 0
      - 137
      - 26
    * - B
      - 20
      - 60
      - 165.21
      - 5.80
      - 163.08
      - 150.59
      - --
      - 0.19
      - 10
      - 727
      - 253
    * - B
      - 20
      - 120
      - 165.24
      - 5.80
      - 162.71
      - 150.57
      - 140.86
      - 0.13
      - 0
      - 135
      - 57
    * - B
      - 20
      - 180
      - 165.22
      - 5.80
      - 162.65
      - 150.59
      - 140.77
      - 0.13
      - 0
      - 134
      - 56
    * - B
      - 25
      - 60
      - 165.21
      - 5.80
      - 163.05
      - 150.56
      - --
      - 0.18
      - 2
      - 274
      - --
    * - B
      - 25
      - 120
      - 165.25
      - 5.80
      - 162.69
      - 150.56
      - 140.86
      - 0.13
      - 0
      - 133
      - 107
    * - B
      - 25
      - 180
      - 165.22
      - 5.80
      - 162.64
      - 150.58
      - 140.78
      - 0.12
      - 0
      - 134
      - 104

.. list-table:: Convergence study, other parameters, from the reference
                `m_{tsl} = 15`, `n_x = 120`. With the halved increment the
                loads at 6 and 7 mm are not on the list of prescribed
                displacements.
    :header-rows: 1

    * - case
      - variation
      - `P_{max}`
      - `w_p(P_{max})`
      - `P(6)`
      - `P(7)`
      - `P(8)`
      - `CV`
      - bisections
      - iterations
      - time
    * - A
      - reference
      - 165.96
      - 5.60
      - 162.70
      - 150.61
      - 140.70
      - 0.27
      - 0
      - 137
      - 26
    * - A
      - `n_{tsl} = 15`
      - 165.96
      - 5.60
      - 162.71
      - 150.61
      - 140.68
      - 0.27
      - 0
      - 141
      - 86
    * - A
      - `n_y = 60`
      - 165.97
      - 5.60
      - 162.69
      - 150.62
      - 140.72
      - 0.27
      - 0
      - 136
      - 24
    * - A
      - `m = n = 12`
      - 165.97
      - 5.60
      - 162.70
      - 150.62
      - 140.70
      - 0.27
      - 0
      - 136
      - 57
    * - A
      - `\Delta w_p` halved
      - 166.16
      - 5.68
      - --
      - --
      - 140.70
      - 0.19
      - 0
      - 262
      - 46
    * - B
      - reference
      - 165.36
      - 5.80
      - 162.76
      - 150.42
      - 140.75
      - 0.16
      - 0
      - 132
      - 27
    * - B
      - `n_{tsl} = 15`
      - 165.35
      - 5.80
      - 162.77
      - 150.42
      - 140.74
      - 0.16
      - 0
      - 133
      - 80
    * - B
      - `n_y = 60`
      - 165.34
      - 5.80
      - 162.77
      - 150.41
      - 140.76
      - 0.16
      - 0
      - 131
      - 25
    * - B
      - `m = n = 12`
      - 165.37
      - 5.80
      - 162.76
      - 150.42
      - 140.75
      - 0.16
      - 0
      - 131
      - 54
    * - B
      - `\Delta w_p` halved
      - 165.60
      - 5.68
      - --
      - --
      - 140.75
      - 0.27
      - 0
      - 208
      - 37

The load after the peak for different numbers of terms along `x` of the
cohesive domain is given below, in N. With 10 terms the load drops in steps;
with 15 or more the curves coincide.

.. list-table:: Load after the peak `P` (N) versus the prescribed
                displacement `w_p` (mm), for `m_{tsl} = 10, n_x = 120`;
                `m_{tsl} = 15, n_x = 120`; and `m_{tsl} = 25, n_x = 180`.
    :header-rows: 1

    * - `w_p`
      - A, 10/120
      - A, 15/120
      - A, 25/180
      - B, 10/120
      - B, 15/120
      - B, 25/180
    * - 4.818
      - 147.66
      - 147.65
      - 147.65
      - 146.78
      - 146.75
      - 146.73
    * - 5.000
      - 152.56
      - 152.54
      - 152.54
      - 151.61
      - 151.51
      - 151.50
    * - 5.200
      - 157.78
      - 157.72
      - 157.72
      - 156.76
      - 156.57
      - 156.56
    * - 5.400
      - 162.69
      - 162.54
      - 162.53
      - 161.67
      - 161.27
      - 161.26
    * - 5.600
      - 166.68
      - 165.96
      - 165.95
      - 166.07
      - 165.03
      - 165.08
    * - 5.800
      - 167.16
      - 165.41
      - 165.39
      - 168.17
      - 165.36
      - 165.22
    * - 6.000
      - 161.95
      - 162.70
      - 162.67
      - 161.19
      - 162.76
      - 162.64
    * - 6.200
      - 160.49
      - 160.01
      - 159.97
      - 159.99
      - 160.08
      - 160.04
    * - 6.400
      - 159.73
      - 157.50
      - 157.49
      - 160.27
      - 157.70
      - 157.48
    * - 6.600
      - 155.19
      - 155.00
      - 155.09
      - 156.00
      - 154.96
      - 155.09
    * - 6.800
      - 151.36
      - 152.99
      - 152.72
      - 150.28
      - 153.01
      - 152.75
    * - 7.000
      - 150.76
      - 150.61
      - 150.58
      - 150.41
      - 150.42
      - 150.58
    * - 7.200
      - 150.00
      - 148.39
      - 148.44
      - 150.32
      - 148.45
      - 148.41
    * - 7.400
      - 145.73
      - 146.54
      - 146.42
      - 144.61
      - 146.46
      - 146.45
    * - 7.600
      - 142.80
      - 144.34
      - 144.43
      - 141.96
      - 144.26
      - 144.47
    * - 7.800
      - 142.10
      - 142.38
      - 142.49
      - 142.02
      - 142.59
      - 142.61
    * - 8.000
      - 141.20
      - 140.70
      - 140.62
      - 141.41
      - 140.75
      - 140.78

The results show that:

- **The Ritz basis of the cohesive domain sets the shape of the response after
  the peak.** With `m_{tsl} = 10` the load drops in steps separated by
  plateaus (`CV = 0.73`--`1.08`), and the peak is 0.7% (A) and 1.8% (B) above
  the converged value. From `m_{tsl} = 15` the curve is smooth (`CV \leq
  0.27`), and `m_{tsl} = 15`, 20 and 25 give the same peak within 0.01 N (A)
  and 0.15 N (B), and the same loads after the peak within 0.3 N. The opening
  ahead of the process zone decays as that of a beam on an elastic
  foundation, with the rate `\lambda = (k_o/4D_{11})^{1/4}`; it is 0.59
  mm⁻¹ in case A and 0.83 mm⁻¹ in case B. The converged `m_{tsl} = 15` over
  `L_{tsl} = 17` mm corresponds to `m_{tsl} = 1.5\,\lambda L_{tsl}` in case A
  and `1.06\,\lambda L_{tsl}` in case B; `m_{tsl} \approx 1.5\,\lambda
  L_{tsl}` is used as a conservative rule for other geometries.
- **The number of Gauss points along** `x` **sets the robustness, not the
  converged answer.** With `n_x = 60` the loads are within 0.5 N of those
  with `n_x \geq 120`, but 1 to 10 increments needed bisection and the
  iterations increased 1.5 to 5.4 times, because the crack front crosses
  several integration points in one increment (see
  :ref:`cohesive_zone_solution`). With `n_x = 120` and 180 no
  increment was bisected in any run. The cohesive zone length, estimated as
  `l_{cz} = 0.88\,E_2 G_{Ic}/\tau_o^2` (Rice's model, Turon et al. 2007
  [turon2007MeshSize]_), is 1.43 mm in case A and 1.96 mm in case B; `n_x =
  120` over 17 mm gives a spacing of the Gauss points at the middle of the
  domain of `\pi L_{tsl}/(2 n_x) = 0.22` mm, about `l_{cz}/6.4` in case A,
  i.e. `n_x \approx 10\,L_{tsl}/l_{cz}`. This is the rule used for other
  geometries.
- **The discretisation along** `y` **and of the other domains does not matter
  here.** `n_{tsl} = 15`, `n_y = 60` and `m = n = 12` change the loads by at
  most 0.02 N, since the specimen is uniform across the width.
- **Halving the increment changes only where the peak is sampled.** The peak
  rises by 0.2 N (0.12%) and moves to `w_p = 5.68` mm, closer to the true
  maximum; the loads on the common increments are the same.

The recommended discretisation for this DCB is therefore `m_{tsl} = 15`, `n_x
= 120`, `n_{tsl} = 10`, `n_y = 30`, with `8 \times 8` terms in the other
domains. It gives `P_{max} = 165.96` N at `w_p = 5.6` mm in case A and 165.36
N at 5.8 mm in case B, in about 26 min of wall time on a shared machine,
against about 171 N at about 6 mm for the finite element model of the thesis
(Section 5.6).


Validation against the literature
---------------------------------

The model was compared with eight references that report mode I DCB
results: Alfano and Crisfield (2001) [alfano2001Interface]_, Camanho et al.
(2003) [camanho2003Delamination]_, Turon et al. (2007) [turon2007MeshSize]_,
Krueger (2008) [krueger2008DCB]_, Tijs et al. (2022) [tijs2022Interlaminar]_,
Leciñana et al. (2023) [lecinana2023Fatigue]_ and the PhD thesis of Tijs
(2023) [tijs2023PhD]_, plus Krueger (2012) [krueger2012MMB]_, whose benchmarks
are for the mixed-mode bending specimen and are only recorded, see below.
Each reference has a notebook in ``notebooks/`` with a dictionary of the FE
cases and one of the experimental tests, a comparison table, a plot and a
discussion; the ``panels`` results are stored in ``notebooks/results/*.npz``
and collected by ``notebooks/validation_summary.ipynb``.

**Model.** ``notebooks/dcb_utils.py`` builds a symmetric DCB: CLPT arms, far
end clamped, arm tips opened to `\pm\delta/2` by prescribed displacements,
the load being the reaction of the prescribed displacement. The bonded region
is split into a cohesive domain (``'SB_TSL'``, bilinear law) of length
`L_{tsl}` ahead of the initial crack front and a perfect bond (``'SB'``) over
the rest, such that only the region reached by the crack needs the resolution
of the cohesive zone. A weld narrower than the specimen (Tijs 2023) is
represented by starting the integration points outside the bonded strip fully
damaged. The solution procedure is the one described in
:ref:`cohesive_zone_solution`. The discretisation follows the rules of the
convergence study (``dcb_utils.discretize``): `m_{tsl} = \lceil 1.5\,\lambda
L_{tsl} \rceil` with at most 25 terms, and `n_x = 10\,L_{tsl}/l_{cz}`, but not
less than `4\,m_{tsl}` to integrate the products of the approximation
functions, rounded up to a multiple of 10 and with at most 300 points, and
`L_{tsl}` covers the crack growth plus a margin
of `2\pi/\lambda + 3\,l_{cz}` for the process zone and the compressive lobe
ahead of it; the range of openings is limited so that `L_{tsl}` stays
resolved, which allows 5--13 mm of crack growth. The penalty stiffness is
`k_o = 10^5` N/mm³: a longer decay length `1/\lambda` needs fewer terms, and
for the DCB of Turon et al. (2007) the peak load with the `k_o = 10^6` N/mm³
of the paper differs by 0.04% (64.07 against 64.05 N), in line with their
Fig. 9. The reference curves of linear elastic fracture mechanics (LEFM) are
the simple and the corrected beam theories, the latter with the crack length
`a + \chi h` of Williams, `\chi = \sqrt{E_{11}/(11 G_{13})\,[3 -
2(\Gamma/(1+\Gamma))^2]}`, `\Gamma = 1.18\sqrt{E_{11}E_{22}}/G_{13}`, and the
bending stiffness `D_{11} b` of the arm.

**Data.** Values given in tables of the references are used as such (`^{t}`
in the table of peak loads below); curves were read from the figures by eye
(`^{d}`), with an uncertainty of about 2--5 N, and `^{m}` marks the mean of
the digitised peaks of several tests (full weld: 150 and 170 N; weld center:
122--136 N). The notebooks state the assumptions needed where the references
are incomplete: the opening convention (Alfano and Crisfield plot the
displacement of one arm, `\delta = 2u`), the crack length of one test of Tijs
et al. (2022) inferred from its compliance, and the elastic properties and
length of the welded specimens of Tijs (2023).

.. list-table:: Peak loads of the mode I DCB references. `P_{max}` in N,
                `\delta` at `P_{max}` in mm, differences of ``panels`` in %,
                `\Delta K_0` is the difference of the initial stiffness of
                ``panels`` to the corrected beam theory (CBT), `\Delta a` the
                crack growth of the ``panels`` run in mm. `^{t}`: value of a
                table of the reference; `^{d}`: digitised; `^{m}`: mean of
                digitised tests; `^{b}`: not an FE curve but the VCCT
                benchmark of Krueger (2008), the onset of delamination at
                mid-width, 2.6% (UD) and 12.0% (D±30) below the load at which
                the width-average energy release rate reaches `G_{Ic}`. The
                two Krueger cases are listed with `\tau_o` = 40 MPa, the
                reference giving no strength; with 60 MPa the peaks are 64.2
                and 109.9 N.
    :header-rows: 1

    * - case
      - ``panels`` `P_{max}`
      - ``panels`` `\delta`
      - FE `P_{max}`
      - diff. FE
      - test `P_{max}`
      - diff. test
      - CBT `P_{max}`
      - diff. CBT
      - `\Delta K_0`
      - `\Delta a`
    * - Alfano 2001, Table I, `t_o` = 57 MPa
      - 64.6
      - 1.83
      - 61.5 `^{d}`
      - +5.0
      - --
      - --
      - 63.2
      - +2.2
      - +17
      - 10.5
    * - Alfano 2001, Table I, `t_o` = 1.7 MPa
      - 50.9
      - 2.90
      - 49.5 `^{d}`
      - +2.9
      - --
      - --
      - 63.2
      - -19.4
      - -24
      - 5.8
    * - Alfano 2001, XAS-913C
      - 93.8
      - 1.91
      - 90.0 `^{d}`
      - +4.2
      - --
      - --
      - 91.6
      - +2.3
      - +17
      - 10.6
    * - Camanho 2003, AS4/PEEK
      - 139.9
      - 4.12
      - 155.3 `^{t}`
      - -9.9
      - 147.1 `^{t}`
      - -4.9
      - 138.9
      - +0.7
      - +13
      - 8.2
    * - Turon 2007, T300/977-2
      - 64.1
      - 4.18
      - 57.0 `^{d}`
      - +12.4
      - 62.0 `^{d}`
      - +3.3
      - 63.3
      - +1.3
      - +12
      - 13.0
    * - Krueger 2008, UD `[0]_{24}`
      - 63.6
      - 1.40
      - 60.7 `^{b}`
      - +4.7
      - --
      - --
      - 62.3
      - +2.1
      - +18
      - 9.9
    * - Krueger 2008, D±30
      - 108.7
      - 1.70
      - 102.4 `^{b}`
      - +6.1
      - --
      - --
      - 114.7
      - -5.2
      - +10
      - 7.6
    * - Tijs 2022, `a_0` = 48 mm
      - 166.2
      - 5.68
      - 167.0 `^{d}`
      - -0.5
      - 142.0 `^{d}`
      - +17.0
      - 164.5
      - +1.1
      - +14
      - 11.8
    * - Tijs 2022, `a_0` = 40 mm
      - 197.5
      - 3.99
      - --
      - --
      - 161.0 `^{d}`
      - +22.7
      - 194.5
      - +1.5
      - +17
      - 11.8
    * - Tijs 2023, full weld, 2.1 N/mm
      - 148.3
      - 6.48
      - 155.0 `^{d}`
      - -4.3
      - 160.0 `^{m}`
      - -7.3
      - 155.5
      - -4.6
      - +4
      - 4.9
    * - Tijs 2023, weld center, 2.1 N/mm
      - 119.2
      - 5.75
      - 128.0 `^{d}`
      - -6.9
      - 129.0 `^{m}`
      - -7.6
      - 123.0
      - -3.1
      - +8
      - 5.2
    * - Tijs 2023, weld center, 1.95 N/mm
      - 115.0
      - 5.56
      - 122.0 `^{d}`
      - -5.7
      - 123.0 `^{d}`
      - -6.5
      - 118.5
      - -3.0
      - +8
      - 5.7
    * - Tijs 2023, weld center, 1.12 N/mm
      - 87.8
      - 4.09
      - 96.0 `^{d}`
      - -8.6
      - --
      - --
      - 89.8
      - -2.3
      - +9
      - 9.2

.. list-table:: Initial compliance: ``panels`` against measured compliances
                and the corrected beam theory (CBT). Differences of
                ``panels`` in %, over the crack lengths listed.
    :header-rows: 1

    * - reference
      - specimens, crack lengths
      - vs test
      - vs CBT
    * - Leciñana et al. (2023), Fig. 10
      - UD, 25 mm, `a_0` = 32.05--42.97 mm
      - -18.9 to -15.5
      - -20.0 to -15.5
    * - Tijs (2023), Table 5.4.1
      - autoclave UD, 25 mm, `a` = 30--45 mm
      - -41.4 to -28.0
      - -20.9 to -14.7
    * - Tijs (2023), Table 5.4.1
      - full weld QI, 25 mm, `a` = 30--45 mm
      - -15.2 to -4.7
      - -12.0 to -6.8
    * - Tijs (2023), Table 5.4.1
      - weld center QI, 12.7 mm, `a` = 30--45 mm
      - -2.1 to +6.6
      - -13.5 to -7.5

The results show that:

- **Same cohesive law as the reference.** For the DCB of Tijs et al. (2022),
  which is also the reference case of the thesis, the FE model of the paper
  uses the same geometry, material and bilinear law (`G_{Ic}` = 1.12 N/mm,
  `\tau^0` = 87 MPa). ``panels`` predicts 166.2 N at 5.68 mm against 167 N at
  5.7 mm, -0.5% on the load and -0.4% on the opening.
- **Cohesive zone beyond LEFM.** With the low strength `t_o` = 1.7 MPa of
  Alfano and Crisfield (2001) the process zone is of the order of the
  specimen length and the peak of their interface model is 22% below the
  corrected beam theory; ``panels`` reproduces it within 2.9%, at the same
  opening and with the same initial stiffness (+3.2%).
- **Nominal strengths.** For the unidirectional specimens with nominal
  strengths, the ``panels`` peak is within 2.3% of the corrected beam theory
  in every case, and after the peak it follows the LEFM propagation branch
  (48.9 against 49.0 N at `\delta` = 3.22 mm for Alfano and Crisfield, Table
  I). The FE results of the references scatter by about `\pm 10\%` around the
  corrected beam theory: 1 mm decohesion elements for a cohesive zone of 1.35
  mm in Camanho et al. (2003), whose FE peak is above both the test and the
  beam theories, and about 57 N in Turon et al. (2007), below both.
- **Tests.** The tests with a bilinear-like response are predicted within 5%
  (Camanho et al. -4.9%, Turon et al. +3.3%). The AS4D/PEKK-FC tests show a
  strong R-curve (initiation about 0.7 N/mm, propagation 1.12 N/mm) that a
  bilinear law with the propagation toughness cannot represent: ``panels``,
  the bilinear FE model of Tijs et al. and the corrected beam theory are all
  15--23% above these tests, which the tabular law of the paper matches.
  Fibre bridging also makes the experimental curves decrease more slowly than
  every model after the peak.
- **Welded joints.** For the welded DCB of Tijs (2023) ``panels`` is 4--9%
  below the FE results of the thesis, 2--5% below the corrected beam theory
  and 6.5--7.6% below the tests with a single toughness value. The cohesive
  zone is long compared with the arm thickness (`l_{cz}` = 2.5--2.7 mm, `h` =
  2.24 mm), which lowers the peak below the LEFM limit; the FE model of the
  thesis is not described in enough detail to separate this effect from its
  modelling assumptions.
- **Initial compliance.** The CLPT arms of ``panels`` are too stiff for the
  unidirectional specimens: the compliance is 15--19% below the tests and
  15--21% below the corrected beam theory, which matches the tests of
  Leciñana et al. within 4%. The arms have no transverse shear deformation,
  which is large for these materials (`E_{11}/G_{13}` = 22--27), and the
  rotation at the crack front comes only from the elastic foundation of the
  cohesive interface. The same appears in the load-opening curves as an
  initial stiffness 12--17% above the corrected beam theory, while the peak
  loads, controlled by the energy release rate, are not affected. For the
  quasi-isotropic welded specimens, with a bending modulus closer to the
  shear modulus, the compliance of ``panels`` is within 15% of the tests.
- **Robustness.** All 16 nonlinear runs converged at every opening without
  bisection.

**Krueger (2008).** NASA/TM-2008-215123 [krueger2008DCB]_ gives two mode I DCB
benchmarks, a unidirectional `[0]_{24}` specimen of T300/1076 and a
multidirectional D±30 specimen of C12K/R6376, built with 3D solid elements and
the virtual crack closure technique, with no cohesive law, no interfacial
strength and no experiment. Their critical load is the one at which `G_T`
reaches `G_c` *at mid-width*, while ``panels`` and beam theory follow the
width average `\bar{G} = P_{crit}^2 a_e^2/(b\,EI)`; the two criteria differ
by `\sqrt{G_c/\bar{G}}`, 1.026 and 1.123, which is the whole offset between
the benchmark and the corrected beam theory. ``panels`` is 2.1% above the
corrected beam theory for the unidirectional specimen and 5.2% below it for
the multidirectional one, whose cohesive zone (`l_{cz}` = 1.99 mm with
`\tau_o` = 40 MPa) is not small compared with its 2.0 mm arms. Raising
`\tau_o` from 40 to 60 MPa moves the peaks by 1.0 and 1.2% only, so the
comparison does not rest on that choice. See
``notebooks/krueger2008_dcb.ipynb``.

**Krueger (2012).** NASA/CR-2012-217562 [krueger2012MMB]_ gives benchmarks
for the mixed-mode bending specimen of IM7/8552 at `G_{II}/G_T` = 0.2, 0.5 and
0.8, with the critical points (1.64 mm, 128.5 N), (1.34 mm, 385 N) and (1.65
mm, 751 N). They are recorded in ``notebooks/krueger2012_mmb_benchmark.ipynb``
but not simulated: the damage of ``'SB_TSL'`` is driven by the normal
separation only, and the loading lever is not modelled.


Differences that remain
-----------------------

- The interpenetration stiffness `k_{ipen}` of Eq. 5.12 is computed by
  :func:`.calc_kw_tsl` but not used: :meth:`.MultiDomain.calc_k_dmg` returns
  `k_o(1-d)` also for negative separations, such that a fully damaged point
  offers no resistance to interpenetration. This does not affect a
  monotonically loaded DCB.
- The damage history ``MultiDomain.dmg_index`` is stored once per assembly,
  which restricts the model to one ``'SB_TSL'`` connection.
- The finite element model of Section 5.6 of the thesis constrains `U_x`
  along the loaded edge, while the Ritz model leaves `u` free there.
- The arms follow CLPT: without transverse shear deformation, the compliance
  of unidirectional DCB specimens is 15--21% below the corrected beam theory
  (see the validation above). A shear deformable kinematics would remove this
  difference.
- The cohesive law is bilinear and the damage is driven by the normal
  separation only: R-curve effects (multilinear laws) and mixed-mode
  delamination are not represented.
