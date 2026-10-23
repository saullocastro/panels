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
- the rules of the discretisation of the cohesive domain,
  :func:`.tsl_resolution`, :func:`.tsl_margin` and
  :func:`.tsl_discretization`, see the convergence study below;
- the methods of :class:`.MultiDomain`: :meth:`.MultiDomain.get_kC_conn`,
  :meth:`.MultiDomain.calc_k_dmg`, :meth:`.MultiDomain.correct_separation`,
  :meth:`.MultiDomain.calc_kT_TSL`, :meth:`.MultiDomain.reaction_line_pd_xcte`
  and :meth:`.MultiDomain.force_out_plane_damage`;
- the nonlinear driver ``tests/multidomain/test_dcb_damage.py`` and its
  counterpart for the validation notebooks, ``notebooks/dcb_utils.py``.

The model of the thesis is the DCB of its Section 7.3: `L = 65` mm, `b = 25`
mm, `a_0 = 48` mm, arms `[0]_{15}` with ply thickness 0.14 mm of
AS4D/PEKK-FC, `G_{Ic} = 1.12` N/mm, three domains per arm, of which the first
one, with length `a_1 = L - a_0 = 17` mm, carries the cohesive zone. It is
the model of the sections below up to the solution procedure and of
``tests/multidomain/test_dcb_damage.py``. The convergence study and the
validation use the model of ``notebooks/dcb_utils.py``, in which the
cohesive domain is followed by a perfect bond, on the longer reference DCB
of the convergence study.


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
:func:`.fkCSB12_dmg` and :func:`.fkCSB22_dmg`. The undamaged connection
``'SB'``, :func:`.fkCSB`, imposes the jump of Eq. :eq:`cz-jump` at zero
separation, such that the pristine ``'SB_TSL'``, with `k^w_{CZ} = k_o` at
every point, has the matrix of ``'SB'`` with `k_t = k_o`, as checked by
``tests/multidomain/test_sb_unified.py``.

The damage kernels, ``kCSB_dmg.pyx``, had the approximation functions in
`\eta` of the top panel in the coupling block built with the flags of the
edge `y_1` in place of `y_2`.
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
loops take 6.5 s per assembly with the code of the thesis (panels 0.6.21), 67%
of the run time in the elastic range.
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
``'SB'`` and ``'SB_TSL'``, raising a ``ValueError`` otherwise.

The cost was measured on the DCB model of the thesis
(``tests/multidomain/test_dcb_damage.py``, `60 \times 30` integration
points, `10 \times 10` and `8 \times 8` terms), on a laptop with 16 threads,
by timing :meth:`.MultiDomain.get_kC_conn` in the driver of the thesis
(panels 0.6.21) and in the current driver (panels 0.13.11), as the medians
of three runs, between which the times varied by up to 50%. One assembly of `[K^{conn}_{dmg}]` takes 6.5 s with the loops of
the thesis (panels 0.6.21) at any damage state, 0.42 s with the
current kernels, and 0.035 s with the matrix products for the pristine
interface and 0.05--0.08 s with 20--100% of the points damaged, 85--190
times faster than the loops of the thesis. Eight displacement increments in
the elastic range take 218 s with the code of the thesis and 6.4 s with
panels 0.13.11, the matrix products and the update of Eq. :eq:`cz-K-update`
together with the changes of the driver described in
:ref:`cohesive_zone_solution`; with the current kernels and without the
predictor and the reuse of the tangent stiffness they take 29 s, with the
same loads to a relative `3.2 \times 10^{-8}`, the order of the convergence
tolerance.


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
integral is not small. The table below compares, for a linear elastic
solution with a tip displacement of 1 mm, the reaction at the loaded edge with
the integrals of the traction computed with the full separation `\Delta` and
with the corrected separation `\Delta^{corr}`, and with the line integral of
`Q_x` of :meth:`.MultiDomain.force_out_plane` at the second last of 100
points along `x` of the free arm. The model is the DCB of the thesis with the
default edge penalties, `10 \times 6` terms in the cohesive domains, `8
\times 6` terms in the other domains and `60 \times 20` integration points.
With `k_o = 2\times10^5` N/mm³ an opening of 1 mm exceeds the separation at
damage onset, `\Delta_o = 4.35\times10^{-4}` mm, therefore the linear
solution is computed with a tip displacement of 0.5 mm, below the onset for
the three values of `k_o`, and scaled to 1 mm.

.. list-table:: Out-of-plane force of the DCB for a tip displacement of 1 mm,
                linear solution, default edge penalties.
    :header-rows: 1

    * - `k_o` [N/mm³]
      - Reaction [N]
      - `\int k \Delta \, dA` [N]
      - `\int k \Delta^{corr} \, dA` [N]
      - `\int Q_x \, dy` [N]
    * - `1\times10^4`
      - 24.98
      - 24.99
      - 388.1
      - 24.41
    * - `5\times10^4`
      - 25.87
      - 25.87
      - 593.1
      - 25.28
    * - `2\times10^5`
      - 26.41
      - 26.42
      - 839.4
      - 25.82

The reaction and the integral of `Q_x` of the loaded arm carry a round-off
error of up to `3\times10^{-4}` relative, which comes from the 0.015 mm
connecting domain and changes, e.g., with the number of integration points of
the panels, whereas the integrals of the traction change by less than
`2\times10^{-6}`. Therefore the reaction and `\int k \Delta \, dA` differ in
the fourth digit, by `2.2\times10^{-4}`; the difference falls below
`3\times10^{-7}` with a connecting domain of 0.15 mm and below
`2\times10^{-9}` with 1.5 or 5 mm. The values are asserted by
``tests/multidomain/test_dcb_linear_measures.py``, whose function
``dcb_thesis_linear`` builds and solves the model:

.. literalinclude:: ../../tests/multidomain/test_dcb_linear_measures.py
    :pyobject: test_load_measures

Earlier versions of this page gave, for this table and the table of the next
section, values computed with the tangential separation of Eq. 4.62 of the thesis
[nathan2024MSc]_, Eq. :eq:`cz-thesis-jump`, before its correction to
Eq. :eq:`cz-jump`; restoring Eq. :eq:`cz-thesis-jump` in the current code
reproduces them, and with Eq. :eq:`cz-jump` the reactions are 1.0--1.2% lower
and `\Delta_{max}/\delta` of the next section is 20% higher.

The ratio between the corrected integral and the reaction, 15.5 to 31.8,
depends on `k_o` and on the damage state, such that a scaling factor fixed at
the first increment distorts the curve after damage onset. The line integral
of `Q_x` with finite differences of the moments is 2.3% low. The
implementation now provides:

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
`\frac{1}{2} k_r (w^i_{,x} - w^j_{,x})^2` per unit width, see
:mod:`panels.multidomain.connections`, the connection behaves as a
rotational spring of stiffness `k_r` per unit width, which adds to a
cantilever loaded at the tip a compliance equal to `3 D_{11}/(k_r a) = 3h/a`
of its bending compliance, about 13% for each connection of the DCB arm (`h
= 2.1` mm, `a = 48` mm). The arms of the
reference model have two connections each, at the crack tip moment. The
table below gives the elastic stiffness of the DCB, `k_o = 2\times10^5`
N/mm³, when the default `k_t` and `k_r` are multiplied by a factor, with the
model and the linear solutions of the previous section.

.. list-table:: Elastic stiffness of the DCB versus the factor applied to the
                default edge penalties.
    :header-rows: 1

    * - Factor
      - `P/\delta` [N/mm]
      - `\Delta_{max}/\delta`
    * - 1
      - 26.41
      - `5.07\times10^{-4}`
    * - 10
      - 32.13
      - `6.26\times10^{-4}`
    * - 100
      - 32.84
      - `6.42\times10^{-4}`
    * - 1000
      - 32.91
      - `6.43\times10^{-4}`

The converged value agrees with a beam on an elastic foundation, `P/\delta
\approx 33` N/mm. With the default penalties the stiffness is 20% lower, and
the separation at the crack tip, and therefore the damage onset, lag the
imposed displacement by 21%. This is independent of the 0.015 mm domain used
between the cohesive domain and the free arm: replacing it by a 5 mm domain
gives 26.94 N/mm with the default penalties and 32.85 N/mm with the factor
100. These values are asserted by:

.. literalinclude:: ../../tests/multidomain/test_dcb_linear_measures.py
    :pyobject: test_edge_penalty_factor

The drivers multiply the
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
correction moves the whole crack front at once, and many points can cross a
kink in one iteration when the integration points are sparse compared with
the cohesive zone. On the reference DCB of the convergence study below (case
A, `m_{tsl} = 25`) with `n_x = 60`, 3.6 points per cohesive zone length, the
iterations without the safeguards below converge up to the peak, but at the
first opening after it, `\delta = 5.8` mm, they enter an exact cycle: from
the eighth iteration on, the ratio of Eq. :eq:`cz-criterion` repeats values
between 0.76 and 0.999 with a period of 12 iterations, four times the refresh
interval `n_{NR} = 3` of `[K_C] + [K_G]`, to a relative `3 \times 10^{-10}`,
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

**Effect of the consistent tangent and of the safeguards.**
``notebooks/dcb_mode_I_solver_study.py`` solves the reference DCB of the
convergence study below, case A, with parts of the procedure switched off.
On the
reference discretisation (`m_{tsl} = 25`, `n_x = 180`) the full procedure
takes 139 iterations in 38 increments, without bisection, and the consistent
tangent alone is enough: without the line search and without bisection,
every opening converges in 2 to 7 iterations, 137 in total, with the same
loads to `10^{-4}` N; the line search reduces the step in only 9 iterations
after the peak. With the secant stiffness as the only tangent of the
interface, also without bisection, the iterations grow as the process zone
develops, from 6 per increment at damage onset to 124 at the peak, `\delta =
5.6` mm: 518 iterations up to the peak against 71 with the consistent
tangent, and at the next opening the ratio of Eq. :eq:`cz-criterion`
decreases only linearly, to `9 \times 10^{-4}` after 200 iterations. With
`n_x = 60`, the line search breaks the cycle described above, 251 of the 332
iterations taking a reduced step: the openings of 5.8, 6.0 and 6.2 mm
converge in 8, 12 and 47 iterations, but at 6.4 mm the ratio stagnates at
0.45--0.47 with steps of `2^{-6}` to `2^{-2}`. With the bisection as well,
the analysis completes with 3 bisected increments and 384 iterations. The
bisection is therefore the safeguard of the coarse discretisations, see the
convergence study. In the 17 nonlinear runs of the validation notebooks, 980
openings, every opening converged without bisection in 1 to 7 iterations,
3.0 on average.

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
the drivers report at every increment. Both are equal by equilibrium of the
loaded arm when the cohesive domain carries the whole load, see
:ref:`cohesive_zone_load`. In the analyses of the reference DCB of the
convergence study below, up to an opening of 8 mm, they agree within `8.7
\times 10^{-4}` with `k_o = 10^5` N/mm³ and `3.7 \times 10^{-5}` with `k_o =
2 \times 10^5` N/mm³ at every opening.


Convergence study
-----------------

**Reference DCB.** The discretisation of the cohesive domain was studied on
the specimen of Tijs et al. (2022) [tijs2022Interlaminar]_ with `a_0 = 48`
mm and the length `L = 100` mm of their finite element model: width 25 mm,
arms of 15 unidirectional plies of 0.14 mm of AS4D/PEKK-FC (`h = 2.1` mm),
`E_1 = 133.15` GPa, `E_2 = 10.95` GPa (averages of the tensile and
compressive moduli), `\nu_{12} = 0.316`, `G_{12} = 5.19` GPa, `G_{Ic} =
1.12` N/mm and `\tau_o = 87` MPa. Two penalty stiffnesses are used:

- case A: `k_o = 10^5` N/mm³, the value of the validation runs below;
- case B: `k_o = 2 \times 10^5` N/mm³, the mode I penalty stiffness used for
  conduction-welded joints of the same material by Tijs et al. (2022), Compos.
  Struct. 281, 114964.

The model of ``notebooks/dcb_utils.py`` is used with ``conn_method='penalty'``
and the driver described in :ref:`cohesive_zone_solution`. The cohesive
domain is `L_{tsl} = 24` mm long, of the 52 mm ligament, and the opening goes
up to `\delta = 8` mm in 38 increments: 23 equal increments up to 5 mm and
increments of 0.2 mm after it. The opening ahead of the process zone decays
with the rate `\lambda = (k_o/2D_{11})^{1/4}` of two arms connected by an
elastic foundation of stiffness `k_o`, the opening being `w^t - w^b`, 0.83
mm⁻¹ in case A and 0.99 mm⁻¹ in case B, and the cohesive zone length estimated with Rice's model,
`l_{cz} = 0.88\,E_2 G_{Ic}/\tau_o^2` (Turon et al. 2007
[turon2007MeshSize]_), is 1.43 mm in both cases.

**Runs.** A full grid `m_{tsl} \in \{10, 15, 20, 25, 30\} \times n_x \in
\{60, 120, 180\}` of terms and Gauss-Legendre points along `x` of the
cohesive domain is run for both cases. From the reference discretisation
`m_{tsl} = 25`, `n_x = 180`, `n_{tsl} = 10`, `n_y = 30` and `8 \times 8`
terms in the other domains, the following are varied one at a time: the terms
along `y` of the cohesive domain (`n_{tsl} = 10 \to 15`), the Gauss points
along `y` (`n_y = 30 \to 60`), the terms of the other domains (`m = n = 8
\to 12`), the increment after `\delta = 5` mm (halved) and the length of the
cohesive domain (`L_{tsl} = 24 \to 17` mm). The metrics are:

- the peak load `P_{max}`, the opening at the peak, `\delta(P_{max})`, and
  the loads at `\delta = 6`, 7 and 8 mm;
- `e_P`, the largest difference of the load to that of the finest run
  (`m_{tsl} = 30`, `n_x = 180`) over all openings, relative to its peak load;
- `e_{eq}`, the largest relative difference between the reaction at the
  loaded edge and the area integral of the normal traction of the
  interface, see :ref:`cohesive_zone_load`. Both are equal by equilibrium of
  the loaded arm when the cohesive domain carries the whole load, so `e_{eq}`
  grows when the basis does not represent the tractions or when the
  tractions ahead of the process zone, its compressive lobe and the decaying
  opening beyond it, reach the perfect bond beyond the cohesive domain,
  whose tractions are not part of the integral;
- the coefficient of variation of the load drop per increment after the
  peak, `CV = \mathrm{std}(\Delta P)/|\mathrm{mean}(\Delta P)|`, which is
  small for a smooth softening and large for a staircase or sawtooth
  response;
- the number of bisected increments, the number of Newton-Raphson iterations
  and the wall time.

The study is reproduced by ``notebooks/dcb_mode_I_convergence_study.py``, and
the metrics and load curves of all runs are stored in
``notebooks/results/dcb_mode_I_convergence_study.json``. The runs were done
with panels 0.13.11 on a laptop with 16 threads, eight runs at a time with two threads
each, so the wall times indicate the relative cost; the reference run takes
10 min in this setting and 4.5 min of wall time (17 min of CPU time) when run
alone.

.. list-table:: Convergence study, terms `m_{tsl}` and Gauss points `n_x`
                along `x` of the cohesive domain, `L_{tsl} = 24` mm. Loads
                in N, openings in mm, `e_P` and `e_{eq}` in %, wall time in
                min. The runs with `m_{tsl} = 10` and `n_x = 60` or 120
                stopped at `\delta = 6.2` and 7.2 mm (case A) and 7.4 mm
                (case B), and the run B with `m_{tsl} = 15`, `n_x = 60` at
                7.8 mm, when an increment did not converge after the last
                bisection; their `e_P` is evaluated up to that opening.
    :header-rows: 1

    * - case
      - `m_{tsl}`
      - `n_x`
      - `P_{max}`
      - `\delta(P_{max})`
      - `P(6)`
      - `P(7)`
      - `P(8)`
      - `CV`
      - `e_P`
      - `e_{eq}`
      - bisections
      - iterations
      - time
    * - A
      - 10
      - 60
      - 168.36
      - 6.20
      - 166.88
      - --
      - --
      - --
      - 5.1
      - 19
      - 7
      - 81
      - 7
    * - A
      - 10
      - 120
      - 168.22
      - 6.20
      - 166.88
      - 154.38
      - --
      - 2.16
      - 5.0
      - 33
      - 7
      - 129
      - 10
    * - A
      - 10
      - 180
      - 168.25
      - 6.20
      - 166.88
      - 154.38
      - 141.13
      - 1.98
      - 5.0
      - 47
      - 3
      - 327
      - 7
    * - A
      - 15
      - 60
      - 166.43
      - 5.60
      - 162.85
      - 150.95
      - 139.52
      - 0.63
      - 1.2
      - 0.32
      - 3
      - 283
      - 8
    * - A
      - 15
      - 120
      - 166.23
      - 5.60
      - 162.52
      - 151.68
      - 139.73
      - 0.56
      - 0.77
      - 0.31
      - 0
      - 145
      - 4
    * - A
      - 15
      - 180
      - 166.19
      - 5.60
      - 162.53
      - 151.71
      - 139.95
      - 0.51
      - 0.71
      - 0.30
      - 0
      - 139
      - 4
    * - A
      - 20
      - 60
      - 166.36
      - 5.60
      - 162.74
      - 150.30
      - 140.96
      - 0.42
      - 0.64
      - 0.087
      - 3
      - 317
      - 15
    * - A
      - 20
      - 120
      - 166.16
      - 5.60
      - 162.57
      - 150.63
      - 140.89
      - 0.26
      - 0.21
      - 0.087
      - 0
      - 143
      - 6
    * - A
      - 20
      - 180
      - 166.12
      - 5.60
      - 162.71
      - 150.66
      - 140.93
      - 0.24
      - 0.078
      - 0.087
      - 0
      - 136
      - 6
    * - A
      - 25
      - 60
      - 166.35
      - 5.60
      - 162.68
      - 150.21
      - 140.78
      - 0.40
      - 0.60
      - 0.088
      - 3
      - 384
      - 31
    * - A
      - 25
      - 120
      - 166.14
      - 5.60
      - 162.51
      - 150.51
      - 140.75
      - 0.24
      - 0.17
      - 0.087
      - 0
      - 135
      - 10
    * - A
      - 25
      - 180
      - 166.10
      - 5.60
      - 162.65
      - 150.54
      - 140.78
      - 0.23
      - 0.018
      - 0.087
      - 0
      - 139
      - 10
    * - A
      - 30
      - 60
      - 166.34
      - 5.60
      - 162.67
      - 150.21
      - 140.78
      - 0.40
      - 0.60
      - 0.088
      - 3
      - 358
      - 43
    * - A
      - 30
      - 120
      - 166.14
      - 5.60
      - 162.51
      - 150.49
      - 140.77
      - 0.24
      - 0.17
      - 0.087
      - 0
      - 135
      - 14
    * - A
      - 30
      - 180
      - 166.10
      - 5.60
      - 162.65
      - 150.53
      - 140.81
      - 0.24
      - --
      - 0.087
      - 0
      - 140
      - 16
    * - B
      - 10
      - 60
      - 169.42
      - 6.40
      - 166.62
      - 154.12
      - --
      - 2.69
      - 7.2
      - 20
      - 7
      - 140
      - 8
    * - B
      - 10
      - 120
      - 169.24
      - 6.40
      - 166.41
      - 153.75
      - --
      - 2.50
      - 7.1
      - 20
      - 7
      - 119
      - 10
    * - B
      - 10
      - 180
      - 169.16
      - 6.40
      - 166.40
      - 153.69
      - 140.42
      - 2.14
      - 7.1
      - 33
      - 0
      - 159
      - 3
    * - B
      - 15
      - 60
      - 166.29
      - 5.80
      - 162.00
      - 151.84
      - --
      - 0.70
      - 1.0
      - 2.2
      - 8
      - 222
      - 18
    * - B
      - 15
      - 120
      - 166.36
      - 5.80
      - 162.54
      - 152.12
      - 139.47
      - 0.59
      - 1.0
      - 2.2
      - 0
      - 141
      - 3
    * - B
      - 15
      - 180
      - 166.39
      - 5.80
      - 162.57
      - 152.08
      - 139.44
      - 0.57
      - 1.0
      - 2.2
      - 0
      - 145
      - 4
    * - B
      - 20
      - 60
      - 166.07
      - 5.60
      - 162.27
      - 150.90
      - 142.31
      - 0.51
      - 0.92
      - 0.031
      - 2
      - 238
      - 11
    * - B
      - 20
      - 120
      - 166.24
      - 5.60
      - 162.67
      - 150.58
      - 141.10
      - 0.25
      - 0.21
      - 0.032
      - 0
      - 159
      - 8
    * - B
      - 20
      - 180
      - 166.20
      - 5.60
      - 162.70
      - 150.65
      - 140.94
      - 0.25
      - 0.14
      - 0.032
      - 0
      - 143
      - 7
    * - B
      - 25
      - 60
      - 166.02
      - 5.60
      - 162.11
      - 150.72
      - 142.14
      - 0.53
      - 0.82
      - 0.0059
      - 1
      - 189
      - 13
    * - B
      - 25
      - 120
      - 166.19
      - 5.60
      - 162.52
      - 150.42
      - 140.90
      - 0.23
      - 0.10
      - 0.0037
      - 0
      - 141
      - 10
    * - B
      - 25
      - 180
      - 166.15
      - 5.60
      - 162.56
      - 150.49
      - 140.75
      - 0.23
      - 0.015
      - 0.0037
      - 0
      - 140
      - 10
    * - B
      - 30
      - 60
      - 166.02
      - 5.60
      - 162.11
      - 150.69
      - 142.13
      - 0.53
      - 0.82
      - 0.0040
      - 2
      - 239
      - 28
    * - B
      - 30
      - 120
      - 166.19
      - 5.60
      - 162.53
      - 150.41
      - 140.99
      - 0.24
      - 0.13
      - 0.0038
      - 0
      - 137
      - 15
    * - B
      - 30
      - 180
      - 166.14
      - 5.60
      - 162.57
      - 150.49
      - 140.77
      - 0.23
      - --
      - 0.0038
      - 0
      - 141
      - 15

.. list-table:: Convergence study, other parameters, varied one at a time
                from the reference discretisation `L_{tsl} = 24` mm,
                `m_{tsl} = 25`, `n_x = 180`, `n_{tsl} = 10`, `n_y = 30`,
                `m = n = 8`. Same quantities and units as above. The halved
                increment gives 53 openings instead of 38.
    :header-rows: 1

    * - case
      - variation
      - `P_{max}`
      - `\delta(P_{max})`
      - `P(6)`
      - `P(7)`
      - `P(8)`
      - `CV`
      - `e_P`
      - `e_{eq}`
      - bisections
      - iterations
      - time
    * - A
      - reference
      - 166.10
      - 5.60
      - 162.65
      - 150.54
      - 140.78
      - 0.23
      - 0.018
      - 0.087
      - 0
      - 139
      - 10
    * - A
      - `n_{tsl} = 15`
      - 166.10
      - 5.60
      - 162.64
      - 150.54
      - 140.77
      - 0.23
      - 0.020
      - 0.087
      - 0
      - 136
      - 25
    * - A
      - `n_y = 60`
      - 166.10
      - 5.60
      - 162.63
      - 150.52
      - 140.77
      - 0.24
      - 0.021
      - 0.085
      - 0
      - 136
      - 12
    * - A
      - `m = n = 12`
      - 166.10
      - 5.60
      - 162.66
      - 150.54
      - 140.78
      - 0.24
      - 0.018
      - 0.086
      - 0
      - 134
      - 13
    * - A
      - `\Delta\delta` halved
      - 166.13
      - 5.70
      - 162.65
      - 150.54
      - 140.78
      - 0.15
      - 0.018
      - 0.087
      - 0
      - 184
      - 13
    * - A
      - `L_{tsl} = 17`
      - 166.08
      - 5.60
      - 162.57
      - 150.49
      - 140.74
      - 0.24
      - 0.064
      - 32
      - 0
      - 138
      - 11
    * - A
      - `L_{tsl} = 17`, `m_{tsl} = 15`, `n_x = 120`
      - 166.10
      - 5.60
      - 162.66
      - 150.38
      - 140.67
      - 0.25
      - 0.11
      - 33
      - 0
      - 142
      - 3
    * - B
      - reference
      - 166.15
      - 5.60
      - 162.56
      - 150.49
      - 140.75
      - 0.23
      - 0.015
      - 0.0037
      - 0
      - 140
      - 10
    * - B
      - `n_{tsl} = 15`
      - 166.14
      - 5.60
      - 162.56
      - 150.49
      - 140.73
      - 0.23
      - 0.028
      - 0.0054
      - 0
      - 136
      - 26
    * - B
      - `n_y = 60`
      - 166.14
      - 5.60
      - 162.55
      - 150.49
      - 140.75
      - 0.24
      - 0.015
      - 0.0042
      - 0
      - 143
      - 12
    * - B
      - `m = n = 12`
      - 166.14
      - 5.60
      - 162.57
      - 150.49
      - 140.75
      - 0.23
      - 0.015
      - 0.0036
      - 0
      - 140
      - 13
    * - B
      - `\Delta\delta` halved
      - 166.21
      - 5.70
      - 162.56
      - 150.49
      - 140.75
      - 0.15
      - 0.015
      - 0.0037
      - 0
      - 176
      - 13
    * - B
      - `L_{tsl} = 17`
      - 166.15
      - 5.60
      - 162.60
      - 150.48
      - 140.78
      - 0.24
      - 0.083
      - 10
      - 0
      - 142
      - 11
    * - B
      - `L_{tsl} = 17`, `m_{tsl} = 15`, `n_x = 120`
      - 166.12
      - 5.60
      - 162.63
      - 150.25
      - 140.64
      - 0.28
      - 0.25
      - 11
      - 0
      - 142
      - 3

The results show that:

- **The Ritz basis of the cohesive domain sets the shape of the response after
  the peak.** The cohesive domain spans `\lambda L_{tsl} = 20.0` (A) and 23.8
  (B) decay lengths. With `m_{tsl} = 10`, 0.50 and 0.42 terms per decay length,
  the basis cannot follow the process zone: the load overshoots the converged
  curve up to 168.2--169.4 N at `\delta = 6.2`--6.4 mm, 1.3--2.0% above the
  converged peak, and then drops by up to 14 N (A) and 19 N (B) in one
  increment, a sawtooth with `CV = 2.0`--2.7. The load differs from the
  converged one by 5--7%, `e_{eq}` is 19--47% from the first increment on,
  and four of the six runs stop at `\delta = 6.2`--7.4 mm. With `m_{tsl} =
  15` the curve is smooth, `CV = 0.51`--0.70, but still 0.7--1.2% off. From
  `m_{tsl} = 20` the error is at most 0.21% for `n_x \geq 120`, and 25 and 30
  terms give the same loads to 0.02% with `n_x = 180`. Against the terms per
  decay length, `m_{tsl}/(\lambda L_{tsl})`, the errors of both cases fall on
  one curve: about 1% at 0.63--0.75, at most 0.14% from 0.84 and below 0.1%
  from 1.0, with `n_x = 180`. The rule `m_{tsl} = 1.25\,\lambda L_{tsl}` keeps
  a margin over this threshold; with at most 25 terms it limits the cohesive
  domain of case A to `25/(1.25\lambda) = 24.0` mm. The Bardell functions go
  up to 30 terms, so a cohesive domain longer than `30/(1.0\lambda)` is not
  resolved.
- **The Gauss points along** `x` **set the robustness and a floor of the
  error.** `n_x = 60`, 120 and 180 over 24 mm correspond to 3.6, 7.1 and 10.7
  points per `l_{cz}`. With `n_x = 60` the error stays at 0.6--0.9% however
  many terms are used, 1 to 3 increments of every run with `m_{tsl} \geq 20`
  needed bisection, and the iterations increased 1.3 to 2.8 times, because
  the crack front crosses several integration points in one increment; the
  run B with `m_{tsl} = 15` stopped at `\delta = 7.8` mm. With `n_x = 120`
  the error is 0.10--0.21%, and with `n_x = 180` at most 0.02% for `m_{tsl}
  \geq 25`, without bisections. This gives the rule `n_x = 10\,L_{tsl}/l_{cz}`,
  168 for the reference DCB, rounded up to 170. Unlike the terms, the
  integration points do not change the shape of the curve: with `n_x = 60`
  the curves are smooth and the peak is within 0.15%.
- **The other parameters do not matter here.** `n_{tsl} = 15`, `n_y = 60`
  and `m = n = 12` change the loads by at most 0.03%, since the specimen is
  uniform across the width. Halving the increment after `\delta = 5` mm
  changes only where the peak is sampled: the peak rises by 0.03 N (A) and
  0.06 N (B) and moves to `\delta = 5.7` mm. The stiffer interface of case B
  raises the initial stiffness by 1.2%, but the peak and the loads after it
  differ from those of case A by less than 0.06%, in line with the
  insensitivity of the DCB to `k_o > 10^4` N/mm³ reported by Turon et al.
  (2007).
- **The length of the cohesive domain is detected by** `e_{eq}`, **not by the
  load.** At the last opening the crack has grown by 9.8 mm, and the margin
  `3 l_{cz} + 8/\lambda` for the process zone and the region ahead of it,
  where the opening decays as `e^{-\lambda x}`, the solution of beams on an
  elastic foundation [kanninen1973AugmentedDCB]_, to below `e^{-8}` of its
  value, is 13.9 mm (A) and 12.4 mm (B), so `L_{tsl} = 24` mm contains
  both up to `\delta = 8` mm and `e_{eq}` stays below 0.09% (A) and 0.004%
  (B), the larger value of case A coming from its longer decay length. With
  `L_{tsl} = 17` mm the loads are within 0.06--0.25% of the converged ones,
  but from `\delta = 7.0` mm (A) and 7.4 mm (B), when the crack has grown by
  6--7.5 mm, the compressive lobe enters the perfect bond and `e_{eq}` grows
  to 10--33% at the last opening.
- **The sawtooth is a discrete snap-back.** With too few terms the basis
  cannot follow the process zone continuously, like a cohesive finite element
  mesh that is coarse compared with the cohesive zone [alfano2001Interface]_
  [turon2007MeshSize]_: the elastic energy builds up while the crack front is
  held and is released when a group of integration points fails at once,
  which a displacement-controlled analysis crosses as a sudden load drop. More
  terms refine the displacement field and remove the sawtooth, while more
  integration points refine the damage field, remove the bisections and
  lower the floor of the error, but do not change the shape of the curve.

The converged discretisation of the reference DCB is therefore `L_{tsl} = 24`
mm, `m_{tsl} = 25`, `n_x = 180`, `n_{tsl} = 10`, `n_y = 30`, with `8 \times
8` terms in the other domains, which gives `P_{max} = 166.10` N (A) and
166.15 N (B) at `\delta = 5.6` mm. The rules for other geometries are:

.. math::
    :label: cz-rules

    m_{tsl} = 1.25\,\lambda L_{tsl}, \qquad
    n_x = 10\,\frac{L_{tsl}}{l_{cz}}, \qquad
    L_{tsl} \geq \Delta a + 3 l_{cz} + \frac{8}{\lambda}

with `\Delta a` the crack growth up to the last opening, `m_{tsl} \leq 25`,
`n_x \geq 4 m_{tsl}` rounded up to a multiple of 10, `n_{tsl} = 10`, `n_y =
30` and `m = n = 8` in the other domains. They are implemented in
:func:`.tsl_resolution`, :func:`.tsl_margin` and :func:`.tsl_discretization`,
and :meth:`.MultiDomain.calc_k_dmg` prints a warning, the first time a point
is damaged, when `m_{tsl} < 1.0\,\lambda L_{tsl}` or `n_x < 7\,L_{tsl}/l_{cz}`.
Before version 0.13.11 the rate was `(k_o/4D_{11})^{1/4}`, that of one arm on
a rigid foundation, `2^{-1/4}` times the rate of the symmetric DCB, and the
factors of the rules were 1.5 and 1.2 instead of 1.25 and 1.0. The margin
was `2\pi/\lambda + 3 l_{cz}`, i.e. `7.47/\lambda + 3 l_{cz}` with the
corrected rate; `2\pi/\lambda + 3 l_{cz}` with the corrected rate lets the
tractions reach the perfect bond in the last openings of the validation runs,
with `e_{eq}` up to 10%, against 0.25% at most with `3 l_{cz} + 8/\lambda`.
The Gauss-Legendre points and weights are computed by
:func:`scipy.special.roots_legendre`, without an upper limit for `n_x`.

**Discretisation of the validation runs.** The validation runs below use the
rules of Eq. :eq:`cz-rules` with 25 terms and the corresponding `L_{tsl} =
25/(1.25\lambda)`, and the range of openings is limited such that the crack
growth plus the margin fits in `L_{tsl}`. The low-strength case of Alfano and
Crisfield needs a cohesive domain over the whole bonded length, `L_{tsl} = 70`
mm, for which the first rule asks for 94 terms with `k_o = 10^5` N/mm³, more
than the 30 Bardell functions. It is therefore run with `k_o = 10^3` N/mm³,
for which `m_{tsl} = 30 = 1.27\,\lambda L_{tsl}` and `n_x = 4 m_{tsl} = 120`
meet both rules. The damage onset, `\Delta_o = \tau_o/k_o = 1.7 \times
10^{-3}` mm, remains far below the separation at failure, `\Delta_f =
2G_{Ic}/\tau_o = 0.33` mm, so the softening branch that governs this case is
hardly affected; Alfano and Crisfield also scaled the penalty stiffness with
the strength. A run with `k_o = 10^5` N/mm³ and the same 30 terms, `0.4\,
\lambda L_{tsl}`, gives a peak of 50.81 N against 50.78 N at the same
opening, and curves that differ by at most 0.7% of the peak, but `e_{eq}`
reaches 1.5% with `k_o = 10^5` N/mm³, as in the runs above with too few
terms, against less than `10^{-5}` with `k_o = 10^3` N/mm³.


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
and collected by ``notebooks/dcb_mode_I_validation_summary.ipynb``. The results were
obtained with panels 0.13.11.

The case of Camanho et al. (2003) is kept for reference, but it is not used
to validate the model: the FE and experimental load-opening curves of the
reference are 12--21% less stiff than the corrected beam theory, more
than transverse shear and root rotation explain, and the reference does not
give enough detail to find out why; the experimental peak, 147.1 N, is 6%
above the corrected beam theory with the same `G_{Ic}`; and the FE model has
about one decohesion element per cohesive zone length, which overpredicts the
peak load [turon2007MeshSize]_. See the note in
``notebooks/dcb_camanho2003.ipynb``.

**Model.** ``notebooks/dcb_utils.py`` builds a symmetric DCB: CLPT arms, far
end clamped, arm tips opened to `\pm\delta/2` by prescribed displacements,
the load being the reaction of the prescribed displacement. The bonded region
is split into a cohesive domain (``'SB_TSL'``, bilinear law) of length
`L_{tsl}` ahead of the initial crack front and a perfect bond (``'SB'``) over
the rest, such that only the region reached by the crack needs the resolution
of the cohesive zone. A weld narrower than the specimen (Tijs 2023) is
represented by starting the integration points outside the bonded strip fully
damaged. The solution procedure is the one described in
:ref:`cohesive_zone_solution`. The discretisation follows the rules of
Eq. :eq:`cz-rules` (``dcb_utils.discretize``): `m_{tsl} = \lceil
1.25\,\lambda L_{tsl} \rceil` with at most 25 terms, and `n_x =
10\,L_{tsl}/l_{cz}`, but not less than `4\,m_{tsl}` to integrate the
products of the approximation functions, rounded up to a multiple of 10, and
`L_{tsl}` covers the crack growth plus a margin of `3\,l_{cz} + 8/\lambda`
for the process zone and the decaying opening ahead of it; the range of
openings is limited so that `L_{tsl}` stays resolved, which allows 4--11 mm
of crack growth. The penalty stiffness is `k_o = 10^5` N/mm³, except for the
low-strength case of Alfano and Crisfield (see the convergence study): a
longer decay length `1/\lambda` needs fewer terms, and for the DCB of Turon
et al. (2007) the peak load with the `k_o = 10^6` N/mm³ of their model
differs by 0.2% (64.13 against 64.02 N), in line with their Fig. 9. The reference curves of linear elastic fracture mechanics (LEFM) are
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
                mid-width; the load at which the width-average energy release
                rate reaches `G_{Ic}` is 2.6% (UD) and 12.3% (D±30) above it. The
                two Krueger cases are listed with `\tau_o` = 40 MPa, the
                reference giving no strength; with 60 MPa the peaks are 64.2
                and 110.0 N. `^{k}`: `k_o = 10^3` N/mm³, see the convergence
                study; with `10^5` N/mm³ the peak is 50.8 N at the same
                opening. `^{n}`: kept for reference, not used to validate the
                model, see above. Tijs 2022, `a_0` = 40 mm: ``panels`` and
                CBT with `G_{Ic}` = 0.95 N/mm, the propagation toughness of
                the test without fibre bridging.
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
      - 64.5
      - 1.85
      - 61.5 `^{d}`
      - +4.8
      - --
      - --
      - 63.2
      - +2.1
      - +17
      - 9.0
    * - Alfano 2001, Table I, `t_o` = 1.7 MPa
      - 50.8
      - 2.90
      - 49.5 `^{d}`
      - +2.6
      - --
      - --
      - 63.2
      - -19.6
      - -25
      - 6.0
    * - Alfano 2001, XAS-913C
      - 93.6
      - 1.87
      - 90.0 `^{d}`
      - +4.0
      - --
      - --
      - 91.6
      - +2.1
      - +17
      - 9.1
    * - Camanho 2003, AS4/PEEK `^{n}`
      - 140.1
      - 4.09
      - 155.3 `^{t}`
      - -9.8
      - 147.1 `^{t}`
      - -4.8
      - 138.9
      - +0.9
      - +14
      - 7.4
    * - Turon 2007, T300/977-2
      - 64.0
      - 4.16
      - 57.0 `^{d}`
      - +12.3
      - 62.0 `^{d}`
      - +3.3
      - 63.3
      - +1.2
      - +12
      - 11.1
    * - Krueger 2008, UD `[0]_{24}`
      - 63.5
      - 1.43
      - 60.7 `^{b}`
      - +4.6
      - --
      - --
      - 62.3
      - +2.0
      - +18
      - 8.5
    * - Krueger 2008, D`\pm`30
      - 108.7
      - 1.70
      - 102.4 `^{b}`
      - +6.1
      - --
      - --
      - 114.7
      - -5.2
      - +10
      - 7.1
    * - Tijs 2022, `a_0` = 48 mm
      - 166.2
      - 5.66
      - 167.0 `^{d}`
      - -0.5
      - 142.0 `^{d}`
      - +17.1
      - 164.5
      - +1.1
      - +14
      - 10.1
    * - Tijs 2022, `a_0` = 40 mm, 0.95 N/mm
      - 182.1
      - 3.71
      - --
      - --
      - 161.0 `^{d}`
      - +13.1
      - 179.1
      - +1.6
      - +17
      - 10.6
    * - Tijs 2023, full weld, 2.1 N/mm
      - 148.0
      - 6.41
      - 155.0 `^{d}`
      - -4.5
      - 160.0 `^{m}`
      - -7.5
      - 155.5
      - -4.8
      - +4
      - 4.9
    * - Tijs 2023, weld center, 2.1 N/mm
      - 119.2
      - 5.82
      - 128.0 `^{d}`
      - -6.9
      - 129.0 `^{m}`
      - -7.6
      - 123.0
      - -3.1
      - +8
      - 5.5
    * - Tijs 2023, weld center, 1.95 N/mm
      - 115.0
      - 5.58
      - 122.0 `^{d}`
      - -5.7
      - 123.0 `^{d}`
      - -6.5
      - 118.5
      - -3.0
      - +8
      - 5.8
    * - Tijs 2023, weld center, 1.12 N/mm
      - 87.9
      - 4.13
      - 96.0 `^{d}`
      - -8.5
      - --
      - --
      - 89.8
      - -2.2
      - +9
      - 8.2

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
  `\tau^0` = 87 MPa). ``panels`` predicts 166.2 N at 5.66 mm against 167 N at
  5.7 mm, -0.5% on the load and -0.7% on the opening.
- **Cohesive zone beyond LEFM.** With the low strength `t_o` = 1.7 MPa of
  Alfano and Crisfield (2001) the process zone is of the order of the
  specimen length and the peak of their interface model is 22% below the
  corrected beam theory; ``panels`` reproduces it within 2.6%, at the same
  opening and with the same initial stiffness (+2.3%), with `k_o = 10^3`
  N/mm³ and the cohesive domain over the whole bonded length (see the
  convergence study).
- **Nominal strengths.** For the unidirectional specimens with nominal
  strengths, the ``panels`` peak is within 2.1% of the corrected beam theory
  in every case, and after the peak it follows the LEFM propagation branch
  (50.7 N, 0.04% below it, at `\delta` = 3.00 mm for Alfano and Crisfield,
  Table I). The FE results of the references scatter by about `\pm 10\%` around the
  corrected beam theory: 1 mm decohesion elements for a cohesive zone of 1.35
  mm in Camanho et al. (2003), whose FE peak is above both the test and the
  beam theories, and about 57 N in Turon et al. (2007), below both.
- **Tests.** The test of Turon et al. (2007), with a bilinear-like response,
  is predicted within 3.3% (and that of Camanho et al., kept for reference,
  within -4.8%). The AS4D/PEKK-FC tests show a
  strong R-curve (initiation about 0.7 N/mm, propagation 1.12 N/mm, and 0.95
  N/mm without fibre bridging for the test with `a_0` = 40 mm, the
  toughness of its ``panels`` run) that a
  bilinear law with the propagation toughness cannot represent: ``panels``,
  the bilinear FE model of Tijs et al. and the corrected beam theory are all
  11--18% above these tests, which the tabular law of the paper matches.
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
- **Robustness.** All 17 nonlinear runs converged at every opening without
  bisection, in 1 to 8 iterations, and the load balances the integral of
  the cohesive tractions within 0.25% (`e_{eq}`), except for the full weld
  (7.6%), whose perfect bond spans the whole width, and the low-strength case
  of Alfano and Crisfield with `k_o = 10^5` N/mm³ (1.5%), whose 30 terms are
  only `0.4\,\lambda L_{tsl}`.

**Krueger (2008).** NASA/TM-2008-215123 [krueger2008DCB]_ gives two mode I DCB
benchmarks, a unidirectional `[0]_{24}` specimen of T300/1076 and a
multidirectional D±30 specimen of C12K/R6376, built with 3D solid elements and
the virtual crack closure technique, with no cohesive law, no interfacial
strength and no experiment. Their critical load is the one at which `G_T`
reaches `G_c` *at mid-width*, while ``panels`` and beam theory follow the
width average `\bar{G} = P_{crit}^2 a_e^2/(b\,EI)`; the two criteria differ
by `\sqrt{G_c/\bar{G}}`, 1.026 and 1.123, which is the whole offset between
the benchmark and the corrected beam theory. ``panels`` is 2.0% above the
corrected beam theory for the unidirectional specimen and 5.2% below it for
the multidirectional one, whose cohesive zone (`l_{cz}` = 1.99 mm with
`\tau_o` = 40 MPa) is not small compared with its 2.0 mm arms. Raising
`\tau_o` from 40 to 60 MPa moves the peaks by 1.1 and 1.2% only, so the
comparison does not rest on that choice. See
``notebooks/dcb_krueger2008.ipynb``.

**Krueger (2012).** NASA/CR-2012-217562 [krueger2012MMB]_ gives benchmarks
for the mixed-mode bending specimen of IM7/8552 at `G_{II}/G_T` = 0.2, 0.5 and
0.8, with the critical points (1.64 mm, 128.5 N), (1.34 mm, 385 N) and (1.65
mm, 751 N). They are recorded in ``notebooks/krueger2012_mmb_benchmark.ipynb``
but not simulated: the damage of ``'SB_TSL'`` is driven by the normal
separation only, and the loading lever is not modelled.


Differences that remain
-----------------------

- The interpenetration stiffness `k_{ipen}` of Eq. 5.12 of the thesis is not
  implemented: :meth:`.MultiDomain.calc_k_dmg` returns `k_o(1-d)` also for
  negative separations, such that a fully damaged point offers no resistance
  to interpenetration. This does not affect a monotonically loaded DCB.
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
