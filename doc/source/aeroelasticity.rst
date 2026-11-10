.. _aeroelasticity:

Panel flutter with the piston theory
====================================

The supersonic flutter of plates, cylindrical panels and stiffened panels
is analysed with the aerodynamic load of the linear piston theory, a
pressure on `w` proportional to the slope of `w` along the flow, with the
parameter `\beta`, plus, for a flow along the axis of a cylinder, a
curvature correction proportional to `w`, with the parameter `\gamma`. The
flow is along `x` or `y`, the attribute ``flow``. The formulation follows
Castro et al. [castro2016FlutterPanel]_, where the stiffener bases of a
stiffened panel are structural elements of the panel, as in
:class:`.StiffPanelBay`, with the sign of the curvature correction
discussed below.


Aerodynamic load
----------------

For a flow along `x`, the aerodynamic load on the shell along `w`, `w`
being positive outwards for a cylinder, is

.. math::
    :label: aero-piston

    q = \beta w_{,x} + \gamma w + \mu_a \dot{w}

and, for a flow along `y`, `q = \beta w_{,y} + \mu_a \dot{w}`, without the
curvature correction. With the Mach number `M`, the air density
`\rho_{air}`, the air speed `V` and the speed of sound `a_\infty`, the
parameters are, Eq. (18) of [castro2016FlutterPanel]_:

.. math::
    :label: aero-parameters

    \beta = \frac{\rho_{air} V^2}{\sqrt{M^2 - 1}}, \qquad
    \gamma = \frac{\beta}{2 r \sqrt{M^2 - 1}}, \qquad
    \mu_a = \frac{\beta}{M a_\infty} \frac{M^2 - 2}{M^2 - 1}

see :meth:`.StiffPanelBay._aero_parameters`. The term `\gamma w` is the
correction of Krumhaar [krumhaar1963Piston]_ for the external flow over a
cylinder of radius `r`: an outward displacement lowers the pressure on the
shell, which is pulled outwards, a softening effect that does not depend on
the direction of the flow; `\gamma = 0` for the plates. The linear piston
theory holds for supersonic flows, `M > 1`, and `\beta` is singular at `M =
1`: a ``Mach`` equal to ``1`` is changed to ``1.0001`` with a warning, and a
lower one raises an error.

The parameters ``beta`` and ``gamma`` are given directly as attributes, or
calculated from ``Mach``, ``rho_air``, ``air_speed`` and the radius when
``beta`` is ``None``, see :meth:`.Shell.calc_kA` and
:meth:`.StiffPanelBay.calc_kA`; the damping coefficient `\mu_a` is the
argument ``aeromu`` of :meth:`.Shell.calc_cA`.


Aerodynamic matrices
--------------------

With `\boldsymbol{S}^w` the row of the approximation functions of `w`,
`w = \boldsymbol{S}^w \boldsymbol{c}`, the virtual work of the load of
Eq. :eq:`aero-piston` gives, for a flow along `x`, the aerodynamic
stiffness and damping matrices

.. math::
    :label: aero-matrices

    \boldsymbol{K}_A = -\int_A \left(\beta (\boldsymbol{S}^w_{,x})^\top
    \boldsymbol{S}^w + \gamma (\boldsymbol{S}^w)^\top \boldsymbol{S}^w\right) dA,
    \qquad
    \boldsymbol{C}_A = -\mu_a \int_A (\boldsymbol{S}^w)^\top \boldsymbol{S}^w \, dA

with `\boldsymbol{S}^w_{,y}` and `\gamma = 0` for a flow along `y`. They are
calculated by:

- :meth:`.Shell.calc_kA` and :meth:`.Shell.calc_cA`, for one domain, the
  latter returning `i \boldsymbol{C}_A`;
- :meth:`.StiffPanelBay.calc_kA` and :meth:`.StiffPanelBay.calc_cA`, the
  sum over the panels of the skin of a bay, `\boldsymbol{K}_A = \sum_p
  \boldsymbol{K}_A^{(p)}`, the stiffeners not contributing;
- in a :class:`.MultiDomain` assembly, by adding the matrices of the
  domains exposed to the flow, ``calc_kA(size=size, row0=p.row_start,
  col0=p.col_start)``, as :func:`.tstiff2d_1stiff_flutter` does for the
  skin.

The closed-form matrices of the kernels of :mod:`panels.models` are used,
or the numerically integrated ones for a partial domain. The term in
`\gamma` is symmetric. Integrating by parts, the term in `\beta` is
skew-symmetric when `w` vanishes along the edges normal to the flow, where
the boundary terms of `\int S_i S_{k,x} \, dx` vanish; otherwise it is
neither symmetric nor skew-symmetric. `\boldsymbol{K}_A` is therefore never
symmetrized. Since the panels of a :class:`.StiffPanelBay` share the
approximation functions of the bay, each one integrated over its own strip,
`\boldsymbol{K}_A` of the bay does not change when the skin is divided into
panels.

The term in `\beta` of Eq. :eq:`aero-matrices` is the transpose of that of
Eq. (21) of [castro2016FlutterPanel]_: with `w = 0` at the edges normal to
the flow the two differ by the sign of `\beta`, i.e. by the direction of the
flow, on which the flutter boundary does not depend. The term in `\gamma`
of Eqs. (18), (21) and (29) of [castro2016FlutterPanel]_ has the opposite
sign, which stiffens the shell; ``panels`` keeps the softening term of
Krumhaar's correction, see :meth:`.Shell.calc_kA`.


Flutter analysis
----------------

`\boldsymbol{K}_A` is added to the stiffness matrix of a frequency
analysis, here without damping,

.. math::
    :label: aero-flutter

    \left(\boldsymbol{K}_C + \boldsymbol{K}_G + \boldsymbol{K}_A + \lambda^2 \boldsymbol{M}\right) \boldsymbol{c} = \boldsymbol{0}

solved with :func:`structsolve.freq`, which handles unsymmetric matrices,
where `\boldsymbol{K}_G` is the geometric stiffness of an optional prestress
and `\lambda^2 = -\omega^2` for the harmonic motion `\boldsymbol{c} e^{i
\omega t}`. Increasing `\beta`, two natural frequencies approach each
other and coalesce at the flutter onset, where the eigenvalues become
complex. With the aerodynamic damping, `\dot{w} = i \omega w`, the problem
becomes quadratic in `\omega`,

.. math::
    :label: aero-flutter-damped

    \left(\boldsymbol{K}_C + \boldsymbol{K}_G + \boldsymbol{K}_A - \omega^2 \boldsymbol{M} + i \omega \boldsymbol{C}_A\right) \boldsymbol{c} = \boldsymbol{0}

Eqs. (29)-(31) of [castro2016FlutterPanel]_, which is solved, e.g., in the
state-space form of their Eq. (32); the factor `i` is already in the matrix
returned by :meth:`.Shell.calc_cA`.

The flutter analysis applies to one domain, with and without prestress, to
a :class:`.StiffPanelBay`, and to assemblies such as the panel with a
T-stiffener of :func:`.tstiff2d_1stiff_flutter`, whose stiffener is made of
plate domains connected to the skin, see :ref:`multidomain_overview`. Only
the domains whose matrices are added are loaded by the flow: the
stiffeners of a bay are not.
