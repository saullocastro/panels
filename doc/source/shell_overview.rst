.. _shell_overview:

Plates and cylindrical shells
=============================

A :class:`.Shell` is one rectangular domain of a plate, `0 \le x \le a`,
`0 \le y \le b`, or of a cylindrical shell of radius `r`, with `y` the arc
length along the circumference. It holds the inputs of the model and
calculates the structural matrices and force vectors, which are solved with
`structsolve <https://github.com/saullocastro/structsolve>`_. The same
class is the building block of the assemblies :class:`.MultiDomain` and
:class:`.StiffPanelBay`. The package :mod:`panels` gathers these classes;
this page states the conventions shared by all of them, the kinematics of
the models and how to choose one.


.. _shell_conventions:

Conventions
-----------

**Coordinates.** `x` runs along the edge of length `a` and, for a
cylinder, along its axis; `y` along the edge of length `b`, the arc length
`y = r \theta` for a cylinder; `z` is normal to the reference surface,
`z = 0`, and `w` is the displacement along `z`, positive outwards for a
cylinder. The shallow shell assumption `r + z \approx r` is adopted, and
`1/r = 0` for the plates. The laminate spans `d - h/2 \le z \le d + h/2`,
`d` being the offset ``offset``, `d = 0` when the reference surface is the
mid-surface, the convention of :func:`composites.laminated_plate`. The
natural coordinates of the approximation functions are `\xi = 2x/a - 1` and
`\eta = 2y/b - 1`.

**Ritz constants.** Each field `F` of `u, v, w` (and `\phi_x, \phi_y` for
the shear deformation theories), of index `k`, is the series

.. math::
    :label: shell-ritz

    F(x, y) = \sum_{j = 0}^{n - 1} \sum_{i = 0}^{m - 1}
    c_{N_d (j m + i) + k} \, P_i^F(\xi) \, P_j^F(\eta)

with the `N_d = 3` or 5 degrees of freedom per term in the order `u, v,
w, \phi_x, \phi_y`, and Bardell's functions `P_i^F`, `P_j^F` with the
boundary flags of `F` along `x` and `y`, see :ref:`theory_func_bardell`.
The vector `\boldsymbol{c}` has the size `N_d \, m \, n`,
:meth:`.Shell.get_size`, and every matrix and vector of a :class:`.Shell`
has its rows and columns in this order. In an assembly, the constants of
each domain start at its ``row_start``, see :ref:`multidomain_overview`.

**Generalized strains and stresses.** The constitutive matrix
``Shell.ABD`` relates the generalized stresses ``Nxx, Nyy, Nxy, Mxx, Myy,
Mxy`` (and ``Qy, Qx``) to the generalized strains ``exx, eyy, gxy, kxx, kyy,
kxy`` (and ``gyz, gxz``), in this order, with the higher-order terms of the
TSDT appended, see :func:`.fsdt_tsdt_field.strain_names`. The changes of
curvature carry the sign `\kappa_{xx} = -w_{,xx}`, and the engineering
shear strains are used.

**Units.** No unit system is assumed: lengths, forces and densities must be
given in one consistent set of units, in which the results are returned.


.. _shell_kinematics:

Kinematics of the models
------------------------

The models of :mod:`panels.models`, selected with ``Shell.model``, differ
in the displacement field through the thickness and in the non-linear
terms of the strains. The integrands of their matrices are derived
symbolically from these kinematics and coded in the modules of
:mod:`panels.models`, see :ref:`shell_models_kernels`.

**Classical laminated plate theory (CLPT)**, 3 degrees of freedom per
term, `u(z) = u - z w_{,x}`, `v(z) = v - z w_{,y}` for the Donnell
kinematics. With `s = 0` for the Donnell kinematics (the von Kármán
strains for the plates) and `s = 1` for the Sanders-Koiter kinematics
[sanders1963Nonlinear]_, and the rotations of the normal `\beta_x =
-w_{,x}` and `\beta_y = -w_{,y} + s v/r`, the strains are

.. math::
    :label: shell-clpt-strains

    \varepsilon_{xx} &= u_{,x} + \frac{1}{2} \beta_x^2, \qquad
    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \beta_y^2, \qquad
    \gamma_{xy} = u_{,y} + v_{,x} + \beta_x \beta_y \\
    \kappa_{xx} &= -w_{,xx}, \qquad
    \kappa_{yy} = -w_{,yy} + s \frac{v_{,y}}{r}, \qquad
    \kappa_{xy} = -2 w_{,xy} + \frac{s}{r} \left( \frac{3}{2} v_{,x}
    - \frac{1}{2} u_{,y} \right)

the changes of curvature of the Sanders-Koiter theory being those written
by Amabili [amabili2008Nonlinear]_, Sec. 1.3.8. With `s = 1` all the
strains vanish for any small rigid-body motion, which matters for long
shells and for deformations of long wavelength, e.g. low circumferential
wave numbers. The non-linear terms omit the rotation about the normal,
`\frac{1}{2}(v_{,x} - u_{,y})`, a simplification of the Sanders-Koiter
theory for small strains and moderately small rotations, which keeps the
geometric stiffness of an axial load equal to that of the Donnell
kinematics, see :mod:`panels.models`. The plate models ``'plate_clpt_gl'``
and ``'plate_clpt_tl'`` replace the membrane strains by the Green-Lagrange
strains, and the latter also the changes of curvature by those of the
deformed normal, see :ref:`plate_clpt_gl` and :ref:`plate_clpt_tl`.

The cylinder ``'cylshell_clpt_gl'`` keeps the changes of curvature of the
Sanders-Koiter kinematics and replaces the membrane strains by the complete
Green-Lagrange strains `\frac{1}{2}(\boldsymbol{F}^\top \boldsymbol{F} -
\boldsymbol{I})` of the mid-surface, with the deformation gradient in the
local basis

.. math::
    :label: shell-cyl-gl-F

    \boldsymbol{F} = \begin{bmatrix} 1 + u_{,x} & u_{,y} \\ v_{,x} & 1 +
    v_{,y} + w/r \\ w_{,x} & w_{,y} - v/r \end{bmatrix}

the strains of the Flügge-Lur'e-Byrne non-linear theory as given by Amabili
[amabili2008Nonlinear]_, Sec. 1.3.6. They vanish for any rigid motion and
contain the rotation about the normal: an axial load `N_{xx}` has the
geometric stiffness `N_{xx} (u_{,x}^2 + v_{,x}^2 + w_{,x}^2)`, instead of
`N_{xx} w_{,x}^2` with Eq. :eq:`shell-clpt-strains`. The difference matters
when `v_{,x}` is of the order of `w_{,x}`: when a long cylinder bends as a
beam, `w = W(x) \cos\theta`, `v = -W(x) \sin\theta`, the work of the axial
load on `v_{,x}^2` equals that on `w_{,x}^2`, and the Sanders-Koiter
kinematics give about twice the Euler load of a pinned tube. For
shell-like modes the difference decreases as the circumferential waves
shorten: for axially compressed panels with `R/h = 100` and `L/R = 2`
supported by shear diaphragms, the Green-Lagrange strains give a buckling
load 4.2 % below that of the Sanders-Koiter kinematics for a panel of 180
degrees and 1.0 % below for a panel of 40 degrees. Linear analyses, static
and natural frequencies, are those of ``'cylshell_clpt_sanders'``.

The cylinder ``'cylshell_clpt_tl'`` is the total-Lagrangian
Kirchhoff-Love counterpart of ``'plate_clpt_tl'``: the membrane strains of
``'cylshell_clpt_gl'`` and changes of curvature measured with the normal of
the deformed mid-surface `\boldsymbol{X}`. With its area vector
`\boldsymbol{a} = \boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y}`, whose
length is the area stretch `1 + \varepsilon_{xx} + \varepsilon_{yy} +
\ldots`:

.. math::
    :label: shell-cyl-tl-k

    \kappa_{xx} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,xx} \qquad
    \kappa_{yy} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,yy} - \frac{1}{r}
    - \frac{\varepsilon_{xx} + 2 \varepsilon_{yy}}{r} \qquad
    \kappa_{xy} = -2 \boldsymbol{a} \cdot \boldsymbol{X}_{,xy} -
    \frac{\gamma_{xy}}{2 r}

where the second derivatives of the position include those of the local
basis, e.g. the component of `\boldsymbol{X}_{,yy}` along the normal is
`w_{,yy} - 2 v_{,y}/r - w/r^2 - 1/r`. The terms `-\boldsymbol{a} \cdot
\boldsymbol{X}_{,\alpha\beta} + (1 + \varepsilon_{xx} + \varepsilon_{yy})
b^0_{\alpha\beta}`, with `b^0_{yy} = -1/r` the only non-zero component of
the second fundamental form `\boldsymbol{n} \cdot
\boldsymbol{X}_{,\alpha\beta}` of the undeformed cylinder, are the changes
of the second fundamental form [vanderheijden2008Koiter]_, Sec. 3.16,
times the area stretch; the terms `-\varepsilon_{yy}/r` and
`-\gamma_{xy}/(2r)` replace, in the linear part, the membrane strains by
the rotation about the normal, which gives the linear changes of curvature
of the Sanders-Koiter kinematics of Eq. :eq:`shell-clpt-strains`
[vanderheijden2008Koiter]_, Eq. (3.16.8). Changes of curvature that differ
by terms of the order of the membrane strain divided by the radius are
equivalent within the accuracy of the classical theory of thin shells
[vanderheijden2008Koiter]_, p. 202. All six strains are invariant under
any rigid rotation of any deformed state, whereas the linear changes of
curvature of ``'cylshell_clpt_gl'`` are exact for small rotations only:
the difference matters where a part of the shell rotates through a large
angle while it bends, e.g. a shallow panel under a point load that snaps
through and inverts its curvature. Up to the quadratic terms, the changes
of curvature differ from those of Novozhilov's theory as given by Amabili
[amabili2008Nonlinear]_, Sec. 1.3.7, by terms of the order of the membrane
strain divided by the radius or times the curvature, and by terms of the
rotation about the normal. The linear analyses and the linear buckling
analyses with a membrane prebuckling state are those of
``'cylshell_clpt_gl'``. The curvatures are cubic in the displacements, and
the exact integration takes `3p + 1` Gauss-Legendre points along each
direction, against `2p + 1` for the other models.

**First-order (FSDT) and third-order (TSDT) shear deformation theories**,
5 degrees of freedom per term, the rotations `\phi_x` and `\phi_y` being
independent fields:

.. math::
    :label: shell-sdt-field

    u(z) &= u + z \phi_x - c_1 \left[(z - d)^3 + d^3\right] (\phi_x + w_{,x}) \\
    v(z) &= v + z \left(\phi_y + s \frac{v}{r}\right)
            - c_1 \left[(z - d)^3 + d^3\right] (\phi_y + w_{,y}) \\
    w(z) &= w

with `c_1 = 0` for the FSDT and `c_1 = 4/(3 h^2)` for the TSDT of Reddy
[reddy1984HigherOrder]_, whose transverse shear strains `(\phi_x +
w_{,x})(1 - 4 (z - d)^2/h^2)` and `(\phi_y + w_{,y})(1 - 4 (z - d)^2/h^2)`
vanish at both faces, such that the TSDT needs no shear correction. The
FSDT uses a shear corrected transverse shear stiffness, by default that of
Rohwer, see ``fsdt_shear_correction`` of :class:`.Shell`. The terms in `s`
correct the cylinders with the Sanders-Koiter kinematics, as in the shell
theory of Reddy and Liu [reddy1985Shells]_ with `r + z \approx r`; the
strains are those of :mod:`panels.models.fsdt_tsdt_field`, and the TSDT
with an offset is described in :ref:`tsdt_offset`.


.. _shell_choosing:

How to choose a model
---------------------

========================================= ===============================================================
structure and behaviour                   model
========================================= ===============================================================
thin plate, moderate rotations            ``'plate_clpt_donnell'``, the default of the plates
thin cylinder or cylindrical panel        ``'cylshell_clpt_sanders'``, the default of the cylinders;
                                          ``'cylshell_clpt_donnell'`` for short wavelengths only
long cylinder bending as a beam, e.g. a   ``'cylshell_clpt_gl'``, also for low circumferential wave
tube in column buckling; curved panels    numbers and for curved panels and stiffener blades
or blades loaded in their own plane       loaded in their own plane
cylindrical panels or cylinders bent      ``'cylshell_clpt_tl'``, see below
through large rotations, e.g. shallow
panels snapping under point loads
plates of an assembly with large          ``'plate_clpt_gl'``; ``'plate_clpt_tl'`` when the plates also
rotations, e.g. a wing box                bend through large rotations, see :ref:`plate_clpt_gl`
plates loaded in their own plane that     ``'plate_clpt_gl'``, see below
bend in their plane when they buckle,
e.g. blade stringers of a stiffened shell
moderately thick plates and shells,       ``'plate_fsdt_donnell'``, ``'cylshell_fsdt_sanders'``
transverse shear flexibility              (or ``_donnell``)
thick laminates, offset reference         ``'plate_tsdt_donnell'``, ``'cylshell_tsdt_sanders'``
surface                                   (or ``_donnell``), see :ref:`tsdt_offset`
========================================= ===============================================================

**Plates bent in their own plane.** A blade stringer modelled as a plate
strip normal to the skin, or the web of a stiffener, carries its share of
the axial load as a membrane stress resultant `N_{xx}` of the plate, and
in a general instability of the stiffened structure it bends in its own
plane: the out-of-plane displacement of the skin at the joint is the
in-plane displacement `v` of the blade. The geometric stiffness is the
second variation of the work of the pre-buckling stresses on the quadratic
part of the membrane strains,

.. math::

    \delta^2 U_G = \int N_{xx} \left( \delta u_{,x}^2 + \delta v_{,x}^2
        + \delta w_{,x}^2 \right) dA

for a pre-buckling state `N_{xx}` alone,

with the Green-Lagrange strains of ``'plate_clpt_gl'``, whereas the von
Kármán strains of ``'plate_clpt_donnell'`` keep only `\delta w_{,x}^2`. A
blade of the von Kármán plate model is therefore loaded without
destabilizing its in-plane (for a stringer, radial) bending: the stringer
behaves as a column whose load does no work on its lateral deflection, and
the general instability load is overestimated, by a factor of two or more
when the stringers carry about half of the load. In the smeared
stiffener theories the total load of the skin and of the stringers enters
the term `\bar{N}_x w_{,x}^2` of the skin [block1965Stiffened]_, which is what
``'plate_clpt_gl'`` recovers for discrete blades. The two models have the
same linear stiffness, mass and aerodynamic matrices, so that linear static
and frequency analyses do not change, and the analytical geometric
stiffness of a constant `N_{xx}` costs the same. Skins and plates that
buckle out of their plane only, e.g. the skin between the stringers, are
not affected.

The thin-plate limit of the FSDT and the TSDT is the CLPT, see
:ref:`boundary_conditions`, at the cost of 5 instead of 3 degrees of
freedom per term. The models with 5 degrees of freedom per term are
supported by :class:`.MultiDomain`, but not by :class:`.StiffPanelBay`,
whose stiffeners assume the 3 degrees of freedom of the CLPT, nor by the
cohesive connection ``'SB_TSL'``, see :ref:`cohesive_zone`.


Inputs
------

**Geometry and model.** The dimensions ``a``, ``b`` and the radius ``r``
(``None`` for a plate). The attribute ``model`` selects the kinematics and
the non-linear equations, see :mod:`panels.models`; when it is not given, a
plate gets ``'plate_clpt_donnell'`` and a cylinder
``'cylshell_clpt_sanders'``; plates with large rotations, e.g. the domains
of a wing box, need ``'plate_clpt_gl'``, see :ref:`plate_clpt_gl`, or
``'plate_clpt_tl'`` when they also bend through large rotations, see
:ref:`plate_clpt_tl`, whose exact integration by default costs about twice
that of ``'plate_clpt_gl'``, and half with ``nx = gauss_points(m)``, ``ny =
gauss_points(n)``, not exact but enough in practice. The models based on the classical laminated
plate theory (CLPT) have the 3 degrees of freedom `u, v, w` per term, those
based on the shear deformation theories (FSDT, TSDT) the 5 degrees of
freedom `u, v, w, \phi_x, \phi_y`.

**Laminate.** The stacking sequence ``stack``, the ply thickness ``plyt``
and the lamina properties ``laminaprop``, or one value per ply with
``plyts`` and ``laminaprops``, the density ``rho`` (or ``rhos``) and the
offset of the laminate ``offset`` build the laminate ``Shell.lam`` with
:func:`composites.laminated_plate`, whose constitutive matrix ``Shell.ABD``
enters every kernel. A stiffness that varies over the domain is given
instead by ``ABDnxny``, one matrix per integration point. The transverse
shear stiffness of the FSDT is controlled by ``fsdt_shear_correction``, and
the TSDT with an offset is described in :ref:`tsdt_offset`.

**Approximation.** The numbers of terms ``m`` and ``n`` of the
approximation functions along `x` and `y`, at most 30, see
:ref:`theory_func_bardell`, and the numbers of integration points ``nx``
and ``ny`` of the numerically integrated matrices, by default
``max(2*m - 1, 7)`` and ``max(2*n - 1, 7)`` for the current ``m`` and ``n``,
the smallest numbers that integrate exactly the products of up to four
approximation functions, see :func:`.gauss_points`. The boundary conditions are the flags of the edges, see
:ref:`boundary_conditions`.

**Loads.** Point, line and pressure loads and the constant stress
resultants of a prestress, see :ref:`loads`.

The constructor sets the defaults, then the given attributes, and builds
the laminate when ``a`` and ``b`` are given; the calculation methods
rebuild the laminate and the size from the current attributes, see
:class:`.Shell`.


Matrices and vectors
--------------------

The size of the matrices and vectors is that of the vector of the Ritz
constants `\boldsymbol{c}`, :meth:`.Shell.get_size`:

============================================= =================================== ==========================
quantity                                      method                               used by
============================================= =================================== ==========================
constitutive stiffness `\boldsymbol{K}_C`     :meth:`.Shell.calc_kC`               all analyses
geometric stiffness `\boldsymbol{K}_G`        :meth:`.Shell.calc_kG`               buckling, prestress
mass `\boldsymbol{M}`                         :meth:`.Shell.calc_kM`               frequency, flutter
aerodynamic stiffness and damping             :meth:`.Shell.calc_kA`,              flutter
                                              :meth:`.Shell.calc_cA`
external force vector `\boldsymbol{F}_{ext}`  :meth:`.Shell.calc_fext`             static analyses
internal force vector `\boldsymbol{F}_{int}`  :meth:`.Shell.calc_fint`             non-linear analyses
tangent stiffness `\boldsymbol{K}_T`          :meth:`.Shell.calc_kT`               non-linear analyses
product `\boldsymbol{K}_T \boldsymbol{v}`     :meth:`.Shell.calc_kTv`              iterative solvers
load stiffness of follower pressures          :meth:`.Shell.calc_kCfollower`       follower loads
============================================= =================================== ==========================

The linear matrices of a constant constitutive matrix over the whole domain
are calculated in closed form; the others, and all of them when ``c``,
``ABDnxny`` or a partial integration domain is given, are integrated
numerically, see :ref:`theory_func_bardell`. The non-linear quantities
follow the Donnell or Sanders-Koiter kinematics of the model, with a tangent
stiffness matrix that is the exact derivative of the internal force vector,
see :ref:`tangent_product`. The integration can be limited to a part of the
domain with ``x1``, ``x2``, ``y1``, ``y2``, see
:meth:`.Shell.integration_limits`, which is how the panels of a
:class:`.StiffPanelBay` share one set of approximation functions.


Analyses
--------

The methods above have the signatures expected by the solvers of
``structsolve``:

- linear static: ``solve(kC, fext)``;
- linear buckling: ``lb(kC, kG)``, with `\boldsymbol{K}_G` of the constant stress
  resultants ``Nxx``, ``Nyy``, ``Nxy`` or of a static solution ``c``;
- frequency: ``freq(kC, kM)``, with an optional prestress `\boldsymbol{K}_G`;
- panel flutter: ``freq(kC + kG + kA, kM)``, see :ref:`aeroelasticity`;
- non-linear static: :class:`structsolve.Analysis` with ``calc_fext``,
  ``calc_fint``, ``calc_kC`` and ``calc_kG``, with the Newton-Raphson or
  the arc-length methods;
- follower pressures, in all the analyses above, see
  :ref:`follower_pressure`.


Results
-------

The fields at points or over a grid are calculated from the Ritz constants
with :meth:`.Shell.uvw`, :meth:`.Shell.strain` and :meth:`.Shell.stress`,
plotted with :func:`.plot_shell`, and the shell with its results is saved
with :meth:`.Shell.save`, see :ref:`postprocessing`.
