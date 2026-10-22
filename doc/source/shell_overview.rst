.. _shell_overview:

Plates and cylindrical shells
=============================

A :class:`.Shell` is one rectangular domain of a plate, `0 \le x \le a`,
`0 \le y \le b`, or of a cylindrical shell of radius `r`, with `y` the arc
length along the circumference. It holds the inputs of the model and
calculates the structural matrices and force vectors, which are solved with
`structsolve <https://github.com/saullocastro/structsolve>`_. The same
class is the building block of the assemblies :class:`.MultiDomain` and
:class:`.StiffPanelBay`.


Inputs
------

**Geometry and model.** The dimensions ``a``, ``b`` and the radius ``r``
(``None`` for a plate). The attribute ``model`` selects the kinematics and
the non-linear equations, see :mod:`panels.models`; when it is not given, a
plate gets ``'plate_clpt_donnell'`` and a cylinder
``'cylshell_clpt_sanders'``. The models based on the classical laminated
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
and ``ny`` of the numerically integrated matrices, ``2*m`` and ``2*n`` by
default. The boundary conditions are the flags of the edges, see
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
constants `\{c\}`, :meth:`.Shell.get_size`:

======================================= =================================== ==========================
quantity                                 method                               used by
======================================= =================================== ==========================
constitutive stiffness `[K_C]`           :meth:`.Shell.calc_kC`               all analyses
geometric stiffness `[K_G]`              :meth:`.Shell.calc_kG`               buckling, prestress
mass `[M]`                               :meth:`.Shell.calc_kM`               frequency, flutter
aerodynamic stiffness and damping        :meth:`.Shell.calc_kA`,              flutter
                                         :meth:`.Shell.calc_cA`
external force vector `\{F_{ext}\}`      :meth:`.Shell.calc_fext`             static analyses
internal force vector `\{F_{int}\}`      :meth:`.Shell.calc_fint`             non-linear analyses
tangent stiffness `[K_T]`                :meth:`.Shell.calc_kT`               non-linear analyses
product `[K_T]\{v\}`                     :meth:`.Shell.calc_kTv`              iterative solvers
load stiffness of follower pressures     :meth:`.Shell.calc_kCfollower`       follower loads
======================================= =================================== ==========================

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

- linear static: ``solve(kC, fext)``, see :doc:`ex_linear_static`;
- linear buckling: ``lb(kC, kG)``, with `[K_G]` of the constant stress
  resultants ``Nxx``, ``Nyy``, ``Nxy`` or of a static solution ``c``, see
  :doc:`ex_linear_buckling`;
- frequency: ``freq(kC, kM)``, with an optional prestress `[K_G]`, see
  :doc:`ex_frequency`;
- panel flutter: ``freq(kC + kG + kA, kM)``, see :ref:`aeroelasticity`;
- non-linear static: :class:`structsolve.Analysis` with ``calc_fext``,
  ``calc_fint``, ``calc_kC`` and ``calc_kG``, with the Newton-Raphson or
  the arc-length methods, see :doc:`ex_newton_raphson` and
  :doc:`ex_arc_length`;
- follower pressures, in all the analyses above, see
  :ref:`follower_pressure`.


Results
-------

The fields at points or over a grid are calculated from the Ritz constants
with :meth:`.Shell.uvw`, :meth:`.Shell.strain` and :meth:`.Shell.stress`,
plotted with :func:`.plot_shell`, and the shell with its results is saved
with :meth:`.Shell.save`, see :ref:`postprocessing`.
