.. _stiffened_panels:

Stiffened panels
================

``panels`` models stiffened panels in two ways:

- a :class:`.StiffPanelBay`, a skin with blade stiffeners
  :class:`.BladeStiff1D` or :class:`.BladeStiff2D` along `x`, whose skin
  panels and stiffener bases share the Ritz constants of the skin, a compact
  model for the global buckling, vibration and flutter of the bay;
- a :class:`.MultiDomain` assembly, in which the skin, the bases and the
  flanges are plate domains connected to each other, e.g. the panel with a
  T-stiffener of :mod:`panels.multidomain.tstiff2d_1stiff_compression`, see
  :ref:`multidomain_overview`, which also represents debonding defects
  between the skin and the base, following Castro and Donadon
  [castro2017Multidomain]_.


Stiffened panel bay
-------------------

The bay is a rectangular plate or cylindrical panel, `0 \le x \le a`, `0
\le y \le b`, with the attributes ``a``, ``b``, ``r``, ``model``, the
laminate (``stack``, ``plyt``, ``laminaprop``, ``rho``), the numbers of
terms ``m`` and ``n``, and the flags of the boundary conditions of its four
edges, see :ref:`boundary_conditions`, by default with `u`, `v` and their
slopes and `w` removed on all edges and the slopes of `w` free. Only the
models with 3 degrees of freedom per term, based on the classical laminated
plate theory, are supported.

The skin is divided along `y` into panels with
:meth:`.StiffPanelBay.add_panel`, each a :class:`.Shell` that spans the
whole bay with the approximation functions and the boundary conditions of
the bay, but is integrated only over its strip `y_1 \le y \le y_2`, see
:meth:`.Shell.integration_limits`; each panel may have its own laminate and
stress resultants ``Nxx``, ``Nyy``, ``Nxy``. The displacement field of the
skin is therefore continuous without any connection, and the matrices of
the bay are the sums of those of the panels and of the stiffeners.

The stiffeners are added after the panels, at a coordinate `y_s`, with
:meth:`.StiffPanelBay.add_bladestiff1d` or
:meth:`.StiffPanelBay.add_bladestiff2d`. Both may have a base, a
:class:`.Shell` bonded below the skin with its offset, using the Ritz
constants of the skin, and a flange:

- :class:`.BladeStiff1D`: the flange is a beam along `x` whose
  displacements are those of the skin along `y = y_s`, which adds no
  degree of freedom, and whose local buckling is not represented;
- :class:`.BladeStiff2D`: the flange is a plate with its own ``mf`` by
  ``nf`` terms, connected to the base, or to the skin, by penalty
  stiffnesses, which represents its local buckling, see
  :meth:`.StiffPanelBay.get_size`.

The matrices :meth:`.StiffPanelBay.calc_kC`,
:meth:`.StiffPanelBay.calc_kG`, :meth:`.StiffPanelBay.calc_kM`,
:meth:`.StiffPanelBay.calc_kA` and :meth:`.StiffPanelBay.calc_cA`, and the
vector :meth:`.StiffPanelBay.calc_fext` are solved with ``structsolve`` as
those of a :class:`.Shell`, e.g. ``lb(bay.calc_kC(), bay.calc_kG())``. The
displacements of the skin and of the stiffeners are calculated with
:meth:`.StiffPanelBay.uvw_skin` and :meth:`.StiffPanelBay.uvw_stiffener`
and plotted with :meth:`.StiffPanelBay.plot_skin` and
:meth:`.StiffPanelBay.plot_stiffener`. The flutter of the bay with the
piston theory is described in :ref:`aeroelasticity`.

The construction of a bay and the linear buckling analyses are shown in
:doc:`ex_stiffpanelbay`.
