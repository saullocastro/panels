.. _cohesive_zone_mixed_mode:

Mixed-mode multilinear cohesive interface ``'CZM'``
===================================================

The connection ``'CZM'`` bonds two domains over their common area with a
mixed-mode cohesive law, for all the models of :mod:`panels.modelDB`, and is
the basis of the delamination coupons of
:mod:`panels.multidomain.delamination`, the double cantilever beam (DCB),
the end-notched flexure (ENF) and the mixed-mode bending (MMB) test. It
differs from the mode I connection ``'SB_TSL'`` of :ref:`cohesive_zone` in
the following points:

- the damage is driven by the normal and the tangential separations, through
  the mixed-mode law of Turon et al. (2018), with a penalty stiffness that
  depends on the mode and the Benzeggagh-Kenane criterion;
- the law can be the superposition of several bilinear laws, each one with
  its own exponent and damage variable, e.g. a crack-tip law and fibre
  bridging laws active in mode I only;
- the separations are those of the perfect bond ``'SB'`` for every model,
  including the first- and third-order shear deformation theories;
- the normal stiffness is kept for a negative normal separation (contact),
  and a contact-only interface can join the arms over a pre-crack;
- each interface has its own history, and several interfaces can evolve
  independently in the same assembly;
- the tangent stiffness matrix is the consistent one, and the interfaces enter
  :meth:`.MultiDomain.calc_fint`, :meth:`.MultiDomain.calc_kC`,
  :meth:`.MultiDomain.calc_kT` and :meth:`.MultiDomain.calc_kTv`, so that the
  reduced functions of the null-space method,
  :meth:`.MultiDomain.get_reduced_functions`, contain them.

Usage
-----

A connection ``'CZM'`` is a dictionary with ``p1`` (the domain on the
positive side of the interface along `z`), ``p2``, the list of laws, and the
numbers of Gauss-Legendre points::

    from panels.multidomain.connections.czm import BilinearLaw, law_with_ratio

    law = BilinearLaw(GIc=0.969, GIIc=1.719, tauI=80., tauII=100., KI=1e5,
                      eta=2.284)
    conn = [..., dict(p1=top, p2=bot, func='CZM', laws=[law],
                      nr_x_gauss=100, nr_y_gauss=6)]
    md = MultiDomain(panels, conn)

The shear stiffness of a :class:`.czm.BilinearLaw` is, unless given, the one
of Eq. (1) of Turon et al. (2018), which keeps the damage from decreasing
when the mode ratio changes. Superposed laws are best built with
:func:`.czm.law_with_ratio` and a common ratio of the stiffnesses, see
:mod:`panels.multidomain.connections.czm`.

The history is updated only with converged states, by
:meth:`.MultiDomain.commit_cohesive`. With ``structsolve`` it is done by the
callable of :meth:`.MultiDomain.get_on_converged`, given to
``Analysis.on_converged``::

    from structsolve import Analysis
    an = Analysis(*md.get_reduced_functions())
    an.on_converged = md.get_on_converged()
    an.static(NLgeom=True)

The coupons of :func:`.delamination.create_coupon` build the whole model,
including the lever of the MMB apparatus, a linear constraint of
:meth:`.MultiDomain.add_linear_constraint`, and
:func:`.delamination.solve_coupon` traces the load-displacement curve::

    from panels.multidomain.delamination import create_coupon, solve_coupon
    md, info = create_coupon('MMB', laminaprop, plyt, [0]*12, b=25.4,
                             a0=34.1, laws=[law], L=50.8, c=44.4,
                             model='plate_fsdt_donnell', nd=7, L_coh=67.5)
    res, an = solve_coupon(md, info, delta_max=8.)

Options for speed
-----------------

With the defaults, the plate kernels keep the tables of the approximation
functions across the iterations, see :ref:`tabulation`, and evaluate the
functions by Horner's rule, see :ref:`theory_func_bardell`: the results
differ from those of panels 0.17.0 at round-off level only. Three options,
off by default, give the same increments, the same Newton-Raphson
iterations and the same loads, to the round-off, in less time; they matter
for the coupons, whose tangent stiffness matrix is assembled and factorized
at every iteration:

- ``create_coupon(..., plate_kernels='blas')``, or ``MultiDomain(...,
  plate_kernels='blas')``: the plate matrices and vectors as products of
  dense matrices, see :ref:`plate_blas`;
- ``solve_coupon(..., reuse_kT=True)``, or
  :meth:`.MultiDomain.get_reduced_functions` with ``reuse_kT=True``: the
  load vector of the prescribed displacement, `\boldsymbol{K}_T
  \boldsymbol{c}_{p1}`, from the tangent stiffness matrix already assembled
  at the same configuration;
- ``solve_coupon(..., linear_solver='pardiso_lu')``, with oneMKL, or
  ``'dense_lu'`` otherwise, both of ``structsolve>=0.13.0``: the
  factorization of the tangent stiffness matrix, forwarded to
  ``structsolve.Analysis``, see the `linear solvers of structsolve
  <https://saullocastro.github.io/structsolve/linear_solvers.html>`_.

::

    md, info = create_coupon('MMB', laminaprop, plyt, [0]*12, b=25.4,
                             a0=34.1, laws=[law], L=50.8, c=44.4,
                             model='plate_fsdt_donnell', nd=7, L_coh=67.5,
                             plate_kernels='blas')
    res, an = solve_coupon(md, info, delta_max=8., reuse_kT=True,
                           linear_solver='pardiso_lu')

``tests/multidomain/test_plate_blas.py`` compares a DCB coupon with these
options to the defaults.

Verification
------------

``tests/multidomain/test_czm.py`` and ``tests/multidomain/test_delamination.py``
verify that:

- along a path of constant mode ratio up to complete failure, the work of the
  tractions equals the sum of the Benzeggagh-Kenane toughnesses of the
  superposed laws, to `10^{-10}`;
- along paths that change the mode ratio, with unloading and closure, no
  damage variable decreases, and the dissipated energy lies between the sums
  of the pure-mode toughnesses;
- the consistent tangent equals central finite differences of the
  tractions;
- the separation operators of ``'CZM'`` equal those of ``'SB_TSL'`` for the
  classical laminated plate theory, and a pristine interface tends to the
  perfect bond ``'SB'`` as the stiffness grows, for CLPT and FSDT plates;
- the lever constraint transmits the line loads of the statics of a rigid
  lever.

API
---

.. automodule:: panels.multidomain.connections.czm
    :members:

.. automodule:: panels.multidomain.delamination
    :members:
