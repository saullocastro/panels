.. _multidomain_overview:

Multi-domain assemblies
=======================

A :class:`.MultiDomain` assembles several :class:`.Shell` domains, each one
with its own approximation functions, see :ref:`theory_func_bardell`, and
imposes the compatibility of their displacements along edges or over areas,
following Castro and Donadon [castro2017Multidomain]_. Assemblies model
cylinders and panels divided in domains, stiffened panels whose stiffeners
are made of plate domains, panels with debonding defects and, with cohesive
connections, delaminations.


Assembly
--------

The domains are given as a list, and the vector of the Ritz constants of the
assembly is the concatenation of those of the domains, in that order: the
constructor sets the attributes ``row_start``, ``col_start``, ``row_end``
and ``col_end`` of each domain. The matrices of the assembly,
:meth:`.MultiDomain.calc_kC`, :meth:`.MultiDomain.calc_kG`,
:meth:`.MultiDomain.calc_kM`, :meth:`.MultiDomain.calc_kT`, and the vectors,
:meth:`.MultiDomain.calc_fext`, :meth:`.MultiDomain.calc_fint`, are the sums
of those of the domains at these positions, see :ref:`shell_overview`, plus
the connections. The boundary conditions and the loads are those of each
domain. The attribute ``group`` of the domains collects them for the plots
and the field outputs, and ``x0``, ``y0`` place a domain in the plane of its
group, see :ref:`postprocessing`.


Connections
-----------

Each connection is a dictionary of the list ``conn``, with the two domains
``p1`` and ``p2``, the key ``func`` and the position of the connection in
each domain:

============= ============================================================ ===================================
``func``       connection                                                    position
============= ============================================================ ===================================
``'SSxcte'``   two skins along an edge of constant `x`                       ``xcte1``, ``xcte2``
``'SSycte'``   two skins along an edge of constant `y`                       ``ycte1``, ``ycte2``
``'BFycte'``   base, or skin, ``p1`` and flange ``p2`` at constant `y`       ``ycte1``, ``ycte2``
``'BFxcte'``   base, or skin, ``p1`` and flange ``p2`` at constant `x`       ``xcte1``, ``xcte2``
``'SB'``       skin and base over their common area                          ``p1`` on the positive side of `z`
``'SB_TSL'``   ``'SB'`` with a cohesive traction-separation law              as ``'SB'``
============= ============================================================ ===================================

The kernels and the conventions of each connection are described in
:mod:`panels.multidomain.connections`. The connections of the models with
5 degrees of freedom per term, FSDT and TSDT, are those of
:mod:`panels.multidomain.connections.kCsdt`, and two connected domains must
have the same number of degrees of freedom per term; ``'SB_TSL'`` is
available for the classical laminated plate theory only, see
:ref:`cohesive_zone`.


Imposing the connections
------------------------

With ``conn_method='null-space'``, the default, the connections are linear
constraints `[B]\{c\} = \{0\}`, imposed exactly with a basis `[Z]` of the
null space of `[B]`, :meth:`.MultiDomain.get_Z`:

.. math::

    \{c\} = [Z]\{c_r\}, \qquad [K_r] = [Z]^T [K] [Z], \qquad
    \{f_r\} = [Z]^T \{f\}

The matrices and vectors are reduced with :meth:`.MultiDomain.reduce`
before solving, and the solution is expanded with
:meth:`.MultiDomain.expand`; :meth:`.MultiDomain.get_reduced_functions`
gives the reduced functions for the non-linear analyses of
:class:`structsolve.Analysis`. The theory and the verification are in
:ref:`null_space`.

With ``conn_method='penalty'``, penalty stiffnesses are added to the
stiffness matrix by :meth:`.MultiDomain.get_kC_conn`, with the constants
``'kt'`` and ``'kr'`` of each connection, by default those of
:func:`.calc_kt_kr`, and the matrices are solved directly. The continuity
is then approximate, with an error that decreases with the penalty
constants, while the conditioning of the stiffness matrix worsens.

The prescribed displacements of the domains are imposed with
``pd_method``, see :ref:`boundary_conditions`.
:meth:`.MultiDomain.linear_static` and :meth:`.MultiDomain.linear_buckling`
solve the reduced problems, with the prescribed displacements when
``pd_method='null-space'``, in which case
:meth:`.MultiDomain.reaction_pd` and :meth:`.MultiDomain.reaction_resultant`
give the reactions.


Ready-made assemblies
---------------------

The functions of ``panels.multidomain`` build and analyse complete
assemblies, and show how to build others:

- :func:`.create_cylinder`, :func:`.create_cylinder_blade_stiffened` and
  the analyses of :mod:`panels.multidomain.cylinder` and
  :mod:`panels.multidomain.cylinder_blade_stiffened`, see
  :doc:`ex_multidomain_cylinder`;
- the panel with one T-stiffener of
  :mod:`panels.multidomain.tstiff2d_1stiff_compression`,
  :mod:`panels.multidomain.tstiff2d_1stiff_freq` and
  :mod:`panels.multidomain.tstiff2d_1stiff_flutter`, see
  :doc:`ex_multidomain_tstiff2d`.


Example
-------

A square plate divided in two or four domains, simply supported on the
outer edges, with the connections ``'SSxcte'`` and ``'SSycte'``:

.. literalinclude:: ../../tests/multidomain/test_multidomain_fsdt_tsdt.py
    :pyobject: domains

The deflection of the plate under a sinusoidal pressure, solved with the
null-space method, reduced and expanded back, agrees with the solution of
Pagano (1970) given in the test:

.. literalinclude:: ../../tests/multidomain/test_null_space.py
    :pyobject: test_bending_divided_plate_pagano

Two plates in series under tension, connected with the penalty method, and
the section force computed by :meth:`.MultiDomain.force`:

.. literalinclude:: ../../tests/multidomain/test_multidomain_postprocessing.py
    :pyobject: tension

.. literalinclude:: ../../tests/multidomain/test_multidomain_postprocessing.py
    :pyobject: test_force_equals_applied_load
