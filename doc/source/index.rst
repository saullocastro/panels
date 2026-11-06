Semi-analytical methods for plates, shells and stiffened panels - panels
========================================================================

The ``panels`` module provides semi-analytical (Ritz) models for plates,
cylindrical shells and stiffened panels, using
:ref:`Bardell's hierarchical functions <theory_func_bardell>` as approximation
functions. The structural matrices and force vectors are solved with
`structsolve <https://github.com/saullocastro/structsolve>`_. With ``panels``
you can run:

* Linear static analyses, with field outputs of displacements, strains and
  stresses

* Linear buckling analyses, with the pre-buckling stress state given by
  constant stress resultants or calculated with a static analysis

* Frequency analyses, with or without pre-stress

* Panel flutter analyses, using the piston theory

* Geometrically non-linear static analyses based on Donnell's equations, with
  a tangent stiffness matrix that is the exact derivative of the internal force
  vector, using:

  - the Newton-Raphson method
  - the arc-length methods of Riks and Crisfield, able to trace limit points
    and snap-through

* Multi-domain analyses, connecting shell domains exactly with the
  :ref:`null-space method <null_space>`, the default, or with penalty
  stiffnesses, to model cylinders,
  stiffened panels and panels with debonding defects, also with the shear
  deformation theories

The models are identified by the type of structure, the kinematic theory and
the non-linear equations, see :mod:`panels.models`:

* ``'plate_clpt_donnell'``: plates, classical laminated plate theory (CLPT)
* ``'plate_clpt_gl'``: plates, CLPT with the Green-Lagrange membrane strains,
  which vanish for any rigid rotation, for assemblies with large rotations,
  see :ref:`plate_clpt_gl`
* ``'plate_clpt_tl'``: plates, total-Lagrangian Kirchhoff-Love, with the
  Green-Lagrange membrane strains and the curvatures of the deformed normal,
  for plates bent through large rotations, see :ref:`plate_clpt_tl`;
  integrated exactly by default, at about twice the cost of
  ``'plate_clpt_gl'``, or with the points of ``'plate_clpt_gl'``,
  ``nx = gauss_points(m)``, ``ny = gauss_points(n)``, not exact but with the
  same results in practice at about half the cost
* ``'plate_fsdt_donnell'``: plates, first-order shear deformation theory
* ``'plate_tsdt_donnell'``: plates, third-order shear deformation theory of
  Reddy
* ``'cylshell_clpt_donnell'``: cylindrical shells, CLPT with Donnell's
  kinematics
* ``'cylshell_clpt_sanders'``: cylindrical shells, CLPT with Sanders-Koiter's
  kinematics, which give zero strains for rigid-body motions
* ``'cylshell_fsdt_donnell'``, ``'cylshell_tsdt_donnell'``: cylindrical
  shells, first-order and third-order shear deformation theories with
  Donnell's kinematics
* ``'cylshell_fsdt_sanders'``, ``'cylshell_tsdt_sanders'``: cylindrical
  shells, first-order and third-order shear deformation theories with
  Sanders-Koiter's kinematics


How the models, the boundary conditions, the loads and the assemblies fit
together is described in the user guide, starting with
:ref:`shell_overview`.


Code repository
---------------

https://github.com/saullocastro/panels


User guide
----------

.. toctree::
    :maxdepth: 1

    shell_overview.rst
    bardell.rst
    boundary_conditions.rst
    loads.rst
    multidomain_overview.rst
    stiffened_panels.rst
    aeroelasticity.rst
    postprocessing.rst
    follower_pressure.rst
    plate_clpt_gl.rst
    plate_clpt_tl.rst
    tsdt_offset.rst
    multidomain_null_space.rst
    tangent_product.rst
    tabulation.rst
    cohesive_zone.rst
    pyodide.rst


.. toctree::
    :maxdepth: 1
    :caption: Worked examples

    notebooks/ex_linear_static.nblink
    notebooks/ex_linear_buckling.nblink
    notebooks/ex_frequency.nblink
    notebooks/ex_flutter.nblink
    notebooks/ex_newton_raphson.nblink
    notebooks/ex_arc_length.nblink
    notebooks/ex_follower_pressure.nblink
    notebooks/ex_prescribed_displacements.nblink
    notebooks/ex_face_clamp.nblink
    notebooks/ex_multidomain_cylinder.nblink
    notebooks/ex_multidomain_tstiff2d.nblink
    notebooks/ex_multidomain_wingbox.nblink
    notebooks/ex_stiffpanelbay.nblink


.. toctree::
    :maxdepth: 1
    :caption: Tutorials

    notebooks/pagano1970_plates.nblink
    notebooks/varadan1991_cylinders.nblink
    notebooks/noor1975_buckling.nblink
    notebooks/kardomateas1993_thick_rings.nblink
    notebooks/kardomateas1993b_1995_thick_cylinders.nblink
    notebooks/kardomateas2003_sandwich_rings.nblink
    notebooks/han2004_sandwich_rings.nblink
    notebooks/nasa_sp8007_external_pressure.nblink
    notebooks/schweizerhof1984_pressure_loads.nblink
    notebooks/schweizerhof1984_cantilever_postbuckling.nblink
    notebooks/sze2004_hinged_roofs.nblink
    notebooks/sabir1972_roof_snap_back.nblink
    notebooks/large_rotation_plates.nblink
    notebooks/null_space_vs_penalty.nblink
    notebooks/tangent_stiffness_product.nblink
    notebooks/tsdt_offset_connections.nblink
    notebooks/fk1952_shallow_arches.nblink
    notebooks/simo1990_cylinder_postbuckling.nblink
    notebooks/stamatelos_labeas_2023.nblink
    notebooks/stamatelos_labeas_2023_multidomain.nblink
    notebooks/sewall1968_blade_cylinder_vibration.nblink
    notebooks/egle1968_few_stringer_cylinder_vibration.nblink
    notebooks/card1966_blade_cylinder_buckling.nblink
    notebooks/weller1977_blade_cylinder_buckling.nblink
    notebooks/peterson1959_blade_cylinder_buckling.nblink
    notebooks/abramovich2009_blade_panels.nblink
    notebooks/lanzi2003_torsion_cylinder.nblink
    notebooks/dcb_alfano2001.nblink
    notebooks/dcb_camanho2003.nblink
    notebooks/dcb_turon2007.nblink
    notebooks/dcb_krueger2008.nblink
    notebooks/dcb_tijs2022.nblink
    notebooks/dcb_tijs2023_phd.nblink
    notebooks/dcb_lecinana2023.nblink
    notebooks/dcb_mode_I_validation_summary.nblink
    notebooks/dcb_mode_I_convergence_study.nblink
    notebooks/dcb_mode_I_solver_study.nblink
    notebooks/cohesive_zone_verification.nblink
    notebooks/krueger2012_mmb_benchmark.nblink
    notebooks/conn_method_comparison.nblink
    notebooks/test_quadrature_partial_integration_matrix.nblink
    notebooks/test_quadrature_partial_integration_of_domain.nblink


panels API
----------

.. toctree::
    :maxdepth: 2

    api.rst


Installing panels
-----------------

Install from the distributed packages by simply doing::

    python -m pip install panels

or from the source code, which requires a C++ compiler to build the Cython
extensions, using::

    python -m pip install .

In the browser, with `Pyodide <https://pyodide.org>`_ 314 (CPython 3.14),
the WebAssembly wheel published on PyPI is installed with ``micropip``, see
:ref:`pyodide_page`.


.. toctree::
    :maxdepth: 1
    :caption: Changelog

    changelog.rst


.. toctree::
    :maxdepth: 1
    :caption: About this documentation

    documentation_levels.rst


License
-------

.. literalinclude:: ../../LICENSE
    :encoding: latin-1


Citing this library
-------------------

Saullo G. P. Castro, Nathan D'Souza. (2026). Semi-analytical methods for plates, shells and stiffened panels (Version 0.15.0). Zenodo. DOI: https://doi.org/10.5281/zenodo.2541522.

In BibTeX:

.. code-block:: bibtex

    @misc{castro_panels,
      author    = {Castro, Saullo G. P. and D'Souza, Nathan},
      title     = {Semi-analytical methods for plates, shells and stiffened
                   panels},
      year      = {2026},
      version   = {0.15.0},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.2541522},
      url       = {https://doi.org/10.5281/zenodo.2541522}
    }

The DOI represents all the versions and resolves to the latest one. The
repository also has a ``CITATION.cff`` file with the same data.


References
----------

.. toctree::
    :maxdepth: 1

    ref.rst


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
