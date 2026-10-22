.. _postprocessing:

Field outputs, plots and files
==============================

The solvers return the Ritz constants `\{c\}`, from which the fields are
calculated at points or over a grid, plotted, and saved with the model.


Field outputs
-------------

=========================== ================================================ ==============================================
object                       displacements, strains, stress resultants        other outputs
=========================== ================================================ ==============================================
:class:`.Shell`              :meth:`.Shell.uvw`, :meth:`.Shell.strain`,
                             :meth:`.Shell.stress`
:class:`.MultiDomain`        :meth:`.MultiDomain.uvw`,                         :meth:`.MultiDomain.force`,
                             :meth:`.MultiDomain.strain`,                      :meth:`.MultiDomain.force_out_plane`,
                             :meth:`.MultiDomain.stress`,                      :meth:`.MultiDomain.reaction_pd`,
                             :meth:`.MultiDomain.calc_results`                 :meth:`.MultiDomain.reaction_resultant`
:class:`.StiffPanelBay`      :meth:`.StiffPanelBay.uvw_skin`,
                             :meth:`.StiffPanelBay.uvw_stiffener`
=========================== ================================================ ==============================================

The methods of :class:`.Shell` store the fields in the dictionary
``Shell.fields`` and the coordinates of the points in ``Shell.plot_mesh``;
the strains and stress resultants include the non-linear terms of the
strains of the model by default. Those of :class:`.MultiDomain` evaluate
the domains of a ``group``, or one domain ``eval_panel``, at a grid or at
the Gauss-Legendre points, and :meth:`.MultiDomain.force` integrates the
stress resultants along a line of constant `x` or `y`. The eigenvalues,
eigenvectors and Ritz constants of an analysis may be kept in the
dictionary ``Shell.results``, which is saved with the shell.


Plots
-----

- :func:`.plot_shell`: contour plot of one field of a :class:`.Shell`,
  with :mod:`matplotlib`;
- :meth:`.MultiDomain.plot`: contour plot of the domains of a group, each
  domain placed at ``x0``, ``y0`` in the plane of the group;
- :meth:`.MultiDomain.plot3d`: interactive 3D plot of the deformed
  assembly with plotly, which must be installed, each domain placed in the
  global coordinate system by ``x0``, ``y0``, ``z0``, ``point_x`` and
  ``point_xy``, see :meth:`.Shell.global_frame`;
- :meth:`.StiffPanelBay.plot_skin` and
  :meth:`.StiffPanelBay.plot_stiffener`: contour plots of a bay.


Saving and loading
------------------

:meth:`.Shell.save`, :meth:`.StiffPanelBay.save` and
:meth:`.MultiDomain.save` write a zip file with the inputs in JSON and the
arrays, e.g. the fields and the results, in NumPy's format, and
:func:`panels.shell.load`, :func:`panels.stiffpanelbay.load` and
:func:`panels.multidomain.load` read it back; the matrices are not stored
and are calculated again. File objects, e.g. :class:`io.BytesIO`, are
accepted in place of file names. The format and the attributes stored are
described in :mod:`panels.json_io`.


Examples
--------

A shell saved after a linear buckling and a static analysis, with the
eigenpairs, the Ritz constants and the stress resultants, gives the same
matrices and results once loaded:

.. literalinclude:: ../../tests/test_json.py
    :pyobject: _shell

.. literalinclude:: ../../tests/test_json.py
    :pyobject: test_shell_after_analysis

A contour plot of the bending moment ``Mxx`` of a :class:`.Shell` with the
options of :func:`.plot_shell`:

.. literalinclude:: ../../tests/tests_shell/test_plot_shell.py
    :pyobject: test_plot_shell_options
