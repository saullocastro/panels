Non-linear static analysis with the Newton-Raphson method
=========================================================

The geometrically non-linear models, based on Donnell's equations, provide the
internal force vector `\boldsymbol{F}_{int}`, calculated with :meth:`.Shell.calc_fint`,
and the tangent stiffness matrix

.. math::

    \boldsymbol{K}_T = \boldsymbol{K}_C + \boldsymbol{K}_G

calculated with :meth:`.Shell.calc_kT`, or with :meth:`.Shell.calc_kC` and
:meth:`.Shell.calc_kG` using ``NLgeom=True``. The tangent stiffness matrix is
the exact derivative of the internal force vector, such that the
Newton-Raphson iterations converge quadratically.

The example below applies a compression of about 37 times the first linear
buckling load in a single step, deep in the postbuckling regime, starting from
the linear solution, and verifies the order of convergence of the iterations:

.. literalinclude:: ../../tests/tests_shell/test_nonlinear.py
    :pyobject: test_nonlinear

The consistency between the tangent stiffness matrix and the internal force
vector is verified by a directional Taylor test, where the error of the linear
approximation of `\boldsymbol{F}_{int}` must decrease quadratically with the step size:

.. literalinclude:: ../../tests/tests_shell/test_tangent_consistency.py
    :pyobject: test_tangent_is_derivative_of_fint

The product of the tangent stiffness matrix by a vector, `\boldsymbol{K}_T \boldsymbol{v}`, is
available without assembling `\boldsymbol{K}_T`, at the cost of one evaluation of the
internal force vector, with :meth:`.Shell.calc_kTv` and
:meth:`.MultiDomain.calc_kTv`, see :ref:`tangent_product`.

The methods ``calc_fext``, ``calc_fint``, ``calc_kC`` and ``calc_kG`` of
:class:`.Shell` and :class:`.MultiDomain` have the signatures required by
:class:`structsolve.Analysis`, which implements the Newton-Raphson method with
load control and the arc-length methods, see :doc:`ex_arc_length`.
