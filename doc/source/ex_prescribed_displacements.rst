.. _ex_prescribed_displacements:

Prescribed displacements
========================

The prescribed displacements of :meth:`.Shell.add_point_pd`,
:meth:`.Shell.add_distr_pd_fixed_x` and :meth:`.Shell.add_distr_pd_fixed_y`
are imposed by a :class:`.MultiDomain`, exactly with
``pd_method='null-space'`` or approximately with ``pd_method='penalty'``,
see :ref:`boundary_conditions` and :ref:`null_space_pd`. The plates below
have the in-plane displacements free at the edges and are compressed by an
end shortening:

.. literalinclude:: ../../tests/multidomain/test_null_space_pd.py
    :pyobject: plate

.. literalinclude:: ../../tests/multidomain/test_null_space_pd.py
    :pyobject: end_shortening

A single shell is wrapped in an assembly without connections, ``conn=[]``,
and :meth:`.MultiDomain.linear_static` returns the configuration with the
prescribed displacements, here the uniform strain of the patch test:

.. literalinclude:: ../../tests/multidomain/test_null_space_pd.py
    :pyobject: test_single_shell_without_conn

A cylinder of five domains under a uniform end shortening gives the same
prebuckling state as the axial load of the corresponding reaction,
:meth:`.MultiDomain.reaction_resultant`, and a slightly higher critical
load; with the penalty method, the springs of :func:`.fkCld_xcte` are added
by the user to the stiffness matrix, and the critical load converges to the
one of the null-space method as the penalty constant increases:

.. literalinclude:: ../../tests/multidomain/test_null_space_pd.py
    :pyobject: cylinder

.. literalinclude:: ../../tests/multidomain/test_null_space_pd.py
    :pyobject: test_cylinder_end_shortening

The geometrically non-linear analyses under displacement control are shown
in :ref:`null_space_pd`. Without an assembly, a point of a :class:`.Shell`
is supported by the penalty springs of
:meth:`.Shell.calc_stiffness_point_constraint`, added to the stiffness
matrix:

.. literalinclude:: ../../tests/tests_shell/test_shell_api.py
    :pyobject: test_stiffness_point_constraint_fixes_the_point
