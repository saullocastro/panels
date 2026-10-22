.. _loads:

Loads
=====

The loads of a :class:`.Shell` are stored as lists of its attributes and
converted into the external force vector `\{F_{ext}\}` by
:meth:`.Shell.calc_fext`, see :func:`.shell_fext`, as the virtual work of
the loads on the approximation functions. In an assembly, the loads are
added to its panels and :meth:`.MultiDomain.calc_fext` or
:meth:`.StiffPanelBay.calc_fext` assembles their vectors.


Types of loads
--------------

=============================================== ======================================================
load                                             method
=============================================== ======================================================
point force `(f_x, f_y, f_z)` at `(x, y)`        :meth:`.Shell.add_point_load`
line force per unit length along `y` at `x`      :meth:`.Shell.add_distr_load_fixed_x`
line force per unit length along `x` at `y`      :meth:`.Shell.add_distr_load_fixed_y`
pressure over the domain or over a patch         :meth:`.Shell.add_pressure_load`
=============================================== ======================================================

The components are along the local axes of the shell, `z` being the normal
of the mid-surface, positive outwards for the cylinders, such that an
external pressure on a cylinder is negative. The line forces are functions
of the coordinate along the line, and the pressure a constant or a function
`p(x, y)`. A pressure is a dead load by default, or a follower load, normal
to the deformed mid-surface, with ``follower=True``, see
:ref:`follower_pressure`. :meth:`.Shell.clear_loads` removes all of them.

Each load is constant (``cte=True``, the default) or incremented
(``cte=False``), stored in separate lists, and

.. math::

    \{F_{ext}(\lambda)\} = \{F_{ext_0}\} + \lambda \{F_{ext_\lambda}\}

is the vector returned by ``calc_fext(inc=lambda)``. The non-linear solvers
of `structsolve <https://github.com/saullocastro/structsolve>`_ call
``calc_fext(inc=1.)`` and scale the whole vector by their load factor, and
pass their load factor as ``inc`` to the methods that accept it, e.g.
:meth:`.Shell.calc_fint` and :meth:`.Shell.calc_kC` for the follower loads
proportional to the load factor.

Prescribed displacements, also constant or incremented, enter the external
force vector with the penalty method, see :ref:`boundary_conditions`.


Stress resultants of a prestress
--------------------------------

The geometric stiffness matrix of a linear buckling analysis is calculated
from the membrane stress resultants of a prestress: the constant attributes
``Nxx``, ``Nyy`` and ``Nxy`` of the shell, in closed form, superposed with
those of a static solution ``c`` given to :meth:`.Shell.calc_kG`, the
matrix being then integrated numerically. The attributes ``Nxx_cte``,
``Nyy_cte`` and ``Nxy_cte`` give instead a constant prestress, whose
geometric stiffness is added to the constitutive stiffness by
:meth:`.Shell.calc_kC` and is therefore not multiplied by the eigenvalue of
``lb(kC, kG)``. See :doc:`ex_linear_buckling`.


Example
-------

A simply supported isotropic square plate under a uniform pressure, whose
central deflection is compared with the Navier series:

.. literalinclude:: ../../tests/tests_shell/test_pressure_load.py
    :pyobject: make_shell

.. literalinclude:: ../../tests/tests_shell/test_pressure_load.py
    :pyobject: central_deflection

.. literalinclude:: ../../tests/tests_shell/test_pressure_load.py
    :pyobject: test_uniform_pressure_timoshenko

The constant and incremented parts of the external force vector:

.. literalinclude:: ../../tests/tests_shell/test_pressure_load.py
    :pyobject: test_increment
