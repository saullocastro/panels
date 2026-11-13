.. _loads:

Loads
=====

The loads of a :class:`.Shell` are stored as lists of its attributes and
converted into the external force vector `\boldsymbol{F}_{ext}` by
:meth:`.Shell.calc_fext`, see :func:`.shell_fext`, as the virtual work of
the loads on the approximation functions. In a :class:`.MultiDomain`, the
loads are added to its domains and :meth:`.MultiDomain.calc_fext`
assembles their vectors, whereas :meth:`.StiffPanelBay.calc_fext` takes
the point forces of the attribute ``forces_skin`` of the bay and of the
attributes ``forces_flange`` of its :class:`.BladeStiff2D` stiffeners.


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

    \boldsymbol{F}_{ext}(\lambda) = \boldsymbol{F}_{ext_0} + \lambda \boldsymbol{F}_{ext_\lambda}

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
``lb(kC, kG)``.


Virtual work and convergence
----------------------------

With `\boldsymbol{S}^u`, `\boldsymbol{S}^v` and `\boldsymbol{S}^w` the rows of
the approximation functions of `u`, `v` and `w`, the external force vector
is the virtual work of the loads, e.g. for a pressure `p(x, y)` and a point
force `(f_x, f_y, f_z)` at `(x_0, y_0)`:

.. math::
    :label: loads-fext

    \boldsymbol{F}_{ext} = \int_A (\boldsymbol{S}^w)^\top p \, dA
    + \left. \left( (\boldsymbol{S}^u)^\top f_x + (\boldsymbol{S}^v)^\top f_y
    + (\boldsymbol{S}^w)^\top f_z \right) \right|_{(x_0, y_0)}

the pressure being integrated over its patch, see
:func:`.pressure_patches`, and the line forces along their line, with
``m`` Gauss-Legendre points along `x` or ``n`` along `y`, see
:func:`.shell_fext`. The penalty forces `k u_p` of the prescribed
displacements enter in the same way. For a uniform pressure `q` on a simply
supported isotropic plate, the Ritz solution converges to the Navier series
of the central deflection

.. math::
    :label: loads-navier

    w\left(\frac{a}{2}, \frac{b}{2}\right) = \frac{16 q}{\pi^6 D}
    \sum_{i = 1, 3, \dots} \sum_{j = 1, 3, \dots}
    \frac{(-1)^{(i + j)/2 - 1}}{i j \left(i^2/a^2 + j^2/b^2\right)^2}

`D = E h^3/(12(1 - \nu^2))`, which is `0.00406235 \, q a^4/D` for the
square plate, reached to `10^{-4}` with ``m = n = 15``.
