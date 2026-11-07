.. _multidomain_connections:

Multidomain connections
=======================

.. currentmodule:: panels.multidomain.connections

Each domain of a :class:`.MultiDomain` assembly is a :class:`.Shell` with its
own set of Bardell approximation functions, see :ref:`theory_func_bardell`.
The domains are joined by connections, given to the assembly as dictionaries
with the key ``'func'``, e.g. ``'SSycte'``, see :meth:`.MultiDomain.get_kC_conn`.
This page describes the connections currently supported. They are imposed
exactly with the null-space method by default, see :ref:`null_space` and
:mod:`panels.multidomain.connections.nullspace`, or with the penalty
stiffnesses of the kernels below with ``conn_method='penalty'``.


Penalty method
--------------

Each penalty kernel adds, for each quantity `q_k` that must be continuous
across the connection, the energy:

.. math::
    :label: conn-penalty

    U_k = \frac{1}{2} k_k \int \left( q_k^{(1)} - q_k^{(2)} \right)^2

integrated along the connected edges, or over the connected area for
``'SB'``, with the translation or rotation penalty constant `k_k`, ``kt`` or
``kr`` [castro2017Multidomain]_. The quantities `q_k` of each connection are
given in the sections below.

With the null-space method the quantities penalized by the kernels become
the linear constraints:

.. math::
    :label: conn-constraints

    \boldsymbol{B} \boldsymbol{c} = \boldsymbol{0}

which are eliminated with a basis `\boldsymbol{Z}` of their null space,
`\boldsymbol{c} = \boldsymbol{Z} \boldsymbol{c}_r`. The results are the limit
of the penalty method for penalty constants going to infinity, see
:ref:`null_space`.


Base-flange, BF
---------------

kCBFycte
~~~~~~~~

Connection of type::

                          _
                           |
       || --> Flange       |
       ||                  |-> Can be used to model a stiffener
     ======  --> Base      |
                          _|

``ycte`` indicates the connection exists at a constant `y_1` for panel 1
(base) and `y_2` for panel 2 (flange), see
:mod:`panels.multidomain.connections.kCBFycte`. The connection ``'BFxcte'``,
along `y`, is the same at a constant `x_1` and `x_2`, see
:mod:`panels.multidomain.connections.kCBFxcte`. In both, ``p1`` is the base
and ``p2`` the flange.

The rotation penalty of ``'BFycte'`` uses the rotation of the normal about
`x` of each panel, `-w_{,y}` for the classical laminated plate theory,
including the term `v/r` for the Sanders-Koiter kinematics, and `\phi_y` for
the FSDT and TSDT; the connection ``'BFxcte'`` penalizes `-w_{,x}` or
`\phi_x`.

kCBFxycte
~~~~~~~~~

Connection of type, seen along the common edge, for the default angle of
90 degrees::

                     ^ z1 = y2
                     |
                     |  p2, e.g. a rib, edge y2 = 0
                     |
      ===============o     o: the common edge, y1 = x2 (into the page)
      p1, e.g. a spar, edge x1 = a1       ---> x1 = z2

between the edge `x_1 =` ``xcte1`` of ``p1`` and the edge `y_2 =`
``ycte2`` of ``p2``, the same straight line, with `y_1 = x_2` and ``p1.b
== p2.a``, the two domains at the angle ``'angle'``, by default 90
degrees, measured from the normal of ``p1`` to that of ``p2`` about the
common axis. It joins, for example, a spar ``p1``, whose edge at the end of
a bay is `x_1 = a_1`, to a rib ``p2``, whose edges `y_2 = 0` or `y_2 =
b_2` run along the height of a wing-box, when the spar is connected to the
skins with ``'BFycte'`` and the rib with ``'BFxcte'``. The displacements of
``p2`` are rotated to the frame of ``p1``, and the rotations about the
common axis are equal: `-w_{1,x}` of ``p1`` and `w_{2,y}` of ``p2``, with
`-v_2/r_2` for the Sanders-Koiter kinematics, or `\theta_{x1}` and
`-\theta_{y2}` for the FSDT and TSDT, see
:mod:`panels.multidomain.connections.kCBFxycte`.


Skin-base, SB
-------------

Connection of type::

               ======        ==> base, p1
               ------        --> skin, p2
                               |
                               v -z

The two panels share the axes `x, y, z`, `z` being normal to their
reference surfaces, and ``p1`` is the panel on the positive side of the
interface along `z`, ``p2`` the one on the negative side, whatever their
roles or their order in the assembly: the lower face `z_1 = d_1 - h_1/2` of
``p1`` is connected to the upper face `z_2 = d_2 + h_2/2` of ``p2``, with
the offsets `d_1`, `d_2` of the laminates, `u_1(z_1) = u_2(z_2)`, `v_1(z_1)
= v_2(z_2)` and `w_1 = w_2`, each panel with its own kinematics, for all the
models, see :mod:`panels.multidomain.connections.skin_base`. The sketch
shows a stiffener base on the side of positive `z` of the skin; a pad-up on
the side of negative `z` of a skin is ``p2``, with the skin ``p1``. The same
holds for ``'SB_TSL'``.

The connection ``'SB'`` penalizes the displacements `u(z), v(z), w` of the
lower face of ``p1`` and of the upper face of ``p2``, with the offsets of the
laminates and the kinematics of each panel, with its slope and, with the
Sanders-Koiter kinematics, the term `v/r`, see :func:`.kCsdt._sb_components`
and :func:`.fkCSB`. The rotation penalty ``'kr'`` of ``'SB'`` applies only to
the FSDT and TSDT, whose rotations are independent of `u, v, w`: the
rotations of the normal at the reference surfaces are then penalized with
``kr``. The rotations of the classical laminated plate theory follow from
`w`, already tied by `w_1 = w_2`, and tying them too would over-constrain the
connection with the Sanders-Koiter kinematics.


Skin-skin, SS
-------------

kCSSxcte
~~~~~~~~

Connection of type::

      __________
      |        |
      |        |      /^\ x2
      |   S2   |       |
      |        |   y2  |
      |        |   <----
      |________| (connection at x2=xcte2)
      __________ (connection at x1=xcte1)
      |        |
      |        |      /^\ x1
      |   S1   |       |
      |        |   y1  |
      |________|   <----

kCSSycte
~~~~~~~~

Connection of type::

                 /-> (connection at y1=ycte1)
                /
               /  /->(connection at y2=ycte2)
      _________| |_________
      |        | |        |
      |        | |        |
      |   S1   | |   S2   |
      |        | |        |
      |________| |________|

          /^\ x1       /^\ x2
           |            |
       y1  |        y2  |
       <----        <----


Damaged skin-base connection
----------------------------

The connection ``'SB_TSL'`` has a traction-separation law at the interface,
see :ref:`cohesive_zone`, :mod:`panels.multidomain.connections.kCSB_dmg` and
:mod:`panels.multidomain.connections.tsl`. Its out-of-plane stiffness is that
of the cohesive interface, with the damage mapped by the traction-separation
law of :func:`.calc_kw_tsl`, and it keeps this penalty stiffness also with
``conn_method='null-space'``. The damaged connections were implemented by
D'Souza [nathan2024MSc]_.


Models based on shear deformation theories
-------------------------------------------

The kernels above, apart from :func:`.fkCSB` of ``'SB'``, which serves
all the models, assume the 3 DOFs `u, v, w` of the models based on the
classical laminated plate theory. The connections ``'SSxcte'`` and
``'SSycte'`` of the plate and cylindrical shell models based on the FSDT
and TSDT, with the 5 DOFs `u, v, w, \phi_x, \phi_y`, are in
:mod:`panels.multidomain.connections.kCsdt`, with :func:`.fkCSB_sdt`, the
matrix of :func:`.fkCSB` for these models, and the connections ``'BFycte'``
and ``'BFxcte'`` of these models are the functions ``fkCBFycte*_sdt`` and
``fkCBFxcte*_sdt`` of :mod:`panels.multidomain.connections.kCBFycte` and
:mod:`panels.multidomain.connections.kCBFxcte`. The coupling block of
``'BFxycte'``, :func:`.fkCBFxycte12`, serves the models with 3 and with 5
DOFs, with the blocks :func:`.fkCBFxycte11_sdt` and
:func:`.fkCBFxycte22_sdt` of each panel.


Exact connections: the null-space method
----------------------------------------

With ``MultiDomain(..., conn_method='null-space')``, the default, the
connections ``'SSxcte'``, ``'SSycte'``, ``'BFxcte'``, ``'BFycte'``,
``'BFxycte'`` and ``'SB'`` are not added as penalty stiffnesses, but imposed
exactly: the quantities penalized by the kernels above become the linear
constraints of Eq. :eq:`conn-constraints` on the Ritz constants, which are
eliminated from them. No penalty constant is needed, see
:mod:`panels.multidomain.connections.nullspace`.


Prescribed displacements with penalties
---------------------------------------

With ``MultiDomain(..., pd_method='penalty')``, the default, a displacement
prescribed at a point or along a line, see :meth:`.Shell.add_point_pd`,
:meth:`.Shell.add_distr_pd_fixed_x` and :meth:`.Shell.add_distr_pd_fixed_y`,
is imposed with a spring between the panel and the support: the penalty
stiffness is added to the stiffness matrix with the kernels of
:mod:`panels.multidomain.connections.kCpd`, for the models with 3 and with 5
DOFs per term, and the force of the spring is in the external force vector of
:func:`.shell_fext`. The in-plane displacements are those at the thickness
coordinate ``z`` of the kernels, the one given to the methods of
:class:`.Shell`, by default the reference surface. With
``pd_method='null-space'`` the prescribed displacements are imposed exactly,
see :func:`.nullspace.pd_constraint_matrix`.


Penalty constants
-----------------

Function :func:`.calc_kt_kr` calculates the penalty constants for
translation (``kt``) and rotation (``kr``) from the laminates of the two
connected panels, based on [castro2017Multidomain]_, with a strain
compatibility criterion. The aim is to have penalty constants that are just
high enough to produce the desired compatibility, but not too high such that
numerical stability issues start to appear. The penalty constants are not
used by the connections imposed with the null-space method, apart from the
key ``'kr'`` of ``'SB'``, which tells whether the rotations of the FSDT and
TSDT are connected. For ``'SB'`` the function gives no ``kr``.

The discretisation of the cohesive domain of a ``'SB_TSL'`` connection
follows from :func:`.tsl_resolution`, :func:`.tsl_margin` and
:func:`.tsl_discretization`, see :mod:`panels.multidomain.connections.tsl`.


API
---

.. automodule:: panels.multidomain.connections

.. automodule:: panels.multidomain.connections.kCBFxycte
    :members: fkCBFxycte11, fkCBFxycte12, fkCBFxycte22, fkCBFxycte11_sdt,
              fkCBFxycte22_sdt, xycte_components, check_xycte

.. automodule:: panels.multidomain.connections.skin_base
    :members: fkCSB

.. automodule:: panels.multidomain.connections.kCSB_dmg
    :members: fkCSB11_dmg, fkCSB12_dmg, fkCSB22_dmg

.. automodule:: panels.multidomain.connections.kCsdt
    :members: fkCSSxcte_sdt, fkCSSycte_sdt, fkCSB_sdt

.. automodule:: panels.multidomain.connections.kCBFycte
    :members: fkCBFycte11, fkCBFycte12, fkCBFycte22, fkCBFycte11_sdt,
              fkCBFycte12_sdt, fkCBFycte22_sdt
    :private-members: _block_sdt

.. automodule:: panels.multidomain.connections.kCBFxcte
    :members: fkCBFxcte11, fkCBFxcte12, fkCBFxcte22, fkCBFxcte11_sdt,
              fkCBFxcte12_sdt, fkCBFxcte22_sdt
    :private-members: _block_sdt

.. autofunction:: panels.multidomain.connections.kCpd.fkCpd

.. autofunction:: panels.multidomain.connections.kCpd.fkCld_xcte

.. autofunction:: panels.multidomain.connections.kCpd.fkCld_ycte

.. autofunction:: panels.multidomain.connections.calc_kt_kr

.. autofunction:: panels.multidomain.connections.calc_kw_tsl

.. autofunction:: panels.multidomain.connections.tsl_resolution
.. autofunction:: panels.multidomain.connections.tsl_margin
.. autofunction:: panels.multidomain.connections.tsl_discretization
