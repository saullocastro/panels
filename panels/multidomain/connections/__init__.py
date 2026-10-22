r"""
===============================================================
Multidomain connections (:mod:`panels.multidomain.connections`)
===============================================================

.. currentmodule:: panels.multidomain.connections

Connection between panel domains. Each panel domain has its own set of Bardell
approximation functions. Below it is shown the connections currently supported.
They are imposed exactly with the null-space method by default, see
:ref:`null_space`, or with the penalty stiffnesses of the kernels below with
``conn_method='penalty'``.

kCBFycte
---------

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

kCSB
---------

Connection of type::

               ======        ==> base, p1
               ------        --> skin, p2
                               |
                               v -z

The two panels share the axes `x, y, z`, `z` being normal to their
mid-surfaces, and ``p1`` is the panel on the positive side of the interface
along `z`, ``p2`` the one on the negative side, whatever their roles or
their order in the assembly: the face `z = -h_1/2` of ``p1`` is connected to
the face `z = +h_2/2` of ``p2``. The sketch shows a stiffener base on the
side of positive `z` of the skin; a pad-up on the side of negative `z` of a
skin is ``p2``, with the skin ``p1``. The same holds for ``'SB_TSL'``.


kCSSxcte
---------

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
---------

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

The connection ``'SB_TSL'``, with a traction-separation law at the
interface, see :ref:`cohesive_zone`.

.. automodule:: panels.multidomain.connections.kCSB_dmg
    :members: fkCSB11_dmg, fkCSB12_dmg, fkCSB22_dmg


Models based on shear deformation theories
-------------------------------------------

The kernels above assume the 3 DOFs `u, v, w` of the models based on the
classical laminated plate theory. The connections ``'SSxcte'``, ``'SSycte'``
and ``'SB'`` of the plate and cylindrical shell models based on the FSDT
and TSDT, with the 5 DOFs `u, v, w, \phi_x, \phi_y`, are in
:mod:`panels.multidomain.connections.kCsdt`, and the connections ``'BFycte'``
and ``'BFxcte'`` of these models are the functions ``fkCBFycte*_sdt`` and
``fkCBFxcte*_sdt`` of :mod:`panels.multidomain.connections.kCBFycte` and
:mod:`panels.multidomain.connections.kCBFxcte`.

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


Exact connections: the null-space method
----------------------------------------

With ``MultiDomain(..., conn_method='null-space')``, the default, the
connections
``'SSxcte'``, ``'SSycte'``, ``'BFxcte'``, ``'BFycte'`` and ``'SB'`` are not
added as penalty stiffnesses, but imposed exactly: the quantities penalized
by the kernels above become linear constraints on the Ritz constants, which
are eliminated from them. No penalty constant is needed, see
:ref:`null_space` and :mod:`panels.multidomain.connections.nullspace`.


Calculating Penalty Constants
------------------------------

Function :func:`.calc_kt_kr` calculates the penalty constants for
translation (``kt``) and rotation (``kr``) from the laminates of the two
connected panels. The penalty constants are not used by the connections
imposed with the null-space method, see :ref:`null_space`.

.. autofunction:: panels.multidomain.connections.calc_kt_kr

The out-of-plane connectivity stiffness of a damaged connection, with the
damage mapped using a traction-separation law, is calculated using
:func:`.calc_kw_tsl`.

.. autofunction:: panels.multidomain.connections.calc_kw_tsl

The discretisation of the cohesive domain of a ``'SB_TSL'`` connection
follows from :func:`.tsl_resolution`, :func:`.tsl_margin` and
:func:`.tsl_discretization`, see :mod:`panels.multidomain.connections.tsl`.

.. autofunction:: panels.multidomain.connections.tsl_resolution
.. autofunction:: panels.multidomain.connections.tsl_margin
.. autofunction:: panels.multidomain.connections.tsl_discretization

Notes
-----
Each penalty kernel adds, for each quantity `q_k` that must be continuous
across the connection, the energy `\frac{1}{2} k_k \int (q_k^{(1)} -
q_k^{(2)})^2`, integrated along the connected edges, or over the connected
area for ``'SB'``, with the translation or rotation penalty constant
`k_k`, ``kt`` or ``kr``. The rotation penalty of ``'BFycte'`` uses the
rotation of the normal about `x` of each panel, `-w_{,y}` for the classical
laminated plate theory, including the term `v/r` for the Sanders-Koiter
kinematics, and `\phi_y` for the FSDT and TSDT, see
:mod:`panels.multidomain.connections.kCBFycte`; the connection ``'BFxcte'``
penalizes `-w_{,x}` or `\phi_x`, see
:mod:`panels.multidomain.connections.kCBFxcte`. The connection ``'SB'``
takes into account the offset between the two mid-surfaces, see
:func:`.nullspace._clpt_components` and :func:`.kCsdt.fkCSB_sdt`.

With the null-space method the quantities penalized by the kernels become
the linear constraints `[B] \{c\} = \{0\}`, which are eliminated with a
basis `[T]` of their null space, `\{c\} = [T] \{c_r\}`. The results are
the limit of the penalty method for penalty constants going to infinity,
see :mod:`panels.multidomain.connections.nullspace`.

Function :func:`.calc_kt_kr` is based on [1]_ and uses a strain
compatibility criterion to calculate the penalty constants. The aim is to
have penalty constants that are just high enough to produce the desired
compatibility, but not too high such that numerical stability issues start
to appear.

Damaged connections were implemented by Nathan (2024) [2]_, with the
traction-separation law of :func:`.calc_kw_tsl`.

References
----------
.. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
   models to Address Linear Buckling and Vibration of Stiffened Composite
   Panels with Debonding Defect, Composite Structures, 2017.
   https://doi.org/10.1016/j.compstruct.2016.10.026
.. [2] D'Souza, N., Multi-Domain Semi-Analytical Cohesive Zone Approach,
   MSc thesis, Delft University of Technology, 2024.
   https://resolver.tudelft.nl/uuid:4709f888-c032-4d9f-97e3-402dfdd47cf1

"""
from . kCBFycte import *
from . kCSB import *
from . kCSSxcte import *
from . kCSSycte import *
from . kCBFxcte import *
from . kCpd import *
from . kCSB_dmg import *
from . penalties import *
from . tsl import tsl_resolution, tsl_margin, tsl_discretization
from . import kCsdt
from . import nullspace
