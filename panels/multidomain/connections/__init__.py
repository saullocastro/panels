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
(base) and `y_2` for panel 2 (flange). The rotation penalty uses the rotation
of the normal about `x` of each panel, `-w_{,y}` for the classical laminated
plate theory, including the term `v/r` for the Sanders-Koiter kinematics, and
`\phi_y` for the FSDT and TSDT, see
:mod:`panels.multidomain.connections.kCBFycte`. The connection ``'BFxcte'``,
along `y`, penalizes `-w_{,x}` or `\phi_x`, see
:mod:`panels.multidomain.connections.kCBFxcte`. In both, ``p1`` is the base
and ``p2`` the flange.

kCSB
---------

Connection of type::

               ======        ==> base, p1
               ------        --> skin, p2
                               |
                               v -z

Takes into account the offset between the two mid-surfaces. The two panels
share the axes `x, y, z`, `z` being normal to their mid-surfaces, and ``p1``
is the panel on the positive side of the interface along `z`, ``p2`` the one
on the negative side, whatever their roles or their order in the assembly:
the face `z = -h_1/2` of ``p1`` is connected to the face `z = +h_2/2` of
``p2``. The sketch shows a stiffener base on the side of positive `z` of the
skin; a pad-up on the side of negative `z` of a skin is ``p2``, with the skin
``p1``. The same holds for ``'SB_TSL'``.


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
:mod:`panels.multidomain.connections.kCBFxcte`, with the rotation penalty on
`\phi_y` and `\phi_x`.

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
by the kernels above become linear constraints `[B] \{c\} = \{0\}`, which
are eliminated with a basis `[T]` of their null space, `\{c\} = [T]
\{c_r\}`. No penalty constant is needed, and the results are the limit of
the penalty method for penalty constants going to infinity, see
:ref:`null_space` and :mod:`panels.multidomain.connections.nullspace`.


Calculating Penalty Constants
------------------------------

Function :func:`.calc_kt_kr` is based on Ref [castro2017Multidomain]_ and
uses a strain compatibility criterion to calculate penalty constants for
translation (``kt``) and rotation (``kr``). The aim is to have penalty constants
that are just high enough to produce the desired compatibility, but not too
high such that numerical stability issues start to appear. The penalty
constants are not used by the connections imposed with the null-space
method, see :ref:`null_space`.

.. autofunction:: panels.multidomain.connections.calc_kt_kr

Damaged connections were implemented by Nathan (2024) [nathan2024MSc]_ and the
out-of-plane connectivity stiffness in the presence of a damage mapped using
traction-separation law is calculated using :func:`.calc_kw_tsl`.

.. autofunction:: panels.multidomain.connections.calc_kw_tsl

"""
from . kCBFycte import *
from . kCSB import *
from . kCSSxcte import *
from . kCSSycte import *
from . kCBFxcte import *
from . kCpd import *
from . kCSB_dmg import *
from . penalties import *
from . import kCsdt
from . import nullspace
