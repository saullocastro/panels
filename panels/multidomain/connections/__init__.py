r"""Connections between the domains of a :class:`.MultiDomain`

The kernels that add the penalty stiffness of a connection to the
stiffness matrix of the assembly, by the name of the connection given in
``'func'``, see :meth:`.MultiDomain.get_kC_conn`:

- ``'BFycte'`` and ``'BFxcte'``, base-flange at a constant `y` or `x`:
  :mod:`panels.multidomain.connections.kCBFycte` and
  :mod:`panels.multidomain.connections.kCBFxcte`, and ``'BFxcte'`` and
  ``'BFycte'`` at a face of the base, key ``'face'``:
  :mod:`panels.multidomain.connections.kCBFxcte_face` and
  :mod:`panels.multidomain.connections.kCBFycte_face`;
- ``'BFxycte'``, between the edge `x_1` = constant of ``p1`` and the edge
  `y_2` = constant of ``p2``: :mod:`panels.multidomain.connections.kCBFxycte`;
- ``'SSxcte'`` and ``'SSycte'``, skin-skin at a constant `x` or `y`:
  :mod:`panels.multidomain.connections.kCSSxcte` and
  :mod:`panels.multidomain.connections.kCSSycte`, for the models with 3 DOFs
  per term, and :mod:`panels.multidomain.connections.kCsdt`, for the models
  based on the FSDT and TSDT;
- ``'SB'``, skin-base over an area: :func:`.fkCSB` of
  :mod:`panels.multidomain.connections.skin_base`;
- ``'SB_TSL'``, skin-base with a traction-separation law:
  :mod:`panels.multidomain.connections.kCSB_dmg`.

The package also imports the penalty kernels of the prescribed
displacements, :mod:`panels.multidomain.connections.kCpd`, the penalty
constants of :mod:`panels.multidomain.connections.penalties`, e.g.
:func:`.calc_kt_kr` and :func:`.calc_kw_tsl`, the discretisation of the
cohesive zone, :func:`.tsl_resolution`, :func:`.tsl_margin` and
:func:`.tsl_discretization`, and the modules ``kCsdt``, ``skin_base`` and
``nullspace``, the latter with the constraints of the connections imposed
with the null-space method.

"""
from . kCBFycte import *
from . skin_base import fkCSB, fkCSB11, fkCSB12, fkCSB22
from . kCSSxcte import *
from . kCSSycte import *
from . kCBFxcte import *
from . kCBFxycte import *
from . kCpd import *
from . kCSB_dmg import *
from . penalties import *
from . tsl import tsl_resolution, tsl_margin, tsl_discretization
from . import kCsdt
from . import kCBFxcte_face
from . import kCBFycte_face
from . import skin_base
from . import nullspace
