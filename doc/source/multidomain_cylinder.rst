Cylinders modeled as multi-domain assemblies
============================================

Unstiffened cylinders (:mod:`panels.multidomain.cylinder`)
----------------------------------------------------------

.. automodule:: panels.multidomain.cylinder
    :members:

Blade-stiffened cylinders (:mod:`panels.multidomain.cylinder_blade_stiffened`)
------------------------------------------------------------------------------

.. automodule:: panels.multidomain.cylinder_blade_stiffened
    :members:

The blades stand on the inner or the outer surface of the skin: the
connection ``'BFycte'`` ties the root of each blade to that face of the
skin, key ``'face'``, and the width of a blade is its height from the
surface of the skin, see :mod:`panels.multidomain.connections.kCBFycte_face`.

Z-stiffened cylinders (:mod:`panels.multidomain.cylinder_z_stiffened`)
----------------------------------------------------------------------

.. automodule:: panels.multidomain.cylinder_z_stiffened
    :members:

Root of the web
~~~~~~~~~~~~~~~

The attached flange is part of the laminate of the strip of skin under it,
and the web starts at the face of that strip on the side of the stringer:
the connection ``'BFycte'`` ties its root to the face of the attached
flange, key ``'face'``, ``'lower'`` for internal and ``'upper'`` for
external stringers. The width of the web runs from that face to the middle
surface of the free flange, which is connected at the middle surface of
the web. For an overall depth `H` from the face of the attached flange on
the skin to the outer face of the free flange, and the same thickness `t_z`
in the three plates, the web is `H - 3 t_z/2` wide; the corner between the
attached flange and the web belongs to the flange.

Laminate of a bent sheet
~~~~~~~~~~~~~~~~~~~~~~~~

A Z stringer formed from one laminated sheet has, in every plate, the plies
of that sheet. The sheet has its own triad: `x` along the stringer, the
transverse direction `t` following the sheet around the bends, and the
normal `n`, with `x, t, n` right-handed; the angles of its plies are
measured from `x` towards `t`. Each plate of the assembly describes its
laminate in its own local axes `x, y, z`, the axis `x` being the same as
that of the sheet. Where the axis `z` of a plate is `n`, its `y` is `t` and
the plate has the stack of the sheet. Where its axis `z` is `-n`, both
triads being right-handed with `x` in common, its axis `y` is `-t`:
flipping `z` flips `y`. A ply at the angle `\theta` from `x` towards `t` is
then at `-\theta` from `x` towards `y`, so that a `+45` ply of the sheet is
a `-45` ply of the plate, and the plies read along `z` are those of the
sheet in reverse order. With `\theta \to -\theta` the terms `A_{16},
A_{26}, D_{16}, D_{26}` of the laminate change sign, and with the reversal
of the order so do `B_{11}, B_{12}, B_{22}, B_{66}`; a cross-ply or
isotropic sheet, symmetric, gives the same laminate in both descriptions.
In a Z stringer the normal of the sheet is the axis `z` of the skin at the
attached flange; it is opposite to the axis `z` of the free flange always,
and to that of the web in two of the four orientations of the stringer
(inside or outside, attached flange on either side of the web), see
:func:`.create_cylinder_z_stiffened`.
