Wing-box modeled as a multi-domain assembly
===========================================

:func:`.create_wingbox` builds the rectangular, untapered wing-box of
Elalfy, De Breuker and Castro, case High-AR-t5 (their Tables 1 and 2, p. 7),
with skins, spars, ribs and optionally blade stringers, all of them plate
domains, ``'plate_clpt_donnell'``, connected with penalty stiffnesses, see
:mod:`panels.multidomain.wingbox`. The wing-box is clamped at the root and
loaded by a vertical force at the tip. The ribs are connected to the skins
only, not to the spars, see the note of :mod:`panels.multidomain.wingbox`.

:func:`.wingbox_tip_force_lb` runs the linear buckling analysis: the
pre-buckling state is the linear static solution, solved with
``structsolve.solve``, which removes the null rows and columns of the
clamped root, and gives the geometric stiffness matrix of
:meth:`.MultiDomain.calc_kG`. Without stringers, with 45 domains and 6264
Ritz constants, the first buckling load, 172.77 N, a local buckling of the
upper skin of the root bay, is 0.20% below the 173.11 N of their Abaqus
model (Table 3, p. 8):

.. literalinclude:: ../../tests/multidomain/test_wingbox.py
    :pyobject: test_wingbox_tip_force_lb

The blade stringers are added with ``nstr``, the number of stringers per
skin, which also splits the skins into ``nstr + 1`` strips:

.. code-block:: python

    from panels.multidomain import wingbox_tip_force_lb

    md, c, eigvals, eigvecs = wingbox_tip_force_lb(nstr=4, hs=3.e-3,
                                                   ts=1.5e-3)

With 4 stringers per skin, ``hs = 3`` mm and ``ts = 1.5`` mm, 225 domains and
19872 Ritz constants, the first buckling load is 709.2 N (panels 0.14.2,
3 min), a buckling of the upper skin and of its stringers in the two bays
next to the root, with a global bending part; no reference value is
available for this configuration, which is not one of the paper.
