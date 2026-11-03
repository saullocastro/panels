Wing-box modeled as a multi-domain assembly
===========================================

:func:`.create_wingbox` builds the rectangular, untapered wing-box of
Elalfy, De Breuker and Castro, case High-AR-t5 (their Tables 1 and 2, p. 7),
with skins, spars, ribs and optionally blade stringers, all of them plate
domains, ``'plate_clpt_donnell'``, connected with penalty stiffnesses, see
:mod:`panels.multidomain.wingbox`. The wing-box is clamped at the root and
loaded by a vertical force at the tip. The ribs are connected to the skins
with ``'BFxcte'`` and to the spars with ``'BFxycte'`` (``rib_spar=True``, the
default); the stringers pass through the ribs without a connection.

:func:`.wingbox_tip_force_lb` runs the linear buckling analysis: the
pre-buckling state is the linear static solution, solved with
``structsolve.solve``, which removes the null rows and columns of the
clamped root, and gives the geometric stiffness matrix of
:meth:`.MultiDomain.calc_kG`. Without stringers, with 45 domains, the first
buckling mode is a local buckling of the upper skin of the root bay. The
first buckling load of their Abaqus model is 173.11 N (Table 3, p. 8). With
the connections imposed exactly, ``conn_method='null-space'``, the load
converges from above to 176.4 N, 1.9% above Abaqus, whereas with the
default penalty connections it converges to 161.9 N, 6.5% below: the
default penalty constants of :func:`.calc_kt_kr` leave some relative motion
at the joints. The 172.78 N of the default model, penalty connections with
6264 Ritz constants, close to Abaqus, is therefore a coincidence of the
number of terms and the penalty constants.

First buckling load (N) against the number of terms ``(m, n)`` of the
domains (panels 0.14.2, one BLAS thread, the time of
:func:`.wingbox_tip_force_lb` in parentheses):

.. list-table::
    :header-rows: 1

    * - skins
      - spars
      - ribs
      - Ritz constants
      - null-space
      - penalty
    * - 8 x 8
      - 8 x 5
      - 4 x 6
      - 6264
      - 190.41 (8 s)
      - 172.78 (12 s)
    * - 10 x 10
      - 10 x 6
      - 5 x 7
      - 9585
      - 177.74 (13 s)
      - 162.75 (20 s)
    * - 12 x 12
      - 12 x 7
      - 6 x 8
      - 13608
      - 176.44 (24 s)
      - 161.85 (37 s)
    * - 14 x 14
      - 14 x 8
      - 7 x 9
      - 18333
      - 176.48 (67 s)
      - 161.86 (83 s)
    * - 16 x 16
      - 16 x 9
      - 8 x 10
      - 23760
      - 176.47 (117 s)
      - 161.85 (123 s)

The defaults of :func:`.create_wingbox` are the first row. The test uses the
third one, with the null-space method:

.. literalinclude:: ../../tests/multidomain/test_wingbox.py
    :pyobject: test_wingbox_tip_force_lb

The blade stringers are added with ``nstr``, the number of stringers per
skin, which also splits the skins into ``nstr + 1`` strips:

.. code-block:: python

    from panels.multidomain import wingbox_tip_force_lb

    md, c, eigvals, eigvecs = wingbox_tip_force_lb(nstr=4, hs=3.e-3,
                                                   ts=1.5e-3)

With 4 stringers per skin, ``hs = 3`` mm and ``ts = 1.5`` mm, 225 domains and
19872 Ritz constants, the default terms and penalty connections, the first
buckling load is 708.9 N (43 s with one BLAS thread), a buckling of the upper skin and of its stringers in the two bays
next to the root, with a global bending part; no reference value is
available for this configuration, which is not one of the paper.
