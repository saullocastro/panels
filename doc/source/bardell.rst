.. _theory_func_bardell:

Bardell's Hierarchical Functions
================================

Introduction
------------

Bardell (1991) [bardell1991]_ applied a very convinient set of approximation
functions based on hierarchical Legendre polynomials using Rodrigues form. The
convenience comes from the fast convergence and the easiness to simulate
practically any type of boundary conditions.

The boundary condition is controlled by the first 4 terms of the approximation
function, herein defined as:

- ``t1``: the translation at extremity 1 (`\xi = -1`)
- ``r1``: the rotation at extremity 1
- ``t2``: the translation at extremity 2 (`\xi = +1`)
- ``r2``: the rotation at extremity 2

Generating Bardell's functions
------------------------------

The following code can be used to generate the Bardell functions for a given
number of terms ``nmax``. The substitution ``replace('**', '^')`` makes the
written output ``bardell.txt`` more readable.

.. literalinclude:: ../../theory/func/bardell/bardell.py
    :caption:

In order to calculate the displacement, strain of stress fields using Cython,
the above output is not adequate due to very long integer numbers that will
cause precision overflows. The code below should be used to create an input to
Cython:

.. literalinclude:: ../../theory/func/bardell/bardell_floating_point.py
    :caption:

Integrals of Bardell's functions
--------------------------------

The integrals over `\xi \in [-1, +1]` of the products of Bardell's functions
and of their first and second derivatives, used to compute the structural
matrices, are calculated exactly with SymPy using rational arithmetic. The
code below writes the C++ source ``panels/core/src/bardell.cpp`` and the
corresponding header ``panels/core/include/bardell.hpp``, which contain the
functions ``integral_ff``, ``integral_ffp``, ``integral_ffpp``,
``integral_fpfp``, ``integral_fpfpp`` and ``integral_fppfpp``:

.. literalinclude:: ../../theory/func/bardell/bardell_integrals_C.py
    :caption:

Displacement field
------------------

The displacement field of each model is a sum over the terms `(i, j)` of the
products of Bardell's functions along `x` and `y`, with the boundary flags of
each field. The code below derives the contribution of each term to `u, v, w,
\phi_x, \phi_y` for the models based on the classical laminated plate theory,
with Donnell's or the Sanders-Koiter kinematics, and on the shear deformation
theories, and checks it against the function ``fg`` of the field modules
:mod:`panels.models.clpt_field` and :mod:`panels.models.fsdt_tsdt_field`:

.. literalinclude:: ../../theory/func/bardell/fuvw.py
    :caption:


