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
number of terms ``nmax``. The substitution ``replace('**', '^')`` aims to
create an input to Mathematica.

.. literalinclude:: ../../theory/func/bardell/bardell.py
    :caption:

In order to calculate the displacement, strain of stress fields using Cython,
the above output is not adequate due to very long integer numbers that will
cause precision overflows. The code below should be used to create an input to
Cython:

.. literalinclude:: ../../theory/func/bardell/bardell_floating_point.py
    :caption:


