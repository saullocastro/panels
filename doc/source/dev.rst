Development tools (:mod:`panels.dev`)
=====================================

Tabulation of the approximation functions (:mod:`panels.dev.tabulate`)
----------------------------------------------------------------------

.. automodule:: panels.dev.tabulate
    :members:

Generator of Bardell's functions
--------------------------------

The C++ evaluation of Bardell's functions,
``panels/core/src/bardell_functions.cpp``, is written by the SymPy script
``theory/func/bardell/bardell_functions_C.py``, which expands the exact
polynomials of the definition and their derivatives and writes each one by
Horner's rule, see :ref:`theory_func_bardell`::

    python theory/func/bardell/bardell_functions_C.py

followed by a new build of the extensions. The script is not part of the
package.
