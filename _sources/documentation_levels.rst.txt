How the documentation is organised
==================================

The documentation of ``panels`` is written in four cumulative levels, each
with its own place, so that a reader, or a tool, can go from what a
function does to why it does it without the levels repeating each other.

.. list-table::
    :header-rows: 1

    * - Level
      - Where
      - Content
    * - 1, functional
      - docstrings of the modules, classes, functions and methods of
        ``panels/``, Python and Cython, except their ``Notes`` and
        ``References`` sections; comments in the code; the README;
        docstrings and comments of the tests
      - what each resource does: a one-line summary, an extended summary
        (what, not why), ``Parameters`` with type, unit, shape and
        default, ``Returns``, ``Raises``, ``See Also`` and, for the main
        entry points only, a short ``Examples``
    * - 2, theory
      - the ``Notes`` and ``References`` sections of the docstrings, used
        only for theory
      - the theory that each resource implements, local to it: the
        governing equations in the notation of the code, the assumptions
        and limits of validity, the meaning of each parameter in that
        context, the algorithm, and the published sources
    * - 3, overview
      - the pages of this documentation, starting from
        :ref:`shell_overview`, and the API pages
      - the theory in a larger context, like a textbook: the kinematics,
        the approximation functions, the boundary conditions, the loads,
        the connections between domains and the solution methods, with
        links to the API instead of repeating level 2
    * - 4, worked examples
      - the notebooks in ``notebooks/``, rendered in the sections "Worked
        examples" and "Tutorials" of this documentation
      - complete analyses, step by step, with their results against
        reference values of the literature, with stored outputs

Each level points only to the levels below it: the docstrings and the
comments name no page and no notebook, and the pages name no notebook. The
API pages render levels 1 and 2 together. To find the theory of a resource,
read the ``Notes`` of its docstring; to find how a class or a method fits in
an analysis, read the pages of level 3.


The tests
---------

The tests in ``tests/`` are part of level 1. Each test states in its
docstring what it verifies and against what:

* a published value, a table or an equation of the literature, named in
  the docstring with its location, and cited in full in its ``Notes`` and
  ``References``;
* a closed-form solution written in the test, with its source;
* a property that holds for any input, e.g. the symmetry of a stiffness
  matrix, or the equilibrium between a reaction and the integral of the
  tractions;
* another formulation of the same model, e.g. a connection imposed with
  the null-space method and with penalties;
* a value of a previous run of the same model, stated as such in the
  docstring.

A notebook may import a module of ``tests/``, e.g. a model that it shares
with a test; a test imports nothing from ``notebooks/`` or ``doc/``.
