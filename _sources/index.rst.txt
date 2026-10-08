Semi-analytical methods for plates, shells and stiffened panels - panels
========================================================================

The ``panels`` module provides semi-analytical (Ritz) models for plates,
cylindrical shells and stiffened panels, using
:ref:`Bardell's hierarchical functions <theory_func_bardell>` as approximation
functions. The structural matrices and force vectors are solved with
`structsolve <https://github.com/saullocastro/structsolve>`_. With ``panels``
you can run:

* Linear static analyses, with field outputs of displacements, strains and
  stresses

* Linear buckling analyses, with the pre-buckling stress state given by
  constant stress resultants or calculated with a static analysis

* Frequency analyses, with or without pre-stress

* Panel flutter analyses, using the piston theory

* Geometrically non-linear static analyses based on Donnell's equations, with
  a tangent stiffness matrix that is the exact derivative of the internal force
  vector, using:

  - the Newton-Raphson method
  - the arc-length methods of Riks and Crisfield, able to trace limit points
    and snap-through

* Multi-domain analyses, connecting shell domains exactly with the
  :ref:`null-space method <null_space>`, the default, or with penalty
  stiffnesses, to model cylinders,
  stiffened panels and panels with debonding defects, also with the shear
  deformation theories

The models are identified by the type of structure, the kinematic theory and
the non-linear equations, see :mod:`panels.models`:

* ``'plate_clpt_donnell'``: plates, classical laminated plate theory (CLPT)
* ``'plate_fsdt_donnell'``: plates, first-order shear deformation theory
* ``'plate_tsdt_donnell'``: plates, third-order shear deformation theory of
  Reddy
* ``'cylshell_clpt_donnell'``: cylindrical shells, CLPT with Donnell's
  kinematics
* ``'cylshell_clpt_sanders'``: cylindrical shells, CLPT with Sanders-Koiter's
  kinematics, which give zero strains for rigid-body motions
* ``'cylshell_fsdt_donnell'``, ``'cylshell_tsdt_donnell'``: cylindrical
  shells, first-order and third-order shear deformation theories with
  Donnell's kinematics
* ``'cylshell_fsdt_sanders'``, ``'cylshell_tsdt_sanders'``: cylindrical
  shells, first-order and third-order shear deformation theories with
  Sanders-Koiter's kinematics


Code repository
---------------

https://github.com/saullocastro/panels


Citing this library
-------------------

Saullo G. P. Castro, Nathan D'Souza. (2026). Semi-analytical methods for plates, shells and stiffened panels (Version 0.11.1). Zenodo. DOI: https://doi.org/10.5281/zenodo.2541522.


Usage examples
--------------

.. toctree::
    :maxdepth: 1

    ex_linear_static.rst
    ex_linear_buckling.rst
    ex_frequency.rst
    ex_flutter.rst
    ex_newton_raphson.rst
    ex_arc_length.rst
    ex_follower_pressure.rst
    ex_multidomain_cylinder.rst
    ex_multidomain_tstiff2d.rst


panels API
----------

.. toctree::
    :maxdepth: 2

    api.rst


Theory
------

.. toctree::
    :maxdepth: 1

    bardell.rst
    multidomain_null_space.rst
    follower_pressure.rst
    cohesive_zone.rst
    ref.rst


Installing panels
-----------------

Install from the distributed packages by simply doing::

    python -m pip install panels

or from the source code, which requires a C++ compiler to build the Cython
extensions, using::

    python -m pip install .

In the browser, with `Pyodide <https://pyodide.org>`_ 314 (CPython 3.14),
which already distributes NumPy, SciPy and matplotlib, the WebAssembly wheel
published on PyPI is installed with ``micropip``, together with
``composites`` and ``structsolve``::

    import micropip
    await micropip.install("panels")

    import io
    from structsolve import lb
    from panels.shell import Shell, load

    s = Shell(a=0.6, b=0.4, stack=[0, 45, -45, 90], plyt=0.125e-3,
              laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.0e9),
              m=10, n=10)
    s.Nxx = -1.
    eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
    s.results['eigvals'] = eigvals
    s.results['eigvecs'] = eigvecs

    # saving to memory, since a web page has no file system
    buf = io.BytesIO()
    s.save(buf)
    data = buf.getvalue()

    # offering the file for download
    from js import Blob, Object, URL, document
    from pyodide.ffi import to_js
    blob = Blob.new(to_js([data]), to_js({'type': 'application/zip'},
                                         dict_converter=Object.fromEntries))
    link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = 'plate.shell.zip'
    link.click()

    # loading the bytes back, e.g. from a file uploaded by the user
    s2 = load(io.BytesIO(data))

In Pyodide the field outputs run serially, without OpenMP. See
:mod:`panels.json_io` for the format of the saved files.


Changelog
---------

https://github.com/saullocastro/panels/blob/main/CHANGELOG.md


License
-------

.. literalinclude:: ../../LICENSE
    :encoding: latin-1


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
