.. _pyodide_page:

Running in the browser with Pyodide
===================================

A WebAssembly wheel of ``panels`` is published on PyPI for
`Pyodide <https://pyodide.org>`_ 314 (CPython 3.14), such that ``panels``
runs in the browser, e.g. in JupyterLite or in a web page with Pyodide.
NumPy, SciPy and matplotlib are distributed with Pyodide, and ``micropip``
installs ``panels`` together with ``composites`` and ``structsolve`` from
PyPI::

    import micropip
    await micropip.install("panels")

In Pyodide the field outputs run serially, without OpenMP.

A web page has no file system: a model is saved to memory with
:meth:`.Shell.save`, which accepts a file-like object, offered to the user
for download through the JavaScript objects of the page, and loaded back with
:func:`.shell.load`, e.g. from a file uploaded by the user. The linear
buckling analysis of a plate, saved and loaded in memory::

    import io
    from structsolve import lb
    from panels.shell import Shell, load

    s = Shell(a=0.6, b=0.4, stack=[0, 45, -45, 90], plyt=0.125e-3,
              laminaprop=(142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.0e9),
              m=10, n=10)
    s.Nxx = -1.
    eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
    print('critical load: Nxx = %1.2f N/m' % -eigvals[0])
    s.results['eigvals'] = eigvals
    s.results['eigvecs'] = eigvecs

    # saving to memory
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

See :mod:`panels.json_io` for the format of the saved files.
