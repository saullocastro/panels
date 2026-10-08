r"""
Saving and loading in a JSON + NumPy zip container (:mod:`panels.json_io`)
==========================================================================

.. currentmodule:: panels.json_io

The objects :class:`.Shell`, :class:`.StiffPanelBay`, with its stiffeners
:class:`.BladeStiff1D` and :class:`.BladeStiff2D`, and :class:`.MultiDomain`
are saved to and loaded from a zip file with::

    from panels import Shell
    from panels.shell import load

    s = Shell(a=1., b=0.5, stack=[0, 90, 90, 0], plyt=1e-3,
              laminaprop=(71e9, 0.33), m=8, n=8)
    s.name = 'plate'
    s.save()              # writes plate.shell.zip
    s2 = load('plate')    # or load('plate.shell.zip')

or with the functions :func:`.save` and :func:`.load` of this module, which
handle the three classes. Instead of a file name, both accept a binary file
object, e.g. :class:`io.BytesIO`, which is how a browser application running
on Pyodide saves without a file system::

    import io
    buf = io.BytesIO()
    s.save(buf)
    data = buf.getvalue()  # bytes, e.g. offered for download

The file is a zip archive, compressed with ``ZIP_DEFLATED``, containing:

- ``model.json``: the inputs, such as the geometry, the number of terms
  ``m`` and ``n``, the boundary conditions, the stacking sequence, the ply
  thicknesses, the lamina properties and the loads, in the envelope::

      {
        "type": "Shell",
        "format_version": 1,
        "panels_version": "0.11.0",
        "data": {...}
      }

- ``arrays/<key>.npy``: every NumPy array, e.g. the constitutive matrix
  ``ABD`` and the results ``fields`` or ``results['eigvecs']``, written by
  :func:`numpy.save` with ``allow_pickle=False``. In ``model.json`` an array
  is replaced by ``{"__ndarray__": "arrays/<key>.npy"}``.

The laminates, e.g. :attr:`.Shell.lam`, are embedded in ``model.json`` with
:func:`composites.to_dict`. The JSON is strict, readable by ``JSON.parse`` in
JavaScript: the floats ``nan``, ``inf`` and ``-inf`` are stored as the
strings ``"NaN"``, ``"Infinity"`` and ``"-Infinity"``. Tuples are loaded as
lists and NumPy scalars as Python ``int`` or ``float``.

Only an explicit list of attributes of each class is stored, such that the
format does not depend on implementation details. Saving an attribute whose
value is not supported raises ``TypeError``, e.g. the functions of the
distributed loads :meth:`.Shell.add_distr_load_fixed_x` or of a pressure
``p(x, y)``. The following are not stored and are recomputed on demand: the
structural matrices (``Shell.matrices``, ``StiffPanelBay.kC``,
``MultiDomain.kC``, ``MultiDomain.T``, ...), the number of threads
``out_num_cores``, which takes the default of the machine that loads the
file, and the attributes of Python subclasses, the loaded object having the
type of the base class.

Loading is safe for files from untrusted sources, unlike pickle: the arrays
are read with ``allow_pickle=False``, the members are read in memory and
never extracted, only the members referenced by ``model.json`` are read, and
unknown keys, missing or extra members and newer format versions raise
``ValueError``.

The pickle files written by older versions of panels are still loaded by
:func:`.load`, with a ``DeprecationWarning``, since loading a pickle file
from an untrusted source can execute arbitrary code.

"""
import io
import json
import math
import numbers
import os
import pickle
import re
import warnings
import zipfile

import numpy as np
from composites import to_dict as lam_to_dict, from_dict as lam_from_dict
from composites.core import Laminate

from .version import __version__


FORMAT_VERSION = 1

_ENVELOPE_KEYS = ('type', 'format_version', 'panels_version', 'data')
_MODEL_JSON = 'model.json'
_MEMBER_RE = re.compile(r'arrays/[A-Za-z0-9_.\-]+\.npy')
# NOTE booleans, integers, floats and complex numbers
_ARRAY_KINDS = 'biufc'

# NOTE attributes stored for each class, the others are either recomputed or
#      references to the parent objects, see the module docstring
_BC_FLAGS = tuple(edge + d + r for edge in ('x1', 'x2', 'y1', 'y2')
                  for d in ('u', 'v', 'w') for r in ('', 'r'))
_SHELL_ATTRS = (
    'a', 'x1', 'x2', 'b', 'y1', 'y2', 'r',
    'stack', 'plyt', 'laminaprop', 'rho', 'offset',
    'group', 'x0', 'y0', 'z0', 'point_x', 'point_xy',
    'row_start', 'col_start', 'row_end', 'col_end',
    'name', 'model', 'fsdt_shear_correction',
    'm', 'n', 'nx', 'ny', 'size',
    'point_loads', 'point_loads_inc', 'distr_loads', 'distr_loads_inc',
    'pressure_loads', 'pressure_loads_inc',
    'point_pds', 'point_pds_inc', 'distr_pds', 'distr_pds_inc',
    'Nxx', 'Nyy', 'Nxy', 'Nxx_cte', 'Nyy_cte', 'Nxy_cte',
    ) + _BC_FLAGS + tuple(
        edge + phi + r for edge in ('x1', 'x2', 'y1', 'y2')
        for phi in ('phix', 'phiy') for r in ('', 'r')) + (
    'plyts', 'laminaprops', 'rhos',
    'flow', 'beta', 'gamma', 'aeromu', 'rho_air', 'speed_sound', 'Mach',
    'air_speed',
    'ABD', 'force_orthotropic_laminate',
    'num_eigvalues', 'num_eigvalues_print',
    'increments', 'results', 'fields', 'plot_mesh',
    )
_STIFFPANELBAY_ATTRS = (
    'name', 'forces_skin', 'flow', 'bc', 'model',
    'stack', 'laminaprop', 'laminaprops', 'plyt', 'plyts', 'rho',
    'm', 'n', 'size', 'a', 'b', 'r',
    ) + _BC_FLAGS + (
    'beta', 'gamma', 'aeromu', 'rho_air', 'speed_sound', 'Mach', 'V',
    'u', 'v', 'w', 'phix', 'phiy', 'Xs', 'Ys',
    )
_BLADESTIFF1D_ATTRS = (
    'model', 'rho', 'ys', 'bb', 'hb', 'bf', 'hf',
    'bstack', 'bplyts', 'blaminaprops', 'fstack', 'fplyts', 'flaminaprops',
    'As', 'Asb', 'Asf', 'Jxx', 'Iyy', 'Fx', 'dbf', 'E1', 'S1', 'F1',
    )
_BLADESTIFF2D_ATTRS = (
    'rho', 'ys', 'bb', 'forces_flange', 'bstack', 'bplyts', 'blaminaprops',
    'dpb',
    )
_MULTIDOMAIN_ATTRS = ('name', 'conn_method', 'size', 'dmg_index', 'fint',
                      'fext')
# NOTE attributes whose strings are never non-finite floats
_STR_ATTRS = ('name', 'model', 'group', 'flow', 'conn_method')


def _encode_float(value):
    value = float(value)
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return 'NaN'
    return 'Infinity' if value > 0 else '-Infinity'


_NON_FINITE = {'NaN': math.nan, 'Infinity': math.inf, '-Infinity': -math.inf}


class _Writer(object):
    r"""Collects the arrays of the zip file while encoding"""
    def __init__(self):
        self.arrays = {}

    def array(self, value, path):
        if value.dtype.kind not in _ARRAY_KINDS or value.dtype.fields:
            raise TypeError('Cannot save %s, an array of dtype %s, only '
                            'numeric and boolean arrays are supported'
                            % (path, value.dtype))
        key = re.sub(r'[^A-Za-z0-9_.\-]+', '.', path).strip('.') or 'array'
        member = 'arrays/%s.npy' % key
        i = 1
        while member in self.arrays:
            i += 1
            member = 'arrays/%s-%d.npy' % (key, i)
        self.arrays[member] = value
        return {'__ndarray__': member}


class _Reader(object):
    r"""Reads the arrays referenced by ``model.json``"""
    def __init__(self, zf):
        self.zf = zf
        self.names = set(zf.namelist())
        self.used = set()

    def array(self, member):
        if not isinstance(member, str) or not _MEMBER_RE.fullmatch(member):
            raise ValueError('Invalid array member name: %r' % (member, ))
        if member not in self.names:
            raise ValueError('Missing member in the zip file: %r' % member)
        self.used.add(member)
        # NOTE read_array reads only the .npy format, whereas np.load would
        #      also open a nested .npz archive
        try:
            value = np.lib.format.read_array(io.BytesIO(self.zf.read(member)),
                                             allow_pickle=False)
        except ValueError as e:
            raise ValueError('Invalid array %r: %s' % (member, e))
        if value.dtype.kind not in _ARRAY_KINDS or value.dtype.fields:
            raise ValueError('Invalid array %r: dtype %s is not supported'
                             % (member, value.dtype))
        return value


def _encode(value, path, w):
    r"""Encode a value into JSON-compatible types, arrays go to ``w``"""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str):
        if value in _NON_FINITE:
            raise ValueError('Cannot save %s, the string %r is reserved for '
                             'the non-finite floats' % (path, value))
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return _encode_float(value)
    if isinstance(value, np.ndarray):
        return w.array(value, path)
    if isinstance(value, (list, tuple)):
        return [_encode(v, '%s[%d]' % (path, i), w)
                for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if not isinstance(k, str) or k.startswith('__'):
                raise TypeError('Cannot save %s, the dictionary keys must be '
                                'strings not starting with "__", got %r'
                                % (path, k))
            out[k] = _encode(v, '%s[%r]' % (path, k), w)
        return out
    raise TypeError('Cannot save %s, of type %r' % (path, type(value).__name__))


def _decode(value, r):
    r"""Inverse of :func:`._encode`"""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _NON_FINITE.get(value, value)
    if isinstance(value, list):
        return [_decode(v, r) for v in value]
    if isinstance(value, dict):
        if '__ndarray__' in value:
            if len(value) != 1:
                raise ValueError('Invalid array reference: %r' % (value, ))
            return r.array(value['__ndarray__'])
        for k in value:
            if k.startswith('__'):
                raise ValueError('Unknown marker: %r' % k)
        return {k: _decode(v, r) for k, v in value.items()}
    raise ValueError('Invalid value: %r' % (value, ))


def _check_keys(data, allowed, typename):
    if not isinstance(data, dict):
        raise ValueError('Expected a JSON object for %s, got %r'
                         % (typename, type(data).__name__))
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError('Unknown keys for %s: %s'
                         % (typename, ', '.join(sorted(unknown))))


def _encode_attrs(obj, names, path, w):
    # NOTE attributes set only by some methods, e.g. BladeStiff1D.E1, are
    #      skipped while absent
    data = {}
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if name in _STR_ATTRS and isinstance(value, str):
            data[name] = value
        else:
            data[name] = _encode(value, '%s.%s' % (path, name), w)
    return data


def _decode_attrs(obj, data, names, r):
    for name in names:
        if name not in data:
            continue
        value = data[name]
        if name in _STR_ATTRS and isinstance(value, str):
            setattr(obj, name, value)
        else:
            setattr(obj, name, _decode(value, r))


def _encode_lam(lam, path):
    if lam is None:
        return None
    if not isinstance(lam, Laminate):
        raise TypeError('Cannot save %s, expected a composites Laminate, got %r'
                        % (path, type(lam).__name__))
    return lam_to_dict(lam)


def _decode_lam(data):
    if data is None:
        return None
    lam = lam_from_dict(data)
    if not isinstance(lam, Laminate):
        raise ValueError('Expected a Laminate, got %r' % type(lam).__name__)
    return lam


def _index(obj, objs, path, what):
    for i, o in enumerate(objs):
        if o is obj:
            return i
    raise ValueError('Cannot save %s, it is not one of the %s' % (path, what))


def _get(objs, i, what):
    if (not isinstance(i, numbers.Integral) or isinstance(i, bool)
            or not 0 <= i < len(objs)):
        raise ValueError('Invalid index of the %s: %r' % (what, i))
    return objs[i]


def _list(data, name):
    value = data.get(name, [])
    if not isinstance(value, list):
        raise ValueError('Expected a list for %s, got %r' % (name, value))
    return value


def _shell_to_data(s, path, w):
    data = _encode_attrs(s, _SHELL_ATTRS, path, w)
    data['lam'] = _encode_lam(s.lam, path + '.lam')
    return data


def _shell_from_data(data, r):
    from .shell import Shell
    _check_keys(data, _SHELL_ATTRS + ('lam', ), 'Shell')
    s = Shell()
    _decode_attrs(s, data, _SHELL_ATTRS, r)
    if 'lam' in data:
        s.lam = _decode_lam(data['lam'])
    return s


def _bladestiff1d_to_data(s, bay, path, w):
    data = _encode_attrs(s, _BLADESTIFF1D_ATTRS, path, w)
    for name in ('panel1', 'panel2'):
        data[name] = _index(getattr(s, name), bay.panels,
                            '%s.%s' % (path, name), 'panels of the bay')
    data['base'] = (None if s.base is None
                    else _shell_to_data(s.base, path + '.base', w))
    data['flam'] = _encode_lam(s.flam, path + '.flam')
    return data


def _bladestiff1d_from_data(data, bay, r):
    from .stiffener import BladeStiff1D
    _check_keys(data, _BLADESTIFF1D_ATTRS
                + ('panel1', 'panel2', 'base', 'flam'), 'BladeStiff1D')
    # NOTE without __init__, which needs all parameters and rebuilds the
    #      stiffener, the stored attributes being complete
    s = BladeStiff1D.__new__(BladeStiff1D)
    s.bay = bay
    s.panel1 = _get(bay.panels, data.get('panel1'), 'panels of the bay')
    s.panel2 = _get(bay.panels, data.get('panel2'), 'panels of the bay')
    s.kC = s.kM = s.kG = None
    s.base = s.flam = None
    _decode_attrs(s, data, _BLADESTIFF1D_ATTRS, r)
    if data.get('base') is not None:
        s.base = _shell_from_data(data['base'], r)
    s.flam = _decode_lam(data.get('flam'))
    return s


def _bladestiff2d_to_data(s, bay, path, w):
    data = _encode_attrs(s, _BLADESTIFF2D_ATTRS, path, w)
    for name in ('panel1', 'panel2'):
        data[name] = _index(getattr(s, name), bay.panels,
                            '%s.%s' % (path, name), 'panels of the bay')
    for name in ('base', 'flange'):
        shell = getattr(s, name)
        data[name] = (None if shell is None
                      else _shell_to_data(shell, '%s.%s' % (path, name), w))
    return data


def _bladestiff2d_from_data(data, bay, r):
    from .stiffener import BladeStiff2D
    _check_keys(data, _BLADESTIFF2D_ATTRS
                + ('panel1', 'panel2', 'base', 'flange'), 'BladeStiff2D')
    s = BladeStiff2D.__new__(BladeStiff2D)
    s.bay = bay
    s.panel1 = _get(bay.panels, data.get('panel1'), 'panels of the bay')
    s.panel2 = _get(bay.panels, data.get('panel2'), 'panels of the bay')
    s.kC = s.kM = s.kG = None
    s.base = s.flange = None
    s.forces_flange = []
    _decode_attrs(s, data, _BLADESTIFF2D_ATTRS, r)
    for name in ('base', 'flange'):
        if data.get(name) is not None:
            setattr(s, name, _shell_from_data(data[name], r))
    return s


def _stiffpanelbay_to_data(bay, path, w):
    data = _encode_attrs(bay, _STIFFPANELBAY_ATTRS, path, w)
    data['panels'] = [_shell_to_data(p, '%s.panels[%d]' % (path, i), w)
                      for i, p in enumerate(bay.panels)]
    data['bladestiff1ds'] = [
        _bladestiff1d_to_data(s, bay, '%s.bladestiff1ds[%d]' % (path, i), w)
        for i, s in enumerate(bay.bladestiff1ds)]
    data['bladestiff2ds'] = [
        _bladestiff2d_to_data(s, bay, '%s.bladestiff2ds[%d]' % (path, i), w)
        for i, s in enumerate(bay.bladestiff2ds)]
    # NOTE the stiffeners in the order they were added, as references
    stiffeners = []
    for i, s in enumerate(bay.stiffeners):
        p = '%s.stiffeners[%d]' % (path, i)
        if any(s is s1 for s1 in bay.bladestiff1ds):
            stiffeners.append(['bladestiff1ds', _index(s, bay.bladestiff1ds,
                                                       p, 'bladestiff1ds')])
        else:
            stiffeners.append(['bladestiff2ds', _index(s, bay.bladestiff2ds,
                                                       p, 'stiffeners')])
    data['stiffeners'] = stiffeners
    return data


def _stiffpanelbay_from_data(data, r):
    from .stiffpanelbay import StiffPanelBay
    _check_keys(data, _STIFFPANELBAY_ATTRS + ('panels', 'bladestiff1ds',
                'bladestiff2ds', 'stiffeners'), 'StiffPanelBay')
    bay = StiffPanelBay()
    _decode_attrs(bay, data, _STIFFPANELBAY_ATTRS, r)
    bay.panels = [_shell_from_data(p, r) for p in _list(data, 'panels')]
    bay.bladestiff1ds = [_bladestiff1d_from_data(s, bay, r)
                         for s in _list(data, 'bladestiff1ds')]
    bay.bladestiff2ds = [_bladestiff2d_from_data(s, bay, r)
                         for s in _list(data, 'bladestiff2ds')]
    bay.stiffeners = []
    for ref in _list(data, 'stiffeners'):
        if (not isinstance(ref, list) or len(ref) != 2
                or ref[0] not in ('bladestiff1ds', 'bladestiff2ds')):
            raise ValueError('Invalid reference to a stiffener: %r' % (ref, ))
        bay.stiffeners.append(_get(getattr(bay, ref[0]), ref[1], ref[0]))
    return bay


def _conn_to_data(conn, md, path, w):
    if conn is None:
        return None
    if not isinstance(conn, (list, tuple)):
        raise TypeError('Cannot save %s, expected a list of dictionaries, '
                        'got %r' % (path, type(conn).__name__))
    out = []
    for i, connecti in enumerate(conn):
        p = '%s[%d]' % (path, i)
        if not isinstance(connecti, dict):
            raise TypeError('Cannot save %s, expected a dictionary, got %r'
                            % (p, type(connecti).__name__))
        rest = {k: v for k, v in connecti.items() if k not in ('p1', 'p2')}
        data = _encode(rest, p, w)
        for name in ('p1', 'p2'):
            if name in connecti:
                data[name] = _index(connecti[name], md.panels,
                                    '%s[%r]' % (p, name), 'panels of the '
                                    'MultiDomain')
        out.append(data)
    return out


def _conn_from_data(data, panels, r):
    if data is None:
        return None
    if not isinstance(data, list):
        raise ValueError('Expected a list of connections, got %r' % (data, ))
    conn = []
    for connecti in data:
        if not isinstance(connecti, dict):
            raise ValueError('Expected a JSON object for a connection, got %r'
                             % (connecti, ))
        rest = {k: v for k, v in connecti.items() if k not in ('p1', 'p2')}
        out = _decode(rest, r)
        for name in ('p1', 'p2'):
            if name in connecti:
                out[name] = _get(panels, connecti[name], 'panels')
        conn.append(out)
    return conn


def _multidomain_to_data(md, path, w):
    data = _encode_attrs(md, _MULTIDOMAIN_ATTRS, path, w)
    data['panels'] = [_shell_to_data(p, '%s.panels[%d]' % (path, i), w)
                      for i, p in enumerate(md.panels)]
    data['conn'] = _conn_to_data(md.conn, md, path + '.conn', w)
    # NOTE the connections used last matter only without md.conn, see
    #      MultiDomain._resolve_conn()
    if md.conn is None and md._conn_used is not None:
        data['conn_used'] = _conn_to_data(md._conn_used, md,
                                          path + '._conn_used', w)
    data['T_tol'] = _encode(md._T_tol, path + '._T_tol', w)
    return data


def _multidomain_from_data(data, r):
    from .multidomain import MultiDomain
    _check_keys(data, _MULTIDOMAIN_ATTRS + ('panels', 'conn', 'conn_used',
                'T_tol'), 'MultiDomain')
    panels = [_shell_from_data(p, r) for p in _list(data, 'panels')]
    conn = _conn_from_data(data.get('conn'), panels, r)
    conn_method = data.get('conn_method', 'null-space')
    if conn_method not in ('penalty', 'null-space'):
        raise ValueError('Invalid conn_method: %r' % (conn_method, ))
    md = MultiDomain(panels, conn, conn_method=conn_method)
    _decode_attrs(md, data, _MULTIDOMAIN_ATTRS, r)
    md._conn_used = _conn_from_data(data.get('conn_used'), panels, r)
    if 'T_tol' in data:
        md._T_tol = _decode(data['T_tol'], r)
    return md


def _classes():
    r"""The classes that can be saved, imported on demand, since
    :mod:`panels.multidomain` imports matplotlib"""
    from .shell import Shell
    yield 'Shell', Shell, _shell_to_data, _shell_from_data
    from .stiffpanelbay import StiffPanelBay
    yield ('StiffPanelBay', StiffPanelBay, _stiffpanelbay_to_data,
           _stiffpanelbay_from_data)
    from .multidomain import MultiDomain
    yield ('MultiDomain', MultiDomain, _multidomain_to_data,
           _multidomain_from_data)


_DECODERS = {
    'Shell': _shell_from_data,
    'StiffPanelBay': _stiffpanelbay_from_data,
    'MultiDomain': _multidomain_from_data,
}


def _reject_constant(name):
    raise ValueError('Invalid JSON constant: %s' % name)


def save(obj, fname):
    r"""Save an object to a zip file

    Parameters
    ----------
    obj : :class:`.Shell`, :class:`.StiffPanelBay` or :class:`.MultiDomain`
        The object to be saved. It is not modified.
    fname : str, path-like or file object
        Name of the file, or a binary file object opened for writing, e.g.
        :class:`io.BytesIO`.

    Raises
    ------
    TypeError
        If ``obj`` is not one of the supported classes, or if one of its
        stored attributes has a value that cannot be saved, e.g. a function.

    """
    for typename, cls, encode, _ in _classes():
        if isinstance(obj, cls):
            break
    else:
        raise TypeError('Cannot save an object of type %r'
                        % type(obj).__name__)
    w = _Writer()
    envelope = {'type': typename,
                'format_version': FORMAT_VERSION,
                'panels_version': __version__,
                'data': encode(obj, typename, w)}
    # NOTE encoded completely before opening the file, such that an error
    #      does not leave a partial file
    s = json.dumps(envelope, allow_nan=False, indent=1)
    if not hasattr(fname, 'write'):
        fname = os.fspath(fname)
    with zipfile.ZipFile(fname, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_MODEL_JSON, s)
        for member, value in w.arrays.items():
            buf = io.BytesIO()
            np.save(buf, value, allow_pickle=False)
            zf.writestr(member, buf.getvalue())


def _load_zip(f):
    with zipfile.ZipFile(f, 'r') as zf:
        names = zf.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicated members in the zip file')
        if _MODEL_JSON not in names:
            raise ValueError('Missing member in the zip file: %r'
                             % _MODEL_JSON)
        try:
            d = json.loads(zf.read(_MODEL_JSON).decode('utf-8'),
                           parse_constant=_reject_constant)
        except UnicodeDecodeError as e:
            raise ValueError('Invalid %s: %s' % (_MODEL_JSON, e))
        if not isinstance(d, dict):
            raise ValueError('Expected a JSON object, got %r'
                             % type(d).__name__)
        _check_keys(d, _ENVELOPE_KEYS, 'the saved object')
        format_version = d.get('format_version')
        if (not isinstance(format_version, numbers.Integral)
                or isinstance(format_version, bool) or format_version < 1):
            raise ValueError('Invalid format_version: %r'
                             % (format_version, ))
        if format_version > FORMAT_VERSION:
            raise ValueError('format_version %d was saved by a newer version '
                             'of panels (%s), this version reads up to %d'
                             % (format_version, d.get('panels_version'),
                                FORMAT_VERSION))
        typename = d.get('type')
        if typename not in _DECODERS:
            raise ValueError('Unknown type: %r' % (typename, ))
        r = _Reader(zf)
        obj = _DECODERS[typename](d.get('data', {}), r)
        extra = r.names - r.used - {_MODEL_JSON}
        if extra:
            raise ValueError('Members not referenced by %s: %s'
                             % (_MODEL_JSON, ', '.join(sorted(extra))))
    return typename, obj


def _load_pickle(f, stacklevel):
    warnings.warn('Loading a pickle file of an older version of panels. '
                  'Pickle files are unsafe from untrusted sources, since '
                  'they can execute arbitrary code; save the object again to '
                  'convert it to the zip format', DeprecationWarning,
                  stacklevel=stacklevel)
    return pickle.load(f)


def _load(fname, suffixes=(), stacklevel=2):
    r"""Load a zip or a legacy pickle file, returns ``(typename, obj)``

    A name without the file extension is completed with the first suffix of
    ``suffixes`` for which the file exists. ``stacklevel`` is that of
    :func:`warnings.warn` as if called by this function.

    """
    if hasattr(fname, 'read'):
        f = fname
        if zipfile.is_zipfile(f):
            f.seek(0)
            return _load_zip(f)
        f.seek(0)
        obj = _load_pickle(f, stacklevel + 1)
        return type(obj).__name__, obj
    fname = os.fspath(fname)
    if not os.path.isfile(fname):
        for suffix in suffixes:
            if os.path.isfile(fname + suffix):
                fname = fname + suffix
                break
    if zipfile.is_zipfile(fname):
        return _load_zip(fname)
    with open(fname, 'rb') as f:
        obj = _load_pickle(f, stacklevel + 1)
    return type(obj).__name__, obj


def load(fname):
    r"""Load an object from a zip file created by :func:`.save`

    Pickle files saved by older versions of panels are also loaded, with a
    ``DeprecationWarning``.

    Parameters
    ----------
    fname : str, path-like or file object
        Name of the file, or a binary file object opened for reading, e.g.
        :class:`io.BytesIO`.

    Returns
    -------
    obj : :class:`.Shell`, :class:`.StiffPanelBay` or :class:`.MultiDomain`
        The object.

    Raises
    ------
    ValueError
        If the file is not valid, has unknown keys, missing or extra members,
        or was saved by a newer format version.

    """
    return _load(fname, stacklevel=3)[1]


def _load_type(fname, typename, suffixes):
    r"""Load an object that must be of type ``typename``, used by the
    functions ``load`` of the modules :mod:`panels.shell`,
    :mod:`panels.stiffpanelbay` and :mod:`panels.multidomain`"""
    loaded, obj = _load(fname, suffixes, stacklevel=4)
    if loaded != typename:
        raise ValueError('Expected a saved %s, got %s' % (typename, loaded))
    return obj
