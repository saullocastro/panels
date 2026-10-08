r"""Saving and loading in the JSON + NumPy zip container, see panels.json_io"""
import copy
import io
import json
import pickle
import zipfile

import numpy as np
from numpy.testing import assert_array_equal
import pytest
from structsolve import lb, static

from panels import json_io
from panels.shell import Shell
from panels.shell import load as load_shell
from panels.stiffpanelbay import StiffPanelBay
from panels.stiffpanelbay import load as load_bay
from panels.multidomain import MultiDomain
from panels.multidomain import load as load_md


LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.0e9)


def _shell(model='plate_clpt_donnell', **kwargs):
    kw = dict(a=0.6, b=0.4, stack=[0, 45, -45, 90], plyt=0.2e-3,
              laminaprop=LAMINAPROP, rho=1600., m=7, n=6, model=model)
    if 'cylshell' in model:
        kw['r'] = 0.8
    kw.update(kwargs)
    s = Shell(**kw)
    s.Nxx = -1.
    s.y1w = 1.
    s.add_point_load(0.3, 0.2, 0, 0, -10.)
    s.add_point_load(0.6, 0.2, -5., 0, 0, cte=False)
    s.add_pressure_load(-1.e3, x1=0.1, follower=False)
    s.add_point_pd(0.2, 0.1, 1.e6, 0., 1.e6, 0., 1.e6, 1.e-4)
    return s


def _save(obj):
    buf = io.BytesIO()
    obj.save(buf)
    return buf.getvalue()


def _members(data):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _model_json(data):
    def reject(name):
        raise ValueError('non-strict JSON constant %s' % name)
    return json.loads(_members(data)['model.json'].decode('utf-8'),
                      parse_constant=reject)


def _zip(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for name, value in members.items():
            zf.writestr(name, value)
    return buf.getvalue()


def _rewrite(data, model=None, members=None):
    r"""A zip file with ``model.json`` and the other members replaced"""
    old = _members(data)
    if model is not None:
        old['model.json'] = json.dumps(model).encode('utf-8')
    if members is not None:
        old = dict(members, **{'model.json': old['model.json']})
    return io.BytesIO(_zip(old))


def _assert_same_matrices(m1, m2):
    assert_array_equal(m1.toarray(), m2.toarray())


def _shell_lb(s):
    kC = s.calc_kC(c=None)
    kG = s.calc_kG()
    eigvals, eigvecs = lb(kC, kG, silent=True, num_eigvalues=3)
    return kC, kG, eigvals, eigvecs


def _shell_static(s):
    kC = s.calc_kC()
    fext = s.calc_fext()
    _, cs = static(kC, fext, silent=True)
    return cs[0]


SHELL_MODELS = ['plate_clpt_donnell', 'cylshell_clpt_sanders',
                'plate_fsdt_donnell', 'cylshell_tsdt_donnell']


@pytest.mark.parametrize('model', SHELL_MODELS)
def test_shell_before_analysis(model):
    s = _shell(model, fsdt_shear_correction=5/6. if 'fsdt' in model
               else 'rohwer')
    s2 = load_shell(io.BytesIO(_save(s)))
    assert type(s2) is Shell
    for name in json_io._SHELL_ATTRS:
        v, v2 = getattr(s, name), getattr(s2, name)
        if isinstance(v, np.ndarray):
            assert_array_equal(v2, v)
        elif isinstance(v, tuple):
            assert v2 == list(v)
        elif name in ('laminaprops', 'pressure_loads'):
            assert v2 == [list(lp) for lp in v]
        else:
            assert v2 == v, name
    assert_array_equal(s2.lam.ABD, s.lam.ABD)
    kC, kG, eigvals, eigvecs = _shell_lb(s)
    kC2, kG2, eigvals2, eigvecs2 = _shell_lb(s2)
    _assert_same_matrices(kC2, kC)
    _assert_same_matrices(kG2, kG)
    assert_array_equal(eigvals2, eigvals)
    assert_array_equal(eigvecs2, eigvecs)
    assert_array_equal(_shell_static(s2), _shell_static(s))


@pytest.mark.parametrize('model', SHELL_MODELS)
def test_shell_after_analysis(model):
    s = _shell(model)
    kC, kG, eigvals, eigvecs = _shell_lb(s)
    s.results['eigvals'] = eigvals
    s.results['eigvecs'] = eigvecs
    c = _shell_static(s)
    s.increments = [0.5, 1.]
    s.results['c'] = c
    s.stress(c, gridx=7, gridy=5)
    data = _save(s)
    # the saved object is not modified
    assert s.matrices['kC'] is not None
    assert s.lam is not None
    s2 = load_shell(io.BytesIO(data))
    for k in ('eigvals', 'eigvecs', 'c'):
        assert s2.results[k].dtype == s.results[k].dtype
        assert_array_equal(s2.results[k], s.results[k])
    assert s2.increments == [0.5, 1.]
    for k, v in s.fields.items():
        if v is None:
            assert s2.fields[k] is None
        else:
            assert s2.fields[k].shape == v.shape == (5, 7)
            assert_array_equal(s2.fields[k], v)
    for k, v in s.plot_mesh.items():
        assert_array_equal(s2.plot_mesh[k], v)
    assert all(v is None for v in s2.matrices.values())
    kC2, kG2, eigvals2, eigvecs2 = _shell_lb(s2)
    _assert_same_matrices(kC2, kC)
    assert_array_equal(eigvals2, eigvals)
    assert_array_equal(eigvecs2, eigvecs)
    assert_array_equal(_shell_static(s2), c)
    s2.stress(c, gridx=7, gridy=5)
    for k, v in s.fields.items():
        if v is not None:
            assert_array_equal(s2.fields[k], v)


def _bay(stiffener):
    tskin = 4*0.2e-3
    b = 100*tskin
    bay = StiffPanelBay()
    bay.name = 'bay'
    bay.a = 2*b
    bay.b = b
    bay.model = 'plate_clpt_donnell'
    bay.stack = [0, 45, -45, 90]
    bay.plyt = 0.2e-3
    bay.laminaprop = LAMINAPROP
    bay.rho = 1600.
    bay.m = 7
    bay.n = 7
    bay.add_panel(0, b/2, Nxx=-1.)
    bay.add_panel(b/2, b, Nxx=-1.)
    kw = dict(ys=b/2, bb=10*tskin, bstack=[0, 90, 90, 0], bplyt=0.2e-3,
              blaminaprop=LAMINAPROP, bf=9*tskin, fstack=[0, 90, 90, 0],
              fplyt=0.2e-3, flaminaprop=LAMINAPROP)
    if stiffener == '1d':
        bay.add_bladestiff1d(Fx=-1., **kw)
    elif stiffener == '2d':
        s = bay.add_bladestiff2d(mf=6, nf=5, **kw)
        s.flange.Nxx = -1.
        s.forces_flange.append([bay.a/2, 0.5*s.flange.b, 0., 0., 1.])
    else:
        bay.add_bladestiff1d(Fx=-1., **kw)
        bay.add_bladestiff2d(mf=6, nf=5, ys=b, bf=9*tskin,
                             fstack=[0, 90, 90, 0], fplyt=0.2e-3,
                             flaminaprop=LAMINAPROP)
    bay.forces_skin.append([bay.a/2, b/4, 0., 0., -1.])
    return bay


def _bay_lb(bay):
    kC = bay.calc_kC(silent=True)
    kG = bay.calc_kG(silent=True)
    eigvals, eigvecs = lb(kC, kG, silent=True, num_eigvalues=3)
    return kC, kG, eigvals, eigvecs


@pytest.mark.parametrize('stiffener', ['1d', '2d', 'both'])
def test_stiffpanelbay(stiffener):
    bay = _bay(stiffener)
    bay2 = load_bay(io.BytesIO(_save(bay)))
    kC, kG, eigvals, eigvecs = _bay_lb(bay)
    kC2, kG2, eigvals2, eigvecs2 = _bay_lb(bay2)
    _assert_same_matrices(kC2, kC)
    _assert_same_matrices(kG2, kG)
    assert_array_equal(eigvals2, eigvals)
    assert_array_equal(eigvecs2, eigvecs)
    assert_array_equal(bay2.calc_fext(silent=True), bay.calc_fext(silent=True))

    # after the analysis
    bay.uvw_skin(eigvecs[:, 0], gridx=6, gridy=5)
    data = _save(bay)
    assert bay.kC is not None
    bay3 = load_bay(io.BytesIO(data))
    assert bay3.kC is None
    for name in ('u', 'v', 'w', 'phix', 'phiy', 'Xs', 'Ys'):
        assert_array_equal(getattr(bay3, name), getattr(bay, name))
    assert len(bay3.stiffeners) == len(bay.stiffeners)
    for s, s3 in zip(bay.stiffeners, bay3.stiffeners):
        assert type(s3) is type(s)
        assert s3.bay is bay3
        assert s3.panel1 is bay3.panels[bay.panels.index(s.panel1)]
        assert s3.panel2 is bay3.panels[bay.panels.index(s.panel2)]
    assert ([s in bay3.bladestiff1ds for s in bay3.stiffeners]
            == [s in bay.bladestiff1ds for s in bay.stiffeners])
    kC3, _, eigvals3, eigvecs3 = _bay_lb(bay3)
    _assert_same_matrices(kC3, kC)
    assert_array_equal(eigvals3, eigvals)
    assert_array_equal(eigvecs3, eigvecs)


def _multidomain(conn_method):
    kw = dict(b=0.3, stack=[0, 45, -45, 90], plyt=0.2e-3, laminaprop=LAMINAPROP,
              model='plate_clpt_donnell', x0=0., y0=0., group='skin')
    p1 = Shell(a=0.4, m=6, n=6, **kw)
    p2 = Shell(a=0.5, m=7, n=6, **dict(kw, x0=0.4))
    p3 = Shell(a=0.4, m=6, n=5, **dict(kw, stack=[0, 90, 90, 0],
                                         group='padup'))
    for p in (p1, p2, p3):
        p.x1u = p.x1v = p.x1w = 1.
        p.x2u = p.x2v = p.x2w = 1.
    p1.x1u = p1.x1v = p1.x1w = 0.
    p2.x2w = 0.
    p2.add_point_load(0.5, 0.15, 0, 0, -10., cte=False)
    conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0., kt=1.e8,
                 kr=1.e4),
            dict(p1=p3, p2=p1, func='SB')]
    return MultiDomain([p1, p2, p3], conn, conn_method=conn_method,
                       name='plate')


def _md_static(md):
    kC = md.calc_kC()
    fext = md.calc_fext()
    _, cs = static(md.reduce(kC), md.reduce(fext), silent=True)
    return kC, md.expand(cs[0])


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_multidomain(conn_method):
    md = _multidomain(conn_method)
    md2 = load_md(io.BytesIO(_save(md)))
    assert type(md2) is MultiDomain
    assert md2.name == 'plate'
    assert md2.conn_method == conn_method
    kC, c = _md_static(md)
    kC2, c2 = _md_static(md2)
    _assert_same_matrices(kC2, kC)
    assert_array_equal(c2, c)

    # after the analysis
    md.fext = md.calc_fext()
    md.update_TSL_history(np.linspace(0., 0.5, 12).reshape(3, 4))
    data = _save(md)
    assert md.T is not None or conn_method == 'penalty'
    md3 = load_md(io.BytesIO(data))
    assert md3.T is None
    assert_array_equal(md3.dmg_index, md.dmg_index)
    assert_array_equal(md3.fext, md.fext)
    for connecti, connecti3 in zip(md.conn, md3.conn):
        assert connecti3['p1'] is md3.panels[md.panels.index(connecti['p1'])]
        assert connecti3['p2'] is md3.panels[md.panels.index(connecti['p2'])]
        assert ({k: v for k, v in connecti3.items() if k not in ('p1', 'p2')}
                == {k: v for k, v in connecti.items() if k not in ('p1', 'p2')})
    for p, p3 in zip(md.panels, md3.panels):
        assert (p3.row_start, p3.row_end) == (p.row_start, p.row_end)
        assert p3.group == p.group
    kC3, c3 = _md_static(md3)
    _assert_same_matrices(kC3, kC)
    assert_array_equal(c3, c)


def test_multidomain_conn_used():
    md = _multidomain('penalty')
    conn = md.conn
    md.conn = None
    md.calc_kC(conn)
    md2 = load_md(io.BytesIO(_save(md)))
    assert md2.conn is None
    assert len(md2._conn_used) == len(conn)
    _assert_same_matrices(md2.calc_kC(), md.calc_kC())


def test_multidomain_panel_not_in_assembly():
    md = _multidomain('penalty')
    md.conn[0] = dict(md.conn[0], p2=_shell())
    with pytest.raises(ValueError, match='not one of the panels'):
        _save(md)


def test_strict_json_and_arrays():
    s = _shell()
    s.Mach = float('inf')
    s.beta = float('nan')
    s.gamma = -np.inf
    s.air_speed = np.float32(2.5)
    s.num_eigvalues = np.int64(7)
    s.force_orthotropic_laminate = np.bool_(True)
    s.point_x = (1., 0., 0.)
    arrays = {
        'f8': np.arange(6.).reshape(2, 3),
        'f4': np.arange(4, dtype=np.float32),
        'i4': np.arange(3, dtype=np.int32),
        'bool': np.array([True, False]),
        'c16': np.array([1 + 2j, np.nan]),
        'fortran': np.asfortranarray(np.arange(12.).reshape(3, 4)),
        'scalar': np.array(3.),
        'empty': np.zeros((0, 4)),
        'nonfinite': np.array([np.nan, np.inf, -np.inf]),
    }
    s.results.update(arrays)
    data = _save(s)
    d = _model_json(data)
    assert d['type'] == 'Shell'
    assert d['format_version'] == 1
    assert d['data']['Mach'] == 'Infinity'
    assert d['data']['beta'] == 'NaN'
    assert d['data']['gamma'] == '-Infinity'
    assert d['data']['lam']['type'] == 'Laminate'
    assert d['data']['results']['f8'] == {'__ndarray__':
                                          'arrays/Shell.results.f8.npy'}
    assert set(_members(data)) == ({'model.json'}
        | {'arrays/Shell.ABD.npy'}
        | {'arrays/Shell.results.%s.npy' % k for k in arrays})
    s2 = load_shell(io.BytesIO(data))
    assert s2.Mach == np.inf and s2.gamma == -np.inf and np.isnan(s2.beta)
    assert s2.air_speed == 2.5 and type(s2.air_speed) is float
    assert s2.num_eigvalues == 7 and type(s2.num_eigvalues) is int
    assert s2.force_orthotropic_laminate is True
    assert s2.point_x == [1., 0., 0.]
    for k, v in arrays.items():
        v2 = s2.results[k]
        assert v2.dtype == v.dtype and v2.shape == v.shape, k
        assert v2.flags.f_contiguous == v.flags.f_contiguous
        assert_array_equal(v2, v)


def test_file_names(tmp_path):
    s = _shell()
    s.name = str(tmp_path / 'plate')
    s.save()
    assert (tmp_path / 'plate.shell.zip').is_file()
    for name in (s.name, s.name + '.shell.zip', tmp_path / 'plate.shell.zip'):
        assert_array_equal(load_shell(name).ABD, s.ABD)
    assert_array_equal(json_io.load(s.name + '.shell.zip').ABD, s.ABD)
    bay = _bay('2d')
    bay.name = str(tmp_path / 'bay')
    bay.save()
    assert (tmp_path / 'bay.stiffpanelbay.zip').is_file()
    assert len(load_bay(bay.name).panels) == 2
    md = _multidomain('penalty')
    assert MultiDomain([]).name == 'multidomain'
    md.name = str(tmp_path / 'md')
    md.save()
    assert (tmp_path / 'md.multidomain.zip').is_file()
    assert len(load_md(md.name).panels) == 3
    s.save(tmp_path / 'other.zip')
    assert type(json_io.load(tmp_path / 'other.zip')) is Shell
    with pytest.raises(ValueError, match='Expected a saved StiffPanelBay, '
                       'got Shell'):
        load_bay(tmp_path / 'other.zip')
    with pytest.raises(FileNotFoundError):
        load_shell(tmp_path / 'missing')


def test_unsupported_values(tmp_path):
    s = _shell()
    s.add_distr_load_fixed_x(0.3, funcz=lambda y: 1.)
    fname = tmp_path / 'distr.zip'
    with pytest.raises(TypeError, match=r"Shell\.distr_loads\[0\]\[4\], of "
                       "type 'function'"):
        s.save(fname)
    assert not fname.exists()
    s = _shell()
    s.add_pressure_load(lambda x, y: 1.e3*x)
    with pytest.raises(TypeError, match='pressure_loads'):
        _save(s)
    s = _shell()
    s.results['object'] = np.array([1, 'a'], dtype=object)
    with pytest.raises(TypeError, match='dtype object'):
        _save(s)
    s = _shell()
    s.results['matrix'] = s.calc_kC()
    with pytest.raises(TypeError, match=r"Shell\.results\['matrix'\]"):
        _save(s)
    s = _shell()
    s.results['Infinity'] = 'NaN'
    with pytest.raises(ValueError, match='reserved'):
        _save(s)
    with pytest.raises(TypeError, match='Cannot save an object'):
        json_io.save(object(), io.BytesIO())
    # a string attribute is stored as it is
    s = _shell(model='plate_clpt_donnell')
    s.name = 'NaN'
    assert load_shell(io.BytesIO(_save(s))).name == 'NaN'


def test_invalid_files():
    s = _shell()
    s.results['eigvals'] = np.arange(3.)
    data = _save(s)
    d = _model_json(data)

    def load(model=None, members=None):
        return load_shell(_rewrite(data, model, members))

    load()
    with pytest.raises(ValueError, match='newer version'):
        load(dict(d, format_version=2))
    for v in (0, '1', True, None):
        with pytest.raises(ValueError, match='Invalid format_version'):
            load(dict(d, format_version=v))
    with pytest.raises(ValueError, match='Unknown type'):
        load(dict(d, type='Laminate'))
    with pytest.raises(ValueError, match='Unknown keys for the saved'):
        load(dict(d, extra=1))
    with pytest.raises(ValueError, match='Unknown keys for Shell: unknown'):
        load(dict(d, data=dict(d['data'], unknown=1)))
    members = _members(data)
    missing = dict(members)
    del missing['arrays/Shell.results.eigvals.npy']
    with pytest.raises(ValueError, match='Missing member'):
        load(members=missing)
    with pytest.raises(ValueError, match='not referenced'):
        load(members=dict(members, **{'arrays/extra.npy':
                                      members['arrays/Shell.ABD.npy']}))
    with pytest.raises(ValueError, match="Missing member in the zip file: "
                       "'model.json'"):
        load_shell(io.BytesIO(_zip({k: v for k, v in members.items()
                                    if k != 'model.json'})))
    buf = io.BytesIO()
    np.save(buf, np.array([1, 'a'], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError, match='Object arrays cannot be loaded'):
        load(members=dict(members, **{'arrays/Shell.ABD.npy':
                                      buf.getvalue()}))
    buf = io.BytesIO()
    np.savez(buf, a=np.arange(3.))
    with pytest.raises(ValueError, match='Invalid array'):
        load(members=dict(members, **{'arrays/Shell.ABD.npy':
                                      buf.getvalue()}))
    for ref in ('../model.json', 'arrays/missing.npy', 'arrays/a b.npy', 1):
        bad = copy.deepcopy(d)
        bad['data']['ABD'] = {'__ndarray__': ref}
        with pytest.raises(ValueError, match='member'):
            load(bad)
    bad = copy.deepcopy(d)
    bad['data']['ABD'] = {'__sparse__': {}}
    with pytest.raises(ValueError, match='Unknown marker'):
        load(bad)
    bad = copy.deepcopy(d)
    bad['data']['lam']['data']['unknown'] = 1.
    with pytest.raises(ValueError, match='Unknown keys for Laminate'):
        load(bad)
    raw = json.dumps(dict(d, data=dict(d['data'], Mach=float('nan'))))
    assert 'NaN' in raw
    with pytest.raises(ValueError, match='Invalid JSON constant: NaN'):
        load_shell(io.BytesIO(_zip(dict(members, **{'model.json':
                                                    raw.encode('utf-8')}))))


UNPICKLED = []


def _unpickled():
    UNPICKLED.append(1)
    return 'executed'


class _Payload:
    def __reduce__(self):
        return (_unpickled, ())


def test_pickle_files_are_not_loaded(tmp_path):
    """Pickle files, e.g. of older versions, can execute arbitrary code"""
    data = pickle.dumps(_Payload())
    (tmp_path / 'plate.Shell').write_bytes(data)
    for f in (tmp_path / 'plate.Shell', io.BytesIO(data)):
        with pytest.raises(ValueError, match='Not a zip file saved by panels'):
            load_shell(f)
        with pytest.raises(ValueError, match='Not a zip file saved by panels'):
            json_io.load(f)
    # the extension of the old pickle files is no longer completed
    with pytest.raises(FileNotFoundError):
        load_shell(tmp_path / 'plate')
    assert UNPICKLED == []


def test_max_size(monkeypatch):
    s = _shell()
    s.results['eigvals'] = np.arange(3.)
    data = _save(s)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        total = sum(zi.file_size for zi in zf.infolist())
        json_size = zf.getinfo('model.json').file_size
    load_shell(io.BytesIO(data), max_size=total)
    json_io.load(io.BytesIO(data), max_size=total)
    with pytest.raises(ValueError, match='exceeds max_size'):
        load_shell(io.BytesIO(data), max_size=total - 1)
    with pytest.raises(ValueError, match="exceeds max_size, reading "
                       "'model.json'"):
        load_shell(io.BytesIO(data), max_size=json_size - 1)

    # a highly compressed member is rejected before it is decompressed
    big = io.BytesIO()
    np.save(big, np.zeros(2**23))
    members = _members(data)
    members['arrays/Shell.ABD.npy'] = big.getvalue()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name, value in members.items():
            zf.writestr(name, value)
    assert len(buf.getvalue()) < 2**20
    read = []
    original = zipfile.ZipFile.read

    def spy(self, name, *args, **kwargs):
        read.append(name)
        return original(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, 'read', spy)
    with pytest.raises(ValueError, match="exceeds max_size, reading "
                       "'arrays/Shell.ABD.npy'"):
        load_shell(io.BytesIO(buf.getvalue()), max_size=2**25)
    assert 'arrays/Shell.ABD.npy' not in read
    # the default of 4 GiB
    assert json_io.MAX_SIZE == 4*2**30
    assert type(load_shell(io.BytesIO(buf.getvalue()))) is Shell


def test_still_picklable():
    s = _shell()
    s2 = copy.deepcopy(s)
    assert_array_equal(s2.calc_kC().toarray(), s.calc_kC().toarray())
    bay = pickle.loads(pickle.dumps(_bay('both')))
    assert len(bay.stiffeners) == 2
    md = pickle.loads(pickle.dumps(_multidomain('null-space')))
    assert md.name == 'plate'
