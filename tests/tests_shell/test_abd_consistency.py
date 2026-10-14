r"""Consistency of the laminate stiffness used by the kernels

Every kernel must integrate the same constitutive matrix, ``Shell.ABD``:

- the analytical `[K_0]` of :meth:`.Shell.calc_kC` must be the numerically
  integrated one, ``calc_kC(c=zeros)``, to machine precision, for every model
  and for an anisotropic, unsymmetric laminate, with and without
  ``force_orthotropic_laminate``
- the linear static solution with the analytical `[K_0]` must be in
  equilibrium according to :meth:`.Shell.calc_fint`
- with ``force_orthotropic_laminate=True`` a simply supported plate under
  uniaxial compression is specially orthotropic and its buckling load is the
  closed-form solution, also when divided in domains with
  :class:`.MultiDomain`

``Shell.ABD`` is calculated from the laminate and cannot be set; another
stiffness, constant or one matrix per integration point, is the attribute
``Shell.ABDnxny``, ``None`` by default:

- ``ABDnxny[i, j]`` is the matrix of the point ``(xs[i], ys[j])`` of
  :meth:`.Shell.integration_points`, for full and partial domains, checked
  against the strain energy summed over these points
- the attribute gives the matrices of the argument ``ABDnxny``, also within
  a :class:`.MultiDomain`, and the stresses at the integration points
- a matrix whose blocks are not symmetric, or with the transverse shear
  coupled to the other strains, is rejected, the kernels reading only the
  upper entries of each block

"""
import io
import json
import sys
import zipfile
sys.path.append('../..')

import numpy as np
import pytest
from structsolve import lb, solve

from panels import modelDB
from panels.shell import Shell, load
from panels.multidomain import MultiDomain


MODELS = sorted(modelDB.db.keys())
#NOTE A16, A26, B, D16 and D26 all non-zero
STACK = [30, -60, 15]
LAMINAPROP = (140e9, 9e9, 0.3, 5e9, 5e9, 3e9)


def _shell(model, **kwargs):
    kw = dict(model=model, a=0.6, b=0.4, stack=STACK, plyt=0.5e-3,
              laminaprop=LAMINAPROP, m=6, n=6)
    if modelDB.db[model].get('requires_r'):
        kw['r'] = 0.8
    kw.update(kwargs)
    return Shell(**kw)


def _rel(A, B):
    A = A.toarray() if hasattr(A, 'toarray') else A
    B = B.toarray() if hasattr(B, 'toarray') else B
    return np.linalg.norm(A - B)/np.linalg.norm(B)


def test_laminate_is_anisotropic_and_unsymmetric():
    s = _shell('plate_clpt_donnell')
    lam = s.lam
    assert np.all(np.abs(lam.B) > 1e-6*np.abs(lam.A).max()*s.lam.h)
    for X in (lam.A, lam.D):
        assert abs(X[0, 2]) > 1e-3*X[0, 0]
        assert abs(X[1, 2]) > 1e-3*X[0, 0]


@pytest.mark.parametrize('ortho', [False, True])
@pytest.mark.parametrize('model', MODELS)
def test_analytical_equals_numerical_kC(model, ortho):
    s = _shell(model, force_orthotropic_laminate=ortho)
    kC_an = s.calc_kC()
    kC_num = s.calc_kC(c=np.zeros(s.get_size()))
    assert _rel(kC_an, kC_num) < 1e-12


@pytest.mark.parametrize('model', MODELS)
def test_force_orthotropic_laminate_changes_both_kC(model):
    aniso = _shell(model)
    ortho = _shell(model, force_orthotropic_laminate=True)
    zeros = np.zeros(aniso.get_size())
    d_an = (ortho.calc_kC() - aniso.calc_kC()).toarray()
    d_num = (ortho.calc_kC(c=zeros) - aniso.calc_kC(c=zeros)).toarray()
    # the flag is not ignored
    assert np.linalg.norm(d_an) > 1e-3*np.linalg.norm(aniso.calc_kC().toarray())
    assert np.linalg.norm(d_an - d_num) < 1e-12*np.linalg.norm(d_num)


@pytest.mark.parametrize('ortho', [False, True])
@pytest.mark.parametrize('model', MODELS)
def test_linear_static_is_in_equilibrium(model, ortho):
    r"""`\{F_{int}(c)\} = \{F_{ext}\}` for the linear solution, where the
    small load makes the non-linear terms of the internal forces vanish:
    the odd part `(\{F_{int}(c)\} - \{F_{int}(-c)\})/2` is free of the
    quadratic terms, and the cubic ones are `(w/h)^2 \approx 10^{-12}` times
    smaller"""
    s = _shell(model, force_orthotropic_laminate=ortho)
    # without the in-plane rigid body motions of the default boundary
    # conditions
    for edge in ('x1', 'x2', 'y1', 'y2'):
        setattr(s, edge + 'u', 0.)
        setattr(s, edge + 'v', 0.)
    s.add_point_load(0.4*s.a, 0.6*s.b, 0., 0., -1.e-6)
    fext = s.calc_fext()
    c = solve(s.calc_kC(), fext, silent=True)
    fint = (s.calc_fint(c) - s.calc_fint(-c))/2
    assert np.linalg.norm(fint - fext) < 1e-9*np.linalg.norm(fext)


def _navier_Nxx_cr(D, a, b, mmax=10):
    r"""Buckling load of a simply supported, specially orthotropic plate
    under uniaxial compression, minimum over the half-waves along `x`,
    one half-wave along `y`"""
    D11, D12, D22, D66 = D[0, 0], D[0, 1], D[1, 1], D[2, 2]
    m = np.arange(1, mmax + 1)
    return np.min(np.pi**2*(D11*(m/a)**2 + 2*(D12 + 2*D66)/b**2
                            + D22*(a/m)**2/b**4))


def _ortho_D(s):
    D = s.lam.D.copy()
    D[0, 2] = D[2, 0] = D[1, 2] = D[2, 1] = 0.
    return D


SYM_STACK = [30, -30, -30, 30]


def test_force_orthotropic_buckling_single_domain():
    s = _shell('plate_clpt_donnell', stack=SYM_STACK, m=8, n=8,
               force_orthotropic_laminate=True)
    s.Nxx = -1.
    eigvals, _ = lb(s.calc_kC(), s.calc_kG(), silent=True, num_eigvalues=3)
    ref = _navier_Nxx_cr(_ortho_D(s), s.a, s.b)
    assert np.isclose(eigvals[0], ref, rtol=1e-6)
    # without the flag the D16 and D26 terms lower the buckling load
    s.force_orthotropic_laminate = False
    eigvals, _ = lb(s.calc_kC(), s.calc_kG(), silent=True, num_eigvalues=3)
    assert eigvals[0] < 0.99*ref


def test_force_orthotropic_buckling_multidomain():
    r"""The plate divided in two domains along `x`, connected exactly with
    the null-space method, gives the buckling load of the single domain"""
    a, b = 0.6, 0.4
    xs = 0.4
    kw = dict(stack=SYM_STACK, b=b, m=8, n=8, force_orthotropic_laminate=True)
    p1 = _shell('plate_clpt_donnell', x0=0., y0=0., a=xs*a, **kw)
    p2 = _shell('plate_clpt_donnell', x0=xs*a, y0=0., a=(1 - xs)*a, **kw)
    # free edges at the interface, the connection gives the continuity
    p1.x2u = p1.x2v = p1.x2w = p1.x2wr = 1.
    p2.x1u = p2.x1v = p2.x1w = p2.x1wr = 1.
    for p in (p1, p2):
        p.Nxx = -1.
    conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0.)]
    md = MultiDomain([p1, p2], conn, conn_method='null-space')
    kC = md.reduce(md.calc_kC())
    kG = md.reduce(md.calc_kG(silent=True))
    eigvals, _ = lb(kC, kG, silent=True, num_eigvalues=3)
    ref = _navier_Nxx_cr(_ortho_D(p1), a, b)
    assert np.isclose(eigvals[0], ref, rtol=1e-4)


# The constitutive matrix is that of the laminate, Shell.ABD, unless the
# attribute Shell.ABDnxny is set

def test_ABD_is_read_only_and_follows_the_laminate():
    s = _shell('plate_clpt_donnell')
    with pytest.raises(AttributeError, match='ABDnxny'):
        s.ABD = 2*s.ABD
    with pytest.raises(AttributeError, match='ABDnxny'):
        _shell('plate_clpt_donnell', ABD=np.eye(6))
    assert np.array_equal(s.ABD, s.lam.ABD)
    s.force_orthotropic_laminate = True
    assert s.ABD[0, 2] == 0 and s.lam.ABD[0, 2] != 0
    s.force_orthotropic_laminate = False
    s.stack = [0., 90.]
    s.calc_kC()
    assert np.array_equal(s.ABD, s.lam.ABD) and s.lam.B[0, 0] != 0
    assert Shell().ABD is None


@pytest.mark.parametrize('model', MODELS)
def test_ABDnxny_is_not_allocated_by_default(model):
    s = _shell(model)
    c = 1e-6*np.random.default_rng(0).standard_normal(s.get_size())
    s.calc_kC()
    s.calc_kC(c=c, NLgeom=True)
    s.calc_kG(c=c)
    s.calc_fint(c)
    s.calc_kTv(c, c)
    s.stress(c, gridx=5, gridy=5)
    assert s.ABDnxny is None


def _field(s, nx, ny, seed=0):
    r"""Stiffness that differs at every integration point, with symmetric
    blocks: two laminates mixed with independent random fractions"""
    other = _shell(s.model, stack=[0., 45., 90.], plyt=0.4e-3,
                   force_orthotropic_laminate=s.force_orthotropic_laminate)
    rng = np.random.default_rng(seed)
    alpha = 1 + rng.random((nx, ny))[:, :, None, None]
    beta = rng.random((nx, ny))[:, :, None, None]
    return np.ascontiguousarray(alpha*s.ABD + beta*other.ABD)


def _partial(s):
    s.x1 = 0.1*s.a
    s.x2 = 0.8*s.a
    s.y1 = 0.25*s.b
    return s


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('model', MODELS)
def test_ABDnxny_point_indexing(model, partial):
    r"""`\{v\}^T [K_0(F)] \{c\}` against the sum of `w_x w_y
    \{\varepsilon(v)\}^T [F] \{\varepsilon(c)\}` over the points of
    :meth:`.Shell.integration_points`, with `nx \neq ny` taken from the shape
    of ``ABDnxny``; the reversed field must give another result"""
    s = _shell(model)
    if partial:
        _partial(s)
    nx, ny = 9, 7
    F = _field(s, nx, ny)
    rng = np.random.default_rng(1)
    c, v = rng.standard_normal((2, s.get_size()))
    zeros = np.zeros(s.get_size())
    kC = s.calc_kC(c=zeros, ABDnxny=F)
    xs, ys, wxs, wys = s.integration_points(nx, ny)
    x1, x2, y1, y2 = s.integration_limits()
    assert np.isclose(wxs.sum(), x2 - x1) and np.isclose(wys.sum(), y2 - y1)
    assert np.all((xs > x1) & (xs < x2)) and np.all((ys > y1) & (ys < y2))
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    names = s._strain_names()

    def eps(c):
        fields = s.strain(c, xs=X, ys=Y, NLgeom=False)[1]
        return np.stack([fields[name] for name in names], axis=-1)

    W = np.outer(wxs, wys)
    ref = np.einsum('ij,ijk,ijkl,ijl', W, eps(v), F, eps(c))
    #NOTE the round-off is relative to the terms of the sum, not to the sum
    scale = np.einsum('ij,ijk,ijkl,ijl', W, abs(eps(v)), abs(F), abs(eps(c)))
    assert abs(v @ kC @ c - ref) <= 1e-13*scale
    kC_rev = s.calc_kC(c=zeros, ABDnxny=F[::-1, ::-1])
    assert abs(v @ kC_rev @ c - ref) > 1e-6*abs(ref)


@pytest.mark.parametrize('model', MODELS)
def test_ABDnxny_attribute_equals_argument(model):
    s = _shell(model)
    nx, ny = 9, 7
    F = _field(s, nx, ny)
    c, v = 1e-4*np.random.default_rng(2).standard_normal((2, s.get_size()))
    zeros = np.zeros(s.get_size())
    ref = dict(
        kC=s.calc_kC(c=zeros, ABDnxny=F),
        kCNL=s.calc_kC(c=c, ABDnxny=F, NLgeom=True),
        kG=s.calc_kG(c=c, ABDnxny=F),
        fint=s.calc_fint(c, ABDnxny=F),
        kTv=s.calc_kTv(c, v, ABDnxny=F),
    )
    s.ABDnxny = F
    # without c the analytical kC, which ignores ABDnxny, must not be used
    assert _rel(s.calc_kC(), ref['kC']) == 0
    assert _rel(s.calc_kC(c=c, NLgeom=True), ref['kCNL']) == 0
    assert _rel(s.calc_kG(c=c), ref['kG']) == 0
    assert _rel(s.calc_fint(c), ref['fint']) == 0
    assert _rel(s.calc_kTv(c, v), ref['kTv']) == 0
    # nx and ny come from the shape of ABDnxny, not from the attributes
    assert (s.nx, s.ny) != (nx, ny)
    with pytest.raises(ValueError):
        s.calc_kC(nx=s.nx, ny=s.ny)


@pytest.mark.parametrize('model', MODELS)
def test_ABDnxny_constant(model):
    r"""A constant matrix, e.g. a stiffness that is not that of a stack of
    plies, gives the analytical matrix of that stiffness"""
    s = _shell(model)
    kC = s.calc_kC()
    s.ABDnxny = 2*s.ABD
    assert _rel(s.calc_kC(), 2*kC) < 1e-12


def test_ABDnxny_multidomain():
    r"""The panels of a :class:`.MultiDomain` use their attribute
    ``ABDnxny``, the penalty connections are not changed"""
    def model():
        p1 = _shell('plate_clpt_donnell', x0=0., y0=0., a=0.3)
        p2 = _shell('plate_clpt_donnell', x0=0.3, y0=0., a=0.3)
        conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=p1.a, xcte2=0.)]
        return MultiDomain([p1, p2], conn), p1
    md, p1 = model()
    kC = md.calc_kC()
    size = md.get_size()
    F = _field(p1, 9, 7)
    kp1_lam = p1.calc_kC(size=size, row0=p1.row_start, col0=p1.col_start)
    kp1 = p1.calc_kC(c=np.zeros(size), ABDnxny=F, size=size,
                     row0=p1.row_start, col0=p1.col_start)
    md2, p1b = model()
    p1b.ABDnxny = F
    assert _rel(md2.calc_kC(), kC - kp1_lam + kp1) < 1e-12


def test_stress_variable_stiffness():
    s = _shell('plate_fsdt_donnell')
    nx, ny = 9, 7
    F = _field(s, nx, ny)
    c = np.random.default_rng(3).standard_normal(s.get_size())
    s.ABDnxny = F
    plot_mesh, fields = s.stress(c, NLgeom=False)
    # at the integration points, ABDnxny[i, j] at (xs[i], ys[j])
    xs, ys = s.integration_points(nx, ny)[:2]
    assert np.array_equal(plot_mesh['Xs'], np.meshgrid(xs, ys, indexing='ij')[0])
    assert np.array_equal(plot_mesh['Ys'], np.meshgrid(xs, ys, indexing='ij')[1])
    eps = np.stack([fields[name] for name in s._strain_names()], axis=-1)
    N = np.einsum('ijkl,ijl->ijk', F, eps)
    #NOTE summed in another order, the round-off is relative to the terms,
    #     which cancel out at some points
    scale = np.einsum('ijkl,ijl->ijk', abs(F), abs(eps))
    for k, name in enumerate(s._stress_names()):
        assert fields[name].shape == (nx, ny)
        assert np.all(abs(fields[name] - N[..., k]) <= 1e-14*scale[..., k])
    with pytest.raises(ValueError, match='integration points'):
        s.stress(c, xs=xs, ys=ys)
    # any points, with one matrix per point
    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.55]])
    Y = 0.5*X
    ABD = F[:2, :3]
    fields = s.stress(c, ABD=ABD, xs=X, ys=Y, NLgeom=False)[1]
    eps = np.stack([fields[name] for name in s._strain_names()], axis=-1)
    Nxx = np.einsum('ijl,ijl->ij', ABD[..., 0, :], eps)
    scale = np.einsum('ijl,ijl->ij', abs(ABD[..., 0, :]), abs(eps))
    assert np.all(abs(fields['Nxx'] - Nxx) <= 1e-14*scale)
    with pytest.raises(ValueError, match='shape'):
        s.stress(c, ABD=F, xs=X, ys=Y)
    # a constant ABDnxny is used at any point
    s.ABDnxny = 2*s.ABD
    N2 = s.stress(c, xs=X, ys=Y, NLgeom=False)[1]['Nxx'].copy()
    s.ABDnxny = None
    N1 = s.stress(c, xs=X, ys=Y, NLgeom=False)[1]['Nxx']
    assert np.allclose(N2, 2*N1, rtol=1e-14, atol=0)


def test_save_load(tmp_path):
    s = _shell('plate_clpt_donnell', force_orthotropic_laminate=True)
    s.ABDnxny = _field(s, 9, 7)
    s.name = str(tmp_path / 'plate')
    s.save()
    s2 = load(s.name)
    assert np.array_equal(s2.ABDnxny, s.ABDnxny)
    assert np.array_equal(s2.ABD, s.ABD)
    s2.ABDnxny = None
    assert _rel(s2.calc_kC(), s.calc_kC(ABDnxny=s.ABD)) < 1e-12


def test_load_ABD_of_older_versions(tmp_path):
    r"""The files of panels < 0.13.9 store the constitutive matrix as
    ``ABD``, it is read and discarded, being that of the laminate"""
    s = _shell('plate_clpt_donnell')
    s.name = str(tmp_path / 'old')
    s.save()
    fname = s.name + '.shell.zip'
    with zipfile.ZipFile(fname) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    d = json.loads(members['model.json'])
    assert d['data']['ABDnxny'] is None
    del d['data']['ABDnxny']
    d['data']['ABD'] = {'__ndarray__': 'arrays/Shell.ABD.npy'}
    members['model.json'] = json.dumps(d).encode('utf-8')
    buf = io.BytesIO()
    np.save(buf, 3*s.ABD)
    members['arrays/Shell.ABD.npy'] = buf.getvalue()
    with zipfile.ZipFile(fname, 'w') as zf:
        for name, value in members.items():
            zf.writestr(name, value)
    s2 = load(fname)
    assert s2.ABDnxny is None
    assert np.array_equal(s2.ABD, s.ABD)


@pytest.mark.parametrize('model', MODELS)
def test_ABDnxny_checks(model):
    s = _shell(model)
    F = _field(s, 9, 7)
    ne = F.shape[-1]
    s.calc_kC(ABDnxny=F)
    # one wrong entry, the matrix not symmetric
    G = F.copy()
    G[2, 3, 0, 1] *= 1.01
    with pytest.raises(ValueError, match=r'symmetric the matrix.*\[2, 3\]'):
        s.calc_kC(ABDnxny=G)
    # symmetric matrix, B not symmetric: B12 != B21
    G = F.copy()
    d = 0.01*abs(G[2, 3, 0, 4])
    G[2, 3, 0, 4] += d
    G[2, 3, 4, 0] += d
    with pytest.raises(ValueError, match=r'symmetric the block B'):
        s.calc_kC(ABDnxny=G)
    with pytest.raises(ValueError, match=r'symmetric the block B'):
        s.calc_fint(np.zeros(s.get_size()), ABDnxny=G[2, 3])
    if ne == 13:
        for name, (i, j) in (('E', (0, 7)), ('F', (3, 7)), ('Dts', (9, 12))):
            G = F.copy()
            d = 0.01*abs(G[2, 3, 0, 0])
            G[2, 3, i, j] += d
            G[2, 3, j, i] += d
            with pytest.raises(ValueError, match='symmetric the block ' + name):
                s.calc_kC(ABDnxny=G)
    if ne > 6:
        G = F.copy()
        d = 0.01*abs(G[2, 3, 0, 0])
        G[2, 3, 0, ne - 1] = G[2, 3, ne - 1, 0] = d
        with pytest.raises(ValueError, match='zero the coupling'):
            s.calc_kC(ABDnxny=G)
    # round-off is accepted
    G = F*(1 + 1e-14*np.random.default_rng(4).standard_normal(F.shape))
    s.calc_kC(ABDnxny=G)
    with pytest.raises(ValueError, match='shape'):
        s.calc_kC(ABDnxny=np.zeros((ne + 1, ne + 1)))
    with pytest.raises(ValueError, match='shape'):
        s.calc_kC(ABDnxny=np.zeros((9, ne, ne)))
    # the attribute is checked as well
    s.ABDnxny = G[0, 0] + 0.01*abs(G[0, 0]).max()*np.eye(ne, k=1)
    with pytest.raises(ValueError, match='symmetric the matrix'):
        s.calc_kC()
