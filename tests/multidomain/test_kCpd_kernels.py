r"""Penalty kernels of the prescribed displacements

The kernels of :mod:`panels.multidomain.connections.kCpd` are compared with
the penalty stiffness `k \int \{g\} \{g\}^T ds` integrated from the rows of
``fg`` of the field module, see ``theory/multidomain_penalization/kCpd.py``,
and with the penalty energy computed from the displacement field of
:meth:`.Shell.uvw`, for the models with 3 and 5 degrees of freedom per term.
A plate whose displacements are prescribed with these kernels and the forces
of :func:`.shell_fext` reaches the values imposed exactly with the null-space
method.

"""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
import panels.modelDB as modelDB
from panels.multidomain import MultiDomain
from panels.multidomain.connections import fkCpd, fkCld_xcte, fkCld_ycte


#NOTE the CLPT models have 3 DOFs per term, the FSDT and TSDT models 5
MODELS = ['plate_clpt_donnell', 'plate_fsdt_donnell', 'plate_tsdt_donnell',
          'cylshell_clpt_sanders', 'cylshell_fsdt_sanders',
          'cylshell_tsdt_donnell']


def plate(a, b, m, n, model='plate_clpt_donnell'):
    r = 0.8 if model.startswith('cylshell') else None
    return Shell(a=a, b=b, r=r, m=m, n=n, model=model, stack=[0],
                 plyt=1e-3, laminaprop=(70e9, 0.3))


def random_bcs(p, rng):
    for d in 'xy':
        for e in '12':
            for f in 'uvw':
                for r in ('', 'r'):
                    setattr(p, d + e + f + r, float(rng.integers(0, 2)))


def line_penalty(p, comp, x, y, k, size):
    r"""Penalty stiffness `k \int g_i g_j ds` of the component ``comp`` along
    the line ``x`` or ``y``, integrated from the rows of ``fg``"""
    fg = modelDB.db[p.model]['field'].fg
    g = np.zeros((5, p.get_size()))
    eta, w = leggauss(max(2*p.m, 2*p.n, 20))
    K = np.zeros((size, size))
    sl = slice(p.col_start, p.col_start + p.get_size())
    for e, wi in zip(eta, w):
        if y is None:
            fg(g, x, (e + 1)*p.b/2, p)
            jac = p.b/2
        else:
            fg(g, (e + 1)*p.a/2, y, p)
            jac = p.a/2
        K[sl, sl] += k*wi*jac*np.outer(g[comp], g[comp])
    return K


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('line', ['x', 'y'])
@pytest.mark.parametrize('comp', [0, 1, 2])
def test_line_penalty_kernels(line, comp, model):
    r"""The kernels of the line prescribed displacements, :func:`.fkCld_xcte`
    and :func:`.fkCld_ycte`, are `k \int g_i g_j ds` of the component; the
    stiffnesses of ``ku`` and ``kv`` were placed at the rows and columns of
    ``w`` up to version 0.11.1, and those of the models with 5 DOFs per term
    with a stride of 3 DOFs per term up to version 0.13.20"""
    rng = np.random.default_rng(comp)
    p = plate(0.4, 0.3, m=7, n=6, model=model)
    random_bcs(p, rng)
    p.col_start = p.get_size()
    size = 2*p.get_size()
    ks = [0., 0., 0.]
    ks[comp] = 1.e6
    if line == 'x':
        K = fkCld_xcte(*ks, p, 0.27, size, p.col_start, p.col_start)
        ref = line_penalty(p, comp, 0.27, None, 1e6, size)
    else:
        K = fkCld_ycte(*ks, p, 0.11, size, p.col_start, p.col_start)
        ref = line_penalty(p, comp, None, 0.11, 1e6, size)
    K = finalize_symmetric_matrix(K).toarray()
    assert np.allclose(K, ref, rtol=0, atol=1e-8*np.abs(ref).max())


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('comp', [0, 1, 2])
def test_point_penalty_kernel(comp, model):
    r"""The kernel of the point prescribed displacements, :func:`.fkCpd`, is
    `k \{g\} \{g\}^T` of the component"""
    p = plate(0.4, 0.3, m=7, n=6, model=model)
    p.col_start = 0
    size = p.get_size()
    ks = [0., 0., 0.]
    ks[comp] = 1.e6
    K = finalize_symmetric_matrix(fkCpd(*ks, p, 0.27, 0.11, size, 0,
                                        0)).toarray()
    g = np.zeros((5, size))
    modelDB.db[p.model]['field'].fg(g, 0.27, 0.11, p)
    ref = 1e6*np.outer(g[comp], g[comp])
    assert np.allclose(K, ref, rtol=0, atol=1e-8*np.abs(ref).max())


def displacements(p, c, xs, ys):
    r"""`u, v, w` of :meth:`.Shell.uvw` at the points ``xs``, ``ys``"""
    _, fields = p.uvw(c, xs=np.asarray(xs, dtype=float),
                      ys=np.asarray(ys, dtype=float))
    return [np.asarray(fields[k]).ravel() for k in ('u', 'v', 'w')]


@pytest.mark.parametrize('model', MODELS)
def test_penalty_energy(model):
    r"""For random Ritz constants `\{c\}`, `\{c\}^T [K] \{c\}` of the
    kernels is twice the penalty energy of a zero prescribed displacement,
    `k_u u^2 + k_v v^2 + k_w w^2` at the point and its integral along the
    lines, with `u, v, w` of :meth:`.Shell.uvw`, the panel being placed after
    another of the same size in the assembly"""
    rng = np.random.default_rng(3)
    p = plate(0.4, 0.3, m=7, n=6, model=model)
    random_bcs(p, rng)
    n = p.get_size()
    size = 2*n
    row0 = n
    ks = (1.3e6, 0.7e6, 2.1e6)
    xp, yp = 0.27, 0.11
    xi, wi = leggauss(20)
    for _ in range(3):
        c = rng.uniform(-1, 1, n)
        cc = np.zeros(size)
        cc[row0:] = c

        K = finalize_symmetric_matrix(fkCpd(*ks, p, xp, yp, size, row0,
                                            row0))
        uvw = displacements(p, c, [xp], [yp])
        ref = sum(k*d[0]**2 for k, d in zip(ks, uvw))
        assert np.isclose(cc @ K @ cc, ref, rtol=1e-10)

        K = finalize_symmetric_matrix(fkCld_xcte(*ks, p, xp, size, row0,
                                                 row0))
        ys = (xi + 1)*p.b/2
        uvw = displacements(p, c, np.full_like(ys, xp), ys)
        ref = sum(k*(wi*p.b/2) @ d**2 for k, d in zip(ks, uvw))
        assert np.isclose(cc @ K @ cc, ref, rtol=1e-10)

        K = finalize_symmetric_matrix(fkCld_ycte(*ks, p, yp, size, row0,
                                                 row0))
        xs = (xi + 1)*p.a/2
        uvw = displacements(p, c, xs, np.full_like(xs, yp))
        ref = sum(k*(wi*p.a/2) @ d**2 for k, d in zip(ks, uvw))
        assert np.isclose(cc @ K @ cc, ref, rtol=1e-10)


@pytest.mark.parametrize('model', ['plate_clpt_donnell', 'plate_fsdt_donnell',
                                   'plate_tsdt_donnell'])
def test_penalty_reaches_prescribed_values(model):
    r"""A simply supported plate with `u, v, w` prescribed at a point, `w`
    along a line `x = x_l` and `v` along a line `y = y_l`, imposed with the
    penalty stiffnesses of the kernels and the forces of :func:`.shell_fext`,
    reaches the prescribed values, as with the null-space method"""
    a, b = 0.4, 0.3
    xp, yp = 0.13, 0.21
    xl, yl = 0.25, 0.09
    up, vp, wp = 1e-5, -2e-5, 1e-3
    wl = lambda y: 2e-3*4*y*(b - y)/b**2
    vl = lambda x: 3e-5*4*x*(a - x)/a**2
    #NOTE the error of the penalty method decreases with 1/k
    k = 1e4*70e9*1e-3
    p = plate(a, b, m=8, n=8, model=model)
    p.add_point_pd(xp, yp, k, up, k, vp, k, wp)
    p.add_distr_pd_fixed_x(xl, kw=k, funcw=wl)
    p.add_distr_pd_fixed_y(yl, kv=k, funcv=vl)

    md = MultiDomain([p], conn=[])
    size = md.get_size()
    kCp = finalize_symmetric_matrix(
          fkCpd(k, k, k, p, xp, yp, size, p.row_start, p.col_start)
        + fkCld_xcte(0., 0., k, p, xl, size, p.row_start, p.col_start)
        + fkCld_ycte(0., k, 0., p, yl, size, p.row_start, p.col_start))
    c = solve(md.calc_kC() + kCp, md.calc_fext(), silent=True)
    c_ns = MultiDomain([p], conn=[], pd_method='null-space').linear_static()

    ys = np.linspace(0, b, 7)
    xs = np.linspace(0, a, 7)
    for ci, tol in ((c_ns, 1e-10), (c, 1e-2)):
        u, v, w = displacements(p, ci, [xp], [yp])
        assert np.allclose([u[0], v[0], w[0]], [up, vp, wp], rtol=tol)
        w = displacements(p, ci, np.full_like(ys, xl), ys)[2]
        assert np.allclose(w, wl(ys), rtol=0, atol=tol*2e-3)
        v = displacements(p, ci, xs, np.full_like(xs, yl))[1]
        assert np.allclose(v, vl(xs), rtol=0, atol=tol*3e-5)
    assert np.linalg.norm(c - c_ns) < 1e-2*np.linalg.norm(c_ns)


def test_invalid_model():
    r"""The number of DOFs per term is read from :mod:`panels.modelDB`, a
    panel of an unknown model raises a ``ValueError``"""
    p = plate(0.4, 0.3, m=4, n=4)
    size = p.get_size()
    p.model = 'unknown_model'
    for K in (lambda: fkCpd(0., 0., 1., p, 0.1, 0.1, size, 0, 0),
              lambda: fkCld_xcte(0., 0., 1., p, 0.1, size, 0, 0),
              lambda: fkCld_ycte(0., 0., 1., p, 0.1, size, 0, 0)):
        with pytest.raises(ValueError, match='unknown_model'):
            K()
