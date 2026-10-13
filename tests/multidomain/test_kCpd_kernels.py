r"""Penalty kernels of the prescribed displacements

The kernels of :mod:`panels.multidomain.connections.kCpd` are compared with
the penalty stiffness `k \int \{g\} \{g\}^T ds` integrated from the rows of
``fg`` of the field module, see ``theory/multidomain_penalization/kCpd.py``.

"""
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
import panels.modelDB as modelDB
from panels.multidomain.connections import fkCpd, fkCld_xcte, fkCld_ycte


def plate(a, b, m, n):
    return Shell(a=a, b=b, m=m, n=n, model='plate_clpt_donnell', stack=[0],
                 plyt=1e-3, laminaprop=(70e9, 0.3))


def line_penalty(p, comp, x, y, k, size):
    r"""Penalty stiffness `k \int g_i g_j ds` of the component ``comp`` along
    the line ``x`` or ``y``, integrated from the rows of ``fg``"""
    fg = modelDB.db[p.model]['field'].fg
    g = np.zeros((5, p.get_size()))
    eta, w = leggauss(max(2*p.m, 2*p.n, 20))
    K = np.zeros((size, size))
    sl = slice(p.col_start, p.col_end)
    for e, wi in zip(eta, w):
        if y is None:
            fg(g, x, (e + 1)*p.b/2, p)
            jac = p.b/2
        else:
            fg(g, (e + 1)*p.a/2, y, p)
            jac = p.a/2
        K[sl, sl] += k*wi*jac*np.outer(g[comp], g[comp])
    return K


@pytest.mark.parametrize('line', ['x', 'y'])
@pytest.mark.parametrize('comp', [0, 1, 2])
def test_line_penalty_kernels(line, comp):
    r"""The kernels of the line prescribed displacements, :func:`.fkCld_xcte`
    and :func:`.fkCld_ycte`, are `k \int g_i g_j ds` of the component; the
    stiffnesses of ``ku`` and ``kv`` were placed at the rows and columns of
    ``w`` up to version 0.11.1"""
    rng = np.random.default_rng(comp)
    p = plate(0.4, 0.3, m=7, n=6)
    for d in 'xy':
        for e in '12':
            for f in 'uvw':
                for r in ('', 'r'):
                    setattr(p, d + e + f + r, float(rng.integers(0, 2)))
    p.col_start = 3*7*6
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


@pytest.mark.parametrize('comp', [0, 1, 2])
def test_point_penalty_kernel(comp):
    r"""The kernel of the point prescribed displacements, :func:`.fkCpd`, is
    `k \{g\} \{g\}^T` of the component"""
    p = plate(0.4, 0.3, m=7, n=6)
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
