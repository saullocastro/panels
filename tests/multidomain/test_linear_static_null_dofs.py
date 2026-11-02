r"""Null rows and columns of the stiffness matrix in a linear static analysis

A boundary condition flag set to zero, here ``x1u = x1v = x1w = x1wr = 0``
at the clamped root of a cantilever, multiplies Ritz functions by zero: their
Ritz constants have no stiffness and the stiffness matrix of
:meth:`.MultiDomain.calc_kC` has null rows and columns, so that it is
exactly singular. ``structsolve.solve``, used by ``structsolve.static``,
``structsolve.Analysis.static`` and :meth:`.MultiDomain.linear_static`,
removes them before solving, whereas ``scipy.sparse.linalg.spsolve`` of the
full system fails, see :class:`.MultiDomain`.

"""
import sys
sys.path.append('../..')

import warnings

import numpy as np
import pytest
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from structsolve import solve

from panels.shell import Shell
from panels.multidomain import MultiDomain


E, NU = 73.e9, 0.33
A, B, T = 0.5, 0.1, 2.e-3
P = 10.


def cantilever(conn_method):
    r"""Plate strip of length ``2*A`` in two domains connected by
    ``'SSxcte'``, clamped at `x = 0`, with a transverse force ``P`` at the
    middle of the free end"""
    panels = []
    for i in range(2):
        p = Shell(a=A, b=B, r=None, stack=[0.], plyt=T, laminaprop=(E, NU),
                  model='plate_clpt_donnell', m=8, n=4, x0=i*A, y0=0.)
        for edge in ('x1', 'x2', 'y1', 'y2'):
            for dof in ('u', 'v', 'w', 'wr'):
                setattr(p, edge + dof, 1.)
        panels.append(p)
    p1, p2 = panels
    p1.x1u = p1.x1v = p1.x1w = p1.x1wr = 0.
    p2.add_point_load(A, B/2, 0., 0., P)
    conn = [dict(p1=p1, p2=p2, func='SSxcte', xcte1=A, xcte2=0.)]
    return MultiDomain(panels, conn, conn_method=conn_method)


def tip_w(md, c):
    p2 = md.panels[1]
    return md.uvw(c, None, gridx=3, gridy=3, eval_panel=p2)['w'][0][1, -1]


@pytest.mark.parametrize('conn_method', ['penalty', 'null-space'])
def test_clamped_multidomain_null_dofs(conn_method):
    md = cantilever(conn_method)
    kC = md.reduce(md.calc_kC(silent=True)).tocsc()
    fext = md.reduce(md.calc_fext(silent=True))

    # the clamped root leaves null rows and columns in kC, with null loads
    null = np.flatnonzero(abs(kC).sum(axis=0).A1 == 0)
    assert null.size > 0
    assert np.all(abs(kC).sum(axis=1).A1[null] == 0)
    assert np.all(fext[null] == 0)

    # a plain spsolve of the full system fails: kC is exactly singular
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        c_sp = spsolve(kC, fext)
    assert (any(issubclass(wi.category, MatrixRankWarning) for wi in w)
            or not np.all(np.isfinite(c_sp)))

    # structsolve.solve removes the null rows and columns
    c_r = solve(kC, fext, silent=True)
    assert np.all(np.isfinite(c_r))
    assert np.all(c_r[null] == 0)
    # 2.6e-8 with the penalty connections, whose stiffnesses worsen the
    # conditioning
    assert np.linalg.norm(kC @ c_r - fext) < 1e-6*np.linalg.norm(fext)
    c = md.expand(c_r)

    # the panels path gives the same configuration
    assert np.allclose(md.linear_static(silent=True), c,
                       rtol=1e-6, atol=1e-9*abs(c).max())

    # tip deflection of the cantilever, against the Euler-Bernoulli beam,
    # the strip being narrow, b/L = 0.1: 1.5% smaller, with the
    # anticlastic curvature restrained near the root
    w_beam = P*(2*A)**3/(3*E*B*T**3/12)
    w = tip_w(md, c)
    assert abs(w/w_beam - 1) < 0.03, w/w_beam
