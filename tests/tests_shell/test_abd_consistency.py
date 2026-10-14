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

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from structsolve import lb, solve

from panels import modelDB
from panels.shell import Shell
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

