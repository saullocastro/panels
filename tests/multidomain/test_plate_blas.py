r"""Options for speed: ``plate_kernels='blas'`` and ``reuse_kT=True``

- The operators of :mod:`panels.multidomain.plate_blas`, selected with
  ``MultiDomain(..., plate_kernels='blas')``, give the same ``kC`` and
  ``kG`` (``NLgeom=True``), ``fint`` and ``kTv`` as the Cython kernels of
  the default ``plate_kernels='cython'``, to ``1e-12`` of the largest
  entry, with the same sparsity pattern (the upper triangle by pairs of
  terms with ``finalize=False``), for the CLPT and the FSDT, at a
  non-trivial configuration, with boundary flags other than the defaults
  and an unsymmetric laminate (bending-extension coupling ``B``).
- A domain that is not eligible (constitutive matrix per point, partial
  integration domain, another model, constant membrane stresses) falls
  back to the Cython kernels: exactly the result of ``'cython'``.
- The operators follow a change of the domain (boundary flags).
- An end-to-end DCB coupon with cohesive interfaces, a few increments past
  the onset of damage, with ``plate_kernels='blas'``, ``reuse_kT=True`` and,
  when ``structsolve>=0.13.0`` provides it, a faster linear solver, gives
  the same increments, the same number of Newton-Raphson iterations per
  increment and the same load within ``1e-6`` N as the defaults.

"""
import numpy as np
import pytest

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain import plate_blas
from panels.multidomain.delamination import create_coupon, solve_coupon
from panels.multidomain.connections.czm import BilinearLaw

LAMINAPROP = (120e3, 9e3, 0.3, 5e3, 5e3, 3e3)
MODELS = ['plate_clpt_donnell', 'plate_fsdt_donnell']


def _assembly(model, kernels, **attrs):
    r"""Two unconnected domains, unsymmetric laminate (``B != 0``),
    non-default boundary flags"""
    panels = []
    for x0 in (0., 30.):
        p = Shell(model=model, a=30., b=20., m=7, n=5, x0=x0,
                  stack=[0, 45, 90, -30], plyt=0.2, laminaprop=LAMINAPROP)
        p.x1u = 0.
        p.x2wr = 0.5
        p.y1v = 0.
        p.y2w = 0.7
        for k, v in attrs.items():
            setattr(p, k, v)
        panels.append(p)
    return MultiDomain(panels, conn=[], plate_kernels=kernels)


def _state(size, seed=0):
    rng = np.random.default_rng(seed)
    return 0.05*rng.standard_normal(size), rng.standard_normal(size)


def _all(md, c, v, finalize=False):
    kw = dict(c=c, NLgeom=True, finalize=finalize, silent=True)
    return dict(kC=md.calc_kC(**kw).toarray(), kG=md.calc_kG(**kw).toarray(),
                fint=md.calc_fint(c), kTv=md.calc_kTv(c, v))


def test_unsymmetric_laminate_has_coupling():
    r"""The laminate of these tests has a bending-extension coupling ``B``
    (check of the test itself)"""
    md = _assembly('plate_clpt_donnell', 'cython')
    p = md.panels[0]
    p._rebuild()
    assert np.abs(p.ABD[:3, 3:6]).max() > 1e-3*np.abs(p.ABD[:3, :3]).max()


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('finalize', [False, True])
def test_blas_equals_cython(model, finalize):
    r"""``kC``, ``kG`` (``NLgeom=True``), ``fint`` and ``kTv`` of
    ``plate_kernels='blas'`` against the Cython kernels: within ``1e-12``
    of the largest entry, and with the same pattern of non-zero entries,
    which with ``finalize=False`` is the upper triangle by pairs of terms
    of the Cython kernels"""
    mc = _assembly(model, 'cython')
    mb = _assembly(model, 'blas')
    c, v = _state(mc.get_size())
    rc = _all(mc, c, v, finalize)
    rb = _all(mb, c, v, finalize)
    for k in rc:
        scale = np.abs(rc[k]).max()
        assert scale > 0
        assert np.abs(rb[k] - rc[k]).max() <= 1e-12*scale, k
        assert np.array_equal(rb[k] != 0, rc[k] != 0), k
    assert all(i in mb._plate_ops for i in range(len(mb.panels)))


@pytest.mark.parametrize('model', MODELS)
def test_operators_follow_the_domain(model):
    r"""After a change of a boundary flag the operators are rebuilt: the
    results stay equal to those of the Cython kernels"""
    mc = _assembly(model, 'cython')
    mb = _assembly(model, 'blas')
    c, v = _state(mc.get_size(), seed=1)
    _all(mb, c, v)
    ops = mb._plate_ops[0][1]
    for md in (mc, mb):
        md.panels[0].x1w = 1.
    rc = _all(mc, c, v)
    rb = _all(mb, c, v)
    assert mb._plate_ops[0][1] is not ops
    for k in rc:
        assert np.abs(rb[k] - rc[k]).max() <= 1e-12*np.abs(rc[k]).max(), k


@pytest.mark.parametrize('model, attrs', [
    ('plate_clpt_donnell', dict(x1=5.)),
    ('plate_fsdt_donnell', dict(y2=15.)),
    ('plate_clpt_donnell', dict(Nxx=-10.)),
    ('plate_fsdt_donnell', dict(Nxy_cte=3.)),
    ('plate_clpt_donnell', 'ABDnxny'),
    ('plate_fsdt_gl', {}),
])
def test_fallback_is_cython(model, attrs):
    r"""Domains that are not eligible, :func:`.plate_blas.is_eligible`
    (partial integration domain, constant membrane stresses, constitutive
    matrix per point, another model), use the Cython kernels with
    ``plate_kernels='blas'``: exactly the results of ``'cython'``"""
    if attrs == 'ABDnxny':
        p = _assembly(model, 'cython').panels[0]
        p._rebuild()
        ABD = np.tile(p.ABD, (p.nx, p.ny, 1, 1))
        attrs = dict(ABDnxny=ABD)
    mc = _assembly(model, 'cython', **attrs)
    mb = _assembly(model, 'blas', **attrs)
    assert not plate_blas.is_eligible(mb.panels[0])
    c, v = _state(mc.get_size(), seed=2)
    rc = _all(mc, c, v)
    rb = _all(mb, c, v)
    assert not mb._plate_ops
    for k in rc:
        assert np.array_equal(rb[k], rc[k]), k


def test_plate_kernels_validated():
    r"""An unknown ``plate_kernels`` raises a ``ValueError``; the default
    is ``'cython'``"""
    p = Shell(a=1., b=1., m=4, n=4, stack=[0.], plyt=0.1,
              laminaprop=LAMINAPROP)
    assert MultiDomain([p], conn=[]).plate_kernels == 'cython'
    with pytest.raises(ValueError, match='plate_kernels'):
        MultiDomain([p], conn=[], plate_kernels='fast')


def _linear_solver():
    r"""``'pardiso_lu'`` with oneMKL, else ``'dense_lu'``, when
    ``structsolve`` has them (0.13.0 or newer), else ``None``"""
    try:
        from structsolve.static import LINEAR_SOLVERS
    except ImportError:
        return None
    if 'pardiso_lu' in LINEAR_SOLVERS:
        try:
            from structsolve.pardiso import load_mkl
            load_mkl()
            return 'pardiso_lu'
        except Exception:
            pass
    if 'dense_lu' in LINEAR_SOLVERS:
        return 'dense_lu'
    return None


def _dcb(on, **analysis_kw):
    r"""DCB of AS4/PEEK (Camanho et al. 2003, Table 1, verification only),
    small, up to 1.5 mm, the damage starting at about 0.75 mm; returns the
    results and the Newton-Raphson iterations of each increment, counted as
    the evaluations of the tangent"""
    law = BilinearLaw(0.969, 1.719, 80., 100., 1e5, 2.284)
    md, info = create_coupon('DCB', (122700., 10100., 0.25, 5500., 5500.,
                             3700.), 0.13, [0]*12, 25.4, 30., [law],
                             x_end=60., nd=1, L_coh=10., m_crack=6, m_coh=8,
                             m_far=6, n=4, nr_y_gauss=4,
                             plate_kernels='blas' if on else 'cython')
    calls = []
    calc_kC = md.calc_kC

    def counted(*args, **kwargs):
        calls.append(1)
        return calc_kC(*args, **kwargs)
    md.calc_kC = counted
    ends = []
    res, an = solve_coupon(md, info, 1.5, initialInc=0.25, maxInc=0.25,
                           reuse_kT=on, record=lambda an, c, row:
                           ends.append(len(calls)), **analysis_kw)
    return res, np.diff([0] + ends)


def _compare(on_kw):
    res0, nit0 = _dcb(False)
    res1, nit1 = _dcb(True, **on_kw)
    assert res0['max_d'][-1] > 0.5
    assert np.array_equal(res1['delta'], res0['delta'])
    assert np.array_equal(nit1, nit0)
    assert np.abs(res1['P'] - res0['P']).max() <= 1e-6


def test_coupon_options_on_equal_off():
    r"""DCB with ``plate_kernels='blas'`` and ``reuse_kT=True`` against the
    defaults: same increments, same iterations, loads within 1e-6 N"""
    _compare({})


def test_coupon_linear_solver_equal_default():
    r"""As :func:`test_coupon_options_on_equal_off`, also with the linear
    solver ``'pardiso_lu'`` (or ``'dense_lu'`` without oneMKL) of
    ``structsolve>=0.13.0``, forwarded to ``structsolve.Analysis`` by
    :func:`.solve_coupon`"""
    solver = _linear_solver()
    if solver is None:
        pytest.skip('structsolve without pardiso_lu and dense_lu')
    _compare(dict(linear_solver=solver))
