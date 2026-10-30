r"""Prescribed displacements imposed exactly with the null-space method

With ``MultiDomain(..., pd_method='null-space')`` the prescribed
displacements of :meth:`.Shell.add_point_pd`,
:meth:`.Shell.add_distr_pd_fixed_x` and :meth:`.Shell.add_distr_pd_fixed_y`
are the constraints `\boldsymbol{B}_d \boldsymbol{c} = \bar{\boldsymbol{d}}_0 + \lambda \bar{\boldsymbol{d}}_1`, eliminated
with the connections, `\boldsymbol{c} = \boldsymbol{Z} \boldsymbol{c}_r + \boldsymbol{c}_{p0} + \lambda
\boldsymbol{c}_{p1}`, see ``doc/source/multidomain_null_space.rst``:

- the membrane patch test is exact, for single and multiple domains, with
  the reaction `P = E A u/L`
- the penalty method converges to it as the penalty constants increase
- linear buckling under a prescribed end-shortening is the same as under the
  equivalent load
- the geometrically non-linear analyses of ``structsolve`` converge under
  displacement control, with the exact Jacobian and `\partial \boldsymbol{R}/\partial
  \lambda`
- redundant prescriptions are dropped, inconsistent ones raise errors that
  name the panels, the prescriptions and the flags involved

"""
import sys
sys.path.append('../..')

import io

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from structsolve import Analysis, lb, solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels import json_io
import panels.modelDB as modelDB
from panels.multidomain import MultiDomain
from panels.multidomain.cylinder import create_cylinder
from panels.multidomain.connections import nullspace, fkCpd, fkCld_xcte


E = 70e9
NU = 0.3
LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)


def plate(a, b, h=2e-3, m=6, n=6, model='plate_clpt_donnell', x0=0., y0=0.,
          stack=(0, ), laminaprop=(E, NU), membrane_free=True):
    r"""Plate with the in-plane displacements free at all the edges, ``w``
    restrained at the edges"""
    s = Shell(a=a, b=b, m=m, n=n, x0=x0, y0=y0, model=model,
              stack=list(stack), plyt=h/len(stack), laminaprop=laminaprop)
    if membrane_free:
        for edge in ('x1', 'x2', 'y1', 'y2'):
            setattr(s, edge + 'u', 1.)
            setattr(s, edge + 'v', 1.)
            #NOTE also the rotations of the shear deformation theories
            setattr(s, edge + 'phix', 1.)
            setattr(s, edge + 'phiy', 1.)
    return s


def end_shortening(p, delta, x1=True, x2=True, cte=False):
    r"""``u = 0`` at ``x = 0`` and ``u = -delta`` at ``x = a``"""
    if x1:
        p.add_distr_pd_fixed_x(0., ku=1., funcu=lambda y: 0., cte=True)
    if x2:
        p.add_distr_pd_fixed_x(p.a, ku=1., funcu=lambda y: -delta, cte=cte)


def two_by_two(a=1., b=0.6, xs=0.4, ys=0.25, **kwargs):
    ps = [plate(xs, ys, **kwargs), plate(a - xs, ys, x0=xs, **kwargs),
          plate(xs, b - ys, y0=ys, **kwargs),
          plate(a - xs, b - ys, x0=xs, y0=ys, **kwargs)]
    p00, p10, p01, p11 = ps
    conn = [dict(p1=p00, p2=p10, func='SSxcte', xcte1=xs, xcte2=0),
            dict(p1=p01, p2=p11, func='SSxcte', xcte1=xs, xcte2=0),
            dict(p1=p00, p2=p01, func='SSycte', ycte1=ys, ycte2=0),
            dict(p1=p10, p2=p11, func='SSycte', ycte1=ys, ycte2=0)]
    return ps, conn


def fields(p, c, gridx=7, gridy=5):
    mesh, f = p.uvw(c[p.col_start:p.col_end], gridx=gridx, gridy=gridy)
    return mesh['Xs'] + p.x0, mesh['Ys'] + p.y0, f


@pytest.mark.parametrize('model', ['plate_clpt_donnell', 'plate_fsdt_donnell',
                                   'cylshell_clpt_sanders',
                                   'cylshell_fsdt_sanders'])
def test_rows_are_those_of_fg(model):
    r"""The rows of a prescribed component are the rows of ``fg`` of the
    field module of the model, for random boundary condition flags"""
    rng = np.random.default_rng(3)
    p = Shell(a=0.4, b=0.3, r=0.8 if 'cylshell' in model else None, m=7, n=6,
              model=model, stack=[0, 90], plyt=1e-3, laminaprop=LAMINAPROP)
    for d in 'xy':
        for e in '12':
            for f in nullspace.FIELDS:
                for r in ('', 'r'):
                    setattr(p, d + e + f + r, float(rng.integers(0, 2)))
    p.col_start = 0
    size = p.get_size()
    fg = modelDB.db[model]['field'].fg
    g = np.zeros((5, size))
    for x, y in ((0.1, 0.2), (0., 0.13), (0.4, 0.3), (0.27, 0.)):
        fg(g, x, y, p)
        for comp in range(5):
            B = nullspace.pd_constraint_matrix(p, comp, x, y, size).toarray()
            assert np.allclose(B[0], g[comp], rtol=0, atol=1e-12), (comp, x, y)


@pytest.mark.parametrize('layout', ['single', '2x2'])
@pytest.mark.parametrize('model', ['plate_clpt_donnell', 'plate_fsdt_donnell'])
def test_membrane_patch_test_isotropic(layout, model):
    r"""Uniform end-shortening `\delta`, with the lateral contraction free:
    exact uniform strain, `u = -\delta x/a`, `v = \nu \delta y/a`, and the
    reactions `P = N \delta` and `F_u = -N`, `N = E h b \delta/a`"""
    a, b, h, delta = 1., 0.6, 2e-3, 1e-3
    if layout == 'single':
        ps = [plate(a, b, h, model=model)]
        conn = []
        loaded, fixed = ps, ps
    else:
        ps, conn = two_by_two(a, b, h=h, model=model)
        loaded, fixed = (ps[1], ps[3]), (ps[0], ps[2])
    for p in fixed:
        end_shortening(p, delta, x2=False)
    for p in loaded:
        end_shortening(p, delta, x1=False)
    ps[0].add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    md = MultiDomain(ps, conn, pd_method='null-space')
    c = md.linear_static()
    for p in ps:
        X, Y, f = fields(p, c)
        assert np.allclose(f['u'], -delta*X/a, rtol=0, atol=1e-12*delta)
        assert np.allclose(f['v'], NU*delta*Y/a, rtol=0, atol=1e-12*delta)
        # w is decoupled, but its round-off is the membrane round-off
        # amplified by the stiffness ratio, of order (a/h)**2
        assert np.allclose(f['w'], 0, rtol=0, atol=1e-8*delta)
    N = E*h*b*delta/a
    P = md.reaction_pd(c, NLgeom=False)
    assert np.isclose(P['inc'], N*delta, rtol=1e-10)
    assert P['cte'] == 0.
    F = md.reaction_resultant(c, [(p, 'x', p.a) for p in loaded],
                              NLgeom=False)
    assert np.isclose(F[0], -N, rtol=1e-10)
    assert np.all(np.isnan(F[1:]))
    F = md.reaction_resultant(c, [(p, 'x', 0.) for p in fixed], NLgeom=False)
    assert np.isclose(F[0], N, rtol=1e-10)


def test_membrane_patch_test_laminate():
    r"""The same for a quasi-isotropic laminate: `N = b \delta/(a
    a_{11})`, `\nu_{xy} = -a_{12}/a_{11}`, with `\boldsymbol{a} = \boldsymbol{A}^{-1}`"""
    a, b, delta = 0.5, 0.3, 1e-4
    stack = [0, 45, -45, 90, 90, -45, 45, 0]
    ps, conn = two_by_two(a, b, xs=0.2, ys=0.1, h=1e-3, stack=stack,
                          laminaprop=LAMINAPROP)
    for p in (ps[0], ps[2]):
        end_shortening(p, delta, x2=False)
    for p in (ps[1], ps[3]):
        end_shortening(p, delta, x1=False)
    ps[0].add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    md = MultiDomain(ps, conn, pd_method='null-space')
    c = md.linear_static()
    A = ps[0].lam.ABD[:3, :3]
    aa = np.linalg.inv(A)
    nuxy = -aa[0, 1]/aa[0, 0]
    for p in ps:
        X, Y, f = fields(p, c)
        assert np.allclose(f['u'], -delta*X/a, rtol=0, atol=1e-11*delta)
        assert np.allclose(f['v'], nuxy*delta*Y/a, rtol=0, atol=1e-11*delta)
    N = b*delta/(a*aa[0, 0])
    assert np.isclose(md.reaction_pd(c, NLgeom=False)['inc'], N*delta,
                      rtol=1e-10)
    F = md.reaction_resultant(c, [(ps[1], 'x', ps[1].a),
                                  (ps[3], 'x', ps[3].a)], NLgeom=False)
    assert np.isclose(F[0], -N, rtol=1e-10)


def test_single_shell_without_conn():
    r"""A single panel is wrapped in a :class:`.MultiDomain` without
    connections, ``conn=[]``; ``conn=None`` raises an error, as before"""
    p = plate(1., 0.6)
    end_shortening(p, 1e-3)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    c = md.linear_static()
    assert np.isclose(c @ np.zeros_like(c), 0)
    X, Y, f = fields(p, c)
    assert np.allclose(f['u'], -1e-3*X, atol=1e-15)
    with pytest.raises(RuntimeError, match='No connectivity'):
        MultiDomain([plate(1., 1.)], pd_method='null-space').get_Z()


def penalty_static(p, delta, k):
    r"""Linear static analysis of :func:`end_shortening` with the penalty
    method, the penalty stiffnesses being added to the stiffness matrix"""
    p.clear_disps()
    end_shortening(p, delta)
    p.add_point_pd(0., 0., 0., 0., k, 0., 0., 0.)
    for pd in p.distr_pds + p.distr_pds_inc:
        pd[2] = (lambda f: (lambda y: k*f(y)))(pd[2])
    md = MultiDomain([p], conn=[])
    size = md.get_size()
    kCp = finalize_symmetric_matrix(fkCld_xcte(k, 0., 0., p, 0., size, 0, 0)
        + fkCld_xcte(k, 0., 0., p, p.a, size, 0, 0)
        + fkCpd(0., k, 0., p, 0., 0., size, 0, 0))
    K = md.calc_kC() + kCp
    return solve(K, md.calc_fext(), silent=True)


def test_limit_of_the_penalty_method():
    r"""The penalty method converges to the null-space method, with an error
    decreasing with the inverse of the penalty constant, here for a plate
    whose lateral contraction is restrained at the edges `y = 0, b`, a
    non-uniform membrane state"""
    a, b, delta = 1., 0.6, 1e-3
    p = plate(a, b, m=8, n=8)
    p.y1v = p.y2v = 0.
    end_shortening(p, delta)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    ref = md.linear_static()
    errors = []
    kref = E*2e-3
    for factor in (1e1, 1e2, 1e3, 1e4):
        c = penalty_static(p, delta, factor*kref)
        errors.append(np.linalg.norm(c - ref)/np.linalg.norm(ref))
    for e1, e2 in zip(errors[:-1], errors[1:]):
        assert 8 < e1/e2 < 12, errors


def test_penalty_unchanged_and_excluded_from_fext():
    r"""The forces `k u_p` are in ``calc_fext`` with ``pd_method='penalty'``,
    the default, and not with ``pd_method='null-space'``"""
    p = plate(1., 0.6)
    p.add_point_load(0.5, 0.3, 0., 0., 1.)
    md0 = MultiDomain([p], conn=[])
    f_load = md0.calc_fext()
    p.add_point_pd(0.5, 0.3, 0., 0., 0., 0., 1e6, 1e-3)
    p.add_distr_pd_fixed_x(0.2, kw=1e6, funcw=lambda y: 1e-3*y)
    fext = MultiDomain([p], conn=[]).calc_fext()
    assert MultiDomain([p], conn=[]).pd_method == 'penalty'
    assert not np.allclose(fext, f_load)
    fext_ns = MultiDomain([p], conn=[], pd_method='null-space').calc_fext()
    assert np.array_equal(fext_ns, f_load)
    with pytest.raises(ValueError, match='pd_method'):
        MultiDomain([p], conn=[], pd_method='lagrange')
    c = np.zeros(fext.size)
    with pytest.raises(ValueError, match='requires pd_method'):
        MultiDomain([p], conn=[]).reaction_pd(c)
    with pytest.raises(ValueError, match='requires pd_method'):
        MultiDomain([p], conn=[]).reaction_resultant(c, [(p, 'x', 0.2)])
    p.add_point_pd(0.1, 0.1, None, None, None, None, None, None,
                   kphix=1., phixp=0.)
    with pytest.raises(NotImplementedError, match='rotations'):
        MultiDomain([p], conn=[]).calc_fext()


def test_mixed_constant_and_incremented_linear():
    r"""`\boldsymbol{c}(\lambda) = \boldsymbol{c}_0 + \lambda \boldsymbol{c}_1`, with `\boldsymbol{c}_0`
    due to the constant prescribed displacements and `\boldsymbol{c}_1` due to the
    load and the incremented ones"""
    p = plate(1., 0.6)
    p.add_distr_pd_fixed_x(0., ku=1., funcu=lambda y: 0.)
    p.add_distr_pd_fixed_x(1., ku=1., funcu=lambda y: -1e-3*(1 + y), cte=True)
    p.add_distr_pd_fixed_y(0.6, kv=1., funcv=lambda x: 2e-4*x, cte=False)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    p.add_point_load(0.5, 0.3, 100., 0., 0.)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    c0, c1, c2 = (md.linear_static(lbd) for lbd in (0., 1., 2.))
    assert np.allclose(c2 - c1, c1 - c0, rtol=1e-10, atol=1e-16)
    c_p0, c_p1 = md.get_cp()
    assert np.any(c_p0) and np.any(c_p1)
    X, Y, f = fields(p, c2, gridx=5, gridy=6)
    assert np.allclose(f['u'][:, -1], -1e-3*(1 + Y[:, -1]), atol=1e-14)
    assert np.allclose(f['v'][-1, :], 2*2e-4*X[-1, :], atol=1e-14)
    # the work conjugate reactions are independent of the particular solution
    P = md.reaction_pd(c2, lbd=2., NLgeom=False)
    Z = md.get_Z()
    rng = np.random.default_rng(0)
    for name, cp in (('cte', c_p0), ('inc', c_p1)):
        other = cp + Z @ rng.normal(size=Z.shape[1])
        r = md._reaction_residual(c2, 2., False)
        assert np.isclose(other @ r, P[name], rtol=1e-9)


def test_values_are_read_again_and_T_reused():
    r"""`\boldsymbol{Z}` depends on the positions and components only, the particular
    solutions on the values, which are read again from the panels"""
    p = plate(1., 0.6)
    end_shortening(p, 1e-3)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    c1 = md.linear_static()
    Z = md.get_Z()
    p.clear_disps()
    end_shortening(p, 2e-3)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    assert md.get_Z() is Z
    assert np.allclose(md.linear_static(), 2*c1, rtol=1e-12)
    p.add_point_pd(0.5, 0.3, 0, None, 0, None, 1., 0.)
    assert md.get_Z() is not Z


def test_conflicting_prescriptions_name_panels():
    r"""Inconsistent prescriptions at a corner, on the same panel or across
    a connection, raise errors naming the panels and the prescriptions"""
    p = plate(1., 0.6)
    p.name = 'skin'
    p.add_distr_pd_fixed_x(1., ku=1., funcu=lambda y: -1e-3)
    p.add_distr_pd_fixed_y(0., ku=1., funcu=lambda x: 0.)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    with pytest.raises(ValueError, match=r"u along y = 0 of panel 0 \(name "
                                         r"'skin'.*u along x = 1 of panel 0"):
        md.get_cp()
    # consistent at the corner: the redundant row is dropped
    p.clear_disps()
    p.add_distr_pd_fixed_x(1., ku=1., funcu=lambda y: -1e-3*y)
    p.add_distr_pd_fixed_y(0., ku=1., funcu=lambda x: 0.)
    c_p0, _ = md.get_cp()
    X, Y, f = fields(p, c_p0)
    assert np.allclose(f['u'][:, -1], -1e-3*Y[:, -1], atol=1e-15)
    assert np.allclose(f['u'][0, :], 0., atol=1e-15)
    # across a connection
    ps, conn = two_by_two()
    ps[1].add_distr_pd_fixed_x(ps[1].a, ku=1., funcu=lambda y: -1e-3)
    ps[3].add_distr_pd_fixed_x(ps[3].a, ku=1., funcu=lambda y: -2e-3)
    md = MultiDomain(ps, conn, pd_method='null-space')
    with pytest.raises(ValueError, match="connection 'SSycte' between "
                                         "panel 1.* and panel 3"):
        md.get_cp()
    # each set must be consistent by itself
    ps[3].clear_disps()
    ps[3].add_distr_pd_fixed_x(ps[3].a, ku=1., funcu=lambda y: -1e-3,
                               cte=False)
    with pytest.raises(ValueError, match='each set must be consistent'):
        md.get_cp()


def test_prescription_removed_by_flag():
    r"""A prescription on a term removed by a boundary condition flag: a
    zero value is dropped, a non-zero one raises an error naming the flag"""
    p = plate(1., 0.6, membrane_free=False)
    p.add_distr_pd_fixed_x(1., ku=1., funcu=lambda y: 0.)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    ref = MultiDomain([plate(1., 0.6, membrane_free=False)], conn=[]).get_Z()
    assert md.get_Z().shape == ref.shape
    assert not np.any(md.get_cp()[0])
    p.clear_disps()
    p.add_distr_pd_fixed_x(1., ku=1., funcu=lambda y: 1e-3)
    with pytest.raises(ValueError, match='flags x2u = 0 of the panel; set '
                                         'them to 1'):
        md.get_cp()


@pytest.mark.parametrize('model', ['plate_clpt_donnell', 'plate_fsdt_donnell'])
def test_prescribed_rotation(model):
    r"""A prescribed rotation `\phi_x = \theta` along `x = a`, with `w = 0`
    along `x = 0`, the other edges free: a rigid rotation `w = -\theta x`,
    with null reactions, and a clamped edge by prescription equals the
    clamped edge by flags"""
    theta = 1e-3
    p = plate(1., 0.6, model=model, membrane_free=True)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        setattr(p, edge + 'w', 1.)
    p.add_distr_pd_fixed_x(0., ku=1., kv=1., kw=1., funcu=lambda y: 0.,
                           funcv=lambda y: 0., funcw=lambda y: 0.)
    p.add_distr_pd_fixed_x(1., kphix=1., funcphix=lambda y: theta, cte=False)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    c = md.linear_static()
    X, Y, f = fields(p, c)
    assert np.allclose(f['w'], -theta*X, atol=1e-14)
    assert abs(md.reaction_pd(c, NLgeom=False)['inc']) < 1e-9

    def tip(clamp_by_flags):
        p = plate(1., 0.6, model=model)
        for edge in ('x2', 'y1', 'y2'):
            setattr(p, edge + 'w', 1.)
        p.x1w = 1.
        if clamp_by_flags:
            p.x1w = 0.
            if 'clpt' in model:
                p.x1wr = 0.
            else:
                p.x1phix = 0.
        else:
            p.add_distr_pd_fixed_x(0., kw=1., kphix=1., funcw=lambda y: 0.,
                                   funcphix=lambda y: 0.)
        p.add_point_load(1., 0.3, 0., 0., 10.)
        md = MultiDomain([p], conn=[], pd_method='null-space')
        return fields(p, md.linear_static())[2]['w']
    assert np.allclose(tip(False), tip(True), rtol=1e-9, atol=0)


@pytest.mark.slow
def test_linear_buckling_plate_end_shortening_equals_load():
    r"""Simply supported plate, the lateral edges free in-plane: a uniform
    end-shortening `\delta` and the equivalent load `N_x = E h \delta/a`
    give the same membrane state and the same critical load"""
    a, b, h = 0.3, 0.2, 1e-3
    p = plate(a, b, h, m=10, n=10)
    end_shortening(p, 1.)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    eigvals, eigvecs = md.linear_buckling(num_eigvalues=3)
    assert eigvecs.shape == (p.get_size(), 3)

    q = plate(a, b, h, m=10, n=10)
    q.x1u = 0.
    q.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    q.add_distr_load_fixed_x(a, funcx=lambda y: -E*h/a)
    mdq = MultiDomain([q], conn=[], pd_method='null-space')
    eigvals_N, _ = mdq.linear_buckling(num_eigvalues=3)
    assert np.allclose(eigvals, eigvals_N, rtol=1e-9)
    # a constant end-shortening delta0 at x = a and an incremented one
    # lbd*delta1 at x = 0 give the same uniform state when delta0 + lbd*delta1
    # is the critical one
    dcr = eigvals[0]
    p.clear_disps()
    p.add_distr_pd_fixed_x(0., ku=1., funcu=lambda y: dcr, cte=False)
    p.add_distr_pd_fixed_x(a, ku=1., funcu=lambda y: -0.5*dcr, cte=True)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    eigvals_mixed, _ = md.linear_buckling(num_eigvalues=3)
    assert np.isclose(eigvals_mixed[0], 0.5, rtol=1e-8)
    # the same end-shortening, constant and incremented at the same line,
    # is a conflict: the line has a single prescribed value
    p.clear_disps()
    p.add_distr_pd_fixed_x(a, ku=1., funcu=lambda y: -0.5*dcr, cte=True)
    p.add_distr_pd_fixed_x(a, ku=1., funcu=lambda y: -dcr, cte=False)
    with pytest.raises(ValueError, match='each set must be consistent'):
        md.get_cp()



def test_linear_buckling_constant_stress_resultants_rigid_translation():
    r"""The constant stress resultants ``Nxx`` of a panel are a pre-stress
    of :meth:`.MultiDomain.linear_buckling`, counted once: a constant
    prescribed rigid translation, which produces no stress, does not change
    the critical load factor, and with the pre-stress `\bar{N}` and the
    incremented uniform `N_{xx} = -1` the critical value is `\lambda_{cr} =
    N_{cr} - \bar{N}`; before, ``Nxx`` entered the geometric stiffness of the
    constant state and that of the incremented one, and the translation
    lowered `\lambda_{cr}` by `\bar{N}`"""
    a, b = 0.5, 0.3

    def run(N0, u0):
        p = plate(a, b, h=0.5e-3, m=8, n=8, stack=(0, 90, 90, 0),
                  laminaprop=LAMINAPROP)
        p.Nxx = -N0
        # uniform N_xx = -1, self-equilibrated
        p.add_distr_load_fixed_x(0., funcx=lambda y: 1.)
        p.add_distr_load_fixed_x(a, funcx=lambda y: -1.)
        # in-plane rigid motion, a constant translation u0
        p.add_point_pd(0., 0., 1., u0, 1., 0., None, None, cte=True)
        p.add_point_pd(0., b, 1., u0, None, None, None, None, cte=True)
        md = MultiDomain([p], conn=[], pd_method='null-space')
        eigvals, _ = md.linear_buckling(num_eigvalues=3)
        assert p.Nxx == -N0
        return eigvals[0]

    Ncr = run(0., 0.)
    assert np.isclose(Ncr, 161.93108924810818, rtol=1e-8)
    for u0 in (0., 1e-6, 1e-3):
        assert np.isclose(run(1., u0), Ncr - 1., rtol=1e-10)
        assert np.isclose(run(0.5*Ncr, u0), 0.5*Ncr, rtol=1e-10)

    # Nxx alone is not multiplied by the load factor
    p = plate(a, b, h=0.5e-3, m=8, n=8, stack=(0, 90, 90, 0),
              laminaprop=LAMINAPROP)
    p.Nxx = -1.
    p.add_point_pd(0., 0., 1., 0., 1., 0., None, None)
    p.add_point_pd(0., b, 1., 0., None, None, None, None)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    with pytest.raises(ValueError, match='no geometric stiffness'):
        md.linear_buckling(num_eigvalues=3)


def cylinder(pd_method, **kw):
    md, conns = create_cylinder(height=0.5, r=0.25, plyt=0.125e-3,
            stack=[0, 45, -45, 90, -45, 45], laminaprop=LAMINAPROP,
            npanels=5, m=8, n=12)
    md = MultiDomain(md.panels, conns, pd_method=pd_method)
    for p in md.panels:
        p.x2u = 1
    return md


@pytest.mark.slow
def test_cylinder_end_shortening():
    r"""Cylinder of 5 panels, simply supported, compressed by a uniform
    end-shortening, see ``test_cylinder_compression_lb_Nxx_from_static``

    - the penalty method converges to the null-space method with the error
      decreasing as `1/k`, from above, the penalty springs shortening the
      cylinder less than `\delta`
    - the end-shortening of the axial load `N_{xx}` gives the reaction
      `2 \pi r N_{xx}`, the same pre-buckling state, and a larger critical
      load, the buckling mode having `u = 0` at the loaded edge under
      displacement control
    """
    Nxx = -100.
    md = cylinder('null-space')
    for p in md.panels:
        p.add_distr_load_fixed_x(p.a, lambda y: Nxx, None, None)
    eig_N, _ = md.linear_buckling(num_eigvalues=3)
    assert np.isclose(Nxx*eig_N[0], -47162.406, rtol=2e-3)
    c_N = md.linear_static()
    delta = -fields(md.panels[0], c_N)[2]['u'][0, -1]

    md = cylinder('null-space')
    for p in md.panels:
        p.add_distr_pd_fixed_x(p.a, ku=1., funcu=lambda y: -delta, cte=False)
    eig_d, modes = md.linear_buckling(num_eigvalues=3)
    assert modes.shape[1] == 3
    c = md.linear_static()
    assert np.allclose(c, c_N, rtol=0, atol=1e-6*np.abs(c_N).max())
    F = md.reaction_resultant(c, [(p, 'x', p.a) for p in md.panels],
                              NLgeom=False)
    assert np.isclose(F[0], Nxx*2*np.pi*0.25, rtol=1e-6)
    assert 1.02 < eig_d[0]/eig_N[0] < 1.05

    errors = []
    for factor in (1e3, 1e4):
        md = cylinder('penalty')
        size = md.get_size()
        k = factor*md.panels[0].lam.ABD[0, 0]
        kCp = 0.
        for p in md.panels:
            p.add_distr_pd_fixed_x(p.a, ku=k, funcu=lambda y: -delta,
                                   cte=False)
            kCp = kCp + fkCld_xcte(k, 0., 0., p, p.a, size, p.row_start,
                                   p.col_start)
        K = md.calc_kC(silent=True) + finalize_symmetric_matrix(kCp)
        c = md.expand(solve(md.reduce(K), md.reduce(md.calc_fext()),
                            silent=True))
        eig, _ = lb(md.reduce(K), md.reduce(md.calc_kG(c=c, silent=True)),
                    silent=True, num_eigvalues=3)
        errors.append(eig[0]/eig_d[0] - 1)
    assert 0 < errors[1] < 2e-4
    assert 9 < errors[0]/errors[1] < 11


def plate_postbuckling(delta_factor=3., m=8, n=8, P=2.):
    r"""Simply supported plate under end-shortening, the lateral edges free
    in-plane, with a small lateral load as imperfection"""
    a, b, h = 0.3, 0.2, 1e-3
    p = plate(a, b, h, m=m, n=n)
    end_shortening(p, 1.)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    eigvals, _ = md.linear_buckling(num_eigvalues=1)
    delta = delta_factor*eigvals[0]
    p.clear_disps()
    end_shortening(p, delta)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    p.add_point_load(a/2, b/2, 0., 0., P)
    return md, p, delta


@pytest.mark.parametrize('method', ['NR', 'arc_length_riks',
                                    'arc_length_crisfield'])
@pytest.mark.slow
def test_nonlinear_displacement_control(method):
    r"""Post-buckling under displacement control: every converged increment
    satisfies `\|\boldsymbol{Z}^\top \boldsymbol{r}\| \leq` ``relTOL`` `\|\lambda \partial \boldsymbol{R}_r/\partial
    \lambda\|`, the methods agree, and the reaction is below the linear one"""
    md, p, delta = plate_postbuckling()
    an = Analysis(*md.get_reduced_functions())
    an.NL_method = method
    an.initialInc = 0.2
    #NOTE the arc length is scaled by the linear solution, without the
    #     out-of-plane deflection of the post-buckling path
    an.maxArcLength = 100.
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    fext, fint, kC, kG = md.get_reduced_functions()
    Z = md.get_Z()
    for c_r, lbd in zip(an.cs, an.increments):
        c = md.expand(c_r, lbd)
        r = md._reaction_residual(c, lbd, True)
        q = fext(inc=1., c=c_r, lbd=lbd)
        assert np.linalg.norm(Z.T @ r) <= 1.01e-6*np.linalg.norm(lbd*q)
    P = md.reaction_pd(c, 1.)['inc']/delta
    N_lin = E*p.plyt*p.b*delta/p.a
    assert 0.5 < P/N_lin < 0.7
    assert np.isclose(P, 2420., rtol=2e-3)


@pytest.mark.slow
def test_nonlinear_reaction_matches_penalty():
    r"""The reaction of the null-space method is the limit of the reaction
    `\int k (u_p - u) dy` of the penalty method, with an imperfection large
    enough for a single equilibrium path"""
    md, p, delta = plate_postbuckling(delta_factor=2., m=6, n=6, P=30.)
    an = Analysis(*md.get_reduced_functions())
    an.relTOL = 1e-10
    an.static(NLgeom=True, silent=True)
    c = md.expand(an.cs[-1], 1.)
    F_ns = md.reaction_resultant(c, [(p, 'x', p.a)])[0]

    #NOTE the error of the penalty method decreases with 1/k until the
    #     convergence tolerance, relative to the penalty forces k*u_p,
    #     dominates, see the penalty vs null-space section of the docs
    k = 1e4*E*p.plyt
    mdp = MultiDomain([p], conn=[])
    p.clear_disps()
    p.add_distr_pd_fixed_x(0., ku=k, funcu=lambda y: 0.)
    p.add_distr_pd_fixed_x(p.a, ku=k, funcu=lambda y: -delta)
    p.add_point_pd(0., 0., 0., 0., k, 0., 0., 0.)
    size = mdp.get_size()
    kCp = finalize_symmetric_matrix(fkCld_xcte(k, 0., 0., p, 0., size, 0, 0)
        + fkCld_xcte(k, 0., 0., p, p.a, size, 0, 0)
        + fkCpd(0., k, 0., p, 0., 0., size, 0, 0))
    an = Analysis(lambda inc=1., silent=True: mdp.calc_fext(inc=inc),
                  lambda c, silent=True: mdp.calc_fint(c) + kCp @ c,
                  lambda c=None, NLgeom=False, silent=True:
                      mdp.calc_kC(c=c, NLgeom=NLgeom) + kCp,
                  lambda c=None, NLgeom=False, silent=True:
                      mdp.calc_kG(c=c, NLgeom=NLgeom, silent=True))
    an.relTOL = 1e-10
    an.static(NLgeom=True, silent=True)
    assert np.isclose(an.increments[-1], 1.)
    c_pen = an.cs[-1]
    eta, w = leggauss(40)
    ys = (eta + 1)*p.b/2
    _, f = p.uvw(c_pen, xs=np.full_like(ys, p.a), ys=ys)
    F_pen = k*np.dot(w, -delta - f['u'])*p.b/2
    assert np.isclose(F_pen, F_ns, rtol=1e-3)
    assert not np.isclose(F_pen, F_ns, rtol=1e-5)


@pytest.mark.slow
def test_finite_differences_of_reduced_functions():
    r"""`\boldsymbol{K}_{T,r}` is the Jacobian of ``calc_fint`` and ``calc_fext(c,
    lbd)`` is `-\partial \boldsymbol{R}_r/\partial \lambda`, at a buckled configuration"""
    md, p, delta = plate_postbuckling(delta_factor=2., m=6, n=6)
    p.add_distr_pd_fixed_y(p.b, kv=1., funcv=lambda x: 1e-3*delta*x,
                           cte=True)
    fext, fint, kC, kG = md.get_reduced_functions()
    an = Analysis(fext, fint, kC, kG)
    an.static(NLgeom=True, silent=True)
    c_r = an.cs[-1]
    lbd = 0.8
    K = (kC(c_r, NLgeom=True, inc=lbd) + kG(c_r, NLgeom=True, inc=lbd)
         ).toarray()
    rng = np.random.default_rng(1)
    v = rng.normal(size=c_r.size)*np.abs(c_r).max()
    for h in (1e-4, ):
        dF = (fint(c_r + h*v, inc=lbd) - fint(c_r - h*v, inc=lbd))/(2*h)
        assert np.linalg.norm(dF - K @ v) <= 1e-6*np.linalg.norm(K @ v)

    def R(lbd):
        return fint(c_r, inc=lbd) - lbd*fext(inc=1.)
    h = 1e-5
    dR = (R(lbd + h) - R(lbd - h))/(2*h)
    q = fext(inc=1., c=c_r, lbd=lbd)
    assert np.linalg.norm(dR + q) <= 1e-6*np.linalg.norm(q)


@pytest.mark.slow
def test_constant_end_shortening_with_load_parameter():
    r"""A constant end-shortening, held during the analysis, with a lateral
    load as the load parameter: at `\lambda = 0` the solution is the linear
    one of the end-shortening, and equilibrium holds at every increment"""
    a, b, h = 0.3, 0.2, 1e-3
    p = plate(a, b, h)
    delta = 1e-6
    end_shortening(p, delta, cte=True)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    p.add_point_load(a/2, b/2, 0., 0., 50.)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    c_p0, c_p1 = md.get_cp()
    assert np.any(c_p0) and not np.any(c_p1)
    for method in ('NR', 'arc_length_riks'):
        an = Analysis(*md.get_reduced_functions())
        an.NL_method = method
        an.static(NLgeom=True, silent=True)
        assert np.isclose(an.increments[-1], 1.)
        Z = md.get_Z()
        for c_r, lbd in zip(an.cs, an.increments):
            c = md.expand(c_r, lbd)
            r = md._reaction_residual(c, lbd, True)
            assert np.linalg.norm(Z.T @ r) <= 1e-5*np.linalg.norm(
                lbd*Z.T @ md.calc_fext())
            X, Y, f = fields(p, c)
            assert np.allclose(f['u'][:, -1], -delta, atol=1e-12*delta)
    # without the incremented set, its reaction is null; at lbd = 0 the
    # reaction of the constant set is N delta, N = E h b delta/a
    assert md.reaction_pd(c, 1.)['inc'] == 0.
    c0 = md.linear_static(0.)
    N = E*h*b*delta/a
    assert np.isclose(md.reaction_pd(c0, 0., NLgeom=False)['cte'], N*delta,
                      rtol=1e-9)


def test_linear_static_through_analysis_raises():
    p = plate(1., 0.6)
    end_shortening(p, 1e-3)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    an = Analysis(*md.get_reduced_functions())
    with pytest.raises(ValueError, match='linear_static'):
        an.static(NLgeom=False, silent=True)


def test_without_prescriptions_unchanged():
    r"""Without prescribed displacements, ``pd_method='null-space'`` gives
    the same `\boldsymbol{Z}` and the same reduced functions as the default"""
    ps, conn = two_by_two()
    md1 = MultiDomain(ps, conn)
    md2 = MultiDomain(ps, conn, pd_method='null-space')
    Z1, Z2 = md1.get_Z(), md2.get_Z()
    assert (Z1 != Z2).nnz == 0
    for f1, f2 in zip(md1.get_reduced_functions(),
                      md2.get_reduced_functions()):
        assert f1.__code__.co_varnames == f2.__code__.co_varnames
    c_r = np.random.default_rng(0).normal(size=Z1.shape[1])
    assert np.array_equal(md1.expand(c_r), md2.expand(c_r, lbd=0.5))


def test_save_and_load():
    p = plate(1., 0.6)
    p.add_point_pd(0., 0., 0, None, 1., 0., 0, None)
    p.add_point_pd(1., 0.3, 1., -1e-3, 0, None, 0, None, cte=False)
    md = MultiDomain([p], conn=[], pd_method='null-space')
    md.get_Z(rhs_tol=1e-6)
    f = io.BytesIO()
    md.save(f)
    f.seek(0)
    md2 = json_io.load(f)
    assert md2.pd_method == 'null-space'
    assert md2._rhs_tol == 1e-6
    q = md2.panels[0]
    assert q.point_pds_values == p.point_pds_values
    assert q.point_pds_values_inc == p.point_pds_values_inc
    assert np.allclose(md2.get_cp()[1], md.get_cp()[1])
