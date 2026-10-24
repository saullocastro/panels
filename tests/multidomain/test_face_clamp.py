r"""Clamped edges of the TSDT: planar clamp by the flags and face clamp by
prescribed displacements at the faces

The in-plane displacement of the third-order shear deformation theory
(TSDT), with the laminate of thickness `h` at `d - h/2 \le z \le d + h/2`,

.. math::

    u(z) = u + z \phi_x - c_1 \left[(z - d)^3 + d^3\right] \left(\phi_x +
    w_{,x}\right), \qquad c_1 = \frac{4}{3 h^2}

is a cubic in `z` that depends on `u`, `\phi_x` and `w_{,x}`. It vanishes
for every `z` only when `u = \phi_x = w_{,x} = 0`: the planar clamp of the
flags ``x1u = x1phix = x1wr = 0`` (with ``x1v = x1w = x1phiy = 0``). The face
clamp prescribes `u = v = 0` at the faces `z = d \mp h/2` and `w = 0`, two
conditions on `u`, `\phi_x` and `w_{,x}`,

.. math::

    \phi_x = \frac{w_{,x}}{2}, \qquad u = -\frac{d}{2} \left(1 - \frac{4
    d^2}{h^2}\right) w_{,x}

which leave the section free to warp, `u(z) = \frac{w_{,x}}{2} \zeta (1 - 4
\zeta^2/h^2)` with `\zeta = z - d`. For the FSDT, `u(z) = u + z \phi_x`, and
for the CLPT, `u(z) = u - z w_{,x}`, the face clamp is the clamp of the
flags.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.connections import nullspace, fkCld_xcte, fkCpd
from panels.multidomain.connections import kCpd

E = 70e9
NU = 0.3
A = 0.5
B = 0.5
H = 0.05
Q = 1e6
FIELDS = ('u', 'v', 'w', 'phix', 'phiy')
U, V, W = nullspace.U, nullspace.V, nullspace.W


def plate(model, offset=0., m=10, n=10):
    r"""Square isotropic plate, `h/a = 0.1`, all the flags ``1``, under a
    uniform pressure"""
    s = Shell(model=model, a=A, b=B, stack=[0.], plyt=H, laminaprop=(E, NU),
              offset=offset, m=m, n=n)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for field in FIELDS:
            setattr(s, edge + field, 1.)
            setattr(s, edge + field + 'r', 1.)
    s.add_pressure_load(Q)
    return s


def clamp_by_flags(s, edge):
    r"""Clamped edge by the boundary condition flags: `u, v, w` and the
    rotations, plus the slope of `w` normal to the edge for the CLPT and the
    TSDT"""
    for field in FIELDS:
        setattr(s, edge + field, 0.)
    if 'fsdt' not in s.model:
        setattr(s, edge + 'wr', 0.)


def face_clamp(s, x, h, k=1., cte=True):
    r"""Face clamp of the edge at ``x``: `u = v = 0` at the faces `z = d \mp
    h/2` and `w = 0`, prescribed with :meth:`.Shell.add_distr_pd_fixed_x`
    and the keyword ``z``"""
    zero = lambda y: 0.
    for z in (s.offset - h/2, s.offset + h/2):
        s.add_distr_pd_fixed_x(x, ku=k, kv=k, funcu=zero, funcv=zero, z=z,
                               cte=cte)
    s.add_distr_pd_fixed_x(x, kw=k, funcw=zero, cte=cte)


def static(s, pd_method='null-space'):
    s._rebuild()
    md = MultiDomain([s], conn=[], pd_method=pd_method)
    return md, md.linear_static()


def fields(s, c, xs, ys):
    r"""`u, v, w, \phi_x, \Phi_y` of :meth:`.Shell.uvw` and, for the shear
    deformation theories, `\gamma_{xz} = \phi_x + w_{,x}` and `\gamma_{yz}`
    of :meth:`.Shell.strain`"""
    xs = np.atleast_1d(np.asarray(xs, dtype=float))
    ys = np.atleast_1d(np.asarray(ys, dtype=float))
    c = np.ascontiguousarray(c)
    s.uvw(c, xs=xs, ys=ys)
    out = {k: s.fields[k].ravel().copy() for k in FIELDS}
    if 'clpt' in s.model:
        out['gxz'] = out['gyz'] = 0*out['u']
    else:
        s.strain(c, xs=xs, ys=ys, NLgeom=False)
        out['gxz'] = s.fields['gxz'].ravel().copy()
        out['gyz'] = s.fields['gyz'].ravel().copy()
    return out


def uv_at(s, f, z, h):
    r"""`u(z)` and `v(z)` from the fields of :func:`fields`"""
    c1 = 4/(3*h**2) if 'tsdt' in s.model else 0.
    d = s.offset
    p = -c1*((z - d)**3 + d**3)
    return (f['u'] + z*f['phix'] + p*f['gxz'],
            f['v'] + z*f['phiy'] + p*f['gyz'])


def center_w(s, c):
    return fields(s, c, A/2, B/2)['w'][0]


@pytest.mark.parametrize('model', ['plate_clpt_donnell', 'plate_fsdt_donnell',
                                   'plate_tsdt_donnell',
                                   'cylshell_clpt_sanders',
                                   'cylshell_tsdt_sanders'])
@pytest.mark.parametrize('offset', [0., 0.3*H, -0.5*H])
def test_rows_are_the_displacements_at_z(model, offset):
    r"""The rows of the prescribed ``'u'`` and ``'v'`` at the coordinate
    ``z`` applied to any Ritz constants give `u(z)` and `v(z)` evaluated
    from the fields, with `\Phi_y = \phi_y + v/r` of the Sanders-Koiter
    kinematics"""
    s = Shell(model=model, a=A, b=B, r=1. if 'cylshell' in model else None,
              stack=[0, 90, 0], plyt=H/3,
              laminaprop=(10*E, E, 0.25, 0.6*E, 0.6*E, 0.5*E),
              offset=offset, m=6, n=7)
    for edge in ('x1', 'x2', 'y1', 'y2'):
        for field in FIELDS:
            setattr(s, edge + field, 1.)
    s._rebuild()
    s.col_start = 0
    size = s.get_size()
    c = np.random.default_rng(3).standard_normal(size)
    for z in (offset - H/2, offset + H/2, 0.37*H):
        for x, y in ((0., 0.21), (0.13, 0.3), (A, B)):
            f = fields(s, c, x, y)
            u, v = uv_at(s, f, z, H)
            Bu = nullspace.pd_constraint_matrix(s, U, x, y, size, z)
            Bv = nullspace.pd_constraint_matrix(s, V, x, y, size, z)
            scale = 1e-12*np.abs(c).sum()
            assert np.isclose((Bu @ c)[0], u[0], rtol=1e-10, atol=scale)
            assert np.isclose((Bv @ c)[0], v[0], rtol=1e-10, atol=scale)


def test_penalty_kernels_with_z():
    r"""The penalty stiffness at ``z`` is `k \{g(z)\}^T \{g(z)\}`, at a
    point, and the integral along the line; the generic route of ``z != 0``
    with ``z = 0`` gives the closed-form integrals of the kernels"""
    s = plate('plate_tsdt_donnell', offset=0.2*H, m=6, n=7)
    s._rebuild()
    s.col_start = 0
    size = s.get_size()
    k = 3.
    z = s.offset + H/2
    K = finalize_symmetric_matrix(fkCpd(k, 2*k, 0., s, 0.1, 0.2, size, 0, 0,
                                        z=z)).toarray()
    Bu = nullspace.pd_constraint_matrix(s, U, 0.1, 0.2, size, z).toarray()
    Bv = nullspace.pd_constraint_matrix(s, V, 0.1, 0.2, size, z).toarray()
    ref = k*Bu.T @ Bu + 2*k*Bv.T @ Bv
    assert np.allclose(K, ref, rtol=0, atol=1e-12*np.abs(ref).max())
    for x in (0., 0.3):
        K0 = fkCld_xcte(k, 2*k, 5*k, s, x, size, 0, 0).toarray()
        K1 = kCpd._fkC_z((k, 2*k, 5*k), s, x, None, 0., size, 0, 0).toarray()
        assert np.allclose(K0, K1, rtol=0, atol=1e-12*np.abs(K0).max())
    K0 = kCpd.fkCld_ycte(k, 2*k, 5*k, s, 0.1, size, 0, 0).toarray()
    K1 = kCpd._fkC_z((k, 2*k, 5*k), s, None, 0.1, 0., size, 0, 0).toarray()
    assert np.allclose(K0, K1, rtol=0, atol=1e-12*np.abs(K0).max())


@pytest.mark.parametrize('model', ['plate_clpt_donnell', 'plate_fsdt_donnell'])
def test_face_clamp_equals_clamp_for_clpt_and_fsdt(model):
    r"""For the CLPT and the FSDT `u(z)` is linear in `z`: `u = v = 0` at
    both faces is the clamp of the flags"""
    s = plate(model)
    for x in (0., A):
        face_clamp(s, x, H)
    _, c = static(s)
    t = plate(model)
    clamp_by_flags(t, 'x1')
    clamp_by_flags(t, 'x2')
    _, ct = static(t)
    xs, ys = np.meshgrid(np.linspace(0, A, 7), np.linspace(0, B, 5))
    f = fields(s, c, xs.ravel(), ys.ravel())
    ft = fields(t, ct, xs.ravel(), ys.ravel())
    for k in FIELDS:
        assert np.allclose(f[k], ft[k], rtol=0, atol=1e-8*np.abs(ft['w']).max())


@pytest.mark.parametrize('offset', [0., 0.3*H])
def test_planar_clamp_is_u_zero_at_three_coordinates(offset):
    r"""For the TSDT `u(z)` depends on three constants, `u`, `\phi_x` and
    `w_{,x}`: `u = v = 0` at three coordinates `z`, here the faces and `z =
    d + h/4`, gives `u(z) = v(z) = 0` for every `z`, the clamp of the flags
    ``x1u = x1v = x1w = x1phix = x1phiy = x1wr = 0``; the mid-surface `z =
    d` would not do, `u(d) = 0` following from the faces. Without ``x1wr =
    0`` the edge is softer"""
    model = 'plate_tsdt_donnell'
    s = plate(model, offset)
    zero = lambda y: 0.
    for x in (0., A):
        face_clamp(s, x, H)
        s.add_distr_pd_fixed_x(x, ku=1., kv=1., funcu=zero, funcv=zero,
                               z=offset + H/4)
    _, c = static(s)
    t = plate(model, offset)
    clamp_by_flags(t, 'x1')
    clamp_by_flags(t, 'x2')
    _, ct = static(t)
    xs, ys = np.meshgrid(np.linspace(0, A, 7), np.linspace(0, B, 5))
    f = fields(s, c, xs.ravel(), ys.ravel())
    ft = fields(t, ct, xs.ravel(), ys.ravel())
    for k in FIELDS:
        assert np.allclose(f[k], ft[k], rtol=0, atol=1e-8*np.abs(ft['w']).max())
    # the former clamp of the shear deformation theories, with w,x free
    t.x1wr = t.x2wr = 1.
    _, cf = static(t)
    assert center_w(t, cf) > 1.05*center_w(t, ct)


def test_face_clamp_tsdt_thick_plate():
    r"""TSDT plate, `h/a = 0.1`, clamped at `x = 0, a` and free at `y = 0,
    b`: the face clamp is softer than the planar clamp and stiffer than the
    simple support; `u` and `v` vanish at both faces along the clamped
    edges, whereas the section warps, `\phi_x = w_{,x}/2`, i.e. `\gamma_{xz}
    = 3 \phi_x`, with `u(z) \neq 0` between the faces"""
    model = 'plate_tsdt_donnell'
    s = plate(model)
    for x in (0., A):
        face_clamp(s, x, H)
    _, c = static(s)
    planar = plate(model)
    clamp_by_flags(planar, 'x1')
    clamp_by_flags(planar, 'x2')
    _, cpl = static(planar)
    ss = plate(model)
    for edge in ('x1', 'x2'):
        for field in ('u', 'v', 'w', 'phiy'):
            setattr(ss, edge + field, 0.)
    _, css = static(ss)
    w_face = center_w(s, c)
    w_planar = center_w(planar, cpl)
    w_ss = center_w(ss, css)
    assert w_planar < 0.99*w_face
    assert w_face < 0.99*w_ss
    # central deflections w D/(q a^4) of the documentation
    D = E*H**3/(12*(1 - NU**2))
    for w, ref in ((w_planar, 0.002901), (w_face, 0.002953), (w_ss, 0.013459)):
        assert np.isclose(w*D/(Q*A**4), ref, rtol=2e-4)
    ys = np.linspace(0, B, 11)
    for x in (0., A):
        f = fields(s, c, np.full_like(ys, x), ys)
        # the face displacement inside the plate
        scale = np.abs(uv_at(s, fields(s, c, np.full_like(ys, A/4), ys),
                             H/2, H)[0]).max()
        for z in (-H/2, H/2):
            u, v = uv_at(s, f, z, H)
            assert np.abs(u).max() < 1e-8*scale
            assert np.abs(v).max() < 1e-8*scale
        assert np.abs(f['w']).max() < 1e-8*w_face
        # warping: phi = w,x/2 and u(z) = (w,x/2) zeta (1 - 4 zeta^2/h^2)
        assert np.allclose(f['gxz'], 3*f['phix'], rtol=1e-6, atol=0)
        u, _ = uv_at(s, f, H/4, H)
        assert np.abs(u).max() > 1e3*np.abs(uv_at(s, f, H/2, H)[0]).max()
        assert np.allclose(u, f['gxz']/3*H/4*(1 - 4*(H/4)**2/H**2),
                           rtol=1e-6, atol=0)


def test_face_clamp_does_not_depend_on_the_reference_surface():
    r"""The faces are physical surfaces: the face-clamped homogeneous plate
    gives the same deflection for the reference surface at the
    mid-surface, `d = 0`, and at `d = 0.3 h` or `d = h/2`, as the planar
    clamp of the flags"""
    model = 'plate_tsdt_donnell'
    xs, ys = np.meshgrid(np.linspace(0, A, 5), np.linspace(0, B, 5))
    res = {}
    for offset in (0., 0.3*H, 0.5*H):
        s = plate(model, offset)
        for x in (0., A):
            face_clamp(s, x, H)
        _, c = static(s)
        t = plate(model, offset)
        clamp_by_flags(t, 'x1')
        clamp_by_flags(t, 'x2')
        _, ct = static(t)
        res[offset] = (fields(s, c, xs.ravel(), ys.ravel())['w'],
                       fields(t, ct, xs.ravel(), ys.ravel())['w'])
    for offset in (0.3*H, 0.5*H):
        for i in (0, 1):
            assert np.allclose(res[offset][i], res[0.][i], rtol=1e-7, atol=0)
    assert res[0.][1].max() < 0.99*res[0.][0].max()


@pytest.mark.parametrize('offset', [0., 0.3*H])
def test_face_clamp_penalty_and_null_space_agree(offset):
    r"""Face clamp at `x = 0` and the faces of `x = a` pushed by `\delta`,
    `u(d \mp h/2) = -\delta`, with the pressure: the penalty method, with the
    kernels of the prescribed displacements at the same `z`, converges to
    the null-space method"""
    model = 'plate_tsdt_donnell'
    delta = 1e-5

    def build(k):
        s = plate(model, offset)
        face_clamp(s, 0., H, k=k)
        for z in (offset - H/2, offset + H/2):
            s.add_distr_pd_fixed_x(A, ku=k, kv=k, funcu=lambda y: -delta,
                                   funcv=lambda y: 0., z=z)
        s.add_distr_pd_fixed_x(A, kw=k, funcw=lambda y: 0.)
        s._rebuild()
        return s

    s = build(1.)
    _, c = static(s)
    k = 1e4*E
    p = build(k)
    md = MultiDomain([p], conn=[])
    size = md.get_size()
    kCp = 0
    for x in (0., A):
        for z in (offset - H/2, offset + H/2):
            kCp = kCp + fkCld_xcte(k, k, 0., p, x, size, 0, 0, z=z)
        kCp = kCp + fkCld_xcte(0., 0., k, p, x, size, 0, 0)
    K = md.calc_kC() + finalize_symmetric_matrix(kCp)
    cp = solve(K, md.calc_fext(), silent=True)
    xs, ys = np.meshgrid(np.linspace(0, A, 6), np.linspace(0, B, 5))
    f = fields(s, c, xs.ravel(), ys.ravel())
    fp = fields(p, cp, xs.ravel(), ys.ravel())
    for key in ('u', 'w'):
        assert np.allclose(fp[key], f[key], rtol=0,
                           atol=1e-3*np.abs(f[key]).max())
    # the faces of x = a are at -delta
    f = fields(s, c, np.full(5, A), np.linspace(0, B, 5))
    for z in (offset - H/2, offset + H/2):
        assert np.allclose(uv_at(s, f, z, H)[0], -delta, rtol=1e-8)


def test_z_is_saved_and_files_without_z_load_with_z_zero():
    r"""The coordinate ``z`` is stored with the prescribed displacements; the
    lists of files written before it get ``z = 0``, the reference
    surface"""
    import io
    from panels import json_io
    s = plate('plate_tsdt_donnell', m=4, n=4)
    s.add_point_pd(0.1, 0.2, 1., 0., None, None, None, None, z=H/2)
    f = io.BytesIO()
    s.save(f)
    f.seek(0)
    q = json_io.load(f)
    assert q.point_pds == s.point_pds
    assert q.point_pds_values == s.point_pds_values
    assert q.point_pds_values[0][7] == H/2
    s.point_pds = [pd[:5] for pd in s.point_pds]
    s.point_pds_values = [values[:7] for values in s.point_pds_values]
    f = io.BytesIO()
    s.save(f)
    f.seek(0)
    q = json_io.load(f)
    assert q.point_pds == [[0.1, 0.2, 0., 0., 0., 0.]]
    assert q.point_pds_values == [[0.1, 0.2, 0., None, None, None, None, 0.]]
