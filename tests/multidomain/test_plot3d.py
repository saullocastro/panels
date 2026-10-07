import base64
import json

import numpy as np
import pytest

from panels.shell import Shell
from panels.multidomain import tstiff2d_1stiff_compression
from panels.multidomain.cylinder import cylinder_compression_lb_Nxx_cte


laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)


def global_displ(p, c, x, y):
    r"""Global positions and displacement vectors of points of panel ``p``"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)*np.ones_like(x)
    p.uvw(c[p.col_start:p.col_end], xs=x, ys=y)
    X, ex, ey, ez = p.global_coords(x, y)
    d = (p.fields['u'][:, None]*ex + p.fields['v'][:, None]*ey
         + p.fields['w'][:, None]*ez)
    return X, d


def place_cylinder(md, r):
    r"""Place the panels of :func:`.create_cylinder` around the global `X`
    axis, with `w` outwards"""
    for p in md.panels:
        phi = p.y0/r
        origin = np.array([0., r*np.cos(phi), -r*np.sin(phi)])
        tangent = np.array([0., -np.sin(phi), -np.cos(phi)])
        p.x0, p.y0, p.z0 = origin
        p.point_x = origin + [1., 0., 0.]
        p.point_xy = origin + tangent


def test_global_frame():
    p = Shell(a=2., b=1., stack=[0, 90], plyt=1e-3, laminaprop=laminaprop)
    origin, axes = p.global_frame()
    assert np.allclose(origin, 0.)
    assert np.allclose(axes, np.eye(3))

    p.x0, p.y0, p.z0 = 1., 2., 3.
    p.point_x = (1., 2., 5.)
    # not orthogonal to x, only on the xy plane
    p.point_xy = (4., 2., 4.)
    origin, (ex, ey, ez) = p.global_frame()
    assert np.allclose(origin, [1., 2., 3.])
    assert np.allclose(ex, [0., 0., 1.])
    assert np.allclose(ey, [1., 0., 0.])
    assert np.allclose(ez, [0., 1., 0.])
    X, ex_, ey_, ez_ = p.global_coords([0., 2., 2.], [0., 0., 1.])
    assert np.allclose(X, [[1., 2., 3.], [1., 2., 5.], [2., 2., 5.]])

    p.point_xy = (1., 2., 7.)
    with pytest.raises(ValueError):
        p.global_frame()
    p.point_x = (1., 2., 3.)
    with pytest.raises(ValueError):
        p.global_frame()


def test_global_coords_cylinder():
    r = 2.
    p = Shell(a=3., b=np.pi*r/2, r=r, stack=[0, 90], plyt=1e-3,
              laminaprop=laminaprop)
    assert p.is_curved()
    y = np.linspace(0, p.b, 7)
    X, ex, ey, ez = p.global_coords(np.ones_like(y), y)
    centre = np.array([1., 0., -r])
    assert np.allclose(np.linalg.norm(X - centre, axis=1), r)
    # w is along the outward normal
    assert np.allclose(ez, (X - centre)/r)
    assert np.allclose(np.cross(ex, ey), ez)
    assert np.allclose(X[-1], [1., r, -r])

    p = Shell(a=3., b=1., r=r, stack=[0, 90], plyt=1e-3,
              laminaprop=laminaprop, model='plate_clpt_donnell')
    assert not p.is_curved()


def test_plot3d_cylinder(tmp_path):
    pytest.importorskip('plotly')
    r = 0.2
    npanels = 4
    md, eigvals, eigvecs = cylinder_compression_lb_Nxx_cte(
        height=0.4, r=r, stack=[0, 45, -45, 90, -45, 45, 0],
        plyt=0.125e-3, laminaprop=laminaprop, npanels=npanels,
        Nxxs=[-100.]*npanels, m=8, n=8, num_eigvalues=5)
    c = eigvecs[:, 0]
    place_cylinder(md, r)

    # the panels close the cylinder and the connections 'SSycte' give the
    # same global displacement vectors on both sides of each joint
    x = np.linspace(0, 0.4, 11)
    dmax = 0.
    for i, p1 in enumerate(md.panels):
        p2 = md.panels[(i + 1) % npanels]
        X1, d1 = global_displ(p1, c, x, p1.b)
        X2, d2 = global_displ(p2, c, x, 0.)
        assert np.allclose(X1, X2, atol=1e-12)
        dmax = max(dmax, np.abs(d1).max())
        assert np.allclose(d1, d2, atol=1e-6*dmax)
    assert dmax > 0

    filename = str(tmp_path / 'cylinder.html')
    fig, data = md.plot3d(c, group='skin', vecs=['w', 'u', 'Nxx'],
                          filename=filename)
    assert data['vecs'] == ['w', 'u', 'Nxx']
    assert len(fig.data) == npanels + 1
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == data['vecs']
    assert fig.layout.scene.aspectmode == 'data'
    # the hover shows the transposed field, see plot3d()
    assert np.allclose(fig.data[0].customdata, fig.data[0].surfacecolor.T)
    assert fig.data[0].surfacecolor.shape == fig.data[0].x.shape
    with open(filename, encoding='utf-8') as f:
        html = f.read()
    assert 'panels3d-controls' in html

    # the deformed geometry of the figure uses the scale factor
    state = json.loads(data['post_script'].split('var D = ', 1)[1]
                       .split(';\n', 1)[0])
    t = state['traces'][0]
    ny, nx = t['shape']
    X = np.frombuffer(base64.b64decode(t['X']), '<f4').reshape(ny, nx, 3)
    d = sum(np.frombuffer(base64.b64decode(t[k]), '<f4').reshape(ny, nx, 3)
            for k in ('du', 'dv', 'dw'))
    Xd = X + data['scale']*d
    assert np.allclose(fig.data[0].x, Xd[..., 0], atol=1e-5)
    assert np.allclose(fig.data[0].z, Xd[..., 2], atol=1e-5)
    # the largest displacement is drawn with 10 % of the size by default
    size = np.linalg.norm([0.4, 2*r, 2*r])
    assert np.isclose(data['scale']*dmax, 0.1*size, rtol=0.2)


def test_plot3d_tstiff2d(tmp_path):
    pytest.importorskip('plotly')
    b = 1.
    bb = b/5.
    bf = bb/2.
    ys = b/2.
    md, c, eigvals, eigvecs = tstiff2d_1stiff_compression(
        b=b, bb=bb, bf=bf, a=3., ys=ys, defect_a=0.4, rho=1.3e3,
        plyt=0.125e-3, laminaprop=laminaprop,
        stack_skin=[0, 45, -45, 90, -45, 45, 0], stack_base=[0, 90, 0]*4,
        stack_flange=[0, 90, 0]*8, Nxx_skin=-60., Nxx_base=-5.,
        Nxx_flange=-10., run_static_case=True, m=6, n=6, num_eigvalues=3)
    skin = [p for p in md.panels if p.group == 'skin']
    bases = [p for p in md.panels if p.group == 'base']
    flanges = [p for p in md.panels if p.group == 'flange']
    h_skin = sum(skin[0].plyts)
    h_base = sum(bases[0].plyts)
    dsb = (h_skin + h_base)/2.
    # the connection 'SB' places the base at z = -dsb, and 'BFycte' rotates
    # the axes of the flange such that its y axis is the -z axis of the base
    for p in bases:
        p.z0 = -dsb
    for base, flange in zip(bases, flanges):
        assert np.isclose(base.x0, flange.x0)
        flange.x0, flange.y0, flange.z0 = base.x0, ys, -dsb
        flange.point_x = (base.x0 + 1., ys, -dsb)
        flange.point_xy = (base.x0, ys, -dsb - 1.)

    for vec in (c, eigvecs[:, 0]):
        for base, flange in zip(bases, flanges):
            x = np.linspace(0, base.a, 9)
            X1, d1 = global_displ(base, vec, x, base.b/2.)
            X2, d2 = global_displ(flange, vec, x, 0.)
            assert np.allclose(X1, X2, atol=1e-12)
            dmax = max(np.abs(d1).max(), np.abs(d2).max())
            assert dmax > 0
            assert np.allclose(d1, d2, atol=1e-6*dmax)

    filename = str(tmp_path / 'tstiff.html')
    fig, data = md.plot3d(eigvecs[:, 0], vec='Nxx', displ='w',
                          filename=filename, include_plotlyjs='cdn')
    assert data['vecs'][0] == 'Nxx'
    for v in ('u', 'v', 'w', 'exx', 'Mxy'):
        assert v in data['vecs']
    assert len(fig.data) == len(md.panels) + 1
    for i, v in enumerate(data['vecs']):
        assert data['vecmin'][v] <= data['vecmax'][v]
    # only w displaces the geometry initially
    state = json.loads(data['post_script'].split('var D = ', 1)[1]
                       .split(';\n', 1)[0])
    assert state['on'] == dict(u=False, v=False, w=True)

    fig, data = md.plot3d(c, group=['base', 'flange'], vecs=['u'], displ='')
    assert len(fig.data) == len(bases) + len(flanges) + 1

    with pytest.raises(ValueError):
        md.plot3d(c, group='stringer')
    with pytest.raises(ValueError):
        md.plot3d(c, vec='Fxx')
    with pytest.raises(ValueError):
        md.plot3d(c, displ='uz')
