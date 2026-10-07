r"""
3D plots of multi-domain assemblies (:mod:`panels.multidomain.plot3d`)
======================================================================

.. currentmodule:: panels.multidomain.plot3d

Interactive 3D plots with `plotly <https://plotly.com/python/>`_, see
:meth:`.MultiDomain.plot3d`. Each domain is placed in the global coordinate
system with :meth:`.Shell.global_frame` and :meth:`.Shell.global_coords`.

"""
import base64
import json

import numpy as np

from panels.logger import msg
from panels.shell import check_c


DISPLS = ('u', 'v', 'w', 'phix', 'phiy')
STRAINS = ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz',
           'kxx3', 'kyy3', 'kxy3', 'gyz2', 'gxz2')
STRESSES = ('Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qy', 'Qx',
            'Pxx', 'Pyy', 'Pxy', 'Ry', 'Rx')


def _b64(a):
    r"""Array as base64 little-endian float32, decoded by the post script"""
    return base64.b64encode(np.ascontiguousarray(a, dtype='<f4').tobytes()
                            ).decode('ascii')


def _typed_array(a):
    r"""2D array as a typed array of plotly.js, which plotly does not encode
    in the arguments of the buttons of the menus"""
    return dict(dtype='f4', bdata=_b64(a), shape='{0}, {1}'.format(*a.shape))


def _outline(X):
    r"""Coordinates of the edges of a grid of points, with shape ``(ny, nx,
    3)``, as one closed line followed by ``None`` to separate it from the
    next one"""
    line = np.concatenate([X[0, :], X[1:, -1], X[-1, -2::-1], X[-2::-1, 0]])
    return [list(line[:, i]) + [None] for i in range(3)]


# NOTE the post script runs in the page after the figure is created, with
#      {plot_id} replaced by plotly with the id of the figure's div. It adds
#      the controls of the displacements and the keyboard shortcuts, which
#      act while the mouse pointer is over the figure or its controls
_POST_SCRIPT = r"""
(function() {
var gd = document.getElementById('{plot_id}');
if (!gd || gd._panels3d) { return; }
gd._panels3d = true;
var D = __DATA__;
function f32(s) {
    var b = atob(s), u = new Uint8Array(b.length);
    for (var i = 0; i < b.length; i++) { u[i] = b.charCodeAt(i); }
    return new Float32Array(u.buffer);
}
D.traces.forEach(function(t) {
    t.X = f32(t.X);
    t.d = [f32(t.du), f32(t.dv), f32(t.dw)];
});
var comps = ['u', 'v', 'w'];

var ctrl = document.createElement('div');
ctrl.className = 'panels3d-controls';
ctrl.style.cssText = 'font: 13px sans-serif; display: flex; ' +
    'flex-wrap: wrap; gap: 6px 14px; align-items: center; ' +
    'padding: 4px 8px;';
var html = '<span>Displacements:</span>';
comps.forEach(function(c) {
    html += '<label><input type="checkbox" data-comp="' + c + '"> ' + c +
            '</label>';
});
html += '<label>scale <input type="range" data-role="slider" min="0" ' +
        'max="1000" step="1" style="vertical-align: middle; width: 14em">' +
        '</label>' +
        '<input type="number" data-role="scale" step="any" min="0" ' +
        'style="width: 9em">' +
        '<button type="button" data-role="reset">reset</button>' +
        '<span style="opacity: 0.65">keys over the plot: u, v, w toggle; ' +
        '+/- scale; 0 undeformed; r reset; n/p next/previous output; ' +
        'o orthographic/perspective</span>';
ctrl.innerHTML = html;
gd.parentNode.insertBefore(ctrl, gd);
var boxes = {};
comps.forEach(function(c) {
    boxes[c] = ctrl.querySelector('input[data-comp="' + c + '"]');
});
var slider = ctrl.querySelector('[data-role="slider"]');
var input = ctrl.querySelector('[data-role="scale"]');
var sliderMax = 5*D.scale0;

function refreshControls() {
    comps.forEach(function(c) { boxes[c].checked = D.on[c]; });
    if (D.scale > sliderMax) { sliderMax = 2*D.scale; }
    slider.value = sliderMax > 0 ? Math.round(1000*D.scale/sliderMax) : 0;
    if (document.activeElement !== input) {
        input.value = Number(D.scale.toPrecision(4));
    }
}

var pending = false;
function redraw() {
    refreshControls();
    if (pending) { return; }
    pending = true;
    window.requestAnimationFrame(function() {
        pending = false;
        var k = comps.map(function(c) { return D.on[c] ? D.scale : 0; });
        var xs = [], ys = [], zs = [], idx = [];
        D.traces.forEach(function(t) {
            var ny = t.shape[0], nx = t.shape[1], X = [], Y = [], Z = [];
            for (var i = 0; i < ny; i++) {
                var rx = new Array(nx), ry = new Array(nx), rz = new Array(nx);
                for (var j = 0; j < nx; j++) {
                    var p = 3*(i*nx + j);
                    var x = t.X[p], y = t.X[p + 1], z = t.X[p + 2];
                    for (var c = 0; c < 3; c++) {
                        if (k[c] !== 0) {
                            x += k[c]*t.d[c][p];
                            y += k[c]*t.d[c][p + 1];
                            z += k[c]*t.d[c][p + 2];
                        }
                    }
                    rx[j] = x; ry[j] = y; rz[j] = z;
                }
                X.push(rx); Y.push(ry); Z.push(rz);
            }
            xs.push(X); ys.push(Y); zs.push(Z); idx.push(t.i);
        });
        Plotly.restyle(gd, {x: xs, y: ys, z: zs}, idx);
    });
}

function setOutput(step) {
    var menu = gd.layout.updatemenus[0];
    var nb = menu.buttons.length;
    var k = ((((menu.active || 0) + step) % nb) + nb) % nb;
    var args = menu.buttons[k].args;
    var layout = Object.assign({}, args[1], {'updatemenus[0].active': k});
    Plotly.update(gd, args[0], layout, args[2]);
}

comps.forEach(function(c) {
    boxes[c].addEventListener('change', function() {
        D.on[c] = boxes[c].checked;
        redraw();
    });
});
slider.addEventListener('input', function() {
    D.scale = sliderMax*slider.value/1000;
    redraw();
});
input.addEventListener('change', function() {
    var v = parseFloat(input.value);
    if (isFinite(v) && v >= 0) { D.scale = v; }
    redraw();
});
ctrl.querySelector('[data-role="reset"]').addEventListener('click',
    function() {
        D.scale = D.scale0;
        comps.forEach(function(c) { D.on[c] = D.on0[c]; });
        sliderMax = 5*D.scale0;
        redraw();
    });

var hover = false;
[gd, ctrl].forEach(function(el) {
    el.addEventListener('mouseenter', function() { hover = true; });
    el.addEventListener('mouseleave', function() { hover = false; });
});
document.addEventListener('keydown', function(e) {
    if (!hover || e.ctrlKey || e.altKey || e.metaKey) { return; }
    var tag = (e.target && e.target.tagName) || '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') { return; }
    var key = e.key;
    if (key === 'u' || key === 'v' || key === 'w') {
        D.on[key] = !D.on[key];
    } else if (key === '+' || key === '=') {
        D.scale = D.scale > 0 ? 1.5*D.scale : D.scale0;
    } else if (key === '-' || key === '_') {
        D.scale = D.scale/1.5;
    } else if (key === '0') {
        D.scale = 0;
    } else if (key === 'r') {
        D.scale = D.scale0;
        comps.forEach(function(c) { D.on[c] = D.on0[c]; });
        sliderMax = 5*D.scale0;
    } else if (key === 'o') {
        var proj = gd._fullLayout.scene.camera.projection.type;
        Plotly.relayout(gd, {'scene.camera.projection.type':
            proj === 'orthographic' ? 'perspective' : 'orthographic'});
        e.preventDefault();
        e.stopPropagation();
        return;
    } else if (key === 'n' || key === 'p') {
        setOutput(key === 'n' ? 1 : -1);
        e.preventDefault();
        e.stopPropagation();
        return;
    } else {
        return;
    }
    e.preventDefault();
    e.stopPropagation();
    redraw();
}, true);
refreshControls();
})();
"""


def plot3d(md, c, group=None, vec='w', vecs=None, displ='uvw', scale=None,
           gridx=30, gridy=30, colormap='jet', title='', filename='',
           show=False, show_undeformed=True, include_plotlyjs=True,
           silent=True):
    r"""Interactive 3D plot of a multi-domain assembly, see
    :meth:`.MultiDomain.plot3d`"""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError('MultiDomain.plot3d() requires plotly, install it '
                          'with "pip install plotly"')

    msg('Plotting 3D...', silent=silent)
    check_c(c, md.get_size())
    if group is None:
        selected = [(i, p) for i, p in enumerate(md.panels)]
    else:
        groups = [group] if isinstance(group, str) else list(group)
        selected = [(i, p) for i, p in enumerate(md.panels)
                    if p.group in groups]
    if not selected:
        raise ValueError('No panel belongs to group {0!r}'.format(group))
    valid = DISPLS + STRAINS + STRESSES
    if vecs is not None:
        vecs = list(vecs)
        if vec not in vecs:
            vecs.insert(0, vec)
        for v in vecs:
            if v not in valid:
                raise ValueError('{0!r} is not a valid output for plot3d, '
                                 'use one of: {1}'.format(v, ', '.join(valid)))
    elif vec not in valid:
        raise ValueError('{0!r} is not a valid output for plot3d, use one '
                         'of: {1}'.format(vec, ', '.join(valid)))
    for comp in displ:
        if comp not in 'uvw':
            raise ValueError('displ must contain only "u", "v" and "w", got '
                             '{0!r}'.format(displ))
    need_strain = vecs is None or any(v in STRAINS for v in vecs)
    need_stress = vecs is None or any(v in STRESSES for v in vecs)

    # NOTE the fields of each panel, calculated one panel at a time
    results = []
    for i, p in selected:
        res = {k: v[0] for k, v in md.uvw(c, None, gridx=gridx, gridy=gridy,
                                           eval_panel=p).items()}
        if need_strain:
            res.update({k: v[0] for k, v in md.strain(c, None, gridx=gridx,
                        gridy=gridy, eval_panel=p).items()})
        if need_stress:
            res.update({k: v[0] for k, v in md.stress(c, None, gridx=gridx,
                        gridy=gridy, eval_panel=p).items()})
        results.append(res)
    if vecs is None:
        vecs = [v for v in valid if all(v in res for res in results)]
        if vec in vecs:
            vecs.remove(vec)
        vecs.insert(0, vec)
    for v in vecs:
        missing = [p for (i, p), res in zip(selected, results)
                   if v not in res]
        if missing:
            raise ValueError('Output {0!r} is not available for the model '
                             '{1!r}'.format(v, missing[0].model))

    # NOTE global coordinates and displacement vectors of each grid point
    geoms = []
    for (i, p), res in zip(selected, results):
        X, ex, ey, ez = p.global_coords(res['x'], res['y'])
        d = [res['u'][..., None]*ex, res['v'][..., None]*ey,
             res['w'][..., None]*ez]
        geoms.append((X, d))
    allX = np.concatenate([X.reshape(-1, 3) for X, d in geoms])
    size = np.linalg.norm(allX.max(axis=0) - allX.min(axis=0))
    dmax = max(np.linalg.norm(sum(d), axis=-1).max() for X, d in geoms)
    if scale is None:
        scale = 0.1*size/dmax if dmax > 0 and size > 0 else 1.
    on = {comp: comp in displ for comp in 'uvw'}

    vecmin = {v: float(min(res[v].min() for res in results)) for v in vecs}
    vecmax = {v: float(max(res[v].max() for res in results)) for v in vecs}
    labels = ['P{0:02d}'.format(i+1) if p.group is None
              else 'P{0:02d} ({1})'.format(i+1, p.group)
              for i, p in selected]

    # NOTE plotly.js looks up the per-point values of the hover of a surface
    #      with the index [column, row] of the picked point, which gives
    #      transposed values of surfacecolor, hence the hover shows
    #      customdata, the transposed field
    def hovertemplate(label, v):
        return ('{0}<br>{1}: %{{customdata:.4g}}<br>X: %{{x:.4g}}, '
                'Y: %{{y:.4g}}, Z: %{{z:.4g}}<extra></extra>'.format(label, v))

    fig = go.Figure()
    data_traces = []
    for k, ((i, p), res, (X, d)) in enumerate(zip(selected, results, geoms)):
        Xd = X + scale*sum(dc for dc, comp in zip(d, 'uvw') if on[comp])
        fig.add_trace(go.Surface(
            x=Xd[..., 0], y=Xd[..., 1], z=Xd[..., 2],
            surfacecolor=res[vec].astype(np.float32),
            customdata=res[vec].T.astype(np.float32),
            coloraxis='coloraxis', name=labels[k], showlegend=True,
            legendgroup=p.group, hovertemplate=hovertemplate(labels[k], vec),
            lighting=dict(ambient=0.75, diffuse=0.35, specular=0.05,
                          roughness=0.9),
            ))
        data_traces.append(dict(i=k, shape=list(X.shape[:2]), X=_b64(X),
                                du=_b64(d[0]), dv=_b64(d[1]), dw=_b64(d[2])))
    surf_ids = list(range(len(selected)))

    lines = [[], [], []]
    for X, d in geoms:
        for axis, coords in zip(lines, _outline(X)):
            axis.extend(coords)
    fig.add_trace(go.Scatter3d(x=lines[0], y=lines[1], z=lines[2],
                               mode='lines', name='undeformed',
                               line=dict(color='rgba(80, 80, 80, 0.8)',
                                         width=2),
                               hoverinfo='skip',
                               visible=True if show_undeformed
                               else 'legendonly'))

    buttons = []
    for v in vecs:
        buttons.append(dict(label=v, method='update', args=[
            {'surfacecolor': [_typed_array(res[v]) for res in results],
             'customdata': [_typed_array(res[v].T) for res in results],
             'hovertemplate': [hovertemplate(label, v) for label in labels]},
            {'coloraxis.cmin': vecmin[v], 'coloraxis.cmax': vecmax[v],
             'coloraxis.colorbar.title.text': v},
            surf_ids]))
    fig.update_layout(
        title=dict(text=title) if title else None,
        coloraxis=dict(colorscale=colormap, cmin=vecmin[vec],
                       cmax=vecmax[vec], colorbar=dict(title=dict(text=vec))),
        scene=dict(aspectmode='data', xaxis_title='X', yaxis_title='Y',
                   zaxis_title='Z',
                   camera=dict(eye=dict(x=1.6, y=1.6, z=1.3),
                               projection=dict(type='orthographic'))),
        updatemenus=[dict(type='dropdown', buttons=buttons, active=0,
                          x=0., xanchor='left', y=1., yanchor='top',
                          showactive=True)],
        legend=dict(itemsizing='constant', x=0., xanchor='left', y=0.9,
                    yanchor='top', bgcolor='rgba(255, 255, 255, 0.5)'),
        margin=dict(l=0, r=0, t=40 if title else 10, b=0),
        )

    state = dict(scale=float(scale), scale0=float(scale), on=on,
                 on0=dict(on), traces=data_traces)
    post_script = _POST_SCRIPT.replace('__DATA__', json.dumps(state))
    if filename:
        fig.write_html(filename, post_script=post_script,
                       include_plotlyjs=include_plotlyjs, full_html=True)
    if show:
        fig.show(post_script=post_script)
    msg('finished!', silent=silent)

    return fig, dict(vecs=vecs, vecmin=vecmin, vecmax=vecmax,
                     scale=float(scale), post_script=post_script)
