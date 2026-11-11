r"""Runs of the notebooks ``moore1944_ring_cylinders`` (``stage=1``) and
``moore1944_stiffened_cylinders`` (``stage=2``)

Usage, from the folder ``notebooks``::

    python moore1944_runs.py 1 8     # stage, number of processes
    python moore1944_runs.py 1 8 "u free"   # only the cases of one name

Writes ``results/moore1944_stage<stage>.json`` (or, for the cases of one
name, ``results/moore1944_stage<stage>_<name>.json``, whose entries replace
those of the same case and specimen in the first file with
:func:`merge`): one entry per case, with the values of
:func:`moore1944_utils.run_case`, written after each case.

"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import moore1944_utils as mu

RING_SPECS = ['16', '17', '18', '13', '13A']
STIFF_SPECS = ['19', '14', '20', '15', '21']

#: stage 1: sectors, terms along the circumference, terms per inch along
#: the axis, terms across the ring webs
MESHES_1 = {'A': dict(nsec=8, n=12, m_per_in=0.8, m_ring=6),
            'B': dict(nsec=12, n=14, m_per_in=1.0, m_ring=6),
            'C': dict(nsec=12, n=18, m_per_in=1.3, m_ring=8)}
#: stage 2: terms per inch across the bays between the stringers and along
#: the axis; terms across strips, webs and lips
MESHES_2 = {'A': dict(n_per_in=1.5, m_per_in=0.8, n_strip=4, n_web=6,
                      n_lip=4),
            'B': dict(n_per_in=2.0, m_per_in=1.0, n_strip=4, n_web=6,
                      n_lip=4),
            'C': dict(n_per_in=2.5, m_per_in=1.2, n_strip=5, n_web=8,
                      n_lip=5)}
#: sensitivities, on mesh B
SENS = {'clamped': dict(ends='clamped'),
        'u free': dict(u_end=1),
        'rigid rings': dict(rings='rigid'),
        'Donnell skin': dict(skin_model='cylshell_clpt_donnell')}
SENS_2 = dict(SENS, **{'no stringers': dict(stringers=False, nsec=16,
                                            n=14),
                       'lips 0.31 in': dict(lip_width=0.31 + 0.016)})


def cases(stage):
    out = []
    if stage == 1:
        for mesh, kw in MESHES_1.items():
            for s in RING_SPECS:
                out.append(('mesh ' + mesh, s, kw, mesh == 'C'))
        for name, kw in SENS.items():
            for s in RING_SPECS:
                out.append((name, s, dict(MESHES_1['B'], **kw), False))
        out.append(('end rotation', '18',
                    dict(MESHES_1['B'], end_rotation=1e-3), False))
    else:
        for mesh, kw in MESHES_2.items():
            for s in STIFF_SPECS:
                out.append(('mesh ' + mesh, s, kw, mesh == 'C'))
        for name, kw in SENS_2.items():
            for s in STIFF_SPECS:
                out.append((name, s, dict(MESHES_2['A'], **kw), False))
    return out


def _run(case):
    name, spec, kw, keep = case
    try:
        res = mu.run_case(spec, kw, keep_mode=keep)
    except Exception as e:      # recorded in the results file
        res = dict(spec=spec, kwargs=dict(kw), error=repr(e))
    res['case'] = name
    return res


def merge(stage, name):
    """Replace the entries of the cases ``name`` of the results file of
    ``stage`` by those of its partial file"""
    fname = os.path.join('results', 'moore1944_stage%d.json' % stage)
    part = fname[:-5] + '_%s.json' % name.replace(' ', '_')
    with open(fname) as fh:
        res = json.load(fh)
    with open(part) as fh:
        new = json.load(fh)
    keys = {(r['case'], r['spec']) for r in new}
    res = [r for r in res if (r['case'], r['spec']) not in keys] + new
    with open(fname, 'w') as fh:
        json.dump(res, fh)


if __name__ == '__main__':
    stage, nproc = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs('results', exist_ok=True)
    fname = os.path.join('results', 'moore1944_stage%d.json' % stage)
    # the largest cases first
    todo = cases(stage)[::-1]
    if len(sys.argv) > 3:
        todo = [c for c in todo if c[0] == sys.argv[3]]
        fname = fname[:-5] + '_%s.json' % sys.argv[3].replace(' ', '_')
    done = []
    with ProcessPoolExecutor(nproc) as ex:
        futs = [ex.submit(_run, c) for c in todo]
        for f in as_completed(futs):
            r = f.result()
            done.append(r)
            print(r['case'], r['spec'], r.get('tau'), r.get('error', ''),
                  flush=True)
            with open(fname, 'w') as fh:
                json.dump(done, fh)
