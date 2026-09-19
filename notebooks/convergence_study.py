r"""Convergence study of the cohesive zone ('SB_TSL') on the reference DCB

Reproduces the convergence study, the elastic-range timing and the check of
``dcb_utils`` against the driver, documented in
theory/multidomain_penalization/cohesive_zone_deviations_from_thesis.tex,
Section "Convergence study".

The reference DCB is the one of ``tests/multidomain/test_dcb_damage.py``:
length 65 mm, width 25 mm, precrack 48 mm, arms of 15 plies of 0.14 mm of
AS4D/PEKK-FC, G1c = 1.12 N/mm, a cohesive domain of 17 mm, prescribed
displacement up to 8 mm. Two sets of cohesive parameters are used:

- case A: k_o = 5e4 N/mm^3, tau_o = 87 MPa;
- case B: k_o = 2e5 N/mm^3, tau_o = 74.2 MPa.

Usage, from the root of the repository::

    python notebooks/convergence_study.py run [--jobs 4] [--only A_m15_nx120 ...]
    python notebooks/convergence_study.py collect [--json notebooks/results/convergence_study.json]
    python notebooks/convergence_study.py timing
    python notebooks/convergence_study.py compare-dcb-utils

``run`` executes the driver once per case (32 cases, about 10 min to 3 h
each, depending on the case and on the machine), in parallel, writing the log
of case TAG to ``<out>/TAG.log`` and the driver files to ``<out>/TAG/``.
``collect`` reads the logs and prints the metrics of the study. The metrics of
the runs of the theory document are stored in
``notebooks/results/convergence_study.json``.

``timing`` runs the eight elastic increments of the driver with the matrix
products and with the Cython kernels, without the predictor and the reuse of
the panel tangent. The 341 s of the theory document were measured with the
code before these changes (commit 0b619a1), and this comparison isolates the
part of the gain that the current code can still switch off.

``compare-dcb-utils`` solves the reference DCB, case A, m_tsl = 15,
n_x = 120, with ``dcb_utils.solve_dcb`` and compares it with the run
A_m15_nx120 of the driver, which must be available in ``<out>``.

"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# (k_o, tau_o) of the two sets of cohesive parameters
PARAMS = {'A': (5e4, 87.), 'B': (2e5, 74.2)}
# reference discretisation: terms m_tsl, n_tsl of the cohesive domain, terms
# m, n of the other domains, Gauss points nx, ny, number of increments
REF = dict(m_tsl=15, n_tsl=10, m=8, n=8, nx=120, ny=30, npts=40)


def cases():
    """Returns {tag: parameters} of the 32 runs of the study"""
    out = {}
    for c in PARAMS:
        for m_tsl in (10, 15, 20, 25):
            for nx in (60, 120, 180):
                out[f'{c}_m{m_tsl}_nx{nx}'] = dict(REF, case=c, m_tsl=m_tsl, nx=nx)
        out[f'{c}_m15_nx120_n15'] = dict(REF, case=c, n_tsl=15)
        out[f'{c}_m15_nx120_ny60'] = dict(REF, case=c, ny=60)
        out[f'{c}_m15_nx120_mn12'] = dict(REF, case=c, m=12, n=12)
        # halved increment after w_p = 5 mm
        out[f'{c}_m15_nx120_w80'] = dict(REF, case=c, npts=80)
    return out


def _driver():
    import matplotlib
    matplotlib.use('Agg')
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, 'tests', 'multidomain'))
    from test_dcb_damage import dcb_damage_prop_no_f_kcrack
    return dcb_damage_prop_no_f_kcrack


def run_driver(tag, p, w_max=8., **kwargs):
    """Runs the driver for one case in the current directory"""
    k_i, tau_o = PARAMS[p['case']]
    driver = _driver()
    return driver(phy_dim=[3, 65, 25, 48],
                  nr_terms=[p['m_tsl'], p['n_tsl'], p['m'], p['n']],
                  name=[tag, ''], k_i=k_i, tau_o=tau_o, nr_x_gauss=p['nx'],
                  nr_y_gauss=p['ny'], w_iter_info=[p['npts'], w_max],
                  G1c=1.12, **kwargs)


def worker(tag, out):
    p = cases()[tag]
    path = os.path.join(out, tag)
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    t0 = time.time()
    run_driver(tag, p)
    print(f'WALLTIME_S {time.time() - t0:.1f}')


def run(tags, out, jobs):
    """Runs the cases in parallel, one process per case"""
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    threads = str(max(1, (os.cpu_count() or 1)//jobs))
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env.setdefault(k, threads)
    pending = list(tags)
    running = []
    while pending or running:
        while pending and len(running) < jobs:
            tag = pending.pop(0)
            log = open(os.path.join(out, tag + '.log'), 'w')
            proc = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__),
                                     '_worker', tag, '--out', out],
                                    stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((tag, proc, log))
            print('started', tag)
        time.sleep(5)
        for item in running[:]:
            tag, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                print('finished', tag, 'exit code', proc.returncode)


def parse(tag, out):
    """Reads the log of one run of the driver"""
    log = open(os.path.join(out, tag + '.log'), errors='ignore').read()
    wp = [float(v) for v in re.findall(rf'% - {tag} -- wp=([0-9.]+)', log)]
    P = [float(v) for v in re.findall(r'Force - Reaction: ([-0-9.]+)', log)]
    n = min(len(wp), len(P))
    iters = [len(re.findall('crisfield', blk))
             for blk in log.split('------------ wp =')[1:]]
    # the driver prints WAKE UP also after an aborted run
    if 'ABORTED' in log:
        status = 'aborted'
    elif 'WAKE UP' in log:
        status = 'done'
    elif 'Traceback' in log:
        status = 'error'
    else:
        status = 'incomplete'
    m = re.search(r'WALLTIME_S ([0-9.]+)', log)
    wall = float(m.group(1)) if m else None
    if wall is None:
        # runs made without this script, the driver writes the time in hours
        txt = os.path.join(out, tag, tag + '.txt')
        if os.path.exists(txt):
            m = re.search(r'took ([0-9.eE+-]+) hrs', open(txt).read())
            if m:
                wall = 3600*float(m.group(1))
    return dict(tag=tag, wp=np.array(wp[:n]), P=np.array(P[:n]), status=status,
                bisections=log.count('bisection: no'), iters=sum(iters),
                wall_min=None if wall is None else wall/60)


def metrics(r):
    """Metrics of the study, see the theory document"""
    wp, P = r['wp'], r['P']
    out = {k: v for k, v in r.items() if k not in ('wp', 'P')}
    out['wp'] = wp.tolist()
    out['P'] = P.tolist()
    if len(P) == 0:
        return out
    i = int(np.argmax(P))
    out['Pmax'] = float(P[i])
    out['wp_Pmax'] = float(wp[i])
    post = np.diff(P[i:])
    # coefficient of variation of the load drop per increment after the peak,
    # small for a smooth softening, large for a staircase
    out['cv'] = float(np.std(post)/abs(np.mean(post))) if len(post) > 2 else None
    for w in (6., 7., 8.):
        j = np.flatnonzero(np.isclose(wp, w))
        out[f'P{w:.0f}'] = float(P[j[0]]) if len(j) else None
    out['wp_end'] = float(wp[-1])
    return out


def collect(out, json_path=None):
    res = {}
    for tag in cases():
        if os.path.exists(os.path.join(out, tag + '.log')):
            res[tag] = metrics(parse(tag, out))
    f = lambda v, fmt: '-' if v is None else format(v, fmt)
    print('| case | status | w_p end | P_max | w_p(P_max) | P(6) | P(7) | P(8) | CV '
          '| bisections | iterations | wall time (min) |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for tag, r in res.items():
        if 'Pmax' not in r:
            print(f'| {tag} | {r["status"]} |' + ' |'*10)
            continue
        print(f'| {tag} | {r["status"]} | {r["wp_end"]:.2f} | {r["Pmax"]:.2f} | {r["wp_Pmax"]:.2f} '
              f'| {f(r["P6"], ".2f")} | {f(r["P7"], ".2f")} | {f(r["P8"], ".2f")} | {f(r["cv"], ".2f")} '
              f'| {r["bisections"]} | {r["iters"]} | {f(r["wall_min"], ".1f")} |')
    if json_path:
        with open(json_path, 'w') as fid:
            json.dump(res, fid, indent=1)
        print('saved', json_path)
    return res


def timing(out):
    """Elastic range: matrix products against the Cython kernels"""
    p = dict(REF, case='A', m_tsl=10, n_tsl=10, nx=60, ny=30, npts=10)
    variants = {
        'matrix products, predictor, tangent reuse': dict(),
        'Cython kernels, no predictor, no tangent reuse': dict(
            use_kernels=True, predictor=False, kT_pan_reuse_steps=0),
    }
    loads = {}
    for i, (label, kwargs) in enumerate(variants.items()):
        path = os.path.join(out, f'timing_{i}')
        os.makedirs(path, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(path)
        t0 = time.time()
        # w_max = 0.3 mm gives eight increments, all in the elastic range
        _, _, _, force, _ = run_driver(f'timing_{i}', p, w_max=0.3, **kwargs)
        dt = time.time() - t0
        os.chdir(cwd)
        loads[label] = force[:, 1]
        print(f'{label}: {dt:.1f} s')
    a, b = loads.values()
    n = np.flatnonzero(a)
    print('max relative difference of the loads', np.max(np.abs(a[n] - b[n])/np.abs(b[n])))


def compare_dcb_utils(out):
    """dcb_utils.solve_dcb against the driver, case A, m_tsl = 15, nx = 120"""
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    import dcb_utils as du
    tag = 'A_m15_nx120'
    ref = parse(tag, out)
    if ref['status'] != 'done':
        raise RuntimeError(f'run {tag} not available in {out}, use "run --only {tag}"')
    k_o, tau_o = PARAMS['A']
    case = dict(E1=(138300. + 128000.)/2, E2=(10400. + 11500.)/2, nu12=0.316,
                G12=5190., h=15*0.14, b=25., L=65., a0=48., L_tsl=17.,
                G1c=1.12, tau_o=tau_o, k_o=k_o, m_tsl=15, n_tsl=10, nx=120, ny=30)
    res = du.run_or_load(case, np.linspace(0.2, 8, 40),
                         os.path.join(out, 'dcb_utils_A_m15_nx120.npz'))
    P_driver = np.interp(res['delta'], ref['wp'], ref['P'])
    dif = np.abs(res['P'] - P_driver)/np.abs(P_driver)
    for d, p, q in zip(res['delta'], res['P'], P_driver):
        print(f'delta {d:5.2f} mm: dcb_utils {p:8.2f} N, driver {q:8.2f} N, {100*(p/q - 1):+.3f}%')
    print(f'max relative difference {100*dif.max():.3f}%; '
          f'peak {res["P"].max():.2f} N (dcb_utils) against {ref["P"].max():.2f} N (driver)')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=['run', 'collect', 'timing',
                                            'compare-dcb-utils', '_worker'])
    parser.add_argument('tag', nargs='?')
    parser.add_argument('--out', default='convergence_study_runs',
                        help='folder of the runs (default: %(default)s)')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--only', nargs='+', help='tags of the cases to run')
    parser.add_argument('--json', help='file to save the metrics of collect')
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    if args.command == 'run':
        tags = args.only or list(cases())
        unknown = set(tags) - set(cases())
        if unknown:
            parser.error(f'unknown cases {sorted(unknown)}')
        run(tags, out, args.jobs)
    elif args.command == 'collect':
        collect(out, args.json)
    elif args.command == 'timing':
        timing(out)
    elif args.command == 'compare-dcb-utils':
        compare_dcb_utils(out)
    else:
        worker(args.tag, out)


if __name__ == '__main__':
    main()
