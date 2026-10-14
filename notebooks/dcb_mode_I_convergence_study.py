r"""Convergence study of the cohesive zone ('SB_TSL') on the reference DCB

The reference DCB is the specimen of Tijs et al. (2022) with a0 = 48 mm, as
in their finite element model (Table 2): length 100 mm, width 25 mm, arms of
2.1 mm of AS4D/PEKK-FC (average of the tensile and compressive moduli of
their Table 1), G1c = 1.12 N/mm and tau_o = 87 MPa, the matrix tensile
strength. It is solved with the validation model of ``dcb_utils``
(``conn_method='penalty'``): perfect bond, cohesive domain of length L_tsl
and free arms, clamped far end, arms opened to +-delta/2, up to delta = 8 mm.
Two penalty stiffnesses of the interface are used, with the same strength:

- case A: k_o = 1e5 N/mm^3, the value of the validation runs;
- case B: k_o = 2e5 N/mm^3, the mode I penalty stiffness used for
  AS4D/PEKK-FC by Tijs et al. (2022b), Compos. Struct. 281, 114964.

Usage, from the root of the repository::

    python notebooks/dcb_mode_I_convergence_study.py run [--jobs 4]
        [--only A_m15_nx120 ...]
        [--longest-first notebooks/results/dcb_mode_I_convergence_study.json]
    python notebooks/dcb_mode_I_convergence_study.py collect
        [--json notebooks/results/dcb_mode_I_convergence_study.json]

``run`` solves each case in its own process, in parallel, writing the log of
case TAG to ``<out>/TAG.log`` and the results to ``<out>/TAG.npz``; a case
whose ``.npz`` is complete is not run again. Each process uses
``cpu_count/jobs`` threads. With ``--longest-first`` the cases are started
from the longest to the shortest wall time stored in a ``.json`` of a
previous ``collect``. ``collect`` prints the metrics of the study and
optionally stores them in a ``.json`` file, with the wall time ``wall_min``
and the CPU time ``cpu_min`` of each run, in minutes.

The results in ``notebooks/results/dcb_mode_I_convergence_study.json`` were
obtained with ``run --jobs 8 --longest-first`` on a machine with 16 threads,
eight runs at a time, and ``collect --json``. The folder of the runs,
``dcb_mode_I_convergence_study_runs``, is not part of the repository.

"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# AS4D/PEKK-FC, Tijs et al. (2022), Table 1, average of the tensile and
# compressive moduli, and their finite element model (Table 2)
DCB = dict(E1=(138300. + 128000.)/2, E2=(10400. + 11500.)/2, nu12=0.316,
           G12=5190., h=2.1, b=25., L=100., a0=48., G1c=1.12, tau_o=87.,
           conn_method='penalty')
# penalty stiffness of the interface of the two cases
K_O = {'A': 1e5, 'B': 2e5}
# reference discretisation: length L_tsl and terms m_tsl, n_tsl of the
# cohesive domain, terms m, n of the other domains, Gauss points nx, ny
REF = dict(L_tsl=24., m_tsl=25, n_tsl=10, m=8, n=8, nx=180, ny=30, dw=0.2)
W_MAX = 8.
SOLVER = {}


def openings(dw):
    """23 equal increments up to 5 mm, then increments of ``dw`` up to 8 mm"""
    n = int(round((W_MAX - 5.)/dw))
    return np.concatenate((np.linspace(5./23, 5., 23), 5. + dw*np.arange(1, n + 1)))


def cases():
    """Returns {tag: parameters} of the runs of the study"""
    out = {}
    for c in K_O:
        for m_tsl in (10, 15, 20, 25, 30):
            for nx in (60, 120, 180):
                out[f'{c}_m{m_tsl}_nx{nx}'] = dict(REF, case=c, m_tsl=m_tsl, nx=nx)
        # variations from the reference discretisation
        out[f'{c}_n15'] = dict(REF, case=c, n_tsl=15)
        out[f'{c}_ny60'] = dict(REF, case=c, ny=60)
        out[f'{c}_mn12'] = dict(REF, case=c, m=12, n=12)
        # halved increment after delta = 5 mm
        out[f'{c}_w01'] = dict(REF, case=c, dw=0.1)
        # shorter cohesive domain, the compressive lobe ahead of the process
        # zone reaches the perfect bond at the end of the analysis
        out[f'{c}_L17_m25_nx180'] = dict(REF, case=c, L_tsl=17.)
        out[f'{c}_L17_m15_nx120'] = dict(REF, case=c, L_tsl=17., m_tsl=15, nx=120)
    return out


def case_dict(p):
    keys = ('L_tsl', 'm_tsl', 'n_tsl', 'm', 'n', 'nx', 'ny')
    return dict(DCB, k_o=K_O[p['case']], **{k: p[k] for k in keys})


def worker(tag, out):
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    import panels
    import dcb_utils as du
    p = cases()[tag]
    case = case_dict(p)
    model = du.build_dcb(case)
    print(f'panels {panels.__version__} from {os.path.dirname(panels.__file__)}, '
          f'conn_method={model["assy"].conn_method}', flush=True)
    assert model['assy'].conn_method == 'penalty'
    print(tag, case, flush=True)
    res = du.run_or_load(case, openings(p['dw']), os.path.join(out, tag + '.npz'),
                         verbose=True, **SOLVER)
    print(f'WALLTIME_S {res["time"]:.1f}')
    print(f'CPUTIME_S {res["cpu_time"]:.1f}')


def run(tags, out, jobs):
    """Runs the cases in parallel, one process per case"""
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ)
    threads = str(max(1, (os.cpu_count() or 1)//jobs))
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[k] = threads
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
            print('started', tag, flush=True)
        time.sleep(5)
        for item in running[:]:
            tag, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                print('finished', tag, 'exit code', proc.returncode, flush=True)


def metrics(res):
    """Metrics of the study, see doc/source/cohesive_zone.rst"""
    w, P = np.asarray(res['delta']), np.asarray(res['P'])
    out = dict(delta=w.tolist(), P=P.tolist(), P_traction=np.asarray(res['P_traction']).tolist(),
               a=np.asarray(res['a']).tolist(), iters=int(np.sum(res['iters'])),
               bisections=int(res['failed']), aborted=bool(res['aborted']),
               wall_min=float(res['time'])/60)
    if 'cpu_time' in res:
        out['cpu_min'] = float(res['cpu_time'])/60
    if len(P) == 0:
        return out
    i = int(np.argmax(P))
    out['Pmax'] = float(P[i])
    out['w_Pmax'] = float(w[i])
    post = np.diff(P[i:])
    # coefficient of variation of the load drop per increment after the peak,
    # small for a smooth softening, large for a staircase
    out['cv'] = float(np.std(post)/abs(np.mean(post))) if len(post) > 2 else None
    for v in (6., 7., 8.):
        j = np.flatnonzero(np.isclose(w, v))
        out[f'P{v:.0f}'] = float(P[j[0]]) if len(j) else None
    out['w_end'] = float(w[-1])
    out['a_end'] = float(res['a'][-1])
    Pt = np.asarray(res['P_traction'])
    out['max_rel_P_traction'] = float(np.max(np.abs(Pt - P)/np.abs(P)))
    return out


def collect(out, json_path=None):
    res = {}
    for tag in cases():
        path = os.path.join(out, tag + '.npz')
        if os.path.exists(path):
            data = np.load(path)
            res[tag] = metrics({k: data[k] for k in data.files})
    f = lambda v, fmt: '-' if v is None else format(v, fmt)
    print('| case | delta end | a end | P_max | delta(P_max) | P(6) | P(7) | P(8) | CV '
          '| bisections | iterations | wall time (min) | max |P_tr/P - 1| |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for tag, r in res.items():
        if 'Pmax' not in r:
            print(f'| {tag} |' + ' |'*12)
            continue
        print(f'| {tag} | {r["w_end"]:.2f} | {r["a_end"]:.2f} | {r["Pmax"]:.2f} | {r["w_Pmax"]:.2f} '
              f'| {f(r["P6"], ".2f")} | {f(r["P7"], ".2f")} | {f(r["P8"], ".2f")} | {f(r["cv"], ".2f")} '
              f'| {r["bisections"]} | {r["iters"]} | {r["wall_min"]:.1f} | {r["max_rel_P_traction"]:.1e} |')
    if json_path:
        with open(json_path, 'w') as fid:
            json.dump(dict(reference_dcb=DCB, k_o=K_O, ref=REF, runs=res), fid, indent=1)
        print('saved', json_path)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=['run', 'collect', '_worker'])
    parser.add_argument('tag', nargs='?')
    parser.add_argument('--out', default='dcb_mode_I_convergence_study_runs',
                        help='folder of the runs (default: %(default)s)')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--only', nargs='+', help='tags of the cases to run')
    parser.add_argument('--json', help='file to save the metrics of collect')
    parser.add_argument('--longest-first', metavar='JSON',
                        help='starts the runs from the longest wall time of this file')
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    if args.command == 'run':
        tags = args.only or list(cases())
        unknown = set(tags) - set(cases())
        if unknown:
            parser.error(f'unknown cases {sorted(unknown)}')
        if args.longest_first:
            prev = json.load(open(args.longest_first))['runs']
            tags.sort(key=lambda t: -prev.get(t, {}).get('wall_min', 0.))
        run(tags, out, args.jobs)
    elif args.command == 'collect':
        collect(out, args.json)
    else:
        worker(args.tag, out)


if __name__ == '__main__':
    main()
