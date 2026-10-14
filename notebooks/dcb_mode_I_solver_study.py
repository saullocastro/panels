r"""Solution procedure of the cohesive zone on the reference DCB

Runs the reference DCB of ``dcb_mode_I_convergence_study.py``, case A, with
the reference discretisation, and with parts of the solution procedure
switched off:

- ``ref``: consistent tangent, line search and bisection (the default of
  ``dcb_utils.solve_dcb``), for the wall time and CPU time of the reference
  analysis, to be run alone on the machine;
- ``secant``: secant stiffness of the cohesive zone only
  (``consistent_tangent=False``);
- ``nols``: consistent tangent without line search (``line_search_max=0``);
- ``nols_nx60`` and ``ls_nx60``: without and with line search, with
  ``nx = 60``, where the runs of the convergence study needed bisections.

The runs other than ``ref`` have no bisection (``max_bisections=0``) and
up to ``max_NR_iter=200`` iterations per opening, so they stop at the first
opening that does not converge. Every Newton-Raphson iteration is traced,
``(delta, iteration, criterion, step)``, to detect a limit cycle: ``collect``
reports the smallest period with which the criterion of the last opening
repeats. The results are discussed in Section *Solution procedure* of
``doc/source/cohesive_zone.rst``.

Usage, from the root of the repository::

    python notebooks/dcb_mode_I_solver_study.py run ref
    python notebooks/dcb_mode_I_solver_study.py run secant nols --jobs 2
        [--threads 1]
    python notebooks/dcb_mode_I_solver_study.py collect

``run`` solves each run in its own process, ``--jobs`` at a time, writing to
``dcb_mode_I_solver_study_runs`` the log of run TAG, ``TAG.log``, its
results, ``TAG.npz``, and its iterations, ``TAG_trace.npy``. ``--threads``
sets the threads of each process, by default ``cpu_count/jobs`` when ``jobs
> 1``. The folder ``dcb_mode_I_solver_study_runs`` is not part of the
repository.

"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, 'dcb_mode_I_solver_study_runs')

RUNS = {
    'ref': dict(),
    'secant': dict(consistent_tangent=False, max_NR_iter=200, max_bisections=0),
    'nols': dict(line_search_max=0, max_NR_iter=200, max_bisections=0),
    # integration points too sparse (3.6 per cohesive zone length), where the
    # runs of the convergence study needed bisections: without and with line
    # search, both without bisection
    'nols_nx60': dict(line_search_max=0, max_NR_iter=200, max_bisections=0),
    'ls_nx60': dict(max_NR_iter=200, max_bisections=0),
}
# discretisation of each run, default: the reference A_m25_nx180
BASE = {'nols_nx60': 'A_m25_nx60', 'ls_nx60': 'A_m25_nx60'}


def worker(tag):
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    import panels
    import dcb_utils as du
    import dcb_mode_I_convergence_study as cs
    p = cs.cases()[BASE.get(tag, 'A_m25_nx180')]
    case = cs.case_dict(p)
    assert case['conn_method'] == 'penalty'
    print(f'panels {panels.__version__} from {os.path.dirname(panels.__file__)}', flush=True)
    print(tag, RUNS[tag], case, flush=True)
    trace = []
    res = du.solve_dcb(case, cs.openings(p['dw']), verbose=True, trace=trace,
                       save=os.path.join(OUT, tag + '.npz'), **RUNS[tag])
    np.save(os.path.join(OUT, tag + '_trace.npy'), np.array(trace))
    print(f'WALLTIME_S {res["time"]:.1f}')
    print(f'CPUTIME_S {res["cpu_time"]:.1f}')
    print(f'ABORTED {res["aborted"]} at delta={res["delta"][-1] if len(res["delta"]) else 0}')


def run(tags, jobs, threads=None):
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ)
    if jobs > 1 or threads:
        threads = str(threads or max(1, (os.cpu_count() or 1)//jobs))
        for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[k] = threads
    pending, running = list(tags), []
    while pending or running:
        while pending and len(running) < jobs:
            tag = pending.pop(0)
            log = open(os.path.join(OUT, tag + '.log'), 'w')
            proc = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__),
                                     '_worker', tag], stdout=log,
                                    stderr=subprocess.STDOUT, env=env)
            running.append((tag, proc, log))
            print('started', tag, flush=True)
        time.sleep(5)
        for item in running[:]:
            tag, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                print('finished', tag, 'exit code', proc.returncode, flush=True)


def cycle(crit, max_period=12, repeats=3, rtol=1e-9):
    """Smallest period with which the last values of ``crit`` repeat"""
    crit = np.asarray(crit)
    for period in range(1, max_period + 1):
        n = period*repeats
        if len(crit) < n:
            break
        tail = crit[-n:].reshape(repeats, period)
        if np.allclose(tail, tail[-1], rtol=rtol, atol=0):
            return period, tail[-1]
    return None, None


def collect():
    for tag in RUNS:
        f = os.path.join(OUT, tag + '.npz')
        if not os.path.exists(f):
            print(tag, 'not run')
            continue
        r = np.load(f)
        it = r['iters']
        print(f"{tag}: openings {len(r['delta'])}, last delta {r['delta'][-1]:.2f}, "
              f"aborted {bool(r['aborted'])}, bisections {int(r['failed'])}, "
              f"iterations {int(it.sum())} (max {int(it.max())} per opening), "
              f"wall {float(r['time'])/60:.1f} min"
              + (f", CPU {float(r['cpu_time'])/60:.1f} min" if 'cpu_time' in r else ''))
        ft = os.path.join(OUT, tag + '_trace.npy')
        if os.path.exists(ft):
            tr = np.load(ft)
            last = tr[tr[:, 0] == tr[-1, 0]]
            period, values = cycle(last[:, 2])
            print(f'    last opening tried {tr[-1, 0]:.4f}: {len(last)} iterations, '
                  f'last criteria {np.array2string(last[-6:, 2], precision=4)}, '
                  f'steps {last[-6:, 3]}, cycle period {period}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=['run', 'collect', '_worker'])
    parser.add_argument('tags', nargs='*')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--threads', type=int, default=None)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.tags or list(RUNS), args.jobs, args.threads)
    elif args.command == 'collect':
        collect()
    else:
        worker(args.tags[0])
