r"""Reruns the nonlinear runs of the mode I DCB validation notebooks in parallel

The notebooks run their cases one after the other with
``dcb_utils.run_or_load``. This script executes the code cells of each
notebook with ``run_or_load`` replaced by a recorder, which stores the case,
the openings and the results file of every run and returns the cached results
(if any) so that the remaining cells execute. The recorded runs are then
solved again, one process per run, overwriting ``results/*.npz``.

Usage, from the root of the repository::

    python notebooks/dcb_mode_I_validation_rerun.py list
    python notebooks/dcb_mode_I_validation_rerun.py run [--jobs 7] [--only FILE ...]

The logs of the runs are written to ``dcb_mode_I_validation_rerun_runs``, a
folder that is not part of the repository. After the runs, execute the
notebooks again (``RERUN = False``) to update their outputs.

"""
import argparse
import contextlib
import io
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(REPO, 'dcb_mode_I_validation_rerun_runs')
NOTEBOOKS = ['dcb_alfano2001.ipynb', 'dcb_camanho2003.ipynb', 'dcb_turon2007.ipynb',
             'dcb_krueger2008.ipynb', 'dcb_tijs2022.ipynb', 'dcb_tijs2023_phd.ipynb']


def record():
    """Executes the notebooks and returns the runs they ask for"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    os.chdir(HERE)
    import dcb_utils as du
    try:
        import IPython.display
        IPython.display.display = lambda *args, **kwargs: None
    except ImportError:
        pass
    runs = []

    def recorder(case, openings, path, rerun=False, **kwargs):
        runs.append(dict(notebook=name, path=os.path.abspath(path), case=case,
                         openings=np.asarray(openings), kwargs=kwargs))
        if os.path.exists(path):
            return du.load(path)
        n = len(openings)
        return dict(delta=np.asarray(openings), P=np.zeros(n), P_traction=np.zeros(n),
                    a=np.full(n, case['a0']), dmax=np.zeros(n), iters=np.zeros(n, int),
                    failed=0, aborted=False, time=0.)

    du.run_or_load = recorder
    for name in NOTEBOOKS:
        nb = json.load(open(os.path.join(HERE, name), encoding='utf-8'))
        ns = {'__name__': 'notebook'}
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(''.join(cell['source']), name, 'exec'), ns)
            except Exception as e:
                print(f'{name}, cell {i}: {type(e).__name__}: {e}')
            plt.close('all')
    return runs


def cost(run):
    """Wall time of the cached run, to start the longest runs first"""
    try:
        return float(np.load(run['path'])['time'])
    except Exception:
        return np.inf


def worker(index):
    sys.path.insert(0, REPO)
    sys.path.insert(0, HERE)
    import panels
    import dcb_utils as du
    run = pickle.load(open(os.path.join(OUT, 'runs.pkl'), 'rb'))[index]
    case = run['case']
    print(f'panels {panels.__version__} from {os.path.dirname(panels.__file__)}', flush=True)
    print(run['notebook'], run['path'], flush=True)
    print({k: v for k, v in case.items() if k not in ('curves', 'source')}, flush=True)
    print('openings', len(run['openings']), run['openings'][-1], run['kwargs'], flush=True)
    assert case.get('conn_method', 'penalty') == 'penalty'
    if os.path.exists(run['path']):
        os.remove(run['path'])
    res = du.solve_dcb(case, run['openings'], save=run['path'], verbose=True, **run['kwargs'])
    print(f'WALLTIME_S {res["time"]:.1f}')
    print(f'CPUTIME_S {res["cpu_time"]:.1f}')
    print(f'ABORTED {res["aborted"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=['list', 'run', '_worker'])
    parser.add_argument('index', nargs='?', type=int)
    parser.add_argument('--jobs', type=int, default=7)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    if args.command == '_worker':
        worker(args.index)
        return
    runs = record()
    runs.sort(key=cost, reverse=True)
    if args.command == 'list':
        for i, r in enumerate(runs):
            c = r['case']
            print(f"{i:2d} {os.path.basename(r['path']):45s} L_tsl={c['L_tsl']:.1f} "
                  f"m_tsl={c['m_tsl']} nx={c['nx']} k_o={c['k_o']:.0e} "
                  f"openings={len(r['openings'])} cached {cost(r)/3600:.1f} h")
        return
    if args.only:
        runs = [r for r in runs if os.path.basename(r['path']) in args.only]
    os.makedirs(OUT, exist_ok=True)
    pickle.dump(runs, open(os.path.join(OUT, 'runs.pkl'), 'wb'))
    env = dict(os.environ)
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[k] = str(args.threads)
    pending, running = list(range(len(runs))), []
    while pending or running:
        while pending and len(running) < args.jobs:
            i = pending.pop(0)
            tag = os.path.splitext(os.path.basename(runs[i]['path']))[0]
            log = open(os.path.join(OUT, tag + '.log'), 'w')
            proc = subprocess.Popen([sys.executable, '-u', os.path.abspath(__file__),
                                     '_worker', str(i)], stdout=log,
                                    stderr=subprocess.STDOUT, env=env, cwd=HERE)
            running.append((tag, proc, log))
            print('started', tag, time.strftime('%H:%M:%S'), flush=True)
        time.sleep(5)
        for item in running[:]:
            tag, proc, log = item
            if proc.poll() is not None:
                log.close()
                running.remove(item)
                print('finished', tag, 'exit code', proc.returncode,
                      time.strftime('%H:%M:%S'), flush=True)


if __name__ == '__main__':
    main()
