r"""Benchmark notebooks with the penalty and the null-space connections

Reruns the ``panels`` models of the benchmark notebooks with both methods
of imposing the multidomain connections, see ``doc/source/multidomain_null_space.rst``:

- ``conn_method='penalty'``: the penalty stiffnesses used by the notebooks
- ``conn_method='null-space'``: the connections ``'SSxcte'``, ``'SSycte'``,
  ``'BFxcte'``, ``'BFycte'`` and ``'SB'`` imposed exactly

The cohesive zone ``'SB_TSL'`` and the prescribed displacements are penalty
stiffnesses in both cases. All the models are based on the classical
laminated plate theory (CLPT).

The DCB cases are taken from the notebooks themselves: their code is
executed with ``dcb_utils.run_or_load`` and ``dcb_utils.elastic_compliance``
replaced by functions that record the cases, so that the cases, the
discretisations and the openings are exactly those of the notebooks.

Usage, from the root of the repository::

    python notebooks/conn_method_study.py list
    python notebooks/conn_method_study.py run [--jobs 7] [--methods penalty null-space] [--only NAME ...]
    python notebooks/conn_method_study.py compliance
    python notebooks/conn_method_study.py stamatelos
    python notebooks/conn_method_study.py profile [--only NAME ...]

The penalty results of the DCB cases are those stored by the notebooks in
``notebooks/results/*.npz``: the penalty runs repeated with the current code
reproduce them to the round-off, so that ``conn_method_comparison.ipynb``
only needs ``run --methods null-space``.

``run`` solves the DCB cases, in parallel, longest first, writing
``notebooks/results/conn_methods/<method>/<NAME>.npz`` and the log
``<NAME>.log``, where ``<NAME>`` is the name of the results file of the
notebook. Complete results are not run again, interrupted ones are run from
the start. ``compliance`` computes the elastic compliances of the notebooks
and ``stamatelos`` runs ``stamatelos_labeas_2023_multidomain.ipynb`` with
both methods, saving ``compliance.json`` and ``stamatelos.json`` in the same
folder. ``profile`` times, for each DCB case and each method, the
components of one Newton-Raphson iteration of ``dcb_utils.solve_dcb`` at the
opening of the stored peak load, and the tangent stiffness of each panel,
updating ``profile.json``; with the iteration counts of the runs, these give
the distribution of the cost of the runs, see ``predicted_cost``. The
results are compared in ``conn_method_comparison.ipynb``.

"""
import argparse
import contextlib
import io
import json
import os
import sys
import time

#NOTE one thread per process, the cases run in parallel
for _v in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'results', 'conn_methods')
METHODS = ('penalty', 'null-space')
DCB_NOTEBOOKS = ('alfano2001_dcb.ipynb', 'camanho2003_dcb.ipynb',
                 'krueger2008_dcb.ipynb', 'tijs2022_dcb.ipynb',
                 'tijs2023_phd_dcb.ipynb', 'turon2007_dcb.ipynb',
                 'lecinana2023_dcb.ipynb')
STAMATELOS = 'stamatelos_labeas_2023_multidomain.ipynb'


def method_dir(method):
    return os.path.join(OUT, method.replace('-', '_'))


def _code_cells(notebook):
    with open(os.path.join(HERE, notebook), encoding='utf-8') as fid:
        nb = json.load(fid)
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def _exec_notebook(notebook, ns=None, after=None, quiet=True):
    r"""Executes the code cells of ``notebook`` in the folder of the
    notebooks, ``after(i, ns)`` is called after the cell ``i``"""
    ns = {'__name__': '__notebook__'} if ns is None else ns
    cwd = os.getcwd()
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    try:
        for i, src in enumerate(_code_cells(notebook)):
            out = io.StringIO()
            with contextlib.redirect_stdout(out if quiet else sys.stdout):
                exec(compile(src, '%s[%d]' % (notebook, i), 'exec'), ns)
            if after is not None:
                after(i, ns)
    finally:
        os.chdir(cwd)
        sys.path.remove(HERE)
    return ns


def collect():
    r"""DCB cases of the notebooks

    Returns
    -------
    jobs : list of dict
        ``run_or_load`` cases, with the keys ``name``, ``notebook``,
        ``case``, ``openings``, ``kwargs`` and ``time``, the run time of
        the results of the notebook.
    compliances : list of dict
        ``elastic_compliance`` cases, with the keys ``notebook``, ``case``,
        ``a0`` and ``kwargs``.

    """
    os.environ.pop('DCB_CASES', None)
    sys.path.insert(0, HERE)
    import dcb_utils as du
    sys.path.remove(HERE)
    jobs, compliances = [], []
    current = {}
    orig = du.run_or_load, du.elastic_compliance

    def record_run(case, openings, path, rerun=False, **kwargs):
        name = os.path.splitext(os.path.basename(path))[0]
        stored = os.path.join(HERE, path)
        t = float(du.load(stored)['time']) if os.path.exists(stored) else np.inf
        jobs.append(dict(name=name, notebook=current['nb'], case=dict(case),
                         openings=np.asarray(openings), kwargs=kwargs, time=t))
        if os.path.exists(stored):
            return du.load(stored)
        raise RuntimeError('no stored results for %s' % name)

    def record_compliance(case, a0, **kwargs):
        compliances.append(dict(notebook=current['nb'], case=dict(case),
                                a0=float(a0), kwargs=kwargs))
        return 1.

    du.run_or_load, du.elastic_compliance = record_run, record_compliance
    #NOTE the tables and figures of the notebooks are not shown, also when
    #     collect() is called from a notebook: display() is replaced in the
    #     namespace of the cells, patching IPython.display.display would
    #     also patch the inline backend of matplotlib if it is loaded here
    import matplotlib.pyplot as plt
    show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        for nb in DCB_NOTEBOOKS:
            current['nb'] = nb
            ns = {'__name__': '__notebook__'}
            for i, src in enumerate(_code_cells(nb)):
                #NOTE the cells that use results that are not stored, or the
                #     dummy compliances, may fail, the cases are recorded
                #     before
                try:
                    _exec_notebook_cell(nb, i, src, ns)
                except Exception:
                    pass
                ns['display'] = lambda *args, **kwargs: None
    finally:
        du.run_or_load, du.elastic_compliance = orig
        plt.show = show
        plt.close('all')
    return jobs, compliances


def _exec_notebook_cell(nb, i, src, ns):
    cwd = os.getcwd()
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, '%s[%d]' % (nb, i), 'exec'), ns)
    finally:
        os.chdir(cwd)
        sys.path.remove(HERE)


def _run_job(job, method):
    r"""Runs one DCB case in a worker process, with its own log"""
    sys.path.insert(0, HERE)
    import dcb_utils as du
    folder = method_dir(method)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, job['name'] + '.npz')
    log = os.path.join(folder, job['name'] + '.log')
    case = dict(job['case'], conn_method=method)
    kwargs = dict(job['kwargs'])
    kwargs.setdefault('verbose', True)
    t0 = time.time()
    with open(log, 'a', buffering=1) as fid, contextlib.redirect_stdout(fid):
        print('%s %s, %s, started %s' % (job['name'], method, job['notebook'],
                                        time.ctime()), flush=True)
        res = du.run_or_load(case, job['openings'], path, rerun=False, **kwargs)
        print('finished %s, %.0f s, %d openings, aborted %s'
              % (time.ctime(), time.time() - t0, len(res['delta']),
                 bool(res['aborted'])), flush=True)
    return job['name'], method, time.time() - t0


def run(jobs, methods, n_jobs):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    todo = sorted(((j, m) for j in jobs for m in methods),
                  key=lambda jm: -jm[0]['time'])
    print('%d runs, %d in parallel' % (len(todo), n_jobs), flush=True)
    with ProcessPoolExecutor(max_workers=n_jobs,
                             mp_context=mp.get_context('spawn')) as ex:
        futures = {ex.submit(_run_job, j, m): (j['name'], m) for j, m in todo}
        for f in as_completed(futures):
            name, method = futures[f]
            try:
                _, _, t = f.result()
                print('%s: done %-40s %-10s %.0f s' % (time.ctime(), name,
                                                       method, t), flush=True)
            except Exception as e:
                print('%s: FAILED %-40s %-10s %r' % (time.ctime(), name,
                                                     method, e), flush=True)


def compliance(compliances):
    sys.path.insert(0, HERE)
    import dcb_utils as du
    out = []
    for c in compliances:
        row = dict(notebook=c['notebook'], a0=c['a0'])
        for method in METHODS:
            case = dict(c['case'], conn_method=method)
            row[method] = float(du.elastic_compliance(case, c['a0'], **c['kwargs']))
        print('%s a0=%.2f: %s' % (row['notebook'], row['a0'], ', '.join(
              '%s %.6g' % (m, row[m]) for m in METHODS)), flush=True)
        out.append(row)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'compliance.json'), 'w') as fid:
        json.dump(out, fid, indent=1)


def _stamatelos_patch(method):
    r"""After the helpers of the notebook, ``MultiDomain`` gets the method
    and ``buckling`` solves the reduced problem"""
    def after(i, ns):
        if 'buckling' not in ns or ns.get('_patched'):
            return
        from functools import partial
        from scipy.sparse import csr_matrix
        from structsolve import lb
        from panels.multidomain import MultiDomain
        ns['MultiDomain'] = partial(MultiDomain, conn_method=method)

        def buckling(md, kG, num_eigvalues=6, c=None):
            kC = md.calc_kC(c=c, silent=True)
            ev, evec = lb(md.reduce(csr_matrix(kC)), md.reduce(csr_matrix(kG)),
                          silent=True, num_eigvalues=num_eigvalues)
            return ev, md.expand(evec)
        ns['buckling'] = buckling
        ns['_patched'] = True
    return after


def stamatelos():
    r"""Eigenvalues of the multidomain Stamatelos and Labeas (2023) models"""
    out = {}
    for method in METHODS:
        t0 = time.time()
        ns = _exec_notebook(STAMATELOS, after=_stamatelos_patch(method))
        r = dict(time=time.time() - t0)
        r['ref_sd'] = [float(v) for v in ns['ref_sd'][:3]]
        r['strips'] = {'%s_%g' % k: [float(v) for v in ev[:3]]
                       for k, ev in ns['res_pen'].items()}
        r['table2'] = {k: dict(Nxx=float(v['Nxx']), sd=float(v['sd']))
                       for k, v in ns['res_t2'].items()}
        r['table3'] = [float(v) for v in ns['ev_t3'][:3]]
        r['table3_sd'] = [float(v) for v in ns['EV_SD_T3'][:3]]
        r['scale'] = float(ns['SCALE'])
        r['sb_mono'] = [float(v) for v in ns['ev_mono'][:3]]
        r['sb'] = {'%s, %s' % k: [float(v) for v in ev[:3]]
                   for k, ev in ns['res_sb'].items()}
        r['table45'] = {}
        for (table, load), d in ns['res_t45'].items():
            r['table45']['%s/%s' % (table, load)] = dict(
                sd=float(d['sd']), **{'%s_%g' % k: float(v['Nxx'])
                                      for k, v in d.items()
                                      if isinstance(k, tuple)})
        r['figure7'] = {k: dict(P_md=[float(v) for v in d['P_md']],
                                P_sd=[float(v) for v in d['P_sd']])
                        for k, d in ns['res_f7'].items()}
        out[method] = r
        print('%s: %.0f s' % (method, r['time']), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'stamatelos.json'), 'w') as fid:
        json.dump(out, fid, indent=1)


def _timeit(f, rep=3):
    ts = []
    for _ in range(rep):
        t = time.perf_counter()
        out = f()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)), out


def profile_case(job, method):
    r"""Timings of the components of one iteration of ``solve_dcb``

    The state is the linear solution at the opening of the stored peak load,
    where the damage grows ahead of the crack tip, so that the damage-rate
    stiffness of the cohesive zone is not empty. The components are:

    - ``kT_pan``: tangent stiffness of the panels, ``calc_kT``, timed once
      at ``c = 0``, the cost of its numerical integration does not depend on
      the state
    - ``kC_conn`` and ``fint``: the residual, ``get_kC_conn`` and
      ``calc_fint``
    - ``kT_TSL``: damage-rate stiffness of the cohesive zone
    - ``sum``, ``reduce``, ``vec``: sum of the matrices, `T^T K T` and the
      products of the vectors by `T` (null-space only)
    - ``solve``: ``structsolve.solve``
    - ``accept`` and ``output``: damage update and reactions of an opening

    The components other than ``kT_pan``, ``accept`` and ``output`` are the
    median of 3 repetitions.

    Returns
    -------
    out : dict
        Timings in ``'t'``, the setup times ``t_build``, ``t_T`` and
        ``t_prescribe``, the sizes and the ``cProfile`` statistics of one
        iteration without ``kT_pan``.

    """
    import cProfile
    import pstats
    from scipy.sparse import csr_matrix
    from structsolve import solve
    from structsolve.sparseutils import finalize_symmetric_matrix
    from panels.multidomain.connections import fkCld_xcte
    sys.path.insert(0, HERE)
    import dcb_utils as du

    case = dict(job['case'], conn_method=method)
    kw = job['kwargs'].get('kw', 1.e6)
    out = dict(name=job['name'], method=method)
    out['t_build'], model = _timeit(lambda: du.build_dcb(case), rep=1)
    assy = model['assy']
    top_arm, bot_arm = model['top_arm'], model['bot_arm']
    top_tsl, bot_tsl = model['top_tsl'], model['bot_tsl']
    nx, ny, size = model['nx'], model['ny'], model['size']
    out['size'] = size
    if method == 'null-space':
        t = time.perf_counter()
        T = assy.get_T()
        out['t_T'] = time.perf_counter() - t
        out['size_r'] = T.shape[1]
        out['nnz_T'] = T.nnz
        red = lambda v: T.T @ v
        red_mat = lambda K: csr_matrix(T.T @ K @ T)
        expand = lambda v: T @ v
    else:
        out['t_T'] = 0.
        out['size_r'] = size
        red = red_mat = expand = lambda x: x

    history = np.zeros((ny, nx))
    if case.get('bond_width') is not None:
        eta, _ = np.polynomial.legendre.leggauss(ny)
        y = top_tsl.b/2*(eta + 1)
        history[np.abs(y - top_tsl.b/2) > case['bond_width']/2, :] = 1.
    assy.update_TSL_history(curr_max_dmg_index=history)

    def prescribe(delta):
        kCp = 0
        for p, sign in ((top_arm, +1), (bot_arm, -1)):
            p.clear_disps()
            kCp += fkCld_xcte(0., 0., kw, p, p.a, size, p.row_start, p.col_start)
            p.add_distr_pd_fixed_x(p.a, None, None, kw, funcu=None, funcv=None,
                                   funcw=lambda y, s=sign: s*delta/2)
        kCp = finalize_symmetric_matrix(kCp)
        return kCp, assy.calc_fext()

    st = du.load(os.path.join(HERE, 'results', job['name'] + '.npz'))
    delta = float(st['delta'][int(np.argmax(st['P']))])
    out['delta'] = delta
    out['t_prescribe'], (kCp, fext) = _timeit(lambda: prescribe(delta), rep=1)

    tm = {}
    c = np.zeros(size)
    kC0 = assy.get_kC_conn(c=c)
    tm['kT_pan'], kT_pan = _timeit(lambda: assy.calc_kT(c=c, kC_conn=0.), rep=1)
    k0 = red_mat(kT_pan + kC0 + kCp)
    c = expand(solve(k0, red(fext), silent=True))

    tm['kC_conn'], kC_conn = _timeit(lambda: assy.get_kC_conn(c=c))
    tm['fint'], fint = _timeit(lambda: np.asarray(assy.calc_fint(c=c, kC_conn=kC_conn)))
    tm['kT_TSL'], kT_TSL = _timeit(lambda: assy.calc_kT_TSL(c=c))
    tm['sum'], K = _timeit(lambda: kT_pan + kC_conn + kT_TSL + kCp)
    tm['reduce'], k0 = _timeit(lambda: red_mat(K))
    R = fint - fext + kCp*c
    tm['solve'], dc = _timeit(lambda: expand(solve(k0, -red(R), silent=True)))
    tm['vec'], _ = _timeit(lambda: (red(R), expand(red(R))))
    tm['accept'], _ = _timeit(lambda: assy.calc_k_dmg(c=c, pA=top_tsl, pB=bot_tsl,
            nr_x_gauss=nx, nr_y_gauss=ny, tsl_type='bilinear',
            prev_max_dmg_index=assy.dmg_index, k_i=case['k_o'],
            tau_o=case['tau_o'], G1c=case['G1c']), rep=1)
    tm['output'], _ = _timeit(lambda: (assy.reaction_line_pd_xcte(c, top_arm,
            top_arm.a, kw, lambda y: delta/2),
            assy.force_out_plane_damage(conn=model['conn'], c=c)), rep=1)
    out['t'] = tm
    out['nnz_K'] = int(csr_matrix(K).nnz)
    out['nnz_k0'] = int(csr_matrix(k0).nnz)
    out['kT_TSL_nnz'] = int(csr_matrix(kT_TSL).nnz) if not np.isscalar(kT_TSL) else 0

    pr = cProfile.Profile()
    pr.enable()
    kC_conn = assy.get_kC_conn(c=c)
    fint = np.asarray(assy.calc_fint(c=c, kC_conn=kC_conn))
    k0 = red_mat(kT_pan + kC_conn + assy.calc_kT_TSL(c=c) + kCp)
    dc = expand(solve(k0, -red(fint - fext + kCp*c), silent=True))
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(12)
    out['cprofile'] = s.getvalue()
    return out


def profile_panels(job):
    r"""Time of ``calc_kC`` and ``calc_kG`` of each panel, with
    ``NLgeom=True``, the parts of ``kT_pan`` of :func:`profile_case`"""
    sys.path.insert(0, HERE)
    import dcb_utils as du
    model = du.build_dcb(dict(job['case']))
    assy, size = model['assy'], model['size']
    c = np.zeros(size)
    rows = []
    for p in assy.panels:
        r = dict(group=p.group, a=p.a, m=p.m, n=p.n, nx=p.nx, ny=p.ny,
                 tsl=any(p is q for q in (model['top_tsl'], model['bot_tsl'])))
        t = time.perf_counter()
        p.calc_kC(c=c, size=size, row0=p.row_start, col0=p.col_start,
                  silent=True, finalize=False, NLgeom=True)
        r['t_kC'] = time.perf_counter() - t
        t = time.perf_counter()
        p.calc_kG(c=c, size=size, row0=p.row_start, col0=p.col_start,
                  silent=True, finalize=False, NLgeom=True)
        r['t_kG'] = time.perf_counter() - t
        rows.append(r)
    return rows


def profile(jobs, panels=True):
    r"""Runs :func:`profile_case` for the cases, with both methods one after
    the other for each case, and :func:`profile_panels`, updating the
    entries of the cases in ``profile.json``"""
    path = os.path.join(OUT, 'profile.json')
    out = dict(iterations={}, panels={})
    if os.path.exists(path):
        with open(path) as fid:
            out = json.load(fid)
    for j in jobs:
        for method in METHODS:
            r = profile_case(j, method)
            print('%-36s %-10s kT_pan %.1f s, rest of the iteration %.2f s'
                  % (j['name'], method, r['t']['kT_pan'], sum(v for k, v in
                     r['t'].items() if k not in ('kT_pan', 'accept', 'output'))),
                  flush=True)
            out['iterations']['%s|%s' % (j['name'], method)] = r
        if panels:
            out['panels'][j['name']] = profile_panels(j)
    os.makedirs(OUT, exist_ok=True)
    with open(path, 'w') as fid:
        json.dump(out, fid, indent=1)


def iteration_counts(res, bond_width=None, kT_pan_reuse_steps=5,
                     NR_kT_update=3):
    r"""Calls of the components of ``solve_dcb`` in a run

    Follows the rules of ``dcb_utils.solve_dcb`` opening by opening, from
    the iterations of each opening in ``res['iters']``: the tangent of the
    panels is computed at the first iteration of each opening, unless it is
    reused while no point is softening, and then every ``NR_kT_update``
    iterations. The softening is detected from ``res['dmax']`` of the
    previous opening or, with a ``bond_width``, where the points outside
    the bonded strip are fully damaged from the start, from the change of
    the secant stiffness `P/\delta`. One trial of the line search per
    iteration is assumed, the number of trials is not stored; the runs have
    no bisections.

    Returns
    -------
    n : dict
        ``kT_pan`` (tangents of the panels), ``iter`` (iterations,
        each with ``kT_TSL``, ``sum``, ``reduce``, ``vec`` and ``solve``),
        ``residual`` and ``opening``.

    """
    it = np.asarray(res['iters'])
    dmax = np.asarray(res['dmax'])
    P, d = np.asarray(res['P']), np.asarray(res['delta'])
    n = dict(kT_pan=0, iter=0, residual=0, opening=len(it))
    age = None
    for i, N in enumerate(it):
        if i == 0:
            elastic = True
        elif bond_width is not None:
            elastic = abs(P[i-1]/d[i-1]/(P[0]/d[0]) - 1) < 1e-5
        else:
            elastic = dmax[i-1] <= 0.
        if age is None or not elastic or age >= kT_pan_reuse_steps:
            n['kT_pan'] += 1
            age = 0
        else:
            age += 1
        for count in range(1, N):
            if count % NR_kT_update == 1 and not (elastic and count == 1):
                n['kT_pan'] += 1
                age = 0
        n['iter'] += N
        n['residual'] += N + 1
    return n


def predicted_cost(prof, n, kT_pan=None):
    r"""Cost of a run, in s, by component, from the timings ``prof`` of
    :func:`profile_case` and the counts ``n`` of :func:`iteration_counts`;
    ``kT_pan`` replaces the timing of the tangent of the panels"""
    t = prof['t']
    kT = t['kT_pan'] if kT_pan is None else kT_pan
    return dict(
        setup=prof['t_build'] + prof['t_T'],
        kT_pan=n['kT_pan']*kT,
        residual=n['residual']*(t['kC_conn'] + t['fint']),
        kT_TSL=n['iter']*t['kT_TSL'],
        reduce=n['iter']*(t['reduce'] + t['sum'] + t['vec']),
        solve=n['iter']*t['solve'],
        opening=n['opening']*(prof['t_prescribe'] + t['accept'] + t['output']),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=('list', 'run', 'compliance',
                                            'stamatelos', 'profile'))
    parser.add_argument('--jobs', type=int, default=7)
    parser.add_argument('--methods', nargs='+', default=list(METHODS),
                        choices=METHODS)
    parser.add_argument('--only', nargs='+')
    args = parser.parse_args()
    if args.command == 'stamatelos':
        return stamatelos()
    jobs, compliances = collect()
    if args.only:
        jobs = [j for j in jobs if j['name'] in args.only]
    if args.command == 'list':
        for j in sorted(jobs, key=lambda j: -j['time']):
            print('%-36s %-24s %3d openings, stored run %6.0f s'
                  % (j['name'], j['notebook'], len(j['openings']), j['time']))
        print('%d elastic compliances: %s' % (len(compliances), sorted(set(
              c['notebook'] for c in compliances))))
    elif args.command == 'run':
        run(jobs, args.methods, args.jobs)
    elif args.command == 'profile':
        profile(jobs)
    else:
        compliance(compliances)


if __name__ == '__main__':
    main()
