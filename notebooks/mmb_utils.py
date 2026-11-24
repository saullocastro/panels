r"""DCB, ENF and MMB coupons with mixed-mode cohesive laws

Helpers shared by the notebooks that compare the coupons of
:func:`panels.multidomain.delamination.create_coupon` with mixed-mode
delamination tests and analyses from the literature:

- :func:`build_coupon`, :func:`discretisation`: the model from a dictionary
  of inputs, the laws given as dictionaries (:func:`law_dict`);
- :func:`run_or_load`, :func:`solve`, :func:`load`: the load-displacement
  curve of :func:`.solve_coupon`, stored in a ``.npz`` file with the inputs,
  the solver settings, the versions, the run time and the machine;
- :func:`lefm_curve` and the functions before it: the load-displacement
  curve of linear elastic fracture mechanics, corrected beam theory of ASTM
  D6671/D6671M-22 [1]_;
- :func:`peak_load`, :func:`peaks_table`, :func:`read_curve`,
  :func:`plot_test`: results and references.

Units: N, mm, MPa, N/mm.

References
----------
.. [1] ASTM D6671/D6671M-22, Standard Test Method for Mixed Mode I-Mode II
   Interlaminar Fracture Toughness of Unidirectional Fiber Reinforced
   Polymer Matrix Composites, ASTM International, 2022.

"""
import json
import math
import os
import platform
import subprocess
import time

import numpy as np

import panels
from panels.multidomain.delamination import create_coupon, solve_coupon
from panels.multidomain.connections.czm import BilinearLaw

MODELS = {'clpt': 'plate_clpt_donnell', 'fsdt': 'plate_fsdt_donnell'}

# keys of ``inputs`` passed to create_coupon besides the specimen and laws
_SPECIMEN_KEYS = ('kind', 'laminaprop', 'plyt', 'stack', 'b', 'a0')
_SOLVER_DEFAULTS = dict(inc=0.1, min_inc=1e-5, relTOL=1e-9, control='load',
                        a_stop=None)


# ------------------------------------------------------------------ model
def law_dict(law):
    """Dictionary of the arguments of a :class:`.czm.BilinearLaw`

    The dictionary is stored with the results and rebuilds the law with
    ``BilinearLaw(**d)``, e.g. for a law of :func:`.czm.law_with_ratio`.
    """
    return dict(GIc=law.GIc, GIIc=law.GIIc, tauI=law.tauI, tauII=law.tauII,
                KI=law.KI, eta=law.eta, KII=law.KII, name=law.name,
                open_only=law.open_only, slip_offset=law.slip_offset,
                opening_driven=law.opening_driven,
                wake_offset=law.wake_offset)


def make_laws(laws):
    """Laws from a list of :class:`.czm.BilinearLaw` or of :func:`law_dict`"""
    return [BilinearLaw(**l) if isinstance(l, dict) else l for l in laws]


def discretisation(laminaprop, h, laws, L_coh, nd, model='clpt', m_coh=None,
                   nr_x_gauss=None, kappa_s=5/6.):
    r"""Terms and integration points of the cohesive domains

    Rules of the study of the mixed-mode cohesive model, for identical
    unidirectional arms of thickness ``h``: decay rates of the opening,
    `\lambda_I = (K_I/(2 D_{11}))^{1/4}`, and of the sliding, `\mu =
    \sqrt{8 K_{sh}/A_{11}}`, with `D_{11} = E_{11} h^3/12`, `A_{11} =
    E_{11} h` and `K_I`, `K_{sh}` summed over the laws; for FSDT arms also
    the fast decay rate of the opening of Timoshenko arms.

    Parameters
    ----------
    laminaprop : tuple
        ``(E11, E22, nu12, G12, G13, G23)``.
    h : float
        Arm thickness.
    laws : list
        Laws of the interface, see :func:`make_laws`.
    L_coh : float
        Length of the cohesive region, split into ``nd`` domains.
    nd : int
        Number of cohesive domains; 0 chooses it so that ``m_coh <= 25``.
    model : str, optional
        ``'clpt'`` or ``'fsdt'``.
    m_coh, nr_x_gauss : int, optional
        Given values are kept.
    kappa_s : float, optional
        Shear correction factor of the FSDT arms.

    Returns
    -------
    disc : dict
        ``lambda_I``, ``mu``, ``r_fast_fsdt``, ``l_cz`` (shortest mode I
        cohesive zone, `0.88 E_{22} G_{Ic}/\tau_I^2`, among the laws with at
        least 10% of the largest strength), ``L_dom``, ``nd``, ``m_coh`` =
        `\min(25, \lceil 1.25 r_{max} L_{dom} \rceil)` and ``nr_x_gauss`` =
        `\max(4 m_{coh}, 10 L_{dom}/l_{cz})` rounded up to a multiple of 10.

    """
    laws = make_laws(laws)
    E11, E22, _, _, G13, _ = laminaprop
    A11, D11 = E11*h, E11*h**3/12.
    KI = sum(l.KI for l in laws)
    KII = sum(l.KII for l in laws)
    lam = (KI/(2*D11))**0.25
    mu = math.sqrt(8*KII/A11)
    r_fast = 0.
    if model == 'fsdt':
        a = 2*KI/(kappa_s*G13*h)
        disc = a*a - 4*lam**4
        r_fast = math.sqrt((a + math.sqrt(disc))/2 if disc > 0 else a/2)
    rmax = max(lam, mu, r_fast)
    tmax = max(l.tauI for l in laws)
    lcz = min(0.88*E22*l.GIc/l.tauI**2 for l in laws if l.tauI >= 0.1*tmax)
    if not nd:
        nd = max(1, int(math.ceil(L_coh/(25/(1.25*rmax)))))
    Ld = L_coh/nd
    if m_coh is None:
        m_coh = min(25, math.ceil(1.25*rmax*Ld))
    if nr_x_gauss is None:
        nr_x_gauss = int(math.ceil(max(4*m_coh, 10*Ld/lcz)/10.)*10)
    return dict(lambda_I=lam, mu=mu, r_fast_fsdt=r_fast, l_cz=lcz, L_dom=Ld,
                nd=nd, m_coh=m_coh, nr_x_gauss=nr_x_gauss)


def build_coupon(inputs):
    """Coupon of :func:`.create_coupon` from a dictionary of inputs

    Parameters
    ----------
    inputs : dict
        ``kind``, ``laminaprop``, ``plyt``, ``stack``, ``b``, ``a0``,
        ``laws`` (see :func:`make_laws`) and any other keyword argument of
        :func:`.create_coupon` (``L``, ``c``, ``x_end``, ``model``, ``nd``,
        ``L_coh``, ``m_coh``, ...); ``model`` may be ``'clpt'`` or
        ``'fsdt'``. Optional ``tip_laws``: indices of the crack-tip laws, see
        :func:`.crack_length`.

    Returns
    -------
    md, info
        See :func:`.create_coupon`.

    """
    kw = {k: v for k, v in inputs.items()
          if k not in _SPECIMEN_KEYS + ('laws', 'tip_laws')}
    kw['model'] = MODELS.get(kw.get('model', 'clpt'), kw.get('model'))
    md, info = create_coupon(inputs['kind'], tuple(inputs['laminaprop']),
                             inputs['plyt'], list(inputs['stack']),
                             inputs['b'], inputs['a0'],
                             make_laws(inputs['laws']), **kw)
    if inputs.get('tip_laws') is not None:
        info['tip_laws'] = list(inputs['tip_laws'])
    return md, info


# ------------------------------------------------------------------ runs
def _git_commit(module):
    path = os.path.dirname(os.path.dirname(os.path.abspath(module.__file__)))
    try:
        return subprocess.check_output(['git', '-C', path, 'rev-parse',
                                        'HEAD'], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ''


def machine():
    """Processor, number of logical processors and operating system"""
    cpu = platform.processor()
    try:
        if platform.system() == 'Windows':
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION'
                               r'\System\CentralProcessor\0')
            cpu = winreg.QueryValueEx(k, 'ProcessorNameString')[0]
        elif os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo') as f:
                cpu = next(l.split(':', 1)[1] for l in f
                           if l.startswith('model name'))
    except Exception:
        pass
    return '{0}, {1} logical processors, {2} {3}'.format(
        cpu.strip(), os.cpu_count(), platform.system(), platform.release())


def _jsonable(inputs):
    out = dict(inputs)
    out['laws'] = [law_dict(l) if not isinstance(l, dict) else dict(l)
                   for l in inputs['laws']]
    return json.loads(json.dumps(out))


def solve(inputs, solver, path=None, verbose=False):
    r"""Load-displacement curve of the coupon of ``inputs``

    Parameters
    ----------
    inputs : dict
        See :func:`build_coupon`.
    solver : dict
        ``delta_max`` and, optionally, ``inc`` (initial and largest
        increment, default 0.1 mm), ``min_inc`` (1e-5), ``relTOL`` (1e-9),
        ``control`` (``'load'`` or ``'load-line'``) and ``a_stop``, see
        :func:`.solve_coupon`.
    path : str, optional
        ``.npz`` file written after every converged increment.

    Returns
    -------
    res : dict
        ``delta``, ``P``, ``a``, ``dissipated``, ``max_d``, ``t`` (wall time
        at each increment, s) and ``inputs``, ``solver``, ``status``,
        ``time``, ``size``, ``panels_version``, ``panels_commit``,
        ``structsolve_version``, ``machine``.

    """
    import structsolve
    inputs = _jsonable(inputs)
    solver = dict(_SOLVER_DEFAULTS, **solver)
    t0 = time.time()
    md, info = build_coupon(inputs)
    meta = dict(inputs=inputs, solver=solver, status='running',
                size=md.get_size(), panels_version=panels.__version__,
                panels_commit=_git_commit(panels),
                structsolve_version=getattr(structsolve, '__version__', ''),
                machine=machine())
    rows = []

    def record(an, c, row):
        rows.append(dict(row, t=time.time() - t0))
        if verbose:
            print('delta {delta:.4f} P {P:.3f} a {a:.3f} t {t:.0f}s'.format(
                  **rows[-1]), flush=True)
        if path is not None:
            _save(path, rows, meta, time.time() - t0)

    try:
        solve_coupon(md, info, solver['delta_max'],
                     initialInc=solver['inc'], maxInc=solver['inc'],
                     minInc=solver['min_inc'], relTOL=solver['relTOL'],
                     record=record, control=solver['control'],
                     a_stop=solver['a_stop'])
        meta['status'] = 'finished'
        if solver['control'] == 'load' and (
                not rows or rows[-1]['delta'] < solver['delta_max']*(1 - 1e-9)):
            meta['status'] = 'stopped at delta = {0:.4f}'.format(
                rows[-1]['delta'] if rows else 0.)
    except Exception as e:
        meta['status'] = 'error: {0!r}'.format(e)
    if path is not None:
        _save(path, rows, meta, time.time() - t0)
    return load(path) if path is not None else _as_res(rows, meta,
                                                       time.time() - t0)


_ROW_KEYS = ('delta', 'P', 'a', 'dissipated', 'max_d', 't')
_JSON_KEYS = ('inputs', 'solver')


def _as_res(rows, meta, wall):
    res = {k: np.array([r[k] for r in rows], dtype=float) for k in _ROW_KEYS}
    res.update(meta, time=wall)
    return res


def _save(path, rows, meta, wall):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = {k: np.array([r[k] for r in rows], dtype=float) for k in _ROW_KEYS}
    for k, v in meta.items():
        out[k] = np.asarray(json.dumps(v) if k in _JSON_KEYS else v)
    np.savez(path, time=wall, **out)


def load(path):
    """Results saved by :func:`solve`, ``inputs`` and ``solver`` as dicts"""
    data = np.load(path)
    res = {k: data[k] for k in data.files}
    for k in res:
        if res[k].ndim == 0:
            res[k] = res[k].item()
    for k in _JSON_KEYS:
        res[k] = json.loads(res[k])
    return res


def run_or_load(inputs, solver, path, rerun=False, verbose=False):
    """Loads the results in ``path`` or runs :func:`solve` and saves them

    A stored run whose inputs or solver settings differ from ``inputs`` and
    ``solver`` is loaded with a warning; with ``rerun=True`` it is run again.
    """
    if os.path.exists(path) and not rerun:
        res = load(path)
        want = dict(inputs=_jsonable(inputs),
                    solver=dict(_SOLVER_DEFAULTS, **solver))
        for k, v in want.items():
            diff = sorted(set(v) | set(res[k]))
            diff = [d for d in diff if v.get(d) != res[k].get(d)]
            if diff:
                print('WARNING: {0} was obtained with other {1}: {2}'.format(
                      path, k, ', '.join('{0} = {1} instead of {2}'.format(
                          d, res[k].get(d), v.get(d)) for d in diff)))
        return res
    return solve(inputs, solver, path, verbose=verbose)


def peak_load(res, L=None):
    """Largest load and its displacement, before the crack length reaches
    ``L`` (the saddle of the ENF and MMB) when given"""
    P, d = np.asarray(res['P']), np.asarray(res['delta'])
    keep = np.ones(len(P), bool) if L is None else np.asarray(res['a']) < L
    i = np.flatnonzero(keep)[np.argmax(P[keep])]
    return float(P[i]), float(d[i])


# ------------------------------------------------------------------ LEFM
def chi(E11, E22, G13):
    r"""Crack length correction parameter, [1]_ Sec. 3.3

    `\chi = \sqrt{E_{11}/(11 G_{13}) [3 - 2 (\Gamma/(1 + \Gamma))^2]}` with
    `\Gamma = 1.18 \sqrt{E_{11} E_{22}}/G_{13}`.
    """
    g = 1.18*math.sqrt(E11*E22)/G13
    return math.sqrt(E11/(11.*G13)*(3. - 2.*(g/(1. + g))**2))


def bk(GIc, GIIc, eta, B):
    """Toughness of the Benzeggagh-Kenane criterion at the mode ratio B"""
    return GIc + (GIIc - GIc)*B**eta


def lever_length(B, a, h, chi_, L):
    """Lever length for the mode ratio ``B`` = G_II/G, [1]_ Eqs. 5-7,
    without the lever weight; with ``chi_ = 0``, simple beam theory"""
    al = (1. - B)/B
    be = (a + chi_*h)/(a + 0.42*chi_*h)
    return (12.*be**2 + 3.*al + 8.*be*math.sqrt(3.*al))/(36.*be**2 - 3.*al)*L


def compliance(kind, a, b, h, E11, chi_, L=None, c=None):
    """Load-point compliance of corrected beam theory

    MMB: [1]_ Eq. 10. DCB (opening of the load points over the load) and
    ENF (mid-span deflection over the mid-span load): its mode I and mode II
    parts.
    """
    aI, aII = a + chi_*h, a + 0.42*chi_*h
    if kind == 'DCB':
        return 8.*aI**3/(b*h**3*E11)
    if kind == 'ENF':
        return (2.*L**3 + 3.*aII**3)/(8.*b*h**3*E11)
    return (8.*aI**3*(3.*c - L)**2 + (6.*aII**3 + 4.*L**3)*(c + L)**2)/(
        16.*L**2*b*h**3*E11)


def critical_load(kind, a, b, h, E11, chi_, Gc, L=None, c=None, P_g=0.,
                  c_g=0.):
    """Load at which `G_I + G_{II} = G_c`, [1]_ Eqs. 12-13 (MMB, with the
    lever weight ``P_g`` at ``c_g``: Eqs. 17-18); DCB and ENF with their
    mode I and mode II parts. Without mode I load where `3c < L`."""
    if kind == 'DCB':
        return math.sqrt(Gc*b**2*h**3*E11/(12.*(a + chi_*h)**2))
    if kind == 'ENF':
        return math.sqrt(Gc*16.*b**2*h**3*E11/(9.*(a + 0.42*chi_*h)**2))
    AI = 12.*(a + chi_*h)**2/(16.*b**2*h**3*L**2*E11)
    AII = 9.*(a + 0.42*chi_*h)**2/(16.*b**2*h**3*L**2*E11)
    kI, gI = 3.*c - L, P_g*(3.*c_g - L)
    kII, gII = c + L, P_g*(c_g + L)
    if kI < 0.:
        AI = 0.
    qa = AI*kI**2 + AII*kII**2
    qb = 2.*(AI*kI*gI + AII*kII*gII)
    qc = AI*gI**2 + AII*gII**2 - Gc
    return (-qb + math.sqrt(qb**2 - 4.*qa*qc))/(2.*qa)


def lefm_curve(kind, E11, E22, G13, b, h, a0, Gc, L=None, c=None, P_g=0.,
               c_g=0., a_max=None, n=200):
    r"""Load-displacement curve of corrected beam theory, [1]_

    Linear up to the critical load at ``a0``, then self-similar growth at
    the constant toughness ``Gc``, e.g. :func:`bk` at the nominal mode
    ratio: `P(a)` from :func:`critical_load` and `\delta = C(a) P` from
    :func:`compliance`. With the lever weight, the displacement is measured
    from the lever resting on the specimen and the mode I and mode II loads
    of the lever are `[P(3c - L) + P_g(3c_g - L)]/(4L)` and `[P(c + L) +
    P_g(c_g + L)]/L`, [1]_ Eqs. 17-18.

    Parameters
    ----------
    kind : str
        ``'DCB'``, ``'ENF'`` or ``'MMB'``.
    E11, E22, G13 : float
        Moduli, for the compliance and for :func:`chi`.
    b, h : float
        Width and arm thickness.
    a0 : float
        Initial crack length.
    Gc : float
        Toughness.
    L, c : float, optional
        Half-span and lever length (MMB, ENF).
    P_g, c_g : float, optional
        Lever weight and the lever length of its centre of gravity.
    a_max : float, optional
        Last crack length, default ``L`` (ENF, MMB) or ``a0 + 50``.
    n : int, optional
        Points of the propagation branch.

    Returns
    -------
    delta, P, a : np.ndarray

    """
    x = chi(E11, E22, G13)
    if a_max is None:
        a_max = L - 1e-6 if kind in ('ENF', 'MMB') else a0 + 50.

    def disp(P, a):
        if kind != 'MMB':
            return compliance(kind, a, b, h, E11, x, L, c)*P
        PI = (P*(3.*c - L) + P_g*(3.*c_g - L))/(4.*L)
        PII = (P*(c + L) + P_g*(c_g + L))/L
        return ((3.*c - L)/(4.*L)*compliance('DCB', a, b, h, E11, x)*PI
                + (c + L)/L*compliance('ENF', a, b, h, E11, x, L)*PII)
    d0 = disp(0., a0)
    a = np.linspace(a0, a_max, n)
    P = np.array([critical_load(kind, ai, b, h, E11, x, Gc, L, c, P_g, c_g)
                  for ai in a])
    d = np.array([disp(Pi, ai) - d0 for Pi, ai in zip(P, a)])
    return np.r_[0., d], np.r_[0., P], np.r_[a0, a]


# ------------------------------------------------------------------ output
def read_curve(path, loading_only=False):
    """Digitised curve ``delta_mm, P_N`` of a CSV file

    Records told apart by a column ``subpath`` are separated by NaN. With
    ``loading_only=True`` each record is cut after its largest displacement,
    which drops the unloading branch of a test record.
    """
    with open(path) as f:
        head = f.readline().strip().split(',')
    data = np.genfromtxt(path, delimiter=',', skip_header=1, ndmin=2)
    x, y = data[:, head.index('delta_mm')], data[:, head.index('P_N')]
    if 'subpath' in head:
        s = data[:, head.index('subpath')]
        parts = [(x[s == v], y[s == v]) for v in np.unique(s)]
    else:
        parts = [(x, y)]
    xs, ys = [], []
    for px, py in parts:
        if loading_only:
            k = int(np.argmax(px)) + 1
            px, py = px[:k], py[:k]
        xs += list(px) + [np.nan]
        ys += list(py) + [np.nan]
    return np.array(xs[:-1]), np.array(ys[:-1])


def peaks_table(rows, ref=None):
    """Markdown table ``| test | source | P_max (N) | difference |``

    Parameters
    ----------
    rows : list
        ``(test, source, P_max)``; ``P_max`` may be ``None``.
    ref : str, optional
        Source to which the differences of the other rows of the same test
        are computed.

    """
    lines = ['| test | source | $P_{max}$ (N) |'
             + (' vs {0} |'.format(ref) if ref else ''),
             '|---|---|---|' + ('---|' if ref else '')]
    refs = {t: p for t, s, p in rows if s == ref}
    for t, s, p in rows:
        line = '| {0} | {1} | {2} |'.format(t, s, '-' if p is None
                                            else '{0:.1f}'.format(p))
        if ref:
            r = refs.get(t)
            line += (' - |' if s == ref or r is None or p is None else
                     ' {0:+.1f}% |'.format(100*(p/r - 1)))
        lines.append(line)
    return '\n'.join(lines)


def plot_test(ax, curves, title=None):
    """Plots ``{label: (delta, P, kwargs)}`` on ``ax``; tests are drawn in
    grey by the caller through ``kwargs``"""
    for label, (d, p, kw) in curves.items():
        ax.plot(d, p, label=label, **kw)
    ax.set_xlabel(r'load-point displacement $\delta$ (mm)')
    ax.set_ylabel(r'load $P$ (N)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)


# ------------------------------------------------------------- study runs
def convert_study_run(run_dir, inputs, solver, path, machine,
                      panels_commit=None, structsolve_commit=None):
    r"""Stores a run of the study's driver as a results file of :func:`solve`

    The driver of the study of the mixed-mode cohesive model writes
    ``meta.json`` (inputs, discretisation, solver settings, status, wall
    time, git commits) and ``results.csv`` (``delta_mm, P_N, a_mm,
    dissipated_Nmm, max_d, t_s`` per converged increment) and calls
    :func:`.create_coupon` and :func:`.solve_coupon` as :func:`solve` does.
    The run is stored only if it was made with the model of ``inputs`` and
    the settings of ``solver``: the laws (their ``repr``), the specimen, the
    keyword arguments of :func:`.create_coupon` (missing ones at their
    defaults), the solver settings, the crack-tip laws when recorded, and
    the size of the model built here from ``inputs``. Otherwise a
    ``ValueError`` lists the differences.

    Parameters
    ----------
    run_dir : str
        Folder of the study run.
    inputs, solver : dict
        See :func:`solve`.
    path : str
        ``.npz`` file to write, read by :func:`load`.
    machine : str
        Computer of the run (the driver does not record it).
    panels_commit, structsolve_commit : str, optional
        Commits, when ``meta.json`` has none (runs made elsewhere).

    Returns
    -------
    res : dict
        As :func:`load`, with ``source``, the name of ``run_dir``.

    """
    import inspect
    with open(os.path.join(run_dir, 'meta.json')) as f:
        meta = json.load(f)
    inp = _jsonable(inputs)
    sol = dict(_SOLVER_DEFAULTS, **solver)
    bad = []

    def check(what, got, want):
        same = (abs(got - want) <= 1e-12*max(1., abs(want))
                if isinstance(got, (int, float)) and isinstance(
                    want, (int, float)) and not isinstance(got, bool)
                else got == want)
        if not same:
            bad.append('{0}: run {1!r}, inputs {2!r}'.format(what, got, want))

    check('kind', meta['kind'], inp['kind'])
    check('a0', meta['a0'], inp['a0'])
    for k in ('L', 'c'):
        if meta.get(k) is not None:
            check(k, meta[k], inp.get(k, 0. if k == 'c' else None))
    check('laminaprop', list(meta['material_used']), list(inp['laminaprop']))
    check('laws', list(meta['laws']),
          [repr(l) for l in make_laws(inp['laws'])])
    defaults = {k: p.default for k, p in
                inspect.signature(create_coupon).parameters.items()}
    model = MODELS.get(inp.get('model', 'clpt'), inp.get('model'))
    for k, v in meta['coupon'].items():
        want = model if k == 'model' else inp.get(k, defaults.get(k))
        check('coupon ' + k, v, want)
    for k, kk in (('delta_max', 'delta_max'), ('inc', 'inc'),
                  ('min_inc', 'min_inc'), ('relTOL', 'relTOL'),
                  ('control', 'control'), ('a_stop', 'a_stop')):
        check('solver ' + k, meta[kk], sol[k])
    if meta.get('tip_laws') is not None:
        check('tip_laws', list(meta['tip_laws']), inp.get('tip_laws'))
    md, info = build_coupon(inp)
    check('size', meta['size'], md.get_size())
    if bad:
        raise ValueError('{0} differs from the inputs:\n  {1}'.format(
            run_dir, '\n  '.join(bad)))
    data = np.genfromtxt(os.path.join(run_dir, 'results.csv'),
                         delimiter=',', names=True, ndmin=1)
    rows = [dict(delta=r['delta_mm'], P=r['P_N'], a=r['a_mm'],
                 dissipated=r['dissipated_Nmm'], max_d=r['max_d'],
                 t=r['t_s']) for r in data]
    vers = meta.get('versions') or {}
    out = dict(inputs=inp, solver=sol, status=meta['status'],
               size=meta['size'], panels_version=panels.__version__,
               panels_commit=vers.get('panels') or panels_commit or '',
               structsolve_version=(vers.get('structsolve')
                                    or structsolve_commit or ''),
               machine=machine, source=os.path.basename(
                   os.path.normpath(run_dir)))
    _save(path, rows, out, meta['wall_time_s'])
    return load(path)


def initial_stiffness(delta, P, lo=0.2, hi=0.6, delta_peak_max=5.):
    """Least-squares slope of the points before the peak with ``lo*P_max <=
    P <= hi*P_max``, the peak taken among ``delta < delta_peak_max``"""
    d, p = np.asarray(delta, float), np.asarray(P, float)
    ok = np.isfinite(d) & np.isfinite(p)
    d, p = d[ok], p[ok]
    i = int(np.argmax(np.where(d < delta_peak_max, p, -np.inf)))
    pre = (np.arange(d.size) <= i) & (p >= lo*p[i]) & (p <= hi*p[i])
    return float(np.polyfit(d[pre], p[pre], 1)[0])


# ------------------------------------------------------------------ superposed laws
def _law_ifaces(md, info):
    """Cohesive interfaces with laws, in the order of ``info['cohesive']``
    (increasing `x`)"""
    out = []
    for ci in info['cohesive']:
        f = md._czm.get(id(ci))
        if f is not None and len(f.laws):
            out.append(f)
    return out


def solve_laws(inputs, solver, path=None, verbose=False):
    r"""As :func:`solve`, with the share of each law of superposed laws

    Besides the results of :func:`solve`, the ``.npz`` file and the returned
    dictionary hold, after every converged increment:

    - ``diss_laws`` (increments x laws): the energy dissipated by each law,
      `\sum_g w_g \sum_k G_c^{[\ell]}(B)\,\Delta r^{[\ell]}`, the split per
      law of ``dissipated`` (:meth:`.CohesiveInterface.commit`); for a law
      with ``opening_driven=True`` and ``KII = 0`` it is the energy that the
      law dissipates, exactly;
    - ``a_laws`` (increments x laws): the length from the load line up to
      which each law has failed across the width, :func:`.crack_length`
      with that law alone;

    and, at the largest load before the crack length reaches ``inputs['L']``
    (MMB, ENF; DCB: the largest load), the state of the interface along
    `x`, mean across the width: ``x_peak`` (positions of the integration
    points from the load line), ``d_peak`` and ``tau_peak`` (laws x points:
    damage and normal traction `(1 - d) k_I \langle\Delta_w\rangle` of each
    law), ``i_peak`` (index of the increment).

    Parameters and returns as :func:`solve`.

    """
    import structsolve
    from panels.multidomain.delamination import crack_length
    inputs = _jsonable(inputs)
    solver = dict(_SOLVER_DEFAULTS, **solver)
    t0 = time.time()
    md, info = build_coupon(inputs)
    meta = dict(inputs=inputs, solver=solver, status='running',
                size=md.get_size(), panels_version=panels.__version__,
                panels_commit=_git_commit(panels),
                structsolve_version=getattr(structsolve, '__version__', ''),
                machine=machine())
    nl = len(inputs['laws'])
    L = inputs.get('L')
    rows, extra = [], dict(diss_laws=[], a_laws=[])
    acc = dict(book=np.zeros(nl), prev_r={}, Pmax=-np.inf)

    def record(an, c, row):
        ifaces = _law_ifaces(md, info)
        for k, f in enumerate(ifaces):
            r0 = acc['prev_r'].get(k, np.zeros_like(f.r))
            Gc = f._Gc_laws(c)
            acc['book'] += np.sum(f.weights*Gc*(f.r - r0), axis=1)
            acc['prev_r'][k] = f.r.copy()
        extra['diss_laws'].append(acc['book'].copy())
        extra['a_laws'].append([crack_length(md, dict(info, tip_laws=[i]))
                                for i in range(nl)])
        rows.append(dict(row, t=time.time() - t0))
        if (L is None or row['a'] < L) and row['P'] > acc['Pmax']:
            acc['Pmax'] = row['P']
            xs, ds, ts = [], [], []
            for f in ifaces:
                nx, ny = f.nr_x_gauss, f.nr_y_gauss
                dw = f.separations(c)[2]
                wp = np.maximum(dw, 0.)
                xs.append(f.p_top.x0 + f.x[:nx])
                ds.append(f.d.reshape(nl, ny, nx).mean(axis=1))
                ts.append(np.array([((1. - f.d[i])*l.KI*wp).reshape(ny, nx)
                                    .mean(axis=0)
                                    for i, l in enumerate(f.laws)]))
            extra['x_peak'] = np.concatenate(xs)
            extra['d_peak'] = np.concatenate(ds, axis=1)
            extra['tau_peak'] = np.concatenate(ts, axis=1)
            extra['i_peak'] = len(rows) - 1
        if verbose:
            print('delta {delta:.4f} P {P:.3f} a {a:.3f} t {t:.0f}s'.format(
                  **rows[-1]), flush=True)
        if path is not None:
            _save_laws(path, rows, extra, meta, time.time() - t0)

    try:
        solve_coupon(md, info, solver['delta_max'],
                     initialInc=solver['inc'], maxInc=solver['inc'],
                     minInc=solver['min_inc'], relTOL=solver['relTOL'],
                     record=record, control=solver['control'],
                     a_stop=solver['a_stop'])
        meta['status'] = 'finished'
        if solver['control'] == 'load' and (
                not rows or rows[-1]['delta'] < solver['delta_max']*(1 - 1e-9)):
            meta['status'] = 'stopped at delta = {0:.4f}'.format(
                rows[-1]['delta'] if rows else 0.)
    except Exception as e:
        meta['status'] = 'error: {0!r}'.format(e)
    if path is not None:
        _save_laws(path, rows, extra, meta, time.time() - t0)
        return load(path)
    res = _as_res(rows, meta, time.time() - t0)
    res.update({k: np.asarray(v) for k, v in extra.items()})
    return res


def _save_laws(path, rows, extra, meta, wall):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    out = {k: np.array([r[k] for r in rows], dtype=float) for k in _ROW_KEYS}
    out.update({k: np.asarray(v, dtype=float) for k, v in extra.items()})
    for k, v in meta.items():
        out[k] = np.asarray(json.dumps(v) if k in _JSON_KEYS else v)
    np.savez(path, time=wall, **out)


def run_or_load_laws(inputs, solver, path, rerun=False, verbose=False):
    """As :func:`run_or_load`, with :func:`solve_laws`"""
    if os.path.exists(path) and not rerun:
        return run_or_load(inputs, solver, path)
    return solve_laws(inputs, solver, path, verbose=verbose)
