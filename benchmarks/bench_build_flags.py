r"""Computational cost of the Ritz matrices, to assess the build flags

Times ``Shell.calc_kC()``, ``calc_kG()`` and ``calc_kM()``, integrated
analytically, and ``calc_kC(c=c)``, ``calc_kG(c=c)`` and ``calc_fint(c)``,
integrated numerically, for several models with ``m = n = 12``. It is used to
assess the Cython directives and the compiler flags of ``setup.py``, see
``CHANGELOG.md``.

Run it against the build of panels to be assessed, e.g.::

    PYTHONPATH=<path to panels repository> python bench_build_flags.py

"""
import json
import sys
import time
import warnings

import numpy as np

import panels
from panels.shell import Shell

REPEAT = 5
MODELS = ['plate_clpt_donnell', 'cylshell_clpt_sanders',
          'plate_fsdt_donnell', 'cylshell_fsdt_sanders',
          'cylshell_tsdt_donnell']


def pin_process():
    # NOTE a single core and a high priority reduce the timing noise
    try:
        import psutil
        p = psutil.Process()
        p.cpu_affinity([p.cpu_affinity()[-1]])
        if hasattr(psutil, 'HIGH_PRIORITY_CLASS'):
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10)
    except Exception:
        pass


def timeit(func):
    best = np.inf
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - t0)
    return best


def make(model):
    s = Shell()
    s.m = 12
    s.n = 12
    s.stack = [0, 90, -45, +45, +45, -45, 90, 0]
    s.plyt = 0.125e-3
    s.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    s.rho = 1600.
    s.model = model
    s.a = 2.
    s.b = 1.
    s.r = 1. if model.startswith('cyl') else 1.e8
    s.Nxx = -1
    return s


def main():
    pin_process()
    warnings.simplefilter('ignore')
    results = {'panels': panels.__file__, 'cases': {}}
    for model in MODELS:
        s = make(model)
        s.calc_kC()
        c = 1e-6*np.random.default_rng(0).standard_normal(s.get_size())
        funcs = {
            'kC': lambda: s.calc_kC(),
            'kG': lambda: s.calc_kG(),
            'kM': lambda: s.calc_kM(),
            'kC(c)': lambda: s.calc_kC(c=c),
            'kG(c)': lambda: s.calc_kG(c=c),
            'fint(c)': lambda: s.calc_fint(c),
        }
        case = {name: timeit(f)*1e3 for name, f in funcs.items()}
        results['cases'][model] = case
        print('%-22s ' % model + '  '.join('%s %8.2f' % item
                                           for item in case.items())
              + '  [ms]')
        sys.stdout.flush()
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'w') as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
