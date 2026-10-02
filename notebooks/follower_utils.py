r"""Utilities of the follower (hydrostatic) pressure validation notebooks

The models, the closed-form solutions and the data of the references are
those of the validation tests,
``tests/tests_shell/test_follower_pressure_validation.py``, imported here,
such that the notebooks and the tests use the same definitions. The notebooks
are organized per reference:

- ``han2004_sandwich_rings.ipynb``
- ``kardomateas2003_sandwich_rings.ipynb``
- ``kardomateas1993_thick_rings.ipynb``
- ``schweizerhof1984_pressure_loads.ipynb``
- ``nasa_sp8007_external_pressure.ipynb``

The results of panels are saved in ``results/<name>.json`` and loaded unless
``rerun=True``, see :func:`run_or_load`.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.abspath(os.path.join(HERE, '..', 'tests', 'tests_shell'))
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)

import test_follower_pressure_validation as val  # noqa: E402


def run_or_load(path, func, rerun=False):
    r"""Loads the JSON results in ``path``, or computes them with ``func()``,
    which returns a JSON-serializable object, and saves them"""
    if os.path.exists(path) and not rerun:
        with open(path) as f:
            return json.load(f)
    t0 = time.time()
    res = func()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(res, f, indent=1)
    print('computed in %.0f s, saved to %s' % (time.time() - t0, path))
    return res


def pct(value, ref):
    r"""Relative difference in percent, ``100 (value/ref - 1)``"""
    if value is None or ref is None:
        return None
    return 100*(value/ref - 1)


def fmt(value, digits=0, percent=None):
    r"""``value`` formatted with ``digits`` decimals, followed by the
    relative difference ``percent`` when given"""
    if value is None:
        return '-'
    out = '{0:,.{1}f}'.format(value, digits)
    if percent is not None:
        out += ' ({0:+.1f}%)'.format(percent)
    return out


def markdown_table(header, rows):
    r"""Markdown table from a list of column names and a list of rows of
    strings"""
    lines = ['| ' + ' | '.join(header) + ' |',
             '|' + '---|'*len(header)]
    for row in rows:
        lines.append('| ' + ' | '.join(str(v) for v in row) + ' |')
    return '\n'.join(lines)


#: shear corrections of the FSDT compared in the sandwich notebooks
FSDT_CORRECTIONS = {'rohwer': 'rohwer', 'vlachoutsis': 'vlachoutsis',
                    '5/6': 5/6}


def sandwich_ring_case(face, ratio, f=None, c=None):
    r"""Critical pressures (Pa) of the sandwich ring of Han et al. (2004) or
    Kardomateas and Simitses (2003) with the faces ``face`` and ``R0/h =
    ratio``, for the models of the notebooks:

    - ``'CLPT'``: the classical laminated plate theory;
    - ``'FSDT core'``, ``'FSDT Gbar'``: the FSDT with the transverse shear
      stiffness of the shell formulas, Eqs. (14a) and (14c) of Han et al.;
    - ``'FSDT <correction>'``: the FSDT with the shear corrections of
      :data:`FSDT_CORRECTIONS`;
    - ``'TSDT'``: the third-order theory;

    with the pressure on the mid-surface, and with the suffix ``', zp =
    h/2'`` on the outer surface.
    """
    f = val.F if f is None else f
    c = val.C if c is None else c
    h = 2*f + c
    out = {}
    s = val.sandwich_ring(face, ratio, model='cylshell_clpt_sanders', f=f,
                          c=c)
    out['CLPT'] = val.ring_buckling_pressure(s)
    for kind in ('core', 'Gbar'):
        s = val.sandwich_ring(face, ratio, f=f, c=c,
                              shear=val.han_shear_stiffness(face, kind, f, c))
        out['FSDT ' + kind] = val.ring_buckling_pressure(s)
    for label, k in FSDT_CORRECTIONS.items():
        s = val.sandwich_ring(face, ratio, f=f, c=c)
        s.fsdt_shear_correction = k
        s._rebuild()
        out['FSDT %s' % label] = val.ring_buckling_pressure(s)
        out['FSDT %s, zp = h/2' % label] = val.ring_buckling_pressure(s,
                                                                      zp=h/2)
    s = val.sandwich_ring(face, ratio, model='cylshell_tsdt_sanders', f=f,
                          c=c)
    out['TSDT'] = val.ring_buckling_pressure(s)
    out['TSDT, zp = h/2'] = val.ring_buckling_pressure(s, zp=h/2)
    return out
