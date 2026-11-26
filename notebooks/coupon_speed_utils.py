r"""Helpers of the notebook ``coupon_speed_options``

The options ``plate_kernels`` and ``reuse_kT`` are passed only when
requested, such that the same code measures any checkout of ``panels``. Run
as a script, the functions use the ``panels`` found first on ``sys.path``,
e.g. a checkout given by ``PYTHONPATH``, and print the results as JSON on
the last line::

    python coupon_speed_utils.py domain <model> <out.npz> [repeat]
    python coupon_speed_utils.py dcb

"""
import json
import sys
import time
import warnings

import numpy as np

from panels.shell import Shell
from panels.multidomain import MultiDomain
from panels.multidomain.delamination import create_coupon, solve_coupon
from panels.multidomain.connections.czm import BilinearLaw

#: AS4/PEEK, Camanho et al. (2003), Table 1: E11, E22, nu12, G12, G13, G23 (MPa)
LAM = (122700., 10100., 0.25, 5500., 5500., 3700.)


def one_domain(model, m=15, n=6, nx=60, ny=10, plate_kernels=None):
    r"""``MultiDomain`` of one domain of `20 \times 25.4` mm, `[0]_{12}`
    AS4/PEEK, with ``m`` by ``n`` terms and ``nx`` by ``ny`` Gauss-Legendre
    points; ``plate_kernels`` is passed only when given"""
    p = Shell(model=model, a=20., b=25.4, m=m, n=n, stack=[0]*12, plyt=0.13,
              laminaprop=LAM)
    p.nx, p.ny = nx, ny
    kw = {} if plate_kernels is None else dict(plate_kernels=plate_kernels)
    return MultiDomain([p], conn=[], **kw)


def tangent_and_fint(md, c):
    """``calc_kC`` and ``calc_kG`` with ``NLgeom=True`` and ``calc_fint`` at
    ``c``"""
    return (md.calc_kC(c=c, NLgeom=True, silent=True),
            md.calc_kG(c=c, NLgeom=True, silent=True),
            md.calc_fint(c))


def time_coupon_domain(model, repeat=9):
    r"""Times of :func:`tangent_and_fint` for a coupon-sized domain, `m =
    24`, `n = 4` terms and `240 \times 6` points, at a random configuration

    Returns
    -------
    times : dict
        ``'first'``: median of the first call on ``repeat`` new domains, which
        fills the tables of the approximation functions; ``'later'``: median
        of ``repeat`` later calls on the same domain (s).
    arrays : list of arrays
        `K_C`, `K_G` (dense) and `F_{int}` of the first call.

    """
    md = one_domain(model, m=24, n=4, nx=240, ny=6)
    c = 1e-2*np.random.default_rng(0).standard_normal(md.get_size())
    kC, kG, fint = tangent_and_fint(md, c)
    later = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        tangent_and_fint(md, c)
        later.append(time.perf_counter() - t0)
    first = []
    for _ in range(repeat):
        md_new = one_domain(model, m=24, n=4, nx=240, ny=6)
        t0 = time.perf_counter()
        tangent_and_fint(md_new, c)
        first.append(time.perf_counter() - t0)
    times = dict(first=float(np.median(first)), later=float(np.median(later)))
    return times, [kC.toarray(), kG.toarray(), np.asarray(fint)]


def dcb(on=False, **analysis_kw):
    r"""DCB of AS4/PEEK up to an opening of 3 mm

    Arms of `[0]_{12}`, 0.13 mm plies, 25.4 mm wide, `a_0 = 30` mm, 60 mm
    long, bilinear law, CLPT, three cohesive domains of 15 terms over 15 mm,
    increments of 0.25 mm. With ``on=True``, ``plate_kernels='blas'`` and
    ``reuse_kT=True``; ``analysis_kw`` goes to ``solve_coupon``.

    Returns
    -------
    res : dict
        The results of ``solve_coupon``, with ``'time'`` (s) and
        ``'iterations'``, the evaluations of the tangent per increment.

    """
    law = BilinearLaw(0.969, 1.719, 80., 100., 1e5, 2.284)
    kw = dict(plate_kernels='blas') if on else {}
    md, info = create_coupon('DCB', LAM, 0.13, [0]*12, 25.4, 30., [law],
                             x_end=60., nd=3, L_coh=15., m_crack=10,
                             m_coh=15, m_far=10, n=6, nr_y_gauss=6, **kw)
    calls = []
    calc_kC = md.calc_kC

    def counted(*args, **kwargs):
        calls.append(1)
        return calc_kC(*args, **kwargs)
    md.calc_kC = counted
    ends = []
    if on:
        analysis_kw = dict(analysis_kw, reuse_kT=True)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res, an = solve_coupon(md, info, 3., initialInc=0.25, maxInc=0.25,
                               record=lambda an, c, row: ends.append(len(calls)),
                               **analysis_kw)
    res['time'] = time.perf_counter() - t0
    res['iterations'] = np.diff([0] + ends)
    return res


if __name__ == '__main__':
    if sys.argv[1] == 'domain':
        repeat = int(sys.argv[4]) if len(sys.argv) > 4 else 9
        times, arrays = time_coupon_domain(sys.argv[2], repeat=repeat)
        np.savez(sys.argv[3], *arrays)
        print(json.dumps(times))
    elif sys.argv[1] == 'dcb':
        res = dcb(False)
        print(json.dumps(dict(delta=np.asarray(res['delta']).tolist(),
                              P=np.asarray(res['P']).tolist(),
                              iterations=res['iterations'].tolist(),
                              max_d=np.asarray(res['max_d']).tolist(),
                              time=res['time'])))
