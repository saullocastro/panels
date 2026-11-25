r"""Reference outputs of the numerically integrated kernels of every model

:func:`compute` evaluates every kernel of ``panels/models/<model>_num.pyx``
that tabulates the approximation functions, for a small shell of each of
the 30 models, and returns the matrices, as dense arrays, and the vectors.
Run as a script, it stores them in ``kernel_references.npz``, next
to this file::

    python tests/tests_shell/kernel_references.py

It was run with the commit 25d1ecb of ``main`` (the kernels of 0.17.0),
whose kernels fill the tables of the functions at every call, see
``test_kernel_references.py``.

"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, 'kernel_references.npz')

MODELS = [geo + '_' + th + '_' + kin
          for geo in ('plate', 'cylshell', 'annular')
          for th in ('clpt', 'fsdt', 'tsdt')
          for kin in ('donnell', 'gl', 'tl')
          ] + ['cylshell_clpt_sanders', 'cylshell_fsdt_sanders',
               'cylshell_tsdt_sanders']

#: number of terms and of integration points, different along x and y
M, N, NX, NY = 3, 4, 5, 6


def make_shell(model, m=M, n=N, partial=False):
    r"""A small laminated shell of the model ``model``, with boundary flags
    differing along each edge, and optionally a partial integration domain
    """
    from panels.shell import Shell
    geo = model.split('_')[0]
    r = {'plate': None, 'cylshell': 400., 'annular': 300.}[geo]
    s = Shell(model=model, a=200., b=150., r=r, m=m, n=n,
              stack=[0, 45, -45, 90, 30], plyt=0.125,
              laminaprop=(142.5e3, 8.7e3, 0.28, 5.1e3, 5.1e3, 3.4e3),
              rho=1.6e-9)
    s.x1w = 0.; s.x1wr = 1.
    s.y2u = 0.; s.y2v = 0.
    s.x2wr = 0.
    s.y1phix = 0.
    s.x2phiy = 0.
    s.beta = 2.5
    s.gamma = 0.3
    s.nx = NX
    s.ny = NY
    if partial:
        s.x1, s.x2 = 30., 170.
        s.y1, s.y2 = 0., 90.
    return s


def _dense(k, size):
    return np.asarray(k.toarray() if hasattr(k, 'toarray') else k,
                      dtype=np.float64).reshape(size, -1)


def compute(model, shell=None, partial=False):
    r"""The outputs of the kernels of the model ``model``

    Parameters
    ----------
    model : str
        The name of the model.
    shell : :class:`.Shell`, optional
        The shell, by default a new one of :func:`make_shell`; a shell given
        twice gives the outputs read from the tables kept on it.
    partial : bool, optional
        A partial integration domain, see :func:`make_shell`.

    Returns
    -------
    out : dict
        The outputs, with keys ``'<model>/<kernel>[/<variant>]'``.

    """
    import importlib
    num = importlib.import_module('panels.models.%s_num' % model)
    s = make_shell(model, partial=partial) if shell is None else shell
    size = s.get_size()
    F, nx, ny = s._get_ABDnxny(None, None, None)
    rng = np.random.default_rng(1)
    c = 1e-2*rng.standard_normal(size)
    v = rng.standard_normal(size)
    pre = model + ('/partial' if partial else '') + '/'
    out = {}
    out[pre + 'fkC_num'] = _dense(num.fkC_num(c, F, s, size, 0, 0, nx, ny,
                                              NLgeom=0), size)
    out[pre + 'fkC_num/NLgeom'] = _dense(num.fkC_num(c, F, s, size, 0, 0, nx,
                                                     ny, NLgeom=1), size)
    out[pre + 'fkG_num'] = _dense(num.fkG_num(c, F, s, size, 0, 0, nx, ny,
                                              -1.5, 0.7, 0.3), size)
    out[pre + 'calc_fint'] = np.asarray(num.calc_fint(c, F, s, size, 0, nx,
                                                      ny))
    out[pre + 'calc_kTv'] = np.asarray(num.calc_kTv(c, v, F, s, size, 0, nx,
                                                    ny, -1.5, 0.7, 0.3))
    if partial:
        return out
    out[pre + 'fkM_num'] = _dense(s.calc_kM(silent=True, finalize=False),
                                  size)
    if not model.startswith('annular'):
        out[pre + 'fkAx_num'] = _dense(num.fkAx_num(s, size, 0, 0, nx, ny),
                                       size)
        out[pre + 'fkAy_num'] = _dense(num.fkAy_num(s, size, 0, 0, nx, ny),
                                       size)
    if hasattr(num, 'calc_fext_follower'):
        p = 0.1 + rng.random((4, 3))
        for q in (0, 1):
            out[pre + 'calc_fext_follower/%d' % q] = np.asarray(
                num.calc_fext_follower(c, p, 20., 150., 10., 140., s, size, 0,
                                       quadratic=q, reference=1 - q))
            out[pre + 'fkCfollower_num/%d' % q] = _dense(
                num.fkCfollower_num(c, p, 20., 150., 10., 140., s, size, 0, 0,
                                    quadratic=q), size)
    return out


def compute_all():
    r"""The outputs of :func:`compute` of every model, with the full and a
    partial integration domain"""
    out = {}
    for model in MODELS:
        out.update(compute(model))
        out.update(compute(model, partial=True))
    return out


if __name__ == '__main__':
    import panels
    out = compute_all()
    np.savez_compressed(PATH, **out)
    print('panels', panels.__version__, 'from', panels.__file__)
    print(len(out), 'outputs in', PATH, os.path.getsize(PATH), 'bytes')
