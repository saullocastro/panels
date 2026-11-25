r"""The numerical kernels of every model against the stored outputs of 0.17.0

The kernels ``fkC_num`` (with and without ``NLgeom``), ``fkG_num``,
``fkM_num``, ``fkAx_num``, ``fkAy_num``, ``calc_fint``, ``calc_kTv``,
``calc_fext_follower`` and ``fkCfollower_num`` of the 30 modules
``panels/models/<model>_num.pyx`` keep their tables of the approximation
functions on the shell across calls, see :mod:`panels.basis_tables`. Their
outputs, for a small shell of each model, with the full and a partial
integration domain, see ``kernel_references.py``, are compared with those of
the kernels of 0.17.0 (commit 25d1ecb), which filled the tables at every
call, stored in ``kernel_references.npz``:

1. at the first call, which fills the tables, and at the second call on the
   same shell, which reads them, the outputs are equal to the references
   bit for bit.

The references fix the values of the approximation functions of 0.17.0. A
change of the evaluation of the functions that is exact only to round-off,
e.g. a Horner scheme for the polynomials of
``panels/core/src/bardell_functions.cpp``, keeps this test meaningful by
comparing to a relative tolerance of about ``1e-12`` instead of bit for bit.

"""
import numpy as np
import pytest

import kernel_references as kr


@pytest.fixture(scope='module')
def references():
    with np.load(kr.PATH) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('model', kr.MODELS)
def test_equal_to_references(model, partial, references):
    """Outputs of the first and second call on a shell, bit for bit equal to
    those of 0.17.0"""
    s = kr.make_shell(model, partial=partial)
    for call in ('first', 'second'):
        out = kr.compute(model, shell=s, partial=partial)
        expected = [k for k in references
                    if k.startswith(model + ('/partial/' if partial else '/'))
                    and (partial or not k.startswith(model + '/partial/'))]
        assert sorted(out) == sorted(expected)
        for k, v in out.items():
            ref = references[k]
            assert v.shape == ref.shape, k
            assert np.array_equal(v, ref), '%s, %s call' % (k, call)
    # every kernel with tables keeps them on the shell
    assert s._basis_tables
