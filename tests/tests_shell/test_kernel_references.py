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

1. at the first call, which fills the tables, the outputs agree with the
   references to ``1e-12`` of the largest absolute entry of each output,
   with an absolute floor of ``1e-15`` for the outputs that vanish in exact
   arithmetic (``fkAx_num`` of the plates, entries of about ``1e-17``);
2. at the second call on the same shell, which reads the tables, the
   outputs are equal to those of the first call bit for bit.

The references fix the values of the approximation functions of 0.17.0,
which summed the monomials of Bardell's polynomials with ``pow()``; since
then ``panels/core/src/bardell_functions.cpp`` evaluates them by Horner's
rule, which changes the outputs at round-off only (largest relative
difference about ``1.3e-15``, of ``fkG_num``), hence the tolerance of item 1.

"""
import numpy as np
import pytest

import kernel_references as kr

RTOL = 1e-12
ATOL = 1e-15


@pytest.fixture(scope='module')
def references():
    with np.load(kr.PATH) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('model', kr.MODELS)
def test_equal_to_references(model, partial, references):
    """Outputs of the first call on a shell equal to those of 0.17.0 to
    ``1e-12`` of the largest entry; outputs of the second call equal to
    those of the first bit for bit"""
    s = kr.make_shell(model, partial=partial)
    expected = [k for k in references
                if k.startswith(model + ('/partial/' if partial else '/'))
                and (partial or not k.startswith(model + '/partial/'))]
    first = kr.compute(model, shell=s, partial=partial)
    assert sorted(first) == sorted(expected)
    for k, v in first.items():
        ref = references[k]
        assert v.shape == ref.shape, k
        tol = max(RTOL*np.abs(ref).max(), ATOL)
        assert np.abs(v - ref).max() <= tol, k
    second = kr.compute(model, shell=s, partial=partial)
    assert sorted(second) == sorted(first)
    for k, v in second.items():
        assert np.array_equal(v, first[k]), '%s, second call' % k
    # every kernel with tables keeps them on the shell
    assert s._basis_tables
