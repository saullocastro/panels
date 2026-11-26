r"""Tables of the approximation functions kept on the shell across calls

The kernels of ``panels/models/<model>_num.pyx`` store their tables on the
shell, see :mod:`panels.basis_tables`, with the key of all the values they
depend on. The checks, against a new shell with the same data, whose
kernels fill new tables:

1. after a change of the numbers of terms ``m`` and ``n``, of a boundary
   flag, of the limits of the integration domain, of the numbers of
   integration points ``nx`` and ``ny``, or of the dimension ``a``, the
   outputs of a shell whose tables were filled before the change are equal,
   bit for bit, to those of the new shell;
2. a shell keeps one table per kernel and table name, replaced, not
   modified, when its key changes, such that a table read by a kernel
   running in another thread is never modified;
3. :func:`panels.basis_tables.clear_tables` empties the tables, and the
   outputs filled again are the same.

"""
import numpy as np
import pytest

import kernel_references as kr
from panels.basis_tables import clear_tables


def set_m(s):
    s.m = kr.M + 2


def set_n(s):
    s.n = kr.N + 1


def set_flag_x(s):
    s.x2u = 1. - s.x2u


def set_flag_y(s):
    s.y1wr = 1. - s.y1wr


def set_limits_x(s):
    s.x1, s.x2 = 40., 120.


def set_limits_y(s):
    s.y1, s.y2 = 50., 150.


def set_nx(s):
    s.nx = kr.NX + 2


def set_ny(s):
    s.ny = kr.NY + 3


def set_a(s):
    s.a = 230.


CHANGES = [set_m, set_n, set_flag_x, set_flag_y, set_limits_x, set_limits_y,
           set_nx, set_ny, set_a]
MODELS = ['plate_clpt_donnell', 'plate_fsdt_donnell', 'cylshell_tsdt_gl',
          'annular_clpt_tl']


def assert_same(out, ref):
    assert sorted(out) == sorted(ref)
    for k in out:
        assert out[k].shape == ref[k].shape, k
        assert np.array_equal(out[k], ref[k]), k


@pytest.mark.parametrize('change', CHANGES, ids=lambda c: c.__name__)
@pytest.mark.parametrize('model', MODELS)
def test_invalidation(model, change):
    """Outputs after a change of the data of a shell with stored tables,
    equal to those of a new shell"""
    s = kr.make_shell(model)
    first = kr.compute(model, shell=s)
    before = {k: v[1] for k, v in s._basis_tables.items()}
    change(s)
    out = kr.compute(model, shell=s)
    fresh = kr.make_shell(model)
    change(fresh)
    assert_same(out, kr.compute(model, shell=fresh))
    # the change changes the outputs
    assert any(k not in out or out[k].shape != first[k].shape
               or not np.array_equal(out[k], first[k]) for k in first)
    # one table per kernel and table name, the old ones replaced; the mass
    # matrix of a partial domain is integrated numerically
    assert set(before) <= set(s._basis_tables)
    assert set(s._basis_tables) == set(fresh._basis_tables)
    assert any(v[1] is not before.get(k) for k, v in s._basis_tables.items())


def test_stored_tables_not_modified():
    """A table replaced after a change keeps its values"""
    model = 'plate_clpt_donnell'
    s = kr.make_shell(model)
    kr.compute(model, shell=s)
    before = {k: (v[1], v[1].copy()) for k, v in s._basis_tables.items()}
    set_limits_x(s)
    kr.compute(model, shell=s)
    for k, (tab, values) in before.items():
        assert np.array_equal(tab, values), k


def test_clear_tables():
    """Outputs after clear_tables equal to the first ones"""
    model = 'cylshell_clpt_sanders'
    s = kr.make_shell(model)
    out = kr.compute(model, shell=s)
    assert s._basis_tables
    clear_tables(s)
    assert s._basis_tables is None
    assert_same(kr.compute(model, shell=s), out)
    assert s._basis_tables
