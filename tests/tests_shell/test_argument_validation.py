"""The argument-handling layer of the Shell stiffness API must not let bad
input reach the nogil kernels.

The kernels in ``panels/models/*_num.pyx`` declare their Ritz-constant vectors
as ``double [::1]`` and are compiled with ``boundscheck=False``, so anything
that is not a contiguous 1-D array of length ``size`` is read out of bounds
instead of raising. The same applies to the radius: the cylindrical kernels
divide by ``r`` and ``r*r``, so an unset radius silently becomes ``inf``.

These tests pin the validation, not the values, of the matrices.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest

from panels.shell import Shell


def make_shell(model='plate_clpt_donnell', r=None):
    s = Shell()
    s.model = model
    s.a = 0.3
    s.b = 0.2
    s.r = r
    s.stack = [30, -45, 0, 90]
    s.plyt = 0.125e-3
    E11 = 142.5e9
    E22 = E11/20
    G12 = 0.5*E22
    s.laminaprop = (E11, E22, 0.25, G12, G12, G12)
    s.m = 4
    s.n = 4
    s.nx = 6
    s.ny = 6
    s.Nxx = s.Nyy = s.Nxy = 0.
    s.Nxx_cte = s.Nyy_cte = s.Nxy_cte = 0.
    s._rebuild()
    return s


def test_calc_kC_ABDnxny_without_c():
    """calc_kC(ABDnxny=...) with no c must behave as c = 0

    Before the fix the zero-filled default was installed only when both c and
    ABDnxny were None, so np.ascontiguousarray(None) produced the shape (1,)
    array [nan], which fkC_num happily accepted and then indexed far past its
    end.
    """
    s = make_shell()
    size = s.get_size()
    ABDnxny = np.zeros((s.nx, s.ny, 6, 6))
    ABDnxny[:, :] = s.ABD

    for NLgeom in (False, True):
        implicit = make_shell().calc_kC(ABDnxny=ABDnxny, NLgeom=NLgeom)
        explicit = make_shell().calc_kC(ABDnxny=ABDnxny, NLgeom=NLgeom,
                                        c=np.zeros(size))
        assert implicit.shape == (size, size)
        assert np.allclose(implicit.toarray(), explicit.toarray(),
                           rtol=0, atol=0)


def test_calc_kC_cte_stress_without_c_cte():
    """A constant stress state with no c_cte must not reach fkG_num as None

    fkG_num takes a ``double [::1]``, and Cython dereferences a None
    memoryview without checking, which segfaulted the interpreter.
    """
    s = make_shell()
    s.Nxx_cte = -100.
    size = s.get_size()
    implicit = s.calc_kC(NLgeom=True)

    s2 = make_shell()
    s2.Nxx_cte = -100.
    explicit = s2.calc_kC(NLgeom=True, c_cte=np.zeros(size))

    assert implicit.shape == (size, size)
    assert np.allclose(implicit.toarray(), explicit.toarray(), rtol=0, atol=0)


def test_calc_fint_rejects_wrongly_sized_c():
    s = make_shell()
    size = s.get_size()
    # a good c still works
    fint = s.calc_fint(c=np.zeros(size))
    assert fint.shape == (size,)
    with pytest.raises(ValueError):
        s.calc_fint(c=np.zeros(size + 1))
    with pytest.raises(ValueError):
        s.calc_fint(c=None)


@pytest.mark.parametrize('bad_r', [None, 0., -0.4])
def test_cylshell_requires_positive_radius(bad_r):
    """An unset or non-positive radius must be caught at the API boundary

    The flat limit of a cylindrical shell is r -> infinity, not r = 0, so
    r = 0 is never a valid radius. Before the fix it reached the kernels,
    where every 1/r term became inf, and the failure surfaced as a bare
    AssertionError raised inside structsolve while finalizing the matrix.
    """
    s = make_shell(model='cylshell_clpt_donnell', r=0.4)
    size = s.get_size()
    s.r = bad_r
    for call in (lambda: s.calc_kC(),
                 lambda: s.calc_kG(),
                 lambda: s.calc_kT(c=np.zeros(size)),
                 lambda: s.calc_kM(),
                 lambda: s.calc_fint(c=np.zeros(size))):
        with pytest.raises(ValueError, match='requires Shell.r'):
            call()


def test_plate_still_accepts_unset_radius():
    """The plate kernels never read r, an unset r must keep working"""
    s = make_shell(model='plate_clpt_donnell', r=None)
    assert s.calc_kC().shape == (s.get_size(), s.get_size())
    assert s.r == 0.


def test_cylshell_approaches_plate_as_radius_grows():
    """r -> infinity, and not r = 0, is the flat-plate limit"""
    size = None
    plate = make_shell(model='plate_clpt_donnell', r=None)
    size = plate.get_size()
    c = np.linspace(0.1, 1., size)*1e-3
    kplate = plate.calc_kC(c=c, NLgeom=True).toarray()
    scale = np.abs(kplate).max()

    previous = None
    for r in (1., 1e4, 1e8):
        cyl = make_shell(model='cylshell_clpt_donnell', r=r)
        kcyl = cyl.calc_kC(c=c, NLgeom=True).toarray()
        rel = np.abs(kcyl - kplate).max()/scale
        if previous is not None:
            assert rel < previous
        previous = rel
    assert previous < 1e-8


def test_fkG_num_has_no_NLgeom_parameter():
    """fkG_num must not advertise an effect it does not have

    It used to accept an NLgeom argument that it never read, because the
    stress of the non-linear strain enters KT through KGNL in fkC_num. The
    argument is gone, so a caller that still expects KG(N0 + N_L + N_NL) now
    fails loudly instead of silently getting KG(N0 + N_L).
    """
    from panels import modelDB
    for model in ('plate_clpt_donnell', 'cylshell_clpt_donnell'):
        fkG_num = modelDB.db[model]['matrices_num'].fkG_num
        # the embedded signature is the first line of the docstring
        signature = fkG_num.__doc__.splitlines()[0]
        assert signature.startswith('fkG_num(')
        assert 'NLgeom' not in signature


def test_fkG_num_rejects_NLgeom():
    """Passing the removed argument must raise, not be silently swallowed"""
    from panels import modelDB
    s = make_shell()
    size = s.get_size()
    fkG_num = modelDB.db[s.model]['matrices_num'].fkG_num
    # the call without it still works
    fkG_num(np.zeros(size), s.ABD, s, size, 0, 0, s.nx, s.ny, -100., 0., 0.)
    with pytest.raises(TypeError):
        fkG_num(np.zeros(size), s.ABD, s, size, 0, 0, s.nx, s.ny, NLgeom=1)


def test_legacy_model_names_are_gone():
    """The _bardell aliases were dropped, an unknown model must raise"""
    from panels import modelDB
    for legacy in ('plate_clpt_donnell_bardell', 'cylshell_clpt_donnell_bardell'):
        assert legacy not in modelDB.db
    s = make_shell()
    s.model = 'plate_clpt_donnell_bardell'
    with pytest.raises(ValueError, match='valid models are'):
        s._rebuild()


def test_fkG_num_is_homogeneous_of_degree_one_in_c():
    """The property the ignored NLgeom protects, pinned directly"""
    s = make_shell()
    size = s.get_size()
    c = np.linspace(0.1, 1., size)*1e-3
    k1 = s.calc_kG(c=c, NLgeom=True).toarray()
    k2 = make_shell().calc_kG(c=2*c, NLgeom=True).toarray()
    assert np.allclose(k2, 2*k1, rtol=1e-10, atol=1e-12*np.abs(k1).max())
def test_NLgeom_and_not_c_selects_the_large_displacement_matrix():
    """calc_kC's docstring claim, pinned

    Giving c only switches calc_kC from the analytical closed form to the
    numerically integrated kernels. The large displacement terms K0L, KL0,
    KLL and KGNL are all built from w,x and w,y, which fkC_num zeroes unless
    NLgeom == 1, so c alone still yields plain K0.
    """
    s = make_shell()
    size = s.get_size()
    c = np.linspace(0.1, 1., size)*1e-3

    k0_analytical = make_shell().calc_kC().toarray()
    k0_at_zero_c = make_shell().calc_kC(c=np.zeros(size)).toarray()
    k_at_c = make_shell().calc_kC(c=c).toarray()
    k_at_c_NL = make_shell().calc_kC(c=c, NLgeom=True).toarray()

    scale = np.abs(k0_analytical).max()
    # c without NLgeom changes nothing but the quadrature
    assert np.array_equal(k_at_c, k0_at_zero_c)
    assert np.abs(k_at_c - k0_analytical).max()/scale < 1e-12
    # NLgeom is what adds the large displacement terms
    assert np.abs(k_at_c_NL - k_at_c).max()/scale > 1e-9


def test_KGNL_is_in_calc_kC_and_not_in_calc_kG():
    """The grouping documented in calc_kC, calc_kG and calc_kT

    calc_kG must stay homogeneous of degree one in c, so the geometric
    stiffness of the non-linear membrane stress has to sit in calc_kC.
    """
    s = make_shell()
    size = s.get_size()
    c = np.linspace(0.1, 1., size)*1e-3

    kG1 = make_shell().calc_kG(c=c, NLgeom=True).toarray()
    kG2 = make_shell().calc_kG(c=2*c, NLgeom=True).toarray()
    assert np.allclose(kG2, 2*kG1, rtol=1e-10, atol=1e-12*np.abs(kG1).max())

    # NLgeom does not change the value of kG, it only picks the quadrature
    kG_lin = make_shell().calc_kG(c=c, NLgeom=False).toarray()
    assert np.array_equal(kG_lin, kG1)

    # kT is exactly the sum of the two
    kT = make_shell().calc_kT(c=c).toarray()
    kC = make_shell().calc_kC(c=c, NLgeom=True).toarray()
    assert np.allclose(kT, kC + kG1, rtol=0, atol=1e-9*np.abs(kT).max())
