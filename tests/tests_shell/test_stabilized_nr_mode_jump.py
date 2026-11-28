r"""Mode jump of a compressed plate passed by the stabilized Newton-Raphson method

A simply supported flat plate, ``a/b = 1.3``, under uniaxial compression,
buckles with one half-wave along the load and changes to two half-waves deep
in the post-buckling regime. The test verifies that the load-controlled
Newton-Raphson method of structsolve, ``NL_method='NR'``, stops at the load
where the one-half-wave pattern ends, whereas ``NL_method='stabilized_NR'``
passes it and reaches the final load with the two-half-wave pattern; that
every state it accepts is an equilibrium, ``'undamped'`` or ``'corrected'``
in ``Analysis.stabilization_status``; and that its path before the jump is
that of ``'NR'``.

Units: N, mm, MPa.

Notes
-----
The plate, isotropic, ``E = 70000``, ``nu = 0.3``, ``h = 1``, ``a = 130``,
``b = 100``, is simply supported at its four edges, with the in-plane
displacement normal to the edge restrained at ``x = 0``, ``u = 0``, and at
``y = 0`` and ``y = b``, ``v = 0``. The compression `N_{xx}` is applied at
``x = a``. The restrained Poisson expansion gives
the pre-buckling state `N_{yy} = \nu N_{xx}`, `N_{xy} = 0`, and the
critical load of a pattern with `m` half-waves along `x` and one along `y`
is

.. math::

    N_{xx,cr} = \frac{\pi^2 D}{b^2}
    \frac{\left((m b/a)^2 + 1\right)^2}{(m b/a)^2 + \nu},
    \qquad D = \frac{E h^3}{12 (1 - \nu^2)}

which follows from the equilibrium of the buckled plate with the pre-buckling
stresses, the deflection being `\sin(m \pi x/a) \sin(\pi y/b)`: 17.98 for
`m = 1` and 26.89 for `m = 2`. The model, 12 x 10 terms of the
``plate_clpt_donnell`` model, gives both to a relative difference below
1e-10.

The reference load is `N_{xx} = 4 N_0`, `N_0 = 4 \pi^2 D/b^2 = 25.31`, the
critical load of a long plate with unloaded edges free to move in-plane,
i.e. 5.63 times the critical load of this plate. A lateral point load at
``x = 0.49 a``, ``y = b/2``, proportional to the compression, triggers the
buckling and makes the one-half-wave pattern unsymmetric: near the limit
point below, its deflection, of one sign, has two crests of different
heights.

In the post-buckling regime, the one-half-wave equilibrium path has a limit
point at about 2.6 `N_0` (load factor 0.654): there is no equilibrium with
one half-wave beyond it, and under load control the solution must jump to the
two-half-wave pattern. The Newton-Raphson method cuts the increments down to
``minInc`` there and stops. The stabilized method solves the first increment
beyond the limit point with viscous forces and corrects the damped state to
the undamped equilibrium, ``'corrected'``; the plain iterations converge
again from there up to the reference load. The path depends on the
stabilization factor and the increments, see the notebook
``ex_stabilized_newton_raphson``: with ``stabilization_factor = 1.e-4``
instead of the default ``1.e-5`` the increment of the jump is cut and solved
by the plain iterations, and the state at the reference load is the same to
the round-off. The largest deflection on the line ``y = b/2`` at the
reference load, 3.132, is a value of the code (12 x 10 terms); with 16 x 14
terms it is 3.145, and the limit point is at the load factor 0.652 instead
of 0.654.

"""
import numpy as np
import pytest
from structsolve import Analysis, lb, solve
from structsolve.convergence import crisfield_weights, crisfield_zeta

from panels.shell import Shell


E = 70e3
NU = 0.3
H = 1.
A = 130.
B = 100.
D = E*H**3/(12*(1 - NU**2))
N0 = 4*np.pi**2*D/B**2


def plate(Nxx, perturbation=True):
    s = Shell()
    s.model = 'plate_clpt_donnell'
    s.a = A
    s.b = B
    s.stack = [0.]
    s.plyt = H
    s.laminaprop = (E, NU)
    s.m = 12
    s.n = 10
    # simply supported; u = 0 at x = 0, v = 0 at y = 0 and y = b
    s.x1u = 0; s.x1ur = 1; s.x2u = 1; s.x2ur = 1
    s.x1v = 1; s.x1vr = 1; s.x2v = 1; s.x2vr = 1
    s.x1w = 0; s.x1wr = 1; s.x2w = 0; s.x2wr = 1
    s.y1u = 1; s.y1ur = 1; s.y2u = 1; s.y2ur = 1
    s.y1v = 0; s.y1vr = 1; s.y2v = 0; s.y2vr = 1
    s.y1w = 0; s.y1wr = 1; s.y2w = 0; s.y2wr = 1
    s.add_distr_load_fixed_x(s.a, funcx=lambda y: -Nxx, funcy=None,
                             funcz=None, cte=False)
    if perturbation:
        s.add_point_load(0.49*s.a, s.b/2, 0, 0, 3e-4*Nxx*s.b, cte=False)
    return s


def run(s, NL_method, stabilization_factor=1.e-5):
    an = Analysis(s.calc_fext, s.calc_fint, s.calc_kC, s.calc_kG)
    an.NL_method = NL_method
    an.initialInc = 0.02
    an.maxInc = 0.02
    an.stabilization_factor = stabilization_factor
    an.static(NLgeom=True, silent=True)
    return an


def half_waves(s, c):
    # 1 or 2 half-waves along x, from the deflections at x = a/4 and x = 3a/4
    # on the line y = b/2
    w = s.uvw(c=c, gridx=5, gridy=3)[1]['w'][1]
    return 1 if w[1]*w[3] > 0 else 2


def test_linear_buckling():
    s = plate(1., perturbation=False)
    kC = s.calc_kC()
    c0 = solve(kC, s.calc_fext(), silent=True)
    eigvals, eigvecs = lb(kC, s.calc_kG(c=c0), silent=True, num_eigvalues=2)
    for k, m in enumerate((1, 2)):
        r = (m*B/A)**2
        ref = np.pi**2*D/B**2*(r + 1)**2/(r + NU)
        print('m = %d, critical load %.4f, closed form %.4f'
              % (m, eigvals[k], ref))
        assert np.isclose(eigvals[k], ref, rtol=1e-6)
        assert half_waves(s, eigvecs[:, k]) == m


@pytest.mark.slow
def test_stabilized_nr_mode_jump():
    Nxx = 4*N0
    s = plate(Nxx)
    fext = s.calc_fext()

    nr = run(s, 'NR')
    print('NR stops at load factor %.5f' % nr.increments[-1])
    assert 0.650 < nr.increments[-1] < 0.656
    assert half_waves(s, nr.cs[-1]) == 1

    st = run(s, 'stabilized_NR')
    status = st.stabilization_status
    for lbd, c, stat in zip(st.increments, st.cs, status):
        print('  load factor %.5f, %-9s, %d half-wave(s)'
              % (lbd, stat, half_waves(s, c)))
    assert np.isclose(st.increments[-1], 1.)
    assert set(status) <= {'undamped', 'corrected'}
    assert 'corrected' in status
    assert all(r is None for r in st.damped_residuals)

    # every accepted state is an equilibrium without viscous forces, to the
    # tolerance of the convergence criterion of structsolve
    w = crisfield_weights(s.calc_kC())
    for lbd, c in zip(st.increments, st.cs):
        fint = s.calc_fint(c)
        zeta = crisfield_zeta(fint - lbd*fext, lbd*fext, fint, w)
        assert zeta <= st.relTOL

    # before the jump the path is that of NR, same increments, same states
    ijump = status.index('corrected')
    assert st.increments[ijump - 1] < nr.increments[-1] < st.increments[ijump]
    lbd_nr = np.asarray(nr.increments)
    for lbd, c in zip(st.increments[:ijump], st.cs[:ijump]):
        i = np.flatnonzero(np.isclose(lbd_nr, lbd, rtol=0, atol=1e-12))
        assert i.size == 1
        assert np.allclose(c, nr.cs[i[0]], rtol=0,
                           atol=1e-8*np.abs(c).max())

    # one half-wave up to the jump, two after it
    assert all(half_waves(s, c) == 1 for c in st.cs[2:ijump])
    assert all(half_waves(s, c) == 2 for c in st.cs[ijump:])
    # largest deflection on the line y = b/2
    wmax = np.abs(s.uvw(c=st.cs[-1], gridx=301, gridy=3)[1]['w'][1]).max()
    print('largest deflection at the reference load %.4f' % wmax)
    # value of the code, 12 x 10 terms
    assert np.isclose(wmax, 3.132, rtol=1e-3)


if __name__ == '__main__':
    test_linear_buckling()
    test_stabilized_nr_mode_jump()
