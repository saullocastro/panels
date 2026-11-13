r"""Prescribed displacement at a point

The deflection ``w`` at the centre of a simply supported isotropic
rectangular plate and of a cylindrical panel with a large radius is
prescribed with a penalty stiffness of the multi-domain connections,
``Shell.add_point_pd``; the linear static solution gives the prescribed
deflection at the point, and the force that it transmits equals the
concentrated load that produces this deflection according to Timoshenko and
Woinowsky-Krieger (1959), eq. (147) and Table 23.

Notes
-----
The force transmitted by the penalty spring at the point is obtained from
the strain energy of the plate, `U = \frac{1}{2} \boldsymbol{c}^\top
\boldsymbol{K}_C \boldsymbol{c} = \frac{1}{2} P w_c` (Clapeyron), with
`\boldsymbol{K}_C` without the penalty and `w_c` the deflection of the
centre. For a simply supported rectangular plate of sides `a \le b` under a
concentrated load `P` at the centre, eq. (147) of [1]_:

.. math::

    w_{max} = \frac{P a^2}{2 \pi^3 D} \sum_{m = 1, 3, 5, ...}
    \frac{1}{m^3} \left(\tanh \alpha_m - \frac{\alpha_m}{\cosh^2
    \alpha_m}\right) = \alpha \frac{P a^2}{D}, \qquad \alpha_m = \frac{m \pi
    b}{2 a}

summed in the test to `m = 199`: `\alpha = 0.016934` for `b/a = 3`, which
Table 23 of [1]_ gives as 0.01690. The reference is `P = w_c D/(\alpha
a^2)`. The Ritz solution converges to it from above, slowly because of the
concentrated load: `+9.4 \%`, `+4.1 \%`, `+2.2 \%`, `+1.4 \%`, `+0.97 \%` for
``m = n = 10, 14, 18, 22, 26``; with ``m = n = 22`` the tolerance is `2
\%`. The penalty constant ``kw = A11`` (:func:`.calc_ku_kv_kw_point_pd`)
gives `w_c/w_p - 1 = -5 \times 10^{-6}`; the tolerance is `10^{-4}`. The
cylinder has ``r = 1e9`` and gives the plate values.

References
----------
.. [1] Timoshenko, S., Woinowsky-Krieger, S., Theory of Plates and Shells,
   2nd ed., McGraw-Hill, New York, 1959.

"""
import sys
sys.path.append('../..')

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest
from structsolve import solve
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.multidomain.connections import calc_ku_kv_kw_point_pd
from panels.multidomain.connections import fkCpd
from panels.plot_shell import plot_shell


def alpha_central_load(a, b, mmax=199):
    """Factor of eq. (147) of Timoshenko and Woinowsky-Krieger (1959)"""
    alpham = np.arange(1, mmax + 1, 2)*np.pi*b/(2*a)
    m = np.arange(1, mmax + 1, 2)
    return np.sum((np.tanh(alpham) - alpham/np.cosh(alpham)**2)/m**3)/(2*np.pi**3)


@pytest.mark.slow
def test_point_pd_linear_static():
    m = 22
    n = 22
    for model in [
            'plate_clpt_donnell',
            'cylshell_clpt_donnell',
                  ]:
        print('Testing model: %s' % model)
        s = Shell()

        s.model = model
        s.a = 3.
        s.b = 1.
        s.r = 1.e9
        E = 70e9
        nu = 0.3
        thickness = 1e-3
        D = E*thickness**3/(12*(1 - nu**2))
        s.stack = [0]
        s.plyt = thickness
        s.laminaprop = (E, nu)
        s.m = m
        s.n = n

        kC = s.calc_kC()
        ku, kv, kw = calc_ku_kv_kw_point_pd(s)
        kCp = fkCpd(0., 0., kw, s, s.a/2, s.b/2, s.get_size(), 0, 0)
        kCp = finalize_symmetric_matrix(kCp)
        k0 = kC + kCp
        wp = 0.001
        s.add_point_pd(s.a/2, s.b/2, 0., 0., 0., 0., kw, wp)
        fext = s.calc_fext()
        c0 = solve(k0, fext)
        plot_shell(s, c0, vec='w', filename='test_point_load_pd.png', colorbar=True)

        s.uvw(c0, xs=np.array([s.a/2]), ys=np.array([s.b/2]))
        wc = s.fields['w'][0]
        assert np.isclose(wc, wp, rtol=1e-4)

        P = c0 @ (kC @ c0)/wc
        short, long = min(s.a, s.b), max(s.a, s.b)
        P_ref = wc*D/(alpha_central_load(short, long)*short**2)
        assert np.isclose(P, P_ref, rtol=0.02)


if __name__ == '__main__':
    test_point_pd_linear_static()
