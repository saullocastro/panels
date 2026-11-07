r"""Web of a ring frame connected to a cylinder with ``'BFxcte'``

An annular web, model ``'frame_web_clpt_donnell'``, connected at the end
`x = 0` of a cylinder clamped there: the connection clamps the outer edge of
the web, whose frequencies must be those of the annulus clamped outside and
free inside, from the frequency equation of Leissa (1969), with the penalty
and with the null-space connections.

Notes
-----
With ``dict(p1=cylinder, p2=web, func='BFxcte', xcte1=x_f, xcte2=0)`` the
connection imposes `u_{cyl} = w_{web}`, `v_{cyl} = v_{web}`, `w_{cyl} =
-u_{web}` and the continuity of the rotation about the circumferential
axis, `w_{cyl,x} = w_{web,x}`, at the station `x_f`, see
:mod:`panels.models.frame_web_clpt_donnell_num`. At a clamped end of the
cylinder all of them vanish, which clamps the outer edge of the web through
the connection alone (the web itself has free edges). The reference values
are the roots of the frequency equation of the annulus, from the
Bessel-function solution of Leissa [1]_, Eq. (1.17), with the edge
conditions of his Eqs. (1.11)-(1.13), computed with SciPy (see
``tests/tests_shell/test_frame_web_clpt_donnell.py``): 13.5004, 19.3892 and
31.3384 for `n = 0, 1, 2`, `
u = 1/3`, `eta = 0.4`; Leissa's Table 2.21
gives 13.54, 19.80 and 31.34. The null-space connection reproduces them to
the digits given; the penalty connection, with the default penalty
constants, gives values 0.3 % lower. The cylinder, two
half cylinders of length `0.5 r` clamped at both ends, has its lowest
frequencies well above those of the web, and the lowest modes of the
assembly are those of the web: `\lambda^2 = \omega r^2 \sqrt{\rho_m h/D}`
for `n = 0, 1, 1, 2, 2` nodal diameters against Leissa [1]_, Table 2.21,
`\nu = 1/3`.

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, 1969, Eqs.
   (1.11)-(1.13) and (1.17), p. 2, and Table 2.21, p. 23.

"""
import numpy as np
import pytest
from structsolve import freq

from panels.shell import Shell
from panels.multidomain import MultiDomain


E, NU, H, RHO = 70e9, 1/3, 1e-3, 2700.
R = 1.
D = E*H**3/(12*(1 - NU**2))
#: roots of the frequency equation of Leissa (1969), Eqs. (1.11)-(1.13) and
#: (1.17), beta = 0.4, nu = 1/3, n = 0, 1, 2
EXACT = (13.5004, 19.3892, 31.3384)
RTOL = {'null-space': 1e-5, 'penalty': 5e-3}


@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_web_clamped_by_cylinder_end(conn_method):
    """Web at the clamped end of a cylinder against the roots of the
    frequency equation of Leissa (1969), `\\beta = 0.4`"""
    beta = 0.4
    cyls, webs = [], []
    for k in range(2):
        c = Shell(model='cylshell_clpt_sanders', a=0.5*R, b=np.pi*R, r=R,
                  m=8, n=12, plyt=H, stack=[0], laminaprop=(E, NU), rho=RHO)
        w = Shell(model='frame_web_clpt_donnell', a=R*(1 - beta), b=np.pi*R,
                  r=R, m=10, n=12, plyt=H, stack=[0], laminaprop=(E, NU),
                  rho=RHO)
        for f in 'uvw':
            for e in ('x1', 'x2'):
                setattr(c, e + f, 0)
                setattr(c, e + f + 'r', 0)
            for e in ('x1', 'x2', 'y1', 'y2'):
                setattr(w, e + f, 1)
                setattr(w, e + f + 'r', 1)
            for e in ('y1', 'y2'):
                setattr(c, e + f, 1)
                setattr(c, e + f + 'r', 1)
        cyls.append(c)
        webs.append(w)
    conn = []
    for p1, p2 in (cyls, webs):
        conn += [dict(p1=p1, p2=p2, func='SSycte', ycte1=p1.b, ycte2=0),
                 dict(p1=p1, p2=p2, func='SSycte', ycte1=0, ycte2=p2.b)]
    for c, w in zip(cyls, webs):
        conn.append(dict(p1=c, p2=w, func='BFxcte', xcte1=0, xcte2=0))
    md = MultiDomain(cyls + webs, conn, conn_method=conn_method)
    K = md.reduce(md.calc_kC(silent=True))
    M = md.reduce(md.calc_kM(silent=True))
    eigvals, _ = freq(K, M, sparse_solver=False, silent=True, num_eigvalues=8)
    l2 = np.sort(np.sqrt(-eigvals.real))*R**2*np.sqrt(RHO*H/D)
    ref = np.array(EXACT)[[0, 1, 1, 2, 2]]
    assert np.allclose(l2[:5], ref, rtol=RTOL[conn_method])
