r"""CLPT cylindrical shell with Sanders-Koiter kinematics

The coordinate `y = r \theta` is the arc length along the circumference,
such that `\frac{1}{r} (\cdot)_{,\theta} = (\cdot)_{,y}`, and the domain
`0 \le x \le a`, `0 \le y \le b` is mapped onto `-1 \le \xi, \eta \le +1`.

Following Sanders (1959, 1963), the mid-surface strains and the changes of
curvature are:

.. math::

    \varepsilon_{xx} = u_{,x} + \frac{1}{2} \phi_x^2

    \varepsilon_{yy} = v_{,y} + \frac{w}{r} + \frac{1}{2} \phi_y^2

    \gamma_{xy} = u_{,y} + v_{,x} + \phi_x \phi_y

    \kappa_{xx} = -w_{,xx}

    \kappa_{yy} = -w_{,yy} + \frac{1}{r} v_{,y}

    \kappa_{xy} = -2 w_{,xy} + \frac{3}{2} \frac{1}{r} v_{,x}
                  - \frac{1}{2} \frac{1}{r} u_{,y}

with the rotations of the normal

.. math::

    \phi_x = -w_{,x} \qquad \phi_y = -w_{,y} + \frac{v}{r}

Dropping the `v/r` term of `\phi_y` and the `1/r` terms of the changes of
curvature recovers the Donnell kinematics of ``cylshell_clpt_donnell``.

The CLPT displacement field with Sanders' correction, used for the mass
matrix, is:

.. math::

    u(z) = u + z \phi_x \qquad v(z) = v + z \phi_y \qquad w(z) = w

In the numerically integrated matrices, ``phix`` and ``phiy`` are the values
of `\phi_x` and `\phi_y` at the integration point, which vanish unless the
geometric non-linearity is considered.

"""
import os

import numpy as np
from sympy import Matrix as M, collect, expand, var

var('a, b, r, h, d, rho, phix, phiy')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('Nxx, Nyy, Nxy, Mxx, Myy, Mxy')

var('fAu, gAu, fAv, gAv, fAw, gAw, fAuxi, fAvxi, fAwxi, gAueta, gAveta, gAweta, fAwxixi, gAwetaeta')
var('fBu, gBu, fBv, gBv, fBw, gBw, fBuxi, fBvxi, fBwxi, gBueta, gBveta, gBweta, fBwxixi, gBwetaeta')

ABD = (A11, A12, A16, A22, A26, A66,
       B11, B12, B16, B22, B26, B66,
       D11, D12, D16, D22, D26, D66)

F = M([[A11, A12, A16, B11, B12, B16],
       [A12, A22, A26, B12, B22, B26],
       [A16, A26, A66, B16, B26, B66],
       [B11, B12, B16, D11, D12, D16],
       [B12, B22, B26, D12, D22, D26],
       [B16, B26, B66, D16, D26, D66]])


def shape_functions(side):
    r"""Shape functions of side ``'A'`` or ``'B'``, as rows of 3 DOFs"""
    s = globals()
    def f(name):
        return s[name.replace('?', side)]
    su = M([[f('f?u')*f('g?u'), 0, 0]])
    sv = M([[0, f('f?v')*f('g?v'), 0]])
    sw = M([[0, 0, f('f?w')*f('g?w')]])
    suxi = M([[f('f?uxi')*f('g?u'), 0, 0]])
    svxi = M([[0, f('f?vxi')*f('g?v'), 0]])
    swxi = M([[0, 0, f('f?wxi')*f('g?w')]])
    sueta = M([[f('f?u')*f('g?ueta'), 0, 0]])
    sveta = M([[0, f('f?v')*f('g?veta'), 0]])
    sweta = M([[0, 0, f('f?w')*f('g?weta')]])
    swxixi = M([[0, 0, f('f?wxixi')*f('g?w')]])
    swxieta = M([[0, 0, f('f?wxi')*f('g?weta')]])
    swetaeta = M([[0, 0, f('f?w')*f('g?wetaeta')]])

    # variations of the rotations phix and phiy
    Gx = -(2/a)*swxi
    Gy = -(2/b)*sweta + sv/r
    G = M([Gx, Gy])

    B = M([(2/a)*suxi + phix*Gx,
           (2/b)*sveta + sw/r + phiy*Gy,
           (2/b)*sueta + (2/a)*svxi + phix*Gy + phiy*Gx,
           -(2/a)**2*swxixi,
           -(2/b)**2*swetaeta + (2/b)*sveta/r,
           -2*(2/a)*(2/b)*swxieta + 3/(2*r)*(2/a)*svxi - 1/(2*r)*(2/b)*sueta])
    B0 = B.subs({phix: 0, phiy: 0})

    # u(z), v(z), w(z) = u0 + z*phix, v0 + z*phiy, w0
    g5 = M([su, sv, sw, Gx, Gy])

    return B, B0, G, g5


def build():
    r"""Integrands of all the matrices

    The integrands are given per unit area of the natural domain, such that
    the integral over the physical domain `x_1 \le x \le x_2`,
    `y_1 \le y \le y_2` is ``intx*inty/4`` times the integral over
    `\xi` and `\eta`, where ``intx = x2 - x1`` and ``inty = y2 - y1``.

    """
    BA, B0A, GA, gA5 = shape_functions('A')
    BB, B0B, GB, gB5 = shape_functions('B')

    # Constitutive stiffness matrix, with K0L, KL0 and KLL
    kC = BA.T*F*BB

    # Linear constitutive stiffness matrix
    kC0 = B0A.T*F*B0B

    # Internal force vector
    fint = BA.T*M([[Nxx, Nyy, Nxy, Mxx, Myy, Mxy]]).T

    # Geometric stiffness matrix
    Nmat = M([[Nxx, Nxy],
              [Nxy, Nyy]])
    kG = GA.T*Nmat*GB

    # Mass matrix
    maux = M([[ 1,  0, 0, -d,  0],
              [ 0,  1, 0,  0, -d],
              [ 0,  0, 1,  0,  0],
              [-d,  0, 0, h**2/12 + d**2, 0],
              [ 0, -d, 0, 0, h**2/12 + d**2]])
    kM = h*rho*gA5.T*maux*gB5

    return dict(kC=kC, kC0=kC0, fint=fint, kG=kG, kM=kM)


def collect_ABD(m, vars=ABD):
    m = m.copy()
    for (i, j), mij in np.ndenumerate(m):
        m[i, j] = collect(expand(mij), vars)
    return m


if __name__ == '__main__':
    from panels.dev.matrixtools import mprint_as_sparse

    matrices = build()
    outdir = './output_expressions_python/'
    os.makedirs(outdir, exist_ok=True)
    for name, m in matrices.items():
        if name == 'kG':
            m = collect_ABD(m, (Nxx, Nyy, Nxy))
        elif name != 'kM':
            m = collect_ABD(m)
        out = mprint_as_sparse(m, name, '11', print_file=False)
        with open(outdir + 'sympy_%s.txt' % name, 'w') as f:
            f.write(out)
