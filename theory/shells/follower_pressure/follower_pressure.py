r"""Follower (hydrostatic) pressure: virtual work, load vector and load stiffness

A pressure ``p`` that stays normal to the deformed mid-surface and acts on its
deformed area does the virtual work

    dW_p = int p n.du dA = int p (X,x x X,y).du dx dy,     X = X0 + u

where ``x`` and ``y`` are the coordinates of the undeformed mid-surface, ``y``
being the arc length for the cylinders, and ``u = u ex + v ey + w ez`` is the
displacement of the mid-surface. In the local basis of the undeformed
mid-surface, where ``ez`` is the normal, positive outwards for the cylinders,
and ``ey,y = -ez/r``, ``ez,y = ey/r``::

    X,x = (1 + u,x) ex + v,x ey + w,x ez = A
    X,y = u,y ex + (1 + v,y + w/r) ey + (w,y - v/r) ez = B

The area vector ``a = A x B`` (per unit undeformed area) is bilinear in the
displacement gradients, ``a = a0 + a1 + a2``:

- ``a0 = ez``, the dead load;
- ``a1``, linear in the gradients, the rotation of the normal and the change
  of area;
- ``a2``, quadratic in the gradients.

The kinematics of each model enter through ``B``:

- plate: ``1/r = 0``;
- Donnell cylinder: the plate rotations, ``B = (u,y, 1 + v,y + w/r, w,y)``,
  i.e. the ``v/r`` of the rotation is neglected as in the Donnell strains;
- Sanders cylinder: ``B = (u,y, 1 + v,y + w/r, w,y - v/r)``, i.e. the exact
  geometry, with the rotation ``phiy = -w,y + v/r`` of the kernels.

For the FSDT and TSDT models the pressure acts on the mid-surface, so only
``u, v, w`` enter, never ``phix, phiy``.

Truncation
----------
With the moderate-rotation ordering of the von Karman, Donnell and Sanders
strains, ``u,x ~ eps`` and ``w,x ~ eps**(1/2)``, the virtual work is

    p a.du = p (a_z dw + a_x du + a_y dv),  du ~ eps**(1/2) dw

such that ``a0 + a1`` retains every term up to ``O(eps) p dw``, the order of
the strains themselves, and ``a2`` is ``O(eps**2) p dw``, of the order of the
terms the strains neglect. The default (``follower='linear'``) is therefore

    a = (-w,x,  -w,y + v/r,  1 + u,x + v,y + w/r)

with ``v/r`` only for the Sanders kinematics and ``w/r`` for both cylinders.
``F_p(c)`` is then affine in ``c`` and the load stiffness is constant. The
option ``follower='quadratic'`` keeps ``a2`` (the complete area vector of the
kinematics above), whose load stiffness depends on ``c``.

Symmetry
--------
For the first-order area vector the load stiffness, as a bilinear form
``P(d, D) = int p d.a1(D) dA``, has the skew part (see ``check_symmetry``)

    P(d, D) - P(D, d) =
        - oint p [(du Dw - dw Du) nx + (dv Dw - dw Dv) ny] ds
        + int [p,x (du Dw - dw Du) + p,y (dv Dw - dw Dv)] dA

over the loaded region. ``K_p`` is therefore symmetric when ``p`` is uniform
over the whole integration domain and, on each edge, either ``w = 0`` or the
displacement normal to the edge is zero (``u`` on the edges ``x = const``,
``v`` on ``y = const``), e.g. simply supported or clamped panels, or the
symmetry planes of a ring. Otherwise, e.g. a free or partially restrained
edge, a patch (whose edges are inside the shell), or ``p(x, y)`` varying, it
is unsymmetric: the load is not conservative. This is the finite-element
version of the result of Hibbitt (1979) and Schweizerhof and Ramm (1984), for
whom the load stiffness of a pressure is symmetric for closed surfaces or
when the boundary of the loaded surface is fixed, with the pressure a
function of the reference coordinates. A hydrostatic head, a function of the
current position, would add a further term, not implemented.

Load surface
------------
A pressure ``p`` acting on the surface ``z = zp`` of a cylinder of radius
``r``, e.g. the outer face ``zp = h/2`` of a thick sandwich, gives the
resultant ``p (r + zp)`` per unit length of mid-surface arc, which is applied
as ``p (1 + zp/r)`` per unit mid-surface area. The direction and the change of
area are those of the mid-surface, i.e. the terms of relative order ``zp k``,
``k`` the change of curvature, and the moment of the in-plane components
about the mid-surface are neglected, as in a shallow shell. For a ring mode
``n`` they are of relative order ``n**2 zp/r`` in the load stiffness, which is
the error to expect at ``r/h = 15``.

"""
import os

import sympy
from sympy import (Matrix, Rational, Symbol, Float, symbols, diff, expand,
                   simplify, Function, cos, sin, pi, integrate, solve)
from sympy.printing.str import StrPrinter

HERE = os.path.dirname(os.path.abspath(__file__))

#: model -> (number of dofs per term, geometry, kinematics)
MODELS = {
    'plate_clpt_donnell': (3, 'plate', 'donnell'),
    'cylshell_clpt_donnell': (3, 'cylshell', 'donnell'),
    'cylshell_clpt_sanders': (3, 'cylshell', 'sanders'),
    'plate_fsdt_donnell': (5, 'plate', 'donnell'),
    'plate_tsdt_donnell': (5, 'plate', 'donnell'),
    'cylshell_fsdt_donnell': (5, 'cylshell', 'donnell'),
    'cylshell_fsdt_sanders': (5, 'cylshell', 'sanders'),
    'cylshell_tsdt_donnell': (5, 'cylshell', 'donnell'),
    'cylshell_tsdt_sanders': (5, 'cylshell', 'sanders'),
}

a, b, r = symbols('a, b, r', positive=True)
#: qf = 1 keeps the quadratic part a2 of the area vector, ref = 1 the dead
#: part a0, both are 0. or 1. in the kernels
qf, ref = symbols('qf, ref')
#: displacement gradients at an integration point, in physical coordinates
ux, uy, vx, vy, vv, ww, wx, wy = symbols('ux, uy, vx, vy, vv, ww, wx, wy')
POINT = (ux, uy, vx, vy, vv, ww, wx, wy)
#: how each point quantity is accumulated from the Ritz constants, with the
#: field index (0 = u, 1 = v, 2 = w) and the basis functions in the
#: notation of the kernels
POINT_SUM = {
    'ux': (0, '(2/a)*fAuxi*gAu'),
    'uy': (0, '(2/b)*fAu*gAueta'),
    'vx': (1, '(2/a)*fAvxi*gAv'),
    'vy': (1, '(2/b)*fAv*gAveta'),
    'vv': (1, 'fAv*gAv'),
    'ww': (2, 'fAw*gAw'),
    'wx': (2, '(2/a)*fAwxi*gAw'),
    'wy': (2, '(2/b)*fAw*gAweta'),
}
FIELDS = ('u', 'v', 'w')


def kinematic_factors(geometry, kinematics):
    r"""``(kw, kv)``: the factors of ``w`` in ``1 + v,y + kw w`` and of ``v``
    in ``w,y - kv v``"""
    if geometry == 'plate':
        return 0, 0
    if kinematics == 'donnell':
        return 1/r, 0
    if kinematics == 'sanders':
        return 1/r, 1/r
    raise ValueError(kinematics)


def area_vector(geometry, kinematics):
    r"""Complete area vector ``a = A x B`` of the kinematics, and its parts
    ``a0, a1, a2`` of order 0, 1 and 2 in the displacement gradients"""
    kw, kv = kinematic_factors(geometry, kinematics)
    A = Matrix([1 + ux, vx, wx])
    B = Matrix([uy, 1 + vy + kw*ww, wy - kv*vv])
    acomp = A.cross(B)
    t = Symbol('t')
    scaled = acomp.subs({q: t*q for q in POINT}, simultaneous=True)
    parts = []
    for order in range(3):
        parts.append(scaled.applyfunc(
            lambda e: expand(diff(e, t, order).subs(t, 0)/sympy.factorial(order))))
    a0, a1, a2 = parts
    assert (a0 + a1 + a2 - acomp).applyfunc(expand) == Matrix([0, 0, 0])
    return acomp, a0, a1, a2


def kernel_area_vector(geometry, kinematics):
    r"""Area vector of the kernels, ``ref a0 + a1 + qf a2``"""
    _, a0, a1, a2 = area_vector(geometry, kinematics)
    return (ref*a0 + a1 + qf*a2).applyfunc(expand)


def trial_derivatives():
    r"""Derivative of each point quantity with respect to a Ritz constant of
    each field, in the notation of the kernels (side ``B``)"""
    fBu, fBuxi, gBu, gBueta = symbols('fBu, fBuxi, gBu, gBueta')
    fBv, fBvxi, gBv, gBveta = symbols('fBv, fBvxi, gBv, gBveta')
    fBw, fBwxi, gBw, gBweta = symbols('fBw, fBwxi, gBw, gBweta')
    zero = {q: 0 for q in POINT}
    du = {**zero, ux: (2/a)*fBuxi*gBu, uy: (2/b)*fBu*gBueta}
    dv = {**zero, vx: (2/a)*fBvxi*gBv, vy: (2/b)*fBv*gBveta, vv: fBv*gBv}
    dw = {**zero, ww: fBw*gBw, wx: (2/a)*fBwxi*gBw, wy: (2/b)*fBw*gBweta}
    return du, dv, dw


def test_functions():
    fAu, gAu, fAv, gAv, fAw, gAw = symbols('fAu, gAu, fAv, gAv, fAw, gAw')
    return fAu*gAu, fAv*gAv, fAw*gAw


def kernel_expressions(geometry, kinematics):
    r"""Integrands of the kernels, to be multiplied by ``weight``, which
    already contains ``p`` and the Jacobian of the patch

    Returns
    -------
    acode : list
        ``[ax, ay, az]``, the area vector at the integration point, with
        ``F_p = int p (Nu ax + Nv ay + Nw az) dA``.
    kblocks : dict
        ``{(row_field, col_field): expr}`` of the load stiffness ``-dF_p/dc``,
        only the blocks that are not identically zero.

    """
    acode = list(kernel_area_vector(geometry, kinematics))
    tests = test_functions()
    trials = trial_derivatives()
    kblocks = {}
    for i, Ni in enumerate(tests):
        for j, dq in enumerate(trials):
            dai = sum(diff(acode[i], q)*dq[q] for q in POINT)
            e = expand(-Ni*dai)
            if e != 0:
                kblocks[(i, j)] = e
    return acode, kblocks


def check_symmetry():
    r"""Skew part of the first-order load stiffness, see the module notes

    For each kinematics, ``P(d, D) - P(D, d)`` is checked against the
    divergence (boundary) and pressure gradient terms.
    """
    x, y = symbols('x, y')
    p = Function('p')(x, y)
    fu = [Function(n)(x, y) for n in ('du', 'dv', 'dw')]
    FU = [Function(n)(x, y) for n in ('Du', 'Dv', 'Dw')]

    def a1_of(f, kw, kv):
        u, v, w = f
        vals = {ux: diff(u, x), uy: diff(u, y), vx: diff(v, x),
                vy: diff(v, y), vv: v, ww: w, wx: diff(w, x), wy: diff(w, y)}
        return [e.subs(vals) for e in a1]

    for geometry, kinematics in (('plate', 'donnell'), ('cylshell', 'donnell'),
                                 ('cylshell', 'sanders')):
        kw, kv = kinematic_factors(geometry, kinematics)
        _, _, a1, _ = area_vector(geometry, kinematics)
        P = lambda d, D: p*sum(di*ai for di, ai in zip(d, a1_of(D, kw, kv)))
        skew = P(fu, FU) - P(FU, fu)
        du, dv, dw = fu
        Du, Dv, Dw = FU
        qx = du*Dw - dw*Du
        qy = dv*Dw - dw*Dv
        ref_skew = (-diff(p*qx, x) - diff(p*qy, y)
                    + diff(p, x)*qx + diff(p, y)*qy)
        assert simplify(expand(skew - ref_skew)) == 0, (geometry, kinematics)
    return True


def ring_buckling(kinematics, load, n, A=None, D=None, As=None, R=None):
    r"""Buckling pressure of a ring in plane strain, mode ``cos(n theta)``

    The displacements are ``w = W cos(n y/R)`` and ``v = V sin(n y/R)``, plus
    ``phiy = P sin(n y/R)`` for the FSDT, the prestress is the membrane state
    ``Nyy = p R`` of the pressure ``p`` (positive outwards, an external
    pressure is negative), and the second variation of the total potential
    is

        U2 = 1/2 int (A eyy**2 + D kyy**2 + As gyz**2 + Nyy byy**2) dy
             - 1/2 int p u.a1(u) dy

    with ``A``, ``D`` and ``As`` the plane-strain membrane, bending and
    transverse shear stiffnesses (``A22``, ``D22`` and ``A44`` of the
    laminate; ``As = None`` for the CLPT), ``byy = w,y - kv v`` the rotation
    of the von Karman terms, and the linear strains of the kernels:

    - CLPT, Sanders: ``eyy = v,y + w/R``, ``kyy = -w,yy + v,y/R``;
    - CLPT, Donnell: ``eyy = v,y + w/R``, ``kyy = -w,yy``;
    - FSDT, Sanders: ``kyy = phiy,y + v,y/R``, ``gyz = phiy + w,y``;
    - FSDT, Donnell: ``kyy = phiy,y``, ``gyz = phiy + w,y``.

    The load term ``u.a1(u)`` is:

    - ``'follower'``: ``v (-w,y + kv v) + w (v,y + w/R)``, the first-order area
      vector of this module;
    - ``'dead'``: zero, a pressure of constant direction, as
      :meth:`.Shell.add_pressure_load` with ``follower=False``;
    - ``'central'``: ``v**2/R``, a load of constant magnitude per unit
      undeformed length directed to the centre of the ring.

    With numeric ``A``, ``D``, ``R`` (and ``As``) the pressure of smallest
    modulus that makes the stiffness matrix singular is returned as a float,
    negative for an external pressure, exact, i.e. not assuming inextensional
    modes. Otherwise the symbolic roots are returned.

    """
    y, W, V, P, p = symbols('y, W, V, P, p')
    Rs, As_, Ds, Ss = symbols('R, A, D, S', positive=True)
    w = W*cos(n*y/Rs)
    v = V*sin(n*y/Rs)
    sanders = kinematics == 'sanders'
    kv = 1/Rs if sanders else 0
    eyy = diff(v, y) + w/Rs
    fsdt = As is not None
    if fsdt:
        phiy = P*sin(n*y/Rs)
        kyy = diff(phiy, y) + (diff(v, y)/Rs if sanders else 0)
        gyz = phiy + diff(w, y)
        dofs = (W, V, P)
    else:
        kyy = -diff(w, y, 2) + (diff(v, y)/Rs if sanders else 0)
        gyz = 0
        dofs = (W, V)
    byy = diff(w, y) - kv*v
    if load == 'follower':
        load_term = v*(-diff(w, y) + kv*v) + w*(diff(v, y) + w/Rs)
    elif load == 'dead':
        load_term = 0
    elif load == 'central':
        load_term = v**2/Rs
    else:
        raise ValueError(load)
    integrand = (As_*eyy**2 + Ds*kyy**2 + Ss*gyz**2 + p*Rs*byy**2
                 - p*load_term)/2
    U2 = integrate(expand(integrand), (y, 0, 2*pi*Rs))
    H = Matrix([[diff(U2, qi, qj) for qj in dofs] for qi in dofs])
    roots = solve(sympy.factor(H.det()), p)
    vals = {Rs: R, As_: A, Ds: D}
    if fsdt:
        vals[Ss] = As
    if None in vals.values():
        return roots
    roots = [complex(rt.subs(vals)) for rt in roots]
    roots = [rt.real for rt in roots if abs(rt.imag) <= 1e-12*abs(rt)]
    roots = [rt for rt in roots if rt != 0]
    return min(roots, key=abs)


def ring_buckling_inextensional(kinematics, load, n):
    r"""``p R**3/D`` of :func:`ring_buckling` (CLPT) for an inextensional
    ring, ``A -> infinity``"""
    roots = ring_buckling(kinematics, load, n)
    Rs, As_, Ds = symbols('R, A, D', positive=True)
    out = []
    for rt in roots:
        e = sympy.limit(simplify(rt*Rs**3/Ds), As_, sympy.oo)
        if e != 0 and e.is_finite:
            out.append(e)
    return min(out, key=abs)


class Printer(StrPrinter):
    def _print_Float(self, e):
        return repr(float(e))


def pstr(e):
    import re
    reps = {q: Float(q) for q in e.atoms(Rational) if not q.is_Integer}
    out = Printer().doprint(e.xreplace(reps))
    # x**n -> x*x*...*x, as panels.dev.matrixtools.pow2mult
    for pw in sorted(set(re.findall(r'\w+\*\*\d+', out)), key=len)[::-1]:
        var, exp = pw.split('**')
        out = out.replace(pw, '(' + '*'.join([var]*int(exp)) + ')')
    return out


if __name__ == '__main__':
    check_symmetry()
    print('symmetry identity verified for all kinematics')
    outdir = os.path.join(HERE, 'output_expressions_python')
    os.makedirs(outdir, exist_ok=True)
    for geometry, kinematics in (('plate', 'donnell'), ('cylshell', 'donnell'),
                                 ('cylshell', 'sanders')):
        acomp, a0, a1, a2 = area_vector(geometry, kinematics)
        acode, kblocks = kernel_expressions(geometry, kinematics)
        lines = ['# %s %s' % (geometry, kinematics)]
        for name, vec in (('a0', a0), ('a1', a1), ('a2', a2)):
            for comp, e in zip('xyz', vec):
                lines.append('%s_%s = %s' % (name, comp, e))
        for (i, j), e in sorted(kblocks.items()):
            lines.append('kCfollower[%s, %s] = %s' % (FIELDS[i], FIELDS[j], pstr(e)))
        out = '\n'.join(lines)
        print(out)
        with open(os.path.join(outdir, 'sympy_follower_%s_%s.txt'
                               % (geometry, kinematics)), 'w') as f:
            f.write(out + '\n')
    print('ring, inextensional limit, p R**3/D for n = 2:')
    for kin in ('sanders', 'donnell'):
        for load in ('follower', 'dead', 'central'):
            print('   %-8s %-8s %s' % (kin, load,
                  ring_buckling_inextensional(kin, load, 2)))
