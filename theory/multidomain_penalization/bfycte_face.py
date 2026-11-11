# -*- coding: utf-8 -*-
r"""
Connection 'BFycte' at a face of a cylinder: symbolic checks

The quantities of panels/multidomain/connections/kCBFycte_face.py, for a
cylinder of radius r (base, panel 1, models cylshell_*) and a flat plate
(flange, panel 2, e.g. a blade stringer or the web of a Z stringer) whose
edge y2 = ycte2 is attached to the generator phi = phi0, z = z_f of the
cylinder:

    u1(z_f) = u2,  v1(z_f) = w2,  w1 = -v2,  theta_y1(z_f) = theta_y2

with the Kirchhoff-Love kinematics of the cylinder

    u1(z) = u1 - z w1,x,  v1(z) = (1 + z/r) v1 - z w1,y,  w1(z) = w1
    theta_y1(z) = d v1(z)/dz = -w1,y + v1/r

the terms z v1/r and v1/r being those of the Sanders-Koiter kinematics,
used at the face for every cylinder model by default (key 'kinematics' =
'sanders'), and omitted with kinematics='model' for a Donnell cylinder;
theta_y2 = -w2,y2 for the plate.

Checked here:

1. The linearized displacement of the point X0 + z n of the normal of the
   cylinder, n being the unit normal of the deformed reference surface,
   is u1(z), v1(z), w1(z) above, and the rotation of the normal about the
   axis x, -(delta n).e_y, is minus theta_y1 = -w1,y + v1/r, independent
   of z; for the plate the same rotation is minus theta_y2 = -w2,y2.
2. For the six rigid motions t + omega x X of the assembly, the fields of
   the two panels give zero residuals of the four quantities if and only if
   the root of the plate is at the radius of the connected line, rho_e = r
   + z_f; with the Donnell kinematics (no z v/r, no v/r) the residuals are
   -z_f v1/r and -v1/r.
3. Rotation about the axis, v1 = r Omega: free of strain with the Donnell
   and the Sanders-Koiter kinematics (see bfxcte_face.py); with the Donnell
   kinematics at the face the residuals are -z_f Omega and -Omega.

Axes: the cylinder axis is the global X, the point of the reference surface
at (x, phi) is X0 = x e_x + r e_z(phi) = (x, r cos(phi), -r sin(phi)), y =
r phi, e_y = dX0/dy and e_z = e_x x e_y pointing outwards; the plate along
the generator phi0 has e_x2 = e_x, e_y2 = -e_z(phi0) (towards the axis) and
e_z2 = e_y(phi0), its point at (x, y2) being X0(x, phi0) + (rho_e - (y2 -
ycte2)) e_z(phi0) - r e_z(phi0), i.e. at the radius rho_e - (y2 - ycte2).

Run: python bfycte_face.py (sympy only); it prints the checks and raises
AssertionError on a failure.
"""
import sympy as sp

x, phi, z, eps, zf, s, phi0 = sp.symbols('x phi z epsilon z_f s phi0',
                                         real=True)
r = sp.symbols('r', positive=True)
u = sp.Function('u')(x, phi)
v = sp.Function('v')(x, phi)
w = sp.Function('w')(x, phi)


def frame(phi):
    ex = sp.Matrix([1, 0, 0])
    ey = sp.Matrix([0, -sp.sin(phi), -sp.cos(phi)])
    ez = ex.cross(ey)
    return ex, ey, ez


ex, ey, ez = frame(phi)
X0 = sp.Matrix([x, r*sp.cos(phi), -r*sp.sin(phi)])
assert sp.simplify(X0 - (x*ex + r*ez)) == sp.zeros(3, 1)
assert sp.simplify(X0.diff(phi)/r - ey) == sp.zeros(3, 1)

# 1. Kirchhoff-Love: X(z) = S + z n(S), S the deformed reference surface
S = X0 + eps*(u*ex + v*ey + w*ez)
n = S.diff(x).cross(S.diff(phi))
n = n/sp.sqrt(n.dot(n))
dn = n.diff(eps).subs(eps, 0)
Uz = (u*ex + v*ey + w*ez) + z*dn
wx, wy = w.diff(x), w.diff(phi)/r
assert sp.simplify(Uz.dot(ex) - (u - z*wx)) == 0
assert sp.simplify(Uz.dot(ey) - ((1 + z/r)*v - z*wy)) == 0
assert sp.simplify(Uz.dot(ez) - w) == 0
theta_y1 = sp.simplify(dn.dot(ey))
assert sp.simplify(theta_y1 - (-wy + v/r)) == 0
# rotation about e_x: delta n = omega x n gives omega_x = -(delta n).e_y
assert sp.simplify(ex.cross(ez) + ey) == sp.zeros(3, 1)
print('1. v(z) =', sp.simplify(Uz.dot(ey)), '; theta_y = dv(z)/dz =',
      theta_y1)


# 2. rigid motions
def fields_cylinder(Ufun):
    Ur = Ufun(X0)
    return Ur.dot(ex), Ur.dot(ey), Ur.dot(ez)


ex0, ey0, ez0 = frame(phi0)


def fields_plate(Ufun, rho_e):
    r"""u2, v2, w2 of the plate at (x, s), s = y2 - ycte2 the distance from
    its root along e_y2 = -e_z(phi0)"""
    ex2, ey2, ez2 = ex0, -ez0, ey0
    Y = x*ex0 + (rho_e - s)*ez0
    Uy = Ufun(Y)
    return Uy.dot(ex2), Uy.dot(ey2), Uy.dot(ez2)


t1, t2, t3, o1, o2, o3, rho = sp.symbols('t1 t2 t3 o1 o2 o3 rho', real=True)
t = sp.Matrix([t1, t2, t3])
om = sp.Matrix([o1, o2, o3])
rigid = lambda P: t + om.cross(P)
u1, v1, w1 = fields_cylinder(rigid)
w1_x, w1_y = w1.diff(x), w1.diff(phi)/r
u2, v2, w2 = fields_plate(rigid, rho)
theta_y2 = -w2.diff(s)
for sanders in (True, False):
    rinv = 1/r if sanders else 0
    uf = u1 - zf*w1_x
    vf = (1 + zf*rinv)*v1 - zf*w1_y
    th1 = -w1_y + rinv*v1
    res = [uf - u2, vf - w2, w1 + v2, th1 - theta_y2]
    res = [sp.simplify(q.subs(phi, phi0).subs(s, 0)) for q in res]
    at_face = [sp.simplify(q.subs(rho, r + zf)) for q in res]
    if sanders:
        assert all(q == 0 for q in at_face), at_face
        assert any(sp.simplify(q) != 0 for q in res)
        print('2. Sanders-Koiter: residuals of the rigid motions with the '
              'root at rho_e = r + z_f:', at_face)
        print('   otherwise:', [sp.factor(q) for q in res])
    else:
        v1_0 = v1.subs(phi, phi0)
        assert sp.simplify(at_face[0]) == 0 and sp.simplify(at_face[2]) == 0
        assert sp.simplify(at_face[1] + zf*v1_0/r) == 0
        assert sp.simplify(at_face[3] + v1_0/r) == 0
        print('   Donnell: residuals with the root at rho_e = r + z_f:',
              at_face)

# 3. rotation about the axis, the field of the tests
Om = sp.symbols('Omega', real=True)
rot = lambda P: sp.Matrix([-Om, 0, 0]).cross(P)
ur, vr, wr = fields_cylinder(rot)
assert sp.simplify(ur) == 0 and sp.simplify(vr - r*Om) == 0
assert sp.simplify(wr) == 0
u2r, v2r, w2r = fields_plate(rot, r + zf)
assert sp.simplify(u2r) == 0 and sp.simplify(v2r) == 0
assert sp.simplify(w2r - (r + zf - s)*Om) == 0
assert sp.simplify(-w2r.diff(s) - Om) == 0
print('3. rotation about the axis: v1 =', sp.simplify(vr), ', w2 =',
      sp.simplify(w2r), ', theta_y2 =', sp.simplify(-w2r.diff(s)))
print('all checks passed')
