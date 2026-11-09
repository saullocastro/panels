# -*- coding: utf-8 -*-
r"""
Connection 'BFxcte' at a face of a cylinder: symbolic checks

The quantities of panels/multidomain/connections/kCBFxcte_face.py, for a
cylinder of radius r (base, panel 1, models cylshell_*) and an annular plate
(flange, panel 2, models annular_*) whose edge x2 = xcte2, at the radius
rho_e = r2 - xcte2, is attached to the line x = x_f, z = z_f of the
cylinder:

    u1(z_f) = w2,  v1(z_f) = v2,  w1 = -u2,  -w1,x = -w2,x

with the Kirchhoff-Love kinematics of the cylinder

    u1(z) = u1 - z w1,x,  v1(z) = (1 + z/r) v1 - z w1,y,  w1(z) = w1

Checked here:

1. The linearized displacement of the point X0 + z n of the normal of the
   cylinder, n being the unit normal of the deformed reference surface,
   is u1(z), v1(z), w1(z) above, the term z v1/r being that of the
   Sanders-Koiter kinematics (rotation of the normal about x: -w,y + v/r).
2. The rotation of the normal about the circumferential axis is -w1,x for
   the cylinder and -w2,x for the annular plate, both about the same
   vector e_y = e_y2.
3. For the six rigid motions t + omega x X of the assembly, the fields of
   the two panels give zero residuals of the four quantities if and only if
   the edge of the plate is at the radius of the connected line, rho_e = r
   + z_f; with the Donnell kinematics (no z v/r) the residual of v is -z_f
   v1/r, i.e. -z_f Omega for a rotation about the axis, v1 = r Omega.

Axes: the cylinder axis is the global X, the point of the reference surface
at (x, phi) is X0 = (x, r cos(phi), -r sin(phi)), y = r phi, so that e_x,
e_y = dX0/dy and e_z = e_x x e_y are right-handed with e_z pointing
outwards; the annular plate has e_x2 = -e_z (inwards), e_y2 = e_y and e_z2 =
e_x2 x e_y2 = e_x, its point at (x2, phi) being at the radius r2 - x2.

Run: python bfxcte_face.py (sympy only); it prints the checks and raises
AssertionError on a failure.
"""
import sympy as sp

x, phi, z, eps, xf, zf, x2 = sp.symbols('x phi z epsilon x_f z_f x2',
                                        real=True)
r, r2 = sp.symbols('r r2', positive=True)
u = sp.Function('u')(x, phi)
v = sp.Function('v')(x, phi)
w = sp.Function('w')(x, phi)


def frame(phi):
    ex = sp.Matrix([1, 0, 0])
    ey = sp.Matrix([0, -sp.sin(phi), -sp.cos(phi)])
    ez = ex.cross(ey)
    return ex, ey, ez


ex, ey, ez = frame(phi)
assert sp.simplify(ez - sp.Matrix([0, sp.cos(phi), -sp.sin(phi)])) == sp.zeros(3, 1)
X0 = sp.Matrix([x, r*sp.cos(phi), -r*sp.sin(phi)])
assert sp.simplify(X0.diff(phi)/r - ey) == sp.zeros(3, 1)

# 1. Kirchhoff-Love: X(z) = S + z n(S), S the deformed reference surface
S = X0 + eps*(u*ex + v*ey + w*ez)
n = S.diff(x).cross(S.diff(phi))
n = n/sp.sqrt(n.dot(n))
U = (u*ex + v*ey + w*ez) + z*n.diff(eps).subs(eps, 0)
U = U.applyfunc(sp.simplify)
wx, wy = w.diff(x), w.diff(phi)/r
uz = sp.simplify(U.dot(ex))
vz = sp.simplify(U.dot(ey))
wz = sp.simplify(U.dot(ez))
assert sp.simplify(uz - (u - z*wx)) == 0
assert sp.simplify(vz - ((1 + z/r)*v - z*wy)) == 0
assert sp.simplify(wz - w) == 0
print('1. u(z) =', uz, ';  v(z) =', sp.simplify(vz), ';  w(z) =', wz)

# 2. rotation about e_y: delta n = omega x n, omega_y = (delta n).e_x for
#    n = e_z (cylinder); omega_y = (delta n).e_x2 for n = e_z2 (plate)
dn = n.diff(eps).subs(eps, 0)
omega_y1 = sp.simplify(dn.dot(ex))
assert sp.simplify(omega_y1 + wx) == 0
assert sp.simplify(ey.cross(ez) - ex) == sp.zeros(3, 1)
print('2. rotation of the cylinder normal about e_y:', omega_y1)


# 3. rigid motions
def fields_cylinder(Ufun):
    r"""u, v, w of the cylinder at its reference surface for the
    displacement field Ufun(P)"""
    Ur = Ufun(X0)
    return Ur.dot(ex), Ur.dot(ey), Ur.dot(ez)


def fields_plate(Ufun):
    r"""u2, v2, w2 of the annular plate at (x2, phi), plane x = x_f"""
    ex2, ey2, ez2 = -ez, ey, ex
    Y = sp.Matrix([xf, (r2 - x2)*sp.cos(phi), -(r2 - x2)*sp.sin(phi)])
    Uy = Ufun(Y)
    return Uy.dot(ex2), Uy.dot(ey2), Uy.dot(ez2)


t1, t2, t3, o1, o2, o3 = sp.symbols('t1 t2 t3 o1 o2 o3', real=True)
t = sp.Matrix([t1, t2, t3])
om = sp.Matrix([o1, o2, o3])
rigid = lambda P: t + om.cross(P)
u1, v1, w1 = fields_cylinder(rigid)
u1_x = u1.diff(x)
w1_x, w1_y = w1.diff(x), w1.diff(phi)/r
uf = u1 - zf*w1_x
for sanders in (True, False):
    vf = (1 + zf/r)*v1 - zf*w1_y if sanders else v1 - zf*w1_y
    u2, v2, w2 = fields_plate(rigid)
    res = [uf - w2, vf - v2, w1 + u2, -w1_x + w2.diff(x2)]
    res = [sp.simplify(q.subs(x, xf)) for q in res]
    at_face = [sp.simplify(q.subs(x2, r2 - r - zf)) for q in res]
    if sanders:
        assert all(q == 0 for q in at_face), at_face
        # and only there: the residuals are proportional to r2 - x2 - r - zf
        assert any(sp.simplify(q) != 0 for q in res)
        print('3. Sanders-Koiter: residuals of the rigid motions with the '
              'edge at rho_e = r + z_f:', at_face)
        print('   otherwise:', [sp.factor(q) for q in res])
    else:
        print('   Donnell: residuals with the edge at rho_e = r + z_f:',
              at_face)
        assert sp.simplify(at_face[1] + zf*v1.subs(x, xf)/r) == 0
print('all checks passed')
