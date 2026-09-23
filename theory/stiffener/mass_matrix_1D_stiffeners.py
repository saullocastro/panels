# Through-thickness integration of the kinetic energy density of the panel,
# of the stiffener's base and of the stiffener's flange, giving the terms of
# the mass matrices of the panel, of the base (with offset) and of the 1D
# flange of the blade stiffeners
#
# The in-plane displacement at a distance z from the mid-surface of the
# panel is
#
#     u = u0 + z phi,  with phi = -w,x (or v = v0 + z psi, psi = -w,y)
#
# and the base and the flange are below the panel:
#
#     panel:   -h/2 <= z <= h/2
#     base:    -h/2 - hb <= z <= -h/2
#     flange:  -h/2 - hb - bf <= z <= -h/2 - hb,  -hf/2 <= y <= hf/2
#
# Writing each result as
#
#     int rho u^2 = rho A [u0, phi] [[1, -d], [-d, I/A + d^2]] [u0, phi]^T
#
# gives the offset d of the centroid (below the mid-surface of the panel),
# the area A and the moment of inertia I about the centroid, which are the
# coefficients of the matrix maux of the shells (d is the offset of the
# Shell, see theory/shells/plate_clpt_donnell) and of the matrix msf of the
# 1D flange, see
# theory/multidomain_panels/bladestiff1d_clt_donnell/bladestiff1d_clt_donnell.py
#
# The same results are obtained with the local coordinates zb and zf of the
# base and of the flange, with u = u0 + (zb - db) phi and u = u0 + (zf - df) phi
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sympy import var, integrate, collect, simplify, expand, Matrix as M

var('z, y, zb, zf, u0, phi, rho, h, hb, bf, hf, db, df')

u = u0 + z*phi

results = {}

print('Panel')
panel = collect(integrate(rho*u**2, (z, -h/2, h/2)), (h, u0, phi), simplify)
print(panel)
results['panel'] = (panel, h, 0, h**3/12)

print('Stiffener Base')
base = collect(integrate(rho*u**2, (z, -h/2 - hb, -h/2)), (u0, phi), simplify)
print(base)
results['base'] = (base, hb, h/2 + hb/2, hb**3/12)

print('Stiffener Flange')
flange = collect(integrate(rho*u**2, (z, -h/2 - hb - bf, -h/2 - hb),
                           (y, -hf/2, hf/2)), (u0, phi), simplify)
print(flange)
results['flange'] = (flange, bf*hf, h/2 + hb + bf/2, hf*bf**3/12)

# checking the matrix form and the offsets

for name, (res, A, d, I) in results.items():
    mat = rho*A*M([[1, -d], [-d, I/A + d**2]])
    q = M([[u0, phi]])
    assert simplify(expand(res - (q*mat*q.T)[0, 0])) == 0, name
    print('%s: A = %s, d = %s, I = %s' % (name, A, d, I))

# local coordinates of the base and of the flange

ub = u0 + (zb - db)*phi
base_local = integrate(rho*ub**2, (zb, -hb/2, hb/2))
print('Stiffener Base, local coordinate zb')
print(collect(base_local, (hb, u0, phi), simplify))
assert simplify(base_local.subs(db, h/2 + hb/2) - base) == 0

uf = u0 + (zf - df)*phi
flange_local = integrate(rho*uf**2, (zf, -bf/2, bf/2), (y, -hf/2, hf/2))
print('Stiffener Flange, local coordinate zf')
print(collect(flange_local, (bf, u0, phi), simplify))
assert simplify(flange_local.subs(df, h/2 + hb + bf/2) - flange) == 0

# the term kmf44 = kmf55 of the matrix msf of the 1D flange

kmf44 = bf**2/3 + bf*(h + 2*hb)/2 + (h + 2*hb)**2/4
assert simplify(kmf44 - (bf**2/12 + (h/2 + hb + bf/2)**2)) == 0
