import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sympy import Matrix as M, Symbol, init_printing, var

var('a, b, r, h, d, intx, inty, rho, aeromu, beta, gamma, w_xi, w_xi_o, w_eta, w_eta_o')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('Nxx, Nyy, Nxy, Mxx, Myy, Mxy')
var('Nxx_o, Nyy_o, Nxy_o, Mxx_o, Myy_o, Mxy_o')

var('fAu, gAu, fAv, gAv, fAw, gAw, fAu_xi, fAv_xi, fAw_xi, gAu_eta, gAv_eta, gAw_eta, fAw_xixi, gAw_etaeta')
var('fBu, gBu, fBv, gBv, fBw, gBw, fBu_xi, fBv_xi, fBw_xi, gBu_eta, gBv_eta, gBw_eta, fBw_xixi, gBw_etaeta')

var('uA, vA, wA, uA_xi, vA_xi, wA_xi, uA_eta, vA_eta, wA_eta, wA_xixi, wA_xieta, wA_etaeta')
var('uB, vB, wB, uB_xi, vB_xi, wB_xi, uB_eta, vB_eta, wB_eta, wB_xixi, wB_xieta, wB_etaeta')

F = M([[A11, A12, A16, B11, B12, B16],
       [A12, A22, A26, B12, B22, B26],
       [A16, A26, A66, B16, B26, B66],
       [B11, B12, B16, D11, D12, D16],
       [B12, B22, B26, D12, D22, D26],
       [B16, B26, B66, D16, D26, D66]])

suA = M([[fAu*gAu, 0, 0]])
svA = M([[0, fAv*gAv, 0]])
swA = M([[0, 0, fAw*gAw]])

suA_xi = M([[fAu_xi*gAu, 0, 0]])
svA_xi = M([[0, fAv_xi*gAv, 0]])
swA_xi = M([[0, 0, fAw_xi*gAw]])

suA_eta = M([[fAu*gAu_eta, 0, 0]])
svA_eta = M([[0, fAv*gAv_eta, 0]])
swA_eta = M([[0, 0, fAw*gAw_eta]])

swA_xixi = M([[0, 0, fAw_xixi*gAw]])
swA_xieta = M([[0, 0, fAw_xi*gAw_eta]])
swA_etaeta = M([[0, 0, fAw*gAw_etaeta]])

suB = M([[fBu*gBu, 0, 0]])
svB = M([[0, fBv*gBv, 0]])
swB = M([[0, 0, fBw*gBw]])

suB_xi = M([[fBu_xi*gBu, 0, 0]])
svB_xi = M([[0, fBv_xi*gBv, 0]])
swB_xi = M([[0, 0, fBw_xi*gBw]])

suB_eta = M([[fBu*gBu_eta, 0, 0]])
svB_eta = M([[0, fBv*gBv_eta, 0]])
swB_eta = M([[0, 0, fBw*gBw_eta]])

swB_xixi = M([[0, 0, fBw_xixi*gBw]])
swB_xieta = M([[0, 0, fBw_xi*gBw_eta]])
swB_etaeta = M([[0, 0, fBw*gBw_etaeta]])

GA = M([(2/a)*swA_xi, (2/b)*swA_eta])
BA = M([(2/a)*suA_xi + (2/a)**2*swA_xi*w_xi,
        (2/b)*svA_eta + 1/r*swA + (2/b)**2*swA_eta*w_eta,
        (2/b)*suA_eta + (2/a)*svA_xi
        + (2/a)*(2/b)*(w_eta*swA_xi + w_xi*swA_eta),
       -(2/a)*(2/a)*swA_xixi,
       -(2/b)*(2/b)*swA_etaeta,
       -(2/a)*(2/b)*2*swA_xieta])
B0A = BA.subs({w_xi.name:0, w_eta.name:0})

GB = M([(2/a)*swB_xi, (2/b)*swB_eta])
BB = M([(2/a)*suB_xi + (2/a)**2*swB_xi*w_xi,
        (2/b)*svB_eta + 1/r*swB + (2/b)**2*swB_eta*w_eta,
        (2/b)*suB_eta + (2/a)*svB_xi
        + (2/a)*(2/b)*(w_eta*swB_xi + w_xi*swB_eta),
       -(2/a)*(2/a)*swB_xixi,
       -(2/b)*(2/b)*swB_etaeta,
       -(2/a)*(2/b)*2*swB_xieta])
B0B = BB.subs({w_xi.name:0, w_eta.name:0})

# Constitutive stiffness matrix
kC = intx*inty/4*BA.T*F*BB

# Internal force vector
fint = intx*inty/4*BA.T*M([[Nxx, Nyy, Nxy, Mxx, Myy, Mxy]]).T

# Linear constitutive stiffness matrix

kC0 = intx*inty/4*B0A.T*F*B0B;

# Geometric stiffness matrix

Nmat = M([[Nxx, Nxy],
          [Nxy, Nyy]])

kG = intx*inty/4*GA.T*Nmat*GB

# Mass matrix

# u(z) = u - z w,x, v(z) = v - z w,y and w, with z measured from the
# reference surface, the laminate spanning -h/2 + d <= z <= h/2 + d, such
# that the density enters through the moments I_n = int rho z^n dz of
# composites.Laminate, intrho, intrhoz and intrhoz2
var('I0, I1, I2')
maux = M([[I0,  0,  0, I1,  0],
          [ 0, I0,  0,  0, I1],
          [ 0,  0, I0,  0,  0],
          [I1,  0,  0, I2,  0],
          [ 0, I1,  0,  0, I2]])

gA5 = M([suA, svA, swA, -(2/a)*swA_xi, -(2/b)*swA_eta])
gB5 = M([suB, svB, swB, -(2/a)*swB_xi, -(2/b)*swB_eta])
kM = intx*inty/4*gA5.T*maux*gB5

# Aerodynamic and damping matrix using piston's theory, with the aerodynamic
# load q = beta*w,x + gamma*w along w and the curvature term gamma*w, see
# Shell.calc_kA()

kAx = -intx*inty/4 * (beta*(2/a)*swA_xi.T*swB + gamma*swA.T*swB)
kAy = -intx*inty/4 * beta*(2/b)*swA_eta.T*swB

cA = -intx*inty/4*aeromu*swA.T*swB

# Printing results

from sympy import collect, simplify

def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m

vars = (A11, A12, A16, A22, A26, A66,
        B11, B12, B16, B22, B26, B66,
        D11, D12, D16, D22, D26, D66)

piece_wise_simplify(kC, vars)
piece_wise_simplify(kC0, vars)
piece_wise_simplify(kG, (Nxx, Nyy, Nxy))
piece_wise_simplify(fint, vars)

from panels.dev.matrixtools import mprint_as_sparse

outdir = './output_expressions_python/'
import os
try: os.makedirs(outdir)
except: pass

matrices = [
    [kC, 'kC'],
    [kC0, 'kC0'],
    [kG, 'kG'],
    [kM, 'kM'],
    [kAx, 'kAx'],
    [kAy, 'kAy'],
    [cA, 'cA'],
    [fint, 'fint'],
]

for m in matrices:
    try:
        out = mprint_as_sparse(m[0], m[1], '11', print_file=False)
    except:
        print(m)
    with open(outdir + 'sympy_%s.txt' % m[1], 'w') as f:
        f.write(out)
