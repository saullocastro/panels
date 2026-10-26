# Symbolically calculates the matrix elements from a given expression
# In this case, the matrix expressions for prescribed displacements
#       Point
#       Line along x and y constant

import sys
sys.path.append(r'C:\Users\natha\Documents\GitHub\panels')

import numpy as np
from sympy import Matrix as M, Symbol, init_printing, var

var('a, b, h, d, w_xi, w_eta')
var('ku, kv, kw')

var('fAu, gAu, fAv, gAv, fAw, gAw, fAu_xi, fAv_xi, fAw_xi, gAu_eta, gAv_eta, gAw_eta, fAw_xixi, gAw_etaeta')
var('fBu, gBu, fBv, gBv, fBw, gBw, fBu_xi, fBv_xi, fBw_xi, gBu_eta, gBv_eta, gBw_eta, fBw_xixi, gBw_etaeta')

var('uA, vA, wA, uA_xi, vA_xi, wA_xi, uA_eta, vA_eta, wA_eta, wA_xixi, wA_xieta, wA_etaeta')
var('uB, vB, wB, uB_xi, vB_xi, wB_xi, uB_eta, vB_eta, wB_eta, wB_xixi, wB_xieta, wB_etaeta')

# SF already in terms of xi and eta
suA = M([[fAu*gAu, 0, 0]])
svA = M([[0, fAv*gAv, 0]])
swA = M([[0, 0, fAw*gAw]])

# suA_xi is deri of suA wrt xi
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

# Constitutive (penalty) stiffness matrix
    # All 3 disps stem from the same term so just club them into 1 stiffeness term
    # But its actually 3x3 matrix with just the middle terms populated
kCpd = (ku*suA.T*suB + kv*svA.T*svB + kw*swA.T*swB)           # no integral in the expression
kCld_xcte = b/2*(ku*suA.T*suB + kv*svA.T*svB + kw*swA.T*swB)  # b/2 comes from converting from y to eta from the original formulation
kCld_ycte = a/2*(ku*suA.T*suB + kv*svA.T*svB + kw*swA.T*swB)  # a/2 comes from converting from x to xi

# Printing results

from sympy import collect, simplify

def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m


piece_wise_simplify(kCpd, [])

from panels.dev.matrixtools import mprint_as_sparse

outdir = './output_expressions_python/'
import os
try: os.makedirs(outdir)
except: pass

matrices = [
    [kCpd, 'kCpd'],
    [kCld_xcte, 'kCld_xcte'],
    [kCld_ycte, 'kCld_ycte'],
]

for m in matrices:
    try:
        out = mprint_as_sparse(m[0], m[1], '11', print_file=False)
    except:
        print(m)
    with open(outdir + 'sympy_%s.txt' % m[1], 'w') as f:
        f.write(out)
