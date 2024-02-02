# Symbolically calculates the matrix elements from a given expression
# In this case, the matrix expressions for prescribed displacements
#       Point
#       Line along x and y constant

import sys
sys.path.append(r'C:\Users\natha\Documents\GitHub\panels')

import numpy as np
from sympy import Matrix as M, Symbol, init_printing, var

var('a, b, h, d, wxi, weta')
var('ku, kv, kw')

var('fAu, gAu, fAv, gAv, fAw, gAw, fAuxi, fAvxi, fAwxi, gAueta, gAveta, gAweta, fAwxixi, gAwetaeta')
var('fBu, gBu, fBv, gBv, fBw, gBw, fBuxi, fBvxi, fBwxi, gBueta, gBveta, gBweta, fBwxixi, gBwetaeta')

var('uA, vA, wA, uAxi, vAxi, wAxi, uAeta, vAeta, wAeta, wAxixi, wAxieta, wAetaeta')
var('uB, vB, wB, uBxi, vBxi, wBxi, uBeta, vBeta, wBeta, wBxixi, wBxieta, wBetaeta')

# SF already in terms of xi and eta
suA = M([[fAu*gAu, 0, 0]])
svA = M([[0, fAv*gAv, 0]])
swA = M([[0, 0, fAw*gAw]])

# suAxi is deri of suA wrt xi
suAxi = M([[fAuxi*gAu, 0, 0]])
svAxi = M([[0, fAvxi*gAv, 0]])
swAxi = M([[0, 0, fAwxi*gAw]])

suAeta = M([[fAu*gAueta, 0, 0]])
svAeta = M([[0, fAv*gAveta, 0]])
swAeta = M([[0, 0, fAw*gAweta]])

swAxixi = M([[0, 0, fAwxixi*gAw]])
swAxieta = M([[0, 0, fAwxi*gAweta]])
swAetaeta = M([[0, 0, fAw*gAwetaeta]])

suB = M([[fBu*gBu, 0, 0]])
svB = M([[0, fBv*gBv, 0]])
swB = M([[0, 0, fBw*gBw]])

suBxi = M([[fBuxi*gBu, 0, 0]])
svBxi = M([[0, fBvxi*gBv, 0]])
swBxi = M([[0, 0, fBwxi*gBw]])

suBeta = M([[fBu*gBueta, 0, 0]])
svBeta = M([[0, fBv*gBveta, 0]])
swBeta = M([[0, 0, fBw*gBweta]])

swBxixi = M([[0, 0, fBwxixi*gBw]])
swBxieta = M([[0, 0, fBwxi*gBweta]])
swBetaeta = M([[0, 0, fBw*gBwetaeta]])

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
