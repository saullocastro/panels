# Symbolically calculates the matrices of the flange of the blade stiffener
# with a 1D formulation (panels.stiffener.BladeStiff1D), model
# 'bladestiff1d_clt_donnell'
#
# The flange is a beam along x, at the constant coordinate y = ys of the
# panel, using the same approximation functions and the same degrees of
# freedom as the panel (and the stiffener's base, which is modelled as a 2D
# Shell domain, see ../bladestiff2d_clt_donnell). The integration is done
# along x only, the functions along y (g*) are evaluated at eta(ys), such
# that the matrices are:
#
#     k0f = (a/2) bf int gAsf^T ksf gBsf dxi
#     kG0f = (a/2) int GA^T Fx GB dxi
#     kMf = (a/2) hf bf rho int gA5^T msf gB5 dxi
#
# with (see panels.stiffener.BladeStiff1D._rebuild()):
#
#     E1 = sum_k h_k (Q11 - Q12^2/Q22), per unit width of the flange
#     F1 = bf^2/12 E1
#     S1 = -sum_k y_k h_k (Q16 - Q12 Q16/Q22)
#     Jxx = hf bf^3/12 + bf hf^3/12
#     df = bf/2 + hb + h/2, the distance between the mid-surface of the panel
#          and the centroid of the flange
#
# The kernels that correspond to each matrix are in
# panels/stiffener/models/bladestiff1d_clt_donnell.pyx:
#
#     k0f: fkCf,  kG0f: fkGf,  kMf: fkMf
#
#NOTE the second and third rows of gAsf and gBsf are +(2/a)w,xi and
#     -(2/a)w,xi, the corresponding contributions of E2, E3 and S2 cancel out
#     and do not appear in the kernels
#
#NOTE the coupling terms of the mass matrix msf are -df, as given by the
#     through-thickness integration of the flange, see
#     theory/stiffener/mass_matrix_1D_stiffeners.py, and as used for the
#     offset d of the shells, see theory/shells/plate_clpt_donnell. Eq. (26)
#     of Castro et al. (2016), https://doi.org/10.1016/j.compstruct.2015.12.056,
#     and the kernel fkMf up to version 0.7.1 had -2*df
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sympy import Matrix as M, var, collect, simplify

var('a, b, h, hb, hf, bf, df, rho')
var('E1, E2, E3, S1, S2, F1, Jxx, Fx')

var('fAu, gAu, fAv, gAv, fAw, gAw, fAuxi, fAvxi, fAwxi, gAueta, gAveta, gAweta, fAwxixi, gAwetaeta')
var('fBu, gBu, fBv, gBv, fBw, gBw, fBuxi, fBvxi, fBwxi, gBueta, gBveta, gBweta, fBwxixi, gBwetaeta')


def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m


suA = M([[fAu*gAu, 0, 0]])
svA = M([[0, fAv*gAv, 0]])
swA = M([[0, 0, fAw*gAw]])

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

# Constitutive stiffness matrix of the flange

ksf = M([[     E1,      E2,      E2,     -df*E1,      S1],
         [     E2,      E3,      E3,     -df*E2,      S2],
         [     E2,      E3,      E3,     -df*E2,      S2],
         [ -df*E1,  -df*E2,  -df*E2, F1 + df**2*E1, -df*S1],
         [     S1,      S2,      S2,     -df*S1,     Jxx]])

gAsf = M([(2/a)*suAxi,
          (2/a)*swAxi,
         -(2/a)*swAxi,
         -(2/a)*(2/a)*swAxixi,
         -(2/a)*(2/b)*swAxieta])
gBsf = M([(2/a)*suBxi,
          (2/a)*swBxi,
         -(2/a)*swBxi,
         -(2/a)*(2/a)*swBxixi,
         -(2/a)*(2/b)*swBxieta])

k0f = (a/2)*bf*gAsf.T*ksf*gBsf

# Geometric stiffness matrix of the flange, due to the axial force Fx

GA = M([(2/a)*swAxi])
GB = M([(2/a)*swBxi])

kG0f = (a/2)*GA.T*M([[Fx]])*GB

# Mass matrix of the flange

kmf44 = kmf55 = bf**2/3 + bf*(h + 2*hb)/2 + (h + 2*hb)**2/4
msf = M([[     1,      0, 0,   -df,      0],
         [     0,      1, 0,     0,    -df],
         [     0,      0, 1,     0,      0],
         [   -df,      0, 0, kmf44,      0],
         [     0,    -df, 0,     0,  kmf55]])

gA5 = M([suA, svA, swA, -(2/a)*swAxi, -(2/b)*swAeta])
gB5 = M([suB, svB, swB, -(2/a)*swBxi, -(2/b)*swBeta])

kMf = (a/2)*hf*bf*rho*gA5.T*msf*gB5

piece_wise_simplify(k0f, (E1, E2, E3, S1, S2, F1, Jxx))
piece_wise_simplify(kG0f, (Fx, ))
piece_wise_simplify(kMf, (rho, ))

# Printing results

from panels.dev.matrixtools import mprint_as_sparse

outdir = './output_expressions_python/'
try: os.makedirs(outdir)
except: pass

matrices = [
    [k0f, 'k0f'],
    [kG0f, 'kG0f'],
    [kMf, 'kMf'],
]

for m in matrices:
    out = mprint_as_sparse(m[0], m[1], '11', print_file=False)
    with open(outdir + 'sympy_%s.txt' % m[1], 'w') as f:
        f.write(out)
