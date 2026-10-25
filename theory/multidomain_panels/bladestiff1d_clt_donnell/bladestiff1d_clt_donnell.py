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
#     S1 = -sum_k y_k h_k (Q16 - Q12 Q26/Q22), y_k of the mid-surface of
#          each ply from the mid-plane of the flange laminate
#     Jxx, the torsion entry of ksf, which receives GJ/bf, with the
#          torsional stiffness GJ = 4 bf D66 of the flange laminate, a thin
#          strip under Saint-Venant torsion
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

var('fAu, gAu, fAv, gAv, fAw, gAw, fAu_xi, fAv_xi, fAw_xi, gAu_eta, gAv_eta, gAw_eta, fAw_xixi, gAw_etaeta')
var('fBu, gBu, fBv, gBv, fBw, gBw, fBu_xi, fBv_xi, fBw_xi, gBu_eta, gBv_eta, gBw_eta, fBw_xixi, gBw_etaeta')


def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m


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

# Constitutive stiffness matrix of the flange

ksf = M([[     E1,      E2,      E2,     -df*E1,      S1],
         [     E2,      E3,      E3,     -df*E2,      S2],
         [     E2,      E3,      E3,     -df*E2,      S2],
         [ -df*E1,  -df*E2,  -df*E2, F1 + df**2*E1, -df*S1],
         [     S1,      S2,      S2,     -df*S1,     Jxx]])

gAsf = M([(2/a)*suA_xi,
          (2/a)*swA_xi,
         -(2/a)*swA_xi,
         -(2/a)*(2/a)*swA_xixi,
         -(2/a)*(2/b)*swA_xieta])
gBsf = M([(2/a)*suB_xi,
          (2/a)*swB_xi,
         -(2/a)*swB_xi,
         -(2/a)*(2/a)*swB_xixi,
         -(2/a)*(2/b)*swB_xieta])

k0f = (a/2)*bf*gAsf.T*ksf*gBsf

# Geometric stiffness matrix of the flange, due to the axial force Fx

GA = M([(2/a)*swA_xi])
GB = M([(2/a)*swB_xi])

kG0f = (a/2)*GA.T*M([[Fx]])*GB

# Mass matrix of the flange

kmf44 = kmf55 = bf**2/3 + bf*(h + 2*hb)/2 + (h + 2*hb)**2/4
msf = M([[     1,      0, 0,   -df,      0],
         [     0,      1, 0,     0,    -df],
         [     0,      0, 1,     0,      0],
         [   -df,      0, 0, kmf44,      0],
         [     0,    -df, 0,     0,  kmf55]])

gA5 = M([suA, svA, swA, -(2/a)*swA_xi, -(2/b)*swA_eta])
gB5 = M([suB, svB, swB, -(2/a)*swB_xi, -(2/b)*swB_eta])

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
