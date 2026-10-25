# Symbolically calculates the matrices of the blade stiffener with a 2D
# formulation for the flange and for the base (panels.stiffener.BladeStiff2D),
# model 'bladestiff2d_clt_donnell'
#
# The stiffener's base (padup) uses the same approximation functions (f, g)
# and degrees of freedom as the panel, with the offset d of its mid-surface,
# and is integrated between y1 = ys - bb/2 and y2 = ys + bb/2 (matrices
# *y1y2). The flange is a plate of length a and width bf, normal to the
# base, with its own approximation functions (r along x, s along the width
# of the flange) and degrees of freedom (matrices *f).
#
# The matrices of the base and of the flange are the same of the shells,
# see theory/shells/cylshell_clpt_donnell and theory/shells/plate_clpt_donnell
# (with intx = a, inty = b or bf), and are calculated by the Shell objects
# BladeStiff2D.base and BladeStiff2D.flange, using the kernels in
# panels/models/cylshell_clpt_donnell.pyx (or plate_clpt_donnell.pyx for a
# flat panel, r -> infinity) and panels/models/plate_clpt_donnell.pyx:
#
#     k0y1y2, k0f: fk0
#     kG0y1y2, kG0f: fkG0
#     kMy1y2, kMf: fkM
#     kAx, kAy, cA: fkAx, fkAy, fcA
#
# The connection between the base (or the panel, when there is no base) at
# y = ys and the edge of the flange at eta_f = -1 is enforced with the
# penalty constants kt and kr, with the flange normal to the base:
#
#     u = uf,  v = wf,  w = -vf,  w,y = wf,y
#
# giving the matrices kCss, kCsf and kCff, whose kernels are in
# panels/stiffener/models/bladestiff2d_clt_donnell.pyx:
#
#     kCss: fkCss,  kCsf: fkCsf,  kCff: fkCff
#
# In the kernels the functions g and s of the connection are evaluated at
# eta(ys) and eta_f = -1, respectively, and only the functions f and r are
# integrated along x
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sympy import Matrix as M, var, collect, simplify

var('a, b, r, bf, h, hf, d, rho, aeromu, beta, gamma')
var('kt, kr')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('Nxx, Nyy, Nxy')

# base (and panel)
var('fAu, gAu, fAv, gAv, fAw, gAw, fAu_xi, fAv_xi, fAw_xi, gAu_eta, gAv_eta, gAw_eta, fAw_xixi, gAw_etaeta')
var('fBu, gBu, fBv, gBv, fBw, gBw, fBu_xi, fBv_xi, fBw_xi, gBu_eta, gBv_eta, gBw_eta, fBw_xixi, gBw_etaeta')

# flange
var('rAu, sAu, rAv, sAv, rAw, sAw, rAu_xi, rAv_xi, rAw_xi, sAu_eta, sAv_eta, sAw_eta, rAw_xixi, sAw_etaeta')
var('rBu, sBu, rBv, sBv, rBw, sBw, rBu_xi, rBv_xi, rBw_xi, sBu_eta, sBv_eta, sBw_eta, rBw_xixi, sBw_etaeta')


def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m


def vectors(f, g, AB):
    r"""Shape functions of the term ``AB`` ('A' or 'B') of a domain with
    approximation functions named ``f`` along x and ``g`` along y"""
    s = lambda fname, gname: (var('%s%s%s' % (f, AB, fname))
                              *var('%s%s%s' % (g, AB, gname)))
    return dict(
        u=M([[s('u', 'u'), 0, 0]]),
        v=M([[0, s('v', 'v'), 0]]),
        w=M([[0, 0, s('w', 'w')]]),
        u_xi=M([[s('u_xi', 'u'), 0, 0]]),
        v_xi=M([[0, s('v_xi', 'v'), 0]]),
        w_xi=M([[0, 0, s('w_xi', 'w')]]),
        u_eta=M([[s('u', 'u_eta'), 0, 0]]),
        v_eta=M([[0, s('v', 'v_eta'), 0]]),
        w_eta=M([[0, 0, s('w', 'w_eta')]]),
        w_xixi=M([[0, 0, s('w_xixi', 'w')]]),
        w_xieta=M([[0, 0, s('w_xi', 'w_eta')]]),
        w_etaeta=M([[0, 0, s('w', 'w_etaeta')]]),
        )


sA, sB = vectors('f', 'g', 'A'), vectors('f', 'g', 'B')
sAf, sBf = vectors('r', 's', 'A'), vectors('r', 's', 'B')

F = M([[A11, A12, A16, B11, B12, B16],
       [A12, A22, A26, B12, B22, B26],
       [A16, A26, A66, B16, B26, B66],
       [B11, B12, B16, D11, D12, D16],
       [B12, B22, B26, D12, D22, D26],
       [B16, B26, B66, D16, D26, D66]])

Nmat = M([[Nxx, Nxy],
          [Nxy, Nyy]])


def B0(s, a, b, r=None):
    r"""Linear strain-displacement matrix, Donnell's cylindrical shell for
    ``r`` given, plate otherwise"""
    curv = s['w']/r if r is not None else 0*s['w']
    return M([(2/a)*s['u_xi'],
              (2/b)*s['v_eta'] + curv,
              (2/b)*s['u_eta'] + (2/a)*s['v_xi'],
             -(2/a)*(2/a)*s['w_xixi'],
             -(2/b)*(2/b)*s['w_etaeta'],
             -(2/a)*(2/b)*2*s['w_xieta']])


def G(s, a, b):
    return M([(2/a)*s['w_xi'], (2/b)*s['w_eta']])


def g5(s, a, b):
    return M([s['u'], s['v'], s['w'], -(2/a)*s['w_xi'], -(2/b)*s['w_eta']])


def maux(h, d):
    return M([[ 1,  0, 0,             -d,              0],
              [ 0,  1, 0,              0,             -d],
              [ 0,  0, 1,              0,              0],
              [-d,  0, 0, (h**2/12 + d**2),            0],
              [ 0, -d, 0,              0, (h**2/12 + d**2)]])


# Base of the stiffener, with offset d

k0y1y2 = (a*b/4)*B0(sA, a, b, r).T*F*B0(sB, a, b, r)
kG0y1y2 = (a*b/4)*G(sA, a, b).T*Nmat*G(sB, a, b)
kMy1y2 = (a*b/4)*h*rho*g5(sA, a, b).T*maux(h, d)*g5(sB, a, b)

# Aerodynamic and damping matrices using piston's theory

kAx = -(a*b/4)*(beta*(2/a)*sA['w_xi'].T*sB['w'] + gamma*sA['w'].T*sB['w'])
kAy = -(a*b/4)*beta*(2/b)*sA['w_eta'].T*sB['w']
cA = -(a*b/4)*aeromu*sA['w'].T*sB['w']

# Flange, with offset d

k0f = (a*bf/4)*B0(sAf, a, bf).T*F*B0(sBf, a, bf)
kG0f = (a*bf/4)*G(sAf, a, bf).T*Nmat*G(sBf, a, bf)
kMf = (a*bf/4)*hf*rho*g5(sAf, a, bf).T*maux(hf, d)*g5(sBf, a, bf)

# Penalty connection between the base (s) and the flange (f)
#
#     U = 1/2 int_0^a [kt (u - uf)^2 + kt (v - wf)^2 + kt (w + vf)^2
#                      + kr (w,y - wf,y)^2] dx
#
# only the upper triangle of the connectivity matrix is needed, the block
# kCfs = kCsf^T is not written

kCss = (a/2)*(kt*sA['u'].T*sB['u']
            + kt*sA['v'].T*sB['v']
            + kt*sA['w'].T*sB['w']
            + kr*((2/b)*sA['w_eta']).T*((2/b)*sB['w_eta']))

kCsf = (a/2)*(-kt*sA['u'].T*sBf['u']
              -kt*sA['v'].T*sBf['w']
              +kt*sA['w'].T*sBf['v']
              -kr*((2/b)*sA['w_eta']).T*((2/bf)*sBf['w_eta']))

kCff = (a/2)*(kt*sAf['u'].T*sBf['u']
            + kt*sAf['v'].T*sBf['v']
            + kt*sAf['w'].T*sBf['w']
            + kr*((2/bf)*sAf['w_eta']).T*((2/bf)*sBf['w_eta']))

vars = (A11, A12, A16, A22, A26, A66,
        B11, B12, B16, B22, B26, B66,
        D11, D12, D16, D22, D26, D66)

piece_wise_simplify(k0y1y2, vars)
piece_wise_simplify(kG0y1y2, (Nxx, Nyy, Nxy))
piece_wise_simplify(kMy1y2, (rho, ))
piece_wise_simplify(k0f, vars)
piece_wise_simplify(kG0f, (Nxx, Nyy, Nxy))
piece_wise_simplify(kMf, (rho, ))
for m in (kAx, kAy, cA, kCss, kCsf, kCff):
    piece_wise_simplify(m, [])

# Printing results

from panels.dev.matrixtools import mprint_as_sparse

outdir = './output_expressions_python/'
try: os.makedirs(outdir)
except: pass

matrices = [
    [k0y1y2, 'k0y1y2'],
    [kG0y1y2, 'kG0y1y2'],
    [kMy1y2, 'kMy1y2'],
    [kAx, 'kAx'],
    [kAy, 'kAy'],
    [cA, 'cA'],
    [k0f, 'k0f'],
    [kG0f, 'kG0f'],
    [kMf, 'kMf'],
    [kCss, 'kCss'],
    [kCsf, 'kCsf'],
    [kCff, 'kCff'],
]

for m in matrices:
    out = mprint_as_sparse(m[0], m[1], '11', print_file=False)
    with open(outdir + 'sympy_%s.txt' % m[1], 'w') as f:
        f.write(out)
