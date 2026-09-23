# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:07:49 2024

@author: Nathan
"""

# Symbolically calculates the matrix elements from a given expression
# In this case, the penalty matrices for the connection between domains
#
# The penalty energy of a connection is
#
#     U = 1/2 sum_k k_k int (q_k^1 - q_k^2)^2
#
# with the penalty constants k_k (kt for translations, kr for rotations) and
# the quantities q_k^1 and q_k^2 of each domain that must be compatible.
# Expanding the square gives the blocks 11, 12 and 22 of the connection
# matrix, the block 21 = 12^T is not written since only the upper triangle
# of the matrix is assembled.
#
# The kernels in panels/multidomain/connections/ that correspond to each
# matrix are:
#
#   kCSSxcte, kCSSycte, kCBFycte, kCBFxcte, kCSB: kCSS*.pyx, kCBF*.pyx, kCSB.pyx
#   kCSSxcte_sdt, kCSSycte_sdt, kCSB_sdt: kCsdt.py, for the models based on
#   shear deformation theories (FSDT and TSDT) with 5 DOFs per term
#   kCBFycte_sdt, kCBFxcte_sdt: the functions *_sdt of kCBF*.pyx, for the
#   same models
#
# The kernels kCpd.pyx and kCSB_dmg.pyx are not derived here.

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from sympy import Matrix as M, Symbol, init_printing, var

# Printing results

from sympy import collect, simplify

def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m


var('a1, b1, a2, b2, h, dsb, wxi, weta')
var('kt, kr')

# Displacements and rotations for panel 1 (or ith panel in eq 32 - http://dx.doi.org/10.1016/j.compstruct.2016.10.026)
var('f1Au, g1Au, f1Av, g1Av, f1Aw, g1Aw, f1Auxi, f1Avxi, f1Awxi, g1Aueta, g1Aveta, g1Aweta, f1Awxixi, g1Awetaeta')
var('f1Bu, g1Bu, f1Bv, g1Bv, f1Bw, g1Bw, f1Buxi, f1Bvxi, f1Bwxi, g1Bueta, g1Bveta, g1Bweta, f1Bwxixi, g1Bwetaeta')
    # A and B are when 2 of the same terms are present i.e. u^2 then it becomes uA*uB so that the matrix can be populated independently

var('u1A, v1A, w1A, u1Axi, v1Axi, w1Axi, u1Aeta, v1Aeta, w1Aeta, w1Axixi, w1Axieta, w1Aetaeta')
var('u1B, v1B, w1B, u1Bxi, v1Bxi, w1Bxi, u1Beta, v1Beta, w1Beta, w1Bxixi, w1Bxieta, w1Betaeta')

# Displacements and rotations for panel 2 (or jth panel in eq 32 - http://dx.doi.org/10.1016/j.compstruct.2016.10.026)
var('f2Au, g2Au, f2Av, g2Av, f2Aw, g2Aw, f2Auxi, f2Avxi, f2Awxi, g2Aueta, g2Aveta, g2Aweta, f2Awxixi, g2Awetaeta')
var('f2Bu, g2Bu, f2Bv, g2Bv, f2Bw, g2Bw, f2Buxi, f2Bvxi, f2Bwxi, g2Bueta, g2Bveta, g2Bweta, f2Bwxixi, g2Bwetaeta')

var('u2A, v2A, w2A, u2Axi, v2Axi, w2Axi, u2Aeta, v2Aeta, w2Aeta, w2Axixi, w2Axieta, w2Aetaeta')
var('u2B, v2B, w2B, u2Bxi, v2Bxi, w2Bxi, u2Beta, v2Beta, w2Beta, w2Bxixi, w2Bxieta, w2Betaeta')

######## IMPORTANT ########################
# All of this should be su1A NOT u1A - As it alreay has removed the c and del_c from the equations
# That is why when the equations for K are written below it directly has u,v,w instead of su, sv, sw etc
# It should be like the KC and KG terms!

# PANEL 1 - A terms
# SF already in terms of xi and eta
u1A = M([[f1Au*g1Au, 0, 0]])
v1A = M([[0, f1Av*g1Av, 0]])
w1A = M([[0, 0, f1Aw*g1Aw]])

# u1Axi is deri of suA wrt xi
w1Axi = M([[0, 0, f1Awxi*g1Aw]])
w1Aeta = M([[0, 0, f1Aw*g1Aweta]])

w1Axixi = M([[0, 0, f1Awxixi*g1Aw]])
w1Aetaeta = M([[0, 0, f1Aw*g1Awetaeta]])

# PANEL 1 - B terms
u1B = M([[f1Bu*g1Bu, 0, 0]])
v1B = M([[0, f1Bv*g1Bv, 0]])
w1B = M([[0, 0, f1Bw*g1Bw]])

w1Bxi = M([[0, 0, f1Bwxi*g1Bw]])
w1Beta = M([[0, 0, f1Bw*g1Bweta]])

w1Bxixi = M([[0, 0, f1Bwxixi*g1Bw]])
w1Betaeta = M([[0, 0, f1Bw*g1Bwetaeta]])


# PANEL 2 - A terms
u2A = M([[f2Au*g2Au, 0, 0]])
v2A = M([[0, f2Av*g2Av, 0]])
w2A = M([[0, 0, f2Aw*g2Aw]])

w2Axi = M([[0, 0, f2Awxi*g2Aw]])
w2Aeta = M([[0, 0, f2Aw*g2Aweta]])

w2Axixi = M([[0, 0, f2Awxixi*g2Aw]])
w2Aetaeta = M([[0, 0, f2Aw*g2Awetaeta]])

# PANEL 2 - B terms
u2B = M([[f2Bu*g2Bu, 0, 0]])
v2B = M([[0, f2Bv*g2Bv, 0]])
w2B = M([[0, 0, f2Bw*g2Bw]])

w2Bxi = M([[0, 0, f2Bwxi*g2Bw]])
w2Beta = M([[0, 0, f2Bw*g2Bweta]])

w2Bxixi = M([[0, 0, f2Bwxixi*g2Bw]])
w2Betaeta = M([[0, 0, f2Bw*g2Bwetaeta]])


######## IMPORTANT ########################
# All of this should be su1A NOT u1A - As it alreay has removed the c and del_c from the equations
# That is why when the equations for K are written below it directly has u,v,w instead of su, sv, sw etc
# It should be like the KC and KG terms!
# So the kC terms here already have del_c*del_c removed


# Note below that in (u1 - u2')^2 = (u1*u1 - u1*u2 - u2*u1 + u2*u2^2)
  # we only want the upper triangle of the connectivity matrix, thus -u2*u1 is ignored
  
# XCTE Penalty Matrix
#NOTE the curvature penalty kk*(w1,xx - w2,xx)^2 was removed from the
#     kernels, see commit 6be356b, and is therefore not included here
kCSSxcte11 =  (b1/2)*kt*(u1A.T*u1B + v1A.T*v1B + w1A.T*w1B + ((2/a1)*w1Axi.T) * ((2/a1)*w1Bxi)*(kr/kt))
kCSSxcte12 = -(b1/2)*kt*(u1A.T*u2B + v1A.T*v2B + w1A.T*w2B + ((2/a1)*w1Axi.T) * ((2/a2)*w2Bxi)*(kr/kt))
kCSSxcte22 =  (b1/2)*kt*(u2A.T*u2B + v2A.T*v2B + w2A.T*w2B + ((2/a2)*w2Axi.T) * ((2/a2)*w2Bxi)*(kr/kt))

piece_wise_simplify(kCSSxcte11, [])
piece_wise_simplify(kCSSxcte12, [])
piece_wise_simplify(kCSSxcte22, [])

# YCTE Penalty Matrix 
kCSSycte11 =  (a1/2)*kt*(u1A.T*u1B + v1A.T*v1B + w1A.T*w1B + ((2/b1)*w1Aeta.T) * ((2/b1)*w1Beta)*(kr/kt))
kCSSycte12 = -(a1/2)*kt*(u1A.T*u2B + v1A.T*v2B + w1A.T*w2B + ((2/b1)*w1Aeta.T) * ((2/b2)*w2Beta)*(kr/kt))
kCSSycte22 =  (a1/2)*kt*(u2A.T*u2B + v2A.T*v2B + w2A.T*w2B + ((2/b2)*w2Aeta.T) * ((2/b2)*w2Beta)*(kr/kt))

piece_wise_simplify(kCSSycte11, [])
piece_wise_simplify(kCSSycte12, [])
piece_wise_simplify(kCSSycte22, [])

# Connection between stiffener's base (1) and flange (2) at y1 = ycte1 and
# y2 = ycte2, with the flange normal to the base: v1 = w2, w1 = -v2 and
# omega1 = omega2, where omega is the rotation of the normal about x, with
# the local axes of the flange rotated by 90 degrees about x. For the CLPT
# omega = -phiy = w,y - v/r with the Sanders-Koiter kinematics, and omega =
# w,y for flat plates and the Donnell kinematics, obtained with rinv = 1/r =
# 0. rinv1 and rinv2 are the arguments of the kernels of kCBFycte.pyx. For
# the FSDT and TSDT omega = -phiy, see the section of these models below
var('rinv1, rinv2')
om1A = (2/b1)*w1Aeta - rinv1*v1A
om1B = (2/b1)*w1Beta - rinv1*v1B
om2A = (2/b2)*w2Aeta - rinv2*v2A
om2B = (2/b2)*w2Beta - rinv2*v2B
kCBFycte11=  (a1/2)*kt*(u1A.T*u1B + v1A.T*v1B + w1A.T*w1B + (kr/kt)*om1A.T*om1B)
kCBFycte12 = -(a1/2)*kt*(u1A.T*u2B + v1A.T*w2B - w1A.T*v2B + (kr/kt)*om1A.T*om2B)
kCBFycte22 =  (a1/2)*kt*(u2A.T*u2B + v2A.T*v2B + w2A.T*w2B + (kr/kt)*om2A.T*om2B)

piece_wise_simplify(kCBFycte11, [])
piece_wise_simplify(kCBFycte12, [])
piece_wise_simplify(kCBFycte22, [])

# Connection between stiffener's base (1) and flange (2) at x1 = xcte1 and
# x2 = xcte2, with the flange normal to the base: u1 = w2, w1 = -u2 and
# omega1 = omega2, where omega is the rotation of the normal about y, with
# the local axes of the flange rotated by 90 degrees about y. For the CLPT
# omega = phix = -w,x, also with the Sanders-Koiter kinematics. For the
# FSDT and TSDT omega = phix, see the section of these models below
kCBFxcte11 =  (b1/2)*kt*(u1A.T*u1B + v1A.T*v1B + w1A.T*w1B + (kr/kt)*((2/a1)*w1Axi.T) * ((2/a1)*w1Bxi))
kCBFxcte12 = -(b1/2)*kt*(u1A.T*w2B + v1A.T*v2B - w1A.T*u2B + (kr/kt)*((2/a1)*w1Axi.T) * ((2/a2)*w2Bxi))
kCBFxcte22 =  (b1/2)*kt*(u2A.T*u2B + v2A.T*v2B + w2A.T*w2B + (kr/kt)*((2/a2)*w2Axi.T) * ((2/a2)*w2Bxi))

piece_wise_simplify(kCBFxcte11, [])
piece_wise_simplify(kCBFxcte12, [])
piece_wise_simplify(kCBFxcte22, [])

# Connection between panel skin and stiffener's base - SB CONNECTION
# connection panel-base integrated over xi' and eta'
# The top panel (1) is extrapolated to the mid-surface of the bottom panel
# (2), at a distance dsb = h1/2 + h2/2 below the mid-surface of (1):
#     u1 + dsb*w1,x = u2,  v1 + dsb*w1,y = v2,  w1 = w2
kCSB11 = (a1*b1/4)*kt*(u1A.T*u1B + v1A.T*v1B + w1A.T*w1B + u1A.T*w1Bxi*(2*dsb/a1) + w1Axi.T*u1B*(2*dsb/a1) + v1A.T*w1Beta*(2*dsb/b1) + w1Aeta.T*v1B*(2*dsb/b1) + w1Axi.T*w1Bxi*(2*dsb/a1)*(2*dsb/a1) + w1Aeta.T*w1Beta*(2*dsb/b1)*(2*dsb/b1))
    # Contains all squared terms - so all terms of (u + d..w,xi)^2 comes here and so on 
                                    # thats split up: u^2 + u*d..w,xi + d..w,xi*u + (d..w,xi)^2
############ THERE IS SOME EXTRA c1 HERE - removed ##############################
kCSB12 = -(a1*b1/4)*kt*(u1A.T*u2B + v1A.T*v2B + w1A.T*w2B + w1Axi.T*u2B*(2*dsb/a1) + w1Aeta.T*v2B*(2*dsb/b1))
# Instead of 2ab only ab is presented in _12 as ba goes into _21 which is symmetric and is hence not added here
kCSB22 =  (a1*b1/4)*kt*(u2A.T*u2B + v2A.T*v2B + w2A.T*w2B)

piece_wise_simplify(kCSB11, [])
piece_wise_simplify(kCSB12, [])
piece_wise_simplify(kCSB22, [])


# MODELS BASED ON SHEAR DEFORMATION THEORIES (FSDT and TSDT)
# 5 DOFs per term u, v, w, phix, phiy, with the displacement field
#
#     u(z) = u + z phix - c1 z^3 (phix + w,x)
#     v(z) = v + z Phiy - c1 z^3 (phiy + w,y)
#     w(z) = w
#
# where c1 = 0 for the FSDT and c1 = 4/(3 h^2) for the TSDT, and Phiy =
# phiy + v/r is the rotation of the normal about x with the Sanders-Koiter
# kinematics, see theory/shells/fsdt_tsdt, obtained with rinv1 = 1/r1 and
# rinv2 = 1/r2 of each panel, and Phiy = phiy with rinv = 0 otherwise
var('krw, cphi1, cw1, cphi2, cw2, zi1, zi2')
var('f1Aphix, g1Aphix, f1Aphiy, g1Aphiy, f1Bphix, g1Bphix, f1Bphiy, g1Bphiy')
var('f2Aphix, g2Aphix, f2Aphiy, g2Aphiy, f2Bphix, g2Bphix, f2Bphiy, g2Bphiy')

def sdt_vectors(p, s):
    r"""Shape functions of panel ``p`` (1 or 2) and term ``s`` (A or B)"""
    f = lambda name: Symbol('f%d%s%s' % (p, s, name))
    g = lambda name: Symbol('g%d%s%s' % (p, s, name))
    return dict(
        u=M([[f('u')*g('u'), 0, 0, 0, 0]]),
        v=M([[0, f('v')*g('v'), 0, 0, 0]]),
        w=M([[0, 0, f('w')*g('w'), 0, 0]]),
        wxi=M([[0, 0, f('wxi')*g('w'), 0, 0]]),
        weta=M([[0, 0, f('w')*g('weta'), 0, 0]]),
        phix=M([[0, 0, 0, f('phix')*g('phix'), 0]]),
        phiy=M([[0, 0, 0, 0, f('phiy')*g('phiy')]]),
        )

s1A, s1B = sdt_vectors(1, 'A'), sdt_vectors(1, 'B')
s2A, s2B = sdt_vectors(2, 'A'), sdt_vectors(2, 'B')

def Phiy(s, p):
    r"""Rotation of the normal about x of panel ``p``"""
    return s['phiy'] + (rinv1 if p == 1 else rinv2)*s['v']

# XCTE and YCTE: penalty kt on u, v, w, kr on phix, Phiy and, for the TSDT,
# krw = kr on the derivative of w normal to the edge, which enters the
# displacement field, krw = 0 for the FSDT
def sdt_edge(sA, pA, sB, pB, jac, normal, LA, LB):
    return jac*(kt*(sA['u'].T*sB['u'] + sA['v'].T*sB['v'] + sA['w'].T*sB['w'])
              + kr*(sA['phix'].T*sB['phix'] + Phiy(sA, pA).T*Phiy(sB, pB))
              + krw*((2/LA)*sA[normal].T)*((2/LB)*sB[normal]))

kCSSxcte_sdt11 =  sdt_edge(s1A, 1, s1B, 1, b1/2, 'wxi', a1, a1)
kCSSxcte_sdt12 = -sdt_edge(s1A, 1, s2B, 2, b1/2, 'wxi', a1, a2)
kCSSxcte_sdt22 =  sdt_edge(s2A, 2, s2B, 2, b1/2, 'wxi', a2, a2)

kCSSycte_sdt11 =  sdt_edge(s1A, 1, s1B, 1, a1/2, 'weta', b1, b1)
kCSSycte_sdt12 = -sdt_edge(s1A, 1, s2B, 2, a1/2, 'weta', b1, b2)
kCSSycte_sdt22 =  sdt_edge(s2A, 2, s2B, 2, a1/2, 'weta', b2, b2)

# SB: penalty kt on the displacements at the interface, z = -h1/2 for the
# top panel (1) and z = +h2/2 for the bottom panel (2), and kr on the
# difference of rotations (kr = 0 connects only the interface)
#
#     u_p(z) = u_p + cphi_p phix_p + cw_p w_p,x
#     v_p(z) = (1 + zi_p rinv_p) v_p + cphi_p phiy_p + cw_p w_p,y
#
# with zi_p the coordinate z of the interface in panel p, cphi_p = zi_p -
# c1_p zi_p^3 and cw_p = -c1_p zi_p^3, which for the TSDT give cphi1 =
# -h1/3, cw1 = h1/6, cphi2 = h2/3, cw2 = -h2/6 and for the FSDT cphi1 =
# -h1/2, cphi2 = h2/2, cw1 = cw2 = 0, and zi1 = -h1/2, zi2 = h2/2
def sdt_interface(s, p):
    cphi, cw, zi, rinv = ((cphi1, cw1, zi1, rinv1) if p == 1 else
                          (cphi2, cw2, zi2, rinv2))
    L = dict(x=a1 if p == 1 else a2, y=b1 if p == 1 else b2)
    qu = s['u'] + cphi*s['phix'] + cw*(2/L['x'])*s['wxi']
    qv = (1 + zi*rinv)*s['v'] + cphi*s['phiy'] + cw*(2/L['y'])*s['weta']
    return qu, qv

def sdt_sb(sA, pA, sB, pB):
    quA, qvA = sdt_interface(sA, pA)
    quB, qvB = sdt_interface(sB, pB)
    return (a1*b1/4)*(kt*(quA.T*quB + qvA.T*qvB + sA['w'].T*sB['w'])
                    + kr*(sA['phix'].T*sB['phix'] + Phiy(sA, pA).T*Phiy(sB, pB)))

kCSB_sdt11 =  sdt_sb(s1A, 1, s1B, 1)
kCSB_sdt12 = -sdt_sb(s1A, 1, s2B, 2)
kCSB_sdt22 =  sdt_sb(s2A, 2, s2B, 2)

# BFycte and BFxcte: base (1) and flange (2), as for the CLPT above, with
# the rotation penalty on the rotations of the normals Phiy (BFycte) and
# phix (BFxcte) instead of -w,y (+ v/r for Sanders) and -w,x. For the TSDT
# the derivative of w normal to the connection is not penalized, at a
# T-joint only the rotation of the normals is common to both panels. The
# other rotation of each panel is a drilling rotation of the other panel and
# is not penalized
def sdt_bfycte(sA, pA, sB, pB, flangeB):
    vB, wB = (sB['w'], -sB['v']) if flangeB else (sB['v'], sB['w'])
    return (a1/2)*(kt*(sA['u'].T*sB['u'] + sA['v'].T*vB + sA['w'].T*wB)
                 + kr*Phiy(sA, pA).T*Phiy(sB, pB))

def sdt_bfxcte(sA, sB, flangeB):
    uB, wB = (sB['w'], -sB['u']) if flangeB else (sB['u'], sB['w'])
    return (b1/2)*(kt*(sA['u'].T*uB + sA['v'].T*sB['v'] + sA['w'].T*wB)
                 + kr*sA['phix'].T*sB['phix'])

kCBFycte_sdt11 =  sdt_bfycte(s1A, 1, s1B, 1, False)
kCBFycte_sdt12 = -sdt_bfycte(s1A, 1, s2B, 2, True)
kCBFycte_sdt22 =  sdt_bfycte(s2A, 2, s2B, 2, False)

kCBFxcte_sdt11 =  sdt_bfxcte(s1A, s1B, False)
kCBFxcte_sdt12 = -sdt_bfxcte(s1A, s2B, True)
kCBFxcte_sdt22 =  sdt_bfxcte(s2A, s2B, False)

for m in (kCSSxcte_sdt11, kCSSxcte_sdt12, kCSSxcte_sdt22,
          kCSSycte_sdt11, kCSSycte_sdt12, kCSSycte_sdt22,
          kCSB_sdt11, kCSB_sdt12, kCSB_sdt22):
    piece_wise_simplify(m, [])



# Printing results
from panels.dev.matrixtools import mprint_as_sparse

outdir = './output_expressions_python_new/'
import os
try: os.makedirs(outdir)
except: pass

matrices = [
    [kCSSxcte11, 'kCSSxcte11'],
    [kCSSxcte12, 'kCSSxcte12'],
    [kCSSxcte22, 'kCSSxcte22'],
    
    [kCSSycte11, 'kCSSycte11'],
    [kCSSycte12, 'kCSSycte12'],
    [kCSSycte22, 'kCSSycte22'],
    
    [kCBFycte11, 'kCBFycte11'],
    [kCBFycte12, 'kCBFycte12'],
    [kCBFycte22, 'kCBFycte22'],

    [kCBFxcte11, 'kCBFxcte11'],
    [kCBFxcte12, 'kCBFxcte12'],
    [kCBFxcte22, 'kCBFxcte22'],

    [kCSB11, 'kCSB11'],
    [kCSB12, 'kCSB12'],
    [kCSB22, 'kCSB22'],

    [kCSSxcte_sdt11, 'kCSSxcte_sdt11'],
    [kCSSxcte_sdt12, 'kCSSxcte_sdt12'],
    [kCSSxcte_sdt22, 'kCSSxcte_sdt22'],

    [kCSSycte_sdt11, 'kCSSycte_sdt11'],
    [kCSSycte_sdt12, 'kCSSycte_sdt12'],
    [kCSSycte_sdt22, 'kCSSycte_sdt22'],

    [kCSB_sdt11, 'kCSB_sdt11'],
    [kCSB_sdt12, 'kCSB_sdt12'],
    [kCSB_sdt22, 'kCSB_sdt22'],

    [kCBFycte_sdt11, 'kCBFycte_sdt11'],
    [kCBFycte_sdt12, 'kCBFycte_sdt12'],
    [kCBFycte_sdt22, 'kCBFycte_sdt22'],

    [kCBFxcte_sdt11, 'kCBFxcte_sdt11'],
    [kCBFxcte_sdt12, 'kCBFxcte_sdt12'],
    [kCBFxcte_sdt22, 'kCBFxcte_sdt22'],
]

for m in matrices:
    try:
        out = mprint_as_sparse(m[0], m[1], '11', print_file=False)
    except:
        print(m)
    with open(outdir + 'sympy_%s.txt' % m[1], 'w') as f:
        f.write(out)
