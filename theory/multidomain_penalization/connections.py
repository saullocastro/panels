# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:07:49 2024

@author: Nathan
"""

# Symbolically calculates the matrix elements from a given expression
# In this case, the penalty matrices for the connection between domains

import sys
import os
# sys.path.append(r'C:\Users\natha\Documents\GitHub\panels\theory\multidomain_penalization')
os.chdir(r'C:\Users\natha\Documents\GitHub\panels\theory\multidomain_penalization')

import numpy as np
from sympy import Matrix as M, Symbol, init_printing, var

# Printing results

from sympy import collect, simplify

def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m


var('a1, b1, a2, b2, h, dsb, wxi, weta')
var('kt, kr, kk')

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
kCSSxcte11 =  (b1/2)*kt*(u1A.T*u1B + v1A.T*v1B + w1A.T*w1B + ((2/a1)*w1Axi.T) * ((2/a1)*w1Bxi)*(kr/kt)) + (b1/2)*kk*((4/a1**2)**2)*w1Axixi.T*w1Bxixi
kCSSxcte12 = -(b1/2)*kt*(u1A.T*u2B + v1A.T*v2B + w1A.T*w2B + ((2/a1)*w1Axi.T) * ((2/a2)*w2Bxi)*(kr/kt)) - (b1/2)*kk*((4/a1**2)*(4/a2**2))*w1Axixi.T*w2Bxixi
kCSSxcte22 =  (b1/2)*kt*(u2A.T*u2B + v2A.T*v2B + w2A.T*w2B + ((2/a2)*w2Axi.T) * ((2/a2)*w2Bxi)*(kr/kt)) + (b1/2)*kk*((4/a2**2)**2)*w2Axixi.T*w2Bxixi

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

# NOT NEEDED
# Connection between stiffener' s base and flange 
kCBFycte11 =  (a1/2)*kt*(u1A.T*u1B + v1A.T*v1B + w1A.T*w1B + (kr/kt)*((2/b1)*w1Aeta.T) * ((2/b1)*w1Beta))
kCBFycte12 = -(a1/2)*kt*(u1A.T*u2B + v1A.T*w2B - w1A.T*v2B + (kr/kt)*((2/b1)*w1Aeta.T) * ((2/b2)*w2Beta))
kCBFycte22 =  (a1/2)*kt*(u2A.T*u2B + v2A.T*v2B + w2A.T*w2B + (kr/kt)*((2/b2)*w2Aeta.T) * ((2/b2)*w2Beta))

piece_wise_simplify(kCBFycte11, [])
piece_wise_simplify(kCBFycte12, [])
piece_wise_simplify(kCBFycte22, [])

# Connection between panel skin and stiffener's base - SB CONNECTION
# connection panel-base integrated over xi' and eta'
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
    
    [kCSB11, 'kCSB11'],
    [kCSB12, 'kCSB12'],
    [kCSB22, 'kCSB22'],
]

for m in matrices:
    try:
        out = mprint_as_sparse(m[0], m[1], '11', print_file=False)
    except:
        print(m)
    with open(outdir + 'sympy_%s.txt' % m[1], 'w') as f:
        f.write(out)
