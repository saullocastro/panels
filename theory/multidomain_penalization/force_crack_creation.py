# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:07:49 2024

@author: Nathan
"""

# Symbolically calculates the matrix elements from a given expression
# In this case, terms in the TPE that deal with the creation of a crack

import sys
import os
# sys.path.append(r'C:\Users\natha\Documents\GitHub\panels\theory\multidomain_penalization')


import numpy as np
from sympy import Matrix as M, Symbol, init_printing, var

# Printing results

from sympy import collect, simplify

def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m

#%% 

var('a_1, b_1, a_2, b_2, weight')
var('del_i_1, k_i_1, del_i, k_i, k_o, del_f, del_o')
var('w1A, w1B, w1C, w2A, w2B, w2C, c1, c2')

# Displacements and rotations for panel 1 (or ith panel in eq 32 - http://dx.doi.org/10.1016/j.compstruct.2016.10.026)
var('f1Au, g1Au, f1Av, g1Av, f1Aw, g1Aw, f1Au_xi, f1Av_xi, f1Aw_xi, g1Au_eta, g1Av_eta, g1Aw_eta, f1Aw_xixi, g1Aw_etaeta')
var('f1Bu, g1Bu, f1Bv, g1Bv, f1Bw, g1Bw, f1Bu_xi, f1Bv_xi, f1Bw_xi, g1Bu_eta, g1Bv_eta, g1Bw_eta, f1Bw_xixi, g1Bw_etaeta')
var('f1Cu, g1Cu, f1Cv, g1Cv, f1Cw, g1Cw, f1Cu_xi, f1Cv_xi, f1Cw_xi, g1Cu_eta, g1Cv_eta, g1Cw_eta, f1Cw_xixi, g1Cw_etaeta')
    # A and B are when 2 of the same terms are present i.e. u^2 then it becomes uA*uB so that the matrix can be populated independently

var('u1A, v1A, w1A, u1A_xi, v1A_xi, w1A_xi, u1A_eta, v1A_eta, w1A_eta, w1A_xixi, w1A_xieta, w1A_etaeta')
var('u1B, v1B, w1B, u1B_xi, v1B_xi, w1B_xi, u1B_eta, v1B_eta, w1B_eta, w1B_xixi, w1B_xieta, w1B_etaeta')

# Displacements and rotations for panel 2 (or jth panel in eq 32 - http://dx.doi.org/10.1016/j.compstruct.2016.10.026)
var('f2Au, g2Au, f2Av, g2Av, f2Aw, g2Aw, f2Au_xi, f2Av_xi, f2Aw_xi, g2Au_eta, g2Av_eta, g2Aw_eta, f2Aw_xixi, g2Aw_etaeta')
var('f2Bu, g2Bu, f2Bv, g2Bv, f2Bw, g2Bw, f2Bu_xi, f2Bv_xi, f2Bw_xi, g2Bu_eta, g2Bv_eta, g2Bw_eta, f2Bw_xixi, g2Bw_etaeta')
var('f2Cu, g2Cu, f2Cv, g2Cv, f2Cw, g2Cw, f2Cu_xi, f2Cv_xi, f2Cw_xi, g2Cu_eta, g2Cv_eta, g2Cw_eta, f2Cw_xixi, g2Cw_etaeta')

var('u2A, v2A, w2A, u2A_xi, v2A_xi, w2A_xi, u2A_eta, v2A_eta, w2A_eta, w2A_xixi, w2A_xieta, w2A_etaeta')
var('u2B, v2B, w2B, u2B_xi, v2B_xi, w2B_xi, u2B_eta, v2B_eta, w2B_eta, w2B_xixi, w2B_xieta, w2B_etaeta')

var('c1u, c1v, c1w')

######## IMPORTANT ########################
# All of this should be su1A NOT u1A - As it alreay has removed the c and del_c from the equations
# That is why when the equations for K are written below it directly has u,v,w instead of su, sv, sw etc
# It should be like the KC and KG terms!

# PANEL 1 - A terms
# SF already in terms of xi and eta
u1A = M([[f1Au*g1Au, 0, 0]])
v1A = M([[0, f1Av*g1Av, 0]])
w1A = M([[0, 0, f1Aw*g1Aw]])

# u1A_xi is deri of suA wrt xi
w1A_xi = M([[0, 0, f1Aw_xi*g1Aw]])
w1A_eta = M([[0, 0, f1Aw*g1Aw_eta]])

w1A_xixi = M([[0, 0, f1Aw_xixi*g1Aw]])
w1A_etaeta = M([[0, 0, f1Aw*g1Aw_etaeta]])

# PANEL 1 - B terms
u1B = M([[f1Bu*g1Bu, 0, 0]])
v1B = M([[0, f1Bv*g1Bv, 0]])
w1B = M([[0, 0, f1Bw*g1Bw]])

w1B_xi = M([[0, 0, f1Bw_xi*g1Bw]])
w1B_eta = M([[0, 0, f1Bw*g1Bw_eta]])

w1B_xixi = M([[0, 0, f1Bw_xixi*g1Bw]])
w1B_etaeta = M([[0, 0, f1Bw*g1Bw_etaeta]])

# PANEL 1 - C terms
u1C = M([[f1Cu*g1Cu, 0, 0]])
v1C = M([[0, f1Cv*g1Cv, 0]])
w1C = M([[0, 0, f1Cw*g1Cw]])

# PANEL 2 - A terms
u2A = M([[f2Au*g2Au, 0, 0]])
v2A = M([[0, f2Av*g2Av, 0]])
w2A = M([[0, 0, f2Aw*g2Aw]])

w2A_xi = M([[0, 0, f2Aw_xi*g2Aw]])
w2A_eta = M([[0, 0, f2Aw*g2Aw_eta]])

w2A_xixi = M([[0, 0, f2Aw_xixi*g2Aw]])
w2A_etaeta = M([[0, 0, f2Aw*g2Aw_etaeta]])

# PANEL 2 - B terms
u2B = M([[f2Bu*g2Bu, 0, 0]])
v2B = M([[0, f2Bv*g2Bv, 0]])
w2B = M([[0, 0, f2Bw*g2Bw]])

w2B_xi = M([[0, 0, f2Bw_xi*g2Bw]])
w2B_eta = M([[0, 0, f2Bw*g2Bw_eta]])

w2B_xixi = M([[0, 0, f2Bw_xixi*g2Bw]])
w2B_etaeta = M([[0, 0, f2Bw*g2Bw_etaeta]])

# PANEL 2 - C terms
u2C = M([[f2Cu*g2Cu, 0, 0]])
v2C = M([[0, f2Cv*g2Cv, 0]])
w2C = M([[0, 0, f2Cw*g2Cw]])

# c1
c1 = M([[c1u, c1v, c1w]])



######## IMPORTANT ########################
# All of this should be su1A NOT u1A - As it alreay has removed the c and del_c from the equations
# That is why when the equations for K are written below it directly has u,v,w instead of su, sv, sw etc
# It should be like the KC and KG terms!
# So the kC terms here already have del_c*del_c removed


# Note below that in (u1 - u2')^2 = (u1*u1 - u1*u2 - u2*u1 + u2*u2^2)
  # we only want the upper triangle of the connectivity matrix, thus -u2*u1 is ignored
  
# 1st term of the fcrack vector 
f_crack_1st_term_p1 = weight*(a_1*b_1/4)*(del_i_1/2)*(k_i_1 - k_i)*(w1A)
f_crack_1st_term_p2 = weight*(a_2*b_2/4)*(del_i_1/2)*(k_i_1 - k_i)*(-w2A)

piece_wise_simplify(f_crack_1st_term_p1, [])
piece_wise_simplify(f_crack_1st_term_p2, [])
  
# Stiffness matrix for the crack creation
Q = (k_o*del_o)*del_f/(del_f - del_o)
# Only terms of panel 1
k_crack11 = weight*(a_1*b_1/4) * ((del_i_1*Q)/(del_i)**2) *  (w1A.T*w1B)
# Coupling terms of panel 1 and 2 - ab modelled as 2 in 2ab comes when its made symmetric like 12 and 21 positions
k_crack12 = weight*(a_1*b_1/4) * ((del_i_1*Q)/(del_i)**2) * (-w1A.T*w2B)
# Terms of panel 2
k_crack22 = weight*(a_2*b_2/4) * ((del_i_1*Q)/(del_i)**2) *  (w2A.T*w2B)

piece_wise_simplify(k_crack11, [])
piece_wise_simplify(k_crack12, [])
piece_wise_simplify(k_crack22, [])

k_crack_term2_11 = weight*(a_2*b_2/4) * ((del_i_1*Q)/(del_i)**3) * w1A.T*w1B*c1.T*w1C
piece_wise_simplify(k_crack_term2_11, [])


# Printing results
os.chdir('C:/Users/natha/Documents/GitHub/panels')
from panels.dev.matrixtools import mprint_as_sparse

os.chdir(r'C:\Users\natha\Documents\GitHub\panels\theory\multidomain_penalization')
outdir = './output_expressions_python_new/'
import os
try: os.makedirs(outdir)
except: pass

matrices = [
    [f_crack_1st_term_p1, 'f_crack_1st_term_p1'],
    [f_crack_1st_term_p2, 'f_crack_1st_term_p2'],
    
    [k_crack11, 'k_crack11'],
    [k_crack12, 'k_crack12'],
    [k_crack22, 'k_crack22'],
    [k_crack_term2_11, 'k_crack_term2_11'],
]

for m in matrices:
    try:
        out = mprint_as_sparse(m[0], m[1], '11', print_file=False)
    except:
        print(m)
    with open(outdir + 'sympy_%s.txt' % m[1], 'w') as f:
        f.write(out)

# %% Generate terms for kcrack 2nd term

var('sw1, sw2, c1, c2')
s_delta = M([[sw1, -sw2]])
c = M([[c1, c2]]).T

kcrack_temp = (s_delta.T*s_delta)*(c*s_delta)
piece_wise_simplify(kcrack_temp, [])
print(kcrack_temp)

# %%
var('w1A, w1B, w1C, w2A, w2B, w2C, c1, c2')
s_delta_A = M([[w1A, -w2A]])
s_delta_B = M([[w1B, -w2B]])
s_delta_C = M([[w1C, -w2C]])
c = M([[c1, c2]]).T

kcrack_temp = (s_delta_A.T*s_delta_B)*(c*s_delta_C)
print(kcrack_temp)
print()
piece_wise_simplify(kcrack_temp, [])
print(kcrack_temp)



