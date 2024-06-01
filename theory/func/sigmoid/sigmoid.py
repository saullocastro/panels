# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:18:02 2024

@author: Nathan
"""

import numpy as np
from sympy import Matrix as M, Symbol, init_printing, var
import sympy as sp
import os

var("xi1t, xi2t, xi1r, xi2r")
var('xi')

sig_ftn = sp.zeros(1,30)
b = 15
a = -15
pos = 0
# Increasing c1 = 1
c1 = 1
for c2 in range(-12,18,2):
    sig_ftn[0,pos] = 1/(1 + sp.exp(-c1*((xi+1)*(b-a)/2 + a-c2)))
    pos += 1
# Decreasing c2 = 1
# for c2 in range(-18,18,2):
#     sig_ftn[0,pos] = 1/(1 + sp.exp(c1*((xi+1)*(b-a)/2 + a-c2)))
#     pos += 1
# print(sig_ftn)

if True:
    c1 = 2
    for c2 in np.arange(-8.85,15,3):
        sig_ftn[0,pos] = 1/(1 + sp.exp(-c1*((xi+1)*(b-a)/2 + a-c2)))
        pos += 1
    c1 = 3
    for c2 in np.arange(-5.3, 14+6, 4):
        sig_ftn[0,pos] = 1/(1 + sp.exp(-c1*((xi+1)*(b-a)/2 + a-c2)))
        pos += 1

deri_sig_ftn = sp.diff(sig_ftn, xi)
# print(deri_sig_ftn)

# %% Writing to a file
os.chdir('C:/Users/natha/Documents/GitHub/panels/theory/func/sigmoid')

with open('sigmoid.txt', 'w') as sigmoid_file:
    sigmoid_file.write('Function \n')
    for i in range(sig_ftn.shape[1]):
        sigmoid_file.write(f'f[{i}] = {sig_ftn[0,i]};\n')
    sigmoid_file.close()
    print('Sigmoid function printed to file :)')
    
with open('sigmoid.txt', 'a') as sigmoid_file:
    sigmoid_file.write('\nDerivatives \n')
    for i in range(deri_sig_ftn.shape[1]):
        sigmoid_file.write(f'fp[{i}] = {deri_sig_ftn[0,i]};\n')
    sigmoid_file.close()
    print('Sigmoid function\'s derivative printed to file :)')
    