# -*- coding: utf-8 -*-
"""
Created on Sat May 18 14:09:40 2024

@author: Nathan
"""


#%% Sigmoid function derivatives 

import sympy as sym
xi1t, xi2t, xi = sym.symbols('xi1t xi2t xi')
f_0 = xi1t*(1/(1 + sym.exp((xi+1)*15 - 15)))
fp_0 = sym.diff(f_0, xi)
f_2 = xi2t*(1/(1 + sym.exp(-((xi+1)*15 - 15))))
fp_2 = sym.diff(f_2, xi)
print(f'fp_0 = {fp_0}')
print(f'fp_2 = {fp_2}')