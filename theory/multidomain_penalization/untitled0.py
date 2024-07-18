# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:44:24 2024

@author: Nathan
"""

from sympy import collect, simplify
from sympy import Matrix as M, Symbol, init_printing, var

var('a,b,c,d,e,f,g,h,i')

SF = M([[a, b, c]])
c = M([d,e,f])
dc = M([g,h,i])

def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m


# Proof that c.T*SF.T*SF*dc - dc.T*SF.T*SF*c
print(c.T*SF.T)
print(SF*dc)
print(SF.T*SF)
print(c.T*SF.T*SF*dc)
print(dc.T*SF.T*SF*c)
print(c.T*SF.T*SF*dc - dc.T*SF.T*SF*c)
print(simplify(c.T*SF.T*SF*dc - dc.T*SF.T*SF*c))

piece_wise_simplify(c.T*SF.T*SF*dc - dc.T*SF.T*SF*c, [])
