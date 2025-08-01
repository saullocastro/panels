# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 16:44:24 2024

@author: Nathan
"""

from sympy import collect, simplify
from sympy import Matrix as M, Symbol, init_printing, var
def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m

#%%

var('a,b,c,d,e,f,g,h,i')

SF = M([[a, b, c]])
c = M([d,e,f])
dc = M([g,h,i])

# Proof that c.T*SF.T*SF*dc - dc.T*SF.T*SF*c
print(c.T*SF.T)
print(SF*dc)
print(SF.T*SF)
print(c.T*SF.T*SF*dc)
print(dc.T*SF.T*SF*c)
print(c.T*SF.T*SF*dc - dc.T*SF.T*SF*c)
print(simplify(c.T*SF.T*SF*dc - dc.T*SF.T*SF*c))

piece_wise_simplify(c.T*SF.T*SF*dc - dc.T*SF.T*SF*c, [])

print()
print()

#%%
var('a,b,c,d,e,f,g,h,i,j,k,l,m,n,o')
matrix = M([[a, b, c], [d, e, f], [g, h, i]])
# print(matrix.shape)

c = M([j, k, l]).T
SF = M([m, n, o])

print((c*SF))
print((c*SF).shape)
print()
print(SF*c)
print((SF*c).shape)
print()
print(simplify(matrix*(c.dot(SF))))
print((simplify(matrix*(c.dot(SF)))).shape)
print()
print(simplify(matrix*(SF*c)))
print(simplify(matrix*(SF*c)).shape)
print()
print(simplify(simplify(matrix*(c.dot(SF))) - simplify(matrix*(SF*c))))
print(simplify(simplify(matrix*(c.dot(SF))) - simplify(matrix*(SF*c))).shape)


