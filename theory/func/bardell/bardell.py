from __future__ import division
import numpy as np
from sympy import var, factorial, factorial2, sympify

nmax = 40

xi = var('xi')
var('t1, r1, t2, r2')

u = map(sympify, ['1/2 - 3/4*xi + 1/4*xi**3',
                  '1/8 - 1/8*xi - 1/8*xi**2 + 1/8*xi**3',
                  '1/2 + 3/4*xi - 1/4*xi**3',
                  '-1/8 - 1/8*xi + 1/8*xi**2 + 1/8*xi**3'])
u = list(u)

for r in range(5, nmax+1):
    utmp = []
    for n in range(0, r//2+1):
        den = 2**n*factorial(n)*factorial(r-2*n-1)
        utmp.append((-1)**n*factorial2(2*r - 2*n - 7)/den * xi**(r-2*n-1))
    u.append(sum(utmp))

u[0] = t1*u[0]
u[1] = r1*u[1]
u[2] = t2*u[2]
u[3] = r2*u[3]

with open('bardell.txt', 'w') as f:
    f.write("Bardell's hierarchical functions\n\n")
    f.write('Number of terms: {0}\n\n'.format(len(u)))
    f.write('\n'.join(map(str, u)).replace('**', '^') + '\n\n')

