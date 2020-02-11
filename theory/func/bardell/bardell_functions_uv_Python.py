from sympy import var, factorial, factorial2, sympify, diff
from sympy.printing import pycode

nmax = 42

xi = var('xi')

u = list(map(sympify, ['-(xi+1)/2 + 1',
                       '(xi+1)/2']))

for r in range(5, nmax+1):
    utmp = []
    for n in range(0, r//2+1):
        den = 2**n*factorial(n)*factorial(r-2*n-1)
        utmp.append((-1)**n*factorial2(2*r - 2*n - 7)/den * xi**(r-2*n-1)/1.)
    u.append(sum(utmp))

with open('./output_bardell_functions_uv.py', 'w') as f:
    f.write("# Modified Legendre polynomial's hierarchical functions\n")
    f.write("# - two first terms correspond to linear variation of displacements\n")
    f.write("#   for completeness of in-plane displacement functions\n")
    f.write("# - xi1t and xi2t control the displacements at extremity along each\n")
    f.write("#   natural coordinate\n")
    f.write('\n')
    f.write('# Number of terms: {0}\n\n'.format(len(u)))
    f.write('def fuv(xi, i, xi1t, xi2t):\n')
    f.write('    if i == 0:\n')
    f.write('        return xi1t*(%s)\n' % pycode(u[0]))
    f.write('    elif i == 1:\n')
    f.write('        return xi2t*(%s)\n' % pycode(u[1]))
    for i in range(2, len(u)):
        f.write('    elif i == %d:\n' % i)
        f.write('        return %s\n' % (pycode(u[i])))
    f.write('\n')
    f.write('\n\n')
    f.write('def fduv_dxi(xi, i, xi1t, xi2t):\n')
    f.write('    if i == 0:\n')
    f.write('        return xi1t*(%s)\n' % pycode(diff(u[0], xi)))
    f.write('    elif i == 1:\n')
    f.write('        return xi2t*(%s)\n' % pycode(diff(u[1], xi)))
    for i in range(2, len(u)):
        f.write('    elif i == %d:\n' % i)
        f.write('        return %s\n' % (pycode(diff(u[i], xi))))
    f.write('\n')
    f.write('\n\n')
    f.write('def fd2uv_dxi2(xi, i, xi1t, xi2t):\n')
    f.write('    if i == 0:\n')
    f.write('        return xi1t*(%s)\n' % pycode(diff(u[0], xi, xi)))
    f.write('    elif i == 1:\n')
    f.write('        return xi2t*(%s)\n' % pycode(diff(u[1], xi, xi)))
    for i in range(2, len(u)):
        f.write('    elif i == %d:\n' % i)
        f.write('        return %s\n' % (pycode(diff(u[i], xi, xi))))
    f.write('\n')
    f.write('\n\n')

