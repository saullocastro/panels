from __future__ import division
import numpy as np
from sympy import var, factorial, factorial2, sympify, diff
from sympy.printing import ccode

nmax = 30

xi = var('xi')

u = list(map(sympify, ['-(xi+1)/2 + 1',
                       '(xi+1)/2']))

for r in range(5, nmax+1):
    utmp = []
    for n in range(0, r//2+1):
        den = 2**n*factorial(n)*factorial(r-2*n-1)
        utmp.append((-1)**n*factorial2(2*r - 2*n - 7)/den * xi**(r-2*n-1)/1.)
    u.append(sum(utmp))

with open('../../../panels/core/src/bardell_functions_uv.cpp', 'w') as f:
    f.write("// Bardell's hierarchical functions\n\n")
    f.write('// Number of terms: {0}\n\n'.format(len(u)))
    f.write('#include <stdlib.h>\n')
    f.write('#include <math.h>\n\n')
    f.write('#if defined(_WIN32) || defined(__WIN32__)\n')
    f.write('  #define EXPORTIT __declspec(dllexport)\n')
    f.write('#else\n')
    f.write('  #define EXPORTIT\n')
    f.write('#endif\n\n')
    f.write('EXPORTIT void vec_fuv(double *f, double xi, double xi1t, double xi2t) {\n')
    consts = {0:'xi1t', 1:'xi2t'}
    for i in range(len(u)):
        const = consts.get(i)
        if const is None:
            f.write('    f[%d] = %s;\n' % (i, ccode(u[i])))
        else:
            f.write('    f[%d] = %s*(%s);\n' % (i, const, ccode(u[i])))
    f.write('}\n')

    f.write('\n\n')
    f.write('EXPORTIT void vec_fuv_x(double *fxi, double xi, double xi1t, double xi2t) {\n')
    for i in range(len(u)):
        const = consts.get(i)
        if const is None:
            f.write('    fxi[%d] = %s;\n' % (i, ccode(diff(u[i], xi))))
        else:
            f.write('    fxi[%d] = %s*(%s);\n' % (i, const, ccode(diff(u[i], xi))))
    f.write('}\n')

    f.write('\n\n')
    f.write('EXPORTIT void vec_fuv_xx(double *fxixi, double xi, double xi1t, double xi2t) {\n')
    for i in range(len(u)):
        const = consts.get(i)
        if const is None:
            f.write('    fxixi[%d] = %s;\n' % (i, ccode(diff(u[i], xi, xi))))
        else:
            f.write('    fxixi[%d] = %s*(%s);\n' % (i, const, ccode(diff(u[i], xi, xi))))
    f.write('}\n')

    f.write('\n\n')
    f.write('EXPORTIT double fuv(int i, double xi, double xi1t, double xi2t) {\n')
    f.write('    switch(i) {\n')
    for i in range(len(u)):
        const = consts.get(i)
        f.write('    case %d:\n' % i)
        if const is None:
            f.write('        return %s;\n' % ccode(u[i]))
        else:
            f.write('        return %s*(%s);\n' % (const, ccode(u[i])))
    f.write('    default:\n')
    f.write('        return 0.;\n')
    f.write('    }\n')
    f.write('}\n')

    f.write('\n\n')
    f.write('EXPORTIT double fuv_x(int i, double xi, double xi1t, double xi2t) {\n')
    f.write('    switch(i) {\n')
    for i in range(len(u)):
        const = consts.get(i)
        f.write('    case %d:\n' % i)
        if const is None:
            f.write('        return %s;\n' % ccode(diff(u[i], xi)))
        else:
            f.write('        return %s*(%s);\n' % (const, ccode(diff(u[i], xi))))
    f.write('    default:\n')
    f.write('        return 0.;\n')
    f.write('    }\n')
    f.write('}\n')

    f.write('\n\n')
    f.write('EXPORTIT double fuv_xx(int i, double xi, double xi1t, double xi2t) {\n')
    f.write('    switch(i) {\n')
    for i in range(len(u)):
        const = consts.get(i)
        f.write('    case %d:\n' % i)
        if const is None:
            f.write('        return %s;\n' % ccode(diff(u[i], xi, xi)))
        else:
            f.write('        return %s*(%s);\n' % (const, ccode(diff(u[i], xi, xi))))
    f.write('    default:\n')
    f.write('        return 0.;\n')
    f.write('    }\n')
    f.write('}\n')

with open('../../../panels/core/include/bardell_functions_uv.hpp', 'w') as g:
    g.write('#if defined(_WIN32) || defined(__WIN32__)\n')
    g.write('  #define IMPORTIT __declspec(dllimport)\n')
    g.write('#else\n')
    g.write('  #define IMPORTIT\n')
    g.write('#endif\n\n')
    g.write('#ifndef BARDELL_FUNCTIONS_UV_H\n')
    g.write('#define BARDELL_FUNCTIONS_UV_H\n')
    g.write('\n')
    g.write('IMPORTIT void vec_fuv(double *f, double xi,\n' +
            '        double xi1t, double xi2t);\n')
    g.write('\n')
    g.write('IMPORTIT void vec_fuv_x(double *fxi, double xi,\n' +
            '        double xi1t, double xi2t);\n')
    g.write('\n')
    g.write('IMPORTIT void vec_fuv_xx(double *fxixi, double xi,\n' +
            '        double xi1t, double xi2t);\n')
    g.write('\n')
    g.write('IMPORTIT double fuv(int i, double xi,\n' +
            '        double xi1t, double xi2t);\n')
    g.write('\n')
    g.write('IMPORTIT double fuv_x(int i, double xi,\n' +
            '        double xi1t, double xi2t);\n')
    g.write('\n')
    g.write('IMPORTIT double fuv_xx(int i, double xi,\n' +
            '        double xi1t, double xi2t);\n')
    g.write('\n')
    g.write('#endif /** BARDELL_FUNCTIONS_UV_H */')
