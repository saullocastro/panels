"""
Integrals of Bardell's hierarchical functions
=============================================

Computes exactly (rational arithmetic) the integrals over xi in [-1, +1] of
the products of Bardell's functions and of their derivatives, and writes the
C++ source ``panels/core/src/bardell.cpp`` and the header
``panels/core/include/bardell.hpp``.

The first four functions of the row index ``i`` are multiplied by the flags
``x1t, x1r, x2t, x2r``, and the first four functions of the column index ``j``
by ``y1t, y1r, y2t, y2r``.

"""
import os

from sympy import var, factorial, factorial2, Rational, Poly, Float
from sympy.printing.c import C99CodePrinter
from sympy.printing.str import StrPrinter

nmax = 30

here = os.path.dirname(os.path.abspath(__file__))
path_src = os.path.join(here, '..', '..', '..', 'panels', 'core', 'src',
        'bardell.cpp')
path_include = os.path.join(here, '..', '..', '..', 'panels', 'core',
        'include', 'bardell.hpp')

xi = var('xi')
xflags = var('x1t, x1r, x2t, x2r')
yflags = var('y1t, y1r, y2t, y2r')

# Bardell's functions (same definition as in bardell_functions_w_C.py), using
# exact rational arithmetic
u = [Rational(1, 2) - Rational(3, 4)*xi + Rational(1, 4)*xi**3,
     Rational(1, 8) - Rational(1, 8)*xi - Rational(1, 8)*xi**2 + Rational(1, 8)*xi**3,
     Rational(1, 2) + Rational(3, 4)*xi - Rational(1, 4)*xi**3,
     -Rational(1, 8) - Rational(1, 8)*xi + Rational(1, 8)*xi**2 + Rational(1, 8)*xi**3]

for r in range(5, nmax+1):
    utmp = []
    for n in range(0, r//2+1):
        if r - 2*n - 1 < 0:
            continue
        den = 2**n*factorial(n)*factorial(r-2*n-1)
        utmp.append((-1)**n*factorial2(2*r - 2*n - 7)/den * xi**(r-2*n-1))
    u.append(sum(utmp))

u = [Poly(ui, xi, domain='QQ') for ui in u]
# derivatives of order 0, 1 and 2
du = [u, [ui.diff(xi) for ui in u], [ui.diff((xi, 2)) for ui in u]]


def flag(flags, i):
    return flags[i] if i < len(flags) else 1


class BardellCodePrinter(C99CodePrinter):
    # floating-point numbers printed with 15 significant digits
    _print_Float = StrPrinter._print_Float


printer = BardellCodePrinter()

# (name in the C++ code, name of the header guard, derivative order of f_i,
#  derivative order of f_j)
integrals = [('ff', 'ff', 0, 0),
             ('ffp', 'ffxi', 0, 1),
             ('ffpp', 'ffxixi', 0, 2),
             ('fpfp', 'fxifxi', 1, 1),
             ('fpfpp', 'fxifxixi', 1, 2),
             ('fppfpp', 'fxixifxixi', 2, 2)]

header_c = """
#include <stdlib.h>
#include <math.h>
#if defined(_WIN32) || defined(__WIN32__)
  #define EXPORTIT __declspec(dllexport)
#else
  #define EXPORTIT
#endif
"""
printstr_full = header_c

header_h = """
#if defined(_WIN32) || defined(__WIN32__)
  #define IMPORTIT __declspec(dllimport)
#else
  #define IMPORTIT
#endif
"""
printstr_full_h = header_h

for name, guard, di, dj in integrals:
    print('integral_%s' % name)
    printstr = ''
    printstr += 'EXPORTIT double integral_%s(int i, int j,\n' % name
    printstr += '           double x1t, double x1r, double x2t, double x2r,\n'
    printstr += '           double y1t, double y1r, double y2t, double y2r) {\n'

    printstr_h = '\n'
    printstr_h += '#ifndef BARDELL_%s_H\n' % guard.upper()
    printstr_h += '#define BARDELL_%s_H\n' % guard.upper()
    printstr_h += printstr.replace(' {', ';').replace('EXPORTIT', 'IMPORTIT')
    printstr_h += '#endif /** BARDELL_%s_H */\n' % guard.upper()
    printstr_h += '\n'

    firstrow = True
    for i in range(nmax):
        activerow = False
        for j in range(nmax):
            p = (du[di][i]*du[dj][j]).integrate()
            value = p.eval(1) - p.eval(-1)
            if value == 0:
                continue
            if not activerow:
                activerow = True
                if firstrow:
                    firstrow = False
                    printstr += '    switch(i) {\n'
                else:
                    printstr += '        default:\n'
                    printstr += '            return 0.;\n'
                    printstr += '        }\n'
                printstr += '    case %d:\n' % i
                printstr += '        switch(j) {\n'
            if not value.is_integer:
                # exact value rounded to the nearest double-precision number,
                # later printed with 15 significant digits
                value = Float(float(value))
            expr = value*flag(xflags, i)*flag(yflags, j)
            printstr += '        case %d:\n' % j
            printstr += '            return %s;\n' % printer.doprint(expr.evalf())
    printstr += '        default:\n'
    printstr += '            return 0.;\n'
    printstr += '        }\n'
    printstr += '    default:\n'
    printstr += '        return 0.;\n'
    printstr += '    }\n'
    printstr += '}\n'

    printstr_full += printstr
    printstr_full_h += printstr_h

with open(path_include, 'w') as g:
    g.write(printstr_full_h)

with open(path_src, 'w') as g:
    g.write(printstr_full)
