from __future__ import division
import numpy as np
from sympy import var, factorial, factorial2, sympify, diff

nmax = 30

xi = var('xi')

u = map(sympify, ['1./2. - 3./4.*xi + 1./4.*xi**3',
                  '1./8. - 1./8.*xi - 1./8.*xi**2 + 1./8.*xi**3',
                  '1./2. + 3./4.*xi - 1./4.*xi**3',
                  '-1./8. - 1./8.*xi + 1./8.*xi**2 + 1./8.*xi**3'])

consts = {0:'xi1t', 1:'xi1r', 2:'xi2t', 3:'xi2r'}

for r in range(5, nmax+1):
    utmp = []
    for n in range(0, r//2+1):
        den = 2**n*factorial(n)*factorial(r-2*n-1)
        utmp.append((-1)**n*factorial2(2*r - 2*n - 7)/den * xi**(r-2*n-1)/1.)
    u.append(sum(utmp))

with open('../../../fortran/bardell/bardell_functions.f90', 'w') as f:
    f.write("! Bardell's hierarchical functions\n\n")
    f.write('! Number of terms: {0}\n\n'.format(len(u)))
    f.write('SUBROUTINE calc_vec_f(f, xi, xi1t, xi1r, xi2t, xi2r)\n')
    f.write('    REAL*8, INTENT(IN) :: xi, xi1t, xi1r, xi2t, xi2r\n')
    f.write('    REAL*8, INTENT(OUT) :: f(%d)\n' % len(u))
    for i in range(len(u)):
        const = consts.get(i)
        if const is None:
            f.write('    f(%d) = %s\n' % (i+1, str(u[i])))
        else:
            f.write('    f(%d) = %s*(%s)\n' % (i+1, const, str(u[i])))
    f.write('END SUBROUTINE\n')

    f.write('\n\n')
    f.write('SUBROUTINE calc_vec_fxi(fxi, xi, xi1t, xi1r, xi2t, xi2r)\n')
    f.write('    REAL*8, INTENT(IN) :: xi, xi1t, xi1r, xi2t, xi2r\n')
    f.write('    REAL*8, INTENT(OUT) :: fxi(%d)\n' % len(u))
    for i in range(len(u)):
        const = consts.get(i)
        if const is None:
            f.write('    fxi(%d) = %s\n' % (i+1, str(diff(u[i], xi))))
        else:
            f.write('    fxi(%d) = %s*(%s)\n' % (i+1, const, str(diff(u[i], xi))))
    f.write('END SUBROUTINE\n')

    f.write('\n\n')

    f.write('SUBROUTINE calc_f(i, xi, xi1t, xi1r, xi2t, xi2r, out)\n')
    f.write('    INTEGER, INTENT(IN) :: i\n')
    f.write('    REAL*8, INTENT(IN) :: xi, xi1t, xi1r, xi2t, xi2r\n')
    f.write('    REAL*8, INTENT(OUT) :: out\n')
    f.write('    out = 0.\n')
    f.write('    SELECT CASE (i)\n')
    for i in range(len(u)):
        const = consts.get(i)
        f.write('    CASE (%d)\n' % (i+1))
        if const is None:
            f.write('        out = %s\n' % str(u[i]))
        else:
            f.write('        out = %s*(%s)\n' % (const, str(u[i])))
    f.write('    END SELECT\n')
    f.write('END SUBROUTINE\n')

    f.write('\n\n')
    f.write('SUBROUTINE calc_fxi(i, xi, xi1t, xi1r, xi2t, xi2r, out)\n')
    f.write('    INTEGER, INTENT(IN) :: i\n')
    f.write('    REAL*8, INTENT(IN) :: xi, xi1t, xi1r, xi2t, xi2r\n')
    f.write('    REAL*8, INTENT(OUT) :: out\n')
    f.write('    out = 0.\n')
    f.write('    SELECT CASE (i)\n')
    for i in range(len(u)):
        const = consts.get(i)
        f.write('    CASE (%d)\n' % (i+1))
        if const is None:
            f.write('        out = %s\n' % str(diff(u[i], xi)))
        else:
            f.write('        out = %s*(%s)\n' % (const, str(diff(u[i], xi))))
    f.write('    END SELECT\n')
    f.write('END SUBROUTINE\n')
