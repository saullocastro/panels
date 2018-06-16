import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var

var('xi1, xi2')
var('x1t, x1r, x2t, x2r')
var('y1t, y1r, y2t, y2r')

subs = {
       }

def List(*e):
    return list(e)

printstr_12 = ''
printstr_full = ''
for i, filepath in enumerate(
        glob.glob(r'.\bardell_integrals_mathematica\fortran_*.txt')):
    print(filepath)
    with open(filepath) as f:
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_')
        lines = [line.strip() for line in f.readlines()]
        string = ''.join(lines)
        string = string.replace('\\','')
        tmp = eval(string)
        if '_12' in filepath:
            name = '_'.join(names[1:3])
        else:
            name = names[1]
        matrix = sympy.Matrix(np.atleast_2d(tmp)).evalf()
        if '_12' in filepath:
            printstr = 'SUBROUTINE integral_%s(xi1, xi2, i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)\n' % name
            printstr += '    REAL*8, INTENT(IN) :: xi1, xi2, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r\n'
        else:
            printstr = 'SUBROUTINE integral_%s(i, j, x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r, out)\n' % name
            printstr += '    REAL*8, INTENT(IN) :: x1t, x1r, x2t, x2r, y1t, y1r, y2t, y2r\n'
        printstr += '    INTEGER, INTENT(IN) :: i, j\n'
        printstr += '    REAL*8, INTENT(OUT) :: out\n'
        printstr += '    out = 0\n'
        printstr += '    SELECT CASE (i)\n'
        for i in range(matrix.shape[0]):
            activerow = False
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 0:
                    continue
                if not activerow:
                    activerow = True
                    printstr += '    CASE (%d)\n' % (i+1)
                    printstr += '        SELECT CASE (j)\n'
                printstr += '        CASE (%d)\n' % (j+1)
                printstr += '            out = %s\n' % str(matrix[i, j])
                printstr += '            RETURN\n'
            printstr += '        END SELECT\n'
        printstr += '    END SELECT\n'
        printstr += 'END SUBROUTINE integral_%s\n\n\n' % name
        if '_12' in filepath:
            printstr_12 += printstr
        else:
            printstr_full += printstr

with open('.\\bardell_integrals_fortran\\bardell_integrals_fortran_12.txt', 'w') as f:
    f.write(printstr_12)
with open('.\\bardell_integrals_fortran\\bardell_integrals_fortran_full.txt', 'w') as f:
    f.write(printstr_full)
