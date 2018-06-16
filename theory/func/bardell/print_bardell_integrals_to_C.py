import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var
from sympy.printing import ccode

from compmech.conecyl.sympytools import mprint_as_sparse, pow2mult

var('x1t, x1r, x2t, x2r')
var('y1t, y1r, y2t, y2r')
var('xi1, xi2')
var('c0, c1')

subs = {
       }

def List(*e):
    return list(e)

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
printstr_12 = header_c
printstr_c0c1 = header_c

header_h = """
#if defined(_WIN32) || defined(__WIN32__)
  #define IMPORTIT __declspec(dllimport)
#else
  #define IMPORTIT
#endif
"""
printstr_full_h = header_h
printstr_12_h = header_h
printstr_c0c1_h = header_h


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
        print '\tfinished eval'
        printstr = ''
        if '_12' in filepath:
            name = '_'.join(names[1:3])
            printstr += 'EXPORTIT double integral_%s(double xi1, double xi2, int i, int j,\n' % name
            printstr += '                   double x1t, double x1r, double x2t, double x2r,\n'
            printstr += '                   double y1t, double y1r, double y2t, double y2r) {\n'

        elif '_c0c1' in filepath:
            name = '_'.join(names[1:3])
            printstr += 'EXPORTIT double integral_%s(double c0, double c1, int i, int j,\n' % name
            printstr += '                   double x1t, double x1r, double x2t, double x2r,\n'
            printstr += '                   double y1t, double y1r, double y2t, double y2r) {\n'

        else:
            name = names[1]
            printstr += 'EXPORTIT double integral_%s(int i, int j,\n' % name
            printstr += '           double x1t, double x1r, double x2t, double x2r,\n'
            printstr += '           double y1t, double y1r, double y2t, double y2r) {\n'

        printstr_h = '\n'
        printstr_h += '#ifndef BARDELL_%s_H\n' % name.upper()
        printstr_h += '#define BARDELL_%s_H\n' % name.upper()
        printstr_h += printstr.replace(' {', ';').replace('EXPORTIT', 'IMPORTIT')
        printstr_h += '#endif /** BARDELL_%s_H */\n' % name.upper()
        printstr_h += '\n'

        matrix = sympy.Matrix(np.atleast_2d(tmp))
        for i in range(matrix.shape[0]):
            activerow = False
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 0:
                    continue
                if not activerow:
                    activerow = True
                    if i == 0:
                        printstr += '    switch(i) {\n'
                    else:
                        printstr += '        default:\n'
                        printstr += '            return 0.;\n'
                        printstr += '        }\n'
                    printstr += '    case %d:\n' % i
                    printstr += '        switch(j) {\n'
                printstr += '        case %d:\n' % j
                printstr += '            return %s;\n' % ccode(matrix[i, j].evalf())
        printstr += '        default:\n'
        printstr += '            return 0.;\n'
        printstr += '        }\n'
        printstr += '    default:\n'
        printstr += '        return 0.;\n'
        printstr += '    }\n'
        printstr += '}\n'

        if '_12' in filepath:
            printstr_12_h += printstr_h
            filepath = r'..\..\..\compmech\lib\src\bardell_integral_%s_12.c' % name[:-3]
            with open(filepath, 'w') as g:
                g.write(printstr_12 + printstr)
        elif '_c0c1' in filepath:
            printstr_c0c1_h += printstr_h
            filepath = r'..\..\..\compmech\lib\src\bardell_integral_%s_c0c1.c' % name[:-5]
            with open(filepath, 'w') as g:
                g.write(printstr_c0c1 + printstr)
        else:
            printstr_full += printstr
            printstr_full_h += printstr_h


with open(r'..\..\..\compmech\include\bardell.h', 'w') as g:
    g.write(printstr_full_h)

with open(r'..\..\..\compmech\lib\src\bardell.c', 'w') as g:
    g.write(printstr_full)

with open(r'..\..\..\compmech\include\bardell_12.h', 'w') as g:
    g.write(printstr_12_h)

with open(r'..\..\..\compmech\include\bardell_c0c1.h', 'w') as g:
    g.write(printstr_c0c1_h)
