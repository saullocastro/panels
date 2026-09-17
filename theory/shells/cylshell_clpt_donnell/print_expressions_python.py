import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var

from panels.dev.matrixtools import mprint_as_sparse

var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('fAuxi, fBuxi, gAu, gBu, fAu, fBu, gAueta, gBueta')
var('fAvxi, fBvxi, fAv, fBv, gAv, gBv, gAveta, gBveta')
var('fAwxixi, fBwxixi, gAw, gBw, fAwxi, fBwxi, gAweta, gBweta')
var('fAw, fBw, gAwetaeta, gBwetaeta')
var('r, a, b, Nxx, Nyy, Nxy, Mxx, Myy, Mxy, intx, inty')
var('wxi, weta')
var('aeromu, beta, gamma, h, rho, d')

subs = {
       }

def List(*e):
    return list(e)

for i, filepath in enumerate(
        glob.glob(r'.\output_expressions_mathematica\fortran_*.txt')):
    print(filepath)
    with open(filepath) as f:
        filename = os.path.basename(filepath)
        names = filename[:-4].split('_')
        lines = [line.strip() for line in f.readlines()]
        string = ''.join(lines)
        string = string.replace('\\','')
        tmp = eval(string)
        matrix = sympy.Matrix(np.atleast_2d(tmp))
        printstr = ''
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i,j] == 0:
                    continue
                else:
                    printstr += '%s[row+%d, col+%d] = %s\n' % (names[1], i, j, str(matrix[i, j]))
        printstr = mprint_as_sparse(matrix, names[1], "11",
                                    print_file=False, collect_for=None,
                                    subs=subs)

    with open('.\\output_expressions_python\\' + filename, 'w') as f:
        f.write(printstr)
