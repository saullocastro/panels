import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var

from compmech.conecyl.sympytools import mprint_as_sparse

var('fAu, fAuxi, gAu, gAueta, fAv, fAvxi, gAv, gAveta, fAw, fAwxi, fAwxixi, gAw, gAweta, gAwetaeta')
var('fBu, fBuxi, gBu, gBueta, fBv, fBvxi, gBv, gBveta, fBw, fBwxi, fBwxixi, gBw, gBweta, gBwetaeta')

var('pAu, pAuxi, qAu, qAueta, pAv, pAvxi, qAv, qAveta, pAw, pAwxi, pAwxixi, qAw, qAweta, qAwetaeta')
var('pBu, pBuxi, qBu, qBueta, pBv, pBvxi, qBv, qBveta, pBw, pBwxi, pBwxixi, qBw, qBweta, qBwetaeta')

var('kt, kr, a, b, bb, bf, c1')

subs = {
       }

def List(*e):
    return list(e)

for i, filepath in enumerate(
        glob.glob(r'./output_expressions_mathematica/fortran_*.txt')):
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
