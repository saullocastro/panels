import os
import glob
from ast import literal_eval

import numpy as np
import sympy
from sympy import pi, sin, cos, var

from compmech.conecyl.sympytools import mprint_as_sparse

var('f1Au, g1Au, f1Av, g1Av, f1Aw, f1Awxi, g1Aw, g1Aweta')
var('f1Bu, g1Bu, f1Bv, g1Bv, f1Bw, f1Bwxi, g1Bw, g1Bweta')
var('f2Au, g2Au, f2Av, g2Av, f2Aw, f2Awxi, g2Aw, g2Aweta')
var('f2Bu, g2Bu, f2Bv, g2Bv, f2Bw, f2Bwxi, g2Bw, g2Bweta')

var('kt, kr, a1, a2, b1, b2, c1, dsb')

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
