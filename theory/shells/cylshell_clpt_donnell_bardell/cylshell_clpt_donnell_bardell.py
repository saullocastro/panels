from sympy import Matrix as M, Symbol, init_printing, var

var('a, b, r, h, d, intx, inty, rho, aeromu, beta, gamma, wxi, weta')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('Nxx, Nyy, Nxy, Mxx, Myy, Mxy')

var('fAu, gAu, fAv, gAv, fAw, gAw, fAuxi, fAvxi, fAwxi, gAueta, gAveta, gAweta, fAwxixi, gAwetaeta')
var('fBu, gBu, fBv, gBv, fBw, gBw, fBuxi, fBvxi, fBwxi, gBueta, gBveta, gBweta, fBwxixi, gBwetaeta')

F = M([[A11, A12, A16, B11, B12, B16],
       [A12, A22, A26, B12, B22, B26],
       [A16, A26, A66, B16, B26, B66],
       [B11, B12, B16, D11, D12, D16],
       [B12, B22, B26, D12, D22, D26],
       [B16, B26, B66, D16, D26, D66]])

uA = M([[fAu*gAu, 0, 0]])
vA = M([[0, fAv*gAv, 0]])
wA = M([[0, 0, fAw*gAw]])

uAxi = M([[fAuxi*gAu, 0, 0]])
vAxi = M([[0, fAvxi*gAv, 0]])
wAxi = M([[0, 0, fAwxi*gAw]])

uAeta = M([[fAu*gAueta, 0, 0]])
vAeta = M([[0, fAv*gAveta, 0]])
wAeta = M([[0, 0, fAw*gAweta]])

wAxixi = M([[0, 0, fAwxixi*gAw]])
wAxieta = M([[0, 0, fAwxi*gAweta]])
wAetaeta = M([[0, 0, fAw*gAwetaeta]])

uB = M([[fBu*gBu, 0, 0]])
vB = M([[0, fBv*gBv, 0]])
wB = M([[0, 0, fBw*gBw]])

uBxi = M([[fBuxi*gBu, 0, 0]])
vBxi = M([[0, fBvxi*gBv, 0]])
wBxi = M([[0, 0, fBwxi*gBw]])

uBeta = M([[fBu*gBueta, 0, 0]])
vBeta = M([[0, fBv*gBveta, 0]])
wBeta = M([[0, 0, fBw*gBweta]])

wBxixi = M([[0, 0, fBwxixi*gBw]])
wBxieta = M([[0, 0, fBwxi*gBweta]])
wBetaeta = M([[0, 0, fBw*gBwetaeta]])

GA = M([(2/a)*wAxi, (2/b)*wAeta])
BA = M([(2/a)*uAxi + (2/a)**2*wAxi*wxi,
        (2/b)*vAeta + 1/r*wA + (2/b)**2*wAeta*weta,
        (2/b)*uAeta + (2/a)*vAxi + (2/a)*(2/b)*(weta*wAxi + wxi*wAeta),
       -(2/a)*(2/a)*wAxixi,
       -(2/b)*(2/b)*wAetaeta,
       -(2/a)*(2/b)*2*wAxieta])
B0A = BA.subs({wxi.name:0, weta.name:0})

GB = M([(2/a)*wBxi, (2/b)*wBeta])
BB = M([(2/a)*uBxi + (2/a)**2*wBxi*wxi,
        (2/b)*vBeta + 1/r*wB + (2/b)**2*wBeta*weta,
        (2/b)*uBeta + (2/a)*vBxi + (2/a)*(2/b)*(weta*wBxi + wxi*wBeta),
       -(2/a)*(2/a)*wBxixi,
       -(2/b)*(2/b)*wBetaeta,
       -(2/a)*(2/b)*2*wBxieta])
B0B = BB.subs({wxi.name:0, weta.name:0})

kC = intx*inty/4*BA.T*F*BB
fint = intx*inty/4*BA.T*M([[Nxx, Nyy, Nxy, Mxx, Myy, Mxy]]).T
kC0 = intx*inty/4*B0A.T*F*B0B;

Nmat = M([[Nxx, Nxy],
          [Nxy, Nyy]])

kG = intx*inty/4*GA.T*Nmat*GB

maux = M([[   1,   0,   0,  -d,  0],
          [   0,   1,   0,   0, -d],
          [   0,   0,   1,   0,  0],
          [ -d,   0, 0, (h^2/12 + d^2), 0],
          [    0, -d, 0, 0, (h^2/12 + d^2)]])

gA5 = M([uA, vA, wA, -(2/a)*wAxi, -(2/b)*wAeta])
gB5 = M([uB, vB, wB, -(2/a)*wBxi, -(2/b)*wBeta])
kM = intx*inty/4*h*rho*gA5.T*maux*gB5

kAx = -intx*inty/4 * (beta*(2/a)*wAxi.T*wB + gamma*wA.T*wB)
kAy = -intx*inty/4 * beta*(2/b)*wAeta.T*wB

cA = -intx*inty/4*aeromu*wA.T*wB

from sympy import collect, simplify

def piece_wise_simplify(m, vars):
    import numpy as np
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m

vars = (A11, A12, A16, A22, A26, A66,
        B11, B12, B16, B22, B26, B66,
        D11, D12, D16, D22, D26, D66)

piece_wise_simplify(kC, vars)
piece_wise_simplify(kC0, vars)
piece_wise_simplify(kG, (Nxx, Nyy, Nxy))

from panels.dev.matrixtools import mprint_as_sparse

outdir = './output_expressions_python/'
import os
try: os.makedirs(outdir)
except: pass

matrices = [
    [kC, 'kC'],
    [kC0, 'kC0'],
    [kG, 'kG'],
    [kM, 'kM'],
    [kAx, 'kAx'],
    [kAy, 'kAy'],
    [cA, 'cA'],
]

for m in matrices:
    try:
        out = mprint_as_sparse(m[0], m[1], '11', print_file=False)
    except:
        print(m)
    with open(outdir + 'sympy_%s.txt' % m[1], 'w') as f:
        f.write(out)
