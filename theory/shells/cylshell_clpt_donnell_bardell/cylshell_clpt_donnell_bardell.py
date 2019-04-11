import sys
sys.path.append(r'C:\repositories\panels')

import numpy as np
from sympy import Matrix as M, Symbol, init_printing, var

var('a, b, r, h, d, intx, inty, rho, aeromu, beta, gamma, wxi, wxi_o, weta, weta_o')
var('A11, A12, A16, A22, A26, A66')
var('B11, B12, B16, B22, B26, B66')
var('D11, D12, D16, D22, D26, D66')
var('Nxx, Nyy, Nxy, Mxx, Myy, Mxy')
var('Nxx_o, Nyy_o, Nxy_o, Mxx_o, Myy_o, Mxy_o')

var('fAu, gAu, fAv, gAv, fAw, gAw, fAuxi, fAvxi, fAwxi, gAueta, gAveta, gAweta, fAwxixi, gAwetaeta')
var('fBu, gBu, fBv, gBv, fBw, gBw, fBuxi, fBvxi, fBwxi, gBueta, gBveta, gBweta, fBwxixi, gBwetaeta')
var('fCu, gCu, fCv, gCv, fCw, gCw, fCuxi, fCvxi, fCwxi, gCueta, gCveta, gCweta, fCwxixi, gCwetaeta')

var('uA, vA, wA, uAxi, vAxi, wAxi, uAeta, vAeta, wAeta, wAxixi, wAxieta, wAetaeta')
var('uB, vB, wB, uBxi, vBxi, wBxi, uBeta, vBeta, wBeta, wBxixi, wBxieta, wBetaeta')
var('uC, vC, wC, uCxi, vCxi, wCxi, uCeta, vCeta, wCeta, wCxixi, wCxieta, wCetaeta')

F = M([[A11, A12, A16, B11, B12, B16],
       [A12, A22, A26, B12, B22, B26],
       [A16, A26, A66, B16, B26, B66],
       [B11, B12, B16, D11, D12, D16],
       [B12, B22, B26, D12, D22, D26],
       [B16, B26, B66, D16, D26, D66]])

suA = M([[fAu*gAu, 0, 0]])
svA = M([[0, fAv*gAv, 0]])
swA = M([[0, 0, fAw*gAw]])

suAxi = M([[fAuxi*gAu, 0, 0]])
svAxi = M([[0, fAvxi*gAv, 0]])
swAxi = M([[0, 0, fAwxi*gAw]])

suAeta = M([[fAu*gAueta, 0, 0]])
svAeta = M([[0, fAv*gAveta, 0]])
swAeta = M([[0, 0, fAw*gAweta]])

swAxixi = M([[0, 0, fAwxixi*gAw]])
swAxieta = M([[0, 0, fAwxi*gAweta]])
swAetaeta = M([[0, 0, fAw*gAwetaeta]])

suB = M([[fBu*gBu, 0, 0]])
svB = M([[0, fBv*gBv, 0]])
swB = M([[0, 0, fBw*gBw]])

suBxi = M([[fBuxi*gBu, 0, 0]])
svBxi = M([[0, fBvxi*gBv, 0]])
swBxi = M([[0, 0, fBwxi*gBw]])

suBeta = M([[fBu*gBueta, 0, 0]])
svBeta = M([[0, fBv*gBveta, 0]])
swBeta = M([[0, 0, fBw*gBweta]])

swBxixi = M([[0, 0, fBwxixi*gBw]])
swBxieta = M([[0, 0, fBwxi*gBweta]])
swBetaeta = M([[0, 0, fBw*gBwetaeta]])

suC = M([[fCu*gCu, 0, 0]])
svC = M([[0, fCv*gCv, 0]])
swC = M([[0, 0, fCw*gCw]])

suCxi = M([[fCuxi*gCu, 0, 0]])
svCxi = M([[0, fCvxi*gCv, 0]])
swCxi = M([[0, 0, fCwxi*gCw]])

suCeta = M([[fCu*gCueta, 0, 0]])
svCeta = M([[0, fCv*gCveta, 0]])
swCeta = M([[0, 0, fCw*gCweta]])

swCxixi = M([[0, 0, fCwxixi*gCw]])
swCxieta = M([[0, 0, fCwxi*gCweta]])
swCetaeta = M([[0, 0, fCw*gCwetaeta]])

GA = M([(2/a)*swAxi, (2/b)*swAeta])
BA = M([(2/a)*suAxi + (2/a)**2*swAxi*wxi,
        (2/b)*svAeta + 1/r*swA + (2/b)**2*swAeta*weta,
        (2/b)*suAeta + (2/a)*svAxi + (2/a)*(2/b)*(weta*swAxi + wxi*swAeta),
       -(2/a)*(2/a)*swAxixi,
       -(2/b)*(2/b)*swAetaeta,
       -(2/a)*(2/b)*2*swAxieta])
ZEROS = BA[:1, :]*0
BA_o = M([(2/a)**2*swAxi*wxi_o,
        (2/b)**2*swAeta*weta_o,
        (2/a)*(2/b)*(weta_o*swAxi + wxi_o*swAeta),
         ZEROS,
         ZEROS,
         ZEROS])
B0A = BA.subs({wxi.name:0, weta.name:0})

GB = M([(2/a)*swBxi, (2/b)*swBeta])
BB = M([(2/a)*suBxi + (2/a)**2*swBxi*wxi,
        (2/b)*svBeta + 1/r*swB + (2/b)**2*swBeta*weta,
        (2/b)*suBeta + (2/a)*svBxi + (2/a)*(2/b)*(weta*swBxi + wxi*swBeta),
       -(2/a)*(2/a)*swBxixi,
       -(2/b)*(2/b)*swBetaeta,
       -(2/a)*(2/b)*2*swBxieta])
ZEROS = BB[:1, :]*0
BB_o = M([(2/a)**2*swBxi*wxi_o,
        (2/b)**2*swBeta*weta_o,
        (2/a)*(2/b)*(weta_o*swBxi + wxi_o*swBeta),
         ZEROS,
         ZEROS,
         ZEROS])
B0B = BB.subs({wxi.name:0, weta.name:0})

GC = M([(2/a)*swCxi, (2/b)*swCeta])
BC = M([(2/a)*suCxi + (2/a)**2*swCxi*wxi,
        (2/b)*svCeta + 1/r*swC + (2/b)**2*swCeta*weta,
        (2/b)*suCeta + (2/a)*svCxi + (2/a)*(2/b)*(weta*swCxi + wxi*swCeta),
       -(2/a)*(2/a)*swCxixi,
       -(2/b)*(2/b)*swCetaeta,
       -(2/a)*(2/b)*2*swCxieta])
ZEROS = BC[:1, :]*0
BC_o = M([(2/a)**2*swCxi*wxi_o,
        (2/b)**2*swCeta*weta_o,
        (2/a)*(2/b)*(weta_o*swCxi + wxi_o*swCeta),
         ZEROS,
         ZEROS,
         ZEROS])
B0C = BC.subs({wxi.name:0, weta.name:0})


# Constitutive stiffness matrix
kC = intx*inty/4*BA.T*F*BB

# Internal force vector
fint = intx*inty/4*BA.T*M([[Nxx, Nyy, Nxy, Mxx, Myy, Mxy]]).T

# Koiter's phi2 and phi2_o as 2nd order tensors
NeAB = (Nxx * 4/(a*a)*swAxi.T*swBxi
        + Nyy * 4/(b*b)*swAeta.T*swBeta
        + Nxy * 4/(a*b)*(swAxi.T*swBeta + swAeta.T*swBxi))
N_oeAB = (Nxx_o * 4/(a*a)*swAxi.T*swBxi
        + Nyy_o * 4/(b*b)*swAeta.T*swBeta
        + Nxy_o * 4/(a*b)*(swAxi.T*swBeta + swAeta.T*swBxi))
eA = BA[:3, :]
kA = BA[3:, :]
NB = (F*BB)[:3, :]
MB = (F*BB)[3:, :]

eA_o = BA_o[:3, :]
NB_o = (F*BB_o)[:3, :]
MB_o = (F*BB_o)[3:, :]

phi2 = intx*inty/4*(NeAB + NB*eA + MB*kA)
phi2_o = intx*inty/4*(N_oeAB + eA*NB_o + eA_o*NB + kA*MB_o)

A = F[:3, :3]
B = F[:3, 3:]
D = F[3:, 3:]

eAB1 = (2/a)**2*swAxi.T*swBxi
eAB2 = (2/b)**2*swAeta.T*swBeta
eAB3 = (2/a)*(2/b)*(swAxi.T*swBeta + swAeta.T*swBxi)


#FIXME attempt to already compute phi3*c1*c1, which would result in a vector
if False:
    # Koiter's phi3 is a 3rd order tensor, here I attempt to calculate
    # phi3*ca*cb to reduce two dimensions, making a vector at the end
    eA = M([[(2/a)*uAxi, 0, (2/a)**2*wAxi*wxi],
            [0, (2/b)*vAeta,  1/r*wA + (2/b)**2*wAeta*weta],
            [(2/b)*uAeta, (2/a)*vAxi, (2/a)*(2/b)*(weta*wAxi + wxi*wAeta)]])
    kA = M([[0, 0, -(2/a)*(2/a)*wAxixi],
            [0, 0, -(2/b)*(2/b)*wAetaeta],
            [0, 0, -(2/a)*(2/b)*2*wAxieta]])
    eB = M([[(2/a)*uBxi, 0, (2/a)**2*wBxi*wxi],
            [0, (2/b)*vBeta,  1/r*wB + (2/b)**2*wBeta*weta],
            [(2/b)*uBeta, (2/a)*vBxi, (2/a)*(2/b)*(weta*wBxi + wxi*wBeta)]])
    kB = M([[0, 0, -(2/a)*(2/a)*wBxixi],
            [0, 0, -(2/b)*(2/b)*wBetaeta],
            [0, 0, -(2/a)*(2/b)*2*wBxieta]])
    eC = M([[(2/a)*uCxi, 0, (2/a)**2*wCxi*wxi],
            [0, (2/b)*vCeta,  1/r*wC + (2/b)**2*wCeta*weta],
            [(2/b)*uCeta, (2/a)*vCxi, (2/a)*(2/b)*(weta*wCxi + wxi*wCeta)]])
    kC = M([[0, 0, -(2/a)*(2/a)*wCxixi],
            [0, 0, -(2/b)*(2/b)*wCetaeta],
            [0, 0, -(2/a)*(2/b)*2*wCxieta]])
    eAB = M([[0, 0, (2/a)**2*wAxi*wBxi],
             [0, 0, (2/b)**2*wAeta*wBeta],
             [0, 0, (2/a)*(2/b)*(wAxi*wBeta + wAeta*wBxi)]])
    eAC = M([(2/a)**2*wAxi*swCxi,
             (2/b)**2*wAeta*swCeta,
             (2/a)*(2/b)*(wAxi*swCeta + wAeta*swCxi)])
    eBC = M([(2/a)**2*wBxi*swCxi,
             (2/b)**2*wBeta*swCeta,
             (2/a)*(2/b)*(swCeta*wBxi + swCxi*wBeta)])
    NB = A*eB + B*kB
    NBC = A*eBC
    MBC = B*eBC

    NC = A*eC + B*kC
    NCeAB = ZEROS
    NBCeA = ZEROS
    NBeAC = ZEROS
    MBCkA = ZEROS
    for i in range(3):
        tmp =  (NC[i, :].T*eAB[i, :])
        NCeAB[i] = (NC[i, :].T*eAB[i, :])
        #NBCeA[i] = NBC[i, :]*eA[i, :]
        #NBeAC[i] = NB[i, :]*eAC[i, :]
        #MBCkA[i] = MBC[i, :]*kA[i, :]
    phi3 = intx*inty/4 * ( NCeAB + NBCeA + NBeAC + MBCkA )




# Linear constitutive stiffness matrix

kC0 = intx*inty/4*B0A.T*F*B0B;

# Geometric stiffness matrix

Nmat = M([[Nxx, Nxy],
          [Nxy, Nyy]])

kG = intx*inty/4*GA.T*Nmat*GB

# Mass matrix

maux = M([[   1,   0,   0,  -d,  0],
          [   0,   1,   0,   0, -d],
          [   0,   0,   1,   0,  0],
          [ -d,   0, 0, (h^2/12 + d^2), 0],
          [    0, -d, 0, 0, (h^2/12 + d^2)]])

gA5 = M([suA, svA, swA, -(2/a)*swAxi, -(2/b)*swAeta])
gB5 = M([suB, svB, swB, -(2/a)*swBxi, -(2/b)*swBeta])
kM = intx*inty/4*h*rho*gA5.T*maux*gB5

# Aerodynamic and damping matrix using piston's theory

kAx = -intx*inty/4 * (beta*(2/a)*swAxi.T*swB + gamma*swA.T*swB)
kAy = -intx*inty/4 * beta*(2/b)*swAeta.T*swB

cA = -intx*inty/4*aeromu*swA.T*swB

# Printing results

from sympy import collect, simplify

def piece_wise_simplify(m, vars):
    for (i,j), mij in np.ndenumerate(m):
        m[i,j] = collect(mij, vars, simplify)
    return m

vars = (A11, A12, A16, A22, A26, A66,
        B11, B12, B16, B22, B26, B66,
        D11, D12, D16, D22, D26, D66)

#piece_wise_simplify(kC, vars)
#piece_wise_simplify(kC0, vars)
#piece_wise_simplify(kG, (Nxx, Nyy, Nxy))

from panels.dev.matrixtools import mprint_as_sparse

outdir = './output_expressions_python/'
import os
try: os.makedirs(outdir)
except: pass

matrices = [
    #[kC, 'kC'],
    #[kC0, 'kC0'],
    #[kG, 'kG'],
    [kM, 'kM'],
    [kAx, 'kAx'],
    [kAy, 'kAy'],
    [cA, 'cA'],
    [phi2, 'phi2'],
    [phi2, 'phi2'],
    #[phi3, 'phi3'], #FIXME
    [BA, 'eAkA'],
    [eAB1, 'eAB1'],
    [eAB2, 'eAB2'],
    [eAB3, 'eAB3'],
]

for m in matrices:
    try:
        out = mprint_as_sparse(m[0], m[1], '11', print_file=False)
    except:
        print(m)
    with open(outdir + 'sympy_%s.txt' % m[1], 'w') as f:
        f.write(out)
