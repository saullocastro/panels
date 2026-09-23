"""Write the Sanders .pyx modules from sanders_exprs.py

Run ``derive_expressions.py`` first, which writes ``sanders_exprs.py``
"""
import os
import re

from sanders_exprs import ANALYTICAL, NUMERICAL

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '..', '..', '..'))

HEADER = """#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
"""

BC_CDEF = """    cdef double x1u, x1ur, x2u, x2ur
    cdef double x1v, x1vr, x2v, x2vr
    cdef double x1w, x1wr, x2w, x2wr
    cdef double y1u, y1ur, y2u, y2ur
    cdef double y1v, y1vr, y2v, y2vr
    cdef double y1w, y1wr, y2w, y2wr"""

BC_BLOCK = """    x1u = shell.x1u; x1ur = shell.x1ur; x2u = shell.x2u; x2ur = shell.x2ur
    x1v = shell.x1v; x1vr = shell.x1vr; x2v = shell.x2v; x2vr = shell.x2vr
    x1w = shell.x1w; x1wr = shell.x1wr; x2w = shell.x2w; x2wr = shell.x2wr
    y1u = shell.y1u; y1ur = shell.y1ur; y2u = shell.y2u; y2ur = shell.y2ur
    y1v = shell.y1v; y1vr = shell.y1vr; y2v = shell.y2v; y2vr = shell.y2vr
    y1w = shell.y1w; y1wr = shell.y1wr; y2w = shell.y2w; y2wr = shell.y2wr"""

LIMITS = """    #NOTE physical limits of the integration domain, resolved and validated
    #      by Shell.integration_limits(); for the full domain the mapping
    #      below gives exactly xi1 = eta1 = -1 and xi2 = eta2 = +1
    x1, x2, y1, y2 = shell.integration_limits()"""

MAPPING = """    xinf = 0
    xsup = shell.a
    xi1 = (x1 - xinf)/(xsup - xinf)*2 - 1
    xi2 = (x2 - xinf)/(xsup - xinf)*2 - 1

    yinf = 0
    ysup = shell.b
    eta1 = (y1 - yinf)/(ysup - yinf)*2 - 1
    eta2 = (y2 - yinf)/(ysup - yinf)*2 - 1

    intx = x2 - x1
    inty = y2 - y1"""

POINT = """        for ptx in range(nx):
            for pty in range(ny):
                xi = xis[ptx]
                eta = etas[pty]
                xi = (xi - (-1))/2 * (xi2 - xi1) + xi1
                eta = (eta - (-1))/2 * (eta2 - eta1) + eta1

                weight = weights_xi[ptx] * weights_eta[pty]
"""

FINPUT = """    # F as 4-D matrix, must be [nx, ny, 6, 6], when there is one ABD[6, 6] for
    # each of the nx * ny integration points
    cdef double F[6 * 6]
    cdef double [:, :, :, ::1] Fnxny

    cdef int one_F_each_point = 0

    Finput = np.ascontiguousarray(Finput, dtype=DOUBLE)
    if Finput.shape == (nx, ny, 6, 6):
        Fnxny = Finput
        one_F_each_point = 1
    elif Finput.shape == (6, 6):
        # creating dummy 4-D array that is not used
        Fnxny = np.empty(shape=(0, 0, 0, 0), dtype=DOUBLE)
        # using a constant F for all the integration domain
        for i in range(6):
            for j in range(6):
                F[i*6 + j] = Finput[i, j]
    else:
        raise ValueError('Invalid shape for Finput!')"""


def abd_read(indent, D=True):
    s = ' '*indent
    lines = [s + 'if one_F_each_point == 1:',
             s + '    for i in range(6):',
             s + '        for j in range(6):',
             s + '            #TODO could assume symmetry',
             s + '            F[i*6 + j] = Fnxny[ptx, pty, i, j]',
             '']
    blocks = [('A', [(0, 0, 11), (0, 1, 12), (0, 2, 16), (1, 1, 22), (1, 2, 26), (2, 2, 66)]),
              ('B', [(0, 3, 11), (0, 4, 12), (0, 5, 16), (1, 4, 22), (1, 5, 26), (2, 5, 66)])]
    if D:
        blocks.append(('D', [(3, 3, 11), (3, 4, 12), (3, 5, 16), (4, 4, 22), (4, 5, 26), (5, 5, 66)]))
    for name, items in blocks:
        for i, j, ij in items:
            lines.append(s + '{0}{1} = F[{2}*6 + {3}]'.format(name, ij, i, j))
        lines.append('')
    return '\n'.join(lines)


F_NAMES = ['u', 'uxi', 'v', 'vxi', 'w', 'wxi', 'wxixi']
G_NAMES = ['u', 'ueta', 'v', 'veta', 'w', 'weta', 'wetaeta']


def basis_line(fg, side, name, indent):
    field = name[0]
    der = name[1:]
    if fg == 'f':
        order = der.count('xi')
        idx = 'i' if side == 'A' else 'k'
        var, xy = 'xi', 'x'
    else:
        order = der.count('eta')
        idx = 'j' if side == 'A' else 'l'
        var, xy = 'eta', 'y'
    func = ['f', 'fp', 'fpp'][order]
    return '{0}{1}{2}{3} = {4}({5}, {6}, {7}1{8}, {7}1{8}r, {7}2{8}, {7}2{8}r)'.format(
        ' '*indent, fg, side, name, func, idx, var, xy, field)


def used(names, exprs, fg, side):
    text = ' '.join(exprs)
    return [n for n in names if re.search(r'\b{0}{1}{2}\b'.format(fg, side, n), text)]


def basis_loops(mname, exprs, entry_lines, symmetric=True):
    """The i, k, j, l loops of the numerical kernels"""
    fA = used(F_NAMES, exprs, 'f', 'A')
    fB = used(F_NAMES, exprs, 'f', 'B')
    gA = used(G_NAMES, exprs, 'g', 'A')
    gB = used(G_NAMES, exprs, 'g', 'B')
    lines = []
    lines.append('                # {0}'.format(mname))
    lines.append('                c = -1')
    lines.append('                for i in range(m):')
    lines += [basis_line('f', 'A', n, 20) for n in fA]
    lines.append('')
    lines.append('                    for k in range(m):')
    lines += [basis_line('f', 'B', n, 24) for n in fB]
    lines.append('')
    lines.append('                        for j in range(n):')
    lines += [basis_line('g', 'A', n, 28) for n in gA]
    lines.append('')
    lines.append('                            for l in range(n):')
    lines.append('')
    lines.append('                                row = row0 + DOF*(j*m + i)')
    lines.append('                                col = col0 + DOF*(l*m + k)')
    lines.append('')
    if symmetric:
        lines.append('                                #NOTE symmetry assumption True if no follower forces are used')
        lines.append('                                if row > col:')
        lines.append('                                    continue')
        lines.append('')
    lines += [basis_line('g', 'B', n, 32) for n in gB]
    lines.append('')
    lines += entry_lines
    return ('\n'.join(lines), ['fA' + n for n in fA] + ['fB' + n for n in fB],
            ['gA' + n for n in gA] + ['gB' + n for n in gB])


def entry(mname, i, j, expr, extra=None):
    lines = ['                                c += 1',
             '                                if ptx == 0 and pty == 0:',
             '                                    {0}r[c] = row+{1}'.format(mname, i),
             '                                    {0}c[c] = col+{1}'.format(mname, j),
             '                                {0}v[c] += weight*(intx*inty/4)*( {1} )'.format(mname, expr)]
    if extra is not None:
        lines.append('                                # KGNL')
        lines.append('                                {0}v[c] += weight*(intx*inty/4)*( {1} )'.format(mname, extra))
    return lines


def basis_cdefs(fnames, gnames):
    out = []
    for side in 'AB':
        names = ['f{0}{1}'.format(side, n) for n in F_NAMES if 'f{0}{1}'.format(side, n) in fnames]
        if names:
            out.append('    cdef double ' + ', '.join(names))
    for side in 'AB':
        names = ['g{0}{1}'.format(side, n) for n in G_NAMES if 'g{0}{1}'.format(side, n) in gnames]
        if names:
            out.append('    cdef double ' + ', '.join(names))
    return '\n'.join(out)


def analytical_module():
    lines = [HEADER + '''r"""
Analytical matrices of the cylindrical shell using the CLPT with the
Sanders-Koiter kinematics, integrated over the full domain

See the kinematic equations in
``theory/shells/cylshell_clpt_sanders/cylshell_clpt_sanders.py``, from where
the integrands herein have been generated.

"""
from scipy.sparse import coo_matrix
import numpy as np

from panels import INT, DOUBLE


cdef int DOF = 3


cdef extern from 'bardell.hpp':
    double integral_ff(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_ffpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fpfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil
    double integral_fppfpp(int i, int j, double x1t, double x1r, double x2t, double x2r,
                       double y1t, double y1r, double y2t, double y2r) nogil

''']
    lines.append(ANALYTICAL['k0'])
    lines.append('\n')
    lines.append(ANALYTICAL['kG0'])
    lines.append('\n')
    lines.append(ANALYTICAL['kM'])
    lines.append('''

#NOTE the aerodynamic matrices depend only on w and are therefore the same as
#     for the Donnell kinematics
def fkAx(double beta, double gamma, object shell,
         int size, int row0, int col0):
    from .cylshell_clpt_donnell import fkAx as donnell_fkAx
    return donnell_fkAx(beta, gamma, shell, size, row0, col0)


def fkAy(double beta, object shell, int size, int row0, int col0):
    from .cylshell_clpt_donnell import fkAy as donnell_fkAy
    return donnell_fkAy(beta, shell, size, row0, col0)


def fcA(double aeromu, object shell, int size, int row0, int col0):
    from .cylshell_clpt_donnell import fcA as donnell_fcA
    return donnell_fcA(aeromu, shell, size, row0, col0)
''')
    return '\n'.join(lines)


def numerical_module():
    kC = NUMERICAL['kC']
    kGNL = NUMERICAL['kGNL']
    kG = NUMERICAL['kG']
    kM = NUMERICAL['kM']
    fint = NUMERICAL['fint']

    out = [HEADER + '''r"""
Numerically integrated matrices of the cylindrical shell using the CLPT with
the Sanders-Koiter kinematics

See the kinematic equations in
``theory/shells/cylshell_clpt_sanders/cylshell_clpt_sanders.py``, from where
the integrands herein have been generated. In short, with ``y`` the arc
length along the circumference, the rotations of the normal are::

    phix = -w,x
    phiy = -w,y + v/r

and the strains are::

    exx = u,x + phix**2/2
    eyy = v,y + w/r + phiy**2/2
    gxy = u,y + v,x + phix*phiy
    kxx = -w,xx
    kyy = -w,yy + v,y/r
    kxy = -2*w,xy + 3/2*v,x/r - 1/2*u,y/r

"""
from scipy.sparse import coo_matrix
import numpy as np
from scipy.special import roots_legendre

from panels import INT, DOUBLE


cdef extern from 'bardell_functions.hpp':
    double f(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil
    double fpp(int i, double xi, double xi1t, double xi1r, double xi2t, double xi2r) nogil

cdef int DOF = 3
''']

    # fkC_num
    ent_lines = []
    for i, j, v in kC:
        ent_lines += entry('kC', i, j, v, kGNL.get((i, j)))
    loops, fn, gn = basis_loops('kC', [v for _, _, v in kC] + list(kGNL.values()), ent_lines)
    out.append('''
def fkC_num(double [::1] cs, object Finput, object shell,
        int size, int row0, int col0, int nx, int ny, int NLgeom=0):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, r, intx, inty
    cdef int m, n
{BC_CDEF}

    cdef int i, j, k, l, c, row, col, ptx, pty
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66

    cdef long [::1] kCr, kCc
    cdef double [::1] kCv

{basis}
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double phix, phiy, NxxNL, NyyNL, NxyNL

    cdef double [::1] xis, etas, weights_xi, weights_eta

{FINPUT}

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    r = shell.r
    m = shell.m
    n = shell.n
{LIMITS}
{BC_BLOCK}

    fdim = {nnz}*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kCr = np.zeros((fdim,), dtype=INT)
    kCc = np.zeros((fdim,), dtype=INT)
    kCv = np.zeros((fdim,), dtype=DOUBLE)

{MAPPING}

    with nogil:
{POINT}
                # rotations of the normal, phix = -w,x and phiy = -w,y + v/r
                phix = 0
                phiy = 0
                if NLgeom == 1:
                    for j in range(n):
                        #TODO put these in a lookup vector
                        gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                        gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                        gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                        for i in range(m):
                            #TODO put these in a lookup vector
                            fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                            fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                            fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                            col = col0 + DOF*(j*m + i)

                            phix += -(2/a)*cs[col+2]*fAwxi*gAw
                            phiy += -(2/b)*cs[col+2]*fAw*gAweta + cs[col+1]*fAv*gAv/r

{ABD}
                # Membrane stress carried by the nonlinear strain
                # eps_NL = {{phix^2/2, phiy^2/2, phix*phiy}}. With it, KGNL = KG(N_NL)
                # is collected in kC such that
                #     KT = K0 + K0L + KL0 + KLL + KGNL (fkC_num) + KG(N0 + N_L) (fkG_num)
                # is the exact Jacobian of calc_fint, and fkG_num stays
                # homogeneous of degree one in cs, as linear buckling requires.
                # phix = phiy = 0 when NLgeom == 0, then KGNL vanishes
                NxxNL = A11*0.5*phix*phix + A12*0.5*phiy*phiy + A16*phix*phiy
                NyyNL = A12*0.5*phix*phix + A22*0.5*phiy*phiy + A26*phix*phiy
                NxyNL = A16*0.5*phix*phix + A26*0.5*phiy*phiy + A66*phix*phiy

{loops}

    kC = coo_matrix((kCv, (kCr, kCc)), shape=(size, size))

    return kC
'''.format(BC_CDEF=BC_CDEF, BC_BLOCK=BC_BLOCK, LIMITS=LIMITS, MAPPING=MAPPING,
           POINT=POINT, FINPUT=FINPUT, ABD=abd_read(16), loops=loops,
           basis=basis_cdefs(fn + ['fAv', 'fAw', 'fAwxi'], gn + ['gAv', 'gAw', 'gAweta']),
           nnz=len(kC)))

    # fkG_num
    ent_lines = []
    for i, j, v in kG:
        ent_lines += entry('kG', i, j, v)
    loops, fn, gn = basis_loops('kG', [v for _, _, v in kG], ent_lines)
    lin = ['fAu', 'fAuxi', 'fAv', 'fAvxi', 'fAw', 'fAwxi', 'fAwxixi',
           'gAu', 'gAueta', 'gAv', 'gAveta', 'gAw', 'gAweta', 'gAwetaeta']
    out.append('''

def fkG_num(double [::1] cs, object Finput, object shell,
            int size, int row0, int col0, int nx, int ny,
            double Nxx0=0, double Nyy0=0, double Nxy0=0):
    """Geometric stiffness matrix of the linear membrane stress state

    The membrane stress used here comes from the *linear* part of the strain
    evaluated at ``cs``, superposed with the constant stress state
    ``(Nxx0, Nyy0, Nxy0)``. The result is therefore homogeneous of degree one
    in ``cs``, which is what the linear buckling eigenvalue problem requires.

    With Sanders' kinematics the rotation ``phiy = -w,y + v/r`` depends on
    ``v``, such that this matrix couples the ``v`` and ``w`` DOFs.

    There is deliberately no switch to include the stress of the non-linear
    strain here. That contribution, ``KGNL``, is collected by
    :func:`.fkC_num` instead, so that

        KT = K0 + K0L + KL0 + KLL + KGNL   (fkC_num)
             + KG(N0 + N_L)                (fkG_num)

    is the exact Jacobian of :func:`.calc_fint` while this function stays
    homogeneous of degree one in ``cs``. Adding ``KGNL`` here would
    double-count it in the tangent stiffness matrix and break linear
    buckling. See the comments in :func:`.fkC_num` and :func:`.calc_fint`.

    """
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, r, intx, inty
    cdef int m, n
{BC_CDEF}

    cdef int i, k, j, l, c, row, col, ptx, pty
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef long [::1] kGr, kGc
    cdef double [::1] kGv

{basis}

    cdef double exx, eyy, gxy, kxx, kyy, kxy
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double Nxx, Nyy, Nxy

    cdef double [::1] xis, etas, weights_xi, weights_eta

{FINPUT}

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    r = shell.r
    m = shell.m
    n = shell.n
{LIMITS}
{BC_BLOCK}

    fdim = {nnz}*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kGr = np.zeros((fdim,), dtype=INT)
    kGc = np.zeros((fdim,), dtype=INT)
    kGv = np.zeros((fdim,), dtype=DOUBLE)

{MAPPING}

    with nogil:
{POINT}
                # Reading laminate constitutive data
{ABD}
                # Calculating the linear strain components. The stress of the
                # nonlinear strain enters KT through KGNL in fkC_num, such that
                # kG is homogeneous of degree one in cs
                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.
                for j in range(n):
                    #TODO put these in a lookup vector
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)

                    for i in range(m):
                        fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                        fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

{LINSTRAIN}

                # Calculating membrane stress components
                Nxx = Nxx0 + A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = Nyy0 + A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = Nxy0 + A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy

{loops}

    kG = coo_matrix((kGv, (kGr, kGc)), shape=(size, size))

    return kG
'''.format(BC_CDEF=BC_CDEF, BC_BLOCK=BC_BLOCK, LIMITS=LIMITS, MAPPING=MAPPING,
           POINT=POINT, FINPUT=FINPUT, ABD=abd_read(16, D=False), loops=loops,
           basis=basis_cdefs(fn + lin, gn + lin), nnz=len(kG),
           LINSTRAIN=LINSTRAIN.format(ind=' '*24)))

    # fkM_num
    ent_lines = []
    for i, j, v in kM:
        ent_lines += entry('kM', i, j, v)
    loops, fn, gn = basis_loops('kM', [v for _, _, v in kM], ent_lines)
    out.append('''

def fkM_num(object shell, double offset, object hrho_input, int size,
        int row0, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, r, intx, inty
    cdef int m, n
{BC_CDEF}

    cdef int i, j, k, l, c, row, col, ptx, pty

    cdef long [::1] kMr, kMc
    cdef double [::1] kMv

{basis}
    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2

    cdef double [::1] xis, etas, weights_xi, weights_eta

    # hrho as 3-D matrix, must be [nx, ny, 2], when there is one pair (h, rho)
    # for each of the nx * ny integration points
    cdef double h, rho, d
    cdef double [:, :, ::1] hrho_nxny

    cdef int one_hrho_each_point = 0

    d = offset

    hrho_input = np.asarray(hrho_input, dtype=DOUBLE)
    if hrho_input.shape == (nx, ny, 2):
        hrho_nxny = np.ascontiguousarray(hrho_input)
        one_hrho_each_point = 1
    elif hrho_input.shape == (2,):
        # creating dummy 3-D array that is not used
        hrho_nxny = np.empty(shape=(0, 0, 0), dtype=DOUBLE)
        # using a constant (h, rho) for all the integration domain
        hrho_input = np.ascontiguousarray(hrho_input)
        h, rho = hrho_input
    else:
        raise ValueError('Invalid shape for hrho_input!')

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    r = shell.r
    m = shell.m
    n = shell.n
{LIMITS}
{BC_BLOCK}

    fdim = {nnz}*m*m*n*n

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    kMr = np.zeros((fdim,), dtype=INT)
    kMc = np.zeros((fdim,), dtype=INT)
    kMv = np.zeros((fdim,), dtype=DOUBLE)

{MAPPING}

    with nogil:
{POINT}
                if one_hrho_each_point == 1:
                    h = hrho_nxny[ptx, pty, 0]
                    rho = hrho_nxny[ptx, pty, 1]

{loops}

    kM = coo_matrix((kMv, (kMr, kMc)), shape=(size, size))

    return kM


#NOTE the aerodynamic matrices depend only on w and are therefore the same as
#     for the Donnell kinematics
def fkAx_num(object shell, int size, int row0, int col0, int nx, int ny):
    from .cylshell_clpt_donnell_num import fkAx_num as donnell_fkAx_num
    return donnell_fkAx_num(shell, size, row0, col0, nx, ny)


def fkAy_num(object shell, int size, int row0, int col0, int nx, int ny):
    from .cylshell_clpt_donnell_num import fkAy_num as donnell_fkAy_num
    return donnell_fkAy_num(shell, size, row0, col0, nx, ny)
'''.format(BC_CDEF=BC_CDEF, BC_BLOCK=BC_BLOCK, LIMITS=LIMITS, MAPPING=MAPPING,
           POINT=POINT, loops=loops, basis=basis_cdefs(fn, gn), nnz=len(kM)))

    # calc_fint
    fint_lines = []
    for i, j, v in fint:
        fint_lines.append('                        fint[col+{0}] += weight*(intx*inty/4)*( {1} )'.format(i, v))
    out.append('''

def calc_fint(double [::1] cs, object Finput, object shell,
        int size, int col0, int nx, int ny):
    cdef double x1, x2, y1, y2, xinf, xsup, yinf, ysup
    cdef double a, b, r, intx, inty
    cdef int m, n
{BC_CDEF}

    cdef int i, j, c, col, ptx, pty
    cdef double A11, A12, A16, A22, A26, A66
    cdef double B11, B12, B16, B22, B26, B66
    cdef double D11, D12, D16, D22, D26, D66
    cdef double Nxx, Nyy, Nxy, Mxx, Myy, Mxy
    cdef double exx, eyy, gxy, kxx, kyy, kxy

    cdef double xi, eta, weight
    cdef double xi1, xi2, eta1, eta2
    cdef double phix, phiy

    cdef double fAu, fAuxi, fAv, fAvxi, fAw, fAwxi, fAwxixi
    cdef double gAu, gAueta, gAv, gAveta, gAw, gAweta, gAwetaeta

    cdef double [::1] xis, etas, weights_xi, weights_eta, fint

{FINPUT}

    if not 'Shell' in shell.__class__.__name__:
        raise ValueError('a Shell object must be given as input')
    a = shell.a
    b = shell.b
    r = shell.r
    m = shell.m
    n = shell.n
{LIMITS}
{BC_BLOCK}

    xis, weights_xi = roots_legendre(nx)
    etas, weights_eta = roots_legendre(ny)

    fint = np.zeros(size, dtype=DOUBLE)

{MAPPING}

    with nogil:
{POINT}
{ABD}
                # rotations of the normal, phix = -w,x and phiy = -w,y + v/r
                phix = 0
                phiy = 0
                for j in range(n):
                    #TODO save in buffer
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    for i in range(m):
                        #TODO save in buffer
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

                        phix += -(2/a)*cs[col+2]*fAwxi*gAw
                        phiy += -(2/b)*cs[col+2]*fAw*gAweta + cs[col+1]*fAv*gAv/r

                # current strain state
                exx = 0.
                eyy = 0.
                gxy = 0.
                kxx = 0.
                kyy = 0.
                kxy = 0.

                for j in range(n):
                    #TODO save in buffer
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)

                    for i in range(m):
                        #TODO save in buffer
                        fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                        fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

{LINSTRAIN}

                # nonlinear strain eps_NL = {{phix^2/2, phiy^2/2, phix*phiy}}
                exx += 0.5*phix*phix
                eyy += 0.5*phiy*phiy
                gxy += phix*phiy

                # current stress state
                Nxx = A11*exx + A12*eyy + A16*gxy + B11*kxx + B12*kyy + B16*kxy
                Nyy = A12*exx + A22*eyy + A26*gxy + B12*kxx + B22*kyy + B26*kxy
                Nxy = A16*exx + A26*eyy + A66*gxy + B16*kxx + B26*kyy + B66*kxy
                Mxx = B11*exx + B12*eyy + B16*gxy + D11*kxx + D12*kyy + D16*kxy
                Myy = B12*exx + B22*eyy + B26*gxy + D12*kxx + D22*kyy + D26*kxy
                Mxy = B16*exx + B26*eyy + B66*gxy + D16*kxx + D26*kyy + D66*kxy

                for j in range(n):
                    gAu = f(j, eta, y1u, y1ur, y2u, y2ur)
                    gAueta = fp(j, eta, y1u, y1ur, y2u, y2ur)
                    gAv = f(j, eta, y1v, y1vr, y2v, y2vr)
                    gAveta = fp(j, eta, y1v, y1vr, y2v, y2vr)
                    gAw = f(j, eta, y1w, y1wr, y2w, y2wr)
                    gAweta = fp(j, eta, y1w, y1wr, y2w, y2wr)
                    gAwetaeta = fpp(j, eta, y1w, y1wr, y2w, y2wr)
                    for i in range(m):
                        fAu = f(i, xi, x1u, x1ur, x2u, x2ur)
                        fAuxi = fp(i, xi, x1u, x1ur, x2u, x2ur)
                        fAv = f(i, xi, x1v, x1vr, x2v, x2vr)
                        fAvxi = fp(i, xi, x1v, x1vr, x2v, x2vr)
                        fAw = f(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxi = fp(i, xi, x1w, x1wr, x2w, x2wr)
                        fAwxixi = fpp(i, xi, x1w, x1wr, x2w, x2wr)

                        col = col0 + DOF*(j*m + i)

{FINT}

    return fint
'''.format(BC_CDEF=BC_CDEF, BC_BLOCK=BC_BLOCK, LIMITS=LIMITS, MAPPING=MAPPING,
           POINT=POINT, FINPUT=FINPUT, ABD=abd_read(16),
           LINSTRAIN=LINSTRAIN.format(ind=' '*24), FINT='\n'.join(fint_lines)))
    return ''.join(out)


LINSTRAIN = """{ind}exx += cs[col+0]*(2/a)*fAuxi*gAu
{ind}eyy += cs[col+1]*(2/b)*fAv*gAveta + 1/r*cs[col+2]*fAw*gAw
{ind}gxy += cs[col+0]*(2/b)*fAu*gAueta + cs[col+1]*(2/a)*fAvxi*gAv
{ind}kxx += -cs[col+2]*(2/a*2/a)*fAwxixi*gAw
{ind}kyy += -cs[col+2]*(2/b*2/b)*fAw*gAwetaeta + 1/r*cs[col+1]*(2/b)*fAv*gAveta
{ind}kxy += (-2*cs[col+2]*(2/a)*fAwxi*(2/b)*gAweta
{ind}        + 1.5/r*cs[col+1]*(2/a)*fAvxi*gAv - 0.5/r*cs[col+0]*(2/b)*fAu*gAueta)"""


if __name__ == '__main__':
    with open(os.path.join(REPO, 'panels', 'models', 'cylshell_clpt_sanders.pyx'), 'w', newline='\n') as f:
        f.write(analytical_module())
    with open(os.path.join(REPO, 'panels', 'models', 'cylshell_clpt_sanders_num.pyx'), 'w', newline='\n') as f:
        f.write(numerical_module())
    print('written')
