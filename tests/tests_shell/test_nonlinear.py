"""Geometrically nonlinear static analysis with a full Newton-Raphson method

The applied compression is about 37 times the first linear buckling load, deep
in the postbuckling regime, where several stable equilibrium branches exist.
The reference deflection corresponds to the branch reached by a load-controlled
path-following analysis with a secant predictor, which returns the same value
with 40, 80 and 160 load increments; coarser incrementations may jump to other
stable branches. Here the full load is applied in a single step, starting from
the linear solution, which reaches that same branch.

Besides the converged deflection, the test checks that the Newton-Raphson
iterations converge quadratically, which only happens when the tangent
stiffness matrix is the exact derivative of the internal force vector.
"""
import sys
sys.path.append('../..')

import numpy as np
from structsolve import solve

from panels.shell import Shell


def scaling(vec, D):
    """
        A. Peano and R. Riccioni, Automated discretisatton error
        control in finite element analysis. In Finite Elements m
        the Commercial Enviror&ent (Editei by J. 26.  Robinson),
        pp. 368-387. Robinson & Assoc., Verwood.  England (1978)
    """
    non_nulls = ~np.isclose(D, 0)
    vec = vec[non_nulls]
    D = D[non_nulls]
    return np.sqrt((vec*np.abs(1/D))@vec)


def test_nonlinear():
    m = 6
    n = 6
    for model in [
            'plate_clpt_donnell',
            'cylshell_clpt_donnell',
            'cylshell_clpt_sanders',
            'plate_fsdt_donnell',
            'plate_tsdt_donnell',
                  ]:
        print('Testing model: %s' % model)
        s = Shell()

        s.model = model
        s.x1u = 0
        s.x1ur = 1
        s.x2u = 1
        s.x2ur = 1

        s.x1v = 1
        s.x1vr = 1
        s.x2v = 1
        s.x2vr = 1

        s.x1w = 0
        s.x1wr = 1
        s.x2w = 0
        s.x2wr = 1

        s.y1u = 1
        s.y1ur = 1
        s.y2u = 1
        s.y2ur = 1

        s.y1v = 0
        s.y1vr = 1
        s.y2v = 1
        s.y2vr = 1

        s.y1w = 0
        s.y1wr = 1
        s.y2w = 0
        s.y2wr = 1

        s.a = 4.
        s.b = 1.
        s.r = 1.e15
        s.stack = [90, 0, 90, 0]
        s.plyt = 1e-3*0.125
        E11 = 142.5e9
        E22 = E11/20
        G12 = G13 = G23 = 0.5*E22
        s.laminaprop = (E11, E22, 0.25, G12, G12, G12)
        s.m = m
        s.n = n

        load = 700
        Nxx = load/s.b
        # distributed axial load
        s.add_distr_load_fixed_x(s.a, funcx=lambda y: -Nxx, funcy=None, funcz=None, cte=False)
        # perturbation load
        s.add_point_load(s.a/2., s.b/2., 0, 0, 0.001, cte=True)

        #initial
        fext = s.calc_fext()
        c = solve(s.calc_kC(), fext, silent=True)
        plot_mesh, fields = s.uvw(c=c)
        print('  linear wmax', fields['w'].max())
        assert np.isclose(fields['w'].max(), 0.0026619, rtol=0.01)

        # solving using the full Newton-Raphson method, with the tangent
        # stiffness matrix updated at every iteration
        D = s.calc_kC().diagonal() # at beginning of load increment
        epsilon = 1.e-10
        errors = []
        while True:
            fint = s.calc_fint(c=c)
            Ri = fint - fext
            crisfield_test = scaling(Ri, D)/max(scaling(fext, D), scaling(fint, D))
            errors.append(crisfield_test)
            print('  iteration %d, crisfield_test %1.3e' % (len(errors) - 1, crisfield_test))
            if crisfield_test < epsilon:
                break
            if len(errors) > 30:
                raise RuntimeError('Not converged!')
            KT = s.calc_kT(c=c)
            c = c + solve(KT, -Ri, silent=True)

        plot_mesh, fields = s.uvw(c=c)
        print('  nonlinear wmax', fields['w'].max())
        assert np.isclose(fields['w'].max(), 0.004841, rtol=0.01)

        # quadratic convergence: once in the asymptotic range, the order
        # log(e_k+1)/log(e_k) approaches 2, whereas an inconsistent tangent
        # gives an order of 1 and needs hundreds of iterations
        assert len(errors) <= 15, 'too many iterations: %d' % len(errors)
        orders = [np.log(e1)/np.log(e0) for e0, e1 in zip(errors[:-1], errors[1:])
                  if 1.e-15 < e1 and e0 < 1.e-2]
        print('  convergence orders', orders)
        if 'fsdt' in model or 'tsdt' in model:
            #NOTE for this very thin plate, a/h = 8000, the residual of the
            #     shear deformation theories reaches its round-off floor, of
            #     the order of eps*(G/E)*(a/h)**2 ~ 1e-9, within the asymptotic
            #     range, where the order cannot be measured; their quadratic
            #     convergence is pinned by test_tangent_consistency.py
            continue
        assert len(orders) >= 2
        assert min(orders) > 1.6, 'convergence is not quadratic: %s' % orders


if __name__ == '__main__':
    test_nonlinear()
