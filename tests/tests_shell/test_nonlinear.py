import numpy as np
from structsolve import solve

from panels import Shell


def test_nonlinear():
    m = 6
    n = 6
    for model in [
            'plate_clpt_donnell_bardell',
            'cylshell_clpt_donnell_bardell',
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

        # solving using Modified Newton-Raphson method
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

        #initial
        fext = s.calc_fext()
        c0 = solve(s.calc_kC(), fext, silent=True)
        plot_mesh, fields = s.uvw(c=c0)
        print('  linear wmax', fields['w'].max())
        assert np.isclose(fields['w'].max(), 0.0026619, rtol=0.01)

        count = 0
        N = s.get_size()
        fint = s.calc_fint(c=c0)
        Ri = fint - fext
        dc = np.zeros(N)
        ci = c0.copy()
        epsilon = 1.e-4
        KT = s.calc_kT(c=c0)
        D = s.calc_kC().diagonal() # at beginning of load increment
        while True:
            #print('  count', count)
            dc = solve(KT, -Ri, silent=True)
            c = ci + dc
            fint = np.asarray(s.calc_fint(c=c))
            Ri = fint - fext
            crisfield_test = scaling(Ri, D)/max(scaling(fext, D), scaling(fint, D))
            #print('    crisfield_test', crisfield_test)
            if crisfield_test < epsilon:
                #print('    converged')
                break
            count += 1
            KT = s.calc_kT(c=c)
            ci = c.copy()
            if count > 1000:
                raise RuntimeError('Not converged!')

        plot_mesh, fields = s.uvw(c=c)
        print('  nonlinear wmax', fields['w'].max())
        assert np.isclose(fields['w'].max(), 0.004574, rtol=0.01)


if __name__ == '__main__':
    test_nonlinear()
