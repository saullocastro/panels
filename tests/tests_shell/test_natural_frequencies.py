import numpy as np
from structsolve import freq

from panels import Shell


models = ['plate_clpt_donnell_bardell', 'cylshell_clpt_donnell_bardell']

def test_panel_freq():
    for model in models:
        for prestress in [True, False]:
            print('Frequency Analysis, prestress={0}, model={1}'.format(
                  prestress, model))
            p = Shell()
            p.model = model
            p.a = 1.
            p.b = 0.5
            p.r = 1.e8
            p.stack = [0, 90, -45, +45]
            p.plyt = 1e-3*0.125
            p.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
            p.rho = 1.3e3
            p.m = 8
            p.n = 8
            p.Nxx = -60.
            p.Nyy = -5.
            k0 = p.calc_kC(silent=True)
            M = p.calc_kM(silent=True)
            if prestress:
                kG0 = p.calc_kG(silent=True)
                k0 += kG0
            eigvals, eigvecs = freq(k0, M, sparse_solver=True, silent=True)
            omegan = np.sqrt(-eigvals)
            if prestress:
                assert np.isclose(omegan[0], 17.6002, rtol=0.001)
            else:
                assert np.isclose(omegan[0], 39.1962, rtol=0.001)


if __name__ == '__main__':
    test_panel_freq()
