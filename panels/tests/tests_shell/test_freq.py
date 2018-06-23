import numpy as np

from panels import Shell


def test_panel_freq():
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell',
                  ]:
        for prestress in [True, False]:
            print('Frequency Analysis, prestress={0}, model={1}'.format(
                  prestress, model))
            s = Shell()
            s.model = model
            s.a = 1.
            s.b = 0.5
            s.r = 1.e8
            s.alphadeg = 0.
            s.stack = [0, 90, -45, +45]
            s.plyt = 1e-3*0.125
            s.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
            s.rho = 1.3e3
            s.m = 11
            s.n = 12
            if prestress:
                s.Nxx_cte = -60.
                s.Nyy_cte = -5.
            eigvals, eigvecs = s.freq(sparse_solver=True, silent=False)
            if prestress:
                assert np.isclose(eigvals[0], 17.85875, rtol=0.001)
            else:
                assert np.isclose(eigvals[0], 39.31476, rtol=0.001)


if __name__ == '__main__':
    test_panel_freq()
