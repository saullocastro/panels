import numpy as np

from structsolve import lb
from panels import Shell
from panels.plot_shell import plot_shell


def test_shell_lb():
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell',
                  ]:
        # ssss
        s = Shell()
        s.m = 12
        s.n = 12
        s.stack = [0, 90, -45, +45, +45, -45, 90, 0]
        s.plyt = 0.125e-3
        s.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        s.model = model
        s.a = 2.
        s.b = 1.
        s.r = 1.e8
        s.Nxx = -1
        eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
        assert np.isclose(eigvals[0], 157.5369886, atol=0.1, rtol=0)

        s.Nxx = 0
        s.Nyy = -1
        eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
        assert np.isclose(eigvals[0], 58.1488566, atol=0.1, rtol=0)

        # ssfs
        s = Shell()
        s.y2u = 1
        s.y2v = 1
        s.y2w = 1
        s.m = 12
        s.n = 13
        s.stack = [0, 90, -45, +45]
        s.plyt = 0.125e-3
        s.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
        s.model = model
        s.a = 1.
        s.b = 0.5
        s.r = 1.e8
        s.Nxx = -1
        eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
        assert np.isclose(eigvals[0], 15.8106238, atol=0.1, rtol=0)

        s.x2u = 1
        s.x2v = 1
        s.x2w = 1
        s.y2u = 0
        s.y2v = 0
        s.y2w = 0
        s.Nxx = 0
        s.Nyy = -1
        eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
        assert np.isclose(eigvals[0], 13.9105, atol=0.1, rtol=0)
        plot_shell(s, eigvecs[:, 0], vec='w')


if __name__ == '__main__':
    test_shell_lb()
