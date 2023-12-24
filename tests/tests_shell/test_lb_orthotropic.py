import numpy as np
from structsolve import lb
import matplotlib

from panels import Shell
from panels.plot_shell import plot_shell


def test_lb_orthotropic():
    #NOTE ssss boundary conditions by default
    s = Shell()
    s.m = 12
    s.n = 12
    s.stack = [0, 90, -45, +45, +45, -45, 90, 0]
    s.plyt = 0.125e-3
    s.laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    s.model = 'cylshell_clpt_donnell_bardell'
    s.a = 2.
    s.b = 1.
    # radius
    s.r = 2.

    # compression
    s.Nxx = -10
    # shear
    s.Nxy = 1

    eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
    assert np.isclose(eigvals[0], 1375.367, atol=0.1, rtol=0.01)
    plot_shell(s, eigvecs[:, 0], vec='w')
