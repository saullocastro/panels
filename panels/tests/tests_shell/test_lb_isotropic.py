import numpy as np
from structsolve import lb
import matplotlib

from panels import Shell
from panels.plot_shell import plot_shell


def test_lb_isotropic():
    #NOTE ssss boundary conditions by default
    s = Shell()
    s.m = 12
    s.n = 12
    s.stack = [0, 90, -45, +45, +45, -45, 90, 0]
    thickness = 1e-3
    E = 70e9
    nu = 0.33
    s.a = 2.
    s.b = 1.
    s.r = 2.

    # compression
    # it is possible to apply any static load if necessary
    s.Nxx = -1
    # shear
    s.Nxy = -1

    s.plyt = thickness
    s.laminaprop = (E, nu)
    s.model = 'cylshell_clpt_donnell_bardell'
    # radius

    eigvals, eigvecs = lb(s.calc_kC(), s.calc_kG(), silent=True)
    plot_shell(s, eigvecs[:, 0], vec='w', filename='example_isotropic.png')
