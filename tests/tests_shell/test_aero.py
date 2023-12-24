import numpy as np

from structsolve import freq, lb
from panels.shell import Shell


def test_aero():
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell']:
        for pre_stress in [False, True]:
            print('Flutter Analysis Piston Theory, consider_pre_stress={0}, model={1}'.
                  format(pre_stress, model))
            s = Shell()
            s.model = model
            s.a = 1.
            s.b = 0.5
            s.r = 1.e8
            s.stack = [0, 90, -45, +45]
            s.plyt = 1e-3*0.125
            E2 = 8.7e9
            s.laminaprop = (142.5e9, E2, 0.28, 5.1e9, 5.1e9, 5.1e9)
            s.rho = 1.5e3
            s.m = 8
            s.n = 9

            if pre_stress:
                s.Nxx = -60.
                s.Nyy = -5.
            else:
                s.Nxx = 0
                s.Nyy = 0

            # testing commong methodology based on betastar
            if pre_stress:
                betasstar = np.linspace(50, 350, 40)
            else:
                betasstar = np.linspace(300, 690, 40)
            betas = betasstar/(s.a**3/E2/(len(s.stack)*s.plyt)**3)
            s.beta = betas[0]
            kC = s.calc_kC()
            kG = s.calc_kG()
            kA = s.calc_kA()
            kM = s.calc_kM()

            eigvals, eigvecs = freq(kC + kG + kA, kM, sparse_solver=True, silent=True)
            out = np.zeros((len(betasstar), eigvals.shape[0]),
                    dtype=eigvals.dtype)
            for i, beta in enumerate(betas):
                s.beta = beta
                kA = s.calc_kA()
                eigvals, eigvecs = freq(kC + kG + kA, kM, sparse_solver=True, silent=True)
                omegan = (-eigvals)**0.5
                omegan = omegan*s.a**2/(np.pi**2*sum(s.plyts))*np.sqrt(s.rho/E2)
                out[i, :] = omegan

            ind = np.where(np.any(out.imag != 0, axis=1))[0][0]
            if pre_stress:
                assert np.isclose(betas[ind], 129.663, atol=0.1, rtol=0)
            else:
                assert np.isclose(betas[ind], 358.875, atol=0.1, rtol=0)

if __name__ == '__main__':
    test_aero()
