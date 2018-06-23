import numpy as np

from panels.shell import Shell


def test_aero():
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell']:
        for atype in [3, 4]:
            print('Flutter Analysis Piston Theory, atype={0}, model={1}'.
                  format(atype, model))
            s = Shell()
            s.model = model
            s.a = 1.
            s.b = 0.5
            s.r = 1.e8
            s.alphadeg = 0.
            s.stack = [0, 90, -45, +45]
            s.plyt = 1e-3*0.125
            E2 = 8.7e9
            s.laminaprop = (142.5e9, E2, 0.28, 5.1e9, 5.1e9, 5.1e9)
            s.rho = 1.5e3
            s.m = 8
            s.n = 9

            # pre-stress applied when atype == 4
            s.Nxx = -60.
            s.Nyy = -5.

            # testing commong methodology based on betastar
            if atype == 4:
                betasstar = np.linspace(150, 350, 40)
            elif atype == 3:
                betasstar = np.linspace(670, 690, 40)
            betas = betasstar/(s.a**3/E2/(len(s.stack)*s.plyt)**3)
            s.beta = betas[0]
            eigvals, eigvecs = s.freq(atype=4, sparse_solver=False, silent=True)
            out = np.zeros((len(betasstar), eigvals.shape[0]),
                    dtype=eigvals.dtype)
            for i, beta in enumerate(betas):
                s.beta = beta
                eigvals, eigvecs = s.freq(atype=3, sparse_solver=False, silent=True)
                eigvals = eigvals*s.a**2/(np.pi**2*sum(s.plyts))*np.sqrt(s.rho/E2)
                out[i, :] = eigvals

            ind = np.where(np.any(out.imag != 0, axis=1))[0][0]
            if atype == 4:
                assert np.isclose(betas[ind], 347.16346, atol=0.1, rtol=0)
            elif atype == 3:
                assert np.isclose(betas[ind], 728.625, atol=0.1, rtol=0)

if __name__ == '__main__':
    out = test_aero()
