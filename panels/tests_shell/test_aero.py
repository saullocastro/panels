import numpy as np

from panels.shell import Shell


def test_aero():
    for model in ['plate_clpt_donnell_bardell',
                  'cylshell_clpt_donnell_bardell']:
        for atype in [3, 4]:
            print('Flutter Analysis Piston Theory, atype={0}, model={1}'.
                  format(atype, model))
            p = Shell()
            p.model = model
            p.a = 1.
            p.b = 0.5
            p.r = 1.e8
            p.alphadeg = 0.
            p.stack = [0, 90, -45, +45]
            p.plyt = 1e-3*0.125
            E2 = 8.7e9
            p.laminaprop = (142.5e9, E2, 0.28, 5.1e9, 5.1e9, 5.1e9)
            p.rho = 1.5e3
            p.m = 8
            p.n = 9

            # pre-stress applied when atype == 4
            p.Nxx = -60.
            p.Nyy = -5.

            # testing commong methodology based on betastar
            if atype == 4:
                betasstar = np.linspace(150, 350, 40)
            elif atype == 3:
                betasstar = np.linspace(670, 690, 40)
            betas = betasstar/(p.a**3/E2/(len(p.stack)*p.plyt)**3)
            p.beta = betas[0]
            p.freq(atype=4, sparse_solver=False, silent=True)
            out = np.zeros((len(betasstar), p.eigvals.shape[0]),
                    dtype=p.eigvals.dtype)
            for i, beta in enumerate(betas):
                p.beta = beta
                p.freq(atype=3, sparse_solver=False, silent=True)
                eigvals = p.eigvals*p.a**2/(np.pi**2*sum(p.plyts))*np.sqrt(p.rho/E2)
                out[i, :] = eigvals

            ind = np.where(np.any(out.imag != 0, axis=1))[0][0]
            if atype == 4:
                assert np.isclose(betas[ind], 347.16346, atol=0.1, rtol=0)
            elif atype == 3:
                assert np.isclose(betas[ind], 728.625, atol=0.1, rtol=0)

if __name__ == '__main__':
    out = test_aero()
