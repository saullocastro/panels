r"""Mass matrix of the 1D flange of :class:`.BladeStiff1D`

The kinetic energy of the flange, of height `b_f` and thickness `h_f`, below
the panel with its centroid at `z = -d_f`, is obtained integrating
`\rho (u_f^2 + v_f^2 + w^2)` with `u_f = u + z \phi_x` and `v_f = v + z
\phi_y` over the height, exactly, and along `x` with Gauss-Legendre points.
This checks the coupling terms `-d_f` of the kernel ``fkMf``, which were `-2
d_f` up to version 0.7.1 as in Eq. (26) of Castro et al. (2016), and the
thickness `h_b` of the base entering `d_f` and `k_{mf44}`.
"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.stiffener import modelDB
from panels.stiffpanelbay import StiffPanelBay


laminaprop = (105.e9, 10.5e9, 0.25, 5.25e9, 5.25e9, 5.25e9)
stack = [0, 90, 90, 0, 0]
plyt = 0.134e-3
tskin = len(stack)*plyt
b = 100*tskin


@pytest.mark.parametrize('base', [False, True])
def test_flange_mass_matrix(base):
    bay = StiffPanelBay()
    bay.a = 2*b
    bay.b = b
    bay.r = None
    bay.model = 'plate_clpt_donnell'
    bay.stack = stack
    bay.plyt = plyt
    bay.laminaprop = laminaprop
    bay.rho = 1600.
    bay.m = 6
    bay.n = 5
    bay.add_panel(0, b/3)
    bay.add_panel(b/3, b)
    kwargs = dict(bb=10*tskin, bstack=stack, bplyt=plyt,
                  blaminaprop=laminaprop) if base else {}
    bay.add_bladestiff1d(ys=b/3, bf=9*tskin, fstack=stack, fplyt=plyt,
                         flaminaprop=laminaprop, **kwargs)
    st = bay.bladestiff1ds[0]
    st._rebuild()
    h = 0.5*sum(st.panel1.plyts) + 0.5*sum(st.panel2.plyts)
    assert np.isclose(st.hb, len(stack)*plyt if base else 0.)
    assert np.isclose(st.dbf, st.bf/2 + st.hb + h/2)

    size = bay.get_size()
    mod = modelDB.db[st.model]['matrices']
    kM = finalize_symmetric_matrix(mod.fkMf(st.ys, st.rho, h, st.hb, st.hf,
            bay.a, bay.b, st.bf, st.dbf, bay.m, bay.n,
            bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
            bay.x1v, bay.x1vr, bay.x2v, bay.x2vr,
            bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
            bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
            bay.y1v, bay.y1vr, bay.y2v, bay.y2vr,
            bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
            size=size, row0=0, col0=0)).toarray()

    s = Shell(model=bay.model, a=bay.a, b=bay.b, m=bay.m, n=bay.n,
              stack=bay.stack, plyt=bay.plyt, laminaprop=bay.laminaprop)
    for k in ['x1u', 'x1ur', 'x2u', 'x2ur', 'x1v', 'x1vr', 'x2v', 'x2vr',
              'x1w', 'x1wr', 'x2w', 'x2wr', 'y1u', 'y1ur', 'y2u', 'y2ur',
              'y1v', 'y1vr', 'y2v', 'y2vr', 'y1w', 'y1wr', 'y2w', 'y2wr']:
        setattr(s, k, getattr(bay, k))
    assert s.get_size() == size

    xi, wx = np.polynomial.legendre.leggauss(2*bay.m + 10)
    xs = (xi + 1)*bay.a/2
    ys = np.full_like(xs, st.ys)
    zc = -st.dbf
    rng = np.random.default_rng(0)
    for _ in range(3):
        c = rng.standard_normal(size)
        s.uvw(c, xs=xs, ys=ys)
        u, v, w = s.fields['u'], s.fields['v'], s.fields['w']
        phix, phiy = s.fields['phix'], s.fields['phiy']
        e = st.bf*((u + zc*phix)**2 + st.bf**2/12*phix**2
                 + (v + zc*phiy)**2 + st.bf**2/12*phiy**2
                 + w**2)
        ref = st.rho*st.hf*np.sum(wx*e)*bay.a/2
        assert np.isclose(c @ kM @ c, ref, rtol=1e-12)
