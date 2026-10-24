r"""Python code called from inside the compiled kernels

The kernels of :mod:`panels.models`, e.g. ``fk0`` and ``fkM``, read the
property :attr:`.Shell.ABD` and call :meth:`.Shell._get_lam_mid` after some
of their C ``double`` variables are already set. Whatever runs in these
calls must preserve the registers that the calling convention requires to be
preserved, otherwise the compiled caller continues with corrupted variables.

The matrix-matrix product of Intel MKL 2025.0 on Windows x64
(``cblas_dgemm``, reached by ``numpy.matmul`` and ``scipy.linalg.blas.dgemm``
for e.g. 13 x 13 matrices) was seen to return with the registers
``xmm8``-``xmm15`` changed, which the Windows x64 calling convention
requires to be preserved; ``MKL_CBWR=COMPATIBLE`` made it preserve them,
the matrix-vector and the dot products preserved them. With a single
product ``np.eye(13) @ ABD`` added to :meth:`.Shell._get_lam_ABD`, the
values of ``ABD`` being unchanged, ``cylshell_tsdt_sanders.fk0`` returned
7761 of its 22500 entries ``nan`` and others wrong by up to `10^{303}`
times the largest entry, with and without a laminate offset, while the
kernels of the other models were not affected. Hence the TSDT constitutive matrix
is assembled by blocks, see :meth:`.Shell._get_lam_ABD`. With a minimal
reproducer independent of panels, the oneMKL builds 2024.2.0, 2024.2.1,
2024.2.2 and 2025.0.0 changed these registers in ``cblas_dgemm`` for square
matrices of size 12 or more, 2023.2.0 to 2024.1.0 and 2025.0.1 to 2026.2.0
did not (AMD processor, Windows 11).

The analytical kernels must be finite and agree with the numerically
integrated ones, for every model and with and without a laminate offset.

"""
import sys
sys.path.append('../..')

import numpy as np
import pytest
from structsolve.sparseutils import finalize_symmetric_matrix

from panels import modelDB
from panels.shell import Shell


LAMINAPROP = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 3.8e9)
FOAM = (0.1e9, 0.1e9, 0.3, 0.04e9, 0.04e9, 0.04e9)


def _rel(A, B):
    A, B = A.toarray(), B.toarray()
    return np.abs(A - B).max()/np.abs(B).max()


@pytest.mark.parametrize('model', sorted(modelDB.db.keys()))
@pytest.mark.parametrize('offset', [0., 1.2e-3])
def test_analytical_kernels_finite_and_equal_numerical(model, offset):
    s = Shell(model=model, a=0.4, b=0.3,
              r=(0.5 if modelDB.db[model]['requires_r'] else None),
              stack=[0, 45, 90], plyts=[0.5e-3, 2e-3, 0.5e-3],
              laminaprops=[LAMINAPROP, FOAM, LAMINAPROP],
              rhos=[1600., 100., 1600.], offset=offset, m=6, n=5)
    size = s.get_size()
    mats = modelDB.db[model]['matrices']
    num = modelDB.db[model]['matrices_num']
    nx, ny = 2*s.m + 4, 2*s.n + 4
    k0 = mats.fk0(s, size, 0, 0)
    assert np.isfinite(k0.data).all()
    kC_num = num.fkC_num(np.zeros(size), s.ABD, s, size, 0, 0, nx, ny, 0)
    assert _rel(finalize_symmetric_matrix(k0),
                finalize_symmetric_matrix(kC_num)) < 1e-12
    kM = mats.fkM(s, size, 0, 0)
    assert np.isfinite(kM.data).all()
