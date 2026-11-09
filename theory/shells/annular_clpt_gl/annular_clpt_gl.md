# Annular-sector plate, `annular_clpt_gl`

Derivation and symbolic checks: `annular_clpt_gl.py` (sympy, run
`python annular_clpt_gl.py`). Kernels: `write_pyx.py`, the same file in every
folder `annular_clpt_*` (the model is the name of the folder), which writes
`panels/models/annular_clpt_gl_num.pyx` and `panels/models/annular_clpt_gl.pyx`
(`python write_pyx.py [folder]`).

Coordinates, local basis, inputs (`a` radial width, `b` arc length of the outer
edge, `r` outer radius; the web of a ring frame is an application) and the
laminate offset as in `../annular_clpt_donnell/annular_clpt_donnell.md`. The model has
the complete Green-Lagrange membrane strains of Amabili (2008), Eqs.
(1.166)-(1.168), and the linear changes of curvature of `annular_clpt_donnell`;
the membrane strains vanish for any finite rigid motion, as those of
`plate_clpt_gl`.

The checks are listed in the docstring of `annular_clpt_gl.py`; the kernels
use `[H(sigma, d)] = sum_r sigma_r d^2 eps_r/d d^2`, split as in every model:
`fkG_num` = `[H(sigma_L + Nbar, 0)]`, `fkC_num` = `[B]^T [ABD] [B] + [H(sigma, d)]
- [H(sigma_L, 0)]`.
