# Annular-sector plate, `annular_clpt_tl`

Derivation and symbolic checks: `annular_clpt_tl.py` (sympy, run
`python annular_clpt_tl.py`). Kernels: `write_pyx.py`, the same file in every
folder `annular_clpt_*` (the model is the name of the folder), which writes
`panels/models/annular_clpt_tl_num.pyx` and `panels/models/annular_clpt_tl.pyx`
(`python write_pyx.py [folder]`).

Coordinates, local basis, inputs (`a` radial width, `b` arc length of the outer
edge, `r` outer radius; the web of a ring frame is an application) and the
laminate offset as in `../annular_clpt_donnell/annular_clpt_donnell.md`. The model has
the Green-Lagrange membrane strains of `annular_clpt_gl` and the changes of
curvature of the deformed normal, measured with the area vector `A = g1 x g2`
as in `plate_clpt_tl`; all six strains vanish for any finite rigid motion.
No published total-Lagrangian formulation in polar coordinates was found; the
model rests on the symbolic checks below and on the tests of `panels`.

The checks are listed in the docstring of `annular_clpt_tl.py`; the kernels
use `[H(sigma, d)] = sum_r sigma_r d^2 eps_r/d d^2`, split as in every model:
`fkG_num` = `[H(sigma_L + Nbar, 0)]`, `fkC_num` = `[B]^T [ABD] [B] + [H(sigma, d)]
- [H(sigma_L, 0)]`.
