# T stiffener of three 2D domains, connections (`tstiff2d_clt_donnell`)

Derivation: `tstiff2d_clt_donnell.nb` (Mathematica 9), which writes the
connection matrices in Fortran form to `output_expressions_mathematica/`
(`fortran_<name>.txt`); `print_expressions_python.py` converts them into
Python statements in `output_expressions_python/`. The conversion script
imports `compmech.conecyl.sympytools`, a package that `panels` does not
depend on. `failed_attempts/` holds two earlier variants of the notebook
(`_conn2`, `_conn3`) with their outputs.

No kernel of `panels` is generated from this folder, and `panels` has no
`tstiff2d_clt_donnell` model: the T-stiffened panels of
`panels.multidomain.tstiff2d_1stiff_compression`, `tstiff2d_1stiff_freq`
and `tstiff2d_1stiff_flutter` are `MultiDomain` assemblies of `Shell`
domains (skin, base and flange) joined by the connections `'SSxcte'`,
`'SSycte'`, `'SB'` and `'BFycte'`, see `../../multidomain_penalization`.

## Model

Skin (approximation functions `f`, `g`), stiffener base (`p`, `q`) and
flange (`r`, `s`), each with its own Ritz constants and the degrees of
freedom `u, v, w`. The stiffness, geometric stiffness, mass and
aerodynamic matrices of each domain are those of the CLPT shell models
(`../../shells`); the notebook defines only the connection matrices, with
the penalty constants `kt` (translations) and `kr` (rotation):

- skin-base, over the area of the base (`a b/4`): the displacements of the
  skin at the offset `dpb`, `u + dpb w,x` and `v + dpb w,y`, and `w`, tied
  to those of the base, blocks `kCpp`, `kCpb` and `kCbbpb`;
- base-flange, along the line of the flange (`a/2`), with the flange
  normal to the base, `u_b = u_f`, `v_b = w_f`, `w_b = -v_f`, `w_b,y =
  w_f,y`, blocks `kCbbbf`, `kCbf` and `kCff`.

Only the upper triangle of the connection matrix is written. The blocks
`kCpb` and `kCbbpb` carry a factor `c1` that the notebook does not define.

## References

- Castro, S. G. P., and Donadon, M. V., "Assembly of semi-analytical models
  to address linear buckling and vibration of stiffened composite panels
  with debonding defect", Composite Structures, 160, 232-247, 2017.
  https://doi.org/10.1016/j.compstruct.2016.10.026
