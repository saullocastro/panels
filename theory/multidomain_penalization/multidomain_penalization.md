# Connections between domains and prescribed displacements

Derivations (sympy) of the penalty matrices of `MultiDomain`:

- `connections.py`: the connections between two domains, written to
  `output_expressions_python_new/sympy_<name>.txt`;
- `bfxcte_face.py`: symbolic checks of the connection `'BFxcte'` at a face
  of a cylinder (key `'face'`): the Kirchhoff-Love displacements of a point
  of the normal, the rotation about the circumferential axis, the zero
  residuals of the six rigid motions of a cylinder and an annular plate
  whose edge is at the radius of the face with the Sanders-Koiter term
  `z v/r` at the face (key `'kinematics'`, default `'sanders'`), and, for
  a Donnell cylinder with `kinematics='model'`, the residual `-z_f Omega`
  of a rigid rotation about the axis, of relative order `h/(2 r)`;
- `bfycte_face.py`: symbolic checks of the connection `'BFycte'` at a face
  of a cylinder, for a flat plate along a generator (a blade, the web of a
  Z stringer): the Kirchhoff-Love displacements of a point of the normal
  and its rotation about the axis `x`, `-w,y + v/r`, the zero residuals of
  the six rigid motions when the root of the plate is at the radius of the
  face, with the Sanders-Koiter terms at the face, and the residuals
  `-z_f v/r` and `-v/r` with the Donnell kinematics (`kinematics='model'`);
- `kCpd.py`: the prescribed displacements at a point and along a line,
  written to `output_expressions_python/sympy_<name>.txt`;
- `force_crack_creation.py`: the terms of the creation of a crack in a
  skin-base interface with a traction-separation law (D'Souza 2024); it
  contains absolute paths of its author's machine and does not run as is.

No script writes the kernels, which are in
`panels/multidomain/connections/`.

## Penalty energy

Each connection adds, for each quantity `q_k` that must be continuous,

    U = 1/2 sum_k k_k int (q_k^1 - q_k^2)^2

along the connected edges, or over the connected area for `'SB'`, with the
penalty constants `kt` (translations) and `kr` (rotations) of
`calc_kt_kr` (`panels/multidomain/connections/penalties.py`). Expanding the
square gives the blocks `11`, `12` and `22`; `21 = 12^T` is not written,
only the upper triangle is assembled. With the null-space method, the
default (`conn_method='null-space'`), the same quantities give the linear
constraints `B c = 0` instead.

## Connections (`connections.py`)

Models with 3 DOFs `u, v, w` (CLPT), `x`, `y` the local axes of each
domain, `rinv = 1/r` for the Sanders-Koiter kinematics and `0` for plates
and the Donnell kinematics:

| `func`     | where                                   | quantities                                  | kernels |
|------------|-----------------------------------------|---------------------------------------------|---------|
| `'SSxcte'` | edges `x1 = xcte1`, `x2 = xcte2`        | `u, v, w` (`kt`), `w,x` (`kr`)              | `kCSSxcte.pyx` |
| `'SSycte'` | edges `y1 = ycte1`, `y2 = ycte2`        | `u, v, w` (`kt`), `w,y` (`kr`)              | `kCSSycte.pyx` |
| `'BFycte'` | base (1) and flange (2), normal, along `x` | `u1 = u2, v1 = w2, w1 = -v2`; `w,y - rinv v` (`kr`) | `kCBFycte.pyx` |
| `'BFxcte'` | base (1) and flange (2), normal, along `y` | `u1 = w2, v1 = v2, w1 = -u2`; `w,x` (`kr`) | `kCBFxcte.pyx` |
| `'BFxycte'`| edge `x1 = xcte1` of (1), edge `y2 = ycte2` of (2), angle `theta` | `u1 = -cos(theta) v2 + sin(theta) w2`, `v1 = u2`, `w1 = sin(theta) v2 + cos(theta) w2`; rotation about the common axis | `kCBFxycte.pyx` (block `12`) |
| `'BFxcte'`, `face='lower'`/`'upper'` | as `'BFxcte'`, the line `x1 = xcte1`, `z = d1 -+ h1/2` of the base | `u1(z) = w2, v1(z) = v2, w1 = -u2`; `w,x` (`kr`) | `kCBFxcte_face.fkCBFxcte_face` |
| `'BFycte'`, `face='lower'`/`'upper'` | as `'BFycte'`, the line `y1 = ycte1`, `z = d1 -+ h1/2` of the base | `u1(z) = u2, v1(z) = w2, w1 = -v2`; `w,y - rinvf1 v` (`kr`) | `kCBFycte_face.fkCBFycte_face` |
| `'SB'`     | panel (1) above panel (2), over the area | `u(z), v(z), w` at the common faces `z = d1 - h1/2`, `z = d2 + h2/2` | `skin_base.fkCSB` |

For `'BFxcte'` with the key `'face'` the edge `x2 = xcte2` of the flange
(`0` for `'lower'`, `a2` for `'upper'`) is tied to the face of the base,
with the Kirchhoff kinematics of the base below, the term `z v/r` with
`rinvf1 = 1/r1` for every `cylshell_*` model by default (`kinematics=
'sanders'`), or `rinv1` of the model with `kinematics='model'`; `'lower'`
and `'upper'` refer to the coordinate `z1` of the base, as the faces of
`'SB'` to the `z` of each panel; the two edges share `eta`
and span the same angle, e.g. a cylinder and an annular plate whose edge is
at the radius `r1 + z` of the face. `'BFycte'` with the key `'face'` is
the same along a line `y1 = ycte1` (edge `y2 = 0` of the flange for
`'lower'`, `b2` for `'upper'`, `a1 = a2`), the rotation of the normal about
`x` at the face being `-w1,y + rinvf1 v1`, with the same `rinvf1` as the
term `z v/r`. The key `'kinematics'` without `'face'` raises a `ValueError`
in both connections. For `'SB'` each panel has its own Kirchhoff kinematics, `u(z) = u - z w,x`,
`v(z) = (1 + z rinv) v - z w,y`, with `z` measured from its reference
surface and the offsets `d1`, `d2`.

Models with 5 DOFs `u, v, w, phix, phiy` (FSDT, TSDT, see
`../shells/<model>`, e.g. `../shells/cylshell_tsdt_sanders`), with the
displacement field

    u(z) = u + z phix - c1 [(z - d)^3 + d^3] (phix + w,x)
    v(z) = v + z Phiy - c1 [(z - d)^3 + d^3] (phiy + w,y)
    w(z) = w,     Phiy = phiy + rinv v

`'SSxcte'` and `'SSycte'` penalize `u, v, w` (`kt`), `phix`, `Phiy` and,
for the TSDT, the derivative of `w` normal to the edge (`kr`): kernels
`fkCSSxcte_sdt`, `fkCSSycte_sdt` of `kCsdt.py`. `'SB'` ties `u(z), v(z),
w` at the faces, with an optional rotation penalty: `fkCSB_sdt` of
`kCsdt.py`. `'BFycte'` and `'BFxcte'` penalize the rotations of the
normals, `Phiy` and `phix`: the functions `*_sdt` of `kCBFycte.pyx` and
`kCBFxcte.pyx`.

The kernels of `kCSB_dmg.pyx` (`'SB_TSL'`) are not derived here.

## Prescribed displacements (`kCpd.py`)

Penalty energy `1/2 (ku u^2 + kv v^2 + kw w^2)` at a point `(xp, yp)`, or
integrated along a line `x = const` (`b/2` the Jacobian) or `y = const`
(`a/2`), with the constants of `calc_ku_kv_kw_point_pd`,
`calc_ku_kv_kw_line_pd_xcte` and `calc_ku_kv_kw_line_pd_ycte`: kernels
`fkCpd`, `fkCld_xcte`, `fkCld_ycte` of `kCpd.pyx`.

## Selection in panels

`MultiDomain(panels, conn)` with `conn` a list of dictionaries, e.g.
`dict(p1=p01, p2=p02, func='SSycte', ycte1=0, ycte2=p02.b)`, see
`MultiDomain.get_kC_conn`; `conn_method='penalty'` selects the penalty
kernels above.

## References

- Castro, S. G. P., and Donadon, M. V., "Assembly of semi-analytical models
  to address linear buckling and vibration of stiffened composite panels
  with debonding defect", Composite Structures, 160, 232-247, 2017.
  https://doi.org/10.1016/j.compstruct.2016.10.026
- D'Souza, N., "Multi-Domain Semi-Analytical Cohesive Zone Approach", MSc
  thesis, Delft University of Technology, 2024.
  https://resolver.tudelft.nl/uuid:4709f888-c032-4d9f-97e3-402dfdd47cf1
