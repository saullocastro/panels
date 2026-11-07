# Blade stiffener with a 1D flange (`bladestiff1d_clt_donnell`)

Derivation: `bladestiff1d_clt_donnell.py` (sympy, run `python
bladestiff1d_clt_donnell.py`), which writes the integrands of the matrices
of the flange to `output_expressions_python/sympy_<name>.txt`. The
through-thickness integration of the mass terms is in
`../../stiffener/mass_matrix_1D_stiffeners.py`. Kernels:
`panels/stiffener/models/bladestiff1d_clt_donnell.pyx`, `k0f: fkCf`, `kG0f:
fkGf`, `kMf: fkMf`; no script writes this module.

## Model

Blade stiffener of a `StiffPanelBay` along `x` at `y = ys`, below the skin:

                 || --> flange (beam, height bf, thickness hf)
               ======  --> base (padup, width bb, thickness hb)
      =========================  --> skin (thickness h)

- The base is a `Shell` with the approximation functions, Ritz constants
  and boundary conditions of the bay, integrated over `ys - bb/2 <= y <= ys
  + bb/2` with the offset `-(h/2 + hb/2)`: perfectly bonded below the skin
  and following its displacement field. Its matrices are those of the
  shell models (`../../shells/plate_clpt_donnell`, ...).
- The flange is a beam along `x` whose displacements are those of the skin
  along `y = ys`: the functions along `y` (`g`) are evaluated at
  `eta(ys)`, and only the functions along `x` (`f`) are integrated. Its
  cross-section does not deform, so local buckling of the flange is not
  represented; see `../bladestiff2d_clt_donnell`.

## Kinematics and strains of the flange

Euler-Bernoulli beam with the Kirchhoff kinematics of the skin, the
centroid of the flange at the distance `df = bf/2 + hb + h/2` below the
mid-surface of the skin. The generalized strains of the script are
`(u,x, w,x, -w,x, -w,xx, -w,xy)`; the contributions of the second and third
rows, with `E2`, `E3`, `S2`, cancel, and the strain energy of the flange is

    U = 1/2 int_0^a bf [E1 (u,x + df w,xx)^2 + F1 w,xx^2
                        - 2 S1 (u,x + df w,xx) w,xy + (GJ/bf) w,xy^2] dx

all at `y = ys`.

## Constitutive law

Laminated flange whose plies are free along its height (`sigma_2 = 0`),
with the reduced stiffnesses `Qbar_ij` of each ply `k` of thickness `h_k`
at the distance `y_k` from the mid-plane of the flange laminate
(`BladeStiff1D._rebuild()`):

    E1 = sum_k h_k (Q11 - Q12^2/Q22)               axial stiffness per unit height
    S1 = -sum_k y_k h_k (Q16 - Q12 Q26/Q22)         axial-twist coupling
    F1 = bf^2/12 E1                                 bending about the centroid
    GJ = 4 bf D66                                   Saint-Venant torsion of a thin strip

as Eq. (16) of Castro et al. (2016) for `E1`, `S1`.

## Matrices

    k0f  = (a/2) bf int gAsf^T ksf gBsf dxi            constitutive stiffness
    kG0f = (a/2) int Fx w,x^T w,x dxi                   geometric stiffness, axial force Fx
    kMf  = (a/2) hf bf rho int gA5^T msf gB5 dxi        mass, gA5 = (u, v, w, -w,x, -w,y)

`msf` couples the translations and rotations with `-df` and has the
rotational inertia `df^2 + bf^2/12` (written `bf^2/3 + bf (h + 2 hb)/2 + (h
+ 2 hb)^2/4` in the script).

## Selection in panels

`StiffPanelBay.add_bladestiff1d(ys, ...)` creates a `BladeStiff1D`, whose
attribute `model` is `'bladestiff1d_clt_donnell'`;
`panels.stiffener.modelDB.db['bladestiff1d_clt_donnell']`: `dofs = 3`,
`e_num = 6`, linear static and linear buckling analyses (no non-linear
static analysis). The skin must use a model with 3 DOFs.

## References

- Castro, S. G. P., Guimarães, T. A. M., Rade, D. A., and Donadon, M. V.,
  "Flutter of stiffened composite panels considering the stiffener's base
  as a structural element", Composite Structures, 140, 36-43, 2016.
  https://doi.org/10.1016/j.compstruct.2015.12.056
