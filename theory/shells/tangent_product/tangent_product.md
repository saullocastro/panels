# Product of the tangent stiffness matrix by a vector

Derivation and symbolic verification: `tangent_product.py` (sympy), kernels:
`write_pyx.py`, which writes `calc_kTv` at the end of every
`panels/models/<model>_num.pyx` (also called by `fsdt_tsdt/write_pyx.py` and
`cylshell_clpt_sanders/write_pyx.py`). Tests:
`tests/tests_shell/test_tangent_product.py`,
`tests/multidomain/test_tangent_product_multidomain.py`. Documentation:
`doc/source/tangent_product.rst`.

## Kinematics shared by all the models

All 9 geometrically non-linear models (plate, Donnell and Sanders cylinders;
CLPT, FSDT and TSDT) have, at an integration point, generalized strains

    eps(c) = ([B_L] + [B_NL]/2) {c},      {beta} = [G] {c}

with `[B_NL]` linear in the two rotations `beta` of the von Kármán terms,
which enter only the membrane rows:

    d[B_NL]/dbeta_x = [Gx; 0; Gy; 0],     d[B_NL]/dbeta_y = [0; Gy; Gx; 0]

| model                    | `beta`                              | source of `[B]`, `[G]`             |
|--------------------------|-------------------------------------|------------------------------------|
| plate/cylshell CLPT Donnell | `w,x`, `w,y`                     | `tangent_product._clpt_donnell` (as `plate_clpt_donnell.py`, `cylshell_clpt_donnell.py`) |
| cylshell CLPT Sanders    | `phix = -w,x`, `phiy = -w,y + v/r`  | `cylshell_clpt_sanders.shape_functions` |
| FSDT/TSDT                | `w,x`, `w,y - sk v/r`               | `fsdt_tsdt.shape_functions`        |

The factor 1/2 of `[B_NL]` gives `beta_x^2/2`, `beta_y^2/2` and
`beta_x beta_y`.

## Product

    F_int = int ([B_L] + [B_NL])^T {sigma} dA,    {sigma} = [ABD] eps(c)

    K_T v = d/dt F_int(c + t v) |_(t=0)
          = int ( ([B_L] + [B_NL])^T [ABD] ([B_L] + [B_NL]) v
                  +  [G]^T [N] [G] v ) dA

since `d eps(c + t v)/dt = ([B_L] + [B_NL]) v` (the strains are a quadratic
form whose bilinear form is `[B_L] + [B_NL]`) and
`d[B_NL]^T/dt sigma = [G]^T [N] [G] v` from the derivatives above,
`[N] = [[Nxx, Nxy], [Nxy, Nyy]]`, `[ABD]` the constitutive matrix. These are
the two parts assembled by `fkC_num` (`KC0 + KCNL + KGNL`) and
`fkG_num` (`KG(Nbar + NL)`). The constant membrane stresses `Nxx_cte`,
`Nyy_cte`, `Nxy_cte`, whose geometric stiffness `Shell.calc_kC` adds to
`K_T`, are added to `[N]`.

Per integration point, with the basis functions tabulated once:

1. `beta = [G] c` and `dbeta = [G] v` (one loop over the terms);
2. `eps = ([B_L] + [B_NL]/2) c` and `deps = ([B_L] + [B_NL]) v` (one
   loop);
3. `sigma = [ABD] eps`, `dsigma = [ABD] deps`, `[N]` from `sigma` plus the
   constant stresses;
4. `kTv_i += w (([B_L] + [B_NL])_i^T dsigma + [G_i]^T [N] dbeta)` (one
   loop).

The cost is `O(nx ny m n)`, against `O(nx ny m^2 n^2)` for the assembly of
`K_T`.

`check(model)` differentiates `F_int` of one term symbolically, with a
symbolic symmetric constitutive matrix `[F]`, and compares it with the
generated expressions; `[B_NL]` and `[G]` being linear in the degrees of
freedom, the identity extends
to any number of terms. Run `python tangent_product.py`.

## Follower pressure loads

`Shell.calc_kTv` adds the product of the load stiffness `-dF_p/dc` as the
central difference

    -dF_p/dc v = -(F_p(c + t v) - F_p(c - t v)) / (2 t)

exact for any `t`, `F_p(c)` being affine (`follower='linear'`) or quadratic
(`follower='quadratic'`) in `c`; `t = max(|c|/|v|, 1)` avoids the
cancellation of large and small terms.
