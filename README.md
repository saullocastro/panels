Github Actions status:

[![Actions Status](https://github.com/saullocastro/panels/workflows/pytest/badge.svg)](https://github.com/saullocastro/panels/actions)

Coverage status:

[![Codecov Status](https://codecov.io/gh/saullocastro/panels/branch/main/graph/badge.svg?token=KD9D8G8D2P)](https://codecov.io/gh/saullocastro/panels)


Semi-analytical methods for plates, shells and stiffened panels
===============================================================

- Linear and non-linear static analyses, with field outputs of displacements, strains and
  stresses
- Linear buckling
- Vibration, with or without pre-stress
- Panel flutter using the piston theory
- Geometrically non-linear static analyses (postbuckling) using the
  Newton-Raphson method or the arc-length methods of Riks and Crisfield
- Multi-domain assemblies of plates, cylindrical shells, cylinders, stiffened panels, with the domains connected exactly with the null-space method, the default, or by penalty stiffnesses. Debonding defects or cohesive zone can be simulated
- CLPT, FSDT and TSDT kinematics for plates and cylindrical shells
- Donnell and Sanders-Koiter kinematics for cylindrical shells

The structural matrices are solved with
[structsolve](https://github.com/saullocastro/structsolve).


Citing this library
===================

Saullo G. P. Castro, Nathan D'Souza. (2026). Semi-analytical methods for plates, shells and stiffened panels (Version 0.9.0). Zenodo. DOI: https://doi.org/10.5281/zenodo.2541522.


Documentation
=============

The documentation is available on: https://saullocastro.github.io/panels.


Installation
============

To get the latest version:

    python -m pip install panels


History
=======

See [CHANGELOG.md](CHANGELOG.md) for the details of each version.
* version 0.9.0 (2026-09-24)
    - Null-space connection, exact, new default, as alternative to penalty-based connection
* version 0.8.0 (2026-09-23)
    - FSDT and TSDT for plate and cylindrical shells
    - Sanders-Koiter for cylindrical shells
    - Shell and Multidomain connections supporting FSDT, TSDT and Sanders-Koiter kinematics
    - Many improvements in the documentation
    - New tests based on literature
    - BUG fixes on aerodynamic matrix of cylindrical shells and mass matrix of stiffener 1D
* version 0.7.1 (2026-09-21)
    - Fixed the cohesive zone of the `MultiDomain` connection `'SB_TSL'`,
      which could not capture the onset of failure of the DCB: tangential
      separation with the rotation of each panel, tractions with the
      uncorrected separation and separation correction as in the thesis of
      D'Souza (2024)
    - Consistent tangent of the cohesive zone, `MultiDomain.calc_kT_TSL()`,
      and faster assembly of the `'SB_TSL'` secant stiffness
    - The kernels `fkCSB11_dmg`, `fkCSB12_dmg` and `fkCSB22_dmg` take the
      distances `dt` and `db` from each mid-surface to the interface
    - Validation of the cohesive zone against mode I DCB results in the
      literature, in `notebooks/`, with a convergence study
    - Fixed `StiffPanelBay.save()` with `composites>=0.9.0`
* version 0.6.21 (2026-09-18)
    - Fixed the mass of the base of `BladeStiff2D`, which was missing, and the
      section forces of `MultiDomain.force()`, which lacked the Jacobian
    - `StiffPanelBay.calc_kA()` sums the contribution of every panel, and
      `calc_kA`, `calc_cA`, `calc_fext`, `save`, `get_size` and
      `uvw_stiffener` of `StiffPanelBay` and `Shell.calc_cA` no longer raise
    - `Shell.strain()` and `Shell.stress()` work for a plate with `r=None`
    - Line coverage of the tests raised from 67 % to 92 %
    - Fixed `StiffPanelBay.add_panel` and the partial-domain integration of
      `Shell.calc_kC`, `calc_kG` and `calc_kM`
    - The limits `Shell.x1, x2, y1, y2` of the integration domain default to
      `None` instead of the sentinels `-1` and `+1`, which silently integrated
      the full domain for a limit at `1.0` and ignored a limit given on one
      side only; the old `-1` now raises
    - Fixed the `MultiDomain` skin-base connections: the `'SB'` penalty was a
      hard-coded mm-unit value, the top and bottom panels depended on their
      order in the assembly, and the connection dictionaries were modified
    - Penalty constants of each `MultiDomain` connection can be given with
      `'kt'` and `'kr'`
    - Reproduction of Stamatelos and Labeas with plate domains only in
      `tests/multidomain/test_stamatelos_labeas_2023_multidomain.py`, using
      every `MultiDomain` connection
    - Fixed the argument handling of the stiffness API: an invalid `c` or
      `c_cte` no longer reaches the nogil kernels, and a model that needs a
      radius raises instead of computing with `r = 0`
    - Reproduction of Stamatelos and Labeas (Computation 2023, 11, 110) in
      `notebooks/stamatelos_labeas_2023.ipynb`
    - Tangent stiffness matrix that is the exact derivative of the internal
      force vector, giving quadratic convergence to the Newton-Raphson method
    - `Shell` and `MultiDomain` can be passed directly to
      `structsolve.Analysis`, including the arc-length methods
    - Gauss-Legendre points and weights from `scipy.special.roots_legendre`
    - Modules of the models renamed without the `_bardell` suffix, and the
      legacy `_bardell` model names removed from `panels.modelDB.db`
    - Sphinx documentation with usage examples
* versions 0.5.x
    - Damaged multidomain connections
* up to version 0.3.x
    - Multidomain methods for pristine structures


License
=======

Distributed under the 3-Clause BSD license
(https://raw.github.com/saullocastro/panels/main/LICENSE).

Contact: S.G.P.Castro@tudelft.nl
