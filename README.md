Github Actions status:

[![Actions Status](https://github.com/saullocastro/panels/workflows/pytest/badge.svg)](https://github.com/saullocastro/panels/actions)

Coverage status:

[![Codecov Status](https://codecov.io/gh/saullocastro/panels/branch/master/graph/badge.svg?token=KD9D8G8D2P)](https://codecov.io/gh/saullocastro/panels)


Semi-analytical methods for plates, shells and stiffened panels
===============================================================

- Linear static analyses, with field outputs of displacements, strains and
  stresses
- Linear buckling
- Vibration, with or without pre-stress
- Panel flutter using the piston theory
- Geometrically non-linear static analyses (postbuckling) using the
  Newton-Raphson method or the arc-length methods of Riks and Crisfield
- Multi-domain assemblies of cylinders, stiffened panels and panels with
  debonding defects

The structural matrices are solved with
[structsolve](https://github.com/saullocastro/structsolve).


Citing this library
===================

Saullo G. P. Castro, Nathan D'Souza. (2026). Semi-analytical methods for plates, shells and stiffened panels (Version 0.6.0). Zenodo. DOI: https://doi.org/10.5281/zenodo.2541522.


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

* version 0.6.0 (2026-09-17)
    - Tangent stiffness matrix that is the exact derivative of the internal
      force vector, giving quadratic convergence to the Newton-Raphson method
    - `Shell` and `MultiDomain` can be passed directly to
      `structsolve.Analysis`, including the arc-length methods
    - Gauss-Legendre points and weights from `scipy.special.roots_legendre`
    - Modules of the models renamed without the `_bardell` suffix
    - Sphinx documentation with usage examples
* versions 0.5.x
    - Damaged multidomain connections
* up to version 0.3.x
    - Multidomain methods for pristine structures


License
=======

Distributed under the 3-Clause BSD license
(https://raw.github.com/saullocastro/panels/master/LICENSE).

Contact: S.G.P.Castro@tudelft.nl
