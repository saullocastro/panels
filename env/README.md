|  Version	| Download | Travis CI | Test Coverage |
| :-------:	| :--- 	   | :---      | :---          |
|   Master	|          | [![Linux Status](https://img.shields.io/travis/compmech/structsolve/master.svg)](https://travis-ci.org/compmech/structsolve) | [![Coverage Status](https://coveralls.io/repos/github/compmech/structsolve/badge.svg?branch=master)](https://coveralls.io/github/compmech/structsolve?branch=master) |


Structural Analysis Solvers
===========================

- Linear statics: [K]{u} = {f}
- Eigensolver for Linear buckling: ([K] + lambda[KG]){u} = 0
- Eigensolver for dynamics: ([K] + lambda^2[M]){u} = 0
- Nonlinear statics using Newton-Raphson 
- Nonlinear statics using the Arc-Length method


ROADMAP
=======

Currently these solvers are pretty much compatible with my other repository
"panels", such that future work should focus on generalizing this module to
work with other applications.


License
-------
Distrubuted in the 2-Clause BSD license (https://raw.github.com/compmech/structsolve/master/LICENSE).

Contact: castrosaullo@gmail.com

