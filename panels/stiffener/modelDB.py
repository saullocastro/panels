r"""
Used to configure the main parameters for each implemented model.

The dictionary ``db`` has one entry per stiffener model,
``'bladestiff1d_clt_donnell'`` and ``'bladestiff2d_clt_donnell'``, with the
flags ``'linear static'``, ``'linear buckling'`` and ``'non-linear static'``
and the module of the compiled kernels of :mod:`panels.stiffener.models`,
under the key ``'matrices'`` for ``'bladestiff1d_clt_donnell'``, with also
``'dofs'``, ``'e_num'`` and ``'num1'``, and under the key ``'connections'``
for ``'bladestiff2d_clt_donnell'``.

.. currentmodule:: panels.stiffener.modelDB


"""
from . models import *
from panels.models import (clpt_field,
                           plate_clpt_donnell,
                           cylshell_clpt_donnell,
                           )
db = {
    'bladestiff1d_clt_donnell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'matrices': bladestiff1d_clt_donnell,
                    'dofs': 3,
                    'e_num': 6,
                    'num1': 3,
                    },
    'bladestiff2d_clt_donnell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'connections': bladestiff2d_clt_donnell,
                    },
    }
