r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: panels.stiffener.modelDB


"""
from . models import *
from panels.models import (clpt_bardell_field,
                           plate_clpt_donnell_bardell,
                           cylshell_clpt_donnell_bardell,
                           )
db = {
    'bladestiff1d_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'matrices': bladestiff1d_clt_donnell_bardell,
                    'dofs': 3,
                    'e_num': 6,
                    'num1': 3,
                    },
    'bladestiff2d_clt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': False,
                    'connections': bladestiff2d_clt_donnell_bardell,
                    },
    }
