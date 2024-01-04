r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: panels.modelDB


"""
from .models import *

# Nested dict of type of shell (with its info) - Has possible model options
db = {
    'cylshell_clpt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clpt_bardell_field,
                    'matrices': cylshell_clpt_donnell_bardell,
                    'matrices_num': cylshell_clpt_donnell_bardell_num,
                    'dofs': 3,
                    'e_num': 6, # no of strain comp
                    },
    'plate_clpt_donnell_bardell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clpt_bardell_field,
                    'matrices': plate_clpt_donnell_bardell,
                    'matrices_num': plate_clpt_donnell_bardell_num,
                    'dofs': 3,
                    'e_num': 6,
                    },
    }
