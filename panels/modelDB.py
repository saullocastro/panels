r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: panels.modelDB


"""
from .models import *

# Nested dict of type of shell (with its info) - Has possible model options
db = {
    'cylshell_clpt_donnell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clpt_field,
                    'matrices': cylshell_clpt_donnell,
                    'matrices_num': cylshell_clpt_donnell_num,
                    'dofs': 3,
                    'e_num': 6, # no of strain comp
                    # the kernels divide by r and r*r, a radius is mandatory
                    'requires_r': True,
                    },
    'plate_clpt_donnell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clpt_field,
                    'matrices': plate_clpt_donnell,
                    'matrices_num': plate_clpt_donnell_num,
                    'dofs': 3,
                    'e_num': 6,
                    # the plate kernels never read r
                    'requires_r': False,
                    },
    }
