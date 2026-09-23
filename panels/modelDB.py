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
    'cylshell_clpt_sanders': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clpt_field,
                    'matrices': cylshell_clpt_sanders,
                    'matrices_num': cylshell_clpt_sanders_num,
                    'dofs': 3,
                    'e_num': 6,
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
    'plate_fsdt_donnell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': fsdt_tsdt_field,
                    'matrices': plate_fsdt_donnell,
                    'matrices_num': plate_fsdt_donnell_num,
                    'dofs': 5,
                    'e_num': 8,
                    'requires_r': False,
                    },
    'plate_tsdt_donnell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': fsdt_tsdt_field,
                    'matrices': plate_tsdt_donnell,
                    'matrices_num': plate_tsdt_donnell_num,
                    'dofs': 5,
                    'e_num': 13,
                    'requires_r': False,
                    },
    }
