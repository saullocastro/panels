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
                    },
    }

# legacy names, kept for backward compatibility
for _old, _new in (('cylshell_clpt_donnell_bardell', 'cylshell_clpt_donnell'),
                   ('plate_clpt_donnell_bardell', 'plate_clpt_donnell')):
    db[_old] = db[_new]
