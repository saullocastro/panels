r"""
Used to configure the main parameters for each implemented model.

.. currentmodule:: panels.modelDB

The dictionary ``db`` maps the name of each model, the attribute
``Shell.model``, to a dictionary with the keys:

- ``'field'``: the module of the field variables, ``clpt_field`` or
  ``fsdt_tsdt_field`` of :mod:`panels.models`;
- ``'matrices'``: the module of the analytically integrated matrices;
- ``'matrices_num'``: the module of the numerically integrated matrices
  and of the internal force vector;
- ``'dofs'``: the number of degrees of freedom per term of the
  approximation, ``3`` or ``5``;
- ``'e_num'``: the number of generalized strains, ``6``, ``8`` or ``13``;
- ``'requires_r'``: whether the kernels need a positive radius ``r``;
- ``'integrand_degree'``, optional: ``(k, j)`` such that the integrands of
  the numerical kernels are polynomials of degree at most ``k p + j`` along
  a direction, for approximation functions of degree ``p``, from which
  :func:`.gauss_points` gives the default number of integration points;
  ``(4, 0)`` when absent;
- ``'linear static'``, ``'linear buckling'``, ``'non-linear static'``: flags,
  ``True`` for every model.

Notes
-----
The ``dofs`` are ``u, v, w`` for the classical laminated plate theory
(CLPT), whose rotations derive from `w`, and ``u, v, w, phix, phiy`` for
the shear deformation theories, with independent rotations. The ``e_num``
generalized strains are the membrane strains and the changes of curvature
for the CLPT, ``6``, plus the transverse shear strains for the first-order
theory (FSDT), ``8``, plus the third-order terms of the in-plane strains
and the second-order terms of the transverse shear strains for the
third-order theory (TSDT), ``13``, see :mod:`panels.models`.

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
    'cylshell_clpt_gl': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clpt_field,
                    #NOTE fk0, fkM, fkAx, fkAy and fcA of
                    #     cylshell_clpt_sanders, fkG0 with the in-plane
                    #     blocks of the Green-Lagrange strains
                    'matrices': cylshell_clpt_gl,
                    'matrices_num': cylshell_clpt_gl_num,
                    'dofs': 3,
                    'e_num': 6,
                    # the kernels divide by r and r*r, a radius is mandatory
                    'requires_r': True,
                    },
    'cylshell_clpt_tl': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clpt_field,
                    #NOTE the analytical kernels of cylshell_clpt_gl: the
                    #     linear strains are those of cylshell_clpt_sanders
                    #     and the geometric stiffness of a constant membrane
                    #     stress state that of the Green-Lagrange strains
                    'matrices': cylshell_clpt_gl,
                    'matrices_num': cylshell_clpt_tl_num,
                    'dofs': 3,
                    'e_num': 6,
                    # the kernels divide by r and r*r, a radius is mandatory
                    'requires_r': True,
                    # curvatures cubic in the displacements: integrands of
                    # degree 6p - 2 along x and 6p along y, integrated
                    # exactly by 3p + 1 points
                    'integrand_degree': (6, 0),
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
    'plate_clpt_gl': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clpt_field,
                    #NOTE fk0, fkM, fkAx, fkAy and fcA of plate_clpt_donnell,
                    #     fkG0 with the in-plane blocks of the Green-Lagrange
                    #     strains
                    'matrices': plate_clpt_gl,
                    'matrices_num': plate_clpt_gl_num,
                    'dofs': 3,
                    'e_num': 6,
                    # the plate kernels never read r
                    'requires_r': False,
                    },
    'plate_clpt_tl': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': clpt_field,
                    #NOTE the analytical kernels of plate_clpt_gl: the linear
                    #     strains are those of plate_clpt_donnell and the
                    #     geometric stiffness of a constant membrane stress
                    #     state that of the Green-Lagrange strains
                    'matrices': plate_clpt_gl,
                    'matrices_num': plate_clpt_tl_num,
                    'dofs': 3,
                    'e_num': 6,
                    # the plate kernels never read r
                    'requires_r': False,
                    # curvatures cubic in the displacements: integrands of
                    # degree 6p - 2, integrated exactly by 3p points
                    'integrand_degree': (6, -2),
                    },
    'frame_web_clpt_donnell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    #NOTE the strains in polar coordinates of the web of a
                    #     ring frame need their own fields
                    'field': frame_web_clpt_donnell_num,
                    #NOTE the matrices of the whole domain are integrated
                    #     numerically, the strains having terms in 1/rho
                    'matrices': frame_web_clpt_donnell,
                    'matrices_num': frame_web_clpt_donnell_num,
                    'dofs': 3,
                    'e_num': 6,
                    # outer radius of the web, the kernels divide by r - x
                    'requires_r': True,
                    },
    'cylshell_fsdt_donnell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': fsdt_tsdt_field,
                    'matrices': cylshell_fsdt_donnell,
                    'matrices_num': cylshell_fsdt_donnell_num,
                    'dofs': 5,
                    'e_num': 8,
                    # the kernels divide by r, a radius is mandatory
                    'requires_r': True,
                    },
    'cylshell_fsdt_sanders': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': fsdt_tsdt_field,
                    'matrices': cylshell_fsdt_sanders,
                    'matrices_num': cylshell_fsdt_sanders_num,
                    'dofs': 5,
                    'e_num': 8,
                    # the kernels divide by r, a radius is mandatory
                    'requires_r': True,
                    },
    'cylshell_tsdt_donnell': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': fsdt_tsdt_field,
                    'matrices': cylshell_tsdt_donnell,
                    'matrices_num': cylshell_tsdt_donnell_num,
                    'dofs': 5,
                    'e_num': 13,
                    # the kernels divide by r, a radius is mandatory
                    'requires_r': True,
                    },
    'cylshell_tsdt_sanders': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': fsdt_tsdt_field,
                    'matrices': cylshell_tsdt_sanders,
                    'matrices_num': cylshell_tsdt_sanders_num,
                    'dofs': 5,
                    'e_num': 13,
                    # the kernels divide by r, a radius is mandatory
                    'requires_r': True,
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
    'plate_fsdt_gl': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': fsdt_tsdt_field,
                    #NOTE fk0, fkM, fkAx, fkAy and fcA of plate_fsdt_donnell, fkG0 with
                    #     the in-plane blocks of the Green-Lagrange strains
                    'matrices': plate_fsdt_gl,
                    'matrices_num': plate_fsdt_gl_num,
                    'dofs': 5,
                    'e_num': 8,
                    # the plate kernels never read r
                    'requires_r': False,
                    },
    'plate_tsdt_gl': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': fsdt_tsdt_field,
                    #NOTE fk0, fkM, fkAx, fkAy and fcA of plate_tsdt_donnell, fkG0 with
                    #     the in-plane blocks of the Green-Lagrange strains
                    'matrices': plate_tsdt_gl,
                    'matrices_num': plate_tsdt_gl_num,
                    'dofs': 5,
                    'e_num': 13,
                    # the plate kernels never read r
                    'requires_r': False,
                    },
    'cylshell_fsdt_gl': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': fsdt_tsdt_field,
                    #NOTE fk0, fkM, fkAx, fkAy and fcA of cylshell_fsdt_sanders, fkG0 with
                    #     the in-plane blocks of the Green-Lagrange strains
                    'matrices': cylshell_fsdt_gl,
                    'matrices_num': cylshell_fsdt_gl_num,
                    'dofs': 5,
                    'e_num': 8,
                    # the kernels divide by r, a radius is mandatory
                    'requires_r': True,
                    },
    'cylshell_tsdt_gl': {
                    'linear static': True,
                    'linear buckling': True,
                    'non-linear static': True,
                    'field': fsdt_tsdt_field,
                    #NOTE fk0, fkM, fkAx, fkAy and fcA of cylshell_tsdt_sanders, fkG0 with
                    #     the in-plane blocks of the Green-Lagrange strains
                    'matrices': cylshell_tsdt_gl,
                    'matrices_num': cylshell_tsdt_gl_num,
                    'dofs': 5,
                    'e_num': 13,
                    # the kernels divide by r, a radius is mandatory
                    'requires_r': True,
                    },
    }
