.. automodule:: panels.models

Model database (:mod:`panels.modelDB`)
--------------------------------------

.. automodule:: panels.modelDB

Field variables (:mod:`panels.models.clpt_field`, :mod:`panels.models.fsdt_tsdt_field`)
---------------------------------------------------------------------------------------

.. automodule:: panels.models.clpt_field
    :members:

.. automodule:: panels.models.fsdt_tsdt_field
    :members:


.. _shell_models_kernels:

Kernels of the structural matrices
----------------------------------

Each model of :mod:`panels.modelDB` has two compiled modules, whose
integrands are derived symbolically from the kinematics of the model, see
:ref:`shell_kinematics`. The modules without suffix integrate in closed
form over the whole domain, with the integrals of Bardell's functions of
:mod:`panels.bardell`, see Eq. :eq:`bardell-separation`: the constitutive
stiffness ``fk0``, the geometric stiffness ``fkG0`` of constant stress
resultants, the mass ``fkM``, the aerodynamic matrices ``fkAx`` and
``fkAy`` of a flow along `x` or `y` and the aerodynamic damping ``fcA``. The
modules ``_num`` integrate with ``nx`` by ``ny`` Gauss-Legendre points and
also give the matrices that depend on the Ritz constants: ``fkC_num``,
``fkG_num``, ``fkM_num``, ``fkAx_num``, ``fkAy_num``, the internal force
vector ``calc_fint``, the product ``calc_kTv`` of the tangent stiffness by
a vector, see :ref:`tangent_product`, and the follower pressure
``calc_fext_follower`` with its load stiffness ``fkCfollower_num``, see
:ref:`follower_pressure`; ``plate_clpt_gl_num``, ``plate_clpt_tl_num``,
``cylshell_clpt_gl_num``, ``cylshell_clpt_tl_num`` and the ``_num`` modules of the
four models ``*_fsdt_gl`` and ``*_tsdt_gl`` import the mass and aerodynamic kernels of
the model with the same linear strains. The rows and columns are those of the Ritz
constants of Eq. :eq:`shell-ritz`.

========================== ============================================================ ===================================================================
model                      closed form                                                  numerical integration
========================== ============================================================ ===================================================================
``plate_clpt_donnell``     :mod:`panels.models.plate_clpt_donnell`                      :mod:`panels.models.plate_clpt_donnell_num`
``plate_clpt_gl``          :mod:`panels.models.plate_clpt_gl`, the linear kernels of    :mod:`panels.models.plate_clpt_gl_num`
                           ``plate_clpt_donnell``
``plate_clpt_tl``          :mod:`panels.models.plate_clpt_gl`                           :mod:`panels.models.plate_clpt_tl_num`
``frame_web_clpt_donnell`` :mod:`panels.models.frame_web_clpt_donnell`, numerical       :mod:`panels.models.frame_web_clpt_donnell_num`
                           integration of the whole domain
``plate_fsdt_donnell``     :mod:`panels.models.plate_fsdt_donnell`                      :mod:`panels.models.plate_fsdt_donnell_num`
``plate_tsdt_donnell``     :mod:`panels.models.plate_tsdt_donnell`                      :mod:`panels.models.plate_tsdt_donnell_num`
``cylshell_clpt_donnell``  :mod:`panels.models.cylshell_clpt_donnell`                   :mod:`panels.models.cylshell_clpt_donnell_num`
``cylshell_clpt_sanders``  :mod:`panels.models.cylshell_clpt_sanders`                   :mod:`panels.models.cylshell_clpt_sanders_num`
``cylshell_clpt_gl``       :mod:`panels.models.cylshell_clpt_gl`, the linear kernels    :mod:`panels.models.cylshell_clpt_gl_num`
                           of ``cylshell_clpt_sanders``
``cylshell_clpt_tl``       :mod:`panels.models.cylshell_clpt_gl`                        :mod:`panels.models.cylshell_clpt_tl_num`
``cylshell_fsdt_donnell``  :mod:`panels.models.cylshell_fsdt_donnell`                   :mod:`panels.models.cylshell_fsdt_donnell_num`
``cylshell_fsdt_sanders``  :mod:`panels.models.cylshell_fsdt_sanders`                   :mod:`panels.models.cylshell_fsdt_sanders_num`
``cylshell_tsdt_donnell``  :mod:`panels.models.cylshell_tsdt_donnell`                   :mod:`panels.models.cylshell_tsdt_donnell_num`
``cylshell_tsdt_sanders``  :mod:`panels.models.cylshell_tsdt_sanders`                   :mod:`panels.models.cylshell_tsdt_sanders_num`
``plate_fsdt_gl``          :mod:`panels.models.plate_fsdt_gl`, the linear kernels of    :mod:`panels.models.plate_fsdt_gl_num`
                           ``plate_fsdt_donnell``
``plate_tsdt_gl``          :mod:`panels.models.plate_tsdt_gl`, the linear kernels of    :mod:`panels.models.plate_tsdt_gl_num`
                           ``plate_tsdt_donnell``
``cylshell_fsdt_gl``       :mod:`panels.models.cylshell_fsdt_gl`, the linear kernels of :mod:`panels.models.cylshell_fsdt_gl_num`
                           ``cylshell_fsdt_sanders``
``cylshell_tsdt_gl``       :mod:`panels.models.cylshell_tsdt_gl`, the linear kernels of :mod:`panels.models.cylshell_tsdt_gl_num`
                           ``cylshell_tsdt_sanders``
``plate_fsdt_tl``          :mod:`panels.models.plate_fsdt_gl`                           :mod:`panels.models.plate_fsdt_tl_num`
``cylshell_fsdt_tl``       :mod:`panels.models.cylshell_fsdt_gl`                        :mod:`panels.models.cylshell_fsdt_tl_num`
========================== ============================================================ ===================================================================

The stiffeners of a :class:`.StiffPanelBay` have their own kernels, for
the classical laminated plate theory:
:mod:`panels.stiffener.models.bladestiff1d_clt_donnell`, the flange of a
:class:`.BladeStiff1D` as a beam along `x` following the skin at `y =
y_s`, and :mod:`panels.stiffener.models.bladestiff2d_clt_donnell` and
:mod:`panels.stiffener.models.bladestiff2d_clt_sanders`, the flange of a
:class:`.BladeStiff2D` as a plate connected by penalties to a skin with the
Donnell or the Sanders kinematics, see :ref:`stiffened_panels` and
:mod:`panels.stiffener.models`. On a curved bay the two differ in the
rotation of the normal of the skin about `x` that the flange follows,
`w_{,y} - v/r` with Sanders' kinematics and `w_{,y}` with Donnell's; the
base of the stiffener has the kinematics of its skin.
