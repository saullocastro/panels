r"""Plates and cylindrical shells (:class:`.Shell`)

The class :class:`.Shell` holds the geometry, the laminate, the boundary
conditions, the loads and the prescribed displacements of a single domain,
and computes its structural matrices (``calc_kC``, ``calc_kG``, ``calc_kT``,
``calc_kM``, ``calc_kA``, ``calc_cA``), its force vectors (``calc_fext``,
``calc_fint``) and its field variables (``uvw``, ``strain``, ``stress``).
The model of each shell, e.g. ``'plate_clpt_donnell'``, selects the kernels
listed in :mod:`panels.modelDB`. The function :func:`load` reads a shell
saved by :meth:`.Shell.save`.

"""
import gc
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import eig
from scipy.special import roots_legendre
from numpy import linspace
from composites import laminated_plate
from composites.core import Laminate
from structsolve.sparseutils import finalize_symmetric_matrix

from .logger import msg, warn
from . import modelDB
from . shell_fext import shell_fext, pressure_patches
from . import json_io

DOUBLE = np.float64


def load(name, max_size=json_io.MAX_SIZE):
    r"""Load a :class:`.Shell` saved by :meth:`.Shell.save`

    A name without the extension is completed with ``'.shell.zip'`` when no
    file of that name exists. The file is read with :func:`panels.json_io.load`
    and must contain a :class:`.Shell`, otherwise a ``ValueError`` is raised.

    Parameters
    ----------
    name : str, path-like or file object
        Name of the file, with or without the extension ``'.shell.zip'``,
        or a binary file object opened for reading, e.g.
        :class:`io.BytesIO`.
    max_size : int, optional
        Maximum total uncompressed size of the members read, in bytes, 4 GiB
        by default, see :func:`panels.json_io.load`.

    Returns
    -------
    shell : :class:`.Shell`
        The loaded shell.

    """
    return json_io._load_type(name, 'Shell', ('.shell.zip', ), max_size)


def check_c(c, size):
    r"""Check that ``c`` is a vector of Ritz constants of size ``size``

    Nothing is returned, the function only raises when the check fails.

    Parameters
    ----------
    c : np.ndarray
        The vector to check.
    size : int
        The expected number of entries, e.g. :meth:`.Shell.get_size`.

    Raises
    ------
    TypeError
        If ``c`` is not a :class:`numpy.ndarray`.
    ValueError
        If ``c`` is not 1-D or does not have ``size`` entries.

    """
    # Conducts a check on c
    if not isinstance(c, np.ndarray):
        raise TypeError('"c" must be a NumPy ndarray object')
    if c.ndim != 1:
        raise ValueError('"c" must be a 1-D ndarray object')
    if c.shape[0] != size:
        raise ValueError('"c" must have the same size as the global stiffness matrix')


def gauss_points(num):
    r"""Default number of Gauss-Legendre points along a direction

    Returns ``max(2*num - 1, 7)``, the smallest number of points that
    integrates exactly every integrand of the numerical kernels of
    :class:`.Shell` along a direction with ``num`` approximation terms and a
    constant constitutive matrix, see :meth:`.Shell.integration_points`.

    Parameters
    ----------
    num : int
        Number of terms of the approximation functions along the direction,
        ``m`` along `x` or ``n`` along `y`.

    Returns
    -------
    points : int
        Number of Gauss-Legendre points.

    Notes
    -----
    Bardell's functions of the terms `i = 0, 1, 2, 3` are cubic and that of
    the term `i \ge 4` has degree `i`, so that the functions of ``num`` terms
    have degree at most `p = \max(num - 1, 3)`. The integrands of the
    internal force vector and of the tangent stiffness matrix with the
    geometrically non-linear strains are products of four of them, or of
    their derivatives, along each direction, of degree at most `4 p`, which
    the Gauss-Legendre rule of `n` points integrates exactly when `2 n - 1
    \ge 4 p`, i.e. `n \ge 2 p + 1 = \max(2\,num - 1, 7)`.

    """
    return max(2*num - 1, 7)


class Shell(object):
    r"""General shell class that can be used for plates or shells

    It works for both plates and cylindrical shells. When the attribute
    ``model`` is not set, it is selected according to parameter ``r``
    (radius): ``'plate_clpt_donnell'`` when ``r`` is ``None`` and
    ``'cylshell_clpt_sanders'`` otherwise. The Donnell kinematics of
    cylindrical shells remain available with ``model =
    'cylshell_clpt_donnell'``, see :mod:`panels.models`.

    The approximation functions for the displacement fields are built using
    :ref:`Bardell's functions <theory_func_bardell>`.

    The constructor sets the default attributes, e.g. simply supported
    boundary conditions, no loads and the numbers of integration points
    ``nx`` and ``ny`` following ``m`` and ``n``, then the attributes given
    in ``kwargs``. When both ``a`` and
    ``b`` are given the shell is also built: the model is selected, the
    laminate ``Shell.lam`` is created from ``stack``, ``plyt`` (or
    ``plyts``), ``laminaprop`` (or ``laminaprops``), ``rho`` and ``offset``
    with :func:`composites.laminated_plate`, and ``size`` is calculated;
    this raises a ``ValueError`` when ``stack``, ``laminaprop`` or ``plyt``
    is missing, the model is unknown or the integration limits are invalid.
    Otherwise the shell is built by the calculation methods, e.g.
    :meth:`.Shell.calc_kC`.

    The per-ply lists ``plyts``, ``laminaprops`` and ``rhos`` that are not
    given by the user are filled with ``plyt``, ``laminaprop`` and ``rho``
    at every rebuild, such that a later change of ``plyt``,
    ``laminaprop``, ``rho`` or ``stack`` is taken into account by the next
    rebuild, e.g. by :meth:`.Shell.calc_kC`. A list given by the user, as
    an argument, as an attribute, or by changing in place a list filled by
    the rebuild, is never overwritten and takes precedence over the
    corresponding scalar.

    The numbers of Gauss-Legendre points ``nx`` and ``ny`` of the
    numerically integrated matrices and vectors are, unless set by the user,
    ``gauss_points(m)`` and ``gauss_points(n)``, i.e. ``max(2*m - 1, 7)`` and
    ``max(2*n - 1, 7)`` for the values of ``m`` and ``n`` at the time of the
    calculation, which integrate exactly every integrand with a constant
    constitutive matrix, see :func:`.gauss_points` and
    :meth:`.Shell.integration_points`. A value assigned to ``nx`` or ``ny``,
    as an attribute or in ``kwargs``, is kept, the rebuild only raising it
    to ``m`` or ``n``; assigning ``None`` restores the default.

    The boundary conditions are controlled by the flags ``x1u, x1ur, x2u,
    x2ur, ..., y2w, y2wr``, where for instance ``x1v`` multiplies the value
    and ``x1vr`` the derivative of the approximation functions of `v` at the
    edge `x = 0`, ``x2v`` and ``x2vr`` those at `x = a`, and the flags
    starting with ``y1`` and ``y2`` those at `y = 0` and `y = b`, with ``0``
    removing and ``1`` keeping that degree of freedom. By default the values
    of `u`, `v` and `w` are removed and their derivatives kept on the four
    edges. The models based on shear deformation theories, e.g.
    ``'plate_fsdt_donnell'`` or ``'cylshell_tsdt_sanders'``, have the
    rotations `\phi_x` and `\phi_y` as independent fields, which are
    controlled by the analogous flags ``x1phix, x1phixr, ..., y2phiy,
    y2phiyr``, all ``1`` by default except ``x1phiy = x2phiy = 0`` and
    ``y1phix = y2phix = 0``. A clamped edge, e.g. `x = 0`, has ``x1u = x1v
    = x1w = 0`` and also ``x1wr = 0`` for the classical laminated plate
    theory (CLPT), ``x1phix = x1phiy = 0`` for the first-order shear
    deformation theory (FSDT), and both ``x1wr = 0`` and ``x1phix = x1phiy
    = 0`` for the third-order one (TSDT), see :ref:`boundary_conditions`.
    An edge clamped by its faces, where the TSDT section may warp, is
    prescribed with the keyword ``z`` of :meth:`.add_distr_pd_fixed_x` and
    :meth:`.add_distr_pd_fixed_y`.

    The attribute ``fsdt_shear_correction`` controls the transverse shear
    stiffness of the first-order shear deformation theory (FSDT):
    ``'rohwer'``, the default, ``'vlachoutsis'``, ``'constant'`` or ``None``
    select the corresponding method of
    :meth:`composites.Laminate.calc_transverse_shear_stiffness`, whereas a
    float ``k`` multiplies the uncorrected stiffness, ``A_ts = k*Abar_ts``.

    The attributes ``x0, y0, z0, point_x, point_xy`` place the shell in the
    global coordinate system of the 3D plots, see :meth:`.Shell.global_frame`
    and :meth:`.MultiDomain.plot3d`.

    The attributes ``x1, x2, y1, y2`` limit the integration domain to
    ``x1 <= x <= x2`` and ``y1 <= y <= y2``, in physical coordinates, while
    the approximation functions still span the whole ``0 <= x <= a`` and
    ``0 <= y <= b``. A limit that is ``None``, the default, is the
    corresponding edge of the shell, see :meth:`.Shell.integration_limits`.

    The constitutive matrix comes from the laminate, ``Shell.lam``, built
    from ``stack``, ``plyts``, ``laminaprops`` and ``offset``: it is the
    read-only property ``Shell.ABD``, see :meth:`.Shell._get_lam_ABD`, with
    the anisotropic coupling terms zeroed when the attribute
    ``force_orthotropic_laminate`` is ``True``. Every kernel, analytical or
    numerical, integrates this same matrix. A stiffness that varies over the
    domain, or that does not come from a stack of plies, e.g. a
    functionally graded wall, is given by the attribute ``ABDnxny``, one
    matrix per integration point, see :meth:`.Shell.integration_points`.
    When it is set, the stiffness matrices and the internal forces are
    integrated numerically with it, also within a :class:`.MultiDomain` and
    by the analyses of ``structsolve``. The laminate still gives the mass,
    :meth:`.Shell.calc_kM`, the thickness used by the kinematics of the
    TSDT, and the penalty constants of the connections, see
    :func:`.calc_kt_kr`.

    Parameters
    ----------
    a : float, optional
        Length (along the `x` coordinate). Default ``None``.
    b : float, optional
        Width (along the `y` coordinate), the arc length for a cylindrical
        shell. Default ``None``.
    r : float, optional
        Radius for cylindrical shell, a length. Default ``None``, which
        selects a plate model.
    stack : list or tuple, optional
        A sequence representing the angles for each ply, angles in degrees,
        passed to :func:`composites.laminated_plate`. Default ``None``.
    plyt : float, optional
        Ply thickness, a length, used for every ply unless the attribute
        ``plyts`` is given by the user. Default ``None``.
    laminaprop : list or tuple, optional
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13}, G_{23}`.
        The moduli are forces per unit area and `\nu_{12}` is dimensionless.
        Used for every ply unless the attribute ``laminaprops`` is given
        by the user. Default ``None``.
    rho : float, optional
        Material density, a mass per unit volume, used for every ply unless
        the attribute ``rhos`` is given by the user. Default ``0``.
    m, n : int, optional
        Number of terms for the approximation functions along `x` and `y`,
        respectively. At most ``30``. Default ``11``.
    offset : float, optional
        Laminate offset about shell mid-surface. The offset is measured along
        the normal (`z`) axis. A length. Default ``0.``.
    **kwargs : dict, optional
        Any other attribute of the class, e.g. ``model``, ``x1w`` or
        ``ABDnxny``, set with :func:`setattr` after the default attributes
        and before the shell is built.

    Raises
    ------
    ValueError
        If ``m > 30`` or ``n > 30``, the Bardell functions of the order 31
        and above not being coded.

    Notes
    -----
    The displacement field is approximated with the Ritz method [1]_, as a
    series of products of Bardell's hierarchical functions [2]_ along `x`
    and `y`, of the natural coordinates `\xi = 2x/a - 1` and `\eta = 2y/b -
    1`, `-1 \le \xi, \eta \le 1`:

    .. math::

        u(x, y) = \sum_{j=0}^{n-1} \sum_{i=0}^{m-1} c_{d (j m + i)}
                  P_i^u(\xi) P_j^u(\eta)

    and likewise for `v`, `w` and, for the shear deformation theories,
    `\phi_x`, `\phi_y`, with the offsets 1 to 4 in place of 0 in the index
    of the Ritz constant, where `d` is the number of degrees of freedom per
    term, 3 or 5, see :mod:`panels.modelDB`. `P_i^u` is the term `i` of
    Bardell's functions with the flags ``x1u, x1ur, x2u, x2ur`` and `P_j^u`
    the term `j` with the flags ``y1u, y1ur, y2u, y2ur``, see
    :mod:`panels.bardell`, and the derivatives follow from `\partial/\partial
    x = (2/a) \partial/\partial \xi` and `\partial/\partial y = (2/b)
    \partial/\partial \eta`. The structural matrices and force vectors are
    the derivatives with respect to the Ritz constants `\boldsymbol{c}` of the
    energy, or of the virtual work, integrated over the integration domain,
    and their size is that of `\boldsymbol{c}`, see :meth:`.Shell.get_size`.

    Only the first four terms of Bardell's functions have a value or a first
    derivative different from zero at the ends, `\xi = \pm 1`, see
    :mod:`panels.bardell`. A flag ``0`` therefore imposes exactly, as an
    essential boundary condition, a zero value (``x1w = 0``: `w = 0` at `x
    = 0`) or a zero first derivative normal to the edge (``x1wr = 0``:
    `w_{,x} = 0` at `x = 0`; ``x1ur = 0``: `u_{,x} = 0`) of the field on the
    whole edge; with ``1`` the value or the derivative is free and the
    natural boundary condition results from the energy. The default,
    `u = v = w = 0` with free derivatives, is the simply supported edge,
    and ``x1w = x1wr = 0`` a clamped one. For the shear deformation
    theories the rotations are independent fields, and their default flags
    give the hard simply supported condition, with the rotation tangential
    to each edge removed, consistent with the default boundary conditions
    of the models based on the classical laminated plate theory, where the
    rotation along the edge, a derivative of `w` along it, is zero with `w`.

    A clamped edge keeps the cross-section planar and unrotated, `u(z) =
    v(z) = 0` for every `z` across the thickness. On `x = 0`, with `u(z) = u
    - z w_{,x}` for the CLPT, this is `u = w_{,x} = 0`, and with `u(z) = u +
    z \phi_x` for the FSDT, `u = \phi_x = 0` with `w_{,x}` free. The TSDT
    field `u(z) = u + z \phi_x - c_1 [(z - d)^3 + d^3] (\phi_x + w_{,x})`,
    `c_1 = 4/(3 h^2)`, with the offset `d` and the thickness `h` of the
    laminate, is a cubic in `z` whose coefficients vanish only for `u =
    \phi_x = w_{,x} = 0`, hence ``x1wr = 0`` together with ``x1phix = 0``;
    the rotation at the reference surface, `\theta_x = u_{,z}(0)`, is then
    zero for any offset. On `x = 0`, `w = 0` gives `w_{,y} = 0` along the
    edge, and `v = \phi_y = 0` gives `v(z) = 0`. A face clamp, `u = v = 0`
    only at the faces `z = d \mp h/2`, gives the same conditions for the
    CLPT and the FSDT, whereas for the TSDT it gives the two conditions
    `\phi_x = w_{,x}/2` and `u = -(d/2)(1 - 4 d^2/h^2) w_{,x}`, leaving the
    section free to warp, `u(z) = (w_{,x}/2) \zeta (1 - 4 \zeta^2/h^2)` with
    `\zeta = z - d`; flags cannot express it, and it is imposed with the
    prescribed displacements at the faces, see
    :ref:`boundary_conditions_face_clamp`.

    The equilibrium approach of Rohwer (1988) [3]_, Sec. 3.4, Eqs. (28)-(43),
    gives ``k = 5/6`` for a homogeneous plate (Sec. 4.1) and accounts for
    the stacking sequence otherwise. The third-order shear deformation
    theory (TSDT) needs no shear correction.

    Limiting the integration domain to ``x1 <= x <= x2`` and ``y1 <= y <=
    y2`` while the approximation functions span the whole shell is how
    several domains, for instance the panels of a :class:`.StiffPanelBay`,
    share one set of approximation functions: the energy of each domain is
    integrated over its own region only.

    References
    ----------
    .. [1] Castro, S. G. P., and Donadon, M. V., "Assembly of Semi-Analytical
       models to Address Linear Buckling and Vibration of Stiffened
       Composite Panels with Debonding Defect", Composite Structures, 2017.
       https://doi.org/10.1016/j.compstruct.2016.10.026
    .. [2] Bardell, N. S., "Free vibration analysis of a flat plate using the
       hierarchical finite element method", Journal of Sound and Vibration,
       1991. https://doi.org/10.1016/0022-460X(91)90855-E
    .. [3] Rohwer, K., "Improved transverse shear stiffnesses for layered
       finite elements", DFVLR-FB 88-32, Deutsche Forschungs- und
       Versuchsanstalt für Luft- und Raumfahrt (DFVLR), Institut für
       Strukturmechanik, Braunschweig, 1988.

    """
    # Declare all the variables/attributes here to preallocate mem, speed it up. Var not declared here cant be used
    __slots__ = [ 'a', 'x1', 'x2', 'b', 'y1', 'y2', 'r',
        'stack', 'plyt', 'laminaprop', 'rho', 'offset',
        'group', 'x0', 'y0', 'z0', 'point_x', 'point_xy',
        'row_start', 'col_start', 'row_end', 'col_end',
        'name', 'bay', 'model',
        'fsdt_shear_correction',
        'm', 'n', '_nx', '_ny', 'size',
        'point_loads', 'point_loads_inc', 'distr_loads', 'distr_loads_inc',
        'pressure_loads', 'pressure_loads_inc',
        'point_pds', 'point_pds_inc', 'distr_pds', 'distr_pds_inc',
        'point_pds_values', 'point_pds_values_inc',
        'distr_pds_values', 'distr_pds_values_inc',
        'Nxx', 'Nyy', 'Nxy', 'Nxx_cte', 'Nyy_cte', 'Nxy_cte',
        'x1u', 'x1ur', 'x2u', 'x2ur',
        'x1v', 'x1vr', 'x2v', 'x2vr',
        'x1w', 'x1wr', 'x2w', 'x2wr',
        'y1u', 'y1ur', 'y2u', 'y2ur',
        'y1v', 'y1vr', 'y2v', 'y2vr',
        'y1w', 'y1wr', 'y2w', 'y2wr',
        'x1phix', 'x1phixr', 'x2phix', 'x2phixr',
        'x1phiy', 'x1phiyr', 'x2phiy', 'x2phiyr',
        'y1phix', 'y1phixr', 'y2phix', 'y2phixr',
        'y1phiy', 'y1phiyr', 'y2phiy', 'y2phiyr',
        'plyts', 'laminaprops', 'rhos', '_derived_lists', '_lam_mid',
        'flow', 'beta', 'gamma', 'aeromu', 'rho_air', 'speed_sound', 'Mach', 'air_speed',
        'ABDnxny', 'force_orthotropic_laminate',
        'num_eigvalues', 'num_eigvalues_print',
        'out_num_cores', 'increments', 'results',
        'lam', 'matrices', 'fields', 'plot_mesh',
        ]

    def __init__(self, a=None, b=None, r=None,
            stack=None, plyt=None, laminaprop=None, rho=0,
            m=11, n=11, offset=0., **kwargs):
        self.a = a
        self.x1 = None # limits of the integration domain along x, None will use 0
        self.x2 = None # and a, see Shell.integration_limits()
        self.b = b
        self.y1 = None # limits of the integration domain along y, None will use 0
        self.y2 = None # and b, see Shell.integration_limits()
        self.r = r # rad of curvature of panel (for curved panels)
        self.stack = stack
        self.plyt = plyt
        self.laminaprop = laminaprop
        self.rho = rho
        self.offset = offset

        # assembly
        self.group = None # Group name (useful when plotting multiple panels together)
        self.x0 = None # starting position of the panel in the global CS
        self.y0 = None
        self.z0 = None # used by the 3D plots, see Shell.global_frame()
        self.point_x = None # point on the local x axis, in the global CS
        self.point_xy = None # point on the local xy plane, in the global CS
        self.row_start = None
        self.col_start = None
        self.row_end = None
        self.col_end = None

        self.name = 'shell'
        self.bay = None

        # model
        self.model = None
        self.fsdt_shear_correction = 'rohwer' # in case of First-order Shear Deformation Theory

        # approximation series - no of terms in SFs
        self.m = m
        self.n = n
        if m > 30 or n > 30:
            raise ValueError('Bardell functions of the order 31 and above are not coded')
        self.size = None

        # numerical integration - no of points, None follows m and n, see
        # the properties nx and ny
        self._nx = None
        self._ny = None

        # loads
        self.point_loads = [] #NOTE see add_point_load
        self.point_loads_inc = [] #NOTE see add_point_load
        self.distr_loads = [] #NOTE see add_distr_load_fixed_x and add_distr_load_fixed_y
        self.distr_loads_inc = [] # NOTE see add_distr_load_fixed_x and add_distr_load_fixed_y
        self.pressure_loads = [] #NOTE see add_pressure_load
        self.pressure_loads_inc = [] #NOTE see add_pressure_load
        # prescribed displacements
        self.point_pds = [] #NOTE see add_point_pd
        self.point_pds_inc = [] #NOTE see add_point_pd
        self.distr_pds = [] #NOTE see add_distr_pd_fixed_x and add_distr_pd_fixed_y
        self.distr_pds_inc = [] # NOTE see add_distr_pd_fixed_x and add_distr_pd_fixed_y
            # Stored as [x pos, y pos of applied displ, force x, y, z due to that displ, z coordinate of u and v]
        # prescribed values, used by MultiDomain(..., pd_method='null-space'):
        # [x, y, up, vp, wp, phixp, phiyp, z] and [x, None, funcu, funcv,
        # funcw, funcphix, funcphiy, z] or [None, y, ...], None for a free
        # component, z the thickness coordinate of u and v
        self.point_pds_values = []
        self.point_pds_values_inc = []
        self.distr_pds_values = []
        self.distr_pds_values_inc = []
        # uniform membrane stress state
        self.Nxx = 0.
        self.Nyy = 0.
        self.Nxy = 0.
        # uniform constant membrane stress state (not multiplied by lambda)
        self.Nxx_cte = 0.
        self.Nyy_cte = 0.
        self.Nxy_cte = 0.

        #NOTE default boundary conditions:
            # Controls disp/rotation at boundaries i.e. flags
            # 0 = no disp or rotation
            # 1 = disp or rotation permitted

            # x1 and x2 are limits of x -- represent BCs with lines x = const
            # y1 and y2 ............. y -- .................. lines y = const
        # - displacement at 4 edges is zero
        # - free to rotate at 4 edges (simply supported by default)

        self.x1u = 0.
        self.x1ur = 1.
        self.x2u = 0.
        self.x2ur = 1.
        self.x1v = 0.
        self.x1vr = 1.
        self.x2v = 0.
        self.x2vr = 1.
        self.x1w = 0.
        self.x1wr = 1.
        self.x2w = 0.
        self.x2wr = 1.

        self.y1u = 0.
        self.y1ur = 1.
        self.y2u = 0.
        self.y2ur = 1.
        self.y1v = 0.
        self.y1vr = 1.
        self.y2v = 0.
        self.y2vr = 1.
        self.y1w = 0.
        self.y1wr = 1.
        self.y2w = 0.
        self.y2wr = 1.

        #NOTE rotations, only used by the shear deformation theories, the
        #     default is the hard simply supported condition, where the
        #     rotation tangential to each edge is zero
        self.x1phix = 1.
        self.x1phixr = 1.
        self.x2phix = 1.
        self.x2phixr = 1.
        self.x1phiy = 0.
        self.x1phiyr = 1.
        self.x2phiy = 0.
        self.x2phiyr = 1.

        self.y1phix = 0.
        self.y1phixr = 1.
        self.y2phix = 0.
        self.y2phixr = 1.
        self.y1phiy = 1.
        self.y1phiyr = 1.
        self.y2phiy = 1.
        self.y2phiyr = 1.

        # material
        self.plyts = None
        self.laminaprops = None
        self.rhos = None
        # the per-ply lists filled by _rebuild(), see _is_derived()
        self._derived_lists = {}

        # aeroelastic parameters for panel flutter
        self.flow = 'x'
        self.beta = None
        self.gamma = None
        self.aeromu = None
        self.rho_air = None
        self.speed_sound = None
        self.Mach = None
        self.air_speed = None

        # constitutive law, see the property ABD
        self.ABDnxny = None
        self.force_orthotropic_laminate = False

        # eigenvalue analysis
        self.num_eigvalues = 5
        self.num_eigvalues_print = 5

        # output queries
        #NOTE os.cpu_count() may return None, e.g. in WebAssembly (Pyodide),
        #     where the kernels run serially without OpenMP
        self.out_num_cores = os.cpu_count() or 1

        # outputs
        self.increments = None
        self.results = dict(eigvecs=None, eigvals=None)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._clear_matrices()

        # Build it if all the parameters are given
        if a is not None and b is not None:
            self._rebuild()


    def _clear_matrices(self):
        self.lam = None
        self.matrices = dict(kC=None, kG=None, kT=None, kM=None, kA=None, cA=None,
                             kCfollower=None)
        self.fields = dict(
                u=None, v=None, w=None, phix=None, phiy=None,
                exx=None, eyy=None, gxy=None, kxx=None, kyy=None, kxy=None, gyz=None, gxz=None,
                Nxx=None, Nyy=None, Nxy=None, Mxx=None, Myy=None, Mxy=None, Qy=None, Qx=None,
                )
        self.plot_mesh = dict(Xs=None, Ys=None)

        #NOTE memory cleanup
        gc.collect()


    @property
    def nx(self):
        r"""Number of Gauss-Legendre points along `x` of the numerical
        integration

        :func:`.gauss_points` of ``m`` at the time of the calculation unless
        a value was assigned, see :class:`.Shell`; assigning ``None``
        restores this default.

        """
        if self._nx is None:
            return gauss_points(self.m)
        return self._nx

    @nx.setter
    def nx(self, value):
        self._nx = value

    @property
    def ny(self):
        r"""Number of Gauss-Legendre points along `y` of the numerical
        integration

        :func:`.gauss_points` of ``n`` at the time of the calculation unless
        a value was assigned, see :class:`.Shell`; assigning ``None``
        restores this default.

        """
        if self._ny is None:
            return gauss_points(self.n)
        return self._ny

    @ny.setter
    def ny(self, value):
        self._ny = value

    def _rebuild(self):
        #NOTE only the values assigned by the user, the defaults being
        #     larger than m and n
        if self._nx is not None:
            self._nx = max(self._nx, self.m)
        if self._ny is not None:
            self._ny = max(self._ny, self.n)
        if self.model is None:
            if self.r is None:
                self.model = 'plate_clpt_donnell'
            elif self.r is not None:
                self.model = 'cylshell_clpt_sanders'

        valid_models = sorted(modelDB.db.keys())

        if not self.model in valid_models:
            raise ValueError('ERROR - valid models are:\n    ' +
                     '\n    '.join(valid_models))

        if not self.stack:
            raise ValueError('stack must be defined')

        #NOTE the lists not given by the user are derived again from the
        #     scalars, which may have changed since the last rebuild
        if self._is_derived('laminaprops'):
            if not self.laminaprop:
                raise ValueError('laminaprop must be defined')
            self._set_derived('laminaprops', [self.laminaprop for i in self.stack])

        if self._is_derived('rhos'):
            self._set_derived('rhos', [self.rho for i in self.stack])

        if self._is_derived('plyts'):
            if self.plyt is None:
                raise ValueError('plyt must be defined')
            self._set_derived('plyts', [self.plyt for i in self.stack])

        if self.stack is not None:
            k = self.fsdt_shear_correction
            if k is None or isinstance(k, str):
                shear_correction = k
            else:
                shear_correction = 'rohwer'
            lam = laminated_plate(stack=self.stack, plyts=self.plyts,
                                      laminaprops=self.laminaprops,
                                      rhos=self.rhos,
                                      offset=self.offset,
                                      shear_correction=shear_correction)
            self.lam = lam
        self.size = self.get_size()
        # fails early on invalid limits of the integration domain
        self.integration_limits()


    def _is_derived(self, name):
        r"""Whether the per-ply list ``name`` was not given by the user

        ``name`` is ``'plyts'``, ``'laminaprops'`` or ``'rhos'``. A list that
        is ``None`` or empty, or that is still the very list filled by
        :meth:`._set_derived`, unchanged, is derived from ``plyt``,
        ``laminaprop`` or ``rho`` at the next rebuild; any other list was
        given by the user and is kept.

        """
        value = getattr(self, name)
        if not value:
            return True
        derived = getattr(self, '_derived_lists', None) or {}
        if name not in derived:
            return False
        obj, copy = derived[name]
        return value is obj and list(value) == copy


    def _set_derived(self, name, value):
        r"""Set the per-ply list ``name`` filled by the rebuild, see
        :meth:`._is_derived`"""
        setattr(self, name, value)
        if getattr(self, '_derived_lists', None) is None:
            self._derived_lists = {}
        self._derived_lists[name] = (value, list(value))


    def _check_r(self):
        r"""Check that ``Shell.r`` is consistent with the selected model

        Models whose kernels divide by the radius (flagged with ``requires_r``
        in :mod:`.modelDB`) must be given a positive ``r``. Without this check
        an unset radius would reach those kernels as ``r = 0``, making every
        ``1/r`` term infinite and surfacing much later as a bare
        ``AssertionError`` raised while finalizing the sparse matrix.

        For models that ignore the radius (the plate kernels never read it) an
        unset ``r`` is normalized to ``0.``, which is the value the rest of the
        code compares against.

        ``r = 0.`` is never a valid radius for a cylindrical model. Use the
        ``'plate_clpt_donnell'`` model for flat panels.

        Notes
        -----
        The flat-plate limit of a cylindrical shell is ``r -> infinity``
        (``1/r -> 0``) and not ``r = 0``.

        """
        if modelDB.db[self.model].get('requires_r', False):
            if self.r is None or self.r <= 0:
                raise ValueError(
                    "model '{0}' requires Shell.r to be set to a positive "
                    "radius (got {1!r}); use 'plate_clpt_donnell' for a flat "
                    "panel, the flat limit of a cylinder is r -> infinity, "
                    "not r = 0".format(self.model, self.r))
        elif self.r is None:
            self.r = 0.


    def integration_limits(self):
        r"""Physical limits of the integration domain

        The attributes ``x1, x2, y1, y2`` are physical coordinates, with
        ``None`` standing for the corresponding edge of the shell: ``x1 = 0``,
        ``x2 = a``, ``y1 = 0`` and ``y2 = b``. Each limit is independent, so
        setting only ``x1 = 0.2`` integrates ``0.2 <= x <= a``.

        Returns
        -------
        limits : tuple
            The floats ``(x1, x2, y1, y2)``.

        Raises
        ------
        ValueError
            Unless ``0 <= x1 < x2 <= a`` and ``0 <= y1 < y2 <= b``. Limits
            outside the shell by a relative ``1e-12`` of ``a`` or ``b`` are
            taken as the edge, to absorb round-off. Also when ``a`` or ``b``
            is ``None``.

        """
        limits = []
        for name1, name2, length, dim in (('x1', 'x2', self.a, 'a'),
                                          ('y1', 'y2', self.b, 'b')):
            if length is None:
                raise ValueError('Shell.{0} must be defined'.format(dim))
            v1 = getattr(self, name1)
            v2 = getattr(self, name2)
            v1 = 0. if v1 is None else float(v1)
            v2 = float(length) if v2 is None else float(v2)
            tol = 1e-12*length
            if -tol <= v1 < 0.:
                v1 = 0.
            if length < v2 <= length + tol:
                v2 = float(length)
            if not (0. <= v1 < v2 <= length):
                raise ValueError(
                    'The integration limits must satisfy 0 <= {0} < {1} <= '
                    '{2}, got {0}={3!r}, {1}={4!r} and {2}={5!r}. Use None '
                    'for an edge of the shell'.format(
                        name1, name2, dim, getattr(self, name1),
                        getattr(self, name2), length))
            limits += [v1, v2]
        return tuple(limits)


    def is_partial_domain(self):
        r"""Tell whether this shell integrates only part of its domain

        See :meth:`.Shell.integration_limits`. Only the numerically integrated
        matrices honour a partial domain, therefore the analytical
        closed-form matrices must not be used when this returns ``True``.

        Returns
        -------
        partial : bool
            ``True`` when ``x1 > 0``, ``x2 < a``, ``y1 > 0`` or ``y2 < b``.

        Notes
        -----
        The closed-form matrices are sums of products of the integrals over
        `-1 \le \xi \le 1` and `-1 \le \eta \le 1` of products of Bardell's
        functions, see :func:`.calc_integral_ff`, i.e. integrals over the
        whole shell. Over a part of it the integrals of the functions are
        not tabulated, and the Gauss-Legendre quadrature of
        :meth:`.Shell.integration_points` is used instead.

        """
        x1, x2, y1, y2 = self.integration_limits()
        return bool(x1 > 0 or x2 < self.a or y1 > 0 or y2 < self.b)


    def integration_points(self, nx=None, ny=None):
        r"""Points and weights of the numerical integration

        The Legendre-Gauss rule of ``nx`` by ``ny`` points mapped to the
        integration limits, :meth:`.Shell.integration_limits`, which for a
        partial domain are not ``[0, a] x [0, b]``. The matrix of point
        ``(xs[i], ys[j])`` is ``ABDnxny[i, j]``, see :meth:`.Shell.calc_kC`,
        and the integral of a function over the integration domain is the
        sum over the points of the products of ``wxs[i]*wys[j]`` by the
        function at ``(xs[i], ys[j])``.

        Parameters
        ----------
        nx, ny : int or None, optional
            Number of points along `x` and `y`, by default the attributes
            ``nx`` and ``ny``, which are used by the kernels.

        Returns
        -------
        xs, ys, wxs, wys : np.ndarray
            The physical coordinates of the points along `x` and `y` and
            their weights, which sum to ``x2 - x1`` and ``y2 - y1``. ``xs``
            and ``wxs`` are 1-D arrays of size ``nx``, ``ys`` and ``wys`` of
            size ``ny``.

        Notes
        -----
        With the Gauss-Legendre points `\xi_i` and weights `w_i` over `-1 \le
        \xi \le 1`, `x_i = x_1 + (\xi_i + 1)(x_2 - x_1)/2` and `w_{x_i} = w_i
        (x_2 - x_1)/2`, and likewise along `y`, such that

        .. math::

            \int_{x_1}^{x_2} \int_{y_1}^{y_2} f(x, y) dy dx \approx
            \sum_i \sum_j w_{x_i} w_{y_j} f(x_i, y_j)

        which is exact for polynomials of degree up to `2 n_x - 1` in `x`
        and `2 n_y - 1` in `y`. The default ``nx = gauss_points(m)``, ``ny
        = gauss_points(n)`` of :class:`.Shell`, see :func:`.gauss_points`,
        is the smallest number of points for the exact integration. Bardell's
        functions, see Castro and Donadon (2017) [1]_, Eqs. (29) and (30),
        are of degree at most `p = \max(m - 1, 3)` in `x`. With a constant
        constitutive matrix the integrands of the constitutive stiffness
        and mass matrices are products of two of them or of their
        derivatives along each direction, of degree at most `2 p`, those of
        the geometric stiffness matrix of the linear stress state of ``c``
        of three, of degree at most `3 p`, and those of the tangent
        stiffness matrix and of the internal force vector with the
        geometrically non-linear strains of four, of degree at most `4 p`,
        which `n_x = 2 p + 1 = \max(2 m - 1, 7)` points integrate exactly,
        and likewise along `y`. The terms in `1/r` of the cylinders, also
        those in `v/r` of the Sanders-Koiter kinematics, only multiply these
        products by a constant, and the transverse shear strains and the
        higher-order terms of the TSDT are again products of the functions
        and of their derivatives. The pressure loads ``p(x, y)`` of
        :meth:`.add_pressure_load` multiply one function by `p`, the
        follower loads with the first-order area vector the product of two
        and with the complete area vector, ``follower='quadratic'``, of
        three, and are integrated exactly for a polynomial `p` of low
        enough degree, e.g. a constant one, and approximately otherwise.
        ``nx`` and ``ny`` follow ``m`` and ``n`` unless they are assigned,
        see :class:`.Shell`.

        References
        ----------
        .. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
           models to Address Linear Buckling and Vibration of Stiffened
           Composite Panels with Debonding Defect, Composite Structures, 160,
           232-247, 2017. https://doi.org/10.1016/j.compstruct.2016.10.026

        """
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
        x1, x2, y1, y2 = self.integration_limits()
        xis, wxis = roots_legendre(nx)
        etas, wetas = roots_legendre(ny)
        xs = x1 + (xis + 1)/2*(x2 - x1)
        ys = y1 + (etas + 1)/2*(y2 - y1)
        return xs, ys, wxis*(x2 - x1)/2, wetas*(y2 - y1)/2


    def _get_ABDnxny(self, ABDnxny, nx, ny):
        r"""Constitutive matrix and points of the numerical kernels

        ``ABDnxny`` if given, else the attribute ``ABDnxny`` if set, else the
        constant :attr:`.Shell.ABD`. For a matrix per point, the number of
        points ``nx`` and ``ny`` that are not given default to its shape,
        otherwise to the attributes ``nx`` and ``ny``.

        """
        if ABDnxny is None:
            ABDnxny = self.ABDnxny
        if ABDnxny is None:
            ABDnxny = self.ABD
        else:
            ABDnxny = np.asarray(ABDnxny, dtype=DOUBLE)
            self._check_ABDnxny(ABDnxny)
            if ABDnxny.ndim == 4:
                nx = ABDnxny.shape[0] if nx is None else nx
                ny = ABDnxny.shape[1] if ny is None else ny
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
        return ABDnxny, nx, ny


    def _check_ABDnxny(self, F, rtol=1e-10):
        r"""Check that the kernels integrate ``F`` as given

        The kernels read only the upper entries of each block of the
        constitutive matrix, e.g. `B_{12}` from ``F[0, 4]`` and never
        ``F[1, 3]``, and skip the blocks that couple the transverse shear to
        the other strains. ``F``, constant or one matrix per point, must
        therefore be symmetric, have symmetric blocks, ``B`` and, for the
        TSDT, ``E``, ``F`` and ``Dts``, and zero coupling blocks, as any
        laminate, to a tolerance ``rtol`` times the largest entry of each
        matrix.

        """
        ne = len(self._strain_names())
        if F.ndim not in (2, 4) or F.shape[-2:] != (ne, ne):
            raise ValueError('ABDnxny must have the shape ({0}, {0}) or (nx, '
                    'ny, {0}, {0}) for the model {1}, got {2}'.format(ne,
                    self.model, F.shape))
        Fp = F.reshape(-1, ne, ne)
        tol = rtol*np.abs(Fp).max(axis=(1, 2))
        checks = [('the matrix', F.swapaxes(-1, -2) - F)]
        blocks = {6: [('B', 0, 3)], 8: [('B', 0, 3)],
                  13: [('B', 0, 3), ('E', 0, 6), ('F', 3, 6), ('Dts', 9, 11)]}
        for name, i, j in blocks[ne]:
            k = 2 if name == 'Dts' else 3
            X = F[..., i:i+k, j:j+k]
            checks.append(('the block ' + name, X.swapaxes(-1, -2) - X))
        if ne == 8:
            checks.append(('the coupling to the transverse shear',
                           F[..., :6, 6:]))
        elif ne == 13:
            checks.append(('the coupling to the transverse shear',
                           F[..., :9, 9:]))
        for what, R in checks:
            err = np.abs(R.reshape(len(Fp), -1)).max(axis=1)
            bad = np.flatnonzero(err > tol)
            if bad.size:
                where = ('' if F.ndim == 2 else ' at the point [{0}, {1}]'
                         .format(*np.unravel_index(bad[0], F.shape[:2])))
                adj = 'zero' if what.startswith('the coupling') else 'symmetric'
                raise ValueError('ABDnxny must have {0} {1}, as any laminate, '
                    'since the kernels read only the upper entries of each '
                    'block{2}: error {3:.3e}, tolerance {4:.3e}'.format(adj,
                    what, where, err[bad[0]], tol[bad[0]]))


    def global_frame(self):
        r"""Origin and axes of the shell in the global coordinate system

        The origin is ``(x0, y0, z0)``, a ``None`` standing for ``0``, and the
        local axes are defined by the point ``point_x``, on the `x` axis, and
        by the point ``point_xy``, on the `xy` plane, both given with three
        global coordinates::

            vec_x = point_x - (x0, y0, z0)
            vec_xy = point_xy - (x0, y0, z0)
            vec_z = np.cross(vec_x, vec_xy)
            vec_y = np.cross(vec_z, vec_x)

        By default ``point_x = (x0 + 1, y0, z0)`` and ``point_xy = (x0, y0 +
        1, z0)``, such that the local axes are those of the global coordinate
        system, as in the 2D plots of :meth:`.MultiDomain.plot`. For a
        cylindrical shell these are the axes at the point `x = y = 0` of the
        shell, see :meth:`.Shell.global_coords`.

        Returns
        -------
        origin : np.ndarray
            The global coordinates of the origin, with shape ``(3,)``.
        axes : np.ndarray
            The unit vectors of the local axes `x`, `y` and `z` in the rows,
            with shape ``(3, 3)``.

        Raises
        ------
        ValueError
            If ``point_x`` or ``point_xy`` does not have three coordinates,
            if ``point_x`` coincides with the origin, or if ``point_xy`` lies
            on the local `x` axis, to a relative tolerance of ``1e-12``.

        """
        origin = np.array([0. if v is None else v for v in
                           (self.x0, self.y0, getattr(self, 'z0', None))],
                          dtype=DOUBLE)
        point_x = getattr(self, 'point_x', None)
        point_xy = getattr(self, 'point_xy', None)
        #NOTE checked before subtracting the origin, which would otherwise
        #     raise a less clear broadcasting error
        point_x = origin + [1., 0., 0.] if point_x is None else point_x
        point_xy = origin + [0., 1., 0.] if point_xy is None else point_xy
        point_x = np.asarray(point_x, dtype=DOUBLE)
        point_xy = np.asarray(point_xy, dtype=DOUBLE)
        if point_x.shape != (3,) or point_xy.shape != (3,):
            raise ValueError('point_x and point_xy must have three coordinates')
        vec_x = point_x - origin
        vec_xy = point_xy - origin
        vec_z = np.cross(vec_x, vec_xy)
        vec_y = np.cross(vec_z, vec_x)
        norms = np.array([np.linalg.norm(v) for v in (vec_x, vec_y, vec_z)])
        # relative to the lengths, since vec_xy may be parallel to vec_x
        if (norms[0] == 0 or norms[2] <= 1e-12*norms[0]
                *np.linalg.norm(vec_xy)):
            raise ValueError('point_x must differ from the origin (x0, y0, '
                             'z0) and point_xy must not lie on the x axis, '
                             'got point_x={0!r}, point_xy={1!r}'.format(
                                 point_x, point_xy))
        axes = np.array([vec_x, vec_y, vec_z])/norms[:, None]
        return origin, axes


    def is_curved(self):
        r"""Tell whether the mid-surface is a cylindrical surface

        ``True`` for the models of cylindrical shells, flagged with
        ``requires_r`` in :mod:`.modelDB`, with a positive radius ``r``.
        When ``model`` is not yet set, a positive ``r`` is enough, since the
        default model with a radius is ``'cylshell_clpt_sanders'``.

        Returns
        -------
        curved : bool
            ``False`` when ``r`` is ``None`` or not positive.

        """
        if self.r is None or self.r <= 0:
            return False
        if self.model is None:
            return True
        return bool(modelDB.db[self.model].get('requires_r', False))


    def global_coords(self, x, y):
        r"""Global coordinates and local axes at points of the mid-surface

        The local axes are those of :meth:`.Shell.global_frame`. For a
        cylindrical shell, see :meth:`.Shell.is_curved`, `y` is the arc length
        along the circumference, of radius ``r``, whose centre is at `z = -r`
        on the local axes of the origin, such that `w` is positive outwards.

        Parameters
        ----------
        x, y : array-like
            Coordinates of the points in the local coordinate system of the
            shell, `0 \le x \le a`, `0 \le y \le b`, with the same shape or
            broadcastable.

        Returns
        -------
        X, ex, ey, ez : np.ndarray
            With the shape of the points and an additional last axis of size 3,
            the global coordinates of the points and the unit vectors along
            which the displacements `u`, `v` and `w` act at each point. The
            global position of the displaced point is ``X + u[..., None]*ex +
            v[..., None]*ey + w[..., None]*ez``.

        Notes
        -----
        For a cylindrical shell, with `\theta = y/r` and the local axes
        `\hat{x}`, `\hat{y}`, `\hat{z}` of the origin, the mid-surface is `x
        \hat{x} + r \sin \theta \hat{y} + r (\cos \theta - 1) \hat{z}`, the
        tangent along `y` is `\cos \theta \hat{y} - \sin \theta \hat{z}` and
        the normal is `\sin \theta \hat{y} + \cos \theta \hat{z}`. The
        outward normal is consistent with the hoop strain `\varepsilon_{yy} =
        v_{,y} + w/r` of the kinematics, see :mod:`panels.models`.

        """
        origin, (ax, ay, az) = self.global_frame()
        x, y = np.broadcast_arrays(np.asarray(x, dtype=DOUBLE),
                                   np.asarray(y, dtype=DOUBLE))
        ones = np.ones(x.shape + (1,))
        ex = ones*ax
        if self.is_curved():
            theta = (y/self.r)[..., None]
            sin, cos = np.sin(theta), np.cos(theta)
            X = (origin + x[..., None]*ax + self.r*sin*ay
                 + self.r*(cos - 1)*az)
            ey = cos*ay - sin*az
            ez = sin*ay + cos*az
        else:
            X = origin + x[..., None]*ax + y[..., None]*ay
            ey = ones*ay
            ez = ones*az
        return X, ex, ey, ez


    def get_size(self):
        r"""Calculate the size of the stiffness matrices

        The size of the stiffness matrices can be interpreted as the number of
        rows or columns, recalling that this will be the size of the Ritz
        constants' vector `\boldsymbol{c}`, the internal force vector `\boldsymbol{F}_{int}` and
        the external force vector `\boldsymbol{F}_{ext}`.

        ONLY RETURNS THE NUMBER OF ROWS ''OR'' COLS

        The size is ``dofs*m*n``, where ``dofs`` is the number of degrees of
        freedom per term of the model, ``3`` or ``5``, see
        :mod:`panels.modelDB`. The attribute ``size`` is updated.

        Returns
        -------
        size : int
            The size of the stiffness matrices of this shell, ``dofs*m*n``.

        """
        dofs = modelDB.db[self.model]['dofs']
        self.size = dofs*self.m*self.n
        return self.size


    def _default_field(self, xs, ys, gridx, gridy):
        if xs is None or ys is None:
            xs = linspace(0, self.a, gridx)
            ys = linspace(0, self.b, gridy)
            xs, ys = np.meshgrid(xs, ys, copy=True)
        xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
        ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
        xshape = xs.shape
        yshape = ys.shape
        if xshape != yshape:
            raise ValueError('Arrays xs and ys must have the same shape')
        self.plot_mesh['Xs'] = xs
        self.plot_mesh['Ys'] = ys
        xs = np.ascontiguousarray(xs.ravel(), dtype=DOUBLE)
        ys = np.ascontiguousarray(ys.ravel(), dtype=DOUBLE)

        return xs, ys, xshape, yshape


    @property
    def ABD(self):
        r"""Constitutive matrix of the laminate ``Shell.lam``, read-only

        Calculated by :meth:`.Shell._get_lam_ABD` at each access, therefore
        always consistent with ``Shell.lam`` and with the attribute
        ``force_orthotropic_laminate``; ``None`` before the laminate is
        built. To use another stiffness, constant or varying over the domain,
        set the attribute ``ABDnxny``, see :meth:`.Shell.calc_kC`.

        """
        if self.lam is None:
            return None
        return self._get_lam_ABD(silent=True)


    @ABD.setter
    def ABD(self, value):
        r"""Setter of :attr:`.Shell.ABD`, which always raises

        Nothing is set, the property is read-only.

        Parameters
        ----------
        value : object
            Ignored.

        Raises
        ------
        AttributeError
            Always, ``Shell.ABD`` is calculated from ``Shell.lam``; the
            attribute ``ABDnxny`` gives another stiffness.

        """
        raise AttributeError('Shell.ABD is calculated from the laminate '
            'Shell.lam and cannot be set, use the attribute Shell.ABDnxny '
            'for a stiffness that is not that of the laminate')


    def _get_constant_ABD(self):
        r"""Constant constitutive matrix of the post-processing

        The attribute ``ABDnxny`` when it is a constant matrix, else
        :attr:`.Shell.ABD`. A stiffness per integration point raises a
        ``ValueError``, since the stresses are evaluated at any point.

        """
        if self.ABDnxny is None:
            ABD = self.ABD
        else:
            ABD = np.asarray(self.ABDnxny, dtype=DOUBLE)
            if ABD.ndim != 2:
                raise ValueError('The stresses need a constant stiffness, '
                    'Shell.ABDnxny has one per integration point')
        if ABD is None:
            raise ValueError('Laminate ABD matrix not defined for shell')
        return ABD


    def _get_lam_mid(self):
        r"""Laminate ``Shell.lam`` about its mid-surface

        A :class:`composites.Laminate` with the plies, stacking sequence and
        shear correction of ``Shell.lam`` and a zero offset, whose stiffness
        matrices and moments of the density, e.g. ``intrhoz6``, are the
        integrals through the thickness with `\zeta = z - d` measured from
        the mid-surface, used by the TSDT, see :meth:`._get_lam_ABD` and
        :meth:`.calc_kM`. It is ``Shell.lam`` itself when the offset is
        zero, and it is computed again only when ``Shell.lam`` changes.

        Returns
        -------
        lam_mid : :class:`composites.Laminate`
            The laminate about its mid-surface.

        Raises
        ------
        RuntimeError
            When ``Shell.lam`` is ``None``.

        """
        lam = self.lam
        if lam is None:
            raise RuntimeError('lam object is None!')
        if lam.offset == 0:
            return lam
        #NOTE called from the kernels fkM, no matrix-matrix BLAS product
        #     herein, see the note in Shell._get_lam_ABD
        cache = getattr(self, '_lam_mid', None)
        if cache is not None and cache[0] is lam and cache[2] == lam.offset:
            return cache[1]
        mid = Laminate()
        mid.offset = 0.
        mid.stack = list(lam.stack)
        mid.shear_correction = lam.shear_correction
        mid.plies = lam.plies
        mid.h = lam.h
        mid.calc_constitutive_matrix()
        mid.calc_equivalent_properties()
        self._lam_mid = (lam, mid, lam.offset)
        return mid


    def _get_lam_ABD(self, silent=False):
        r"""Constitutive matrix of the laminate, as required by the model

        - Classical laminated plate theory (CLPT), ``6 x 6``: the ``ABD``
          matrix.

        - First-order shear deformation theory (FSDT), ``8 x 8``: ``[[A, B,
          0], [B, D, 0], [0, 0, Ats]]``, where ``Ats`` is the shear corrected
          transverse shear stiffness of the `(yz, xz)` components, see the
          attribute ``fsdt_shear_correction``.

        - Third-order shear deformation theory (TSDT), ``13 x 13``: ``T^T
          F_m T``, with ``F_m = [[A, B, E, 0, 0], [B, D, F, 0, 0], [E, F, H,
          0, 0], [0, 0, 0, Abar_ts, Dts], [0, 0, 0, Dts, Fts]]`` of the
          laminate about its mid-surface, :meth:`._get_lam_mid`, without
          shear correction, see the notes.

        The rows and columns are in the order of the generalized strains of
        each model, see :meth:`.Shell.strain`.

        Notes
        -----
        The blocks are the integrals through the thickness of the laminate,
        computed by :mod:`composites`, of the plane-stress stiffness
        `\bar{Q}` of the plies, `A, B, D = \int \bar{Q} \{1, z, z^2\} dz`
        and, for the TSDT, `E, F, H = \int \bar{Q} \{z^3, z^4, z^6\} dz`, and
        of their transverse shear stiffness `\bar{Q}_s`, ``Abar_ts``,
        ``Dts``, ``Fts`` `= \int \bar{Q}_s \{1, z^2, z^4\} dz`. With the
        strains of the displacement field of each theory, see
        :mod:`panels.models`, the product of this matrix by the generalized
        strains gives the conjugate stress resultants, see
        :func:`.fsdt_tsdt_field.stress_names`, and half the product of the
        strains by it the strain energy per unit area. ``Ats`` is the
        transverse shear stiffness corrected for the constant transverse
        shear strain of the FSDT, see :class:`.Shell`.

        For the TSDT the cubic term of the displacement field is written
        about the mid-surface of the laminate, `z = d` with the offset `d`,
        such that the transverse shear strains vanish at both faces, see
        :mod:`panels.models`: the in-plane strains are `\varepsilon =
        \varepsilon^{(0)} + z \varepsilon^{(1)} + \left[(z - d)^3 +
        d^3\right] \varepsilon^{(3)}` and the transverse shear strains
        `\gamma = \gamma^{(0)} + (z - d)^2 \gamma^{(2)}`. With `\zeta = z -
        d`, `\varepsilon = \left(\varepsilon^{(0)} + d \varepsilon^{(1)} +
        d^3 \varepsilon^{(3)}\right) + \zeta \varepsilon^{(1)} + \zeta^3
        \varepsilon^{(3)}`, such that the strain energy is that of the
        laminate about its mid-surface, ``F_m``, whose integrals of `\bar
        Q \{1, \zeta, \zeta^2, \zeta^3, \zeta^4, \zeta^6\}` and `\bar Q_s
        \{1, \zeta^2, \zeta^4\}` composites computes with a zero offset,
        for the strains ``T e``, ``T`` being the identity but for the blocks
        ``T[0:3, 3:6] = d I`` and ``T[0:3, 6:9] = d^3 I``. With `d = 0` it
        is the matrix of the laminate ``Shell.lam``.

        """
        if self.lam is None:
            raise RuntimeError('lam object is None!')
        lam = self.lam
        if 'clpt' in self.model:
            #NOTE a copy, the coupling terms are zeroed below
            ABD = np.array(lam.ABD, dtype=DOUBLE)
        elif 'fsdt' in self.model:
            ABD = np.zeros((8, 8), dtype=DOUBLE)
            ABD[:6, :6] = lam.ABD
            k = self.fsdt_shear_correction
            if k is None or isinstance(k, str):
                ABD[6:, 6:] = lam.Ats
            else:
                ABD[6:, 6:] = k*lam.Abar_ts
        elif 'tsdt' in self.model:
            mid = self._get_lam_mid()
            A, B, D = mid.A, mid.B, mid.D
            E, F, H = mid.E, mid.F, mid.H
            d = lam.offset
            #NOTE T^T F_m T written by blocks, see the notes; the products by
            #     the scalar d do not call BLAS. The kernels of
            #     panels.models read Shell.ABD, hence run this code, while
            #     their C variables may be held in registers. The
            #     matrix-matrix product of MKL 2024.2.0 to 2025.0.0 on
            #     Windows x64 (cblas_dgemm, reached by numpy.matmul; fixed in
            #     2025.0.1) returns with xmm8 to
            #     xmm15 changed, which the Windows x64 calling convention
            #     requires to be preserved: one product T.T @ F_m @ T, or
            #     even np.eye(13) @ ABD, here gives nan in
            #     cylshell_tsdt_sanders.fk0. Do not call matrix-matrix BLAS
            #     products in code reached from the kernels, see
            #     tests/tests_shell/test_kernel_python_calls.py
            ABD = np.zeros((13, 13), dtype=DOUBLE)
            B1 = d*A + B
            E1 = d**3*A + E
            F1 = d**4*A + d**3*B + d*E + F
            ABD[0:3, 0:3] = A
            ABD[0:3, 3:6] = B1
            ABD[0:3, 6:9] = E1
            ABD[3:6, 0:3] = B1.T
            ABD[3:6, 3:6] = d**2*A + 2*d*B + D
            ABD[3:6, 6:9] = F1
            ABD[6:9, 0:3] = E1.T
            ABD[6:9, 3:6] = F1.T
            ABD[6:9, 6:9] = d**6*A + 2*d**3*E + H
            ABD[9:11, 9:11] = mid.Abar_ts
            ABD[9:11, 11:13] = mid.Dts
            ABD[11:13, 9:11] = mid.Dts
            ABD[11:13, 11:13] = mid.Fts

        if self.force_orthotropic_laminate and 'tsdt' in self.model:
            msg('', silent=silent)
            msg('Forcing orthotropic laminate...', level=2, silent=silent)
            # the 16 and 26 terms of A, B, D, E, F, H and the 45 shear terms
            for i in (0, 1, 3, 4, 6, 7):
                for j in (2, 5, 8):
                    ABD[i, j] = ABD[j, i] = 0.
            for i, j in ((9, 10), (11, 12), (9, 12), (10, 11)):
                ABD[i, j] = ABD[j, i] = 0.
        elif self.force_orthotropic_laminate:
            msg('', silent=silent)
            msg('Forcing orthotropic laminate...', level=2, silent=silent)
            ABD[0, 2] = 0. # A16
            ABD[1, 2] = 0. # A26
            ABD[2, 0] = 0. # A61
            ABD[2, 1] = 0. # A62

            ABD[0, 5] = 0. # B16
            ABD[5, 0] = 0. # B61
            ABD[1, 5] = 0. # B26
            ABD[5, 1] = 0. # B62

            ABD[3, 2] = 0. # B16
            ABD[2, 3] = 0. # B61
            ABD[4, 2] = 0. # B26
            ABD[2, 4] = 0. # B62

            ABD[3, 5] = 0. # D16
            ABD[4, 5] = 0. # D26
            ABD[5, 3] = 0. # D61
            ABD[5, 4] = 0. # D62

            if ABD.shape[0] == 8:
                ABD[6, 7] = 0. # A45
                ABD[7, 6] = 0. # A54

        return ABD


    def calc_kC(self, size=None, row0=0, col0=0, silent=True, finalize=True,
            c=None, c_cte=None, nx=None, ny=None, ABDnxny=None, NLgeom=False,
            inc=1.):
        r"""Calculate the constitutive stiffness matrix

        It is ``NLgeom``, and not ``c``, that selects the large displacement
        matrix. With ``NLgeom=False`` the result is the linear constitutive
        stiffness matrix, whether or not ``c`` is given. With ``NLgeom=True``
        the result is the part of the tangent stiffness matrix of
        :meth:`.Shell.calc_kT` that :meth:`.Shell.calc_kG` does not return,
        evaluated at ``c``. What ``c`` alone changes is the integration:
        giving it selects the numerically integrated matrices over the
        analytical closed form, which is also replaced by the numerical
        integration when ``ABDnxny`` is given or set as attribute, when
        ``NLgeom=True`` or when the integration domain is partial, see
        :meth:`.Shell.is_partial_domain`. When using ``c`` its size must be
        the same as the attribute ``size``.

        Despite the name, the returned matrix also contains:

        - with ``NLgeom=True``, the geometric stiffness of the membrane
          stresses of the non-linear part of the strains, which
          :meth:`.Shell.calc_kG` does not include;
        - with ``NLgeom=True`` and ``finalize=True``, the load stiffness of
          the follower pressure loads, :meth:`.Shell.calc_kCfollower` at
          ``c`` and load factor ``inc``, added after the symmetrization. With
          ``finalize=False``, as in an assembly, it is not added; it must
          then be added after the finalization, see
          :meth:`.MultiDomain.calc_kC`;
        - if ``c_cte`` is given, or any of the attributes ``Nxx_cte``,
          ``Nyy_cte`` or ``Nxy_cte`` is non-zero, the geometric stiffness of
          that constant stress state, i.e. :meth:`.Shell.calc_kG` evaluated
          with ``c_cte`` and ``Nxx_cte``, ``Nyy_cte``, ``Nxy_cte`` in place
          of ``c`` and ``Nxx``, ``Nyy``, ``Nxy``, in closed form unless
          ``c_cte`` is given, ``NLgeom=True`` or the integration domain is
          partial.

        In multi-domain semi-analytical models the sparse matrices that are
        calculated may have the ``size`` of the assembled global model, and the
        current constitutive matrix being calculated starts at position
        ``row0`` and ``col0``.

        Parameters
        ----------
        size : int or str, optional
            The size of the calculated sparse matrices. When using a string,
            for example, if '+1' is given it will add 1 to the shell's size
            obtained by the :meth:`.Shell.get_size`. Default ``None``, the
            size of this shell, :meth:`.Shell.get_size`.
        row0, col0 : int, optional
            Offset to populate the output sparse matrix (necessary when
            assemblying shells). Default ``0``.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
            Default ``True``, nothing is printed.
        finalize : bool, optional
            Asserts validity of output data and makes the output matrix
            symmetric, should be ``False`` when assemblying. Default
            ``True``.
        c : np.ndarray or None, optional
            This must be the result of a static analysis, used to compute
            non-linear terms based on the actual displacement field. A 1-D
            :class:`numpy.ndarray` of size ``size``. Default ``None``, a
            vector of zeros when needed.
        c_cte : np.ndarray or None, optional
            This must be the result of a static analysis, used to compute
            initial stress state not affected by the load multiplier of the
            linear buckling eigenvalue analysis. A 1-D
            :class:`numpy.ndarray` of size ``size``. Default ``None``.
        nx, ny : int or None, optional
            Number of integration points along `x` and `y`, respectively, for
            the Legendre-Gauss quadrature rule applied in the numerical
            integration. Only used by the numerical integration. By default
            the shape of ``ABDnxny`` when it has one matrix per point, else
            the attributes ``nx`` and ``ny``.
        ABDnxny : array-like or None, optional
            The constitutive matrix, by default the attribute ``ABDnxny``
            or, when it is ``None``, the matrix of the laminate
            :attr:`.Shell.ABD`. Any other than :attr:`.Shell.ABD` selects the
            numerical integration. Either a constant ``(ne, ne)`` matrix or
            one matrix per integration point, of shape ``(nx, ny, ne, ne)``,
            where ``ne`` is 6 for the classical laminated plate theory
            (CLPT), 8 for the first-order (FSDT) and 13 for the third-order
            (TSDT) shear deformation theories, the rows and columns being in
            the order of :meth:`.Shell._get_lam_ABD`. ``ABDnxny[i, j]`` is
            the matrix of the point ``(xs[i], ys[j])`` of
            :meth:`.Shell.integration_points`, i.e. the Legendre-Gauss points
            mapped to the integration limits
            :meth:`.Shell.integration_limits`, which are not ``[0, a] x [0,
            b]`` for a partial domain. The kernels read only the upper
            entries of each block, therefore the matrix and each of its
            blocks, e.g. ``A``, ``B`` and ``D`` for the CLPT, must be
            symmetric, and the transverse shear uncoupled from the other
            strains, as for any laminate, otherwise a ``ValueError`` is
            raised. It is used as given, ``force_orthotropic_laminate``
            applies only to :attr:`.Shell.ABD`.
        NLgeom : bool, optional
            Flag to indicate if geometrically non-linearities should be
            considered. Default ``False``.
        inc : float, optional
            Load factor of the incremented follower pressure loads, used only
            with ``NLgeom=True``, see :meth:`.Shell.calc_kCfollower`.
            Dimensionless. Default ``1.``.

        Returns
        -------
        kC : csr_matrix
            The matrix described above. Also stored in ``Shell.matrices``
            under the key ``'kC'``.

        Notes
        -----
        With the strain operator `\boldsymbol{B} = \boldsymbol{B}_L + \boldsymbol{B}_{NL}`, whose non-linear
        part `\boldsymbol{B}_{NL}` is linear in the rotations `\beta_x`, `\beta_y` of
        the von Kármán terms, `w_{,x}` and `w_{,y}` (`w_{,y} - v/r` for the
        Sanders-Koiter kinematics, see :mod:`panels.models`), and the
        constitutive matrix `\boldsymbol{ABD}` of ``ABDnxny``, the linear and the
        non-linear constitutive stiffness matrices are

        .. math::

            \boldsymbol{K}_{C0} = \int \boldsymbol{B}_L^\top \boldsymbol{ABD} \boldsymbol{B}_L dA \qquad
            \boldsymbol{K}_{CNL} = \int \left( \boldsymbol{B}_L^\top \boldsymbol{ABD} \boldsymbol{B}_{NL} + \boldsymbol{B}_{NL}^\top \boldsymbol{ABD}
            \boldsymbol{B}_L + \boldsymbol{B}_{NL}^\top \boldsymbol{ABD} \boldsymbol{B}_{NL} \right) dA

        over the integration domain. With ``NLgeom=False`` the result is
        `\boldsymbol{K}_{C0}`, whether or not ``c`` is given, because the kernels zero
        `w_{,x}` and `w_{,y}` and every large displacement term is built from
        them. With ``NLgeom=True`` the result is `\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} +
        \boldsymbol{K}_{GNL}`, evaluated at ``c``. Despite the name, the returned matrix
        is not purely constitutive in two cases:

        - With ``NLgeom=True`` the returned matrix also contains
          `\boldsymbol{K}_{GNL}`, the geometric stiffness of the membrane stress
          carried by the non-linear part of the strain,
          `\boldsymbol{\varepsilon}_{NL} = \{w_{,x}^2/2, w_{,y}^2/2, w_{,x} w_{,y}\}`,
          i.e. `\boldsymbol{K}_G` of :meth:`.Shell.calc_kG` with the stresses
          `\boldsymbol{A}\boldsymbol{\varepsilon}_{NL}`. This term is deliberately collected here
          instead of in :meth:`.Shell.calc_kG`, so that

          .. math::
              \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL} + \boldsymbol{K}_G(\bar{\boldsymbol{N}} + \boldsymbol{N}_L)

          with the first three terms returned by this method and the last one
          by :meth:`.Shell.calc_kG`, is the exact Jacobian of
          :meth:`.Shell.calc_fint`, while :meth:`.Shell.calc_kG` stays
          homogeneous of degree one in ``c``, as linear buckling requires.

        - If ``c_cte`` is given, or any of the attributes ``Nxx_cte``,
          ``Nyy_cte`` or ``Nxy_cte`` is non-zero, the geometric stiffness
          `\boldsymbol{K}_G(\boldsymbol{N}_{cte})` of that constant stress state is added into the
          returned matrix, which is then no longer `\boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} +
          \boldsymbol{K}_{GNL}`.
          This is the device used to superpose combined load cases, where the
          eigenvalue of a linear buckling analysis must multiply only part of
          the applied loads. The constant stress state does not scale with
          that eigenvalue precisely because it is added here and not in
          :meth:`.Shell.calc_kG`.

        The load stiffness of the follower pressure loads makes the sum of
        this matrix and :meth:`.Shell.calc_kG` the exact Jacobian of
        :meth:`.Shell.calc_fint` at the same ``inc``. It is unsymmetric,
        which is why it is not added with ``finalize=False``: the caller of
        an assembly symmetrizes the assembled upper triangle.

        The closed-form matrices separate the two directions, with the
        integrals of Bardell's functions of :mod:`panels.bardell` over the
        whole domain, which requires a constant `\boldsymbol{ABD}`; the numerical
        integration uses the points of :meth:`.Shell.integration_points`.
        See :ref:`tangent_product` for the derivation of the terms.

        """
        msg('Calculating kC... ', level=2, silent=silent)

        self._rebuild()
        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()

        analytical_kC = True
        analytical_kG = True

        #NOTE the analytical matrices always integrate the full domain, only
        #     the numerical ones honour (x1, x2, y1, y2)
        if self.is_partial_domain():
            analytical_kC = False
            analytical_kG = False

        # This means a linear analysis is already performed (check panels\tests\tests_shell\test_nonlinear.py)
        # So, the next step is NL. So no analytical
        if c is not None:
            check_c(c, size)
            analytical_kC = False
        # For variable stiffness laminates
        if ABDnxny is not None or self.ABDnxny is not None:
            analytical_kC = False
        # For NL Geos, KC (and not KC0) needs to be used so only do it numerically
        if NLgeom:
            analytical_kC = False
            analytical_kG = False

        matrices = modelDB.db[self.model]['matrices']  # selects what matrix functions to use
            # self.model = model of the current shell obj
        matrices_num = modelDB.db[self.model]['matrices_num']

        # Num integration points
        ABDnxny, nx, ny = self._get_ABDnxny(ABDnxny, nx, ny)
        self._check_r()

        #NOTE c is forwarded to a ``double [::1]`` kernel argument, so it must
        #     always be a contiguous 1-D array of length ``size``. Passing
        #     None (or letting np.ascontiguousarray turn None into the shape
        #     (1,) array [nan]) makes the kernels read out of bounds, since
        #     bounds checking is disabled in the .pyx files.
        if c is not None:
            # returns a contiguous array, how matrices in C are stored. 1 after the other like matlab
            c = np.ascontiguousarray(c, dtype=DOUBLE)
        else:
            # Empty c if the interest is only on the heterogeneous
            # laminate properties
            c = np.zeros(size, dtype=DOUBLE)

        #NOTE ABDnxny is checked by Shell._check_ABDnxny() and its shape
        #     against nx, ny within the .pyx files

        # Calc Kc as per panels/panels/models then the pyx files given by ''matrices'' defined earlier
        # This calc KC0 - linear consitutive stiff mat (SA formulation paper - eq 11)
            # This will happen by default unless a something is specified that NL Geo is needed
        if analytical_kC:
            kC = matrices.fk0(self, size, row0, col0)
        # This is what happens for NL Geo
        else:
            kC = matrices_num.fkC_num(c, ABDnxny, self,
                     size, row0, col0, nx, ny, NLgeom=int(NLgeom))

        if c_cte is not None or any((self.Nxx_cte, self.Nyy_cte, self.Nxy_cte)):
            if any((self.Nxx_cte, self.Nyy_cte, self.Nxy_cte)):
                msg('NOTE: constant stress state taken into account by (Nxx_cte, Nyy_cte, Nxy_cte)', level=3, silent=silent)
            if c_cte is not None:
                msg('NOTE: constant stress state taken into account by c_cte', level=3, silent=silent)
                check_c(c_cte, size)
                c_cte = np.ascontiguousarray(c_cte, dtype=DOUBLE)
                analytical_kG = False
            else:
                #NOTE same as for c above, fkG_num takes a ``double [::1]``
                #     and a None would be dereferenced inside the nogil loop
                c_cte = np.zeros(size, dtype=DOUBLE)
            # This calc KG0 - Geo stiff mat at initial membrane stress state (SA formulation paper - eq 12) and adds it to K0 calc earlier
            # this is required for combined load cases where the eigenvalue
            # lambda should be applied to only some of the applied forces
            if analytical_kG:
                kC += matrices.fkG0(self.Nxx_cte, self.Nyy_cte, self.Nxy_cte, self, size, row0, col0)
            else:
                kC += matrices_num.fkG_num(c_cte, ABDnxny, self, size,
                                           row0, col0, nx, ny,
                                           self.Nxx_cte, self.Nyy_cte,
                                           self.Nxy_cte)

        if finalize:
            kC = finalize_symmetric_matrix(kC)
            #NOTE unsymmetric, hence added after the symmetrization
            if NLgeom and self.has_follower_loads():
                kC = kC + self.calc_kCfollower(c=c, inc=inc, size=size,
                        row0=row0, col0=col0, silent=silent)
        self.matrices['kC'] = kC

        msg('finished!', level=2, silent=silent)

        return kC


    def calc_kG(self, size=None, row0=0, col0=0, silent=True, finalize=True,
            c=None, nx=None, ny=None, ABDnxny=None, NLgeom=False):
        r"""Calculate the (initial stress or) geometric stiffness matrix

        The returned matrix is the geometric stiffness of the membrane stress
        state obtained from the *linear* part of the strain evaluated at
        ``c``, superposed with the constant stress state given by the
        attributes ``Nxx``, ``Nyy`` and ``Nxy``. When ``c`` is not given, only
        the latter contributes.

        See :meth:`.Shell.calc_kC` for details on each parameter.

        The closed-form matrix of the constant stress state is used unless
        ``c`` is given, ``NLgeom=True`` or the integration domain is partial,
        see :meth:`.Shell.is_partial_domain`; then the matrix is integrated
        numerically with ``ABDnxny``, ``nx`` and ``ny``.

        Parameters
        ----------
        size : int or str, optional
            Size of the matrix, see :meth:`.Shell.calc_kC`. Default ``None``,
            the size of this shell.
        row0, col0 : int, optional
            Offset in the matrix of an assembly. Default ``0``.
        silent : bool, optional
            Default ``True``, no log message is printed.
        finalize : bool, optional
            Symmetrize and check the matrix, ``False`` when assembling.
            Default ``True``.
        c : np.ndarray or None, optional
            Ritz constants of the configuration, a 1-D array of size
            ``size``, whose linear membrane stresses enter the matrix.
            Default ``None``, only ``Nxx``, ``Nyy`` and ``Nxy`` contribute.
        nx, ny : int or None, optional
            Number of integration points, see :meth:`.Shell.calc_kC`.
            Default ``None``.
        ABDnxny : array-like or None, optional
            Constitutive matrix, see :meth:`.Shell.calc_kC`. Default
            ``None``.
        NLgeom : bool, optional
            Only forces the numerical integration, see the notes. Default
            ``False``.

        Returns
        -------
        kG : csr_matrix
            The geometric stiffness matrix. Also stored in ``Shell.matrices``
            under the key ``'kG'``.

        Notes
        -----
        With the rows `\boldsymbol{G}` of the approximation functions of the rotations
        `\beta_x = w_{,x}` and `\beta_y = w_{,y}` (`w_{,y} - v/r` for the
        Sanders-Koiter kinematics) of the von Kármán terms, see
        :mod:`panels.models`, the geometric stiffness of the membrane
        stresses `\boldsymbol{N}` is

        .. math::

            \boldsymbol{K}_G(\boldsymbol{N}) = \int \boldsymbol{G}^\top \boldsymbol{N} \boldsymbol{G} dA \qquad
            \boldsymbol{N} = \begin{bmatrix} N_{xx} & N_{xy} \\ N_{xy} & N_{yy}
                  \end{bmatrix}

        over the integration domain, with `\boldsymbol{N} = \bar{\boldsymbol{N}} + \boldsymbol{N}_L`, `\bar{\boldsymbol{N}}` the
        stresses of the attributes ``Nxx``, ``Nyy``, ``Nxy`` and `\boldsymbol{N}_L` those
        of the linear part of the strains at ``c``. The linear buckling
        problem of a pre-buckling (fundamental) state `\hat{\boldsymbol{c}}`, e.g. of
        a linear static analysis, is `(\boldsymbol{K}_{C0} + \lambda
        \boldsymbol{K}_G(\hat{\boldsymbol{c}}))\boldsymbol{u} = \boldsymbol{0}`, see
        :meth:`.Shell.calc_kC` for the stresses that must not scale with
        `\lambda`.

        The returned matrix is homogeneous of degree one in ``c``, which is
        what makes it usable as the right-hand side of the linear buckling
        eigenvalue problem. The geometric stiffness of the stress carried by
        the *non-linear* part of the strain is therefore not included here, it
        is returned by :meth:`.Shell.calc_kC` when ``NLgeom=True``. For the
        same reason ``NLgeom`` does not change the value of this matrix, it
        only forces the numerical instead of the analytical integration.

        """
        msg('Calculating kG... ', level=2, silent=silent)

        self._rebuild()
        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()

        analytical_kG = True

        #NOTE see the note in Shell.calc_kC()
        if self.is_partial_domain():
            analytical_kG = False

        if c is not None:
            check_c(c, size)
            c = np.ascontiguousarray(c, dtype=DOUBLE)
            if any((self.Nxx, self.Nyy, self.Nxy)):
                msg('NOTE: stress state taken into account using ALSO (Nxx, Nyy, Nxy)', level=3, silent=silent)
            analytical_kG = False
        else:
            c = np.zeros(size, dtype=DOUBLE)
            if any((self.Nxx, self.Nyy, self.Nxy)):
                msg('NOTE: stress state taken into account using ONLY (Nxx, Nyy, Nxy)', level=3, silent=silent)
        if NLgeom:
            analytical_kG = False

        matrices = modelDB.db[self.model]['matrices']
        matrices_num = modelDB.db[self.model]['matrices_num']

        self._check_r()

        if analytical_kG:
            kG = matrices.fkG0(self.Nxx, self.Nyy, self.Nxy, self, size, row0, col0)

        else:
            ABDnxny, nx, ny = self._get_ABDnxny(ABDnxny, nx, ny)
            kG = matrices_num.fkG_num(c, ABDnxny, self, size, row0, col0,
                                      nx, ny, self.Nxx, self.Nyy, self.Nxy)

        if finalize:
            kG = finalize_symmetric_matrix(kG)
        self.matrices['kG'] = kG

        msg('finished!', level=2, silent=silent)

        return kG


    def calc_kT(self, size=None, row0=0, col0=0, silent=True, finalize=True,
            c=None, nx=None, ny=None, ABDnxny=None, inc=1.):
        r"""Calculate the tangent stiffness matrix `\boldsymbol{K}_T`

        The tangent stiffness matrix is the derivative of
        :meth:`.Shell.calc_fint` with respect to ``c``, assembled as the sum
        of :meth:`.Shell.calc_kC` and :meth:`.Shell.calc_kG`, both with
        ``NLgeom=True`` and at ``c``, always with the numerical integration.
        With follower pressure loads, the load stiffness
        :meth:`.Shell.calc_kCfollower` at the load factor ``inc`` is
        included, through :meth:`.Shell.calc_kC`, and the matrix is then
        unsymmetric in general.

        See :meth:`.Shell.calc_kC` for details on each parameter. Note that
        ``c_cte`` is not forwarded, so a constant stress state can only be
        given through the attributes ``Nxx_cte``, ``Nyy_cte`` and ``Nxy_cte``.
        The matrices of both calls are also stored in ``Shell.matrices``
        under the keys ``'kC'`` and ``'kG'``.

        Parameters
        ----------
        size : int or str, optional
            Size of the matrix, see :meth:`.Shell.calc_kC`. Default ``None``,
            the size of this shell.
        row0, col0 : int, optional
            Offset in the matrix of an assembly. Default ``0``.
        silent : bool, optional
            Default ``True``, no log message is printed.
        finalize : bool, optional
            Passed to :meth:`.Shell.calc_kC` and :meth:`.Shell.calc_kG`,
            ``False`` when assembling. Default ``True``.
        c : np.ndarray or None, optional
            Ritz constants of the configuration, a 1-D array of size
            ``size``. Default ``None``, the undeformed state.
        nx, ny : int or None, optional
            Number of integration points, see :meth:`.Shell.calc_kC`.
            Default ``None``.
        ABDnxny : array-like or None, optional
            Constitutive matrix, see :meth:`.Shell.calc_kC`. Default
            ``None``.
        inc : float, optional
            Load factor of the incremented follower pressure loads,
            dimensionless. Default ``1.``.

        Returns
        -------
        kT : csr_matrix
            The tangent stiffness matrix. Also stored in ``Shell.matrices``
            under the key ``'kT'``.

        Notes
        -----
        The tangent stiffness matrix is the exact Jacobian of
        :meth:`.Shell.calc_fint` with respect to ``c``, assembled as

        .. math::
            \boldsymbol{K}_T = \boldsymbol{K}_{C0} + \boldsymbol{K}_{CNL} + \boldsymbol{K}_{GNL} + \boldsymbol{K}_G(\bar{\boldsymbol{N}} + \boldsymbol{N}_L)

        where all terms but the last come from
        :meth:`.Shell.calc_kC` with ``NLgeom=True``, and the last one from
        :meth:`.Shell.calc_kG` with ``NLgeom=True``. Note that `\boldsymbol{K}_{GNL}`
        is grouped with the constitutive matrix and not with the geometric
        one, see the notes in :meth:`.Shell.calc_kC` and
        :meth:`.Shell.calc_kG`. Being the exact Jacobian, it gives the
        quadratic convergence of the Newton-Raphson iterations.

        """
        kC = self.calc_kC(size=size, row0=row0, col0=col0, silent=silent, finalize=finalize,
            c=c, nx=nx, ny=ny, ABDnxny=ABDnxny, NLgeom=True, inc=inc)
        kG = self.calc_kG(size=size, row0=row0, col0=col0, silent=silent, finalize=finalize,
            c=c, nx=nx, ny=ny, ABDnxny=ABDnxny, NLgeom=True)
        kT = kC + kG
        self.matrices['kT'] = kT

        return kT


    def calc_kTv(self, c, v, size=None, col0=0, silent=True, nx=None,
            ny=None, ABDnxny=None, inc=1.):
        r"""Product of the tangent stiffness matrix by a vector

        Returns `\boldsymbol{K}_T(\boldsymbol{c}) \boldsymbol{v}`, with `\boldsymbol{K}_T` of :meth:`.Shell.calc_kT` at
        the load factor ``inc``, i.e. the directional derivative of
        :meth:`.Shell.calc_fint` along ``v``, without assembling `\boldsymbol{K}_T`,
        including the constant membrane stresses ``Nxx_cte``, ``Nyy_cte`` and
        ``Nxy_cte``. The cost is about that of :meth:`.Shell.calc_fint`,
        instead of the assembly of `\boldsymbol{K}_T`, whose cost grows with the square
        of the number of terms. With follower pressure loads, the product of
        their load stiffness :meth:`.Shell.calc_kCfollower` is obtained from
        two evaluations of :meth:`.Shell.calc_fext_follower`, see the notes.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the configuration, 1-D array of size ``size``.
        v : np.ndarray
            The vector, with the size of ``c``.
        size : int or str, optional
            See :meth:`.Shell.calc_fint`. Default ``None``, the size of this
            shell.
        col0 : int, optional
            See :meth:`.Shell.calc_fint`. Default ``0``.
        silent : bool, optional
            See :meth:`.Shell.calc_fint`. Default ``True``.
        nx, ny : int or None, optional
            See :meth:`.Shell.calc_fint`. Default ``None``.
        ABDnxny : array-like or None, optional
            See :meth:`.Shell.calc_fint`. Default ``None``.
        inc : float, optional
            See :meth:`.Shell.calc_fint`. Dimensionless. Default ``1.``.

        Returns
        -------
        kTv : np.ndarray
            The product, 1-D array of size ``size``.

        Raises
        ------
        NotImplementedError
            If the numerical kernels of the model have no ``calc_kTv``.

        Notes
        -----
        At each integration point, see :ref:`tangent_kTv`,

        .. math::

            \boldsymbol{K}_T \boldsymbol{v} = \int \left( (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top \boldsymbol{ABD} (\boldsymbol{B}_L +
            \boldsymbol{B}_{NL}) \boldsymbol{v} + \boldsymbol{G}^\top \boldsymbol{N} \boldsymbol{G} \boldsymbol{v} \right) dA

        where `\boldsymbol{\beta} = \boldsymbol{G} \boldsymbol{c}` are the rotations of the von Kármán
        terms, `\boldsymbol{B}_L + \boldsymbol{B}_{NL}` the strain operator, `\boldsymbol{B}_{NL}` being
        linear in `\boldsymbol{\beta}`, `\boldsymbol{ABD}` the constitutive matrix of
        ``ABDnxny`` and `\boldsymbol{N}` the membrane stresses at ``c``, plus the
        constant ones ``Nxx_cte``, ``Nyy_cte`` and ``Nxy_cte``. Since the
        strains are `\boldsymbol{\varepsilon}(\boldsymbol{c}) = (\boldsymbol{B}_L + \tfrac{1}{2} \boldsymbol{B}_{NL})
        \boldsymbol{c}`, a quadratic form whose bilinear form is `\boldsymbol{B}_L + \boldsymbol{B}_{NL}`,
        the first term is the derivative of the strains and the second that
        of `\boldsymbol{B}_{NL}^\top` in the internal force vector `\int (\boldsymbol{B}_L +
        \boldsymbol{B}_{NL})^\top \boldsymbol{ABD} \boldsymbol{\varepsilon}(\boldsymbol{c}) dA`. Each
        step at a point is a loop over the `m n` terms, against the `m^2
        n^2` pairs of terms of the assembly of `\boldsymbol{K}_T`.

        With follower pressure loads, the product of their load stiffness
        `\boldsymbol{K}_F`, :meth:`.Shell.calc_kCfollower`, is the central difference

        .. math::

            \boldsymbol{K}_F \boldsymbol{v} = -\frac{\partial \boldsymbol{F}_p}{\partial \boldsymbol{c}} \boldsymbol{v} =
            -\frac{\boldsymbol{F}_p(\boldsymbol{c} + t \boldsymbol{v}) - \boldsymbol{F}_p(\boldsymbol{c} - t \boldsymbol{v})}{2 t}

        which is exact for any `t`, `\boldsymbol{F}_p(\boldsymbol{c})` being at most quadratic in
        `\boldsymbol{c}`; `t = \max(\|\boldsymbol{c}\|/\|\boldsymbol{v}\|, 1)` avoids the cancellation of large
        and small terms.

        """
        msg('Calculating kT*v...', level=2, silent=silent)
        matrices_num = modelDB.db[self.model].get('matrices_num')
        calc_kTv = getattr(matrices_num, 'calc_kTv', None)
        if calc_kTv is None:
            raise NotImplementedError('calc_kTv not implemented for model {0}'
                                      .format(self.model))
        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()

        self._check_r()
        ABDnxny, nx, ny = self._get_ABDnxny(ABDnxny, nx, ny)

        #NOTE see the note in Shell.calc_kC() on why c and v must be checked
        #     before being forwarded
        c = np.ascontiguousarray(c, dtype=DOUBLE)
        v = np.ascontiguousarray(v, dtype=DOUBLE)
        check_c(c, size)
        check_c(v, size)
        kTv = np.asarray(calc_kTv(c, v, ABDnxny, self, size, col0, nx, ny,
                                  self.Nxx_cte, self.Nyy_cte, self.Nxy_cte))
        if self.has_follower_loads():
            norm_v = np.linalg.norm(v)
            if norm_v > 0:
                t = max(np.linalg.norm(c)/norm_v, 1.)
                fp = lambda cc: self.calc_fext_follower(cc, inc=inc,
                        size=size, col0=col0, reference=False)
                kTv = kTv - (fp(c + t*v) - fp(c - t*v))/(2*t)

        msg('finished!', level=2, silent=silent)

        return kTv


    def has_follower_loads(self):
        r"""Tell whether any pressure load is a follower load, see
        :meth:`.Shell.add_pressure_load`

        Both ``pressure_loads`` and ``pressure_loads_inc`` are checked.

        Returns
        -------
        has_follower : bool
            ``True`` when a pressure load was added with a ``follower``
            other than ``False`` or ``None``.

        """
        return any(len(load) > 5 and load[5] is not None
                   for load in self.pressure_loads + self.pressure_loads_inc)


    def _follower_matrices(self):
        matrices_num = modelDB.db[self.model]['matrices_num']
        if not hasattr(matrices_num, 'fkCfollower_num'):
            raise NotImplementedError('follower pressure loads are not '
                                      'implemented for model {0}'.format(
                                          self.model))
        return matrices_num


    def calc_kCfollower(self, c=None, inc=1., size=None, row0=0, col0=0,
            silent=True, finalize=True):
        r"""Calculate the load stiffness matrix of the follower pressure loads

        The follower pressure loads, see :meth:`.Shell.add_pressure_load`,
        have a force vector, :meth:`.Shell.calc_fext_follower`, which
        depends on the configuration, and the returned matrix is minus its
        derivative with respect to the Ritz constants, evaluated at ``c``,
        with the pressures of the incremented loads (``cte=False``)
        multiplied by the load factor ``inc`` and those of the constant loads
        by ``1``, integrated over the patch of each load. It enters the
        tangent stiffness matrix through :meth:`.Shell.calc_kC`, see
        :meth:`.Shell.calc_kT`, and the linear buckling problem of a follower
        pressure as ``lb(kC0, kG + kCfollower)``, with ``kG`` and
        ``kCfollower`` for a unit load factor.

        The matrix is unsymmetric in general, therefore it is assembled in
        full and never symmetrized. For ``follower='linear'`` it does not
        depend on ``c``.

        Parameters
        ----------
        c : array-like or None, optional
            The Ritz constants, with the size ``size``. Only used by the
            loads with ``follower='quadratic'``. ``None`` is the undeformed
            state.
        inc : float, optional
            Load factor of the incremented follower loads. Dimensionless.
            Default ``1.``.
        size : int or str, optional
            See :meth:`.Shell.calc_kC`. Default ``None``, the size of this
            shell.
        row0, col0 : int, optional
            See :meth:`.Shell.calc_kC`. Default ``0``.
        silent : bool, optional
            See :meth:`.Shell.calc_kC`. Default ``True``.
        finalize : bool, optional
            Checks for ``nan`` and ``inf`` values. The matrix is never
            symmetrized. Default ``True``.

        Returns
        -------
        kCfollower : csr_matrix
            Also stored in ``Shell.matrices`` under the key ``'kCfollower'``.

        Notes
        -----
        With the force vector `\boldsymbol{F}_p(\boldsymbol{c})` of :meth:`.Shell.add_pressure_load`,
        the returned matrix is the load stiffness

        .. math::
            \boldsymbol{K}_F = - \frac{\partial \boldsymbol{F}_p}{\partial \boldsymbol{c}}

        which enters the tangent stiffness matrix as `\boldsymbol{K}_T = \boldsymbol{K}_C + \boldsymbol{K}_G +
        \boldsymbol{K}_F`. For the first-order area vector `\boldsymbol{F}_p(\boldsymbol{c})` is
        affine in `\boldsymbol{c}`, whence a constant matrix, and for the complete
        area vector it is quadratic. In the linear buckling problem of a
        hydrostatic prestress, `p = \lambda p_0`, the matrix multiplies
        `\lambda` together with `\boldsymbol{K}_G` of the linear prestress state.

        For the first-order area vector, the skew part of the load stiffness
        is a sum of a contour integral over the edges of the loaded region,
        of `p` times the products of the variations of `w` and of the
        displacement normal to the edge, and of an integral of the
        derivatives of `p` over the region. The matrix is therefore
        symmetric, a conservative load, when the pressure is uniform over the
        whole integration domain and, on each edge, either `w` or the
        displacement normal to the edge is zero, e.g. simply supported or
        clamped panels, symmetry planes and closed rings, in agreement with
        Hibbitt [1]_, pp. 938-939, and, for pressures depending on the
        reference coordinates, with Schweizerhof and Ramm [2]_, Table 2 and
        the conditions of their Eqs. (32)-(35); it is unsymmetric for free
        or partially restrained edges, for patches, whose edges are inside
        the shell, and for a varying ``p(x, y)``. For
        ``follower='quadratic'`` the matrix at ``c`` different from zero is
        not symmetric in general. See :ref:`follower_pressure`.

        References
        ----------
        .. [1] Hibbitt, H. D., "Some follower forces and load stiffness",
           International Journal for Numerical Methods in Engineering,
           14(6), 937-941, 1979.
        .. [2] Schweizerhof, K., and Ramm, E., "Displacement dependent
           pressure loads in nonlinear finite element analyses", Computers &
           Structures, 18(6), 1099-1114, 1984.
           https://doi.org/10.1016/0045-7949(84)90154-8

        """
        msg('Calculating kCfollower... ', level=2, silent=silent)
        self._rebuild()
        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()
        self._check_r()
        if c is None:
            c = np.zeros(size, dtype=DOUBLE)
        else:
            c = np.ascontiguousarray(c, dtype=DOUBLE)
            check_c(c, size)
        kCf = csr_matrix((size, size), dtype=DOUBLE)
        patches = pressure_patches(self, inc, follower_only=True)
        if patches:
            matrices_num = self._follower_matrices()
        for patch in patches:
            x1, x2, y1, y2 = patch['limits']
            quadratic = int(patch['follower'] == 'quadratic')
            kCf = kCf + csr_matrix(matrices_num.fkCfollower_num(c,
                patch['pnxny'], x1, x2, y1, y2, self, size, row0, col0,
                quadratic))
        kCf.eliminate_zeros()
        if finalize:
            assert not np.any(np.isnan(kCf.data))
            assert not np.any(np.isinf(kCf.data))
        self.matrices['kCfollower'] = kCf
        msg('finished!', level=2, silent=silent)
        return kCf


    def calc_fext_follower(self, c, inc=1., size=None, col0=0,
            reference=True):
        r"""Force vector `\boldsymbol{F}_p(\boldsymbol{c})` of the follower pressure loads

        The pressures of the incremented loads are multiplied by ``inc``. With
        ``reference=False`` the part of the undeformed state, `\boldsymbol{F}_p(0)`,
        which :meth:`.Shell.calc_fext` already contains, is excluded. See
        :meth:`.Shell.add_pressure_load`. Only the loads added with a
        ``follower`` contribute; the dead pressure loads do not.

        Parameters
        ----------
        c : array-like
            Ritz constants of the configuration, 1-D of size ``size``.
        inc : float, optional
            Load factor of the incremented follower loads. Dimensionless.
            Default ``1.``.
        size : int or str, optional
            Size of the vector, see :meth:`.Shell.calc_fext`. Default
            ``None``, the size of this shell.
        col0 : int, optional
            Offset in the force vector of an assembly. Default ``0``.
        reference : bool, optional
            Whether the part of the undeformed state is included. Default
            ``True``.

        Returns
        -------
        fp : np.ndarray
            The force vector, 1-D array of size ``size``.

        """
        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()
        self._check_r()
        c = np.ascontiguousarray(c, dtype=DOUBLE)
        check_c(c, size)
        fp = np.zeros(size, dtype=DOUBLE)
        patches = pressure_patches(self, inc, follower_only=True)
        if patches:
            matrices_num = self._follower_matrices()
        for patch in patches:
            x1, x2, y1, y2 = patch['limits']
            quadratic = int(patch['follower'] == 'quadratic')
            fp += np.asarray(matrices_num.calc_fext_follower(c,
                patch['pnxny'], x1, x2, y1, y2, self, size, col0, quadratic,
                int(reference)))
        return fp


    def calc_kM(self, size=None, row0=0, col0=0, h_nxny=None, rho_nxny=None,
            nx=None, ny=None, silent=True, finalize=True, I_nxny=None):
        r"""Calculate the mass matrix

        By default the density enters through the moments of the laminate
        ``Shell.lam``, `I_n = \int \rho z^n dz`, ``intrho``, ``intrhoz``,
        ..., ``intrhoz6``, with `z` measured from the reference surface and
        the laminate spanning `-h/2 + d \le z \le h/2 + d`, `d` being the
        offset ``Shell.offset``, which accounts exactly for the density of
        each ply, and the closed-form matrix is returned. The matrix is
        integrated numerically with ``nx`` by ``ny`` points, with the same
        moments, when the integration domain is partial, see
        :meth:`.Shell.is_partial_domain`, and with the moments at each
        integration point when ``h_nxny``, ``rho_nxny`` or ``I_nxny`` is
        given: ``h_nxny`` and ``rho_nxny``, each defaulting to ``lam.h`` and
        to the mean density ``lam.intrho/lam.h``, give the moments of a
        uniform density over `-h/2 + d \le z \le h/2 + d`, whereas
        ``I_nxny`` gives the moments directly. The laminate ``Shell.lam``
        must already be built, e.g. by the constructor or by
        :meth:`.Shell.calc_kC`; its ply densities are ``rhos`` when given
        by the user, otherwise ``rho`` for every ply, as of the last
        rebuild, see :class:`.Shell`.

        Parameters
        ----------
        size : int or str, optional
            Size of the matrix, see :meth:`.Shell.calc_kC`. Default ``None``,
            the size of this shell.
        row0, col0 : int, optional
            Offset in the matrix of an assembly. Default ``0``.
        h_nxny : (nx, ny) array-like or None, optional
            The thickness of the laminate at each integration point, a
            length. With ``I_nxny`` it is used only by the TSDT, for `c_1 =
            4/(3h^2)`. Default ``None``.
        rho_nxny : (nx, ny) array-like or None, optional
            The density of the laminate at each integration point, uniform
            through the thickness, a mass per unit volume. Not accepted with
            ``I_nxny``. Default ``None``.
        nx, ny : int or None, optional
            Number of integration points along `x` and `y`. Default
            ``None``, the attributes ``nx`` and ``ny``.
        silent : bool, optional
            Default ``True``, no log message is printed.
        finalize : bool, optional
            Symmetrize and check the matrix, ``False`` when assembling.
            Default ``True``.
        I_nxny : (nx, ny, K) array-like or None, optional
            The moments `I_n` at each integration point, `n = 0, 1, 2`,
            ``K = 3``, for the CLPT and the FSDT, and for the TSDT the
            moments `J_n = \int \rho (z - d)^n dz` about the mid-surface,
            `n = 0, 1, 2, 3, 4, 6`, ``K = 6``, a mass per unit area times
            length**n. Default ``None``.

        Returns
        -------
        kM : csr_matrix or coo_matrix
            The mass matrix, a ``csr_matrix`` with ``finalize=True``,
            otherwise the matrix returned by the kernel. Also stored in
            ``Shell.matrices`` under the key ``'kM'``.

        Raises
        ------
        ValueError
            When ``I_nxny`` is given with ``rho_nxny``, or with ``h_nxny``
            for a model that is not based on the TSDT, or when the shape of
            ``I_nxny`` is not ``(nx, ny, K)``.

        Notes
        -----
        The mass matrix is the second derivative with respect to the
        velocities of the Ritz constants of the kinetic energy of the
        displacement field through the thickness, with the rotary and, for
        the TSDT, the higher-order inertia:

        .. math::

            \boldsymbol{M} = \int_A \int_{-h/2 + d}^{h/2 + d} \rho(z) \left(
            (\boldsymbol{S}^u(z))^\top \boldsymbol{S}^u(z) +
            (\boldsymbol{S}^v(z))^\top \boldsymbol{S}^v(z) +
            (\boldsymbol{S}^w)^\top \boldsymbol{S}^w \right) dz \, dA

        where the rows `\boldsymbol{S}^u(z)`, `\boldsymbol{S}^v(z)`,
        `\boldsymbol{S}^w` of the approximation functions give
        `u(z) = u + z \phi_x - c_1 z^3 (\phi_x + w_{,x})`, `v(z) = v + z
        \phi_y - c_1 z^3 (\phi_y + w_{,y})` and `w`. For the CLPT `\phi_x =
        -w_{,x}`, `\phi_y = -w_{,y}` and `c_1 = 0`; for the FSDT `c_1 = 0`;
        for the TSDT `c_1 = 4/(3h^2)`; with the Sanders-Koiter kinematics
        `v/r` is added to the rotation `\phi_y` multiplied by `z`, see
        :mod:`panels.models`, and for the TSDT `z^3` is replaced by `(z -
        d)^3 + d^3`, the cubic term about the mid-surface, see
        :meth:`._get_lam_ABD`. The integrand being a polynomial in `z`, the
        integral through the thickness is a combination of the moments
        `I_n` of the density, `n \le 2`; for the TSDT, whose field is in
        `\zeta = z - d` a polynomial with the powers `\zeta^0`, `\zeta` and
        `\zeta^3`, of the moments `J_n` about the mid-surface, `n \le 6`
        without `n = 5`, those of :meth:`._get_lam_mid`, with `d` in the
        coefficients. The convention of the offset, a laminate centred at
        `z = d`, is that of :func:`composites.laminated_plate`, also used
        by the stiffness matrices. A uniform density `\rho` gives

        .. math::

            I_n = \rho \frac{(h/2 + d)^{n + 1} - (-h/2 + d)^{n + 1}}{n + 1}

        and `J_n` is `I_n` with `d = 0`.

        """
        msg('Calculating kM... ', level=2, silent=silent)

        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny

        matrices = modelDB.db[self.model]['matrices']
        matrices_num = modelDB.db[self.model]['matrices_num']

        self._check_r()

        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()

        tsdt = '_tsdt_' in self.model
        powers = (0, 1, 2, 3, 4, 6) if tsdt else (0, 1, 2)
        lam = self.lam
        #NOTE the TSDT uses the moments about the mid-surface
        mom = self._get_lam_mid() if tsdt else lam
        lam_moments = [mom.intrho, mom.intrhoz, mom.intrhoz2, mom.intrhoz3,
                       mom.intrhoz4, None, mom.intrhoz6]

        if I_nxny is not None:
            if rho_nxny is not None:
                raise ValueError('I_nxny and rho_nxny cannot be given together')
            if h_nxny is not None and not tsdt:
                raise ValueError('h_nxny is used with I_nxny only by the TSDT')
            I_nxny = np.asarray(I_nxny, dtype=DOUBLE)
            if I_nxny.shape != (nx, ny, len(powers)):
                raise ValueError('I_nxny must have the shape {0}'.format(
                                 (nx, ny, len(powers))))

        if I_nxny is None and h_nxny is None and rho_nxny is None:
            I_input = np.array([lam_moments[k] for k in powers]
                               + ([lam.h] if tsdt else []), dtype=DOUBLE)
            #NOTE see the note in Shell.calc_kC()
            if self.is_partial_domain():
                kM = matrices_num.fkM_num(self, I_input, size, row0, col0, nx, ny)
            else:
                kM = matrices.fkM(self, size, row0, col0)
        else:
            h = np.full((nx, ny), lam.h, dtype=DOUBLE)
            if h_nxny is not None:
                h[:, :] = h_nxny
            if I_nxny is None:
                rho = np.full((nx, ny), lam.intrho/lam.h, dtype=DOUBLE)
                if rho_nxny is not None:
                    rho[:, :] = rho_nxny
                d = 0. if tsdt else self.offset
                z1 = -h/2 + d
                z2 = h/2 + d
                I_nxny = np.stack([rho*(z2**(k + 1) - z1**(k + 1))/(k + 1)
                                   for k in powers], axis=2)
            if tsdt:
                I_nxny = np.concatenate((I_nxny, h[..., None]), axis=2)
            I_input = np.ascontiguousarray(I_nxny, dtype=DOUBLE)
            kM = matrices_num.fkM_num(self, I_input, size, row0, col0, nx, ny)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.matrices['kM'] = kM

        msg('finished!', level=2, silent=silent)

        return kM


    def calc_kA(self, size=None, row0=0, col0=0, silent=True, finalize=True):
        r"""Calculate the aerodynamic matrix using the linear piston theory

        The matrix of the aerodynamic load along `w` of a supersonic flow
        along `x` or `y`, with the parameters ``beta`` and, for a flow along
        `x` over a cylinder, ``gamma``, which multiply the terms of the load
        in `w_{,x}` (or `w_{,y}`) and in `w`, see the notes. Flat plates have
        ``gamma = 0``, and a ``gamma`` given for a plate model is ignored
        with a warning. For a flow along `y` the term in ``gamma`` is not
        included.

        The flow direction is the attribute ``flow``, ``'x'`` or ``'y'``.
        When the attribute ``beta`` is ``None``, ``beta`` and ``gamma`` are
        calculated from the attributes ``Mach``, ``rho_air``, ``air_speed``
        and, for the cylinders, ``r``, a ``Mach`` equal to ``1`` being
        changed to ``1.0001`` with a warning; otherwise the attributes
        ``beta`` and ``gamma`` are used, a ``gamma`` of ``None`` meaning
        ``0``. The attributes ``beta`` and ``gamma`` are updated with the
        values used. The closed-form matrix is used, or the numerically
        integrated one for a partial domain, see
        :meth:`.Shell.is_partial_domain`. The matrix is added to the
        stiffness matrix of a frequency analysis, e.g. ``freq(kC + kG + kA,
        kM)`` of ``structsolve``.

        Parameters
        ----------
        size : int or str, optional
            Size of the matrix, see :meth:`.Shell.calc_kC`. Default ``None``,
            the size of this shell.
        row0, col0 : int, optional
            Offset in the matrix of an assembly. Default ``0``.
        silent : bool, optional
            Default ``True``, no log message is printed.
        finalize : bool, optional
            Checks for ``nan`` and ``inf`` values. The matrix is never
            symmetrized. Default ``True``.

        Returns
        -------
        kA : coo_matrix
            The aerodynamic matrix, as returned by the kernel. Also stored in
            ``Shell.matrices`` under the key ``'kA'``.

        Raises
        ------
        ValueError
            If ``beta`` is ``None`` and ``Mach`` is ``None`` or lower than
            ``1``, or if ``flow`` is neither ``'x'`` nor ``'y'``.

        Notes
        -----
        In the linear piston theory, for a flow along `x`, the aerodynamic
        load on the shell along `w` is

        .. math::

            q = \beta w_{,x} + \gamma w

        and, for a flow along `y`, `q = \beta w_{,y}`. When ``beta`` is not
        given it is calculated from the Mach number `M` as `\beta =
        \rho_{air} U^2/\sqrt{M^2 - 1}`, with `U` the attribute ``air_speed``.
        The matrix computed by the kernels, rows and columns in the order of
        the Ritz constants, is

        .. math::

            \boldsymbol{K}_A = -\int \left(\beta (\boldsymbol{S}^w_{,x})^\top
            \boldsymbol{S}^w + \gamma (\boldsymbol{S}^w)^\top
            \boldsymbol{S}^w\right) dA

        with `\boldsymbol{S}^w_{,y}` and `\gamma = 0` for a flow along `y`,
        where `\boldsymbol{S}^w` is the row of the approximation functions of
        `w`,
        entering the equations of motion as `(\boldsymbol{K} + \boldsymbol{K}_A + \lambda^2
        \boldsymbol{M})\boldsymbol{c} = \boldsymbol{0}`, `\lambda^2 = -\omega^2` for a harmonic motion,
        e.g. with :func:`structsolve.freq`, see :meth:`.Shell.calc_cA` for
        the damping. This is the convention of panels: the term in `\gamma`
        is `-\gamma \int (\boldsymbol{S}^w)^\top \boldsymbol{S}^w dA` added to
        the stiffness, which
        softens the shell, as Krumhaar's correction does [1]_, see below.

        Since `\int (\boldsymbol{S}^w_{,x})^\top \boldsymbol{S}^w dA = -\int
        (\boldsymbol{S}^w)^\top \boldsymbol{S}^w_{,x} dA` when `w` vanishes at
        `x = 0` and `x = a`, the term in `\beta` is then `\beta \int
        (\boldsymbol{S}^w)^\top \boldsymbol{S}^w_{,x} dA`. The term in `\beta` of
        panels is the transpose of that of Eq. (21) of [2]_, `\boldsymbol{K}_{A_x} =
        \int (\boldsymbol{S}^w)^\top (-\beta \boldsymbol{S}^w_{,x} + \gamma \boldsymbol{S}^w) dA`, which is
        added to the stiffness in Eqs. (29)-(31) of [2]_, `(\boldsymbol{K}_{ps} +
        \boldsymbol{K}_A - \omega^2 \boldsymbol{M} + i \omega \boldsymbol{C}_A)\boldsymbol{c}_0 = \boldsymbol{0}`: with `w = 0`
        at `x = 0` and `x = a` the two differ by the sign of `\beta`, i.e.
        by the direction of the flow, on which the flutter boundary does not
        depend. The sign of the term in `\gamma` of Eqs. (18), (21) and (29)
        of [2]_ is reversed, a mistake of the paper: added with a positive
        sign to the stiffness it stiffens the shell, whereas Krumhaar's
        correction softens it, as derived independently from the potential
        flow over a cylinder in pyfe3d; panels keeps the softening term.

        The term `\gamma w` is Krumhaar's correction of the piston theory for
        the external flow over a cylinder [1]_, with `w` positive outwards:

        .. math::

            \gamma = \frac{\beta}{2 r \sqrt{M^2 - 1}}

        such that an outward displacement lowers the pressure on the shell,
        which is pulled outwards, a softening effect that does not depend on
        the flow direction. It is obtained expanding the exact linear
        potential flow over a cylinder with a sinusoidal radial displacement
        for short wavelengths.

        References
        ----------
        .. [1] Krumhaar, H., "The accuracy of linear piston theory when
           applied to cylindrical shells", AIAA Journal, 1(6), 1448-1449,
           1963.
        .. [2] Castro, S. G. P., Guimaraes, T. A. M., Rade, D. A., and
           Donadon, M. V., "Flutter of stiffened composite panels considering
           the stiffener's base as a structural element", Composite
           Structures, 140, 36-43, 2016.
           https://doi.org/10.1016/j.compstruct.2015.12.056

        """
        msg('Calculating kA... ', level=2, silent=silent)

        matrices = modelDB.db[self.model]['matrices']
        matrices_num = modelDB.db[self.model]['matrices_num']

        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()

        self._check_r()

        if self.beta is None:
            if self.Mach is None:
                raise ValueError('Mach number cannot be a NoneValue')
            elif self.Mach < 1:
                raise ValueError('Mach number must be >= 1')
            elif self.Mach == 1:
                warn('Mach number forced to 1.0001')
                self.Mach = 1.0001
            Mach = self.Mach
            beta = self.rho_air * self.air_speed**2 / (Mach**2 - 1)**0.5
            #NOTE the curvature term exists only for cylindrical shells
            if modelDB.db[self.model].get('requires_r', False):
                gamma = beta*1./(2.*self.r*(Mach**2 - 1)**0.5)
            else:
                gamma = 0.
        else:
            beta = self.beta
            gamma = self.gamma if self.gamma is not None else 0.
            if not modelDB.db[self.model].get('requires_r', False):
                if gamma != 0.:
                    warn('gamma = {0} ignored, plates have gamma = 0'.format(
                         gamma), level=1, silent=silent)
                gamma = 0.

        self.beta = beta
        self.gamma = gamma

        #NOTE see the note in Shell.calc_kC()
        partial_domain = self.is_partial_domain()
        if self.flow.lower() == 'x':
            if partial_domain:
                kA = matrices_num.fkAx_num(self, size, row0, col0, self.nx, self.ny)
            else:
                kA = matrices.fkAx(beta, gamma, self, size, row0, col0)

        elif self.flow.lower() == 'y':
            if partial_domain:
                kA = matrices_num.fkAy_num(self, size, row0, col0, self.nx, self.ny)
            else:
                kA = matrices.fkAy(beta, self, size, row0, col0)

        else:
            raise ValueError('Invalid flow value, must be x or y')

        if finalize:
            assert np.any(np.isnan(kA.data)) == False
            assert np.any(np.isinf(kA.data)) == False
        self.matrices['kA'] = kA

        msg('finished!', level=2, silent=silent)

        return kA


    def calc_cA(self, aeromu, silent=True, size=None, finalize=True):
        r"""Calculate the aerodynamic damping matrix using the piston theory

        The closed-form matrix of the kernel, always over the whole domain
        and at the offset ``0`` of the matrix, is multiplied by the
        imaginary unit ``1j``.

        Parameters
        ----------
        aeromu : float
            Aerodynamic damping coefficient `\mu` of the piston theory, a
            pressure per unit velocity, mass/(area time), see
            :meth:`.StiffPanelBay._aero_parameters`.
        silent : bool, optional
            Default ``True``, no log message is printed.
        size : int or None, optional
            Size of the matrix. Default ``None``, the size of this shell. A
            string is not accepted, unlike :meth:`.Shell.calc_kC`.
        finalize : bool, optional
            Symmetrize and check the matrix. Default ``True``.

        Returns
        -------
        cA : csr_matrix or coo_matrix
            The complex damping matrix, a ``csr_matrix`` with
            ``finalize=True``. Also stored in ``Shell.matrices`` under the
            key ``'cA'``.

        Notes
        -----
        The kernel computes `-\mu \int (\boldsymbol{S}^w)^\top \boldsymbol{S}^w
        dA` over the whole shell, with `\mu` the parameter ``aeromu`` and
        `\boldsymbol{S}^w` the row of
        the approximation functions of `w`, i.e. the matrix of the
        aerodynamic damping load of the piston theory, proportional to `\mu
        \dot{w}`; :class:`.StiffPanelBay` computes `\mu = \beta (M^2 - 2)/(M
        a_\infty (M^2 - 1))`, with `a_\infty` the speed of sound, Eq. (18)
        of [1]_, see :meth:`.Shell.calc_kA` for `\beta`. This is `\boldsymbol{C}_A =
        \int (\boldsymbol{S}^w)^\top (-\mu \boldsymbol{S}^w) dA` of Eq. (23) of [1]_. For the
        harmonic motion `\boldsymbol{c} e^{i \omega t}`, `\dot{w} = i \omega w`, and
        the damping enters Eqs. (29)-(31) of [1]_ as `+i \omega \boldsymbol{C}_A`; the
        factor ``1j`` is already in ``cA`` `= i \boldsymbol{C}_A`, such that the
        eigenvalue problem is

        .. math::

            \left(\boldsymbol{K} + \boldsymbol{K}_A - \omega^2 \boldsymbol{M} + \omega \boldsymbol{C}_A\right)\boldsymbol{c} =
            \boldsymbol{0}

        with `\boldsymbol{K}_A` of :meth:`.Shell.calc_kA`, quadratic in `\omega`, e.g.
        solved in the state-space form of Eq. (32) of [1]_.

        References
        ----------
        .. [1] Castro, S. G. P., Guimaraes, T. A. M., Rade, D. A., and
           Donadon, M. V., "Flutter of stiffened composite panels considering
           the stiffener's base as a structural element", Composite
           Structures, 140, 36-43, 2016.
           https://doi.org/10.1016/j.compstruct.2015.12.056

        """
        msg('Calculating cA... ', level=2, silent=silent)

        if size is None:
            size = self.get_size()
        #NOTE fcA is only implemented analytically, over the whole domain
        matrices = modelDB.db[self.model]['matrices']
        cA = matrices.fcA(aeromu, self, size, 0, 0)
        cA = cA*(0+1j)

        if finalize:
            cA = finalize_symmetric_matrix(cA)
        self.matrices['cA'] = cA

        msg('finished!', level=2, silent=silent)

        return cA


    def uvw(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the ``fields`` parameter of the
        :class:`.Shell` object.

        The entries ``'u'``, ``'v'``, ``'w'``, ``'phix'`` and ``'phiy'`` of
        ``Shell.fields`` and ``'Xs'``, ``'Ys'`` of ``Shell.plot_mesh`` are
        updated, all with the shape of ``xs``. When ``xs`` or ``ys`` is
        ``None``, the points are the grid of ``gridx`` equally spaced values
        of `x` in ``[0, a]`` by ``gridy`` of `y` in ``[0, b]``, built by
        :func:`numpy.meshgrid`, of shape ``(gridy, gridx)``.

        Parameters
        ----------
        c : array-like
            The full set of Ritz constants, 1-D of size
            :meth:`.Shell.get_size`.
        xs : np.ndarray
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
            A length.
        ys : np.ndarray
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
            A length, with the same shape as ``xs``.
        gridx : int
            Number of points along the `x` axis where to calculate the
            displacement field. Default ``300``.
        gridy : int
            Number of points along the `y` where to calculate the
            displacement field. Default ``300``.

        Returns
        -------
        out : tuple
            Containing ``plot_mesh`` and ``fields``, the dictionaries
            ``Shell.plot_mesh`` and ``Shell.fields``.


        """
        #NOTE fuvw/fstrain take a ``double [::1]``, see the note in
        #     Shell.calc_kC() on why c must be checked before being forwarded
        c = np.ascontiguousarray(c, dtype=DOUBLE)
        check_c(c, self.get_size())

        xs, ys, xshape, yshape = self._default_field(xs, ys, gridx, gridy)
        fuvw = modelDB.db[self.model]['field'].fuvw
        us, vs, ws, phixs, phiys = fuvw(c, self, xs, ys, self.out_num_cores)

        self.plot_mesh['Xs'] = xs.reshape(xshape)
        self.plot_mesh['Ys'] = ys.reshape(yshape)
        self.fields['u'] = us.reshape(xshape)
        self.fields['v'] = vs.reshape(xshape)
        self.fields['w'] = ws.reshape(xshape)
        self.fields['phix'] = phixs.reshape(xshape)
        self.fields['phiy'] = phiys.reshape(xshape)

        return self.plot_mesh, self.fields


    def strain(self, c, xs=None, ys=None, gridx=300, gridy=300, NLgeom=True):
        r"""Calculate the strain field

        The generalized strains of the model are stored in ``Shell.fields``,
        with the shape of ``xs``, and the points in ``'Xs'`` and ``'Ys'`` of
        ``Shell.plot_mesh``. The points are chosen as in
        :meth:`.Shell.uvw`. An unset radius ``r`` of a plate is set to
        ``0.``.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the strain field
            calculation, 1-D of size :meth:`.Shell.get_size`.
        xs : np.ndarray, optional
            The `x` coordinates where to calculate the strains. Default
            ``None``.
        ys : np.ndarray, optional
            The `y` coordinates where to calculate the strains, must
            have the same shape as ``xs``. Default ``None``.
        gridx : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used. Default ``300``.
        gridy : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used. Default ``300``.
        NLgeom : bool
            Flag to indicate whether non-linear strain components should be considered.
            Default ``True``.

        Returns
        -------
        plot_mesh : dict
            ``Shell.plot_mesh``, with the points in ``'Xs'`` and ``'Ys'``.
        fields : dict
            ``Shell.fields``, a dictionary of ``np.ndarrays`` with the keys:
            ``(exx, eyy, gxy, kxx, kyy, kxy)``. The models based on
            shear deformation theories also return the transverse shear
            strains ``(gyz, gxz)`` and, for the third-order theory, the
            higher-order terms ``(kxx3, kyy3, kxy3, gyz2, gxz2)``. The other
            entries of ``Shell.fields`` are left as they were.

        Notes
        -----
        The strains at a distance `z` from the reference surface are
        ``exx + z*kxx + ((z - d)**3 + d**3)*kxx3`` and ``gyz + (z -
        d)**2*gyz2``, `d` being the offset, see :meth:`._get_lam_ABD`, and likewise
        for the other components, with the generalized strains of the
        kinematics of each model, see :mod:`panels.models` and
        :mod:`panels.models.fsdt_tsdt_field`. With ``NLgeom=False`` the
        quadratic terms of the membrane strains are omitted.

        """
        #NOTE fuvw/fstrain take a ``double [::1]``, see the note in
        #     Shell.calc_kC() on why c must be checked before being forwarded
        c = np.ascontiguousarray(c, dtype=DOUBLE)
        check_c(c, self.get_size())
        #NOTE fstrain reads Shell.r as a double, an unset radius of a plate
        #     must be normalized first
        self._check_r()
        xs, ys, xshape, yshape = self._default_field(xs, ys, gridx, gridy)
        field = modelDB.db[self.model]['field']
        strains = field.fstrain(c, self, xs, ys, self.out_num_cores, int(NLgeom))

        self.plot_mesh['Xs'] = xs.reshape(xshape)
        self.plot_mesh['Ys'] = ys.reshape(yshape)
        for name, value in zip(self._strain_names(), strains):
            self.fields[name] = value.reshape(xshape)

        return self.plot_mesh, self.fields


    def _strain_names(self):
        field = modelDB.db[self.model]['field']
        if hasattr(field, 'strain_names'):
            return field.strain_names(self)
        return ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy')


    def _stress_names(self):
        field = modelDB.db[self.model]['field']
        if hasattr(field, 'stress_names'):
            return field.stress_names(self)
        return ('Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy')


    def stress(self, c, ABD=None, xs=None, ys=None, gridx=300, gridy=300, NLgeom=True):
        r"""Calculate the stress field

        The strains are calculated with :meth:`.Shell.strain`, which updates
        ``Shell.fields`` and ``Shell.plot_mesh``, and the generalized
        stresses of the model, the products of ``ABD`` by the strains, are
        stored in ``Shell.fields`` too, with the shape of ``xs``.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the strain field
            calculation, 1-D of size :meth:`.Shell.get_size`.
        ABD : np.ndarray, optional
            The constitutive matrix, either constant, of shape ``(ne, ne)``,
            or one for each point, of shape ``xs.shape + (ne, ne)``, see
            ``ABDnxny`` in :meth:`.Shell.calc_kC`. By default the attribute
            ``ABDnxny`` or, when it is ``None``, :attr:`.Shell.ABD`.
        xs : np.ndarray, optional
            The `x` coordinates where to calculate the strains. Default
            ``None``.
        ys : np.ndarray, optional
            The `y` coordinates where to calculate the strains, must
            have the same shape as ``xs``. Default ``None``, see
            :meth:`.Shell.uvw`. When neither ``xs`` and ``ys`` nor
            ``ABD`` are given and the attribute ``ABDnxny`` has one matrix
            per integration point, the stresses are calculated at the
            integration points, where the stiffness is known: ``xs[i, j],
            ys[i, j]`` is the point of ``ABDnxny[i, j]``, see
            :meth:`.Shell.integration_points`.
        gridx : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used. Default ``300``.
        gridy : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used. Default ``300``.
        NLgeom : bool
            Flag to indicate whether non-linear strain components should be considered.
            Default ``True``.

        Returns
        -------
        plot_mesh : dict
            ``Shell.plot_mesh``, with the points in ``'Xs'`` and ``'Ys'``.
        fields : dict
            ``Shell.fields``, a dictionary of ``np.ndarrays`` with the keys:
            ``(Nxx, Nyy, Nxy, Mxx, Myy, Mxy)``. The models based on
            shear deformation theories also return the transverse shear
            forces ``(Qy, Qx)`` and, for the third-order theory, the
            higher-order resultants ``(Pxx, Pyy, Pxy, Ry, Rx)``, in the
            positions of ``(kxx3, kyy3, kxy3, gyz2, gxz2)`` among the
            strains, see :meth:`.Shell.strain`. It also contains the strains
            of :meth:`.Shell.strain`.

        Raises
        ------
        ValueError
            If the attribute ``ABDnxny`` has one matrix per integration point,
            ``ABD`` is not given and ``xs`` or ``ys`` is given; if no
            constitutive matrix is defined; or if ``ABD`` has neither the
            shape ``(ne, ne)`` nor ``xs.shape + (ne, ne)``.

        Notes
        -----
        The generalized stresses are the resultants through the thickness
        conjugate to the generalized strains, e.g. ``Pxx`` to ``kxx3`` and
        ``Ry`` to ``gyz2``, see :func:`.fsdt_tsdt_field.stress_names` and
        :meth:`.Shell._get_lam_ABD`. With ``NLgeom=True`` the membrane
        stresses include those of the quadratic terms of the strains, as in
        :meth:`.Shell.calc_fint`.

        """
        if ABD is None and self.ABDnxny is not None:
            ABD = self.ABDnxny
            if np.ndim(ABD) == 4:
                if xs is not None or ys is not None:
                    raise ValueError('Shell.ABDnxny is known only at the '
                        'integration points, see Shell.integration_points(), '
                        'give either xs = ys = None or the stiffness ABD of '
                        'the points xs, ys')
                xs, ys = self.integration_points(*np.shape(ABD)[:2])[:2]
                xs, ys = np.meshgrid(xs, ys, indexing='ij')
        elif ABD is None:
            ABD = self.ABD
            if ABD is None:
                raise ValueError('Laminate ABD matrix not defined for shell')
        ABD = np.asarray(ABD, dtype=DOUBLE)

        plot_mesh, fields = self.strain(c, xs, ys, gridx, gridy, NLgeom)
        strains = [fields[name] for name in self._strain_names()]
        ne = len(strains)
        shape = strains[0].shape
        if ABD.shape not in ((ne, ne), shape + (ne, ne)):
            raise ValueError('ABD must have the shape {0} or {1}, got {2}'
                             .format((ne, ne), shape + (ne, ne), ABD.shape))

        self.plot_mesh = plot_mesh
        #NOTE ABD[..., i, j] is a scalar for a constant ABD, otherwise an
        #     array of the shape of the strains
        for i, name in enumerate(self._stress_names()):
            self.fields[name] = sum(e*ABD[..., i, j] for j, e in enumerate(strains))

        return self.plot_mesh, self.fields


    def add_point_load(self, x, y, fx, fy, fz, cte=True):
        r"""Add a point load with three components

        The load ``[x, y, fx, fy, fz]`` is appended to the attribute
        ``point_loads`` when ``cte=True``, otherwise to ``point_loads_inc``,
        and enters the external force vector, see :func:`.shell_fext`.

        Parameters
        ----------
        x : float
            The `x` position, a length.
        y : float
            The `y` position, a length, the arc length for the cylindrical
            shells.
        fx : float
            The `x` component of the force vector, a force.
        fy : float
            The `y` component of the force vector, a force.
        fz : float
            The `z` component of the force vector, a force.
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis. Default ``True``.

        """
        if cte:
            self.point_loads.append([x, y, fx, fy, fz])
        else:
            self.point_loads_inc.append([x, y, fx, fy, fz])


    def add_distr_load_fixed_x(self, x, funcx=None, funcy=None, funcz=None, cte=True):
        r"""Add a distributed force g(y) at a fixed x position

        The load ``[x, None, funcx, funcy, funcz]`` is appended to the
        attribute ``distr_loads`` when ``cte=True``, otherwise to
        ``distr_loads_inc``. :func:`.shell_fext` integrates it along `y` with
        ``n`` Gauss-Legendre points, a function that is ``None`` giving a
        zero component.

        Parameters
        ----------
        x : float
            The fixed `x` position, a length.
        funcx, funcy, funcz : function, optional
            The functions of the distributed force components, will be used
            from `y=0` to `y=b`. At least one of the three must be defined
            Each is called with the coordinate `y` and returns a force per
            unit length. Default ``None``.
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis. Default ``True``.

        Raises
        ------
        ValueError
            If ``funcx``, ``funcy`` and ``funcz`` are all ``None``.

        """
        if not any((funcx, funcy, funcz)):
            raise ValueError('At least one function must be different than None')
        if cte:
            self.distr_loads.append([x, None, funcx, funcy, funcz])
        else:
            self.distr_loads_inc.append([x, None, funcx, funcy, funcz])


    def add_distr_load_fixed_y(self, y, funcx=None, funcy=None, funcz=None, cte=True):
        r"""Add a distributed force g(x) at a fixed y position

        The load ``[None, y, funcx, funcy, funcz]`` is appended to the
        attribute ``distr_loads`` when ``cte=True``, otherwise to
        ``distr_loads_inc``. :func:`.shell_fext` integrates it along `x` with
        ``m`` Gauss-Legendre points, a function that is ``None`` giving a
        zero component.

        Parameters
        ----------
        y : float
            The fixed `y` position, a length.
        funcx, funcy, funcz : function, optional
            The functions of the distributed force components, will be used
            from `x=0` to `x=a`. At least one of the three must be defined
            Each is called with the coordinate `x` and returns a force per
            unit length. Default ``None``.
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis. Default ``True``.

        Raises
        ------
        ValueError
            If ``funcx``, ``funcy`` and ``funcz`` are all ``None``.

        """
        if not any((funcx, funcy, funcz)):
            raise ValueError('At least one function must be different than None')
        if cte:
            self.distr_loads.append([None, y, funcx, funcy, funcz])
        else:
            self.distr_loads_inc.append([None, y, funcx, funcy, funcz])


    def add_pressure_load(self, p, x1=None, x2=None, y1=None, y2=None,
            cte=True, follower=False, zp=0.):
        r"""Add a pressure load distributed over the domain or over a patch

        The pressure acts along the normal `z` of the undeformed mid-surface,
        with the same sign convention as ``fz`` in :meth:`.add_point_load`,
        i.e. a positive ``p`` pushes along `+z`. For the cylindrical shells
        `w` is positive outwards, see :meth:`.Shell.global_coords`, such that
        an external pressure is negative.

        With ``follower=False`` the load is a dead load: its direction does
        not follow the deformation in the non-linear analyses. Otherwise it
        is a follower (hydrostatic) pressure, normal to the deformed
        mid-surface and acting on its deformed area, with the area vector of
        the deformed mid-surface truncated to first order in the
        displacement gradients for ``follower=True`` or ``'linear'``, and
        complete for ``'quadratic'``, see the notes.

        In the undeformed state a follower load equals the dead load, and
        :meth:`.Shell.calc_fext` returns this reference vector. The part that
        depends on the configuration enters :meth:`.Shell.calc_fint` (with a
        negative sign) and its derivative the tangent stiffness matrix
        through :meth:`.Shell.calc_kCfollower`, both at the load factor
        ``inc`` of the incremented loads. The configuration-dependent force
        vector is :meth:`.Shell.calc_fext` with ``c``.

        The equivalent nodal force vector is computed by
        :func:`.shell_fext`, integrating `p(x, y)` times the approximation
        functions of `w` with ``nx*ny`` Gauss-Legendre points over the loaded
        patch, the same points used by the follower terms.

        Parameters
        ----------
        p : float or function
            The pressure, as a force per unit area of mid-surface. Either a
            constant, or a function ``p(x, y)`` of the physical coordinates,
            being `y` the arc length for the cylindrical shells.
        x1, x2, y1, y2 : float, optional
            Physical limits of the loaded patch, ``x1 <= x <= x2`` and ``y1 <=
            y <= y2``. A limit that is ``None`` is the corresponding limit of
            the integration domain of the shell, see
            :meth:`.Shell.integration_limits`, such that by default the whole
            domain is loaded. A patch is intersected with the integration
            domain, which matters only for the partial domains of an
            assembly. Lengths. Default ``None``.
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis. Note that the non-linear solvers of ``structsolve``
            scale the whole :meth:`.Shell.calc_fext` by their load factor,
            so with them use ``cte=False`` for loads that follow the load
            factor. Default ``True``, the load is appended to the attribute
            ``pressure_loads``, otherwise to ``pressure_loads_inc``, as
            ``[x1, x2, y1, y2, p, follower, zp]`` with ``follower`` ``None``,
            ``'linear'`` or ``'quadratic'``.
        follower : bool or str, optional
            ``False`` for a dead load, ``True`` or ``'linear'`` for a follower
            pressure with the first-order area vector, ``'quadratic'`` for the
            complete area vector. ``None`` is taken as ``False``. Default
            ``False``.
        zp : float, optional
            Position along `z` of the surface on which the pressure acts,
            e.g. ``h/2`` for the outer face of a cylinder. For the cylindrical
            shells the pressure is applied as ``p*(1 + zp/r)`` per unit area
            of mid-surface, see :func:`.pressure_factor`. Plates are not
            affected. A length. Default ``0.``.

        Raises
        ------
        ValueError
            If ``p`` is ``None``; if a limit lies outside ``[0, a]`` or ``[0,
            b]`` by more than a relative ``1e-12``, checked when ``a`` or
            ``b`` is set; if ``x1 >= x2`` or ``y1 >= y2``; or if
            ``follower`` is not one of the values above.

        Notes
        -----
        The follower pressure has the force vector, the virtual work of `p`
        on the displacements,

        .. math::
            \boldsymbol{F}_p(\boldsymbol{c}) = \int p \left((\boldsymbol{S}^u)^\top a_x +
                         (\boldsymbol{S}^v)^\top a_y + (\boldsymbol{S}^w)^\top
                         a_z\right) dx dy

        where `\boldsymbol{S}^u`, `\boldsymbol{S}^v`, `\boldsymbol{S}^w` are
        the rows of the approximation functions of `u`, `v`, `w` and `(a_x,
        a_y, a_z)` is the area vector of the deformed mid-surface per unit
        undeformed area, `\boldsymbol{a} = \boldsymbol{X}_{,x} \times
        \boldsymbol{X}_{,y}`, with the tangent vectors of the kinematics
        `\boldsymbol{X}_{,x} = (1 + u_{,x}, v_{,x}, w_{,x})` and
        `\boldsymbol{X}_{,y} = (u_{,y}, 1 + v_{,y} + k_w w, w_{,y} - k_v v)`
        on the local axes of each point, `k_w = 1/r`
        for the cylinders and `k_v = 1/r` for the Sanders kinematics only,
        zero otherwise. The Donnell cylinder neglects `v/r` in the rotation,
        as its strains do. With ``follower=True`` or ``'linear'`` it is
        truncated to first order in the displacement gradients, consistently
        with the moderate-rotation strains of the models:

        .. math::
            a_x = -w_{,x}, \quad a_y = -w_{,y} + k_v v, \quad
            a_z = 1 + u_{,x} + v_{,y} + k_w w

        With the moderate-rotation ordering of the strains, `u_{,x} \sim
        \varepsilon` and `w_{,x} \sim \varepsilon^{1/2}`, this keeps every
        term up to the order `\varepsilon p \delta w`, while the second-order
        part of the area vector contributes terms of the order of what the
        von Kármán, Donnell and Sanders strains neglect. With
        ``'quadratic'`` the complete (bilinear) area vector is used, e.g.
        for sensitivity checks of the truncation. The rotations of the shear
        deformation theories do not enter the load. A hydrostatic head, a
        pressure depending on the current position, is not implemented. See
        :ref:`follower_pressure`.

        """
        if p is None:
            raise ValueError('p must be a float or a function p(x, y)')
        if not callable(p):
            p = float(p)
        for name, value, length in (('x1', x1, self.a), ('x2', x2, self.a),
                                    ('y1', y1, self.b), ('y2', y2, self.b)):
            if value is not None and length is not None:
                tol = 1e-12*length
                if not (-tol <= value <= length + tol):
                    raise ValueError('{0}={1!r} is outside the shell, 0 <= '
                                     '{0} <= {2!r}'.format(name, value, length))
        if x1 is not None and x2 is not None and not x1 < x2:
            raise ValueError('x1 < x2 is required, got x1={0!r} and '
                             'x2={1!r}'.format(x1, x2))
        if y1 is not None and y2 is not None and not y1 < y2:
            raise ValueError('y1 < y2 is required, got y1={0!r} and '
                             'y2={1!r}'.format(y1, y2))
        if follower is False or follower is None:
            follower = None
        elif follower is True or follower == 'linear':
            follower = 'linear'
        elif follower != 'quadratic':
            raise ValueError("follower must be False, True, 'linear' or "
                             "'quadratic', got {0!r}".format(follower))
        zp = float(zp)
        if cte:
            self.pressure_loads.append([x1, x2, y1, y2, p, follower, zp])
        else:
            self.pressure_loads_inc.append([x1, x2, y1, y2, p, follower, zp])


    def add_point_pd(self, x, y, ku, up, kv, vp, kw, wp, cte=True,
                     kphix=None, phixp=None, kphiy=None, phiyp=None, z=0.):
        r"""Add a point prescribed displacement with up to five components

        o/p = [location and equivalent force]

        The in-plane displacements `u` and `v` are prescribed at the
        thickness coordinate ``z``, by default at the reference surface,
        e.g. at the faces of the laminate to clamp an edge by its faces, see
        :ref:`boundary_conditions_face_clamp`.

        With the penalty method, the default of :class:`.MultiDomain`, the
        prescribed displacement enters the external force vector as `k u_p`,
        see :func:`.shell_fext`, and the penalty stiffness must be added to
        the stiffness matrix, e.g. with :func:`.fkCpd`. With
        ``MultiDomain(..., pd_method='null-space')`` it is imposed exactly,
        see :ref:`null_space_pd`: the values of the penalty constants are
        not used, a component being prescribed when its penalty constant is
        neither ``None`` nor zero and its value is not ``None``.

        The list ``[x, y, ku*up, kv*vp, kw*wp, z]``, a component being
        ``0.`` when its penalty constant or value is ``None``, is appended to
        the attribute ``point_pds``, and ``[x, y, up, vp, wp, phixp, phiyp,
        z]``, with ``None`` for the components not prescribed, to
        ``point_pds_values``; with ``cte=False`` to ``point_pds_inc`` and
        ``point_pds_values_inc`` instead.

        Parameters
        ----------
        x : float
            The `x` position, a length.
        y : float
            The `y` position, a length, the arc length for the cylindrical
            shells.
        ku, kv, kw : float
            The `x,y,z` component of the penalty stiffness of the prescribed displacement.
            A force per unit length, or ``None``.
        up, vp, wp : float
            The `x,y,z` components of the prescribed displacement. Lengths,
            or ``None``.
        cte : bool, optional
            Constant prescribed displacements are not incremented
            during the non-linear analysis. Default ``True``.
        kphix, phixp, kphiy, phiyp : float, optional
            Prescribed rotations, only with ``pd_method='null-space'``, the
            rotations of the normal at the reference surface `z = 0`:
            `-w_{,x}` and `-w_{,y}` for the classical laminated plate
            theory, `\phi_x` and `\phi_y` for the first-order shear
            deformation theory, and `\theta_x` and `\theta_y` of the Notes
            for the third-order one, plus `v/r` about `x` with the
            Sanders-Koiter kinematics. The rotations ``phixp`` and
            ``phiyp`` are dimensionless; of ``kphix`` and ``kphiy`` only
            whether they are ``None`` or zero is used. Default ``None``.
        z : float, optional
            Thickness coordinate, measured from the reference surface, where
            ``up`` and ``vp`` are prescribed, a length, normally within the
            laminate, ``offset - h/2 <= z <= offset + h/2``; `w` and the
            rotations do not depend on it. With the penalty method, the
            same ``z`` must be given to the penalty stiffness, e.g. to
            :func:`.fkCpd`. Default ``0.``, the reference surface.

        Notes
        -----
        The in-plane displacements at the coordinate `z` are those of the
        displacement field of the model, e.g. for the third-order shear
        deformation theory (TSDT), with the offset `d` and the thickness
        `h` of the laminate:

        .. math::

            u(z) = u + z \phi_x - c_1 \left[(z - d)^3 + d^3\right]
            \left(\phi_x + w_{,x}\right), \qquad c_1 = \frac{4}{3 h^2}

        `u(z) = u + z \phi_x` for the first-order theory (FSDT) and `u(z) =
        u - z w_{,x}` for the classical laminated plate theory (CLPT), and
        likewise `v(z)`, with the term `z v/r` of the Sanders-Koiter
        kinematics, see :func:`.nullspace.pd_constraint_matrix`. A
        prescribed `u(z)` is therefore a linear constraint on `u`, `\phi_x`
        and `w_{,x}`.

        The prescribed rotations are those of the normal at the reference
        surface, the rotations also tied by the connections, see
        :func:`.kCsdt._rotation_x` and :func:`.kCsdt._rotation_y`. With the
        third-order shear deformation theory (TSDT) and the offset `d` of
        the laminate, of thickness `h`, they are

        .. math::

            \theta_x = u_{,z}\big|_{z=0} = \phi_x - \frac{4 d^2}{h^2}
            \left(\phi_x + w_{,x}\right) \qquad
            \theta_y = v_{,z}\big|_{z=0} = \Phi_y - \frac{4 d^2}{h^2}
            \left(\phi_y + w_{,y}\right)

        with `\phi_x` and `\phi_y` the rotations at the mid-surface, `z =
        d`, and `\Phi_y = \phi_y + v/r` with the Sanders-Koiter kinematics,
        `\Phi_y = \phi_y` otherwise. For the FSDT, and for the TSDT with
        `d = 0`, `\theta_x = \phi_x` and `\theta_y = \Phi_y`. See
        :ref:`tsdt_offset_prescribed_rotations`. For the TSDT, `u = 0`, `w
        = 0` and `\theta_x = 0` along an edge `x = cte` do not clamp it,
        the warping term of `u(z)` remaining: the clamp also needs `w_{,x}
        = 0`, which `u(z) = 0` prescribed at three coordinates `z` gives;
        prescribed at the two faces only, `u(z) = 0` gives the face clamp,
        see :ref:`boundary_conditions_face_clamp`.

        The penalty method adds to the energy, for each prescribed
        component, e.g. `u`,

        .. math::

            \frac{k_u}{2} \left(u(x, y) - u_p\right)^2 = \frac{k_u}{2}
            \left(\boldsymbol{g}_u \boldsymbol{c} - u_p\right)^2

        with `\boldsymbol{g}_u` the row of the approximation functions of `u` at the
        point, see the function ``fg`` of the model, whose derivatives give
        the penalty stiffness `k_u \boldsymbol{g}_u^\top \boldsymbol{g}_u` and the force `k_u u_p
        \boldsymbol{g}_u^\top`. The prescribed value is reached only in the limit of an
        infinite penalty constant: the error decreases with `k_u` relative to
        the stiffness of the structure, while the conditioning of the
        stiffness matrix worsens.

        """
        def given(k, value):
            return k is not None and k != 0 and value is not None
        z = float(z)
        values = [x, y] + [value if given(k, value) else None
                           for k, value in ((ku, up), (kv, vp), (kw, wp),
                                            (kphix, phixp), (kphiy, phiyp))]
        values.append(z)
        pd = [x, y] + [k*value if k is not None and value is not None else 0.
                       for k, value in ((ku, up), (kv, vp), (kw, wp))] + [z]
        if cte:
            self.point_pds.append(pd)
            # Adds the location and force
            self.point_pds_values.append(values)
        else:
            self.point_pds_inc.append(pd)
            self.point_pds_values_inc.append(values)


    def _add_distr_pd(self, x, y, ks, funcs, cte, z):
        r"""See :meth:`.add_distr_pd_fixed_x` and
        :meth:`.add_distr_pd_fixed_y`, ``ks`` and ``funcs`` for ``u, v, w,
        phix, phiy``"""
        if not any(ks):
            raise ValueError('At least one penalty constant must be different than None')
        if not any(funcs):
            raise ValueError('At least one function must be different than None')
        # Force funtns = k * displ ftn
        new_funcs = []
        for name, k, func in zip(('u', 'v', 'w', 'phix', 'phiy'), ks, funcs):
            new_func = None
            if (k is not None) or (func is not None): # k or func is specified
                if k is None or func is None: # if atmost 1 is specified the component is incomplete
                    raise ValueError('Both k{0} and func{0} must be specified'.format(name))
                new_func = lambda s, k=k, func=func: k*func(s) # s is the coordinate along the line
            new_funcs.append(new_func)
        z = float(z)
        values = [x, y] + [func if k is not None and k != 0 else None
                           for k, func in zip(ks, funcs)] + [z]
        pd = [x, y] + new_funcs[:3] + [z]
        if cte:
            self.distr_pds.append(pd)
            self.distr_pds_values.append(values)
        else:
            self.distr_pds_inc.append(pd)
            self.distr_pds_values_inc.append(values)


    def add_distr_pd_fixed_x(self, x, ku=None, kv=None, kw=None,
                             funcu=None, funcv=None, funcw=None, cte=True,
                             kphix=None, kphiy=None, funcphix=None,
                             funcphiy=None, z=0.):
        r"""Add a distributed prescribed displacement g(y) at a fixed x position

        The in-plane displacements `u` and `v` are prescribed at the
        thickness coordinate ``z``, see :meth:`.add_point_pd`.

        With the penalty method, the default of :class:`.MultiDomain`, the
        prescribed displacement enters the external force vector as `k
        g(y)`, see :func:`.shell_fext`, and the penalty stiffness must be
        added to the stiffness matrix, e.g. with :func:`.fkCld_xcte`. With
        ``MultiDomain(..., pd_method='null-space')`` it is imposed exactly,
        see :ref:`null_space_pd`: the values of the penalty constants are
        not used, a component being prescribed when its penalty constant is
        neither ``None`` nor zero.

        The list ``[x, None, ku*funcu, kv*funcv, kw*funcw, z]``, with
        ``None`` for a component without penalty constant and function, is
        appended to the attribute ``distr_pds``, and ``[x, None, funcu,
        funcv, funcw, funcphix, funcphiy, z]``, with ``None`` for a component
        whose penalty constant is ``None`` or zero, to ``distr_pds_values``;
        with
        ``cte=False`` to ``distr_pds_inc`` and ``distr_pds_values_inc``
        instead. A ``ValueError`` is raised when all penalty constants or all
        functions are ``None``, or when only one of the penalty constant and
        the function of a component is given.

        Parameters
        ----------
        x : float
            The fixed `x` position, a length.
        ku, kv, kw : float, optional
            The `x,y,z` components of the penalty stiffness of the prescribed
            displacement.  At least one of the three must be defined, and
            corresponding to the funcu, funcv, funcw specified.
            A force per unit length of the line per unit displacement, i.e.
            a force per unit area. Default ``None``.
        funcu, funcv, funcw : function, optional
            Specify in normal coordinates (x,y) not natural
            The functions of the distributed prescribed displacements, will be used
            from `y=0` to `y=b`. At least one of the three must be defined, and
            corresponding to the ku, kv, kw specified. Each is called with
            the coordinate `y` and returns a displacement, a length. Default
            ``None``.
        cte : bool, optional
            Constant prescribed displacements are not incremented during the non-linear
            analysis. Default ``True``.
        kphix, kphiy : float, optional
            Penalty constants of the prescribed rotations, only with
            ``pd_method='null-space'``, see :meth:`.add_point_pd`. Default
            ``None``.
        funcphix, funcphiy : function, optional
            Prescribed rotations of the normal at the reference surface,
            dimensionless, as functions of `y`, only with
            ``pd_method='null-space'``, see :meth:`.add_point_pd`.
            Default ``None``.
        z : float, optional
            Thickness coordinate, measured from the reference surface, of
            the prescribed `u` and `v`, a length, see :meth:`.add_point_pd`;
            with the penalty method the same ``z`` must be given to
            :func:`.fkCld_xcte`. Default ``0.``, the reference surface.

        Notes
        -----
        With the penalty method the energy of each prescribed component,
        e.g. `u`, is the integral along the line of the energy of
        :meth:`.add_point_pd`, `\int_0^b \frac{k_u}{2} (u(x, y) - g(y))^2
        dy`, with ``ku`` a penalty constant per unit length of the line,
        which gives the force `\int_0^b k_u g(y) \boldsymbol{g}_u^\top dy` of
        :func:`.shell_fext` and the penalty stiffness of :func:`.fkCld_xcte`.

        """
        self._add_distr_pd(x, None, (ku, kv, kw, kphix, kphiy),
                           (funcu, funcv, funcw, funcphix, funcphiy), cte, z)


    def add_distr_pd_fixed_y(self, y, ku=None, kv=None, kw=None,
                             funcu=None, funcv=None, funcw=None, cte=True,
                             kphix=None, kphiy=None, funcphix=None,
                             funcphiy=None, z=0.):
        r"""Add a distributed prescribed displacement g(x) at a fixed y position

        The in-plane displacements `u` and `v` are prescribed at the
        thickness coordinate ``z``, see :meth:`.add_point_pd`.

        With the penalty method, the default of :class:`.MultiDomain`, the
        prescribed displacement enters the external force vector as `k
        g(x)`, see :func:`.shell_fext`, and the penalty stiffness must be
        added to the stiffness matrix, e.g. with :func:`.fkCld_ycte`. With
        ``MultiDomain(..., pd_method='null-space')`` it is imposed exactly,
        see :ref:`null_space_pd`: the values of the penalty constants are
        not used, a component being prescribed when its penalty constant is
        neither ``None`` nor zero.

        The list ``[None, y, ku*funcu, kv*funcv, kw*funcw, z]`` is appended
        to the attribute ``distr_pds`` and ``[None, y, funcu, funcv, funcw,
        funcphix, funcphiy, z]`` to ``distr_pds_values``, or to
        ``distr_pds_inc`` and ``distr_pds_values_inc`` with ``cte=False``,
        and a ``ValueError`` is raised, as for
        :meth:`.add_distr_pd_fixed_x`.

        Parameters
        ----------
        y : float
            The fixed `y` position, a length.
        ku, kv, kw : float, optional
            The `x,y,z` components of the penalty stiffness of the prescribed
            displacement.  At least one of the three must be defined, and
            corresponding to the funcu, funcv, funcw specified.
            A force per unit length of the line per unit displacement, i.e.
            a force per unit area. Default ``None``.
        funcu, funcv, funcw : function, optional
            The functions of the distributed prescribed displacements, will be used
            from `x=0` to `x=a`. At least one of the three must be defined, and
            corresponding to the ku, kv, kw specified. Each is called with
            the coordinate `x` and returns a displacement, a length. Default
            ``None``.
        cte : bool, optional
            Constant prescribed displacements are not incremented during the non-linear
            analysis. Default ``True``.
        kphix, kphiy : float, optional
            Penalty constants of the prescribed rotations, only with
            ``pd_method='null-space'``, see :meth:`.add_point_pd`. Default
            ``None``.
        funcphix, funcphiy : function, optional
            Prescribed rotations of the normal at the reference surface,
            dimensionless, as functions of `x`, only with
            ``pd_method='null-space'``, see :meth:`.add_point_pd`.
            Default ``None``.
        z : float, optional
            Thickness coordinate, measured from the reference surface, of
            the prescribed `u` and `v`, a length, see :meth:`.add_point_pd`;
            with the penalty method the same ``z`` must be given to
            :func:`.fkCld_ycte`. Default ``0.``, the reference surface.

        Notes
        -----
        The penalty energy is that of :meth:`.add_distr_pd_fixed_x`,
        integrated along `x` from `0` to `a`.

        """
        self._add_distr_pd(None, y, (ku, kv, kw, kphix, kphiy),
                           (funcu, funcv, funcw, funcphix, funcphiy), cte, z)

    def has_rotation_pds(self):
        r"""Whether rotations are prescribed, which only
        ``MultiDomain(..., pd_method='null-space')`` supports

        Reads the lists ``point_pds_values``, ``point_pds_values_inc``,
        ``distr_pds_values`` and ``distr_pds_values_inc``, see
        :meth:`.Shell.add_point_pd`.

        Returns
        -------
        has_rotations : bool
            ``True`` when a value of ``phix`` or ``phiy`` is prescribed.

        """
        return any(values[5] is not None or values[6] is not None
                   for values in self.point_pds_values
                   + self.point_pds_values_inc + self.distr_pds_values
                   + self.distr_pds_values_inc)

    def clear_disps(self):

        '''Remove all the prescribed displacements

            Used to clear exisiting displacements for the panels
                Useful for non-linear runs where displacements are added for each increment

        The attributes ``point_pds``, ``point_pds_inc``, ``distr_pds``,
        ``distr_pds_inc``, ``point_pds_values``, ``point_pds_values_inc``,
        ``distr_pds_values`` and ``distr_pds_values_inc`` are set to empty
        lists.

        '''
        self.point_pds = []
        self.point_pds_inc = []
        self.distr_pds = []
        self.distr_pds_inc = []
        self.point_pds_values = []
        self.point_pds_values_inc = []
        self.distr_pds_values = []
        self.distr_pds_values_inc = []

    def clear_loads(self):
        '''Remove all the loads

            Used to clear exisiting loads for the panels
                Useful for non-linear runs where loads are added for each increment

        The attributes ``point_loads``, ``point_loads_inc``,
        ``distr_loads``, ``distr_loads_inc``, ``pressure_loads`` and
        ``pressure_loads_inc`` are set to empty lists. The prescribed
        displacements are kept, see :meth:`.Shell.clear_disps`.

        '''
        self.point_loads = []
        self.point_loads_inc = []
        self.distr_loads = []
        self.distr_loads_inc = []
        self.pressure_loads = []
        self.pressure_loads_inc = []

    def calc_stiffness_point_constraint(self, x, y, u=True, v=True, w=True, phix=False,
            phiy=False, kuvw=1.e6, kphi=1.e5):
        r"""Add a point constraint

        This can used to create different types of support and boundary
        conditions.

        The matrix is ``k*outer(g, g)`` summed over the constrained degrees
        of freedom, with ``g`` the row of the approximation functions of
        that degree of freedom at ``(x, y)``, see the function ``fg`` of the
        model. It is returned, not stored nor added to ``Shell.matrices``.

        Parameters
        ----------
        x, y : float
            Coordinates of the point contraint, lengths.
        u, v, w : bool, optional
            Translational degrees of freedom to be constrained. Default
            ``True``.
        phix, phiy : bool, optional
            Rotational degrees of freedom to be constrained. Default
            ``False``.
        kuvw : float, optional
            Penalty constant used for the translational constraints, a force
            per unit length. Default ``1.e6``.
        kphi : float, optional
            Penalty constant used for the rotational constraints, a moment
            per unit rotation. Default ``1.e5``.

        Returns
        -------
        kPC : numpy.matrix or csr_matrix
           Stiffness matrix concenrning this point constraint. It must be added
           to the constitutive stiffness matrix in order to be taken into
           account in the calculations. Of shape ``(size, size)``, with
           ``size`` from :meth:`.Shell.get_size`. Adding the dense outer
           products to the empty ``csr_matrix`` gives a dense
           :class:`numpy.matrix` when any degree of freedom is constrained.

        Notes
        -----
        The matrix is the second derivative of the penalty energy
        `\frac{k}{2} (\boldsymbol{g} \boldsymbol{c})^2` of each constrained degree of freedom,
        which prescribes a zero value approximately, with an accuracy and a
        conditioning that grow and worsen with `k`, see
        :meth:`.Shell.add_point_pd`. The rotations are those of the rows
        ``phix`` and ``phiy`` of ``fg``, which include `v/r` for the
        Sanders-Koiter kinematics.

        """
        fg = modelDB.db[self.model]['field'].fg
        size = self.get_size()
        g = np.zeros((5, size), dtype=DOUBLE)
        fg(g, x, y, self)
        gu, gv, gw, gphix, gphiy = g
        kPC = csr_matrix((size, size), dtype=DOUBLE)
        if u:
            kPC += kuvw*np.outer(gu, gu)
        if v:
            kPC += kuvw*np.outer(gv, gv)
        if w:
            kPC += kuvw*np.outer(gw, gw)
        if phix:
            kPC += kphi*np.outer(gphix, gphix)
        if phiy:
            kPC += kphi*np.outer(gphiy, gphiy)
        return kPC


    def calc_fext(self, inc=1., size=None, col0=0, silent=True, c=None,
                  include_pds=True):
        r"""Calculate the external force vector `\boldsymbol{F}_{ext}`

        The vector is the sum of the constant terms `\boldsymbol{F}_{ext_0}`, of the
        loads and prescribed displacements added with ``cte=True``, and of
        the terms `\boldsymbol{F}_{ext_\lambda}`, added with ``cte=False``, which
        are scaled by the parameter ``inc``.

        See the documentation of :func:`.shell_fext` for more details. The
        follower pressure loads, see :meth:`.Shell.add_pressure_load`,
        contribute with their value in the undeformed state, unless ``c`` is
        given.

        Parameters
        ----------
        inc : float, optional
            Since this function is called during the non-linear analysis,
            ``inc`` will multiply the terms `\boldsymbol{F}_{ext_\lambda}`.
            Dimensionless. Default ``1.``.
        size : int or str, optional
            The size of the force vector. Can be the size of the total internal
            force vector of a multidomain assembly. When using a string, for
            example, if '+1' is given it will add 1 to the shell's size obtained
            by the :meth:`.Shell.get_size`. Default ``None``, the size of
            this shell.
        col0 : int, optional
            Offset in a global force vector of an assembly. Default ``0``.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
            Default ``True``, nothing is printed.
        c : array-like or None, optional
            The Ritz constants, with the size ``size``. When given, the
            follower pressure loads are evaluated in this configuration,
            i.e. ``fext(c) = fext + (F_p(c) - F_p(0))``, the load vector that
            the arc-length solvers of ``structsolve`` use as the derivative
            of the residual with respect to the load factor. Default
            ``None``.
        include_pds : bool, optional
            Whether the forces `k u_p` of the prescribed displacements
            imposed with the penalty method are included.
            ``MultiDomain(..., pd_method='null-space')`` imposes them exactly
            instead, and excludes these forces. Default ``True``.

        Returns
        -------
        fext : np.ndarray
            The external force vector, 1-D array of size ``size``.

        """
        self._rebuild()
        msg('Calculating external forces...', level=2, silent=silent)
        fext = shell_fext(self, inc=inc, size=size, col0=col0,
                          include_pds=include_pds)
        if c is not None and self.has_follower_loads():
            fext += self.calc_fext_follower(c, inc=inc, size=fext.shape[0],
                                            col0=col0, reference=False)
        return fext


    def calc_fint(self, c, size=None, col0=0, silent=True, nx=None,
            ny=None, ABDnxny=None, inc=1.):
        r"""Calculate the internal force vector `\boldsymbol{F}_{int}`

        The internal force vector of the strains at ``c``, integrated
        numerically with the constitutive matrix ``ABDnxny``. With follower
        pressure loads, see :meth:`.Shell.add_pressure_load`, the part of
        their force vector that depends on the configuration,
        :meth:`.Shell.calc_fext_follower` with ``reference=False`` at the
        load factor ``inc``, is subtracted, such that the tangent stiffness
        matrix :meth:`.Shell.calc_kT` at ``inc`` is its derivative with
        respect to ``c``.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the internal forces
            calculation, 1-D of size ``size``.
        size : int or str, optional
            The size of the internal force vector. Can be the size of a global
            internal force vector of an assembly. When using a string,
            for example, if '+1' is given it will add 1 to the shell's size
            obtained by the :meth:`.Shell.get_size`. Default ``None``, the
            size of this shell.
        col0 : int, optional
            Offset in a global internal force vector of an assembly. Default
            ``0``.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
            Default ``True``, nothing is printed.
        nx, ny : int, optional
            Number of integration points along `x` and `y`, see
            :meth:`.Shell.calc_kC`. Default ``None``.
        ABDnxny : np.ndarray, optional
            The constitutive matrix, constant or per integration point, by
            default the attribute ``ABDnxny`` or :attr:`.Shell.ABD`, see
            :meth:`.Shell.calc_kC`.
        inc : float, optional
            Load factor of the incremented follower pressure loads.
            Dimensionless. Default ``1.``.

        Returns
        -------
        fint : np.ndarray
            The internal force vector, 1-D array of size ``size``.

        Raises
        ------
        ValueError
            If the attribute ``model`` is not a valid model, or if its
            numerical kernels or their ``calc_fint`` are not implemented.

        Notes
        -----
        The elastic internal force vector is the derivative of the strain
        energy with respect to the Ritz constants,

        .. math::

            \boldsymbol{F}_{int}^{elastic}(\boldsymbol{c}) = \int (\boldsymbol{B}_L + \boldsymbol{B}_{NL})^\top \boldsymbol{ABD}
            \boldsymbol{\varepsilon}(\boldsymbol{c}) dA

        with the generalized strains `\boldsymbol{\varepsilon}(\boldsymbol{c}) = (\boldsymbol{B}_L +
        \tfrac{1}{2} \boldsymbol{B}_{NL}) \boldsymbol{c}` of the kinematics of the model, the
        strain operator `\boldsymbol{B}_L + \boldsymbol{B}_{NL}`, `\boldsymbol{B}_{NL}` being linear in the
        rotations `\boldsymbol{\beta}` of the von Kármán terms, and the constitutive
        matrix `\boldsymbol{ABD}` of ``ABDnxny``, see :meth:`.Shell.calc_kTv`. With
        follower pressure loads the
        residual is

        .. math::
            \boldsymbol{R} = \boldsymbol{F}_{ext} - \boldsymbol{F}_{int}
                  = \boldsymbol{F}_{ext} + (\boldsymbol{F}_p(\boldsymbol{c}) - \boldsymbol{F}_p(0))
                    - \boldsymbol{F}_{int}^{elastic}(\boldsymbol{c})

        with :meth:`.Shell.calc_fext` at the same ``inc``, and the tangent
        stiffness matrix :meth:`.Shell.calc_kT` at ``inc`` is its exact
        Jacobian.

        """
        msg('Calculating internal forces...', level=2, silent=silent)
        model = self.model
        if not model in modelDB.db.keys():
            raise ValueError(
                    '{0} is not a valid model option'.format(model))
        matrices_num = modelDB.db[model].get('matrices_num')
        if matrices_num is None:
            raise ValueError('matrices_num not implemented for model {0}'.
                    format(model))
        calc_fint = getattr(matrices_num, 'calc_fint', None)
        if calc_fint is None:
            raise ValueError('calc_fint not implemented for model {0}'.
                    format(model))

        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()

        self._check_r()
        ABDnxny, nx, ny = self._get_ABDnxny(ABDnxny, nx, ny)

        #NOTE calc_fint takes a ``double [::1]``, see the note in
        #     Shell.calc_kC() on why c must be checked before being forwarded
        c = np.ascontiguousarray(c, dtype=DOUBLE)
        check_c(c, size)
        fint = np.asarray(calc_fint(c, ABDnxny, self, size, col0, nx, ny))
        if self.has_follower_loads():
            fint = fint - self.calc_fext_follower(c, inc=inc, size=size,
                                                  col0=col0, reference=False)

        msg('finished!', level=2, silent=silent)

        return fint


    def save(self, fname=None):
        r"""Save the ``Shell`` object to a zip file

        The inputs are stored in JSON and the arrays, such as ``ABD`` and the
        results in ``fields``, in NumPy's ``.npy`` format, see
        :mod:`panels.json_io`. The matrices in ``Shell.matrices`` are not
        stored. The object is not modified.

        Parameters
        ----------
        fname : str, path-like or file object, optional
            Name of the file, or a binary file object opened for writing,
            e.g. :class:`io.BytesIO`. By default the name stored in
            ``Shell.name`` followed by the extension ``'.shell.zip'``.

        """
        if fname is None:
            fname = self.name + '.shell.zip'
        if not hasattr(fname, 'write'):
            msg('Saving Shell to {}'.format(fname))
        json_io.save(self, fname)
