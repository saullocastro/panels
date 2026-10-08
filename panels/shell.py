import gc
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import eig
from numpy import linspace
from composites import laminated_plate
from structsolve.sparseutils import finalize_symmetric_matrix

from .logger import msg, warn
from . import modelDB
from . shell_fext import shell_fext, pressure_patches
from . import json_io

DOUBLE = np.float64


def load(name):
    r"""Load a :class:`.Shell` saved by :meth:`.Shell.save`

    Parameters
    ----------
    name : str, path-like or file object
        Name of the file, with or without the extension ``'.shell.zip'``,
        or a binary file object opened for reading, e.g.
        :class:`io.BytesIO`. The pickle files ``'.Shell'`` saved by older
        versions of panels are also loaded, with a ``DeprecationWarning``,
        see :mod:`panels.json_io`.

    Returns
    -------
    shell : :class:`.Shell`

    """
    return json_io._load_type(name, 'Shell', ('.shell.zip', '.Shell'))


def check_c(c, size):
    # Conducts a check on c
    if not isinstance(c, np.ndarray):
        raise TypeError('"c" must be a NumPy ndarray object')
    if c.ndim != 1:
        raise ValueError('"c" must be a 1-D ndarray object')
    if c.shape[0] != size:
        raise ValueError('"c" must have the same size as the global stiffness matrix')


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

    Parameters
    ----------
    a : float, optional
        Length (along the `x` coordinate).
    b : float, optional
        Width (along the `y` coordinate).
    r : float, optional
        Radius for cylindrical shell.
    stack : list or tuple, optional
        A sequence representing the angles for each ply.
    plyt : float, optional
        Ply thickness.
    laminaprop : list or tuple, optional
        Orthotropic lamina properties: `E_1, E_2, \nu_{12}, G_{12}, G_{13}, G_{23}`.
    rho : float, optional
        Material density.
    m, n : int, optional
        Number of terms for the approximation functions along `x` and `y`,
        respectively.
    offset : float, optional
        Laminate offset about shell mid-surface. The offset is measured along
        the normal (`z`) axis.

    Notes
    -----
    The boundary conditions are controlled by the flags ``x1u, x1ur, x2u,
    x2ur, ..., y2w, y2wr``, where for instance ``x1v`` multiplies the value
    and ``x1vr`` the derivative of the approximation functions of `v` at the
    edge `x = x_1`, with ``0`` removing and ``1`` keeping that degree of
    freedom. The models based on shear deformation theories, e.g.
    ``'plate_fsdt_donnell'`` or ``'cylshell_tsdt_sanders'``, have the
    rotations `\phi_x` and `\phi_y` as independent fields, which are
    controlled by the analogous flags ``x1phix, x1phixr, ..., y2phiy,
    y2phiyr``. Their default is the hard simply supported condition, with
    the tangential rotation removed at each edge (``x1phiy = x2phiy = 0`` and
    ``y1phix = y2phix = 0``), consistent with the default boundary
    conditions of the models based on the classical laminated plate theory.

    The attribute ``fsdt_shear_correction`` controls the transverse shear
    stiffness of the first-order shear deformation theory (FSDT):
    ``'rohwer'``, the default, ``'vlachoutsis'``, ``'constant'`` or ``None``
    select the corresponding method of
    :meth:`composites.Laminate.calc_transverse_shear_stiffness`, whereas a
    float ``k`` multiplies the uncorrected stiffness, ``A_ts = k*Abar_ts``.
    The equilibrium approach of Rohwer (1988) gives ``k = 5/6`` for a
    homogeneous plate and accounts for the stacking sequence otherwise. The
    third-order shear deformation theory (TSDT) needs no shear correction.

    The attributes ``x0, y0, z0, point_x, point_xy`` place the shell in the
    global coordinate system of the 3D plots, see :meth:`.Shell.global_frame`
    and :meth:`.MultiDomain.plot3d`.

    The attributes ``x1, x2, y1, y2`` limit the integration domain to
    ``x1 <= x <= x2`` and ``y1 <= y <= y2``, in physical coordinates, while
    the approximation functions still span the whole ``0 <= x <= a`` and
    ``0 <= y <= b``. This is how several domains, for instance the panels of
    a :class:`.StiffPanelBay`, share one set of approximation functions. A
    limit that is ``None``, the default, is the corresponding edge of the
    shell, see :meth:`.Shell.integration_limits`.

    """
    # Declare all the variables/attributes here to preallocate mem, speed it up. Var not declared here cant be used
    __slots__ = [ 'a', 'x1', 'x2', 'b', 'y1', 'y2', 'r',
        'stack', 'plyt', 'laminaprop', 'rho', 'offset',
        'group', 'x0', 'y0', 'z0', 'point_x', 'point_xy',
        'row_start', 'col_start', 'row_end', 'col_end',
        'name', 'bay', 'model',
        'fsdt_shear_correction',
        'm', 'n', 'nx', 'ny', 'size',
        'point_loads', 'point_loads_inc', 'distr_loads', 'distr_loads_inc',
        'pressure_loads', 'pressure_loads_inc',
        'point_pds', 'point_pds_inc', 'distr_pds', 'distr_pds_inc',
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
        'plyts', 'laminaprops', 'rhos',
        'flow', 'beta', 'gamma', 'aeromu', 'rho_air', 'speed_sound', 'Mach', 'air_speed',
        'ABD', 'force_orthotropic_laminate',
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

        # numerical integration - no of points
        self.nx = 2*m
        self.ny = 2*n

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
            # Stored as [x pos, y pos of applied displ, force x, y, z due to that displ]
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

        # aeroelastic parameters for panel flutter
        self.flow = 'x'
        self.beta = None
        self.gamma = None
        self.aeromu = None
        self.rho_air = None
        self.speed_sound = None
        self.Mach = None
        self.air_speed = None

        # constitutive law
        self.ABD = None
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


    def _rebuild(self):
        self.nx = max(self.nx, self.m)
        self.ny = max(self.ny, self.n)
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

        if not self.laminaprops:
            if not self.laminaprop:
                raise ValueError('laminaprop must be defined')
            self.laminaprops = [self.laminaprop for i in self.stack]

        if not self.rhos:
            self.rhos = [self.rho for i in self.stack]

        if not self.plyts:
            if self.plyt is None:
                raise ValueError('plyt must be defined')
            self.plyts = [self.plyt for i in self.stack]

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
            self.ABD = self._get_lam_ABD()
        self.size = self.get_size()
        # fails early on invalid limits of the integration domain
        self.integration_limits()


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

        Note that the flat-plate limit of a cylindrical shell is
        ``r -> infinity`` (``1/r -> 0``) and not ``r = 0``, hence
        ``r = 0.`` is never a valid radius for a cylindrical model. Use the
        ``'plate_clpt_donnell'`` model for flat panels.

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
            taken as the edge, to absorb round-off.

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

        """
        x1, x2, y1, y2 = self.integration_limits()
        return bool(x1 > 0 or x2 < self.a or y1 > 0 or y2 < self.b)


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

        """
        origin = np.array([0. if v is None else v for v in
                           (self.x0, self.y0, getattr(self, 'z0', None))],
                          dtype=DOUBLE)
        point_x = getattr(self, 'point_x', None)
        point_xy = getattr(self, 'point_xy', None)
        vec_x = (np.array([1., 0., 0.]) if point_x is None
                 else np.asarray(point_x, dtype=DOUBLE) - origin)
        vec_xy = (np.array([0., 1., 0.]) if point_xy is None
                  else np.asarray(point_xy, dtype=DOUBLE) - origin)
        if vec_x.shape != (3,) or vec_xy.shape != (3,):
            raise ValueError('point_x and point_xy must have three coordinates')
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
        on the local axes of the origin, such that `w` is positive outwards,
        consistent with `varepsilon_{yy} = v_{,y} + w/r`. Then, with `	heta
        = y/r`, the mid-surface is `x \hat{x} + r \sin 	heta \hat{y} + r
        (\cos 	heta - 1) \hat{z}`, the tangent along `y` is `\cos 	heta
        \hat{y} - \sin 	heta \hat{z}` and the normal is `\sin 	heta
        \hat{y} + \cos 	heta \hat{z}`.

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
        constants' vector `\{c\}`, the internal force vector `\{F_{int}\}` and
        the external force vector `\{F_{ext}\}`.

        ONLY RETURNS THE NUMBER OF ROWS ''OR'' COLS

        Returns
        -------
        size : int
            The size of the stiffness matrices. Can be the size of a global
            internal force vector of an assembly. When using a string, for
            example, if '+1' is given it will add 1 to the Shell`s size
            obtained by the :meth:`.Shell.get_size`

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


    def _get_lam_ABD(self, silent=False):
        r"""Constitutive matrix of the laminate, as required by the model

        - Classical laminated plate theory (CLPT), ``6 x 6``: the ``ABD``
          matrix.

        - First-order shear deformation theory (FSDT), ``8 x 8``: ``[[A, B,
          0], [B, D, 0], [0, 0, Ats]]``, where ``Ats`` is the shear corrected
          transverse shear stiffness of the `(yz, xz)` components, see the
          attribute ``fsdt_shear_correction``.

        - Third-order shear deformation theory (TSDT), ``13 x 13``: ``[[A, B,
          E, 0, 0], [B, D, F, 0, 0], [E, F, H, 0, 0], [0, 0, 0, Abar_ts,
          Dts], [0, 0, 0, Dts, Fts]]``, without shear correction.

        The rows and columns are in the order of the generalized strains of
        each model, see :meth:`.Shell.strain`.

        """
        if self.lam is None:
            raise RuntimeError('lam object is None!')
        lam = self.lam
        if 'clpt' in self.model:
            ABD = lam.ABD
        elif 'fsdt' in self.model:
            ABD = np.zeros((8, 8), dtype=DOUBLE)
            ABD[:6, :6] = lam.ABD
            k = self.fsdt_shear_correction
            if k is None or isinstance(k, str):
                ABD[6:, 6:] = lam.Ats
            else:
                ABD[6:, 6:] = k*lam.Abar_ts
        elif 'tsdt' in self.model:
            ABD = np.zeros((13, 13), dtype=DOUBLE)
            ABD[0:3, 0:3] = lam.A
            ABD[0:3, 3:6] = lam.B
            ABD[0:3, 6:9] = lam.E
            ABD[3:6, 0:3] = lam.B
            ABD[3:6, 3:6] = lam.D
            ABD[3:6, 6:9] = lam.F
            ABD[6:9, 0:3] = lam.E
            ABD[6:9, 3:6] = lam.F
            ABD[6:9, 6:9] = lam.H
            ABD[9:11, 9:11] = lam.Abar_ts
            ABD[9:11, 11:13] = lam.Dts
            ABD[11:13, 9:11] = lam.Dts
            ABD[11:13, 11:13] = lam.Fts

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
        stiffness matrix `[K_0]`, whether or not ``c`` is given, because the
        kernels zero `w_{,x}` and `w_{,y}` and every large displacement term
        is built from them. With ``NLgeom=True`` the result is
        `[K_0] + [K_{0L}] + [K_{L0}] + [K_{LL}] + [K_{G_{NL}}]`, evaluated at
        ``c``. What ``c`` alone changes is the integration: giving it selects
        the numerically integrated matrices over the analytical closed form.
        When using ``c`` its size must be the same as the attribute ``size``.

        Returns
        -------
        kC : csr_matrix
            The matrix described in the notes below. Also stored in
            ``Shell.matrices`` under the key ``'kC'``.

        Notes
        -----
        Despite the name, the returned matrix is not purely constitutive in
        two cases:

        - With ``NLgeom=True`` the returned matrix also contains
          `[K_{G_{NL}}]`, the geometric stiffness of the membrane stress
          carried by the non-linear part of the strain,
          `\{\varepsilon_{NL}\} = \{w_{,x}^2/2, w_{,y}^2/2, w_{,x} w_{,y}\}`.
          This term is deliberately collected here instead of in
          :meth:`.Shell.calc_kG`, so that

          .. math::
              [K_T] = [K_0] + [K_{0L}] + [K_{L0}] + [K_{LL}] + [K_{G_{NL}}]
                      + [K_G(N_0 + N_L)]

          with the first five terms returned by this method and the last one
          by :meth:`.Shell.calc_kG`, is the exact Jacobian of
          :meth:`.Shell.calc_fint`, while :meth:`.Shell.calc_kG` stays
          homogeneous of degree one in ``c``, as linear buckling requires.

        - With ``NLgeom=True`` and ``finalize=True`` the load stiffness of the
          follower pressure loads, :meth:`.Shell.calc_kCfollower` at ``c``
          and load factor ``inc``, is added, such that the sum of this matrix
          and :meth:`.Shell.calc_kG` is the exact Jacobian of
          :meth:`.Shell.calc_fint` at the same ``inc``. With
          ``finalize=False``, as in an assembly, it is not added, because the
          load stiffness is unsymmetric and the caller symmetrizes the
          assembled upper triangle; it must then be added after the
          finalization, see :meth:`.MultiDomain.calc_kC`.

        - If ``c_cte`` is given, or any of the attributes ``Nxx_cte``,
          ``Nyy_cte`` or ``Nxy_cte`` is non-zero, the geometric stiffness
          `[K_G(N_{cte})]` of that constant stress state is added into the
          returned matrix, which is then no longer `[K_0] + [K_{C_{NL}}]`.
          This is the device used to superpose combined load cases, where the
          eigenvalue of a linear buckling analysis must multiply only part of
          the applied loads. The constant stress state does not scale with
          that eigenvalue precisely because it is added here and not in
          :meth:`.Shell.calc_kG`.

        In multi-domain semi-analytical models the sparse matrices that are
        calculated may have the ``size`` of the assembled global model, and the
        current constitutive matrix being calculated starts at position
        ``row0`` and ``col0``.

        Parameters
        ----------
        size : int or str, optional
            The size of the calculated sparse matrices. When using a string,
            for example, if '+1' is given it will add 1 to the Shell`s size
            obtained by the :meth:`.Shell.get_size`
        row0, col0: int or None, optional
            Offset to populate the output sparse matrix (necessary when
            assemblying shells).
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        finalize : bool, optional
            Asserts validity of output data and makes the output matrix
            symmetric, should be ``False`` when assemblying.
        c : array-like or None, optional
            This must be the result of a static analysis, used to compute
            non-linear terms based on the actual displacement field.
        c_cte : array-like or None, optional
            This must be the result of a static analysis, used to compute
            initial stress state not affected by the load multiplier of the
            linear buckling eigenvalue analysis.
        nx, ny : int or None, optional
            Number of integration points along `x` and `y`, respectively, for
            the Legendre-Gauss quadrature rule applied in the numerical
            integration. Only used when ``c`` is given.
        ABDnxny : 4-D array-like or None, optional
            The constitutive relations for the laminate at each integration
            point. Must be a 4-D array of shape ``(nx, ny, 6, 6)`` when using
            classical laminated plate theory models.
        NLgeom : bool, optional
            Flag to indicate if geometrically non-linearities should be
            considered.
        inc : float, optional
            Load factor of the incremented follower pressure loads, used only
            with ``NLgeom=True``, see :meth:`.Shell.calc_kCfollower`.

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
        if ABDnxny is not None:
            analytical_kC = False
        # For NL Geos, KC (and not KC0) needs to be used so only do it numerically
        if NLgeom:
            analytical_kC = False
            analytical_kG = False

        matrices = modelDB.db[self.model]['matrices']  # selects what matrix functions to use
            # self.model = model of the current shell obj
        matrices_num = modelDB.db[self.model]['matrices_num']

        # Num integration points
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
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

        #NOTE the consistency checks for ABDnxny are done within the .pyx files
        ABDnxny = self.ABD if ABDnxny is None else ABDnxny

        # Calc Kc as per panels/panels/models then the pyx files given by ''matrices'' defined earlier
        # This calc K0 - linear consitutive stiff mat (SA formulation paper - eq 11)
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

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

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

        Returns
        -------
        kG : csr_matrix
            The geometric stiffness matrix. Also stored in ``Shell.matrices``
            under the key ``'kG'``.

        Notes
        -----
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

        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny

        if analytical_kG:
            kG = matrices.fkG0(self.Nxx, self.Nyy, self.Nxy, self, size, row0, col0)

        else:
            if ABDnxny is None:
                ABDnxny = self._get_lam_ABD()
            kG = matrices_num.fkG_num(c, ABDnxny, self, size, row0, col0,
                                      nx, ny, self.Nxx, self.Nyy, self.Nxy)

        if finalize:
            kG = finalize_symmetric_matrix(kG)
        self.matrices['kG'] = kG

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kG


    def calc_kT(self, size=None, row0=0, col0=0, silent=True, finalize=True,
            c=None, nx=None, ny=None, ABDnxny=None, inc=1.):
        r"""Calculate the tangent stiffness matrix `[K_T]`

        The tangent stiffness matrix is the exact Jacobian of
        :meth:`.Shell.calc_fint` with respect to ``c``, assembled as

        .. math::
            [K_T] = [K_0] + [K_{0L}] + [K_{L0}] + [K_{LL}] + [K_{G_{NL}}]
                    + [K_G(N_0 + N_L)]

        where all terms but the last come from
        :meth:`.Shell.calc_kC` with ``NLgeom=True``, and the last one from
        :meth:`.Shell.calc_kG` with ``NLgeom=True``. With follower pressure
        loads, the load stiffness :meth:`.Shell.calc_kCfollower` at the load
        factor ``inc`` is included, through :meth:`.Shell.calc_kC`, and the
        matrix is then unsymmetric in general. Note that `[K_{G_{NL}}]`
        is grouped with the constitutive matrix and not with the geometric
        one, see the notes in :meth:`.Shell.calc_kC` and
        :meth:`.Shell.calc_kG`.

        See :meth:`.Shell.calc_kC` for details on each parameter. Note that
        ``c_cte`` is not forwarded, so a constant stress state can only be
        given through the attributes ``Nxx_cte``, ``Nyy_cte`` and ``Nxy_cte``.

        Returns
        -------
        kT : csr_matrix
            The tangent stiffness matrix. Also stored in ``Shell.matrices``
            under the key ``'kT'``.

        """
        kC = self.calc_kC(size=size, row0=row0, col0=col0, silent=silent, finalize=finalize,
            c=c, nx=nx, ny=ny, ABDnxny=ABDnxny, NLgeom=True, inc=inc)
        kG = self.calc_kG(size=size, row0=row0, col0=col0, silent=silent, finalize=finalize,
            c=c, nx=nx, ny=ny, ABDnxny=ABDnxny, NLgeom=True)
        kT = kC + kG
        self.matrices['kT'] = kT

        return kT


    def has_follower_loads(self):
        r"""Tell whether any pressure load is a follower load, see
        :meth:`.Shell.add_pressure_load`"""
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
        have the force vector `\{F_p(c)\}`, which depends on the
        configuration, and the returned matrix is the stiffness contribution

        .. math::
            [K_{C_{follower}}] = - \frac{\partial \{F_p\}}{\partial \{c\}}

        evaluated at ``c``, with the pressures of the incremented loads
        (``cte=False``) multiplied by the load factor ``inc`` and those of the
        constant loads by ``1``. It enters the tangent stiffness matrix as
        `[K_T] = [K_C] + [K_G] + [K_{C_{follower}}]`, see
        :meth:`.Shell.calc_kT`, and the linear buckling problem of a follower
        pressure as ``lb(kC0, kG + kCfollower)``, with ``kG`` and
        ``kCfollower`` for a unit load factor.

        The matrix is unsymmetric in general, see the notes in
        ``theory/shells/follower_pressure/follower_pressure.py``: it is
        symmetric when the pressure is uniform over the whole integration
        domain and, on each edge, either `w` or the displacement normal to
        the edge is zero. It is therefore assembled in full and never
        symmetrized. For the first-order area vector (``follower='linear'``)
        it does not depend on ``c``.

        Parameters
        ----------
        c : array-like or None, optional
            The Ritz constants, with the size ``size``. Only used by the
            loads with ``follower='quadratic'``. ``None`` is the undeformed
            state.
        inc : float, optional
            Load factor of the incremented follower loads.
        size, row0, col0, silent :
            See :meth:`.Shell.calc_kC`.
        finalize : bool, optional
            Checks for ``nan`` and ``inf`` values. The matrix is never
            symmetrized.

        Returns
        -------
        kCfollower : csr_matrix
            Also stored in ``Shell.matrices`` under the key ``'kCfollower'``.

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
        r"""Force vector `\{F_p(c)\}` of the follower pressure loads

        The pressures of the incremented loads are multiplied by ``inc``. With
        ``reference=False`` the part of the undeformed state, `\{F_p(0)\}`,
        which :meth:`.Shell.calc_fext` already contains, is excluded. See
        :meth:`.Shell.add_pressure_load`.

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
            nx=None, ny=None, silent=True, finalize=True):
        r"""Calculate the mass matrix

        Parameters
        ----------
        h_nxny : (nx, ny) array-like or None, optional
            The constitutive relations for the laminate at each integration
            point.
        rho_nxny : (nx, ny) array-like or None, optional
            The material density for the laminate at each integration
            point. If multiple materials exist for the different plies,
            calculate ``rho`` using a weighted average.

        """
        msg('Calculating kM... ', level=2, silent=silent)

        analytical_kM = True
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny

        #NOTE see the note in Shell.calc_kC()
        if self.is_partial_domain():
            analytical_kM = False

        matrices = modelDB.db[self.model]['matrices']
        matrices_num = modelDB.db[self.model]['matrices_num']

        self._check_r()

        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()

        # calculate one h and rho for each integration point OR one for the
        # whole domain

        if h_nxny is None:
            h_nxny = np.zeros((nx, ny), dtype=DOUBLE)
            h_nxny[:, :] = self.lam.h
        else:
            analytical_kM = False
        if rho_nxny is None:
            rho_nxny = np.zeros((nx, ny), dtype=DOUBLE)
            #TODO change the whole code to handle the more general intrho,
            #     allowing different materials along the laminated plate
            rho_nxny[:, :] = self.lam.intrho/self.lam.h
        else:
            analytical_kM = False

        if analytical_kM:
            kM = matrices.fkM(self, self.offset, size, row0, col0)
        else:
            hrho_input = np.concatenate((h_nxny[..., None], rho_nxny[..., None]), axis=2)
            kM = matrices_num.fkM_num(self, self.offset, hrho_input, size, row0, col0, nx, ny)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.matrices['kM'] = kM

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kM


    def calc_kA(self, size=None, row0=0, col0=0, silent=True, finalize=True):
        r"""Calculate the aerodynamic matrix using the linear piston theory

        For a flow along `x`, the aerodynamic load on the shell along `w` is

        .. math::

            q = \beta w_{,x} + \gamma w

        and the aerodynamic matrix is `[K_A] = -\int \{N_w\}^T (\beta
        \{N_w\}_{,x} + \gamma \{N_w\}) dA`, entering the equations of motion
        as `([K] + [K_A] + \lambda^2 [M])\{c\} = \{0\}`, where `\{N_w\}` are
        the approximation functions of `w`. The flutter boundary does not
        depend on the sign of `\beta`, i.e. on the flow direction. When
        ``beta`` is not given it is calculated from the Mach number `M` as
        `\beta = \rho_{air} U^2/\sqrt{M^2 - 1}`.

        The term `\gamma w` is Krumhaar's correction of the piston theory for
        the external flow over a cylinder, with `w` positive outwards:

        .. math::

            \gamma = \frac{\beta}{2 r \sqrt{M^2 - 1}}

        such that an outward displacement lowers the pressure on the shell,
        which is pulled outwards, a softening effect that does not depend on
        the flow direction. It is obtained expanding the exact linear
        potential flow over a cylinder with a sinusoidal radial displacement
        for short wavelengths. Flat plates have `\gamma = 0`, and a ``gamma``
        given for a plate model is ignored. For a flow along `y` the
        curvature correction is not included.

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

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kA


    def calc_cA(self, aeromu, silent=True, size=None, finalize=True):
        r"""Calculate the aerodynamic damping matrix using the piston theory
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

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return cA


    def uvw(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the ``fields`` parameter of the
        :class:`.Shell`` object.

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        xs : np.ndarray
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        ys : np.ndarray
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int
            Number of points along the `y` where to calculate the
            displacement field.

        Returns
        -------
        out : tuple
            Containing ``plot_mesh`` and ``fields``


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

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the strain field
            calculation.
        xs : np.ndarray, optional
            The `x` coordinates where to calculate the strains.
        ys : np.ndarray, optional
            The `y` coordinates where to calculate the strains, must
            have the same shape as ``xs``.
        gridx : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used.
        gridy : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used.
        NLgeom : bool
            Flag to indicate whether non-linear strain components should be considered.

        Returns
        -------
        res : dict
            A dictionary of ``np.ndarrays`` with the keys:
            ``(x, y, exx, eyy, gxy, kxx, kyy, kxy)``. The models based on
            shear deformation theories also return the transverse shear
            strains ``(gyz, gxz)`` and, for the third-order theory, the
            higher-order terms ``(kxx3, kyy3, kxy3, gyz2, gxz2)``, such that
            the strains at a distance `z` from the mid-surface are
            ``exx + z*kxx + z**3*kxx3`` and ``gyz + z**2*gyz2``.

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

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the strain field
            calculation.
        ABD : np.ndarray, optional
            The laminate stiffness matrix. Can be a 6 x 6 (ABD) matrix for
            homogeneous laminates over the whole domain.
        xs : np.ndarray, optional
            The `x` coordinates where to calculate the strains.
        ys : np.ndarray, optional
            The `y` coordinates where to calculate the strains, must
            have the same shape as ``xs``.
        gridx : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used.
        gridy : int, optional
            When ``xs`` and ``ys`` are not supplied, ``gridx`` and ``gridy``
            are used.
        NLgeom : bool
            Flag to indicate whether non-linear strain components should be considered.

        Returns
        -------
        res : dict
            A dictionary of ``np.ndarrays`` with the keys:
            ``(x, y, Nxx, Nyy, Nxy, Mxx, Myy, Mxy)``. The models based on
            shear deformation theories also return the transverse shear
            forces ``(Qy, Qx)`` and, for the third-order theory, the
            higher-order resultants ``(Pxx, Pyy, Pxy, Ry, Rx)`` conjugate to
            ``(kxx3, kyy3, kxy3, gyz2, gxz2)``, see :meth:`.Shell.strain`.

        """
        plot_mesh, fields = self.strain(c, xs, ys, gridx, gridy, NLgeom)
        strains = [fields[name] for name in self._strain_names()]
        if ABD is None:
            ABD = self.ABD
        if ABD is None:
            raise ValueError('Laminate ABD matrix not defined for shell')
        #TODO implement for variable stiffness!

        self.plot_mesh = plot_mesh
        for i, name in enumerate(self._stress_names()):
            self.fields[name] = sum(e*ABD[i, j] for j, e in enumerate(strains))

        return self.plot_mesh, self.fields


    def add_point_load(self, x, y, fx, fy, fz, cte=True):
        r"""Add a point load with three components

        Parameters
        ----------
        x : float
            The `x` position.
        y : float
            The `y` position in radians.
        fx : float
            The `x` component of the force vector.
        fy : float
            The `y` component of the force vector.
        fz : float
            The `z` component of the force vector.
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis.

        """
        if cte:
            self.point_loads.append([x, y, fx, fy, fz])
        else:
            self.point_loads_inc.append([x, y, fx, fy, fz])


    def add_distr_load_fixed_x(self, x, funcx=None, funcy=None, funcz=None, cte=True):
        r"""Add a distributed force g(y) at a fixed x position

        Parameters
        ----------
        x : float
            The fixed `x` position.
        funcx, funcy, funcz : function, optional
            The functions of the distributed force components, will be used
            from `y=0` to `y=b`. At least one of the three must be defined
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis.

        """
        if not any((funcx, funcy, funcz)):
            raise ValueError('At least one function must be different than None')
        if cte:
            self.distr_loads.append([x, None, funcx, funcy, funcz])
        else:
            self.distr_loads_inc.append([x, None, funcx, funcy, funcz])


    def add_distr_load_fixed_y(self, y, funcx=None, funcy=None, funcz=None, cte=True):
        r"""Add a distributed force g(x) at a fixed y position

        Parameters
        ----------
        y : float
            The fixed `y` position.
        funcx, funcy, funcz : function, optional
            The functions of the distributed force components, will be used
            from `x=0` to `x=a`. At least one of the three must be defined
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis.

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
        mid-surface and acting on its deformed area, with the force vector

        .. math::
            \{F_p(c)\} = \int p \left(\{N_u\} a_x + \{N_v\} a_y
                         + \{N_w\} a_z\right) dx dy

        where `(a_x, a_y, a_z)` is the area vector of the deformed
        mid-surface per unit undeformed area. With ``follower=True`` or
        ``'linear'`` it is truncated to first order in the displacement
        gradients, consistently with the moderate-rotation strains of the
        models:

        .. math::
            a_x = -w_{,x}, \quad a_y = -w_{,y} + k_v v, \quad
            a_z = 1 + u_{,x} + v_{,y} + k_w w

        with `k_w = 1/r` for the cylinders and `k_v = 1/r` for the Sanders
        kinematics only, zero otherwise. With ``'quadratic'`` the complete
        (bilinear) area vector of the kinematics is used. See
        ``theory/shells/follower_pressure/follower_pressure.py``.

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
            assembly.
        cte : bool, optional
            Constant forces are not incremented during the non-linear
            analysis. Note that the non-linear solvers of ``structsolve``
            scale the whole :meth:`.Shell.calc_fext` by their load factor,
            so with them use ``cte=False`` for loads that follow the load
            factor.
        follower : bool or str, optional
            ``False`` for a dead load, ``True`` or ``'linear'`` for a follower
            pressure with the first-order area vector, ``'quadratic'`` for the
            complete area vector.
        zp : float, optional
            Position along `z` of the surface on which the pressure acts,
            e.g. ``h/2`` for the outer face of a cylinder. For the cylindrical
            shells the pressure is applied as ``p*(1 + zp/r)`` per unit area
            of mid-surface, which is the resultant of ``p`` acting on the
            radius ``r + zp``, with the direction and change of area of the
            mid-surface (shallow shell). Plates are not affected.

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


    def add_point_pd(self, x, y, ku, up, kv, vp, kw, wp, cte=True):
        r"""Add a point prescribed displacement with three components

        o/p = [location and equivalent force]

        Parameters
        ----------
        x : float
            The `x` position.
        y : float
            The `y` position in radians.
        ku, kv, kw : float
            The `x,y,z` component of the penalty stiffness of the prescribed displacement.
        up, vp, wp : float
            The `x,y,z` components of the prescribed displacement.
        cte : bool, optional
            Constant prescribed displacements are not incremented
            during the non-linear analysis.

        """
        if cte:
            self.point_pds.append([x, y, ku*up, kv*vp, kw*wp])
            # Adds the location and force
        else:
            self.point_pds_inc.append([x, y, ku*up, kv*vp, kw*wp])


    def add_distr_pd_fixed_x(self, x, ku=None, kv=None, kw=None,
                             funcu=None, funcv=None, funcw=None, cte=True):
        r"""Add a distributed prescribed displacement g(y) at a fixed x position

        Parameters
        ----------
        x : float
            The fixed `x` position.
        ku, kv, kw : float, optional
            The `x,y,z` components of the penalty stiffness of the prescribed
            displacement.  At least one of the three must be defined, and
            corresponding to the funcu, funcv, funcw specified.
        funcu, funcv, funcw : type: function, optional
            Specify in normal coordinates (x,y) not natural
            The functions of the distributed prescribed displacements, will be used
            from `y=0` to `y=b`. At least one of the three must be defined, and
            corresponding to the ku, kv, kw specified.
        cte : bool, optional
            Constant prescribed displacements are not incremented during the non-linear
            analysis.

        """
        if not any((ku, kv, kw)):
            raise ValueError('At least one penalty constant must be different than None')
        if not any((funcu, funcv, funcw)):
            raise ValueError('At least one function must be different than None')
        # Force funtns = k * displ ftn
        new_funcu = None
        new_funcv = None
        new_funcw = None
        if (ku is not None) or (funcu is not None): # ku or funcu is specified
            if ku is None or funcu is None: # if atmost 1 is specified for u means u is to be specified, but is currently incomplete
                raise ValueError('Both ku and funcu must be specified')
            new_funcu = lambda y: ku*funcu(y) # y is param in ftn
        if (kv is not None) or (funcv is not None):
            if kv is None or funcv is None:
                raise ValueError('Both kv and funcv must be specified')
            new_funcv = lambda y: kv*funcv(y)
        if (kw is not None) or (funcw is not None):
            if kw is None or funcw is None:
                raise ValueError('Both kw and funcw must be specified')
            new_funcw = lambda y: kw*funcw(y)
        if cte:
            self.distr_pds.append([x, None, new_funcu, new_funcv, new_funcw])
        else:
            self.distr_pds_inc.append([x, None, new_funcu, new_funcv, new_funcw])


    def add_distr_pd_fixed_y(self, y, ku=None, kv=None, kw=None,
                             funcu=None, funcv=None, funcw=None, cte=True):
        r"""Add a distributed prescribed displacement g(x) at a fixed y position

        Parameters
        ----------
        y : float
            The fixed `y` position.
        ku, kv, kw : float, optional
            The `x,y,z` components of the penalty stiffness of the prescribed
            displacement.  At least one of the three must be defined, and
            corresponding to the funcu, funcv, funcw specified.
        funcu, funcv, funcw : type: function, optional
            The functions of the distributed prescribed displacements, will be used
            from `y=0` to `y=b`. At least one of the three must be defined, and
            corresponding to the ku, kv, kw specified.
        cte : bool, optional
            Constant prescribed displacements are not incremented during the non-linear
            analysis.

        """
        if not any((ku, kv, kw)):
            raise ValueError('At least one penalty constant must be different than None')
        if not any((funcu, funcv, funcw)):
            raise ValueError('At least one function must be different than None')
        new_funcu = None
        new_funcv = None
        new_funcw = None
        if (ku is not None) or (funcu is not None):
            if ku is None or funcu is None:
                raise ValueError('Both ku and funcu must be specified')
            new_funcu = lambda x: ku*funcu(x)
        if (kv is not None) or (funcv is not None):
            if kv is None or funcv is None:
                raise ValueError('Both kv and funcv must be specified')
            new_funcv = lambda x: kv*funcv(x)
        if (kw is not None) or (funcw is not None):
            if kw is None or funcw is None:
                raise ValueError('Both kw and funcw must be specified')
            new_funcw = lambda x: kw*funcw(x)
        if cte:
            self.distr_pds.append([None, y, new_funcu, new_funcv, new_funcw])
        else:
            self.distr_pds_inc.append([None, y, new_funcu, new_funcv, new_funcw])

    def clear_disps(self):

        '''
            Used to clear exisiting displacements for the panels
                Useful for non-linear runs where displacements are added for each increment
        '''
        self.point_pds = []
        self.point_pds_inc = []
        self.distr_pds = []
        self.distr_pds_inc = []

    def clear_loads(self):
        '''
            Used to clear exisiting loads for the panels
                Useful for non-linear runs where loads are added for each increment
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

        Parameters
        ----------
        x, y : float
            Coordinates of the point contraint
        u, v, w : bool, optional
            Translational degrees of freedom to be constrained
        phix, phiy : bool, optional
            Rotational degrees of freedom to be constrained
        kuvw : float, optional
            Penalty constant used for the translational constraints
        kphi :
            Penalty constant used for the rotational constraints

        Returns
        -------
        kPC : csr_matrix
           Stiffness matrix concenrning this point constraint. It must be added
           to the constitutive stiffness matrix in order to be taken into
           account in the calculations.

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


    def calc_fext(self, inc=1., size=None, col0=0, silent=True, c=None):
        r"""Calculate the external force vector `\{F_{ext}\}`

        Recall that:

        .. math::

            \{F_{ext}\}=\{{F_{ext}}_0\} + \{{F_{ext}}_\lambda\}

        such that the terms in `\{{F_{ext}}_0\}` are constant and the terms in
        `\{{F_{ext}}_\lambda\}` will be scaled by the parameter ``inc``.

        See the documentation of :func:`.shell_fext` for more details. The
        follower pressure loads, see :meth:`.Shell.add_pressure_load`,
        contribute with their value in the undeformed state, unless ``c`` is
        given.

        Parameters
        ----------
        inc : float, optional
            Since this function is called during the non-linear analysis,
            ``inc`` will multiply the terms `\{{F_{ext}}_\lambda\}`.
        size : int or str, optional
            The size of the force vector. Can be the size of the total internal
            force vector of a multidomain assembly. When using a string, for
            example, if '+1' is given it will add 1 to the Shell`s size obtained
            by the :meth:`.Shell.get_size`
        col0 : int, optional
            Offset in a global force vector of an assembly.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        c : array-like or None, optional
            The Ritz constants, with the size ``size``. When given, the
            follower pressure loads are evaluated in this configuration,
            i.e. ``fext(c) = fext + (F_p(c) - F_p(0))``, the load vector that
            the arc-length solvers of ``structsolve`` use as the derivative
            of the residual with respect to the load factor.

        Returns
        -------
        fext : np.ndarray
            The external force vector

        """
        self._rebuild()
        msg('Calculating external forces...', level=2, silent=silent)
        fext = shell_fext(self, inc=inc, size=size, col0=col0)
        if c is not None and self.has_follower_loads():
            fext += self.calc_fext_follower(c, inc=inc, size=fext.shape[0],
                                            col0=col0, reference=False)
        return fext


    def calc_fint(self, c, size=None, col0=0, silent=True, nx=None,
            ny=None, ABDnxny=None, inc=1.):
        r"""Calculate the internal force vector `\{F_{int}\}`

        With follower pressure loads, see :meth:`.Shell.add_pressure_load`,
        the part of their force vector that depends on the configuration is
        subtracted, such that

        .. math::
            \{R\} = \{F_{ext}\} - \{F_{int}\}
                  = \{F_{ext}\} + (\{F_p(c)\} - \{F_p(0)\})
                    - \{F_{int}^{elastic}(c)\}

        with :meth:`.Shell.calc_fext` at the same ``inc``, and the tangent
        stiffness matrix :meth:`.Shell.calc_kT` at ``inc`` is its exact
        Jacobian.


        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the internal forces
            calculation.
        size : int or str, optional
            The size of the internal force vector. Can be the size of a global
            internal force vector of an assembly. When using a string,
            for example, if '+1' is given it will add 1 to the Shell`s size
            obtained by the :meth:`.Shell.get_size`
        col0 : int, optional
            Offset in a global internal force vector of an assembly.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        nx : int, optional
            Number of integration points along `x`.
        ny : int, optional
            Number of integration points along `y`.
        ABDnxny : np.ndarray, optional
            Laminate stiffness for each integration point, if not supplied it
            will assume constant properties over the shell domain.
        inc : float, optional
            Load factor of the incremented follower pressure loads.

        Returns
        -------
        fint : np.ndarray
            The internal force vector

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
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
        ABDnxny = self.ABD if ABDnxny is None else ABDnxny

        #NOTE calc_fint takes a ``double [::1]``, see the note in
        #     Shell.calc_kC() on why c must be checked before being forwarded
        c = np.ascontiguousarray(c, dtype=DOUBLE)
        check_c(c, size)
        fint = np.asarray(calc_fint(c, ABDnxny, self, size, col0, nx, ny))
        if self.has_follower_loads():
            fint = fint - self.calc_fext_follower(c, inc=inc, size=size,
                                                  col0=col0, reference=False)

        gc.collect()

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
