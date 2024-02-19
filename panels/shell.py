import gc
import pickle
from multiprocessing import cpu_count

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig
from numpy import linspace, deg2rad
from composites import laminated_plate
from structsolve.sparseutils import remove_null_cols, make_skew_symmetric, finalize_symmetric_matrix

from .logger import msg, warn
from . import modelDB
from . shell_fext import shell_fext

DOUBLE = np.float64


def load(name):
    if '.Shell' in name:
        return pickle.load(open(name, 'rb'))
    else:
        return pickle.load(open(name + '.Shell', 'rb'))


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

    It works for both plates and cylindrical shells. The right model is selected
    according to parameter ``r`` (radius).

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

    """
    # Declare all the variables/attributes here to preallocate mem, speed it up. Var not declared here cant be used
    __slots__ = [ 'a', 'x1', 'x2', 'b', 'y1', 'y2', 'r',
        'stack', 'plyt', 'laminaprop', 'rho', 'offset',
        'group', 'x0', 'y0', 'row_start', 'col_start', 'row_end', 'col_end',
        'name', 'bay', 'model',
        'fsdt_shear_correction',
        'm', 'n', 'nx', 'ny', 'size',
        'point_loads', 'point_loads_inc', 'distr_loads', 'distr_loads_inc',
        'point_pds', 'point_pds_inc', 'distr_pds', 'distr_pds_inc',
        'Nxx', 'Nyy', 'Nxy', 'Nxx_cte', 'Nyy_cte', 'Nxy_cte',
        'x1u', 'x1ur', 'x2u', 'x2ur',
        'x1v', 'x1vr', 'x2v', 'x2vr',
        'x1w', 'x1wr', 'x2w', 'x2wr',
        'y1u', 'y1ur', 'y2u', 'y2ur',
        'y1v', 'y1vr', 'y2v', 'y2vr',
        'y1w', 'y1wr', 'y2w', 'y2wr',
        'plyts', 'laminaprops', 'rhos',
        'flow', 'beta', 'gamma', 'aeromu', 'rho_air', 'speed_sound', 'Mach', 'V',
        'ABD', 'force_orthotropic_laminate',
        'num_eigvalues', 'num_eigvalues_print',
        'out_num_cores', 'increments', 'results',
        'lam', 'matrices', 'fields', 'plot_mesh',
        ]

    def __init__(self, a=None, b=None, r=None,
            stack=None, plyt=None, laminaprop=None, rho=0,
            m=11, n=11, offset=0., **kwargs):
        self.a = a
        self.x1 = -1 # used to integrate part of the shell domain, -1 will use 0
        self.x2 = +1 # used to integrate part of the shell domain, +1 will use a
        self.b = b
        self.y1 = -1 # used to integrate part of the shell domain, -1 will use 0
        self.y2 = +1 # used to integrate part of the shell domain, +1 will use b
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
        self.row_start = None
        self.col_start = None
        self.row_end = None
        self.col_end = None

        self.name = 'shell'
        self.bay = None

        # model
        self.model = None
        self.fsdt_shear_correction = 5/6. # in case of First-order Shear Deformation Theory

        # approximation series - no of terms in SFs
        self.m = m
        self.n = n
        self.size = None

        # numerical integration - no of points
        self.nx = 2*m
        self.ny = 2*n

        # loads
        self.point_loads = [] #NOTE see add_point_load
        self.point_loads_inc = [] #NOTE see add_point_load
        self.distr_loads = [] #NOTE see add_distr_load_fixed_x and add_distr_load_fixed_y
        self.distr_loads_inc = [] # NOTE see add_distr_load_fixed_x and add_distr_load_fixed_y
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
        self.V = None

        # constitutive law
        self.ABD = None
        self.force_orthotropic_laminate = False

        # eigenvalue analysis
        self.num_eigvalues = 5
        self.num_eigvalues_print = 5

        # output queries
        self.out_num_cores = cpu_count()

        # outputs
        self.increments = None
        self.results = dict(eigvecs=None, eigvals=None)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self._clear_matrices()


    def _clear_matrices(self):
        self.lam = None
        self.matrices = dict(kC=None, kG=None, kT=None, kM=None, kA=None, cA=None)
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
                self.model = 'plate_clpt_donnell_bardell'
            elif self.r is not None:
                self.model = 'cylshell_clpt_donnell_bardell'

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
            lam = laminated_plate(stack=self.stack, plyts=self.plyts,
                                      laminaprops=self.laminaprops,
                                      rhos=self.rhos,
                                      offset=self.offset)
            self.lam = lam
            self.ABD = self._get_lam_ABD()
        self.size = self.get_size()


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
            obtained by the :method:`.Shell.get_size`

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
        if self.lam is None:
            raise RuntimeError('lam object is None!')
        if 'clpt' in self.model:
            ABD = self.lam.ABD
        elif 'fsdt' in self.model:
            ABD = self.lam.ABDE
            ABD[6:, 6:] *= self.fsdt_shear_correction

        if self.force_orthotropic_laminate:
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
            c=None, c_cte=None, nx=None, ny=None, ABDnxny=None, NLgeom=False):
        r"""Calculate the constitutive stiffness matrix
        ---------- Notation as per MD paper: kP_i ----------- 

        If ``c`` is not given it calculates the linear constitutive stiffness
        matrix, otherwise the large displacement linear constitutive stiffness
        matrix is calculated. When using ``c`` the size of ``c`` must be the
        same as the attribute ``size``.

        In multi-domain semi-analytical models the sparse matrices that are
        calculated may have the ``size`` of the assembled global model, and the
        current constitutive matrix being calculated starts at position
        ``row0`` and ``col0``.

        Parameters
        ----------
        size : int or str, optional
            The size of the calculated sparse matrices. When using a string,
            for example, if '+1' is given it will add 1 to the Shell`s size
            obtained by the :method:`.Shell.get_size`
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
            initial stress state not affected by the load multiplier.
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

        """
        self._rebuild()
        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()
        msg('Calculating kC... ', level=2, silent=silent)

        analytical_kC = True
        analytical_kG = True
        # This means a linear analysis is already performed (check panels\tests\tests_shell\test_nonlinear.py)
        # So, the next step is NL. So no analytical
        if c is not None:
            check_c(c, size)
            analytical_kC = False
        # ??????????
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
        self.r = self.r if self.r is not None else 0.

        if c is None and ABDnxny is None:
            # Empty c if the interest is only on the heterogeneous
            # laminate properties
            c = np.zeros(size, dtype=DOUBLE)
        c = np.ascontiguousarray(c, dtype=DOUBLE)
        # returns a contiguous array, how matrices in C are stored. 1 after the other like matlab

        #NOTE the consistency checks for ABDnxny are done within the .pyx
        #     files
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
                analytical_kG = False
            # This calc KG0 - Geo stiff mat at initial membrane stress state (SA formulation paper - eq 12) and adds it to K0 calc earlier
            
            # WHY IS KG0 ADDED TO KC ???????????????????????????
            if analytical_kG:
                kC += matrices.fkG0(self.Nxx_cte, self.Nyy_cte, self.Nxy_cte, self, size, row0, col0)
            else:
                kC += matrices_num.fkG_num(c_cte, ABDnxny, self,
                        size, row0, col0, nx, ny, NLgeom,
                        self.Nxx_cte, self.Nyy_cte, self.Nxy_cte)

        if finalize:
            kC = finalize_symmetric_matrix(kC)
        self.matrices['kC'] = kC

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kC


    def calc_kG(self, size=None, row0=0, col0=0, silent=True, finalize=True,
            c=None, nx=None, ny=None, ABDnxny=None, NLgeom=False):
        r"""Calculate the (inital stress or) geometric stiffness matrix
        ---------- Notation as per MD paper: kGp_i ----------- 
        
        See :meth:`.Shell.calc_kC` for details on each parameter.

        """
        msg('Calculating kG... ', level=2, silent=silent)
        self._rebuild()
        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()
        analytical_kG = True
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

        self.r = self.r if self.r is not None else 0.

        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny

        if analytical_kG:
            kG = matrices.fkG0(self.Nxx, self.Nyy, self.Nxy, self, size, row0, col0)
        else:
            if ABDnxny is None:
                ABDnxny = self._get_lam_ABD()
            kG = matrices_num.fkG_num(c, ABDnxny, self, size, row0, col0,
                                      nx, ny, int(NLgeom), self.Nxx, self.Nyy, self.Nxy)

        if finalize:
            kG = finalize_symmetric_matrix(kG)
        self.matrices['kG'] = kG

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kG


    def calc_kT(self, size=None, row0=0, col0=0, silent=True, finalize=True,
            c=None, nx=None, ny=None, ABDnxny=None):
        kC = self.calc_kC(size=size, row0=row0, col0=col0, silent=silent, finalize=finalize,
            c=c, nx=nx, ny=ny, ABDnxny=ABDnxny, NLgeom=True)
        kG = self.calc_kG(size=size, row0=row0, col0=col0, silent=silent, finalize=finalize,
            c=c, nx=nx, ny=ny, ABDnxny=ABDnxny, NLgeom=True)
        kT = kC + kG
        self.matrices['kT'] = kT

        return kT


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

        matrices = modelDB.db[self.model]['matrices']
        matrices_num = modelDB.db[self.model]['matrices_num']

        self.r = self.r if self.r is not None else 0.

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
        """
        msg('Calculating kA... ', level=2, silent=silent)


        matrices_num = modelDB.db[self.model]['matrices_num']

        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()

        self.r = self.r if self.r is not None else 0.

        if self.beta is None:
            if self.Mach is None:
                raise ValueError('Mach number cannot be a NoneValue')
            elif self.Mach < 1:
                raise ValueError('Mach number must be >= 1')
            elif self.Mach == 1:
                self.Mach = 1.0001
            Mach = self.Mach
            beta = self.rho_air * self.V**2 / (Mach**2 - 1)**0.5
            if self.r != 0.:
                gamma = beta*1./(2.*self.r*(Mach**2 - 1)**0.5)
            else:
                gamma = 0.
        else:
            beta = self.beta
            gamma = self.gamma if self.gamma is not None else 0.

        self.beta = beta
        self.gamma = gamma

        if self.flow.lower() == 'x':
            kA = matrices_num.fkAx_num(self, size, row0, col0, self.nx, self.ny)
        elif self.flow.lower() == 'y':
            kA = matrices_num.fkAy_num(self, size, row0, col0, self.nx, self.ny)
        else:
            raise ValueError('Invalid flow value, must be x or y')

        if finalize:
            assert np.any(np.isnan(kA.data)) == False
            assert np.any(np.isinf(kA.data)) == False
            kA = csr_matrix(make_skew_symmetric(kA))
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
        matrices_num = modelDB.db[self.model]['matrices_num']
        cA = matrices_num.fcA(aeromu, self, size, 0, 0)
        cA = cA*(0+1j)

        if finalize:
            cA = finalize_symmetric_matrix(cA)
        self.matrices['cA'] = cA

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)


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
        c = np.ascontiguousarray(c, dtype=DOUBLE)

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
            ``(x, y, exx, eyy, gxy, kxx, kyy, kxy)``

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)
        xs, ys, xshape, yshape = self._default_field(xs, ys, gridx, gridy)
        fstrain = modelDB.db[self.model]['field'].fstrain
        exx, eyy, gxy, kxx, kyy, kxy = fstrain(c, self, xs, ys, self.out_num_cores, int(NLgeom))

        self.plot_mesh['Xs'] = xs.reshape(xshape)
        self.plot_mesh['Ys'] = ys.reshape(yshape)
        self.fields['exx'] = exx.reshape(xshape)
        self.fields['eyy'] = eyy.reshape(xshape)
        self.fields['gxy'] = gxy.reshape(xshape)
        self.fields['kxx'] = kxx.reshape(xshape)
        self.fields['kyy'] = kyy.reshape(xshape)
        self.fields['kxy'] = kxy.reshape(xshape)

        return self.plot_mesh, self.fields


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
            ``(x, y, Nxx, Nyy, Nxy, Mxx, Myy, Mxy)``

        """
        plot_mesh, fields = self.strain(c, xs, ys, gridx, gridy, NLgeom)
        exx = fields['exx']
        eyy = fields['eyy']
        gxy = fields['gxy']
        kxx = fields['kxx']
        kyy = fields['kyy']
        kxy = fields['kxy']
        if ABD is None:
            ABD = self.ABD
        if ABD is None:
            raise ValueError('Laminate ABD matrix not defined for shell')
        #TODO implement for variable stiffness!

        self.plot_mesh = plot_mesh
        self.fields['Nxx'] = exx*ABD[0, 0] + eyy*ABD[0, 1] + gxy*ABD[0, 2] + kxx*ABD[0, 3] + kyy*ABD[0, 4] + kxy*ABD[0, 5]
        self.fields['Nyy'] = exx*ABD[1, 0] + eyy*ABD[1, 1] + gxy*ABD[1, 2] + kxx*ABD[1, 3] + kyy*ABD[1, 4] + kxy*ABD[1, 5]
        self.fields['Nxy'] = exx*ABD[2, 0] + eyy*ABD[2, 1] + gxy*ABD[2, 2] + kxx*ABD[2, 3] + kyy*ABD[2, 4] + kxy*ABD[2, 5]
        self.fields['Mxx'] = exx*ABD[3, 0] + eyy*ABD[3, 1] + gxy*ABD[3, 2] + kxx*ABD[3, 3] + kyy*ABD[3, 4] + kxy*ABD[3, 5]
        self.fields['Myy'] = exx*ABD[4, 0] + eyy*ABD[4, 1] + gxy*ABD[4, 2] + kxx*ABD[4, 3] + kyy*ABD[4, 4] + kxy*ABD[4, 5]
        self.fields['Mxy'] = exx*ABD[5, 0] + eyy*ABD[5, 1] + gxy*ABD[5, 2] + kxx*ABD[5, 3] + kyy*ABD[5, 4] + kxy*ABD[5, 5]

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
        r"""Add a distributed prescribed displacement g(y) ??? at a fixed x position

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


    def calc_fext(self, inc=1., size=None, col0=0, silent=True):
        r"""Calculate the external force vector `\{F_{ext}\}`

        Recall that:

        .. math::

            \{F_{ext}\}=\{{F_{ext}}_0\} + \{{F_{ext}}_\lambda\}

        such that the terms in `\{{F_{ext}}_0\}` are constant and the terms in
        `\{{F_{ext}}_\lambda\}` will be scaled by the parameter ``inc``.

        Parameters
        ----------
        inc : float, optional
            Since this function is called during the non-linear analysis,
            ``inc`` will multiply the terms `\{{F_{ext}}_\lambda\}`.

        size : int or str, optional
            The size of the force vector. Can be the size of a global internal
            force vector of an assembly. When using a string, for example, if
            '+1' is given it will add 1 to the Shell`s size obtained by the
            :method:`.Shell.get_size`

        col0 : int, optional
            Offset in a global forcce vector of an assembly.

        silent : bool, optional
            A boolean to tell whether the log messages should be printed.

        Returns
        -------
        fext : np.ndarray
            The external force vector

        """
        self._rebuild()
        msg('Calculating external forces...', level=2, silent=silent)
        return shell_fext(self, inc=inc, size=size, col0=col0)


    def calc_fint(self, c, size=None, col0=0, silent=True, nx=None,
            ny=None, ABDnxny=None):
        r"""Calculate the internal force vector `\{F_{int}\}`


        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the internal forces
            calculation.
        size : int or str, optional
            The size of the internal force vector. Can be the size of a global
            internal force vector of an assembly. When using a string,
            for example, if '+1' is given it will add 1 to the Shell`s size
            obtained by the :method:`.Shell.get_size`
        col0 : int, optional
            Offset in a global internal forcce vector of an assembly.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        nx : int, optional
            Number of integration points along `x`.
        ny : int, optional
            Number of integration points along `y`.
        ABDnxny : np.ndarray, optional
            Laminate stiffness for each integration point, if not supplied it
            will assume constant properties over the shell domain.

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

        self.r = self.r if self.r is not None else 0.
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
        ABDnxny = self.ABD if ABDnxny is None else ABDnxny

        c = np.ascontiguousarray(c, dtype=DOUBLE)
        fint = calc_fint(c, ABDnxny, self, size, col0, nx, ny)

        gc.collect()

        msg('finished!', level=2, silent=silent)

        return fint


    def save(self):
        r"""Save the ``Shell`` object using ``pickle``

        Notes
        -----
        The pickled file will have the name stored in ``Shell.name``
        followed by a ``'.Shell'`` extension.

        """
        name = self.name + '.Shell'
        msg('Saving Shell to {}'.format(name))

        self._clear_matrices()

        with open(name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

