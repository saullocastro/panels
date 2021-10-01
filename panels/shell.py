from __future__ import division, absolute_import

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


def load(name):
    if '.Shell' in name:
        return pickle.load(open(name, 'rb'))
    else:
        return pickle.load(open(name + '.Shell', 'rb'))


def check_c(c, size):
    if not isinstance(c, np.ndarray):
        raise TypeError('"c" must be a NumPy ndarray object')
    if c.ndim != 1:
        raise ValueError('"c" must be a 1-D ndarray object')
    if c.shape[0] != size:
        raise ValueError('"c" must have the same size as the global stiffness matrix')


class Shell(object):
    r"""General shell class that can be used for plates or shells

    It works for both plates, cylindrical and conical shells. The right
    model is selected according to parameters ``r`` (radius) and ``alphadeg``
    (semi-vertex angle).

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
    alphadeg : float, optional
        Semi-vertex angle for conical shell.
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
    __slots__ = [ 'a', 'x1', 'x2', 'b', 'y1', 'y2', 'r', 'alphadeg', 'alpharad',
        'stack', 'plyt', 'laminaprop', 'rho', 'offset',
        'group', 'x0', 'y0', 'row_start', 'col_start', 'row_end', 'col_end',
        'name', 'model',
        'fsdt_shear_correction',
        'm', 'n', 'nx', 'ny', 'size',
        'point_loads', 'point_loads_inc', 'distr_loads', 'distr_loads_inc',
        'Nxx', 'Nyy', 'Nxy', 'Nxx_cte', 'Nyy_cte', 'Nxy_cte',
        'x1u', 'x2u', 'x1v', 'x2v',
        'x1w', 'x1wr', 'x2w', 'x2wr', 'y1u', 'y2u',
        'y1v', 'y2v', 'y1w', 'y1wr', 'y2w', 'y2wr',
        'plyts', 'laminaprops', 'rhos',
        'flow', 'beta', 'gamma', 'aeromu', 'rho_air', 'speed_sound', 'Mach', 'V',
        'F', 'force_orthotropic_laminate',
        'num_eigvalues', 'num_eigvalues_print',
        'out_num_cores', 'increments', 'results',
        'lam', 'matrices', 'fields', 'plot_mesh',
        ]

    def __init__(self, a=None, b=None, r=None, alphadeg=None,
            stack=None, plyt=None, laminaprop=None, rho=0,
            m=11, n=11, offset=0., **kwargs):
        self.a = a
        self.x1 = -1 # used to integrate part of the shell domain, -1 will use 0
        self.x2 = -1 # used to integrate part of the shell domain, -1 will se a
        self.b = b
        self.y1 = -1 # used to integrate part of the shell domain, -1 will use 0
        self.y2 = -1 # used to integrate part of the shell domain, -1 will use b
        self.r = r
        self.alphadeg = alphadeg
        self.alpharad = None
        self.stack = stack
        self.plyt = plyt
        self.laminaprop = laminaprop
        self.rho = rho
        self.offset = offset

        # assembly
        self.group = None
        self.x0 = None
        self.y0 = None
        self.row_start = None
        self.col_start = None
        self.row_end = None
        self.col_end = None

        self.name = 'shell'

        # model
        self.model = None
        self.fsdt_shear_correction = 5/6. # in case of First-order Shear Deformation Theory

        # approximation series
        self.m = m
        self.n = n
        self.size = None

        # numerical integration
        self.nx = 2*m
        self.ny = 2*n

        # loads
        self.point_loads = [] #NOTE see add_point_load
        self.point_loads_inc = [] #NOTE see add_point_load
        self.distr_loads = [] #NOTE see add_distr_load_fixed_x and add_distr_load_fixed_y
        self.distr_loads_inc = [] # NOTE see add_distr_load_fixed_x and add_distr_load_fixed_y
        # uniform membrane stress state
        self.Nxx = 0.
        self.Nyy = 0.
        self.Nxy = 0.
        # uniform constant membrane stress state (not multiplied by lambda)
        self.Nxx_cte = 0.
        self.Nyy_cte = 0.
        self.Nxy_cte = 0.

        #NOTE default boundary conditions:
        # - displacement at 4 edges is zero
        # - free to rotate at 4 edges (simply supported by default)
        self.x1u = 0.
        self.x2u = 0.
        self.x1v = 0.
        self.x2v = 0.
        self.x1w = 0.
        self.x1wr = 1.
        self.x2w = 0.
        self.x2wr = 1.

        self.y1u = 0.
        self.y2u = 0.
        self.y1v = 0.
        self.y2v = 0.
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
        self.F = None
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
            if self.r is None and self.alphadeg is None:
                self.model = 'plate_clpt_donnell_bardell'
            elif self.r is not None and self.alphadeg is None:
                self.model = 'cylshell_clpt_donnell_bardell'
            elif self.r is not None and self.alphadeg is not None:
                self.model = 'coneshell_clpt_donnell_bardell'

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
            self.F = self._get_lam_F()
        self.size = self.get_size()


    def get_size(self):
        r"""Calculate the size of the stiffness matrices

        The size of the stiffness matrices can be interpreted as the number of
        rows or columns, recalling that this will be the size of the Ritz
        constants' vector `\{c\}`, the internal force vector `\{F_{int}\}` and
        the external force vector `\{F_{ext}\}`.

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
        xs = np.atleast_1d(np.array(xs, dtype=np.float64))
        ys = np.atleast_1d(np.array(ys, dtype=np.float64))
        xshape = xs.shape
        yshape = ys.shape
        if xshape != yshape:
            raise ValueError('Arrays xs and ys must have the same shape')
        self.plot_mesh['Xs'] = xs
        self.plot_mesh['Ys'] = ys
        xs = np.ascontiguousarray(xs.ravel(), dtype=np.float64)
        ys = np.ascontiguousarray(ys.ravel(), dtype=np.float64)

        return xs, ys, xshape, yshape


    def _get_lam_F(self):
        if self.lam is None:
            raise RuntimeError('lam object is None!')
        if 'clpt' in self.model:
            F = self.lam.ABD
        elif 'fsdt' in self.model:
            F = self.lam.ABDE
            F[6:, 6:] *= self.fsdt_shear_correction

        if self.force_orthotropic_laminate:
            msg('')
            msg('Forcing orthotropic laminate...', level=2)
            F[0, 2] = 0. # A16
            F[1, 2] = 0. # A26
            F[2, 0] = 0. # A61
            F[2, 1] = 0. # A62

            F[0, 5] = 0. # B16
            F[5, 0] = 0. # B61
            F[1, 5] = 0. # B26
            F[5, 1] = 0. # B62

            F[3, 2] = 0. # B16
            F[2, 3] = 0. # B61
            F[4, 2] = 0. # B26
            F[2, 4] = 0. # B62

            F[3, 5] = 0. # D16
            F[4, 5] = 0. # D26
            F[5, 3] = 0. # D61
            F[5, 4] = 0. # D62

            if F.shape[0] == 8:
                F[6, 7] = 0. # A45
                F[7, 6] = 0. # A54

        return F


    def calc_kC(self, size=None, row0=0, col0=0, silent=True, finalize=True,
            c=None, c_cte=None, nx=None, ny=None, Fnxny=None,
            NLgeom=False):
        """Calculate the constitutive stiffness matrix

        If ``c`` is not given it calculates the linear constitutive stiffness
        matrix, otherwise the large displacement linear constitutive stiffness
        matrix is calculated. When using ``c`` the size of ``c`` must be the
        same as ``size``.

        In assemblies of semi-analytical models the sparse matrices that are
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
        Fnxny : 4-D array-like or None, optional
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
        if c is not None:
            check_c(c, size)

        alphadeg = self.alphadeg if self.alphadeg is not None else 0.
        self.alpharad = deg2rad(alphadeg)
        self.r = self.r if self.r is not None else 0.

        matrices_num = modelDB.db[self.model]['matrices_num']
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny

        #NOTE the consistence checks for Fnxny are done within the .pyx
        #     files
        Fnxny = self.F if Fnxny is None else Fnxny

        if c is None:
            # Empty c if the interest is only on the heterogeneous
            # laminate properties
            c = np.zeros(size, dtype=np.float64)
        c = np.ascontiguousarray(c, dtype=np.float64)

        kC = matrices_num.fkC_num(c, Fnxny, self,
                 size, row0, col0, nx, ny, NLgeom=int(NLgeom))

        if c_cte is not None or any((self.Nxx_cte, self.Nyy_cte, self.Nxy_cte)):
            if any((self.Nxx_cte, self.Nyy_cte, self.Nxy_cte)):
                msg('NOTE: constant stress state taken into account by (Nxx_cte, Nyy_cte, Nxy_cte)', level=3, silent=silent)
            if c_cte is not None:
                msg('NOTE: constant stress state taken into account by c_cte', level=3, silent=silent)
                check_c(c_cte, size)
            else:
                c_cte = 0 * c # creating a dummy c_cte
            NLgeom = int(NLgeom)
            kC += matrices_num.fkG_num(c_cte, Fnxny, self,
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
            c=None, nx=None, ny=None, Fnxny=None, NLgeom=False):
        """Calculate the geometric stiffness matrix

        See :meth:`.Shell.calc_kC` for details on each parameter.

        """
        msg('Calculating kG... ', level=2, silent=silent)
        self._rebuild()
        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()
        if c is not None:
            check_c(c, size)
            c = np.ascontiguousarray(c, dtype=np.float64)
            if any((self.Nxx, self.Nyy, self.Nxy)):
                msg('NOTE: stress state taken into account using ALSO (Nxx, Nyy, Nxy)', level=3, silent=silent)
        else:
            c = np.zeros(size, dtype=np.float64)
            if any((self.Nxx, self.Nyy, self.Nxy)):
                msg('NOTE: stress state taken into account using ONLY (Nxx, Nyy, Nxy)', level=3, silent=silent)

        matrices = modelDB.db[self.model]['matrices_num']

        alphadeg = self.alphadeg if self.alphadeg is not None else 0.
        self.alpharad = deg2rad(alphadeg)
        self.r = self.r if self.r is not None else 0.

        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
        if Fnxny is None:
            Fnxny = self._get_lam_F()
        NLgeom = int(NLgeom)
        kG = matrices.fkG_num(c, Fnxny, self,
                   size, row0, col0, nx, ny, NLgeom,
                   self.Nxx, self.Nyy, self.Nxy)

        if finalize:
            kG = finalize_symmetric_matrix(kG)
        self.matrices['kG'] = kG

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kG


    def calc_kT(self, size=None, row0=0, col0=0, silent=True, finalize=True,
            c=None, nx=None, ny=None, Fnxny=None):
        kC = self.calc_kC(size=size, row0=row0, col0=col0, silent=silent, finalize=finalize,
            c=c, nx=nx, ny=ny, Fnxny=Fnxny, NLgeom=True)
        kG = self.calc_kG(size=size, row0=row0, col0=col0, silent=silent, finalize=finalize,
            c=c, nx=nx, ny=ny, Fnxny=Fnxny, NLgeom=True)
        kT = kC + kG
        self.matrices['kT'] = kT

        return kT


    def calc_kM(self, size=None, row0=0, col0=0, h_nxny=None, rho_nxny=None,
            nx=None, ny=None, silent=True, finalize=True):
        """Calculate the mass matrix

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
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny

        matrices = modelDB.db[self.model]['matrices_num']

        alphadeg = self.alphadeg if self.alphadeg is not None else 0.
        self.alpharad = deg2rad(alphadeg)
        self.r = self.r if self.r is not None else 0.

        if size is None:
            size = self.get_size()
        elif isinstance(size, str):
            size = int(size) + self.get_size()

        # calculate one h and rho for each integration point OR one for the
        # whole domain

        if h_nxny is None:
            h_nxny = np.zeros((nx, ny), dtype=np.float64)
            h_nxny[:, :] = self.lam.h
        if rho_nxny is None:
            rho_nxny = np.zeros((nx, ny), dtype=np.float64)
            #TODO change the whole code to handle the more general intrho,
            #     allowing different materials along the laminated plate
            rho_nxny[:, :] = self.lam.intrho/self.lam.h

        hrho_input = np.concatenate((h_nxny[..., None], rho_nxny[..., None]), axis=2)
        kM = matrices.fkM_num(self, self.offset, hrho_input, size, row0, col0, nx, ny)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.matrices['kM'] = kM

        #NOTE memory cleanup
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kM


    def calc_kA(self, size=None, row0=0, col0=0, silent=True, finalize=True):
        """Calculate the aerodynamic matrix using the linear piston theory
        """
        msg('Calculating kA... ', level=2, silent=silent)

        if 'coneshell' in self.model:
            raise NotImplementedError('Conical shells not supported')

        matrices = modelDB.db[self.model]['matrices_num']

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
            kA = matrices.fkAx_num(self, size, row0, col0, self.nx, self.ny)
        elif self.flow.lower() == 'y':
            kA = matrices.fkAy_num(self, size, row0, col0, self.nx, self.ny)
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
        """Calculate the aerodynamic damping matrix using the piston theory
        """
        msg('Calculating cA... ', level=2, silent=silent)

        if size is None:
            size = self.get_size()
        matrices = modelDB.db[self.model]['matrices_num']
        cA = matrices.fcA(aeromu, self, size, 0, 0)
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
        c = np.ascontiguousarray(c, dtype=np.float64)

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
        c = np.ascontiguousarray(c, dtype=np.float64)
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


    def stress(self, c, F=None, xs=None, ys=None, gridx=300, gridy=300, NLgeom=True):
        r"""Calculate the stress field

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants vector to be used for the strain field
            calculation.
        F : np.ndarray, optional
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
        if F is None:
            F = self.F
        if F is None:
            raise ValueError('Laminate ABD matrix not defined for shell')
        #TODO implement for variable stiffness!

        self.plot_mesh = plot_mesh
        self.fields['Nxx'] = exx*F[0, 0] + eyy*F[0, 1] + gxy*F[0, 2] + kxx*F[0, 3] + kyy*F[0, 4] + kxy*F[0, 5]
        self.fields['Nyy'] = exx*F[1, 0] + eyy*F[1, 1] + gxy*F[1, 2] + kxx*F[1, 3] + kyy*F[1, 4] + kxy*F[1, 5]
        self.fields['Nxy'] = exx*F[2, 0] + eyy*F[2, 1] + gxy*F[2, 2] + kxx*F[2, 3] + kyy*F[2, 4] + kxy*F[2, 5]
        self.fields['Mxx'] = exx*F[3, 0] + eyy*F[3, 1] + gxy*F[3, 2] + kxx*F[3, 3] + kyy*F[3, 4] + kxy*F[3, 5]
        self.fields['Myy'] = exx*F[4, 0] + eyy*F[4, 1] + gxy*F[4, 2] + kxx*F[4, 3] + kyy*F[4, 4] + kxy*F[4, 5]
        self.fields['Mxy'] = exx*F[5, 0] + eyy*F[5, 1] + gxy*F[5, 2] + kxx*F[5, 3] + kyy*F[5, 4] + kxy*F[5, 5]

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
        g = np.zeros((5, size), dtype=np.float64)
        fg(g, x, y, self)
        gu, gv, gw, gphix, gphiy = g
        kPC = csr_matrix((size, size), dtype=np.float64)
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
        """Calculate the external force vector `\{F_{ext}\}`

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
            ny=None, Fnxny=None):
        """Calculate the internal force vector `\{F_{int}\}`


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
        Fnxny : np.ndarray, optional
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

        alphadeg = self.alphadeg if self.alphadeg is not None else 0.
        self.alpharad = deg2rad(alphadeg)
        self.r = self.r if self.r is not None else 0.
        nx = self.nx if nx is None else nx
        ny = self.ny if ny is None else ny
        Fnxny = self.F if Fnxny is None else Fnxny

        c = np.ascontiguousarray(c, dtype=np.float64)
        fint = calc_fint(c, Fnxny, self, size, col0, nx, ny)

        gc.collect()

        msg('finished!', level=2, silent=silent)

        return fint


    def save(self):
        """Save the ``Shell`` object using ``pickle``

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

