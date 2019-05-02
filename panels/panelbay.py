r"""
==============================================================================
Stiffened Panel Bay (:mod:`panels.panelbay`)
==============================================================================

.. currentmodule:: panels.panelbay

Main features:

- possibility to use many panels with different properties. In such case the
  panels are separated by their `y` (circumferential) coordinate. Usually
  there is a stiffener positioned at the `y` coordinate between two panels.
- possibility to use stiffeners (blade) modeled with 1D or 2D formulation.

.. autoclass:: PanelBay
    :members:

"""
from __future__ import division, absolute_import

import platform
import gc
import os
import sys
import traceback
from collections import Iterable
import time
import pickle
from multiprocessing import cpu_count
from copy import deepcopy

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eig, eigh
from numpy import linspace
from structsolve.sparseutils import finalize_symmetric_matrix
import composites.laminate as laminate

from . logger import msg, warn
from . shell import Shell
from . import modelDB as shellmDB
from . stiffener import BladeStiff1D, BladeStiff2D


def load(name):
    if '.PanelBay' in name:
        return pickle.load(open(name, 'rb'))
    else:
        return pickle.load(open(name + '.PanelBay', 'rb'))


class PanelBay(object):
    r"""Stiffened Shell Bay

    Can be used for supersonic Aeroelastic studies with the Piston Theory.

    Stiffeners are modeled either with 1D or 2D formulations.

    Main characteristics:

    - Supports both airflows along x (axis) or y (circumferential).
      Controlled by the parameter ``flow``
    - ``bladestiff1ds`` contains the :class:`.BladeStiff1D` stiffeners
    - ``bladestiff2ds`` contains the :class:`.BladeStiff2D` stiffeners

    """
    def __init__(self):
        self.name = ''

        # boundary conditions
        # "inf" is used to define the high stiffnesses (removed dofs)
        #       a high value will cause numerical instabilities
        #TODO use a marker number for self.inf and self.maxinf if the
        #     normalization of edge stiffnesses is adopted
        #     now it is already independent of self.inf and more robust
        self.point_loads_skin = []
        self.flow = 'x'
        self.bc = None
        self.model = None
        self.stack = []
        self.laminaprop = None
        self.laminaprops = []
        self.plyt = None
        self.plyts = []
        self.rho = None

        # approximation series
        self.m = 11
        self.n = 12

        # shells
        self.shells = []

        # stiffeners
        self.stiffeners = []
        self.bladestiff1ds = []
        self.bladestiff2ds = []

        # geometry
        self.a = None
        self.b = None
        self.r = None
        self.alphadeg = None

        # boundary conditions
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

        # aerodynamic properties for the Piston theory
        self.beta = None
        self.gamma = None
        self.aeromu = None
        self.rho_air = None
        self.speed_sound = None
        self.Mach = None
        self.V = None

        # output queries
        self.out_num_cores = cpu_count()

        self._clear_matrices()


    def _clear_matrices(self):
        self.matrices = dict(kC=None, kT=None, kM=None, kA=None, cA=None)
        self.fields = dict(u=None, v=None, w=None, phix=None, phiy=None)
        self.plot_mesh = dict(Xs=None, Ys=None)

        for shell in self.shells:
            shell.matrices['kC'] = None
            shell.matrices['kG'] = None
            shell.matrices['kM'] = None

        for s in self.bladestiff1ds:
            s.matrices['kC'] = None
            s.matrices['kG'] = None
            s.matrices['kM'] = None

        for s in self.bladestiff2ds:
            s.matrices['kC'] = None
            s.matrices['kG'] = None
            s.matrices['kM'] = None

        gc.collect()


    def _rebuild(self):
        if self.a is None:
            raise ValueError('The length a must be specified')

        if self.b is None:
            raise ValueError('The width b must be specified')

        for s in self.shells:
            s._rebuild()
            if self.model is not None:
                assert self.model == s.model
            else:
                self.model = s.model

        for s in self.bladestiff1ds:
            s._rebuild()

        for s in self.bladestiff2ds:
            s._rebuild()


    def _default_field(self, xs, a, ys, b, gridx, gridy):
        if xs is None or ys is None:
            xs = linspace(0., a, gridx)
            ys = linspace(0., b, gridy)
            xs, ys = np.meshgrid(xs, ys, copy=False)
        xs = np.atleast_1d(np.array(xs, dtype=np.float64))
        ys = np.atleast_1d(np.array(ys, dtype=np.float64))
        xshape = xs.shape
        yshape = ys.shape
        if xshape != yshape:
            raise ValueError('Arrays xs and ys must have the same shape')
        self.plot_mesh['Xs'] = xs
        self.plot_mesh['Ys'] = ys
        xs = xs.ravel()
        ys = ys.ravel()

        return xs, ys, xshape, yshape


    def get_size(self):
        r"""Calculate the size of the stiffness matrices

        The size of the stiffness matrices can be interpreted as the number of
        rows or columns, recalling that this will be the size of the Ritz
        constants' vector `\{c\}`, the internal force vector `\{F_{int}\}` and
        the external force vector `\{F_{ext}\}`.

        It takes into account the independent degrees of freedom from each of
        the `.Stiffener2D` objects that belong to the current assembly.

        Returns
        -------
        size : int
            The size of the stiffness matrices.

        """
        num = shellmDB.db[self.model]['num']
        self.size = num*self.m*self.n

        for s in self.bladestiff2ds:
            self.size += s.flange.get_size()

        return self.size


    def add_bladestiff1d(self, ys, rho=None, bb=None, bstack=None,
            bplyts=None, bplyt=None, blaminaprops=None, blaminaprop=None,
            bf=None, fstack=None, fplyts=None, fplyt=None, flaminaprops=None,
            flaminaprop=None, **kwargs):
        """Add a new BladeStiff1D to the current panel bay

        Parameters
        ----------
        ys : float
            Stiffener position.
        rho : float, optional
            Stiffener's material density. If not given the bay density will be
            used.
        bb : float, optional
            Stiffener base width.
        bstack : list, optional
            Stacking sequence for the stiffener base laminate.
        bplyts : list, optional
            Thicknesses for each stiffener base ply.
        bplyt : float, optional
            Unique thickness for all stiffener base plies.
        blaminaprops : list, optional
            Lamina properties for each stiffener base ply.
        blaminaprop : float, optional
            Unique lamina properties for all stiffener base plies.
        bf : float
            Stiffener flange width.
        fstack : list, optional
            Stacking sequence for the stiffener flange laminate.
        fplyts : list, optional
            Thicknesses for each stiffener flange ply.
        fplyt : float, optional
            Unique thickness for all stiffener flange plies.
        flaminaprops : list, optional
            Lamina properties for each stiffener flange ply.
        flaminaprop : float, optional
            Unique lamina properties for all stiffener flange plies.

        Returns
        -------
        s : :class:`.BladeStiff1D` object

        Notes
        -----
        Additional parameters can be passed using the ``kwargs``.

        """
        if rho is None:
            rho = self.rho

        if bstack is None and fstack is None:
            raise ValueError('bstack or fstack must be defined!')

        if bstack is not None:
            if bplyts is None:
                if bplyt is None:
                    raise ValueError('bplyts or bplyt must be defined!')
                else:
                    bplyts = [bplyt for _ in bstack]
            if blaminaprops is None:
                if blaminaprop is None:
                    raise ValueError('blaminaprops or blaminaprop must be defined!')
                else:
                    blaminaprops = [blaminaprop for _ in bstack]

        if fstack is not None:
            if fplyts is None:
                if fplyt is None:
                    raise ValueError('fplyts or fplyt must be defined!')
                else:
                    fplyts = [fplyt for _ in fstack]
            if flaminaprops is None:
                if flaminaprop is None:
                    raise ValueError('flaminaprops or flaminaprop must be defined!')
                else:
                    flaminaprops = [flaminaprop for _ in fstack]

        if len(self.shells) == 0:
            raise RuntimeError('The shells must be added before the stiffeners')

        # finding shell1 and shell2
        shell1 = None
        shell2 = None

        for s in self.shells:
            if s.y2 == ys:
                shell1 = s
            if s.y1 == ys:
                shell2 = s
            if np.isclose(ys, 0):
                if np.isclose(s.y1, ys):
                    shell1 = shell2 = s
            if np.isclose(ys, self.b):
                if np.isclose(s.y2, ys):
                    shell1 = shell2 = s

        if shell1 is None or shell2 is None:
            raise RuntimeError('shell1 and shell2 could not be found!')

        s = BladeStiff1D(bay=self, rho=rho, shell1=shell1, shell2=shell2, ys=ys,
                bb=bb, bf=bf, bstack=bstack, bplyts=bplyts,
                blaminaprops=blaminaprops, fstack=fstack, fplyts=fplyts,
                flaminaprops=flaminaprops)

        for k, v in kwargs.items():
            setattr(s, k, v)

        self.bladestiff1ds.append(s)
        self.stiffeners.append(s)

        return s


    def add_bladestiff2d(self, ys, rho=None, bb=None, bstack=None,
            bplyts=None, bplyt=None, blaminaprops=None, blaminaprop=None,
            bf=None, fstack=None, fplyts=None, fplyt=None, flaminaprops=None,
            flaminaprop=None, mf=14, nf=11, **kwargs):
        """Add a new BladeStiff2D to the current panel bay

        Parameters
        ----------
        ys : float
            Stiffener position.
        rho : float, optional
            Stiffener's material density. If not given the bay density will be
            used.
        bb : float, optional
            Stiffener base width.
        bstack : list, optional
            Stacking sequence for the stiffener base laminate.
        bplyts : list, optional
            Thicknesses for each stiffener base ply.
        bplyt : float, optional
            Unique thickness for all stiffener base plies.
        blaminaprops : list, optional
            Lamina properties for each stiffener base ply.
        blaminaprop : float, optional
            Unique lamina properties for all stiffener base plies.
        bf : float
            Stiffener flange width.
        fstack : list, optional
            Stacking sequence for the stiffener flange laminate.
        fplyts : list, optional
            Thicknesses for each stiffener flange ply.
        fplyt : float, optional
            Unique thickness for all stiffener flange plies.
        flaminaprops : list, optional
            Lamina properties for each stiffener flange ply.
        flaminaprop : float, optional
            Unique lamina properties for all stiffener flange plies.
        mf : int, optional
            Number of approximation terms for flange, along `x`.
        nf : int, optional
            Number of approximation terms for flange, along `y`.

        Returns
        -------
        s : :class:`.BladeStiff2D` object

        Notes
        -----
        Additional parameters can be passed using the ``kwargs``.

        """
        if rho is None:
            rho = self.rho

        if bstack is None and fstack is None:
            raise ValueError('bstack or fstack must be defined!')

        if bstack is not None:
            if bplyts is None:
                if bplyt is None:
                    raise ValueError('bplyts or bplyt must be defined!')
                else:
                    bplyts = [bplyt for _ in bstack]
            if blaminaprops is None:
                if blaminaprop is None:
                    raise ValueError('blaminaprops or blaminaprop must be defined!')
                else:
                    blaminaprops = [blaminaprop for _ in bstack]

        if fstack is not None:
            if fplyts is None:
                if fplyt is None:
                    raise ValueError('fplyts or fplyt must be defined!')
                else:
                    fplyts = [fplyt for _ in fstack]
            if flaminaprops is None:
                if flaminaprop is None:
                    raise ValueError('flaminaprops or flaminaprop must be defined!')
                else:
                    flaminaprops = [flaminaprop for _ in fstack]

        if len(self.shells) == 0:
            raise RuntimeError('The shells must be added before the stiffeners')

        # finding shell1 and shell2
        shell1 = None
        shell2 = None

        for s in self.shells:
            if s.y2 == ys:
                shell1 = s
            if s.y1 == ys:
                shell2 = s
            if np.isclose(ys, 0):
                if np.isclose(s.y1, ys):
                    shell1 = shell2 = s
            if np.isclose(ys, self.b):
                if np.isclose(s.y2, ys):
                    shell1 = shell2 = s

        if shell1 is None or shell2 is None:
            raise RuntimeError('shell1 and shell2 could not be found!')

        s = BladeStiff2D(bay=self, rho=rho, shell1=shell1, shell2=shell2, ys=ys,
                bb=bb, bf=bf, bstack=bstack, bplyts=bplyts,
                blaminaprops=blaminaprops, fstack=fstack, fplyts=fplyts,
                flaminaprops=flaminaprops, mf=mf, nf=nf)

        for k, v in kwargs.items():
            setattr(s, k, v)

        self.bladestiff2ds.append(s)
        self.stiffeners.append(s)

        return s


    def add_shell(self, y1, y2, stack=None, plyts=None, plyt=None,
            laminaprops=None, laminaprop=None, model=None, rho=None, **kwargs):
        """Add a new shell to the current panel bay

        Parameters
        ----------
        y1 : float
            Position of the first shell edge along `y`.
        y2 : float
            Position of the second shell edge along `y`.
        stack : list, optional
            Shell laminate stacking sequence. If not given the stacking sequence of the
            bay will be used.
        plyts : list, optional
            Thicknesses for each shell ply. If not supplied the bay ``plyts``
            attribute will be used.
        plyt : float, optional
            Unique thickness to be used for all shell plies. If not supplied
            the bay ``plyt`` attribute will be used.
        laminaprops : list, optional
            Lamina properties for each shell ply.
        laminaprop : list, optional
            Unique lamina properties for all shell plies.
        model : str, optional
            Not recommended to pass this parameter, but the user can use a
            different model for each shell. It is recommended to defined
            ``model`` for the bay object.
        rho : float, optional
            Shell material density. If not given the bay density will be used.

        Notes
        -----
        Additional parameters can be passed using the ``kwargs``.

        """
        s = Shell(a=self.a, b=self.b, m=self.m, n=self.n, r=self.r,
                  alphadeg=self.alphadeg, y1=y1, y2=y2,
                  x1u=self.x1u, x2u=self.x2u,
                  x1v=self.x1v, x2v=self.x2v,
                  x1w=self.x1w, x1wr=self.x1wr, x2w=self.x2w, x2wr=self.x2wr,
                  y1u=self.y1u, y2u=self.y2u,
                  y1v=self.y1v, y2v=self.y2v,
                  y1w=self.y1w, y1wr=self.y1wr, y2w=self.y2w, y2wr=self.y2wr)
        s.model = model if model is not None else self.model
        s.stack = stack if stack is not None else self.stack
        s.plyt = plyt if plyt is not None else self.plyt
        s.plyts = plyts if plyts is not None else self.plyts
        s.laminaprop = laminaprop if laminaprop is not None else self.laminaprop
        s.laminaprops = laminaprops if laminaprops is not None else self.laminaprops
        s.rho = rho if rho is not None else self.rho

        for k, v in kwargs.items():
            setattr(s, k, v)

        self.shells.append(s)

        return s


    def calc_kC(self, silent=False):
        self._rebuild()
        msg('Calculating kC... ', level=2, silent=silent)

        size = self.get_size()

        kC = 0.
        for s in self.shells:
            s.calc_kC(size=size, row0=0, col0=0, silent=True, finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kC += s.matrices['kC']

        for s in self.bladestiff1ds:
            s.calc_kC(size=size, row0=0, col0=0, silent=True, finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kC += s.matrices['kC']

        num = shellmDB.db[self.model]['num']
        m = self.m
        n = self.n
        row0 = num*m*n
        col0 = num*m*n
        for i, s in enumerate(self.bladestiff2ds):
            if i > 0:
                s_1 = self.bladestiff2ds[i-1]
            s.calc_kC(size=size, row0=row0, col0=col0, silent=True, finalize=False)
            if s.flange is not None:
                row0 += s.flange.get_size()
                col0 += s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kC += s.matrices['kC']

        kC = finalize_symmetric_matrix(kC)
        self.matrices['kC'] = kC

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kC


    def calc_kG(self, silent=False, c=None):
        self._rebuild()
        msg('Calculating kG... ', level=2, silent=silent)

        size = self.get_size()

        kG = 0.
        for s in self.shells:
            s.calc_kG(size=size, row0=0, col0=0, silent=True,
                       finalize=False, c=c)
            #TODO summing up coo_matrix objects may be slow!
            kG += s.matrices['kG']

        for s in self.bladestiff1ds:
            s.calc_kG(size=size, row0=0, col0=0, silent=True,
                       finalize=False, c=c)
            #TODO summing up coo_matrix objects may be slow!
            kG += s.matrices['kG']

        num = shellmDB.db[self.model]['num']
        m = self.m
        n = self.n
        row0 = num*m*n
        col0 = num*m*n

        for i, s in enumerate(self.bladestiff2ds):
            if i > 0:
                s_1 = self.bladestiff2ds[i-1]
            s.calc_kG(size=size, row0=row0, col0=col0, silent=True,
                       finalize=False, c=c)
            if s.flange is not None:
                row0 += s.flange.get_size()
                col0 += s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kG += s.matrices['kG']

        kG = finalize_symmetric_matrix(kG)
        self.matrices['kG'] = kG

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kG


    def calc_kM(self, silent=False):
        self._rebuild()
        msg('Calculating kM... ', level=2, silent=silent)
        model = self.model

        size = self.get_size()

        kM = 0.

        for s in self.shells:
            s.calc_kM(size=size, row0=0, col0=0, silent=True,
                      finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kM += s.matrices['kM']

        for s in self.bladestiff1ds:
            s.calc_kM(size=size, row0=0, col0=0, silent=True,
                      finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kM += s.matrices['kM']

        num = shellmDB.db[self.model]['num']
        m = self.m
        n = self.n
        row0 = num*m*n
        col0 = num*m*n

        for i, s in enumerate(self.bladestiff2ds):
            if i > 0:
                s_1 = self.bladestiff2ds[i-1]
            s.calc_kM(size=size, row0=row0, col0=col0, silent=True,
                    finalize=False)
            if s.flange is not None:
                row0 += s.flange.get_size()
                col0 += s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kM += s.matrices['kM']

        kM = finalize_symmetric_matrix(kM)
        self.matrices['kM'] = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kM


    def calc_kA(self, silent=False):
        self._rebuild()
        msg('Calculating kA... ', level=2, silent=silent)
        model = self.model
        a = self.a
        b = self.b
        r = self.r if self.r is not None else 0.
        m = self.m
        n = self.n
        size = self.get_size()

        if self.beta is None:
            if self.Mach < 1:
                raise ValueError('Mach number must be >= 1')
            elif self.Mach == 1:
                self.Mach = 1.0001
            Mach = self.Mach
            beta = self.rho_air * self.V**2 / (Mach**2 - 1)**0.5
            if r != 0.:
                gamma = beta*1./(2.*self.r*(Mach**2 - 1)**0.5)
            else:
                gamma = 0.
            ainf = self.speed_sound
            aeromu = beta/(Mach*ainf)*(Mach**2 - 2)/(Mach**2 - 1)
        else:
            beta = self.beta
            gamma = self.gamma if self.gamma is not None else 0.
            aeromu = self.aeromu if self.aeromu is not None else 0.

        # contributions from shells
        #TODO summing up coo_matrix objects may be slow!
        #FIXME this only works if the first shell represent the full
        #      panelbay domain (mainly integration interval, boundary
        #      conditions)
        s = self.shells[0]
        #FIXME the initialization below looks terrible
        #      we should move as quick as possible to the strategy of using
        #      classes more to carry data, avoiding these intrincated methods
        #      shared among classes... (calc_kC, calc_kG etc)
        s.flow = self.flow
        s.Mach = self.Mach
        s.rho_air = self.rho_air
        s.speed_sound = self.speed_sound
        s.size = self.size
        s.V = self.V
        s.r = self.r
        s.calc_kA(silent=True, finalize=False)
        kA = s.matrices['kA']

        assert np.any(np.isnan(kA.data)) == False
        assert np.any(np.isinf(kA.data)) == False
        kA = csr_matrix(make_skew_symmetric(kA))
        self.matrices['kA'] = kA

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kA


    def calc_cA(self, silent=False):
        self._rebuild()
        msg('Calculating cA... ', level=2, silent=silent)
        model = self.model
        a = self.a
        b = self.b
        r = self.r
        m = self.m
        n = self.n
        size = self.get_size()

        if self.beta is None:
            if self.Mach < 1:
                raise ValueError('Mach number must be >= 1')
            elif self.Mach == 1:
                self.Mach = 1.0001
            Mach = self.Mach
            beta = self.rho_air * self.V**2 / (Mach**2 - 1)**0.5
            gamma = beta*1./(2.*self.r*(Mach**2 - 1)**0.5)
            ainf = self.speed_sound
            aeromu = beta/(Mach*ainf)*(Mach**2 - 2)/(Mach**2 - 1)
        else:
            beta = self.beta
            gamma = self.gamma if self.gamma is not None else 0.
            aeromu = self.aeromu if self.aeromu is not None else 0.

        # contributions from shells
        s = self.shells[0]
        s.calc_cA(size=size, row0=0, col0=0, silent=silent)
        cA = s.cA

        cA = finalize_symmetric_matrix(cA)
        self.matrices['cA'] = cA

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return cA


    def uvw_skin(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement field is
        calculated and stored in the ``fields`` parameter of the
        :class:`.PanelBay` object.

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
            A tuple of ``np.ndarrays`` containing
            ``(u, v, w, phix, phiy)``.

        Notes
        -----
        The returned values ``plot_mesh``, ``fields`` are also
        stored as parameters with the same name in the
        :class:`.PanelBay` object.

        """
        c = np.ascontiguousarray(c, dtype=np.float64)

        m = self.m
        n = self.n
        a = self.a
        b = self.b

        if xs is None or ys is None:
            xs, ys, xshape, yshape = self._default_field(xs, a, ys, b, gridx, gridy)
        else:
            xshape = xs.shape

        if c.shape[0] == self.get_size():
            num = shellmDB.db[self.model]['num']
            c = c[:num*self.m*self.n]
        else:
            raise ValueError('c must be the full vector of Ritz constants')

        fuvw = shellmDB.db[self.model]['field'].fuvw
        us, vs, ws, phixs, phiys = fuvw(c, self, xs, ys, self.out_num_cores)

        self.plot_mesh['Xs'] = xs.reshape(xshape)
        self.plot_mesh['Ys'] = ys.reshape(xshape)
        self.fields['u'] = us.reshape(xshape)
        self.fields['v'] = vs.reshape(xshape)
        self.fields['w'] = ws.reshape(xshape)
        self.fields['phix'] = phixs.reshape(xshape)
        self.fields['phiy'] = phiys.reshape(xshape)

        return self.plot_mesh, self.fields


    def uvw_stiffener(self, c, si, region='flange', xs=None, ys=None,
            gridx=300, gridy=300):
        r"""Calculate the displacement field on a stiffener

        For a given full set of Ritz constants ``c``, the displacement field is
        calculated and stored in the ``fields`` parameter of the
        :class:`PanelBay` object.

        Parameters
        ----------
        c : float
            The full set of Ritz constants.
        si : int
            Stiffener index.
        region : str, optional
            Stiffener region ('base', 'flange' etc).
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
            A tuple of ``np.ndarrays`` containing
            ``(u, v, w, phix, phiy)``.

        Notes
        -----
        The returned values ``plot_mesh``, ``fields`` are also
        stored as parameters with the same name in the
        :class:`.PanelBay` object.

        """
        c = np.ascontiguousarray(c, dtype=np.float64)

        stiff = self.stiffeners[si]
        if isinstance(stiff, BladeStiff1D):
            raise RuntimeError('Use plot_skin for BladeStiff1D')
        if region.lower() == 'base' and isinstance(stiff, BladeStiff2D):
            #TODO why this case isn't working?
            raise RuntimeError('Use plot_skin for the base of BladeStiff2D')

        num = shellmDB.db[self.model]['num']
        row_init = num*self.m*self.n

        # getting array position
        for i, s in enumerate(self.stiffeners):
            if i > 0:
                s_1 = self.stiffeners[i-1]
                if isinstance(s, BladeStiff2D):
                    row_init += s_1.get_size()
            if i == si:
                break

        if region.lower() == 'base':
            bstiff = stiff.base.b
            raise NotImplementedError('lascou-se')
        elif region.lower() == 'flange':
            bstiff = stiff.flange.b
            if isinstance(s, BladeStiff2D):
                row_final = row_init + s.flange.get_size()
        else:
            raise ValueError('Invalid region')

        if c.shape[0] == self.get_size():
            c = c[row_init: row_final]
        else:
            raise ValueError('c must be the full vector of Ritz constants')

        if xs is None or ys is None:
            xs, ys, xshape, yshape = self._default_field(xs, self.a, ys, bstiff, gridx, gridy)
        else:
            xshape = xs.shape

        if region.lower() == 'flange':
            fuvw = shellmDB.db[s.flange.model]['field'].fuvw
            us, vs, ws, phixs, phiys = fuvw(c, s.flange, xs, ys, self.out_num_cores)

        elif region.lower() == 'base':
            fuvw = shellmDB.db[s.base.model]['field'].fuvw
            us, vs, ws, phixs, phiys = fuvw(c, s.base, xs, ys, self.out_num_cores)

        self.plot_mesh['Xs'] = xs.reshape(xshape)
        self.plot_mesh['Ys'] = ys.reshape(xshape)
        self.fields['u'] = us.reshape(xshape)
        self.fields['v'] = vs.reshape(xshape)
        self.fields['w'] = ws.reshape(xshape)
        self.fields['phix'] = phixs.reshape(xshape)
        self.fields['phiy'] = phiys.reshape(xshape)

        return self.plot_mesh, self.fields


    def plot_skin(self, c, invert_y=False, plot_type=1, vec='w',
            deform_u=False, deform_u_sf=100., filename='', ax=None,
            figsize=(3.5, 2.), save=True, title='', colorbar=False,
            cbar_nticks=2, cbar_format=None, cbar_title='', cbar_fontsize=10,
            aspect='equal', clean=True, dpi=400, texts=[], xs=None, ys=None,
            gridx=300, gridy=300, num_levels=400, vecmin=None, vecmax=None,
            silent=False):
        r"""Contour plot for a Ritz constants vector.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the field contour.
        vec : str, optional
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``
            - Strain: ``'exx'``, ``'eyy'``, ``'gxy'``, ``'kxx'``, ``'kyy'``,
              ``'kxy'``, ``'gyz'``, ``'gxz'``
            - Stress: ``'Nxx'``, ``'Nyy'``, ``'Nxy'``, ``'Mxx'``, ``'Myy'``,
              ``'Mxy'``, ``'Qy'``, ``'Qx'``
        deform_u : bool, optional
            If ``True`` the contour plot will look deformed.
        deform_u_sf : float, optional
            The scaling factor used to deform the contour.
        invert_y : bool, optional
            Inverts the `y` axis of the plot. It may be used to match
            the coordinate system of the finite element models created
            using the ``desicos.abaqus`` module.
        plot_type : int, optional
            For cylinders only ``4`` and ``5`` are valid.
            For cones all the following types can be used:

            - ``1``: concave up (with ``invert_y=False``) (default)
            - ``2``: concave down (with ``invert_y=False``)
            - ``3``: stretched closed
            - ``4``: stretched opened (`r \times y` vs. `a`)
            - ``5``: stretched opened (`y` vs. `a`)

        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
        dpi : int, optional
            Resolution of the saved file in dots per inch.
        filename : str, optional
            The file name for the generated image file. If no value is given,
            the `name` parameter of the :class:`PanelBay` object
            will be used.
        ax : AxesSubplot, optional
            When ``ax`` is given, the contour plot will be created inside it.
        figsize : tuple, optional
            The figure size given by ``(width, height)``.
        title : str, optional
            If any string is given it is added as title to the contour plot.
        colorbar : bool, optional
            If a colorbar should be added to the contour plot.
        cbar_nticks : int, optional
            Number of ticks added to the colorbar.
        cbar_format : [ None | format string | Formatter object ], optional
            See the ``matplotlib.pyplot.colorbar`` documentation.
        cbar_fontsize : int, optional
            Fontsize of the colorbar labels.
        cbar_title : str, optional
            Colorbar title. If ``cbar_title == ''`` no title is added.
        aspect : str, optional
            String that will be passed to the ``AxesSubplot.set_aspect()``
            method.
        clean : bool, optional
            Clean axes ticks, grids, spines etc.
        xs : np.ndarray, optional
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        ys : np.ndarray, optional
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).
        vecmin : float, optional
            Minimum value for the contour scale (useful to compare with other
            results). If not specified it will be taken from the calculated
            field.
        vecmax : float, optional
            Maximum value for the contour scale.
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib object that can be used to modify the current plot
            if needed.

        """
        msg('Plotting contour...', silent=silent)

        fields_bkp = deepcopy(self.fields)

        import matplotlib
        if platform.system().lower() == 'linux':
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        msg('Computing field variables...', level=1, silent=silent)
        displs = ['u', 'v', 'w', 'phix', 'phiy']

        if vec in displs:
            self.uvw_skin(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
            field = self.fields.get(vec)
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))

        msg('Finished!', level=1, silent=silent)

        Xs = self.plot_mesh['Xs']
        Ys = self.plot_mesh['Ys']

        if vecmin is None:
            vecmin = field.min()
        if vecmax is None:
            vecmax = field.max()

        levels = linspace(vecmin, vecmax, num_levels)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            if isinstance(ax, matplotlib.axes.Axes):
                ax = ax
                fig = ax.figure
                save = False
            else:
                raise ValueError('ax must be an Axes object')

        x = Ys
        y = Xs

        if deform_u:
            if vec not in displs:
                self.uvw_skin(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
            field_u = self.fields['u']
            field_v = self.fields['v']
            y -= deform_u_sf*field_u
            x += deform_u_sf*field_v
        contour = ax.contourf(x, y, field, levels=levels)

        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fsize = cbar_fontsize
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbarticks = linspace(vecmin, vecmax, cbar_nticks)
            cbar = plt.colorbar(contour, ticks=cbarticks, format=cbar_format,
                                cax=cax)
            if cbar_title:
                cax.text(0.5, 1.05, cbar_title, horizontalalignment='center',
                         verticalalignment='bottom', fontsize=fsize)
            cbar.outline.remove()
            cbar.ax.tick_params(labelsize=fsize, pad=0., tick2On=False)

        if invert_y == True:
            ax.invert_yaxis()
        ax.invert_xaxis()

        if title != '':
            ax.set_title(str(title))

        fig.tight_layout()
        ax.set_aspect(aspect)

        ax.grid(False)
        ax.set_frame_on(False)
        if clean:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
        else:
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        for kwargs in texts:
            ax.text(transform=ax.transAxes, **kwargs)

        if save:
            if not filename:
                filename = 'test.png'
            fig.savefig(filename, transparent=True,
                        bbox_inches='tight', pad_inches=0.05, dpi=dpi)
            plt.close()

        for k, v in fields_bkp.items():
            if v is not None:
                self.fields[k] = v

        msg('finished!', silent=silent)

        return ax


    def plot_stiffener(self, c, si, region='flange', invert_y=False,
            plot_type=1, vec='w', deform_u=False, deform_u_sf=100.,
            filename='', ax=None, figsize=(3.5, 2.), save=True, title='',
            colorbar=False, cbar_nticks=2, cbar_format=None, cbar_title='',
            cbar_fontsize=10, aspect='equal', clean=True, dpi=400, texts=[],
            xs=None, ys=None, gridx=300, gridy=300, num_levels=400,
            vecmin=None, vecmax=None, silent=False):
        r"""Contour plot for a Ritz constants vector.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the field contour.
        si : int
            Stiffener index.
        region : str, optional
            Stiffener region ('base', 'flange' etc).
        vec : str, optional
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``
            - Strain: ``'exx'``, ``'eyy'``, ``'gxy'``, ``'kxx'``, ``'kyy'``,
              ``'kxy'``, ``'gyz'``, ``'gxz'``
            - Stress: ``'Nxx'``, ``'Nyy'``, ``'Nxy'``, ``'Mxx'``, ``'Myy'``,
              ``'Mxy'``, ``'Qy'``, ``'Qx'``
        deform_u : bool, optional
            If ``True`` the contour plot will look deformed.
        deform_u_sf : float, optional
            The scaling factor used to deform the contour.
        invert_y : bool, optional
            Inverts the `y` axis of the plot. It may be used to match
            the coordinate system of the finite element models created
            using the ``desicos.abaqus`` module.
        plot_type : int, optional
            For cylinders only ``4`` and ``5`` are valid.
            For cones all the following types can be used:

            - ``1``: concave up (with ``invert_y=False``) (default)
            - ``2``: concave down (with ``invert_y=False``)
            - ``3``: stretched closed
            - ``4``: stretched opened (`r \times y` vs. `a`)
            - ``5``: stretched opened (`y` vs. `a`)

        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
        dpi : int, optional
            Resolution of the saved file in dots per inch.
        filename : str, optional
            The file name for the generated image file. If no value is given,
            the `name` parameter of the :class:`PanelBay` object
            will be used.
        ax : AxesSubplot, optional
            When ``ax`` is given, the contour plot will be created inside it.
        figsize : tuple, optional
            The figure size given by ``(width, height)``.
        title : str, optional
            If any string is given it is added as title to the contour plot.
        colorbar : bool, optional
            If a colorbar should be added to the contour plot.
        cbar_nticks : int, optional
            Number of ticks added to the colorbar.
        cbar_format : [ None | format string | Formatter object ], optional
            See the ``matplotlib.pyplot.colorbar`` documentation.
        cbar_fontsize : int, optional
            Fontsize of the colorbar labels.
        cbar_title : str, optional
            Colorbar title. If ``cbar_title == ''`` no title is added.
        aspect : str, optional
            String that will be passed to the ``AxesSubplot.set_aspect()``
            method.
        clean : bool, optional
            Clean axes ticks, grids, spines etc.
        xs : np.ndarray, optional
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        ys : np.ndarray, optional
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).
        vecmin : float, optional
            Minimum value for the contour scale (useful to compare with other
            results). If not specified it will be taken from the calculated
            field.
        vecmax : float, optional
            Maximum value for the contour scale.
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib object that can be used to modify the current plot
            if needed.

        """
        msg('Plotting contour...', silent=silent)

        fields_bkp = deepcopy(self.fields)

        import matplotlib.pyplot as plt
        import matplotlib

        msg('Computing field variables...', level=1, silent=silent)
        displs = ['u', 'v', 'w', 'phix', 'phiy']

        if vec in displs:
            self.uvw_stiffener(c, si=si, region=region, xs=xs, ys=ys,
                               gridx=gridx, gridy=gridy)
            field = self.fields.get(vec)
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))

        msg('Finished!', level=1, silent=silent)

        Xs = self.plot_mesh['Xs']
        Ys = self.plot_mesh['Ys']

        if vecmin is None:
            vecmin = field.min()
        if vecmax is None:
            vecmax = field.max()

        levels = linspace(vecmin, vecmax, num_levels)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            if isinstance(ax, matplotlib.axes.Axes):
                ax = ax
                fig = ax.figure
                save = False
            else:
                raise ValueError('ax must be an Axes object')

        x = Ys
        y = Xs

        if deform_u:
            if vec not in displs:
                self.uvw_stiffener(c, si=si, region=region, xs=xs, ys=ys,
                                   gridx=gridx, gridy=gridy)
            field_u = self.fields['u']
            field_v = self.fields['v']
            y -= deform_u_sf*field_u
            x += deform_u_sf*field_v
        contour = ax.contourf(x, y, field, levels=levels)

        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fsize = cbar_fontsize
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbarticks = linspace(vecmin, vecmax, cbar_nticks)
            cbar = plt.colorbar(contour, ticks=cbarticks, format=cbar_format,
                                cax=cax)
            if cbar_title:
                cax.text(0.5, 1.05, cbar_title, horizontalalignment='center',
                         verticalalignment='bottom', fontsize=fsize)
            cbar.outline.remove()
            cbar.ax.tick_params(labelsize=fsize, pad=0., tick2On=False)

        if invert_y == True:
            ax.invert_yaxis()
        ax.invert_xaxis()

        if title != '':
            ax.set_title(str(title))

        fig.tight_layout()
        ax.set_aspect(aspect)

        ax.grid(False)
        ax.set_frame_on(False)
        if clean:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
        else:
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        for kwargs in texts:
            ax.text(transform=ax.transAxes, **kwargs)

        if save:
            if not filename:
                filename = 'test.png'
            fig.savefig(filename, transparent=True,
                        bbox_inches='tight', pad_inches=0.05, dpi=dpi)
            plt.close()

        for k, v in fields_bkp.items():
            if v is not None:
                self.fields[k] = v

        msg('finished!', silent=silent)

        return ax


    def save(self):
        """Save the :class:`PanelBay` object using ``pickle``

        Notes
        -----
        The pickled file will have the name stored in
        :property:`.PanelBay.name` followed by a
        ``'.PanelBay'`` extension.

        """
        name = self.name + '.PanelBay'
        msg('Saving PanelBay to {0}'.format(name))

        self._clear_matrices()

        with open(name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


    def calc_fext(self, silent=False):
        """Calculates the external force vector `\{F_{ext}\}`

        Parameters
        ----------
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.

        Returns
        -------
        fext : np.ndarray
            The external force vector

        """
        msg('Calculating external forces...', level=2, silent=silent)
        num = shellmDB.db[self.model]['num']
        fg = shellmDB.db[self.model]['field'].fg

        # punctual forces on skin
        size = num*self.m*self.n
        g = np.zeros((3, size), dtype=np.float64)
        fext_skin = np.zeros(size, dtype=np.float64)
        for i, load in enumerate(self.point_loads_skin):
            x, y, fx, fy, fz = load
            fg(g, x, y, self)

            fpt = np.array([[fx, fy, fz]])
            fext_skin += fpt.dot(g).ravel()

        fext = fext_skin
        # punctual forces on bladestiff2ds
        # flange
        for s in self.bladestiff2ds:
            fg_flange = shellmDB.db[s.flange.model]['field'].fg
            size = s.flange.get_size()
            g_flange = np.zeros((3, size), dtype=np.float64)
            fext_stiffener = np.zeros(size, dtype=np.float64)
            for i, load in enumerate(s.flange.point_loads):
                xf, yf, fx, fy, fz = load
                fg_flange(g_flange, xf, yf, s.flange)
                fpt = np.array([[fx, fy, fz]])
                fext_stiffener += fpt.dot(g_flange).ravel()

            fext = np.concatenate((fext, fext_stiffener))

        msg('finished!', level=2, silent=silent)

        return fext
