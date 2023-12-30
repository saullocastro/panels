import gc
import pickle
from multiprocessing import cpu_count

import numpy as np
from scipy.sparse import csr_matrix
from numpy import linspace, reshape
from structsolve.sparseutils import finalize_symmetric_matrix, make_skew_symmetric

from panels.logger import msg
from panels import Shell, modelDB as panelmDB
from panels.stiffener import BladeStiff1D, BladeStiff2D


DOUBLE = np.float64


def load(name):
    if '.StiffPanelBay' in name:
        return pickle.load(open(name, 'rb'))
    else:
        return pickle.load(open(name + '.StiffPanelBay', 'rb'))


class StiffPanelBay(object):
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
        self.forces_skin = []
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

        # panels
        self.panels = []

        # stiffeners
        self.stiffeners = []
        self.bladestiff1ds = []
        self.bladestiff2ds = []
        self.tstiff2ds = []

        # geometry
        self.a = None
        self.b = None
        self.r = None

        # boundary conditions
        self.x1u = 0.
        self.x1ur = 0.
        self.x2u = 0.
        self.x2ur = 0.
        self.x1v = 0.
        self.x1vr = 0.
        self.x2v = 0.
        self.x2vr = 0.
        self.x1w = 0.
        self.x1wr = 1.
        self.x2w = 0.
        self.x2wr = 1.

        self.y1u = 0.
        self.y1ur = 0.
        self.y2u = 0.
        self.y2ur = 0.
        self.y1v = 0.
        self.y1vr = 0.
        self.y2v = 0.
        self.y2vr = 0.
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
        self.kC = None
        self.kT = None
        self.kM = None
        self.kA = None
        self.cA = None
        self.u = None
        self.v = None
        self.w = None
        self.phix = None
        self.phiy = None
        self.Xs = None
        self.Ys = None

        for panel in self.panels:
            panel.kC = None
            panel.kM = None
            panel.kG = None

        for s in self.bladestiff1ds:
            s.kC = None
            s.kM = None
            s.kG = None

        for s in self.bladestiff2ds:
            s.kC = None
            s.kM = None
            s.kG = None

        for s in self.tstiff2ds:
            s.kC = None
            s.kM = None
            s.kG = None

        gc.collect()


    def _rebuild(self):
        if self.a is None:
            raise ValueError('The length a must be specified')

        if self.b is None:
            raise ValueError('The width b must be specified')

        for p in self.panels:
            p._rebuild()
            if self.model is not None:
                assert self.model == p.model
            else:
                self.model = p.model

        for s in self.bladestiff1ds:
            s._rebuild()

        for s in self.bladestiff2ds:
            s._rebuild()

        for s in self.tstiff2ds:
            s._rebuild()


    def _default_field(self, xs, a, ys, b, gridx, gridy):
        if xs is None or ys is None:
            xs = linspace(0., a, gridx)
            ys = linspace(0., b, gridy)
            xs, ys = np.meshgrid(xs, ys, copy=False)
        xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
        ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
        xshape = xs.shape
        yshape = ys.shape
        if xshape != yshape:
            raise ValueError('Arrays xs and ys must have the same shape')
        self.Xs = xs
        self.Ys = ys
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
        dofs = panelmDB.db[self.model]['dofs']
        self.size = dofs*self.m*self.n

        for s in self.bladestiff2ds:
            self.size += s.flange.get_size()

        for s in self.tstiff2ds:
            self.size += s.base.get_size() + s.flange.get_size()

        return self.size


    def add_bladestiff1d(self, ys, rho=None, bb=None, bstack=None,
            bplyts=None, bplyt=None, blaminaprops=None, blaminaprop=None,
            bf=None, fstack=None, fplyts=None, fplyt=None, flaminaprops=None,
            flaminaprop=None, **kwargs):
        r"""Add a new BladeStiff1D to the current panel bay

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

        if len(self.panels) == 0:
            raise RuntimeError('The panels must be added before the stiffeners')

        # finding panel1 and panel2
        panel1 = None
        panel2 = None

        for p in self.panels:
            if p.y2 == ys:
                panel1 = p
            if p.y1 == ys:
                panel2 = p
            if np.isclose(ys, 0):
                if np.isclose(p.y1, ys):
                    panel1 = panel2 = p
            if np.isclose(ys, self.b):
                if np.isclose(p.y2, ys):
                    panel1 = panel2 = p

        if panel1 is None or panel2 is None:
            raise RuntimeError('panel1 and panel2 could not be found!')

        s = BladeStiff1D(bay=self, rho=rho, panel1=panel1, panel2=panel2, ys=ys,
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
        r"""Add a new BladeStiff2D to the current panel bay

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

        if len(self.panels) == 0:
            raise RuntimeError('The panels must be added before the stiffeners')

        # finding panel1 and panel2
        panel1 = None
        panel2 = None

        for p in self.panels:
            if p.y2 == ys:
                panel1 = p
            if p.y1 == ys:
                panel2 = p
            if np.isclose(ys, 0):
                if np.isclose(p.y1, ys):
                    panel1 = panel2 = p
            if np.isclose(ys, self.b):
                if np.isclose(p.y2, ys):
                    panel1 = panel2 = p

        if panel1 is None or panel2 is None:
            raise RuntimeError('panel1 and panel2 could not be found!')

        s = BladeStiff2D(bay=self, rho=rho, panel1=panel1, panel2=panel2, ys=ys,
                bb=bb, bf=bf, bstack=bstack, bplyts=bplyts,
                blaminaprops=blaminaprops, fstack=fstack, fplyts=fplyts,
                flaminaprops=flaminaprops, mf=mf, nf=nf)

        for k, v in kwargs.items():
            setattr(s, k, v)

        self.bladestiff2ds.append(s)
        self.stiffeners.append(s)

        return s


    def add_panel(self, y1, y2, stack=None, plyts=None, plyt=None,
            laminaprops=None, laminaprop=None, model=None, rho=None, **kwargs):
        r"""Add a new panel to the current panel bay

        Parameters
        ----------
        y1 : float
            Position of the first panel edge along `y`.
        y2 : float
            Position of the second panel edge along `y`.
        stack : list, optional
            Shell stacking sequence. If not given the stacking sequence of the
            bay will be used.
        plyts : list, optional
            Thicknesses for each panel ply. If not supplied the bay ``plyts``
            attribute will be used.
        plyt : float, optional
            Unique thickness to be used for all panel plies. If not supplied
            the bay ``plyt`` attribute will be used.
        laminaprops : list, optional
            Lamina properties for each panel ply.
        laminaprop : list, optional
            Unique lamina properties for all panel plies.
        model : str, optional
            Not recommended to pass this parameter, but the user can use a
            different model for each panel. It is recommended to defined
            ``model`` for the bay object.
        rho : float, optional
            Shell material density. If not given the bay density will be used.

        Notes
        -----
        Additional parameters can be passed using the ``kwargs``.

        """
        p = Shell(a=self.a, b=self.b, m=self.m, n=self.n, r=self.r,
                  y1=y1, y2=y2,
                  x1u=self.x1u, x1ur=self.x1ur, x2u=self.x2u, x2ur=self.x2ur,
                  x1v=self.x1v, x1vr=self.x1vr, x2v=self.x2v, x2vr=self.x2vr,
                  x1w=self.x1w, x1wr=self.x1wr, x2w=self.x2w, x2wr=self.x2wr,
                  y1u=self.y1u, y1ur=self.y1ur, y2u=self.y2u, y2ur=self.y2ur,
                  y1v=self.y1v, y1vr=self.y1vr, y2v=self.y2v, y2vr=self.y2vr,
                  y1w=self.y1w, y1wr=self.y1wr, y2w=self.y2w, y2wr=self.y2wr)
        p.model = model if model is not None else self.model
        p.stack = stack if stack is not None else self.stack
        p.plyt = plyt if plyt is not None else self.plyt
        p.plyts = plyts if plyts is not None else self.plyts
        p.laminaprop = laminaprop if laminaprop is not None else self.laminaprop
        p.laminaprops = laminaprops if laminaprops is not None else self.laminaprops
        p.rho = rho if rho is not None else self.rho

        for k, v in kwargs.items():
            setattr(p, k, v)

        self.panels.append(p)

        return p


    def calc_kC(self, silent=False):
        self._rebuild()
        msg('Calculating kC... ', level=2, silent=silent)

        size = self.get_size()

        kC = 0.
        for p in self.panels:
            p.calc_kC(size=size, row0=0, col0=0, silent=True,
                          finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kC += p.matrices['kC']

        for s in self.bladestiff1ds:
            s.calc_kC(size=size, row0=0, col0=0, silent=True,
                      finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kC += s.kC

        dofs = panelmDB.db[self.model]['dofs']
        m = self.m
        n = self.n
        row0 = dofs*m*n
        col0 = dofs*m*n
        for i, s in enumerate(self.bladestiff2ds):
            if i > 0:
                s_1 = self.bladestiff2ds[i-1]
            s.calc_kC(size=size, row0=row0, col0=col0, silent=True,
                      finalize=False)
            if s.flange is not None:
                row0 += s.flange.get_size()
                col0 += s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kC += s.kC

        for i, s in enumerate(self.tstiff2ds):
            if i > 0:
                s_1 = self.tstiff2ds[i-1]
            s.calc_kC(size=size, row0=row0, col0=col0, silent=True,
                      finalize=False)
            row0 += s.base.get_size() + s.flange.get_size()
            col0 += s.base.get_size() + s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kC += s.kC

        kC = finalize_symmetric_matrix(kC)
        self.kC = kC

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
        for p in self.panels:
            p.calc_kG(size=size, row0=0, col0=0, silent=True,
                       finalize=False, c=c)
            #TODO summing up coo_matrix objects may be slow!
            kG += p.matrices['kG']

        for s in self.bladestiff1ds:
            s.calc_kG(size=size, row0=0, col0=0, silent=True,
                       finalize=False, c=c)
            #TODO summing up coo_matrix objects may be slow!
            kG += s.kG

        dofs = panelmDB.db[self.model]['dofs']
        m = self.m
        n = self.n
        row0 = dofs*m*n
        col0 = dofs*m*n

        for i, s in enumerate(self.bladestiff2ds):
            if i > 0:
                s_1 = self.bladestiff2ds[i-1]
            s.calc_kG(size=size, row0=row0, col0=col0, silent=True,
                       finalize=False, c=c)
            if s.flange is not None:
                row0 += s.flange.get_size()
                col0 += s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kG += s.kG

        for i, s in enumerate(self.tstiff2ds):
            if i > 0:
                s_1 = self.tstiff2ds[i-1]
            s.calc_kG(size=size, row0=row0, col0=col0, silent=True,
                       finalize=False, c=c)
            row0 += s.base.get_size() + s.flange.get_size()
            col0 += s.base.get_size() + s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kG += s.kG


        kG = finalize_symmetric_matrix(kG)
        self.kG = kG

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kG


    def calc_kM(self, silent=False):
        self._rebuild()
        msg('Calculating kM... ', level=2, silent=silent)

        size = self.get_size()

        kM = 0.

        for p in self.panels:
            p.calc_kM(size=size, row0=0, col0=0, silent=True,
                      finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kM += p.matrices['kM']

        for s in self.bladestiff1ds:
            s.calc_kM(size=size, row0=0, col0=0, silent=True,
                      finalize=False)
            #TODO summing up coo_matrix objects may be slow!
            kM += s.kM

        dofs = panelmDB.db[self.model]['dofs']
        m = self.m
        n = self.n
        row0 = dofs*m*n
        col0 = dofs*m*n

        for i, s in enumerate(self.bladestiff2ds):
            if i > 0:
                s_1 = self.bladestiff2ds[i-1]
            s.calc_kM(size=size, row0=row0, col0=col0, silent=True,
                    finalize=False)
            if s.flange is not None:
                row0 += s.flange.get_size()
                col0 += s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kM += s.kM

        for i, s in enumerate(self.tstiff2ds):
            if i > 0:
                s_1 = self.tstiff2ds[i-1]
            s.calc_kM(size=size, row0=row0, col0=col0, silent=True,
                    finalize=False)
            row0 += s.base.get_size() + s.flange.get_size()
            col0 += s.base.get_size() + s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kM += s.kM

        kM = finalize_symmetric_matrix(kM)
        self.kM = kM

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kM


    def calc_kA(self, silent=False):
        self._rebuild()
        msg('Calculating kA... ', level=2, silent=silent)
        a = self.a
        b = self.b
        r = self.r if self.r is not None else 0.
        m = self.m
        n = self.n

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

        # contributions from panels
        #TODO summing up coo_matrix objects may be slow!
        #FIXME this only works if the first panel represent the full
        #      stiffpanelbay domain (mainly integration interval, boundary
        #      conditions)
        p = self.panels[0]
        #FIXME the initialization below looks terrible
        #      we should move as quick as possible to the strategy of using
        #      classes more to carry data, avoiding these intrincated methods
        #      shared among classes... (calc_kC, calc_kG etc)
        p.flow = self.flow
        p.Mach = self.Mach
        p.rho_air = self.rho_air
        p.speed_sound = self.speed_sound
        p.size = self.size
        p.V = self.V
        p.r = self.r
        p.calc_kA(silent=True, finalize=False)
        kA = p.matrices['kA']

        assert np.any(np.isnan(kA.data)) == False
        assert np.any(np.isinf(kA.data)) == False
        kA = csr_matrix(make_skew_symmetric(kA))
        self.kA = kA

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return kA


    def calc_cA(self, silent=False):
        self._rebuild()
        msg('Calculating cA... ', level=2, silent=silent)
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

        # contributions from panels
        p = self.panels[0]
        p.calc_cA(size=size, row0=0, col0=0, silent=silent)
        cA = p.matrices['cA']

        cA = finalize_symmetric_matrix(cA)
        self.cA = cA

        #NOTE forcing Python garbage collector to clean the memory
        #     it DOES make a difference! There is a memory leak not
        #     identified, probably in the csr_matrix process
        gc.collect()

        msg('finished!', level=2, silent=silent)

        return cA


    def uvw_skin(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the
        :class:`.StiffPanelBay` object.

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
        The returned values ``u```, ``v``, ``w``, ``phix``, ``phiy`` are
        stored as parameters with the same name in the
        :class:`.StiffPanelBay` object.

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        m = self.m
        n = self.n
        a = self.a
        b = self.b

        if xs is None or ys is None:
            xs, ys, xshape, yshape = self._default_field(xs, a, ys, b, gridx, gridy)
        else:
            xshape = xs.shape

        if c.shape[0] == self.get_size():
            dofs = panelmDB.db[self.model]['dofs']
            c = c[:dofs*self.m*self.n]
        else:
            raise ValueError('c must be the full vector of Ritz constants')

        fuvw = panelmDB.db[self.model]['field'].fuvw
        us, vs, ws, phixs, phiys = fuvw(c, self, xs, ys, self.out_num_cores)

        self.u = reshape(us, xshape)
        self.v = reshape(vs, xshape)
        self.w = reshape(ws, xshape)
        self.phix = reshape(phixs, xshape)
        self.phiy = reshape(phiys, xshape)

        return self.u, self.v, self.w, self.phix, self.phiy


    def uvw_stiffener(self, c, si, region='flange', xs=None, ys=None,
            gridx=300, gridy=300):
        r"""Calculate the displacement field on a stiffener

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the
        :class:`StiffPanelBay` object.

        Parameters
        ----------
        c : float
            The full set of Ritz constants
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
        The returned values ``u```, ``v``, ``w``, ``phix``, ``phiy`` are
        stored as parameters with the same name in the
        :class:`StiffPanelBay` object.

        """
        c = np.ascontiguousarray(c, dtype=DOUBLE)

        stiff = self.stiffeners[si]
        if isinstance(stiff, BladeStiff1D):
            raise RuntimeError('Use plot_skin for BladeStiff1D')
        if region.lower() == 'base' and isinstance(stiff, BladeStiff2D):
            #TODO why this case isn't working?
            raise RuntimeError('Use plot_skin for the base of BladeStiff2D')

        dofs = panelmDB.db[self.model]['dofs']
        row_init = dofs*self.m*self.n

        # getting array position
        for i, s in enumerate(self.stiffeners):
            if i > 0:
                s_1 = self.stiffeners[i-1]
                if isinstance(s, BladeStiff2D):
                    row_init += s_1.get_size()
            if i == si:
                break

        if region.lower() == 'flange':
            bstiff = stiff.flange.b
            if isinstance(s, BladeStiff2D):
                row_final = row_init + s.flange.get_size()
            else:
                raise ValueError('Invalid region')
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
            fuvw = panelmDB.db[s.flange.model]['field'].fuvw
            us, vs, ws, phixs, phiys = fuvw(c, s.flange, xs, ys, self.out_num_cores)

        elif region.lower() == 'base':
            fuvw = panelmDB.db[s.base.model]['field'].fuvw
            us, vs, ws, phixs, phiys = fuvw(c, s.base, xs, ys, self.out_num_cores)

        self.u = reshape(us, xshape)
        self.v = reshape(vs, xshape)
        self.w = reshape(ws, xshape)
        self.phix = reshape(phixs, xshape)
        self.phiy = reshape(phiys, xshape)

        return self.u, self.v, self.w, self.phix, self.phiy


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
            the `name` parameter of the :class:`StiffPanelBay` object
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

        ubkp, vbkp, wbkp, phixbkp, phiybkp = (self.u, self.v, self.w,
                                              self.phix, self.phiy)

        import matplotlib
        import matplotlib.pyplot as plt

        msg('Computing field variables...', level=1, silent=silent)
        displs = ['u', 'v', 'w', 'phix', 'phiy']

        if vec in displs:
            self.uvw_skin(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
            field = getattr(self, vec)
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))

        msg('Finished!', level=1, silent=silent)

        Xs = self.Xs
        Ys = self.Ys

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
            if vec in displs:
                pass
            else:
                self.uvw_skin(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
            field_u = self.u
            field_v = self.v
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
            try:
                cbar.outline.remove()
            except NotImplementedError:
                pass
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

        if ubkp is not None:
            self.u = ubkp
        if vbkp is not None:
            self.v = vbkp
        if wbkp is not None:
            self.w = wbkp
        if phixbkp is not None:
            self.phix = phixbkp
        if phiybkp is not None:
            self.phiy = phiybkp

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
            the `name` parameter of the :class:`StiffPanelBay` object
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

        ubkp, vbkp, wbkp, phixbkp, phiybkp = (self.u, self.v, self.w,
                                              self.phix, self.phiy)

        import matplotlib.pyplot as plt
        import matplotlib

        msg('Computing field variables...', level=1, silent=silent)
        displs = ['u', 'v', 'w', 'phix', 'phiy']

        if vec in displs:
            self.uvw_stiffener(c, si=si, region=region, xs=xs, ys=ys,
                               gridx=gridx, gridy=gridy)
            field = getattr(self, vec)
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))

        msg('Finished!', level=1, silent=silent)

        Xs = self.Xs
        Ys = self.Ys

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
            if vec in displs:
                pass
            else:
                self.uvw_stiffener(c, si=si, region=region, xs=xs, ys=ys,
                                   gridx=gridx, gridy=gridy)
            field_u = self.u
            field_v = self.v
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
            try:
                cbar.outline.remove()
            except NotImplementedError:
                pass
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

        if ubkp is not None:
            self.u = ubkp
        if vbkp is not None:
            self.v = vbkp
        if wbkp is not None:
            self.w = wbkp
        if phixbkp is not None:
            self.phix = phixbkp
        if phiybkp is not None:
            self.phiy = phiybkp

        msg('finished!', silent=silent)

        return ax


    def save(self):
        r"""Save the :class:`StiffPanelBay` object using ``pickle``

        Notes
        -----
        The pickled file will have the name stored in
        :property:`.StiffPanelBay.name` followed by a
        ``'.StiffPanelBay'`` extension.

        """
        name = self.name + '.StiffPanelBay'
        msg('Saving StiffPanelBay to {0}'.format(name))

        self._clear_matrices()

        with open(name, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)


    def calc_fext(self, silent=False):
        r"""Calculates the external force vector `\{F_{ext}\}`

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
        dofs = panelmDB.db[self.model]['dofs']
        fg = panelmDB.db[self.model]['field'].fg

        # punctual forces on skin
        size = dofs*self.m*self.n
        g = np.zeros((3, size), dtype=DOUBLE)
        fext_skin = np.zeros(size, dtype=DOUBLE)
        for i, force in enumerate(self.forces_skin):
            x, y, fx, fy, fz = force
            fg(g, x, y, self)

            fpt = np.array([[fx, fy, fz]])
            fext_skin += fpt.dot(g).ravel()

        fext = fext_skin
        # punctual forces on bladestiff2ds
        # flange
        for s in self.bladestiff2ds:
            fg_flange = panelmDB.db[s.flange.model]['field'].fg
            size = s.flange.get_size()
            g_flange = np.zeros((3, size), dtype=DOUBLE)
            fext_stiffener = np.zeros(size, dtype=DOUBLE)
            for i, force in enumerate(s.flange.forces):
                xf, yf, fx, fy, fz = force
                fg_flange(g_flange, xf, yf, s.flange)
                fpt = np.array([[fx, fy, fz]])
                fext_stiffener += fpt.dot(g_flange).ravel()

            fext = np.concatenate((fext, fext_stiffener))

        # punctual forces on tstiff2ds
        for s in self.tstiff2ds:
            # base
            size = s.base.get_size()
            g_base = np.zeros((3, size), dtype=DOUBLE)
            fext_base = np.zeros(size, dtype=DOUBLE)
            fg_base = panelmDB.db[s.base.model]['field'].fg
            for i, force in enumerate(s.base.forces):
                xb, yb, fx, fy, fz = force
                fg_base(g_base, xb, yb, s.base)
                fpt = np.array([[fx, fy, fz]])
                fext_base += fpt.dot(g_base).ravel()

            # flange
            size = s.flange.get_size()
            g_flange = np.zeros((3, size), dtype=DOUBLE)
            fext_flange = np.zeros(size, dtype=DOUBLE)
            fg_flange = panelmDB.db[s.flange.model]['field'].fg
            for i, force in enumerate(s.flange.forces):
                xf, yf, fx, fy, fz = force
                fg_flange(g_flange, xf, yf, s.flange)
                fpt = np.array([[fx, fy, fz]])
                fext_flange += fpt.dot(g_flange).ravel()

            fext = np.concatenate((fext, fext_base, fext_flange))

        msg('finished!', level=2, silent=silent)

        return fext
