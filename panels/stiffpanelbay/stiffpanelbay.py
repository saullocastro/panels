r"""Stiffened panel bay (:class:`.StiffPanelBay`)

The class :class:`.StiffPanelBay` assembles the skin panels of a bay,
:class:`.Shell` objects separated along `y`, and the blade stiffeners
:class:`.BladeStiff1D` and :class:`.BladeStiff2D`, and computes the
stiffness, geometric stiffness, mass and aerodynamic matrices, the external
force vector and the displacement fields of the assembly. The function
:func:`.load` reads a bay saved with :meth:`.StiffPanelBay.save`.

"""
import gc
import os

import numpy as np
from scipy.sparse import csr_matrix
from numpy import linspace, reshape
from structsolve.sparseutils import finalize_symmetric_matrix, make_skew_symmetric

from panels import json_io
from panels.logger import msg
from panels.shell import Shell, modelDB as panelmDB
from panels.stiffener import BladeStiff1D, BladeStiff2D


DOUBLE = np.float64


def load(name, max_size=json_io.MAX_SIZE):
    r"""Load a :class:`.StiffPanelBay` saved by :meth:`.StiffPanelBay.save`

    The file is read by :mod:`panels.json_io`, which raises a ``ValueError``
    when it holds an object of another type than :class:`.StiffPanelBay`.

    Parameters
    ----------
    name : str, path-like or file object
        Name of the file, with or without the extension
        ``'.stiffpanelbay.zip'``, or a binary file object opened for
        reading, e.g. :class:`io.BytesIO`.
    max_size : int, optional
        Maximum total uncompressed size of the members read, in bytes, 4 GiB
        by default, see :func:`panels.json_io.load`.

    Returns
    -------
    bay : :class:`.StiffPanelBay`

    """
    return json_io._load_type(name, 'StiffPanelBay', ('.stiffpanelbay.zip', ),
                              max_size)


def _first_not_none(*values):
    """Return the first value that is not ``None``"""
    for v in values:
        if v is not None:
            return v
    return None


class StiffPanelBay(object):
    r"""Stiffened Shell Bay

    Can be used for supersonic Aeroelastic studies with the Piston Theory.

    Stiffeners are modeled either with 1D or 2D formulations.

    Main characteristics:

    - Supports both airflows along x (axis) or y (circumferential).
      Controlled by the parameter ``flow``
    - ``bladestiff1ds`` contains the :class:`.BladeStiff1D` stiffeners
    - ``bladestiff2ds`` contains the :class:`.BladeStiff2D` stiffeners

    Notes
    -----
    The panels of the skin are :class:`.Shell` objects with the same
    approximation functions, spanning the whole bay with its boundary
    conditions, each one integrated only over its strip `y_1 \le y \le
    y_2`. They share the Ritz constants of the skin, so that the
    displacement field of the skin is continuous across the panels without
    any connection, and the matrices of the bay are the sums of those of the
    panels and of the stiffeners. The bases of the stiffeners also use the
    Ritz constants of the skin, and only the flanges of the
    :class:`.BladeStiff2D` stiffeners have their own Ritz constants, which
    follow those of the skin, see :meth:`.get_size`.

    For the flutter analysis with the piston theory, the matrices enter the
    equations of motion as `([K] + [K_A] + \lambda^2 [M])\{c\} = \{0\}`, see
    :meth:`.Shell.calc_kA` and :meth:`.calc_kA`, as in [1]_.

    TODO(author): confirm that the model of the bay is that of [1]_.

    References
    ----------
    .. [1] Castro, S. G. P., Guimaraes, T. A. M., Rade, D. A., and Donadon,
       M. V., Flutter of stiffened composite panels considering the
       stiffener's base as a structural element, Composite Structures, 2016.
       https://doi.org/10.1016/j.compstruct.2015.12.056

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
        self.out_num_cores = os.cpu_count() or 1

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
            panel._clear_matrices()

        #NOTE the stiffeners rebuild their laminates before each calculation
        for s in self.bladestiff1ds:
            s.kC = None
            s.kM = None
            s.kG = None
            s.flam = None
            if s.base is not None:
                s.base._clear_matrices()

        for s in self.bladestiff2ds:
            s.kC = None
            s.kM = None
            s.kG = None
            if s.base is not None:
                s.base._clear_matrices()
            if s.flange is not None:
                s.flange._clear_matrices()

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
        #NOTE the stiffener kernels assume the 3 DOFs u, v, w of the models
        #     based on the classical laminated plate theory
        if self.model is not None and panelmDB.db[self.model]['dofs'] != 3:
            raise NotImplementedError(
                "StiffPanelBay only supports models with 3 DOFs per term "
                "(u, v, w), got model '{0}'".format(self.model))

        for s in self.bladestiff1ds:
            s._rebuild()

        for s in self.bladestiff2ds:
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

        It takes into account the independent degrees of freedom from the
        flange of each of the :class:`.BladeStiff2D` objects that belong to
        the current assembly.

        The size is ``dofs*m*n`` for the skin, ``dofs`` being the number of
        DOFs per term of the model in the attribute ``model``, plus the size
        of each flange. It is also stored in the attribute ``size``.

        Returns
        -------
        size : int
            The size of the stiffness matrices.

        """
        dofs = panelmDB.db[self.model]['dofs']
        self.size = dofs*self.m*self.n

        for s in self.bladestiff2ds:
            if s.flange is not None:
                self.size += s.flange.get_size()

        return self.size


    def add_bladestiff1d(self, ys, rho=None, bb=None, bstack=None,
            bplyts=None, bplyt=None, blaminaprops=None, blaminaprop=None,
            bf=None, fstack=None, fplyts=None, fplyt=None, flaminaprops=None,
            flaminaprop=None, **kwargs):
        r"""Add a new BladeStiff1D to the current panel bay

        The panels must be added before. The panel ``panel1`` of the
        stiffener is the panel whose edge ``y2`` is equal to ``ys``, and
        ``panel2`` the one whose edge ``y1`` is equal to ``ys``; for ``ys``
        close to ``0`` or to the width ``b`` of the bay, both are the panel
        with that edge. When only ``bplyt``, ``blaminaprop``, ``fplyt`` or
        ``flaminaprop`` is given, it is repeated for each ply. The new
        stiffener is appended to the attributes ``bladestiff1ds`` and
        ``stiffeners``.

        Parameters
        ----------
        ys : float
            Stiffener position, the coordinate `y`, length.
        rho : float, optional
            Stiffener's material density. If not given the bay density will be
            used. Mass per unit volume. Default ``None``.
        bb : float, optional
            Stiffener base width, length. Default ``None``.
        bstack : list, optional
            Stacking sequence for the stiffener base laminate. Default
            ``None``, no base.
        bplyts : list, optional
            Thicknesses for each stiffener base ply, length. Default
            ``None``.
        bplyt : float, optional
            Unique thickness for all stiffener base plies, length. Default
            ``None``.
        blaminaprops : list, optional
            Lamina properties for each stiffener base ply. Default ``None``.
        blaminaprop : list or tuple, optional
            Unique lamina properties for all stiffener base plies. Default
            ``None``.
        bf : float, optional
            Stiffener flange width, length. Default ``None``.
        fstack : list, optional
            Stacking sequence for the stiffener flange laminate. Default
            ``None``, no flange.
        fplyts : list, optional
            Thicknesses for each stiffener flange ply, length. Default
            ``None``.
        fplyt : float, optional
            Unique thickness for all stiffener flange plies, length. Default
            ``None``.
        flaminaprops : list, optional
            Lamina properties for each stiffener flange ply. Default ``None``.
        flaminaprop : list or tuple, optional
            Unique lamina properties for all stiffener flange plies. Default
            ``None``.

        **kwargs : dict, optional
            Additional parameters, set as attributes of the new stiffener
            after it is created.

        Returns
        -------
        s : :class:`.BladeStiff1D` object

        Raises
        ------
        ValueError
            When neither ``bstack`` nor ``fstack`` is given; when ``bstack``
            is given without ``bplyts`` or ``bplyt``, or without
            ``blaminaprops`` or ``blaminaprop``; when ``fstack`` is given
            without ``fplyts`` or ``fplyt``, or without ``flaminaprops`` or
            ``flaminaprop``.
        RuntimeError
            When no panel was added; when ``panel1`` or ``panel2`` is not
            found.

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

        The panels must be added before. The panel ``panel1`` of the
        stiffener is the panel whose edge ``y2`` is equal to ``ys``, and
        ``panel2`` the one whose edge ``y1`` is equal to ``ys``; for ``ys``
        close to ``0`` or to the width ``b`` of the bay, both are the panel
        with that edge. When only ``bplyt``, ``blaminaprop``, ``fplyt`` or
        ``flaminaprop`` is given, it is repeated for each ply. The new
        stiffener is appended to the attributes ``bladestiff2ds`` and
        ``stiffeners``.

        Parameters
        ----------
        ys : float
            Stiffener position, the coordinate `y`, length.
        rho : float, optional
            Stiffener's material density. If not given the bay density will be
            used. Mass per unit volume. Default ``None``.
        bb : float, optional
            Stiffener base width, length. Default ``None``.
        bstack : list, optional
            Stacking sequence for the stiffener base laminate. Default
            ``None``, no base.
        bplyts : list, optional
            Thicknesses for each stiffener base ply, length. Default
            ``None``.
        bplyt : float, optional
            Unique thickness for all stiffener base plies, length. Default
            ``None``.
        blaminaprops : list, optional
            Lamina properties for each stiffener base ply. Default ``None``.
        blaminaprop : list or tuple, optional
            Unique lamina properties for all stiffener base plies. Default
            ``None``.
        bf : float, optional
            Stiffener flange width, length. Default ``None``.
        fstack : list, optional
            Stacking sequence for the stiffener flange laminate. Default
            ``None``, no flange.
        fplyts : list, optional
            Thicknesses for each stiffener flange ply, length. Default
            ``None``.
        fplyt : float, optional
            Unique thickness for all stiffener flange plies, length. Default
            ``None``.
        flaminaprops : list, optional
            Lamina properties for each stiffener flange ply. Default ``None``.
        flaminaprop : list or tuple, optional
            Unique lamina properties for all stiffener flange plies. Default
            ``None``.
        mf : int, optional
            Number of approximation terms for flange, along `x`. Default
            ``14``.
        nf : int, optional
            Number of approximation terms for flange, along `y`. Default
            ``11``.
        **kwargs : dict, optional
            Additional parameters, set as attributes of the new stiffener
            after it is created.

        Returns
        -------
        s : :class:`.BladeStiff2D` object

        Raises
        ------
        ValueError
            When neither ``bstack`` nor ``fstack`` is given; when ``bstack``
            is given without ``bplyts`` or ``bplyt``, or without
            ``blaminaprops`` or ``blaminaprop``; when ``fstack`` is given
            without ``fplyts`` or ``fplyt``, or without ``flaminaprops`` or
            ``flaminaprop``.
        RuntimeError
            When no panel was added; when ``panel1`` or ``panel2`` is not
            found.

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

        The panel is a :class:`.Shell` with the attributes ``a``, ``b``,
        ``m``, ``n``, ``r`` and the boundary conditions of the bay,
        integrated between ``y1`` and ``y2``. It is appended to the
        attribute ``panels``.

        Parameters
        ----------
        y1 : float
            Position of the first panel edge along `y`, length.
        y2 : float
            Position of the second panel edge along `y`, length.
        stack : list, optional
            Shell stacking sequence. If not given the stacking sequence of the
            bay will be used. Default ``None``.
        plyts : list, optional
            Thicknesses for each panel ply. If not supplied the bay ``plyts``
            attribute will be used. Length. Default ``None``.
        plyt : float, optional
            Unique thickness to be used for all panel plies. If not supplied
            the bay ``plyt`` attribute will be used. Length. Default ``None``.
        laminaprops : list, optional
            Lamina properties for each panel ply. If not supplied the bay
            ``laminaprops`` attribute is used. Default ``None``.
        laminaprop : list, optional
            Unique lamina properties for all panel plies. If not supplied the
            bay ``laminaprop`` attribute is used. Default ``None``.
        model : str, optional
            Not recommended to pass this parameter, but the user can use a
            different model for each panel. It is recommended to defined
            ``model`` for the bay object. Default ``None``, the attribute
            ``model`` of the bay.
        rho : float, optional
            Shell material density. If not given the bay density will be used.
            Mass per unit volume. Default ``None``; ``0.`` when the density of
            the bay is also ``None``.
        **kwargs : dict, optional
            Additional parameters, set as attributes of the new panel after
            it is created.

        Returns
        -------
        p : :class:`.Shell`
            The new panel.

        """
        #NOTE the laminate and model attributes must be passed to the
        #     constructor, since Shell.__init__() already calls
        #     Shell._rebuild(), which needs them to build the laminate
        p = Shell(a=self.a, b=self.b, m=self.m, n=self.n, r=self.r,
                  y1=y1, y2=y2,
                  model=model if model is not None else self.model,
                  stack=stack if stack is not None else self.stack,
                  plyt=plyt if plyt is not None else self.plyt,
                  plyts=plyts if plyts is not None else self.plyts,
                  laminaprop=(laminaprop if laminaprop is not None
                              else self.laminaprop),
                  laminaprops=(laminaprops if laminaprops is not None
                               else self.laminaprops),
                  rho=_first_not_none(rho, self.rho, 0.),
                  x1u=self.x1u, x1ur=self.x1ur, x2u=self.x2u, x2ur=self.x2ur,
                  x1v=self.x1v, x1vr=self.x1vr, x2v=self.x2v, x2vr=self.x2vr,
                  x1w=self.x1w, x1wr=self.x1wr, x2w=self.x2w, x2wr=self.x2wr,
                  y1u=self.y1u, y1ur=self.y1ur, y2u=self.y2u, y2ur=self.y2ur,
                  y1v=self.y1v, y1vr=self.y1vr, y2v=self.y2v, y2vr=self.y2vr,
                  y1w=self.y1w, y1wr=self.y1wr, y2w=self.y2w, y2wr=self.y2wr)

        for k, v in kwargs.items():
            setattr(p, k, v)

        self.panels.append(p)

        return p


    def calc_kC(self, silent=False):
        r"""Calculate the constitutive stiffness matrix of the assembly

        The bay is rebuilt and its size computed with :meth:`.get_size`.
        The matrices of the panels and of the :class:`.BladeStiff1D`
        stiffeners are placed at row and column ``0``, the Ritz constants of
        the skin; those of the :class:`.BladeStiff2D` stiffeners at the Ritz
        constants of their flanges, which follow those of the skin in the
        order of the attribute ``bladestiff2ds``. The sum is finalized with
        :func:`structsolve.sparseutils.finalize_symmetric_matrix` and stored
        in the attribute ``kC``. Messages are printed unless
        ``silent=True``.

        Parameters
        ----------
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.

        Returns
        -------
        kC : sparse matrix
            The constitutive stiffness matrix, of shape ``(size, size)``.

        """
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
        for s in self.bladestiff2ds:
            s.calc_kC(size=size, row0=row0, col0=col0, silent=True,
                      finalize=False)
            if s.flange is not None:
                row0 += s.flange.get_size()
                col0 += s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kC += s.kC

        kC = finalize_symmetric_matrix(kC)
        self.kC = kC

        msg('finished!', level=2, silent=silent)

        return kC


    def calc_kG(self, silent=False, c=None):
        r"""Calculate the geometric stiffness matrix of the assembly

        The bay is rebuilt and its size computed with :meth:`.get_size`.
        The matrices of the panels, :meth:`.Shell.calc_kG`, and of the
        :class:`.BladeStiff1D` stiffeners are placed at row and column
        ``0``; those of the :class:`.BladeStiff2D` stiffeners at the Ritz
        constants of their flanges, as in :meth:`.calc_kC`. The sum is
        finalized with
        :func:`structsolve.sparseutils.finalize_symmetric_matrix` and stored
        in the attribute ``kG``. Messages are printed unless
        ``silent=True``.

        Parameters
        ----------
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.
        c : array-like or None, optional
            Ritz constants of the assembly, passed to the panels and to the
            stiffeners. :meth:`.BladeStiff1D.calc_kG` raises a
            ``NotImplementedError`` when it is not ``None``, and
            :meth:`.BladeStiff2D.calc_kG` does not use it. Default ``None``.

        Returns
        -------
        kG : sparse matrix
            The geometric stiffness matrix, of shape ``(size, size)``.

        """
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

        for s in self.bladestiff2ds:
            s.calc_kG(size=size, row0=row0, col0=col0, silent=True,
                       finalize=False, c=c)
            if s.flange is not None:
                row0 += s.flange.get_size()
                col0 += s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kG += s.kG


        kG = finalize_symmetric_matrix(kG)
        self.kG = kG

        msg('finished!', level=2, silent=silent)

        return kG


    def calc_kM(self, silent=False):
        r"""Calculate the mass matrix of the assembly

        The bay is rebuilt and its size computed with :meth:`.get_size`.
        The matrices of the panels and of the :class:`.BladeStiff1D`
        stiffeners are placed at row and column ``0``; those of the
        :class:`.BladeStiff2D` stiffeners at the Ritz constants of their
        flanges, as in :meth:`.calc_kC`. The sum is finalized with
        :func:`structsolve.sparseutils.finalize_symmetric_matrix` and stored
        in the attribute ``kM``. Messages are printed unless
        ``silent=True``.

        Parameters
        ----------
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.

        Returns
        -------
        kM : sparse matrix
            The mass matrix, of shape ``(size, size)``.

        """
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

        for s in self.bladestiff2ds:
            s.calc_kM(size=size, row0=row0, col0=col0, silent=True,
                    finalize=False)
            if s.flange is not None:
                row0 += s.flange.get_size()
                col0 += s.flange.get_size()
            #TODO summing up coo_matrix objects may be slow!
            kM += s.kM

        kM = finalize_symmetric_matrix(kM)
        self.kM = kM

        msg('finished!', level=2, silent=silent)

        return kM


    def _aero_parameters(self):
        r"""Piston theory parameters ``beta``, ``gamma`` and ``aeromu``

        When the attribute ``beta`` is ``None`` they are computed from the
        attributes ``Mach``, ``rho_air``, ``V``, ``speed_sound`` and, for
        the models that need the radius, ``r``; a ``ValueError`` is raised
        when ``Mach`` is ``None`` or lower than ``1``, and ``Mach == 1`` is
        replaced by ``1.0001`` in the attribute ``Mach``. Otherwise the
        attributes ``beta``, ``gamma`` and ``aeromu`` are returned, ``None``
        meaning ``0.`` for the last two.

        Returns
        -------
        beta, gamma, aeromu : float
            Parameters of the aerodynamic load, of the curvature correction
            and of the aerodynamic damping. When computed, ``gamma`` is
            ``0.`` for the models that do not need the radius.

        Notes
        -----
        With the Mach number `M`, the air density `\rho_{air}`, the air speed
        `V` and the speed of sound `a_\infty`:

        .. math::

            \beta = \frac{\rho_{air} V^2}{\sqrt{M^2 - 1}}, \qquad
            \gamma = \frac{\beta}{2 r \sqrt{M^2 - 1}}, \qquad
            \mu_a = \frac{\beta}{M a_\infty} \frac{M^2 - 2}{M^2 - 1}

        `\gamma` is the curvature correction of the piston theory for
        cylindrical shells, see :meth:`.Shell.calc_kA`, zero for the models
        without radius, and `\mu_a`, ``aeromu``, the coefficient of the
        aerodynamic damping, see :meth:`.Shell.calc_cA`. The linear piston
        theory holds for supersonic flows only, and `\beta` is singular at
        `M = 1`, hence the replacement of ``Mach == 1`` by ``1.0001``.

        TODO(author): give a source for the expression of ``aeromu``.

        """
        r = self.r if self.r is not None else 0.
        if self.beta is None:
            if self.Mach is None:
                raise ValueError('Mach number cannot be a NoneValue')
            elif self.Mach < 1:
                raise ValueError('Mach number must be >= 1')
            elif self.Mach == 1:
                self.Mach = 1.0001
            Mach = self.Mach
            beta = self.rho_air * self.V**2 / (Mach**2 - 1)**0.5
            #NOTE the curvature term exists only for cylindrical shells, see
            #     Shell.calc_kA()
            if panelmDB.db[self.model].get('requires_r', False):
                gamma = beta*1./(2.*r*(Mach**2 - 1)**0.5)
            else:
                gamma = 0.
            ainf = self.speed_sound
            aeromu = beta/(Mach*ainf)*(Mach**2 - 2)/(Mach**2 - 1)
        else:
            beta = self.beta
            gamma = self.gamma if self.gamma is not None else 0.
            aeromu = self.aeromu if self.aeromu is not None else 0.
        return beta, gamma, aeromu


    def calc_kA(self, silent=False):
        r"""Calculate the aerodynamic matrix of the assembly

        The bay is rebuilt and its size computed with :meth:`.get_size`.
        The parameters ``beta`` and ``gamma`` of the piston theory are the
        attributes ``beta`` and ``gamma`` of the bay when ``beta`` is given,
        otherwise they are computed from ``Mach``, ``rho_air``, ``V`` and,
        for the models that need the radius, ``r``; a ``ValueError`` is
        raised by ``_aero_parameters()`` when ``Mach`` is then ``None`` or
        lower than ``1``, and ``Mach == 1`` is replaced by ``1.0001`` in the
        attribute ``Mach``. The attributes ``flow``, ``beta`` and ``gamma``
        of each panel are set to those of the bay, and the matrices of the
        panels, :meth:`.Shell.calc_kA`, each one integrated over its own
        domain, are summed at row and column ``0``. The stiffeners do not
        contribute. The sum is made skew-symmetric with
        :func:`structsolve.sparseutils.make_skew_symmetric` and stored in
        the attribute ``kA``. Messages are printed unless ``silent=True``.

        Parameters
        ----------
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.

        Returns
        -------
        kA : scipy.sparse.csr_matrix
            The aerodynamic matrix, of shape ``(size, size)``.

        Notes
        -----
        The aerodynamic load of the linear piston theory and the parameters
        ``beta`` and ``gamma`` are those of :meth:`.Shell.calc_kA` and
        :meth:`._aero_parameters`. The skew-symmetrization keeps the diagonal
        and the upper triangle of the sum of the panels and replaces the
        lower triangle by the opposite of the upper one.

        TODO(author): the term `\gamma w` gives a symmetric matrix and the
        term `\beta w_{,x}` is skew-symmetric only when the boundary terms
        of `\int N_i N_{k,x} dx` vanish; is the skew-symmetrization intended,
        given that :meth:`.Shell.calc_kA` never symmetrizes the matrix?

        """
        self._rebuild()
        msg('Calculating kA... ', level=2, silent=silent)

        size = self.get_size()
        beta, gamma, aeromu = self._aero_parameters()

        # contributions from panels, each one integrated over its own domain
        #TODO summing up coo_matrix objects may be slow!
        kA = 0.
        for p in self.panels:
            p.flow = self.flow
            p.beta = beta
            p.gamma = gamma
            p.calc_kA(size=size, row0=0, col0=0, silent=True, finalize=False)
            kA += p.matrices['kA']

        assert np.any(np.isnan(kA.data)) == False
        assert np.any(np.isinf(kA.data)) == False
        kA = csr_matrix(make_skew_symmetric(kA))
        self.kA = kA

        msg('finished!', level=2, silent=silent)

        return kA


    def calc_cA(self, silent=False):
        r"""Calculate the aerodynamic damping matrix of the assembly

        The bay is rebuilt and its size computed with :meth:`.get_size`.
        The parameter ``aeromu`` of the piston theory is the attribute
        ``aeromu`` of the bay (``0.`` when ``None``) when ``beta`` is given,
        otherwise it is computed from ``Mach``, ``rho_air``, ``V`` and
        ``speed_sound``, see :meth:`.calc_kA`. The matrix is that of the
        first panel, :meth:`.Shell.calc_cA`, which is integrated over the
        whole bay; the stiffeners do not contribute. It is stored in the
        attribute ``cA``. Messages are printed unless ``silent=True``.

        Parameters
        ----------
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.

        Returns
        -------
        cA : sparse matrix
            The aerodynamic damping matrix, with complex values, of shape
            ``(size, size)``.

        Notes
        -----
        The damping term of the piston theory, with the coefficient
        ``aeromu`` of :meth:`._aero_parameters`, is proportional to the
        velocity of `w`, see :meth:`.Shell.calc_cA`, whose matrix is
        multiplied by the imaginary unit.

        TODO(author): state how ``cA`` enters the flutter eigenvalue
        problem, e.g. with the factor of the frequency of a harmonic motion.

        """
        self._rebuild()
        msg('Calculating cA... ', level=2, silent=silent)
        size = self.get_size()
        beta, gamma, aeromu = self._aero_parameters()

        #NOTE Shell.calc_cA() integrates analytically over the whole bay,
        #     regardless of the panel limits, so a single panel is used
        p = self.panels[0]
        cA = p.calc_cA(aeromu, size=size, silent=True, finalize=True)
        self.cA = cA

        msg('finished!', level=2, silent=silent)

        return cA


    def uvw_skin(self, c, xs=None, ys=None, gridx=300, gridy=300):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the
        :class:`.StiffPanelBay` object.

        Only the Ritz constants of the skin, the first ``dofs*m*n``, are
        used, with the function ``fuvw`` of the field module of the model.
        When ``xs`` or ``ys`` is ``None``, the field is computed on a grid of
        ``gridx`` points along `x`, from ``0`` to ``a``, and ``gridy``
        points along `y`, from ``0`` to ``b``, also stored in the attributes
        ``Xs`` and ``Ys``.

        Parameters
        ----------
        c : array-like
            The full set of Ritz constants, 1-D array of size
            :meth:`.get_size`.
        xs : np.ndarray, optional
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
            Length.
        ys : np.ndarray, optional
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
            Length, same shape as ``xs``.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field. Default ``300``.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field. Default ``300``.

        Returns
        -------
        out : tuple
            A tuple of ``np.ndarrays`` containing
            ``(u, v, w, phix, phiy)``, each one with the shape of ``xs``, or
            ``(gridy, gridx)`` for the default grid. The arrays are also
            stored in the attributes ``u``, ``v``, ``w``, ``phix`` and
            ``phiy`` of the :class:`.StiffPanelBay` object.

        Raises
        ------
        ValueError
            When the size of ``c`` is not :meth:`.get_size`.

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

        Only the flanges of the :class:`.BladeStiff2D` stiffeners have their
        own Ritz constants: those of the stiffener ``si`` follow the Ritz
        constants of the skin and of the flanges of the previous
        :class:`.BladeStiff2D` stiffeners. The field is computed with the
        function ``fuvw`` of the field module of the model of the flange.
        When ``xs`` or ``ys`` is ``None``, the field is computed on a grid of
        ``gridx`` points along `x`, from ``0`` to ``a``, and ``gridy``
        points along the width of the flange, also stored in the attributes
        ``Xs`` and ``Ys``.

        Parameters
        ----------
        c : array-like
            The full set of Ritz constants, 1-D array of size
            :meth:`.get_size`.
        si : int
            Stiffener index, in the attribute ``stiffeners``.
        region : str, optional
            Stiffener region ('base', 'flange' etc). Only ``'flange'`` is
            implemented. Default ``'flange'``.
        xs : np.ndarray, optional
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
            Length.
        ys : np.ndarray, optional
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
            Length, same shape as ``xs``.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field. Default ``300``.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field. Default ``300``.

        Returns
        -------
        out : tuple
            A tuple of ``np.ndarrays`` containing
            ``(u, v, w, phix, phiy)``, each one with the shape of ``xs``, or
            ``(gridy, gridx)`` for the default grid. The arrays are also
            stored in the attributes ``u``, ``v``, ``w``, ``phix`` and
            ``phiy`` of the :class:`.StiffPanelBay` object.

        Raises
        ------
        RuntimeError
            When the stiffener ``si`` is a :class:`.BladeStiff1D`; when
            ``region`` is ``'base'``; when the stiffener has no flange.
        ValueError
            When ``region`` is neither ``'flange'`` nor ``'base'``; when the
            size of ``c`` is not :meth:`.get_size`.

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

        # getting array position, only the flanges of the BladeStiff2D
        # stiffeners have their own Ritz constants, see get_size()
        for s in self.stiffeners[:si]:
            if isinstance(s, BladeStiff2D) and s.flange is not None:
                row_init += s.flange.get_size()
        s = stiff

        if region.lower() == 'flange':
            if stiff.flange is None:
                raise RuntimeError('This BladeStiff2D has no flange')
            bstiff = stiff.flange.b
            row_final = row_init + stiff.flange.get_size()
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

        Contour plot of a displacement field of the skin, computed by
        :meth:`.uvw_skin`.

        The displacement component ``vec`` is computed at the points of
        the grid and drawn with ``ax.contourf``, with `y` along the
        horizontal axis, which is inverted, and `x` along the vertical
        axis. When ``save`` is ``True`` and ``ax`` is not given, the figure
        is saved to ``filename`` and closed. The attributes ``u``, ``v``,
        ``w``, ``phix`` and ``phiy`` are restored afterwards to their
        values before the call, except those that were ``None``, which keep
        the computed field; the attributes ``Xs`` and ``Ys`` keep the grid,
        shifted in place by the displacements when ``deform_u=True``.
        Messages are printed unless ``silent=True``.

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

            Only the displacement components are implemented, the other
            values raise a ``ValueError``. Default ``'w'``.
        deform_u : bool, optional
            If ``True`` the contour plot will look deformed. The coordinates
            of the plot are then shifted by ``deform_u_sf`` times the
            displacements `u` and `v`. Default ``False``.
        deform_u_sf : float, optional
            The scaling factor used to deform the contour. Default ``100.``.
        invert_y : bool, optional
            Inverts the `y` axis of the plot. It may be used to match
            the coordinate system of the finite element models created
            using the ``desicos.abaqus`` module. Default ``False``.
        plot_type : int, optional
            For cylinders only ``4`` and ``5`` are valid.
            For cones all the following types can be used:

            - ``1``: concave up (with ``invert_y=False``) (default)
            - ``2``: concave down (with ``invert_y=False``)
            - ``3``: stretched closed
            - ``4``: stretched opened (`r \times y` vs. `a`)
            - ``5``: stretched opened (`y` vs. `a`)

            The value is not used by this method.
        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
            Default ``True``. Ignored when ``ax`` is given.
        dpi : int, optional
            Resolution of the saved file in dots per inch. Default ``400``.
        filename : str, optional
            The file name for the generated image file. If no value is given,
            ``'test.png'`` is used. Default ``''``.
        ax : AxesSubplot, optional
            When ``ax`` is given, the contour plot will be created inside it,
            and the figure is not saved. Default ``None``, a new figure is
            created.
        figsize : tuple, optional
            The figure size given by ``(width, height)``, used when ``ax`` is
            not given. Default ``(3.5, 2.)``.
        title : str, optional
            If any string is given it is added as title to the contour plot.
            Default ``''``.
        colorbar : bool, optional
            If a colorbar should be added to the contour plot. Default
            ``False``.
        cbar_nticks : int, optional
            Number of ticks added to the colorbar. Default ``2``.
        cbar_format : [ None | format string | Formatter object ], optional
            See the ``matplotlib.pyplot.colorbar`` documentation. Default
            ``None``.
        cbar_fontsize : int, optional
            Fontsize of the colorbar labels. Default ``10``.
        cbar_title : str, optional
            Colorbar title. If ``cbar_title == ''`` no title is added.
            Default ``''``.
        aspect : str, optional
            String that will be passed to the ``AxesSubplot.set_aspect()``
            method. Default ``'equal'``.
        clean : bool, optional
            Clean axes ticks, grids, spines etc. Default ``True``.
        texts : list of dict, optional
            Keyword arguments of ``ax.text``, one dictionary per text, with
            the position in the coordinates of the axes
            (``transform=ax.transAxes``). Default ``[]``.
        xs : np.ndarray, optional
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        ys : np.ndarray, optional
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field. Default ``300``.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field. Default ``300``.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).
            Default ``400``.
        vecmin : float, optional
            Minimum value for the contour scale (useful to compare with other
            results). If not specified it will be taken from the calculated
            field. Default ``None``.
        vecmax : float, optional
            Maximum value for the contour scale. If not specified it is taken
            from the calculated field. Default ``None``.
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.
            Default ``False``.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib object that can be used to modify the current plot
            if needed.

        Raises
        ------
        ValueError
            When ``vec`` is not one of the displacement components; when
            ``ax`` is given and is not a :class:`matplotlib.axes.Axes`.

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

        Contour plot of a displacement field of the flange of a
        :class:`.BladeStiff2D`, computed by :meth:`.uvw_stiffener`.

        The displacement component ``vec`` is computed at the points of
        the grid and drawn with ``ax.contourf``, with `y` along the
        horizontal axis, which is inverted, and `x` along the vertical
        axis. When ``save`` is ``True`` and ``ax`` is not given, the figure
        is saved to ``filename`` and closed. The attributes ``u``, ``v``,
        ``w``, ``phix`` and ``phiy`` are restored afterwards to their
        values before the call, except those that were ``None``, which keep
        the computed field; the attributes ``Xs`` and ``Ys`` keep the grid,
        shifted in place by the displacements when ``deform_u=True``.
        Messages are printed unless ``silent=True``.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the field contour.
        si : int
            Stiffener index.
        region : str, optional
            Stiffener region ('base', 'flange' etc). Only ``'flange'`` is
            implemented, see :meth:`.uvw_stiffener`. Default ``'flange'``.
        vec : str, optional
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``
            - Strain: ``'exx'``, ``'eyy'``, ``'gxy'``, ``'kxx'``, ``'kyy'``,
              ``'kxy'``, ``'gyz'``, ``'gxz'``
            - Stress: ``'Nxx'``, ``'Nyy'``, ``'Nxy'``, ``'Mxx'``, ``'Myy'``,
              ``'Mxy'``, ``'Qy'``, ``'Qx'``

            Only the displacement components are implemented, the other
            values raise a ``ValueError``. Default ``'w'``.
        deform_u : bool, optional
            If ``True`` the contour plot will look deformed. The coordinates
            of the plot are then shifted by ``deform_u_sf`` times the
            displacements `u` and `v`. Default ``False``.
        deform_u_sf : float, optional
            The scaling factor used to deform the contour. Default ``100.``.
        invert_y : bool, optional
            Inverts the `y` axis of the plot. It may be used to match
            the coordinate system of the finite element models created
            using the ``desicos.abaqus`` module. Default ``False``.
        plot_type : int, optional
            For cylinders only ``4`` and ``5`` are valid.
            For cones all the following types can be used:

            - ``1``: concave up (with ``invert_y=False``) (default)
            - ``2``: concave down (with ``invert_y=False``)
            - ``3``: stretched closed
            - ``4``: stretched opened (`r \times y` vs. `a`)
            - ``5``: stretched opened (`y` vs. `a`)

            The value is not used by this method.
        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
            Default ``True``. Ignored when ``ax`` is given.
        dpi : int, optional
            Resolution of the saved file in dots per inch. Default ``400``.
        filename : str, optional
            The file name for the generated image file. If no value is given,
            ``'test.png'`` is used. Default ``''``.
        ax : AxesSubplot, optional
            When ``ax`` is given, the contour plot will be created inside it,
            and the figure is not saved. Default ``None``, a new figure is
            created.
        figsize : tuple, optional
            The figure size given by ``(width, height)``, used when ``ax`` is
            not given. Default ``(3.5, 2.)``.
        title : str, optional
            If any string is given it is added as title to the contour plot.
            Default ``''``.
        colorbar : bool, optional
            If a colorbar should be added to the contour plot. Default
            ``False``.
        cbar_nticks : int, optional
            Number of ticks added to the colorbar. Default ``2``.
        cbar_format : [ None | format string | Formatter object ], optional
            See the ``matplotlib.pyplot.colorbar`` documentation. Default
            ``None``.
        cbar_fontsize : int, optional
            Fontsize of the colorbar labels. Default ``10``.
        cbar_title : str, optional
            Colorbar title. If ``cbar_title == ''`` no title is added.
            Default ``''``.
        aspect : str, optional
            String that will be passed to the ``AxesSubplot.set_aspect()``
            method. Default ``'equal'``.
        clean : bool, optional
            Clean axes ticks, grids, spines etc. Default ``True``.
        texts : list of dict, optional
            Keyword arguments of ``ax.text``, one dictionary per text, with
            the position in the coordinates of the axes
            (``transform=ax.transAxes``). Default ``[]``.
        xs : np.ndarray, optional
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        ys : np.ndarray, optional
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field. Default ``300``.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field. Default ``300``.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).
            Default ``400``.
        vecmin : float, optional
            Minimum value for the contour scale (useful to compare with other
            results). If not specified it will be taken from the calculated
            field. Default ``None``.
        vecmax : float, optional
            Maximum value for the contour scale. If not specified it is taken
            from the calculated field. Default ``None``.
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.
            Default ``False``.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib object that can be used to modify the current plot
            if needed.

        Raises
        ------
        ValueError
            When ``vec`` is not one of the displacement components; when
            ``ax`` is given and is not a :class:`matplotlib.axes.Axes`.

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


    def save(self, fname=None):
        r"""Save the :class:`StiffPanelBay` object to a zip file

        The panels and stiffeners are saved with the bay, see
        :mod:`panels.json_io`. The matrices are not stored. The object is not
        modified.

        Parameters
        ----------
        fname : str, path-like or file object, optional
            Name of the file, or a binary file object opened for writing,
            e.g. :class:`io.BytesIO`. By default the name stored in
            ``StiffPanelBay.name`` followed by the extension
            ``'.stiffpanelbay.zip'``.

        """
        if fname is None:
            fname = self.name + '.stiffpanelbay.zip'
        if not hasattr(fname, 'write'):
            msg('Saving StiffPanelBay to {0}'.format(fname))
        json_io.save(self, fname)


    def calc_fext(self, silent=False):
        r"""Calculates the external force vector `\{F_{ext}\}`

        The point forces of the skin, the attribute ``forces_skin``, and
        those of the flange of each :class:`.BladeStiff2D`, its attribute
        ``forces_flange``, each one a sequence ``(x, y, fx, fy, fz)`` of the
        coordinates and of the components along `x`, `y` and `z`, are
        projected on the approximation functions with the function ``fg`` of
        the field module of the model, using the first panel for the skin.
        The vector of the skin is followed by those of the flanges, in the
        order of the attribute ``bladestiff2ds``; the stiffeners without
        flange are skipped. The bay is not rebuilt, the attribute ``model``
        must be set. Messages are printed unless ``silent=True``.

        Parameters
        ----------
        silent : bool, optional
            A boolean to tell whether the msg messages should be printed.
            Default ``False``.

        Returns
        -------
        fext : np.ndarray
            The external force vector, 1-D array of size :meth:`.get_size`.

        Notes
        -----
        The external force vector is the work-equivalent of the point
        forces, the virtual work of each force `\{f\} = \{f_x, f_y, f_z\}^T`
        at `(x, y)` being `\delta\{c\}^T [N(x, y)]^T \{f\}`, where the rows
        of `[N]` are the approximation functions of `u`, `v` and `w` at the
        point, given by ``fg``. The rows of the rotations are multiplied by
        zero: no moment is applied.

        """
        msg('Calculating external forces...', level=2, silent=silent)
        dofs = panelmDB.db[self.model]['dofs']
        fg = panelmDB.db[self.model]['field'].fg

        # punctual forces on skin
        #NOTE fg() needs a Shell; the shape functions of any panel span the
        #     whole bay, with the boundary conditions of the bay
        size = dofs*self.m*self.n
        #NOTE fg() fills the rows of u, v, w, phix and phiy
        g = np.zeros((5, size), dtype=DOUBLE)
        fext_skin = np.zeros(size, dtype=DOUBLE)
        for i, force in enumerate(self.forces_skin):
            x, y, fx, fy, fz = force
            fg(g, x, y, self.panels[0])

            fpt = np.array([[fx, fy, fz, 0., 0.]])
            fext_skin += fpt.dot(g).ravel()

        fext = fext_skin
        # punctual forces on bladestiff2ds
        # flange
        for s in self.bladestiff2ds:
            if s.flange is None:
                continue
            fg_flange = panelmDB.db[s.flange.model]['field'].fg
            size = s.flange.get_size()
            g_flange = np.zeros((5, size), dtype=DOUBLE)
            fext_stiffener = np.zeros(size, dtype=DOUBLE)
            for i, force in enumerate(s.forces_flange):
                xf, yf, fx, fy, fz = force
                fg_flange(g_flange, xf, yf, s.flange)
                fpt = np.array([[fx, fy, fz, 0., 0.]])
                fext_stiffener += fpt.dot(g_flange).ravel()

            fext = np.concatenate((fext, fext_stiffener))

        msg('finished!', level=2, silent=silent)

        return fext
