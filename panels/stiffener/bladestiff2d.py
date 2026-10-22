r"""Blade stiffener with a 2D flange (:class:`.BladeStiff2D`)

The class :class:`.BladeStiff2D` computes the stiffness, geometric stiffness
and mass matrices of a blade stiffener of a :class:`.StiffPanelBay`, whose
base and flange are :class:`.Shell` objects. The base uses the Ritz
constants of the skin of the bay, the flange has its own Ritz constants,
connected to the base, or to the skin without base, with the penalty
kernels of ``bladestiff2d_clt_donnell``, see :mod:`panels.stiffener.modelDB`.

"""

from composites import laminated_plate
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.logger import msg
from panels.multidomain.connections import calc_kt_kr
from .modelDB import db

class BladeStiff2D(object):
    r"""Blade Stiffener using 2D Formulation for Flange

    Blade-type of stiffener model using a 2D formulation for the flange and a
    2D formulation for the base (padup)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> panels
         Panel1      Panel2

    Both the flange and the base are optional. The stiffener's base is modeled
    using the same approximation functions as the skin, with the proper
    offset.

    Each stiffener has a constant `y_s` coordinate.

    When ``bstack`` is given, the base is the :class:`.Shell` ``base``
    spanning ``ys - bb/2 <= y <= ys + bb/2``, with the dimensions, number of
    terms and boundary conditions of ``bay`` and the offset ``-h/2 - hb/2``,
    ``h`` being the mean of the thicknesses of ``panel1`` and ``panel2`` and
    ``hb`` the thickness of the base. When ``fstack`` is given, the flange
    is the :class:`.Shell` ``flange`` with the model
    ``'plate_clpt_donnell'``, the length ``bay.a``, the width ``bf`` and
    ``mf`` by ``nf`` terms, with the flags of the boundary conditions
    ``x1u = x1ur = x2u = x2ur = 0``, ``x1v = x1vr = x2v = x2vr = 0``,
    ``x1w = x2w = 0``, ``x1wr = x2wr = 1`` and all the flags of the edges
    ``y1`` and ``y2`` equal to ``1``. The constructor then calls
    ``_rebuild()``, which checks with ``assert`` that ``panel1`` and
    ``panel2`` have the same ``model``, ``m``, ``n`` and ``r``, rebuilds the
    laminates of the base and of the flange and sets the attribute ``dpb``,
    ``h/2 + hb/2``, when there is a base. The attributes ``kC``, ``kG`` and
    ``kM`` are ``None``, and ``forces_flange``, the point forces
    ``(xf, yf, fx, fy, fz)`` applied to the flange and used by
    :meth:`.StiffPanelBay.calc_fext`, is an empty list.

    Parameters
    ----------
    bay : :class:`.StiffPanelBay`
        Panel bay of the stiffener, whose ``a``, ``b``, ``r``, ``m``, ``n``
        and boundary conditions are used.
    rho : float
        Material density of the base and of the flange, mass per unit
        volume.
    panel1, panel2 : :class:`.Shell`
        Panels of the bay on each side of the stiffener.
    ys : float
        Coordinate `y` of the stiffener, length.
    bb : float or None
        Width of the base, length. Used only when ``bstack`` is given.
    bf : float or None
        Width (height) of the flange, length. Used only when ``fstack`` is
        given.
    bstack : list or None
        Stacking sequence of the base, as ``stack`` of :class:`.Shell`.
        ``None`` for a stiffener without base.
    bplyts : list or None
        Thickness of each ply of the base, length.
    blaminaprops : list or None
        Lamina properties of each ply of the base, as ``laminaprop`` of
        :class:`.Shell`.
    fstack : list or None
        Stacking sequence of the flange, as ``stack`` of :class:`.Shell`.
        ``None`` for a stiffener without flange.
    fplyts : list or None
        Thickness of each ply of the flange, length.
    flaminaprops : list or None
        Lamina properties of each ply of the flange, as ``laminaprop`` of
        :class:`.Shell`.
    mf : int, optional
        Number of terms of the flange along `x`. Default ``14``.
    nf : int, optional
        Number of terms of the flange along its width. Default ``11``.

    Notes
    -----
    The base is a :class:`.Shell` with the approximation functions and the
    Ritz constants of the skin, whose laminate is offset by `-(h/2 +
    h_b/2)`, i.e. the base is perfectly bonded below the skin and follows
    its displacement field. The flange is a plate based on the classical
    laminated plate theory, normal to the base, with its own approximation
    functions and Ritz constants, simply supported at its ends `x = 0` and
    `x = a`, where only the rotation `w_{,x}` is free, and free along its
    two edges `y_f = 0` and `y_f = b_f`. Its edge `y_f = 0` is connected to
    the skin along `y = y_s` with the penalty energy of :meth:`.calc_kC`,
    such that the local buckling of the flange is represented, unlike with
    :class:`.BladeStiff1D`. The base as a structural element follows [1]_.

    TODO(author): confirm that the 2D flange model is also that of [1]_.

    References
    ----------
    .. [1] Castro, S. G. P., Guimaraes, T. A. M., Rade, D. A., and Donadon,
       M. V., Flutter of stiffened composite panels considering the
       stiffener's base as a structural element, Composite Structures, 2016.
       https://doi.org/10.1016/j.compstruct.2015.12.056

    """
    def __init__(self, bay, rho, panel1, panel2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops, mf=14, nf=11):
        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.rho = rho
        self.ys = ys
        self.bb = bb
        self.forces_flange = []

        self.bstack = bstack
        self.bplyts = bplyts
        self.blaminaprops = blaminaprops

        self.kC = None
        self.kM = None
        self.kG = None

        self.base = None
        if bstack is not None:
            y1 = self.ys - bb/2.
            y2 = self.ys + bb/2.
            h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
            hb = sum(self.bplyts)
            self.base = Shell(a=bay.a, b=bay.b, r=bay.r,
                    stack=bstack, plyts=bplyts, laminaprops=blaminaprops,
                    rho=rho, m=bay.m, n=bay.n, offset=(-h/2.-hb/2.),
                    x1u=bay.x1u, x1ur=bay.x1ur, x2u=bay.x2u, x2ur=bay.x2ur,
                    x1v=bay.x1v, x1vr=bay.x1vr, x2v=bay.x2v, x2vr=bay.x2vr,
                    x1w=bay.x1w, x1wr=bay.x1wr, x2w=bay.x2w, x2wr=bay.x2wr,
                    y1u=bay.y1u, y1ur=bay.y1ur, y2u=bay.y2u, y2ur=bay.y2ur,
                    y1v=bay.y1v, y1vr=bay.y1vr, y2v=bay.y2v, y2vr=bay.y2vr,
                    y1w=bay.y1w, y1wr=bay.y1wr, y2w=bay.y2w, y2wr=bay.y2wr,
                    y1=y1, y2=y2)

        self.flange = None
        if fstack is not None:
            self.flange = Shell(m=mf, n=nf, a=bay.a, b=bf, rho=rho,
                    stack=fstack, plyts=fplyts, laminaprops=flaminaprops,
                    model='plate_clpt_donnell',
                    x1u=0., x1ur=0., x2u=0., x2ur=0.,
                    x1v=0., x1vr=0., x2v=0., x2vr=0.,
                    x1w=0., x1wr=1., x2w=0., x2wr=1.,
                    y1u=1., y1ur=0., y2u=1., y2ur=0.,
                    y1v=1., y1vr=0., y2v=1., y2vr=0.,
                    y1w=1., y1wr=1., y2w=1., y2wr=1.)

        self._rebuild()


    def _rebuild(self):
        assert self.panel1.model == self.panel2.model
        assert self.panel1.m == self.panel2.m
        assert self.panel1.n == self.panel2.n
        assert self.panel1.r == self.panel2.r
        if self.flange is not None:
            #NOTE the ply densities are needed by the numerical mass matrix
            self.flange.lam = laminated_plate(self.flange.stack, plyts=self.flange.plyts,
                                            laminaprops=self.flange.laminaprops,
                                            rhos=self.flange.rhos)
            self.flange.lam.calc_equivalent_properties()

        if self.base is not None:
            h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
            hb = sum(self.bplyts)
            self.dpb = h/2. + hb/2.
            self.base.lam = laminated_plate(self.bstack, plyts=self.bplyts,
                                            laminaprops=self.blaminaprops,
                                            rhos=self.base.rhos,
                                            offset=(-h/2.-hb/2.))
            self.base.lam.calc_equivalent_properties()


    def calc_kC(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        r"""Calculate the linear constitutive stiffness matrix

        The stiffener is rebuilt, then the following matrices are summed:
        that of the base, :meth:`.Shell.calc_kC`, at row and column ``0``,
        the Ritz constants of the skin; that of the flange at ``row0`` and
        ``col0``; and the connection between the flange and the base, or
        the skin ``panel1`` without base, computed by the kernels ``fkCss``
        (at row and column ``0``), ``fkCsf`` (at row ``0`` and column
        ``col0``) and ``fkCff`` (at ``row0`` and ``col0``) with the penalty
        constants of :func:`.calc_kt_kr` of type ``'ycte'``. The result is
        stored in the attribute ``kC``; nothing is returned. Messages are
        printed unless ``silent=True``.

        Parameters
        ----------
        size : int or None, optional
            Size of the assembly. Default ``None``, accepted only for a
            stiffener without flange, the base then using its own size, see
            :meth:`.Shell.calc_kC`; the connection kernels require an
            integer.
        row0, col0 : int, optional
            Row and column of the assembly where the Ritz constants of the
            flange start. Default ``0``.
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.
        finalize : bool, optional
            Whether the matrix is completed from its upper triangle with
            :func:`structsolve.sparseutils.finalize_symmetric_matrix`; with
            ``False`` the sum of the upper triangles is stored. Default
            ``True``.

        Notes
        -----
        The flange is normal to the base, its local axes being rotated by 90
        degrees about the common axis `x`. The connection between the skin
        at `y = y_s` and the edge `\eta_f = -1` of the flange adds the
        penalty energy

        .. math::

            U = \frac{1}{2} \int_0^a k_t \left[ (u - u_f)^2 + (v - w_f)^2 +
                (w + v_f)^2 \right] + k_r \left( w_{,y} - w_{f,y} \right)^2
                dx

        where `u, v, w` are the displacements of the skin and `u_f, v_f,
        w_f` those of the flange in its own axes, and `w_{f,y}` the
        derivative along the width of the flange, i.e. the rotations of the
        normals of the skin and of the flange about `x` are the same. The
        kernels ``fkCss``, ``fkCsf`` and ``fkCff`` give the blocks of the
        skin, of the coupling and of the flange of this energy. The
        penalty constants of :func:`.calc_kt_kr` of type ``'ycte'`` are
        computed from the laminates of the base, or of ``panel1`` without
        base, and of the flange.

        The connection uses the approximation functions of the skin at `y =
        y_s`: the edge of the flange follows the mid-surface of the skin, and
        the offset of the base, `h/2 + h_b/2`, is not considered in the
        position of the flange.

        """
        self._rebuild()
        msg('Calculating kC... ', level=2, silent=silent)

        bay = self.bay
        a = bay.a
        b = bay.b
        m = bay.m
        n = bay.n

        kC = 0.
        if self.base is not None:
            kC += self.base.calc_kC(size=size, row0=0, col0=0, silent=True,
                    finalize=False)
        if self.flange is not None:
            kC += self.flange.calc_kC(size=size, row0=row0, col0=col0,
                    silent=True, finalize=False)

            # connectivity between stiffener'base and stiffener's flange
            if self.base is None:
                ktbf, krbf = calc_kt_kr(self.panel1, self.flange, 'ycte')
            else:
                ktbf, krbf = calc_kt_kr(self.base, self.flange, 'ycte')

            mod = db['bladestiff2d_clt_donnell']['connections']
            kC += mod.fkCss(ktbf, krbf, self.ys, a, b, m, n,
                            bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
                            bay.x1v, bay.x1vr, bay.x2v, bay.x2vr,
                            bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                            bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
                            bay.y1v, bay.y1vr, bay.y2v, bay.y2vr,
                            bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                            size, 0, 0)
            bf = self.flange.b
            kC += mod.fkCsf(ktbf, krbf, self.ys, a, b, bf, m, n, self.flange.m, self.flange.n,
                            bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
                            bay.x1v, bay.x1vr, bay.x2v, bay.x2vr,
                            bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                            bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
                            bay.y1v, bay.y1vr, bay.y2v, bay.y2vr,
                            bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                            self.flange.x1u, self.flange.x1ur, self.flange.x2u, self.flange.x2ur,
                            self.flange.x1v, self.flange.x1vr, self.flange.x2v, self.flange.x2vr,
                            self.flange.x1w, self.flange.x1wr, self.flange.x2w, self.flange.x2wr,
                            self.flange.y1u, self.flange.y1ur, self.flange.y2u, self.flange.y2ur,
                            self.flange.y1v, self.flange.y1vr, self.flange.y2v, self.flange.y2vr,
                            self.flange.y1w, self.flange.y1wr, self.flange.y2w, self.flange.y2wr,
                            size, 0, col0)
            kC += mod.fkCff(ktbf, krbf, a, bf, self.flange.m, self.flange.n,
                            self.flange.x1u, self.flange.x1ur, self.flange.x2u, self.flange.x2ur,
                            self.flange.x1v, self.flange.x1vr, self.flange.x2v, self.flange.x2vr,
                            self.flange.x1w, self.flange.x1wr, self.flange.x2w, self.flange.x2wr,
                            self.flange.y1u, self.flange.y1ur, self.flange.y2u, self.flange.y2ur,
                            self.flange.y1v, self.flange.y1vr, self.flange.y2v, self.flange.y2vr,
                            self.flange.y1w, self.flange.y1wr, self.flange.y2w, self.flange.y2wr,
                            size, row0, col0)

        if finalize:
            kC = finalize_symmetric_matrix(kC)
        self.kC = kC

        msg('finished!', level=2, silent=silent)


    def calc_kG(self, size=None, row0=0, col0=0, silent=False, finalize=True,
            c=None, NLgeom=False):
        """Calculate the linear geometric stiffness matrix

        The stiffener is rebuilt and only the flange contributes, with
        :meth:`.Shell.calc_kG` at ``row0`` and ``col0`` and the stress state
        given by the attributes ``Nxx``, ``Nyy`` and ``Nxy`` of the flange;
        the base adds no term. The result is stored in the attribute ``kG``;
        nothing is returned. Messages are printed unless ``silent=True``.

        Parameters
        ----------
        size : int or None, optional
            Size of the assembly. Default ``None``, the flange then using
            its own size, see :meth:`.Shell.calc_kG`.
        row0, col0 : int, optional
            Row and column of the assembly where the Ritz constants of the
            flange start. Default ``0``.
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.
        finalize : bool, optional
            Whether the matrix is completed from its upper triangle with
            :func:`structsolve.sparseutils.finalize_symmetric_matrix`.
            Default ``True``.
        c : array-like or None, optional
            Not used by this method. Default ``None``.
        NLgeom : bool, optional
            Passed to :meth:`.Shell.calc_kG` of the flange. Default
            ``False``.

        Notes
        -----
        The geometric stiffness is that of the flange for the membrane
        forces ``Nxx``, ``Nyy`` and ``Nxy`` prescribed in its attributes,
        see :meth:`.Shell.calc_kG`. The base and the skin under the
        stiffener add no geometric stiffness here: the load carried by the
        base is not considered.

        """
        self._rebuild()
        msg('Calculating kG... ', level=2, silent=silent)

        kG = 0.
        if self.base is not None:
            #TODO include kG for pad-up and Nxx load that arrives there
            pass
        if self.flange is not None:
            kG += self.flange.calc_kG(size=size, row0=row0, col0=col0,
                    silent=True, finalize=False, NLgeom=NLgeom)

        if finalize:
            kG = finalize_symmetric_matrix(kG)
        self.kG = kG

        msg('finished!', level=2, silent=silent)


    def calc_kM(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        """Calculate the mass matrix

        The stiffener is rebuilt, then the mass matrix of the base,
        :meth:`.Shell.calc_kM`, at row and column ``0``, the Ritz constants
        of the skin, and that of the flange at ``row0`` and ``col0`` are
        summed. The result is stored in the attribute ``kM``; nothing is
        returned. Messages are printed unless ``silent=True``.

        Parameters
        ----------
        size : int or None, optional
            Size of the assembly. Default ``None``, each part then using its
            own size, see :meth:`.Shell.calc_kM`.
        row0, col0 : int, optional
            Row and column of the assembly where the Ritz constants of the
            flange start. Default ``0``.
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.
        finalize : bool, optional
            Whether the matrix is completed from its upper triangle with
            :func:`structsolve.sparseutils.finalize_symmetric_matrix`.
            Default ``True``.

        """
        self._rebuild()
        msg('Calculating kM... ', level=2, silent=silent)

        kM = 0.
        if self.base is not None:
            kM += self.base.calc_kM(size=size, row0=0, col0=0, silent=True, finalize=False)
        if self.flange is not None:
            kM += self.flange.calc_kM(size=size, row0=row0, col0=col0, silent=True, finalize=False)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.kM = kM

        msg('finished!', level=2, silent=silent)

