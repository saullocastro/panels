
r"""Blade stiffener with a 1D flange (:class:`.BladeStiff1D`)

The class :class:`.BladeStiff1D` computes the stiffness, geometric stiffness
and mass matrices of a blade stiffener of a :class:`.StiffPanelBay`, whose
base is a :class:`.Shell` and whose flange is a beam described by the
compiled kernels of ``bladestiff1d_clt_donnell``, see
:mod:`panels.stiffener.modelDB`. Both use the Ritz constants of the skin of
the bay.

"""
from composites import laminated_plate
from structsolve.sparseutils import finalize_symmetric_matrix

from panels.shell import Shell
from panels.logger import msg
from . import modelDB


class BladeStiff1D(object):
    r"""Blade Stiffener using 1D Formulation for Flange

    Blade-type of stiffener model using a 1D formulation for the flange and a
    2D formulation for the padup (base)::


                 || --> flange       |
                 ||                  |-> stiffener
               ======  --> padup     |
      =========================  --> panels
         Panel1      Panel2

    Both the flange and the padup are optional, but one must exist.

    Each stiffener has a constant `y` coordinate.

    The constructor stores the input and calls ``_rebuild()``, which checks
    with ``assert`` that ``panel1`` and ``panel2`` have the same ``model``,
    ``m``, ``n`` and ``r``. When ``bstack`` is given, the base is the
    :class:`.Shell` ``base`` spanning ``ys - bb/2 <= y <= ys + bb/2``, with
    the dimensions, number of terms and boundary conditions of ``bay`` and
    the offset ``-h/2 - hb/2``, ``h`` being the mean of the thicknesses of
    ``panel1`` and ``panel2`` and ``hb`` the thickness of the base, and the
    area of the base is stored in ``Asb``. When ``fstack`` is given, the
    laminate of the flange is stored in ``flam`` and its properties in
    ``hf``, ``Asf``, ``E1``, ``S1``, ``F1`` and ``GJ``. The attributes ``hb`` and
    ``As`` are also set and, when ``bf`` is given, ``Iyy``, ``Jxx`` and
    ``dbf``, the distance ``bf/2 + hb + h/2`` from the mid-surface of the
    panels to the centroid of the flange, which are otherwise ``None``.
    The attribute ``model`` is
    ``'bladestiff1d_clt_donnell'``, and the attribute ``Fx``, the axial
    force of the flange used by :meth:`.calc_kG`, is ``None``.

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
        Width (height) of the flange, length. Required when ``fstack`` is
        given, ``None`` is accepted for a stiffener without flange.
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

    Notes
    -----
    The model is that of the blade stiffener of [1]_, with the stiffener's
    base as a structural element. The base is a :class:`.Shell` with the
    approximation functions and the Ritz constants of the skin, whose
    laminate is offset by `-(h/2 + h_b/2)`, i.e. the base is perfectly
    bonded below the skin and follows its displacement field. The flange is
    a beam along `x` at `y = y_s`, normal to the skin and below it, whose
    displacements are those of the skin along the line `y = y_s`: the
    cross-section of the flange does not deform, and the local buckling of
    the flange is not represented.

    The properties of the flange, a laminate of height `b_f` and thickness
    `h_f`, computed by ``_rebuild()``, are, with the reduced stiffnesses
    `\bar Q_{ij}` of each ply `k` in the axes of the laminate (``q11L``,
    ``q12L``, ``q16L``, ``q22L``, ``q26L``), its thickness `h_k` and the
    distance `y_k` of its mid-surface from the mid-plane of the flange
    laminate:

    .. math::

        E_1 = \sum_k h_k \left(\bar Q_{11} - \frac{\bar Q_{12}^2}{\bar
        Q_{22}}\right), \qquad
        S_1 = -\sum_k y_k h_k \left(\bar Q_{16} - \frac{\bar Q_{12} \bar
        Q_{26}}{\bar Q_{22}}\right), \qquad
        F_1 = \frac{b_f^2}{12} E_1

    The reduced stiffnesses are those of a beam whose plies are free along
    the height `s` of the flange, `\sigma_2 = 0`: eliminating `\varepsilon_2
    = -(\bar Q_{12} \varepsilon_1 + \bar Q_{26} \gamma_{12})/\bar Q_{22}`
    from `\sigma_1 = \bar Q_{11} \varepsilon_1 + \bar Q_{12} \varepsilon_2 +
    \bar Q_{16} \gamma_{12}` gives

    .. math::

        \sigma_1 = \left(\bar Q_{11} - \frac{\bar Q_{12}^2}{\bar Q_{22}}
        \right) \varepsilon_1 + \left(\bar Q_{16} - \frac{\bar Q_{12} \bar
        Q_{26}}{\bar Q_{22}}\right) \gamma_{12}

    the first coefficient entering `E_1` and the second, the coupling of
    the axial strain with the shear strain of the twist, `S_1`, as in Eq.
    (16) of [1]_.

    `E_1` being the axial stiffness per unit height of the flange, and the
    section properties `I_{yy} = h_f b_f^3/12` and `J_{xx} = h_f b_f^3/12 +
    b_f h_f^3/12`, the polar moment of area, which is not used by the
    kernels. The distance from the mid-surface of the skin to the centroid
    of the flange is `d_f = b_f/2 + h_b + h/2`.

    The torsional stiffness of the flange is that of a thin laminated strip
    under uniform (Saint-Venant) torsion:

    .. math::

        G J = 4 b_f D_{66}

    with `D_{66}` of the flange laminate about its own mid-plane, ``flam``.
    Twisted at the rate `\theta_{,x}`, the strip, `h_f \ll b_f`, has with
    respect to its mid-plane the displacement `\tilde w = -\theta_{,x} x
    s`, `s` along the height and `\tilde w` along the thickness, whose
    curvatures are `\kappa_{xx} = -\tilde w_{,xx} = 0`, `\kappa_{ss} = 0`
    and `\kappa_{xs} = -2 \tilde w_{,xs} = 2 \theta_{,x}`, such that the
    bending energy per unit length of the strip is

    .. math::

        U = b_f \frac{1}{2} D_{66} \kappa_{xs}^2 = \frac{1}{2} \left(4 b_f
        D_{66}\right) \theta_{,x}^2 = \frac{1}{2} G J \theta_{,x}^2

    For an isotropic flange `D_{66} = G h_f^3/12` and `G J = G b_f
    h_f^3/3`, the Saint-Venant constant `J = b_f h_f^3/3` of a thin
    rectangle, smaller than the polar moment `J_{xx}`, which overestimates
    the torsional stiffness of open sections. The couplings `D_{16}` and
    `D_{26}`, and the effect of the free edges of the strip, of relative
    order `h_f/b_f`, are neglected.

    The distances `y_k` are measured from the mid-plane of the flange
    laminate, at the mid-surface of each ply, as `y_{ply}` of Eq. (16) of
    [1]_, which does not state the origin: `S_1` couples the axial strain
    with the shear strain of the twist, which varies linearly through the
    thickness of the flange about its mid-plane, i.e. it is a coupling of the
    type `B` of the flange laminate, zero for a symmetric flange, as `F_1`
    is taken about the centroid of the flange. Measured from the first face,
    `S_1` would contain the spurious term `-(h_f/2) \sum_k h_k (\bar Q_{16} -
    \bar Q_{12} \bar Q_{26}/\bar Q_{22})`.

    References
    ----------
    .. [1] Castro, S. G. P., Guimaraes, T. A. M., Rade, D. A., and Donadon,
       M. V., Flutter of stiffened composite panels considering the
       stiffener's base as a structural element, Composite Structures, 2016.
       https://doi.org/10.1016/j.compstruct.2015.12.056

    """
    def __init__(self, bay, rho, panel1, panel2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops):
        self.bay = bay
        self.panel1 = panel1
        self.panel2 = panel2
        self.model = 'bladestiff1d_clt_donnell'
        self.rho = rho
        self.ys = ys
        self.bb = bb
        self.hb = 0.
        self.bf = bf
        self.hf = 0.

        self.bstack = bstack
        self.bplyts = bplyts
        self.blaminaprops = blaminaprops
        self.base = None
        self.fstack = fstack
        self.fplyts = fplyts
        self.flaminaprops = flaminaprops
        self.flam = None

        self.As = None
        self.Asb = None
        self.Asf = None
        self.Jxx = None
        self.Iyy = None
        self.GJ = None

        self.Fx = None

        self._rebuild()


    def _rebuild(self):
        assert self.panel1.model == self.panel2.model
        assert self.panel1.m == self.panel2.m
        assert self.panel1.n == self.panel2.n
        assert self.panel1.r == self.panel2.r

        if self.fstack is not None:
            if self.bf is None:
                raise ValueError('The flange width bf must be given with fstack')
            self.hf = sum(self.fplyts)
            self.Asf = self.bf*self.hf
            #NOTE the density of the flange plies gives the mass of the flange
            rho = self.rho if self.rho is not None else 0.
            self.flam = laminated_plate(self.fstack, plyts=self.fplyts,
                                        laminaprops=self.flaminaprops,
                                        rhos=[rho for _ in self.fstack])
            self.flam.calc_equivalent_properties()

        h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
        hb = 0.
        bay = self.bay
        if self.bstack is not None:
            hb = sum(self.bplyts)
            y1 = self.ys - self.bb/2.
            y2 = self.ys + self.bb/2.
            self.base = Shell(a=bay.a, b=bay.b, r=bay.r,
                    stack=self.bstack, plyts=self.bplyts,
                    rho=self.rho, m=bay.m, n=bay.n,
                    laminaprops=self.blaminaprops, offset=(-h/2.-hb/2.),
                    x1u=bay.x1u, x1ur=bay.x1ur, x2u=bay.x2u, x2ur=bay.x2ur,
                    x1v=bay.x1v, x1vr=bay.x1vr, x2v=bay.x2v, x2vr=bay.x2vr,
                    x1w=bay.x1w, x1wr=bay.x1wr, x2w=bay.x2w, x2wr=bay.x2wr,
                    y1u=bay.y1u, y1ur=bay.y1ur, y2u=bay.y2u, y2ur=bay.y2ur,
                    y1v=bay.y1v, y1vr=bay.y1vr, y2v=bay.y2v, y2vr=bay.y2vr,
                    y1w=bay.y1w, y1wr=bay.y1wr, y2w=bay.y2w, y2wr=bay.y2wr,
                    y1=y1, y2=y2)
            self.Asb = self.bb*hb
        self.hb = hb

        #NOTE the properties of the flange are not needed without flange,
        #     whose width bf may then be None
        if self.bf is not None:
            #TODO check offset effect on curved panels
            self.dbf = self.bf/2. + hb + h/2.
            self.Iyy = self.hf*self.bf**3/12.
            self.Jxx = self.hf*self.bf**3/12. + self.bf*self.hf**3/12.
        else:
            self.dbf = self.Iyy = self.Jxx = None

        Asb = self.Asb if self.Asb is not None else 0.
        Asf = self.Asf if self.Asf is not None else 0.
        self.As = Asb + Asf

        if self.fstack is not None:
            self.E1 = 0
            #E3 = 0
            self.S1 = 0
            #NOTE y of the mid-surface of each ply from the mid-plane of the
            #     flange laminate, see the notes of the class
            yply = self.flam.plies[0].h/2. - self.hf/2.
            for i, ply in enumerate(self.flam.plies):
                if i > 0:
                    yply += self.flam.plies[i-1].h/2. + self.flam.plies[i].h/2.
                self.E1 += ply.h*(ply.q11L - ply.q12L**2/ply.q22L)
                self.S1 += -yply*ply.h*(ply.q16L - ply.q12L*ply.q26L/ply.q22L)

            self.F1 = self.bf**2/12.*self.E1
            # Saint-Venant torsion of a thin laminated strip
            self.GJ = 4.*self.bf*self.flam.D66


    def calc_kC(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        r"""Calculate the linear constitutive stiffness matrix

        The stiffener is rebuilt, then the matrix of the base,
        :meth:`.Shell.calc_kC`, and that of the flange, computed by the
        kernel ``fkCf``, are summed, both at ``row0`` and ``col0``. The
        result is stored in the attribute ``kC``; nothing is returned.
        Messages are printed unless ``silent=True``.

        Parameters
        ----------
        size : int or None, optional
            Size of the assembly. Default ``None``, accepted only for a
            stiffener without flange, the base then using its own size, see
            :meth:`.Shell.calc_kC`; the kernel of the flange requires an
            integer.
        row0, col0 : int, optional
            Row and column of the assembly where the Ritz constants of the
            skin start. Default ``0``.
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.
        finalize : bool, optional
            Whether the matrix is completed from its upper triangle with
            :func:`structsolve.sparseutils.finalize_symmetric_matrix`; with
            ``False`` the sum of the upper triangles is stored. Default
            ``True``.

        Notes
        -----
        The base is a :class:`.Shell` whose laminate is offset from the
        mid-surface of the skin, see :meth:`.Shell.calc_kC`. The strain
        energy of the flange, integrated along `x` at `y = y_s` with the
        approximation functions of the skin, is

        .. math::

            U_f = \frac{b_f}{2} \int_0^a \{\varepsilon_f\}^\top [k_{sf}]
            \{\varepsilon_f\} dx, \qquad
            \{\varepsilon_f\} = \begin{Bmatrix} u_{,x} \\ -w_{,xx} \\
            -w_{,xy} \end{Bmatrix}, \qquad
            [k_{sf}] = \begin{bmatrix}
                E_1 & -d_f E_1 & S_1 \\
                -d_f E_1 & F_1 + d_f^2 E_1 & -d_f S_1 \\
                S_1 & -d_f S_1 & G J/b_f
            \end{bmatrix}

        with the properties of :class:`.BladeStiff1D`, i.e. `u_{,x} + d_f
        w_{,xx}` is the axial strain at the centroid of the flange, below the
        skin, with the axial stiffness `b_f E_1`, `b_f F_1 = E_1 b_f^3/12` is
        the bending stiffness of the flange about its own centroid, and
        `-w_{,xy}` is the twist rate of the flange, which rotates with the
        skin about `x`, the rotation being `w_{,y}`, such that the torsion
        term is `\frac{1}{2} \int_0^a G J w_{,xy}^2 dx`, with the torsional
        stiffness `G J = 4 b_f D_{66}` of :class:`.BladeStiff1D`, passed to
        the kernel ``fkCf`` as `G J/b_f`.

        """
        self._rebuild()
        msg('Calculating kC... ', level=2, silent=silent)

        kC = 0.
        if self.base is not None:
            kC += self.base.calc_kC(size=size, row0=row0, col0=col0,
                    silent=True, finalize=False)
        if self.flam is not None:
            mod = modelDB.db[self.model]['matrices']
            bay = self.bay
            #NOTE the kernel multiplies the torsion entry by bf
            kC += mod.fkCf(self.ys, bay.a, bay.b, self.bf, self.dbf, self.E1, self.F1,
                           self.S1, self.GJ/self.bf, bay.m, bay.n,
                           bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
                           bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                           bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
                           bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                           size=size, row0=row0, col0=col0)

        if finalize:
            kC = finalize_symmetric_matrix(kC)
        self.kC = kC

        msg('finished!', level=2, silent=silent)


    def calc_kG(self, size=None, row0=0, col0=0, silent=False, finalize=True,
            c=None):
        r"""Calculate the linear geometric stiffness matrix

        The stiffener is rebuilt and only the flange contributes, with the
        kernel ``fkGf`` and the axial force of the attribute ``Fx`` (``0.``
        when ``Fx`` is ``None``); the base adds no term. The result is
        stored in the attribute ``kG``; nothing is returned. Messages are
        printed unless ``silent=True``.

        Parameters
        ----------
        size : int or None, optional
            Size of the assembly. Default ``None``; the kernel of the flange
            requires an integer.
        row0, col0 : int, optional
            Row and column of the assembly where the Ritz constants of the
            skin start. Default ``0``.
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.
        finalize : bool, optional
            Whether the matrix is completed from its upper triangle with
            :func:`structsolve.sparseutils.finalize_symmetric_matrix`.
            Default ``True``.
        c : None, optional
            Ritz constants of a numerical geometric stiffness matrix, not
            implemented. Default ``None``, the only value accepted.

        Raises
        ------
        NotImplementedError
            When ``c`` is not ``None``.

        Notes
        -----
        The geometric stiffness of the flange comes from the energy of its
        axial force `F_x` (``Fx``), integrated along `x` at `y = y_s`:

        .. math::

            U_{G} = \frac{1}{2} \int_0^a F_x w_{,x}^2 dx

        The base adds no geometric stiffness: the whole axial load of the
        stiffener is assumed to be carried by the flange, through ``Fx``.

        """
        #TODO
        if c is not None:
            raise NotImplementedError('numerical kG not implemented')

        self._rebuild()
        msg('Calculating kG... ', level=2, silent=silent)

        kG = 0.
        if self.base is not None:
            # TODO include kG for padup
            #      now it is assumed that all the load goes to the flange
            pass
        if self.flam is not None:
            Fx = self.Fx if self.Fx is not None else 0.
            mod = modelDB.db[self.model]['matrices']
            bay = self.bay
            kG += mod.fkGf(self.ys, Fx, bay.a, bay.b, self.bf, bay.m, bay.n,
                             bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                             bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                             size, row0, col0)

        if finalize:
            kG = finalize_symmetric_matrix(kG)
        self.kG = kG

        msg('finished!', level=2, silent=silent)


    def calc_kM(self, size=None, row0=0, col0=0, silent=False, finalize=True):
        r"""Calculate the mass matrix

        The stiffener is rebuilt, then the mass matrix of the base,
        :meth:`.Shell.calc_kM`, and that of the flange, computed by the
        kernel ``fkMf`` with the mass per unit area ``flam.intrho`` of the
        flange laminate, ``hb``, ``hf``, ``bf``, ``dbf`` and the mean
        thickness of ``panel1`` and ``panel2``, are summed, both at
        ``row0`` and ``col0``. The result is stored in the attribute ``kM``;
        nothing is returned. Messages are printed unless ``silent=True``,
        also those of the base.

        Parameters
        ----------
        size : int or None, optional
            Size of the assembly. Default ``None``, accepted only for a
            stiffener without flange, the base then using its own size; the
            kernel of the flange requires an integer.
        row0, col0 : int, optional
            Row and column of the assembly where the Ritz constants of the
            skin start. Default ``0``.
        silent : bool, optional
            Whether the messages are suppressed. Default ``False``.
        finalize : bool, optional
            Whether the matrix is completed from its upper triangle with
            :func:`structsolve.sparseutils.finalize_symmetric_matrix`.
            Default ``True``.

        Notes
        -----
        The kinetic energy of the flange is integrated through its height,
        its in-plane displacements at the centroid being `u - d_f \phi_x` and
        `v - d_f \phi_y`, with the rotations `\phi_x = -w_{,x}` and `\phi_y
        = -w_{,y}` of the skin at `y = y_s`, see the kernel ``fkMf`` of
        ``bladestiff1d_clt_donnell``, whose coupling terms are `-d_f` instead
        of the `-2 d_f` of Eq. (26) of [1]_, the centroid being at `z =
        -d_f`. The mass matrix of the base is that of a :class:`.Shell` with
        the offset `-(h/2 + h_b/2)`, see :meth:`.Shell.calc_kM`, whose
        laminate is centred at `z = -(h/2 + h_b/2)`, below the skin as the
        flange, such that the coupling terms of `u` and `\phi_x` of the base
        and of the flange have the same sign, `I_1 < 0` and `-d_f < 0`.

        References
        ----------
        .. [1] Castro, S. G. P., Guimaraes, T. A. M., Rade, D. A., and
           Donadon, M. V., Flutter of stiffened composite panels considering
           the stiffener's base as a structural element, Composite
           Structures, 2016. https://doi.org/10.1016/j.compstruct.2015.12.056

        """
        self._rebuild()
        msg('Calculating kM... ', level=2, silent=silent)

        mod = modelDB.db[self.model]['matrices']

        kM = 0.
        if self.base is not None:
            kM += self.base.calc_kM(size=size, row0=row0, col0=col0, silent=silent, finalize=False)
        if self.flam is not None:
            bay = self.bay
            h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
            #NOTE the kernel multiplies rho by hf, the mean density of the
            #     flange laminate gives its mass per unit area, flam.intrho
            kM += mod.fkMf(self.ys, self.flam.intrho/self.hf, h, self.hb, self.hf, bay.a, bay.b,
                           self.bf, self.dbf, bay.m, bay.n,
                           bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
                           bay.x1v, bay.x1vr, bay.x2v, bay.x2vr,
                           bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                           bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
                           bay.y1v, bay.y1vr, bay.y2v, bay.y2vr,
                           bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                           size=size, row0=row0, col0=col0)

        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.kM = kM

        msg('finished!', level=2, silent=silent)

