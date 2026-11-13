r"""Blade stiffener with a 2D flange (:class:`.BladeStiff2D`)

The class :class:`.BladeStiff2D` computes the stiffness, geometric stiffness
and mass matrices of a blade stiffener of a :class:`.StiffPanelBay`, whose
base and flange are :class:`.Shell` objects. The base uses the Ritz
constants of the skin of the bay, the flange has its own Ritz constants,
connected to the base, or to the skin without base, with the penalty
kernels of the stiffener model, ``bladestiff2d_clt_donnell`` or
``bladestiff2d_clt_sanders``, see :mod:`panels.stiffener.modelDB`.

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
    ``hb`` the thickness of the base, and the model of the base given by
    the stiffener model, see ``model``. When ``fstack`` is given, the flange
    is the :class:`.Shell` ``flange`` with the model
    ``'plate_clpt_donnell'``, the length ``bay.a``, the width ``bf`` and
    ``mf`` by ``nf`` terms. The flags of the boundary conditions of its
    ends ``x1`` and ``x2`` are copied from those of ``bay`` when the
    stiffener is created, each displacement of the flange taking the flags
    of the displacement of the skin it equals at the root of the blade:
    ``x1u, x1ur`` from ``bay.x1u, bay.x1ur`` (axial displacement),
    ``x1v, x1vr`` from ``bay.x1w, bay.x1wr`` (the in-plane displacement
    `v_f` of the flange along its width, normal to the skin) and ``x1w,
    x1wr`` from ``bay.x1v, bay.x1vr`` (the displacement `w_f` normal to the
    flange, along `y` of the skin), and likewise at ``x2``; all the flags
    of the edges ``y1`` and ``y2`` of the flange are ``1``, free. With the
    default flags of :class:`.StiffPanelBay` the ends of the flange have
    `u_f = u_{f,x} = v_f = w_f = w_{f,x} = 0`, `v_{f,x}` free. To give the
    flange other end conditions, e.g. for a stiffener that ends at the
    bay or runs through the mouse holes of the ribs, assign the flags of
    the attribute ``flange`` after the stiffener is created, see Examples.
    The constructor then calls
    ``_rebuild()``, which checks with ``assert`` that ``panel1`` and
    ``panel2`` have the same ``model``, ``m``, ``n`` and ``r``, sets the
    model of the base, rebuilds the
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
    model : str or None, optional
        Stiffener model, a key of :mod:`panels.stiffener.modelDB`, stored
        in the attribute ``model``: ``'bladestiff2d_clt_donnell'``, with the
        base ``'plate_clpt_donnell'`` on a flat bay and
        ``'cylshell_clpt_donnell'`` on a curved bay, or
        ``'bladestiff2d_clt_sanders'``, only on a curved bay, with the base
        ``'cylshell_clpt_sanders'``. Default ``None``, selected from the
        skin ``panel1`` when the stiffener is created:
        ``'bladestiff2d_clt_sanders'`` when the skin is curved, see
        :meth:`.Shell.is_curved`, and its model is not
        ``'cylshell_clpt_donnell'``, ``'bladestiff2d_clt_donnell'``
        otherwise.

    Raises
    ------
    ValueError
        When ``model`` is not a model of :class:`.BladeStiff2D` in
        :mod:`panels.stiffener.modelDB`, or is
        ``'bladestiff2d_clt_sanders'`` on a flat bay.

    Notes
    -----
    The base is a :class:`.Shell` with the approximation functions and the
    Ritz constants of the skin, whose laminate is offset by `-(h/2 +
    h_b/2)`, i.e. the base is perfectly bonded below the skin and follows
    its displacement field. The flange is a plate based on the classical
    laminated plate theory, normal to the base, with its own approximation
    functions and Ritz constants, free along its two edges `y_f = 0` and
    `y_f = b_f`. Its edge `y_f = 0` is connected to the skin along `y =
    y_s` with the penalty energy of :meth:`.calc_kC`,
    such that the local buckling of the flange is represented, unlike with
    :class:`.BladeStiff1D`. The base as a structural element follows [1]_,
    Sec. 2, Eqs. (11) and (12), where the flange is a one-dimensional
    Euler-Bernoulli beam, as in :class:`.BladeStiff1D`. The two-dimensional
    flange and its connection to the skin are a model of this package, not
    taken from a publication.

    The flange lies in the plane `x`-`z` of the skin, its width along `-z`
    and its normal along `y`. The connection makes its displacements at
    the root equal to those of the skin, `u_f = u`, `v_f = -w` and `w_f =
    v` along `y = y_s`, and so also their derivatives along `x`: `u_{f,x} =
    u_{,x}`, `v_{f,x} = -w_{,x}` and `w_{f,x} = v_{,x}`. The end flags of
    the flange are those of the skin under this correspondence: a flag
    ``0`` of the skin at an end constrains the root of the flange through
    the connection anyway, and a flag ``1`` of the skin is not contradicted
    by an end condition of the flange that its root would impose on the
    skin, e.g. `v_{f,x} = 0` where the skin has `w_{,x}` free, or `u_f = 0`
    where the skin has `u` free. The flags hold over the whole height of
    the flange: the rib or support that holds the skin also holds the
    blade. Where it does not, as for a stiffener that ends
    at the bay without being attached to the rib, or one that runs
    through a mouse hole of the rib, the flags of the displacements that
    are not held are set to ``1`` by the analyst; the root of the flange
    then still follows the skin through the connection.

    On a curved bay the base is a cylindrical shell with the kinematics of
    the skin, and the connection constrains the rotation of the normal of
    the skin about `x`, `-\phi_y`, to that of the flange: `-\phi_y =
    w_{,y} - v/r` with the Sanders kinematics [2]_
    (``'bladestiff2d_clt_sanders'``), `-\phi_y = w_{,y}` with the Donnell
    kinematics (``'bladestiff2d_clt_donnell'``). The flange remains a
    plate. A rigid rotation of the cylinder about its axis, `v = r \alpha`,
    `w = 0`, turns the flange by `\alpha`: it leaves the connection
    unstrained with Sanders' rotation, `w_{,y} - v/r = -\alpha`, but not
    with Donnell's, `w_{,y} = 0`.

    References
    ----------
    .. [1] Castro, S. G. P., Guimaraes, T. A. M., Rade, D. A., and Donadon,
       M. V., Flutter of stiffened composite panels considering the
       stiffener's base as a structural element, Composite Structures, 140,
       36-43, 2016. https://doi.org/10.1016/j.compstruct.2015.12.056
    .. [2] Sanders, J. L., Nonlinear theories for thin shells, Quarterly of
       Applied Mathematics, 21(1), 21-36, 1963.
       https://doi.org/10.1090/qam/147023

    Examples
    --------
    A blade stiffener with a plate flange is created by
    :meth:`.StiffPanelBay.add_bladestiff2d` between two panels of a bay; the
    flange adds its own degrees of freedom:

    >>> from structsolve import lb
    >>> from panels.stiffpanelbay import StiffPanelBay
    >>> laminaprop = (142.5e9, 8.7e9, 0.28, 5.1e9, 5.1e9, 5.1e9)
    >>> bay = StiffPanelBay()
    >>> bay.a, bay.b, bay.r = 0.4, 0.3, None
    >>> bay.stack = [0, 90, 90, 0]
    >>> bay.plyt, bay.laminaprop = 0.125e-3, laminaprop
    >>> bay.m, bay.n, bay.rho = 8, 8, 1600.
    >>> bay.model = 'plate_clpt_donnell'
    >>> p1 = bay.add_panel(y1=0., y2=0.15, Nxx=-1.)
    >>> p2 = bay.add_panel(y1=0.15, y2=0.3, Nxx=-1.)
    >>> s = bay.add_bladestiff2d(ys=0.15, bf=0.02, fstack=[0, 90, 90, 0],
    ...                          fplyt=0.125e-3, flaminaprop=laminaprop,
    ...                          mf=6, nf=4)
    >>> print(type(s).__name__, s.flange.get_size(), bay.get_size())
    BladeStiff2D 72 264
    >>> eigvals, eigvecs = lb(bay.calc_kC(silent=True),
    ...                       bay.calc_kG(silent=True),
    ...                       silent=True, num_eigvalues=1)
    >>> print('%.1f' % eigvals[0])
    707.3

    The ends of the flange take the flags of the bay. For a stiffener that
    runs through mouse holes of the ribs, which do not hold the blade over
    its height, the displacements `v_f` and `w_f` of the ends of the flange
    are freed:

    >>> for end in ('x1', 'x2'):
    ...     setattr(s.flange, end + 'v', 1.)
    ...     setattr(s.flange, end + 'w', 1.)
    >>> eigvals, eigvecs = lb(bay.calc_kC(silent=True),
    ...                       bay.calc_kG(silent=True),
    ...                       silent=True, num_eigvalues=1)
    >>> print('%.1f' % eigvals[0])
    678.7

    """
    def __init__(self, bay, rho, panel1, panel2, ys, bb, bf, bstack, bplyts,
            blaminaprops, fstack, fplyts, flaminaprops, mf=14, nf=11,
            model=None):
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
        self.model = model
        base_model = self._base_model()

        self.base = None
        if bstack is not None:
            y1 = self.ys - bb/2.
            y2 = self.ys + bb/2.
            h = 0.5*sum(self.panel1.plyts) + 0.5*sum(self.panel2.plyts)
            hb = sum(self.bplyts)
            self.base = Shell(a=bay.a, b=bay.b, r=bay.r,
                    stack=bstack, plyts=bplyts, laminaprops=blaminaprops,
                    rho=rho, m=bay.m, n=bay.n, offset=(-h/2.-hb/2.),
                    model=base_model,
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
                    x1u=bay.x1u, x1ur=bay.x1ur, x2u=bay.x2u, x2ur=bay.x2ur,
                    x1v=bay.x1w, x1vr=bay.x1wr, x2v=bay.x2w, x2vr=bay.x2wr,
                    x1w=bay.x1v, x1wr=bay.x1vr, x2w=bay.x2v, x2wr=bay.x2vr,
                    y1u=1., y1ur=1., y2u=1., y2ur=1.,
                    y1v=1., y1vr=1., y2v=1., y2vr=1.,
                    y1w=1., y1wr=1., y2w=1., y2wr=1.)

        self._rebuild()


    def _base_model(self):
        r"""Select the stiffener model and return the model of the base

        When the attribute ``model`` is ``None`` it is set to
        ``'bladestiff2d_clt_sanders'`` for a curved skin ``panel1``, see
        :meth:`.Shell.is_curved`, whose model is not
        ``'cylshell_clpt_donnell'``, and to ``'bladestiff2d_clt_donnell'``
        otherwise.

        Returns
        -------
        base_model : str
            Model of the :class:`.Shell` of the base, from the key ``'base
            models'`` of :mod:`panels.stiffener.modelDB`.

        Raises
        ------
        ValueError
            When ``model`` is not a model of :class:`.BladeStiff2D`, or has
            no base model for the bay, i.e. ``'bladestiff2d_clt_sanders'``
            on a flat bay.

        """
        curved = self.panel1.is_curved()
        if self.model is None:
            if curved and self.panel1.model != 'cylshell_clpt_donnell':
                self.model = 'bladestiff2d_clt_sanders'
            else:
                self.model = 'bladestiff2d_clt_donnell'
        valid = sorted(k for k in db if 'connections' in db[k])
        if self.model not in valid:
            raise ValueError("the models of BladeStiff2D are %s, got '%s'"
                             % (', '.join(valid), self.model))
        base_model = db[self.model]['base models'][
                'curved' if curved else 'flat']
        if base_model is None:
            raise ValueError("model '%s' requires a curved bay, with a radius"
                             % self.model)
        return base_model


    def _rebuild(self):
        assert self.panel1.model == self.panel2.model
        assert self.panel1.m == self.panel2.m
        assert self.panel1.n == self.panel2.n
        assert self.panel1.r == self.panel2.r
        base_model = self._base_model()
        if self.flange is not None:
            #NOTE the ply densities are needed by the numerical mass matrix
            self.flange.lam = laminated_plate(self.flange.stack, plyts=self.flange.plyts,
                                            laminaprops=self.flange.laminaprops,
                                            rhos=self.flange.rhos)
            self.flange.lam.calc_equivalent_properties()

        if self.base is not None:
            self.base.model = base_model
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
        the skin ``panel1`` without base, computed by the kernels of the
        stiffener model ``model``, ``fkCss`` (at row and column ``0``),
        ``fkCsf`` (at row ``0`` and column ``col0``) and ``fkCff`` (at
        ``row0`` and ``col0``) with the penalty constants of
        :func:`.calc_kt_kr` of type ``'ycte'``. The result is
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
                (w + v_f)^2 \right] + k_r \left( -\phi_y - w_{f,y} \right)^2
                dx

        where `u, v, w` are the displacements of the skin and `u_f, v_f,
        w_f` those of the flange in its own axes, `-\phi_y` the rotation of
        the normal of the skin about `x`, `w_{,y} - v/r` with the Sanders
        kinematics (``'bladestiff2d_clt_sanders'``) and `w_{,y}` with the
        Donnell kinematics (``'bladestiff2d_clt_donnell'``), and `w_{f,y}`
        the derivative along the width of the flange, i.e. the rotations of
        the normals of the skin and of the flange about `x` are the same. The
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

            mod = db[self.model]['connections']
            #NOTE the kernels of the Sanders kinematics take the radius
            geom = ((a, b, self.panel1.r)
                    if self.model == 'bladestiff2d_clt_sanders' else (a, b))
            kC += mod.fkCss(ktbf, krbf, self.ys, *geom, m, n,
                            bay.x1u, bay.x1ur, bay.x2u, bay.x2ur,
                            bay.x1v, bay.x1vr, bay.x2v, bay.x2vr,
                            bay.x1w, bay.x1wr, bay.x2w, bay.x2wr,
                            bay.y1u, bay.y1ur, bay.y2u, bay.y2ur,
                            bay.y1v, bay.y1vr, bay.y2v, bay.y2vr,
                            bay.y1w, bay.y1wr, bay.y2w, bay.y2wr,
                            size, 0, 0)
            bf = self.flange.b
            kC += mod.fkCsf(ktbf, krbf, self.ys, *geom, bf, m, n, self.flange.m, self.flange.n,
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

