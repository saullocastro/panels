r"""External force vector of a :class:`.Shell` (:mod:`panels.shell_fext`)

:func:`shell_fext` integrates the point loads, distributed loads, pressure
loads and the forces of the prescribed displacements of a shell into its
external force vector, the work of :meth:`.Shell.calc_fext`.
:func:`pressure_patches` and :func:`pressure_factor` prepare the pressure
loads, which are also used by the follower pressure terms of
:meth:`.Shell.calc_kCfollower` and :meth:`.Shell.calc_fext_follower`.

"""
import numpy as np

from . import modelDB
from scipy.special import roots_legendre

def shell_fext(shell, inc, size, col0, include_pds=True):
    r"""Calculate the external force vector

    The function reads the following attributes of the :class:`.Shell` object:

        - ``shell.point_loads`` : list of point loads, each described with ``[x,
          y, fx, fy, fz]``. See :meth:`.Shell.add_point_load`.

        - ``shell.point_loads_inc`` : similar to ``point_loads`` but this is
          affected by the load increment in nonlinear analyses.

        - ``shell.distr_loads`` : list of distributed loads, each described with
          ``[x, None, funcx, funcy, funcz]`` or ``[None, y, funcx, funcy,
          funcz]`` where ``x`` or ``y`` are the variable over which the load is
          distributed. See :meth:`.Shell.add_distr_load_fixed_x` or
          :meth:`.Shell.add_distr_load_fixed_y`.

        - ``shell.distr_loads_inc`` : similar to ``distr_loads``, but this is
          affected by the load increment in nonlinear analyses.

        - ``shell.pressure_loads`` : list of pressure loads, each described
          with ``[x1, x2, y1, y2, p, follower, zp]``, where ``p`` is a float
          or a function ``p(x, y)`` acting along `z` over the patch ``x1 <= x
          <= x2``, ``y1 <= y <= y2``, a ``None`` limit standing for the
          corresponding limit of :meth:`.Shell.integration_limits`,
          ``follower`` is ``None`` for a dead load or the truncation of the
          area vector of a follower load, ``'linear'`` or ``'quadratic'``,
          and ``zp`` is the position of the loaded surface, see
          :func:`.pressure_patches`. A follower load contributes here with
          its value in the undeformed state, equal to the dead load, see
          :meth:`.Shell.add_pressure_load`.

        - ``shell.pressure_loads_inc`` : similar to ``pressure_loads``, but
          this is affected by the load increment in nonlinear analyses.

        - ``shell.point_pds`` : list of prescribed point displacements, each
          described with ``[x, y, ku*up, kv*vp, kw*wp, z]``, ``z`` being the
          thickness coordinate of `u` and `v`. See
          :meth:`.Shell.add_point_pd`.

        - ``shell.point_pds_inc`` : similar to ``point_pds``, but this is
          affected by the load increment in nonlinear analyses.

        - ``shell.distr_pds`` : list of distributed prescribed displacements,
          each described with ``[x, None, ku*funcu, kv*funcv, kw*funcw, z]``
          or ``[None, y, ku*funcu, kv*funcv, kw*funcw, z]`` where ``x`` or
          ``y`` are the variable over which the displacement is distributed.
          See
          :meth:`.Shell.add_distr_pd_fixed_x` or
          :meth:`.Shell.add_distr_pd_fixed_y`.

        - ``shell.distr_pds_inc`` : similar as ``distr_pds``, but this is
          affected by the load increment in nonlinear analyses.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell object.
    inc : float
        Since this function is called during the non-linear analysis,
        ``inc`` will multiply the terms `\{{F_{ext}}_\lambda\}`.
        col0 = starting col in the global matrix ?????
        Dimensionless.
    size : int, str or None
        The size of the force vector. Can be the size of the total internal
        force vector of a multidomain assembly. When using a string, for
        example, if '+1' is given it will add 1 to the Shell`s size obtained
        by the :meth:`.Shell.get_size`. ``None`` is the size of the shell.
    col0 : int
        Offset in a global internal force vector of an assembly.
    include_pds : bool, optional
        Whether the forces of the prescribed displacements are included. The
        prescribed rotations, see :meth:`.Shell.add_point_pd`, are only
        available with ``MultiDomain(..., pd_method='null-space')``, which
        excludes these forces. Default ``True``.

    Returns
    -------
    fext : np.ndarray
        The external force vector, 1-D array of size ``size``, zero outside
        of ``fext[col0:col0 + shell.get_size()]``.

    Raises
    ------
    ValueError
        If ``shell.model`` is not a valid model, or if a distributed load or
        a distributed prescribed displacement has both or none of ``x`` and
        ``y`` set to ``None``.
    NotImplementedError
        If ``include_pds=True`` and rotations are prescribed, see
        :meth:`.Shell.has_rotation_pds`.

    Notes
    -----
    Each load enters with its virtual work on the approximation functions,
    the rows `\{g_u\}`, `\{g_v\}`, `\{g_w\}` of the function ``fg`` of the
    model: a point load gives `f_x \{g_u\} + f_y \{g_v\} + f_z \{g_w\}` at
    `(x, y)`; a distributed load at a fixed `y` gives `\int_0^a (f_x
    \{g_u\} + f_y \{g_v\} + f_z \{g_w\}) dx`, computed with ``shell.m``
    Gauss-Legendre points, and at a fixed `x` the integral along `y` with
    ``shell.n`` points; a pressure gives `\int \int p \{g_w\} dx dy` over its
    patch, see :func:`.pressure_patches`. The prescribed displacements
    imposed with the penalty method give the same vectors with the forces
    `k u_p`, the right-hand side of the penalty energy, see
    :meth:`.Shell.add_point_pd`, with the rows of `u(z)` and `v(z)` of
    :func:`.nullspace.pd_rows` in place of `\{g_u\}` and `\{g_v\}` for a
    thickness coordinate `z` different from zero.

    """

    model = shell.model
    if not model in modelDB.db.keys():
        raise ValueError('{} is not a valid model option'.format(model))
    db = modelDB.db
    fg = db[model]['field'].fg # model is the model (name) to be used, specified in the shell object
    # Field is in \panels\panels\models

    if size is None:
        size = shell.get_size()
    elif isinstance(size, str):
        size = int(size) + shell.get_size()
    col1 = col0 + shell.get_size()
    g = np.zeros((5, shell.get_size()), dtype=np.float64)
    fext = np.zeros(size, dtype=np.float64) # A vector bec size = single integer

    # %% Prescribed LOADS

    # %%% point loads
    # - grouping point_loads and point_loads_inc
    point_loads = []
    for load in shell.point_loads:
        point_loads.append(load + [1.]) #NOTE adding inc = 1.
    for load in shell.point_loads_inc:
        point_loads.append(load + [inc])
    # - calculating
    for x, y, fx, fy, fz, inc_i in point_loads:
        fg(g, x, y, shell)
        fpt = np.array([[fx, fy, fz, 0, 0]])*inc_i
        fext[col0:col1] += fpt.dot(g).ravel()

    # %%% distributed loads
    # - grouping distr_loads and distr_loads_inc
    distr_loads = []
    for load in shell.distr_loads:
        distr_loads.append(load + [1.]) #NOTE adding inc = 1.
    for load in shell.distr_loads_inc:
        distr_loads.append(load + [inc])

    for x, y, funcx, funcy, funcz, inc_i in distr_loads:
        if x is None and y is None:
            raise ValueError('x and y cannot be None when defining distributed loads')
        if x is not None and y is not None:
            raise ValueError('One of x or y must be None when defining distributed loads')
        funcx = funcx if funcx is not None else lambda _: 0
        funcy = funcy if funcy is not None else lambda _: 0
        funcz = funcz if funcz is not None else lambda _: 0
        if x is None:
            ycte = y
            # getting integration points and weights
            points, weights = roots_legendre(shell.m)
            # integrating g(x,ycte)*s(x,ycte)*dx = sum(weight_i * ( (a/2) * g(xvar, ycte) * s(xvar, ycte) ))
            for xi, weight in zip(points, weights):
                xvar = (xi + 1)*shell.a/2
                fpt = np.array([[funcx(xvar), funcy(xvar), funcz(xvar), 0, 0]]) * inc_i
                fg(g, xvar, ycte, shell)
                fext[col0:col1] += weight * (shell.a/2) * fpt.dot(g).ravel()
        else:
            xcte = x
            # getting integration points and weights
            points, weights = roots_legendre(shell.n)
            # integrating g(xcte,y)*s(xcte,y)*dy = sum(weight_i * ( (b/2) * g(xcte, yvar) * s(xcte, yvar) ))
            for eta, weight in zip(points, weights):
                yvar = (eta + 1)*shell.b/2
                fpt = np.array([[funcx(yvar), funcy(yvar), funcz(yvar), 0, 0]]) * inc_i
                fg(g, xcte, yvar, shell)
                fext[col0:col1] += weight * (shell.b/2) * fpt.dot(g).ravel()

    # %%% pressure loads
    # integrating p(x,y)*gw(x,y)*dx*dy over each patch with nx*ny
    # Gauss-Legendre points, gw being the row of g associated with w; the
    # follower loads enter with their value in the undeformed state
    if shell.pressure_loads or shell.pressure_loads_inc:
        xis, wxs = roots_legendre(shell.nx)
        etas, wys = roots_legendre(shell.ny)
    for patch in pressure_patches(shell, inc):
        x1, x2, y1, y2 = patch['limits']
        xs = x1 + (xis + 1)*(x2 - x1)/2
        ys = y1 + (etas + 1)*(y2 - y1)/2
        jac = (x2 - x1)/2*(y2 - y1)/2
        pnxny = patch['pnxny']
        for j, (yvar, wy) in enumerate(zip(ys, wys)):
            for i, (xvar, wx) in enumerate(zip(xs, wxs)):
                fg(g, xvar, yvar, shell)
                fext[col0:col1] += (wx*wy*jac*pnxny[i, j])*g[2]


    # %% Prescribed DISPLACEMENTS
    if not include_pds:
        return fext
    if shell.has_rotation_pds():
        raise NotImplementedError('Prescribed rotations are only available '
                'with MultiDomain(..., pd_method="null-space")')

    # %%% point prescribed displacements
    # - grouping point_pds and point_pds_inc
    point_pds = []
    for pd in shell.point_pds:
        # pd = [x, y, ku*up, kv*vp, kw*wp, z]
        # Adding each pd as an elem of a list, so now point_pds = nestes lists
        # len(point_pds) = no of pds
        point_pds.append(pd + [1.]) #NOTE adding inc = 1. as the last elem of the list
    for pd in shell.point_pds_inc:
        point_pds.append(pd + [inc])
    # - calculating
    for x, y, fx, fy, fz, z, inc_i in point_pds:
        # This modifies g (based on its memory address so nothing needs to be returned)
        fg(g, x, y, shell)
        if z != 0:
            #NOTE u(z) and v(z) combine the fields with the rotations and
            #     the slopes of w, see nullspace.pd_rows
            fext[col0:col1] += inc_i*(fz*g[2]
                                      + _uv_force(shell, fx, fy, x, y, z))
            continue
        # Essentially does this:
            # for j in range(n):
            #     for i in range(m):
            #         col = DOF*(j*m + i)
                    # g[0, col+0] = fu[i]*gu[j]
                    # g[1, col+1] = fv[i]*gv[j]
                    # g[2, col+2] = fw[i]*gw[j]
                    # g[3, col+2] = -(2/a)*fw_xi[i]*gw[j]
                    # g[4, col+2] = -(2/b)*fw[i]*gw_eta[j]
        fpt = np.array([[fx, fy, fz, 0, 0]])*inc_i
            # fx, fy and fz are defined in pd i.e. ku*up, kv*vp, kw*wp
        fext[col0:col1] += fpt.dot(g).ravel()
            # Equivalent to: k*w*S  --- returns contiguous flattened array (1D array)

    # %%% distributed prescribed displacements
    # - grouping distr_pds and distr_pds_inc
    distr_pds = []
    for pd in shell.distr_pds:
        # Adding the increment size to distr_pds
        # Finally: distr_pds = [x pos, y pos of applied displ, force x, y, z due to that displ, z coordinate of u and v, incremenent size] -- size: (7,)
        distr_pds.append(pd + [1.]) #NOTE adding inc = 1.
    for pd in shell.distr_pds_inc:
        distr_pds.append(pd + [inc])

    for x, y, funcu, funcv, funcw, z, inc_i in distr_pds:
        # For distr loads only 1 x or y is needed not both
        # funcu = Force = k * ftn of that disp
        if x is None and y is None:
            raise ValueError('x and y cannot be None when defining displacements')
        if x is not None and y is not None:
            raise ValueError('One of x or y must be None when defining displacements')
        funcu = funcu if funcu is not None else lambda _: 0 # _ ignores ip var
        funcv = funcv if funcv is not None else lambda _: 0
        funcw = funcw if funcw is not None else lambda _: 0

        # Disp along y = const
        if x is None:
            ycte = y
            # getting integration points and weights
                # Integration over x so m terms
            points, weights = roots_legendre(shell.m)
            # integrating g(x,ycte)*s(x,ycte)*dx = sum(weight_i * ( (a/2) * g(xvar, ycte) * s(xvar, ycte) ))
            for xi, weight in zip(points, weights):
                xvar = (xi + 1)*shell.a/2
                fg(g, xvar, ycte, shell)
                if z != 0:
                    fext[col0:col1] += weight*(shell.a/2)*inc_i*(
                        funcw(xvar)*g[2] + _uv_force(shell, funcu(xvar),
                                                     funcv(xvar), xvar, ycte, z))
                    continue
                fpt = np.array([[funcu(xvar), funcv(xvar), funcw(xvar), 0, 0]]) * inc_i # Bec funcu etc are ftns of x,y not xi,eta -- # (1x5)
                    # THis is k_i * u_i ... i = u,v,w
                    # funci = force due to that displ_i
                # fg is in terms of x and y not xi and eta
                # Essentially does this:
                    # g[ fu[i]*gu[j],    fv[i]*gv[j],    fw[i]*gw[j],    -(2/a)*fw_xi[i]*gw[j],   -(2/b)*fw[i]*gw_eta[j] ]
                fext[col0:col1] += weight * (shell.a/2) * fpt.dot(g).ravel()
                # same as np.dot()  -  ravel() to flatten it into a 1D array

        # disp along x = const
        else:
            xcte = x
            # getting integration points and weights
            points, weights = roots_legendre(shell.n)
            for eta, weight in zip(points, weights):
                yvar = (eta + 1)*shell.b/2
                fg(g, xcte, yvar, shell)
                if z != 0:
                    fext[col0:col1] += weight*(shell.b/2)*inc_i*(
                        funcw(yvar)*g[2] + _uv_force(shell, funcu(yvar),
                                                     funcv(yvar), xcte, yvar, z))
                    continue
                fpt = np.array([[funcu(yvar), funcv(yvar), funcw(yvar), 0, 0]]) * inc_i
                fext[col0:col1] += weight * (shell.b/2) * fpt.dot(g).ravel()

    return fext


def _uv_force(shell, fu, fv, x, y, z):
    r"""Force vector `f_u \{g_u(z)\} + f_v \{g_v(z)\}` of the prescribed
    in-plane displacements at the thickness coordinate ``z``, with the size
    of the Ritz constants of ``shell``, see :func:`.nullspace.pd_rows`"""
    from .multidomain.connections import nullspace
    xis = np.array([2*x/shell.a - 1])
    etas = np.array([2*y/shell.b - 1])
    out = 0.
    for comp, f in ((nullspace.U, fu), (nullspace.V, fv)):
        if f != 0:
            row = nullspace.pd_rows(shell, comp, xis, etas, z).toarray()[0]
            out = out + f*row
    return out


def pressure_factor(shell, zp):
    r"""Factor of a pressure acting on the surface ``z = zp``

    For a cylindrical shell, see :meth:`.Shell.is_curved`, a pressure ``p``
    on the surface ``z = zp`` is applied as ``p (1 + zp/r)`` per unit area
    of mid-surface. For plates the factor is ``1``.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell object, whose ``r`` is read when it is curved.
    zp : float
        Position along `z` of the loaded surface, a length.

    Returns
    -------
    factor : float
        ``1 + zp/r`` for a cylindrical shell with ``zp`` nonzero, otherwise
        ``1.``. Dimensionless.

    Notes
    -----
    A pressure ``p`` on the surface of radius ``r + zp`` has the resultant
    ``p (r + zp)`` per unit angle, i.e. ``p (1 + zp/r)`` per unit length of
    mid-surface arc. Its direction and change of area are those of the
    mid-surface, the shallow-shell approximation: the terms of relative
    order `z_p \kappa`, with `\kappa` the change of curvature, in the change
    of area and the moment of the in-plane components about the mid-surface
    are neglected, which are of relative order `n^2 z_p/r` in the load
    stiffness of a ring mode with `n` circumferential waves, e.g. about
    ``0.13`` for `n = 2`, `z_p = h/2` and `r/h = 15`.

    """
    if zp and shell.is_curved():
        return 1. + zp/shell.r
    return 1.


def pressure_patches(shell, inc, follower_only=False):
    r"""Patches and integration point values of the pressure loads

    The loads of ``shell.pressure_loads`` enter with the factor ``1`` and
    those of ``shell.pressure_loads_inc`` with ``inc``. A load whose patch,
    intersected with :meth:`.Shell.integration_limits`, is empty is
    skipped. The pressure is evaluated at the ``shell.nx`` by ``shell.ny``
    Gauss-Legendre points of the patch, calling ``p(x, y)`` when it is a
    function.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell object, whose ``pressure_loads`` and ``pressure_loads_inc``
        are read, see :func:`.shell_fext`.
    inc : float
        Load factor that multiplies the pressures of ``pressure_loads_inc``.
        Dimensionless.
    follower_only : bool, optional
        Return only the follower loads. Default ``False``.

    Returns
    -------
    patches : list of dict
        For each load whose patch intersects the integration domain of the
        shell, a dictionary with:

        - ``'limits'``: ``(x1, x2, y1, y2)``, the patch intersected with
          :meth:`.Shell.integration_limits`;
        - ``'pnxny'``: array of shape ``(shell.nx, shell.ny)`` with the
          pressure at the Gauss-Legendre points of the patch, multiplied by
          the load factor and by :func:`.pressure_factor`;
        - ``'follower'``: ``None``, ``'linear'`` or ``'quadratic'``.

    """
    loads = [(load, 1.) for load in shell.pressure_loads]
    loads += [(load, inc) for load in shell.pressure_loads_inc]
    if not loads:
        return []
    dx1, dx2, dy1, dy2 = shell.integration_limits()
    xis = roots_legendre(shell.nx)[0]
    etas = roots_legendre(shell.ny)[0]
    patches = []
    for load, inc_i in loads:
        x1, x2, y1, y2, p = load[:5]
        follower = load[5] if len(load) > 5 else None
        zp = load[6] if len(load) > 6 else 0.
        if follower_only and follower is None:
            continue
        # the patch, intersected with the integration domain of the shell
        x1 = dx1 if x1 is None else max(x1, dx1)
        x2 = dx2 if x2 is None else min(x2, dx2)
        y1 = dy1 if y1 is None else max(y1, dy1)
        y2 = dy2 if y2 is None else min(y2, dy2)
        if not (x1 < x2 and y1 < y2):
            continue
        xs = x1 + (xis + 1)*(x2 - x1)/2
        ys = y1 + (etas + 1)*(y2 - y1)/2
        if callable(p):
            pnxny = np.array([[p(x, y) for y in ys] for x in xs],
                             dtype=np.float64)
        else:
            pnxny = np.full((xs.shape[0], ys.shape[0]), p, dtype=np.float64)
        pnxny *= inc_i*pressure_factor(shell, zp)
        patches.append(dict(limits=(x1, x2, y1, y2),
                            pnxny=np.ascontiguousarray(pnxny),
                            follower=follower))
    return patches
