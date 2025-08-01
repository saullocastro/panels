import numpy as np

from . import modelDB
from . legendre_gauss_quadrature import get_points_weights

def shell_fext(shell, inc, size, col0):
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

        - ``shell.point_pds`` : list of prescribed point displacements, each
          described with ``[x, y, ku*up, kv*vp, kw*wp]``. See
          :meth:`.Shell.add_point_pd`.

        - ``shell.point_pds_inc`` : similar to ``point_pds``, but this is
          affected by the load increment in nonlinear analyses.

        - ``shell.distr_pds`` : list of distributed prescribed displacements,
          each described with ``[x, None, ku*funcu, kv*funcv, kw*funcw]`` or
          ``[None, y, ku*funcu, kv*funcv, kw*funcw]`` where ``x`` or ``y`` are
          the variable over which the displacement is distributed. See
          :meth:`.Shell.add_distr_pd_fixed_x` or
          :meth:`.Shell.add_distr_pd_fixed_y`.

        - ``shell.distr_pds_inc`` : similar as ``distr_pds``, but this is
          affected by the load increment in nonlinear analyses.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell object.
    inc : float, optional
        Since this function is called during the non-linear analysis,
        ``inc`` will multiply the terms `\{{F_{ext}}_\lambda\}`.
        col0 = starting col in the global matrix ?????
    size : int or str, optional
        The size of the force vector. Can be the size of the total internal
        force vector of a multidomain assembly. When using a string, for
        example, if '+1' is given it will add 1 to the Shell`s size obtained
        by the :meth:`.Shell.get_size`
    col0 : int, optional
        Offset in a global internal force vector of an assembly.

    Returns
    -------
    fext : np.ndarray
        The external force vector.

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
            points = np.zeros(shell.m, dtype=np.float64)
            weights = np.zeros(shell.m, dtype=np.float64)
            get_points_weights(shell.m, points, weights)
            # integrating g(x,ycte)*s(x,ycte)*dx = sum(weight_i * ( (a/2) * g(xvar, ycte) * s(xvar, ycte) ))
            for xi, weight in zip(points, weights):
                xvar = (xi + 1)*shell.a/2
                fpt = np.array([[funcx(xvar), funcy(xvar), funcz(xvar), 0, 0]]) * inc_i
                fg(g, xvar, ycte, shell)
                fext[col0:col1] += weight * (shell.a/2) * fpt.dot(g).ravel()
        else:
            xcte = x
            # getting integration points and weights
            points = np.zeros(shell.n, dtype=np.float64)
            weights = np.zeros(shell.n, dtype=np.float64)
            get_points_weights(shell.n, points, weights)
            # integrating g(xcte,y)*s(xcte,y)*dy = sum(weight_i * ( (b/2) * g(xcte, yvar) * s(xcte, yvar) ))
            for eta, weight in zip(points, weights):
                yvar = (eta + 1)*shell.b/2
                fpt = np.array([[funcx(yvar), funcy(yvar), funcz(yvar), 0, 0]]) * inc_i
                fg(g, xcte, yvar, shell)
                fext[col0:col1] += weight * (shell.b/2) * fpt.dot(g).ravel()


    # %% Prescribed DISPLACEMENTS

    # %%% point prescribed displacements
    # - grouping point_pds and point_pds_inc
    point_pds = []
    for pd in shell.point_pds:
        # pd = [x, y, ku*up, kv*vp, kw*wp]
        # Adding each pd as an elem of a list, so now point_pds = nestes lists
        # len(point_pds) = no of pds
        point_pds.append(pd + [1.]) #NOTE adding inc = 1. as the last elem of the list
    for pd in shell.point_pds_inc:
        point_pds.append(pd + [inc])
    # - calculating
    for x, y, fx, fy, fz, inc_i in point_pds:
        # This modifies g (based on its memory address so nothing needs to be returned)
        fg(g, x, y, shell)
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
        # Finally: distr_pds = [x pos, y pos of applied displ, force x, y, z due to that displ, incremenent size] -- size: (6,)
        distr_pds.append(pd + [1.]) #NOTE adding inc = 1.
    for pd in shell.distr_pds_inc:
        distr_pds.append(pd + [inc])

    for x, y, funcu, funcv, funcw, inc_i in distr_pds:
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
            points = np.zeros(shell.m, dtype=np.float64)
            weights = np.zeros(shell.m, dtype=np.float64)
            get_points_weights(shell.m, points, weights)
                # Addresses of points and weights get passed -- ftn def uses pointers so changes the address so no return arg
            # integrating g(x,ycte)*s(x,ycte)*dx = sum(weight_i * ( (a/2) * g(xvar, ycte) * s(xvar, ycte) ))
            for xi, weight in zip(points, weights):
                xvar = (xi + 1)*shell.a/2
                fpt = np.array([[funcu(xvar), funcv(xvar), funcw(xvar), 0, 0]]) * inc_i # Bec funcu etc are ftns of x,y not xi,eta -- # (1x5)
                    # THis is k_i * u_i ... i = u,v,w
                    # funci = force due to that displ_i
                # fg is in terms of x and y not xi and eta
                fg(g, xvar, ycte, shell)
                # Essentially does this:
                    # g[ fu[i]*gu[j],    fv[i]*gv[j],    fw[i]*gw[j],    -(2/a)*fw_xi[i]*gw[j],   -(2/b)*fw[i]*gw_eta[j] ]
                fext[col0:col1] += weight * (shell.a/2) * fpt.dot(g).ravel()
                # same as np.dot()  -  ravel() to flatten it into a 1D array

        # disp along x = const
        else:
            xcte = x
            # getting integration points and weights
            points = np.zeros(shell.n, dtype=np.float64)
            weights = np.zeros(shell.n, dtype=np.float64)
            get_points_weights(shell.n, points, weights)
            for eta, weight in zip(points, weights):
                yvar = (eta + 1)*shell.b/2
                fpt = np.array([[funcu(yvar), funcv(yvar), funcw(yvar), 0, 0]]) * inc_i
                fg(g, xcte, yvar, shell)
                fext[col0:col1] += weight * (shell.b/2) * fpt.dot(g).ravel()

    return fext
