import numpy as np

from . import modelDB
from . legendre_gauss_quadrature import get_points_weights

def shell_fext(shell, inc, size, col0):
    model = shell.model
    if not model in modelDB.db.keys():
        raise ValueError('{} is not a valid model option'.format(model))
    db = modelDB.db
    fg = db[model]['field'].fg # model is the model to be used, specified in the shell object

    if size is None:
        size = shell.get_size()
    elif isinstance(size, str):
        size = int(size) + shell.get_size()
    col1 = col0 + shell.get_size()
    g = np.zeros((5, shell.get_size()), dtype=np.float64)
    fext = np.zeros(size, dtype=np.float64)

    # %% point loads
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

    # %% distributed loads
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

    # %% point prescribed displacements
    # - grouping point_pds and point_pds_inc
    point_pds = []
    for pd in shell.point_pds:
        point_pds.append(pd + [1.]) #NOTE adding inc = 1.
    for pd in shell.point_pds_inc:
        point_pds.append(pd + [inc])
    # - calculating
    for x, y, fx, fy, fz, inc_i in point_pds:
        fg(g, x, y, shell)
        fpt = np.array([[fx, fy, fz, 0, 0]])*inc_i
        fext[col0:col1] += fpt.dot(g).ravel()



    # %% distributed prescribed displacements
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
        if x is None and y is None:
            raise ValueError('x and y cannot be None when defining displacements')
        if x is not None and y is not None:
            raise ValueError('One of x or y must be None when defining displacements')
        funcu = funcu if funcu is not None else lambda _: 0 # _ ignores ip var
        funcv = funcv if funcv is not None else lambda _: 0
        funcw = funcw if funcw is not None else lambda _: 0
        if x is None:
            ycte = y
            # getting integration points and weights
                # Integration over x so m terms
            points = np.zeros(shell.m, dtype=np.float64)
            weights = np.zeros(shell.m, dtype=np.float64)
            get_points_weights(shell.m, points, weights)
            # integrating g(x,ycte)*s(x,ycte)*dx = sum(weight_i * ( (a/2) * g(xvar, ycte) * s(xvar, ycte) ))
            for xi, weight in zip(points, weights):
                xvar = (xi + 1)*shell.a/2
                fpt = np.array([[funcu(xvar), funcv(xvar), funcw(xvar), 0, 0]]) * inc_i # Bec funcu etc are ftns of x,y not xi,eta -- # (1x5)
                fg(g, xvar, ycte, shell)  # ??????????????????
                fext[col0:col1] += weight * (shell.a/2) * fpt.dot(g).ravel() 
                # same as np.dot()  -  ravel() to flatten it into a 1D array
        else:
            xcte = x
            # getting integration points and weights
            points = np.zeros(shell.n, dtype=np.float64)
            weights = np.zeros(shell.n, dtype=np.float64)
            get_points_weights(shell.n, points, weights)
            # integrating g(xcte,y)*s(xcte,y)*dy = sum(weight_i * ( (b/2) * g(xcte, yvar) * s(xcte, yvar) ))
            for eta, weight in zip(points, weights):
                yvar = (eta + 1)*shell.b/2
                fpt = np.array([[funcu(yvar), funcv(yvar), funcw(yvar), 0, 0]]) * inc_i
                fg(g, xcte, yvar, shell)
                fext[col0:col1] += weight * (shell.b/2) * fpt.dot(g).ravel() 

    return fext
