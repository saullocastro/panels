import numpy as np

from . import modelDB
from . legendre_gauss_quadrature import get_points_weights

def shell_fext(shell, inc, size, col0):
    model = shell.model
    if not model in modelDB.db.keys():
        raise ValueError('{} is not a valid model option'.format(model))
    db = modelDB.db
    fg = db[model]['field'].fg

    if size is None:
        size = shell.get_size()
    elif isinstance(size, str):
        size = int(size) + shell.get_size()
    col1 = col0 + shell.get_size()
    g = np.zeros((5, shell.get_size()), dtype=np.float64)
    fext = np.zeros(size, dtype=np.float64)

    # point loads
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

    # distributed loads
    # - grouping distr_loads and distr_loads_inc
    distr_loads = []
    for load in shell.distr_loads:
        distr_loads.append(load + [1.])
    for load in shell.distr_loads_inc:
        distr_loads.append(load + [inc])

    for x, y, funcx, funcy, funcz, inc_i in distr_loads:
        if x is None and y is None:
            raise ValueError('x and y cannot be None when defining distributed loads')
        if x is not None and y is not None:
            raise ValueError('One of x or y must be None when defining distributed loads')
        funcx = funcx if funcx is not None else lambda x: 0
        funcy = funcy if funcy is not None else lambda x: 0
        funcz = funcz if funcz is not None else lambda x: 0
        if x is None:
            # getting integration points and weights
            points = np.zeros(shell.m, dtype=np.float64)
            weights = np.zeros(shell.m, dtype=np.float64)
            get_points_weights(shell.m, points, weights)
            # integrating g(x,ycte)*s(x,ycte)*dx = sum(weight_i * ( (a/2) * g(xcte, ycte) * s(xcte, ycte) ))
            for xi, weight in zip(points, weights):
                xcte = (xi + 1)*shell.a/2
                fpt = np.array([[funcx(xcte), funcy(xcte), funcz(xcte), 0, 0]]) * inc_i
                fg(g, xcte, y, shell)
                fext[col0:col1] += weight * (shell.a/2) * fpt.dot(g).ravel()
        else:
            # getting integration points and weights
            points = np.zeros(shell.n, dtype=np.float64)
            weights = np.zeros(shell.n, dtype=np.float64)
            get_points_weights(shell.n, points, weights)
            # integrating g(x,ycte)*s(x,ycte)*dx = sum(weight_i * ( (a/2) * g(xcte, ycte) * s(xcte, ycte) ))
            for eta, weight in zip(points, weights):
                ycte = (eta + 1)*shell.b/2
                fpt = np.array([[funcx(ycte), funcy(ycte), funcz(ycte), 0, 0]]) * inc_i
                fg(g, x, ycte, shell)
                fext[col0:col1] += weight * (shell.b/2) * fpt.dot(g).ravel()

    return fext
