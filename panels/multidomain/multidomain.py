r"""Assembly of several domains (:mod:`panels.multidomain.multidomain`)

The class :class:`.MultiDomain` assembles the matrices and vectors of
several :class:`.Shell` domains, imposes the connections between them and
the prescribed displacements, with the null-space or the penalty method,
computes the reactions, the cohesive-zone connections ``'SB_TSL'`` and the
fields of the results, and plots them. The function :func:`.load` reads an
assembly saved with :meth:`.MultiDomain.save`. Both are imported in
:mod:`panels.multidomain`.

"""
import platform

import numpy as np
from numpy import linspace, reshape
from scipy.sparse import csr_matrix, coo_matrix
from structsolve import lb, solve
from structsolve.sparseutils import finalize_symmetric_matrix
from matplotlib import pyplot as plt

from scipy.special import roots_legendre
from panels import json_io
from panels.logger import msg, warn
from panels.shell import DOUBLE, check_c, Shell
import panels.modelDB as modelDB
from panels.multidomain import connections
from panels.multidomain.connections import nullspace


def load(name, max_size=json_io.MAX_SIZE):
    r"""Load a :class:`.MultiDomain` saved by :meth:`.MultiDomain.save`

    The assembly is restored with its panels and connections, see
    :mod:`panels.json_io`. The matrices, not stored in the file, must be
    computed again.

    Parameters
    ----------
    name : str, path-like or file object
        Name of the file, with or without the extension
        ``'.multidomain.zip'``, or a binary file object opened for reading,
        e.g. :class:`io.BytesIO`.
    max_size : int, optional
        Maximum total uncompressed size of the members read, in bytes, 4 GiB
        by default, see :func:`panels.json_io.load`.

    Returns
    -------
    md : :class:`.MultiDomain`

    """
    return json_io._load_type(name, 'MultiDomain', ('.multidomain.zip', ),
                              max_size)


def _penalties(connecti, pA, pB, connection_type):
    r"""Penalty constants of one connection

    The defaults are given by :func:`.calc_kt_kr`, and the keys ``'kt'`` and
    ``'kr'`` of the connection dictionary override them.

    """
    kt, kr = connections.calc_kt_kr(pA, pB, connection_type)
    kt = connecti.get('kt', kt)
    kr = connecti.get('kr', kr)
    return kt, kr


def _sb_top_bottom(connecti):
    r"""Panels on the positive and negative sides of a skin-base connection

    The connections ``'SB'`` and ``'SB_TSL'`` connect the lower face of
    ``p1``, `z = d_1 - h_1/2`, to the upper face of ``p2``, `z = d_2 +
    h_2/2`, each in the coordinate `z` of its panel, with the offsets `d_1`,
    `d_2` and the thicknesses `h_1`, `h_2`, the `z` axis being normal to the
    reference surfaces of both panels, therefore ``p1`` must be the panel on
    the positive side of the interface along `z`, and ``p2`` the one on the
    negative side, independently of their order in the assembly. The two
    panels must have the same axes `x, y, z`. Both must cover the same area,
    with the same dimensions ``a`` and ``b``, otherwise a ``ValueError`` is
    raised.

    Notes
    -----
    The reference surfaces of ``p1`` and ``p2`` are `h_1/2 - d_1 + h_2/2 +
    d_2` apart, see :func:`.kCsdt._sb_distances`: `h_1/2 + h_2/2` without
    offsets, and zero when both panels share the reference surface, e.g. a
    stiffener base with `d_1 = (h_1 + h_2)/2` on a skin with `d_2 = 0`.
    Both panels must cover the same area, since :func:`.skin_base.fkCSB`,
    the null-space constraints of :func:`.nullspace.constraint_matrix`, the
    kernels of ``kCSB_dmg.pyx`` and the operators of
    :meth:`.MultiDomain._tsl_operators` evaluate the functions of ``p2`` at
    the natural coordinates of ``p1``, which correspond to the same physical
    point only when the two domains coincide.

    """
    p_top = connecti['p1']
    p_bot = connecti['p2']
    if not (np.isclose(p_top.a, p_bot.a) and np.isclose(p_top.b, p_bot.b)):
        raise ValueError('The panels of a "{0}" connection must have the '
                         'same dimensions, got a={1}, b={2} and a={3}, b={4}'
                         .format(connecti['func'], p_top.a, p_top.b,
                                 p_bot.a, p_bot.b))
    return p_top, p_bot


def _bf_base_flange(connecti):
    r"""Base and flange of a base-flange connection

    The kernels of ``'BFycte'`` and ``'BFxcte'`` rotate the local axes of
    the flange by 90 degrees about the axis of the connection with respect to
    those of the base, see :mod:`panels.multidomain.connections.kCBFycte`,
    therefore ``p1`` must be the base, or skin, and ``p2`` the flange,
    independently of their order in the assembly. The coordinates of the
    connection, ``xcte1``, ``xcte2`` or ``ycte1``, ``ycte2``, refer to ``p1``
    and ``p2``, respectively.

    """
    return connecti['p1'], connecti['p2']


def _upper_block(k12, p1, p2):
    r"""Coupling block between ``p1`` and ``p2`` in the upper triangle

    ``k12`` couples the rows of ``p1`` to the columns of ``p2``. The
    assembled connection matrix is made symmetric from its upper triangle
    only, see :func:`.finalize_symmetric_matrix`, so when ``p1`` comes after
    ``p2`` in the assembly the block is transposed.

    """
    if p1.col_start > p2.col_start:
        return k12.T
    return k12


def _rinv_sanders(p):
    r"""`1/r` of a panel with the kinematics of Sanders-Koiter, zero otherwise

    The value is passed to the kernels of the rotation penalty of the
    connection ``'BFycte'``, see
    :mod:`panels.multidomain.connections.kCBFycte`. A panel without a model
    gives zero.

    Notes
    -----
    With the kinematics of Sanders-Koiter the rotation of the normal about
    the `x` axis includes the term `v/r`, `\phi_y = -w_{,y} + v/r`, which
    enters the rotation penalty of the connection ``'BFycte'``, see
    :mod:`panels.multidomain.connections.kCBFycte`. The kinematics of
    Donnell neglect this term, hence the zero.

    """
    if p.model is not None and 'sanders' in p.model:
        return 1./p.r
    return 0.


def _dofs(p):
    r"""Number of DOFs per term of the model of ``p``

    A panel without a model gets one of the default models, which are based
    on the classical laminated plate theory with 3 DOFs, see
    :meth:`.Shell._rebuild`.

    """
    if p.model is None:
        return 3
    return modelDB.db[p.model]['dofs']


def default_field(panel, gridx, gridy):
    r"""Grid of points over the domain of a panel

    The points are ``gridx`` equally spaced values of `x` from ``0`` to
    ``panel.a`` combined with ``gridy`` equally spaced values of `y` from
    ``0`` to ``panel.b``, as given by :func:`numpy.meshgrid`, and returned
    flattened.

    Parameters
    ----------
    panel : :class:`.Shell`
        The panel, whose attributes ``a`` and ``b`` are used.
    gridx : int
        Number of points along `x`.
    gridy : int
        Number of points along `y`.

    Returns
    -------
    xs, ys : np.ndarray
        Coordinates `x` and `y` of the points, length, 1-D arrays of size
        ``gridx*gridy``.
    shape : tuple
        The shape of the grid, ``(gridy, gridx)``, to reshape ``xs``,
        ``ys`` and the fields computed at them.

    """
    xs = linspace(0, panel.a, gridx)
    ys = linspace(0, panel.b, gridy)
    xs, ys = np.meshgrid(xs, ys, copy=False)
    # Now xs and ys have the x and y coord of all the grid points i.e. size = no_grid_x * no_grid_y
    # Scalar inputs are converted to 1D arrays, whilst higher-dimensional inputs are preserved
    xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
    ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
    shape = xs.shape
    xs = xs.ravel()
    ys = ys.ravel()
    return xs, ys, shape


class MultiDomain(object):
    r"""Class for multi-domain semi-analytical approach

    This class has some useful methods that will help plotting output for
    different panel groups within the multi-domain and so forth.

    The constructor places the panels one after the other in the vector of
    Ritz constants of the assembly, in the order given, and sets the
    attributes ``row_start``, ``col_start``, ``row_end`` and ``col_end`` of
    each panel accordingly, with 3 degrees of freedom per term for the
    models based on the classical laminated plate theory, and for a panel
    without ``model``, and 5 for the models based on shear deformation
    theories. Creating another assembly with the same panels overwrites
    these attributes. No matrix is computed.


    Parameters
    ----------

    panels : iterable
        A list, tuple etc of :class:`.Shell` objects.
    conn : list of dict or None, optional
        A connectivity dictionary, a list with one dictionary per
        connection, see :meth:`.get_kC_conn`. With ``None``, the default,
        the connections must be given to :meth:`.calc_kC` or
        :meth:`.get_kC_conn`, which keep the last ones used.

        Concerning the conn dictionary, the key 'func' stands for the connection function that
        builds the compatibility relations between the panels.

        The connections functions available are:
            - 'SSycte' : defines a skin-skin connection for const y and calls the following functions ``fkCSSycte11``, ``fkCSSycte12``, ``fkCSSycte22``
            - 'SSxcte' : defines a skin-skin connection for const x and calls the following functions ``fkCSSxcte11``, ``fkCSSxcte12``, ``fkCSSxcte22``
            - 'SB' : defines a skin-base connection over an area and calls the function :func:`.skin_base.fkCSB`, see :meth:`.get_kC_conn` for the roles of ``p1`` and ``p2``
            - 'BFycte': defines a base-flange connection and calls the following functions ``fkCBFycte11``, ``fkCBFycte12``, ``fkCBFycte22``

        The connections ``'BFxcte'`` and ``'SB_TSL'`` are also available,
        see :meth:`.get_kC_conn`.

        Explanations about the connetion functions are found in ``connections`` module.
    conn_method : str, optional
        How the connections are imposed:

        - ``'null-space'`` (default): exactly, eliminating the constrained
          Ritz constants with a basis of the null space of the constraints,
          see :meth:`.get_Z` and
          :mod:`panels.multidomain.connections.nullspace`. The matrices and
          vectors must then be reduced with :meth:`.reduce` before solving,
          and the solutions expanded with :meth:`.expand`. The penalty
          constants ``'kt'`` and ``'kr'`` of the connections are not used,
          apart from the key ``'kr'`` of ``'SB'`` for the shear deformation
          theories, which tells whether the rotations are connected. The
          damaged connection ``'SB_TSL'`` keeps its penalty stiffness
        - ``'penalty'``: with penalty stiffnesses added to the stiffness
          matrix, see :meth:`.get_kC_conn`, solving directly for all the
          Ritz constants
    pd_method : str, optional
        How the prescribed displacements of the panels are imposed, see
        :meth:`.Shell.add_point_pd`, :meth:`.Shell.add_distr_pd_fixed_x` and
        :meth:`.Shell.add_distr_pd_fixed_y`:

        - ``'penalty'`` (default): their penalty forces enter
          :meth:`.calc_fext`, and the penalty stiffnesses must be added to
          the stiffness matrix by the user, e.g. with :func:`.fkCpd`,
          :func:`.fkCld_xcte` or :func:`.fkCld_ycte`
        - ``'null-space'``: exactly, as linear constraints eliminated
          together with the connections, see
          :meth:`.get_cp`, :meth:`.expand`, :meth:`.get_reduced_functions`,
          :meth:`.linear_static`, :meth:`.linear_buckling`,
          :meth:`.reaction_pd` and :meth:`.reaction_resultant`. The penalty
          constants are not used, and :meth:`.calc_fext` does not contain
          the forces of the prescribed displacements. This works with any
          ``conn_method``, also for a single panel without connections,
          ``MultiDomain([shell], conn=[], pd_method='null-space')``
    name : str, optional
        Name of the assembly, used by :meth:`.save`. Default
        ``'multidomain'``.

    Raises
    ------
    ValueError
        When ``conn_method`` or ``pd_method`` is not ``'penalty'`` or
        ``'null-space'``.

    Notes
    -----
    For more details about the theory involved, see Castro and Donadon
    (2017) [1]_. The vector of Ritz constants of the assembly is the
    concatenation of those of the panels, and the total potential energy of
    the assembly is the sum of the energies of the panels, whose matrices
    form a block-diagonal matrix, see :meth:`.calc_kC`, plus the energy of
    the connections, which make the displacements of the domains continuous:

    - with ``conn_method='penalty'``, by penalty energies, see
      :meth:`.get_kC_conn`, such that the continuity is approximate, with an
      error that decreases with the inverse of the penalty constants, while
      the condition number of the stiffness matrix grows with them;
    - with ``conn_method='null-space'``, as the exact linear constraints
      `[B] \{c\} = \{0\}`, eliminated with `\{c\} = [Z] \{c_r\}`, see
      :meth:`.get_Z`, as with penalty constants going to infinity.

    The connection ``'SB_TSL'`` represents the physical stiffness of a
    cohesive interface, which degrades with the damage, see
    :meth:`.calc_k_dmg`, and not a continuity condition, therefore it keeps
    its penalty stiffness with both methods, see :meth:`._null_space`.

    With ``pd_method='null-space'`` the prescribed displacements are the
    inhomogeneous linear constraints `[B_d] \{c\} = \{d_0\} + \lambda
    \{d_1\}`, and every admissible configuration is `\{c\} = [Z] \{c_r\} +
    \{c_{p0}\} + \lambda \{c_{p1}\}`, see :meth:`.get_cp` and
    :ref:`null_space_pd`. With ``pd_method='penalty'`` a prescribed value
    `u_p` is imposed with the energy `\frac{1}{2} k (u - u_p)^2`, integrated
    along the line for the distributed prescriptions, i.e. a spring of
    stiffness `k` between the structure and the support, whose force `k
    u_p` enters :meth:`.calc_fext`; the prescribed value is then reached
    only approximately, with an error that decreases with `1/k`. See
    :ref:`null_space`.

    References
    ----------
    .. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
       models to Address Linear Buckling and Vibration of Stiffened
       Composite Panels with Debonding Defect, Composite Structures, 2017.
       https://doi.org/10.1016/j.compstruct.2016.10.026

    Examples
    --------

    >>> panel_1 = Shell(*args)
    >>> panel_2 = Shell(*args)
    >>> panel_3 = Shell(*args)
    >>> panels = [panel_1, panel_2]
    >>> conn = [
            dict(p1=panel_1, p2=panel_2, func='SSxcte', xcte1=0, xcte2=panel_2.a),
            dict(p1=panel_2, p2=panel_3, func='SSycte', ycte1=0, ycte2=panel_3.b)
            ] # A list of dictionary that indicates two connections: (panel_1-panel_2) and (panel_2-panel_3)

    xcte1 = location (x const) in panel 1 which connects to panel 2
    xcte2 = ........................... 2 ....................... 2

    >>> md = MultiDomain(panels, conn)

    The connections are imposed exactly with the default
    ``conn_method='null-space'``, see :meth:`.get_Z`.

    Point or distributed forces or displacements can be easily added by using
    one of the methods:

        - :meth:`.Shell.add_point_load`
        - :meth:`.Shell.add_distr_load_fixed_x`
        - :meth:`.Shell.add_distr_load_fixed_y`
        - :meth:`.Shell.add_point_pd`
        - :meth:`.Shell.add_distr_pd_fixed_x`
        - :meth:`.Shell.add_distr_pd_fixed_y`

    With this, calculating the stiffness matrix and the external force
    vector is straightforward:

    >>> kC = md.calc_kC()
    >>> fext = md.calc_fext()

    These have the size of all the Ritz constants. With the default
    ``conn_method='null-space'`` they are reduced to the independent Ritz
    constants before solving, here with the Python module ``structsolve``,
    and the solution is expanded back to all the Ritz constants:

    >>> from structsolve import static
    >>>
    >>> incs, cs = static(md.reduce(kC), md.reduce(fext), silent=True)
    >>> c = md.expand(cs[0])

    Solving ``kC`` and ``fext`` directly would leave the domains
    disconnected, since ``kC`` then contains no connection stiffness. With
    ``conn_method='penalty'`` the matrices contain the penalty stiffnesses of
    the connections and are solved directly, ``md.reduce()`` and
    ``md.expand()`` being then the identity:

    >>> md = MultiDomain(panels, conn, conn_method='penalty')
    >>> incs, cs = static(md.calc_kC(), md.calc_fext(), silent=True)
    >>> c = cs[0]

    And plotting the results for the group named ``'skin'``.

    >>> md.plot(c, 'skin', filename='tmp_cylinder_compression_lb_Nxx_cte.png')


    The group name attribute belongs to each domain, and is passed while
    instantiating the :class:``.Shell`` object. Furthermore, it is important to
    pass the correct relative position of each domain within the group for
    generating consistent plots. For examples on how to create complex
    multidomains, see :func:`.create_cylinder` and
    :func:`.create_cylinder_blade_stiffened`.

    """
    def __init__(self, panels, conn=None, conn_method='null-space',
                 name='multidomain', pd_method='penalty'):
        if conn_method not in ('penalty', 'null-space'):
            raise ValueError("conn_method must be 'penalty' or 'null-space', "
                             "got '{0}'".format(conn_method))
        if pd_method not in ('penalty', 'null-space'):
            raise ValueError("pd_method must be 'penalty' or 'null-space', "
                             "got '{0}'".format(pd_method))
        # Initialize the assmbly obj with these values
        self.name = name
        self.conn = conn
        self.conn_method = conn_method
        self.pd_method = pd_method
        self.kC_conn = None
        self.Z = None
        self._Z_key = None
        self._Z_tol = 1.e-9
        self._rhs_tol = 1.e-8
        self._elim = None
        self._pd_blocks = []
        self._pd_conn = []
        self._cp_cache = None
        self.c_p0 = None
        self.c_p1 = None
        self._conn_used = None
        self.panels = panels
        self.size = None
        self.out_num_cores = 4
        row0 = 0
        col0 = 0
        for p in panels:
            assert isinstance(p, Shell) # Check if its Shell obj
            # This actually modifies the shell obj (i.e. passed panels)
            p.row_start = row0 # 1st panel starts at 0,0. Rest start at end of last matrix  --- Assembly of Kp_i from the paper
            p.col_start = col0
            #NOTE 3 DOFs u, v, w for the classical laminated plate theory and
            #     5 DOFs u, v, w, phix, phiy for the shear deformation theories
            row0 += _dofs(p)*p.m*p.n
            col0 += _dofs(p)*p.m*p.n
            p.row_end = row0 # This is now the start for the next panel to be assembled
            p.col_end = col0


    def save(self, fname=None):
        r"""Save the :class:`.MultiDomain` object to a zip file

        The panels and the connections are saved with the assembly, see
        :mod:`panels.json_io`. The matrices, including the null-space basis
        ``Z``, are not stored. The object is not modified.

        Parameters
        ----------
        fname : str, path-like or file object, optional
            Name of the file, or a binary file object opened for writing,
            e.g. :class:`io.BytesIO`. By default the name stored in
            ``MultiDomain.name`` followed by the extension
            ``'.multidomain.zip'``.

        """
        if fname is None:
            fname = self.name + '.multidomain.zip'
        if not hasattr(fname, 'write'):
            msg('Saving MultiDomain to {0}'.format(fname))
        json_io.save(self, fname)


    def get_size(self):
        """
        Size of K of the assembly

        The number of Ritz constants of the whole assembly, the sum over the
        panels of the number of degrees of freedom per term times ``m*n``,
        which is also the size of the matrices of the assembly. It is stored
        in the attribute ``size``.

        Returns
        -------
        size : int
            Number of Ritz constants of the assembly.

        """
        self.size = sum([_dofs(p)*p.m*p.n for p in self.panels])
        return self.size


    def _resolve_conn(self, conn):
        r"""Connections given, or those of the assembly, or those used last
        by :meth:`.get_kC_conn`, e.g. through ``calc_kC(conn)``"""
        if conn is not None:
            return conn
        if self.conn is not None:
            return self.conn
        if self._conn_used is not None:
            return self._conn_used
        raise RuntimeError('No connectivity dictionary defined!')


    def _null_space(self, connecti):
        r"""Whether the connection is imposed with the null-space method

        ``True`` when ``conn_method='null-space'`` and the function of the
        connection is in :data:`.nullspace.NULL_SPACE_FUNCS`.

        Notes
        -----
        The connections of :data:`.nullspace.NULL_SPACE_FUNCS` are continuity
        conditions, homogeneous linear constraints on the Ritz constants,
        `[B] \{c\} = \{0\}`, see :func:`.nullspace.constraint_matrix`,
        eliminated exactly by :meth:`.get_Z`. The other connection,
        ``'SB_TSL'``, represents the stiffness of a cohesive interface that
        degrades with the damage, and it always enters :meth:`.get_kC_conn`
        as a penalty stiffness.

        """
        return (self.conn_method == 'null-space'
                and connecti['func'] in nullspace.NULL_SPACE_FUNCS)


    def _Z_signature(self, conn):
        r"""Data that define `[Z]`, used to reuse it while unchanged

        A tuple with the data of the connections imposed exactly, of the
        panels and, with ``pd_method='null-space'``, of the positions and
        the components of the prescribed displacements, compared by
        :meth:`.get_Z`.

        Notes
        -----
        `[Z]` spans the null space of the constraints of :meth:`.get_Z`,
        whose rows depend on the function, the position and the panels of
        each connection imposed exactly, on whether the rotations of
        ``'SB'`` are connected (key ``'kr'``, used only by the FSDT and
        TSDT), on the number of terms, the
        dimensions, the radius, the model and the offset of each panel, on
        its ply thicknesses, through the faces connected by ``'SB'``, and on
        its boundary condition flags, which remove terms
        of the approximation functions. With ``pd_method='null-space'`` the
        positions and the prescribed components of the prescribed
        displacements add rows to the constraints, whereas the prescribed
        values change only the right-hand sides, i.e. the particular
        solutions of :meth:`.get_cp`, and are not part of the signature.

        """
        keys = ('func', 'xcte1', 'xcte2', 'ycte1', 'ycte2')
        sig_conn = tuple((tuple(ci.get(k) for k in keys), id(ci['p1']),
                          id(ci['p2']), bool(ci.get('kr', 0.)))
                         for ci in conn if self._null_space(ci))
        sig_panels = []
        for p in self.panels:
            bcs = tuple(float(getattr(p, d + e + f + r))
                        for d in 'xy' for e in '12'
                        for f in nullspace.FIELDS[:_dofs(p)] for r in ('', 'r'))
            plyts = () if p.plyts is None else tuple(p.plyts)
            sig_panels.append((id(p), p.m, p.n, p.a, p.b, p.r, p.model,
                               p.col_start, plyts, bcs))
        if self.pd_method != 'null-space':
            return sig_conn, tuple(sig_panels)
        #NOTE the positions and the prescribed components define [Z], the
        #     prescribed values only the particular solutions, see get_cp()
        sig_pds = tuple((id(p), cte, tuple(values[:2]), values[7],
                         tuple(v is not None for v in values[2:7]))
                        for p in self.panels
                        for cte, values_list in (
                            (True, p.point_pds_values + p.distr_pds_values),
                            (False, p.point_pds_values_inc
                                    + p.distr_pds_values_inc))
                        for values in values_list)
        return sig_conn, tuple(sig_panels), sig_pds


    def get_Z(self, conn=None, tol=None, rhs_tol=None):
        r"""Basis of the null space of the constraints imposed exactly

        With ``conn_method='null-space'``, the default, returns the matrix
        ``Z`` such that the Ritz constants of the assembly ``c = Z @ c_r``
        satisfy all the connections of :data:`.nullspace.NULL_SPACE_FUNCS`
        for any vector ``c_r`` of independent Ritz constants, see
        :mod:`panels.multidomain.connections.nullspace`. The matrices and
        vectors computed by this class, which do not contain these
        connections, are reduced with :meth:`.reduce` before solving, and
        the solution is expanded with :meth:`.expand`::

            kC = md.calc_kC()
            kG = md.calc_kG(c=c)
            eigvals, eigvecs_r = lb(md.reduce(kC), md.reduce(kG))
            eigvecs = md.expand(eigvecs_r)

        With ``conn_method='penalty'``, and without prescribed displacements
        imposed exactly, `[Z]` is the identity matrix and the same code
        solves the problem with the penalty method.

        With ``pd_method='null-space'``, the prescribed displacements of the
        panels are also constraints, eliminated after the connections, and
        `[Z]` is the basis of the null space of all the constraints; the
        particular solutions are given by :meth:`.get_cp`.

        `[Z]` is stored in the attribute ``Z`` and reused while the
        connections, the number of terms and the boundary conditions of the
        panels, and the positions and components of the prescribed
        displacements with ``pd_method='null-space'``, do not change, also
        by :meth:`.reduce` and :meth:`.expand`.

        Parameters
        ----------
        conn : list of dict, optional
            The connections, see :meth:`.get_kC_conn`. If ``None``, those of
            the assembly are used, or those used last by
            :meth:`.get_kC_conn`, e.g. through ``calc_kC(conn)``.
        tol : float, optional
            Tolerance to find the rank of the constraints, see
            :func:`.nullspace.null_space_basis`. If ``None``, the last one
            given is used, initially ``1e-9``.
        rhs_tol : float, optional
            Relative tolerance of the consistency of the prescribed
            displacements, see :meth:`.nullspace.Elimination.particular`.
            If ``None``, the last one given is used, initially ``1e-8``.

        Returns
        -------
        Z : scipy.sparse.csr_matrix
            Matrix with shape ``(size, size_r)``, with ``size_r`` the number
            of independent Ritz constants.

        Notes
        -----
        The connections imposed exactly are the homogeneous linear
        constraints `[B] \{c\} = \{0\}`, one block of rows per connection,
        see :func:`.nullspace.constraint_matrix`, and all the vectors that
        satisfy them are `\{c\} = [Z] \{c_r\}`, with `[B] [Z] = [0]`, where
        `\{c_r\}` are the independent Ritz constants. Substituting them into
        the total potential energy gives the reduced matrices and vectors of
        :meth:`.reduce`. With ``pd_method='null-space'``, the prescribed
        displacements add the inhomogeneous constraints `[B_d] \{c\} =
        \{d_0\} + \lambda \{d_1\}`, and `[Z]` spans the null space of all
        the constraints, see :meth:`.get_cp` and :ref:`null_space_pd`.

        The blocks of constraints are eliminated one at a time, the
        connections first, by :func:`.nullspace.null_space_basis` or, with
        prescribed displacements, by :class:`.nullspace.Elimination`, which
        keeps the data of the decompositions for the particular solutions:
        the rows of each block are normalized to a unit norm and written in
        the independent Ritz constants left by the previous blocks, and a QR
        decomposition with column pivoting reveals the rank of the block,
        each pivot larger than ``tol`` giving one independent constraint,
        whose Ritz constant is eliminated. The redundant constraints are
        dropped, e.g. at a corner shared by four domains, around a closed
        cylinder, or on a term already removed by a boundary condition flag.
        ``tol`` applies to the normalized constraints: the independent
        constraints have pivots many orders of magnitude above the round-off
        errors of the redundant ones, and a warning is issued when this gap
        is smaller than `10^4`. ``rhs_tol`` is the tolerance of the
        consistency of the redundant prescribed displacements, see
        :meth:`.get_cp`.

        The constraints could be imposed with Lagrange multipliers, which add
        one unknown per constraint and give an indefinite system; for the
        linear buckling and frequency analyses the geometric stiffness matrix
        and the mass matrix would then have zero rows and columns for the
        multipliers, giving infinite eigenvalues. The null-space method
        removes unknowns instead of adding them, and gives the same solution.
        Since the constraints are linear in `\{c\}`, `[Z]` does not change
        during a non-linear analysis. See :ref:`null_space`.

        """
        size = self.get_size()
        conn = self._resolve_conn(conn)
        if tol is not None:
            self._Z_tol = tol
        tol = self._Z_tol
        if rhs_tol is not None and rhs_tol != self._rhs_tol:
            self._rhs_tol = rhs_tol
            self._cp_cache = None
        key = (self._Z_signature(conn), tol)
        if self.Z is not None and self._Z_key == key:
            return self.Z
        Bs = [nullspace.constraint_matrix(ci, size)
              for ci in conn if self._null_space(ci)]
        blocks = []
        if self.pd_method == 'null-space':
            for p in self.panels:
                blocks += nullspace.pd_blocks(p, size)
        if blocks:
            elim = nullspace.Elimination(size, tol)
            for B in Bs:
                elim.add(B)
            for i, block in enumerate(blocks):
                elim.add(block['B'], rhs=True, info=block)
            self.Z = elim.finish()
            self._elim = elim
        else:
            self.Z = nullspace.null_space_basis(Bs, size, tol=tol)
            self._elim = None
        self._pd_blocks = blocks
        self._pd_conn = [ci for ci in conn if self._null_space(ci)]
        self._cp_cache = None
        self._Z_key = key
        return self.Z


    def _panel_label(self, p):
        i = [j for j, pj in enumerate(self.panels) if pj is p][0]
        return "panel {0} (name '{1}', group '{2}')".format(i, p.name, p.group)


    def _pd_label(self, block):
        comp = nullspace.PD_COMPONENTS[block['comp']]
        if block['z'] != 0 and block['comp'] in (nullspace.U, nullspace.V):
            comp = '{0} at z = {1:g}'.format(comp, block['z'])
        x, y = block['x'], block['y']
        if x is not None and y is not None:
            where = 'at (x, y) = ({0:g}, {1:g})'.format(x, y)
        elif y is None:
            where = 'along x = {0:g}'.format(x)
        else:
            where = 'along y = {0:g}'.format(y)
        return '{0} {1} of {2} (cte={3})'.format(comp, where,
               self._panel_label(block['panel']), block['cte'])


    def _pd_zero_flags(self, block, ends=True):
        r"""Boundary condition flags equal to zero that restrain the
        prescribed component at its points, with ``ends=False`` only those
        that restrain it along the whole line"""
        p = block['panel']
        flags = []
        for field, der_x, der_y, _ in nullspace._pd_terms(p, block['comp'],
                                                          block['z']):
            for direction, cte, length, der in (('x', block['x'], p.a, der_x),
                                                ('y', block['y'], p.b, der_y)):
                edges = []
                if cte is None and not ends:
                    continue
                if cte is None or np.isclose(cte, 0.):
                    edges.append('1')
                if cte is None or np.isclose(cte, length):
                    edges.append('2')
                for edge in edges:
                    name = (direction + edge + nullspace.FIELDS[field]
                            + ('r' if der else ''))
                    if float(getattr(p, name)) == 0. and name not in flags:
                        flags.append(name)
        return flags


    def _pd_describe(self, resultant=False):
        r"""Message of the inconsistent prescribed displacements, see
        :meth:`.nullspace.Elimination.particular`"""
        def describe(i, block, res, atol):
            p = block['panel']
            k = int(np.argmax(np.abs(res)))
            label = self._pd_label(block)
            flags = self._pd_zero_flags(block)
            if resultant:
                return ('The reaction resultant of the selected lines is not '
                        'defined: a unit translation of the selected lines '
                        'is not compatible with the other constraints at {0}, '
                        'e.g. a corner shared with a prescribed line not '
                        'selected, or connected to it; select all the lines '
                        'of the edge'.format(label))
            if block['B'][k].nnz == 0:
                flags = self._pd_zero_flags(block, ends=False) or flags
                return ('The prescribed {0} is not possible: this component '
                        'is removed by the boundary condition flags {1} = 0 '
                        'of the panel; set them to 1 to prescribe a non-zero '
                        'value, or remove the prescription'.format(label,
                        ', '.join(flags) or '(none found)'))
            panels = {id(p)}
            for ci in self._pd_conn:
                if ci['p1'] is p or ci['p2'] is p:
                    panels.update((id(ci['p1']), id(ci['p2'])))
            before = ['connection {0!r} between {1} and {2}'.format(
                      ci['func'], self._panel_label(ci['p1']),
                      self._panel_label(ci['p2'])) for ci in self._pd_conn
                      if ci['p1'] is p or ci['p2'] is p]
            before += ['prescribed ' + self._pd_label(b)
                       for b in self._pd_blocks[:i]
                       if id(b['panel']) in panels]
            msg_flags = ''
            if flags:
                msg_flags = ('; the boundary condition flags {0} = 0 of the '
                             'panel also restrain it'.format(', '.join(flags)))
            return ('The prescribed {0} is not consistent with the '
                    'constraints imposed before it, the redundant constraint '
                    'at its point {1} has the residual {2:.2e} > {3:.2e}{4}. '
                    'Constraints imposed before it on the same or connected '
                    'panels: {5}. With constant (cte=True) and incremented '
                    '(cte=False) prescriptions, each set must be consistent '
                    'by itself; with functions that are not polynomials, '
                    'see the parameter "rhs_tol" of MultiDomain.get_Z'
                    .format(label, k, abs(res[k]), atol, msg_flags,
                            '; '.join(before) or 'none'))
        return describe


    def get_cp(self):
        r"""Particular solutions of the prescribed displacements

        With ``pd_method='null-space'``, returns the vectors ``c_p0`` and
        ``c_p1`` that satisfy the constraints of the connections imposed
        exactly and of the prescribed displacements, ``c_p0`` with the
        values of the constant prescriptions (``cte=True``) and ``c_p1``
        with those of the prescriptions incremented with the load factor
        (``cte=False``), the other prescribed values being zero, such that
        the configuration at the load factor ``lbd`` is ``Z @ c_r + c_p0 +
        lbd*c_p1``, see :meth:`.expand`. They are computed in the
        elimination of :meth:`.get_Z`, which is reused while only the
        prescribed values change, the values being read again from the
        panels at each call, and they are stored in the attributes ``c_p0``
        and ``c_p1``.

        Returns
        -------
        c_p0, c_p1 : np.ndarray
            Vectors with the size of the assembly, null without prescribed
            displacements or with ``pd_method='penalty'``.

        Raises
        ------
        ValueError
            When the prescribed displacements are not consistent, with the
            panels, the prescriptions and the constraints involved.

        Notes
        -----
        With ``pd_method='null-space'`` the prescribed displacements are the
        constraints `[B_d] \{c\} = \{d_0\} + \lambda \{d_1\}`, `\{d_0\}`
        from the constant prescriptions (``cte=True``) and `\{d_1\}` from
        those incremented with the load factor (``cte=False``), and every
        admissible configuration is

        .. math::

            \{c\} = [Z] \{c_r\} + \{c_{p0}\} + \lambda \{c_{p1}\}

        with `[B] \{c_{p0}\} = \{0; d_0\}` and `[B] \{c_{p1}\} = \{0;
        d_1\}` for `[B]` the constraints of the connections and of the
        prescribed displacements, see :ref:`null_space_pd`. Each row of
        `[B_d]` is a prescribed component at a point, or at a Gauss-Legendre
        point of a prescribed line, see
        :func:`.nullspace.pd_constraint_matrix`, such that the constraint
        holds along the whole line when the prescribed function is a
        polynomial that the approximation functions represent, and only at
        these points otherwise.

        The right-hand sides are solved with the QR decompositions stored by
        :meth:`.get_Z`, see :meth:`.nullspace.Elimination.particular`,
        without any new decomposition. Since the constraints hold for every
        `\lambda`, the constant and the incremented sets must each be
        consistent by themselves. A redundant constraint is dropped only when
        the particular solution satisfies it within ``rhs_tol`` of
        :meth:`.get_Z`, relative to the largest prescribed value; otherwise
        the prescriptions are inconsistent, e.g. two prescribed lines with
        different values at a common corner, or a non-zero value on a term
        removed by a boundary condition flag.

        """
        self.get_Z()
        size = self.get_size()
        blocks = self._pd_blocks
        if not blocks:
            self.c_p0 = np.zeros(size, dtype=DOUBLE)
            self.c_p1 = np.zeros(size, dtype=DOUBLE)
            return self.c_p0, self.c_p1
        #NOTE [Z] is reused while the positions and the components do not
        #     change, the values are read again from the panels
        entries = [e for p in self.panels for e in nullspace.pd_entries(p)]
        assert len(entries) == len(blocks)
        for block, entry in zip(blocks, entries):
            block['value'] = entry[4]
        ds = [nullspace.pd_rhs(b) for b in blocks]
        key = np.concatenate(ds)
        if (self._cp_cache is not None
                and np.array_equal(self._cp_cache[0], key)):
            return self.c_p0, self.c_p1
        d0 = [d if b['cte'] else 0*d for d, b in zip(ds, blocks)]
        d1 = [0*d if b['cte'] else d for d, b in zip(ds, blocks)]
        describe = self._pd_describe()
        self.c_p0 = self._elim.particular(d0, self._rhs_tol, describe)
        self.c_p1 = self._elim.particular(d1, self._rhs_tol, describe)
        self._cp_cache = (key, )
        return self.c_p0, self.c_p1


    def reduce(self, A):
        r"""Reduce a matrix or a vector to the independent Ritz constants

        Returns ``Z.T @ A @ Z`` for a matrix and ``Z.T @ A`` for a vector,
        with ``Z`` from :meth:`.get_Z`.

        Parameters
        ----------
        A : np.ndarray or sparse matrix
            A vector, 1-D :class:`numpy.ndarray` of size :meth:`.get_size`,
            or a matrix with shape ``(size, size)``. Any other input,
            including a 2-D :class:`numpy.ndarray`, is treated as a matrix.

        Returns
        -------
        A_r : np.ndarray or scipy.sparse.csr_matrix
            1-D array of size ``size_r`` for a vector, or
            :class:`scipy.sparse.csr_matrix` with shape ``(size_r,
            size_r)`` for a matrix, with ``size_r`` the number of
            independent Ritz constants, the number of columns of `[Z]`.

        Notes
        -----
        Substituting `\{c\} = [Z] \{c_r\}` into the total potential energy,
        `\frac{1}{2} \{c\}^T [K] \{c\} - \{c\}^T \{f\}`, gives the reduced
        stiffness matrix `[K_r] = [Z]^T [K] [Z]` and the reduced force
        vector `\{f_r\} = [Z]^T \{f\}`, and likewise `[K_{G r}] = [Z]^T [K_G]
        [Z]` and `[M_r] = [Z]^T [M] [Z]` for the linear buckling and
        frequency analyses. `[K_r]` is symmetric and positive definite
        whenever the connected assembly has no rigid-body motion. With
        prescribed displacements imposed exactly, the reduced force vector
        also contains the forces of the particular solutions, see
        :meth:`.linear_static`. See :ref:`null_space`.

        """
        Z = self.get_Z()
        if isinstance(A, np.ndarray) and A.ndim == 1:
            return Z.T @ A
        return csr_matrix(Z.T @ A @ Z)


    def expand(self, c_r, lbd=None):
        r"""Ritz constants of the assembly from the independent ones

        Returns ``Z @ c_r``, with ``Z`` from :meth:`.get_Z`. ``c_r`` can
        also be a 2D array, e.g. with one eigenvector per column.

        With ``pd_method='null-space'`` and prescribed displacements, the
        configuration of the structure at the load factor ``lbd`` is
        returned when ``lbd`` is given, ``Z @ c_r + c_p0 + lbd*c_p1`` with
        ``c_p0`` and ``c_p1`` from :meth:`.get_cp`, added to each column of
        a 2D ``c_r``, whereas a mode, e.g. an eigenvector of a linear
        buckling or frequency analysis, or an increment of the
        configuration, is expanded without ``lbd``, ``Z @ c_r``::

            c = md.expand(c_r, lbd)        # configuration
            phi = md.expand(eigvecs_r)     # modes
            cs = [md.expand(c_r, lbd) for c_r, lbd in zip(an.cs, an.increments)]

        Parameters
        ----------
        c_r : array-like
            Independent Ritz constants, 1D, or 2D with one vector per
            column.
        lbd : float or None, optional
            Load factor of a configuration. With ``None``, the default,
            ``Z @ c_r`` is returned, unchanged with respect to the
            previous versions.

        Returns
        -------
        c : np.ndarray
            Ritz constants of the assembly, 1-D array of size
            :meth:`.get_size`, or 2-D with one vector per column when
            ``c_r`` is 2-D.

        Notes
        -----
        A configuration of the structure at the load factor `\lambda`
        satisfies the inhomogeneous constraints of the prescribed
        displacements, `\{c\} = [Z] \{c_r\} + \{c_{p0}\} + \lambda
        \{c_{p1}\}`, see :meth:`.get_cp`, whereas a mode, or an increment of
        the configuration at a constant load factor, satisfies the
        homogeneous constraints only, its prescribed components being zero,
        and is `[Z] \{c_r\}`.

        """
        Z = self.get_Z()
        if lbd is None:
            return Z @ c_r
        c_p0, c_p1 = self.get_cp()
        c_p = c_p0 + lbd*c_p1
        c = Z @ c_r
        if c.ndim == 2:
            return c + c_p[:, None]
        return c + c_p


    def get_reduced_functions(self):
        r"""Functions of the reduced problem for ``structsolve.Analysis``

        The functions take and return quantities reduced to the independent
        Ritz constants, see :meth:`.get_Z`, and can be used for the
        non-linear analyses::

            from structsolve import Analysis
            an = Analysis(*md.get_reduced_functions())
            an.static(NLgeom=True)
            cs = [md.expand(c_r, lbd) for c_r, lbd in zip(an.cs,
                                                          an.increments)]

        Without prescribed displacements imposed exactly, the functions
        evaluate :meth:`.calc_fext`, :meth:`.calc_fint`, :meth:`.calc_kC` and
        :meth:`.calc_kG` at the Ritz constants ``Z @ c`` and return ``Z.T @
        f`` for the vectors and ``Z.T @ K @ Z`` for the matrices.

        With ``pd_method='null-space'`` and prescribed displacements, they
        are evaluated at the configuration ``c_full = Z @ c + c_p0 +
        lbd*c_p1``, see :meth:`.get_cp`, with ``lbd`` equal to ``inc``,
        except in ``calc_fext``:

        - ``calc_fint(c, inc)``: the reduced :meth:`.calc_fint` at the load
          factor ``inc``;
        - ``calc_kC(c, NLgeom, inc)`` and ``calc_kG(c, NLgeom, inc)``: the
          reduced :meth:`.calc_kC` and :meth:`.calc_kG` at the configuration,
          whose sum is the reduced tangent stiffness matrix;
        - ``calc_fext(inc=1., c=None, lbd=None)``: without ``c``, the
          reduced reference vector ``Z.T @ calc_fext(inc)``, which the
          solvers scale by the load factor; with ``c`` and ``lbd``, ``Z.T @
          (calc_fext(inc, c=c_full) - inc*calc_kTv(c_full, c_p1, inc=lbd))``
          at the configuration of the load factor ``lbd``, the product being
          computed only when ``c_p1`` is not null, which ``structsolve``
          0.7.0 or newer passes in the non-linear analyses. The product of
          :meth:`.calc_kTv` avoids the assembly of the tangent stiffness
          matrix.

        The linear static analysis ``Analysis.static(NLgeom=False)`` cannot
        contain the constant prescribed displacements and is not available
        then: ``calc_fext`` raises an error when called with ``c`` and
        without ``lbd``; see :meth:`.linear_static` instead. The prescribed
        values used are those when this method is called.

        Returns
        -------
        calc_fext, calc_fint, calc_kC, calc_kG : tuple of functions

        Notes
        -----
        Without prescribed displacements imposed exactly, the reduced
        problem is `[Z]^T \{R([Z] \{c_r\}, \lambda)\} = \{0\}`, with the
        residual `\{R(c, \lambda)\} = \lambda \{F_{ext}\} - \{F_{int}(c,
        \lambda)\}` of the unconstrained system and the constant `[Z]`,
        since the constraints are linear.

        With ``pd_method='null-space'`` and prescribed displacements, the
        configuration is `\{c\} = [Z] \{c_r\} + \{c_{p0}\} + \lambda
        \{c_{p1}\}`, and substituting it into the residual and projecting it
        with `[Z]^T` gives the reduced problem solved by ``structsolve``:

        .. math::

            \{F_{int, r}(c_r, \lambda)\} &= [Z]^T \{F_{int}(c)\} \\
            [K_{T, r}] &= [Z]^T [K_T(c)] [Z] = -\frac{\partial R_r}{\partial
                c_r} \\
            \frac{\partial R_r}{\partial \lambda} &= [Z]^T \left(\{F_{ext}\}
                - [K_T(c)] \{c_{p1}\}\right)

        `[K_{T, r}]`, the sum of ``calc_kC`` and ``calc_kG``, is the exact
        Jacobian, and `\partial R_r/\partial \lambda`, returned by
        ``calc_fext`` with ``c`` and ``lbd``, depends on the configuration
        and on the load factor. ``structsolve`` uses it in the predictors
        and in the constraint of the arc-length methods, and `\lambda
        \partial R_r/\partial \lambda` as a reference quantity of the
        convergence criteria, Crisfield's scaled criterion, the default of
        ``structsolve`` 0.9.0, and the relative one, since under a prescribed
        displacement alone both `[Z]^T \{F_{ext}\}` and `[Z]^T \{F_{int}\}`
        vanish at equilibrium. See :ref:`null_space_pd`.

        """
        Z = self.get_Z()
        if not self._pd_blocks:
            def calc_fext(inc=1., silent=True, c=None):
                c = None if c is None else Z @ c
                return Z.T @ self.calc_fext(inc=inc, silent=silent, c=c)

            def calc_fint(c, inc=1., silent=True):
                return Z.T @ self.calc_fint(c=Z @ c, inc=inc, silent=silent)

            def calc_kC(c=None, NLgeom=False, inc=1., silent=True):
                c = None if c is None else Z @ c
                return csr_matrix(Z.T @ self.calc_kC(c=c, NLgeom=NLgeom, inc=inc,
                                  silent=silent) @ Z)

            def calc_kG(c=None, NLgeom=False, silent=True):
                c = None if c is None else Z @ c
                return csr_matrix(Z.T @ self.calc_kG(c=c, NLgeom=NLgeom,
                                  silent=silent) @ Z)

            return calc_fext, calc_fint, calc_kC, calc_kG

        c_p0, c_p1 = self.get_cp()
        has_p1 = bool(np.any(c_p1))

        def full(c_r, lbd):
            return Z @ c_r + c_p0 + lbd*c_p1

        def calc_fext(inc=1., silent=True, c=None, lbd=None):
            if c is None:
                return Z.T @ self.calc_fext(inc=inc, silent=silent)
            if lbd is None:
                raise ValueError('With the prescribed displacements of '
                    'pd_method="null-space", calc_fext(c=c) requires the '
                    'load factor "lbd", passed by structsolve>=0.7.0 in the '
                    'non-linear analyses; the linear static analysis '
                    'Analysis.static(NLgeom=False) is not available, use '
                    'MultiDomain.linear_static()')
            cf = full(c, lbd)
            fext = self.calc_fext(inc=inc, silent=silent, c=cf)
            if has_p1:
                #NOTE the product, without the assembly of [K_T]
                fext = fext - inc*self.calc_kTv(cf, c_p1, silent=silent,
                                                inc=lbd)
            return Z.T @ fext

        def calc_fint(c, inc=1., silent=True):
            return Z.T @ self.calc_fint(c=full(c, inc), inc=inc, silent=silent)

        def calc_kC(c=None, NLgeom=False, inc=1., silent=True):
            c = None if c is None else full(c, inc)
            kC = self.calc_kC(c=c, NLgeom=NLgeom, inc=inc, silent=silent)
            return csr_matrix(Z.T @ kC @ Z)

        def calc_kG(c=None, NLgeom=False, inc=1., silent=True):
            c = None if c is None else full(c, inc)
            kG = self.calc_kG(c=c, NLgeom=NLgeom, silent=silent)
            return csr_matrix(Z.T @ kG @ Z)

        return calc_fext, calc_fint, calc_kC, calc_kG


    def _linear_state(self, kC, fext, c_p, silent=True):
        r"""Linear static solution with a particular solution ``c_p``

        Solves the system of the reduced matrix ``Z.T @ kC @ Z`` with the
        right-hand side ``Z.T @ (fext - kC @ c_p)``, with
        ``structsolve.solve``, and returns ``Z @ c_r + c_p``.

        Notes
        -----
        Substituting `\{c\} = [Z] \{c_r\} + \{c_p\}` into `[K] \{c\} =
        \{F\}` and projecting it with `[Z]^T` gives `[K_r] \{c_r\} = [Z]^T
        (\{F\} - [K] \{c_p\})`, with `[K_r] = [Z]^T [K] [Z]`.

        """
        Z = self.get_Z()
        rhs = Z.T @ (fext - kC @ c_p)
        c_r = solve(csr_matrix(Z.T @ kC @ Z), rhs, silent=silent)
        return Z @ c_r + c_p


    def linear_static(self, lbd=1., silent=True):
        r"""Linear static analysis with exact prescribed displacements

        Solves the reduced linear system of the stiffness matrix of
        :meth:`.calc_kC`, for the external force vector ``calc_fext(inc=1.)``
        scaled by ``lbd``, as in the analyses of ``structsolve``, and for the
        prescribed displacements ``c_p0 + lbd*c_p1`` of :meth:`.get_cp`, see
        :meth:`._linear_state`, and returns the configuration. Without
        prescribed displacements imposed exactly it is the linear static
        analysis of the reduced problem, with ``pd_method='penalty'`` the
        penalty stiffnesses not being added.

        Parameters
        ----------
        lbd : float, optional
            Load factor.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.

        Returns
        -------
        c : np.ndarray
            Ritz constants of the assembly.

        Notes
        -----
        Solves

        .. math::

            [Z]^T [K] [Z] \{c_r\} = [Z]^T \left(\lambda \{F_{ext}\} - [K]
            (\{c_{p0}\} + \lambda \{c_{p1}\})\right)

        with `[K]` from :meth:`.calc_kC`, `\{F_{ext}\}` from
        ``calc_fext(inc=1.)``, scaled by `\lambda` as in the analyses of
        ``structsolve``, and `\{c_{p0}\}`, `\{c_{p1}\}` from
        :meth:`.get_cp`, and returns `\{c\} = [Z] \{c_r\} + \{c_{p0}\} +
        \lambda \{c_{p1}\}`. See :ref:`null_space_pd`.

        """
        c_p0, c_p1 = self.get_cp()
        kC = self.calc_kC(silent=silent)
        fext = lbd*self.calc_fext(inc=1., silent=silent)
        return self._linear_state(kC, fext, c_p0 + lbd*c_p1, silent=silent)


    def linear_buckling(self, num_eigvalues=25, c=None, silent=True,
                        **kwargs):
        r"""Linear buckling analysis with exact prescribed displacements

        The pre-buckling state is computed with linear static solutions, see
        :meth:`._linear_state`: ``c1`` for the external force vector
        ``calc_fext(inc=1.)`` and the incremented prescribed displacements
        ``c_p1``, and ``c0`` for the constant prescribed displacements
        ``c_p0`` alone. The reduced geometric stiffness matrix of ``c0``,
        which also contains the constant stress resultants ``Nxx``, ``Nyy``
        and ``Nxy`` of the panels, see :meth:`.Shell.calc_kG`, is added to
        the reduced constitutive stiffness matrix of :meth:`.calc_kC`, and
        the reduced geometric stiffness matrix of the stresses of ``c1``
        alone, computed with the constant stress resultants of the panels
        set to zero, is the second matrix of ``structsolve.lb``, whose
        eigenvalues are the critical load factors. Without constant
        prescribed displacements and constant stress resultants, it is the
        usual linear buckling analysis about the linear static solution for
        a load factor of 1. The matrices are reduced with :meth:`.reduce`.

        Parameters
        ----------
        num_eigvalues : int, optional
            Number of eigenvalues, passed to ``structsolve.lb``.
        c : array-like, optional
            A pre-buckling state of the assembly, ``c1``, instead of the
            linear static solution, with ``c0`` zero; the constant stress
            resultants of the panels are still a pre-stress.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        kwargs : dict, optional
            Other arguments of ``structsolve.lb``.

        Returns
        -------
        eigvals, eigvecs : tuple
            The critical load factors and the modes of the assembly, one
            per column, expanded with :meth:`.expand` without load factor.

        Raises
        ------
        ValueError
            When the incremented state ``c1`` has no geometric stiffness,
            e.g. when the only load is given by the constant stress
            resultants ``Nxx``, ``Nyy`` and ``Nxy`` of the panels, which are
            not multiplied by the load factor; apply such a load with
            :meth:`.Shell.add_distr_load_fixed_x` or
            :meth:`.Shell.add_distr_load_fixed_y` instead, or use
            :meth:`.calc_kG` and ``structsolve.lb`` directly.

        Notes
        -----
        The pre-buckling state is linear in the load factor, `\{c(\lambda)\}
        = \{c_0\} + \lambda \{c_1\}`, see :meth:`.linear_static`, with
        `\{c_0\}` due to the constant prescribed displacements alone and
        `\{c_1\}` due to the load and the incremented prescribed
        displacements. The geometric stiffness matrix of the linear buckling
        analysis is linear in the membrane stress resultants of the linear
        strains of the pre-buckling state, see :meth:`.Shell.calc_kG`, to
        which the constant stress resultants `\{N_0\}` of the attributes
        ``Nxx``, ``Nyy`` and ``Nxy`` of the panels are superposed. These
        belong to the constant part of the state, they are not multiplied
        by the load factor, so that `[K_G(N_0, c_0 + \lambda c_1)] =
        [K_G(N_0, c_0)] + \lambda [K_G(0, c_1)]`, and the critical load
        factors `\lambda` solve

        .. math::

            \left([K_r] + [K_{G,r}(N_0, c_0)] + \lambda [K_{G,r}(0,
            c_1)]\right) \{\phi_r\} = \{0\}

        with the matrices reduced with `[Z]`, see :meth:`.reduce`. Each
        constant contribution enters once: a constant prescribed rigid
        translation, which produces no stress, leaves the critical load
        factors unchanged. Without constant prescribed displacements and
        constant stress resultants it is the usual linear buckling analysis
        about the linear static solution for `\lambda = 1`. The modes
        satisfy the homogeneous constraints, `\{\phi\} = [Z] \{\phi_r\}`:
        under displacement control the prescribed components of the modes
        are zero, whereas under the equivalent load they are free. See
        :ref:`null_space_pd`.

        """
        Z = self.get_Z()
        kC = self.calc_kC(silent=silent)
        kC_r = csr_matrix(Z.T @ kC @ Z)
        c0 = None
        if c is None:
            c_p0, c_p1 = self.get_cp()
            fext = self.calc_fext(inc=1., silent=silent)
            c1 = self._linear_state(kC, fext, c_p1, silent=silent)
            if np.any(c_p0):
                c0 = self._linear_state(kC, 0*fext, c_p0, silent=silent)
        else:
            c1 = c
        has_N0 = any(any((p.Nxx, p.Nyy, p.Nxy)) for p in self.panels)
        if c0 is not None or has_N0:
            #NOTE the constant stress resultants of the panels are in the
            #     geometric stiffness matrix of c0, also when c0 is zero
            kC_r = kC_r + self.reduce(self.calc_kG(c=c0, silent=silent))
        N0 = [(p.Nxx, p.Nyy, p.Nxy) for p in self.panels]
        try:
            for p in self.panels:
                p.Nxx = p.Nyy = p.Nxy = 0.
            kG_r = self.reduce(self.calc_kG(c=c1, silent=silent))
        finally:
            for p, (Nxx, Nyy, Nxy) in zip(self.panels, N0):
                p.Nxx, p.Nyy, p.Nxy = Nxx, Nyy, Nxy
        if abs(kG_r).max() == 0:
            raise ValueError('The incremented pre-buckling state has no '
                'geometric stiffness: there is no load or incremented '
                'prescribed displacement. The constant stress resultants '
                'Nxx, Nyy and Nxy of the panels are a pre-stress, not '
                'multiplied by the load factor')
        eigvals, eigvecs = lb(kC_r, kG_r, num_eigvalues=num_eigvalues,
                              silent=silent, **kwargs)
        return eigvals, self.expand(eigvecs)


    def _reaction_residual(self, c, lbd, NLgeom):
        r"""Residual of the unconstrained system at ``c`` and ``lbd``

        ``calc_fint(c, inc=lbd) - lbd*calc_fext(inc=1.)`` with
        ``NLgeom=True``, otherwise ``calc_kC() @ c - lbd*calc_fext(inc=1.)``.

        Notes
        -----
        `\{r(c, \lambda)\} = \{F_{int}(c)\} - \lambda \{F_{ext}\}`. At an
        equilibrium configuration that satisfies the constraints imposed
        exactly, `[Z]^T \{r\} = \{0\}`, and `\{r\}` holds the generalized
        forces applied by the constraints on the structure, see
        :meth:`.reaction_pd`.

        """
        c = np.asarray(c, dtype=DOUBLE)
        fext = self.calc_fext(inc=1., silent=True)
        if NLgeom:
            return self.calc_fint(c=c, inc=lbd, silent=True) - lbd*fext
        return self.calc_kC(silent=True) @ c - lbd*fext


    def reaction_pd(self, c, lbd=1., NLgeom=True):
        r"""Generalized reactions of the prescribed displacements

        With ``pd_method='null-space'``, returns the reaction work-conjugate
        to each set of prescribed displacements, the constant ones
        (``cte=True``) and the incremented ones (``cte=False``): the dot
        products ``c_p0 @ r`` and ``c_p1 @ r`` of the particular solutions
        of :meth:`.get_cp` with the residual ``r`` of the unconstrained
        system at ``c`` and ``lbd``, see :meth:`._reaction_residual`.

        Parameters
        ----------
        c : array-like
            Ritz constants of the assembly at equilibrium, see
            :meth:`.expand`.
        lbd : float, optional
            Load factor of ``c``, the external force vector being
            ``calc_fext(inc=1.)`` scaled by ``lbd`` as in the analyses of
            ``structsolve``.
        NLgeom : bool, optional
            With ``True``, the internal force vector is :meth:`.calc_fint`,
            for the geometrically non-linear analyses; with ``False``, the
            product of the stiffness matrix of :meth:`.calc_kC` by ``c``,
            consistent with :meth:`.linear_static`.

        Returns
        -------
        P : dict
            ``{'cte': P_0, 'inc': P_1}``.

        Raises
        ------
        ValueError
            When ``pd_method`` is not ``'null-space'``.

        Notes
        -----
        The reaction work-conjugate to the set `j` of prescribed
        displacements is

        .. math::

            P_j = \{c_{pj}\}^T \{r(c, \lambda)\}, \qquad \{r(c, \lambda)\} =
            \{F_{int}(c)\} - \lambda \{F_{ext}\}

        with `\{c_{pj}\}` from :meth:`.get_cp`. It does not depend on the
        choice of the particular solution, because `[Z]^T \{r\} = 0` at
        equilibrium, and `P_j \delta` is the work of the reactions for the
        prescribed values multiplied by `\delta`. For a uniform
        end-shortening `\delta` of an edge, `P_j = N \delta` with `N` the
        total axial reaction, see :meth:`.reaction_resultant`, and `P_1 =
        \partial W/\partial \lambda` is the force conjugate to the load
        factor of a displacement-controlled analysis. See
        :ref:`null_space_pd`.

        """
        if self.pd_method != 'null-space':
            raise ValueError('reaction_pd() requires pd_method="null-space"')
        c_p0, c_p1 = self.get_cp()
        r = self._reaction_residual(c, lbd, NLgeom)
        return dict(cte=float(c_p0 @ r), inc=float(c_p1 @ r))


    def reaction_resultant(self, c, lines, lbd=1., NLgeom=True):
        r"""Resultant reaction forces along prescribed lines or points

        With ``pd_method='null-space'``, returns the resultants of the
        reactions of the selected prescriptions along the local directions
        `u`, `v` and `w` of the panels, the dot products ``c_u @ r`` of the
        residual ``r`` of the unconstrained system, as in
        :meth:`.reaction_pd`, with the particular solution ``c_u`` of the
        constraints whose prescribed value is 1 for the selected
        prescriptions of that component and 0 for all the other ones. For
        instance, the total axial force at the loaded end of a cylinder made
        of several panels is ``F[0]`` with all the lines of that end
        selected. A line whose ends are shared with a prescription not
        selected, directly or through a connection, has no resultant of its
        own, and an error is raised.

        Parameters
        ----------
        c : array-like
            Ritz constants of the assembly at equilibrium.
        lines : list of tuple
            The prescriptions selected, ``(panel, 'x', x)`` for the lines
            ``x = cte`` of :meth:`.Shell.add_distr_pd_fixed_x`, ``(panel,
            'y', y)`` for :meth:`.Shell.add_distr_pd_fixed_y` and ``(panel,
            'xy', (x, y))`` for the points of :meth:`.Shell.add_point_pd`.
        lbd : float, optional
            Load factor of ``c``, see :meth:`.reaction_pd`.
        NLgeom : bool, optional
            See :meth:`.reaction_pd`.

        Returns
        -------
        F : np.ndarray
            ``[F_u, F_v, F_w]``, ``nan`` for a direction not prescribed in
            the selected prescriptions.

        Raises
        ------
        ValueError
            When ``pd_method`` is not ``'null-space'``, when the kind of a
            selected prescription is not ``'x'``, ``'y'`` or ``'xy'``, or
            when no prescribed displacement of the panel matches a selected
            prescription.

        Notes
        -----
        The resultant along `u` is

        .. math::

            F_u = \{c_u\}^T \{r(c, \lambda)\}

        with `\{c_u\}` a solution of the constraints for a unit translation
        `u = 1` of the selected prescriptions that prescribe `u`, all the
        other prescribed values being zero, i.e. the virtual work of the
        reactions for this unit translation of the selected supports, see
        :meth:`.reaction_pd`. When a selected line shares an end with a
        prescription not selected, directly or through a connection, the
        unit translation is not compatible with the other constraints and
        the force at the shared corner is undetermined. The sign is that of
        the force applied by the support on the structure, `F_u = -N` at the
        shortened end, with `N` the total axial reaction of
        :meth:`.reaction_pd`. See :ref:`null_space_pd`.

        """
        if self.pd_method != 'null-space':
            raise ValueError('reaction_resultant() requires '
                             'pd_method="null-space"')
        self.get_Z()
        blocks = self._pd_blocks

        def matches(block, sel):
            p, kind, where = sel
            if block['panel'] is not p:
                return False
            x, y = block['x'], block['y']
            if kind == 'x':
                return y is None and np.isclose(x, where)
            if kind == 'y':
                return x is None and np.isclose(y, where)
            if kind == 'xy':
                return (x is not None and y is not None
                        and np.isclose(x, where[0]) and np.isclose(y, where[1]))
            raise ValueError("The kind of a selected prescription must be "
                             "'x', 'y' or 'xy', got {0!r}".format(kind))

        selected = [any(matches(b, sel) for sel in lines) for b in blocks]
        for sel in lines:
            if not any(matches(b, sel) for b in blocks):
                raise ValueError('No prescribed displacement {0!r} along {1} '
                                 '= {2} of {3}'.format(sel[1], sel[1], sel[2],
                                 self._panel_label(sel[0])))
        r = self._reaction_residual(c, lbd, NLgeom)
        F = np.full(3, np.nan)
        for comp in (nullspace.U, nullspace.V, nullspace.W):
            unit = [sel and b['comp'] == comp
                    for sel, b in zip(selected, blocks)]
            if not any(unit):
                continue
            ds = [np.full(b['B'].shape[0], 1. if u else 0.)
                  for u, b in zip(unit, blocks)]
            c_u = self._elim.particular(ds, self._rhs_tol,
                                        self._pd_describe(resultant=True))
            F[comp] = c_u @ r
        return F


    def plot(self, c=None, group=None, invert_y=False, vec='w', filename='', ax=None,
            figsize=(3.5, 2.), save=True, title='', identify=False,
            show_boundaries=False, boundary_line='--k', boundary_linewidth=1.,
            colorbar=False, cbar_nticks=10, cbar_format=None, cbar_title='',
            cbar_fontsize=7, colormap='jet', aspect='equal', clean=True,
            dpi=400, texts=[], xs=None, ys=None, gridx=50, gridy=50,
            num_levels=400, vecmin=None, vecmax=None,
            nr_x_gauss=None, nr_y_gauss=None,
            res=None, silent=True, display_zero=False, flip_plot=False,
            eval_panel=None):
        r"""Contour plot for a Ritz constants vector.

        The field ``vec`` of the panels of ``group`` is computed with
        :meth:`.calc_results`, unless ``res`` is given, and drawn with
        :meth:`matplotlib.axes.Axes.contourf`, one contour per panel, at
        the position ``(y0 + y, x0 + x)`` of the plot, i.e. with `y` along
        the horizontal axis and `x` along the vertical axis, swapped with
        ``flip_plot=True``. The horizontal axis is always inverted. The
        levels span ``vecmin`` to ``vecmax`` for all the panels.
        When ``ax`` is not given a new figure is created; the backend of
        matplotlib is set to ``'Agg'`` on Linux. With ``save=True`` and no
        ``ax``, the figure is saved to ``filename`` and closed.

        Parameters
        ----------
        c : np.ndarray or None
            The Ritz constants that will be used to compute the field contour.
            1-D array of size :meth:`.get_size`. Default ``None``, then
            ``res`` must be given.
        group : str or None
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together. Default ``None``.
        vec : str, optional
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``
            - Strain: ``'exx'``, ``'eyy'``, ``'gxy'``, ``'kxx'``, ``'kyy'``,
              ``'kxy'``, ``'gyz'``, ``'gxz'``
            - Stress: ``'Nxx'``, ``'Nyy'``, ``'Nxy'``, ``'Mxx'``, ``'Myy'``,
              ``'Mxy'``, ``'Qy'``, ``'Qx'``

            Default ``'w'``. The other names accepted by
            :meth:`.calc_results` can also be given.
        invert_y : bool, optional
            Inverts the `y` axis of the plot. Default ``False``.
        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
            Default ``True``, ignored when ``ax`` is given.
        dpi : int, optional
            Resolution of the saved file in dots per inch. Default ``400``.
        filename : str, optional
            The file name for the generated image file. If no value is given,
            ``group + '.png'`` is used.
        ax : AxesSubplot, optional
            When ``ax`` is given, the contour plot will be created inside it.
        figsize : tuple, optional
            The figure size given by ``(width, height)``. Default ``(3.5,
            2.)``.
        title : str, optional
            If any string is given a title is added to the contour plot.
        identify : bool, optional
            If domains should be identified. If yes, the text ``'P 01'``,
            ``'P 02'`` etc, with the position of the panel in the assembly,
            is written at the middle of each panel. Default ``False``.
        show_boundaries : bool, optional
            If boundaries between domains should be drawn. Default ``False``.
        boundary_line : str, optional
            Matplotlib string to define line type and color. Default
            ``'--k'``.
        boundary_linewidth : float, optional
            Matplotlib float to define line width. Default ``1.``.
        colorbar : bool, optional
            If a colorbar should be added to the contour plot. Default
            ``False``.
        cbar_nticks : int, optional
            Number of ticks added to the colorbar. Default ``10``.
        cbar_format : [ None | format string | Formatter object ], optional
            See the ``matplotlib.pyplot.colorbar`` documentation.
        cbar_title : str, optional
            Colorbar title. If ``cbar_title == ''`` no title is added.
        cbar_fontsize : int, optional
            Fontsize of the colorbar labels. Default ``7``.
        colormap : string, optional
            Name of a matplotlib available colormap. Default ``'jet'``, also
            used, with a warning, when the name is not found.
        aspect : str, optional
            String that will be passed to the ``AxesSubplot.set_aspect()``
            method. Default ``'equal'``.
        clean : bool, optional
            Clean axes ticks, grids, spines etc. Default ``True``.
        texts : list of dict, optional
            Each dictionary is passed as keyword arguments to
            ``ax.text(transform=ax.transAxes, **kwargs)``. Default ``[]``.
        xs : np.ndarray, optional
            Not used, the points come from ``gridx`` and ``gridy`` or from
            ``nr_x_gauss`` and ``nr_y_gauss``.
        ys : np.ndarray, optional
            Not used, see ``xs``.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field. Default ``50``.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field. Default ``50``.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).
            Default ``400``.
        vecmin : float, optional
            Minimum value for the contour scale (useful to compare with other
            results). If not specified it will be taken from the calculated
            field.
        vecmax : float, optional
            Maximum value for the contour scale.
        nr_x_gauss, nr_y_gauss : int or None, optional
            Number of Gauss-Legendre points along `x` and `y`, respectively,
            used instead of ``gridx`` or ``gridy`` when not ``None``, the
            default, see :meth:`.uvw`.
        res : dict, optional
            Contains the results of a previous run, as returned by
            :meth:`.calc_results`. If results already exist, then they're not
            computed again
        silent : bool, optional
            True = doesnt print any messages
        display_zero : bool, optional
            Decides whether 0 should be displayed as a tick in the colour bar or not
        flip_plot : bool, optional
            Flips plot so that x axis is now horizontal. Default ``False``;
            ``True`` prints a warning that it is still under development.
        eval_panel : :class:`.Shell` or None, optional
            When given, only this panel is drawn, instead of the panels of
            ``group``, with its field computed by :meth:`.calc_results` with
            the same ``eval_panel``. A ``res`` given with ``eval_panel``
            must contain the field of this panel only.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib object that can be used to modify the current plot
            if needed.
        data : dict
            Data calculated during the plotting procedure, with the keys
            ``'vecmin'`` and ``'vecmax'``.

        Raises
        ------
        ValueError
            When both ``c`` and ``res`` are ``None``, or when ``ax`` is not a
            :class:`matplotlib.axes.Axes`.

        """
        msg('Plotting contour.../../panels/multidomain/connections/penalties.py..', silent=True)

        import matplotlib.cm as cm
        import matplotlib
        if platform.system().lower() == 'linux':
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # If no results (res) is passed, then compute them
        if res is None:
            if c is None:
                raise ValueError('Either "c" or "res" must be given as input')
            res = self.calc_results(c, group, vec = vec, gridx = gridx, gridy = gridy,
                             nr_x_gauss = nr_x_gauss, nr_y_gauss = nr_y_gauss,
                             eval_panel=eval_panel)

        field = np.array(res[vec])
        msg('Finished!', level=1, silent=silent)

        if vecmin is None:
            vecmin = field.min()
        if vecmax is None:
            vecmax = field.max()

        data = dict(vecmin=vecmin, vecmax=vecmax)

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

        if invert_y == True:
            ax.invert_yaxis()
        ax.invert_xaxis()

        colormap_obj = getattr(cm, colormap, None)
        if colormap_obj is None:
            warn('Invalid colormap, using "jet"', level=1)
            colormap_obj = cm.jet

        # Used generally when the field for the entire group is needed
        if eval_panel is None:
            loop_panels = self.panels
        # Used when the field for just a single panel is needed
        if eval_panel is not None:
            loop_panels = [eval_panel]

        count = -1

        for i, panel in enumerate(loop_panels):
            if eval_panel is None:
                if panel.group != group:
                    continue

        # for i, panel in enumerate(self.panels):
        #     if panel.group != group:
        #         continue
            count += 1
            # x y flipped bec it needs to be plotted
            if flip_plot == False:
                xplot = res['y'][count] + panel.y0
                yplot = res['x'][count] + panel.x0
            if flip_plot == True:
                print('WARNING: flipping of plots still under development !!!!')
                xplot = res['x'][count] + panel.x0
                yplot = res['y'][count] + panel.y0
            field = res[vec][count]

            # print(np.shape(xplot), np.shape(yplot), np.shape(field))
            contour = ax.contourf(xplot, yplot, field, levels=levels,
                    cmap=colormap_obj)
            if identify:
                ax.text(xplot.mean(), yplot.mean(), 'P {0:02d}'.format(i+1),
                        transform=ax.transData, ha='center')
            if show_boundaries:
                # Also takes care of gauss point field
                if flip_plot == False:
                    x1_plot = panel.y0
                    x2_plot = panel.y0 + panel.b
                    y1_plot = panel.x0
                    y2_plot = panel.x0 + panel.a
                if flip_plot == True:
                    x1_plot = panel.x0
                    x2_plot = panel.x0 + panel.a
                    y1_plot = panel.y0
                    y2_plot = panel.y0 + panel.b
                # x1, x2 = xplot.min(), xplot.max()
                # y1, y2 = yplot.min(), yplot.max()
                # Plots both points as a line
                ax.plot((x1_plot, x2_plot), (y1_plot, y1_plot), boundary_line, lw=boundary_linewidth)
                ax.plot((x1_plot, x2_plot), (y2_plot, y2_plot), boundary_line, lw=boundary_linewidth)
                ax.plot((x1_plot, x1_plot), (y1_plot, y2_plot), boundary_line, lw=boundary_linewidth)
                ax.plot((x2_plot, x2_plot), (y1_plot, y2_plot), boundary_line, lw=boundary_linewidth)

        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fsize = cbar_fontsize
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='50%', pad=0.05)
                # Change colorbar size by changing size='')
            cbarticks = linspace(vecmin, vecmax, cbar_nticks)
            # Adding 0 to the ticks
            if display_zero :
                cbarticks = np.sort(np.append(cbarticks, np.array([0.])))
            cbar = plt.colorbar(contour, ticks=cbarticks, format=cbar_format,
                                cax=cax)
            if cbar_title:
                cax.text(0.5, 1.05, cbar_title, horizontalalignment='center',
                         verticalalignment='bottom', fontsize=fsize)
            # cbar.outline.remove()
            cbar.ax.tick_params(labelsize=fsize, pad=0.) #, tick2On=False) # Hides the tick marks

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
                filename = group + '.png'
            fig.savefig(filename, transparent=True,
                        bbox_inches='tight', pad_inches=0.05, dpi=dpi)
            plt.close()

        msg('finished!', silent=silent)

        return ax, data


    def plot3d(self, c, group=None, vec='w', vecs=None, displ='uvw',
               scale=None, gridx=30, gridy=30, colormap='jet', title='',
               filename='', show=False, show_undeformed=True,
               include_plotlyjs=True, silent=True):
        r"""Interactive 3D plot of the assembly with plotly

        Each domain is placed in the global coordinate system by its
        attributes ``x0, y0, z0``, the origin, ``point_x``, a point on its
        `x` axis, and ``point_xy``, a point on its `xy` plane, see
        :meth:`.Shell.global_frame`. The cylindrical shells are drawn curved,
        see :meth:`.Shell.global_coords`. The domains are drawn deformed by
        the displacements `u`, `v` and `w` times a scale factor, and colored
        by one of the outputs, which is selected in a drop-down menu.

        When the figure is shown in a web page, i.e. with ``filename`` or
        with ``show=True`` and a renderer of plotly based on HTML, such as
        ``'browser'`` or ``'notebook'``, controls above the figure toggle
        the displacements `u`, `v` and `w` and change the scale factor, and
        while the mouse pointer is over the figure the keys:

        - ``u``, ``v``, ``w``: toggle the displacement
        - ``+`` and ``-``: multiply and divide the scale factor by 1.5
        - ``0``: undeformed geometry, with a zero scale factor
        - ``r``: reset the displacements and the scale factor
        - ``n`` and ``p``: next and previous output
        - ``o``: toggle the orthographic and perspective projections

        The renderers without JavaScript, e.g. that of VS Code, keep the
        drop-down menu of the outputs and the initial deformed geometry.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants of the assembly, e.g. ``md.expand(c_r)``.
        group : str, list of str or None, optional
            The groups of the panels to plot, all the panels if ``None``.
        vec : str, optional
            The output initially shown, see :meth:`.plot`.
        vecs : list of str or None, optional
            The outputs that can be selected: the displacements ``'u'``,
            ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``, the strains and the
            stress resultants, see :meth:`.plot`. If ``None``, all those
            available for all the plotted panels.
        displ : str, optional
            The displacements initially applied to the geometry, any
            combination of ``'u'``, ``'v'`` and ``'w'``, or ``''`` for the
            undeformed geometry.
        scale : float or None, optional
            Scale factor of the displacements. If ``None``, the largest
            displacement is drawn with 10 % of the size of the assembly.
        gridx, gridy : int, optional
            Number of points along the `x` and `y` axes of each panel.
        colormap : str, optional
            A colorscale of plotly, e.g. ``'jet'`` or ``'viridis'``.
        title : str, optional
            Title of the figure.
        filename : str, optional
            If given, the figure is saved to this HTML file, with the
            controls of the displacements.
        show : bool, optional
            Show the figure with ``fig.show()``.
        show_undeformed : bool, optional
            If the edges of the undeformed panels are initially visible, they
            are toggled in the legend.
        include_plotlyjs : bool or str, optional
            See ``plotly.io.write_html``, ``True`` gives a file that works
            offline, ``'cdn'`` a much smaller one.
        silent : bool, optional
            Do not print messages.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            The figure.
        data : dict
            With the keys ``'vecs'``, the outputs, ``'vecmin'`` and
            ``'vecmax'``, dictionaries with the range of each output,
            ``'scale'``, the scale factor, and ``'post_script'``, the
            JavaScript of the controls, to show them with
            ``fig.show(post_script=data['post_script'])`` or
            ``fig.write_html(filename, post_script=data['post_script'])``.

        Examples
        --------

        A T-stiffened panel whose flange stands along the global `z` axis,
        above the line `y = y_s` of the skin::

            flange.x0, flange.y0, flange.z0 = 0., ys, 0.
            flange.point_x = (1., ys, 0.)
            flange.point_xy = (0., ys, 1.)
            fig, data = md.plot3d(c, filename='tstiff.html')

        """
        from panels.multidomain.plot3d import plot3d
        return plot3d(self, c, group=group, vec=vec, vecs=vecs, displ=displ,
                      scale=scale, gridx=gridx, gridy=gridy,
                      colormap=colormap, title=title, filename=filename,
                      show=show, show_undeformed=show_undeformed,
                      include_plotlyjs=include_plotlyjs, silent=silent)


    def calc_results(self, c, group=None, vec='w', gridx=50, gridy=50,
                     nr_x_gauss = None, nr_y_gauss = None,
                     eval_panel=None, x_cte_force=None, y_cte_force=None):

        r"""Results of a Ritz constants vector for a contour plot.

        Computes the results needed to plot the field ``vec``, without
        plotting: :meth:`.uvw` for a displacement, :meth:`.strain` for a
        strain, :meth:`.stress` for a stress resultant and :meth:`.force`
        for a force along a line, ``'Fxx'``, ``'Fyy'`` or ``'Fxy'``. The
        result of that method is returned with all its keys, not only
        ``vec``.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the field contour.
            1-D array of size :meth:`.get_size`.
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together. Default ``None``, then
            ``eval_panel`` must be given.
        vec : str, optional
            Variable that needs to be calculated
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``
            - Strain: ``'exx'``, ``'eyy'``, ``'gxy'``, ``'kxx'``, ``'kyy'``,
              ``'kxy'``, ``'gyz'``, ``'gxz'``, ``'kxx3'``, ``'kyy3'``,
              ``'kxy3'``, ``'gyz2'``, ``'gxz2'``
            - Stress: ``'Nxx'``, ``'Nyy'``, ``'Nxy'``, ``'Mxx'``, ``'Myy'``,
              ``'Mxy'``, ``'Qy'``, ``'Qx'``, ``'Pxx'``, ``'Pyy'``,
              ``'Pxy'``, ``'Ry'``, ``'Rx'``
            - Force: ``'Fxx'``, ``'Fyy'``, ``'Fxy'``

            Default ``'w'``.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field. Default ``50``.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field. Default ``50``.
        nr_x_gauss, nr_y_gauss : int or None, optional
            Number of gauss sampling points along x and y respectively where the displacement field is to be calculated,
            used instead of ``gridx`` or ``gridy`` when not ``None``, the
            default.
            The Gauss-Legendre points come from :func:`scipy.special.roots_legendre`,
            without an upper limit
        eval_panel : Shell object
            Used to evaluate results for a single panel (that is passed) instead of an entire group of panels
            Default ``None``.
        x_cte_force, y_cte_force : float or None, optional
            Constant `x` or `y` coordinate of a line at which the force is to be
            calculated. Coordinates are relative to specific panel, i.e. `0 \le
            x \le a`, `0 \le y \le b`. Used only for the forces, see
            :meth:`.force`. Default ``None``.

        Returns
        -------
        res : dict
            The output of :meth:`.uvw`, :meth:`.strain`, :meth:`.stress` or
            :meth:`.force`, see there.

        Raises
        ------
        ValueError
            When ``vec`` is not one of the names above.

        """

        displs = ['u', 'v', 'w', 'phix', 'phiy']
        strains = ['exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz',
                   'kxx3', 'kyy3', 'kxy3', 'gyz2', 'gxz2']
        stresses = ['Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qy', 'Qx',
                    'Pxx', 'Pyy', 'Pxy', 'Ry', 'Rx']
        forces = ['Fxx', 'Fyy', 'Fxy']

        msg('Computing field variables...', level=1, silent=True)

        # res = dict of all keywords in that specific input dict
        # size of each variable = no_panel_in_group x grid_pts_Y x grid_pts_X
        if vec in displs:
            res = self.uvw(c, group, gridx=gridx, gridy=gridy, nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss,
                           eval_panel=eval_panel)
        elif vec in strains:
            res = self.strain(c, group, gridx=gridx, gridy=gridy, nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss,
                              eval_panel=eval_panel)
        elif vec in stresses:
            res = self.stress(c, group, gridx=gridx, gridy=gridy, nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss,
                              eval_panel=eval_panel)
        elif vec in forces:
            res = self.force(c, group, eval_panel=eval_panel, x_cte_force=x_cte_force, y_cte_force=y_cte_force,
                             gridx=gridx, gridy=gridy, nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss)
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))

        return res


    def uvw(self, c, group, gridx=50, gridy=50, nr_x_gauss=None, nr_y_gauss=None,
            eval_panel=None):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated with the field functions of the model of each
        panel, at a grid of points of each panel of ``group``, or of
        ``eval_panel`` only. The returned values ``u``, ``v``, ``w``,
        ``phix``, ``phiy`` are not stored in the ``Shell`` objects.

        Along `x` the points are ``gridx`` equally spaced values from ``0``
        to ``a``, or, when ``nr_x_gauss`` is given, the ``nr_x_gauss``
        Gauss-Legendre points of :func:`scipy.special.roots_legendre` mapped
        to ``0 <= x <= a``; along `y` likewise with ``gridy``,
        ``nr_y_gauss`` and ``b``.

        Parameters
        ----------
        c : np.ndarray
            The full set of Ritz constants, 1-D array of size
            :meth:`.get_size`.
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together. Not used when ``eval_panel`` is
            given; ``group`` and ``eval_panel`` cannot both be ``None``.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field. Default ``50``.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field. Default ``50``.
        nr_x_gauss, nr_y_gauss : int or None, optional
            Number of gauss sampling points along x and y respectively where the displacement field is to be calculated
            Each one can be given or not, independently of the other.
            Both can be different. Default ``None``, the grid is used.
        eval_panel : Shell object
            Used to evaluate results for a single panel (that is passed) instead of an entire group of panels
            Default ``None``.

        Returns
        -------
        out : dict
            A dictionary with the keys ``'x'``, ``'y'``, ``'u'``, ``'v'``,
            ``'w'``, ``'phix'``, ``'phiy'``, each a list with one
            ``np.ndarray`` per evaluated panel, in the order of the
            assembly, with shape ``(ny, nx)``, ``nx`` and ``ny`` the numbers
            of points along `x` and `y`. ``'x'`` and ``'y'`` are the
            coordinates of the points in the panel, length, and the others
            the displacements, length, and the rotations, dimensionless.

        Notes
        -----
        The fields of each panel are linear combinations of the
        approximation functions of its model, with its Ritz constants as
        coefficients. ``phix`` and ``phiy`` are the rotations of the normal:
        `-w_{,x}` and `-w_{,y}`, plus `v/r` in ``phiy`` with the
        Sanders-Koiter kinematics, for the models based on the classical
        laminated plate theory, and the independent rotation fields of the
        models based on shear deformation theories. The Gauss-Legendre
        points `\xi_k \in (-1, 1)` are mapped to `x_k = a (\xi_k + 1)/2`, and
        likewise along `y`.

        """
        res = dict(x=[], y=[], u=[], v=[], w=[], phix=[], phiy=[])

        # Used generally when the field for the entire group is needed
        if group is not None:
            loop_panels = self.panels
        # Used when the field for just a single panel is needed
        if eval_panel is not None:
            loop_panels = [eval_panel]

        for panel in loop_panels:
            if eval_panel is None:
                if panel.group != group:
                    continue

            c_panel = c[panel.col_start: panel.col_end]
            c_panel = np.ascontiguousarray(c_panel, dtype=DOUBLE)
            model = panel.model
            fuvw = modelDB.db[model]['field'].fuvw

            if nr_x_gauss is not None:
                # Getting the gauss points and weights for x
                    # Gauss points are between 1 and -1
                x, x_weights = roots_legendre(nr_x_gauss)
                # Converting to physical coord as per panel dimensions
                # Done bec the clpt_field.pyx ftn converts it to natural coord
                x = (panel.a/2)*(x + 1)
            else:
                x = linspace(0, panel.a, gridx)
            if nr_y_gauss is not None:
                # Getting the gauss points and weights for y
                y, y_weights = roots_legendre(nr_y_gauss)
                # Converting to physical coord as per panel dimensions
                # Done bec the clpt_field.pyx ftn converts it to natural coord
                y = (panel.b/2)*(y + 1)
            else:
                y = linspace(0, panel.b, gridy)

            xs, ys = np.meshgrid(x, y, copy=False)
            # Size = size_y x size_x
            # Scalar inputs are converted to 1D arrays, whilst higher-dimensional inputs are preserved
            xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
            ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
            shape = xs.shape # gets the shape of xs which is of the grid
            x = xs.ravel()
            y = ys.ravel()

            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)

            u, v, w, phix, phiy = fuvw(c_panel, panel, x, y, self.out_num_cores)

            res['x'].append(reshape(x, shape))
            res['y'].append(reshape(y, shape))
            res['u'].append(reshape(u, shape))
            res['v'].append(reshape(v, shape))
            res['w'].append(reshape(w, shape))
            res['phix'].append(reshape(phix, shape))
            res['phiy'].append(reshape(phiy, shape))

        return res


    def strain(self, c, group, gridx=50, gridy=50, NLterms=True, nr_x_gauss=None, nr_y_gauss=None,
               eval_panel=None):
        r"""Calculate the strain field

        The strain field consists of in-plane strains (`\varepsilon_{xx}`,
        `\varepsilon_{yy}`, `\gamma_{xy}`) and curvatures (`\kappa_{xx}`,
        `\kappa_{yy}`, `\kappa_{xy}`) at each point over the grid of interest.
        The size of the grid is defined using ``gridx`` and ``gridy``. In the
        multidomain assembly, each domain will have a corresponding grid
        defined by this amount of points. The points are those of
        :meth:`.uvw`, with ``nr_x_gauss`` and ``nr_y_gauss`` replacing
        ``gridx`` and ``gridy`` when given.

        Parameters
        ----------
        c : np.ndarray
            The full set of Ritz constants, 1-D array of size
            :meth:`.get_size`.
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            that should be plotted together. Not used when ``eval_panel`` is
            given; ``group`` and ``eval_panel`` cannot both be ``None``.
        gridx : int, optional
            Number of points along the `x` axis, where the strain field is
            calculated. Default ``50``.
        gridy : int, optional
            Number of points along the `y` axis, where the strain field is
            calculated. Default ``50``.
        NLterms : bool, optional
            Flag to indicate whether non-linear strain components should be considered.
            Default ``True``.
        nr_x_gauss, nr_y_gauss : int or None, optional
            Number of gauss sampling points along x and y respectively where
            the field is to be calculated. Each one can be given or not,
            independently of the other. They can be different. Default
            ``None``, the grid is used.
        eval_panel : Shell object
            Used to evaluate results for a single panel (that is passed)
            instead of an entire group of panels. Default ``None``.

        Returns
        -------
        out : dict
            A dictionary of ``np.ndarrays`` with the keys:
            ``(x, y, exx, eyy, gxy, kxx, kyy, kxy)``.
            Each is a list with one array per evaluated panel, of shape
            ``(ny, nx)``, ``nx`` and ``ny`` the numbers of points along `x`
            and `y`. The strains are dimensionless and the curvatures have
            the dimension of 1/length.
            The panels of the models based on shear deformation theories
            also give the strains ``(gyz, gxz)`` and, for the third-order
            theory, ``(kxx3, kyy3, kxy3, gyz2, gxz2)``, see
            :meth:`.Shell.strain`.

        Notes
        -----
        The strains are those of the kinematics of the model of each panel,
        see :meth:`.Shell.strain`. With ``NLterms=True`` they include the
        non-linear terms of the strain-displacement relations, products of
        the rotations, e.g. `\frac{1}{2} \phi_y^2` in `\varepsilon_{yy}` for
        the models based on the classical laminated plate theory, and with
        ``NLterms=False`` only the linear terms.

        """
        res = dict(x=[], y=[], exx=[], eyy=[], gxy=[], kxx=[], kyy=[], kxy=[])

        if group is not None:
            loop_panels = self.panels

        # Used when the field for just a single panel is needed
        if eval_panel is not None:
            loop_panels = [eval_panel]

        for panel in loop_panels:
            if eval_panel is None:
                if panel.group != group:
                    continue

            c_panel = c[panel.col_start: panel.col_end]
            c_panel = np.ascontiguousarray(c_panel, dtype=DOUBLE)
            model = panel.model
            # In panels\panels\models - for plates, clpt_field
            fstrain = modelDB.db[model]['field'].fstrain

            # Here x and y are the complete unravelled grid in the x and y coord resp
            # So both have the size = gridx*gridy
            # x and y now have the x and y coord of all the grid points i.e. size = no_grid_x * no_grid_y
            if nr_x_gauss is not None:
                # Getting the gauss points and weights for x
                    # Gauss points are between 1 and -1
                x, x_weights = roots_legendre(nr_x_gauss)
                # Converting to physical coord as per panel dimensions
                # Done bec the clpt_field.pyx ftn converts it to natural coord
                x = (panel.a/2)*(x + 1)
            else:
                x = linspace(0, panel.a, gridx)
            if nr_y_gauss is not None:
                # Getting the gauss points and weights for y
                y, y_weights = roots_legendre(nr_y_gauss)
                # Converting to physical coord as per panel dimensions
                # Done bec the clpt_field.pyx ftn converts it to natural coord
                y = (panel.b/2)*(y + 1)
            else:
                y = linspace(0, panel.b, gridy)

            xs, ys = np.meshgrid(x, y, copy=False)
            # Scalar inputs are converted to 1D arrays, whilst higher-dimensional inputs are preserved
            xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
            ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
            shape = xs.shape # gets the shape of xs which is of the grid
            x = xs.ravel()
            y = ys.ravel()

            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)

            strains = fstrain(c_panel, panel, x, y, self.out_num_cores,
                              NLgeom=int(NLterms))
            res['x'].append(reshape(x, shape))
            res['y'].append(reshape(y, shape))
            #NOTE the models based on shear deformation theories have more
            #     strain components, see Shell.strain()
            for name, value in zip(panel._strain_names(), strains):
                res.setdefault(name, []).append(reshape(value, shape))

        return res


    def stress(self, c, group, gridx=50, gridy=50, NLterms=True, nr_x_gauss=None, nr_y_gauss=None,
               eval_panel=None, x_cte_force=None, y_cte_force=None):

        r"""Calculate the stress field

        The stress field consists of in-plane distributed forces (Nxx, Nyy,
        Nxy) and distributed moments (Mxx, Myy, Mxy) at each point over the
        grid of interest. The size of the grid is defined using ``gridx`` and
        ``gridy``. In the multidomain assembly, each domain will have a
        corresponding grid defined by this amount of points. The points are
        those of :meth:`.uvw`, with ``nr_x_gauss`` and ``nr_y_gauss``
        replacing ``gridx`` and ``gridy`` when given. The stress resultants
        are the products of the strains, computed as in :meth:`.strain`, by
        the constant constitutive matrix of each panel,
        ``Shell._get_constant_ABD()``.

        Parameters
        ----------
        c : np.ndarray
            The full set of Ritz constants, 1-D array of size
            :meth:`.get_size`.
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together. Not used when ``eval_panel`` is
            given; ``group`` and ``eval_panel`` cannot both be ``None``.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field. Default ``50``.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field. Default ``50``.
        NLterms : bool, optional
            Flag to indicate whether non-linear strain components should be considered.
            Default ``True``.
        nr_x_gauss, nr_y_gauss : int or None, optional
            Number of gauss sampling points along x and y respectively where the displacement field is to be calculated
            Each one can be given or not, independently of the other.
            Both can be different. Default ``None``, the grid is used.
        eval_panel : Shell object
            Used to evaluate results for a single panel (that is passed) instead of an entire group of panels
            Default ``None``.
        x_cte_force, y_cte_force : float or None, optional
            Not used. Default ``None``.

        Returns
        -------
        out : dict
            A dict containing many ``np.ndarrays``, with the keys:
            ``(x, y, Nxx, Nyy, Nxy, Mxx, Myy, Mxy)``, each a list with one
            array per evaluated panel, of shape ``(ny, nx)``, ``nx`` and
            ``ny`` the numbers of points along `x` and `y`. ``Nxx``, ``Nyy``
            and ``Nxy`` are forces per unit length, ``Mxx``, ``Myy`` and
            ``Mxy`` moments per unit length, i.e. forces. The panels of the
            models based on shear deformation theories also give the
            resultants ``(Qy, Qx)`` and, for the third-order theory,
            ``(Pxx, Pyy, Pxy, Ry, Rx)``, see :meth:`.Shell.stress`.

        Notes
        -----
        For the classical laminated plate theory the stress resultants are
        `\{N_{xx}, N_{yy}, N_{xy}, M_{xx}, M_{yy}, M_{xy}\} = [ABD]
        \{\varepsilon_{xx}, \varepsilon_{yy}, \gamma_{xy}, \kappa_{xx},
        \kappa_{yy}, \kappa_{xy}\}`, with `[ABD]` the constitutive matrix of
        the laminate, and the shear deformation theories add the transverse
        shear and higher-order terms with their larger constitutive
        matrices, see :meth:`.Shell.stress`. The strains are those of
        :meth:`.strain`, with the non-linear terms when ``NLterms=True``.
        The constitutive matrix must be constant over the panel, a stiffness
        per integration point, ``Shell.ABDnxny``, not being supported, since
        the stresses are evaluated at any point.

        """
        res = dict(x=[], y=[], Nxx=[], Nyy=[], Nxy=[], Mxx=[], Myy=[], Mxy=[])

        # Used generally when the stress field for the entire group is needed
        if group is not None:
            loop_panels = self.panels
        # Used when the stress field for just a single panel is needed
        if eval_panel is not None:
            loop_panels = [eval_panel]

        for panel in loop_panels:
            if eval_panel is None:
                if panel.group != group:
                    continue
            c_panel = c[panel.col_start: panel.col_end]
            c_panel = np.ascontiguousarray(c_panel, dtype=DOUBLE)
            model = panel.model
            fstrain = modelDB.db[model]['field'].fstrain

            if nr_x_gauss is not None:
                # Getting the gauss points and weights for x
                x, x_weights = roots_legendre(nr_x_gauss)
                # Converting to physical coord as per panel dimensions
                x = (panel.a/2)*(x + 1)
            else:
                x = linspace(0, panel.a, gridx)
            if nr_y_gauss is not None:
                # Getting the gauss points and weights for y
                y, y_weights = roots_legendre(nr_y_gauss)
                # Converting to physical coord as per panel dimensions
                y = (panel.b/2)*(y + 1)
            else:
                y = linspace(0, panel.b, gridy)

            xs, ys = np.meshgrid(x, y, copy=False)
            # Scalar inputs are converted to 1D arrays, whilst higher-dimensional inputs are preserved
            xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
            ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
            shape = xs.shape # gets the shape of xs which is of the grid
            x = xs.ravel()
            y = ys.ravel()

            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)

            strains = [reshape(e, shape) for e in fstrain(c_panel, panel, x,
                       y, self.out_num_cores, NLgeom=int(NLterms))]
            F = panel._get_constant_ABD()

            # For x: Each row goes from 0 to panel.a
            # For y: Each col goes from 0 to panel.b
            res['x'].append(reshape(x, shape))
            res['y'].append(reshape(y, shape))
            #NOTE the models based on shear deformation theories have more
            #     stress resultants, see Shell.stress()
            for i, name in enumerate(panel._stress_names()):
                res.setdefault(name, []).append(
                        sum(e*F[i, j] for j, e in enumerate(strains)))
        return res


    def force(self, c, group, eval_panel, x_cte_force=None, y_cte_force=None,
              gridx=50, gridy=50, NLterms=True, nr_x_gauss=None, nr_y_gauss=None):

        r"""Calculate the force along a line (xcte or ycte)

        The stress resultants ``Nxx``, ``Nyy`` and ``Nxy`` of one panel are
        computed with :meth:`.stress` and integrated with the Gauss-Legendre
        rule along the line `x` = ``x_cte_force``, over the ``nr_y_gauss``
        points along `y`, or along the line `y` = ``y_cte_force``, over the
        ``nr_x_gauss`` points along `x`. The line must pass through the
        points where the stress resultants are computed: ``x_cte_force``
        must be one of the points along `x`, the ``gridx`` equally spaced
        values from ``0`` to ``a`` or the ``nr_x_gauss`` Gauss-Legendre
        points, and likewise for ``y_cte_force``. The panel is
        ``eval_panel`` or, when it is ``None``, the first panel of ``group``.

        Parameters
        ----------
        c : np.ndarray
            The full set of Ritz constants, 1-D array of size
            :meth:`.get_size`.
        group : str
            A group of panels. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            belong together. Used only when ``eval_panel`` is ``None``, see
            above.
        eval_panel : Shell object
            Used to evaluate results for a single panel (that is passed) instead of an entire group of panels
        x_cte_force, y_cte_force : float or None, optional
            Constant `x` or `y` coordinate of a line at which the force is to be
            calculated. Coordinates are relative to specific panel, i.e. `0 \le
            x \le a`, `0 \le y \le b`. Exactly one of them must be given.
            Default ``None``.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field. Default ``50``.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field. Default ``50``.
        NLterms : bool, optional
            Flag to indicate whether non-linear strain components should be considered.
            Not passed to :meth:`.stress`, which uses its default ``True``.
        nr_x_gauss, nr_y_gauss : int or None, optional
            Number of gauss sampling points along `x` and `y`, respectively. The
            displacement field is to be calculated along either one of
            ``nr_x_gauss`` or ``nr_y_gauss``. ``nr_y_gauss`` is required with
            ``x_cte_force`` and ``nr_x_gauss`` with ``y_cte_force``. Default
            ``None``.

        Returns
        -------
        out : dict
            A dict containing many ``np.ndarrays``, with the keys:
            ``(Fxx, Fyy, Fxy)``. Each value is a scalar, the integral of
            ``Nxx``, ``Nyy`` or ``Nxy`` along the line, a force.

        Raises
        ------
        ValueError
            When both or none of ``x_cte_force`` and ``y_cte_force`` are
            given, when ``nr_y_gauss`` is missing with ``x_cte_force`` or
            ``nr_x_gauss`` with ``y_cte_force``, when the points found on the
            line are not in a single column or row of the grid, or when
            their number differs from ``nr_y_gauss`` or ``nr_x_gauss``.

        Notes
        -----
        The force is the integral of the stress resultant along the line,
        e.g. for ``x_cte_force``:

        .. math::

            F_{xx} = \int_0^b N_{xx}(x_c, y) \, dy \approx \frac{b}{2}
                     \sum_k w_k N_{xx}(x_c, y_k)

        with the ``nr_y_gauss`` Gauss-Legendre points `y_k` and weights
        `w_k`, exact when `N_{xx}` is a polynomial in `y` of degree up to
        ``2*nr_y_gauss - 1``, and likewise along `x` for ``y_cte_force``.

        """
        # Empty dict - keys added later on
        res = dict()

        if x_cte_force is None and y_cte_force is None:
            raise ValueError('x_cte or y_cte need to be specified')
        if x_cte_force is not None and y_cte_force is not None:
            raise ValueError('Integration can only be performed along a single line - specify either x_cte or y_cte')
        if x_cte_force is not None and nr_y_gauss is None:
            raise ValueError('Both x_cte_force and nr_y_gauss need to be specified')
        if y_cte_force is not None and nr_x_gauss is None:
            raise ValueError('Both y_cte_force and nr_x_gauss need to be specified')

        # Stress field for a single panel of interest
        res_stress = self.stress(c=c, group=group, eval_panel=eval_panel,
                                 x_cte_force=x_cte_force, y_cte_force=y_cte_force,
                                 gridx=gridx, gridy=gridy,
                                 nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss)
        # the panel whose results are in res_stress[...][0]
        if eval_panel is not None:
            panel = eval_panel
        else:
            panel = [p for p in self.panels if p.group == group][0]

        for vec in ['Nxx', 'Nyy', 'Nxy']:
            if x_cte_force is not None:
                [_, col] = np.where(np.isclose(res_stress['x'][0], x_cte_force))
                    # Find which col corresponds to values at x_cte_forces
                if col.max() != col.min():
                    # Check that you've picked the right col no - if its right, all x's should have the same col no
                    raise ValueError('Error - Check force values')
                # Extracting stress field for that particular column
                stress_field = res_stress[vec][0][:, col.min()] # So that only 1 col is taken
                if np.shape(stress_field)[0] != nr_y_gauss:
                    raise ValueError('Size mismatch')
                # Getting the gauss points and weights
                y_temp, weights = roots_legendre(nr_y_gauss)
                # Jacobian of the mapping from the natural coordinate
                jacobian = panel.b/2

            if y_cte_force is not None:
                [row, _] = np.where(np.isclose(res_stress['y'][0], y_cte_force))
                    # Find which row corresponds to values at y_cte_forces
                if row.max() != row.min():
                    # Check that you've picked the right row no
                    raise ValueError('Error - Check force values')
                # Extracting stress field for that particular column
                stress_field = res_stress[vec][0][row.min(), :]
                if np.shape(stress_field)[0] != nr_x_gauss:
                    raise ValueError('Size mismatch')
                # Getting the gauss points and weights
                x_temp, weights = roots_legendre(nr_x_gauss)
                jacobian = panel.a/2

            # Integration
            force_intgn = np.dot(weights, stress_field)*jacobian

            # Adding keys by modifying keys of res_stress (F added, N removed)
            res[f'F{vec[1:]}'] = force_intgn

        return res


    def force_out_plane(self, c, group, eval_panel, x_cte_force=None, y_cte_force=None,
              gridx=50, gridy=50, NLterms=True, nr_x_gauss=None, nr_y_gauss=None):
        r"""Out-of-plane force along a line of a panel from its moments

        The transverse shear force ``Qx`` is integrated along the line `x`
        = ``x_cte_force``, over ``nr_y_gauss`` Gauss-Legendre points along
        `y`, or ``Qy`` along the line `y` = ``y_cte_force``, over
        ``nr_x_gauss`` Gauss-Legendre points along `x`. The shear force is
        obtained from the derivatives of the moments ``Mxx``, ``Myy`` and
        ``Mxy`` of ``eval_panel``, computed as in :meth:`.stress` at the
        points of the line and at the points at a distance `a/(gridx - 1)`
        (or `b/(gridy - 1)`) on each side of it, inside the panel. When
        neither line is given, the line is `x = a (gridx - 2)/(gridx - 1)`,
        the second last of ``gridx`` equally spaced points from ``0`` to
        ``a``. The code keeps two other integrals, an area integral and one
        based on the derivatives of `w`, which are disabled.

        Parameters
        ----------
        c : np.ndarray
            The full set of Ritz constants, 1-D array of size
            :meth:`.get_size`.
        group : str or None
            Not used, since ``eval_panel`` is given.
        eval_panel : :class:`.Shell`
            The panel, required, with a model of 3 degrees of freedom per
            term.
        x_cte_force, y_cte_force : float or None, optional
            Constant `x` or `y` coordinate of the line, relative to the
            panel, `0 \le x \le a` or `0 \le y \le b`. At most one of them.
            Default ``None``, the line `x = a (gridx - 2)/(gridx - 1)`.
        gridx, gridy : int, optional
            The step of the finite differences across the line `x` = cte is
            `a/(gridx - 1)`, and across the line `y` = cte `b/(gridy - 1)`,
            at least ``3``. Default ``50``.
        NLterms : bool, optional
            Whether the non-linear terms of the strains enter the moments,
            see :meth:`.stress`. Default ``True``.
        nr_x_gauss, nr_y_gauss : int or None, optional
            Number of Gauss-Legendre points along `x` and `y`, ``nr_y_gauss``
            required for a line `x` = cte, ``nr_x_gauss`` for a line `y` =
            cte. Default ``None``.

        Returns
        -------
        force_intgn : float
            The integral of ``Qx`` along `y`, or of ``Qy`` along `x`, a
            force.

        Raises
        ------
        NotImplementedError
            When the model of ``eval_panel`` does not have 3 degrees of
            freedom per term.
        ValueError
            When both ``x_cte_force`` and ``y_cte_force`` are given, when
            ``nr_y_gauss`` is missing for a line `x` = cte or ``nr_x_gauss``
            for a line `y` = cte, when the line is outside the panel, or
            when ``gridx`` or ``gridy`` is lower than ``3``.

        Notes
        -----
        In the classical laminated plate theory the transverse shear forces
        are not given by the constitutive relations, and they are obtained
        from the equilibrium of the moments:

        .. math::

            Q_x = M_{xx,x} + M_{xy,y}, \qquad Q_y = M_{xy,x} + M_{yy,y}

        with the derivatives across the line approximated by the
        second-order finite differences of :func:`numpy.gradient`, central,
        or one-sided when the line is closer than one step to an edge, and
        the derivatives along the line by those of :func:`numpy.gradient`
        over the Gauss-Legendre points. The transverse shear stress
        `\tau_{xz}` acts on the faces `x` = cte, whose resultant per unit
        length is `Q_x`, and `\tau_{yz}` on the faces `y` = cte, such that
        the out-of-plane force transmitted across the line `x = x_c` is

        .. math::

            F = \int_0^b Q_x(x_c, y) \, dy \approx \frac{b}{2} \sum_k w_k
                Q_x(x_c, y_k)

        as in the line integral of `Q_x` of the cohesive zone page,
        :ref:`cohesive_zone_load`, and likewise `F = \int_0^a Q_y(x, y_c)
        \, dx` across the line `y = y_c`. For a free edge `y = 0` or `y = b`
        the condition of the classical theory is on the effective shear
        `Q_y + M_{xy,x}`, not on `M_{xy}`: the twisting moment at the ends of
        the line, statically equivalent to concentrated forces, is not
        included in `F`.

        """

        #NOTE the out-of-plane force is obtained from the derivatives of the
        #     moments of the classical laminated plate theory
        if eval_panel is not None and _dofs(eval_panel) != 3:
            raise NotImplementedError(
                "force_out_plane only supports models with 3 DOFs per term "
                "(u, v, w), got model '{0}'".format(eval_panel.model))

        #NOTE only the line integral is enabled, the branches of the other
        #     two integrals are kept for reference and excluded from the
        #     coverage
        line_int = True
        area_int = False
        von_karman_NL_int = False

        # Original one - uses gauss points in both x and y
        if area_int: # pragma: no cover
            # nr_x_gauss = 128
            # nr_y_gauss = 100

            res_stress = self.stress(c=c, group=group,
                                     eval_panel = eval_panel, x_cte_force = x_cte_force, y_cte_force = y_cte_force,
                                     gridx=gridx, gridy=gridy, nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss)

            # Gauss points and weights
            y_gauss, weights_y = roots_legendre(nr_y_gauss)

            x_gauss, weights_x = roots_legendre(nr_x_gauss)

            # Points used for distance to calc derivatives
            if False: # unchanged dx, dy
                dx = x_gauss.copy()
                dy = y_gauss.copy()
            else: # convert to physical dimensions
                dx = (eval_panel.a/2)*(x_gauss + 1)
                dy = (eval_panel.b/2)*(y_gauss + 1)

            # Derivatives wrt x, y
            dMxx_dy, dMxx_dx = np.gradient(res_stress['Mxx'][0], dy, dx)
            dMxy_dy, dMxy_dx = np.gradient(res_stress['Mxy'][0], dy, dx)
            dMyy_dy, dMyy_dx = np.gradient(res_stress['Myy'][0], dy, dx)

            Qx = dMxx_dx + dMxy_dy
            Qy = dMxy_dx + dMyy_dy

            # Line Integral of Qx - considers gauss points in x and y
            if False:
                # FOR XCTE
                Q_tot = Qx + Qy
                # print(Q_tot)
                # HARD CODED - WRONG - CHANGE
                Q_int = Q_tot[:,-1]
                force_intgn = np.dot(weights_y, Q_int)*(eval_panel.b/2)
                print(f'Force {force_intgn} WARNING: hardcoded for the tip for xcte !!!')
                # print(res_stress['Mxx'][0][:,-1])

            # Area integral of q
            else:
                dQx_dy, dQx_dx = np.gradient(Qx, dy, dx)
                dQy_dy, dQy_dx = np.gradient(Qy, dy, dx)

                q = -(dQx_dx + dQy_dy)

                weights_x_mesh, weights_y_mesh = np.meshgrid(weights_x, weights_y, copy=False)
                eff_weight = np.multiply(weights_x_mesh, weights_y_mesh)
                force_intgn = np.sum(np.multiply(eff_weight, q))*(eval_panel.b*eval_panel.a/4)
                print(f'Force (int area q) {force_intgn}')

                return force_intgn

        # Gauss-Legendre points along the line and three equally spaced
        # points across it, for the finite differences of the moments
        if line_int:
            panel = eval_panel
            if x_cte_force is not None and y_cte_force is not None:
                raise ValueError('Integration can only be performed along a '
                                 'single line - specify either x_cte_force or '
                                 'y_cte_force')
            xline = y_cte_force is None
            if xline:
                if nr_y_gauss is None:
                    raise ValueError('nr_y_gauss must be given for a line '
                                     'x = cte')
                ngrid, ngauss, length, width = gridx, nr_y_gauss, panel.b, panel.a
                cte = x_cte_force
                if cte is None:
                    cte = panel.a*(gridx - 2)/(gridx - 1)
            else:
                if nr_x_gauss is None:
                    raise ValueError('nr_x_gauss must be given for a line '
                                     'y = cte')
                ngrid, ngauss, length, width = gridy, nr_x_gauss, panel.a, panel.b
                cte = y_cte_force
            if ngrid < 3:
                raise ValueError('gridx and gridy must be at least 3')
            if not (0. <= cte <= width):
                raise ValueError('The line must be inside the panel')
            step = width/(ngrid - 1)
            if cte - step < 0.:
                across, i_cte = cte + step*np.arange(3.), 0
            elif cte + step > width:
                across, i_cte = cte + step*np.arange(-2., 1.), 2
            else:
                across, i_cte = cte + step*np.arange(-1., 2.), 1
            s_gauss, weights = roots_legendre(ngauss)
            along = (length/2)*(s_gauss + 1)
            x, y = (across, along) if xline else (along, across)

            # moments as in MultiDomain.stress(), rows along y, columns along x
            xs, ys = np.meshgrid(x, y, copy=False)
            c_panel = np.ascontiguousarray(c[panel.col_start: panel.col_end],
                                           dtype=DOUBLE)
            fstrain = modelDB.db[panel.model]['field'].fstrain
            strains = [reshape(e, xs.shape) for e in fstrain(c_panel, panel,
                       np.ascontiguousarray(xs.ravel(), dtype=DOUBLE),
                       np.ascontiguousarray(ys.ravel(), dtype=DOUBLE),
                       self.out_num_cores, NLgeom=int(NLterms))]
            F = panel._get_constant_ABD()
            names = panel._stress_names()
            M = {name: sum(e*F[names.index(name), j] for j, e in enumerate(strains))
                 for name in ('Mxx', 'Myy', 'Mxy')}

            if xline:
                Q = (np.gradient(M['Mxx'], x, axis=1, edge_order=2)
                     + np.gradient(M['Mxy'], y, axis=0))[:, i_cte]
            else:
                Q = (np.gradient(M['Mxy'], x, axis=1)
                     + np.gradient(M['Myy'], y, axis=0, edge_order=2))[i_cte, :]

            force_intgn = np.dot(weights, Q)*(length/2)

            return force_intgn


        if von_karman_NL_int: # pragma: no cover

            res_uvw = self.uvw(c=c, group=group, gridx=gridx, gridy=gridy,
                               nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss,
                    eval_panel=eval_panel)
            res_stress = self.stress(c=c, group=group,
                                     eval_panel = eval_panel, x_cte_force = x_cte_force, y_cte_force = y_cte_force,
                                     gridx=gridx, gridy=gridy,
                                     nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss)

            # Gauss points and weights
            y_gauss, weights_y = roots_legendre(nr_y_gauss)

            x_gauss, weights_x = roots_legendre(nr_x_gauss)

            # Points used for distance to calc derivatives
            if False: # unchanged dx, dy
                dx = x_gauss.copy()
                dy = y_gauss.copy()
            else: # convert to physical dimensions
                dx = (eval_panel.a/2)*(x_gauss + 1)
                dy = (eval_panel.b/2)*(y_gauss + 1)

            # Derivatives wrt x, y
            dw_dy, dw_dx = np.gradient(res_uvw['w'][0], dy, dx)
            d2w_dxdy, d2w_dx2 = np.gradient(dw_dx, dy, dx)
            d3w_dx2dy, d3w_dx3 = np.gradient(d2w_dx2, dy, dx)
            d4w_dx3dy, d4w_dx4 = np.gradient(d3w_dx3, dy, dx)

            d2w_dy2, d2w_dxdy = np.gradient(dw_dy, dy, dx)
            d3w_dy3, d3w_dxdy2 = np.gradient(d2w_dy2, dy, dx)
            d4w_dy4, d4w_dxdy3 = np.gradient(d3w_dy3, dy, dx)

            d4w_dx2dy2, d4w_dx3dy = np.gradient(d3w_dx2dy, dy, dx)

            # ABD
            F = eval_panel._get_constant_ABD()
            D11 = F[3,3]
            D12 = F[3,4]
            D22 = F[4,4]
            D66 = F[5,5]

            Nxx = res_stress['Nxx'][0]
            Nxy = res_stress['Nxy'][0]
            Nyy = res_stress['Nyy'][0]

            pz = D11*d4w_dx4 + 2*(D12 + 2*D66)*d4w_dx2dy2 + D22*d4w_dy4
            - (np.multiply(Nxx, d2w_dx2) + 2*np.multiply(Nxy, d2w_dxdy) + np.multiply(Nyy, d2w_dy2))

            # Area integral
            if True:
                weights_x_mesh, weights_y_mesh = np.meshgrid(weights_x, weights_y, copy=False)
                eff_weight = np.multiply(weights_x_mesh, weights_y_mesh)
                force_intgn = np.sum(np.multiply(eff_weight, pz))*(eval_panel.b*eval_panel.a/4)

                print(f'Force area int qz (CK) {force_intgn} WARNING: hardcoded !!!')

            # Line integral
            else:
                force_intgn = np.dot(weights_y, pz[:,-1])*(eval_panel.b/2)
                print(f'Force line int qz (CK) {force_intgn} WARNING: hardcoded !!!')

            return force_intgn


    def force_out_plane_damage(self, conn, c):
        r"""Area integral of the normal traction of an ``SB_TSL`` connection

        Integrates, with the ``nr_x_gauss`` by ``nr_y_gauss`` Gauss-Legendre
        points of the connection over the domain of the panel ``p1`` of the
        first ``SB_TSL`` connection of ``conn``, the product of the
        stiffness of the interface of :meth:`.calc_k_dmg`, with the damage
        history ``self.dmg_index`` when it exists, by the normal separation
        of :meth:`.calc_separation`, not corrected by
        :meth:`.correct_separation`. The damage history is not modified.

        Parameters
        ----------
        conn : list of dict
            Connections, the first ``SB_TSL`` connection is used.
        c : array-like
            Ritz constants of the assembly.

        Returns
        -------
        force_intgn : float
            Out-of-plane force transmitted by the cohesive interface.
            ``None`` when ``conn`` has no ``'SB_TSL'`` connection.

        Notes
        -----
        The force is

        .. math::

            F = \int_A k^w_{CZ} \Delta_w \, dA

        with `k^w_{CZ} = k_o (1 - d)` from :meth:`.calc_k_dmg` and the full
        normal separation `\Delta_w = w^t - w^b`, the same that enters the
        internal force vector. The corrected separation of
        :meth:`.correct_separation`, with the compressive region behind the
        crack front set to zero, is used only to compute the damage: using it
        here would drop the compressive tractions that balance the tensile
        ones, and the thesis of D'Souza (2024) [1]_ needed to
        scale that integral by the ratio to the reaction at the first
        increment. Computed with the full separation, `F` matches the
        reaction of the prescribed displacement,
        :meth:`.reaction_line_pd_xcte`, by equilibrium of the loaded arm, and
        needs no scaling. See :ref:`cohesive_zone`.

        References
        ----------
        .. [1] D'Souza, N., Multi-Domain Semi-Analytical Cohesive Zone
           Approach, MSc thesis, Delft University of Technology, 2024.
           https://resolver.tudelft.nl/uuid:4709f888-c032-4d9f-97e3-402dfdd47cf1

        """
        for connecti in conn:
            if connecti['func'] == 'SB_TSL':
                nr_x_gauss = connecti['nr_x_gauss']
                nr_y_gauss = connecti['nr_y_gauss']
                tsl_type = connecti['tsl_type']
                p_top = connecti['p1']
                p_bot = connecti['p2']
                k_i = connecti['k_o']
                tau_o = connecti['tau_o']
                G1c = connecti['G1c']

                # Gauss points and weights
                x_gauss, weights_x = roots_legendre(nr_x_gauss)

                y_gauss, weights_y = roots_legendre(nr_y_gauss)

                prev = self.dmg_index if hasattr(self, "dmg_index") else None
                kw_tsl, dmg_index_max, del_d, dmg_index_curr = self.calc_k_dmg(c=c, pA=p_top, pB=p_bot,
                                     nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss, tsl_type=tsl_type,
                                     prev_max_dmg_index=prev, k_i=k_i, tau_o=tau_o, G1c=G1c)
                #NOTE the traction must be computed with the uncorrected
                #      separation, the same that enters the internal force
                #      vector through kC_conn. The corrected separation, with
                #      the compressive region set to zero, is only used to
                #      evaluate the damage. Using it here drops the
                #      compressive tractions that balance the tensile ones
                res_pan_top = self.calc_results(c=c, eval_panel=p_top, vec='w',
                                        nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss)
                res_pan_bot = self.calc_results(c=c, eval_panel=p_bot, vec='w',
                                        nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss)
                del_d_raw = self.calc_separation(res_pan_top, res_pan_bot)
                tau = np.multiply(kw_tsl, del_d_raw)

                force_intgn = 0

                for pty in range(nr_y_gauss):
                    for ptx in range(nr_x_gauss):
                        # Makes it more efficient when reading data (kw_tsl) from memory as memory is read along a row
                        # So also accessing memory in the same way helps it out so its not deleting and reaccessing the
                        # same memory everytime.

                        weight = weights_x[ptx] * weights_y[pty]

                        tau_xieta = tau[pty, ptx]
                        force_intgn += weight * ((p_top.a*p_top.b)/4) * tau_xieta

                return force_intgn


    def get_kC_conn(self, conn=None, finalize=True, c=None):
        r"""Stiffness matrix due to the multidomain connectivities

        These are based on penalty stiffnesses. With the default
        ``conn_method='null-space'`` only the connections that are not
        imposed exactly by :meth:`.get_Z` enter this matrix, see below.

        Parameters
        ----------
        conn : list of dict, optional
            Each element of the list is a dictionary describing a single
            connection pair, for example::

                conn = [dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0)]

            The possible options for ``func`` are:

            - ``'SSxcte'`` and ``'SSycte'``: between 2 skins along an edge
            - ``'BFxcte'`` and ``'BFycte'``: between the base, or skin,
              ``p1`` and the flange ``p2`` of a stiffener, whatever their
              order in the assembly, see
              :mod:`panels.multidomain.connections.kCBFycte` and
              :mod:`panels.multidomain.connections.kCBFxcte`
            - ``'SB'``: between 2 skins connected over an area, e.g. the base
              of a stiffener bonded to a skin, a pad-up of a skin, or the
              arms of a double cantilever beam. The two panels share the
              axes `x, y, z`, `z` being normal to their reference
              surfaces, and ``p1`` is the panel on the positive side of the
              interface along `z`, ``p2`` the one on the negative side,
              whatever their order in the assembly: the lower face of
              ``p1``, `z_1 = d_1 - h_1/2`, is connected to the upper face
              of ``p2``, `z_2 = d_2 + h_2/2`, with the offset ``offset``
              and the thickness of each panel, `u_1(z_1) = u_2(z_2)`,
              `v_1(z_1) = v_2(z_2)` and `w_1 = w_2`, each panel with its
              own kinematics, for all the models, see
              :func:`.skin_base.fkCSB` and :func:`._sb_top_bottom`. For
              example,
              a stiffener base on the side of positive `z` of the skin is
              ``p1`` and the skin ``p2``, while a pad-up on the side of
              negative `z` is ``p2`` and the skin ``p1``. For a double
              cantilever beam with `z` pointing up, ``p1`` is the upper arm
            - ``'SB_TSL'``: between 2 skins connected over an area with a
              traction-separation law (TSL) at the interface, requiring
              ``tsl_type``, ``nr_x_gauss``, ``nr_y_gauss``, ``k_o``,
              ``tau_o`` and ``G1c``, with ``p1`` and ``p2`` as for ``'SB'``,
              so that the normal separation `w_1 - w_2` of the interface is
              positive, opening it, when ``p1`` moves along the positive `z`
              direction with respect to ``p2``. The matrix is computed by matrix
              products, see :meth:`._kC_TSL`, or, with the key
              ``use_kernels=True``, by the slower kernels of
              :mod:`panels.multidomain.connections.kCSB_dmg`, which give the
              same matrix. ``nr_x_gauss`` and ``nr_y_gauss`` are the numbers
              of Gauss-Legendre points of the interface, computed by
              :func:`scipy.special.roots_legendre` without an upper limit;
              the terms and points along `x` that resolve the cohesive zone
              are given by :func:`.tsl_discretization`

            The panels of ``'SB'`` and ``'SB_TSL'`` must have the same
            dimensions `a`, `b`, otherwise a ``ValueError`` is raised, see
            :func:`._sb_top_bottom`.

            The penalty constants default to the values of
            :func:`.calc_kt_kr`. They can be given for each connection with
            the keys ``'kt'`` and ``'kr'``; for the shear deformation theories
            ``'SB'`` connects the rotations only when ``'kr'`` is given, with
            units of force/length, and for all the models the penalty
            ``'kt'`` of ``'SB'`` has units of force/length**3 and must be
            consistent with the units of the model. The key ``'kr'`` of
            ``'SB'`` applies only to the FSDT and TSDT: the rotations of the
            classical laminated plate theory follow from `w`, already tied
            by `w_1 = w_2`, and a ``'kr'`` given to these models is not used,
            with a ``UserWarning``, see :func:`.skin_base.fkCSB`.

            For the models based on shear deformation theories, e.g.
            ``'plate_fsdt_donnell'`` or ``'cylshell_tsdt_sanders'``,
            ``'SSxcte'``, ``'SSycte'`` and ``'SB'`` are available, see
            :mod:`panels.multidomain.connections.kCsdt`, and ``'BFycte'`` and
            ``'BFxcte'``, see the functions ``*_sdt`` of
            :mod:`panels.multidomain.connections.kCBFycte` and
            :mod:`panels.multidomain.connections.kCBFxcte`, and
            :meth:`._kC_conn_sdt`. ``'SB_TSL'`` is not available for these
            models. The edge connections also connect the independent
            rotations with ``'kr'``, and the rotation flags of the connected
            edges (e.g. ``x2phiy``) must be released as the other flags.

            The dictionaries are not modified.

            If ``None``, the connectivity defined for the assembly is used.

            With ``conn_method='null-space'`` the connections of
            :data:`.nullspace.NULL_SPACE_FUNCS` are imposed exactly by
            :meth:`.get_Z` and do not enter this matrix, which then contains
            only the connections ``'SB_TSL'``.
        finalize : bool, optional
            Asserts validity of output data and makes the output matrix
            symmetric.
        c : array-like or None, optional
            Ritz constants, used by the damaged connections, whose
            stiffness is computed with :meth:`.calc_k_dmg` at ``c``.

        Returns
        -------
        kC_conn : sparse matrix
            Matrix with shape ``(size, size)``, ``size`` from
            :meth:`.get_size`, symmetric when ``finalize=True``, otherwise
            with the upper triangle only. It is also stored in the attribute
            ``kC_conn``, and the connections used are kept for the later
            calls without ``conn``, see :meth:`.get_Z`. The damage history
            of the ``'SB_TSL'`` connections, ``self.dmg_index``, is read and
            not modified.

        Raises
        ------
        ValueError
            When a connection has ``p1`` and ``p2`` that are the same panel,
            or a ``'func'`` that is not recognized.

        Notes
        -----
        The connections are based on penalty stiffnesses, as detailed in
        Castro and Donadon (2017) [1]_. For each quantity `q_k` that must
        be continuous across a connection, the penalty energy is

        .. math::

            U_k = \frac{1}{2} k_k \int \left( q_k^{(1)} - q_k^{(2)}
                  \right)^2

        where `k_k` is the translation or rotation penalty constant,
        ``'kt'`` or ``'kr'``, and the integral is along the connected edges,
        or over the area of ``'SB'``. Since `q_k^{(1)}` and `q_k^{(2)}` are
        linear in the Ritz constants of the two panels, expanding the square
        gives the blocks 11, 12 and 22 of the connection matrix, of the
        kernels ``fk*11``, ``fk*12`` and ``fk*22`` of
        :mod:`panels.multidomain.connections`; only the upper triangle is
        assembled, the block 12 being transposed by :func:`._upper_block`
        when ``p1`` comes after ``p2``, and the matrix is made symmetric with
        ``finalize=True``. For the classical laminated plate theory the
        quantities are `u, v, w` and the slope `w_{,x}` for ``'SSxcte'`` or
        `w_{,y}` for ``'SSycte'``; for ``'BFycte'`` and ``'BFxcte'`` the
        displacements of the flange, rotated by 90 degrees about the axis of
        the connection, and the rotation about that axis, see
        :func:`._rinv_sanders` and :func:`._bf_base_flange`. For ``'SB'``,
        for all the models, the quantities are the displacements of the
        faces in contact, `u_1(z_1)`, `v_1(z_1)` and `w_1` against
        `u_2(z_2)`, `v_2(z_2)` and `w_2`, with `z_1 = d_1 - h_1/2`, `z_2 =
        d_2 + h_2/2`, `d_1` and `d_2` being the offsets, `u(z) = u - z
        w_{,x}` and `v(z) = (1 + z/r) v - z w_{,y}` for the classical
        laminated plate theory, with the slope of each panel and the term
        `z v/r` only with the Sanders-Koiter kinematics, see
        :func:`.skin_base.fkCSB` and :func:`._sb_top_bottom`, and the
        rotations of the normal of the shear deformation theories when
        ``'kr'`` is given; ``'kr'`` is not used for the classical laminated
        plate theory, whose rotations follow from `w`. The other connections of the shear deformation
        theories are described in :meth:`._kC_conn_sdt`.

        The continuity is then only approximate. Low penalty constants make
        the assembly too flexible, and high ones make the stiffness matrix
        ill-conditioned; the default constants of :func:`.calc_kt_kr` are a
        compromise. The connections of :data:`.nullspace.NULL_SPACE_FUNCS`
        are imposed exactly with ``conn_method='null-space'``, see
        :meth:`.get_Z`.

        The connection ``'SB_TSL'`` is a cohesive interface whose secant
        stiffness `k^w_{CZ} = k_o (1 - d)`, the same for the three components
        of the separation, is computed at ``c`` at each Gauss-Legendre point
        of the interface by :meth:`.calc_k_dmg`, with the damage history
        ``self.dmg_index``, and integrated by :meth:`._kC_TSL` or by the
        kernels of :mod:`panels.multidomain.connections.kCSB_dmg`. The
        matrix is the secant stiffness of the interface, `\{f^{coh}_{int}\}
        = [K^{conn}_{dmg}] \{c\}`; its consistent tangent also needs the
        damage-rate part of :meth:`.calc_kT_TSL`. See :ref:`cohesive_zone`.

        References
        ----------
        .. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
           models to Address Linear Buckling and Vibration of Stiffened
           Composite Panels with Debonding Defect, Composite Structures, 2017.
           https://doi.org/10.1016/j.compstruct.2016.10.026

        """
        conn = self._resolve_conn(conn)
        self._conn_used = conn

        size = self.get_size()

        kC_conn = coo_matrix((size, size), dtype=DOUBLE)

        # Looping through each connection pair
        for connecti in conn:
            if self._null_space(connecti):
                continue
            if connecti['func'] != 'SB_force':
                #NOTE a local copy is used, because the coordinates of the
                #     connection are swapped below when p1 comes after p2 in
                #     the assembly. Swapping them in the user's dictionary
                #     would swap them back in every other call, for instance
                #     calc_kC() followed by calc_kT()
                connecti = dict(connecti)
                # connecti = ith connection pair
                p1_temp = connecti['p1']
                p2_temp = connecti['p2']
                if p1_temp is p2_temp:
                    raise ValueError('A connection must be between two '
                                     'different panels')
                # pA and pB are the two panels that are finally passed on
                if p1_temp.col_start < p2_temp.col_start:
                    pA = p1_temp
                    pB = p2_temp
                elif p1_temp.col_start > p2_temp.col_start:
                    pA = p2_temp
                    pB = p1_temp
                    #NOTE the kernels of 'SB', 'SB_TSL', 'BFycte' and
                    #     'BFxcte' distinguish p1 from p2, see _bf_base_flange()
                    if connecti['func'] in ('SB', 'SB_TSL', 'BFycte', 'BFxcte'):
                        pass
                    else:
                        if 'xcte1' in connecti.keys() and 'xcte2' in connecti.keys():
                            temp_xcte = connecti['xcte1']
                            connecti['xcte1'] = connecti['xcte2']
                            connecti['xcte2'] = temp_xcte
                        if 'ycte1' in connecti.keys() and 'ycte2' in connecti.keys():
                            temp_ycte = connecti['ycte1']
                            connecti['ycte1'] = connecti['ycte2']
                            connecti['ycte2'] = temp_ycte

                connection_function = connecti['func'] # Type of connection

                #NOTE models based on shear deformation theories, with the 5
                #     DOFs u, v, w, phix, phiy per term
                if _dofs(pA) != 3 or _dofs(pB) != 3:
                    kC_conn += self._kC_conn_sdt(connecti, pA, pB, size)
                    continue

                if connection_function == 'SSycte':
                    # ftn in panels/multidomain/connections/penalties.py
                    kt, kr = _penalties(connecti, pA, pB, 'ycte')

                    # ftn in panels/panels/multidomain/connections
                    # Eq 32 MD paper - expanding squares gives i^2, ij, ji and j^2 which form
                    #       the 11, 12, 21 and 22 terms of the kC_conn matrix
                    # adds the penalty stiffness to ycte of panel pA position
                    kC_conn += connections.kCSSycte.fkCSSycte11(
                            kt, kr, pA, connecti['ycte1'],
                            size, row0=pA.row_start, col0=pA.col_start)
                    # adds the penalty stiffness to ycte of panel 1 and panel 2 coupling position
                    kC_conn += connections.kCSSycte.fkCSSycte12(
                            kt, kr, pA, pB, connecti['ycte1'], connecti['ycte2'],
                            size, row0=pA.row_start, col0=pB.col_start)
                    # adds the penalty stiffness to ycte of panel pB position
                    kC_conn += connections.kCSSycte.fkCSSycte22(
                            kt, kr, pA, pB, connecti['ycte2'],
                            size, row0=pB.row_start, col0=pB.col_start)

                elif connection_function == 'SSxcte':
                    kt, kr = _penalties(connecti, pA, pB, 'xcte')
                    kC_conn += connections.kCSSxcte.fkCSSxcte11(
                            kt=kt, kr=kr, p1=pA, xcte1=connecti['xcte1'],
                            size=size, row0=pA.row_start, col0=pA.col_start)
                    kC_conn += connections.kCSSxcte.fkCSSxcte12(
                            kt=kt, kr=kr, p1=pA, p2=pB, xcte1=connecti['xcte1'], xcte2=connecti['xcte2'],
                            size=size, row0=pA.row_start, col0=pB.col_start)
                    kC_conn += connections.kCSSxcte.fkCSSxcte22(
                            kt=kt, kr=kr, p1=pA, p2=pB, xcte2=connecti['xcte2'],
                            size=size, row0=pB.row_start, col0=pB.col_start)

                elif connection_function == 'BFycte':
                    pb, pf = _bf_base_flange(connecti)
                    kt, kr = _penalties(connecti, pA, pB, 'ycte')
                    #NOTE the rotation about x of a panel with the Sanders-
                    #     Koiter kinematics is w,y - v/r, see kCBFycte.pyx.
                    #     The rotation about y of 'BFxcte' is -w,x for both
                    #     kinematics
                    rinvb = _rinv_sanders(pb)
                    rinvf = _rinv_sanders(pf)
                    kC_conn += connections.kCBFycte.fkCBFycte11(
                            kt, kr, pb, connecti['ycte1'],
                            size, row0=pb.row_start, col0=pb.col_start,
                            rinv1=rinvb)
                    kC_conn += _upper_block(connections.kCBFycte.fkCBFycte12(
                            kt, kr, pb, pf, connecti['ycte1'], connecti['ycte2'],
                            size, row0=pb.row_start, col0=pf.col_start,
                            rinv1=rinvb, rinv2=rinvf), pb, pf)
                    kC_conn += connections.kCBFycte.fkCBFycte22(
                            kt, kr, pb, pf, connecti['ycte2'],
                            size, row0=pf.row_start, col0=pf.col_start,
                            rinv2=rinvf)

                elif connection_function == 'BFxcte':
                    pb, pf = _bf_base_flange(connecti)
                    kt, kr = _penalties(connecti, pA, pB, 'xcte')
                    kC_conn += connections.kCBFxcte.fkCBFxcte11(
                            kt, kr, pb, connecti['xcte1'],
                            size, row0=pb.row_start, col0=pb.col_start)
                    kC_conn += _upper_block(connections.kCBFxcte.fkCBFxcte12(
                            kt, kr, pb, pf, connecti['xcte1'], connecti['xcte2'],
                            size, row0=pb.row_start, col0=pf.col_start), pb, pf)
                    kC_conn += connections.kCBFxcte.fkCBFxcte22(
                            kt, kr, pb, pf, connecti['xcte2'],
                            size, row0=pf.row_start, col0=pf.col_start)

                elif connection_function == 'SB':
                    #NOTE p1 is the panel on the positive side of the
                    #     interface along z and p2 the one on the negative
                    #     side, whatever their order in the assembly, see
                    #     _sb_top_bottom()
                    p_top, p_bot = _sb_top_bottom(connecti)
                    kt, _ = _penalties(connecti, p_top, p_bot, 'bot-top')
                    #NOTE the faces in contact with the kinematics of each
                    #     panel, the same matrix for all the models, upper
                    #     triangle whatever the order of the panels
                    kC_conn += connections.skin_base.fkCSB(kt, p_top, p_bot,
                            size, kr=connecti.get('kr', 0.))

                # Traction Seperation Law introduced at the interface
                elif connection_function == 'SB_TSL': # and c is not None:
                    # Executed when c is not None i.e some displacements have occured so damage 'might' be created
                    tsl_type = connecti['tsl_type']
                    k_i = connecti['k_o']
                    tau_o = connecti['tau_o']
                    G1c = connecti['G1c']

                    nr_x_gauss = connecti['nr_x_gauss']
                    nr_y_gauss = connecti['nr_y_gauss']
                    p_top, p_bot = _sb_top_bottom(connecti)

                    if hasattr(self, "dmg_index"):
                        kw_tsl, dmg_index_max, del_d, dmg_index_curr = self.calc_k_dmg(c=c, pA=p_top, pB=p_bot,
                                             nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss, tsl_type=tsl_type,
                                             prev_max_dmg_index=self.dmg_index, k_i=k_i, tau_o=tau_o, G1c=G1c)
                    else:
                        kw_tsl, dmg_index_max, del_d, dmg_index_curr = self.calc_k_dmg(c=c, pA=p_top, pB=p_bot,
                                             nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss, tsl_type=tsl_type,
                                             prev_max_dmg_index=None, k_i=k_i, tau_o=tau_o, G1c=G1c)

                    # Overwriting kw_tsl to the original value
                    if False:
                        kw_tsl = np.zeros_like(del_d)
                        kw_tsl[:,:] = k_i
                        print(f'kw_tsl_overwrite kw_tsl min: {np.min(kw_tsl):.3e}')
                    # print(f'   kw MD class {np.min(kw_tsl):.1e}      dmg {np.max(dmg_index):.3f}')

                    #NOTE the tangential separation uses the slope of each
                    #      panel, dt and db are the distances from each
                    #      mid-plane to the interface, see _tsl_operators(),
                    #      the docstring of kCSB_dmg.pyx and
                    #      doc/source/cohesive_zone.rst. The kernels and
                    #      the matrix products of _kC_TSL() give the same
                    #      matrix, both require panels of the same
                    #      dimensions, checked by _sb_top_bottom()
                    if connecti.get('use_kernels', False):
                        dt, db = connections.kCsdt._sb_distances(p_top, p_bot)
                        rinvt = _rinv_sanders(p_top)
                        rinvb = _rinv_sanders(p_bot)
                        kC_conn += connections.kCSB_dmg.fkCSB11_dmg(dt=dt, p1=p_top,
                                size=size, row0=p_top.row_start, col0=p_top.col_start,
                                nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss, kw_tsl=kw_tsl,
                                rinv1=rinvt)
                        kC_conn += _upper_block(connections.kCSB_dmg.fkCSB12_dmg(
                                dt=dt, db=db, p1=p_top, p2=p_bot,
                                size=size, row0=p_top.row_start, col0=p_bot.col_start,
                                nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss, kw_tsl=kw_tsl,
                                rinv1=rinvt, rinv2=rinvb),
                                p_top, p_bot)
                        kC_conn += connections.kCSB_dmg.fkCSB22_dmg(db=db, p1=p_top, p2=p_bot,
                                size=size, row0=p_bot.row_start, col0=p_bot.col_start,
                                nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss, kw_tsl=kw_tsl,
                                rinv2=rinvb)
                    else:
                        kC_conn += self._kC_TSL(p_top, p_bot, nr_x_gauss,
                                nr_y_gauss, k_i, kw_tsl, size)

                else:
                    raise ValueError(f'{connection_function} not recognized. Provide a correct function if you expect results')

        if finalize:
            kC_conn = finalize_symmetric_matrix(kC_conn)
        self.kC_conn = kC_conn

        return kC_conn


    def _kC_conn_sdt(self, connecti, pA, pB, size):
        r"""Connection matrix between panels of the models based on shear
        deformation theories, see :mod:`panels.multidomain.connections.kCsdt`
        and the functions ``*_sdt`` of
        :mod:`panels.multidomain.connections.kCBFycte` and
        :mod:`panels.multidomain.connections.kCBFxcte`

        ``pA`` comes before ``pB`` in the assembly and the coordinates of the
        connection in ``connecti`` are already ordered accordingly.

        Notes
        -----
        The models based on shear deformation theories have the independent
        rotations `\phi_x` and `\phi_y`, which the edge connections ``'SSxcte'``
        and ``'SSycte'`` penalize with ``'kr'`` together with `u, v, w`, and
        for the third-order theory also the slope `w_{,x}` or `w_{,y}`. The
        rotation penalty of ``'BFycte'`` and ``'BFxcte'`` is on `\phi_y`
        (`\phi_y + v/r` with the Sanders-Koiter kinematics) and `\phi_x`
        instead of `-w_{,y}` and `-w_{,x}`. ``'SB'`` penalizes the
        displacements `u, v, w` at the interface, `z = d_1 - h_1/2` of
        ``p1`` and `z = d_2 + h_2/2` of ``p2`` with the offsets `d_1`,
        `d_2`, as for the classical laminated plate theory, and the
        difference of the rotations only when ``'kr'`` is given and not
        zero, which only these models do, see :func:`.skin_base.fkCSB`. The
        default rotation
        penalty does not grow as the domains become narrower, so thin and
        narrow domains may need a higher ``'kr'`` to converge to the
        single-domain result. See
        :meth:`.get_kC_conn` for the penalty energy.

        """
        func = connecti['func']
        if _dofs(pA) != _dofs(pB):
            raise NotImplementedError(
                "Connection '{0}' between models with a different number of "
                "DOFs per term, got models '{1}' and '{2}'".format(func,
                pA.model, pB.model))
        kCsdt = connections.kCsdt
        if func == 'SSxcte':
            kt, kr = _penalties(connecti, pA, pB, 'xcte')
            return kCsdt.fkCSSxcte_sdt(kt, kr, pA, pB, connecti['xcte1'],
                                       connecti['xcte2'], size)
        elif func == 'SSycte':
            kt, kr = _penalties(connecti, pA, pB, 'ycte')
            return kCsdt.fkCSSycte_sdt(kt, kr, pA, pB, connecti['ycte1'],
                                       connecti['ycte2'], size)
        elif func == 'SB':
            p_top, p_bot = _sb_top_bottom(connecti)
            kt, _ = _penalties(connecti, p_top, p_bot, 'bot-top')
            kr = connecti.get('kr', 0.)
            return kCsdt.fkCSB_sdt(kt, p_top, p_bot, size, kr=kr)
        elif func in ('BFycte', 'BFxcte'):
            #NOTE the rotation penalty is on phiy ('BFycte') or phix
            #     ('BFxcte'), see kCBFycte.pyx and kCBFxcte.pyx
            pb, pf = _bf_base_flange(connecti)
            cte = func[2:]
            mod = getattr(connections, 'kCBF' + cte)
            kt, kr = _penalties(connecti, pA, pB, cte)
            c1, c2 = connecti[cte + '1'], connecti[cte + '2']
            #NOTE with the Sanders-Koiter kinematics the rotation about x is
            #     phiy + v/r, the rotation about y of 'BFxcte' is phix
            kw1, kw12, kw2 = {}, {}, {}
            if cte == 'ycte':
                rb, rf = _rinv_sanders(pb), _rinv_sanders(pf)
                kw1, kw12, kw2 = dict(rinv1=rb), dict(rinv1=rb, rinv2=rf), dict(rinv2=rf)
            k11 = getattr(mod, 'fkCBF%s11_sdt' % cte)(kt, kr, pb, c1, size,
                    pb.row_start, pb.col_start, **kw1)
            k12 = getattr(mod, 'fkCBF%s12_sdt' % cte)(kt, kr, pb, pf, c1, c2,
                    size, pb.row_start, pf.col_start, **kw12)
            k22 = getattr(mod, 'fkCBF%s22_sdt' % cte)(kt, kr, pb, pf, c2, size,
                    pf.row_start, pf.col_start, **kw2)
            return k11 + _upper_block(k12, pb, pf) + k22
        raise NotImplementedError(
            "Connection '{0}' is not implemented for the models based on "
            "shear deformation theories, got models '{1}' and '{2}'".format(
            func, pA.model, pB.model))


    def calc_kC(self, conn=None, c=None, silent=True, finalize=True, inc=1.,
            NLgeom=False):
        r"""Calculate the constitutive stiffness matrix of the assembly

        The matrices :meth:`.Shell.calc_kC` of the panels are assembled at
        their offsets ``row_start`` and ``col_start``, symmetrized when
        ``finalize=True``, and the matrix of the connections,
        :meth:`.get_kC_conn` at ``c``, is added. With ``NLgeom=True`` and
        ``finalize=True`` the load stiffness of the follower pressure loads
        of the panels is added last. The penalty stiffnesses of the
        prescribed displacements are not included. The result is stored in
        the attribute ``kC``, and the matrix of the connections in
        ``kC_conn``.

        Parameters
        ----------

        conn : list of dict, optional
            A connectivity dictionary, see :meth:`.get_kC_conn`. Optional if
            already defined for the assembly.
        c : array-like or None, optional
            This must be the result of a static analysis, used to compute
            non-linear terms based on the actual displacement field.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        finalize : bool, optional
            Asserts validity of output data and makes the output matrix
            symmetric, should be ``False`` when assemblying.
        inc : float, optional
            Load factor of the incremented follower pressure loads of the
            panels, see :meth:`.Shell.calc_kCfollower`.
        NLgeom : bool, optional
            If ``True``, the constitutive part of the tangent stiffness matrix
            at ``c`` is calculated, see :meth:`.Shell.calc_kC`, plus the load
            stiffness of the follower pressure loads of the panels when
            ``finalize=True``.

        Returns
        -------
        kC : sparse matrix
            Matrix with shape ``(size, size)``, ``size`` from
            :meth:`.get_size`.

        Raises
        ------
        ValueError
            When the attribute ``row_start`` or ``col_start`` of a panel is
            ``None``.

        Notes
        -----
        This is kP (made by diagonally assemblying kP_i from the MD paper)
        [1]_, the block-diagonal matrix of the constitutive stiffness matrices
        of the panels, to which the matrix of the connections is added:

        .. math::

            [K_C] = \sum_p [L_p]^T [K_C^{(p)}] [L_p] + [K_C^{conn}]

        where `[L_p]` selects the degrees of freedom of the panel ``p``. The
        total potential energy of the assembly is the sum of those of the
        panels and of the penalty energies of the connections, see
        :meth:`.get_kC_conn`; with ``conn_method='null-space'`` the
        connections imposed exactly are not in `[K_C^{conn}]`, see
        :meth:`.get_Z`. The load stiffness of the follower pressure loads
        is not symmetric, therefore it is added after the symmetrization,
        see :meth:`.Shell.calc_kCfollower`.

        References
        ----------
        .. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
           models to Address Linear Buckling and Vibration of Stiffened
           Composite Panels with Debonding Defect, Composite Structures, 2017.
           https://doi.org/10.1016/j.compstruct.2016.10.026

        """
        size = self.get_size()
        msg('Calculating kC for assembly...', level=2, silent=silent)
        if c is not None: # If c passed, check if its correct (dim and dtype)
            check_c(c, size)

        kC = 0.
        for p in self.panels:
            if p.row_start is None or p.col_start is None:
                raise ValueError('Shell attributes "row_start" and "col_start" must be defined!')
            # Calc Kc per panel (from the Shell class)
            kC += p.calc_kC(c=c, row0=p.row_start, col0=p.col_start, size=size,
                    silent=silent, finalize=False, NLgeom=NLgeom)

        # Make the matrix symm at the end
        if finalize:
            kC = finalize_symmetric_matrix(kC)

        # NOTE move this to another class method? it's a bid hidden
        # Adding kC_conn to KC panel
        kC_conn = self.get_kC_conn(conn=conn, c=c)

        kC += kC_conn

        #NOTE unsymmetric, added after the symmetrization, see Shell.calc_kC
        if NLgeom and finalize:
            kC = kC + self._calc_kCfollower(c, inc, silent)

        self.kC = kC
        msg('finished!', level=2, silent=silent)

        return kC


    def calc_kG(self, c=None, silent=False, finalize=True, NLgeom=False):
        r"""Calculate the geometric stiffness matrix of the assembly

        The matrices :meth:`.Shell.calc_kG` of the panels are assembled at
        their offsets ``row_start`` and ``col_start`` and symmetrized when
        ``finalize=True``. The connections do not contribute. The result is
        stored in the attribute ``kG``.

        Parameters
        ----------

        c : array-like or None, optional
            This must be the result of a static analysis, used to compute
            non-linear terms based on the actual displacement field.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
            Default ``False``.
        finalize : bool, optional
            Asserts validity of output data and makes the output matrix
            symmetric, should be ``False`` when assemblying.
        NLgeom : bool, optional
            If ``True``, the geometric part of the tangent stiffness matrix at
            ``c`` is calculated, see :meth:`.Shell.calc_kG`.

        Returns
        -------
        kG : sparse matrix
            Matrix with shape ``(size, size)``, ``size`` from
            :meth:`.get_size`.

        Raises
        ------
        ValueError
            When the attribute ``row_start`` or ``col_start`` of a panel is
            ``None``.

        Notes
        -----
        `[K_G] = \sum_p [L_p]^T [K_G^{(p)}] [L_p]`, where `[L_p]` selects the
        degrees of freedom of the panel ``p``. The penalty energies of the
        connections are quadratic forms of the Ritz constants and the
        constraints of the null-space method are linear, so that neither has
        a geometric stiffness.

        """
        size = self.get_size()
        msg('Calculating kG for assembly...', level=2, silent=silent)
        if c is not None:
            check_c(c, size)

        kG = 0.
        for p in self.panels:
            if p.row_start is None or p.col_start is None:
                raise ValueError('Shell attributes "row_start" and "col_start" must be defined!')
            kG += p.calc_kG(c=c, row0=p.row_start, col0=p.col_start, size=size,
                            silent=silent, finalize=False, NLgeom=NLgeom)
        if finalize:
            kG = finalize_symmetric_matrix(kG)
        self.kG = kG
        msg('finished!', level=2, silent=silent)
        return kG


    def _calc_kCfollower(self, c, inc, silent):
        r"""Load stiffness of the follower pressure loads of the panels,
        unsymmetric, see :meth:`.Shell.calc_kCfollower`"""
        size = self.get_size()
        kCf = csr_matrix((size, size))
        for p in self.panels:
            if p.has_follower_loads():
                kCf = kCf + p.calc_kCfollower(c=c, inc=inc, size=size,
                        row0=p.row_start, col0=p.col_start, silent=silent)
        return kCf


    def calc_kM(self, silent=False, finalize=True):
        r"""Calculate the mass matrix of the assembly

        The matrices :meth:`.Shell.calc_kM` of the panels are assembled at
        their offsets ``row_start`` and ``col_start`` and symmetrized when
        ``finalize=True``. The connections do not contribute. The result is
        stored in the attribute ``kM``.

        Parameters
        ----------
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
            Default ``False``.
        finalize : bool, optional
            Makes the output matrix symmetric, should be ``False`` when
            assemblying. Default ``True``.

        Returns
        -------
        kM : sparse matrix
            Matrix with shape ``(size, size)``, ``size`` from
            :meth:`.get_size`.

        Raises
        ------
        ValueError
            When the attribute ``row_start`` or ``col_start`` of a panel is
            ``None``.

        Notes
        -----
        `[M] = \sum_p [L_p]^T [M^{(p)}] [L_p]`, where `[L_p]` selects the
        degrees of freedom of the panel ``p``: the connections, penalty
        springs or constraints, carry no mass.

        """
        msg('Calculating kM for assembly...', level=2, silent=silent)
        size = self.get_size()
        kM = 0
        for p in self.panels:
            if p.row_start is None or p.col_start is None:
                raise ValueError('Shell attributes "row_start" and "col_start" must be defined!')
            kM += p.calc_kM(row0=p.row_start, col0=p.col_start, size=size,
                            silent=silent, finalize=False)
        if finalize:
            kM = finalize_symmetric_matrix(kM)
        self.kM = kM
        msg('finished!', level=2, silent=silent)
        return kM


    def calc_kT(self, c=None, silent=True, finalize=True, inc=None, kC_conn=None):
        r"""Tangent stiffness matrix of the assembly

        The tangent stiffness matrix of :meth:`.MultiDomain.calc_fint`: the
        matrices :meth:`.Shell.calc_kC` and :meth:`.Shell.calc_kG` with
        ``NLgeom=True`` at ``c`` of each panel, i.e. the two terms of
        :meth:`.Shell.calc_kT`, assembled at the offsets ``row_start`` and
        ``col_start`` of the panel, plus the matrix of the connections,
        :meth:`.MultiDomain.get_kC_conn`. The panels are assembled with
        ``finalize=False``, each filling its upper triangle, and the sum is
        symmetrized once. The load stiffness of the follower pressure loads
        of the panels, :meth:`.Shell.calc_kCfollower` at the load factor
        ``inc``, unsymmetric, is added after the symmetrization, when
        ``finalize=True``.

        Parameters
        ----------
        c : np.ndarray or None, optional
            Ritz constants of the assembly, of size :meth:`.get_size`.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        finalize : bool, optional
            Symmetrize the sum of the panels and add the load stiffness of
            the follower pressure loads.
        inc : float or None, optional
            Load factor of the follower pressure loads, ``None`` for ``1.``.
        kC_conn : sparse matrix or None, optional
            The matrix of the connections, by default
            :meth:`.get_kC_conn` at ``c``. The stiffness of a damaged
            cohesive connection, ``'SB_TSL'``, is part of it, see
            :meth:`.get_kC_conn`.

        Returns
        -------
        kT : sparse matrix
            The tangent stiffness matrix, also stored in the attribute
            ``kT``.

        Raises
        ------
        ValueError
            When the attribute ``row_start`` or ``col_start`` of a panel is
            ``None``.

        Notes
        -----
        The exact Jacobian of :meth:`.MultiDomain.calc_fint`:

        .. math::
            [K_T] = \sum_p [L_p]^T \left( [K_C^{(p)}] + [K_G^{(p)}] \right)
                    [L_p] + [K_C^{conn}]

        where, for each panel ``p``, `[K_C^{(p)}]` and `[K_G^{(p)}]` are
        :meth:`.Shell.calc_kC` and :meth:`.Shell.calc_kG` with
        ``NLgeom=True`` at ``c``, the operator `[L_p]` selecting its
        degrees of freedom, and `[K_C^{conn}]` is the matrix of the
        connections, :meth:`.MultiDomain.get_kC_conn`, constant for the
        penalty connections, whose energy is quadratic. See
        :ref:`tangent_clpt_donnell_multidomain`. The secant stiffness of a
        damaged cohesive connection ``'SB_TSL'`` depends on ``c`` through
        the damage, and where the damage grows the exact Jacobian also
        contains the damage-rate part of :meth:`.calc_kT_TSL`, which is not
        added here.

        """
        msg('Calculating kT for assembly...', level=2, silent=silent)
        size = self.get_size()
        kT = 0
        #TODO use multiprocessing.Pool here

        # Parallelizing doesnt work
        # Parallelizing the calc of kT of all panels separately
        # Only parallelize this if the main function is not parallized. Otherwise parallelize that
        # so that many runs can be performed simultaneously and this runs on one core
        for p in self.panels:
            if p.row_start is None or p.col_start is None:
                raise ValueError('Shell attributes "row_start" and "col_start" must be defined!')
            # Both kC and kG are not symmetric now
            kT += p.calc_kC(c=c, size=size, row0=p.row_start, col0=p.col_start,
                    silent=silent, finalize=False, NLgeom=True) # inc=inc,  - REMOVED bec not in shell
            kT += p.calc_kG(c=c, size=size, row0=p.row_start,
                    col0=p.col_start, silent=silent, finalize=False, NLgeom=True)

        if finalize:
            kT = finalize_symmetric_matrix(kT)
            #NOTE unsymmetric, added after the symmetrization
            kT = kT + self._calc_kCfollower(c, 1. if inc is None else inc,
                                            silent)
        if kC_conn is None:
            kC_conn = self.get_kC_conn(c=c)
        kT += kC_conn
        # print(f'kCconn {np.max(kC_conn):.3e}')
        self.kT = kT
        msg('finished!', level=2, silent=silent)
        return kT


    def calc_kTv(self, c, v, silent=True, inc=1.):
        r"""Product of the tangent stiffness matrix of the assembly by a
        vector, without assembly

        Returns the product by ``v`` of the tangent stiffness matrix at
        ``c``, the sum of :meth:`.calc_kC` and :meth:`.calc_kG` with
        ``NLgeom=True`` at the load factor ``inc``, i.e. the directional
        derivative of :meth:`.calc_fint` along ``v``. The product of each
        panel comes from :meth:`.Shell.calc_kTv`, whose cost is about that
        of :meth:`.Shell.calc_fint`, and the connections add
        ``get_kC_conn(c=c) @ v``.

        Parameters
        ----------
        c : array-like
            Ritz constants of the assembly.
        v : array-like
            The vector, with the size of ``c``.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        inc : float, optional
            Load factor of the incremented follower pressure loads of the
            panels, see :meth:`.Shell.calc_kCfollower`.

        Returns
        -------
        kTv : np.ndarray
            The product.

        Raises
        ------
        ValueError
            When the attribute ``col_start`` of a panel is ``None``.

        Notes
        -----
        `[K_T(c)] \{v\} = \left. \frac{d}{d\epsilon} \{F_{int}(c + \epsilon
        v)\} \right|_{\epsilon = 0}`, computed without assembling `[K_T]`,
        e.g. for the product `[K_T(c)] \{c_{p1}\}` of
        :meth:`.get_reduced_functions`. For a cohesive connection
        ``'SB_TSL'`` only its secant stiffness enters the product, without
        the damage-rate part of :meth:`.calc_kT_TSL`.

        """
        msg('Calculating kT*v for assembly...', level=2, silent=silent)
        size = self.get_size()
        c = np.asarray(c, dtype=DOUBLE)
        v = np.asarray(v, dtype=DOUBLE)
        check_c(c, size)
        check_c(v, size)
        kTv = np.zeros(size, dtype=DOUBLE)
        for p in self.panels:
            if p.col_start is None:
                raise ValueError('Shell attributes "col_start" must be defined!')
            kTv += p.calc_kTv(c, v, size=size, col0=p.col_start,
                              silent=silent, inc=inc)
        kTv += self.get_kC_conn(c=c) @ v
        msg('finished!', level=2, silent=silent)
        return kTv


    def calc_fint(self, c, silent=True, inc=1., kC_conn=None):
        r"""Internal force vector of the assembly

        The sum of :meth:`.Shell.calc_fint` of each panel at the load factor
        ``inc``, which includes the part of the follower pressure loads that
        depends on the configuration, assembled at the offset ``col_start``
        of the panel, and of the product of the matrix of the connections,
        :meth:`.get_kC_conn`, by ``c``. :meth:`.MultiDomain.calc_kT` gives
        its tangent stiffness matrix.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, of size :meth:`.get_size`.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        inc : float, optional
            Load factor of the follower pressure loads.
        kC_conn : sparse matrix or None, optional
            The matrix of the connections, by default
            :meth:`.get_kC_conn` at ``c``.

        Returns
        -------
        fint : np.ndarray
            The internal force vector, also stored in the attribute
            ``fint``.

        Raises
        ------
        ValueError
            When the attribute ``col_start`` of a panel is ``None``.

        Notes
        -----
        .. math::
            \{F_{int}\} = \sum_p [L_p]^T \{F_{int}^{(p)}([L_p] \{c\})\}
                          + [K_C^{conn}] \{c\}

        where `\{F_{int}^{(p)}\}` is :meth:`.Shell.calc_fint` of each panel,
        `[L_p]` selects its degrees of freedom, and `[K_C^{conn}]` is the
        matrix of the connections, :meth:`.get_kC_conn`, the penalty
        connections being linear in ``c``. :meth:`.MultiDomain.calc_kT` is
        its exact Jacobian, see :ref:`tangent_clpt_donnell_multidomain`. For
        a cohesive connection ``'SB_TSL'``, `[K_C^{conn}]` is the secant
        stiffness at ``c``, which depends on ``c`` through the damage, see
        :meth:`.calc_kT_TSL`.

        """
        msg('Calculating internal forces for assembly...', level=2, silent=silent)
        size = self.get_size()
        # fint = 0
        fint = np.zeros_like(c)
        for p in self.panels:
            if p.col_start is None:
                raise ValueError('Shell attributes "col_start" must be defined!')
            fint += p.calc_fint(c=c, size=size, col0=p.col_start, silent=silent,
                                inc=inc)
        if kC_conn is None:
            kC_conn = self.get_kC_conn(c=c)
        fint += kC_conn*c
        self.fint = fint
        msg('finished!', level=2, silent=silent)
        return fint


    def calc_fext(self, inc=1., silent=True, c=None):
        r"""External force vector of the assembly

        With ``c``, the follower pressure loads of the panels are evaluated
        in this configuration, see :meth:`.Shell.calc_fext`.

        With ``pd_method='null-space'``, the penalty forces of the prescribed
        displacements are not included, since they are imposed exactly, see
        :meth:`.get_cp`.

        The vectors :meth:`.Shell.calc_fext` of the panels are assembled at
        their offsets ``col_start``. The result is stored in the attribute
        ``fext``.

        Parameters
        ----------
        inc : float, optional
            Load factor that multiplies the incremented terms of the panels
            (``cte=False``), see :meth:`.Shell.calc_fext`, dimensionless.
            Default ``1.``.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
            Default ``True``.
        c : np.ndarray or None, optional
            Ritz constants of the assembly, 1-D array of size
            :meth:`.get_size`, at which the follower pressure loads are
            evaluated. Default ``None``, the undeformed configuration.

        Returns
        -------
        fext : np.ndarray
            External force vector, 1-D array of size :meth:`.get_size`.

        Raises
        ------
        ValueError
            When the attribute ``col_start`` of a panel is ``None``.

        Notes
        -----
        With ``pd_method='penalty'``, a prescribed displacement `u_p` with
        the penalty stiffness `k` adds the energy `\frac{1}{2} k (u -
        u_p)^2`, integrated along the line for the distributed
        prescriptions, whose first variation gives the force `k u_p` in this
        vector and the penalty stiffness `k`, which the user adds to the
        stiffness matrix. With ``pd_method='null-space'`` the prescribed
        displacements are constraints instead, see :meth:`.get_cp`.

        """
        msg('Calculating external forces for assembly...', level=2, silent=silent)
        size = self.get_size()
        fext = 0
        for p in self.panels:
            if p.col_start is None:
                raise ValueError('Shell attributes "col_start" must be defined!')
            fext += p.calc_fext(inc=inc, size=size, col0=p.col_start,
                                silent=silent, c=c,
                                include_pds=self.pd_method == 'penalty')
        self.fext = fext
        msg('finished!', level=2, silent=silent)
        return fext


    def calc_separation(self, res_pan_top, res_pan_bot):

        r"""
            Calculates the separation between two panels

            The normal separation is the difference of the displacements
            `w` of the first panel of each result, ``res_pan_top['w'][0] -
            res_pan_bot['w'][0]``, which must be evaluated at the same
            points.

            Parameters
            ----------
            res_pan_top, res_pan_bot : dict
                Results of :meth:`.uvw` or :meth:`.calc_results` with
                ``vec='w'``, with the keys ``'x'``, ``'y'`` and ``'w'``.

                ``res_pan_top`` must be the results of the panel on the
                positive side of the interface along `z`, ``p1`` of the
                ``'SB_TSL'`` connection, see :meth:`.get_kC_conn`

            Returns
            -------
            del_d : np.ndarray
                Normal separation at each point, length, with the shape of
                the grid of the results.

            Raises
            ------
            ValueError
                When the coordinates ``'x'`` or ``'y'`` of the two results
                differ.

            Notes
            -----
            The normal separation of the interface of an ``'SB_TSL'``
            connection is `\Delta_w = w^t - w^b`, positive when the
            interface opens, `t` and `b` referring to ``p1`` and ``p2``.
            The tangential separations, which also depend on the slopes of
            both panels, are given by the operators of
            :meth:`._tsl_operators`.
        """

        if not (np.all(res_pan_top['x'][0] == res_pan_bot['x'][0]) and np.all(res_pan_top['y'][0] == res_pan_bot['y'][0])):
            raise ValueError('Grid used to evaluate the displ of both panels dont match')

        # [0] bec its a list of results per panel and since theres only 1 panel, its the first one
        del_d = res_pan_top['w'][0] - res_pan_bot['w'][0]

        return del_d


    def calc_k_dmg(self, c, pA, pB, nr_x_gauss, nr_y_gauss, tsl_type, prev_max_dmg_index,
                   k_i=None, tau_o=None, G1c=None):
        r"""Calculate the damaged k_tsl and the damage index

        At the ``nr_x_gauss`` by ``nr_y_gauss`` Gauss-Legendre points of the
        connection, the normal separation of :meth:`.calc_separation` is
        corrected by :meth:`.correct_separation`, the damage of the current
        state is computed from the corrected separation by
        :func:`.calc_kw_tsl`, the damage is the element-wise maximum of this
        value and ``prev_max_dmg_index``, and the stiffness of the interface
        is ``k_i*(1 - damage)``, see :meth:`.calc_damaged_stiffness`, the
        same for the three components of the traction and also for negative
        separations.

        The first time a point is damaged, the discretisation of ``pA`` along
        `x` is checked against the rules of
        :mod:`panels.multidomain.connections.tsl`, and a warning is printed
        when ``pA.m < 1.0*lam*pA.a`` or ``nr_x_gauss < 7*pA.a/lcz``, with
        ``lam, lcz`` of :func:`.tsl_resolution` for ``pA`` and ``pB``. The check is done once per
        ``pA`` and ``nr_x_gauss``.

        NOTE: Currently this only works for 1 contact region bec of the way
            the damage history is being stored in the MD object. To ensure
            that it works for multiple connected domamins, it needs to be
            stored with the shell object instead and modify the rest
            accordingly

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the assembly, 1-D array of size
            :meth:`.get_size`.
        pA : :class:`.Shell`
            Panel on the positive side of the interface along `z`, ``p1`` of
            the ``'SB_TSL'`` connection.
        pB : :class:`.Shell`
            Panel on the negative side of the interface, ``p2``.
        nr_x_gauss, nr_y_gauss : int
            Number of Gauss-Legendre points of the interface along `x` and
            `y`.
        tsl_type : str
            Type of the traction-separation law, passed to
            :func:`.calc_kw_tsl`, which accepts only ``'bilinear'``.
        prev_max_dmg_index : np.ndarray or None
            Max damage index per integration point for
            the previous converged NR iteration, or ``None``
            With shape ``(nr_y_gauss, nr_x_gauss)``, dimensionless.
        k_i : float
            Pristine stiffness of the interface, `k_o`, force/length**3.
            Required, although the default is ``None``.
        tau_o : float
            Traction at damage onset, force/length**2. Required, although
            the default is ``None``.
        G1c : float
            Critical energy release rate in mode I, force/length. Required,
            although the default is ``None``.

        Returns
        -------
        kw_tsl, max_dmg_index, corrected_del_d, dmg_index_curr : np.ndarray
            The stiffness of the interface, the damage, the corrected
            normal separation and the damage of the corrected separation of
            the current state, all with shape ``(nr_y_gauss, nr_x_gauss)``. The damage
            history ``self.dmg_index`` is not modified, see
            :meth:`.update_TSL_history`.

        Notes
        -----
        The traction-separation law is bilinear, `\tau_\alpha = k_o (1 - d)
        \Delta_\alpha` for `\alpha = u, v, w`, with one damage variable `d`
        [1]_. The damage is driven by the normal separation `\Delta_w = w^t
        - w^b` at the Gauss-Legendre points of the connection, after the
        correction of :meth:`.correct_separation`, `\bar\Delta_w`, and it is
        irreversible:

        .. math::

            d = \max\left( d^{h}, \hat d(\bar\Delta_w) \right), \qquad
            \hat d(\Delta) = \begin{cases}
                0 & \Delta \le \Delta_o \\
                \frac{\Delta_f (\Delta - \Delta_o)}{(\Delta_f - \Delta_o) \Delta}
                  & \Delta_o < \Delta < \Delta_f \\
                1 & \Delta \ge \Delta_f
            \end{cases}

        where `d^h` is ``prev_max_dmg_index``, the damage of the last
        converged increment, `\hat d` is computed by :func:`.calc_kw_tsl`,
        `\Delta_o = \tau_o/k_o` and `\Delta_f = 2 G_{Ic}/\tau_o`, such that
        the area under the softening branch of the law is `G_{Ic}`. This is
        the bilinear law of Camanho et al. [2]_, Eqs. (28) and (29), for mode
        I, with the onset and final separations of its p. 1421, with one
        damage variable driven by the normal separation for the three
        components instead of one per mode. The
        stiffness of the interface is `k^w_{CZ} = k_o (1 - d)`, the same for
        the three components of the traction, see :ref:`cohesive_zone`. It is
        also used for negative separations: there is no additional stiffness
        against interpenetration, the interpenetration stiffness of Eq. 5.12
        of [1]_ is not implemented, which does not affect a monotonically
        loaded double cantilever beam.

        The resolution check warns when the terms or the integration points
        along `x` of the cohesive domain are too few for the decay length
        `1/\lambda` of the opening and for the length `l_{cz}` of the
        cohesive zone, see :func:`.tsl_resolution` and
        :func:`.tsl_discretization`, which give the recommended
        discretisation; below these limits the load-displacement curve after
        the peak may become a sawtooth, and the increments may need
        bisection.

        References
        ----------
        .. [1] D'Souza, N., Multi-Domain Semi-Analytical Cohesive Zone
           Approach, MSc thesis, Delft University of Technology, 2024.
           https://resolver.tudelft.nl/uuid:4709f888-c032-4d9f-97e3-402dfdd47cf1
        .. [2] Camanho, P. P., Dávila, C. G., and de Moura, M. F., Numerical
           simulation of mixed-mode progressive delamination in composite
           materials, Journal of Composite Materials, 37(16), 1415-1438,
           2003. https://doi.org/10.1177/0021998303034505

        """
        # Calculating the displacements of each panel
        res_pan_top = self.calc_results(c=c, eval_panel=pA, vec='w',
                                nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss)
        res_pan_bot = self.calc_results(c=c, eval_panel=pB, vec='w',
                                nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss)
        # Separation for the current NR iteration
        del_d_curr = self.calc_separation(res_pan_top, res_pan_bot)

        corrected_del_d = self.correct_separation(del_d_curr)

        # Calculating dmg index corr to current separation
        _, dmg_index_curr = connections.calc_kw_tsl(pA=pA, pB=pB, tsl_type=tsl_type, k_i=k_i,
                                                    del_d=corrected_del_d, tau_o=tau_o, G1c=G1c)
        if np.any(dmg_index_curr > 0):
            self._check_tsl_resolution(pA, nr_x_gauss, k_i, tau_o, G1c, pB=pB)

        # Considering max dmg_index per intgn point for all displ steps
        # prev_max_dmg_index is already the max over all disp steps at each integration point
        if prev_max_dmg_index is not None:
            max_dmg_index = np.amax(np.array([prev_max_dmg_index, dmg_index_curr]), axis=0)
        else: # First iteration
            max_dmg_index = dmg_index_curr.copy()

        # Calculating stiffness grid
        kw_tsl = self.calc_damaged_stiffness(dmg_index=max_dmg_index, k_i=k_i)

        return kw_tsl, max_dmg_index, corrected_del_d, dmg_index_curr


    def _check_tsl_resolution(self, pA, nr_x_gauss, k_o, tau_o, G1c, pB=None):
        r"""Warns once if the cohesive domain ``pA`` is under-resolved along `x`

        Warns when ``pA.m < 1.0*lam*pA.a`` or ``nr_x_gauss < 7*pA.a/lcz``,
        with ``lam, lcz`` of :func:`.tsl_resolution`, once per ``pA`` and
        ``nr_x_gauss``. See :meth:`.calc_k_dmg` and
        :mod:`panels.multidomain.connections.tsl`.

        Notes
        -----
        ``lam`` is the decay rate `\lambda` of the opening ahead of the
        process zone and ``lcz`` the length `l_{cz}` of the cohesive zone.
        The limits of 1.0 terms per decay length `1/\lambda` and 7 points
        per `l_{cz}` are below the rules `m = 1.25 \lambda a` and `n_x = 10
        a/l_{cz}` of :func:`.tsl_discretization`, which the convergence
        study of the double cantilever beam gave, see :ref:`cohesive_zone`.

        """
        checked = self.__dict__.setdefault('_tsl_resolution_checked', set())
        key = (id(pA), nr_x_gauss)
        if key in checked:
            return
        checked.add(key)
        lam, lcz = connections.tsl_resolution(pA, k_o, tau_o, G1c, p2=pB)
        terms = pA.m/(lam*pA.a)
        points = nr_x_gauss*lcz/pA.a
        if terms < 1.0:
            warn(f'SB_TSL: {pA.m} terms along x over {pA.a:g} give {terms:.2f} '
                 f'terms per decay length 1/lambda = {1/lam:.3g}, below 1.0; '
                 f'recommended m = 1.25*lambda*a = {1.25*lam*pA.a:.1f}, see '
                 'panels.multidomain.connections.tsl_discretization')
        if points < 7:
            warn(f'SB_TSL: {nr_x_gauss} Gauss points along x over {pA.a:g} give '
                 f'{points:.2f} points per cohesive zone length l_cz = {lcz:.3g}, '
                 f'below 7; recommended nx = 10*a/l_cz = {10*pA.a/lcz:.0f}, see '
                 'panels.multidomain.connections.tsl_discretization')


    @staticmethod
    def correct_separation(del_d):
        r"""Remove the separation behind the first compressive point

        Assumes a single crack front that advances along ``-x`` (DCB loaded
        at ``x=a`` of the panel ``p1``). For each row ``y = cte``, starting
        from the point of maximum separation (location `A`) and moving
        towards ``x=0``, the first point with a non-positive separation is
        found (location `B`), and the separation from ``x=0`` up to `B` is
        set to zero. Points on the other side of `A` are not modified. Rows
        without non-positive values between ``x=0`` and `A`, or without
        positive values, are not modified. The corrected separation is used
        only to compute the damage, see :meth:`.calc_k_dmg`.

        Parameters
        ----------
        del_d : np.ndarray
            Separation field with shape ``(nr_y_gauss, nr_x_gauss)``.

        Returns
        -------
        corrected : np.ndarray
            The corrected separation field.

        Notes
        -----
        Implements Section 6.3.1 of the thesis of D'Souza (2024) [1]_:

        .. math::

            \Delta^{corr}_{i,j} = \begin{cases}
                0 & j \le B_i \\
                \Delta_{i,j} & j > B_i
            \end{cases}
            \qquad
            B_i = \max \left\{ j \le A_i : \Delta_{i,j} \le 0 \right\}, \quad
            A_i = \arg\max_j \Delta_{i,j}

        with `i` the row and `j` the column of the grid of integration
        points, `j` increasing with `x`. A previous implementation set to
        zero everything from ``x=0`` up to the *last* non-positive
        separation of the row, which removed the whole fracture process zone
        when the approximation functions gave a non-positive value between
        `A` and the edge of the domain. The negative separations are
        physical: for a beam on an elastic foundation the separation behind
        the crack front oscillates with a decaying amplitude. The tractions,
        the internal force vector and the stiffness matrices use the full
        separation. See :ref:`cohesive_zone_correction`.

        References
        ----------
        .. [1] D'Souza, N., Multi-Domain Semi-Analytical Cohesive Zone
           Approach, MSc thesis, Delft University of Technology, 2024.
           https://resolver.tudelft.nl/uuid:4709f888-c032-4d9f-97e3-402dfdd47cf1

        """
        corrected = np.array(del_d, dtype=DOUBLE, copy=True)
        for i in range(corrected.shape[0]):
            row = corrected[i, :]
            A = np.argmax(row)
            if row[A] <= 0:
                continue
            nonpos = np.flatnonzero(row[:A+1] <= 0)
            if nonpos.size > 0:
                B = nonpos[-1]
                row[:B+1] = 0.
        return corrected


    def _gauss_field_operators(self, panel, nr_x_gauss, nr_y_gauss):
        r"""Matrices giving `u, v, w, w_{,x}, w_{,y}` at the Gauss points

        The field is linear in the Ritz constants of ``panel``, such that
        ``N[k] @ c_panel`` gives, for ``k = 0, 1, 2, 3, 4``, respectively `u`,
        `v`, `w`, `w_{,x}` and `w_{,y}` at the ``nr_y_gauss x nr_x_gauss``
        Gauss-Legendre points of ``panel``, flattened in the same order used
        by :meth:`.uvw` and by the ``kw_tsl`` grids of the ``SB_TSL``
        connection. The matrices are cached because they only depend on the
        geometry and on the approximation functions.

        Returns
        -------
        N : np.ndarray
            Array with shape ``(5, nr_y_gauss*nr_x_gauss, n_dofs_panel)``.
        weights : np.ndarray
            Integration weights, including the Jacobian `ab/4`.

        Notes
        -----
        Since the fields are linear in the Ritz constants, the column `j` of
        each matrix is the field of the unit vector `\{e_j\}`, evaluated by
        the function ``fuvw`` of the model, and the weights are `w_{\xi}
        w_{\eta} \, ab/4`. The slopes come from the rotations of ``fuvw``,
        `w_{,x} = -\phi_x` and `w_{,y} = -\phi_y`, which hold for the models
        based on the classical laminated plate theory with the Donnell
        kinematics. With the Sanders-Koiter kinematics ``fuvw`` gives
        `\phi_y = -w_{,y} + v/r`, the rotation of the normal about `x`, and
        the last matrix gives `w_{,y} - v/r`, such that the operator `[B_v]`
        of :meth:`._tsl_operators` gives the displacement `v(z) = v + z (v/r
        - w_{,y})` of the Kirchhoff-Love kinematics of the cylinder at the
        interface, the one also used for the prescribed displacements at
        `z`, see :mod:`panels.multidomain.connections.nullspace`. A rigid
        rotation about the axis of the cylinder then opens no tangential
        separation when the radius of each panel is that of its reference
        surface. The kernels of ``kCSB_dmg.pyx``, used with the key
        ``'use_kernels'``, and the connection ``'SB'``, see
        :func:`.kCsdt._surface_terms`, contain the same term `v/r`.

        """
        cache = self.__dict__.setdefault('_gauss_field_cache', {})
        key = (id(panel), panel.m, panel.n, panel.a, panel.b, nr_x_gauss,
               nr_y_gauss)
        if key in cache:
            return cache[key]
        xi, wxi = roots_legendre(nr_x_gauss)
        eta, weta = roots_legendre(nr_y_gauss)
        xs, ys = np.meshgrid((panel.a/2)*(xi + 1), (panel.b/2)*(eta + 1))
        xs = np.ascontiguousarray(xs.ravel(), dtype=DOUBLE)
        ys = np.ascontiguousarray(ys.ravel(), dtype=DOUBLE)
        weights = np.outer(weta, wxi).ravel()*panel.a*panel.b/4
        fuvw = modelDB.db[panel.model]['field'].fuvw
        ndof = panel.col_end - panel.col_start
        N = np.zeros((5, xs.shape[0], ndof), dtype=DOUBLE)
        e = np.zeros(ndof, dtype=DOUBLE)
        for j in range(ndof):
            e[:] = 0.
            e[j] = 1.
            out = fuvw(e, panel, xs, ys, self.out_num_cores)
            for k in range(5):
                N[k, :, j] = out[k]
        #NOTE fuvw returns phix = -w,x and phiy = -w,y
        N[3:] *= -1
        cache[key] = (N, weights)
        return N, weights


    def _tsl_operators(self, p_top, p_bot, nr_x_gauss, nr_y_gauss):
        r"""Separation operators of an ``SB_TSL`` connection

        Returns ``(Bu, Bv, Bw, weights, dofs)``, where ``Bu @ cc``, ``Bv @
        cc`` and ``Bw @ cc`` give the separations along `x`, `y` and `z`
        at the Gauss-Legendre points of the connection, with ``cc =
        c[dofs]`` the Ritz constants of ``p_top``, the panel ``p1`` of the
        connection, on the positive side of the interface along `z`, followed
        by those of ``p_bot``, the panel ``p2``, on the negative side, see
        :meth:`.get_kC_conn`. They are built from the matrices of
        :meth:`._gauss_field_operators` of both panels and from the
        distances of their reference surfaces to the interface, see
        :func:`.kCsdt._sb_distances`; they depend only on the geometry and
        are cached.

        Notes
        -----
        The separations are the displacement jump between the two surfaces in
        contact, with the slope of each panel:

        .. math::

            \begin{aligned}
                \Delta_u &= u^t + d^t w^t_{,x} - u^b + d^b w^b_{,x} \\
                \Delta_v &= v^t + d^t w^t_{,y} - v^b + d^b w^b_{,y} \\
                \Delta_w &= w^t - w^b
            \end{aligned}

        where the superscripts `t` and `b` refer to ``p1`` and ``p2``, `u`,
        `v` are the displacements of their reference surfaces, and `d^t =
        h_1/2 - d_1` and `d^b = h_2/2 + d_2` are the distances from their
        reference surfaces to the interface, the lower face `z = d_1 -
        h_1/2` of ``p1`` and the upper face `z = d_2 + h_2/2` of ``p2``,
        with the offsets `d_1`, `d_2`, from the Kirchhoff kinematics `u(z) =
        u - z w_{,x}`. Without offsets they are `h_1/2` and `h_2/2`, the
        distances from the mid-planes. With the matrices `[N]` of
        :meth:`._gauss_field_operators`, whose row `g` holds the
        approximation functions of each field at the point `g`, the operators
        are the `n_g \times n_{dof}` matrices:

        .. math::

            \begin{aligned}
                {}[B_u] &= \left[ [N^t_u] + d^t [N^t_{w,x}] \;\;
                           -[N^b_u] + d^b [N^b_{w,x}] \right] \\
                [B_v] &= \left[ [N^t_v] + d^t [N^t_{w,y}] \;\;
                         -[N^b_v] + d^b [N^b_{w,y}] \right] \\
                [B_w] &= \left[ [N^t_w] \;\; -[N^b_w] \right]
            \end{aligned}

        Unlike the compatibility of the thesis of D'Souza (2024)
        [1]_, Eq. 4.62, which uses the slope of ``p1``
        only, this gives no spurious tangential separation in the fracture
        process zone, where the two arms rotate in opposite directions. See
        :ref:`cohesive_zone`. With `z_1 = -d^t` and `z_2 = d^b`, the rows of
        the operators are those of the quantities of ``'SB'``, `u_1(z_1) -
        u_2(z_2)`, `v_1(z_1) - v_2(z_2)` and `w_1 - w_2`, see
        :func:`.kCsdt._sb_components`, at the Gauss-Legendre points: a
        pristine interface, with `k_o` at every point, has the matrix of
        :func:`.skin_base.fkCSB` with `k_t = k_o` when the rule integrates
        the products of the approximation functions exactly.

        References
        ----------
        .. [1] D'Souza, N., Multi-Domain Semi-Analytical Cohesive Zone
           Approach, MSc thesis, Delft University of Technology, 2024.
           https://resolver.tudelft.nl/uuid:4709f888-c032-4d9f-97e3-402dfdd47cf1

        """
        cache = self.__dict__.setdefault('_tsl_operators_cache', {})
        key = (id(p_top), id(p_bot), p_top.col_start, p_bot.col_start,
               nr_x_gauss, nr_y_gauss)
        if key in cache:
            return cache[key]
        Nt, weights = self._gauss_field_operators(p_top, nr_x_gauss, nr_y_gauss)
        Nb, _ = self._gauss_field_operators(p_bot, nr_x_gauss, nr_y_gauss)
        dt, db = connections.kCsdt._sb_distances(p_top, p_bot)
        Bu = np.hstack((Nt[0] + dt*Nt[3], -Nb[0] + db*Nb[3]))
        Bv = np.hstack((Nt[1] + dt*Nt[4], -Nb[1] + db*Nb[4]))
        Bw = np.hstack((Nt[2], -Nb[2]))
        dofs = np.concatenate((np.arange(p_top.col_start, p_top.col_end),
                               np.arange(p_bot.col_start, p_bot.col_end)))
        cache[key] = (Bu, Bv, Bw, weights, dofs)
        return cache[key]


    def _kC_TSL(self, p_top, p_bot, nr_x_gauss, nr_y_gauss, k_o, kw_tsl, size):
        r"""Secant stiffness of an ``SB_TSL`` connection by matrix products

        Computes the same matrix as :func:`.fkCSB11_dmg`,
        :func:`.fkCSB12_dmg` and :func:`.fkCSB22_dmg`, with the same
        Gauss-Legendre rule, from the operators of :meth:`._tsl_operators`
        and the stiffness ``kw_tsl`` at the integration points: the
        stiffness of the pristine interface, with ``k_o`` at every point,
        computed once and cached, minus a correction computed only over the
        damaged points, where ``kw_tsl`` differs from ``k_o``. Only the
        upper triangle is returned, see :func:`.finalize_symmetric_matrix`.
        The kernels are still used when the connection has
        ``use_kernels=True``.

        Notes
        -----
        The matrix is

        .. math::

            [K] = [K_o] - \sum_{\alpha=u,v,w} [B_\alpha]^T
                  \text{diag}\left(w_g (k_o - k_g)\right) [B_\alpha]

        where `[K_o] = k_o \sum_\alpha [B_\alpha]^T \text{diag}(w_g)
        [B_\alpha]` is the stiffness of the pristine interface, `w_g` are
        the integration weights and `k_g` the entries of ``kw_tsl``. The
        second term runs only over the points where `k_g \ne k_o`, the
        damaged points.

        The expression above is the Gauss-Legendre rule

        .. math::

            [K] = \sum_{\alpha=u,v,w} [B_\alpha]^T
                  \text{diag}\left(w_g k_g\right) [B_\alpha]

        regrouped with `k_g = k_o - (k_o - k_g)`, therefore it is exact, not
        an approximation, and it does not depend on how the damage is
        computed. Before damage onset the assembly reduces to a
        copy of `[K_o]`, after onset the cost of the correction grows with
        the number of points of the fracture process zone and of the cracked
        region, not with the size of the whole interface. The operators
        `[B_\alpha]` are those of :meth:`._tsl_operators`. The result agrees
        with the kernels of ``kCSB_dmg.pyx`` to a relative `10^{-12}`, see
        ``tests/multidomain/test_sb_tsl.py``. See :ref:`cohesive_zone`.

        """
        Bu, Bv, Bw, weights, dofs = self._tsl_operators(p_top, p_bot,
                nr_x_gauss, nr_y_gauss)
        cache = self.__dict__.setdefault('_tsl_K0_cache', {})
        key = (id(p_top), id(p_bot), p_top.col_start, p_bot.col_start,
               nr_x_gauss, nr_y_gauss, k_o)
        if key not in cache:
            WB = [(k_o*weights)[:, None]*B for B in (Bu, Bv, Bw)]
            K0 = Bu.T @ WB[0] + Bv.T @ WB[1] + Bw.T @ WB[2]
            upper = dofs[None, :] >= dofs[:, None]
            i, j = np.nonzero(upper)
            cache[key] = (K0, dofs[i], dofs[j], upper)
        K0, rows, cols, upper = cache[key]
        dk = k_o - np.ravel(kw_tsl)
        dmg = np.flatnonzero(dk)
        if dmg.size > 0:
            g = (weights[dmg]*dk[dmg])[:, None]
            K = K0.copy()
            for B in (Bu, Bv, Bw):
                Bd = B[dmg]
                K -= Bd.T @ (g*Bd)
        else:
            K = K0
        return coo_matrix((K[upper], (rows, cols)), shape=(size, size))


    def calc_kT_TSL(self, c, conn=None):
        r"""Damage-rate part of the tangent stiffness of the cohesive zone

        Returns the matrix to be added to the secant stiffness of the
        ``SB_TSL`` connections of :meth:`.get_kC_conn` to obtain the
        consistent tangent stiffness of the cohesive zone at ``c``, with the
        damage history ``self.dmg_index`` when it exists. It is computed
        with the operators of :meth:`._tsl_operators` at the Gauss-Legendre
        points of each connection where the damage grows in the current
        state, i.e. where the damage of the current separation, see
        :meth:`.calc_k_dmg`, is larger than the damage history and the
        corrected normal separation is between the separations at damage
        onset and at complete failure. The matrix is not symmetric. Only
        connections with ``tsl_type='bilinear'`` contribute.

        Parameters
        ----------
        c : array-like
            Ritz constants of the assembly.
        conn : list of dicts, optional
            Connections, the default is ``self.conn``.

        Returns
        -------
        kD : csr_matrix
            Sparse matrix with the size of the assembly, zero when no point
            of the cohesive zone is softening.

        Notes
        -----
        The internal force of an ``SB_TSL`` connection is `f = K_s(c) c`,
        where the secant stiffness `K_s` is the matrix of :meth:`.get_kC_conn`,
        computed with `k = k_o (1 - d)`. Its consistent tangent is `K_s +
        K_d`, where:

        .. math::

            K_d = - \int_A k_o \frac{\partial d}{\partial \Delta_n}
                    \left( B_u^T \Delta_u + B_v^T \Delta_v + B_w^T \Delta_n
                    \right) B_w \, dA

        with `B_u, B_v, B_w` the operators giving the separations `\Delta_u,
        \Delta_v, \Delta_n` from the Ritz constants, see
        :meth:`._tsl_operators`. `\partial d/\partial
        \Delta_n` is only non-zero at the points where the damage grows in
        the current state, i.e. where the damage computed from the current
        separation is larger than the damage history and `\Delta_o < \Delta_n <
        \Delta_f`. The matrix is not symmetric. See :ref:`cohesive_zone`.

        For the bilinear law of :meth:`.calc_k_dmg`, `\tau_\alpha = k_o (1 -
        d) \Delta_\alpha` for `\alpha = u, v, w`, with the damage `d =
        \max(d^h, \hat d(\bar\Delta_n))` driven by the corrected normal
        separation `\bar\Delta_n`:

        .. math::

            \frac{\partial d}{\partial \Delta_n} = \chi \, \hat d'(\Delta_n),
            \qquad
            \hat d'(\Delta) = \frac{\Delta_f \Delta_o}{(\Delta_f - \Delta_o)
            \Delta^2}, \qquad
            \chi = \begin{cases}
                1 & \Delta_o < \bar\Delta_n < \Delta_f \text{ and }
                    \hat d(\bar\Delta_n) > d^h \\
                0 & \text{otherwise}
            \end{cases}

        Where `\chi = 1` the correction of :meth:`.correct_separation` leaves
        the separation unchanged, `\bar\Delta_n = \Delta_n`, and the
        derivative of the correction itself is zero almost everywhere. The
        tangential separations use the uncorrected values, the same used in
        the internal force vector. Some properties:

        - `K_d` is not symmetric, because the tangential tractions depend on
          `\Delta_n` through `d`, while `d` does not depend on `\Delta_u`,
          `\Delta_v`.
        - The term `B_w^T \Delta_n B_w` is symmetric and negative
          semi-definite; it makes the normal tangent stiffness of a softening
          point `k_o (1 - d - \hat d' \Delta_n) = -k_o \Delta_o / (\Delta_f -
          \Delta_o)`, the slope of the descending branch of the law.
        - With the secant stiffness alone the Newton-Raphson iterations
          converge linearly once the crack grows. The iteration matrix
          `[K_C] + [K_G] + K_s + K_d + [K^{pen}_{PD}]` is non-symmetric,
          ``structsolve.solve`` accepts it.

        The matrix was verified against central finite differences of `K_s(c)
        c`, for random states in the softening range, with and without points
        where `d^h > \hat d`: the relative error of `K_s + K_d` is below `2
        \times 10^{-9}`, against 9--62% for `K_s` alone, see
        ``tests/multidomain/test_sb_tsl.py``.

        """
        size = self.get_size()
        if conn is None:
            conn = self.conn
        rows = []
        cols = []
        vals = []
        for connecti in conn:
            if connecti['func'] != 'SB_TSL':
                continue
            if connecti['tsl_type'] != 'bilinear':
                continue
            k_o = connecti['k_o']
            tau_o = connecti['tau_o']
            G1c = connecti['G1c']
            nr_x_gauss = connecti['nr_x_gauss']
            nr_y_gauss = connecti['nr_y_gauss']
            p_top, p_bot = _sb_top_bottom(connecti)
            prev = self.dmg_index if hasattr(self, "dmg_index") else None
            _, _, del_n, dmg_index_curr = self.calc_k_dmg(c=c, pA=p_top,
                    pB=p_bot, nr_x_gauss=nr_x_gauss, nr_y_gauss=nr_y_gauss,
                    tsl_type='bilinear', prev_max_dmg_index=prev, k_i=k_o,
                    tau_o=tau_o, G1c=G1c)
            del_n = np.ravel(del_n)
            dmg_index_curr = np.ravel(dmg_index_curr)
            if prev is None:
                prev = np.zeros_like(dmg_index_curr)
            prev = np.ravel(prev)
            del_o = tau_o/k_o
            del_f = 2*G1c/tau_o
            active = ((del_n > del_o) & (del_n < del_f)
                      & (dmg_index_curr > prev))
            if not np.any(active):
                continue
            # derivative of d = del_f (del_n - del_o)/((del_f - del_o) del_n)
            dd = del_f*del_o/((del_f - del_o)*del_n[active]**2)

            Bu, Bv, Bw, weights, dofs = self._tsl_operators(p_top, p_bot,
                    nr_x_gauss, nr_y_gauss)
            Bu = Bu[active]
            Bv = Bv[active]
            Bw = Bw[active]
            cc = c[dofs]
            g = k_o*dd*weights[active]
            #NOTE the separations are the uncorrected ones, the same used in
            #     the internal force vector
            Du = Bu @ cc
            Dv = Bv @ cc
            Dw = Bw @ cc
            kD = -(Bu.T @ ((g*Du)[:, None]*Bw)
                   + Bv.T @ ((g*Dv)[:, None]*Bw)
                   + Bw.T @ ((g*Dw)[:, None]*Bw))
            rows.append(np.repeat(dofs, dofs.size))
            cols.append(np.tile(dofs, dofs.size))
            vals.append(kD.ravel())
        if len(vals) == 0:
            return csr_matrix((size, size), dtype=DOUBLE)
        return csr_matrix((np.concatenate(vals), (np.concatenate(rows),
                           np.concatenate(cols))), shape=(size, size))


    def reaction_line_pd_xcte(self, c, panel, x, kw, funcw, nr_y_gauss=None):
        r"""Out-of-plane reaction of a prescribed displacement along ``x=cte``

        For a prescribed displacement ``funcw`` imposed with the penalty
        stiffness ``kw`` along the line ``x`` of ``panel``, see
        :meth:`.Shell.add_distr_pd_fixed_x` and :func:`.fkCld_xcte`, returns
        the force along `z` applied by the support on the structure: the
        product of ``kw`` by the difference between the prescribed and the
        computed displacements `w` along the line, integrated along `y` with
        ``nr_y_gauss`` Gauss-Legendre points. It is used for the
        load-displacement curves of the cohesive zone model.

        Parameters
        ----------
        c : array-like
            Ritz constants of the assembly.
        panel : :class:`.Shell`
            Panel where the displacement is prescribed.
        x : float
            Position of the line, in the coordinates of ``panel``.
        kw : float
            Penalty stiffness used to prescribe the displacement.
        funcw : callable
            Prescribed displacement `w_p(y)`.
        nr_y_gauss : int, optional
            Number of Gauss-Legendre points along `y`. The default is
            ``max(2*panel.n, 20)``.

        Returns
        -------
        R : float
            Reaction force along `z`.

        Notes
        -----
        The force applied by the support on the structure is:

        .. math::

            R = \int_0^b k_w \left( w_p(y) - w(x, y) \right) dy

        the force of the penalty spring between the structure and the
        support, which is the load measured by the load cell. It is exact for
        any state of damage of the structure, and it is used for the
        load-displacement curves of the cohesive zone model, see
        :ref:`cohesive_zone_load`.

        """
        if nr_y_gauss is None:
            nr_y_gauss = max(2*panel.n, 20)
        eta, weights = roots_legendre(nr_y_gauss)
        y = np.ascontiguousarray((panel.b/2)*(eta + 1), dtype=DOUBLE)
        xs = np.ascontiguousarray(np.full_like(y, x), dtype=DOUBLE)
        c_panel = np.ascontiguousarray(c[panel.col_start: panel.col_end], dtype=DOUBLE)
        fuvw = modelDB.db[panel.model]['field'].fuvw
        _, _, w, _, _ = fuvw(c_panel, panel, xs, y, self.out_num_cores)
        wp = np.array([funcw(yi) for yi in y])
        return kw*np.dot(weights, wp - np.asarray(w))*panel.b/2


    def calc_traction_stiffness(self, kw_tsl, corrected_max_del_d):
        r"""
            Calculates traction for the approach where SB_TSL connection is used

            The traction is the element-wise product ``kw_tsl *
            corrected_max_del_d``. Nothing is stored.

            Parameters
            ----------
            kw_tsl : np.ndarray
                Stiffness of the interface at each point, force/length**3,
                e.g. from :meth:`.calc_k_dmg`.
            corrected_max_del_d : np.ndarray
                Separation at each point, length, with the shape of
                ``kw_tsl``.

            Returns
            -------
            tau : np.ndarray
                Traction at each point, force/length**2, with the shape of
                the inputs.

            Notes
            -----
            The traction-separation law of :meth:`.calc_k_dmg`, `\tau =
            k^w_{CZ} \Delta` with `k^w_{CZ} = k_o (1 - d)`. The traction that
            enters the internal force vector uses the full separation, not
            the corrected one of :meth:`.correct_separation`, see
            :meth:`.force_out_plane_damage`.
        """
        tau = np.multiply(kw_tsl, corrected_max_del_d)

        return tau


    def update_TSL_history(self, curr_max_dmg_index):
        r"""Store the damage history of the cohesive zone

        Sets ``self.dmg_index``, the damage history used by
        :meth:`.calc_k_dmg`, :meth:`.get_kC_conn`, :meth:`.calc_kT_TSL` and
        :meth:`.force_out_plane_damage`, which take the element-wise maximum
        of this history and of the damage of the current separation. The
        drivers call it only with converged states, typically with the
        damage returned by :meth:`.calc_k_dmg`, so the iterates of an
        increment that is discarded and bisected, or the predicted state at
        the start of an increment, do not modify the history. It is stored
        once per assembly, which restricts the model to one ``'SB_TSL'``
        connection.

        Parameters
        ----------
        curr_max_dmg_index : np.ndarray
            Damage at the Gauss-Legendre points of the connection, with
            shape ``(nr_y_gauss, nr_x_gauss)``.

        Notes
        -----
        The damage is irreversible, `d = \max(d^h, \hat d)`, see
        :meth:`.calc_k_dmg`, such that the existing damage does not vanish
        when the separation decreases. Discarding the iterates of an
        increment is consistent because `d^h` is only updated with converged
        states, and every converged sub-increment of a bisected increment
        updates it, so that the irreversible law follows the finer loading
        path. See :ref:`cohesive_zone`.

        """
        self.dmg_index = curr_max_dmg_index


    def calc_damaged_stiffness(self, dmg_index, k_i):
        """Calculate the reduced stiffness given the damage index

        Returns ``k_i*(1 - dmg_index)``. Nothing is stored.

        Parameters
        ----------
        dmg_index : np.ndarray
            Damage index at each point, dimensionless, between ``0`` and
            ``1``.
        k_i : float
            Pristine stiffness of the interface, `k_o`, force/length**3.

        Returns
        -------
        kw_tsl : np.ndarray
            Reduced stiffness at each point, force/length**3, with the shape
            of ``dmg_index``.

        Notes
        -----
        The secant stiffness of the bilinear traction-separation law,
        `k^w_{CZ} = k_o (1 - d)`, the same for the three components of the
        traction, see :meth:`.calc_k_dmg`.

        """
        kw_tsl = k_i*(1-dmg_index)

        return kw_tsl
