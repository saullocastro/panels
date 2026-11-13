r"""
Base-flange connection ``'BFxcte'`` at a face of the base

With the key ``'face'`` of the connection dictionary, ``'lower'`` or
``'upper'``, the connection ``'BFxcte'`` ties the edge `x_2 =` ``xcte2`` of
the flange ``p2`` to the face of the base ``p1`` at `z_1 = d_1 \mp h_1/2`,
along the line `x_1 =` ``xcte1``, instead of its reference surface `z_1 =
0`. ``'lower'`` and ``'upper'`` refer to the coordinate `z_1` of the base
``p1``, as the faces of the connection ``'SB'`` refer to the coordinate `z`
of the connected domains. The face is a line of points of the normals of
``p1``, which follow the displacements and the rotation of its reference
surface, see :func:`.kCsdt._surface_terms`. For a ``'cylshell_*'`` base,
whose `z` points outwards, ``'lower'`` is the inner surface and
``'upper'`` the outer surface, and the points of the face follow the
Sanders-Koiter kinematics, with the term `z v_1/r_1`, whatever the model
of the cylinder; the key ``'kinematics'``, ``'sanders'`` (default) or
``'model'``, selects the kinematics of the model of the cylinder instead,
see :func:`.face_rinv`. For other bases the points of the face follow the
kinematics of the model of ``p1``. E.g. the web of an
internal ring frame, an ``'annular_*'`` plate with its outer edge
``xcte2=0``, is connected to the inner surface, and the web of an
external ring frame, with its inner edge ``xcte2=p2.a``, to the outer
surface::

    dict(p1=cylinder, p2=web, func='BFxcte', xcte1=x_f, xcte2=0,
         face='lower')
    dict(p1=cylinder, p2=web, func='BFxcte', xcte1=x_f, xcte2=web.a,
         face='upper')

Without the key, or with ``face=None``, the connection ties the reference
surfaces, see :mod:`panels.multidomain.connections.kCBFxcte`, and the key
``'kinematics'`` must not be given (``ValueError``). The connection
``'BFycte'`` has the same keys, see
:mod:`panels.multidomain.connections.kCBFycte_face`.
:func:`.check_bfxcte` checks the geometry of the connection,
:func:`.bfxcte_face_components` gives the connected quantities and
:func:`.fkCBFxcte_face` the penalty stiffness matrix; the null-space method
imposes the same quantities, see
:func:`.nullspace.constraint_matrix`.

Notes
-----
The local axes of the flange are those of the base rotated by 90 degrees
about the common axis `y`, `\boldsymbol{e}_{x2} = -\boldsymbol{e}_{z1}`,
`\boldsymbol{e}_{y2} = \boldsymbol{e}_{y1}` and `\boldsymbol{e}_{z2} =
\boldsymbol{e}_{x1}`, see :mod:`panels.multidomain.connections.kCBFxcte`.
The flange therefore lies on the side of negative `z_1` of the line
connected to its edge `x_2 = 0`, and on the side of positive `z_1` of the
line connected to its edge `x_2 = a_2`: a flange attached to a face, and
not crossing the base, has ``xcte2 = 0`` at the lower face and ``xcte2 =
p2.a`` at the upper face.

At the face `z_1 = z_f`, with `z_f = d_1 - h_1/2` or `d_1 + h_1/2`, the
penalty energy is

.. math::

    U = \frac{1}{2} \int_0^{b_1} k_t \left[ \left(u_1(z_f) - w_2\right)^2
        + \left(v_1(z_f) - v_2\right)^2 + \left(w_1 + u_2\right)^2 \right]
        + k_r \left(\theta_{x1}(z_f) - \theta_{x2}\right)^2 dy

with the Kirchhoff-Love kinematics of the classical laminated plate theory
(CLPT), the normal being rigid and normal to the deformed reference
surface,

.. math::

    u_1(z) = u_1 - z w_{1,x}, \qquad v_1(z) = \left(1 + \frac{z}{r_1}\right)
    v_1 - z w_{1,y}, \qquad \theta_{x1}(z) = -w_{1,x}

the term `z v_1/r_1` being that of the Sanders-Koiter kinematics, whose
rotation of the normal about `x` is `-w_{1,y} + v_1/r_1`, see
:func:`.kCsdt._rinv_sanders`, used for every ``'cylshell_*'`` base by
default and, with ``kinematics='model'``, only for the models with these
kinematics, see :func:`.face_rinv`. For the first-order (FSDT) and third-order
(TSDT) shear deformation theories, `u_1(z)` and `v_1(z)` are those of
:func:`.kCsdt._surface_terms` and `\theta_{x1}(z) = \partial
u_1/\partial z` at `z = z_f`, see :func:`._rotation_x_at`. The flange is
connected at its reference surface, `\theta_{x2}` being the rotation of
:func:`.kCsdt._rotation_x`. A rigid-body motion of the assembly gives
`U = 0` when the edge of the flange is at the face, i.e., for a
``'cylshell_*'`` base and an ``'annular_*'`` flange, when the radius of
the edge, `r_2 - x_{cte2}`, is the radius of the face, `r_1 + z_f`: the
displacement of a point at the distance `z` along the normal of a rigid
motion `\boldsymbol{t} + \boldsymbol{\omega} \times \boldsymbol{X}` is
that of the Kirchhoff-Love kinematics with the Sanders-Koiter rotation
`v_1/r_1`. The rigid motions of a cylinder have the same fields `u_1,
v_1, w_1` at its reference surface for every model, and the rotation
about the axis, `v_1 = r_1 \Omega`, is free of strain also with the Donnell
kinematics, which neglect `z v_1/r_1` at the face: with
``kinematics='model'`` a Donnell cylinder gives `v_1(z_f) - v_2 = -z_f
\Omega`, of the order of `h_1/r_1` relative to `v_2`, within the
approximation of these kinematics, and :func:`.check_bfxcte` issues a
``UserWarning``. The term `z v_1/r_1` at the face does not change the
strains of the cylinder, which remain those of its model.

Each panel keeps its own natural coordinate `\eta = 2 y/b - 1` along the
connection, so that the two edges correspond point by point when they
span the same angle, `b_1/r_1 = b_2/r_2` for a cylinder and an annular
plate, also when their lengths differ, and the integral along the
connection is `\int_0^{b_1} dy = (b_1/2) \int_{-1}^{1} d\eta`, with the
derivatives `\partial/\partial y = (2/b) \partial/\partial \eta` of each
panel. The penalty stiffness matrix is computed with the closed-form
integrals of the Legendre-based (Bardell) approximation functions [1]_ of
:func:`.kCsdt._integral`.

References
----------
.. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
   models to Address Linear Buckling and Vibration of Stiffened Composite
   Panels with Debonding Defect, Composite Structures, 160, 232-247, 2017.
   https://doi.org/10.1016/j.compstruct.2016.10.026

"""
import warnings

import numpy as np
from scipy.sparse import coo_matrix, triu

from panels import INT, DOUBLE
from . import kCsdt


#: Key of the connection dictionary that selects the face of the base
FACE_KEY = 'face'
#: Faces of the base and the sign of `h/2` in `z = d \pm h/2`
FACES = {'lower': -1, 'upper': 1}
#: Key of the connection dictionary that selects the kinematics of the face
#: of a ``'cylshell_*'`` base, see :func:`.face_rinv`
KINEMATICS_KEY = 'kinematics'
#: Kinematics of the face of a ``'cylshell_*'`` base, the first the default
KINEMATICS = ('sanders', 'model')
#: Section of the documentation named by the warning of :func:`.check_bfxcte`
DONNELL_SECTION = ("the section 'Rigid rotation at a face of a Donnell "
        "cylinder' of the page 'Multidomain connections' of the "
        "documentation, https://saullocastro.github.io/panels/"
        "multidomain_connections.html"
        "#rigid-rotation-at-a-face-of-a-donnell-cylinder")
#: Prefixes of the names of the models of the cylinders and of the annular
#: plates, whose radii are checked by :func:`.check_bfxcte`
CYLINDER_PREFIX = 'cylshell_'
ANNULAR_PREFIX = 'annular_'

U, V, W, PHIX, PHIY = range(5)


def _has_prefix(p, prefix):
    return p.model is not None and p.model.startswith(prefix)


def face_coordinate(p, face, func='BFxcte'):
    r"""
    Coordinate `z` of a face of the base

    The coordinate is measured from the reference surface of ``p``, along
    its axis `z`, with its offset and its ply thicknesses.

    Parameters
    ----------
    p : :class:`.Shell`
        Base, with its ``offset`` and ply thicknesses ``plyts``.
    face : str or None
        ``'lower'``, ``'upper'`` or ``None`` for the reference surface.
    func : str, optional
        Connection named in the message of the error, ``'BFxcte'`` (default)
        or ``'BFycte'``.

    Returns
    -------
    z : float
        `d - h/2` for ``'lower'``, `d + h/2` for ``'upper'``, with the offset
        `d` and the thickness `h` of ``p``, and ``0.`` for ``None``, length.

    Raises
    ------
    ValueError
        When ``face`` is not one of ``None``, ``'lower'`` and ``'upper'``.

    """
    if face is None:
        return 0.
    if face not in FACES:
        raise ValueError("The key '{0}' of a '{1}' connection must be one "
                         "of {2} or None, got {3!r}".format(FACE_KEY, func,
                         tuple(FACES), face))
    if p.plyts is None:
        #NOTE the ply thicknesses are derived from plyt and stack
        p._rebuild()
    return kCsdt._face(p, FACES[face])


def kinematics_of(connecti, func='BFxcte'):
    r"""
    Kinematics of the face of the base of a connection ``'BFxcte'`` or
    ``'BFycte'``

    The value of the key ``'kinematics'`` of the connection dictionary,
    checked, with its default. The key is accepted only with the key
    ``'face'``: at the reference surface the kinematics of the face do not
    enter the connection.

    Parameters
    ----------
    connecti : dict
        Connection ``'BFxcte'`` or ``'BFycte'``, optionally with the keys
        ``'face'`` and ``'kinematics'``.
    func : str, optional
        Connection named in the messages of the errors, ``'BFxcte'``
        (default) or ``'BFycte'``.

    Returns
    -------
    kinematics : str
        ``connecti['kinematics']``, ``'sanders'`` when the key is absent or
        ``None``, see :func:`.face_rinv`.

    Raises
    ------
    ValueError
        When ``connecti['kinematics']`` is not one of ``None``,
        ``'sanders'`` and ``'model'``, or when it is not ``None`` and
        ``connecti['face']`` is absent or ``None``.

    """
    kinematics = connecti.get(KINEMATICS_KEY)
    if kinematics is None:
        return KINEMATICS[0]
    if connecti.get(FACE_KEY) is None:
        raise ValueError("The key '{0}' of a '{1}' connection requires the "
                         "key '{2}': without it the reference surfaces are "
                         "connected, got {0}={3!r}".format(KINEMATICS_KEY,
                         func, FACE_KEY, kinematics))
    if kinematics not in KINEMATICS:
        raise ValueError("The key '{0}' of a '{1}' connection must be one "
                         "of {2} or None, got {3!r}".format(KINEMATICS_KEY,
                         func, KINEMATICS, kinematics))
    return kinematics


def face_rinv(p, kinematics='sanders'):
    r"""
    Factor `1/r` of the term `z v/r` of `v(z)` at a face of the base

    The Sanders-Koiter kinematics of a ``'cylshell_*'`` base, whatever its
    model, by default, or the kinematics of the model of the base.

    Parameters
    ----------
    p : :class:`.Shell`
        Base.
    kinematics : str, optional
        ``'sanders'``, the default, for the Sanders-Koiter kinematics at
        the face of a ``'cylshell_*'`` base, whatever its model; ``'model'``
        for the kinematics of the model of ``p``.

    Returns
    -------
    rinv : float
        ``1/p.r`` for a ``'cylshell_*'`` base with ``'sanders'``, otherwise
        that of the kinematics of the model of ``p``,
        :func:`.kCsdt._rinv_sanders`: ``1/p.r`` for the models with the
        Sanders-Koiter rotation of the normal and ``0.`` for the Donnell
        models and the plates, 1/length.

    Raises
    ------
    ValueError
        When ``kinematics`` is not ``'sanders'`` or ``'model'``.

    Notes
    -----
    The displacement of a point at the distance `z` along the normal of a
    cylinder is `v(z) = (1 + z/r) v - z w_{,y}` with the Sanders-Koiter
    rotation of the normal about `x`, `-w_{,y} + v/r`; the Donnell
    kinematics neglect `z v/r`. A rigid rotation `\Omega` about the axis,
    `v = r \Omega`, `u = w = 0`, is free of strain with both kinematics,
    and moves the face `z_f` by `(r + z_f) \Omega`, which `v(z_f)` gives
    only with the term `z v/r`. Using it at the face of every cylinder
    model keeps the rigid rotation of the assembly free of penalty energy,
    or of constraint residual, without changing the strains of the
    cylinder, see the Notes of
    :mod:`panels.multidomain.connections.kCBFxcte_face`.

    """
    if kinematics not in KINEMATICS:
        raise ValueError("kinematics must be one of {0}, got {1!r}".format(
                         KINEMATICS, kinematics))
    if kinematics == 'sanders' and _has_prefix(p, CYLINDER_PREFIX):
        return 1./p.r
    return kCsdt._rinv_sanders(p)


def check_bfxcte(connecti, rtol=1.e-8):
    r"""
    Check the geometry of a connection ``'BFxcte'``

    With the key ``'face'``, the edge ``xcte2`` of the flange ``p2`` must
    be ``0`` at the ``'lower'`` face and ``p2.a`` at the ``'upper'`` face of
    the base ``p1``, the flange lying on the side of the face, and the
    lengths of the connected lines must be equal: `b_1 (1 + z_f/r_1)` of
    the face of a ``'cylshell_*'`` base, `b_1` otherwise, and `b_2 (r_2 -
    x_{cte2})/r_2` of the edge of an ``'annular_*'`` flange, `b_2`
    otherwise. With or without the key, an ``'annular_*'`` flange on a
    ``'cylshell_*'`` base must have its edge `x_2 =` ``xcte2`` at the
    radius of the connected surface of the cylinder, `r_2 - x_{cte2} = r_1
    + z_f`, and span the same angle, `b_2/r_2 = b_1/r_1`, `z_f = 0` being
    the reference surface.

    Parameters
    ----------
    connecti : dict
        Connection ``'BFxcte'``, with the base ``p1``, the flange ``p2``,
        ``xcte1``, ``xcte2`` and optionally ``'face'`` and
        ``'kinematics'``, see :meth:`.MultiDomain.get_kC_conn`.
    rtol : float, optional
        Relative tolerance of the comparisons of lengths, radii and angles.

    Returns
    -------
    z : float
        Coordinate `z_f` of the connected line in ``p1``, measured from its
        reference surface, see :func:`.face_coordinate`, length.

    Raises
    ------
    ValueError
        When ``'face'`` is not ``None``, ``'lower'`` or ``'upper'``, when
        ``xcte2`` is not the edge of the flange on the side of the face, or
        when the radius, the angle or the length of the connected lines do
        not match, see above; when ``'kinematics'`` is not ``None``,
        ``'sanders'`` or ``'model'``, or is given without ``'face'``, see
        :func:`.kinematics_of`.

    Warns
    -----
    UserWarning
        With ``kinematics='model'``, when ``p1`` is a ``'cylshell_*'``
        model with the Donnell kinematics, ``'cylshell_clpt_donnell'``,
        ``'cylshell_fsdt_donnell'`` or ``'cylshell_tsdt_donnell'``, and the
        connected line is not its reference surface, `z_f \neq 0`. The
        message names the section of the documentation on the residual of
        a rigid rotation about the axis.

    Notes
    -----
    The two edges share the natural coordinate `\eta`, and the points of
    the edge of the flange coincide with those of the connected line of
    the base when both have the same length and, for an annular plate on a
    cylinder, span the same angle at the same radius `r_1 + z_f`. A rigid
    motion of the assembly then gives no residual of the connected
    quantities with the Sanders-Koiter kinematics at the face; with the
    Donnell kinematics the residual of `v_1(z_f) = v_2` is `-z_f v_1/r_1`,
    hence the warning, see the Notes of
    :mod:`panels.multidomain.connections.kCBFxcte_face`.

    """
    p1, p2 = connecti['p1'], connecti['p2']
    face = connecti.get(FACE_KEY)
    z = face_coordinate(p1, face)
    xcte2 = connecti['xcte2']
    if face is not None:
        edge = 0. if face == 'lower' else p2.a
        if not np.isclose(xcte2, edge, rtol=0, atol=rtol*p2.a):
            raise ValueError("A flange connected to the {0} face of the base "
                    "with 'BFxcte' must have xcte2={1}, its edge on the side "
                    "of that face, got xcte2={2}".format(face, edge, xcte2))
    cylinder = _has_prefix(p1, CYLINDER_PREFIX)
    annular = _has_prefix(p2, ANNULAR_PREFIX)
    if cylinder and annular:
        rho1 = p1.r + z
        rho2 = p2.r - xcte2
        where = ('reference surface' if face is None
                 else '{0} face'.format(face))
        if not np.isclose(rho2, rho1, rtol=rtol, atol=0):
            raise ValueError("The edge xcte2={0} of the annular plate is at "
                    "the radius {1}, the {2} of the cylinder at the radius "
                    "r + z = {3} + {4} = {5}: the radii must be equal"
                    .format(xcte2, rho2, where, p1.r, z, rho1))
        if not np.isclose(p2.b/p2.r, p1.b/p1.r, rtol=rtol, atol=0):
            raise ValueError("The annular plate and the cylinder must span "
                    "the same angle, got b2/r2 = {0} and b1/r1 = {1}"
                    .format(p2.b/p2.r, p1.b/p1.r))
    elif face is not None:
        L1 = p1.b*(1 + z/p1.r) if cylinder else p1.b
        L2 = p2.b*(p2.r - xcte2)/p2.r if annular else p2.b
        if not np.isclose(L1, L2, rtol=rtol, atol=0):
            raise ValueError("The connected lines of the base and of the "
                    "flange must have the same length, got {0} and {1}"
                    .format(L1, L2))
    kinematics = kinematics_of(connecti)
    if (kinematics == 'model' and cylinder and z != 0
            and kCsdt._rinv_sanders(p1) == 0):
        warnings.warn("The connection 'BFxcte' at the {0} face of the "
                "cylinder, model '{1}', uses the Donnell kinematics of the "
                "model ({2}='model'), without the term z*v/r: a rigid "
                "rotation about the axis leaves a residual of the order of "
                "h/r at the face, see {3}".format(face, p1.model,
                KINEMATICS_KEY, DONNELL_SECTION), UserWarning, stacklevel=2)
    return z


def _rotation_x_at(p, z):
    r"""
    Terms of the rotation of the normal about `y` at the coordinate `z`

    Parameters
    ----------
    p : :class:`.Shell`
        Panel.
    z : float
        Coordinate measured from the reference surface of ``p``, length.

    Returns
    -------
    terms : list
        The rotation as a list of ``(field, der_x, der_y, coeff)``, see
        :func:`.kCsdt._connection_matrix`.

    Notes
    -----
    `\theta_x(z) = \partial u(z)/\partial z`. For the CLPT, `\theta_x =
    -w_{,x}` at every `z`. For the shear deformation theories, from `u(z)`
    of :func:`.kCsdt._surface_terms`,

    .. math::

        \theta_x(z) = \left[1 - 3 c_1 (z - d)^2\right] \phi_x - 3 c_1
        (z - d)^2 w_{,x}

    with `c_1 = 4/(3 h^2)` for the TSDT and `c_1 = 0` for the FSDT; at `z =
    0` it is :func:`.kCsdt._rotation_x`, and at a face of a TSDT laminate,
    `z - d = \pm h/2`, it is `-w_{,x}`, the transverse shear strain
    vanishing there.

    """
    if kCsdt._dofs(p) == 3:
        return [(W, 1, 0, -1.)]
    if not kCsdt._is_tsdt(p):
        return [(PHIX, 0, 0, 1.)]
    a = 4.*(z - p.offset)**2/sum(p.plyts)**2
    terms = [(PHIX, 0, 0, 1. - a)]
    if a != 0:
        terms.append((W, 1, 0, -a))
    return terms


def bfxcte_face_components(p1, p2, z, kt=1., kr=1., kinematics='sanders'):
    r"""
    Quantities of the connection ``'BFxcte'`` at a coordinate `z` of the base

    The displacements and the rotation of the base at `z` and those of the
    flange at its reference surface, connected with ``kt`` and ``kr``.

    Parameters
    ----------
    p1 : :class:`.Shell`
        Base.
    p2 : :class:`.Shell`
        Flange, a model with the same number of DOFs per term as ``p1``.
    z : float
        Coordinate `z_f` of the connected line in ``p1``, see
        :func:`.check_bfxcte`, length.
    kt, kr : float, optional
        Penalties of the displacements and of the rotation.
    kinematics : str, optional
        Kinematics of the face, ``'sanders'`` or ``'model'``, see
        :func:`.face_rinv`.

    Returns
    -------
    comps : list
        Four ``(penalty, ((p1, terms1), (p2, terms2)))``, see
        :func:`.kCsdt._connection_matrix`: `u_1(z_f) = w_2`, `v_1(z_f) =
        v_2`, `w_1 = -u_2` with ``kt`` and `\theta_{x1}(z_f) =
        \theta_{x2}` with ``kr``.

    Notes
    -----
    The quantities of the energy of the Notes of
    :mod:`panels.multidomain.connections.kCBFxcte_face`, with `u_1(z)`,
    `v_1(z)` of :func:`.kCsdt._surface_terms`, the term `z v_1/r_1` with
    the `1/r_1` of :func:`.face_rinv`, `\theta_{x1}(z)` of
    :func:`._rotation_x_at` and `\theta_{x2}` of :func:`.kCsdt._rotation_x`.
    For `z = 0` they are those of the connection at the reference surface,
    see :func:`.nullspace._clpt_components` and
    :func:`.nullspace._sdt_components`.

    """
    tu, tv = kCsdt._surface_terms(p1, z, face_rinv(p1, kinematics))
    return [(kt, ((p1, tu), (p2, [(W, 0, 0, 1.)]))),
            (kt, ((p1, tv), (p2, [(V, 0, 0, 1.)]))),
            (kt, ((p1, [(W, 0, 0, 1.)]), (p2, [(U, 0, 0, -1.)]))),
            (kr, ((p1, _rotation_x_at(p1, z)), (p2, kCsdt._rotation_x(p2))))]


def fkCBFxcte_face(kt, kr, p1, p2, xcte1, xcte2, z, size,
                   kinematics='sanders'):
    r"""
    Penalty stiffness matrix of the connection ``'BFxcte'`` at a face

    The edge ``xcte2`` of the flange ``p2`` is connected to the line
    ``xcte1`` at the coordinate `z_f` of the base ``p1``, with the
    quantities of :func:`.bfxcte_face_components`.

    Parameters
    ----------
    kt : float
        Translation penalty stiffness, force per unit area.
    kr : float
        Rotation penalty stiffness, force.
    p1 : :class:`.Shell`
        Base, or skin.
    p2 : :class:`.Shell`
        Flange, a model with the same number of DOFs per term as ``p1``.
    xcte1 : float
        Coordinate `x_1` of the connection in ``p1``, length, between ``0``
        and ``p1.a``.
    xcte2 : float
        Coordinate `x_2` of the connection in ``p2``, length, ``0`` or
        ``p2.a``, see :func:`.check_bfxcte`.
    z : float
        Coordinate `z_f` of the connected line in ``p1``, measured from its
        reference surface, length, see :func:`.face_coordinate`.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    kinematics : str, optional
        Kinematics of the face, ``'sanders'``, the default, or ``'model'``,
        see :func:`.face_rinv`.

    Returns
    -------
    kC : scipy.sparse.coo_matrix
        Upper triangle of the penalty stiffness matrix of the connection,
        with the blocks of ``p1`` and ``p2`` at their ``row_start`` and
        ``col_start``, shape ``(size, size)``.

    Raises
    ------
    NotImplementedError
        When the models of ``p1`` and ``p2`` have a different number of
        DOFs per term.

    Notes
    -----
    The second derivatives of the penalty energy of
    :mod:`panels.multidomain.connections.kCBFxcte_face` with respect to the
    Ritz constants of the two panels: each quantity of
    :func:`.bfxcte_face_components` is a linear combination `q_k^p = \sum_t
    \alpha_t \boldsymbol{N}_t \boldsymbol{c}^p`, and

    .. math::

        \boldsymbol{K}_{PQ} = \pm k_k \frac{b_1}{2} \sum_{t \in P} \sum_{s
        \in Q} \alpha_t \alpha_s \int_{-1}^{1} \boldsymbol{N}_t^\top
        \boldsymbol{N}_s \, d\eta

    with the sign `+` for `P = Q` and `-` for `P \neq Q`, the functions
    across the connection evaluated at `\xi_1 = 2 x_{cte1}/a_1 - 1` and
    `\xi_2 = 2 x_{cte2}/a_2 - 1`. With `b_1 = b_2` the matrix is that of
    :func:`.kCsdt._connection_matrix`, which integrates over the length of
    each panel of the rows.

    """
    if kCsdt._dofs(p1) != kCsdt._dofs(p2):
        raise NotImplementedError("Connection 'BFxcte' between models with "
                "a different number of DOFs per term, got models '{0}' and "
                "'{1}'".format(p1.model, p2.model))
    comps = bfxcte_face_components(p1, p2, z, kt, kr, kinematics)
    ctes = (xcte1, xcte2)
    rows, cols, vals = [], [], []
    for penalty, sides in comps:
        if penalty == 0:
            continue
        for iP, (pP, termsP) in enumerate(sides):
            for iQ, (pQ, termsQ) in enumerate(sides):
                sign = 1 if iP == iQ else -1
                for termP in termsP:
                    for termQ in termsQ:
                        coeff = sign*penalty*termP[3]*termQ[3]
                        if coeff == 0:
                            continue
                        X = np.outer(
                            kCsdt._point(pP, termP[0], termP[1], 'x',
                                         ctes[iP]),
                            kCsdt._point(pQ, termQ[0], termQ[1], 'x',
                                         ctes[iQ]))
                        #NOTE kCsdt._integral integrates over the length of
                        #     pP, here over the length b1 of the base
                        Y = (kCsdt._integral(pP, termP[0], termP[2],
                                             pQ, termQ[0], termQ[2], 'y')
                             *p1.b/pP.b)
                        #NOTE Ritz constant of term (i, j) at dofs*(j*m + i)
                        block = coeff*np.kron(Y, X)
                        r = (pP.row_start + kCsdt._dofs(pP)
                             *np.arange(block.shape[0]) + termP[0])
                        c = (pQ.col_start + kCsdt._dofs(pQ)
                             *np.arange(block.shape[1]) + termQ[0])
                        R, C = np.meshgrid(r, c, indexing='ij')
                        rows.append(R.ravel())
                        cols.append(C.ravel())
                        vals.append(block.ravel())
    if not rows:
        return coo_matrix((size, size), dtype=DOUBLE)
    kC = coo_matrix((np.concatenate(vals), (np.concatenate(rows).astype(INT),
                     np.concatenate(cols).astype(INT))), shape=(size, size))
    kC.sum_duplicates()
    return triu(kC, format='coo')
