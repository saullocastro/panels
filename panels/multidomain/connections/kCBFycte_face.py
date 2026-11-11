r"""
Base-flange connection ``'BFycte'`` at a face of the base

With the key ``'face'`` of the connection dictionary, ``'lower'`` or
``'upper'``, the connection ``'BFycte'`` ties the edge `y_2 =` ``ycte2`` of
the flange ``p2`` to the face of the base ``p1`` at `z_1 = d_1 \mp h_1/2`,
along the line `y_1 =` ``ycte1``, instead of its reference surface `z_1 =
0`. ``'lower'`` and ``'upper'`` refer to the coordinate `z_1` of the base
``p1``, as for ``'BFxcte'``, see
:mod:`panels.multidomain.connections.kCBFxcte_face`. On a ``'cylshell_*'``
base the line `y_1 =` const is a generator, and the flange is a flat plate
along it, e.g. a blade stringer or the web of a Z stringer: ``'lower'`` is
the inner surface, for an internal stringer, and ``'upper'`` the outer
surface, for an external one. The points of the face follow the
Sanders-Koiter kinematics of a ``'cylshell_*'`` base, with the term `z
v_1/r_1`, whatever the model of the cylinder, or, with
``kinematics='model'``, the kinematics of its model, see
:func:`.kCBFxcte_face.face_rinv`; the points of the face of another base
follow the kinematics of its model::

    dict(p1=skin, p2=blade, func='BFycte', ycte1=0, ycte2=0,
         face='lower')
    dict(p1=skin, p2=blade, func='BFycte', ycte1=0, ycte2=blade.b,
         face='upper')

Without the key, or with ``face=None``, the connection ties the reference
surfaces, see :mod:`panels.multidomain.connections.kCBFycte`, and the key
``'kinematics'`` must not be given. :func:`.check_bfycte` checks the
geometry of the connection, :func:`.bfycte_face_components` gives the
connected quantities and :func:`.fkCBFycte_face` the penalty stiffness
matrix; the null-space method imposes the same quantities, see
:func:`.nullspace.constraint_matrix`.

Notes
-----
The local axes of the flange are those of the base rotated by 90 degrees
about the common axis `x`, `\boldsymbol{e}_{x2} = \boldsymbol{e}_{x1}`,
`\boldsymbol{e}_{y2} = -\boldsymbol{e}_{z1}` and `\boldsymbol{e}_{z2} =
\boldsymbol{e}_{y1}`, see :mod:`panels.multidomain.connections.kCBFycte`.
The flange therefore lies on the side of negative `z_1` of the line
connected to its edge `y_2 = 0`, and on the side of positive `z_1` of the
line connected to its edge `y_2 = b_2`: a flange attached to a face, and
not crossing the base, has ``ycte2 = 0`` at the lower face and ``ycte2 =
p2.b`` at the upper face.

At the face `z_1 = z_f`, with `z_f = d_1 - h_1/2` or `d_1 + h_1/2`, the
penalty energy is

.. math::

    U = \frac{1}{2} \int_0^{a_1} k_t \left[ \left(u_1(z_f) - u_2\right)^2
        + \left(v_1(z_f) - w_2\right)^2 + \left(w_1 + v_2\right)^2 \right]
        + k_r \left(\theta_{y1}(z_f) - \theta_{y2}\right)^2 dx

with the Kirchhoff-Love kinematics of the classical laminated plate theory
(CLPT), the normal being rigid and normal to the deformed reference
surface,

.. math::

    u_1(z) = u_1 - z w_{1,x}, \qquad v_1(z) = \left(1 + \frac{z}{r_1}\right)
    v_1 - z w_{1,y}, \qquad \theta_{y1}(z) = \frac{\partial v_1(z)}{\partial
    z} = -w_{1,y} + \frac{v_1}{r_1}

the terms `z v_1/r_1` and `v_1/r_1` being those of the Sanders-Koiter
kinematics, with the same factor `1/r_1` of
:func:`.kCBFxcte_face.face_rinv` in both, so that the rotation is that of
the normal whose points give `v_1(z)`. For the first-order (FSDT) and
third-order (TSDT) shear deformation theories, `u_1(z)` and `v_1(z)` are
those of :func:`.kCsdt._surface_terms` and `\theta_{y1}(z) = \partial
v_1/\partial z` at `z = z_f`, see :func:`._rotation_y_at`. The flange is
connected at its reference surface, `\theta_{y2}` being the rotation of
:func:`.kCsdt._rotation_y`. The rotation about the axis of a cylinder,
`v_1 = r_1 \Omega`, moves the face by `(r_1 + z_f) \Omega` and rotates its
normal by `\Omega`; the flange, a flat plate rotating rigidly about the same
axis, has `w_2 = (r_1 + z_f) \Omega` at its edge on the face and
`\theta_{y2} = \Omega`: the motion gives `U = 0` with the Sanders-Koiter
terms. The rigid translations and the rotations about axes normal to the
axis of the cylinder also give `U = 0`. With ``kinematics='model'`` a
Donnell cylinder, which neglects `z v_1/r_1` and `v_1/r_1`, gives for the
rotation about the axis `v_1(z_f) - w_2 = -z_f \Omega` and `\theta_{y1} -
\theta_{y2} = -\Omega`, and :func:`.check_bfycte` issues a
``UserWarning``; at the reference surface, ``face=None``, the kernels of
:mod:`panels.multidomain.connections.kCBFycte` keep the kinematics of the
model.

The two panels share the natural coordinate `\xi = 2 x/a - 1` along the
connection, which requires `a_1 = a_2`. The penalty stiffness matrix is
computed with the closed-form integrals of the Legendre-based (Bardell)
approximation functions [1]_ of :func:`.kCsdt._connection_matrix`.

References
----------
.. [1] Castro, S. G. P., and Donadon, M. V., Assembly of Semi-Analytical
   models to Address Linear Buckling and Vibration of Stiffened Composite
   Panels with Debonding Defect, Composite Structures, 160, 232-247, 2017.
   https://doi.org/10.1016/j.compstruct.2016.10.026

"""
import warnings

import numpy as np

from . import kCsdt
from .kCBFxcte_face import (FACE_KEY, KINEMATICS_KEY, CYLINDER_PREFIX,
        DONNELL_SECTION, _has_prefix, face_coordinate, kinematics_of,
        face_rinv)


U, V, W, PHIX, PHIY = range(5)


def check_bfycte(connecti, rtol=1.e-8):
    r"""
    Check the geometry of a connection ``'BFycte'``

    With the key ``'face'``, the edge ``ycte2`` of the flange ``p2`` must
    be ``0`` at the ``'lower'`` face and ``p2.b`` at the ``'upper'`` face of
    the base ``p1``, the flange lying on the side of the face, and the
    lengths ``p1.a`` and ``p2.a`` of the connected lines must be equal.

    Parameters
    ----------
    connecti : dict
        Connection ``'BFycte'``, with the base ``p1``, the flange ``p2``,
        ``ycte1``, ``ycte2`` and optionally ``'face'`` and
        ``'kinematics'``, see :meth:`.MultiDomain.get_kC_conn`.
    rtol : float, optional
        Relative tolerance of the comparisons of lengths.

    Returns
    -------
    z : float
        Coordinate `z_f` of the connected line in ``p1``, measured from its
        reference surface, ``0.`` without the key ``'face'``, see
        :func:`.kCBFxcte_face.face_coordinate`, length.

    Raises
    ------
    ValueError
        When ``'face'`` is not ``None``, ``'lower'`` or ``'upper'``, when
        ``ycte2`` is not the edge of the flange on the side of the face,
        when the lengths of the connected lines differ, when
        ``'kinematics'`` is not ``None``, ``'sanders'`` or ``'model'``, or
        when ``'kinematics'`` is given without ``'face'``, see
        :func:`.kCBFxcte_face.kinematics_of`.

    Warns
    -----
    UserWarning
        With ``kinematics='model'``, when ``p1`` is a ``'cylshell_*'``
        model with the Donnell kinematics and the connected line is not its
        reference surface, `z_f \neq 0`. The message names the section of
        the documentation on the residual of a rigid rotation about the
        axis.

    Notes
    -----
    The two edges share the natural coordinate `\xi`, and the points of
    the edge of the flange coincide with those of the connected line of
    the base when both have the same length. A rigid motion of the
    assembly then gives no residual of the connected quantities with the
    Sanders-Koiter kinematics at the face, see the Notes of
    :mod:`panels.multidomain.connections.kCBFycte_face`.

    """
    p1, p2 = connecti['p1'], connecti['p2']
    face = connecti.get(FACE_KEY)
    kinematics = kinematics_of(connecti, 'BFycte')
    z = face_coordinate(p1, face, 'BFycte')
    if face is None:
        return z
    ycte2 = connecti['ycte2']
    edge = 0. if face == 'lower' else p2.b
    if not np.isclose(ycte2, edge, rtol=0, atol=rtol*p2.b):
        raise ValueError("A flange connected to the {0} face of the base "
                "with 'BFycte' must have ycte2={1}, its edge on the side "
                "of that face, got ycte2={2}".format(face, edge, ycte2))
    if not np.isclose(p1.a, p2.a, rtol=rtol, atol=0):
        raise ValueError("The connected lines of the base and of the "
                "flange must have the same length, got a1={0} and a2={1}"
                .format(p1.a, p2.a))
    if (kinematics == 'model' and _has_prefix(p1, CYLINDER_PREFIX)
            and z != 0 and kCsdt._rinv_sanders(p1) == 0):
        warnings.warn("The connection 'BFycte' at the {0} face of the "
                "cylinder, model '{1}', uses the Donnell kinematics of the "
                "model ({2}='model'), without the term z*v/r: a rigid "
                "rotation about the axis leaves a residual of the order of "
                "h/r at the face, see {3}".format(face, p1.model,
                KINEMATICS_KEY, DONNELL_SECTION), UserWarning, stacklevel=2)
    return z


def _rotation_y_at(p, z, rinv):
    r"""
    Terms of the rotation of the normal about `x` at the coordinate `z`

    Parameters
    ----------
    p : :class:`.Shell`
        Panel.
    z : float
        Coordinate measured from the reference surface of ``p``, length.
    rinv : float
        `1/r` of the term `z v/r` of `v(z)`, see
        :func:`.kCBFxcte_face.face_rinv`, 1/length.

    Returns
    -------
    terms : list
        The rotation as a list of ``(field, der_x, der_y, coeff)``, see
        :func:`.kCsdt._connection_matrix`.

    Notes
    -----
    `\theta_y(z) = \partial v(z)/\partial z`, with `v(z)` of
    :func:`.kCsdt._surface_terms`. For the CLPT, `\theta_y = -w_{,y} + v/r`
    at every `z`. For the shear deformation theories

    .. math::

        \theta_y(z) = \frac{v}{r} + \left[1 - 3 c_1 (z - d)^2\right]
        \phi_y - 3 c_1 (z - d)^2 w_{,y}

    with `c_1 = 4/(3 h^2)` for the TSDT and `c_1 = 0` for the FSDT; at a
    face of a TSDT laminate, `z - d = \pm h/2`, it is `v/r - w_{,y}`, the
    transverse shear strain vanishing there.

    """
    if kCsdt._dofs(p) == 3:
        terms = [(W, 0, 1, -1.)]
    elif not kCsdt._is_tsdt(p):
        terms = [(PHIY, 0, 0, 1.)]
    else:
        a = 4.*(z - p.offset)**2/sum(p.plyts)**2
        terms = [(PHIY, 0, 0, 1. - a)]
        if a != 0:
            terms.append((W, 0, 1, -a))
    if rinv != 0:
        terms.append((V, 0, 0, rinv))
    return terms


def bfycte_face_components(p1, p2, z, kt=1., kr=1., kinematics='sanders'):
    r"""
    Quantities of the connection ``'BFycte'`` at a coordinate `z` of the base

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
        :func:`.check_bfycte`, length.
    kt, kr : float, optional
        Penalties of the displacements and of the rotation.
    kinematics : str, optional
        Kinematics of the face, ``'sanders'`` or ``'model'``, see
        :func:`.kCBFxcte_face.face_rinv`.

    Returns
    -------
    comps : list
        Four ``(penalty, ((p1, terms1), (p2, terms2)))``, see
        :func:`.kCsdt._connection_matrix`: `u_1(z_f) = u_2`, `v_1(z_f) =
        w_2`, `w_1 = -v_2` with ``kt`` and `\theta_{y1}(z_f) =
        \theta_{y2}` with ``kr``.

    Notes
    -----
    The quantities of the energy of the Notes of
    :mod:`panels.multidomain.connections.kCBFycte_face`, with `u_1(z)`,
    `v_1(z)` of :func:`.kCsdt._surface_terms` and `\theta_{y1}(z)` of
    :func:`._rotation_y_at`, both with the `1/r_1` of
    :func:`.kCBFxcte_face.face_rinv`, and `\theta_{y2}` of
    :func:`.kCsdt._rotation_y`. For `z = 0` and ``kinematics='model'``
    they are those of the connection at the reference surface, see
    :func:`.nullspace._clpt_components` and
    :func:`.nullspace._sdt_components`.

    """
    rinv = face_rinv(p1, kinematics)
    tu, tv = kCsdt._surface_terms(p1, z, rinv)
    return [(kt, ((p1, tu), (p2, [(U, 0, 0, 1.)]))),
            (kt, ((p1, tv), (p2, [(W, 0, 0, 1.)]))),
            (kt, ((p1, [(W, 0, 0, 1.)]), (p2, [(V, 0, 0, -1.)]))),
            (kr, ((p1, _rotation_y_at(p1, z, rinv)),
                  (p2, kCsdt._rotation_y(p2))))]


def fkCBFycte_face(kt, kr, p1, p2, ycte1, ycte2, z, size,
                   kinematics='sanders'):
    r"""
    Penalty stiffness matrix of the connection ``'BFycte'`` at a face

    The edge ``ycte2`` of the flange ``p2`` is connected to the line
    ``ycte1`` at the coordinate `z_f` of the base ``p1``, with the
    quantities of :func:`.bfycte_face_components`.

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
    ycte1 : float
        Coordinate `y_1` of the connection in ``p1``, length, between ``0``
        and ``p1.b``.
    ycte2 : float
        Coordinate `y_2` of the connection in ``p2``, length, ``0`` or
        ``p2.b``, see :func:`.check_bfycte`.
    z : float
        Coordinate `z_f` of the connected line in ``p1``, measured from its
        reference surface, length, see
        :func:`.kCBFxcte_face.face_coordinate`.
    size : int
        Size of the stiffness matrix of the assembly, see
        :meth:`.MultiDomain.get_size`.
    kinematics : str, optional
        Kinematics of the face, ``'sanders'``, the default, or ``'model'``,
        see :func:`.kCBFxcte_face.face_rinv`.

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
    :mod:`panels.multidomain.connections.kCBFycte_face` with respect to the
    Ritz constants of the two panels, integrated along `x` and evaluated at
    `\eta_1 = 2 y_{cte1}/b_1 - 1` and `\eta_2 = 2 y_{cte2}/b_2 - 1`, see
    :func:`.kCsdt._connection_matrix`.

    """
    if kCsdt._dofs(p1) != kCsdt._dofs(p2):
        raise NotImplementedError("Connection 'BFycte' between models with "
                "a different number of DOFs per term, got models '{0}' and "
                "'{1}'".format(p1.model, p2.model))
    comps = bfycte_face_components(p1, p2, z, kt, kr, kinematics)
    return kCsdt._connection_matrix(comps, dict(x=None, y=(ycte1, ycte2)),
                                    size)
