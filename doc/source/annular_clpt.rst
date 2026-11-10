.. _annular_clpt:

Annular-sector plates
=====================

The models ``'annular_clpt_donnell'``, ``'annular_clpt_gl'`` and
``'annular_clpt_tl'`` are flat annular sectors between the outer radius `r`
and the inner radius `r - a`, with the classical laminated plate theory
(CLPT) in polar coordinates [amabili2008Nonlinear]_, Sec. 1.4: the von
Kármán strains, the complete Green-Lagrange membrane strains, and these with
the changes of curvature of the deformed normal. They are the polar
counterparts of ``'plate_clpt_donnell'``, ``'plate_clpt_gl'`` and
``'plate_clpt_tl'``. The web of a ring frame of a stiffened cylinder, in the
plane normal to the axis and meeting the skin at its outer radius, is an
application. The kernels :mod:`panels.models.annular_clpt_donnell`,
:mod:`panels.models.annular_clpt_gl`, :mod:`panels.models.annular_clpt_tl`
and their numerically integrated modules ``*_num`` are generated from
symbolic derivations with SymPy of the strains below. The models
``'annular_fsdt_donnell'``, ``'annular_fsdt_gl'`` and ``'annular_fsdt_tl'``
are the same sectors with the first-order shear deformation theory (FSDT),
the polar counterparts of ``'plate_fsdt_donnell'``, ``'plate_fsdt_gl'`` and
``'plate_fsdt_tl'``, see :ref:`annular_fsdt`.


Geometry and coordinates
------------------------

An annular sector is a :class:`.Shell` with ``a`` the radial width, ``b``
the arc length of the outer edge and ``r`` the outer radius; for the web of
a ring frame, ``a`` is the height of the web. With the polar coordinates
`(\rho, \theta)` of the plane, the coordinates of the Ritz series of Eq.
:eq:`shell-ritz` are

.. math::

    x = r - \rho, \quad 0 \le x \le a \qquad\qquad
    y = r \theta, \quad 0 \le y \le b

`x` runs from the outer edge inwards and `y` along the outer edge. The
displacements `u, v, w` are measured along the local basis

.. math::

    \boldsymbol{e}_x = -\boldsymbol{e}_\rho, \qquad
    \boldsymbol{e}_y = \boldsymbol{e}_\theta, \qquad
    \boldsymbol{e}_z = \boldsymbol{e}_x \times \boldsymbol{e}_y

which rotates with `\theta`. In the notation of Amabili, with the radial,
circumferential and normal displacements `u_1, u_2, u_3` along
`\boldsymbol{e}_\rho`, `\boldsymbol{e}_\theta` and `\boldsymbol{e}_\rho \times
\boldsymbol{e}_\theta = -\boldsymbol{e}_z`:

.. math::

    u_1 = -u, \quad u_2 = v, \quad u_3 = -w, \qquad
    \frac{\partial}{\partial \rho} = -\frac{\partial}{\partial x}, \quad
    \frac{\partial}{\partial \theta} = r \frac{\partial}{\partial y}


Strains
-------

With `s = r/\rho` the ratio of the arc length along the outer edge to the
arc length at the radius `\rho`, the Kirchhoff displacement field through
the thickness is `u(z) = u - z w_{,x}`, `v(z) = v - z s w_{,y}`, `w(z) =
w`. The von Kármán strains of ``'annular_clpt_donnell'``, Eqs.
(1.175)-(1.180) of [amabili2008Nonlinear]_, become

.. math::

    \varepsilon_{xx} = u_{,x} + \tfrac{1}{2} w_{,x}^2 \qquad
    \varepsilon_{yy} = s v_{,y} - \frac{u}{\rho} + \tfrac{1}{2} s^2 w_{,y}^2
    \qquad
    \gamma_{xy} = s u_{,y} + v_{,x} + \frac{v}{\rho} + s w_{,x} w_{,y}

    \kappa_{xx} = -w_{,xx} \qquad
    \kappa_{yy} = \frac{w_{,x}}{\rho} - s^2 w_{,yy} \qquad
    \kappa_{xy} = -2 s \left(w_{,xy} + \frac{w_{,y}}{\rho}\right)

with `\varepsilon_{xx} = \varepsilon_\rho`, `\varepsilon_{yy} =
\varepsilon_\theta`, `\gamma_{xy} = -\gamma_{\rho\theta}` and the curvatures
of opposite sign to Amabili's, the axis `\boldsymbol{e}_z` being opposite to
his. The terms `-u/\rho` and `v/\rho` are those of the stretching of a
circumferential fibre that moves radially and of the rotation of the local
basis; `w_{,x}/\rho` is the curvature of a circumferential fibre whose
radial slope changes. The six small rigid-body motions of the plane give
zero linear strains, and for `r \to \infty` the strains are those of
``'plate_clpt_donnell'``. A flat plate has no curvature term, and Donnell's
shallow-shell strains reduce to von Kármán's [amabili2008Nonlinear]_, Sec.
1.3.10: the name follows ``'plate_clpt_donnell'``.

The deformed reference plane has, per unit length of the reference lines,
the tangent vectors

.. math::

    \boldsymbol{g}_1 = \boldsymbol{X}_{,x} = \begin{Bmatrix} 1 + u_{,x} \\
    v_{,x} \\ w_{,x} \end{Bmatrix}, \qquad
    \boldsymbol{g}_2 = s \boldsymbol{X}_{,y} = \begin{Bmatrix} s u_{,y} +
    v/\rho \\ 1 + s v_{,y} - u/\rho \\ s w_{,y} \end{Bmatrix}

in the local basis. ``'annular_clpt_gl'`` has the complete Green-Lagrange
membrane strains, Eqs. (1.166)-(1.168) of [amabili2008Nonlinear]_,

.. math::

    \varepsilon_{xx} = \tfrac{1}{2} (\boldsymbol{g}_1 \cdot \boldsymbol{g}_1
    - 1) \qquad
    \varepsilon_{yy} = \tfrac{1}{2} (\boldsymbol{g}_2 \cdot \boldsymbol{g}_2
    - 1) \qquad
    \gamma_{xy} = \boldsymbol{g}_1 \cdot \boldsymbol{g}_2

which vanish for any finite rigid motion, and the linear changes of
curvature above. ``'annular_clpt_tl'`` also measures the changes of
curvature with the area vector `\boldsymbol{A} = \boldsymbol{g}_1 \times
\boldsymbol{g}_2` of the deformed plane, as ``'plate_clpt_tl'`` does:

.. math::

    \kappa_{xx} = -\boldsymbol{A} \cdot \boldsymbol{X}_{,xx} \qquad
    \kappa_{yy} = -s^2 \boldsymbol{A} \cdot \boldsymbol{X}_{,yy} \qquad
    \kappa_{xy} = -2 s \boldsymbol{A} \cdot \boldsymbol{X}_{,xy}

with the second derivatives of the position, in the local basis,

.. math::

    \boldsymbol{X}_{,xx} = \begin{Bmatrix} u_{,xx} \\ v_{,xx} \\ w_{,xx}
    \end{Bmatrix}, \quad
    \boldsymbol{X}_{,xy} = \begin{Bmatrix} u_{,xy} + v_{,x}/r \\ v_{,xy} - (1
    + u_{,x})/r \\ w_{,xy} \end{Bmatrix}, \quad
    \boldsymbol{X}_{,yy} = \begin{Bmatrix} u_{,yy} + 2 v_{,y}/r + (\rho -
    u)/r^2 \\ v_{,yy} - 2 u_{,y}/r - v/r^2 \\ w_{,yy} \end{Bmatrix}

The terms without displacement are those of the flat reference plane in
polar coordinates, normal to `\boldsymbol{A}` of the undeformed plane. All
six strains vanish for any finite rigid motion; their linear part is that of
the von Kármán strains. For `r \to \infty` the strains of
``'annular_clpt_gl'`` and ``'annular_clpt_tl'`` are those of
``'plate_clpt_gl'`` and ``'plate_clpt_tl'``. The element of area is

.. math::

    dA = \rho \, d\rho \, d\theta = \frac{\rho}{r} \, dx \, dy

The constitutive law, the internal force vector and the split of the tangent
stiffness into ``fkC_num`` and ``fkG_num`` are those of the flat plates;
the geometric stiffness of a constant membrane state `N_{xx} =
N_{\rho\rho}`, `N_{yy} = N_{\theta\theta}`, `N_{xy} = -N_{\rho\theta}` is
the variation of its work on the second-order terms of the membrane
strains, on `(w_{,x}, s w_{,y})` for ``'annular_clpt_donnell'`` and on all
the displacement gradients for ``'annular_clpt_gl'`` and
``'annular_clpt_tl'``.


.. _annular_fsdt:

First-order shear deformation theory
------------------------------------

The models ``'annular_fsdt_donnell'``, ``'annular_fsdt_gl'`` and
``'annular_fsdt_tl'`` have the geometry, the coordinates and the local
basis above, with the rotations `\phi_x`, `\phi_y` of the normal as
independent fields: five degrees of freedom `u, v, w, \phi_x, \phi_y` per
term and the displacement field

.. math::

    u(z) = u + z \phi_x, \qquad v(z) = v + z \phi_y, \qquad w(z) = w

of ``'plate_fsdt_donnell'``. The von Kármán strains of the Mindlin plate in
polar coordinates [bagheri2019Torque]_, Eqs. (2) and (6), are those of
``'annular_fsdt_donnell'``:

.. math::

    \varepsilon_{xx} = u_{,x} + \tfrac{1}{2} w_{,x}^2 \qquad
    \varepsilon_{yy} = s v_{,y} - \frac{u}{\rho} + \tfrac{1}{2} s^2 w_{,y}^2
    \qquad
    \gamma_{xy} = s u_{,y} + v_{,x} + \frac{v}{\rho} + s w_{,x} w_{,y}

    \kappa_{xx} = \phi_{x,x} \qquad
    \kappa_{yy} = s \phi_{y,y} - \frac{\phi_x}{\rho} \qquad
    \kappa_{xy} = s \phi_{x,y} + \phi_{y,x} + \frac{\phi_y}{\rho}

    \gamma_{yz} = \phi_y + s w_{,y} \qquad
    \gamma_{xz} = \phi_x + w_{,x}

In the notation of [bagheri2019Torque]_, whose `z` is along
`-\boldsymbol{e}_z`, the radial and circumferential displacements and
rotations are `u_0 = -u`, `v_0 = v`, `w_0 = -w`, `\psi_r = \phi_x` and
`\psi_\theta = -\phi_y`. The terms `-\phi_x/\rho` and `\phi_y/\rho` are those
of `-u/\rho` and `v/\rho` for the displacements `z \phi_x`, `z \phi_y` of
the points off the reference plane. With `\phi_x = -w_{,x}` and `\phi_y = -s
w_{,y}` the transverse shear strains vanish and the changes of curvature are
those of ``'annular_clpt_donnell'``.

``'annular_fsdt_gl'`` has the Green-Lagrange membrane strains of
``'annular_clpt_gl'`` and the linear changes of curvature and transverse
shear strains above, as ``'plate_fsdt_gl'`` for the flat plate.
``'annular_fsdt_tl'`` is total-Lagrangian with the second-order director of
``'plate_fsdt_tl'`` [schmidt1988Refined]_, [palmerio1990Moderate]_: the
deformed position of a point is `\boldsymbol{X} + z \boldsymbol{d}`, with

.. math::

    \boldsymbol{d} = \phi_x \boldsymbol{e}_x + \phi_y \boldsymbol{e}_y +
    \left(1 - \tfrac{1}{2} (\phi_x^2 + \phi_y^2)\right) \boldsymbol{e}_z

and, the local basis turning with `\theta` (`\boldsymbol{e}_{x,y} =
-\boldsymbol{e}_y/r`, `\boldsymbol{e}_{y,y} = \boldsymbol{e}_x/r`), the
derivatives of the director per unit length of the reference lines

.. math::

    \boldsymbol{d}_{,x} = \begin{Bmatrix} \phi_{x,x} \\ \phi_{y,x} \\
    -(\phi_x \phi_{x,x} + \phi_y \phi_{y,x}) \end{Bmatrix}, \qquad
    s \boldsymbol{d}_{,y} = \begin{Bmatrix} s \phi_{x,y} + \phi_y/\rho \\
    s \phi_{y,y} - \phi_x/\rho \\ -s (\phi_x \phi_{x,y} + \phi_y \phi_{y,y})
    \end{Bmatrix}

The strains are the coefficients of `z^0` and `z^1` of the Green-Lagrange
strains of this field, the metric of the flat reference plane not depending
on `z`:

.. math::

    \varepsilon_{xx} = \tfrac{1}{2} (\boldsymbol{g}_1 \cdot \boldsymbol{g}_1
    - 1) \qquad
    \varepsilon_{yy} = \tfrac{1}{2} (\boldsymbol{g}_2 \cdot \boldsymbol{g}_2
    - 1) \qquad
    \gamma_{xy} = \boldsymbol{g}_1 \cdot \boldsymbol{g}_2

    \kappa_{xx} = \boldsymbol{g}_1 \cdot \boldsymbol{d}_{,x} \qquad
    \kappa_{yy} = \boldsymbol{g}_2 \cdot s \boldsymbol{d}_{,y} \qquad
    \kappa_{xy} = \boldsymbol{g}_1 \cdot s \boldsymbol{d}_{,y} +
    \boldsymbol{g}_2 \cdot \boldsymbol{d}_{,x}

    \gamma_{yz} = \boldsymbol{g}_2 \cdot \boldsymbol{d} \qquad
    \gamma_{xz} = \boldsymbol{g}_1 \cdot \boldsymbol{d}

Their linear part is that of ``'annular_fsdt_donnell'``: the reference
plane has no curvature of its own, and no correction of the linear part is
needed, unlike for the cylinder of :ref:`fsdt_tsdt_tl`. A finite rigid
rotation by the angle `t` changes the strains by `O(t^4)`, through the
length of the second-order director; the linear transverse shear strains
of ``'annular_fsdt_gl'`` change by `O(t^2)` under a rotation about an axis
inclined to the plane. For `r \to \infty` the strains of the three models
are those of ``'plate_fsdt_donnell'``, ``'plate_fsdt_gl'`` and
``'plate_fsdt_tl'``.

The constitutive matrix ``Shell.ABD`` of these models is the `8 \times 8`
matrix of the FSDT, with the transverse shear stiffness and the shear
correction chosen with ``Shell.fsdt_shear_correction``; the mass matrix has
the moments `I_0, I_1, I_2` of the density with the displacement field
above. The transverse shear lowers the buckling loads and frequencies of
thick annuli: under a torque, the buckling torque of a clamped annulus with
`b/a = 0.5` and `h/a = 0.1` is 53 % of the classical one
[bagheri2019Torque]_, Table 2.


Laminate offset
---------------

The laminate spans `d - h/2 \le z \le d + h/2` about the reference plane,
`d` being ``Shell.offset``, as for the other models of the CLPT and of the
FSDT. The reference plane being flat, `z` enters only the integrals through the
thickness: the stiffness through the matrix ``ABD`` of the laminate about
the reference plane, the coupling `B` included, and the mass through the
moments `I_0, I_1, I_2` of the density, with the displacement field
`u(z)`, `v(z)`, `w(z)` above. The connections tie the reference planes of
the connected panels.


Numerical integration
---------------------

The coefficients `1/\rho` make the integrands rational functions of `x`.
The integrals of Bardell's functions, Eq. :eq:`bardell-separation`, do not
apply, and every matrix, also of the whole domain with a constant
laminate, is integrated with the Gauss-Legendre rule. The rule converges
exponentially, the faster the farther the centre `\rho = 0` is from the
sector, i.e. the smaller `a/r`.


Connection to a cylinder
------------------------

The local axes of the annular sector are those of the flange of the
connection ``'BFxcte'``, see :ref:`multidomain_connections`: the axes of the
cylinder rotated by 90 degrees about the common circumferential axis `y`,
with `\boldsymbol{e}_x` of the sector along `-\boldsymbol{e}_z` of the
cylinder (inwards) and `\boldsymbol{e}_z` of the sector along the axis
`\boldsymbol{e}_x` of the cylinder. The web of a ring frame whose outer
radius is the radius of the cylinder and which spans the angle of a
cylinder domain shares the natural coordinate `\eta` with the edge `x = x_f`
of that domain, and

.. code-block:: python

    dict(p1=cylinder, p2=web, func='BFxcte', xcte1=x_f, xcte2=0)

imposes, at the frame station `x_f`, `u_{cyl} = w_{web}`, `v_{cyl} =
v_{web}`, `w_{cyl} = -u_{web}` and the continuity of the rotation about the
circumferential axis, with penalties or with the null space of
:class:`.MultiDomain`. A closed ring is made of several sectors connected
with ``'SSycte'`` along their radial edges, as the domains of the cylinder.

The web of a ring frame is attached to a surface of the skin. With the
key ``'face'`` of ``'BFxcte'`` the edge of the sector is connected to the
inner surface of the cylinder, `z = d - h/2`, or to its outer surface, `z
= d + h/2`, `d` being the offset and `h` the thickness of the cylinder,
see :ref:`multidomain_connections`. An internal frame, its outer edge on
the inner surface, has the outer radius `r - h/2 + d`:

.. code-block:: python

    dict(p1=cylinder, p2=web, func='BFxcte', xcte1=x_f, xcte2=0,
         face='lower')

and an external frame, its inner edge `x = a` on the outer surface, the
inner radius `r + h/2 + d`:

.. code-block:: python

    dict(p1=cylinder, p2=web, func='BFxcte', xcte1=x_f, xcte2=web.a,
         face='upper')

the points of the surface following the normal of the cylinder, `u(z) = u
- z w_{,x}`, `v(z) = (1 + z/r) v - z w_{,y}` for the Sanders-Koiter
kinematics. The sector has no eccentricity of its own: the radius of the
connected edge must be that of the connected surface, and the sector must
span the angle of the cylinder domain, `b_{web}/r_{web} = b/r`, or the
connection raises a ``ValueError``. The cylinder with its reference
surface at its inner face, ``offset = h/2``, connected without the key,
is the same internal frame.

The sectors of the first-order shear deformation theory connect in the
same way to the cylinders with five degrees of freedom per term, e.g.
``'cylshell_fsdt_sanders'``, with or without the key ``'face'``; the
rotation about the circumferential axis tied by the connection is then
`\phi_x` of both panels. The two panels of a connection must have the same
number of degrees of freedom per term: an ``'annular_fsdt_*'`` sector on a
cylinder of the classical theory, or the converse, raises a
``NotImplementedError``.
