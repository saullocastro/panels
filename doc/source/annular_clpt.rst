.. _annular_clpt_donnell:

Ring frames: the web as an annular-sector plate
===============================================

The model ``'annular_clpt_donnell'`` is the web of a ring frame of a
stiffened cylinder: a flat annular sector in the plane normal to the axis of
the cylinder, between the outer radius `r`, where it meets the skin, and the
inner radius `r - a`. It uses the classical laminated plate theory (CLPT)
with the von Kármán strains of circular plates [amabili2008Nonlinear]_, Sec.
1.4.2. The kernels :mod:`panels.models.annular_clpt_donnell` and
:mod:`panels.models.annular_clpt_donnell_num` are generated from a symbolic
derivation with SymPy of the strains below.


Geometry and coordinates
------------------------

A web is a :class:`.Shell` with ``a`` the height of the web, ``b`` the arc
length of its outer edge and ``r`` the outer radius. With the polar
coordinates `(\rho, \theta)` of the plane of the web, the coordinates of the
Ritz series of Eq. :eq:`shell-ritz` are

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
w`, and the von Kármán strains, Eqs. (1.175)-(1.180) of
[amabili2008Nonlinear]_, become

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
``'plate_clpt_donnell'``. The element of area is

.. math::

    dA = \rho \, d\rho \, d\theta = \frac{\rho}{r} \, dx \, dy

The constitutive law, the internal force vector and the split of the tangent
stiffness into ``fkC_num`` and ``fkG_num`` are those of the flat plates; the
geometric stiffness of the membrane stresses `N_{xx} = N_{\rho\rho}`,
`N_{yy} = N_{\theta\theta}`, `N_{xy} = -N_{\rho\theta}` is

.. math::

    \boldsymbol{K}_G = \int \boldsymbol{G}^\top
    \begin{bmatrix} N_{xx} & N_{xy} \\ N_{xy} & N_{yy} \end{bmatrix}
    \boldsymbol{G} \, dA, \qquad
    \boldsymbol{G} \boldsymbol{c} = \{w_{,x}, s w_{,y}\}^\top


Numerical integration
---------------------

The coefficients `1/\rho` make the integrands rational functions of `x`.
The integrals of Bardell's functions, Eq. :eq:`bardell-separation`, do not
apply, and every matrix, also of the whole domain with a constant
laminate, is integrated with the Gauss-Legendre rule. The rule converges
exponentially, the faster the farther the centre `\rho = 0` is from the web,
i.e. the smaller `a/r`.


Connection to a cylinder
------------------------

The local axes of the web are those of the flange of the connection
``'BFxcte'``, see :ref:`multidomain_connections`: the axes of the cylinder
rotated by 90 degrees about the common circumferential axis `y`, with
`\boldsymbol{e}_x` of the web along `-\boldsymbol{e}_z` of the cylinder
(inwards) and `\boldsymbol{e}_z` of the web along the axis
`\boldsymbol{e}_x` of the cylinder. A web whose outer radius is the radius of
the cylinder and which spans the angle of a cylinder domain shares the
natural coordinate `\eta` with the edge `x = x_f` of that domain, and

.. code-block:: python

    dict(p1=cylinder, p2=web, func='BFxcte', xcte1=x_f, xcte2=0)

imposes, at the frame station `x_f`, `u_{cyl} = w_{web}`, `v_{cyl} =
v_{web}`, `w_{cyl} = -u_{web}` and the continuity of the rotation about the
circumferential axis, with penalties or with the null space of
:class:`.MultiDomain`. A closed ring is made of several webs connected with
``'SSycte'`` along their radial edges, as the domains of the cylinder.


Variants
--------

``'annular_clpt_donnell'`` follows the convention of
``'plate_clpt_donnell'``, the von Kármán flat plate: a flat plate has no
curvature term, and Donnell's shallow-shell strains reduce to von Kármán's
[amabili2008Nonlinear]_, Sec. 1.3.10. The complete Green-Lagrange strains in
polar coordinates, Eqs. (1.166)-(1.168) of [amabili2008Nonlinear]_, would
give the counterpart of ``'plate_clpt_gl'``.
