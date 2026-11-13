.. _cylshell_clpt_tl:

Large rotations: total-Lagrangian Kirchhoff-Love cylindrical shells
===================================================================

The model ``'cylshell_clpt_tl'`` is the cylinder of ``'cylshell_clpt_gl'``,
with the complete Green-Lagrange membrane strains, whose changes of
curvature are measured with the normal of the deformed mid-surface instead
of the fixed local basis, as ``'plate_clpt_tl'`` does for the plate, see
:ref:`plate_clpt_tl`. All its strains are invariant under any rigid rotation
of any deformed state, and their linear part is exactly that of the
Sanders-Koiter kinematics of ``'cylshell_clpt_sanders'``, so that linear
analyses with ``'cylshell_clpt_tl'`` are those of ``'cylshell_clpt_gl'`` and
``'cylshell_clpt_sanders'``. This page derives the changes of curvature from
the position of the deformed mid-surface and shows, step by step, why their
linear part is Sanders'. Every equation of the derivation is verified
symbolically with SymPy.


Strains
-------

With `y = r\theta` the arc length along the circumference and `w` positive
outwards, the membrane strains are the Green-Lagrange strains of the
mid-surface, those of ``'cylshell_clpt_gl'``,

.. math::
    :label: cyl-tl-membrane

    \varepsilon_{xx} &= u_{,x} + \tfrac{1}{2}\left(u_{,x}^2 + v_{,x}^2 + w_{,x}^2\right) \\
    \varepsilon_{yy} &= v_{,y} + \frac{w}{r} + \tfrac{1}{2}\left(u_{,y}^2
        + \left(v_{,y} + \frac{w}{r}\right)^2 + \left(w_{,y} - \frac{v}{r}\right)^2\right) \\
    \gamma_{xy} &= u_{,y} + v_{,x} + u_{,x} u_{,y} + v_{,x} \left(v_{,y} + \frac{w}{r}\right)
        + w_{,x} \left(w_{,y} - \frac{v}{r}\right)

the strains of the Flügge-Lur'e-Byrne non-linear theory as given by Amabili
[amabili2008Nonlinear]_, Eqs. (1.118)-(1.120), and the changes of curvature
are

.. math::
    :label: cyl-tl-k

    \kappa_{xx} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,xx} \qquad
    \kappa_{yy} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,yy} - \frac{1}{r}
    - \frac{\varepsilon_{xx} + 2 \varepsilon_{yy}}{r} \qquad
    \kappa_{xy} = -2 \boldsymbol{a} \cdot \boldsymbol{X}_{,xy} -
    \frac{\gamma_{xy}}{2 r}

with `\boldsymbol{X}` the deformed mid-surface and `\boldsymbol{a} =
\boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y}` its area vector, as derived
below.


Derivation of the changes of curvature
--------------------------------------

**The undeformed cylinder.** The local basis `(\boldsymbol{e}_x,
\boldsymbol{e}_y, \boldsymbol{e}_z)` is orthonormal, with `\boldsymbol{e}_x`
along the axis, `\boldsymbol{e}_y` along the circumference, the outward
normal `\boldsymbol{e}_z` and `\boldsymbol{e}_x \times \boldsymbol{e}_y =
\boldsymbol{e}_z`. Only `\boldsymbol{e}_y` and `\boldsymbol{e}_z` vary, along
the circumference, and the position `\boldsymbol{X}_0` of the mid-surface has
the derivatives

.. math::
    :label: cyl-tl-basis

    \boldsymbol{e}_{y,y} = -\frac{\boldsymbol{e}_z}{r} \qquad
    \boldsymbol{e}_{z,y} = \frac{\boldsymbol{e}_y}{r} \qquad
    \boldsymbol{X}_{0,x} = \boldsymbol{e}_x \qquad
    \boldsymbol{X}_{0,y} = \boldsymbol{e}_y \qquad
    \boldsymbol{X}_{0,yy} = -\frac{\boldsymbol{e}_z}{r} \qquad
    \boldsymbol{X}_{0,xx} = \boldsymbol{X}_{0,xy} = \boldsymbol{0}

The second fundamental form of a surface is `b_{\alpha\beta} =
\boldsymbol{n} \cdot \boldsymbol{X}_{,\alpha\beta}`, with `\boldsymbol{n}`
its unit normal [vanderheijden2008Koiter]_, Sec. 3.16, p. 182; for the
undeformed cylinder, `\boldsymbol{n} = \boldsymbol{e}_z` and

.. math::
    :label: cyl-tl-b0

    b^0_{xx} = b^0_{xy} = 0 \qquad b^0_{yy} = -\frac{1}{r}

**The deformed mid-surface.** `\boldsymbol{X} = \boldsymbol{X}_0 +
u \boldsymbol{e}_x + v \boldsymbol{e}_y + w \boldsymbol{e}_z`. Differentiating
with Eq. :eq:`cyl-tl-basis`, the components in the local basis are

.. math::
    :label: cyl-tl-Xa

    \boldsymbol{X}_{,x} = \left(1 + u_{,x},\; v_{,x},\; w_{,x}\right) \qquad
    \boldsymbol{X}_{,y} = \left(u_{,y},\; 1 + v_{,y} + \frac{w}{r},\;
    w_{,y} - \frac{v}{r}\right)

.. math::
    :label: cyl-tl-Xab

    \boldsymbol{X}_{,xx} &= \left(u_{,xx},\; v_{,xx},\; w_{,xx}\right) \qquad
    \boldsymbol{X}_{,xy} = \left(u_{,xy},\; v_{,xy} + \frac{w_{,x}}{r},\;
    w_{,xy} - \frac{v_{,x}}{r}\right) \\
    \boldsymbol{X}_{,yy} &= \left(u_{,yy},\; v_{,yy} + \frac{2 w_{,y}}{r} -
    \frac{v}{r^2},\; w_{,yy} - \frac{2 v_{,y}}{r} - \frac{w}{r^2} -
    \frac{1}{r}\right)

`\boldsymbol{F} = [\boldsymbol{X}_{,x}\; \boldsymbol{X}_{,y}]` is the
deformation gradient of ``'cylshell_clpt_gl'``, and Eq.
:eq:`cyl-tl-membrane` is `\frac{1}{2}(\boldsymbol{F}^\top \boldsymbol{F} -
\boldsymbol{I})`.

**The area vector.** The normal of the deformed mid-surface is measured with

.. math::
    :label: cyl-tl-a

    \boldsymbol{a} = \boldsymbol{X}_{,x} \times \boldsymbol{X}_{,y} = \left(
    v_{,x} \left(w_{,y} - \frac{v}{r}\right) - w_{,x} \left(1 + v_{,y} +
    \frac{w}{r}\right),\;
    w_{,x} u_{,y} - (1 + u_{,x}) \left(w_{,y} - \frac{v}{r}\right),\;
    (1 + u_{,x}) \left(1 + v_{,y} + \frac{w}{r}\right) - v_{,x} u_{,y}\right)

`\boldsymbol{a} = |\boldsymbol{a}| \boldsymbol{n}`, with the area stretch

.. math::
    :label: cyl-tl-a-norm

    |\boldsymbol{a}|^2 = \det(\boldsymbol{F}^\top \boldsymbol{F}) = (1 + 2
    \varepsilon_{xx})(1 + 2 \varepsilon_{yy}) - \gamma_{xy}^2

so that `\boldsymbol{a} \cdot \boldsymbol{X}_{,\alpha\beta} =
|\boldsymbol{a}|\, b_{\alpha\beta}`. The area vector is used instead of the
unit normal to keep the strains polynomial in the displacements, as in
``'plate_clpt_tl'``. In the notation of Koiter [vanderheijden2008Koiter]_,
Eqs. (3.16.5)-(3.16.7), p. 183, written for the cylinder in Eq. (3.17.3),
p. 202, with the linear membrane strains `\theta_{xx} = u_{,x}`,
`\theta_{yy} = v_{,y} + w/r`, `\theta_{xy} = \frac{1}{2}(u_{,y} + v_{,x})`,
the rotation about the normal `\omega_{xy} = \frac{1}{2}(v_{,x} - u_{,y})`
and the rotations of the normal `\varphi_x = w_{,x}` and `\varphi_y =
w_{,y} - v/r`, the linear part of Eqs. :eq:`cyl-tl-a` and
:eq:`cyl-tl-a-norm` is

.. math::
    :label: cyl-tl-a-lin

    \boldsymbol{a} \approx (1 + \theta_{xx} + \theta_{yy})\, \boldsymbol{e}_z
    - \varphi_x \boldsymbol{e}_x - \varphi_y \boldsymbol{e}_y \qquad
    |\boldsymbol{a}| \approx 1 + \theta_{xx} + \theta_{yy}

the normal rotated by `\varphi_x` and `\varphi_y` and stretched with the
area.

**Linearization.** With `\boldsymbol{a} = \boldsymbol{e}_z + \delta
\boldsymbol{a}` and `\boldsymbol{X}_{,\alpha\beta} =
\boldsymbol{X}_{0,\alpha\beta} + \delta \boldsymbol{X}_{,\alpha\beta}`, the
products are, to first order, `\boldsymbol{e}_z \cdot
\boldsymbol{X}_{0,\alpha\beta} + \delta \boldsymbol{a} \cdot
\boldsymbol{X}_{0,\alpha\beta} + \boldsymbol{e}_z \cdot \delta
\boldsymbol{X}_{,\alpha\beta}`. Only `\boldsymbol{X}_{0,yy}` is not zero,
and it is normal to the surface, so that the rotations of the normal do not
enter and the stretch of Eq. :eq:`cyl-tl-a-lin` enters only `yy`:

.. math::
    :label: cyl-tl-aX-lin

    \boldsymbol{a} \cdot \boldsymbol{X}_{,xx} \approx w_{,xx} \qquad
    \boldsymbol{a} \cdot \boldsymbol{X}_{,xy} \approx w_{,xy} -
    \frac{v_{,x}}{r} \qquad
    \boldsymbol{a} \cdot \boldsymbol{X}_{,yy} \approx -\frac{1}{r} -
    \frac{\theta_{xx} + \theta_{yy}}{r} + w_{,yy} - \frac{2 v_{,y}}{r} -
    \frac{w}{r^2}

Dividing by `|\boldsymbol{a}|` removes the stretch, `-(\theta_{xx} +
\theta_{yy})/r = (|\boldsymbol{a}| - 1)\, b^0_{yy}`, and leaves the linear
changes of the second fundamental form

.. math::
    :label: cyl-tl-db

    \delta b_{xx} = w_{,xx} \qquad
    \delta b_{xy} = w_{,xy} - \frac{v_{,x}}{r} \qquad
    \delta b_{yy} = w_{,yy} - \frac{2 v_{,y}}{r} - \frac{w}{r^2}

**Sanders' linear changes of curvature.** Koiter's linearized tensor of
changes of curvature is [vanderheijden2008Koiter]_, Eq. (3.16.8), p. 183,

.. math::

    \rho_{\alpha\beta} = \frac{1}{2}\left(\varphi_{\alpha|\beta} +
    \varphi_{\beta|\alpha}\right) - \frac{1}{2}\left(b^\kappa_\alpha
    \omega_{\kappa\beta} + b^\kappa_\beta \omega_{\kappa\alpha}\right)

For the cylinder, `x` and `y` are Cartesian coordinates of the developed
surface, the covariant derivatives are partial derivatives, and the only
non-zero component of `b^\kappa_\alpha` is `b^y_y = -1/r`:

.. math::
    :label: cyl-tl-koiter

    \rho_{xx} = \varphi_{x,x} = w_{,xx} \qquad
    \rho_{yy} = \varphi_{y,y} = w_{,yy} - \frac{v_{,y}}{r} \qquad
    \rho_{xy} = \frac{1}{2}\left(\varphi_{x,y} + \varphi_{y,x}\right)
    - \frac{\omega_{xy}}{2 r} = w_{,xy} - \frac{3}{4} \frac{v_{,x}}{r} +
    \frac{1}{4} \frac{u_{,y}}{r}

the Koiter-Sanders relations of [vanderheijden2008Koiter]_, Eq. (3.17.3),
p. 202, written there in the coordinates `x/r` and `y/r`. With the sign
and the engineering shear of panels they are the linear changes of
curvature of Sanders [sanders1963Nonlinear]_, those of
``'cylshell_clpt_sanders'`` in Eq. :eq:`shell-clpt-strains` with `s = 1`:

.. math::
    :label: cyl-tl-sanders

    \kappa^S_{xx} = -\rho_{xx} = -w_{,xx} \qquad
    \kappa^S_{yy} = -\rho_{yy} = -w_{,yy} + \frac{v_{,y}}{r} \qquad
    \kappa^S_{xy} = -2 \rho_{xy} = -2 w_{,xy} + \frac{3}{2} \frac{v_{,x}}{r}
    - \frac{1}{2} \frac{u_{,y}}{r}

**The change of the second fundamental form is not Sanders'.** Writing Eq.
:eq:`cyl-tl-db` with the same quantities, `\delta b_{yy} = \varphi_{y,y} -
\theta_{yy}/r` and, since `v_{,x} = \theta_{xy} + \omega_{xy}`, `\delta
b_{xy} = \frac{1}{2}(\varphi_{x,y} + \varphi_{y,x}) - (\theta_{xy} +
\omega_{xy})/(2r)`:

.. math::
    :label: cyl-tl-db-rho

    \delta b_{xx} = \rho_{xx} \qquad
    \delta b_{yy} = \rho_{yy} - \frac{\theta_{yy}}{r} \qquad
    \delta b_{xy} = \rho_{xy} - \frac{\theta_{xy}}{2 r}

The second fundamental form sees the whole in-plane gradient `v_{,x}`,
strain and rotation about the normal, and the circumferential strain
`\theta_{yy}`, where Koiter's `\rho_{\alpha\beta}` keeps only the rotation.
The plain change of the second fundamental form, `\kappa^b_{xx} = -(b_{xx}
- b^0_{xx})`, `\kappa^b_{yy} = -(b_{yy} - b^0_{yy})`, `\kappa^b_{xy} =
-2(b_{xy} - b^0_{xy})`, therefore has the linear part

.. math::
    :label: cyl-tl-kb

    \kappa^b_{xx} = \kappa^S_{xx} \qquad
    \kappa^b_{yy} = \kappa^S_{yy} + \frac{\varepsilon_{yy}}{r} \qquad
    \kappa^b_{xy} = \kappa^S_{xy} + \frac{\gamma_{xy}}{2 r}

with the linear membrane strains `\varepsilon_{yy} = \theta_{yy}` and
`\gamma_{xy} = 2 \theta_{xy}`. For a uniform radial displacement `w = w_0`,
for instance,

.. math::
    :label: cyl-tl-expansion

    \varepsilon_{yy} = \frac{w_0}{r} + \frac{w_0^2}{2 r^2} \qquad
    \kappa^b_{yy} = \frac{w_0}{r^2} \qquad
    \kappa^S_{yy} = \kappa_{yy} = 0

exactly: the change of the second fundamental form gives a uniform
expansion the change of curvature `w_0/r^2`, to first order the membrane
strain divided by the radius, which Sanders' changes of curvature and Eq.
:eq:`cyl-tl-k` do not. Koiter has shown that, within the classical theory
of thin shells, the changes of curvature can be altered by terms of the
order of the membrane strain divided by the smallest radius of curvature
without altering the accuracy of the theory [vanderheijden2008Koiter]_,
p. 202: both measures are admissible, but only the first has the linear
part of ``'cylshell_clpt_sanders'``, whose kernels give the linear analyses
of ``'cylshell_clpt_gl'``.

**The changes of curvature of the model.** Eq. :eq:`cyl-tl-k` is the
change of the second fundamental form times the area stretch, with the
reference term multiplied by the first-order stretch `1 + \varepsilon_{xx} +
\varepsilon_{yy}`, plus the two membrane terms of Eq. :eq:`cyl-tl-kb`
with the opposite sign:

.. math::

    \kappa_{yy} = -\left(\boldsymbol{a} \cdot \boldsymbol{X}_{,yy} - (1 +
    \varepsilon_{xx} + \varepsilon_{yy})\, b^0_{yy}\right) -
    \frac{\varepsilon_{yy}}{r} \qquad
    \kappa_{xy} = -2 \left(\boldsymbol{a} \cdot \boldsymbol{X}_{,xy} - (1 +
    \varepsilon_{xx} + \varepsilon_{yy})\, b^0_{xy}\right) -
    \frac{\gamma_{xy}}{2 r}

and `\kappa_{xx} = -\boldsymbol{a} \cdot \boldsymbol{X}_{,xx}`, since
`b^0_{xx} = b^0_{xy} = 0`. Term by term, with Eq. :eq:`cyl-tl-aX-lin`, the
linear parts are

.. math::
    :label: cyl-tl-k-lin

    \kappa_{xx} &\approx -w_{,xx} \\
    \kappa_{yy} &\approx \underbrace{\frac{1}{r} + \frac{\theta_{xx} +
    \theta_{yy}}{r} - w_{,yy} + \frac{2 v_{,y}}{r} +
    \frac{w}{r^2}}_{-\boldsymbol{a} \cdot \boldsymbol{X}_{,yy}}
    \underbrace{- \frac{1}{r} - \frac{\theta_{xx} +
    \theta_{yy}}{r}}_{(1 + \varepsilon_{xx} + \varepsilon_{yy})\, b^0_{yy}}
    \underbrace{- \frac{v_{,y}}{r} - \frac{w}{r^2}}_{-\varepsilon_{yy}/r}
    = -w_{,yy} + \frac{v_{,y}}{r} \\
    \kappa_{xy} &\approx \underbrace{-2 w_{,xy} + \frac{2
    v_{,x}}{r}}_{-2 \boldsymbol{a} \cdot \boldsymbol{X}_{,xy}}
    \underbrace{- \frac{u_{,y} + v_{,x}}{2 r}}_{-\gamma_{xy}/(2 r)}
    = -2 w_{,xy} + \frac{3}{2} \frac{v_{,x}}{r} - \frac{1}{2}
    \frac{u_{,y}}{r}

which are `\kappa^S_{xx}`, `\kappa^S_{yy}` and `\kappa^S_{xy}` of Eq.
:eq:`cyl-tl-sanders`. In `\kappa_{yy}` the curvature `1/r` of the
undeformed cylinder cancels; the stretch of the area vector, `(\theta_{xx}
+ \theta_{yy})/r`, cancels with the stretch put on the reference term
(without it, `-\boldsymbol{a} \cdot \boldsymbol{X}_{,yy} - 1/r` would keep
the further membrane term `(\theta_{xx} + \theta_{yy})/r`); and
`-\varepsilon_{yy}/r` cancels `w/r^2` and half of `2 v_{,y}/r`, the terms of
`\delta b_{yy}` that are not in `\rho_{yy}`. In `\kappa_{xy}`,
`-\gamma_{xy}/(2r)` takes the strain part out of the `v_{,x}/r` of `\delta
b_{xy}` and leaves the rotation `\omega_{xy}` with the weight of Eq.
:eq:`cyl-tl-koiter`.

**Invariance.** For a rigid rotation of a deformed state, `\boldsymbol{X}
\to \boldsymbol{R} \boldsymbol{X}`, the vectors `\boldsymbol{X}_{,\alpha}`,
`\boldsymbol{X}_{,\alpha\beta}` and `\boldsymbol{a}` rotate with
`\boldsymbol{R}`, their dot products and the Green-Lagrange strains do not
change, and so neither do the curvatures of Eq. :eq:`cyl-tl-k`. The plain
change of the second fundamental form is invariant as well, but its linear
part is not Sanders', Eq. :eq:`cyl-tl-kb`; Sanders' linear changes of
curvature, those of ``'cylshell_clpt_gl'``, are measured in the fixed local
basis and change when a bent state is rotated rigidly. Eq. :eq:`cyl-tl-k`
has both properties: invariant, and Sanders' linear part. Under a uniform
radial displacement it gives no bending strain at any amplitude, Eq.
:eq:`cyl-tl-expansion`.

**Comparison with Novozhilov's curvatures.** Up to the quadratic terms,
Eq. :eq:`cyl-tl-k` differs from the non-linear changes of curvature of
Novozhilov's theory as given by Amabili [amabili2008Nonlinear]_, Sec. 1.3.7,
Eqs. (1.130)-(1.132), by terms of the order of the membrane strain divided
by the radius or times the curvature, and by terms of the rotation about
the normal, which those equations do not treat exactly.


Internal forces, tangent stiffness and integration
--------------------------------------------------

The strains depend at a point on the seventeen generalized derivatives
`\boldsymbol{d} = \boldsymbol{\Phi} \boldsymbol{c}`: `u_{,x}`, `u_{,y}`,
`v_{,x}`, `v_{,y}`, `w_{,x}`, `w_{,y}`, `v`, `w` and the nine second
derivatives of `u`, `v`, `w`. The internal force vector and the tangent
stiffness are those of ``'plate_clpt_tl'``, see :ref:`plate_clpt_tl`, with
these derivatives: :meth:`.Shell.calc_kG` assembles `\boldsymbol{H}(\boldsymbol{\sigma}_L +
\bar{\boldsymbol{N}}, \boldsymbol{0})`, and :meth:`.Shell.calc_kC` with
``NLgeom=True`` the rest of the Jacobian of :meth:`.Shell.calc_fint`. With a
constant membrane stress the geometric stiffness is that of
``'cylshell_clpt_gl'``, whose analytical kernels the model uses: linear
static solutions, natural frequencies and linear buckling loads with a
membrane prebuckling state are those of ``'cylshell_clpt_gl'``. Follower
pressures use the exact area vector of the Sanders cylinder, and the
connections of :class:`.MultiDomain` those of the Sanders-Koiter rotations.
The curvatures are cubic in the displacements; the integrands are
polynomials of degree `6p - 2` along `x` and `6p` along `y`, where `v` and
`w` enter without derivative, and the default number of Gauss-Legendre
points, :func:`.gauss_points` with the key ``'integrand_degree'`` of
:mod:`panels.modelDB`, is `3p + 1` along each direction, against `2p + 1`
for the other cylinders. :func:`.clpt_field.fstrain` gives the curvatures
of Eq. :eq:`cyl-tl-k`.
