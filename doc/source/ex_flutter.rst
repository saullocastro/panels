Panel flutter analysis
======================

The aerodynamic stiffness matrix of the linear piston theory is calculated
with :meth:`.Shell.calc_kA`, as a function of the aerodynamic parameter
``beta`` of the :class:`.Shell` object. The flutter onset is found by
increasing ``beta`` and solving the eigenvalue problem
`([K_C] + [K_G] + [K_A] + \lambda^2 [M])\{\bar{c}\} = \{0\}` with
:func:`structsolve.freq`, until two natural frequencies coalesce and the
eigenvalues become complex. In this convention `[K_A] = -\int \left(\beta
(\boldsymbol{S}^w_{,x})^\top \boldsymbol{S}^w + \gamma (\boldsymbol{S}^w)^\top
\boldsymbol{S}^w\right) dA`, with the row `\boldsymbol{S}^w` of the
approximation functions of `w`, for the load
`q = \beta w_{,x} + \gamma w` of a flow along `x`, where the curvature
correction `\gamma` of Krumhaar softens a cylindrical shell, see
:meth:`.Shell.calc_kA`. The term in `\beta` is the transpose of that of Eq.
(21) of [castro2016FlutterPanel]_, which for `w = 0` at the leading and
trailing edges corresponds to reversing the flow direction and does not
change the flutter boundary, whereas the sign of the term in `\gamma` of
Eqs. (21) and (29) of that paper is reversed, a mistake of the paper. The
aerodynamic damping, :meth:`.Shell.calc_cA`, is `i [C_A]` with `[C_A]` of
Eq. (23) of the paper, entering the eigenvalue problem as `+\omega [c_A]`.
The code below is extracted from one of the ``panels`` unit tests, which
verifies the critical ``beta`` with and without pre-stress:

.. literalinclude:: ../../tests/tests_shell/test_aero.py
    :pyobject: test_aero
