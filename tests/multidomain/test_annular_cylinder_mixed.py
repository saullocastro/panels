r"""Webs and cylinders of different shear deformation theories connected with
``'BFxcte'``

An annular web of the first-order shear deformation theory (FSDT),
``'annular_fsdt_donnell'``, on a cylinder of the third-order one (TSDT),
``'cylshell_tsdt_sanders'`` or ``'cylshell_tsdt_donnell'``, and an annular
web of the TSDT, ``'annular_tsdt_donnell'``, on an FSDT cylinder,
``'cylshell_fsdt_sanders'`` or ``'cylshell_fsdt_donnell'``. Small models (few
terms). Verified, with the null-space and the penalty connections, at the
reference surface of the cylinder and at its inner face (key ``'face'``):

- a thin web, `h/r_o = 0.001`, `\beta = 0.4`, clamped through the
  connection by the clamped end of the cylinder: the frequencies against
  those of the same web clamped directly, and these against the roots of
  the frequency equation of the annulus clamped outside and free inside
  (Leissa 1969), with
  :func:`test_annular_cylinder.check_web_clamped`;
- a free cylinder with two webs: the translation along the axis and the
  rotation about it give no strain energy and, with the default
  ``conn_method`` of :class:`.MultiDomain`, ``'null-space'``, satisfy the
  constraints.

Notes
-----
Both models have the 5 DOFs `u, v, w, \phi_x, \phi_y` per term, and the
connection ties `u_{cyl}(z_f) = w_{web}`, `v_{cyl}(z_f) = v_{web}`,
`w_{cyl} = -u_{web}` and the rotations of the normal about the
circumferential axis, `\theta_{x,cyl}(z_f) = \theta_{x,web}`, at the
coordinate `z_f` of the cylinder, `z_f = 0` without ``'face'``, see
:mod:`panels.multidomain.connections.kCBFxcte_face`. The rotation of each
side follows its own model: `\theta_x = \phi_x` at every `z` of an FSDT
cylinder and at the reference surface of an FSDT web; at the reference
surface of a TSDT cylinder or web without offset, `\theta_x = \phi_x`, and
at a face of a TSDT cylinder `\theta_x = -w_{,x}`, where the transverse
shear strain of the TSDT vanishes. The displacements at a face are those of
the model of the cylinder, `u(z) = u + z \phi_x` for the FSDT, with the
cubic terms of the TSDT, and `v(z)` with the Sanders-Koiter term `z v/r`
by default. At the clamped end of the cylinder all of them vanish, and the
connection sets `u = v = w = \phi_x = 0` on the connected edge of the web.

The reference roots are `\lambda^2 = \omega r_o^2 \sqrt{\rho h/D}` =
13.5004, 19.3892 and 31.3384 for `n = 0, 1, 2` nodal diameters, `\nu =
1/3`, `r_o` the outer radius of the web, from Leissa [1]_, Eqs.
(1.11)-(1.13) and (1.17). With webs of `m = 10`, `n = 12` terms and the null
space, the FSDT web on the TSDT cylinder gives the roots to `-2.4 \times
10^{-5}` and the TSDT web on the FSDT cylinder to `-3.1 \times 10^{-5}`, the
effect of the transverse shear of the web, `h/r_o = 0.001`.

References
----------
.. [1] Leissa, A. W., Vibration of Plates, NASA SP-160, 1969, Eqs.
   (1.11)-(1.13) and (1.17), p. 2.

"""
import numpy as np
import pytest

from panels.multidomain.connections import nullspace

import test_annular_cylinder as tac


#: (cylinder, web) of different shear deformation theories
MIXED = [('cylshell_tsdt_sanders', 'annular_fsdt_donnell'),
         ('cylshell_fsdt_sanders', 'annular_tsdt_donnell')]
#: (cylinder, web) for the rigid-body motions, Sanders-Koiter and Donnell
#: cylinders
MIXED_RIGID = MIXED + [('cylshell_tsdt_donnell', 'annular_fsdt_donnell'),
                       ('cylshell_fsdt_donnell', 'annular_tsdt_donnell')]


@pytest.mark.parametrize('cyl, web', MIXED)
@pytest.mark.parametrize('face', [None, 'lower'])
@pytest.mark.parametrize('conn_method', ['null-space', 'penalty'])
def test_mixed_web_exact_roots(conn_method, face, cyl, web):
    r"""Web, `h/r_o = 0.001`, `\beta = 0.4`, its outer edge on the reference
    surface (no ``'face'``) or on the inner surface (``face='lower'``) of a
    cylinder of the other shear deformation theory, clamped at `x = 0`:
    the five lowest frequencies against those of the web clamped directly,
    see :func:`test_annular_cylinder.check_web_clamped`"""
    tac.check_web_clamped(cyl, web, face, conn_method)


@pytest.mark.parametrize('faces', [True, False])
@pytest.mark.parametrize('cyl, web', MIXED_RIGID)
@pytest.mark.parametrize('conn_method', [None, 'penalty'])
def test_mixed_rigid_body_motions(conn_method, cyl, web, faces):
    """Free cylinder with two webs of the other shear deformation theory,
    an internal frame on its inner surface at `x = 0` and an external frame
    on its outer surface at `x = L` or, with ``faces`` false, two internal
    frames at its ends connected at its reference surface: the translation
    along the axis and the rotation about it give no strain energy and,
    with the default ``conn_method``, ``'null-space'``, satisfy the
    constraints"""
    md, cyls, webs, conn_webs = tac._frames(cyl, conn_method=conn_method,
                                            web=web, faces=faces)
    if conn_method is None:
        assert md.conn_method == 'null-space'
        assert all(md._null_space(ci) for ci in conn_webs)
    K = md.calc_kC(silent=True)
    size = md.get_size()
    c_t, c_r = tac._rigid(md, cyls, webs)
    for c in (c_t, c_r):
        energy = c @ (K @ c)
        assert abs(energy) < 1e-10*abs(K.diagonal()).max()*(c @ c)
        if conn_method is None:
            for ci in md.conn:
                B = nullspace.constraint_matrix(ci, size)
                assert np.abs(B @ c).max() < 1e-10*np.abs(c).max()
