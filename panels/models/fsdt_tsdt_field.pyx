#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: initializedcheck=False
#cython: overflowcheck=False
#cython: embedsignature=True
#cython: infer_types=False
r"""
Field variables of the plate and cylindrical shell models using the
first-order (FSDT) and Reddy's third-order (TSDT) shear deformation theories

The degrees of freedom of each term of the approximation are ``u, v, w, phix,
phiy``. For the models with the Sanders-Koiter kinematics, the rotation of
the normal about `x` is ``phiy + v/r``, which is the field ``phiy`` returned
by :func:`.fuvw` and the approximation of ``phiy`` of :func:`.fg`.

Notes
-----
Each field `F` of `u, v, w, \phi_x, \phi_y`, of index `k = 0, \dots, 4`, is
the Ritz series of :class:`.Shell`, `F = \sum_{j} \sum_{i} c_{5(j m + i) +
k} P_i^F(\xi) P_j^F(\eta)`, with `\xi = 2x/a - 1`, `\eta = 2y/b - 1` and
Bardell's functions of the boundary flags of `F`, see
:mod:`panels.bardell`; the derivatives are `F_{,x} = (2/a) F_{,\xi}` and
`F_{,y} = (2/b) F_{,\eta}`.

The displacement field through the thickness is, with `c_1 = 4/(3 h^2)` for
the TSDT of Reddy [1]_ and `c_1 = 0` for the FSDT, `s = 1` for the
Sanders-Koiter kinematics [2]_ and `s = 0` for the Donnell kinematics,
`1/r = 0` for the plates, `z` measured from the reference surface and the
laminate spanning `d - h/2 \le z \le d + h/2` with the offset `d`:

.. math::

    u(z) = u + z \phi_x - c_1 \left[(z - d)^3 + d^3\right] (\phi_x + w_{,x})

    v(z) = v + z \left(\phi_y + s \frac{v}{r}\right)
           - c_1 \left[(z - d)^3 + d^3\right] (\phi_y + w_{,y})

    w(z) = w

The generalized strains of :func:`.fstrain`, in the order of
:func:`.strain_names`, are `\varepsilon = \varepsilon^{(0)} + z
\varepsilon^{(1)} + \left[(z - d)^3 + d^3\right] \varepsilon^{(3)}` for
``exx, eyy, gxy``, ``kxx, kyy, kxy`` and ``kxx3, kyy3, kxy3``, and `\gamma
= \gamma^{(0)} + (z - d)^2 \gamma^{(2)}` for ``gyz, gxz`` and ``gyz2,
gxz2``, with the rotations
`\beta_x = -w_{,x}` and `\beta_y = -w_{,y} + s v/r` of the von Kármán
non-linear terms:

.. math::

    \varepsilon^{(0)} = \{ u_{,x} + \frac{1}{2} \beta_x^2,\
                           v_{,y} + \frac{w}{r} + \frac{1}{2} \beta_y^2,\
                           u_{,y} + v_{,x} + \beta_x \beta_y \}

    \varepsilon^{(1)} = \{ \phi_{x,x},\
                           \phi_{y,y} + \frac{s}{r} v_{,y},\
                           \phi_{x,y} + \phi_{y,x}
                           + \frac{s}{r} \left(\frac{3}{2} v_{,x}
                           - \frac{1}{2} u_{,y}\right) \}

    \varepsilon^{(3)} = -c_1 \{ \phi_{x,x} + w_{,xx},\
                                \phi_{y,y} + w_{,yy},\
                                \phi_{x,y} + \phi_{y,x} + 2 w_{,xy} \}

    \gamma^{(0)} = \{ \phi_y + w_{,y},\ \phi_x + w_{,x} \} \qquad
    \gamma^{(2)} = -3 c_1 \gamma^{(0)}

The terms in `s` are the corrections of Sanders, which make all the
strains vanish for any small rigid-body motion: the rotation of the
transverse normal about `x` is `\phi_y + v/r`, and the rotation about the
normal, `\frac{1}{2}(v_{,x} - u_{,y})`, enters the twist. With this
correction of the displacement field, the term `-v/r` of the transverse
shear strain `\gamma_{yz}` cancels, and no correction appears in the `z^3`
terms, as in the shell theory of Reddy and Liu [3]_, Eqs. (5), (7) and
(11), specialised to a circular cylinder with the shallow shell assumption
`r + z \approx r`. As for the CLPT, the non-linear terms do not include
the rotation about the normal, a simplification of the Sanders-Koiter
theory [2]_, see :mod:`panels.models`. The transverse shear strain `\gamma^{(0)}(1 - 4 (z - d)^2/h^2)` of the TSDT
vanishes at both faces `z = d \pm h/2`; the cubic term about the
mid-surface keeps `u`, `v` the displacements of the reference surface,
`z = 0`, and `\phi_x`, `\phi_y` the rotations of the normal at the
mid-surface, see :meth:`.Shell._get_lam_ABD`.

References
----------
.. [1] Reddy, J. N., "A simple higher-order theory for laminated composite
   plates", J. Appl. Mech., 51, 745-752, 1984.
.. [2] Sanders, J. L., "Nonlinear theories for thin shells", Quarterly of
   Applied Mathematics, 21(1), 21-36, 1963.
   https://doi.org/10.1090/qam/147023
.. [3] Reddy, J. N., and Liu, C. F., "A higher-order shear deformation
   theory of laminated elastic shells", International Journal of
   Engineering Science, 23(3), 319-330, 1985.
   https://doi.org/10.1016/0020-7225(85)90051-5

"""
import numpy as np
from libc.stdlib cimport malloc, free

cdef extern from 'bardell_functions.hpp':
    double vec_f(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) noexcept nogil
    double vec_fp(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) noexcept nogil
    double vec_fpp(double *f, double xi, double xi1t, double xi1r,
                  double xi2t, double xi2r) noexcept nogil

DOUBLE = np.float64

cdef int NMAX = 30
cdef int DOF = 5
cdef int NFIELDS = 5

FIELDS = ('u', 'v', 'w', 'phix', 'phiy')


def is_tsdt(object s):
    r'''Tell whether the model of ``s`` uses the third-order theory (TSDT)

    The TSDT models have thirteen generalized strains, the FSDT ones eight,
    see :func:`.strain_names`.

    Parameters
    ----------
    s : :class:`.Shell`
        The shell, whose attribute ``model`` is read.

    Returns
    -------
    tsdt : int
        ``1`` when ``'tsdt'`` is in the name of the model, else ``0``.

    '''
    return 1 if 'tsdt' in s.model else 0


def is_sanders(object s):
    r'''Tell whether the model of ``s`` uses Sanders-Koiter kinematics

    For these models the field ``phiy`` of :func:`.fuvw` and :func:`.fg`
    is the rotation of the normal `\phi_y + v/r`.

    Parameters
    ----------
    s : :class:`.Shell`
        The shell, whose attribute ``model`` is read.

    Returns
    -------
    sanders : int
        ``1`` when ``'sanders'`` is in the name of the model, else ``0``.

    '''
    return 1 if ('sanders' in s.model
                 or s.model in ('cylshell_fsdt_gl', 'cylshell_tsdt_gl',
                                'cylshell_fsdt_tl')) else 0


def is_gl(object s):
    r'''Tell whether the model of ``s`` has Green-Lagrange membrane strains

    Parameters
    ----------
    s : :class:`.Shell`
        The shell, whose attribute ``model`` is read.

    Returns
    -------
    gl : int
        ``1`` when the name of the model ends with ``'_gl'`` or the model is
        one of :func:`.is_tl`, else ``0``.

    Notes
    -----
    The membrane strains of these models are the Green-Lagrange strains of
    the mid-surface, `\varepsilon_{xx} = u_{,x} + (u_{,x}^2 + v_{,x}^2 +
    \beta_x^2)/2`, `\varepsilon_{yy} = v_{,y} + w/r + (u_{,y}^2 + (v_{,y} +
    w/r)^2 + \beta_y^2)/2`, `\gamma_{xy} = u_{,y} + v_{,x} + u_{,x} u_{,y}
    + v_{,x} (v_{,y} + w/r) + \beta_x \beta_y`, with `\beta_x = -w_{,x}`,
    `\beta_y = -w_{,y} + v/r` and `1/r = 0` for the plates, see
    :mod:`panels.models`.

    '''
    return 1 if (s.model.endswith('_gl') or is_tl(s) == 1) else 0


def is_tl(object s):
    r'''Tell whether the model of ``s`` is a total-Lagrangian FSDT model

    Parameters
    ----------
    s : :class:`.Shell`
        The shell, whose attribute ``model`` is read.

    Returns
    -------
    tl : int
        ``1`` when the model is ``'plate_fsdt_tl'`` or ``'cylshell_fsdt_tl'``,
        else ``0``.

    Notes
    -----
    The membrane strains of these models are those of :func:`.is_gl`; the
    curvatures and transverse shear strains are those of the second-order
    director `\mathbf{d} = (\psi_x, \psi_y, 1 - (\psi_x^2 + \psi_y^2)/2)`,
    `\psi_x = \phi_x`, `\psi_y = \phi_y + v/r`, in the local basis, with
    the tangent vectors `\mathbf{X}_{,x} = (1 + u_{,x}, v_{,x}, w_{,x})`,
    `\mathbf{X}_{,y} = (u_{,y}, 1 + v_{,y} + w/r, w_{,y} - v/r)` and `1/r =
    0` for the plate:

    .. math::

        \kappa_{xx} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,x} \qquad
        \kappa_{yy} = \mathbf{X}_{,y} \cdot \mathbf{d}_{,y} - \frac{1}{r} -
        \frac{\varepsilon_{yy}}{r}

        \kappa_{xy} = \mathbf{X}_{,x} \cdot \mathbf{d}_{,y} +
        \mathbf{X}_{,y} \cdot \mathbf{d}_{,x} - \frac{\gamma_{xy}}{2 r}
        \qquad \gamma_{yz} = \mathbf{X}_{,y} \cdot \mathbf{d} \qquad
        \gamma_{xz} = \mathbf{X}_{,x} \cdot \mathbf{d}

    with `\mathbf{d}_{,x} = (\psi_{x,x}, \psi_{y,x}, \psi_{z,x})` and
    `\mathbf{d}_{,y} = (\psi_{x,y}, \psi_{y,y} + \psi_z/r, \psi_{z,y} -
    \psi_y/r)`, see :mod:`panels.models.plate_fsdt_tl_num` and
    :mod:`panels.models.cylshell_fsdt_tl_num`.

    '''
    return 1 if s.model in ('plate_fsdt_tl', 'cylshell_fsdt_tl') else 0


def rinv(object s):
    r'''`1/r` for the cylinders and zero for the plates, which may have
    ``r = None``

    A cylinder is a model whose name contains ``'cylshell'``.

    Parameters
    ----------
    s : :class:`.Shell`
        The shell, whose attributes ``model`` and ``r`` are read.

    Returns
    -------
    rinv : float
        ``1/s.r``, per unit length, or ``0.``.

    '''
    return 1./s.r if 'cylshell' in s.model else 0.


def strain_names(object s):
    r'''Names of the generalized strains returned by :func:`.fstrain`

    They are in the order of the rows and columns of the constitutive matrix
    ``Shell.ABD``: the mid-surface strains, the first-order changes of
    curvature and the mid-surface transverse shear strains, plus, for the
    TSDT, the third-order terms ``kxx3, kyy3, kxy3`` of the in-plane strains
    and the second-order terms ``gyz2, gxz2`` of the transverse shear strains.

    Parameters
    ----------
    s : :class:`.Shell`
        The shell, whose attribute ``model`` is read.

    Returns
    -------
    names : tuple of str
        Eight names, or thirteen for the TSDT.

    '''
    if is_tsdt(s) == 1:
        return ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'kxx3', 'kyy3',
                'kxy3', 'gyz', 'gxz', 'gyz2', 'gxz2')
    return ('exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz')


def stress_names(object s):
    r'''Names of the generalized stresses, conjugate to :func:`.strain_names`

    The stress of each position is conjugate to the strain of the same
    position of :func:`.strain_names`, e.g. ``'Qy'`` to ``'gyz'``; the
    names are used as keys of ``Shell.fields`` by :meth:`.Shell.stress`.

    Parameters
    ----------
    s : :class:`.Shell`
        The shell, whose attribute ``model`` is read.

    Returns
    -------
    names : tuple of str
        Eight names, or thirteen for the TSDT.

    Notes
    -----
    The generalized stresses are the products of the constitutive matrix of
    :meth:`.Shell._get_lam_ABD` by the generalized strains, see
    :meth:`.Shell.stress`. With the in-plane stresses `\sigma = \{\sigma_{xx},
    \sigma_{yy}, \tau_{xy}\}` and the transverse shear stresses `\tau =
    \{\tau_{yz}, \tau_{xz}\}` of the displacement field of
    :mod:`panels.models.fsdt_tsdt_field`, they are the resultants

    .. math::

        \{N, M, P\} = \int_h \sigma \{1, z, z^3\} dz \qquad
        \{Q, R\} = \int_h \tau \{1, z^2\} dz

    conjugate to `\varepsilon^{(0)}`, `\varepsilon^{(1)}`,
    `\varepsilon^{(3)}`, `\gamma^{(0)}` and `\gamma^{(2)}`, i.e. the
    internal virtual work per unit area is `N \cdot \delta\varepsilon^{(0)}
    + M \cdot \delta\varepsilon^{(1)} + P \cdot \delta\varepsilon^{(3)} + Q
    \cdot \delta\gamma^{(0)} + R \cdot \delta\gamma^{(2)}`. For the FSDT,
    `Q` is computed with the shear corrected stiffness, see
    :class:`.Shell`.

    '''
    if is_tsdt(s) == 1:
        return ('Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Pxx', 'Pyy', 'Pxy',
                'Qy', 'Qx', 'Ry', 'Rx')
    return ('Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qy', 'Qx')


cdef void read_bcs(object s, double *bcs):
    # bcs[field*8 + 0..3] along x, bcs[field*8 + 4..7] along y
    cdef int k
    for k, field in enumerate(FIELDS):
        bcs[k*8 + 0] = getattr(s, 'x1' + field)
        bcs[k*8 + 1] = getattr(s, 'x1' + field + 'r')
        bcs[k*8 + 2] = getattr(s, 'x2' + field)
        bcs[k*8 + 3] = getattr(s, 'x2' + field + 'r')
        bcs[k*8 + 4] = getattr(s, 'y1' + field)
        bcs[k*8 + 5] = getattr(s, 'y1' + field + 'r')
        bcs[k*8 + 6] = getattr(s, 'y2' + field)
        bcs[k*8 + 7] = getattr(s, 'y2' + field + 'r')


cdef void basis(double *bcs, double xi, double eta, double *f, double *f_xi,
        double *f_xixi, double *g, double *g_eta, double *g_etaeta) noexcept nogil:
    # values and derivatives of the approximation functions of all the fields
    # at (xi, eta), stored as f[field*NMAX + i] and g[field*NMAX + j]
    cdef int k
    cdef double *bc
    for k in range(NFIELDS):
        bc = &bcs[k*8]
        vec_f(&f[k*NMAX], xi, bc[0], bc[1], bc[2], bc[3])
        vec_fp(&f_xi[k*NMAX], xi, bc[0], bc[1], bc[2], bc[3])
        vec_fpp(&f_xixi[k*NMAX], xi, bc[0], bc[1], bc[2], bc[3])
        vec_f(&g[k*NMAX], eta, bc[4], bc[5], bc[6], bc[7])
        vec_fp(&g_eta[k*NMAX], eta, bc[4], bc[5], bc[6], bc[7])
        vec_fpp(&g_etaeta[k*NMAX], eta, bc[4], bc[5], bc[6], bc[7])


cdef void derivatives(double *c, int m, int n, double a, double b,
        double *f, double *f_xi, double *f_xixi, double *g, double *g_eta,
        double *g_etaeta, double *val, double *val_x, double *val_y,
        double *val_xx, double *val_yy, double *val_xy) noexcept nogil:
    # values and physical derivatives of each field at the point
    cdef int i, j, k, col
    cdef double ck
    for k in range(NFIELDS):
        val[k] = 0; val_x[k] = 0; val_y[k] = 0
        val_xx[k] = 0; val_yy[k] = 0; val_xy[k] = 0
    for j in range(n):
        for i in range(m):
            col = DOF*(j*m + i)
            for k in range(NFIELDS):
                ck = c[col+k]
                val[k] += ck*f[k*NMAX+i]*g[k*NMAX+j]
                val_x[k] += ck*(2/a)*f_xi[k*NMAX+i]*g[k*NMAX+j]
                val_y[k] += ck*(2/b)*f[k*NMAX+i]*g_eta[k*NMAX+j]
                val_xx[k] += ck*(2/a)*(2/a)*f_xixi[k*NMAX+i]*g[k*NMAX+j]
                val_yy[k] += ck*(2/b)*(2/b)*f[k*NMAX+i]*g_etaeta[k*NMAX+j]
                val_xy[k] += ck*(2/a)*(2/b)*f_xi[k*NMAX+i]*g_eta[k*NMAX+j]


def fuvw(double [::1] c, object s, double [::1] xs, double [::1] ys,
        int num_cores=4):
    '''
        Calculates the displacement field at all points in the provided grid

        Returns ``u, v, w, phix, phiy``, where ``phix`` and ``phiy`` are the
        rotations of the normal, `\phi_x` and `\phi_y`, and `\phi_y + v/r`
        for the Sanders-Koiter kinematics.

        The points are evaluated serially, one after the other.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the shell ``s``, 1-D array of size
            ``5*s.m*s.n``, see :meth:`.Shell.get_size`.
        s : :class:`.Shell`
            The shell, whose dimensions, numbers of terms, boundary
            condition flags and, for the cylinders, radius are read.
        xs, ys : np.ndarray
            Coordinates `x` and `y` of the points, lengths, 1-D arrays of
            the same size.
        num_cores : int, optional
            Not used. Default ``4``.

        Returns
        -------
        u, v, w, phix, phiy : np.ndarray
            The displacements, lengths, and the rotations, dimensionless,
            1-D arrays of the size of ``xs``.
    '''
    cdef double a, b, sr
    cdef int m, n, pti, k, size
    cdef double bcs[40]
    cdef double [:, ::1] out
    cdef double *f
    cdef double *f_xi
    cdef double *f_xixi
    cdef double *g
    cdef double *g_eta
    cdef double *g_etaeta
    cdef double val[5]
    cdef double val_x[5]
    cdef double val_y[5]
    cdef double val_xx[5]
    cdef double val_yy[5]
    cdef double val_xy[5]
    a = s.a
    b = s.b
    m = s.m
    n = s.n
    #NOTE r is only read for the cylinders, a plate may have r = None
    sr = rinv(s)*is_sanders(s)
    read_bcs(s, bcs)
    size = xs.shape[0]
    out = np.zeros((NFIELDS, size), dtype=DOUBLE)
    f = <double *>malloc(6*NFIELDS*NMAX*sizeof(double))
    f_xi = &f[NFIELDS*NMAX]
    f_xixi = &f[2*NFIELDS*NMAX]
    g = &f[3*NFIELDS*NMAX]
    g_eta = &f[4*NFIELDS*NMAX]
    g_etaeta = &f[5*NFIELDS*NMAX]
    with nogil:
        for pti in range(size):
            basis(bcs, 2*xs[pti]/a - 1., 2*ys[pti]/b - 1., f, f_xi, f_xixi, g,
                  g_eta, g_etaeta)
            derivatives(&c[0], m, n, a, b, f, f_xi, f_xixi, g, g_eta, g_etaeta,
                        val, val_x, val_y, val_xx, val_yy, val_xy)
            for k in range(NFIELDS):
                out[k, pti] = val[k]
            out[4, pti] += sr*val[1]
    free(f)
    return tuple(np.asarray(out[k]) for k in range(NFIELDS))


def fstrain(double [::1] c, object s, double [::1] xs, double [::1] ys,
        int num_cores=4, int NLgeom=0):
    r'''
        Calculates the generalized strains at all points in the provided grid

        The strains are returned in the order given by :func:`.strain_names`.

        The points are evaluated serially, one after the other. For the
        TSDT the thickness ``h`` is the sum of ``s.plyts``.

        Parameters
        ----------
        c : np.ndarray
            Ritz constants of the shell ``s``, 1-D array of size
            ``5*s.m*s.n``, see :meth:`.Shell.get_size`.
        s : :class:`.Shell`
            The shell, whose dimensions, numbers of terms, boundary
            condition flags, ply thicknesses and, for the cylinders, radius
            are read.
        xs, ys : np.ndarray
            Coordinates `x` and `y` of the points, lengths, 1-D arrays of
            the same size.
        num_cores : int, optional
            Not used. Default ``4``.
        NLgeom : int, optional
            ``1`` to include the non-linear terms of the membrane strains,
            ``0`` otherwise. Default ``0``.

        Returns
        -------
        strains : tuple of np.ndarray
            Eight, or thirteen for the TSDT, 1-D arrays of the size of
            ``xs``.

        Notes
        -----
        The strains are those of the notes of
        :mod:`panels.models.fsdt_tsdt_field`, with `c_1 = 4/(3 h^2)` for the
        TSDT, and with the non-linear terms `\beta_x^2/2`, `\beta_y^2/2`
        and `\beta_x \beta_y` multiplied by ``NLgeom``; for the models of
        :func:`.is_gl` the membrane strains are the Green-Lagrange strains
        given there, with their quadratic terms multiplied by ``NLgeom``,
        and for those of :func:`.is_tl` with ``NLgeom=1`` the curvatures and
        transverse shear strains are those given there.
    '''
    cdef double a, b, h, c1, ri, sr, bx, by, gl, ux, uy, vx, vy
    cdef int m, n, pti, size, tsdt
    cdef double bcs[40]
    cdef double [:, ::1] out
    cdef double *f
    cdef double *f_xi
    cdef double *f_xixi
    cdef double *g
    cdef double *g_eta
    cdef double *g_etaeta
    cdef double val[5]
    cdef double val_x[5]
    cdef double val_y[5]
    cdef double val_xx[5]
    cdef double val_yy[5]
    cdef double val_xy[5]
    cdef double gyz, gxz
    cdef int tl
    cdef double px, py, pz, px_x, px_y, py_x, py_y, pz_x, pz_y
    cdef double xx0, xx1, xx2, xy0, xy1, xy2, dy1, dy2
    a = s.a
    b = s.b
    m = s.m
    n = s.n
    tsdt = is_tsdt(s)
    c1 = 0.
    if tsdt == 1:
        h = sum(s.plyts)
        c1 = 4./(3.*h*h)
    #NOTE r is only read for the cylinders, a plate may have r = None
    ri = rinv(s)
    sr = ri*is_sanders(s)
    gl = is_gl(s)
    tl = is_tl(s)
    read_bcs(s, bcs)
    size = xs.shape[0]
    out = np.zeros((13 if tsdt == 1 else 8, size), dtype=DOUBLE)
    f = <double *>malloc(6*NFIELDS*NMAX*sizeof(double))
    f_xi = &f[NFIELDS*NMAX]
    f_xixi = &f[2*NFIELDS*NMAX]
    g = &f[3*NFIELDS*NMAX]
    g_eta = &f[4*NFIELDS*NMAX]
    g_etaeta = &f[5*NFIELDS*NMAX]
    with nogil:
        for pti in range(size):
            basis(bcs, 2*xs[pti]/a - 1., 2*ys[pti]/b - 1., f, f_xi, f_xixi, g,
                  g_eta, g_etaeta)
            derivatives(&c[0], m, n, a, b, f, f_xi, f_xixi, g, g_eta, g_etaeta,
                        val, val_x, val_y, val_xx, val_yy, val_xy)
            # u, v, w, phix, phiy = 0, 1, 2, 3, 4
            # rotations of the von Kármán terms, beta_x = -w,x and
            # beta_y = -w,y + v/r (Sanders)
            bx = -val_x[2]
            by = -val_y[2] + sr*val[1]
            out[0, pti] = val_x[0] + NLgeom*0.5*bx*bx
            out[1, pti] = val_y[1] + ri*val[2] + NLgeom*0.5*by*by
            out[2, pti] = val_y[0] + val_x[1] + NLgeom*bx*by
            # Green-Lagrange membrane strains: quadratic terms of the
            # in-plane gradients u,x, u,y, v,x and v,y + w/r
            ux = val_x[0]
            uy = val_y[0]
            vx = val_x[1]
            vy = val_y[1] + ri*val[2]
            out[0, pti] += gl*NLgeom*0.5*(ux*ux + vx*vx)
            out[1, pti] += gl*NLgeom*0.5*(uy*uy + vy*vy)
            out[2, pti] += gl*NLgeom*(ux*uy + vx*vy)
            out[3, pti] = val_x[3]
            out[4, pti] = val_y[4] + sr*val_y[1]
            out[5, pti] = (val_y[3] + val_x[4]
                           + sr*(1.5*val_x[1] - 0.5*val_y[0]))
            gyz = val[4] + val_y[2]
            gxz = val[3] + val_x[2]
            if tsdt == 1:
                out[6, pti] = -c1*(val_x[3] + val_xx[2])
                out[7, pti] = -c1*(val_y[4] + val_yy[2])
                out[8, pti] = -c1*(val_y[3] + val_x[4] + 2*val_xy[2])
                out[9, pti] = gyz
                out[10, pti] = gxz
                out[11, pti] = -3*c1*gyz
                out[12, pti] = -3*c1*gxz
            else:
                out[6, pti] = gyz
                out[7, pti] = gxz
            if tl == 1 and NLgeom == 1:
                # second-order director d = (px, py, pz) and its derivatives
                px = val[3]
                py = val[4] + ri*val[1]
                px_x = val_x[3]
                px_y = val_y[3]
                py_x = val_x[4] + ri*val_x[1]
                py_y = val_y[4] + ri*val_y[1]
                pz = 1. - 0.5*(px*px + py*py)
                pz_x = -(px*px_x + py*py_x)
                pz_y = -(px*px_y + py*py_y)
                # tangent vectors X,x and X,y, and the components of d,y
                xx0 = 1. + ux
                xx1 = vx
                xx2 = val_x[2]
                xy0 = uy
                xy1 = 1. + vy
                xy2 = val_y[2] - ri*val[1]
                dy1 = py_y + ri*pz
                dy2 = pz_y - ri*py
                out[3, pti] = xx0*px_x + xx1*py_x + xx2*pz_x
                out[4, pti] = (xy0*px_y + xy1*dy1 + xy2*dy2 - ri
                               - ri*out[1, pti])
                out[5, pti] = (xx0*px_y + xx1*dy1 + xx2*dy2 + xy0*px_x
                               + xy1*py_x + xy2*pz_x - 0.5*ri*out[2, pti])
                out[6, pti] = xy0*px + xy1*py + xy2*pz
                out[7, pti] = xx0*px + xx1*py + xx2*pz
    free(f)
    return tuple(np.asarray(out[k]) for k in range(out.shape[0]))


def fg(double[:, ::1] g, double x, double y, object s):
    '''
        Approximation functions of ``u, v, w, phix, phiy`` at ``(x, y)``,
        stored in the rows of ``g``, which must have shape ``(5, size)``. For
        the Sanders-Koiter kinematics the row of ``phiy`` gives the rotation
        of the normal `\phi_y + v/r`

        The functions are written in place: for each term of column ``col
        = 5*(j*m + i)``, ``g[k, col+k]`` is the function of the field ``k``
        of ``u, v, w, phix, phiy`` and ``g[4, col+1]`` the term ``v/r`` of
        the Sanders-Koiter kinematics, zero otherwise. The other entries
        are not written, so ``g`` is usually created with zeros. Nothing is
        returned.

        Parameters
        ----------
        g : np.ndarray
            Output array of shape ``(5, size)``, with ``size`` at least
            ``5*s.m*s.n``, see :meth:`.Shell.get_size`, C-contiguous.
        x, y : float
            Coordinates of the point, lengths.
        s : :class:`.Shell`
            The shell, whose dimensions, numbers of terms, boundary
            condition flags and, for the cylinders, radius are read.

        Raises
        ------
        ValueError
            If the class of ``s`` is not named ``'Shell'``.
    '''
    cdef int m, n, i, j, k, col
    cdef double a, b, sr
    cdef double bcs[40]
    cdef double *f
    cdef double *f_xi
    cdef double *f_xixi
    cdef double *gg
    cdef double *g_eta
    cdef double *g_etaeta
    if s.__class__.__name__ != 'Shell':
        raise ValueError('A Shell object must be passed')
    a = s.a
    b = s.b
    m = s.m
    n = s.n
    sr = rinv(s)*is_sanders(s)
    read_bcs(s, bcs)
    f = <double *>malloc(6*NFIELDS*NMAX*sizeof(double))
    f_xi = &f[NFIELDS*NMAX]
    f_xixi = &f[2*NFIELDS*NMAX]
    gg = &f[3*NFIELDS*NMAX]
    g_eta = &f[4*NFIELDS*NMAX]
    g_etaeta = &f[5*NFIELDS*NMAX]
    basis(bcs, 2*x/a - 1., 2*y/b - 1., f, f_xi, f_xixi, gg, g_eta, g_etaeta)
    for j in range(n):
        for i in range(m):
            col = DOF*(j*m + i)
            for k in range(NFIELDS):
                g[k, col+k] = f[k*NMAX+i]*gg[k*NMAX+j]
            g[4, col+1] = sr*f[1*NMAX+i]*gg[1*NMAX+j]
    free(f)
