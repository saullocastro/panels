r"""Contour plots of the field variables of a :class:`.Shell`

The function :func:`plot_shell` draws a filled contour plot of a
displacement, strain or stress field with :mod:`matplotlib`. On Linux the
non-interactive backend ``'Agg'`` is selected when this module is imported.

"""
import platform
from copy import deepcopy

from numpy import linspace
import matplotlib
if platform.system().lower() == 'linux':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .logger import msg, warn


def plot_shell(shell, c, invert_y=False, vec='w', deform_u=False,
        deform_u_sf=100., filename='', ax=None, figsize=(3.5, 2.),
        save=True, title='', colorbar=False, cbar_nticks=2,
        cbar_format=None, cbar_title='', cbar_fontsize=10, colormap='jet',
        aspect='equal', clean=True, dpi=400, texts=[], xs=None, ys=None,
        gridx=300, gridy=300, num_levels=400, vecmin=None, vecmax=None,
        plot_offset_x=0., plot_offset_y=0., NLgeom=True):
    r"""Contour plot for a Ritz constants vector.

    The field ``vec`` is calculated with :meth:`.Shell.uvw`,
    :meth:`.Shell.strain` or :meth:`.Shell.stress`, which update
    ``shell.fields`` and ``shell.plot_mesh``; afterwards the entries of
    ``shell.fields`` that were not ``None`` before the call are restored,
    the others keep the calculated values. The plot shows `x` along the
    vertical axis and `y` along the horizontal axis, and the horizontal axis
    is inverted. Log messages are printed. When ``save`` is ``True`` the
    figure is written to ``filename`` and closed with
    :func:`matplotlib.pyplot.close`.

    Parameters
    ----------
    shell : :class:`.Shell`
        The shell whose field is plotted.
    c : array_like
        The Ritz constants that will be used to compute the field contour.
    vec : str, optional
        Can be one of the components:

        - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``
        - Strain: ``'exx'``, ``'eyy'``, ``'gxy'``, ``'kxx'``, ``'kyy'``,
          ``'kxy'``, ``'gyz'``, ``'gxz'``
        - Stress: ``'Nxx'``, ``'Nyy'``, ``'Nxy'``, ``'Mxx'``, ``'Myy'``,
          ``'Mxy'``, ``'Qy'``, ``'Qx'``

        Default ``'w'``. The transverse shear components exist only for the
        models based on shear deformation theories.
    deform_u : bool, optional
        If ``True`` the contour plot will look deformed. The plotted `x`
        of each point is decreased by ``deform_u_sf*u`` and the plotted `y`
        increased by ``deform_u_sf*v``. Default ``False``.
    deform_u_sf : float, optional
        The scaling factor used to deform the contour. Dimensionless.
        Default ``100.``.
    invert_y : bool, optional
        Inverts the `y` axis of the plot. Default ``False``.
    save : bool, optional
        Flag telling whether the contour should be saved to an image file.
        Default ``True``, ignored when ``ax`` is given.
    dpi : int, optional
        Resolution of the saved file in dots per inch. Default ``400``.
    filename : str, optional
        The file name for the generated image file. If no value is given,
        ``'test.png'`` is used. Default ``''``.
    ax : AxesSubplot, optional
        When ``ax`` is given, the contour plot will be created inside it.
        Default ``None``, a new figure is created.
    figsize : tuple, optional
        The figure size given by ``(width, height)``, in inches, used only
        for a new figure. Default ``(3.5, 2.)``.
    title : str, optional
        If any string is given it is added as title to the contour plot.
        Default ``''``.
    colorbar : bool, optional
        If a colorbar should be added to the contour plot. Default ``False``.
    cbar_nticks : int, optional
        Number of ticks added to the colorbar. Default ``2``.
    cbar_format : [ None | format string | Formatter object ], optional
        See the ``matplotlib.pyplot.colorbar`` documentation. Default
        ``None``.
    cbar_title : str, optional
        Colorbar title. If ``cbar_title == ''`` no title is added. Default
        ``''``.
    cbar_fontsize : int, optional
        Fontsize of the colorbar labels. Default ``10``.
    colormap : string, optional
        Name of a matplotlib available colormap. An invalid name, or a
        value that is not a string, gives ``'jet'`` with a warning. Default
        ``'jet'``.
    aspect : str, optional
        String that will be passed to the ``AxesSubplot.set_aspect()``
        method. Default ``'equal'``.
    clean : bool, optional
        Clean axes ticks, grids, spines etc. Default ``True``.
    texts : list of dict, optional
        Keyword arguments of :meth:`matplotlib.axes.Axes.text`, one
        dictionary per text, placed in axes coordinates
        (``transform=ax.transAxes``). Default ``[]``.
    xs : array_like, optional
        The `x` positions where to calculate the displacement field.
        Default is ``None`` and the method ``_default_field`` is used.
    ys : array_like, optional
        The ``y`` positions where to calculate the displacement field.
        Default is ``None`` and the method ``_default_field`` is used.
    gridx : int, optional
        Number of points along the `x` axis where to calculate the
        displacement field. Default ``300``.
    gridy : int, optional
        Number of points along the `y` where to calculate the
        displacement field. Default ``300``.
    num_levels : int, optional
        Number of contour levels (higher values make the contour smoother).
        Default ``400``.
    vecmin : float, optional
        Minimum value for the contour scale (useful to compare with other
        results). If not specified it will be taken from the calculated
        field. Default ``None``.
    vecmax : float, optional
        Maximum value for the contour scale. If not specified it will be
        taken from the calculated field. Default ``None``.
    plot_offset_x, plot_offset_y : float, optional
        Offsets added to the coordinates ``x`` and ``y`` of the plotted
        points, lengths. Default ``0.``.
    NLgeom : bool, optional
        Passed to :meth:`.Shell.stress`, used only for the stresses.
        Default ``True``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Matplotlib object that can be used to modify the current plot
        if needed.

    Raises
    ------
    ValueError
        If ``vec`` is not one of the components above, or if ``ax`` is not
        a :class:`matplotlib.axes.Axes`.

    """
    msg('Plotting contour...')

    fields_bkp = deepcopy(shell.fields)

    msg('Computing field variables...', level=1)
    displs = ['u', 'v', 'w', 'phix', 'phiy']
    strains = ['exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz']
    stresses = ['Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qy', 'Qx']
    if vec in displs:
        shell.uvw(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
    elif vec in strains:
        shell.strain(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
    elif vec in stresses:
        shell.stress(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy, NLgeom=NLgeom)
    else:
        raise ValueError(
                '{0} is not a valid vec parameter value!'.format(vec))
    field = shell.fields.get(vec)
    msg('Finished!', level=1)

    Xs = shell.plot_mesh['Xs'] + plot_offset_x
    Ys = shell.plot_mesh['Ys'] + plot_offset_y

    if vecmin is None:
        vecmin = field.min()
    if vecmax is None:
        vecmax = field.max()

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

    x = Ys # in matplotlib x goes vertically (axis=0)
    y = Xs # and y goes horizontally (axis=1)

    if deform_u:
        if vec in displs:
            pass
        else:
            shell.uvw(c, xs=xs, ys=ys, gridx=gridx, gridy=gridy)
        field_u = shell.fields['u']
        field_v = shell.fields['v']
        y -= deform_u_sf*field_u
        x += deform_u_sf*field_v

    if isinstance(colormap, str):
        colormap_obj = getattr(cm, colormap, None)
        if colormap_obj is None:
            warn('Invalid colormap, using "jet"', level=1)
            colormap_obj = cm.jet
    else:
        warn('Invalid colormap (must be a string), using "jet"', level=1)
        colormap_obj = cm.jet

    contour = ax.contourf(x, y, field, levels=levels, cmap=colormap_obj)

    if colorbar:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fsize = cbar_fontsize
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbarticks = linspace(vecmin, vecmax, cbar_nticks)
        cbar = plt.colorbar(contour, ticks=cbarticks, format=cbar_format,
                            cax=cax, cmap=colormap_obj)
        if cbar_title:
            cax.text(0.5, 1.05, cbar_title, horizontalalignment='center',
                     verticalalignment='bottom', fontsize=fsize)
        #cbar.outline.remove()
        cbar.ax.tick_params(labelsize=fsize, pad=0., tick2On=False)

    if invert_y == True:
        ax.invert_yaxis()
    ax.invert_xaxis()

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
            filename = 'test.png'
        fig.savefig(filename, transparent=True,
                    bbox_inches='tight', pad_inches=0.05, dpi=dpi)
        plt.close()

    for k, v in fields_bkp.items():
        if v is not None:
            shell.fields[k] = v

    msg('finished!')

    return ax

