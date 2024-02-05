import platform
import gc

import numpy as np
from numpy import linspace, reshape
from scipy.sparse import csr_matrix

from structsolve.sparseutils import finalize_symmetric_matrix

from panels import Shell
from panels.logger import msg, warn
from panels.shell import DOUBLE, check_c
import panels.modelDB as modelDB

from . import connections


def default_field(panel, gridx, gridy):
    xs = linspace(0, panel.a, gridx)
    ys = linspace(0, panel.b, gridy)
    xs, ys = np.meshgrid(xs, ys, copy=False)
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

    For more details about the theory involved, see
    [castro2017AssemblyModels]_.

    Parameters
    ----------
    panels : iterable
        A list, tuple etc of :class:`.Shell` objects.
    conn : dict
        A connectivity dictionary.

    Example
    -------
    >>> panel_1 = Shell(*args)
    >>> panel_2 = Shell(*args)
    >>> panel_3 = Shell(*args)
    >>> panels = [panel_1, panel_2]
    >>> conn = [
            dict(p1=panel_1, p2=panel_2, func='SSxcte', xcte1=0, xcte2=panel_2.a),
            dict(p1=panel_2, p2=panel_3, func='SSycte', ycte1=0, ycte2=panel_3.b)
            ] # A list of dictionary that indicates two connections: (panel_1-panel_2) and (panel_2-panel_3)
    >>> assembly_1 = MultiDomain(panels, conn)

    Notes
    -----

    Concerning the conn dictionary, the key 'func' stands for the connection function that
    builds the compatibility relations between the panels.

    The connections functions available are:
        - 'SSycte' : defines a skin-skin connection and calls the following functions ``fkCSSycte11``, ``fkCSSycte12``, ``fkCSSycte22``
        - 'SSxcte' : defines a skin-skin connection and calls the following functions ``fkCSSxcte11``, ``fkCSSxcte12``, ``fkCSSxcte22``
        - 'SB' : defines a skin-base connection and calls the following functions ``fkCBFycte11``, ``fkCBFycte12``, ``fkCBFycte22``
        - 'BFycte': defines a base-flange connection and calls the following functions ``fkCBFycte11``, ``fkCBFycte12``, ``fkCBFycte22``

    Explanations about the connetion functions are found in ``connections`` module.
    """
    def __init__(self, panels, conn=None):
        self.conn = conn
        self.kC_conn = None
        self.panels = panels
        self.size = None
        self.out_num_cores = 4
        row0 = 0
        col0 = 0
        for p in panels:
            assert isinstance(p, Shell)
            p.row_start = row0
            p.col_start = col0
            row0 += 3*p.m*p.n
            col0 += 3*p.m*p.n
            p.row_end = row0
            p.col_end = col0


    def get_size(self):
        self.size = sum([3*p.m*p.n for p in self.panels])
        return self.size


    def plot(self, c, group, invert_y=False, vec='w', filename='', ax=None,
            figsize=(3.5, 2.), save=True, title='', identify=False,
            show_boundaries=False, boundary_line='--k', boundary_linewidth=1.,
            colorbar=False, cbar_nticks=2, cbar_format=None, cbar_title='',
            cbar_fontsize=10, colormap='jet', aspect='equal', clean=True,
            dpi=400, texts=[], xs=None, ys=None, gridx=50, gridy=50,
            num_levels=400, vecmin=None, vecmax=None, calc_data_only=False):
        r"""Contour plot for a Ritz constants vector.

        Parameters
        ----------
        c : np.ndarray
            The Ritz constants that will be used to compute the field contour.
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together.
        vec : str, optional
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``
            - Strain: ``'exx'``, ``'eyy'``, ``'gxy'``, ``'kxx'``, ``'kyy'``,
              ``'kxy'``, ``'gyz'``, ``'gxz'``
            - Stress: ``'Nxx'``, ``'Nyy'``, ``'Nxy'``, ``'Mxx'``, ``'Myy'``,
              ``'Mxy'``, ``'Qy'``, ``'Qx'``
        invert_y : bool, optional
            Inverts the `y` axis of the plot.
        save : bool, optional
            Flag telling whether the contour should be saved to an image file.
        dpi : int, optional
            Resolution of the saved file in dots per inch.
        filename : str, optional
            The file name for the generated image file. If no value is given,
            the `name` parameter of the ``Shell`` object will be used.
        ax : AxesSubplot, optional
            When ``ax`` is given, the contour plot will be created inside it.
        figsize : tuple, optional
            The figure size given by ``(width, height)``.
        title : str, optional
            If any string is given a title is added to the contour plot.
        indentify : bool, optional
            If domains should be identified. If yes, the name of each panel is
            used.
        show_boundaries : bool, optional
            If boundaries between domains should be drawn.
        boundary_line : str, optional
            Matplotlib string to define line type and color.
        boundary_linewidth : float, optional
            Matplotlib float to define line width.
        colorbar : bool, optional
            If a colorbar should be added to the contour plot.
        cbar_nticks : int, optional
            Number of ticks added to the colorbar.
        cbar_format : [ None | format string | Formatter object ], optional
            See the ``matplotlib.pyplot.colorbar`` documentation.
        cbar_title : str, optional
            Colorbar title. If ``cbar_title == ''`` no title is added.
        cbar_fontsize : int, optional
            Fontsize of the colorbar labels.
        colormap : string, optional
            Name of a matplotlib available colormap.
        aspect : str, optional
            String that will be passed to the ``AxesSubplot.set_aspect()``
            method.
        clean : bool, optional
            Clean axes ticks, grids, spines etc.
        xs : np.ndarray, optional
            The `x` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        ys : np.ndarray, optional
            The ``y`` positions where to calculate the displacement field.
            Default is ``None`` and the method ``_default_field`` is used.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.
        num_levels : int, optional
            Number of contour levels (higher values make the contour smoother).
        vecmin : float, optional
            Minimum value for the contour scale (useful to compare with other
            results). If not specified it will be taken from the calculated
            field.
        vecmax : float, optional
            Maximum value for the contour scale.
        calc_data_only : bool, optional
            If only calculated data should be returned.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib object that can be used to modify the current plot
            if needed.
        data : dict
            Data calculated during the plotting procedure.

        """
        msg('Plotting contour...')

        import matplotlib.cm as cm
        import matplotlib
        if platform.system().lower() == 'linux':
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        msg('Computing field variables...', level=1)
        displs = ['u', 'v', 'w', 'phix', 'phiy']
        strains = ['exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz']
        stresses = ['Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qy', 'Qx']
        if vec in displs:
            res = self.uvw(c, group, gridx=gridx, gridy=gridy)
        elif vec in strains:
            res = self.strain(c, group, gridx=gridx, gridy=gridy)
        elif vec in stresses:
            res = self.stress(c, group, gridx=gridx, gridy=gridy)
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))
        field = np.array(res[vec])
        msg('Finished!', level=1)

        if vecmin is None:
            vecmin = field.min()
        if vecmax is None:
            vecmax = field.max()

        data = dict(vecmin=vecmin, vecmax=vecmax)

        if calc_data_only:
            return None, data

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

        count = -1
        for i, panel in enumerate(self.panels):
            if panel.group != group:
                continue
            count += 1
            xplot = res['y'][count] + panel.y0
            yplot = res['x'][count] + panel.x0
            field = res[vec][count]
            contour = ax.contourf(xplot, yplot, field, levels=levels,
                    cmap=colormap_obj)
            if identify:
                ax.text(xplot.mean(), yplot.mean(), 'P {0:02d}'.format(i+1),
                        transform=ax.transData, ha='center')
            if show_boundaries:
                x1, x2 = xplot.min(), xplot.max()
                y1, y2 = yplot.min(), yplot.max()
                ax.plot((x1, x2), (y1, y1), boundary_line, lw=boundary_linewidth)
                ax.plot((x1, x2), (y2, y2), boundary_line, lw=boundary_linewidth)
                ax.plot((x1, x1), (y1, y2), boundary_line, lw=boundary_linewidth)
                ax.plot((x2, x2), (y1, y2), boundary_line, lw=boundary_linewidth)

        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fsize = cbar_fontsize
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            cbarticks = linspace(vecmin, vecmax, cbar_nticks)
            cbar = plt.colorbar(contour, ticks=cbarticks, format=cbar_format,
                                cax=cax)
            if cbar_title:
                cax.text(0.5, 1.05, cbar_title, horizontalalignment='center',
                         verticalalignment='bottom', fontsize=fsize)
            cbar.outline.remove()
            cbar.ax.tick_params(labelsize=fsize, pad=0., tick2On=False)

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

        msg('finished!')

        return ax, data


    def uvw(self, c, group, gridx=50, gridy=50):
        r"""Calculate the displacement field

        For a given full set of Ritz constants ``c``, the displacement
        field is calculated and stored in the parameters
        ``u``, ``v``, ``w``, ``phix``, ``phiy`` of the ``Shell`` object.

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.

        Returns
        -------
        out : tuple
            A tuple of ``np.ndarrays`` containing
            ``(xs, ys, u, v, w, phix, phiy)``.

        Notes
        -----
        The returned values ``u```, ``v``, ``w``, ``phix``, ``phiy`` are
        stored as parameters with the same name in the ``Shell`` object.

        """
        res = dict(x=[], y=[], u=[], v=[], w=[], phix=[], phiy=[])
        for panel in self.panels:
            if panel.group != group:
                continue
            c_panel = c[panel.col_start: panel.col_end]
            c_panel = np.ascontiguousarray(c_panel, dtype=DOUBLE)
            model = panel.model
            fuvw = modelDB.db[model]['field'].fuvw
            x, y, shape = default_field(panel, gridx, gridy)
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


    def strain(self, c, group, gridx=50, gridy=50, NLterms=True):
        r"""Calculate the strain field

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.
        NLterms : bool
            Flag to indicate whether non-linear strain components should be considered.

        Returns
        -------
        out : dict
            A dictionary of ``np.ndarrays`` with the keys:
            ``(x, y, exx, eyy, gxy, kxx, kyy, kxy)``.

        """
        res = dict(x=[], y=[], exx=[], eyy=[], gxy=[], kxx=[], kyy=[], kxy=[])
        for panel in self.panels:
            if panel.group != group:
                continue
            c_panel = c[panel.col_start: panel.col_end]
            c_panel = np.ascontiguousarray(c_panel, dtype=DOUBLE)
            model = panel.model
            fstrain = modelDB.db[model]['field'].fstrain
            x, y, shape = default_field(panel, gridx, gridy)
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            exx, eyy, gxy, kxx, kyy, kxy = fstrain(c_panel, panel, x, y,
                    self.out_num_cores, NLterms=int(NLterms))
            res['x'].append(reshape(x, shape))
            res['y'].append(reshape(y, shape))
            res['exx'].append(reshape(exx, shape))
            res['eyy'].append(reshape(eyy, shape))
            res['gxy'].append(reshape(gxy, shape))
            res['kxx'].append(reshape(kxx, shape))
            res['kyy'].append(reshape(kyy, shape))
            res['kxy'].append(reshape(kxy, shape))

        return res


    def stress(self, c, group, gridx=50, gridy=50, NLterms=True):
        r"""Calculate the stress field

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together.
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.
        NLterms : bool
            Flag to indicate whether non-linear strain components should be considered.

        Returns
        -------
        out : dict
            A dict containing many ``np.ndarrays``, with the keys:
            ``(x, y, Nxx, Nyy, Nxy, Mxx, Myy, Mxy)``.

        """
        res = dict(x=[], y=[], Nxx=[], Nyy=[], Nxy=[], Mxx=[], Myy=[], Mxy=[])
        for panel in self.panels:
            if panel.group != group:
                continue
            c_panel = c[panel.col_start: panel.col_end]
            c_panel = np.ascontiguousarray(c_panel, dtype=DOUBLE)
            model = panel.model
            fstrain = modelDB.db[model]['field'].fstrain
            x, y, shape = default_field(panel, gridx, gridy)
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            exx, eyy, gxy, kxx, kyy, kxy = fstrain(c_panel, panel, x, y,
                    self.out_num_cores, NLterms=int(NLterms))
            exx = reshape(exx, shape)
            eyy = reshape(eyy, shape)
            gxy = reshape(gxy, shape)
            kxx = reshape(kxx, shape)
            kyy = reshape(kyy, shape)
            kxy = reshape(kxy, shape)
            Ns = np.zeros((exx.shape + (6,)))
            F = panel.F
            if F is None:
                raise ValueError('Laminate ABD matrix not defined for panel')
            for i in range(6):
                Ns[..., i] = (exx*F[i, 0] + eyy*F[i, 1] + gxy*F[i, 2]
                            + kxx*F[i, 3] + kyy*F[i, 4] + kxy*F[i, 5])
            res['x'].append(reshape(x, shape))
            res['y'].append(reshape(y, shape))
            res['Nxx'].append(Ns[..., 0])
            res['Nyy'].append(Ns[..., 1])
            res['Nxy'].append(Ns[..., 2])
            res['Mxx'].append(Ns[..., 3])
            res['Myy'].append(Ns[..., 4])
            res['Mxy'].append(Ns[..., 5])
        return res


    def get_kC_conn(self, conn=None, finalize=True):
        if conn is None:
            if self.conn is None:
                raise RuntimeError('No connectivity dictionary defined!')
            conn = self.conn

        if self.kC_conn is not None:
            return self.kC_conn

        size = self.get_size()

        kC_conn = 0.
        for connecti in conn:
            p1 = connecti['p1']
            p2 = connecti['p2']
            connection_function = connecti['func']
            if connection_function == 'SSycte':
                kt, kr = connections.calc_kt_kr(p1, p2, 'ycte')
                kC_conn += connections.kCSSycte.fkCSSycte11(
                        kt, kr, p1, connecti['ycte1'],
                        size, p1.row_start, col0=p1.col_start)
                kC_conn += connections.kCSSycte.fkCSSycte12(
                        kt, kr, p1, p2, connecti['ycte1'], connecti['ycte2'],
                        size, p1.row_start, col0=p2.col_start)
                kC_conn += connections.kCSSycte.fkCSSycte22(
                        kt, kr, p1, p2, connecti['ycte2'],
                        size, p2.row_start, col0=p2.col_start)
            elif connection_function == 'SSxcte':
                kt, kr = connections.calc_kt_kr(p1, p2, 'xcte')
                kC_conn += connections.kCSSxcte.fkCSSxcte11(
                        kt, kr, p1, connecti['xcte1'],
                        size, p1.row_start, col0=p1.col_start)
                kC_conn += connections.kCSSxcte.fkCSSxcte12(
                        kt, kr, p1, p2, connecti['xcte1'], connecti['xcte2'],
                        size, p1.row_start, col0=p2.col_start)
                kC_conn += connections.kCSSxcte.fkCSSxcte22(
                        kt, kr, p1, p2, connecti['xcte2'],
                        size, p2.row_start, col0=p2.col_start)
            elif connection_function == 'SB':
                kt, kr = connections.calc_kt_kr(p1, p2, 'bot-top')
                dsb = sum(p1.plyts)/2. + sum(p2.plyts)/2.
                kC_conn += connections.kCSB.fkCSB11(kt, dsb, p1,
                        size, p1.row_start, col0=p1.col_start)
                kC_conn += connections.kCSB.fkCSB12(kt, dsb, p1, p2,
                        size, p1.row_start, col0=p2.col_start)
                kC_conn += connections.kCSB.fkCSB22(kt, p1, p2,
                        size, p2.row_start, col0=p2.col_start)
            elif connection_function == 'BFycte':
                kt, kr = connections.calc_kt_kr(p1, p2, 'ycte')
                kC_conn += connections.kCBFycte.fkCBFycte11(
                        kt, kr, p1, connecti['ycte1'],
                        size, p1.row_start, col0=p1.col_start)
                kC_conn += connections.kCBFycte.fkCBFycte12(
                        kt, kr, p1, p2, connecti['ycte1'], connecti['ycte2'],
                        size, p1.row_start, col0=p2.col_start)
                kC_conn += connections.kCBFycte.fkCBFycte22(
                        kt, kr, p1, p2, connecti['ycte2'],
                        size, p2.row_start, col0=p2.col_start)
            elif connection_function == 'BFxcte':
                kt, kr = connections.calc_kt_kr(p1, p2, 'xcte')
                kC_conn += connections.kCBFxcte.fkCBFxcte11(
                        kt, kr, p1, connecti['xcte1'],
                        size, p1.row_start, col0=p1.col_start)
                kC_conn += connections.kCBFxcte.fkCBFxcte12(
                        kt, kr, p1, p2, connecti['xcte1'], connecti['xcte2'],
                        size, p1.row_start, col0=p2.col_start)
                kC_conn += connections.kCBFxcte.fkCBFxcte22(
                        kt, kr, p1, p2, connecti['xcte2'],
                        size, p2.row_start, col0=p2.col_start)
            elif connection_function == 'kCLTxycte':
                kt, kr = connections.calc_kt_kr(p1, p2, 'xcte-ycte')
                kC_conn += connections.kCLTxycte.fkCBFxcte11(
                        kt, kr, p1, connecti['xcte1'],
                        size, p1.row_start, col0=p1.col_start)
                kC_conn += connections.kCLTxycte.fkCBFxycte12(
                        kt, kr, p1, p2, connecti['xcte1'], connecti['ycte2'],
                        size, p1.row_start, col0=p2.col_start)
                kC_conn += connections.kCLTxycte.fkCBFycte22(
                        kt, kr, p1, p2, connecti['ycte2'],
                        size, p2.row_start, col0=p2.col_start)
            else:
                raise ValueError(f'{connection_function} not recognized.')

        if finalize:
            kC_conn = finalize_symmetric_matrix(kC_conn)
        self.kC_conn = kC_conn
        #NOTE memory cleanup
        gc.collect()
        return kC_conn


    def calc_kC(self, conn=None, c=None, silent=False, finalize=True, inc=1.):
        """Calculate the constitutive stiffness matrix of the assembly

        Parameters
        ----------

        conn : dict, optional
            A connectivity dictionary. Optional if already defined for the
            assembly.
        c : array-like or None, optional
            This must be the result of a static analysis, used to compute
            non-linear terms based on the actual displacement field.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        finalize : bool, optional
            Asserts validity of output data and makes the output matrix
            symmetric, should be ``False`` when assemblying.
        inc : float, optional
            Dummy argument needed for non-linear analyses.

        """
        size = self.get_size()
        msg('Calculating kC for assembly...', level=2, silent=silent)
        if c is not None:
            check_c(c, size)

        kC = 0.
        for p in self.panels:
            if p.row_start is None or p.col_start is None:
                raise ValueError('Shell attributes "row_start" and "col_start" must be defined!')
            kC += p.calc_kC(c=c, row0=p.row_start, col0=p.col_start, size=size,
                    silent=silent, finalize=False)

        if finalize:
            kC = finalize_symmetric_matrix(kC)

        # NOTE move this to another class method? it's a bid hidden
        kC_conn = self.get_kC_conn(conn=conn)
        kC += self.kC_conn

        self.kC = kC
        msg('finished!', level=2, silent=silent)
        return kC


    def calc_kG(self, c=None, silent=False, finalize=True):
        """Calculate the geometric stiffness matrix of the assembly

        Parameters
        ----------

        c : array-like or None, optional
            This must be the result of a static analysis, used to compute
            non-linear terms based on the actual displacement field.
        silent : bool, optional
            A boolean to tell whether the log messages should be printed.
        finalize : bool, optional
            Asserts validity of output data and makes the output matrix
            symmetric, should be ``False`` when assemblying.

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
                            silent=silent, finalize=False)
        if finalize:
            kG = finalize_symmetric_matrix(kG)
        self.kG = kG
        msg('finished!', level=2, silent=silent)
        return kG


    def calc_kM(self, silent=False, finalize=True):
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


    def calc_kT(self, c=None, silent=False, finalize=True, inc=None):
        msg('Calculating kT for assembly...', level=2, silent=silent)
        size = self.get_size()
        kT = 0
        #TODO use multiprocessing.Pool here
        for p in self.panels:
            if p.row_start is None or p.col_start is None:
                raise ValueError('Shell attributes "row_start" and "col_start" must be defined!')
            kT += p.calc_kC(c=c, size=size, row0=p.row_start, col0=p.col_start,
                    silent=silent, finalize=False, inc=inc, NLgeom=True)
            kT += p.calc_kG(c=c, size=size, row0=p.row_start,
                    col0=p.col_start, silent=silent, finalize=False, NLgeom=True)
        if finalize:
            kT = finalize_symmetric_matrix(kT)
        kC_conn = self.get_kC_conn()
        kT += kC_conn
        self.kT = kT
        msg('finished!', level=2, silent=silent)
        return kT


    def calc_fint(self, c, silent=False, inc=1.):
        msg('Calculating internal forces for assembly...', level=2, silent=silent)
        size = self.get_size()
        fint = 0
        for p in self.panels:
            if p.col_start is None:
                raise ValueError('Shell attributes "col_start" must be defined!')
            fint += p.calc_fint(c=c, size=size, col0=p.col_start, silent=silent)
        kC_conn = self.get_kC_conn()
        fint += kC_conn*c
        self.fint = fint
        msg('finished!', level=2, silent=silent)
        return fint


    def calc_fext(self, inc=1., silent=False):
        msg('Calculating external forces for assembly...', level=2, silent=silent)
        size = self.get_size()
        fext = 0
        for p in self.panels:
            if p.col_start is None:
                raise ValueError('Shell attributes "col_start" must be defined!')
            fext += p.calc_fext(inc=inc, size=size, col0=p.col_start,
                                silent=silent)
        self.fext = fext
        msg('finished!', level=2, silent=silent)
        return fext
