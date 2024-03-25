import platform
import gc

import numpy as np
from numpy import linspace, reshape
from scipy.sparse import csr_matrix

from panels.legendre_gauss_quadrature import get_points_weights
from structsolve.sparseutils import finalize_symmetric_matrix

from panels import Shell
from panels.logger import msg, warn
from panels.shell import DOUBLE, check_c
import panels.modelDB as modelDB

import sys
sys.path.append('C:/Users/natha/Documents/GitHub/panels')
from . import connections


def default_field(panel, gridx, gridy):
    xs = linspace(0, panel.a, gridx)
    ys = linspace(0, panel.b, gridy)
    xs, ys = np.meshgrid(xs, ys, copy=False)
        # Now xs and ys have the x and y coord of all the grid points i.e. size = no_grid_x * no_grid_y
    # Scalar inputs are converted to 1D arrays, whilst higher-dimensional inputs are preserved
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
    
    xcte1 = location (x const) in panel 1 which connects to panel 2
    xcte2 = ........................... 2 ....................... 2
    
    >>> assembly_1 = MultiDomain(panels, conn)

    Notes
    -----

    Concerning the conn dictionary, the key 'func' stands for the connection function that
    builds the compatibility relations between the panels.

    The connections functions available are:
        - 'SSycte' : defines a skin-skin connection for const x and calls the following functions ``fkCSSycte11``, ``fkCSSycte12``, ``fkCSSycte22``
        - 'SSxcte' : defines a skin-skin connection for const y and calls the following functions ``fkCSSxcte11``, ``fkCSSxcte12``, ``fkCSSxcte22``
        - 'SB' : defines a skin-base connection and calls the following functions ``fkCBFycte11``, ``fkCBFycte12``, ``fkCBFycte22``
        - 'BFycte': defines a base-flange connection and calls the following functions ``fkCBFycte11``, ``fkCBFycte12``, ``fkCBFycte22``

    Explanations about the connetion functions are found in ``connections`` module.
    """
    def __init__(self, panels, conn=None):
        # Initialize the assmbly obj with these values
        self.conn = conn
        self.kC_conn = None
        self.panels = panels
        self.size = None
        self.out_num_cores = 4
        row0 = 0
        col0 = 0
        for p in panels:
            assert isinstance(p, Shell) # Check if its Shell obj
            # This actually modifies the shell obj (i.e. passed panels)
            p.row_start = row0 # 1st panel starts at 0,0. Rest start at end of last matrix  --- Assembly of Kp_i from the paper
            p.col_start = col0
            row0 += 3*p.m*p.n # 3 bec 3 dof i.e. u,v,w - For FSDT, change to 5
            col0 += 3*p.m*p.n
            p.row_end = row0 # This is now the start for the next panel to be assembled
            p.col_end = col0


    def get_size(self):
        '''
        Size of K of a single panel
        '''
        self.size = sum([3*p.m*p.n for p in self.panels])
        return self.size


    def plot(self, c='None', group='None', invert_y=False, vec='w', filename='', ax=None,
            figsize=(3.5, 2.), save=True, title='', identify=False,
            show_boundaries=False, boundary_line='--k', boundary_linewidth=1.,
            colorbar=False, cbar_nticks=10, cbar_format=None, cbar_title='',
            cbar_fontsize=7, colormap='jet', aspect='equal', clean=True,
            dpi=400, texts=[], xs=None, ys=None, gridx=50, gridy=50,
            num_levels=400, vecmin=None, vecmax=None,
            no_x_gauss = None, no_y_gauss = None, 
            res = None, silent=True, display_zero = False, flip_plot=False,
            eval_panel=None):
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
        no_x_gauss, no_y_gauss : int
            Number of gauss sampling points along x and y respectively where the displacement field is to be calculated
            Either one of no_x_gauss and no_y_gauss needs to be specified or both
            When specified, uvw_gauss_points, stress_gauss_points and strain_gauss points are called instead
        calc_res_field_only : bool, optional
            Only calcuates the result fields and returns it back
        res : whatever data the stress, strain and uvw functions output, optional
            Contains the results of a previous run. If results already exist, then they're not computed again
        silent : bool, optional
            True = doesnt print any messages
        display_zero : bool, optional
            Decides whether 0 should be displayed as a tick in the colour bar or not
        flip_plot : bool, optional
            Flips plot so that x axis is now horizontal

        Returns
        -------
        ax : matplotlib.axes.Axes
            The Matplotlib object that can be used to modify the current plot
            if needed.
        data : dict
            Data calculated during the plotting procedure.

        """
        msg('Plotting contour...', silent=True)

        import matplotlib.cm as cm
        import matplotlib
        if platform.system().lower() == 'linux':
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # If no results (res) is passed, then compute them
        if res is None:
            res = self.calc_results(c, group, vec = vec, gridx = gridx, gridy = gridy,  
                             no_x_gauss = no_x_gauss, no_y_gauss = no_y_gauss)
            
        field = np.array(res[vec])
        msg('Finished!', level=1, silent=silent)

        if vecmin is None:
            vecmin = field.min()
        if vecmax is None:
            vecmax = field.max()

        data = dict(vecmin=vecmin, vecmax=vecmax)

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

        # Used generally when the field for the entire group is needed
        if eval_panel is None:
            loop_panels = self.panels
        # Used when the field for just a single panel is needed
        if eval_panel is not None:
            loop_panels = [eval_panel]

        count = -1
        
        for i, panel in enumerate(loop_panels):
            if eval_panel is None:
                if panel.group != group:
                    continue
        
        # for i, panel in enumerate(self.panels):
        #     if panel.group != group:
        #         continue
            count += 1
            # x y flipped bec it needs to be plotted
            if flip_plot == False:
                xplot = res['y'][count] + panel.y0
                yplot = res['x'][count] + panel.x0
            if flip_plot == True:
                print('WARNING: flipping of plots still under development !!!!')
                xplot = res['x'][count] + panel.x0
                yplot = res['y'][count] + panel.y0
            field = res[vec][count]
            
            # print(np.shape(xplot), np.shape(yplot), np.shape(field))
            contour = ax.contourf(xplot, yplot, field, levels=levels,
                    cmap=colormap_obj)
            if identify:
                ax.text(xplot.mean(), yplot.mean(), 'P {0:02d}'.format(i+1),
                        transform=ax.transData, ha='center')
            if show_boundaries:
                # Also takes care of gauss point field
                if flip_plot == False:
                    x1_plot = panel.y0
                    x2_plot = panel.y0 + panel.b
                    y1_plot = panel.x0
                    y2_plot = panel.x0 + panel.a
                if flip_plot == True:
                    x1_plot = panel.x0
                    x2_plot = panel.x0 + panel.a
                    y1_plot = panel.y0
                    y2_plot = panel.y0 + panel.b
                # x1, x2 = xplot.min(), xplot.max()
                # y1, y2 = yplot.min(), yplot.max()
                # Plots both points as a line
                ax.plot((x1_plot, x2_plot), (y1_plot, y1_plot), boundary_line, lw=boundary_linewidth)
                ax.plot((x1_plot, x2_plot), (y2_plot, y2_plot), boundary_line, lw=boundary_linewidth)
                ax.plot((x1_plot, x1_plot), (y1_plot, y2_plot), boundary_line, lw=boundary_linewidth)
                ax.plot((x2_plot, x2_plot), (y1_plot, y2_plot), boundary_line, lw=boundary_linewidth)

        if colorbar:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fsize = cbar_fontsize
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='50%', pad=0.05)
                # Change colorbar size by changing size='')
            cbarticks = linspace(vecmin, vecmax, cbar_nticks)
            # Adding 0 to the ticks
            if display_zero :
                cbarticks = np.sort(np.append(cbarticks, np.array([0.])))
            cbar = plt.colorbar(contour, ticks=cbarticks, format=cbar_format,
                                cax=cax)
            if cbar_title:
                cax.text(0.5, 1.05, cbar_title, horizontalalignment='center',
                         verticalalignment='bottom', fontsize=fsize)
            # cbar.outline.remove()
            cbar.ax.tick_params(labelsize=fsize, pad=0.) #, tick2On=False) # Hides the tick marks

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

        msg('finished!', silent=silent)

        return ax, data

    def calc_results(self, c, group=None, vec='w', gridx=50, gridy=50,  
                     no_x_gauss = None, no_y_gauss = None, 
                     eval_panel=None, x_cte_force=None, y_cte_force=None):
        
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
            Variable that needs to be calculated    
            Can be one of the components:

            - Displacement: ``'u'``, ``'v'``, ``'w'``, ``'phix'``, ``'phiy'``
            - Strain: ``'exx'``, ``'eyy'``, ``'gxy'``, ``'kxx'``, ``'kyy'``,
              ``'kxy'``, ``'gyz'``, ``'gxz'``
            - Stress: ``'Nxx'``, ``'Nyy'``, ``'Nxy'``, ``'Mxx'``, ``'Myy'``,
              ``'Mxy'``, ``'Qy'``, ``'Qx'``
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.
        use_gauss_points : bool, optional 
            Uses the gauss integration points (x_gauss and y_gauss) to evaluate the fields
        no_x_gauss, no_y_gauss : Array
            Number of gauss sampling points along x and y respectively where the displacement field is to be calculated
            Either one of no_x_gauss and no_y_gauss needs to be specified or both
            When specified, stress_gauss_points and strain_gauss points are called instead
        eval_panel : Shell object
            Used to evaluate results for a single panel (that is passed) instead of an entire group of panels
        """
        
        displs = ['u', 'v', 'w', 'phix', 'phiy']
        strains = ['exx', 'eyy', 'gxy', 'kxx', 'kyy', 'kxy', 'gyz', 'gxz']
        stresses = ['Nxx', 'Nyy', 'Nxy', 'Mxx', 'Myy', 'Mxy', 'Qy', 'Qx']
        forces = ['Fxx', 'Fyy', 'Fxy']
        
        msg('Computing field variables...', level=1, silent=True)
        
        if no_x_gauss is not None:
            if no_x_gauss > 128:
                raise ValueError('Gauss points more than 128 not coded')
                
        if no_y_gauss is not None:
            if no_y_gauss > 128:
                raise ValueError('Gauss points more than 128 not coded')

        # res = dict of all keywords in that specific input dict
        # size of each variable = no_panel_in_group x grid_pts_Y x grid_pts_X
        if vec in displs:
            res = self.uvw(c, group, gridx=gridx, gridy=gridy, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss,
                           eval_panel=eval_panel)
        elif vec in strains:
            res = self.strain(c, group, gridx=gridx, gridy=gridy, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss,
                              eval_panel=eval_panel)
        elif vec in stresses:
            res = self.stress(c, group, gridx=gridx, gridy=gridy, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss,
                              eval_panel=eval_panel)
        elif vec in forces:
            res = self.force(c, group, eval_panel=eval_panel, x_cte_force=x_cte_force,
                            y_cte_force=y_cte_force, gridx=gridx, gridy=gridy,
                            no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss)
        else:
            raise ValueError(
                    '{0} is not a valid vec parameter value!'.format(vec))
                
        return res

    def uvw(self, c, group, gridx=50, gridy=50, no_x_gauss=None, no_y_gauss=None,
            eval_panel=None):
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
        no_x_gauss, no_y_gauss : int
            Number of gauss sampling points along x and y respectively where the displacement field is to be calculated
            Either one of no_x_gauss and no_y_gauss needs to be specified or both
            Both can be different
        eval_panel : Shell object
            Used to evaluate results for a single panel (that is passed) instead of an entire group of panels

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
        
        # Used generally when the field for the entire group is needed
        if eval_panel is None:
            loop_panels = self.panels
        # Used when the field for just a single panel is needed
        if eval_panel is not None:
            loop_panels = [eval_panel]
        
        for panel in loop_panels:
            if eval_panel is None:
                if panel.group != group:
                    continue

            c_panel = c[panel.col_start: panel.col_end]
            c_panel = np.ascontiguousarray(c_panel, dtype=DOUBLE)
            model = panel.model
            fuvw = modelDB.db[model]['field'].fuvw
                
            if no_x_gauss is not None:
                # Getting the gauss points and weights for x 
                    # Gauss points are between 1 and -1
                x = np.zeros(no_x_gauss, dtype=np.float64)
                x_weights = np.zeros(no_x_gauss, dtype=np.float64)
                get_points_weights(no_x_gauss, x, x_weights)
                # Converting to physical coord as per panel dimensions 
                # Done bec the clpt_bardell_field.pyx ftn converts it to natural coord
                x = (panel.a/2)*(x + 1)
            else:
                x = linspace(0, panel.a, gridx)
            if no_y_gauss is not None:
                # Getting the gauss points and weights for y
                y = np.zeros(no_y_gauss, dtype=np.float64)
                y_weights = np.zeros(no_y_gauss, dtype=np.float64)
                get_points_weights(no_y_gauss, y, y_weights)
                # Converting to physical coord as per panel dimensions
                # Done bec the clpt_bardell_field.pyx ftn converts it to natural coord
                y = (panel.b/2)*(y + 1)
            else:
                y = linspace(0, panel.b, gridy) 
                
            xs, ys = np.meshgrid(x, y, copy=False)
            # Size = size_y x size_x
            # Scalar inputs are converted to 1D arrays, whilst higher-dimensional inputs are preserved
            xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
            ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
            shape = xs.shape # gets the shape of xs which is of the grid
            x = xs.ravel()
            y = ys.ravel()
            
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


    def strain(self, c, group, gridx=50, gridy=50, NLterms=True, no_x_gauss=None, no_y_gauss=None,
               eval_panel=None):
        r"""Calculate the strain field
        Strains and curvatures at each point ???? all x and y or just diagonal terms ????

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        group : str
            A group to plot. Each panel in ``panels`` should contain an
            attribute ``group``, which is used to identify which entities
            should be plotted together.
        
        EACH PANEL HAS gridx x gridy POINTS !!!
        
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.
        NLterms : bool
            Flag to indicate whether non-linear strain components should be considered.
        no_x_gauss, no_y_gauss : int
            Number of gauss sampling points along x and y respectively where the displacement field is to be calculated
            Either one of no_x_gauss and no_y_gauss needs to be specified or both
            Both can be different
        eval_panel : Shell object
            Used to evaluate results for a single panel (that is passed) instead of an entire group of panels

        Returns
        -------
        out : dict
            A dictionary of ``np.ndarrays`` with the keys:
            ``(x, y, exx, eyy, gxy, kxx, kyy, kxy)``.
            Each has the shape: (no_panels_in_group) x gridx x gridy

        """
        
        # res = results
        res = dict(x=[], y=[], exx=[], eyy=[], gxy=[], kxx=[], kyy=[], kxy=[])
        
        if eval_panel is None:
            loop_panels = self.panels
        # Used when the field for just a single panel is needed
        if eval_panel is not None:
            loop_panels = [eval_panel]
        
        for panel in loop_panels:
            if eval_panel is None:
                if panel.group != group:
                    continue
                
            c_panel = c[panel.col_start: panel.col_end]
            c_panel = np.ascontiguousarray(c_panel, dtype=DOUBLE)
            model = panel.model
            # In panels\panels\models - for plates, clpt_bardell_field
            fstrain = modelDB.db[model]['field'].fstrain
            
            # Here x and y are the complete unravelled grid in the x and y coord resp
            # So both have the size = gridx*gridy
            # x and y now have the x and y coord of all the grid points i.e. size = no_grid_x * no_grid_y
            if no_x_gauss is not None:
                # Getting the gauss points and weights for x 
                    # Gauss points are between 1 and -1
                x = np.zeros(no_x_gauss, dtype=np.float64)
                x_weights = np.zeros(no_x_gauss, dtype=np.float64)
                get_points_weights(no_x_gauss, x, x_weights)
                # Converting to physical coord as per panel dimensions 
                # Done bec the clpt_bardell_field.pyx ftn converts it to natural coord
                x = (panel.a/2)*(x + 1)
            else:
                x = linspace(0, panel.a, gridx)
            if no_y_gauss is not None:
                # Getting the gauss points and weights for y
                y = np.zeros(no_y_gauss, dtype=np.float64)
                y_weights = np.zeros(no_y_gauss, dtype=np.float64)
                get_points_weights(no_y_gauss, y, y_weights)
                # Converting to physical coord as per panel dimensions
                # Done bec the clpt_bardell_field.pyx ftn converts it to natural coord
                y = (panel.b/2)*(y + 1)
            else:
                y = linspace(0, panel.b, gridy) 
            
            xs, ys = np.meshgrid(x, y, copy=False)
            # Scalar inputs are converted to 1D arrays, whilst higher-dimensional inputs are preserved
            xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
            ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
            shape = xs.shape # gets the shape of xs which is of the grid
            x = xs.ravel()
            y = ys.ravel()
            
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            
            exx, eyy, gxy, kxx, kyy, kxy = fstrain(c_panel, panel, x, y,
                    self.out_num_cores, NLgeom=int(NLterms))
            res['x'].append(reshape(x, shape))
            res['y'].append(reshape(y, shape))
            res['exx'].append(reshape(exx, shape))
            res['eyy'].append(reshape(eyy, shape))
            res['gxy'].append(reshape(gxy, shape))
            res['kxx'].append(reshape(kxx, shape))
            res['kyy'].append(reshape(kyy, shape))
            res['kxy'].append(reshape(kxy, shape))

        return res


    def stress(self, c, group, gridx=50, gridy=50, NLterms=True, no_x_gauss=None, no_y_gauss=None,
               eval_panel=None, x_cte_force=None, y_cte_force=None):
        r"""Calculate the stress (Nx) field

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
        no_x_gauss, no_y_gauss : int
            Number of gauss sampling points along x and y respectively where the displacement field is to be calculated
            Either one of no_x_gauss and no_y_gauss needs to be specified or both
            Both can be different
        eval_panel : Shell object
            Used to evaluate results for a single panel (that is passed) instead of an entire group of panels

        Returns
        -------
        out : dict
            A dict containing many ``np.ndarrays``, with the keys:
            ``(x, y, Nxx, Nyy, Nxy, Mxx, Myy, Mxy)``.

        """
        res = dict(x=[], y=[], Nxx=[], Nyy=[], Nxy=[], Mxx=[], Myy=[], Mxy=[])
        
        # Used generally when the stress field for the entire group is needed
        if eval_panel is None:
            loop_panels = self.panels
        # Used when the stress field for just a single panel is needed
        if eval_panel is not None:
            loop_panels = [eval_panel]
        
        for panel in loop_panels:
            if eval_panel is None:
                if panel.group != group:
                    continue
            c_panel = c[panel.col_start: panel.col_end]
            c_panel = np.ascontiguousarray(c_panel, dtype=DOUBLE)
            model = panel.model
            fstrain = modelDB.db[model]['field'].fstrain
            
            if no_x_gauss is not None:
                # Getting the gauss points and weights for x
                x = np.zeros(no_x_gauss, dtype=np.float64)
                x_weights = np.zeros(no_x_gauss, dtype=np.float64)
                get_points_weights(no_x_gauss, x, x_weights)
                # Converting to physical coord as per panel dimensions
                x = (panel.a/2)*(x + 1)
            else:
                x = linspace(0, panel.a, gridx)
            if no_y_gauss is not None:
                # Getting the gauss points and weights for y
                y = np.zeros(no_y_gauss, dtype=np.float64)
                y_weights = np.zeros(no_y_gauss, dtype=np.float64)
                get_points_weights(no_y_gauss, y, y_weights)
                # Converting to physical coord as per panel dimensions
                y = (panel.b/2)*(y + 1)
            else:
                y = linspace(0, panel.b, gridy)
                
            # Adding extra points where the force is requested    
            if x_cte_force is not None:
                x = np.sort(np.append(x, np.array([x_cte_force])))
                    # not sorting it so that the last i
            if y_cte_force is not None:
                y = np.sort(np.append(y, np.array([y_cte_force])))
                
            xs, ys = np.meshgrid(x, y, copy=False)
            # Scalar inputs are converted to 1D arrays, whilst higher-dimensional inputs are preserved
            xs = np.atleast_1d(np.array(xs, dtype=DOUBLE))
            ys = np.atleast_1d(np.array(ys, dtype=DOUBLE))
            shape = xs.shape # gets the shape of xs which is of the grid
            x = xs.ravel()
            y = ys.ravel()
            
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            
            exx, eyy, gxy, kxx, kyy, kxy = fstrain(c_panel, panel, x, y,
                    self.out_num_cores, NLgeom=int(NLterms))
            exx = reshape(exx, shape)
            eyy = reshape(eyy, shape)
            gxy = reshape(gxy, shape)
            kxx = reshape(kxx, shape)
            kyy = reshape(kyy, shape)
            kxy = reshape(kxy, shape)
            Ns = np.zeros((exx.shape + (6,)))
            F = panel.ABD
            if F is None:
                raise ValueError('Laminate ABD matrix not defined for panel')
            for i in range(6):
                Ns[..., i] = (exx*F[i, 0] + eyy*F[i, 1] + gxy*F[i, 2]
                            + kxx*F[i, 3] + kyy*F[i, 4] + kxy*F[i, 5])
            
            # For x: Each row goes from 0 to panel.a
            # For y: Each col goes from 0 to panel.b
            res['x'].append(reshape(x, shape))
            res['y'].append(reshape(y, shape))
            res['Nxx'].append(Ns[..., 0])
            res['Nyy'].append(Ns[..., 1])
            res['Nxy'].append(Ns[..., 2])
            res['Mxx'].append(Ns[..., 3])
            res['Myy'].append(Ns[..., 4])
            res['Mxy'].append(Ns[..., 5])
        return res

    def force(self, c, group, eval_panel, x_cte_force=None, y_cte_force=None,
              gridx=50, gridy=50, NLterms=True, no_x_gauss=None, no_y_gauss=None):
        
        """Calculate the force along a line (xcte or ycte)

        Parameters
        ----------
        c : float
            The full set of Ritz constants
        group : str
            Dummy variable in this function - just used to avoid changing the stress 
                function when results for a single panel need to be evaluated
            A group to plot. Each panel in ``panels`` should contain an
                attribute ``group``, which is used to identify which entities
                should be plotted together.
        eval_panel : Shell object
            Used to evaluate results for a single panel (that is passed) instead of an entire group of panels
        x_cte_force or y_cte_force : x or y coordinates of line at which force is to be calc
            Coordinates are relative to specific panel
        gridx : int, optional
            Number of points along the `x` axis where to calculate the
            displacement field.
        gridy : int, optional
            Number of points along the `y` where to calculate the
            displacement field.
        NLterms : bool
            Flag to indicate whether non-linear strain components should be considered.
        no_x_gauss, no_y_gauss : int
            Number of gauss sampling points along x and y respectively where the displacement field is to be calculated
            Either one of no_x_gauss and no_y_gauss needs to be specified or both
            Both can be different

        Returns
        -------
        out : dict
            A dict containing many ``np.ndarrays``, with the keys:
            ``(Fxx, Fyy, Fxy)``. 
        """
    
        
        # Empty dict - keys added later on
        res = dict()
        
        if x_cte_force is None and y_cte_force is None:
            raise ValueError('x_cte or y_cte needs to be specified')
        if x_cte_force is not None and y_cte_force is not None:
            raise ValueError('Integration can only be performed along a single line - specify either x_cte or y_cte')
        if x_cte_force is not None and no_y_gauss is None:
            raise ValueError('Both x_cte_force and no_y_gauss need to be specified')
        if y_cte_force is not None and no_x_gauss is None:
            raise ValueError('Both y_cte_force and no_x_gauss need to be specified')
        if no_x_gauss > 64 or no_y_gauss > 64:
            raise ValueError('Gauss points more than 64 not coded')
            
        res_stress = self.stress(c=c, group=group, 
                                 eval_panel = eval_panel, x_cte_force = x_cte_force, y_cte_force = y_cte_force,
                                 gridx=gridx, gridy=gridy, no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss)
        
        for vec in ['Nxx', 'Nyy', 'Nxy']:
            if x_cte_force is not None:
                [_, col] = np.where(np.isclose(res_stress['x'][0], x_cte_force))
                    # Find which col corresponds to values at x_cte_forces
                if col.max() != col.min():
                    # Check that you've picked the right col no - if its right, all x's should have the same col no
                    raise ValueError('Error - Check force values')
                # Extracting stress field for that particular column
                stress_field = res_stress[vec][0][:, col.min()] # So that only 1 col is taken
                if np.shape(stress_field)[0] != no_y_gauss:
                    raise ValueError('Size mismatch')
                # Getting the gauss points and weights
                y_temp = np.zeros(no_y_gauss, dtype=np.float64)
                weights = np.zeros(no_y_gauss, dtype=np.float64)
                get_points_weights(no_y_gauss, y_temp, weights)
                
            if y_cte_force is not None:
                [row, _] = np.where(np.isclose(res_stress['y'][0], y_cte_force))
                    # Find which row corresponds to values at x_cte_forces
                if row.max() != row.min():
                    # Check that you've picked the right row no
                    raise ValueError('Error - Check force values')
                # Extracting stress field for that particular column
                stress_field = res_stress[vec][0][row.min(), :]
                if np.shape(stress_field)[0] != no_x_gauss:
                    raise ValueError('Size mismatch')
                # Getting the gauss points and weights
                x_temp = np.zeros(no_x_gauss, dtype=np.float64)
                weights = np.zeros(no_x_gauss, dtype=np.float64)
                get_points_weights(no_x_gauss, x_temp, weights)
            
            # Integration
            force_intgn = np.dot(weights, stress_field)
            # Adding keys by modifying keys of res_stress (F added, N removed)
            res[f'F{vec[1:]}'] = force_intgn
    
        return res

    def get_kC_conn(self, conn=None, finalize=True, c=None, kw_tsl=None):
        '''
            Calc the stiffness matrix due to the connectivities
            
            conn = List of dicts 
                Each elem of the list = dict for a single connection pair 
                Each dict contains info of that specific with connection pair
                    Example: conn = [dict(p1=top1, p2=top2, func='SSxcte', xcte1=top1.a, xcte2=0)]
                
                For func, possible options are:
                    'SSxcte' and 'SSycte' = between 2 skins along an edge
                    'BFxcte' and 'BFycte' = between stiffener base and flange
                    'SB'                  = btwn 2 skins connected over an area
                    'SB_TSL'              = btwn 2 skins connected over an area with a TSL introduced 
                                                at the interface
                            Required param:
                                tsl_type, no_x_gauss, no_x_gauss
        '''
        if conn is None:
            if self.conn is None:
                raise RuntimeError('No connectivity dictionary defined!')
            conn = self.conn

        if self.kC_conn is not None:
            return self.kC_conn

        size = self.get_size()

        kC_conn = 0.
        
        # Looping through each connection pair 
        for connecti in conn:
            # connecti = ith connection pair
            p1_temp = connecti['p1']
            p2_temp = connecti['p2']
            # pA and pB are the two panels that are finally passed on
            if p1_temp.col_start < p2_temp.col_start:
                pA = p1_temp
                pB = p2_temp
            elif p1_temp.col_start > p2_temp.col_start:
                pA = p2_temp
                pB = p1_temp
                if connecti['func'] == 'SB' or connecti['func'] == 'SB_TSL':
                    pass
                else:
                    if 'xcte1' in connecti.keys() and 'xcte2' in connecti.keys():
                        temp_xcte = connecti['xcte1']
                        connecti['xcte1'] = connecti['xcte2']
                        connecti['xcte2'] = temp_xcte
                    # y needs to be tested
                    if 'ycte1' in connecti.keys() and 'ycte2' in connecti.keys():
                        temp_ycte = connecti['ycte1']
                        connecti['ycte1'] = connecti['ycte2']
                        connecti['ycte2'] = temp_ycte
            
            connection_function = connecti['func'] # Type of connection
            
            if connection_function == 'SSycte':
                # ftn in panels/multidomain/connections/penalties.py
                kt, kr = connections.calc_kt_kr(pA, pB, 'ycte') 
                
                # ftn in panels/panels/multidomain/connections
                # Eq 32 MD paper - expanding squares gives i^2, ij, ji and j^2 which form 
                #       the 11, 12, 21 and 22 terms of the kC_conn matrix
                # adds the penalty stiffness to ycte of panel pA position
                kC_conn += connections.kCSSycte.fkCSSycte11(
                        kt, kr, pA, connecti['ycte1'],
                        size, row0=pA.row_start, col0=pA.col_start)
                # adds the penalty stiffness to ycte of panel 1 and panel 2 coupling position
                kC_conn += connections.kCSSycte.fkCSSycte12(
                        kt, kr, pA, pB, connecti['ycte1'], connecti['ycte2'],
                        size, row0=pA.row_start, col0=pB.col_start)
                # adds the penalty stiffness to ycte of panel pB position
                kC_conn += connections.kCSSycte.fkCSSycte22(
                        kt, kr, pA, pB, connecti['ycte2'],
                        size, row0=pB.row_start, col0=pB.col_start) 
            
            elif connection_function == 'SSxcte':
                kt, kr = connections.calc_kt_kr(pA, pB, 'xcte')
                # print('kt, kr XCTE', kt, kr)
                # kt = 1*kt
                kr = 2.5*kr
                kk = 1*kt
                # print(f'        Modified kt, kr, kk XCTE : {kt:.1e} {kr:.1e} {kk:.1e}')
    
                kC_conn += connections.kCSSxcte.fkCSSxcte11(
                        kt=kt, kr=kr, kk=kk, p1=pA, xcte1=connecti['xcte1'],
                        size=size, row0=pA.row_start, col0=pA.col_start)
                kC_conn += connections.kCSSxcte.fkCSSxcte12(
                        kt=kt, kr=kr, kk=kk, p1=pA, p2=pB, xcte1=connecti['xcte1'], xcte2=connecti['xcte2'],
                        size=size, row0=pA.row_start, col0=pB.col_start)
                kC_conn += connections.kCSSxcte.fkCSSxcte22(
                        kt=kt, kr=kr, kk=kk, p1=pA, p2=pB, xcte2=connecti['xcte2'],
                        size=size, row0=pB.row_start, col0=pB.col_start)
            
            elif connection_function == 'BFycte':
                kt, kr = connections.calc_kt_kr(pA, pB, 'ycte')
                kC_conn += connections.kCBFycte.fkCBFycte11(
                        kt, kr, pA, connecti['ycte1'],
                        size, row0=pA.row_start, col0=pA.col_start)
                kC_conn += connections.kCBFycte.fkCBFycte12(
                        kt, kr, pA, pB, connecti['ycte1'], connecti['ycte2'],
                        size, row0=pA.row_start, col0=pB.col_start)
                kC_conn += connections.kCBFycte.fkCBFycte22(
                        kt, kr, pA, pB, connecti['ycte2'],
                        size, row0=pB.row_start, col0=pB.col_start)
            
            elif connection_function == 'BFxcte':
                kt, kr = connections.calc_kt_kr(pA, pB, 'xcte')
                kC_conn += connections.kCBFxcte.fkCBFxcte11(
                        kt, kr, pA, connecti['xcte1'],
                        size, row0=pA.row_start, col0=pA.col_start)
                kC_conn += connections.kCBFxcte.fkCBFxcte12(
                        kt, kr, pA, pB, connecti['xcte1'], connecti['xcte2'],
                        size, row0=pA.row_start, col0=pB.col_start)
                kC_conn += connections.kCBFxcte.fkCBFxcte22(
                        kt, kr, pA, pB, connecti['xcte2'],
                        size, row0=pB.row_start, col0=pB.col_start)
                        
            elif connection_function == 'SB': # or (connection_function == 'SB_TSL' and c is None):
                # c is None with SB_TSL implies the inital state of loading so no damage
                # so original SB connection still applies 
                
                kt, kr = connections.calc_kt_kr(pA, pB, 'bot-top')
                # print('kt SB connection : ', kt)
                kt = 1*kt
                # print(f'        Modified kt SB :       {kt:.1e}')
                dsb = sum(pA.plyts)/2. + sum(pB.plyts)/2.
                kC_conn += connections.kCSB.fkCSB11(kt, dsb, pA,
                        size, row0=pA.row_start, col0=pA.col_start)
                kC_conn += connections.kCSB.fkCSB12(kt, dsb, pA, pB,
                        size, row0=pA.row_start, col0=pB.col_start)
                kC_conn += connections.kCSB.fkCSB22(kt, pA, pB,
                        size, row0=pB.row_start, col0=pB.col_start)
            
            # Traction Seperation Law introduced at the interface
            elif connection_function == 'SB_TSL': # and c is not None:
                # Executed when c is not None i.e some displacements have occured so damage 'might' be created
                tsl_type = connecti['tsl_type']
                
                # ATTENTION: pA NEEDS to be the top one and pB, the bottom panel
                no_x_gauss = connecti['no_x_gauss']
                no_y_gauss = connecti['no_y_gauss']

                if kw_tsl is None:
                    kw_tsl, dmg_index, del_d = self.calc_k_dmg(c=c, pA=pA, pB=pB, 
                                             no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, tsl_type=tsl_type)
            
                dsb = sum(pA.plyts)/2. + sum(pB.plyts)/2.
                kC_conn += connections.kCSB_dmg.fkCSB11_dmg(dsb=dsb, p1=pA,
                        size=size, row0=pA.row_start, col0=pA.col_start,
                        no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, kw_tsl=kw_tsl)
                kC_conn += connections.kCSB_dmg.fkCSB12_dmg(dsb=dsb, p1=pA, p2=pB,
                        size=size, row0=pA.row_start, col0=pB.col_start, 
                        no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, kw_tsl=kw_tsl)
                kC_conn += connections.kCSB_dmg.fkCSB22_dmg(p1=pA, p2=pB,
                        size=size, row0=pB.row_start, col0=pB.col_start, 
                        no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss, kw_tsl=kw_tsl)
                
            else:
                raise ValueError(f'{connection_function} not recognized. Provide a correct function if you expect results')

        if finalize:
            kC_conn = finalize_symmetric_matrix(kC_conn)
        self.kC_conn = kC_conn
        #NOTE memory cleanup
        gc.collect()
        return kC_conn


    def calc_kC(self, conn=None, c=None, silent=True, finalize=True, inc=1.):
        """Calculate the constitutive stiffness matrix of the assembly
        --- this is kP (made by diagonally assemblying kP_i from the MD paper)

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
        if c is not None: # If c passed, check if its correct (dim and dtype)
            check_c(c, size)

        kC = 0.
        for p in self.panels:
            if p.row_start is None or p.col_start is None:
                raise ValueError('Shell attributes "row_start" and "col_start" must be defined!')
            # Calc Kc per panel (from the Shell class)
            kC += p.calc_kC(c=c, row0=p.row_start, col0=p.col_start, size=size,
                    silent=silent, finalize=False) 

        # Make the matrix symm at the end
        if finalize:
            kC = finalize_symmetric_matrix(kC)

        # NOTE move this to another class method? it's a bid hidden
        # Adding kC_conn to KC panel
        kC_conn = self.get_kC_conn(conn=conn, c=c)
        
        # print(f'max kC_conn : {np.max(kC_conn):.2e}      kC_wo_conn : {np.max(kC):.2e}')
        
        kC += kC_conn
        # kC += self.kC_conn

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


    def calc_kT(self, c=None, silent=True, finalize=True, inc=None, kw_tsl=None):
        msg('Calculating kT for assembly...', level=2, silent=silent)
        size = self.get_size()
        kT = 0
        #TODO use multiprocessing.Pool here
        for p in self.panels:
            if p.row_start is None or p.col_start is None:
                raise ValueError('Shell attributes "row_start" and "col_start" must be defined!')
            # Both kC and kG are not symmetric now
            kT += p.calc_kC(c=c, size=size, row0=p.row_start, col0=p.col_start,
                    silent=silent, finalize=False, NLgeom=True) # inc=inc,  - REMOVED bec not in shell
            kT += p.calc_kG(c=c, size=size, row0=p.row_start,
                    col0=p.col_start, silent=silent, finalize=False, NLgeom=True)
        if finalize:
            kT = finalize_symmetric_matrix(kT)
        kC_conn = self.get_kC_conn(c=c, kw_tsl=kw_tsl)
        kT += kC_conn
        self.kT = kT
        msg('finished!', level=2, silent=silent)
        return kT


    def calc_fint(self, c, silent=True, inc=1.):
        msg('Calculating internal forces for assembly...', level=2, silent=silent)
        size = self.get_size()
        # fint = 0
        fint = np.zeros_like(c)
        for p in self.panels:
            if p.col_start is None:
                raise ValueError('Shell attributes "col_start" must be defined!')
            fint += p.calc_fint(c=c, size=size, col0=p.col_start, silent=silent)
        kC_conn = self.get_kC_conn(c=c)
        fint += kC_conn*c
        self.fint = fint
        msg('finished!', level=2, silent=silent)
        return fint


    def calc_fext(self, inc=1., silent=True):
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

    def calc_separation(self, res_pan_top, res_pan_bot):
        
        '''
            Calculates the separation between two panels
            
            INPUT ARGUMENTS:
                res_pan_A,B 
                
                !!!!!! PANEL TOP NEEDS TO BE THE ONE ON THE TOP !!!!!!!
        '''

        if not (np.all(res_pan_top['x'][0] == res_pan_bot['x'][0]) and np.all(res_pan_top['y'][0] == res_pan_bot['y'][0])):
            raise ValueError('Grid used to evaluate the displ of both panels dont match')
          
        # [0] bec its a list of results per panel and since theres only 1 panel, its the first one
        del_d = res_pan_top['w'][0] - res_pan_bot['w'][0]
        
        return del_d
    
    def calc_k_dmg(self, c, pA, pB, no_x_gauss, no_y_gauss, tsl_type):
        
        '''
            Calculates the damaged k_tsl and the damage index
        '''
        
        # Calculating the separation between the panels
        res_pan_top = self.calc_results(c=c, eval_panel=pA, vec='w', 
                                no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss)
        res_pan_bot = self.calc_results(c=c, eval_panel=pB, vec='w', 
                                no_x_gauss=no_x_gauss, no_y_gauss=no_y_gauss)
        del_d = self.calc_separation(res_pan_top, res_pan_bot)
        
        # Calculating stiffness grid
        kw_tsl, dmg_index = connections.calc_kw_tsl(pA=pA, pB=pB, tsl_type=tsl_type, del_d=del_d)
        
        return kw_tsl, dmg_index, del_d
