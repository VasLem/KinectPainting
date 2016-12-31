from matplotlib import cm
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import operator


class DiscreteSlider(Slider):
    '''
    A matplotlib slider widget with discrete steps.
    '''
    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 1)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this 
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: 
            return
        for cid, func in self.observers.iteritems():
            func(discrete_val)

class Hist4D(object):
    def __init__(self):
        self.fig=None
        self.cubes_info=None
        self.slow=None
        self.shigh=None
        self.colormap=None
    def draw_cubes(self,_axes, vals, edges):
        '''
        ax=Axes3D handle
        edges=matrix L+1xM+1xN+1 result of histogramdd
        vals=matrix LxMxN result of histogramdd
        colormap=color map to be matched with nonzero vals
        '''
        edx, edy, edz = np.meshgrid(edges[0], edges[1], edges[2])
        edx_rolled = np.roll(edx, -1, axis=1)
        edy_rolled = np.roll(edy, -1, axis=0)
        edz_rolled = np.roll(edz, -1, axis=2)
        edx_rolled = edx_rolled[:-1, :-1, :-1].ravel()
        edy_rolled = edy_rolled[:-1, :-1, :-1].ravel()
        edz_rolled = edz_rolled[:-1, :-1, :-1].ravel()
        edx = edx[:-1, :-1, :-1].ravel()
        edy = edy[:-1, :-1, :-1].ravel()
        edz = edz[:-1, :-1, :-1].ravel()
        vals = vals.ravel()
        vdraw_cube = np.vectorize(self.draw_cube, excluded='axes')
        cubes_handles = vdraw_cube(_axes, edx[vals>0],
                                   edx_rolled[vals>0],
                                   edy[vals>0],
                                   edy_rolled[vals>0],
                                   edz[vals > 0],
                                   edz_rolled[vals > 0],
                                   vals[vals>0]/float(np.max(vals)))
        cubes_data = [a for a in zip(vals[vals>0],cubes_handles)]
        self.cubes_info=dict()
        for k, v in cubes_data:
            self.cubes_info[k] = self.cubes_info.get(k, ()) + tuple(v)
    def set_sliders(self,splot1,splot2):
        maxlim=max(self.cubes_info.keys())
        axcolor = 'lightgoldenrodyellow'
        #low_vis = self.fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        #high_vis  = self.fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        self.slow = Slider(splot1,'low', 0.0, maxlim, valfmt='%0.0f')
        self.shigh = Slider(splot2, 'high', 0.0 , maxlim, valfmt='%0.0f')
        
        self.slow.on_changed(self.update)
        self.shigh.on_changed(self.update)
        self.slow.set_val(0)
        self.shigh.set_val(maxlim)
    def update(self,val):
        visible = [v for k, v in self.cubes_info.items() if k >
                   self.slow.val and k<=
                   self.shigh.val]
        invisible = [v for k, v in self.cubes_info.items() if k <=
                     self.slow.val or k>
                     self.shigh.val]
        print len(visible),len(invisible)
        for item in [item for sublist in visible for item in sublist]:
            item.set_alpha(1)
        for item in [item for sublist in invisible for item in sublist]:
            item.set_alpha(0)
        total=[v for k,v in self.cubes_info.items()]
        self.fig.canvas.draw_idle()

    def draw_cube(self,_axes, x1_coord, x2_coord,
                  y1_coord, y2_coord,
                  z1_coord, z2_coord,
                  color_ind):
        '''
        draw a cube given cube limits and color
        '''
        _x_coord, _y_coord, _z_coord = np.meshgrid([x1_coord, x2_coord],
                                                   [y1_coord, y2_coord],
                                                   [z1_coord, z2_coord])
        tmp1 = np.concatenate((_x_coord.ravel()[None, :], _y_coord.ravel()[
            None, :], _z_coord.ravel()[None, :]), axis=0)
        tmp2 = tmp1.copy()
        tmp2[:, [0, 1]], tmp2[:, [6, 7]] = tmp2[
            :, [6, 7]].copy(), tmp2[:, [0, 1]].copy()
        tmp3 = tmp2.copy()
        tmp3[:, [0, 2]], tmp3[:, [5, 7]] = tmp3[
            :, [5, 7]].copy(), tmp3[:, [0, 2]].copy()
        points = np.concatenate((tmp1, tmp2, tmp3), axis=1)
        points = points.T.reshape(6, 4, 3)
        surf = []
        for count in range(6):
            surf.append(_axes.plot_surface(points[count, :, 0].reshape(2, 2),
                                          points[count, :, 1].reshape(2, 2),
                                          points[count, :, 2].reshape(2, 2),
                                          color=self.colormap(float(color_ind)),
                                          linewidth=0,
                                          alpha=0,
                                          antialiased=True,
                                          shade=False))
        return surf

    def array2cmap(self,X):
        N = X.shape[0]
        r = np.linspace(0., 1., N+1)
        r = np.sort(np.concatenate((r, r)))[1:-1]
        rd = np.concatenate([[X[i, 0], X[i, 0]] for i in xrange(N)])
        gr = np.concatenate([[X[i, 1], X[i, 1]] for i in xrange(N)])
        bl = np.concatenate([[X[i, 2], X[i, 2]] for i in xrange(N)])
        rd = tuple([(r[i], rd[i], rd[i]) for i in xrange(2 * N)])
        gr = tuple([(r[i], gr[i], gr[i]) for i in xrange(2 * N)])
        bl = tuple([(r[i], bl[i], bl[i]) for i in xrange(2 * N)])
        cdict = {'red': rd, 'green': gr, 'blue': bl}
        return colors.LinearSegmentedColormap('my_colormap', cdict, N)

    def draw_colorbar(self,_axes,unique_vals,cax):
        xmin, xmax = _axes.get_xlim()
        ymin, ymax = _axes.get_ylim()
        zmin, zmax = _axes.get_zlim()
        invis=_axes.scatter(unique_vals,
                           unique_vals,
                           c=np.arange(unique_vals.size),
                           cmap=self.colormap,alpha=1)
        _axes.set_xlim([xmin,xmax])
        _axes.set_ylim([ymin,ymax])
        _axes.set_zlim([zmin,zmax])
        cbar=self.fig.colorbar(invis,ax=_axes,cax=cax)
        cbar.set_ticks(np.linspace(0,np.size(unique_vals),5))
        cbar.set_ticklabels(np.around(np.linspace(0,np.max(unique_vals),5),2))

        invis.set_alpha(0)
    def create_brightness_colormap(self,principal_rgb_color, scale_size):
        '''
        Create brightness colormap based on one principal RGB color
        '''
        if np.any(principal_rgb_color > 1):
            raise Exception('principal_rgb_color values should  be in range [0,1]')
        hsv_color = colors.rgb_to_hsv(principal_rgb_color)
        hsv_colormap = np.concatenate((np.tile(hsv_color[:-1][None, :], (scale_size, 1))[:],
                                       np.linspace(0, 1, scale_size)[:, None]),
                                      axis=1)
        self.colormap=self.array2cmap(colors.hsv_to_rgb(hsv_colormap))

    def draw(self,hist,edges,
             fig=None,gs=None,subplot=None,
             color=np.array([1,0,0]),all_axes=None):
        '''
        fig=figure handle
        gs= contiguous slice (or whole) of gridspec to host plot
        hist,edges=histogramdd output

        '''
        if fig is not None:
            self.fig=fig
        else:
            self.fig=plt.figure()

        if gs is None:
            gs = gridspec.GridSpec(50, 50)
        if all_axes is None:
            _axes = self.fig.add_subplot(gs[:-5,:45],projection='3d')
            cax=self.fig.add_subplot(gs[:-5,45:])
            ax1=self.fig.add_subplot(gs[-4:-2,:])
            ax2=self.fig.add_subplot(gs[-2:,:])
        else:
            _axes,cax,ax1,ax2=all_axes
            _axes.clear()
            cax.clear()
            ax1.clear()
            ax2.clear()
        unique_hist=np.unique(hist)
        colormap = self.create_brightness_colormap(color,
                                                 np.size(unique_hist))
        
        
        self.draw_cubes(_axes, hist, edges)
        
        self.draw_colorbar(_axes,unique_hist,cax)
        _axes.set_xlim((edges[0].min(),edges[0].max()))
        _axes.set_ylim((edges[1].min(),edges[1].max()))
        _axes.set_zlim((edges[2].min(),edges[2].max()))
        self.fig.patch.set_facecolor('white')
        _axes.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
        _axes.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
        _axes.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
        self.set_sliders(ax1, ax2)
        return _axes,ax1,ax2



def main():
    '''
    example caller function
    '''
    data = np.random.randn(100, 3)
    hist, edges = np.histogramdd(data, bins=(5, 8, 4))
    fig = plt.figure()
    hist4d=Hist4D()
    hist4d.draw(hist,edges,fig)
    plt.show()

if __name__ == '__main__':
    main()
