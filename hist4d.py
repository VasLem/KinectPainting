from matplotlib import cm
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import operator

class Hist4D(object):
    def __init__(self):
        self.fig=None
        self.cubes_info=None
        self.slow=None
        self.shigh=None
    def draw_cubes(self,_axes, vals, edges, colormap):
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
        colormap_scale = colormap.shape[0]
        tmp = np.arange(colormap_scale-1) / float(colormap_scale - 1)
        bins = (1 - tmp) * np.min(vals) + tmp * np.max(vals)
        digitized_vals = np.digitize(vals, bins)
        valid_vals = colormap[digitized_vals, ...][vals > 0]
        vdraw_cube = np.vectorize(self.draw_cube, excluded='axes')
        cubes_handles = vdraw_cube(_axes, edx[vals>0],
                                   edx_rolled[vals>0],
                                   edy[vals>0],
                                   edy_rolled[vals>0],
                                   edz[vals > 0],
                                   edz_rolled[vals > 0],
                                   valid_vals[:, 0], valid_vals[:, 1],
                                   valid_vals[:, 2])
        cubes_data = [a for a in zip(vals[vals>0],cubes_handles)]
        self.cubes_info=dict()
        for k, v in cubes_data:
            self.cubes_info[k] = self.cubes_info.get(k, ()) + tuple(v)
    def set_sliders(self,splot1,splot2):
        maxlim=max(self.cubes_info.keys())
        axcolor = 'lightgoldenrodyellow'
        #low_vis = self.fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
        #high_vis  = self.fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)
        self.slow = Slider(splot1,'low', 0, maxlim, valinit=0)
        self.shigh = Slider(splot2, 'high', 0 , maxlim, valinit=maxlim)
        self.slow.on_changed(self.update)
        self.shigh.on_changed(self.update)
    def update(self,val):
        visible = [v for k, v in self.cubes_info.items() if k >
                   np.floor(self.slow.val) and k<
                   np.ceil(self.shigh.val)]
        invisible = [v for k, v in self.cubes_info.items() if k <=
                     np.floor(self.slow.val) or k>=
                     np.ceil(self.shigh.val)]
        for item in [item for sublist in visible for item in sublist]:
            item.set_alpha(1)
        for item in [item for sublist in invisible for item in sublist]:
            item.set_alpha(0)
        self.fig.canvas.draw_idle()

    def draw_cube(self,_axes, x1_coord, x2_coord,
                  y1_coord, y2_coord,
                  z1_coord, z2_coord,
                  colorx, colory, colorz):
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
        for count in range(0, 6):
            surf.append(_axes.plot_surface(points[count, :, 0].reshape(2, 2),
                                          points[count, :, 1].reshape(2, 2),
                                          points[count, :, 2].reshape(2, 2),
                                          rstride=1,
                                          cstride=1,
                                          color=[colorx, colory, colorz],
                                          linewidth=0,
                                          antialiased=False))
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

    def draw_colorbar(self,_axes,colormap,minval,maxval):
        xmin, xmax = _axes.get_xlim()
        ymin, ymax = _axes.get_ylim()
        zmin, zmax = _axes.get_zlim()
        invis=_axes.scatter(np.arange(minval,maxval+1),
                           np.arange(minval,maxval+1),
                           c=np.arange(minval,maxval+1),
                           cmap=self.array2cmap(colormap),alpha=1)
        _axes.set_xlim([xmin,xmax])
        _axes.set_ylim([ymin,ymax])
        _axes.set_zlim([zmin,zmax])
        self.fig.colorbar(invis)
        invis.set_alpha(0)
    def create_brightness_colormap(self,principal_rgb_color, scale_size):
        '''
        Create brightness colormap based on one principal RGB color
        '''
        norm=plt.Normalize()
        if np.any(principal_rgb_color > 1):
            raise Exception('principal_rgb_color values should  be in range [0,1]')
        from matplotlib import colors
        hsv_color = colors.rgb_to_hsv(principal_rgb_color)
        hsv_colormap = np.concatenate((np.tile(hsv_color[:-1][None, :], (scale_size, 1))[:],
                                       np.arange(0.5, 1, 1 / float(2 * scale_size))[:, None]),
                                      axis=1)
        return colors.hsv_to_rgb(hsv_colormap)

    def draw_cubes_plot(self,fig,colormap,hist,edges):
        self.fig=fig
        gs = gridspec.GridSpec(50, 50)
        _axes = self.fig.add_subplot(gs[:-5,:],projection='3d')
        self.draw_cubes(_axes, hist, edges, colormap)
        self.draw_colorbar(_axes,colormap,np.min(hist),np.max(hist))
        ax1=self.fig.add_subplot(gs[-4:-2,:])
        ax2=self.fig.add_subplot(gs[-2:,:])
        self.set_sliders(ax1, ax2)
def main():
    '''
    example caller function
    '''
    hist4d=Hist4D()
    colormap = hist4d.create_brightness_colormap(np.array([1,0,0]), 256)
    data = np.random.randn(100, 3)
    hist, edges = np.histogramdd(data, bins=(5, 8, 4))
    fig = plt.figure()
    hist4d.draw_cubes_plot(fig,colormap,hist,edges)
    plt.show()

if __name__ == '__main__':
    main()
