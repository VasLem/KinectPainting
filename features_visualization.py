import logging
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
from matplotlib import pyplot as plt
from __init__ import initialize_logger, timeit, PRIM_X, PRIM_Y, FLNT, find_nonzero
import hist4d as h4d
import class_objects as co
class FeatureVisualization(object):

    def __init__(self, offline_vis=False, n_frames=None,
                 n_saved=4, init_frames=50):
        import matplotlib.pyplot as plt
        self.logger = logging.getLogger(self.__class__.__name__)
        if offline_vis:
            self.n_to_plot = n_frames / n_saved
        else:
            self.n_to_plot = None
        self.offline_vis = offline_vis
        self.n_frames = n_frames
        self.init_frames = init_frames
        gs = gridspec.GridSpec(120, 100)
        initialize_logger(self.logger)
        if not self.offline_vis:
            plt.ion()
            self.fig = plt.figure()
            self.patches3d_plot = self.fig.add_subplot(
                gs[:50, 60:100], projection='3d')
            self.patches2d_plot = self.fig.add_subplot(gs[:50, :50])
            self.hist4d = h4d.Hist4D()
            self.hof_plots = (self.fig.add_subplot(gs[60:100 - 5, :45], projection='3d'),
                              self.fig.add_subplot(gs[60:100 - 5, 45:50]),
                              self.fig.add_subplot(gs[100 - 4:100 - 2, :50]),
                              self.fig.add_subplot(gs[100 - 2:100, :50]))
            self.plotted_hof = False
            self.pause_key = Button(
                self.fig.add_subplot(gs[110:120, 25:75]), 'Next')
            self.pause_key.on_clicked(self.unpause)
            self.hog_plot = self.fig.add_subplot(gs[70:100, 70:100])
            plt.show()
        else:
            self.fig = plt.figure()
            self.hog_fig = plt.figure()
            self.patches3d_fig = plt.figure()
            self.patches2d_fig = plt.figure()
            self.xypca_fig = plt.figure()
            self.patches3d_plot = self.patches2d_fig.add_subplot(111)
            self.patches2d_plot = self.patches3d_fig.add_subplot(111)
            self.xypca_plot = self.xypca_fig.add_subplot(111)
            self.hog_plot = self.hog_fig.add_subplot(111)
            self.hof_plots = (self.fig.add_subplot(gs[60:100 - 5, :45], projection='3d'),
                              self.fig.add_subplot(gs[60:100 - 5, 45:50]),
                              self.fig.add_subplot(gs[100 - 4:100 - 2, :50]),
                              self.fig.add_subplot(gs[100 - 2:100, :50]))
            self.patches3d_fig.tight_layout()
            self.patches2d_fig.tight_layout()
            self.fig.tight_layout()
            self.hog_fig.tight_layout()
            self.xypca_fig.tight_layout()
            self.hogs = []
            self.patches2d = []
            self.patches3d = []
            self.xypca = []
            self.hof = None
        self.curr_frame = None

    def to_plot(self):
        if not self.offline_vis or self.curr_frame is None:
            return True
        return not ((
            self.curr_frame - self.init_frames) % self.n_to_plot)

    def set_curr_frame(self, num):
        self.curr_frame = num

    def plot(self, name, features, edges):
        if 'hog' in name:
            self.plot_hog(features, edges)
        elif 'hof' in name:
            self.plot_hof(features, edges)
        elif 'pca' in name:
            self.plot_xypca(features, edges)

    def plot_hog(self, ghog_features, ghog_edges):
        if self.to_plot():
            hog_hist = ghog_features
            hog_bins = ghog_edges
            width = 0.7 * (hog_bins[0][1] - hog_bins[0][0])
            center = (hog_bins[0][:-1] + hog_bins[0][1:]) / 2
            self.hog_plot.clear()
            self.hog_plot.bar(center, hog_hist, align='center', width=width)

    def plot_hof(self, hof_features, hof_edges):
        if self.to_plot():
            if not self.plotted_hof:
                if self.offline_vis:
                    self.plotted_hof = True
                self.hist4d.draw(
                    hof_features,
                    hof_edges,
                    fig=self.fig,
                    all_axes=self.hof_plots)
                self.hof = self.convert_plot2array(self.fig)

    def plot_xypca(self, pca_features, xticklabels):
        if self.to_plot():
            width = 0.35
            ind = np.arange(len(xticklabels))
            self.xypca_plot.clear()
            self.xypca_plot.set_xticks(ind + width / 2)
            self.xypca_plot.set_xticklabels(xticklabels)
            self.xypca.append(self.convert_plot2array(self.xypca_fig))

    def plot_3d_projection(self, roi, prev_roi_patch, curr_roi_patch):
        if self.to_plot():
            nonzero_mask = (prev_roi_patch * curr_roi_patch) > 0
            yx_coords = (find_nonzero(nonzero_mask.astype(np.uint8)).astype(float) -
                         np.array([[PRIM_Y - roi[0, 0],
                                    PRIM_X - roi[1, 0]]]))
            prev_z_coords = prev_roi_patch[nonzero_mask][:,
                                                         None].astype(float)
            curr_z_coords = curr_roi_patch[nonzero_mask][:,
                                                         None].astype(float)
            prev_yx_proj = yx_coords * prev_z_coords / (FLNT)
            curr_yx_proj = yx_coords * curr_z_coords / (FLNT)
            prev_yx_proj = prev_yx_proj[prev_z_coords.ravel() != 0]
            curr_yx_proj = curr_yx_proj[curr_z_coords.ravel() != 0]
            self.patches3d_plot.clear()
            self.patches3d_plot.scatter(prev_yx_proj[:, 1], prev_yx_proj[:, 0],
                                        prev_z_coords[prev_z_coords != 0],
                                        zdir='z', s=4, c='r', depthshade=False, alpha=0.5)
            self.patches3d_plot.scatter(curr_yx_proj[:, 1], curr_yx_proj[:, 0],
                                        curr_z_coords[curr_z_coords != 0],
                                        zdir='z', s=4, c='g', depthshade=False, alpha=0.5)
            if self.offline_vis:
                self.patches3d.append(
                    self.convert_plot2array(
                        self.patches3d_fig))
            '''
            zprevmin,zprevmax=self.patches3d_plot.get_zlim()
            yprevmin,yprevmax=self.patches3d_plot.get_ylim()
            xprevmin,xprevmax=self.patches3d_plot.get_xlim()
            minlim=min(xprevmin,yprevmin,zprevmin)
            maxlim=max(xprevmax,yprevmax,zprevmax)
            self.patches3d_plot.set_zlim([minlim,maxlim])
            self.patches3d_plot.set_xlim([minlim,maxlim])
            self.patches3d_plot.set_ylim([minlim,maxlim])
            '''

    def convert_plot2array(self, fig):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    def plot_3d_patches(self, roi, prev_roi_patch, curr_roi_patch):
        if self.to_plot:
            self.patches3d_plot.clear()
            x_range = np.arange(roi[0, 0], roi[0, 1])
            y_range = np.arange(roi[1, 0], roi[1, 1])
            xmesh, ymesh = np.meshgrid(y_range, x_range)
            xmesh = xmesh.ravel()
            ymesh = ymesh.ravel()
            curr_vals = curr_roi_patch.ravel()
            self.patches3d_plot.scatter(xmesh[curr_vals > 0],
                                        ymesh[curr_vals > 0],
                                        zs=curr_vals[curr_vals > 0],
                                        zdir='z',
                                        s=4,
                                        c='r',
                                        depthshade=False,
                                        alpha=0.5)
            prev_vals = prev_roi_patch.ravel()
            self.patches3d_plot.scatter(xmesh[prev_vals > 0],
                                        ymesh[prev_vals > 0],
                                        zs=prev_vals[prev_vals > 0],
                                        zdir='z',
                                        s=4,
                                        c='g',
                                        depthshade=False,
                                        alpha=0.5)
            if self.offline_vis:
                self.patches3d.append(
                    self.convert_plot2array(
                        self.patches3d_fig))

    def plot_2d_patches(self, prev_roi_patch, curr_roi_patch):
        self.patches2d_plot.clear()
        self.patches2d_plot.imshow(prev_roi_patch, cmap='Reds', alpha=0.5)
        self.patches2d_plot.imshow(curr_roi_patch, cmap='Greens', alpha=0.5)
        if self.offline_vis:
            self.patches2d.append(self.convert_plot2array(self.patches2d_fig))

    def _draw_single(self, fig):
        import time
        if not self.offline_vis:
            fig.canvas.draw()
            try:
                fig.canvas.start_event_loop(30)
            except BaseException:
                time.sleep(1)

    def draw(self):
        if not self.offline_vis:
            self._draw_single(self.fig)
        else:
            if self.to_plot():
                self._draw_single(self.hog_fig)
                self._draw_single(self.fig)
                self._draw_single(self.patches3d_fig)
                self._draw_single(self.patches2d_fig)

            if self.curr_frame == self.n_frames - 1:
                import pickle
                with open('visualized_features', 'w') as out:
                    pickle.dump((self.hof, self.hogs,
                                 self.patches2d,
                                 self.patches3d), out)
                tmp_fig = plt.figure()
                tmp_axes = tmp_fig.add_subplot(111)
                if self.hogs:
                    hogs_im = co.draw_oper.create_montage(
                        self.hogs, draw_num=False)
                    tmp_axes.imshow(hogs_im[:, :, :3])
                    plt.axis('off')
                    tmp_fig.savefig('ghog.pdf', bbox_inches='tight')
                if self.hof is not None:
                    tmp_axes.imshow(self.hof[:, :, :3])
                    plt.axis('off')
                    tmp_fig.savefig('3dhof.pdf', bbox_inches='tight')
                if self.patches2d:
                    patches2d_im = co.draw_oper.create_montage(
                        self.patches2d, draw_num=False)
                    tmp_axes.imshow(patches2d_im[:, :, :3])
                    plt.axis('off')
                    tmp_fig.savefig('patches2d.pdf', bbox_inches='tight')
                if self.patches3d:
                    patches3d_im = co.draw_oper.create_montage(
                        self.patches3d, draw_num=False)
                    tmp_axes.imshow(patches3d_im[:, :, :3])
                    plt.axis('off')
                    tmp_fig.savefig('patches3d.pdf', bbox_inches='tight')
                if self.xypca:
                    tmp_axes.imshow(self.hof[:, :, :3])
                    plt.axis('off')
                    tmp_fig.savefig('3dxypca.pdf', bbox_inches='tight')

    def unpause(self, val):
        plt.gcf().canvas.stop_event_loop()
