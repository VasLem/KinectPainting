
import cv2
import numpy as np
import class_objects as co
from __init__ import initialize_logger, timeit, PRIM_X, PRIM_Y, FLNT, find_nonzero
from math import pi

class _Descriptor(object):
    '''
    <parameters>: dictionary with parameters
    <datastreamer> : FramesPreprocessing Class
    <viewer>: FeatureVisualization Class
    '''


    def __init__(self, parameters, datastreamer, viewer=None,
                 reset_time=True):
        self.name = ''
        self.features = None
        self.roi = None
        self.roi_original = None
        self.parameters = parameters
        self.plots = None
        self.edges = None
        self.ds = datastreamer
        initialize_logger(self)
        self.action_type = parameters['action_type']
        if reset_time:
            self.time = []
        self.view = viewer


    def reset(self, visualize=False, reset_time=False):
        self.__init__(self.parameters, self.ds, visualize,
                      reset_time=reset_time)

    def draw_flow(self, img, flow, step=16):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2:h:step, step /
                        2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def draw_hsv(self, flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:, :, 0], flow[:, :, 1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx * fx + fy * fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = 255
        hsv[..., 2] = np.minimum(v * 4, 255)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    def convert_to_uint8(self, patch, _min=None, _max=None):
        # We keep 0 value to denote pixels outside of mask and use the rest band
        # for the other depth values
        uint8 = np.zeros(patch.shape, np.uint8)
        nnz_pixels_mask = patch > 0
        nnz_pixels = patch[patch > 0]
        uint8[nnz_pixels_mask] = ((nnz_pixels - _min) / float(
            _max - _min) * 254 + 1).astype(np.uint8)
        return uint8

    def visualize_projection(self):
        self.view.plot_3d_projection(self.roi,
                                     self.ds.prev_roi_patch,
                                     self.ds.curr_roi_patch)

    def visualize_roi(self):
        self.view.plot_2d_patches(self.ds.prev_roi_patch,
                                  self.ds.curr_roi_patch)

    def visualize(self):
        self.view.plot(self.name.lower(), self.features, self.edges)

    def draw(self):
        self.view.draw()

    def plot(self):
        self.view.plot()

    def set_curr_frame(self, frame):
        self.view.set_curr_frame(frame)

    def find_outliers(self, data, m=2.):
        '''
        median of data must not be 0
        '''
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev > 0 else 0
        return s > m

    def find_roi(self, prev_patch, curr_patch, prev_patch_pos, curr_patch_pos):
        '''
        Find unified ROI, concerning 2 consecutive frames
        '''
        if prev_patch is None:
            prev_patch = curr_patch
            prev_patch_pos = curr_patch_pos
        roi = np.array([[
            min(prev_patch_pos[0], curr_patch_pos[0]),
            max((prev_patch.shape[0] + prev_patch_pos[0],
                 curr_patch.shape[0] + curr_patch_pos[0]))],
            [min(prev_patch_pos[1], curr_patch_pos[1]),
             max(prev_patch.shape[1] + prev_patch_pos[1],
                 curr_patch.shape[1] + curr_patch_pos[1])]])
        return roi

    def extract(self):
        pass



class _Property(object):
    '''
    Base class for required frame properties
    <datastreamer> : FramesPreprocessing Class
    '''
    def __init__(self, datastreamer):
        self.ds = datastreamer
        self.time = []

    def extract(self):
        pass

    def reset(self):
        pass

class VAR(_Property):
    '''
    Depth variance absolute difference of current patch with previous
    '''

    @timeit
    def extract(self):
        if self.ds.prev_patch is None or self.ds.curr_patch is None:
            return [None]
        return [np.abs(np.var(self.ds.curr_patch) - np.var(self.ds.prev_patch))]

class MEDIAN(_Property):
    '''
    Median Depth of Current Patch
    '''

    @timeit
    def extract(self):
        if self.ds.curr_path is None:
            return [None]
        return [np.median(self.ds.curr_patch)]

class SpaceHistogram(object):
    '''
    Create Histograms for 3DHOG and GHOF
    '''

    def __init__(self):
        self.bin_size = None
        self.range = None

    def hist_data(self, sample):
        '''
        Compute normalized N-D histograms
        '''
        hist, edges = np.histogramdd(sample, self.bin_size, range=self.range)
        return hist, edges

class TDHOF(_Descriptor):

    def __init__(self, *args, **kwargs):
        _Descriptor.__init__(self, *args, **kwargs)
        self.name = '3dhof'
        self.bin_size = co.CONST['3DHOF_bin_size']
        self.hist = SpaceHistogram()


    def compute_scene_flow(self):
        '''
        Computes scene flow for 3DHOF
        '''
        if self.ds.prev_depth_im is None or self.ds.curr_depth_im is None:
            return None
        roi = self.roi
        prev_depth_im = self.ds.prev_depth_im
        curr_depth_im = self.ds.curr_depth_im
        self.prev_roi_patch = prev_depth_im[roi[0, 0]:roi[0, 1],
                                            roi[1, 0]:roi[1, 1]].astype(float)
        self.curr_roi_patch = curr_depth_im[roi[0, 0]:roi[0, 1],
                                            roi[1, 0]:roi[1, 1]].astype(float)
        curr_z = self.curr_roi_patch
        prev_z = self.prev_roi_patch
        # DEBUGGING
        # cv2.imshow('curr_roi_patch',(self.curr_roi_patch_original).astype(np.uint8))
        # cv2.waitKey(10)
        prev_nnz_mask = self.prev_roi_patch > 0
        curr_nnz_mask = self.curr_roi_patch > 0
        nonzero_mask = prev_nnz_mask * curr_nnz_mask
        if np.sum(nonzero_mask) == 0:
            return None
        _max = max(np.max(self.prev_roi_patch[prev_nnz_mask]),
                   np.max(self.curr_roi_patch[curr_nnz_mask]))
        _min = min(np.min(self.prev_roi_patch[prev_nnz_mask]),
                   np.min(self.curr_roi_patch[curr_nnz_mask]))
        prev_uint8 = self.convert_to_uint8(self.prev_roi_patch,
                                           _min=_min, _max=_max)
        curr_uint8 = self.convert_to_uint8(self.curr_roi_patch,
                                           _min=_min, _max=_max)
        flow = cv2.calcOpticalFlowFarneback(prev_uint8,
                                            curr_uint8, None,
                                            0.3, 3, 40,
                                            3, 7, 1.5, 0)
        # DEBUGGING
        '''
        cv2.imshow('flow HSV', self.draw_hsv(flow))
        cv2.imshow('prev',  prev_uint8)
        cv2.imshow('flow', self.draw_flow(curr_uint8, flow, step = 14))
        cv2.waitKey(500)
        '''
        y_old, x_old = np.mgrid[:self.prev_roi_patch.shape[0],
                                :self.prev_roi_patch.shape[1]].reshape(
                                    2, -1).astype(int)
        mask = prev_z[y_old, x_old] > 0
        y_old = y_old[mask.ravel()]
        x_old = x_old[mask.ravel()]
        fx, fy = flow[y_old, x_old].T
        y_new, x_new = ((y_old + fy).astype(int), (x_old + fx).astype(int))
        y_new = np.minimum(curr_z.shape[0] - 1, y_new)
        y_new = np.maximum(0, y_new)
        x_new = np.minimum(curr_z.shape[1] - 1, x_new)
        x_new = np.maximum(0, x_new)
        mask = (self.find_outliers(curr_z[y_new, x_new], 5)
                + self.find_outliers(prev_z[y_old, x_old], 5)) == 0
        if np.size(mask)<10:
            return None
        y_new = y_new[mask]
        y_old = y_old[mask]
        x_new = x_new[mask]
        x_old = x_old[mask]
        princ_coeff = co.pol_oper.derotate_points(
            self.ds.curr_depth_im,
            np.array([PRIM_Y - self.roi_original[0, 0],
                      PRIM_X - self.roi_original[0, 1]]),
            self.ds.angle,
            self.ds.center)
        y_true_old = ((y_old - princ_coeff[0]) *
                      prev_z[y_old,
                             x_old] / float(FLNT))
        x_true_old = ((x_old - princ_coeff[1]) *
                      prev_z[y_old,
                             x_old] / float(FLNT))
        y_true_new = ((y_new - princ_coeff[0]) *
                      curr_z[y_new,
                             x_new] / float(FLNT))
        x_true_new = ((x_new - princ_coeff[1]) *
                      curr_z[y_new,
                             x_new] / float(FLNT))
        # DEBUGGING
        #cv2.imshow('test', (self.curr_roi_patch).astype(np.uint8))
        # cv2.waitKey(10)
        dx = x_true_new - x_true_old
        dy = y_true_new - y_true_old
        dz = curr_z[y_new, x_new] - prev_z[y_old, x_old]

        return np.concatenate((dx.reshape(-1, 1),
                               dy.reshape(-1, 1),
                               dz.reshape(-1, 1)), axis=1)
    @timeit
    def extract(self,bin_size=None):
        '''
        Compute 3DHOF features
        '''
        self.roi = self.find_roi(self.ds.prev_patch, self.ds.curr_patch,
                                 self.ds.prev_patch_pos, self.ds.curr_patch_pos)
        self.roi_original = self.find_roi(
            self.ds.prev_patch_original, self.ds.curr_patch_original,
            self.ds.prev_patch_pos_original,
            self.ds.curr_patch_pos_original)
        if bin_size is None:
            self.hist.bin_size = self.bin_size
        self.hist.range = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
        disp = self.compute_scene_flow()
        if disp is None:
            return None
        disp_norm = np.sqrt((disp[:, 0] * disp[:, 0] + disp[:, 1] *
                             disp[:, 1] + disp[:, 2] * disp[:, 2]))[:, None]
        disp_norm[disp_norm == 0] = 1
        disp = disp / disp_norm.astype(float)
        hist, edges = self.hist.hist_data(disp)
        self.edges = edges
        self.features = hist / float(np.sum(hist))
        self.features = self.features.ravel()
        return self.features


class ZHOF(_Descriptor):

    def __init__(self, *args, **kwargs):
        _Descriptor.__init__(self, *args, **kwargs)
        self.name = 'zhof'
        initialize_logger(self)
        self.bin_size = co.CONST['ZHOF_bin_size']
        self.hist = SpaceHistogram()


    def z_flow(self, prev_depth_im, curr_depth_im):
        '''
        Computes vertical displacement to the camera, using static frame
        xy-coordinates and changing z ones.
        '''
        roi = self.roi
        self.prev_roi_patch = prev_depth_im[
            roi[0, 0]:roi[0, 1],
            roi[1, 0]:roi[1, 1]].astype(float)
        self.curr_roi_patch = curr_depth_im[
            roi[0, 0]:roi[0, 1],
            roi[1, 0]:roi[1, 1]].astype(float)
        '''
        y_size = 30
        resize_rat =y_size/float(np.shape(self.prev_roi_patch)[0]) 
        x_size = int(resize_rat * np.shape(self.prev_roi_patch)[1])
        self.prev_roi_patch = cv2.resize(self.prev_roi_patch, (x_size, y_size),
                               interpolation=cv2.INTER_NEAREST)
        self.curr_roi_patch = cv2.resize(self.curr_roi_patch, (x_size, y_size),
                               interpolation=cv2.INTER_NEAREST)
        '''
        resize_rat = 1
        nonzero_mask = (self.prev_roi_patch * self.curr_roi_patch) > 0
        if np.sum(nonzero_mask) == 0:
            return None
        '''
        #DEBUGGING
        cv2.imshow('test_prev',(self.prev_roi_patch%255).astype(np.uint8))
        cv2.imshow('test_curr', (self.curr_roi_patch%255).astype(np.uint8))
        cv2.waitKey(30)
        '''
        try:
            yx_coords = (find_nonzero(
            nonzero_mask.astype(np.uint8)).astype(float)/resize_rat
            -
            np.array([[PRIM_Y - self.roi[0, 0],
                       PRIM_X - self.roi[1, 0]]]))
        except ValueError:
            return None
        prev_z_coords = self.prev_roi_patch[nonzero_mask][:,
                                                          None].astype(float)
        curr_z_coords = self.curr_roi_patch[nonzero_mask][:,
                                                          None].astype(float)
        dz_coords = (curr_z_coords - prev_z_coords).astype(float)
        # invariance to environment height variance:
        dz_outliers = self.find_outliers(dz_coords, 3.).ravel()
        dz_coords = dz_coords[dz_outliers == 0]
        yx_coords = yx_coords[dz_outliers == 0, :]
        yx_coords_in_space = (yx_coords * dz_coords / FLNT)
        return np.concatenate((yx_coords_in_space,
                               dz_coords), axis=1)

    @timeit
    def extract(self, bin_size=None):
        '''
        Compute ZHOF features
        '''
        '''
        #DEBUGGING
        if self.ds.prev_patch_pos is not None:
            print 'extract',self.ds.prev_patch.shape, self.ds.curr_patch.shape
        '''
        if self.ds.prev_patch is None or self.ds.curr_patch is None:
            return None
        '''
        #DEBUGGING
        print self.ds.prev_patch_pos, self.ds.curr_patch_pos
        exit()
        '''
        if self.ds.curr_count - self.ds.prev_count > co.CONST[
            'min_frame_count_diff']:
            return None
        self.roi = self.find_roi(self.ds.prev_patch, self.ds.curr_patch,
                                 self.ds.prev_patch_pos, self.ds.curr_patch_pos)
        self.roi_original = self.find_roi(
            self.ds.prev_patch_original, self.ds.curr_patch_original,
            self.ds.prev_patch_pos_original,
            self.ds.curr_patch_pos_original)
        '''
        #DEBUGGING
        print self.roi
        '''
        if bin_size is None:
            self.hist.bin_size = self.bin_size
        else:
            self.hist.bin_size = self.bin_size
        self.hist.range = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
        disp = self.z_flow(self.ds.prev_depth_im, self.ds.curr_depth_im)
        if disp is None:
            return None
        disp_norm = np.sqrt((disp[:, 0] * disp[:, 0] + disp[:, 1] *
                             disp[:, 1] + disp[:, 2] * disp[:, 2]))[:, None]
        disp_norm[disp_norm == 0] = 1
        disp = disp / disp_norm.astype(float)
        # print np.unique(np.around(disp,1))
        hist, edges = self.hist.hist_data(disp)
        self.edges = edges
        features = hist / float(np.sum(hist))
        features = features.ravel()
        return features

def grad_angles(patch):
    '''
    Compute gradient angles on image patch for GHOG
    '''
    y_size = 30
    x_size = int(y_size / float(np.shape(patch)[0])
                 * np.shape(patch)[1])
    patch = cv2.resize(patch, (x_size, y_size),
                       interpolation=cv2.INTER_NEAREST)
    grady, gradx = np.gradient(patch)
    ang = np.arctan2(grady, gradx)
    #ang[ang < 0] = ang[ang < 0] + pi

    return ang.ravel()  # returns values 0 to pi

class GHOG(_Descriptor):

    def __init__(self, *args, **kwargs):
        _Descriptor.__init__(self, *args, **kwargs)
        self.name = 'ghog'
        initialize_logger(self)
        self.bin_size = co.CONST['GHOG_bin_size']
        self.hist = SpaceHistogram()


    @timeit
    def extract(self, bin_size=None):
        '''
        Compute GHOG features
        '''
        im_patch = self.ds.curr_patch.astype(int)
        if bin_size is None:
            self.hist.bin_size = self.bin_size
        else:
            self.hist.bin_size = bin_size
        # DEBUGGING: added -pi (check grad_angles too)
        self.hist.range = [[-pi, pi]]
        gradients = grad_angles(im_patch)
        hist, edges = self.hist.hist_data(gradients)
        self.edges = edges
        #hist[0] = max(0, hist[0] - np.sum(im_patch==0))
        hist = hist / float(np.sum(hist))

        return hist


class ContourStatistics(_Descriptor):

    def __init__(self, *args, **kwargs):
        _Descriptor.__init__(self, *args, **kwargs)
        self.name = 'contour_stats'
        initialize_logger(self)

    @timeit
    def extract(self, resize_size=None):
         return np.ravel(cv2.HuMoments(cv2.moments(self.ds.hand_contour)))






class TDXYPCA(_Descriptor):

    def __init__(self, *args, **kwargs):
        _Descriptor.__init__(self, *args, **kwargs)
        self.name = '3dxypca'
        initialize_logger(self)
        self.pca_resize_size = co.CONST['3DXYPCA_size']
        self.edges = [['X' + str(cnt) for cnt in range(self.pca_resize_size)]+
                      ['Y' + str(cnt) for cnt in range(self.pca_resize_size)]]

    @timeit
    def extract(self, resize_size=None):
        '''
        Compute 3DXYPCA features
        '''
        if resize_size is not None:
            self.pca_resize_size = resize_size
        _, pca_along_2 = cv2.PCACompute(
            cv2.findNonZero(self.ds.curr_patch.astype(np.uint8)).squeeze().
            astype(float),
            np.array([]), maxComponents=1)
        rot_angle = np.arctan2(pca_along_2[0][1], pca_along_2[0][0])
        patch = co.pol_oper.derotate(self.ds.curr_patch, rot_angle,
                                     (self.ds.curr_patch.shape[0] / 2,
                                      self.ds.curr_patch.shape[1] / 2))
        # DEBUGGING
        # cv2.imshow('test',patch.astype(np.uint8))
        # cv2.waitKey(10)
        patch_res = cv2.resize(patch, (self.pca_resize_size,
                                       self.pca_resize_size),
                               interpolation=cv2.INTER_NEAREST)
        patch_res_mask = patch_res == 0

        masked_array = np.ma.array(patch_res, mask=patch_res_mask)
        masked_mean_0 = np.ma.mean(masked_array, axis=0)
        masked_mean_1 = np.ma.mean(masked_array, axis=1)
        cor_patch_res_0 = patch_res.copy()
        cor_patch_res_1 = patch_res.copy()
        cor_patch_res_0[patch_res_mask] = np.tile(masked_mean_0[None,
                                                                :], (patch_res.shape[0], 1))[
            patch_res_mask]
        cor_patch_res_1[patch_res_mask] = np.tile(masked_mean_1[:, None], (
            1, patch_res.shape[1]))[
            patch_res_mask]
        _, pca_along_0 = cv2.PCACompute(
            cor_patch_res_0, np.array(
                []), maxComponents=1)
        _, pca_along_1 = cv2.PCACompute(cor_patch_res_1.T, np.array([]),
                                        maxComponents=1)
        features = np.concatenate((pca_along_0[0], pca_along_1[0]), axis=0)
        return features
