import sys
import os
import warnings
import logging
import glob
from math import pi
import numpy as np
from numpy.linalg import pinv
import cv2
import class_objects as co
import sparse_coding as sc
import hand_segmentation_alg as hsa
import hist4d as h4d
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import cPickle as pickle

# Kinect Intrinsics
'''
PRIM_X = 479.75
PRIM_Y = 269.75
FLNT = 540.68603515625
'''
# Senz3d Intrinsics
PRIM_X = 317.37514566554989
PRIM_Y = 246.61273826510859
FLNT = 595.333159044648 / (30.48 / 1000.0)


logging.basicConfig(format='%(levelname)s:%(message)s')


def checktypes(objects, classes):
    '''
    Checks type of input objects and prints caller's doc string
    and exits if there is a problem
    '''
    frame = sys._getframe(1)
    try:
        if not all([isinstance(obj, instance) for
                    obj, instance in zip(objects, classes)]):
            raise TypeError(getattr(frame.f_locals['self'].__class__,
                                    frame.f_code.co_name).__doc__)
    finally:
        del frame


def find_nonzero(arr):
    '''
    Finds nonzero elements positions
    '''
    return np.fliplr(cv2.findNonZero(arr).squeeze())


def grad_angles(patch):
    '''
    Compute gradient angles on image patch for GHOG
    '''
    gradx, grady = np.gradient(patch.astype(float))
    ang = np.arctan2(grady, gradx)
    ang[ang < 0] = pi + ang[ang < 0]

    return ang.ravel()  # returns values 0 to pi


def prepare_dexter_im(img):
    '''
    Compute masks for images
    '''
    binmask = img < 6000
    contours = cv2.findContours(
        (binmask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours_area = [cv2.contourArea(contour) for contour in contours]
    hand_contour = contours[np.argmax(contours_area)].squeeze()
    hand_patch = img[np.min(hand_contour[:, 1]):np.max(hand_contour[:, 1]),
                     np.min(hand_contour[:, 0]):np.max(hand_contour[:, 0])]
    hand_patch_max = np.max(hand_patch)
    hand_patch[hand_patch == hand_patch_max] = 0
    img[img == hand_patch_max] = 0
    med_filt = np.median(hand_patch[hand_patch != 0])
    thres = np.min(img) + 0.1 * (np.max(img) - np.min(img))
    binmask[np.abs(img - med_filt) > thres] = False
    hand_patch[np.abs(hand_patch - med_filt) > thres] = 0
    hand_patch_pos = np.array(
        [np.min(hand_contour[:, 1]), np.min(hand_contour[:, 0])])
    return img * binmask,\
        hand_patch, hand_patch_pos


def prepare_im(img):
    contours = cv2.findContours(
        (img).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours_area = [cv2.contourArea(contour) for contour in contours]
    hand_contour = contours[np.argmax(contours_area)].squeeze()
    hand_patch = img[np.min(hand_contour[:, 1]):np.max(hand_contour[:, 1]),
                     np.min(hand_contour[:, 0]):np.max(hand_contour[:, 0])]
    hand_patch_pos = np.array(
        [np.min(hand_contour[:, 1]), np.min(hand_contour[:, 0])])
    return img,\
        hand_patch, hand_patch_pos


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
        weights = np.ones(sample.shape[0]) / float(sample.shape[0])
        hist, edges = np.histogramdd(sample, self.bin_size, range=self.range,
                                     weights=weights)
        return hist, edges


class Action(object):
    '''
    Class to hold an action
    '''

    def __init__(self):
        self.features = None
        self.sparse_features = None
        self.sync = []
        self.samples_indices = None

    def add_features(self, features=None, sparse_features=None):
        '''
        Add frame features to action
        '''
        if features is not None:
            if self.features is None:
                self.features = [np.atleast_2d(feature).T for feature in
                                 features]
            else:
                for count, feature in enumerate(features):
                    self.features[count] = np.concatenate((self.features[count],
                                                           np.atleast_2d(feature).T),
                                                          axis=1)

        if sparse_features is not None:
            if self.sparse_features is None:
                self.sparse_features = [np.atleast_2d(feature).T for feature in
                                        sparse_features]
            else:
                for count, feature in enumerate(sparse_features):
                    self.sparse_features[count] = np.concatenate((self.sparse_features[count],
                                                                  np.atleast_2d(feature).T),
                                                                 axis=1)

    def update_sparse_features(self, dictionaries):
        '''
        Update sparse features using trained dictionaries
        '''
        self.flush_sparse_features()
        self.sparse_features = []
        for feature, dictionary in zip(self.features, dictionaries):
            self.sparse_features.append(np.dot(dictionary, feature))

    def flush_sparse_features(self):
        '''
        Empty sparse_features
        '''
        self.sparse_features = None

    def flush_features(self):
        '''
        Empty features
        '''
        self.features = None


class Actions(object):
    '''
    Class to hold multiple actions
    '''

    def __init__(self):
        self.actions = []
        self.testing = Action()
        self.save_path = (os.getcwd() +
                          os.sep + 'saved_actions.pkl')

    def remove_action(self, act_num):
        '''
        Remove an action using act_num. Set -1 to remove last
        added action, or 'all' to remove every action
        '''
        if act_num == 'all':
            del self.actions
        else:
            del self.actions[act_num]

    def add_action(self, dictionaries=None,
                   depthdata=None,
                   masks_needed=True,
                   masksdata=None,
                   use_dexter=False,
                   visualize_=False,
                   for_testing=False):
        '''
        features_extract= FeatureExtraction Class
        dictionaries= SparseDictionaries Class
        depthdata= (Directory with depth frames) OR (list of depth frames)
        masksdata=>Only used when use_dexter is False=>
            (Directory with hand masks) OR (list of hand masks)
        use_dexter= True if Dexter 1 TOF Dataset is used
        visualize= True to visualize features extracted from frames
        for_testing = True if input data is testing data and features are
                      returned as (3DHOF,GHOG)
        '''
        features_extract = FeatureExtraction(visualize_=visualize_)
        checktypes([dictionaries, features_extract], [
            SparseDictionaries, FeatureExtraction])
        if depthdata is None:
            raise Exception("Depth data frames are at least  needed")
        if for_testing:
            action = self.testing
        else:
            self.actions.append(Action())
            action = self.actions[-1]
        if not use_dexter:
            if masks_needed:
                if masksdata is None:
                    raise Exception('masksdata must be given if ' +
                                    'masks_needed')
                if isinstance(masksdata, str):
                    files = glob.glob(masksdata + '/*.png')
                    masks = [cv2.imread(filename, 0) for filename
                             in files]
                else:
                    masks = masksdata[:]
            else:
                masks = None
        if isinstance(depthdata, str):
            files = []
            samples_indices = []
            if os.path.isdir(os.path.join(depthdata, '0')):
                for root, dirs, filenames in os.walk(depthdata):
                    for filename in filenames:
                        if filename.endswith('.png'):
                            fil = os.path.join(root, filename)
                            par_folder = os.path.normpath(fil).split(
                                os.sep)[-2]
                            try:
                                ind = int(par_folder)
                                files.append(fil)
                                samples_indices.append(ind)
                            except ValueError:
                                pass
                action.samples_indices = np.array(samples_indices)
            else:
                files = glob.glob(os.path.join(depthdata,'*.png'))
                action.samples_indices = np.zeros(len(files))
            imgs = [cv2.imread(filename, -1) for filename
                    in files]
            action.sync = [int(filter(str.isdigit,
                                                os.path.basename(filename)))
                                     for filename in files]
        else:
            imgs = depthdata[:]
        for img_count, img in enumerate(imgs):
            features_extract.update(img, img_count, use_dexter,
                                    masks_needed, masks)
            # Extract Features
            features = features_extract.extract_features()
            # Save action to actions object
            if features is not None:
                if visualize_:
                    features_extract.visualize()
                if len(dictionaries.dicts) == 0:
                    action.add_features(features=features)
                else:
                    sparse_features = []
                    for feat_num, feature in enumerate(features):
                        sparse_features.append(np.dot(dictionaries.dicts[feat_num],
                                                      feature))

                    action.add_features(features=features,
                                                  sparse_features=sparse_features)
        if for_testing:
            return self.testing.sparse_features

        return 0

    def update_sparse_features(self, dicts,
                               act_num='all', ret_sparse=False):
        '''
        Update sparse features for all Actions or a single one, specified by
        act_num.
        Requirement is that existent dictionaries have been trained
        '''
        if any([(dicti is None) for dicti in dicts]):
            raise Exception('Dictionaries for existent features must' +
                            'have been trained before calling')
        if act_num == 'all':
            iter_quant = self.actions
        else:
            iter_quant = [self.actions[act_num]]
        for action in iter_quant:
            action.update_sparse_features(dicts)
        if ret_sparse:
            sparse_features = []
            for action in self.actions:
                sparse_features.append(action.sparse_features)
            return sparse_features

    def save(self, save_path=None):
        '''
        Save actions to file
        '''
        if save_path is None:
            actions_path = self.save_path
        else:
            actions_path = save_path

        logging.info('Saving actions to ' + actions_path)
        with open(actions_path, 'wb') as output:
            pickle.dump(self.actions, output, -1)


class SparseDictionaries(object):
    '''
    Class to hold sparse coding dictionaries
    '''

    def __init__(self):
        self.inv_dicts = []
        self.dicts = []
        self.initialized = True
        self.save_path = (os.getcwd() +
                          os.sep + 'saved_dictionaries.pkl')

    def train(self, actions, feat_num, act_num=0, bmat=None,
              final_iter=False, ret_errors=False, train_max_im_num=-1):
        '''
        actions=Actions class
        act_num=0: action postion inside actions.actions list
        feat_num=0: features position inside
                    actions.actions[act_num].features list
        final_iter=False: if True, it computes the dictionary matrix, which
                       converts input features to sparse
        '''
        checktypes([actions, feat_num], [Actions, int])
        self.inv_dicts[feat_num].inp_features = actions.actions[
            act_num].features[feat_num][:, :train_max_im_num].copy()
        self.inv_dicts[feat_num].out_features = (actions.actions[act_num].
                                                 sparse_features[feat_num].copy())
        if bmat is not None:
            self.inv_dicts[feat_num].bmat = bmat.copy()
        elif self.inv_dicts[feat_num].bmat is None:
            raise Exception('Dictionaries not correctly initialized')
        if ret_errors:
            init_error = np.linalg.norm(self.inv_dicts[feat_num].inp_features -
                                        np.dot(self.inv_dicts[feat_num].bmat,
                                               self.inv_dicts[feat_num].
                                               out_features))
        self.inv_dicts[feat_num].bmat = self.inv_dicts[
            feat_num].dictionary_training()
        if final_iter:
            self.dicts[feat_num] = (pinv(self.inv_dicts[feat_num].bmat))
        if ret_errors:
            final_error = np.linalg.norm(self.inv_dicts[feat_num].inp_features -
                                         np.dot(self.inv_dicts[feat_num].bmat,
                                                self.inv_dicts[feat_num].
                                                out_features))
            return init_error, final_error

    def initialize(self, total_features_num):
        '''
        total_features_num is the total features types number of the actions
        '''
        self.dicts = []
        checktypes([total_features_num], [int])
        self.inv_dicts = []
        for _ in range(total_features_num):
            self.inv_dicts.append(sc.SparseCoding())
            self.dicts.append(None)
        self.initialized = True

    def add_dict(self, actions=None, feat_dim=0,
                 act_num=0, feat_num=0):
        '''
        actions=Actions class
        feat_num=Position where dict will be put
        feat_dim=Current features dimension
        In case feat_dim is not set:
            actions is a necessary variable
            act_num=0: action postion inside actions.actions list
            feat_num=0: features position inside
                        actions.actions[act_num].features list
                        AND position where dict will be put
        '''
        if not self.initialized:
            raise Exception('First call initialize function')
        if feat_num > len(self.inv_dicts):
            raise Exception('feat_num is more than total dictionaries ' +
                            'length. Check initialize stage')
        if feat_num == -1:
            logging.warning('feat_num argument in add_dict should' +
                            ' be set when feat_dim is not, ' +
                            'else dictionary for feature 0 will be ovewritten')
            feat_num = 0
        if feat_dim == 0:
            checktypes([actions], [Actions])
            if len(actions.actions) == 0:
                raise Exception('Actions should have at least ' +
                                'one entry, or set feat_dim')
            self.inv_dicts[feat_num].initialize(actions.actions[act_num].
                                                features[feat_num].shape[0])
        else:
            self.inv_dicts[feat_num].initialize(feat_dim)

    def flush(self, dict_num='all'):
        '''
        Reinitialize all or one dictionary
        '''
        if dict_num == 'all':
            iter_quant = self.inv_dicts
            iter_range = range(len(self.inv_dicts))
        else:
            iter_quant = [self.inv_dicts[dict_num]]
            iter_range = [dict_num]
        feat_dims = [] * len(iter_quant)
        for count, inv_dict in enumerate(iter_quant):
            feat_dims[count] = None
            try:
                feat_dim = inv_dict.bmat.shape[0]
                feat_dims[count] = feat_dim
            except AttributeError:
                feat_dims[count] = None
        for count in iter_range:
            if feat_dims[count] is not None:
                self.inv_dicts[count].initialize(feat_dims[count],
                                                 flush_variables=True)

    def save(self, save_path=None):
        '''
        Save dictionaries to file
        '''
        if save_path is None:
            dictionaries_path = self.save_path
        else:
            dictionaries_path = save_path

        logging.info('Saving Dictionaries to ' + dictionaries_path)
        with open(dictionaries_path, 'wb') as output:
            pickle.dump(self.dicts, output, -1)


class FeatureExtraction(object):
    '''
    Features computation class
    '''

    def __init__(self, visualize_=False):
        self.features = np.zeros(0)
        self.prev_projection = np.zeros(0)
        self.curr_projection = np.zeros(0)
        self.roi = np.zeros(0)
        self.prev_patch = None
        self.curr_patch = None
        self.prev_roi_patch = None
        self.curr_roi_patch = None
        self.prev_patch_pos = None
        self.curr_patch_pos = None
        self.prev_count = 0
        self.curr_count = 0
        self.prev_depth_im = np.zeros(0)
        self.curr_depth_im = np.zeros(0)
        self.hofhist = SpaceHistogram()
        self.hoghist = SpaceHistogram()
        self.fig = None
        self.hof_features = None
        self.hog_features = None
        self.hof_plots = None
        self.hog_plot = None
        if visualize_:
            self.view = FeatureVisualization()

    def visualize(self):
        self.view.plot_hof(self.hof_features)
        self.view.plot_hog(self.hog_features)
        self.view.plot_3d_projection(self.roi,
                                     self.prev_roi_patch,
                                     self.curr_roi_patch)
        self.view.plot_2d_patches(self.prev_roi_patch,
                                  self.curr_roi_patch)
        self.view.draw()

    def compute_scene_flow(self, prev_depth_im, curr_depth_im):
        '''
        Computes scene flow for 3DHOF
        '''
        self.prev_roi_patch = prev_depth_im[self.roi[0, 0]:self.roi[0, 1],
                                            self.roi[1, 0]:self.roi[1, 1]]
        self.curr_roi_patch = curr_depth_im[self.roi[0, 0]:self.roi[0, 1],
                                            self.roi[1, 0]:self.roi[1, 1]]
        nonzero_mask = (self.prev_roi_patch + self.curr_roi_patch) > 0
        yx_coords = (find_nonzero(nonzero_mask.astype(np.uint8)).astype(float) -
                     np.array([[PRIM_Y - self.roi[0, 0],
                                PRIM_X - self.roi[1, 0]]]))
        prev_z_coords = self.prev_roi_patch[nonzero_mask][:,
                                                          None].astype(float)
        curr_z_coords = self.curr_roi_patch[nonzero_mask][:,
                                                          None].astype(float)
        dz_coords = (curr_z_coords - prev_z_coords).astype(float)
        yx_coords_in_space = yx_coords * dz_coords / FLNT
        return np.concatenate((yx_coords_in_space,
                               dz_coords), axis=1)

    def find_roi(self, prev_patch, curr_patch, prev_patch_pos, curr_patch_pos):
        '''
        Find unified ROI, concerning 2 consecutive frames
        '''
        self.roi = np.array([[
            min(prev_patch_pos[0], curr_patch_pos[0]),
            max((prev_patch.shape[0] + prev_patch_pos[0],
                 curr_patch.shape[0] + curr_patch_pos[0]))],
            [min(prev_patch_pos[1], curr_patch_pos[1]),
             max(prev_patch.shape[1] + prev_patch_pos[1],
                 curr_patch.shape[1] + curr_patch_pos[1])]])

    def hof3d(self, prev_depth_im, curr_depth_im):
        '''
        Compute 3DHOF features
        '''
        if self.hofhist.bin_size is None:
            self.hofhist.bin_size = co.CONST['3DHOF_bin_size']
            #self.hofhist.range = [[-1,1],[-1,1],[-1,1]]
        disp = self.compute_scene_flow(prev_depth_im, curr_depth_im)
        disp_norm = np.sqrt((disp[:, 0] * disp[:, 0] + disp[:, 1] *
                             disp[:, 1] + disp[:, 2] * disp[:, 2]))[:, None]
        disp_norm[disp_norm == 0] = 1
        disp /= disp_norm.astype(float)
        return self.hofhist.hist_data(disp)

    def ghog(self, depth_im):
        '''
        Compute GHOG features
        '''
        im_patch = depth_im[self.roi[0, 0]:self.roi[0, 1],
                            self.roi[1, 0]:self.roi[1, 1]]
        if self.hoghist.range is None:
            self.hoghist.bin_size = co.CONST['GHOG_bin_size']
            self.hoghist.range = [[0, pi]]
        return self.hoghist.hist_data(grad_angles(im_patch))

    def extract_features(self):
        '''
        Returns 3DHOF and GHOG
        '''
        if self.prev_patch is None or \
           self.curr_count - self.prev_count > co.CONST['min_frame_count_diff']:
            return None
        self.find_roi(self.prev_patch, self.curr_patch,
                      self.prev_patch_pos, self.curr_patch_pos)
        self.hof_features = self.hof3d(
            self.prev_depth_im, self.curr_depth_im)
        self.hog_features = self.ghog(self.curr_depth_im)
        return self.hof_features[0].ravel(), self.hog_features[0].ravel()

    def update(self, img, img_count, use_dexter=False, masks_needed=True,
               masks=None):
        '''
        Update frames
        '''

        if use_dexter:
            mask, hand_patch, hand_patch_pos = prepare_dexter_im(
                img)
        else:
            if masks_needed:
                hand_patch, hand_patch_pos = hsa.main_process(masks[img_count])
                if hand_patch.shape[1] == 0:
                    warnings.warn('Problem with frame' + str(img_count))
                    hsa.main_process(masks[img_count], display=1)
                    mask = img * (masks[img_count] > 0)
            else:
                mask, hand_patch, hand_patch_pos = prepare_im(img)
        (self.prev_depth_im,
         self.curr_depth_im) = (self.curr_depth_im,
                                mask)
        (self.curr_count,
         self.prev_count) = (img_count,
                             self.curr_count)
        (self.prev_patch,
         self.curr_patch) = (self.curr_patch,
                             hand_patch)
        (self.prev_patch_pos,
         self.curr_patch_pos) = (self.curr_patch_pos,
                                 hand_patch_pos)


class Callback(object):

    def unpause(self):
        global pause
        pause ^= True


class FeatureVisualization(object):

    def __init__(self):
        import matplotlib.pyplot as plt
        plt.ion()
        self.fig = plt.figure()
        gs = gridspec.GridSpec(120, 100)
        self.patches3d_plot = self.fig.add_subplot(
            gs[:50, 60:100], projection='3d')
        self.patches2d_plot = self.fig.add_subplot(gs[:50, :50])
        self.hist4d = h4d.Hist4D()
        self.hof_plots = (self.fig.add_subplot(gs[60:100 - 5, :45], projection='3d'),
                          self.fig.add_subplot(gs[60:100 - 5, 45:50]),
                          self.fig.add_subplot(gs[100 - 4:100 - 2, :50]),
                          self.fig.add_subplot(gs[100 - 2:100, :50]))
        self.pause_key = Button(
            self.fig.add_subplot(gs[110:120, 25:75]), 'Next')
        self.pause_key.on_clicked(self.unpause)
        self.hog_plot = self.fig.add_subplot(gs[70:100, 70:100])
        plt.show()

    def plot_hog(self, hog_features):
        hog_hist, hog_bins = hog_features
        width = 0.7 * (hog_bins[0][1] - hog_bins[0][0])
        center = (hog_bins[0][:-1] + hog_bins[0][1:]) / 2
        self.hog_plot.clear()
        self.hog_plot.bar(center, hog_hist, align='center', width=width)

    def plot_hof(self, hof_features):
        self.hist4d.draw(hof_features[0], hof_features[
                         1], fig=self.fig, all_axes=self.hof_plots)

    def plot_3d_projection(self, roi, prev_roi_patch, curr_roi_patch):
        self.patches3d_plot.clear()
        nonzero_mask = (prev_roi_patch + curr_roi_patch) > 0
        yx_coords = (find_nonzero(nonzero_mask.astype(np.uint8)).astype(float) -
                     np.array([[PRIM_Y - roi[0, 0],
                                PRIM_X - roi[1, 0]]]))
        prev_z_coords = prev_roi_patch[nonzero_mask][:,
                                                     None].astype(float)
        curr_z_coords = curr_roi_patch[nonzero_mask][:,
                                                     None].astype(float)
        dz_coords = (curr_z_coords - prev_z_coords).astype(float)
        yx_coords_in_space = yx_coords * dz_coords / FLNT
        prev_yx_proj = yx_coords * prev_z_coords / (FLNT)
        curr_yx_proj = yx_coords * curr_z_coords / (FLNT)
        prev_yx_proj = prev_yx_proj[prev_z_coords.ravel() != 0]
        curr_yx_proj = curr_yx_proj[curr_z_coords.ravel() != 0]
        self.patches3d_plot.scatter(prev_yx_proj[:, 1], prev_yx_proj[:, 0],
                                    prev_z_coords[prev_z_coords != 0],
                                    zdir='z', s=4, c='r', depthshade=False, alpha=0.5)
        self.patches3d_plot.scatter(curr_yx_proj[:, 1], curr_yx_proj[:, 0],
                                    curr_z_coords[curr_z_coords != 0],
                                    zdir='z', s=4, c='g', depthshade=False, alpha=0.5)
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

    def plot_3d_patches(self, roi, prev_roi_patch, curr_roi_patch):
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

    def plot_2d_patches(self, prev_roi_patch, curr_roi_patch):
        self.patches2d_plot.clear()
        self.patches2d_plot.imshow(prev_roi_patch, cmap='Reds', alpha=0.5)
        self.patches2d_plot.imshow(curr_roi_patch, cmap='Greens', alpha=0.5)

    def draw(self):
        import time
        self.fig.canvas.draw()
        try:
            self.fig.canvas.start_event_loop(30)
        except:
            time.sleep(1)

    def unpause(self, val):
        plt.gcf().canvas.stop_event_loop()


class ActionRecognition(object):
    '''
    Class to hold everything about action recognition
    '''

    def __init__(self, log_lev='INFO', train_max_im_num=100,
                 train_max_iter=2):
        self.dictionaries = SparseDictionaries()
        self.actions = Actions()
        self.log_lev = log_lev
        logging.getLogger().setLevel(log_lev)
        self.train_max_im_num = train_max_im_num
        self.train_max_iter = train_max_iter

    def add_action(self, depthdata=None,
                   masks_needed=True,
                   masksdata=None,
                   use_dexter=False,
                   visualize=False,
                   for_testing=False):
        '''
        actions.add_action alias
        '''
        res = self.actions.add_action(self.dictionaries,
                                      depthdata,
                                      masks_needed,
                                      masksdata,
                                      use_dexter,
                                      visualize_=visualize,
                                      for_testing=for_testing)
        return res

    def train_sparse_dictionaries(self, act_num=None,
                                  depthdata=None,
                                  masks_needed=True,
                                  masksdata=None,
                                  use_dexter=False,
                                  save_trained=True,
                                  save_path=False):
        '''
        Add Dexter 1 TOF Dataset or depthdata + binarymaskdata and
        set use_dexter to False (directory with .png or list accepted)
        Inputs:
            act_num: action number to use for training
            depthdata: path or list of images
            masksdata: path or list of images
            masks_needed: true if no hand segmentation is done beforehand
                         and Dexter 1 dataset is not used
            use_dexter: true if Dexter 1 dataset is used
            iterations: training iterations
            save_trained: save dictionaries after training
        '''
        checktypes([act_num, depthdata, masks_needed, masksdata, use_dexter,
                    save_trained, save_path],
                   [int, (list, str, type(None)), (bool, int),
                    (list, str, type(None)),
                    (bool, int), (bool, int), (bool, int)])
        if act_num is None and len(self.actions.actions) > 0:
            act_num = 0
        elif len(self.actions.actions) == 0:
            raise Exception('Run add_action first and then call ' +
                            'train_sparse_dictionaries')
        feat_num = len(self.actions.actions[act_num].features)

        if depthdata is None and len(self.actions.actions) == 0:
            raise Exception("Path/List of frames depth data is at least needed" +
                            ' because action data is empty')
        elif not use_dexter and masksdata is None:
            if masks_needed:
                raise Exception("Path/List of masksdata is at least needed" +
                                ' because use_dexter is false and masks_neede'
                                'd is True')

        if depthdata is not None:
            logging.info('Adding action..')
            self.actions.add_action(self.dictionaries,
                                    depthdata,
                                    masks_needed,
                                    masksdata,
                                    use_dexter)
        # Train dictionaries
        self.dictionaries.initialize(feat_num)
        frames_num = self.actions.\
            actions[act_num].features[0].shape[1]
        logging.info('Frames number: ' + str(frames_num))
        train_im_num = min(frames_num, self.train_max_im_num)
        logging.info('Frames to be used for training: ' + str(train_im_num))
        logging.info('Creating dictionaries..')
        for count in range(feat_num):
            self.dictionaries.add_dict(self.actions, feat_num=count)
        train_actions = train_im_num
        final_errors = []
        for iteration in range(self.train_max_iter):
            logging.info('Epoch: ' + str(iteration))
            self.actions.actions[act_num].flush_sparse_features()
            logging.info('Running Feature Sign Search Algorithm..')
            for img_count in range(train_im_num):
                logging.info('Frame ' + str(img_count) + ' is being edited')
                sparse_features = []
                for feat_count in range(feat_num):
                    coding = sc.SparseCoding(log_lev=self.log_lev)
                    coding.feature_sign_search_algorithm(
                        inp_features=np.atleast_2d(
                            self.actions.actions[
                                act_num].features[feat_count]
                            [:, img_count]).T.astype(float),
                        init_bmat=self.dictionaries.inv_dicts[feat_count].
                        bmat.copy(),
                        max_iter=100)
                    sparse_features.append(np.atleast_2d(coding.
                                                         out_features).T)
                self.actions.actions[act_num].add_features(
                    sparse_features=sparse_features)
            logging.info('Training Dictionaries..')
            for feat_count in range(feat_num):
                if iteration == self.train_max_iter - 1:
                    final_errors.append(self.dictionaries.train(self.actions,
                                                                feat_count,
                                                                final_iter=True,
                                                                ret_errors=True,
                                                                train_max_im_num=train_im_num)
                                        [1])
                else:
                    self.dictionaries.train(self.actions, feat_count,
                                            train_max_im_num=train_im_num)
        logging.info('Training is completed with final errors:\n' +
                     str(final_errors))
        if save_trained:
            self.dictionaries.save()
        return(self.dictionaries.dicts)
