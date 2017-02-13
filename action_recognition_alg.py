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
import time
LOG = logging.getLogger('__name__')
CH = logging.StreamHandler(sys.stderr)
CH.setFormatter(logging.Formatter(
    '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
LOG.handlers=[]
LOG.addHandler(CH)
LOG.setLevel(logging.INFO)
# Kinect Intrinsics
PRIM_X = 479.75
PRIM_Y = 269.75
FLNT = 540.68603515625
# Senz3d Intrinsics
'''
PRIM_X = 317.37514566554989
PRIM_Y = 246.61273826510859
FLNT = 595.333159044648 / (30.48 / 1000.0)
'''



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
        hand_patch, hand_patch_pos, contours[np.argmax(contours_area)]


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
        return hist


class Action(object):
    '''
    Class to hold an action
    '''

    def __init__(self):
        self.features = None
        self.sparse_features = None
        self.sync = []
        self.samples_indices = None
        self.angles = None

    def add_features(self, features=None, sparse_features=None):
        '''
        Add frame features to action
        '''
        if features is not None:
            if self.features is None:
                self.features = [np.atleast_2d(feature.ravel()).T for feature in
                                 features]
            else:
                for count, feature in enumerate(features):
                    self.features[count] = np.concatenate((self.features[count],
                                                           np.atleast_2d(feature.ravel()).T),
                                                          axis=1)

        if sparse_features is not None:
            if self.sparse_features is None:
                self.sparse_features = [np.atleast_2d(feature.ravel()).T for feature in
                                        sparse_features]
            else:
                for count, feature in enumerate(sparse_features):
                    self.sparse_features[count] = np.concatenate((self.sparse_features[count],
                                                                  np.atleast_2d(feature.ravel()).T),
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
        self.features_extract = None

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
                   for_testing=False,
                   isderotated=True,
                   isstatic=False,
                   max_act_samples=None,
                   feature_params=None):
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
        self.features_extract = FeatureExtraction(visualize_=visualize_)
        checktypes([dictionaries, self.features_extract], [
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
            angles = []
            centers = []
            action.sync = []
            if os.path.isdir(os.path.join(depthdata, '0')):
                for root, dirs, filenames in os.walk(depthdata):
                    for filename in sorted(filenames):
                        if filename.endswith('.png'):
                            fil = os.path.join(root, filename)
                            par_folder = os.path.normpath(fil).split(
                                os.sep)[-2]
                            try:
                                ind = int(par_folder)
                                files.append(fil)
                                action.sync.append(int(filter(
                                    str.isdigit, fil)))
                                samples_indices.append(ind)
                            except ValueError:
                                pass
                        elif filename.endswith('angles.txt'):
                            fil = os.path.join(root, filename)
                            with open(fil, 'r') as inpf:
                                angles += map(float, inpf)
                        elif filename.endswith('centers.txt'):
                            fil = os.path.join(root, filename)
                            with open(fil, 'r') as inpf:
                                for line in inpf:
                                    center = [
                                        float(num) for num
                                        in line.split(' ')]
                                    centers += [center]
                action.samples_indices = np.array(samples_indices)
                action.angles = angles
            else:
                LOG.error('depthtdata must have numbered'+
                              ' subdirectories, to denote the'+
                              ' different samples')
                raise NotImplementedError
            imgs = [cv2.imread(filename, -1) for filename
                    in files]
        else:
            imgs = depthdata[:]
        if not isderotated:
            imgs = [co.pol_oper.derotate(imgs[count],
                                         angles[count],
                                         centers[count])
                    for count in range(len(imgs))]
        feat_count=0
        for img_count, img in enumerate(imgs):
            #DEBUGGING
            #cv2.imshow('test',(img%255).astype(np.uint8))
            #cv2.waitKey(10)
            self.features_extract.update(img, img_count, use_dexter,
                                    masks_needed, masks)
            # Extract Features
            features = self.features_extract.extract_features(isstatic=isstatic,
                                                         params=feature_params)
            # Save action to actions object
            if features is not None:
                if visualize_:
                    self.features_extract.visualize()
                if len(dictionaries.dicts) == 0:
                    action.add_features(features=features)
                else:
                    sparse_features = []
                    for feat_num, feature in enumerate(features):
                        sparse_features.append(np.dot(dictionaries.dicts[feat_num],
                                                      feature))
                    action.add_features(features=features,
                                                  sparse_features=sparse_features)
                feat_count+=1
                if max_act_samples is not None:
                    if feat_count==max_act_samples:
                        break
        if for_testing:
            if len(dictionaries.dicts) != 0:
                return self.testing.sparse_features, self.testing.sync
            else:
                return self.testing.features, self.testing.sync
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

        LOG.info('Saving actions to ' + actions_path)
        with open(actions_path, 'wb') as output:
            pickle.dump(self.actions, output, -1)


class SparseDictionaries(object):
    '''
    Class to hold sparse coding dictionaries
    '''

    def __init__(self, des_dim=None):
        self.inv_dicts = []
        self.dicts = []
        self.initialized = True
        self.save_path = (os.getcwd() +
                          os.sep + 'saved_dictionaries.pkl')
        self.des_dim = des_dim

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
            self.inv_dicts.append(sc.SparseCoding(des_dim=self.des_dim))
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
            LOG.warning('feat_num argument in add_dict should' +
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
        feat_dims = [] * len(iter_range)
        for count, inv_dict in zip(iter_range, iter_quant):
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

        LOG.info('Saving Dictionaries to ' + dictionaries_path)
        with open(dictionaries_path, 'wb') as output:
            pickle.dump(self.dicts, output, -1)

def grad_angles(patch):
    '''
    Compute gradient angles on image patch for GHOG
    '''
    grady, gradx = np.gradient(patch.astype(float))
    ang = np.arctan2(grady, gradx)
    ang[ang < 0] = ang[ang < 0] + pi

    return ang.ravel()  # returns values 0 to pi

class FeatureExtraction(object):
    '''
    Features computation class
    '''

    def __init__(self, visualize_=False):
        self.extract_time = []
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
        self.hand_contour = None
        self.fig = None
        self.hof_features = None
        self.ghog_features = None
        self.hof_plots = None
        self.hog_plot = None
        self.winSize = (48,48)
        self.blockSize = (16,16)
        self.blockStride = (8,8)
        self.cellSize = (8,8)
        self.nbins = 9
        self.derivAperture = 1
        self.winSigma = 4.
        self.histogramNormType = 0
        self.L2HysThreshold = 2.0000000000000001e-01
        self.gammaCorrection = 0
        self.nlevels = 64
        self.hists = []
        self.winstride = (8, 8)
        self.padding = (4, 4)
        self.hog = cv2.HOGDescriptor(self.winSize, self.blockSize,
                                     self.blockStride, self.cellSize,
                                     self.nbins, self.derivAperture, self.winSigma,
                                     self.histogramNormType, self.L2HysThreshold,
                                     self.gammaCorrection, self.nlevels)

        if visualize_:
            self.view = FeatureVisualization()

    def visualize(self):
        self.view.plot_hof(self.hof_features)
        self.view.plot_hog(self.ghog_features)
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
        yx_coords = (find_nonzero(
            nonzero_mask.astype(np.uint8)).astype(float)
                            -
                     np.array([[PRIM_Y - self.roi[0, 0],
                                PRIM_X - self.roi[1, 0]]]))
        prev_z_coords = self.prev_roi_patch[nonzero_mask][:,
                                                          None].astype(float)
        curr_z_coords = self.curr_roi_patch[nonzero_mask][:,
                                                          None].astype(float)
        dz_coords = (curr_z_coords - prev_z_coords).astype(float)
        #invariance to environment height variance:
        dz_coords[dz_coords>250] = 250
        dz_coords[dz_coords<-250] = -250
        yx_coords_in_space = yx_coords * dz_coords / FLNT
        return np.concatenate((yx_coords_in_space,
                               dz_coords), axis=1)

    def find_roi(self, prev_patch, curr_patch, prev_patch_pos, curr_patch_pos):
        '''
        Find unified ROI, concerning 2 consecutive frames
        '''
        if prev_patch is None:
            prev_patch = curr_patch
            prev_patch_pos = curr_patch_pos
        self.roi = np.array([[
            min(prev_patch_pos[0], curr_patch_pos[0]),
            max((prev_patch.shape[0] + prev_patch_pos[0],
                 curr_patch.shape[0] + curr_patch_pos[0]))],
            [min(prev_patch_pos[1], curr_patch_pos[1]),
             max(prev_patch.shape[1] + prev_patch_pos[1],
                 curr_patch.shape[1] + curr_patch_pos[1])]])

    def hof3d(self, prev_depth_im, curr_depth_im, hof_params=None):
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
        hist = self.hofhist.hist_data(disp)
        hist = hist/float(disp.size)
        return hist

    def pca_features(self, square_edge_size):
        _, pca_along_2 = cv2.PCACompute(
            cv2.findNonZero(self.curr_patch.astype(np.uint8)).squeeze().
            astype(float),
            np.array([]), maxComponents=1)
        rot_angle = np.arctan2(pca_along_2[0][0], pca_along_2[0][1])
        patch = co.pol_oper.derotate(self.curr_patch, rot_angle,
                             (self.curr_patch.shape[0]/2,
                              self.curr_patch.shape[1]/2))
        patch_res = cv2.resize(patch, (square_edge_size, square_edge_size),
                              interpolation=cv2.INTER_NEAREST)
        patch_res_mask = patch_res == 0

        masked_array = np.ma.array(patch_res, mask=patch_res_mask)
        masked_mean_0 = np.ma.mean(masked_array,axis=0)
        masked_mean_1 = np.ma.mean(masked_array,axis=1)
        cor_patch_res_0 = patch_res.copy()
        cor_patch_res_1 = patch_res.copy()
        cor_patch_res_0[patch_res_mask] = np.tile(masked_mean_0[None,
                                                              :], (patch_res.shape[0], 1))[
                                                                  patch_res_mask]
        cor_patch_res_1[patch_res_mask] = np.tile(masked_mean_1[:,None], (
                                                                1, patch_res.shape[1]))[
                                                                patch_res_mask]
        _, pca_along_0 = cv2.PCACompute(cor_patch_res_0,np.array([]), maxComponents=1)
        _, pca_along_1 = cv2.PCACompute(cor_patch_res_1.T, np.array([]),
                                        maxComponents=1)
        features = np.concatenate((pca_along_0[0], pca_along_1[0]),axis=0)
        return features

    def ghog(self, depth_im, ghog_params=None):
        '''
        Compute GHOG features
        '''
        im_patch = depth_im[self.roi[0, 0]:self.roi[0, 1],
                            self.roi[1, 0]:self.roi[1, 1]]
        if self.hoghist.range is None:
            self.hoghist.bin_size = co.CONST['GHOG_bin_size']
            self.hoghist.range = [[0, pi]]
        if ghog_params is not None:
            self.hoghist.bin_size = ghog_params

        hist = self.hoghist.hist_data(grad_angles(im_patch))
        #hist[0] = max(0, hist[0] - np.sum(im_patch==0))
        hist = hist / float(np.sum(hist))
        return hist

    def extract_features(self,isstatic=False, params=([None],[None])):
        '''
        Returns 3DHOF and GHOG . isstatic to return only GHOG.
        features_params is a tuple of lists of variables ([hof_bin_numbers],
        [hog_bin_numbers])
        '''
        t1 = time.time()
        self.find_roi(self.prev_patch, self.curr_patch,
                      self.prev_patch_pos, self.curr_patch_pos)
        self.ghog_features = self.ghog(self.curr_depth_im, params[0][0])
        if not isstatic:
            if self.prev_patch is None or \
               self.curr_count - self.prev_count > co.CONST['min_frame_count_diff']:
                return None
            self.hof_features = self.hof3d(
                self.prev_depth_im, self.curr_depth_im)
            t2 = time.time()
            self.extract_time.append(t2-t1)
            return self.hof_features.ravel(), self.ghog_features.ravel()
        else:
            pca_features = self.pca_features(params[0][0])
            t2 = time.time()
            self.extract_time.append(t2-t1)
            return [pca_features]
            #return [self.ghog_features.ravel()]

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
                try:
                    mask, hand_patch, hand_patch_pos, self.hand_contour = prepare_im(img)
                except ValueError:
                    return
        (self.prev_depth_im,
         self.curr_depth_im) = (self.curr_depth_im,
                                img)
        (self.curr_count,
         self.prev_count) = (img_count,
                             self.curr_count)
        (self.prev_patch,
         self.curr_patch) = (self.curr_patch,
                             hand_patch)
        (self.prev_patch_pos,
         self.curr_patch_pos) = (self.curr_patch_pos,
                                 hand_patch_pos)
        return

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

    def plot_hog(self, ghog_features):
        hog_hist, hog_bins = ghog_features
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
                 train_max_iter=2, des_dim=None):
        self.dictionaries = SparseDictionaries(des_dim=des_dim)
        self.des_dim = des_dim
        self.actions = Actions()
        self.log_lev = log_lev
        LOG.setLevel(log_lev)
        self.train_max_im_num = train_max_im_num
        self.train_max_iter = train_max_iter

    def add_action(self, depthdata=None,
                   masks_needed=True,
                   masksdata=None,
                   use_dexter=False,
                   visualize=False,
                   for_testing=False,
                   isstatic=False,
                   max_act_samples=None,
                   feature_params=None):
        '''
        actions.add_action alias
        '''
        res = self.actions.add_action(self.dictionaries,
                                      depthdata,
                                      masks_needed,
                                      masksdata,
                                      use_dexter,
                                      visualize_=visualize,
                                      for_testing=for_testing,
                                      isstatic=isstatic,
                                      max_act_samples=max_act_samples,
                                      feature_params=feature_params)
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
            LOG.info('Adding action..')
            self.actions.add_action(self.dictionaries,
                                    depthdata,
                                    masks_needed,
                                    masksdata,
                                    use_dexter)
        # Train dictionaries
        self.dictionaries.initialize(feat_num)
        frames_num = self.actions.\
            actions[act_num].features[0].shape[1]
        LOG.info('Frames number: ' + str(frames_num))
        train_im_num = min(frames_num, self.train_max_im_num)
        LOG.info('Frames to be used for training: ' + str(train_im_num))
        LOG.info('Creating dictionaries..')
        for count in range(feat_num):
            self.dictionaries.add_dict(self.actions, feat_num=count)
        train_actions = train_im_num
        final_errors = []
        try_count = 0
        iterat = range(feat_num)
        while True:
            LOG.info("Initializing inverse dictionaries")
            self.actions.actions[act_num].flush_sparse_features()
            for img_count in range(10):
                LOG.debug('Frame ' + str(img_count) + ' is being edited')
                sparse_features = []
                for feat_count in iterat:
                    coding = sc.SparseCoding(log_lev=self.log_lev,
                                             des_dim=self.des_dim)
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
            for feat_count in iterat:
                self.dictionaries.train(self.actions, feat_count,
                                        train_max_im_num=10)
            for iteration in range(self.train_max_iter):
                LOG.info('Epoch: ' + str(iteration))
                self.actions.actions[act_num].flush_sparse_features()
                LOG.info('Running Feature Sign Search Algorithm..')
                for img_count in range(train_im_num):
                    LOG.debug('Frame ' + str(img_count) + ' is being edited')
                    sparse_features = []
                    for feat_count in iterat:
                        coding = sc.SparseCoding(log_lev=self.log_lev,
                                                 des_dim=self.des_dim)
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
                LOG.info('Training Dictionaries..')
                for feat_count in iterat:
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
            LOG.info('Training is completed with final errors:\n' +
                     str(final_errors))
            try_count += 1
            if np.max(final_errors) > co.CONST['max_dict_error']:
                if try_count > co.CONST['max_retries']:
                    LOG.warning('Training has high final error but'+
                                ' reached maximum retries')
                    break
                LOG.warning('Bad inverse dictionary initialization and the'
                            + ' final error is too high, retraining..'+
                            '(max_dict_error can be changed in config.yaml '
                            'if needed)')
                iterat = []
                for count, error in enumerate(final_errors):
                    if error > co.CONST['max_dict_error']:
                        iterat.append(count)
                        self.dictionaries.flush(count)
                final_erros = []
            else:
                break
        if save_trained:
            self.dictionaries.save()
        return(self.dictionaries.dicts)
