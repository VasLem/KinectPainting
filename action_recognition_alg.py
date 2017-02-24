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

    def update_sparse_features(self, coders, max_act_samples=None,
                               fss_max_iter=None):
        '''
        Update sparse features using trained dictionaries
        '''
        self.flush_sparse_features()
        self.sparse_features = []
        for feature, coder in zip(self.features, coders):
            if max_act_samples is not None:
                self.sparse_features.append(coder.multicode(feature[:,
                                                                    :max_act_samples],
                                                                    max_iter=fss_max_iter))
            else:
                self.sparse_features.append(coder.multicode(feature,
                                                            max_iter=fss_max_iter))


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
        self.sparse_time = [[],[]]
        self.preproc_time = []

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
                   isderotated=False,
                   isstatic=False,
                   max_act_samples=None,
                   feature_params=None,
                   fss_max_iter = None,
                   derot_centers = None,
                   derot_angles = None):
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
            self.testing = Action()
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
            derot_info = False
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
                                    str.isdigit, os.path.basename(fil))))
                                samples_indices.append(ind)
                            except ValueError:
                                pass
                        elif filename.endswith('angles.txt'):
                            derot_info = True
                            fil = os.path.join(root, filename)
                            with open(fil, 'r') as inpf:
                                angles += map(float, inpf)
                        elif filename.endswith('centers.txt'):
                            derot_info = True
                            fil = os.path.join(root, filename)
                            with open(fil, 'r') as inpf:
                                for line in inpf:
                                    center = [
                                        float(num) for num
                                        in line.split(' ')]
                                    centers += [center]
            else:
                for root, dirs, filenames in os.walk(depthdata):
                    for filename in sorted(filenames):
                        if filename.endswith('.png'):
                            fil = os.path.join(root, filename)
                            files.append(fil)
                            action.sync.append(int(filter(
                                str.isdigit, os.path.basename(fil))))
                            samples_indices.append(0)
                        elif filename.endswith('angles.txt'):
                            derot_info = True
                            fil = os.path.join(root, filename)
                            with open(fil, 'r') as inpf:
                                angles += map(float, inpf)
                        elif filename.endswith('centers.txt'):
                            derot_info = True
                            fil = os.path.join(root, filename)
                            with open(fil, 'r') as inpf:
                                for line in inpf:
                                    center = [
                                        float(num) for num
                                        in line.split(' ')]
                                    centers += [center]
            action.samples_indices = np.array(samples_indices)
            action.angles = angles
            imgs = [cv2.imread(filename, -1) for filename
                    in files]
        else:
            if not isstatic and isderotated:
                if isinstance(depthdata, list) and len(depthdata)==2:
                    imgs_prev = depthdata[0]
                    imgs_next = depthdata[1]
                else:
                    raise Exception('If the data provided is for actions ' +
                                    'recognition and is derotated, then it ' +
                                    'must be a list of 2 arrays, the first ' +
                                    'including images derotated with next' +
                                    ' rotation frames and the second' +
                                    ' including images '
                                    + 'derotated with current rotation frames')
            else:
                imgs = depthdata[:]
        if not derot_info:
            if derot_angles is not None and derot_centers is not None:
                centers = derot_centers
                angles = derot_angles
                derot_info = True

        if not isderotated and derot_info:
            if isstatic:
                imgs = [co.pol_oper.derotate(imgs[count],
                                         angles[count],
                                         centers[count])
                    for count in range(len(imgs))]
            elif not isstatic:
                imgs_prev = [co.pol_oper.derotate(imgs[count],
                                         angles[count+1],
                                         centers[count+1])
                    for count in range(len(imgs)-1)]
                imgs_next = [co.pol_oper.derotate(imgs[count+1],
                                         angles[count+1],
                                         centers[count+1])
                    for count in range(len(imgs)-1)]
        feat_count=0
        img_len = len(imgs)
        for img_count, img in enumerate(imgs):
            #DEBUGGING
            #cv2.imshow('test',(imgs[img_count]%255).astype(np.uint8))
            #cv2.waitKey(10)
            t1 = time.time()
            if not isstatic and derot_info:
                if img_count == img_len-1:
                    break
                self.features_extract.update(imgs_prev[img_count], img_count-1, use_dexter,
                                    masks_needed, masks)
                self.features_extract.update(imgs_next[img_count], img_count,
                                             use_dexter, masks_needed, masks)
            else:
                self.features_extract.update(img, img_count, use_dexter,
                                             masks_needed, masks)
            t2 = time.time()
            self.preproc_time.append(t2-t1)

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
                    count=0
                    for sparse_coder, feature in zip(dictionaries.sparse_dicts,
                                                     features):
                        t1=time.time()
                        sparse_features.append(sparse_coder.code(feature,
                                                                 max_iter=fss_max_iter))
                        t2=time.time()
                        self.sparse_time[count].append(t2-t1)
                        count+=1
                    action.add_features(features=features,
                                                  sparse_features=sparse_features)
                feat_count+=1
                if max_act_samples is not None:
                    if feat_count==max_act_samples:
                        break
        #DEBUGGING
        #print np.min(self.sparse_time,axis=1) ,\
        #np.max(self.sparse_time,axis=1), np.mean(self.sparse_time,axis=1)\
        #        ,np.median(self.sparse_time,axis=1)
        if for_testing:
            if len(dictionaries.dicts) != 0:
                return self.testing.sparse_features, self.testing.sync
            else:
                return self.testing.features, self.testing.sync
        return 0

    def update_sparse_features(self, coders,
                               act_num='all', ret_sparse=False,
                               max_act_samples=None):
        '''
        Update sparse features for all Actions or a single one, specified by
        act_num.
        Requirement is that existent dictionaries have been trained
        '''
        if any([(coder is None) for coder in coders]):
            raise Exception('Dictionaries for existent features must' +
                            'have been trained before calling')
        if act_num == 'all':
            iter_quant = self.actions
        else:
            iter_quant = [self.actions[act_num]]
        for action in iter_quant:
            action.update_sparse_features(coders,
                                          max_act_samples=max_act_samples)
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
        self.sparse_dicts = []
        self.dicts = []
        self.initialized = True
        self.save_path = (os.getcwd() +
                          os.sep + 'saved_dictionaries.pkl')
        self.des_dim = des_dim

    def train(self, data, feat_num, bmat=None, display=0, min_iterations=10):
        '''
        actions=Actions class
        act_num=none: action postion inside actions.actions list
        feat_num=0: features position inside
                    actions.actions[act_num].features list
        final_iter=False: if True, it computes the dictionary matrix, which
                       converts input features to sparse
        '''
        self.sparse_dicts[feat_num].display = display
        checktypes([feat_num], [int])
        self.sparse_dicts[feat_num].train_sparse_dictionary(data,
                                                           sp_opt_max_iter=200,
                                                           init_bmat = bmat,
                                                            min_iterations=min_iterations)
        self.dicts[feat_num] = (pinv(self.sparse_dicts[feat_num].bmat))

    def initialize(self, total_features_num):
        '''
        total_features_num is the total features types number of the actions
        '''
        self.dicts = []
        checktypes([total_features_num], [int])
        self.sparse_dicts = []
        if not isinstance(self.des_dim, list):
            self.des_dim = total_features_num * [self.des_dim]
        for count in range(total_features_num):
            self.sparse_dicts.append(sc.SparseCoding(des_dim=self.des_dim[count],
                                                     name=str(count)))
            self.dicts.append(None)
        self.initialized = True


    def flush(self, dict_num='all'):
        '''
        Reinitialize all or one dictionary
        '''
        if dict_num == 'all':
            iter_quant = self.sparse_dicts
            iter_range = range(len(self.sparse_dicts))
        else:
            iter_quant = [self.sparse_dicts[dict_num]]
            iter_range = [dict_num]
        feat_dims = [] * (max(iter_range)+1)
        for count, inv_dict in zip(iter_range, iter_quant):
            feat_dims[count] = None
            try:
                feat_dim = inv_dict.bmat.shape[0]
                feat_dims[count] = feat_dim
            except AttributeError:
                feat_dims[count] = None
        for count in iter_range:
            if feat_dims[count] is not None:
                self.sparse_dicts[count].flush_variables()
                self.sparse_dicts[count].initialize(feat_dims[count])

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
            pickle.dump((self.sparse_dicts, self.dicts), output, -1)

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
        self.hof_edges = None
        self.ghog_edges = None
        self.feat_names = None

        if visualize_:
            self.view = FeatureVisualization()

    def visualize(self):
        self.view.plot_hof(self.hof_features, self.hof_edges)
        self.view.plot_hog(self.ghog_features, self.ghog_edges)
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
        nonzero_mask = (self.prev_roi_patch * self.curr_roi_patch) > 0
        if np.sum(nonzero_mask) == 0:
            return None
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
        if disp is None:
            return None
        disp_norm = np.sqrt((disp[:, 0] * disp[:, 0] + disp[:, 1] *
                             disp[:, 1] + disp[:, 2] * disp[:, 2]))[:, None]
        disp_norm[disp_norm == 0] = 1
        disp /= disp_norm.astype(float)
        hist, hof_edges = self.hofhist.hist_data(disp)
        self.hof_edges= hof_edges
        hist = hist/float(disp.size)
        return hist

    def pca_features(self, square_edge_size=None):
        '''
        Compute 3DXYPCA features
        '''
        if square_edge_size is None:
            square_edge_size = co.CONST['3DXYPCA_num']
        _, pca_along_2 = cv2.PCACompute(
            cv2.findNonZero(self.curr_patch.astype(np.uint8)).squeeze().
            astype(float),
            np.array([]), maxComponents=1)
        rot_angle = np.arctan2(pca_along_2[0][1], pca_along_2[0][0])
        patch = co.pol_oper.derotate(self.curr_patch, rot_angle,
                             (self.curr_patch.shape[0]/2,
                              self.curr_patch.shape[1]/2))
		#DEBUGGING
        #cv2.imshow('test',patch.astype(np.uint8))
        #cv2.waitKey(10)
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
        #DEBUGGING: I have switched cor_patch_res and the classification was
        #better, still I dont think I am doing it correctly
        _, pca_along_0 = cv2.PCACompute(cor_patch_res_1,np.array([]), maxComponents=1)
        _, pca_along_1 = cv2.PCACompute(cor_patch_res_0.T, np.array([]),
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
        hist, ghog_edges = self.hoghist.hist_data(grad_angles(im_patch))
        self.ghog_edges = ghog_edges
        #hist[0] = max(0, hist[0] - np.sum(im_patch==0))
        hist = hist / float(np.sum(hist))
        return hist

    def extract_features(self,isstatic=False, params=None, both=True):
        '''
        Returns 3DHOF and GHOG . isstatic to return 3DXYPCA. params is a number
        or (dictionary:not yet implemented/needed)
        '''
        t1 = time.time()
        self.find_roi(self.prev_patch, self.curr_patch,
                      self.prev_patch_pos, self.curr_patch_pos)
        if not isstatic:
            if self.feat_names is None:
                self.feat_names = ['3DHOF', 'GHOG']
            if self.prev_patch is None or \
               self.curr_count - self.prev_count > co.CONST['min_frame_count_diff']:
                return None
            self.hof_features = self.hof3d(
                self.prev_depth_im, self.curr_depth_im)
            if self.hof_features is None:
                return None
            self.ghog_features = self.ghog(self.curr_depth_im,params)
            t2 = time.time()
            self.extract_time.append(t2-t1)
            return self.hof_features.ravel(), self.ghog_features.ravel()
        else:
            if self.feat_names is None:
                self.feat_names = ['3DXYPCA']
            pca_features = self.pca_features(params)
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

    def plot_hog(self, ghog_features, ghog_edges):
        hog_hist = ghog_features
        hog_bins = ghog_edges
        width = 0.7 * (hog_bins[0][1] - hog_bins[0][0])
        center = (hog_bins[0][:-1] + hog_bins[0][1:]) / 2
        self.hog_plot.clear()
        self.hog_plot.bar(center, hog_hist, align='center', width=width)

    def plot_hof(self, hof_features, hof_edges):
        self.hist4d.draw(hof_features, hof_edges
                         , fig=self.fig, all_axes=self.hof_plots)

    def plot_3d_projection(self, roi, prev_roi_patch, curr_roi_patch):
        self.patches3d_plot.clear()
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

    def __init__(self, log_lev='INFO', des_dim=None):
        self.dictionaries = SparseDictionaries(des_dim=des_dim)
        self.des_dim = des_dim
        self.actions = Actions()
        self.log_lev = log_lev
        LOG.setLevel(log_lev)

    def add_action(self, depthdata=None,
                   masks_needed=True,
                   masksdata=None,
                   use_dexter=False,
                   visualize=False,
                   for_testing=False,
                   isstatic=False,
                   max_act_samples=None,
                   feature_params=None,
                   fss_max_iter=None):
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
                                      feature_params=feature_params,
                                      fss_max_iter=fss_max_iter)
        return res

    def train_sparse_dictionaries(self, act_num=None,
                                  depthdata=None,
                                  masks_needed=False,
                                  masksdata=None,
                                  use_dexter=False,
                                  save_trained=True,
                                  min_iterations=10):
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
                    save_trained],
                   [(int, type(None)), (list, str, type(None)), (bool, int),
                    (list, str, type(None)),
                    (bool, int), (bool, int)])
        if len(self.actions.actions) == 0:
            raise Exception('Run add_action first and then call ' +
                            'train_sparse_dictionaries')
        feat_num = len(self.actions.actions[0].features)

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
        for feat_count in range(feat_num):
            if act_num is None:
                data = np.concatenate(
                    [self.actions.actions[ind].features[feat_count] for ind in
                    range(len(self.actions.actions))], axis=1)
            else:
                data = self.actions.actions[act_num].features[feat_count]
            frames_num = data.shape[1]
            LOG.info('Frames number: ' + str(frames_num))
            LOG.info('Creating dictionaries..')
            self.dictionaries.train(data, feat_count, display=1,
                                    min_iterations=min_iterations)
        if save_trained:
            self.dictionaries.save()
        return(self.dictionaries.dicts)
