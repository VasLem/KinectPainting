import sys
import warnings
import glob
from math import pi
import numpy as np
from numpy.linalg import pinv
import cv2
import class_objects as co
import sparse_coding as sc
import hand_segmentation_alg as hsa

# Kinect Intrinsics
'''
PRIM_X = 479.75
PRIM_Y = 269.75
FLNT = 540.68603515625
'''
# Senz3d Intrinsics
PRIM_X = 317.37514566554989
PRIM_Y = 246.61273826510859
FLNT = 595.333159044648


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
    gradx, grady = np.gradient(patch)
    return np.arctan(grady, gradx)  # returns values 0 to pi


def prepare_dexter_im(img):
    '''
    Compute masks for images
    '''
    binmask = img < 6000
    mask = np.zeros_like(img)
    mask[binmask] = img[binmask]
    mask = mask / (np.max(mask)).astype(float)
    contours = cv2.findContours(
        (binmask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    contours_area = [cv2.contourArea(contour) for contour in contours]
    hand_contour = contours[np.argmax(contours_area)].squeeze()
    hand_patch = mask[np.min(hand_contour[:, 1]):np.max(hand_contour[:, 1]),
                      np.min(hand_contour[:, 0]):np.max(hand_contour[:, 0])]
    med_filt = np.median(hand_patch[hand_patch != 0])
    binmask[np.abs(mask - med_filt) > 0.2] = False
    hand_patch[np.abs(hand_patch - med_filt) > 0.2] = 0
    hand_patch *= 256
    hand_patch = hand_patch.astype(np.uint8)
    hand_patch_pos = np.array(
        [np.min(hand_contour[:, 1]), np.min(hand_contour[:, 0])])
    return ((256 * mask).astype(np.uint8)) * binmask,\
        hand_patch, hand_patch_pos


class SpaceHistogram(object):
    '''
    Create Histograms for 3DHOG and GHOF
    '''

    def __init__(self):
        self.binarized_space = []
        self.bin_size = 0

    def binarize_3d(self):
        '''
        Initialize Histrogram for 3DHOF
        '''
        b_x = np.linspace(-1.0, 1.0, self.bin_size)
        b_y = np.linspace(-1.0, 1.0, self.bin_size)
        b_z = np.linspace(-1.0, 1.0, self.bin_size)
        self.binarized_space = [b_x, b_y, b_z]

    def binarize_1d(self):
        '''
        Initialize Histogram for GHOG
        '''
        self.binarized_space = [np.linspace(0.0, pi, self.bin_size)]

    def hist_data(self, sample):
        '''
        Compute N-D histograms
        '''
        return np.histogramdd(sample, self.binarized_space, normed=True)[0].ravel()


class Action(object):
    '''
    Class to hold an action
    '''

    def __init__(self):
        self.features = None
        self.sparse_features = None

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
                   features_extract=None,
                   depthdata=None,
                   masksdata=None,
                   use_dexter=True):
        '''
        features_extract= FeatureExtraction Class
        dictionaries= SparseDictionaries Class
        depthdata= (Directory with depth frames) OR (list of depth frames)
        masksdata=>Only used when use_dexter is False=>
            (Directory with hand masks) OR (list of hand masks)
        use_dexter= True if Dexter 1 TOF Dataset is used
        '''
        checktypes([dictionaries, features_extract], [
            SparseDictionaries, FeatureExtraction])
        if depthdata is None:
            raise Exception("Depth data frames are at least  needed")
        self.actions.append(Action())
        if not use_dexter:
            if isinstance(masksdata, str):
                masks = [cv2.imread(filename, 0) for filename
                         in glob.glob(masksdata + '/*.png')]
            else:
                masks = masksdata[:]
        if isinstance(depthdata, str):
            imgs = [cv2.imread(filename, -1) for filename
                    in glob.glob(depthdata + '/*.png')]
        else:
            imgs = depthdata[:]
        for img_count, img in enumerate(imgs):
            if use_dexter:
                mask, hand_patch, hand_patch_pos = prepare_dexter_im(
                    img)
            else:
                hand_patch, hand_patch_pos = hsa.main_process(masks[img_count])
                if hand_patch.shape[1] == 0:
                    warnings.warn('Problem with frame'+str(img_count))
                    hsa.main_process(masks[img_count], display=1)
                mask = img * (masks[img_count] > 0)
            features_extract.update(mask, img_count, hand_patch, hand_patch_pos)
            # Extract Features
            features = features_extract.extract_features()
            # Save action to actions object
            if features is not None:
                if len(dictionaries.dicts) == 0:
                    self.actions[-1].add_features(features=features)
                else:
                    sparse_features = []
                    for feat_num, feature in enumerate(features):
                        sparse_features.append(np.dot(dictionaries.inv_dicts[feat_num],
                                                      feature))

                    self.actions[-1].add_features(features=features,
                                                  sparse_features=sparse_features)
        return


class SparseDictionaries(object):
    '''
    Class to hold sparse coding dictionaries
    '''
    def __init__(self):
        self.inv_dicts = []
        self.dicts = []
        self.initialized = True

    def train(self, actions, feat_num, act_num=0, bmat=None, final_iter=False):
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
            act_num].features[feat_num].copy()
        self.inv_dicts[feat_num].out_features = (actions.actions[act_num].
                                                 sparse_features[feat_num].copy())
        if bmat is not None:
            self.inv_dicts[feat_num].bmat = bmat.copy()
        self.inv_dicts[feat_num].bmat = self.inv_dicts[
            feat_num].dictionary_training()
        if final_iter:
            self.dicts.append(pinv(self.inv_dicts[feat_num].bmat))

    def initialize(self, total_features_num):
        '''
        total_features_num is the total features types number of the actions
        '''
        self.dicts = []
        checktypes([total_features_num], [int])
        self.inv_dicts = []
        for _ in range(total_features_num):
            self.inv_dicts.append(sc.FeatureSignSearch())
        self.initialized = True

    def add_dict(self, actions=None, feat_dim=0, des_dim=0,
                 act_num=0, feat_num=0):
        '''
        actions=Actions class
        feat_num=Position where dict will be put
        feat_dim=Current features dimension
        des_dim=Desired features dimension
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
        if des_dim == 0:
            des_dim = co.CONST['des_dim']
        if feat_num == -1:
            warnings.warn('feat_num should be set when feat_dim is not, ' +
                          'else dictionary for feature 0 will be ovewritten')
            feat_num = 0
        if feat_dim == 0:
            checktypes([actions], [Actions])
            if len(actions.actions) == 0:
                raise Exception('Actions should have at least ' +
                                'one entry, or set feat_dim')
            self.inv_dicts[feat_num].initialise_vars(actions.actions[act_num].
                                                     features[feat_num].shape[0], des_dim)
        else:
            self.inv_dicts[feat_num].initialise_vars(feat_dim, des_dim)


class FeatureExtraction(object):
    '''
    Features computation class
    '''

    def __init__(self):
        self.features = np.zeros(0)
        self.prev_projection = np.zeros(0)
        self.curr_projection = np.zeros(0)
        self.roi = np.zeros(0)
        self.prev_patch = None
        self.curr_patch = None
        self.prev_patch_pos = None
        self.curr_patch_pos = None
        self.prev_count = 0
        self.curr_count = 0
        self.prev_depth_im = np.zeros(0)
        self.curr_depth_im = np.zeros(0)
        self.hofhist = SpaceHistogram()
        self.hoghist = SpaceHistogram()

    def compute_scene_flow(self, prev_depth_im, curr_depth_im):
        '''
        Computes scene flow for 3DHOF
        '''
        prev_hand_patch = prev_depth_im[self.roi[0, 0]:self.roi[0, 1],
                                        self.roi[1, 0]:self.roi[1, 1]]
        curr_hand_patch = curr_depth_im[self.roi[0, 0]:self.roi[0, 1],
                                        self.roi[1, 0]:self.roi[1, 1]]
        nonzero_mask = prev_hand_patch + curr_hand_patch
        yx_coords = (find_nonzero(nonzero_mask.astype(np.uint8)) -
                     np.array([[PRIM_Y, PRIM_X]]))
        prev_z_coords = prev_hand_patch[nonzero_mask > 0][:, None] / 255.0
        curr_z_coords = curr_hand_patch[nonzero_mask > 0][:, None] / 255.0
        dz_coords = curr_z_coords - prev_z_coords
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
        if len(self.hofhist.binarized_space) == 0:
            self.hofhist.bin_size = 4
            self.hofhist.binarize_3d()
        disp = self.compute_scene_flow(prev_depth_im, curr_depth_im)
        disp_norm = np.sqrt((disp[:, 0] * disp[:, 0] + disp[:, 1] *
                             disp[:, 1] + disp[:, 2] * disp[:, 2]))[:, None]
        disp_norm[disp_norm == 0] = 1
        disp /= disp_norm
        return self.hofhist.hist_data(disp)

    def ghog(self, depth_im):
        '''
        Compute GHOG features
        '''
        im_patch = depth_im[self.roi[0, 0]:self.roi[0, 1],
                            self.roi[1, 0]:self.roi[1, 1]]
        if len(self.hoghist.binarized_space) == 0:
            self.hoghist.bin_size = 9
            self.hoghist.binarize_1d()
        return self.hoghist.hist_data(grad_angles(im_patch).ravel())

    def extract_features(self):
        '''
        Returns 3DHOF and GHOG
        '''
        if self.prev_patch is None or \
           self.curr_count - self.prev_count > co.CONST['min_frame_count_diff']:
            return None
        self.find_roi(self.prev_patch, self.curr_patch,
                      self.prev_patch_pos, self.curr_patch_pos)
        hof_features = self.hof3d(
            self.prev_depth_im, self.curr_depth_im)
        hog_features = self.ghog(self.curr_depth_im)
        return hof_features, hog_features

    def update(self, armmask_uint8, count, hand_patch,
               hand_patch_pos):
        '''
        Update frames
        '''
        (self.prev_depth_im,
         self.curr_depth_im) = (self.curr_depth_im,
                                armmask_uint8)
        (self.curr_count,
         self.prev_count) = (count,
                             self.curr_count)
        (self.prev_patch,
         self.curr_patch) = (self.curr_patch,
                             hand_patch)
        (self.prev_patch_pos,
         self.curr_patch_pos) = (self.curr_patch_pos,
                                 hand_patch_pos)


class ActionRecognition(object):
    '''
    Class to hold everything about action recognition
    '''

    def __init__(self):
        self.features = FeatureExtraction()
        self.dictionaries = SparseDictionaries()
        self.actions = Actions()

    def add_action(self, depthdata=None,
                   masksdata=None,
                   use_dexter=True):
        '''
        actions.add_action alias
        '''
        self.actions.add_action(self.dictionaries,
                                self.features,
                                depthdata,
                                masksdata,
                                use_dexter)

    def train_sparse_dictionaries(self, act_num=None,
                                  depthdata=None,
                                  masksdata=None,
                                  use_dexter=True,
                                  iterations=3,
                                  print_info=False):
        '''
        Add Dexter 1 TOF Dataset or depthdata + binarymaskdata and
        set use_dexter to False (directory with .png or list accepted)
        '''
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
            raise Exception("Path/List of frames masks data is at least needed" +
                            ' because use_dexter flag is set to False')

        if depthdata is not None:
            if print_info:
                print 'Adding action..'
            self.actions.add_action(self.dictionaries,
                                    self.features,
                                    depthdata,
                                    masksdata,
                                    use_dexter)
        # Train dictionaries
        self.dictionaries.initialize(feat_num)
        if print_info:
            print 'Creating dictionaries..'
        for count in range(feat_num):
            self.dictionaries.add_dict(self.actions, feat_num=count,
                                       des_dim=100)
        frames_num=self.actions.\
                    actions[act_num].features[0].shape[1]
        if print_info:
            print 'Frames number:',frames_num
        for iteration in range(iterations):
            if print_info:
                print 'Epoch:',iteration
            self.actions.actions[act_num].flush_sparse_features()
            if print_info:
                print 'Running Feature Sign Search Algorithm..'

            for img_count in range(frames_num):
                sparse_features = []
                for feat_count in range(feat_num):

                    coding = sc.FeatureSignSearch()
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
            if print_info:
                print 'Training Dictionaries..'
            for feat_count in range(feat_num):
                if iteration == iterations - 1:
                    self.dictionaries.train(self.actions, feat_count,
                                            final_iter=True)
                else:
                    self.dictionaries.train(self.actions, feat_count)
