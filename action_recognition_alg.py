import numpy as np
import cv2
from math import pi
import class_objects as co
import sparse_coding as sc
import glob
import hand_segmentation_alg as hsa
import subprocess
import inspect
import sys

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


def checktypes(objects,Classes):
    if not all([isinstance(obj,instance) for
                obj,instance in zip(objects,Classes)]):
        try:
            f=sys._getframe(1) ;
            print 'Type Error in line',f.f_lineno
            print getattr(f.f_locals['self'].__class__,f.f_code.co_name).__doc__
        finally:
             del f
        raise SystemError

def find_nonzero(arr):
    return np.fliplr(cv2.findNonZero(arr).squeeze())

class SpaceHistogram(object):

    def __init__(self):
        self.binarized_space = []
        self.bin_size = 0

    def binarize_3d(self):
        b_x = np.linspace(-1.0, 1.0, self.bin_size)
        b_y = np.linspace(-1.0, 1.0, self.bin_size)
        b_z = np.linspace(-1.0, 1.0, self.bin_size)
        self.binarized_space = [b_x, b_y, b_z]

    def binarize_1d(self):
        self.binarized_space = [np.linspace(0.0, pi, self.bin_size)]

    def hist_data(self, sample):
        return np.histogramdd(sample, self.binarized_space, normed=True)[0].ravel()


class Action(object):

    def __init__(self):
        self.hof_features = None
        self.hog_features = None
        self.sparse_hof_features = None
        self.sparse_hog_features = None

    def add_features(self, hof_features, hog_features, sparse_hof_features=None,
                     sparse_hog_features=None):
        if self.hog_features is None:
            self.hof_features = np.atleast_2d(hof_features).T
            self.hog_features = np.atleast_2d(hog_features).T
            self.sparse_hof_features = np.atleast_2d(sparse_hof_features).T
            self.sparse_hog_features = np.atleast_2d(sparse_hog_features).T
        else:
            self.hof_features = np.concatenate((self.hof_features,
                                                np.atleast_2d(hof_features).T),
                                               axis=1)
            self.hog_features = np.concatenate((self.hog_features,
                                                np.atleast_2d(hog_features).T),
                                               axis=1)
            self.sparse_hof_features = np.concatenate((self.sparse_hof_features,
                                                       np.atleast_2d(sparse_hof_features).T),
                                                      axis=1)
            self.sparse_hog_features = np.concatenate((self.sparse_hog_features,
                                                       np.atleast_2d(sparse_hog_features).T),
                                                      axis=1)

    def flush_sparse_features(self):
        self.sparse_hof_features = None
        self.sparse_hog_features = None

    def update_sparse_features(self, sparse_hof_features, sparse_hog_features):
        if self.sparse_hog_features is None:
            self.sparse_hof_features = np.atleast_2d(sparse_hof_features)
            self.sparse_hog_features = np.atleast_2d(sparse_hog_features)
        else:
            self.sparse_hof_features = np.concatenate((self.sparse_hof_features,
                                                       np.atleast_2d(sparse_hof_features).T),
                                                      axis=1)
            self.sparse_hog_features = np.concatenate((self.sparse_hog_features,
                                                       np.atleast_2d(sparse_hog_features).T),
                                                      axis=1)


class Actions(object):

    def __init__(self):
        self.actions = []

    def add_action(self, dictionaries=None,
                   features=None,
                   depthdata=None,
                   masksdata=None,
                   use_dexter=True,
                   sparse_initialize=False):
        '''
        features= FeatureExtraction Class
        dictionaries= Sparse_Dictionaries Class
        depthdata= (Directory with depth frames) OR (list of depth frames)
        masksdata=>Only used when use_dexter is False=>
            (Directory with hand masks) OR (list of hand masks)
        use_dexter= True if Dexter 1 TOF Dataset is used
        sparse_initialize=True if initialisation of sparse features is also needed
        '''
        checktypes([dictionaries,features],[FeatureExtraction,Sparse_Dictionaries])
        if depthdata is  None:
            print "Depth data frames are at least  needed"
            exit()
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
        hof_bmat = None
        hog_bmat = None
        for img_count, img in enumerate(imgs):
            if use_dexter:
                mask, hand_patch, hand_patch_pos = features.prepare_dexter_im(img)
            else:
                hand_patch, hand_patch_pos = hsa.main_process(masks[img_count])
                if hand_patch.shape[1] == 0:
                    print 'Problem with frame', img_count
                    hsa.main_process(masks[img_count], display=1)
                mask = img * (masks[img_count] > 0)
            features.update(mask, img_count, hand_patch, hand_patch_pos)
            # Extract Features
            hof, hog = features.extract_features()
            # Save action to actions object
            if hof.shape[0] > 0:
                if dictionaries.hof.bmat is None and\
                   not sparse_initialize:
                    self.actions[-1].add_features(hof, hog)
                else:
                    if not sparse_initialize:
                        sparse_hof = np.dot(dictionaries.hof.bmat, hof)
                        sparse_hog = np.dot(dictionaries.hog.bmat, hog)
                    else:
                        hof_coding = sc.FeatureSignSearch()
                        hog_coding = sc.FeatureSignSearch()
                        # Initialise sparse features
                        hof_coding.feature_sign_search_algorithm(hof[:, None].astype(float),
                                                                 init_bmat=hof_bmat)
                        hog_coding.feature_sign_search_algorithm(hog[:, None].astype(float),
                                                                 init_bmat=hog_bmat)
                        if img_count == 0:
                            hof_bmat = hof_coding.bmat.copy()
                            hog_bmat = hog_coding.bmat.copy()

                    self.actions[-1].add_features(hof, hog,
                                                     sparse_hof,
                                                     sparse_hog)
        return hof_bmat, hog_bmat


class Sparse_Dictionaries(object):

    def __init__(self):
        self.hof = sc.FeatureSignSearch()
        self.hog = sc.FeatureSignSearch()

    def train_hof(self, actions, act_num, bmat=None):
        '''
        actions=Actions Class
        act_num=int
        bmat is not needed if Sparse_Dictionaries.hof.bmat is already initialised
        '''
        checktypes([actions,act_num],[Actions,int])
        self.hof.inp_features = actions.actions[act_num].hof_features.copy()
        self.hof.out_features = actions.actions[
            act_num].sparse_hof_features.copy()
        if bmat is not None:
            self.hof.bmat = bmat.copy()
        self.hof.bmat = self.hof.dictionary_training()

    def train_hog(self, actions, act_num, bmat=None):
        '''
        actions=Actions Class
        act_num=int
        bmat is not needed if Sparse_Dictionaries.hof.bmat is already initialised
        '''
        checktypes([actions,act_num],[Actions,int])
        self.hog.inp_features = actions.actions[act_num].hog_features.copy()
        self.hog.out_features = actions.actions[
            act_num].sparse_hog_features.copy()
        if bmat is not None:
            self.hog.bmat = bmat.copy()
        self.hog.bmat = self.hog.dictionary_training()

class FeatureExtraction(object):

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

    def compute_scene_flow(self, prev_depth_im, curr_depth_im):
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
        YX_coords = yx_coords * dz_coords / FLNT
        return np.concatenate((YX_coords,
                               dz_coords), axis=1)

    def find_roi(self, prev_patch, curr_patch, prev_patch_pos, curr_patch_pos):
        '''
        print 'roi'
        print '\t prev_patch shape',prev_patch.shape
        print '\t curr_patch.shape',curr_patch.shape
        '''
        self.roi = np.array([[
            min(prev_patch_pos[0], curr_patch_pos[0]),
            max((prev_patch.shape[0] + prev_patch_pos[0],
                 curr_patch.shape[0] + curr_patch_pos[0]))],
            [min(prev_patch_pos[1], curr_patch_pos[1]),
             max(prev_patch.shape[1] + prev_patch_pos[1],
                 curr_patch.shape[1] + curr_patch_pos[1])]])
        '''
        print '\t roi shape',self.roi[0,1]-self.roi[0,0],\
                self.roi[1,1]-self.roi[1,0]
        '''

    def hof3d(self, prev_depth_im, curr_depth_im):
        if len(hofhist.binarized_space) == 0:
            hofhist.bin_size = 4
            hofhist.binarize_3d()
        disp = self.compute_scene_flow(prev_depth_im, curr_depth_im)
        disp_norm = np.sqrt((disp[:, 0] * disp[:, 0] + disp[:, 1] *
                             disp[:, 1] + disp[:, 2] * disp[:, 2]))[:, None]
        disp_norm[disp_norm == 0] = 1
        disp /= disp_norm
        return hofhist.hist_data(disp)

    def grad_angles(self, patch):
        gradx, grady = np.gradient(patch)
        return np.arctan(grady, gradx)  # returns values 0 to pi

    def ghog(self, depth_im):
        im_patch = depth_im[self.roi[0, 0]:self.roi[0, 1],
                            self.roi[1, 0]:self.roi[1, 1]]
        if len(hoghist.binarized_space) == 0:
            hoghist.bin_size = 9
            hoghist.binarize_1d()
        return hoghist.hist_data(self.grad_angles(im_patch).ravel())

    def extract_features(self):
        if self.prev_patch is None or \
           self.curr_count - self.prev_count > co.CONST['min_frame_count_diff']:
            return np.zeros(0), np.zeros(0)
        self.find_roi(self.prev_patch, self.curr_patch,
                               self.prev_patch_pos, self.curr_patch_pos)
        roi = self.curr_depth_im[self.roi[0, 0]:self.roi[0, 1],
                                 self.roi[1, 0]:self.roi[1, 1]]
        '''
        cv2.imshow('test2',roi/np.max(roi).astype(float))
        cv2.waitKey(0)
        '''
        hof_features = self.hof3d(
            self.prev_depth_im, self.curr_depth_im)
        hog_features = self.ghog(self.curr_depth_im)
        return hof_features, hog_features

    def update(self, armmask_uint8, count, hand_patch,
               hand_patch_pos):
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

    def __init__(self):
        self.features = FeatureExtraction()
        self.dictionaries=Sparse_Dictionaries()
        self.actions=Actions()

    
    def add_action(self,depthdata=None,
                  masksdata=None,
                  use_dexter=True):
        print type(self.dictionaries)
        self.actions.add_action(self.dictionaries,
                                self.features,
                                 depthdata,
                                 masksdata,
                                 use_dexter,
                                 sparse_initialize=False)
        

    def train_sparse_dictionaries(self, depthdata=None,
                                  masksdata=None,
                                  use_dexter=True, action_num=0,
                                  retrain=True):
        '''
        Add Dexter 1 TOF Dataset or depthdata + binarymaskdata and
        set use_dexter to False (directory with .png or list accepted)
        '''
        if retrain:
            self.dictionaries.hof = sc.FeatureSignSearch()
            self.dictionaries.hog = sc.FeatureSignSearch()
        if depthdata == None and len(self.actions.actions) == 0:
            print "Pathname of frames depth data is at least needed" +\
                ' because action data is empty'
            exit()

        hof_bmat, hog_bmat = self.actions.add_action(self.dictionaries,
                                                     self.features,
                                                     depthdata,
                                                     masksdata,
                                                     use_dexter,
                                                     sparse_initialize=True)
        self.dictionaries.hof.bmat = hof_bmat
        self.dictionaries.hog.bmat = hog_bmat

        # Train dictionaries
        hof = self.actions.actions[action_num].hof_features
        hog = self.actions.actions[action_num].hog_features
        max_iter = 3

        for iteration in range(max_iter):
            self.dictionaries.train_hof(self.actions,0)
            self.dictionaries.train_hog(self.actions,0)
            if iteration < max_iter - 1:
                self.actions.actions[action_num].flush_sparse_features()
                for img_count in range(self.actions.actions[0].
                                       hog_features.shape[1]):
                    hof_coding = sc.FeatureSignSearch()
                    hog_coding = sc.FeatureSignSearch()
                    hof_coding.feature_sign_search_algorithm(hof[:, None].astype(float),
                                                             init_bmat=self.dictionaries.hof.bmat.copy())
                    hog_coding.feature_sign_search_algorithm(hog[:, None].astype(float),
                                                             init_bmat=self.dictionaries.hog.bmat.copy())
                    self.actions.actions[action_num].update_sparse_features(hof_coding.out_features,
                                                                       hog_coding.out_features)
        self.dictionaries.hof.bmat = np.pinv(self.dictionaries.hof.bmat)
        self.dictionaries.hog.bmat = np.pinv(self.dictionaries.hog.bmat)
        self.actions.actions[0].sparse_hof_features = np.dot(self.dictionaries.hof,
                                                        self.actions.actions[0].hof_features)
        self.actions.actions[0].sparse_hog_features = np.dot(self.dictionaries.hog,
                                                        self.actions.actions[0].hog_features)

    def prepare_dexter_im(self, img):
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
        return ((256 * mask).astype(np.uint8))*binmask,\
            hand_patch, hand_patch_pos




# pylint: disable=C0103
hofhist = SpaceHistogram()
hoghist = SpaceHistogram()
action_recog = ActionRecognition()
