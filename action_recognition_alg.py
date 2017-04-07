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
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel(logging.INFO)
# Kinect Intrinsics
PRIM_X = 256.92
PRIM_Y = 204.67
FLNT = 365.98
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

def prepare_im(img, contour=None, square=False):
    '''
    <square> for display reasons, it returns a square patch of the hand, with
    the hand centered inside.
    '''
    if img is None:
        return None, None, None
    if contour is None:
        contours = cv2.findContours(
            (img).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours_area = [cv2.contourArea(contour) for contour in contours]
        contour = contours[np.argmax(contours_area)].squeeze()
    hand_contour = contour.squeeze()
    if hand_contour.size == 2:
        return None, None, None
    if square:
        edge_size = max(np.max(hand_contour[:,1]) - np.min(hand_contour[:,1]),
                        np.max(hand_contour[:,0]) - np.min(hand_contour[:,0]))
        center = np.mean(hand_contour,axis=0).astype(int)
        hand_patch = img[center[1] - edge_size/2:
                         center[1] + edge_size/2,
                         center[0] - edge_size/2:
                         center[0] + edge_size/2
                         ]
    else:
        hand_patch = img[np.min(hand_contour[:, 1]):np.max(hand_contour[:, 1]),
                     np.min(hand_contour[:, 0]):np.max(hand_contour[:, 0])]
    hand_patch_pos = np.array(
        [np.min(hand_contour[:, 1]), np.min(hand_contour[:, 0])])
    return hand_patch, hand_patch_pos, contour


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

    def __init__(self, parameters, name, coders=None):
        self.name = name
        self.parameters = parameters
        self.features = [[] for i in range(len(parameters['features']))]
        self.sparse_features = [[] for i in range(len(parameters['features']))]
        self.testing = parameters['testing']
        self.online = parameters['testing_params']['online']
        self.sparsecoded = parameters['sparsecoded']
        self.action_type = parameters['action_type']
        self.isdynamic = name not in parameters['passive_actions']
        self.trained_coders = True
        self.buffer_start_inds = []
        self.buffer_end_inds = []
        if self.sparsecoded:
            self.trained_coders = parameters['sparse_params']['trained_coders']

        if self.sparsecoded and coders is None:
            raise Exception('Sparse Coders required')

        if not self.action_type=='Passive':
            self.post_pca = parameters['dynamic_params']['post_PCA']
            self.post_pca_components = parameters['dynamic_params'][
                'post_PCA_components']
        self.coders = coders
        self.filter_from = [[] for i in range(len(parameters['features']))]
        self.bbuffer = [[] for i in range(len(parameters['features']))]
        if not self.action_type=='Passive' :
            self.buffer_size = parameters['dynamic_params']['buffer_size']
            try:
                self.filter_size = parameters['dynamic_params'][
                    'filter_window_size']
                self.buffer_confidence_tol = parameters['dynamic_params'][
                    'buffer_confidence_tol']
                self.filter_confidence_tol = parameters['dynamic_params'][
                    'filter_window_confidence_tol']
                self.post_pca = parameters['dynamic_params']['post_PCA']
                self.post_pca_components = parameters['dynamic_params'][
                    'post_PCA_components']
            except (KeyError,IndexError,TypeError):
                self.filter_size = None
                self.buffer_confidence_tol = None
                self.filter_confidence_tol = None
            self.buffer_exists = []
            self.pca_features = []
            if self.post_pca and not self.isdynamic:
                self.buffer_size = self.post_pca_components
        else:
            self.buffer_size = 1
        if self.sparsecoded and not self.trained_coders and not self.post_pca:
            self.buffer_size = None
        self.sync = []
        self.frames_inds = []
        self.samples_inds = []
        self.angles = None
        self.name = ''
    def add_buffer(self, buffer):
        '''
        <buffer> should have always the same size.
        <self.bbuffer> is a list of buffers. It can have a size limit, after which it
        acts as a buffer (useful for shifting window
        operations (filtering etc.))
        '''
        inp = buffer[:]
        #check buffer contiguousness
        if not self.action_type=='Passive':
            check_cont = np.all(np.abs(np.diff(self.frames_inds[-self.buffer_size:])) <=
                      self.buffer_size * self.buffer_confidence_tol)
            #check if buffer frames belong to the same sample, in case of training
            check_sam = self.testing or len(np.unique(
                        self.samples_inds[-self.buffer_size:])) == 1
        else:
            check_cont = True
            check_sam= True
        if check_cont and check_sam:
            for count in range(len(inp)):
                if not self.action_type=='Passive' and self.isdynamic and self.post_pca:
                    mean,inp[count] = cv2.PCACompute(
                        np.array(inp[count]),
                        np.array([]),
                        maxComponents=self.post_pca_components)
                    inp[count] = (np.array(inp[count]) + mean).ravel()
                    if self.sparsecoded and self.trained_coders:
                        inp[count]= self.coders[count].code(
                            inp[count]).ravel()
                    inp[count] = [inp[count]]
                if not self.action_type=='Passive':
                    self.filter_from[count] = (self.filter_from[count][-self.filter_size +
                                                          1:] + [True])
            if not self.action_type=='Passive':
                self.buffer_start_inds.append(self.frames_inds[-self.buffer_size])
                self.buffer_end_inds.append(self.frames_inds[-1])
                self.buffer_exists = self.buffer_exists[
                    -self.filter_size + 1:] + [True]
            for count in range(len(inp)):
                if not self.online:
                    self.bbuffer[count] += [inp[count][:]]
                else:
                    self.bbuffer[count] = [inp[count][:]]
        else:
            self.buffer_exists = self.buffer_exists[
                -self.filter_size + 1:] + [False]
        if not self.action_type=='Passive':
            if sum(self.buffer_exists) < len(self.filter_from[0]) * self.filter_confidence_tol:
                self.filter_from = [[] for i in range(len(inp))]
            return self.buffer_exists[-1]
        else:
            return True

    def add_features(self, features_group, features, update_buffer=False):
        if features is not None:
            for count, feature in enumerate(features):
                features_group[count].append(feature.ravel()[:])
                if self.buffer_size is not None:
                    del features_group[count][:-self.buffer_size]
            if (update_buffer):
                if len(features_group[0])<self.buffer_size:
                    self.buffer_exists.append(False)
                    return False
                else:
                    return self.add_buffer(features_group)
            return True
        else:
            return False


    def add_features_group(self, frame_ind, sample_ind=0, features=None,
                           sparse_features=None):
        '''
        Add frame features to action
        '''
        self.frames_inds.append(frame_ind)
        self.samples_inds.append(sample_ind)
        if not self.action_type=='Passive':
            try:
                self.buffer_exists += (self.frames_inds[-1] -
                                       self.frames_inds[-2] - 1) * [False]
            except (IndexError,AttributeError) :
                pass
        update_buffer = ((not self.sparsecoded)
                         or (self.post_pca and
                             self.sparsecoded))
        valid = self.add_features(self.features, features, update_buffer)
        sparse_time = None
        if self.sparsecoded and self.trained_coders and not self.post_pca:
            if sparse_features is None:
                if self.trained_coders:
                    sparse_valid, sparse_time = self.update_sparse_features(self.coders,
                                                              last_only=True)
            else:
                sparse_valid = self.add_features(self.sparse_features,
                                             sparse_features, self.sparsecoded)
            return sparse_valid, sparse_time
        else:
            return valid, None

    def retrieve_features(self):
        '''
        Returns a 3d numpy array, which has as first dimension the number of
        saved features sets inside <self.bbuffer>,
        as second dimension the number of frames from which each features set
        was taken and as third dimension the features shape of each frame. The
        features of each frame are a vector, including concatenated features
        coming from different descriptors.
        '''
        if len(self.features) > 1:
            return np.concatenate(tuple([np.array(feat_type_buffer)
                                         for feat_type_buffer in
                  self.bbuffer]), axis=2)
        else:
            res = np.atleast_3d(np.array(self.bbuffer).squeeze())
            return res

    def update_sparse_features(self, coders, max_act_samples=None,
                               fss_max_iter=None, last_only=False):
        '''
        Update sparse features using trained coders.
        If <last_only>, the sparse features list is not erased, but the new
        sparse features coming from the last entry of <self.features> are added
        to the list. 
        '''


        if not last_only:
            self.flush_sparse_features()
            self.sparse_features = []
            self.bbuffer = [[] for i in range(len(self.parameters['features']))]
            iterat = range(len(self.bbuffer[0]))
        else:
            iterat = [-1]
            sparse_time= []
        for count in iterat:
            sparse_features = []
            if last_only:
                sparse_time.append([])
            for type_count in range(len(self.features)):
                if last_only:
                    t1 = time.time()
                sparse_features.append(coders[type_count].code(
                                  self.features[type_count][count],
                                  max_iter=fss_max_iter))
                if last_only:
                    sparse_time[-1].append(time.time() - t1)

            valid = self.add_features(self.sparse_features,
                              sparse_features,
                              True)
        if last_only:
            return valid,sparse_time
        else:
            return None


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

    def __init__(self, parameters, coders=None, feat_filename=None):
        self.parameters = parameters
        self.sparsecoded = parameters['sparsecoded']
        self.actions = []
        self.coders= coders
        self.save_path = (os.getcwd() +
                          os.sep + 'saved_actions.pkl')
        self.features_extract = None
        self.sparse_time = []
        self.preproc_time = []
        self.extract_time = []
        self.features_db = None
        self.candid_d_actions = None
        self.feat_filename = feat_filename

    def remove_action(self, act_num):
        '''
        Remove an action using act_num. Set -1 to remove last
        added action, or 'all' to remove every action
        '''
        if act_num == 'all':
            self.actions = []
        else:
            self.actions = self.actions[act_num:] + self.actions[:act_num]

    def save_action_features_to_mem(self,filename,
                                    data):
        '''
        features_db has tree structure, with the following order:
            features type->list of instances dicts ->[params which are used to
            identify the features, data of each instance->actions]
        This order allows to search only once for each descriptor and get all
        actions corresponding to a matching instance, as it is assumed that
        descriptors are fewer than actions.
        <data> is a list of length same as the descriptors number
        '''
        if filename is None:
            return
        if self.candid_d_actions is None:
            self.candid_d_actions = [None]* len(self.parameters[
                'features'])
        for dcount,descriptor in enumerate(
            self.parameters['features']):
            if self.instance_ind[dcount] is None:
                feat_params = self.parameters['features_params'][descriptor]
                if not isinstance(self.features_db, dict):
                    self.features_db = {}
                if descriptor not in self.features_db:
                    self.features_db[descriptor] = []
                    self.features_db[descriptor].append(
                        {'params':feat_params,
                         'data':{}})
                self.instance_ind[dcount] = -1
            self.features_db[descriptor][
                self.instance_ind[dcount]]['data'][self.name] = data[dcount]
        co.file_oper.save_using_ujson(self.features_db, filename)


    def load_action_features_from_mem(self, filename):
        '''
        features_db has tree structure, with the following order:
            features type->list of instances dicts ->[params which are used to
            identify the features, data of each instance->actions]
        This order allows to search only once for each descriptor and get all
        actions corresponding to a matching instance, as it is assumed that
        descriptors are fewer than actions
        '''
        features_to_extract = self.parameters['features'][:]
        data = [None] * len(features_to_extract)
        if filename is None:
            return features_to_extract, data
        try:
            self.features_db = co.file_oper.load_using_ujson(
                filename)
        except Exception as e:
            pass
        self.instance_ind = [None] * len(features_to_extract)
        if self.features_db is not None:
            if self.candid_d_actions is None or not self.candid_d_actions:
                self.candid_d_actions = [None]*len(self.parameters['features'])
                for dcount,descriptor in enumerate(
                    self.parameters['features']):
                    try:
                        saved_instances = self.features_db[
                            descriptor]
                        feat_params = self.parameters['features_params'][
                            descriptor]
                        for inst_count,instance in enumerate(saved_instances):
                            found_instance = True
                            for key in feat_params:
                                if isinstance(feat_params[key],dict):
                                    if any([feat_params[key][sub_key]!=
                                            instance['params'][key][sub_key]
                                            for sub_key in feat_params[key]]):
                                        found_instance = False
                                        break
                                else:
                                    if feat_params[key] != instance['params'][key]:
                                        found_instance = False
                                        break
                            if found_instance:
                                self.instance_ind[dcount] = inst_count
                                self.candid_d_actions[dcount] = instance['data']
                    except KeyError:
                        pass
            '''
            Result is <self.candid_d_actions>, a list which holds matching
            instances of actions for each descriptor, or None if not found.
            '''
            for dcount,descriptor in enumerate(self.candid_d_actions):
                if self.candid_d_actions[dcount] is not None:
                    if self.name in self.candid_d_actions[dcount]:
                        data[dcount] = self.candid_d_actions[dcount][self.name]
                        features_to_extract.remove(self.parameters['features']
                                                   [dcount])

        return features_to_extract, data

    def add_action(self, data=None,
                   mv_obj_fold_name=None,
                   hnd_mk_fold_name=None,
                   masks_needed=True,
                   use_dexter=False,
                   visualize_=False,
                   for_testing=False,
                   isderotated=False,
                   action_type='Dynamic',
                   max_act_samples=None,
                   fss_max_iter=None,
                   derot_centers=None,
                   derot_angles=None,
                   name=None,
                   feature_extraction_method=None,
                   save=True,
                   feat_filename=None):
        '''
        parameters=dictionary having at least a 'features' key, which holds
            a sublist of ['3DXYPCA', 'GHOG', '3DHOF', 'ZHOF']. It can have a
            'features_params' key, which holds specific parameters for the
            features to be extracted.
        features_extract= FeatureExtraction Class
        data= (Directory with depth frames) OR (list of depth frames)
        use_dexter= True if Dexter 1 TOF Dataset is used
        visualize= True to visualize features extracted from frames
        for_testing = True if input data is testing data
        '''
        if data is None:
            raise Exception("Depth data frames are at least  needed")
        if isinstance(data, basestring) and name is None:
            name = os.path.basename(data)
        if save:
            if data[:(-len(name)-2)] == co.CONST['test_save_path']:
                feat_filename = name+'_feats'
            else:
                feat_filename = 'train_feats'
        self.name = name
        features_to_extract, feat_data = self.load_action_features_from_mem(
            feat_filename)
        found_feats = []
        found_sync = None
        for feat_datum in feat_data:
            if feat_datum is None:
                found_feats.append(None)
            else:
                found_feats.append(feat_datum[0])
                found_sync = feat_datum[1]
                found_buffer_start_inds = feat_datum[2]
                found_buffer_end_inds = feat_datum[3]
        if for_testing:
            self.testing = Action(self.parameters,name='test',coders=self.coders)
            action = self.testing
        else:
            self.actions.append(Action(self.parameters,name=name,coders=self.coders))
            action = self.actions[-1]
        if features_to_extract:
            self.features_extract = FeatureExtraction(parameters=self.parameters,
                                                      visualize_=visualize_)
            if masks_needed:
                if mv_obj_fold_name is None:
                    mv_obj_fold_name = co.CONST['mv_obj_fold_name']
                if hnd_mk_fold_name is None:
                    hnd_mk_fold_name = co.CONST['hnd_mk_fold_name']
            if isinstance(data, basestring):
                files = []
                masks = []
                samples_indices = []
                angles = []
                centers = []
                action.sync = []
                derot_info = False
                if (os.path.isdir(os.path.join(data, '0')) or
                        os.path.isdir(os.path.join(data, mv_obj_fold_name, '0'))):
                    action.name = os.path.basename(data)
                    for root, dirs, filenames in os.walk(data):
                        for filename in sorted(filenames):
                            fil = os.path.join(root, filename)
                            folder_sep = os.path.normpath(fil).split(os.sep)
                            if filename.endswith('.png'):
                                ismask = False
                                if masks_needed:
                                    ismask = folder_sep[-3] == hnd_mk_fold_name
                                par_folder = folder_sep[-2]
                                try:
                                    ind = int(par_folder)
                                    if ismask:
                                        masks.append(fil)
                                    else:
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
                    for root, dirs, filenames in os.walk(data):
                        folder_sep = os.path.normpath(root).split(os.sep)
                        for filename in sorted(filenames):
                            if filename.endswith('.png'):
                                fil = os.path.join(root, filename)
                                if (masks_needed and
                                        folder_sep[-2] == mv_obj_fold_name):
                                    masks.append(fil)
                                else:
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
                if masks_needed:
                    masks = [cv2.imread(filename, -1) for filename in masks]
            if not derot_info:
                if derot_angles is not None and derot_centers is not None:
                    centers = derot_centers
                    angles = derot_angles
                    derot_info = True

            img_len = len(imgs)
            for img_count, img in enumerate(imgs):
                if img_count > 0:
                    if samples_indices[
                            img_count] != samples_indices[img_count - 1]:
                        self.features_extract.reset()
                # DEBUGGING
                # cv2.imshow('test',(imgs[img_count]%255).astype(np.uint8))
                # cv2.waitKey(10)
                t1 = time.time()
                if isderotated:
                    angles.append([None])
                    centers.append([None])
                if not masks_needed:
                    masks.append([None])
                self.features_extract.update(imgs[img_count],
                                             action.sync[img_count],
                                             mask=masks[img_count],
                                             angle=angles[img_count],
                                             center=centers[img_count])
                t2 = time.time()
                self.preproc_time.append(t2 - t1)

                # Extract Features
                features = self.features_extract.extract_features()
                # Save action to actions object
                if features is not None:
                    if visualize_:
                        self.features_extract.visualize()
                    valid, sparse_time = action.add_features_group(
                        action.sync[img_count],
                        sample_ind=action.samples_indices[img_count],
                        features=features)
                    self.sparse_time += [sparse_time]
                    if max_act_samples is not None:
                        if img_count == max_act_samples:
                            break
            self.extract_time += self.features_extract.extract_time
            # DEBUGGING
            # print np.min(self.sparse_time,axis=1) ,\
            #np.max(self.sparse_time,axis=1), np.mean(self.sparse_time,axis=1)\
            #        ,np.median(self.sparse_time,axis=1)
        count = 0
        for feat_count in range(len(feat_data)):
            if feat_data[feat_count] is None:
                found_feats[feat_count] = action.bbuffer[count]
                count+=1
        count = 0
        for feat_count in range(len(feat_data)):
            if feat_data[feat_count] is None:
                feat_data[feat_count] = [np.array(action.bbuffer[count]).tolist()
                                         ,action.sync,
                                     action.buffer_start_inds,
                                     action.buffer_end_inds]
                count += 1
        self.save_action_features_to_mem(feat_filename, feat_data)
        action.bbuffer = found_feats
        if found_sync is not None:
            action.sync = found_sync
            action.buffer_start_inds = found_buffer_start_inds
            action.buffer_end_inds = found_buffer_end_inds
        if for_testing:
            if action_type=='Passive':
                return (self.testing.retrieve_features().squeeze(),
                        self.testing.sync)
            else:
                return (self.testing.retrieve_features(),
                        [self.testing.sync,
                         self.testing.buffer_start_inds,
                        self.testing.buffer_end_inds])

        return 0

    def update_sparse_features(self, coders,
                               act_num='all',
                               max_act_samples=None,
                               fss_max_iter=None,
                               last_only=False):
        '''
        Update sparse features for all Actions or a single one, specified by
        act_num.
        Requirement is that existent coders have been trained
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
                                          max_act_samples=max_act_samples,
                                          fss_max_iter=fss_max_iter,
                                          last_only=last_only)

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


class ActionsSparseCoding(object):
    '''
    Class to hold sparse coding coders
    '''

    def __init__(self, parameters):
        self.features = parameters['features']
        self.parameters = parameters
        self.sparse_dim = []
        try:
            for feat in self.features:
                self.sparse_dim.append(parameters['sparse_params'][feat])
        except (KeyError, TypeError):
            self.sparse_dim = [None] * len(self.features)
        self.sparse_coders = []
        self.codebooks = []
        self.initialized = True
        self.save_path = (os.getcwd() +
                          os.sep + 'saved_coders.pkl')

    def train(self, data, feat_count, bmat=None, display=0, min_iterations=10,
              init_traindata_num=200, incr_rate=2, sp_opt_max_iter=200,
              debug=False):
        '''
        feat_count: features position inside
                    actions.actions[act_num].features list
        '''
        try:
            self.sparse_coders[feat_count].display = display
        except:
            self.sparse_coders[feat_count] = sc.SparseCoding(
                                  sparse_dim=self.sparse_dim[feat_count],
                                  name=str(feat_count))
            self.sparse_coders[feat_count].display = display

        self.sparse_coders[feat_count].train_sparse_dictionary(data,
                                                           init_traindata_num=
                                                               init_traindata_num,
                                                           incr_rate=incr_rate,
                                                           sp_opt_max_iter=sp_opt_max_iter,
                                                           min_iterations=min_iterations)
        self.codebooks[feat_count] = (
            pinv(self.sparse_coders[feat_count].bmat))
        return 1
    def initialize(self):
        '''
        initialize / reset all codebooks that refer to the given <sparse_dim>
        and feature combination
        '''
        self.sparse_coders = []
        for count,feature in enumerate(self.features):
            self.sparse_coders.append(sc.SparseCoding(
                                  sparse_dim=self.sparse_dim[count],
                                  name=str(count)))
            self.codebooks.append(None)
        self.initialized = True

    def flush(self, feat_count='all'):
        '''
        Reinitialize all or one dictionary
        '''
        if feat_count == 'all':
            iter_quant = self.sparse_coders
            iter_range = range(len(self.features))
        else:
            iter_quant = [self.sparse_coders[feat_count]]
            iter_range = [feat_count]
        feat_dims = []
        for feat_count, inv_dict in zip(iter_range, iter_quant):
            feat_dims[feat_count] = None
            try:
                feat_dim = inv_dict.bmat.shape[0]
                feat_dims[feat_count] = feat_dim
            except AttributeError:
                feat_dims[feat_count]= None
        for feature in self.sparse_coders:
            if feat_dims[feature] is not None:
                self.sparse_coders[feat_count].flush_variables()
                self.sparse_coders[feat_count].initialize(feat_dims[feature])

    def save(self, save_dict=None, save_path=None):
        '''
        Save coders to file
        '''
        if save_dict is not None:
            for feat_count, feature in enumerate(self.features):
                if not self.parameters['dynamic_params']['post_PCA']:
                    save_dict[feature+' '+
                              str(self.sparse_dim[feat_count])] = \
                    self.sparse_coders[feat_count]
                else:
                    save_dict[feature+' '+
                              str(self.sparse_dim[feat_count])+
                              ' PCA ' +
                              str(self.parameters['dynamic_params'][
                              'post_PCA_components'])] = \
                    self.sparse_coders[feat_count]
            return
        if save_path is None:
            coders_path = self.save_path
        else:
            coders_path = save_path

        LOG.info('Saving Dictionaries to ' + coders_path)
        with open(coders_path, 'wb') as output:
            pickle.dump((self.sparse_coders, self.codebooks), output, -1)


def grad_angles(patch):
    '''
    Compute gradient angles on image patch for GHOG
    '''
    grady, gradx = np.gradient(patch.astype(float))
    ang = np.arctan2(grady, gradx)
    #ang[ang < 0] = ang[ang < 0] + pi

    return ang.ravel()  # returns values 0 to pi


class FeatureExtraction(object):
    '''
    Features computation class
    '''

    def __init__(self, parameters, visualize_=False,
                 features_to_extract=None):
        self.parameters = parameters
        self.with_hof3d = '3DHOF' in parameters['features']
        self.with_pca = '3DXYPCA' in parameters['features']
        self.with_ghog = 'GHOG' in parameters['features']
        self.with_zhof = 'ZHOF' in parameters['features']
        if features_to_extract is not None:
            self.with_hof3d = '3DHOF' in features_to_extract
            self.with_pca = '3DXYPCA' in features_to_extract
            self.with_ghog = 'GHOG' in features_to_extract
            self.with_zhof = 'ZHOF' in features_to_extract
        self.extracted_features = [None] * len(self.parameters['features'])
        if self.with_hof3d:
            self.hof3d_ind = self.parameters['features'].index('3DHOF')
        if self.with_ghog:
            self.ghog_ind = self.parameters['features'].index('GHOG')
        if self.with_pca:
            self.pca_ind = self.parameters['features'].index('3DXYPCA')
        if self.with_zhof:
            self.zhof_ind = self.parameters['features'].index('ZHOF')
        self.action_type = parameters['action_type']
        self.use_coders = parameters['sparsecoded']
        if self.with_hof3d or self.with_zhof:
            self.hof_bin_size = co.CONST[['3DHOF' if self.with_hof3d
                                          else 'ZHOF'][0]+'_bin_size']
        if self.with_ghog:
            self.ghog_bin_size = co.CONST['GHOG_bin_size']
        if self.with_pca:
            self.pca_resize_size = co.CONST['3DXYPCA_size']
        self.skeleton = None
        self.extract_time = []
        self.features = np.zeros(0)
        self.prev_projection = np.zeros(0)
        self.curr_projection = np.zeros(0)
        self.roi = np.zeros(0)
        self.roi_original = np.zeros(0)
        self.prev_patch = None
        self.curr_patch = None
        self.prev_patch_original = None
        self.curr_patch_original = None
        self.prev_roi_patch = None
        self.curr_roi_patch = None
        self.prev_roi_patch_original = None
        self.curr_roi_patch_original = None
        self.prev_patch_pos = None
        self.curr_patch_pos = None
        self.prev_patch_pos_original = None
        self.curr_patch_pos_original = None
        self.prev_count = 0
        self.curr_count = 0
        self.prev_depth_im = np.zeros(0)
        self.curr_depth_im = np.zeros(0)
        self.hofhist = SpaceHistogram()
        self.hoghist = SpaceHistogram()
        self.hand_contour = None
        self.fig = None
        self.hof_features = None
        self.zhof_features = None
        self.ghog_features = None
        self.hof_plots = None
        self.hog_plot = None
        self.hof_edges = None
        self.ghog_edges = None
        self.kernel = np.ones((5, 5), np.uint8)
        self.curr_full_depth_im = None
        self.prev_full_depth_im = None
        self.prev_cnt = None
        self.curr_cnt = None
        self.angle = None
        self.center = None
        self.hand_img = None

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

    def find_outliers(self, data, m=2.):
        '''
        median of data must not be 0
        '''
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev>0 else 0
        return s > m

    def compute_scene_flow(self, prev_depth_im, curr_depth_im):
        '''
        Computes scene flow for 3DHOF
        '''
        if prev_depth_im is None:
            return None
        roi = self.roi
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
        mask = (self.find_outliers(curr_z[y_new, x_new], 3)
                + self.find_outliers(prev_z[y_old, x_old], 3)) == 0
        y_new = y_new[mask]
        y_old = y_old[mask]
        x_new = x_new[mask]
        x_old = x_old[mask]
        princ_coeff = co.pol_oper.derotate_points(
            self.curr_depth_im,
            np.array([PRIM_Y - self.roi_original[0, 0],
                      PRIM_X - self.roi_original[0, 1]]),
            self.angle,
            self.center)
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
        #cv2.waitKey(10)
        dx = x_true_new - x_true_old
        dy = y_true_new - y_true_old
        dz = curr_z[y_new, x_new] - prev_z[y_old, x_old]
        return np.concatenate((dx.reshape(-1, 1),
                               dy.reshape(-1, 1),
                               dz.reshape(-1, 1)), axis=1)

    def z_flow(self, prev_depth_im, curr_depth_im):
        '''
        Computes vertical displacement to the camera, using static frame
        xy-coordinates and changing z ones.
        '''
        self.prev_roi_patch = prev_depth_im[
            self.roi[0, 0]:self.roi[0, 1],
            self.roi[1, 0]:self.roi[1, 1]].astype(float)
        self.curr_roi_patch = curr_depth_im[
            self.roi[0, 0]:self.roi[0, 1],
            self.roi[1, 0]:self.roi[1, 1]].astype(float)
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
        dz_outliers = self.find_outliers(dz_coords, 3.).ravel()
        dz_coords = dz_coords[dz_outliers==0]
        yx_coords = yx_coords[dz_outliers==0,:]
        yx_coords_in_space = (yx_coords * dz_coords / FLNT)
        return np.concatenate((yx_coords_in_space,
                               dz_coords), axis=1)
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

    def hof_compute(self, prev_depth_im, curr_depth_im, hof_bin_size=None,
                       simplified=False):
        '''
        Compute 3DHOF features, or ZHOF features if <simplified>
        '''
        if hof_bin_size is None:
            self.hofhist.bin_size = self.hof_bin_size
        else:
            self.hofhist.bin_size = hof_bin_size
        self.hofhist.range = [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]
        if not simplified:
            disp = self.compute_scene_flow(prev_depth_im, curr_depth_im)
        else:
            disp = self.z_flow(prev_depth_im, curr_depth_im)
        if disp is None:
            return None
        # print disp.max(axis=0), disp.min(axis=0)
        disp_norm = np.sqrt((disp[:, 0] * disp[:, 0] + disp[:, 1] *
                             disp[:, 1] + disp[:, 2] * disp[:, 2]))[:, None]
        disp_norm[disp_norm == 0] = 1
        disp = disp / disp_norm.astype(float)
        # print np.unique(np.around(disp,1))
        hist, hof_edges = self.hofhist.hist_data(disp)
        self.hof_edges = hof_edges
        hist = hist / np.sum(hist)
        return hist

    def pca_compute(self, resize_size=None):
        '''
        Compute 3DXYPCA features
        '''
        if resize_size is not None:
            self.pca_resize_size = resize_size
        _, pca_along_2 = cv2.PCACompute(
            cv2.findNonZero(self.curr_patch.astype(np.uint8)).squeeze().
            astype(float),
            np.array([]), maxComponents=1)
        rot_angle = np.arctan2(pca_along_2[0][1], pca_along_2[0][0])
        patch = co.pol_oper.derotate(self.curr_patch, rot_angle,
                                     (self.curr_patch.shape[0] / 2,
                                      self.curr_patch.shape[1] / 2))
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

    def ghog_compute(self, depth_im, ghog_bin_size=None):
        '''
        Compute GHOG features
        '''
        im_patch = self.curr_patch.astype(int)
        if ghog_bin_size is None:
            self.hoghist.bin_size = self.ghog_bin_size
        else:
            self.hoghist.bin_size = ghog_bin_size
        #DEBUGGING: added -pi (check grad_agnels too)
        self.hoghist.range = [[-pi, pi]]
        gradients = grad_angles(im_patch)
        hist, ghog_edges = self.hoghist.hist_data(gradients)
        self.ghog_edges = ghog_edges
        #hist[0] = max(0, hist[0] - np.sum(im_patch==0))
        hist = hist / float(np.sum(hist))
        return hist

    def extract_features(self):
        '''
        Returns Extracted Features
        '''
        #Remember to change list of features in __init__, if
        # features computation turn is changed
        t1 = time.time()
        if self.curr_patch is None:
            return None
        features = []
        if self.with_ghog:
            ghog_features = self.ghog_compute(self.curr_depth_im)
            self.extracted_features[self.ghog_ind] = ghog_features
        if self.with_hof3d or self.with_zhof:
            self.roi = self.find_roi(self.prev_patch, self.curr_patch,
                                     self.prev_patch_pos, self.curr_patch_pos)
            self.roi_original = self.find_roi(
                self.prev_patch_original, self.curr_patch_original,
                self.prev_patch_pos_original,
                self.curr_patch_pos_original)
            if self.prev_patch is None or \
               self.curr_count - self.prev_count > co.CONST['min_frame_count_diff']:
                return None
            if self.with_hof3d:
                hof_features = self.hof_compute(
                    self.prev_depth_im,
                    self.curr_depth_im)
                if hof_features is None:
                    return None
                self.extracted_features[self.hof3d_ind] = hof_features
            if self.with_zhof:
                zhof_features = self.hof_compute(
                    self.prev_depth_im,
                    self.curr_depth_im,
                    simplified=True)
                if zhof_features is None:
                    return None
                self.extracted_features[self.zhof_ind] = zhof_features
        if self.with_pca:
            pca_features = self.pca_compute()
            self.extracted_features[self.pca_ind] = pca_features

        t2 = time.time()
        self.extract_time.append(t2 - t1)
        return self.extracted_features


    def reset(self, visualize=False):
        self.__init__(self.parameters,visualize)
    def update(self, img, img_count, use_dexter=False, mask=None, angle=None,
               center=None, masks_needed=False, isderotated=False):
        '''
        Update frames
        '''

        if use_dexter:
            mask, hand_patch, hand_patch_pos = prepare_dexter_im(
                img)
        else:
            cnt = None
            try:
                if masks_needed and mask is None:
                    mask1 = cv2.morphologyEx(
                        img.copy(), cv2.MORPH_OPEN, self.kernel)
                    _, cnts, _ = cv2.findContours(mask1.astype(np.uint8),
                                                  cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_NONE)
                    cnts_areas = [cv2.contourArea(cnts[i]) for i in
                                  xrange(len(cnts))]
                    cnt = cnts[np.argmax(cnts_areas)]
                    if self.skeleton is None:
                        self.skeleton = hsa.FindArmSkeleton(img.copy())
                    if not self.skeleton.run(img, cnt,
                                         'longest_ray'):
                        return False
                    mask = self.skeleton.hand_mask
                    if mask is None:
                        return False
                    last_link = (self.skeleton.skeleton[-1][1] -
                                 self.skeleton.skeleton[-1][0])
                    angle = np.arctan2(
                        last_link[0], last_link[1])
                    center = self.skeleton.hand_start

                if not isderotated and angle is None:
                    raise Exception('mask is not None, derotation is True ' +
                                    'and angle and center are missing, ' +
                                    'cannot proceed with this combination')
                if self.with_hof3d:
                    (self.prev_full_depth_im,
                     self.curr_full_depth_im) = (self.curr_full_depth_im,
                                                 img)
                    imgs = [self.prev_full_depth_im,
                            img]
                    if self.prev_full_depth_im is None:
                        return False
                else:
                    imgs = [img]
                for img in imgs:
                    if np.sum(mask > 0) == 0:
                        return False
                    img = img * (mask > 0)
                    self.hand_img = img
                    if not isderotated:
                        if angle is not None and center is not None:
                            self.angle = angle
                            self.center = center
                            processed_img = co.pol_oper.derotate(
                                img,
                                angle, center)
                    else:
                        processed_img = img
                    hand_patch, hand_patch_pos, self.hand_contour = prepare_im(
                        processed_img)
                    # DEBUGGING
                    #cv2.imshow('test',((hand_patch)%255).astype(np.uint8))
                    #cv2.waitKey(10)
                    if hand_patch is None:
                        return False
                    (self.prev_depth_im,
                     self.curr_depth_im) = (self.curr_depth_im,
                                            processed_img)
                    (self.curr_count,
                     self.prev_count) = (img_count,
                                         self.curr_count)
                    (self.prev_patch,
                     self.curr_patch) = (self.curr_patch,
                                         hand_patch)
                    (self.prev_patch_pos,
                     self.curr_patch_pos) = (self.curr_patch_pos,
                                             hand_patch_pos)
                    if not self.action_type=='Passive':
                        (hand_patch_original,
                         hand_patch_pos_original,
                         self.hand_contour_original) = prepare_im(
                            img)
                        (self.prev_patch_original,
                         self.curr_patch_original) = (self.curr_patch_original,
                                                      hand_patch_original)
                        (self.prev_patch_pos_original,
                         self.curr_patch_pos_original) = (
                             self.curr_patch_pos_original,
                             hand_patch_pos_original)


            except ValueError:
                return False
        return True


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
        self.hist4d.draw(
            hof_features,
            hof_edges,
            fig=self.fig,
            all_axes=self.hof_plots)

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
    <parameters> must be a dictionary.
    '''

    def __init__(self, parameters, coders=None, feat_filename=None, log_lev='INFO'):
        self.parameters = parameters
        self.sparse_helper = ActionsSparseCoding(parameters)
        self.sparse_helper.sparse_coders = coders
        self.dict_names = self.sparse_helper.features
        self.actions = Actions(parameters,
                               coders=self.sparse_helper.sparse_coders,
                               feat_filename=feat_filename)
        self.log_lev = log_lev
        LOG.setLevel(log_lev)

    def add_action(self, *args, **kwargs):
        '''
        actions.add_action alias
        '''
        res = self.actions.add_action(*args, **kwargs)
        return res

    def train_sparse_coders(self,
                            use_dexter=False,
                            trained_coders_list = None,
                            coders_to_train = None,
                            codebooks_dict = None,
                            coders_savepath = None,
                            min_iterations=10,
                            incr_rate=2,
                            sp_opt_max_iter=200,
                            init_traindata_num=200,
                            save=True,
                            debug=False):
        '''
        Add Dexter 1 TOF Dataset or depthdata + binarymaskdata and
        set use_dexter to False (directory with .png or list accepted)
        Inputs:
            act_num: action number to use for training
            use_dexter: true if Dexter 1 dataset is used
            iterations: training iterations
            save_trained: save sparse coders after training
        '''
        if len(self.actions.actions) == 0:
            raise Exception('Run add_action first and then call ' +
                            'train_sparse_coders')
        feat_num = len(self.parameters['features'])
        # Train coders
        self.sparse_helper.initialize()
        for ind,coder in enumerate(trained_coders_list):
            self.sparse_helper.sparse_coders[ind] = coder
        all_sparse_coders = codebooks_dict
        for count,feat_name in enumerate(self.parameters['features']):
            if count in coders_to_train:
                if self.parameters['dynamic_params']['post_PCA']:
                    data = [np.array(self.actions.actions[ind].
                                     bbuffer[count]).reshape(
                                        np.array(self.actions.actions[ind].
                                         bbuffer[count]).shape[0], -1) for ind in
                         range(len(self.actions.actions))]
                else:
                    data = [np.array(self.actions.actions[ind].features[count]) for ind in
                     range(len(self.actions.actions))]
                data = np.concatenate(
                    data, axis=0).T
                frames_num = data.shape[1]
                LOG.info('Frames number: ' + str(frames_num))
                LOG.info('Creating coder for ' + feat_name)
                self.sparse_helper.train(data,
                                        count,
                                        display=1,
                                        init_traindata_num=init_traindata_num,
                                        incr_rate=incr_rate,
                                        sp_opt_max_iter=sp_opt_max_iter,
                                        min_iterations=min_iterations,
                                        debug=debug)
                save_name = feat_name + ' ' + str(self.parameters['sparse_params'][
                    feat_name])
                if self.parameters['dynamic_params']['post_PCA']:
                    comp = self.parameters['dynamic_params']['post_PCA_components']
                    save_name += ' PCA ' + str(comp)
                all_sparse_coders[save_name
                                   ] = self.sparse_helper.sparse_coders[
                    count]
                if codebooks_dict is not None and save:
                    self.sparse_helper.save(save_dict=codebooks_dict)
                    if coders_savepath is not None:
                        LOG.info('Saving '+
                                 str(self.parameters['features'][count])+
                                 ' coder..')
                        with open(coders_savepath, 'w') as output:
                            pickle.dump(all_sparse_coders, output, -1)
        self.parameters['sparse_params']['trained_coders'] = True
        return
