
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
def initialize_logger(logger):
    if not getattr(logger, 'handler_set', None):
        CH = logging.StreamHandler()
        CH.setFormatter(logging.Formatter(
            '%(name)s-%(funcName)s()(%(lineno)s)-%(levelname)s:%(message)s'))
        logger.addHandler(CH)
        logger.handler_set = True
    logger.propagate = False

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

def timeit(func):
    '''
    Decorator to time extraction
    '''
    def wrapper(self,*arg, **kw):
        t1 = time.time()
        res = func(self,*arg, **kw)
        t2 = time.time()
        self.time.append(t2-t1)
        del self.time[:-5000]
        return res
    return wrapper

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
        try:
            contour = contours[np.argmax(contours_area)].squeeze()
        except ValueError:
            return None, None, None
    hand_contour = contour.squeeze()
    if hand_contour.size == 2:
        return None, None, None
    if square:
        edge_size = max(np.max(hand_contour[:, 1]) - np.min(hand_contour[:, 1]),
                        np.max(hand_contour[:, 0]) - np.min(hand_contour[:, 0]))
        center = np.mean(hand_contour, axis=0).astype(int)
        hand_patch = img[center[1] - edge_size / 2:
                         center[1] + edge_size / 2,
                         center[0] - edge_size / 2:
                         center[0] + edge_size / 2
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


class BufferOperations(object):

    def __init__(self, parameters, reset_time=True):
        self.logger = logging.getLogger('BufferOperations')
        initialize_logger(self.logger)
        self.parameters = parameters
        self.buffer = []
        self.depth = []
        self.testing = parameters['testing']
        self.action_type = parameters['action_type']
        self.samples_indices = []
        self.buffer_start_inds = []
        self.buffer_end_inds = []
        if not self.action_type == 'Passive':
            self.ptpca = parameters['PTPCA']
            self.ptpca_components = parameters['PTPCA_params'][
                'PTPCA_components']
        self.bbuffer = [[] for i in range(len(parameters['descriptors']))]
        if not self.action_type == 'Passive':
            self.buffer_size = parameters['dynamic_params']['buffer_size']
            try:
                self.buffer_confidence_tol = parameters['dynamic_params'][
                    'buffer_confidence_tol']
                self.ptpca = parameters['PTPCA']
                self.ptpca_components = parameters['PTPCA_params'][
                    'PTPCA_components']
            except (KeyError, IndexError, TypeError):
                self.buffer_confidence_tol = None
            self.pca_features = []
        else:
            self.buffer_size = 1
        self.sync = []
        self.frames_inds = []
        self.samples_inds = []
        self.buffer_components = []
        self.depth_components = []
        self.real_samples_inds = []
        if reset_time:
            self.time = []

    def reset(self, reset_time=False):
        self.__init__(self.parameters, reset_time=reset_time)


    def check_buffer_integrity(self, buffer):
        check_sam = True
        check_cont = True
        check_len = len(buffer) == self.buffer_size
        if check_len:
            if not self.action_type == 'Passive':

                check_cont = np.all(np.abs(np.diff(self.frames_inds[-self.buffer_size:])) <=
                                    self.buffer_size * self.buffer_confidence_tol)
                # check if buffer frames belong to the same sample, in case of
                # training
                check_sam = self.testing or len(np.unique(
                    self.samples_inds[-self.buffer_size:])) == 1
            else:
                check_cont = True
                check_sam = True
                check_len = True
        return check_len and check_cont and check_sam

    @timeit
    def perform_post_time_pca(self, inp):
        reshaped = False
        if self.buffer_size == 1:
            return
        if np.shape(inp)[0] == 1 or len(np.shape(inp))==1:
            reshaped = True
            inp = np.reshape(inp, (self.buffer_size, -1))
        mean, inp = cv2.PCACompute(
            np.array(inp),
            np.array([]),
            maxComponents=self.ptpca_components)
        inp = (np.array(inp) + mean)
        if reshaped:
            return inp.ravel()
        return inp

    def update_buffer_info(self, sync, samples_index=0,
                           samples=None, depth=None):
        self.frames_inds.append(sync)
        self.samples_inds.append(samples_index)
        if samples is not None:
            self.buffer_components.append(samples)
            del self.buffer_components[:-self.buffer_size]
        if depth is not None:
            self.depth_components.append(depth)
            del self.depth_components[:-self.buffer_size]
    def add_buffer(self, buffer=None, depth=None, sample_count=None,
                   already_checked=False):
        '''
        <buffer> should have always the same size.
        <self.bbuffer> is a list of buffers. It can have a size limit, after which it
        acts as a buffer (useful for shifting window
        operations (filtering etc.))
        '''
        # check buffer contiguousness
        if buffer is None:
            buffer = self.buffer_components
        if depth is None:
            fmask = np.isfinite(self.depth_components)
            if np.sum(fmask):
                depth = np.mean(np.array(self.depth_components)[fmask])
        if not already_checked:
            check = self.check_buffer_integrity(buffer[-self.buffer_size:])
        else:
            check = True
        if not self.parameters['testing_params']['online']:
                self.real_samples_inds += [-1] * (self.frames_inds[-1] + 1 -
                               len(self.buffer))
                self.depth += [None] * (self.frames_inds[-1] + 1
                                         - len(self.buffer))
                self.buffer += [None] * (self.frames_inds[-1] + 1
                                         - len(self.buffer))
        if check:
            self.buffer_start_inds.append(self.frames_inds[-self.buffer_size])
            self.buffer_end_inds.append(self.frames_inds[-1])
            if not self.parameters['testing_params']['online']:
                self.buffer[self.frames_inds[-1]] = np.array(
                    buffer)
                self.depth[self.frames_inds[-1]] = depth
            else:
                self.buffer = np.array(buffer)
                self.depth = depth
            if not self.parameters['testing_params']['online']:
                self.real_samples_inds[self.frames_inds[-1]] = (np.unique(self.samples_inds[
                -self.buffer_size:])[0])
        else:
            if self.parameters['testing_params']['online']:
                self.buffer = None
                self.depth = None

    def extract_buffer_list(self):
        '''
        Returns a 2d numpy array, which has as first dimension the number of
        saved features sets inside <self.bbuffer>,
        as second dimension a flattened buffer. In case it is online, the first
        dimension is 1. In case there are None samples inside, those are turned
        to None arrays.
        '''
        if self.parameters['testing_params']['online']:
            if self.bbuffer is None:
                return None
        else:
            buffer_len = 0
            for _buffer in self.buffer:
                if _buffer is not None:
                    buffer_len = np.size(_buffer)
                    break
            if not buffer_len:
                self.logger.debug('No valid buffer')
                return None
        npbuffer = np.zeros((len(self.buffer),buffer_len))
        for buffer_count in range(len(self.buffer)):
            if self.buffer[buffer_count] is None:
                self.buffer[buffer_count] = np.zeros(buffer_len)
                self.buffer[buffer_count][:] = np.nan
            npbuffer[buffer_count, ...] =\
                np.array(self.buffer[buffer_count]).T.ravel()
        return npbuffer, self.real_samples_inds, self.depth


class Action(object):
    '''
    Class to hold an action
    '''

    def __init__(self, parameters, name, coders=None):
        self.name = name
        self.parameters = parameters
        self.features = []
        self.sync = []
        self.frames_inds = []
        self.samples_inds = []
        self.length = 0
        self.start_inds = []
        self.end_inds = []
        self.real_data = []

class Actions(object):
    '''
    Class to hold multiple actions
    '''

    def __init__(self, parameters, coders=None, feat_filename=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        initialize_logger(self.logger)
        self.slflogger = logging.getLogger('save_load_features')
        FH = logging.FileHandler('save_load_features.log', mode='w')
        FH.setFormatter(logging.Formatter(
            '%(asctime)s (%(lineno)s): %(message)s',
            "%Y-%m-%d %H:%M:%S"))
        self.slflogger.addHandler(FH)
        self.slflogger.setLevel(logging.INFO)
        self.parameters = parameters
        
        self.sparsecoded = parameters['sparsecoded']
        self.available_descriptors = {'3DHOF': Descriptor3DHOF,
                                      'ZHOF': DescriptorZHOF,
                                      'GHOG': DescriptorGHOG,
                                      '3DXYPCA': Descriptor3DXYPCA}
        self.actions = []
        self.names = []
        self.coders = coders
        if coders is None:
            self.coders = [None] * len(self.parameters['descriptors'])
        self.save_path = (os.getcwd() +
                          os.sep + 'saved_actions.pkl')
        self.features_extract = None
        self.preproc_time = []
        self.features_db = None
        self.feat_filename = feat_filename
        self.candid_d_actions = None
        self.valid_feats = None
        self.all_data = [None] * len(self.parameters['descriptors'])
        self.name = None
        self.frames_preproc = None
        self.descriptors = {feature:None for feature in
                            self.parameters['descriptors']}
        self.descriptors_id = [None] * len(self.parameters['descriptors'])
        self.coders_info = [None] * len(self.parameters['descriptors'])
        self.buffer_class= ([BufferOperations(self.parameters)] *
                            len(self.parameters['descriptors']))

    def save_action_features_to_mem(self, data, filename=None,
                                    action_name=None):
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
            if self.feat_filename is None:
                return
            else:
                filename = self.feat_filename
        if action_name is None:
            action_name = self.name
        for dcount, descriptor in enumerate(
                self.parameters['descriptors']):
            if self.candid_d_actions[dcount] is None:
                self.candid_d_actions[dcount] = {}
            self.candid_d_actions[dcount][action_name] = data[dcount]
            co.file_oper.save_labeled_data([descriptor,
                                            str(co.dict_oper.
                                                create_sorted_dict_view(
                                            self.parameters[
                                                'features_params'][
                                                    descriptor]))],
                                           self.candid_d_actions[dcount],
                                           filename, fold_lev=1)

    def load_action_features_from_mem(self, filename=None):
        '''
        features_db has tree structure, with the following order:
            features type->list of instances dicts ->[params which are used to
            identify the features, data of each instance->actions]
        This order allows to search only once for each descriptor and get all
        actions corresponding to a matching instance, as it is assumed that
        descriptors are fewer than actions
        '''
        features_to_extract = self.parameters['descriptors'][:]
        data = [None] * len(features_to_extract)
        if self.candid_d_actions is None:
            self.candid_d_actions = []
            if filename is None:
                if self.feat_filename is None:
                    return features_to_extract, data
                else:
                    filename = self.feat_filename
            for descriptor in self.parameters['descriptors']:
                self.candid_d_actions.append(
                    co.file_oper.load_labeled_data([descriptor,
                                                    str(co.dict_oper.create_sorted_dict_view(
                                                    self.parameters[
                                                        'features_params'][
                                                            descriptor]))],
                                                   filename, fold_lev=1))
            '''
            Result is <candid_d_actions>, a list which holds matching
            instances of actions for each descriptor, or None if not found.
            '''
        for dcount, instance in enumerate(self.candid_d_actions):
            self.slflogger.info('Descriptor: ' + self.parameters['descriptors'][
                dcount])
            if instance is not None:
                self.slflogger.info('Finding action \'' + self.name +
                            '\' inside matching instance')
                if self.name in instance and np.array(
                        instance[self.name][0]).size > 0:
                    self.slflogger.info('Action Found')
                    data[dcount] = instance[self.name]
                    features_to_extract.remove(self.parameters['descriptors']
                                               [dcount])
                else:
                    self.slflogger.info('Action not Found')
            else:
                self.slflogger.info('No matching instance exists')

        return features_to_extract, data



    def train_sparse_dictionary(self):
        '''
        Train missing sparse dictionaries. add_action should have been executed
        first
        '''
        for count, (data, info) in enumerate(
                zip(self.all_data, self.coders_info)):
            if not self.coders[count]:
                if data is not None:
                    coder = sc.SparseCoding(
                        sparse_dim_rat=self.parameters['features_params'][
                            self.parameters['descriptors'][count]][
                                'sparse_params']['_dim_rat'],
                        name=self.parameters['descriptors'][count])
                    finite_samples = np.prod(np.isfinite(data),
                                             axis=1).astype(bool)
                    coder.train_sparse_dictionary(data[finite_samples,:])
                    co.file_oper.save_labeled_data(info, coder)
                else:
                    raise Exception('No data available, run add_action first')
                self.coders[count] = coder
                co.file_oper.save_labeled_data(self.coders_info[count], self.coders[count])

    def load_sparse_coder(self, count):
        self.coders_info[count] = (['Sparse Coders']+
                                   [self.parameters['sparsecoded']]+
                    [str(self.parameters['descriptors'][
                        count])]+
                    [str(co.dict_oper.create_sorted_dict_view(
                        self.parameters['coders_params'][
                    str(self.parameters['descriptors'][count])]))])
        if self.coders[count] is None:
            self.coders[count] = co.file_oper.load_labeled_data(
                self.coders_info[count])
        return self.coders_info[count]


    def retrieve_descriptor_possible_ids(self, count, assume_existence=False):
        descriptor = self.parameters['descriptors'][count]
        file_ids = [co.dict_oper.create_sorted_dict_view(
            {'Descriptor':descriptor}),
                    co.dict_oper.create_sorted_dict_view(
            {'ActionType':self.parameters['action_type']}),
                    co.dict_oper.create_sorted_dict_view(
                        {'DescriptorParams':co.dict_oper.create_sorted_dict_view(
                        self.parameters['features_params'][descriptor]['params'])})]
        ids = ['Features']
        if self.sparsecoded:
            self.load_sparse_coder(count)
        if (self.parameters['sparsecoded'] == 'Features'
                and (self.coders[count] is not None or assume_existence)):
            file_ids.append(co.dict_oper.create_sorted_dict_view(
                {'SparseFeaturesParams':
                             co.dict_oper.create_sorted_dict_view(
                                 self.parameters[
                'features_params'][descriptor]['sparse_params'])}))
            ids.append('Sparse Features')
        file_ids.append(co.dict_oper.create_sorted_dict_view(
            {'BufferParams':
                         co.dict_oper.create_sorted_dict_view(
                             self.parameters['dynamic_params'])}))
        ids.append('Buffered Features')
        if self.parameters['action_type']!='Passive':
            if (self.parameters['sparsecoded'] == 'Buffer'
                    and (self.coders[count] is not None or assume_existence)):
                file_ids.append(co.dict_oper.create_sorted_dict_view(
                    {'SparseBufferParams':
                                 co.dict_oper.create_sorted_dict_view(
                                     self.parameters[
                    'features_params'][descriptor]['sparse_params'])}))
                ids.append('Sparse Buffers')
            if not (self.parameters['sparsecoded'] == 'Buffer'
                    and self.coders[count] is None) or assume_existence:
                if self.parameters['PTPCA']:
                    file_ids.append(co.dict_oper.create_sorted_dict_view(
                        {'PTPCAParams':
                                     co.dict_oper.create_sorted_dict_view(
                                         self.parameters[
                    'PTPCA_params'])}))
                    ids.append('PTPCA')
        return ids, file_ids


    def add_action(self, data=None,
                   mv_obj_fold_name=None,
                   hnd_mk_fold_name=None,
                   masks_needed=True,
                   use_dexter=False,
                   visualize_=False,
                   isderotated=False,
                   action_type='Dynamic',
                   max_act_samples=None,
                   fss_max_iter=None,
                   derot_centers=None,
                   derot_angles=None,
                   name=None,
                   feature_extraction_method=None,
                   save=True,
                   load=True,
                   feat_filename=None,
                   calc_mean_depths=False,
                   to_visualize=[],
                   n_vis_frames=9,
                   exit_after_visualization=False,
                   offline_vis=False):
        '''
        parameters=dictionary having at least a 'descriptors' key, which holds
            a sublist of ['3DXYPCA', 'GHOG', '3DHOF', 'ZHOF']. It can have a
            'features_params' key, which holds specific parameters for the
            features to be extracted.
        features_extract= FeatureExtraction Class
        data= (Directory with depth frames) OR (list of depth frames)
        use_dexter= True if Dexter 1 TOF Dataset is used
        visualize= True to visualize features extracted from frames
        '''
        self.name = name
        if name is None:
            self.name = os.path.basename(data)
        loaded_data = [[] for i in range(len(self.parameters['descriptors']))]
        readimagedata = False
        features = [None] * len(self.parameters['descriptors'])
        buffers = [None] * len(self.parameters['descriptors'])
        samples_indices = [None] * len(self.parameters['descriptors'])
        median_depth = [None] * len(self.parameters['descriptors'])
        times = {}
        valid = False
        redo = False
        if 'raw' in to_visualize:
            load = False
        while not valid:
            for count, descriptor in enumerate(self.parameters['descriptors']):
                nloaded_ids = {}
                loaded_ids = {}
                ids, file_ids = self.retrieve_descriptor_possible_ids(count)
                try_ids = ids[:]
                for try_count in range(len(try_ids)):
                    loaded_data = co.file_oper.load_labeled_data(
                        [try_ids[-1]] + file_ids + [self.name])
                    if loaded_data is not None and not redo and load:
                        loaded_ids[try_ids[-1]] = file_ids[:]
                        break
                    else:
                        nloaded_ids[try_ids[-1]] = file_ids[:]
                        try_ids = try_ids[:-1]
                        file_ids = file_ids[:-1]
                for _id in ids:
                    try:
                        nloaded_file_id = nloaded_ids[_id]
                        nloaded_id = _id
                    except:
                        continue
                    if nloaded_id == 'Features':
                        if not readimagedata:
                            (imgs, masks, sync, angles,
                             centers,
                             samples_inds) = co.imfold_oper.load_frames_data(
                                 data,mv_obj_fold_name,
                                 hnd_mk_fold_name, masks_needed,
                                 derot_centers,derot_angles)
                            if 'raw' in to_visualize:
                                montage = co.draw_oper.create_montage(imgs[:],
                                                                      max_ims=n_vis_frames,
                                                                      draw_num=False)
                                fig = plt.figure()
                                tmp_axes = fig.add_subplot(111)
                                tmp_axes.imshow(montage[:,:,:-1])
                                plt.axis('off')
                                fig.savefig('frames_sample.pdf',
                                            bbox_inches='tight')
                                to_visualize.remove('raw')
                                if (not to_visualize and
                                    exit_after_visualization):
                                    return
                            for cnt in range(len(samples_indices)):
                                samples_indices[cnt] = samples_inds.copy()
                            readimagedata = True
                        if not self.frames_preproc:
                            self.frames_preproc = FramesPreprocessing(self.parameters)
                        else:
                            self.frames_preproc.reset()
                        if not self.descriptors[descriptor]:
                            self.descriptors[
                                descriptor] = self.available_descriptors[
                                    descriptor](parameters=self.parameters,
                                                datastreamer=self.frames_preproc,
                                               viewer=(
                                                   FeatureVisualization(
                                                    offline_vis=offline_vis,
                                                    n_frames=len(imgs)) if
                                               to_visualize else None))
                        else:
                            self.descriptors[descriptor].reset()
                        features[count] = []
                        median_depth[count] = []
                        valid = []
                        for img_count, img in enumerate(imgs):
                            '''
                            #DEBUGGING
                            cv2.imshow('t', (img%256).astype(np.uint8))
                            cv2.waitKey(30)
                            '''
                            check = self.frames_preproc.update(img,
                                                  sync[img_count],
                                                  mask=masks[img_count],
                                                  angle=angles[img_count],
                                                          center=centers[img_count])
                            if 'features' in to_visualize:
                                self.descriptors[descriptor].set_curr_frame(img_count)
                            if check:

                                extracted_features = self.descriptors[descriptor].extract()
                                if extracted_features is not None:
                                    features[count].append(extracted_features)
                                    median_depth[count].append(np.median(self.frames_preproc.curr_patch))
                                else:
                                    features[count].append(None)
                                    median_depth[count].append(None)
                                if 'features' in to_visualize:
                                    self.descriptors[descriptor].visualize()
                                    self.descriptors[descriptor].draw()
                                    if (len(to_visualize) == 1 and
                                        exit_after_visualization):
                                        continue
                            else:
                                if (len(to_visualize) == 1
                                    and exit_after_visualization):
                                    self.descriptors[descriptor].draw()
                                    continue
                                features[count].append(None)
                                median_depth[count].append(None)
                        if 'Features' not in times:
                            times['Features'] = []
                        times['Features'] += self.descriptors[descriptor].time
                        if self.preproc_time is None:
                            self.preproc_time = []
                        self.preproc_time+=self.frames_preproc.time
                        loaded_ids[nloaded_id] = nloaded_file_id
                        co.file_oper.save_labeled_data([nloaded_id]
                                                       +loaded_ids[nloaded_id]+
                                                       [self.name],
                                                       [np.array(features[count]),
                                                        (sync,
                                                        samples_indices[count]),
                                                        median_depth[count],
                                                        times])
                    elif nloaded_id == 'Sparse Features':
                        if features[count] is None:
                            [features[count],
                             (sync,
                             samples_indices[count]),
                             median_depth[count],
                             times] = co.file_oper.load_labeled_data(
                                 [ids[ids.index(nloaded_id)-1]]+
                                 loaded_ids[ids[ids.index(nloaded_id)-1]]+[self.name])
                        if self.coders[count] is None:
                            self.load_sparse_coder(count)
                        features[count] = self.coders[
                            count].multicode(features[count])
                        if 'Sparse Features' not in times:
                            times['Sparse Features'] = []
                        times['Sparse Features'] += self.coders[
                            count].time
                        loaded_ids[nloaded_id] = nloaded_file_id
                        co.file_oper.save_labeled_data([nloaded_id] +
                                                       loaded_ids[nloaded_id]+
                                                       [self.name],
                                                       [np.array(features[count]),
                                                        (sync,
                                                        samples_indices[count]),
                                                        median_depth[count],
                                                        times])
                    elif nloaded_id == 'Buffered Features':
                        if features[count] is None or samples_indices[count] is None:
                            [features[
                                count],
                             (sync,
                             samples_indices[count]),
                             median_depth[count],
                            times] = co.file_oper.load_labeled_data(
                                [ids[ids.index(nloaded_id) -1]] +
                                loaded_ids[
                                    ids[ids.index(nloaded_id) - 1]] +
                                [self.name])
                        self.buffer_class[count].reset()
                        new_samples_indices = []
                        for sample_count in range(len(features[count])):
                            self.buffer_class[count].update_buffer_info(
                                sync[sample_count],
                                samples_indices[count][sample_count],
                                samples = features[count][sample_count],
                                depth=median_depth[count][sample_count])
                            self.buffer_class[count].add_buffer()
                        features[count],samples_indices[count],median_depth[count] = self.buffer_class[count].extract_buffer_list()
                        loaded_ids[nloaded_id] = nloaded_file_id
                        co.file_oper.save_labeled_data([nloaded_id]+loaded_ids[nloaded_id]
                                                       +[self.name],
                                                       [np.array(features[count]),
                                                        samples_indices[count],
                                                        median_depth[count],
                                                         times])
                    elif nloaded_id == 'Sparse Buffers':
                        if features[count] is None:
                            [features[count],
                             samples_indices[count],
                             median_depth[count],
                             times] = co.file_oper.load_labeled_data(
                                ['Buffered Features']+loaded_ids['Buffered Features']
                            +[self.name])
                        if self.coders[count] is None:
                            self.load_sparse_coder(count)
                        features[count] = self.coders[count].multicode(features[count])
                        if 'Sparse Buffer' not in times:
                            times['Sparse Buffer'] = []
                        times['Sparse Buffer'] += self.coders[
                            count].time
                        loaded_ids[nloaded_id] = nloaded_file_id
                        co.file_oper.save_labeled_data([nloaded_id] + 
                                                       loaded_ids[nloaded_id]
                                                       +[self.name],
                                                       [np.array(features[count]),
                                                        samples_indices[count],
                                                        median_depth[count],
                                                        times])
                    elif nloaded_id == 'PTPCA':
                        if features[count] is None:
                            [features[count],
                             samples_indices[count],
                             median_depth[count],
                             times] = co.file_oper.load_labeled_data(
                                [ids[ids.index('PTPCA')-1]] +
                                 loaded_ids[ids[ids.index('PTPCA') - 1]]
                                +[self.name])
                        self.buffer_class[count].reset()
                        features[count] = [
                            self.buffer_class[count].perform_post_time_pca(
                                _buffer) for _buffer in features[count]]
                        if 'PTPCA' not in times:
                            times['PTPCA'] = []
                        times['PTPCA'] += self.buffer_class[
                            count].time
                        loaded_ids[nloaded_id] = nloaded_file_id
                        co.file_oper.save_labeled_data([nloaded_id] +
                                                       loaded_ids[nloaded_id]+
                                                       [self.name],
                                                       [np.array(
                                                           features[count]),
                                                        samples_indices[count],
                                                        median_depth[count],
                                                        times])
                if features[count] is None:
                    try:
                        [features[count],
                        samples_indices[count],
                        median_depth[count],
                        times] = loaded_data
                        if isinstance(samples_indices[count], tuple):
                            samples_indices[count] = samples_indices[count][-1]
                    except TypeError:
                        pass
                self.descriptors_id[count] = loaded_ids[ids[-1]]
                if (self.parameters['sparsecoded'] and not self.coders[count]):
                    finite_features = []
                    for feat in features[count]:
                        if feat is not None:
                            finite_features.append(feat)
                    if self.all_data[count] is None:
                        self.all_data[count] = np.array(finite_features)
                    else:
                        self.all_data[count] = np.concatenate((self.all_data[count],
                            finite_features),axis=0)
            try:
                if np.unique([len(feat) for feat in features]).size == 1:
                    valid = True
                    redo = False
                else:
                    self.logger.warning('Unequal samples dimension of loaded features:'
                                + str([len(feat) for feat in features])
                                +' ...repeating')
                    redo = True
            except Exception as e:
                for count,feat in enumerate(features):
                    if feat is None:
                        print 'Features[' + str(count) + '] is None'
                self.logger.warning(str(e))
                redo = True
                pass
        return (features,
                samples_indices[np.argmax([len(sample) for sample in
                                                        samples_indices])],
                median_depth[np.argmax([len(median_depth) for
                                                 median_depth in
                                                 samples_indices])],
                self.name, self.coders, self.descriptors_id)

    def save(self, save_path=None):
        '''
        Save actions to file
        '''
        if save_path is None:
            actions_path = self.save_path
        else:
            actions_path = save_path

        self.logger.info('Saving actions to ' + actions_path)
        with open(actions_path, 'wb') as output:
            pickle.dump(self.actions, output, -1)


class ActionsSparseCoding(object):
    '''
    Class to hold sparse coding coders
    '''

    def __init__(self, parameters):
        self.features = parameters['descriptors']
        self.logger = logging.getLogger(self.__class__.__name__)
        initialize_logger(self.logger)
        self.parameters = parameters
        self.sparse_dim_rat = []
        try:
            for feat in self.features:
                self.sparse_dim_rat.append(parameters['sparse_params'][feat])
        except (KeyError, TypeError):
            self.sparse_dim_rat = [None] * len(self.features)
        self.sparse_coders = []
        self.codebooks = []
        self.initialized = True
        self.save_path = (os.getcwd() +
                          os.sep + 'saved_coders.pkl')

    def train(self, data, feat_count, display=0, min_iterations=10,
              init_traindata_num=200, incr_rate=2, sp_opt_max_iter=200,
              debug=False, save_traindata=True):
        '''
        feat_count: features position inside
                    actions.actions[act_num].features list
        '''
        try:
            self.sparse_coders[feat_count].display = display
        except:
            self.sparse_coders[feat_count] = sc.SparseCoding(
                sparse_dim_rat=self.sparse_dim_rat[feat_count],
                name=str(feat_count))
            self.sparse_coders[feat_count].display = display
            self.logger.info('Training Dictionaries using data of shape:'
                     + str(data.shape))
            if save_traindata:
                savepath = ('SparseTraining-' +
                            self.parameters['descriptors'][
                                feat_count] + '.npy')
                self.logger.info('TrainData is saved to ' + savepath)
                np.save(savepath, data, allow_pickle=False)
        self.sparse_coders[feat_count].train_sparse_dictionary(data,
                                                               init_traindata_num=init_traindata_num,
                                                               incr_rate=incr_rate,
                                                               sp_opt_max_iter=sp_opt_max_iter,
                                                               min_iterations=min_iterations,
                                                               n_jobs=4)
        self.codebooks[feat_count] = (
            pinv(self.sparse_coders[feat_count].codebook_comps))
        return 1

    def initialize(self):
        '''
        initialize / reset all codebooks that refer to the given <sparse_dim_rat>
        and feature combination
        '''
        self.sparse_coders = []
        for count, feature in enumerate(self.features):
            self.sparse_coders.append(sc.SparseCoding(
                sparse_dim_rat=self.sparse_dim_rat[count],
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
                feat_dim = inv_dict.codebook_comps.shape[0]
                feat_dims[feat_count] = feat_dim
            except AttributeError:
                feat_dims[feat_count] = None
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
                if not self.parameters['PTPCA']:
                    save_dict[feature + ' ' +
                              str(self.sparse_dim_rat[feat_count])] = \
                        self.sparse_coders[feat_count]
                else:
                    save_dict[feature + ' ' +
                              str(self.sparse_dim_rat[feat_count]) +
                              ' PCA ' +
                              str(self.parameters['PTPCA_params'][
                                  'PTPCA_components'])] = \
                        self.sparse_coders[feat_count]
            return
        if save_path is None:
            coders_path = self.save_path
        else:
            coders_path = save_path

        self.logger.info('Saving Dictionaries to ' + coders_path)
        with open(coders_path, 'wb') as output:
            pickle.dump((self.sparse_coders, self.codebooks), output, -1)


def grad_angles(patch):
    '''
    Compute gradient angles on image patch for GHOG
    '''
    y_size = 30
    x_size = int(y_size/float(np.shape(patch)[0])
                               * np.shape(patch)[1])
    patch = cv2.resize(patch, (x_size, y_size),
                               interpolation=cv2.INTER_NEAREST)
    grady, gradx = np.gradient(patch)
    ang = np.arctan2(grady, gradx)
    #ang[ang < 0] = ang[ang < 0] + pi

    return ang.ravel()  # returns values 0 to pi


class FramesPreprocessing(object):

    def __init__(self, parameters,reset_time=True):
        self.logger = logging.getLogger(self.__class__.__name__)
        initialize_logger(self.logger)
        self.parameters = parameters
        self.action_type = parameters['action_type']
        self.skeleton = None
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
        self.hand_contour = None
        self.fig = None
        self.kernel = np.ones((5, 5), np.uint8)
        self.curr_full_depth_im = None
        self.prev_full_depth_im = None
        self.prev_cnt = None
        self.curr_cnt = None
        self.angle = None
        self.center = None
        self.hand_img = None
        if reset_time:
            self.time = []

    def reset(self,reset_time=False):
        self.__init__(self.parameters, reset_time=reset_time)

    @timeit
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
            #try:
            if masks_needed and mask is None:
                mask1 = cv2.morphologyEx(
                    img.copy(), cv2.MORPH_OPEN, self.kernel)
                _, cnts, _ = cv2.findContours(mask1.astype(np.uint8),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
                if not cnts:
                    img = None
                else:
                    cnts_areas = [cv2.contourArea(cnts[i]) for i in
                                  xrange(len(cnts))]
                    cnt = cnts[np.argmax(cnts_areas)]
                    if self.skeleton is None:
                        self.skeleton = hsa.FindArmSkeleton(img.copy())
                    skeleton_found = self.skeleton.run(img, cnt,
                                             'longest_ray')
                    if skeleton_found:    
                        mask = self.skeleton.hand_mask
                        last_link = (self.skeleton.skeleton[-1][1] -
                                     self.skeleton.skeleton[-1][0])
                        angle = np.arctan2(
                            last_link[0], last_link[1])
                        center = self.skeleton.hand_start
                    else:
                        img = None


            if img is not None and not isderotated and angle is None:
                raise Exception('mask is not None, derotation is True ' +
                                'and angle and center are missing, ' +
                                'cannot proceed with this combination')
            if self.action_type is not 'Passive':
                if img is not None:
                    (self.prev_full_depth_im,
                     self.curr_full_depth_im) = (self.curr_full_depth_im,
                                                 img)
                imgs = [self.prev_full_depth_im,
                        self.curr_full_depth_im]
                #if self.prev_full_depth_im is None:
                #    return False
            else:
                imgs = [img]
            curr_img = img
            any_none = any([im is None for im in imgs])
            if not any_none:
                imgs = [im.copy() for im in imgs]
            for im in imgs:
                if not any_none:
                    if np.sum(mask * img > 0) == 0:
                        any_none = True
                if not any_none:
                    im = im * (mask > 0)
                    if not isderotated:
                        if angle is not None and center is not None:
                            self.angle = angle
                            self.center = center
                            processed_img = co.pol_oper.derotate(
                                im,
                                angle, center)
                    else:
                        processed_img = im
                else:
                    processed_img = None
                self.hand_img = im
                if processed_img is not None:
                    hand_patch, hand_patch_pos, self.hand_contour = prepare_im(
                        processed_img)
                else:
                    hand_patch, hand_patch_pos, self.hand_contour = (None,
                                                                     None,
                                                                     None)
                # DEBUGGING
                # cv2.imshow('test',((hand_patch)%255).astype(np.uint8))
                # cv2.waitKey(10)
                #if hand_patch is None:
                #    return False
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
                if not self.action_type == 'Passive':
                    if not any_none:
                        (hand_patch_original,
                         hand_patch_pos_original,
                        self.hand_contour_original) = prepare_im(
                            im)
                    else:
                        (hand_patch_original,
                         hand_patch_pos_original,
                         self.hand_contour_original) = (None, None, None)
                    (self.prev_patch_original,
                     self.curr_patch_original) = (self.curr_patch_original,
                                                  hand_patch_original)
                    (self.prev_patch_pos_original,
                     self.curr_patch_pos_original) = (
                         self.curr_patch_pos_original,
                         hand_patch_pos_original)
            #except ValueError:
            #    return False
        return not (any_none or self.curr_patch is None)


class Descriptor(object):
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


class Descriptor3DHOF(Descriptor):

    def __init__(self, *args, **kwargs):
        Descriptor.__init__(self, *args, **kwargs)
        self.name = '3dhof'
        self.logger = logging.getLogger(self.__class__.__name__)
        initialize_logger(self.logger)
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


class DescriptorZHOF(Descriptor):

    def __init__(self, *args, **kwargs):
        Descriptor.__init__(self, *args, **kwargs)
        self.name = 'zhof'
        self.logger = logging.getLogger(self.__class__.__name__)
        initialize_logger(self.logger)
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
        # print disp.max(axis=0), disp.min(axis=0)
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


class DescriptorGHOG(Descriptor):

    def __init__(self, *args, **kwargs):
        Descriptor.__init__(self, *args, **kwargs)
        self.name = 'ghog'
        self.logger = logging.getLogger(self.__class__.__name__)
        initialize_logger(self.logger)
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


class Descriptor3DXYPCA(Descriptor):

    def __init__(self, *args, **kwargs):
        Descriptor.__init__(self, *args, **kwargs)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = '3dxypca'
        initialize_logger(self.logger)
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
                self.patches3d.append(self.convert_plot2array(self.patches3d_fig))
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
    def convert_plot2array(self,fig):
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
                self.patches3d.append(self.convert_plot2array(self.patches3d_fig))

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
            except:
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
                with open('visualized_features','w') as out:
                    pickle.dump((self.hof,self.hogs,
                                 self.patches2d,
                                 self.patches3d), out)
                tmp_fig = plt.figure()
                tmp_axes = tmp_fig.add_subplot(111)
                if self.hogs:
                    hogs_im = co.draw_oper.create_montage(
                        self.hogs, draw_num=False)
                    tmp_axes.imshow(hogs_im[:,:,:3])
                    plt.axis('off')
                    tmp_fig.savefig('ghog.pdf', bbox_inches='tight')
                if self.hof is not None:
                    tmp_axes.imshow(self.hof[:,:,:3])
                    plt.axis('off')
                    tmp_fig.savefig('3dhof.pdf', bbox_inches='tight')
                if self.patches2d:
                    patches2d_im = co.draw_oper.create_montage(
                        self.patches2d, draw_num=False)
                    tmp_axes.imshow(patches2d_im[:,:,:3])
                    plt.axis('off')
                    tmp_fig.savefig('patches2d.pdf', bbox_inches='tight')
                if self.patches3d:
                    patches3d_im = co.draw_oper.create_montage(
                        self.patches3d, draw_num=False)
                    tmp_axes.imshow(patches3d_im[:,:,:3])
                    plt.axis('off')
                    tmp_fig.savefig('patches3d.pdf', bbox_inches='tight')
                if self.xypca:
                    tmp_axes.imshow(self.hof[:,:,:3])
                    plt.axis('off')
                    tmp_fig.savefig('3dxypca.pdf', bbox_inches='tight')



    def unpause(self, val):
        plt.gcf().canvas.stop_event_loop()


class ActionRecognition(object):
    '''
    Class to hold everything about action recognition
    <parameters> must be a dictionary.
    '''

    def __init__(self, parameters, coders=None,
                 feat_filename=None, log_lev='INFO'):
        self.parameters = parameters
        self.sparse_helper = ActionsSparseCoding(parameters)
        self.sparse_helper.sparse_coders = coders
        self.dict_names = self.sparse_helper.features
        self.actions = Actions(parameters,
                               coders=self.sparse_helper.sparse_coders,
                               feat_filename=feat_filename)
        self.log_lev = log_lev
        self.logger = logging.getLogger('ActionRecognition')
        initialize_logger(self.logger)
        self.logger.setLevel(log_lev)
        # self.logger.setLevel('SAVE_LOAD_FEATURES')

    def add_action(self, *args, **kwargs):
        '''
        actions.add_action alias
        '''
        res = self.actions.add_action(*args, **kwargs)
        return res

    def train_sparse_coders(self,
                            use_dexter=False,
                            trained_coders_list=None,
                            coders_to_train=None,
                            codebooks_dict=None,
                            coders_savepath=None,
                            min_iterations=10,
                            incr_rate=2,
                            sp_opt_max_iter=200,
                            init_traindata_num=200,
                            save=True,
                            debug=False,
                            save_traindata=True):
        '''
        Add Dexter 1 TOF Dataset or depthdata + binarymaskdata and
        set use_dexter to False (directory with .png or list accepted)
        Inputs:
            act_num: action number to use for training
            use_dexter: true if Dexter 1 dataset is used
            iterations: training iterations
            save: save sparse coders after training
            save_traindata: save training data used for sparsecoding
        '''
        if len(self.actions.actions) == 0:
            raise Exception('Run add_action first and then call ' +
                            'train_sparse_coders')
        feat_num = len(self.parameters['descriptors'])
        # Train coders
        self.sparse_helper.initialize()
        for ind, coder in enumerate(trained_coders_list):
            self.sparse_helper.sparse_coders[ind] = coder
        all_sparse_coders = codebooks_dict
        all_data = [self.actions.actions[ind].retrieve_features(concat=False) for
                    ind in range(len(self.actions.actions))]
        '''
        all_data is a list of actions. Each action is a list of descriptors.
        Each descriptor is a 3d numpy array of samples-buffers with shape =
        (samples number, buffer size, features size)
        '''
        for count, feat_name in enumerate(self.parameters['descriptors']):
            if count in coders_to_train:
                if self.parameters['PTPCA']:
                    self.logger.info('Using PCA with ' + str(
                        self.parameters['PTPCA_params']['PTPCA_components']) +
                        ' components')
                data = [
                    all_data[ind][count].reshape(
                        all_data[ind][count].shape[0], -1)
                    for ind in
                    range(len(self.actions.actions))]
                for ind, d in enumerate(data):
                    self.logger.info('Descriptor of ' + feat_name + ' for action \'' +
                             str(self.actions.actions[ind].name) +
                             '\' has shape ' + str(d.shape))
                data = np.concatenate(
                    data, axis=0)
                frames_num = data.shape[0]
                self.logger.info('Frames number: ' + str(frames_num))
                self.logger.info('Creating coder for ' + feat_name)
                self.sparse_helper.train(data[np.prod(
                    np.isfinite(data), axis=1).astype(bool),
                    :],
                    count,
                    display=1,
                    init_traindata_num=init_traindata_num,
                    incr_rate=incr_rate,
                    sp_opt_max_iter=sp_opt_max_iter,
                    min_iterations=min_iterations,
                    debug=debug,
                    save_traindata=save_traindata)
                save_name = feat_name + ' ' + str(self.parameters['sparse_params'][
                    feat_name])
                if self.parameters['PTPCA']:
                    comp = self.parameters['PTPCA_params'][
                        'PTPCA_components']
                    save_name += ' PCA ' + str(comp)
                all_sparse_coders[save_name
                                  ] = self.sparse_helper.sparse_coders[
                    count]
                if codebooks_dict is not None and save:
                    self.sparse_helper.save(save_dict=codebooks_dict)
                    if coders_savepath is not None:
                        self.logger.info('Saving ' +
                                 str(self.parameters['descriptors'][count]) +
                                 ' coder..')
                        with open(coders_savepath, 'w') as output:
                            pickle.dump(all_sparse_coders, output, -1)
        self.parameters['sparse_params']['trained_coders'] = True
        return
