import logging
import sys
import os
import warnings
import glob
from math import pi
import numpy as np
from numpy.linalg import pinv
import cv2
from __init__ import initialize_logger, timeit, PRIM_X, PRIM_Y, FLNT, find_nonzero
import class_objects as co
import sparse_coding as sc
import hand_segmentation_alg as hsa
from matplotlib import pyplot as plt
import cPickle as pickle
import descriptors
from features_visualization import FeatureVisualization
from buffer_operations import BufferOperations
from data_streamer import DataLoader

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
        self.available_descriptors = {'3DHOF': descriptors.TDHOF,
                                      'ZHOF': descriptors.ZHOF,
                                      'GHOG': descriptors.GHOG,
                                      '3DXYPCA': descriptors.TDXYPCA,
                                      'CONTOUR_STATS':descriptors.ContourStatistics}
        self.required_properties = [('VAR',descriptors.VAR),
                                    ('MEDIAN', descriptors.MEDIAN)]
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
        self.descriptors = {feature: None for feature in
                            self.parameters['descriptors']}
        self.descriptors_id = [None] * len(self.parameters['descriptors'])
        self.coders_info = [None] * len(self.parameters['descriptors'])
        self.buffer_class = ([BufferOperations(self.parameters)] *
                             len(self.parameters['descriptors']))
        self.frames_preproc = FramesPreprocessing(
                                self.parameters)

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
                    coder.train_sparse_dictionary(data[finite_samples, :])
                    co.file_oper.save_labeled_data(info, coder)
                else:
                    raise Exception('No data available, run add_action first')
                self.coders[count] = coder
                co.file_oper.save_labeled_data(
                    self.coders_info[count], self.coders[count])

    def load_sparse_coder(self, count):
        self.coders_info[count] = (['Sparse Coders'] +
                                   [self.parameters['sparsecoded']] +
                                   [str(self.parameters['descriptors'][
                                       count])] +
                                   [str(co.dict_oper.create_sorted_dict_view(
                                       self.parameters['coders_params'][
                                           str(self.parameters['descriptors'][count])]))])
        if self.coders[count] is None:
            self.coders[count] = co.file_oper.load_labeled_data(
                self.coders_info[count])
        return self.coders_info[count]

    def retrieve_descriptor_possible_ids(self, count, assume_existence=False):
        if isinstance(count, int):
            is_property = False
            descriptor = self.parameters['descriptors'][count]
        else:
            is_property = True
            descriptor = count
        file_ids = [co.dict_oper.create_sorted_dict_view(
            {'Descriptor': descriptor}),
            co.dict_oper.create_sorted_dict_view(
            {'ActionType': self.parameters['action_type']}),
            (co.dict_oper.create_sorted_dict_view(
            {'DescriptorParams': co.dict_oper.create_sorted_dict_view(
                self.parameters['features_params'][descriptor]['params'])})
            if not is_property else None)]
        ids = ['Features']
        if is_property:
            return ids, file_ids
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
        if self.parameters['action_type'] != 'Passive':
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


    def get_action_properties(self, action, path):
        total_properties = {}
        dl = None
        for prop_name, prop_constructor in self.required_properties:
            ids, file_ids = self.retrieve_descriptor_possible_ids(prop_name)
            key_ids = [ids[0]] + file_ids + [action]
            loaded_data = co.file_oper.load_labeled_data(key_ids)
            if loaded_data is None:
                extracted_properties = []
                self.frames_preproc.reset()
                extractor = prop_constructor(self.frames_preproc)
                if dl is None:
                    dl = DataLoader(path)
                for img, img_count in enumerate(dl.imgs):

                    check = self.frames_preproc.update(img,
                                                       dl.sync[img_count],
                                                       mask=dl.masks[img_count],
                                                       angle=dl.angles[img_count],
                                                       center=dl.centers[img_count])
                    if not check:
                        extracted_properties.append(None)
                    else:
                        extracted_properties.append(extractor.extract())
                total_properties[prop_name] = extracted_properties
                co.file_oper.save_labeled_data(key_ids, total_properties[prop_name])
            else:
                total_properties[prop_name] = loaded_data
        return total_properties




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
        readimagedata = False
        descriptors_num = len(self.parameters['descriptors'])
        loaded_data = [[] for i in range(descriptors_num)]
        features = [None] * descriptors_num
        samples_indices = [None] * descriptors_num
        times = {}
        valid = False
        redo = False
        if 'raw' in to_visualize:
            load = False
        properties, dl = self.get_action_properties(self.name, data)
        while not valid:
            for count, descriptor in enumerate(descriptors):
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
                    except BaseException:
                        continue
                    if nloaded_id == 'Features':
                        if not readimagedata:
                            if dl is None:
                                dl = DataLoader(data)
                            for cnt in range(len(samples_indices)):
                                samples_indices[cnt] = dl.samples_inds.copy()
                            readimagedata = True
                        self.frames_preproc.reset()
                        if not self.descriptors[descriptor]:
                            self.descriptors[
                                descriptor] = self.available_descriptors[
                                    descriptor](parameters=self.parameters,
                                                datastreamer=self.frames_preproc,
                                                viewer=(
                                                    FeatureVisualization(
                                                        offline_vis=offline_vis,
                                                        n_frames=len(dl.imgs)) if
                                                    to_visualize else None))
                        else:
                            self.descriptors[descriptor].reset()
                        features[count] = []
                        valid = []
                        for img_count, img in enumerate(dl.imgs):
                            '''
                            #DEBUGGING
                            cv2.imshow('t', (img%256).astype(np.uint8))
                            cv2.waitKey(30)
                            '''
                            check = self.frames_preproc.update(img,
                                                               dl.sync[img_count],
                                                               mask=dl.masks[img_count],
                                                               angle=dl.angles[img_count],
                                                               center=dl.centers[img_count])
                            if 'features' in to_visualize:
                                self.descriptors[descriptor].set_curr_frame(
                                    img_count)
                            if check:
                                extracted_features = self.descriptors[descriptor].extract()

                                if extracted_features is not None:
                                    features[count].append(extracted_features)
                                else:
                                    features[count].append(None)
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
                        if 'Features' not in times:
                            times['Features'] = []
                        times['Features'] += self.descriptors[descriptor].time

                        if self.preproc_time is None:
                            self.preproc_time = []
                        self.preproc_time += self.frames_preproc.time
                        loaded_ids[nloaded_id] = nloaded_file_id
                        co.file_oper.save_labeled_data([nloaded_id]
                                                       + loaded_ids[nloaded_id] +
                                                       [self.name],
                                                       [np.array(features[count]),
                                                        (dl.sync,
                                                         samples_indices[count]),
                                                        times])
                    elif nloaded_id == 'Sparse Features':
                        if features[count] is None:
                            [features[count],
                             (sync,
                              samples_indices[count]),
                             times] = co.file_oper.load_labeled_data(
                                 [ids[ids.index(nloaded_id) - 1]] +
                                 loaded_ids[ids[ids.index(nloaded_id) - 1]] + [self.name])
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
                                                       loaded_ids[nloaded_id] +
                                                       [self.name],
                                                       [np.array(features[count]),
                                                        (sync,
                                                         samples_indices[count]),
                                                        times])
                    elif nloaded_id == 'Buffered Features':
                        if features[count] is None or samples_indices[count] is None:
                            [features[
                                count],
                             (sync,
                              samples_indices[count]),
                             times] = co.file_oper.load_labeled_data(
                                [ids[ids.index(nloaded_id) - 1]] +
                                loaded_ids[
                                    ids[ids.index(nloaded_id) - 1]] +
                                [self.name])
                        self.buffer_class[count].reset()
                        new_samples_indices = []
                        for sample_count in range(len(features[count])):
                            self.buffer_class[count].update_buffer_info(
                                sync[sample_count],
                                samples_indices[count][sample_count],
                                samples=features[count][sample_count],
                                depth=properties['MEDIAN'][sample_count])
                            self.buffer_class[count].add_buffer()
                        features[count], samples_indices[count], properties['MEDIAN'] = self.buffer_class[count].extract_buffer_list(
                        )
                        loaded_ids[nloaded_id] = nloaded_file_id
                        co.file_oper.save_labeled_data([nloaded_id] + loaded_ids[nloaded_id]
                                                       + [self.name],
                                                       [np.array(features[count]),
                                                        samples_indices[count],
                                                        times])
                    elif nloaded_id == 'Sparse Buffers':
                        if features[count] is None:
                            [features[count],
                             samples_indices[count],
                             times] = co.file_oper.load_labeled_data(
                                ['Buffered Features'] +
                                loaded_ids['Buffered Features']
                                + [self.name])
                        if self.coders[count] is None:
                            self.load_sparse_coder(count)
                        features[count] = self.coders[count].multicode(
                            features[count])
                        if 'Sparse Buffer' not in times:
                            times['Sparse Buffer'] = []
                        times['Sparse Buffer'] += self.coders[
                            count].time
                        loaded_ids[nloaded_id] = nloaded_file_id
                        co.file_oper.save_labeled_data([nloaded_id] +
                                                       loaded_ids[nloaded_id]
                                                       + [self.name],
                                                       [np.array(features[count]),
                                                        samples_indices[count],
                                                        times])
                    elif nloaded_id == 'PTPCA':
                        if features[count] is None:
                            [features[count],
                             samples_indices[count],
                             times] = co.file_oper.load_labeled_data(
                                [ids[ids.index('PTPCA') - 1]] +
                                 loaded_ids[ids[ids.index('PTPCA') - 1]]
                                + [self.name])
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
                                                       loaded_ids[nloaded_id] +
                                                       [self.name],
                                                       [np.array(
                                                           features[count]),
                                                        samples_indices[count],
                                                        times])
                if features[count] is None:
                    try:
                        [features[count],
                         samples_indices[count],
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
                                                               finite_features), axis=0)
            try:
                if np.unique([len(feat) for feat in features]).size == 1:
                    valid = True
                    redo = False
                else:
                    self.logger.warning('Unequal samples dimension of loaded features:'
                                        + str([len(feat) for feat in features])
                                        + ' ...repeating')
                    redo = True
            except Exception as e:
                for count, feat in enumerate(features):
                    if feat is None:
                        print 'Features[' + str(count) + '] is None'
                self.logger.warning(str(e))
                redo = True
                pass
        return (features,
                samples_indices[np.argmax([len(sample) for sample in
                                           samples_indices])],
                properties['MEDIAN'],
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
        except BaseException:
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


class FramesPreprocessing(object):

    def __init__(self, parameters, reset_time=True):
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

    def reset(self, reset_time=False):
        self.__init__(self.parameters, reset_time=reset_time)

    @timeit
    def update(self, img, img_count, use_dexter=False, mask=None, angle=None,
               center=None, masks_needed=False, isderotated=False,
               preprocessed=False):
        '''
        Update frames
        '''
        if isinstance(img, int):
            self.logger.warning('Supplied image is integer')

        if use_dexter:
            mask, hand_patch, hand_patch_pos = prepare_dexter_im(
                img)
        else:
            cnt = None
            # try:
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
                # if self.prev_full_depth_im is None:
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
                # if hand_patch is None:
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
            # except ValueError:
            #    return False
        return not (any_none or self.curr_patch is None)




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
        self.logger = None
        initialize_logger(self, log_lev)

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
