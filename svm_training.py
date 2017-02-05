'''
Implementation of SVM Training, partly described inside Fanello et al.
'''
import sys
import glob
import numpy as np
import class_objects as co
from action_recognition_alg import *
import os.path
import cPickle as pickle
import logging
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
# pylint: disable=no-member


class SVM(object):
    '''
    Class to hold all SVM specific methods.

    '''

    def __init__(self, log_lev='INFO',
                 visualize=False, masks_needed=False,
                 buffer_size=20, train_max_im_num=100,
                 train_max_iter=2):
        logging.basicConfig(format='%(levelname)s:%(message)s')
        logging.getLogger(__name__).setLevel(log_lev)
        self.log_lev = log_lev
        self.visualize = visualize
        self.buffer_size = buffer_size
        self.buffer_exists = []
        self.masks_needed = masks_needed
        self.action_recog = ActionRecognition(log_lev, train_max_im_num,
                                              train_max_iter)
        self.sparse_features_lists = None
        self.unified_classifier = None
        self.sync = []
        self.dicts = None
        self.svms_1_v_all_traindata = None
        self.features_extraction = FeatureExtraction()
        self.scores = None
        self.recognized_classes = []
        self.img_count = -1
        self.train_classes = None
        self._buffer = []
        self.running_small_mean_vec = []
        self.running_big_mean_vec = []
        self.filtered_scores_std = []
        self.filtered_scores_std_mean = []
        self.filtered_scores = None
        self.saved_buffers_scores = []
        self.new_action_starts_count = 0
        self.ground_truth = None
        self.test_ground_truth = None
        self.test_classes = None
        self.train_classes = None
        self.mean_from = -1
        self.on_action = False
        self.crossings = None
        self.act_inds = []
        self.correction_count = 0
        self.less_filtered_scores_std = None
        self.high_filtered_scores_std = None

    def run_training(self, dict_train_act_num=0, retrain_dicts=False,
                     retrain_svms=False, test_against_training=False,
                     training_datapath=None, svms_savepath='train_svms.pkl',
                     num_of_cores=4, buffer_size=20, svms_save=True):
        '''
        <Arguments>
        For testing:
            If a testing using training data is to be used, then
            <test_against_training> is to be True
        For dictionaries training:
            Train dictionaries using <dict_train_act_num>-th action. Do not
            train dictionaries if save file already exists or <retrain_dicts>
            is False.
        For svm training:
            Train SVMS with <num_of_cores> and buffer size <buffer_size>.
            Save them if <svms_save> is True to <svms_savepath>. Do not train
            if <svms_savepath> already exists and <retrain_svms> is False.
        '''
        self.buffer_size = buffer_size
        if retrain_svms or retrain_dicts or test_against_training:
            self.prepare_training_data(training_datapath)
        self.process_dictionaries(dict_train_act_num, retrain_dicts)
        self.process_svms(num_of_cores, buffer_size, retrain_svms,
                          svms_savepath, svms_save, test_against_training)

    def prepare_training_data(self, path=None):
        '''
        Read actions from the <path> and name them according to their parent
        folder name
        '''
        logging.info('Adding actions..')
        if path is None:
            path = co.CONST['actions_path']
        self.train_classes = [name for name in os.listdir(path)
                              if os.path.isdir(os.path.join(path, name))][::-1]
        self.sync = []
        for action in self.train_classes:
            self.action_recog.add_action(os.path.join(path, action),
                                         self.masks_needed,
                                         use_dexter=False)
            self.sync.append(self.action_recog.actions.actions[-1].sync)

    def process_dictionaries(self, train_act_num, retrain=False):
        '''
        Train dictionaries using <train_act_num>-th action or load them if
            <retrain> is not True and file exists (which one?)
        '''
        read_from_mem = 0
        if not retrain:
            try:
                with open(self.action_recog.dictionaries.save_path, 'r') as inp:
                    logging.info('Loading dictionaries..')
                    self.dicts = pickle.load(inp)
                    self.action_recog.dictionaries.dicts = self.dicts
                read_from_mem = 1
            except (IOError, EOFError):
                read_from_mem = 0
        else:
            logging.info('retrain switch is True, so the dictionaries ' +
                         'are retrained')
        if not read_from_mem:
            logging.info('Training dictionaries..')
            self.dicts = self.action_recog.train_sparse_dictionaries(
                act_num=train_act_num, masks_needed=self.masks_needed)
        logging.info('Making Sparse Features..')
        self.sparse_features_lists = (self.action_recog.
                                      actions.update_sparse_features(self.dicts,
                                                                     ret_sparse=True))

    def process_svms(self, num_of_cores=4, buffer_size=20, retrain=False,
                     savepath='train_svms.pkl', save=True,
                     against_training=False):
        '''
        Train (or load trained) SVMs with number of cores num_of_cores, with buffer size (stride
            is 1) <buffer_size>. If <retrain> is True, SVMs are retrained, even if
            <save_path> exists. If against_training, the training data is
            prepared, assuming that it will be used after that for testing.
        '''
        loaded = 0
        if against_training or retrain or not os.path.exists(savepath):
            if against_training and (not retrain or os.path.exists(savepath)):
                logging.info('Preparing SVMs Train Data for testing..')
            else:
                logging.info('Preparing SVMs Train Data..')
            self.buffer_size = buffer_size
            svm_initial_training_data = []
            for sparse_features in self.sparse_features_lists:
                svm_initial_training_data.append(np.concatenate(tuple(sparse_features),
                                                                axis=0))
            svm_training_samples_inds = [
                action.samples_indices
                for action in self.action_recog.actions.actions]
            svm_training_frames_inds = [
                action.sync for action in self.action_recog.actions.actions]
            svms_buffers = []
            for data, frames_samples_inds, frames_inds in zip(
                    svm_initial_training_data, svm_training_samples_inds,
                    svm_training_frames_inds):
                svm_buffers = []
                for count in range(data.shape[1] - self.buffer_size):
                    # Checking if corresponding frames belong to same sample
                    # and are not too timespace distant from each other
                    # If every condition holds then the buffer is generated
                    if (len(np.unique(
                        frames_samples_inds[count:count +
                                            self.buffer_size])) == 1
                        and
                        np.all(np.abs(np.diff(frames_inds[count:count +
                                                          self.buffer_size])) <
                               self.buffer_size / 4)):
                        svm_buffers.append(np.atleast_2d(
                            data[:, count:count + self.buffer_size].ravel()))
                svms_buffers.append(svm_buffers)
            logging.info('Train Data has ' + str(len(svms_buffers)) +
                         ' buffer lists. First buffer list has length ' +
                         str(len(svms_buffers[0])) +
                         ' and last buffer has shape ' +
                         str(svms_buffers[0][-1].shape))
            logging.info('Joining buffers..')
            svms_training_data = []
            for svm_buffers in svms_buffers:
                svms_training_data.append(
                    np.concatenate(tuple(svm_buffers), axis=0))

            logging.info('Train Data has ' + str(len(svms_training_data)) +
                         ' training datasets for each action. Shape of first dataset is ' +
                         str(svms_training_data[0].shape))
            logging.info(
                'Creating ground truth vector and concatenating remaining data..')
            self.ground_truth = []
            self.svms_1_v_all_traindata = np.concatenate(
                tuple(svms_training_data), axis=0)
            for count, data in enumerate(svms_training_data):
                self.ground_truth.append(count *
                                         np.ones(data.shape[0]))
            self.ground_truth = np.concatenate(
                tuple(self.ground_truth), axis=0)
            logging.info('Final TrainData to be used as input to OneVsRestClassifier' +
                         ' has shape ' + str(self.svms_1_v_all_traindata.shape))
        if not retrain and os.path.exists(savepath):
            logging.info('Loading trained SVMs..')
            with open(savepath, 'r') as inp:
                (self.unified_classifier, self.ground_truth,
                 self.train_classes) = pickle.load(inp)
            loaded = 1
        else:
            if retrain and not os.path.exists(savepath):
                logging.info('retrain switch is True, so the SVMs ' +
                             'are retrained')
            logging.info(
                'Training SVMs using ' +
                str(num_of_cores) +
                ' cores..')
            self.unified_classifier = OneVsRestClassifier(LinearSVC(),
                                                          num_of_cores).fit(
                                                              self.svms_1_v_all_traindata,
                                                              self.ground_truth)
        if save and not loaded:
            logging.info('Saving trained SVMs..')
            with open(savepath, 'w') as out:
                pickle.dump((self.unified_classifier, self.ground_truth,
                             self.train_classes), out)

    def offline_testdata_processing(self, datapath, ground_truth_type):
        '''
        Offline testing data processing, using data in <datapath>.
        Refer to <construct_ground_truth> for <ground_truth_type> info
        '''
        logging.info('Processing test data..')
        logging.info('Constructing ground truth vector..')
        logging.info('Extracting features..')
        sparse_features = self.action_recog.add_action(datapath, masks_needed=False,
                                                       for_testing=True)
        sparse_features = np.concatenate(tuple(sparse_features), axis=0)
        svm_buffers = []
        frames_inds = self.action_recog.actions.testing.sync
        test_buffers_start_inds = []
        test_buffers_end_inds = []
        for count in range(sparse_features.shape[1] - self.buffer_size):
            if np.all(np.abs(np.diff(frames_inds[count:count +
                                                 self.buffer_size])) <
                      self.buffer_size / 4):

                svm_buffers.append(np.atleast_2d(sparse_features[:, count:count +
                                                                 self.buffer_size].ravel()))
                test_buffers_start_inds.append(frames_inds[count])
                test_buffers_end_inds.append(frames_inds[count +
                                                         self.buffer_size])
        svm_testing_data = np.concatenate(tuple(svm_buffers), axis=0)
        return svm_testing_data, test_buffers_start_inds, test_buffers_end_inds

    def construct_ground_truth(self, data=None, ground_truth_type=None,
                               testing=True):
        '''
        <ground_truth_type>:'*.csv'(wildcat) to load csv
                                whose rows have format
                                class | start_index | end_index
                            'filename' to load class from datapath
                                filenames which have format
                                'number-class.png' or 'number.png'
                                if it has no class
        <data> should be either a string refering to the datapath
        of the numbered frames or a boolean list/array.
        if <testing> is True, compare ground truth classes with
            <self.train_classes> and remove classes that do not exist
            inside <self.train_classes>
        Returns <ground_truth> vectors which holds indices that refer
            to <classes> vector, which is also returned. Any item with
            no class has corresponding ground truth NaN
        '''
        if isinstance(data, basestring):
            if not os.path.exists(data):
                raise Exception(data + ' is a non existent path')
            files = [os.path.splitext(filename)[0] for filename in
                     glob.glob(os.path.join(data, '*.png'))]
            if not files:
                raise Exception(data + ' does not include any png file')
        else:
            data = np.array(data)
            if np.max(data) != 1:
                raise Exception('data should be boolean')
            else:
                data = data.astype(bool)
        ground_truth_init = {}
        if ground_truth_type[-4::] == '.csv':
            try:
                with open(ground_truth_type, 'r') as inp:
                    for line in inp:
                        if '\n' in line:
                            line.replace('\n', '')
                        items = line.split(':')
                        try:
                            ground_truth_init[items[0]].append([
                                int(item) for item in items[1].split(',')])
                            ground_truth_init[items[0]].append([
                                int(item) for item in items[2].split(',')])
                        except (AttributeError, KeyError):
                            ground_truth_init[items[0]] = []
                            ground_truth_init[items[0]].append([
                                int(item) for item in items[1].split(',')])
                            ground_truth_init[items[0]].append([
                                int(item) for item in items[2].split(',')])
            except (EOFError, IOError):
                raise Exception('Invalid csv file given\n' +
                                self.construct_ground_truth.__doc__)
            keys = ground_truth_init.keys()
            if testing:
                class_match = {}
                for key in keys:
                    try:
                        class_match[key] = self.train_classes.index(key)
                    except ValueError:
                        ground_truth_init.pop(key, None)
                if not ground_truth_init:
                    raise Exception(
                        'No classes found matching with training data ones')
            else:
                class_match = {}
                for count, key in enumerate(keys):
                    class_match[key] = count
            length = max([max(ground_truth_init[item]) for item in
                          ground_truth_init])[0]
            if not isinstance(data, basestring):
                ground_truth = np.zeros(len(data))
            else:
                ground_truth = np.zeros(length + 1)
            ground_truth[:] = np.NaN
            all_bounds = [map(list, zip(*ground_truth_init[key])) for key in
                          ground_truth_init.keys()]
            if isinstance(data, basestring):
                iterat = [int(filter(str.isdigit, filename))
                          for filename in files]
            else:
                iterat = np.where(data)[0]
                print iterat
            for count, ind in enumerate(iterat):
                for key, bounds in zip(ground_truth_init, all_bounds):
                    for bound in bounds:
                        if ind <= bound[1] and ind >= bound[0]:
                            ground_truth[ind] = class_match[key]
                            break
        elif ground_truth_type == 'filename':
            ground_truth_vecs = [filename.split('-') for filename
                                 in files]
            classes = []
            ground_truth = np.zeros(len(files))
            ground_truth[:] = np.NaN
            inval_format = True
            for count, item in enumerate(ground_truth_vecs):
                if len(item) > 2:
                    inval_format = True
                    break
                if len(item) == 2:
                    inval_format = False
                    if testing:
                        if item[1] not in self.train_classes:
                            continue
                    if item[1] not in classes:
                        classes.append(item[1])
                    ground_truth[count] = classes.index(items[1])
            if inval_format:
                print 'Invalid format'
                raise Exception(self.construct_ground_truth.__doc__)
        else:
            raise Exception('Invalid ground_truth_type\n' +
                            self.construct_ground_truth.__doc__)
        return ground_truth

    def masked_mean(self, data, win_size):
        mask = np.isnan(data)
        K = np.ones(win_size, dtype=int)
        return np.convolve(np.where(mask,0,data), K)/np.convolve(~mask,K)

    def upgr_filter(self, data, win_size):
        '''
        Mean filter data with missing values, along axis 1
        '''
        if len(data.shape) == 1:
            inp = np.atleast_2d(data).T
        else:
            inp = data
        return np.apply_along_axis(self.masked_mean, 0, data, win_size)

    def run_testing(self, data=None, online=True, against_training=False,
                    scores_filter_shape=20,
                    std_small_filter_shape=co.CONST['STD_small_filt_window'],
                    std_big_filter_shape=co.CONST['STD_big_filt_window'],
                    ground_truth_type='test_ground_truth.csv',
                    img_count=None, save=True, save_path='svm_test_results.pkl',
                    load=False, like_paper=True):
        '''
        Test SVMS using data (.png files) located in <data>. If <online>, the
        testing is online, with <data> being a numpy array, which has been
        firstly processed by <hand_segmentation_alg>. If <against_training>,
        the testing happens using the concatenated training data, so <data> is
        not used. The scores retrieved from testing are filtered using a
        box filter of shape <box_filter_shape>. The running mean along a buffer
        of the data is computed with a running window of length
        <mean_filter_shape>. The ground truth for the testing data is given by
        <ground_truth_type> (for further info about the variable refer to
        <self.construct_ground_truth>). If the training is online, the count of
        the frame is passed by <img_count>.If <save> is True, testing
        results are saved to <save_path>.
        If <load> is True and <save_path> exists, testing is bypassed and all the
        necessary results are loaded from memory.
        '''
        if not online:
            self.buffer_exists = []
            if load and os.path.exists(save_path):
                with open(save_path, 'r') as inp:
                    self.scores = pickle.load(inp)
                # DEBUGGING
            else:
                logging.info('Classifier contains ' +
                             str(len(self.unified_classifier.estimators_)) + ' estimators')
                if against_training:
                    logging.info('Testing SVMS against training data..')
                    self.scores = self.unified_classifier.decision_function(
                        self.svms_1_v_all_traindata)
                else:
                    (svm_testing_data,
                     test_buffers_start_inds,
                     test_buffers_end_inds) = self.offline_testdata_processing(data,
                                                                               ground_truth_type)
                    logging.info('Testing SVMS..')
                    print svm_testing_data.shape
                    print svm_testing_data.max(), svm_testing_data.min()
                    self.scores = self.unified_classifier.decision_function(
                        svm_testing_data)
                    expanded_scores = np.zeros((test_buffers_end_inds[-1],
                                                self.scores.shape[1]))
                    expanded_scores[:] = np.NaN
                    for score, start, end in zip(self.scores,
                                                 test_buffers_start_inds,
                                                 test_buffers_end_inds):
                        expanded_scores[start:end + 1, :] = score[None, :]
                    self.scores = expanded_scores
                    if save:
                        with open(save_path, 'w') as out:
                            pickle.dump(self.scores, out)
            self.test_ground_truth = self.construct_ground_truth(
                data, ground_truth_type)
            self.filtered_scores = self.upgr_filter(self.scores,
                                                    scores_filter_shape)

            if like_paper:
                self.process_offline_scores_original(scores_filter_shape,
                                                     std_small_filter_shape,
                                                     std_big_filter_shape)
            else:
                self.process_scores_remade()
        else:
            '''
            input is processed from hand_segmentation_alg (any data
            processed in such way, that the result is the same with my processing,
            is acceptable, eg. Dexter)
            There must be a continuous data streaming (method called in every
            loop), even if the result of the previous algorithm is None
            '''
            self.img_count += 1
            self.mean_from += 1
            # self.buffer_exists = self.buffer_exists[
            #    -std_big_filter_shape:]
            if not self.img_count or (img_count == 0):
                self._buffer = []
                self.mean_from = 0
                self.buffer_exists = []
                self.scores = []
                self.filtered_scores = []
                self.filtered_scores_std_mean = []
                self.filtered_scores_std = []
                self.act_inds = []
                self.crossings = []
            if img_count is not None:
                self.buffer_exists += ((img_count - self.img_count) * [False])
                self.img_count = img_count
            if data is None:
                self.buffer_exists.append(False)
                return
            self.features_extraction.update(
                data, self.img_count, use_dexter=False, masks_needed=False)
            features = self.features_extraction.extract_features()
            if features is None:
                self.buffer_exists.append(False)
                return
            sparse_features = np.concatenate(
                tuple([np.dot(dic, feature) for (dic, feature) in
                       zip(self.dicts, features)]), axis=0)
            if len(self._buffer) < self.buffer_size:
                self._buffer = self._buffer + [sparse_features]
                self.buffer_exists.append(False)
                return
            else:
                self._buffer = self._buffer[1:] + [sparse_features]
                self.buffer_exists.append(True)
            existence = self.buffer_exists[-self.buffer_size:]
            if sum(existence) < 3 * self.buffer_size / 4:
                return
            elif sum(existence) == 3 * self.buffer_size / 4:
                self.mean_from = 0

            inp = np.array(self._buffer).T.reshape(1, -1)
            # inp = np.atleast_2d(np.concatenate(tuple(self._buffer),
            #                                   axis=0))
            score = (self.unified_classifier.
                     decision_function(inp))
            self.scores.append(score)
            if len(self.running_small_mean_vec) < std_small_filter_shape:
                self.running_small_mean_vec.append(score.ravel())
            else:
                self.running_small_mean_vec = (self.running_small_mean_vec[1:]
                                               + [score.ravel()])
            start_from = np.sum(self.buffer_exists[-std_small_filter_shape:])
            self.filtered_scores.append(
                np.mean(np.array(self.running_small_mean_vec[
                    -start_from:]
                ), axis=0))
            filtered_score_std = np.std(self.filtered_scores[-1])
            self.filtered_scores_std.append(filtered_score_std)
            if len(self.running_big_mean_vec) < std_big_filter_shape:
                self.running_big_mean_vec.append(filtered_score_std)
            else:
                self.running_big_mean_vec = (self.running_big_mean_vec[1:]
                                             + [filtered_score_std])
            start_from = np.sum(self.buffer_exists[
                -min(std_big_filter_shape, self.mean_from + 1):])
            self.filtered_scores_std_mean.append(
                np.mean(self.running_big_mean_vec[-start_from:]))
            mean_diff = self.filtered_scores_std_mean[
                -1] - self.filtered_scores_std[-1]
            if (mean_diff > co.CONST['action_separation_thres'] and not
                self.on_action):
                self.crossings.append(self.img_count)
                self.on_action = True
                self.mean_from = self.img_count
                if self.recognized_classes:
                    self.recognized_classes[-1].add(length=self.img_count -
                                                self.new_action_starts_count +
                                                   1)
                    logging.info('Frame ' + str(self.img_count) + ': ' +
                                 self.recognized_classes[-1].name +
                                 ', starting from frame ' +
                                 str(self.recognized_classes[-1].start) +
                                ' with length ' +
                                 str(self.recognized_classes[-1].length))
                self.recognized_classes.append(ClassObject(self.train_classes))
                '''
                #this is for evaluating previous result
                index = np.mean(
                    np.array(self.saved_buffers_scores), axis=0).argmax()
                '''
                index = np.argmax(self.filtered_scores[-1])
                self.act_inds = [index]
                self.new_action_starts_count = self.img_count
                self.recognized_classes[-1].add(
                                                index=index,
                                                start=self.new_action_starts_count)
                '''
                logging.info('Frame ' + str(self.img_count) + ': ' +
                             self.recognized_classes[-1].name +
                             ', starting from frame ' +
                             str(self.recognized_classes[-1].start) +
                            ' with length ' +
                             str(self.recognized_classes[-1].length))
                '''
                self.saved_buffers_scores = []
            else:
                if self.recognized_classes:
                    if self.correction_count < 10:
                        index = np.argmax(self.filtered_scores[-1])
                        self.act_inds.append(index)
                        self.recognized_classes[-1].add(
                            index=np.median(self.act_inds))
                        self.correction_count += 1
                if mean_diff < co.CONST['action_separation_thres']:
                    self.on_action = False
                if len(self.recognized_classes) > 0:
                    logging.info('Frame ' + str(self.img_count) +': ' +
                                 self.recognized_classes[-1].name)
                self.saved_buffers_scores.append(score)


    def process_offline_scores_original(self, scores_filter_shape,
                                        std_small_filter_shape,
                                        std_big_filter_shape, display=True):
        '''
        Process scores using stds as proposed by the paper
        '''
        fmask = np.prod(np.isfinite(self.scores), axis=1).astype(bool)
        self.filtered_scores_std = np.zeros(self.scores.shape[0])
        self.filtered_scores_std[:] = None
        self.filtered_scores_std[fmask] = np.std(self.scores[fmask, :],
                                                 axis=1)
        self.less_filtered_scores_std = self.upgr_filter(self.filtered_scores_std,
                                                         std_small_filter_shape)

        self.high_filtered_scores_std = self.upgr_filter(self.filtered_scores_std,
                                                         std_big_filter_shape)

        positive = np.zeros_like(self.scores.shape[0])
        positive[:] = None
        positive[fmask] = ((self.high_filtered_scores_std -
                            self.less_filtered_scores_std)[fmask] > 0).astype(int)
        # We are interested only in finding negative to positive zero crossings,
        # because this is where std falls below its mean
        neg_to_pos_zero_crossings = np.where(positive[1:] -
                                             positive[:-1] ==
                                             -1)[0]
        self.crossings = neg_to_pos_zero_crossings
        interesting_crossings = np.concatenate((np.array([0]),
                                                neg_to_pos_zero_crossings,
                                                np.array([self.scores.shape[0]])),
                                               axis=0)
        self.recognized_classes = []
        count = 0
        for cross1, cross2 in zip(interesting_crossings[
                :-1], interesting_crossings[1:]):
            act_scores = self.filtered_scores[cross1:cross2, :]
            mask = fmask[cross1:cross2]
            if not np.any(mask):
                act = np.zeros(cross2 - cross1)
                act[:] = None
                self.recognized_classes.append(act)
                continue
            index = np.mean(
                act_scores[mask, :], axis=0).argmax()
            '''
            index = np.median(
                act_scores[mask, :], axis=0).argmax()
            '''
            act = index * np.ones(cross2 - cross1)
            act[np.logical_not(mask)] = None
            self.recognized_classes.append(act)
        self.recognized_classes = np.concatenate(
            tuple(self.recognized_classes), axis=0)
        if display:
            import matplotlib
            matplotlib.rcParams['text.usetex'] = True
            matplotlib.rcParams['text.latex.unicode'] = True
            plt.style.use('seaborn-ticks')
            plt.figure()
            axes = plt.subplot(111)
            axes.plot(
                np.array(
                    self.less_filtered_scores_std),
                color='r',
                label=r'STD')
            axes.plot(np.array(self.high_filtered_scores_std),
                      color='g', label='STD Mean')
            self.put_legend_outside_plot(axes)
            axes.set_title('Filtered Scores Statistics')
            axes.set_xlabel('Frames')

            plt.figure()
            axes = plt.subplot(111)
            for count, score in enumerate(np.transpose(np.array(
                    self.filtered_scores))):
                axes.plot(score, label='%s' % self.train_classes[count])
            self.put_legend_outside_plot(axes)
            plt.title('Filtered Scores')
            plt.xlabel('Frames')

            plt.figure()
            axes = plt.subplot(111)
            mean_diff = (np.array(self.high_filtered_scores_std) -
                         np.array(self.less_filtered_scores_std))
            axes.plot((mean_diff) / float(np.max(np.abs(mean_diff))),
                      label='Filtered scores\nnormalized\nmean difference')
            axes.plot(self.crossings, np.zeros_like(self.crossings), 'o',
                      label='Selected Zero\n Crossings')
            if self.test_ground_truth is not None:
                axes.plot((self.test_ground_truth - np.mean(self.test_ground_truth[
                    np.isfinite(self.test_ground_truth)]))
                          / float(np.max(self.test_ground_truth[
                              np.isfinite(self.test_ground_truth)])),
                          label='Ground Truth', linewidth=1.5)
            self.put_legend_outside_plot(axes)
            plt.title('Measure of actions starting and ending points')
            plt.xlabel('Frames')

    def process_scores_remade(self):
        '''
        By examining svms maximums, they seem to provide a more efficient
        measure of when the action starts and ends, although they are
        susceptible to SVMs local accuracy
        '''
        self.recognized_classes = np.zeros(
            self.filtered_scores.shape[0], float)
        self.recognized_classes[:] = None
        mask = (np.prod(np.isfinite(
            self.filtered_scores), axis=1)).astype(bool)
        self.recognized_classes[mask] = np.argmax(self.filtered_scores,
                                                  axis=1)[mask].astype(int)
        tmp = np.diff(self.recognized_classes)
        tmp[np.logical_not(np.isfinite(tmp))] = 0
        self.crossings = np.where(tmp != 0)[0]
        '''
        # throw away crossings with small distance from each other. This will
        # happen locally in the online training
        interesting_crossings = np.concatenate((np.array([0]),
                                                self.crossings,
                                                np.array([self.scores.shape[0]])), axis=0)
        dist = abs(np.diff(np.concatenate((interesting_crossings, [0]))))
        close_cross = dist < 20
        self.crossings = interesting_crossings[np.logical_not(close_cross)]
        for start, end in zip (self.crossings[:-1], self.crossings[1:]):
            self.recognized_classes[start:end+1][mask[start:end+1]] = np.bincount(
                self.recognized_classes[start:end+1][mask[start:end+1]].astype(int)).argmax()
        '''
        cut_filtered_scores = self.filtered_scores[:]
        cut_mask = np.zeros_like(cut_filtered_scores)
        cut_mask[range(self.filtered_scores[mask].shape[0]),
                 self.recognized_classes[mask].astype(int)] = 1
        cut_mask = (1 - cut_mask).astype(bool)
        cut_filtered_scores[cut_mask] = np.NaN
        plt.figure()
        axes = plt.subplot(111)
        for count, score in enumerate(np.transpose(np.array(
                cut_filtered_scores))):
            axes.plot(score, label='%s' % self.train_classes[count])
        self.put_legend_outside_plot(axes)
        plt.title('Filtered Scores Maximums along time')
        plt.xlabel('Frames')

    def put_legend_outside_plot(self, axes):
        '''
        Remove legend from the insides of the plots
        '''
        # Shrink current axis by 20%
        box = axes.get_position()
        axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def compute_performance_measures(self):
        from sklearn import metrics
        fmask = \
        np.isfinite(self.test_ground_truth)*np.isfinite(self.recognized_classes)
        y_true = self.test_ground_truth[fmask]
        y_pred = self.recognized_classes[fmask]
        f1_scores = metrics.f1_score(y_true,y_pred, average = None)
        print f1_scores
        confusion_mat = metrics.confusion_matrix(y_true, y_pred)
        print confusion_mat
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print accuracy

    def visualize_scores(self, title=''):
        '''
        Plot results with title <title>
        '''
        if self.buffer_exists:
            self.buffer_exists = np.array(self.buffer_exists)
            expanded_recognized_classes = np.zeros(self.buffer_exists.size)
            expanded_recognized_classes[:] = None
            for clas in self.recognized_classes:
                expanded_recognized_classes[clas.start:clas.start + clas.length + 1][
                    self.buffer_exists[clas.start:clas.start + clas.length + 1]] = clas.index
            self.recognized_classes = expanded_recognized_classes
            self.crossings = np.array(self.crossings)
        self.compute_performance_measures()
        plt.figure()
        axes = plt.subplot(111)
        if self.test_ground_truth is not None:
            axes.plot(
                self.test_ground_truth,
                label='Ground Truth',
                linewidth=1.5)
        if self.crossings is not None:
            axes.set_xticks(self.crossings, minor=True)
            axes.xaxis.grid(True, which='minor')
            axes.plot(self.crossings, np.zeros_like(self.crossings) - 1, 'o',
                      label='Actions\nbreak-\n points')
        axes.plot(self.recognized_classes, label='Identified\nClasses')
        self.put_legend_outside_plot(axes)
        axes.set_xlabel('Frames')
        labels = self.train_classes
        plt.yticks(range(len(labels)), labels)
        plt.ylim((-1, len(labels) + 1))
        if title is None:
            axes.set_title('Result')
        else:
            axes.set_title(title)
        plt.show()


class ClassObject(object):
    '''
    Class to hold classification classes
    '''

    def __init__(self, class_names):
        self.name = ''
        self.index = 0
        self.start = 0
        self.length = 0
        self.names = class_names

    def add(self, index=None, start=None, length=None):
        '''
        Add Class with name, corresponding index in self.train_classes,
            starting from start frame, with length length frames
        '''
        if index is not None:
            self.index = int(index)
            self.name = self.names[int(index)]
        if start is not None:
            self.start = start
        if length is not None:
            self.length = length


def fake_online_testing(svm, path=None):
    '''
    Immitate online testing for performance testing reasons
    '''
    if path is None:
        path = co.CONST['rosbag_res_save_path']
    filenames = glob.glob(os.path.join(path, '*.png'))
    for filename in filenames:
        img = cv2.imread(filename, -1)
        img_count = int(filter(str.isdigit, filename))
        svm.run_testing(img, img_count=img_count, online=True, load=False,
                        like_paper=True)
    svm.recognized_classes[-1].add(length=img_count)
    svm.test_ground_truth = svm.construct_ground_truth(
        svm.buffer_exists, ground_truth_type='test_ground_truth.csv')
    plt.figure()
    axes = plt.subplot(111)
    for count, scores in enumerate(np.transpose(np.array(
            svm.scores))):
        axes.plot(scores.ravel(), label='%s' % svm.train_classes[count])
    svm.put_legend_outside_plot(axes)
    plt.title('Filtered Scores')
    plt.xlabel('Frames')


def main():
    '''
    Example Usage
    '''
    svm = SVM('INFO', train_max_im_num=100, train_max_iter=2)
    svm.run_training(retrain_svms=False, retrain_dicts=False, buffer_size=10)
    '''
    # testing against training data
    svm.run_training(test_against_training=True)
    svm.run_testing(online=False, against_training=True)
    '''
    # testing with other offline data
    '''
    svm.run_testing(co.CONST['rosbag_res_save_path'], online=False, load=True,
                    like_paper=True)
    svm.visualize_scores('Results as paper proposes')
    '''
    '''
    svm.run_testing(co.CONST['rosbag_res_save_path'], online=False, load=True,
                    like_paper=False)
    svm.visualize_scores('Results with simple approach')
    '''
    fake_online_testing(svm)
    svm.visualize_scores('Fake Online Testing')
if __name__ == '__main__':
    main()
