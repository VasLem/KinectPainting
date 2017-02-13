'''
Implementation of Classifier Training, partly described inside Fanello et al.
'''
import sys
import glob
import numpy as np
import class_objects as co
from action_recognition_alg import *
import os.path
import cPickle as pickle
import logging
# pylint: disable=no-member


class Classifier(object):
    '''
    Class to hold all Classifier specific methods.

    '''

    def __init__(self, log_lev='INFO',
                 visualize=False, masks_needed=False,
                 buffer_size=20, train_max_im_num=100,
                 train_max_iter=2, des_dim=None,
                 isstatic = False,
                 use='svms', num_of_cores=4, name='',
                 hog_num=None):
        self.log_lev = log_lev
        self.visualize = visualize
        self.buffer_size = buffer_size
        self.buffer_exists = None
        self.masks_needed = masks_needed
        self.action_recog = ActionRecognition(log_lev, train_max_im_num,
                                              train_max_iter, des_dim=des_dim)
        self.sparse_features_lists = None
        self.unified_classifier = None
        self.sync = []
        self.dicts = None
        self.one_v_all_traindata = None
        self.features_extraction = FeatureExtraction()
        self.scores = None
        self.recognized_classes = []
        self.img_count = -1
        self.train_classes = None
        self._buffer = []
        self.scores_running_mean_vec = []
        self.big_std_running_mean_vec = []
        self.small_std_running_mean_vec = []
        self.filtered_scores_std = []
        self.filtered_scores_std_mean = []
        self.filtered_scores = None
        self.saved_buffers_scores = []
        self.new_action_starts_count = 0
        self.train_ground_truth = None
        self.test_ground_truth = None
        self.test_classes = None
        self.train_classes = None
        self.mean_from = -1
        self.on_action = False
        self.crossings = None
        self.act_inds = []
        self._max= 0
        self.less_filtered_scores_std = None
        self.high_filtered_scores_std = None
        self.isstatic = isstatic
        self.testdata = None
        self.test_sync = None
        self.use = use
        self.num_of_cores =  num_of_cores
        self.name = name
        self.allowed_train_actions = None
        self.hog_num = hog_num
        self.already_expanded = False
        if self.use == 'svms':
            from sklearn.svm import LinearSVC
            from sklearn.multiclass import OneVsRestClassifier
            self.classifier_type = OneVsRestClassifier(LinearSVC(),
                                                      self.num_of_cores)
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.classifier_type = RandomForestClassifier(10)
        if isstatic:
            fil = os.path.join(co.CONST['rosbag_location'],
                               'gestures_type.csv')
            self.allowed_train_actions = []
            if os.path.exists(fil):
                with open(fil,'r') as inp:
                    for line in inp:
                        if line.split(':')[0] == 'Static':
                            self.allowed_train_actions = line.split(
                                ':')[1].rstrip('\n').split(',')
            else:
                self.allowed_train_actions=None

    def initialize_classifier(self,classifier=None):
        if classifier is not None:
            self.unified_classifier = classifier
        else:
            if self.use == 'svms':
                from sklearn.svm import LinearSVC
                from sklearn.multiclass import OneVsRestClassifier
                self.unified_classifier = OneVsRestClassifier(LinearSVC(),
                                                          self.num_of_cores)
            else:
                from sklearn.ensemble import RandomForestClassifier
                self.unified_classifier = RandomForestClassifier(10)

        if self.use == 'svms':
            self.decide = self.unified_classifier.decision_function
            self.predict = self.unified_classifier.predict
        else:
            self.decide = self.unified_classifier.predict_proba
            self.predict = self.unified_classifier.predict

    def run_training(self, dict_train_act_num=0, dicts_retrain=False,
                     classifiers_retrain=False, test_against_training=False,
                     training_datapath=None, classifiers_savepath=None,
                     num_of_cores=4, buffer_size=None, classifiers_save=True,
                     max_act_samples=None, use_dicts = True):
        '''
        <Arguments>
        For testing:
            If a testing using training data is to be used, then
            <test_against_training> is to be True
        For dictionaries training:
            Train dictionaries using <dict_train_act_num>-th action. Do not
            train dictionaries if save file already exists or <dicts_retrain>
            is False.
        For svm training:
            Train ClassifierS with <num_of_cores> and buffer size <buffer_size>.
            Save them if <classifiers_save> is True to <classifiers_savepath>. Do not train
            if <classifiers_savepath> already exists and <classifiers_retrain> is False.
        '''
        if classifiers_savepath is None:
            classifiers_savepath = self.use + '_' + self.name + '_train'
            if self.isstatic:
                classifiers_savepath = classifiers_savepath + '_static'
            else:
                classifiers_savepath = classifiers_savepath + '_active'
            classifiers_savepath += '.pkl'
        if buffer_size is not None:
            self.buffer_size = buffer_size
        if classifiers_retrain or dicts_retrain or test_against_training:
            self.prepare_training_data(training_datapath, max_act_samples)
        if use_dicts:
            self.process_dictionaries(dict_train_act_num, dicts_retrain)
        self.process_training(num_of_cores, classifiers_retrain,
                          classifiers_savepath, classifiers_save,
                              test_against_training, use_dicts=use_dicts)

    def prepare_training_data(self, path=None, max_act_samples=None):
        '''
        Read actions from the <path> and name them according to their parent
        folder name
        '''
        LOG.info('Adding actions..')
        if path is None:
            path = co.CONST['actions_path']
        self.train_classes = [name for name in os.listdir(path)
                              if os.path.isdir(os.path.join(path, name))][::-1]
        self.sync = []
        if self.allowed_train_actions is not None:
            self.train_classes = [clas for clas in self.train_classes if clas
                                  in self.allowed_train_actions]
        for action in self.train_classes:
            self.action_recog.add_action(os.path.join(path, action),
                                         self.masks_needed,
                                         use_dexter=False,
                                         isstatic=self.isstatic,
                                         max_act_samples=max_act_samples,
                                         feature_params=([self.hog_num],[None]))
            self.sync.append(self.action_recog.actions.actions[-1].sync)

    def process_dictionaries(self, train_act_num, retrain=False,
                             dictionaries_savepath=None):
        '''
        Train dictionaries using <train_act_num>-th action or load them if
            <retrain> is not True and file exists (which one?)
        '''
        if dictionaries_savepath is None:
            dictionaries_savepath = self.use + '_dict_' + self.name + '_train'
            if self.isstatic:
                dictionaries_savepath = dictionaries_savepath + '_static'
            else:
                dictionaries_savepath = dictionaries_savepath + '_active'
            dictionaries_savepath += '.pkl'
        read_from_mem = 0
        self.action_recog.dictionaries.save_path = dictionaries_savepath
        if not retrain:
            try:
                with open(self.action_recog.dictionaries.save_path, 'r') as inp:
                    LOG.info('Loading dictionaries..')
                    self.dicts = pickle.load(inp)
                    self.action_recog.dictionaries.dicts = self.dicts
                read_from_mem = 1
            except (IOError, EOFError):
                read_from_mem = 0
        else:
            LOG.info('retrain switch is True, so the dictionaries ' +
                         'are retrained')
        if not read_from_mem:
            LOG.info('Training dictionaries..')
            self.dicts = self.action_recog.train_sparse_dictionaries(
                act_num=train_act_num, masks_needed=self.masks_needed)
        LOG.info('Making Sparse Features..')
        self.sparse_features_lists = (self.action_recog.
                                      actions.update_sparse_features(self.dicts,
                                                                     ret_sparse=True))

    def process_training(self, num_of_cores=4, retrain=False,
                     savepath=None, save=True,
                     against_training=False, use_dicts=True):
        '''
        Train (or load trained) Classifiers with number of cores num_of_cores, with buffer size (stride
            is 1) <self.buffer_size>. If <retrain> is True, Classifiers are retrained, even if
            <save_path> exists. If against_training, the training data is
            prepared, assuming that it will be used after that for testing.
        '''
        loaded = 0
        if save and savepath is None:
            raise('savepath needed')
        if against_training or retrain or not os.path.exists(savepath):
            if against_training and (not retrain or os.path.exists(savepath)):
                LOG.info('Preparing Classifiers Train Data for testing..')
            else:
                LOG.info('Preparing Classifiers Train Data..')
            if not self.isstatic:
                initial_traindata = []
                if use_dicts:
                    for sparse_features in self.sparse_features_lists:
                        initial_traindata.append(np.concatenate(tuple(sparse_features),
                                                                axis=0))
                else:
                    for action in self.action_recog.actions.actions:
                        initial_traindata.append(np.concatenate(tuple(action.features),
                                                                axis=0))

                traindata_samples_inds = [
                    action.samples_indices
                    for action in self.action_recog.actions.actions]
                traindata_frames_inds = [
                    action.sync for action in self.action_recog.actions.actions]
                acts_buffers = []
                for data, frames_samples_inds, frames_inds in zip(
                        initial_traindata, traindata_samples_inds,
                        traindata_frames_inds):
                    act_buffers = []
                    for count in range(data.shape[1] - self.buffer_size):
                        # Checking if corresponding frames belong to same sample
                        # and are not too timespace distant from each other
                        # If every condition holds then the buffer is generated
                        if (len(np.unique(
                            frames_samples_inds[count:count +
                                                self.buffer_size])) == 1
                            and
                            np.all(np.abs(np.diff(frames_inds[count:count +
                                                              self.buffer_size]))
                                    <=
                                   self.buffer_size / 4)):
                            act_buffers.append(np.atleast_2d(
                                data[:, count:count + self.buffer_size].ravel()))
                    acts_buffers.append(act_buffers)
                LOG.info('Train Data has ' + str(len(acts_buffers)) +
                             ' buffer lists. First buffer list has length ' +
                             str(len(acts_buffers[0])) +
                             ' and last buffer has shape ' +
                             str(acts_buffers[0][-1].shape))
                LOG.info('Joining buffers..')
                multiclass_traindata = []
                for act_buffers in acts_buffers:
                    multiclass_traindata.append(
                        np.concatenate(tuple(act_buffers), axis=0))
            else:
                if use_dicts:
                    multiclass_traindata= [np.concatenate(tuple(feature_list),axis=0).T for
                                     feature_list in
                                     self.sparse_features_lists]
                else:
                    multiclass_traindata= [np.concatenate(tuple(action.features),axis=0).T for
                                        action in
                                     self.action_recog.actions.actions]

            LOG.info('Train Data has ' + str(len(multiclass_traindata)) +
                         ' training datasets for each action. Shape of first dataset is ' +
                         str(multiclass_traindata[0].shape))
            LOG.info(
                'Creating ground truth vector and concatenating remaining data..')
            self.train_ground_truth = []
            self.one_v_all_traindata = np.concatenate(
                tuple(multiclass_traindata), axis=0)
            for count, data in enumerate(multiclass_traindata):
                self.train_ground_truth.append(count *
                                         np.ones(data.shape[0]))
            self.train_ground_truth = np.concatenate(
                tuple(self.train_ground_truth), axis=0)
            LOG.info('Final TrainData to be used as input to Classifier' +
                         ' has shape ' + str(self.one_v_all_traindata.shape))
        if not retrain and os.path.exists(savepath):
            LOG.info('Loading trained Classifiers..')
            with open(savepath, 'r') as inp:
                (self.unified_classifier, self.train_ground_truth,
                 self.train_classes) = pickle.load(inp)
                self.initialize_classifier(self.unified_classifier)
            loaded = 1
        else:
            if retrain and not os.path.exists(savepath):
                LOG.info('retrain switch is True, so the Classifiers ' +
                             'are retrained')
            LOG.info(
                'Training Classifiers using ' +
                str(num_of_cores) +
                ' cores..')
            self.initialize_classifier(self.classifier_type.fit(self.one_v_all_traindata,
                                                                  self.train_ground_truth))
        if save and not loaded:
            LOG.info('Saving trained Classifiers..')
            with open(savepath, 'w') as out:
                pickle.dump((self.unified_classifier, self.train_ground_truth,
                             self.train_classes), out)

    def offline_testdata_processing(self, datapath):
        '''
        Offline testing data processing, using data in <datapath>.
        '''
        LOG.info('Processing test data..')
        LOG.info('Constructing ground truth vector..')
        LOG.info('Extracting features..')
        sparse_features, self.test_sync = self.action_recog.add_action(datapath, masks_needed=False,
                                                   for_testing=True,
                                                   isstatic=self.isstatic,
                                         feature_params=([self.hog_num],[None]))
        sparse_features = np.concatenate(tuple(sparse_features), axis=0)
        if not self.isstatic:
            act_buffers = []
            frames_inds = self.action_recog.actions.testing.sync
            test_buffers_start_inds = []
            test_buffers_end_inds = []
            for count in range(sparse_features.shape[1] - self.buffer_size):
                if np.all(np.abs(np.diff(frames_inds[count:count +
                                                     self.buffer_size])) <=
                          self.buffer_size / 4):

                    act_buffers.append(np.atleast_2d(sparse_features[:, count:count +
                                                                     self.buffer_size].ravel()))
                    test_buffers_start_inds.append(frames_inds[count])
                    test_buffers_end_inds.append(frames_inds[count +
                                                             self.buffer_size])
            testdata = np.concatenate(tuple(act_buffers), axis=0)
            return testdata, test_buffers_start_inds, test_buffers_end_inds
        else:
            testdata = sparse_features.T
            return testdata

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
                ground_truth = np.zeros(max([int(filter(str.isdigit,filename)) for
                                             filename in files]) + 1)
            ground_truth[:] = np.NaN
            all_bounds = [map(list, zip(*ground_truth_init[key])) for key in
                          ground_truth_init.keys()]
            if isinstance(data, basestring):
                iterat = [int(filter(str.isdigit, filename))
                          for filename in files]
            else:
                iterat = np.where(data)[0]
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
                LOG.error('Invalid format')
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
        return np.apply_along_axis(self.masked_mean,
                                   0, data, win_size)[:-win_size+1]

    def run_testing(self, data=None, online=True, against_training=False,
                    scores_filter_shape=5,
                    std_small_filter_shape=co.CONST['STD_small_filt_window'],
                    std_big_filter_shape=co.CONST['STD_big_filt_window'],
                    ground_truth_type= co.CONST['test_ground_truth'],
                    img_count=None, save=True, scores_savepath=None,
                    load=False):
        '''
        Test ClassifierS using data (.png files) located in <data>. If <online>, the
        testing is online, with <data> being a numpy array, which has been
        firstly processed by <hand_segmentation_alg>. If <against_training>,
        the testing happens using the concatenated training data, so <data> is
        not used. The scores retrieved from testing are filtered using a
        box filter of shape <box_filter_shape>. The running mean along a buffer
        of the data is computed with a running window of length
        <mean_filter_shape>. The ground truth for the testing data is given by
        <ground_truth_type> (for further info about the variable refer to
        <self.construct_ground_truth>). If the training is online, the count of
        the frame is passed by <img_count>. If <save> is True,
        testing results are saved to <scores_savepath>.
        If <load> is True and <scores_save_path> exists, testing is bypassed and all the
        necessary results are loaded from memory.
        '''
        if save or load:
            if scores_savepath is None:
                scores_savepath = self.use + '_' + self.name
                if self.isstatic:
                    scores_savepath = scores_savepath + '_static'
                else:
                    scores_savepath = scores_savepath + '_active'
                scores_savepath += '_testscores.pkl'
        if not online:
            if load and os.path.exists(scores_savepath):
                with open(scores_savepath, 'r') as inp:
                    self.scores,self.test_sync = pickle.load(inp)
                # DEBUGGING
            else:
                if self.use == 'svms':
                    LOG.info('Classifier contains ' +
                                 str(len(self.unified_classifier.estimators_)) + ' estimators')
                if against_training:
                    LOG.info('Testing Classifiers against training data..')
                    self.scores = self.decide(
                        self.one_v_all_traindata)
                else:
                    if not self.isstatic:
                        (testdata,
                         test_buffers_start_inds,
                         test_buffers_end_inds) = self.offline_testdata_processing(
                             data)
                    else:
                        testdata = self.offline_testdata_processing(
                            data)
                    self.testdata = testdata
                    LOG.info(self.name+':')
                    LOG.info('Testing Classifiers..')
                    self.scores = self.decide(
                        testdata)
                    if not self.isstatic:
                        expanded_scores = np.zeros((self.test_sync[-1] + 1,
                                                    self.scores.shape[1]))
                        expanded_scores[:] = np.NaN
                        for score, start, end in zip(self.scores,
                                                     test_buffers_start_inds,
                                                     test_buffers_end_inds):
                            expanded_scores[start:end + 1, :] = score[None, :]
                        self.scores = expanded_scores
                    if save:
                        with open(scores_savepath, 'w') as out:
                            pickle.dump((self.scores,self.test_sync), out)
            self.test_ground_truth = self.construct_ground_truth(
                os.path.join(data, '0'), ground_truth_type)
            if not self.isstatic:
                self.filtered_scores = self.upgr_filter(self.scores,
                                                        scores_filter_shape)
                self.filtered_scores = self.scores
                self.process_offline_scores_original(scores_filter_shape,
                                                     std_small_filter_shape,
                                                     std_big_filter_shape)
            else:
                #self.filtered_scores = self.upgr_filter(self.scores,
                #                                        3)
                self.filtered_scores = self.scores
                self.recognized_classes = []
                for score in self.filtered_scores:
                    if np.max(score) >= 0.7 or len(self.recognized_classes)==0:
                        self.recognized_classes.append(score.argmax())
                    else:
                        self.recognized_classes.append(self.recognized_classes[-1])
                self.recognized_classes = np.array(self.recognized_classes)
                #self.recognized_classes = self.scores.argmax(axis=1)
                plt.figure()
                axes = plt.subplot(111)
                for count, score in enumerate(np.transpose(np.array(
                        self.filtered_scores))):
                    axes.plot(score, label='%s' % self.train_classes[count])
                self.put_legend_outside_plot(axes)
                plt.title(self.use+' '+self.name+' Filtered Scores')
                plt.xlabel('Frames')
            if (self.action_recog.actions.
                features_extract is not None):
                LOG.info('Mean feature extraction time ' +
                         str(np.mean(self.action_recog.actions.
                                 features_extract.extract_time)))
                LOG.info('Max feature extraction time ' +
                         str(np.max(self.action_recog.actions.
                                 features_extract.extract_time)))
                LOG.info('Min feature extraction time ' +
                         str(np.min(self.action_recog.actions.
                                 features_extract.extract_time)))
                LOG.info('Median feature extraction time ' +
                         str(np.median(self.action_recog.actions.
                                 features_extract.extract_time)))
        else:
            '''
            input is processed from hand_segmentation_alg (any data
            processed in such way, that the result is the same with my processing,
            is acceptable, eg. Dexter)
            There must be a continuous data streaming (method called in every
            loop), even if the result of the previous algorithm is None
            '''
            self.process_online_scores(data, img_count,
                                       scores_filter_shape,
                                       std_small_filter_shape,
                                       std_big_filter_shape)


    def process_online_scores(self, data, img_count=None,
                              scores_filter_shape=5,
                              std_small_filter_shape=co.CONST[
                                  'STD_small_filt_window'],
                              std_big_filter_shape=co.CONST[
                                  'STD_big_filt_window']):
        '''
        <data> is the frame with frame number <img_count> or increasing by one
        relatively to the previous frame. Scores are filtered with a filter of
        length <scores_filter_shape>. <std_small_filter_shape> is the shape
        of the filter used to remove the temporal noise from the scores std.
        <std_big_filter_shape> is the shape of the filter to compute the mean
        of the scores std.
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
            self.small_std_running_mean_vec = []
            self.big_std_running_mean_vec = []
            self.scores_running_mean_vec = []
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
        '''
        existence = self.buffer_exists[-self.buffer_size:]
        if sum(existence) < 3 * self.buffer_size / 4:
            return
        elif sum(existence) == 3 * self.buffer_size / 4:
            self.mean_from = 0
        '''
        inp = np.array(self._buffer).T.reshape(1, -1)
        # inp = np.atleast_2d(np.concatenate(tuple(self._buffer),
        #                                   axis=0))
        score = (self.unified_classifier.
                 decision_function(inp))
        self.scores.append(score)
        if len(self.scores_running_mean_vec) < scores_filter_shape:
            self.scores_running_mean_vec.append(score.ravel())
        else:
            self.scores_running_mean_vec = (self.scores_running_mean_vec[1:]
                                           + [score.ravel()])
        start_from = np.sum(self.buffer_exists[-std_small_filter_shape:])
        self.filtered_scores.append(
            np.mean(np.array(self.scores_running_mean_vec[
                -start_from:]
            ), axis=0))
        score_std = np.std(self.filtered_scores[-1])
        if len(self.small_std_running_mean_vec) < std_small_filter_shape:
            self.small_std_running_mean_vec.append(score_std)
        else:
            self.small_std_running_mean_vec = (
                self.small_std_running_mean_vec[1:] +
                [score_std])
        filtered_score_std = np.mean(self.small_std_running_mean_vec)
        self.filtered_scores_std.append(filtered_score_std)
        if len(self.big_std_running_mean_vec) < std_big_filter_shape:
            self.big_std_running_mean_vec.append(filtered_score_std)
        else:
            self.big_std_running_mean_vec = (self.big_std_running_mean_vec[1:]
                                         + [filtered_score_std])
        start_from = np.sum(self.buffer_exists[
            -min(std_big_filter_shape, self.mean_from + 1):])
        self.filtered_scores_std_mean.append(
            np.mean(self.big_std_running_mean_vec[-start_from:]))
        mean_diff = self.filtered_scores_std_mean[
            -1] - self.filtered_scores_std[-1]
        if (np.min(mean_diff) > co.CONST['action_separation_thres'] and not
           self.on_action):
            self.crossings.append(self.img_count)
            self.on_action = True
            #self.mean_from = self.img_count
            if self.recognized_classes:
                self.recognized_classes[-1].add(length=self.img_count -
                                            self.new_action_starts_count +
                                               1)
                LOG.info('Frame ' + str(self.img_count) + ': ' +
                             self.recognized_classes[-1].name +
                             ', starting from frame ' +
                             str(self.recognized_classes[-1].start) +
                            ' with length ' +
                             str(self.recognized_classes[-1].length))
            self.recognized_classes.append(ClassObject(self.train_classes))
            index = np.argmax(self.filtered_scores[-1])
            self._max = self.filtered_scores[-1][index]
            self.act_inds = [index]
            self.new_action_starts_count = self.img_count
            self.recognized_classes[-1].add(
                                            index=index,
                                            start=self.new_action_starts_count)
            self.saved_buffers_scores = []
        else:
            if len(self.recognized_classes) > 0:
                _arg = np.argmax(self.filtered_scores[-1])
                if self._max < self.filtered_scores[-1][_arg]:
                    self._max = self.filtered_scores[-1][_arg]
                    self.recognized_classes[-1].add(index=_arg)
                if mean_diff < co.CONST['action_separation_thres']:
                    self.on_action = False
                self.saved_buffers_scores.append(score)
                LOG.info('Frame ' + str(self.img_count) +': ' +
                             self.recognized_classes[-1].name)
                return self.recognized_classes[-1].name
            else:
                return None

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
        positive = np.zeros(self.scores.shape[0])
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
        finite_scores = self.filtered_scores[
                    np.isfinite(self.filtered_scores)].reshape(
                        -1, self.scores.shape[1])
        time_norm = np.max(finite_scores, axis=0) - np.min(finite_scores, axis
                                                          =0)
        self.time_normalized_filtered_scores = ((self.filtered_scores -
                                                 np.min(finite_scores, axis=0))/
                                                time_norm)
        self.time_normalized_filtered_scores[
            self.time_normalized_filtered_scores<0]=0
        #self.filtered_scores = self.time_normalized_filtered_scores
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
        self.already_expanded = True
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
            axes.plot((mean_diff) / float(np.max(np.abs(mean_diff[
                np.isfinite(mean_diff)]))),
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


    def put_legend_outside_plot(self, axes):
        '''
        Remove legend from the insides of the plots
        '''
        # Shrink current axis by 20%
        box = axes.get_position()
        axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def compute_performance_measures(self, fmask):
        from sklearn import metrics
        y_true = self.test_ground_truth[fmask]
        y_pred = self.recognized_classes[fmask]
        print y_true, y_pred
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
        fmask = None
        if self.buffer_exists is not None:
            self.buffer_exists = np.array(self.buffer_exists)
            expanded_recognized_classes = np.zeros(self.buffer_exists.size)
            expanded_recognized_classes[:] = None
            for clas in self.recognized_classes:
                expanded_recognized_classes[clas.start:clas.start + clas.length + 1][
                    self.buffer_exists[clas.start:clas.start + clas.length + 1]] = clas.index
            self.recognized_classes = expanded_recognized_classes
            self.crossings = np.array(self.crossings)
            if self.test_ground_truth is not None:
                fmask = self.buffer_exists
        elif self.test_ground_truth is not None:
            if self.isstatic:
                recognized_classes_expanded = np.zeros_like(self.test_ground_truth)
                recognized_classes_expanded[self.test_sync
                                   ] = self.recognized_classes[np.isfinite(
                                       self.recognized_classes)]
                fmask = \
                    np.isfinite(self.test_ground_truth)*np.isfinite(
                        recognized_classes_expanded)
                self.recognized_classes = recognized_classes_expanded
            else:
                fmask = (np.isfinite(self.recognized_classes)*np.isfinite(
                    self.test_ground_truth)).astype(bool)


        if fmask is not None:
            self.compute_performance_measures(fmask)
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


def fake_online_testing(classifier, data='train', path=None):
    '''
    Immitate online testing for performance testing reasons
    '''
    if path is None:
        path = os.path.join(co.CONST[data + '_path'],'0')
    filenames = glob.glob(os.path.join(path, '*.png'))
    sync = [int(filter(str.isdigit,
                                        os.path.basename(filename)))
                             for filename in filenames]
    [sync,filenames] = map(list,zip(*[[y,x] for (y,x) in
                                 sorted(zip(sync,filenames),
                                        key=lambda pair: pair[0])]))
    txts = glob.glob(os.path.join(path, '*.txt'))
    for count, filename in enumerate(filenames):
        img = cv2.imread(filename, -1)
        cv2.imshow('test', (img%255).astype(np.uint8))
        cv2.waitKey(10)
        img_count = int(filter(str.isdigit, filename))
        classifier.run_testing(img, img_count=img_count, online=True, load=False)
    classifier.recognized_classes[-1].add(length=img_count)
    classifier.test_ground_truth = classifier.construct_ground_truth(
        classifier.buffer_exists, ground_truth_type=co.CONST[data + '_ground_truth'])
    plt.figure()
    axes = plt.subplot(111)
    classifier.filtered_scores = np.array(classifier.filtered_scores).squeeze()
    classifier.buffer_exists = np.array(classifier.buffer_exists)
    expanded_scores = np.zeros((len(classifier.buffer_exists),classifier.filtered_scores.shape[1]))
    expanded_scores[:] = np.NaN
    expanded_scores[classifier.buffer_exists.astype(bool), :] = classifier.filtered_scores
    for count, scores in enumerate(np.transpose(np.array(
            expanded_scores))):
        axes.plot(scores.ravel(), label='%s' % classifier.train_classes[count])
    classifier.put_legend_outside_plot(axes)
    plt.title('Filtered Scores')
    plt.xlabel('Frames')
    plt.figure()
    axes = plt.subplot(111)
    axes.plot(
        np.array(
            classifier.filtered_scores_std),
        color='r',
        label=r'STD')
    axes.plot(np.array(classifier.filtered_scores_std_mean),
              color='g', label='STD Mean')
    classifier.put_legend_outside_plot(axes)
    axes.set_title('Filtered Scores Statistics')
    axes.set_xlabel('Frames')


def main():
    '''
    Example Usage
    '''
    actions_svm = Classifier('INFO', isstatic=False,
                             train_max_im_num=300, train_max_iter=3,
                            name='actions', des_dim=300)

    static_forest = Classifier('INFO', isstatic=True, train_max_im_num=100,
                               name='poses', use='forest',
                        hog_num=32,
                        des_dim=300)
    actions_svm.run_training(classifiers_retrain=False, use_dicts=False,
                             buffer_size=25, max_act_samples=300)

    static_forest.run_training(classifiers_retrain=False,
                            max_act_samples=2000, use_dicts=False)
    data = 'train'

    static_forest.run_testing(co.CONST[data + '_path'],
                    ground_truth_type=co.CONST[data + '_ground_truth'],
                    online=False, load=False)
    actions_svm.run_testing(co.CONST[data + '_path'],
                    ground_truth_type=co.CONST[data + '_ground_truth'],
                    online=False, load=False)



    actions_svm.visualize_scores('Actions SVM testing')
    static_forest.visualize_scores('Poses Forest testing')
    '''
    fake_online_testing(svm,data)
    svm.visualize_scores('Fake Online Testing')
    '''
    plt.show()



LOG = logging.getLogger('__name__')
CH = logging.StreamHandler(sys.stderr)
CH.setFormatter(logging.Formatter(
    '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel(logging.INFO)
if __name__ == '__main__':
    main()
